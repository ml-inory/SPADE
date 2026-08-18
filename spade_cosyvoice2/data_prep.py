"""Prepare a small LibriTTS subset in CosyVoice2's parquet format.

The CosyVoice2 training pipeline consumes parquet shards with columns
``utt / audio_data / wav / text / spk / utt_embedding / spk_embedding /
speech_token``. This module downloads a small LibriTTS part (e.g. dev-clean,
~54 MB), selects a deterministic subset, extracts speaker embeddings
(campplus.onnx) and 25 Hz speech tokens (speech_tokenizer_v2.onnx) with CPU
onnxruntime, and writes the parquet files plus ``data.list`` train/eval
splits -- enough to run SPADE distillation and WLI evaluation on real speech.
"""

from __future__ import annotations

import argparse
import json
import random
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import whisper
from tqdm import tqdm

from spade.config_utils import dataclass_from_dict, load_yaml, resolve_device
from spade_cosyvoice2.paths import cosyvoice_root


OPENS_LR_BASE = "https://www.openslr.org/resources/60"


@dataclass
class DataPrepConfig:
    part: str = "dev-clean"
    hf_parquet: str = ""       # optional: openslr/librispeech_asr parquet path/URL
    num_utts: int = 200
    train_utts: int = 150
    num_threads: int = 8
    seed: int = 0
    data_root: str = ""  # default: <cosyvoice_root>/data/libritts
    out_root: str = ""   # default: <cosyvoice_root>/data/spade

    def __post_init__(self) -> None:
        if not self.data_root:
            self.data_root = str(cosyvoice_root() / "data" / "libritts")
        if not self.out_root:
            self.out_root = str(cosyvoice_root() / "data" / "spade")


def download_part(part: str, root: Path) -> Path:
    """Download and extract a LibriTTS part (e.g. dev-clean)."""
    url = f"{OPENS_LR_BASE}/{part}.tar.gz"
    tarball = root / f"{part}.tar.gz"
    root.mkdir(parents=True, exist_ok=True)
    if not tarball.exists():
        print(f"[data] downloading {url}")
        urllib.request.urlretrieve(url, tarball)
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(root)
    return root / "LibriTTS" / part


def collect_utterances(src_dir: Path) -> list[dict[str, str]]:
    """Pair each wav with its normalized transcript (like prepare_data.py)."""
    utts: list[dict[str, str]] = []
    for wav in sorted(src_dir.rglob("*.wav")):
        txt = wav.with_suffix(".normalized.txt")
        if not txt.exists():
            continue
        utt = wav.stem
        text = txt.read_text().strip()
        if not text:
            continue
        utts.append({"utt": utt, "wav": str(wav), "text": text, "spk": utt.split("_")[0]})
    return utts


def collect_from_hf_parquet(
    parquet_path: str | Path,
    out_root: Path,
    num_utts: int,
    seed: int,
) -> list[dict[str, str]]:
    """Build the utterance table from a LibriSpeech HF parquet (audio bytes).

    Audio bytes are written as flac files so the rest of the pipeline
    (embedding / speech-token extraction) is identical to the openslr path.
    """
    import random

    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path).to_pylist()
    rng = random.Random(seed)
    rng.shuffle(table)
    rows = table[:num_utts]
    audio_dir = out_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    utts: list[dict[str, str]] = []
    for r in rows:
        audio = r.get("audio")
        if isinstance(audio, dict):
            audio_bytes = audio.get("bytes")
        else:
            audio_bytes = audio
        if audio_bytes is None:
            continue
        utt = str(r.get("id", r.get("file", "")).split("/")[-1]).replace(".flac", "").replace(".wav", "")
        if not utt:
            continue
        path = audio_dir / f"{utt}.flac"
        if not path.exists():
            path.write_bytes(audio_bytes)
        spk = str(r.get("speaker_id", utt.split("_")[0]))
        utts.append({"utt": utt, "wav": str(path), "text": r["text"].strip(), "spk": spk})
    return utts


def _embedding_utt(wav: str, sess: onnxruntime.InferenceSession) -> list[float]:
    audio, sr = read_audio(wav)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)
    feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    out = sess.run(None, {sess.get_inputs()[0].name: feat.unsqueeze(0).numpy()})[0]
    return out.flatten().tolist()


def _speech_token_utt(wav: str, sess: onnxruntime.InferenceSession) -> list[int]:
    audio, sr = read_audio(wav)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    feat = whisper.log_mel_spectrogram(audio, n_mels=128)
    names = [i.name for i in sess.get_inputs()]
    feeds = {
        names[0]: feat.detach().cpu().numpy(),
        names[1]: np.array([feat.shape[2]], dtype=np.int32),
    }
    return sess.run(None, feeds)[0].flatten().tolist()


def read_audio(wav: str, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    """Load audio with soundfile (torchaudio.load is broken in this env)."""
    import soundfile as sf

    data, sr = sf.read(wav, dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.T)  # (channels, T)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != target_sr:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(audio)
    return audio, target_sr


def _make_onnx(path: Path, threads: int) -> onnxruntime.InferenceSession:
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1
    return onnxruntime.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def prepare(cfg: DataPrepConfig) -> dict[str, Any]:
    root = Path(cfg.data_root)
    out = Path(cfg.out_root)
    model_dir = cosyvoice_root() / "pretrained_models" / "CosyVoice2-0.5B"
    if cfg.hf_parquet:
        utts = collect_from_hf_parquet(cfg.hf_parquet, out, cfg.num_utts, cfg.seed)
        print(f"[data] loaded {len(utts)} utterances from HF parquet {cfg.hf_parquet}")
    else:
        src_dir = download_part(cfg.part, root)
        utts = collect_utterances(src_dir)
        rng = random.Random(cfg.seed)
        rng.shuffle(utts)
        utts = utts[: cfg.num_utts]
        print(f"[data] selected {len(utts)} utterances from {cfg.part}")

    campplus = _make_onnx(model_dir / "campplus.onnx", cfg.num_threads)
    tokenizer = _make_onnx(model_dir / "speech_tokenizer_v2.onnx", cfg.num_threads)

    utt2embedding: dict[str, list[float]] = {}
    with ThreadPoolExecutor(max_workers=cfg.num_threads) as pool:
        futs = {pool.submit(_embedding_utt, u["wav"], campplus): u for u in utts}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="embeddings"):
            utt2embedding[futs[fut]["utt"]] = fut.result()

    utt2token: dict[str, list[int]] = {}
    with ThreadPoolExecutor(max_workers=cfg.num_threads) as pool:
        futs = {pool.submit(_speech_token_utt, u["wav"], tokenizer): u for u in utts}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="speech tokens"):
            utt2token[futs[fut]["utt"]] = fut.result()

    spk2emb: dict[str, list[float]] = {}
    for u in utts:
        emb = utt2embedding[u["utt"]]
        spk2emb.setdefault(u["spk"], []).append(emb)
    spk2embedding = {k: np.mean(v, axis=0).tolist() for k, v in spk2emb.items()}

    rows = [
        {
            "utt": u["utt"],
            "audio_data": Path(u["wav"]).read_bytes(),
            "wav": u["wav"],
            "text": u["text"],
            "spk": u["spk"],
            "utt_embedding": utt2embedding[u["utt"]],
            "spk_embedding": spk2embedding[u["spk"]],
            "speech_token": utt2token[u["utt"]],
        }
        for u in utts
    ]
    parquet_dir = out / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    train_rows = rows[: cfg.train_utts]
    eval_rows = rows[cfg.train_utts :]
    for name, part_rows in (("train", train_rows), ("eval", eval_rows)):
        path = parquet_dir / f"{name}.parquet"
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(part_rows)), path)
        (parquet_dir / f"{name}.data.list").write_text(f"{path}\n")
        print(f"[data] wrote {path} ({len(part_rows)} utts)")
    meta = {
        "part": cfg.part,
        "num_utts": len(utts),
        "train_utts": len(train_rows),
        "eval_utts": len(eval_rows),
        "train_list": str(parquet_dir / "train.data.list"),
        "eval_list": str(parquet_dir / "eval.data.list"),
        "parquet_dir": str(parquet_dir),
    }
    (out / "data_prep.json").write_text(json.dumps(meta, indent=2))
    print(f"[data] summary -> {out / 'data_prep.json'}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="CosyVoice2 data preparation")
    parser.add_argument("--config", default="")
    parser.add_argument("--part", default="")
    parser.add_argument("--num-utts", type=int, default=0)
    parser.add_argument("--train-utts", type=int, default=0)
    args = parser.parse_args()
    if args.config:
        cfg = dataclass_from_dict(DataPrepConfig, load_yaml(args.config), "DataPrepConfig")
    else:
        cfg = DataPrepConfig()
    if args.part:
        cfg.part = args.part
    if args.num_utts:
        cfg.num_utts = args.num_utts
    if args.train_utts:
        cfg.train_utts = args.train_utts
    prepare(cfg)


if __name__ == "__main__":
    main()
