"""Spectral comparison of synthesized audio vs the real reference.

The eval samples are LibriSpeech utterances whose *own* wav is the reference:
same text, same speaker. This script compares each condition's synthesized
waveform with that reference in the spectral domain:

* log-mel spectrograms (CosyVoice2 mel config: 24 kHz, n_fft=1920, hop=480,
  80 mels, fmin=0, fmax=8000);
* DTW alignment over mel frames (removes prosody/timing differences);
* per-frame mel L2 distance (lower = closer to reference) and spectral
  convergence;
* side-by-side spectrogram images per utterance and condition.

This complements WER/ASR-confidence: it measures spectral fidelity, which is
what "audio quality" degradation shows up as.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from spade.config_utils import dataclass_from_dict, load_yaml
from spade_cosyvoice2.wli import load_eval_samples


MEL_KW = dict(sr=24000, n_fft=1920, hop_length=480, n_mels=80, fmin=0, fmax=8000, center=False)


def log_mel(path: str) -> np.ndarray:
    """Log-mel spectrogram (dB), shape (n_mels, T)."""
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != MEL_KW["sr"]:
        y = librosa.resample(y, orig_sr=sr, target_sr=MEL_KW["sr"])
    mel = librosa.feature.melspectrogram(y=y, **MEL_KW)
    return librosa.power_to_db(mel, ref=np.max, top_db=80.0)


def dtw_align(a: np.ndarray, b: np.ndarray, radius: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """DTW alignment of mel frame sequences (band-limited for speed).

    Returns aligned index arrays into ``a`` and ``b`` (same length).
    """
    n, m = a.shape[1], b.shape[1]
    cost = np.zeros((n + 1, m + 1))
    cost[1:, 0] = np.inf
    cost[0, 1:] = np.inf
    # squared L2 between frames
    d = ((a.T[:, None, :] - b.T[None, :, :]) ** 2).sum(-1)
    for i in range(1, n + 1):
        lo = max(1, i - radius)
        hi = min(m + 1, i + radius + 1)
        cost[i, lo:hi] = d[i - 1, lo - 1 : hi - 1] + np.minimum.reduce(
            [cost[i - 1, lo - 1 : hi - 1], cost[i - 1, lo:hi], cost[i, lo - 1 : hi - 1]]
        )
    # backtrack
    i, j = n, m
    ia, ib = [], []
    while i > 0 and j > 0:
        ia.append(i - 1)
        ib.append(j - 1)
        cand = [
            (cost[i - 1, j - 1], i - 1, j - 1),
            (cost[i - 1, j], i - 1, j),
            (cost[i, j - 1], i, j - 1),
        ]
        _, i, j = min(cand, key=lambda x: x[0])
    ia.reverse()
    ib.reverse()
    return np.array(ia), np.array(ib)


@dataclass
class SpectralCompareConfig:
    eval_list: str = ""
    out_dir: str = "outputs/cosyvoice2/ablation_torch"
    conditions: list[str] = field(default_factory=list)  # subdirs under out_dir
    subset_size: int = 8


def run(cfg: SpectralCompareConfig) -> dict:
    out = Path(cfg.out_dir)
    samples = load_eval_samples(cfg.eval_list, cfg.subset_size)
    conditions = cfg.conditions or [
        d.name for d in out.iterdir() if d.is_dir() and (d / "ablation_report.json").exists()
    ]
    if not conditions:
        # fall back to any condition subdir with wavs
        conditions = [d.name for d in out.iterdir() if d.is_dir() and list(d.glob("*.wav"))]
    if not conditions:
        raise ValueError(f"no condition dirs with wavs found under {out}")

    fig_dir = out / "spectra"
    fig_dir.mkdir(parents=True, exist_ok=True)
    per_cond: dict[str, dict[str, float]] = {
        c: {"mel_l2": [], "spec_conv": []} for c in conditions
    }
    summary: dict[str, Any] = {"reference": "eval utterance's own wav", "conditions": {}}

    for sample in samples[: cfg.subset_size]:
        ref_mel = log_mel(sample["wav"])
        for cond in conditions:
            wav_path = out / cond / f"{sample['utt']}.wav"
            if not wav_path.exists():
                continue
            syn_mel = log_mel(str(wav_path))
            ia, ib = dtw_align(ref_mel, syn_mel)
            diff = ref_mel[:, ia] - syn_mel[:, ib]
            mel_l2 = float(np.sqrt((diff**2).mean()))
            spec_conv = float(np.linalg.norm(diff) / np.linalg.norm(ref_mel[:, ia]))
            per_cond[cond]["mel_l2"].append(mel_l2)
            per_cond[cond]["spec_conv"].append(spec_conv)

            # side-by-side spectrogram
            fig, axes = plt.subplots(3, 1, figsize=(14, 8))
            for ax, mel, title in zip(
                axes,
                (ref_mel, syn_mel, syn_mel[:, ib]),
                (f"reference (real)", f"{cond}", f"{cond} DTW-aligned"),
            ):
                img = ax.imshow(mel, aspect="auto", origin="lower", cmap="magma")
                ax.set_title(title)
                fig.colorbar(img, ax=ax, fraction=0.02)
            fig.suptitle(f"{sample['utt']} | mel L2={mel_l2:.2f} | conv={spec_conv:.3f}")
            fig.tight_layout()
            fig.savefig(fig_dir / f"{sample['utt']}_{cond}.png", dpi=110)
            plt.close(fig)

    for cond in conditions:
        if per_cond[cond]["mel_l2"]:
            summary["conditions"][cond] = {
                "mel_l2_mean": round(float(np.mean(per_cond[cond]["mel_l2"])), 3),
                "mel_l2_per_utt": [round(v, 3) for v in per_cond[cond]["mel_l2"]],
                "spec_conv_mean": round(float(np.mean(per_cond[cond]["spec_conv"])), 4),
                "spec_conv_per_utt": [round(v, 4) for v in per_cond[cond]["spec_conv"]],
            }
            print(
                f"[spectra] {cond:<16} mel_l2={summary['conditions'][cond]['mel_l2_mean']:.3f} "
                f"spec_conv={summary['conditions'][cond]['spec_conv_mean']:.4f}"
            )

    report_path = out / "spectra_report.json"
    report_path.write_text(json.dumps(summary, indent=2))
    print(f"[spectra] images -> {fig_dir}")
    print(f"[spectra] report  -> {report_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Spectral comparison vs real reference")
    parser.add_argument("--config", default="")
    parser.add_argument("--eval-list", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--conditions", nargs="+", default=[])
    args = parser.parse_args()
    cfg = SpectralCompareConfig()
    if args.config:
        cfg = dataclass_from_dict(SpectralCompareConfig, load_yaml(args.config), "SpectralCompareConfig")
    if args.eval_list:
        cfg.eval_list = args.eval_list
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.conditions:
        cfg.conditions = args.conditions
    if not cfg.eval_list:
        raise SystemExit("eval_list is required")
    run(cfg)


if __name__ == "__main__":
    main()
