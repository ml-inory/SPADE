"""WER-based layer importance (WLI) for the CosyVoice2 LLM."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torchaudio

from spade.adapters.hf import bypass_block
from spade.config_utils import dataclass_from_dict, load_yaml, resolve_device
from spade.metrics import word_error_rate
from spade.pruning import compute_wli
from spade_cosyvoice2.model_utils import load_cosyvoice2_with_llm, synthesize


@dataclass
class WLIConfig:
    eval_list: str
    subset_size: int = 8
    whisper_model: str = "base"
    model_dir: str = ""
    device: str = "auto"

    def __post_init__(self) -> None:
        if not self.model_dir:
            from spade_cosyvoice2.paths import model_dir

            self.model_dir = str(model_dir())


def load_eval_samples(eval_list: str, subset_size: int) -> list[dict]:
    """Read utterances + same-speaker prompts from the eval parquet."""
    with open(eval_list) as fh:
        paths = [line.strip() for line in fh if line.strip()]
    rows = [dict(r) for p in paths for r in pq.read_table(p).to_pylist()]
    by_spk: dict[str, list[dict]] = {}
    for r in rows:
        by_spk.setdefault(r["spk"], []).append(r)
    samples = []
    for r in rows[:subset_size]:
        prompt = next((p for p in by_spk.get(r["spk"], []) if p["utt"] != r["utt"]), r)
        samples.append(
            {
                "utt": r["utt"],
                "text": r["text"],
                "wav": r["wav"],
                "spk": r["spk"],
                "prompt_utt": prompt["utt"],
                "prompt_text": prompt["text"],
                "prompt_wav": prompt["wav"],
            }
        )
    return samples


class CosyVoice2WERScorer:
    """WER scorer: synthesize zero-shot with the (bypassed) LLM, transcribe."""

    def __init__(self, cosyvoice, whisper_model: str, device: str) -> None:
        import whisper

        self.cv = cosyvoice
        self.whisper = whisper.load_model(whisper_model, device=device)

    def score(self, qwen_model, sample: dict) -> float:
        speech = synthesize(
            self.cv, sample["text"], sample["prompt_text"], sample["prompt_wav"]
        )
        return self.score_speech(speech, sample["text"])

    def score_speech(self, speech: torch.Tensor, reference: str) -> float:
        speech = speech.to("cpu")
        if self.cv.sample_rate != 16000:
            speech = torchaudio.transforms.Resample(
                self.cv.sample_rate, 16000
            )(speech)
        text = self.whisper.transcribe(
            speech[0].numpy().astype(np.float32), fp16=False
        )["text"].strip()
        return word_error_rate(reference, text)


def run_wli(config: WLIConfig) -> dict:
    device = resolve_device(config.device)
    cosyvoice = load_cosyvoice2_with_llm(config.model_dir)
    qwen = cosyvoice.model.llm.llm.model
    samples = load_eval_samples(config.eval_list, config.subset_size)
    print(f"[wli] {len(samples)} eval samples, {qwen.config.num_hidden_layers} layers")
    scorer = CosyVoice2WERScorer(cosyvoice, config.whisper_model, device)
    wli = compute_wli(
        qwen,
        scorer,
        samples,
        bypass_fn=bypass_block,
        n_layers=qwen.config.num_hidden_layers,
        show_progress=True,
    )
    report = {
        "wli": wli.tolist(),
        "eval_subset_size": len(samples),
        "eval_list": config.eval_list,
        "whisper_model": config.whisper_model,
    }
    out = Path(config.eval_list).parent / "wli_report.json"
    out.write_text(json.dumps(report, indent=2))
    print("[wli] per-layer WLI: " + ", ".join(f"{v:.3f}" for v in wli))
    print(f"[wli] report -> {out}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="WLI for CosyVoice2 LLM")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = dataclass_from_dict(WLIConfig, load_yaml(args.config), "WLIConfig")
    run_wli(cfg)


if __name__ == "__main__":
    main()
