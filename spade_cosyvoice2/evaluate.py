"""Evaluate a (pruned/distilled) CosyVoice2 LLM: WER, RTF, params."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from spade.config_utils import dataclass_from_dict, load_yaml, resolve_device
from spade_cosyvoice2.model_utils import load_cosyvoice2_with_llm, synthesize
from spade_cosyvoice2.wli import CosyVoice2WERScorer, load_eval_samples


@dataclass
class EvalConfig:
    eval_list: str
    llm_pt: str
    retained: list[int] = field(default_factory=list)
    subset_size: int = 8
    whisper_model: str = "base"
    model_dir: str = ""
    device: str = "auto"

    def __post_init__(self) -> None:
        if not self.model_dir:
            from spade_cosyvoice2.paths import model_dir

            self.model_dir = str(model_dir())


def evaluate(config: EvalConfig) -> dict:
    device = resolve_device(config.device)
    cosyvoice = load_cosyvoice2_with_llm(
        config.model_dir,
        llm_pt=config.llm_pt,
        retained=config.retained or None,
    )
    qwen = cosyvoice.model.llm.llm.model
    samples = load_eval_samples(config.eval_list, config.subset_size)
    scorer = CosyVoice2WERScorer(cosyvoice, config.whisper_model, device)

    wers, rtfs = [], []
    for sample in samples:
        try:
            start = time.perf_counter()
            speech = synthesize(
                cosyvoice, sample["text"], sample["prompt_text"], sample["prompt_wav"]
            )
            elapsed = time.perf_counter() - start
            wers.append(scorer.score_speech(speech, sample["text"]))
            audio_seconds = speech.shape[1] / cosyvoice.sample_rate
            rtfs.append(elapsed / audio_seconds)
        except Exception as exc:  # degenerate output: count as fully wrong
            print(
                f"[eval] sample {sample['utt']} failed ({type(exc).__name__}); WER=1.0",
                file=sys.stderr,
            )
            wers.append(1.0)
            rtfs.append(0.0)

    metrics = {
        "wer": round(float(np.mean(wers)), 4),
        "wer_per_sample": [round(w, 4) for w in wers],
        "rtf": round(float(np.mean(rtfs)), 4),
        "rtf_per_sample": [round(r, 4) for r in rtfs],
        "params": sum(p.numel() for p in qwen.parameters()),
        "depth": qwen.config.num_hidden_layers,
        "retained_indices": config.retained,
    }
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CosyVoice2 LLM variant")
    parser.add_argument("--config", required=True)
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    cfg = dataclass_from_dict(EvalConfig, load_yaml(args.config), "EvalConfig")
    metrics = evaluate(cfg)
    if args.json:
        Path(args.json).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
