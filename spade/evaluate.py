"""Evaluation stage: WER + efficiency metrics for any checkpoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from spade.checkpoints import load_checkpoint
from spade.config_utils import dataclass_from_dict, load_yaml, resolve_device
from spade.data import CharTokenizer, ToySpeechCodec, make_synthetic_texts
from spade.data.dataset import SyntheticTTSDataset
from spade.evaluation import (
    TokenCodecWERScorer,
    count_parameters,
    measure_generation_latency,
    model_depth,
)


@dataclass
class EvalConfig:
    subset_size: int = 8
    max_new_tokens: int = 128
    hop_seconds: float = 0.08
    n_runs: int = 3
    device: str = "auto"
    data: dict[str, Any] = field(default_factory=dict)


def evaluate_checkpoint(
    model_path: str,
    cfg: EvalConfig,
    data_cfg: dict[str, Any],
) -> dict:
    device = resolve_device(cfg.device)
    model, _ = load_checkpoint(model_path, device=device)
    tokenizer = CharTokenizer()
    codec = ToySpeechCodec(
        alphabet=tokenizer.alphabet,
        code_vocab_size=int(data_cfg.get("code_vocab_size", 512)),
    )
    texts = make_synthetic_texts(
        max(int(data_cfg.get("num_eval_samples", 24)), cfg.subset_size),
        seed=int(data_cfg.get("seed", 0)) + 1,
        max_words=int(data_cfg.get("max_words", 12)),
    )
    ds = SyntheticTTSDataset(
        texts,
        codec,
        tokenizer,
        num_speakers=int(data_cfg.get("num_speakers", 4)),
        use_prompt=bool(data_cfg.get("use_prompt", False)),
        max_len=int(data_cfg.get("max_len", 128)),
    )
    samples = [
        {
            "input_ids": ds[i]["input_ids"],
            "speaker_ids": ds[i]["speaker_ids"],
            "text": ds[i]["text"],
            "cond_len": int(ds[i]["cond_len"]),
        }
        for i in range(cfg.subset_size)
    ]
    scorer = TokenCodecWERScorer(
        codec=codec, tokenizer=tokenizer, max_new_tokens=cfg.max_new_tokens, device=device
    )
    wers = [scorer.score(model, s) for s in samples]
    latency = measure_generation_latency(
        model,
        samples[0]["input_ids"][: samples[0]["cond_len"]].unsqueeze(0),
        max_new_tokens=cfg.max_new_tokens,
        n_runs=cfg.n_runs,
        device=device,
        hop_seconds=cfg.hop_seconds,
    )
    metrics = {
        "wer": round(float(np.mean(wers)), 4),
        "wer_per_sample": [round(w, 4) for w in wers],
        "params": count_parameters(model),
        "depth": model_depth(model),
        **latency,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an LLM-TTS checkpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    cfg = dataclass_from_dict(EvalConfig, load_yaml(args.config), "EvalConfig")
    metrics = evaluate_checkpoint(args.model, cfg, cfg.data)
    print(json.dumps(metrics, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()
