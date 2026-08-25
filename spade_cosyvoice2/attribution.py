"""Attribute audio degradation to the LLM (token errors) vs Flow/vocoder.

Two complementary measurements on the torch path:

1. **Token error rate (TER)**: the teacher / distilled LLM's generated speech
   tokens vs the reference utterance's ground-truth tokens. If the distilled
   LLM's TER is much higher, the LLM is the source of degradation.
2. **GT-token reconstruction**: audio synthesized from the *reference* tokens
   through the same Flow + HiFi-GAN. Comparing its spectrum to the real audio
   isolates how much quality the Flow/vocoder itself loses (independent of
   the LLM).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import soundfile as sf
import torch

from spade.config_utils import dataclass_from_dict, load_yaml, resolve_device
from spade_cosyvoice2.model_utils import (
    load_cosyvoice2_with_llm,
    synthesize_from_tokens,
)
from spade_cosyvoice2.wli import load_eval_samples


@dataclass
class Variant:
    name: str
    llm_pt: str
    retained: list[int] = field(default_factory=list)


@dataclass
class AttributionConfig:
    eval_list: str
    subset_size: int = 8
    model_dir: str = ""
    device: str = "auto"
    out_dir: str = "outputs/cosyvoice2/attribution"
    variants: list[dict[str, Any]] = field(default_factory=list)
    gt_flow_steps: int = 10

    def __post_init__(self) -> None:
        if not self.model_dir:
            from spade_cosyvoice2.paths import model_dir

            self.model_dir = str(model_dir())


def _edit_distance(a: list, b: list) -> int:
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, y in enumerate(b, start=1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (x != y))
            prev = cur
    return dp[-1]


def token_error_rate(reference: list[int], hypothesis: list[int]) -> float:
    if not reference:
        return 1.0 if hypothesis else 0.0
    return _edit_distance(reference, hypothesis) / len(reference)


def _ref_token_map(eval_list: str) -> dict[str, list[int]]:
    with open(eval_list) as fh:
        paths = [line.strip() for line in fh if line.strip()]
    out: dict[str, list[int]] = {}
    for p in paths:
        for r in pq.read_table(p).to_pylist():
            out[str(r["utt"])] = [int(t) for t in r["speech_token"]]
    return out


def generate_tokens(cosyvoice, sample: dict, device: str) -> list[int]:
    mi = cosyvoice.frontend.frontend_zero_shot(
        sample["text"], sample["prompt_text"], sample["prompt_wav"],
        cosyvoice.sample_rate, "",
    )
    tokens: list[int] = []
    for t in cosyvoice.model.llm.inference(
        text=mi["text"].to(device),
        text_len=mi["text_len"].to(device),
        prompt_text=mi["prompt_text"].to(device),
        prompt_text_len=mi["prompt_text_len"].to(device),
        prompt_speech_token=mi["llm_prompt_speech_token"].to(device),
        prompt_speech_token_len=mi["llm_prompt_speech_token_len"].to(device),
        embedding=mi["llm_embedding"].to(device),
    ):
        tokens.append(int(t))
    return tokens


def run(cfg: AttributionConfig) -> dict:
    device = resolve_device(cfg.device)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = load_eval_samples(cfg.eval_list, cfg.subset_size)
    ref_tokens = _ref_token_map(cfg.eval_list)
    variants = [dataclass_from_dict(Variant, v, "variant") for v in cfg.variants]
    if not variants:
        raise ValueError("at least one LLM variant is required")

    report: dict[str, Any] = {
        "eval_list": cfg.eval_list,
        "subset_size": len(samples),
        "variants": {},
    }

    # 1) token error rates per LLM variant
    for var in variants:
        cv = load_cosyvoice2_with_llm(
            cfg.model_dir, llm_pt=var.llm_pt, retained=var.retained or None
        )
        ters: list[float] = []
        token_paths = []
        for sample in samples:
            hyp = generate_tokens(cv, sample, device)
            ref = ref_tokens.get(sample["utt"], [])
            ters.append(token_error_rate(ref, hyp))
            tok_file = out / f"{var.name}_{sample['utt']}.json"
            tok_file.write_text(json.dumps({"ref_len": len(ref), "hyp": hyp}))
            token_paths.append(str(tok_file))
        report["variants"][var.name] = {
            "ter_mean": round(float(sum(ters) / len(ters)), 4),
            "ter_per_utt": [round(t, 4) for t in ters],
            "token_files": token_paths,
        }
        print(
            f"[attr] {var.name:<10} TER={report['variants'][var.name]['ter_mean']:.4f}"
        )

    # 2) GT-token reconstruction (reference tokens -> Flow -> HiFi-GAN)
    cv = load_cosyvoice2_with_llm(cfg.model_dir)  # teacher, flow/hift only
    gt_dir = out / "gt_tokens"
    gt_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        ref = ref_tokens.get(sample["utt"], [])
        if not ref:
            continue
        speech = synthesize_from_tokens(
            cv, ref, sample["prompt_text"], sample["prompt_wav"],
            flow_steps=cfg.gt_flow_steps,
        )
        sf.write(
            str(gt_dir / f"{sample['utt']}.wav"),
            speech.squeeze(0).cpu().numpy(),
            cv.sample_rate,
        )
    report["gt_tokens"] = {
        "flow_steps": cfg.gt_flow_steps,
        "wav_dir": str(gt_dir),
        "n": len(samples),
    }
    print(f"[attr] GT-token reconstruction -> {gt_dir}")

    report_path = out / "attribution_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[attr] report -> {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="SPADE/CosyVoice2 attribution")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = dataclass_from_dict(AttributionConfig, load_yaml(args.config), "AttributionConfig")
    run(cfg)


if __name__ == "__main__":
    main()
