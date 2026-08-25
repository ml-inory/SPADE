"""Controlled ablation: isolate SPADE-LLM vs Flow-step effects on audio quality.

Every condition synthesizes the same evaluation utterances with a specific
LLM checkpoint and Flow solver step count, then reports Whisper WER, RTF,
and saves the waveforms for listening. This is the torch-side controlled
experiment used to locate where quality degrades relative to the original
CosyVoice2 (the golden standard).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from spade.config_utils import dataclass_from_dict, load_yaml, resolve_device
from spade_cosyvoice2.model_utils import load_cosyvoice2_with_llm, synthesize
from spade_cosyvoice2.wli import CosyVoice2WERScorer, load_eval_samples


@dataclass
class Condition:
    name: str
    llm_pt: str
    retained: list[int] = field(default_factory=list)
    flow_steps: int = 10


@dataclass
class AblationConfig:
    eval_list: str
    subset_size: int = 8
    whisper_model: str = "base"
    model_dir: str = ""
    device: str = "auto"
    out_dir: str = "outputs/cosyvoice2/ablation"
    conditions: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.model_dir:
            from spade_cosyvoice2.paths import model_dir

            self.model_dir = str(model_dir())


def run_ablation(cfg: AblationConfig) -> dict:
    device = resolve_device(cfg.device)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = load_eval_samples(cfg.eval_list, cfg.subset_size)
    conditions = [dataclass_from_dict(Condition, c, "condition") for c in cfg.conditions]
    if not conditions:
        raise ValueError("at least one condition is required")

    # Load each distinct LLM once and evaluate all its flow-step variants.
    import soundfile as sf

    cosyvoice_cache: dict[str, Any] = {}
    report: dict[str, Any] = {"eval_list": cfg.eval_list, "subset_size": len(samples)}
    for cond in conditions:
        key = cond.llm_pt
        if key not in cosyvoice_cache:
            cosyvoice_cache[key] = load_cosyvoice2_with_llm(
                cfg.model_dir, llm_pt=cond.llm_pt, retained=cond.retained or None
            )
        cv = cosyvoice_cache[key]
        scorer = CosyVoice2WERScorer(cv, cfg.whisper_model, device)
        cond_dir = out / cond.name
        cond_dir.mkdir(parents=True, exist_ok=True)

        wers, rtfs, audios = [], [], []
        for sample in samples:
            try:
                start = time.perf_counter()
                speech = synthesize(
                    cv, sample["text"], sample["prompt_text"], sample["prompt_wav"],
                    flow_steps=cond.flow_steps,
                )
                elapsed = time.perf_counter() - start
                wer = scorer.score_speech(speech, sample["text"])
                wers.append(wer)
                rtfs.append(elapsed / (speech.shape[1] / cv.sample_rate))
                sf.write(
                    str(cond_dir / f"{sample['utt']}.wav"),
                    speech.squeeze(0).cpu().numpy(),
                    cv.sample_rate,
                )
                audios.append(speech.shape[1] / cv.sample_rate)
            except Exception as exc:  # degenerate output -> fully wrong
                print(f"[ablation] {cond.name}/{sample['utt']} failed ({type(exc).__name__})")
                wers.append(1.0)
                rtfs.append(0.0)
                audios.append(0.0)

        report[cond.name] = {
            "flow_steps": cond.flow_steps,
            "depth": cv.model.llm.llm.model.config.num_hidden_layers,
            "params": sum(p.numel() for p in cv.model.llm.llm.model.parameters()),
            "wer": round(float(np.mean(wers)), 4),
            "wer_per_sample": [round(w, 4) for w in wers],
            "rtf": round(float(np.mean(rtfs)), 4),
            "avg_audio_seconds": round(float(np.mean(audios)), 2),
            "wav_dir": str(cond_dir),
        }
        print(
            f"[ablation] {cond.name:<14} flow={cond.flow_steps:>3} "
            f"WER={report[cond.name]['wer']:.4f} RTF={report[cond.name]['rtf']:.3f}"
        )

    report_path = out / "ablation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[ablation] report -> {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="SPADE/CosyVoice2 ablation")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = dataclass_from_dict(AblationConfig, load_yaml(args.config), "AblationConfig")
    run_ablation(cfg)


if __name__ == "__main__":
    main()

