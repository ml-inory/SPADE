"""One-command SPADE pipeline: train -> WLI prune -> distill -> evaluate."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spade.checkpoints import save_checkpoint
from spade.config_utils import dataclass_from_dict, load_yaml, resolve_device
from spade.distill_cli import DistillRunConfig, distill_student
from spade.evaluate import EvalConfig, evaluate_checkpoint
from spade.prune import PruneConfig, prune_teacher
from spade.train import TrainConfig, train_teacher


@dataclass
class PipelineConfig:
    output_dir: str = "outputs"
    device: str = "auto"
    seed: int = 0
    teacher: dict[str, Any] = field(default_factory=dict)
    prune: dict[str, Any] = field(default_factory=dict)
    distill: dict[str, Any] = field(default_factory=dict)
    eval: dict[str, Any] = field(default_factory=dict)


def run_pipeline(cfg: PipelineConfig) -> dict:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)

    # --- Stage 1: train the teacher ---
    train_cfg = dataclass_from_dict(TrainConfig, dict(cfg.teacher), "teacher")
    train_cfg.device = device
    print("=" * 60, "\n[1/4] training teacher\n", "=" * 60, sep="")
    teacher, meta = train_teacher(train_cfg)
    teacher_path = out / "teacher.pt"
    save_checkpoint(teacher, teacher_path, meta=meta)

    # --- Stage 2: WLI pruning ---
    print("=" * 60, "\n[2/4] WLI pruning\n", "=" * 60, sep="")
    prune_cfg = dataclass_from_dict(PruneConfig, dict(cfg.prune), "prune")
    prune_cfg.device = device
    prune_cfg.data = dict(train_cfg.data)
    _, student_path = prune_teacher(str(teacher_path), prune_cfg, out)
    with open(out / "wli_report.json", encoding="utf-8") as fh:
        prune_report = json.load(fh)

    # --- Stage 3: baseline evaluation (teacher vs pruned student) ---
    print("=" * 60, "\n[3/4] baseline evaluation\n", "=" * 60, sep="")
    eval_cfg = dataclass_from_dict(EvalConfig, dict(cfg.eval), "eval")
    eval_cfg.device = device
    eval_cfg.data = dict(train_cfg.data)
    teacher_metrics = evaluate_checkpoint(str(teacher_path), eval_cfg, dict(train_cfg.data))
    student_metrics = evaluate_checkpoint(student_path, eval_cfg, dict(train_cfg.data))
    print(f"teacher WER={teacher_metrics['wer']:.4f} | "
          f"pruned WER={student_metrics['wer']:.4f} | "
          f"params {teacher_metrics['params']} -> {student_metrics['params']}")

    # --- Stage 4: distillation ---
    print("=" * 60, "\n[4/4] adaptive distillation\n", "=" * 60, sep="")
    distill_cfg = dataclass_from_dict(DistillRunConfig, dict(cfg.distill), "distill")
    distill_cfg.device = device
    distill_cfg.data = dict(train_cfg.data)
    distilled_path = out / "distilled.pt"
    history = distill_student(
        str(teacher_path), student_path, distill_cfg, str(distilled_path)
    )
    distilled_metrics = evaluate_checkpoint(
        str(distilled_path), eval_cfg, dict(train_cfg.data)
    )

    speedup = {
        "depth_ratio": round(
            teacher_metrics["depth"] / distilled_metrics["depth"], 3
        ),
        "param_ratio": round(
            teacher_metrics["params"] / distilled_metrics["params"], 3
        ),
        "rtf_ratio": round(
            teacher_metrics["synthetic_rtf"] / distilled_metrics["synthetic_rtf"], 3
        ),
    }
    report = {
        "teacher": {"metrics": teacher_metrics, "train_history": meta["history"]},
        "pruning": prune_report,
        "student_before_distill": {"metrics": student_metrics},
        "student_after_distill": {"metrics": distilled_metrics, "history": history},
        "speedup": speedup,
        "paths": {
            "teacher": str(teacher_path),
            "student_pruned": student_path,
            "student_distilled": str(distilled_path),
            "wli_report": str(out / "wli_report.json"),
            "pipeline_report": str(out / "pipeline_report.json"),
        },
    }
    with open(out / "pipeline_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("=" * 60)
    print("SPADE pipeline finished.")
    print(f"  teacher : depth={teacher_metrics['depth']} params={teacher_metrics['params']} "
          f"WER={teacher_metrics['wer']:.4f} RTF={teacher_metrics['synthetic_rtf']}")
    print(f"  pruned  : depth={student_metrics['depth']} params={student_metrics['params']} "
          f"WER={student_metrics['wer']:.4f}")
    print(f"  distilled: depth={distilled_metrics['depth']} "
          f"params={distilled_metrics['params']} WER={distilled_metrics['wer']:.4f} "
          f"RTF={distilled_metrics['synthetic_rtf']}")
    print(f"  speedup : depth x{speedup['depth_ratio']}, params x{speedup['param_ratio']}, "
          f"RTF x{speedup['rtf_ratio']}")
    print(f"  report  : {out / 'pipeline_report.json'}")
    print("=" * 60)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full SPADE pipeline")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = dataclass_from_dict(PipelineConfig, load_yaml(args.config), "PipelineConfig")
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
