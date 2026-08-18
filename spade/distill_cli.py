"""Distillation stage CLI: restore the pruned student with a frozen teacher."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from spade.checkpoints import load_checkpoint, save_checkpoint
from spade.config_utils import dataclass_from_dict, load_yaml, resolve_device
from spade.distill import DistillationConfig, DistillTrainer
from spade.train import build_benchmark


@dataclass
class DistillRunConfig:
    data: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 16
    device: str = "auto"
    distill: dict[str, Any] = field(default_factory=dict)


def distill_student(
    teacher_path: str,
    student_path: str,
    cfg: DistillRunConfig,
    out_path: str | Path,
) -> dict:
    device = resolve_device(cfg.device)
    teacher, _ = load_checkpoint(teacher_path, device=device)
    student, _ = load_checkpoint(student_path, device=device)
    _, _, train_ds, _ = build_benchmark(cfg.data)
    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate(b),
    )
    dcfg = dataclass_from_dict(
        DistillationConfig,
        dict(cfg.distill),
        "distill",
    )
    dcfg.device = device
    trainer = DistillTrainer(teacher, student, dcfg)
    print(
        f"[distill] teacher depth={teacher.config.n_layer}, "
        f"student depth={student.config.n_layer}, "
        f"matching={dcfg.matching}, targets={trainer.teacher_targets}"
    )
    history = trainer.train(loader)
    save_checkpoint(
        student,
        out_path,
        meta={
            "history": history,
            "teacher_targets": trainer.teacher_targets,
            "distill_config": asdict(dcfg),
        },
    )
    with open(Path(out_path).with_suffix(".history.json"), "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    print(f"[distill] saved distilled student -> {out_path}")
    return history


def _collate(batch):
    from spade.data import collate_llmtts_batch

    return collate_llmtts_batch(batch)


def main() -> None:
    parser = argparse.ArgumentParser(description="SPADE distillation stage")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = dataclass_from_dict(DistillRunConfig, load_yaml(args.config), "DistillRunConfig")
    distill_student(args.teacher, args.student, cfg, args.out)


if __name__ == "__main__":
    main()
