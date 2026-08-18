"""WLI pruning stage: importance analysis + layer removal."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from spade.checkpoints import load_checkpoint, save_checkpoint
from spade.config_utils import dataclass_from_dict, load_yaml, resolve_device
from spade.data import CharTokenizer, ToySpeechCodec, make_synthetic_texts
from spade.data.dataset import SyntheticTTSDataset
from spade.evaluation import TokenCodecWERScorer
from spade.pruning import (
    ImportanceReport,
    compute_cli,
    compute_wli,
    prune_layers,
    select_layers_to_keep,
)


@dataclass
class PruneConfig:
    target_layers: int
    eval_subset_size: int = 8
    force_keep_first_last: bool = True
    compute_cli: bool = True
    max_new_tokens: int = 128
    device: str = "auto"
    data: dict[str, Any] = field(default_factory=dict)


def eval_samples(data_cfg: dict[str, Any], subset_size: int):
    tokenizer = CharTokenizer()
    codec = ToySpeechCodec(
        alphabet=tokenizer.alphabet,
        code_vocab_size=int(data_cfg.get("code_vocab_size", 512)),
    )
    texts = make_synthetic_texts(
        max(int(data_cfg.get("num_eval_samples", 24)), subset_size),
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
    return tokenizer, codec, [
        {
            "input_ids": ds[i]["input_ids"],
            "speaker_ids": ds[i]["speaker_ids"],
            "text": ds[i]["text"],
            "cond_len": int(ds[i]["cond_len"]),
        }
        for i in range(subset_size)
    ]


def prune_teacher(
    teacher_path: str,
    cfg: PruneConfig,
    out_dir: str | Path,
) -> tuple[ImportanceReport, str]:
    device = resolve_device(cfg.device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher, meta = load_checkpoint(teacher_path, device=device)
    tokenizer, codec, samples = eval_samples(cfg.data, cfg.eval_subset_size)

    print(f"[prune] teacher depth={teacher.config.n_layer}, "
          f"params={sum(p.numel() for p in teacher.parameters()):,}")
    scorer = TokenCodecWERScorer(
        codec=codec,
        tokenizer=tokenizer,
        max_new_tokens=cfg.max_new_tokens,
        device=device,
    )
    baseline_wer = float(np.mean([scorer.score(teacher, s) for s in samples]))
    print(f"[prune] teacher WER on {len(samples)} eval samples: {baseline_wer:.4f}")

    wli = compute_wli(teacher, scorer, samples, show_progress=True)
    cli = (
        compute_cli(teacher, samples, batch_size=4, device=device)
        if cfg.compute_cli
        else None
    )
    retained = select_layers_to_keep(
        wli,
        cfg.target_layers,
        force_keep_first_last=cfg.force_keep_first_last,
    )
    report = ImportanceReport(
        wli=wli,
        cli=cli,
        retained_indices=retained,
        target_layers=cfg.target_layers,
    )
    print(report.summary())
    student = prune_layers(teacher, retained).to(device)
    student_path = out_dir / "student_pruned.pt"
    save_checkpoint(
        student,
        student_path,
        meta={"wli": wli.tolist(), "cli": None if cli is None else cli.tolist(),
              "retained_indices": retained, "baseline_wer": baseline_wer},
    )
    report_json = {
        "wli": wli.tolist(),
        "cli": None if cli is None else cli.tolist(),
        "baseline_wer": baseline_wer,
        "target_layers": cfg.target_layers,
        "retained_indices": retained,
    }
    with open(out_dir / "wli_report.json", "w", encoding="utf-8") as fh:
        import json

        json.dump(report_json, fh, indent=2)
    print(f"[prune] saved student -> {student_path}")
    return report, str(student_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="WLI pruning stage")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = dataclass_from_dict(PruneConfig, load_yaml(args.config), "PruneConfig")
    prune_teacher(args.teacher, cfg, args.out)


if __name__ == "__main__":
    main()
