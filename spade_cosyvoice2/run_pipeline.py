"""End-to-end SPADE-on-CosyVoice2 runner.

train teacher already exists (official CosyVoice2-0.5B); this pipeline runs:
  data prep -> WLI -> prune (24->12) -> distill -> evaluate (teacher/pruned/distilled)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from spade.config_utils import dataclass_from_dict, load_yaml
from spade_cosyvoice2.data_prep import DataPrepConfig, prepare
from spade_cosyvoice2.evaluate import EvalConfig, evaluate
from spade_cosyvoice2.prune_llm import PruneConfig, prune_llm_checkpoint
from spade_cosyvoice2.wli import WLIConfig, run_wli


@dataclass
class PipelineConfig:
    model_dir: str = ""
    out_dir: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    wli: dict[str, Any] = field(default_factory=dict)
    prune: dict[str, Any] = field(default_factory=dict)
    distill: dict[str, Any] = field(default_factory=dict)
    eval: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_dir:
            from spade_cosyvoice2.paths import model_dir

            self.model_dir = str(model_dir())
        if not self.out_dir:
            self.out_dir = str(Path(self.model_dir).parent / "spade_outputs")


def run_pipeline(cfg: PipelineConfig) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. data prep
    data_cfg = dataclass_from_dict(DataPrepConfig, dict(cfg.data), "data")
    data_meta = prepare(data_cfg)

    # 2. WLI on the full 24-layer LLM
    wli_cfg = dataclass_from_dict(WLIConfig, dict(cfg.wli), "wli")
    wli_cfg.eval_list = data_meta["eval_list"]
    wli_report = run_wli(wli_cfg)
    wli = wli_report["wli"]

    # 3. prune to target depth
    prune_cfg = dataclass_from_dict(PruneConfig, dict(cfg.prune), "prune")

    if not prune_cfg.retained:
        from spade.pruning import select_layers_to_keep

        prune_cfg.retained = select_layers_to_keep(
            np.array(wli),
            prune_cfg.target_layers,
            force_keep_first_last=prune_cfg.force_keep_first_last,
        )
    pruned_llm = out / "pruned_llm.pt"
    retained = prune_llm_checkpoint(
        f"{cfg.model_dir}/llm.pt",
        prune_cfg.target_layers,
        pruned_llm,
        retained=prune_cfg.retained,
    )

    # 4. distillation
    from spade_cosyvoice2.distill import CosyVoice2DistillConfig, distill

    distill_cfg = CosyVoice2DistillConfig(
        train_list=data_meta["train_list"],
        llm_pt=f"{cfg.model_dir}/llm.pt",
        student_llm_pt=str(pruned_llm),
        retained=retained,
        model_dir=cfg.model_dir,
        out_llm_pt=str(out / "distilled_llm.pt"),
        **{k: v for k, v in cfg.distill.items()
           if k not in ("train_list", "llm_pt", "student_llm_pt", "retained", "model_dir", "out_llm_pt")},
    )
    distill_result = distill(distill_cfg)

    # 5. evaluate teacher / pruned / distilled
    eval_base = dict(cfg.eval)
    eval_base["eval_list"] = data_meta["eval_list"]
    teacher_metrics = evaluate(
        EvalConfig(llm_pt=f"{cfg.model_dir}/llm.pt", **eval_base)
    )
    pruned_metrics = evaluate(
        EvalConfig(llm_pt=str(pruned_llm), retained=retained, **eval_base)
    )
    distilled_metrics = evaluate(
        EvalConfig(
            llm_pt=str(out / "distilled_llm.pt"), retained=retained, **eval_base
        )
    )

    report = {
        "data": data_meta,
        "wli": wli_report,
        "pruning": {"retained_indices": retained},
        "teacher": teacher_metrics,
        "pruned": pruned_metrics,
        "distilled": distilled_metrics,
        "distillation_history": distill_result["history"],
        "paths": {
            "pruned_llm": str(pruned_llm),
            "distilled_llm": str(out / "distilled_llm.pt"),
        },
    }
    report_path = out / "pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[pipeline] report -> {report_path}")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="SPADE-on-CosyVoice2 pipeline")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = dataclass_from_dict(PipelineConfig, load_yaml(args.config), "PipelineConfig")
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
