"""SPADE distillation training for the CosyVoice2 LLM."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from spade.config_utils import dataclass_from_dict, load_yaml, resolve_device
from spade.distill import distill_loss
from spade.distill.matching import dynamic_teacher_targets, static_teacher_targets
from spade_cosyvoice2.model_utils import build_dataloader, build_llm, load_configs, spade_forward
from spade_cosyvoice2.prune_llm import count_qwen_layers


@dataclass
class CosyVoice2DistillConfig:
    train_list: str
    llm_pt: str                     # teacher llm.pt (24 layers)
    student_llm_pt: str             # pruned student llm.pt
    retained: list[int] = field(default_factory=list)
    model_dir: str = ""
    epochs: int = 1
    lr: float = 1e-5
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    accum_grad: int = 2
    alpha: float = 0.25
    beta: float = 0.1
    skew_mode: str = "forward"
    use_ce: bool = True
    use_logit: bool = True
    use_latent: bool = True
    use_attention: bool = True
    use_embedding: bool = True
    matching: str = "dynamic"
    device: str = "auto"
    seed: int = 0
    log_every: int = 10
    save_every: int = 0          # 0 = save only at the end
    use_amp: bool = True         # bf16 autocast to fit 24GB L4
    resume: bool = True          # continue from out_llm_pt if it exists
    out_llm_pt: str = ""

    def __post_init__(self) -> None:
        if not self.model_dir:
            from spade_cosyvoice2.paths import model_dir

            self.model_dir = str(model_dir())
        if not self.out_llm_pt:
            self.out_llm_pt = str(
                Path(self.student_llm_pt).with_name("distilled_llm.pt")
            )


def distill(config: CosyVoice2DistillConfig) -> dict:
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    use_amp = config.use_amp and device.startswith("cuda")
    configs = load_configs(config.model_dir)

    teacher = build_llm(configs, device, llm_pt=config.llm_pt)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    student = build_llm(
        configs, device, llm_pt=config.student_llm_pt, retained=config.retained
    )
    start_epoch = 1
    if config.resume and Path(config.out_llm_pt).exists():
        saved = torch.load(config.out_llm_pt, map_location=device, weights_only=True)
        student.load_state_dict(saved, strict=False)
        print(f"[distill] resumed from {config.out_llm_pt}")
    if config.matching == "dynamic":
        targets = dynamic_teacher_targets(config.retained, count_qwen_layers(
            torch.load(config.llm_pt, map_location="cpu", weights_only=True)))
    else:
        targets = static_teacher_targets(config.retained)
    print(f"[distill] teacher depth={count_qwen_layers(torch.load(config.llm_pt, map_location='cpu', weights_only=True))}, "
          f"student depth={len(config.retained)}, targets={targets}")

    loader = build_dataloader(configs, config.train_list)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    history: list[dict] = []
    step = 0
    for epoch in range(start_epoch, config.epochs + 1):
        student.train()
        accum = 0.0
        n_batches = 0
        optimizer.zero_grad()
        for batch in loader:
            amp_ctx = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if use_amp
                else torch.nullcontext()
            )
            with amp_ctx:
                with torch.no_grad():
                    teacher_out = spade_forward(teacher, batch, device)
                student_out = spade_forward(student, batch, device)
                comp = distill_loss(
                    teacher_out,
                    student_out,
                    student_out["lm_target_masked"],
                    teacher_targets=targets,
                    alpha=config.alpha,
                    beta=config.beta,
                    skew_mode=config.skew_mode,
                    use_ce=config.use_ce,
                    use_logit=config.use_logit,
                    use_latent=config.use_latent,
                    use_attention=config.use_attention,
                    use_embedding=config.use_embedding,
                )
            (comp["total"] / config.accum_grad).backward()
            accum += float(comp["total"].detach())
            n_batches += 1
            step += 1
            if step % config.accum_grad == 0:
                nn.utils.clip_grad_norm_(student.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            if config.log_every and step % config.log_every == 0:
                print(f"[distill] epoch {epoch} step {step} "
                      + " ".join(f"{k}={v:.4f}" for k, v in comp.items()))
            if config.save_every and step % config.save_every == 0:
                torch.save(student.state_dict(), config.out_llm_pt)
                print(f"[distill] checkpoint saved at step {step}")
            del teacher_out, student_out, comp
        if config.save_every:
            torch.save(student.state_dict(), config.out_llm_pt)
        history.append({"epoch": epoch, "avg_total": accum / max(n_batches, 1)})
        print(f"[distill] epoch {epoch} done: avg total={accum / max(n_batches, 1):.4f}")

    out = Path(config.out_llm_pt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), out)
    (out.with_suffix(".json")).write_text(
        json.dumps({"retained_indices": config.retained, "history": history}, indent=2)
    )
    print(f"[distill] saved {out}")
    return {"history": history, "targets": targets, "out_llm_pt": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description="SPADE distillation for CosyVoice2")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = dataclass_from_dict(
        CosyVoice2DistillConfig, load_yaml(args.config), "CosyVoice2DistillConfig"
    )
    distill(cfg)


if __name__ == "__main__":
    main()
