"""Distillation training loop: restore a pruned student from the teacher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spade.distill.losses import (
    attention_mse_loss,
    cross_entropy_loss,
    embedding_mse_loss,
    latent_mse_loss,
    skew_kl_loss,
    spade_composite_loss,
)
from spade.models import LLMTTSBackbone


def distill_loss(
    teacher_out: dict,
    student_out: dict,
    labels: torch.Tensor,
    teacher_targets: list[int],
    alpha: float = 0.25,
    beta: float = 0.1,
    skew_mode: str = "forward",
    use_ce: bool = True,
    use_logit: bool = True,
    use_latent: bool = True,
    use_attention: bool = True,
    use_embedding: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute all SPADE loss components for one batch.

    ``teacher_targets[j]`` is the teacher layer index whose latent and
    attention maps student layer ``j`` is aligned against (from
    :func:`spade.distill.matching.dynamic_teacher_targets`).
    """
    components: dict[str, torch.Tensor] = {}

    if use_ce:
        components["ce"] = cross_entropy_loss(student_out["logits"], labels)
    else:
        components["ce"] = labels.sum() * 0.0

    if use_logit:
        components["logit"] = skew_kl_loss(
            student_out["logits"],
            teacher_out["logits"],
            labels=labels,
            beta=beta,
            mode=skew_mode,
        )
    else:
        components["logit"] = labels.sum() * 0.0

    latent_terms: list[torch.Tensor] = []
    attention_terms: list[torch.Tensor] = []
    for j, t in enumerate(teacher_targets):
        if use_latent:
            latent_terms.append(
                latent_mse_loss(
                    student_out["hidden_states"][j],
                    teacher_out["hidden_states"][t],
                )
            )
        if use_attention:
            attention_terms.append(
                attention_mse_loss(
                    student_out["attentions"][j],
                    teacher_out["attentions"][t],
                )
            )
    components["latent"] = (
        torch.stack(latent_terms).mean() if latent_terms else labels.sum() * 0.0
    )
    components["attention"] = (
        torch.stack(attention_terms).mean() if attention_terms else labels.sum() * 0.0
    )

    if use_embedding:
        components["embedding"] = embedding_mse_loss(
            student_out["embedding_out"],
            teacher_out["embedding_out"],
        )
    else:
        components["embedding"] = labels.sum() * 0.0

    components["total"] = spade_composite_loss(
        components["ce"],
        components["logit"],
        components["latent"],
        components["attention"],
        components["embedding"],
        alpha=alpha,
    )
    return components


@dataclass
class DistillationConfig:
    """Hyperparameters for SPADE distillation."""

    alpha: float = 0.25          # paper's supervised/distillation balance
    beta: float = 0.1            # DistiLLM skew-KL teacher mixing weight
    skew_mode: str = "forward"   # 'forward' (KL(p||mix)) or 'reverse'
    use_ce: bool = True
    use_logit: bool = True
    use_latent: bool = True
    use_attention: bool = True
    use_embedding: bool = True
    matching: str = "dynamic"    # 'dynamic' or 'static'
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    epochs: int = 1
    device: str | torch.device = "cpu"
    log_every: int = 10

    def __post_init__(self) -> None:
        if self.matching not in ("dynamic", "static"):
            raise ValueError("matching must be 'dynamic' or 'static'")


class DistillTrainer:
    """Fine-tune a pruned student under a frozen teacher with SPADE's loss."""

    def __init__(
        self,
        teacher: LLMTTSBackbone,
        student: LLMTTSBackbone,
        config: DistillationConfig,
        teacher_targets: Optional[list[int]] = None,
    ) -> None:
        self.teacher = teacher.to(config.device).eval()
        self.student = student.to(config.device)
        self.config = config
        if teacher_targets is None:
            from spade.distill.matching import dynamic_teacher_targets, static_teacher_targets

            if config.matching == "dynamic":
                teacher_targets = dynamic_teacher_targets(
                    student.retained_indices, teacher.config.n_layer
                )
            else:
                teacher_targets = static_teacher_targets(student.retained_indices)
        self.teacher_targets = teacher_targets
        if len(self.teacher_targets) != student.config.n_layer:
            raise ValueError(
                "teacher_targets length must equal student n_layer "
                f"({len(self.teacher_targets)} != {student.config.n_layer})"
            )

        # Freeze the teacher; only the student is optimized.
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def _forward_pair(self, batch: dict) -> tuple[dict, dict]:
        input_ids = batch["input_ids"].to(self.config.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.config.device)
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_hidden_states=True,
            return_attentions=True,
        )
        with torch.no_grad():
            teacher_out = self.teacher(**kwargs)
        student_out = self.student(**kwargs)
        return teacher_out, student_out

    def train(
        self,
        dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ) -> dict:
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        history: dict[str, list] = {"train": [], "val": []}
        step = 0
        for epoch in range(1, self.config.epochs + 1):
            self.student.train()
            epoch_metrics: dict[str, float] = {
                "ce": 0.0, "logit": 0.0, "latent": 0.0,
                "attention": 0.0, "embedding": 0.0, "total": 0.0,
            }
            n_batches = 0
            for batch in dataloader:
                teacher_out, student_out = self._forward_pair(batch)
                comp = distill_loss(
                    teacher_out,
                    student_out,
                    batch["labels"].to(self.config.device),
                    teacher_targets=self.teacher_targets,
                    alpha=self.config.alpha,
                    beta=self.config.beta,
                    skew_mode=self.config.skew_mode,
                    use_ce=self.config.use_ce,
                    use_logit=self.config.use_logit,
                    use_latent=self.config.use_latent,
                    use_attention=self.config.use_attention,
                    use_embedding=self.config.use_embedding,
                )
                optimizer.zero_grad()
                comp["total"].backward()
                nn.utils.clip_grad_norm_(
                    self.student.parameters(), self.config.grad_clip
                )
                optimizer.step()
                for k in epoch_metrics:
                    epoch_metrics[k] += float(comp[k].detach())
                n_batches += 1
                step += 1
                if step % self.config.log_every == 0:
                    print(
                        f"[distill] epoch {epoch} step {step} "
                        + " ".join(f"{k}={v:.4f}" for k, v in comp.items())
                    )
            if n_batches:
                epoch_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}
            history["train"].append(epoch_metrics)

            if val_dataloader is not None:
                history["val"].append(self.evaluate(val_dataloader))
        return history

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        self.student.eval()
        totals = {"ce": 0.0, "logit": 0.0, "latent": 0.0,
                  "attention": 0.0, "embedding": 0.0, "total": 0.0}
        n = 0
        for batch in dataloader:
            teacher_out, student_out = self._forward_pair(batch)
            comp = distill_loss(
                teacher_out,
                student_out,
                batch["labels"].to(self.config.device),
                teacher_targets=self.teacher_targets,
                alpha=self.config.alpha,
                beta=self.config.beta,
                skew_mode=self.config.skew_mode,
            )
            for k in totals:
                totals[k] += float(comp[k])
            n += 1
        return {k: (v / n if n else 0.0) for k, v in totals.items()}

