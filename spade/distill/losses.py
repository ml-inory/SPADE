"""Losses for SPADE's adaptive multi-level knowledge distillation.

The paper's composite objective (Eq. 2) is::

    L = alpha * L_CE + (1 - alpha) / 4 * (L_logit + L_l + L_a + L_e)

with ``alpha = 0.25``. ``L_logit`` aligns output distributions with the
Skew KL divergence from DistiLLM (Ko et al., 2024) -- a mixed-distribution
KL that avoids the instability of plain forward/reverse KL for
autoregressive models. ``L_l``, ``L_a`` and ``L_e`` are MSE losses on
intermediate latents, attention maps, and embedding outputs.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _valid_mask(labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if labels is None:
        return None
    return labels != -100


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Supervised cross-entropy over the code-token vocabulary."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


def skew_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    beta: float = 0.1,
    mode: str = "forward",
) -> torch.Tensor:
    """Skew KL divergence between teacher ``p`` and student ``q`` logits.

    Faithful to DistiLLM (arXiv:2402.03898)::

        forward:  mixed = beta * p + (1 - beta) * q,  KL(p || mixed)
        reverse:  mixed = (1 - beta) * p + beta * q,  KL(q || mixed)

    ``beta`` is the teacher's mixing weight (DistiLLM default 0.1). Positions
    with ``labels == -100`` are masked out (DistiLLM masks by label).
    Gradient flows only into the student's logits because the teacher is
    detached by construction (``teacher_logits`` come from ``no_grad``).
    """
    if mode not in ("forward", "reverse"):
        raise ValueError("mode must be 'forward' or 'reverse'")

    p = F.softmax(teacher_logits.float(), dim=-1)
    q = F.softmax(student_logits.float(), dim=-1)

    if mode == "forward":
        mixed = beta * p + (1.0 - beta) * q
        loss = (p * (p.clamp_min(1e-12).log() - mixed.clamp_min(1e-12).log())).sum(-1)
    else:
        mixed = (1.0 - beta) * p + beta * q
        loss = (q * (q.clamp_min(1e-12).log() - mixed.clamp_min(1e-12).log())).sum(-1)

    valid = _valid_mask(labels)
    if valid is not None:
        loss = loss.masked_select(valid.bool())
    return loss.mean() if loss.numel() > 0 else loss.sum() * 0.0


def latent_mse_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """MSE between intermediate latent states (``L_l``)."""
    return F.mse_loss(student, teacher)


def attention_mse_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """MSE between attention maps (``L_a``), averaged over heads/positions."""
    return F.mse_loss(student, teacher)


def embedding_mse_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """MSE between input embedding outputs (``L_e``)."""
    return F.mse_loss(student, teacher)


def spade_composite_loss(
    ce: torch.Tensor,
    logit: torch.Tensor,
    latent: torch.Tensor,
    attention: torch.Tensor,
    embedding: torch.Tensor,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Combine the five loss terms exactly as in the paper (Eq. 2)."""
    return alpha * ce + (1.0 - alpha) / 4.0 * (logit + latent + attention + embedding)

