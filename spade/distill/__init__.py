from spade.distill.losses import (
    attention_mse_loss,
    cross_entropy_loss,
    embedding_mse_loss,
    latent_mse_loss,
    skew_kl_loss,
    spade_composite_loss,
)
from spade.distill.matching import dynamic_teacher_targets, static_teacher_targets
from spade.distill.trainer import DistillationConfig, DistillTrainer, distill_loss

__all__ = [
    "cross_entropy_loss",
    "skew_kl_loss",
    "latent_mse_loss",
    "attention_mse_loss",
    "embedding_mse_loss",
    "spade_composite_loss",
    "dynamic_teacher_targets",
    "static_teacher_targets",
    "DistillTrainer",
    "DistillationConfig",
    "distill_loss",
]

