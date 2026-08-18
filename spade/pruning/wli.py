"""WER-based layer importance (WLI) pruning -- the pruning stage of SPADE.

Following the paper (Eq. 1), the importance of Transformer layer ``i`` is
the average WER of the model when that layer is removed::

    WLI_i = E_D[ WER( model(x_2; theta_{i}, x_1, y_1), y_2 ) ]

where ``theta_{i}`` denotes the model parameters *without* layer ``i``.
Layers whose removal does not hurt intelligibility (low WLI) are pruned.
The cosine-based layer importance (CLI) baseline used for ablation is also
provided: the average cosine distance between a layer's input and output
latents.
"""

from __future__ import annotations

from dataclasses import replace
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spade.models import LLMTTSBackbone, TransformerBlock


@dataclass
class ImportanceReport:
    """Per-layer importance and the resulting pruning decision."""

    wli: np.ndarray
    cli: Optional[np.ndarray]
    retained_indices: list[int]
    target_layers: int

    @property
    def n_layers(self) -> int:
        return len(self.wli)

    def summary(self) -> str:
        lines = [
            f"target depth: {self.target_layers} (from {self.n_layers})",
            "per-layer WLI: " + ", ".join(f"{v:.3f}" for v in self.wli),
        ]
        if self.cli is not None:
            lines.append(
                "per-layer CLI: " + ", ".join(f"{v:.3f}" for v in self.cli)
            )
        lines.append("retained indices: " + ", ".join(map(str, self.retained_indices)))
        return "\n".join(lines)


def compute_wli(
    model: LLMTTSBackbone,
    scorer: Callable[[LLMTTSBackbone, dict], float],
    samples: Sequence[dict],
    bypass_fn: Optional[Callable] = None,
    n_layers: Optional[int] = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Leave-one-out WER-based layer importance.

    For each layer, temporarily bypass it, score every evaluation sample with
    ``scorer`` (WER per sample), and average. Layers are restored afterwards.

    ``bypass_fn`` is an optional ``contextmanager(model, layer_index)`` that
    temporarily neutralizes a layer (e.g. :func:`spade.adapters.hf.bypass_block`
    for Hugging Face backbones); when omitted, the built-in backbone's
    :meth:`LLMTTSBackbone.disable_blocks` is used. ``n_layers`` overrides the
    layer count for backbones whose config uses a different attribute name.
    """
    model.eval()
    n = n_layers if n_layers is not None else model.config.n_layer
    if hasattr(scorer, "score"):
        scorer = scorer.score  # type: ignore[assignment]
    wli = np.zeros(n, dtype=np.float64)
    iterator = range(n)
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="WLI (leave-one-out)")

    for i in iterator:
        wers: list[float] = []
        if bypass_fn is not None:
            with bypass_fn(model, i):
                with torch.no_grad():
                    for sample in samples:
                        wers.append(float(scorer(model, sample)))
        else:
            model.disable_blocks([i])
            try:
                with torch.no_grad():
                    for sample in samples:
                        wers.append(float(scorer(model, sample)))
            finally:
                model.enable_all_blocks()
        wli[i] = float(np.mean(wers))
    return wli


def compute_cli(
    model: LLMTTSBackbone,
    samples: Sequence[dict],
    batch_size: int = 8,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Cosine-based layer importance: mean 1 - cos(x^{l-1}, x^l) per layer.

    A higher value means the layer transforms its input more. This is the
    baseline the paper shows does *not* align with WER-based importance.
    """
    model.eval()
    n_layers = model.config.n_layer
    sums = np.zeros(n_layers, dtype=np.float64)
    counts = np.zeros(n_layers, dtype=np.float64)
    model = model.to(device)

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        input_ids = nn.utils.rnn.pad_sequence(
            [s["input_ids"] for s in batch], batch_first=True, padding_value=2
        ).to(device)
        mask = (input_ids != 2).long()
        with torch.no_grad():
            out = model(
                input_ids,
                attention_mask=mask,
                return_hidden_states=True,
            )
        emb = out["embedding_out"]                     # (B, T, D)
        hidden = out["hidden_states"]                  # [B,T,D] * (n_layers+1)
        valid = mask.bool()                            # (B, T)
        for i in range(n_layers):
            x_in = emb if i == 0 else hidden[i - 1]
            x_out = hidden[i]
            cos = F.cosine_similarity(x_in, x_out, dim=-1)  # (B, T)
            sums[i] += float(cos[valid].sum().item())
            counts[i] += int(valid.sum().item())

    return 1.0 - sums / np.maximum(counts, 1)


def select_layers_to_keep(
    importance: Sequence[float] | np.ndarray,
    target_layers: int,
    force_keep_first_last: bool = True,
) -> list[int]:
    """Select layers to retain under an importance ranking.

    ``importance`` is *higher-is-more-important* (WLI or CLI). The lowest-
    importance layers are pruned; optionally the first and last layers are
    always retained (boundary layers consistently score important in the
    paper and are structurally safe to keep).
    """
    n = len(importance)
    if target_layers > n:
        raise ValueError(f"target_layers ({target_layers}) exceeds n_layer ({n})")
    if target_layers < 1:
        raise ValueError("target_layers must be >= 1")
    if target_layers == n:
        return list(range(n))

    keep: set[int] = set()
    if force_keep_first_last:
        keep.update({0, n - 1})
        if target_layers < len(keep):
            raise ValueError(
                f"target_layers ({target_layers}) is smaller than the forced "
                f"first/last layers ({len(keep)}); disable force_keep_first_last"
            )

    candidates = [i for i in range(n) if i not in keep]
    ranked = sorted(candidates, key=lambda i: float(importance[i]), reverse=True)
    keep.update(ranked[: target_layers - len(keep)])
    return sorted(keep)


def prune_layers(
    teacher: LLMTTSBackbone,
    retained_indices: Sequence[int],
) -> LLMTTSBackbone:
    """Build a pruned student by copying the retained teacher layers.

    Embeddings, final layer norm, and LM heads are copied directly; each
    student block ``j`` receives the parameters of teacher block
    ``retained_indices[j]``. This is the "parameters are copied from
    retained layers" initialization described in the paper.
    """
    retained = sorted(set(int(i) for i in retained_indices))
    n_teacher = teacher.config.n_layer
    if not retained:
        raise ValueError("retained_indices must be non-empty")
    if min(retained) < 0 or max(retained) >= n_teacher:
        raise ValueError(f"retained_indices out of range [0, {n_teacher})")

    student = LLMTTSBackbone(
        replace(teacher.config, n_layer=len(retained))
    )
    student.wte.load_state_dict(teacher.wte.state_dict())
    student.wpe.load_state_dict(teacher.wpe.state_dict())
    student.ln_f.load_state_dict(teacher.ln_f.state_dict())
    for s_head, t_head in zip(student.lm_heads, teacher.lm_heads):
        s_head.load_state_dict(t_head.state_dict())

    blocks: list[TransformerBlock] = []
    for j, i in enumerate(retained):
        block = TransformerBlock(student.config)
        block.load_state_dict(teacher.blocks[i].state_dict())
        blocks.append(block)
    student.set_blocks(nn.ModuleList(blocks))
    student.retained_indices = retained  # type: ignore[attr-defined]
    return student


def prune_by_importance(
    teacher: LLMTTSBackbone,
    importance: Sequence[float] | np.ndarray,
    target_layers: int,
    force_keep_first_last: bool = True,
) -> tuple[LLMTTSBackbone, list[int]]:
    """One-shot pruning: rank by importance, keep top-k, copy weights."""
    retained = select_layers_to_keep(
        importance, target_layers, force_keep_first_last=force_keep_first_last
    )
    return prune_layers(teacher, retained), retained
