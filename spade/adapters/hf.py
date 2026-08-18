"""Adapters for Hugging Face-style residual Transformer backbones.

SPADE's pruning and distillation machinery only assumes a residual
Transformer stack (``x^l = x^{l-1} + f_l(x^{l-1})``) whose blocks live in a
``nn.ModuleList``. This module adapts that machinery to Hugging Face models
such as GPT-2, LLaMA/Qwen-style stacks, and the CosyVoice 2 / LLaSA LLM-TTS
backbones built on them:

* :func:`get_layer_stack` locates the block ``ModuleList``;
* :func:`prune_hf_layers` rebuilds the model with fewer layers by remapping
  state-dict keys and copying the retained blocks' parameters;
* :func:`bypass_block` temporarily zeroes a block (the residual structure
  turns this into removal) for leave-one-out WLI scoring;
* :class:`HFDistillTrainer` runs SPADE's composite distillation loss on HF
  hidden states / attentions with dynamic layer matching.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from copy import deepcopy
from typing import Iterator, Optional

import torch
import torch.nn as nn

from spade.distill import DistillationConfig, DistillTrainer


LAYER_STACK_CANDIDATES: list[tuple[str, str]] = [
    ("transformer", "h"),      # GPT-2 and many encoder-decoder LMs
    ("model", "layers"),       # LLaMA / Qwen / CosyVoice-2-style stacks
    ("", "layers"),
    ("", "h"),
    ("", "blocks"),
    ("decoder", "layers"),     # T5-style decoders
]


def get_layer_stack(model: nn.Module) -> tuple[nn.Module, str, nn.ModuleList]:
    """Locate the ``nn.ModuleList`` of Transformer blocks in an HF model."""
    for parent_attr, stack_attr in LAYER_STACK_CANDIDATES:
        parent = getattr(model, parent_attr, None) if parent_attr else model
        if parent is not None and hasattr(parent, stack_attr):
            stack = getattr(parent, stack_attr)
            if isinstance(stack, nn.ModuleList) and len(stack) > 0:
                return parent, stack_attr, stack
    raise ValueError(
        "could not locate a ModuleList of transformer blocks; add a "
        "(parent_attr, stack_attr) pair to LAYER_STACK_CANDIDATES"
    )


def n_layers(model: nn.Module) -> int:
    return len(get_layer_stack(model)[2])


def _set_n_layers(config, n: int) -> None:
    for attr in ("n_layer", "num_hidden_layers", "num_layers"):
        if hasattr(config, attr):
            setattr(config, attr, n)


def _layer_index_re(stack_name: str) -> re.Pattern:
    return re.compile(re.escape(stack_name) + r"\.(\d+)\.")


def remap_layer_keys(
    state_dict: dict[str, torch.Tensor],
    stack_name: str,
    retained_indices: list[int],
) -> dict[str, torch.Tensor]:
    """Drop pruned blocks and renumber retained block keys to 0..k-1."""
    retained = sorted(int(i) for i in retained_indices)
    index_map = {old: new for new, old in enumerate(retained)}
    rx = _layer_index_re(stack_name)
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        m = rx.search(key)
        if m is None:
            out[key] = value
            continue
        old = int(m.group(1))
        if old not in index_map:
            continue  # pruned layer: drop its parameters
        new_key = key[: m.start(1)] + str(index_map[old]) + key[m.end(1) :]
        out[new_key] = value
    return out


def prune_hf_layers(teacher: nn.Module, retained_indices: list[int]) -> nn.Module:
    """Build a pruned HF student by copying the retained teacher layers.

    The student is the same model class with the layer count reduced; shared
    parameters (embeddings, final norm, LM head) are copied unchanged and
    each retained block is copied to its new position.
    """
    retained = sorted(set(int(i) for i in retained_indices))
    _, stack_name, stack = get_layer_stack(teacher)
    if not retained:
        raise ValueError("retained_indices must be non-empty")
    if min(retained) < 0 or max(retained) >= len(stack):
        raise ValueError(f"retained_indices out of range [0, {len(stack)})")

    config = deepcopy(teacher.config)
    _set_n_layers(config, len(retained))
    student = teacher.__class__(config)
    student.load_state_dict(
        remap_layer_keys(teacher.state_dict(), stack_name, retained)
    )
    student.retained_indices = retained  # type: ignore[attr-defined]
    return student


@contextmanager
def bypass_block(model: nn.Module, index: int) -> Iterator[None]:
    """Temporarily neutralize block ``index`` for leave-one-out WLI.

    All of the block's parameters are zeroed, which -- under the residual
    structure ``x^l = x^{l-1} + f_l(x^{l-1})`` -- makes the block an exact
    identity map regardless of the block's forward signature. Parameters are
    restored afterwards.
    """
    _, _, stack = get_layer_stack(model)
    block = stack[index]
    saved = {k: v.detach().clone() for k, v in block.state_dict().items()}
    with torch.no_grad():
        for param in block.parameters():
            param.zero_()
    try:
        yield
    finally:
        block.load_state_dict(saved)


class HFDistillTrainer(DistillTrainer):
    """SPADE distillation for HF backbones (reuses the composite loss).

    The teacher is frozen; the student is trained with ``alpha * CE +
    (1-alpha)/4 * (skew-KL logit + latent MSE + attention MSE + embedding
    MSE)`` using HF ``output_hidden_states``/``output_attentions``.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: DistillationConfig,
        teacher_targets: Optional[list[int]] = None,
    ) -> None:
        from spade.distill.matching import dynamic_teacher_targets, static_teacher_targets

        self.teacher = teacher.to(config.device).eval()
        self.student = student.to(config.device)
        self.config = config
        retained = list(student.retained_indices)  # type: ignore[attr-defined]
        if teacher_targets is None:
            if config.matching == "dynamic":
                teacher_targets = dynamic_teacher_targets(retained, n_layers(teacher))
            else:
                teacher_targets = static_teacher_targets(retained)
        self.teacher_targets = teacher_targets
        if len(self.teacher_targets) != n_layers(student):
            raise ValueError(
                "teacher_targets length must equal the student's layer count"
            )
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    def _forward_pair(self, batch: dict) -> tuple[dict, dict]:
        input_ids = batch["input_ids"].to(self.config.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.config.device)

        def run(model: nn.Module) -> dict:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
            )
            # HF includes the embedding output at index 0; strip it so the
            # list follows our convention: [post-block 0 .. n-1, final LN].
            hidden_states = list(out.hidden_states)[1:]
            return {
                "logits": out.logits,
                "hidden_states": hidden_states,
                "attentions": list(out.attentions),
                "embedding_out": out.hidden_states[0],
            }

        with torch.no_grad():
            teacher_out = run(self.teacher)
        student_out = run(self.student)
        return teacher_out, student_out
