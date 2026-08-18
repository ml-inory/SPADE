"""Checkpoint save/load for LLM-TTS backbones."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch

from spade.models import LLMTTSBackbone, LLMTTSConfig


def save_checkpoint(
    model: LLMTTSBackbone,
    path: str | Path,
    meta: Optional[dict[str, Any]] = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "config": asdict(model.config),
        "model_state": model.state_dict(),
        "retained_indices": getattr(model, "retained_indices", None),
        "meta": meta or {},
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[LLMTTSBackbone, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = LLMTTSConfig(**ckpt["config"])
    model = LLMTTSBackbone(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    retained = ckpt.get("retained_indices")
    if retained is not None:
        model.retained_indices = list(retained)  # type: ignore[attr-defined]
    return model, ckpt

