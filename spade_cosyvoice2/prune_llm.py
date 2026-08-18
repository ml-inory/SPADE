"""Prune the CosyVoice2 LLM (Qwen2, 24 layers) with SPADE's layer selection.

The CosyVoice2 LLM checkpoint (``llm.pt``) is a plain state dict whose keys
include ``llm.llm.model.model.layers.<i>.*`` for the Qwen2 transformer stack.
Pruning reuses :func:`spade.adapters.hf.remap_layer_keys` to drop the
pruned blocks and renumber the retained ones, then saves a checkpoint that
the standard ``CosyVoice2`` loader can consume after shrinking the Qwen2
stack to the target depth.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from spade.adapters.hf import remap_layer_keys, _set_n_layers
from spade.config_utils import dataclass_from_dict, load_yaml
from spade.pruning import select_layers_to_keep


@dataclass
class PruneConfig:
    target_layers: int = 12
    retained: Optional[list[int]] = None
    force_keep_first_last: bool = True


def default_retained_indices(n_layers: int, target: int, force_keep_first_last: bool = True) -> list[int]:
    """Evenly spaced retention when no WLI ranking is available yet."""
    if target >= n_layers:
        return list(range(n_layers))
    if force_keep_first_last:
        if target < 2:
            raise ValueError("target_layers must be >= 2 with force_keep_first_last")
        middle = target - 2
        idx = torch.linspace(1, n_layers - 2, steps=middle).round().long().unique().tolist()
        if len(idx) < middle:  # fill with evenly spaced fallback
            step = (n_layers - 2) / middle
            idx = [min(n_layers - 2, 1 + int(round(i * step))) for i in range(middle)]
        return [0] + idx + [n_layers - 1]
    return list(torch.linspace(0, n_layers - 1, steps=target).round().long().unique().tolist())


def count_qwen_layers(state_dict: dict) -> int:
    keys = [k for k in state_dict if "model.model.layers." in k and ".self_attn." in k]
    idx = {int(k.split("model.model.layers.")[1].split(".")[0]) for k in keys}
    return max(idx) + 1 if idx else 0


def prune_llm_checkpoint(
    llm_pt: str | Path,
    target_layers: int,
    out_path: str | Path,
    retained: Optional[list[int]] = None,
    force_keep_first_last: bool = True,
) -> list[int]:
    state_dict = torch.load(llm_pt, map_location="cpu", weights_only=True)
    n_layers = count_qwen_layers(state_dict)
    if retained is None:
        retained = select_layers_to_keep(
            # Without WLI, fall back to evenly spaced retention.
            [1.0 / (1.0 + abs(i - n_layers / 2)) for i in range(n_layers)],
            target_layers,
            force_keep_first_last=force_keep_first_last,
        )
    retained = sorted(int(i) for i in retained)
    pruned_sd = remap_layer_keys(state_dict, "layers", retained)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pruned_sd, out_path)
    (out_path.with_suffix(".json")).write_text(
        json.dumps({"retained_indices": retained, "n_layers": n_layers}, indent=2)
    )
    print(
        f"[prune] {n_layers} -> {len(retained)} Qwen2 layers; retained={retained}; "
        f"saved {out_path}"
    )
    return retained


def shrink_qwen2_for_llm(llm, retained_indices: list[int]) -> None:
    """Replace the Qwen2 inside a CosyVoice ``Qwen2LM`` with a pruned copy.

    ``llm`` is the ``Qwen2LM`` instance (its ``.llm.model`` is a
    ``Qwen2ForCausalLM``). After this call the wrapper expects a state dict
    whose ``model.model.layers`` keys are renumbered to ``0..k-1``.
    """
    qwen = llm.llm.model
    config = deepcopy(qwen.config)
    _set_n_layers(config, len(retained_indices))
    from transformers import AutoModelForCausalLM

    new_qwen = AutoModelForCausalLM.from_config(config)
    llm.llm.model = new_qwen


def load_pruned_llm_state(llm, pruned_llm_pt: str | Path, device: str = "cuda") -> None:
    sd = torch.load(pruned_llm_pt, map_location=device, weights_only=True)
    llm.load_state_dict(sd, strict=False)
    llm.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune the CosyVoice2 LLM")
    parser.add_argument("--llm-pt", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--target-layers", type=int, default=0)
    parser.add_argument("--retained", type=int, nargs="+", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = PruneConfig()
    if args.config:
        cfg = dataclass_from_dict(PruneConfig, load_yaml(args.config), "PruneConfig")
    if args.target_layers:
        cfg.target_layers = args.target_layers
    if args.retained:
        cfg.retained = args.retained
    prune_llm_checkpoint(
        args.llm_pt,
        cfg.target_layers,
        args.out,
        retained=cfg.retained,
        force_keep_first_last=cfg.force_keep_first_last,
    )


if __name__ == "__main__":
    main()

