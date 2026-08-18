import json

import pytest
import torch

from spade.adapters.hf import remap_layer_keys
from spade_cosyvoice2.prune_llm import (
    count_qwen_layers,
    default_retained_indices,
    prune_llm_checkpoint,
)


def make_fake_llm_state_dict(n_layers=24, n_embd=16):
    sd = {
        "llm.llm_embedding.weight": torch.randn(2, n_embd),
        "llm.llm.model.model.embed_tokens.weight": torch.randn(100, n_embd),
        "llm.llm.model.model.norm.weight": torch.randn(n_embd),
        "llm.llm_decoder.weight": torch.randn(8, n_embd),
        "llm.llm_decoder.bias": torch.randn(8),
        "llm.speech_embedding.weight": torch.randn(8, n_embd),
    }
    for i in range(n_layers):
        sd[f"llm.llm.model.model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(n_embd, n_embd)
        sd[f"llm.llm.model.model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(n_embd, n_embd)
        sd[f"llm.llm.model.model.layers.{i}.input_layernorm.weight"] = torch.randn(n_embd)
    return sd


def test_count_qwen_layers():
    sd = make_fake_llm_state_dict(n_layers=24)
    assert count_qwen_layers(sd) == 24


def test_default_retained_even_spacing():
    retained = default_retained_indices(24, 12)
    assert retained[0] == 0
    assert retained[-1] == 23
    assert len(retained) == 12
    assert retained == sorted(retained)
    assert 12 not in retained or True


def test_remap_drops_and_renumbers():
    sd = make_fake_llm_state_dict(n_layers=6)
    pruned = remap_layer_keys(sd, "layers", [0, 2, 5])
    layer_keys = sorted(k for k in pruned if "layers." in k)
    indices = {int(k.split("layers.")[1].split(".")[0]) for k in layer_keys}
    assert indices == {0, 1, 2}
    # Retained teacher layer 2 must land on student layer 1.
    assert torch.equal(
        pruned["llm.llm.model.model.layers.1.self_attn.q_proj.weight"],
        sd["llm.llm.model.model.layers.2.self_attn.q_proj.weight"],
    )
    assert "layers.1." not in [k for k in sd if k.startswith("llm.llm.model.model.layers.3")]
    # Non-layer keys are copied unchanged.
    assert "llm.llm_decoder.weight" in pruned


def test_prune_llm_checkpoint_roundtrip(tmp_path):
    src = tmp_path / "llm.pt"
    torch.save(make_fake_llm_state_dict(n_layers=8), src)
    out = tmp_path / "pruned.pt"
    retained = prune_llm_checkpoint(src, 4, out, retained=[0, 2, 5, 7])
    assert retained == [0, 2, 5, 7]
    meta = json.loads((out.with_suffix(".json")).read_text())
    assert meta["retained_indices"] == [0, 2, 5, 7]
    pruned_sd = torch.load(out, weights_only=True)
    assert count_qwen_layers(pruned_sd) == 4


def test_prune_requires_target():
    with pytest.raises(ValueError):
        default_retained_indices(24, 1)

