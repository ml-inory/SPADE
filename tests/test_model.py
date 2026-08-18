import torch
import pytest

from spade.models import LLMTTSBackbone, LLMTTSConfig


def make_model(n_layer=2, n_embd=32, n_head=4, vocab=64):
    cfg = LLMTTSConfig(
        vocab_size=vocab,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        max_position_embeddings=64,
    )
    return LLMTTSBackbone(cfg)


def test_forward_shapes():
    model = make_model()
    ids = torch.randint(0, model.vocab_size, (2, 12))
    out = model(ids, return_hidden_states=True, return_attentions=True)
    assert out["logits"].shape == (2, 12, model.vocab_size)
    assert len(out["hidden_states"]) == model.config.n_layer + 1
    assert len(out["attentions"]) == model.config.n_layer
    att = out["attentions"][0]
    assert att.shape == (2, model.config.n_head, 12, 12)
    assert out["embedding_out"].shape == (2, 12, model.config.n_embd)


def test_causal_attention_is_upper_triangular_zero():
    model = make_model(n_layer=1)
    ids = torch.randint(0, model.vocab_size, (1, 8))
    w = model(ids, return_attentions=True)["attentions"][0][0]  # (H, 8, 8)
    # Causal: upper triangle (col > row) must be exactly zero.
    assert torch.allclose(w, w.tril())
    # Rows (per query position) must sum to 1.
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)))


def test_generation_appends_tokens():
    model = make_model()
    ids = torch.tensor([[3, 4, 5]])
    gen = model.generate(ids, max_new_tokens=6, temperature=0.0)
    assert gen.shape == (1, 9)
    assert (gen[:, :3] == ids).all()


def test_multicodebook_heads():
    cfg = LLMTTSConfig(vocab_size=64, n_layer=1, n_head=2, n_embd=16, num_codebooks=3)
    model = LLMTTSBackbone(cfg)
    ids = torch.randint(0, 64, (1, 8))
    logits = model(ids)["logits"]
    assert isinstance(logits, list) and len(logits) == 3
    assert all(l.shape == (1, 8, 64) for l in logits)


def test_config_validation():
    with pytest.raises(ValueError):
        LLMTTSConfig(n_embd=16, n_head=3)
    with pytest.raises(ValueError):
        LLMTTSConfig(num_codebooks=0)


def test_state_dict_copy_detached():
    model = make_model()
    copy = model.state_dict_copy()
    assert set(copy) == set(model.state_dict())
    with torch.no_grad():
        model.blocks[0].attn.c_attn.weight.add_(1.0)
    assert not torch.allclose(copy["blocks.0.attn.c_attn.weight"], model.state_dict()["blocks.0.attn.c_attn.weight"])
