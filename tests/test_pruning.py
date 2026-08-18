import numpy as np
import pytest
import torch

from spade.data import CharTokenizer, ToySpeechCodec, make_synthetic_texts
from spade.evaluation import TokenCodecWERScorer
from spade.models import LLMTTSBackbone, LLMTTSConfig
from spade.pruning import compute_cli, compute_wli, prune_by_importance, prune_layers, select_layers_to_keep


def make_teacher(n_layer=4, vocab=64, n_embd=32):
    cfg = LLMTTSConfig(vocab_size=vocab, n_layer=n_layer, n_head=4, n_embd=n_embd)
    return LLMTTSBackbone(cfg)


def make_samples(n=6):
    tok = CharTokenizer()
    codec = ToySpeechCodec(alphabet=tok.alphabet)
    texts = make_synthetic_texts(n, seed=3, max_words=5)
    from spade.data import SyntheticTTSDataset

    ds = SyntheticTTSDataset(texts, codec, tok, num_speakers=2, use_prompt=False)
    samples = []
    for i in range(n):
        item = ds[i]
        samples.append(
            {
                "input_ids": item["input_ids"],
                "speaker_ids": item["speaker_ids"],
                "text": item["text"],
                "cond_len": int(item["cond_len"]),
            }
        )
    return samples


class _FakeWERScorer:
    """Returns a known WER per disabled layer so WLI wiring is testable."""

    def __init__(self, per_layer_wer):
        self.per_layer_wer = per_layer_wer

    def score(self, model, sample):
        assert len(model._disabled) == 1
        layer = next(iter(model._disabled))
        return self.per_layer_wer[layer]


def test_compute_wli_leave_one_out_and_restores():
    model = make_teacher()
    samples = make_samples(4)
    fake = _FakeWERScorer({0: 0.1, 1: 0.9, 2: 0.2, 3: 0.3})
    wli = compute_wli(model, fake, samples)
    assert wli.shape == (4,)
    assert np.allclose(wli, [0.1, 0.9, 0.2, 0.3])
    assert model._disabled == set(), "layers must be restored after WLI"


def test_select_layers_keeps_top_and_boundaries():
    importance = np.array([0.9, 0.1, 0.2, 0.8])
    assert select_layers_to_keep(importance, 2) == [0, 3]
    assert select_layers_to_keep(importance, 3) == [0, 2, 3]
    assert select_layers_to_keep(importance, 4) == [0, 1, 2, 3]
    # Without boundary forcing, pure top-k by importance.
    assert select_layers_to_keep(importance, 2, force_keep_first_last=False) == [0, 3]
    with pytest.raises(ValueError):
        select_layers_to_keep(importance, 5)
    with pytest.raises(ValueError):
        select_layers_to_keep(np.array([0.5, 0.5]), 1)


def test_prune_layers_copies_weights():
    teacher = make_teacher(n_layer=4)
    # Give blocks distinct weights for a strong assertion.
    with torch.no_grad():
        for i, block in enumerate(teacher.blocks):
            block.attn.c_attn.weight.fill_(float(i + 1))
        teacher.wte.weight.fill_(7.0)
    student = prune_layers(teacher, [1, 3])
    assert student.config.n_layer == 2
    assert student.retained_indices == [1, 3]
    assert student.wte.weight.shape == teacher.wte.weight.shape
    assert torch.allclose(student.wte.weight, teacher.wte.weight)
    assert torch.allclose(student.blocks[0].attn.c_attn.weight, teacher.blocks[1].attn.c_attn.weight)
    assert torch.allclose(student.blocks[1].attn.c_attn.weight, teacher.blocks[3].attn.c_attn.weight)
    assert student.num_parameters() < teacher.num_parameters()


def test_prune_layers_validation():
    teacher = make_teacher(n_layer=4)
    with pytest.raises(ValueError):
        prune_layers(teacher, [])
    with pytest.raises(ValueError):
        prune_layers(teacher, [4])
    with pytest.raises(ValueError):
        prune_layers(teacher, [-1])


def test_prune_by_importance_wiring():
    teacher = make_teacher(n_layer=4)
    student, retained = prune_by_importance(teacher, [0.9, 0.1, 0.2, 0.8], target_layers=2)
    assert retained == [0, 3]
    assert student.config.n_layer == 2


def test_compute_cli_finite_and_in_range():
    model = make_teacher(n_layer=3, vocab=512)
    samples = make_samples(4)
    cli = compute_cli(model, samples, batch_size=2)
    assert cli.shape == (3,)
    assert np.isfinite(cli).all()
    assert ((cli >= 0.0) & (cli <= 2.0)).all()


def test_token_codec_wer_scorer_returns_bounded_score():
    tok = CharTokenizer()
    codec = ToySpeechCodec(alphabet=tok.alphabet)
    from spade.data import SyntheticTTSDataset

    texts = make_synthetic_texts(2, seed=5, max_words=4)
    ds = SyntheticTTSDataset(texts, codec, tok, num_speakers=1, use_prompt=False)
    sample = ds[0]
    sample["cond_len"] = int(sample["cond_len"])
    model = make_teacher(vocab=codec.code_vocab_size, n_embd=64, n_layer=2)
    scorer = TokenCodecWERScorer(codec=codec, tokenizer=tok, max_new_tokens=24)
    wer = scorer.score(model, {k: sample[k] for k in ("input_ids", "speaker_ids", "text", "cond_len")})
    assert 0.0 <= wer <= 1.0
