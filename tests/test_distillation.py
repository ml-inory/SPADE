import pytest
import torch

from spade.data import CharTokenizer, SyntheticTTSDataset, ToySpeechCodec, make_synthetic_texts
from spade.distill import (
    DistillationConfig,
    DistillTrainer,
    attention_mse_loss,
    cross_entropy_loss,
    distill_loss,
    dynamic_teacher_targets,
    embedding_mse_loss,
    latent_mse_loss,
    skew_kl_loss,
    spade_composite_loss,
    static_teacher_targets,
)
from spade.models import LLMTTSBackbone, LLMTTSConfig
from spade.pruning import prune_layers
from torch.utils.data import DataLoader


def make_pair(n_teacher=4, vocab=128, n_embd=32, n_head=4):
    teacher = LLMTTSBackbone(
        LLMTTSConfig(vocab_size=vocab, n_layer=n_teacher, n_head=n_head, n_embd=n_embd)
    )
    retained = [0, n_teacher - 1]
    student = prune_layers(teacher, retained)
    return teacher, student


def make_dataloader(batch_size=4, n=12):
    tok = CharTokenizer()
    codec = ToySpeechCodec(alphabet=tok.alphabet)
    texts = make_synthetic_texts(n, seed=7, max_words=5)
    ds = SyntheticTTSDataset(texts, codec, tok, num_speakers=2)
    return DataLoader(ds, batch_size=batch_size, collate_fn=lambda b: _collate(b))


def _collate(batch):
    from spade.data import collate_llmtts_batch

    return collate_llmtts_batch(batch)


def _tiny_forward(model, B=2, T=10, vocab=128):
    ids = torch.randint(0, vocab, (B, T))
    labels = torch.randint(0, vocab, (B, T))
    labels[:, -2:] = -100
    return model(ids, return_hidden_states=True, return_attentions=True), labels


def test_skew_kl_zero_for_identical():
    logits = torch.randn(2, 6, 32)
    loss = skew_kl_loss(logits, logits.clone(), beta=0.1)
    assert loss.item() < 1e-4
    # Reverse mode also zero for identical distributions.
    loss_r = skew_kl_loss(logits, logits.clone(), beta=0.1, mode="reverse")
    assert loss_r.item() < 1e-4


def test_skew_kl_masking_and_modes():
    student = torch.randn(2, 6, 32, requires_grad=True)
    teacher = torch.randn(2, 6, 32)
    labels = torch.full((2, 6), -100)
    labels[0, :4] = 1
    loss = skew_kl_loss(student, teacher, labels=labels)
    assert loss.requires_grad
    assert torch.isfinite(loss)
    assert skew_kl_loss(student, teacher, labels=None).item() > 0
    with pytest.raises(ValueError):
        skew_kl_loss(student, teacher, mode="sideways")


def test_mse_losses():
    a = torch.randn(2, 8, 16)
    b = torch.randn(2, 8, 16)
    assert torch.allclose(latent_mse_loss(a, a), torch.tensor(0.0), atol=1e-6)
    assert attention_mse_loss(a, a).item() == 0.0
    assert embedding_mse_loss(a, a).item() == 0.0
    assert latent_mse_loss(a, b) > 0


def test_composite_loss_weights():
    ce = torch.tensor(1.0, requires_grad=True)
    logit = torch.tensor(2.0, requires_grad=True)
    latent = torch.tensor(3.0, requires_grad=True)
    attn = torch.tensor(4.0, requires_grad=True)
    emb = torch.tensor(5.0, requires_grad=True)
    total = spade_composite_loss(ce, logit, latent, attn, emb, alpha=0.25)
    expected = 0.25 * 1.0 + 0.75 / 4.0 * (2 + 3 + 4 + 5)
    assert torch.allclose(total, torch.tensor(expected))


def test_dynamic_matching():
    assert dynamic_teacher_targets([0, 2, 5], n_teacher_layers=8) == [1, 4, 7]
    assert dynamic_teacher_targets([0, 1, 2], n_teacher_layers=3) == [0, 1, 2]
    assert dynamic_teacher_targets([0], n_teacher_layers=6) == [5]
    assert static_teacher_targets([2, 0, 5]) == [0, 2, 5]
    with pytest.raises(ValueError):
        dynamic_teacher_targets([], n_teacher_layers=4)
    with pytest.raises(ValueError):
        dynamic_teacher_targets([0, 7], n_teacher_layers=6)


def test_distill_loss_components_shapes_and_total():
    teacher, student = make_pair(n_teacher=4, vocab=128)
    t_out, labels = _tiny_forward(teacher)
    s_out, _ = _tiny_forward(student)
    targets = dynamic_teacher_targets(student.retained_indices, teacher.config.n_layer)
    comp = distill_loss(t_out, s_out, labels, teacher_targets=targets)
    assert set(comp) == {"ce", "logit", "latent", "attention", "embedding", "total"}
    assert all(torch.isfinite(v) for v in comp.values())
    assert comp["total"].item() > 0
    comp["total"].backward()
    assert student.blocks[0].attn.c_attn.weight.grad is not None


def test_trainer_updates_student_not_teacher():
    teacher, student = make_pair(n_teacher=4, vocab=512, n_embd=64, n_head=4)
    loader = make_dataloader(batch_size=4, n=12)
    config = DistillationConfig(epochs=1, device="cpu", log_every=1000)
    trainer = DistillTrainer(teacher, student, config)
    teacher_params_before = {k: v.clone() for k, v in teacher.state_dict().items()}
    student_params_before = {k: v.clone() for k, v in student.state_dict().items()}
    history = trainer.train(loader)
    assert len(history["train"]) == 1
    assert all(torch.isfinite(torch.tensor(v)) for v in history["train"][0].values())
    # Teacher must be untouched.
    for k, v in teacher.state_dict().items():
        assert torch.equal(v, teacher_params_before[k]), k
    # Student must have moved.
    moved = any(
        not torch.equal(student.state_dict()[k], student_params_before[k])
        for k in student_params_before
    )
    assert moved


def test_trainer_static_matching_and_ablations():
    teacher, student = make_pair(n_teacher=4, vocab=512, n_embd=64, n_head=4)
    loader = make_dataloader(batch_size=4, n=8)
    config = DistillationConfig(
        epochs=1, matching="static", use_logit=False, use_attention=False, device="cpu"
    )
    trainer = DistillTrainer(teacher, student, config)
    assert trainer.teacher_targets == static_teacher_targets(student.retained_indices)
    history = trainer.train(loader)
    assert history["train"][0]["attention"] == 0.0
    assert history["train"][0]["logit"] == 0.0


def test_trainer_validates_targets():
    teacher, student = make_pair(n_teacher=4, vocab=128)
    with pytest.raises(ValueError):
        DistillTrainer(teacher, student, DistillationConfig(), teacher_targets=[0])

