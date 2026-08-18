import torch

import pytest
from torch.utils.data import DataLoader, Dataset

from spade.adapters import HFDistillTrainer, bypass_block, get_layer_stack, n_layers, prune_hf_layers
from spade.data import CharTokenizer, make_synthetic_texts
from spade.distill import DistillationConfig, dynamic_teacher_targets
from spade.pruning import compute_wli

transformers = pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel


def make_hf_teacher(n_layer=4, vocab=512, n_embd=64, n_head=4):
    cfg = GPT2Config(
        vocab_size=vocab,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_positions=128,
        n_ctx=128,
    )
    return GPT2LMHeadModel(cfg)


class _CharDataset(Dataset):
    def __init__(self, n=16, max_len=20):
        self.tok = CharTokenizer()
        self.texts = make_synthetic_texts(n, seed=11, max_words=4)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        ids = self.tok.encode(self.texts[i], add_bos=True, add_eos=True)[:20]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(ids[1:] + [self.tok.pad_id], dtype=torch.long),
        }


def _collate(batch):
    pad = 2
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=pad
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch], batch_first=True, padding_value=-100
    )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": (input_ids != pad).long(),
    }


class _FakeCharScorer:
    def __init__(self, per_layer_wer):
        self.per_layer_wer = per_layer_wer
        self.calls = 0

    def score(self, model, sample):
        i = self.calls % len(self.per_layer_wer)
        self.calls += 1
        return self.per_layer_wer[i]


def test_layer_stack_detection():
    model = make_hf_teacher()
    parent, stack_name, stack = get_layer_stack(model)
    assert stack_name == "h"
    assert len(stack) == 4
    assert n_layers(model) == 4


def test_prune_hf_layers_copies_weights():
    teacher = make_hf_teacher(n_layer=4)
    with torch.no_grad():
        for i, block in enumerate(teacher.transformer.h):
            block.attn.c_attn.weight.fill_(float(i + 1))
    student = prune_hf_layers(teacher, [0, 3])
    assert n_layers(student) == 2
    assert student.retained_indices == [0, 3]
    assert torch.allclose(
        student.transformer.h[0].attn.c_attn.weight,
        teacher.transformer.h[0].attn.c_attn.weight,
    )
    assert torch.allclose(
        student.transformer.h[1].attn.c_attn.weight,
        teacher.transformer.h[3].attn.c_attn.weight,
    )
    assert sum(p.numel() for p in student.parameters()) < sum(
        p.numel() for p in teacher.parameters()
    )
    # Forward pass works on the pruned model.
    ids = torch.randint(0, 512, (2, 10))
    assert student(input_ids=ids).logits.shape == (2, 10, 512)


def test_bypass_block_zeroes_and_restores():
    model = make_hf_teacher(n_layer=2)
    before = model.transformer.h[1].attn.c_attn.weight.clone()
    with bypass_block(model, 1):
        assert torch.allclose(
            model.transformer.h[1].attn.c_attn.weight, torch.zeros_like(before)
        )
    assert torch.allclose(model.transformer.h[1].attn.c_attn.weight, before)


def test_wli_with_bypass_fn():
    teacher = make_hf_teacher(n_layer=3)
    # Random init: verify wiring, not quality.
    samples = [{"input_ids": torch.randint(0, 512, (12,)), "labels": None}]
    fake = _FakeCharScorer([0.1, 0.5, 0.2])
    wli = compute_wli(teacher, fake, samples, bypass_fn=bypass_block, n_layers=3)
    assert wli.shape == (3,)
    assert torch.allclose(
        torch.tensor(wli), torch.tensor([0.1, 0.5, 0.2], dtype=torch.float64)
    )


def test_hf_distill_trainer_updates_student_only():
    teacher = make_hf_teacher(n_layer=4, vocab=64, n_embd=48)
    student = prune_hf_layers(teacher, [0, 3])
    loader = DataLoader(
        _CharDataset(n=12), batch_size=4, collate_fn=_collate
    )
    config = DistillationConfig(epochs=1, device="cpu", log_every=1000)
    trainer = HFDistillTrainer(teacher, student, config)
    assert trainer.teacher_targets == dynamic_teacher_targets([0, 3], 4)
    teacher_before = {k: v.clone() for k, v in teacher.state_dict().items()}
    student_before = {k: v.clone() for k, v in student.state_dict().items()}
    history = trainer.train(loader)
    assert len(history["train"]) == 1
    assert all(torch.isfinite(torch.tensor(v)) for v in history["train"][0].values())
    for k, v in teacher.state_dict().items():
        assert torch.equal(v, teacher_before[k]), k
    assert any(
        not torch.equal(student.state_dict()[k], student_before[k])
        for k in student_before
    )


def test_hf_trainer_validates_targets():
    teacher = make_hf_teacher(n_layer=4, vocab=64, n_embd=48)
    student = prune_hf_layers(teacher, [0, 3])
    with pytest.raises(ValueError):
        HFDistillTrainer(teacher, student, DistillationConfig(), teacher_targets=[0])
