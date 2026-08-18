"""Teacher training for the built-in LLM-TTS benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spade.checkpoints import save_checkpoint
from spade.config_utils import dataclass_from_dict, load_yaml, resolve_device
from spade.data import (
    CharTokenizer,
    SyntheticTTSDataset,
    ToySpeechCodec,
    collate_llmtts_batch,
    make_synthetic_texts,
)
from spade.models import LLMTTSBackbone, LLMTTSConfig


@dataclass
class TrainConfig:
    data: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 16
    epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 0
    log_every: int = 20


def build_benchmark(data_cfg: dict[str, Any]):
    """Build tokenizer/codec/datasets from a data config."""
    torch.manual_seed(int(data_cfg.get("seed", 0)))
    tokenizer = CharTokenizer()
    codec = ToySpeechCodec(
        alphabet=tokenizer.alphabet,
        code_vocab_size=int(data_cfg.get("code_vocab_size", 512)),
    )
    texts = make_synthetic_texts(
        int(data_cfg.get("num_samples", 400)),
        seed=int(data_cfg.get("seed", 0)),
        max_words=int(data_cfg.get("max_words", 12)),
    )
    eval_texts = make_synthetic_texts(
        int(data_cfg.get("num_eval_samples", 24)),
        seed=int(data_cfg.get("seed", 0)) + 1,
        max_words=int(data_cfg.get("max_words", 12)),
    )
    train_ds = SyntheticTTSDataset(
        texts,
        codec,
        tokenizer,
        num_speakers=int(data_cfg.get("num_speakers", 4)),
        use_prompt=bool(data_cfg.get("use_prompt", False)),
        max_len=int(data_cfg.get("max_len", 128)),
    )
    eval_ds = SyntheticTTSDataset(
        eval_texts,
        codec,
        tokenizer,
        num_speakers=int(data_cfg.get("num_speakers", 4)),
        use_prompt=bool(data_cfg.get("use_prompt", False)),
        max_len=int(data_cfg.get("max_len", 128)),
    )
    return tokenizer, codec, train_ds, eval_ds


def train_teacher(cfg: TrainConfig) -> tuple[LLMTTSBackbone, dict]:
    device = resolve_device(cfg.device)
    tokenizer, codec, train_ds, eval_ds = build_benchmark(cfg.data)
    model_cfg = dict(cfg.model)
    model_cfg.setdefault("vocab_size", codec.code_vocab_size)
    model = LLMTTSBackbone(LLMTTSConfig(**model_cfg)).to(device)
    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_llmtts_batch,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    history: list[dict] = []
    step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask=mask)["logits"]
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.detach())
            n += 1
            step += 1
            if cfg.log_every and step % cfg.log_every == 0:
                print(f"[train] epoch {epoch} step {step} ce={float(loss.detach()):.4f}")
        history.append({"epoch": epoch, "ce": epoch_loss / max(n, 1), "step": step})
        print(f"[train] epoch {epoch} done: avg ce={epoch_loss / max(n, 1):.4f}")
    return model, {"history": history, "codec_vocab_size": codec.code_vocab_size}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an LLM-TTS teacher")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = dataclass_from_dict(TrainConfig, load_yaml(args.config), "TrainConfig")
    model, meta = train_teacher(cfg)
    save_checkpoint(model, args.out, meta=meta)
    print(f"saved teacher checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
