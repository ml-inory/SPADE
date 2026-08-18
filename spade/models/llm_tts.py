"""Minimal GPT-style autoregressive backbone for LLM-TTS.

This is a compact, self-contained implementation of the LLM-TTS backbone
class used by recent systems (e.g. CosyVoice / VALL-E style codec language
models): a causal Transformer that consumes text tokens (and, optionally,
prompt audio-code tokens) and autoregressively predicts discrete speech-code
tokens.

The forward pass optionally exposes, per Transformer block:

* ``hidden_states`` -- the post-block latent used by SPADE's latent
  alignment loss ``L_l``;
* ``attentions``    -- the causal attention maps used by SPADE's attention
  alignment loss ``L_a``;
* ``embedding_out`` -- the input embedding output used by SPADE's embedding
  reconstruction loss ``L_e``.

The design mirrors the residual structure ``x^l = x^{l-1} + f_l(x^{l-1})``
assumed by the paper, which is what makes structured layer pruning safe:
removing a block is equivalent to ``f_l -> 0``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LLMTTSConfig:
    """Configuration for :class:`LLMTTSBackbone`."""

    vocab_size: int = 256
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 128
    max_position_embeddings: int = 512
    num_codebooks: int = 1
    ff_mult: int = 4
    dropout: float = 0.0
    attn_dropout: float = 0.0
    norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.num_codebooks < 1:
            raise ValueError("num_codebooks must be >= 1")


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention that can expose attention maps."""

    def __init__(self, config: LLMTTSConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        if attention_mask is not None:
            att = att + attention_mask
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, H, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, att if return_attn else None


class TransformerBlock(nn.Module):
    """Residual Transformer block: ``x + Attn(LN(x))`` then ``x + MLP(LN(x))``."""

    def __init__(self, config: LLMTTSConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.ff_mult * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(config.ff_mult * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        h, attn = self.attn(self.ln1(x), attention_mask, return_attn=return_attn)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, attn


class LLMTTSBackbone(nn.Module):
    """Autoregressive LLM-TTS backbone.

    Inputs are token ids (text tokens and/or speech-code tokens) from a
    shared vocabulary. The model predicts, at every position, the next
    speech-code token for each configured codebook (``num_codebooks`` LM
    heads, VALL-E/CosyVoice style).
    """

    def __init__(self, config: LLMTTSConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_codebooks = config.num_codebooks

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.norm_eps)
        self.lm_heads = nn.ModuleList(
            [
                nn.Linear(config.n_embd, config.vocab_size, bias=False)
                for _ in range(config.num_codebooks)
            ]
        )
        self._disabled: set[int] = set()
        if config.tie_word_embeddings:
            for head in self.lm_heads:
                head.weight = self.wte.weight

        self.apply(self._init_weights)
        # Scale projection residuals (GPT-2 style) for stable training.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=config.initializer_range / (2 * config.n_layer) ** 0.5)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_block(self, index: int) -> TransformerBlock:
        return self.blocks[index]

    def set_blocks(self, blocks: nn.ModuleList) -> None:
        """Replace the Transformer block stack (used by pruning)."""
        self.blocks = blocks

    def disable_blocks(self, indices: list[int] | set[int] | tuple[int, ...]) -> None:
        """Temporarily bypass the given blocks (used for leave-one-out WLI).

        Bypassed blocks are skipped entirely, which -- thanks to the residual
        structure ``x^l = x^{l-1} + f_l(x^{l-1})`` -- is equivalent to
        removing them.
        """
        self._disabled = set(indices)

    def enable_all_blocks(self) -> None:
        self._disabled = set()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        return_attentions: bool = False,
    ) -> dict:
        """Forward pass.

        Args:
            input_ids: LongTensor ``(B, T)`` of token ids.
            position_ids: optional ``(B, T)`` absolute positions.
            attention_mask: optional ``(B, T)`` causal mask (1 = attend,
                0 = blocked); converted to additive mask internally.
            return_hidden_states: also return per-block hidden states.
            return_attentions: also return per-block attention maps.

        Returns:
            Dict with keys:
                ``logits``         -- ``(B, T, vocab)`` when a single
                    codebook, otherwise list of ``(B, T, vocab)`` tensors;
                ``hidden_states``  -- list of per-block hidden states
                    (post residual, including the final layer norm output);
                ``attentions``     -- list of per-block attention maps
                    ``(B, H, T, T)``;
                ``embedding_out``  -- the embedding output ``(B, T, n_embd)``.
        """
        B, T = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(T, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(B, -1)

        # Always enforce causality; optionally combine with a padding mask.
        causal = torch.tril(
            torch.ones(T, T, dtype=torch.bool, device=input_ids.device)
        ).unsqueeze(0)
        if attention_mask is not None:
            # Convert (B, T) 0/1 mask to additive mask.
            pad = attention_mask.unsqueeze(1).unsqueeze(2).bool()  # (B, 1, 1, T)
            valid = causal & pad
        else:
            valid = causal
        dtype = next(self.parameters()).dtype
        attn_mask = torch.where(
            valid,
            torch.zeros((), dtype=dtype, device=input_ids.device),
            torch.full((), torch.finfo(dtype).min, device=input_ids.device),
        )

        embedding_out = self.drop(self.wte(input_ids) + self.wpe(position_ids))
        x = embedding_out

        hidden_states: list[torch.Tensor] = []
        attentions: list[torch.Tensor] = []
        for idx, block in enumerate(self.blocks):
            if idx in self._disabled:
                continue
            x, attn = block(x, attn_mask, return_attn=return_attentions)
            if return_hidden_states:
                hidden_states.append(x)
            if return_attentions:
                attentions.append(attn)

        x = self.ln_f(x)
        if return_hidden_states:
            hidden_states.append(x)  # final normed latent, used for logits

        logits = [head(x) for head in self.lm_heads]
        if self.num_codebooks == 1:
            logits = logits[0]

        return {
            "logits": logits,
            "hidden_states": hidden_states if return_hidden_states else None,
            "attentions": attentions if return_attentions else None,
            "embedding_out": embedding_out,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Greedy/sampled autoregressive generation over codebook 0.

        Returns the generated token ids appended to ``input_ids``.
        """
        self.eval()
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            out = self.forward(generated)
            logits = out["logits"]
            if isinstance(logits, list):
                logits = logits[0]
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                k = min(top_k, next_logits.size(-1))
                topk = torch.topk(next_logits, k, dim=-1)
                next_logits = next_logits.masked_fill(
                    next_logits < topk.values[:, -1:], float("-inf")
                )
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = next_logits.sort(dim=-1, descending=True)
                cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum - sorted_logits.softmax(dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                next_logits = torch.empty_like(sorted_logits).scatter_(
                    -1, sorted_idx, sorted_logits
                )

            probs = F.softmax(next_logits, dim=-1)
            if temperature == 0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=-1)
            if eos_token_id is not None and (next_id == eos_token_id).all():
                break
        return generated

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)

    @classmethod
    def from_config_dict(cls, config_dict: dict) -> "LLMTTSBackbone":
        known = set(asdict(LLMTTSConfig()).keys())
        extra = set(config_dict) - known
        if extra:
            raise ValueError(f"Unknown LLMTTSConfig fields: {sorted(extra)}")
        return cls(LLMTTSConfig(**config_dict))

    def state_dict_copy(self) -> dict:
        return {k: v.detach().clone() for k, v in self.state_dict().items()}
