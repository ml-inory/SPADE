"""Efficiency metrics for pruned vs. teacher LLM-TTS backbones."""

from __future__ import annotations

import time

import torch


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_depth(model: torch.nn.Module) -> int:
    return model.config.n_layer


def measure_generation_latency(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    n_runs: int = 3,
    device: str | torch.device = "cpu",
    hop_seconds: float = 0.08,
) -> dict:
    """Wall-clock generation latency and a synthetic real-time factor (RTF).

    ``hop_seconds`` maps one speech-code token to an audio duration (a
    codec-dependent constant); RTF = generation wall time / synthetic audio
    duration, so values below 1 mean faster than real time.
    """
    model.eval()
    x = input_ids.to(device)
    timings: list[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            model.generate(x, max_new_tokens=max_new_tokens, temperature=0.0)
            timings.append(time.perf_counter() - start)
    avg_seconds = sum(timings) / len(timings)
    audio_seconds = max_new_tokens * hop_seconds
    return {
        "avg_generation_seconds": round(avg_seconds, 4),
        "synthetic_rtf": round(avg_seconds / audio_seconds, 4) if audio_seconds > 0 else float("inf"),
        "tokens_per_second": round(max_new_tokens / avg_seconds, 1) if avg_seconds > 0 else 0.0,
    }

