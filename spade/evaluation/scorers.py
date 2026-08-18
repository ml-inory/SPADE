"""WER scorers used by SPADE's WLI pruning and final evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import torch

from spade.data.text_tokenizer import CharTokenizer
from spade.data.toy_codec import ToySpeechCodec
from spade.metrics import word_error_rate


class WERScorer(Protocol):
    """Protocol: score a model on one evaluation sample (lower is better)."""

    def score(self, model: torch.nn.Module, sample: dict) -> float:
        ...


@dataclass
class TokenCodecWERScorer:
    """WER scorer for the built-in synthetic speech-codec benchmark.

    Conditions the model on the sample's text prefix (``cond_len`` tokens),
    generates ``max_new_tokens`` speech-code tokens, decodes them back to
    text with the synthetic codec, and returns the word error rate against
    the reference transcript.
    """

    codec: ToySpeechCodec
    tokenizer: CharTokenizer
    max_new_tokens: int = 128
    temperature: float = 0.0
    device: str | torch.device = "cpu"

    def score(self, model: torch.nn.Module, sample: dict) -> float:
        model.eval()
        input_ids = sample["input_ids"].unsqueeze(0).to(self.device)
        cond_len = int(sample.get("cond_len", len(sample["input_ids"])))
        condition = input_ids[:, :cond_len]
        speaker = int(sample["speaker_ids"])
        reference = str(sample["text"])

        generated = model.generate(
            condition,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        code_ids = generated[0, cond_len:].tolist()
        # Strip EOS if the model produced one before max_new_tokens.
        if self.codec.eos_id in code_ids:
            code_ids = code_ids[: code_ids.index(self.codec.eos_id)]
        hypothesis = self.codec.decode(code_ids, speaker)
        return word_error_rate(reference, hypothesis)


@dataclass
class WhisperWERScorer:
    """WER scorer for real audio using OpenAI Whisper.

    ``decode_fn`` turns a generated code sequence into a waveform so the
    scorer is model-agnostic::

        def decode_fn(codes: list[int], speaker: int) -> tuple[torch.Tensor, int]:
            return waveform, sample_rate

    Requires ``openai-whisper`` (``pip install spade-llm-tts[wer]``).
    """

    decode_fn: Optional[Callable[[list[int], int], tuple[torch.Tensor, int]]]
    model_name: str = "base"
    device: str | torch.device | None = None
    language: Optional[str] = None

    def __post_init__(self) -> None:
        try:
            import whisper  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai-whisper is required for WhisperWERScorer; "
                "install with `pip install spade-llm-tts[wer]`"
            ) from exc
        import whisper

        self._model = whisper.load_model(self.model_name, device=self.device)

    def score(self, model: torch.nn.Module, sample: dict) -> float:
        if self.decode_fn is None:
            raise ValueError(
                "decode_fn (codes -> waveform) must be provided to score real audio"
            )
        # Generate code tokens with the model, then decode to audio and
        # transcribe. Sample layout must match TokenCodecWERScorer.
        input_ids = sample["input_ids"].unsqueeze(0).to(self.device)
        cond_len = int(sample.get("cond_len", len(sample["input_ids"])))
        condition = input_ids[:, :cond_len]
        speaker = int(sample["speaker_ids"])
        max_new = int(sample.get("max_new_tokens", 256))

        generated = model.generate(condition, max_new_tokens=max_new, temperature=0.0)
        code_ids = generated[0, cond_len:].tolist()
        waveform, sample_rate = self.decode_fn(code_ids, speaker)
        waveform = waveform.to(self.device) if torch.is_tensor(waveform) else waveform
        text = self._model.transcribe(
            waveform, language=self.language, fp16=False
        )["text"]
        return word_error_rate(str(sample["text"]), text)

