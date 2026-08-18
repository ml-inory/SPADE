"""Dataset + collation for the built-in synthetic LLM-TTS benchmark."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from spade.data.text_tokenizer import CharTokenizer
from spade.data.toy_codec import ToySpeechCodec


WORDS = (
    "the quick brown fox jumps over lazy dog hello world speech synthesis "
    "structured pruning adaptive distillation efficient language model "
    "autoregressive codec transformer attention knowledge transfer "
    "intelligibility naturalness speaker similarity zero shot generalization "
    "training data quality latency memory footprint deployment on device"
).split()


def make_synthetic_texts(num_samples: int, seed: int = 0, max_words: int = 12) -> list[str]:
    """Generate ``num_samples`` short pseudo-sentences for the benchmark."""
    rng = random.Random(seed)
    texts: list[str] = []
    for _ in range(num_samples):
        n_words = rng.randint(4, max_words)
        words = [rng.choice(WORDS) for _ in range(n_words)]
        text = " ".join(words)
        if rng.random() < 0.25:
            text += rng.choice([".", "!", "?"])
        texts.append(text)
    return texts


@dataclass
class SyntheticTTSDataset(Dataset):
    """Text/audio-code pairs from the :class:`ToySpeechCodec`.

    Each sample contains:

    * ``input_ids``: ``[BOS, speaker_token, <audio codes with prompt>]``
      followed by the text tokens -- the autoregressive conditioning;
    * ``labels``: the audio-code tokens (target positions only).

    ``use_prompt`` optionally prepends a reference speaker utterance (audio
    codes of another sentence spoken by the same speaker), mirroring the
    zero-shot LLM-TTS setup used by the paper's WLI evaluation.
    """

    texts: list[str]
    codec: ToySpeechCodec
    tokenizer: CharTokenizer
    num_speakers: int = 4
    use_prompt: bool = False
    max_len: int = 96
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def __len__(self) -> int:
        return len(self.texts)

    def _speaker(self, index: int) -> int:
        return index % self.num_speakers

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        text = self.texts[index]
        speaker = self._speaker(index)
        text_ids = self.tokenizer.encode(text)

        # Audio codes for the utterance (the prediction target).
        audio_codes = self.codec.encode(text, speaker)

        prompt_codes: list[int] = []
        if self.use_prompt:
            other = self._rng.choice([i for i in range(len(self.texts)) if i != index])
            other_text = self.texts[other]
            other_speaker = self._speaker(other)
            # Reference prompt: another utterance by a *different* speaker is
            # allowed but keeps speaker consistency for the demo.
            prompt_codes = self.codec.encode(
                other_text,
                speaker,
                add_bos=False,
                add_eos=True,
            )

        # Condition: prompt codes (if any) + text tokens + audio codes.
        input_ids = prompt_codes + text_ids + audio_codes
        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]

        # Labels: predict audio codes at their positions; -100 elsewhere.
        prompt_len = len(prompt_codes)
        text_len = len(text_ids)
        audio_start = min(prompt_len + text_len, len(input_ids))
        labels = [-100] * len(input_ids)
        for i in range(audio_start, min(len(input_ids), audio_start + len(audio_codes))):
            labels[i] = input_ids[i]
        # Keep sequence length consistent: truncation above may cut codes.

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "speaker_ids": torch.tensor(speaker, dtype=torch.long),
            "text": text,
        }


def collate_llmtts_batch(
    batch: list[dict[str, torch.Tensor]],
    pad_id: int = 2,
) -> dict[str, torch.Tensor]:
    """Pad a list of samples into a batched dict (dropping text metadata)."""
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [s["input_ids"] for s in batch], batch_first=True, padding_value=pad_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [s["labels"] for s in batch], batch_first=True, padding_value=-100
    )
    speaker_ids = torch.stack([s["speaker_ids"] for s in batch])
    attention_mask = (input_ids != pad_id).long()
    return {
        "input_ids": input_ids,
        "labels": labels,
        "speaker_ids": speaker_ids,
        "attention_mask": attention_mask,
    }
