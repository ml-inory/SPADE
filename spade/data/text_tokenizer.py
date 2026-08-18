"""Small character-level tokenizer for the built-in TTS benchmark."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_ALPHABET = (
    " abcdefghijklmnopqrstuvwxyz0123456789"
    ".,!?-'"
)


@dataclass
class CharTokenizer:
    """Character-level tokenizer with BOS/EOS/PAD special tokens."""

    alphabet: str = DEFAULT_ALPHABET
    bos_id: int = 0
    eos_id: int = 1
    pad_id: int = 2

    def __post_init__(self) -> None:
        chars = list(dict.fromkeys(self.alphabet))
        self._char_to_id = {c: i + 3 for i, c in enumerate(chars)}
        self._id_to_char = {v: k for k, v in self._char_to_id.items()}
        self.vocab_size = 3 + len(chars)
        if self.vocab_size > 65535:
            raise ValueError("Alphabet too large for token ids")

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in text.lower():
            if ch not in self._char_to_id:
                ch = " "  # unknown chars collapse to space
            ids.append(self._char_to_id[ch])
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        return "".join(self._id_to_char.get(int(i), " ") for i in ids).strip()

    @property
    def special_ids(self) -> tuple[int, int, int]:
        return self.bos_id, self.eos_id, self.pad_id

