"""Synthetic speech-codec benchmark.

The paper's pipeline is evaluated with a *word-error-rate based layer
importance* (WLI), i.e. WER of synthesized speech measured against the
reference transcript. To make the whole SPADE loop runnable end-to-end
without multi-gigabyte speech corpora, this module provides a deterministic
"speech codec": text is mapped to a sequence of discrete code tokens that
depends on the speaker and on neighbouring characters, and can be decoded
back to text. An LLM-TTS model must therefore learn a genuinely causal,
context-dependent mapping -- decoding mistakes translate directly into WER.

The mapping is intentionally *not* a trivial lookup:

    code_t = base(char_t) + jitter(speaker, char_{t-1}, char_t, t)

where ``jitter`` is a deterministic function of the speaker id, the previous
character, the current character, and the absolute position. It is chosen to
be a *learnable* linear combination of those features (not a random hash),
so a small Transformer can generalize from the training texts to unseen
ones -- while still being context-, speaker-, and position-dependent.
Character bases are spaced ``jitter_range`` apart, so for a fixed previous
character each character maps to a *distinct* code; the decoder replays the
same function causally and recovers the text exactly.
"""

from __future__ import annotations


class ToySpeechCodec:
    """Invertible synthetic codec producing discrete speech-code tokens."""

    def __init__(
        self,
        alphabet: str,
        code_vocab_size: int = 512,
        jitter_range: int = 7,
        bos_id: int = 0,
        eos_id: int = 1,
        pad_id: int = 2,
        num_speaker_slots: int = 64,
    ) -> None:
        self.speaker_bos_id = code_vocab_size - num_speaker_slots
        max_char_code = 3 + len(alphabet) * jitter_range
        if max_char_code >= self.speaker_bos_id:
            raise ValueError(
                "alphabet too large for the reserved speaker-token range; "
                "increase code_vocab_size or reduce jitter_range"
            )
        self.alphabet = alphabet
        self.code_vocab_size = code_vocab_size
        self.jitter_range = jitter_range
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        # Bases are spaced jitter_range apart so the mapping is injective
        # for a fixed previous character (see module docstring).
        self._char_to_base = {
            c: 4 + i * jitter_range for i, c in enumerate(alphabet)
        }
        self._chars = list(alphabet)

    def _jitter(self, speaker: int, prev_char: str, char: str, position: int) -> int:
        # Learnable smooth mapping: linear combination of speaker id,
        # previous-char id, current-char id, and position, reduced mod J.
        prev_id = self._char_to_base.get(prev_char, 0)
        char_id = self._char_to_base.get(char, 0)
        return (
            speaker * 2 + prev_id * 3 + char_id * 5 + position
        ) % self.jitter_range

    def encode(
        self,
        text: str,
        speaker: int,
        prev_char: str = "\x00",
        start_position: int = 0,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        """Encode ``text`` into code tokens.

        When ``add_bos``, the speaker token is emitted right after BOS so
        the model sees the speaker conditioning before any code.
        """
        codes: list[int] = []
        if add_bos:
            codes.append(self.bos_id)
            codes.append(self.speaker_bos_id + speaker)
        position = start_position
        for ch in text.lower():
            if ch not in self._char_to_base:
                ch = " "
            base = self._char_to_base[ch]
            jitter = self._jitter(speaker, prev_char, ch, position)
            codes.append(base + jitter)
            prev_char = ch
            position += 1
        if add_eos:
            codes.append(self.eos_id)
        return codes

    def decode(
        self,
        codes: list[int] | tuple[int, ...],
        speaker: int = 0,
        prev_char: str = "\x00",
        start_position: int = 0,
    ) -> str:
        """Deterministically decode code tokens back to text."""
        out: list[str] = []
        position = start_position
        for code in codes:
            code = int(code)
            if code in (self.bos_id, self.pad_id):
                continue
            if code == self.eos_id:
                break
            if code >= self.speaker_bos_id:
                continue  # speaker token region
            # Block index uniquely identifies the character for a fixed
            # previous character; verify the jitter term as a sanity check.
            block = (code - 4) // self.jitter_range
            if not (0 <= block < len(self._chars)):
                found = " "
            else:
                found = self._chars[block]
                base = 4 + block * self.jitter_range
                if self._jitter(speaker, prev_char, found, position) != code - base:
                    found = " "
            out.append(found)
            prev_char = found
            position += 1
        return "".join(out).strip()

    def decode_viterbi(
        self,
        codes: list[int] | tuple[int, ...],
        speaker: int = 0,
        prev_char: str = "\x00",
        start_position: int = 0,
    ) -> str:
        """ASR-style maximum-consistency decoding (Viterbi).

        The exact decoder (:meth:`decode`) is brittle: one wrong code changes
        the previous character, which cascades into the next jitter term.
        Real systems evaluate intelligibility with an ASR model that tolerates
        such errors, so this decoder finds the character sequence that is
        *maximally consistent* with all code tokens under the codec's
        context-dependent jitter function -- analogous to decoding with
        language constraints. A single corrupted code only degrades its own
        character instead of derailing the whole utterance.
        """
        seq: list[int] = []
        for code in codes:
            code = int(code)
            if code in (self.bos_id, self.pad_id):
                continue
            if code == self.eos_id:
                break
            if code >= self.speaker_bos_id:
                continue
            seq.append(code)
        if not seq:
            return ""

        chars = self._chars
        n_chars = len(chars)
        n = len(seq)
        neg = -1e9

        def consistent(prev: str, cur: str, code: int, pos: int) -> float:
            base = self._char_to_base[cur]
            return 1.0 if (base + self._jitter(speaker, prev, cur, pos)) == code else neg

        best = [consistent(prev_char, c, seq[0], start_position) for c in chars]
        back: list[list[int]] = [[-1] * n for _ in range(n_chars)]

        for t in range(1, n):
            new: list[float] = []
            for j, cur in enumerate(chars):
                vals = [
                    best[i]
                    + consistent(chars[i], cur, seq[t], start_position + t)
                    for i in range(n_chars)
                ]
                j_star = max(range(n_chars), key=lambda i: vals[i])
                new.append(vals[j_star])
                back[j][t] = j_star
            best = new

        j = max(range(n_chars), key=lambda i: best[i])
        path: list[str] = []
        for t in range(n - 1, -1, -1):
            path.append(chars[j])
            j = back[j][t]
        return "".join(reversed(path)).strip()
