"""Word error rate (WER) computation used by SPADE's WLI pruning."""

from __future__ import annotations


def _edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance between two token sequences."""
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, y in enumerate(b, start=1):
            cur = dp[j]
            dp[j] = min(
                dp[j] + 1,          # deletion
                dp[j - 1] + 1,      # insertion
                prev + (x != y),    # substitution
            )
            prev = cur
    return dp[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER between two transcripts (0.0 == perfect, 1.0 == completely wrong).

    Falls back to character-level comparison when either side has no words,
    so degenerate empty hypotheses still produce a bounded error rate.
    """
    ref = reference.strip().lower().split()
    hyp = hypothesis.strip().lower().split()
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    return _edit_distance(ref, hyp) / len(ref)

