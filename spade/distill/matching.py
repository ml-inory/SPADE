"""Layer matching between the pruned student and the teacher.

The paper's *dynamic* matching: student layer ``l_n`` (copied from retained
teacher layer ``l_m``) is distilled against the *last teacher layer before
the next retained layer* -- i.e. the end of the pruned segment. This forces
the student layer to absorb the transformations of every removed layer in
that segment. The final student layer distills against the final teacher
layer. The *static* matching (student layer ``n`` against its own retained
teacher layer) is provided as the ablation baseline.
"""

from __future__ import annotations

from typing import Sequence


def dynamic_teacher_targets(
    retained_indices: Sequence[int],
    n_teacher_layers: int,
) -> list[int]:
    """Teacher layer targets per student layer (dynamic matching).

    For student layer ``j`` (retained teacher layer ``r_j``), the target is
    ``r_{j+1} - 1`` -- the last teacher layer before the next retained layer
    -- or ``n_teacher_layers - 1`` for the final student layer.
    """
    retained = sorted(int(i) for i in retained_indices)
    if not retained:
        raise ValueError("retained_indices must be non-empty")
    if min(retained) < 0 or max(retained) >= n_teacher_layers:
        raise ValueError("retained_indices out of range")
    targets: list[int] = []
    for j, r in enumerate(retained):
        if j + 1 < len(retained):
            targets.append(retained[j + 1] - 1)
        else:
            targets.append(n_teacher_layers - 1)
    return targets


def static_teacher_targets(retained_indices: Sequence[int]) -> list[int]:
    """Ablation: each student layer distills from its own retained layer."""
    return [int(i) for i in sorted(retained_indices)]

