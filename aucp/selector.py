"""Checkpoint selection helper built on top of AUCp.

The canonical workflow during training is:

1. After each epoch (or every N iterations), score *all* training-normal
   samples and *all* unlabeled-test samples with the current model.
2. Call ``aucp_score`` to get one number per checkpoint.
3. After training finishes, pick the checkpoint with the highest AUCp.

``select_best_checkpoint`` packages step 3 so you can call it on the result of
a training loop without writing boilerplate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np

from aucp.metric import aucp_score


@dataclass(frozen=True)
class CheckpointScore:
    """One entry in the per-checkpoint AUCp log."""

    checkpoint: object
    aucp: float


def select_best_checkpoint(
    scored_checkpoints: Iterable[Tuple[object, np.ndarray, np.ndarray]]
    | Iterable[Tuple[object, float]],
    *,
    score_fn: Optional[Callable[[object], Tuple[np.ndarray, np.ndarray]]] = None,
) -> CheckpointScore:
    """Pick the checkpoint that maximizes AUCp.

    Two calling conventions are supported.

    **Pre-scored** (cheapest, recommended): pass an iterable of
    ``(checkpoint, aucp_value)`` tuples — for example, the rows of the CSV
    your training script already writes::

        rows = [(path, float(aucp)) for path, aucp in csv_rows]
        best = select_best_checkpoint(rows)

    **Score-on-demand**: pass an iterable of
    ``(checkpoint, train_normal_scores, test_unlabeled_scores)`` and we'll
    compute AUCp for each one::

        triples = [(ckpt, train_s, test_s) for ckpt, train_s, test_s in stream]
        best = select_best_checkpoint(triples)

    You can also supply ``score_fn`` to compute scores lazily from a checkpoint
    identifier, which is useful when you don't want to materialize all
    score arrays in memory at once.

    Returns
    -------
    CheckpointScore
        The winning entry. Ties are broken by the first occurrence.

    Raises
    ------
    ValueError
        If the input is empty or every checkpoint produced a non-finite AUCp.
    """
    best: Optional[CheckpointScore] = None
    n_seen = 0

    for entry in scored_checkpoints:
        n_seen += 1
        if score_fn is not None:
            ckpt = entry
            train_s, test_s = score_fn(ckpt)
            value = aucp_score(train_s, test_s)
        elif len(entry) == 2:
            ckpt, value = entry
            value = float(value)
        elif len(entry) == 3:
            ckpt, train_s, test_s = entry
            value = aucp_score(train_s, test_s)
        else:
            raise ValueError(
                "Each entry must be (ckpt, aucp), (ckpt, train_scores, test_scores), "
                f"or a checkpoint id when score_fn is given. Got tuple of length {len(entry)}."
            )

        if not np.isfinite(value):
            continue
        if best is None or value > best.aucp:
            best = CheckpointScore(checkpoint=ckpt, aucp=value)

    if n_seen == 0:
        raise ValueError("scored_checkpoints is empty")
    if best is None:
        raise ValueError("All checkpoints produced non-finite AUCp values")
    return best
