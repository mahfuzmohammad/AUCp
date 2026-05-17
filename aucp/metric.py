"""Core AUCp metric.

AUCp is the area under the ROC curve computed with *pseudo* labels:
every sample in the training (normal-only) set is treated as negative, and
every sample in the unlabeled test set is treated as positive. With a large
representative normal set, this approximates the true AUC up to a known bias
(Proposition 1 + Eq. 12 in the paper).

The implementation deliberately depends on nothing beyond NumPy and
scikit-learn so it can be dropped into any pipeline.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score


ArrayLike = np.ndarray


def _as_1d(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def aucp_score(
    train_normal_scores: ArrayLike,
    test_unlabeled_scores: ArrayLike,
) -> float:
    """Compute AUCp from raw anomaly scores.

    Parameters
    ----------
    train_normal_scores
        Anomaly scores assigned to the *known-normal* training samples.
        Higher means more anomalous.
    test_unlabeled_scores
        Anomaly scores assigned to the *unlabeled* test samples. AUCp treats
        these as pseudo-positive (abnormal).

    Returns
    -------
    float
        AUCp in [0, 1]. Higher means the model separates training-normal from
        unlabeled-test more cleanly, which under the paper's assumptions
        correlates with the true AUC.
    """
    train_scores = _as_1d(train_normal_scores, "train_normal_scores")
    test_scores = _as_1d(test_unlabeled_scores, "test_unlabeled_scores")

    scores = np.concatenate([train_scores, test_scores])
    pseudo_labels = np.concatenate([
        np.zeros(train_scores.size, dtype=np.int8),
        np.ones(test_scores.size, dtype=np.int8),
    ])
    return float(roc_auc_score(pseudo_labels, scores))


def aucp_from_labels(
    pseudo_labels: ArrayLike,
    scores: ArrayLike,
) -> float:
    """Compute AUCp when pseudo labels are already assembled.

    Equivalent to ``sklearn.metrics.roc_auc_score(pseudo_labels, scores)``
    with the convention that 0 = known-normal (from training) and 1 = unlabeled
    test sample. Provided as a thin wrapper so callers can be explicit about
    the pseudo-label semantics.
    """
    y = np.asarray(pseudo_labels).ravel()
    s = _as_1d(scores, "scores")
    if y.shape != s.shape:
        raise ValueError("pseudo_labels and scores must have the same length")
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("pseudo_labels must contain only 0 and 1")
    return float(roc_auc_score(y, s))


def estimate_auc_from_aucp(
    aucp: float,
    contamination: float,
) -> float:
    """Debias AUCp into an estimate of the true AUC.

    Implements Eq. 12 of the paper::

        AUC ~= (AUCp - 0.5 * rho) / (1 - rho)

    where ``rho`` is the fraction of *true normals* hiding inside the
    "unlabeled" (pseudo-positive) test set. If you know the dataset's anomaly
    prevalence, ``rho = 1 - prevalence``. If you only have a mixture-proportion
    estimate (KM-MPE, AlphaMax, TIcE, ...), feed it in here.

    Notes
    -----
    The correction assumes the no-covariate-shift regime (Eq. 9, 10 in the
    paper). When ``contamination`` is close to 1.0 the denominator becomes
    unstable and the estimate is not meaningful; we raise in that case.
    """
    if not 0.0 <= contamination < 1.0:
        raise ValueError("contamination must be in [0, 1)")
    return (float(aucp) - 0.5 * contamination) / (1.0 - contamination)
