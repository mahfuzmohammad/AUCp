"""Synthetic demo of AUCp on a 2D toy problem.

What this shows
---------------
We simulate a fictitious "training run" where 30 candidate checkpoints score a
held-out set with progressively less noise. The true AUC (computed with the
hidden labels) and AUCp (computed without them) both rise together, and the
checkpoint that maximizes AUCp also has near-best true AUC — which is the
empirical claim of the paper.

Run it::

    python examples/synthetic_demo.py

It prints the AUC of the AUCp-selected checkpoint, the AUC of the best
possible checkpoint, and the Pearson correlation between AUCp and AUC across
all 30 checkpoints. No GPUs, no datasets, no PyTorch.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from aucp import aucp_score, select_best_checkpoint


def simulate_checkpoints(
    n_normal_train: int = 500,
    n_normal_test: int = 200,
    n_abnormal_test: int = 200,
    n_checkpoints: int = 30,
    seed: int = 0,
):
    """Generate per-checkpoint score arrays where signal grows with the epoch index."""
    rng = np.random.default_rng(seed)
    true_labels = np.concatenate([
        np.zeros(n_normal_test, dtype=np.int8),
        np.ones(n_abnormal_test, dtype=np.int8),
    ])

    results = []
    for epoch in range(n_checkpoints):
        # As "training" progresses, the gap between normal and abnormal grows.
        signal = epoch / (n_checkpoints - 1)  # 0 -> 1
        noise = 1.0 - 0.7 * signal            # 1.0 -> 0.3

        train_scores = rng.normal(loc=0.0, scale=noise, size=n_normal_train)
        normal_test = rng.normal(loc=0.0, scale=noise, size=n_normal_test)
        abnormal_test = rng.normal(loc=signal, scale=noise, size=n_abnormal_test)

        test_scores = np.concatenate([normal_test, abnormal_test])
        true_auc = roc_auc_score(true_labels, test_scores)
        pseudo_auc = aucp_score(train_scores, test_scores)
        results.append({
            "epoch": epoch,
            "train_scores": train_scores,
            "test_scores": test_scores,
            "true_labels": true_labels,
            "auc": true_auc,
            "aucp": pseudo_auc,
        })
    return results


def main() -> None:
    runs = simulate_checkpoints()
    aucs = np.array([r["auc"] for r in runs])
    aucps = np.array([r["aucp"] for r in runs])

    selected = select_best_checkpoint([(r["epoch"], r["aucp"]) for r in runs])
    best_true = int(np.argmax(aucs))

    corr = float(np.corrcoef(aucs, aucps)[0, 1])

    print("checkpoint | true AUC | AUCp")
    print("-----------+----------+------")
    for r in runs:
        print(f"  epoch {r['epoch']:>2}  |  {r['auc']:.4f}  | {r['aucp']:.4f}")

    print()
    print(f"AUCp-selected checkpoint : epoch {selected.checkpoint} "
          f"(AUCp={selected.aucp:.4f}, true AUC={aucs[selected.checkpoint]:.4f})")
    print(f"Best possible checkpoint : epoch {best_true} (true AUC={aucs[best_true]:.4f})")
    print(f"Pearson corr(AUC, AUCp)  : {corr:.4f}")


if __name__ == "__main__":
    main()
