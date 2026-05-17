"""AUCp: Pseudo-AUC for inference-time model selection without labels.

Quick start
-----------
>>> from aucp import aucp_score, select_best_checkpoint
>>> import numpy as np
>>> train_normal_scores = np.array([0.10, 0.12, 0.08, 0.15])     # known normal
>>> test_unlabeled_scores = np.array([0.92, 0.18, 0.87, 0.40])   # unknown mix
>>> aucp_score(train_normal_scores, test_unlabeled_scores)
1.0

The score above treats every test sample as pseudo-abnormal. With a large,
representative normal training set, AUCp ranks checkpoints almost identically
to the true AUC (see Proposition 1 in the paper).
"""

from aucp.metric import aucp_score, aucp_from_labels, estimate_auc_from_aucp
from aucp.paths import data_root, dataset_path, output_root
from aucp.selector import select_best_checkpoint

__version__ = "0.1.0"
__all__ = [
    "aucp_score",
    "aucp_from_labels",
    "estimate_auc_from_aucp",
    "select_best_checkpoint",
    "data_root",
    "dataset_path",
    "output_root",
]
