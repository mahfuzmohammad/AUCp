"""Summarize AUC vs AUCp results across reconstruction-method runs.

Reads per-epoch metrics from ``<output_root>/reconstruction/<dataset>/<method>/fold_0``
and prints, for each (dataset, method), the AUC of the last-epoch model and
the AUC of the model that maximized AUCp. Output root is configurable via the
``AUCP_OUTPUT_ROOT`` environment variable; see ``aucp/paths.py``.
"""

import os
import sys
from glob import glob

_REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aucp.paths import output_root


DATASETS = "rsna vin brain lag".split()
METHODS = "ae ae-l1".split()


def main() -> None:
    base = output_root() / "reconstruction"
    for dataset in DATASETS:
        for method in METHODS:
            run_dir = base / dataset / method / "fold_0"
            aucs, aucps = [], []

            auc_txts = sorted(glob(str(run_dir / "auc_metrics_*.txt")))
            aucp_txts = sorted(glob(str(run_dir / "aucp_metrics_*.txt")))

            for aucp_txt in aucp_txts:
                auc_txt = aucp_txt.replace("aucp_metrics", "auc_metrics")
                if auc_txt in auc_txts:
                    with open(auc_txt) as f:
                        for line in f:
                            if "AUC" in line and "PixAUC" not in line:
                                aucs.append(float(line.split()[-1]))
                    with open(aucp_txt) as f:
                        for line in f:
                            if "AUC" in line and "PixAUC" not in line:
                                aucps.append(float(line.split()[-1]))

            if not aucps:
                print(f"Dataset: {dataset}, Method: {method} — no runs found in {run_dir}")
                print("-" * 64)
                continue

            max_aucp = max(v for v in aucps if v > 0)
            idx = aucps.index(max_aucp)
            max_auc = aucs[idx]
            last_auc = aucs[-1]

            print(
                f"Dataset: {dataset}, Method: {method}, "
                f"Last AUC: {last_auc:.4f}, AUC for max AUCp: {max_auc:.4f}"
            )
            print("-" * 64)


if __name__ == "__main__":
    main()
