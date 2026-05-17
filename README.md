# AUCp — pick the best checkpoint without any labels

[![IEEE TMI](https://img.shields.io/badge/IEEE_TMI-10.1109%2FTMI.2026.3684946-00629B?logo=ieee)](https://doi.org/10.1109/TMI.2026.3684946)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Code style: PEP8](https://img.shields.io/badge/code%20style-PEP8-1f425f.svg)](https://www.python.org/dev/peps/pep-0008/)

**TL;DR** — Training an anomaly detector but you don't have labels to pick the best epoch? Treat every test sample as pseudo-abnormal, every training sample as pseudo-normal, compute AUC. That number is **AUCp**, and it ranks checkpoints almost identically to the true AUC (Pearson r > 0.95 on 5 of 7 medical-imaging benchmarks).

> Rahman Siddiquee et al., *AUCp: Pseudo-AUC for Inference Model Selection with Unlabeled Validation Data in Abnormality Detection*, **IEEE Transactions on Medical Imaging**, 2026. [`doi:10.1109/TMI.2026.3684946`](https://doi.org/10.1109/TMI.2026.3684946)

---

## The idea in one picture

```
       known normal                       unlabeled
   ┌──────────────────┐         ┌──────────────────────────┐
   │  D_train (N − k) │         │   D_test  (k normals +   │
   │   ↓  pseudo y=0  │         │           d abnormals)   │
   │                  │         │   ↓  pseudo y=1  (all)   │
   └────────┬─────────┘         └───────────┬──────────────┘
            │                               │
            └───────────► score with ◄──────┘
                    candidate model M_θ
                          │
                          ▼
                ┌──────────────────┐
                │  AUCp = AUC on   │      Higher AUCp ≈ higher true AUC
                │  pseudo labels   │  ──► pick the checkpoint with max(AUCp)
                └──────────────────┘
```

**Why it works** (Proposition 1, paper §II): when the training-normal set is large and representative, the pseudo labels differ from the true labels only on the `k` normals hiding inside the test mixture — and that gap shrinks as `1 − k/n`. So argmax over checkpoints lines up with the true argmax even when AUCp's absolute value is biased downward.

If you also know (or can estimate) the test-set contamination `ρ = k / (k + d)`, you can recover an unbiased AUC estimate via Eq. 12 (`estimate_auc_from_aucp`).

---

## Install

```bash
# Just the metric (NumPy + scikit-learn, no PyTorch)
pip install -e .

# Plus dependencies needed to reproduce the paper's experiments
pip install -e ".[experiments]"
```

## 60-second example

```python
import numpy as np
from aucp import aucp_score, select_best_checkpoint

# Score every training-normal sample and every unlabeled test sample
# with each candidate checkpoint.
train_normal_scores = model.score(train_loader)   # higher = more anomalous
test_unlabeled_scores = model.score(test_loader)

# One number per checkpoint:
aucp = aucp_score(train_normal_scores, test_unlabeled_scores)

# Across a training run:
log = [(epoch, aucp_score(s_tr, s_te)) for epoch, (s_tr, s_te) in scores.items()]
best = select_best_checkpoint(log)
print(f"Pick epoch {best.checkpoint} (AUCp = {best.aucp:.4f})")
```

That's the whole interface. Run [`examples/synthetic_demo.py`](examples/synthetic_demo.py) for a no-GPU sanity check, or open [`examples/aucp_walkthrough.ipynb`](examples/aucp_walkthrough.ipynb) for a guided tour with plots.

---

## Headline results

The paper applies AUCp to **18 anomaly-detection methods × 7 medical-imaging datasets**. A few of the largest gains, taken from Tab. VI in the paper (last-epoch AUC vs AUCp-selected AUC, %):

| Method        | Dataset      | Last-epoch AUC | AUCp-selected AUC | Δ           |
|---------------|--------------|---------------:|------------------:|------------:|
| FPI           | Brain Tumor  |          29.32 |             88.81 |  **+59.49** |
| CutPaste      | Brain Tumor  |          22.08 |             82.77 |  **+60.69** |
| FPI-Poisson   | Brain Tumor  |          24.25 |             78.09 |  **+53.84** |
| MemAE         | Brain Tumor  |          60.30 |             84.35 |  **+24.05** |
| NSA           | VinDr-CXR    |          49.99 |             76.22 |  **+26.23** |
| FPI           | VinDr-CXR    |          47.46 |             69.14 |  **+21.68** |
| FPI-Poisson   | Camelyon16   |          54.68 |             71.31 |  **+16.63** |
| FPI           | BraTS21      |          51.42 |             76.45 |  **+25.03** |

On the unsupervised brain-MRI side (Tab. II), swapping FID for AUCp during model selection lifts **HealthyGAN** by **+21.59% AUC on Alzheimer's** and **Brainomaly** to a benchmark-leading **0.8960 AUC on headache** (+1.86% over FID-selected, +28.83% over the strongest competitor).

Across the four self-supervised methods that benefit most (CutPaste, FPI, FPI-Poisson, NSA), Pearson correlation between AUCp and the true AUC exceeds **0.95** on 5 of 7 datasets (Tab. VIII).

---

## Repository map

```
AUCp/
├── aucp/                      ← standalone Python package (start here)
│   ├── metric.py              ← aucp_score, estimate_auc_from_aucp
│   ├── selector.py            ← select_best_checkpoint
│   └── paths.py               ← AUCP_DATA_ROOT / AUCP_OUTPUT_ROOT resolver
├── examples/
│   ├── synthetic_demo.py      ← runs in seconds, no datasets needed
│   └── aucp_walkthrough.ipynb ← guided notebook with plots
├── reconstruction/            ← image- and feature-reconstruction baselines
│   ├── train.py / test.py     ← AE, AE-U, MemAE, AE-Perceptual, ...
│   └── feature-autoencoder/   ← FAE-MSE, FAE-SSIM
├── ssl/                       ← self-supervised baselines
│   ├── one_stage/             ← CutPaste, FPI, FPI-Poisson, NSA
│   └── two_stage/             ← CutPaste / AnatPaste / PANDA / Mean-Shifted
├── pyproject.toml             ← `pip install -e .`
├── requirements.txt           ← experiment-side dependencies
├── CITATION.cff
├── CONTRIBUTING.md
└── LICENSE                    ← MIT
```

---

## Reproducing the paper

### 1. Data

The seven public benchmarks are pre-processed in [MedIAnomaly-Data](https://zenodo.org/records/12677223) (same splits as [MedIAnomaly](https://arxiv.org/abs/2404.04518)). Download, extract, and either place everything under `~/MedIAnomaly-Data/` or point the env var:

```bash
export AUCP_DATA_ROOT=/path/to/MedIAnomaly-Data
```

ISIC2018 ships separately — fetch `ISIC2018_Task3_Training`, `ISIC2018_Task3_Test`, and both ground-truth CSVs from the [ISIC archive](https://challenge.isic-archive.com/data/#2018) and drop them inside the data root.

Expected layout:

```
$AUCP_DATA_ROOT
├── RSNA/         {images/, data.json}
├── VinCXR/       {images/, data.json}
├── BrainTumor/   {images/, data.json}
├── LAG/          {images/, data.json}
├── ISIC2018_Task3/
├── Camelyon16/   {train/good, test/good, test/Ungood}
└── BraTS2021/    {train, test/{normal,tumor,annotation}}
```

Experiment artifacts (checkpoints, per-epoch AUC/AUCp CSVs, logs) land in `./output/` by default; override with `export AUCP_OUTPUT_ROOT=/path/to/runs`.

### 2. Reconstruction methods (`reconstruction/`)

Trains and evaluates AE, AE-L1, AE-SSIM, AE-Perceptual, MemAE, AE-U, GANomaly, VAE variants, and constrained / cascade AE:

```bash
cd reconstruction
bash train_eval.sh                  # whole sweep
# or one model:
python train.py -d rsna -m ae-perceptual -g 0 -f 0
python test.py  -d rsna -m ae-perceptual -g 0 -f 0
```

After every epoch, both AUC and AUCp are written to `auc_metrics_*.txt` / `aucp_metrics_*.txt`. Summarize across datasets with:

```bash
python reconstruction/all_aucps.py
```

### 3. Self-supervised methods (`ssl/`)

```bash
# one-stage (CutPaste, FPI, FPI-Poisson, NSA)
cd ssl/one_stage
bash train_eval.sh
python AUCP_test.py -d rsna -s NSA -g 0 -f 0

# two-stage (CutPaste / AnatPaste / PANDA / Mean-Shifted-AD)
cd ssl/two_stage
bash train_eval.sh
python aucp_eval.py --type rsna --variant anatpaste
```

Every training script saves a checkpoint per epoch; the eval scripts iterate them, compute both AUC (using the held-out labels for reference) and AUCp (label-free), and write a CSV with one row per epoch so you can plot the correlation yourself.

---

## When AUCp works — and when it doesn't

AUCp is reliable when **both** of the following hold (paper §V, Fig. 5):

1. The known-normal training set is **large and representative** of normal anatomy. If it's small or biased, AUCp drifts.
2. The unlabeled evaluation set contains a **non-trivial fraction of abnormals**. As `d → 0`, AUCp → 0.5 (random guessing).

Outside those conditions, fall back to standard label-based validation or use AUCp together with an [MPE](https://arxiv.org/abs/1604.05924) method (KM-MPE, AlphaMax, TIcE) feeding `estimate_auc_from_aucp(aucp, rho)`.

---

## Citation

```bibtex
@article{rahmansiddiquee2026aucp,
  title   = {AUCp: Pseudo-AUC for Inference Model Selection with Unlabeled Validation Data in Abnormality Detection},
  author  = {Rahman Siddiquee, Md Mahfuzur and Rafsani, Fazle and Shah, Jay and Wu, Teresa and Chong, Catherine D and Schwedt, Todd J and Li, Baoxin},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2026},
  doi     = {10.1109/TMI.2026.3684946},
  publisher = {IEEE}
}
```

## Acknowledgements

This work was supported by the U.S. Department of Defense (W81XWH-15-1-0286, W81XWH1910534), the National Institutes of Health (K23NS070891, 1R61NS113315-01), and Amgen ISS 20187183.

The experiment harness builds on [MedIAnomaly](https://github.com/caiyu6666/MedIAnomaly), [Brainomaly](https://arxiv.org/abs/2302.09200), [HealthyGAN](https://arxiv.org/abs/2202.08208), [PANDA](https://github.com/talreiss/PANDA), and [Mean-Shifted-AD](https://github.com/talreiss/Mean-Shifted-Anomaly-Detection). The base self-supervised pipelines extend [CutPaste](https://arxiv.org/abs/2104.04015), [FPI](https://arxiv.org/abs/2011.04197), [PII](https://arxiv.org/abs/2107.02788), and [NSA](https://arxiv.org/abs/2109.15222).

## License

[MIT](LICENSE). Pull requests welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
