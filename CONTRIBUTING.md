# Contributing to AUCp

Thanks for your interest in making this project better. The most useful kinds of contributions, roughly in order:

1. **Bug reports** that include a minimal reproducer.
2. **New benchmarks** — wiring AUCp into another abnormality-detection codebase and showing the gain (or loss) over `last`-checkpoint selection.
3. **Documentation improvements**, especially clarifications to the math or the assumptions.
4. **New mixture-proportion estimators** plugged into `estimate_auc_from_aucp`.

## Quick development setup

```bash
git clone https://github.com/mahfuzmohammad/AUCp
cd AUCp
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python examples/synthetic_demo.py   # sanity check
```

The standalone `aucp/` package depends only on NumPy and scikit-learn. The two experiment subtrees (`reconstruction/`, `ssl/`) additionally require PyTorch and dataset-specific packages — install only what you need (see `requirements.txt`).

## Tests

We don't yet ship a formal test suite, but `examples/synthetic_demo.py` is a regression check for the public API. New behavior should add a section to that demo (or a Jupyter cell in `examples/aucp_walkthrough.ipynb`) that exercises it.

## Style

- Public-facing helpers go in `aucp/`. Keep its dependency surface tiny — NumPy + scikit-learn only.
- Cluster-specific paths belong behind the `AUCP_DATA_ROOT` / `AUCP_OUTPUT_ROOT` env vars defined in `aucp/paths.py`. Don't hardcode absolute paths.
- Match the existing formatting in the file you're editing rather than reformatting the whole repo.

## Pull requests

- One concern per PR.
- Update the README / notebook if you change the public API.
- Squash to a single descriptive commit before merge.

## Questions / proposals

Open a GitHub issue. For research-direction questions (new datasets, new selection heuristics, alternatives to `estimate_auc_from_aucp`) tag it `discussion` so we can scope before code lands.
