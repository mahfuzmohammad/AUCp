"""Path resolution for datasets and experiment outputs.

Both the reconstruction and self-supervised training scripts used to ship with
absolute paths that pointed at one author's filesystem. This module replaces
those with two environment variables that have sensible defaults:

- ``AUCP_DATA_ROOT``    — root of the MedIAnomaly-Data download.
                          Default: ``~/MedIAnomaly-Data``
- ``AUCP_OUTPUT_ROOT``  — where experiment artifacts (checkpoints, CSVs, logs)
                          are written. Default: ``./output`` inside the repo.

You can also pass an explicit path to ``data_root()`` / ``output_root()`` to
override both the environment variable and the default — useful from tests.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, os.PathLike]


_DATASET_DIRS = {
    "rsna": "RSNA",
    "vin": "VinCXR",
    "brain": "BrainTumor",
    "lag": "LAG",
    "brats": "BraTS2021",
    "c16": "Camelyon16",
    "isic": "ISIC2018_Task3",
}


def data_root(override: Optional[PathLike] = None) -> Path:
    """Return the root directory of MedIAnomaly-Data."""
    if override is not None:
        return Path(override).expanduser()
    env = os.environ.get("AUCP_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    return Path.home() / "MedIAnomaly-Data"


def output_root(override: Optional[PathLike] = None) -> Path:
    """Return the root directory where experiment artifacts are written."""
    if override is not None:
        return Path(override).expanduser()
    env = os.environ.get("AUCP_OUTPUT_ROOT")
    if env:
        return Path(env).expanduser()
    # Default sits next to the repo, not deep in the user's home, so a checkout
    # is self-contained.
    return Path.cwd() / "output"


def dataset_path(dataset: str, override: Optional[PathLike] = None) -> Path:
    """Resolve the on-disk directory for a named dataset.

    ``dataset`` is the short slug the training scripts use (``rsna``, ``vin``,
    ``brain``, ``lag``, ``brats``, ``c16``, ``isic``).
    """
    key = dataset.lower()
    if key not in _DATASET_DIRS:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Known: {sorted(_DATASET_DIRS)}"
        )
    return data_root(override) / _DATASET_DIRS[key]
