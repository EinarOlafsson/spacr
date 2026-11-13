"""The static helpers of ``spacr.submodules.CellposeLazyDataset``.

The dataset itself is tested in ``tests/test_submodules_cellpose.py``.
"""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")

from spacr import submodules as S


# ---------------------------------------------------------------------------
# CellposeLazyDataset
# ---------------------------------------------------------------------------

def test_lazy_dataset_static_helpers():
    rgb = np.zeros((4, 4, 3), dtype=np.float32); rgb[..., 0] = 3.0
    gray = S.CellposeLazyDataset._to_grayscale(rgb)
    assert gray.shape == (4, 4)
    # already 2-D → unchanged
    assert S.CellposeLazyDataset._to_grayscale(gray).shape == (4, 4)
    scaled = S.CellposeLazyDataset._scale_to_unit_interval(
        np.array([[0, 255]], dtype=np.float32))
    assert scaled.max() <= 1.0
    # max<=1 → returned unchanged
    small = np.array([[0.0, 0.5]], dtype=np.float32)
    assert S.CellposeLazyDataset._scale_to_unit_interval(small).max() == 0.5
