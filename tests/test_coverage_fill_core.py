"""Coverage-fill for spacr.core cheap validation branches (no GPU)."""
from __future__ import annotations

import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import core as C


def test_preprocess_generate_masks_bad_src_type():
    with pytest.raises(ValueError):
        C.preprocess_generate_masks({"src": 12345})


def test_preprocess_generate_masks_missing_src():
    with pytest.raises(ValueError):
        C.preprocess_generate_masks({})
