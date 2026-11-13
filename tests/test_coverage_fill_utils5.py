"""Coverage-fill batch 5 for spacr.utils stack-filter / regex helpers."""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")

from spacr import utils as U


def test_merge_touching_objects():
    m = np.zeros((16, 16), dtype=np.int32)
    m[4:12, 4:9] = 1
    m[4:12, 9:12] = 2   # adjacent to label 1
    out = U.merge_touching_objects(m.copy(), threshold=0.1)
    assert out is not None


# ---------------------------------------------------------------------------
# _object_filter
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _get_regex
# ---------------------------------------------------------------------------
