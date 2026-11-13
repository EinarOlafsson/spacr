"""Coverage-fill batch 3 for spacr.utils mask/segmentation-metric helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from spacr import utils as U


# ---------------------------------------------------------------------------
# circular mask / apply_mask
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# IoU / matching / AP
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# jaccard / dice / boundary
# ---------------------------------------------------------------------------

def test_jaccard_and_dice():
    a = np.zeros((8, 8), dtype=np.int32); a[1:5, 1:5] = 1
    b = np.zeros((8, 8), dtype=np.int32); b[1:5, 1:5] = 1
    assert U.jaccard_index(a, b) == 1.0
    assert U.dice_coefficient(a, b) == 1.0
    # both empty → dice defined as 1.0
    z = np.zeros((4, 4), dtype=np.int32)
    assert U.dice_coefficient(z, z) == 1.0


def test_extract_boundaries_and_bf1():
    m = np.zeros((16, 16), dtype=np.int32); m[4:12, 4:12] = 1
    b = U.extract_boundaries(m)
    assert b.any()
    f1 = U.boundary_f1_score(m, m)
    assert f1 > 0.9


# ---------------------------------------------------------------------------
# mask object helpers
# ---------------------------------------------------------------------------

def test_filter_object():
    m = np.zeros((16, 16), dtype=np.int32)
    m[0, 0] = 1              # 1 px → removed
    m[4:8, 4:8] = 2          # 16 px → kept
    out = U._filter_object(m.copy(), min_value=5)
    assert 1 not in np.unique(out) and 2 in np.unique(out)


# ---------------------------------------------------------------------------
# dataframe helpers
# ---------------------------------------------------------------------------

def test_generate_fraction_map():
    df = pd.DataFrame({
        "gene": ["g1", "g1", "g2", "g3", "g3", "g3"],
        "prc": ["p_r1_c1", "p_r1_c1", "p_r1_c1",
                "p_r1_c2", "p_r1_c2", "p_r1_c2"],
        "count": [10, 20, 30, 5, 5, 5],
        "well_read_sum": [60, 60, 60, 15, 15, 15],
    })
    out = U.generate_fraction_map(df, gene_column="gene", min_frequency=0.0)
    assert isinstance(out, pd.DataFrame)
    assert out.index.name == "prc"
