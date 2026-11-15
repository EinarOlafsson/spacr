"""The row matcher's answer for a frame with no rows.

The key builders' blank-type, empty-frame and missing-column cases live in
``tests/test_selection_object_keys.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spacr import selection


def _frame(**extra) -> pd.DataFrame:
    data = {"plateID": ["p1", "p1"], "rowID": ["r1", "r2"],
            "columnID": ["c1", "c1"], "fieldID": ["f1", "f1"],
            "object_label": [1, 2]}
    data.update(extra)
    return pd.DataFrame(data)


def test_matching_an_empty_frame_is_an_empty_mask():
    """The row matcher's own answer for no rows is a zero-length mask."""
    mask = selection._match_frame(_frame().iloc[:0], ["p1_r1_c1_f1_1"])
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == (0,)
