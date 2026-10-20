"""The invasion assay adds its per-parasite columns in one step.

Existing names are replaced where they stand, new names are appended in the
order given, and a call that only replaces returns the same frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spacr.submodules import _with_invasion_columns


def test_replacing_only_keeps_the_frame_and_its_column_order():
    df = pd.DataFrame({"a": [1, 2], "cell_id": [0.0, 3.0]}, index=[10, 11])
    out = _with_invasion_columns(df, {"cell_id": np.array([0, 3])})
    assert out is df
    assert list(out.columns) == ["a", "cell_id"]
    assert out["cell_id"].tolist() == [0, 3]


def test_new_columns_are_appended_in_order_aligned_with_the_index():
    df = pd.DataFrame({"a": [1, 2]}, index=[10, 11])
    out = _with_invasion_columns(df, {"a": [5, 6], "z": pd.Series([7, 8], index=[10, 11]),
                                      "b": np.array([True, False])})
    assert list(out.columns) == ["a", "z", "b"]
    assert out.loc[11].tolist() == [6, 8, False]
    assert out.index.tolist() == [10, 11]
