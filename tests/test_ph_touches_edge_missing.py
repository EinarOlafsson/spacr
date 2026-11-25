import numpy as np
import pandas as pd

from spacr.ops_engine import _ph_touches_edge


def test_missing_touches_field_edge_counts_as_touching():
    out = _ph_touches_edge(pd.Series([True, False, np.nan, None]))
    assert out.dtype == bool
    assert out.tolist() == [True, False, True, True]
    assert _ph_touches_edge(pd.Series([], dtype=object)).tolist() == []
