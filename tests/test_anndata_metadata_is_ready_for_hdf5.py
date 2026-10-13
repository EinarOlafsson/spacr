"""Item 288: AnnData export's metadata normalisation for HDF5, case by case.

``_hdf5_metadata`` prepares ``obs`` for AnnData's HDF5 string encoding
without changing any value:

* a categorical with ordinary (object) categories is left exactly as it is;
* one whose categories are the nullable string dtype has them turned into
  plain objects, with the same labels;
* a nullable string column becomes an object column;
* an object column that is entirely missing becomes an empty categorical,
  which HDF5 can store and a column of Nones is not;
* numbers are untouched, and the index becomes object.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spacr.anndata_export import _hdf5_metadata


def test_each_kind_of_metadata_column_is_normalised():
    frame = pd.DataFrame({
        "plain_cat": pd.Categorical(["a", "b", "a"]),
        "string_cat": pd.Categorical(
            pd.array(["x", "y", "x"], dtype="string")),
        "text": pd.array(["p", None, "q"], dtype="string"),
        "missing": pd.Series([None, None, None], dtype=object),
        "area": [1.0, 2.0, 3.0],
    }, index=pd.RangeIndex(3))
    _hdf5_metadata(frame)

    assert list(frame["plain_cat"]) == ["a", "b", "a"]
    assert frame["plain_cat"].cat.categories.dtype == object
    assert list(frame["string_cat"]) == ["x", "y", "x"]
    assert frame["string_cat"].cat.categories.dtype == object
    assert frame["text"].dtype == object
    assert frame["text"].iloc[0] == "p" and pd.isna(frame["text"].iloc[1])
    assert isinstance(frame["missing"].dtype, pd.CategoricalDtype)
    assert len(frame["missing"].cat.categories) == 0
    assert frame["missing"].isna().all()
    assert frame["area"].dtype == np.float64
    assert frame.index.dtype == object
