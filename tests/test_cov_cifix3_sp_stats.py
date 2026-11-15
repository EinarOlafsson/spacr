"""Coverage for the per-cell export edge cases and the feature drop list reasons of the profiling statistics."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from spacr import sp_stats as s


def _keys(prcf, labels, time=None):
    plate, row, column, field = prcf.split("_")
    keys = {"plateID": plate, "rowID": row, "columnID": column,
            "fieldID": field, "prcf": prcf, "object_label": labels}
    if time is not None:
        keys["timeID"] = time
    return keys


def test_only_numeric_measured_columns_are_features():
    """Metadata, provenance, text and boolean columns are skipped; a bare name is refused."""
    frame = pd.DataFrame({
        "Metadata_x": [1.0, 2.0], "plateID": ["p", "p"],
        "cell_label": ["a", "b"], "cell_flag": [True, False],
        "cell_area": [1.0, 2.0]})
    assert s._per_cell_features(frame, "cell") == ["cell_area"]
    with pytest.raises(ValueError, match="without the cell_ prefix"):
        s._per_cell_features(pd.DataFrame({"area": [1.0]}), "cell")


def test_field_ids_that_are_not_numbers_stay_as_they_are():
    """A field id that does not parse keeps every id unchanged."""
    fields = pd.Series(["f1", "fx"])
    assert s._per_cell_site(fields).tolist() == ["f1", "fx"]
    assert s._per_cell_site(pd.Series(["f1", "f2"])).tolist() == [1, 2]


def test_a_pair_without_a_well_name_gets_a_missing_value():
    """A row and column that name no well give None for that cell only."""
    names = s._per_cell_wells(pd.Series(["r2", "r0"]), pd.Series(["c3", "c0"]))
    assert names.tolist() == ["B03", None]


def test_the_drop_list_gives_every_rule_a_reason_even_without_values():
    """Each rule explains itself, with and without the feature present in the normalised table."""
    normalized = pd.DataFrame({
        "a": [1.0, np.nan, np.nan, 2.0], "b": [np.nan] * 4,
        "c": [1.0, 2.0, 3.0, 4.0]})
    excluded = {
        "constant_in_controls": ["c"],
        "drop_na_columns": ["a", "ghost"],
        "drop_outliers": ["c", "b"],
        "variance_threshold": ["c", "b"],
        "frequency_threshold": ["c"],
        "correlation": ["a"]}
    result = SimpleNamespace(options={}, normalized=normalized, excluded=excluded)
    drops = s._feature_drop_list(result)
    reasons = {(row.rule, row.feature): row.reason for row in drops.itertuples()}
    assert reasons[("constant_in_controls", "c")].startswith("no variation")
    assert "non-finite in 50.0%" in reasons[("drop_na_columns", "a")]
    assert reasons[("drop_na_columns", "ghost")] == "non-finite"
    assert "exceeds 500" in reasons[("drop_outliers", "c")]
    assert "nan" in reasons[("drop_outliers", "b")]
    assert "variance 1.25" in reasons[("variance_threshold", "c")]
    assert reasons[("variance_threshold", "b")] == "variance undefined"
    assert reasons[("frequency_threshold", "c")].startswith("near-constant")
    assert "correlated above 0.9" in reasons[("correlation", "a")]


def test_no_features_are_constant_in_the_reference():
    """An empty feature list has nothing constant."""
    frame = pd.DataFrame({"plateID": ["p"], "x": [1.0]})
    assert s._constant_in_reference(frame, []) == []
