"""The canonical schema refuses the shapes that would corrupt a database.

Everything here is a boundary a writer or a model-input selector crosses. A
frame handed over as something other than a table, a required identity column
that is absent, a duplicated column label that ``to_sql`` would reject, a
timelapse column on a table declared non-timelapse, an empty well id that
would group every row together -- each has to be refused where it is
detectable, because past that point the wrong number is already stored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import schema


def _cell_frame(**extra) -> pd.DataFrame:
    data = {
        "object_label": [1, 2],
        "plateID": ["plate1", "plate1"],
        "rowID": ["r1", "r1"],
        "columnID": ["c1", "c1"],
        "fieldID": ["f1", "f1"],
        "prcf": ["plate1_r1_c1_f1", "plate1_r1_c1_f1"],
        "file_name": ["plate1_A01_1", "plate1_A01_1"],
        "path_name": ["/data/plate1_A01_1.npy"] * 2,
        "cell_area": [10.0, 20.0],
    }
    data.update(extra)
    return pd.DataFrame(data)


# -- the vocabulary itself ---------------------------------------------------

def test_two_canonical_names_cannot_claim_one_folded_spelling(monkeypatch):
    """A spelling meaning two things would rename a column at random."""
    monkeypatch.setattr(
        schema, "LEGACY_COLUMN_NAMES",
        {"plate_id": schema.PLATE_KEY, "PlateID": schema.ROW_KEY})
    with pytest.raises(RuntimeError) as excinfo:
        schema._build_folded_aliases()
    assert "plateid" in str(excinfo.value)


def test_the_shipped_vocabulary_has_no_contradiction():
    """The guard must be green against the aliases spaCR actually ships."""
    folded = schema._build_folded_aliases()
    assert folded[schema.fold_column_name("Plate_ID")] == schema.PLATE_KEY


# -- screen ids --------------------------------------------------------------

def test_a_missing_screen_label_becomes_the_default_screen():
    """NaN is what pandas puts in a column a source never filled in."""
    assert schema.screen_id(float("nan")) == schema.DEFAULT_SCREEN


def test_a_frame_with_blank_screen_labels_is_filled_in():
    """A blank screen id is not an identity: every row would group as one."""
    frame = pd.DataFrame({schema.SCREEN_KEY: [np.nan, "tsg101", ""]})
    out = schema.add_screen_column(frame, "rescue")
    assert list(out[schema.SCREEN_KEY]) == ["rescue", "tsg101", "rescue"]


# -- stems -------------------------------------------------------------------


# -- table contracts ---------------------------------------------------------

def test_a_contract_validates_a_frame_through_itself():
    """The contract object is the handle a writer already holds."""
    contract = schema.object_table_schema("cell")
    validated = contract.validate(_cell_frame())
    pd.testing.assert_frame_equal(validated, _cell_frame())


def test_a_field_provenance_table_is_keyed_by_field_not_by_object():
    """Rescale factors are per field; an object key would multiply them."""
    assert schema.table_key_columns("intensity_rescale") == \
        schema.FIELD_KEY_COLUMNS


def test_a_parent_link_column_is_provenance_not_a_measurement():
    """A link between compartments is identity and must never be modelled."""
    assert schema.is_provenance_column("nucleus_pathogen") is True


# -- model inputs ------------------------------------------------------------

def test_coercing_features_needs_a_frame_not_a_series():
    """A Series has no columns to select features from."""
    with pytest.raises(schema.ModelFeatureSchemaError) as excinfo:
        schema.coerce_model_feature_types(pd.Series([1.0, 2.0]))
    assert "Series" in str(excinfo.value)


def test_selecting_features_needs_a_frame_not_a_dict():
    """The refusal names the type so the caller can see what they passed."""
    with pytest.raises(schema.ModelFeatureSchemaError) as excinfo:
        schema.model_feature_columns({"cell_area": [1.0]})
    assert "dict" in str(excinfo.value)


def test_a_boolean_column_is_not_a_model_input():
    """A flag is a category; numeric selectors have never included bools."""
    frame = pd.DataFrame({"cell_area": [1.0, 2.0],
                          "cell_flag": [True, False]})
    chosen = schema.model_feature_columns(frame,
                                          extra_features=("cell_flag",))
    assert chosen == ["cell_area"]


# -- collision reporting -----------------------------------------------------

def test_disagreeing_metadata_columns_are_reported_to_the_given_sink():
    """The caller supplied a sink, so the message must go there, not to
    the warnings machinery where a test run would swallow it."""
    frame = pd.DataFrame({"rowID": ["r1", "r2"], "row_name": ["r1", "r9"]})
    captured = []
    out, collisions = schema.resolve_metadata_collisions(
        frame, warn=captured.append)
    assert len(captured) == 1, captured
    assert "rowID" in captured[0] and "row_name" in captured[0]
    assert [c.agreed for c in collisions] == [False]
    assert list(out.columns) == ["rowID"]


# -- comparable key values ---------------------------------------------------


# -- object-table validation -------------------------------------------------


