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

def test_a_stem_without_a_plate_component_is_refused():
    """Escaping the plate of a stem that has none would invent a plate."""
    with pytest.raises(schema.KeyParseError) as excinfo:
        schema.escape_field_stem_plate("r1_c1.npy")
    assert "r1_c1" in str(excinfo.value)


def test_a_timelapse_stem_needs_its_timepoint_too():
    """Three components cannot carry plate, well, field and time."""
    with pytest.raises(schema.KeyParseError):
        schema.escape_field_stem_plate("plate1_r1_c1", timelapse=True)


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

def test_a_missing_key_value_compares_as_blank():
    """NaN is not a well id, so it must not compare equal to the text 'nan'."""
    assert schema.comparable_key_value(float("nan")) == ""


def test_a_not_a_number_that_is_not_a_python_float_also_compares_as_blank():
    """A numpy float32 column and a CSV holding the text 'nan' arrive here
    too, and both mean the same thing: the row has no value."""
    assert schema.comparable_key_value(np.float32("nan")) == ""
    assert schema.comparable_key_value("NaN") == ""


def test_a_fractional_key_value_keeps_its_fraction():
    """1.5 is not 1; truncating it would merge two different rows."""
    assert schema.comparable_key_value(1.5) == "1.5"
    assert schema.comparable_key_value(2.0) == "2"


# -- object-table validation -------------------------------------------------

def test_validating_something_that_is_not_a_frame_is_refused():
    """The writer boundary is where a non-frame can still be seen."""
    with pytest.raises(schema.ObjectTableSchemaError) as excinfo:
        schema.validate_object_table_frame([{"object_label": 1}], "cell")
    assert "list" in str(excinfo.value)


def test_a_duplicated_column_label_is_refused_by_name():
    """pandas allows two columns named alike; to_sql does not."""
    frame = _cell_frame()
    doubled = pd.concat([frame, frame[["cell_area"]]], axis=1)
    with pytest.raises(schema.ObjectTableSchemaError) as excinfo:
        schema.validate_object_table_frame(doubled, "cell")
    assert "cell_area" in str(excinfo.value)


def test_a_missing_required_column_names_what_is_missing():
    """Without path_name the rows cannot be traced back to an image."""
    frame = _cell_frame().drop(columns=["path_name"])
    with pytest.raises(schema.ObjectTableSchemaError) as excinfo:
        schema.validate_object_table_frame(frame, "cell")
    assert "path_name" in str(excinfo.value)


def test_a_resolver_that_hands_back_an_incomplete_frame_is_still_refused(
        monkeypatch):
    """The writer boundary re-checks rather than trusting the resolver: a
    frame that reaches to_sql without prcf writes rows nothing can join."""
    from spacr import metadata_resolution

    incomplete = _cell_frame().drop(columns=["prcf"])

    class _Result:
        frame = incomplete

    monkeypatch.setattr(metadata_resolution, "resolve_metadata_columns",
                        lambda *a, **k: _Result())
    with pytest.raises(schema.ObjectTableSchemaError) as excinfo:
        schema.validate_object_table_frame(incomplete, "cell")
    message = str(excinfo.value)
    assert "prcf" in message
    assert "got [" in message


def test_a_timepoint_column_on_a_non_timelapse_table_is_refused():
    """Declaring non-timelapse and carrying timeID collapses the timepoints."""
    frame = _cell_frame(timeID=["t1", "t2"])
    with pytest.raises(schema.ObjectTableSchemaError) as excinfo:
        schema.validate_object_table_frame(frame, "cell", timelapse=False)
    assert "timeID" in str(excinfo.value)


def test_an_empty_well_id_is_refused_with_the_offending_rows():
    """A blank rowID groups with every other blank rowID in the database."""
    frame = _cell_frame(rowID=["r1", "   "])
    with pytest.raises(schema.ObjectTableSchemaError) as excinfo:
        schema.validate_object_table_frame(frame, "cell")
    message = str(excinfo.value)
    assert "rowID" in message
    assert "[1]" in message
