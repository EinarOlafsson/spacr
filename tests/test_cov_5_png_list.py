"""A png_list row that cannot name an object is dropped, not mis-cut.

Every branch here decides which crop a row turns into. Getting the object id
wrong does not fail loudly -- it cuts a DIFFERENT cell out of the merged
array and files it under the first one's name, so the scores in the table
belong to an object nobody looked at. These tests pin the cases where the
answer must be "no label" rather than a guess: an id that is absent, an id
that is a float NaN, a database that is not on disk, and a frame that
carries no object column at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.png_list import (PNG_LIST_ID_COLUMNS, _merged_field_paths,
                            _object_id_int, crop_rows_from_png_list)


def test_a_missing_object_id_is_not_object_zero():
    """``None`` comes back as None, never coerced to a label.

    ``int(None)`` raises and ``float(None)`` is not reached, so the guard is
    the only thing between a null id and an exception mid-frame.
    """
    assert _object_id_int(None) is None


def test_a_nan_id_names_no_object_but_a_whole_float_does():
    """A float id is a label only when it is a real number.

    Crop ids arrive as floats whenever the column holds a null, because
    pandas widens the whole column. ``int(nan)`` is a garbage label
    (a platform-dependent huge integer), so NaN must return None while an
    ordinary float still resolves.
    """
    assert _object_id_int(float("nan")) is None
    assert _object_id_int(np.float64(12.0)) == 12


@pytest.mark.parametrize(
    "value",
    [12.5, np.float32(12.5), np.inf, -np.inf, True, np.bool_(True)],
)
def test_only_an_exact_finite_non_boolean_number_can_be_an_object_id(value):
    """Coercion must not turn a different or absent value into a label."""
    assert _object_id_int(value) is None
    assert _object_id_int(12.0) == 12


def test_an_unknown_object_type_is_refused_before_labels_are_mapped(tmp_path):
    """A typo cannot silently fall back to cell labels."""
    frame = pd.DataFrame({"cell_id": ["o4"], "path_name": ["field.npy"]})
    with pytest.raises(ValueError, match="object_type must be one of"):
        crop_rows_from_png_list(
            str(tmp_path / "measurements.db"), frame,
            object_type="nucleuz", verbose=False,
        )


def test_multiple_alternate_id_columns_are_ambiguous(tmp_path):
    """Dictionary order cannot decide which mask a requested crop belongs to."""
    frame = pd.DataFrame({
        "nucleus_id": ["o4"],
        "pathogen_id": ["o7"],
        "path_name": ["field.npy"],
    })
    with pytest.raises(ValueError, match="multiple alternate object ID"):
        crop_rows_from_png_list(
            str(tmp_path / "measurements.db"), frame,
            object_type="cell", verbose=False,
        )


def test_a_database_that_is_not_there_yields_no_field_paths():
    """A missing measurements.db gives an empty map, not a sqlite error.

    ``crop_rows_from_png_list`` calls this before it knows whether the
    database was ever written; sqlite would happily CREATE the file and then
    fail on the first SELECT, leaving an empty database behind.
    """
    assert _merged_field_paths("/nonexistent/never/made/measurements.db") == {}


def test_a_frame_with_no_object_column_keeps_no_rows(tmp_path):
    """A frame naming neither an id column nor object_label cuts nothing.

    The fallbacks must produce a null label and a null path rather than
    raising a KeyError, so a caller handed the wrong table gets an empty
    result it can report instead of a traceback halfway through a run.
    """
    df = pd.DataFrame({"png_path": ["a.png", "b.png"]})
    out = crop_rows_from_png_list(str(tmp_path / "absent.db"), df,
                                  object_type="cell", verbose=False)
    assert list(out.columns) == ["png_path", "path_name", "object_label",
                                 "object_type"]
    assert len(out) == 0


def test_every_crop_mode_names_its_own_id_column():
    """Each supported mode maps to ``<mode>_id`` and nothing else.

    The fallback loop in ``crop_rows_from_png_list`` walks these values, so a
    mode whose column were misspelled would silently borrow another object's
    ids.
    """
    for mode, column in PNG_LIST_ID_COLUMNS.items():
        assert column == f"{mode}_id"
