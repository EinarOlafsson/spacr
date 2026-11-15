"""The png_list helpers find the right field and the right id column.

A database that is not on disk gives no field paths rather than a sqlite
error, and every crop mode maps to its own ``<mode>_id`` column.
"""
from __future__ import annotations

from spacr.png_list import PNG_LIST_ID_COLUMNS, _merged_field_paths


def test_a_database_that_is_not_there_yields_no_field_paths():
    """A missing measurements.db gives an empty map, not a sqlite error.

    ``crop_rows_from_png_list`` calls this before it knows whether the
    database was ever written; sqlite would happily CREATE the file and then
    fail on the first SELECT, leaving an empty database behind.
    """
    assert _merged_field_paths("/nonexistent/never/made/measurements.db") == {}


def test_every_crop_mode_names_its_own_id_column():
    """Each supported mode maps to ``<mode>_id`` and nothing else.

    The fallback loop in ``crop_rows_from_png_list`` walks these values, so a
    mode whose column were misspelled would silently borrow another object's
    ids.
    """
    for mode, column in PNG_LIST_ID_COLUMNS.items():
        assert column == f"{mode}_id"
