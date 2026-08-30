"""Six last branches across control naming, model comparison and table merges.

Four of the six are loop arcs: the iteration that does NOT match. A loop whose
every fixture item matched has never proved it can pass one over, and in each
case here passing one over is the behaviour that keeps a report honest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# control_names.resolve_controls — arc 143 -> 145, a prefix supplied by hand
# ---------------------------------------------------------------------------

def test_a_supplied_prefix_is_used_instead_of_being_measured():
    """The ``if prefix is None:`` branch not taken.

    The prefix is measured from the library once and then handed down, which
    is the whole reason the parameter exists: measuring it again per call
    would be O(library) per control, and -- worse -- a caller that already
    knows the prefix could get a DIFFERENT answer from a shorter name list,
    silently resolving the same control two ways in one run.
    """
    from spacr.control_names import resolve_controls

    supplied = resolve_controls(["TGGT1_999999"], prefix="TGGT1_")
    assert supplied and supplied[0] is not None


def test_no_typed_controls_at_all_is_an_empty_result():
    """The early return above, so the prefix branch is reached deliberately."""
    from spacr.control_names import resolve_controls

    assert resolve_controls(None) == ()
    assert resolve_controls([]) == ()


# ---------------------------------------------------------------------------
# model_compare.ModelConfig — arcs 396 -> 395 and 412 -> 408, loops passing over
# ---------------------------------------------------------------------------

def test_only_the_ignored_extras_are_listed_as_ignored():
    """The ``if key in IGNORED_ARGUMENTS:`` loop arc that skips.

    ``extra`` carries both kinds at once in real use. Listing an honoured key
    among the ignored ones would tell the user their setting had no effect
    when it did -- and this report exists precisely so that two runs which
    look identical can be shown not to be.
    """
    from spacr.model_compare import (HONOURED_EVAL_ARGUMENTS,
                                     IGNORED_ARGUMENTS, ModelConfig)

    ignored_key = next(iter(IGNORED_ARGUMENTS))
    honoured_key = next(iter(HONOURED_EVAL_ARGUMENTS))

    config = ModelConfig(name="A", extra={ignored_key: 7, honoured_key: 3})
    listed = config.ignored_parameters()

    assert listed.get(ignored_key) == 7
    assert honoured_key not in listed


def test_an_honoured_extra_earns_no_note_at_all():
    """The ``elif key not in HONOURED_EVAL_ARGUMENTS:`` arc that skips.

    Three outcomes share one loop: ignored keys get an "is ignored" note,
    unknown keys get a "not a Cellpose 4 eval" note, and honoured keys get
    NOTHING. The silent third case is the one no fixture had produced, and a
    regression there would put every legitimate setting into the report as a
    warning.
    """
    from spacr.model_compare import (HONOURED_EVAL_ARGUMENTS,
                                     IGNORED_ARGUMENTS, ModelConfig)

    honoured_key = next(iter(HONOURED_EVAL_ARGUMENTS))
    ignored_key = next(iter(IGNORED_ARGUMENTS))

    notes = ModelConfig(name="A", extra={honoured_key: 3}).notes()
    assert not [n for n in notes if honoured_key in n]

    both = ModelConfig(name="A", extra={honoured_key: 3, ignored_key: 7,
                                        "not_a_real_argument": 1}).notes()
    assert any(ignored_key in n and "is ignored" in n for n in both)
    assert any("not_a_real_argument" in n for n in both)
    assert not [n for n in both if honoured_key in n]


# ---------------------------------------------------------------------------
# merge_tables._align_keys — arc 292 -> 286, two numeric key columns
# ---------------------------------------------------------------------------

def test_two_numeric_key_columns_are_left_as_numbers():
    """The ``elif not (both numeric):`` arc that skips the cast.

    The cast to string exists because a plate called ``1`` is read as an int
    from one table and a str from another, and the merge then matches nothing.
    When BOTH sides are already numeric there is no such disagreement, and
    casting anyway would turn 1.0 into "1.0" on one side and "1" on the other
    -- introducing exactly the failure the function exists to prevent.
    """
    from spacr.merge_tables import _align_keys

    left = pd.DataFrame({"plateID": [1, 2], "value": [10, 20]})
    right = pd.DataFrame({"plateID": [1, 2], "other": [30, 40]})

    _align_keys(left, right, ["plateID"])

    assert pd.api.types.is_numeric_dtype(left["plateID"])
    assert pd.api.types.is_numeric_dtype(right["plateID"])


def test_mixed_type_key_columns_are_cast_to_string():
    """The taken side, which is the failure the function was written for."""
    from spacr.merge_tables import _align_keys

    left = pd.DataFrame({"plateID": [1, 2]})
    right = pd.DataFrame({"plateID": ["1", "2"]})

    _align_keys(left, right, ["plateID"])

    assert left["plateID"].tolist() == right["plateID"].tolist()


def test_a_key_missing_from_one_side_is_skipped():
    """The ``continue`` above both, so neither cast is reached by accident."""
    from spacr.merge_tables import _align_keys

    left = pd.DataFrame({"plateID": [1]})
    right = pd.DataFrame({"other": [1]})

    _align_keys(left, right, ["plateID"])           # must not raise

    assert pd.api.types.is_numeric_dtype(left["plateID"])


# ---------------------------------------------------------------------------
# merge_tables._merge_crops — arc 664 -> 666, png_list with no path column
# ---------------------------------------------------------------------------

def test_crops_merge_by_object_even_when_no_path_column_is_present():
    """The ``if path_column:`` branch not taken.

    ``png_list`` is one row per crop, and a table written by a mode that
    recorded object ids but no path still identifies WHICH objects were
    cropped. Refusing to merge it, or inventing an empty path column, would
    lose that -- so the join happens and simply carries no path.
    """
    from spacr.merge_tables import OBJECT_COLUMN, _merge_crops
    from spacr.utils import PNG_OBJECT_ID_COLUMNS

    id_column = list(PNG_OBJECT_ID_COLUMNS.values())[0]
    merged = pd.DataFrame({"plateID": ["p1", "p1"],
                           OBJECT_COLUMN: ["1", "2"],
                           "area": [10.0, 20.0]})
    png = pd.DataFrame({"plateID": ["p1", "p1"], id_column: [1, 2]})

    out = _merge_crops(merged, png, ["plateID"])

    assert len(out) == 2
    assert not [c for c in out.columns if c.endswith("_path")]


def test_a_png_list_with_a_path_column_carries_it_through():
    """The taken side, so the absence above is visibly a decision."""
    from spacr.merge_tables import OBJECT_COLUMN, PNG_TABLE, _merge_crops
    from spacr.utils import PNG_OBJECT_ID_COLUMNS

    id_column = list(PNG_OBJECT_ID_COLUMNS.values())[0]
    merged = pd.DataFrame({"plateID": ["p1"], OBJECT_COLUMN: ["1"],
                           "area": [10.0]})
    png = pd.DataFrame({"plateID": ["p1"], id_column: [1],
                        "png_path": ["/crops/a.png"]})

    out = _merge_crops(merged, png, ["plateID"])

    assert out[f"{PNG_TABLE}_path"].tolist() == ["/crops/a.png"]


def test_a_png_list_with_no_object_id_column_merges_nothing():
    """The guard above both, which returns the table untouched."""
    from spacr.merge_tables import OBJECT_COLUMN, _merge_crops

    merged = pd.DataFrame({"plateID": ["p1"], OBJECT_COLUMN: ["1"]})
    png = pd.DataFrame({"plateID": ["p1"], "unrelated": [1]})

    out = _merge_crops(merged, png, ["plateID"])

    assert out.equals(merged)
