"""``_check_integrity`` deduplicates measurement columns before they hit the DB.

Two things were wrong with the old one-liner:

  df.columns = [col + f'_{i}' if df.columns.tolist().count(col) > 1 and i != 0
                else col for i, col in enumerate(df.columns)]

  * ``i`` is the column's position in the WHOLE frame, not the occurrence index
    of the duplicate. A second ``mean_intensity`` sitting at position 57 became
    ``mean_intensity_57`` — a name indistinguishable from a genuinely
    parameterised feature such as ``homogeneity_distance_8``, and one that moved
    whenever an unrelated column was added upstream.
  * ``and i != 0`` meant the first occurrence kept its bare name only when it
    happened to sit at index 0, even though ``object_label`` is read out of the
    first label column.

It also re-scanned the full column list once per column, which is O(n^2) on the
~1000-column measurement frames this runs over, twice per field per object type.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import utils as U


def _frame(columns, n_rows=2):
    data = np.arange(len(columns) * n_rows).reshape(n_rows, len(columns))
    df = pd.DataFrame(data)
    df.columns = columns
    return df


def _labelled(columns, n_rows=2):
    """A frame shaped like a real measurement frame: it always has a label."""
    return _frame(list(columns) + ["label"], n_rows=n_rows)


# ---------------------------------------------------------------------------
# The duplicate suffix
# ---------------------------------------------------------------------------

def test_first_occurrence_keeps_its_bare_name():
    """object_label is read from the first label column, so it must not move."""
    out = U._check_integrity(_frame(["label", "value", "label"]))
    # both label columns are consumed into label_list and dropped
    assert "label" not in out.columns
    assert out["object_label"].tolist() == [0, 3]   # column 0 of each row


def test_repeat_is_suffixed_with_its_occurrence_index_not_its_position():
    df = _labelled(["a", "b", "c", "d", "e", "dup", "f", "g", "dup"])
    out = U._check_integrity(df)
    # the repeat sits at position 8, but it is occurrence 1
    assert f"dup{U.DUPLICATE_COLUMN_SUFFIX}1" in out.columns
    assert "dup_8" not in out.columns
    assert "dup" in out.columns


def test_the_suffix_cannot_be_mistaken_for_a_parameterised_feature():
    """`mean_intensity_57` is a plausible real feature name. `__dup1` is not."""
    out = U._check_integrity(_labelled(["mean_intensity"] * 2 + ["x"] * 55
                                       + ["mean_intensity"]))
    suffixed = [c for c in out.columns if c.startswith("mean_intensity")]
    assert all(c == "mean_intensity" or U.DUPLICATE_COLUMN_SUFFIX in c
               for c in suffixed), suffixed
    assert not any(c.rsplit("_", 1)[-1].isdigit() and
                   U.DUPLICATE_COLUMN_SUFFIX not in c for c in suffixed)


def test_three_copies_get_distinct_names():
    out = U._check_integrity(_labelled(["v", "v", "v", "keep"]))
    assert "v" in out.columns
    assert f"v{U.DUPLICATE_COLUMN_SUFFIX}1" in out.columns
    assert f"v{U.DUPLICATE_COLUMN_SUFFIX}2" in out.columns
    assert len(set(out.columns)) == len(out.columns)


def test_a_duplicate_whose_first_copy_is_not_at_index_zero():
    """The old `and i != 0` guard renamed the first copy in exactly this case."""
    out = U._check_integrity(_labelled(["other", "dup", "dup"]))
    assert "dup" in out.columns
    assert "dup_1" not in out.columns


# ---------------------------------------------------------------------------
# A frame with no label column
# ---------------------------------------------------------------------------

def test_a_frame_with_no_label_column_says_so(capsys):
    """It used to die on `IndexError: list index out of range` from x[0], with
    nothing to indicate which frame or what was missing."""
    with pytest.raises(ValueError) as exc:
        U._check_integrity(_frame(["cell_area", "value"]))
    msg = str(exc.value)
    assert "label" in msg
    assert "cell_area" in msg          # names the columns it did see


def test_an_empty_frame_is_not_an_error():
    """_merge_and_save_to_database calls this BEFORE its len() > 0 guard."""
    out = U._check_integrity(pd.DataFrame(columns=["cell_area"]))
    assert len(out) == 0


def test_a_completely_empty_frame_is_not_an_error():
    out = U._check_integrity(pd.DataFrame())
    assert len(out) == 0


def test_unique_columns_are_left_completely_alone():
    cols = ["cell_area", "cell_channel_0_mean_intensity", "object_label_x"]
    out = U._check_integrity(_frame(cols))
    # object_label_x contains 'label', so it is collapsed; the others survive
    assert "cell_area" in out.columns
    assert "cell_channel_0_mean_intensity" in out.columns
    assert not any(U.DUPLICATE_COLUMN_SUFFIX in c for c in out.columns)


def test_substring_matches_are_not_treated_as_duplicates():
    """`.count(col)` compares whole names: `label` and `cell_label` differ."""
    df = _frame(["label", "cell_label", "value"])
    out = U._check_integrity(df)
    # both are label columns, both collapse, neither is suffixed as a duplicate
    assert not any(U.DUPLICATE_COLUMN_SUFFIX in c for c in out.columns)


# ---------------------------------------------------------------------------
# The label collapse itself must be unchanged
# ---------------------------------------------------------------------------

def test_label_columns_collapse_into_label_list_and_object_label():
    out = U._check_integrity(_frame(["label", "cell_label", "value"]))
    assert "object_label" in out.columns
    assert "label_list" in out.columns
    assert "label" not in out.columns and "cell_label" not in out.columns
    assert "value" in out.columns


def test_object_label_is_the_first_label_column_in_column_order():
    df = _frame(["cell_label", "label", "value"])
    out = U._check_integrity(df)
    # column 0 is cell_label -> values 0 and 3
    assert out["object_label"].tolist() == [0, 3]


def test_label_list_is_stringified():
    out = U._check_integrity(_frame(["label", "cell_label", "v"]))
    assert out["label_list"].map(type).eq(str).all()


def test_duplicate_label_columns_both_reach_label_list():
    out = U._check_integrity(_frame(["label", "label", "v"]))
    assert out.loc[0, "label_list"] == str([0, 1])


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

def test_column_list_is_not_rescanned_once_per_column():
    """The old form called df.columns.tolist() inside the comprehension, i.e.
    once per column. On a real ~1000-column frame that is a million operations
    per call, and it runs twice per field per object type."""
    cols = [f"f{i}" for i in range(400)] + ["dup", "dup", "label"]
    df = _frame(cols, n_rows=1)

    calls = {"n": 0}
    real_tolist = pd.Index.tolist

    def counting_tolist(self):
        calls["n"] += 1
        return real_tolist(self)

    pd.Index.tolist = counting_tolist
    try:
        U._check_integrity(df)
    finally:
        pd.Index.tolist = real_tolist

    assert calls["n"] < 10, (
        f"tolist() called {calls['n']} times for {len(cols)} columns — "
        "the per-column rescan is back")


# ---------------------------------------------------------------------------
# The feature dictionary must recognise what this writes
# ---------------------------------------------------------------------------

def test_feature_dict_recognises_the_current_duplicate_suffix():
    """A column this function emits must not land in family='unknown' — that is
    exactly the gap the feature dictionary exists to close."""
    from spacr.feature_dict import parse_column

    entry = parse_column(
        f"cell_channel_0_mean_intensity{U.DUPLICATE_COLUMN_SUFFIX}1")
    assert entry.family == "intensity"
    assert entry.object_type == "cell"
    assert entry.channel == 0
    assert "duplicated" in (entry.notes or "")


def test_feature_dict_still_recognises_the_legacy_positional_suffix():
    """Databases written before the change carry `_<position>`; reading an old
    measurements.db must keep working."""
    from spacr.feature_dict import parse_column

    entry = parse_column("cell_channel_0_mean_intensity_57")
    assert entry.family == "intensity"
    assert "_check_integrity" in (entry.notes or "")
