"""Joining cell to pathogen warned about nearly every row of a healthy screen.

    the same column arrived from two tables with different values for the
    same object:
      'object_label': 60095 of 60816 objects disagree between cell and
      pathogen (e.g. prcfo 0, 1, 2, 3, 4)

`object_label` was in :data:`spacr.merge_tables.MUST_AGREE`, whose members
are described as columns that "name WHICH object a row is, rather than
measuring it. Both tables read them off the same image, so they must match".
That is true of plateID and prcfo. It is not true of object_label: a cell's
object_label is its label in the CELL mask and a pathogen's is its label in
the PATHOGEN mask -- two separate labellings of two separate objects. There
is no reason for them to coincide and every reason for them not to.

So the warning fired on 99% of the objects in a correct screen, over data
that is exactly right, using the words "a defect in the data no analysis
should quietly average over". A warning that fires on the normal case
teaches its reader to ignore it, and takes the real conflicts with it.

Cytoplasm is why the column is not simply dropped from the check: it is the
cell minus its nucleus and carries the CELL's label, so a disagreement
between those two is a genuine mismatch and is still reported.
"""
from __future__ import annotations

import logging
import os
import sqlite3

import pandas as pd
import pytest

from spacr.merge_tables import SAME_LABEL_SPACE, reconcile_duplicates


PLATE1_DB = os.environ.get("SPACR_PLATE1_DB", "")


@pytest.fixture
def warnings_from(caplog):
    caplog.set_level(logging.WARNING, logger="spacr.merge_tables")
    return caplog


def _pair(suffix, right_values):
    return pd.DataFrame({
        "prcfo": [f"p_r1_c1_f1_o{i}" for i in range(5)],
        "object_label": [1, 2, 3, 4, 5],
        f"object_label{suffix}": list(right_values),
    })


def test_a_cell_and_a_pathogen_label_may_differ(warnings_from):
    frame = _pair("_pathogen", [11, 12, 13, 14, 15])

    out = reconcile_duplicates(frame.copy(), "_pathogen", key="prcfo",
                               left_name="cell", right_name="pathogen")

    assert "object_label" not in warnings_from.text
    # BOTH are kept: they are two different facts, not one fact twice.
    assert "object_label" in out.columns
    assert "object_label_pathogen" in out.columns


def test_a_cell_and_a_cytoplasm_label_may_not(warnings_from):
    """Same label space, so a mismatch there is a real one."""
    frame = _pair("_cytoplasm", [1, 2, 9, 4, 5])

    reconcile_duplicates(frame.copy(), "_cytoplasm", key="prcfo",
                         left_name="cell", right_name="cytoplasm")

    assert "object_label" in warnings_from.text
    assert "1 of 5" in warnings_from.text


def test_matching_cytoplasm_labels_collapse_to_one_column(warnings_from):
    frame = _pair("_cytoplasm", [1, 2, 3, 4, 5])

    out = reconcile_duplicates(frame.copy(), "_cytoplasm", key="prcfo",
                               left_name="cell", right_name="cytoplasm")

    assert "object_label_cytoplasm" not in out.columns
    assert not warnings_from.text.strip()


def test_the_label_space_names_the_tables_it_applies_to():
    """A guard that applied to everything is what produced the false alarm."""
    assert SAME_LABEL_SPACE["object_label"] == ("cell", "cytoplasm")


def test_plate_identity_still_has_to_agree(warnings_from):
    """Narrowing object_label may not narrow the columns that were right."""
    frame = pd.DataFrame({
        "prcfo": ["a", "b"],
        "plateID": ["plate1", "plate1"],
        "plateID_pathogen": ["plate1", "plate9"],
    })

    reconcile_duplicates(frame.copy(), "_pathogen", key="prcfo",
                         left_name="cell", right_name="pathogen")

    assert "plateID" in warnings_from.text


@pytest.mark.skipif(not PLATE1_DB or not os.path.exists(PLATE1_DB),
                    reason="set SPACR_PLATE1_DB to a real measurements.db")
def test_a_real_cell_pathogen_join_is_silent(warnings_from):
    """The screen this was reported from is a correct one."""
    from spacr.qt.widgets.measurement_scan_panel import merge_across_databases

    merged = merge_across_databases([PLATE1_DB], ["cell", "pathogen"])
    frame = merged[0] if isinstance(merged, tuple) else merged

    assert len(frame) > 0, "the join returned nothing, so it proved nothing"
    assert "object_label" not in warnings_from.text
