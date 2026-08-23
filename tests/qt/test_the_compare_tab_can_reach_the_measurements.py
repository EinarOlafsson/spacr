"""Two defects that between them emptied the Compare tab and froze the window.

Reported as "still dont see the measurement database columns in measurement
in the cell tab under compare" and "pressing join the measurements table in
the cell tab in the regression module makes spacr unresponsive".

They are one button. It reads every object table out of every attached
database and joins them onto the crop rows.

WHY NOTHING ARRIVED. `object_identity` builds a row's identity from
``prcfo``, or from ``prcf`` plus an object label. ``png_list`` -- the crop
table the panel starts from -- carries neither: it has plateID, rowID,
columnID and fieldID, which are the four columns ``prcf`` is nothing but a
paste of, and it spells its object ``cell_id = 'o2'`` where the object
tables spell it ``object_label = 2``. So the join refused every row with
"these object rows carry no object identity", and every morphological
measurement in the screen was unreachable from the panel that exists to
compare them. Both halves are handled now: the field key is built from the
four columns, and the label is translated out of the prcfo spelling.

WHY THE WINDOW FROZE. The join ran on the GUI thread. Measured at 3.2 s for
one plate's 553 objects, so a four-plate screen is minutes of a dead window.
It is a JobRunner job now, like every other long run in spaCR.

A third thing turned up while checking the result: the reconciler counted an
ABSENT value as a disagreement, so 172 uninfected cells -- kept on purpose,
with no pathogen row and therefore no pathogen columns -- were reported as
172 objects disagreeing about their own plateID.
"""
from __future__ import annotations

import os
import sqlite3
import time

import pandas as pd
import pytest

from spacr.gene_measurement_compare import object_identity


PLATE1_DB = os.environ.get("SPACR_PLATE1_DB", "")


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

def test_the_four_identity_columns_are_enough():
    """`prcf` is a paste of them, and png_list carries them without the paste."""
    frame = pd.DataFrame({
        "plateID": ["plate1"], "rowID": ["r5"],
        "columnID": ["c1"], "fieldID": ["f17"], "cell_id": ["o2"],
    })

    identity = object_identity(frame)

    assert identity is not None, "the crop table was refused an identity"
    assert identity.iloc[0] == "plate1_r5_c1_f17_2"


def test_the_prcfo_spelling_of_the_label_is_translated():
    """'o2' and 2 are the same object and must produce the same identity."""
    crop = pd.DataFrame({"plateID": ["plate1"], "rowID": ["r5"],
                         "columnID": ["c1"], "fieldID": ["f17"],
                         "cell_id": ["o2"]})
    measured = pd.DataFrame({"prcf": ["plate1_r5_c1_f17"],
                             "object_label": [2]})

    assert object_identity(crop).iloc[0] == object_identity(measured).iloc[0]


def test_an_existing_prcf_is_still_preferred():
    """The new branch is a fallback, not a replacement."""
    frame = pd.DataFrame({
        "plateID": ["ignored"], "rowID": ["ignored"],
        "columnID": ["ignored"], "fieldID": ["ignored"],
        "prcf": ["plate1_r5_c1_f17"], "object_label": [2],
    })

    assert object_identity(frame).iloc[0] == "plate1_r5_c1_f17_2"


def test_rows_with_no_label_still_have_no_identity():
    frame = pd.DataFrame({"plateID": ["p"], "rowID": ["r1"],
                          "columnID": ["c1"], "fieldID": ["f1"]})
    assert object_identity(frame) is None


# ---------------------------------------------------------------------------
# an absence is not a disagreement
# ---------------------------------------------------------------------------

def test_an_uninfected_cell_does_not_disagree_with_its_missing_pathogen(caplog):
    import logging

    from spacr.merge_tables import reconcile_duplicates

    caplog.set_level(logging.WARNING, logger="spacr.merge_tables")
    frame = pd.DataFrame({
        "prcfo": ["a", "b", "c"],
        "plateID": ["plate1", "plate1", "plate1"],
        # The third cell has no pathogen row, so every pathogen column is NA.
        "plateID_pathogen": ["plate1", "plate1", None],
    })

    out = reconcile_duplicates(frame.copy(), "_pathogen", key="prcfo",
                               left_name="cell", right_name="pathogen")

    assert "plateID" not in caplog.text
    assert "plateID_pathogen" not in out.columns   # agreed, so collapsed


def test_two_present_values_that_differ_still_disagree(caplog):
    """Ignoring absences may not become ignoring conflicts."""
    import logging

    from spacr.merge_tables import reconcile_duplicates

    caplog.set_level(logging.WARNING, logger="spacr.merge_tables")
    frame = pd.DataFrame({
        "prcfo": ["a", "b"],
        "plateID": ["plate1", "plate1"],
        "plateID_pathogen": ["plate1", "plate9"],
    })

    reconcile_duplicates(frame.copy(), "_pathogen", key="prcfo",
                         left_name="cell", right_name="pathogen")

    assert "plateID" in caplog.text


# ---------------------------------------------------------------------------
# the button, on real data
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PLATE1_DB or not os.path.exists(PLATE1_DB),
                    reason="set SPACR_PLATE1_DB to a real measurements.db")
def test_the_join_returns_at_once_and_fills_the_panel(qapp):
    """The click must not block, and what arrives must be the measurements."""
    from spacr.qt.widgets.measurement_compare_dialog import (
        MeasurementComparePanel)

    with sqlite3.connect(PLATE1_DB) as db:
        objects = pd.read_sql(
            "SELECT plateID, rowID, columnID, fieldID, cell_id, png_path "
            "FROM png_list", db)

    panel = MeasurementComparePanel(objects, {}, databases=[PLATE1_DB])
    try:
        before = panel._objects.shape[1]

        started = time.time()
        panel.join_the_tables()
        # THE CLICK ITSELF. Whatever the join costs, the handler returns.
        assert time.time() - started < 0.5

        deadline = time.time() + 120
        while panel._joining and time.time() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert not panel._joining, "the join never finished"

        after = panel._objects.shape[1]
        assert after > before + 100, (
            f"the join added {after - before} columns; the object tables "
            f"hold hundreds")
        measured = [c for c in panel._objects.columns
                    if c.startswith(("cell_", "nucleus_", "pathogen_",
                                     "cytoplasm_"))]
        assert len(measured) > 100
        assert "cell_area" in panel._objects.columns
    finally:
        panel.deleteLater()
        qapp.processEvents()
