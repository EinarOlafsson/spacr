"""Compare panel — the wells dialog, the join, and everything that fails.

The panel's job is to make one claim about one measurement, and every path
here is one the panel has to survive without a traceback in front of the
user:

* a table with no measurement in it at all, and one whose columns cannot be
  classified -- both end in a sentence in the report box, not an empty
  window;
* the well chooser, including the case where the object rows do not say
  which well they came from;
* the threaded join: the guard against a second press, the worker's own
  failure, the completion handler's, and the dependent-variable table that
  is asked for and is not there;
* saving, both when the folder dialog is cancelled and when the write
  itself refuses.

Offscreen, no modal dialogs, no sleeps: the JobRunner is driven to
completion through its own callback rather than by waiting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from PySide6.QtWidgets import QDialog

from spacr.qt.widgets import measurement_compare_dialog as mcd
from spacr.qt.widgets.measurement_compare_dialog import (
    MeasurementCompareDialog, MeasurementComparePanel,
)


@pytest.fixture
def objects():
    """200 cells over two plates, with the four identity columns."""
    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame({
        "cell_area": rng.normal(10.0, 2.0, n),
        "nucleus_area": rng.normal(5.0, 1.0, n),
        "plateID": np.repeat(["p1", "p2"], n // 2),
        "rowID": rng.choice(["r1", "r2"], n),
        "columnID": rng.choice(["c1", "c2", "c3"], n),
        "fieldID": rng.choice(["f1", "f2"], n),
        "object_label": np.arange(n),
    })
    frame.loc[frame.index[:50], "cell_area"] += 3.0
    return frame


@pytest.fixture
def groups(objects):
    return {"TGGT1_220950": objects.index[:50]}


@pytest.fixture
def panel(qtbot, objects, groups):
    widget = MeasurementComparePanel(objects, groups)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# Nothing to compare
# ---------------------------------------------------------------------------

def test_columns_that_cannot_be_classified_leave_the_list_empty(panel,
                                                                monkeypatch):
    """A broken classifier is a shorter list, never a dead panel."""
    from spacr import gene_measurement_sweep

    def explode(_name):
        raise RuntimeError("the measurement vocabulary could not be read")

    monkeypatch.setattr(gene_measurement_sweep, "is_measurement", explode)

    assert panel._numeric_columns() == []
    panel._offer_second()
    assert panel.second.count() == 0


def test_a_view_change_before_the_first_build_draws_nothing(qtbot, objects):
    widget = MeasurementComparePanel(objects, {})
    qtbot.addWidget(widget)
    widget._comparison = None
    widget._draw_and_report()
    assert widget.comparison() is None


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def test_control_wells_that_cannot_be_resolved_are_no_wells(panel,
                                                            monkeypatch):
    panel._counts = pd.DataFrame({"prc": ["p1_r1_c1"], "grna": ["g1"],
                                  "count": [10]})
    panel.controls.setText("TGGT1_000001")

    def explode(_counts, _typed):
        raise RuntimeError("the count table has no guide column")

    monkeypatch.setattr(mcd, "control_wells", explode)
    assert panel._control_wells() == ()


def test_no_typed_control_asks_the_count_table_nothing(panel):
    panel._counts = pd.DataFrame({"prc": ["p1_r1_c1"]})
    panel.controls.setText("   ")
    assert panel._typed_controls() == []
    assert panel._control_wells() == ()


def test_controls_are_split_on_commas_and_trimmed(panel):
    panel.controls.setText(" TGGT1_000001 , , TGGT1_000002 ")
    assert panel._typed_controls() == ["TGGT1_000001", "TGGT1_000002"]


# ---------------------------------------------------------------------------
# The well chooser
# ---------------------------------------------------------------------------

def test_object_rows_with_no_well_offer_nothing_to_choose_between(qtbot):
    frame = pd.DataFrame({"cell_area": [1.0, 2.0, 3.0, 4.0]})
    widget = MeasurementComparePanel(frame, {})
    qtbot.addWidget(widget)

    assert widget.choose_wells() is False
    assert "do not say which well" in widget.report.toPlainText()


def test_cancelling_the_well_chooser_changes_nothing(panel, monkeypatch):
    monkeypatch.setattr(mcd._WellChoice, "exec",
                        lambda self: QDialog.Rejected)
    before = panel.chosen_wells()
    assert panel.choose_wells() is False
    assert panel.chosen_wells() == before


def test_choosing_the_same_wells_again_is_not_a_change(panel, monkeypatch):
    offered = panel.wells_on_offer()
    monkeypatch.setattr(mcd._WellChoice, "exec",
                        lambda self: QDialog.Accepted)
    monkeypatch.setattr(mcd._WellChoice, "chosen",
                        lambda self: list(offered))

    assert panel.choose_wells() is True
    # Second time round the answer is identical, so nothing is rebuilt.
    assert panel.choose_wells() is False


def test_choosing_a_subset_of_wells_narrows_the_annotated_group(panel,
                                                                monkeypatch):
    """The choice trims the ANNOTATED wells, and the comparison says which
    ones it left out -- an annotated well silently dropped would change the
    claim without changing the picture."""
    offered = panel.wells_on_offer()
    keep = list(offered[:2])
    monkeypatch.setattr(mcd._WellChoice, "exec",
                        lambda self: QDialog.Accepted)
    monkeypatch.setattr(mcd._WellChoice, "chosen", lambda self: keep)

    assert panel.choose_wells() is True
    assert panel.chosen_wells() == keep

    comparison = panel.comparison()
    assert "annotated well(s) left out" in comparison.note
    annotated = comparison.frame.loc[
        comparison.frame["group"] == "TGGT1_220950", "unit"]
    assert set(annotated) <= set(keep)


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------

def test_an_outcome_that_is_not_a_dict_is_reported_not_unpacked(panel):
    assert panel._finish_join(None) == "the join returned nothing"
    assert "Could not join" in panel.join_note.text()


def test_moving_a_join_box_re_joins_only_after_a_join_has_been_made(panel,
                                                                    tmp_path,
                                                                    monkeypatch):
    """Otherwise the box takes effect on the NEXT press of a pressed button."""
    presses = []
    monkeypatch.setattr(type(panel), "join_the_tables",
                        lambda self, *a: presses.append(1) or "")

    panel._joined_once = False
    panel._on_join_choice()
    assert presses == []

    panel._joined_once = True
    panel._on_join_choice()
    assert presses == [1]


# ---------------------------------------------------------------------------
# The dependent variable
# ---------------------------------------------------------------------------

def test_the_dependent_variable_is_left_alone_when_it_is_not_asked_for(
        panel, objects):
    panel.join_dependent.setChecked(False)
    wide, note = panel._join_the_dependent_variable(objects)
    assert wide is objects
    assert note == ""


# ---------------------------------------------------------------------------
# Wells from guides
# ---------------------------------------------------------------------------

def test_explicit_wells_win_over_the_ones_derived_from_guides(panel):
    panel._selected_wells = ["p1_r1_c1"]
    assert panel.selected_wells() == ["p1_r1_c1"]


def test_changing_the_scope_re_derives_the_wells(panel):
    panel._selected_wells = ["p1_r1_c1"]
    panel._on_scope()
    assert panel._selected_wells is None


def test_setting_the_guides_clears_the_derived_wells(panel):
    panel._selected_wells = ["p1_r1_c1"]
    panel.set_selected_guides(["TGGT1_220950_1"])
    assert panel._selected_guides == ["TGGT1_220950_1"]
    assert panel._selected_wells is None


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_the_report_is_left_alone_before_the_first_comparison(qtbot,
                                                              objects):
    widget = MeasurementComparePanel(objects, {})
    qtbot.addWidget(widget)
    widget.report.setPlainText("untouched")
    widget._comparison = None

    widget._report()

    assert widget.report.toPlainText() == "untouched"


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The window around the panel
# ---------------------------------------------------------------------------

def test_the_window_forwards_everything_to_its_panel(qtbot, objects, groups,
                                                     tmp_path, monkeypatch):
    """The dialog reimplements nothing: every accessor is the panel's."""
    window = MeasurementCompareDialog(objects, groups)
    qtbot.addWidget(window)

    assert window.measurement is window.panel.measurement
    assert window.level is window.panel.level
    assert window.kind is window.panel.kind
    assert window.report is window.panel.report
    assert window.contrast is window.panel.contrast
    assert window.controls is window.panel.controls
    assert window.refresh() is window.panel.comparison()
    assert window.comparison() is window.panel.comparison()
    assert window.join_the_tables() == (
        "no measurements database is attached")
    assert window.save_everything(str(tmp_path))

    monkeypatch.setattr(mcd._WellChoice, "exec",
                        lambda self: QDialog.Rejected)
    assert window.choose_wells() is False
