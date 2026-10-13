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

from PySide6.QtWidgets import QDialog, QFileDialog

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

def test_a_table_with_no_measurement_says_so_rather_than_drawing(qtbot):
    frame = pd.DataFrame({"plateID": ["p1", "p1"], "rowID": ["r1", "r1"],
                          "columnID": ["c1", "c2"], "fieldID": ["f1", "f1"],
                          "object_label": [1, 2]})
    widget = MeasurementComparePanel(frame, {})
    qtbot.addWidget(widget)

    assert widget.refresh() is None
    assert "no measurement column" in widget.report.toPlainText()


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

def test_a_panel_with_no_database_says_there_is_nothing_to_join(panel):
    assert panel.join_the_tables() == "no measurements database is attached"
    assert panel.join_button.isVisible() is False


def test_a_second_press_while_a_join_is_running_is_ignored(panel, tmp_path):
    panel._databases = [str(tmp_path / "measurements.db")]
    panel._joining = True
    assert panel.join_the_tables() == ""
    # The button text is untouched, so the running join still owns it.
    assert panel.join_button.text() != "Joining…"


def test_a_join_that_raises_is_a_message_in_the_panel(panel, tmp_path,
                                                      monkeypatch):
    """"a failed join is a message in the panel, not a traceback"."""
    panel._databases = [str(tmp_path / "measurements.db")]
    seen = {}

    def explode(_objects, _databases, png_list=False):
        raise RuntimeError("that file is not a measurements database")

    monkeypatch.setattr(mcd, "join_measurements", explode)
    monkeypatch.setattr(panel._jobs, "submit",
                        lambda work, done: seen.update(outcome=work()) or True)

    panel.join_the_tables()

    assert seen["outcome"] == {
        "error": "that file is not a measurements database"}
    why = panel._finish_join(seen["outcome"])
    assert why == "that file is not a measurements database"
    assert "Could not join" in panel.join_note.text()
    assert panel.join_button.isEnabled() is True


def test_an_outcome_that_is_not_a_dict_is_reported_not_unpacked(panel):
    assert panel._finish_join(None) == "the join returned nothing"
    assert "Could not join" in panel.join_note.text()


def test_a_join_replaces_the_object_rows_and_keeps_the_groups(panel, objects,
                                                              groups,
                                                              tmp_path,
                                                              monkeypatch):
    wider = objects.copy()
    wider["pathogen_area"] = 1.0
    panel._databases = [str(tmp_path / "measurements.db")]
    monkeypatch.setattr(mcd, "join_measurements",
                        lambda o, d, png_list=False: (wider, ""))
    monkeypatch.setattr(panel._jobs, "submit",
                        lambda work, done: done(work()) or True)

    panel.join_the_tables()

    assert panel._joined_once is True
    names = [panel.measurement.itemData(i)
             for i in range(panel.measurement.count())]
    assert "pathogen_area" in names


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


def test_asking_for_a_dependent_variable_that_is_not_attached_says_so(
        panel, objects):
    panel.join_dependent.setChecked(True)
    panel.set_dependent_frame(None)

    _wide, note = panel._join_the_dependent_variable(objects)

    assert "no table carrying it is attached" in note


def test_an_empty_dependent_table_is_the_same_as_none(panel, objects):
    panel.join_dependent.setChecked(True)
    panel.set_dependent_frame(pd.DataFrame({"pred": []}))

    _wide, note = panel._join_the_dependent_variable(objects)

    assert "no table carrying it is attached" in note


def test_a_dependent_join_that_matches_nothing_is_reported_not_swallowed(
        panel, objects, monkeypatch):
    """A route that matches nothing is a failure, not an empty answer."""
    panel.join_dependent.setChecked(True)
    panel.set_dependent_frame(pd.DataFrame({"pred": [0.5]}))

    from spacr import dependent_join

    def explode(_wide, _frame):
        raise ValueError("no identifier column in common")

    monkeypatch.setattr(dependent_join, "join", explode)

    wide, note = panel._join_the_dependent_variable(objects)

    assert wide is objects
    assert "did not join: no identifier column in common" in note


def test_a_dependent_join_that_works_reports_the_route_it_took(panel,
                                                               objects,
                                                               monkeypatch):
    panel.join_dependent.setChecked(True)
    panel.set_dependent_frame(pd.DataFrame({"pred": [0.5]}))
    joined = objects.copy()
    joined["pred"] = 0.5

    from spacr import dependent_join

    monkeypatch.setattr(dependent_join, "join",
                        lambda wide, frame: (joined, {"route": "prcfo"}))
    monkeypatch.setattr(dependent_join, "describe",
                        lambda report: "matched on prcfo")

    wide, note = panel._join_the_dependent_variable(objects)

    assert "pred" in wide.columns
    assert note == "matched on prcfo"


def test_the_dependent_note_is_appended_to_the_join_note(panel, objects,
                                                          tmp_path,
                                                          monkeypatch):
    panel._databases = [str(tmp_path / "m.db")]
    panel.join_dependent.setChecked(True)
    panel.set_dependent_frame(None)

    outcome = {"wide": objects, "trouble": "one plate had no rows"}
    trouble = panel._finish_join(outcome)

    assert "one plate had no rows" in trouble
    assert "no table carrying it is attached" in trouble


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

def test_a_heading_keeps_its_plain_tooltip_when_the_filter_cannot_install(
        qtbot, objects, monkeypatch):
    """A worse tooltip rather than none."""
    from spacr.qt.screens import settings_model

    class Refuses:
        def __init__(self, *_args):
            raise RuntimeError("no event filter here")

    monkeypatch.setattr(settings_model, "_ApiTooltipFilter", Refuses)

    widget = MeasurementComparePanel(objects, {})
    qtbot.addWidget(widget)

    assert widget.headings["level"].toolTip()
    assert getattr(widget, "_heading_filter", None) is None


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def test_a_pyqtgraph_failure_falls_back_to_a_matplotlib_canvas(panel,
                                                               monkeypatch):
    """The figure still appears; only the interactive plot is lost."""
    import spacr.qt.widgets.grouped_plot as gp

    class Refuses:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("pyqtgraph is not available")

    monkeypatch.setattr(gp, "GroupedPlot", Refuses)

    panel.refresh()

    holder = panel._figure_holder
    assert holder.count() >= 1
    assert holder.itemAt(holder.count() - 1).widget() is not None


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

def test_saving_before_there_is_a_comparison_writes_nothing(qtbot, objects):
    widget = MeasurementComparePanel(objects, {})
    qtbot.addWidget(widget)
    widget._comparison = None
    assert widget.save_everything("/tmp") == {}


def test_a_cancelled_folder_dialog_writes_nothing(panel, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    assert panel.save_everything() == {}


def test_a_save_that_refuses_is_appended_to_the_report(panel, monkeypatch):
    before = panel.report.toPlainText()

    def explode(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(mcd, "save", explode)

    assert panel.save_everything("/nowhere") == {}
    assert before in panel.report.toPlainText()
    assert "Could not save: read-only file system" in panel.report.toPlainText()


def test_a_save_says_how_many_items_went_where(panel, tmp_path):
    written = panel.save_everything(str(tmp_path))

    assert written
    assert f"Saved {len(written)} item(s) to {tmp_path}" in \
        panel.report.toPlainText()
    assert any(tmp_path.iterdir())


def test_the_folder_dialog_supplies_the_destination(panel, tmp_path,
                                                    monkeypatch):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))
    assert panel.save_everything()
    assert any(tmp_path.iterdir())


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
