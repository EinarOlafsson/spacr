"""Instruction 154 F: the tab reaches its own purpose.

    "the point of the measurements tab is to merge measurements so that
     regression can be run on any column in the databases", as four steps --

        1. LOAD the measurement databases
        2. MERGE THE TABLES within each database
        3. MERGE THE DATABASES into one frame
        4. PICK A COLUMN and regress on it

Steps 1-3 shipped and step 4 did not, so the tab ENDED BEFORE ITS OWN PURPOSE
-- which is most of "i dont understand how this is all set up, its probably
broken". These tests are about the fourth step and about the three things it
needs to be honest:

  * THE MERGED FRAME IS AN ARTEFACT, written once and named. Twelve fits that
    re-merged four databases twelve times would not only be twelve times
    slower, they would not be guaranteed to have been fitted on the same
    numbers.
  * A QUEUE OF N FITS IS A LONG JOB: off the GUI thread, saying which column
    it is on, cancellable -- and A FIT THAT FAILS DOES NOT TAKE THE OTHER N-1
    WITH IT. A queue where the fourth column raises and the remaining eight
    never run is a queue that silently did a third of what was asked.
  * EACH COLUMN IS ITS OWN RUN, with its own folder and its own row in the
    Runs tab, "each gets saved as a run that i can evaluate".
"""
from __future__ import annotations

import os
import sqlite3
import threading

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


# --------------------------------------------------------------------------- #
#  Two plates, with a column that varies, one that does not, and a text one
# --------------------------------------------------------------------------- #

def _database(folder, plate, cells=4):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(str(folder), "measurements.db")
    identity = {"rowID": ["r1"] * cells, "columnID": ["c1"] * cells,
                "fieldID": ["f1"] * cells}
    cell = pd.DataFrame({
        "plateID": [plate] * cells, **identity,
        "object_label": list(range(1, cells + 1)),
        "file_name": [f"{plate}.tif"] * cells,
        "area": [100.0 * i for i in range(1, cells + 1)],
        "wobble": [0.5 * i for i in range(1, cells + 1)],
        "constant": [7.0] * cells,
    })
    with sqlite3.connect(path) as db:
        cell.to_sql("cell", db, index=False)
    return path


@pytest.fixture()
def two_plates(tmp_path):
    return [_database(tmp_path / "plate1", "plate1"),
            _database(tmp_path / "plate2", "plate2")]


def _rows(paths, tmp_path):
    return [{"plate": f"plate{i + 1}",
             "score": str(tmp_path / f"s{i + 1}.csv"),
             "count": str(tmp_path / f"c{i + 1}.csv"),
             "database": path}
            for i, path in enumerate(paths)]


def _tab(qtbot, paths, tmp_path, *, threaded=False, fit=None, settings=None):
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    rows = _rows(paths, tmp_path)
    panel = MeasurementScanPanel(
        database_provider=lambda: rows,
        destination_provider=lambda: str(tmp_path / "measurements"),
        settings_provider=lambda: dict(
            settings or {"regression_type": "ols", "paired_data": rows,
                         "src": str(tmp_path)}),
        threaded=threaded, fit=fit)
    qtbot.addWidget(panel)
    return panel


# --------------------------------------------------------------------------- #
#  Which columns a regression could actually take
# --------------------------------------------------------------------------- #

def test_only_numeric_measurements_that_vary_are_offered():
    """Both halves are checked, and both were needed.

    Identity survives every dtype filter -- `object_label` is an integer and
    is a NAME -- and a constant column makes a degenerate fit that every
    backend reports differently, turning one unusable choice into N
    different-looking failures.
    """
    from spacr.qt.widgets.measurement_scan_panel import regressable_columns

    frame = pd.DataFrame({
        "plateID": ["p1", "p1"], "rowID": ["r1", "r1"],
        "columnID": ["c1", "c1"], "fieldID": ["f1", "f1"],
        "object_label": [1, 2], "source_database": ["a", "b"],
        "cell_file_name": ["x.tif", "y.tif"],
        "cell_area": [10.0, 20.0],
        "cell_constant": [7.0, 7.0],
        "cell_flag": [True, False],
    })

    assert regressable_columns(frame) == ("cell_area",)


def test_no_frame_offers_no_columns():
    from spacr.qt.widgets.measurement_scan_panel import regressable_columns

    assert regressable_columns(None) == ()
    assert regressable_columns(pd.DataFrame()) == ()


# --------------------------------------------------------------------------- #
#  The artefact
# --------------------------------------------------------------------------- #

def test_the_merged_frame_is_written_once_and_named(qtbot, two_plates,
                                                    tmp_path):
    """It survives the tab, which is what "regress on any column" needs.

    The suffix is no longer asserted as `.csv`: the merge stages the frame,
    and a stage writes Parquet where an engine is installed and CSV where none
    is. What has to be true either way is that the artefact is named for the
    merge, is on disk, and reads back as the frame that was merged -- so this
    reads it through `tabular.read_table`, which dispatches on the suffix,
    instead of assuming one.
    """
    from spacr import tabular

    panel = _tab(qtbot, two_plates, tmp_path)

    assert panel.databases.merged_frame_path() == ""
    frame = panel.databases.merge()

    written = panel.databases.merged_frame_path()
    assert written and os.path.isfile(written)
    assert os.path.splitext(os.path.basename(written))[0] == \
        "merged_measurements"
    assert len(tabular.read_table(written)) == len(frame)


def test_the_merged_frame_reaches_the_fit_without_being_parsed(
        qtbot, two_plates, tmp_path):
    """The artefact is written for the user; the fit gets the object.

    The frame is already in this process when it is written, so parsing it
    back is pure cost -- 2.75 GB of it on a four-plate screen. A queue that
    left the fit to read the file would be paying that per fit.
    """
    from spacr import frame_handoff

    panel = _tab(qtbot, two_plates, tmp_path)
    frame = panel.databases.merge()
    written = panel.databases.merged_frame_path()

    assert frame_handoff.held(written) is frame

    handed = []
    panel.regression._fit = lambda settings: handed.append(
        [frame_handoff.held(pair["score"]) is frame
         for pair in settings["paired_data"]]) or {
             "results": pd.DataFrame({"feature": ["a"]}),
             "res_folder": str(tmp_path / "ols_1")}
    panel.regression.set_selected_columns(["cell_area"])
    panel.regression.start_regressions()

    # Two plate rows, both pointed at the one merged frame -- which is the
    # shape that made the old loader parse the same file once per row.
    assert handed == [[True, True]]


def test_a_second_queue_is_handed_the_frame_again(qtbot, two_plates,
                                                  tmp_path):
    """Finishing withdraws the offer, so starting must make it again.

    Without that, only the first queue over a merge takes the object and
    every queue after it parses the artefact back once per fit while the
    panel is still holding the frame it wrote.
    """
    from spacr import frame_handoff

    panel = _tab(qtbot, two_plates, tmp_path)
    frame = panel.databases.merge()
    written = panel.databases.merged_frame_path()

    seen = []
    panel.regression._fit = lambda settings: seen.append(
        frame_handoff.held(written) is frame) or {
            "results": pd.DataFrame({"feature": ["a"]}),
            "res_folder": str(tmp_path / "ols_1")}

    panel.regression.set_selected_columns(["cell_area"])
    panel.regression.start_regressions()
    assert frame_handoff.held(written) is None, (
        "a finished queue must withdraw what it offered")

    panel.regression.set_selected_columns(["cell_wobble"])
    panel.regression.start_regressions()

    assert seen == [True, True]
    frame_handoff.release(written)


def test_the_artefact_is_columnar_and_the_frame_is_offered(tmp_path):
    """The producer half of the handoff, on its own.

    `write_merged_frame` used to call `frame.to_csv` and offer nothing, which
    made the Measurements queue the one producer the handoff never saw: it
    wrote the slowest format there is and then let every fit parse it back.
    """
    from spacr import frame_handoff, tabular
    from spacr.qt.widgets.measurement_scan_panel import write_merged_frame

    frame = pd.DataFrame({"plateID": ["plate1", "plate2"],
                          "rowID": ["r1", "r1"], "columnID": ["c1", "c1"],
                          "cell_area": [10.0, 20.0]})
    said = []

    written = write_merged_frame(frame, tmp_path / "measurements",
                                 report=said.append)
    try:
        assert os.path.splitext(written)[1] in (".parquet", ".csv")
        assert frame_handoff.held(written) is frame
        assert tabular.read_table(written)["cell_area"].tolist() == [10.0,
                                                                     20.0]
        assert "merged_measurements" in said[0]
    finally:
        frame_handoff.release(written)


def test_nothing_to_write_writes_nothing_and_offers_nothing(tmp_path):
    """An empty merge must not leave an offer standing under a path that has
    no file, which a later reader would take as the frame."""
    from spacr import frame_handoff
    from spacr.qt.widgets.measurement_scan_panel import write_merged_frame

    frame = pd.DataFrame({"cell_area": [1.0]})

    assert write_merged_frame(None, tmp_path) == ""
    assert write_merged_frame(pd.DataFrame(), tmp_path) == ""
    assert write_merged_frame(frame, "") == ""
    assert frame_handoff.held(tmp_path / "merged_measurements.parquet") is None


def test_the_merge_says_where_it_wrote(qtbot, two_plates, tmp_path):
    panel = _tab(qtbot, two_plates, tmp_path)
    panel.databases.merge()

    assert panel.databases.merged_frame_path() in panel.databases.statement()


def test_a_merge_with_nowhere_to_write_still_merges(qtbot, two_plates,
                                                    tmp_path):
    """A merged frame nobody can write is still a merged frame.

    So the artefact's absence is a NOTE and never a refusal -- the panel is
    used headless and in tests where there is nowhere to put a file.
    """
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    panel = MeasurementScanPanel(
        database_provider=lambda: _rows(two_plates, tmp_path), threaded=False)
    qtbot.addWidget(panel)

    assert panel.databases.merge() is not None
    assert panel.databases.merged_frame_path() == ""
    assert "not been written" in panel.regression.state.text() \
        or "no destination" in panel.databases.step_states()[3]


# --------------------------------------------------------------------------- #
#  One fit's settings
# --------------------------------------------------------------------------- #

def test_each_fit_changes_the_response_and_nothing_else():
    """What makes the runs comparable is that only one thing varied."""
    from spacr.qt.widgets.measurement_scan_panel import column_run_settings

    base = {"regression_type": "ols", "fdr_alpha": 0.05,
            "dependent_variable": "pred",
            "paired_data": [{"plate": "plate1", "score": "old.csv",
                             "count": "counts1.csv"},
                            {"plate": "plate2", "score": "older.csv",
                             "count": "counts2.csv"}]}

    settings = column_run_settings(base, "cell_area", "/data/merged.csv")

    assert settings["dependent_variable"] == "cell_area"
    assert settings["regression_type"] == "ols"
    # THE COUNT SIDE IS LEFT ALONE. The guides each well got are the same
    # guides; only what is regressed onto them changed.
    assert [pair["count"] for pair in settings["paired_data"]] == [
        "counts1.csv", "counts2.csv"]
    assert {pair["score"] for pair in settings["paired_data"]} == {
        "/data/merged.csv"}


def test_the_base_settings_are_never_mutated():
    """Twelve fits built by mutating one dict are twelve fits of the last one."""
    from spacr.qt.widgets.measurement_scan_panel import column_run_settings

    base = {"dependent_variable": "pred",
            "paired_data": [{"score": "old.csv", "count": "c.csv"}]}

    column_run_settings(base, "cell_area", "/data/merged.csv")

    assert base["dependent_variable"] == "pred"
    assert base["paired_data"][0]["score"] == "old.csv"


# --------------------------------------------------------------------------- #
#  The queue
# --------------------------------------------------------------------------- #

def test_a_fit_that_fails_does_not_take_the_others_with_it():
    """The whole reason this is a queue and not a loop."""
    from spacr.qt.widgets.measurement_scan_panel import run_column_fits

    def fit(settings):
        if settings["dependent_variable"] == "b":
            raise RuntimeError("singular design")
        return {"results": [1, 2], "res_folder": f"/runs/{settings['dependent_variable']}"}

    fits = run_column_fits(["a", "b", "c"], lambda c: {"dependent_variable": c},
                           fit)

    assert [f.column for f in fits] == ["a", "b", "c"]
    assert [f.ok for f in fits] == [True, False, True]
    assert "singular design" in fits[1].error
    assert fits[2].folder == "/runs/c"


def test_the_failure_says_which_error_it_was():
    """"did not fit" is not an answer; the exception's own words are."""
    from spacr.qt.widgets.measurement_scan_panel import run_column_fits

    def fit(_settings):
        raise ValueError("no variance left after blocking")

    fits = run_column_fits(["a"], lambda c: {"dependent_variable": c}, fit)

    assert fits[0].error == "ValueError: no variance left after blocking"
    assert "no variance left" in fits[0].describe()


def test_a_cancel_stops_between_fits_and_keeps_the_finished_ones():
    """Not mid-fit, and the honesty is the point.

    A regression stopped half-way has written part of a results folder and
    there is no way to say what that folder means. The fits that finished are
    complete runs.
    """
    from spacr.qt.widgets.measurement_scan_panel import (QueueCancelled,
                                                         run_column_fits)

    done = []
    stop = {"now": False}

    def fit(settings):
        done.append(settings["dependent_variable"])
        stop["now"] = True
        return {"results": [1], "res_folder": "/runs/x"}

    with pytest.raises(QueueCancelled) as raised:
        run_column_fits(["a", "b", "c"], lambda c: {"dependent_variable": c},
                        fit, cancelled=lambda: stop["now"])

    assert done == ["a"]
    assert "1 of 3" in str(raised.value)


def test_each_outcome_is_handed_over_as_it_is_decided():
    """A queue of twelve fills the Runs tab as it goes, not at the end."""
    from spacr.qt.widgets.measurement_scan_panel import run_column_fits

    seen = []
    run_column_fits(
        ["a", "b"], lambda c: {"dependent_variable": c},
        lambda s: {"results": [1], "res_folder": "/runs"},
        on_result=lambda outcome: seen.append(outcome.column))

    assert seen == ["a", "b"]


def test_a_fit_that_returns_a_path_is_read_too():
    """`perform_regression` returns a dict through the GUI and a PATH direct."""
    from spacr.qt.widgets.measurement_scan_panel import run_column_fits

    fits = run_column_fits(
        ["a"], lambda c: {"dependent_variable": c},
        lambda s: "/runs/ols_1/results.csv")

    assert fits[0].ok and fits[0].folder == "/runs/ols_1"


def test_a_fit_that_returns_nothing_is_not_reported_as_a_success():
    from spacr.qt.widgets.measurement_scan_panel import run_column_fits

    fits = run_column_fits(["a"], lambda c: {"dependent_variable": c},
                           lambda s: None)

    assert fits[0].ok is False
    assert "nothing to look at" in fits[0].error


# --------------------------------------------------------------------------- #
#  Step 4 on the tab
# --------------------------------------------------------------------------- #

def test_the_tab_reads_as_four_named_steps(qtbot, two_plates, tmp_path):
    """A page with no headings is what "i dont understand how this is all set
    up" is a complaint about."""
    from PySide6.QtWidgets import QLabel

    panel = _tab(qtbot, two_plates, tmp_path)
    panel.resize(700, 1000)
    panel.show()
    # SORTED BY WHERE THEY ARE ON SCREEN, not by findChildren order. The
    # traversal order is an accident of construction and reparenting -- the
    # sections became QSplitter children on 2026-08-19 and it changed -- while
    # "reads as four named steps" is a claim about the ORDER A READER SEES.
    steps = [w for w in panel.findChildren(QLabel)
             if w.objectName() == "WorkflowStep"]
    steps.sort(key=lambda w: w.mapTo(panel, w.rect().topLeft()).y())
    headings = [w.text() for w in steps]

    assert len(headings) == 4
    assert headings[0].startswith("1.")
    assert headings[3].startswith("4.")
    assert "REGRESS" in headings[3]


def test_each_step_says_where_it_stands(qtbot, two_plates, tmp_path):
    panel = _tab(qtbot, two_plates, tmp_path)

    before = panel.databases.step_states()
    assert "2 database(s) attached" in before[1]
    assert "cell" in before[2]
    assert before[3] == "Not merged yet."

    panel.databases.merge()

    after = panel.databases.step_states()
    assert "Merged:" in after[3] and "Written to" in after[3]


def test_step_four_offers_the_merged_frames_columns(qtbot, two_plates,
                                                    tmp_path):
    panel = _tab(qtbot, two_plates, tmp_path)
    assert panel.regression.columns() == ()

    panel.databases.merge()

    assert set(panel.regression.columns()) == {"cell_area", "cell_wobble"}
    # The constant and the text identifier are left out, not hidden away.
    assert "cell_constant" not in panel.regression.columns()
    assert "cell_file_name" not in panel.regression.columns()


def test_step_four_says_why_it_can_do_nothing_yet(qtbot, two_plates,
                                                  tmp_path):
    """A control that does nothing and says nothing is the failure this file
    keeps fixing."""
    panel = _tab(qtbot, two_plates, tmp_path)

    assert "merge the databases in step 3" in panel.regression.state.text()
    assert panel.regression.run_button.isEnabled() is False


def test_the_picker_is_multi_select_and_each_column_is_a_run(
        qtbot, two_plates, tmp_path):
    fitted = []

    def fit(settings):
        fitted.append(settings["dependent_variable"])
        return {"results": pd.DataFrame({"feature": ["a"]}),
                "res_folder": str(tmp_path / settings["dependent_variable"])}

    panel = _tab(qtbot, two_plates, tmp_path, fit=fit)
    panel.databases.merge()
    assert panel.regression.set_selected_columns(
        ["cell_area", "cell_wobble"]) == 2

    started, finished, ended = [], [], []
    panel.regression.fit_started.connect(lambda c, s: started.append(c))
    panel.regression.fit_finished.connect(
        lambda c, outcome: finished.append((c, outcome["ok"])))
    panel.regression.queue_finished.connect(
        lambda ok, bad: ended.append((ok, bad)))

    assert panel.regression.start_regressions() is True

    assert fitted == ["cell_area", "cell_wobble"]
    # ONE `fit_started` PER FIT. It went out twice for the first column while
    # the queue also announced itself before submitting, which is two rows in
    # the Runs tab for one fit -- and the first says "running" for ever
    # because the second overwrote its handle.
    assert started == ["cell_area", "cell_wobble"]
    assert finished == [("cell_area", True), ("cell_wobble", True)]
    assert ended == [(2, 0)]


def test_one_columns_failure_is_reported_and_the_others_still_run(
        qtbot, two_plates, tmp_path):
    def fit(settings):
        if settings["dependent_variable"] == "cell_wobble":
            raise RuntimeError("singular design")
        return {"results": pd.DataFrame({"feature": ["a"]}),
                "res_folder": str(tmp_path / "ols_1")}

    panel = _tab(qtbot, two_plates, tmp_path, fit=fit)
    panel.databases.merge()
    panel.regression.set_selected_columns(["cell_area", "cell_wobble"])
    panel.regression.start_regressions()

    assert [f.ok for f in panel.regression.outcomes()] == [True, False]
    assert "singular design" in panel.regression.outcomes_box.toPlainText()
    assert "1 run(s) fitted, 1 did not" in panel.regression.progress.text()


def test_every_fit_reads_the_one_file_the_merge_wrote(qtbot, two_plates,
                                                      tmp_path):
    """The merge is paid for once and the runs are on the same numbers."""
    scores = []

    def fit(settings):
        scores.append({pair["score"] for pair in settings["paired_data"]})
        return {"results": pd.DataFrame({"feature": ["a"]}),
                "res_folder": str(tmp_path / "ols_1")}

    panel = _tab(qtbot, two_plates, tmp_path, fit=fit)
    panel.databases.merge()
    written = panel.databases.merged_frame_path()
    panel.regression.set_selected_columns(["cell_area", "cell_wobble"])
    panel.regression.start_regressions()

    assert scores == [{written}, {written}]


def test_a_frame_provider_that_fails_still_lets_the_queue_read_the_file(
        qtbot, two_plates, tmp_path):
    """The offer is an optimisation; losing it costs a parse, not the run.

    The provider is the live merging panel, so it can raise -- and a queue
    that refused to start because the in-memory shortcut was unavailable
    would have turned a fast path into a requirement.
    """
    panel = _tab(qtbot, two_plates, tmp_path)
    panel.databases.merge()
    written = panel.databases.merged_frame_path()

    def _explode():
        raise RuntimeError("the merged frame was dropped")

    panel.regression._frame_provider = _explode
    assert panel.regression._offer_frame(written) is False
    assert panel.regression._offer_frame("") is False

    panel.regression._fit = lambda settings: {
        "results": pd.DataFrame({"feature": ["a"]}),
        "res_folder": str(tmp_path / "ols_1")}
    panel.regression._columns = ("cell_area",)
    panel.regression.columns_list.addItem("cell_area")
    panel.regression.columns_list.item(0).setSelected(True)

    assert panel.regression.start_regressions() is True
    assert [f.ok for f in panel.regression.outcomes()] == [True]


def test_the_queue_refuses_with_nothing_selected_and_says_so(
        qtbot, two_plates, tmp_path):
    panel = _tab(qtbot, two_plates, tmp_path)
    panel.databases.merge()
    panel.regression.set_selected_columns([])

    assert panel.regression.start_regressions() is False
    assert "Pick at least one column" in panel.regression.progress.text()


def test_the_queue_refuses_without_the_artefact_and_names_the_reason(
        qtbot, two_plates, tmp_path):
    """A fit with nothing to read must not be started and then fail N times."""
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    panel = MeasurementScanPanel(
        database_provider=lambda: _rows(two_plates, tmp_path),
        settings_provider=lambda: {},
        threaded=False, fit=lambda settings: None)
    qtbot.addWidget(panel)
    panel.databases.merge()
    panel.regression.refresh()
    # Nothing to select from is a different refusal; force one so the artefact
    # is the only thing missing.
    panel.regression._columns = ("cell_area",)
    panel.regression.columns_list.addItem("cell_area")
    panel.regression.columns_list.item(0).setSelected(True)

    assert panel.regression.start_regressions() is False
    assert "not been written" in panel.regression.progress.text()


def test_filtering_the_list_never_changes_what_is_selected(
        qtbot, two_plates, tmp_path):
    """A filter that deselected what it hid would silently shorten the queue."""
    panel = _tab(qtbot, two_plates, tmp_path)
    panel.databases.merge()
    panel.regression.set_selected_columns(["cell_area", "cell_wobble"])

    panel.regression.filter.setText("wobble")

    assert panel.regression.selected_columns() == ("cell_area", "cell_wobble")
    hidden = [panel.regression.columns_list.item(i).isHidden()
              for i in range(panel.regression.columns_list.count())]
    assert any(hidden), "the filter hid nothing"


def test_a_second_merge_re_offers_the_columns(qtbot, two_plates, tmp_path):
    """Otherwise the picker holds the previous merge's columns and every fit
    reads a file that has been overwritten underneath it."""
    panel = _tab(qtbot, two_plates, tmp_path)
    assert panel.regression.columns() == ()

    panel.databases.merge()

    assert panel.regression.columns(), "the merged signal did not reach step 4"


def test_the_selection_survives_a_refresh(qtbot, two_plates, tmp_path):
    panel = _tab(qtbot, two_plates, tmp_path)
    panel.databases.merge()
    panel.regression.set_selected_columns(["cell_area"])

    panel.regression.refresh()

    assert panel.regression.selected_columns() == ("cell_area",)


# --------------------------------------------------------------------------- #
#  A QUEUE OF N FITS IS A LONG JOB (the 154 A lesson, one screen along)
# --------------------------------------------------------------------------- #

def test_the_queue_runs_off_the_gui_thread(qtbot, two_plates, tmp_path):
    """The only test of this that cannot be faked.

    A progress label that is never repainted is indistinguishable from a
    frozen window, so what is asserted is WHICH THREAD the fits ran on.
    """
    gui_thread = threading.current_thread()
    ran_on = []

    def fit(settings):
        ran_on.append(threading.current_thread())
        return {"results": pd.DataFrame({"feature": ["a"]}),
                "res_folder": str(tmp_path / "ols_1")}

    panel = _tab(qtbot, two_plates, tmp_path, threaded=False, fit=fit)
    panel.databases.merge()
    # The MERGE above is driven synchronously on purpose; the QUEUE is what
    # this test is about, so only it is threaded.
    panel.regression._threaded = True
    panel.regression._jobs._threaded = True
    panel.regression.set_selected_columns(["cell_area", "cell_wobble"])

    ended = []
    panel.regression.queue_finished.connect(
        lambda ok, bad: ended.append((ok, bad)))
    assert panel.regression.start_regressions() is True
    # The click handler has ALREADY returned while the fits are still going.
    assert panel.regression.is_running() is True
    qtbot.waitUntil(lambda: bool(ended), timeout=30000)

    assert ran_on and gui_thread not in ran_on, ran_on
    assert ended == [(2, 0)]


def test_while_the_queue_runs_stop_is_the_button_that_is_enabled(
        qtbot, two_plates, tmp_path):
    panel = _tab(qtbot, two_plates, tmp_path,
                 fit=lambda s: {"results": pd.DataFrame({"feature": ["a"]}),
                                "res_folder": str(tmp_path)})
    panel.databases.merge()
    panel.regression.set_selected_columns(["cell_area"])

    assert panel.regression.run_button.isEnabled() is True
    assert panel.regression.cancel_button.isEnabled() is False

    states = []
    panel.regression.queue_progress.connect(
        lambda *_a: states.append((panel.regression.run_button.isEnabled(),
                                   panel.regression.cancel_button.isEnabled())))
    panel.regression.start_regressions()

    assert states == [(False, True)]
    assert panel.regression.run_button.isEnabled() is True


def test_stopping_a_queue_keeps_the_runs_that_finished(qtbot, two_plates,
                                                       tmp_path):
    panel = _tab(qtbot, two_plates, tmp_path)

    def fit(settings):
        panel.regression.cancel()
        return {"results": pd.DataFrame({"feature": ["a"]}),
                "res_folder": str(tmp_path / "ols_1")}

    panel.regression._fit = fit
    panel.databases.merge()
    panel.regression.set_selected_columns(["cell_area", "cell_wobble"])
    panel.regression.start_regressions()

    assert [f.column for f in panel.regression.outcomes()] == ["cell_area"]
    assert "Stopped after 1 of 2" in panel.regression.progress.text()
    assert "1 run(s) finished" in panel.regression.progress.text()


def test_cancelling_when_nothing_is_running_says_so(qtbot, two_plates,
                                                    tmp_path):
    panel = _tab(qtbot, two_plates, tmp_path)
    assert panel.regression.cancel() is False


# --------------------------------------------------------------------------- #
#  The whole workflow, on the real screen, ending in the Runs tab
# --------------------------------------------------------------------------- #

def test_three_columns_become_three_rows_in_the_runs_tab(qtbot, two_plates,
                                                         tmp_path):
    """154 F's own acceptance test: "load 4 databases, merge, pick 3 columns,
    get 3 runs, and evaluate them against each other in the Runs tab, without
    leaving the app"."""
    from spacr.qt.screens.app_screen import AppScreen

    def fit(settings):
        response = settings["dependent_variable"]
        if response == "cell_wobble":
            raise ValueError("no variance left after blocking")
        folder = tmp_path / "results" / response
        folder.mkdir(parents=True, exist_ok=True)
        return {"results": pd.DataFrame({"feature": ["a", "b", "c"]}),
                "res_folder": str(folder)}

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    rows = _rows(two_plates, tmp_path)
    screen._settings_model._widgets["paired_data"].set_value(rows)
    screen._settings_model.set_value_for_key("src", str(tmp_path))

    scan = screen._scan_panel
    scan.refresh_databases()
    assert scan.databases.merge() is not None
    assert scan.databases.merged_frame_path()
    scan.regression.refresh()

    # The queue driven inline; the threading is asserted on its own above.
    scan.regression._threaded = False
    scan.regression._jobs._threaded = False
    scan.regression._fit = fit
    scan.regression.set_selected_columns(["cell_area", "cell_wobble"])
    assert scan.regression.start_regressions() is True

    table = screen._sweep_runs._frame
    assert table is not None and len(table) == 2
    # THE RESPONSE IS A COLUMN. A comparison table whose only varying column
    # is missing is a list of identical-looking rows.
    assert list(table["dependent_variable"]) == ["cell_area", "cell_wobble"]
    assert list(table["status"]) == ["ok", "failed"]
    assert str(table["folder"].iloc[0]).endswith("cell_area")
    # A run that finished is the loaded run (154 G); one that failed is not.
    assert list(table["loaded"]) == ["loaded", ""]


def test_the_destination_is_where_the_runs_write(qtbot, two_plates, tmp_path):
    """Beside the results it produced, not in a folder nobody can find.

    THE REGRESSION SCREEN HAS NO `src`, checked below rather than assumed --
    which is why the count file is the live branch of this rule, exactly as
    it is the live branch of `spacr.refit.destination`.
    """
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    assert "src" not in screen._settings_model.collect()

    screen._settings_model._widgets["paired_data"].set_value(
        _rows(two_plates, tmp_path))

    assert screen._measurements_destination() == os.path.join(
        str(tmp_path), "measurements")


def test_a_module_that_does_have_a_src_uses_it(qtbot, tmp_path):
    """One rule, in one order, so the frame and its runs share a project."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    screen._settings_model.collect = lambda: {
        "src": str(tmp_path / "project"),
        "paired_data": [{"count": "/elsewhere/counts.csv"}]}

    assert screen._measurements_destination() == os.path.join(
        str(tmp_path / "project"), "measurements")


def test_a_placeholder_src_is_not_a_destination(qtbot, tmp_path):
    """`path` is what the settings dicts ship as "not chosen yet"."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    screen._settings_model.collect = lambda: {
        "src": "path",
        "paired_data": [{"count": str(tmp_path / "counts.csv")}]}

    assert screen._measurements_destination() == os.path.join(
        str(tmp_path), "measurements")


def test_with_no_counts_named_the_databases_plate_folder_is_used(
        qtbot, two_plates, tmp_path):
    """A project whose counts are not named yet still has somewhere to write."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    screen._settings_model.collect = lambda: {}
    screen._attached_database_rows = lambda: [
        {"plate": "plate1", "database": two_plates[0]}]

    assert screen._measurements_destination() == os.path.join(
        str(tmp_path / "plate1"), "measurements")


def test_a_database_in_spacrs_own_layout_resolves_to_its_plate(qtbot,
                                                               tmp_path):
    """`<plate>/measurements/measurements.db` is two levels up, not one.

    And a LOOSE database is one, which is why the rule looks at the parent's
    name: assuming the deep layout for a loose file puts the merged frame in
    the plate's parent, which on a project root is everybody's folder.
    """
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    screen._settings_model.collect = lambda: {}
    deep = _database(tmp_path / "plate9" / "measurements", "plate9")
    screen._attached_database_rows = lambda: [
        {"plate": "plate9", "database": deep}]

    assert screen._measurements_destination() == os.path.join(
        str(tmp_path / "plate9"), "measurements")


def test_with_nothing_at_all_there_is_no_destination(qtbot):
    """And step 4 then SAYS the frame was never written, rather than fitting
    against a file that is not there."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    screen._settings_model.collect = lambda: {}
    screen._attached_database_rows = lambda: []

    assert screen._measurements_destination() == ""


def test_the_legacy_flat_count_list_is_still_understood(qtbot, tmp_path):
    """`count_data` is what `perform_regression` migrates the pairs into."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    screen._settings_model.collect = lambda: {
        "count_data": [str(tmp_path / "counts.csv")]}

    assert screen._measurements_destination() == os.path.join(
        str(tmp_path), "measurements")
