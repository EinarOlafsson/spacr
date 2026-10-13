"""Instruction 154 G — which run is loaded, and how a view finds out.

Reported 2026-08-18:

    "and the cell table just says that no regression results are loaded. but i
     just ran a gegression so that should be loaded automatically and
     otherwise there should be a checkbox in runs that specifies which run is
     loaded if there are several runs if there is one run then that is the
     loaded there should also be the opertunity to loaded rund."

Four rules, and each has its own test below:

* a run that FINISHES becomes the loaded run, with no step in between;
* ONE run means that run is loaded -- there is nothing to choose between;
* SEVERAL runs means the Runs tab carries the choice, and the choice is
  VISIBLE from the views that depend on it, not only from the tab that sets
  it;
* a run can be loaded deliberately, INCLUDING one from an earlier session on
  disk. A run on disk is a first-class run, not a degraded one.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt

from spacr.qt.widgets.sweep_runs import (  # noqa: E402
    LOADED_COLUMN, LOADED_MARK, SOURCE_DISK, SweepRunsPanel,
)


def _settings():
    return {"regression_type": "ols", "inference": "parametric",
            "fdr_alpha": 0.05}


def _trials(n=3):
    return pd.DataFrame({
        "trial_id": list(range(1, n + 1)),
        "status": ["ok"] * n,
        "regression_type": ["ols"] * n,
        "n_results": [10 * i for i in range(1, n + 1)],
    })


def _run_folder(root, name="ols_1"):
    """A folder that reads as a finished run: a results table in it."""
    folder = root / "results" / name
    folder.mkdir(parents=True)
    pd.DataFrame({"feature": ["a"], "coefficient": [0.3],
                  "p_value": [0.01]}).to_csv(folder / "results.csv",
                                             index=False)
    return str(folder)


def _marked(panel):
    """The run names carrying the loaded mark, in table order."""
    frame = panel._frame
    if frame is None or LOADED_COLUMN not in frame.columns:
        return []
    return [str(row["run"]) for _i, row in frame.iterrows()
            if str(row[LOADED_COLUMN]) == LOADED_MARK]


# --------------------------------------------------------------------------- #
#  A run that finishes is the run that is loaded
# --------------------------------------------------------------------------- #

def test_a_run_that_finishes_becomes_the_loaded_run(qtbot, tmp_path):
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    folder = _run_folder(tmp_path)
    handle = panel.record_run("run 10:00:00", "run", _settings(),
                              folder=folder)

    panel.update_run(handle, status="ok")

    assert panel.loaded_run()["run"] == "run 10:00:00"
    assert panel.loaded_run_folder() == folder
    assert _marked(panel) == ["run 10:00:00"]


def test_a_run_still_going_is_not_the_loaded_run(qtbot, tmp_path):
    """A row claiming to be loaded is a click that opens nothing."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.record_run("run 10:00:00", "run", _settings(),
                     folder=_run_folder(tmp_path))

    assert panel.loaded_run() is None
    assert _marked(panel) == []


def test_a_run_that_failed_is_not_the_loaded_run(qtbot):
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    handle = panel.record_run("run 10:00:00", "run", _settings())

    panel.update_run(handle, status="failed")

    assert panel.loaded_run() is None


def test_the_newest_finished_run_takes_over(qtbot, tmp_path):
    """"i just ran a regression so that should be loaded automatically"."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    first = panel.record_run("run A", "run", _settings(),
                             folder=_run_folder(tmp_path, "ols_1"))
    panel.update_run(first, status="ok")
    second = panel.record_run("run B", "run", _settings(),
                              folder=_run_folder(tmp_path, "ols_2"))
    assert panel.loaded_run()["run"] == "run A"

    panel.update_run(second, status="ok")

    assert panel.loaded_run()["run"] == "run B"
    assert _marked(panel) == ["run B"]


def test_starting_another_run_does_not_move_the_mark(qtbot, tmp_path):
    """Only FINISHING does. A run in flight has produced nothing to show."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    first = panel.record_run("run A", "run", _settings(),
                             folder=_run_folder(tmp_path, "ols_1"))
    panel.update_run(first, status="ok")

    panel.record_run("run B", "run", _settings())

    assert panel.loaded_run()["run"] == "run A"


# --------------------------------------------------------------------------- #
#  One run is the loaded run; several is a choice
# --------------------------------------------------------------------------- #

def test_one_trial_in_the_table_is_the_loaded_run(qtbot):
    """"if there is one run then that is the loaded"."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)

    panel.set_frame(_trials(1))

    assert panel.loaded_run()["run"] == "trial 1"


def test_several_runs_and_no_choice_is_said_rather_than_guessed(qtbot):
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)

    panel.set_frame(_trials(3))

    assert panel.loaded_run() is None
    assert "No run is loaded" in panel._status.text()


def test_picking_a_run_loads_it(qtbot):
    """The Runs tab carries the choice, per the request."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.set_frame(_trials(3))

    assert panel.set_loaded_run("trial 2") is True

    assert panel.loaded_run()["run"] == "trial 2"
    assert _marked(panel) == ["trial 2"]


def test_choosing_a_run_this_table_does_not_hold_is_refused(qtbot):
    """A tick against nothing reads as an answer."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.set_frame(_trials(3))
    panel.set_loaded_run("trial 2")

    assert panel.set_loaded_run("trial 99") is False
    assert panel.loaded_run()["run"] == "trial 2"


def test_choosing_a_run_shows_it(qtbot):
    """Moving the mark and leaving every view on the previous run is the
    failure 154 G is about."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.set_frame(_trials(3))
    activated = []
    panel.trial_activated.connect(lambda row: activated.append(row["run"]))

    panel.set_loaded_run("trial 2")

    assert activated == ["trial 2"]


def test_choosing_the_run_already_loaded_shows_it_once(qtbot):
    """A re-emit per call would re-load the results panel on every rebuild."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.set_frame(_trials(3))
    panel.set_loaded_run("trial 2")
    activated = []
    panel.trial_activated.connect(lambda row: activated.append(row["run"]))

    assert panel.set_loaded_run("trial 2") is True

    assert activated == []


def test_the_status_line_does_not_name_two_runs_at_once(qtbot, tmp_path):
    """"Loaded: ols_1. Loaded the run in .../ols_2." was one sentence naming
    two runs -- the note from the last load outliving the choice."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    first = _run_folder(tmp_path, "ols_1")
    second = _run_folder(tmp_path, "ols_2")
    panel.load_run_from_disk(first)
    panel.load_run_from_disk(second)

    panel.set_loaded_run(first)

    said = panel._status.text()
    assert "Loaded: ols_1" in said
    assert "ols_2" not in said, said


def test_the_choice_is_announced(qtbot):
    """A view that only learns about deliberate choices shows the wrong run
    after the common case."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.set_frame(_trials(3))
    seen = []
    panel.loaded_run_changed.connect(lambda row: seen.append(row["run"]))

    panel.set_loaded_run("trial 3")

    assert seen == ["trial 3"]


def test_the_status_line_names_the_loaded_run(qtbot):
    """The mark is in a column that can be sorted away; the sentence is not."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.set_frame(_trials(3))
    panel.set_loaded_run("trial 2")

    assert "Loaded: trial 2" in panel._status.text()


def test_a_mark_on_a_run_that_is_gone_does_not_survive_a_reload(qtbot):
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.set_frame(_trials(3))
    panel.set_loaded_run("trial 3")

    panel.set_frame(_trials(2))

    assert panel.loaded_run() is None
    assert panel._loaded_key == ""


def test_selecting_a_row_is_NOT_loading_it(qtbot):
    """INVERTED BY 190, and the inversion is the point.

    This used to assert that selection loaded, on the reasoning that the
    selection already re-pointed the results panel so the mark had to follow
    it. The maintainer asked for the opposite on 2026-08-20: "for some reason
    clicking once on a run shows the results. double click should loade the
    results". Both cannot be true.

    ARROWING DOWN A LIST OF FIVE RUNS LOADED FIVE RUNS -- five multi-second
    reads to look at five names. Selection now costs nothing; double-click is
    the gesture that costs time, and the mark follows THAT.
    """
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.set_frame(_trials(3))
    activated = []
    panel.trial_activated.connect(lambda row: activated.append(row["run"]))

    assert panel.table.select_key("trial 2")

    assert panel.loaded_run() is None
    assert activated == []


def test_but_double_clicking_it_is(qtbot):
    """The other half, so the pair cannot both be deleted as obsolete."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.set_frame(_trials(3))
    assert panel.table.select_key("trial 2")

    panel._on_double_click()

    assert panel.loaded_run()["run"] == "trial 2"


def test_recording_a_run_does_not_re_activate_the_loaded_one(qtbot, tmp_path):
    """`_rebuild` puts the highlight back, and that is this panel talking to
    itself -- not the user picking a row. Re-emitting would re-load the
    results panel on every recorded run."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    handle = panel.record_run("run A", "run", _settings(),
                              folder=_run_folder(tmp_path, "ols_1"))
    panel.update_run(handle, status="ok")
    activated = []
    panel.trial_activated.connect(lambda row: activated.append(row["run"]))

    panel.record_run("run B", "run", _settings())

    assert activated == []


# --------------------------------------------------------------------------- #
#  A run on disk is a first-class run
# --------------------------------------------------------------------------- #

def test_a_run_from_an_earlier_session_can_be_opened(qtbot, tmp_path):
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    folder = _run_folder(tmp_path, "ols_7")

    assert panel.load_run_from_disk(folder) is True

    record = panel.loaded_run()
    assert record["run"] == "ols_7"
    assert record["source"] == SOURCE_DISK
    assert panel.loaded_run_folder() == folder


def test_an_opened_run_is_shown_not_only_marked(qtbot, tmp_path):
    """A run marked loaded and displayed nowhere is a broken click."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    activated = []
    panel.trial_activated.connect(lambda row: activated.append(row["folder"]))
    folder = _run_folder(tmp_path, "ols_7")

    panel.load_run_from_disk(folder)

    assert activated == [folder]


def test_an_opened_run_carries_its_own_settings(qtbot, tmp_path):
    """Described by the SAME columns as a session run, or it cannot be
    compared with one -- which is what the tab is for."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    folder = _run_folder(tmp_path, "ols_7")
    # The file a run actually leaves beside its results -- `save_settings`'s
    # Key/Value CSV, which is what `settings_of_run` reads.
    pd.DataFrame({"Key": ["regression_type", "fdr_alpha"],
                  "Value": ["glm", 0.1]}).to_csv(
        os.path.join(folder, "regression_settings.csv"), index=False)

    panel.load_run_from_disk(folder)

    assert panel.loaded_run()["regression_type"] == "glm"


def test_a_folder_that_holds_no_run_says_so(qtbot, tmp_path):
    """The user picked a folder; being told nothing happened is the failure
    this repository keeps fixing."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    empty = tmp_path / "not_a_run"
    empty.mkdir()

    assert panel.load_run_from_disk(str(empty)) is False
    assert str(empty) in panel._status.text()
    assert "results.csv" in panel._status.text()


def test_opening_the_same_run_twice_is_one_row(qtbot, tmp_path):
    """A second row would be indistinguishable from a re-run."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    folder = _run_folder(tmp_path, "ols_7")

    panel.load_run_from_disk(folder)
    panel.load_run_from_disk(folder)

    assert list(panel._frame["run"]) == ["ols_7"]


def test_a_parent_folder_resolves_to_the_run_inside_it(qtbot, tmp_path):
    """`Load run…` accepts what a user actually has to hand."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    folder = _run_folder(tmp_path, "ols_7")

    assert panel.load_run_from_disk(str(tmp_path)) is True
    assert panel.loaded_run_folder() == folder


def test_the_button_is_there_and_does_not_open_a_dialog_when_given_a_folder(
        qtbot, tmp_path):
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)

    assert panel._open.text().startswith("Load run")
    # No QFileDialog is reached: a static modal runs its event loop in C++
    # and would hang a headless run, which is why the folder is a parameter.
    assert panel.load_run_from_disk(_run_folder(tmp_path, "ols_9")) is True


# --------------------------------------------------------------------------- #
#  The views that depend on it
# --------------------------------------------------------------------------- #

def test_the_montage_tab_names_the_run_it_is_describing(qtbot, tmp_path):
    from spacr.qt.widgets.cell_montage_view import CellMontageView

    folder = _run_folder(tmp_path, "ols_4")
    view = CellMontageView(results_provider=lambda: folder, threaded=False)
    qtbot.addWidget(view)

    assert view.loaded_run_name() == "ols_4"


def test_the_montage_status_says_which_run_it_would_use(qtbot, tmp_path):
    """A montage built from the wrong run looks exactly like one built from
    the right one, so the view says which run it is on."""
    from spacr.qt.widgets.cell_montage_view import CellMontageView

    folder = _run_folder(tmp_path, "ols_4")
    view = CellMontageView(results_provider=lambda: folder, threaded=False)
    qtbot.addWidget(view)
    view._key, view._name, view._effect = "fraction:grna[a_1]", "a_1", 0.3
    # The inputs a montage needs (a database with per-object rows) are
    # `spacr.cell_montage`'s subject, not this one's: what is under test is
    # the sentence a READY tab shows, so the refusals are stood down.
    view.reason = lambda: ""

    view._announce()

    said = view._status_text
    assert said.startswith("Ready")
    assert "a_1" in said
    assert "ols_4" in said, said


def test_no_run_loaded_says_how_to_load_one(qtbot):
    """"No regression results are loaded" named neither which run it wanted
    nor any way to give it one."""
    from spacr.qt.widgets.cell_montage_view import CellMontageView

    view = CellMontageView(results_provider=lambda: "", threaded=False)
    qtbot.addWidget(view)
    view._key, view._name, view._effect = "fraction:grna[a_1]", "a_1", 0.3

    said = view.reason()

    assert said == view.NO_RUN_LOADED
    assert "Runs tab" in said and "Load run" in said


def test_a_table_with_no_folder_is_a_different_sentence(qtbot):
    """The coefficients are on screen, so "no results are loaded" reads as a
    contradiction of what the user is looking at."""
    from spacr.qt.widgets.cell_montage_view import CellMontageView

    frame = pd.DataFrame({"feature": ["fraction:grna[a_1]"],
                          "coefficient": [0.3], "p_value": [0.01]})
    view = CellMontageView(frame_provider=lambda: frame,
                           results_provider=lambda: "", threaded=False)
    qtbot.addWidget(view)
    view._key, view._name, view._effect = "fraction:grna[a_1]", "a_1", 0.3

    assert view.reason() == view.RESULTS_WITHOUT_A_FOLDER
