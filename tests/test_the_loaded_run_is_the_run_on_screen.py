"""Instruction 157 — the loaded mark and the views cannot disagree.

Reported 2026-08-18:

    "i ran a mixed model and a ols model and eaven if the ols model is marked
     as loaded i think i still see the mixed results and no summary (because
     the ols is actually not loaded)"

`loaded_run_changed` was emitted from four places in the Runs tab and
connected in none, and the path that a user reaches most -- A RUN THAT
BECOMES LOADED BY FINISHING -- did not even emit: `update_run` moved the mark
and told nobody. The three DELIBERATE paths (a row clicked, a run chosen, a
folder opened) each announced themselves, which is why the model layer looked
right and the application did not.

So the tests below drive the FINISHING path first and hardest. That is the
one nothing pressed, and handoff section 0b is about exactly this: a green
model-layer suite over a control no test reaches.

Four rules:

* a run that finishes re-points the results, the figures and the summary, not
  only the mark;
* both entry points end in ONE function, so they cannot drift;
* a load that FAILS leaves the mark where it was -- the mark is a consequence
  of the run being shown;
* the results panel NAMES ITS OWN RUN, so a divergence is visible from the
  view that is wrong rather than by comparing two views.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

pytestmark = pytest.mark.qt

from spacr.qt.widgets.sweep_runs import (  # noqa: E402
    LOADED_COLUMN, LOADED_MARK, SweepRunsPanel,
)


def _coefficients(n, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feature": [f"fraction:grna[{seed}_{i}]" for i in range(n)],
        "coefficient": rng.normal(0, .5, n),
        "p_value": rng.uniform(size=n),
        "q_value": rng.uniform(size=n),
    })


def _run_folder(root, name, rows, seed):
    """A folder shaped like a finished run: a coefficient table in it."""
    folder = root / "results" / name
    folder.mkdir(parents=True, exist_ok=True)
    _coefficients(rows, seed).to_csv(folder / "results.csv", index=False)
    return str(folder)


def _marked(panel):
    frame = panel._frame
    if frame is None or LOADED_COLUMN not in frame.columns:
        return []
    return [str(row["run"]) for _i, row in frame.iterrows()
            if str(row[LOADED_COLUMN]) == LOADED_MARK]


# --------------------------------------------------------------------------- #
#  The Runs tab announces the run that finished
# --------------------------------------------------------------------------- #

def test_a_run_that_finishes_is_announced(qtbot, tmp_path):
    """THE PATH THAT WAS NEVER DRIVEN. `update_run` moved the mark and
    emitted nothing, so every view stayed on the previous run."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    folder = _run_folder(tmp_path, "ols_1", 5, 1)
    seen = []
    panel.loaded_run_changed.connect(lambda row: seen.append(row["run"]))
    handle = panel.record_run("run A", folder=folder)

    panel.update_run(handle, status="ok")

    assert seen == ["run A"]


def test_the_second_run_to_finish_takes_the_views_with_it(qtbot, tmp_path):
    """The reported sequence: run `mixed`, run `ols`. The mark moved and the
    results did not."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    first = panel.record_run("mixed", folder=_run_folder(
        tmp_path, "mixed_1", 5, 1))
    panel.update_run(first, status="ok")
    seen = []
    panel.loaded_run_changed.connect(lambda row: seen.append(row["run"]))
    second = panel.record_run("ols", folder=_run_folder(
        tmp_path, "ols_1", 7, 2))

    panel.update_run(second, status="ok")

    assert seen == ["ols"]
    assert _marked(panel) == ["ols"]


def test_a_run_that_fails_announces_nothing(qtbot, tmp_path):
    """It is not the loaded run, so nothing should be re-pointed at it."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    seen = []
    panel.loaded_run_changed.connect(lambda row: seen.append(row["run"]))
    handle = panel.record_run("run A", folder=str(tmp_path / "nothing"))

    panel.update_run(handle, status="failed")

    assert seen == []


def test_a_folder_holding_one_run_announces_it(qtbot):
    """"if there is one run then that is the loaded" -- a view that only
    learns about deliberate choices shows the wrong run after the common
    case."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    seen = []
    panel.loaded_run_changed.connect(lambda row: seen.append(row["run"]))

    panel.set_frame(pd.DataFrame({"trial_id": [1], "status": ["ok"]}))

    assert seen == ["trial 1"]


def test_every_path_announces_through_one_funnel(qtbot, tmp_path):
    """Clicking, choosing, opening and FINISHING all emit both names for the
    one event -- so a screen connecting either is on the same path."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    loaded, activated = [], []
    panel.loaded_run_changed.connect(lambda row: loaded.append(row["run"]))
    panel.trial_activated.connect(lambda row: activated.append(row["run"]))
    handle = panel.record_run("run A", folder=_run_folder(
        tmp_path, "ols_1", 5, 1))

    panel.update_run(handle, status="ok")
    panel.load_run_from_disk(_run_folder(tmp_path, "ols_2", 5, 2))

    assert loaded == ["run A", "ols_2"]
    assert activated == loaded


def test_a_load_that_failed_puts_the_mark_back(qtbot, tmp_path):
    """A mark on a run that is not on screen is the same disagreement,
    pointing the other way.

    Driven the way the screen drives it: the refusal comes back from the
    listener that was asked to show the run, inside the announcement.
    """
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.loaded_run_changed.connect(
        lambda row: row["run"] == "run B" and panel.the_load_failed(
            "run B has no saved results on disk."))
    first = panel.record_run("run A", folder=_run_folder(
        tmp_path, "ols_1", 5, 1))
    panel.update_run(first, status="ok")
    second = panel.record_run("run B", folder=str(tmp_path / "gone"))

    panel.update_run(second, status="ok")

    assert panel.loaded_run()["run"] == "run A"
    assert _marked(panel) == ["run A"]
    assert "no saved results" in panel._status.text()


def test_the_first_run_keeps_its_mark_even_when_it_cannot_be_drawn(
        qtbot, tmp_path):
    """There is nothing to go back to, and clearing the mark would answer "no
    run is loaded" right after a run the user watched finish -- 154 G's
    report, reached from the other direction."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.loaded_run_changed.connect(
        lambda row: panel.the_load_failed("run A wrote no results table."))

    handle = panel.record_run("run A", folder=str(tmp_path / "empty"))
    panel.update_run(handle, status="ok")

    assert panel.loaded_run()["run"] == "run A"
    assert _marked(panel) == ["run A"]


def test_the_undo_is_spent_on_the_announcement_it_answers(qtbot, tmp_path):
    """A refusal arriving two choices later must not drag the mark back.

    The window in which a load can fail is the announcement itself; a stale
    undo would move the mark to a run nobody chose and nothing is showing.
    """
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    first = panel.record_run("run A", folder=_run_folder(
        tmp_path, "ols_1", 5, 1))
    panel.update_run(first, status="ok")
    second = panel.record_run("run B", folder=_run_folder(
        tmp_path, "ols_2", 5, 2))
    panel.update_run(second, status="ok")

    # THE LOAD REPORTED SUCCESS, which is what spends the undo since the read
    # moved to a worker (instruction 159). Before that it was spent when the
    # announcement returned -- the same moment, while the listener read the run
    # synchronously, and the wrong moment once the answer arrives later.
    panel.the_load_succeeded()

    assert panel.the_load_failed("late") is False

    assert panel.loaded_run()["run"] == "run B"


# --------------------------------------------------------------------------- #
#  The results panel names its own run
# --------------------------------------------------------------------------- #

def test_the_results_panel_names_the_run_it_is_showing(qtbot, tmp_path):
    """The user could only tell the two had diverged by comparing views."""
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)

    assert panel.load(_run_folder(tmp_path, "ols_4", 6, 1)) is True

    assert panel.run_name() == "ols_4"
    assert panel._run_label.text() == "Run: ols_4"


def test_a_live_run_knows_its_folder(qtbot, tmp_path):
    """155 A: `perform_regression` hands `res_folder` back with the
    coefficients, so a run made IN the application is not a stray CSV."""
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    folder = _run_folder(tmp_path, "ols_5", 6, 1)

    # What `_on_pipeline_result` does: the frame from the run itself, and the
    # run's own folder as the source -- a DIRECTORY, not the CSV `load` finds.
    assert panel.set_frame(_coefficients(6, 1), source=folder) is True

    assert panel.run_folder() == os.path.abspath(folder)
    assert panel.run_name() == "ols_5"


def test_a_table_from_no_run_says_so_rather_than_naming_one(qtbot):
    """A frame handed straight in genuinely has no run folder, and that case
    keeps its own answer."""
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)

    assert panel.set_frame(_coefficients(6, 1)) is True

    assert panel.run_folder() == ""
    assert panel.run_name() == ""
    assert panel._run_label.text() == panel.NO_RUN_NAMED


# --------------------------------------------------------------------------- #
#  The real screen: the finishing path, end to end
# --------------------------------------------------------------------------- #

@pytest.fixture()
def screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    widget = AppScreen("regression")
    qtbot.addWidget(widget)
    assert widget._results_panel is not None
    assert widget._sweep_runs is not None
    return widget


def _settle(screen, timeout: float = 30.0) -> None:
    """Wait for an asynchronous run load to finish.

    Since instruction 159 the results panel reads a run on a worker, so the
    coefficients are not on screen when the call that started the read
    returns. Every assertion about WHAT is up has to come after this.
    """
    import time

    from PySide6.QtWidgets import QApplication

    panel = screen._results_panel
    loading = getattr(panel, "is_loading", lambda: False)

    def unsettled():
        # BOTH conditions. `is_loading` goes False when the WORKER finishes,
        # and the screen's own half -- the figures, and the rollback of the
        # loaded mark when the read failed -- runs afterwards on a queued
        # signal. Waiting on the worker alone returns before the rollback has
        # been dispatched, which is how a failed load looked like a successful
        # one for exactly one processEvents.
        return loading() or getattr(screen, "_pending_trial", None) is not None

    deadline = time.monotonic() + timeout
    while unsettled() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    QApplication.processEvents()
    assert not unsettled(), "the run was still settling when the wait expired"


def _finish(screen, label, folder):
    """Start a run and let it finish, the way the screen does it.

    AND WAIT FOR THE READ. Since instruction 159 the results panel loads a run
    on a worker, so the coefficients are not on screen when `update_run`
    returns -- the answer arrives through `load_finished`. Waiting here rather
    than in each test keeps every assertion about WHAT is on screen instead of
    about when: a test that polled the frame itself would pass on the previous
    run's table while the new one was still being read.
    """
    handle = screen._sweep_runs.record_run(label, folder=folder)
    screen._sweep_runs.update_run(handle, status="ok")
    _settle(screen)
    return handle


def test_the_run_that_finishes_replaces_the_one_on_screen(screen, tmp_path):
    """THE REPORTED BUG, through the real widgets. `mixed` wrote 30
    coefficients and `ols` wrote 70, so the length says which is up -- a
    length cannot be produced by accident from the other run's table."""
    _finish(screen, "mixed", _run_folder(tmp_path, "mixed_1", 30, 1))
    assert len(screen._results_panel.results_frame()) == 30

    _finish(screen, "ols", _run_folder(tmp_path, "ols_1", 70, 2))

    frame = screen._results_panel.results_frame()
    assert frame is not None and len(frame) == 70
    assert screen._results_panel.run_name() == "ols_1"
    assert _marked(screen._sweep_runs) == ["ols"]


def test_the_run_on_screen_and_the_mark_name_the_same_run(screen, tmp_path):
    """The two facts that were allowed to disagree."""
    _finish(screen, "mixed", _run_folder(tmp_path, "mixed_1", 30, 1))
    _finish(screen, "ols", _run_folder(tmp_path, "ols_1", 70, 2))

    assert (screen._results_panel.run_folder()
            == os.path.abspath(screen._sweep_runs.loaded_run_folder()))


def test_a_finished_run_whose_folder_is_gone_leaves_the_mark_alone(
        screen, tmp_path):
    """A failed load leaves the previous run on screen AND the mark on it."""
    _finish(screen, "mixed", _run_folder(tmp_path, "mixed_1", 30, 1))

    _finish(screen, "ols", str(tmp_path / "results" / "never_written"))

    assert len(screen._results_panel.results_frame()) == 30
    assert _marked(screen._sweep_runs) == ["mixed"]
    assert screen._results_panel.run_name() == "mixed_1"


def test_the_run_already_on_screen_is_not_re_read(screen, tmp_path, monkeypatch):
    """ONE PATH, and it costs one load. The Runs tab emits both names for the
    event; the live run's model, diagnostics and summary are already in the
    panel and re-reading the folder would replace them with what is on disk."""
    folder = _run_folder(tmp_path, "ols_1", 30, 1)
    panel = screen._results_panel
    panel.set_frame(_coefficients(30, 1), source=folder)
    loads = []
    monkeypatch.setattr(type(panel), "load",
                        lambda self, path: loads.append(path) or True)

    _finish(screen, "ols", folder)

    assert loads == []
    # THE PANEL IS INSIDE A SPLITTER since 7486d492 put two runs on screen at
    # once, so the tab's current widget is that splitter and not the panel.
    # Asserting the panel is REACHABLE from the current tab keeps what this
    # line was for -- the results tab is the one showing -- without pinning the
    # widget tree, which is what broke when the second run arrived.
    current = screen._results_tabs.currentWidget()
    assert current is panel or panel in current.findChildren(type(panel)), (
        f"the results panel is not on the current tab ({current!r})")


def test_the_montage_gets_the_folder_of_a_run_made_in_the_application(
        screen, tmp_path):
    """155 A: the application told the user it could not find a folder it had
    made ten seconds earlier."""
    folder = _run_folder(tmp_path, "ols_1", 30, 1)

    _finish(screen, "ols", folder)

    assert screen._results_source_path() == os.path.abspath(folder)


def test_the_screen_follows_the_loaded_run_signal(screen, tmp_path):
    """The missing connection, on its own.

    `loaded_run_changed` is the signal that says WHICH RUN IS LOADED. It was
    emitted from four places in the Runs tab and

        grep -rn "loaded_run_changed" spacr/qt/screens/  ->  NOTHING

    so the mark moved and nothing followed. Emitted alone here, because that
    is the only way to tell the connection from the one beside it.
    """
    folder = _run_folder(tmp_path, "ols_9", 42, 3)

    screen._sweep_runs.loaded_run_changed.emit(
        {"run": "ols", "status": "ok", "folder": folder})
    _settle(screen)

    frame = screen._results_panel.results_frame()
    assert frame is not None and len(frame) == 42
    assert screen._results_panel.run_name() == "ols_9"
