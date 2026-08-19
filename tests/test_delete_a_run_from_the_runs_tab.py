"""Delete a run from the Runs tab, not only from the figures (146).

    "the user should be able to delete runs from the figures (currently
     possible) and from the run tab (not possible)"

THE ASYMMETRY. The figure grid has a "Clear figures" control and groups its
tiles by run, so a run can be got rid of from the pictures. `SweepRunsPanel`
had `load`, `reload`, `set_frame`, `record_run`, `update_run` and
`selected_trial` -- a run entered the table and stayed. The same run could be
dismissed from the pictures and not from the list of runs, which is the place
a user goes to manage them.

TWO DIFFERENT THINGS A USER COULD MEAN, and a single "Delete" that does not
distinguish them is how a screen's results are lost. Most of what is below is
about keeping them apart.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


def _run_folder(root, name, *, figures=2, tables=1, size=1024):
    folder = os.path.join(str(root), name)
    os.makedirs(folder, exist_ok=True)
    for index in range(figures):
        with open(os.path.join(folder, f"fig_{index}.png"), "wb") as handle:
            handle.write(b"x" * size)
    for index in range(tables):
        with open(os.path.join(folder, f"table_{index}.csv"), "w") as handle:
            handle.write("a,b\n1,2\n")
    return folder


def _panel(qtbot):
    from spacr.qt.widgets.sweep_runs import SweepRunsPanel

    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    return panel


def _select(panel, *names):
    """Select the rows whose `run` column holds these names.

    Through the selection MODEL, not `selectRow`: under ExtendedSelection
    `selectRow` behaves like a plain click and clears whatever was selected,
    so three calls to it leave one row selected and a multi-select test
    quietly checks the single-row case.
    """
    from PySide6.QtCore import QItemSelectionModel

    table = panel.table.table
    table.clearSelection()
    model = table.selectionModel()
    wanted = set(names)
    for row in range(table.rowCount()):
        for column in range(table.columnCount()):
            item = table.item(row, column)
            if item is None:
                continue
            index = item.data(0x0100)
            if index is None:
                continue
            record = panel._frame.iloc[int(index)].to_dict()
            if str(record.get("run")) in wanted:
                model.select(table.model().index(row, 0),
                             QItemSelectionModel.Select
                             | QItemSelectionModel.Rows)
            break


# ---------------------------------------------------------------------------
# A. what delete means, and it must be said
# ---------------------------------------------------------------------------

def test_removing_from_the_list_leaves_the_folder(qtbot, tmp_path):
    """REMOVE FROM THE LIST: the row goes, the folder on disk is untouched."""
    panel = _panel(qtbot)
    folder = _run_folder(tmp_path, "ols_1")
    handle = panel.record_run("ols_1", folder=folder)
    panel.update_run(handle, status="ok")
    assert len(panel._frame) == 1

    removed = panel.remove_runs(panel._all_rows())
    assert removed == 1
    assert panel._frame is None
    assert os.path.isdir(folder), "removing a row deleted the run folder"


def test_removing_says_reload_brings_it_back(qtbot, tmp_path):
    """Recoverable, so it needs no confirmation -- only the ability to get
    it back, said out loud (instruction 146 C)."""
    panel = _panel(qtbot)
    folder = _run_folder(tmp_path, "ols_1")
    panel.update_run(panel.record_run("ols_1", folder=folder), status="ok")
    panel.remove_runs(panel._all_rows())
    assert "Reload brings it back" in panel._status.text()


def test_reload_does_bring_a_removed_sweep_trial_back(qtbot, tmp_path):
    """The promise is kept, not merely printed."""
    from spacr.qt.widgets.sweep_runs import RESULTS_FILENAME

    panel = _panel(qtbot)
    pd.DataFrame([{"trial_id": 1, "status": "ok"},
                  {"trial_id": 2, "status": "ok"}]).to_csv(
        os.path.join(str(tmp_path), RESULTS_FILENAME), index=False)
    panel.load(str(tmp_path))
    assert len(panel._frame) == 2

    panel.remove_runs([row for row in panel._all_rows()
                       if row.get("run") == "trial 1"])
    assert len(panel._frame) == 1

    panel.reload()
    assert len(panel._frame) == 2


def test_deleting_from_disk_shows_the_path_and_what_is_in_it(qtbot,
                                                             tmp_path):
    """A user deciding whether to destroy an overnight fit needs to see what
    they are destroying, and a folder path alone is not that."""
    panel = _panel(qtbot)
    folder = _run_folder(tmp_path, "ols_1", figures=12, tables=4,
                         size=2 * 1024 * 1024)
    panel.update_run(panel.record_run("ols_1", folder=folder), status="ok")

    asked = {}

    def confirm(message, folders):
        asked["message"] = message
        asked["folders"] = folders
        return True

    assert panel.delete_runs_from_disk(panel._all_rows(), confirm=confirm) == 1
    assert folder in asked["message"]
    assert "12 figures" in asked["message"]
    assert "4 CSVs" in asked["message"]
    assert "MB" in asked["message"]
    assert "cannot be undone" in asked["message"]
    assert not os.path.exists(folder)


def test_a_refused_confirmation_deletes_nothing(qtbot, tmp_path):
    panel = _panel(qtbot)
    folder = _run_folder(tmp_path, "ols_1")
    panel.update_run(panel.record_run("ols_1", folder=folder), status="ok")

    assert panel.delete_runs_from_disk(panel._all_rows(),
                                       confirm=lambda *_a: False) == 0
    assert os.path.isdir(folder)
    assert len(panel._frame) == 1


def test_the_folder_description_is_the_words_a_decision_needs(tmp_path):
    from spacr.qt.widgets.sweep_runs import SweepRunsPanel

    folder = _run_folder(tmp_path, "ols_1", figures=1, tables=1, size=10)
    described = SweepRunsPanel.describe_folder(folder)
    assert "1 figure," in described and "1 CSV" in described
    assert SweepRunsPanel.describe_folder(
        str(tmp_path / "nowhere")) == "nothing on disk"


def test_a_running_run_is_refused_and_says_why(qtbot, tmp_path):
    """NEVER DELETE WHAT IS RUNNING. Refused with the reason, per 106 --
    not silently ignored."""
    panel = _panel(qtbot)
    folder = _run_folder(tmp_path, "ols_1")
    panel.record_run("ols_1", folder=folder)          # status: running

    assert panel.delete_runs_from_disk(panel._all_rows(),
                                       confirm=lambda *_a: True) == 0
    assert os.path.isdir(folder)
    said = panel._status.text()
    assert "still going" in said
    assert "Stop it first" in said


# ---------------------------------------------------------------------------
# B. the gestures
# ---------------------------------------------------------------------------

def test_the_delete_key_removes_the_selection(qtbot, tmp_path):
    panel = _panel(qtbot)
    folder = _run_folder(tmp_path, "ols_1")
    panel.update_run(panel.record_run("ols_1", folder=folder), status="ok")
    panel.update_run(panel.record_run("ols_2"), status="ok")
    _select(panel, "ols_1")

    panel.eventFilter(panel.table.table,
                      QKeyEvent(QEvent.KeyPress, Qt.Key_Delete,
                                Qt.NoModifier))
    assert [row["run"] for row in panel._all_rows()] == ["ols_2"]
    # THE SAFE HALF ON THE BARE KEY. Deleting from disk is a separate,
    # explicitly worded choice and a keystroke must not reach it.
    assert os.path.isdir(folder)


def test_the_delete_key_refuses_a_running_run(qtbot):
    panel = _panel(qtbot)
    panel.record_run("ols_1")                          # still going
    _select(panel, "ols_1")

    panel.eventFilter(panel.table.table,
                      QKeyEvent(QEvent.KeyPress, Qt.Key_Delete,
                                Qt.NoModifier))
    assert [row["run"] for row in panel._all_rows()] == ["ols_1"]
    assert "still going" in panel._status.text()


def test_several_rows_go_at_once(qtbot, tmp_path):
    """A sweep writes one folder per trial; cleaning up after one by hand
    twenty times is not a feature."""
    panel = _panel(qtbot)
    for index in range(4):
        panel.update_run(panel.record_run(f"ols_{index}"), status="ok")
    _select(panel, "ols_0", "ols_2", "ols_3")

    assert panel.remove_runs(panel.selected_runs()) == 3
    assert [row["run"] for row in panel._all_rows()] == ["ols_1"]


def test_the_context_menu_puts_the_safe_entry_first(qtbot, tmp_path):
    """A user reaching for a menu takes what is at the top, and what is at
    the top must not be the irreversible one.

    Read off the built menu rather than off a shown one: `QMenu.exec` is a
    C++ event loop, it cannot be monkeypatched off a PySide type, and a test
    that calls it hangs.
    """
    panel = _panel(qtbot)
    panel.update_run(panel.record_run("ols_1", folder=str(tmp_path)),
                     status="ok")
    _select(panel, "ols_1")

    from PySide6.QtWidgets import QAbstractItemView

    assert panel.table.table.selectionMode() == \
        QAbstractItemView.ExtendedSelection
    actions = [action
               for action in panel._build_run_menu(
                   panel.selected_runs()).actions()
               if not action.isSeparator()]
    verbs = [action.data() for action in actions]
    # LOAD IS FIRST since 2026-08-18, and the rule this test guards is
    # unchanged by that: the top entry must not be the irreversible one, and
    # Load is the least destructive entry in the menu -- it shows a run. It is
    # also what a user opens this menu FOR, which is why it went to the top;
    # the menu previously offered no way to load at all, so the only route to
    # another run was a single click.
    assert verbs[0] == "load"
    assert actions[0].text() == "Load this run"
    assert verbs[1] == "remove"
    assert actions[1].text().startswith("Remove 1 run from the list")
    # DESTRUCTIVE LAST, whatever else the menu grows. 116's "open beside"
    # arrived between the two and the ordering rule is the one that matters:
    # what is at the top must not be the irreversible one.
    assert verbs[-1] == "delete"
    assert actions[-1].text().startswith("Delete 1 run from disk")
    assert all(action.isEnabled() for action in actions)


def test_load_is_greyed_while_the_run_is_still_going(qtbot, tmp_path):
    """A run still going has no results, so loading it would show nothing.

    An empty screen under a mark claiming a run is worse than a disabled
    entry, which at least says why (instruction 106).
    """
    panel = _panel(qtbot)
    panel.record_run("ols_2", folder=str(tmp_path / "ols_2"))
    _select(panel, "ols_2")
    actions = {action.data(): action
               for action in panel._build_run_menu(panel.selected_runs()).actions()
               if action.data()}
    assert "load" in actions
    assert not actions["load"].isEnabled()
    assert "still going" in actions["load"].toolTip()
    # And the seam a test drives is guarded too, not only the paint.
    assert panel._apply_run_menu("load", panel.selected_runs()) is False


def test_the_menu_greys_both_entries_for_a_running_run(qtbot):
    panel = _panel(qtbot)
    panel.record_run("ols_1")
    _select(panel, "ols_1")

    actions = [action
               for action in panel._build_run_menu(
                   panel.selected_runs()).actions()
               if not action.isSeparator()]
    assert not [action for action in actions if action.isEnabled()], (
        "a running run offered a delete")
    assert any("still going" in action.toolTip() for action in actions)


# ---------------------------------------------------------------------------
# C. what leaves the list leaves the other views with it
# ---------------------------------------------------------------------------

def test_the_panel_announces_what_left(qtbot):
    panel = _panel(qtbot)
    panel.update_run(panel.record_run("ols_1", folder="/tmp/ols_1"),
                     status="ok")
    heard = []
    panel.runs_removed.connect(heard.append)

    panel.remove_runs(panel._all_rows())
    assert len(heard) == 1
    assert heard[0][0]["run"] == "ols_1"


def test_the_loaded_mark_does_not_survive_the_run_being_removed(qtbot):
    panel = _panel(qtbot)
    panel.update_run(panel.record_run("ols_1", folder="/tmp/ols_1"),
                     status="ok")
    assert panel.loaded_run() is not None

    panel.remove_runs(panel._all_rows())
    assert panel.loaded_run() is None
    assert panel._loaded_key == ""


def test_a_deleted_run_takes_its_retained_plot_state_with_it(qtbot,
                                                             tmp_path):
    """116's hook, and the trap it wrote down.

    Without this a later run written into the same folder inherits the
    deleted one's level and colouring, with nothing on screen saying where
    they came from.
    """
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    folder = _run_folder(tmp_path, "ols_1")
    panel._plot_states[panel._plot_state_key(folder)] = {"level": "gene"}

    assert panel.forget_run(folder) is True
    assert panel.remembered_runs() == ()


def test_deleting_the_run_on_screen_clears_the_panel_and_does_not_refile_it(
        qtbot, tmp_path):
    """The ORDER is the whole point.

    `set_frame` remembers the outgoing run's view on the way out, so
    forgetting first and clearing second files the deleted run again -- under
    the same key, with the state it was just relieved of.
    """
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    folder = _run_folder(tmp_path, "ols_1")
    panel.set_frame(pd.DataFrame({"feature": ["a"], "coefficient": [1.0],
                                  "p_value": [0.01]}), source=folder)
    assert panel.run_folder() == os.path.abspath(folder)

    assert panel.forget_run(folder) is True
    assert panel.remembered_runs() == (), (
        "the deleted run was filed again by the clear")
    assert panel.run_folder() == ""
    assert "has been deleted" in panel._status


def test_another_runs_state_is_left_alone(qtbot, tmp_path):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    one = _run_folder(tmp_path, "ols_1")
    two = _run_folder(tmp_path, "ols_2")
    panel._plot_states[panel._plot_state_key(one)] = {"level": "gene"}
    panel._plot_states[panel._plot_state_key(two)] = {"level": "grna"}

    panel.forget_run(one)
    assert panel.remembered_runs() == (panel._plot_state_key(two),)


def test_the_screen_wires_the_removal_to_the_results_panel(qtbot, tmp_path):
    """Driven on the real screen: the Runs tab's signal reaches the panel."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    runs = getattr(screen, "_sweep_runs", None)
    panel = getattr(screen, "_results_panel", None)
    assert runs is not None and panel is not None

    folder = _run_folder(tmp_path, "ols_1")
    panel._plot_states[panel._plot_state_key(folder)] = {"level": "gene"}
    runs.update_run(runs.record_run("ols_1", folder=folder), status="ok")

    runs.remove_runs(runs._all_rows())
    assert panel.remembered_runs() == (), (
        "the run left the table and its retained view did not")
