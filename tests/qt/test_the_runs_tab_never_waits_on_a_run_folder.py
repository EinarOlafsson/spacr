"""The Runs tab never touches a run folder from the click that asks for it.

THE FREEZE, measured 2026-09-04 on the maintainer's machine: a single
``os.path.exists`` on a path under ``/nas_mnt`` -- an ``autofs`` mount whose
share was asleep -- had NOT RETURNED AFTER TWENTY SECONDS, because the stat
is what triggers the automount. A stalled event loop is not a crash and
leaves no traceback, which is why it was reported as "opening map barcodes
crashes spacr", as hover flicker, and as glimpses of other screens.

Every row on this tab carries a folder the user chose, and `SweepRunsPanel`
reached into those folders from five places on the GUI thread:

  * the "Load run…" chooser, which handed Qt the remembered folder as its
    start directory and then walked whatever came back (`find_results_table`);
  * `load`, which stat-ed and read ``sweep_results.csv`` on a tab change;
  * `_build_run_menu`, which asked `workspace.has_workspace` -- a bare
    ``Path.is_file`` -- before it could show the menu;
  * `delete_runs_from_disk`, which walked and sized every run folder to
    compose its confirmation, then ``shutil.rmtree``-d them;
  * the menu's "Save the state", which wrote a bundle into each folder.

Each test below makes one of those primitives park until the test lets it go
and then asserts the call came back anyway, in well under a second. The panel
is built with ``threaded=True`` because that is what the application gets;
under pytest it defaults to inline so the rest of the suite can still read
what these methods returned.
"""
from __future__ import annotations

import os
import shutil
import threading
import time

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from spacr.qt import path_probe                                  # noqa: E402
from spacr.qt.widgets.sweep_runs import SweepRunsPanel           # noqa: E402

#: Longer than any human would call responsive, far shorter than the twenty
#: seconds actually measured: a test that waited the real duration is a test
#: nobody runs.
SLOW_S = 6.0

#: What "did not block" means here. The work is handed to a worker, so the
#: call should return in microseconds; a tenth of a second is slack for a
#: loaded machine, and still a hundredth of what a sleeping mount costs.
FAST_S = 0.5


@pytest.fixture(autouse=True)
def _fresh_cache():
    """`path_probe` caches for the life of the process, so tests would leak."""
    path_probe.forget()
    yield
    path_probe.forget()


@pytest.fixture
def asleep():
    """A share that answers nothing until this test is finished with it.

    Set at teardown rather than left to time out, so the parked workers are
    released before the panel shuts them down and the file does not spend
    :data:`SLOW_S` per test winding threads down.
    """
    return threading.Event()


def _parked(released, answer=None):
    """A filesystem call that behaves like a stat on a sleeping automount."""
    def wait(*_args, **_kwargs):
        released.wait(SLOW_S)
        return answer
    return wait


@pytest.fixture
def panel(qtbot, asleep):
    # THE SHARE WAKES BEFORE THE PANEL CLOSES. pytest-qt closes the widgets it
    # was given from `pytest_runtest_teardown`, which runs BEFORE any fixture
    # finaliser -- so releasing the workers in this fixture's own teardown is
    # too late: `closeEvent` gets there first, finds a worker still inside its
    # stat, and `bridge.drain_thread` parks the QThread for the rest of the
    # session rather than terminating it (correctly -- see its docstring).
    # `before_close_func` is the one hook that runs on the near side of that.
    made = SweepRunsPanel(threaded=True)
    qtbot.addWidget(made, before_close_func=lambda _widget: asleep.set())
    yield made
    asleep.set()


def _a_run(root, name="ols_1"):
    """A folder shaped like a finished run."""
    folder = os.path.join(str(root), name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "results.csv"), "w") as handle:
        handle.write("feature,coefficient,p_value\na,1.0,0.01\n")
    return folder


# ---------------------------------------------------------------------------
# the chooser
# ---------------------------------------------------------------------------

def test_the_chooser_does_not_stat_the_remembered_folder(panel, monkeypatch,
                                                         asleep, qtbot):
    """Qt stats the start directory before it draws the dialog.

    So handing `QFileDialog` a remembered ``/nas_mnt`` path freezes the click
    that opened the chooser -- inside C++, where no amount of care in this
    file would help. The path is offered only once the cache has confirmed it.
    """
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(os.path, "isdir", _parked(asleep, True))
    panel._folder = "/nas_mnt/data/sequencing/seq_3"
    offered = []

    def chooser(_parent, _caption, start, *_args, **_kwargs):
        offered.append(start)
        return ""

    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(chooser))

    started = time.monotonic()
    panel.load_run_from_disk()
    elapsed = time.monotonic() - started

    assert elapsed < FAST_S, (
        f"opening the chooser took {elapsed:.1f}s -- something stat-ed the "
        f"remembered folder on the GUI thread")
    assert offered == [""], "a folder nothing has confirmed was handed to Qt"

    # AND THE FOLDER IS NOT LOST, which is the other half of the promise: the
    # answer lands in the cache and the next click opens where it left off.
    asleep.set()
    qtbot.waitUntil(lambda: path_probe.known(panel._folder, want_dir=True)
                    is True, timeout=5000)
    panel.load_run_from_disk()
    assert offered[-1] == panel._folder


def test_opening_a_run_does_not_walk_the_folder_first(panel, monkeypatch,
                                                      asleep, tmp_path):
    """`find_results_table` is an `os.walk` of a folder just chosen."""
    from spacr.qt.widgets import regression_results

    monkeypatch.setattr(regression_results, "find_results_table",
                        _parked(asleep, ""))
    folder = _a_run(tmp_path)

    started = time.monotonic()
    answered = panel.load_run_from_disk(folder)
    elapsed = time.monotonic() - started

    assert elapsed < FAST_S, (
        f"the search for a run took {elapsed:.1f}s on the calling thread")
    assert answered is True, "the search should report that it was started"
    assert not panel._open.isEnabled(), (
        "a second click would queue a second walk of the same folder")


# ---------------------------------------------------------------------------
# the sweep's own table, read on a tab change
# ---------------------------------------------------------------------------

def test_reading_the_sweep_table_does_not_block_the_tab_change(panel,
                                                               monkeypatch,
                                                               asleep,
                                                               tmp_path):
    """`AppScreen._on_results_tab_changed` calls this with the destination.

    Which is a folder the user typed, so the ``isfile`` in front of the read
    is the exact stat that did not come back after twenty seconds.
    """
    monkeypatch.setattr(os.path, "isfile", _parked(asleep, True))

    started = time.monotonic()
    answered = panel.load(str(tmp_path))
    elapsed = time.monotonic() - started

    assert elapsed < FAST_S, (
        f"the tab change waited {elapsed:.1f}s for the results table")
    assert answered is True, "the read should report that it was started"


# ---------------------------------------------------------------------------
# the row menu
# ---------------------------------------------------------------------------

def test_the_row_menu_is_built_without_asking_the_disk(panel, monkeypatch,
                                                       asleep):
    """A menu that cannot be drawn until a mount wakes is not a menu.

    The restore entry is offered optimistically while the answer is unknown,
    for the reason `path_probe` gives: an entry that is enabled and then
    quietly correct is better than one withdrawn because a share was slow.
    """
    from spacr import workspace

    monkeypatch.setattr(workspace, "has_workspace", _parked(asleep, True))
    records = [{"run": "ols_1", "status": "ok",
                "folder": "/nas_mnt/runs/ols_1"}]

    started = time.monotonic()
    menu = panel._build_run_menu(records)
    elapsed = time.monotonic() - started

    assert elapsed < FAST_S, (
        f"the right-click took {elapsed:.1f}s to produce a menu")
    entries = {action.data(): action for action in menu.actions()
               if action.data()}
    assert entries["restore"].isEnabled(), (
        "the restore entry was withdrawn because a mount was slow")


def test_saving_a_runs_state_does_not_write_from_the_menu_click(panel,
                                                                monkeypatch,
                                                                asleep,
                                                                tmp_path):
    """The bundle is written into the run folder, which is the slow part."""
    from spacr import workspace

    monkeypatch.setattr(workspace, "save_for_run", _parked(asleep, {}))
    records = [{"run": "ols_1", "status": "ok", "folder": _a_run(tmp_path)}]

    started = time.monotonic()
    answered = panel._apply_run_menu("save_state", records)
    elapsed = time.monotonic() - started

    assert elapsed < FAST_S, (
        f"the menu entry held the interface for {elapsed:.1f}s")
    assert answered is True, "the save should report that it was started"


# ---------------------------------------------------------------------------
# deleting from disk
# ---------------------------------------------------------------------------

def test_the_delete_confirmation_is_composed_off_the_gui_thread(panel,
                                                                monkeypatch,
                                                                asleep,
                                                                tmp_path):
    """The confirmation names what is in each folder, so it has to walk them.

    That walk used to happen in the click, BEFORE the modal appeared -- so
    the user got no dialog and no application either.
    """
    monkeypatch.setattr(os, "walk", _parked(asleep, []))
    monkeypatch.setattr(shutil, "rmtree", _parked(asleep, None))
    records = [{"run": "ols_1", "status": "ok", "folder": _a_run(tmp_path)}]
    asked = []

    started = time.monotonic()
    answered = panel.delete_runs_from_disk(
        records,
        confirm=lambda message, _folders: asked.append(message) or True)
    elapsed = time.monotonic() - started

    assert elapsed < FAST_S, (
        f"the delete spent {elapsed:.1f}s reading folders before asking")
    assert answered, "the delete should report that it was started"
    assert asked == [], (
        "the modal went up before the panel knew what it was deleting")


def test_the_delete_itself_leaves_the_gui_thread(panel, monkeypatch, asleep,
                                                 qtbot, tmp_path):
    """`shutil.rmtree` walks and unlinks every file under the run folder.

    On a sleeping mount it blocks at least as long as the walk that described
    it did, so a "yes" to the confirmation must not be the last thing the
    interface does.
    """
    monkeypatch.setattr(shutil, "rmtree", _parked(asleep, None))
    records = [{"run": "ols_1", "status": "ok", "folder": _a_run(tmp_path)}]
    asked = []

    started = time.monotonic()
    panel.delete_runs_from_disk(
        records,
        confirm=lambda message, _folders: asked.append(message) or True)
    elapsed = time.monotonic() - started

    assert elapsed < FAST_S, (
        f"the click that said yes waited {elapsed:.1f}s for the removal")

    # AND THE WORK STILL HAPPENS. The description is a job of its own and the
    # confirmation is shown from its callback, so the question arrives on the
    # next spin of the event loop -- which is also what proves the loop is
    # still spinning while the removal is under way.
    qtbot.waitUntil(lambda: bool(asked), timeout=5000)
    assert "ols_1" in asked[0], "the confirmation must still name the folder"

    started = time.monotonic()
    qtbot.wait(50)
    assert time.monotonic() - started < FAST_S, (
        "the event loop was blocked by the removal")


# ---------------------------------------------------------------------------
# the sentences the wait leaves behind
#
# Moving the work off the GUI thread means the click can no longer say what
# happened -- so a placeholder goes up in its place and something else has to
# take it down. `JobRunner._on_settled` calls `on_done` ONLY for a job that
# came back cleanly, and a modal the user declines produces no job at all, so
# "the answer will clear it" is not true of every path. Each test below is one
# path where nothing was clearing it and the tab's last words were a
# placeholder for the rest of the session.
# ---------------------------------------------------------------------------

def test_saying_no_to_the_delete_puts_the_status_line_back(panel, qtbot,
                                                           tmp_path):
    """Declining the confirmation used to leave the interface as it was.

    It still must. The walk that composes the confirmation now happens behind
    "Working out what these runs hold…", and a No means no worker is running,
    no arrival handler is coming and no `job_failed` will ever fire -- so
    unless the decline takes its own placeholder down, that sentence is the
    last thing this tab ever says.
    """
    panel._say("Nothing run yet.")
    records = [{"run": "ols_1", "status": "ok", "folder": _a_run(tmp_path)}]
    declined = []

    panel.delete_runs_from_disk(
        records,
        confirm=lambda message, _folders: (declined.append(message), False)[1])

    qtbot.waitUntil(lambda: bool(declined), timeout=5000)
    assert "Working out what these runs hold" not in panel._status.text(), (
        "the placeholder outlived the question it was standing in for")
    assert panel._source_note == "Nothing run yet.", (
        "declining the delete lost the sentence the line was already showing")


def test_a_finished_delete_is_an_answer_and_not_a_placeholder(panel, qtbot,
                                                              tmp_path):
    """"Deleted 1 run folder from disk." is the answer, not a stand-in.

    The delete writes two placeholders in a row and its arrival handler has to
    retire the second, or the next thing that writes to this line reads the
    finished sentence as one still waiting to be replaced.
    """
    records = [{"run": "ols_1", "status": "ok", "folder": _a_run(tmp_path)}]

    panel.delete_runs_from_disk(records,
                                confirm=lambda _message, _folders: True)
    qtbot.waitUntil(lambda: panel._source_note.startswith("Deleted"),
                    timeout=5000)

    # A LATER, UNRELATED FAILURE MUST NOT TOUCH IT. `job_failed` carries no
    # job id, so the only thing telling it whether this line is fair game is
    # whether a placeholder is outstanding -- and none is.
    panel._on_job_failed("some other job fell over")
    assert panel._source_note.startswith("Deleted"), (
        "a finished delete was overwritten by an unrelated job's failure")


def test_a_save_that_fails_takes_its_own_placeholder_down(panel, qtbot,
                                                          monkeypatch,
                                                          tmp_path):
    """The bundle write goes to a worker, and a worker can raise.

    `on_done` never runs for a job that threw, so "Saving the state of 1 run…"
    has to be a placeholder the failure handler can recognise -- written with
    `_start_waiting`, not with `_say`.
    """
    from spacr.qt.widgets import sweep_runs

    def the_share_went_away(_folders):
        raise OSError("the share went away")

    monkeypatch.setattr(sweep_runs, "save_run_states", the_share_went_away)
    records = [{"run": "ols_1", "status": "ok", "folder": _a_run(tmp_path)}]

    assert panel._apply_run_menu("save_state", records) is True
    assert "Saving the state" in panel._status.text(), (
        "the menu entry said nothing about what it had started")

    qtbot.waitUntil(
        lambda: "Saving the state" not in panel._source_note, timeout=5000)
    assert panel._source_note.startswith("That did not finish"), (
        "the save's placeholder was left standing when the worker raised")


def test_a_workspace_probe_answered_after_the_run_finished_is_dropped(
        panel, qtbot, monkeypatch, asleep, tmp_path):
    """A bundle appears when the run CLOSES, which is after the right-click.

    Right-clicking a run that is still going submits a `has_workspace` on a
    folder that has no bundle in it yet. `update_run` drops the cached answer
    when the run finishes for exactly that reason -- but the probe itself is
    still out on a slow mount, and if it is allowed to land it writes the same
    "no" straight back and greys the restore entry for the rest of the
    session.
    """
    from spacr import workspace

    folder = _a_run(tmp_path)
    monkeypatch.setattr(workspace, "has_workspace", _parked(asleep, False))
    handle = panel.record_run("ols_1", folder=folder)

    # The right-click while it was still going: optimistic now, probe out.
    assert panel._workspace_answer(folder) is True

    panel.update_run(handle, status="ok")
    asleep.set()
    qtbot.waitUntil(lambda: panel._jobs.pending_jobs() == 0, timeout=8000)

    assert folder not in panel._workspace_answers, (
        "a probe describing the run as it was before it finished was filed "
        "as the answer")

    records = [{"run": "ols_1", "status": "ok", "folder": folder}]
    entries = {action.data(): action
               for action in panel._build_run_menu(records).actions()
               if action.data()}
    assert entries["restore"].isEnabled(), (
        "the restore entry was greyed by a stale answer about a run that had "
        "not finished yet")


def test_the_second_placeholder_of_a_delete_remembers_the_first_sentence(
        panel, monkeypatch, asleep, qtbot, tmp_path):
    """A delete writes two placeholders, and only the first has a sentence.

    "Working out what these runs hold…" then "Deleting…" -- and by the time
    the second is written the status line is already showing the first, so
    capturing what it replaced a second time records a placeholder as the
    thing to go back to. The next decline then restores THAT, and the tab
    settles on a sentence about work that finished long ago.
    """
    monkeypatch.setattr(shutil, "rmtree", _parked(asleep, None))
    panel._say("Nothing run yet.")

    first = [{"run": "ols_1", "status": "ok", "folder": _a_run(tmp_path)}]
    panel.delete_runs_from_disk(first,
                                confirm=lambda _message, _folders: True)
    qtbot.waitUntil(lambda: panel._source_note == "Deleting…", timeout=5000)

    # A SECOND DELETE, DECLINED, while the first is still unlinking files.
    second = [{"run": "ols_2", "status": "ok",
               "folder": _a_run(tmp_path, "ols_2")}]
    declined = []
    panel.delete_runs_from_disk(
        second,
        confirm=lambda message, _folders: (declined.append(message), False)[1])
    qtbot.waitUntil(lambda: bool(declined), timeout=5000)

    assert panel._source_note == "Nothing run yet.", (
        "declining put back a placeholder instead of the sentence the line "
        f"actually had, and settled on {panel._source_note!r}")


def test_a_delete_whose_handler_throws_takes_its_placeholder_down(
        panel, qtbot, monkeypatch, tmp_path):
    """The arrival handler is code, and code raises.

    `JobRunner._on_settled` routes an exception from `on_done` to
    `job_failed` -- the same door a worker's own failure comes through -- so
    a handler that retires its placeholder BEFORE writing the sentence that
    replaces it hands `_on_job_failed` a line it can no longer see is
    outstanding. "Deleting…" then stands for the rest of the session, which
    is the one thing the placeholder was introduced to prevent.
    """
    def the_table_fell_over(self, *_args, **_kwargs):
        raise RuntimeError("the table fell over")

    monkeypatch.setattr(SweepRunsPanel, "remove_runs", the_table_fell_over)
    records = [{"run": "ols_1", "status": "ok", "folder": _a_run(tmp_path)}]

    panel.delete_runs_from_disk(records,
                                confirm=lambda _message, _folders: True)
    qtbot.waitUntil(lambda: panel._jobs.pending_jobs() == 0, timeout=8000)
    qtbot.waitUntil(
        lambda: panel._status.text().startswith("That did not finish"),
        timeout=5000)
    assert "Deleting" not in panel._status.text(), (
        "the delete's placeholder was left on the line by a handler that "
        "cleared it and then threw")


def test_a_save_whose_rebuild_throws_takes_its_placeholder_down(
        panel, qtbot, monkeypatch, tmp_path):
    """`_on_states_saved` rebuilds the table, and that is the throw.

    It is the one `_apply_run_menu` names as the reason the save's sentence
    has to be a placeholder at all, so it is the one the placeholder must
    actually survive.
    """
    def the_table_fell_over(self, *_args, **_kwargs):
        raise RuntimeError("the table fell over")

    monkeypatch.setattr(SweepRunsPanel, "_rebuild", the_table_fell_over)
    records = [{"run": "ols_1", "status": "ok", "folder": _a_run(tmp_path)}]

    assert panel._apply_run_menu("save_state", records) is True
    qtbot.waitUntil(lambda: panel._jobs.pending_jobs() == 0, timeout=8000)
    qtbot.waitUntil(
        lambda: panel._status.text().startswith("That did not finish"),
        timeout=5000)
    assert "Saving the state" not in panel._status.text(), (
        "the save's placeholder was left on the line by a handler that "
        "cleared it and then threw")


def test_the_older_workspace_probe_does_not_outrank_the_newer_one(
        panel, qtbot, monkeypatch, tmp_path):
    """Two probes for one folder, and the STALE one answers first.

    `update_run` drops the folder so the reply describing the run before it
    finished cannot be filed -- but the next right-click starts a second
    probe, and a guard that only asks "is this folder pending" sees the
    folder pending again and accepts the older answer, discarding the newer
    probe along with it. The restore entry is then greyed by a fact that
    stopped being true before the second question was even asked.
    """
    import threading

    from spacr import workspace

    folder = _a_run(tmp_path)
    handle = panel.record_run("ols_1", folder=folder)
    gates = [threading.Event(), threading.Event()]
    asked = []

    def probe(_target):
        turn = len(asked)
        asked.append(turn)
        gates[min(turn, 1)].wait(SLOW_S)
        # The run wrote its bundle between the two questions.
        return turn != 0

    monkeypatch.setattr(workspace, "has_workspace", probe)

    assert panel._workspace_answer(folder) is True       # the first probe
    qtbot.waitUntil(lambda: len(asked) == 1, timeout=5000)
    panel.update_run(handle, status="ok")                # the bundle appears
    assert panel._workspace_answer(folder) is True       # the second probe
    qtbot.waitUntil(lambda: len(asked) == 2, timeout=5000)

    gates[0].set()                                       # the stale one lands
    qtbot.wait(200)
    assert folder not in panel._workspace_answers, (
        "the reply from before the run finished was filed as the answer")

    gates[1].set()
    qtbot.waitUntil(lambda: panel._workspace_answers.get(folder) is True,
                    timeout=8000)

    records = [{"run": "ols_1", "status": "ok", "folder": folder}]
    entries = {action.data(): action
               for action in panel._build_run_menu(records).actions()
               if action.data()}
    assert entries["restore"].isEnabled(), (
        "the restore entry was greyed by an answer the newer probe had "
        "already corrected")
