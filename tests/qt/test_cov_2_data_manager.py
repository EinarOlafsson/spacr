"""The Data Manager's guard rails: the dialogs, the refusals, the worker seam.

The screen owns the only call in spaCR that deletes a user's data, so every
path that leads to it is a path that has to be pinned. What is driven here is
what the happy-path tests cannot reach: a cancelled confirmation, a folder
picker the user closed, a second job asked for while one is running, a handler
that raises after the work is done, and the screen being closed with a thread
still alive.

The file dialogs and the message box are replaced by stand-ins that answer
without blocking -- a modal dialog in a headless run is a hang, and what is
being tested is what the screen does with the answer, not Qt's dialog.

Offscreen, CPU-only, offline.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal                    # noqa: E402
from PySide6.QtWidgets import QDialog                         # noqa: E402

from spacr import data_manager as dm                          # noqa: E402
from spacr.qt.screens import data_manager as screen_module    # noqa: E402

from tests.test_data_manager import (build_project,            # noqa: E402
                                     register_pipeline)

pytestmark = pytest.mark.qt


@pytest.fixture()
def project(tmp_path):
    """A registered project on disk."""
    root = str(tmp_path / "plate1")
    build_project(root)
    register_pipeline(root)
    return root


@pytest.fixture()
def screen(qtbot, project):
    """The screen, opened on the project, running inline."""
    widget = screen_module.DataManagerScreen(project=project, threaded=False)
    qtbot.addWidget(widget)
    return widget


class _Chooser:
    """A folder picker that answers ``path`` without opening anything."""

    def __init__(self, path):
        self.path = str(path)
        self.calls = []

    def getExistingDirectory(self, *args):     # noqa: N802 - Qt spelling
        self.calls.append(args)
        return self.path


def _answering(dialog_result):
    class _Dialog:
        """A confirmation dialog that answers without being shown."""

        def __init__(self, plan, parent=None, *, threaded=True):
            """Take the real dialog's signature, threading included.

            The real one enumerates the files off the GUI thread and is
            handed the screen's own ``threaded`` flag, so a stand-in that
            did not accept it would raise a TypeError instead of answering.
            """
            self.plan = plan
            self.threaded = threaded

        def exec(self):
            return dialog_result

    return _Dialog


#: The stand-in's answers. Integers because the screen ORs them together to
#: build the button set, exactly as ``QMessageBox.StandardButton`` allows.
_YES, _NO = 0x4000, 0x10000


def _message_box(answer):
    class _Box:
        Yes = _YES
        No = _NO
        asked = []

        @classmethod
        def question(cls, *args, **kwargs):
            cls.asked.append(args)
            return answer

    return _Box


# ---------------------------------------------------------------------------
# the folder pickers
# ---------------------------------------------------------------------------

def test_the_project_picker_scans_what_it_was_given(qtbot, monkeypatch,
                                                    project):
    """Choosing a folder is the same as setting it: it scans immediately.

    A picker that only recorded the path would leave the screen showing the
    previous project's numbers under the new project's name.
    """
    widget = screen_module.DataManagerScreen(threaded=False)
    qtbot.addWidget(widget)
    chooser = _Chooser(project)
    monkeypatch.setattr(screen_module, "QFileDialog", chooser)

    widget.choose_project()

    assert widget.project == project
    assert widget.usage is not None
    assert widget.usage.total_bytes > 0
    assert chooser.calls, "the picker was actually consulted"


def test_a_cancelled_project_picker_leaves_the_screen_where_it_was(screen,
                                                                   monkeypatch,
                                                                   project):
    """Cancelling must not blank the screen or rescan the empty string.

    ``set_project("")`` would clear every table and then try to measure the
    working directory, which is not the user's project.
    """
    before = screen.usage.total_bytes
    monkeypatch.setattr(screen_module, "QFileDialog", _Chooser(""))

    screen.choose_project()

    assert screen.project == project
    assert screen.usage.total_bytes == before


def test_the_destination_picker_sets_the_archive_target(screen, monkeypatch,
                                                        tmp_path):
    """The archive destination is chosen the same way and changes no files.

    Setting it must also drop any archive plan already made, because a plan
    names the destination it was made for.
    """
    target = str(tmp_path / "cold")
    monkeypatch.setattr(screen_module, "QFileDialog", _Chooser(target))

    screen.choose_destination()

    assert screen.destination_label.text() == target
    assert screen._archive_plan is None
    assert not os.path.exists(target), "choosing a destination moves nothing"

    monkeypatch.setattr(screen_module, "QFileDialog", _Chooser(""))
    screen.choose_destination()
    assert screen.destination_label.text() == target, "cancel keeps the target"


# ---------------------------------------------------------------------------
# what the usage table shows
# ---------------------------------------------------------------------------

def test_a_kind_with_no_bytes_and_no_registry_rows_is_not_a_row(qtbot,
                                                                tmp_path):
    """An empty file must not put an otherwise-absent kind in the table.

    A zero-byte file is enough to invent a kind bucket. Showing "model
    weights — 0 B" for a project that was never trained tells the user a
    stage ran that did not.
    """
    root = str(tmp_path / "plate1")
    build_project(root, crops=False, model=False)
    register_pipeline(root)
    os.makedirs(os.path.join(root, "model"), exist_ok=True)
    open(os.path.join(root, "model", "empty.pth"), "wb").close()

    widget = screen_module.DataManagerScreen(project=root, threaded=False)
    qtbot.addWidget(widget)

    empty_kinds = [row.label for row in widget.usage.kinds
                   if not row.size_bytes and not row.n_artifacts]
    assert empty_kinds, "the scan has to produce such a row for this to mean anything"
    shown = {widget.usage_table.item(r, 0).text()
             for r in range(widget.usage_table.rowCount())}
    assert not shown & set(empty_kinds)
    assert shown, "the kinds that do carry bytes are still listed"


def test_a_symlink_in_the_project_is_reported_and_not_followed(qtbot,
                                                               tmp_path):
    """A symlink is counted once, as a link, and said out loud.

    Following it would double-count the target -- or walk out of the project
    entirely -- and the user would be shown a size that is not what deleting
    the project would free.
    """
    root = str(tmp_path / "plate1")
    build_project(root)
    register_pipeline(root)
    os.symlink(os.path.join(root, "orig"), os.path.join(root, "shortcut"))

    widget = screen_module.DataManagerScreen(project=root, threaded=False)
    qtbot.addWidget(widget)

    assert len(widget.usage.symlinks) == 1
    assert "symlink" in widget.note_label.text()
    assert "not followed" in widget.note_label.text()


# ---------------------------------------------------------------------------
# planning refusals
# ---------------------------------------------------------------------------

def test_planning_without_a_project_asks_for_one(qtbot):
    """The Plan button is reachable before a folder is chosen.

    Running the planner on ``""`` would raise inside a click handler; the
    screen answers with the sentence that says what to do instead.
    """
    widget = screen_module.DataManagerScreen(threaded=False)
    qtbot.addWidget(widget)

    assert widget.plan_prune() is False
    assert "Choose a project folder first." in widget.note_label.text()
    assert widget.plan is None


def test_a_plan_with_nothing_to_delete_says_to_read_the_reasons(qtbot,
                                                                tmp_path):
    """An empty plan is a result, not a failure, and it has to explain itself.

    A project whose only contents are protected -- raw images and files with
    no registry row -- frees nothing. Leaving the previous "N bytes can be
    deleted" text on screen would be a lie about what the button now does.
    """
    root = str(tmp_path / "plate1")
    os.makedirs(os.path.join(root, "orig"))
    with open(os.path.join(root, "orig", "plate1_A01_1.tif"), "wb") as handle:
        handle.write(b"\x00" * 2048)

    widget = screen_module.DataManagerScreen(project=root, threaded=False)
    qtbot.addWidget(widget)
    widget.plan_prune()

    assert widget.plan is not None
    assert widget.plan.candidates == ()
    assert "Nothing here can be deleted safely" in widget.freed_label.text()
    assert widget.prune_table.rowCount() == 0


# ---------------------------------------------------------------------------
# the confirmation dialogs
# ---------------------------------------------------------------------------

def test_a_truncated_file_list_says_the_totals_still_cover_everything(
        qtbot, screen, monkeypatch):
    """A plan too large to list must not read as a plan that small.

    The dialog shows the files that would go. Past the cap the list is cut
    short, and the sentence that replaces the rest has to say that the totals
    above are still complete -- otherwise the user reads a truncated list as
    the whole deletion.

    The cap is lowered rather than the project grown to a hundred thousand
    files, which is the same branch reached in a tenth of a second.
    """
    screen.plan_prune()
    plan = screen.plan
    assert len(plan.file_list()[0]) > 1

    monkeypatch.setattr(dm, "MAX_RECORDED_FILES", 1)
    dialog = screen_module.ConfirmDeleteDialog(plan, screen)
    qtbot.addWidget(dialog)

    text = dialog.describe()
    files, truncated = plan.file_list()
    assert truncated and len(files) == 1
    assert "the list is cut short" in text
    assert "cover all of them" in text
    assert dm.human_bytes(plan.total_bytes) in text


def test_a_cancelled_confirmation_deletes_nothing(screen, project,
                                                  monkeypatch):
    """Closing the dialog is a decision, and the screen has to record it.

    Nothing may be removed, and the note has to say so -- a silent no-op is
    indistinguishable from a deletion that failed.
    """
    screen.plan_prune()
    monkeypatch.setattr(screen_module, "ConfirmDeleteDialog",
                        _answering(QDialog.Rejected))
    before = sorted(
        os.path.join(folder, name)
        for folder, _dirs, names in os.walk(project) for name in names)

    assert screen.confirm_and_prune() is False

    assert "Nothing was deleted." in screen.note_label.text()
    after = sorted(
        os.path.join(folder, name)
        for folder, _dirs, names in os.walk(project) for name in names)
    assert after == before
    assert screen.plan is not None, "the plan survives a cancel"


def test_an_accepted_confirmation_deletes_exactly_the_plan(screen, project,
                                                           monkeypatch):
    """The accepted dialog runs the plan the dialog was built from.

    The files the plan listed go, the originals stay, and the screen drops the
    spent plan so the delete button cannot be pressed twice.
    """
    screen.plan_prune()
    plan = screen.plan
    planned, _ = plan.file_list()
    assert planned
    monkeypatch.setattr(screen_module, "ConfirmDeleteDialog",
                        _answering(QDialog.Accepted))

    assert screen.confirm_and_prune() is True

    assert not any(os.path.exists(path) for path in planned)
    assert os.path.isdir(os.path.join(project, "orig"))
    assert screen.plan is None
    assert "Freed" in screen.freed_label.text()


def test_confirm_and_prune_does_nothing_without_a_plan(screen, monkeypatch):
    """No plan means no dialog, not an empty one.

    Opening a confirmation for nothing would ask the user to acknowledge a
    deletion of zero files.
    """
    class _Forbidden:
        def __init__(self, plan, parent=None):
            raise AssertionError("a confirmation opened with no plan")

    monkeypatch.setattr(screen_module, "ConfirmDeleteDialog", _Forbidden)
    assert screen.plan is None
    assert screen.confirm_and_prune() is False


# ---------------------------------------------------------------------------
# archiving
# ---------------------------------------------------------------------------

def test_declining_the_archive_question_moves_nothing(screen, project,
                                                      tmp_path, monkeypatch):
    """"No" leaves the project where it is, and says so.

    An archive is a move: an accidental yes relocates somebody's plate. The
    question is asked once and the default is No.
    """
    destination = str(tmp_path / "cold")
    screen.set_destination(destination)
    screen.plan_archive()
    assert screen._archive_plan is not None

    box = _message_box(_NO)
    monkeypatch.setattr(screen_module, "QMessageBox", box)

    assert screen.confirm_and_archive() is False

    assert "Nothing was moved." in screen.note_label.text()
    assert box.asked, "the user was actually asked"
    assert os.path.isdir(os.path.join(project, "merged"))
    assert not os.path.exists(destination)


def test_accepting_the_archive_question_moves_and_leaves_the_record(
        screen, project, tmp_path, monkeypatch):
    """"Yes" moves the files and reports where the record was left.

    The record is the whole point of archiving rather than copying: the note
    has to name the ledger, because that is what the user needs to find the
    data again.
    """
    destination = str(tmp_path / "cold")
    screen.set_destination(destination)
    screen.plan_archive()
    monkeypatch.setattr(screen_module, "QMessageBox", _message_box(_YES))
    # The archive handler re-scans the project it just emptied, and the scan
    # writes its own note, so the notes are collected as they are set rather
    # than read off the label afterwards.
    said = []
    original_note = screen._note
    monkeypatch.setattr(screen, "_note",
                        lambda text, warn=False: (said.append(text),
                                                  original_note(text,
                                                                warn=warn))[0])

    assert screen.confirm_and_archive() is True

    assert os.path.isdir(os.path.join(destination, "merged"))
    assert not os.path.exists(os.path.join(project, "merged"))
    moved = [line for line in said if line.startswith("Moved")]
    assert len(moved) == 1
    assert destination in moved[0]
    assert "Record left at" in moved[0]
    assert screen._archive_plan is None
    assert screen.archive_table.rowCount() == 0


def test_confirm_and_archive_does_nothing_without_a_plan(screen, monkeypatch):
    """No archive plan means the question is never asked."""
    class _Forbidden:
        @staticmethod
        def question(*_args, **_kwargs):
            raise AssertionError("asked to archive with no plan")

    monkeypatch.setattr(screen_module, "QMessageBox", _Forbidden)
    assert screen._archive_plan is None
    assert screen.confirm_and_archive() is False


# ---------------------------------------------------------------------------
# the job seam
# ---------------------------------------------------------------------------

def test_a_second_job_is_refused_while_one_is_running(screen):
    """One job at a time: the screen's tables are filled by the handler.

    Two overlapping scans would interleave their table writes. The refusal is
    reported by the return value rather than by an exception, because it is
    a normal thing for a double-click to do.
    """
    screen._set_busy(True)
    try:
        assert screen.scan() is False
        assert screen.plan_prune() is False
    finally:
        screen._set_busy(False)
    assert screen.scan() is True


def test_a_handler_that_raises_after_the_work_is_reported_not_swallowed(
        screen):
    """The job succeeded; drawing its result did not. Both facts are kept.

    ``job_finished`` must carry False so a caller waiting on it is not told
    the screen is showing something it is not, and the message has to reach
    the note strip -- there is no console in a packaged build.
    """
    seen = []
    screen.job_finished.connect(seen.append)

    def _explodes(_result):
        raise RuntimeError("the table refused the row")

    ok = screen._run(lambda: "the result", _explodes)

    assert ok is True, "the work itself ran"
    assert seen == [False], "but the job is reported as failed"
    assert "the table refused the row" in screen.note_label.text()
    assert screen._busy is False, "and the screen is usable again"


class _StubWorker(QObject):
    """Stands in for ``PipelineWorker``: the two signals the screen connects."""

    error = Signal(str)
    finished = Signal(bool)


class _StubThread(QObject):
    """A thread stand-in that runs the job body on the calling thread.

    Coverage cannot see inside a ``QThread``, and a real one would also make
    the assertions racy. Running the same callable here drives the identical
    code with the identical signals.
    """

    finished = Signal()

    def __init__(self, fn, settings, worker):
        super().__init__()
        self._fn = fn
        self._settings = settings
        self.worker = worker
        self.quit_calls = 0
        self.waited = []

    def start(self):
        self._fn(self._settings)
        self.worker.finished.emit(True)

    def retire(self):
        """Emit what a real thread emits when its event loop ends."""
        self.finished.emit()

    def quit(self):
        self.quit_calls += 1

    def wait(self, msecs=0):
        self.waited.append(msecs)
        return True


@pytest.fixture()
def stub_threads(monkeypatch):
    """Replace ``make_thread`` so the threaded path runs inline."""
    made = []

    def _make_thread(fn, settings, app_key="", **_kwargs):
        worker = _StubWorker()
        thread = _StubThread(fn, settings, worker)
        made.append(thread)
        return thread, worker

    from spacr.qt import bridge
    monkeypatch.setattr(bridge, "make_thread", _make_thread)
    return made


def test_the_threaded_path_runs_the_same_job_and_settles_it(qtbot, project,
                                                            stub_threads):
    """The body handed to the worker computes the result the handler draws.

    The threaded and inline paths must not diverge: the same callable, the
    same box, the same handler. What the worker thread does is put the
    result where ``_job_settled`` looks for it.
    """
    widget = screen_module.DataManagerScreen(threaded=True)
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.job_finished, timeout=5000) as blocker:
        widget.set_project(project)

    assert blocker.args == [True]
    assert stub_threads, "the threaded path was taken"
    assert widget.usage is not None
    assert widget.usage.total_bytes > 0
    assert widget.usage_table.rowCount() > 0


def test_a_worker_error_reaches_the_note_strip_as_its_last_line(qtbot, project,
                                                                stub_threads):
    """A traceback from the worker is shown as the line that matters.

    The screen has no console. Showing the whole traceback in a one-line
    label would show the user the first frame of it; the exception line is
    the part that says what went wrong.
    """
    widget = screen_module.DataManagerScreen(threaded=True)
    qtbot.addWidget(widget)
    widget.set_project(project)
    thread = stub_threads[-1]

    thread.worker.error.emit(
        "Traceback (most recent call last):\n  File x\nOSError: disk is full")
    assert widget.note_label.text() == "OSError: disk is full"

    thread.worker.error.emit("")
    assert widget.note_label.text() == "unknown error"


def test_closing_the_screen_stops_a_thread_that_is_still_alive(qtbot, project,
                                                               stub_threads):
    """A QThread collected while running takes the process with it.

    ``closeEvent`` is the last chance to stop one, so a job still on the
    books has to be asked to quit and waited for -- not merely dropped.
    """
    widget = screen_module.DataManagerScreen(threaded=True)
    qtbot.addWidget(widget)
    widget.set_project(project)
    thread = stub_threads[-1]
    assert widget._jobs, "the job is still on the books until the thread retires"

    widget.close()

    assert thread.quit_calls == 1
    assert thread.waited == [2000]
    assert widget._jobs == []
