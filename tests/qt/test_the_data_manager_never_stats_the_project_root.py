"""The Data Manager asks about its project root without waiting for it.

THE DEFECT, 2026-09-04. Every enable/disable pass on this screen stat-ed the
user's project root inline:

    DataManagerScreen._update_controls
      -> os.path.isdir(self._root)

and `self._root` is exactly the path a user is most likely to keep on a
share: the plate folder they dropped or picked. `_update_controls` runs on
construction, on every prune-kind checkbox, on every job settle and on every
destination change, and `scan()` and `plan_prune()` stat-ed the same root
again as their guard. Measured on the maintainer's machine that day, a
single ``os.path.isdir`` under ``/nas_mnt`` -- an ``autofs`` mount whose
share was asleep -- had NOT RETURNED AFTER TWENTY SECONDS, because the stat
is what triggers the automount.

A stalled event loop is not a crash and leaves no traceback, which is why
this arrived as "opening map barcodes crashes spacr", hover flicker and
glimpses of other screens rather than as one bug.

What is asserted below is the property the freeze violated: these calls
return long before the filesystem does. The correction still has to arrive,
so the last test waits for a genuinely missing root to grey the buttons.
"""
from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("PySide6")

from spacr.qt import path_probe                              # noqa: E402
from spacr.qt.screens import data_manager as screen_module   # noqa: E402

pytestmark = pytest.mark.qt

#: Long enough that a single inline stat is unmissable in the timing,
#: shorter than the twenty seconds actually measured -- a test that waited
#: the real duration is a test nobody runs.
SLOW_S = 8.0

#: Shaped like the path that froze the maintainer's machine.
SLEEPING = "/nas_mnt/data/plate1"


@pytest.fixture(autouse=True)
def _fresh_cache():
    """No answer carried in from another test's probes."""
    path_probe.forget()
    yield
    path_probe.forget()


@pytest.fixture()
def screen(qtbot, qt_theme_applied):
    """The screen with no project, built before the filesystem goes slow."""
    widget = screen_module.DataManagerScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture()
def sleeping_share(monkeypatch):
    """Make every ``isdir`` behave like a stat on a share that is asleep.

    Patched on ``os.path`` itself rather than on the screen's reference to
    it, so an inline stat anywhere under the call -- including one moved
    into a helper -- is caught rather than stepped around.
    """
    def never(_path):
        time.sleep(SLOW_S)
        return True

    monkeypatch.setattr(os.path, "isdir", never)
    return never


def test_refreshing_the_controls_does_not_wait_for_the_root(
        screen, sleeping_share):
    """The call that ran on every settle, checkbox and drop."""
    screen._root = SLEEPING

    started = time.monotonic()
    screen._update_controls()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"_update_controls() took {elapsed:.1f}s -- it is stat-ing the "
        "project root on the GUI thread again, which is the freeze")
    assert screen.rescan_button.isEnabled(), (
        "an unprobed root must stay usable: the buttons it enables all hand "
        "the real question to a worker")


def test_a_scan_starts_without_stat_ing_the_root_first(screen,
                                                       sleeping_share):
    """The guard in `scan()` must not become the block the worker avoids."""
    calls = []
    screen._run = lambda fn, on_done: (calls.append(fn) or True)
    screen._root = SLEEPING

    started = time.monotonic()
    ok = screen.scan()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"scan() took {elapsed:.1f}s before reaching the worker -- its guard "
        "is stat-ing the root inline")
    assert ok and calls, (
        "the first scan of a freshly chosen folder was refused: the guard "
        "must answer optimistically while the probe is out")


def test_planning_a_prune_starts_without_stat_ing_the_root_first(
        screen, sleeping_share):
    """Same guard, same root, same thread."""
    calls = []
    screen._run = lambda fn, on_done: (calls.append(fn) or True)
    screen._root = SLEEPING

    started = time.monotonic()
    ok = screen.plan_prune()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"plan_prune() took {elapsed:.1f}s -- its guard is stat-ing the root")
    assert ok and calls, "the plan was refused on an unprobed root"


def test_the_folder_dialog_opens_at_home_rather_than_on_a_sleeping_mount(
        screen, sleeping_share, monkeypatch):
    """Handing the dialog a sleeping path moves the stat, it does not remove it.

    `path_probe.isdir` answers False for a path it has not probed, which is
    the pessimistic direction and the right one for a start directory:
    opening at home costs one click, opening on a mount that is asleep costs
    the application.
    """
    seen = {}

    def dialog(_parent, _caption, directory, *args, **kwargs):
        seen["directory"] = directory
        return ""

    monkeypatch.setattr(screen_module.QFileDialog, "getExistingDirectory",
                        staticmethod(dialog))
    screen._root = SLEEPING

    started = time.monotonic()
    screen.choose_project()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"choose_project() took {elapsed:.1f}s before the dialog opened")
    assert seen["directory"] != SLEEPING, (
        "an unprobed root was handed to QFileDialog, which stats it")


def test_a_root_that_is_gone_greys_the_buttons_once_the_probe_lands(
        screen, tmp_path, qtbot):
    """Optimism is only safe because the correction arrives.

    No slow filesystem here on purpose: this is the other half of the fix,
    and it is the half that keeps the user's warning.
    """
    screen._root = str(tmp_path / "never-created")
    screen._update_controls()
    assert screen.rescan_button.isEnabled(), "the optimistic answer is first"

    qtbot.waitUntil(lambda: not screen.rescan_button.isEnabled(),
                    timeout=10000)
    assert not screen.plan_button.isEnabled()


def test_closing_the_screen_lets_go_of_the_process_wide_signal(qtbot,
                                                               qt_theme_applied):
    """`probes.answered` outlives the screen; a live slot on a dead widget
    is what turns a redraw into a hard crash."""
    widget = screen_module.DataManagerScreen(threaded=False)
    qtbot.addWidget(widget)
    assert widget._path_probe_redraw is not None

    widget.close()
    assert widget._path_probe_redraw is None


# ---------------------------------------------------------------------------
# The sibling site: the confirmation dialog walked the project to open
# ---------------------------------------------------------------------------
#
# THE SECOND DEFECT, same screen, one step downstream of the first. The root
# was taken off the GUI thread and the delete confirmation still walked the
# whole project on it:
#
#     ConfirmDeleteDialog.__init__
#       -> describe()
#         -> PrunePlan.file_list()
#           -> data_manager._enumerate()
#             -> os.walk(candidate.path)          # every candidate, every file
#
# The plan deliberately does not carry its file list -- a project with
# millions of crops would carry millions of strings -- so that walk runs
# fresh every time the dialog is opened, on the thread that has to paint it.
# On the mount that started all of this the walk's FIRST stat is where the
# twenty seconds go, and the user is looking at a window that has not
# appeared while spaCR asks a sleeping share to wake up.
#
# The tests below hold the walk open on purpose. If it is back on the GUI
# thread the constructor cannot return until the walk does, so the elapsed
# time is the assertion; the rest is the promise that nothing was lost --
# the totals are up immediately, every path arrives, and Delete stays locked
# until it does.

import threading                                              # noqa: E402

from PySide6.QtWidgets import QDialogButtonBox                # noqa: E402

from spacr import data_manager as dm                          # noqa: E402


def _a_plan(root, *, files=("a.tif", "b.tif")):
    """A real one-candidate PrunePlan over a real folder.

    A stand-in would not do: the whole defect lives in
    :meth:`spacr.data_manager.PrunePlan.file_list`, which is a method of the
    real dataclass and is what the dialog calls.

    :param root: a folder to plan over; the candidate is ``root/measure``.
    :param files: names to create in the candidate folder.
    :returns: the plan.
    """
    candidate_dir = root / "measure"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        (candidate_dir / name).write_bytes(b"x" * 8)
    candidate = dm.PruneCandidate(
        path=str(candidate_dir), kind="measurements", module="measure",
        artifact_ids=("a1",), size_bytes=16, n_files=len(files),
        inventory_digest="d", regenerate_with="run measure again")
    return dm.PrunePlan(root=str(root), candidates=(candidate,),
                        total_bytes=16, total_files=len(files), token="t")


@pytest.fixture()
def held_walk(monkeypatch):
    """Make ``file_list`` block until the test lets it finish.

    Patched on the class, so it catches the call wherever it is made from.
    The wait is bounded by :data:`SLOW_S` rather than left open, so a walk
    that went back onto the GUI thread makes the timing assertion fail
    instead of hanging the whole session.
    """
    release = threading.Event()
    real = dm.PrunePlan.file_list

    def blocking(self):
        """Wait for the test, then enumerate for real."""
        release.wait(SLOW_S)
        return real(self)

    monkeypatch.setattr(dm.PrunePlan, "file_list", blocking)
    return release


def test_the_confirmation_opens_before_the_project_has_been_walked(
        qtbot, qt_theme_applied, tmp_path, held_walk):
    """The dialog the user is waiting for must not be the walk."""
    plan = _a_plan(tmp_path)

    started = time.monotonic()
    dialog = screen_module.ConfirmDeleteDialog(plan)
    elapsed = time.monotonic() - started
    qtbot.addWidget(dialog)

    assert elapsed < 1.0, (
        f"ConfirmDeleteDialog took {elapsed:.1f}s to build -- it is walking "
        "the project on the GUI thread again")
    text = dialog.listing.toPlainText()
    assert "Reading the disk" in text, "no placeholder while the walk is out"
    assert dm.human_bytes(plan.total_bytes) in text, (
        "the plan itself needs nothing from the disk and must be up at once")
    assert "run measure again" in text, "the candidate rows went missing"


def test_delete_stays_locked_until_the_file_list_is_on_screen(
        qtbot, qt_theme_applied, tmp_path, held_walk):
    """Later is allowed; unread is not.

    The acknowledgement says "I have read the list above". While the list is
    a placeholder there is nothing to have read, so the box that arms Delete
    must not move -- and it must be released the moment the list lands.
    """
    plan = _a_plan(tmp_path)
    dialog = screen_module.ConfirmDeleteDialog(plan)
    qtbot.addWidget(dialog)

    assert not dialog.acknowledged.isEnabled(), (
        "the user could acknowledge a file list that is not there yet")
    assert not dialog.buttons.button(QDialogButtonBox.Ok).isEnabled()

    held_walk.set()
    qtbot.waitUntil(dialog.acknowledged.isEnabled, timeout=10000)

    text = dialog.listing.toPlainText()
    expected, truncated = plan.file_list()
    assert expected and not truncated
    for path in expected:
        assert path in text, "a path the user was shown before is now missing"
    assert "Reading the disk" not in text
    dialog.acknowledged.setChecked(True)
    assert dialog.buttons.button(QDialogButtonBox.Ok).isEnabled()


def test_the_walk_never_claims_a_run_banner(qtbot, qt_theme_applied,
                                            tmp_path, held_walk):
    """Home filters its banners on ``user_visible``.

    Listing the files of one dialog is housekeeping for that dialog: the
    runner carries nothing else, and a "data_manager — running" banner for it
    would tell the user a run had started when none had.
    """
    dialog = screen_module.ConfirmDeleteDialog(_a_plan(tmp_path))
    qtbot.addWidget(dialog)

    assert dialog._runner is not None
    assert dialog._runner._user_visible is False


def test_a_walk_that_fails_does_not_leave_reading_the_disk_up(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """A result is delivered only for a job that succeeded.

    So a placeholder written before the job and cleared only in its
    completion handler stays on screen for good when the worker raises.
    Unthreaded, so the failure has landed by the time the constructor
    returns.
    """
    def raising(self):
        """Fail the way a walk over a share that has gone does."""
        raise OSError("the share went away")

    monkeypatch.setattr(dm.PrunePlan, "file_list", raising)
    dialog = screen_module.ConfirmDeleteDialog(_a_plan(tmp_path),
                                               threaded=False)
    qtbot.addWidget(dialog)

    text = dialog.listing.toPlainText()
    assert "Reading the disk" not in text, (
        "the placeholder outlived the walk that was meant to replace it")
    assert "the share went away" in text
    assert dm.human_bytes(16) in text, "the plan is still readable and true"
    assert dialog.acknowledged.isEnabled(), (
        "the user was locked out of a deletion they can still read the "
        "plan for")


def test_a_plan_with_nothing_to_delete_starts_no_thread_at_all(
        qtbot, qt_theme_applied):
    """``file_list`` over no candidates touches no disk.

    Sent to a worker anyway it would cost a thread to learn what an empty
    tuple already says, and the dialog would flash a placeholder for a list
    that is instantly complete.
    """
    dialog = screen_module.ConfirmDeleteDialog(dm.PrunePlan(root="/tmp"))
    qtbot.addWidget(dialog)

    assert dialog._runner is None
    assert "Reading the disk" not in dialog.listing.toPlainText()
    assert dialog.acknowledged.isEnabled()


def test_closing_the_dialog_retires_the_walk_it_started(
        qtbot, qt_theme_applied, tmp_path, held_walk):
    """Qt aborts the process if a running QThread is destroyed with its owner.

    Accept and Reject both go through ``done()`` and NEITHER sends a close
    event, which is why the teardown cannot live in ``closeEvent`` alone.
    """
    dialog = screen_module.ConfirmDeleteDialog(_a_plan(tmp_path))
    qtbot.addWidget(dialog)
    assert dialog._runner is not None

    dialog.reject()      # no close event is sent by this

    assert dialog._runner is None, (
        "the walk's thread outlived the dialog that owns it")
    held_walk.set()


def test_the_screen_hands_the_dialog_its_own_threading(screen, tmp_path,
                                                       monkeypatch):
    """An unthreaded screen must not open a dialog that is still reading.

    The screen's ``threaded`` flag is the seam every test drives it through;
    a dialog that ignored it would answer questions about a file list that
    had not arrived.
    """
    seen = {}

    class _Dialog:
        """Records how it was built and cancels."""

        def __init__(self, plan, parent=None, *, threaded=True):
            """Record the threading the screen asked for."""
            seen["threaded"] = threaded

        def exec(self):
            """Cancel, so nothing is deleted."""
            return 0

    monkeypatch.setattr(screen_module, "ConfirmDeleteDialog", _Dialog)
    screen._plan = _a_plan(tmp_path)

    assert screen.confirm_and_prune() is False
    assert seen["threaded"] is False


# ---------------------------------------------------------------------------
# The third defect: the pickers forgot where the user had just been
# ---------------------------------------------------------------------------
#
# Taking the start-directory stat off the GUI thread was right, and the first
# way it was written quietly changed what the user sees. `path_probe` keys its
# cache on (path, want_dir), because a path can exist and not be a directory:
#
#     path_probe.prime(chosen, True)   ->  writes  (chosen, False) = True
#     path_probe.isdir(chosen)         ->  reads   (chosen, True)  = MISSING
#
# so priming after the dialog answered a question nobody asks, and `isdir`
# answers False for a path it has not probed. The result was a picker that
# reopened at HOME having forgotten the folder the previous press landed on --
# every time for the archive destination, which nothing else on this screen
# ever probes with `want_dir=True`, and on a race for the project root, which
# `_update_controls` happens to probe and usually wins.
#
# "Just later" is the whole licence this exercise operates under. A remembered
# location that is silently dropped is not later, it is gone, so these tests
# hold the picker to what it did before the freeze was fixed -- while still
# refusing to hand a never-vouched-for path to QFileDialog.


@pytest.fixture()
def picker(monkeypatch):
    """Record every start directory QFileDialog is opened at.

    :returns: a callable ``picker(returns)`` that makes the dialog hand back
        ``returns``, and a list ``picker.starts`` of the directories it was
        opened at, in order.
    """
    starts = []
    answer = {"value": ""}

    def dialog(_parent, _caption, directory, *args, **kwargs):
        """Stand in for the folder chooser, recording where it opened."""
        starts.append(directory)
        return answer["value"]

    monkeypatch.setattr(screen_module.QFileDialog, "getExistingDirectory",
                        staticmethod(dialog))

    def configure(returns):
        """Set what the next dialog returns."""
        answer["value"] = returns
        return starts

    configure.starts = starts
    return configure


def test_the_destination_picker_reopens_where_the_user_last_left_it(
        screen, picker, tmp_path):
    """The regression the probe cache's second key introduced.

    Nothing else on this screen asks ``want_dir`` about the destination, so
    the pessimistic answer here had nothing to correct it: the picker opened
    at home on the second press, and on the tenth.
    """
    chosen = str(tmp_path / "archive")
    os.makedirs(chosen)
    starts = picker(chosen)

    screen.choose_destination()
    screen.choose_destination()

    assert starts[0] == os.path.expanduser("~"), (
        "with no destination yet the picker opens at home")
    assert starts[1] == chosen, (
        "the picker forgot the folder the user chose one press ago -- "
        "path_probe.prime records (path, False) and a start directory asks "
        "(path, True)")


def test_the_project_picker_reopens_where_the_user_last_left_it(
        screen, picker, tmp_path):
    """The same fault on the root, where a probe race had been hiding it."""
    chosen = str(tmp_path / "plate")
    os.makedirs(chosen)
    starts = picker(chosen)

    screen.choose_project()
    screen.choose_project()

    assert starts[1] == chosen, "the project picker forgot the last project"


def test_a_picker_never_opens_on_a_path_nobody_vouched_for(
        screen, picker, sleeping_share):
    """Remembering a picked folder must not become trusting every folder.

    A root restored from a previous session was never handed back by a
    dialog in THIS screen, so it is still only a hint, and handing it to
    QFileDialog is handing the freeze to the dialog instead.
    """
    picker("")
    screen._root = SLEEPING

    started = time.monotonic()
    screen.choose_project()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"choose_project() took {elapsed:.1f}s"
    assert picker.starts[0] == os.path.expanduser("~"), (
        "an unprobed root was handed to QFileDialog, which stats it")


def test_a_picked_folder_that_has_since_gone_falls_back_to_home(
        screen, picker, tmp_path, qtbot):
    """The optimistic default is a starting position, not a verdict.

    Once the background probe has actually answered, its answer is what
    ``path_probe`` returns, and a folder that has been deleted since the user
    picked it must stop being offered as the place to open.
    """
    chosen = str(tmp_path / "gone")
    os.makedirs(chosen)
    picker(chosen)
    screen.choose_destination()          # remembers it
    assert screen._dialog_start(chosen) == chosen

    os.rmdir(chosen)
    path_probe.forget(chosen)

    qtbot.waitUntil(
        lambda: screen._dialog_start(chosen) == os.path.expanduser("~"),
        timeout=10000)


def test_remembering_a_folder_costs_no_stat(screen, picker, monkeypatch):
    """The record is made from what the dialog returned, not by asking again.

    ``QFileDialog`` only hands back a directory it reached, so the stat has
    already happened; repeating it on the GUI thread is the exact call this
    screen exists to have removed.
    """
    gui_thread = threading.current_thread()

    def forbidden(_path):
        """Fail loudly rather than slowly, and only where it matters.

        `path_probe`'s own workers stat for a living; the rule this guards
        is that the GUI thread does not.
        """
        if threading.current_thread() is gui_thread:
            raise AssertionError("the GUI thread stat-ed the chosen folder")
        return True

    picker(SLEEPING)
    monkeypatch.setattr(os.path, "isdir", forbidden)
    monkeypatch.setattr(os.path, "exists", forbidden)

    screen.choose_destination()

    assert screen._dialog_start(SLEEPING) == SLEEPING, (
        "the folder the dialog just returned was not remembered")


def test_the_controls_redraw_only_for_this_screens_own_root(screen):
    """`probes.answered` is process-wide; this screen has one path in it.

    Every remembered path anywhere in spaCR comes through this signal --
    `file_list.py` alone probes the lot at start-up -- and `_update_controls`
    reads exactly one path back out of `path_probe`. Answering for the rest
    is a full enable pass to reach the conclusion it already had.
    """
    calls = []
    screen._root = "/some/root"
    screen._update_controls = lambda: calls.append(1)

    screen._path_probe_redraw("/an/unrelated/path", True)
    assert not calls, (
        "an answer about somebody else's path re-ran the enable pass")

    screen._path_probe_redraw("/some/root", False)
    assert calls, "the screen ignored the answer about its own root"


# ---------------------------------------------------------------------------
# The fourth defect: Cancel inherited the freeze the walk had given up
# ---------------------------------------------------------------------------
#
# Moving the walk to a thread gave the dialog a thread to retire, and the
# house teardown waits three seconds for one. That put a NEW freeze on the
# way out: press Cancel while the walk is still out and the GUI thread is
# held for the full budget before the window will close. Smaller than the
# twenty seconds this exercise started with, and the same defect.
#
# The wait could not have worked in any case. `bridge.drain_thread` stops a
# thread by asking its event loop to quit, and this worker is inside
# `os.walk`, which has no interruption point; the budget always ran out in
# full and always ended where it would have ended immediately, with the
# thread parked. Parking is what makes letting go safe -- the process-wide
# park list holds the reference Qt would otherwise see dropped on a running
# QThread -- so the correct wait is no wait.


def test_cancelling_a_dialog_does_not_wait_for_the_walk_to_finish(
        qtbot, qt_theme_applied, tmp_path, held_walk):
    """Closing must not block on work whose whole point was not to block.

    ``held_walk`` keeps the walk inside ``file_list`` for :data:`SLOW_S`, so
    a teardown that waits for it cannot return before the test's bound.
    """
    dialog = screen_module.ConfirmDeleteDialog(_a_plan(tmp_path))
    qtbot.addWidget(dialog)
    assert dialog._runner is not None

    started = time.monotonic()
    dialog.reject()
    elapsed = time.monotonic() - started

    held_walk.set()
    assert elapsed < 1.0, (
        f"reject() held the GUI thread for {elapsed:.1f}s waiting on a walk "
        "that cannot be interrupted -- the teardown budget is the new freeze")
    assert dialog._runner is None, (
        "the walk's thread outlived the dialog that owns it")


def test_a_walk_that_has_already_finished_is_reaped_rather_than_parked(
        qtbot, qt_theme_applied, tmp_path):
    """The grace is short, not zero, and this is what it buys.

    A dialog closed after its list arrived has a thread that is already on
    its way out; giving it a moment retires it normally instead of adding a
    parked thread to the process for no reason.
    """
    dialog = screen_module.ConfirmDeleteDialog(_a_plan(tmp_path))
    qtbot.addWidget(dialog)
    qtbot.waitUntil(dialog.acknowledged.isEnabled, timeout=10000)

    runner = dialog._runner
    dialog.reject()

    assert dialog._runner is None
    qtbot.waitUntil(lambda: runner.active_jobs() == 0, timeout=10000)
