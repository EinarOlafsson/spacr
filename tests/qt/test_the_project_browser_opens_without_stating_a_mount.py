"""Opening the Project Browser never stats a remembered folder inline.

THE FREEZE, 2026-09-04. `make_project_browser_screen` is the factory the
registry hands to `MainWindow._build_screen`, and that build cannot leave
the GUI thread -- Qt forbids making widgets anywhere else. The factory
seeded its search folders like this:

    remembered = get_recent_sources("project_browser") + ...("mask")
    roots = tuple(p for p in remembered if p and os.path.isdir(p))

and those are folders the user last pointed some other screen at. One of
the maintainer's was under ``/nas_mnt``, an ``autofs`` mount whose share was
asleep, and a single ``os.path.isdir`` on it had NOT RETURNED AFTER TWENTY
SECONDS -- the stat is what triggers the automount. So clicking "Project
Browser" in the sidebar froze the whole application, with no traceback,
because a stalled event loop is not a crash.

The fix asks :mod:`spacr.qt.path_probe` instead, optimistically: a folder
nobody has probed yet is seeded anyway and checked in the background. That
is safe precisely here, because the walk those roots feed runs on a
`JobRunner` worker -- a root that turns out to be gone is discovered off the
GUI thread and just lists no projects.

Two properties, and both matter: the factory must return at once (below),
and it must still seed the folders it always did, because a browser that
opens empty is the defect this seeding was written to prevent.

AND THE SIBLING SITE, one button along. Once the browser is open, "Add
folder…" opened its dialog on ``self._roots[-1]`` -- a remembered root, and
on the maintainer's machine the very same sleeping ``/nas_mnt`` path. Qt
stats and then LISTS the start directory before it draws the dialog, so
seeding without blocking bought nothing if the first click on the chooser
froze instead. `ProjectBrowserScreen._start_directory` asks the probe cache
there too, pessimistically this time: an unprobed folder is not offered,
because offering it is what stats it.

AND THE LATE FAILURE. `JobRunner.job_failed` is not generation-guarded, so a
first walk that fails slowly used to overwrite the second walk's finished
table. `_scan` returns its failure as a value, through the generation-guarded
completion handler, so a superseded walk's error is dropped with its result.
"""
from __future__ import annotations

import os
import threading
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog

from spacr.projects import ProjectSummary
from spacr.qt import path_probe
from spacr.qt.screens import project_browser as module
from spacr.qt.screens.project_browser import (
    APP_KEY, ProjectBrowserScreen, make_project_browser_screen)

pytestmark = pytest.mark.qt

#: The mount that will not answer. Everything under it sleeps.
ASLEEP = "/nas_asleep"

#: Longer than any human would call responsive, far shorter than the twenty
#: seconds actually measured. A test that waited the real duration is a test
#: nobody runs.
SLOW_S = 8.0


@pytest.fixture(autouse=True)
def _fresh_cache():
    path_probe.forget()
    yield
    path_probe.forget()


@pytest.fixture
def sleeping_mount(monkeypatch):
    """Make ``isdir`` under :data:`ASLEEP` take :data:`SLOW_S`, as autofs did.

    Patched on ``os.path`` itself, which is what both the old inline filter
    and `path_probe`'s background worker call -- so the unfixed factory
    sleeps and the fixed one hands the sleep to a probe thread. Paths outside
    the fake mount still get the real answer: a blanket patch would slow the
    theme, the drop handler and every other stat the screen makes on the way
    up, and this test would then be measuring the wrong thing.
    """
    real_isdir = os.path.isdir

    def isdir(path):
        if str(path).startswith(ASLEEP):
            time.sleep(SLOW_S)
            return True
        return real_isdir(path)

    monkeypatch.setattr(path_probe.os.path, "isdir", isdir)
    return isdir


@pytest.fixture
def no_walk(monkeypatch):
    """Stop the screen's first scan from walking the fake mount for real.

    The scan is already off the GUI thread and is not what is under test;
    stubbing it keeps the worker from parking on a path that does not exist.
    """
    import spacr.projects as projects

    monkeypatch.setattr(projects, "browse", lambda *a, **k: ())


def _remember(monkeypatch, project_paths, mask_paths=()):
    """Seed what the user last worked in, without touching QSettings."""
    import spacr.qt.prefs as prefs

    def get_recent_sources(key, limit=4):
        if key == APP_KEY:
            return list(project_paths)[:limit]
        if key == "mask":
            return list(mask_paths)[:limit]
        return []

    monkeypatch.setattr(prefs, "get_recent_sources", get_recent_sources)


def _close(qtbot, widget):
    qtbot.addWidget(widget)
    widget.close()


def test_the_factory_returns_before_a_sleeping_mount_answers(
        qtbot, monkeypatch, sleeping_mount, no_walk, qt_theme_applied):
    """The property the freeze violated: opening the screen does not wait."""
    _remember(monkeypatch, [f"{ASLEEP}/data/experiment/plate1"],
              [f"{ASLEEP}/data/masks"])

    started = time.monotonic()
    widget = make_project_browser_screen()
    elapsed = time.monotonic() - started
    _close(qtbot, widget)

    assert isinstance(widget, ProjectBrowserScreen)
    assert elapsed < 1.0, (
        f"the factory took {elapsed:.1f}s -- it is stat-ing remembered "
        "folders on the GUI thread again, which is the freeze")


def test_the_remembered_folders_are_still_seeded(
        qtbot, monkeypatch, sleeping_mount, no_walk, qt_theme_applied):
    """Not blocking is only correct if the browser still opens on something.

    `path_probe.isdir` would answer False for a folder it has not probed,
    which on the first open of every session is every folder -- an empty
    browser, which is the whole thing this seeding exists to prevent.
    """
    _remember(monkeypatch, [f"{ASLEEP}/data/experiment/plate1"],
              [f"{ASLEEP}/data/masks"])

    widget = make_project_browser_screen()
    roots = widget.roots()
    _close(qtbot, widget)

    assert f"{ASLEEP}/data/experiment" in roots, (
        "the parent of the last project folder is no longer seeded")
    assert f"{ASLEEP}/data/masks" in roots


def test_a_folder_known_to_be_gone_is_still_dropped(
        qtbot, monkeypatch, tmp_path, no_walk, qt_theme_applied):
    """Optimism is temporary: once the probe has answered, it is obeyed.

    The filter still does its job -- a remembered folder that has since been
    deleted does not become a search root -- it just learns the answer from a
    probe thread rather than from the GUI thread.
    """
    present = tmp_path / "still-here"
    present.mkdir()
    gone = tmp_path / "deleted-last-week"

    path_probe.exists(gone, want_dir=True)
    qtbot.waitUntil(
        lambda: path_probe.known(gone, want_dir=True) is not None,
        timeout=5000)

    _remember(monkeypatch, [], [str(present), str(gone)])

    widget = make_project_browser_screen()
    roots = widget.roots()
    _close(qtbot, widget)

    assert str(present) in roots
    assert str(gone) not in roots


# ---------------------------------------------------------------------------
# The sibling site: the folder chooser's start directory
# ---------------------------------------------------------------------------

def test_the_folder_chooser_does_not_open_on_a_sleeping_root(
        qtbot, monkeypatch, sleeping_mount, no_walk, qt_theme_applied):
    """Opening "Add folder…" must not wait on the root it would start in.

    The stand-in chooser stats its start directory, which is what Qt does
    before it draws the real dialog -- and under :data:`ASLEEP` that stat
    takes :data:`SLOW_S`. So this measures the freeze rather than merely
    describing it: with the start directory taken from ``self._roots[-1]``
    the click blocks, and with it taken from the probe cache it does not.
    """
    asleep = f"{ASLEEP}/data/experiment"
    widget = ProjectBrowserScreen(threaded=False)
    qtbot.addWidget(widget)
    widget._roots.append(asleep)

    seen = {}

    def _chooser(parent, title, start):
        seen["start"] = start
        os.path.isdir(start)        # what Qt does before it draws anything
        return ""

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", _chooser)

    started = time.monotonic()
    widget.choose_root()
    elapsed = time.monotonic() - started
    widget.close()

    assert seen["start"] != asleep, (
        "the chooser was handed a remembered root nothing has probed -- "
        "Qt stats and lists the start directory, so this is the freeze")
    assert elapsed < 1.0, f"the chooser took {elapsed:.1f}s to open"


def test_the_chooser_still_offers_the_folder_being_searched(
        qtbot, tmp_path, no_walk, qt_theme_applied, monkeypatch):
    """Not blocking is only correct if the convenience survives.

    The point of starting in the remembered folder is that the second folder
    is no more expensive to reach than the first. Once the probe has answered
    -- which `add_root` asks it to do the moment a root is added -- the
    chooser opens exactly where it always did.
    """
    widget = ProjectBrowserScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.add_root(str(tmp_path), scan=False)
    root = os.path.abspath(str(tmp_path))
    qtbot.waitUntil(
        lambda: path_probe.known(root, want_dir=True) is not None,
        timeout=5000)

    seen = {}
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory",
        lambda parent, title, start: seen.setdefault("start", start) and "")

    widget.choose_root()
    widget.close()

    assert seen["start"] == root


def test_a_root_still_waiting_does_not_hide_the_ones_that_answered(
        qtbot, tmp_path, monkeypatch, sleeping_mount, no_walk,
        qt_theme_applied):
    """The roots are tried newest-first, not "newest or nothing".

    A user who adds a sleeping mount as their most recent search folder must
    not thereby lose the folder chooser for every OTHER folder they search.
    `_start_directory` walks back through the roots until one the probe cache
    has confirmed, so the newest unanswered root costs one cache miss rather
    than the whole convenience.
    """
    answered = os.path.abspath(str(tmp_path))
    widget = ProjectBrowserScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.add_root(answered, scan=False)
    qtbot.waitUntil(
        lambda: path_probe.known(answered, want_dir=True) is True,
        timeout=5000)
    # Appended rather than added, so it is the newest root AND one nothing
    # has ever probed -- which is what a remembered mount looks like.
    widget._roots.append(f"{ASLEEP}/added/last")

    seen = {}

    def _chooser(parent, title, start):
        seen["start"] = start
        os.path.isdir(start)        # what Qt does before it draws anything
        return ""

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", _chooser)

    started = time.monotonic()
    widget.choose_root()
    elapsed = time.monotonic() - started
    widget.close()

    assert seen["start"] == answered, (
        "one unprobed root at the head of the list cost the user every "
        "folder behind it")
    assert elapsed < 1.0, f"the chooser took {elapsed:.1f}s to open"


def test_the_chosen_folder_is_remembered_under_one_spelling(
        qtbot, tmp_path, no_walk, qt_theme_applied, monkeypatch):
    """What the dialog hands back is primed under the name others ask about.

    `path_probe`'s cache is keyed on the path STRING, and `add_root` and
    `push_recent_source` both store ``os.path.abspath``'d one. Priming the
    dialog's own spelling -- which can carry a trailing separator -- would
    leave the spelling every other screen asks about still unknown, and the
    prime would have bought a cache entry nobody reads.
    """
    chosen = os.path.abspath(str(tmp_path))
    path_probe.forget(chosen)
    widget = ProjectBrowserScreen(threaded=False)
    qtbot.addWidget(widget)

    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory",
        lambda parent, title, start: chosen + os.sep)

    widget.choose_root()
    roots = widget.roots()
    widget.close()

    assert roots == (chosen,)
    assert path_probe.known(chosen) is True, (
        "the folder the dialog just listed was primed under a name nothing "
        "else uses")


def test_adding_a_root_asks_the_probe_about_it(
        qtbot, tmp_path, no_walk, qt_theme_applied):
    """The recovery for the pessimistic gate, and the only one it needs.

    Nothing on screen is painted from `_start_directory`, so no
    `path_probe.probes.answered` subscription is required to undo a *no*.
    What is required is that the question gets asked at all -- otherwise a
    root added this session would be refused by the chooser forever.
    """
    root = os.path.abspath(str(tmp_path))
    path_probe.forget(root)
    widget = ProjectBrowserScreen(threaded=False)
    qtbot.addWidget(widget)

    widget.add_root(root, scan=False)
    qtbot.waitUntil(
        lambda: path_probe.known(root, want_dir=True) is True, timeout=5000)
    widget.close()


# ---------------------------------------------------------------------------
# The failure that arrives after the scan that replaced it
# ---------------------------------------------------------------------------

def test_a_superseded_scan_cannot_overwrite_the_one_that_replaced_it(
        qtbot, tmp_path, monkeypatch, qt_theme_applied):
    """A slow first walk's failure must not land on the second walk's table.

    `JobRunner.cancel` abandons the *results* of the jobs it supersedes, and
    `job_failed` is emitted regardless -- so a walk failing on a folder that
    is no longer being searched used to replace the project count with its
    error and re-enable the Scan button underneath a walk still running.
    Routing the failure through the generation-guarded completion handler is
    what makes it get dropped with everything else the cancel dropped.
    """
    entered = threading.Event()
    release = threading.Event()

    def slow_and_doomed(roots, depth):
        entered.set()
        release.wait(20)
        raise RuntimeError("the first mount went away")

    widget = ProjectBrowserScreen(threaded=True)
    qtbot.addWidget(widget)
    widget.add_root(str(tmp_path), scan=False)
    complaints = []
    widget.failed.connect(complaints.append)

    monkeypatch.setattr(module, "_browse", slow_and_doomed)
    widget.rescan()
    qtbot.waitUntil(entered.is_set, timeout=10000)

    listed = ProjectSummary(root=os.path.join(str(tmp_path), "plate1"),
                            name="plate1", known=True)
    monkeypatch.setattr(module, "_browse", lambda roots, depth: (listed,))
    with qtbot.waitSignal(widget.scanned, timeout=10000):
        widget.rescan()
    assert widget._table.rowCount() == 1

    release.set()
    qtbot.waitUntil(lambda: widget.active_jobs() == 0, timeout=20000)
    qtbot.wait(150)             # let anything queued to the GUI thread land
    text = widget._status.text()
    widget.close()

    assert complaints == [], (
        f"a superseded walk's failure was announced: {complaints}")
    assert "went away" not in text, (
        f"the abandoned walk's error overwrote the finished table: {text!r}")
    assert "1 project(s)" in text


def test_a_failing_scan_still_says_what_went_wrong(
        qtbot, tmp_path, monkeypatch, qt_theme_applied):
    """Dropping a SUPERSEDED failure must not drop the current one too."""
    def doomed(roots, depth):
        raise RuntimeError("the mount went away")

    monkeypatch.setattr(module, "_browse", doomed)
    widget = ProjectBrowserScreen(threaded=False)
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.failed, timeout=5000) as caught:
        widget.add_root(str(tmp_path))
    text = widget._status.text()
    widget.close()

    assert "the mount went away" in caught.args[0]
    assert "the mount went away" in text
    assert widget._rescan.isEnabled()
