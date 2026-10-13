"""The QC dashboard decides on a worker, never on the GUI thread.

THE DEFECT, 2026-09-04. `QCDashboardScreen.refresh` handed only the PARSE to
its `JobRunner`. The two decisions in front of it ran inline:

    refresh()
      -> os.path.isdir(src)                       # the folder guard
      -> self._fingerprint(src)
        -> spacr.seg_qc.find_scorecards(src)      # lists the qc folder
        -> os.stat(...)                           # once per artifact

and `src` is whatever the user typed, browsed to or dropped on the screen.
Measured on the maintainer's machine that day: a single `os.path.exists`
under `/nas_mnt` -- an `autofs` mount whose share was asleep -- had NOT
RETURNED AFTER TWENTY SECONDS, because the stat is what triggers the
automount. A project folder on a network mount is precisely what this screen
exists to read, so pointing it at one stopped the whole application. It was
reported as "opening map barcodes crashes spacr", plus hover flicker and
glimpses of other screens, and it left no traceback: a stalled event loop is
not a crash.

The tests below pin both halves of the fix -- that `refresh` returns at once,
and that every message it used to show still arrives, a moment later.
"""
from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import qc_dashboard as qc_module

#: Longer than any human would call responsive, shorter than the twenty
#: seconds actually measured. A test that waited the real duration would be
#: a test nobody runs.
SLOW_S = 8.0

#: Stands in for `/nas_mnt`. Only paths under it are made slow: patching
#: `os.path.isdir` for everything would park whatever else the GUI thread
#: does during the eight seconds, and the failure under test would be lost
#: in the noise of a test that hangs for its own reasons.
ASLEEP = "/nas-asleep"


@pytest.fixture
def sleeping_share(monkeypatch):
    """Make every folder check under :data:`ASLEEP` take :data:`SLOW_S`."""
    real_isdir = os.path.isdir

    def slow_isdir(path, _real=real_isdir):
        if str(path).startswith(ASLEEP):
            time.sleep(SLOW_S)
            return True
        return _real(path)

    monkeypatch.setattr(qc_module.os.path, "isdir", slow_isdir)
    return slow_isdir


@pytest.fixture
def sleeping_scorecards(monkeypatch):
    """Make the scorecard listing take :data:`SLOW_S`, as a cold mount does."""
    from spacr import seg_qc

    def crawl(_src):
        time.sleep(SLOW_S)
        raise AssertionError("the GUI thread waited for the qc folder")

    monkeypatch.setattr(seg_qc, "find_scorecards", crawl)
    return crawl


def _screen(qtbot, **kwargs):
    screen = qc_module.QCDashboardScreen(threaded=True, **kwargs)
    qtbot.addWidget(screen)
    return screen


def test_refresh_returns_before_the_folder_guard_answers(qtbot,
                                                         sleeping_share):
    """The property the freeze violated: asking does not mean waiting."""
    screen = _screen(qtbot)
    screen._src_edit.setText(f"{ASLEEP}/data/plate1")

    started = time.monotonic()
    screen.refresh()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"refresh() took {elapsed:.1f}s -- it is stat-ing the user's folder "
        "on the GUI thread again, which is the freeze")


def test_refresh_returns_before_the_fingerprint_walks_the_project(
        qtbot, tmp_path, sleeping_scorecards):
    """The second half: the cache key is a walk of the disk, not a guess.

    The folder exists here, so the guard passes immediately and the only
    slow thing left is the fingerprint. It has to be on the worker too --
    moving one of the two would have left the freeze in place.
    """
    screen = _screen(qtbot)
    screen._src_edit.setText(str(tmp_path))

    started = time.monotonic()
    screen.refresh()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"refresh() took {elapsed:.1f}s -- it is fingerprinting the project "
        "on the GUI thread")


def test_setting_the_source_does_not_wait_either(qtbot, sleeping_share):
    """Every route in is one route: browse, drop and Enter all land here."""
    screen = _screen(qtbot)

    started = time.monotonic()
    screen.set_source(f"{ASLEEP}/data/plate1")
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"set_source() took {elapsed:.1f}s -- a dropped or browsed folder "
        "still freezes the window")


def test_a_folder_that_is_not_there_still_says_so(qtbot, tmp_path):
    """Off the GUI thread is only correct if the warning still appears.

    The guard moved to the worker; the sentence it produces did not change,
    it only arrives once the answer does.
    """
    screen = _screen(qtbot)
    screen.set_source(str(tmp_path / "nope"))

    qtbot.waitUntil(lambda: "is not a folder" in screen.status_text(),
                    timeout=10000)
    assert screen._status.property("spacrError") == "true"
    assert screen._verdict.text() == "That folder does not exist."
    assert screen.dashboard() is None


def test_the_verdicts_and_the_cache_both_survive_the_move(qtbot, tmp_path):
    """The read still lands, and a second look still skips the parse."""
    calls = []

    def reader(src):
        from spacr.qt.widgets.qc_summary import Dashboard
        calls.append(src)
        return Dashboard(root=str(src), verdict="pass",
                         headline="nothing to report")

    (tmp_path / "measurements").mkdir()
    (tmp_path / "measurements" / "measurements.db").write_bytes(b"x")

    screen = _screen(qtbot, reader=reader)
    screen.set_source(str(tmp_path))

    qtbot.waitUntil(lambda: screen.dashboard() is not None, timeout=10000)
    assert "nothing was recomputed" in screen.status_text()
    assert calls == [str(tmp_path)]

    screen.refresh()
    qtbot.waitUntil(
        lambda: "has changed" in screen.status_text(), timeout=10000)
    assert calls == [str(tmp_path)], (
        "the fingerprint cache stopped working when it moved to the worker")
