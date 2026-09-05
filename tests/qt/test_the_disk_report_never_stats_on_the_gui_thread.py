""""Check disk space" must not stat a remembered folder on the GUI thread.

THE FREEZE, 2026-09-04. Preferences > Performance > "Check disk space" ran
`resource_cleanup.disk_report()` straight from the button's `clicked` slot,
and its first act was `project_paths()`, which called `os.path.isdir` on
every source folder the user had ever pointed a module at -- read back out
of QSettings, one per module, plus three recents each.

One of the maintainer's was under `/nas_mnt`, an `autofs` mount with
`timeout=600`. Measured on that machine: a single stat on that path had NOT
RETURNED AFTER TWENTY SECONDS, because the stat is what wakes the automount
and the share was asleep. The whole application stopped, with no traceback,
because a stalled event loop is not a crash.

WHAT IS ASSERTED HERE is everything that lives in
`spacr/qt/resource_cleanup.py`, and it is two things rather than one.

FIRST, `project_paths()` never stats a remembered folder when the caller is
the GUI thread; it answers from `path_probe`'s cache instead.

SECOND -- and this is the sibling site, one step downstream, that a fix to
`project_paths` alone leaves wide open -- `disk_report()` itself does not
freeze the GUI thread either. It cannot answer from a cache: `os.stat` and
`shutil.disk_usage` are what produce the device ids and the byte counts, and
a remembered number is an invented one. So on the GUI thread it BOUNDS the
wait instead, and a folder that misses the budget is counted in the note.
That branch is a backstop: putting the whole call on a worker is the Qt
caller's job, in `spacr/qt/preferences.py::run_resource_action`, and is
covered with that dialog.

The reason the backstop is needed rather than merely tidy: `path_probe`
caches TRUE for a path whose stat never answers (`_stat_with_timeout` gives
up after `PROBE_TIMEOUT_S` and reports it present), so a second GUI-thread
report is handed back exactly the sleeping mount the first one skipped.

And the other direction is asserted for both, because it is the one a cache
and a timeout are easy to break: on a worker the stat still happens and the
wait is unbounded, so the drive table is complete rather than one refresh
behind.
"""
from __future__ import annotations

import collections
import os
import threading
import time

import pytest

pytest.importorskip("PySide6")

from spacr.qt import path_probe
from spacr.qt import resource_cleanup as rc

#: Long enough that nobody would call it responsive, far short of the twenty
#: seconds actually measured -- a test that waited the real duration is a
#: test nobody runs.
SLOW_S = 8.0

#: What the interface is allowed to spend. Generous: the point is the
#: difference between "returned" and "did not return", not a benchmark.
BUDGET_S = 1.0

ASLEEP = "/nas_mnt/data/sequencing/seq_3"

#: What a mount that is waking up costs the caller that touches it. Shorter
#: than `SLOW_S` because these tests wait for the helper thread to finish
#: after the assertion rather than abandoning it, and a test suite is not the
#: place to hold a thread for eight seconds.
DOZING_S = 3.0

#: `shutil.disk_usage`'s shape, for the pretend drive behind `ASLEEP`. The
#: real call cannot be made on a path that is not there, and this file must
#: never touch `/nas_mnt` for real -- on the maintainer's machine that is the
#: automount whose twenty seconds started all of this.
_Usage = collections.namedtuple("_Usage", "total used free")


@pytest.fixture(autouse=True)
def _fresh_cache():
    """No cached answers, because an empty cache is the case that froze."""
    path_probe.forget()
    yield
    path_probe.forget()


@pytest.fixture
def sleeping_mount(monkeypatch, tmp_path):
    """Make `ASLEEP` behave like the maintainer's dozing autofs share.

    Only that path. A blanket slow `isdir` would also catch the home and
    temp directories, which spaCR reads from the environment rather than
    from anything the user typed, and which the fix deliberately still
    checks directly.
    """
    real_isdir = os.path.isdir

    def slow(path):
        if str(path).startswith("/nas_mnt"):
            time.sleep(SLOW_S)
            return True
        return real_isdir(path)

    monkeypatch.setattr(rc.os.path, "isdir", slow)
    monkeypatch.setattr(path_probe.os.path, "isdir", slow)

    kept = tmp_path / "kept"
    kept.mkdir()

    from spacr.qt import app as app_mod
    from spacr.qt import prefs
    monkeypatch.setattr(app_mod, "APPS", [("map_barcodes",)])
    monkeypatch.setattr(prefs, "get_last_source", lambda key: ASLEEP)
    monkeypatch.setattr(prefs, "get_recent_sources",
                        lambda key, limit=3: [str(kept)])
    return kept


def test_project_paths_returns_before_the_mount_wakes(qapp, sleeping_mount):
    """The property the freeze violated: the GUI thread does not wait."""
    started = time.monotonic()
    paths = rc.project_paths()
    elapsed = time.monotonic() - started

    assert elapsed < BUDGET_S, (
        f"project_paths() took {elapsed:.1f}s on the GUI thread -- it is "
        "stat-ing remembered folders again, which is the freeze")
    assert ASLEEP not in paths, (
        "a folder nothing has probed yet must be left out, not waited for")


def test_the_readings_that_never_needed_a_user_path_still_arrive(
        qapp, sleeping_mount):
    """A refusal to stat must not empty the table it was protecting.

    The home and temp directories are spaCR's own -- nobody typed them, and
    every start-up has already stat-ed them -- so they are still in the
    first report even while the sleeping mount is unknown.
    """
    paths = rc.project_paths()
    assert os.path.expanduser("~") in paths
    assert any(os.path.realpath(p) == os.path.realpath("/tmp") or
               "tmp" in p for p in paths)


def test_a_second_ask_is_still_immediate(qapp, sleeping_mount):
    """The probe for a path that never answers must not be joined.

    Every later click asks again; if the second ask waited on the first
    probe the freeze would simply have moved one click later.
    """
    rc.project_paths()
    started = time.monotonic()
    rc.project_paths()
    assert time.monotonic() - started < BUDGET_S


def test_a_worker_still_measures_the_folder_the_gui_would_skip(
        qapp, monkeypatch, tmp_path):
    """Off the GUI thread the stat is the point, not the hazard.

    `disk_report` runs on a JobRunner, and there a cached guess would be an
    invented number. This is the assertion that stops the cache from being
    wired in everywhere and quietly costing the user their drive table.
    """
    kept = tmp_path / "kept"
    kept.mkdir()
    gone = tmp_path / "deleted"

    from spacr.qt import app as app_mod
    from spacr.qt import prefs
    monkeypatch.setattr(app_mod, "APPS", [("map_barcodes",)])
    monkeypatch.setattr(prefs, "get_last_source", lambda key: str(gone))
    monkeypatch.setattr(prefs, "get_recent_sources",
                        lambda key, limit=3: [str(kept)])

    answer = {}

    def on_a_worker():
        answer["paths"] = rc.project_paths()

    worker = threading.Thread(target=on_a_worker, name="spacr-test-disk")
    worker.start()
    worker.join(30)
    assert not worker.is_alive()

    assert str(kept) in answer["paths"], (
        "the worker must read the real filesystem -- a report that answers "
        "from a cache is not a disk reading")
    assert str(gone) not in answer["paths"]


# ---------------------------------------------------------------------------
# The sibling site: the reading itself
# ---------------------------------------------------------------------------

@pytest.fixture
def dozing_drive(monkeypatch, tmp_path):
    """Make the drive behind `ASLEEP` take `DOZING_S` to answer a stat.

    Both calls are stood in for, and neither ever reaches `/nas_mnt`: this
    suite runs on the machine whose automount is the bug, and a test that
    woke it for real would be indistinguishable from the defect.

    :returns: a real local folder, on a drive that answers instantly, which
        must keep its line in every report the sleeping one appears in.
    """
    real_stat = os.stat
    real_usage = rc.shutil.disk_usage
    kept = tmp_path / "kept"
    kept.mkdir()

    class _Stat:
        """Just enough of `os.stat_result` for a device id."""

        st_dev = 0xDECAF

    def slow_stat(path, *args, **kwargs):
        """`os.stat`, dozing for the pretend mount and honest elsewhere."""
        if str(path).startswith("/nas_mnt"):
            time.sleep(DOZING_S)
            return _Stat()
        return real_stat(path, *args, **kwargs)

    def usage(path):
        """`shutil.disk_usage`, answering for the pretend mount."""
        if str(path).startswith("/nas_mnt"):
            return _Usage(100, 60, 40)
        return real_usage(path)

    monkeypatch.setattr(rc.os, "stat", slow_stat)
    monkeypatch.setattr(rc.shutil, "disk_usage", usage)
    return kept


def test_the_reading_gives_up_rather_than_freezing_the_interface(
        qapp, dozing_drive):
    """`disk_report` on the GUI thread returns; the dozing drive is a note.

    THE SIBLING SITE. `project_paths` can refuse to stat a remembered folder
    and it buys nothing on its own, because `path_probe` answers TRUE for a
    path whose probe timed out -- and then `os.stat` here waits the full
    twenty seconds for it. This is that call, bounded.
    """
    started = time.monotonic()
    report = rc.disk_report([ASLEEP, str(dozing_drive)])
    elapsed = time.monotonic() - started

    assert elapsed < rc._GUI_DISK_BUDGET_S + BUDGET_S, (
        f"disk_report() held the GUI thread for {elapsed:.1f}s -- it is "
        "waiting on the mount again, which is the freeze")
    assert ASLEEP not in [entry.path for entry in report.entries]
    assert report.note == "1 folder(s) could not be read."


def test_the_drives_that_do_answer_keep_their_lines(qapp, dozing_drive):
    """One dozing mount must not cost the folders listed after it.

    The reason the readings are started together rather than one after the
    other: a budget spent in order would leave every later folder nothing,
    and the user would lose the local drive as well as the sleeping one.
    """
    report = rc.disk_report([ASLEEP, str(dozing_drive)])
    assert [entry.path for entry in report.entries] == [str(dozing_drive)]
    assert report.entries[0].total > 0


def test_a_worker_still_waits_for_the_drive_the_gui_gave_up_on(
        qapp, dozing_drive):
    """Off the GUI thread the reading is complete, however long it takes.

    The bound is a backstop for a caller on the wrong thread, not a new
    policy for the report. `preferences._start_disk_report` runs this on a
    `JobRunner`, and there a dozing mount is waited for and reported.
    """
    landed = {}

    def on_a_worker():
        """Read the disk the way the button actually reads it."""
        landed["report"] = rc.disk_report([ASLEEP, str(dozing_drive)])

    worker = threading.Thread(target=on_a_worker, name="spacr-test-disk-read")
    worker.start()
    worker.join(30)
    assert not worker.is_alive()

    report = landed["report"]
    assert ASLEEP in [entry.path for entry in report.entries], (
        "a worker that stops waiting is inventing the drive table it was "
        "asked to measure")
    assert report.note == ""


def test_an_ordinary_local_reading_is_unchanged_on_the_gui_thread(
        qapp, tmp_path):
    """The bound must cost nothing when nothing is asleep.

    Every existing promise of the report -- a line per drive, folders that
    are gone counted rather than crashed on -- still holds when the caller
    is the GUI thread and the budget is in force.
    """
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()

    report = rc.disk_report([str(first), str(second)])
    assert len(report.entries) == 1, "two folders on one drive are one line"
    assert report.entries[0].total > 0
    assert report.note == ""

    gone = rc.disk_report([str(tmp_path / "unplugged"), str(first)])
    assert [entry.path for entry in gone.entries] == [str(first)]
    assert gone.note == "1 folder(s) could not be read."
