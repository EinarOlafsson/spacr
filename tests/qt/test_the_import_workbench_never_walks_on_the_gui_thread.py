"""A drop onto the import workbench does not walk the folder inline.

THE FREEZE, 2026-09-04. `ImportWorkbench.add_files` -- the one funnel for
both `dropEvent` and the "Add files…" dialog -- called `images_under()`
straight through:

    ImportWorkbench.dropEvent
      -> add_files(paths)
        -> images_under(paths)
          -> os.path.isdir(path)          for every dropped path
          -> os.walk(path)                for every dropped folder

and the dropped path is, by definition, one the user chose: a plate folder,
usually on the microscope's share. Measured on the maintainer's machine that
day, a single `os.path.exists` under an `autofs` mount whose share was asleep
had NOT RETURNED AFTER TWENTY SECONDS -- the stat is what triggers the
automount. A recursive walk is thousands of those, run on the thread that
paints. The application froze with no traceback, because a stalled event loop
is not a crash; it surfaced as "spacr crashes", hover flicker, and glimpses of
other screens.

The property asserted here is the one the freeze violated: the drop RETURNS,
and the files arrive when the worker has found them.
"""
from __future__ import annotations

import os
import threading
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl      # noqa: E402
from PySide6.QtGui import QDropEvent                         # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog      # noqa: E402

from spacr.qt.widgets import import_workbench as iw          # noqa: E402

#: Longer than any human would call responsive, shorter than the twenty
#: seconds actually measured. A test that waited the real duration would be
#: a test nobody runs.
SLOW_S = 8.0


@pytest.fixture
def slow_walk(monkeypatch):
    """Make the walk take :data:`SLOW_S`, as a sleeping mount does.

    Released at teardown rather than simply slept through: the worker that
    is stuck in it is a real thread, and `shutdown` would otherwise wait its
    whole budget for a folder nobody is looking at any more.
    """
    released = threading.Event()

    def crawl(_paths, **_kwargs):
        released.wait(SLOW_S)
        return []

    monkeypatch.setattr(iw, "images_under", crawl)
    yield crawl
    released.set()


@pytest.fixture
def plate_folder(tmp_path):
    """Eight Yokogawa-ish images and one file that is not an image."""
    folder = tmp_path / "plate1"
    folder.mkdir()
    for well in ("A01", "A02"):
        for field in (1, 2):
            for channel in (1, 2):
                (folder / f"{well}_T0001F{field:03d}L01A01Z01C{channel:02d}.tif"
                 ).write_bytes(b"II*\x00")
    (folder / "notes.txt").write_text("not an image")
    return folder


@pytest.fixture
def panel(qtbot):
    made = iw.ImportWorkbench()
    qtbot.addWidget(made)
    return made


def _settle(panel, timeout_s: float = 10.0) -> None:
    """Spin the loop until the walk has landed, as the real one would."""
    deadline = time.monotonic() + timeout_s
    while panel.is_scanning() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.005)
    QApplication.processEvents()


def _drop(paths):
    data = QMimeData()
    data.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return QDropEvent(QPointF(4, 4), Qt.CopyAction, data,
                      Qt.LeftButton, Qt.NoModifier), data


# ---------------------------------------------------------------------------
# the property the freeze violated
# ---------------------------------------------------------------------------

def test_add_files_returns_before_the_folder_is_walked(panel, slow_walk):
    started = time.monotonic()
    panel.add_files(["/nas_mnt/data/plate1"])
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"add_files took {elapsed:.1f}s -- it is walking the dropped folder "
        "on the GUI thread again, which is the freeze")


def test_a_drop_returns_before_the_folder_is_walked(panel, slow_walk):
    """The entry point the user actually uses, through Qt's own event."""
    event, data = _drop(["/nas_mnt/data/plate1"])

    started = time.monotonic()
    panel.dropEvent(event)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"dropEvent took {elapsed:.1f}s -- the drag is still holding the "
        "GUI thread while the share is stat-ed")
    assert event.isAccepted()
    del event, data


def test_the_file_dialog_does_not_wait_for_the_walk(panel, slow_walk,
                                                    monkeypatch):
    """"Add files…" picks names on a share too."""
    monkeypatch.setattr(
        QFileDialog, "getOpenFileNames",
        staticmethod(lambda *a, **k: (["/nas_mnt/data/a.tif"], "")))

    started = time.monotonic()
    panel.ask_for_files()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"ask_for_files took {elapsed:.1f}s -- it walks what the dialog "
        "returned before it gives the thread back")


def test_the_panel_says_it_is_looking_rather_than_nothing(panel, slow_walk):
    """A drop that answers later must still answer the drop immediately."""
    panel.add_files(["/nas_mnt/data/plate1"])

    assert panel.dropped.text().strip(), (
        "the panel said nothing at all while the walk was running, which "
        "reads as a drop that was ignored")
    assert panel.is_scanning()


# ---------------------------------------------------------------------------
# and nothing the user saw before has gone
# ---------------------------------------------------------------------------

def test_every_image_still_arrives_a_moment_later(panel, plate_folder):
    """Rule two: the only change is that it arrives later."""
    panel.add_files([str(plate_folder)])
    _settle(panel)

    assert len(panel.files()) == 8
    assert all(not f.endswith(".txt") for f in panel.files())
    assert panel.table.rowCount() == 8
    assert "8 of 8" in panel.evidence.text()


def test_a_drop_still_fills_the_table(panel, plate_folder):
    event, data = _drop([plate_folder])

    panel.dropEvent(event)
    _settle(panel)

    assert len(panel.files()) == 8
    assert panel.the_plan() is not None
    del event, data


def test_a_second_drop_of_the_same_folder_adds_nothing(panel, plate_folder):
    panel.add_files([str(plate_folder)])
    _settle(panel)
    panel.add_files([str(plate_folder)])
    _settle(panel)

    assert len(panel.files()) == 8


def test_two_drops_are_both_kept_rather_than_coalesced(panel, plate_folder,
                                                       tmp_path):
    """Unlike a refresh, two drops ask DIFFERENT questions.

    Dropping one plate and then another while the first is still being
    walked must add both; a coalescing runner would silently lose one.
    """
    other = tmp_path / "plate2"
    other.mkdir()
    (other / "B01_T0001F001L01A01Z01C01.tif").write_bytes(b"II*\x00")

    panel.add_files([str(plate_folder)])
    panel.add_files([str(other)])
    _settle(panel)

    assert len(panel.files()) == 9


def test_an_empty_choice_starts_no_walk_and_says_nothing_new(panel,
                                                             plate_folder):
    """Cancelling the dialog must not leave the panel claiming to be busy."""
    panel.add_files([str(plate_folder)])
    _settle(panel)
    summary = panel.dropped.text()

    panel.add_files([])

    assert panel.dropped.text() == summary
    assert not panel.is_scanning()


def test_closing_the_dialog_does_not_leave_a_walk_running(qtbot, monkeypatch):
    """Qt aborts the process if a running QThread is destroyed.

    A walk of a fifth of a second rather than :data:`SLOW_S`: the point here
    is that `done` waits for the worker at all, and a worker deliberately
    stuck for eight seconds would only prove that the wait is bounded --
    which `JobRunner.shutdown` already promises -- at the cost of parking a
    thread for the rest of the session.
    """
    monkeypatch.setattr(iw, "images_under",
                        lambda _paths, **_kw: (time.sleep(0.2), [])[1])
    dialog = iw.ImportWorkbenchDialog()
    qtbot.addWidget(dialog)
    dialog.workbench.add_files(["/nas_mnt/data/plate1"])

    dialog.reject()

    assert dialog.workbench._scanner.active_jobs() == 0


# ---------------------------------------------------------------------------
# the edges the first pass left open
# ---------------------------------------------------------------------------

def _drain(panel, timeout_s: float = 10.0) -> None:
    """Wait until no walk THREAD is left, cancelled ones included.

    `_settle` waits on `is_scanning`, which a cancel clears immediately --
    the whole point of a cancel being that it does not join. These tests
    care what a cancelled walk does when it finally lands, so they wait for
    the thread itself to retire.
    """
    deadline = time.monotonic() + timeout_s
    while panel._scanner.active_jobs() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.005)
    for _ in range(5):
        QApplication.processEvents()


@pytest.fixture
def gated_walk(monkeypatch):
    """A walk held open until the test releases it, then answering fully.

    Unlike `slow_walk` this one RETURNS THE FILES, so a test can ask what
    happens when a walk the user gave up on finally lands.
    """
    release = threading.Event()
    answer: list = []

    def crawl(_paths, **_kwargs):
        release.wait(SLOW_S)
        return list(answer)

    monkeypatch.setattr(iw, "images_under", crawl)
    crawl.release = release                                  # type: ignore
    crawl.answer = answer                                    # type: ignore
    yield crawl
    release.set()


def test_a_walk_that_fails_does_not_leave_the_panel_saying_it_is_working(
        panel, monkeypatch):
    """`JobRunner` calls `on_done` only for a job that SUCCEEDED.

    A caption written before `submit` therefore stays on screen for the rest
    of the session when the worker raises -- and a share that is refusing is
    exactly where `os.walk` raises. The failure has to come back.
    """
    monkeypatch.setattr(iw, "images_under", lambda *_a, **_k: (_ for _ in ()
                                                               ).throw(
        OSError("the share is not answering")))

    panel.add_files(["/nas_mnt/data/plate1"])
    _settle(panel)

    said = panel.dropped.text()
    assert "Working" not in said, (
        f"the panel is still claiming to be working: {said!r}")
    assert "the share is not answering" in said, (
        f"the walk failed and the panel did not say so: {said!r}")
    assert not panel.is_scanning()


def test_a_worker_that_dies_outright_still_clears_the_working_caption(
        panel, slow_walk):
    """The case `_walk` cannot catch: the job never delivers at all.

    `job_failed` is the only notice of it, and nothing was listening.
    """
    panel.add_files(["/nas_mnt/data/plate1"])
    assert "Working" in panel.dropped.text()

    panel._scanner.job_failed.emit("the walk thread died")

    said = panel.dropped.text()
    assert "Working" not in said, (
        f"a job that failed left its placeholder behind: {said!r}")
    assert "the walk thread died" in said


def test_a_failure_that_lands_after_the_panel_is_gone_is_not_an_exception(
        panel, slow_walk):
    """A parked worker outlives the widget; the C++ half is already gone."""
    panel.add_files(["/nas_mnt/data/plate1"])
    runner = panel._scanner
    panel.shutdown()
    panel.deleteLater()
    QApplication.processEvents()

    runner._on_worker_error_text("too late")                # must not raise


def test_clearing_the_panel_is_not_undone_by_a_walk_that_lands_later(
        panel, plate_folder, gated_walk):
    """Clear, on a share that is not answering, is the ordinary case.

    Without a cancel the walk lands half a minute later and quietly refills
    the table the user just emptied -- files they cleared, back, with no
    action of theirs in between.
    """
    gated_walk.answer.extend(
        str(f) for f in sorted(plate_folder.glob("*.tif")))
    panel.add_files([str(plate_folder)])

    panel.set_files([])                                # the Clear button
    gated_walk.release.set()
    _drain(panel)

    assert panel.files() == [], (
        "a walk the user cancelled refilled the table behind them")
    assert panel.table.rowCount() == 0
    assert "Working" not in panel.dropped.text()


def test_a_second_drop_is_not_cancelled_by_the_first_one_landing(
        panel, plate_folder, tmp_path):
    """`_files_found` must not take the cancelling path out of `set_files`."""
    other = tmp_path / "plate2"
    other.mkdir()
    (other / "B01_T0001F001L01A01Z01C01.tif").write_bytes(b"II*\x00")

    panel.add_files([str(plate_folder)])
    panel.add_files([str(other)])
    _drain(panel)
    _settle(panel)

    assert len(panel.files()) == 9


def test_the_waiting_caption_is_one_the_catalogs_already_carry(panel,
                                                               slow_walk):
    """A caption invented here would be English-only in nine languages.

    `tools/build_i18n_catalogs.py` harvests literal Qt strings out of
    `spacr/qt`, and `tests/qt/test_i18n_caption_ratchet.py` fails on any it
    finds that no catalog owns. Reusing a registered one costs nothing.
    """
    from spacr.qt.i18n_catalogs import en

    panel.add_files(["/nas_mnt/data/plate1"])

    assert panel.dropped.text() in en.UI_SOURCES, (
        f"{panel.dropped.text()!r} is not in the generated UI catalog -- "
        "register it or reuse a caption that is")


def test_nothing_left_on_the_panel_stats_anything(panel, plate_folder,
                                                  monkeypatch):
    """The inventory, asserted: the walk was not the only candidate.

    Everything else the panel does with a path -- the plate name, the
    proposal, the plan, the table, the tree -- works on BASENAMES, and a
    basename is a string operation. This fails if anyone puts a stat back
    on the redraw path.
    """
    panel.add_files([str(plate_folder)])
    _settle(panel)

    def forbidden(*_a, **_k):
        raise AssertionError("the GUI thread touched the filesystem")

    for name in ("walk", "listdir", "scandir"):
        monkeypatch.setattr(os, name, forbidden)
    for name in ("isdir", "isfile", "exists"):
        monkeypatch.setattr(os.path, name, forbidden)

    panel.propose_from_the_names()
    panel.regex.setText(r"(?P<plateID>[^_]+)_(?P<wellID>[A-Z]\d+)")
    panel.refresh()

    assert panel.the_plan() is not None
    assert panel.table.rowCount() == 8
