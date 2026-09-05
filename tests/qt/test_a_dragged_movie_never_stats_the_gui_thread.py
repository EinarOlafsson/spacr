"""Dragging a folder over the timelapse preview must not stat it inline.

THE FREEZE, 2026-09-04. `TimelapsePreviewPanel` sets `setAcceptDrops(True)`
and answered every drag with:

    dragEnterEvent / dragMoveEvent / dropEvent
      -> _dropped_path
        -> Path(url.toLocalFile()).is_dir()

on the path the user was dragging. A timelapse field of view is normally a
folder on the plate storage, so that path is normally a network one, and a
stat under the maintainer's ``/nas_mnt`` (``autofs``, share asleep) had not
returned after TWENTY SECONDS. `dragMoveEvent` fires on every mouse-move, so
merely holding a folder over the panel was enough to stall the event loop --
reported as a crash, because a frozen window leaves no traceback.

The fix asks :mod:`spacr.qt.path_probe` instead, optimistically: an unknown
path is accepted during the drag and the real read happens on the JobRunner
worker inside `load_sequence_async`, which reports an unusable path as a
line in the status label. Guessing wrong costs a sentence; stat-ing a
sleeping mount costs the application.
"""
from __future__ import annotations

import os
import pathlib
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

#: Longer than any human would call responsive, far shorter than the twenty
#: seconds actually measured. A test that waited the real duration would be
#: a test nobody runs.
SLOW_S = 8.0


@pytest.fixture()
def sleeping_mount(monkeypatch):
    """Every real directory test takes :data:`SLOW_S`, as a sleeping mount does.

    Both doors are shut: `Path.is_dir` is what the freeze called, and
    `path_probe`'s own `os.path.isdir` is what the background worker calls --
    if the fix were to wait on that worker the test would catch it too.
    """
    from spacr.qt import path_probe

    def never(*_args, **_kwargs):
        time.sleep(SLOW_S)
        return True

    monkeypatch.setattr(pathlib.Path, "is_dir", never)
    monkeypatch.setattr(path_probe.os.path, "isdir", never)
    path_probe.forget()
    return never


class _DragEvent:
    """The little a drag handler asks of the event it is given."""

    def __init__(self, mime):
        self._mime = mime
        self.accepted = False
        self.ignored = False

    def mimeData(self):
        return self._mime

    def acceptProposedAction(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


def _drag_of(path):
    from PySide6.QtCore import QMimeData, QUrl

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    return _DragEvent(mime)


@pytest.fixture()
def panel(qtbot):
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel

    widget = TimelapsePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_hovering_a_sleeping_folder_returns_at_once(panel, sleeping_mount,
                                                    tmp_path):
    """The property the freeze violated: the drag handler does not wait."""
    event = _drag_of(tmp_path / "plate_on_the_nas")

    started = time.monotonic()
    panel.dragEnterEvent(event)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"dragEnterEvent took {elapsed:.1f}s -- _dropped_path is stat-ing the "
        "dragged path on the GUI thread again, which is the freeze")
    assert event.accepted is True, (
        "an unprobed folder must still be accepted during the drag; the open "
        "is what discovers it is unusable")


def test_every_mouse_move_over_the_panel_stays_free(panel, sleeping_mount,
                                                    tmp_path):
    """`dragMoveEvent` fires per mouse-move, so one slow answer is many."""
    started = time.monotonic()
    for _ in range(20):
        panel.dragMoveEvent(_drag_of(tmp_path / "plate_on_the_nas"))
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"twenty drag-moves took {elapsed:.1f}s -- each one is stat-ing the "
        "dragged path")


def test_dropping_a_sleeping_folder_returns_at_once(panel, sleeping_mount,
                                                    tmp_path, monkeypatch):
    """The drop itself hands the path on without ever touching the disk."""
    handed = []
    monkeypatch.setattr(type(panel), "load_sequence_async",
                        lambda _self, path, **kw: handed.append(path) or True)
    event = _drag_of(tmp_path / "plate_on_the_nas")

    started = time.monotonic()
    panel.dropEvent(event)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"dropEvent took {elapsed:.1f}s -- the drop is stat-ing the path")
    assert handed == [str(tmp_path / "plate_on_the_nas")], (
        "the dropped folder must still reach the loader")


def test_a_frame_file_is_still_recognised_without_asking_the_disk(
        panel, sleeping_mount, tmp_path):
    """A stack is decided by its suffix, which is a string test and free.

    Kept because the cheap half must stay first: putting the probe before
    the suffix check would put a network round trip in front of an answer
    already in hand.
    """
    event = _drag_of(tmp_path / "movie.tif")

    started = time.monotonic()
    panel.dragEnterEvent(event)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert event.accepted is True


def test_a_remote_url_is_still_refused(panel, sleeping_mount):
    """Nothing about not blocking may make the panel accept a browser drag."""
    from PySide6.QtCore import QMimeData, QUrl

    mime = QMimeData()
    mime.setUrls([QUrl("https://example.org/movie.tif")])
    event = _DragEvent(mime)

    panel.dragEnterEvent(event)

    assert event.accepted is False
    assert event.ignored is True
