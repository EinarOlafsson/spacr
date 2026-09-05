"""Dragging a folder over the motility preview does not stat it.

THE FREEZE, 2026-09-04. `MotilityPreviewPanel._dropped_path` decided whether
to accept a drag with ``Path(url.toLocalFile()).is_dir()``, and Qt delivers
`dragEnterEvent` / `dragMoveEvent` / `dropEvent` on the GUI thread by
definition. `dragMoveEvent` re-fires on every mouse move, so holding a folder
over the panel was one stat per pixel of travel.

Measured on the maintainer's machine: a single stat under ``/nas_mnt`` -- an
``autofs`` mount with ``timeout=600`` whose share was asleep -- had NOT
RETURNED AFTER TWENTY SECONDS, because the stat is what triggers the
automount. Dragging a folder off that share froze the whole application
before the user had even let go of the mouse button, with no traceback,
because a stalled event loop is not a crash.

The panel's real work was already threaded -- `load_folder_async` submits
`scan_plate_payload` to a `JobRunner`. The block sat UPSTREAM of that, in the
accept/reject decision that gates it.

The answer now comes from `spacr.qt.path_probe`, from cache, optimistically:
a folder nobody has probed yet is accepted, because the decision has to be
made before any probe can finish, and a wrongly accepted drop degrades to the
worker's error message in the status label rather than a frozen window.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QMimeData, QPoint, QPointF, QUrl, Qt
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent

from spacr.qt import path_probe
from spacr.qt.widgets.motility_preview import MotilityPreviewPanel

pytestmark = pytest.mark.qt

#: Longer than any human would call responsive, far shorter than the twenty
#: seconds actually measured. A test that waited the real duration would be a
#: test nobody runs.
SLOW_S = 8.0


@pytest.fixture(autouse=True)
def _fresh_cache():
    path_probe.forget()
    yield
    path_probe.forget()


@pytest.fixture
def panel(qtbot, qt_theme_applied):
    widget = MotilityPreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    yield widget
    widget.shutdown()


def _urls(paths):
    """A mime payload of local file URLs.

    The caller keeps the reference: PySide6 drag events do not own the
    :class:`QMimeData` they are handed, and letting it be collected while the
    event is alive segfaults the interpreter.
    """
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return mime


@pytest.fixture
def sleeping_mount(monkeypatch):
    """Every directory check takes :data:`SLOW_S`, as a sleeping share does."""
    def slow_isdir(_path):
        time.sleep(SLOW_S)
        return True

    from pathlib import Path

    def slow_path_isdir(_self):
        time.sleep(SLOW_S)
        return True

    # Both spellings: the old code used `Path.is_dir`, and `path_probe`'s own
    # worker uses `os.path.isdir`. Patching only one would let the block come
    # back through the other.
    monkeypatch.setattr(Path, "is_dir", slow_path_isdir)
    monkeypatch.setattr(path_probe.os.path, "isdir", slow_isdir)


def test_a_drag_over_the_panel_returns_before_the_mount_wakes(
        panel, sleeping_mount, tmp_path):
    """The property the freeze violated: the decision does not wait."""
    mime = _urls([tmp_path / "plate_on_a_sleeping_share"])
    enter = QDragEnterEvent(QPoint(1, 1), Qt.CopyAction, mime,
                            Qt.LeftButton, Qt.NoModifier)

    started = time.monotonic()
    panel.dragEnterEvent(enter)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"dragEnterEvent took {elapsed:.1f}s -- the panel is stat-ing the "
        "dragged folder on the GUI thread again, which is the freeze")
    assert enter.isAccepted(), (
        "an unprobed folder must still be accepted: the answer cannot be "
        "known yet, and refusing it would break every first drop")


def test_dragging_across_the_panel_stays_responsive(
        panel, sleeping_mount, tmp_path):
    """`dragMoveEvent` fires per mouse move; none of them may block."""
    mime = _urls([tmp_path / "plate_on_a_sleeping_share"])

    started = time.monotonic()
    for step in range(40):
        move = QDragMoveEvent(QPoint(step, step), Qt.CopyAction, mime,
                              Qt.LeftButton, Qt.NoModifier)
        panel.dragMoveEvent(move)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"forty drag-move events took {elapsed:.1f}s -- dragging a folder "
        "across the panel is stat-ing it once per mouse move")


def test_the_drop_still_dispatches_the_scan(panel, sleeping_mount, tmp_path,
                                            monkeypatch):
    """Off the GUI thread is only correct if the plate still gets loaded."""
    asked = []
    monkeypatch.setattr(panel, "load_folder_async",
                        lambda path: asked.append(path) or True)
    target = tmp_path / "plate_on_a_sleeping_share"
    mime = _urls([target])
    dropped = QDropEvent(QPointF(1, 1), Qt.CopyAction, mime,
                         Qt.LeftButton, Qt.NoModifier)

    started = time.monotonic()
    panel.dropEvent(dropped)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"dropEvent took {elapsed:.1f}s"
    assert dropped.isAccepted()
    assert asked == [str(target)], (
        "the drop no longer hands the folder to the threaded loader")


def test_a_drag_with_no_urls_at_all_is_still_refused(panel):
    """Text dragged in from elsewhere is not a path, and never was."""
    mime = QMimeData()
    mime.setText("/somewhere/else")
    enter = QDragEnterEvent(QPoint(1, 1), Qt.CopyAction, mime,
                            Qt.LeftButton, Qt.NoModifier)

    panel.dragEnterEvent(enter)

    assert not enter.isAccepted()
    assert panel._dropped_path(enter) is None


def test_a_file_the_probe_has_already_answered_for_is_refused(panel, tmp_path,
                                                              qtbot):
    """Optimism is only for the unknown: a known non-folder is still refused.

    Without this the fix would have quietly turned the drop zone into one
    that accepts anything at all, forever.
    """
    stray = tmp_path / "notes.txt"
    stray.write_text("not a plate")
    path_probe.exists(str(stray), want_dir=True)
    qtbot.waitUntil(
        lambda: path_probe.known(str(stray), want_dir=True) is False,
        timeout=10000)

    mime = _urls([stray])
    enter = QDragEnterEvent(QPoint(1, 1), Qt.CopyAction, mime,
                            Qt.LeftButton, Qt.NoModifier)
    panel.dragEnterEvent(enter)

    assert not enter.isAccepted(), (
        "a path the probe has answered NOT a directory should be refused")
