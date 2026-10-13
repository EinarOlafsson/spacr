"""Dragging a settings splitter and Ctrl + wheel over a column stay smooth.

The tutorial capture of lesson 05 (clip 14_resize_panels, Mask Generation)
showed the app stop painting while the settings splitter was dragged and the
right-hand column's text was sized with Ctrl + wheel. Measured offscreen at
2560x1440 with the field animating, on 2026-10-13:

* every move of a drag rewrote the whole preferences file to remember the
  pane sizes, 7.3 ms of the 11.5 ms the move held the GUI thread;
* every notch of the wheel restyled the whole column, 130-180 ms a notch,
  about half of it Qt looking up fallback fonts for the two glyphs Open Sans
  lacks (``▾`` and ``↓``) at each new size.

Now a drag writes the sizes once, when the handle is let go or rests (a move
made in code still writes at once), and a turn of the wheel restyles the
column once, when the wheel rests. The end state is the same: the same sizes,
the same sheet, the same pixels, and both are saved.

Timing is compared with a reference measured in the same test on the same
machine (the write a move used to make, a restyle a notch used to make), so a
loaded runner slows both sides alike.
"""
from __future__ import annotations

import statistics
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, QSettings, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent, QWheelEvent                 # noqa: E402
from PySide6.QtWidgets import (QApplication, QLabel, QPushButton,  # noqa: E402
                               QVBoxLayout, QWidget)

from spacr.qt.widgets import collapsible_splitter as cs            # noqa: E402


def _pump(n: int = 6) -> None:
    for _ in range(n):
        QApplication.processEvents()


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A private preferences file the size of a real one."""
    from spacr.qt import preferences

    path = tmp_path / "spacr-smooth.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    settings = preferences._settings()
    for i in range(400):
        settings.setValue(f"filler/key{i:03d}", "x" * 40)
    settings.sync()
    return preferences


@pytest.fixture
def writes(monkeypatch):
    """Every pane-size write, as ``(key, extents)``."""
    seen = []
    real = cs.set_pane_extents

    def spy(key, extents):
        seen.append((key, dict(extents)))
        real(key, extents)

    monkeypatch.setattr(cs, "set_pane_extents", spy)
    return seen


def _labels(parent, count, text):
    column = QVBoxLayout(parent)
    for i in range(count):
        column.addWidget(QLabel(f"{text} {i} ▾ …", parent))
    return parent


def _splitter(qtbot, key):
    split = cs.CollapsibleSplitter(Qt.Horizontal, persist_key=key)
    split.add_pane(_labels(QWidget(), 12, "Setting"), "Settings",
                   mode=cs.EDGE, extent=300)
    split.add_pane(_labels(QWidget(), 12, "Runtime"), "Runtime", stretch=2)
    qtbot.addWidget(split)
    split.resize(1200, 400)
    split.show()
    qtbot.waitExposed(split)
    _pump()
    return split


class _Drag:
    """A press on a handle, moves from a fixed point, and the release."""

    def __init__(self, handle):
        self.handle = handle
        self.start = handle.rect().center()
        self.origin = handle.mapToGlobal(self.start)
        self._send(QEvent.MouseButtonPress, 0, Qt.LeftButton, Qt.LeftButton)

    def _send(self, kind, dx, button, buttons):
        glob = self.origin + QPoint(dx, 0)
        local = self.handle.mapFromGlobal(glob)
        QApplication.sendEvent(self.handle, QMouseEvent(
            kind, QPointF(local), QPointF(glob), button, buttons,
            Qt.NoModifier))

    def move(self, dx):
        self._send(QEvent.MouseMove, dx, Qt.NoButton, Qt.LeftButton)

    def release(self, dx):
        self._send(QEvent.MouseButtonRelease, dx, Qt.LeftButton, Qt.NoButton)


def _drag(split, steps=30, step=8):
    drag = _Drag(split.handle(1))
    for i in range(steps):
        drag.move(step * (i + 1))
        _pump(1)
    return drag, step * steps


# --------------------------------------------------------------------------
# the drag
# --------------------------------------------------------------------------

def test_a_drag_writes_its_sizes_once_when_the_handle_is_let_go(
        qtbot, store, writes):
    split = _splitter(qtbot, "smooth::drag")
    before = split.sizes()[0]
    drag, travelled = _drag(split)
    assert writes == [], "a move of a drag wrote the preferences file"
    assert split.pane("Settings").extent == split.sizes()[0]
    assert split.sizes()[0] > before
    drag.release(travelled)
    assert writes == [("smooth::drag", split.extents())]
    assert cs.get_pane_extents("smooth::drag")["Settings"] == split.sizes()[0]
    qtbot.wait(cs._EXTENTS_SAVE_MS + 100)
    assert len(writes) == 1, "the release left a second write waiting"


def test_a_held_handle_that_rests_is_written_once(qtbot, store, writes):
    split = _splitter(qtbot, "smooth::rest")
    _drag(split, steps=5)
    assert writes == []
    qtbot.waitUntil(lambda: len(writes) == 1, timeout=3000)
    assert cs.get_pane_extents("smooth::rest")["Settings"] == split.sizes()[0]


def test_a_move_made_in_code_is_written_at_once(qtbot, store, writes):
    split = _splitter(qtbot, "smooth::code")
    split.moveSplitter(350, 1)
    assert writes == [("smooth::code", split.extents())]


def test_hiding_the_splitter_writes_a_size_that_is_still_waiting(
        qtbot, store, writes):
    split = _splitter(qtbot, "smooth::hide")
    _drag(split, steps=5)
    assert writes == []
    split.hide()
    assert writes == [("smooth::hide", split.extents())]
    split.show()
    _pump()
    split.hide()
    assert len(writes) == 1, "a hide with nothing waiting wrote again"


def test_a_drag_ends_where_it_did_when_every_move_wrote(
        qtbot, store, writes, monkeypatch):
    """The same drag, with the old write on every move, ends identically."""
    split = _splitter(qtbot, "smooth::same")
    start = split.sizes()
    drag, travelled = _drag(split)
    drag.release(travelled)
    _pump()
    new = (split.sizes(), split.extents(), split.grab().toImage())

    split.setSizes(start)
    _pump()
    monkeypatch.setattr(split._extents_timer, "start", split._save_extents)
    drag, travelled = _drag(split)
    drag.release(travelled)
    _pump()
    old = (split.sizes(), split.extents(), split.grab().toImage())
    assert new[0] == old[0] and new[1] == old[1]
    assert new[2] == old[2], "the drag's end state is not pixel-identical"


def test_a_drag_move_costs_a_fraction_of_the_write_it_used_to_make(
        qtbot, store):
    split = _splitter(qtbot, "smooth::time")
    drag = _Drag(split.handle(1))
    moves = []
    for i in range(40):
        started = time.perf_counter()
        drag.move(4 * (i % 20 + 1))
        moves.append(time.perf_counter() - started)
    drag.release(4)
    _pump()
    old = []
    for _ in range(15):
        started = time.perf_counter()
        cs.set_pane_extents("smooth::time", split.extents())
        old.append(time.perf_counter() - started)
    move, write = statistics.median(moves), statistics.median(old)
    assert move < 0.5 * write, (
        f"a drag move took {move * 1000:.2f} ms against "
        f"{write * 1000:.2f} ms for the write each move used to make")


# --------------------------------------------------------------------------
# Ctrl + wheel over a column
# --------------------------------------------------------------------------

class _Column(QWidget):
    """A column of labels that counts the sheets it is given."""

    def __init__(self, rows=60):
        super().__init__()
        self.sheets = 0
        layout = QVBoxLayout(self)
        self.labels = [QLabel(f"Console line {i} ▾ ↓ …", self)
                       for i in range(rows)]
        for label in self.labels:
            layout.addWidget(label)
        self.button = QPushButton("↓ jump to the end", self)
        layout.addWidget(self.button)

    def setStyleSheet(self, sheet):                     # noqa: N802
        self.sheets += 1
        super().setStyleSheet(sheet)

    def state(self):
        return (self.styleSheet(),
                [w.font().pixelSize() for w in self.labels + [self.button]])


@pytest.fixture
def column(qtbot, qt_theme_applied, store):
    from spacr.qt.live_zoom import ColumnTextScale

    store.set_runtime_text_scale(1.0)
    root = _Column()
    qtbot.addWidget(root)
    root.resize(700, 1400)
    root.show()
    qtbot.waitExposed(root)
    scale = ColumnTextScale()
    scale.register(root)
    _pump()
    root.sheets = 0
    yield scale, root
    scale.deleteLater()


def _notch(scale, widget, notches=1):
    event = QWheelEvent(QPointF(5, 5), QPointF(widget.mapToGlobal(QPoint(5, 5))),
                        QPoint(0, 0), QPoint(0, int(120 * notches)),
                        Qt.NoButton, Qt.ControlModifier, Qt.NoScrollPhase,
                        False)
    return scale.eventFilter(widget, event)


def test_ten_notches_restyle_the_column_once_when_the_wheel_rests(
        qtbot, column):
    scale, root = column
    for _ in range(10):
        assert _notch(scale, root.labels[5])
        _pump(2)
    assert scale.scale() == pytest.approx(2.0)
    assert root.sheets == 0, "a notch restyled the column"
    qtbot.waitUntil(lambda: root.sheets == 1, timeout=3000)
    qtbot.wait(300)
    assert root.sheets == 1


def test_the_size_a_turn_of_the_wheel_leaves_is_saved(qtbot, column, store):
    from spacr.qt.live_zoom import ColumnTextScale

    scale, root = column
    for _ in range(4):
        _notch(scale, root.button)
    qtbot.waitUntil(
        lambda: store.get_runtime_text_scale() == pytest.approx(1.4),
        timeout=3000)
    assert ColumnTextScale().scale() == pytest.approx(1.4)


def test_the_wheel_ends_where_a_restyle_a_notch_did(qtbot, column):
    """Same sheet, same font sizes, same pixels as the old path."""
    scale, root = column
    for _ in range(6):
        _notch(scale, root.labels[3])
    qtbot.waitUntil(lambda: root.sheets == 1, timeout=3000)
    _pump()
    new = root.state(), root.grab().toImage()

    scale.reset()
    _pump()
    for step in range(1, 7):
        scale.set_scale(1.0 + 0.1 * step)
        scale.restyle_all()
        _pump(2)
    _pump()
    old = root.state(), root.grab().toImage()
    assert new[0] == old[0]
    assert new[1] == old[1], "the zoom's end state is not pixel-identical"


def test_a_notch_costs_a_fraction_of_the_restyle_it_used_to_make(
        qtbot, column):
    scale, root = column
    restyles = []
    for step in range(4):
        scale.set_scale(1.05 + 0.1 * step)
        started = time.perf_counter()
        scale.restyle_all()
        restyles.append(time.perf_counter() - started)
    scale.reset()
    _pump()
    notches = []
    for i in range(10):
        started = time.perf_counter()
        _notch(scale, root.labels[i], 1 if i < 5 else -1)
        _pump(1)
        notches.append(time.perf_counter() - started)
    notch, restyle = statistics.median(notches), statistics.median(restyles)
    assert notch < 0.25 * restyle, (
        f"a Ctrl + wheel notch took {notch * 1000:.2f} ms against "
        f"{restyle * 1000:.2f} ms for the restyle each notch used to make")
