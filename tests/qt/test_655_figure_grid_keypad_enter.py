"""Real numeric Enter activates the same saved/live/pinned figure route."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QVBoxLayout, QWidget

from spacr.qt.widgets.figure_grid_view import FigureGridView


@pytest.fixture
def grid(qtbot):
    host = QWidget()
    layout = QVBoxLayout(host)
    view = FigureGridView()
    layout.addWidget(view)
    pixmap = QPixmap(80, 60)
    pixmap.fill(Qt.white)
    view.set_figures([pixmap, None, pixmap], ["First", "", "Third"])
    view.set_live_tiles([("qq", pixmap, "Q-Q"),
                         ("regression", pixmap, "Volcano")])
    host.resize(640, 800)
    qtbot.addWidget(host)
    host.show()
    host.activateWindow()
    yield view


@pytest.mark.parametrize("kind", ["saved", "live", "pinned"])
def test_numeric_enter_keeps_the_actual_figure_route(grid, qtbot, kind):
    saved, live, pinned = [], [], []
    grid.figure_activated.connect(saved.append)
    grid.live_tile_activated.connect(live.append)
    grid.pinned_activated.connect(lambda: pinned.append(True))
    cell = {"saved": grid._cells[1], "live": grid._live[0],
            "pinned": grid._live[1]}[kind]
    cell.setFocus(Qt.TabFocusReason)
    qtbot.waitUntil(cell.hasFocus)
    qtbot.keyClick(cell, Qt.Key_Enter, Qt.KeypadModifier)
    assert saved == ([2] if kind == "saved" else [])
    assert live == (["qq"] if kind == "live" else
                    ["regression"] if kind == "pinned" else [])
    assert pinned == ([True] if kind == "pinned" else [])


@pytest.mark.parametrize("modifier", [Qt.ControlModifier, Qt.AltModifier,
                                      Qt.ShiftModifier, Qt.MetaModifier])
def test_modified_numeric_enter_does_not_open_a_figure(grid, qtbot, modifier):
    opened = []
    grid.figure_activated.connect(opened.append)
    cell = grid._cells[0]
    cell.setFocus(Qt.TabFocusReason)
    qtbot.waitUntil(cell.hasFocus)
    qtbot.keyClick(cell, Qt.Key_Enter, Qt.KeypadModifier | modifier)
    assert opened == []


@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Space])
def test_keypad_modifier_does_not_expand_other_activation_keys(grid, qtbot, key):
    opened = []
    grid.figure_activated.connect(opened.append)
    cell = grid._cells[0]
    cell.setFocus(Qt.TabFocusReason)
    qtbot.waitUntil(cell.hasFocus)
    qtbot.keyClick(cell, key, Qt.KeypadModifier)
    assert opened == []
