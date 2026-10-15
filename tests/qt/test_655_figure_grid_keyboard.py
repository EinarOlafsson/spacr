"""Keyboard users can reach, identify and open every kind of figure tile."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QAccessible, QColor, QPalette, QPixmap
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget

from spacr.qt.widgets.figure_grid_view import FigureGridView, _FigureCell


@pytest.fixture
def figure_grid(qtbot):
    host = QWidget()
    layout = QVBoxLayout(host)
    before = QPushButton("Before")
    grid = FigureGridView()
    after = QPushButton("After")
    for widget in (before, grid, after):
        layout.addWidget(widget)
    pixmap = QPixmap(80, 60)
    pixmap.fill(Qt.white)
    grid.set_figures([pixmap, None, pixmap], ["Plate one", "", "Plate three"])
    grid.set_live_tiles([("qq", pixmap, "Q-Q"),
                         ("regression", pixmap, "Volcano")])
    host.resize(640, 800)
    qtbot.addWidget(host)
    host.show()
    host.activateWindow()
    qtbot.wait(50)
    before.setFocus(Qt.TabFocusReason)
    qtbot.waitUntil(before.hasFocus)
    yield before, grid, after


def test_tab_reaches_saved_and_live_tiles_and_can_leave_the_grid(figure_grid, qtbot):
    before, grid, after = figure_grid
    cells = set(grid._cells + grid._live)
    reached = set()
    left_the_grid = False
    for _ in range(20):
        focused = QApplication.focusWidget()
        assert focused is not None
        qtbot.keyClick(focused, Qt.Key_Tab)
        focused = QApplication.focusWidget()
        if focused in cells:
            reached.add(focused)
        left_the_grid = left_the_grid or focused is after
        if focused is before:
            break
    assert reached == cells
    assert left_the_grid


@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space])
@pytest.mark.parametrize("kind", ["saved", "live", "pinned"])
def test_keyboard_activation_keeps_the_saved_index_or_live_key(
        figure_grid, qtbot, kind, key):
    _, grid, _ = figure_grid
    saved, live, pinned = [], [], []
    grid.figure_activated.connect(saved.append)
    grid.live_tile_activated.connect(live.append)
    grid.pinned_activated.connect(lambda: pinned.append(True))
    cell = {"saved": grid._cells[1], "live": grid._live[0],
            "pinned": grid._live[1]}[kind]
    cell.setFocus(Qt.TabFocusReason)
    qtbot.keyClick(cell, key)
    assert saved == ([2] if kind == "saved" else [])
    assert live == (["qq"] if kind == "live" else
                    ["regression"] if kind == "pinned" else [])
    assert pinned == ([True] if kind == "pinned" else [])


@pytest.mark.parametrize("modifier", [Qt.ControlModifier, Qt.AltModifier,
                                      Qt.ShiftModifier])
def test_modified_activation_keys_do_not_open_figures(figure_grid, qtbot, modifier):
    _, grid, _ = figure_grid
    opened = []
    grid.figure_activated.connect(opened.append)
    cell = grid._cells[0]
    cell.setFocus(Qt.TabFocusReason)
    for key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
        qtbot.keyClick(cell, key, modifier)
    assert opened == []


@pytest.mark.parametrize("title,letter,live_key,expected", [
    ("Plate three", "B", "", "Plate three"),
    ("", "B", "", "B"),
    ("", "", "qq", "qq"),
])
def test_accessibility_names_reuse_existing_figure_identity(
        qtbot, title, letter, live_key, expected):
    pixmap = QPixmap(80, 60)
    pixmap.fill(Qt.white)
    cell = _FigureCell(2, pixmap, title, letter=letter, live_key=live_key)
    qtbot.addWidget(cell)
    interface = QAccessible.queryAccessibleInterface(cell)
    assert interface.text(QAccessible.Name) == expected


def test_keyboard_focus_is_visible_on_the_figure_border(figure_grid):
    before, grid, _ = figure_grid
    cell = grid._cells[0]
    palette = cell.palette()
    highlight = QColor("#da21a6")
    palette.setColor(QPalette.Highlight, highlight)
    cell.setPalette(palette)
    before.setFocus()
    plain = cell.grab().toImage()
    cell.setFocus(Qt.TabFocusReason)
    assert cell.hasFocus()
    focused = cell.grab().toImage()
    point = (1, cell.height() // 2)
    assert focused.pixelColor(*point) == highlight
    assert plain.pixelColor(*point) != highlight
