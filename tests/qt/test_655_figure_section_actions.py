"""Assistive clients can identify and operate actual figure-section folds."""

import gc
import json

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QAccessible, QAccessibleActionInterface, QPixmap
from PySide6.QtWidgets import QApplication, QFrame

from spacr.qt import i18n
from spacr.qt.widgets.figure_grid_view import FigureGridView


@pytest.fixture
def grid(qtbot):
    view = FigureGridView()
    image = QPixmap(120, 90)
    image.fill()
    view.set_figures([image, image], ["First", "Second"], [("Run A", 0, 2)])
    qtbot.addWidget(view)
    view.resize(640, 480)
    view.show()
    qtbot.waitExposed(view)
    yield view


def _header(grid, name="Run A"):
    return next(header for header in grid._headers if header.text() == name)


def _press(grid, name="Run A"):
    action = QAccessible.queryAccessibleInterface(_header(grid, name)).actionInterface()
    assert QAccessibleActionInterface.pressAction() in action.actionNames()
    action.doAction(QAccessibleActionInterface.pressAction())
    QApplication.processEvents()


def test_actual_header_has_fold_button_role_and_expansion_state(grid):
    header = _header(grid)
    interface = QAccessible.queryAccessibleInterface(header)
    state = interface.state()
    print(json.dumps({"role": interface.role().name,
                      "actions": interface.actionInterface().actionNames(),
                      "expandable": bool(state.expandable),
                      "expanded": bool(state.expanded),
                      "collapsed": bool(state.collapsed)}))
    assert interface.role() == QAccessible.Button
    assert state.focusable and state.expandable and state.expanded
    assert not state.collapsed and not state.checkable
    for language in ("en", "ko"):
        i18n.retranslate_widget_tree(grid, language)
        assert interface.text(QAccessible.Name) == "Run A"
    header.setAccessibleName("Caller supplied run identity")
    assert interface.text(QAccessible.Name) == "Caller supplied run identity"


def test_press_folds_real_cells_and_preserves_saved_state_on_rebuild(grid):
    width = grid.workspace_state()["cell_width"]
    cell_sizes = [(cell.width(), cell.height()) for cell in grid._cells]
    assert all(cell.isVisible() for cell in grid._cells)
    _press(grid)
    assert grid.is_section_collapsed("Run A", 0)
    assert not any(cell.isVisible() for cell in grid._cells)
    folded = QAccessible.queryAccessibleInterface(_header(grid)).state()
    assert folded.expandable and folded.collapsed and not folded.expanded
    state = json.loads(json.dumps(grid.workspace_state()))
    assert state["collapsed"] == [["Run A", 0]]
    assert state["cell_width"] == width
    image = QPixmap(120, 90)
    image.fill()
    grid.set_figures([image, image], ["First", "Second"], [("Run A", 0, 2)])
    assert QAccessible.queryAccessibleInterface(_header(grid)).state().collapsed
    assert not any(cell.isVisible() for cell in grid._cells)
    _press(grid)
    expanded = QAccessible.queryAccessibleInterface(_header(grid)).state()
    assert expanded.expanded and not expanded.collapsed
    assert all(cell.isVisible() for cell in grid._cells)
    assert [(cell.width(), cell.height()) for cell in grid._cells] == cell_sizes
    assert grid.workspace_state() == {"cell_width": width, "collapsed": []}
    assert _header(grid).text() == "Run A"


def test_offscreen_section_is_reached_before_it_is_folded(grid):
    image = QPixmap(120, 90)
    image.fill()
    grid.resize(700, 300)
    grid.set_figures([image] * 24, [str(i) for i in range(24)],
                     [("Run A", 0, 12), ("Run B", 12, 12)])
    QApplication.processEvents()
    bar = grid.verticalScrollBar()
    bar.setValue(0)
    assert bar.maximum() > 0 and not grid._is_raised(_header(grid, "Run B"))
    _press(grid, "Run B")
    assert not grid.is_section_collapsed("Run B", 12)
    assert bar.value() > 0 and grid._is_raised(_header(grid, "Run B"))
    _press(grid, "Run B")
    assert grid.is_section_collapsed("Run B", 12)


def test_disabled_header_refuses_assistive_press(grid):
    header = _header(grid)
    interface = QAccessible.queryAccessibleInterface(header)
    action = interface.actionInterface()
    header.setEnabled(False)
    assert interface.state().disabled and action.actionNames() == []
    before = grid.workspace_state()
    action.doAction(QAccessibleActionInterface.pressAction())
    assert grid.workspace_state() == before
    assert _header(grid) is header


def test_rebuilt_header_retires_its_qt_owned_accessible_interface(grid):
    header = _header(grid)
    interface = QAccessible.queryAccessibleInterface(header)
    identity = QAccessible.uniqueId(interface)
    assert interface.parent().object() is header.parentWidget()
    del interface
    gc.collect()
    assert QAccessible.uniqueId(QAccessible.queryAccessibleInterface(header)) == identity
    _press(grid)
    assert _header(grid) is not header
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert QAccessible.accessibleInterface(identity) is None
    current = QAccessible.queryAccessibleInterface(_header(grid))
    assert current.isValid() and current.state().collapsed


def test_unrelated_frames_keep_native_accessibility(qtbot):
    frame = QFrame()
    qtbot.addWidget(frame)
    interface = QAccessible.queryAccessibleInterface(frame)
    assert interface.role() != QAccessible.Button
    assert not interface.state().expandable
    assert QAccessibleActionInterface.pressAction() not in interface.actionInterface().actionNames()
