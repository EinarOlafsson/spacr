"""Item 471, slice C: the figures, tables and panels of the G-to-P screens fold.

Screens from Gate Editor to PCA. Each one's figures, tables and side panels
fold by the one heading control (or, where a heading would break a layout
the maintainer asked for, by the handle beside them), a folded one sits at
the bottom of its room, a dragged width or height is remembered, and a
panel built on first use or kept hidden by its owner stays that way.

The preference store is isolated per test by ``tests/qt/conftest.py``, so
the screens' own keys are used as they are.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt              # noqa: E402
from PySide6.QtGui import QMouseEvent                       # noqa: E402
from PySide6.QtWidgets import QApplication                  # noqa: E402

from spacr.qt.widgets import collapsible_splitter as cs     # noqa: E402


def _pump(n: int = 12) -> None:
    for _ in range(n):
        QApplication.processEvents()


def _click(label) -> None:
    at = QPointF(3, 3)
    QApplication.sendEvent(label, QMouseEvent(
        QEvent.MouseButtonRelease, at, at, Qt.LeftButton, Qt.LeftButton,
        Qt.NoModifier))


def _bottom_of(widget, container) -> int:
    """Where ``widget``'s bottom edge is, in ``container``'s coordinates."""
    return widget.mapTo(container, widget.rect().bottomLeft()).y()


def _assert_heading_at_bottom(section, slack: int = 8) -> None:
    """A folded section's heading is at the bottom of the section's room."""
    assert section.shut
    assert not section.body.isVisible()
    assert _bottom_of(section.heading, section) >= section.height() - slack


def _show(widget, width=1400, height=900):
    widget.resize(width, height)
    widget.show()
    _pump()
    return widget


# ---------------------------------------------------------------------------
# Gate Editor
# ---------------------------------------------------------------------------

@pytest.fixture()
def gate_screen(qapp, qtbot):
    from spacr.qt.screens.gate_editor import GateEditorScreen

    screen = GateEditorScreen(threaded=False)
    qtbot.addWidget(screen)
    return _show(screen)


@pytest.mark.qt
def test_the_gate_graph_and_gate_table_fold_by_their_headings(gate_screen):
    gates = gate_screen.gates
    assert isinstance(gates.body, cs.CollapsibleSplitter)
    assert gates.canvas_section.body is gates.canvas
    assert gates.tree_section.body is gates.tree

    _click(gates.tree_section.heading)
    _pump()
    _assert_heading_at_bottom(gates.tree_section)
    assert gates.canvas.isVisible(), "the graph went with the table"

    gates.tree_section.set_folded(False)
    gates.canvas_section.set_folded(True)
    _pump()
    _assert_heading_at_bottom(gates.canvas_section)
    assert gates.tree.isVisible()


@pytest.mark.qt
def test_the_side_tabs_and_console_collapse_to_the_right(gate_screen):
    from spacr.qt.screens.gate_editor import CONSOLE_PANE, SIDE_PANE

    body = gate_screen._body
    assert body.pane(SIDE_PANE).mode == cs.EDGE
    assert body.is_collapsed(CONSOLE_PANE), "the console starts collapsed"
    assert body.sizes()[2] == 0
    assert 200 <= body.sizes()[1] <= 320, body.sizes()

    body.toggle_pane(SIDE_PANE)
    _pump()
    assert body.sizes()[1] == 0
    body.toggle_pane(CONSOLE_PANE)
    _pump()
    assert body.sizes()[2] >= gate_screen.console.minimumWidth()


@pytest.mark.qt
def test_the_tab_strip_stays_level_with_the_graphs_one_row(gate_screen):
    """The maintainer's one-row layout survives the fold machinery."""
    screen = gate_screen
    tabs_top = screen.side_tabs.mapTo(screen, screen.side_tabs.rect()
                                      .topLeft()).y()
    panel_top = screen.gates.mapTo(screen, screen.gates.rect()
                                   .topLeft()).y()
    assert tabs_top == panel_top
    screen.align_side_panel()
    _pump()
    page = screen.side_tabs.currentWidget()
    graph = screen.gates.body
    assert abs(page.mapTo(screen, page.rect().topLeft()).y()
               - graph.mapTo(screen, graph.rect().topLeft()).y()) <= 2


@pytest.mark.qt
def test_a_dragged_gate_table_width_is_remembered(gate_screen, qtbot):
    from spacr.qt.screens.gate_editor import GateEditorScreen
    from spacr.qt.widgets.gate_editor import GRAPH_SPLIT_KEY

    split = gate_screen.gates.body
    split.moveSplitter(split.sizes()[0] - 150, 1)
    _pump()
    dragged = cs.get_pane_extents(GRAPH_SPLIT_KEY).get("Gate table", 0)
    assert dragged >= 350

    again = GateEditorScreen(threaded=False)
    qtbot.addWidget(again)
    _show(again)
    assert abs(again.gates.body.sizes()[1] - dragged) <= 8


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------
