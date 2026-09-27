"""The console's waiting AI heading, its column scale and empty warnings.

Pinned here, each as what the user sees:

* the "spaCR AI" heading a question left waiting is taken down only while
  nothing has been written under it; a reply already started keeps it; a
  heading that has already gone is left to Qt; a heading with no block is
  still taken down; a layout that has already let go does not stop it;
* the column's text scale re-zooms the console only when it changes;
* an empty warning adds nothing to the console, and a heading that is not
  inside a console copies nothing.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import console_panel as cp
from spacr.qt.widgets.console_panel import ConsolePanel

pytestmark = pytest.mark.qt


@pytest.fixture
def panel(qtbot):
    widget = ConsolePanel()
    qtbot.addWidget(widget)
    widget.resize(600, 400)
    return widget


def _waiting(panel, with_block=True):
    """The state a question leaves: a heading, and an empty reply block."""
    bar = panel.begin_topic("spaCR AI")
    block = None
    if with_block:
        block = cp._StdoutBlock(text_color="#4c8dff")
        panel._insert_entry(block)
        panel._current_stdout = block
    panel._pending_ai_topic = bar
    panel._pending_ai_block = block
    return bar, block


def _in_console(panel, widget):
    return any(panel._entries.itemAt(i) is not None
               and panel._entries.itemAt(i).widget() is widget
               for i in range(panel._entries.count()))


def test_an_empty_waiting_heading_is_taken_down(panel):
    bar, block = _waiting(panel)
    panel._take_down_the_waiting_heading()
    assert not _in_console(panel, bar)
    assert not _in_console(panel, block)
    assert panel._current_stdout is None
    assert panel._pending_ai_topic is None


def test_a_reply_already_started_keeps_its_heading(panel):
    bar, block = _waiting(panel)
    block.appendPlainText("The mask looks under-segmented because")
    assert panel._take_down_the_waiting_heading() is None
    assert _in_console(panel, bar)
    assert _in_console(panel, block)
    assert panel._pending_ai_topic is None


def test_a_heading_that_has_already_gone_is_left_to_qt(panel):
    class Gone:
        def text(self):
            raise RuntimeError("Internal C++ object already deleted.")

    before = panel._entries.count()
    panel._pending_ai_topic = Gone()
    panel._pending_ai_block = None
    assert panel._take_down_the_waiting_heading() is None
    assert panel._entries.count() == before


def test_a_heading_with_no_reply_block_is_still_taken_down(panel):
    bar, _block = _waiting(panel, with_block=False)
    panel._take_down_the_waiting_heading()
    assert not _in_console(panel, bar)


def test_a_layout_that_has_let_go_does_not_stop_the_take_down(panel,
                                                             monkeypatch):
    bar, block = _waiting(panel)
    dots = cp._WorkingDots(color="#4c8dff")
    panel._working_dots = dots

    def gone(_widget):
        raise RuntimeError("Internal C++ object already deleted.")

    monkeypatch.setattr(panel._entries, "removeWidget", gone)
    assert panel._take_down_the_waiting_heading() is dots
    assert dots.parent() is panel
    assert panel._current_stdout is None


def test_the_column_scale_rezooms_only_when_it_changes(panel, monkeypatch):
    zooms = []
    monkeypatch.setattr(panel, "apply_zoom", lambda: zooms.append(True))
    panel.apply_column_text_scale(1.0)
    assert zooms == []
    panel.apply_column_text_scale(1.5)
    assert zooms == [True]
    panel.apply_column_text_scale(1.5)
    assert zooms == [True]


def test_an_empty_warning_adds_nothing(panel):
    before = panel._entries.count()
    panel.append_warning("")
    assert panel._entries.count() == before


@pytest.mark.parametrize("depth", [0, 7])
def test_a_heading_outside_any_console_copies_nothing(qtbot, depth):
    from PySide6.QtWidgets import QApplication, QWidget

    top = QWidget()
    qtbot.addWidget(top)
    holder = top
    for _ in range(depth - 1):
        holder = QWidget(holder)
    bar = cp._TopicBar("spaCR output", parent=holder if depth else None)
    if not depth:
        qtbot.addWidget(bar)
    QApplication.clipboard().setText("kept")
    bar._copy_section()
    assert QApplication.clipboard().text() == "kept"
