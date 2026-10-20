"""A hidden chip takes no room in a FlowLayout.

Counted, a hidden last button could wrap onto a line of its own: the
Measure action row then reported a height one line taller than it drew, the
runtime splitter gave the Actions pane that height, and switching the QC
toggle on moved the Run row (N47 serial run, 2026-10-11).
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QPushButton, QWidget

from spacr.qt.widgets.flow import FlowHost, FlowLayout

pytestmark = pytest.mark.qt


def test_a_hidden_trailing_button_adds_no_line(qtbot):
    host = FlowHost()
    qtbot.addWidget(host)
    layout = FlowLayout(host, spacing=8)
    shown = [QPushButton(f"Button {n}", host) for n in range(3)]
    for button in shown:
        layout.addWidget(button)
    hidden = QPushButton("File as issue", host)
    layout.addWidget(hidden)
    hidden.hide()
    host.show()
    one_line = max(button.sizeHint().height() for button in shown)
    width = sum(b.sizeHint().width() for b in shown) + 2 * 8

    assert layout.heightForWidth(width) == one_line

    hidden.show()
    assert layout.heightForWidth(width) > one_line
