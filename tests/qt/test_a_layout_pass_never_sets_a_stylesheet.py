"""Asking a wrapping row for its height must not restyle anything.

A setStyleSheet inside heightForWidth or sizeHint re-polishes during layout
and can cascade into more layout. Counted, not timed: a chip row is laid out
at several widths and no widget may receive a stylesheet meanwhile.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel, QWidget

from spacr.qt.widgets.flow import FlowHost, FlowLayout

pytestmark = pytest.mark.qt


def test_wrapping_a_chip_row_sets_no_stylesheet(qtbot, qt_theme_applied,
                                                monkeypatch):
    host = QWidget()
    qtbot.addWidget(host)
    row = FlowHost(host)
    layout = FlowLayout(row)
    for index in range(30):
        layout.addWidget(QLabel(f"chip {index}"))
    host.show()
    calls = []
    original = QWidget.setStyleSheet

    def counting(self, text):
        calls.append(text)
        return original(self, text)

    monkeypatch.setattr(QWidget, "setStyleSheet", counting)
    for width in (120, 300, 640, 1200, 300):
        row.resize(width, 10)
        layout.heightForWidth(width)
        row.heightForWidth(width)
        row.sizeHint()
    assert calls == []
