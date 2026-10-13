"""The Gate Editor graph has one rounded opaque surface in 2D and 3D."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QVBoxLayout, QWidget

from spacr.qt import preferences, theme
from spacr.qt.widgets.gate_editor import GateCanvas
from spacr.qt.widgets import gate_editor
from spacr.qt.widgets.graph_spec import GraphSpec


class _ChangingBackdrop(QWidget):
    def __init__(self):
        super().__init__()
        self.ink = QColor("#fa00fa")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.ink)
        painter.end()


def test_gate_graph_ends_painter_when_panel_paint_fails(qtbot, monkeypatch):
    canvas = GateCanvas()
    qtbot.addWidget(canvas)
    ended = []

    class Painter:
        def __init__(self, widget):
            assert widget is canvas

        def end(self):
            ended.append(True)

    def fail(*args, **kwargs):
        raise RuntimeError("panel paint failed")

    monkeypatch.setattr(gate_editor, "QPainter", Painter)
    monkeypatch.setattr(gate_editor, "paint_panel", fail)
    with pytest.raises(RuntimeError, match="panel paint failed"):
        canvas.paintEvent(None)
    assert ended == [True]


@pytest.mark.parametrize("appearance", ("dark", "light"))
def test_gate_graph_hides_backdrop_but_keeps_rounded_corners_in_2d_and_3d(
        appearance, qtbot, qt_theme_applied, monkeypatch, tmp_path):
    store = QSettings(str(tmp_path / "preferences.ini"), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: store)
    original_palette = qt_theme_applied.palette()
    original_sheet = qt_theme_applied.styleSheet()
    preferences.set_theme(appearance)
    theme.apply_qpalette(qt_theme_applied)
    qt_theme_applied.setStyleSheet(theme.stylesheet())
    try:
        host = _ChangingBackdrop()
        qtbot.addWidget(host)
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        canvas = GateCanvas(host)
        layout.addWidget(canvas)
        host.resize(800, 560)
        host.show()
        qt_theme_applied.processEvents()
        canvas.set_frame(pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [5.0, 6.0, 7.0, 8.0],
            "c": [2.0, 3.0, 5.0, 7.0],
        }))
        canvas.set_spec(GraphSpec(x="a", y="b"))

        for width, height, mode in ((800, 560, "2D"),
                                    (1000, 620, "3D"),
                                    (760, 540, "2D")):
            host.ink = QColor("#fa00fa")
            host.resize(width, height)
            host.update()
            canvas.set_mode(mode, z_column="c")
            canvas._canvas.draw()
            qt_theme_applied.processEvents()
            assert canvas._axes[(0, 0)].name == (
                "3d" if mode == "3D" else "rectilinear")
            assert canvas._figure.patch.get_alpha() == 0.0
            figure = canvas._canvas.geometry()
            assert figure.left() >= 8 and figure.top() >= 8
            assert figure.right() <= width - 9

            first = host.grab().toImage()
            host.ink = QColor("#00ef00")
            host.update()
            qt_theme_applied.processEvents()
            second = host.grab().toImage()
            assert first.pixelColor(0, 0) != second.pixelColor(0, 0)
            for x, y in ((22, 22), (width // 2, height // 2),
                         (width - 22, height - 22)):
                assert first.pixelColor(x, y) == second.pixelColor(x, y), (
                    appearance, mode, width, height, x, y)
            assert second.pixelColor(22, 22) == QColor(
                theme.active_palette()["surface_alt"])
    finally:
        qt_theme_applied.setPalette(original_palette)
        qt_theme_applied.setStyleSheet(original_sheet)
