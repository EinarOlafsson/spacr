"""Toggle closes its native painter on success and at every failing paint stage."""

import pytest
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QStyle

from spacr.qt.widgets import toggle as module


@pytest.mark.parametrize("fault", [
    None, "render_hint", "widget_focus", "focus_rect", "palette", "switch", "caption",
])
def test_painter_is_ended_even_when_a_paint_stage_raises(qtbot, monkeypatch, fault):
    image = QImage(240, 50, QImage.Format_ARGB32_Premultiplied)
    painters = []
    switch = module.Toggle("Preview")
    switch.resize(image.size())
    qtbot.addWidget(switch)

    def fail_at(stage):
        if fault == stage:
            raise RuntimeError("paint stage: " + stage)

    class RetainedPainter(QPainter):
        def __init__(self, widget):
            super().__init__(image)
            painters.append(self)

        def setRenderHint(self, *args):
            fail_at("render_hint")
            return super().setRenderHint(*args)

        def drawRoundedRect(self, rect, x_radius, y_radius):
            if rect == switch.rect().adjusted(4, 4, -4, -4):
                fail_at("focus_rect")
            return super().drawRoundedRect(rect, x_radius, y_radius)

    class FocusStyle:
        def drawPrimitive(self, primitive, option, painter, widget):
            if primitive == QStyle.PE_Widget:
                fail_at("widget_focus")

    original_palette = module._painting_palette
    original_switch = switch._paint_switch
    original_caption = switch._paint_caption

    def palette():
        fail_at("palette")
        return original_palette()

    def paint_switch(*args):
        fail_at("switch")
        return original_switch(*args)

    def paint_caption(*args):
        fail_at("caption")
        return original_caption(*args)

    monkeypatch.setattr(module, "QPainter", RetainedPainter)
    monkeypatch.setattr(module, "_painting_palette", palette)
    monkeypatch.setattr(switch, "hasFocus", lambda: True)
    monkeypatch.setattr(switch, "style", lambda: FocusStyle())
    monkeypatch.setattr(switch, "_paint_switch", paint_switch)
    monkeypatch.setattr(switch, "_paint_caption", paint_caption)
    try:
        if fault is None:
            switch.paintEvent(None)
        else:
            with pytest.raises(RuntimeError, match="paint stage: " + fault):
                switch.paintEvent(None)
        assert len(painters) == 1
        assert not painters[0].isActive(), "Native painter survived paintEvent"
    finally:
        for painter in painters:
            if painter.isActive():
                painter.end()
