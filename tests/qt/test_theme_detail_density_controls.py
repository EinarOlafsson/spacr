"""Rendered theme controls preserve population and bound actual pixel sampling."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QImage, QPainter

from spacr.qt.widgets import ambient


@pytest.mark.parametrize("theme", ambient.AMBIENT_THEMES)
def test_detail_changes_sampling_without_removing_elements(theme):
    engine = ambient.make_engine(theme, "spacr", "#101418", seed=23,
                                 density=3.0, resolution=0.25)
    engine.set_time(36.0)
    low_size = engine.buffer_size(1280, 720)
    low_count = len(engine.geometry(1280, 720))
    engine.set_resolution(2.0)
    high_size = engine.buffer_size(1280, 720)
    assert high_size[0] * high_size[1] > low_size[0] * low_size[1]
    assert len(engine.geometry(1280, 720)) == low_count
    assert high_size[0] * high_size[1] <= engine.max_pixels
    engine.set_density(0.25)
    assert len(engine.geometry(1280, 720)) < low_count


def test_stratified_detail_changes_rendered_pixels_without_changing_stars():
    engine = ambient.make_engine("drift", "spacr", "#101418", seed=19,
                                 resolution=1.0, density=2.0)
    engine.set_time(13.0)
    before = engine.geometry(960, 540)

    def render():
        image = QImage(960, 540, QImage.Format_RGB32)
        image.fill(QColor("#101418"))
        painter = QPainter(image)
        try:
            engine.paint(painter, 960, 540)
        finally:
            painter.end()
        return bytes(image.constBits())

    detailed = render()
    engine.set_resolution(0.25)
    assert engine.geometry(960, 540) == before
    assert engine.buffer_size(960, 540) == (240, 135)
    assert render() != detailed
    engine.set_resolution(1.0)
    assert render() == detailed


def test_stratified_render_buffer_respects_display_budget():
    engine = ambient.make_engine("drift", "spacr", "#101418", seed=19)
    engine.set_max_pixels(320 * 180)
    bw, bh = engine.buffer_size(3840, 2160)
    assert bw * bh <= 320 * 180
    assert len(engine.geometry(3840, 2160)) > len(engine.geometry(320, 180))


@pytest.mark.parametrize("theme", ambient.AMBIENT_THEMES)
def test_legacy_blur_cannot_soften_displayed_themes(qtbot, theme):
    widget = ambient.AmbientWidget(theme=theme, palette="spacr", seed=19,
                                   blur=3.0, speed=1.0, size=1.0,
                                   resolution=1.0, density=1.0, direction="up")
    qtbot.addWidget(widget)
    assert widget.blur() == widget.engine.blur == 0.0
    widget.set_blur(2.0)
    widget.set_theme("drift")
    assert widget.blur() == widget.engine.blur == 0.0
