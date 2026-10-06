"""Native point frames retain exact raster output and independent ownership."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QImage, QPainter
from spacr.qt.widgets import ambient

KEYS = ("data_art_point_atlas", "data_art_impulse_lens", "data_art_genetic_advection")


def _engine(key, background, blur):
    engine = ambient.make_engine(key, "spacr", background, seed=42,
                                 resolution=2.0, blur=blur)
    engine.max_pixels = 3840 * 2160
    engine.set_gravity_radius(0.65)
    engine.set_pointer((0.3, 0.6))
    engine.set_time(11.0)
    if engine.family == "impulse_lens":
        engine._add_impulse((0.4, 0.5))
    return engine


def _bytes(image):
    return bytes(image.constBits())


def _buffered_reference(engine, width, height):
    bw, bh = engine.buffer_size(width, height)
    image = QImage(bw, bh, QImage.Format_RGB32)
    painter = QPainter(image)
    try:
        painter.fillRect(image.rect(), engine.identity)
        painter.setCompositionMode(engine.mode)
        engine._paint_field(painter, bw, bh)
    finally:
        painter.end()
    return engine._soften(image, width, height)


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("background", ("#101418", "#ffffff"))
@pytest.mark.parametrize("blur", (0.0, 0.6))
def test_native_owned_frame_matches_the_original_buffered_composition(
    key, background, blur, qapp
):
    direct = _engine(key, background, blur)
    buffered = _engine(key, background, blur)
    actual = direct.shade(3840, 2160)
    expected = _buffered_reference(buffered, 3840, 2160)
    assert actual.size() == expected.size()
    assert _bytes(actual) == _bytes(expected)


@pytest.mark.parametrize("key", KEYS)
def test_published_image_survives_later_shades_resizes_and_caller_mutation(key, qapp):
    engine = _engine(key, "#101418", 0.0)
    first = engine.shade(640, 360)
    original = _bytes(first)
    engine.advance(0.25)
    second = engine.shade(640, 360)
    assert _bytes(first) == original
    assert first is not second
    first.fill("red")
    before_resize = _bytes(second)
    engine.shade(800, 450)
    engine.set_gravity_radius(0.0)
    engine.shade(320, 180)
    assert _bytes(second) == before_resize
    assert engine.shade(0, 180) is None
    assert engine.shade(320, 0) is None
