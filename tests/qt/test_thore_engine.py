"""The rain and lightning backdrop stays seeded, sparse and restrained."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QImage, QPainter

from spacr.qt.widgets.ambient import _ThoreEngine


PALETTE = ("#93bce8", "#a8d9ee", "#d7e2f2")


def _engine(background="#09121b", **options):
    """Construct the worker-safe private painter with a fixed seed."""
    return _ThoreEngine(PALETTE, background, seed=17, **options)


def _frame(engine, second, width=640, height=360):
    """Composite through the public engine paint path at a chosen clock."""
    engine.set_time(second)
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(engine.background)
    painter = QPainter(image)
    engine.paint(painter, width, height)
    painter.end()
    return image


def _pixels(image):
    """Read BGR pixels from the completed image without copying Qt state."""
    return np.frombuffer(image.bits(), dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine())[:, :image.width() * 4].reshape(
            image.height(), image.width(), 4)[:, :, :3]


def _digest(image):
    """Identify all output pixels at this exact clock."""
    return hashlib.sha256(image.bits().tobytes()).hexdigest()


def test_rain_continues_and_lightning_is_local_without_a_full_screen_flash():
    """A bright event leaves most of the screen near its resting luminance."""
    engine = _engine()
    flash = _pixels(_frame(engine, 0.17)).copy()
    rest = _pixels(_frame(engine, 1.0)).copy()
    assert np.mean(np.any(flash != rest, axis=2)) > 0.5
    assert abs(float(flash.mean()) - float(rest.mean())) < 12.0
    assert np.mean(np.max(flash, axis=2) > 100) < 0.03
    assert _digest(_frame(engine, 1.0)) != _digest(_frame(engine, 1.1))


@pytest.mark.parametrize("background", ("#09121b", "#f0f1ed"))
def test_rain_is_visible_on_dark_and_light_pages(background):
    """The composition mode leaves a sparse legible field on either page."""
    engine = _engine(background)
    frame = _pixels(_frame(engine, 1.0))
    page = np.array([engine.background.blue(), engine.background.green(),
                     engine.background.red()])
    painted = float(np.any(frame != page, axis=2).mean())
    assert 0.001 < painted < 0.25


def test_seeded_bolts_and_rain_survive_arbitrary_clock_seek_with_bounded_cache():
    """A time seek rebuilds the same event without retaining an hour of rain."""
    first = _engine()
    second = _engine()
    near = _digest(_frame(first, 0.17))
    cached = first._bolt_cache[0]
    assert _digest(_frame(first, 0.17)) == near
    assert first._bolt_cache[0] is cached
    expected = _digest(_frame(first, 3600.17))
    assert _digest(_frame(second, 3600.17)) == expected
    for index in range(100):
        _frame(first, index * 8.4 + 0.17, 160, 90)
        assert len(first._bolt_cache) <= 4
    assert _digest(_frame(first, 3600.17)) == expected
    assert _digest(_frame(_ThoreEngine(PALETTE, "#09121b", seed=18),
                          3600.17)) != expected


def test_controls_change_density_geometry_clock_and_native_detail():
    """The four user controls have an observable effect on the actual field."""
    engine = _engine()
    engine.set_max_pixels(3840 * 2160)
    assert engine.buffer_size(3840, 2160) == (3840, 2160)
    original = _digest(_frame(engine, 2.3))
    engine.set_density(3.0)
    dense = _digest(_frame(engine, 2.3))
    assert dense != original
    engine.set_size(2.0)
    resized = _digest(_frame(engine, 2.3))
    assert resized != dense
    engine.set_colors(("#db8d82", "#a8d9ee"))
    assert _digest(_frame(engine, 2.3)) != resized
    engine.set_time(0.0)
    engine.set_speed(2.0)
    engine.advance(1.25)
    assert engine.time == 2.5
