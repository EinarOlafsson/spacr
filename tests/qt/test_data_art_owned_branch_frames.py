"""Growth and rain retain exact field pixels with independent native frames."""

from __future__ import annotations

import pytest
from PySide6.QtGui import QPainter

from spacr.qt.widgets import ambient

KEYS = ('data_art_fungal_growth', 'data_art_thore')


def _engine(key, background='#101418', blur=0.0):
    engine = ambient.make_engine(key, 'spacr', background, seed=42,
                                 resolution=2, density=1, blur=blur)
    engine.set_colors(['#ff7733', '#33ddff'])
    engine.set_max_pixels(3840 * 2160)
    return engine


@pytest.mark.parametrize('key', KEYS)
@pytest.mark.parametrize('background', ['#101418', '#f6f7f9'])
@pytest.mark.parametrize('blur', [0.0, 0.6])
def test_native_owned_branches_match_the_original_buffered_painter(
        key, background, blur, qapp):
    actual = _engine(key, background, blur)
    reference = _engine(key, background, blur)
    for stamp in [0.0, 16.97, 97.0, 180.0, 16.97]:
        actual.set_time(stamp)
        reference.set_time(stamp)
        image = actual.shade(3840, 2160)
        original = ambient._BufferedEngine._shade(reference, 3840, 2160)
        assert image.size() == original.size()
        assert image.bits().tobytes() == original.bits().tobytes()
    assert actual._buffer is None


@pytest.mark.parametrize('key', KEYS)
def test_published_branch_image_survives_reset_resize_and_caller_mutation(key, qapp):
    engine = _engine(key)
    engine.set_time(97)
    first = engine.shade(640, 360)
    original = first.bits().tobytes()
    engine.advance(.25)
    second = engine.shade(640, 360)
    second_original = second.bits().tobytes()
    assert first is not second
    assert first.bits().tobytes() == original
    first.fill('red')
    engine.shade(800, 450)
    engine.set_time(0)
    engine.set_colors(['#33cc88', '#ff44aa'])
    engine.shade(320, 180)
    assert second.bits().tobytes() == second_original
    assert engine.shade(0, 180) is None
    assert engine.shade(320, 0) is None


@pytest.mark.parametrize('key', KEYS)
def test_branch_draw_error_ends_painter_and_preserves_published_frame(
        key, qapp, monkeypatch):
    engine = _engine(key)
    first = engine.shade(640, 360)
    original = first.bits().tobytes()
    paint_field = engine._paint_field
    probes = []

    def fail_after_partial_draw(painter, width, height):
        painter.fillRect(0, 0, 10, 10, 'red')
        probes.append(painter)
        raise RuntimeError('injected field failure')

    monkeypatch.setattr(engine, '_paint_field', fail_after_partial_draw)
    with pytest.raises(RuntimeError, match='injected field failure'):
        engine.shade(640, 360)
    assert probes and not probes[0].isActive()
    assert first.bits().tobytes() == original
    monkeypatch.setattr(engine, '_paint_field', paint_field)
    recovered = engine.shade(640, 360)
    assert recovered.bits().tobytes() == original


@pytest.mark.parametrize('key', KEYS)
def test_owned_branch_synchronous_paint_keeps_original_blur_composition(key, qapp):
    from PySide6.QtGui import QImage

    engine = _engine(key, '#f6f7f9', .6)
    engine.set_time(16.97)
    painted = QImage(640, 360, QImage.Format_RGB32)
    composed = QImage(640, 360, QImage.Format_RGB32)
    for image in (painted, composed):
        image.fill('#f6f7f9')
    painter = QPainter(painted)
    try:
        engine.paint(painter, 640, 360)
    finally:
        painter.end()
    original = ambient._BufferedEngine._shade(engine, 640, 360)
    painter = QPainter(composed)
    try:
        engine.blit(painter, original, 640, 360)
    finally:
        painter.end()
    assert painted.bits().tobytes() == composed.bits().tobytes()
