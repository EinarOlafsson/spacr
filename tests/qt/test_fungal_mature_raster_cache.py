"""Mature filament reuse preserves native pixels, order, ownership and recovery."""

import gc
import weakref

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath

from spacr.qt import preferences
from spacr.qt.widgets import ambient


def _engine(palette='spacr', background='#101418', **options):
    engine = ambient.make_engine('data_art_fungal_growth', palette, background,
                                 seed=42, resolution=2, density=3, size=2.5,
                                 blur=0, **options)
    engine.set_max_pixels(3840 * 2160)
    engine.set_time(95)
    return engine


def _bytes(image):
    pixels = np.frombuffer(image.constBits(), dtype=np.uint32)
    assert np.all(pixels >> 24 == 255)
    return image.constBits().tobytes()


@pytest.mark.parametrize('palette', ['spacr', 'random'])
def test_native_cache_hits_keep_exact_original_pixels_and_independent_frames(palette, qapp,
                                                                            monkeypatch):
    engine = _engine(palette)
    reference = _engine(palette)
    reference._fungal_raster_failed = True
    hits = []
    reuse = engine._reuse_fungal_raster

    def record(*args):
        hit = reuse(*args)
        hits.append(hit)
        return hit

    monkeypatch.setattr(engine, '_reuse_fungal_raster', record)
    for _ in range(32):
        first = engine.shade(3840, 2160)
    original = _bytes(reference.shade(3840, 2160))
    assert any(hits) and _bytes(first) == original
    assert engine._owned_fungal_image is None
    entries = engine._fungal_rasters
    assert sum(entry[1].nbytes + entry[2].nbytes for entry in entries.values()) <= 8 * 1024**2
    assert all(entry[1].base is None and entry[2].base is None for entry in entries.values())
    frame_ref = weakref.ref(first)
    for stamp in [95.125, 95.25, 95.5, 95]:
        engine.set_time(stamp)
        reference.set_time(stamp)
        second = engine.shade(3840, 2160)
        assert _bytes(second) == _bytes(reference.shade(3840, 2160))
        assert _bytes(first) == original
    del first
    gc.collect()
    assert frame_ref() is None
    engine.set_size(1)
    assert not engine._fungal_rasters and not engine._fungal_observed
    assert _bytes(second) == original


@pytest.mark.parametrize('palette', ['spacr', 'random'])
def test_partial_optional_failure_repaints_once_and_keeps_traceback_pixels_alive(
        palette, qapp, monkeypatch):
    engine = _engine(palette)
    reference = _engine(palette)
    reference._fungal_raster_failed = True
    original = _bytes(reference.shade(960, 540))
    errors = []
    images = []
    paint = engine._paint_cached_fungal_paths

    def fail(painter, width, height, paths, mature):
        paint(painter, width, height, paths, mature)
        device = painter.device()
        borrowed = np.frombuffer(device.bits(), dtype=np.uint32)
        borrowed[:20] = 0xffaabbcc
        painter.translate(20, 20)
        painter.setOpacity(.2)
        images.append(weakref.ref(device))
        try:
            raise RuntimeError('injected optional raster failure')
        except RuntimeError as error:
            errors.append(error)
            raise

    monkeypatch.setattr(engine, '_paint_cached_fungal_paths', fail)
    image = engine.shade(960, 540)
    assert _bytes(image) == original
    assert engine._owned_fungal_image is None and engine._fungal_raster_failed
    assert not engine._fungal_rasters and not engine._fungal_observed
    assert _bytes(engine.shade(960, 540)) == original
    del image
    gc.collect()
    assert images[0]() is not None
    traceback = errors[0].__traceback__
    while traceback.tb_frame.f_code.co_name != 'fail':
        traceback = traceback.tb_next
    assert int(traceback.tb_frame.f_locals['borrowed'][0]) >> 24 == 255
    del traceback
    errors.clear()
    gc.collect()
    assert images[0]() is None


@pytest.mark.parametrize('variant', ['unowned', 'transform', 'device_ratio', 'opacity',
                                     'clip', 'ARGB32', 'size', 'light', 'failed',
                                     'dimensions', 'composition'])
def test_unsupported_context_keeps_original_qt_paths(variant, qapp, monkeypatch):
    engine = _engine(background='#f6f7f9' if variant == 'light' else '#101418')
    reference = _engine(background='#f6f7f9' if variant == 'light' else '#101418')
    frames = []
    for current in [reference, engine]:
        if variant == 'size':
            current.set_size(1)
        elif variant == 'failed':
            current._fungal_raster_failed = True
        frame = QImage(480, 270, QImage.Format_ARGB32 if variant == 'ARGB32'
                       else QImage.Format_RGB32)
        frame.fill(current.identity)
        if variant == 'device_ratio':
            frame.setDevicePixelRatio(2)
        painter = QPainter(frame)
        painter.setCompositionMode(current.mode)
        current._owned_fungal_image = None if variant == 'unowned' else frame
        if variant == 'transform':
            painter.translate(QPointF(8, 5))
        elif variant == 'opacity':
            painter.setOpacity(.5)
        elif variant == 'clip':
            painter.setClipRect(20, 20, 250, 160)
        elif variant == 'composition':
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        width = 481 if variant == 'dimensions' else 480
        assert not current._can_reuse_fungal_raster(painter, width, 270)
        if current is engine:
            def forbidden(*args):
                raise AssertionError('unsupported painter reached optional raster cache')
            monkeypatch.setattr(current, '_paint_cached_fungal_paths', forbidden)
        current._paint_field(painter, width, 270)
        painter.end()
        current._owned_fungal_image = None
        frames.append(frame)
        assert not current._fungal_rasters
    assert frames[0].constBits().tobytes() == frames[1].constBits().tobytes()


@pytest.mark.parametrize('palette', ['spacr', 'random', 'custom', 'white'])
@pytest.mark.parametrize('density', [2, 3])
def test_native_dense_growth_reuses_exact_pixels_and_preserves_owned_memory_bounds(
        palette, density, qapp, monkeypatch):
    monkeypatch.setattr(preferences, '_ambient_custom_colors',
                        lambda: ('#fffe00017777', '#0303fffefefe'))
    engine = _engine('spacr' if palette == 'white' else palette)
    reference = _engine('spacr' if palette == 'white' else palette)
    for current in (engine, reference):
        current.set_resolution(1)
        current.set_density(density)
        if palette == 'white':
            current.set_colors(['white'])
    reference._fungal_raster_failed = True
    assert engine.effective_density() == density
    hits = []
    reuse = engine._reuse_fungal_raster

    def record(*args):
        result = reuse(*args)
        hits.append(result)
        return result

    monkeypatch.setattr(engine, '_reuse_fungal_raster', record)
    for _ in range(32):
        frame = engine.shade(3840, 2160)
    assert any(hits)
    before = _bytes(frame)
    assert before == _bytes(reference.shade(3840, 2160))
    assert frame.width() == 3840 and frame.height() == 2160
    for clock in (95.125, 98):
        engine.set_time(clock)
        reference.set_time(clock)
        actual = engine.shade(3840, 2160)
        assert _bytes(actual) == _bytes(reference.shade(3840, 2160))
        assert _bytes(frame) == before
    entries = engine._fungal_rasters
    assert len(entries) <= 64 and len(engine._fungal_observed) <= 64
    assert sum(entry[1].nbytes + entry[2].nbytes for entry in entries.values()) <= 8 * 1024**2
    assert all(entry[1].base is None and entry[2].base is None for entry in entries.values())
    assert engine._owned_fungal_image is None
    engine.set_size(1)
    assert not engine._fungal_rasters and not engine._fungal_observed
    assert _bytes(frame) == before


def test_saturation_guard_preserves_destination_and_unsaturated_addition_is_opaque(qapp):
    engine = _engine()
    path = QPainterPath()
    path.moveTo(0, 0)
    path.lineTo(10, 10)
    key = ('sample',)
    color = QColor('white')
    color.setAlphaF(.2)
    engine._fungal_rasters[key] = (path, np.array([0, 1], np.int32),
                                   np.array([0x101010, 0x202020], np.uint32))
    target = np.array([0xffeeeeee, 0xff000000], np.uint32)
    original = target.copy()
    assert not engine._reuse_fungal_raster(key, path, color, target)
    assert np.array_equal(target, original)
    target[:] = [0xff010203, 0xff040506]
    assert engine._reuse_fungal_raster(key, path, color, target)
    assert target.tolist() == [0xff111213, 0xff242526]
    changed = QPainterPath(path)
    changed.lineTo(20, 20)
    assert not engine._reuse_fungal_raster(key, changed, color, target)
    assert not engine._reuse_fungal_raster(('missing',), path, color, target)


def test_sparse_arrays_and_path_history_remain_bounded_and_empty_regions_are_skipped(qapp):
    engine = _engine()
    path = QPainterPath()
    path.moveTo(8, 8)
    path.lineTo(40, 20)
    color = QColor('white')
    color.setAlphaF(.2)
    huge = (QPainterPath(path), np.zeros(2 * 1024**2, np.int32),
            np.zeros(2 * 1024**2, np.uint32))
    engine._fungal_rasters[('oversized',)] = huge
    assert engine._warm_fungal_raster(('sample',), path, color, 1, 64, 32)
    assert ('oversized',) not in engine._fungal_rasters
    for index in range(70):
        engine._fungal_rasters[('seed', index)] = (QPainterPath(path), np.array([0], np.int32),
                                                np.array([1], np.uint32))
        engine._observe_fungal_path(('observation', index), path)
    assert engine._warm_fungal_raster(('new',), path, color, 1, 64, 32)
    assert len(engine._fungal_rasters) <= 64 and len(engine._fungal_observed) <= 64
    assert sum(entry[1].nbytes + entry[2].nbytes
               for entry in engine._fungal_rasters.values()) <= 8 * 1024**2
    offscreen = QPainterPath()
    offscreen.moveTo(-500, -500)
    offscreen.lineTo(-400, -400)
    assert not engine._warm_fungal_raster(('offscreen',), offscreen, color, 1, 64, 32)
