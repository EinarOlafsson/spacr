"""Stable individual colours have exact compiled/fallback grain ownership."""

import gc
import sys
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtGui import QImage, QPainter

from spacr.qt.widgets import ambient


@pytest.fixture
def numpy_colors(monkeypatch):
    monkeypatch.setattr(ambient, '_COLORED_SCATTER', None)
    monkeypatch.setattr(ambient, '_COLORED_SCATTER_FAILED', False)
    monkeypatch.setattr(ambient, '_PACKED_SCATTER_FAILED', True)
    monkeypatch.setattr(ambient, '_SATIN_COMPILER', SimpleNamespace(ready=lambda: None))


@pytest.fixture(scope='module')
def colored_kernel():
    from numba import njit

    kernel = njit(nogil=True, cache=False)(ambient._scatter_colored_grains)
    coordinates = np.zeros(1, np.int32)
    table = np.zeros(256, np.uint64)
    kernel(np.zeros(1, np.uint32), np.zeros(1, np.uint8), coordinates,
           coordinates, np.zeros(1, np.uint16), table, table, table, 1, 1, True, True)
    assert kernel.nopython_signatures
    return kernel


def frame(engine, width=960, height=540):
    if hasattr(engine, 'shade'):
        return engine.shade(width, height)
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(engine.identity)
    painter = QPainter(image)
    painter.setCompositionMode(engine.mode)
    try:
        engine.paint(painter, width, height)
    finally:
        painter.end()
    return image


def test_advection_preserves_particle_identity_across_trail_samples(
        qapp, numpy_colors, monkeypatch):
    engine = ambient.make_engine('data_art_genetic_advection', 'random',
                                 '#101418', seed=42, resolution=2)
    observed = []
    material = engine._colored_point_material

    def capture(width, height, x, y, light, spread):
        observed.append((x.shape, y.shape, light.shape))
        return material(width, height, x, y, light, spread)

    monkeypatch.setattr(engine, '_colored_point_material', capture)
    image = frame(engine)
    assert not image.isNull()
    assert len(observed) == 1
    x, y, light = observed[0]
    assert len(x) == 2 and x[0] == 12 and x[1] > 100
    assert x == y == light


@pytest.mark.parametrize('theme', ambient.AMBIENT_THEMES)
@pytest.mark.parametrize('background', ['#101418', '#f6f7f9'])
def test_random_palette_is_offered_seeded_and_restores_previous_style(
        theme, background, qapp, numpy_colors):
    assert 'random' in ambient.palettes_for(theme)
    assert ambient.palette_label(theme, 'random')
    engine = ambient.make_engine(theme, 'spacr', background, seed=42, resolution=2)
    engine.set_max_pixels(960 * 540)
    engine.set_time(47)
    image = frame(engine)
    original = image.bits().tobytes()
    palette = ambient.palette_colors(theme, 'random')
    engine.set_colors(palette)
    assert engine._random_palette
    image = frame(engine)
    colored = image.bits().tobytes()
    assert colored != original
    same = ambient.make_engine(theme, 'random', background, seed=42, resolution=2)
    same.set_max_pixels(960 * 540)
    same.set_time(47)
    matching = frame(same)
    assert matching.bits().tobytes() == colored
    engine.set_time(3600.33)
    later = frame(engine)
    assert image.bits().tobytes() == colored
    assert not later.isNull()
    engine.set_time(47)
    engine.set_colors(ambient.palette_colors(theme, 'spacr'))
    image = frame(engine)
    assert image.bits().tobytes() == original


@pytest.mark.parametrize('background', ['#101418', '#f6f7f9'])
@pytest.mark.parametrize('family', ['point_atlas', 'impulse_lens', 'genetic_advection'])
def test_compiled_multi_hue_matches_complete_fallback_frames(
        family, background, qapp, numpy_colors, colored_kernel, monkeypatch):
    engine = ambient.make_engine('data_art_' + family, 'random', background,
                                 seed=42, resolution=2)
    engine.set_max_pixels(1920 * 1080)
    engine.set_time(17)
    image = frame(engine, 1920, 1080)
    reference = image.bits().tobytes()
    monkeypatch.setattr(ambient, '_COLORED_SCATTER', colored_kernel)
    compiled = frame(engine, 1920, 1080)
    assert compiled.bits().tobytes() == reference
    cached = list(engine._material_cache.values())
    assert not any(isinstance(value, np.ndarray) and value.size == 1920 * 1080
                   for value in cached)


@pytest.mark.parametrize('background', ['#101418', '#f6f7f9'])
def test_duplicate_grains_keep_strongest_intensity_across_hues(
        background, qapp, numpy_colors, colored_kernel, monkeypatch):
    engine = ambient.make_engine('data_art_impulse_lens', 'random', background, seed=42)
    x = np.array([0, 0, 15, 15, -1, 16], np.float32)
    y = np.array([0, 0, 8, 8, 2, 2], np.float32)
    light = np.array([.125, .875, .75, .25, 1, 1], np.float32)
    image = engine._point_material(16, 9, x, y, light, spread=True)
    reference = image.bits().tobytes()
    table = engine._material_cache[('random_grain_palette', engine.dark)][0]
    identities = np.arange(x.size, dtype=np.uint32)
    mixed = identities * np.uint32(0x9e3779b1) + np.uint32(engine._art_seed & 0xffffffff)
    mixed ^= mixed >> 16
    hues = mixed % 32
    for point in [1, 2]:
        slot = int(hues[point]) * 256 + round(float(light[point]) * 255)
        assert image.pixel(int(x[point]), int(y[point])) == int(table[slot] & np.uint64(0xffffffff))
    monkeypatch.setattr(ambient, '_COLORED_SCATTER', colored_kernel)
    compiled = engine._point_material(16, 9, x, y, light, spread=True)
    assert compiled.bits().tobytes() == reference


def test_particle_trail_samples_keep_same_hue_as_identity_moves(qapp, numpy_colors):
    engine = ambient.make_engine('data_art_genetic_advection', 'random', '#101418', seed=42)
    x = np.array([[2, 10], [4, 12]], np.float32)
    y = np.array([[2, 2], [5, 5]], np.float32)
    first = engine._point_material(20, 12, x, y, np.ones_like(x), spread=False)
    assert first.pixel(2, 2) == first.pixel(4, 5)
    assert first.pixel(10, 2) == first.pixel(12, 5)
    assert first.pixel(2, 2) != first.pixel(10, 2)
    second = engine._point_material(20, 12, x + 1, y + 1, np.ones_like(x), spread=False)
    assert first.pixel(2, 2) == second.pixel(3, 3)
    assert first.pixel(10, 2) == second.pixel(11, 3)


@pytest.mark.parametrize('background', ['#101418', '#f6f7f9'])
@pytest.mark.parametrize('spread', [False, True])
def test_python_ranked_kernel_matches_duplicate_and_perimeter_fallback(
        background, spread, qapp, numpy_colors, monkeypatch):
    engine = ambient.make_engine('data_art_impulse_lens', 'random', background, seed=42)
    rng = np.random.default_rng(42)
    x = rng.integers(0, 8, size=400).astype(np.float32)
    y = rng.integers(0, 5, size=400).astype(np.float32)
    light = rng.choice(np.array([0, .25, .5, .75, 1], np.float32), size=400)
    reference = engine._point_material(8, 5, x, y, light, spread=spread)
    monkeypatch.setattr(ambient, '_COLORED_SCATTER', ambient._scatter_colored_grains)
    actual = engine._point_material(8, 5, x, y, light, spread=spread)
    assert actual.bits().tobytes() == reference.bits().tobytes()


def test_partial_compiler_failure_restores_whole_frame_and_disables_fast_path(
        qapp, numpy_colors, monkeypatch):
    engine = ambient.make_engine('data_art_impulse_lens', 'random', '#101418', seed=42)
    image = frame(engine)
    reference = image.bits().tobytes()
    calls = []

    def fail(flat, *_args):
        calls.append(True)
        flat[:100] = 0xffff0000
        raise RuntimeError('injected partial colored grain failure')

    monkeypatch.setattr(ambient, '_COLORED_SCATTER', fail)
    recovered = frame(engine)
    assert recovered.bits().tobytes() == reference
    assert ambient._COLORED_SCATTER_FAILED
    assert ambient._COLORED_SCATTER is None
    repeated = frame(engine)
    assert repeated.bits().tobytes() == reference
    assert len(calls) == 1


def test_rank_plane_is_frame_local_and_palette_cache_is_bounded(
        qapp, numpy_colors, monkeypatch):
    engine = ambient.make_engine('data_art_impulse_lens', 'random', '#101418', seed=42)
    references = []

    def capture(flat, ranks, *arguments):
        references.append(weakref.ref(ranks))
        ambient._scatter_colored_grains(flat, ranks, *arguments)

    monkeypatch.setattr(ambient, '_COLORED_SCATTER', capture)
    image = frame(engine, 320, 180)
    assert not image.isNull()
    gc.collect()
    assert references and all(reference() is None for reference in references)
    tables = engine._material_cache[('random_grain_palette', engine.dark)]
    assert sum(table.nbytes for table in tables) == 32 * 256 * 8 * 3
    assert all(table.ndim == 1 for table in tables)


@pytest.mark.parametrize('failure', ['raise', 'disabled'])
def test_optional_colored_compiler_failure_keeps_existing_single_hue_kernel(
        failure, numpy_colors, monkeypatch):
    original_kernel = SimpleNamespace(nopython_signatures=(True,))

    def original(*_args):
        pass

    original.nopython_signatures = original_kernel.nopython_signatures

    def decorator(*_args, **_kwargs):
        def compile(function):
            if function is ambient._scatter_packed_grains:
                return original
            if failure == 'raise':
                raise RuntimeError('ranked compiler unavailable')
            return function
        return compile

    monkeypatch.setitem(sys.modules, 'numba', SimpleNamespace(njit=decorator))
    monkeypatch.setattr(ambient, '_PACKED_SCATTER', None)
    monkeypatch.setattr(ambient, '_PACKED_SCATTER_FAILED', False)
    ambient._complete_ambient_startup()
    ambient._warm_packed_scatter()
    assert ambient._PACKED_SCATTER is original
    assert not ambient._PACKED_SCATTER_FAILED
    assert ambient._COLORED_SCATTER_FAILED
    assert ambient._ready_colored_scatter() is None
    assert ambient._ready_packed_scatter() is original
