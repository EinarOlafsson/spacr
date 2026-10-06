"""Low density percentages change actual offered theme rasters."""

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtGui import QImage, QPainter

from spacr.qt.widgets import ambient

OFFERED = ('blobs', 'aurora', 'drift', 'data_art_point_atlas',
           'data_art_tissue_facets', 'data_art_chromatin_ribbon',
           'data_art_genetic_advection', 'data_art_impulse_lens',
           'data_art_fungal_growth', 'data_art_thore')


@pytest.mark.parametrize('theme', OFFERED)
def test_one_ten_fifty_percent_change_actual_theme_rasters(theme, qapp, monkeypatch):
    monkeypatch.setattr(ambient, '_SATIN_COMPILER',
                        SimpleNamespace(ready=lambda: None))
    engine = ambient.make_engine(theme, 'spacr', '#101418', seed=42,
                                 resolution=2)
    engine.set_max_pixels(960 * 540)
    engine.time = 97
    rasters = []
    for density in (.01, .10, .50):
        engine.set_density(density)
        if isinstance(engine, ambient.DriftEngine):
            image = QImage(960, 540, QImage.Format_RGB32)
            image.fill(engine.identity)
            painter = QPainter(image)
            painter.setCompositionMode(engine.mode)
            try:
                engine.paint(painter, 960, 540)
            finally:
                painter.end()
        else:
            image = engine.shade(960, 540)
        rasters.append(image.bits().tobytes())
    assert len(set(rasters)) == 3


@pytest.mark.parametrize('theme,base', [('blobs', 14), ('aurora', 3)])
def test_fractional_single_element_remains_quieter_than_ten_percent(theme, base, qapp):
    engine = ambient.make_engine(theme, 'spacr', '#101418', seed=42)
    light = []
    counts = []
    for density in (.01, .10):
        engine.set_density(density)
        counts.append(engine.element_count(base, base * 3))
        image = engine.shade(960, 540)
        pixels = np.frombuffer(image.bits(), np.uint8).reshape(-1, 4)
        light.append(float(pixels[:, :3].sum()))
    assert counts == [1, 1]
    assert 0 < light[0] < light[1] * .3


def test_fractional_opacity_leaves_default_and_higher_density_unchanged():
    engine = ambient.make_engine('aurora', 'spacr', '#101418', seed=42)
    for density in (.5, 1, 2, 3):
        engine.set_density(density)
        assert engine._fractional_alpha_scale(3) == engine.alpha_scale()


def test_aurora_density_roundtrip_rebuilds_alpha_dependent_cached_textures(qapp):
    engine = ambient.make_engine('aurora', 'spacr', '#101418', seed=42, density=.01)
    image = engine.shade(960, 540)
    initial = image.bits().tobytes()
    engine.set_density(.10)
    image = engine.shade(960, 540)
    assert image.bits().tobytes() != initial
    engine.set_density(.01)
    image = engine.shade(960, 540)
    assert image.bits().tobytes() == initial


def test_rain_and_facets_have_distinct_low_density_populations(qapp):
    rain = ambient.make_engine('data_art_thore', 'spacr', '#101418', seed=42)
    paper = ambient.make_engine('data_art_tissue_facets', 'spacr', '#101418', seed=42)
    rain_counts = []
    tile_counts = []
    for density in (.01, .10, .50):
        rain.set_density(density)
        rain_counts.append(len(rain.geometry(1920, 1080)[0]))
        paper.set_density(density)
        image = paper.shade(1920, 1080)
        assert not image.isNull()
        material = next(value for key, value in paper._material_cache.items()
                        if key[0] == 'tissue_facets')
        tile_counts.append(len(material))
    assert rain_counts == [1, 10, 52]
    assert tile_counts[0] < tile_counts[1] < tile_counts[2]


def test_satin_fractional_opacity_is_restored_after_draw_failure(qapp, monkeypatch):
    monkeypatch.setattr(ambient, '_SATIN_COMPILER',
                        SimpleNamespace(ready=lambda: None))
    engine = ambient.make_engine('data_art_chromatin_ribbon', 'spacr', '#101418',
                                 seed=42, density=.01)

    class FailingPainter:
        def __init__(self):
            self.current_opacity = .6
            self.draw_opacity = None

        def opacity(self):
            return self.current_opacity

        def setOpacity(self, value):
            self.current_opacity = value

        def drawImage(self, *_args):
            self.draw_opacity = self.current_opacity
            raise RuntimeError('injected draw failure')

    painter = FailingPainter()
    with pytest.raises(RuntimeError, match='injected draw failure'):
        engine._paint_chromatin_ribbon(painter, 320, 180)
    assert painter.draw_opacity == pytest.approx(.6 * .07)
    assert painter.opacity() == .6


def test_one_percent_paper_radius_affects_a_seeded_primitive_center(qapp):
    engine = ambient.make_engine('data_art_tissue_facets', 'spacr', '#101418',
                                 seed=42, resolution=2)
    width, height = 1920, 1080
    engine.set_max_pixels(width * height)
    engine.set_gravity_radius(.01)
    image = engine.shade(width, height)
    resting = image.bits().tobytes()
    cells = next(value for key, value in engine._material_cache.items()
                 if key[0] == 'tissue_facets')
    selected = min(cells, key=lambda cell: (cell[0] - width * .5) ** 2
                   + (cell[1] - height * .5) ** 2)
    x, y = selected[:2]
    reach = min(width, height) * .01
    affected_centers = [(cx, cy) for cx, cy, *_ in cells
                        if (cx - x) ** 2 + (cy - y) ** 2 < reach ** 2]
    assert affected_centers == [(x, y)]
    engine.set_pointer((x / width, y / height))
    image = engine.shade(width, height)
    lifted = image.bits().tobytes()
    assert lifted != resting
    engine.set_pointer(None)
    image = engine.shade(width, height)
    assert image.bits().tobytes() == resting
