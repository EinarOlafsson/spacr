"""Fungal backdrop growth stays sparse, continuous and reproducible."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QImage, QPainter

from spacr.qt.widgets.ambient import _FungalGrowthEngine

PALETTE = ("#6ce7c2", "#b19cff", "#e6c17d")


def _engine(background="#101418", **options):
    """Construct the standalone renderer without a widget or timer."""
    return _FungalGrowthEngine(PALETTE, background, seed=29, **options)


def _frame(engine, width, height):
    """Composite exactly as the widget does onto the page colour."""
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(engine.background)
    painter = QPainter(image)
    engine.paint(painter, width, height)
    painter.end()
    return image


def _occupancy(image, background):
    """Fraction of actual painted output pixels differing from the page."""
    color = QColor(background)
    pixels = np.frombuffer(image.bits(), dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine())
    rgb = pixels[:, :image.width() * 4].reshape(
        image.height(), image.width(), 4)[:, :, :3]
    return float(np.any(rgb != [color.blue(), color.green(), color.red()],
                        axis=2).mean())


def _digest(image):
    """Hash the full raster, including the background and thin tips."""
    return hashlib.sha256(image.bits().tobytes()).hexdigest()


@pytest.mark.parametrize("canvas", ((160, 90), (384, 216), (640, 360)))
def test_fungal_raster_occupies_less_than_twenty_five_percent_at_control_extremes(
        canvas):
    """The limit applies to real light/dark pixels, softness and low detail."""
    width, height = canvas
    for background in ("#101418", "#f0f1ed"):
        for resolution, density, size, blur in (
                (0.25, 0.25, 0.25, 0.0),
                (0.25, 3.0, 2.5, 3.0),
                (1.0, 1.0, 1.0, 0.0),
                (1.0, 3.0, 2.5, 3.0),
                (2.0, 3.0, 2.5, 3.0)):
            engine = _engine(background, resolution=resolution,
                             density=density, size=size, blur=blur)
            engine.set_max_pixels(width * height)
            for second in (0.0, 11.37, 30.01, 93.81, 3600.33):
                engine.set_time(second)
                occupied = _occupancy(_frame(engine, width, height),
                                      background)
                assert occupied <= 0.25, (
                    canvas, background, resolution, density,
                    size, blur, second, occupied)


def test_native_4k_uses_the_physical_pixel_budget_and_remains_sparse():
    """High detail keeps fine lines native without exceeding screen memory."""
    width, height = 3840, 2160
    engine = _engine("#f0f1ed", density=3.0, size=2.5, blur=3.0)
    engine.set_max_pixels(width * height)
    assert engine.buffer_size(width, height) == (width, height)
    for second in (47.5, 3600.33):
        engine.set_time(second)
        image = _frame(engine, width, height)
        assert _occupancy(image, "#f0f1ed") <= 0.25
    assert engine._buffer is None
    owned = engine.shade(width, height)
    assert owned.bytesPerLine() * owned.height() <= width * height * 4
    engine.set_resolution(0.5)
    assert engine.buffer_size(width, height) == (width // 2, height // 2)
    capped = _engine()
    capped.set_max_pixels(1920 * 1080)
    assert capped.buffer_size(width, height) == (1920, 1080)
    assert capped.geometry(0, height) == ()
    assert capped.geometry(width, 0) == ()


def test_fractional_tip_growth_and_cycle_boundaries_are_continuous():
    """A new lineage enters invisibly while older tips keep their paths."""
    engine = _engine()
    width, height = 384, 216
    engine.set_time(4.1)
    before = {edge[:6] + edge[-1:]: edge[6]
              for edge in engine.geometry(width, height)
              if 0.0 < edge[6] < 1.0}
    engine.set_time(4.2)
    after = {edge[:6] + edge[-1:]: edge[6]
             for edge in engine.geometry(width, height)
             if 0.0 < edge[6] < 1.0}
    growing = [after[key] - progress for key, progress in before.items()
               if key in after and after[key] > progress]
    assert growing and max(growing) < 0.1

    engine.set_time(29.999)
    old = _frame(engine, width, height)
    engine.set_time(30.001)
    new = _frame(engine, width, height)
    assert _occupancy(old, "#101418") > 0.0
    assert _occupancy(new, "#101418") > 0.0
    old_bytes = np.frombuffer(old.bits(), dtype=np.uint8)
    new_bytes = np.frombuffer(new.bits(), dtype=np.uint8)
    assert float(np.mean(old_bytes != new_bytes)) < 0.03


def test_arbitrary_hour_later_clock_is_seeded_and_history_stays_bounded():
    """Revisiting a frame recreates identical pixels after cache eviction."""
    first = _engine()
    same = _engine()
    for engine in (first, same):
        engine.set_time(3600.33)
    expected = _digest(_frame(first, 384, 216))
    assert _digest(_frame(same, 384, 216)) == expected
    for second in range(0, 7200, 30):
        first.set_time(float(second))
        assert first.geometry(384, 216)
        assert len(first._lineage_cache) <= 8
    first.set_time(3600.33)
    assert _digest(_frame(first, 384, 216)) == expected
    other = _FungalGrowthEngine(PALETTE, "#101418", seed=30)
    other.set_time(3600.33)
    assert _digest(_frame(other, 384, 216)) != expected


def test_each_colony_has_a_common_origin_and_progressive_connected_forks():
    """Every branch starts at its parent's completed tip and grows from it."""
    engine = _engine()
    width, height = 640, 360
    first = engine._lineage(0, width, height)
    second = engine._lineage(1, width, height)
    assert first[0][1:3] == (engine._origin[0] * width, engine._origin[1] * height)
    assert first[0][1:3] != second[0][1:3]
    for colony in (first, second):
        assert len(colony) == 240 * 6
        for index in range(240):
            branch = colony[index * 6:(index + 1) * 6]
            assert all(edge[0] == index for edge in branch)
            assert all(before[5:7] == after[1:3]
                       for before, after in zip(branch, branch[1:]))
            if index:
                parent = colony[((index - 1) // 2) * 6 + 5]
                assert branch[0][1:3] == parent[5:7]
                assert branch[0][7] > parent[7] + parent[8]
    engine.set_time(3600.33)
    visible = engine.geometry(width, height)
    assert visible
    assert len(engine._lineage_cache) <= 8


def test_live_tips_are_brighter_than_established_branches_and_density_adds_forks():
    engine = _engine()
    engine.set_time(18.0)
    edges = engine.geometry(1920, 1080)
    active = [edge[7] for edge in edges if edge[6] < 1 and edge[6] > .2]
    established = [edge[7] for edge in edges if edge[6] == 1]
    assert active and established
    assert max(active) > min(established) * 1.5
    engine.set_time(80.0)
    counts = []
    for density in (.01, .10, .50):
        engine.set_density(density)
        counts.append(len(engine.geometry(1920, 1080)))
    assert 0 < counts[0] < counts[1] < counts[2]


def test_density_size_speed_and_palette_change_the_actual_frame():
    """Each user control changes shape, abundance, clock or coloured ink."""
    engine = _engine()
    engine.set_time(15.0)
    original = _digest(_frame(engine, 1920, 1080))
    engine.set_density(3.0)
    dense = engine.geometry(1920, 1080)
    assert len(dense) > len(_engine().geometry(1920, 1080))
    assert _digest(_frame(engine, 1920, 1080)) != original
    engine.set_size(2.0)
    assert engine.geometry(1920, 1080) != dense
    engine.set_colors(("#f5a35d", "#f05f8c"))
    recolored = _digest(_frame(engine, 384, 216))
    engine.set_colors(PALETTE)
    assert _digest(_frame(engine, 384, 216)) != recolored
    engine.set_time(0.0)
    engine.set_speed(2.0)
    engine.advance(1.25)
    assert engine.time == 2.5
