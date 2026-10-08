"""Fungal backdrop growth stays sparse, continuous and reproducible."""

from __future__ import annotations

import hashlib
import math

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
    growing = [((after[key] - progress)
                * (math.hypot(key[2] - key[0], key[3] - key[1])
                   + math.hypot(key[4] - key[2], key[5] - key[3])))
               for key, progress in before.items()
               if key in after and after[key] > progress]
    assert growing and max(growing) < 3.0

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


def test_wandering_tips_fork_recursively_more_often_near_the_front():
    """Three nearby roots explore and their offspring fork in turn."""
    engine = _engine()
    width, height = 960, 540
    for block in range(4):
        colony = engine._lineage(block, width, height)
        endpoints = {edge[5:7] for edge in colony}
        roots = [edge for edge in colony if edge[1:3] not in endpoints]
        assert len(roots) == 3
        assert len({edge[1:3] for edge in roots}) == 1
        assert all(edge[1:3] in endpoints or edge in roots for edge in colony)
        assert max(edge[10] for edge in colony) >= 3
        starts = {}
        for edge in colony:
            starts[edge[1:3]] = starts.get(edge[1:3], 0) + 1
        forks = [edge for edge in colony
                 if starts.get(edge[5:7], 0) > 1
                 and block * 30 <= edge[7] < block * 30 + 30]
        thirds = [sum(block * 30 + start <= edge[7] < block * 30 + start + 10
                      for edge in forks) for start in (0, 10, 20)]
        assert thirds[0] < thirds[1] < thirds[2]
        early = [edge for edge in colony if block * 30 <= edge[7] < block * 30 + 5]
        late = [edge for edge in colony if block * 30 + 18 <= edge[7] < block * 30 + 25]
        primary = [edge for edge in colony if edge[0] == -1]
        if block < 2:
            shift = (sum(edge[5] for edge in late) / len(late)
                     - sum(edge[1] for edge in early) / len(early))
            assert (shift if block == 0 else -shift) > width * 0.4
            assert (max(edge[5] for edge in primary) if block == 0 else
                    width - min(edge[5] for edge in primary)) > width * 0.8
        else:
            shift = (sum(edge[6] for edge in late) / len(late)
                     - sum(edge[2] for edge in early) / len(early))
            assert (shift if block == 2 else -shift) > height * 0.4
            assert (max(edge[6] for edge in primary) if block == 2 else
                    height - min(edge[6] for edge in primary)) > height * 0.8
    engine.set_time(3600.33)
    visible = engine.geometry(width, height)
    assert visible
    assert len(engine._lineage_cache) <= 8


def test_first_front_actually_advances_across_the_rendered_screen():
    """Later frames paint new distant growth rather than recolouring old ink."""
    engine = _engine(density=2.0, size=1.25)
    farthest = []
    for second in (7.0, 21.0):
        engine.set_time(second)
        image = engine.shade(640, 360)
        pixels = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(360, 640, 4)
        _, columns = np.nonzero(np.any(pixels[:, :, :3] != 0, axis=2))
        farthest.append(int(np.quantile(columns, 0.99)))
    assert farthest[1] > farthest[0] + 200


def test_child_filaments_wait_for_their_parent_to_reach_the_fork():
    """A daughter never appears disconnected ahead of its growing parent."""
    engine = _engine()
    for block in range(4):
        lineage = engine._lineage(block, 960, 540)
        arrival = {edge[5:7]: edge[7] + edge[8] for edge in lineage}
        for edge in lineage:
            if edge[1:3] in arrival:
                assert edge[7] >= arrival[edge[1:3]] - 1e-9


def test_empty_colony_can_later_publish_a_connected_front(monkeypatch):
    """An empty frame must not leave stale ancestry when growth resumes."""
    engine = _engine(density=3.0)
    engine.set_time(10.0)
    lineage = engine._lineage
    monkeypatch.setattr(engine, "_lineage", lambda *_: ())
    assert engine.geometry(640, 360) == ()
    monkeypatch.setattr(engine, "_lineage", lineage)
    edges = engine.geometry(640, 360)
    assert edges
    colony = lineage(0, 640, 360)
    endpoints = {edge[5:7] for edge in colony}
    roots = {edge[1:3] for edge in colony if edge[1:3] not in endpoints}
    visible_ends = {edge[4:6] for edge in edges}
    assert all(edge[:2] in roots or edge[:2] in visible_ends for edge in edges)


@pytest.mark.parametrize("cheap_parent_last", (False, True))
def test_colliding_endpoints_resolve_to_the_last_parent_before_budgeting(
        monkeypatch, cheap_parent_last):
    """The admitted daughter includes the selected parent within the ink budget."""
    engine = _engine(density=3.0)
    costly = (0, 0, 0, 20, 20, 5, 5, 0, 1, 0, 0)
    cheap = (1, 4, 4, 4.5, 4.5, 5, 5, 0, 1, 1, 0)
    daughter = (2, 5, 5, 5.5, 5.5, 6, 6, 0, 1, 2, 0)
    parents = (costly, cheap) if cheap_parent_last else (cheap, costly)
    monkeypatch.setattr(engine, "_lineage", lambda *_: (*parents, daughter))
    engine.set_time(10.0)
    admitted = [edge[-1] for edge in engine.geometry(20, 20)]
    assert admitted == ([1, 2] if cheap_parent_last else [1])


def test_selected_expensive_parent_still_admits_its_unselected_sibling(monkeypatch):
    """An admitted ancestor must not hide a fitting sister from frontier order."""
    engine = _engine(density=3.0)
    lineage = (
        (0, 0, 0, .25, 0, .5, 0, 0, 1, 0, 0),
        (1, 0, 10, 1.5, 10, 3, 10, 0, 1, 1, 0),
        (2, 0, 0, 6, 0, 12, 0, 0, 1, 2, 0),
        (3, 12, 0, 12.05, 0, 12.1, 0, 0, 1, 3, 0),
    )
    monkeypatch.setattr(engine, "_lineage", lambda *_: lineage)
    engine.set_time(10.0)
    admitted = engine.geometry(20, 20)
    assert [edge[-1] for edge in admitted] == [0, 2, 3]
    assert admitted[-1][:2] == admitted[-2][4:6]


@pytest.mark.parametrize("ulp_offset", (-8, 0, 8))
def test_remaining_ink_budget_keeps_the_branch_on_the_fitting_side_of_equality(
        monkeypatch, ulp_offset):
    """A tiny fitting filament survives next to an almost budget-filling root."""
    engine = _engine(density=3.0)
    radius = .65 * .5 + 1.5
    tiny_cost = 2.0 * radius * .01 + math.pi * radius ** 2
    length = (88.0 - tiny_cost - math.pi * radius ** 2) / (2.0 * radius)
    for _ in range(abs(ulp_offset)):
        length = math.nextafter(length, math.inf if ulp_offset > 0 else -math.inf)
    lineage = (
        (0, 0, 0, .005, 0, .01, 0, 0, 1, 0, 0),
        (1, 0, 10, length / 2.0, 10, length, 10, 0, 1, 1, 0),
    )
    monkeypatch.setattr(engine, "_lineage", lambda *_: lineage)
    engine.set_time(10.0)
    admitted = [edge[-1] for edge in engine.geometry(20, 20)]
    assert admitted == ([0, 1] if ulp_offset <= 0 else [1])


@pytest.mark.parametrize("canvas", ((20, 1000), (1000, 20)))
def test_narrow_canvases_keep_reflected_filaments_connected_and_inside(canvas):
    """Tips can turn at either nearby wall without leaving a narrow view."""
    width, height = canvas
    engine = _engine(density=0.1)
    for block in range(4):
        lineage = engine._lineage(block, width, height)
        endpoints = {edge[5:7] for edge in lineage}
        for edge in lineage:
            x0, y0, cx, cy, x1, y1 = edge[1:7]
            assert all(math.isfinite(value) for value in (x0, y0, cx, cy, x1, y1))
            assert 0 <= x0 <= width and 0 <= x1 <= width
            assert 0 <= y0 <= height and 0 <= y1 <= height
            assert edge[1:3] in endpoints or edge[0] == -1
        if block == (2 if width < height else 0):
            assert len(lineage) > 100


def test_tiny_canvas_cannot_show_a_disconnected_subpixel_branch():
    """Below one pixel of travel, the renderer leaves the page clean."""
    engine = _engine(density=3.0)
    for block in range(4):
        assert engine._lineage(block, 12, 12) == ()
    for second in (0.0, 45.0):
        engine.set_time(second)
        assert engine.geometry(12, 12) == ()
        assert _occupancy(_frame(engine, 12, 12), "#101418") == 0.0


def test_compact_canvas_retains_connected_visible_growth_as_tips_get_subpixel():
    """Tiny daughter steps fade out while their visible parent paths remain."""
    engine = _engine(density=3.0, size=2.5)
    lineage = engine._lineage(0, 30, 30)
    endpoints = {edge[5:7] for edge in lineage}
    roots = {edge[1:3] for edge in lineage if edge[1:3] not in endpoints}
    assert len(lineage) > 100
    assert len(roots) == 1
    assert all(edge[1:3] in endpoints or edge[1:3] in roots
               for edge in lineage)
    engine.set_time(5.0)
    assert engine.geometry(30, 30)
    assert 0.0 < _occupancy(_frame(engine, 30, 30), "#101418") <= 0.25


def test_visible_front_has_more_connected_forks_later_and_at_higher_density():
    """Density adds live tips while visible daughter paths retain their parents."""
    counts = []
    for density in (1.0, 3.0):
        engine = _engine(density=density)
        engine.set_time(31.0)
        lineage = engine._lineage(0, 960, 540)
        by_path = {edge[1:7]: edge for edge in lineage}
        visible = {edge[:6] for edge in engine.geometry(960, 540)
                   if edge[:6] in by_path}
        endpoints = {edge[5:7] for edge in lineage}
        origins = {edge[1:3] for edge in lineage if edge[1:3] not in endpoints}
        parent_ends = {path[4:6] for path in visible}
        assert all(path[:2] in parent_ends or path[:2] in origins for path in visible)
        children = {}
        for path in visible:
            children[path[:2]] = children.get(path[:2], 0) + 1
        forks = [by_path[path] for path in visible if path in by_path
                 and children.get(path[4:6], 0) > 1]
        assert forks
        counts.append(len(forks))
    assert counts[0] < counts[1]


def test_default_density_starts_at_one_point_and_keeps_forking_in_short_segments():
    engine = _engine(density=.1)
    lineage = engine._lineage(0, 960, 540)
    endpoints = {edge[5:7] for edge in lineage}
    roots = {edge[1:3] for edge in lineage if edge[1:3] not in endpoints}
    assert len(roots) == 1
    starts = {}
    for edge in lineage:
        starts[edge[1:3]] = starts.get(edge[1:3], 0) + 1
        length = math.hypot(edge[5] - edge[1], edge[6] - edge[2])
        assert length < 960 * .013
        middle = ((edge[1] + edge[5]) / 2, (edge[2] + edge[6]) / 2)
        assert math.hypot(edge[3] - middle[0], edge[4] - middle[1]) <= length * .016
    for start in (0, 10, 20):
        forks = [edge for edge in lineage if start <= edge[7] < start + 10
                 and starts.get(edge[5:7], 0) > 1]
        assert len(forks) >= 20


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


def test_density_changes_fork_population_at_both_detail_settings():
    """More detail changes pixel sampling, not whether Density adds branches."""
    populations = []
    for resolution in (1.0, 2.0):
        engine = _engine(resolution=resolution)
        engine.set_time(80.0)
        counts = []
        for density in (1.0, 2.0, 3.0):
            engine.set_density(density)
            counts.append(len(engine.geometry(1920, 1080)))
        assert counts[0] < counts[1] < counts[2]
        populations.append(counts)
    assert populations[0][0] == populations[1][0]


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
