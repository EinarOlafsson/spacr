"""N684: the CPU paths that the graphics renderer mirrors, at their edges.

Blurred stars get a halo pass under every dot pass, spinning paper paints
the same with an explicit layout as with the pending one, and only hairline
filaments get Qt's half-pixel cap extension.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QImage, QPainter

from spacr.qt.widgets import ambient


def test_blurred_stars_draw_a_wider_halo_under_each_dot_pass(qapp):
    engine = ambient.make_engine("drift", ambient.default_palette_for("drift"),
                                 "#191919", seed=4, blur=1.0, blink_percent=0)
    engine.set_time(2.0)
    passes = engine._dot_passes(1280, 720)
    half = len(passes) // 2
    assert len(passes) == 2 * half > 0
    for (halo, halo_points), (dot, dot_points) in zip(passes[:half], passes[half:]):
        assert halo_points is dot_points
        assert halo.widthF() > dot.widthF()
        assert halo.color().alphaF() < dot.color().alphaF()
    sharp = ambient.make_engine("drift", ambient.default_palette_for("drift"),
                                "#191919", seed=4, blur=0.0, blink_percent=0)
    sharp.set_time(2.0)
    assert len(sharp._dot_passes(1280, 720)) == half


def _paper(layout_mode, engine, layout, size=(240, 160)):
    image = QImage(*size, QImage.Format_RGB32)
    image.fill(engine.identity)
    painter = QPainter(image)
    try:
        if layout_mode == "explicit":
            engine._paint_tissue_facets(painter, *size, layout=layout)
        else:
            engine._tissue_pending = layout
            engine._paint_tissue_facets(painter, *size)
    finally:
        painter.end()
    return image


def test_spinning_paper_paints_an_explicit_layout_like_the_pending_one(qapp):
    engine = ambient.make_engine("data_art_tissue_facets", "spacr", "#191919", seed=8)
    engine.pointer = (0.5, 0.5)
    engine.gravity_radius = 0.7
    engine.set_time(1.0)
    engine._tissue_layout(240, 160)
    engine.set_time(1.05)
    layout = engine._tissue_layout(240, 160)
    assert any(angle for *_rest, angle in layout[1])
    explicit = _paper("explicit", engine, layout)
    assert getattr(engine, "_tissue_pending", None) is None
    pending = _paper("pending", engine, layout)
    assert engine._tissue_pending is None
    assert explicit == pending
    blank = QImage(240, 160, QImage.Format_RGB32)
    blank.fill(engine.identity)
    assert explicit != blank


def test_only_hairline_filaments_get_the_half_pixel_cap_extension(qapp):
    engine = ambient.make_engine("data_art_fungal_growth", "spacr", "#191919", seed=684)
    engine.set_time(41.5)
    geometry = engine.geometry(1920, 1080)
    starts = {(np.float32(x0), np.float32(y0)) for x0, y0, *_rest in geometry}
    _groups, segments, _tips, _dark = engine._fungal_strokes(1920, 1080)
    first = segments[np.fmod(segments[:, 5], 2.0) >= 1.0]
    thick = first[first[:, 4] > 0.5]
    thin = first[first[:, 4] <= 0.5]
    assert len(thick) and len(thin)
    assert all((x, y) in starts for x, y in thick[:, 0:2])
    moved = [(x, y) not in starts for x, y in thin[:, 0:2]]
    assert sum(moved) >= 0.9 * len(moved)
    pushes = []
    for row in thin[:50]:
        x, y = row[0], row[1]
        candidates = [(sx, sy) for sx, sy in starts
                      if abs(sx - x) <= 0.51 and abs(sy - y) <= 0.51]
        pushes.append(bool(candidates))
    assert all(pushes)
