"""Item 606: galaxies pass by in the space pattern, on both renderers.

"for the space mode there are small stars that look great passing like
little dots. but there should also be galaxies passing i want the user to
feel like they are really passing through space."

Pinned here: the pictures are drawn once and fade to black at their edge;
a galaxy approaches with perspective -- it grows and speeds up as it nears
-- and fades in and out; galaxies are capped and sparse; the CPU fallback
lays them over its star field; and the shader declares one set of uniforms
per slot, which the GPU canvas fills every frame.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

space = pytest.importorskip("spacr.qt.widgets.fractal_space")


def test_the_pictures_are_drawn_once_and_fade_out_at_their_edge():
    atlas = space._galaxy_atlas()
    size = space._GALAXY_SPRITE_SIZE
    assert atlas.shape == (size, size * space._GALAXY_SPRITES, 3)
    assert space._galaxy_atlas() is atlas, "redrawn on every call"
    assert atlas.min() >= 0.0 and atlas.max() <= 1.0
    sprites = [atlas[:, i * size:(i + 1) * size] for i in
               range(space._GALAXY_SPRITES)]
    for sprite in sprites:
        assert sprite[:, 0].max() < 0.01 and sprite[0, :].max() < 0.01
        assert sprite[size // 2, size // 2].max() > 0.5, "no bright core"
    assert not np.allclose(sprites[0], sprites[1])
    assert {"spiral", "barred", "elliptical"} <= set(space._GALAXY_KINDS)


def _track(slot, start, stop, step=0.5):
    """One slot's row over time, while it carries the same galaxy."""
    rows = []
    for t in np.arange(start, stop, step):
        rows.append((t, space._galaxies_at(float(t), 1.0)[slot]))
    return rows


def _one_life(slot=0):
    """The times one galaxy in ``slot`` is alive, start to finish."""
    pace = 1.4 / (1.4 * space._GALAXY_LIFETIME)
    for epoch in range(2, 40):
        start = (epoch - (slot + 0.37) / space.GALAXY_SLOTS) / pace
        row = space._galaxies_at(start + 0.5 * space._GALAXY_LIFETIME,
                                 1.0)[slot]
        if row[3] > 0.0:
            return start, start + space._GALAXY_LIFETIME
    raise AssertionError("no galaxy ever flew in this slot")


def test_a_galaxy_grows_and_speeds_up_as_it_nears_then_fades():
    start, end = _one_life()
    rows = _track(0, start + 0.5, end - 0.5)
    sizes = np.array([row[2] for _t, row in rows])
    places = np.array([np.hypot(row[0], row[1]) for _t, row in rows])
    depths = np.array([row[11] for _t, row in rows])
    assert (np.diff(depths) < 0).all(), "it is not approaching"
    assert (np.diff(sizes) > 0).all(), "it does not grow as it nears"
    outward = np.diff(places)
    assert (outward > 0).all()
    assert outward[-1] > 5 * outward[0], "it does not speed up as it nears"
    brightness = np.array([row[3] for _t, row in rows])
    assert brightness[0] < 0.1 * brightness.max()
    assert brightness[-1] < 0.1 * brightness.max()


def test_galaxies_are_capped_and_sparse():
    seen = []
    for t in np.arange(0.0, 900.0, 1.0):
        rows = space._galaxies_at(float(t), 1.0)
        assert rows.shape == (space.GALAXY_SLOTS, 12)
        visible = rows[:, 3] > 0.05
        seen.append(int(visible.sum()))
    assert max(seen) <= space.GALAXY_SLOTS
    assert 0.5 < np.mean(seen) < space.GALAXY_SLOTS
    assert space.GALAXY_SLOTS < space.STAR_LAYERS * 10


def test_the_layout_is_a_pure_function_of_the_clock():
    assert np.array_equal(space._galaxies_at(123.4, 1.0),
                          space._galaxies_at(123.4, 1.0))
    assert not np.array_equal(space._galaxies_at(123.4, 1.0),
                              space._galaxies_at(130.0, 1.0))


def _a_close_galaxy():
    for t in np.arange(0.0, 600.0, 0.5):
        rows = space._galaxies_at(float(t), 1.0)
        for row in rows:
            if row[3] > 0.3 and row[2] > 0.3 and abs(row[0]) < 1.0 \
                    and abs(row[1]) < 0.6:
                return float(t), row
    raise AssertionError("no galaxy ever comes close")


def test_the_cpu_fallback_draws_them(monkeypatch):
    pytest.importorskip("numba")
    t, row = _a_close_galaxy()
    engine = space.SpaceEngine(2)
    engine.samples = 1
    with_them = engine.render(160, 90, t, 1.0)
    monkeypatch.setattr(space, "_galaxies_at",
                        lambda *_a: np.zeros((space.GALAXY_SLOTS, 12)))
    without = engine.render(160, 90, t, 1.0)
    added = with_them.astype(int) - without.astype(int)
    assert added.min() >= 0, "a galaxy darkened the sky"
    assert (added.max(axis=2) > 8).sum() > 60, "no galaxy was drawn"
    assert with_them.mean() < 40, "the sky is no longer mostly black"


def test_the_shader_has_one_set_of_uniforms_per_slot():
    source = space.FRAGMENT_SHADER
    assert "uniform sampler2D u_galaxies;" in source
    assert f"/ {float(space._GALAXY_SPRITES)}," in source
    declared = set(re.findall(r"uniform\s+\w+\s+(u_galaxy\d_\w+)\s*;",
                              source))
    filled = set(space._galaxy_uniforms(space._galaxies_at(50.0, 1.0)))
    assert declared == filled
    assert len(declared) == 3 * space.GALAXY_SLOTS
    for slot in range(space.GALAXY_SLOTS):
        assert f"galaxy_light(p, u_galaxy{slot}_place" in source


def test_the_texture_holds_the_square_root_of_the_light():
    texture = space._galaxy_texture()
    atlas = space._galaxy_atlas()
    assert texture.dtype == np.uint8 and texture.shape[2] == 4
    probe = atlas[64, 64 + 3 * space._GALAXY_SPRITE_SIZE]
    stored = texture[64, 64 + 3 * space._GALAXY_SPRITE_SIZE, :3] / 255.0
    assert np.allclose(stored ** 2, probe, atol=0.01)


def test_the_gpu_canvas_uploads_them_and_moves_them_every_frame(
        gpu_backdrop, stand_in_vispy):  # noqa: F811
    canvas = gpu_backdrop("space")._canvas
    assert "u_galaxies" in canvas._program
    uploaded = canvas._program["u_galaxies"]
    assert uploaded.data.shape[2] == 4
    names = [f"u_galaxy{slot}_place" for slot in range(space.GALAXY_SLOTS)]
    canvas._update_uniforms(200.0)
    first = [canvas._program[name] for name in names]
    canvas._update_uniforms(230.0)
    assert [canvas._program[name] for name in names] != first


from tests.qt.test_cov_r5_fractal_travel import (  # noqa: E402,F401
    gpu_backdrop,
    stand_in_vispy,
)
