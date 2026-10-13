"""The galaxy compositing kernel, run as Python so coverage can see it.

The kernel is a Numba function; its ``py_func`` is the same body, so these
tests read what it does pixel by pixel rather than through the JIT.
"""
from __future__ import annotations

import numpy as np
import pytest

space = pytest.importorskip("spacr.qt.widgets.fractal_space")
pytest.importorskip("numba")


@pytest.fixture
def kernel(monkeypatch):
    monkeypatch.setattr(space, "_atlas_texel", space._atlas_texel.py_func)
    return space._add_galaxies.py_func


def _galaxy(x=0.0, y=0.0, half=0.2, brightness=1.0, squash=1.0):
    row = np.zeros(12)
    row[:9] = [x, y, half, brightness, 1.0, 0.0, squash, 0.0, 1.0]
    row[9:11] = [1.0, 1.0]
    return row


def test_a_galaxy_in_view_adds_light_and_never_removes_it(kernel):
    atlas = space._galaxy_atlas()
    frame = np.full((24, 32, 3), 10, np.uint8)
    before = frame.copy()
    kernel(frame, 0.0, 0.0, 0.0, np.array([_galaxy()]), atlas)
    added = frame.astype(int) - before
    assert added.min() >= 0
    assert added.max() > 0
    assert frame.max() <= 255


def test_dark_empty_and_out_of_view_galaxies_draw_nothing(kernel):
    atlas = space._galaxy_atlas()
    frame = np.zeros((24, 32, 3), np.uint8)
    galaxies = np.array([_galaxy(brightness=0.0), _galaxy(half=0.0),
                         _galaxy(x=50.0, y=50.0)])
    kernel(frame, 0.0, 0.0, 0.0, galaxies, atlas)
    assert frame.max() == 0


def test_a_flat_galaxy_skips_the_pixels_outside_its_disc(kernel):
    atlas = space._galaxy_atlas()
    flat = np.zeros((24, 32, 3), np.uint8)
    round_ = np.zeros_like(flat)
    kernel(flat, 0.0, 0.0, 0.0, np.array([_galaxy(squash=0.05)]), atlas)
    kernel(round_, 0.0, 0.0, 0.0, np.array([_galaxy(squash=1.0)]), atlas)
    assert (flat.max(axis=2) > 0).sum() < (round_.max(axis=2) > 0).sum()


def test_the_texel_read_clamps_to_the_atlas_edge():
    texel = space._atlas_texel.py_func
    atlas = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    assert texel(atlas, -5.0, -5.0, 0) == atlas[0, 0, 0]
    assert texel(atlas, 99.0, 99.0, 1) == atlas[1, 2, 1]
    middle = texel(atlas, 0.5, 0.0, 2)
    assert atlas[0, 0, 2] < middle < atlas[0, 1, 2]
