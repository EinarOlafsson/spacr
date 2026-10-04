"""Wound-closure fallbacks: each guard that keeps the cut it was given."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage as ndi

from spacr import measure as m
from tests.test_scratch_wound_closure import SHAPE, _brightfield, _straight


def _scattered_series():
    first, later = _straight(192, 80), _straight(192, 50)
    rng = np.random.default_rng(1)
    image = _brightfield(later, seed=11).astype(float)
    texture = ndi.gaussian_filter(rng.standard_normal(SHAPE), 1.5)
    texture /= texture.std()
    yy, xx = np.indices(SHAPE)
    zone = later & (xx < 0.6 * SHAPE[1])
    ys, xs = np.nonzero(zone)
    cells = np.zeros(SHAPE, dtype=bool)
    for i in rng.choice(ys.size, int(zone.sum() / 49), replace=False):
        cells |= (yy - ys[i]) ** 2 + (xx - xs[i]) ** 2 <= 4
    image += cells * texture * 45
    return [_brightfield(first, seed=10), image]


@pytest.mark.parametrize("name, value", [
    ("_WOUND_RELEVEL_MIN_FAR", 2.0),
    ("_WOUND_FLOOR_REACH", -1),
    ("_WOUND_FLOOR_MIN_CORE", 1e6),
    ("_WOUND_FLOOR_MIN_GAP", 1e6),
    ("_WOUND_FRONT_ONLY", False),
    ("_WOUND_SCATTERED_RADIUS", 0.0),
    ("_WOUND_FRONT_MIN_WIDTH", 1e6),
])
def test_each_fallback_still_measures_the_series(monkeypatch, name, value):
    monkeypatch.setattr(m, name, value)
    frame, status, _masks = m._wound_series(_scattered_series(), (0, 1),
                                            source="texture")
    assert status == "ok"
    assert np.isfinite(frame["relative_open_area"].iloc[1])


def test_fronts_with_no_measurable_position_keep_the_wound(monkeypatch):
    wound = _straight(192, 40)
    monkeypatch.setattr(m, "map_coordinates", None, raising=False)
    import scipy.ndimage as nd

    monkeypatch.setattr(nd, "map_coordinates",
                        lambda array, coords, **k: np.zeros(coords[0].shape))
    assert m._wound_fronts(wound, 15) is wound
