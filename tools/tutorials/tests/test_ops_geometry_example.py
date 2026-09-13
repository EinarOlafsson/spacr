"""Check geometry guards with fixtures; the tutorial uses actual NAS pixels."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ops_geometry_example as example


def test_overlap_uses_the_correct_coordinates_and_rejects_flat_or_tiny_overlap():
    canvas = np.random.default_rng(13).normal(size=(144, 144))
    first, second = canvas[:96, :96], canvas[16:112, 8:104]
    assert example.overlap_correlation(first, second, (16, 8)) == pytest.approx(1)
    assert example.overlap_correlation(first, second, (33, 8)) < .1
    with pytest.raises(ValueError, match='32 pixels'):
        example.overlap_correlation(first, second, (80, 0))
    with pytest.raises(ValueError, match='nonconstant'):
        example.overlap_correlation(first, np.zeros_like(second), (16, 8))


@pytest.fixture
def valid_geometry(monkeypatch):
    edges = {(0, 1): (1267, 9), (0, 11): (-9, 1268),
             (1, 10): (-9, 1267), (11, 10): (1267, 9)}
    well = SimpleNamespace(placements=dict.fromkeys(example.SITES, (0, 0)),
                           proposed=4, accepted=4, canvas=(2756, 2756),
                           residuals=np.array([.25] * 4),
                           edges={pair: SimpleNamespace(dy=dy, dx=dx, shift=(dy, dx))
                                  for pair, (dy, dx) in edges.items()})
    planes = dict.fromkeys(example.SITES)
    monkeypatch.setattr(example, 'overlap_correlation',
                        lambda a, b, shift: .85 if tuple(shift) in edges.values() else .15)
    assert len(example.require_geometry(well, planes)) == 4  # Positive counterpart first.
    return well, planes


@pytest.mark.parametrize('field,value,message', [
    ('accepted', 3, 'adjacencies'),
    ('placements', {0: (0, 0)}, 'adjacencies'),
    ('canvas', (3500, 3500), 'canvas'),
    ('residuals', np.array([3.] * 4), 'loop'),
])
def test_geometry_requires_every_independent_check(valid_geometry, field, value, message):
    well, planes = valid_geometry
    broken = deepcopy(well)
    setattr(broken, field, value)
    with pytest.raises(ValueError, match=message):
        example.require_geometry(broken, planes)


def test_good_loop_statistics_cannot_replace_actual_pixel_support(valid_geometry, monkeypatch):
    well, planes = valid_geometry
    monkeypatch.setattr(example, 'overlap_correlation', lambda *args: .15)
    with pytest.raises(ValueError, match='independent pixel support'):
        example.require_geometry(well, planes)
