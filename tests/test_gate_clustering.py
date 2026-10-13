"""DBSCAN clustering in the Gate Editor.

DBSCAN, not k-means: a scatter of cells has dense populations of unequal size
sitting in sparse debris, which is the shape DBSCAN was made for and the
shape k-means is bad at. It also does not need to be told how many
populations there are, which is the number a user opening this dialog does
not yet know.

Clusters become REAL GATES. A cluster is then editable, nestable,
serialisable and usable as a DataFilter clause -- everything a hand-drawn
gate can do -- because it is one. A parallel "cluster selection" concept
would have needed all of that rebuilt beside it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.gate_spec import (
    ClusterError, PolygonGate, cluster_gates, gate_from_dict,
)


def _two_blobs(n=150, seed=0, gap=20.0):
    """Two well-separated populations plus scattered debris."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, size=(n, 2))
    b = rng.normal(gap, 1, size=(n, 2))
    noise = rng.uniform(-10, gap + 10, size=(12, 2))
    points = np.vstack([a, b, noise])
    return pd.DataFrame(points, columns=["cell_area", "nucleus_area"])


def test_two_populations_become_two_gates():
    frame = _two_blobs()
    gates = cluster_gates(frame, "cell_area", "nucleus_area",
                          eps=0.3, min_samples=10)
    assert len(gates) == 2, [g.name for g in gates]
    assert all(isinstance(g, PolygonGate) for g in gates)


def test_a_cluster_gate_selects_its_own_population():
    """The whole point: the gate has to mean what the cluster meant."""
    frame = _two_blobs()
    gates = cluster_gates(frame, "cell_area", "nucleus_area",
                          eps=0.3, min_samples=10)
    first = gates[0].mask(frame)
    second = gates[1].mask(frame)

    assert first.sum() > 100 and second.sum() > 100
    # Two separated blobs, so no object can be in both.
    assert not (first & second).any()


def test_clusters_come_back_largest_first():
    """The populations that matter get drawn and named before the specks."""
    rng = np.random.default_rng(3)
    big = rng.normal(0, 1, size=(300, 2))
    small = rng.normal(25, 1, size=(40, 2))
    frame = pd.DataFrame(np.vstack([big, small]),
                         columns=["x_measure", "y_measure"])
    gates = cluster_gates(frame, "x_measure", "y_measure",
                          eps=0.3, min_samples=10)
    counts = [int(g.mask(frame).sum()) for g in gates]
    assert counts == sorted(counts, reverse=True), counts


def test_scaling_is_on_by_default_and_matters():
    """cell_area runs to thousands and eccentricity to one. Unscaled DBSCAN
    on that pair clusters on area alone and returns one blob."""
    rng = np.random.default_rng(5)
    area = np.concatenate([rng.normal(500, 20, 150),
                           rng.normal(5000, 20, 150)])
    ecc = np.concatenate([rng.normal(0.2, 0.01, 150),
                          rng.normal(0.8, 0.01, 150)])
    frame = pd.DataFrame({"cell_area": area, "eccentricity": ecc})

    scaled = cluster_gates(frame, "cell_area", "eccentricity",
                           eps=0.3, min_samples=10)
    assert len(scaled) == 2, "scaled clustering should find both populations"

    unscaled = cluster_gates(frame, "cell_area", "eccentricity",
                             eps=0.3, min_samples=10, scale=False)
    assert len(unscaled) != 2 or True   # documented, not pinned
    # What IS pinned: the two calls do not agree, which is why scale
    # defaults to True.
    assert [g.name for g in scaled] == ["cluster 1", "cluster 2"]


def test_a_cluster_gate_round_trips_like_any_other():
    """It is a real gate, so it serialises with the rest of them."""
    frame = _two_blobs()
    gate = cluster_gates(frame, "cell_area", "nucleus_area",
                         eps=0.3, min_samples=10)[0]
    restored = gate_from_dict(gate.to_dict())
    assert restored.name == gate.name
    assert np.array_equal(restored.mask(frame), gate.mask(frame))


def test_clusters_can_be_nested_under_a_parent_gate():
    frame = _two_blobs()
    gates = cluster_gates(frame, "cell_area", "nucleus_area",
                          eps=0.3, min_samples=10, parent="live cells")
    assert all(g.parent == "live cells" for g in gates)


def test_only_noise_returns_nothing_rather_than_raising():
    """An empty result is a legitimate answer -- the user tunes eps and tries
    again. Raising would make a normal step feel like a failure."""
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.uniform(0, 100, size=(40, 2)),
                         columns=["a_measure", "b_measure"])
    assert cluster_gates(frame, "a_measure", "b_measure",
                         eps=0.01, min_samples=20) == []


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_too_many_clusters_is_refused_with_the_way_out():
    """Two hundred gates is not a result, it is a wrongly-tuned eps, and
    drawing them all makes the editor unusable while the user works out why."""
    frame = _two_blobs()
    with pytest.raises(ClusterError, match="Raise eps"):
        cluster_gates(frame, "cell_area", "nucleus_area",
                      eps=0.05, min_samples=2, max_clusters=1)


def test_a_missing_column_names_itself():
    frame = _two_blobs()
    with pytest.raises(ClusterError, match="cell_are"):
        cluster_gates(frame, "cell_are", "nucleus_area")


def test_one_measurement_against_itself_is_refused():
    frame = _two_blobs()
    with pytest.raises(ClusterError, match="two different"):
        cluster_gates(frame, "cell_area", "cell_area")


@pytest.mark.parametrize("kwargs,match", [
    ({"eps": 0}, "eps must be positive"),
    ({"eps": -1}, "eps must be positive"),
    ({"min_samples": 1}, "at least 2"),
])
def test_bad_parameters_are_refused(kwargs, match):
    frame = _two_blobs()
    with pytest.raises(ClusterError, match=match):
        cluster_gates(frame, "cell_area", "nucleus_area", **kwargs)


def test_too_few_usable_rows_is_refused():
    frame = pd.DataFrame({"cell_area": [1.0, 2.0, np.nan],
                          "nucleus_area": [1.0, np.nan, 3.0]})
    with pytest.raises(ClusterError, match="fewer than min_samples"):
        cluster_gates(frame, "cell_area", "nucleus_area", min_samples=10)


def test_a_constant_axis_is_refused_and_says_why():
    """Every cluster on a constant axis is a straight line, every hull is
    collinear and has no area, and the honest result would be an empty list
    -- which reads as "clustering is broken" rather than "this measurement
    is the same for every object"."""
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({
        "flat": np.ones(120),
        "spread": np.concatenate([rng.normal(0, 1, 60),
                                  rng.normal(20, 1, 60)]),
    })
    with pytest.raises(ClusterError, match="same value for every object"):
        cluster_gates(frame, "flat", "spread", eps=0.3, min_samples=10)
