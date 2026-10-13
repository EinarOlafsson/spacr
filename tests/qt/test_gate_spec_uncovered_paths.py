"""Gate paths that only run at the edges: the bare shape, one-sided bounds,
and the two places gate_spec depends on scikit-learn.

Three things are asserted here that the ordinary gate tests cannot reach.

The BARE SHAPE. :class:`~spacr.qt.widgets.gate_spec.Gate` is a concrete
dataclass rather than an ABC, so it can be built, and every question that
only a real shape can answer -- what kind it is, which columns it reads,
which rows it keeps, how it reads, how it serialises -- has to refuse rather
than return a default. A base class that answered ``()`` for ``columns`` or
``True`` for every row would make a half-written gate kind look like a
working one.

ONE-SIDED BOUNDS. A gate open on one side is the normal case -- a threshold
dragged to the edge of a histogram, a quadrant gate -- and the open side has
to stay open through masking and through handle placement. The tables below
are small enough that the kept rows are listed rather than counted.

THE TWO SCIKIT-LEARN DOORS. A Walk needs ``silhouette_score`` and HDBSCAN
needs scikit-learn 1.3; both are asked for at call time, and both have to
come back as a :class:`~spacr.qt.widgets.gate_spec.ClusterError` sentence
naming what is missing rather than as a bare ``ImportError`` out of a
button press. The tests take the attribute away from the installed
scikit-learn, which is what an older one looks like from inside the import.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.gate_spec import (ClusterError, Gate, RectGate,
                                        ThresholdGate, cluster_gates,
                                        cluster_walk_candidates)


def _table() -> pd.DataFrame:
    """Five objects, one of which was never measured.

    ====  ====  =========
    idx   area  intensity
    ====  ====  =========
    0       10         10
    1       20         30
    2       30         50
    3       40         70
    4      NaN         90
    ====  ====  =========
    """
    return pd.DataFrame({"area": [10.0, 20.0, 30.0, 40.0, np.nan],
                         "intensity": [10.0, 30.0, 50.0, 70.0, 90.0]})


def _kept(gate, frame) -> list:
    """The row labels the gate keeps, in order."""
    return list(frame.index[gate.mask(frame)])


# ---------------------------------------------------------------------------
# The bare shape answers nothing
# ---------------------------------------------------------------------------

def test_the_bare_shape_will_not_say_what_kind_it_is():
    """A kind is what the tool buttons and the file format are read by."""
    with pytest.raises(NotImplementedError):
        Gate(name="bare").kind


def test_the_bare_shape_will_not_name_the_columns_it_reads():
    """An empty column list would make a gate look re-appliable anywhere."""
    with pytest.raises(NotImplementedError):
        Gate(name="bare").columns


def test_the_bare_shape_selects_no_rows_and_refuses_to_pretend_otherwise():
    """Not "everything" and not "nothing" -- a shape with no geometry has no
    answer, and either default would be silently wrong."""
    with pytest.raises(NotImplementedError):
        Gate(name="bare").mask(_table())


def test_the_bare_shape_has_no_description_to_show_in_the_tree():
    with pytest.raises(NotImplementedError):
        Gate(name="bare").describe()


def test_the_bare_shape_cannot_be_written_to_a_gate_file():
    """Serialising it would write a row nothing can read back."""
    with pytest.raises(NotImplementedError):
        Gate(name="bare").to_dict()


# ---------------------------------------------------------------------------
# One-sided thresholds
# ---------------------------------------------------------------------------

def test_a_threshold_open_below_offers_only_its_upper_anchor():
    """The open side gets no handle: an anchor there is a bound the gate does
    not have, and dragging it would invent one."""
    gate = ThresholdGate(name="dim", column="intensity", high=40.0)
    handles = gate.handles((0.0, 100.0, 0.0, 80.0))
    assert [(h.role, h.x, h.y) for h in handles] == [("high", 40.0, 40.0)]


def test_a_threshold_open_below_keeps_everything_under_its_cut():
    gate = ThresholdGate(name="dim", column="intensity", high=40.0)
    assert _kept(gate, _table()) == [0, 1]


def test_a_threshold_open_above_offers_only_its_lower_anchor():
    gate = ThresholdGate(name="bright", column="intensity", low=40.0)
    handles = gate.handles((0.0, 100.0, 0.0, 80.0))
    assert [(h.role, h.x, h.y) for h in handles] == [("low", 40.0, 40.0)]


def test_a_threshold_below_every_value_keeps_nothing():
    """The empty population is a real answer, reported as an empty set rather
    than as the whole table."""
    gate = ThresholdGate(name="impossible", column="area", low=1000.0)
    assert _kept(gate, _table()) == []


def test_a_threshold_spanning_the_data_still_drops_the_unmeasured_object():
    """"Keeps everything" means every object that HAS the measurement. Row 4
    has no area, and an object with no value is not an object inside a region
    the user defined by value."""
    gate = ThresholdGate(name="all", column="area", low=0.0, high=1000.0)
    assert _kept(gate, _table()) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# A rectangle open on one side of each axis
# ---------------------------------------------------------------------------

def test_a_quadrant_gate_bounds_each_axis_only_where_it_was_drawn():
    """Open above in x and open below in y -- the quadrant gate.

    ``area >= 25`` keeps rows 2 and 3; ``intensity <= 60`` keeps rows 0, 1
    and 2; the rectangle is the intersection, so row 2 alone. Row 4 has no
    area and is outside regardless.
    """
    frame = _table()
    gate = RectGate(name="quadrant", x_column="area", y_column="intensity",
                    x_low=25.0, y_high=60.0)
    assert _kept(gate, frame) == [2]


def test_a_quadrant_gate_and_its_range_clauses_keep_the_same_rows():
    """The clauses handed to the Local Data Filter have to be the gate, not
    an approximation of it -- otherwise nudging them in the panel changes the
    population before the user has touched anything."""
    frame = _table()
    gate = RectGate(name="quadrant", x_column="area", y_column="intensity",
                    x_low=25.0, y_high=60.0)
    combined = np.ones(len(frame), dtype=bool)
    for clause in gate.range_filters():
        combined &= clause.mask(frame)
    assert list(combined) == list(gate.mask(frame))


def test_a_rectangle_drawn_past_the_data_keeps_nothing():
    gate = RectGate(name="empty", x_column="area", y_column="intensity",
                    x_low=100.0, x_high=200.0)
    assert _kept(gate, _table()) == []


def test_a_rectangle_around_the_whole_cloud_keeps_every_measured_object():
    gate = RectGate(name="everything", x_column="area", y_column="intensity",
                    x_low=0.0, x_high=1000.0, y_low=0.0, y_high=1000.0)
    assert _kept(gate, _table()) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# The two scikit-learn doors
# ---------------------------------------------------------------------------

def _two_blobs() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    a = rng.normal(0.0, 0.4, size=(40, 2))
    b = rng.normal(8.0, 0.4, size=(40, 2))
    points = np.vstack([a, b])
    return pd.DataFrame({"x": points[:, 0], "y": points[:, 1]})


def test_a_walk_without_the_silhouette_metric_says_scikit_learn_is_missing(
        monkeypatch):
    """The Walk scores each radius, so it needs the metric before it starts.
    Losing it must read as a sentence in the dialog, not as an ImportError
    escaping the button."""
    metrics = pytest.importorskip("sklearn.metrics")
    monkeypatch.delattr(metrics, "silhouette_score")
    with pytest.raises(ClusterError) as excinfo:
        cluster_walk_candidates(_two_blobs(), "x", "y",
                                eps=0.5, min_samples=5, steps=3)
    assert "scikit-learn" in str(excinfo.value)


def test_choosing_hdbscan_on_an_older_scikit_learn_names_the_version_needed(
        monkeypatch):
    """HDBSCAN arrived in scikit-learn 1.3. Asking for it on an older one has
    to name the version and offer DBSCAN, rather than fall back to DBSCAN and
    return its answer under another name."""
    cluster = pytest.importorskip("sklearn.cluster")
    monkeypatch.delattr(cluster, "HDBSCAN")
    with pytest.raises(ClusterError) as excinfo:
        cluster_gates(_two_blobs(), "x", "y", eps=0.5, min_samples=5,
                      method="hdbscan")
    message = str(excinfo.value)
    assert "1.3" in message
    assert "DBSCAN" in message


def test_dbscan_still_runs_when_hdbscan_is_unavailable(monkeypatch):
    """The other method is offered by the message, so it has to work."""
    cluster = pytest.importorskip("sklearn.cluster")
    monkeypatch.delattr(cluster, "HDBSCAN")
    gates = cluster_gates(_two_blobs(), "x", "y", eps=0.5, min_samples=5,
                          method="dbscan")
    assert [g.name for g in gates] == ["cluster 1", "cluster 2"]


def test_a_rectangle_bounding_one_axis_still_drops_rows_the_other_never_measured():
    """A gate names two columns even when it constrains only one of them.

    ``RectGate.__post_init__`` requires just one of the four bounds, so a
    rectangle drawn across x alone leaves y unbounded. The mask still has to
    exclude a row whose y was never measured: the gate reports on a pair of
    measurements, and a row missing half the pair is not inside it.
    """
    frame = pd.DataFrame({"area": [10.0, 20.0, 30.0],
                          "intensity": [5.0, float("nan"), 50.0]})
    gate = RectGate(name="x only", x_column="area", y_column="intensity",
                    x_low=0.0, x_high=100.0)

    keep = gate.mask(frame)

    assert list(frame.index[keep]) == [0, 2]
    assert not bool(keep[1]), "a row with no intensity is not inside the gate"
