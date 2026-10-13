"""Brushing across a categorical axis selects the groups under the sweep.

A brush is a predicate over the frame, not a hit test against drawn marks --
that is what keeps it exact when the panel was binned or sampled. On a
continuous axis the predicate is an interval on the values. On a **categorical**
axis the drawn positions are tick indices, so the predicate has to be
translated back through the shared level order first: sweeping from 0.6 to 2.4
on a box plot of four plates means "the second and third boxes", not "values
between 0.6 and 2.4".

The level order is the one in :class:`~spacr.qt.widgets.graph_spec.Scales`,
computed once over the whole frame, because that is what the renderer drew the
ticks from. Re-deriving it here would put the boxes in a different order than
the panel shows and select the wrong groups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spacr.qt.widgets.graph_spec import (GraphSpec, brush_mask, column_kinds,
                                         scales_for)

PLATES = ("p1", "p2", "p3", "p4")


def _frame(n=24):
    generator = np.random.default_rng(0)
    return pd.DataFrame({
        "area": generator.normal(100.0, 20.0, n),
        "plateID": [PLATES[i % len(PLATES)] for i in range(n)],
    })


def _box_plot(frame):
    spec = GraphSpec(x="plateID", y="area", kind="box")
    kinds = column_kinds(frame)
    return spec, kinds, scales_for(frame, spec, kinds)


def test_the_level_order_the_ticks_were_drawn_from_is_the_one_used():
    frame = _frame()
    _spec, _kinds, scales = _box_plot(frame)

    assert scales.x_levels == PLATES


def test_a_sweep_across_two_boxes_selects_exactly_those_two_groups():
    frame = _frame()
    spec, kinds, scales = _box_plot(frame)

    mask = brush_mask(frame, spec, kinds, 0.6, -1e9, 2.4, 1e9, scales)

    assert sorted(set(frame.loc[mask, "plateID"])) == ["p2", "p3"]
    # Every row of those groups, not a sample of them: the predicate is
    # evaluated on the frame, not on the marks that were drawn.
    assert int(mask.sum()) == int(frame["plateID"].isin(["p2", "p3"]).sum())


def test_a_sweep_over_one_tick_selects_one_group():
    frame = _frame()
    spec, kinds, scales = _box_plot(frame)

    mask = brush_mask(frame, spec, kinds, -0.4, -1e9, 0.4, 1e9, scales)

    assert sorted(set(frame.loc[mask, "plateID"])) == ["p1"]


def test_a_sweep_that_misses_every_tick_selects_nothing():
    """Between two boxes is not "the nearer box"; it is no box."""
    frame = _frame()
    spec, kinds, scales = _box_plot(frame)

    mask = brush_mask(frame, spec, kinds, 9.0, -1e9, 12.0, 1e9, scales)

    assert not mask.any()


def test_without_the_shared_levels_a_categorical_axis_constrains_nothing():
    """No level order means no way to turn a tick position into a group.

    Guessing one would select whichever groups the local sort happened to put
    under the sweep, which is worse than selecting on the other axis alone.
    """
    frame = _frame()
    spec, kinds, _scales = _box_plot(frame)

    mask = brush_mask(frame, spec, kinds, 0.6, -1e9, 2.4, 1e9, None)

    assert mask.all()
