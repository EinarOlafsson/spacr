"""A time series with no z axis, which is half the movies spaCR reads.

``AxisOrder`` and ``TStackSpec`` both treat ``z_axis=None`` as an ordinary
value -- a flat TYX time series is what a widefield movie is -- and every
uncovered arc here is that None being handled. The fixtures in the suite are
all 4-D, which is why the 3-D case was never exercised: the common shape got
the coverage and the other common shape did not.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# AxisOrder — a flat time series
# ---------------------------------------------------------------------------

def test_a_flat_time_series_validates_without_a_z_axis():
    """Arc 1603 -> 1605: the z axis is not added to the uniqueness check.

    ``__post_init__`` collects the axes to prove none is used twice. With no z
    there are three, and appending a None would make the check compare an
    integer with a None and either raise or silently pass.
    """
    from spacr.zstack import AxisOrder

    order = AxisOrder(t_axis=0, z_axis=None, y_axis=1, x_axis=2)

    assert order.z_axis is None
    assert order.t_axis == 0


def test_a_flat_time_series_is_named_tyx():
    """Arc 1618 -> 1620: no Z letter in the name.

    The name is what gets written next to the tracks, so a TYX movie labelled
    TZYX would misdescribe the data it came from in the run's own record.
    """
    from spacr.zstack import AxisOrder

    assert AxisOrder(t_axis=0, z_axis=None, y_axis=1, x_axis=2).name == "TYX"


def test_a_volume_time_series_is_named_tzyx():
    """The taken side, so the absence above is visibly a decision."""
    from spacr.zstack import AxisOrder

    assert AxisOrder(t_axis=0, z_axis=1, y_axis=2,
                     x_axis=3).name == "TZYX"


def test_a_channel_axis_is_named_and_ordered():
    """Both optional axes together, in ascending axis order as documented."""
    from spacr.zstack import AxisOrder

    order = AxisOrder(t_axis=0, z_axis=1, y_axis=3, x_axis=4, channel_axis=2)

    assert order.name == "TZCYX"


def test_an_axis_used_twice_is_refused():
    """The uniqueness check the None must not break."""
    from spacr.zstack import AxisOrder

    with pytest.raises(Exception):
        AxisOrder(t_axis=0, z_axis=0, y_axis=1, x_axis=2)


# ---------------------------------------------------------------------------
# TStackSpec — a channel axis on a flat series
# ---------------------------------------------------------------------------

def test_a_channel_axis_is_checked_against_time_alone_when_there_is_no_z():
    """Arc 1925 -> 1927.

    The clash check builds the set of axes already taken. Without a z axis
    that set is just time, and the channel must still be checked against it --
    a TCYX movie whose channel axis is 0 is a real mistake, and one that
    produces a plausible-looking wrong movie rather than an error.
    """
    from spacr.zstack import TStackError, TStackSpec

    good = TStackSpec(t_axis=0, z_axis=None, channel_axis=1)
    assert good.channel_axis == 1

    with pytest.raises(TStackError):
        TStackSpec(t_axis=0, z_axis=None, channel_axis=0)


def test_a_channel_axis_clashing_with_z_is_refused():
    """The taken side of the same check, with z present."""
    from spacr.zstack import TStackError, TStackSpec

    with pytest.raises(TStackError):
        TStackSpec(t_axis=0, z_axis=1, channel_axis=1)


# ---------------------------------------------------------------------------
# _apply_track_maps — a label the frame does not have
# ---------------------------------------------------------------------------

def test_a_track_map_naming_a_label_beyond_the_frame_is_passed_over():
    """Arc 2504 -> 2503: the loop skips it rather than growing the lookup.

    The map comes from the linker and the frame from the masks, and they can
    disagree -- a label dropped by a later filter still has a map entry.
    Writing it would index past the end of the LUT, which is an IndexError in
    the middle of relabelling a whole movie.
    """
    from spacr.zstack import _apply_track_maps

    labels = np.zeros((2, 3, 3), dtype=np.int64)
    labels[0, 0, 0] = 1
    labels[1, 1, 1] = 1

    # The second entry names a label this frame does not contain.
    maps = [{1: 10, 99: 20}, {1: 10}]

    out = _apply_track_maps(labels, maps)

    assert out.shape == labels.shape
    assert int(out.max()) >= 1
