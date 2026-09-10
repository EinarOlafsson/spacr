"""Axis classification survives a frame that cannot be scanned for order.

Deciding whether an x-axis is a series or a cloud means reading its values.
A frame whose column access fails part-way through must still yield a usable
shape, because the plot picker calls this before it can offer any graph at
all -- an exception here would leave the user with no graph types to choose.
"""
from __future__ import annotations

import pandas as pd

from spacr.graph_types import shape_of, types_for


class _FrameThatBreaksOnRescan:
    """Reports numeric columns, then raises when the values are re-read."""

    def __init__(self):
        self.columns = ("x", "y")
        self._reads = 0

    def __getitem__(self, name):
        self._reads += 1
        if self._reads > 2:
            raise RuntimeError("column vanished mid-scan")
        return pd.Series([1.0, 2.0, 3.0])


def test_a_frame_that_cannot_be_rescanned_is_a_cloud_not_a_crash():
    """Unreadable ordering falls back to the unordered two-axis shape."""
    frame = _FrameThatBreaksOnRescan()
    assert shape_of(frame, "x", "y") == "continuous_continuous"
    assert types_for("continuous_continuous")


def test_a_numeric_x_against_a_categorical_y_is_a_single_measurement():
    """Only x carries a measurement, so the shape is the one-axis shape."""
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": ["a", "b", "c"]})
    assert shape_of(frame, "x", "y") == "continuous_only"


def test_a_sorted_unique_x_is_an_ordered_series():
    """The ordered branch is what makes a line an offered graph type."""
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    assert shape_of(frame, "x", "y") == "ordered_continuous"
