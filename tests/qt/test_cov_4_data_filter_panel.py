"""Filter rows are restored against the data in front of the user, not the
data they were saved against.

A saved filter set is a set of numbers; the frame it is loaded onto may hold
an entirely different range. Trusting the saved bounds gives an empty
selection, which on screen is indistinguishable from a filter set that failed
to load. The column classifier has its own edge: a frame-shaped object that
cannot be weak-referenced must simply not be cached, because an ``id()``
remembered without a weakref can be re-used by a different object.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.qt.widgets import data_filter_panel as dfp
from spacr.qt.widgets.data_filter_panel import DataFilterPanel, classify_columns


class _FrameThatCannotBeWeakReferenced:
    """A frame-shaped view with no ``__weakref__`` slot."""

    __slots__ = ("_frame",)

    def __init__(self, frame):
        self._frame = frame

    @property
    def columns(self):
        return self._frame.columns

    @property
    def shape(self):
        return self._frame.shape

    def __getitem__(self, name):
        return self._frame[name]


def test_a_frame_that_cannot_be_weak_referenced_is_still_classified():
    """The cache is an optimisation; losing it must not lose the answer."""
    frame = pd.DataFrame({"cell_area": range(100), "plate": ["p1"] * 100})
    view = _FrameThatCannotBeWeakReferenced(frame)
    before = dict(dfp._KINDS_CACHE)
    kinds = classify_columns(view)
    assert kinds == classify_columns(frame)
    assert dfp._KINDS_CACHE.keys() == before.keys() or \
        id(view) not in dfp._KINDS_CACHE


def _panel(qapp, frame):
    class _Link:
        def __init__(self):
            self.published = []

        def publish_filter(self, *args, **kwargs):
            self.published.append((args, kwargs))

        def set_filter(self, *args, **kwargs):
            self.published.append((args, kwargs))

        def subscribe(self, *args, **kwargs):
            return None

    panel = DataFilterPanel(link=_Link())
    panel.set_frame(frame)
    return panel


def test_a_saved_row_names_the_column_and_the_kind_it_was(qapp):
    """A column that has gone away has to be reported, not guessed."""
    frame = pd.DataFrame({"cell_area": [float(i) for i in range(100)]})
    panel = _panel(qapp, frame)
    panel.add_column("cell_area")
    saved = panel.state()
    assert saved["version"] == 1
    assert saved["filters"] == [{"kind": "range", "column": "cell_area",
                                 "low": 0.0, "high": 99.0}]


def test_saved_bounds_are_clamped_to_what_the_current_data_allows(qapp):
    """A set saved against another plate must not silently select nothing."""
    frame = pd.DataFrame({"cell_area": [float(i) for i in range(100)]})
    panel = _panel(qapp, frame)
    missing = panel.restore({"version": 1, "filters": [
        {"kind": "range", "column": "cell_area",
         "low": -1e30, "high": 1e30}]})
    assert missing == []
    row = panel._rows["cell_area"]
    assert row._low.value() == pytest.approx(row._low.minimum())
    assert row._high.value() == pytest.approx(row._high.maximum())


def test_restoring_bounds_inside_the_range_keeps_them_exactly(qapp):
    """Clamping must not move a bound that was already legal."""
    frame = pd.DataFrame({"cell_area": [float(i) for i in range(100)]})
    panel = _panel(qapp, frame)
    panel.restore({"version": 1, "filters": [
        {"kind": "range", "column": "cell_area", "low": 10.0, "high": 20.0}]})
    row = panel._rows["cell_area"]
    assert row._low.value() == pytest.approx(10.0)
    assert row._high.value() == pytest.approx(20.0)
    assert panel.state()["filters"][0]["low"] == pytest.approx(10.0)
