"""The data filter panel with a frame it cannot weak-reference and a row it could not add."""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.qt.widgets import data_filter_panel as dfp
from spacr.qt.widgets.data_filter_panel import DataFilterPanel, classify_columns

pytestmark = pytest.mark.qt


class _FrameThatCannotBeWeakReferenced:
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


def _panel(qapp, frame):
    class _Link:
        def publish_filter(self, *a, **k):
            pass

        def set_filter(self, *a, **k):
            pass

        def subscribe(self, *a, **k):
            return None

    panel = DataFilterPanel(link=_Link())
    panel.set_frame(frame)
    return panel


def test_known_kinds_for_a_frame_that_cannot_be_weak_referenced_are_not_cached():
    frame = pd.DataFrame({"cell_area": range(100), "plate": ["p1"] * 100})
    view = _FrameThatCannotBeWeakReferenced(frame)
    kinds = {"cell_area": "range", "plate": "categorical", "extra": "range"}
    dfp._remember_column_kinds(view, kinds)
    assert id(view) not in dfp._KINDS_CACHE
    # the frame-shaped view still classifies from the table itself
    assert classify_columns(view) == classify_columns(frame)


def test_a_saved_row_whose_clause_was_not_added_is_skipped_not_restored(qapp, monkeypatch):
    frame = pd.DataFrame({"cell_area": [float(i) for i in range(100)]})
    panel = _panel(qapp, frame)
    monkeypatch.setattr(panel, "add_column", lambda column: None)
    missing = panel.restore({"version": 1, "filters": [
        {"kind": "range", "column": "cell_area", "low": 10.0, "high": 20.0},
        {"kind": "range", "column": "gone", "low": 0, "high": 1}]})
    assert missing == ["gone"]
    assert panel._rows == {}
    assert panel.state() == {"version": 1, "filters": []}
