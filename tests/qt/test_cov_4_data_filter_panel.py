"""Filter rows are restored against the data in front of the user, not the
data they were saved against.

A saved filter set is a set of numbers; the frame it is loaded onto may hold
an entirely different range. Trusting the saved bounds gives an empty
selection, which on screen is indistinguishable from a filter set that failed
to load.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.qt.widgets.data_filter_panel import DataFilterPanel


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
