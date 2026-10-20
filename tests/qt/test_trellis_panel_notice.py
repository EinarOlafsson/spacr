"""The trellis panel shows its caveats under the grid: the ignored two-way wrap and the shared-axis warning.

Lesson 63 reads ``panel.canvas.notice()`` after typing a wrap into a grid that is faceted both ways, and expects
"wrap ignored: ..." there. This pins the app side of that contract through the whole panel (not only the canvas).
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.widgets.graph_builder import GraphSpec
from spacr.qt.widgets.trellis_spec import TrellisSpec
from spacr.qt.widgets.trellis_view import TrellisPanelWidget


@pytest.fixture
def panel(qtbot):
    frame = pd.DataFrame({"rowID": ["r1", "r1", "r2", "r2"] * 5, "columnID": ["c1", "c2"] * 10,
                          "a": np.arange(20.0), "b": np.arange(20.0)})
    widget = TrellisPanelWidget(link=LinkedSelection())
    qtbot.addWidget(widget)
    widget.set_frame(frame)
    return widget


def test_two_way_wrap_is_reported_under_the_grid(panel):
    panel.canvas.set_trellis_spec(TrellisSpec(
        graph=GraphSpec(x="a", y="b", facet_row="rowID", facet_col="columnID"), wrap=5))
    notice = panel.canvas.notice()
    assert "2 × 2 = 4 panels" in notice
    assert "wrap ignored" in notice


def test_the_notice_is_still_set_when_matplotlib_cannot_lay_out_a_zero_size_canvas(panel, monkeypatch):
    """Lesson 63 dry run: tight_layout raised 'Singular matrix' on an unsized canvas and the notice was never written."""
    def singular(*_a, **_k):
        raise np.linalg.LinAlgError("Singular matrix")
    monkeypatch.setattr(panel.canvas._figure, "tight_layout", singular)
    panel.canvas.set_trellis_spec(TrellisSpec(
        graph=GraphSpec(x="a", y="b", facet_row="rowID", facet_col="columnID"), wrap=5))
    assert "wrap ignored" in panel.canvas.notice()
