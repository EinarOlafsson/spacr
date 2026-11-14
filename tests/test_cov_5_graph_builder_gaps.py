"""The chart's quieter selection branches: redraw instead of restyle, and too early.

An aggregate chart answers a selection made elsewhere by redrawing, and a
selection that arrives before the first draw is ignored.  The draws, sweeps
and shelf controls live in ``tests/qt/test_graph_builder_canvas_draws_and_sweeps.py``.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.linked_selection import LinkedSelection            # noqa: E402
from spacr.qt.widgets import graph_builder as gb                 # noqa: E402
from spacr.qt.widgets.graph_spec import (BAR, GraphSpec)
from spacr.selection import Selection, as_key_index              # noqa: E402

pytestmark = pytest.mark.qt


def _frame(rows=60):
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "plateID": [f"p{i % 2 + 1}" for i in range(rows)],
        "rowID": [f"r{i % 3 + 1}" for i in range(rows)],
        "columnID": [f"c{i % 4 + 1}" for i in range(rows)],
        "fieldID": [f"f{i % 2 + 1}" for i in range(rows)],
        "object_label": list(range(rows)),
        "area": rng.normal(100, 10, rows),
        "intensity": rng.normal(size=rows),
        "gene": [f"g{i % 3}" for i in range(rows)],
    })


@pytest.fixture()
def canvas(qtbot):
    view = gb.GraphCanvas(link=LinkedSelection(), source="cov5")
    qtbot.addWidget(view)
    view.set_frame(_frame())
    return view


# ---------------------------------------------------------------------------
# Answering a selection made elsewhere
# ---------------------------------------------------------------------------

def test_an_aggregate_chart_redraws_rather_than_restyling(canvas,
                                                           monkeypatch):
    """A bar's highlight is a recomputed reduction, not a re-coloured artist."""
    canvas.set_spec(GraphSpec(x="gene", y="area", kind=BAR), immediate=True)
    assert canvas._live_highlight is False
    drawn = []
    monkeypatch.setattr(canvas, "render_now", lambda: drawn.append(True))

    canvas.link.set_selection(
        Selection(keys=as_key_index(["p1_r1_c1_f1_1"]), source="elsewhere"))

    assert drawn == [True]


def test_a_selection_arriving_before_the_first_draw_is_ignored(qtbot):
    view = gb.GraphCanvas(link=LinkedSelection(), source="cov5empty")
    qtbot.addWidget(view)

    view.on_linked_selection_changed(
        Selection(keys=as_key_index(["p1_r1_c1_f1_1"]), source="elsewhere"))

    assert view._selected_mask is None
