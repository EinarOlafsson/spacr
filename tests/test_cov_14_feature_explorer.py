"""The explorer says why it is empty instead of drawing an empty picture.

Three ways there is nothing to rank or draw, and each has to leave the panel
readable rather than blank:

* no table has been loaded yet, so the summary says so and the panel does not
  try to rank columns it does not have;
* a ranking that kept nothing still repaints the canvas, because a stale
  picture under a new empty result is a picture of the previous table;
* a feature whose column holds no finite value has no bin edges to draw on,
  and a bar chart drawn on no edges is a row of nothing with a title over it
  claiming a score.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _score(feature, **over):
    from spacr.qt.widgets.feature_rank import FeatureScore

    fields = dict(feature=feature, statistic="auc", score=0.8, auc=0.9,
                  cohen_d=1.0, ks=0.5, mutual_info=0.2, higher_in="a",
                  against="b", n_by_class={"a": 4, "b": 4})
    fields.update(over)
    return FeatureScore(**fields)


def _result(scores, spec, label="label"):
    from spacr.qt.widgets.feature_rank import ExplorerResult

    return ExplorerResult(spec=spec, label=label, classes=("a", "b"),
                          scores=tuple(scores), n_rows=8, n_considered=1)


def test_a_panel_with_no_table_says_so_and_ranks_nothing(qtbot):
    """Ranking before a table is loaded refuses and explains the refusal."""
    from spacr.qt.widgets.feature_explorer import FeatureExplorerPanel

    panel = FeatureExplorerPanel()
    qtbot.addWidget(panel)

    assert panel.rank_now() is None
    assert panel.summary() == "no table loaded"
    assert panel.result is None


def test_the_panel_reports_the_spec_it_is_holding(qtbot):
    """``spec`` hands back the spec the controls currently describe."""
    from spacr.qt.widgets.feature_explorer import FeatureExplorerPanel
    from spacr.qt.widgets.feature_rank import ExplorerSpec

    panel = FeatureExplorerPanel()
    qtbot.addWidget(panel)
    wanted = ExplorerSpec(label="label", top=3, bins=8)
    panel.set_spec(wanted)

    assert panel.spec.top == 3
    assert panel.spec.bins == 8


def test_an_empty_ranking_still_repaints_the_canvas(qtbot):
    """A result with no kept features clears rather than leaving the old plot.

    The canvas would otherwise keep showing the previous table's histograms
    under a summary describing the new one.
    """
    from spacr.qt.widgets.feature_explorer import FeatureExplorerPanel
    from spacr.qt.widgets.feature_rank import ExplorerSpec

    panel = FeatureExplorerPanel()
    qtbot.addWidget(panel)
    frame = pd.DataFrame({"label": ["a", "b"] * 4,
                          "area": np.arange(8.0)})
    panel.set_frame(frame)

    panel._draw(_result((), ExplorerSpec(label="label")))

    assert not panel._figure.get_axes()


def test_a_feature_with_no_finite_values_is_left_undrawn(qtbot):
    """A column of NaN yields no bin edges, so its strip is skipped.

    Drawing it would put a title asserting a score over an empty axis.
    """
    from spacr.qt.widgets.feature_explorer import FeatureExplorerPanel
    from spacr.qt.widgets.feature_rank import ExplorerSpec

    panel = FeatureExplorerPanel()
    qtbot.addWidget(panel)
    frame = pd.DataFrame({"label": ["a", "b"] * 4,
                          "blank": [np.nan] * 8})
    panel.set_frame(frame)
    spec = ExplorerSpec(label="label")
    panel._spec = spec

    panel._draw(_result((_score("blank"),), spec))

    drawn = panel._figure.get_axes()
    assert len(drawn) == 1
    assert not drawn[0].patches
    assert drawn[0].get_title(loc="left") == ""


def test_a_feature_with_values_is_drawn(qtbot):
    """The same path does draw bars when the column has finite values."""
    from spacr.qt.widgets.feature_explorer import FeatureExplorerPanel
    from spacr.qt.widgets.feature_rank import ExplorerSpec

    panel = FeatureExplorerPanel()
    qtbot.addWidget(panel)
    frame = pd.DataFrame({"label": ["a", "b"] * 4,
                          "area": np.arange(8.0)})
    panel.set_frame(frame)
    spec = ExplorerSpec(label="label")
    panel._spec = spec

    panel._draw(_result((_score("area"),), spec))

    drawn = panel._figure.get_axes()
    assert len(drawn) == 1
    assert drawn[0].patches
    assert "area" in drawn[0].get_title(loc="left")
