"""What the PCA surface does when the decomposition behind it is not sound.

The arrows are a decoration over a scatter that stands on its own, so each
of these takes something away from the loadings -- a correlation, a feature
name, the fit itself -- and asks for the points anyway.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.widgets.graph_spec import SCATTER, GraphSpec
from spacr.qt.widgets.pca_model import (PCAError, PCASpec, candidate_features,
                                        pca)
from spacr.qt.widgets import pca_view
from spacr.qt.widgets.pca_view import FeaturePicker, PCAPanel, PCAScoresCanvas

pytestmark = pytest.mark.qt

FEATURES = ("area", "perimeter", "intensity")


def cluster_frame(n: int = 120, seed: int = 7) -> pd.DataFrame:
    """Two groups apart along one direction, in three different units."""
    rng = np.random.default_rng(seed)
    half = n // 2
    group = np.array(["control"] * half + ["knockdown"] * (n - half))
    shift = np.where(group == "control", -1.0, 1.0)
    return pd.DataFrame({
        "plateID": ["p1"] * n,
        "rowID": ["r1"] * n,
        "columnID": [f"c{i % 4 + 1}" for i in range(n)],
        "fieldID": ["f1"] * n,
        "object_label": np.arange(n),
        "area": 900.0 + 120.0 * shift + rng.normal(scale=25.0, size=n),
        "perimeter": 110.0 + 9.0 * shift + rng.normal(scale=2.0, size=n),
        "intensity": 140.0 - 6.0 * shift + rng.normal(scale=3.0, size=n),
        "gene": group,
    })


@pytest.fixture
def link() -> LinkedSelection:
    """A PRIVATE link -- never the process-wide one."""
    return LinkedSelection()


def a_decomposition():
    frame = cluster_frame()
    return frame, pca(frame, PCASpec(features=FEATURES, n_components=2))


# ---------------------------------------------------------------------------
# The feature list
# ---------------------------------------------------------------------------

def test_the_picker_offers_exactly_the_tables_continuous_columns(qtbot):
    """``available`` is what the tick list was built from -- the plate and
    well identifiers are deliberately not among them."""
    picker = FeaturePicker()
    qtbot.addWidget(picker)
    frame = cluster_frame()

    picker.set_frame(frame)

    assert picker.available() == tuple(candidate_features(frame))
    assert set(FEATURES) <= set(picker.available())
    assert "plateID" not in picker.available()
    assert "gene" not in picker.available()

    picker.set_frame(None)
    assert picker.available() == ()


# ---------------------------------------------------------------------------
# The colour list
# ---------------------------------------------------------------------------

def test_a_panel_pointed_at_no_table_offers_nothing_to_colour_by(qtbot, link):
    """Emptying the panel has to empty the colour list with it, or the box
    goes on offering columns of a table that is no longer loaded."""
    panel = PCAPanel(link=link)
    qtbot.addWidget(panel)
    panel.set_frame(cluster_frame())
    assert panel._colour.count() > 1

    panel.set_frame(None, compute=False)

    assert panel._colour.count() == 1
    assert panel._colour.itemText(0) == "none"
    assert panel.features.available() == ()


# ---------------------------------------------------------------------------
# Arrows that cannot be drawn
# ---------------------------------------------------------------------------

def test_a_feature_whose_correlation_is_not_a_number_gets_no_arrow(qtbot,
                                                                   link):
    """The other features keep theirs. An arrow to a NaN would be a line to
    nowhere on a plot the reader takes at face value."""
    frame, result = a_decomposition()
    scores = result.scores_frame(frame)

    canvas = PCAScoresCanvas(link=link)
    qtbot.addWidget(canvas)
    canvas.set_result(result, scores)
    canvas.set_spec(GraphSpec(x="PC1", y="PC2", kind=SCATTER))
    whole = len(canvas.axes_at(0, 0).texts)
    assert whole == 2 * len(result.features)

    broken = np.array(result.correlations, dtype=float)
    broken[1, :] = np.nan
    canvas.set_result(replace(result, correlations=broken), scores)
    canvas.set_spec(GraphSpec(x="PC1", y="PC2", kind=SCATTER))

    assert canvas.arrow_scale > 0.0                 # the ruler still stands
    assert len(canvas.axes_at(0, 0).texts) == whole - 2
    assert result.features[1] not in [
        text.get_text() for text in canvas.axes_at(0, 0).texts]


def test_a_result_the_arrows_cannot_be_read_from_still_draws_the_points(
        qtbot, link):
    """A result whose feature names do not reach as far as its correlations
    cannot be labelled -- and the scatter it decorates is not its to lose."""
    frame, result = a_decomposition()
    scores = result.scores_frame(frame)
    mismatched = replace(result, features=result.features[:1])

    canvas = PCAScoresCanvas(link=link)
    qtbot.addWidget(canvas)
    canvas.set_result(mismatched, scores)
    canvas.set_spec(GraphSpec(x="PC1", y="PC2", kind=SCATTER))

    ax = canvas.axes_at(0, 0)
    assert ax is not None
    assert ax.collections, "the scores are still on the axes"
    assert len(ax.texts) < 2 * len(result.features)


# ---------------------------------------------------------------------------
# A fit that fails in a way the model did not name
# ---------------------------------------------------------------------------

def test_a_fit_that_fails_for_an_unnamed_reason_says_so_in_the_report(
        qtbot, link, monkeypatch):
    """``PCAError`` carries its own sentence; anything else has to be turned
    into one here rather than escaping onto the worker's thread."""
    panel = PCAPanel(link=link)
    qtbot.addWidget(panel)
    panel.set_frame(cluster_frame())
    assert panel.result is not None

    def _explodes(frame, spec=None):
        raise MemoryError("not enough room for the Gram matrix")

    monkeypatch.setattr(pca_view, "pca", _explodes)

    with qtbot.waitSignal(panel.failed, timeout=2000) as caught:
        panel.recompute()

    assert caught.args[0].startswith("PCA failed: ")
    assert "Gram matrix" in caught.args[0]
    assert panel.report.text() == caught.args[0]
    assert panel.result is None
    assert panel.canvas.result is None


def test_a_refusal_the_model_names_is_shown_as_the_model_worded_it(qtbot,
                                                                  link):
    """The other side of the same seam: a ``PCAError`` is not re-worded."""
    panel = PCAPanel(link=link)
    qtbot.addWidget(panel)
    frame = cluster_frame()
    frame["area"] = 1.0
    frame["perimeter"] = 2.0
    frame["intensity"] = 3.0

    with qtbot.waitSignal(panel.failed, timeout=2000) as caught:
        panel.set_frame(frame)

    assert not caught.args[0].startswith("PCA failed: ")
    assert "no continuous columns to decompose" in caught.args[0]
    assert panel.report.text() == caught.args[0]


# ---------------------------------------------------------------------------
# A component the pickers do not carry
# ---------------------------------------------------------------------------

def test_a_scree_pick_for_a_component_that_is_not_offered_moves_no_axis(
        qtbot, link):
    """The scree plot never emits one, but the slot is public and a stale
    index must not put a name the pickers do not hold onto an axis."""
    panel = PCAPanel(link=link)
    qtbot.addWidget(panel)
    panel.set_frame(cluster_frame())
    assert panel.result is not None
    before = panel._pc_x.currentData()

    panel.scree.component_picked.emit(panel.result.n_components + 40)

    assert panel._pc_x.currentData() == before
