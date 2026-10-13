"""One name for one idea, and a metric field that fails where you can see it.

"there should be a clustering algorithm option with a hyperparameter search
and which should include a hyperparameter walk (that is what the setting
should be called, Walk, this name should also apply to the new image UMAP
adaptive setting!)"

A directional search through hyperparameter space is the same thing in the
Image UMAP screen and in the Gate Editor's clustering, and it was called
"Adaptive 2x2" in one and "Walk" in the other. "2x2" also described the
current two-parameter special case rather than the design.
"""

from __future__ import annotations

import pytest


def test_the_umap_search_calls_it_walk(qtbot):
    from spacr.qt.screens.hyperparam import HyperparamPanel

    panel = HyperparamPanel("umap")
    qtbot.addWidget(panel)
    assert panel._adaptive.text() == "Walk"
    assert "2×2" not in panel._adaptive.text()


def test_the_gate_editor_calls_it_walk_too(qtbot):
    from spacr.qt.widgets.gate_settings import (GateEditorSettings,
                                                GateSettingsDialog)

    dialog = GateSettingsDialog(GateEditorSettings())
    qtbot.addWidget(dialog)
    assert dialog._walk.text() == "Walk"


def test_the_tooltip_says_what_a_walk_is(qtbot):
    """A control named after a metaphor has to define it once."""
    from spacr.qt.screens.hyperparam import HyperparamPanel

    panel = HyperparamPanel("umap")
    qtbot.addWidget(panel)
    tip = panel._adaptive.toolTip().lower()
    assert "direction" in tip or "steps" in tip


# ---------------------------------------------------------------------------
# The metric field
# ---------------------------------------------------------------------------

def test_a_bad_metric_is_refused_where_the_user_can_see_it():
    """It used to fail deep inside the run, after the embedding started.

    That is the difference between a sentence under the control and a
    traceback twenty minutes in.
    """
    from spacr.qt.screens.hyperparam import parse_values

    with pytest.raises(ValueError, match="metric"):
        parse_values("euclidian", "metric", "metric")   # the usual typo


def test_a_real_metric_passes():
    from spacr.qt.screens.hyperparam import parse_values

    assert parse_values("euclidean, cosine", "metric", "metric") == \
        ["euclidean", "cosine"]


def test_the_list_comes_from_the_installed_umap_when_there_is_one():
    """So it cannot drift from the version that has to accept it."""
    from spacr.qt.screens.hyperparam import UMAP_METRICS, umap_metrics

    metrics = umap_metrics()
    assert "euclidean" in metrics
    try:
        from umap.distances import named_distances
    except Exception:
        assert metrics == UMAP_METRICS
    else:
        assert set(metrics) == set(named_distances)


def test_the_panel_builds_without_umap_installed(monkeypatch):
    """A user configuring a run on a laptop and executing it elsewhere is
    ordinary; the settings panel must not need the library."""
    import builtins

    from spacr.qt.screens import hyperparam

    real_import = builtins.__import__

    def _no_umap(name, *args, **kwargs):
        if name.startswith("umap"):
            raise ImportError("no umap here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_umap)
    assert hyperparam.umap_metrics() == hyperparam.UMAP_METRICS
