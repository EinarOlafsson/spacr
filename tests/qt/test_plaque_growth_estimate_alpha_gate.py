"""Plaque Assay's experimental scale / time estimate is an alpha feature.

The Figure-preview button is registered as ``PlaqueEstimateScaleTime`` (and
its reference note as ``PlaqueEstimateScaleTimeNote``) in
``spacr.settings.ALPHA_FEATURES``: shown only while Preferences -> "Show
alpha features" is on, and a value saved while it is hidden still reaches
the run.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def _panel(qtbot, monkeypatch, shown):
    from spacr.qt import preferences
    from spacr.qt.widgets import plaque_preview as pv

    monkeypatch.setattr(pv, "missing_papers_packages", lambda *a, **k: [])
    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: shown)
    panel = pv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    return panel


def test_registered_under_its_item():
    from spacr.settings import ALPHA_FEATURES

    assert {"PlaqueEstimateScaleTime", "PlaqueEstimateScaleTimeNote"} <= set(
        ALPHA_FEATURES[501]["widgets"])
    assert "plaque_estimate_growth" in ALPHA_FEATURES[501]["settings"]


def test_the_button_is_hidden_unless_alpha_features_are_shown(
        qtbot, monkeypatch):
    from spacr.qt import preferences

    for shown in (False, True):
        panel = _panel(qtbot, monkeypatch, shown)
        button = panel._growth_btn
        assert button.objectName() == "PlaqueEstimateScaleTime"
        assert button.isHidden() is not shown
        monkeypatch.setattr(preferences, "_get_show_alpha_features",
                            lambda s=shown: not s)
        preferences._apply_alpha_widgets(panel)
        assert button.isHidden() is shown


def test_a_value_saved_while_hidden_still_reaches_the_run(qtbot, monkeypatch):
    from spacr.qt.widgets import plaque_preview as pv

    panel = _panel(qtbot, monkeypatch, False)
    panel.set_mode(pv.FIGURE_MODE)
    panel.apply_settings({"plaque_estimate_growth": True})
    assert panel._growth_btn.isHidden()
    assert panel._growth_note.isHidden()
    assert panel.current_settings()["plaque_estimate_growth"] is True
    assert panel.settings_for_propagation()["plaque_estimate_growth"] is True
