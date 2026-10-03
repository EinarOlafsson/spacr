"""The generative counterfactuals in Activation Maps are an alpha feature.

Their three settings form the Counterfactuals heading of the Activation Maps
form, hidden until Preferences -> Show alpha features is on; a value saved
while hidden still reaches the run.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

KEYS = ("counterfactuals", "counterfactual_crops", "counterfactual_epochs",
        "counterfactual_condition")


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_the_settings_are_registered_under_their_item():
    from spacr.settings import (ALPHA_FEATURES, _is_alpha,
                                get_default_generate_activation_map_settings)

    assert ALPHA_FEATURES[564] == {
        "settings": KEYS, "widgets": ("ActivationCounterfactualViewer",)}
    assert set(KEYS) <= set(get_default_generate_activation_map_settings({}))
    assert all(_is_alpha("settings", key) for key in KEYS)


def test_the_gate_hides_them_until_alpha_features_are_on(prefs):
    assert not any(prefs._is_alpha_visible("settings", k) for k in KEYS)
    prefs._set_show_alpha_features(True)
    assert all(prefs._is_alpha_visible("settings", k) for k in KEYS)


def test_the_activation_rows_follow_the_switch(qtbot, prefs):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    screen = AppScreen("activation")
    try:
        screen._open_the_heading_of(KEYS[0])
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(k) for k in KEYS)
        assert screen._alpha_hidden_sections()
        assert screen._settings_model.set_value_for_key(KEYS[0], True)
        assert screen._settings_model.set_value_for_key(KEYS[2], 5)

        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert all(screen.setting_row_is_visible(k) for k in KEYS)

        prefs._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(k) for k in KEYS)
        collected = screen._settings_model.collect()
        assert collected[KEYS[0]] is True
        assert collected[KEYS[2]] == 5
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()


def test_the_viewer_button_follows_the_switch_and_shows_sequences(
        qtbot, prefs, tmp_path):
    import numpy as np
    import pandas as pd

    from spacr.qt.screens.activation import (_add_counterfactual_viewer_button,
                                             _counterfactual_viewer)
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus
    from spacr.tabular import write_table

    screen = AppScreen("activation")
    try:
        button = _add_counterfactual_viewer_button(screen)
        assert button.objectName() == "ActivationCounterfactualViewer"
        assert button.isHidden()
        prefs._set_show_alpha_features(True)
        from spacr.qt.preferences import _apply_alpha_widgets
        _apply_alpha_widgets(button)
        assert not button.isHidden()
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()

    write_table(pd.DataFrame([{
        "name": "p1_A01_1_1.png", "source_class": 0, "target_class": 1,
        "source_condition": "c1", "target_condition": "c2",
        "score_start": 0.1, "score_end": 0.9, "score_path": "0.1;0.5;0.9",
        "flipped": True}]), tmp_path / "counterfactual_cells.csv")
    np.save(tmp_path / "counterfactual_frames.npy",
            np.random.default_rng(0).random((1, 3, 1, 8, 8)).astype("float32"))
    dialog = _counterfactual_viewer(str(tmp_path))
    qtbot.addWidget(dialog)
    assert dialog.listing.count() == 1
    assert "c1→c2" in dialog.listing.item(0).text()
    assert len(dialog.strip_labels) == 6
    assert dialog.strip_labels[5].text() == "0.9"
