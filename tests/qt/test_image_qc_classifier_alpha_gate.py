"""The learned image-quality classifier in Mask is an alpha feature.

Its four settings sit in the Image Quality heading of the Mask form and stay
hidden until Preferences -> Show alpha features is on; a value saved while
hidden still reaches the run.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

KEYS = ("image_qc_classifier", "image_qc_classifier_model",
        "image_qc_classifier_labels", "image_qc_classifier_threshold")


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_the_settings_are_registered_under_their_item():
    from spacr.image_quality import DEFAULTS
    from spacr.settings import ALPHA_FEATURES, _is_alpha

    assert ALPHA_FEATURES[559] == {
        "settings": KEYS,
        "widgets": ("AnnotateFieldQCButton", "AnnotateFieldQCDialog",
                    "QCClassifierCard")}
    assert set(KEYS) <= set(DEFAULTS)
    assert all(_is_alpha("settings", key) for key in KEYS)


def test_the_gate_hides_them_until_alpha_features_are_on(prefs):
    assert not any(prefs._is_alpha_visible("settings", k) for k in KEYS)
    prefs._set_show_alpha_features(True)
    assert all(prefs._is_alpha_visible("settings", k) for k in KEYS)


def test_the_mask_rows_follow_the_switch(qtbot, prefs):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    screen = AppScreen("mask")
    try:
        screen._open_the_heading_of(KEYS[0])
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(k) for k in KEYS)
        assert screen.setting_row_is_visible("image_qc_mode")
        assert screen._settings_model.set_value_for_key(KEYS[0], True)
        assert screen._settings_model.set_value_for_key(KEYS[3], 0.8)

        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert all(screen.setting_row_is_visible(k) for k in KEYS)

        prefs._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(k) for k in KEYS)
        collected = screen._settings_model.collect()
        assert collected[KEYS[0]] is True
        assert collected[KEYS[3]] == 0.8
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()
