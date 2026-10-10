"""The vvvv export setting of Mask Generation and Measure is alpha (N685).

``export_vvvv`` is registered under item 685 with the Make Masks vvvv
buttons. Its row is hidden on both forms until Preferences -> Show alpha
features is on; a value set while hidden still reaches the run.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

KEY = "export_vvvv"


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_the_setting_is_registered_under_item_685():
    from spacr.settings import ALPHA_FEATURES, _is_alpha, tooltips

    assert ALPHA_FEATURES[685]["settings"] == (KEY,)
    assert _is_alpha("settings", KEY)
    assert tooltips[KEY].startswith("(bool) - ")
    assert tooltips[KEY].endswith("Default False.")


@pytest.mark.parametrize("app_key,heading", [
    ("mask", "Visualization & Diagnostics"),
    ("measure", "Crop Output"),
])
def test_the_row_follows_the_alpha_switch(qtbot, prefs, app_key, heading):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.screens.settings_model import _APP_CATEGORY_SPECS
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    spec = dict(_APP_CATEGORY_SPECS[app_key])
    assert KEY in spec[heading]
    screen = AppScreen(app_key)
    try:
        screen._open_the_heading_of(KEY)
        screen._refresh_alpha_visibility()
        assert not screen.setting_row_is_visible(KEY)
        assert screen._settings_model.collect()[KEY] is False
        assert screen._settings_model.set_value_for_key(KEY, True)
        assert screen._settings_model.collect()[KEY] is True

        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert screen.setting_row_is_visible(KEY)

        prefs._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not screen.setting_row_is_visible(KEY)
        assert screen._settings_model.collect()[KEY] is True
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()
