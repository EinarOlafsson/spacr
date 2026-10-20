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


@pytest.mark.parametrize("app_key", ["mask", "measure", "timelapse"])
def test_no_module_form_shows_the_row(qtbot, prefs, app_key):
    """The maintainer, 2026-10-16: the switch is Preferences -> Plugins.

    The key stays in the run's settings, hidden, at its default; a run
    started from the app takes its value from the preference.
    """
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.screens.settings_model import (_APP_CATEGORY_SPECS,
                                                 _APP_HIDDEN_KEYS)
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    assert KEY in _APP_HIDDEN_KEYS[app_key]
    for _heading, keys in dict(_APP_CATEGORY_SPECS.get(app_key, ())).items():
        assert KEY not in keys
    screen = AppScreen(app_key)
    try:
        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert not screen.setting_row_is_visible(KEY)
        assert screen._settings_model.collect()[KEY] is False
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()
