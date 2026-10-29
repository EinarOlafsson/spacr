"""A public registry unload keeps native menus and Help navigation usable."""

import pytest

from tests.app_registry_state import app_registry_snapshot, restore_app_registry_to
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile

pytestmark = pytest.mark.qt


def test_unloading_a_section_keeps_native_help_fallback_and_home_navigation(
        qtbot, qt_theme_applied, private_ai_profile, monkeypatch):
    import spacr.qt
    from spacr.qt import app, i18n
    from spacr.qt.screens.run_history import RunHistoryScreen

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(False)
    preferences.set_reduce_motion(True)
    monkeypatch.setenv(i18n.ENV_LANGUAGE, 'en')
    spacr.qt.register_self_registering_modules()
    saved = app_registry_snapshot(app)
    removed = [row[0] for row in app.APPS if row[3] == app.SECTION_DATA]
    assert 'run_history' in removed and removed
    window = None
    try:
        for key in removed:
            assert app.unregister_app(key)
        assert app.SECTION_DATA in app.SECTION_ORDER
        assert app.SECTION_DATA not in app.SECTIONS
        window = app.MainWindow(initial_app='__home__')
        qtbot.addWidget(window)
        window.show()
        assert app.SECTION_DATA not in window._section_menus
        assert app.SECTION_CORE in window._section_menus
        assert all(key not in window._app_actions for key in removed)
        assert window._app_actions['mask'].isEnabled()
        fallback = next(action for action in window._easy_help_menu.actions()
                        if action.text().replace('&', '') == 'Run History')
        assert fallback.isEnabled() and fallback.statusTip()
        fallback.trigger()
        assert isinstance(window._screens['run_history'], RunHistoryScreen)
        assert window._stack.currentWidget() is window._screens['run_history']
        home = window.menuBar().actions()[0].menu().actions()[0]
        assert home.text().replace('&', '') == 'Home'
        home.trigger()
        assert window._stack.currentWidget() is window._startup
        assert window._app_actions['mask'].isEnabled()
    finally:
        if window is not None:
            window.close()
        restore_app_registry_to(app, saved)
    assert list(app.APPS) == saved[0]
    assert app.APP_FACTORIES == saved[1]
    assert app.APP_STAGE == saved[2]
    assert app.APP_META == saved[3]
    assert all(table == original for table, original in saved[4])
