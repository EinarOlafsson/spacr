"""Native navigation queries stay distinct from inspected filesystem sources."""
from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtTest import QSignalSpy

from tests.qt.test_733_easy_mode_home import _image
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


def test_navigation_protocol_context_and_cache_shortcut_preserve_local_inputs(
        qtbot, private_ai_profile, tmp_path, monkeypatch):
    import webbrowser

    from spacr.qt.app import MainWindow
    from spacr.qt.i18n import tr

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_ambient_enabled(False)
    preferences.set_reduce_motion(True)
    preferences._set_flow_graphics_backend('cpu')
    monkeypatch.chdir(tmp_path)
    source = tmp_path / 'source.tif'
    _image(source)
    original = source.read_bytes()
    dispatched = []
    monkeypatch.setattr(webbrowser, 'open', lambda url, *args, **kwargs:
                        dispatched.append(('browser', str(url))) or True)
    monkeypatch.setattr(QDesktopServices, 'openUrl', lambda url:
                        dispatched.append(('desktop', url.toString())) or True)
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1100, 850)
    window.show()
    page = window._startup
    chosen = QSignalSpy(page.module_requested)
    screens = set(window._screens)
    try:
        page.path_edit.setText('Measure')
        qtbot.keyClick(page.path_edit, Qt.Key_Return)
        qtbot.waitUntil(lambda: bool(page._last_result) and any(
            card._key == 'measure' for card in page.field.cards), timeout=5000)
        result = page._last_result
        assert result['path'] == 'Measure'
        groups = result['context']['action_groups']
        assert groups and all(group['kind'] == 'navigation' for group in groups.values())
        assert not dispatched and chosen.count() == 0
        details = page.field.details
        link = details.document().find(tr('Protocols'))
        assert not link.isNull()
        link.setPosition(link.selectionStart() + 2)
        details.setTextCursor(link)
        details.ensureCursorVisible()
        qtbot.mouseClick(details.viewport(), Qt.LeftButton, pos=details.cursorRect(link).center())
        dialog = window._protocol_browser_dialog
        assert dialog.isVisible() and dialog._source_path == ''
        assert dialog._context == result['context'] and page._last_result is result
        assert not dispatched and window._stack.currentWidget() is page
        dialog.close()
        window.activateWindow()
        qtbot.waitUntil(window.isActiveWindow, timeout=2000)
        page.path_edit.setText('Cache')
        qtbot.keyClick(page.path_edit, Qt.Key_Return)
        cache = tmp_path / 'spacr-home' / 'cache'
        assert cache.is_dir()
        assert dispatched == [('browser', QUrl.fromLocalFile(str(cache.resolve())).toString())]
        assert chosen.count() == 0 and set(window._screens) == screens
        assert window._stack.currentWidget() is page and not page._runner.active_jobs()
        assert source.read_bytes() == original and not (tmp_path / 'Measure').exists()
    finally:
        window.close()
