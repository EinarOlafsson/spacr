"""Native description links and child controls keep separate keyboard traversal."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


def test_details_controls_leave_links_to_the_browser_and_keep_keyboard_actions(
        qtbot, qapp, private_ai_profile, monkeypatch):
    from spacr.qt.tutorials import tutorial_url
    from spacr.qt.widgets.home import _EasyHomePage

    preferences, _, _ = private_ai_profile
    preferences.set_ambient_enabled(False)
    preferences.set_reduce_motion(True)
    preferences._set_flow_graphics_backend('cpu')
    opened = []
    monkeypatch.setattr(QDesktopServices, 'openUrl', lambda url: opened.append(url.toString()) or True)
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1100, 850)
    page.show()
    page.activateWindow()
    qtbot.waitUntil(page.isActiveWindow)
    details, grip = page.field.details, page.field._details_grip
    highlighted = []
    details.highlighted.connect(lambda url: highlighted.append(url.toString()))
    try:
        assert details.toHtml().count('<a ') >= 7 and opened == []
        details.setFocus(Qt.TabFocusReason)
        qtbot.keyClick(details, Qt.Key_Tab)
        qtbot.keyClick(details, Qt.Key_Return)
        assert opened == [tutorial_url('mask')]
        qtbot.keyClick(details, Qt.Key_Tab)
        qtbot.keyClick(details, Qt.Key_Return)
        assert opened == [tutorial_url('mask'), tutorial_url('measure')]
        qtbot.keyClick(details, Qt.Key_Backtab, Qt.ShiftModifier)
        qtbot.keyClick(details, Qt.Key_Return)
        assert opened == [tutorial_url('mask'), tutorial_url('measure'), tutorial_url('mask')]
        before = details.geometry()
        grip.setFocus(Qt.OtherFocusReason)
        qtbot.keyClick(grip, Qt.Key_Up)
        assert details.height() > before.height() and details.geometry().bottom() == before.bottom()
        assert grip.focusPolicy() == Qt.StrongFocus and grip.isVisible()
        for control in (details._collapse, grip):
            for key, modifiers in ((Qt.Key_Tab, Qt.NoModifier), (Qt.Key_Backtab, Qt.ShiftModifier)):
                control.setFocus(Qt.OtherFocusReason)
                qapp.processEvents()
                assert qapp.focusWidget() is control
                highlighted.clear()
                before_opened = list(opened)
                qtbot.keyClick(control, key, modifiers)
                qapp.processEvents()
                assert qapp.focusWidget() is not control, (type(control).__name__, highlighted)
                assert opened == before_opened
        assert page.path_edit.text() == '' and page._last_result is None
    finally:
        page.close()
