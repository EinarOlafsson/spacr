"""The custom Easy search surface owns its rounded idle and focused chrome."""

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QLineEdit

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


@pytest.mark.parametrize('theme', ['dark', 'light'])
def test_easy_search_keeps_only_its_rounded_surface_with_field_fade_enabled(
        qtbot, qapp, private_ai_profile, tmp_path, theme):
    from spacr.qt import theme as styling
    from spacr.qt.widgets import field_fade
    from spacr.qt.widgets.home import _EasyHomePage

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_field_fade_enabled(True)
    preferences.set_ambient_enabled(True)
    preferences.set_reduce_motion(False)
    previous_sheet = qapp.styleSheet()
    installed = field_fade.install_field_fade(qapp)
    page = None
    try:
        qapp.setStyleSheet(styling.stylesheet(theme))
        page = _EasyHomePage()
        qtbot.addWidget(page)
        ordinary = QLineEdit(page)
        ordinary.setGeometry(16, 16, 160, 36)
        page.resize(1000, 700)
        page.show()
        page.activateWindow()
        ordinary.setFocus()
        qtbot.waitUntil(ordinary.hasFocus)
        panel = page.path_edit.geometry()
        origin = page.path_edit.mapTo(page, QPoint())
        frames = {}
        for focused in (False, True):
            if focused:
                page.path_edit.setFocus()
                qtbot.waitUntil(page.path_edit.hasFocus)
                qtbot.mouseMove(page.path_edit, QPoint(18, 8))
            else:
                ordinary.setFocus()
                qtbot.waitUntil(lambda: not page.path_edit.hasFocus())
            image = page.grab().toImage()
            destination = tmp_path / f'easy-search-{theme}-{"focused" if focused else "idle"}.png'
            assert image.save(str(destination))
            print('EASY_FOCUS_FRAME', destination, flush=True)
            frames[focused] = image
        assert not field_fade.fades(page.path_edit)
        assert field_fade.fades(ordinary)
        assert page.field.rim.isVisible() and not hasattr(page.field.rim, '_timer')
        assert not page.path_edit.hasFrame()
        assert page.path_edit.rect().contains(page.path_edit.browse.geometry())
        for image in frames.values():
            ratio = image.devicePixelRatioF()
            reference = origin + QPoint(-5, 3)
            background = image.pixelColor(round(reference.x() * ratio), round(reference.y() * ratio))
            for offset in (QPoint(3, 3), QPoint(8, 3), QPoint(3, 8)):
                point = origin + offset
                assert image.pixelColor(round(point.x() * ratio), round(point.y() * ratio)) == background
            center = origin + QPoint(panel.width() // 2, panel.height() // 2)
            assert image.pixelColor(round(center.x() * ratio), round(center.y() * ratio)) != background
    finally:
        if page is not None:
            page.close()
        if installed:
            field_fade.uninstall_field_fade(qapp)
        qapp.setStyleSheet(previous_sheet)
