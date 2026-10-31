"""Standalone Navigator learning links keep local input and handoff ownership."""
import copy

import pytest
from PySide6.QtCore import QPoint, QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtTest import QSignalSpy

from tests.qt.test_733_easy_mode_home import (
    _bytes, _image, isolated_input_root as isolated_input_root,
    trace_navigator_native_input_jobs as trace_navigator_native_input_jobs,
)
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile

pytestmark = pytest.mark.qt


def test_standalone_learning_links_preserve_input_until_explicit_module_choice(
        qtbot, qapp, private_ai_profile, isolated_input_root, monkeypatch):
    from spacr.qt.i18n import tr
    from spacr.qt.tutorials import tutorial_url
    from spacr.qt.widgets.home import _EasyHomePage
    from tests.qt.test_739_navigator_protocols import _choose, _click_url

    preferences, _, _ = private_ai_profile
    preferences.set_ambient_enabled(False)
    preferences.set_reduce_motion(True)
    image = isolated_input_root / 'field.tif'
    _image(image)
    before = _bytes([image])
    entries = set(isolated_input_root.iterdir())
    opened_urls = []

    def external_url(url):
        opened_urls.append(QUrl(url))
        return True

    monkeypatch.setattr(QDesktopServices, 'openUrl', external_url)
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1000, 800)
    page.show()
    chosen = QSignalSpy(page.module_requested)
    try:
        assert page.window() is page and not callable(getattr(page, '_open_navigator_protocols', None))
        page.accept_paths([str(image)])
        qtbot.waitUntil(lambda: not page._runner.is_busy() and any(
            card._key == 'make_masks' for card in page.field.cards), timeout=5000)
        card = next(card for card in page.field.cards if card._key == 'make_masks')
        qtbot.mouseMove(card, card.rect().center())
        details = page.field.details
        expected = tutorial_url('make_masks')
        qtbot.waitUntil(lambda: 'spacr://sidebar' in details.toHtml() and 'spacr://protocols' not in details.toHtml(),
                        timeout=5000)
        result = copy.deepcopy(page._last_result)
        anchors = QSignalSpy(details.anchorClicked)
        assert opened_urls == [] and chosen.count() == 0

        def click_link(label, url):
            cursor = details.document().find(tr(label))
            assert not cursor.isNull()
            cursor.setPosition(cursor.selectionStart() + 2)
            details.setTextCursor(cursor)
            details.ensureCursorVisible()
            qapp.processEvents()
            point = details.cursorRect(cursor).center() + QPoint(1, 0)
            assert details.anchorAt(point) == url
            qtbot.mouseClick(details.viewport(), Qt.LeftButton, pos=point)

        click_link('Open Dock', 'spacr://sidebar')
        assert anchors.count() == 1 and anchors.at(0)[0].toString() == 'spacr://sidebar'
        assert opened_urls == [] and chosen.count() == 0
        assert page.path_edit.text() == str(image) and page._last_result == result
        assert _bytes([image]) == before
        qtbot.mouseClick(page.path_edit.protocols, Qt.LeftButton)
        _choose(qtbot, page._protocol_select, 'small_images')
        qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
        _choose(qtbot, page._protocol_select, 'tiff_yokogawa')
        qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
        dialog = page._protocol_browser_dialog
        qtbot.waitUntil(dialog.isVisible)
        _click_url(qtbot, dialog._text, 'spacr://protocol/route/make_masks')
        assert dialog._recipe_result['protocol']['active_start'] == 'make_masks'
        assert page._last_result == result and page.path_edit.text() == str(image)
        _click_url(qtbot, dialog._text, expected)
        assert anchors.count() == 1
        assert [url.toString() for url in opened_urls] == [expected]
        assert opened_urls[0].scheme() == 'https' and not opened_urls[0].isLocalFile()
        assert str(image) not in expected
        assert chosen.count() == 0 and page._last_result == result
        assert page.path_edit.text() == str(image) and _bytes([image]) == before
        dialog.close()
        card = next(card for card in page.field.cards if card._key == 'make_masks')
        card.click()
        assert page._workflow is not None and chosen.count() == 0
        page._workflow.central.click()
        assert chosen.count() == 1 and chosen.at(0) == ['make_masks', str(image)]
        assert _bytes([image]) == before and set(isolated_input_root.iterdir()) == entries
    finally:
        page.close()
