"""Protocols choices stay explicit, responsive and bound to native input ownership."""
from copy import deepcopy

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QPlainTextEdit
from shiboken6 import isValid

from spacr.qt import app
from spacr.qt.widgets.home import _EasyHomePage
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


@pytest.fixture
def protocol_profile(private_ai_profile):
    """Choose the supported static CPU path before constructing a Home widget."""
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    preferences.set_ambient_enabled(False)
    return preferences


def _choose(qtbot, combo, key):
    """Select a real native popup item after it is laid out and visible."""
    index = combo.findData(key)
    assert index >= 0, key
    combo.showPopup()
    view = combo.view()
    qtbot.waitUntil(view.isVisible)
    view.doItemsLayout()
    item = combo.model().index(index, 0)
    view.scrollTo(item)
    qtbot.waitUntil(lambda: view.viewport().rect().contains(view.visualRect(item).center()))
    qtbot.mouseClick(view.viewport(), Qt.LeftButton, pos=view.visualRect(item).center())
    qtbot.waitUntil(lambda: not isValid(view) or not view.isVisible())


def _details(qtbot, page):
    """Complete each current required field using a real catalog option."""
    for _ in range(8):
        pending = [(field, page._protocol_detail_combos[field['name']])
                   for field in page._protocol_fields()
                   if field.get('required', True)
                   and page._protocol_detail_combos[field['name']].currentIndex() < 0]
        if not pending:
            return
        field, combo = pending[0]
        _choose(qtbot, combo, field['options'][0]['key'])
    pytest.fail('dependent protocol fields did not become valid')


def _finish(qtbot, page, family, format):
    """Choose a family, format and its genuine dependent inputs."""
    qtbot.mouseClick(page.path_edit.protocols, Qt.LeftButton)
    _choose(qtbot, page._protocol_select, family)
    qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
    _choose(qtbot, page._protocol_select, format)
    _details(qtbot, page)
    qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
    return page.window()._protocol_browser_dialog


def _click_url(qtbot, browser, url):
    """Click an actual rendered anchor, retaining native text layout and dispatch."""
    block = browser.document().begin()
    while block.isValid():
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            if fragment.isValid() and fragment.charFormat().anchorHref() == url:
                cursor = QTextCursor(browser.document())
                cursor.setPosition(fragment.position() + min(1, len(fragment.text()) - 1))
                browser.setTextCursor(cursor)
                browser.ensureCursorVisible()
                point = browser.cursorRect(cursor).center() + QPoint(1, 0)
                assert browser.anchorAt(point) == url
                qtbot.mouseClick(browser.viewport(), Qt.LeftButton, pos=point)
                return
            iterator += 1
        block = block.next()
    pytest.fail('missing rendered protocol anchor: ' + url)


@pytest.mark.parametrize('family,format', [
    ('small_images', 'pdf'), ('plate_images', 'pdf'),
    ('features', 'database'), ('sequencing', 'fastq_gz'), ('sequencing', 'other'),
])
def test_three_step_native_choices_keep_back_state_and_never_start_analysis(
        qtbot, protocol_profile, family, format):
    """All four families, both PDF lanes and unavailable formats use one chooser."""
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1000, 800)
    page.show()
    qtbot.waitExposed(page)
    from PySide6.QtTest import QSignalSpy

    activated_text = QSignalSpy(page._protocol_select.textActivated)
    search_geometry = page.path_edit.geometry()
    intro = page.field.details.toPlainText()
    assert intro.index('spaCR AI:') < intro.index('Protocols:') < intro.index('File/folder:')
    assert 'spacr://protocols' not in page.field.details.toHtml()
    assert page.path_edit.protocols.width() == page.path_edit.ai.width() == 36
    assert page.path_edit.protocols.x() < page.path_edit.ai.x()
    qtbot.mouseClick(page.path_edit.protocols, Qt.LeftButton)
    assert page.path_edit.geometry() == search_geometry
    page._protocol_select.showPopup()
    inline = page._protocol_select.view()
    assert inline.window() is page.window()
    qtbot.keyClick(inline, Qt.Key_Escape)
    assert not inline.isVisible() and page.path_edit.protocols.isChecked()
    assert not page._protocol_next_button.isEnabled()
    assert not page._protocol_back_button.isEnabled()
    _choose(qtbot, page._protocol_select, family)
    assert activated_text.count() == 1
    assert activated_text.at(0)[0] == page._protocol_select.currentText()
    assert page._protocol_next_button.isEnabled()
    qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
    assert not page._protocol_next_button.isEnabled()
    _choose(qtbot, page._protocol_select, format)
    _details(qtbot, page)
    if family == 'features':
        _choose(qtbot, page._protocol_detail_combos['experiment'], 'other')
        assert 'barcodes' not in page._protocol_choices
        assert not page._protocol_next_button.isEnabled()
        _details(qtbot, page)
    qtbot.waitUntil(lambda: all(combo.isVisible() and combo.height() >= 28
                                 for combo in page._protocol_detail_combos.values()))
    assert all(page.field.protocol_controls.rect().contains(combo.parentWidget().geometry())
               for combo in page._protocol_detail_combos.values())
    assert page.path_edit.geometry() == search_geometry
    retained = deepcopy(page._protocol_choices)
    qtbot.mouseClick(page._protocol_back_button, Qt.LeftButton)
    assert page._protocol_select.currentData() == family
    qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
    assert page._protocol_select.currentData() == format
    assert page._protocol_choices == retained
    qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
    dialog = page._protocol_browser_dialog
    qtbot.waitUntil(dialog.isVisible)
    assert page.path_edit.geometry() == search_geometry
    assert page._protocol_step == 2 and not page._protocol_next_button.isEnabled()
    assert page._last_result is None
    assert page._protocol_result['source_path'] == ''
    assert page.field.protocol_controls.width() == round(page.path_edit.width() * 0.75)
    assert page.field.protocol_controls.y() > page.path_edit.geometry().bottom()
    assert all(card.geometry().bottom() < page.path_edit.y() for card in page.field.cards)
    if format == 'other':
        assert not page.field.cards
        assert page._protocol_result['requirements']
        assert page._protocol_result['requirements'][0] in page.field.details.toPlainText()
    assert not page.grab().isNull()
    dialog.close()
    page.activateWindow()
    qtbot.waitUntil(page.isActiveWindow)
    page.path_edit.setFocus()
    qtbot.keyClick(page.path_edit, Qt.Key_Escape)
    assert page.path_edit.geometry() == search_geometry
    assert not page.path_edit.protocols.isChecked()
    qtbot.mouseClick(page.path_edit.protocols, Qt.LeftButton)
    assert page._protocol_step == 2 and page._protocol_choices == retained
    assert not dialog.isVisible()
    assert not page._runner.is_busy()


@pytest.mark.parametrize('accept', [False, True])
def test_protocol_metadata_uses_real_tiff_native_handoff_and_preserves_inspection(
        qtbot, protocol_profile, monkeypatch, tmp_path, accept):
    """An explicit card previews CQ1 settings before the real Mask file adapter."""
    image = tmp_path / 'userW0025F0001T0001Z000C1.tif'
    pixels = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
    tifffile.imwrite(image, pixels)
    original_bytes = image.read_bytes()
    window = app.MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1200, 850)
    window.show()
    page = window._startup
    page.accept_paths([str(image)])
    qtbot.waitUntil(lambda: not page._runner.is_busy() and page._last_result is not None,
                    timeout=10000)
    inspected = deepcopy(page._last_result)
    overrides = deepcopy(page._settings_overrides)
    qtbot.mouseClick(page.path_edit.protocols, Qt.LeftButton)
    _choose(qtbot, page._protocol_select, 'small_images')
    qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
    _choose(qtbot, page._protocol_select, 'tiff_other')
    _choose(qtbot, page._protocol_detail_combos['metadata_type'], 'cq1')
    if accept:
        active = page._protocol_detail_combos['metadata_type']
        page.accept_paths([str(image)])
        active.showPopup()
        assert active.view().isVisible()
        qtbot.waitUntil(lambda: not page._runner.is_busy() and page._last_result is not None,
                        timeout=10000)
        qtbot.waitUntil(lambda: not isValid(active))
        assert page.field.protocol_controls._inline_choice is None
        qtbot.mouseClick(page.path_edit.protocols, Qt.LeftButton)
        assert not page.path_edit.protocols.isChecked()
        qtbot.mouseClick(page.path_edit.protocols, Qt.LeftButton)
        assert page.path_edit.protocols.isChecked()
        assert page._protocol_choices['metadata_type'] == 'cq1'
        assert page.field.protocol_controls._inline_choice is None
    qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
    dialog = window._protocol_browser_dialog
    qtbot.waitUntil(dialog.isVisible)
    dialog.close()
    assert page._last_result == inspected and page._settings_overrides == overrides
    screens = dict(window._screens)
    seen = []

    def preview(widget):
        widget.show()
        qtbot.waitExposed(widget)
        text = widget.findChild(QPlainTextEdit, 'EasySettingsPreview')
        assert text is not None and text.isReadOnly()
        seen.append(text.toPlainText())
        buttons = widget.findChild(QDialogButtonBox)
        qtbot.mouseClick(buttons.button(QDialogButtonBox.Open if accept else QDialogButtonBox.Cancel),
                         Qt.LeftButton)
        return widget.result()

    monkeypatch.setattr(QDialog, 'exec', preview)
    card = next(card for card in page.field.cards if card._key == 'mask')
    qtbot.mouseClick(card, Qt.LeftButton)
    qtbot.waitUntil(lambda: page._workflow is not None)
    assert window._screens == screens
    qtbot.mouseClick(page._workflow.central, Qt.LeftButton)
    assert seen and 'cq1' in seen[0]
    assert image.read_bytes() == original_bytes
    np.testing.assert_array_equal(tifffile.imread(image), pixels)
    assert page._last_result == inspected and page._settings_overrides == overrides
    if accept:
        screen = window._screens['mask']
        settings = screen._settings_model.collect()
        assert settings['metadata_type'] == 'cq1'
        assert settings['src'] == str(tmp_path)
        assert screen._thread is None
    else:
        assert window._screens == screens
        assert window._stack.currentWidget() is page


def test_open_recipe_cannot_carry_an_earlier_inspection_into_a_replacement_path(
        qtbot, protocol_profile, tmp_path):
    """A visible old recipe refuses its native anchor after a real source edit."""
    images = [tmp_path / name for name in ('first.tif', 'replacement.tif')]
    for number, image in enumerate(images):
        tifffile.imwrite(image, np.full((16, 16), number, dtype=np.uint16))
    before = [image.read_bytes() for image in images]
    window = app.MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.show()
    page = window._startup
    page.accept_paths([str(images[0])])
    qtbot.waitUntil(lambda: not page._runner.is_busy() and page._last_result is not None,
                    timeout=10000)
    dialog = _finish(qtbot, page, 'small_images', 'tiff_yokogawa')
    qtbot.waitUntil(dialog.isVisible)
    screens = dict(window._screens)
    page.path_edit.setFocus()
    qtbot.keyClick(page.path_edit, Qt.Key_A, Qt.ControlModifier)
    qtbot.keyClicks(page.path_edit, str(images[1]))
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    qtbot.waitUntil(lambda: not page._runner.is_busy() and page._last_result is not None
                    and page._last_result['path'] == str(images[1]), timeout=10000)
    _click_url(qtbot, dialog._text, 'spacr://module/mask')
    assert window._screens == screens
    assert 'inspected input changed' in window.statusBar().currentMessage()
    assert [image.read_bytes() for image in images] == before
    assert page.path_edit.text() == str(images[1])
    dialog.close()


def test_single_dock_link_opens_real_modules_without_changing_mode(qtbot, protocol_profile, tmp_path):
    from spacr.qt.app import MainWindow
    from tests.qt.test_739_navigator_protocols import _click_url

    preferences = protocol_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    preferences.set_ambient_enabled(False)
    preferences._settings().setValue('prefs/navigator_dock_collapsed', True)
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1366, 768)
    window.show()
    page = window._startup
    split = page._navigator_aside_split
    split.set_collapsed('Widgets', True, by_user=False)
    qtbot.waitUntil(lambda: page.field.width() > 1000)
    assert window._dock_edge.is_collapsed()
    assert not window._dock_slot.isVisible()
    from spacr.qt.i18n import tr

    assert tr('Open Dock') in page.field.details.toPlainText()
    assert 'spacr://sidebar' in page.field.details.toHtml()
    anchor = QRect(page.path_edit.mapToGlobal(QPoint()), page.path_edit.size())
    aside_state = (QRect(page._navigator_aside.geometry()), split.mask())
    text = page.path_edit.text()
    result = page._last_result
    opened = QSignalSpy(page.module_requested)
    toggles = QSignalSpy(window._dock_edge.collapsedChanged)
    try:
        for _ in range(2):
            _click_url(qtbot, page.field.details, 'spacr://sidebar')
            qtbot.waitUntil(lambda: window._dock_slot.isVisible())
            assert not window._dock_edge.is_collapsed()
            assert window._sidebar.isVisible()
            assert window._dock_slot.width() >= window._sidebar.width() > 0
            assert split.is_collapsed('Widgets')
            assert (QRect(page._navigator_aside.geometry()), split.mask()) == aside_state
            assert QRect(page.path_edit.mapToGlobal(QPoint()), page.path_edit.size()) == anchor
            assert window._stack.geometry() == window.centralWidget().rect()
            assert page.path_edit.text() == text and page._last_result is result
            assert opened.count() == 0
        assert toggles.count() == 1 and toggles.at(0) == [False]
        assert protocol_profile.get_easy_mode()
        assert window._stack.currentWidget() is page
        assert page._last_result is None
        from PySide6.QtGui import QColor
        from spacr.qt.theme import font_px, palette_for

        for scale in (1.0, 1.5):
            preferences.set_font_scale(scale)
            preferences.apply_preferences_to_app()
            page.refresh()
            from PySide6.QtWidgets import QApplication

            QApplication.processEvents()
            qtbot.waitUntil(lambda: page.path_edit.font().pixelSize() == font_px(16))
            qtbot.waitUntil(lambda: page.path_edit.protocols.size() == page.path_edit.ai.size())
            button = page.path_edit.protocols
            assert button.geometry().center().y() == page.path_edit.ai.geometry().center().y()
            assert button.geometry().center().y() == page.path_edit.rect().center().y()
            assert page._protocol_select.font().pixelSize() == font_px(16)
            assert page.field.details.font().pixelSize() == font_px(16)
            qtbot.mouseMove(button, button.rect().center())
            assert window.grab().save(str(tmp_path / f'{scale}-p-hover.png'))
            current_anchor = QRect(page.path_edit.mapToGlobal(QPoint()), page.path_edit.size())
            qtbot.mouseClick(button, Qt.LeftButton)
            qtbot.mouseMove(window, QPoint(2, 2))
            qtbot.waitUntil(page.field.protocol_controls.isVisible)
            assert QRect(page.path_edit.mapToGlobal(QPoint()), page.path_edit.size()) == current_anchor
            fragment = page.field.details.document().begin().begin().fragment()
            assert fragment.charFormat().foreground().color() == QColor(palette_for('dark')['fg'])
            assert not fragment.charFormat().isAnchor()
            assert window.grab().save(str(tmp_path / f'{scale}-protocols.png'))
            qtbot.mouseClick(button, Qt.LeftButton)
            cursor = page.field.details.document().find(tr('Open Dock'))
            assert not cursor.isNull()
            cursor.setPosition(cursor.selectionStart() + 1)
            page.field.details.setTextCursor(cursor)
            page.field.details.ensureCursorVisible()
            point = page.field.details.cursorRect(cursor).center() + QPoint(1, 0)
            assert page.field.details.anchorAt(point) == 'spacr://sidebar'
            assert cursor.charFormat().foreground().color() == QColor(palette_for('dark')['fg'])
            qtbot.mouseMove(page.field.details.viewport(), QPoint(2, 2))
            qtbot.mouseMove(page.field.details.viewport(), point)
            qtbot.waitUntil(lambda: len(page.field.details.extraSelections()) == 1)
            assert page.field.details.extraSelections()[0].cursor.selectedText() == tr('Open Dock')
            assert window.grab().save(str(tmp_path / f'{scale}-dock-hover.png'))
            qtbot.mouseClick(page.field.details.viewport(), Qt.LeftButton, pos=point)
            assert not window._dock_edge.is_collapsed()
            assert QRect(page.path_edit.mapToGlobal(QPoint()), page.path_edit.size()) == current_anchor
            assert opened.count() == 0 and page._last_result is None
    finally:
        window.close()


def test_protocols_retires_a_real_native_ai_stream_before_showing_local_choices(
        qtbot, protocol_profile, monkeypatch):
    """Late provider chunks cannot replace the explicitly selected local workflow."""
    import threading
    from spacr.qt.ai import providers, worker

    entered, release = threading.Event(), threading.Event()
    created = []

    class OfflineProvider:
        def is_configured(self):
            return True

        def stream_chat(self, messages, system='', model=None):
            entered.set()
            assert release.wait(3)
            yield 'Late provider answer'

        def cancel_stream(self):
            release.set()

    monkeypatch.setattr(providers, 'get_provider', lambda name: OfflineProvider())
    monkeypatch.setattr(worker, 'question_cache_key', lambda messages: '')
    real_make = worker.make_stream_thread

    def capture(*args, **kwargs):
        pair = real_make(*args, **kwargs)
        created.append(pair)
        return pair

    monkeypatch.setattr(worker, 'make_stream_thread', capture)
    protocol_profile._settings().setValue('home/navigator_ai_provider', 'spacr_hosted')
    protocol_profile.set_preferred_provider('spacr_hosted')
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1000, 800)
    page.show()
    try:
        qtbot.mouseClick(page.path_edit.ai, Qt.LeftButton)
        page.path_edit.setText('Explain an unfamiliar imaging experiment')
        qtbot.keyClick(page.path_edit, Qt.Key_Return)
        qtbot.waitUntil(entered.is_set)
        assert page._ai_thread is created[0][0]
        qtbot.mouseClick(page.path_edit.protocols, Qt.LeftButton)
        assert page._ai_thread is None and not page.path_edit.ai._busy
        _choose(qtbot, page._protocol_select, 'features')
        explanation = page.field.details.toPlainText()
        assert explanation and 'Late provider answer' not in explanation
        thread, _work = created[0]
        qtbot.waitUntil(lambda: not isValid(thread) or not thread.isRunning(), timeout=5000)
        qtbot.wait(60)
        assert page.field.details.toPlainText() == explanation
        assert page._protocol_next_button.isEnabled()
        assert page._last_result is None
    finally:
        release.set()
        for thread, work in created:
            if isValid(thread) and thread.isRunning():
                work.cancel()
                thread.quit()
                assert thread.wait(3000)


def test_downstream_recipe_clears_cached_raw_source_and_fits_visible_sidebar(
        qtbot, protocol_profile, monkeypatch, tmp_path):
    """An explicit later step awaits its own outputs instead of reusing raw input."""
    image = tmp_path / 'userW0025F0001T0001Z000C1.tif'
    pixels = np.arange(24 * 24, dtype=np.uint16).reshape(24, 24)
    tifffile.imwrite(image, pixels)
    before = image.read_bytes()
    unrelated = tmp_path / 'previous_measurement_source'
    unrelated.mkdir()
    window = app.MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1366, 768)
    window.show()
    window.open_module('measure')
    screen = window._screens['measure']
    screen.apply_settings_dict({'src': str(unrelated)})
    assert screen._settings_model.collect()['src'] == str(unrelated)
    window.open_module('__home__')
    page = window._startup
    page.accept_paths([str(image)])
    qtbot.waitUntil(lambda: not page._runner.is_busy() and page._last_result is not None,
                    timeout=10000)
    inspected = deepcopy(page._last_result)
    overrides = deepcopy(page._settings_overrides)
    qtbot.mouseClick(page.path_edit.protocols, Qt.LeftButton)
    _choose(qtbot, page._protocol_select, 'small_images')
    qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
    _choose(qtbot, page._protocol_select, 'other_metadata')
    _choose(qtbot, page._protocol_detail_combos['image_format'], 'png')
    from spacr.qt.i18n import tr

    explanation = tr('Inspect a representative file and verify reader availability and axes before conversion.')
    assert page.field.details.toPlainText().count(explanation) == 1
    _choose(qtbot, page._protocol_select, 'tiff_yokogawa')
    _details(qtbot, page)
    qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
    dialog = window._protocol_browser_dialog
    qtbot.waitUntil(dialog.isVisible)
    assert 'measure' not in {row[0] for row in page._protocol_result['actions']}
    assert any(step['module'] == 'measure' for step in page._protocol_result['protocol']['steps'])
    page._show_navigator_sidebar()
    qtbot.waitUntil(lambda: not page._navigator_aside_split.is_collapsed('Widgets'))
    qtbot.waitUntil(lambda: page.field._protocol_details_width is not None)
    details_right = page.field.details.mapToGlobal(page.field.details.rect().topRight()).x()
    sidebar_left = page._navigator_aside.mapToGlobal(QPoint(0, 0)).x()
    assert details_right < sidebar_left
    seen = []

    def preview(widget):
        widget.show()
        qtbot.waitExposed(widget)
        text = widget.findChild(QPlainTextEdit, 'EasySettingsPreview')
        assert text is not None and text.isReadOnly()
        seen.append(text.toPlainText())
        assert str(image) not in seen[-1] and str(unrelated) not in seen[-1]
        buttons = widget.findChild(QDialogButtonBox)
        qtbot.mouseClick(buttons.button(QDialogButtonBox.Open), Qt.LeftButton)
        return widget.result()

    monkeypatch.setattr(QDialog, 'exec', preview)
    _click_url(qtbot, dialog._text, 'spacr://module/measure')
    assert seen
    assert window._stack.currentWidget() is screen
    assert screen._settings_model.collect()['src'] is None
    assert screen._thread is None
    assert page._last_result == inspected and page._settings_overrides == overrides
    assert page.path_edit.text() == str(image)
    assert image.read_bytes() == before
    np.testing.assert_array_equal(tifffile.imread(image), pixels)
    dialog.close()
    window.open_module('__home__')
    qtbot.mouseClick(page.path_edit.protocols, Qt.LeftButton)
    qtbot.waitUntil(lambda: page.field._protocol_details_width is None)
    assert page.field.details.width() == min(1560, max(180, page.field.width() - 32))
