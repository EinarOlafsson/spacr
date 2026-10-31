"""Protocol arrivals preserve complete card masks, reachable choices and native input."""
from copy import deepcopy

import numpy as np
import tifffile
from PySide6.QtCore import QAbstractAnimation, QCoreApplication, QEvent, QPoint, QRect, Qt
from PySide6.QtGui import QColor, QRegion
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QPlainTextEdit

from tests.qt.test_739_navigator_protocols import _choose
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


def test_animated_tiff_protocol_cards_remain_whole_and_reachable_before_native_handoff(
        qtbot, qapp, private_ai_profile, monkeypatch, tmp_path, record_property, qtlog):
    from spacr.qt.app import MainWindow

    preferences, _, _ = private_ai_profile
    monkeypatch.setenv('SPACR_NO_GL', '1')
    preferences.set_easy_mode(True)
    preferences.set_show_alpha(True)
    preferences._set_show_alpha_features(True)
    preferences.set_show_beta(True)
    preferences.set_ambient_enabled(True)
    preferences.set_reduce_motion(False)
    preferences._set_flow_graphics_backend('cpu')
    source = tmp_path / 'source'
    source.mkdir()
    image = source / 'userW0025F0001T0001Z000C1.tif'
    pixels = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
    tifffile.imwrite(image, pixels)
    original = image.read_bytes()
    entries = set(source.iterdir())
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    with qtbot.waitExposed(window):
        window.show()
    from spacr.qt.widgets.loading_screen import LoadingScreen

    qtbot.waitUntil(lambda: not window.findChildren(LoadingScreen), timeout=10000)
    page = window._startup
    selected = QSignalSpy(page.module_requested)
    try:
        page.accept_paths([str(image)])
        qtbot.waitUntil(lambda: not page._runner.is_busy() and page._last_result is not None,
                        timeout=10000)
        inspected = deepcopy(page._last_result)
        for width, height, scale in ((1366, 768, 1.0), (1920, 1080, 1.5)):
            if page.path_edit.protocols.isChecked():
                qtbot.mouseClick(page.path_edit.protocols, Qt.LeftButton)
            preferences.set_font_scale(scale)
            preferences.apply_preferences_to_app()
            window.resize(width, height)
            qapp.processEvents()
            assert window.size().toTuple() == (width, height)
            header = QRect(page.path_edit.mapToGlobal(QPoint()), page.path_edit.size())
            qtbot.mouseClick(page.path_edit.protocols, Qt.LeftButton)
            while page._protocol_step:
                qtbot.mouseClick(page._protocol_back_button, Qt.LeftButton)
            _choose(qtbot, page._protocol_select, 'small_images')
            qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
            _choose(qtbot, page._protocol_select, 'tiff_other')
            _choose(qtbot, page._protocol_detail_combos['metadata_type'], 'cq1')
            qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
            dialog = window._protocol_browser_dialog
            qtbot.waitUntil(dialog.isVisible)
            assert dialog._recipe_result is not None
            assert not dialog._diagram_host.isVisible() and not dialog._choice_list.isVisible()
            dialog.close()
            window.activateWindow()
            qtbot.waitUntil(window.isActiveWindow)
            field = page.field
            assert field._motion_enabled()
            assert [card._key for card in field.cards] == ['foreign', 'mask', 'make_masks', 'align', 'overview']
            qtbot.waitUntil(lambda: all(not card._growing for card in field.cards)
                           and field.protocol_controls._arrival.state() == QAbstractAnimation.Stopped,
                           timeout=10000)
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            qapp.processEvents()
            assert not any('QItemSelectionModel: Selecting when no model has been set' in record.message
                           for record in qtlog.records)
            assert QRect(page.path_edit.mapToGlobal(QPoint()), page.path_edit.size()) == header
            reached = set()
            for offset in dict.fromkeys((field._scrollbar.value(), field._scrollbar.maximum(), 0)):
                field._scrollbar.setValue(offset)
                qapp.processEvents()
                for card in field.cards:
                    local = field._row_rect.translated(-card.x(), -card.y()).intersected(card.rect())
                    assert card.mask() == QRegion(local), (card._key, card.rect(), card.mask().boundingRect())
                    assert card._content_opacity.opacity() == 1.0
                    if local != card.rect():
                        continue
                    points = (QPoint(5, 5), QPoint(card.width() - 6, 5),
                              QPoint(5, card.height() - 6), QPoint(card.width() - 6, card.height() - 6))
                    hits = [window.childAt(card.mapTo(window, point)) for point in points]
                    if all(hit is card or (hit is not None and card.isAncestorOf(hit)) for hit in hits):
                        reached.add(card._key)
            assert reached == {card._key for card in field.cards}
            button = page.path_edit.protocols
            assert button.width() == button.height()
            qtbot.mouseMove(window, QPoint(4, 4))
            qapp.processEvents()
            rendered = button.grab().toImage()
            assert not any(rendered.pixelColor(x, y).rgb() in (QColor('#303030').rgb(), QColor('#363636').rgb())
                           for y in range(rendered.height()) for x in range(rendered.width())), (
                'the selected control shows a circle at rest; it is a hover and press affordance only')
            for state, colour in (('pressed', '#303030'), ('hover', '#363636')):
                if state == 'pressed':
                    button.setDown(True)
                else:
                    qtbot.mouseMove(button, button.rect().center())
                qapp.processEvents()
                rendered = button.grab().toImage()
                button.setDown(False)
                matches = [(x, y) for y in range(rendered.height()) for x in range(rendered.width())
                           if rendered.pixelColor(x, y).rgb() == QColor(colour).rgb()]
                assert matches
                left, right = min(x for x, _ in matches), max(x for x, _ in matches)
                top, bottom = min(y for _, y in matches), max(y for _, y in matches)
                assert abs((right - left) - (bottom - top)) <= 1
                assert abs(left + right - (rendered.width() - 1)) <= 1
                assert abs(top + bottom - (rendered.height() - 1)) <= 1
                assert rendered.pixelColor(0, 0).rgb() != QColor(colour).rgb()
                assert rendered.save(str(tmp_path / f'protocol-dot-{width}-{state}.png'))
            assert window.grab().save(str(tmp_path / f'protocol-cards-{width}.png'))
            record_property(f'capture_{width}', str(tmp_path / f'protocol-cards-{width}.png'))
            assert selected.count() == 0 and page._last_result == inspected
            assert image.read_bytes() == original and set(source.iterdir()) == entries

        previews = []

        def accept_preview(widget):
            with qtbot.waitExposed(widget):
                widget.show()
            text = widget.findChild(QPlainTextEdit, 'EasySettingsPreview')
            assert text is not None and text.isReadOnly()
            previews.append(text.toPlainText())
            buttons = widget.findChild(QDialogButtonBox)
            qtbot.mouseClick(buttons.button(QDialogButtonBox.Open), Qt.LeftButton)
            return widget.result()

        monkeypatch.setattr(QDialog, 'exec', accept_preview)
        field._scrollbar.setValue(0)
        qapp.processEvents()
        card = next(card for card in field.cards if card._key == 'mask')
        qtbot.mouseClick(card, Qt.LeftButton)
        qtbot.waitUntil(lambda: page._workflow is not None)
        qtbot.mouseClick(page._workflow.central, Qt.LeftButton)
        assert previews and 'cq1' in previews[0]
        assert selected.count() == 1 and selected.at(0) == ['mask', str(source)]
        screen = window._screens['mask']
        assert screen._settings_model.collect()['src'] == str(source)
        assert screen._settings_model.collect()['metadata_type'] == 'cq1'
        assert screen._thread is None
        assert image.read_bytes() == original and set(source.iterdir()) == entries
        np.testing.assert_array_equal(tifffile.imread(image), pixels)
    finally:
        window.close()
        preferences.set_font_scale(1.0)
        preferences.apply_preferences_to_app()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.processEvents()
        assert not any('QItemSelectionModel: Selecting when no model has been set' in record.message
                       for record in qtlog.records)
