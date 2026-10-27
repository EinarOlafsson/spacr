"""Resizing a live workflow never revives its faded suggestion cards."""
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


def test_workflow_resize_keeps_departed_cards_out_of_pixels_and_mouse_targets(
        qtbot, private_ai_profile, tmp_path):
    from spacr.qt.app import make_home_page

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_show_alpha(True)
    preferences.set_ambient_enabled(True)
    preferences.set_reduce_motion(False)
    page = make_home_page()
    qtbot.addWidget(page)
    page.resize(1400, 1000)
    page.show()
    actions = [('mask', 'Mask Generation', '', str(tmp_path)),
               ('make_masks', 'Make Masks', '', str(tmp_path)),
               ('measure', 'Measure', '', str(tmp_path))]
    try:
        page.field.set_cards(actions)
        for card in page.field.cards:
            card.activated.connect(page._select_suggestion)
        qtbot.waitUntil(lambda: all(not card._growing for card in page.field.cards), timeout=3000)
        page._last_result = {'actions': actions}
        selected, *departed = page.field.cards
        for card in departed:
            card.setStyleSheet('background: rgb(255, 0, 255); border: none;')
        page._select_suggestion('mask', str(tmp_path))
        graph = page._workflow
        qtbot.waitUntil(lambda: graph._progress == 1, timeout=2000)
        faded = [card._content_opacity.opacity() for card in departed]
        for size in ((1050, 820), (1450, 1000), (950, 900)):
            page.resize(*size)
            page.field._arrange()
            QApplication.processEvents()
            assert graph.central is selected and selected.isVisible()
            assert not page.path_edit.isVisible()
            assert all(card.isHidden() and not card.isVisible() for card in departed)
            assert [card._content_opacity.opacity() for card in departed] == faded
            assert not page.field._scrollbar.isVisible()
            frame = page.grab().toImage()
            for card in departed:
                point = card.mapToGlobal(card.rect().center())
                target = QApplication.widgetAt(point)
                assert target is not card and not (target is not None and card.isAncestorOf(target))
                local = page.mapFromGlobal(point)
                if page.rect().contains(local):
                    assert frame.pixelColor(local) != QColor(255, 0, 255)
        graph.back.click()
        assert page._workflow is None
        assert page.field._suggestions_active
        assert page.path_edit.isVisible()
        assert selected.parentWidget() is page.field
        assert all(card.isEnabled() for card in page.field.cards)
        assert any(card.isVisible() for card in page.field.cards)
        assert all(card._content_opacity.opacity() == 1 for card in page.field.cards)
        page._select_suggestion('mask', str(tmp_path))
        assert page._workflow.central is selected
    finally:
        page.close()
