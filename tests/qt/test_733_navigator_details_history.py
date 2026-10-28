"""Description controls retain answers without moving the Navigator input."""
from PySide6.QtCore import Qt

from tests.qt.test_733_navigator_details_resize import page as page
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


def test_history_buttons_revisit_rich_content_and_replace_forward_branch(page, qtbot):
    details = page.field.details
    details.setHtml('<p><b>First description</b></p>')
    details.setPlainText('Second answer')
    qtbot.mouseClick(details._back, Qt.LeftButton)
    assert details.toPlainText() == 'First description'
    assert '<span style=" font-weight:700;">First description' in details.toHtml()
    assert details._forward.isEnabled()
    qtbot.mouseClick(details._forward, Qt.LeftButton)
    assert details.toPlainText() == 'Second answer'
    qtbot.mouseClick(details._back, Qt.LeftButton)
    details.setPlainText('New branch')
    assert not details._forward.isEnabled()
    assert details._back.geometry().right() < details._forward.geometry().left()
    assert details._forward.geometry().right() < details._collapse.geometry().left()
    assert details._collapse.geometry().right() >= details.width() - 8


def test_stream_updates_one_answer_without_interrupting_history_browsing(page):
    details = page.field.details
    details.setPlainText('Previous result')
    details.setPlainText('Thinking…')
    count = len(details._entries)
    details._start_reply(42)
    details.setHtml('<p>Partial</p>', stream=42)
    details._navigate(-1)
    details.setHtml('<p><b>Completed answer</b></p>', stream=42)
    assert details.toPlainText() == 'Previous result'
    assert len(details._entries) == count
    details._navigate(1)
    assert details.toPlainText() == 'Completed answer'
    page._reset_for_home()
    assert len(details._entries) == 1 and details._entry == 0
    assert not details._back.isEnabled() and not details._forward.isEnabled()
    assert 'File/folder:' in details.toPlainText()


def test_minimize_restore_keeps_content_height_and_search_anchor(page, qtbot):
    from tests.qt.test_733_navigator_fixed_layout import choices

    field, details = page.field, page.field.details
    card = choices(page, 3)[0]
    page._select_suggestion(card._key, card._path)
    assert details.parentWidget() is page._workflow
    details.setPlainText('Keep this result')
    original = details.geometry()
    search = field.panel.geometry()
    qtbot.mouseClick(details._collapse, Qt.LeftButton)
    assert details._minimized and not details.viewport().isVisible()
    assert not field._details_grip.isVisible()
    assert details.mask().isEmpty()
    assert details.width() == original.width()
    assert details.height() >= details._collapse.height() + 8
    assert details._collapse.isVisible()
    assert field.panel.geometry() == search
    qtbot.mouseClick(details._collapse, Qt.LeftButton)
    assert not details._minimized and details.viewport().isVisible()
    assert details.geometry() == original and details.mask().isEmpty()
    assert details.toPlainText() == 'Keep this result'
    page._show_suggestions()
    assert details.parentWidget() is field and field._details_grip.isVisible()


def test_thin_white_history_arrows_and_blue_expanded_toggle(page, qtbot):
    details = page.field.details

    def ink(button):
        image = button.icon().pixmap(20, 20).toImage()
        pixels = [(x, y, image.pixelColor(x, y)) for x in range(20) for y in range(20)
                  if image.pixelColor(x, y).alpha() > 20]
        return pixels

    for button in (details._back, details._forward):
        pixels = ink(button)
        assert pixels and all(color.red() == color.green() == color.blue() == 255
                              for _, _, color in pixels)
        assert sum(color.alpha() / 255 for _, _, color in pixels) < 20
    expanded = ink(details._collapse)
    assert all(color.blue() > color.red() for _, _, color in expanded)
    assert sum(y for x, y, _ in expanded if x < 8) / sum(x < 8 for x, _, _ in expanded) > 10
    qtbot.mouseClick(details._collapse, Qt.LeftButton)
    collapsed = ink(details._collapse)
    assert all(color.red() == color.green() == color.blue() == 255 for _, _, color in collapsed)
    assert sum(y for x, y, _ in collapsed if x < 8) / sum(x < 8 for x, _, _ in collapsed) < 9
    qtbot.mouseClick(details._collapse, Qt.LeftButton)
    assert details.viewport().isVisible()
    assert all(color.blue() > color.red() for _, _, color in ink(details._collapse))
