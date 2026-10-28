"""The approved lower-panel guide yields to actual input, descriptions and answers."""
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


@pytest.fixture
def page(qtbot, private_ai_profile):
    from spacr.qt.widgets.home import _EasyHomePage
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    value = _EasyHomePage()
    qtbot.addWidget(value)
    value.resize(1100, 900)
    value.show()
    yield value
    value.close()


def test_empty_home_shows_six_approved_bold_examples_without_moving_controls(page):
    text = page.field.details.toPlainText()
    assert page._guide_visible and not page._details_has_content
    assert all(fragment in text for fragment in (
        'File/folder:', '/data/experiment/plate1', '/data/experiment/plate1/image.tif',
        'file type, such as tif', '“segment images”', '“Measure”',
        'image based modules', 'compatible modules', 'Click once to open the module network.',
        'Click twice to open the module.', 'Ask for help using spaCR', 'How do I generate object masks?', 'How do i train a Cellpose or Torch model?'))
    document = page.field.details.document()
    for label in ('File/folder', 'Path', 'Question', 'Module', 'Explore', 'spaCR AI'):
        assert document.find(label + ':').charFormat().fontWeight() == QFont.Bold
        assert document.find(label + ':').charFormat().foreground().color().name() == "#999999"
    assert not page.field.cards and not page._debounce.isActive()
    before = page.path_edit.geometry()
    page._show_empty_guide()
    assert page.path_edit.geometry() == before


def test_typing_removes_guide_immediately_and_empty_clear_restores_it(page, qtbot, monkeypatch):
    monkeypatch.setattr(page._runner, 'submit', lambda *a, **k: pytest.fail('Typing submitted work'))
    page.path_edit.setFocus()
    qtbot.keyClicks(page.path_edit, 'unfamiliar task')
    assert not page._guide_visible and not page.field.details.toPlainText()
    page.path_edit.selectAll()
    qtbot.keyClick(page.path_edit, Qt.Key_Backspace)
    assert page._guide_visible and 'File/folder:' in page.field.details.toPlainText()


def test_drop_and_actual_description_replace_intro(page, qtbot, monkeypatch):
    from spacr.qt.widgets import home
    monkeypatch.setattr(home, '_easy_inspect_path', lambda path: dict(
        path=path, actions=[], context={}, message='Image inspected'))
    page.accept_paths(['/local/image.tif'])
    assert not page._guide_visible
    qtbot.waitUntil(lambda: page.field.details.toPlainText() == 'Image inspected')
    assert 'File/folder:' not in page.field.details.toPlainText()
    page.field.details.setHtml('<p>Real module description</p>')
    assert page._details_has_content
    page._show_empty_guide()
    assert page.field.details.toPlainText() == 'Real module description'


def test_actual_ai_answer_survives_clearing_input_until_fresh_home(page, qtbot):
    page.path_edit.setText('Question awaiting a response')
    page._ai_text = 'Actual analysis advice, retained for reference.'
    page._render_ai_reply()
    assert page._details_has_content and not page._guide_visible
    page.path_edit.selectAll()
    qtbot.keyClick(page.path_edit, Qt.Key_Backspace)
    assert page.field.details.toPlainText() == 'Actual analysis advice, retained for reference.'
    assert not page._guide_visible
    page._reset_for_home()
    assert page._guide_visible and 'spaCR AI:' in page.field.details.toPlainText()
    assert not page._details_has_content
