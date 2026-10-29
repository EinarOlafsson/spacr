"""Unmatched tasks explain local assistance; the literary answer never spends AI quota."""
from PySide6.QtCore import Qt
import pytest

from tests.qt.test_733_navigator_ai import ai_page as ai_page
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


@pytest.mark.parametrize('value', [
    '42', 'THE ANSWER TO LIFE!', 'meaning of life', 'What is the meaning of 42?',
    'What does 42 mean?', 'Life, the universe and everything',
    'meningen med livet', 'Was bedeutet 42?', '¿Qué significa 42?',
    'o sentido da vida', 'Que signifie 42 ?', '42是什么意思？',
    '42 का क्या मतलब है?', '42는 무슨 뜻인가요?', 'Hvað þýðir 42?',
])
@pytest.mark.parametrize('enabled', [False, True])
def test_literal_and_localized_life_tasks_answer_locally_on_enter(ai_page, qtbot, value, enabled):
    page, calls, _drained = ai_page
    page.path_edit.ai.blockSignals(True)
    page.path_edit.ai.setChecked(enabled)
    page.path_edit.ai.blockSignals(False)
    commands = []
    page.command_requested.connect(commands.append)
    page.path_edit.setText(value)
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    text = page.field.details.toPlainText()
    assert '42 — the answer to life, the universe and everything.' in text
    assert 'The Hitchhiker’s Guide to the Galaxy' in text
    assert not commands and not calls
    assert not page._runner.is_busy()


def test_unmatched_nonpath_task_explains_provider_choice_without_ai(ai_page, qtbot):
    page, calls, _drained = ai_page
    page.path_edit.setText('Why is my experiment mysteriously upside down?')
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    text = page.field.details.toPlainText()
    assert text.startswith('No matching tool found.')
    assert 'Right-click ◯' in text and 'Left-click ◯' in text
    assert all(name in text for name in ('spaCR AI', 'Cloudflare', 'Claude', 'Codex', 'Gemini'))
    assert all(example in text for example in ('segment images', 'Measure', 'tif'))
    assert 'Choose an existing local file or folder.' not in text
    assert not calls and not page._runner.is_busy()


@pytest.mark.parametrize('value', ['42 images', '42 channels', '/missing/42', '42.tif', './42', 'life/42'])
def test_numeric_measurements_and_paths_are_not_easter_egg_tasks(value):
    from spacr.qt.widgets.home import _EasyHomePage

    assert not _EasyHomePage._navigator_local_reply(value)


def test_existing_numeric_file_keeps_native_input_priority(ai_page, qtbot, tmp_path, monkeypatch):
    page, calls, _drained = ai_page
    path = tmp_path / '42'
    path.write_text('not an image')
    monkeypatch.chdir(tmp_path)
    commands = []
    page.command_requested.connect(commands.append)
    page.path_edit.setText('42')
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    assert commands == ['42']
    assert 'Hitchhiker' not in page.field.details.toPlainText()
    assert not calls


def test_nonexistent_explicit_path_still_uses_local_file_inspection(ai_page, monkeypatch, tmp_path):
    page, calls, _drained = ai_page
    value = str(tmp_path / 'missing42.tif')
    submitted = []
    monkeypatch.setattr(page._runner, 'submit', lambda action, **kwargs: submitted.append(action()))
    page.path_edit.setText(value)
    page._inspect()
    assert submitted[0]['message'] == 'Choose an existing local file or folder.'
    assert submitted[0]['actions'] == []
    assert not calls


def test_explicit_numeric_file_drop_never_becomes_an_ai_or_literary_task(ai_page, monkeypatch, tmp_path):
    page, calls, _drained = ai_page
    source = tmp_path / '42'
    source.write_bytes(b'numeric filename, not a question')
    before = source.read_bytes()
    results = []
    monkeypatch.setattr(page._runner, 'submit', lambda action, **kwargs: results.append(action()))
    page.path_edit.ai.blockSignals(True)
    page.path_edit.ai.setChecked(True)
    page.path_edit.ai.blockSignals(False)
    page.accept_paths([str(source)])
    assert results and results[0]['path'] == str(source)
    assert 'Hitchhiker' not in page.field.details.toPlainText()
    assert not calls
    assert source.read_bytes() == before
