"""Native task cards accept reviewed locale phrases and natural questions."""
import unicodedata

import pytest
from PySide6.QtCore import Qt

from spacr.qt import app
from spacr.qt.widgets.home import _EasyHomePage
from tests.qt.test_733_easy_commands import commands as commands
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile

_QUESTIONS = [
    ('sv', 'jag vill mäta cellegenskaper', ('measure',)),
    ('de', 'ich möchte Bilder analysieren', ('mask', 'measure', 'qc_dashboard')),
    ('es', 'quiero analizar imágenes', ('mask', 'measure', 'qc_dashboard')),
    ('pt', 'quero analisar imagens', ('mask', 'measure', 'qc_dashboard')),
    ('fr', 'je veux analyser des images', ('mask', 'measure', 'qc_dashboard')),
    ('zh_CN', '我想分析图像', ('mask', 'measure', 'qc_dashboard')),
    ('ko', '이미지 분석을 하고 싶어요', ('mask', 'measure', 'qc_dashboard')),
    ('hi', 'मैं चित्रों का विश्लेषण करना चाहता हूँ', ('mask', 'measure', 'qc_dashboard')),
    ('is', 'ég vil greina myndir', ('mask', 'measure', 'qc_dashboard')),
    ('zh_CN', '请帮我分析FASTQ条形码', ('map_barcodes',)),
    ('zh_CN', '我想分析Z堆栈', ('3d',)),
    ('zh_CN', 'tuxiang fenxi', ('mask', 'measure', 'qc_dashboard')),
]


@pytest.mark.parametrize('locale,question,expected', _QUESTIONS)
def test_natural_locale_questions_make_real_cards_on_enter(
        commands, qtbot, monkeypatch, locale, question, expected):
    monkeypatch.setenv('SPACR_LANGUAGE', locale)
    monkeypatch.setattr(app, 'app_is_visible', lambda key: True)
    page = _EasyHomePage(commands)
    commands.setCentralWidget(page)
    commands._startup = page
    commands._easy_navigation_suggestions = lambda value: app.MainWindow._easy_navigation_suggestions(commands, value)
    page.command_requested.connect(commands._open_easy_command)
    page.path_edit.setText(question)
    assert not page.field.cards and not commands.opened
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    qtbot.waitUntil(lambda: bool(page.field.cards), timeout=3000)
    assert tuple(card._key for card in page.field.cards) == expected
    assert not commands.opened and not commands.navigated


def test_unicode_normalization_preserves_hindi_marks_and_canonical_accents():
    text = 'चित्रों का विश्लेषण'
    assert app._easy_command_text(text) == text
    assert '\u094d' in app._easy_command_text('विश्लेषण')
    assert app._easy_command_text('MÄTNING!') == app._easy_command_text(unicodedata.normalize('NFD', 'mätning'))


@pytest.mark.parametrize('query', [
    '/actual/图像分析/image.npy', './चित्र विश्लेषण', r'C:\이미지 분석\file.tif',
    'चित्र विश्लेषण.npy', '图像分析.sql', 'prefiximage analysispostfix',
    '图', 'imagery analysispostfix',
])
def test_localized_tasks_do_not_capture_paths_or_english_word_fragments(query):
    assert app._easy_task_keys(query) == ()


def test_existing_relative_localized_folder_remains_file_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / '图像分析').mkdir()
    assert app._easy_task_keys('图像分析') == ()


@pytest.mark.parametrize('key', ['overview', 'parascape'])
def test_owned_extra_native_modules_have_canonical_card_metadata(key):
    rows = app._easy_module_search(key)
    assert [row[0] for row in rows] == [key]
    assert app._easy_module_matches(key) == (key,)
    assert rows[0][1] and rows[0][2] and rows[0][3] == ''


@pytest.mark.parametrize('word', ['saliency', 'gradcam', 'grad-cam', '显著性图', '중요도 맵'])
def test_model_attention_aliases_offer_activation(word):
    assert [row[0] for row in app._easy_module_search(word)] == ['activation']


def test_english_and_other_locale_inputs_remain_available_under_another_ui_language(monkeypatch):
    monkeypatch.setenv('SPACR_LANGUAGE', 'fr')
    assert app._easy_task_keys('analyze images') == ('mask', 'measure', 'qc_dashboard')
    assert app._easy_task_keys('我想分析图像') == ('mask', 'measure', 'qc_dashboard')
    assert app._easy_task_keys('मैं चित्रों का विश्लेषण करना चाहता हूँ') == ('mask', 'measure', 'qc_dashboard')


def test_extra_organism_aliases_preserve_species_and_parascape_maturity(monkeypatch):
    from spacr.qt import preferences

    monkeypatch.setattr(app, 'app_is_visible', lambda key: key == 'toxoplasma')
    monkeypatch.setattr(preferences, 'maturity_is_visible', lambda stage: stage != 'alpha')
    rows = app._easy_native_extra_rows()
    assert 'toxoplasma' in rows
    assert not {'virus', 'mammalian', 'parascape'}.intersection(rows)
    assert app._easy_module_search('人类细胞') == []
    assert app._easy_module_search('ParaScape') == []
