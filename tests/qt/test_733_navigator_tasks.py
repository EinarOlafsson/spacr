"""Reviewed Navigator task phrases produce native suggestion cards on Enter."""
import re
from pathlib import Path

import pytest
from PySide6.QtCore import Qt

from spacr.qt import app
from spacr.qt.widgets.home import _EasyHomePage
from tests.qt.test_733_easy_commands import commands
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


_ITEM = Path(__file__).resolve().parents[2] / 'features/new/733_easy_mode_content_guided_home.txt'
_PAIRS = [(original, alternate, tuple(keys.split(', ')))
          for original, alternate, keys in re.findall(
              r'^\d{3}\. (.+) \| (.+) -> (.+)$', _ITEM.read_text(), re.M)]
_CASES = [(original, keys) for original, _alternate, keys in _PAIRS]
_ALTERNATES = [(alternate, keys) for _original, alternate, keys in _PAIRS]
assert len(_CASES) == len(_ALTERNATES) == 100


@pytest.mark.parametrize('phrase,expected', _CASES)
def test_reviewed_tasks_offer_cards_without_opening_modules(commands, qtbot, monkeypatch, phrase, expected):
    monkeypatch.setattr(app, 'app_is_visible', lambda key: True)
    page = _EasyHomePage(commands)
    commands.setCentralWidget(page)
    commands._easy_navigation_suggestions = lambda value: app.MainWindow._easy_navigation_suggestions(commands, value)
    page.command_requested.connect(commands._open_easy_command)
    emitted = []
    page.command_requested.connect(emitted.append)
    query = 'Can you help me ' + phrase.upper() + '?'
    rows = commands._easy_navigation_suggestions(query)
    assert tuple(row[0] for row in rows) == expected
    assert len(rows) == len({row[0] for row in rows})
    assert all(row[1] and row[2] and not row[3] for row in rows)
    page.path_edit.setText(query)
    assert not page.field.cards
    assert not commands.opened
    page._submit()
    assert tuple(card._key for card in page.field.cards) == expected
    assert not commands.opened and not commands.navigated and not emitted
    assert all(not row[0].startswith('__easy_command__:') for row in rows)


@pytest.mark.parametrize('value', [
    '/actual/image analysis/image.tif', r'C:\images\object detection.tif',
    './mask curation', '~/make a graph', 'read a database.sqlite',
    'image analysis.tif', 'analyze spatial imaging.npy',
])
def test_task_phrases_never_override_explicit_file_paths(value):
    assert not app._easy_task_keys(value)
    assert not app._easy_module_search(value)


def test_task_cards_keep_maturity_filter_and_existing_module_partial_search(commands, qtbot, monkeypatch):
    page = _EasyHomePage(commands)
    commands.setCentralWidget(page)
    commands._easy_navigation_suggestions = lambda value: app.MainWindow._easy_navigation_suggestions(commands, value)
    monkeypatch.setattr(app, 'app_is_visible', lambda key: key == 'make_masks')
    page.path_edit.setText('OBJECT DETECTION!')
    page._submit()
    assert [card._key for card in page.field.cards] == ['make_masks']
    assert not commands.opened
    assert 'measure' in {row[0] for row in app._easy_module_search('MeAsUr')}
    assert 'db_browser' in {row[0] for row in app._easy_module_search('DaTaBaSe')}
    assert page.path_edit.placeholderText() == 'Drop a file or folder | Type a Path | Ask a Question ◉'


@pytest.mark.parametrize('phrase,expected', [
    ('image data', ('mask', 'measure', 'qc_dashboard')),
    ('make figure', ('graph_builder',)),
    ('MAKE GRAPH!', ('graph_builder',)),
    ('high content image', ('mask', 'measure', 'qc_dashboard')),
])
def test_short_forms_and_partial_tasks_use_native_cards(commands, phrase, expected):
    rows = app.MainWindow._easy_navigation_suggestions(commands, phrase)
    assert tuple(row[0] for row in rows) == expected
    assert not commands.opened


def test_existing_relative_task_named_folder_keeps_file_inspection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'image data').mkdir()
    assert app._easy_task_keys('image data') == ()
    assert app._easy_module_search('image data') == []


def test_spatial_task_keeps_declared_page_fold_maturity(commands, monkeypatch):
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, 'maturity_is_visible', lambda stage: stage != 'alpha')
    rows = app.MainWindow._easy_navigation_suggestions(commands, 'analyze spatial imaging')
    assert rows == []
    assert not commands.opened


@pytest.mark.parametrize('phrase', ['image analasys', 'HIGH CONTENT IMAGE ANALASYS'])
def test_user_supplied_analysis_spellings_offer_image_tools(commands, phrase):
    rows = app.MainWindow._easy_navigation_suggestions(commands, phrase)
    assert tuple(row[0] for row in rows) == ('mask', 'measure', 'qc_dashboard')
    assert not commands.opened and not commands.navigated


@pytest.mark.parametrize('phrase,expected', _ALTERNATES)
def test_paired_alternate_tasks_offer_cards_without_opening_modules(
        commands, qtbot, monkeypatch, phrase, expected):
    test_reviewed_tasks_offer_cards_without_opening_modules(
        commands, qtbot, monkeypatch, phrase, expected)


@pytest.mark.parametrize('phrase', [
    'analyze masks', 'ANALYSE MASKS', 'mask analysis',
    'analyze segmentation masks', 'segmentation mask analysis',
])
def test_existing_mask_analysis_tasks_offer_native_cards_on_enter(
        commands, qtbot, monkeypatch, phrase):
    test_reviewed_tasks_offer_cards_without_opening_modules(
        commands, qtbot, monkeypatch, phrase, ('measure', 'qc_dashboard', 'make_masks'))


@pytest.mark.parametrize('phrase', [
    'detect features', 'FEATURE DETECTION', 'detect image features',
    'detect morphological features', 'extract features', 'feature extraction',
])
def test_feature_detection_tasks_offer_measure_without_launching(
        commands, qtbot, monkeypatch, phrase):
    test_reviewed_tasks_offer_cards_without_opening_modules(
        commands, qtbot, monkeypatch, phrase, ('measure',))
