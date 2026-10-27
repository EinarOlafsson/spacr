"""Pure format keywords offer actual native cards without consuming file paths."""
import numpy as np
import pytest
from PySide6.QtCore import Qt

from spacr import convert
from spacr.qt import app
from spacr.qt.widgets.home import _EasyHomePage
from tests.qt.test_733_easy_commands import commands
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


_FORMATS = [('sql', {'db_browser', 'graph_builder', 'gate_editor'}),
            ('db', {'db_browser', 'graph_builder', 'gate_editor'}),
            ('sqlite', {'db_browser', 'graph_builder', 'gate_editor'}),
            ('sqlite3', {'db_browser', 'graph_builder', 'gate_editor'}),
            ('npy', {'make_masks', 'measure', 'convert'}),
            ('tiff', {'make_masks', 'mask', 'convert'}),
            ('tif', {'make_masks', 'mask', 'convert'}),
            ('bmp', {'make_masks', 'mask', 'convert'}),
            ('bnp', {'make_masks', 'mask', 'convert'}),
            ('png', {'make_masks', 'mask', 'convert'}),
            ('jpg', {'make_masks', 'mask', 'convert'}),
            ('jpeg', {'make_masks', 'mask', 'convert'}),
            ('csv', {'graph_builder', 'gate_editor'}),
            ('tsv', {'graph_builder', 'gate_editor'}),
            ('fastq', {'map_barcodes'}), ('fq', {'map_barcodes'}),
            ('fastq.gz', {'map_barcodes'}), ('fq.gz', {'map_barcodes'}),
            ('pdf', {'analyze_plaques'})]


@pytest.mark.parametrize('token,expected', _FORMATS)
@pytest.mark.parametrize('style', ['plain', 'dotted_upper'])
def test_format_keywords_on_real_enter_offer_cards_and_never_launch(
        commands, qtbot, monkeypatch, token, expected, style):
    monkeypatch.setattr(app, 'app_is_visible', lambda key: True)
    page = _EasyHomePage(commands)
    commands.setCentralWidget(page)
    commands._startup = page
    commands._easy_navigation_suggestions = lambda value: app.MainWindow._easy_navigation_suggestions(commands, value)
    page.command_requested.connect(commands._open_easy_command)
    text = token if style == 'plain' else '.' + token.upper()
    qtbot.keyClicks(page.path_edit, text)
    assert not commands.opened and not page.field.cards
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    qtbot.waitUntil(lambda: bool(page.field.cards), timeout=3000)
    keys = [card._key for card in page.field.cards]
    assert expected <= set(keys)
    assert len(keys) == len(set(keys))
    assert not commands.opened and not commands.navigated
    if token == 'npy':
        assert 'mask' not in keys
    assert all(not key.startswith('__easy_command__:') for key in keys)


@pytest.mark.parametrize('extension', ['.nd2', '.czi', '.lif'])
@pytest.mark.parametrize('available', [False, True])
def test_vendor_tokens_follow_installed_reader_capability(commands, monkeypatch, extension, available):
    monkeypatch.setattr(convert, 'reader_available', lambda ext: available)
    rows = app.MainWindow._easy_navigation_suggestions(commands, extension.upper())
    assert [row[0] for row in rows] == (['convert'] if available else [])


@pytest.mark.parametrize('token', ['gif', 'webp'])
def test_installed_general_raster_formats_offer_converter_without_false_editor_route(commands, token):
    if '.' + token not in convert._image_extensions():
        pytest.skip('Pillow opener is not installed')
    rows = app.MainWindow._easy_navigation_suggestions(commands, token)
    assert [row[0] for row in rows] == ['convert']


@pytest.mark.parametrize('value', ['image.npy', 'image.tif', 'dump.sql', '/actual/sql',
                                   './npy', '~/tiff', r'C:\data\image.npy', 'reads.fastq.gz'])
def test_actual_filenames_and_paths_are_not_format_keywords(value):
    assert app._easy_format_suggestions(value) is None


def test_existing_format_named_folder_or_file_keeps_local_inspection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'sql').mkdir()
    (tmp_path / '.tif').mkdir()
    np.save(tmp_path / 'image.npy', np.zeros((4, 5), np.uint16))
    (tmp_path / 'dump.sql').write_text('CREATE TABLE cell (value REAL);')
    for value in ('sql', '.tif', 'image.npy', 'dump.sql'):
        assert app._easy_format_suggestions(value) is None


def test_format_cards_keep_visibility_and_native_module_search(commands, monkeypatch):
    monkeypatch.setattr(app, 'app_is_visible', lambda key: key == 'make_masks')
    assert [row[0] for row in app.MainWindow._easy_navigation_suggestions(commands, '.NPY')] == ['make_masks']
    assert 'measure' in {row[0] for row in app.MainWindow._easy_navigation_suggestions(commands, 'MeAsUrE')}
