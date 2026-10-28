"""Local-input invariants and real Navigator widgets, without a native capture."""
import hashlib
import importlib.util
from pathlib import Path
import sqlite3
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / 'tools/tutorials'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('navigator_capture_under_test', TOOLS / 'capture_navigator.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
print('ACTUAL_CAPTURE_IMPORT', driver.__file__, hashlib.sha256(Path(driver.__file__).read_bytes()).hexdigest())


@pytest.fixture
def local_inputs(tmp_path):
    """Synthetic unit-test inputs; no test result is a captured scientific result."""
    from PIL import Image
    root = tmp_path / 'annotate_fresh/example_data/plate1'
    image = root / 'data/multiple_nucleus/multiple_pathogens/plate1_E01/cell_png/plate1_E01_10_1_4.png'
    image.parent.mkdir(parents=True)
    Image.new('RGB', (8, 8), 'gray').save(image)
    database = root / 'measurements/measurements.db'
    database.parent.mkdir()
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, cell_area REAL)')
        connection.execute('CREATE TABLE png_list (png_path TEXT)')
    settings = root / 'settings/annotate_settings.csv'
    settings.parent.mkdir()
    settings.write_text('Key,Value\nannotation_column,infected\n')
    return tmp_path


def test_exclusive_real_byte_copies_and_history_only_scope(local_inputs):
    import spacr.qt
    spacr.qt.register_self_registering_modules()
    rows = driver.prepare_inputs(local_inputs)
    result = driver.checked_input_routes(rows)
    assert [r[0] for r in result['annotation']['actions']] == ['db_browser']
    assert result['annotation']['context']['analysis_inputs'] == {}
    assert driver.digest(rows['file_copy']) == driver.digest(rows['image'])
    assert driver.digest(rows['folder_copy'] / rows['image'].name) == driver.digest(rows['image'])
    assert all(driver.digest(path) == sha for path, sha in rows['original'].items())
    existing = rows['annotation'].read_bytes()
    with pytest.raises(FileExistsError):
        driver.prepare_inputs(local_inputs)
    assert rows['annotation'].read_bytes() == existing


def test_history_example_with_measurements_is_rejected(local_inputs):
    rows = driver.prepare_inputs(local_inputs)
    with sqlite3.connect(rows['annotation']) as connection:
        connection.execute('CREATE TABLE cell (cell_area REAL)')
    with pytest.raises(ValueError, match='history-only'):
        driver.checked_input_routes(rows)


def test_reviewed_entry_point_matches_actual_geometry_and_entry():
    target = Path('/private/file-tray/real.png')
    actual = dict(rect=[0, 32, 1280, 1408])
    review = dict(window_rect=actual['rect'], tray_name='file-tray', entry_name='real.png',
                  source_point=[370, 230])
    assert driver.checked_source_point(review, actual, target) == [370, 230]


@pytest.mark.parametrize('change', [
    dict(window_rect=[1, 32, 1280, 1408]), dict(entry_name='wrong.png'),
    dict(tray_name='other'), dict(source_point=[370, 40]),
    dict(source_point=[1400, 230]), dict(source_point=[float('nan'), 230]),
])
def test_entry_point_rejects_changed_window_wrong_input_or_unreachable_point(change):
    actual = dict(rect=[0, 32, 1280, 1408])
    review = dict(window_rect=actual['rect'], tray_name='file-tray', entry_name='real.png',
                  source_point=[370, 230])
    review.update(change)
    with pytest.raises(ValueError):
        driver.checked_source_point(review, actual, Path('/private/file-tray/real.png'))


def test_actual_widget_enter_case_query_does_not_open_tool(qtbot, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings, Qt
    import spacr.qt
    from spacr.qt import preferences
    from spacr.qt.app import _easy_module_search
    from spacr.qt.widgets.home import _EasyHomePage
    monkeypatch.setattr(preferences, '_settings', lambda: QSettings(str(tmp_path / 'profile.ini'), QSettings.IniFormat))
    preferences.set_ambient_enabled(False)
    preferences.set_reduce_motion(True)
    preferences.set_show_alpha(False)
    spacr.qt.register_self_registering_modules()
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1800, 1000)
    page.show()
    opened = []
    page.module_requested.connect(lambda *args: opened.append(args))
    edit = page.path_edit
    qtbot.keyClicks(edit, 'mEaSuRe')
    assert not page.field.cards
    qtbot.keyClick(edit, Qt.Key_Return)
    qtbot.waitUntil(lambda: bool(page.field.cards), timeout=5000)
    assert 'measure' in {card._key for card in page.field.cards}
    assert [card._key for card in page.field.cards] == [row[0] for row in _easy_module_search('Measure')]
    assert not opened and page._workflow is None and not page.path_edit.ai.isChecked()
    page.close()


def test_scene_order_matches_canonical():
    import json
    canonical = json.loads((TOOLS / 'lessons/88_navigator.json').read_text())
    assert tuple(scene['visual'] for scene in canonical['scenes']) == driver.SCENES


def test_gtk_shadow_inset_preserves_raw_client_and_reachable_frame():
    properties = ('_GTK_FRAME_EXTENTS(CARDINAL) = 20, 20, 20, 20\n'
                  '_NET_FRAME_EXTENTS(CARDINAL) = 0, 0, 0, 0\n')
    client = [-20, -20, 1320, 1480]  # Observed p48 X client; extents here are a unit fixture.
    row = driver._visible_files_geometry(client, properties, [0, 0, 2560, 1440])
    assert row['rect'] == [0, 0, 1280, 1440]
    assert row['client_rect'] == client and row['gtk_frame_extents'] == [20, 20, 20, 20]
    assert row['net_frame_extents'] == [0, 0, 0, 0]
    assert not row['entry_geometry_verified']
    review = dict(window_rect=row['rect'], tray_name='file-tray', entry_name='real.png',
                  source_point=[370, 230])
    assert driver.checked_source_point(review, row, Path('/private/file-tray/real.png')) == [370, 230]
    review['source_point'] = [-10, 230]
    with pytest.raises(ValueError):
        driver.checked_source_point(review, row, Path('/private/file-tray/real.png'))


@pytest.mark.parametrize('gtk,net,client', [
    ('_GTK_FRAME_EXTENTS:  not found.', '0, 0, 0, 0', [-20, -20, 1320, 1480]),
    ('_GTK_FRAME_EXTENTS(CARDINAL) = 20, 20, 20, 20', '0, 0, 0, 0', [-21, -20, 1320, 1480]),
    ('_GTK_FRAME_EXTENTS(CARDINAL) = 20, 20, 20, 20', '0, 0, 0, 0', [-20, -20, 1321, 1481]),
    ('_GTK_FRAME_EXTENTS(CARDINAL) = -20, 20, 20, 20', '0, 0, 0, 0', [-20, -20, 1320, 1480]),
    ('_GTK_FRAME_EXTENTS(STRING) = "20, 20, 20, 20"', '0, 0, 0, 0', [-20, -20, 1320, 1480]),
    ('_GTK_FRAME_EXTENTS(CARDINAL) = 20, 20, 20, 20', '4, 4, 24, 4', [-20, -20, 1320, 1480]),
])
def test_frame_properties_cannot_excuse_visible_offscreen_or_malformed_geometry(gtk, net, client):
    with pytest.raises(RuntimeError):
        driver._visible_files_geometry(client, gtk + '\n_NET_FRAME_EXTENTS(CARDINAL) = ' + net,
                                       [0, 0, 2560, 1440])


def test_net_frame_extents_are_outward_wm_borders_not_inset_shadows():
    properties = ('_GTK_FRAME_EXTENTS:  not found.\n'
                  '_NET_FRAME_EXTENTS(CARDINAL) = 4, 4, 24, 4\n')
    row = driver._visible_files_geometry([80, 60, 800, 600], properties, [0, 0, 2560, 1440])
    assert row['rect'] == row['client_rect'] == [80, 60, 800, 600]
    assert row['gtk_frame_extents'] is None
    assert row['wm_frame_rect'] == [76, 36, 808, 628]


@pytest.fixture
def actual_path_editor(qtbot):
    from PySide6.QtWidgets import QWidget
    from spacr.qt.widgets.home import _EasyPathEdit
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(2560, 1440)
    edit = _EasyPathEdit(host)
    edit.setGeometry(920, 680, 720, 64)
    host.show()
    qtbot.waitUntil(lambda: edit.isVisible() and edit.textMargins().right() > 0)
    return host, edit


@pytest.mark.parametrize('retain_reserved_margin', [True, False])
def test_real_path_editor_drop_point_avoids_tiled_files_and_native_controls(actual_path_editor,
                                                                          retain_reserved_margin):
    from PySide6.QtCore import QPoint, QRect
    host, edit = actual_path_editor
    if not retain_reserved_margin:
        edit.setTextMargins(18, 0, 0, 0)
    origin = host.mapToGlobal(QPoint(0, 0))
    files = [origin.x(), origin.y(), 1280, 1440]
    center = edit.mapToGlobal(edit.rect().center())
    assert QRect(*files).contains(center)
    point = driver.drop_text_point(edit, files)
    actual = edit.mapToGlobal(point)
    assert actual.x() > files[0] + files[2] + 4
    assert not QRect(*files).contains(actual)
    assert edit.contentsRect().marginsRemoved(edit.textMargins()).contains(point)
    assert not edit.browse.geometry().contains(point)
    assert not edit.ai.geometry().contains(point)


def test_real_path_editor_fully_covered_drop_is_rejected(actual_path_editor):
    from PySide6.QtCore import QPoint
    host, edit = actual_path_editor
    origin = host.mapToGlobal(QPoint(0, 0))
    with pytest.raises(RuntimeError, match='no uncovered'):
        driver.drop_text_point(edit, [origin.x(), origin.y(), host.width(), host.height()])


def test_real_readonly_editor_cannot_be_a_drop_target(actual_path_editor):
    _host, edit = actual_path_editor
    edit.setReadOnly(True)
    with pytest.raises(RuntimeError, match='unavailable'):
        driver.drop_text_point(edit, [0, 0, 100, 100])
