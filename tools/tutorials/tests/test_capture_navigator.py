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
