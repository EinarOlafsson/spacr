"""Generated-output suggestions use the native file and table adapters."""
import sqlite3

import numpy as np
from spacr.qt.screens.graph_builder import GraphBuilderScreen
from spacr.qt.screens.make_masks import MakeMasksScreen, QInputDialog
from tests.qt.test_733_easy_commands import commands
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


def _context(key, source, project):
    route = dict(ready=True, source=str(source), project=str(project))
    return {'generated_outputs': {'routes': {key: route}}, 'analysis_inputs': {key: route}}


def test_generated_npy_make_masks_uses_the_exact_file_and_real_editor(commands, qtbot, tmp_path, monkeypatch):
    merged = tmp_path / 'plate1' / 'merged'
    merged.mkdir(parents=True)
    source = merged / 'AssayPlate%5FPerkinElmer%5FCellCarrier-384_A03_8_1.npy'
    pixels = np.arange(40 * 48 * 4, dtype=np.uint16).reshape(40, 48, 4)
    np.save(source, pixels)
    before = source.read_bytes()
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    commands._open_easy_named_module = lambda key: commands.opened.append(key) or screen
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args: ('YXC', True))
    monkeypatch.setattr(QInputDialog, 'getInt', lambda *args: (2, True))
    assert commands._open_easy_input('make_masks', str(tmp_path), _context('make_masks', source, merged.parent))
    qtbot.waitUntil(lambda: not screen._loading and screen._load_worker is None, timeout=20000)
    assert commands.opened == ['make_masks']
    assert screen._folder == str(merged) and screen._image_files == [source.name]
    np.testing.assert_array_equal(screen._canvas.image, pixels[..., 2])
    assert not screen._canvas.mask.any()
    assert source.read_bytes() == before


def test_generated_graph_uses_real_existing_database_and_native_table_loader(commands, qtbot, tmp_path):
    folder = tmp_path / 'plate1' / 'measurements'
    folder.mkdir(parents=True)
    database = folder / 'measurements.db'
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE cell (object_label INTEGER, value REAL, wellID TEXT)')
        connection.executemany('INSERT INTO cell VALUES (?,?,?)', [(1, 3.5, 'A03'), (2, 7.5, 'A03')])
    before = database.read_bytes()
    screen = GraphBuilderScreen(threaded=False)
    qtbot.addWidget(screen)
    commands._open_easy_named_module = lambda key: commands.opened.append(key) or screen
    source = folder.parent / 'merged' / 'selected.npy'
    assert commands._open_easy_input('graph_builder', str(source), _context('graph_builder', database, folder.parent))
    qtbot.waitUntil(lambda: screen._frame is not None, timeout=10000)
    assert commands.opened == ['graph_builder']
    assert screen._path == str(database)
    assert screen._frame['value'].tolist() == [3.5, 7.5]
    assert screen._frame['object_label'].tolist() == [1, 2]
    assert database.read_bytes() == before
