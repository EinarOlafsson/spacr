"""Relocating a tutorial copy must not rewrite original labels or image bytes."""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_annotation_capture import digest, prepare


def source_project(root):
    source = root / 'original'
    (source / 'data').mkdir(parents=True)
    (source / 'measurements').mkdir()
    image = source / 'data/cell.png'
    image.write_bytes(b'original crop bytes')
    database = source / 'measurements/measurements.db'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE png_list (png_path TEXT, infected INTEGER, reviewed TEXT, rowID TEXT)')
        db.execute('INSERT INTO png_list VALUES (?, ?, ?, ?)', (str(image), 1, 'keep', 'r12'))
        second = source / 'data/other-cell.png'
        second.write_bytes(b'other original crop bytes')
        db.execute('INSERT INTO png_list VALUES (?, ?, ?, ?)', (str(second), 0, None, 'r12'))
        db.execute('CREATE TABLE unrelated (value TEXT)')
        db.execute("INSERT INTO unrelated VALUES ('preserve this too')")
    return source, database, image


def test_copy_relocates_only_paths_and_keeps_the_original_unchanged(tmp_path):
    source, database, image = source_project(tmp_path)
    original_database = digest(database)
    destination = tmp_path / 'prepared'
    visible = Path('/tmp/annotation-recording/plate1')
    receipt = prepare(source, destination, visible)
    assert receipt['accepted'] and receipt['rows'] == receipt['unique_images'] == 2
    assert receipt['changed_columns'] == ['png_path']
    assert digest(database) == original_database
    assert image.read_bytes() == (destination / 'data/cell.png').read_bytes()
    with sqlite3.connect(destination / 'measurements/measurements.db') as db:
        assert db.execute('SELECT * FROM png_list').fetchall() == [
            (str(visible / 'data/cell.png'), 1, 'keep', 'r12'),
            (str(visible / 'data/other-cell.png'), 0, None, 'r12')]
        assert db.execute('SELECT * FROM unrelated').fetchall() == [('preserve this too',)]
    with pytest.raises(FileExistsError):
        prepare(source, destination, visible)


def test_a_crop_outside_the_source_is_rejected_before_any_copy(tmp_path):
    source, database, image = source_project(tmp_path)
    outside = tmp_path / 'other.png'
    outside.write_bytes(b'unrelated pixels')
    with sqlite3.connect(database) as db:
        db.execute('UPDATE png_list SET png_path=?', (str(outside),))
    before = digest(database)
    destination = tmp_path / 'prepared'
    with pytest.raises(ValueError):
        prepare(source, destination, Path('/tmp/annotation-recording/plate1'))
    assert not destination.exists()
    assert digest(database) == before
    assert image.read_bytes() == b'original crop bytes'


def test_personal_visible_paths_are_rejected_before_any_copy(tmp_path):
    source, database, _ = source_project(tmp_path)
    destination = tmp_path / 'prepared'
    with pytest.raises(ValueError, match='neutral'):
        prepare(source, destination, Path('/home/maintainer/example'))
    assert not destination.exists()
