"""Filename-derived PASS must not override the example database's actual wells."""
import importlib.util
from pathlib import Path
import sqlite3

spec = importlib.util.spec_from_file_location(
    'capture_classify', Path(__file__).resolve().parents[1] / 'capture_classify.py')
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


def measured_split(tmp_path, test_well):
    database = tmp_path / 'measurements.db'
    with sqlite3.connect(database) as con:
        con.execute('CREATE TABLE png_list (png_path,plateID,rowID,columnID)')
        con.executemany('INSERT INTO png_list VALUES (?,?,?,?)', [
            ('/crops/plate1_A01_1_1_1.png','plate1','r1','c1'),
            ('/crops/plate1_A01_2_1_2.png','plate1',*test_well)])
    for split, name in [('train','plate1_A01_1_1_1.png'), ('test','plate1_A01_2_1_2.png')]:
        folder = tmp_path / split / 'class_1'
        folder.mkdir(parents=True)
        (folder / name).write_bytes(b'lineage fixture; pixels are not read by this check')
    return recorder.inspect_database_split(database, tmp_path)


def test_real_independent_wells_pass_even_when_names_are_ambiguous(tmp_path):
    proof = measured_split(tmp_path, ('r2', 'c1'))
    assert proof['accepted'] is True
    assert proof['sample_counts'] == {'train': 1, 'test': 1}
    assert proof['overlapping_actual_wells'] == []


def test_same_actual_well_in_different_filename_fields_is_rejected(tmp_path):
    proof = measured_split(tmp_path, ('r1', 'c1'))
    assert proof['accepted'] is False
    assert proof['overlapping_actual_wells'] == [('plate1', 'r1', 'c1')]
