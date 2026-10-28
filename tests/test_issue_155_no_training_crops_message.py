"""GitHub #155: Classify on a source with no crop rows names what is missing.

The reporter's source had no ``png_list`` rows, so the crop list fell back to
the ``cell`` measurement table, which gave nothing either. The empty frame
reached ``png_df['png_path']`` and the run died with ``KeyError: 'png_path'``.
It now refuses with a message that says whether the database is missing or
which table is empty, and what to change.
"""
from __future__ import annotations

import sqlite3

import pytest


def _settings(src):
    return {"src": str(src), "dataset_mode": "metadata",
            "class_metadata": [["c1"], ["c2"]], "metadata_type_by": "columnID",
            "class_folder_names": ["nc", "pc"]}


def test_a_source_without_a_measurements_database(tmp_path):
    from spacr.io import generate_training_dataset

    with pytest.raises(FileNotFoundError) as excinfo:
        generate_training_dataset(_settings(tmp_path))
    message = str(excinfo.value)
    assert "png_path" not in message
    assert "No crops to train on" in message
    assert "measurements.db" in message
    assert "run Measure on this plate first" in message


def test_a_database_with_no_png_list_and_an_empty_cell_table(tmp_path):
    from spacr.io import generate_training_dataset

    (tmp_path / "measurements").mkdir()
    db = tmp_path / "measurements" / "measurements.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE cell (object_label INTEGER, plateID TEXT, "
                     "rowID TEXT, columnID TEXT, fieldID TEXT, prcf TEXT, "
                     "file_name TEXT, path_name TEXT)")
    with pytest.raises(ValueError) as excinfo:
        generate_training_dataset(_settings(tmp_path))
    message = str(excinfo.value)
    assert "no 'png_list' table" in message
    assert "'cell' has 0 row(s)" in message
    assert "path_string='cell_png'" in message


def test_a_source_that_is_a_file(tmp_path):
    from spacr.io import _no_training_crops_error

    db = tmp_path / "measurements.db"
    db.write_bytes(b"")
    error = _no_training_crops_error(str(db), str(db / "x"), "cell", "cell_png")
    assert isinstance(error, FileNotFoundError)
    assert "Set src to that plate folder" in str(error)


def test_a_corrupt_database_reports_missing_crops_without_repairing_or_replacing_it(tmp_path):
    from spacr.io import _no_training_crops_error

    folder = tmp_path / 'measurements'
    folder.mkdir()
    database = folder / 'measurements.db'
    original = b'this is not a SQLite database; preserve the evidence\n'
    database.write_bytes(original)
    error = _no_training_crops_error(str(tmp_path), str(database), 'cell', 'cell_png')
    assert isinstance(error, ValueError)
    assert "no 'png_list' table" in str(error) and "no 'cell' table" in str(error)
    assert 'Measure' in str(error)
    assert database.read_bytes() == original
    assert sorted(path.name for path in folder.iterdir()) == ['measurements.db']
