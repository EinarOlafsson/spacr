"""A relative measurements database path reads instead of crashing (GitHub #143).

``_read_db`` opens a database it may not write through a read-only file URI,
and ``pathlib.Path.as_uri`` refuses relative paths, so a ``src`` such as
``plate1/measurements/measurements.db`` stopped Classify with "relative path
can't be expressed as a file URI". The path is now made absolute first, and a
path naming no file says which absolute path was tried.
"""

import os
import sqlite3
import stat

import pandas as pd
import pytest

from spacr.io import _read_db


def _plate(root):
    folder = root / "plate1" / "measurements"
    folder.mkdir(parents=True)
    database = folder / "measurements.db"
    with sqlite3.connect(database) as connection:
        pd.DataFrame({"plateID": ["plate1"], "rowID": ["r1"], "columnID": ["c1"],
                      "fieldID": ["f1"], "object_label": [1], "cell_area": [10.0]}
                     ).to_sql("cell", connection, index=False)
    return database


def test_a_relative_read_only_database_is_read(tmp_path, monkeypatch):
    database = _plate(tmp_path)
    os.chmod(database, stat.S_IRUSR)
    os.chmod(database.parent, stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.chdir(tmp_path)
    try:
        frames = _read_db("plate1/measurements/measurements.db", ["cell"])
    finally:
        os.chmod(database.parent, stat.S_IRWXU)
        os.chmod(database, stat.S_IRUSR | stat.S_IWUSR)
    assert len(frames) == 1 and len(frames[0]) == 1


def test_a_missing_database_names_the_absolute_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as caught:
        _read_db("plate9/measurements/measurements.db", ["cell"])
    assert str(tmp_path / "plate9" / "measurements" / "measurements.db") in str(caught.value)
