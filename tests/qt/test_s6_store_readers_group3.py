"""Unified-store step S6, group 3: the home screen's database evidence and the
drop handlers read measurement databases through :mod:`spacr.store`.

Each reader is run on an older ``measurements.db`` and on a spaCR store
project holding the same tables, and compared with the values the fixture
data implies, so the move onto the store changes neither what an older file
yields nor leaves a store project unread.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from spacr import store
from spacr.qt import dnd_handlers as dh
from spacr.qt.widgets import home
from spacr.store_browse import READ_ERRORS

CELLS = pd.DataFrame({"object_key": [1, 2, 3], "plateID": ["plate1"] * 3,
                      "rowID": ["r1"] * 3, "columnID": ["c1"] * 3,
                      "cell_area": [10.0, 20.5, 31.0]})
PNGS = pd.DataFrame({"png_path": ["a.png", "b.png"], "plateID": ["plate1"] * 2})
SPATIAL = pd.DataFrame({"source": ["/data/xenium/slide1", "/data/xenium/slide2"]})


@pytest.fixture(autouse=True)
def _fresh_shared_stores():
    """Close every shared store a test opened, so no test reads another's cache.

    :returns: nothing; yields once around the test.
    """
    yield
    store.release_shared()


def _legacy(path: Path, tables) -> Path:
    """Write an older SQLite measurements file with ``tables`` in creation order.

    :param path: the file to create.
    :param tables: ``{name: DataFrame}``.
    :returns: the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        for name, frame in tables.items():
            frame.to_sql(name, connection, index=False)
    return path


def _project(folder: Path, tables) -> Path:
    """Write a spaCR store project holding ``tables`` for one well.

    :param folder: the project folder.
    :param tables: ``{name: DataFrame}``.
    :returns: the project folder.
    """
    for name, frame in tables.items():
        sort_by = ("object_key",) if "object_key" in frame else ()
        store.write_table(folder, name, frame, plate="plate1", well="A01",
                          run_id="t", sort_by=sort_by)
    store.register(folder)
    return folder


def test_database_evidence_is_the_same_for_an_older_file(tmp_path):
    database = _legacy(tmp_path / "plate1" / "measurements" / "measurements.db",
                       {"png_list": PNGS, "cell": CELLS})
    assert home._easy_database_evidence(database) == dict(
        tables=["png_list", "cell"], numeric=True, objects=True, crops=True,
        annotation_only=False)


def test_database_evidence_reads_a_store_project(tmp_path):
    project = _project(tmp_path / "plate1" / "measurements",
                       {"cell": CELLS, "png_list": PNGS})
    expected = dict(tables=["cell", "png_list"], numeric=True, objects=True,
                    crops=True, annotation_only=False)
    assert home._easy_database_evidence(project / store.CATALOG) == expected
    assert home._easy_database_evidence(project) == expected


def test_database_evidence_without_numbers_or_identity(tmp_path):
    words = pd.DataFrame({"plateID": ["p"], "note": ["x"]})
    database = _legacy(tmp_path / "old" / "measurements.db", {"cell": words})
    project = _project(tmp_path / "new", {"cell": words})
    for path in (database, project):
        evidence = home._easy_database_evidence(path)
        assert (evidence["numeric"], evidence["objects"], evidence["crops"]) == (
            False, False, False)


def test_an_annotation_only_file_is_still_recognised(tmp_path):
    annotations = pd.DataFrame({"file": ["a.tif"], "annotation": [1]})
    database = _legacy(tmp_path / "measurements.db",
                       {"image_annotations": annotations})
    assert home._easy_database_evidence(database)["annotation_only"] is True


def test_an_unreadable_file_raises_a_read_error(tmp_path):
    broken = tmp_path / "measurements.db"
    broken.write_text("not sqlite")
    with pytest.raises(READ_ERRORS):
        home._easy_database_evidence(broken)
    with pytest.raises(READ_ERRORS):
        home._easy_database_evidence(tmp_path / "missing" / "measurements.db")


def test_the_spatial_source_read_is_the_same_on_both(tmp_path):
    statement = "SELECT substr(source, 1, 4096) FROM spatial_fields LIMIT 8"
    expected = [(value,) for value in SPATIAL["source"]]
    database = _legacy(tmp_path / "old" / "measurements" / "measurements.db",
                       {"spatial_fields": SPATIAL})
    project = _project(tmp_path / "new" / "measurements",
                       {"spatial_fields": SPATIAL})
    for path in (database, project / store.CATALOG):
        rows = home._easy_store_read(
            path, lambda connection: connection.execute(statement).fetchall(),
            lambda opened: list(opened.sql(statement).itertuples(
                index=False, name=None)))
        assert rows == expected


def test_an_older_file_is_read_without_starting_duckdb(tmp_path):
    database = _legacy(tmp_path / "measurements.db", {"cell": CELLS})
    home._easy_database_evidence(database)
    assert store.shared(database)._duck is None


def test_analysis_discovery_finds_both_kinds(tmp_path):
    old = _legacy(tmp_path / "plate1" / "measurements" / "measurements.db",
                  {"cell": CELLS})
    beside = _legacy(tmp_path / "plate1" / "measurements.db", {"cell": CELLS})
    project = _project(tmp_path / "plate2" / "measurements", {"cell": CELLS})
    found = home._easy_analysis_inputs(tmp_path)
    assert set(found) == {old, beside, project / store.CATALOG}
    assert found.index(old) < found.index(beside)


def test_analysis_discovery_prefers_the_store_in_one_folder(tmp_path):
    _legacy(tmp_path / "plate1" / "measurements" / "measurements.db", {"cell": CELLS})
    project = _project(tmp_path / "plate1" / "measurements", {"cell": CELLS})
    assert home._easy_analysis_inputs(tmp_path) == [project / store.CATALOG]


def test_table_names_are_the_same_for_an_older_file(tmp_path):
    database = _legacy(tmp_path / "measurements.db",
                       {"png_list": PNGS, "cell": CELLS})
    assert dh.table_names(database) == ["cell", "png_list"]


def test_table_names_read_a_store_project(tmp_path):
    run = tmp_path / "run"
    project = _project(run / "measurements", {"png_list": PNGS, "cell": CELLS})
    for path in (project, project / store.CATALOG, run):
        assert dh.table_names(path) == ["cell", "png_list"]
    assert dh.table_names(tmp_path / "elsewhere") == []


def test_the_database_browser_takes_a_dropped_store_project(tmp_path):
    run = tmp_path / "run"
    project = _project(run / "measurements", {"cell": CELLS})
    database = _legacy(tmp_path / "old" / "measurements" / "measurements.db",
                       {"cell": CELLS})
    handler = dh.DatabaseDropHandler()
    for path in (run, project, project / store.CATALOG, database,
                 database.parent, database.parent.parent):
        assert handler.can_accept(path), path
    (tmp_path / "empty").mkdir()
    assert not handler.can_accept(tmp_path / "empty")

    class Browser:
        """A stand-in screen that records what it was asked to open."""

        opened = []

        def set_database(self, path):
            """Record the path and accept it.

            :param path: the path the drop handed over.
            :returns: True.
            """
            self.opened.append(path)
            return True

    screen = Browser()
    handler.apply(project / store.CATALOG, screen)
    handler.apply(database, screen)
    assert screen.opened == [str(project / store.CATALOG), str(database)]


def test_plate_view_drops_keep_their_own_rule(tmp_path):
    """Plate View takes what the measurement screens take.

    With step S4 the measurement screens' rule (MeasurementsDropHandler)
    names a store project's measurements folder too, and Plate View reads
    one through the shared store (S7c), so both drops are accepted; a folder
    with neither stays refused.
    """
    project = _project(tmp_path / "run" / "measurements", {"cell": CELLS})
    database = _legacy(tmp_path / "old" / "measurements" / "measurements.db",
                       {"cell": CELLS})
    handler = dh.ResultsDatabaseDropHandler()
    assert handler.can_accept(database)
    assert handler.can_accept(project) is dh.MeasurementsDropHandler().can_accept(project)
    assert handler.can_accept(project)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not handler.can_accept(empty)
