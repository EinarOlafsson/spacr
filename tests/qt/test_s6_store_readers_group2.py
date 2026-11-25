"""S6 group 2: the QC dashboard's units card reads through :mod:`spacr.store`.

The same fixture rows are written once as an older ``measurements.db`` and
once as a store project; the card must report the same verdict, headline and
distinct stamps from both, and the stamps must be the ones the fixture holds.
"""
from __future__ import annotations

import os
import sqlite3

import pandas as pd
import pytest

from spacr import store
from spacr.qt.widgets import qc_summary as qc


def _cells(stamps):
    """A small cell table carrying the given ``(ndim, units)`` stamps.

    :param stamps: one ``(measurement_ndim, measurement_units)`` per row.
    :returns: the frame.
    """
    return pd.DataFrame({
        "object_key": list(range(1, len(stamps) + 1)),
        "plate": ["plate1"] * len(stamps),
        "well": ["A01"] * len(stamps),
        "cell_area": [float(i) for i in range(len(stamps))],
        "measurement_ndim": [s[0] for s in stamps],
        "measurement_units": [s[1] for s in stamps],
    })


def _legacy(root, tables):
    """Write ``tables`` into ``root/measurements/measurements.db``.

    :param root: the project folder.
    :param tables: ``{name: frame}``.
    :returns: the database path.
    """
    path = root / "measurements" / "measurements.db"
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as db:
        for name, frame in tables.items():
            frame.to_sql(name, db, index=False)
    return path


def _project(root, tables):
    """Write ``tables`` as a store project in ``root/measurements``.

    :param root: the project folder.
    :param tables: ``{name: frame}``.
    :returns: the store folder.
    """
    project = root / "measurements"
    for name, frame in tables.items():
        store.write_table(str(project), name, frame, plate="plate1",
                          well="A01", run_id="t", sort_by=("object_key",))
    store.register(str(project))
    return project


def _expected(frame):
    """The distinct stamps of a fixture frame, built without the store.

    :param frame: the fixture.
    :returns: the set of ``(ndim, units)`` tuples.
    """
    held = frame[["measurement_ndim", "measurement_units"]].drop_duplicates()
    return {(int(a), str(b)) for a, b in held.itertuples(index=False)}


def _stamps_in(card):
    """The two example stamps a failing card names, parsed back.

    :param card: a ``fail`` units card.
    :returns: the stamps the detail line quotes.
    """
    import ast

    text = card.detail[0]
    body = text.split("e.g. ", 1)[1].split(". A column", 1)[0]
    first, second = body.split(" and ")
    return {ast.literal_eval(first), ast.literal_eval(second)}


@pytest.mark.parametrize("kind", ["legacy", "store"])
def test_mixed_units_are_found_in_both_kinds_of_project(tmp_path, kind):
    """Two stamps in one table fail the card, naming the fixture's stamps."""
    frame = _cells([(2, "px"), (3, "um"), (2, "px")])
    write = _legacy if kind == "legacy" else _project
    source = write(tmp_path, {"cell": frame})
    card = qc._read_units(str(tmp_path))
    assert card.verdict == "fail"
    assert card.headline == ("1 table(s) hold rows measured in different "
                             "units (cell).")
    assert "2 distinct stamps" in card.detail[0]
    assert _stamps_in(card) == _expected(frame)
    assert card.source == str(source)
    assert all(type(v) in (int, str) for s in _stamps_in(card) for v in s)


@pytest.mark.parametrize("kind", ["legacy", "store"])
def test_one_stamp_beside_an_unstamped_table_is_consistent(tmp_path, kind):
    """A stamped table is checked and an unstamped one only counted."""
    notes = pd.DataFrame({"object_key": [1], "plate": ["plate1"],
                          "well": ["A01"], "text": ["x"]})
    write = _legacy if kind == "legacy" else _project
    write(tmp_path, {"cell": _cells([(2, "px"), (2, "px")]), "notes": notes})
    card = qc._read_units(str(tmp_path))
    assert (card.verdict, card.headline) == (
        "ok", "1 measurement table(s) are internally consistent in units "
              "and dimensionality.")


@pytest.mark.parametrize("kind", ["legacy", "store"])
def test_only_unstamped_tables_warn(tmp_path, kind):
    """Tables without the stamp columns give the warning card."""
    frame = pd.DataFrame({"object_key": [1, 2], "plate": ["plate1"] * 2,
                          "well": ["A01"] * 2, "cell_area": [1.0, 2.0]})
    write = _legacy if kind == "legacy" else _project
    write(tmp_path, {"cell": frame})
    card = qc._read_units(str(tmp_path))
    assert card.verdict == "warn"
    assert card.headline.startswith("1 table(s) carry no units stamp")


def test_a_store_wins_over_an_older_file_and_dates_from_its_catalog(tmp_path):
    """Beside an older file, the store project is the one read."""
    _project(tmp_path, {"cell": _cells([(2, "px"), (3, "um")])})
    with sqlite3.connect(tmp_path / "measurements" / "measurements.db") as db:
        _cells([(2, "px")]).to_sql("cell", db, index=False)
    path, mtime = qc._find_measurements_db(str(tmp_path))
    assert path == str(tmp_path / "measurements")
    assert mtime == os.stat(tmp_path / "measurements" / store.CATALOG).st_mtime
    assert qc._read_units(str(tmp_path)).verdict == "fail"


def test_a_file_that_is_not_a_database_is_an_error_card(tmp_path):
    """A measurements.db that does not open still says so."""
    path = tmp_path / "measurements" / "measurements.db"
    path.parent.mkdir()
    path.write_bytes(b"not a database at all, just bytes" * 10)
    card = qc._read_units(str(tmp_path))
    assert card.verdict == "error"
    assert card.source == str(path)


def test_no_project_says_none_was_found(tmp_path):
    """An empty folder has no measurements to check."""
    card = qc._read_units(str(tmp_path))
    assert card.headline == "No measurements database was found."
