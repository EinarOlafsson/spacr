"""Table-driven tests of releasing an import's canonical copy and merging conversion maps.

``spacr.foreign.release_canonical_copy`` removes an import's rows from
spaCR's own object table on every database state an import really leaves
behind (provenance tables gone, an import record too old to carry the claim,
a column SQLite refuses to drop) and drops an imported column only when the
release left it empty -- never one spaCR also writes into.
``_populate_conversion_map`` merges a map under one transaction: a merge
that cannot be applied leaves the old map, and a map without ``target``
never reaches the merge.  One real import (two fields, two channels, one
mask class) is shared by the module; every test mutates a private copy.
"""
from __future__ import annotations

import os
import shutil
import sqlite3

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import convert as cv
from spacr import foreign as fg
from spacr.errors import ConfigurationError

SIZE = 24


def _write(path, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(path, array)


def _their_folder(root):
    """Two fields x two channels of TIFFs, one cell mask per field, and a results table."""
    images = os.path.join(root, "their_images")
    masks = os.path.join(root, "their_cell_masks")
    label = np.zeros((SIZE, SIZE), np.uint16)
    label[2:8, 2:8] = 1
    label[12:20, 12:20] = 2
    for field in (1, 2):
        for channel, value in ((1, 11), (2, 12)):
            _write(os.path.join(images, f"fov{field:02d}_C{channel}.tif"),
                   np.full((SIZE, SIZE), value, np.uint16))
        _write(os.path.join(masks, f"fov{field:02d}_cell_mask.tif"), label)
    table = os.path.join(root, "results.csv")
    pd.DataFrame([{"ImageNumber": f"fov{field:02d}_C1.tif", "ObjectNumber": obj,
                   "AreaShape_Area": 36.0 * obj,
                   "Metadata_Treatment": "wt" if field == 1 else "ko"}
                  for field in (1, 2) for obj in (1, 2)]).to_csv(table, index=False)
    return images, {"cell": masks}, table


def _sql(db_path, *statements):
    """Run ``statements`` in order, commit, and return the last one's rows."""
    connection = sqlite3.connect(db_path)
    try:
        rows = None
        for statement in statements:
            rows = connection.execute(statement).fetchall()
        connection.commit()
        return rows
    finally:
        connection.close()


def _tables(db_path):
    return sorted(r[0] for r in _sql(
        db_path, "SELECT name FROM sqlite_master WHERE type IN ('table','view')"))


def _columns(db_path, table):
    return [r[1] for r in _sql(db_path, f'PRAGMA table_info("{table}")')]


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    """One finished import: ``(database path, conversion map path)``."""
    root = str(tmp_path_factory.mktemp("their_data"))
    images, masks, table = _their_folder(root)
    plan = fg.plan_import(images, masks, table, um_per_px=0.5)
    result = fg.run_import(plan, os.path.join(root, "imported"))
    return result.db_path, result.conversion.map_path


_CUT_PROVENANCE = ("DROP TABLE foreign_columns", "DROP TABLE foreign_import")
_OLD_RECORD = ("DROP TABLE foreign_columns", "DROP TABLE foreign_import",
               "CREATE TABLE foreign_import (dst TEXT, object_types TEXT)",
               "INSERT INTO foreign_import VALUES ('/old/dst', 'cell')")
_INDEXED = _CUT_PROVENANCE + (
    "CREATE INDEX idx_keep ON cell(foreign_areashape_area)",)

# (case id, statements run first, imported column still on `cell`, foreign_import rows after)
_RELEASE_CASES = [
    ("provenance tables gone", _CUT_PROVENANCE, False, None),
    ("import record too old to carry the claim", _OLD_RECORD, False,
     [("/old/dst", "cell")]),
    ("current import record has its claim withdrawn", ("DROP TABLE foreign_columns",),
     False, [(0, "released back to spaCR; their rows are in foreign_cell only")]),
    ("an indexed column is kept and the release stands", _INDEXED, True, None),
]


def test_release_canonical_copy_removes_the_rows_on_every_database_state(imported, tmp_path, subtests):
    """Four rows go from ``cell``, all four stay in ``foreign_cell``; the record is un-claimed if it can be."""
    db_path, _map = imported
    for index, (case_id, prepare, column_kept, record) in enumerate(_RELEASE_CASES):
        with subtests.test(msg=case_id):
            copy = str(tmp_path / f"m{index}.db")
            shutil.copy(db_path, copy)
            _sql(copy, *prepare)
            assert _sql(copy, "SELECT COUNT(*) FROM cell") == [(4,)]

            assert fg.release_canonical_copy(copy, "cell") == 4

            assert _sql(copy, "SELECT COUNT(*) FROM cell") == [(0,)]
            assert _sql(copy, "SELECT COUNT(*) FROM foreign_cell") == [(4,)]
            assert sorted(r[0] for r in _sql(
                copy, "SELECT foreign_areashape_area FROM foreign_cell")) == [
                36.0, 36.0, 72.0, 72.0]
            assert "cell_with_foreign" in _tables(copy)
            assert ("foreign_areashape_area" in _columns(copy, "cell")) is column_kept
            assert "foreign_areashape_area" in _columns(copy, "foreign_cell")
            if record is not None:
                columns = ("dst, object_types" if case_id.startswith("import record")
                           else "canonical_table_written, canonical_table_note")
                assert _sql(copy, f"SELECT {columns} FROM foreign_import") == record


def test_release_canonical_copy_keeps_a_column_spacr_also_writes(tmp_path):
    """A ColumnMap onto spaCR's ``cell_area`` keeps the column and spaCR's rows; an emptied import column goes."""
    from spacr.utils import _merge_and_save_to_database

    images, masks, table = _their_folder(str(tmp_path))
    plan = fg.plan_import(images, masks, table, um_per_px=0.5, column_maps=[
        fg.ColumnMap(source="AreaShape_Area", target="cell_area"),
        fg.ColumnMap(source="Metadata_Treatment", target="foreign_metadata_treatment")],
        allow_spacr_targets=True)
    dst = str(tmp_path / "imported")
    db_path = fg.run_import(plan, dst).db_path
    _merge_and_save_to_database(
        pd.DataFrame({"label": [1, 2], "cell_area": [11.0, 22.0]}),
        pd.DataFrame({"label": [1, 2], "cell_channel_0_mean_intensity": [3.0, 4.0]}),
        "cell", dst, "plate1_A01_9", "spacr_run")
    assert _sql(db_path, "SELECT COUNT(*) FROM cell") == [(6,)]

    assert fg.release_canonical_copy(db_path, "cell") == 4

    assert _sql(db_path, "SELECT prcf, cell_area FROM cell") == [
        ("plate1_r1_c1_f9", 11.0), ("plate1_r1_c1_f9", 22.0)]
    columns = _columns(db_path, "cell")
    assert "cell_area" in columns
    assert "foreign_metadata_treatment" not in columns


def test_populate_conversion_map_merges_under_one_transaction(imported, tmp_path):
    """A refused merge keeps the old map and leaves staging; the retry clears it; no ``target`` never merges."""
    _db, map_path = imported
    original = str(tmp_path / "already_converted.db")
    cv.populate_db_from_map(original, map_path)
    held = _sql(original, "SELECT target FROM conversion_map")
    assert len(held) == 4

    incoming = pd.read_csv(map_path)
    incoming["target"] = [str(t).replace("plate1", "plateX") for t in incoming["target"]]
    incoming_path = str(tmp_path / "incoming_map.csv")
    incoming.to_csv(incoming_path, index=False)
    _sql(original, "CREATE UNIQUE INDEX uq_source ON conversion_map(source)")

    with pytest.raises(sqlite3.IntegrityError):
        fg._populate_conversion_map(original, incoming_path)
    assert _sql(original, "SELECT target FROM conversion_map") == held
    assert fg._CONVERSION_STAGING in _tables(original)

    _sql(original, "DROP INDEX uq_source")
    fg._populate_conversion_map(original, incoming_path)
    targets = sorted(r[0] for r in _sql(original, "SELECT target FROM conversion_map"))
    assert targets == sorted([t for (t,) in held] + incoming["target"].tolist())
    assert fg._CONVERSION_STAGING not in _tables(original)

    headless = str(tmp_path / "no_target.csv")
    incoming.drop(columns=["target"]).to_csv(headless, index=False)
    with pytest.raises(ConfigurationError) as raised:
        fg._populate_conversion_map(original, headless)
    assert str(raised.value) == (
        f"{headless} is not a spaCR conversion map — missing column(s): target")
    assert sorted(r[0] for r in _sql(original, "SELECT target FROM conversion_map")) == targets
