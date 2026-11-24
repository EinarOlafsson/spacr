"""S7, first batch: library readers take measurement data from the shared store.

Every test builds an older ``measurements.db`` with sqlite3 and compares what
the migrated reader returns with what the reader it replaced returned, written
out here as the plain SQLite call it used to be. Frames are compared with their
dtypes.
"""
import pathlib
import re
import sqlite3

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

SPACR = pathlib.Path(__file__).resolve().parent.parent / "spacr"


def _db(tmp_path):
    path = tmp_path / "measurements.db"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, "
                    "fieldID TEXT, prcf TEXT, object_label INTEGER, "
                    "file_name TEXT, path_name TEXT, cell_area REAL, note)")
        rows = [("p1", "r1", "c1", "f1", "p1_r1_c1_f1", 1, "a.tif", "a.npy", 10.5, None),
                ("p1", "r1", "c2", "f1", "p1_r1_c2_f1", 2, "b.tif", "b.npy", None, "x"),
                ("p2", "r2", "c1", "f2", "p2_r2_c1_f2", 3, "c.tif", "c.npy", 7.0, 4)]
        con.executemany("INSERT INTO cell VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        con.execute("CREATE TABLE png_list (prcfo TEXT, png_path TEXT, plateID TEXT, "
                    "rowID TEXT, columnID TEXT)")
        con.executemany("INSERT INTO png_list VALUES (?,?,?,?,?)",
                        [("p1_r1_c1_f1_o1", "/x/A_cell.png", "p1", "r1", "c1"),
                         ("p1_r1_c2_f1_o2", "/x/B_cell.png", "p1", "r1", "c2"),
                         ("p2_r2_c1_f2_o3", "/x/C_nucleus.png", "p2", "r2", "c1")])
        con.execute("CREATE TABLE object_counts (file_name TEXT, count_type TEXT, "
                    "object_count INTEGER)")
        con.executemany("INSERT INTO object_counts VALUES (?,?,?)",
                        [("a", "cell", 3), ("a", "nucleus", 2), ("b", "cell", 5)])
        con.execute("CREATE TABLE settings_history (run_id TEXT, stage TEXT, "
                    "stamped_utc TEXT, setting_key TEXT, setting_value TEXT)")
        con.executemany("INSERT INTO settings_history VALUES (?,?,?,?,?)",
                        [("r1", "measure", "t1", "k", "1"), ("r1", "measure", "t1", "j", "2"),
                         ("r2", "classify", "t2", "k", "3")])
    return path


def _old(path, sql, params=None):
    with sqlite3.connect(path) as con:
        return pd.read_sql_query(sql, con, params=params)


def test_multi_database_plan_equals_the_sqlite_answers(tmp_path):
    from spacr import multi_database as md
    path = _db(tmp_path)
    plan = md.describe_merge([str(path)], "cell")
    source = plan.sources[0]
    with sqlite3.connect(path) as con:
        columns = [r[1] for r in con.execute('PRAGMA table_info("cell")')]
        rows = con.execute('SELECT COUNT(*) FROM "cell"').fetchone()[0]
        stored = sorted(str(r[0]) for r in con.execute('SELECT DISTINCT "plateID" FROM "cell"'))
    assert list(source.columns) == columns
    assert source.rows == rows
    assert list(source.stored_plates) == stored
    kinds = md.column_kinds(str(path), "cell")
    assert kinds["cell_area"] == "numeric"
    assert kinds["file_name"] == "text"
    assert kinds["note"] == "unknown"
    assert md.column_kinds(str(path), "missing") == {}


def test_merge_tables_reads_the_same_frame(tmp_path):
    from spacr import merge_tables as mt
    path = _db(tmp_path)
    assert set(mt.table_names(str(path))) == {"cell", "png_list", "object_counts",
                                              "settings_history"}
    assert_frame_equal(mt._read(str(path), "cell"), _old(path, 'SELECT * FROM "cell"'))


def test_object_counts_and_path_lists(tmp_path):
    from spacr import measure
    from spacr.utils import generate_path_list_from_db
    path = _db(tmp_path)
    run = tmp_path / "run"
    (run / "measurements").mkdir(parents=True)
    db = run / "measurements" / "measurements.db"
    path.rename(db)
    got = measure.get_object_counts(str(run))
    old = _old(db, "SELECT * FROM object_counts")
    expected = old.groupby("count_type").agg(
        total_object_count=("object_count", "sum"),
        avg_object_count_per_file_name=("object_count", "mean")).reset_index()
    assert_frame_equal(got, expected)
    assert generate_path_list_from_db(str(db), "a_cell") == ["/x/A_cell.png"]
    assert generate_path_list_from_db(str(db), ["A_cell", "C_nuc"]) == ["/x/A_cell.png", "/x/C_nucleus.png"]
    assert len(generate_path_list_from_db(str(db), None)) == 3


def test_io_readers(tmp_path):
    from spacr import io
    path = _db(tmp_path)
    got = io.read_settings_history(str(path))
    assert [r["run_id"] for r in got] == ["r1", "r2"]
    assert got[0]["settings"] == {"k": "1", "j": "2"}
    rows = io.crop_rows_from_object_table(str(path), "cell", verbose=False)
    old = _old(path, 'SELECT object_label, plateID, rowID, columnID, fieldID, prcf, '
                     'file_name, path_name FROM "cell"')
    assert_frame_equal(rows[list(old.columns)], old)
    assert io.crop_rows_from_object_table(str(path), "pathogen", verbose=False).empty
    assert io._read_db(str(path), ["cell"])[0].shape[0] == 3
    with pytest.raises(ValueError, match="Table not found"):
        io._read_db(str(path), ["nope"])


def test_derived_tables_and_original_filenames(tmp_path):
    from spacr import derived_tables as dt, original_filenames as of
    path = _db(tmp_path)
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE conversion_map (a TEXT, b TEXT)")
        con.execute("INSERT INTO conversion_map VALUES ('1', '2')")
    with sqlite3.connect(path) as con:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name")]
        expected = {n: [[r[1], r[2]] for r in con.execute(f'PRAGMA table_info("{n}")')]
                    for n in names}
    assert dt.schemas(str(path)) == expected
    assert_frame_equal(dt.column_sample(str(path), "cell", limit=2),
                       _old(path, 'SELECT * FROM "cell" LIMIT 2'))
    assert path.resolve() in of.discover_maps(path)
    payload, kind = of._map_payload(path)
    assert kind == "conversion_map" and b"1" in payload


def test_hit_investigation_crop_keys(tmp_path):
    from spacr import hit_investigation as hi
    from spacr.schema import normalise_plate_columns
    path = _db(tmp_path)
    got = hi._read_crop_keys(str(path), "png_path")
    assert_frame_equal(got, normalise_plate_columns(
        _old(path, 'SELECT "prcfo", "png_path" FROM "png_list"')))
    with sqlite3.connect(path) as con:
        con.execute("DROP TABLE png_list")
    with pytest.raises(hi.HitAttributionError, match="png_list"):
        hi._read_crop_keys(str(path), "png_path")


def test_run_compare_counts_and_assay_report(tmp_path):
    from spacr import assay_report as ar
    from spacr import run_compare as rc
    path = _db(tmp_path)
    counts = rc.count_database(str(path))
    assert counts.available and counts.overall["cell"] == 3
    assert counts.per_plate["p1"]["cell"] == 2
    assert ar.db_object_tables(str(path)) == ["cell"]
    frame = pd.DataFrame({"plateID": ["p1", "p2"], "rowID": ["r1", "r2"],
                          "columnID": ["c1", "c1"], "grp": ["a", "b"]})
    got = ar.db_well_counts(str(path), frame, "grp", tables=["cell"])
    assert got["objects_per_well"].tolist() == [1, 1]
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"not a database" * 50)
    assert not rc.count_database(str(bad)).available


def test_condition_annotation_receipt(tmp_path):
    from spacr import condition_annotations as ca
    path = _db(tmp_path)
    assert ca.saved_table_annotation(path, "cell", pd.DataFrame({"a": [1]})) is None


def test_no_reader_of_this_batch_opens_sqlite_on_its_own():
    """The modules this batch cleaned hold no direct opener of measurement data."""
    clean = ("multi_database.py", "merge_tables.py", "assay_report.py",
             "derived_tables.py", "hit_investigation.py", "_lineage_measurements.py",
             "original_filenames.py")
    opener = re.compile(r"sqlite3\.connect\(|duckdb\.connect\(|read_parquet\(")
    for name in clean:
        text = (SPACR / name).read_text(encoding="utf-8")
        assert not opener.findall(text), name
