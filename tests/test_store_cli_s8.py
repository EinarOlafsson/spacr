"""S8: ``spacr store compact|verify|compare|export|ui|migrate`` on small projects."""
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import store
from spacr.cli_store import main


def _frame(scale=1.0):
    rows = 6
    return pd.DataFrame({
        "prcfo": [f"p1_r1_c1_f1_o{i}" for i in range(1, rows + 1)],
        "object_key": np.arange(1, rows + 1, dtype=np.int64),
        "area": np.arange(rows, dtype="float64") * scale,
        "name": list("abcdef")})


def _project(path, scale=1.0):
    path.mkdir()
    store.write_table(str(path), "cell", _frame(scale), plate="p1", well="r1c1",
                      run_id="r")
    store.register(str(path))
    return path


def _db(folder):
    folder.mkdir()
    frame = _frame().assign(plateID="p1", rowID="r1", columnID="c1", fieldID="f1")
    with sqlite3.connect(folder / "measurements.db") as con:
        frame.to_sql("cell", con, index=False)
    return folder / "measurements.db"


def test_verify_checks_the_catalog_and_catches_a_changed_file(tmp_path, capsys):
    project = _project(tmp_path / "a")
    assert main(["verify", str(project)]) == 0
    assert "catalog files checked: 2" in capsys.readouterr().out
    assert main(["verify", str(project), "--full", "--json"]) == 0
    part = next((project / "tables").rglob("*.parquet"))
    raw = bytearray(part.read_bytes())
    raw[8] ^= 1
    part.write_bytes(bytes(raw))
    assert main(["verify", str(project), "--full"]) == 1
    assert "changed outside spaCR" in capsys.readouterr().err


def test_compact_folds_the_edit_log(tmp_path, capsys):
    project = _project(tmp_path / "a")
    store.edit(str(project), [{"value": "x"}], kind="annotation",
               source="test", table="cell", column="label", every_row=True)
    assert main(["compact", str(project), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["rows"] >= 1 and set(result) == {"journals", "files", "rows"}


def test_compare_reports_equal_and_different_projects(tmp_path, capsys):
    one, two = _project(tmp_path / "a"), _project(tmp_path / "b")
    assert main(["compare", str(one), str(two)]) == 0
    assert "OK: the projects agree." in capsys.readouterr().out
    three = _project(tmp_path / "c", scale=2.0)
    assert main(["compare", str(one), str(three), "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    column = result["tables"]["cell"]["columns"]["area"]
    assert column["max_diff"] == 5.0 and column["differing"] == 5
    assert result["tables"]["cell"]["columns"]["name"]["ok"] is True


def test_export_writes_sqlite_csv_and_parquet(tmp_path, capsys):
    project = _project(tmp_path / "a")
    target = tmp_path / "out.sqlite"
    assert main(["export", str(project), str(target), "--json"]) == 0
    with sqlite3.connect(target) as con:
        assert con.execute("SELECT COUNT(*) FROM cell").fetchone()[0] == 6
    assert main(["export", str(project), str(tmp_path / "c.csv"), "--table", "cell"]) == 0
    assert len(pd.read_csv(tmp_path / "c.csv")) == 6
    assert main(["export", str(project), str(tmp_path / "c.parquet"), "--table", "cell"]) == 0
    assert len(pd.read_parquet(tmp_path / "c.parquet")) == 6
    capsys.readouterr()
    assert main(["export", str(project), str(tmp_path / "c.txt")]) == 1


def test_ui_prints_the_duckdb_command(tmp_path, capsys):
    project = _project(tmp_path / "a")
    assert main(["ui", str(project), "--print"]) == 0
    assert capsys.readouterr().out.strip() == f"duckdb -readonly {project / store.CATALOG}"
    assert main(["ui", str(tmp_path), "--print"]) == 1


def test_migrate_then_export_gives_the_same_rows(tmp_path, capsys):
    db = _db(tmp_path / "old")
    assert main(["migrate", str(db), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["tables"]["cell"] == {"source": 6, "store": 6}
    target = tmp_path / "back.sqlite"
    assert main(["export", str(tmp_path / "old"), str(target)]) == 0
    with sqlite3.connect(db) as old, sqlite3.connect(target) as new:
        before = pd.read_sql_query("SELECT prcfo, area, name FROM cell ORDER BY prcfo", old)
        after = pd.read_sql_query("SELECT prcfo, area, name FROM cell ORDER BY prcfo", new)
    pd.testing.assert_frame_equal(before, after)
    assert main(["compare", str(tmp_path / "old"), str(tmp_path / "old")]) == 0


def test_migrate_refuses_a_folder_without_a_database(tmp_path):
    with pytest.raises(store.StoreError):
        store.migrate_sqlite(tmp_path)
    assert main(["migrate", str(tmp_path)]) == 1
