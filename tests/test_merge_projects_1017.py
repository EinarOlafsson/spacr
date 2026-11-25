"""Merging tables from several projects, as the Database Browser does (item 1017).

:func:`spacr.plate_measurements.merge_projects` stacks one table from every
project or joins the object tables on their keys; two projects that hold a
plate of the same name stay apart through the project label, and every row
key stays unique. :func:`spacr.plate_measurements.save_merge` writes the same
table into a store project, streaming a stack a chunk at a time.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from spacr import measurement_store, plate_measurements, store
from spacr.multi_database import SCREEN_COLUMN, SOURCE_COLUMN, MergeRefused
from spacr.plate_measurements import merge_projects, save_merge

IDENTITY = ["plateID", "rowID", "columnID", "fieldID", "object_label"]


def _cells(plate, column, count, area=1.0):
    """``count`` cells of one well, with an area and a text column."""
    labels = np.arange(1, count + 1)
    return pd.DataFrame({
        "plateID": [plate] * count, "rowID": ["r1"] * count,
        "columnID": [column] * count, "fieldID": ["f1"] * count,
        "object_label": labels,
        "prcfo": [f"{plate}_r1_{column}_f1_o{i}" for i in labels],
        "cell_area": labels * area,
        "condition": ["NT" if i % 2 else "g1" for i in labels]})


def _nuclei(cells):
    """Two nuclei per cell, the areas of each pair summing to the cell area."""
    rows = []
    for _, cell in cells.iterrows():
        for half in (0.25, 0.75):
            rows.append({**{key: cell[key] for key in IDENTITY[:4]},
                         "object_label": len(rows) + 1,
                         "cell_id": cell["object_label"],
                         "nucleus_area": cell["cell_area"] * half})
    return pd.DataFrame(rows)


def _legacy(root, name, plate="plate1", column="c1", count=6, area=1.0,
            extra=None):
    """An older measurements.db with a cell and a nucleus table."""
    path = root / name / "measurements" / "measurements.db"
    path.parent.mkdir(parents=True)
    cells = _cells(plate, column, count, area)
    if extra:
        cells[extra] = 7.0
    with sqlite3.connect(path) as con:
        cells.to_sql("cell", con, index=False)
        _nuclei(cells).to_sql("nucleus", con, index=False)
    return str(path)


def _store_project(root, name, plate="plate1", column="c2", count=4, area=2.0):
    """A store project with Measure-shaped cell and nucleus tables."""
    project = root / name / "measurements"
    project.mkdir(parents=True)
    cells = _cells(plate, column, count, area)
    measurement_store.write_rows(str(project), "cell", cells)
    measurement_store.write_rows(str(project), "nucleus", _nuclei(cells))
    return str(project)


@pytest.fixture(autouse=True)
def _fresh():
    """No shared store survives a test."""
    store.release_shared()
    yield
    store.release_shared()


def _texts(values):
    """Values as text, a blank as None, whatever the reader typed them as."""
    return [None if pd.isna(value) else str(value) for value in values]


def _sorted(frame, keys):
    """``frame`` in key order with a fresh index."""
    return frame.sort_values(keys, kind="stable").reset_index(drop=True)


def test_stack_puts_one_table_of_every_project_together(tmp_path):
    first = _legacy(tmp_path, "runA", plate="plate1", count=6)
    second = _legacy(tmp_path, "runB", plate="plate2", count=3, area=10.0)
    merged = merge_projects([first, second], "cell")
    assert len(merged) == 9
    assert merged.groupby(SOURCE_COLUMN).size().to_dict() == {"runA": 6, "runB": 3}
    assert not merged.duplicated(list(merged.attrs["key_columns"])).any()
    assert merged.attrs["kept_apart"] == {}
    assert merged.attrs["summary"][0].startswith("9 rows from 2 project(s)")
    two = merged[merged[SOURCE_COLUMN] == "runB"]
    assert two["cell_area"].tolist() == [10.0, 20.0, 30.0]


def test_join_rolls_children_up_and_stacks_the_projects(tmp_path):
    first = _legacy(tmp_path, "runA", plate="plate1", count=4)
    second = _legacy(tmp_path, "runB", plate="plate2", count=2)
    merged = merge_projects([first, second], ["cell", "nucleus"], how="join")
    assert len(merged) == 6
    assert merged.attrs["tables"] == ("cell", "nucleus")
    assert np.allclose(merged["nucleus_area"], merged["cell_area"])
    assert not merged.duplicated(list(merged.attrs["key_columns"])).any()


def test_one_project_joined_is_merge_tables_of_that_project(tmp_path):
    from spacr.merge_tables import merge_tables

    path = _legacy(tmp_path, "runA", count=5)
    merged = merge_projects([path], ["cell", "nucleus"], how="join")
    direct = merge_tables(path, ["cell", "nucleus"])
    assert len(merged) == len(direct)
    assert np.allclose(_sorted(merged, IDENTITY)["nucleus_area"],
                       _sorted(direct, IDENTITY)["nucleus_area"])


def test_two_projects_with_the_same_plate_name_stay_apart(tmp_path):
    first = _legacy(tmp_path, "runA", plate="plate1", count=5)
    second = _legacy(tmp_path, "runB", plate="plate1", count=5, area=3.0)
    with pytest.raises(MergeRefused):
        from spacr.multi_database import read_merged
        read_merged([first, second], "cell")
    for how, tables in (("stack", "cell"), ("join", ["cell", "nucleus"])):
        merged = merge_projects([first, second], tables, how=how,
                                labels=["before", "after"])
        assert len(merged) == 10, how
        assert set(merged[SOURCE_COLUMN]) == {"before", "after"}
        assert set(merged[SCREEN_COLUMN]) == {"before", "after"}
        keys = list(merged.attrs["key_columns"])
        assert SOURCE_COLUMN in keys
        assert not merged.duplicated(keys).any(), how
        assert merged.duplicated(["plateID", "rowID", "columnID", "fieldID",
                                  "object_label"]).sum() == 5
        assert merged.attrs["kept_apart"] == {"plate1": ("before", "after")}
        assert any("Plate plate1 is in before and after" in line
                   for line in merged.attrs["summary"])


def test_labels_are_checked(tmp_path):
    first = _legacy(tmp_path, "runA")
    second = _legacy(tmp_path, "runB", plate="plate2")
    with pytest.raises(MergeRefused, match="its own label"):
        merge_projects([first, second], "cell", labels=["same", "same"])
    with pytest.raises(MergeRefused, match="one label per project"):
        merge_projects([first, second], "cell", labels=["one"])
    with pytest.raises(MergeRefused, match="same project"):
        merge_projects([first, os.path.dirname(first)], "cell")
    with pytest.raises(MergeRefused, match="one table"):
        merge_projects([first, second], ["cell", "nucleus"])
    with pytest.raises(MergeRefused, match="has no table"):
        merge_projects([first, second], "pathogen")


def test_a_store_project_and_an_older_file_merge(tmp_path):
    legacy = _legacy(tmp_path, "runA", plate="plate1", column="c1", count=6)
    project = _store_project(tmp_path, "runB", plate="plate1", column="c1",
                             count=4)
    stacked = merge_projects([legacy, project], "cell")
    assert stacked.groupby(SOURCE_COLUMN).size().to_dict() == {"runA": 6, "runB": 4}
    assert stacked.attrs["kept_apart"] == {"plate1": ("runA", "runB")}
    assert not stacked.duplicated(list(stacked.attrs["key_columns"])).any()
    joined = merge_projects([legacy, project], ["cell", "nucleus"], how="join")
    assert len(joined) == 10
    assert np.allclose(joined["nucleus_area"], joined["cell_area"])


def test_keep_narrows_the_columns_and_holds_the_keys(tmp_path):
    first = _legacy(tmp_path, "runA", extra="cell_perimeter")
    second = _legacy(tmp_path, "runB", plate="plate2")
    merged = merge_projects([first, second], "cell", keep=["cell_area"])
    assert "cell_area" in merged.columns
    assert "condition" not in merged.columns
    assert set(IDENTITY) <= set(merged.columns)
    common = merge_projects([first, second], "cell", columns="common")
    assert "cell_perimeter" not in common.columns
    assert any("only some projects" in line for line in common.attrs["summary"])
    union = merge_projects([first, second], "cell")
    assert union["cell_perimeter"].isna().sum() == 6


def _saved(project, name):
    """A saved table read back through the shared reader, without its store columns."""
    frame = store.read_frame(project, table=name)
    return frame.drop(columns=[c for c in ("plate", "well", "merge_row")
                               if c in frame.columns])


@pytest.mark.parametrize("how, tables", [("stack", "cell"),
                                         ("join", ["cell", "nucleus"])])
def test_a_saved_merge_is_the_merge_in_code(tmp_path, how, tables):
    first = _legacy(tmp_path, "runA", plate="plate1", count=5)
    second = _store_project(tmp_path, "runB", plate="plate1", column="c1",
                            count=3)
    target = str(tmp_path / "merged" / "measurements")
    result = save_merge(target, "pooled", [first, second], tables, how=how,
                        labels=["A", "B"])
    assert result["rows"] == 8
    assert result["rows_per_source"] == {"A": 5, "B": 3}
    assert result["kept_apart"] == {"plate1": ("A", "B")}
    assert any(line.startswith("Saved as pooled") for line in result["summary"])
    in_code = merge_projects([first, second], tables, how=how, labels=["A", "B"])
    saved = _saved(target, "pooled")
    keys = list(result["key_columns"])
    assert keys == list(in_code.attrs["key_columns"])
    shared = [c for c in in_code.columns if c in saved.columns]
    assert set(in_code.columns) - set(shared) <= {"plate", "well", "object_key",
                                                  "crop_store", "crop_index"}
    left = _sorted(saved[shared], keys)
    right = _sorted(in_code[shared], keys)
    for column in shared:
        a, b = left[column], right[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            assert np.allclose(a.astype(float), b.astype(float), equal_nan=True), column
        else:
            assert _texts(a) == _texts(b), column
    assert not saved.duplicated(keys).any()


def test_a_saved_merge_pages_in_the_browser_backend(tmp_path):
    from spacr.store_browse import StoreDb

    first = _legacy(tmp_path, "runA", plate="plate1", count=7)
    second = _legacy(tmp_path, "runB", plate="plate1", count=7)
    target = _store_project(tmp_path, "runC", plate="plate9")
    save_merge(target, "pooled", [first, second], "cell")
    db = StoreDb(target)
    assert db.row_key("pooled") == ("keyset", ["plate", "well", "merge_row"])
    seen, after = [], None
    while True:
        _cols, rows, keys = db.chunk("pooled", ["merge_row"], limit=3, after=after)
        if not rows:
            break
        seen += [row[0] for row in rows]
        after = keys[-1]
    assert sorted(seen) == list(range(14))


def test_save_refuses_an_older_file_and_an_existing_name(tmp_path):
    first = _legacy(tmp_path, "runA")
    with pytest.raises(MergeRefused, match="older measurements.db"):
        save_merge(os.path.dirname(first), "pooled", [first], "cell")
    with pytest.raises(MergeRefused, match="cannot be a table name"):
        save_merge(str(tmp_path / "out"), "bad name", [first], "cell")
    project = _store_project(tmp_path, "runB")
    with pytest.raises(MergeRefused, match="already has a table named cell"):
        save_merge(project, "cell", [first], "cell")


def test_a_large_stack_streams_in_chunks(tmp_path, monkeypatch):
    rows = 120_000
    path = tmp_path / "big" / "measurements" / "measurements.db"
    path.parent.mkdir(parents=True)
    wells = np.array(["c1", "c2", "c3"])
    frame = pd.DataFrame({
        "plateID": "plate1", "rowID": "r1",
        "columnID": wells[np.arange(rows) % 3], "fieldID": "f1",
        "object_label": np.arange(rows) // 3 + 1,
        "cell_area": np.arange(rows, dtype=float)})
    with sqlite3.connect(path) as con:
        frame.to_sql("cell", con, index=False)
    project = _store_project(tmp_path, "small", plate="plate2", count=50)
    sizes = []
    original = plate_measurements._source_chunks

    def counted(*args, **kwargs):
        """Record the size of every chunk read."""
        for chunk in original(*args, **kwargs):
            sizes.append(len(chunk))
            yield chunk
    monkeypatch.setattr(plate_measurements, "_source_chunks", counted)
    import spacr.multi_database as multi_database
    monkeypatch.setattr(multi_database.tabular, "read_database",
                        lambda *a, **k: pytest.fail("a whole-table read"))
    target = str(tmp_path / "out" / "measurements")
    result = save_merge(target, "pooled", [str(path), project], "cell",
                        keep=["cell_area"], chunk_rows=10_000)
    assert result["rows"] == rows + 50
    assert max(sizes) <= 10_000
    assert sum(sizes) == 2 * (rows + 50)
    with store.open(target) as opened:
        assert opened.count("pooled") == rows + 50
        total = opened.sql('SELECT SUM(cell_area) AS s FROM "pooled" '
                           "WHERE source_database = 'big'")["s"].iloc[0]
        assert total == pytest.approx(frame["cell_area"].sum())
        columns = set(opened.columns("pooled")["name"])
    assert "condition" not in columns and "cell_area" in columns


def test_a_cancelled_save_leaves_nothing(tmp_path):
    from spacr.multi_database import MergeCancelled

    first = _legacy(tmp_path, "runA")
    target = str(tmp_path / "out" / "measurements")
    with pytest.raises(MergeCancelled):
        save_merge(target, "pooled", [first], "cell", cancelled=lambda: True)
    assert not os.path.isfile(os.path.join(target, store.CATALOG)) or (
        "pooled" not in store.open(target).tables())
