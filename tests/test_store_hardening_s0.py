"""Store hardening (step S0 of the unified store).

One schema per table (so the catalog views read without ``union_by_name``),
bounded row groups, one batched register with one view rebuild, the keyset
``Store.page`` and the registry ``Store.count``, the edit-log sequence
stamp, and the rest of the S0 row: NaN stored as null, project-relative view
paths, key and content digests, ``spacr_columns``, float16 read as float32,
``STORE_VERSION`` and the register lock file.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from spacr import store


def _well(n=50, seed=0, extra=None, first=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "object_key": np.arange(first, first + n, dtype=np.int64),
        "gene": rng.choice(np.array(["g1", "g2", None], dtype=object), n)})
    for j in range(4):
        frame[f"f{j}"] = rng.normal(size=n).astype(np.float32)
    for name, values in (extra or {}).items():
        frame[name] = values
    return frame


def _put(project, well, frame, **kwargs):
    store.write_table(project, "cell", frame, plate="P1", well=well,
                      sort_by=("object_key",), key_columns=("object_key",),
                      **kwargs)


def _views(project):
    conn = duckdb.connect(os.path.join(project, store.CATALOG),
                          read_only=True)
    try:
        return dict(conn.execute("SELECT view_name, sql FROM duckdb_views() "
                                 "WHERE NOT internal").fetchall())
    finally:
        conn.close()


# 1. one schema per table

def test_a_part_with_the_tables_columns_in_another_order_and_type_is_conformed(
        tmp_path):
    project = str(tmp_path)
    first = _well(seed=1)
    second = _well(seed=2, first=100)
    second = second[["f3", "f0", "object_key", "gene", "f1", "f2"]]
    second["f0"] = second["f0"].astype(np.float64)
    _put(project, "A1", first)
    _put(project, "B1", second)
    store.register(project)
    with store.open(project) as s:
        got = s.query(table="cell").sort_values("object_key")
    assert list(got.columns[:2]) == ["object_key", "gene"] or "object_key" in got
    schemas = {pq.read_schema(os.path.join(project, f["path"])).remove_metadata()
               for f in store.open(project)._files if f["table_name"] == "cell"}
    assert len({tuple((x.name, str(x.type)) for x in sc) for sc in schemas}) == 1
    expected = pd.concat([first, second]).sort_values("object_key")
    for column in ("f0", "f1", "f2", "f3"):
        np.testing.assert_allclose(
            got[column].to_numpy(dtype=np.float64),
            expected[column].to_numpy(dtype=np.float64), rtol=1e-6)
    assert "union_by_name=false" in _views(project)["cell"].replace(" ", "")


def test_a_part_with_a_column_the_table_lacks_is_refused(tmp_path):
    project = str(tmp_path)
    _put(project, "A1", _well(seed=1))
    store.register(project)
    _put(project, "B1", _well(seed=2, first=100, extra={"new": 1.0}))
    with pytest.raises(store.StoreError, match="new"):
        store.register(project)


def test_a_missing_column_becomes_null_and_rows_are_unchanged(tmp_path):
    project = str(tmp_path)
    _put(project, "A1", _well(seed=1))
    _put(project, "B1", _well(seed=2, first=100).drop(columns=["f2"]))
    store.register(project)
    with store.open(project) as s:
        got = s.query({"object_key": list(range(100, 150))}, table="cell")
    assert len(got) == 50 and got["f2"].isna().all()


def test_a_rewritten_well_may_change_the_whole_tables_schema(tmp_path):
    project = str(tmp_path)
    _put(project, "A1", _well(seed=1))
    store.register(project)
    _put(project, "A1", _well(seed=3, extra={"new": 2.0}))
    store.register(project)
    with store.open(project) as s:
        assert "new" in s.query(table="cell").columns


# 2. row groups

def test_row_groups_hold_at_most_8192_rows_and_64_mb():
    assert store._ROW_GROUP_ROWS == 8192
    assert store._ROW_GROUP_BYTES == 64 * 2 ** 20
    wide = pa.schema([pa.field(f"f{i}", pa.float32()) for i in range(2800)])
    assert store._row_group_rows(wide, None) <= 64 * 2 ** 20 // (2800 * 4)
    narrow = pa.schema([pa.field("a", pa.float32())])
    assert store._row_group_rows(narrow, None) == 8192


def test_a_part_is_cut_into_bounded_row_groups(tmp_path):
    project = str(tmp_path)
    _put(project, "A1", _well(n=20_000))
    store.register(project)
    with store.open(project) as s:
        path = os.path.join(project, next(
            f["path"] for f in s._files
            if f["table_name"] == "cell" and f["kind"] == "parquet"))
    part = pq.ParquetFile(path)
    sizes = [part.metadata.row_group(i).num_rows
             for i in range(part.metadata.num_row_groups)]
    assert max(sizes) <= 8192 and sum(sizes) == 20_000


def test_a_query_that_outgrows_the_cap_names_batches(tmp_path, monkeypatch):
    project = str(tmp_path)
    _put(project, "A1", _well(n=5000))
    store.register(project)
    monkeypatch.setattr(store, "_QUERY_BYTES", 1000)
    with store.open(project) as s:
        with pytest.raises(store.StoreError, match="batches"):
            s.query("SELECT * FROM cell", table="cell")
        assert sum(len(b) for b in s.batches(table="cell", as_="pandas",
                                              batch_size=999)) == 5000


# 3. one batched register, one view rebuild

def test_register_inserts_every_manifest_at_once_and_rebuilds_views_once(
        tmp_path, monkeypatch):
    project = str(tmp_path)
    for k in range(30):
        _put(project, f"W{k:02d}", _well(n=20, seed=k, first=100 * k))
    calls = []
    real = store._rebuild_views
    monkeypatch.setattr(store, "_rebuild_views",
                        lambda *a: (calls.append(a[2]), real(*a))[1])
    result = store.register(project)
    assert result["registered"] == 30 and len(calls) == 1
    with store.open(project) as s:
        assert s.count("cell") == 600
        assert len([f for f in s._files if f["table_name"] == "cell"]) == 30


def test_two_manifests_of_one_well_keep_only_the_last(tmp_path):
    project = str(tmp_path)
    _put(project, "A1", _well(n=10, seed=1))
    _put(project, "A1", _well(n=7, seed=2))
    store.register(project)
    with store.open(project) as s:
        assert s.count("cell") == 7
        assert len(s.query(table="cell")) == 7


# 4. Store.page and Store.count

def _paged(project, **kwargs):
    seen, cursor = [], None
    with store.open(project) as s:
        while True:
            frame, cursor = s.page("cell", after=cursor, **kwargs)
            seen.append(frame)
            if cursor is None:
                break
    return pd.concat(seen, ignore_index=True)


def test_pages_cover_every_row_once_in_key_order_across_wells(tmp_path):
    project = str(tmp_path)
    for k, well in enumerate(("A1", "A2", "B1")):
        _put(project, well, _well(n=130 + k, seed=k, first=1000 * k))
    store.register(project)
    got = _paged(project, columns=["f0"], limit=100)
    with store.open(project) as s:
        whole = s.query(table="cell")
    whole = whole.sort_values(["plate", "well", "object_key"]).reset_index(
        drop=True)
    assert len(got) == len(whole) == 130 + 131 + 132
    assert got[["plate", "well", "object_key"]].equals(
        whole[["plate", "well", "object_key"]])
    np.testing.assert_array_equal(got["f0"].to_numpy(), whole["f0"].to_numpy())


def test_a_filtered_page_and_an_empty_one(tmp_path):
    project = str(tmp_path)
    _put(project, "A1", _well(n=200, seed=1))
    store.register(project)
    got = _paged(project, columns=["gene"], where={"gene": "g1"}, limit=17)
    assert (got["gene"] == "g1").all() and len(got) > 0
    with store.open(project) as s:
        empty, cursor = s.page("cell", columns=["gene"],
                               where={"gene": "none"})
    assert len(empty) == 0 and cursor is None


def test_pages_sorted_by_a_feature_follow_the_sort_to_the_end(tmp_path):
    project = str(tmp_path)
    for k, well in enumerate(("A1", "B1")):
        _put(project, well, _well(n=150, seed=k, first=1000 * k))
    store.register(project)
    for direction in ("asc", "desc"):
        got = _paged(project, columns=["f1"], order_by=("f1", direction),
                     limit=64)
        values = got["f1"].to_numpy()
        assert len(got) == 300
        assert (np.diff(values) >= 0).all() if direction == "asc" else (
            np.diff(values) <= 0).all()


def test_a_cursor_of_another_table_is_refused(tmp_path):
    project = str(tmp_path)
    _put(project, "A1", _well(n=10))
    store.register(project)
    with store.open(project) as s:
        with pytest.raises(store.StoreError):
            s.page("cell", after={"plate": "P9", "well": "Z", "after": [1]})


def test_count_without_a_filter_reads_the_registry_only(tmp_path):
    project = str(tmp_path)
    _put(project, "A1", _well(n=40))
    _put(project, "B1", _well(n=60, first=500))
    store.register(project)
    with store.open(project) as s:
        s._conn = None  # a SQL count would need the connection
        assert s.count("cell") == 100
    with store.open(project) as s:
        assert s.count("cell", {"gene": "g1"}) == len(
            s.query({"gene": "g1"}, table="cell"))
        assert s.count("cell", "f0 > 0") == len(s.query("f0 > 0",
                                                        table="cell"))


def test_columns_come_from_the_catalog_record(tmp_path):
    project = str(tmp_path)
    _put(project, "A1", _well(n=10))
    store.register(project)
    with store.open(project) as s:
        columns = s.columns("cell")
    assert list(columns["name"][:2]) == ["object_key", "gene"]
    assert set(columns["name"]) >= {"f0", "f3", "plate", "well"}
    assert columns.set_index("name").loc["f0", "stored_type"] == "float"


# 5. the edit-log stamp

def test_the_edit_stamp_is_two_columns_and_strictly_increasing():
    int64 = np.iinfo(np.int64).max
    stamps = [store._edit_stamp(3) for _ in range(1000)]
    assert stamps == sorted(stamps) and len(set(stamps)) == 1000
    assert all(ns < int64 and writer == 3 for ns, writer in stamps)
    assert time.time_ns() * 1000 > int64  # the one-number form overflows
    schema = store._edit_schema()
    assert schema.field("seq_ns").type == pa.int64()
    assert schema.field("seq_writer").type == pa.int32()
    assert "seq" not in schema.names
    assert store._edit_stamp(4)[1] == 4


# 6. the rest of the S0 row

def test_nan_is_stored_as_null_and_inf_stays(tmp_path):
    project = str(tmp_path)
    frame = _well(n=6)
    frame.loc[1, "f0"] = np.nan
    frame.loc[2, "f0"] = np.inf
    _put(project, "A1", frame)
    store.register(project)
    path = os.path.join(project, store.open(project)._files[0]["path"])
    column = pq.read_table(path).column("f0")
    assert column.null_count == 1 and column[2].as_py() == np.inf


def test_catalog_views_use_project_relative_paths_and_open_from_the_project(
        tmp_path):
    project = str(tmp_path)
    _put(project, "A1", _well(n=25))
    store.register(project)
    assert str(tmp_path) not in _views(project)["cell"]
    code = ("import duckdb;c=duckdb.connect('spacr.duckdb',read_only=True);"
            "print(c.execute('select count(*) from cell').fetchone()[0])")
    out = subprocess.run([sys.executable, "-c", code], cwd=project,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "25"


def test_files_record_key_content_and_schema_digests(tmp_path):
    project = str(tmp_path)
    frame = _well(n=40)
    _put(project, "A1", frame)
    store.register(project)
    conn = duckdb.connect(os.path.join(project, store.CATALOG),
                          read_only=True)
    rows = conn.execute("SELECT path, table_name, key_digest, content_digest,"
                        " sha256, schema_digest, dtype_map_version FROM "
                        "spacr_files").fetchall()
    conn.close()
    cell = [r for r in rows if r[1] == "cell"][0]
    keys = [r for r in rows if r[1] == "cell__keys"][0]
    assert cell[3] == cell[4] == hashlib.sha256(
        open(os.path.join(project, cell[0]), "rb").read()).hexdigest()
    assert cell[2] == keys[2] and len(cell[2]) == 64
    assert cell[5] and keys[5]
    # the same keys give the same digest in another project
    other = str(tmp_path / "other")
    store.write_table(other, "cell", frame, plate="P1", well="A1",
                      sort_by=("object_key",), key_columns=("object_key",))
    store.register(other)
    conn = duckdb.connect(os.path.join(other, store.CATALOG), read_only=True)
    assert conn.execute("SELECT key_digest FROM spacr_files WHERE "
                        "table_name = 'cell'").fetchone()[0] == cell[2]
    conn.close()


def test_a_rerun_writes_byte_identical_parts_and_digests(tmp_path):
    digests = []
    for name in ("a", "b"):
        project = str(tmp_path / name)
        _put(project, "A1", _well(n=500, seed=4))
        store.register(project)
        conn = duckdb.connect(os.path.join(project, store.CATALOG),
                              read_only=True)
        digests.append(conn.execute(
            "SELECT path, content_digest, key_digest, schema_digest FROM "
            "spacr_files ORDER BY path").fetchall())
        conn.close()
    assert digests[0] == digests[1]


def test_spacr_columns_lists_the_tables_columns_in_order(tmp_path):
    project = str(tmp_path)
    _put(project, "A1", _well(n=10))
    store.register(project)
    conn = duckdb.connect(os.path.join(project, store.CATALOG),
                          read_only=True)
    rows = conn.execute("SELECT column_name, stored_type, compute_type FROM "
                        "spacr_columns WHERE table_name = 'cell' ORDER BY "
                        "ordinal").fetchall()
    conn.close()
    assert [r[0] for r in rows][:2] == ["object_key", "gene"]
    assert dict((r[0], r[1]) for r in rows)["object_key"] == "int64"


def test_float16_columns_are_read_as_float32_everywhere(tmp_path):
    project = str(tmp_path)
    frame = _well(n=30)
    half = pa.table({"object_key": pa.array(frame["object_key"]),
                     "h": pa.array(frame["f0"].to_numpy().astype(np.float16))})
    _put(project, "A1", half)
    store.register(project)
    with store.open(project) as s:
        assert s.query(table="cell")["h"].dtype == np.float32
        assert s.sql("SELECT h FROM cell", as_="arrow").schema.field(
            "h").type == pa.float32()
        assert next(s._frames("cell"))["h"].dtype == np.float32
        batch = next(s.batches(table="cell", as_="arrow", batch_size=10))
        assert batch.schema.field("h").type == pa.float32()
        got = s.page("cell", columns=["h"])[0]
        assert got["h"].dtype == np.float32


def test_store_version_is_written_and_a_newer_store_is_refused(tmp_path):
    project = str(tmp_path)
    _put(project, "A1", _well(n=5))
    store.register(project)
    path = os.path.join(project, store.STORE_VERSION_FILE)
    assert open(path).read().strip() == str(store.SCHEMA_VERSION)
    open(path, "w").write(f"{store.SCHEMA_VERSION + 1}\n")
    with pytest.raises(store.StoreError, match="version"):
        store.open(project)


def test_register_waits_for_a_held_lock_and_takes_over_a_stale_one(tmp_path):
    project = str(tmp_path)
    _put(project, "A1", _well(n=5))
    lock = os.path.join(project, store._LOCK)
    open(lock, "w").write(str(os.getpid()))  # held by a live process
    started = time.monotonic()
    with pytest.raises(store.StoreError, match="registering"):
        with store._register_lock(project, wait=0.4):
            pass
    assert time.monotonic() - started >= 0.4
    os.utime(lock, (time.time() - 3600,) * 2)
    open(lock, "w").write("999999999")  # a process that is gone
    os.utime(lock, (time.time() - 3600,) * 2)
    store.register(project)
    assert not os.path.exists(lock)
    with store.open(project) as s:
        assert s.count("cell") == 5


def test_the_lock_is_released_when_register_refuses(tmp_path):
    project = str(tmp_path)
    _put(project, "A1", _well(n=5))
    store.register(project)
    _put(project, "B1", _well(n=5, first=50, extra={"new": 1.0}))
    with pytest.raises(store.StoreError):
        store.register(project)
    assert not os.path.exists(os.path.join(project, store._LOCK))


# the arrival contract of the S0 public API (2026-11-23)

S0_BIRTH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                        "release_contracts",
                        "s0_store_hardening_api_arrivals_2026-11-23.json")
S0_ARRIVALS = frozenset({
    "spacr.store.Store.columns", "spacr.store.Store.count",
    "spacr.store.Store.page", "spacr.power_ops.nominal_calibration",
    "spacr.power_ops.write_ops_screen"})


def _s0_source_docstring(symbol):
    """The docstring of one arrival read from its source file, so no module is imported."""
    import ast
    import inspect

    parts = symbol.split(".")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "spacr", parts[1] + ".py")
    with open(path, encoding="utf-8") as handle:
        scope = ast.parse(handle.read())
    for name in parts[2:]:
        scope = next(node for node in scope.body
                     if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                     and node.name == name)
    return inspect.cleandoc(ast.get_docstring(scope, clean=False))


def _s0_birth_arrivals(existing=None):
    """The S0 public symbols, bound to their source docstrings."""
    import json

    with open(S0_BIRTH, encoding="utf-8") as handle:
        entries = json.load(handle)
    assert set(entries) == S0_ARRIVALS
    for symbol, body in entries.items():
        assert body == _s0_source_docstring(symbol), symbol
    if existing is not None:
        assert not entries.keys() & existing.keys()
    return entries


def _validated_s0_arrivals(callables):
    """Remove only the S0 exact new callables from the frozen boundary."""
    entries = _s0_birth_arrivals()
    found = {item.symbol: item for item in callables
             if item.symbol in entries}
    assert set(found) == S0_ARRIVALS
    for symbol, item in found.items():
        assert item.docstring == entries[symbol], symbol
    return [item for item in callables if item.symbol not in entries]


def test_the_s0_arrivals_are_exactly_the_new_public_api():
    assert len(_s0_birth_arrivals()) == 5
