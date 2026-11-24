"""The spaCR store (item 974, alpha): Parquet tables, Zarr crops, one catalog.

Each case writes a small store and reads it back: the values and the crops
come back identical and in row order, a well rewritten replaces its earlier
files, a write that stops before its manifest is never seen, a file changed
outside spaCR is caught, an older SQLite project opens through the same
API, and the OPS run writes the same rows and crops as ``ph_tables`` and
``ph_crops`` when ``ops_storage`` is ``spacr_store`` and nothing new when it
is not.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")
pytest.importorskip("zarr")

from spacr import ops_engine, store
from tests.test_ops_phenotype_run_v2 import FULL, PLATE, _run, measured, models

__all__ = ["measured", "models"]


def _frame(well: str, n: int = 300, seed: int = 0) -> pd.DataFrame:
    """A well's cells: ids, genes with nulls, keys and 12 float32 features."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "cell_id": np.arange(n, dtype=np.int64) + 10_000 * (ord(well[0]) - 64),
        "gene": rng.choice(np.array(["g1", "g2", "g3", None], dtype=object), n),
        "prcfo": [f"P_{well}_{i}" for i in range(n)]})
    for j in range(12):
        values = rng.normal(size=n).astype(np.float32)
        values[rng.random(n) < 0.05] = np.nan
        frame[f"f{j}"] = values
    return frame


def _crops(n: int, seed: int = 1) -> np.ndarray:
    """``n`` raw uint16 crops of 2 x 8 x 8."""
    return np.random.default_rng(seed).integers(0, 65535, (n, 2, 8, 8),
                                                dtype=np.uint16)


def _write(project, wells=("A1", "B2"), crops=True):
    """Write and register the wells; returns ``{well: (frame, crops, rows)}``."""
    written = {}
    for k, well in enumerate(wells):
        frame = _frame(well, seed=k)
        rows = np.arange(0, len(frame), 3)
        stack = _crops(len(rows), seed=k) if crops else None
        store.write_part(project, "cells", frame, plate="pl/1", well=well,
                         key_columns=("prcfo", "gene"), crops=stack,
                         crop_rows=rows if crops else None)
        written[well] = (frame, stack, rows)
    store.register(project, run={"run_id": "r", "settings": {"x": 1}})
    return written


def _same(found: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Every column of ``expected`` equal in ``found``, NaN equal to NaN."""
    found = found.sort_values("cell_id").reset_index(drop=True)
    expected = expected.sort_values("cell_id").reset_index(drop=True)
    assert len(found) == len(expected)
    for column in expected.columns:
        left, right = found[column].to_numpy(), expected[column].to_numpy()
        if right.dtype.kind == "f":
            assert np.array_equal(left, right, equal_nan=True), column
            assert left.dtype == right.dtype, column
        else:
            assert [None if pd.isna(v) else v for v in left] == [
                None if pd.isna(v) else v for v in right], column


def test_values_and_crops_come_back_identical_in_row_order(tmp_path):
    written = _write(tmp_path)
    with store.open(tmp_path, threads=2, verify="full") as opened:
        assert opened.tables() == ["cells"]
        every = opened.query()
        _same(every, pd.concat([w[0] for w in written.values()]))
        assert set(every["plate"]) == {"pl/1"}
        frame, crops, rows = written["A1"]
        result, stack = opened.query({"well": "A1"}, features=["f3"],
                                     crops=True)
        assert list(result.columns) == ["prcfo", "gene", "plate", "well",
                                        "cell_id", "crop_store",
                                        "crop_index", "f3"]
        expected = {frame["cell_id"].iloc[r]: crops[i]
                    for i, r in enumerate(rows)}
        for position, cell in enumerate(result["cell_id"]):
            want = expected.get(cell)
            if want is None:
                assert pd.isna(result["crop_store"].iloc[position])
                assert not stack[position].any()
            else:
                assert np.array_equal(stack[position], want)
        genes = result["gene"].tolist()
        assert genes == sorted(genes, key=lambda g: (g is None, g or ""))
        (meta, matrix), _ = opened.query("gene = 'g2'", features=["f0", "f1"],
                                         crops=True, as_="numpy")
        assert matrix.dtype == np.float32 and matrix.shape == (len(meta), 2)
        assert opened.sql("SELECT COUNT(*) AS n FROM cells")["n"][0] == 600


def test_batches_cover_every_row_once_and_shuffle_reproducibly(tmp_path):
    _write(tmp_path)
    with store.open(tmp_path) as opened:
        plain = [m for m, _x in opened.batches(batch_size=64)]
        ids = pd.concat(plain)["cell_id"]
        assert len(ids) == 600 and ids.is_unique
        first = [m["cell_id"].tolist() for (m, _x), _c in opened.batches(
            batch_size=50, shuffle=True, seed=3, crops=True, buffer=2)]
        again = [m["cell_id"].tolist() for (m, _x), _c in opened.batches(
            batch_size=50, shuffle=True, seed=3, crops=True, buffer=2)]
        assert first == again
        assert sorted(sum(first, [])) == sorted(ids.tolist())
        assert sum(first, []) != ids.tolist()


def test_key_table_routes_answer_as_the_wide_table_does(tmp_path):
    _write(tmp_path)
    with store.open(tmp_path, threads=2) as opened:
        wide = opened._select({"gene": "g2"}, "cells", [])
        direct = store._arrow_of(opened._conn, *wide).to_pandas()
        routed = opened._routed({"gene": "g2"}, "cells", [])
        assert routed is not None and "_part" not in routed.column_names
        pd.testing.assert_frame_equal(
            routed.to_pandas().sort_values("cell_id").reset_index(drop=True),
            direct.sort_values("cell_id").reset_index(drop=True))
        ids = [10_005, 10_250, 20_017, 20_299, 10_003]
        sql, params = opened._select({"cell_id": ids}, "cells", None)
        expected = store._arrow_of(opened._conn, sql, params).to_pandas()
        located = opened._routed({"cell_id": ids}, "cells", None)
        assert located is not None and located.num_rows == 5
        pd.testing.assert_frame_equal(
            located.to_pandas().sort_values("cell_id").reset_index(drop=True),
            expected.sort_values("cell_id").reset_index(drop=True),
            check_dtype=False)
        projected = opened.query({"cell_id": ids}, features=["f1", "f7"])
        projected = projected.set_index("cell_id").loc[expected["cell_id"]]
        assert np.array_equal(projected["f7"].to_numpy(),
                              expected["f7"].to_numpy(), equal_nan=True)
        assert opened._routed("f3 > 0", "cells", []) is None
        assert opened._wanted("cells", "cells", ["cell_id", "f1"]).count(
            "cell_id") == 1
        assert len(opened.query({"cell_id": [1]})) == 0
        assert opened.tables() == ["cells"]
        frame, stack = opened.query({"well": "B2"}, features=[], crops=True)
        reader = opened._array(frame["crop_store"].dropna().iloc[0])
        assert isinstance(reader, store._ChunkReader)
    import zarr

    rel = frame["crop_store"].dropna().iloc[0]
    reference = zarr.open_array(str(tmp_path / rel / "crops"), mode="r")
    index = frame["crop_index"].dropna().astype(int).to_numpy()
    assert np.array_equal(stack[frame["crop_index"].notna().to_numpy()],
                          np.asarray(reference)[index])


def _bytes(project):
    """``{relative path: bytes}`` of a store's table files."""
    return {str(p.relative_to(project)): p.read_bytes()
            for p in sorted((project / "tables").rglob("*.parquet"))}


def test_the_row_copy_is_off_by_default_and_exact_when_on(tmp_path):
    frame = _frame("A1", n=500)
    frame["f_wide"] = np.linspace(0, 1, 500)
    for name, extra in (("default", {}), ("off", {"row_copy": False}),
                        ("on", {"row_copy": True})):
        store.write_part(tmp_path / name, "cells", frame, plate="P",
                         well="A1", key_columns=("prcfo", "gene"), **extra)
        store.register(tmp_path / name)
    assert _bytes(tmp_path / "default") == _bytes(tmp_path / "off")
    assert _bytes(tmp_path / "on") == _bytes(tmp_path / "off")
    assert not (tmp_path / "off" / "rows").exists()
    assert (tmp_path / "on" / "rows").is_dir()
    ids = frame["cell_id"].to_numpy()[[3, 77, 140, 499, 250]].tolist()
    with store.open(tmp_path / "on", verify="full") as on, \
            store.open(tmp_path / "off") as off:
        sql, params = on._select({"cell_id": ids}, "cells", None)
        wide = store._arrow_of(on._conn, sql, params)
        calls = []
        real = on._read_row_copies
        on._read_row_copies = lambda *a: calls.append(1) or real(*a)
        copied = on._routed({"cell_id": ids}, "cells", None)
        assert calls == [1]
        parquet = off._routed({"cell_id": ids}, "cells", None)
        assert on._columns("cells") == copied.column_names
        order = lambda t: t.sort_by("cell_id")
        assert order(copied).equals(order(parquet))
        assert order(copied).select(wide.column_names).cast(
            order(wide).schema).equals(order(wide))
        assert copied.schema.field("f3").type == wide.schema.field("f3").type
        assert copied.schema.field("f_wide").type == wide.schema.field(
            "f_wide").type


def test_a_rewritten_well_replaces_its_files(tmp_path):
    _write(tmp_path)
    old = sorted((tmp_path / "tables" / "cells").rglob("*.parquet"))
    frame = _frame("A1", n=10, seed=9)
    store.write_part(tmp_path, "cells", frame, plate="pl/1", well="A1",
                     crops=_crops(10), key_columns=("prcfo",))
    report = store.register(tmp_path)
    assert report["superseded"] == 3
    with store.open(tmp_path) as opened:
        assert len(opened.query({"well": "A1"})) == 10
    now = sorted((tmp_path / "tables" / "cells").rglob("*.parquet"))
    assert len(now) == len(old) and now != old
    assert not (tmp_path / "crops" / "plate=pl%2F1" / "well=A1.zarr").exists()


def test_a_write_stopped_before_its_manifest_is_never_read(tmp_path,
                                                          monkeypatch):
    _write(tmp_path, wells=("A1",))
    real = os.replace

    def failing(source, target):
        """Fail when the manifest would be published."""
        if str(target).endswith(".json"):
            raise KeyboardInterrupt
        return real(source, target)

    monkeypatch.setattr(store.os, "replace", failing)
    with pytest.raises(KeyboardInterrupt):
        store.write_part(tmp_path, "cells", _frame("B2"), plate="pl/1",
                         well="B2", crops=_crops(300))
    monkeypatch.setattr(store.os, "replace", real)
    assert store.register(tmp_path)["registered"] == 0
    with store.open(tmp_path, verify="full") as opened:
        assert set(opened.query()["well"]) == {"A1"}
    assert store.cleanup(tmp_path, older_than=-1)


def test_a_file_changed_outside_spacr_is_caught(tmp_path):
    _write(tmp_path, wells=("A1",), crops=False)
    part = next((tmp_path / "tables").rglob("part-*.parquet"))
    stat = part.stat()
    data = bytearray(part.read_bytes())
    data[len(data) // 2] ^= 0xFF
    part.write_bytes(bytes(data))
    os.utime(part, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    store.open(tmp_path).close()
    with pytest.raises(store.StoreError, match="sha256"):
        store.open(tmp_path, verify="full")
    part.write_bytes(bytes(data) + b"x")
    with pytest.raises(store.StoreError, match="size or mtime"):
        store.open(tmp_path)


def test_an_older_sqlite_project_opens_through_the_same_api(tmp_path):
    frame = _frame("A1", n=20)
    with sqlite3.connect(str(tmp_path / "measurements.db"),
                         timeout=30) as conn:
        conn.execute("CREATE TABLE cell (cell_id INTEGER, gene TEXT, "
                     "f0 REAL)")
        conn.executemany("INSERT INTO cell VALUES (?, ?, ?)", [
            (int(c), g, None if np.isnan(v) else float(v))
            for c, g, v in zip(frame["cell_id"], frame["gene"], frame["f0"])])
    with store.open(tmp_path) as opened:
        assert opened.legacy.endswith("measurements.db")
        found = opened.query(table="cell")
        assert found["cell_id"].tolist() == frame["cell_id"].tolist()
        assert np.array_equal(found["f0"].to_numpy(float),
                              frame["f0"].to_numpy(float), equal_nan=True)


def test_ops_storage_is_checked():
    """Item 0an: the store is the only target; 'current' is a retired value."""
    assert ops_engine._ops_storage({}) == "spacr_store"
    assert ops_engine._ops_storage({"ops_storage": "spacr_store"}) == \
        "spacr_store"
    assert ops_engine._ops_storage({"ops_storage": "current"}) == "spacr_store"
    with pytest.raises(ValueError, match="ops_storage"):
        ops_engine._ops_storage({"ops_storage": "hdf5"})


def test_an_older_settings_file_naming_current_loads_as_the_store():
    from spacr.ops_settings import ops_defaults

    assert ops_defaults({"ops_storage": "current"})["ops_storage"] == \
        "spacr_store"
    assert ops_defaults({})["ops_storage"] == "spacr_store"


def test_the_ops_run_writes_the_same_rows_and_crops_to_the_store(measured):
    from spacr.ops_store import write_table

    reads = pd.DataFrame({"plate": PLATE, "well": "A1",
                          "object_id": np.repeat(np.arange(1, 9), 3),
                          "read": ["ACGT", "ACGA", "TTTT"] * 8,
                          "quality": np.linspace(0, 1, 24)})
    write_table(str(measured / "run" / "measurements.db"), "ops_reads", reads)
    out = _run(measured, FULL, ops_merged_root=str(measured / "nas"),
               ops_crop_control_fraction=0.0, ops_crop_masks=True,
               ops_ph_measure_gpu=False, ops_tables_layout="wide",
               ops_storage="spacr_store")
    run = measured / "run"
    project = run / "measurements"
    assert out["store"]["project"] == str(project)
    assert out["store"]["wells"]["A1"] == {"cell": 8, "crops": 8,
                                           "reads": 24, "barcodes": 8}
    assert not (run / store.CATALOG).exists()
    with store.open(project, verify="full") as opened:
        assert {"cell", "nucleus", "cytoplasm", "cell_wide", "reads",
                "barcodes"} <= set(opened.tables())
        wide = opened.query(table="cell_wide")
        found, stack = opened.query(table="cell", crops=True)
        barcodes = opened.query(table="barcodes")
        stored = opened.query(table="reads")
    assert len(barcodes) == 8
    stored = stored.sort_values(["object_id", "read"]).reset_index(drop=True)
    expected = reads.sort_values(["object_id", "read"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(stored[expected.columns], expected)
    found = found.set_index("prcfo").loc[wide["prcfo"]].reset_index()
    for column in [c for c in wide.columns if not c.startswith(
            ("nucleus_", "cytoplasm_"))]:
        left = found[column].to_numpy()
        right = wide[column].to_numpy()
        if right.dtype.kind == "f" or found[column].dtype.kind == "f":
            assert np.array_equal(left.astype(float), right.astype(float),
                                  equal_nan=True), column
        else:
            assert [None if pd.isna(v) else v for v in left] == [
                None if pd.isna(v) else v for v in right], column
    with store.open(project) as opened:
        result, stack = opened.query(table="cell", crops=True)
    import glob

    shard = glob.glob(str(measured / "nas" / "data" / "cell" / PLATE / "A1" /
                          "shard-0000.tar"))[0]
    index = pd.read_parquet(shard[:-4] + ".index.parquet")
    position = {p: i for i, p in enumerate(result["prcfo"])}
    for record in index.itertuples():
        image, _mask = ops_engine._ph_read_crop(shard, record.offset_data,
                                                record.size)
        assert np.array_equal(stack[position[record.prcfo]], image)


def test_the_default_ops_run_writes_the_store_and_no_sqlite(measured):
    """Item 0an: the default run keeps every table in measurements/ only."""
    out = _run(measured, FULL, ops_merged_root=str(measured / "nas"))
    run = measured / "run"
    assert out["store"]["project"] == str(run / "measurements")
    assert not (run / "measurements.db").exists()
    assert not (run / store.CATALOG).exists()
    assert not (run / "tables").exists() and not (run / "crops").exists()
    with store.open(run / "measurements") as opened:
        assert {"cell", "merge", "barcodes"} <= set(opened.tables())


def test_box_sized_crops_stay_in_the_crop_pool(monkeypatch):
    """Item 0an: crops cut with crop_margin_px are each their cell's box.

    One store crop array cannot hold crops of different sizes; the store
    step raised a broadcast error on the A1 check fields (crop_margin_px 16).
    Such crops stay in the OPS crop pool and are not copied; crops of one
    size still are.
    """
    rows = [{"prcfo": "P_r1_c1_f1_o1", "height": 10, "width": 12},
            {"prcfo": "P_r1_c1_f1_o2", "height": 14, "width": 12}]
    monkeypatch.setattr(ops_engine, "_ph_crop_commit",
                        lambda root, plate, well: {"rows": rows})
    prcfo = pd.Series([r["prcfo"] for r in rows])
    assert ops_engine._ops_store_crops("root", "P", "A1", prcfo) == (
        None, None, None)
