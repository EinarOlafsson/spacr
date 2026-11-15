"""The phenotype run's engineering items (item 872).

Part A: the bounding-box loops of the three per-object measurements return
what the whole-field comparison returned, byte for byte, and a measurement
frame is written in one transaction without changing what is stored. Part B:
a well's rows are read by a query on that well (B8), the aggregate takes its
guides from any genotype table (B5), and a field missing from the download is
recorded once and not retried (B18).
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call

pytest.importorskip("scipy")

from spacr import measure, ops_engine, utils
from spacr.ops_store import write_table


def _label_images():
    """Touching objects, objects on the edge, gaps in the numbering, a
    one-pixel object, and an empty mask."""
    rng = np.random.default_rng(1)
    touching = np.zeros((40, 40), np.int32)
    touching[2:12, 2:12] = 1
    touching[2:12, 12:20] = 2
    touching[0:5, 30:40] = 7
    touching[25, 25] = 9
    touching[30:38, 0:6] = 12
    noisy = (rng.random((40, 40)) > 0.7).astype(np.int32)
    from scipy import ndimage

    noisy, _n = ndimage.label(noisy)
    return [touching, noisy.astype(np.int32), np.zeros((40, 40), np.int32)]


def _oracle(monkeypatch, call):
    """``call()`` with the bounding boxes switched off: the whole-field
    comparison the boxes replaced."""
    with monkeypatch.context() as patch:
        patch.setattr(measure, "_label_bounding_boxes", lambda mask: {})
        return call()


@pytest.mark.parametrize("index", [0, 1, 2])
def test_components_and_parents_are_byte_identical_to_the_whole_field_loop(
        monkeypatch, index):
    cells = _label_images()[index]
    nuclei = np.where(cells % 2 == 1, cells, 0).astype(np.int32)
    pathogens = np.roll(cells, 3, axis=1)
    fast = measure.get_components(cells, nuclei, pathogens)
    slow = _oracle(monkeypatch,
                   lambda: measure.get_components(cells, nuclei, pathogens))
    for got, want in zip(fast, slow):
        pd.testing.assert_frame_equal(got, want)
    fast = measure._map_child_to_parent(pathogens, cells)
    slow = _oracle(monkeypatch,
                   lambda: measure._map_child_to_parent(pathogens, cells))
    pd.testing.assert_frame_equal(fast, slow)


@pytest.mark.parametrize("index", [0, 1])
def test_intensity_distances_are_byte_identical_to_the_whole_field_loop(
        monkeypatch, index):
    cells = _label_images()[index]
    nuclei = np.where(cells % 2 == 1, cells, 0).astype(np.int32)
    pathogens = np.zeros_like(cells)
    rng = np.random.default_rng(2)
    channels = rng.random((40, 40, 2)).astype(np.float32)
    settings = {"distance_gaussian_sigma": 2, "pathogen_mask_dim": None}
    fast = measure._measure_intensity_distance(cells, nuclei, pathogens,
                                               channels, settings)
    slow = _oracle(monkeypatch, lambda: measure._measure_intensity_distance(
        cells, nuclei, pathogens, channels, settings))
    pd.testing.assert_frame_equal(fast, slow)
    for column in fast.columns:
        assert fast[column].to_numpy().tobytes() == \
            slow[column].to_numpy().tobytes()


def _frame(n=50):
    """A measurement frame of mixed types."""
    return pd.DataFrame({"prcfo": [f"p_r1_c1_f1_o{i}" for i in range(n)],
                         "area": np.arange(n, dtype=np.int64),
                         "mean": np.linspace(0, 1, n),
                         "flag": [bool(i % 2) for i in range(n)]})


def _connection(path):
    """An autocommit connection with the frame's table, as spaCR opens one."""
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute("CREATE TABLE cell (prcfo TEXT NOT NULL, area INTEGER, "
                 "mean REAL, flag INTEGER)")
    return conn


def test_one_transaction_per_frame_stores_what_row_by_row_stored(tmp_path):
    frame = _frame()
    conn = _connection(tmp_path / "a.db")
    utils._insert_frame(conn, "cell", frame)
    got = conn.execute("SELECT * FROM cell ORDER BY rowid").fetchall()
    schema = conn.execute("PRAGMA table_info(cell)").fetchall()
    conn.close()
    plain = _connection(tmp_path / "b.db")
    for row in frame.itertuples(index=False):
        plain.execute("INSERT INTO cell VALUES (?, ?, ?, ?)",
                      (row.prcfo, int(row.area), float(row.mean),
                       int(row.flag)))
    want = plain.execute("SELECT * FROM cell ORDER BY rowid").fetchall()
    assert got == want
    assert schema == plain.execute("PRAGMA table_info(cell)").fetchall()
    plain.close()


def test_a_frame_that_fails_half_way_leaves_the_table_unchanged(tmp_path):
    conn = _connection(tmp_path / "a.db")
    utils._insert_frame(conn, "cell", _frame(3))
    broken = _frame(10)
    broken.loc[6, "prcfo"] = None
    with pytest.raises(sqlite3.IntegrityError):
        utils._insert_frame(conn, "cell", broken)
    assert conn.execute("SELECT COUNT(*) FROM cell").fetchone()[0] == 3
    assert not conn.in_transaction
    conn.close()


def test_a_frame_inside_an_open_transaction_does_not_raise(tmp_path):
    conn = _connection(tmp_path / "a.db")
    conn.execute("BEGIN")
    utils._insert_frame(conn, "cell", _frame(4))
    assert conn.execute("SELECT COUNT(*) FROM cell").fetchone()[0] == 4
    conn.close()


def _db_with_wells(tmp_path):
    """``ops_objects`` for three wells, interleaved."""
    db = str(tmp_path / "measurements.db")
    rows = pd.DataFrame({"plate": "P1", "well": ["A1", "B2", "A1", "C3"] * 25,
                         "object_id": np.arange(100),
                         "centroid_y": np.arange(100) * 0.5,
                         "centroid_x": np.arange(100) * 2.0})
    write_table(db, "ops_objects", rows)
    return db, rows


def test_a_well_is_read_by_a_query_on_that_well(tmp_path, monkeypatch):
    db, rows = _db_with_wells(tmp_path)
    seen = []
    original = ops_engine.__dict__.get("_read_query")
    from spacr import tabular

    real = tabular._read_query

    def spy(conn, sql, **kwargs):
        """Record each query."""
        seen.append(sql)
        return real(conn, sql, **kwargs)

    monkeypatch.setattr(tabular, "_read_query", spy)
    got = ops_engine._well_rows(db, "ops_objects", "P1", "A1")
    want = rows[rows.well == "A1"].reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want, check_dtype=False)
    assert got["object_id"].tolist() == sorted(got["object_id"])
    assert any("WHERE plate = ? AND well = ?" in sql for sql in seen)
    assert not any(sql.strip() == "SELECT * FROM ops_objects" for sql in seen)
    assert ops_engine._well_rows(db, "ops_objects", "P1", "Z9").empty
    assert ops_engine._well_rows(db, "ops_merge", "P1", "A1") is None
    del original


def test_two_genotype_tables_give_two_aggregates_without_rerunning(tmp_path):
    rng = np.random.default_rng(0)
    db = str(tmp_path / "measurements.db")
    ids = np.arange(1, 81)
    write_table(db, "ops_merge", pd.DataFrame({
        "plate": "P1", "well": "A1", "site": 0, "label": ids,
        "object_id": ids, "distance": 0.5, "refined": 1}))
    pd.DataFrame({"sgRNA": ["AAAATTTT", "CCCCTTTT", "GGGGTTTT"],
                  "gene_symbol": ["nontargeting", "GENE1", "GENE2"]}).to_csv(
        tmp_path / "design.csv", index=False)
    pd.DataFrame({"plate": "P1", "well": "A1", "site": 0, "label": ids,
                  "nucleus_area": rng.normal(100, 5, 80)}).to_parquet(
        tmp_path / "ops_phenotype_cells_A1.parquet")
    tables = {}
    for name, guides in (("first", ["AAAA", "CCCC"]),
                         ("second", ["AAAA", "GGGG"])):
        path = tmp_path / f"{name}.parquet"
        pd.DataFrame({"plate": "P1", "well": "A1", "object_id": ids,
                      "mapped_guide": [guides[i % 2] for i in range(80)]}
                     ).to_parquet(path)
        report = ops_engine._ph_aggregate(
            db, "P1", ["A1"], {"ops_design": str(tmp_path / "design.csv"),
                               "ops_genotype_table": str(path),
                               "ops_ph_channels": "DAPI"}, str(tmp_path))
        tables[name] = set(pd.read_parquet(
            report["outputs"]["genes_parquet"])["gene"])
    assert tables == {"first": {"nontargeting", "GENE1"},
                      "second": {"nontargeting", "GENE2"}}


def test_missing_fields_are_recorded_once_and_not_retried(tmp_path,
                                                          monkeypatch):
    tifffile = pytest.importorskip("tifffile")
    from scipy import ndimage

    class _Model:
        """Label the bright regions of the last channel."""

        def eval(self, image, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
                 rescale=None, diameter=None, flow_threshold=0.4,
                 cellprob_threshold=0.0, do_3D=False, anisotropy=None,
                 flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
                 max_size_fraction=0.4, niter=None, augment=False,
                 tile_overlap=0.1, bsize=None, compute_masks=True, progress=None):
            """Threshold at half the maximum, after checking the call as Cellpose would."""
            check_cellpose_eval_call(image, channel_axis, require_channel_axis=False)
            plane = np.asarray(image, float)
            plane = plane[..., -1] if plane.ndim == 3 else plane
            return ndimage.label(plane > plane.max() / 2.0)[0], None, None

    monkeypatch.setattr(ops_engine, "_cellpose_model",
                        lambda settings, gpu, key="cellpose_model": _Model())
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    plane = np.zeros((32, 32), np.uint16)
    plane[8:16, 8:16] = 500
    for name in ("DAPI", "AF750"):
        tifffile.imwrite(tiles / f"20X_DAPI-AF750_A1_{name}_Site-0.tif", plane)
    tifffile.imwrite(tiles / "20X_DAPI-AF750_A1_DAPI_Site-1.tif", plane)
    (tiles / "20X_DAPI-AF750_A1_DAPI_Site-2.tif").write_bytes(b"broken")
    (tiles / "20X_DAPI-AF750_A1_AF750_Site-2.tif").write_bytes(b"broken")
    (tiles / "20X_DAPI-AF750_A1_DAPI_Site-3.tif").mkdir()
    (tiles / "20X_DAPI-AF750_A1_AF750_Site-3.tif").mkdir()
    db = str(tmp_path / "measurements.db")
    frame = pd.DataFrame([{"plate": "P1", "well": "A1", "site": s,
                           "centre_y": 100.0 * s, "centre_x": 0.0,
                           "is_anchor": int(s == 0)} for s in range(4)])
    write_table(db, "ops_phenotype", ops_engine._ph_transform_columns(
        frame, 0.25, 0.0, (32, 32)))
    settings = {"dst_root": str(tmp_path), "plate": "P1", "ops_gpu": False,
                "phenotype_source": str(tiles), "ops_ph_downsample": 1,
                "ops_ph_channels": "DAPI,AF750", "n_workers": 1}
    first = ops_engine.run_ops(settings, phases=("ph_segment",))
    segment = first["wells"]["A1"]["ph_fields"]["ph_segment"]
    assert segment["segmented"] == 1 and segment["missing_recorded"] == [1, 2, 3]
    recorded = ops_engine._ph_missing_fields(str(tmp_path), "A1")
    assert sorted(recorded) == [1, 2, 3]
    assert "no AF750 plane" in recorded[1]
    assert recorded[3] == "no file for the site in phenotype_source"
    again = ops_engine.run_ops(settings, phases=("ph_segment",))
    segment = again["wells"]["A1"]["ph_fields"]["ph_segment"]
    assert segment["missing_recorded"] == [] and segment["segmented"] == 0
    assert segment["missing_skipped"] == [1, 2, 3]
    retry = ops_engine.run_ops({**settings, "ops_retry_missing": True},
                               phases=("ph_segment",))
    assert retry["wells"]["A1"]["ph_fields"]["ph_segment"][
        "missing_recorded"] == [1, 2, 3]
