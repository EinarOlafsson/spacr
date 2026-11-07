"""Embedding a crop store is one public call: cells, wells, record, files.

``spacr.embeddings.embed_store`` takes a measurements database, a crop
array (in memory or a ``.npy`` opened memory-mapped) or a crop query,
estimates one fixed scale per channel from the store, embeds every crop and
averages the vectors per well. The encoder here is a picklable stand-in, so
nothing is downloaded and everything runs on the CPU.
"""
from __future__ import annotations

import json
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

import spacr.embeddings as emb


class MeanEncoder:
    """Per-plane means of the scaled crop."""

    in_channels = 3

    def __call__(self, stack):
        return stack.mean(axis=(1, 2)).astype(np.float32)


def _crops(n=12, channels=2, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((n, 6, 6, channels)) * 400).astype(np.float32)


def _objects(n=12):
    return pd.DataFrame({
        "plateID": ["p1"] * (n // 2) + ["p2"] * (n - n // 2),
        "rowID": ["A"] * n,
        "columnID": ["01", "02"] * (n // 2),
        "cell_id": np.arange(n),
    })


def test_cells_and_well_aggregates_come_from_one_fixed_scale(tmp_path):
    crops, objects = _crops(), _objects()
    record = {}
    out = emb.embed_store(crops, objects=objects, record=record,
                          encoder=MeanEncoder(), workers=1, threads=1,
                          output_dir=str(tmp_path / "out"))
    scale = emb._estimate_channel_scale(crops)
    expected = emb.embed_array(
        crops, emb.EmbeddingSpec(channel_scale=scale, device="cpu"),
        encoder=MeanEncoder())
    cells = out["cells"]
    assert list(cells.columns[:4]) == list(objects.columns)
    np.testing.assert_allclose(cells[list(expected.columns)].to_numpy(),
                               expected.values, rtol=1e-6)
    assert record["channel_scale"] == {"0": scale[0], "1": scale[1]}
    assert record["n_cells"] == 12 and record["n_wells"] == 4

    wells = out["wells_mean"]
    assert list(wells[["plateID", "columnID"]].itertuples(index=False, name=None)) == [
        ("p1", "01"), ("p1", "02"), ("p2", "01"), ("p2", "02")]
    assert wells["n_cells"].tolist() == [3, 3, 3, 3]
    first = cells[(cells.plateID == "p1") & (cells.columnID == "01")]
    column = expected.columns[0]
    assert wells.loc[0, column] == pytest.approx(first[column].mean())
    assert out["wells_median"].loc[0, column] == pytest.approx(
        first[column].median())

    files = out["files"]
    assert set(files) == {"cells", "wells_mean", "wells_median", "record"}
    from spacr.tabular import read_table

    written = read_table(files["cells"], report=None)
    np.testing.assert_allclose(written[column].to_numpy(),
                               cells[column].to_numpy())
    with open(files["record"], encoding="utf-8") as handle:
        assert json.load(handle)["n_wells"] == 4


def test_a_memory_mapped_shard_matches_the_array_in_memory(tmp_path):
    crops, objects = _crops(20, seed=3), _objects(20)
    path = tmp_path / "shard.npy"
    np.save(path, crops)
    table = tmp_path / "objects.csv"
    objects.to_csv(table, index=False)
    from_memory = emb.embed_store(crops, objects=objects, workers=1,
                                  threads=1, encoder=MeanEncoder())
    from_disk = emb.embed_store(str(path), objects=str(table), workers=2,
                                threads=1, encoder=MeanEncoder())
    columns = [c for c in from_memory["cells"].columns if c.startswith("emb_")]
    np.testing.assert_array_equal(from_disk["cells"][columns].to_numpy(),
                                  from_memory["cells"][columns].to_numpy())
    assert from_disk["record"]["source"] == str(path)


def test_a_recorded_scale_is_reused_and_crops_without_wells_get_no_well_table():
    crops = _crops(6)
    record = {"channel_scale": {"0": 100.0, "1": 200.0}}
    out = emb.embed_store(crops, record=record, workers=1, threads=1,
                          encoder=MeanEncoder())
    assert out["wells_mean"] is None and out["wells_median"] is None
    assert out["cells"]["object"].tolist() == list(range(6))
    assert out["record"]["channel_scale"] == {"0": 100.0, "1": 200.0}
    np.testing.assert_allclose(
        out["cells"]["emb_c0_000"].to_numpy(),
        np.clip(crops[..., 0] / 100.0, 0, 1).mean(axis=(1, 2)), rtol=1e-6)


def test_object_rows_must_match_the_crops():
    with pytest.raises(emb.EmbeddingError, match="one row per crop"):
        emb.embed_store(_crops(6), objects=_objects(4), encoder=MeanEncoder())
    with pytest.raises(emb.EmbeddingError, match="neither"):
        emb.embed_store("crops.tif", encoder=MeanEncoder())


def _png(path, value):
    from PIL import Image

    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)


def _database(tmp_path):
    root = tmp_path / "screen"
    data = root / "data" / "cell_png"
    measurements = root / "measurements"
    data.mkdir(parents=True)
    measurements.mkdir(parents=True)
    rows = []
    for plate in ("plate1", "plate2"):
        for column in ("01", "02"):
            for label in (1, 2):
                name = f"{plate}_A{column}_1_{label}.png"
                _png(str(data / name), 20 * label + (5 if column == "02" else 0))
                rows.append((str(data / name), name, plate, "A", column, "1",
                             f"o{label}"))
    db = measurements / "measurements.db"
    conn = sqlite3.connect(str(db), timeout=30)
    try:
        conn.execute("CREATE TABLE png_list (png_path TEXT, file_name TEXT,"
                     " plateID TEXT, rowID TEXT, columnID TEXT, fieldID TEXT,"
                     " cell_id TEXT)")
        conn.executemany("INSERT INTO png_list VALUES (?,?,?,?,?,?,?)", rows)
        conn.execute("CREATE TABLE cell (plateID TEXT, rowID TEXT,"
                     " columnID TEXT, fieldID TEXT, path_name TEXT,"
                     " file_name TEXT)")
        conn.commit()
    finally:
        conn.close()
    return str(db)


def test_a_measurements_database_is_embedded_per_cell_and_per_well(tmp_path):
    db = _database(tmp_path)
    out = emb.embed_store(db, workers=1, threads=1, encoder=MeanEncoder(),
                          output_dir=str(tmp_path / "embedded"))
    cells, wells = out["cells"], out["wells_mean"]
    assert len(cells) == 8 and len(wells) == 4
    assert {"png_path", "plateID", "rowID", "columnID"} <= set(cells.columns)
    assert wells["n_cells"].tolist() == [2, 2, 2, 2]
    one = emb.embed_store(db, plate="plate1", workers=1, threads=1,
                          encoder=MeanEncoder())
    assert set(one["cells"]["plateID"]) == {"plate1"}
    assert os.path.isfile(out["files"]["wells_median"])
