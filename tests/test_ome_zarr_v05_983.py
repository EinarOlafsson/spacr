"""Item 983: OME-NGFF 0.5 (zarr-format 3, sharded) written by spacr.ome_zarr."""
from __future__ import annotations

import json

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from spacr.ome_zarr import OmeZarrError, read_ome_zarr, write_ome_zarr


def _field(seed=0, shape=(3, 700, 900), dtype="uint16"):
    return np.random.default_rng(seed).integers(0, 60000, shape).astype(dtype)


def test_a_05_store_reads_back_pixel_exact_with_its_metadata(tmp_path):
    data = _field()
    write_ome_zarr(tmp_path / "f.zarr", data, axes=["c", "y", "x"],
                   channel_names=["DAPI", "GFP", "AF750"], ngff_version="0.5")
    group = json.loads((tmp_path / "f.zarr" / "zarr.json").read_text())
    assert group["zarr_format"] == 3
    ome = group["attributes"]["ome"]
    assert ome["version"] == "0.5"
    assert "version" not in ome["multiscales"][0]
    assert [c["label"] for c in ome["omero"]["channels"]] == ["DAPI", "GFP", "AF750"]
    array = json.loads((tmp_path / "f.zarr" / "0" / "zarr.json").read_text())
    assert array["codecs"][0]["name"] == "sharding_indexed"
    inner = [c["name"] for c in array["codecs"][0]["configuration"]["codecs"]]
    assert "zstd" in inner
    back = np.asarray(read_ome_zarr(tmp_path / "f.zarr").read())
    assert back.dtype == data.dtype and np.array_equal(back, data)


def test_one_shard_holds_a_whole_field_by_default(tmp_path):
    write_ome_zarr(tmp_path / "f.zarr", _field(shape=(2, 600, 700)),
                   axes=["c", "y", "x"], ngff_version="0.5")
    chunk_files = [p for p in (tmp_path / "f.zarr" / "0").rglob("*")
                   if p.is_file() and p.name != "zarr.json"]
    assert len(chunk_files) == 1


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int32", "float32", "bool"])
def test_every_spacr_array_kind_round_trips_exactly(tmp_path, dtype):
    rng = np.random.default_rng(3)
    data = (rng.random((2, 130, 150)) > 0.5) if dtype == "bool" else (
        rng.random((2, 130, 150)) * 1000).astype(dtype)
    write_ome_zarr(tmp_path / "f.zarr", data, axes=["c", "y", "x"],
                   ngff_version="0.5")
    back = np.asarray(read_ome_zarr(tmp_path / "f.zarr").read())
    assert back.dtype == data.dtype and np.array_equal(back, data)


def test_labels_keep_their_values_through_a_stride_pyramid(tmp_path):
    labels = np.zeros((512, 512), np.int32)
    labels[10:200, 10:200] = 3
    labels[300:500, 250:480] = 70000
    write_ome_zarr(tmp_path / "l.zarr", labels, axes=["y", "x"], levels=3,
                   downsample="stride", ngff_version="0.5")
    image = read_ome_zarr(tmp_path / "l.zarr")
    assert np.array_equal(np.asarray(image.read(0)), labels)
    assert set(np.unique(np.asarray(image.read(2)))) <= {0, 3, 70000}


def test_the_default_still_writes_04_and_shards_need_05(tmp_path):
    write_ome_zarr(tmp_path / "old.zarr", _field(shape=(1, 64, 64)),
                   axes=["c", "y", "x"])
    assert (tmp_path / "old.zarr" / ".zgroup").is_file()
    assert not (tmp_path / "old.zarr" / "zarr.json").exists()
    with pytest.raises(OmeZarrError, match="needs ngff_version='0.5'"):
        write_ome_zarr(tmp_path / "x.zarr", _field(shape=(1, 64, 64)),
                       axes=["c", "y", "x"], shards=(1, 64, 64))
    with pytest.raises(OmeZarrError, match="multiple of the chunks"):
        write_ome_zarr(tmp_path / "y.zarr", _field(shape=(1, 600, 600)),
                       axes=["c", "y", "x"], ngff_version="0.5",
                       chunks=(1, 256, 256), shards=(1, 300, 300))
    with pytest.raises(OmeZarrError, match="0.4 .* or 0.5"):
        write_ome_zarr(tmp_path / "z.zarr", _field(shape=(1, 8, 8)),
                       axes=["c", "y", "x"], ngff_version="0.6")
