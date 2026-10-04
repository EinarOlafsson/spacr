"""Cloud OME-Zarr edges: odd listings, axis-free images and staging names."""
from __future__ import annotations

import types

import numpy as np
import pytest

fsspec = pytest.importorskip("fsspec")

from spacr import ome_zarr  # noqa: E402
from tests.test_cloud_storage_access import plate_in_memory  # noqa: E402,F401


def test_a_local_path_is_not_a_cloud_address(tmp_path):
    with pytest.raises(ValueError, match="is not a cloud address"):
        ome_zarr._cloud_path(str(tmp_path))
    with pytest.raises(ValueError, match="is not a cloud address"):
        ome_zarr._cloud_listing(str(tmp_path))


def test_an_image_without_time_or_channel_axes_gives_one_plane():
    image = types.SimpleNamespace(
        axis_names=["y", "x"], time_axis=None, channel_axis=None,
        level=lambda level: types.SimpleNamespace(shape=(3, 4)),
        read=lambda level, region, prefer_zarr: np.ones((3, 4)))
    planes = list(ome_zarr._field_planes(image, 0))
    assert [(t, c, p.shape, z) for t, c, p, z in planes] == [(0, 0, (3, 4), False)]


class _Fs:
    def __init__(self, entries):
        self.entries = entries

    def ls(self, key, detail=True):
        return self.entries

    def find(self, key, detail=True):
        return {e["name"]: e for e in self.entries}


def test_listings_trim_prefixes_and_skip_the_folder_itself():
    fs = _Fs([{"name": "bucket/run/", "type": "directory"},
              {"name": "bucket/run/b.tif", "type": "file", "size": 5},
              {"name": "other/a", "type": "directory"}])
    root = ome_zarr._CloudPath(fs, "memory", "bucket/run")
    assert ome_zarr._cloud_listing(root) == [("a", "folder", 0),
                                              ("b.tif", "file", 5)]


def test_folders_and_foreign_keys_are_not_synced(tmp_path):
    fs = _Fs([{"name": "bucket/run/sub", "type": "directory"},
              {"name": "elsewhere/x.tif", "type": "file"}])
    root = ome_zarr._CloudPath(fs, "memory", "bucket/run")
    with pytest.raises(ome_zarr._CloudStorageError, match="holds no images"):
        ome_zarr._sync_cloud_folder(root, tmp_path / "d", report=None)


def test_a_well_outside_a_zarr_named_plate_keeps_the_folder_name(
        plate_in_memory, tmp_path):  # noqa: F811
    url, _arrays, fs, bucket = plate_in_memory
    fs.copy(f"{bucket}/plate.zarr", f"{bucket}/screen7", recursive=True)
    well = f"memory://{bucket.lstrip('/')}/screen7/B/3"
    ome_zarr._stage_ome_zarr(well, tmp_path / "w", fields=1, report=None)
    assert all(p.name.startswith("screen7_B03_")
               for p in (tmp_path / "w").glob("*.tif"))


def test_a_single_cloud_file_is_not_a_source(plate_in_memory, tmp_path):  # noqa: F811
    url, _arrays, fs, bucket = plate_in_memory
    fs.pipe(f"{bucket}/one.tif", b"x")
    settings = {"cloud_cache": str(tmp_path / "cache")}
    with pytest.raises(ome_zarr._CloudStorageError, match="single file"):
        ome_zarr._localize_cloud_source(
            f"memory://{bucket.lstrip('/')}/one.tif", settings, "mask",
            report=None)


def test_staging_keeps_a_cellvoyager_metadata_type(plate_in_memory, tmp_path):  # noqa: F811
    url = plate_in_memory[0]
    said = []
    settings = {"cloud_cache": str(tmp_path / "cache"), "cloud_wells": "B03",
                "cloud_fields": 1, "metadata_type": "cellvoyager"}
    ome_zarr._localize_cloud_source(url, settings, "mask", report=said.append)
    assert not any("metadata_type set" in line for line in said)


def test_describing_an_empty_plate_and_a_well_without_fields(plate_in_memory,  # noqa: F811
                                                             monkeypatch):
    url = plate_in_memory[0]
    monkeypatch.setattr(ome_zarr, "_well_fields", lambda root: [])
    text = ome_zarr._describe_cloud_source(url)
    assert "fields" in text and "\n" in text
    assert ome_zarr._describe_cloud_source(f"{url}/B/3") == \
        "OME-Zarr well: 0 fields"
    monkeypatch.setattr(ome_zarr, "_plate_layout", lambda root: ("p", []))
    assert ome_zarr._describe_cloud_source(url) == "OME-Zarr plate p: 0 wells"


def test_time_and_channel_axes_are_read_one_plane_at_a_time():
    regions = []

    def read(level, region, prefer_zarr):
        regions.append(dict(region))
        return np.ones((1, 1, 3, 4))

    image = types.SimpleNamespace(
        axis_names=["t", "c", "y", "x"],
        time_axis=types.SimpleNamespace(name="t"),
        channel_axis=types.SimpleNamespace(name="c"),
        level=lambda level: types.SimpleNamespace(shape=(2, 1, 3, 4)),
        read=read)
    planes = list(ome_zarr._field_planes(image, 0))
    assert regions == [{"t": 0, "c": 0}, {"t": 1, "c": 0}]
    assert all(p.shape == (3, 4) for _t, _c, p, _z in planes)


def test_a_cloud_folder_of_images_is_mirrored(plate_in_memory, tmp_path):  # noqa: F811
    url, _arrays, fs, bucket = plate_in_memory
    fs.pipe(f"{bucket}/images/a.tif", b"x")
    local = ome_zarr._localize_cloud_source(
        f"memory://{bucket.lstrip('/')}/images",
        {"cloud_cache": str(tmp_path / "cache")}, "mask", report=None)
    assert (tmp_path / "cache").exists() and local
