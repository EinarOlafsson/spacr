"""Cloud OME-Zarr reading: wells and single images, listings and credentials.

The in-memory fsspec store stands in for a bucket, as in
``test_cloud_storage_access``; nothing leaves the machine.
"""
from __future__ import annotations

import sys
import types

import pytest

fsspec = pytest.importorskip("fsspec")

from spacr import ome_zarr  # noqa: E402
from tests.test_cloud_storage_access import plate_in_memory  # noqa: E402,F401


def test_a_cloud_path_object_keeps_its_protocol(plate_in_memory):  # noqa: F811
    url = plate_in_memory[0]
    path = ome_zarr._cloud_path(url)
    assert ome_zarr._cloud_protocol(path) == "memory"
    assert ome_zarr._cloud_path(path) is path
    assert repr(path).startswith("_CloudPath(")
    assert path.parent.key.endswith(path.key.rsplit("/", 2)[-2])
    assert path.joinpath("", "A").key.endswith("/A")
    assert ome_zarr._redact_url("/local/plate.zarr") == "/local/plate.zarr"


def test_a_well_is_described_and_staged_on_its_own(plate_in_memory,  # noqa: F811
                                                   tmp_path):
    url = plate_in_memory[0]
    well = f"{url}/B/3"
    assert ome_zarr._cloud_source_kind(well) == "well"
    assert "OME-Zarr well: 2 fields" in ome_zarr._describe_cloud_source(well)
    summary = ome_zarr._stage_ome_zarr(well, tmp_path / "w", fields=1,
                                       report=None)
    names = sorted(p.name for p in (tmp_path / "w").glob("*.tif"))
    assert summary["written"] == 2
    assert all("_B03_" in name and name.startswith("plate_") for name in names)


def test_a_single_image_is_described_and_staged_as_a01(plate_in_memory,  # noqa: F811
                                                       tmp_path):
    url = plate_in_memory[0]
    image = f"{url}/A/1/0"
    assert ome_zarr._cloud_source_kind(image) == "image"
    assert "DAPI" in ome_zarr._describe_cloud_source(image)
    ome_zarr._stage_ome_zarr(image, tmp_path / "i", report=None)
    assert all("_A01_" in p.name for p in (tmp_path / "i").glob("*.tif"))


def test_a_plain_group_is_not_stageable(plate_in_memory, tmp_path):  # noqa: F811
    url = plate_in_memory[0]
    row = f"{url}/A"
    assert ome_zarr._cloud_source_kind(row) == "group"
    assert ome_zarr._describe_cloud_source(row).startswith("group:")
    with pytest.raises(ome_zarr.OmeZarrError, match="not an OME-Zarr plate"):
        ome_zarr._stage_ome_zarr(row, tmp_path / "g", report=None)


def test_listing_refuses_local_and_http_paths():
    with pytest.raises(ValueError, match="not a cloud address"):
        ome_zarr._cloud_listing("/local/folder")
    with pytest.raises(ome_zarr._CloudStorageError, match="plain HTTP"):
        ome_zarr._cloud_listing("https://host/plate.zarr")
    with pytest.raises(ome_zarr._CloudStorageError, match="plain HTTP"):
        ome_zarr._sync_cloud_folder("https://host/data", "/tmp/never")


def test_wells_are_parsed_or_refused():
    assert ome_zarr._parse_wells(None) == ()
    assert ome_zarr._parse_wells(["A1", "", "b3"]) == ("A01", "B03")
    with pytest.raises(ValueError, match="not a well address"):
        ome_zarr._parse_wells("A1, nonsense")
    assert ome_zarr._well_from_path("") == "A01"


def test_plate_metadata_must_exist_and_bad_wells_are_skipped(monkeypatch):
    monkeypatch.setattr(ome_zarr, "_group_attributes",
                        lambda root: ({}, None))
    with pytest.raises(ome_zarr.OmeZarrError, match="no `plate` metadata"):
        ome_zarr._plate_layout("root")
    monkeypatch.setattr(ome_zarr, "_group_attributes", lambda root: ({
        "plate": {"name": "p", "wells": [{"path": ""}, "junk",
                                         {"path": "C/4"}]}}, None))
    assert ome_zarr._plate_layout("root") == ("p", [("C04", "C/4")])


def test_credentials_are_described_for_each_source(monkeypatch):
    options = ome_zarr._CloudOptions
    assert options(anonymous=True).describe("s3") == "anonymously"
    assert "profile 'lab'" in options(profile="lab").describe("s3")
    monkeypatch.setattr(ome_zarr, "_aws_credentials_found", lambda: True)
    assert "configured" in options().describe("s3")
    assert ome_zarr._storage_options("az", options(anonymous=True)) == {
        "anon": True}
    assert ome_zarr._storage_options("abfs", options()) == {}


def test_aws_credentials_are_looked_up_once(monkeypatch):
    botocore = types.ModuleType("botocore")
    session = types.ModuleType("botocore.session")
    calls = []

    def get_session():
        calls.append(1)
        return types.SimpleNamespace(get_credentials=lambda: object())

    session.get_session = get_session
    botocore.session = session
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.session", session)
    monkeypatch.setattr(ome_zarr, "_AWS_CREDENTIALS_FOUND", {})
    assert ome_zarr._aws_credentials_found() is True
    assert ome_zarr._aws_credentials_found() is True
    assert calls == [1]
    monkeypatch.setattr(ome_zarr, "_AWS_CREDENTIALS_FOUND", {})
    monkeypatch.setitem(sys.modules, "botocore.session", None)
    assert ome_zarr._aws_credentials_found() is False


def test_a_missing_cloud_library_names_its_install(monkeypatch):
    def missing(protocol, **options):
        error = ImportError("no s3fs")
        error.name = "s3fs"
        raise error

    monkeypatch.setattr(fsspec, "filesystem", missing)
    with pytest.raises(ome_zarr._CloudLibraryMissing, match="s3fs"):
        ome_zarr._cloud_filesystem("s3", ome_zarr._CloudOptions())


def test_a_store_that_refuses_access_reads_as_missing(plate_in_memory):  # noqa: F811
    path = ome_zarr._cloud_path(plate_in_memory[0] + "/missing")

    class _Refusing:
        def isfile(self, key):
            raise PermissionError("denied")

        def exists(self, key):
            raise PermissionError("denied")

        def cat_file(self, key, *a, **k):
            raise PermissionError("denied")

    refusing = ome_zarr._CloudPath(_Refusing(), path.protocol, path.key)
    assert refusing.is_file() is False
    assert refusing.exists() is False
    assert refusing.is_dir() is False


def test_a_single_cloud_file_is_not_a_run_source(tmp_path):
    fs = fsspec.filesystem("memory")
    fs.pipe("/bucket-file/one.tif", b"not really a tiff")
    try:
        with pytest.raises(ome_zarr._CloudStorageError, match="single file"):
            ome_zarr._localize_cloud_source(
                "memory://bucket-file/one.tif",
                {"cloud_cache": str(tmp_path)}, "mask", report=None)
    finally:
        fs.rm("/bucket-file", recursive=True)


def test_results_upload_skips_folders_and_reports_a_missing_one(tmp_path):
    said = []
    assert ome_zarr._upload_cloud_results(
        str(tmp_path), "memory://out", "memory://in/plate.zarr/merged",
        ome_zarr._CloudOptions(), report=said.append) == []
    assert "does not exist" in said[0]
    nested = tmp_path / "merged"
    (tmp_path / "measurements" / "sub").mkdir(parents=True)
    (tmp_path / "measurements" / "sub" / "a.csv").write_text("x\n")
    fs = fsspec.filesystem("memory")
    try:
        written = ome_zarr._upload_cloud_results(
            str(nested), "memory://out-bucket", "memory://in/plate.zarr/merged",
            ome_zarr._CloudOptions(), report=None)
        assert written == ["memory://out-bucket/plate/measurements/sub/a.csv"]
    finally:
        fs.rm("/out-bucket", recursive=True)
