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


def test_listing_refuses_local_and_http_paths():
    with pytest.raises(ValueError, match="not a cloud address"):
        ome_zarr._cloud_listing("/local/folder")
    http = ome_zarr._CloudPath(object(), "https", "host/plate.zarr")
    with pytest.raises(ome_zarr._CloudStorageError, match="plain HTTP"):
        ome_zarr._cloud_listing(http)
    with pytest.raises(ome_zarr._CloudStorageError, match="plain HTTP"):
        ome_zarr._sync_cloud_folder(http, "/tmp/never")


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


def test_results_upload_skips_folders_and_reports_a_missing_one(tmp_path):
    said = []
    assert ome_zarr._upload_cloud_results(
        str(tmp_path), "memory://out", "memory://in/plate.zarr/merged",
        ome_zarr._CloudOptions(), report=said.append) == []
    assert "does not exist" in said[0]
    nested = tmp_path / "merged"
    (tmp_path / "measurements" / "sub").mkdir(parents=True)
    (tmp_path / "measurements" / "sub" / "a.csv").write_text("x\n", encoding="utf-8")
    fs = fsspec.filesystem("memory")
    try:
        written = ome_zarr._upload_cloud_results(
            str(nested), "memory://out-bucket", "memory://in/plate.zarr/merged",
            ome_zarr._CloudOptions(), report=None)
        assert written == ["memory://out-bucket/plate/measurements/sub/a.csv"]
    finally:
        fs.rm("/out-bucket", recursive=True)
