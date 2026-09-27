"""Cloud sources: OME-Zarr plates, folders and tables read from object storage.

No test here reaches the internet or uses a credential. The stores are
fsspec's in-memory filesystem, which stands in for S3, Google Cloud Storage
and Azure because spaCR talks to all of them through the same fsspec calls,
and a plain HTTP server on 127.0.0.1, which is how public OME-Zarr such as
IDR's is served.

What is pinned:

* only the chosen wells, fields and pyramid level are fetched, counted at
  :func:`spacr.ome_zarr._read_chunk_bytes`, and a repeated run fetches
  nothing;
* the fetched TIFFs carry the channel planes (z maximum projected) under the
  Yokogawa names Make Masks parses;
* Make Masks and Measure given the same address work in the same local
  folder, and ``cloud_results`` copies the measurements back;
* tables open through :mod:`spacr.tabular` from an address and are cached;
* a missing optional library names its install line, and no secret in an
  address reaches the console or a file.
"""
from __future__ import annotations

import functools
import http.server
import json
import threading
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

fsspec = pytest.importorskip("fsspec")

from spacr import ome_zarr, tabular                               # noqa: E402
from spacr.ome_zarr import Axis, write_ome_zarr                   # noqa: E402

WELLS = (("A", "1", 0, 0), ("B", "3", 1, 2))
FIELDS = ("0", "1")
SHAPE = (2, 3, 64, 64)
CHUNKS = (1, 1, 32, 32)


def _write_plate(root: Path) -> dict:
    """Write a two-well, two-field OME-Zarr 0.4 plate; return its arrays.

    As in IDR's plates, y and x are in micrometers and z carries no unit.
    """
    rng = np.random.default_rng(550)
    root.mkdir(parents=True)
    (root / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (root / ".zattrs").write_text(json.dumps({"plate": {
        "name": "screen_1", "version": "0.4",
        "rows": [{"name": "A"}, {"name": "B"}],
        "columns": [{"name": "1"}, {"name": "2"}, {"name": "3"}],
        "wells": [{"path": f"{r}/{c}", "rowIndex": ri, "columnIndex": ci}
                  for r, c, ri, ci in WELLS]}}))
    arrays = {}
    for row, column, _, _ in WELLS:
        (root / row).mkdir(exist_ok=True)
        (root / row / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        well = root / row / column
        well.mkdir()
        (well / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        (well / ".zattrs").write_text(json.dumps(
            {"well": {"images": [{"path": f} for f in FIELDS]}}))
        for field in FIELDS:
            data = rng.integers(0, 4000, size=SHAPE, dtype=np.uint16)
            write_ome_zarr(
                well / field, data,
                axes=[Axis.channel("c"), Axis.space("z", 1.0, "micrometer"),
                      Axis.space("y", 0.5, "micrometer"),
                      Axis.space("x", 0.5, "micrometer")],
                levels=2, channel_names=("DAPI", "GFP"), chunks=CHUNKS)
            attrs_path = well / field / ".zattrs"
            attrs = json.loads(attrs_path.read_text())
            attrs["multiscales"][0]["axes"][1].pop("unit")
            attrs_path.write_text(json.dumps(attrs))
            arrays[(f"{row}{column}", field)] = data
    return arrays


@pytest.fixture
def plate_in_memory(tmp_path):
    """The plate uploaded to a fresh bucket of the in-memory store."""
    arrays = _write_plate(tmp_path / "plate.zarr")
    fs = fsspec.filesystem("memory")
    bucket = f"/bucket-{uuid.uuid4().hex[:8]}"
    fs.put(str(tmp_path / "plate.zarr"), f"{bucket}/plate.zarr",
           recursive=True)
    try:
        yield f"memory://{bucket.lstrip('/')}/plate.zarr", arrays, fs, bucket
    finally:
        fs.rm(bucket, recursive=True)


@pytest.fixture
def chunk_reads(monkeypatch):
    """Every chunk the reader fetches, in order."""
    seen: list = []
    real = ome_zarr._read_chunk_bytes

    def counting(path):
        seen.append(str(path))
        return real(path)

    monkeypatch.setattr(ome_zarr, "_read_chunk_bytes", counting)
    return seen


def test_addresses_are_recognised_and_secrets_removed():
    assert ome_zarr._is_cloud_url("s3://bucket/plate.zarr")
    assert ome_zarr._is_cloud_url("gs://bucket/x")
    assert ome_zarr._is_cloud_url("az://container/x")
    assert ome_zarr._is_cloud_url("https://host/plate.zarr")
    assert not ome_zarr._is_cloud_url("/data/plate")
    assert not ome_zarr._is_cloud_url("C:\\data\\plate")
    assert not ome_zarr._is_cloud_url("file:///data/plate")
    assert not ome_zarr._is_cloud_url(None)
    assert ome_zarr._redact_url(
        "https://me:hunter2@host/a/b.zarr?X-Amz-Signature=abc#frag"
    ) == "https://host/a/b.zarr"


def test_a_plate_is_described_from_its_metadata_alone(plate_in_memory,
                                                      chunk_reads):
    url, _, _, _ = plate_in_memory
    assert ome_zarr._cloud_source_kind(url) == "plate"
    text = ome_zarr._describe_cloud_source(url)
    assert "screen_1: 2 wells" in text
    assert "well A01: 2 fields" in text
    assert "DAPI, GFP" in text
    with pytest.raises(ome_zarr.OmeZarrError, match="not all in the same"):
        ome_zarr.read_ome_zarr(f"{url}/B/3/1")
    image = ome_zarr._open_ome_zarr(f"{url}/B/3/1", check_units=False)
    assert image.shape == SHAPE
    assert image.channel_names == ("DAPI", "GFP")
    assert chunk_reads == []


def test_only_the_chosen_fields_are_fetched_and_only_once(plate_in_memory,
                                                          chunk_reads,
                                                          tmp_path):
    url, arrays, _, _ = plate_in_memory
    dest = tmp_path / "staged"
    summary = ome_zarr._stage_ome_zarr(url, dest, wells="B3", fields=1,
                                       report=None)
    per_field = SHAPE[0] * SHAPE[1] * (SHAPE[2] // 32) * (SHAPE[3] // 32)
    assert summary["fetched"] == 1 and summary["written"] == 2
    assert len(chunk_reads) == per_field
    assert all("/B/3/0/0/" in path for path in chunk_reads)
    names = sorted(p.name for p in dest.glob("*.tif"))
    assert names == ["screen-1_B03_T0001F001L01A01Z01C01.tif",
                     "screen-1_B03_T0001F001L01A01Z01C02.tif"]
    expected = arrays[("B3", "0")].max(axis=1)
    for channel, name in enumerate(names):
        assert np.array_equal(tifffile.imread(dest / name), expected[channel])

    for name in names:
        (dest / name).unlink()
    chunk_reads.clear()
    again = ome_zarr._stage_ome_zarr(url, dest, wells="B3", fields=1,
                                     report=None)
    assert again["fetched"] == 0 and again["reused"] == 1
    assert chunk_reads == []
    record = json.loads((dest / "spacr_cloud_source.json").read_text())
    assert record["fields"]["B03/F001"]["z_projected"] is True
    assert record["fields"]["B03/F001"]["spacing"] == (
        "y 0.5 micrometer, x 0.5 micrometer")
    assert record["channel_names"] == ["DAPI", "GFP"]


def test_a_coarser_level_fetches_the_coarser_chunks(plate_in_memory,
                                                    chunk_reads, tmp_path):
    url, _, _, _ = plate_in_memory
    ome_zarr._stage_ome_zarr(url, tmp_path / "coarse", wells="A1", fields=1,
                             level=1, report=None)
    assert chunk_reads and all("/A/1/0/1/" in p for p in chunk_reads)
    plane = tifffile.imread(next((tmp_path / "coarse").glob("*C01.tif")))
    assert plane.shape == (32, 32)
    with pytest.raises(ome_zarr.OmeZarrError, match="no well H12"):
        ome_zarr._stage_ome_zarr(url, tmp_path / "bad", wells="H12",
                                 report=None)
    with pytest.raises(ome_zarr.OmeZarrError, match="cloud_level 5"):
        ome_zarr._stage_ome_zarr(url, tmp_path / "bad", wells="A1",
                                 level=5, report=None)


def test_mask_and_measure_share_one_local_folder(plate_in_memory, tmp_path):
    url, _, fs, bucket = plate_in_memory
    seen = []

    def fake_module(settings):
        seen.append(dict(settings))
        root = Path(settings["src"])
        if root.name == "merged":
            root = root.parent
        (root / "merged").mkdir(exist_ok=True)
        (root / "measurements").mkdir(exist_ok=True)
        (root / "measurements" / "measurements.db").write_bytes(b"rows")
        return "done"

    results = f"memory://{bucket.lstrip('/')}/results"
    mask = {"src": url, "cloud_cache": str(tmp_path / "cache"),
            "cloud_wells": "A01", "cloud_fields": 1,
            "metadata_type": "auto", "cloud_results": results}
    assert ome_zarr._needs_cloud_run(mask)
    assert ome_zarr._run_with_cloud_sources(fake_module, mask, "mask",
                                            report=None) == "done"
    staged = Path(seen[0]["src"])
    assert staged.parent == tmp_path / "cache"
    assert seen[0]["metadata_type"] == "cellvoyager"
    assert seen[0]["cloud_results"] == ""
    assert mask["cloud_results"] == results
    assert len(list(staged.glob("*.tif"))) == 2
    assert fs.cat_file(f"{bucket}/results/plate/measurements/"
                       f"measurements.db") == b"rows"

    measure = {"src": url + "/merged", "cloud_cache": str(tmp_path / "cache")}
    ome_zarr._run_with_cloud_sources(fake_module, measure, "measure",
                                     report=None)
    assert Path(seen[1]["src"]) == staged / "merged"

    fresh = {"src": url, "cloud_cache": str(tmp_path / "elsewhere")}
    with pytest.raises(ome_zarr._CloudStorageError, match="Make Masks"):
        ome_zarr._run_with_cloud_sources(fake_module, fresh, "measure",
                                         report=None)


def test_local_runs_are_left_alone():
    assert not ome_zarr._needs_cloud_run({"src": "/data/plate"})
    assert not ome_zarr._needs_cloud_run({"src": ["/a", "/b"],
                                          "cloud_results": ""})
    assert ome_zarr._needs_cloud_run({"src": "/a",
                                      "cloud_results": "s3://b/out"})


def test_a_cloud_folder_is_mirrored_and_cached(tmp_path):
    fs = fsspec.filesystem("memory")
    bucket = f"/bucket-{uuid.uuid4().hex[:8]}"
    try:
        fs.pipe(f"{bucket}/raw/plate1_A01_T0001F001L01A01Z01C01.tif", b"a")
        fs.pipe(f"{bucket}/raw/sub/plate1_A01_T0001F002L01A01Z01C01.tif",
                b"b")
        fs.pipe(f"{bucket}/raw/notes.bin", b"skipped")
        url = f"memory://{bucket.lstrip('/')}/raw"
        dest = tmp_path / "mirror"
        first = ome_zarr._sync_cloud_folder(url, dest, report=None)
        assert first["downloaded"] == 2
        assert (dest / "sub" / "plate1_A01_T0001F002L01A01Z01C01.tif"
                ).read_bytes() == b"b"
        assert not (dest / "notes.bin").exists()
        assert ome_zarr._sync_cloud_folder(url, dest, report=None
                                           )["downloaded"] == 0
        fs.pipe(f"{bucket}/raw/plate1_A01_T0001F001L01A01Z01C01.tif", b"new!")
        assert ome_zarr._sync_cloud_folder(url, dest, report=None
                                           )["downloaded"] == 1
        with pytest.raises(ome_zarr._CloudStorageError, match="no images"):
            ome_zarr._sync_cloud_folder(url + "/sub/none", tmp_path / "e",
                                        report=None)
    finally:
        fs.rm(bucket, recursive=True)


def test_tables_open_from_an_address_and_are_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    fs = fsspec.filesystem("memory")
    bucket = f"/bucket-{uuid.uuid4().hex[:8]}"
    try:
        fs.pipe(f"{bucket}/scores.csv", b"plate,row,col,score\np1,r1,c2,0.5\n")
        url = f"memory://{bucket.lstrip('/')}/scores.csv"
        frame = tabular.read_table(url, report=None)
        assert list(frame["score"]) == [0.5]
        assert "columnID" in tabular.table_columns(url)
        copies = list((tmp_path / "xdg" / "spacr" / "cloud" / "files"
                       ).glob("*/scores.csv"))
        assert len(copies) == 1
        fs.pipe(f"{bucket}/scores.csv", b"plate,row,col,score\np1,r1,c2,0.75\n")
        assert list(tabular.read_table(url, report=None)["score"]) == [0.75]
    finally:
        fs.rm(bucket, recursive=True)


def test_a_missing_library_names_its_install_line(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "gcsfs"
                        else real(name, *a))
    with pytest.raises(ImportError) as caught:
        ome_zarr._cloud_path("gs://bucket/plate.zarr")
    assert isinstance(caught.value, ome_zarr._CloudLibraryMissing)
    assert "python -m pip install fsspec gcsfs" in str(caught.value)


def test_credentials_are_chosen_never_carried(monkeypatch):
    options = ome_zarr._CloudOptions
    monkeypatch.setattr(ome_zarr, "_aws_credentials_found", lambda: True)
    assert ome_zarr._storage_options("s3", options()) == {}
    assert ome_zarr._storage_options("s3", options(profile="lab")) == {
        "profile": "lab"}
    assert ome_zarr._storage_options(
        "s3", options(anonymous=True, endpoint="https://minio.local")) == {
        "anon": True, "client_kwargs": {"endpoint_url": "https://minio.local"}}
    assert ome_zarr._storage_options("gs", options(anonymous=True)) == {
        "token": "anon"}
    monkeypatch.setattr(ome_zarr, "_aws_credentials_found", lambda: False)
    assert ome_zarr._storage_options("s3", options()) == {"anon": True}
    assert options().describe("s3") == (
        "anonymously, as no AWS credentials were found")
    loaded = options._from_settings({"cloud_anonymous": "True",
                                    "cloud_profile": " lab ",
                                    "cloud_endpoint": None})
    assert loaded == options(anonymous=True, profile="lab", endpoint="")


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serves files without logging each request to stderr."""

    def log_message(self, *args):
        """Say nothing."""


@pytest.fixture
def http_plate(tmp_path):
    """The plate served over HTTP from 127.0.0.1, as IDR serves its plates."""
    _write_plate(tmp_path / "www" / "plate.zarr")
    handler = functools.partial(_QuietHandler,
                                directory=str(tmp_path / "www"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/plate.zarr"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_a_plate_served_over_http_stages_without_leaking_secrets(
        http_plate, chunk_reads, tmp_path, capsys):
    pytest.importorskip("aiohttp")
    secret_url = http_plate.replace("http://", "http://me:hunter2@") \
        + "?X-Amz-Signature=topsecret"
    settings = {"src": secret_url, "cloud_cache": str(tmp_path / "cache"),
                "cloud_wells": "B03", "cloud_fields": 1}
    local = ome_zarr._localize_cloud_source(secret_url, settings, "mask")
    assert len(list(Path(local).glob("*.tif"))) == 2
    assert chunk_reads and all("/B/3/0/0/" in p for p in chunk_reads)
    printed = capsys.readouterr().out
    manifest = (Path(local) / "spacr_cloud_source.json").read_text()
    for text in (printed, manifest, local):
        assert "hunter2" not in text and "topsecret" not in text
    with pytest.raises(ome_zarr._CloudStorageError, match="cannot be listed"):
        ome_zarr._cloud_listing(http_plate)
