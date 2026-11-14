"""Cloud OME-Zarr sources in ``spacr.ome_zarr``: addresses, staging, localising.

fsspec's in-memory store stands in for a bucket (the ``plate_in_memory``
fixture from ``test_cloud_storage_access``: plate ``screen_1`` with wells A01
and B03, two fields each, channels DAPI and GFP, z maximum projected).
Nothing leaves the machine.  The tests pin the exact keys a cloud path
resolves to, the exact TIFF names and counts a staging writes, and the exact
refusals for sources a run cannot read.
"""
from __future__ import annotations

from pathlib import Path

import pytest

fsspec = pytest.importorskip("fsspec")

from spacr import ome_zarr  # noqa: E402
from tests.test_cloud_storage_access import plate_in_memory  # noqa: E402,F401


def _bucket_url(bucket, *parts):
    """``memory://`` address of ``parts`` inside the fixture's bucket."""
    return "/".join([f"memory://{bucket.lstrip('/')}", *parts])


def _tifs(folder: Path):
    """Sorted names of the TIFFs a staging wrote into ``folder``."""
    return sorted(p.name for p in folder.glob("*.tif"))


def test_cloud_path_resolves_addresses_to_store_keys(plate_in_memory,  # noqa: F811
                                                    tmp_path, subtests,
                                                    monkeypatch):
    """A cloud address becomes a ``_CloudPath`` with the store's key (and an
    existing one is returned as is); HTTP keeps its query and fragment out of
    the key; a local path is refused; a store refusing access reads as
    missing rather than raising."""
    url, _arrays, _fs, bucket = plate_in_memory
    key = f"{bucket}/plate.zarr"

    with subtests.test(msg="memory address"):
        path = ome_zarr._cloud_path(url)
        assert ome_zarr._cloud_protocol(path) == "memory"
        assert ome_zarr._cloud_path(path) is path
        assert (path.protocol, path.key, path.name) == ("memory", key,
                                                        "plate.zarr")
        assert path.url == str(path) == url
        assert repr(path) == f"_CloudPath({url!r})"
        assert path.parent.key == bucket
        assert path.joinpath("", "A", "/1/").key == f"{key}/A/1"
        assert (path / "B").key == f"{key}/B"
        assert path.is_dir() and path.exists() and not path.is_file()

    with subtests.test(msg="local paths are not cloud addresses"):
        assert ome_zarr._redact_url("/local/plate.zarr") == "/local/plate.zarr"
        for call in (ome_zarr._cloud_path, ome_zarr._cloud_listing):
            with pytest.raises(ValueError, match="is not a cloud address"):
                call(str(tmp_path))

    with subtests.test(msg="store refusing access reads as missing"):
        missing = ome_zarr._cloud_path(url + "/missing")

        class _Refusing:
            def isfile(self, key):
                raise PermissionError("denied")

            def exists(self, key):
                raise PermissionError("denied")

        refusing = ome_zarr._CloudPath(_Refusing(), missing.protocol,
                                       missing.key)
        assert (refusing.is_file(), refusing.exists(),
                refusing.is_dir()) == (False, False, False)

    with subtests.test(msg="http keeps its query out of the key"):
        monkeypatch.setattr(ome_zarr, "_cloud_filesystem",
                            lambda protocol, options: object())
        path = ome_zarr._cloud_path("https://host/a/plate.zarr?sig=1#frag")
        assert path.protocol == "https"
        assert path.key == path.url == "https://host/a/plate.zarr"


def test_stage_ome_zarr_writes_named_tiffs_for_a_well_or_an_image(
        plate_in_memory, tmp_path, subtests):  # noqa: F811
    """A well is staged under its plate's name (the folder name when it is
    not ``*.zarr``), a single image as well A01; a re-run reuses recorded
    fields; a plain group or an unknown well is refused."""
    url, _arrays, fs, bucket = plate_in_memory
    fs.copy(f"{bucket}/plate.zarr", f"{bucket}/screen7", recursive=True)
    names = ["{p}_{w}_T0001F001L01A01Z01C01.tif",
             "{p}_{w}_T0001F001L01A01Z01C02.tif"]
    cases = [
        ("well of a .zarr plate", f"{url}/B/3", {"fields": 1}, "well",
         "plate", "B03", "OME-Zarr well: 2 fields\n0 — OME-NGFF 0.4"),
        ("well outside a .zarr-named plate",
         _bucket_url(bucket, "screen7", "B", "3"), {"fields": 1}, "well",
         "screen7", "B03", "OME-Zarr well: 2 fields\n0 — OME-NGFF 0.4"),
        ("single image", f"{url}/A/1/0", {}, "image", "0", "A01",
         "0 — OME-NGFF 0.4, 2 levels, axes c, z, y, x (0)"),
    ]
    for index, (case_id, source, kwargs, kind, plate, well,
                described) in enumerate(cases):
        with subtests.test(msg=case_id):
            dest = tmp_path / f"stage{index}"
            assert ome_zarr._cloud_source_kind(source) == kind
            text = ome_zarr._describe_cloud_source(source)
            assert text.startswith(described)
            assert "channels: DAPI, GFP" in text
            summary = ome_zarr._stage_ome_zarr(source, dest, report=None,
                                               **kwargs)
            assert summary == {"fetched": 1, "reused": 0, "written": 2,
                               "wells": [well],
                               "channel_names": ["DAPI", "GFP"]}
            assert _tifs(dest) == [n.format(p=plate, w=well) for n in names]

    with subtests.test(msg="a re-run reuses the recorded field"):
        said = []
        again = ome_zarr._stage_ome_zarr(f"{url}/B/3", tmp_path / "stage0",
                                         fields=1, report=said.append)
        assert (again["fetched"], again["reused"], again["written"]) == (0, 1,
                                                                         0)
        assert said == ["cloud: B03 field 1 already fetched"]

    with subtests.test(msg="a plain group has no fields"):
        row = f"{url}/A"
        assert ome_zarr._cloud_source_kind(row) == "group"
        assert ome_zarr._describe_cloud_source(row) == f"group: {row}"
        with pytest.raises(ome_zarr.OmeZarrError,
                           match="not an OME-Zarr plate, well or image"):
            ome_zarr._stage_ome_zarr(row, tmp_path / "g", report=None)
        assert _tifs(tmp_path / "g") == []

    with subtests.test(msg="a well the plate does not have"):
        with pytest.raises(ome_zarr.OmeZarrError,
                           match="has no well C07; it has 2 wells "
                                 r"\(A01, B03\)"):
            ome_zarr._stage_ome_zarr(url, tmp_path / "w", wells="C07",
                                     report=None)


def test_localize_cloud_source_stages_mirrors_or_refuses(plate_in_memory,  # noqa: F811
                                                         tmp_path, subtests):
    """An OME-Zarr is staged for Make Masks (switching ``metadata_type`` to
    ``cellvoyager`` and saying so, silently when it already is), a cloud
    folder of images is mirrored, and a single file or an unstaged Measure
    source is refused."""
    url, _arrays, fs, bucket = plate_in_memory

    for case_id, metadata, announced in (("already cellvoyager",
                                          "cellvoyager", False),
                                         ("unset", None, True)):
        with subtests.test(msg=f"plate staged, metadata_type {case_id}"):
            said = []
            settings = {"cloud_cache": str(tmp_path / case_id),
                        "cloud_wells": "B03", "cloud_fields": 1}
            if metadata:
                settings["metadata_type"] = metadata
            local = Path(ome_zarr._localize_cloud_source(
                url, settings, "mask", report=said.append))
            assert local.parent == tmp_path / case_id
            assert local.name.startswith("plate.zarr-")
            # A whole plate is named by its metadata name, sanitised.
            assert _tifs(local) == ["screen-1_B03_T0001F001L01A01Z01C01.tif",
                                    "screen-1_B03_T0001F001L01A01Z01C02.tif"]
            assert said[1:3] == [
                "cloud: B03 field 1 fetched, 2 images",
                "cloud: 1 fields fetched, 0 reused, from 1 wells; channels "
                "['DAPI', 'GFP']"]
            switched = [line for line in said if "metadata_type set" in line]
            assert switched == (
                ["cloud: metadata_type set to 'cellvoyager' (was None), the "
                 "naming of the fetched TIFFs"] if announced else [])
            assert settings["metadata_type"] == "cellvoyager"

    with subtests.test(msg="a cloud folder of images is mirrored"):
        fs.pipe(f"{bucket}/images/a.tif", b"x")
        said = []
        local = Path(ome_zarr._localize_cloud_source(
            _bucket_url(bucket, "images"),
            {"cloud_cache": str(tmp_path / "cache")}, "mask",
            report=said.append))
        assert local.parent == tmp_path / "cache"
        assert (local / "a.tif").read_bytes() == b"x"
        assert said[-1] == (f"cloud: 1 files downloaded, 0 already cached "
                            f"from {_bucket_url(bucket, 'images')}")

    with subtests.test(msg="a single file is not a source"):
        fs.pipe(f"{bucket}/one.tif", b"not really a tiff")
        with pytest.raises(ome_zarr._CloudStorageError,
                           match="is a single file; src needs a folder"):
            ome_zarr._localize_cloud_source(
                _bucket_url(bucket, "one.tif"),
                {"cloud_cache": str(tmp_path / "file")}, "mask", report=None)

    with subtests.test(msg="measure before make masks"):
        with pytest.raises(ome_zarr._CloudStorageError,
                           match="has not been through Make Masks yet"):
            ome_zarr._localize_cloud_source(
                f"{url}/merged", {"cloud_cache": str(tmp_path / "m")},
                "measure", report=None)
