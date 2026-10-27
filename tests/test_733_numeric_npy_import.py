"""Safe shared NPY Import uses headers for preview and read-only mapped pixels."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import convert as cv


def test_header_preview_never_loads_pixels(tmp_path, monkeypatch):
    image = tmp_path / "field.npy"
    np.save(image, np.arange(20, dtype=np.uint16).reshape(4, 5))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Preview loaded pixels"))
    sources = cv.scan(str(tmp_path))
    assert ".npy" in cv.IMAGE_EXTENSIONS
    assert cv.reader_available(".npy")
    assert len(sources) == 1 and sources[0].readable
    assert sources[0].meta["axes"] == "YX"
    assert sources[0].meta["shape"] == (4, 5)
    assert (sources[0].t, sources[0].z, sources[0].n_channels) == (1, 1, 1)
    assert cv.plan(sources).ok


@pytest.mark.parametrize("fortran", [False, True])
def test_conversion_maps_readonly_without_changing_source(tmp_path, monkeypatch, fortran):
    source = tmp_path / "source"
    source.mkdir()
    image = source / "field.npy"
    expected = np.arange(30, dtype=np.float32).reshape(5, 6)
    np.save(image, np.asfortranarray(expected) if fortran else expected)
    before = image.read_bytes()
    plan = cv.plan(cv.scan(str(source)))
    original_load = np.load
    calls = []
    def mapped(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)
    monkeypatch.setattr(np, "load", mapped)
    result = cv.convert(plan, str(tmp_path / "converted"))
    assert len(result.written) == 1
    assert calls and all(call.get("allow_pickle") is False and call.get("mmap_mode") == "r" for call in calls)
    converted = tmp_path / "converted" / result.written[0].target
    np.testing.assert_array_equal(tifffile.imread(converted), expected)
    assert image.read_bytes() == before
    assert (tmp_path / "converted" / cv.MAP_FILENAME).is_file()


@pytest.mark.parametrize("value,reason", [
    (np.array([[object()]], dtype=object), "real numeric"),
    (np.zeros((2, 3), dtype=[("x", "u2")]), "real numeric"),
    (np.full((2, 3), "value"), "real numeric"),
    (np.ones((2, 3), dtype=np.complex64), "real numeric"),
    (np.zeros((2, 3, 4), dtype=np.uint16), "axis selection"),
    (np.zeros((0, 3), dtype=np.uint16), "two-dimensional"),
])
def test_unsafe_or_ambiguous_headers_are_unreadable(tmp_path, monkeypatch, value, reason):
    np.save(tmp_path / "field.npy", value)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Refusal loaded pixels"))
    sources = cv.scan(str(tmp_path))
    assert len(sources) == 1 and not sources[0].readable
    assert reason in sources[0].error
    assert not cv.plan(sources).mappings


def test_truncated_data_oversized_header_and_seg_bundle_are_refused(tmp_path):
    image = tmp_path / "field.npy"
    np.save(image, np.ones((5, 6), np.uint16))
    image.write_bytes(image.read_bytes()[:-3])
    assert "truncated" in cv.scan(str(tmp_path))[0].error
    image.unlink()
    image = tmp_path / "large_header.npy"
    image.write_bytes(b"\x93NUMPY\x02\x00" + (65537).to_bytes(4, "little"))
    assert "too large" in cv.scan(str(tmp_path))[0].error
    image.unlink()
    np.save(tmp_path / "field_seg.npy", np.ones((5, 6), np.uint16))
    assert "separate editor" in cv.scan(str(tmp_path))[0].error


def test_changed_shape_is_refused_after_preview(tmp_path):
    image = tmp_path / "field.npy"
    np.save(image, np.ones((5, 6), np.uint16))
    source = cv.scan(str(tmp_path))[0]
    np.save(image, np.ones((6, 5), np.uint16))
    with pytest.raises(cv.ConfigurationError, match="changed after"):
        cv._read_source(source)


def test_foreign_import_accepts_numeric_npy_images_and_masks(tmp_path):
    from spacr import foreign
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    pixels = np.arange(48, dtype=np.uint16).reshape(6, 8)
    labels = np.zeros((6, 8), np.uint16)
    labels[1:4, 2:5] = 1
    np.save(images / "field.npy", pixels)
    np.save(masks / "field.npy", labels)
    table = pd.DataFrame({"ImageNumber": ["field.npy"], "ObjectNumber": [1], "Area": [9.]})
    plan = foreign.plan_import(str(images), {"cell": str(masks)}, table)
    assert plan.ok, plan.errors
    result = foreign.run_import(plan, str(tmp_path / "project"))
    assert Path(result.db_path).is_file()
    converted = next((tmp_path / "project" / "images").glob("*.tif"))
    np.testing.assert_array_equal(tifffile.imread(converted), pixels)
    assert list((tmp_path / "project" / "merged").glob("*.npy"))
