"""Crop readers take a run's spaCR store first, PNGs second (item 976, alpha).

``spacr.crops.resolve_crop_source`` is the one entry point: with
``measurements/spacr.duckdb`` holding crops for the object class it returns
the store source (unless ``prefer`` asks for another), and every crop it
returns equals the PNG source's crop for the same object. The direct-path
readers (Annotate's ``load_crop_image`` and the class-folder export) read
through ``_read_crop_preferring_store``, which falls back to the PNG.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("zarr")
pytest.importorskip("cv2")

from spacr.crops import (  # noqa: E402
    _STORE_SOURCES,
    PngCropSource,
    _read_crop_preferring_store,
    read_crop_png,
    resolve_crop_source,
)
from spacr.measure import _write_measure_store  # noqa: E402
from tests.test_measure_store_975 import _db  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_sources():
    """Each test opens its own store sources."""
    _STORE_SOURCES.clear()
    yield
    _STORE_SOURCES.clear()


def _rows(db):
    with sqlite3.connect(str(db), timeout=30) as conn:
        return [dict(zip(("png_path", "prcfo"), r)) for r in conn.execute(
            "SELECT png_path, prcfo FROM png_list")]


@pytest.mark.parametrize("fmt", [None, 2, 4])
def test_the_entry_point_takes_the_store_first_and_matches_the_pngs(tmp_path,
                                                                     fmt):
    db, _cell, _nucleus, _pixels = _db(tmp_path, n=5, fmt=fmt)
    root = str(tmp_path / "run")
    assert resolve_crop_source(root, object_type="cell").kind == "png"
    _write_measure_store(str(db), {"measurement_backend": "spacr_store"})
    source = resolve_crop_source(root, object_type="cell")
    assert source.kind == "store"
    assert resolve_crop_source(root, object_type="cell",
                               prefer="png").kind == "png"
    rows = _rows(db)
    for got, want in zip(source.get_many(rows),
                         PngCropSource(db_path=str(db)).get_many(rows)):
        assert np.array_equal(got, want) and got.dtype == want.dtype
    by_path = source.get_many([r["png_path"] for r in rows])
    assert all(np.array_equal(a, b) for a, b in zip(by_path, source.get_many(rows)))


def test_a_store_without_crops_falls_through_to_the_pngs(tmp_path):
    db, _cell, _nucleus, _pixels = _db(tmp_path, n=3)
    _write_measure_store(str(db), {"measurement_backend": "spacr_store"})
    root = str(tmp_path / "run")
    assert resolve_crop_source(root, object_type="nucleus").kind == "png"
    assert resolve_crop_source(root, object_type="cell",
                               prefer="store").kind == "store"


def test_the_direct_reader_prefers_the_store_and_falls_back(tmp_path):
    db, _cell, _nucleus, _pixels = _db(tmp_path, n=4)
    path = _rows(db)[1]["png_path"]
    want = read_crop_png(path, db_path=str(db))
    assert np.array_equal(_read_crop_preferring_store(path, db_path=str(db)),
                          want)
    _write_measure_store(str(db), {"measurement_backend": "spacr_store"})
    os.remove(path)
    assert np.array_equal(_read_crop_preferring_store(path, db_path=str(db)),
                          want)
    with pytest.raises(Exception):
        _read_crop_preferring_store(path)


def test_annotate_reads_the_same_image_from_the_store(tmp_path):
    from spacr.qt.annotate_engine import load_crop_image

    db, _cell, _nucleus, _pixels = _db(tmp_path, n=3)
    path = _rows(db)[0]["png_path"]
    before = np.asarray(load_crop_image(path, db_path=str(db)))
    _write_measure_store(str(db), {"measurement_backend": "spacr_store"})
    os.remove(path)
    after = np.asarray(load_crop_image(path, db_path=str(db)))
    assert np.array_equal(before, after)


def test_ome_zarr_crops_come_after_the_store_and_before_the_pngs(tmp_path,
                                                                 monkeypatch):
    import sys
    import types

    from spacr.crops import CropSource

    class _Fake(CropSource):
        """Item 983's source, as its interface is specified."""

        kind = "ome-zarr"

        def __init__(self, root=None, reason="", db_path=None, **_kw):
            self.reason = reason

    module = types.ModuleType("spacr.crop_store")
    module.OmeZarrCropSource = _Fake
    module.ome_zarr_crops_exist = lambda root: True
    monkeypatch.setitem(sys.modules, "spacr.crop_store", module)
    db, _cell, _nucleus, _pixels = _db(tmp_path, n=3)
    root = str(tmp_path / "run")
    assert resolve_crop_source(root, object_type="cell").kind == "ome-zarr"
    assert resolve_crop_source(root, object_type="cell",
                               prefer="png").kind == "png"
    _write_measure_store(str(db), {"measurement_backend": "spacr_store"})
    assert resolve_crop_source(root, object_type="cell").kind == "store"
    assert resolve_crop_source(root, object_type="cell",
                               prefer="ome-zarr").kind == "ome-zarr"
