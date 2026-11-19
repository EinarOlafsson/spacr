"""Item 985 end to end: a real Measure PNG (Legacy) project converts to what Measure writes natively.

The same plate is measured twice by spacr.measure.measure_crop, once writing
PNG crops and once writing OME-Zarr crops. Converting the PNG project must
give the native run's crops, pixel for pixel and mask for mask, per class,
with every original kept; a shuffled class index must not match (control).

The project is what Measure leaves with its defaults (no pixel size, no
channel names, measurements.db in WAL mode), plus the plane layout Mask
Generation writes. Until 985 converts it, the setup ends in an expected
failure (2026-11-19 review): a leftover measurements.db-shm, pixel_size_um
None, channel_names None, and every measurement column copied into
ssCell_Crops as text so the converter's own check fails. The strict marker
turns this test red once the conversion works, so the marker comes off then.
"""
from __future__ import annotations

import hashlib
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("zarr")

from tests.test_ome_zarr_round_trip_986 import _measure


def _digest(pixels: np.ndarray, mask: np.ndarray) -> str:
    out = hashlib.md5()
    for array in (np.ascontiguousarray(pixels), np.ascontiguousarray(mask)):
        out.update(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes())
    return out.hexdigest()


def _crops(root: Path, table: pd.DataFrame):
    from spacr.crop_store import read_crop

    out = {}
    for row in table.to_dict("records"):
        pixels, mask = read_crop(root, row)
        out.setdefault(row["object_type"], []).append(_digest(pixels, mask))
    return {k: sorted(v) for k, v in out.items()}


def _table(root: Path) -> pd.DataFrame:
    with sqlite3.connect(str(root / "measurements" / "measurements.db")) as conn:
        return pd.read_sql("SELECT * FROM ssCell_Crops", conn)


pytestmark = pytest.mark.xfail(strict=True, reason="985 review 2026-11-19: real Measure "
                                "projects are not converted yet (see module docstring)")


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    from spacr import ome_zarr_convert

    base = tmp_path_factory.mktemp("e2e985")
    png = _measure(base / "png", "png")
    native = _measure(base / "native", "ome-zarr")
    import json

    from spacr.crops import MERGED_LAYOUT_SIDECAR

    (png / "merged" / MERGED_LAYOUT_SIDECAR).write_text(json.dumps({
        "version": 1, "intensity_channels": [0, 1],
        "mask_plane_order": ["cell", "nucleus"],
        "mask_dims": {"cell": 2, "nucleus": 3}}))
    before = sorted(str(p.relative_to(png)) for p in png.rglob("*") if p.is_file())
    try:
        report = ome_zarr_convert.convert_project(png, base / "converted")
    except ValueError as exc:
        pytest.xfail(f"985 does not yet convert a real Measure project: {exc}")
    return {"png": png, "native": native, "dst": base / "converted",
            "report": report, "before": before}


def test_a_measured_png_project_converts_to_the_native_ome_zarr_crops(converted):
    assert converted["report"]["status"] in {"verified", "raw_verified"}
    got = _crops(converted["dst"], _table(converted["dst"]))
    want = _crops(converted["native"], _table(converted["native"]))
    assert sorted(want) == ["cell", "nucleus"] and len(want["cell"]) == 9
    assert got == want


def test_every_original_is_kept_and_a_shuffled_index_does_not_match(converted):
    after = sorted(str(p.relative_to(converted["png"]))
                   for p in converted["png"].rglob("*") if p.is_file())
    assert after == converted["before"]
    table = _table(converted["dst"])
    want = _crops(converted["native"], _table(converted["native"]))
    misses = 0
    for seed in range(20):
        shuffled = table.copy()
        shuffled["object_type"] = np.random.default_rng(seed).permutation(
            table["object_type"].to_numpy())
        if (shuffled["object_type"] == table["object_type"]).all():
            continue
        misses += _crops(converted["dst"], shuffled) != want
    assert misses >= 19
