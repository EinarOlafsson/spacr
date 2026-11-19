"""Item 986's switch condition: OME-Zarr reproduces the old formats exactly.

Measure is run twice on the same plate, once with PNG (Legacy) crops and once
with OME-Zarr crops. Every PNG crop is rebuilt pixel-exact from the OME-Zarr
store, class membership matches png_list exactly, a shuffled index does not
(control), and a rerun writes byte-identical stores.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("zarr")

PNG_SIZE = [32, 32]
MAPPING = {"r": 0, "g": 1, "b": 0}


def _plate(root: Path) -> Path:
    merged = root / "merged"
    merged.mkdir(parents=True)
    for i, name in enumerate(["plate1_A01_1", "plate1_A01_2", "plate1_B03_1"]):
        rng = np.random.default_rng(10 + i)
        image = rng.integers(1, 60000, (72, 90, 2)).astype(np.uint16)
        cells = np.zeros((72, 90), np.uint16)
        cells[3:25, 4:30] = 1
        cells[30:60, 40:75] = 2
        cells[55:72, 0:20] = 3
        nuclei = np.zeros_like(cells)
        nuclei[8:16, 10:20] = 1
        nuclei[38:50, 50:62] = 2
        np.save(merged / f"{name}.npy", np.dstack([image, cells, nuclei]))
    return merged


def _measure(root: Path, fmt: str, **over) -> Path:
    from spacr.measure import measure_crop
    from spacr.settings import get_measure_crop_settings

    merged = _plate(root)
    settings = get_measure_crop_settings({})
    settings.update({
        "src": str(merged), "channels": [0, 1], "cell_mask_dim": 2,
        "nucleus_mask_dim": 3, "pathogen_mask_dim": None, "cell_min_size": 0,
        "nucleus_min_size": 0, "pathogen_min_size": 0, "cytoplasm_min_size": 0,
        "cytoplasm": False, "save_png": True, "crop_mode": ["cell", "nucleus"],
        "png_size": [PNG_SIZE, PNG_SIZE], "png_channel_mapping": dict(MAPPING),
        "crop_intensity": "raw", "normalize": False, "save_arrays": False,
        "plot": False, "verbose": False, "n_jobs": 1,
        "crop_image_format": fmt})
    settings.update(over)
    measure_crop(settings)
    return root


def _digest(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    return hashlib.md5(str(array.dtype).encode() + str(array.shape).encode()
                       + array.tobytes()).hexdigest()


def _png_crops(root: Path):
    """``{class: sorted digests}`` of every PNG crop, decoded at its stored bit depth."""
    from spacr.crops import _read_crop_array

    out = {}
    for path in sorted(root.rglob("*_png/*.png")):
        kind = path.parent.name[:-len("_png")]
        out.setdefault(kind, []).append(_digest(_read_crop_array(str(path))))
    return {k: sorted(v) for k, v in out.items()}


def _png_list_counts(root: Path):
    with sqlite3.connect(str(root / "measurements" / "measurements.db")) as conn:
        paths = [r[0] for r in conn.execute("SELECT png_path FROM png_list")]
    counts = {}
    for p in paths:
        kind = Path(p).parent.name[:-len("_png")]
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _zarr_views(root: Path, table: pd.DataFrame):
    from spacr.crop_store import legacy_png_view

    out = {}
    planes = [MAPPING["r"], MAPPING["g"], MAPPING["b"]]
    for row in table.to_dict("records"):
        view = legacy_png_view(root, row, PNG_SIZE, planes)
        out.setdefault(row["object_type"], []).append(_digest(view))
    return {k: sorted(v) for k, v in out.items()}


def _table(root: Path) -> pd.DataFrame:
    with sqlite3.connect(str(root / "measurements" / "measurements.db")) as conn:
        return pd.read_sql("SELECT * FROM ssCell_Crops", conn)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("rt986")
    return {"png": _measure(base / "png", "png"),
            "zarr": _measure(base / "zarr", "ome-zarr"),
            "again": _measure(base / "again", "ome-zarr")}


def test_every_png_crop_is_rebuilt_pixel_exact_from_ome_zarr(runs):
    png = _png_crops(runs["png"])
    assert sorted(png) == ["cell", "nucleus"] and len(png["cell"]) == 9
    assert _zarr_views(runs["zarr"], _table(runs["zarr"])) == png


def test_the_crop_table_matches_png_list_membership_and_a_shuffle_does_not(runs):
    table = _table(runs["zarr"])
    assert table.groupby("object_type").size().to_dict() == _png_list_counts(runs["png"])
    png = _png_crops(runs["png"])
    misses = 0
    for seed in range(20):
        shuffled = table.copy()
        shuffled["object_type"] = np.random.default_rng(seed).permutation(
            table["object_type"].to_numpy())
        if (shuffled["object_type"] == table["object_type"]).all():
            continue
        misses += _zarr_views(runs["zarr"], shuffled) != png
    assert misses >= 19


def test_png_mode_writes_no_crop_store_and_ome_zarr_writes_no_pngs(runs):
    assert not (runs["png"] / "crops").exists()
    assert not list(runs["zarr"].rglob("*_png/*.png"))


def test_a_rerun_writes_byte_identical_crop_stores(runs):
    def files(root):
        store = root / "crops"
        return {str(p.relative_to(store)): p.read_bytes()
                for p in sorted(store.rglob("*")) if p.is_file()}

    first, second = files(runs["zarr"]), files(runs["again"])
    assert first.keys() == second.keys() and len(first) > 4
    assert first == second
    cols = [c for c in _table(runs["zarr"]).columns]
    assert _table(runs["zarr"])[cols].equals(_table(runs["again"])[cols])
