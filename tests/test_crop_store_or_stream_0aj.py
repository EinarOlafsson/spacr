"""Item 0aj crop decision (maintainer, 2026-11-23 16:30).

Saving crops ON writes project/crop.ome.zarr through the shared writer;
OFF streams each crop from fov.ome.zarr and its labels. Both go through ONE
reader (crop_store.read_crop / read_crops / read_crop_labels) and give the
same crops byte for byte: pixels, the own-object mask and every class's
labels. The crop index goes through spacr.store.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("zarr")
pytest.importorskip("duckdb")

from tests.test_omezarr_native_0aj import _merge, _run_folder  # noqa: E402


def _project(root):
    fields = _run_folder(root)
    _merge(root, "ome-zarr")
    return fields


def _settings(project, **over):
    from spacr.settings import get_measure_crop_settings

    settings = get_measure_crop_settings({})
    settings.update({
        "src": str(project / "fov.ome.zarr"), "channels": [0, 1, 2],
        "cell_mask_dim": 3, "nucleus_mask_dim": 4, "pathogen_mask_dim": None,
        "cell_min_size": 0, "nucleus_min_size": 0, "pathogen_min_size": 0,
        "cytoplasm_min_size": 0, "cytoplasm": False, "save_png": True,
        "crop_mode": ["cell", "nucleus"], "save_arrays": False, "plot": False,
        "verbose": False, "n_jobs": 1, "crop_margin_px": 3,
    })
    settings.update(over)
    return settings


def _index(project):
    from spacr.crop_store import _read_crop_index

    frame = _read_crop_index(project)
    return {row["object_id"]: row for row in frame.to_dict("records")}


@pytest.mark.parametrize("zero_outside", [False, True])
def test_stored_crops_equal_streamed_crops_byte_for_byte(tmp_path, zero_outside):
    from spacr.crop_store import CROP_STORE_NAME, read_crop, read_crop_labels, read_crops
    from spacr.measure import measure_crop

    saved, streamed = tmp_path / "saved", tmp_path / "streamed"
    _project(saved)
    _project(streamed)
    # A size limit drops objects, so Measure's own labels differ from Mask's:
    # the streamed cut must use the labels Measure measured with.
    common = dict(crop_zero_outside=zero_outside, nucleus_min_size=1500)
    measure_crop(_settings(saved, save_png=True, **common))
    measure_crop(_settings(streamed, save_png=False, **common))
    assert (saved / CROP_STORE_NAME).is_dir() and not (saved / "crops.ome.zarr").exists()
    assert not (streamed / CROP_STORE_NAME).exists()
    stored, cut = _index(saved), _index(streamed)
    assert stored and sorted(stored) == sorted(cut)
    # The 30 x 40 nucleus (1200 px) is below the limit: not a crop, and gone
    # from the labels every crop carries.
    nuclei = sorted(r["object_label"] for r in stored.values() if r["object_type"] == "nucleus")
    assert nuclei == [2, 2, 2]
    assert all(row["crop_array"] for row in stored.values())
    assert not any(row["crop_array"] for row in cut.values())
    for object_id, row in stored.items():
        pixels, mask = read_crop(saved, row)
        s_pixels, s_mask = read_crop(streamed, cut[object_id])
        assert pixels.dtype == s_pixels.dtype and pixels.tobytes() == s_pixels.tobytes()
        assert mask.dtype == s_mask.dtype and mask.tobytes() == s_mask.tobytes()
        labels, s_labels = read_crop_labels(saved, row), read_crop_labels(streamed, cut[object_id])
        assert sorted(labels) == sorted(s_labels)
        assert {"cell", "nucleus"} <= set(labels)
        for cls in labels:
            assert np.array_equal(labels[cls], s_labels[cls])
        assert 1 not in np.unique(s_labels["nucleus"])  # the dropped nucleus
        own = labels[row["object_type"]] == row["object_label"]
        assert np.array_equal(own, mask.astype(bool))
    batch = read_crops(streamed, list(cut.values()), context=False)
    for (pixels, mask), row in zip(batch, cut.values()):
        want = read_crop(saved, stored[row["object_id"]], context=False)
        assert pixels.tobytes() == want[0].tobytes() and mask.tobytes() == want[1].tobytes()


def test_stored_crops_are_the_raw_channels_and_all_labels(tmp_path):
    from spacr.crop_store import read_crop, read_crop_labels
    from spacr.measure import measure_crop

    fields = _project(tmp_path)
    measure_crop(_settings(tmp_path))
    for row in _index(tmp_path).values():
        image, cell, nucleus = fields[f"plate1_{row['well']}_{row['field']}_1"]
        box = (slice(int(row["bbox_y0"]), int(row["bbox_y1"])),
               slice(int(row["bbox_x0"]), int(row["bbox_x1"])))
        pixels, _mask = read_crop(tmp_path, row)
        assert np.array_equal(pixels, np.moveaxis(image[box], -1, 0))
        labels = read_crop_labels(tmp_path, row)
        assert np.array_equal(labels["cell"], cell[box])
        assert np.array_equal(labels["nucleus"], nucleus[box])


def test_the_crop_index_lives_in_spacr_store_and_ssCell_Crops_mirrors_it(tmp_path):
    import sqlite3

    from spacr import store
    from spacr.crop_store import CROP_INDEX_TABLE
    from spacr.measure import measure_crop

    _project(tmp_path)
    measure_crop(_settings(tmp_path, save_png=False))
    with store.open(str(tmp_path)) as opened:
        assert CROP_INDEX_TABLE in opened.tables()
        frame = opened.query(table=CROP_INDEX_TABLE)
    assert {"plate", "well", "field", "object_type", "object_label",
            "object_id", "bbox_y0", "source_field"} <= set(frame.columns)
    with sqlite3.connect(str(tmp_path / "measurements" / "measurements.db")) as conn:
        mirrored = {r[0] for r in conn.execute("SELECT object_id FROM ssCell_Crops")}
    assert mirrored == set(frame["object_id"])


def test_older_crop_stores_still_read(tmp_path):
    from spacr.crop_store import (CROP_STORE_NAME, CropPoolWriter, class_dir,
                                  committed_rows, cut_crops, read_crop)

    image = np.arange(2 * 40 * 40, dtype=np.uint16).reshape(2, 40, 40)
    labels = np.zeros((40, 40), np.uint16)
    labels[5:15, 5:15] = 1
    writer = CropPoolWriter(tmp_path, "plate1", "A01", "cell", 2, np.uint16)
    writer.add(cut_crops(image, labels), field=1, timepoint=None)
    writer.commit()
    os.replace(tmp_path / CROP_STORE_NAME, tmp_path / "crops.ome.zarr")
    rows = committed_rows(tmp_path)
    assert len(rows) == 1
    rows[0]["crop_store"] = "crops.ome.zarr/plate=plate1/well=A01"
    assert class_dir(tmp_path, "plate1", "A01", "cell").parts[-4] == "crops.ome.zarr"
    pixels, mask = read_crop(tmp_path, rows[0])
    assert np.array_equal(pixels, image[:, 5:15, 5:15]) and mask.all()


@pytest.mark.parametrize("save", [True, False])
def test_a_training_dataset_is_its_own_ome_zarr_store_under_datasets(tmp_path, save):
    from spacr.crop_store import (dataset_store, read_crop, read_crop_labels,
                                  read_training_dataset, write_training_dataset)
    from spacr.measure import measure_crop

    _project(tmp_path)
    measure_crop(_settings(tmp_path, save_png=save))
    source = list(_index(tmp_path).values())
    chosen = [dict(r, **{"class": "a" if i % 2 else "b", "split": "train" if i < 4 else "test"})
              for i, r in enumerate(source)]
    written = write_training_dataset(tmp_path, "set1", chosen)
    store = dataset_store(tmp_path, "set1")
    assert store == tmp_path / "datasets" / "set1.ome.zarr" and store.is_dir()
    assert (store / "spacr_dataset.json").is_file()
    back = {r["source_object_id"]: r for r in read_training_dataset(tmp_path, "set1")}
    assert len(back) == len(chosen) == len(written)
    for row, new in zip(chosen, written):
        assert new["crop_store"].startswith("datasets/set1.ome.zarr/")
        stored = back[row["object_id"]]
        assert (stored["class"], stored["split"]) == (row["class"], row["split"])
        assert (stored["plate"], stored["well"], stored["field"]) == (row["plate"], row["well"], row["field"])
        want, got = read_crop(tmp_path, row), read_crop(tmp_path, stored)
        assert want[0].tobytes() == got[0].tobytes() and np.array_equal(want[1], got[1])
        for cls, plane in read_crop_labels(tmp_path, row).items():
            assert np.array_equal(read_crop_labels(tmp_path, stored)[cls], plane)


@pytest.mark.parametrize("save", [True, False])
def test_the_stored_transform_maps_every_crop_pixel_onto_its_fov_pixel(tmp_path, save):
    """Crops (and dataset crops) record a 0.6 scale+translation into their source FOV."""
    import json

    from spacr.crop_store import (crop_to_fov, read_crop, read_crop_labels,
                                  read_training_dataset, write_training_dataset)
    from spacr.image_store import read_image, read_labels
    from spacr.measure import measure_crop

    _project(tmp_path)
    measure_crop(_settings(tmp_path, save_png=save, crop_margin_px=5))
    rows = list(_index(tmp_path).values())
    write_training_dataset(tmp_path, "set1", [dict(r, **{"class": "a", "split": "train"}) for r in rows])
    for row in rows + read_training_dataset(tmp_path, "set1"):
        record = json.loads(row["coordinate_transformations"])
        assert record["source_store"] == "fov.ome.zarr" and record["output"] == f"fov:{record['source_key']}"
        assert [t["type"] for t in record["transformations"]] == ["scale", "translation"]
        fov = read_image(tmp_path / "fov.ome.zarr", key=record["source_key"])
        pixels, _mask = read_crop(tmp_path, row)
        ys, xs = np.indices(pixels.shape[1:])
        fy, fx = crop_to_fov(row, ys, xs)
        assert np.array_equal(fy, np.round(fy)) and np.array_equal(fx, np.round(fx))
        region = fov[:, fy.astype(int), fx.astype(int)]
        assert region.tobytes() == pixels.tobytes()
        nucleus = read_labels(tmp_path / "fov.ome.zarr", "nucleus", key=record["source_key"])
        # The measured labels equal Mask's here (no size filter), so the label
        # crop is the FOV label region the transform points to.
        assert np.array_equal(read_crop_labels(tmp_path, row)["nucleus"],
                              nucleus[fy.astype(int), fx.astype(int)])
