"""Item 984: Measure writes ragged OME-Zarr crops through its single writer."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("zarr")


def _plate(root):
    merged = root / "merged"
    merged.mkdir(parents=True)
    rng = np.random.default_rng(3)
    image = rng.integers(0, 60000, (48, 64, 2)).astype(np.uint16)
    cells = np.zeros((48, 64), np.uint16)
    cells[4:20, 6:30] = 1
    cells[26:44, 36:60] = 2
    nuclei = np.zeros_like(cells)
    nuclei[8:14, 10:20] = 1
    nuclei[30:38, 40:50] = 2
    np.save(merged / "plate1_A01_1.npy", np.dstack([image, cells, nuclei]))
    return merged, image, cells


def _settings(merged, **over):
    from spacr.settings import get_measure_crop_settings

    settings = get_measure_crop_settings({})
    settings.update({
        "src": str(merged), "channels": [0, 1],
        "cell_mask_dim": 2, "nucleus_mask_dim": 3,
        "pathogen_mask_dim": None, "cell_min_size": 0,
        "nucleus_min_size": 0, "pathogen_min_size": 0,
        "cytoplasm_min_size": 0, "cytoplasm": False, "save_png": True,
        "crop_mode": ["cell"], "save_arrays": False, "plot": False,
        "verbose": False, "n_jobs": 1,
    })
    settings.update(over)
    return settings


def _crop_table(root):
    with sqlite3.connect(str(root / "measurements" / "measurements.db")) as conn:
        return pd.read_sql("SELECT * FROM ssCell_Crops", conn)


def test_the_crop_settings_default_to_png_and_refuse_unknown_formats():
    from spacr.measure import _crop_image_format
    from spacr.settings import get_measure_crop_settings

    settings = get_measure_crop_settings({})
    assert settings["crop_image_format"] == "png"
    assert settings["crops_per_shard"] == 4096
    assert settings["crop_margin_px"] == 0 and settings["crop_zero_outside"] is False
    with pytest.raises(ValueError, match="crop_image_format must be one of"):
        _crop_image_format({"crop_image_format": "jpeg"})


def test_measure_writes_ome_zarr_crops_pixel_exact(tmp_path):
    from spacr.crop_store import OmeZarrCropSource, ome_zarr_crops_exist, read_crop
    from spacr.measure import measure_crop

    merged, image, cells = _plate(tmp_path)
    measure_crop(_settings(merged, crop_image_format="ome-zarr", crop_margin_px=2,
                           crop_mode=["cell", "nucleus"]))
    assert ome_zarr_crops_exist(tmp_path)
    everything = _crop_table(tmp_path)
    assert sorted(everything["object_type"]) == ["cell", "cell", "nucleus", "nucleus"]
    table = everything[everything["object_type"] == "cell"].reset_index(drop=True)
    assert list(table["object_label"]) == [1, 2]
    assert list(table["nucleus_id"]) == [1, 2]
    assert table["has_nucleus"].astype(bool).all() and list(table["n_nuclei"]) == [1, 1]
    assert set(table["crop_store"]) == {"crops/plate=plate1/well=A01.zarr"}
    for row in table.to_dict("records"):
        y0, x0, y1, x1 = (int(row[k]) for k in ("bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1"))
        pixels, mask = read_crop(tmp_path, row)
        assert pixels.dtype == np.uint16
        assert np.array_equal(pixels, np.moveaxis(image[y0:y1, x0:x1], -1, 0))
        assert np.array_equal(mask, cells[y0:y1, x0:x1] == row["object_label"])
    first = table.iloc[0]
    assert (first["bbox_y0"], first["bbox_x0"], first["bbox_y1"], first["bbox_x1"]) == (
        2, 4, 22, 32)
    source = OmeZarrCropSource(str(tmp_path), channels=[0, 1])
    rgb = source.get({"object_id": first["object_id"]})
    assert rgb.dtype == np.uint8 and rgb.shape == (20, 28, 3)
    assert not (tmp_path / "data").exists()


def test_the_png_default_writes_no_crop_store(tmp_path):
    from spacr.measure import measure_crop

    merged, _image, _cells = _plate(tmp_path)
    measure_crop(_settings(merged, png_size=[16, 16]))
    assert not (tmp_path / "crops").exists()
    with sqlite3.connect(str(tmp_path / "measurements" / "measurements.db")) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    assert "ssCell_Crops" not in names


def test_the_sink_commits_fields_in_plan_order_whatever_order_they_arrive(tmp_path):
    from spacr.crop_store import CropSink, cut_crops

    rng = np.random.default_rng(1)
    labels = np.zeros((30, 30), np.int32)
    labels[2:10, 2:12] = 1
    labels[15:28, 14:29] = 2
    fields = {f"plate1_A01_{i}": rng.integers(0, 999, (2, 30, 30)).astype(np.uint16)
              for i in (1, 2, 3)}

    def run(order, root):
        sink = CropSink(root, [f"{name}.npy" for name in fields], crops_per_shard=3)
        for name in order:
            field = int(name.rsplit("_", 1)[1])
            sink.receive(f"{name}.npy", [{
                "plate": "plate1", "well": "A01", "object_type": object_type,
                "channels": 2, "dtype": "<u2",
                "identity": {"field": field, "timepoint": None},
                "crops": cut_crops(fields[name], labels)}
                for object_type in ("cell", "nucleus")])
        return sink.close()

    a = run(["plate1_A01_3", "plate1_A01_1", "plate1_A01_2"], tmp_path / "a")
    b = run(["plate1_A01_1", "plate1_A01_2", "plate1_A01_3"], tmp_path / "b")
    key = ["object_type", "field", "object_label", "crop_array", "offset"]
    assert [[r[k] for k in key] for r in a] == [[r[k] for k in key] for r in b]
    assert [(r["object_type"], r["field"]) for r in a] == [
        ("cell", 1), ("cell", 1), ("cell", 2), ("cell", 2), ("cell", 3), ("cell", 3),
        ("nucleus", 1), ("nucleus", 1), ("nucleus", 2), ("nucleus", 2),
        ("nucleus", 3), ("nucleus", 3)]


def test_two_measure_workers_write_the_same_crops_as_one(tmp_path):
    from spacr.crop_store import read_crop
    from spacr.measure import measure_crop

    tables = {}
    for jobs in (1, 2):
        root = tmp_path / f"j{jobs}"
        merged, _image, _cells = _plate(root)
        stack = np.load(merged / "plate1_A01_1.npy")
        for name in ("plate1_A01_2", "plate1_B02_1"):
            np.save(merged / f"{name}.npy", stack[::-1].copy())
        measure_crop(_settings(merged, crop_image_format="ome-zarr", n_jobs=jobs,
                               crop_mode=["cell", "nucleus"]))
        table = _crop_table(root).sort_values("object_id").reset_index(drop=True)
        tables[jobs] = (root, table)
    (root1, one), (root2, two) = tables[1], tables[2]
    assert len(one) == 12
    cols = ["object_id", "crop_store", "crop_array", "offset", "height", "width"]
    assert one[cols].equals(two[cols])
    for a, b in zip(one.to_dict("records"), two.to_dict("records")):
        assert np.array_equal(read_crop(root1, a)[0], read_crop(root2, b)[0])


def _store_bytes(root):
    store = root / "crops"
    return {str(p.relative_to(store)): p.read_bytes()
            for p in sorted(store.rglob("*")) if p.is_file()}


def test_a_measure_rerun_replaces_the_wells_crops_as_a_clean_run_writes_them(tmp_path):
    from spacr.measure import measure_crop

    clean_root = tmp_path / "clean"
    merged, _image, _cells = _plate(clean_root)
    measure_crop(_settings(merged, crop_image_format="ome-zarr",
                           crop_mode=["cell", "nucleus"]))
    rerun_root = tmp_path / "rerun"
    merged2, _image, _cells = _plate(rerun_root)
    for _ in range(2):
        measure_crop(_settings(merged2, crop_image_format="ome-zarr",
                               crop_mode=["cell", "nucleus"]))
    assert _store_bytes(rerun_root) == _store_bytes(clean_root)
    clean, rerun = _crop_table(clean_root), _crop_table(rerun_root)
    assert len(rerun) == 4 and rerun.equals(clean)


def test_a_rerun_of_some_fields_keeps_the_others_and_matches_a_clean_run(tmp_path):
    from spacr.crop_store import CropSink, committed_rows, cut_crops

    rng = np.random.default_rng(4)
    labels = np.zeros((30, 30), np.int32)
    labels[2:10, 2:12] = 1
    labels[15:28, 14:29] = 2
    fields = {i: rng.integers(0, 999, (2, 30, 30)).astype(np.uint16) for i in (1, 2, 3)}

    def payload(i):
        return [{"plate": "plate1", "well": "A01", "object_type": "cell",
                 "channels": 2, "dtype": "<u2",
                 "identity": {"field": i, "timepoint": None},
                 "crops": [{**c, "object_id": f"plate1_A01_{i}_cell_{c['object_label']}"}
                           for c in cut_crops(fields[i], labels)]}]

    def run(root, which):
        sink = CropSink(root, [f"plate1_A01_{i}.npy" for i in which], crops_per_shard=3)
        for i in which:
            sink.receive(f"plate1_A01_{i}.npy", payload(i))
        sink.close()

    run(tmp_path / "clean", [1, 2, 3])
    run(tmp_path / "resumed", [1, 2, 3])
    run(tmp_path / "resumed", [3])
    run(tmp_path / "resumed", [2, 3])
    assert _store_bytes(tmp_path / "resumed") == _store_bytes(tmp_path / "clean")
    assert committed_rows(tmp_path / "resumed") == committed_rows(tmp_path / "clean")
    assert [r["field"] for r in committed_rows(tmp_path / "clean")] == [1, 1, 2, 2, 3, 3]


def test_crops_of_mask_generation_names_with_a_timepoint_get_the_right_well(tmp_path):
    from spacr.measure import measure_crop

    merged, _image, _cells = _plate(tmp_path)
    (merged / "plate1_A01_1.npy").rename(merged / "plate1_E01_10_1.npy")
    measure_crop(_settings(merged, crop_image_format="ome-zarr"))
    table = _crop_table(tmp_path)
    assert set(table["well"]) == {"E01"} and set(table["field"]) == {10}
    assert set(table["crop_store"]) == {"crops/plate=plate1/well=E01.zarr"}
    assert set(table["object_id"]) == {"plate1_E01_10_cell_1", "plate1_E01_10_cell_2"}


def test_the_crop_store_does_not_depend_on_the_folder_listing_order(tmp_path, monkeypatch):
    from spacr import io as spacr_io
    from spacr.measure import measure_crop

    listdir = spacr_io._listdir_visible
    stores = {}
    for order in ("forward", "reverse"):
        root = tmp_path / order
        merged, _image, _cells = _plate(root)
        stack = np.load(merged / "plate1_A01_1.npy")
        for name in ("plate1_A01_2", "plate1_A01_3", "plate1_A01_10"):
            np.save(merged / f"{name}.npy", stack[::-1].copy())

        def listing(folder, _reverse=(order == "reverse")):
            names = sorted(listdir(folder))
            return names[::-1] if _reverse else names

        monkeypatch.setattr(spacr_io, "_listdir_visible", listing)
        measure_crop(_settings(merged, crop_image_format="ome-zarr", n_jobs=2))
        stores[order] = _store_bytes(root)
    assert stores["forward"] and stores["forward"] == stores["reverse"]
