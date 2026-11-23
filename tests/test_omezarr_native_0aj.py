"""Item 0aj: Mask merges straight into merged.ome.zarr; Measure and crops read it."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("zarr")


def _run_folder(root, names=("plate1_A01_1_1", "plate1_A01_2_1", "plate1_B03_1_1"),
                big_label=False):
    """A Mask run before the merge: stack/*.npy and masks/<role>_mask_stack/*.npy."""
    stack = root / "stack"
    stack.mkdir(parents=True)
    for role in ("cell", "nucleus"):
        (root / "masks" / f"{role}_mask_stack").mkdir(parents=True)
    fields = {}
    for i, name in enumerate(names):
        rng = np.random.default_rng(i)
        image = rng.integers(0, 60000, (300, 270, 3)).astype(np.uint16)
        cell = np.zeros((300, 270), np.int32)
        cell[10:120, 20:150] = 1
        cell[150:290, 100:260] = 70000 if big_label else 2
        nucleus = np.zeros((300, 270), np.uint16)
        nucleus[30:60, 40:80] = 1
        nucleus[200:240, 150:200] = 2
        np.save(stack / f"{name}.npy", image)
        np.save(root / "masks" / "cell_mask_stack" / f"{name}.npy", cell)
        np.save(root / "masks" / "nucleus_mask_stack" / f"{name}.npy", nucleus)
        fields[name] = (image, cell, nucleus)
    return fields


def _merge(root, fmt, resume=False):
    from spacr.io import _load_and_concatenate_arrays

    _load_and_concatenate_arrays(str(root), [0, 1, 2], 1, 0, None, None,
                                 resume=resume, image_format=fmt)


@pytest.mark.parametrize("big_label", [False, True])
def test_the_store_field_equals_the_npy_merge_exactly(tmp_path, big_label):
    from spacr.image_store import _list_fields, _read_field

    a, b = tmp_path / "npy", tmp_path / "zarr"
    _run_folder(a, big_label=big_label)
    _run_folder(b, big_label=big_label)
    _merge(a, "npy")
    _merge(b, "ome-zarr")
    store = b / "merged.ome.zarr"
    assert not (b / "merged").exists()
    names = _list_fields(store)
    assert names == sorted(n for n in os.listdir(a / "merged") if n.endswith(".npy"))
    for name in names:
        want = np.load(a / "merged" / name)
        got = _read_field(store / name)
        assert got.dtype == want.dtype and np.array_equal(got, want)
    meta = json.loads((store / "A/1/1/labels/cell/0/zarr.json").read_text())
    assert meta["data_type"] == ("uint32" if big_label else "uint16")
    image = json.loads((store / "A/1/1/0/zarr.json").read_text())
    assert image["data_type"] == "uint16"
    assert image["codecs"][0]["name"] == "sharding_indexed"
    assert image["codecs"][0]["configuration"]["chunk_shape"] == [1, 256, 256]
    shard_files = [f for f in (store / "A/1/1/0/c").rglob("*") if f.is_file()]
    assert len(shard_files) == 1
    from spacr.crops import read_merged_plane_layout
    layout = read_merged_plane_layout(str(store))
    assert layout["mask_dims"] == {"cell": 3, "nucleus": 4}


def test_resume_skips_only_committed_fields(tmp_path, capsys):
    fields = _run_folder(tmp_path)
    _merge(tmp_path, "ome-zarr")
    ledger = tmp_path / "merged.ome.zarr" / "spacr_commits.jsonl"
    lines = ledger.read_text().splitlines()
    ledger.write_text("\n".join(lines[:1]) + "\n")
    _merge(tmp_path, "ome-zarr", resume=True)
    entries = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(entries) == len(fields)
    assert {e["name"] for e in entries} == {f"{n}.npy" for n in fields}


def test_two_plates_fall_back_to_npy_with_a_note(tmp_path, capsys):
    _run_folder(tmp_path, names=("plate1_A01_1_1", "plate2_A01_1_1"))
    _merge(tmp_path, "ome-zarr")
    assert (tmp_path / "merged" / "plate1_A01_1_1.npy").is_file()
    assert not (tmp_path / "merged.ome.zarr").exists()
    assert "one store holds one plate" in capsys.readouterr().out


def test_cleanup_trusts_the_store_commits(tmp_path):
    from spacr.utils import cleanup_pipeline_folders

    _run_folder(tmp_path)
    _merge(tmp_path, "ome-zarr")
    deleted = cleanup_pipeline_folders(str(tmp_path), keep_intermediate=False,
                                       keep_original=True, verbose=False)
    assert str(tmp_path / "stack") in deleted and not (tmp_path / "masks").exists()
    assert (tmp_path / "merged.ome.zarr").is_dir()


def test_cleanup_keeps_intermediates_when_a_field_is_uncommitted(tmp_path):
    from spacr.utils import cleanup_pipeline_folders

    _run_folder(tmp_path)
    _merge(tmp_path, "ome-zarr")
    np.save(tmp_path / "stack" / "plate1_C05_1_1.npy", np.zeros((4, 4, 3), np.uint16))
    cleanup_pipeline_folders(str(tmp_path), keep_intermediate=False,
                             keep_original=True, verbose=False)
    assert (tmp_path / "stack").is_dir()


def test_measure_resolves_and_lists_the_store(tmp_path):
    from spacr.image_store import _field_header, _list_fields
    from spacr.measure import _plate_folder_of_merged, _read_measure_field
    from spacr.validate import _resolve_measure_src

    fields = _run_folder(tmp_path)
    _merge(tmp_path, "ome-zarr")
    store = tmp_path / "merged.ome.zarr"
    for given in (tmp_path, tmp_path / "merged", store, tmp_path / "measurements"):
        assert _resolve_measure_src(str(given))[0] == str(store)
    assert _plate_folder_of_merged(str(store)) == str(tmp_path)
    name = _list_fields(store)[0]
    image, cell, nucleus = fields[os.path.splitext(name)[0]]
    data = _read_measure_field(str(store / name), {})
    assert np.array_equal(data[..., :3], image) and np.array_equal(data[..., 3], cell)
    assert np.array_equal(data[..., 4], nucleus)
    assert _field_header(store / name) == (data.shape, data.dtype)


def test_old_projects_still_read(tmp_path):
    from spacr.measure import _read_measure_field
    from spacr.validate import _resolve_measure_src

    _run_folder(tmp_path)
    _merge(tmp_path, "npy")
    merged = tmp_path / "merged"
    assert _resolve_measure_src(str(tmp_path))[0] == str(merged)
    name = sorted(n for n in os.listdir(merged) if n.endswith(".npy"))[0]
    assert np.array_equal(_read_measure_field(str(merged / name), {}),
                          np.load(merged / name))


def test_crops_keep_every_class_and_rebuild_the_own_mask(tmp_path):
    from spacr.crop_store import (CROP_STORE_NAME, CropPoolWriter, committed_rows,
                                  cut_crops, read_crop, read_crop_labels)

    rng = np.random.default_rng(3)
    image = rng.integers(0, 60000, (3, 120, 100)).astype(np.uint16)
    cell = np.zeros((120, 100), np.uint32)
    cell[5:60, 5:50] = 7
    cell[70:115, 40:95] = 70000
    nucleus = np.zeros((120, 100), np.uint16)
    nucleus[20:30, 20:30] = 3
    masks = {"cell": cell, "nucleus": nucleus}
    crops = cut_crops(image, cell, margin=4, all_labels=masks)
    writer = CropPoolWriter(tmp_path, "plate1", "A01", "cell", 3, np.uint16)
    writer.add(crops, field=1, timepoint=None)
    writer.commit()
    rows = committed_rows(tmp_path)
    assert rows and rows[0]["crop_store"].startswith(CROP_STORE_NAME)
    for row, crop in zip(rows, crops):
        pixels, own = read_crop(tmp_path, row)
        labels = read_crop_labels(tmp_path, row)
        assert row["label_classes"] == "cell,nucleus"
        box = slice(row["bbox_y0"], row["bbox_y1"]), slice(row["bbox_x0"], row["bbox_x1"])
        assert np.array_equal(pixels, image[(slice(None),) + box])
        assert np.array_equal(own, crop["mask"])
        assert np.array_equal(labels["cell"] == row["object_label"], own)
        assert np.array_equal(labels["nucleus"], nucleus[box])
        assert np.array_equal(labels["cell"], cell[box])
    assert not (tmp_path / "crops").exists()


def test_old_crop_folders_still_read(tmp_path):
    from spacr.crop_store import class_dir, committed_rows, cut_crops, read_crop
    from spacr.crop_store import CropPoolWriter

    image = np.ones((2, 40, 40), np.uint16)
    labels = np.zeros((40, 40), np.uint16)
    labels[5:15, 5:15] = 1
    writer = CropPoolWriter(tmp_path, "plate1", "A01", "cell", 2, np.uint16)
    writer.add(cut_crops(image, labels), field=1, timepoint=None)
    writer.commit()
    new = tmp_path / "crops.ome.zarr" / "plate=plate1" / "well=A01"
    old = tmp_path / "crops" / "plate=plate1" / "well=A01.zarr"
    old.parent.mkdir(parents=True)
    os.replace(new, old)
    import shutil
    shutil.rmtree(tmp_path / "crops.ome.zarr")
    rows = committed_rows(tmp_path)
    for row in rows:
        row["crop_store"] = "crops/plate=plate1/well=A01.zarr"
    assert class_dir(tmp_path, "plate1", "A01", "cell") == old / "cell"
    pixels, mask = read_crop(tmp_path, rows[0])
    assert mask.sum() == 100 and pixels.shape == (2, 10, 10)


def test_the_chaining_port_finds_the_store_fields(tmp_path):
    from spacr.ports import MERGED_ARRAYS, module_ports, port_problems, resolve_port

    fields = _run_folder(tmp_path)
    _merge(tmp_path, "ome-zarr")
    port = next(p for p in module_ports("measure").consumes if p.kind == MERGED_ARRAYS)
    resolved = resolve_port(port, str(tmp_path))
    assert resolved.exists and resolved.count == len(fields)
    assert port_problems(port, str(tmp_path)) == ()
