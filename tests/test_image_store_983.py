"""Item 983: the OME-Zarr plate store, its single writer and the any-format readers."""
from __future__ import annotations

import json

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from spacr import image_store
from spacr.image_store import PlateWriter, read_image, read_labels, run_fields

CHANNELS = ["DAPI", "GFP", "AF750"]


def _make_field(task):
    """A field and its two label images, from a seed (top level: spawn-safe)."""
    rng = np.random.default_rng(int(task))
    image = rng.integers(0, 60000, (3, 96, 128)).astype(np.uint16)
    nuclei = np.zeros((96, 128), np.int32)
    nuclei[10:30, 10:30] = 1
    nuclei[50:80, 60:100] = 2 + int(task)
    return image, {"nucleus": nuclei, "cell": (nuclei > 0).astype(np.int32) * 7}


def _plan(n=6):
    wells = ["A01", "A02", "B03"]
    return [(wells[i % 3], i // 3, i) for i in range(n)]


def _arrays(root, plate="P1"):
    """Every array in the plate store, by path, and the ledger order."""
    store = root / f"{plate}.ome.zarr"
    out = {}
    for meta in sorted(store.rglob("zarr.json")):
        node = json.loads(meta.read_text())
        if node["node_type"] == "array":
            out[str(meta.parent.relative_to(store))] = np.asarray(
                zarr.open_array(str(meta.parent), mode="r")[...])
    ledger = [json.loads(l)["key"] for l in
              (store / "spacr_commits.jsonl").read_text().splitlines()]
    return out, ledger


def test_fields_and_labels_round_trip_pixel_exact_in_an_hcs_plate(tmp_path):
    writer = PlateWriter(tmp_path, "P1", channel_names=CHANNELS)
    report = run_fields(_plan(), _make_field, writer)
    assert report == {"written": 6, "skipped": 0, "keys": [
        "A/1/0", "A/2/0", "B/3/0", "A/1/1", "A/2/1", "B/3/1"]}
    for well, field, task in _plan():
        image, labels = _make_field(task)
        got = read_image(tmp_path / "P1.ome.zarr", key=(well, field))
        assert got.dtype == image.dtype and np.array_equal(got, image)
        for name, mask in labels.items():
            back = read_labels(tmp_path / "P1.ome.zarr", name, key=(well, field))
            assert back.dtype == mask.dtype and np.array_equal(back, mask)
    root = json.loads((tmp_path / "P1.ome.zarr" / "zarr.json").read_text())
    plate = root["attributes"]["ome"]["plate"]
    assert [w["path"] for w in plate["wells"]] == ["A/1", "A/2", "B/3"]
    assert plate["field_count"] == 2
    well = json.loads((tmp_path / "P1.ome.zarr" / "A" / "1" / "zarr.json").read_text())
    assert well["attributes"]["ome"]["well"]["images"] == [{"path": "0"}, {"path": "1"}]
    labels = json.loads((tmp_path / "P1.ome.zarr" / "A" / "1" / "0" / "labels"
                         / "zarr.json").read_text())
    assert labels["attributes"]["ome"]["labels"] == ["nucleus", "cell"]
    field = json.loads((tmp_path / "P1.ome.zarr" / "A" / "1" / "0" / "zarr.json").read_text())
    assert [c["label"] for c in field["attributes"]["ome"]["omero"]["channels"]] == CHANNELS


@pytest.mark.parametrize("method", ["spawn", "forkserver"])
def test_parallel_workers_commit_in_plan_order_and_reruns_are_byte_identical(
        tmp_path, method):
    first = PlateWriter(tmp_path / "a", "P1", channel_names=CHANNELS)
    run_fields(_plan(), _make_field, first, workers=2, start_method=method)
    second = PlateWriter(tmp_path / "b", "P1", channel_names=CHANNELS)
    run_fields(_plan(), _make_field, second, workers=1)
    arrays_a, order_a = _arrays(tmp_path / "a")
    arrays_b, order_b = _arrays(tmp_path / "b")
    assert order_a == order_b == [f"{w[0]}/{int(w[1:])}/{f}" for w, f, _t in _plan()]
    assert arrays_a.keys() == arrays_b.keys()
    for name in arrays_a:
        assert np.array_equal(arrays_a[name], arrays_b[name])


def test_a_stopped_run_resumes_with_only_uncommitted_fields(tmp_path):
    writer = PlateWriter(tmp_path, "P1", channel_names=CHANNELS)
    run_fields(_plan(3), _make_field, writer)
    stray = tmp_path / "P1.ome.zarr" / "A" / "1" / "1"
    stray.mkdir(parents=True)
    (stray / "junk").write_text("half-written field")
    again = PlateWriter(tmp_path, "P1", channel_names=CHANNELS)
    report = run_fields(_plan(), _make_field, again)
    assert report["skipped"] == 3 and report["written"] == 3
    assert not (stray / "junk").exists()
    ref = PlateWriter(tmp_path / "ref", "P1", channel_names=CHANNELS)
    run_fields(_plan(), _make_field, ref)
    resumed, order = _arrays(tmp_path)
    clean, clean_order = _arrays(tmp_path / "ref")
    assert order == clean_order
    for name in clean:
        assert np.array_equal(resumed[name], clean[name])


@pytest.mark.parametrize("suffix", [".npy", ".npz", ".tif", ".png", ".zarr"])
def test_the_reader_finds_every_format_from_what_is_on_disk(tmp_path, suffix):
    data = np.random.default_rng(5).integers(0, 255, (40, 50)).astype(np.uint8)
    path = tmp_path / f"img{suffix}"
    if suffix == ".npy":
        np.save(path, data)
    elif suffix == ".npz":
        np.savez(path, data)
    elif suffix == ".tif":
        import tifffile
        tifffile.imwrite(str(path), data)
    elif suffix == ".png":
        from PIL import Image
        Image.fromarray(data).save(path)
    else:
        from spacr.ome_zarr import write_ome_zarr
        write_ome_zarr(path, data, axes=["y", "x"], ngff_version="0.5")
    got = read_image(path)
    assert got.dtype == data.dtype and np.array_equal(got, data)
    assert np.array_equal(read_labels(path, "cell"), data)


def test_bad_wells_and_files_are_refused(tmp_path):
    with pytest.raises(ValueError, match="row letter"):
        image_store._field_key("3B", 0)
    (tmp_path / "x.bmp").write_bytes(b"")
    with pytest.raises(ValueError, match="not an image"):
        read_image(tmp_path / "x.bmp")


def _merged_run(tmp_path):
    """A Mask Generation run folder: merged stacks (Y, X, 3 channels + 2 masks) and the plane layout."""
    from spacr.crops import MERGED_LAYOUT_SIDECAR

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / MERGED_LAYOUT_SIDECAR).write_text(json.dumps({
        "version": 1, "intensity_channels": [0, 1, 2],
        "mask_plane_order": ["cell", "nucleus"],
        "mask_dims": {"cell": 3, "nucleus": 4}}))
    stacks = {}
    for i, name in enumerate(["plate1_A01_1", "plate1_A01_2", "plate1_B03_1", "plate2_C04_1"]):
        rng = np.random.default_rng(i)
        stack = np.zeros((64, 80, 5), np.uint16)
        stack[..., :3] = rng.integers(0, 60000, (64, 80, 3))
        stack[5:20, 5:30, 3] = 1
        stack[30:60, 40:70, 3] = 2
        stack[8:15, 8:20, 4] = 1
        np.save(merged / f"{name}.npy", stack)
        stacks[name] = stack
    return stacks


def test_mask_generation_exports_merged_fields_to_ome_zarr_pixel_exact(tmp_path):
    from spacr.image_store import export_merged

    stacks = _merged_run(tmp_path)
    report = export_merged(tmp_path, "ome-zarr", channel_names=CHANNELS)
    assert sorted(report) == ["plate1", "plate2"]
    assert report["plate1"]["written"] == 3
    for name, stack in stacks.items():
        plate, well, field = name.split("_")
        store = tmp_path / "images" / f"{plate}.ome.zarr"
        image = read_image(store, key=(well, int(field)))
        assert np.array_equal(image, np.moveaxis(stack[..., :3], -1, 0))
        assert np.array_equal(read_labels(store, "cell", key=(well, int(field))), stack[..., 3])
        assert np.array_equal(read_labels(store, "nucleus", key=(well, int(field))), stack[..., 4])
    again = export_merged(tmp_path, "ome-zarr", channel_names=CHANNELS)
    assert again["plate1"] == {"written": 0, "skipped": 3, "keys": []}


def test_mask_generation_exports_tiff_pixel_exact_and_npy_writes_nothing(tmp_path):
    import tifffile

    from spacr.image_store import export_merged

    stacks = _merged_run(tmp_path)
    assert export_merged(tmp_path, "npy") == {}
    assert not (tmp_path / "images").exists() and not (tmp_path / "merged_tiff").exists()
    files = export_merged(tmp_path, "tiff")["tiff"]
    assert len(files) == 4 * 3
    for name, stack in stacks.items():
        image = tifffile.imread(str(tmp_path / "merged_tiff" / f"{name}.tif"))
        assert np.array_equal(image, np.moveaxis(stack[..., :3], -1, 0))
        cell = tifffile.imread(str(tmp_path / "merged_tiff" / f"{name}_cell_mask.tif"))
        assert cell.dtype == stack.dtype and np.array_equal(cell, stack[..., 3])
    with pytest.raises(ValueError, match="mask_image_format must be one of"):
        export_merged(tmp_path, "png")
