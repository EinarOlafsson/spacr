"""Item 0aj P3: t and z axes in fov.ome.zarr, raw z planes stored, z reduced on read.

The fixtures are tiny CellVoyager-named TIFFs written here: a z-stack
plate (each channel three slices) and a time-lapse plate (three frames).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

pytest.importorskip("zarr")


def _write_plate(folder, *, slices=1, frames=1, fields=(1,), wells=("E01",),
                 channels=(1, 2), seed=0, shape=(40, 48)):
    """Raw CellVoyager images; returns ``{(well, field, time, channel): [planes by z]}``."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    planes = {}
    for well in wells:
        for field in fields:
            for t in range(1, frames + 1):
                for c in channels:
                    stack = []
                    for z in range(1, slices + 1):
                        plane = rng.integers(0, 60000, shape, dtype=np.uint16)
                        tifffile.imwrite(folder / (f"plate1_{well}_T{t:04d}F{field:03d}"
                                                   f"L01A01Z{z:02d}C{c:02d}.tif"), plane)
                        stack.append(plane)
                    planes[(well, field, t, c)] = stack
    return planes


def _ingest(folder, *, timelapse=False):
    """Mask preprocessing's ingest into fov.ome.zarr, as a Mask run calls it."""
    from spacr.io import _rename_and_organize_image_files
    from spacr.utils import _get_regex

    regex = _get_regex("cellvoyager", "tif", None)
    return _rename_and_organize_image_files(
        str(folder), regex, metadata_type="cellvoyager", img_format=[".tif"],
        timelapse=timelapse, defer_originals=True, store_fields=True,
        source_plate="plate1")


def _ledger(folder):
    return [json.loads(line) for line in
            (folder / "fov.ome.zarr" / "spacr_commits.jsonl").read_text().splitlines()]


def test_raw_z_planes_are_stored_and_the_maximum_projection_is_computed_on_read(tmp_path):
    from spacr.image_store import (_read_stack, _stack_header, _stack_names,
                                   read_frame_channels, verify_store)
    from spacr.io import _fold_max

    project = tmp_path / "plate1"
    planes = _write_plate(project, slices=3, fields=(1, 2))
    _ingest(project)
    assert not (project / "stack").exists()
    names = _stack_names(project / "stack")
    assert names == ["plate1_E01_1_1.npy", "plate1_E01_2_1.npy"]
    entries = {e["name"]: e for e in _ledger(project)}
    for field, name in ((1, names[0]), (2, names[1])):
        entry = entries[name]
        assert entry["axes"] == "czyx"
        raw = read_frame_channels(project / "fov.ome.zarr" / entry["key"], entry,
                                  z_projection=None)
        assert raw.shape == (2, 3, 40, 48) and raw.dtype == np.uint16
        for ci, c in enumerate((1, 2)):
            for z in range(3):
                # The stored plane IS the original file, byte for byte.
                assert raw[ci, z].tobytes() == planes[("E01", field, 1, c)][z].tobytes()
        # The 2-D view every Mask stage reads is the projection Mask stored before.
        before = np.stack([_fold_max([p.copy() for p in planes[("E01", field, 1, c)]])
                           for c in (1, 2)], axis=-1)
        view = _read_stack(project / "stack" / name)
        assert view.dtype == before.dtype and view.tobytes() == before.tobytes()
        assert _stack_header(project / "stack" / name) == ((40, 48, 2), np.dtype(np.uint16))
        mean = _read_stack(project / "stack" / name, z_projection="mean")
        assert mean.dtype == np.float32
        assert np.allclose(mean[..., 0], np.mean(planes[("E01", field, 1, 1)], axis=0))
        assert _read_stack(project / "stack" / name, z_projection=1)[..., 1].tobytes() == \
            planes[("E01", field, 1, 2)][1].tobytes()
    report = verify_store(project)
    assert report["mismatches"] == [] and report["fields"] == 2
    assert report["planes"] == 2 * 2 * 3  # every z plane of every channel


def test_the_originals_of_a_z_stack_are_settled_only_after_every_plane_is_verified(tmp_path):
    from spacr.image_store import _settle_originals, verify_project

    project = tmp_path / "plate1"
    _write_plate(project, slices=3)
    _ingest(project)
    settings = {"metadata_type": "cellvoyager", "custom_regex": None}
    (project / "settings").mkdir()
    (project / "settings" / "gen_mask_settings.json").write_text(json.dumps(settings))
    checked = verify_project(project, originals=True)
    assert checked["mismatches"] == [] and checked["originals_compared"] == 6
    report = _settle_originals(project, keep=True, settings=settings, source_plate="plate1",
                               printer=lambda *_a: None)
    assert report["problems"] == [] and report["kept"] == []
    assert len(report["moved"]) == 6  # 2 channels x 3 slices
    assert len(list((project / "orig").glob("*.tif"))) == 6 and not list(project.glob("*.tif"))


def test_a_corrupted_z_plane_keeps_every_original_in_place(tmp_path):
    from spacr.image_store import _settle_originals

    project = tmp_path / "plate1"
    _write_plate(project, slices=2)
    _ingest(project)
    # One original changes after the store was written: nothing is deleted.
    victim = sorted(project.glob("*Z02C01.tif"))[0]
    tifffile.imwrite(victim, np.zeros((40, 48), np.uint16))
    report = _settle_originals(project, keep=False,
                               settings={"metadata_type": "cellvoyager"},
                               source_plate="plate1", printer=lambda *_a: None)
    assert report["deleted"] == [] and any("z 1" in p for p in report["problems"])
    assert len(list(project.glob("*.tif"))) == 4


def test_a_time_lapse_field_is_one_field_with_a_t_axis_read_frame_by_frame(tmp_path):
    from spacr.image_store import (_read_stack, _stack_names, _write_merged_field,
                                   _merged_writer, _read_field, _store_entries,
                                   _list_fields, verify_store)

    project = tmp_path / "plate1"
    planes = _write_plate(project, frames=3, fields=(1,))
    _ingest(project, timelapse=True)
    ledger = _ledger(project)
    assert len(ledger) == 1 and ledger[0]["axes"] == "tcyx"
    assert ledger[0]["frames"] == [f"plate1_E01_1_{t}.npy" for t in (1, 2, 3)]
    names = _stack_names(project / "stack")
    assert sorted(names) == sorted(ledger[0]["frames"])
    for t in (1, 2, 3):
        frame = _read_stack(project / "stack" / f"plate1_E01_1_{t}.npy")
        for ci, c in enumerate((1, 2)):
            assert frame[..., ci].tobytes() == planes[("E01", 1, t, c)][0].tobytes()
    assert verify_store(project)["planes"] == 3 * 2

    layout = {"version": 1, "intensity_channels": [0, 1], "mask_plane_order": ["cell"],
              "mask_dims": {"cell": 2}}
    writer = _merged_writer(project, "plate1", layout)
    store = project / "fov.ome.zarr"
    merged = {}
    for t in (2, 1, 3):  # any order: the field commits with its last frame
        name = f"plate1_E01_1_{t}.npy"
        stack = _read_stack(project / "stack" / name)
        cells = np.zeros(stack.shape[:2], np.uint16)
        cells[5 + t:15 + t, 4:20] = 7
        merged[name] = np.concatenate([stack, cells[..., None]], axis=-1)
        _write_merged_field(writer, name, merged[name], layout, image_dtype=np.uint16)
        if t != 3:
            assert _store_entries(store) == {}  # not merged until every frame is in
    assert sorted(_list_fields(store)) == sorted(merged)
    for name, want in merged.items():
        got = _read_field(store / name)
        assert got.dtype == want.dtype and np.array_equal(got, want)
    entry = _ledger(project)[-1]
    assert entry["label_axes"] == "tyx" and entry["merged_frames"] == [0, 1, 2]
    from spacr.image_store import read_labels
    assert read_labels(store / entry["key"], "cell").shape == (3, 40, 48)
    assert verify_store(project)["mismatches"] == []


def test_a_two_d_project_records_no_axes_and_keeps_its_layout(tmp_path):
    from spacr.image_store import _read_stack

    project = tmp_path / "plate1"
    planes = _write_plate(project)
    _ingest(project)
    entry = _ledger(project)[0]
    assert "axes" not in entry and "label_axes" not in entry and "frames" not in entry
    meta = json.loads((project / "fov.ome.zarr" / entry["key"] / "zarr.json").read_text())
    axes = meta["attributes"]["ome"]["multiscales"][0]["coordinateSystems"][0]["axes"]
    assert [a["name"] for a in axes] == ["c", "y", "x"]
    view = _read_stack(project / "stack" / entry["name"])
    assert view[..., 0].tobytes() == planes[("E01", 1, 1, 1)][0].tobytes()


def test_axes_are_checked_and_never_flattened():
    from spacr.image_store import field_axes, project_z

    assert field_axes(3) == "cyx" and field_axes(2, kind="labels") == "yx"
    assert field_axes(5, "tczyx") == "tczyx" and field_axes(3, "tyx", kind="labels") == "tyx"
    with pytest.raises(ValueError, match="never flattened"):
        field_axes(4)
    with pytest.raises(ValueError, match="NGFF order"):
        field_axes(4, "ctyx")
    with pytest.raises(ValueError, match="NGFF order"):
        field_axes(3, "cyx", kind="labels")
    a = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    assert project_z(a, 0).dtype == np.uint16 and np.array_equal(project_z(a, 0), a[1])
    with pytest.raises(ValueError, match="z_projection"):
        project_z(a, 0, "median")


def test_streamed_crops_of_a_z_stack_are_cut_from_the_projection(tmp_path):
    """The crop streamer reads a box of the field frame through the shared reader."""
    from spacr.crop_store import _StoreFieldWindows
    from spacr.image_store import (_merged_writer, _read_field, _read_stack,
                                   _store_entries, _write_merged_field)

    project = tmp_path / "plate1"
    _write_plate(project, slices=3)
    _ingest(project)
    layout = {"version": 1, "intensity_channels": [0, 1], "mask_plane_order": ["cell"],
              "mask_dims": {"cell": 2}}
    writer = _merged_writer(project, "plate1", layout)
    name = "plate1_E01_1_1.npy"
    stack = _read_stack(project / "stack" / name)
    cells = np.zeros(stack.shape[:2], np.uint16)
    cells[4:30, 6:40] = 3
    _write_merged_field(writer, name, np.concatenate([stack, cells[..., None]], -1),
                        layout, image_dtype=np.uint16)
    store = project / "fov.ome.zarr"
    whole = _read_field(store / name)
    windows = _StoreFieldWindows(str(store / name), store, _store_entries(store)[name])
    box = windows.channels([0, 1], 3, 20, 5, 30)
    assert np.array_equal(box, np.moveaxis(whole[3:20, 5:30, :2], -1, 0))
    assert np.array_equal(windows.labels("cell", 3, 20, 5, 30), whole[3:20, 5:30, 2])
    windows.prefetch(0, 40, 0, 48)
    assert np.array_equal(windows.channels([1], 3, 20, 5, 30), box[1:])
