"""Item 0aj P2: no masks/<class>_mask_stack. Every segmentation stage writes its labels into fov.ome.zarr.

Each object mask becomes ``labels/<class>`` of its field through the one
writer (:func:`spacr.image_store._write_mask`, i.e.
``PlateWriter.write_labels``) and is read back through the one reader
(:func:`spacr.image_store._read_mask`). A field+class counts as done only
once its label is committed, so a run killed mid-role resumes field by
field. An older project's ``masks/*_mask_stack/*.npy`` still reads.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zarr")
tifffile = pytest.importorskip("tifffile")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

from tests.test_a_rerun_trusts_nothing_a_killed_run_left import (  # noqa: E402,F401
    _RUN, _settings, fake_cellpose,
)


def _run(plate, **over):
    from spacr.core import preprocess_generate_masks

    run = dict(_RUN, keep_intermediate=False, preprocess=True)
    run.update(over)
    preprocess_generate_masks(_settings(plate, **run))


def _merged(project, store):
    """``{field name: merged array}`` through the shared merged-field reader."""
    from spacr.image_store import _list_fields, _read_field

    folder = Path(project) / ("fov.ome.zarr" if store else "merged")
    return {name: _read_field(folder / name) for name in _list_fields(folder)}


def _mask_stack_paths(root):
    return sorted(str(p) for p in Path(root).rglob("*_mask_stack*"))


@pytest.fixture
def mask_writes(monkeypatch):
    """Every mask write, as ``(label, field name)``; and every .npy saved under a *_mask_stack."""
    import spacr.image_store as IS
    import spacr.io as IO

    calls = {"labels": [], "npy": []}
    real_write, real_save = IS._write_mask, IO._save_array_atomic

    def _write_mask(path, mask):
        calls["labels"].append((IS._mask_label_name(Path(path).parent), Path(path).name))
        return real_write(path, mask)

    def save(path, array):
        if "_mask_stack" in str(path):
            calls["npy"].append(str(path))
        return real_save(path, array)

    for module in ("spacr.object", "spacr.utils", "spacr.host_pathogen"):
        mod = __import__(module, fromlist=["x"])
        if hasattr(mod, "_write_mask"):
            monkeypatch.setattr(mod, "_write_mask", _write_mask)
    monkeypatch.setattr(IS, "_write_mask", _write_mask)
    monkeypatch.setattr(IO, "_save_array_atomic", save)
    return calls


@pytest.mark.parametrize("keep", [False, True])
def test_a_mask_run_writes_no_mask_stack_and_matches_the_npy_run(
        yokogawa_cellvoyager_dir, fake_cellpose, mask_writes, tmp_path, keep):
    from spacr.image_store import _store_entries, read_labels, verify_project

    plate = yokogawa_cellvoyager_dir["src"]
    npy_plate = tmp_path / "npy" / plate.name
    shutil.copytree(plate, npy_plate)
    _run(npy_plate, mask_image_format="npy", keep_intermediate=True)
    assert _mask_stack_paths(npy_plate / "masks")  # the npy run still writes them
    mask_writes["npy"].clear()
    mask_writes["labels"].clear()

    _run(plate, keep_intermediate=keep)
    # No mask-stack folder or file was ever written; every mask went to the store.
    assert mask_writes["npy"] == [] and _mask_stack_paths(plate) == []
    assert sorted(mask_writes["labels"]) == sorted(
        (role, name) for role in ("cell", "nucleus") for name in _merged(plate, True))
    before, after = _merged(npy_plate, False), _merged(plate, True)
    assert sorted(before) == sorted(after) and len(after) == 4
    for name in before:
        assert after[name].dtype == before[name].dtype
        assert np.array_equal(after[name], before[name])
    store = plate / "fov.ome.zarr"
    for name, entry in _store_entries(store).items():
        assert sorted(entry["labels"]) == ["cell", "nucleus"]
        for role in ("cell", "nucleus"):
            label = read_labels(store / entry["key"], role)
            assert label.dtype == np.uint16
            old = np.load(npy_plate / "masks" / f"{role}_mask_stack" / name)
            assert np.array_equal(label, old)
    report = verify_project(plate)
    assert report["mismatches"] == []


def test_a_run_killed_mid_role_resumes_from_the_committed_labels(
        yokogawa_cellvoyager_dir, fake_cellpose, mask_writes, monkeypatch, tmp_path):
    import spacr.image_store as IS
    import spacr.object as O

    plate = yokogawa_cellvoyager_dir["src"]
    clean = tmp_path / "clean" / plate.name
    shutil.copytree(plate, clean)
    _run(clean)
    mask_writes["labels"].clear()

    real = O._write_mask
    nuclei = []

    def die_on_the_third_nucleus(path, mask):
        if IS._mask_label_name(Path(path).parent) == "nucleus":
            if len(nuclei) == 2:
                raise KeyboardInterrupt("killed while segmenting nuclei")
            nuclei.append(Path(path).name)
        return real(path, mask)

    with monkeypatch.context() as patch:
        patch.setattr(O, "_write_mask", die_on_the_third_nucleus)
        with pytest.raises(KeyboardInterrupt):
            _run(plate, plot=False)
    masks = plate / "masks"
    assert _mask_stack_paths(plate) == []
    fields = sorted(IS._image_entries(plate / "fov.ome.zarr"))
    first = set(mask_writes["labels"])
    assert len([1 for role, _n in first if role == "nucleus"]) == 2
    # What is committed is exactly what was written before the kill.
    for role in ("cell", "nucleus"):
        assert IS._mask_names(masks / f"{role}_mask_stack") == sorted(
            name for r, name in first if r == role)
    assert IS._store_entries(plate / "fov.ome.zarr") == {}  # nothing merged yet
    mask_writes["labels"].clear()

    _run(plate, plot=False, resume=True)
    # The resume segments only the field+role pairs that were not committed.
    assert sorted(mask_writes["labels"]) == sorted(
        {(role, name) for role in ("cell", "nucleus") for name in fields} - first)
    before, after = _merged(clean, True), _merged(plate, True)
    assert sorted(before) == sorted(after) == fields
    for name in fields:
        assert np.array_equal(after[name], before[name])
    assert _mask_stack_paths(plate) == []


def test_a_finished_run_segments_again_on_a_rerun(
        yokogawa_cellvoyager_dir, fake_cellpose, mask_writes):
    """The merged labels are the output, not working masks: a rerun of a
    finished project segments every field again, as when masks/ was deleted."""
    from spacr.image_store import _store_entries

    plate = yokogawa_cellvoyager_dir["src"]
    _run(plate)
    first = sorted(mask_writes["labels"])
    store = plate / "fov.ome.zarr"
    merged = {n: e for n, e in _store_entries(store).items()}
    assert all(e.get("masks") == "consumed" for e in merged.values())
    mask_writes["labels"].clear()
    _run(plate)
    assert sorted(mask_writes["labels"]) == first
    assert sorted(_store_entries(store)) == sorted(merged)
    assert _mask_stack_paths(plate) == []


def test_an_older_project_with_mask_stack_npy_still_reads(
        yokogawa_cellvoyager_dir, fake_cellpose, tmp_path):
    import spacr.image_store as IS
    from spacr.seg_qc import find_mask_stacks, score_masks

    plate = yokogawa_cellvoyager_dir["src"]
    _run(plate, mask_image_format="npy", keep_intermediate=True)
    folder = plate / "masks" / "cell_mask_stack"
    names = IS._mask_names(folder)
    assert len(names) == 4 and all((folder / n).is_file() for n in names)
    assert IS._mask_folder_exists(folder) and not IS._masks_in_store(folder)
    array = IS._read_mask(folder / names[0])
    assert np.array_equal(array, np.load(folder / names[0]))
    assert IS._mask_header(folder / names[0]) == (array.shape, array.dtype)
    assert IS._mask_valid(folder / names[0]) == (True, "done")
    assert set(find_mask_stacks(str(plate))) == {"cell", "nucleus"}
    assert len(score_masks(str(folder), object_type="cell")) == 4

    # A project whose channels are in fov.ome.zarr but whose masks are files
    # (an earlier 0aj version): the files win, and the merge reads them.
    store_plate = tmp_path / "store" / plate.name
    shutil.copytree(plate, store_plate)
    shutil.rmtree(store_plate / "merged")
    from spacr.io import _load_and_concatenate_arrays
    from spacr.image_store import _stack_writer, _write_stack_field
    writer = _stack_writer(store_plate, "plate1")
    for name in IS._stack_folder_names(store_plate / "stack"):
        _write_stack_field(writer, name, np.load(store_plate / "stack" / name), [1, 2])
    shutil.rmtree(store_plate / "stack")
    assert not IS._masks_in_store(store_plate / "masks" / "cell_mask_stack")
    _load_and_concatenate_arrays(str(store_plate), [0, 1], 1, 0, None, None,
                                 image_format="ome-zarr")
    old = _merged(plate, False)
    new = _merged(store_plate, True)
    assert sorted(old) == sorted(new)
    for name in old:
        assert np.array_equal(old[name], new[name])


def test_the_mask_reader_and_writer_round_trip_and_keep_the_channels(tmp_path):
    import spacr.image_store as IS

    project = tmp_path / "proj"
    (project / "masks").mkdir(parents=True)
    writer = IS._stack_writer(project, "plate1")
    stack = np.random.default_rng(0).integers(0, 4000, (40, 50, 2)).astype(np.uint16)
    IS._write_stack_field(writer, "plate1_B03_1_1.npy", stack, [1, 2])
    folder = project / "masks" / "cell_mask_stack"
    path = folder / "plate1_B03_1_1.npy"
    assert IS._masks_in_store(folder) and not IS._mask_exists(path)
    assert IS._mask_valid(path) == (False, "missing")
    big = np.arange(40 * 50, dtype=np.int64).reshape(40, 50) * 40  # above 65535
    IS._write_mask(path, big)
    IS._write_mask(project / "masks" / "nucleus_mask_stack" / path.name,
                  (big % 7).astype(np.int32))
    assert not folder.exists() and not (project / "masks" / "nucleus_mask_stack").exists()
    read = IS._read_mask(path)
    assert read.dtype == np.uint32 and np.array_equal(read, big)
    assert IS._read_mask(project / "masks" / "nucleus_mask_stack" / path.name).dtype == np.uint16
    assert IS._mask_names(folder) == [path.name]
    assert set(IS._mask_folders(project / "masks")) == {"cell_mask_stack", "nucleus_mask_stack"}
    assert IS._mask_digest(path) == IS._mask_digest(path)
    # Labels never touch the channels.
    assert np.array_equal(IS._read_stack(project / "stack" / path.name), stack)
    assert IS.verify_store(project)["mismatches"] == []
    assert IS._remove_mask_folder(project / "masks" / "nucleus_mask_stack")
    assert set(IS._mask_folders(project / "masks")) == {"cell_mask_stack"}


def test_a_time_lapse_frame_keeps_its_own_mask_file_not_a_label_of_its_field(tmp_path):
    """Each frame of a t-axis field (0aj P3) keeps its own ``.npy`` mask; none overwrites another.

    A 2-D frame mask cannot be a label of the field's t axis, so the Mask
    writer keeps it as the working file the merge turns into ``(T, Y, X)``
    labels, and the end-of-run steps leave the field's ledger entry alone.
    """
    import spacr.image_store as IS

    project = tmp_path / "proj"
    names = ["plate1_E01_1_1.npy", "plate1_E01_1_2.npy"]
    writer = IS._stack_writer(project, "plate1")
    IS._write_stack_series(writer, names, [np.full((2, 8, 8), t + 1, np.uint16) for t in range(2)],
                           [1, 2], z_planes=False)
    folder = project / "masks" / "cell_mask_stack"
    assert not IS._masks_in_store(folder)
    assert not IS._project_keeps_masks_in_store(project)
    masks = {}
    for t, name in enumerate(names):
        masks[name] = np.zeros((8, 8), np.uint16)
        masks[name][t, t] = 10 + t
        IS._write_mask(folder / name, masks[name])
    for name in names:
        assert (folder / name).is_file()
        assert np.array_equal(IS._read_mask(folder / name), masks[name])
    assert not (project / "fov.ome.zarr" / IS._image_entries(project / "fov.ome.zarr")[names[0]]["key"]
                / "labels" / "cell").exists()
    before = (project / "fov.ome.zarr" / IS._LEDGER).read_text(encoding="utf-8")
    assert IS._consume_working_masks(project) == 0
    assert (project / "fov.ome.zarr" / IS._LEDGER).read_text(encoding="utf-8") == before
