"""Item 987b: the V1 normalised batches are built in memory, and equal what the archives held."""
import os

import numpy as np
import pytest

from spacr import io as IO
from spacr import _mask_batches as MB
from spacr.io import _normalized_npz_field_ids, concatenate_and_normalize

SETTINGS = {"timelapse": False, "randomize": False, "batch_size": 2,
            "lower_percentile": 2, "nucleus_channel": 0, "cell_channel": 1,
            "pathogen_channel": None, "organelle_channel": None,
            "nucleus_background": 100, "nucleus_signal_to_noise": 5,
            "remove_background_nucleus": False, "cell_background": 100,
            "cell_signal_to_noise": 5, "remove_background_cell": False,
            "background": 100, "Signal_to_noise": 5, "remove_background": False,
            "plot": False, "figuresize": 4, "cmap": "inferno", "nr": 1}


@pytest.fixture
def archives_on_disk(monkeypatch):
    """Make ``concatenate_and_normalize`` write archives, the way a plotting or timelapse run does."""
    monkeypatch.setattr(IO, "_mask_batches_stay_in_memory", lambda *a, **k: False)


def _plate(root, shapes):
    stack = root / "stack"
    stack.mkdir(parents=True)
    rng = np.random.default_rng(3)
    for index, (h, w) in enumerate(shapes):
        field = (rng.random((h, w, 3)) * 2900.0 + 60.0).astype(np.float32)
        np.save(stack / f"plate1_A01_F{index:03d}.npy", field)
    return stack


@pytest.mark.parametrize("shapes", [[(8, 8)] * 5, [(8, 8), (8, 10), (9, 8), (8, 8), (7, 7)]])
def test_a_plan_builds_the_batches_the_archives_held(tmp_path, shapes, monkeypatch):
    plan_stack = _plate(tmp_path / "plan", shapes)
    disk_stack = _plate(tmp_path / "disk", shapes)
    concatenate_and_normalize(str(plan_stack), [0, 1], settings=dict(SETTINGS))
    monkeypatch.setattr(IO, "_mask_batches_stay_in_memory", lambda *a, **k: False)
    concatenate_and_normalize(str(disk_stack), [0, 1], settings=dict(SETTINGS))
    plan_masks, disk_masks = tmp_path / "plan" / "masks", tmp_path / "disk" / "masks"

    assert not list(plan_masks.glob("*.npz")) and (plan_masks / MB._PLAN_NAME).is_file()
    archives = sorted(disk_masks.glob("*.npz"))
    assert archives and not (disk_masks / MB._PLAN_NAME).exists()
    paths = MB._batch_paths(plan_masks, os.listdir)
    assert [os.path.basename(p) for p in paths] == [a.name for a in archives]
    for path, archive in zip(paths, archives):
        with MB._open_batch(path) as built, np.load(archive) as saved:
            np.testing.assert_array_equal(built["data"], saved["data"])
            assert built["data"].dtype == saved["data"].dtype
            assert list(built["filenames"]) == list(saved["filenames"])


def test_the_plan_answers_what_resume_and_quality_screening_ask(tmp_path):
    stack = _plate(tmp_path, [(8, 8)] * 3)
    concatenate_and_normalize(str(stack), [0, 1], settings=dict(SETTINGS))
    masks = str(tmp_path / "masks")
    assert _normalized_npz_field_ids(masks) == tuple(
        f"plate1_A01_F{i:03d}" for i in range(3))
    from spacr.io import _check_normalized_archives, _validate_normalized_source_channels

    checked = _check_normalized_archives(masks)
    assert checked["covered"] == set(_normalized_npz_field_ids(masks)) and checked["planes"] == {2}
    virtual = MB._batch_paths(masks, os.listdir)[0]
    assert _validate_normalized_source_channels(virtual, [0, 1]) == [0, 1]
    with pytest.raises(ValueError):
        _validate_normalized_source_channels(virtual, [0, 2])


def test_old_projects_keep_their_archives_and_a_rerun_over_them_writes_archives(
        tmp_path, archives_on_disk, monkeypatch):
    stack = _plate(tmp_path, [(8, 8)] * 3)
    concatenate_and_normalize(str(stack), [0, 1], settings=dict(SETTINGS))
    monkeypatch.undo()
    masks = tmp_path / "masks"
    assert sorted(masks.glob("*.npz"))
    concatenate_and_normalize(str(stack), [0, 1], settings=dict(SETTINGS))
    assert sorted(masks.glob("*.npz")) and not (masks / MB._PLAN_NAME).exists()
    paths = MB._batch_paths(masks, os.listdir)
    assert all(os.path.isfile(p) for p in paths)
    with MB._open_batch(paths[0]) as batch:
        assert batch["data"].shape[-1] == 2


def test_volumetric_fields_keep_writing_archives(tmp_path):
    stack = tmp_path / "stack"
    stack.mkdir()
    np.save(stack / "plate1_A01_F000.npy", np.full((2, 6, 6, 3), 500.0, np.float32))
    concatenate_and_normalize(str(stack), [0, 1], settings=dict(SETTINGS, batch_size=1))
    assert list((tmp_path / "masks").glob("*.npz"))


def test_parallel_workers_get_the_plan_not_archive_paths(tmp_path):
    from spacr.io import _mask_batch_manifest
    from spacr.object import _assigned_mask_archives
    from spacr._mask_workers import _partition_batches

    stack = _plate(tmp_path, [(8, 8)] * 5)
    concatenate_and_normalize(str(stack), [0, 1], settings=dict(SETTINGS))
    masks = str(tmp_path / "masks")
    records = _mask_batch_manifest(masks)
    assert [os.path.basename(r["path"]) for r in records] == [
        "stack_0_norm.npz", "stack_1_norm.npz", "stack_2_norm.npz"]
    assert [len(r["fields"]) for r in records] == [2, 2, 1]
    assert all(set(r) >= {"path", "sha256", "bytes", "fields", "planes"} and r["planes"] == 2
               for r in records)
    assert len({r["sha256"] for r in records}) == 3 and all(r["bytes"] > 0 for r in records)
    assert _mask_batch_manifest(masks) == records
    parts = _partition_batches(records, ["0", "1"], None)
    assert sorted(path for paths in parts.values() for path in paths) == sorted(r["path"] for r in records)
    assert _assigned_mask_archives(masks, parts["0"]) == [os.path.abspath(p) for p in parts["0"]]
    with pytest.raises(ValueError):
        _assigned_mask_archives(masks, [os.path.join(masks, "stack_9_norm.npz")])
    changed = np.load(stack / "plate1_A01_F000.npy") * 2
    np.save(stack / "plate1_A01_F000.npy", changed)
    assert _mask_batch_manifest(masks)[0]["sha256"] != records[0]["sha256"]
