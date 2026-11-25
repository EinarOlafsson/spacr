"""Item 0aj P4: no masks/ working folder. The normalised GPU batches are built in memory.

A project whose fields live in fov.ome.zarr never gets a ``masks/`` folder:
the normalised batches (formerly the ``masks/stack_<n>_norm.npz`` cache, still
written by timelapse, volumetric, illumination and PSF runs) are built in
memory from fov.ome.zarr whenever a role needs them, plotting runs included,
and the small run records (the batch plan with its percentiles, the
adjusted-cell ledger) go to ``fov.ome.zarr/.spacr_mask_run``, removed at the
end of a run that keeps no intermediates. Normalisation is deterministic, so
building a batch again (another role, or a resume after a kill) gives the
same bytes as the archive did. An older project's ``masks/*.npz`` still reads.
"""
from __future__ import annotations

import os
import shutil
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zarr")
pytest.importorskip("tifffile")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call  # noqa: E402
from tests.test_a_rerun_trusts_nothing_a_killed_run_left import (  # noqa: E402
    _RUN, _settings,
)

RECORDS = ".spacr_mask_run"


@pytest.fixture
def value_cellpose(monkeypatch):
    """CPU stand-in for Cellpose whose labels follow the normalised pixel values.

    Objects are the connected pixels above 0.5 of the normalised input, so a
    batch normalised with other percentiles gives other labels: an exact
    match below proves the batch was built with the same values.
    """
    import torch
    from skimage.measure import label

    import spacr.object as O
    import spacr.plot as PL

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(PL, "plot_cellpose4_output", lambda *a, **k: None)
    seen = []

    class _Model:
        def __init__(self, *args, **kwargs):
            pass

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                 normalize=True, rescale=None, diameter=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                 anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=None,
                 compute_masks=True, progress=None):
            check_cellpose_eval_call(x, channel_axis, z_axis=z_axis, do_3D=do_3D,
                                     stitch_threshold=stitch_threshold)
            masks = []
            for image in x:
                image = np.asarray(image, dtype=np.float64)
                seen.append(image.copy())
                if image.ndim == 3:
                    axis = int(np.argmin(image.shape))
                    image = image.max(axis=axis)
                masks.append(label(image > 0.5).astype(np.uint16))
            return masks, [np.zeros_like(m, np.float32) for m in masks], None

    monkeypatch.setattr(O, "cp_models", types.SimpleNamespace(CellposeModel=_Model))
    return seen


@pytest.fixture
def npz_writes(monkeypatch):
    """Every normalised archive written, and every folder created under a masks/ folder."""
    import spacr.io as IO

    calls = {"npz": [], "dirs": []}
    real_save, real_makedirs = IO._save_npz_atomic, os.makedirs

    def save(path, **arrays):
        calls["npz"].append(str(path))
        return real_save(path, **arrays)

    def makedirs(name, *args, **kwargs):
        parts = Path(os.fspath(name)).parts
        if "masks" in parts:
            calls["dirs"].append(os.fspath(name))
        return real_makedirs(name, *args, **kwargs)

    monkeypatch.setattr(IO, "_save_npz_atomic", save)
    monkeypatch.setattr(os, "makedirs", makedirs)
    return calls


def _run(plate, **over):
    from spacr.core import preprocess_generate_masks

    run = dict(_RUN, preprocess=True, batch_size=2)
    run.update(over)
    preprocess_generate_masks(_settings(plate, **run))


def _labels(project):
    """``{(field, class): label}`` of every committed label in fov.ome.zarr."""
    from spacr.image_store import _store_entries, read_labels

    store = Path(project) / "fov.ome.zarr"
    out = {}
    for name, entry in _store_entries(store).items():
        for role in entry["labels"]:
            out[(name, role)] = read_labels(store / entry["key"], role)
    return out


def _merged(project):
    from spacr.image_store import _list_fields, _read_field

    store = Path(project) / "fov.ome.zarr"
    return {name: _read_field(store / name) for name in _list_fields(store)}


def _same(a, b):
    assert sorted(a) == sorted(b) and a
    for key in a:
        assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape, key
        assert np.array_equal(a[key], b[key]), key


@pytest.mark.parametrize("plot", [False, True])
@pytest.mark.parametrize("keep", [True, False])
def test_a_default_run_creates_no_masks_folder_and_writes_no_archive(
        yokogawa_cellvoyager_dir, value_cellpose, npz_writes, plot, keep):
    from spacr.image_store import verify_project

    plate = yokogawa_cellvoyager_dir["src"]
    _run(plate, plot=plot, keep_intermediate=keep)
    assert npz_writes["npz"] == []
    assert npz_writes["dirs"] == []
    assert not (plate / "masks").exists()
    records = plate / "fov.ome.zarr" / RECORDS
    # The run's records are kept with the intermediates, otherwise removed.
    assert (records / "batch_plan.json").is_file() is keep
    if keep:
        import json

        plan = json.loads((records / "batch_plan.json").read_text())
        # A plotting run keeps the batches its archives had (batch_size 2,
        # nothing resized or calibrated), so its masks are those of earlier versions.
        assert bool(plan.get("archive_equivalent")) is plot
        if plot:
            assert plan["sizes"] == [2, 2] and not plan.get("calibration")
    assert len(_merged(plate)) == 4
    assert {role for _n, role in _labels(plate)} == {"cell", "nucleus"}
    assert verify_project(plate)["mismatches"] == []


def test_the_in_memory_batches_equal_the_archive_cache_byte_for_byte(
        yokogawa_cellvoyager_dir, value_cellpose, monkeypatch, tmp_path):
    """The same plate through the old archive cache and through the in-memory
    batches: identical normalised batches, labels and merged fields."""
    import spacr.io as IO
    from spacr import _mask_batches as MB

    plate = yokogawa_cellvoyager_dir["src"]
    cached = tmp_path / "cached" / plate.name
    shutil.copytree(plate, cached)
    with monkeypatch.context() as patch:
        patch.setattr(IO, "_mask_batches_stay_in_memory", lambda *a, **k: False)
        _run(cached, keep_intermediate=True)
    archives = sorted((cached / "masks").glob("stack_*_norm.npz"))
    assert len(archives) == 2

    _run(plate, keep_intermediate=True)
    assert not (plate / "masks").exists()
    _same(_labels(cached), _labels(plate))
    _same(_merged(cached), _merged(plate))

    # Normalisation is deterministic: each batch built again from fov.ome.zarr
    # has exactly the bytes its archive held, every time it is built.
    plan = MB._read_plan(plate / "masks")
    assert plan is not None and len(plan["batches"]) == 2
    for archive in archives:
        index = MB._batch_index_of(archive)
        with np.load(archive) as old:
            data, names = old["data"], list(old["filenames"])
        for _ in range(2):
            with MB._open_batch(plate / "masks" / archive.name) as built:
                assert list(built["filenames"]) == names
                assert built["data"].dtype == data.dtype
                assert built["data"].tobytes() == data.tobytes()
        assert index is not None


def test_a_run_killed_mid_role_resumes_from_the_store_without_a_masks_folder(
        yokogawa_cellvoyager_dir, value_cellpose, monkeypatch, tmp_path):
    """Killed after part of a batch was committed: no masks/ folder, the plan
    survives in fov.ome.zarr, and the resume rebuilds the batches and
    segments only the uncommitted field+role pairs, exactly as a clean run."""
    import spacr.image_store as IS
    import spacr.object as O

    plate = yokogawa_cellvoyager_dir["src"]
    clean = tmp_path / "clean" / plate.name
    shutil.copytree(plate, clean)
    _run(clean)

    real = O._write_mask
    written = []

    def die_on_the_third_nucleus(path, mask):
        role = IS._mask_label_name(Path(path).parent)
        if role == "nucleus" and len([r for r, _ in written if r == "nucleus"]) == 1:
            raise KeyboardInterrupt("killed while segmenting nuclei")
        written.append((role, Path(path).name))
        return real(path, mask)

    with monkeypatch.context() as patch:
        patch.setattr(O, "_write_mask", die_on_the_third_nucleus)
        with pytest.raises(KeyboardInterrupt):
            _run(plate)
    assert not (plate / "masks").exists()
    assert (plate / "fov.ome.zarr" / RECORDS / "batch_plan.json").is_file()
    committed = {(name, role) for name, role in
                 ((n, r) for r in ("cell", "nucleus")
                  for n in IS._mask_names(plate / "masks" / f"{r}_mask_stack"))}
    assert committed == {(name, role) for role, name in written}
    assert 0 < len(committed) < 8

    redone = []

    def record(path, mask):
        redone.append((Path(path).name, IS._mask_label_name(Path(path).parent)))
        return real(path, mask)

    monkeypatch.setattr(O, "_write_mask", record)
    _run(plate, resume=True)
    fields = sorted(IS._image_entries(plate / "fov.ome.zarr"))
    assert set(redone) == {(n, r) for n in fields for r in ("cell", "nucleus")} - committed
    _same(_labels(clean), _labels(plate))
    _same(_merged(clean), _merged(plate))
    assert not (plate / "masks").exists()
    # keep_intermediate is on in these runs: the records stay for a rerun.
    assert (plate / "fov.ome.zarr" / RECORDS / "batch_plan.json").is_file()


def test_an_older_project_with_masks_npz_still_reads(
        yokogawa_cellvoyager_dir, value_cellpose, monkeypatch, tmp_path):
    """A project an earlier version left with masks/*.npz archives (killed
    before segmenting) finishes with preprocess off from those archives;
    the real masks/ folder wins over the store's record folder."""
    import spacr.io as IO
    import spacr.object as O
    from spacr import _mask_batches as MB
    from spacr.image_store import _mask_run_dir

    plate = yokogawa_cellvoyager_dir["src"]
    clean = tmp_path / "clean" / plate.name
    shutil.copytree(plate, clean)
    _run(clean)

    def killed(*args, **kwargs):
        raise KeyboardInterrupt("killed before segmenting")

    with monkeypatch.context() as patch:
        patch.setattr(IO, "_mask_batches_stay_in_memory", lambda *a, **k: False)
        patch.setattr(O, "generate_cellpose_masks_sam", killed)
        with pytest.raises(KeyboardInterrupt):
            _run(plate)
    masks = plate / "masks"
    archives = sorted(p.name for p in masks.glob("stack_*_norm.npz"))
    assert len(archives) == 2
    assert _mask_run_dir(masks) == os.fspath(masks)
    assert [Path(p).name for p in MB._batch_paths(masks, os.listdir)] == archives

    read = []
    real_open = MB._open_batch

    def opened(path, **kwargs):
        read.append((Path(path).name, os.path.isfile(path)))
        return real_open(path, **kwargs)

    monkeypatch.setattr(O, "open_mask_batch", opened)
    _run(plate, preprocess=False, resume=True)
    assert read and all(is_file for _n, is_file in read)
    assert {name for name, _f in read} == set(archives)
    _same(_labels(clean), _labels(plate))
    _same(_merged(clean), _merged(plate))


def test_the_record_folder_follows_the_project_layout(tmp_path):
    import spacr.image_store as IS

    project = tmp_path / "proj"
    project.mkdir()
    masks = project / "masks"
    # No store yet: the masks folder itself (a file-stack run).
    assert IS._mask_run_dir(masks) == os.fspath(masks)
    writer = IS._stack_writer(project, "plate1")
    stack = np.zeros((16, 16, 2), np.uint16)
    IS._write_stack_field(writer, "plate1_B03_1_1.npy", stack, [1, 2])
    expected = os.fspath(project / "fov.ome.zarr" / IS.MASK_RUN_RECORDS)
    assert IS._mask_run_dir(masks) == expected
    assert not masks.exists() and not Path(expected).exists()
    # A real masks/ folder (an older project) always wins.
    masks.mkdir()
    assert IS._mask_run_dir(masks) == os.fspath(masks)
    masks.rmdir()
    # A folder not named masks is never redirected (OPS per-well folders).
    other = project / "phenotype_masks"
    assert IS._mask_run_dir(other) == os.fspath(other)
    Path(expected).mkdir()
    # Once a run keeps its records in the store they stay there, even when
    # something creates masks/ in the middle of the run.
    masks.mkdir()
    assert IS._mask_run_dir(masks) == expected
    masks.rmdir()
    assert IS._remove_mask_run_records(project) and not Path(expected).exists()
    assert not IS._remove_mask_run_records(project)
