"""Item 958: the OPS objects phase resumes per window.

On the synthetic five-field well of
``test_the_ops_engine_runs_a_well_end_to_end``, stitched once, the objects
and decode phases are run whole, stopped part way and resumed, and resumed
after a segmentation setting changed. Each run's ``ops_objects`` and
``ops_barcodes`` are compared with an uninterrupted run's.
"""
from __future__ import annotations

import hashlib
import os
import random
import shutil
import sqlite3

import pytest

pytest.importorskip("tifffile")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")
pd = pytest.importorskip("pandas")

from spacr import ops_engine
from tests.test_the_ops_engine_runs_a_well_end_to_end import (
    OVERLAP, _plate, _ThresholdModel)

PHASES = ("objects", "decode")


class _Stop(RuntimeError):
    """The interruption: raised by the model part way through the well."""


class _CountingModel(_ThresholdModel):
    """The threshold model, counting its windows and stopping after ``stop``."""

    def __init__(self, stop=None):
        self.calls = 0
        self.stop = stop

    def eval(self, image, batch_size=8, resample=True, channels=None,
             channel_axis=None, z_axis=None, normalize=True, rescale=None,
             diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
             do_3D=False, anisotropy=None, flow3D_smooth=0,
             stitch_threshold=0.0, min_size=15, max_size_fraction=0.4,
             niter=None, augment=False, tile_overlap=0.1, bsize=None,
             compute_masks=True, progress=None):
        """Label one window, unless this is the window to stop at.

        The installed cellpose's eval parameters are written out, as in the
        threshold model, so an argument cellpose removed fails here too.
        """
        if self.stop is not None and self.calls >= self.stop:
            raise _Stop(f"stopped after {self.calls} windows")
        self.calls += 1
        return super().eval(image, batch_size=batch_size,
                            channel_axis=channel_axis, diameter=diameter)


@pytest.fixture(scope="module")
def stitched(tmp_path_factory):
    """The synthetic well, stitched once.

    :returns: ``(root, settings, library)``; ``root / 'stitched'`` holds the
        stitched ``measurements.db``.
    """
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ops_engine, "_RASTER_OVERLAP", OVERLAP)
    monkeypatch.setattr(ops_engine, "_WINDOW", 256)
    monkeypatch.setattr(ops_engine, "_WINDOW_OVERLAP", 48)
    root = tmp_path_factory.mktemp("ops_objects_resume")
    raw = root / "raw"
    raw.mkdir()
    _nuclei, codes, _centre, _tops = _plate(raw)
    settings = {"genotype_source": str(raw), "dst_root": str(root / "stitched"),
                "plate": "synthetic", "ops_gpu": False, "n_workers": 1}
    library = sorted(set(codes)) + ["TTTT", "GGGG"]
    ops_engine.run_ops(settings, phases=("stitch",))
    yield root, settings, library
    monkeypatch.undo()


def _run(stitched, folder, monkeypatch, model=None, **extra):
    """Run the objects and decode phases on a copy of the stitched well.

    :returns: ``(result, model)``.
    """
    root, settings, library = stitched
    if not os.path.exists(folder / "measurements.db"):
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / "stitched" / "measurements.db", folder)
    model = model or _CountingModel()
    monkeypatch.setattr(ops_engine, "_cellpose_model", lambda s, gpu: model)
    result = ops_engine.run_ops({**settings, "dst_root": str(folder), **extra},
                                phases=PHASES, library=library)
    return result, model


def _tables(folder):
    """md5 of ``ops_objects`` and ``ops_barcodes`` as stored, row by row."""
    out = {}
    with sqlite3.connect(folder / "measurements.db") as conn:
        for table in ("ops_objects", "ops_barcodes"):
            frame = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            out[table] = hashlib.md5(
                frame.to_csv(index=False).encode("utf-8")).hexdigest()
    return out


def _window_files(folder):
    """Count mask/marker files while checking that their batch grouping is saved."""
    well = folder / "objects_windows" / "A1"
    if not well.is_dir():
        return []
    assert (well / 'mask_batch_plan.json').is_file()
    return sorted(name for name in os.listdir(well) if name.startswith('w'))


@pytest.fixture(scope="module")
def reference(stitched, tmp_path_factory):
    """One uninterrupted run with checkpoints off, as before item 958."""
    monkeypatch = pytest.MonkeyPatch()
    folder = tmp_path_factory.mktemp("reference")
    _result, model = _run(stitched, folder, monkeypatch,
                          ops_objects_resume=False)
    monkeypatch.undo()
    return _tables(folder), model.calls


def test_the_default_run_equals_one_without_checkpoints_and_leaves_no_files(
        stitched, reference, tmp_path, monkeypatch):
    """Checkpoints on (the default), uninterrupted: same tables, no files left."""
    tables, segmented = reference
    assert segmented >= 4
    result, model = _run(stitched, tmp_path, monkeypatch)
    assert model.calls == segmented
    assert _tables(tmp_path) == tables
    assert not (tmp_path / "objects_windows").exists()
    assert "windows_resumed" not in result["wells"]["A1"]["objects"]


def test_a_well_stopped_part_way_resumes_to_the_same_tables(
        stitched, reference, tmp_path, monkeypatch):
    """Stop at a window drawn with a fixed seed, resume: md5-equal tables."""
    tables, segmented = reference
    stop = random.Random(958).randrange(1, segmented)
    with pytest.raises(_Stop):
        _run(stitched, tmp_path, monkeypatch, model=_CountingModel(stop))
    files = _window_files(tmp_path)
    assert len(files) == 2 * stop
    assert all(name.endswith((".npz", ".done.json")) for name in files)

    result, model = _run(stitched, tmp_path, monkeypatch)
    assert model.calls == segmented - stop
    resumed = result["wells"]["A1"]["objects"]["windows_resumed"]
    assert (resumed["reused"], resumed["computed"]) == (stop, segmented - stop)
    assert resumed["deleted"] == 2 * segmented
    assert _tables(tmp_path) == tables
    assert not (tmp_path / "objects_windows").exists()


def test_a_changed_segmentation_setting_reuses_no_window(
        stitched, tmp_path, monkeypatch):
    """Stale-key control: the diameter changes between the stop and the resume.

    Every window is segmented again, and the tables equal a fresh run with
    the new diameter.
    """
    fresh = tmp_path / "fresh"
    _result, model = _run(stitched, fresh, monkeypatch, cellpose_diameter=17)
    segmented = model.calls
    resumed = tmp_path / "resumed"
    with pytest.raises(_Stop):
        _run(stitched, resumed, monkeypatch, model=_CountingModel(2),
             cellpose_diameter=12)
    assert len(_window_files(resumed)) == 4
    result, model = _run(stitched, resumed, monkeypatch, cellpose_diameter=17)
    assert model.calls == segmented
    assert "windows_resumed" not in result["wells"]["A1"]["objects"]
    assert _tables(resumed) == _tables(fresh)


def test_kept_windows_are_all_reused_and_a_touched_tile_is_segmented_again(
        stitched, reference, tmp_path, monkeypatch):
    """With ``ops_objects_keep_windows`` every window stays and a rerun reuses
    them; a tile file whose modification time changes sends exactly the
    windows it covers back to the model."""
    tables, segmented = reference
    _run(stitched, tmp_path, monkeypatch, ops_objects_keep_windows=True)
    assert len(_window_files(tmp_path)) == 2 * segmented

    result, model = _run(stitched, tmp_path, monkeypatch,
                         ops_objects_keep_windows=True)
    assert model.calls == 0
    assert result["wells"]["A1"]["objects"]["windows_resumed"]["reused"] == segmented
    assert _tables(tmp_path) == tables

    raw = stitched[0] / "raw"
    tile = next(path for path in sorted(raw.rglob("*.tif"))
                if "_c1_" in path.name and "Site-0" in path.name)
    stat = tile.stat()
    os.utime(tile, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    try:
        result, model = _run(stitched, tmp_path, monkeypatch,
                             ops_objects_keep_windows=True)
    finally:
        os.utime(tile, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    resumed = result["wells"]["A1"]["objects"]["windows_resumed"]
    assert 0 < model.calls < segmented
    assert (resumed["reused"], resumed["computed"]) == (segmented - model.calls,
                                                        model.calls)
    assert _tables(tmp_path) == tables


def test_an_unreadable_window_file_is_segmented_again(
        stitched, reference, tmp_path, monkeypatch):
    """A labels file cut short is not trusted: that window is recomputed."""
    tables, segmented = reference
    folder = tmp_path / "windows"
    _run(stitched, tmp_path, monkeypatch, ops_objects_keep_windows=True,
         ops_objects_windows_dir=str(folder))
    assert not (tmp_path / "objects_windows").exists()
    labels = sorted((folder / "A1").glob("*.npz"))
    assert len(labels) == segmented
    labels[0].write_bytes(labels[0].read_bytes()[:100])
    result, model = _run(stitched, tmp_path, monkeypatch,
                         ops_objects_windows_dir=str(folder))
    assert model.calls == 1
    assert result["wells"]["A1"]["objects"]["windows_resumed"]["reused"] == segmented - 1
    assert _tables(tmp_path) == tables
    assert not (folder / "A1").exists() and folder.is_dir()


def test_checkpoints_off_write_nothing(stitched, tmp_path, monkeypatch):
    """``ops_objects_resume=False`` neither reads nor writes window files."""
    with pytest.raises(_Stop):
        _run(stitched, tmp_path, monkeypatch, model=_CountingModel(2),
             ops_objects_resume=False)
    assert not (tmp_path / "objects_windows").exists()


def test_the_window_folder_and_its_settings():
    """Where the files go, and the three settings' defaults."""
    from spacr.ops_settings import OPS_DEFAULTS

    assert OPS_DEFAULTS["ops_objects_resume"] is True
    assert OPS_DEFAULTS["ops_objects_keep_windows"] is False
    assert OPS_DEFAULTS["ops_objects_windows_dir"] is None
    db = os.path.join("run", "measurements.db")
    assert ops_engine._objects_windows_folder({}, db, "B2") == os.path.join(
        os.path.abspath("run"), "objects_windows", "B2")
    assert ops_engine._objects_windows_folder(
        {"ops_objects_windows_dir": "elsewhere"}, db, "B2") == os.path.join(
        "elsewhere", "B2")
    assert ops_engine._objects_windows_folder(
        {"ops_objects_resume": False}, db, "B2") is None
