"""CPU-only branch coverage for :func:`spacr.object.generate_cellpose_masks_sam`.

The real Cellpose-SAM network is replaced with a deterministic fake
(``spacr.object.cp_models.CellposeModel`` is monkeypatched), CUDA is forced off,
and the timelapse trackers / movie writer / plot helper are swapped for
recorders. Every branch therefore runs on CPU in milliseconds while the
assertions still check real artefacts:

  * the rows written to ``measurements/measurements.db::object_counts``
  * the ``.npy`` label masks written to ``<src>/<object>_mask_stack``
  * the exact kwargs handed to ``model.eval`` and to each tracking backend
  * the batch shapes/dtypes produced by the channel-selection branches
  * the progress lines printed for skipped files and re-cut timelapse batches

Two tests are ``xfail(strict=True)`` and assert the CORRECT behaviour of paths
that are currently broken (see ``suspected_bugs`` in the run report).
"""
from __future__ import annotations

import os
import sqlite3
import types
from pathlib import Path

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spacr.object as O

from tests.cellpose_api_contract import (
    MISSING_CHANNEL_AXIS,
    configured_eval_arguments,
    emulate_pretrained_model,
    eval_arguments,
    init_arguments,
)
from tests.conftest import check_cellpose_eval_call


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """Force the CPU code path even on a CUDA box and record empty_cache()."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    return calls


@pytest.fixture
def fake_model(monkeypatch):
    """Install a deterministic stand-in for ``cellpose.models.CellposeModel``.

    Returns a mutable holder; set ``holder['n_objects']`` before the call to
    control how many labelled blobs each returned mask carries.
    """
    holder = {"model": None, "n_objects": 2}

    class _M:
        """``CellposeModel`` double declaring the installed 4.0.7 signature.

        Neither method takes ``**kwargs``: ``generate_cellpose_masks_sam`` is a
        real call site, so an argument cellpose 4 removed has to raise
        ``TypeError`` here rather than vanish into a catch-all.
        """

        def __init__(self, gpu=False, pretrained_model="cpsam", model_type=None,
                     diam_mean=None, device=None, nchan=None,
                     use_bfloat16=True):
            self.gpu = gpu
            self.pretrained_model = pretrained_model
            self.device = device
            self.init_kwargs = init_arguments(locals())
            self.loaded_model = emulate_pretrained_model(pretrained_model,
                                                         model_type)
            self.eval_kwargs = []
            self.eval_configured = []
            self.eval_inputs = []
            holder["model"] = self

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                 normalize=True, invert=False, rescale=None, diameter=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                 anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=256,
                 compute_masks=True, progress=None):
            check_cellpose_eval_call(x, channel_axis)
            self.eval_kwargs.append(eval_arguments(locals()))
            self.eval_configured.append(configured_eval_arguments(locals()))
            imgs = [np.asarray(im) for im in x]
            self.eval_inputs.append(imgs)
            masks, flows = [], []
            for im in imgs:
                h, w = im.shape[:2]
                m = np.zeros((h, w), dtype=np.uint16)
                if holder["n_objects"] >= 1:
                    m[2:8, 2:8] = 1
                if holder["n_objects"] >= 2:
                    m[12:18, 12:18] = 2
                masks.append(m)
                flows.append(np.zeros((h, w), dtype=np.float32))
            # THREE values. This used to return four -- the cellpose 3 shape.
            # The installed 4.0.7 returns (masks, flows, styles) on both of its
            # return paths, so a four-value double would bless a
            # ``masks, flows, styles, diams = model.eval(...)`` unpack that
            # raises ValueError on every real run.
            return masks, flows, None

    monkeypatch.setattr(O, "cp_models", types.SimpleNamespace(CellposeModel=_M))
    return holder


@pytest.fixture
def fake_plot(monkeypatch):
    """Replace spacr.plot.plot_cellpose4_output with a recorder."""
    import spacr.plot as PL
    calls = []

    def _rec(batch, masks, flows, **kwargs):
        calls.append({"batch": list(batch), "masks": list(masks),
                      "flows": list(flows), "kwargs": kwargs})

    monkeypatch.setattr(PL, "plot_cellpose4_output", _rec)
    return calls


@pytest.fixture
def fake_timelapse(monkeypatch):
    """Replace the timelapse movie writer / trackers / motility hook."""
    import spacr.timelapse as TL
    rec = {"movie": [], "btrack": [], "trackpy": [], "trackastra": [],
           "motility": []}

    def _movie(arrays, filenames, save_path, fps=10):
        rec["movie"].append({"n_frames": len(arrays),
                             "filenames": list(filenames),
                             "save_path": save_path, "fps": fps})

    def _as_stack(masks):
        return [np.asarray(m, dtype=np.uint16) for m in masks]

    def _btrack(**kw):
        rec["btrack"].append(kw)
        return _as_stack(kw["masks_3D"])

    def _trackpy(**kw):
        rec["trackpy"].append(kw)
        return _as_stack(kw["masks"])

    def _trackastra(**kw):
        rec["trackastra"].append(kw)
        return _as_stack(kw["masks"])

    def _motility(settings):
        rec["motility"].append(settings)
        return "motility-done"

    monkeypatch.setattr(TL, "_npz_to_movie", _movie)
    monkeypatch.setattr(TL, "_btrack_track_cells", _btrack)
    monkeypatch.setattr(TL, "_trackpy_track_cells", _trackpy)
    monkeypatch.setattr(TL, "_trackastra_track_cells", _trackastra)
    monkeypatch.setattr(TL, "automated_motility_assay", _motility)
    return rec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_npz(src: Path, name: str = "batch1.npz", n: int = 3, h: int = 32,
               w: int = 32, c: int = 2, seed: int = 0):
    """Write one pre-batched npz exactly like spacr's preprocessing does."""
    src.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if n == 0:
        data = np.zeros((0, h, w, c), dtype=np.uint16)
        filenames = np.array([], dtype="<U32")
    else:
        data = rng.integers(0, 4000, size=(n, h, w, c)).astype(np.uint16)
        filenames = np.array([f"plate1_A01_{i + 1}.npy" for i in range(n)])
    np.savez(src / name, data=data, filenames=filenames)
    return data, [str(f) for f in filenames]


def _settings(src, **over):
    """Minimal settings dict; the rest is filled by spacr's own defaults."""
    s = {
        "src": str(src),
        "cell_channel": 0,
        "nucleus_channel": 1,
        "pathogen_channel": None,
        "magnification": 20,
        "batch_size": 50,
        "verbose": False,
        "plot": False,
        "save": True,
        "timelapse": False,
        "n_jobs": 1,
        # keep merge/split/filter a no-op unless a test asks for it
        "cell_min_object_area": 0,
        "nucleus_min_object_area": 0,
        "pathogen_min_object_area": 0,
    }
    s.update(over)
    return s


def _counts(db_path):
    """Return [(file_name, count_type, object_count), ...] from the run DB."""
    con = sqlite3.connect(str(db_path))
    try:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        if "object_counts" not in names:
            return None
        return sorted(con.execute(
            "SELECT file_name, count_type, object_count FROM object_counts"))
    finally:
        con.close()


def _mask_files(src, object_type="cell"):
    folder = Path(src) / f"{object_type}_mask_stack"
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir())


# ---------------------------------------------------------------------------
# Happy path — non-timelapse
# ---------------------------------------------------------------------------

def test_basic_run_writes_masks_counts_and_uses_cpu_model(tmp_path, fake_model,
                                                          force_cpu, capsys):
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=3)

    assert O.generate_cellpose_masks_sam(str(src), _settings(src), "cell") is None

    # --- the model was built for CPU with the SAM checkpoint ---------------
    model = fake_model["model"]
    assert model.pretrained_model == "cpsam"
    assert model.gpu is False
    assert str(model.device) == "cpu"
    assert force_cpu == [1], "torch.cuda.empty_cache() must be called once"

    # --- eval got the whole batch with spaCR's SAM parameters --------------
    assert len(model.eval_kwargs) == 1
    kw = model.eval_kwargs[0]
    assert kw["batch_size"] == 3
    assert kw["normalize"] is False
    assert kw["channel_axis"] == -1
    assert kw["diameter"] is None
    assert kw["progress"] is True
    assert kw["min_size"] == 0          # cell_min_area default
    assert kw["resample"] is True       # _get_object_settings('cell')
    assert kw["flow_threshold"] == 1.0  # cell_FT default
    assert kw["cellprob_threshold"] == 0  # cell_CP_prob default

    # two-channel stack -> both cellpose channels handed to the model
    imgs = model.eval_inputs[0]
    assert len(imgs) == 3
    assert imgs[0].shape == (32, 32, 2)
    assert imgs[0].dtype == np.float32
    assert imgs[0].max() <= 1.0

    # --- masks on disk -----------------------------------------------------
    assert _mask_files(src) == sorted(names)
    written = np.load(src / "cell_mask_stack" / names[0])
    assert written.shape == (32, 32)
    assert written.dtype == np.uint16
    assert sorted(np.unique(written).tolist()) == [0, 1, 2]

    # --- object counts in the run database ---------------------------------
    db = tmp_path / "measurements" / "measurements.db"
    assert db.is_file()
    rows = _counts(db)
    assert rows == sorted((n, "cell_before_filtration", 2) for n in names)

    out = capsys.readouterr().out
    assert "Torch CUDA is not available, using CPU" in out
    assert "saving to DB" in out
    assert "Found 2.0 cell/FOV" in out


def test_verbose_displays_settings_table_and_channels(tmp_path, fake_model,
                                                      monkeypatch, capsys):
    src = tmp_path / "stack"
    _write_npz(src, n=2)
    shown = []
    monkeypatch.setattr(O, "display", lambda obj, *a, **k: shown.append(obj))

    O.generate_cellpose_masks_sam(str(src), _settings(src, verbose=True), "cell")

    assert len(shown) == 1
    df = shown[0]
    assert list(df.columns) == ["setting_key", "setting_value"]
    assert "src" in set(df["setting_key"])
    # every value was stringified before display
    assert all(isinstance(v, str) for v in df["setting_value"])

    # cell + nucleus channels remap to [0, 1] and are printed in verbose mode
    assert "[0, 1]" in capsys.readouterr().out


def test_single_channel_stack_falls_back_to_channel_zero(tmp_path, fake_model):
    src = tmp_path / "stack"
    _write_npz(src, n=3, c=1)

    O.generate_cellpose_masks_sam(str(src), _settings(src), "cell")

    imgs = fake_model["model"].eval_inputs[0]
    assert [im.shape for im in imgs] == [(32, 32, 1)] * 3
    assert all(im.dtype == np.float32 for im in imgs)


def test_pathogen_channel_is_mirrored_into_the_cellpose_channel(tmp_path,
                                                                fake_model):
    """pathogen_channel -> cellpose_pathogen_channel, and the pathogen object
    settings (single channel, resample off) reach model.eval."""
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=2, c=3)
    settings = _settings(src, cell_channel=0, nucleus_channel=1,
                         pathogen_channel=2, pathogen_min_area=0)

    O.generate_cellpose_masks_sam(str(src), settings, "pathogen")

    assert settings["cellpose_pathogen_channel"] == 2
    imgs = fake_model["model"].eval_inputs[0]
    # only the remapped pathogen plane is segmented
    assert [im.shape for im in imgs] == [(32, 32, 1)] * 2
    kw = fake_model["model"].eval_kwargs[0]
    assert kw["resample"] is False       # _get_object_settings('pathogen')
    assert kw["min_size"] == 0

    assert _mask_files(src, "pathogen") == sorted(names)
    rows = _counts(tmp_path / "measurements" / "measurements.db")
    assert rows == sorted((n, "pathogen_before_filtration", 2) for n in names)


def test_missing_channel_for_object_type_raises(tmp_path, fake_model):
    src = tmp_path / "stack"
    _write_npz(src, n=1)
    settings = _settings(src, cell_channel=0, nucleus_channel=None,
                         pathogen_channel=None)

    with pytest.raises(ValueError) as exc:
        O.generate_cellpose_masks_sam(str(src), settings, "pathogen")
    assert "pathogen" in str(exc.value)
    # the error is raised before any model is instantiated
    assert fake_model["model"] is None


def test_existing_mask_file_is_skipped_and_preserved(tmp_path, fake_model,
                                                     capsys):
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=3)
    out_folder = src / "cell_mask_stack"
    out_folder.mkdir()
    sentinel = np.full((32, 32), 7, dtype=np.uint16)
    np.save(out_folder / names[0], sentinel)

    O.generate_cellpose_masks_sam(str(src), _settings(src), "cell")

    out = capsys.readouterr().out
    assert f"File {names[0]} already exists in the output folder" in out

    # only the two unprocessed FOVs reached the model
    assert len(fake_model["model"].eval_inputs[0]) == 2
    # the pre-existing mask was not overwritten
    assert np.array_equal(np.load(out_folder / names[0]), sentinel)
    assert _mask_files(src) == sorted(names)

    rows = _counts(tmp_path / "measurements" / "measurements.db")
    assert rows == sorted((n, "cell_before_filtration", 2) for n in names[1:])


def test_fully_processed_batch_is_skipped_entirely(tmp_path, fake_model):
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=3)
    out_folder = src / "cell_mask_stack"
    out_folder.mkdir()
    for n in names:
        np.save(out_folder / n, np.zeros((32, 32), dtype=np.uint16))

    O.generate_cellpose_masks_sam(str(src), _settings(src), "cell")

    # the model was constructed but never evaluated, and nothing was recorded
    assert fake_model["model"] is not None
    assert fake_model["model"].eval_inputs == []
    assert _counts(tmp_path / "measurements" / "measurements.db") is None
    assert _mask_files(src) == sorted(names)


def test_plot_enabled_calls_plot_helper_and_skips_mask_check(tmp_path,
                                                             fake_model,
                                                             fake_plot):
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=3)
    # a pre-existing mask must NOT be filtered out when plot=True
    out_folder = src / "cell_mask_stack"
    out_folder.mkdir()
    np.save(out_folder / names[0], np.zeros((32, 32), dtype=np.uint16))

    O.generate_cellpose_masks_sam(str(src), _settings(src, plot=True), "cell")

    assert len(fake_model["model"].eval_inputs[0]) == 3  # _check_masks skipped
    assert len(fake_plot) == 1
    call = fake_plot[0]
    assert len(call["batch"]) == 3
    assert call["kwargs"]["nr"] == 3
    assert call["kwargs"]["cmap"] == "inferno"
    assert call["kwargs"]["figuresize"] == 10
    assert call["masks"][0].shape == (32, 32)


def test_save_disabled_writes_no_mask_files(tmp_path, fake_model):
    src = tmp_path / "stack"
    _write_npz(src, n=2)

    O.generate_cellpose_masks_sam(str(src), _settings(src, save=False), "cell")

    # the folder is still created, but stays empty
    assert (src / "cell_mask_stack").is_dir()
    assert _mask_files(src) == []
    # counts are still recorded
    assert len(_counts(tmp_path / "measurements" / "measurements.db")) == 2


def test_area_filter_empties_masks_and_reports_zero(tmp_path, fake_model,
                                                    capsys):
    """min_area larger than every object -> filtered masks are empty."""
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=2)
    # each fake object is 6x6 = 36 px, so min_area=100 removes both
    settings = _settings(src, cell_min_area=100)

    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    rows = _counts(tmp_path / "measurements" / "measurements.db")
    assert rows == sorted((n, "cell_before_filtration", 0) for n in names)
    saved = np.load(src / "cell_mask_stack" / names[0])
    assert saved.max() == 0
    assert "Found 0.0 cell/FOV. average size: 0.000 px2" in capsys.readouterr().out


def test_model_returning_empty_masks_reports_zero_objects(tmp_path, fake_model,
                                                          capsys):
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=2)
    fake_model["n_objects"] = 0

    O.generate_cellpose_masks_sam(str(src), _settings(src), "cell")

    rows = _counts(tmp_path / "measurements" / "measurements.db")
    assert rows == sorted((n, "cell_before_filtration", 0) for n in names)
    assert "Found 0.0 cell/FOV. average size: 0.000 px2" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Timelapse paths
# ---------------------------------------------------------------------------

def _tl_settings(src, **over):
    base = dict(timelapse=True, timelapse_objects=["cell"], batch_size=2,
                timelapse_displacement=None, timelapse_memory=3,
                timelapse_remove_transient=False, timelapse_frame_limits=[5])
    base.update(over)
    return _settings(src, **base)


def test_timelapse_objects_must_be_trackable(tmp_path, fake_model,
                                             fake_timelapse, capsys):
    src = tmp_path / "stack"
    _write_npz(src, n=2)
    settings = _tl_settings(src, timelapse_objects=["organelle"])

    assert O.generate_cellpose_masks_sam(str(src), settings, "cell") is None

    assert "must be a subset of" in capsys.readouterr().out
    assert fake_model["model"].eval_inputs == []
    assert fake_timelapse["movie"] == []
    assert not (tmp_path / "movies").exists()


def test_timelapse_btrack_recuts_batch_and_tracks(tmp_path, fake_model,
                                                  fake_timelapse, monkeypatch,
                                                  capsys):
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=3)
    # cpu_count()-2 <= 0 forces the n_jobs floor branch
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    settings = _tl_settings(src, batch_size=2, timelapse_frame_limits=[0, 2],
                            timelapse_displacement=50, timelapse_mode="btrack",
                            n_jobs=1)

    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    out = capsys.readouterr().out
    assert "Changed batch_size:2 to 3" in out
    assert "Cut batch at indecies: [0, 2], New batch_size: 2" in out
    assert settings["timelapse_batch_size"] == 3

    # the movie is written from the (re-cut) 2-frame batch
    assert len(fake_timelapse["movie"]) == 1
    movie = fake_timelapse["movie"][0]
    assert movie["n_frames"] == 2
    assert movie["fps"] == 2
    assert movie["filenames"] == names[:2]
    assert movie["save_path"] == str(tmp_path / "movies" /
                                     "timelapse_cell_batch1.mp4")
    assert (tmp_path / "movies").is_dir()

    assert len(fake_timelapse["btrack"]) == 1
    kw = fake_timelapse["btrack"][0]
    assert kw["radius"] == 50            # timelapse_displacement honoured
    assert kw["n_jobs"] == 1             # cpu_count()-2 floored to 1
    assert kw["object_type"] == "cell"
    assert kw["mode"] == "btrack"
    assert kw["name"] == "batch1"
    assert kw["batch_list"] is None
    assert kw["run_optimization"] is True
    assert kw["max_objects_for_optimization"] == 20000
    assert len(kw["masks_3D"]) == 2
    assert fake_timelapse["trackpy"] == [] and fake_timelapse["trackastra"] == []

    rows = _counts(tmp_path / "measurements" / "measurements.db")
    assert rows == sorted((n, "cell_timelapse", 2) for n in names[:2])
    assert _mask_files(src) == sorted(names[:2])


def test_timelapse_btrack_default_radius(tmp_path, fake_model, fake_timelapse):
    src = tmp_path / "stack"
    _write_npz(src, n=2)
    settings = _tl_settings(src, batch_size=2, timelapse_mode="btrack",
                            timelapse_displacement=None)

    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    assert fake_timelapse["btrack"][0]["radius"] == 100
    assert fake_timelapse["btrack"][0]["n_jobs"] >= 1


def test_timelapse_trackastra_gets_masks_and_images(tmp_path, fake_model,
                                                    fake_timelapse):
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=2)
    settings = _tl_settings(src, batch_size=2, timelapse_mode="trackastra")

    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    assert len(fake_timelapse["trackastra"]) == 1
    kw = fake_timelapse["trackastra"][0]
    assert kw["model_name"] == "general_2d"
    assert kw["linking_mode"] == "greedy"
    assert kw["mode"] == "trackastra"
    assert kw["batch_filenames"] == names
    # trackastra needs the raw intensity batch, not just the labels
    assert np.asarray(kw["images"]).shape == (2, 32, 32, 2)
    assert len(kw["masks"]) == 2
    assert fake_timelapse["btrack"] == []
    assert _mask_files(src) == sorted(names)


@pytest.mark.parametrize("mode,expected_iou", [("trackpy", False), ("iou", True)])
def test_timelapse_trackpy_and_iou_modes(tmp_path, fake_model, fake_timelapse,
                                         mode, expected_iou):
    src = tmp_path / "stack"
    _write_npz(src, n=2)
    settings = _tl_settings(src, batch_size=2, timelapse_mode=mode,
                            timelapse_displacement=25, timelapse_memory=5)

    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    assert len(fake_timelapse["trackpy"]) == 1
    kw = fake_timelapse["trackpy"][0]
    assert kw["track_by_iou"] is expected_iou
    assert kw["mode"] == mode
    assert kw["timelapse_displacement"] == 25
    assert kw["timelapse_memory"] == 5
    assert fake_timelapse["btrack"] == []


def test_timelapse_untracked_object_uses_plain_mask_stack(tmp_path, fake_model,
                                                          fake_timelapse):
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=2)
    # 'cell' is segmented but only nuclei are tracked
    settings = _tl_settings(src, batch_size=2, timelapse_objects=["nucleus"],
                            timelapse_mode="btrack")

    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    assert fake_timelapse["btrack"] == []
    assert fake_timelapse["trackpy"] == []
    assert fake_timelapse["trackastra"] == []
    assert _mask_files(src) == sorted(names)
    rows = _counts(tmp_path / "measurements" / "measurements.db")
    assert rows == sorted((n, "cell_timelapse", 2) for n in names)


def test_timelapse_plot_uses_single_frame_and_object_numbers(tmp_path,
                                                             fake_model,
                                                             fake_timelapse,
                                                             fake_plot):
    src = tmp_path / "stack"
    _write_npz(src, n=2)
    settings = _tl_settings(src, batch_size=2, plot=True,
                            timelapse_objects=["nucleus"])

    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    assert len(fake_plot) == 1
    kwargs = fake_plot[0]["kwargs"]
    assert kwargs["nr"] == 1
    assert kwargs["print_object_number"] is True
    assert len(fake_plot[0]["batch"]) == 2


def test_timelapse_motility_hook_runs(tmp_path, fake_model, fake_timelapse):
    src = tmp_path / "stack"
    _write_npz(src, n=2)
    settings = _tl_settings(src, batch_size=2, timelapse_objects=["nucleus"],
                            motility_analysis=True)

    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    assert len(fake_timelapse["motility"]) == 1
    assert fake_timelapse["motility"][0] is settings


# ---------------------------------------------------------------------------
# Paths that used to be broken. Both bugs are fixed (the plot/save blocks were
# moved inside the per-batch loop in spacr/object.py); the xfail(strict=True)
# markers that documented them have been retired and these now assert the
# correct behaviour directly.
# ---------------------------------------------------------------------------

def test_every_batch_of_a_multi_batch_npz_is_saved(tmp_path, fake_model):
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=4)

    O.generate_cellpose_masks_sam(str(src), _settings(src, batch_size=2), "cell")

    # the database records all four FOVs ...
    rows = _counts(tmp_path / "measurements" / "measurements.db")
    assert rows == sorted((n, "cell_before_filtration", 2) for n in names)
    # ... so all four masks must exist on disk too.
    assert _mask_files(src) == sorted(names)


def test_empty_npz_batch_is_handled_gracefully(tmp_path, fake_model):
    src = tmp_path / "stack"
    _write_npz(src, n=0)

    assert O.generate_cellpose_masks_sam(str(src), _settings(src), "cell") is None
    assert _mask_files(src) == []
