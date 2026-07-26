"""CPU-only behaviour tests for :func:`spacr.object.generate_cellpose_masks`
and the last uncovered corners of :mod:`spacr.object`.

``generate_cellpose_masks`` is the "choose your own model" sibling of
``generate_cellpose_masks_sam``.  It was completely unreachable until this pass:
it called ``utils._get_cellpose_channels(src, nucleus, pathogen, cell)`` while
that helper takes a single settings dict, so every call died with a TypeError
before the first field was read.  Everything below therefore exercises code that
had never run.

Cellpose is the only thing faked.  ``spacr.utils.cp_models`` is swapped for a
recorder, which means :func:`spacr.utils._choose_model` — the function that
decides whether a user-trained checkpoint or the stock ``cpsam`` weights get
loaded — runs for real and its decision is asserted.  Everything else is real:
the SQLite database, the ``.npy`` masks on disk, the channel remapping, the
merge/filter helpers and the timelapse dispatch.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spacr.object as O
import spacr.utils as U


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """Force the CPU path even on a CUDA box and record empty_cache() calls."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    return calls


@pytest.fixture
def fake_cellpose(monkeypatch):
    """Swap ``spacr.utils.cp_models`` for a deterministic recorder.

    ``_choose_model`` still runs for real, so the checkpoint-vs-stock decision
    it makes is observable through ``holder['model'].pretrained_model``.

    Set ``holder['blobs']`` to a list of ``(row_slice, col_slice)`` pairs to
    control the objects every returned mask carries.
    """
    holder = {
        "model": None,
        "blobs": [(slice(2, 8), slice(2, 8)), (slice(12, 18), slice(12, 18))],
    }

    class _M:
        def __init__(self, gpu=None, pretrained_model=None, device=None, **kwargs):
            self.gpu = gpu
            self.pretrained_model = pretrained_model
            self.device = device
            self.init_kwargs = kwargs
            self.eval_kwargs = []
            self.eval_inputs = []
            holder["model"] = self

        def eval(self, x=None, **kwargs):
            self.eval_kwargs.append(kwargs)
            imgs = [np.asarray(im) for im in x]
            self.eval_inputs.append(imgs)
            masks, flows = [], []
            for im in imgs:
                h, w = im.shape[:2]
                m = np.zeros((h, w), dtype=np.uint16)
                for label, (rs, cs) in enumerate(holder["blobs"], start=1):
                    m[rs, cs] = label
                masks.append(m)
                # The per-image flow layout cellpose 4 returns:
                # (rgb, dP, cellprob, p).  parse_cellpose4_output turns this
                # into one flow-rgb array per image.
                flows.append((
                    np.full((h, w, 3), 0.25, dtype=np.float32),
                    np.zeros((2, h, w), dtype=np.float32),
                    np.zeros((h, w), dtype=np.float32),
                    None,
                ))
            return masks, flows, None

    monkeypatch.setattr(U, "cp_models", types.SimpleNamespace(CellposeModel=_M))
    return holder


@pytest.fixture
def fake_timelapse(monkeypatch):
    """Replace the movie writer / trackers / motility hook with recorders."""
    import spacr.timelapse as TL
    rec = {"movie": [], "btrack": [], "trackpy": [], "trackastra": [],
           "ultrack": [], "motility": []}

    def _as_stack(masks):
        return [np.asarray(m, dtype=np.uint16) for m in masks]

    def _movie(arrays, filenames, save_path, fps=10):
        rec["movie"].append({"n_frames": len(arrays),
                             "filenames": list(filenames),
                             "save_path": save_path, "fps": fps})

    def _btrack(**kw):
        rec["btrack"].append(kw)
        return _as_stack(kw["masks_3D"])

    def _trackpy(**kw):
        rec["trackpy"].append(kw)
        return _as_stack(kw["masks"])

    def _trackastra(**kw):
        rec["trackastra"].append(kw)
        return _as_stack(kw["masks"])

    def _ultrack(**kw):
        rec["ultrack"].append(kw)
        return _as_stack(kw["masks"])

    def _motility(settings):
        rec["motility"].append(settings)
        return "motility-done"

    monkeypatch.setattr(TL, "_npz_to_movie", _movie)
    monkeypatch.setattr(TL, "_btrack_track_cells", _btrack)
    monkeypatch.setattr(TL, "_trackpy_track_cells", _trackpy)
    monkeypatch.setattr(TL, "_trackastra_track_cells", _trackastra)
    monkeypatch.setattr(TL, "_ultrack_track_cells", _ultrack)
    monkeypatch.setattr(TL, "automated_motility_assay", _motility)
    return rec


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _write_npz(src, name="batch1.npz", n=3, h=32, w=32, c=2, seed=0):
    """Write one pre-batched .npz exactly like spaCR's preprocessing does."""
    src = Path(src)
    src.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if n == 0:
        data = np.zeros((0, h, w, c), dtype=np.uint16)
        filenames = np.array([], dtype="<U32")
    else:
        data = rng.integers(1, 4000, size=(n, h, w, c)).astype(np.uint16)
        filenames = np.array([f"plate1_A01_{i + 1}.npy" for i in range(n)])
    np.savez(src / name, data=data, filenames=filenames)
    return data, [str(f) for f in filenames]


def _settings(src, **over):
    """Minimal settings; spaCR's own defaults fill the rest."""
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
        "filter": False,
        "timelapse": False,
        "n_jobs": 1,
        # QC is a separate module with its own suite; keep it out of the way
        # unless a test is specifically about it.
        "seg_qc": "off",
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


def _mask_files(src, object_type="nucleus"):
    folder = Path(src) / f"{object_type}_mask_stack"
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir())


def _n_objects(path):
    arr = np.load(path)
    return len(set(np.unique(arr).tolist()) - {0})


# --------------------------------------------------------------------------- #
#  Happy path
# --------------------------------------------------------------------------- #

def test_writes_one_uint16_mask_per_field_and_records_the_counts(
        tmp_path, fake_cellpose, force_cpu):
    """The whole function used to be dead code: this is the end-to-end proof
    that it segments, counts and saves."""
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=3)

    assert O.generate_cellpose_masks(str(src), _settings(src), "nucleus") is None

    assert _mask_files(src) == sorted(names)
    for name in names:
        arr = np.load(src / "nucleus_mask_stack" / name)
        assert arr.dtype == np.uint16
        assert arr.shape == (32, 32)
        # the two blobs the fake model drew, with their areas intact
        assert sorted(np.unique(arr).tolist()) == [0, 1, 2]
        assert int((arr == 1).sum()) == 36
        assert int((arr == 2).sum()) == 36

    assert _counts(tmp_path / "measurements" / "measurements.db") == sorted(
        (n, "nucleus_before_filtration", 2) for n in names)
    assert force_cpu == [1], "torch.cuda.empty_cache() must be called once"


def test_the_model_is_built_for_cpu_with_the_stock_checkpoint(
        tmp_path, fake_cellpose):
    src = tmp_path / "stack"
    _write_npz(src, n=2)

    O.generate_cellpose_masks(str(src), _settings(src), "nucleus")

    model = fake_cellpose["model"]
    assert model.pretrained_model == "cpsam"
    assert model.gpu is False
    assert str(model.device) == "cpu"


def test_eval_receives_every_spacr_parameter_for_this_object_type(
        tmp_path, fake_cellpose):
    """The eval kwargs are where the silent-discard bugs live, so pin them."""
    src = tmp_path / "stack"
    _write_npz(src, n=3)
    settings = _settings(src, batch_size=8, nucleus_FT=0.7, nucleus_CP_prob=-1.5,
                         nucleus_min_area=25)

    O.generate_cellpose_masks(str(src), settings, "nucleus")

    model = fake_cellpose["model"]
    assert len(model.eval_kwargs) == 1
    kw = model.eval_kwargs[0]
    assert kw["batch_size"] == 8
    assert kw["normalize"] is False
    assert kw["channel_axis"] == -1
    # nucleus is the only object at channel 1, remapped to index 0 of the
    # compacted two-channel stack... the cell channel 0 is also extracted, so
    # nucleus lands at 1.
    assert kw["channels"] == [1]
    # _get_diam(20, 'nucleus') == int(0.75 * 20 + 45) == 60
    assert kw["diameter"] == 60
    assert kw["flow_threshold"] == 0.7
    assert kw["cellprob_threshold"] == -1.5
    assert kw["rescale"] is None
    assert kw["resample"] is True
    # nucleus_min_area is documented as "passed to Cellpose as min_size"
    assert kw["min_size"] == 25

    # ... and the images handed over are float32 in [0, 1], one channel per
    # entry of `channels`.
    imgs = model.eval_inputs[0]
    assert len(imgs) == 3
    assert imgs[0].shape == (32, 32, 1)
    assert imgs[0].dtype == np.float32
    assert 0.0 <= imgs[0].min() and imgs[0].max() == pytest.approx(1.0)


def test_pathogen_uses_its_own_thresholds_and_does_not_resample(
        tmp_path, fake_cellpose):
    src = tmp_path / "stack"
    _write_npz(src, n=2, c=3)
    settings = _settings(src, pathogen_channel=2, pathogen_FT=0.3,
                         pathogen_CP_prob=2.0)

    O.generate_cellpose_masks(str(src), settings, "pathogen")

    kw = fake_cellpose["model"].eval_kwargs[0]
    assert kw["flow_threshold"] == 0.3
    assert kw["cellprob_threshold"] == 2.0
    assert kw["resample"] is False, "_get_object_settings turns resample off for pathogens"
    assert kw["diameter"] == 20, "_get_diam(20, 'pathogen') == magnification"
    assert _mask_files(src, "pathogen") == ["plate1_A01_1.npy", "plate1_A01_2.npy"]


@pytest.mark.parametrize(
    "object_type,expected_channels,expected_depth",
    [
        ("nucleus", [0], 1),        # nucleus=1 -> position 0 of [1, 3]
        ("cell", [1, 0], 2),        # cell=3 -> position 1, then the nucleus
        ("pathogen", [2], 1),       # pathogen=5 -> position 2 of [1, 3, 5]
    ],
)
def test_channels_are_remapped_to_the_compacted_stack(
        tmp_path, fake_cellpose, object_type, expected_channels, expected_depth):
    """The .npz only holds the channels of enabled objects, densely
    re-indexed; indexing it with the raw channel number is an IndexError or,
    worse, the wrong channel."""
    src = tmp_path / "stack"
    _write_npz(src, n=2, c=3)
    settings = _settings(src, nucleus_channel=1, cell_channel=3,
                         pathogen_channel=5)

    O.generate_cellpose_masks(str(src), settings, object_type)

    model = fake_cellpose["model"]
    assert model.eval_kwargs[0]["channels"] == expected_channels
    assert model.eval_inputs[0][0].shape == (32, 32, expected_depth)


def test_a_user_trained_checkpoint_reaches_cellpose(tmp_path, fake_cellpose):
    """`pathogen_model` names a fine-tuned checkpoint. If it does not arrive as
    `pretrained_model` the run silently segments with the stock weights."""
    ckpt = tmp_path / "my_pathogens.pth"
    ckpt.write_bytes(b"not really a checkpoint, _choose_model only stats it")

    src = tmp_path / "stack"
    _write_npz(src, n=2, c=3)
    settings = _settings(src, pathogen_channel=2, pathogen_model=str(ckpt))

    O.generate_cellpose_masks(str(src), settings, "pathogen")

    assert fake_cellpose["model"].pretrained_model == str(ckpt)


def test_a_checkpoint_path_that_is_not_there_stops_the_run(tmp_path, fake_cellpose):
    src = tmp_path / "stack"
    _write_npz(src, n=2, c=3)
    settings = _settings(src, pathogen_channel=2,
                         pathogen_model=str(tmp_path / "gone" / "model.pth"))

    with pytest.raises(FileNotFoundError) as excinfo:
        O.generate_cellpose_masks(str(src), settings, "pathogen")
    assert "cpsam" in str(excinfo.value), "the message must name the safe fallback"
    assert _mask_files(src, "pathogen") == []


def test_an_object_type_with_no_channel_raises_before_any_model_loads(
        tmp_path, fake_cellpose):
    src = tmp_path / "stack"
    _write_npz(src, n=2)
    settings = _settings(src, pathogen_channel=None)

    with pytest.raises(ValueError) as excinfo:
        O.generate_cellpose_masks(str(src), settings, "pathogen")
    assert "pathogen" in str(excinfo.value)
    assert fake_cellpose["model"] is None, "no model may be loaded first"


def test_t_stack_is_refused_by_this_generator(tmp_path, fake_cellpose):
    """Only the SAM generator implements 4-D; returning 2-D masks to a user
    whose settings said 4-D is the failure the feature exists to prevent."""
    from spacr.zstack import TStackError

    src = tmp_path / "stack"
    _write_npz(src, n=2)
    settings = _settings(src, t_stack=True, t_axis_order="tzyxc")

    with pytest.raises(TStackError) as excinfo:
        O.generate_cellpose_masks(str(src), settings, "nucleus")
    assert "generate_cellpose_masks_sam" in str(excinfo.value)
    assert fake_cellpose["model"] is None


def test_verbose_prints_the_channel_map_and_the_settings_table(
        tmp_path, fake_cellpose, monkeypatch, capsys):
    shown = []
    monkeypatch.setattr(O, "display", lambda df, *a, **k: shown.append(df))

    src = tmp_path / "stack"
    _write_npz(src, n=2)
    O.generate_cellpose_masks(str(src), _settings(src, verbose=True), "nucleus")

    assert len(shown) == 1
    df = shown[0]
    assert list(df.columns) == ["setting_key", "setting_value"]
    keys = set(df["setting_key"])
    assert {"src", "batch_size", "nucleus_channel"} <= keys
    assert all(isinstance(v, str) for v in df["setting_value"])

    out = capsys.readouterr().out
    # cell=0 and nucleus=1 are both extracted, so the remap is the identity
    assert "{'nucleus': [1], 'cell': [0, 1]}" in out, \
        "the resolved channel map must be printed"
    assert "'diameter': 60" in out, "the object settings must be printed"


def test_a_single_channel_stack_is_duplicated_into_two_channels(
        tmp_path, fake_cellpose):
    """A one-channel .npz cannot be indexed with the remapped channel list;
    the generator duplicates channel 0 instead."""
    src = tmp_path / "stack"
    _write_npz(src, n=2, c=1)
    settings = _settings(src, cell_channel=None, nucleus_channel=0)

    O.generate_cellpose_masks(str(src), settings, "nucleus")

    imgs = fake_cellpose["model"].eval_inputs[0]
    assert imgs[0].shape == (32, 32, 2)
    assert np.array_equal(imgs[0][..., 0], imgs[0][..., 1])


# --------------------------------------------------------------------------- #
#  Batching, skipping and saving
# --------------------------------------------------------------------------- #

def test_every_batch_of_a_multi_batch_npz_is_saved(tmp_path, fake_cellpose):
    """The save block used to sit outside the per-batch loop, so an .npz that
    needed more than one batch wrote only the last one's masks."""
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=5)

    O.generate_cellpose_masks(str(src), _settings(src, batch_size=2), "nucleus")

    model = fake_cellpose["model"]
    assert [len(imgs) for imgs in model.eval_inputs] == [2, 2, 1]
    assert _mask_files(src) == sorted(names)
    assert _counts(tmp_path / "measurements" / "measurements.db") == sorted(
        (n, "nucleus_before_filtration", 2) for n in names)


def test_an_empty_npz_does_not_crash(tmp_path, fake_cellpose):
    """No batch ever ran, so mask_stack was unbound when the save block —
    which used to live outside the loop — dereferenced it."""
    src = tmp_path / "stack"
    _write_npz(src, n=0)

    assert O.generate_cellpose_masks(str(src), _settings(src), "nucleus") is None
    assert _mask_files(src) == []
    assert fake_cellpose["model"].eval_kwargs == []


def test_fields_whose_mask_already_exists_are_not_recomputed(
        tmp_path, fake_cellpose, capsys):
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=3)
    out_dir = src / "nucleus_mask_stack"
    out_dir.mkdir(parents=True)
    np.save(out_dir / names[0], np.zeros((32, 32), dtype=np.uint16))

    O.generate_cellpose_masks(str(src), _settings(src), "nucleus")

    imgs = fake_cellpose["model"].eval_inputs[0]
    assert len(imgs) == 2, "the already-done field must not be segmented again"
    assert "already exists" in capsys.readouterr().out
    # and the pre-existing (empty) mask is left exactly as it was
    assert _n_objects(out_dir / names[0]) == 0
    assert _n_objects(out_dir / names[1]) == 2


def test_a_batch_whose_fields_are_all_done_is_skipped_and_the_next_one_runs(
        tmp_path, fake_cellpose):
    """_check_masks can empty a batch completely; the loop must move on to the
    next batch rather than hand Cellpose a zero-length array."""
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=4)
    out_dir = src / "nucleus_mask_stack"
    out_dir.mkdir(parents=True)
    for name in names[:2]:
        np.save(out_dir / name, np.zeros((32, 32), dtype=np.uint16))

    O.generate_cellpose_masks(str(src), _settings(src, batch_size=2), "nucleus")

    model = fake_cellpose["model"]
    assert [len(imgs) for imgs in model.eval_inputs] == [2], \
        "only the second batch may reach the model"
    assert _mask_files(src) == sorted(names)
    assert _n_objects(out_dir / names[0]) == 0, "the finished field is untouched"
    assert _n_objects(out_dir / names[2]) == 2
    rows = _counts(tmp_path / "measurements" / "measurements.db")
    assert sorted(r[0] for r in rows) == sorted(names[2:])


def test_btrack_n_jobs_never_drops_below_one_on_a_small_machine(
        tmp_path, fake_cellpose, fake_timelapse, monkeypatch):
    """``os.cpu_count() - 2`` is -1 on a single-core box; btrack would be asked
    for a negative worker count."""
    monkeypatch.setattr(os, "cpu_count", lambda: 1)

    src = tmp_path / "stack"
    _write_npz(src, n=3)
    O.generate_cellpose_masks(
        str(src), _tl(src, timelapse_mode="btrack"), "nucleus")

    assert fake_timelapse["btrack"][0]["n_jobs"] == 1


def test_a_batch_with_no_objects_reports_zero_rather_than_dividing_by_nothing(
        tmp_path, fake_cellpose, capsys):
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=2)
    fake_cellpose["blobs"] = []          # every returned mask is empty

    O.generate_cellpose_masks(str(src), _settings(src), "nucleus")

    assert "Found 0.0 nucleus/FOV. average size: 0.000 px2" in capsys.readouterr().out
    assert _mask_files(src) == sorted(names)
    assert _n_objects(src / "nucleus_mask_stack" / names[0]) == 0
    assert _counts(tmp_path / "measurements" / "measurements.db") == sorted(
        (n, "nucleus_before_filtration", 0) for n in names)


def test_save_false_segments_but_writes_no_masks(tmp_path, fake_cellpose):
    src = tmp_path / "stack"
    _write_npz(src, n=2)

    O.generate_cellpose_masks(str(src), _settings(src, save=False), "nucleus")

    assert _mask_files(src) == []
    # the counts still went to the database
    assert len(_counts(tmp_path / "measurements" / "measurements.db")) == 2


def test_plot_true_renders_the_batch_and_keeps_every_field(
        tmp_path, fake_cellpose, monkeypatch):
    import spacr.plot as PL
    calls = []
    monkeypatch.setattr(PL, "plot_cellpose4_output",
                        lambda b, m, f, **kw: calls.append((list(b), list(m), list(f), kw)))

    src = tmp_path / "stack"
    _write_npz(src, n=2)
    O.generate_cellpose_masks(str(src), _settings(src, plot=True, batch_size=2),
                              "nucleus")

    assert len(calls) == 1
    batch, masks, flows, kw = calls[0]
    assert len(batch) == 2 and len(masks) == 2 and len(flows) == 2
    assert kw["nr"] == 2
    assert kw["cmap"] == "inferno"
    # the flow handed to the plot is the whole per-image flow image
    assert np.asarray(flows[0]).shape == (32, 32, 3)


# --------------------------------------------------------------------------- #
#  filter / merge
# --------------------------------------------------------------------------- #

def test_filter_records_after_filtration_counts_and_saves_the_filtered_stack(
        tmp_path, fake_cellpose):
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=2)
    # one blob of 36 px and one of 4 px; a size filter between them keeps one.
    fake_cellpose["blobs"] = [(slice(2, 8), slice(2, 8)),
                              (slice(20, 22), slice(20, 22))]

    settings = _settings(src, filter=True, nucleus_diameter=12)
    O.generate_cellpose_masks(str(src), settings, "nucleus")

    rows = _counts(tmp_path / "measurements" / "measurements.db")
    kinds = {r[1] for r in rows}
    assert kinds == {"nucleus_before_filtration", "nucleus_after_filtration"}
    assert _mask_files(src) == sorted(names)

    # settings._get_object_settings hard-codes filter_size, filter_intensity
    # and remove_border_objects to False for every object type and reads none
    # of the <obj>_remove_border_objects / size settings, so `filter=True`
    # cannot drop anything here — before and after counts are equal by
    # construction. Asserting that keeps the no-op honest and turns into a
    # real failure the day those flags start being read.
    assert sorted(r for r in rows if r[1] == "nucleus_after_filtration") == sorted(
        (n, "nucleus_after_filtration", 2) for n in names)
    assert _n_objects(src / "nucleus_mask_stack" / names[0]) == 2


def test_filter_applies_the_merge_and_the_saved_masks_are_the_merged_ones(
        tmp_path, fake_cellpose):
    """The one thing `filter=True` can really change today: it forwards
    object_settings['merge'], so merge_pathogens has to survive the round trip
    into the database counts and onto disk."""
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=2, c=3)
    fake_cellpose["blobs"] = [(slice(2, 14), slice(2, 10)),
                              (slice(2, 14), slice(10, 18))]
    settings = _settings(src, pathogen_channel=2, merge_pathogens=True,
                         filter=True)

    O.generate_cellpose_masks(str(src), settings, "pathogen")

    rows = _counts(tmp_path / "measurements" / "measurements.db")
    before = sorted(r for r in rows if r[1] == "pathogen_before_filtration")
    after = sorted(r for r in rows if r[1] == "pathogen_after_filtration")
    assert [r[2] for r in before] == [2, 2], "Cellpose returned two objects"
    assert [r[2] for r in after] == [1, 1], "the merge must reach the count"
    for name in names:
        assert _n_objects(src / "pathogen_mask_stack" / name) == 1


def test_filter_processes_every_field_even_when_fields_outnumber_image_rows(
        tmp_path, fake_cellpose):
    """``_filter_cp_masks`` iterates ``zip(masks, flows[0], batch)``, so the
    per-image flow list has to arrive nested one deep. Passing it bare made
    ``flows[0]`` the first image's flow array and the zip ran over its *rows*:
    a batch with more fields than the images are tall lost the trailing masks
    entirely."""
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=6, h=4, w=24)
    fake_cellpose["blobs"] = [(slice(1, 3), slice(1, 5))]

    O.generate_cellpose_masks(str(src), _settings(src, filter=True), "nucleus")

    assert _mask_files(src) == sorted(names), "no field may be silently dropped"
    rows = [r for r in _counts(tmp_path / "measurements" / "measurements.db")
            if r[1] == "nucleus_after_filtration"]
    assert len(rows) == 6


def test_the_filter_plot_gets_a_flow_image_not_one_pixel_row(
        tmp_path, fake_cellpose, monkeypatch):
    """Same nesting bug seen from the other side: ``plot_masks`` was handed a
    single row of the first image's flow field."""
    import spacr.plot as PL
    seen = []
    monkeypatch.setattr(PL, "plot_masks",
                        lambda batch, masks, flows, **kw: seen.append(np.asarray(flows)))
    monkeypatch.setattr(PL, "plot_cellpose4_output", lambda *a, **k: None)

    src = tmp_path / "stack"
    _write_npz(src, n=2)
    O.generate_cellpose_masks(str(src), _settings(src, filter=True, plot=True),
                              "nucleus")

    assert seen, "the filter plot path must have run"
    for flow in seen:
        assert flow.shape == (32, 32, 3), f"got {flow.shape}, a row of the flow image"
        assert np.allclose(flow, 0.25)


def test_merge_is_not_thrown_away_when_filter_is_off(tmp_path, fake_cellpose):
    """`merge_pathogens` routed through ``_filter_cp_masks(merge=True)``, and
    the result was then rebound to the raw Cellpose masks by an unconditional
    ``else`` — so the setting did nothing."""
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=2, c=3)
    # two touching rectangles: merge_touching_objects(threshold=0.66) fuses them
    fake_cellpose["blobs"] = [(slice(2, 14), slice(2, 10)),
                              (slice(2, 14), slice(10, 18))]
    settings = _settings(src, pathogen_channel=2, merge_pathogens=True,
                         filter=False)

    O.generate_cellpose_masks(str(src), settings, "pathogen")

    folder = src / "pathogen_mask_stack"
    for name in names:
        assert _n_objects(folder / name) == 1, "the two touching objects must be merged"
        # ... and the merged object covers both rectangles
        assert int((np.load(folder / name) > 0).sum()) == 12 * 16


def test_merge_off_and_filter_off_saves_the_raw_masks(tmp_path, fake_cellpose):
    """The complement of the test above: with merge off the same two touching
    objects must survive as two."""
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=2, c=3)
    fake_cellpose["blobs"] = [(slice(2, 14), slice(2, 10)),
                              (slice(2, 14), slice(10, 18))]
    settings = _settings(src, pathogen_channel=2, merge_pathogens=False,
                         filter=False)

    O.generate_cellpose_masks(str(src), settings, "pathogen")

    assert _n_objects(src / "pathogen_mask_stack" / names[0]) == 2


# --------------------------------------------------------------------------- #
#  timelapse
# --------------------------------------------------------------------------- #

def _tl(src, **over):
    s = _settings(src, timelapse=True, batch_size=50,
                  timelapse_objects=["nucleus"], timelapse_mode="trackpy",
                  timelapse_displacement=12, timelapse_memory=2,
                  timelapse_remove_transient=False, timelapse_frame_limits=[0, 3])
    s.update(over)
    return s


def test_timelapse_writes_a_movie_tracks_and_records_timelapse_counts(
        tmp_path, fake_cellpose, fake_timelapse):
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=3)

    O.generate_cellpose_masks(str(src), _tl(src), "nucleus")

    assert len(fake_timelapse["movie"]) == 1
    movie = fake_timelapse["movie"][0]
    assert movie["n_frames"] == 3
    assert movie["save_path"].endswith("movies/timelapse_nucleus_batch1.mp4")
    assert movie["fps"] == 2

    assert len(fake_timelapse["trackpy"]) == 1
    kw = fake_timelapse["trackpy"][0]
    assert kw["object_type"] == "nucleus"
    assert kw["timelapse_displacement"] == 12
    assert kw["timelapse_memory"] == 2
    assert kw["track_by_iou"] is False
    assert kw["mode"] == "trackpy"
    assert kw["batch_filenames"] == names

    rows = _counts(tmp_path / "measurements" / "measurements.db")
    assert {r[1] for r in rows} == {"nucleus_timelapse"}
    assert _mask_files(src) == sorted(names)


def test_timelapse_iou_mode_sets_track_by_iou(tmp_path, fake_cellpose,
                                              fake_timelapse):
    src = tmp_path / "stack"
    _write_npz(src, n=3)

    O.generate_cellpose_masks(str(src), _tl(src, timelapse_mode="iou"), "nucleus")

    assert len(fake_timelapse["trackpy"]) == 1
    assert fake_timelapse["trackpy"][0]["track_by_iou"] is True


def test_timelapse_btrack_uses_the_displacement_as_its_radius(
        tmp_path, fake_cellpose, fake_timelapse):
    src = tmp_path / "stack"
    _write_npz(src, n=3)

    O.generate_cellpose_masks(
        str(src), _tl(src, timelapse_mode="btrack", timelapse_displacement=17),
        "nucleus")

    assert len(fake_timelapse["btrack"]) == 1
    kw = fake_timelapse["btrack"][0]
    assert kw["radius"] == 17
    assert kw["n_jobs"] >= 1
    assert len(kw["masks_3D"]) == 3


def test_timelapse_btrack_falls_back_to_a_radius_of_100(
        tmp_path, fake_cellpose, fake_timelapse):
    src = tmp_path / "stack"
    _write_npz(src, n=3)

    O.generate_cellpose_masks(
        str(src), _tl(src, timelapse_mode="btrack", timelapse_displacement=None),
        "nucleus")

    assert fake_timelapse["btrack"][0]["radius"] == 100


def test_timelapse_leaves_untracked_object_types_as_a_plain_stack(
        tmp_path, fake_cellpose, fake_timelapse):
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=3)

    # 'nucleus' is segmented but only 'cell' is tracked
    O.generate_cellpose_masks(str(src), _tl(src, timelapse_objects=["cell"]),
                              "nucleus")

    assert fake_timelapse["trackpy"] == []
    assert _mask_files(src) == sorted(names)
    assert _n_objects(src / "nucleus_mask_stack" / names[0]) == 2


def test_timelapse_objects_outside_the_trackable_set_abort_the_run(
        tmp_path, fake_cellpose, fake_timelapse, capsys):
    src = tmp_path / "stack"
    _write_npz(src, n=3)

    assert O.generate_cellpose_masks(
        str(src), _tl(src, timelapse_objects=["organelle"]), "nucleus") is None

    assert "must be a subset of" in capsys.readouterr().out
    assert _mask_files(src) == []
    assert fake_cellpose["model"].eval_kwargs == [], "nothing may be segmented"


def test_timelapse_recuts_the_batch_to_the_frame_limits(
        tmp_path, fake_cellpose, fake_timelapse, capsys):
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=6)

    O.generate_cellpose_masks(
        str(src), _tl(src, batch_size=4, timelapse_frame_limits=[1, 4]), "nucleus")

    out = capsys.readouterr().out
    assert "Changed batch_size:4 to 6" in out
    assert "New batch_size: 3" in out
    # frames 1..3 only
    assert _mask_files(src) == sorted(names[1:4])
    assert fake_timelapse["movie"][0]["filenames"] == names[1:4]


def test_the_motility_hook_runs_when_both_flags_are_set(
        tmp_path, fake_cellpose, fake_timelapse):
    src = tmp_path / "stack"
    _write_npz(src, n=3)
    settings = _tl(src, motility_analysis=True)

    O.generate_cellpose_masks(str(src), settings, "nucleus")

    assert len(fake_timelapse["motility"]) == 1
    assert fake_timelapse["motility"][0] is settings


def test_timelapse_plot_draws_one_frame_at_a_time(
        tmp_path, fake_cellpose, fake_timelapse, monkeypatch):
    import spacr.plot as PL
    calls = []
    monkeypatch.setattr(PL, "plot_cellpose4_output",
                        lambda b, m, f, **kw: calls.append(kw))

    src = tmp_path / "stack"
    _write_npz(src, n=3)
    O.generate_cellpose_masks(str(src), _tl(src, plot=True), "nucleus")

    assert len(calls) == 1
    assert calls[0]["nr"] == 1
    assert calls[0]["print_object_number"] is True


# --------------------------------------------------------------------------- #
#  segmentation QC is wired into this generator too
# --------------------------------------------------------------------------- #

def test_seg_qc_scores_the_masks_this_generator_just_wrote(
        tmp_path, fake_cellpose, monkeypatch):
    import spacr.seg_qc as Q
    seen = []

    def _qc(mask_folder, object_type=None, dst=None, mode=None, thresholds=None,
            verbose=True):
        seen.append({"mask_folder": mask_folder, "object_type": object_type,
                     "dst": dst, "mode": mode})
        return {"mode": mode, "flags": {"plate1_A01_1.npy": ["tiny"]}}

    monkeypatch.setattr(Q, "run_segmentation_qc", _qc)

    src = tmp_path / "stack"
    _write_npz(src, n=2)
    settings = _settings(src, seg_qc="flag")

    O.generate_cellpose_masks(str(src), settings, "nucleus")

    assert len(seen) == 1
    assert seen[0]["mask_folder"] == str(src / "nucleus_mask_stack")
    assert seen[0]["object_type"] == "nucleus"
    assert seen[0]["dst"] == str(tmp_path)
    assert seen[0]["mode"] == "flag"
    # 'flag' mode records the per-field flags back into settings
    assert settings["seg_qc_flags"]["nucleus"] == {"plate1_A01_1.npy": ["tiny"]}


def test_a_qc_crash_never_costs_the_run_its_masks(tmp_path, fake_cellpose,
                                                  monkeypatch, capsys):
    import spacr.seg_qc as Q

    def _boom(*a, **k):
        raise RuntimeError("scorecard exploded")

    monkeypatch.setattr(Q, "run_segmentation_qc", _boom)

    src = tmp_path / "stack"
    _, names = _write_npz(src, n=2)

    O.generate_cellpose_masks(str(src), _settings(src, seg_qc="report"), "nucleus")

    assert "Segmentation QC skipped for nucleus" in capsys.readouterr().out
    assert _mask_files(src) == sorted(names), "the masks must survive a QC bug"


def test_run_seg_qc_returns_none_and_touches_nothing_when_off(tmp_path,
                                                              monkeypatch):
    import spacr.seg_qc as Q
    called = []
    monkeypatch.setattr(Q, "run_segmentation_qc",
                        lambda *a, **k: called.append(1))

    settings = {"seg_qc": "off"}
    assert O._run_seg_qc(str(tmp_path), settings, "cell") is None
    assert called == []
    assert "seg_qc_flags" not in settings


def test_run_seg_qc_report_mode_does_not_record_flags(tmp_path, monkeypatch):
    import spacr.seg_qc as Q
    monkeypatch.setattr(Q, "run_segmentation_qc",
                        lambda *a, **k: {"mode": "report", "flags": {"a": ["x"]}})

    settings = {"seg_qc": "report"}
    result = O._run_seg_qc(str(tmp_path), settings, "cell")
    assert result["mode"] == "report"
    assert "seg_qc_flags" not in settings, "only 'flag' mode mutates settings"


# --------------------------------------------------------------------------- #
#  GPU selection
# --------------------------------------------------------------------------- #

def test_a_cuda_box_builds_a_gpu_model_on_device_zero(
        tmp_path, fake_cellpose, monkeypatch, capsys):
    """Only the device selection is exercised — no kernel ever runs, because
    Cellpose is the fake."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    src = tmp_path / "stack"
    _write_npz(src, n=2)
    O.generate_cellpose_masks(str(src), _settings(src), "nucleus")

    model = fake_cellpose["model"]
    assert model.gpu is True
    assert str(model.device) == "cuda:0"
    assert "Torch CUDA is not available" not in capsys.readouterr().out


def test_a_run_without_a_nucleus_channel_leaves_the_cellpose_alias_unset(
        tmp_path, fake_cellpose):
    """The `cellpose_<obj>_channel` aliases are only back-filled from the plain
    channel settings; a disabled object must not gain one."""
    src = tmp_path / "stack"
    _write_npz(src, n=2, c=1)
    settings = _settings(src, nucleus_channel=None, cell_channel=None,
                         pathogen_channel=0)

    O.generate_cellpose_masks(str(src), settings, "pathogen")

    assert settings.get("cellpose_nucleus_channel") is None
    assert settings.get("cellpose_cell_channel") is None
    assert settings["cellpose_pathogen_channel"] == 0
    assert fake_cellpose["model"].eval_kwargs[0]["channels"] == [0]


# --------------------------------------------------------------------------- #
#  timelapse_frame_limits — the shapes that must NOT cut the batch
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("limits", [[5], None, (0, 2)])
def test_frame_limits_that_cannot_name_a_range_leave_the_batch_whole(
        tmp_path, fake_cellpose, fake_timelapse, limits):
    """Only a list of two or more entries cuts the stack. The default is `[5]`
    — a single entry — and anything that is not a list at all is ignored."""
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=4)

    O.generate_cellpose_masks(
        str(src), _tl(src, batch_size=2, timelapse_frame_limits=limits), "nucleus")

    assert _mask_files(src) == sorted(names), "all four frames must survive"
    assert fake_timelapse["movie"][0]["n_frames"] == 4


def test_a_batch_size_that_already_matches_the_stack_is_left_alone(
        tmp_path, fake_cellpose, fake_timelapse, capsys):
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=4)

    O.generate_cellpose_masks(
        str(src), _tl(src, batch_size=4, timelapse_frame_limits=[1, 3]), "nucleus")

    out = capsys.readouterr().out
    assert "Changed batch_size" not in out
    assert "Cut batch at indecies" not in out
    assert _mask_files(src) == sorted(names)


# --------------------------------------------------------------------------- #
#  generate_cellpose_masks_sam — branches its own suite does not reach
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_sam_model(monkeypatch):
    """Stand-in for the SAM checkpoint used by generate_cellpose_masks_sam."""
    holder = {"model": None}

    class _M:
        def __init__(self, gpu=None, pretrained_model=None, device=None, **kw):
            self.gpu = gpu
            self.pretrained_model = pretrained_model
            self.device = device
            self.eval_kwargs = []
            holder["model"] = self

        def eval(self, x=None, **kwargs):
            self.eval_kwargs.append(kwargs)
            masks, flows = [], []
            for im in x:
                h, w = np.asarray(im).shape[:2]
                m = np.zeros((h, w), dtype=np.uint16)
                m[2:8, 2:8] = 1
                masks.append(m)
                flows.append((np.zeros((h, w, 3), np.float32), None, None, None))
            return masks, flows, None

    monkeypatch.setattr(O, "cp_models", types.SimpleNamespace(CellposeModel=_M))
    return holder


def test_the_sam_generator_also_builds_a_gpu_model_on_a_cuda_box(
        tmp_path, fake_sam_model, monkeypatch, capsys):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    src = tmp_path / "stack"
    _write_npz(src, n=2)
    O.generate_cellpose_masks_sam(str(src), _settings(src), "cell")

    model = fake_sam_model["model"]
    assert model.gpu is True
    assert str(model.device) == "cuda:0"
    assert model.pretrained_model == "cpsam"
    assert "Torch CUDA is not available" not in capsys.readouterr().out


def test_the_sam_generator_without_a_cell_channel_leaves_that_alias_unset(
        tmp_path, fake_sam_model):
    src = tmp_path / "stack"
    _write_npz(src, n=2, c=1)
    settings = _settings(src, cell_channel=None, nucleus_channel=0,
                         pathogen_channel=None)

    O.generate_cellpose_masks_sam(str(src), settings, "nucleus")

    assert settings.get("cellpose_cell_channel") is None
    assert settings["cellpose_nucleus_channel"] == 0
    assert _mask_files(src, "nucleus") == ["plate1_A01_1.npy", "plate1_A01_2.npy"]


@pytest.mark.parametrize("limits", [[5], None, (0, 2)])
def test_the_sam_generator_keeps_the_batch_whole_for_unusable_frame_limits(
        tmp_path, fake_sam_model, fake_timelapse, limits):
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=4)
    settings = _settings(
        src, timelapse=True, timelapse_objects=["cell"], timelapse_mode="trackpy",
        timelapse_displacement=10, timelapse_memory=3,
        timelapse_remove_transient=False, timelapse_frame_limits=limits,
        batch_size=2, cell_min_object_area=0, nucleus_min_object_area=0,
    )

    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    assert _mask_files(src, "cell") == sorted(names)
    assert fake_timelapse["movie"][0]["n_frames"] == 4


def test_sam_generator_routes_to_ultrack_with_its_solver_parameters(
        tmp_path, fake_sam_model, fake_timelapse):
    """The ultrack branch is the only tracker whose parameters are read
    straight out of settings, so a typo there is invisible without this."""
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=3)
    settings = _settings(
        src, timelapse=True, timelapse_objects=["cell"], timelapse_mode="ultrack",
        timelapse_displacement=None, timelapse_memory=3,
        timelapse_remove_transient=True, timelapse_frame_limits=[0, 3],
        ultrack_max_distance=33.0, ultrack_division_weight=-0.25,
        ultrack_contour_sigma=1.5, ultrack_n_workers=2,
        cell_min_object_area=0, nucleus_min_object_area=0,
    )

    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    assert len(fake_timelapse["ultrack"]) == 1
    kw = fake_timelapse["ultrack"][0]
    assert kw["object_type"] == "cell"
    assert kw["max_distance"] == 33.0
    assert kw["division_weight"] == -0.25
    assert kw["contour_sigma"] == 1.5
    assert kw["n_workers"] == 2
    assert kw["timelapse_remove_transient"] is True
    assert kw["batch_filenames"] == names
    # ultrack gets the raw intensities as well as the labels
    assert len(kw["masks"]) == 3
    assert np.asarray(kw["images"]).shape == (3, 32, 32, 2)
    assert _mask_files(src, "cell") == sorted(names)


# --------------------------------------------------------------------------- #
#  the IPython import fallback at module scope
# --------------------------------------------------------------------------- #

def test_module_imports_and_display_is_a_no_op_when_ipython_is_unavailable(
        monkeypatch):
    """IPython can be mid-init in another thread; importing spacr.object must
    never block or fail on it, and the fallback display() must swallow calls."""
    import importlib

    real = sys.modules.get("IPython.display")

    # A None entry in sys.modules makes `from IPython.display import display`
    # raise ImportError, which is exactly the mid-init condition the fallback
    # is there for.
    monkeypatch.setitem(sys.modules, "IPython.display", None)
    try:
        importlib.reload(O)
        # the fallback is a plain function, not IPython's display
        assert O.display.__module__ == "spacr.object"
        assert O.display("anything", extra=1) is None
    finally:
        if real is not None:
            sys.modules["IPython.display"] = real
        else:
            sys.modules.pop("IPython.display", None)
        importlib.reload(O)

    # and the module is intact afterwards
    assert callable(O.generate_cellpose_masks)
