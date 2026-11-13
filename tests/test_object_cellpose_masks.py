"""Table-driven tests of the chosen-model mask generator ``spacr.object.generate_cellpose_masks``.

Two general tests, each walking a case table with pytest's ``subtests``
fixture:

* still fields: model construction (CPU / CUDA, stock / user checkpoint, the
  refusals that stop a run before any model loads), the eval call, the
  channel remap onto the compacted stack (raw, recorded, stale, text, one
  plane), batching, resume, save, plot, merge / filter and segmentation QC;
* time-lapse batches: the batch re-cut to the frame limits, the movie, every
  tracker the generator can hand a batch to and the steps run after tracking.

Only Cellpose is faked: ``spacr.utils.cp_models`` is swapped for a recording
``CellposeModel`` whose ``eval`` declares the installed 4.0.7 signature
(``channel_axis=MISSING_CHANNEL_AXIS``, no ``**kwargs``) and passes every
call through :func:`tests.conftest.check_cellpose_eval_call`, so
:func:`spacr.utils._choose_model` still decides which checkpoint loads.  The
SQLite counts, the ``.npy`` masks, the channel remap and the merge/filter
helpers are real.  Each case gets its own folder and its own monkeypatch
context.
"""
from __future__ import annotations

import os
import re
import sqlite3
import types
from pathlib import Path

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import spacr.object as O  # noqa: E402
import spacr.utils as U  # noqa: E402
from tests.cellpose_api_contract import (  # noqa: E402
    DEPRECATED_EVAL_ARGUMENTS,
    configured_eval_arguments,
    emulate_pretrained_model,
    eval_arguments,
    init_arguments,
)
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# doubles
# ---------------------------------------------------------------------------

TWO_BLOBS = [(slice(2, 8), slice(2, 8)), (slice(12, 18), slice(12, 18))]
TOUCHING = [(slice(2, 14), slice(2, 10)), (slice(2, 14), slice(10, 18))]


def _install_cellpose(mp, holder):
    """Swap ``spacr.utils.cp_models`` for a recording 4.0.7 ``CellposeModel``.

    Every returned mask carries ``holder['blobs']`` as labels 1, 2, ...; every
    flow is the 4.0.7 per-image tuple ``(rgb, dP, cellprob, p)``.
    """

    class _M:
        def __init__(self, gpu=False, pretrained_model="cpsam", model_type=None,
                     diam_mean=None, device=None, nchan=None, use_bfloat16=True):
            self.gpu = gpu
            self.pretrained_model = pretrained_model
            self.device = device
            self.init_kwargs = init_arguments(locals())
            self.loaded_model = emulate_pretrained_model(pretrained_model, model_type)
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
                for label, (rs, cs) in enumerate(holder["blobs"], start=1):
                    m[rs, cs] = label
                masks.append(m)
                flows.append((np.full((h, w, 3), 0.25, dtype=np.float32),
                              np.zeros((2, h, w), dtype=np.float32),
                              np.zeros((h, w), dtype=np.float32), None))
            return masks, flows, None

    mp.setattr(U, "cp_models", types.SimpleNamespace(CellposeModel=_M))


def _install_device(mp, rec, cuda=False):
    """Pin the device decision (CPU, or a CUDA box) and record empty_cache()."""
    import torch

    device = torch.device("cuda:0" if cuda else "cpu")
    mp.setattr(torch.cuda, "is_available", lambda: cuda)
    O.accelerator._reset_device_cache()  # is_gpu() resolves for real per case
    mp.setattr(O.accelerator, "torch_device", lambda: device)
    mp.setattr(O.accelerator, "cellpose_kwargs", lambda: {"gpu": cuda, "device": device})
    mp.setattr(torch.cuda, "empty_cache", lambda: rec["empty_cache"].append(1))


def _install_timelapse(mp, rec):
    """Record the movie writer, the trackers and the motility hook."""
    import spacr.timelapse as TL

    def stack(masks):
        return [np.asarray(m, dtype=np.uint16) for m in masks]

    def tracker(name, key):
        def _track(**kw):
            rec[name].append(kw)
            return stack(kw[key])
        return _track

    mp.setattr(TL, "_npz_to_movie", lambda arrays, filenames, save_path, fps=10:
               rec["movie"].append({"n_frames": len(arrays), "filenames": list(filenames),
                                    "save_path": save_path, "fps": fps}))
    mp.setattr(TL, "_btrack_track_cells", tracker("btrack", "masks_3D"))
    mp.setattr(TL, "_trackpy_track_cells", tracker("trackpy", "masks"))
    mp.setattr(TL, "automated_motility_assay", lambda s: rec["motility"].append(s))


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def _names(*idx):
    """The field filenames ``_write_npz`` writes, by 1-based index."""
    return [f"plate1_A01_{i}.npy" for i in idx]


def _write_npz(src, n=3, c=2, h=32, w=32, source_channels=None):
    """Pre-batched ``.npz`` whose channel ``k`` carries a bright band on rows 4k..4k+2."""
    src = Path(src)
    src.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    data = rng.integers(100, 400, size=(n, h, w, c)).astype(np.uint16)
    for k in range(c):
        data[:, 4 * k:4 * k + 3, :, k] = 4000
    extra = ({} if source_channels is None else
             {"source_channels": np.asarray(source_channels, dtype=np.int64)})
    np.savez(src / "batch1.npz", data=data,
             filenames=np.array(_names(*range(1, n + 1)), dtype="<U32"), **extra)
    return data


def _settings(src, **over):
    s = {"src": str(src), "cell_channel": 0, "nucleus_channel": 1,
         "pathogen_channel": None, "magnification": 20, "batch_size": 50,
         "verbose": False, "plot": False, "save": True, "filter": False,
         "timelapse": False, "n_jobs": 1, "seg_qc": "off"}
    s.update(over)
    return s


def _tl(**over):
    """Time-lapse overrides: tracked nuclei, trackpy, frames 0..3."""
    base = dict(timelapse=True, batch_size=50, timelapse_objects=["nucleus"],
                timelapse_mode="trackpy", timelapse_displacement=12,
                timelapse_memory=2, timelapse_remove_transient=False,
                timelapse_frame_limits=[0, 3])
    base.update(over)
    return base


def _counts(db_path):
    """``[(file_name, count_type, object_count), ...]`` from the run database."""
    if not Path(db_path).exists():
        return None
    con = sqlite3.connect(str(db_path))
    try:
        tables = [row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        if "object_counts" not in tables:
            return None
        return sorted(con.execute(
            "SELECT file_name, count_type, object_count FROM object_counts"))
    finally:
        con.close()


def _mask_files(src, object_type):
    folder = Path(src) / f"{object_type}_mask_stack"
    return sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []


def _labels(path):
    """Sorted non-zero labels of a saved mask."""
    return sorted(set(np.unique(np.load(path)).tolist()) - {0})


def _which_channel(plane, image):
    """Stack channel whose content ``plane`` reproduces (best correlation)."""
    flat = np.asarray(plane, dtype=float).ravel()
    scores = [abs(np.corrcoef(flat, image[..., k].ravel().astype(float))[0, 1])
              for k in range(image.shape[-1])]
    return int(np.argmax(scores))


def _case_dir(tmp_path, name):
    d = tmp_path / re.sub(r"[^A-Za-z0-9_]+", "_", name)
    d.mkdir()
    return d


def _run(tmp_path, case_id, capsys, *, object_type="nucleus", npz=None, over=None,
         blobs=TWO_BLOBS, pre_done=(), cuda=False, setup=None):
    """Run the generator for one case and return everything it touched."""
    base = _case_dir(tmp_path, case_id)
    src = base / "stack"
    data = _write_npz(src, **(npz or {}))
    if pre_done:
        (src / f"{object_type}_mask_stack").mkdir(parents=True)
        for name in pre_done:
            np.save(src / f"{object_type}_mask_stack" / name, np.zeros((32, 32), np.uint16))
    settings = _settings(src, **(over or {}))
    holder = {"model": None, "blobs": blobs}
    rec = {k: [] for k in ("empty_cache", "movie", "btrack", "trackpy", "motility",
                           "plot", "plot_masks", "display", "qc", "lineage", "division")}
    r = types.SimpleNamespace(base=base, src=src, data=data, settings=settings,
                              holder=holder, rec=rec, error=None, value="unset")
    capsys.readouterr()
    with pytest.MonkeyPatch.context() as mp:
        _install_cellpose(mp, holder)
        _install_device(mp, rec, cuda=cuda)
        _install_timelapse(mp, rec)
        if setup is not None:
            setup(mp, r)
        try:
            r.value = O.generate_cellpose_masks(str(src), settings, object_type)
        except Exception as exc:  # noqa: BLE001 - the refusal rows assert on it
            r.error = exc
    r.out = capsys.readouterr().out
    r.model = holder["model"]
    r.files = _mask_files(src, object_type)
    r.counts = _counts(base / "measurements" / "measurements.db")
    return r


# --- per-case setups --------------------------------------------------------

def _record_plots(mp, r):
    import spacr.plot as PL
    mp.setattr(PL, "plot_cellpose4_output", lambda b, m, f, **kw: r.rec["plot"].append(
        {"n": (len(b), len(m), len(f)), "flow_shape": np.asarray(f[0]).shape, **kw}))
    mp.setattr(PL, "plot_masks", lambda batch, masks, flows, **kw:
               r.rec["plot_masks"].append(np.asarray(flows)))


def _record_display(mp, r):
    mp.setattr(O, "display", lambda df, *a, **k: r.rec["display"].append(df))


def _record_qc(result=None, crash=False):
    def setup(mp, r):
        import spacr.seg_qc as Q

        def _qc(mask_folder, object_type=None, dst=None, mode=None, thresholds=None,
                verbose=True):
            if crash:
                raise RuntimeError("scorecard exploded")
            r.rec["qc"].append({"mask_folder": mask_folder, "object_type": object_type,
                                "dst": dst, "mode": mode})
            return result
        mp.setattr(Q, "run_segmentation_qc", _qc)
    return setup


def _checkpoint(mp, r):
    ckpt = r.base / "my_pathogens.pth"
    ckpt.write_bytes(b"not really a checkpoint, _choose_model only stats it")
    r.settings["pathogen_model"] = str(ckpt)


def _missing_checkpoint(mp, r):
    r.settings["pathogen_model"] = str(r.base / "gone" / "model.pth")


def _one_cpu(mp, r):
    mp.setattr(os, "cpu_count", lambda: 1)


def _after_tracking(mp, r):
    import spacr.division_analysis as DA
    import spacr.timelapse as TL

    mp.setattr(TL, "_run_lineage_step",
               lambda src, name, obj, mode, settings, *, frame_sources, label_stack,
               images=None: r.rec["lineage"].append(
                   ("lineage", name, obj, mode, len(frame_sources), len(label_stack),
                    None if images is None else len(images))))
    mp.setattr(TL, "_run_event_features_step",
               lambda src, name, obj, stack, images, mode, settings:
               r.rec["lineage"].append(("events", name, obj, mode, len(stack))))
    mp.setattr(DA, "_run_division_step",
               lambda src, name, obj, mode, settings, *, label_stack, images=None:
               r.rec["division"].append((name, obj, mode, len(label_stack), len(images))))


def _seq(*setups):
    def setup(mp, r):
        for s in setups:
            s(mp, r)
    return setup


# ---------------------------------------------------------------------------
# still fields
# ---------------------------------------------------------------------------

def test_generate_cellpose_masks_segments_saves_and_counts_still_fields(
        subtests, tmp_path, capsys):
    """Every still-field route ends in exact masks on disk, exact counts in the
    database and exactly the eval call, model and channel planes it should."""
    from spacr.model_zoo import ModelZooError
    from spacr.zstack import TStackError

    n3 = _names(1, 2, 3)
    before = lambda names, k=2, obj="nucleus": sorted(  # noqa: E731
        (n, f"{obj}_before_filtration", k) for n in names)
    after = lambda names, k=2, obj="nucleus": sorted(  # noqa: E731
        (n, f"{obj}_after_filtration", k) for n in names)
    ok_out = "Torch GPU is not available, using CPU"
    cases = [
        # (id, run kwargs, expectations)
        ("writes one uint16 mask per field and counts it", {},
         {"files": n3, "counts": before(n3), "eval_sizes": [3], "empty_cache": [1],
          "labels": {n: [1, 2] for n in n3}, "areas": {1: 36, 2: 36},
          "model": {"pretrained_model": "cpsam", "gpu": False, "device": "cpu"},
          "out": [ok_out, "Found 2.0 nucleus/FOV. average size: 36.000 px2"]}),
        ("eval gets every nucleus parameter",
         {"over": dict(batch_size=8, nucleus_flow_threshold=0.7,
                       nucleus_cellprob_threshold=-1.5, nucleus_min_area=25)},
         {"eval": {"batch_size": 8, "normalize": False, "channel_axis": -1,
                   "diameter": 60, "flow_threshold": 0.7, "cellprob_threshold": -1.5,
                   "rescale": None, "resample": True, "min_size": 25},
          "input_shape": (32, 32, 1), "planes": [0], "eval_sizes": [3]}),
        ("pathogen uses its own thresholds and does not resample",
         {"object_type": "pathogen", "npz": dict(n=2, c=3),
          "over": dict(pathogen_channel=2, pathogen_flow_threshold=0.3,
                       pathogen_cellprob_threshold=2.0)},
         {"eval": {"flow_threshold": 0.3, "cellprob_threshold": 2.0, "resample": False,
                   "diameter": 20},
          "files": _names(1, 2), "counts": before(_names(1, 2), obj="pathogen")}),
        ("nucleus plane of a compacted stack",
         {"npz": dict(n=2, c=3), "over": dict(nucleus_channel=1, cell_channel=3,
                                              pathogen_channel=5)},
         {"input_shape": (32, 32, 1), "planes": [0]}),
        ("cell then nucleus plane of a compacted stack",
         {"object_type": "cell", "npz": dict(n=2, c=3),
          "over": dict(nucleus_channel=1, cell_channel=3, pathogen_channel=5)},
         {"input_shape": (32, 32, 2), "planes": [1, 0]}),
        ("pathogen plane of a compacted stack",
         {"object_type": "pathogen", "npz": dict(n=2, c=3),
          "over": dict(nucleus_channel=1, cell_channel=3, pathogen_channel=5)},
         {"input_shape": (32, 32, 1), "planes": [2]}),
        ("a stale recorded position is recomputed from the raw channels",
         {"npz": dict(n=2, c=2, source_channels=[1, 0]),
          "over": dict(cellpose_nucleus_channel=1, cellpose_cell_channel=0)},
         {"planes": [0], "aliases": {"nucleus": 0, "cell": 1, "pathogen": None},
          "files": _names(1, 2)}),
        ("a text channel is skipped and the run finishes",
         {"npz": dict(n=2, c=3), "over": dict(organelle_channel="")},
         {"value": None, "files": _names(1, 2), "planes": [0]}),
        ("a one-plane stack is duplicated into two planes",
         {"npz": dict(n=2, c=1), "over": dict(cell_channel=None, nucleus_channel=0)},
         {"input_shape": (32, 32, 2), "duplicated": True, "files": _names(1, 2)}),
        ("no nucleus channel leaves its alias unset",
         {"object_type": "pathogen", "npz": dict(n=2, c=1),
          "over": dict(nucleus_channel=None, cell_channel=None, pathogen_channel=0)},
         {"aliases": {"nucleus": None, "cell": None, "pathogen": 0},
          "files": _names(1, 2)}),
        ("a user-trained checkpoint reaches cellpose",
         {"object_type": "pathogen", "npz": dict(n=2, c=3),
          "over": dict(pathogen_channel=2), "setup": _checkpoint},
         {"model": {"pretrained_model": "my_pathogens.pth"}, "files": _names(1, 2)}),
        ("a CUDA box builds a GPU model on device zero", {"npz": dict(n=2), "cuda": True},
         {"model": {"pretrained_model": "cpsam", "gpu": True, "device": "cuda:0"},
          "not_out": [ok_out], "files": _names(1, 2)}),
        ("every batch of a multi-batch npz is saved",
         {"npz": dict(n=5), "over": dict(batch_size=2)},
         {"eval_sizes": [2, 2, 1], "files": _names(1, 2, 3, 4, 5),
          "counts": before(_names(1, 2, 3, 4, 5))}),
        ("an empty npz writes nothing", {"npz": dict(n=0)},
         {"value": None, "files": [], "eval_sizes": [], "counts": None}),
        ("a field whose mask exists is not segmented again", {"pre_done": _names(1)},
         {"eval_sizes": [2], "files": n3, "labels": {_names(1)[0]: [], _names(2)[0]: [1, 2]},
          "out": ["File plate1_A01_1.npy already exists in the output folder. Skipping..."],
          "counts": before(_names(2, 3))}),
        ("a batch whose fields are all done is skipped",
         {"npz": dict(n=4), "over": dict(batch_size=2), "pre_done": _names(1, 2)},
         {"eval_sizes": [2], "files": _names(1, 2, 3, 4),
          "labels": {_names(1)[0]: [], _names(3)[0]: [1, 2]},
          "counts": before(_names(3, 4))}),
        ("an objectless batch reports zero", {"npz": dict(n=2), "blobs": []},
         {"out": ["Found 0.0 nucleus/FOV. average size: 0.000 px2"], "files": _names(1, 2),
          "labels": {_names(1)[0]: []}, "counts": before(_names(1, 2), k=0)}),
        ("save off segments and counts but writes no mask",
         {"npz": dict(n=2), "over": dict(save=False)},
         {"files": [], "counts": before(_names(1, 2)), "eval_sizes": [2]}),
        ("plot on renders the whole batch with its flow images",
         {"npz": dict(n=2), "over": dict(plot=True, batch_size=2), "setup": _record_plots},
         {"plot": [{"n": (2, 2, 2), "flow_shape": (32, 32, 3), "cmap": "inferno",
                    "figuresize": 10, "nr": 2}],
          "files": _names(1, 2), "out": ["plotting"]}),
        ("filter records after-filtration counts and saves the filtered stack",
         {"npz": dict(n=2), "over": dict(filter=True, nucleus_diameter=12),
          "blobs": [(slice(2, 8), slice(2, 8)), (slice(20, 22), slice(20, 22))]},
         {"counts": sorted(before(_names(1, 2)) + after(_names(1, 2))),
          "files": _names(1, 2), "labels": {_names(1)[0]: [1, 2]}}),
        ("filter applies the merge and saves the merged masks",
         {"object_type": "pathogen", "npz": dict(n=2, c=3), "blobs": TOUCHING,
          "over": dict(pathogen_channel=2, merge_pathogens=True, filter=True)},
         {"counts": sorted(before(_names(1, 2), obj="pathogen")
                           + after(_names(1, 2), k=1, obj="pathogen")),
          "labels": {n: [2] for n in _names(1, 2)}}),
        ("filter keeps every field when fields outnumber image rows",
         {"npz": dict(n=6, h=4, w=24), "over": dict(filter=True),
          "blobs": [(slice(1, 3), slice(1, 5))]},
         {"files": _names(1, 2, 3, 4, 5, 6),
          "counts": sorted(before(_names(1, 2, 3, 4, 5, 6), k=1)
                           + after(_names(1, 2, 3, 4, 5, 6), k=1))}),
        ("the filter plot gets whole flow images",
         {"npz": dict(n=2), "over": dict(filter=True, plot=True), "setup": _record_plots},
         {"plot_masks_flows": 1, "files": _names(1, 2)}),
        ("merge without filter still merges",
         {"object_type": "pathogen", "npz": dict(n=2, c=3), "blobs": TOUCHING,
          "over": dict(pathogen_channel=2, merge_pathogens=True, filter=False)},
         {"labels": {n: [2] for n in _names(1, 2)}, "areas": {2: 12 * 16},
          "counts": before(_names(1, 2), obj="pathogen")}),
        ("merge off and filter off saves the raw masks",
         {"object_type": "pathogen", "npz": dict(n=2, c=3), "blobs": TOUCHING,
          "over": dict(pathogen_channel=2, merge_pathogens=False, filter=False)},
         {"labels": {n: [1, 2] for n in _names(1, 2)}, "areas": {1: 96, 2: 96}}),
        ("verbose prints the channel map and shows the settings table",
         {"npz": dict(n=2), "over": dict(verbose=True), "setup": _record_display},
         {"out": ["{'nucleus': [0], 'cell': [1, 0]}", "'diameter': 60"],
          "display": True, "files": _names(1, 2)}),
        ("seg_qc flag scores the written masks and records the flags",
         {"npz": dict(n=2), "over": dict(seg_qc="flag"),
          "setup": _record_qc({"mode": "flag", "flags": {"plate1_A01_1.npy": ["tiny"]}})},
         {"qc": "flag", "qc_flags": {"plate1_A01_1.npy": ["tiny"]}, "files": _names(1, 2)}),
        ("a QC crash never costs the run its masks",
         {"npz": dict(n=2), "over": dict(seg_qc="report"), "setup": _record_qc(crash=True)},
         {"out": ["Segmentation QC skipped for nucleus"], "files": _names(1, 2)}),
        # refusals: nothing is loaded and nothing is written
        ("a checkpoint path that is not there stops the run",
         {"object_type": "pathogen", "npz": dict(n=2, c=3),
          "over": dict(pathogen_channel=2), "setup": _missing_checkpoint},
         {"raises": (ModelZooError, ["gone", "stock weights were not substituted"])}),
        ("an object type with no channel stops before any model loads",
         {"npz": dict(n=2), "object_type": "pathogen", "over": dict(pathogen_channel=None)},
         {"raises": (ValueError, ["Error: No channels were specified for object_type "
                                  "'pathogen'. Check your settings."])}),
        ("t_stack is refused by this generator",
         {"npz": dict(n=2), "over": dict(t_stack=True, t_axis_order="tzyxc")},
         {"raises": (TStackError, ["t_stack is on but object.generate_cellpose_masks "
                                   "does not implement the 4-D path",
                                   "Only object.generate_cellpose_masks_sam reads t_stack"])}),
    ]
    for case_id, kw, expect in cases:
        with subtests.test(msg=case_id):
            object_type = kw.get("object_type", "nucleus")
            r = _run(tmp_path, case_id, capsys, **kw)
            if "raises" in expect:
                exc_type, fragments = expect["raises"]
                assert isinstance(r.error, exc_type), repr(r.error)
                for fragment in fragments:
                    assert fragment in str(r.error)
                assert r.model is None, "no model may be loaded before the refusal"
                assert r.files == []
                continue
            assert r.error is None, repr(r.error)
            assert r.value is expect.get("value", None)
            model = r.model
            for key in ("files", "counts"):
                if key in expect:
                    assert getattr(r, key) == expect[key], key
            if "eval_sizes" in expect:
                assert [len(x) for x in model.eval_inputs] == expect["eval_sizes"]
            if "empty_cache" in expect:
                assert r.rec["empty_cache"] == expect["empty_cache"]
            for configured in model.eval_configured:
                assert not set(configured) & set(DEPRECATED_EVAL_ARGUMENTS)
                assert "channels" not in configured
            for attr, value in expect.get("model", {}).items():
                actual = getattr(model, attr)
                if attr == "pretrained_model" and value.endswith(".pth"):
                    assert actual == str(r.base / value)
                elif attr == "device":
                    assert str(actual) == value
                else:
                    assert actual == value
            for key, value in expect.get("eval", {}).items():
                assert model.eval_kwargs[0][key] == value, key
            if "input_shape" in expect or "planes" in expect or "duplicated" in expect:
                handed = model.eval_inputs[0][0]
                if "input_shape" in expect:
                    assert handed.shape == expect["input_shape"]
                assert handed.dtype == np.float32
                assert handed.min() >= 0.0 and handed.max() == pytest.approx(1.0)
                for position, plane in enumerate(expect.get("planes", [])):
                    assert _which_channel(handed[..., position], r.data[0]) == plane
                if expect.get("duplicated"):
                    np.testing.assert_array_equal(handed[..., 0], handed[..., 1])
            for role, value in expect.get("aliases", {}).items():
                assert r.settings.get(f"cellpose_{role}_channel") == value, role
            folder = r.src / f"{object_type}_mask_stack"
            for name, labels in expect.get("labels", {}).items():
                assert _labels(folder / name) == labels, name
            for label, area in expect.get("areas", {}).items():
                for name in r.files:
                    arr = np.load(folder / name)
                    assert arr.dtype == np.uint16 and arr.shape == (32, 32)
                    assert int((arr == label).sum()) == area
            for text in expect.get("out", []):
                assert text in r.out, text
            for text in expect.get("not_out", []):
                assert text not in r.out, text
            if "plot" in expect:
                assert r.rec["plot"] == expect["plot"]
            if "plot_masks_flows" in expect:
                assert len(r.rec["plot_masks"]) == expect["plot_masks_flows"]
                for flow in r.rec["plot_masks"]:
                    assert flow.shape == (32, 32, 3)
                    assert np.allclose(flow, 0.25)
            if expect.get("display"):
                assert len(r.rec["display"]) == 1
                df = r.rec["display"][0]
                assert list(df.columns) == ["setting_key", "setting_value"]
                assert {"src", "batch_size", "nucleus_channel"} <= set(df["setting_key"])
                assert all(isinstance(v, str) for v in df["setting_value"])
                assert df.set_index("setting_key").loc["src", "setting_value"] == str(r.src)
            if "qc" in expect:
                assert r.rec["qc"] == [{"mask_folder": str(folder), "object_type": "nucleus",
                                        "dst": str(r.base), "mode": expect["qc"]}]
                assert r.settings["seg_qc_flags"] == {"nucleus": expect["qc_flags"]}


# ---------------------------------------------------------------------------
# time-lapse
# ---------------------------------------------------------------------------

def test_generate_cellpose_masks_hands_timelapse_batches_to_the_chosen_tracker(
        subtests, tmp_path, capsys):
    """A time-lapse batch is re-cut to its frame limits, written as a movie and
    tracked by exactly the tracker the settings name, with exact arguments."""
    n3 = _names(1, 2, 3)
    movie3 = [{"n_frames": 3, "filenames": n3, "fps": 2}]
    cases = [
        # (id, run kwargs, expectations)
        ("trackpy tracks, writes a movie and counts per frame", {"over": _tl()},
         {"movie": movie3, "files": n3,
          "counts": sorted((n, "nucleus_timelapse", 2) for n in n3),
          "trackpy": {"object_type": "nucleus", "name": "batch1",
                      "timelapse_displacement": 12, "timelapse_memory": 2,
                      "timelapse_remove_transient": False, "track_by_iou": False,
                      "mode": "trackpy", "batch_filenames": n3, "plot": False,
                      "save": True, "n_masks": 3},
          "btrack_calls": 0}),
        ("iou mode is trackpy tracking by overlap", {"over": _tl(timelapse_mode="iou")},
         {"trackpy": {"track_by_iou": True, "mode": "iou"}, "btrack_calls": 0}),
        ("btrack uses the displacement as its radius",
         {"over": _tl(timelapse_mode="btrack", timelapse_displacement=17)},
         {"btrack": {"radius": 17, "n_jobs": max(1, os.cpu_count() - 2), "n_masks": 3,
                     "mode": "btrack", "batch_filenames": n3, "batch_list": None,
                     "optimizer_time_limit_s": 120, "optimizer_mip_gap": 0.01,
                     "run_optimization": True, "max_objects_for_optimization": 20000},
          "trackpy_calls": 0, "files": n3}),
        ("btrack without a displacement uses a radius of 100",
         {"over": _tl(timelapse_mode="btrack", timelapse_displacement=None)},
         {"btrack": {"radius": 100}}),
        ("btrack never asks for fewer than one worker",
         {"over": _tl(timelapse_mode="btrack"), "setup": _one_cpu},
         {"btrack": {"n_jobs": 1}}),
        ("an untracked object type stays a plain stack",
         {"over": _tl(timelapse_objects=["cell"])},
         {"trackpy_calls": 0, "btrack_calls": 0, "files": n3,
          "labels": {n: [1, 2] for n in n3}, "movie": movie3}),
        ("objects outside the trackable set abort before segmenting",
         {"over": _tl(timelapse_objects=["organelle"])},
         {"files": [], "eval_sizes": [], "movie": [],
          "out": ["timelapse_objects ['organelle'] must be a subset of "
                  "['cell', 'nucleus', 'pathogen']"]}),
        ("the batch is re-cut to the frame limits",
         {"npz": dict(n=6), "over": _tl(batch_size=4, timelapse_frame_limits=[1, 4])},
         {"files": _names(2, 3, 4),
          "movie": [{"n_frames": 3, "filenames": _names(2, 3, 4), "fps": 2}],
          "out": ["Changed batch_size:4 to 6, data length:6",
                  "Cut batch at indecies: [1, 4], New batch_size: 3"],
          "timelapse_batch_size": 6}),
        ("a batch size that matches the stack is left alone",
         {"npz": dict(n=4), "over": _tl(batch_size=4, timelapse_frame_limits=[1, 3])},
         {"files": _names(1, 2, 3, 4), "not_out": ["Changed batch_size", "Cut batch at"]}),
        ("a one-entry frame limit leaves the batch whole",
         {"npz": dict(n=4), "over": _tl(batch_size=2, timelapse_frame_limits=[5])},
         {"files": _names(1, 2, 3, 4), "movie_frames": 4,
          "out": ["Changed batch_size:2 to 4"], "not_out": ["Cut batch at"]}),
        ("no frame limit leaves the batch whole",
         {"npz": dict(n=4), "over": _tl(batch_size=2, timelapse_frame_limits=None)},
         {"files": _names(1, 2, 3, 4), "movie_frames": 4, "not_out": ["Cut batch at"]}),
        ("a tuple frame limit leaves the batch whole",
         {"npz": dict(n=4), "over": _tl(batch_size=2, timelapse_frame_limits=(0, 2))},
         {"files": _names(1, 2, 3, 4), "movie_frames": 4, "not_out": ["Cut batch at"]}),
        ("motility is left to the merged-frame pipeline",
         {"over": _tl(motility_analysis=True)}, {"motility": 0, "files": n3}),
        ("plot draws one frame at a time",
         {"over": _tl(plot=True), "setup": _record_plots},
         {"plot": [{"n": (3, 3, 3), "flow_shape": (32, 32, 3), "cmap": "inferno",
                    "figuresize": 10, "nr": 1, "print_object_number": True}]}),
        ("lineage, events and division run after tracking",
         {"over": _tl(timelapse_lineage=True, timelapse_events=True,
                      timelapse_division_analysis=True), "setup": _after_tracking},
         {"lineage": [("lineage", "batch1", "nucleus", "trackpy", 3, 3, 3),
                      ("events", "batch1", "nucleus", "trackpy", 3)],
          "division": [("batch1", "nucleus", "trackpy", 3, 3)]}),
    ]
    for case_id, kw, expect in cases:
        with subtests.test(msg=case_id):
            r = _run(tmp_path, case_id, capsys, **kw)
            assert r.error is None, repr(r.error)
            assert r.value is None
            if "files" in expect:
                assert r.files == expect["files"]
            if "counts" in expect:
                assert r.counts == expect["counts"]
            if "eval_sizes" in expect:
                assert [len(x) for x in r.model.eval_inputs] == expect["eval_sizes"]
            if "movie" in expect:
                movies = [{k: m[k] for k in ("n_frames", "filenames", "fps")}
                          for m in r.rec["movie"]]
                assert movies == expect["movie"]
                for m in r.rec["movie"]:
                    assert Path(m["save_path"]) == (
                        r.base / "movies" / "timelapse_nucleus_batch1.mp4")
            if "movie_frames" in expect:
                assert [m["n_frames"] for m in r.rec["movie"]] == [expect["movie_frames"]]
            for tracker, mask_key in (("trackpy", "masks"), ("btrack", "masks_3D")):
                if f"{tracker}_calls" in expect:
                    assert len(r.rec[tracker]) == expect[f"{tracker}_calls"]
                if tracker in expect:
                    assert len(r.rec[tracker]) == 1
                    call = r.rec[tracker][0]
                    for key, value in expect[tracker].items():
                        if key == "n_masks":
                            assert len(call[mask_key]) == value
                        else:
                            assert call[key] == value, key
                    assert call["src"] == str(r.src)
            for name, labels in expect.get("labels", {}).items():
                assert _labels(r.src / "nucleus_mask_stack" / name) == labels
            for text in expect.get("out", []):
                assert text in r.out, text
            for text in expect.get("not_out", []):
                assert text not in r.out, text
            if "timelapse_batch_size" in expect:
                assert r.settings["timelapse_batch_size"] == expect["timelapse_batch_size"]
            if "motility" in expect:
                assert len(r.rec["motility"]) == expect["motility"]
            if "plot" in expect:
                assert r.rec["plot"] == expect["plot"]
            for key in ("lineage", "division"):
                if key in expect:
                    assert r.rec[key] == expect[key]
