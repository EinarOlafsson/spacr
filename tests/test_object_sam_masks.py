"""Table-driven tests of the Cellpose-SAM mask generator in ``spacr.object``.

One general test per behaviour family, each walking a case table with
pytest's ``subtests`` fixture:

* ``_segment_cellpose_sam``: channel selection per object type, the ndim and
  bounds refusals, the eval settings it forwards and the fully-done batch;
* ``generate_cellpose_masks_sam`` on still fields: model construction, the
  eval call, channel resolution (raw, recorded, stale, text), resume, plot,
  save, the area filter, per-image diameters and the 4-D notes;
* ``generate_cellpose_masks_sam`` on time-lapse batches: the batch re-cut,
  the movie, every tracker and the steps that run after tracking.

Cellpose is replaced by a double declaring the installed 4.0.7 ``eval``
signature (``channel_axis=MISSING_CHANNEL_AXIS``, no ``**kwargs``) that
passes every call through :func:`tests.conftest.check_cellpose_eval_call`.
CUDA is forced off; trackers, the movie writer and the plot helper are
recorders.  The SQLite counts and the ``.npy`` masks on disk are real.
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
from tests.cellpose_api_contract import (  # noqa: E402
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

class _RecordingCP:
    """``CellposeModel.eval`` double for ``_segment_cellpose_sam`` (4.0.7 signature).

    Returns three values; image ``i`` carries one 3x3 object labelled ``i + 1``.
    """

    def __init__(self):
        self.calls = []
        self.configured = []

    def eval(self, x, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
             invert=False, rescale=None, diameter=None, flow_threshold=0.4,
             cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
             max_size_fraction=0.4, niter=None, augment=False,
             tile_overlap=0.1, bsize=256, compute_masks=True, progress=None):
        check_cellpose_eval_call(x, channel_axis)
        bound = locals()
        self.configured.append(configured_eval_arguments(bound))
        arrs = [np.asarray(a) for a in x]
        self.calls.append(dict(eval_arguments(bound), x=arrs))
        n = len(arrs)
        h, w = arrs[0].shape[:2]
        masks = []
        for i in range(n):
            m = np.zeros((h, w), dtype=np.int32)
            m[1:4, 1:4] = i + 1
            masks.append(m)
        flows = [np.zeros((n, h, w, 3), np.float32), np.zeros((3, n, h, w), np.float32),
                 np.zeros((n, h, w), np.float32), np.zeros((n, h, w), np.float32)]
        return masks, flows, None


def _install_sam_model(mp, holder):
    """Swap ``spacr.object.cp_models`` for a recording ``CellposeModel`` class."""

    class _M:
        def __init__(self, gpu=False, pretrained_model="cpsam", model_type=None,
                     diam_mean=None, device=None, nchan=None, use_bfloat16=True):
            self.gpu = gpu
            self.pretrained_model = pretrained_model
            self.device = device
            self.init_kwargs = init_arguments(locals())
            self.loaded_model = emulate_pretrained_model(pretrained_model, model_type)
            self.eval_kwargs = []
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
                flows.append((np.zeros((h, w, 3), np.float32), None, None, None))
            return masks, flows, None

    mp.setattr(O, "cp_models", types.SimpleNamespace(CellposeModel=_M))


def _install_recorders(mp, rec):
    """Record the movie writer, every tracker, the motility hook and the plot."""
    import spacr.plot as PL
    import spacr.timelapse as TL
    import torch

    cpu = torch.device("cpu")
    mp.setattr(torch.cuda, "is_available", lambda: False)
    O.accelerator._reset_device_cache()  # resolve() runs for real per case
    mp.setattr(O.accelerator, "cellpose_kwargs", lambda: {"gpu": False, "device": cpu})
    mp.setattr(torch.cuda, "empty_cache", lambda: rec["empty_cache"].append(1))

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
    for name in ("trackpy", "trackastra", "ultrack", "timepose"):
        mp.setattr(TL, f"_{name}_track_cells", tracker(name, "masks"))
    mp.setattr(TL, "automated_motility_assay", lambda s: rec["motility"].append(s))
    mp.setattr(PL, "plot_cellpose4_output", lambda batch, masks, flows, **kw:
               rec["plot"].append({"n": len(batch), "mask_shape": np.shape(masks[0]),
                                   **kw}))


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
         "verbose": False, "plot": False, "save": True, "timelapse": False,
         "n_jobs": 1, "cell_min_split_area": 0, "nucleus_min_split_area": 0,
         "pathogen_min_split_area": 0}
    s.update(over)
    return s


def _tl(**over):
    """Time-lapse settings overrides (tracked cells, one 2-frame batch)."""
    base = dict(timelapse=True, timelapse_objects=["cell"], batch_size=2,
                timelapse_displacement=None, timelapse_memory=3,
                timelapse_remove_transient=False, timelapse_frame_limits=[5])
    base.update(over)
    return base


def _counts(db_path):
    if not Path(db_path).exists():
        return None
    con = sqlite3.connect(str(db_path))
    try:
        names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        if "object_counts" not in names:
            return None
        return sorted(con.execute(
            "SELECT file_name, count_type, object_count FROM object_counts"))
    finally:
        con.close()


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


# ---------------------------------------------------------------------------
# _segment_cellpose_sam
# ---------------------------------------------------------------------------

def _sam_unit_settings(object_type, **over):
    s = {"nucleus_channel": None, "cell_channel": None, "pathogen_channel": None,
         "organelle_channel": None, "plot": True,
         f"{object_type}_flow_threshold": 0.45,
         f"{object_type}_cellprob_threshold": -0.25}
    s.update(over)
    return s


def _banded(shape, values):
    """Float32 batch whose channel ``k`` is the constant ``values[k]``."""
    b = np.zeros(shape, dtype=np.float32)
    for k, v in enumerate(values):
        b[..., k] = v
    return b


def test_segment_cellpose_sam_selects_channels_and_forwards_eval_settings(
        subtests, tmp_path, monkeypatch):
    """Channels are picked per object type, the ndim/bounds/type guards refuse
    with exact messages, and eval gets the object-scoped thresholds."""
    import spacr.io as IO
    import spacr.utils as U

    ramp = (np.random.default_rng(7).random((2, 10, 10)) * 50).astype(np.float32)
    rand4 = (np.random.default_rng(2).random((2, 16, 16, 2)) * 255).astype(np.float32)
    common_eval = {"diameter": None, "normalize": False, "channel_axis": -1,
                   "channels": None}
    cases = [
        # (id, object_type, batch, settings overrides, identity prepare, expect)
        ("cell: cell plane then nucleus plane", "cell", _banded((1, 8, 8, 3), (1, 2, 3)),
         dict(cell_channel=2, nucleus_channel=0), True,
         {"planes": [[3.0, 1.0]], "eval": {"batch_size": 1, "flow_threshold": 0.45,
                                           "cellprob_threshold": -0.25, "resample": True}}),
        ("cell: a None nucleus is dropped", "cell", _banded((1, 8, 8, 2), (5, 6)),
         dict(cell_channel=1, nucleus_channel=None), True, {"planes": [[6.0]]}),
        ("cell: two real channels", "cell", rand4, dict(cell_channel=0, nucleus_channel=1),
         False, {"shapes": [(16, 16, 2)] * 2}),
        ("nucleus: one channel", "nucleus", rand4, dict(nucleus_channel=0), False,
         {"shapes": [(16, 16, 1)] * 2}),
        ("pathogen: scoped kwargs, batch_size = number of fields", "pathogen",
         np.zeros((3, 8, 8, 1), np.float32),
         dict(pathogen_channel=0, pathogen_flow_threshold=0.62,
              pathogen_cellprob_threshold=1.25, pathogen_resample=False), False,
         {"eval": {"batch_size": 3, "flow_threshold": 0.62, "cellprob_threshold": 1.25,
                   "resample": False}, "shapes": [(8, 8, 1)] * 3}),
        ("organelle: one channel", "organelle", rand4, dict(organelle_channel=0), False,
         {"shapes": [(16, 16, 1)] * 2}),
        ("nucleus: resample defaults to True", "nucleus", np.zeros((1, 8, 8, 1), np.float32),
         dict(nucleus_channel=0), False, {"eval": {"resample": True}}),
        ("3-D batch gains a channel axis", "organelle", ramp, dict(organelle_channel=0),
         True, {"exact": [ramp[0][..., None], ramp[1][..., None]]}),
        ("2-D batch refused", "nucleus", np.zeros((8, 8), np.float32),
         dict(nucleus_channel=0), False,
         {"raises": "Expected batch with ndim 3 or 4, got ndim=2"}),
        ("5-D batch refused", "nucleus", np.zeros((1, 2, 8, 8, 1), np.float32),
         dict(nucleus_channel=0), False,
         {"raises": "Expected batch with ndim 3 or 4, got ndim=5"}),
        ("out-of-bounds channel refused", "nucleus", np.zeros((1, 8, 8, 2), np.float32),
         dict(nucleus_channel=5), False,
         {"raises": "Selected channels for object_type 'nucleus' are out of bounds "
                    "for batch with 2 channels."}),
        ("unknown object type refused", "mitochondrion", np.zeros((1, 8, 8, 1), np.float32),
         {}, False, {"raises": "Unsupported object_type: mitochondrion"}),
        ("no channel configured refused", "organelle", np.zeros((1, 8, 8, 1), np.float32),
         {}, False, {"raises": "No valid channels defined for object_type 'organelle'."}),
        ("nucleus None refused", "nucleus", np.zeros((1, 16, 16, 1), np.float32),
         dict(nucleus_channel=None, plot=False), False,
         {"raises": "No valid channels defined for object_type 'nucleus'."}),
        ("every mask already on disk", "nucleus", np.zeros((2, 8, 8, 1), np.float32),
         dict(nucleus_channel=0, plot=False), False, {"none": True}),
    ]
    for case_id, obj, batch, over, identity, expect in cases:
        with subtests.test(msg=case_id):
            names = [f"f{i}.npy" for i in range(len(batch))]
            model = _RecordingCP()
            out_dir = str(_case_dir(tmp_path, case_id))
            settings = _sam_unit_settings(obj, **over)
            with monkeypatch.context() as mp:
                if identity:
                    mp.setattr(U, "prepare_batch_for_segmentation", lambda b: b)
                if expect.get("none"):
                    mp.setattr(IO, "_check_masks", lambda b, f, o, resume=False:
                               (np.zeros((0, 8, 8, 1), np.float32), []))
                if "raises" in expect:
                    with pytest.raises(ValueError, match=re.escape(expect["raises"])):
                        O._segment_cellpose_sam(batch, names, model, settings, obj, out_dir)
                    assert model.calls == []
                    continue
                out = O._segment_cellpose_sam(batch, names, model, settings, obj, out_dir)
            if expect.get("none"):
                assert out is None and model.calls == []
                continue
            assert len(model.calls) == 1
            call = model.calls[0]
            # the masks are exactly what the model drew, one per field
            assert [np.unique(m).tolist() for m in out] == [[0, i + 1] for i in range(len(batch))]
            assert {k: call[k] for k in common_eval} == common_eval
            assert "channels" not in model.configured[0]
            if "eval" in expect:
                assert {k: call[k] for k in expect["eval"]} == expect["eval"]
            if "planes" in expect:
                assert [[float(v) for v in np.unique(x.reshape(-1, x.shape[-1]), axis=0)[0]]
                        for x in call["x"]] == expect["planes"]
            if "shapes" in expect:
                assert [x.shape for x in call["x"]] == expect["shapes"]
            if "exact" in expect:
                for got, want in zip(call["x"], expect["exact"]):
                    np.testing.assert_array_equal(got, want)


# ---------------------------------------------------------------------------
# _validate_organelle_settings
# ---------------------------------------------------------------------------

def test_validate_organelle_settings_accepts_exactly_the_documented_pairs(subtests):
    """Every morphology x method pair is accepted iff it is in the method map;
    refusals name the morphology and the allowed methods."""
    allowed = {
        "spots": ("otsu", "adaptive", "log", "dog", "cellpose"),
        "network": ("otsu", "adaptive", "ridge", "hysteresis", "cellpose", "unet"),
        "irregular": ("otsu", "adaptive", "cellpose"),
        "ring": ("otsu", "adaptive", "dog", "log", "cellpose"),
    }
    methods = sorted({m for ms in allowed.values() for m in ms} | {"bogus"})
    for morphology, ok in allowed.items():
        for method in methods:
            with subtests.test(msg=f"{morphology}/{method}"):
                if method in ok:
                    assert O._validate_organelle_settings(morphology, method) is None
                else:
                    with pytest.raises(ValueError) as exc:
                        O._validate_organelle_settings(morphology, method)
                    assert str(exc.value) == (f"For morphology='{morphology}', method must "
                                              f"be one of {ok}, got '{method}'")
    for morphology in ("bogus", "mesh", "Spots"):
        with subtests.test(msg=f"morphology {morphology}"):
            with pytest.raises(ValueError) as exc:
                O._validate_organelle_settings(morphology, "otsu")
            assert str(exc.value) == ("organelle_morphology must be one of ('spots', "
                                      f"'network', 'irregular', 'ring'), got '{morphology}'")


# ---------------------------------------------------------------------------
# generate_cellpose_masks_sam: shared runner
# ---------------------------------------------------------------------------

def _run_generator(tmp_path, case_id, monkeypatch, capsys, case):
    """Run one generator case in its own folder and return what it left behind."""
    root = _case_dir(tmp_path, case_id)
    src = root / "stack"
    data = _write_npz(src, **case.get("npz", {}))
    obj = case.get("object_type", "cell")
    for name in case.get("existing", []):
        folder = src / f"{obj}_mask_stack"
        folder.mkdir(exist_ok=True)
        np.save(folder / name, np.full((32, 32), 7, dtype=np.uint16))
    holder = {"model": None, "n_objects": case.get("n_objects", 2)}
    rec = {k: [] for k in ("movie", "btrack", "trackpy", "trackastra", "ultrack",
                           "timepose", "motility", "plot", "empty_cache")}
    log = []
    settings = _settings(src, **case.get("settings", {}))
    capsys.readouterr()
    with monkeypatch.context() as mp:
        _install_sam_model(mp, holder)
        _install_recorders(mp, rec)
        if "patch" in case:
            case["patch"](mp, log)
        if "raises" in case:
            with pytest.raises(ValueError, match=re.escape(case["raises"])):
                O.generate_cellpose_masks_sam(str(src), settings, obj)
            returned = "raised"
        else:
            returned = O.generate_cellpose_masks_sam(str(src), settings, obj)
    folder = src / f"{obj}_mask_stack"
    files = sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []
    model = holder["model"]
    first = model.eval_inputs[0][0] if model is not None and model.eval_inputs else None
    return {
        "returned": returned, "files": files, "folder": folder.is_dir(),
        "rows": _counts(root / "measurements" / "measurements.db"),
        "model": None if model is None else {
            "pretrained": model.pretrained_model, "gpu": model.gpu,
            "device": str(model.device)},
        "evals": None if model is None else len(model.eval_kwargs),
        "eval": model.eval_kwargs[0] if model is not None and model.eval_kwargs else None,
        "inputs": None if first is None else [
            (im.shape, str(im.dtype)) for im in model.eval_inputs[0]],
        "planes": None if first is None else tuple(
            _which_channel(first[..., k], data[0]) for k in range(first.shape[-1])),
        "unit_range": None if first is None else bool(first.max() <= 1.0),
        "masks": {f: np.load(folder / f) for f in files},
        "out": capsys.readouterr().out, "settings": settings, "rec": rec, "log": log,
        "root": root, "qc": (root / "qc").exists(),
    }


def _check(obs, expect):
    """Compare every expected key with the observation, exactly."""
    for key, want in expect.items():
        if key == "out_has":
            for fragment in want:
                assert fragment in obs["out"], fragment
        elif key == "out_lacks":
            for fragment in want:
                assert fragment not in obs["out"], fragment
        elif key == "eval":
            assert {k: obs["eval"][k] for k in want} == want
        elif key == "settings":
            assert {k: obs["settings"].get(k) for k in want} == want
        elif key == "labels":
            for name, labels in want.items():
                mask = obs["masks"][name]
                assert (mask.dtype, mask.shape) == (np.dtype(np.uint16), (32, 32))
                assert np.unique(mask).tolist() == labels
        elif key == "calls":
            assert {name: len(obs["rec"][name]) for name in want} == want
        elif key == "tracked":
            for name, subset in want.items():
                kw = obs["rec"][name][0]
                assert {k: kw[k] for k in subset} == subset
        elif key == "plot":
            assert [{k: call[k] for k in want[0]} for call in obs["rec"]["plot"]] == want
        elif key == "movie":
            assert obs["rec"]["movie"] == want
        elif key == "empty_cache":
            assert len(obs["rec"]["empty_cache"]) == want
        else:
            assert obs[key] == want, key


def _rows(names, kind, n):
    return sorted((name, kind, n) for name in names)


# ---------------------------------------------------------------------------
# generate_cellpose_masks_sam: still fields
# ---------------------------------------------------------------------------

def _gpu(mp, log):
    import torch
    mp.setattr(torch.cuda, "is_available", lambda: True)
    O.accelerator._reset_device_cache()
    mp.setattr(O.accelerator, "cellpose_kwargs",
               lambda: {"gpu": True, "device": torch.device("cuda:7")})


def _per_image(estimates):
    def patch(mp, log):
        def per_image(model, images, eval_kwargs, fallback, **governor):
            log.append(("per_image", len(images)))
            masks = [np.asarray(m, np.uint16) for m in model.eval(images, channel_axis=-1)[0]]
            return masks, [np.zeros(m.shape, np.float32) for m in masks], estimates
        mp.setattr(O, "_per_image_diameter_masks", per_image)
        mp.setattr(O, "_note_the_diameter", lambda settings, obj, estimate=None:
                   log.append((settings[f"{obj}_diameter"], obj, estimate)))
    return patch


def _four_d_notes(mp, log):
    mp.setattr(O, "display", lambda *a, **k: None)

    class _Result:
        notes = ["t_axis=0 read as time"]
        z_results = [types.SimpleNamespace(notes=["frame 1 projected"]),
                     types.SimpleNamespace(notes=["frame 2 projected"])]

    def fake_t(acquisition, model, t_plan, eval_kwargs):
        frames = np.asarray(acquisition)
        masks = [np.zeros(frames.shape[1:3], np.uint16) for _ in range(frames.shape[0])]
        for m in masks:
            m[2:8, 2:8] = 1
        log.append(("4d", frames.shape[0], eval_kwargs["batch_size"]))
        return masks, _Result(), None

    mp.setattr(O, "_segment_timepoints_with_t", fake_t)


SAM_EVAL = {"normalize": False, "channel_axis": -1, "diameter": 120.0, "progress": True,
            "min_size": 0, "flow_threshold": 0.4, "cellprob_threshold": 0}

STILL_CASES = [
    ("basic run: cpsam on CPU, whole batch, masks and counts", {
        "expect": {
            "returned": None, "model": {"pretrained": "cpsam", "gpu": False, "device": "cpu"},
            "empty_cache": 1, "evals": 1, "qc": True,
            "eval": {**SAM_EVAL, "batch_size": 3, "resample": True},
            "inputs": [((32, 32, 2), "float32")] * 3, "unit_range": True,
            "files": _names(1, 2, 3), "labels": {n: [0, 1, 2] for n in _names(1, 2, 3)},
            "rows": _rows(_names(1, 2, 3), "cell_before_filtration", 2),
            "out_has": ["Torch GPU is not available, using CPU", "saving to DB",
                        "Found 2.0 cell/FOV. average size: 36.000 px2"]}}),
    ("a CUDA box builds a GPU model; seg_qc off writes no QC report", {
        "npz": {"n": 2}, "patch": _gpu, "settings": {"seg_qc": "off"},
        "expect": {"model": {"pretrained": "cpsam", "gpu": True, "device": "cuda:7"},
                   "files": _names(1, 2), "qc": False,
                   "out_lacks": ["Torch GPU is not available"]}}),
    ("single-channel stack keeps channel 0", {
        "npz": {"c": 1},
        "expect": {"inputs": [((32, 32, 1), "float32")] * 3, "files": _names(1, 2, 3)}}),
    ("pathogen channel mirrored, pathogen settings reach eval", {
        "npz": {"n": 2, "c": 3}, "object_type": "pathogen",
        "settings": {"pathogen_channel": 2, "pathogen_min_area": 0},
        "expect": {"settings": {"cellpose_pathogen_channel": 2},
                   "inputs": [((32, 32, 1), "float32")] * 2, "planes": (2,),
                   "eval": {"resample": False, "min_size": 0},
                   "files": _names(1, 2),
                   "rows": _rows(_names(1, 2), "pathogen_before_filtration", 2)}}),
    ("no cell channel leaves its alias unset", {
        "npz": {"n": 2, "c": 1}, "object_type": "nucleus",
        "settings": {"cell_channel": None, "nucleus_channel": 0},
        "expect": {"settings": {"cellpose_cell_channel": None, "cellpose_nucleus_channel": 0},
                   "files": _names(1, 2)}}),
    ("an object type without a channel raises before a model loads", {
        "npz": {"n": 1}, "object_type": "pathogen",
        "settings": {"nucleus_channel": None},
        "raises": "No valid channels defined for object_type 'pathogen'.",
        "expect": {"model": None, "files": []}}),
    ("stale recorded positions are recomputed from the raw channels", {
        "npz": {"n": 2, "c": 2, "source_channels": [1, 0]},
        "settings": {"cellpose_cell_channel": 0, "cellpose_nucleus_channel": 1},
        "expect": {"planes": (1, 0),
                   "settings": {"cellpose_cell_channel": 1, "cellpose_nucleus_channel": 0}}}),
    ("a text channel is skipped, not fatal", {
        "npz": {"n": 2, "c": 3}, "settings": {"organelle_channel": "not-a-channel"},
        "expect": {"returned": None, "files": _names(1, 2), "planes": (1, 0)}}),
    ("an existing mask is skipped and preserved", {
        "existing": _names(1),
        "expect": {"inputs": [((32, 32, 2), "float32")] * 2, "files": _names(1, 2, 3),
                   "labels": {_names(1)[0]: [7], _names(2)[0]: [0, 1, 2]},
                   "rows": _rows(_names(2, 3), "cell_before_filtration", 2),
                   "out_has": ["File plate1_A01_1.npy already exists in the output folder. "
                               "Skipping..."]}}),
    ("a fully processed batch is never evaluated", {
        "existing": _names(1, 2, 3),
        "expect": {"evals": 0, "rows": None, "files": _names(1, 2, 3)}}),
    ("plot=True plots every field and skips the resume check", {
        "existing": _names(1), "settings": {"plot": True},
        "expect": {"inputs": [((32, 32, 2), "float32")] * 3,
                   "plot": [{"n": 3, "nr": 3, "cmap": "inferno", "figuresize": 10,
                             "mask_shape": (32, 32)}]}}),
    ("save=False keeps the counts and writes no masks", {
        "npz": {"n": 2}, "settings": {"save": False},
        "expect": {"folder": True, "files": [],
                   "rows": _rows(_names(1, 2), "cell_before_filtration", 2)}}),
    ("min_area above every object empties the masks", {
        "npz": {"n": 2}, "settings": {"cell_min_area": 100},
        "expect": {"rows": _rows(_names(1, 2), "cell_before_filtration", 0),
                   "labels": {_names(1)[0]: [0]},
                   "out_has": ["Found 0.0 cell/FOV. average size: 0.000 px2"]}}),
    ("a model drawing nothing reports zero", {
        "npz": {"n": 2}, "n_objects": 0,
        "expect": {"rows": _rows(_names(1, 2), "cell_before_filtration", 0),
                   "out_has": ["Found 0.0 cell/FOV. average size: 0.000 px2"]}}),
    ("every batch of a multi-batch archive is saved", {
        "npz": {"n": 4}, "settings": {"batch_size": 2},
        "expect": {"evals": 2, "files": _names(1, 2, 3, 4),
                   "rows": _rows(_names(1, 2, 3, 4), "cell_before_filtration", 2)}}),
    ("an empty archive runs nothing", {
        "npz": {"n": 0},
        "expect": {"returned": None, "evals": 0, "files": [], "rows": None}}),
    ("per-image diameters: the median estimate is noted", {
        "settings": {"estimate_diameters_per_image": True, "cell_diameter": "auto"},
        "patch": _per_image([12.0, 0, 16.0]),
        "expect": {"files": _names(1, 2, 3),
                   "log": [("auto", "cell", None), ("per_image", 3), ("auto", "cell", 14.0)]}}),
    ("per-image diameters without an estimate note nothing", {
        "npz": {"n": 2},
        "settings": {"estimate_diameters_per_image": True, "cell_diameter": "auto"},
        "patch": _per_image([0, None]),
        "expect": {"log": [("auto", "cell", None), ("per_image", 2)]}}),
    ("the 4-D path prints its acquisition and frame notes", {
        "npz": {"n": 2},
        "settings": {"verbose": True, "t_stack": True, "t_axis_order": "TYX"},
        "patch": _four_d_notes,
        "expect": {"log": [("4d", 2, 1)], "files": _names(1, 2),
                   "out_has": ["[4D] batch1: t_axis=0 read as time",
                               "[4D] plate1_A01_1.npy: frame 1 projected",
                               "[4D] plate1_A01_2.npy: frame 2 projected"]}}),
]


def test_generate_cellpose_masks_sam_segments_saves_and_counts_still_fields(
        subtests, tmp_path, monkeypatch, capsys):
    """Still-field runs build the right model, hand eval spaCR's SAM settings on
    the resolved channels, honour resume/plot/save/filters and record counts."""
    for case_id, case in STILL_CASES:
        with subtests.test(msg=case_id):
            obs = _run_generator(tmp_path, case_id, monkeypatch, capsys, case)
            _check(obs, case["expect"])


# ---------------------------------------------------------------------------
# generate_cellpose_masks_sam: time-lapse
# ---------------------------------------------------------------------------

def _recording(module, name, fmt):
    def patch(mp, log):
        import importlib
        mod = importlib.import_module(module)
        mp.setattr(mod, name, lambda *a, **k: log.append(fmt(*a, **k)))
    return patch


def _both(*patches):
    def patch(mp, log):
        for p in patches:
            p(mp, log)
    return patch


def _cpu_count(n):
    return lambda mp, log: mp.setattr(os, "cpu_count", lambda: n)


TL_CASES = [
    ("timelapse objects must be trackable", {
        "npz": {"n": 2}, "settings": _tl(timelapse_objects=["organelle"]),
        "expect": {"returned": None, "evals": 0, "movie": [],
                   "out_has": ["timelapse_objects ['organelle'] must be a subset of "
                               "['cell', 'nucleus', 'pathogen']"]}}),
    ("btrack: batch re-cut, movie and solver arguments", {
        "settings": _tl(timelapse_frame_limits=[0, 2], timelapse_displacement=50,
                        timelapse_mode="btrack"),
        "patch": _cpu_count(1),
        "expect": {"settings": {"timelapse_batch_size": 3},
                   "out_has": ["Changed batch_size:2 to 3, data length:3",
                               "Cut batch at indecies: [0, 2], New batch_size: 2"],
                   "movie": [{"n_frames": 2, "filenames": _names(1, 2), "fps": 2,
                              "save_path": "MOVIE"}],
                   "calls": {"btrack": 1, "trackpy": 0, "trackastra": 0},
                   "tracked": {"btrack": {"radius": 50, "n_jobs": 1, "object_type": "cell",
                                          "mode": "btrack", "name": "batch1",
                                          "batch_list": None, "run_optimization": True,
                                          "max_objects_for_optimization": 20000,
                                          "optimizer_time_limit_s": 120,
                                          "optimizer_mip_gap": 0.01}},
                   "rows": _rows(_names(1, 2), "cell_timelapse", 2),
                   "files": _names(1, 2)}}),
    ("btrack: default radius and cpu_count - 2 workers", {
        "npz": {"n": 2}, "settings": _tl(timelapse_mode="btrack"), "patch": _cpu_count(6),
        "expect": {"tracked": {"btrack": {"radius": 100, "n_jobs": 4}}}}),
    ("trackastra gets labels and raw intensities", {
        "npz": {"n": 2}, "settings": _tl(timelapse_mode="trackastra"),
        "expect": {"tracked": {"trackastra": {"model_name": "general_2d",
                                              "linking_mode": "greedy",
                                              "mode": "trackastra",
                                              "batch_filenames": _names(1, 2)}},
                   "image_shape": ("trackastra", (2, 32, 32, 2), 2),
                   "calls": {"btrack": 0}, "files": _names(1, 2)}}),
    ("trackpy tracks by centroid", {
        "npz": {"n": 2},
        "settings": _tl(timelapse_mode="trackpy", timelapse_displacement=25,
                        timelapse_memory=5),
        "expect": {"tracked": {"trackpy": {"track_by_iou": False, "mode": "trackpy",
                                           "timelapse_displacement": 25,
                                           "timelapse_memory": 5}},
                   "calls": {"btrack": 0}}}),
    ("iou mode is trackpy by overlap", {
        "npz": {"n": 2},
        "settings": _tl(timelapse_mode="iou", timelapse_displacement=25, timelapse_memory=5),
        "expect": {"tracked": {"trackpy": {"track_by_iou": True, "mode": "iou"}}}}),
    ("ultrack reads its solver parameters from settings", {
        "settings": _tl(timelapse_mode="ultrack", timelapse_remove_transient=True,
                        timelapse_frame_limits=[0, 3], batch_size=50,
                        ultrack_max_distance=33.0, ultrack_division_weight=-0.25,
                        ultrack_contour_sigma=1.5, ultrack_n_workers=2),
        "expect": {"tracked": {"ultrack": {"object_type": "cell", "max_distance": 33.0,
                                           "division_weight": -0.25, "contour_sigma": 1.5,
                                           "n_workers": 2,
                                           "timelapse_remove_transient": True,
                                           "batch_filenames": _names(1, 2, 3)}},
                   "image_shape": ("ultrack", (3, 32, 32, 2), 3),
                   "files": _names(1, 2, 3)}}),
    ("timepose gets labels, images and the release key", {
        "npz": {"n": 2},
        "settings": _tl(timelapse_mode="timepose_v1", timelapse_remove_transient=True),
        "expect": {"tracked": {"timepose": {"mode": "timepose_v1", "object_type": "cell",
                                            "batch_filenames": _names(1, 2),
                                            "timelapse_remove_transient": True}},
                   "image_shape": ("timepose", (2, 32, 32, 2), 2),
                   "calls": {"btrack": 0, "trackastra": 0}}}),
    ("an untracked object keeps its plain mask stack", {
        "npz": {"n": 2}, "settings": _tl(timelapse_objects=["nucleus"], timelapse_mode="btrack"),
        "expect": {"calls": {"btrack": 0, "trackpy": 0, "trackastra": 0},
                   "files": _names(1, 2), "rows": _rows(_names(1, 2), "cell_timelapse", 2)}}),
    ("time-lapse plot: one frame with object numbers", {
        "npz": {"n": 2}, "settings": _tl(plot=True, timelapse_objects=["nucleus"]),
        "expect": {"plot": [{"n": 2, "nr": 1, "print_object_number": True}]}}),
    ("motility is deferred to the pipeline", {
        "npz": {"n": 2}, "settings": _tl(timelapse_objects=["nucleus"], motility_analysis=True),
        "expect": {"calls": {"motility": 0}, "files": _names(1, 2)}}),
    ("unusable frame limits keep the whole batch", {
        "npz": {"n": 4}, "settings": _tl(timelapse_mode="trackpy", timelapse_frame_limits=None),
        "expect": {"files": _names(1, 2, 3, 4), "frames": 4}}),
    ("a one-element frame limit keeps the whole batch", {
        "npz": {"n": 4}, "settings": _tl(timelapse_mode="trackpy", timelapse_frame_limits=[5]),
        "expect": {"files": _names(1, 2, 3, 4), "frames": 4}}),
    ("a tuple frame limit is not a list and keeps the whole batch", {
        "npz": {"n": 4}, "settings": _tl(timelapse_mode="trackpy",
                                         timelapse_frame_limits=(0, 2)),
        "expect": {"files": _names(1, 2, 3, 4), "frames": 4}}),
    ("lineage then events run after tracking", {
        "npz": {"n": 2},
        "settings": _tl(timelapse_mode="trackpy", timelapse_lineage=True,
                        timelapse_events=True),
        "patch": _both(
            _recording("spacr.timelapse", "_run_lineage_step",
                       lambda src, name, obj, mode, settings, *, frame_sources, label_stack,
                       images=None: ("lineage", name, obj, mode, len(frame_sources),
                                     len(label_stack), len(images))),
            _recording("spacr.timelapse", "_run_event_features_step",
                       lambda src, name, obj, stack, images, mode, settings:
                       ("events", name, obj, mode, len(stack)))),
        "expect": {"log": [("lineage", "batch1", "cell", "trackpy", 2, 2, 2),
                           ("events", "batch1", "cell", "trackpy", 2)]}}),
    ("tracked fields reach the division analysis", {
        "npz": {"n": 2},
        "settings": _tl(timelapse_mode="trackpy", timelapse_division_analysis=True),
        "patch": _recording("spacr.division_analysis", "_run_division_step",
                            lambda src, name, obj, mode, settings, *, label_stack,
                            images=None: ("division", name, obj, mode, len(label_stack),
                                          len(images))),
        "expect": {"log": [("division", "batch1", "cell", "trackpy", 2, 2)]}}),
]


def test_generate_cellpose_masks_sam_hands_timelapse_batches_to_the_chosen_tracker(
        subtests, tmp_path, monkeypatch, capsys):
    """Time-lapse runs re-cut the batch, write the movie, call exactly the chosen
    tracker with its arguments, then the lineage/event/division steps."""
    for case_id, case in TL_CASES:
        with subtests.test(msg=case_id):
            obs = _run_generator(tmp_path, case_id, monkeypatch, capsys, case)
            expect = dict(case["expect"])
            if "movie" in expect and expect["movie"]:
                expect["movie"] = [dict(m, save_path=str(
                    obs["root"] / "movies" / "timelapse_cell_batch1.mp4"))
                    for m in expect["movie"]]
            if "image_shape" in expect:
                name, shape, n_masks = expect.pop("image_shape")
                kw = obs["rec"][name][0]
                assert (np.asarray(kw["images"]).shape, len(kw["masks"])) == (shape, n_masks)
            if "frames" in expect:
                assert [m["n_frames"] for m in obs["rec"]["movie"]] == [expect.pop("frames")]
            _check(obs, expect)
