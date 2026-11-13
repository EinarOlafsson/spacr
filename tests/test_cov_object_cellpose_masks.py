"""Shared Cellpose / time-lapse doubles for ``generate_cellpose_masks`` plus a few corners.

``fake_cellpose``, ``fake_timelapse``, ``force_cpu``, ``_write_npz`` and ``_tl``
are imported by other suites.  The tests left here cover ``_run_seg_qc`` and
the IPython-free ``display`` fallback; the generator itself is tested in
``tests/test_object_cellpose_masks.py``.
"""
from __future__ import annotations

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

from tests.cellpose_api_contract import (
    MISSING_CHANNEL_AXIS,
    configured_eval_arguments,
    emulate_pretrained_model,
    eval_arguments,
    init_arguments,
)
from tests.conftest import check_cellpose_eval_call


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
    cpu = torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(O.accelerator, "torch_device", lambda: cpu)
    monkeypatch.setattr(
        O.accelerator, "cellpose_kwargs",
        lambda: {"gpu": False, "device": cpu},
    )
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
        """``CellposeModel`` double declaring the installed 4.0.7 signature.

        No ``**kwargs`` on either method: ``generate_cellpose_masks`` is a real
        call site, so an argument cellpose 4 removed must raise ``TypeError``
        here instead of being swallowed. ``eval_configured`` holds only the
        arguments spaCR set away from cellpose's own defaults, which is the
        dict to ask "did spaCR configure this at all".
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
            # object.py names channel_axis here, so the value stays under
            # contract: it goes through the same convert_image the real eval
            # runs before anything else.
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


def _mask_files(src, object_type="nucleus"):
    folder = Path(src) / f"{object_type}_mask_stack"
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir())


def _n_objects(path):
    arr = np.load(path)
    return len(set(np.unique(arr).tolist()) - {0})


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


# --------------------------------------------------------------------------- #
#  segmentation QC is wired into this generator too
# --------------------------------------------------------------------------- #

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
#  generate_cellpose_masks_sam — branches its own suite does not reach
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_sam_model(monkeypatch):
    """Stand-in for the SAM checkpoint used by generate_cellpose_masks_sam."""
    holder = {"model": None}

    class _M:
        """SAM-path ``CellposeModel`` double, installed 4.0.7 signature."""

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
