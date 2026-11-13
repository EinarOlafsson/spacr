"""Shared Cellpose-SAM doubles and builders for the ``generate_cellpose_masks_sam`` suites.

``fake_model``, ``fake_timelapse``, ``force_cpu``, ``_write_npz``, ``_settings``
and ``_tl_settings`` are imported (or loaded via ``pytest_plugins``) by other
test files.  The generator's own tests live in ``tests/test_object_sam_masks.py``.
"""
from __future__ import annotations

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

    cpu = torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(O.accelerator, 'is_gpu', lambda: False)
    monkeypatch.setattr(
        O.accelerator,
        "cellpose_kwargs",
        lambda: {"gpu": False, "device": cpu},
    )
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
        "cell_min_split_area": 0,
        "nucleus_min_split_area": 0,
        "pathogen_min_split_area": 0,
    }
    s.update(over)
    return s


# ---------------------------------------------------------------------------
# Timelapse paths
# ---------------------------------------------------------------------------

def _tl_settings(src, **over):
    base = dict(timelapse=True, timelapse_objects=["cell"], batch_size=2,
                timelapse_displacement=None, timelapse_memory=3,
                timelapse_remove_transient=False, timelapse_frame_limits=[5])
    base.update(over)
    return _settings(src, **base)
