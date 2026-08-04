"""Coverage for spacr.object preprocessing + Cellpose segmentation helpers.

Targets the block of ``spacr/object.py`` that sits between the classical
settings extractor and the Cellpose-SAM wrapper:

    _extract_classical_settings   whitelist / pickle-safe subset
    _preprocess_batch             rolling-ball + CLAHE, incl. the flat-image
                                  (pmax == pmin) fallback
    _apply_cell_mask              exact-path and ``.npy``-suffix fallback
    _load_unet_model              bad-path guard + eval()/device handling
    _segment_cellpose             channel remap/clamp, nucleus fallback,
                                  eval() kwarg forwarding, empty-batch guard
    _segment_cellpose_sam         per-object-type channel selection, ndim
                                  dispatch, bounds guards, empty-batch guard

Everything here is CPU-only and offline: the Cellpose model is a recording
stand-in, ``torch.cuda.is_available`` is forced False where the device choice
matters, and the "all masks already on disk" branches are reached by injecting
an empty return from ``spacr.io._check_masks``.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from tests.cellpose_api_contract import (
    DEPRECATED_EVAL_ARGUMENTS,
    MISSING_CHANNEL_AXIS,
    configured_eval_arguments,
    eval_arguments,
)
from tests.conftest import check_cellpose_eval_call


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_stray_figures():
    """Never let an Agg figure leak between tests."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


class _RecordingCP:
    """Minimal CellposeModel stand-in that records every eval() call.

    ``eval`` declares the installed cellpose 4.0.7 parameter list verbatim,
    with the real defaults and no ``**kwargs``, so an argument cellpose 4
    removed raises ``TypeError`` at the call site instead of disappearing.

    It returns THREE values — ``(masks, flows, styles)`` — which is what 4.0.7
    returns on both of its return paths. The previous docstring claimed
    ``(masks, flows, styles, diams)`` was "cellpose-4 shaped"; that is the
    cellpose 3 shape, and returning it would keep a four-value unpack green
    against a library that raises ValueError on one.
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
        # Both _segment_cellpose and _segment_cellpose_sam name channel_axis,
        # so the value stays under the convert_image contract.
        check_cellpose_eval_call(x, channel_axis)
        # One snapshot, taken before any local of this method exists, so the
        # two views describe exactly the bound parameters.
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
        flows = [
            np.zeros((n, h, w, 3), dtype=np.float32),
            np.zeros((3, n, h, w), dtype=np.float32),
            np.zeros((n, h, w), dtype=np.float32),
            np.zeros((n, h, w), dtype=np.float32),
        ]
        # Three values, not four -- see the class docstring.
        return masks, flows, None


def _patch_prepare_identity(monkeypatch):
    """Make prepare_batch_for_segmentation a pass-through recorder.

    ``_segment_cellpose`` / ``_segment_cellpose_sam`` import it inside the
    function body, so patching the attribute on ``spacr.utils`` is enough.
    Neutralising the normalisation lets the tests assert on the *exact*
    channel content that object.py selected.
    """
    import spacr.utils as U
    monkeypatch.setattr(U, "prepare_batch_for_segmentation", lambda b: b)


def _patch_check_masks_empty(monkeypatch, n_channels=2):
    """Simulate 'every mask in this batch already exists on disk'."""
    import spacr.io as IO
    empty = np.zeros((0, 8, 8, n_channels), dtype=np.float32)
    # The stub must carry ``resume=`` too: io._check_masks grew that parameter
    # and object.py passes it, so a three-parameter lambda raised TypeError
    # rather than exercising the branch under test.
    monkeypatch.setattr(IO, "_check_masks",
                        lambda b, f, o, resume=False: (empty, []))


_CLASSICAL_KEYS = (
    "organelle_morphology", "organelle_method",
    "organelle_min_size", "organelle_max_size",
    "organelle_tophat_radius", "organelle_watershed_spots",
    "organelle_log_min_sigma", "organelle_log_max_sigma",
    "organelle_log_num_sigma", "organelle_log_threshold",
    "organelle_dog_sigma_low", "organelle_dog_sigma_high",
    "organelle_ridge_sigmas", "organelle_ridge_filter",
    "organelle_skeletonize", "organelle_network_threshold",
    "organelle_hysteresis_low", "organelle_hysteresis_high",
    "organelle_adaptive_block_size", "organelle_adaptive_offset",
    "organelle_morph_radius", "organelle_fill_holes",
    "organelle_ring_sigma_inner", "organelle_ring_sigma_outer",
    "organelle_ring_min_prominence", "organelle_ring_fill_method",
)


def _full_organelle_settings():
    """Every classical key plus a pile of keys that must NOT survive."""
    s = {
        "organelle_morphology": "ring", "organelle_method": "dog",
        "organelle_min_size": 7, "organelle_max_size": 900,
        "organelle_tophat_radius": 4, "organelle_watershed_spots": True,
        "organelle_log_min_sigma": 1.0, "organelle_log_max_sigma": 4.0,
        "organelle_log_num_sigma": 6, "organelle_log_threshold": 0.02,
        "organelle_dog_sigma_low": 1.5, "organelle_dog_sigma_high": 3.5,
        "organelle_ridge_sigmas": [1, 2, 3], "organelle_ridge_filter": "sato",
        "organelle_skeletonize": False, "organelle_network_threshold": 0.3,
        "organelle_hysteresis_low": 30.0, "organelle_hysteresis_high": 80.0,
        "organelle_adaptive_block_size": 15, "organelle_adaptive_offset": -0.5,
        "organelle_morph_radius": 3, "organelle_fill_holes": 25,
        "organelle_ring_sigma_inner": 1.25, "organelle_ring_sigma_outer": 4.0,
        "organelle_ring_min_prominence": 0.07,
        "organelle_ring_fill_method": "convex",
        # -- must be dropped --
        "organelle_model_name": "cpsam",
        "organelle_diameter": 30,
        "organelle_resample": True,
        "organelle_channel": 2,
        "src": "/some/where",
        "batch_size": 8,
        "plot": True,
    }
    return s


# ---------------------------------------------------------------------------
# _extract_classical_settings
# ---------------------------------------------------------------------------

def test_extract_classical_settings_keeps_every_whitelisted_key():
    from spacr.object import _extract_classical_settings

    src = _full_organelle_settings()
    out = _extract_classical_settings(src)

    assert set(out) == set(_CLASSICAL_KEYS)
    for key in _CLASSICAL_KEYS:
        assert out[key] == src[key], key
    # Non-whitelisted keys are dropped, and the caller's dict is untouched.
    for dropped in ("organelle_model_name", "organelle_diameter",
                    "organelle_resample", "organelle_channel",
                    "src", "batch_size", "plot"):
        assert dropped not in out
    assert out is not src
    assert len(src) == len(_CLASSICAL_KEYS) + 7


def test_extract_classical_settings_result_is_pickle_safe():
    """The docstring promises a pickle-safe subset for the worker Pool."""
    import pickle
    from spacr.object import _extract_classical_settings

    src = _full_organelle_settings()
    # A deliberately unpicklable value on a NON-whitelisted key must not
    # break round-tripping of the extracted subset.
    src["organelle_progress_callback"] = lambda *a: None
    out = _extract_classical_settings(src)

    restored = pickle.loads(pickle.dumps(out))
    assert restored == out
    assert restored["organelle_ring_fill_method"] == "convex"
    assert "organelle_progress_callback" not in out


def test_extract_classical_settings_ignores_absent_keys():
    from spacr.object import _extract_classical_settings

    out = _extract_classical_settings(
        {"organelle_morphology": "irregular", "unrelated": 1}
    )
    assert out == {"organelle_morphology": "irregular"}


# ---------------------------------------------------------------------------
# _preprocess_batch
# ---------------------------------------------------------------------------

def _sloped_batch(n=2, size=32, base=0.5):
    """Float32 batch: linear background ramp + one bright square per frame."""
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    ramp = base + (yy + xx) / (2.0 * size)   # base .. base+1 background ramp
    frames = []
    for i in range(n):
        img = ramp.copy()
        img[8 + i: 16 + i, 8 + i: 16 + i] += 2.0
        frames.append(img)
    return np.stack(frames).astype(np.float32)


def test_preprocess_batch_noop_returns_same_object():
    from spacr.object import _preprocess_batch

    batch = _sloped_batch()
    out = _preprocess_batch(batch, {"organelle_rolling_ball": False,
                                    "organelle_clahe": False})
    assert out is batch


def test_preprocess_batch_rolling_ball_removes_background_gradient():
    from spacr.object import _preprocess_batch

    batch = _sloped_batch()
    original = batch.copy()
    out = _preprocess_batch(batch, {"organelle_rolling_ball": True,
                                    "organelle_rolling_ball_radius": 6})

    assert out is not batch
    assert np.array_equal(batch, original)          # input not mutated
    assert out.shape == batch.shape and out.dtype == batch.dtype
    assert (out >= 0).all()                          # np.clip(.., 0, None)
    # Background is flattened: the dark corners come down toward zero while
    # the bright square survives.
    assert out[0][0, 0] < batch[0][0, 0]
    assert out[0][-1, -1] < batch[0][-1, -1]
    assert out[0][10, 10] > out[0][0, 0]


def test_preprocess_batch_rolling_ball_and_clahe_together():
    from spacr.object import _preprocess_batch

    batch = _sloped_batch()
    out = _preprocess_batch(
        batch,
        {"organelle_rolling_ball": True, "organelle_rolling_ball_radius": 6,
         "organelle_clahe": True, "organelle_clahe_clip_limit": 0.02},
    )
    assert out.shape == batch.shape
    assert out.dtype == np.float32
    # CLAHE output is renormalised into [0, 1].
    assert out.min() >= 0.0
    assert out.max() <= 1.0 + 1e-6
    # And it is genuinely a different image than plain rolling-ball output.
    rb_only = _preprocess_batch(
        batch, {"organelle_rolling_ball": True,
                "organelle_rolling_ball_radius": 6})
    assert not np.allclose(out, rb_only)


def test_preprocess_batch_clahe_flat_image_uses_zero_fallback():
    """pmax == pmin -> np.zeros_like branch instead of a divide-by-zero.

    A flat frame normalises to all-zeros, and CLAHE of an all-zero frame is a
    constant image -- the point is that nothing becomes NaN/inf and the frame
    carries no spurious structure.
    """
    from spacr.object import _preprocess_batch

    batch = np.full((2, 32, 32), 3.5, dtype=np.float32)
    with np.errstate(divide="raise", invalid="raise"):
        out = _preprocess_batch(batch, {"organelle_clahe": True,
                                        "organelle_clahe_clip_limit": 0.01})
    assert out.shape == batch.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out.min() == out.max()               # still perfectly flat
    assert 0.0 <= out.min() <= 1.0


def test_preprocess_batch_clahe_mixed_flat_and_textured_frames():
    """Only the flat frame takes the zeros branch; the other is equalised."""
    from spacr.object import _preprocess_batch

    textured = _sloped_batch(n=1)[0]
    flat = np.full((32, 32), 0.25, dtype=np.float32)
    batch = np.stack([flat, textured]).astype(np.float32)

    out = _preprocess_batch(batch, {"organelle_clahe": True})
    assert out[0].min() == out[0].max()         # flat frame stays structureless
    assert np.isfinite(out[0]).all()
    assert out[1].max() > out[1].min()          # real contrast survives
    assert out[1].min() >= 0.0
    assert out[1].max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# _apply_cell_mask
# ---------------------------------------------------------------------------

def test_apply_cell_mask_zeroes_outside_cells_without_mutating_input(tmp_path):
    from spacr.object import _apply_cell_mask

    batch = np.ones((2, 8, 8), dtype=np.float32)
    original = batch.copy()

    cell_mask = np.zeros((8, 8), dtype=np.int32)
    cell_mask[2:5, 2:5] = 7            # label id != 1 on purpose
    np.save(str(tmp_path / "f0.npy"), cell_mask)

    out = _apply_cell_mask(batch, ["f0.npy", "absent.npy"], str(tmp_path))

    assert out is not batch
    assert np.array_equal(batch, original)
    # frame 0: only the 3x3 cell footprint survives
    assert out[0].sum() == 9.0
    assert np.all(out[0][2:5, 2:5] == 1.0)
    assert out[0][0, 0] == 0.0
    # frame 1: no mask file at all -> untouched
    assert np.array_equal(out[1], np.ones((8, 8), dtype=np.float32))


def test_apply_cell_mask_npy_suffix_fallback(tmp_path):
    """Filename without the .npy extension resolves via the +'.npy' retry."""
    from spacr.object import _apply_cell_mask

    batch = np.full((1, 6, 6), 4.0, dtype=np.float32)
    cell_mask = np.zeros((6, 6), dtype=np.int32)
    cell_mask[0:2, 0:2] = 1
    np.save(str(tmp_path / "frame"), cell_mask)   # writes frame.npy

    assert not os.path.exists(str(tmp_path / "frame"))
    out = _apply_cell_mask(batch, ["frame"], str(tmp_path))

    assert out[0].sum() == pytest.approx(16.0)    # 4 px * 4.0
    assert np.all(out[0][0:2, 0:2] == 4.0)
    assert out[0][5, 5] == 0.0


def test_apply_cell_mask_all_zero_mask_blanks_the_frame(tmp_path):
    from spacr.object import _apply_cell_mask

    batch = np.ones((1, 5, 5), dtype=np.float32)
    np.save(str(tmp_path / "empty.npy"), np.zeros((5, 5), dtype=np.int32))
    out = _apply_cell_mask(batch, ["empty.npy"], str(tmp_path))
    assert out.sum() == 0.0


# ---------------------------------------------------------------------------
# _load_unet_model
# ---------------------------------------------------------------------------

def test_load_unet_model_rejects_missing_path(tmp_path):
    from spacr.object import _load_unet_model

    bogus = str(tmp_path / "nope.pt")
    with pytest.raises(ValueError) as exc:
        _load_unet_model({"organelle_unet_model_path": bogus})
    assert bogus in str(exc.value)


def test_load_unet_model_rejects_none_path():
    from spacr.object import _load_unet_model

    with pytest.raises(ValueError) as exc:
        _load_unet_model({})
    assert "None" in str(exc.value)


def test_load_unet_model_loads_on_cpu_in_eval_mode(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from spacr.object import _load_unet_model

    # Force the CPU device branch regardless of the host machine.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    model = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1))
    model.train()
    assert model.training is True
    path = tmp_path / "unet.pt"
    torch.save(model, str(path))

    loaded = _load_unet_model({"organelle_unet_model_path": str(path)})

    assert loaded.training is False                     # .eval() was called
    params = list(loaded.parameters())
    assert params and all(p.device.type == "cpu" for p in params)
    with torch.no_grad():
        out = loaded(torch.zeros(1, 1, 8, 8))
    assert tuple(out.shape) == (1, 2, 8, 8)


# ---------------------------------------------------------------------------
# _segment_cellpose
# ---------------------------------------------------------------------------

def _cp_settings(**over):
    s = {
        "nucleus_channel": None, "cell_channel": None,
        "pathogen_channel": None, "organelle_channel": 0,
        "plot": True,                       # skips _check_masks by default
        "batch_size": 2,
        "organelle_diameter": 17,
        "organelle_FT": 0.35,
        "organelle_CP_prob": -1.5,
        "organelle_resample": True,
    }
    s.update(over)
    return s


def test_segment_cellpose_duplicates_channel_when_no_nucleus(tmp_path):
    """nucleus_channel is None -> ch1 falls back to ch0."""
    from spacr.object import _segment_cellpose

    rng = np.random.default_rng(0)
    batch = (rng.random((2, 16, 16, 1)) * 255).astype(np.float32)
    model = _RecordingCP()

    out = _segment_cellpose(batch, ["a.npy", "b.npy"], model,
                            _cp_settings(), "organelle", str(tmp_path))

    assert isinstance(out, list) and len(out) == 2
    assert all(m.shape == (16, 16) for m in out)
    assert out[0].max() == 1 and out[1].max() == 2

    x = model.calls[0]["x"]
    assert len(x) == 2
    assert x[0].shape[-1] == 2
    np.testing.assert_array_equal(x[0][..., 0], x[0][..., 1])


def test_segment_cellpose_forwards_eval_kwargs(tmp_path):
    from spacr.object import _segment_cellpose

    batch = np.zeros((1, 16, 16, 1), dtype=np.float32)
    batch[0, 4:9, 4:9, 0] = 200.0
    settings = _cp_settings(batch_size=5, organelle_diameter=23,
                            organelle_FT=0.9, organelle_CP_prob=0.75,
                            organelle_resample=False)
    model = _RecordingCP()

    _segment_cellpose(batch, ["a.npy"], model, settings,
                      "organelle", str(tmp_path))

    kw = model.calls[0]
    assert kw["batch_size"] == 5
    assert kw["diameter"] == 23
    assert kw["flow_threshold"] == 0.9
    assert kw["cellprob_threshold"] == 0.75
    assert kw["resample"] is False
    assert kw["normalize"] is False
    assert kw["channel_axis"] == -1
    assert kw["channels"] == [0, 1]     # what it passes; see the xfail below
    assert kw["rescale"] is None


@pytest.mark.xfail(strict=True, reason=(
    "spacr/object.py:1853 passes channels=[0, 1] to CellposeModel.eval. "
    "cellpose 4.0.7 logs 'channels deprecated in v4.0.1+. If data contain "
    "more than 3 channels, only the first 3 channels will be used' and never "
    "reads it, so the pair configures nothing -- the network takes the first "
    "three planes of whatever array it is handed. spacr/object.py:1913, the "
    "SAM sibling of this same function, already omits it. Fix: delete the "
    "channels=[0, 1] argument; spacr.model_compare.IGNORED_ARGUMENTS already "
    "documents 'channels' as this no-op."))
def test_segment_cellpose_does_not_pass_a_dead_channels_argument(tmp_path):
    """The hard-coded ``[0, 1]`` reaches nothing and must not be sent.

    It is also wrong on its own terms: ``_segment_cellpose`` is called for
    single-channel object types too, where a ``[0, 1]`` pair names a plane the
    batch does not have. Cellpose 4 ignoring it is the only reason that has
    never surfaced.
    """
    from spacr.object import _segment_cellpose

    batch = np.zeros((1, 16, 16, 1), dtype=np.float32)
    batch[0, 4:9, 4:9, 0] = 200.0
    model = _RecordingCP()

    _segment_cellpose(batch, ["a.npy"], model, _cp_settings(),
                      "organelle", str(tmp_path))

    configured = model.configured[0]
    dead = sorted(set(configured) & set(DEPRECATED_EVAL_ARGUMENTS))
    assert not dead, (
        "cellpose 4 accepts and then discards: "
        + ", ".join(f"{name}={configured[name]!r}" for name in dead)
    )


def test_segment_cellpose_remaps_and_clamps_organelle_channel(tmp_path, monkeypatch):
    """Raw channel ids are remapped onto the compacted stack, then clamped."""
    from spacr.object import _segment_cellpose

    _patch_prepare_identity(monkeypatch)

    # Two-channel compacted stack, but the raw organelle id (3) maps to
    # dense index 3 -> must clamp to shape[3]-1 == 1.
    batch = np.zeros((1, 8, 8, 2), dtype=np.float32)
    batch[..., 0] = 1.0
    batch[..., 1] = 2.0
    settings = _cp_settings(nucleus_channel=0, cell_channel=1,
                            pathogen_channel=2, organelle_channel=3)
    model = _RecordingCP()

    _segment_cellpose(batch, ["a.npy"], model, settings,
                      "organelle", str(tmp_path))

    x = model.calls[0]["x"][0]
    assert x.shape == (8, 8, 2)
    assert np.all(x[..., 0] == 2.0)     # clamped organelle channel (index 1)
    assert np.all(x[..., 1] == 1.0)     # nucleus channel (remapped 0 -> 0)


def test_segment_cellpose_uses_remapped_nucleus_channel(tmp_path, monkeypatch):
    from spacr.object import _segment_cellpose

    _patch_prepare_identity(monkeypatch)

    # Raw ids {nucleus: 3, organelle: 1} -> compacted [1, 3] -> {1:0, 3:1}
    batch = np.zeros((1, 8, 8, 2), dtype=np.float32)
    batch[..., 0] = 10.0      # organelle (raw 1 -> dense 0)
    batch[..., 1] = 20.0      # nucleus   (raw 3 -> dense 1)
    settings = _cp_settings(nucleus_channel=3, organelle_channel=1)
    model = _RecordingCP()

    _segment_cellpose(batch, ["a.npy"], model, settings,
                      "organelle", str(tmp_path))

    x = model.calls[0]["x"][0]
    assert np.all(x[..., 0] == 10.0)
    assert np.all(x[..., 1] == 20.0)


def test_segment_cellpose_ndim3_stacks_the_single_plane(tmp_path, monkeypatch):
    from spacr.object import _segment_cellpose

    _patch_prepare_identity(monkeypatch)

    rng = np.random.default_rng(3)
    batch = (rng.random((2, 12, 12)) * 100).astype(np.float32)
    settings = _cp_settings(organelle_channel=None, batch_size=1)
    model = _RecordingCP()

    out = _segment_cellpose(batch, ["a.npy", "b.npy"], model, settings,
                            "organelle", str(tmp_path))

    assert len(out) == 2
    x = model.calls[0]["x"][0]
    assert x.shape == (12, 12, 2)
    np.testing.assert_array_equal(x[..., 0], batch[0])
    np.testing.assert_array_equal(x[..., 1], batch[0])


def test_segment_cellpose_returns_none_when_batch_fully_filtered(
        tmp_path, monkeypatch):
    """Every mask already on disk -> _check_masks empties the batch."""
    from spacr.object import _segment_cellpose

    _patch_check_masks_empty(monkeypatch)
    batch = np.zeros((2, 8, 8, 1), dtype=np.float32)
    model = _RecordingCP()

    out = _segment_cellpose(batch, ["a.npy", "b.npy"], model,
                            _cp_settings(plot=False), "organelle",
                            str(tmp_path))

    assert out is None
    assert model.calls == []       # the model was never invoked


# ---------------------------------------------------------------------------
# _segment_cellpose_sam
# ---------------------------------------------------------------------------

def _sam_settings(object_type, **over):
    s = {
        "nucleus_channel": None, "cell_channel": None,
        "pathogen_channel": None, "organelle_channel": None,
        "plot": True,
        f"{object_type}_FT": 0.45,
        f"{object_type}_CP_prob": -0.25,
    }
    s.update(over)
    return s


def test_segment_cellpose_sam_cell_uses_cell_then_nucleus_order(
        tmp_path, monkeypatch):
    from spacr.object import _segment_cellpose_sam

    _patch_prepare_identity(monkeypatch)

    batch = np.zeros((1, 8, 8, 3), dtype=np.float32)
    batch[..., 0] = 1.0
    batch[..., 1] = 2.0
    batch[..., 2] = 3.0
    settings = _sam_settings("cell", cell_channel=2, nucleus_channel=0)
    model = _RecordingCP()

    out = _segment_cellpose_sam(batch, ["a.npy"], model, settings,
                                "cell", str(tmp_path))

    assert len(out) == 1
    x = model.calls[0]["x"][0]
    assert x.shape == (8, 8, 2)
    assert np.all(x[..., 0] == 3.0)      # cell channel first
    assert np.all(x[..., 1] == 1.0)      # nucleus channel second


def test_segment_cellpose_sam_cell_drops_none_nucleus(tmp_path, monkeypatch):
    from spacr.object import _segment_cellpose_sam

    _patch_prepare_identity(monkeypatch)

    batch = np.zeros((1, 8, 8, 2), dtype=np.float32)
    batch[..., 0] = 5.0
    batch[..., 1] = 6.0
    settings = _sam_settings("cell", cell_channel=1, nucleus_channel=None)
    model = _RecordingCP()

    _segment_cellpose_sam(batch, ["a.npy"], model, settings,
                          "cell", str(tmp_path))

    x = model.calls[0]["x"][0]
    assert x.shape == (8, 8, 1)
    assert np.all(x[..., 0] == 6.0)


def test_segment_cellpose_sam_forwards_object_type_scoped_kwargs(tmp_path):
    from spacr.object import _segment_cellpose_sam

    batch = np.zeros((3, 8, 8, 1), dtype=np.float32)
    batch[:, 2:6, 2:6, 0] = 90.0
    settings = _sam_settings("pathogen", pathogen_channel=0,
                             pathogen_FT=0.62, pathogen_CP_prob=1.25,
                             pathogen_resample=False)
    model = _RecordingCP()

    out = _segment_cellpose_sam(batch, ["a.npy", "b.npy", "c.npy"], model,
                                settings, "pathogen", str(tmp_path))

    assert len(out) == 3
    kw = model.calls[0]
    assert kw["batch_size"] == 3          # == len(batch_list), not settings
    assert kw["diameter"] is None
    assert kw["flow_threshold"] == 0.62
    assert kw["cellprob_threshold"] == 1.25
    assert kw["resample"] is False
    assert kw["normalize"] is False
    assert kw["channel_axis"] == -1
    # No channel pair reaches cellpose. eval(channels=) is deprecated in
    # v4.0.1+ and dropped, so leaving it at the library default is the only
    # honest way to call it -- and this path does.
    assert kw["channels"] is None
    assert "channels" not in model.configured[0]


def test_segment_cellpose_sam_resample_defaults_to_true(tmp_path):
    from spacr.object import _segment_cellpose_sam

    batch = np.zeros((1, 8, 8, 1), dtype=np.float32)
    settings = _sam_settings("nucleus", nucleus_channel=0)
    assert "nucleus_resample" not in settings
    model = _RecordingCP()

    _segment_cellpose_sam(batch, ["a.npy"], model, settings,
                          "nucleus", str(tmp_path))
    assert model.calls[0]["resample"] is True


def test_segment_cellpose_sam_ndim3_adds_channel_axis(tmp_path, monkeypatch):
    from spacr.object import _segment_cellpose_sam

    _patch_prepare_identity(monkeypatch)

    rng = np.random.default_rng(7)
    batch = (rng.random((2, 10, 10)) * 50).astype(np.float32)
    settings = _sam_settings("organelle", organelle_channel=0)
    model = _RecordingCP()

    out = _segment_cellpose_sam(batch, ["a.npy", "b.npy"], model, settings,
                                "organelle", str(tmp_path))

    assert len(out) == 2
    x = model.calls[0]["x"]
    assert x[0].shape == (10, 10, 1)
    np.testing.assert_array_equal(x[0][..., 0], batch[0])
    np.testing.assert_array_equal(x[1][..., 0], batch[1])


@pytest.mark.parametrize("shape", [(8, 8), (1, 2, 8, 8, 1)])
def test_segment_cellpose_sam_rejects_bad_ndim(tmp_path, shape):
    from spacr.object import _segment_cellpose_sam

    batch = np.zeros(shape, dtype=np.float32)
    settings = _sam_settings("nucleus", nucleus_channel=0)
    with pytest.raises(ValueError) as exc:
        _segment_cellpose_sam(batch, ["a.npy"], _RecordingCP(), settings,
                              "nucleus", str(tmp_path))
    assert "ndim" in str(exc.value)
    assert str(len(shape)) in str(exc.value)


def test_segment_cellpose_sam_rejects_out_of_bounds_channel(tmp_path):
    from spacr.object import _segment_cellpose_sam

    batch = np.zeros((1, 8, 8, 2), dtype=np.float32)
    settings = _sam_settings("nucleus", nucleus_channel=5)
    with pytest.raises(ValueError) as exc:
        _segment_cellpose_sam(batch, ["a.npy"], _RecordingCP(), settings,
                              "nucleus", str(tmp_path))
    msg = str(exc.value)
    assert "out of bounds" in msg
    assert "2 channels" in msg


def test_segment_cellpose_sam_unknown_object_type(tmp_path):
    from spacr.object import _segment_cellpose_sam

    batch = np.zeros((1, 8, 8, 1), dtype=np.float32)
    with pytest.raises(ValueError) as exc:
        _segment_cellpose_sam(batch, ["a.npy"], _RecordingCP(),
                              {"plot": True}, "mitochondrion", str(tmp_path))
    assert "Unsupported object_type" in str(exc.value)
    assert "mitochondrion" in str(exc.value)


def test_segment_cellpose_sam_no_channels_configured(tmp_path):
    from spacr.object import _segment_cellpose_sam

    batch = np.zeros((1, 8, 8, 1), dtype=np.float32)
    settings = _sam_settings("organelle")     # organelle_channel stays None
    with pytest.raises(ValueError) as exc:
        _segment_cellpose_sam(batch, ["a.npy"], _RecordingCP(), settings,
                              "organelle", str(tmp_path))
    assert "No valid channels" in str(exc.value)


def test_segment_cellpose_sam_returns_none_when_batch_fully_filtered(
        tmp_path, monkeypatch):
    from spacr.object import _segment_cellpose_sam

    _patch_check_masks_empty(monkeypatch, n_channels=1)
    batch = np.zeros((2, 8, 8, 1), dtype=np.float32)
    settings = _sam_settings("nucleus", nucleus_channel=0, plot=False)
    model = _RecordingCP()

    out = _segment_cellpose_sam(batch, ["a.npy", "b.npy"], model, settings,
                                "nucleus", str(tmp_path))

    assert out is None
    assert model.calls == []
