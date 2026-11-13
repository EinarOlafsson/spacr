"""Table-driven tests of the mask post-processing and organelle-input helpers in ``spacr.object``.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture:

* ``merge_split_filter_masks``: the no-op gate, the mask / intensity layouts
  it accepts and refuses (with exact messages), perimeter merging, the area,
  border and mean-intensity filters and the per-field progress line;
* ``_segment_cellpose``: the organelle / nucleus plane pair it builds from a
  compacted stack, the eval settings it forwards and the fully-done batch;
* ``_preprocess_batch``, ``_apply_cell_mask``, ``_load_unet_model``,
  ``_extract_classical_settings`` and ``_fill_cellpose_channel_positions``.

Everything is synthetic, tiny, CPU-only and offline.  The Cellpose double
declares the installed 4.0.7 ``eval`` signature
(``channel_axis=MISSING_CHANNEL_AXIS``, no ``**kwargs``) and passes every call
through :func:`tests.conftest.check_cellpose_eval_call`.
"""
from __future__ import annotations

import pickle
import re

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import spacr.object as O  # noqa: E402
from tests.cellpose_api_contract import (  # noqa: E402
    DEPRECATED_EVAL_ARGUMENTS,
    configured_eval_arguments,
    eval_arguments,
)
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    yield
    plt.close("all")


def _case_dir(tmp_path, name):
    d = tmp_path / re.sub(r"[^A-Za-z0-9_]+", "_", name)
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# merge_split_filter_masks
# ---------------------------------------------------------------------------

# Three disjoint 10x10 squares well clear of the border.
_BOXES = {1: (slice(5, 15), slice(5, 15)), 2: (slice(5, 15), slice(25, 35)),
          3: (slice(5, 15), slice(45, 55))}
_LOW_MID_HIGH = {1: 10.25, 2: 50.25, 3: 90.5}
_HIGH_MID_LOW = {1: 80.5, 2: 120.25, 3: 20.5}
_MEAN_BOUNDS = {"cell_min_intensity": 50.25, "cell_max_intensity": 90.5}


def _three_boxes():
    m = np.zeros((64, 64), dtype=np.int32)
    for label, sl in _BOXES.items():
        m[sl] = label
    return m


def _plane(values):
    """Float32 field whose box ``k`` holds ``values[k]`` and is 0 elsewhere."""
    img = np.zeros((64, 64), dtype=np.float32)
    for label, sl in _BOXES.items():
        img[sl] = values[label]
    return img


def _kept(*boxes):
    """The uint16 label image of the surviving boxes, relabelled 1..n."""
    out = np.zeros((64, 64), dtype=np.uint16)
    for new, old in enumerate(boxes, 1):
        out[_BOXES[old]] = new
    return out


def _with_extras():
    """Three boxes plus a border object, a 4 px speck and a 121 px block."""
    m = _three_boxes()
    m[0:5, 20:25] = 4
    m[25:27, 25:27] = 5
    m[30:41, 30:41] = 6
    return m


def _halves():
    """Two touching 20x10 halves of one 20x20 block."""
    m = np.zeros((64, 64), dtype=np.int32)
    m[20:40, 20:30] = 1
    m[20:40, 30:40] = 2
    return m


def _two_objects():
    m = np.zeros((32, 32), dtype=np.int32)
    m[2:10, 2:10] = 1
    m[16:26, 16:26] = 2
    return m


def _speck_and_block():
    m = np.zeros((32, 32), dtype=np.int32)
    m[0, 0] = 1
    m[10:20, 10:20] = 2
    return m


def test_merge_split_filter_masks_merges_filters_and_validates_layouts(
        subtests, monkeypatch):
    """Each enabled operation gives the exact relabelled masks, disabled ones
    hand the caller's object back, and bad layouts are refused by message."""
    import spacr.utils as U

    first, second = _plane(_LOW_MID_HIGH), _plane(_HIGH_MID_LOW)
    off = {"cell_perimeter_fraction": 0, "cell_min_area": 0, "cell_max_area": 0,
           "cell_remove_border_objects": False, "cell_min_intensity": 0,
           "cell_max_intensity": 0}
    block_only = np.zeros((32, 32), np.uint16)
    block_only[10:20, 10:20] = 1
    deep = np.zeros((1, 64, 64, 16))
    deep[0, ..., 0] = first
    same = object()
    cases = [
        # (id, masks, intensity, settings, object_type, expected)
        # expected: ``same`` (the masks object itself), None, a list of label
        # images, or (exception message fragment,)
        ("every operation off hands the masks back", _three_boxes(), None, off, "cell", same),
        ("an empty settings dict hands the masks back", _two_objects(),
         _two_objects().astype(float), {}, "cell", same),
        ("a percentile lower bound alone does nothing", _three_boxes(), first,
         {"cell_min_intensity_percentile": 50}, "cell", same),
        ("a percentile upper bound alone does nothing", _three_boxes(), first,
         {"cell_max_intensity_percentile": 50}, "cell", same),
        ("a percentile band drops nothing when work runs", _three_boxes(), first,
         {"cell_min_area": 1, "cell_min_intensity_percentile": 50,
          "cell_max_intensity_percentile": 50}, "cell", [_kept(1, 2, 3)]),
        ("another object type's settings are ignored", _three_boxes(), None,
         {"nucleus_min_area": 500}, "cell", same),
        ("the object type's own prefix is read", _three_boxes(), _plane({1: 1, 2: 1, 3: 1}),
         {"nucleus_min_area": 500}, "nucleus", [np.zeros((64, 64), np.uint16)]),
        ("no masks with work enabled is None", None, None, {"cell_min_area": 5}, "cell",
         None),
        ("min bound keeps equality", _three_boxes(), first,
         {"cell_min_intensity": 50.25}, "cell", [_kept(2, 3)]),
        ("max bound keeps equality", _three_boxes(), first,
         {"cell_max_intensity": 50.25}, "cell", [_kept(1, 2)]),
        ("a min bound above every mean keeps none", _three_boxes(), first,
         {"cell_min_intensity": 91}, "cell", [_kept()]),
        ("a max bound above every mean keeps all", _three_boxes(), first,
         {"cell_max_intensity": 100}, "cell", [_kept(1, 2, 3)]),
        ("area and border filters need no intensity image", _with_extras(), None,
         {"cell_min_area": 10, "cell_max_area": 100, "cell_remove_border_objects": True,
          "cell_min_intensity": 0, "cell_max_intensity": 0}, "cell", [_kept(1, 2, 3)]),
        ("min area drops a one-pixel speck", _speck_and_block(),
         np.random.default_rng(0).random((32, 32)).astype(np.float32),
         {"cell_min_area": 5}, "cell", [block_only]),
        ("a 3-D batch is one output per field", np.stack([_two_objects()] * 2),
         np.random.default_rng(1).random((2, 32, 32)).astype(np.float32),
         {"cell_min_area": 3}, "cell", [_two_objects().astype(np.uint16)] * 2),
        ("the misspelled perimiter key still merges", _halves(), None,
         {"cell_perimiter_fraction": 0.1}, "cell",
         [np.where(_halves() > 0, 1, 0).astype(np.uint16)]),
        ("without a merge key touching halves stay two", _halves(), None,
         {"cell_min_area": 1}, "cell", [_halves().astype(np.uint16)]),
        ("4-D channel-last intensity uses channel 0 per field",
         np.stack([_three_boxes()] * 2),
         np.stack([np.stack([first, second], -1), np.stack([second, first], -1)]),
         dict(_MEAN_BOUNDS), "cell", [_kept(2, 3), _kept(1)]),
        ("a single 4-D field equals its 3-D equivalent", _three_boxes()[None],
         np.stack([first, second], -1)[None], dict(_MEAN_BOUNDS), "cell", [_kept(2, 3)]),
        ("a 3-D single field", _three_boxes()[None], first[None], dict(_MEAN_BOUNDS),
         "cell", [_kept(2, 3)]),
        ("sixteen channels stay channel-last", _three_boxes()[None], deep,
         dict(_MEAN_BOUNDS), "cell", [_kept(2, 3)]),
        ("lists keep field order", [_three_boxes(), _three_boxes()], [first, second],
         dict(_MEAN_BOUNDS), "cell", [_kept(2, 3), _kept(1)]),
        ("a channel-first field is refused", _three_boxes()[None],
         np.zeros((1, 2, 64, 64)), dict(_MEAN_BOUNDS), "cell",
         ("same shape as the mask",)),
        ("enabled mean bounds need the intensity image", _three_boxes(), None,
         dict(_MEAN_BOUNDS), "cell", ("same shape as the mask",)),
        ("4-D field counts must match", np.stack([_three_boxes()] * 2),
         np.zeros((3, 64, 64, 2), np.float32), {"cell_min_area": 5}, "cell",
         ("Number of masks (2) does not match number of intensity images (3).",)),
        ("list field counts must match", [_two_objects(), _two_objects()],
         [np.zeros((32, 32), np.float32)], {"cell_min_area": 3}, "cell",
         ("Number of masks (2) does not match number of intensity images (1).",)),
        ("1-D intensity refused", _three_boxes(), np.zeros(8, np.float32),
         {"cell_min_area": 5}, "cell", ("Unsupported intensity_images ndim: 1",)),
        ("5-D intensity refused for its rank, not its length", [_three_boxes()],
         np.zeros((1, 2, 2, 2, 2), np.float32), {"cell_min_area": 5}, "cell",
         ("Unsupported intensity_images ndim: 5",)),
        ("4-D masks refused", np.zeros((2, 4, 4, 4), np.int32),
         np.zeros((2, 4, 4), np.float32), {"cell_min_area": 3}, "cell",
         ("Unsupported masks ndim: 4",)),
        ("5-D masks refused", np.zeros((2, 2, 2, 2, 2), np.int32),
         np.zeros((2, 2, 2, 2, 2), np.int32), {"cell_min_area": 3}, "cell",
         ("Unsupported masks ndim: 5",)),
    ]
    for case_id, masks, intensity, settings, object_type, expected in cases:
        with subtests.test(msg=case_id):
            snapshot = None if masks is None else [np.array(m, copy=True) for m in masks]
            if isinstance(expected, tuple):
                with pytest.raises(ValueError) as excinfo:
                    O.merge_split_filter_masks(masks, intensity, settings, object_type)
                assert expected[0] in str(excinfo.value)
                assert "does not match" not in str(excinfo.value) or "Number" in expected[0]
                continue
            out = O.merge_split_filter_masks(masks, intensity, settings, object_type)
            if expected is same:
                assert out is masks
            elif expected is None:
                assert out is None
            else:
                assert isinstance(out, list) and len(out) == len(expected)
                for got, want in zip(out, expected):
                    assert got.dtype == np.uint16 and got.shape == want.shape
                    np.testing.assert_array_equal(got, want)
            if snapshot is not None:
                for before, after in zip(snapshot, masks):
                    np.testing.assert_array_equal(before, after)

    with subtests.test(msg="progress is reported once per named field, in order"):
        progress = []
        monkeypatch.setattr(U, "print_progress", lambda current, total, **kw: progress.append(
            (current, total, kw["n_jobs"], kw["batch_size"], kw["operation_type"],
             len(kw["time_ls"]), all(t >= 0 for t in kw["time_ls"]))))
        out = O.merge_split_filter_masks(
            [_three_boxes(), _three_boxes()], [first, second], dict(_MEAN_BOUNDS), "cell",
            batch_filenames=["fov_a.npy", "fov_b.npy"])
        assert [o.tolist() for o in out] == [_kept(2, 3).tolist(), _kept(1).tolist()]
        assert progress == [(1, 2, 1, None, "merge_cell", 1, True),
                            (2, 2, 1, None, "merge_cell", 2, True)]


# ---------------------------------------------------------------------------
# _segment_cellpose
# ---------------------------------------------------------------------------

class _RecordingCP:
    """``CellposeModel.eval`` double (4.0.7 signature); image ``i`` gets one
    3x3 object labelled ``i + 1``."""

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


def _cp_settings(**over):
    s = {"nucleus_channel": None, "cell_channel": None, "pathogen_channel": None,
         "organelle_channel": 0, "plot": True, "batch_size": 2,
         "organelle_diameter": 17, "organelle_flow_threshold": 0.35,
         "organelle_cellprob_threshold": -1.5, "organelle_resample": True}
    s.update(over)
    return s


def _const(shape, values):
    b = np.zeros(shape, dtype=np.float32)
    for k, v in enumerate(values):
        b[..., k] = v
    return b


def test_segment_cellpose_pairs_the_organelle_plane_with_the_nucleus_plane(
        subtests, tmp_path, monkeypatch):
    """The organelle plane (remapped, clamped) is paired with the remapped
    nucleus plane or itself, eval gets the organelle settings, and a batch
    whose masks all exist is None."""
    import spacr.io as IO
    import spacr.utils as U

    one = (np.random.default_rng(0).random((2, 16, 16, 1)) * 255).astype(np.float32)
    flat3 = (np.random.default_rng(3).random((2, 12, 12)) * 100).astype(np.float32)
    square = np.zeros((1, 16, 16, 1), np.float32)
    square[0, 4:9, 4:9, 0] = 200.0
    rec_eval = {"batch_size": 2, "diameter": 17, "flow_threshold": 0.35,
                "cellprob_threshold": -1.5, "resample": True, "normalize": False,
                "channel_axis": -1, "rescale": None}
    cases = [
        # (id, batch, settings overrides, identity prepare, expected planes of x[0],
        #  expected eval kwargs)
        ("no nucleus: the organelle plane twice", one, {}, True,
         np.concatenate([one[0], one[0]], -1), rec_eval),
        ("organelle settings reach eval", square,
         dict(batch_size=5, organelle_diameter=23, organelle_flow_threshold=0.9,
              organelle_cellprob_threshold=0.75, organelle_resample=False), True,
         np.concatenate([square[0], square[0]], -1),
         dict(rec_eval, batch_size=5, diameter=23, flow_threshold=0.9,
              cellprob_threshold=0.75, resample=False)),
        ("an out-of-range organelle position is clamped", _const((1, 8, 8, 2), (1, 2)),
         dict(nucleus_channel=0, cell_channel=1, pathogen_channel=2, organelle_channel=3),
         True, _const((8, 8, 2), (2, 1)), rec_eval),
        ("raw channels are remapped onto the compacted stack", _const((1, 8, 8, 2), (10, 20)),
         dict(nucleus_channel=3, organelle_channel=1), True,
         _const((8, 8, 2), (10, 20)), rec_eval),
        ("organelle then nucleus in remapped order",
         _const((1, 8, 8, 2), (10, 20)), dict(nucleus_channel=0, organelle_channel=7),
         True, _const((8, 8, 2), (20, 10)), rec_eval),
        ("a nucleus beyond the stack falls back to the organelle plane",
         _const((1, 8, 8, 2), (10, 20)),
         dict(nucleus_channel=3, cell_channel=1, pathogen_channel=2, organelle_channel=0),
         True, _const((8, 8, 2), (10, 10)), rec_eval),
        ("a 3-D batch stacks the single plane twice", flat3,
         dict(organelle_channel=None, batch_size=1), True,
         np.stack([flat3[0], flat3[0]], -1), dict(rec_eval, batch_size=1)),
        ("a 3-D batch without an organelle channel, resample off",
         flat3[:1], dict(organelle_channel=None, nucleus_channel=None, batch_size=1,
                         organelle_resample=False, organelle_diameter=15), True,
         np.stack([flat3[0], flat3[0]], -1),
         dict(rec_eval, batch_size=1, diameter=15, resample=False)),
    ]
    for case_id, batch, over, identity, planes, want_eval in cases:
        with subtests.test(msg=case_id):
            model = _RecordingCP()
            names = [f"f{i}.npy" for i in range(len(batch))]
            with monkeypatch.context() as mp:
                if identity:
                    mp.setattr(U, "prepare_batch_for_segmentation", lambda b: b)
                out = O._segment_cellpose(batch, names, model, _cp_settings(**over),
                                          "organelle", str(_case_dir(tmp_path, case_id)))
            assert len(model.calls) == 1
            call = model.calls[0]
            assert len(call["x"]) == len(batch)
            np.testing.assert_array_equal(call["x"][0], planes)
            for key, value in want_eval.items():
                assert call[key] == value, key
            assert not set(model.configured[0]) & set(DEPRECATED_EVAL_ARGUMENTS)
            assert [m.shape for m in out] == [batch.shape[1:3]] * len(batch)
            assert [sorted(np.unique(m).tolist()) for m in out] == [
                [0, i + 1] for i in range(len(batch))]

    with subtests.test(msg="the real normalisation still pairs equal planes"):
        model = _RecordingCP()
        out = O._segment_cellpose(one, ["a.npy", "b.npy"], model, _cp_settings(),
                                  "organelle", str(_case_dir(tmp_path, "real")))
        x0 = model.calls[0]["x"][0]
        assert x0.shape == (16, 16, 2)
        np.testing.assert_array_equal(x0[..., 0], x0[..., 1])
        assert len(out) == 2

    with subtests.test(msg="a batch whose masks all exist is None and never evaluated"):
        model = _RecordingCP()
        seen = []
        with monkeypatch.context() as mp:
            def _check(b, f, o, resume=False):
                seen.append((b.shape, list(f), resume))
                return np.zeros((0, 8, 8, 2), np.float32), []
            mp.setattr(IO, "_check_masks", _check)
            out = O._segment_cellpose(np.zeros((2, 8, 8, 1), np.float32),
                                      ["a.npy", "b.npy"], model,
                                      _cp_settings(plot=False, resume=True), "organelle",
                                      str(_case_dir(tmp_path, "done")))
        assert out is None
        assert model.calls == []
        assert seen == [((2, 8, 8, 2), ["a.npy", "b.npy"], True)]


# ---------------------------------------------------------------------------
# _preprocess_batch
# ---------------------------------------------------------------------------

def _sloped(n=2, size=32, base=0.5):
    """Float32 batch: linear background ramp + one bright square per frame."""
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    ramp = base + (yy + xx) / (2.0 * size)
    frames = []
    for i in range(n):
        img = ramp.copy()
        img[8 + i:16 + i, 8 + i:16 + i] += 2.0
        frames.append(img)
    return np.stack(frames).astype(np.float32)


def _expected_preprocess(batch, rolling=None, clahe=None):
    """Independent per-frame rolling-ball / CLAHE reference."""
    from skimage.exposure import equalize_adapthist
    from skimage.restoration import rolling_ball

    out = batch.copy()
    for i, img in enumerate(batch):
        if rolling is not None:
            img = np.clip(img - rolling_ball(img, radius=rolling), 0, None)
        if clahe is not None:
            lo, hi = np.percentile(img, (0.5, 99.5))
            norm = np.clip((img - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(img)
            img = equalize_adapthist(norm, clip_limit=clahe).astype(np.float32)
        out[i] = img
    return out


def test_preprocess_batch_applies_rolling_ball_then_clahe(subtests):
    """No option hands the batch back; rolling ball and CLAHE give the exact
    per-frame result into a copy, and a flat frame never divides by zero."""
    sloped = _sloped()
    flat = np.full((2, 32, 32), 3.5, np.float32)
    mixed = np.stack([np.full((32, 32), 0.25, np.float32), _sloped(n=1)[0]])
    cases = [
        # (id, batch, settings, rolling radius, clahe clip) ; None radius/clip = off
        ("nothing requested", sloped, {"organelle_rolling_ball": False,
                                       "organelle_clahe": False}, "same", None),
        ("empty settings", sloped, {}, "same", None),
        ("rolling ball", sloped, {"organelle_rolling_ball": True,
                                  "organelle_rolling_ball_radius": 6}, 6, None),
        ("rolling ball default radius", sloped, {"organelle_rolling_ball": True}, 50, None),
        ("rolling ball then CLAHE", sloped,
         {"organelle_rolling_ball": True, "organelle_rolling_ball_radius": 6,
          "organelle_clahe": True, "organelle_clahe_clip_limit": 0.02}, 6, 0.02),
        ("CLAHE default clip on a flat batch", flat, {"organelle_clahe": True}, None, 0.01),
        ("CLAHE on a flat and a textured frame", mixed, {"organelle_clahe": True}, None, 0.01),
    ]
    for case_id, batch, settings, radius, clip in cases:
        with subtests.test(msg=case_id):
            original = batch.copy()
            with np.errstate(divide="raise", invalid="raise"):
                out = O._preprocess_batch(batch, settings)
            if radius == "same":
                assert out is batch
                continue
            assert out is not batch
            np.testing.assert_array_equal(batch, original)
            assert out.dtype == np.float32 and out.shape == batch.shape
            np.testing.assert_allclose(out, _expected_preprocess(batch, radius, clip),
                                       rtol=0, atol=1e-6)
            assert np.isfinite(out).all() and out.min() >= 0.0
            if clip is not None:
                assert out.max() <= 1.0 + 1e-6
            if case_id.startswith("rolling ball") and clip is None:
                assert out[0][0, 0] < batch[0][0, 0] and out[0][10, 10] > out[0][0, 0]
            if "flat" in case_id:
                assert out[0].min() == out[0].max()
            if "textured" in case_id:
                assert out[1].max() > out[1].min()


# ---------------------------------------------------------------------------
# _apply_cell_mask
# ---------------------------------------------------------------------------

def test_apply_cell_mask_zeroes_pixels_outside_the_saved_cell_mask(subtests, tmp_path):
    """Pixels outside a field's cell mask (found by exact name or name + .npy)
    become 0 in a copy; a field without a mask is untouched."""
    def mask(shape, *slices, label=7):
        m = np.zeros(shape, np.int32)
        for sl in slices:
            m[sl] = label
        return m

    cases = [
        # (id, batch, {saved file name: mask}, batch filenames, expected output)
        ("exact name; a missing mask leaves its field alone", np.ones((2, 8, 8), np.float32),
         {"f0.npy": mask((8, 8), np.s_[2:5, 2:5])}, ["f0.npy", "absent.npy"],
         np.stack([mask((8, 8), np.s_[2:5, 2:5], label=1), np.ones((8, 8))])),
        ("a name without .npy finds name.npy", np.full((1, 6, 6), 4.0, np.float32),
         {"frame.npy": mask((6, 6), np.s_[0:2, 0:2], label=1)}, ["frame"],
         mask((6, 6), np.s_[0:2, 0:2], label=4)[None]),
        ("exact and suffix lookups in one batch", np.ones((2, 8, 8), np.float32),
         {"img0.npy": mask((8, 8), np.s_[2:5, 2:5]), "img1.npy": mask((8, 8), np.s_[0:2, 0:2])},
         ["img0.npy", "img1"],
         np.stack([mask((8, 8), np.s_[2:5, 2:5], label=1),
                   mask((8, 8), np.s_[0:2, 0:2], label=1)])),
        ("an all-zero mask blanks the field", np.ones((1, 5, 5), np.float32),
         {"empty.npy": np.zeros((5, 5), np.int32)}, ["empty.npy"], np.zeros((1, 5, 5))),
        ("no mask anywhere", np.ones((1, 4, 4), np.float32), {}, ["nope"], np.ones((1, 4, 4))),
    ]
    for case_id, batch, saved, names, expected in cases:
        with subtests.test(msg=case_id):
            folder = _case_dir(tmp_path, case_id)
            for name, m in saved.items():
                np.save(folder / name, m)
            original = batch.copy()
            out = O._apply_cell_mask(batch, names, str(folder))
            assert out is not batch
            np.testing.assert_array_equal(batch, original)
            assert out.dtype == batch.dtype
            np.testing.assert_array_equal(out, expected.astype(np.float32))


# ---------------------------------------------------------------------------
# _load_unet_model
# ---------------------------------------------------------------------------

def test_load_unet_model_loads_a_saved_module_in_eval_mode_or_refuses(
        subtests, tmp_path, monkeypatch):
    """A saved module comes back on the CPU in eval mode computing what it
    computed before; a missing or unset path is refused by message."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("SPACR_DEVICE", "cpu")
    monkeypatch.setattr(O.accelerator, "torch_device", lambda: torch.device("cpu"))
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1))
    model.train()
    saved = tmp_path / "unet.pt"
    torch.save(model, str(saved))
    probe = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    with torch.no_grad():
        want = model(probe)

    refusals = [
        ("a path that does not exist", {"organelle_unet_model_path": str(tmp_path / "nope.pt")},
         f"got '{tmp_path / 'nope.pt'}'"),
        ("no path at all", {}, "got 'None'"),
        ("an explicit None", {"organelle_unet_model_path": None}, "got 'None'"),
    ]
    for case_id, settings, tail in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError) as excinfo:
                O._load_unet_model(settings)
            assert str(excinfo.value) == (
                "organelle_unet_model_path must point to a valid .pt/.pth file, " + tail)

    with subtests.test(msg="a saved module loads"):
        loaded = O._load_unet_model({"organelle_unet_model_path": str(saved)})
        assert isinstance(loaded, nn.Sequential)
        assert loaded.training is False
        assert {p.device.type for p in loaded.parameters()} == {"cpu"}
        with torch.no_grad():
            got = loaded(probe)
        assert tuple(got.shape) == (1, 2, 8, 8)
        assert torch.equal(got, want)


# ---------------------------------------------------------------------------
# _extract_classical_settings
# ---------------------------------------------------------------------------

_CLASSICAL = {
    "organelle_morphology": "ring", "organelle_method": "dog",
    "organelle_min_area": 7, "organelle_max_area": 900,
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
    "organelle_ring_min_prominence": 0.07, "organelle_ring_fill_method": "convex",
}
_NOT_CLASSICAL = {
    "organelle_model_name": "cpsam", "organelle_diameter": 30, "organelle_resample": True,
    "organelle_channel": 2, "src": "/some/where", "batch_size": 8, "plot": True,
    "organelle_progress_callback": lambda *a: None,
}


def test_extract_classical_settings_keeps_exactly_the_whitelisted_keys(subtests):
    """The result is the whitelisted keys that are present, unchanged, in a new
    dict that survives a pickle round trip."""
    cases = [
        ("every classical key among others", dict(_CLASSICAL, **_NOT_CLASSICAL), _CLASSICAL),
        ("only the keys that are present", {"organelle_morphology": "irregular",
                                            "unrelated": 1},
         {"organelle_morphology": "irregular"}),
        ("nothing classical", dict(_NOT_CLASSICAL), {}),
        ("empty settings", {}, {}),
    ]
    for case_id, settings, expected in cases:
        with subtests.test(msg=case_id):
            before = dict(settings)
            out = O._extract_classical_settings(settings)
            assert out == expected
            assert out is not settings
            assert settings == before
            assert pickle.loads(pickle.dumps(out)) == expected


# ---------------------------------------------------------------------------
# _fill_cellpose_channel_positions
# ---------------------------------------------------------------------------

def test_fill_cellpose_channel_positions_recomputes_every_derived_alias(subtests):
    """Each ``cellpose_<role>_channel`` becomes the role's dense position in
    role order; a disabled or non-numeric role loses its stale alias."""
    cases = [
        # (id, settings, expected cellpose_* aliases afterwards)
        ("numeric and text channels, shared positions",
         {"nucleus_channel": 5, "cell_channel": "2", "pathogen_channel": 5,
          "organelle_channel": 7},
         {"cellpose_nucleus_channel": 0, "cellpose_cell_channel": 1,
          "cellpose_pathogen_channel": 0, "cellpose_organelle_channel": 2}),
        ("a stale alias is overwritten",
         {"nucleus_channel": 5, "cell_channel": "2", "cellpose_cell_channel": 9},
         {"cellpose_nucleus_channel": 0, "cellpose_cell_channel": 1}),
        ("disabled roles clear, numbered organelle slots refresh",
         {"nucleus_channel": 3, "cell_channel": 3, "pathogen_channel": None,
          "organelle_channel": "invalid", "organelleb_channel": 2,
          "cellpose_pathogen_channel": 9, "cellpose_organelle_channel": 7,
          "cellpose_organelleb_channel": 99},
         {"cellpose_nucleus_channel": 0, "cellpose_cell_channel": 0,
          "cellpose_organelleb_channel": 1}),
        ("a channel that is not a number is skipped",
         {"cell_channel": "rgb", "cellpose_cell_channel": 5, "nucleus_channel": 2},
         {"cellpose_nucleus_channel": 0}),
        ("nothing configured", {"cellpose_nucleus_channel": 4}, {}),
    ]
    for case_id, settings, expected in cases:
        with subtests.test(msg=case_id):
            raw = {k: v for k, v in settings.items() if not k.startswith("cellpose_")}
            assert O._fill_cellpose_channel_positions(settings) is settings
            assert {k: v for k, v in settings.items() if k.startswith("cellpose_")} == expected
            assert {k: v for k, v in settings.items()
                    if not k.startswith("cellpose_")} == raw
