"""588: filling holes keeps touching cells apart, and Apply segments as the preview does.

Two faults found while recording the Train Cellpose tutorial:

* ``spacr.utils.fill_holes_in_mask`` labelled the mask afresh with
  ``ndimage.label`` before filling, so every pair of touching cells became one
  object. Apply's ``fill_in`` defaults to True, and the tutorial's batch masks
  came out as 8/4/6 objects where the preview showed 74. Make Masks' "Fill
  holes" button had the same relabelling.
* Apply's defaults (``channels=[0, 0]``, ``percentiles=None``) went through a
  set-wide percentile search that averaged empty channel slots into numpy
  "Mean of empty slice" warnings, while the preview lets Cellpose normalise
  the one raw plane itself.
"""
from __future__ import annotations

import io
import os
import time
import warnings
import zipfile

import numpy as np
import pytest
import tifffile
from scipy import ndimage

from spacr.qt import mask_engine as engine
from spacr.utils import fill_holes_in_mask

ZIP = os.path.join(os.path.dirname(__file__), os.pardir, "docs", "source", "_extra",
                   "tutorials", "examples", "Cellpose_training_images_masks.zip")


def _touching_ringed_cells():
    """Two square cells sharing an edge, each with a hole of its own."""
    mask = np.zeros((20, 30), np.uint16)
    mask[2:14, 2:14] = 5
    mask[2:14, 14:26] = 9
    mask[6:9, 6:9] = 0
    mask[5:10, 18:22] = 0
    return mask


def _count(mask):
    return int(len(np.unique(mask)) - (1 if (mask == 0).any() else 0))


@pytest.mark.parametrize("fill", [
    fill_holes_in_mask,
    engine.fill_holes,
    lambda m: engine.fill_holes(m, preserve_ids=True),
    engine.fill_label_holes,
], ids=["utils.fill_holes_in_mask", "engine.fill_holes", "engine.fill_holes-exact",
        "engine.fill_label_holes"])
def test_touching_cells_keep_both_ids_and_their_holes_are_filled(fill):
    mask = _touching_ringed_cells()
    before = mask.copy()
    filled = fill(mask)
    np.testing.assert_array_equal(mask, before)
    assert set(np.unique(filled).tolist()) == {0, 5, 9}
    assert (filled[6:9, 6:9] == 5).all()
    assert (filled[5:10, 18:22] == 9).all()
    assert (filled[2:14, 2:14] == 5).all() and (filled[2:14, 14:26] == 9).all()
    assert (filled == 0).sum() == (before == 0).sum() - 9 - 20


def test_a_hole_never_takes_a_neighbour_and_the_inner_ring_owns_its_hole():
    mask = np.zeros((21, 21), np.int32)
    mask[1:20, 1:20] = 3
    mask[3:18, 3:18] = 0
    mask[6:15, 6:15] = 7
    mask[8:13, 8:13] = 0
    mask[10, 10] = 11
    filled = fill_holes_in_mask(mask)
    assert filled[10, 10] == 11
    assert filled[9, 9] == 7
    assert filled[4, 4] == 3
    assert (filled[mask > 0] == mask[mask > 0]).all()
    assert filled.dtype == mask.dtype


def test_a_boolean_or_empty_mask_is_still_handled():
    ring = np.zeros((9, 9), bool)
    ring[1:8, 1:8] = True
    ring[3:6, 3:6] = False
    assert set(np.unique(fill_holes_in_mask(ring)).tolist()) == {0, 1}
    assert (fill_holes_in_mask(ring)[4, 4] == 1)
    empty = np.zeros((6, 6), np.uint16)
    assert not fill_holes_in_mask(empty).any()


def test_filling_a_large_field_of_touching_objects_is_fast():
    side, step = 2000, 100
    mask = np.zeros((side, side), np.int32)
    label = 0
    for y in range(0, side - step + 1, step):
        for x in range(0, side - step + 1, step):
            label += 1
            mask[y:y + step, x:x + step] = label
            mask[y + 30:y + 60, x + 30:x + 60] = 0
    assert label == 400
    start = time.perf_counter()
    filled = fill_holes_in_mask(mask)
    elapsed = time.perf_counter() - start
    assert _count(filled) == 400
    assert not (filled == 0).any()
    assert elapsed < 5.0, f"filling 400 objects on 2000 x 2000 took {elapsed:.2f}s"


def _tutorial_masks():
    if not os.path.exists(ZIP):
        pytest.skip("the Train Cellpose tutorial ZIP is not in this checkout")
    with zipfile.ZipFile(ZIP) as archive:
        return {name: tifffile.imread(io.BytesIO(archive.read(name)))
                for name in sorted(archive.namelist())
                if name.startswith("training/masks/") and name.endswith(".tif")}


def test_real_curated_fields_keep_every_object_after_fill_in():
    masks = _tutorial_masks()
    assert masks
    merged_by_relabelling = 0
    for name, mask in masks.items():
        ids = _count(mask)
        merged_by_relabelling += ids - ndimage.label(mask > 0)[1]
        for fill in (fill_holes_in_mask, engine.fill_holes):
            filled = fill(mask)
            assert _count(filled) == ids, name
            kept = mask > 0
            np.testing.assert_array_equal(filled[kept], mask[kept])
    assert merged_by_relabelling > 200


def test_apply_takes_the_preview_channels_and_normalisation_when_unset():
    from spacr.spacr_cellpose import _apply_input_settings
    assert _apply_input_settings([0, 0], None) == ([0], None, False)
    assert _apply_input_settings([1, 0, 1], None) == ([1, 0], None, False)
    assert _apply_input_settings(None, [2, 99]) == (None, [2, 99], True)
    assert _apply_input_settings([0], ["a", "b"]) == ([0], None, False)


def test_the_percentile_search_no_longer_averages_empty_channels(tmp_path):
    from spacr.io import _load_normalized_images_and_labels
    image = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 3000) + 100
    path = str(tmp_path / "field.tif")
    tifffile.imwrite(path, image)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _load_normalized_images_and_labels([path], None, channels=[0, 0],
                                                 percentiles=None, background=100,
                                                 Signal_to_noise=10)[0]
        raw = _load_normalized_images_and_labels([path], None, rescale=False)[0]
    assert out[0].shape == (64, 64, 1) and 0 <= out[0].min() < out[0].max() <= 1
    np.testing.assert_array_equal(raw[0][..., 0], image)


class _RecordingModel:
    """Stands in for Cellpose and records what Apply hands it."""

    calls = []

    def __init__(self, *args, **kwargs):
        self.pretrained_model = kwargs.get("pretrained_model")

    def eval(self, x, **kwargs):
        _RecordingModel.calls.append((np.array(x, copy=True), kwargs))
        mask = np.zeros(np.asarray(x).shape[:2], np.uint16)
        mask[2:10, 2:10] = 1
        mask[2:10, 10:18] = 2
        mask[4:7, 4:7] = 0
        return mask, [np.zeros((2,) + mask.shape)], None


@pytest.mark.parametrize("extra,normalised_by_cellpose", [
    ({}, True),
    ({"percentiles": [2, 99]}, False),
])
def test_apply_hands_cellpose_the_preview_input_and_keeps_touching_ids(
        tmp_path, monkeypatch, extra, normalised_by_cellpose):
    from spacr import spacr_cellpose

    monkeypatch.setattr(spacr_cellpose.cp_models, "CellposeModel", _RecordingModel)
    monkeypatch.setattr("spacr.accelerator.cellpose_gpu", lambda: False)
    _RecordingModel.calls = []
    image = ((np.arange(32 * 32).reshape(32, 32) * 7) % 4000 + 100).astype(np.uint16)
    tifffile.imwrite(str(tmp_path / "field.tif"), image)
    settings = {"src": str(tmp_path), "save": True, **extra}
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        spacr_cellpose.identify_masks_finetune(settings)
    (x, kwargs), = _RecordingModel.calls
    assert kwargs["normalize"] is normalised_by_cellpose
    assert kwargs["channel_axis"] is None and x.ndim == 2
    if normalised_by_cellpose:
        np.testing.assert_array_equal(x, image)
    else:
        assert 0 <= x.min() and x.max() <= 1
    saved = tifffile.imread(str(tmp_path / "masks" / "field.tif"))
    assert set(np.unique(saved).tolist()) == {0, 1, 2}
    assert (saved[4:7, 4:7] == 1).all()


def _stock_cellpose_available():
    try:
        import cellpose  # noqa: F401
    except Exception:
        return False
    return os.path.exists(os.path.expanduser("~/.cellpose/models/cpsam"))


@pytest.mark.slow
@pytest.mark.heavy
@pytest.mark.skipif(not _stock_cellpose_available(), reason="needs Cellpose and the cpsam weights")
def test_apply_with_defaults_finds_the_preview_count_on_a_real_field(tmp_path, monkeypatch):
    """Stock cpsam on CPU, one real tutorial field cropped to 256 x 256."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr("spacr.accelerator.cellpose_gpu", lambda: False)
    if not os.path.exists(ZIP):
        pytest.skip("the Train Cellpose tutorial ZIP is not in this checkout")
    with zipfile.ZipFile(ZIP) as archive:
        field = tifffile.imread(io.BytesIO(archive.read("apply/cell_field_01.tif")))
    crop = np.ascontiguousarray(field[128:384, 128:384])
    tifffile.imwrite(str(tmp_path / "cell_field_01.tif"), crop)

    from spacr.qt.widgets.live_preview import PreviewRequest, _segment_multi
    preview, _flows = _segment_multi(PreviewRequest(image=crop, model="cpsam"))
    preview_count = _count(preview["cell"])

    from spacr.spacr_cellpose import identify_masks_finetune
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        identify_masks_finetune({"src": str(tmp_path), "save": True, "model_name": "cpsam"})
    applied = tifffile.imread(str(tmp_path / "masks" / "cell_field_01.tif"))

    assert preview_count > 10
    assert _count(applied) == preview_count


def test_postprocessing_renumbers_without_merging_touching_objects():
    from spacr.object import _postprocess_masks
    mask = _touching_ringed_cells()
    out, = _postprocess_masks([mask], min_size=0)
    assert _count(out) == 2
    assert out[3, 3] != out[3, 20]
