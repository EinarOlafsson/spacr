"""Per-image diameters, shared-GPU workers and division analysis in Make Masks.

Cellpose-SAM can estimate the object diameter in each image, a mask worker
may share its GPU, and a timelapse can hand its tracked fields to the
division analysis. These tests hold each of those routes to what the run
passes on.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacr.object as O
from tests.test_cov_object_cellpose_masks import (  # noqa: F401
    _tl, _write_npz, fake_cellpose, fake_timelapse,
)
from tests.test_cov_object_masks_sam import (  # noqa: F401
    _tl_settings, _write_npz as _write_sam_npz, fake_model,
    fake_timelapse as sam_timelapse, _settings as _sam_settings,
)


def test_a_two_dimensional_answer_is_the_one_label_image():
    plane = np.arange(6, dtype=np.uint16).reshape(2, 3)
    assert O._first_mask(plane) is plane
    stack = np.zeros((1, 2, 3), np.uint16)
    assert O._first_mask([stack[0]]).shape == (2, 3)


def test_a_worker_count_that_is_not_a_number_does_not_share_a_gpu():
    assert O._shares_a_gpu({"mask_workers_per_gpu": 2}) is True
    assert O._shares_a_gpu({"mask_workers_per_gpu": 1}) is False
    assert O._shares_a_gpu({"mask_workers_per_gpu": "many"}) is False
    assert O._shares_a_gpu({"mask_workers_per_gpu": [3]}) is False


def test_each_image_is_segmented_at_its_own_diameter(
        tmp_path, fake_model, monkeypatch):
    seen = {}

    def per_image(model, images, eval_kwargs, fallback):
        seen["images"] = len(images)
        seen["fallback"] = fallback
        masks = [np.asarray(m, np.uint16) for m in model.eval(
            images, channel_axis=-1)[0]]
        flows = [np.zeros(m.shape, np.float32) for m in masks]
        return masks, flows, [12.0, 0, 16.0]

    noted = []
    monkeypatch.setattr(O, "_per_image_diameter_masks", per_image)
    monkeypatch.setattr(O, "_note_the_diameter",
                        lambda settings, obj, estimate=None:
                        noted.append((settings["cell_diameter"], obj, estimate)))
    src = tmp_path / "stack"
    _write_sam_npz(src, n=3)
    settings = _sam_settings(src, estimate_diameters_per_image=True,
                             cell_diameter="auto")
    O.generate_cellpose_masks_sam(str(src), settings, "cell")
    assert seen["images"] == 3
    assert [call for call in noted if call[2] is not None] == [
        ("auto", "cell", 14.0)]


def test_each_image_without_an_estimate_notes_no_diameter(
        tmp_path, fake_model, monkeypatch):
    def per_image(model, images, eval_kwargs, fallback):
        masks = [np.zeros(image.shape[:2], np.uint16) for image in images]
        return masks, [np.zeros(m.shape, np.float32) for m in masks], [0, None]

    noted = []
    monkeypatch.setattr(O, "_per_image_diameter_masks", per_image)
    monkeypatch.setattr(O, "_note_the_diameter",
                        lambda *args, **kwargs: noted.append(
                            args + (kwargs.get("estimate"),)))
    src = tmp_path / "stack"
    _write_sam_npz(src, n=2)
    O.generate_cellpose_masks_sam(str(src), _sam_settings(
        src, estimate_diameters_per_image=True, cell_diameter="auto"), "cell")
    assert [call for call in noted
            if len(call) > 2 and call[2] is not None] == []


def test_cellpose_timelapse_fields_reach_the_division_analysis(
        tmp_path, fake_cellpose, fake_timelapse, monkeypatch):
    import spacr.division_analysis as DA

    calls = []
    monkeypatch.setattr(
        DA, "_run_division_step",
        lambda src, name, obj, mode, settings, *, label_stack, images=None:
        calls.append((name, obj, mode, len(label_stack), len(images))))
    src = tmp_path / "stack"
    _write_npz(src, n=3)
    O.generate_cellpose_masks(
        str(src), _tl(src, timelapse_division_analysis=True), "nucleus")
    assert calls == [("batch1", "nucleus", "trackpy", 3, 3)]


def test_sam_timelapse_fields_reach_the_division_analysis(
        tmp_path, fake_model, sam_timelapse, monkeypatch):
    import spacr.division_analysis as DA

    calls = []
    monkeypatch.setattr(
        DA, "_run_division_step",
        lambda src, name, obj, mode, settings, *, label_stack, images=None:
        calls.append((obj, mode, len(label_stack), len(images))))
    src = tmp_path / "stack"
    _write_sam_npz(src, n=2)
    O.generate_cellpose_masks_sam(str(src), _tl_settings(
        src, batch_size=2, timelapse_mode="trackpy",
        timelapse_division_analysis=True), "cell")
    assert calls == [("cell", "trackpy", 2, 2)]
