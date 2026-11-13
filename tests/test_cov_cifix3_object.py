"""The one-label-image helper, shared-GPU workers and division prediction in Make Masks.

A worker may share its GPU, and a time-lapse run can predict divisions once
after its division analysis.  These tests hold each of those routes to what
the run passes on.
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
    fake_timelapse as sam_timelapse,
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


@pytest.mark.parametrize("predict", [True, False])
def test_divisions_are_predicted_once_after_a_cellpose_run_only_when_asked(
        tmp_path, fake_cellpose, fake_timelapse, monkeypatch, predict):
    import spacr.division_analysis as DA
    import spacr.division_prediction as DP

    monkeypatch.setattr(DA, "_run_division_step", lambda *a, **k: None)
    predicted = []
    monkeypatch.setattr(DP, "_predict_after_run",
                        lambda src, obj, mode, settings: predicted.append((src, obj, mode)))
    src = tmp_path / "stack"
    _write_npz(src, n=3)
    O.generate_cellpose_masks(str(src), _tl(
        src, timelapse_division_analysis=True,
        timelapse_division_prediction=predict), "nucleus")
    assert predicted == ([(str(src), "nucleus", "trackpy")] if predict else [])


def test_divisions_are_predicted_once_after_a_sam_run(
        tmp_path, fake_model, sam_timelapse, monkeypatch):
    import spacr.division_analysis as DA
    import spacr.division_prediction as DP

    monkeypatch.setattr(DA, "_run_division_step", lambda *a, **k: None)
    predicted = []
    monkeypatch.setattr(DP, "_predict_after_run",
                        lambda src, obj, mode, settings: predicted.append((src, obj, mode)))
    src = tmp_path / "stack"
    _write_sam_npz(src, n=2)
    O.generate_cellpose_masks_sam(str(src), _tl_settings(
        src, batch_size=2, timelapse_mode="trackpy",
        timelapse_division_analysis=True, timelapse_division_prediction=True), "cell")
    assert predicted == [(str(src), "cell", "trackpy")]
