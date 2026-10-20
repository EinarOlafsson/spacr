"""Item 706: 3D detection refuses overlapping runs and stale, empty or unusable results.

A second run while one is out is refused; a 2-D-only method falls back to
Otsu and a named threshold runs as a two-class split; a result for a changed
volume or mask is discarded, an empty result leaves the mask alone, and a
result the mask cannot hold is reported. The volume Cellpose call adapts to
models whose ``eval`` cannot be inspected or returns a list, and refuses
labels of the wrong shape; the captured request inverts and normalizes the
volume before detecting.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.qt import cpu_modes
from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks

from .test_706_object_detection_3d import screen

__all__ = ["screen"]


class _Uninspectable:
    """A model ``eval`` whose signature cannot be read."""

    __signature__ = "not a signature"

    def __init__(self, labels):
        self.labels, self.kwargs = labels, None

    def __call__(self, field, **kwargs):
        self.kwargs = kwargs
        return [self.labels], None, None


def test_an_uninspectable_model_gets_every_setting_and_a_listed_result_is_unwrapped():
    volume = np.zeros((3, 4, 5), np.float32)
    labels = np.zeros((3, 4, 5), np.int64)
    labels[1, 1:3, 1:3] = 2
    model = type("Model", (), {})()
    model.eval = _Uninspectable(labels)
    out = make_masks.cellpose_detect_volume(volume, model, stitch_threshold=0.25)
    assert out.dtype == np.int32 and np.array_equal(out, labels)
    assert model.eval.kwargs["stitch_threshold"] == 0.25
    assert model.eval.kwargs["do_3D"] is False and model.eval.kwargs["z_axis"] == 0


def test_labels_of_another_shape_are_refused():
    model = type("Model", (), {})()
    model.eval = _Uninspectable(np.zeros((3, 4), np.int32))
    with pytest.raises(ValueError, match=r"labels shaped \(3, 4\) for a volume shaped \(3, 4, 5\)"):
        make_masks.cellpose_detect_volume(np.zeros((3, 4, 5)), model)


def test_a_captured_request_inverts_normalizes_and_loads_its_model_once(monkeypatch):
    image = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    seen = []

    def detect(volume, model, **parameters):
        seen.append((volume.copy(), model, parameters))
        return (volume > volume.mean()).astype(np.int32)

    loaded = []
    monkeypatch.setattr(make_masks, "cellpose_detect_volume", detect)
    monkeypatch.setattr(make_masks, "load_cellpose_model",
                        lambda name: loaded.append(name) or "loaded-model")
    request = {"axes": "ZYX", "image": image, "invert": True, "percentiles": (1.0, 99.0),
               "kind": "cellpose", "model": "cpsam", "parameters": {"diameter": 3}}
    models = {}
    labels = make_masks._detect_volume_snapshot(request, models)
    make_masks._detect_volume_snapshot(request, models)
    assert loaded == ["cpsam"] and models == {"cpsam": "loaded-model"}
    volume, model, parameters = seen[0]
    expected = engine.normalize_for_detection(engine.invert_normalized(image), 1.0, 99.0)
    np.testing.assert_allclose(volume, expected)
    assert model == "loaded-model" and parameters == {"diameter": 3}
    assert labels.shape == image.shape


def test_a_second_run_while_one_is_out_is_refused(screen):
    screen._detection_request = {"busy": True}
    assert screen._start_volume_detection("threshold") is False
    assert "Wait for the current 3D operation" in screen._status_label.text()
    screen._detection_request = None


@pytest.mark.parametrize("mode, classes", [
    ("cellpose", None), (cpu_modes.MULTIOTSU, 3), ("li", 2)])
def test_the_threshold_run_follows_the_method_it_can_run_in_3d(screen, monkeypatch, mode,
                                                                classes):
    submitted = []
    screen._magnifier.mode = mode
    monkeypatch.setattr(screen._otsu_classes, "value", lambda: 2)

    class Worker:
        def submit(self, request):
            submitted.append(request)

        def close(self, **_options):
            return True

    monkeypatch.setattr(screen, "_volume_detection_worker", Worker())
    assert screen._start_volume_detection("threshold")
    request = submitted[0]
    if mode == "cellpose":
        assert request["method"] == "otsu" and request["parameters"]["algorithm"] == "otsu"
    else:
        assert request["parameters"]["classes"] == classes
    if mode == "li":
        assert request["parameters"]["foreground_class"] == 1
    screen._detection_request = None


def _request(screen, **changes):
    session = screen._mask_stage.session
    request = {"session": session, "mask": session.mask.copy(), "label": "Otsu",
               "mode": "replace"}
    request.update(changes)
    return request


def test_a_result_for_a_changed_mask_is_discarded(screen):
    session = screen._mask_stage.session
    labels = (np.asarray(session.image) > 0).astype(np.int32)
    request = _request(screen, mask=np.ones_like(session.mask))
    screen._take_volume_detection(request, labels, None)
    assert "discarded because the volume or its mask changed" in screen._status_label.text()
    assert not session.mask.any()


def test_an_empty_result_leaves_the_mask_unchanged(screen):
    session = screen._mask_stage.session
    screen._take_volume_detection(_request(screen), np.zeros(session.mask.shape, np.int32), None)
    assert "Otsu found no objects in the volume" in screen._status_label.text()
    assert not session.mask.any()


def test_a_result_the_mask_cannot_hold_is_reported(screen, monkeypatch):
    session = screen._mask_stage.session
    labels = np.zeros(session.mask.shape, np.int32)
    labels[0, 0, 0] = 70000
    warned = []
    monkeypatch.setattr(screen, "_warn", lambda title, text: warned.append((title, text)))
    screen._take_volume_detection(_request(screen), labels, None)
    assert warned and warned[0][0] == "Object detection failed"
    assert "No unused object identifier fits" in warned[0][1]
    assert not session.mask.any()


def test_without_its_controls_the_3d_switch_is_left_alone(screen, monkeypatch):
    toggle = screen._detect_3d
    monkeypatch.setattr(screen, "_cp_model", None)
    enabled = toggle.isEnabled()
    screen._sync_detect_3d()
    assert toggle.isEnabled() == enabled

