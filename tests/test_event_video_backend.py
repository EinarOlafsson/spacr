"""Event-video input, checkpoint, worker and trained-head boundaries."""
from types import SimpleNamespace
import hashlib

import numpy as np
import pandas as pd
import pytest

from spacr import _segmentation_backends as backends
from spacr import timelapse as tl


def _checkpoint(tmp_path, monkeypatch):
    """Small files exercise verification without substituting inference weights."""
    folder = tmp_path / "model"
    folder.mkdir()
    expected = {}
    for name in backends._EVENT_VIDEO_FILES:
        data = (name + " pinned fixture").encode()
        (folder / name).write_bytes(data)
        expected[name] = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(backends, "_EVENT_VIDEO_FILES", expected)
    return folder


@pytest.mark.parametrize("channels", [None, [], [0], [0, 0, 2], [True, 0, 0], [0.0, 0, 0]])
def test_channel_mapping_is_explicit_and_never_guessed(channels):
    with pytest.raises(ValueError, match="explicit ordered"):
        backends._event_video_inputs(np.zeros((2, 9, 1, 16, 16), np.float32), channels)


@pytest.mark.parametrize("source", [
    np.zeros((1, 9, 16, 16), np.float32),
    np.zeros((1, 0, 1, 16, 16), np.float32),
    np.zeros((1, 9, 1, 16, 16), np.uint8),
    np.full((1, 9, 1, 16, 16), np.nan, np.float32),
])
def test_invalid_crop_geometry_and_intensities_are_refused(source):
    with pytest.raises(ValueError):
        backends._event_video_inputs(source, [0, 0, 0])


@pytest.mark.parametrize("mutation", ["bytes", "missing", "symlink"])
def test_checkpoint_changes_cannot_reuse_verified_identity(tmp_path, monkeypatch, mutation):
    folder = _checkpoint(tmp_path, monkeypatch)
    assert backends._event_video_identity(folder)["revision"] == backends._EVENT_VIDEO_REVISION
    target = folder / "model.safetensors"
    original = target.read_bytes()
    target.unlink()
    if mutation == "bytes":
        target.write_bytes(original + b"changed")
    elif mutation == "symlink":
        other = tmp_path / "other.safetensors"
        other.write_bytes(original)
        target.symlink_to(other)
    with pytest.raises(ValueError):
        backends._event_video_identity(folder)


@pytest.mark.parametrize("defect", [None, "shape", "dtype", "nonfinite", "provenance", "channels"])
def test_parent_checks_worker_arrays_and_full_provenance(tmp_path, monkeypatch, defect):
    folder = _checkpoint(tmp_path, monkeypatch)
    monkeypatch.setattr(backends, "_backend_state", lambda *_a: SimpleNamespace(
        state=backends._INSTALLED, in_process=False, env="private-env"))

    class Worker:
        def request(self, op, **request):
            assert op == "event_video_features"
            assert request["device"] == "cpu"
            crops = np.load(request["input"], allow_pickle=False)
            assert crops.shape == (2, 9, 1, 16, 16)
            assert request["channel_map"] == [0, 0, 0]
            features = np.zeros((2, 768), np.float32)
            record = {**backends._event_video_identity(folder), "channel_map": [0, 0, 0]}
            if defect == "shape":
                features = features[:1]
            elif defect == "dtype":
                features = features.astype(np.float64)
            elif defect == "nonfinite":
                features[0, 0] = np.nan
            elif defect == "provenance":
                record["revision"] = "wrong"
            elif defect == "channels":
                record["channel_map"] = [1, 1, 1]
            np.save(request["output"], features, allow_pickle=False)
            return {"provenance": record}

    def factory(name, env):
        assert name == "videomae" and env == "private-env"
        return Worker()

    call = lambda: backends._event_video_features(
        np.zeros((2, 9, 1, 16, 16), np.float32), folder, [0, 0, 0],
        device="cpu", worker_for=factory)
    if defect:
        with pytest.raises(backends._BackendError):
            call()
    else:
        features, record = call()
        assert features.shape == (2, 768)
        assert record["weights_license"] == "CC-BY-NC-4.0"


def test_video_operation_cannot_run_in_a_segmentation_worker():
    response = backends._handle("cellpose", {
        "protocol": backends._PROTOCOL, "id": 7, "op": "event_video_features"}, {})
    assert response["ok"] is False
    assert response["id"] == 7
    assert "VideoMAE worker" in response["error"]["message"]


def test_optional_video_backend_is_alpha_and_not_a_segmentation_choice():
    from spacr.settings import ALPHA_FEATURES
    assert backends._SPECS["videomae"].segments is False
    assert backends._SPECS["videomae"].alpha is True
    assert "videomae" not in backends._BACKEND_NAMES
    assert "videomae_v1" in ALPHA_FEATURES[567]["models"]


def test_clip_cache_detects_changed_pixels_and_channel_mapping(tmp_path, monkeypatch):
    folder = _checkpoint(tmp_path, monkeypatch)
    calls = []

    def encode(crops, model_dir, channels, **kwargs):
        calls.append(len(crops))
        features = np.broadcast_to(crops.mean(axis=(1, 2, 3, 4))[:, None],
                                   (len(crops), 768)).copy().astype(np.float32)
        return features, {**backends._event_video_identity(model_dir), "channel_map": list(channels)}

    monkeypatch.setattr(backends, "_event_video_features", encode)
    config = {"model_dir": str(folder), "channel_map": [0, 0, 0], "device": "cpu"}
    windows = np.zeros((2, 9, 2, 16, 16), np.float32)
    index = pd.DataFrame({"track_id": [1, 2], "frame": [3, 4]})
    original = {(3, 1): windows[0, 0], (4, 2): windows[1, 0]}
    first, _ = tl._event_clip_features(config, windows, index, original, 9)
    again, _ = tl._event_clip_features(config, windows, index, original, 9)
    np.testing.assert_array_equal(first, again)
    assert calls == [2]
    windows[0] = 1
    changed, _ = tl._event_clip_features(config, windows, index, original, 9)
    assert calls == [2, 1]
    assert changed[0, 0] == 1 and changed[1, 0] == 0
    config["channel_map"] = [1, 1, 1]
    tl._event_clip_features(config, windows, index, original, 9)
    assert calls == [2, 1, 2]
    del original[(3, 1)]
    with pytest.raises(ValueError, match="every observed track frame"):
        tl._event_clip_features(config, windows, index, original, 9)


def test_fusion_head_trains_saves_and_refuses_missing_video_features(tmp_path):
    import torch
    rng = np.random.default_rng(3)
    inputs = rng.normal(size=(6, 3, 9)).astype(np.float32)
    videos = rng.normal(size=(6, 768)).astype(np.float32)
    labels = np.array([0, 1, 0, 1, 0, 1], np.int64)
    model = tl._event_train([(inputs, None, labels, np.ones(6, np.float32), videos)],
                            ["none", "mitosis"], ["area"], window=9,
                            epochs=1, video_width=768)
    path = tmp_path / "event_model.pt"
    torch.save(model, path)
    restored = torch.load(path, weights_only=True, map_location="cpu")
    probabilities = tl._event_probabilities(restored, inputs, video=videos)
    assert probabilities.shape == (6, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(probabilities,
                                  tl._event_probabilities(model, inputs, video=videos))
    with pytest.raises(ValueError, match="pretrained clip features"):
        tl._event_probabilities(restored, inputs)
