"""CPU event-head routing keeps pretrained vectors and provenance explicit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import _segmentation_backends as backends
from spacr import timelapse
from spacr.tabular import write_table


def _video(monkeypatch, tmp_path):
    identity = {"files_sha256": {"model.safetensors": "verified"},
                "revision": "pinned", "frames": 16, "intensity": "field",
                "frame_sampling": "uniform", "pooling": "mean"}
    calls = []

    def encode(windows, model_dir, channels, *, device):
        assert device == "cpu" and channels == [0, 0, 0]
        calls.append(windows.copy())
        values = windows.mean(axis=(1, 2, 3, 4))
        vectors = np.repeat(values[:, None], 768, axis=1).astype(np.float32)
        return vectors, dict(identity, channel_map=list(channels))

    monkeypatch.setattr(backends, "_event_video_identity", lambda _path: identity)
    monkeypatch.setattr(backends, "_event_video_features", encode)
    config = {"model_dir": str(tmp_path / "declared-checkpoint"),
              "channel_map": [0, 0, 0], "device": "cpu"}
    return config, identity, calls


def _field():
    tracks = pd.DataFrame({"track_id": [7, 7, 8, 8], "frame": [0, 1, 0, 1],
                           "x": [0., 2., 3., 6.], "y": [1., 1., 4., 4.]})
    table = timelapse._event_track_table(tracks)
    crops = {(int(row.frame), int(row.track_id)):
             np.full((1, 4, 4), (i + 1) / 10, np.float32)
             for i, row in enumerate(tracks.itertuples())}
    return table, crops


@pytest.mark.parametrize("defect", ["missing", "wrong_width", "wrong_batch"])
def test_cpu_fusion_head_refuses_missing_or_misaligned_video_vectors(defect):
    import torch

    network = timelapse._event_network(2, 2, video_width=768).eval()
    inputs = torch.zeros((2, 2, 3))
    video = {"missing": None, "wrong_width": torch.zeros((2, 767)),
             "wrong_batch": torch.zeros((1, 768))}[defect]
    with torch.no_grad():
        with pytest.raises(ValueError, match="matching pretrained clip features"):
            network(inputs, video=video)
        scores = network(inputs, video=torch.ones((2, 768)))
    assert scores.device.type == "cpu" and scores.shape == (2, 2)
    assert torch.isfinite(scores).all()


def test_clip_cache_requires_images_and_revalidates_cached_checkpoint(monkeypatch, tmp_path):
    config, _identity, calls = _video(monkeypatch, tmp_path)
    table, collection = _field()
    windows = np.ones((2, 3, 1, 4, 4), np.float32)
    index = table[["track_id", "frame"]].iloc[:2]
    with pytest.raises(ValueError, match="requires saved image crops"):
        timelapse._event_clip_features(config, None, index, collection, 3)
    assert calls == []
    first, provenance = timelapse._event_clip_features(config, windows, index, collection, 3)
    assert first.shape == (2, 768) and provenance["channel_map"] == [0, 0, 0]
    config["_provenance"]["revision"] = "different checkpoint"
    with pytest.raises(ValueError, match="cache does not match"):
        timelapse._event_clip_features(config, windows, index, collection, 3)
    assert len(calls) == 1


def test_empty_clip_request_keeps_verified_width_without_encoding(monkeypatch, tmp_path):
    config, _identity, calls = _video(monkeypatch, tmp_path)
    table, collection = _field()
    windows = np.ones((1, 3, 1, 4, 4), np.float32)
    index = table[["track_id", "frame"]].iloc[:1]
    timelapse._event_clip_features(config, windows, index, collection, 3)
    features, provenance = timelapse._event_clip_features(
        config, windows[:0], index.iloc[:0], collection, 3)
    assert features.shape == (0, 768) and features.dtype == np.float32
    assert provenance["channel_map"] == [0, 0, 0] and len(calls) == 1


def test_video_fit_refuses_a_field_without_crops_before_training(monkeypatch, tmp_path):
    config, _identity, calls = _video(monkeypatch, tmp_path)
    table, _crops = _field()
    annotations = pd.DataFrame({"field": ["field"], "track_id": [7],
                                "frame": [1], "event": ["death"]})

    def forbidden(*_args, **_kwargs):
        raise AssertionError("incomplete image collection reached training")

    monkeypatch.setattr(timelapse, "_event_train", forbidden)
    with pytest.raises(ValueError, match="image crops in every annotated field"):
        timelapse._event_fit({"field": (table, {})}, annotations, video=config)
    assert calls == []


def test_video_fit_routes_owned_vectors_and_persists_explicit_configuration(monkeypatch, tmp_path):
    config, identity, calls = _video(monkeypatch, tmp_path)
    table, crops = _field()
    annotations = pd.DataFrame({"field": ["field"], "track_id": [7],
                                "frame": [1], "event": ["death"]})
    trained = []

    def train(samples, classes, columns, **options):
        trained.append((samples, classes, columns, options))
        return {"classes": classes, "columns": columns, **options}

    monkeypatch.setattr(timelapse, "_event_train", train)
    model = timelapse._event_fit({"field": (table, crops)}, annotations,
                               window=3, epochs=1, video=config)
    samples, classes, columns, options = trained[0]
    inputs, image_windows, labels, weights, vectors = samples[0]
    assert classes == [timelapse._EVENT_BACKGROUND, "death"]
    assert len(inputs) == len(labels) == len(weights) == len(vectors) == len(table)
    assert vectors.shape == (len(table), 768) and options["video_width"] == 768
    np.testing.assert_array_equal(vectors[:, 0], calls[0].mean(axis=(1, 2, 3, 4)))
    assert image_windows.shape == (len(table), 3, 1, 4, 4)
    assert model["video"] == {"model_dir": str(tmp_path / "declared-checkpoint"),
                              "channel_map": [0, 0, 0], "device": "cpu"}
    assert model["video_provenance"] == dict(identity, channel_map=[0, 0, 0])
    assert len(model["mean"]) == len(model["std"]) == len(columns)
    assert np.isfinite(model["mean"]).all() and np.array(model["std"]).min() > 0


@pytest.mark.parametrize("changed", [None, "files_sha256", "revision", "channel_map",
                                    "frames", "intensity", "frame_sampling", "pooling"])
def test_saved_video_detector_checks_every_provenance_component_before_scores(
        monkeypatch, tmp_path, changed):
    config, identity, _calls = _video(monkeypatch, tmp_path)
    table, crops = _field()
    columns = timelapse._event_columns(table)
    expected = dict(identity, channel_map=[0, 0, 0])
    if changed:
        expected[changed] = "different"
    model = {"columns": columns, "window": 3, "mean": [0.] * len(columns),
             "std": [1.] * len(columns), "channels": 1, "video_width": 768,
             "video": dict(config), "video_provenance": expected,
             "classes": [timelapse._EVENT_BACKGROUND, "death"]}
    scoring = []

    def probabilities(_model, inputs, image_windows, features):
        scoring.append(features.copy())
        assert features.shape == (len(inputs), 768)
        return np.tile([.1, .9], (len(inputs), 1))

    monkeypatch.setattr(timelapse, "_event_probabilities", probabilities)
    if changed:
        with pytest.raises(ValueError, match="encoder provenance differ"):
            timelapse._event_detect(model, table, crops)
        assert scoring == []
    else:
        found = timelapse._event_detect(model, table, crops)
        assert len(scoring) == 1 and not found.empty
        assert set(found["event"]) == {"death"}
        assert set(found["track_id"]) == {7, 8}


@pytest.mark.parametrize("checkpoint,channels", [("", [0, 0, 0]), ("declared", None)])
def test_video_selection_requires_both_checkpoint_and_channel_map(tmp_path, checkpoint, channels):
    with pytest.raises(ValueError, match="explicit video_channels"):
        timelapse._event_detection(str(tmp_path), "cell", "trackpy",
                                   encoder="videomae", annotations=pd.DataFrame(),
                                   video_checkpoint=checkpoint, video_channels=channels)


def test_unknown_event_encoder_is_refused_before_reading_tracks(tmp_path):
    with pytest.raises(ValueError, match="small or videomae"):
        timelapse._event_detection(str(tmp_path), "cell", "trackpy", encoder="guessed")


@pytest.mark.parametrize("video_width,explicit_override", [(0, False), (768, False),
                                                         (768, True)])
def test_saved_model_loading_refuses_wrong_head_and_reuses_declared_cpu_encoder(
        tmp_path, monkeypatch, video_width, explicit_override):
    import torch

    tracks = tmp_path / "tracks"
    tracks.mkdir()
    table, crops = _field()
    write_table(table[["track_id", "frame", "x", "y"]],
                tracks / "trackpy_tracks_cell_field.csv", canonicalise=False)
    monkeypatch.setattr(timelapse, "_event_field_inputs", lambda *_args: (table, crops))
    model = {"video_width": video_width, "video": {"model_dir": "saved-checkpoint",
             "channel_map": [0, 0, 0], "device": "auto"},
             "classes": [timelapse._EVENT_BACKGROUND, "death"]}
    loads, routed = [], []

    def load(path, **options):
        loads.append((path, options))
        return model

    def detect(_model, _table, _crops, **options):
        routed.append(options["video"])
        return pd.DataFrame({"track_id": [7], "frame": [1],
                             "event": ["death"], "probability": [.9]})

    monkeypatch.setattr(torch, "load", load)
    monkeypatch.setattr(timelapse, "_event_detect", detect)
    monkeypatch.setattr(timelapse, "_event_timing", lambda *_args: {})
    arguments = dict(model_path="saved-event-model.pt", encoder="videomae",
                     video_device="cpu", plot=False)
    if explicit_override:
        arguments.update(video_checkpoint="current-checkpoint", video_channels=[0, 0, 0])
    if not video_width:
        with pytest.raises(ValueError, match="not trained with VideoMAE"):
            timelapse._event_detection(str(tracks), "cell", "trackpy", **arguments)
        assert routed == []
    else:
        result = timelapse._event_detection(str(tracks), "cell", "trackpy", **arguments)
        assert routed == [{"model_dir": "current-checkpoint" if explicit_override
                           else "saved-checkpoint", "channel_map": [0, 0, 0],
                           "device": "cpu"}]
        assert model["video"]["device"] == "auto"
        assert result["events"]["field"].tolist() == ["field"]
    assert loads == [("saved-event-model.pt", {"map_location": "cpu", "weights_only": True})]
