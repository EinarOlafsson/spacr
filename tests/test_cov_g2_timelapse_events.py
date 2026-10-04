"""Event detection on tracks: empty fields, one-field validation, odd names."""
from __future__ import annotations

import numpy as np
import pandas as pd

from spacr import timelapse as tl


def _tracks(n_tracks=6, n_frames=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for track in range(1, n_tracks + 1):
        x, y = rng.uniform(0, 50, 2)
        for frame in range(n_frames):
            x, y = x + rng.normal(0, 1), y + rng.normal(0, 1)
            rows.append({"frame": frame, "track_id": track, "x": x, "y": y})
    return pd.DataFrame(rows)


def test_a_field_without_rows_has_no_events():
    table = tl._event_track_table(_tracks()).iloc[0:0]
    model = {"columns": ["speed"], "window": 5, "mean": [0.0], "std": [1.0],
             "channels": 0}
    found = tl._event_detect(model, table)
    assert found.empty and list(found.columns) == ["track_id", "frame", "event",
                                                   "probability"]


def test_one_annotated_field_is_validated_by_holding_out_tracks():
    table = tl._event_track_table(_tracks())
    annotations = pd.DataFrame({"field": ["f"] * 3, "track_id": [1, 2, 3],
                                "frame": [5, 6, 7], "event": ["mitosis"] * 3})
    scores, detected = tl._event_cross_validate({"f": (table, {})}, annotations,
                                                 window=3, epochs=1)
    assert "mitosis" in list(scores["event"])
    assert set(detected["field"]) <= {"f"}


def test_existing_parent_links_are_kept_when_correcting_divisions():
    tracks = pd.DataFrame({"frame": [0, 1, 2, 3], "track_id": [1, 1, 1, 2],
                           "x": [10.0, 10, 10, 12], "y": [10.0] * 4,
                           "parent_track_id": [0, 0, 0, "7"]})
    events = pd.DataFrame({"track_id": [1], "frame": [2], "event": ["mitosis"]})
    fixed, links = tl._event_correct_divisions(tracks, events, max_distance=20)
    assert fixed.groupby("track_id")["parent_track_id"].first().to_dict()[2] == 7
    assert links.empty


def test_marks_outside_a_track_are_ignored_when_splitting_cycles():
    spans = pd.DataFrame({"field": ["f"], "track_id": [1], "start": [5],
                          "end": [9]})
    cycles = tl._event_cycles(spans, {("f", 1): [2, 7, 30]})
    assert cycles["event"].tolist() == [1, 0]


def test_timing_accepts_fields_named_outside_the_plate_scheme():
    table = pd.DataFrame({"frame": list(range(10)), "track_id": 1,
                          "x": 0.0, "y": 0.0})
    events = pd.DataFrame({"field": ["movie"], "track_id": [1], "frame": [4],
                           "event": ["death"]})
    timing = tl._event_timing({"movie": table}, events)
    assert set(timing) <= {"death"}


def test_conditions_naming_no_imaged_well_time_nothing():
    table = pd.DataFrame({"frame": list(range(10)), "track_id": 1,
                          "x": 0.0, "y": 0.0})
    events = pd.DataFrame({"field": ["plate1_r1_c1_f1"], "track_id": [1],
                           "frame": [4], "event": ["death"]})
    assert tl._event_timing({"plate1_r1_c1_f1": table}, events,
                            conditions=["control=c9"]) == {}


def test_a_new_object_found_on_two_frames_is_seeded_once():
    masks = np.zeros((3, 8, 8), np.int32)
    masks[1, 2:5, 2:5] = 4
    masks[2, 2:6, 2:6] = 4
    seeds = tl._sam2_new_seeds(masks, np.zeros_like(masks), 10)
    assert list(seeds) == [1] and seeds[1].max() == 10


def test_a_track_barely_touching_another_has_no_parent():
    stack = np.zeros((2, 8, 8), np.int32)
    stack[0, 0:2, 0:2] = 1
    stack[1, 1:7, 1:7] = 2
    assert tl._sam2_parent_links(stack) == {1: 0, 2: 0}


def test_frame_features_without_images_have_no_crops():
    masks = np.zeros((2, 16, 16), np.int32)
    masks[1, 4:9, 4:9] = 3
    features, crops = tl._event_frame_features(masks)
    assert features["frame"].tolist() == [1]
    assert "intensity_mean_c0" not in features.columns


def test_windows_leave_a_missing_column_at_its_mean():
    tracks = pd.DataFrame({"frame": [0, 1, 2], "track_id": [1, 1, 1],
                           "x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0]})
    table = tl._event_track_table(tracks)
    x, _crops, _index = tl._event_windows(table, ["speed", "absent"], 3,
                                          [0.0, 0.0], [1.0, 1.0])
    assert x.shape[1] == 3


def test_a_detection_too_far_from_its_annotation_is_not_a_hit():
    ann = pd.DataFrame({"field": ["a"], "track_id": [1], "frame": [5],
                        "event": ["mitosis"]})
    det = pd.DataFrame({"field": ["a"], "track_id": [1], "frame": [15],
                        "event": ["mitosis"], "probability": [0.9]})
    row = tl._event_scores(det, ann, ["mitosis"], tolerance=2).iloc[0]
    assert row["true_positives"] == 0


def test_a_new_object_that_barely_overlaps_the_last_one_is_seeded_again():
    masks = np.zeros((3, 8, 8), np.int32)
    masks[1, 0:2, 0:2] = 4
    masks[2, 1:7, 1:7] = 4
    seeds = tl._sam2_new_seeds(masks, np.zeros_like(masks), 10)
    assert sorted(seeds) == [1, 2] and seeds[2].max() == 11


def test_annotations_for_another_object_name_no_event(tmp_path):
    (tmp_path / "tracks").mkdir()
    _tracks(n_tracks=2, n_frames=4).to_csv(
        tmp_path / "tracks" / "trackpy_tracks_cell_p_r1_c1_f1.csv", index=False)
    ann = pd.DataFrame({"field": ["p_r1_c1_f1"], "track_id": [1], "frame": [2],
                        "event": ["mitosis"], "object": ["nucleus"]})
    import pytest

    with pytest.raises(ValueError, match="names no event on the cell tracks"):
        tl._event_detection(str(tmp_path / "tracks"), "cell", "trackpy",
                            annotations=ann)


def test_event_features_without_images_and_a_failure_are_reported(tmp_path,
                                                                   capsys):
    src = tmp_path / "plate" / "merged"
    masks = np.zeros((2, 16, 16), np.int32)
    masks[:, 4:9, 4:9] = 1
    tl._run_event_features_step(str(src), "f1", "cell", masks, None, "iou", {})
    out = tmp_path / "plate" / "tracks" / "events"
    assert (out / "trackpy_tracks_cell_f1_features.csv").exists()
    assert not (out / "trackpy_tracks_cell_f1_crops.npz").exists()
    tl._run_event_features_step(str(src), "f2", "cell", None, None, "iou", {})
    assert "could not be stored for f2" in capsys.readouterr().out


def test_the_detection_step_reports_failures_and_held_out_scores(
        tmp_path, monkeypatch, capsys):
    scores = pd.DataFrame({"precision": [0.5], "recall": [0.25],
                           "mean_abs_timing_error": [1.0]})

    def detect(tracks_dir, object_type, prefix, **kwargs):
        if object_type == "nucleus":
            raise ValueError("no tracks")
        return {"events": pd.DataFrame({"event": ["mitosis"]}),
                "scores": scores}

    monkeypatch.setattr(tl, "_event_detection", detect)
    results = tl._run_event_detection_step(
        str(tmp_path), {"timelapse_objects": ["cell", "nucleus"]})
    printed = capsys.readouterr().out
    assert list(results) == ["cell"]
    assert "Event detection (nucleus) failed: no tracks" in printed
    assert "Held-out precision 0.50, recall 0.25" in printed
