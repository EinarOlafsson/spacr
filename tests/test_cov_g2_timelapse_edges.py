"""SAM2 tracking and event detection at their edges, without SAM2 or training."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import timelapse as tl


def _movie(frames=3):
    masks = np.zeros((frames, 12, 12), np.int32)
    masks[:, 2:5, 2:5] = 1
    masks[1:, 7:10, 7:10] = 2
    images = masks.astype(np.float32) * 100
    return masks, images


def test_sam2_refuses_bad_stacks_and_skips_empty_movies(tmp_path, capsys):
    src = str(tmp_path / "masks")
    with pytest.raises(ValueError, match="mask stack"):
        tl._sam2_track_cells(src, "b", [], "cell", np.zeros((4, 4)),
                             images=np.zeros((4, 4)))
    with pytest.raises(ValueError, match="does not match"):
        tl._sam2_track_cells(src, "b", [], "cell", np.zeros((2, 4, 4)),
                             images=np.zeros((3, 4, 4)))
    empty = np.zeros((2, 4, 4), np.int32)
    out = tl._sam2_track_cells(src, "b", [], "cell", empty,
                               images=empty.astype(float))
    assert np.asarray(out).max() == 0
    assert "nothing to follow" in capsys.readouterr().out


def test_sam2_drops_transient_tracks_and_plots_when_asked(tmp_path,
                                                          monkeypatch):
    import spacr.plot as plot

    masks, images = _movie()
    shown = []
    monkeypatch.setattr(plot, "_visualize_and_save_timelapse_stack_with_tracks",
                        lambda *a, **k: shown.append(True))

    def propagate(frames, seeds, model=None, device=None):
        return masks.copy(), {"objects": 2, "seconds": 0.1, "device": "cpu"}

    src = tmp_path / "run" / "masks"
    src.mkdir(parents=True)
    out = tl._sam2_track_cells(str(src), "b", ["a", "b", "c"], "cell", masks,
                               images=images, timelapse_remove_transient=True,
                               plot=True, propagate=propagate)
    assert set(np.unique(out)) == {0, 1}
    assert shown == [True]


def test_a_non_integer_parent_is_refused():
    tracks = pd.DataFrame({"track_id": [1, 2], "frame": [0, 1]})
    with pytest.raises(ValueError, match="non-integer parent"):
        tl._native_lineage_columns(tracks, {2: 1.5}, "sam2")


def test_event_features_of_an_empty_movie_are_empty():
    table, crops = tl._event_frame_features(np.zeros((2, 8, 8), np.int32))
    assert table.empty and crops is None


def test_event_windows_of_no_tracks_are_empty():
    table = pd.DataFrame(columns=["track_id", "frame", "area"])
    x, crops, index = tl._event_windows(table, ["area"], 3, np.zeros(1),
                                        np.ones(1))
    assert x.shape[0] == 0 and crops is None and index.empty


def test_annotations_accept_field_id_and_name_missing_columns(tmp_path):
    path = tmp_path / "ann.csv"
    pd.DataFrame({"fieldID": ["f"], "track_id": [1], "frame": [2],
                  "event": ["mitosis"]}).to_csv(path, index=False)
    assert "field" in tl._event_read_annotations(str(path)).columns
    pd.DataFrame({"field": ["f"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="lacks"):
        tl._event_read_annotations(str(path))


def test_divisions_skip_unknown_mothers_and_taken_daughters():
    tracks = pd.DataFrame({
        "track_id": [1, 1, 2, 3], "frame": [0, 1, 2, 2],
        "x": [5.0, 5.0, 6.0, 6.5], "y": [5.0, 5.0, 5.0, 5.5]})
    events = pd.DataFrame({"track_id": [9, 1, 1], "frame": [1, 1, 1],
                           "event": ["mitosis"] * 3})
    out, links = tl._event_correct_divisions(tracks, events)
    parents = out.groupby("track_id")["parent_track_id"].first()
    assert parents.loc[2] == 1 and parents.loc[3] == 1
    assert sorted(links["track_id"]) == [2, 3]


def test_event_detection_needs_tracks_and_matching_annotations(tmp_path):
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    with pytest.raises(ValueError, match="No trackpy tracks"):
        tl._event_detection(str(tracks), "cell", "trackpy")
