"""SAM2 tracking and event detection at their edges, without SAM2 or training."""
from __future__ import annotations

import inspect
import re

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


def _scn_sam2_refuses_bad_stacks_and_skips_empty_movies(tmp_path, capsys):
    _obs = []
    src = str(tmp_path / "masks")
    with pytest.raises(ValueError, match="mask stack") as _raised:
        tl._sam2_track_cells(src, "b", [], "cell", np.zeros((4, 4)),
                             images=np.zeros((4, 4)))
    _obs.append(_seen(type(_raised.value).__name__ + ": " + str(_raised.value), True))
    with pytest.raises(ValueError, match="does not match") as _raised:
        tl._sam2_track_cells(src, "b", [], "cell", np.zeros((2, 4, 4)),
                             images=np.zeros((3, 4, 4)))
    _obs.append(_seen(type(_raised.value).__name__ + ": " + str(_raised.value), True))
    empty = np.zeros((2, 4, 4), np.int32)
    out = tl._sam2_track_cells(src, "b", [], "cell", empty,
                               images=empty.astype(float))
    assert np.asarray(out).max() == 0
    _obs.append(_seen((np.asarray(out).max()), (np.asarray(out).max() == 0)))
    assert "nothing to follow" in capsys.readouterr().out
    _obs.append(_seen(("nothing to follow" in capsys.readouterr().out), ("nothing to follow" in capsys.readouterr().out)))
    return _obs


def _scn_sam2_drops_transient_tracks_and_plots_when_asked(tmp_path,
                                                          monkeypatch):
    _obs = []
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
    _obs.append(_seen((set(np.unique(out))), (set(np.unique(out)) == {0, 1})))
    assert shown == [True]
    _obs.append(_seen((shown), (shown == [True])))
    return _obs


def _scn_a_non_integer_parent_is_refused():
    _obs = []
    tracks = pd.DataFrame({"track_id": [1, 2], "frame": [0, 1]})
    with pytest.raises(ValueError, match="non-integer parent") as _raised:
        tl._native_lineage_columns(tracks, {2: 1.5}, "sam2")
    _obs.append(_seen(type(_raised.value).__name__ + ": " + str(_raised.value), True))
    return _obs


def _scn_event_features_of_an_empty_movie_are_empty():
    _obs = []
    table, crops = tl._event_frame_features(np.zeros((2, 8, 8), np.int32))
    assert table.empty and crops is None
    _obs.append(_seen((table.empty and crops is None), (table.empty and crops is None)))
    return _obs


def _scn_event_windows_of_no_tracks_are_empty():
    _obs = []
    table = pd.DataFrame(columns=["track_id", "frame", "area"])
    x, crops, index = tl._event_windows(table, ["area"], 3, np.zeros(1),
                                        np.ones(1))
    assert x.shape[0] == 0 and crops is None and index.empty
    _obs.append(_seen((x.shape[0] == 0 and crops is None and index.empty), (x.shape[0] == 0 and crops is None and index.empty)))
    return _obs


def _scn_annotations_accept_field_id_and_name_missing_columns(tmp_path):
    _obs = []
    path = tmp_path / "ann.csv"
    pd.DataFrame({"fieldID": ["f"], "track_id": [1], "frame": [2],
                  "event": ["mitosis"]}).to_csv(path, index=False)
    assert "field" in tl._event_read_annotations(str(path)).columns
    _obs.append(_seen(("field" in tl._event_read_annotations(str(path)).columns), ("field" in tl._event_read_annotations(str(path)).columns)))
    pd.DataFrame({"field": ["f"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="lacks") as _raised:
        tl._event_read_annotations(str(path))
    _obs.append(_seen(type(_raised.value).__name__ + ": " + str(_raised.value), True))
    return _obs


def _scn_divisions_skip_unknown_mothers_and_taken_daughters():
    _obs = []
    tracks = pd.DataFrame({
        "track_id": [1, 1, 2, 3], "frame": [0, 1, 2, 2],
        "x": [5.0, 5.0, 6.0, 6.5], "y": [5.0, 5.0, 5.0, 5.5]})
    events = pd.DataFrame({"track_id": [9, 1, 1], "frame": [1, 1, 1],
                           "event": ["mitosis"] * 3})
    out, links = tl._event_correct_divisions(tracks, events)
    parents = out.groupby("track_id")["parent_track_id"].first()
    assert parents.loc[2] == 1 and parents.loc[3] == 1
    _obs.append(_seen((parents.loc[2] == 1 and parents.loc[3] == 1), (parents.loc[2] == 1 and parents.loc[3] == 1)))
    assert sorted(links["track_id"]) == [2, 3]
    _obs.append(_seen((sorted(links["track_id"])), (sorted(links["track_id"]) == [2, 3])))
    return _obs


def _scn_event_detection_needs_tracks_and_matching_annotations(tmp_path):
    _obs = []
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    with pytest.raises(ValueError, match="No trackpy tracks") as _raised:
        tl._event_detection(str(tracks), "cell", "trackpy")
    _obs.append(_seen(type(_raised.value).__name__ + ": " + str(_raised.value), True))
    return _obs


def _plain(value):
    """Return a plain comparable copy of a value, or raise ValueError."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return re.sub(r"/tmp/pytest-of-[^/]+/pytest-\d+/[^/\s]+", "<tmp>", value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return None if value != value else round(value, 6)
    if isinstance(value, (list, tuple)) and len(value) <= 60:
        return [_plain(v) for v in value]
    if isinstance(value, (set, frozenset)) and len(value) <= 60:
        return sorted((_plain(v) for v in value), key=repr)
    if isinstance(value, dict) and len(value) <= 60:
        return {str(k): _plain(v) for k, v in value.items()}
    raise ValueError(type(value))


def _seen(value, held):
    """Observed value of one check: the value itself when plain, else whether the check held."""
    try:
        return _plain(value)
    except ValueError:
        return bool(held)


def _run_scenario(request, scenario, kwargs):
    """Run one scenario function, resolving its fixtures by name."""
    wanted = [n for n in inspect.signature(scenario).parameters if n not in kwargs]
    fixtures = {n: request.getfixturevalue(n) for n in wanted}
    return scenario(**fixtures, **kwargs)


@pytest.mark.parametrize("scenario, kwargs, expected", [
    pytest.param(_scn_sam2_refuses_bad_stacks_and_skips_empty_movies, {}, ['ValueError: SAM2 needs a (T, Y, X) mask stack, got shape (4, 4).', 'ValueError: Image stack shape (3, 4, 4) does not match mask stack shape (2, 4, 4).', 0, False], id='sam2-refuses-bad-stacks-and-skips-empty-movies'),
    pytest.param(_scn_sam2_drops_transient_tracks_and_plots_when_asked, {}, [[0, 1], [True]], id='sam2-drops-transient-tracks-and-plots-when-asked'),
    pytest.param(_scn_a_non_integer_parent_is_refused, {}, ['ValueError: sam2: non-integer parent for track 2'], id='a-non-integer-parent-is-refused'),
    pytest.param(_scn_event_features_of_an_empty_movie_are_empty, {}, [True], id='event-features-of-an-empty-movie-are-empty'),
    pytest.param(_scn_event_windows_of_no_tracks_are_empty, {}, [True], id='event-windows-of-no-tracks-are-empty'),
    pytest.param(_scn_annotations_accept_field_id_and_name_missing_columns, {}, [True, 'ValueError: Setting: timelapse_events_annotations <tmp>/ann.csv lacks the column(s) event, frame, track_id.'], id='annotations-accept-field-id-and-name-missing-columns'),
    pytest.param(_scn_divisions_skip_unknown_mothers_and_taken_daughters, {}, [True, [2, 3]], id='divisions-skip-unknown-mothers-and-taken-daughters'),
    pytest.param(_scn_event_detection_needs_tracks_and_matching_annotations, {}, ['ValueError: No trackpy tracks of cell in <tmp>/tracks.'], id='event-detection-needs-tracks-and-matching-annotations'),
])
def test_sam2_events_and_lineage_helpers_refuse_bad_input_or_return_empty_results(request, scenario, kwargs, expected):
    observed = _run_scenario(request, scenario, kwargs)
    assert observed == expected
