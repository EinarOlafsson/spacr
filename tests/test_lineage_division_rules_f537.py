"""Edge-entry, vacuole-count and bud rules for inferred divisions (F537).

On a public Toxoplasma endodyogeny movie (Cell Image Library CIL:10524,
public domain) most false inferred divisions came from objects entering at
the field edge and from mature rosettes segmented differently from frame to
frame. Each rule is checked on synthetic tracks, then all of them on the
tracks of that movie against hand-checked labels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr._lineage_trees import (_lineage_bud_signal, _lineage_edge_distance,
                                  _lineage_guard_options, _lineage_object_radius,
                                  _lineage_segments, _lineage_vacuoles,
                                  _run_lineage_step)

DATA = Path(__file__).parent / "data" / "lineage_cil10524"


def _track(track_id, frames, x, y, **extra):
    return [{"track_id": track_id, "frame": f, "x": x, "y": y, **extra} for f in frames]


def _divisions(seg):
    """``(birth frame, mother track)`` of every division."""
    found = []
    for _, mother in seg[seg["n_daughters"] > 0].iterrows():
        kids = seg[seg["parent_segment_id"] == mother["segment_id"]]
        found.append((int(kids["start_frame"].min()), int(mother["track_id"])))
    return sorted(found)


def _budding(x=200.0, y=200.0, start=10, end=30):
    """Mother 1 goes on; track 2 starts beside her at ``start``."""
    return pd.DataFrame(_track(1, range(0, end), x, y)
                        + _track(2, range(start, end), x + 8, y)
                        + _track(9, range(0, end), 20.0, 380.0))


def test_a_track_entering_at_the_edge_is_not_a_daughter():
    df = _budding(x=380.0, y=200.0)
    df.loc[df["track_id"] == 2, "x"] = 392.0
    assert _divisions(_lineage_segments(df, field_shape=(400, 400))) == [(10, 1)]
    seg = _lineage_segments(df, field_shape=(400, 400), entry_margin=10.0)
    assert _divisions(seg) == []
    assert _divisions(_lineage_segments(df, entry_margin=10.0)) == [(10, 1)]


def test_a_mother_still_crossing_into_the_field_does_not_divide():
    df = _budding(x=200.0, y=200.0)
    df["edge_distance"] = 100.0
    df.loc[(df["track_id"] == 1) & (df["frame"] == 8), "edge_distance"] = 4.0
    assert _divisions(_lineage_segments(df, entry_margin=10.0)) == []
    df["edge_distance"] = 100.0
    assert _divisions(_lineage_segments(df, entry_margin=10.0)) == [(10, 1)]


def test_an_edge_distance_column_overrides_the_field_shape():
    df = pd.DataFrame({"x": [1.0, 50.0], "y": [50.0, 50.0], "frame": [0, 0]})
    assert list(_lineage_edge_distance(df, (100, 100))) == [1.0, 49.0]
    df["edge_distance"] = [30.0, np.nan]
    assert list(_lineage_edge_distance(df, (100, 100))) == [30.0, np.inf]
    assert _lineage_edge_distance(df.drop(columns="edge_distance")) is None


def _rosette(n, x=200.0, y=200.0, spacing=8.0):
    return [(x + spacing * (i % 2), y + spacing * (i // 2)) for i in range(n)]


def _reshuffle():
    """A 4-rosette in which track 4 is lost and track 5 later takes its place."""
    rows = []
    for tid, (x, y) in enumerate(_rosette(4), start=1):
        frames = range(0, 10) if tid == 4 else range(0, 30)
        rows += _track(tid, frames, x, y)
    rows += _track(5, range(12, 30), 206.0, 210.0)
    rows += _track(9, range(0, 30), 20.0, 380.0)
    return pd.DataFrame(rows)


def test_a_reshuffle_inside_a_rosette_that_keeps_its_count_is_no_division():
    df = _reshuffle()
    assert _divisions(_lineage_segments(df)) != []
    assert _divisions(_lineage_segments(df, vacuole_window=6)) == []


def test_a_rosette_that_doubles_gives_its_divisions():
    rows = []
    for tid, (x, y) in enumerate(_rosette(2), start=1):
        rows += _track(tid, range(0, 30), x, y)
    rows += _track(3, range(12, 30), 200.0, 208.0)
    rows += _track(4, range(12, 30), 208.0, 208.0)
    rows += _track(9, range(0, 30), 20.0, 380.0)
    seg = _lineage_segments(pd.DataFrame(rows), vacuole_window=6)
    assert len(_divisions(seg)) == 2


def test_daughters_that_move_apart_still_count_together():
    rows = (_track(1, range(0, 10), 200.0, 200.0)
            + _track(2, range(10, 30), 180.0, 200.0)
            + _track(3, range(10, 30), 220.0, 200.0)
            + _track(9, range(0, 30), 20.0, 380.0))
    seg = _lineage_segments(pd.DataFrame(rows), vacuole_window=6)
    assert _divisions(seg) == [(10, 1)]


def test_a_mother_seen_too_briefly_has_no_count_and_does_not_divide():
    df = _budding(start=10)
    df = df[~((df["track_id"] == 1) & (df["frame"] < 9))]
    assert _divisions(_lineage_segments(df)) == [(10, 1)]
    assert _divisions(_lineage_segments(df, vacuole_window=6)) == []


def test_vacuoles_join_chains_of_neighbours():
    df = pd.DataFrame({"frame": [0, 0, 0, 0, 1], "x": [0.0, 10.0, 20.0, 90.0, 0.0],
                       "y": [0.0] * 5})
    keys, size = _lineage_vacuoles(df, 12.0)
    assert keys[0] == keys[1] == keys[2] != keys[3]
    assert size[keys[0]] == 3 and size[keys[3]] == 1 and size[keys[4]] == 1


def test_the_bud_guard_needs_a_bright_mother():
    df = _budding()
    df["bud_signal"] = 1.2
    assert _divisions(_lineage_segments(df)) == [(10, 1)]
    assert _divisions(_lineage_segments(df, bud_threshold=2.6)) == []
    df.loc[(df["track_id"] == 1) & (df["frame"] == 4), "bud_signal"] = 3.0
    assert _divisions(_lineage_segments(df, bud_threshold=2.6)) == [(10, 1)]
    df.loc[(df["track_id"] == 1) & (df["frame"] == 4), "bud_signal"] = np.nan
    df.loc[(df["track_id"] == 1) & (df["frame"] == 1), "bud_signal"] = 3.0
    assert _divisions(_lineage_segments(df, bud_threshold=2.6)) == []
    assert _divisions(_lineage_segments(df.drop(columns="bud_signal"),
                                        bud_threshold=2.6)) == [(10, 1)]


def test_the_bud_signal_scores_bright_outlines_inside_an_object():
    labels = np.zeros((2, 40, 40), np.int32)
    labels[:, 5:25, 5:25] = 3
    labels[1, 28:38, 28:38] = 4
    image = np.ones((2, 40, 40))
    image[0, 6:24, 14:16] = 9.0
    table = _lineage_bud_signal(labels, image)
    scores = {(r.frame, r.track_id): r.bud_signal for r in table.itertuples()}
    assert scores[(0, 3)] > 2.6
    assert scores[(1, 3)] == pytest.approx(1.0)
    assert scores[(1, 4)] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        _lineage_bud_signal(labels, image[:1])
    empty = _lineage_bud_signal(np.zeros((1, 4, 4), int), np.ones((1, 4, 4)))
    assert list(empty.columns) == ["frame", "track_id", "bud_signal"] and empty.empty


def test_the_guard_options_follow_the_settings():
    labels = np.zeros((2, 50, 50), np.int32)
    labels[:, 10:14, 10:30] = 1
    radius = _lineage_object_radius(labels)
    assert radius == pytest.approx(11.0, abs=1.5)
    assert _lineage_object_radius(None) is None
    assert _lineage_object_radius(np.zeros((2, 5, 5), int)) is None
    assert _lineage_guard_options({}, labels) == {"entry_margin": radius, "vacuole_window": 6}
    assert _lineage_guard_options({}, None, 30.0) == {"entry_margin": 15.0, "vacuole_window": 6}
    off = {"timelapse_lineage_edge_entry": False, "timelapse_lineage_vacuole_count": False}
    assert _lineage_guard_options(off) == {}
    bud = _lineage_guard_options({**off, "timelapse_lineage_bud_guard": True})
    assert bud == {"bud_threshold": 2.6}
    blank = {**off, "timelapse_lineage_bud_guard": True, "timelapse_lineage_bud_threshold": ""}
    assert _lineage_guard_options(blank) == {"bud_threshold": 2.6}
    set_ = {**off, "timelapse_lineage_bud_guard": True, "timelapse_lineage_bud_threshold": "2.3"}
    assert _lineage_guard_options(set_) == {"bud_threshold": 2.3}


@pytest.mark.parametrize("value", [True, "bright", float("inf"), [2]])
def test_the_bud_threshold_refuses_non_numbers(value):
    with pytest.raises(ValueError):
        _lineage_guard_options({"timelapse_lineage_bud_guard": True,
                                "timelapse_lineage_bud_threshold": value})


def _run_dir(tmp_path, df):
    src = tmp_path / "merged"
    src.mkdir()
    (tmp_path / "tracks").mkdir()
    df.to_csv(tmp_path / "tracks" / "trackpy_tracks_pathogen_p.csv", index=False)
    return str(src)


def _stack_for(df, shape=(30, 400, 400)):
    labels = np.zeros(shape, np.int32)
    for r in df.itertuples():
        y, x = int(r.y), int(r.x)
        labels[r.frame, y - 3:y + 3, x - 3:x + 3] = r.track_id
    return labels


def test_the_run_step_reads_the_bud_signal_from_the_images(tmp_path):
    df = _budding(start=10)
    src = _run_dir(tmp_path, df)
    labels = _stack_for(df)
    images = np.ones(labels.shape + (1,), np.float32)
    settings = {"save": False, "timelapse_lineage_bud_guard": True,
                "timelapse_lineage_min_division_h": ""}
    dull = _run_lineage_step(src, "p", "pathogen", "iou", settings,
                             label_stack=labels, images=images)
    assert _divisions(dull["segments"]) == []
    images[5, 198:202, 200, 0] = 9.0
    bright = _run_lineage_step(src, "p", "pathogen", "iou", settings,
                               label_stack=labels, images=images)
    assert _divisions(bright["segments"]) == [(10, 1)]
    tracks = pd.read_csv(Path(src).parent / "tracks" / "trackpy_tracks_pathogen_p.csv")
    assert "bud_signal" not in tracks.columns


def test_the_run_step_reports_a_bud_guard_it_cannot_apply(tmp_path, capsys):
    df = _budding(start=10)
    src = _run_dir(tmp_path, df)
    settings = {"save": False, "timelapse_lineage_bud_guard": True,
                "timelapse_lineage_min_division_h": ""}
    run = _run_lineage_step(src, "p", "pathogen", "iou", settings)
    assert _divisions(run["segments"]) == [(10, 1)]
    assert "bud guard skipped" in capsys.readouterr().out
    labels = _stack_for(df)
    run = _run_lineage_step(src, "p", "pathogen", "iou", settings,
                            label_stack=labels, images=np.ones((2, 4, 4)))
    assert _divisions(run["segments"]) == [(10, 1)]
    assert "bud guard skipped" in capsys.readouterr().out


def _cil_divisions(**settings):
    tracks = pd.read_csv(DATA / "tracks.csv")
    if not settings.pop("edge", True):
        tracks = tracks.drop(columns="edge_distance")
    if not settings.pop("bud", False):
        tracks = tracks.drop(columns="bud_signal")
    guards = _lineage_guard_options(settings, max_distance=30.0)
    if "entry_margin" in guards:
        guards["entry_margin"] = 14.5
    seg = _lineage_segments(tracks, max_distance=30.0, min_division_h=6.0,
                            frame_interval_s=600.0, **guards)
    return _divisions(seg)


def _cil_labels():
    labels = pd.read_csv(DATA / "division_labels.csv")
    return {(int(r.frame), int(r.mother_track_id)): r.label for r in labels.itertuples()}


def test_cil_movie_old_rules_give_the_hand_checked_42_divisions():
    found = _cil_divisions(timelapse_lineage_edge_entry=False,
                           timelapse_lineage_vacuole_count=False, edge=False)
    labels = _cil_labels()
    marks = [labels[d] for d in found]
    assert len(found) == 42
    assert (marks.count("R"), marks.count("E"), marks.count("A")) == (14, 4, 24)


def test_cil_movie_default_rules_keep_precision_and_most_real_divisions():
    labels = _cil_labels()
    found = _cil_divisions()
    marks = [labels.get(d, "?") for d in found]
    assert "?" not in marks
    real = marks.count("R") + marks.count("E")
    assert real / len(found) >= 0.8
    assert marks.count("R") / sum(v == "R" for v in labels.values()) >= 0.55


def test_cil_movie_bud_guard_keeps_only_real_divisions():
    labels = _cil_labels()
    found = _cil_divisions(timelapse_lineage_bud_guard=True, bud=True)
    assert found and all(labels.get(d) == "R" for d in found)


def test_cil_movie_bud_signal_is_high_in_a_budding_mother():
    excerpt = np.load(DATA / "bud_excerpt.npz")
    table = _lineage_bud_signal(excerpt["labels"], excerpt["imc1"])
    first = int(excerpt["first_frame"])
    mother = table[table["track_id"] == 8]
    assert mother["bud_signal"].max() >= 2.6
    assert int(mother.loc[mother["bud_signal"].idxmax(), "frame"]) + first < 10
