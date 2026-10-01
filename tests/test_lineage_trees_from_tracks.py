"""Lineage trees drawn from a timelapse's tracks.

A synthetic field with known divisions: generation times, the tree shape,
the Newick export and the per-lineage statistics must all agree with the
tracks they came from.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.timelapse import (_lineage_newick, _lineage_segments,
                             _lineage_statistics, _lineage_trees_from_tracks,
                             _run_lineage_step)


def _track(track_id, frames, x, y, **extra):
    return [{"track_id": track_id, "frame": f, "x": x, "y": y, **extra}
            for f in frames]


def _field():
    """Track 1 divides at frame 5 into 2 and 3 (both new ids).

    Track 2 divides at frame 13 keeping its id, with 4 as the new sister;
    track 3 divides at frame 15 into 5 and 6. Track 7 never divides, 8
    enters mid-movie far from everything, and 9 breaks into 10 (one new
    track beside an ending one is not a division).
    """
    rows = (_track(1, range(0, 5), 50, 50)
            + _track(2, range(5, 21), 45, 50)
            + _track(3, range(5, 15), 55, 50)
            + _track(4, range(13, 21), 42, 52)
            + _track(5, range(15, 21), 58, 50)
            + _track(6, range(15, 21), 55, 54)
            + _track(7, range(0, 21), 200, 200)
            + _track(8, range(10, 21), 300, 300)
            + _track(9, range(0, 7), 100, 100)
            + _track(10, range(7, 21), 100, 101))
    df = pd.DataFrame(rows)
    df["cell_area"] = df["track_id"] * 10.0
    return df


def test_generation_times_match_the_tracks():
    seg = _lineage_segments(_field(), max_distance=30)
    complete = seg.dropna(subset=["generation_time"])
    got = {(int(r.track_id), int(r.start_frame)): r.generation_time
           for r in complete.itertuples()}
    assert got == {(2, 5): 8.0, (3, 5): 10.0}

    root = seg[(seg.track_id == 1)].iloc[0]
    assert root.parent_segment_id == 0 and root.n_daughters == 2
    assert np.isnan(root.generation_time)

    two = seg[seg.track_id == 2].sort_values("start_frame")
    assert two.start_frame.tolist() == [5, 13]
    assert two.end_frame.tolist() == [12, 20]
    assert two.iloc[1].parent_segment_id == two.iloc[0].segment_id
    four = seg[seg.track_id == 4].iloc[0]
    assert four.parent_segment_id == two.iloc[0].segment_id
    assert four.generation == 2

    for lonely in (7, 8, 10):
        row = seg[seg.track_id == lonely].iloc[0]
        assert row.parent_segment_id == 0 and row.lineage_id == row.segment_id


def test_explicit_parent_links_override_inference():
    df = pd.DataFrame(_track(1, range(0, 4), 10, 10, parent_track_id=0)
                      + _track(2, range(4, 9), 400, 400, parent_track_id=1)
                      + _track(3, range(4, 9), 800, 10, parent_track_id=1))
    seg = _lineage_segments(df, max_distance=5)
    mother = seg[seg.track_id == 1].iloc[0]
    assert mother.n_daughters == 2 and mother.division_source == "tracker"
    assert set(seg[seg.parent_segment_id == mother.segment_id].track_id) == {2, 3}


def test_newick_and_statistics_describe_the_same_tree():
    seg = _lineage_segments(_field(), max_distance=30)
    newick = _lineage_newick(seg).splitlines()
    assert len(newick) == 5
    assert newick[0] == ("((t4_f13:8,t2_f13:8)t2_f5:8,(t5_f15:6,t6_f15:6)"
                         "t3_f5:10)t1_f0:5;")

    stats = _lineage_statistics(seg)
    tree = stats[stats.root_track_id == 1].iloc[0]
    assert tree.n_divisions == 3 and tree.n_cells == 7
    assert tree.max_generation == 2 and tree.n_complete_cycles == 2
    assert tree.generation_time_mean == pytest.approx(9.0)
    overall = stats.iloc[-1]
    assert overall.lineage_id == "all" and overall.n_divisions == 3


def test_sibling_correlation_needs_three_pairs():
    rows = []
    tid = 1
    for k, (a, b) in enumerate([(4, 5), (6, 7), (8, 8), (10, 11)]):
        x = 100 * (k + 1)
        mother, d1, d2 = tid, tid + 1, tid + 2
        tid += 7
        rows += _track(mother, range(0, 2), x, 10)
        rows += _track(d1, range(2, 2 + a), x - 3, 10)
        rows += _track(d2, range(2, 2 + b), x + 3, 10)
        for d, n, dx in ((d1, a, -3), (d2, b, 3)):
            rows += _track(tid, range(2 + n, 2 + n + 3), x + dx - 1, 12)
            rows += _track(tid + 1, range(2 + n, 2 + n + 3), x + dx + 1, 12)
            tid += 2
    seg = _lineage_segments(pd.DataFrame(rows), max_distance=5)
    stats = _lineage_statistics(seg)
    overall = stats.iloc[-1]
    assert overall.sibling_pairs == 4
    assert overall.sibling_correlation > 0.9
    assert stats.iloc[0].sibling_pairs == 1
    assert np.isnan(stats.iloc[0].sibling_correlation)


def test_the_run_step_writes_every_output(tmp_path):
    src = tmp_path / "run" / "merged"
    tracks_dir = tmp_path / "run" / "tracks"
    tracks_dir.mkdir(parents=True)
    path = tracks_dir / "trackpy_tracks_cell_plate1_A01_1.csv"
    _field().to_csv(path, index=False)

    result = _run_lineage_step(str(src), "plate1_A01_1", "cell", "iou",
                               {"timelapse_lineage_color_by": "cell_area",
                                "timelapse_lineage_max_distance": 30.0,
                                "save": True})
    paths = result["paths"]
    for key in ("segments", "statistics", "newick", "figure"):
        assert (tmp_path / "run" / "tracks" / "lineage").as_posix() in paths[key]
    segments = pd.read_csv(paths["segments"])
    four = segments[segments.track_id == 4].iloc[0]
    assert four.color_cell_area == pytest.approx(40.0)


def test_an_unknown_colour_falls_back_to_generation_time(tmp_path, capsys):
    path = tmp_path / "tracks.csv"
    _field().to_csv(path, index=False)
    result = _lineage_trees_from_tracks(str(path), color_by="nope", plot=False)
    assert "color_generation_time" in result["segments"].columns
    assert "colouring by generation_time instead" in capsys.readouterr().out
    assert "figure" not in result["paths"]


def test_missing_tracks_are_reported_not_raised(tmp_path, capsys):
    assert _run_lineage_step(str(tmp_path / "merged"), "x", "cell",
                             "trackastra", {}) is None
    assert "no tracks table" in capsys.readouterr().out
