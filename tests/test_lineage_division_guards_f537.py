"""Guards against spurious inferred divisions in lineage trees.

Motility movies of parasites that leave the field, flicker or break gave
only artefact divisions. Inference now needs a movie long enough for a
division, daughters that persist, a mother seen without a gap, a mother that
did not leave at the field edge, and cell cycles no shorter than the
minimum division interval.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr._lineage_trees import (_lineage_drop_short_cycles,
                                  _lineage_min_division_h, _lineage_segments,
                                  _run_lineage_step)


def _track(track_id, frames, x, y):
    return [{"track_id": track_id, "frame": f, "x": x, "y": y} for f in frames]


def _division(mother_end=9, daughter_frames=10, x=200.0, y=200.0, gap=False):
    """Mother 1 ends at ``mother_end``; tracks 2 and 3 start beside her."""
    mother = [f for f in range(mother_end + 1) if not (gap and f == mother_end - 1)]
    start = mother_end + 1
    rows = (_track(1, mother, x, y)
            + _track(2, range(start, start + daughter_frames), x - 4, y)
            + _track(3, range(start, start + daughter_frames), x + 4, y)
            + _track(9, range(0, start + 10), 20.0, 20.0))
    return pd.DataFrame(rows)


def _n_divisions(seg):
    return int((seg["n_daughters"] > 0).sum())


def test_a_clean_division_is_still_found():
    seg = _lineage_segments(_division(), max_distance=30)
    assert _n_divisions(seg) == 1


def test_daughters_that_flicker_are_not_a_division():
    seg = _lineage_segments(_division(daughter_frames=1), max_distance=30)
    assert _n_divisions(seg) == 0
    loose = _lineage_segments(_division(daughter_frames=1), max_distance=30,
                              persist_frames=1)
    assert _n_divisions(loose) == 1


def test_a_mother_seen_after_a_gap_is_not_a_division():
    seg = _lineage_segments(_division(gap=True), max_distance=30)
    assert _n_divisions(seg) == 0


def test_a_mother_ending_at_the_field_edge_left_the_field():
    df = _division(x=5.0, y=200.0)
    assert _n_divisions(_lineage_segments(df, max_distance=30)) == 1
    seg = _lineage_segments(df, max_distance=30, field_shape=(400, 400))
    assert _n_divisions(seg) == 0
    inside = _lineage_segments(_division(), max_distance=30, field_shape=(400, 400))
    assert _n_divisions(inside) == 1


def test_a_movie_much_shorter_than_a_division_has_none():
    df = _division()
    assert _n_divisions(_lineage_segments(df, max_distance=30, min_division_h=6.0)) == 1
    short = _lineage_segments(df, max_distance=30, min_division_h=6.0,
                              frame_interval_s=156.0)
    assert _n_divisions(short) == 0
    timed = df.assign(time_s=df["frame"] * 156.0)
    assert _n_divisions(_lineage_segments(timed, max_distance=30, min_division_h=6.0)) == 0
    long = _lineage_segments(df, max_distance=30, min_division_h=6.0,
                             frame_interval_s=3600.0)
    assert _n_divisions(long) == 1


def _spans(spans):
    return pd.DataFrame(spans, columns=["track_id", "start", "end"]).set_index("track_id")


def test_a_cycle_shorter_than_the_minimum_keeps_the_stronger_division():
    spans = _spans([(1, 0, 9), (2, 10, 12), (3, 10, 40), (4, 13, 40), (5, 13, 40)])
    parents = {2: 1, 3: 1, 4: 2, 5: 2}
    assert _lineage_drop_short_cycles(parents, set(), spans, 12) == {2: 1, 3: 1}
    assert _lineage_drop_short_cycles(parents, set(), spans, 2) == parents
    known = _lineage_drop_short_cycles(parents, {4, 5}, spans, 12)
    assert known == parents
    went_on = _spans([(1, 0, 12), (2, 10, 40), (4, 13, 40), (5, 13, 40)])
    assert _lineage_drop_short_cycles({2: 1, 4: 1, 5: 1}, set(), went_on, 12) == {4: 1, 5: 1}
    three = _spans([(1, 0, 9), (2, 10, 40), (3, 10, 40), (6, 10, 12),
                    (4, 13, 40), (5, 13, 40)])
    kept = _lineage_drop_short_cycles({2: 1, 3: 1, 6: 1, 4: 6, 5: 6}, set(), three, 12)
    assert kept == {2: 1, 3: 1, 4: 6, 5: 6}


@pytest.mark.parametrize("value, expected", [
    (None, None), ("", None), (0, None), (-1, None), (6, 6.0), ("7.5", 7.5)])
def test_the_minimum_division_setting_reads_hours(value, expected):
    assert _lineage_min_division_h({"timelapse_lineage_min_division_h": value}) == expected
    assert _lineage_min_division_h({}) == 6.0


@pytest.mark.parametrize("value", [True, "soon", float("nan"), [6]])
def test_the_minimum_division_setting_refuses_non_numbers(value):
    with pytest.raises(ValueError):
        _lineage_min_division_h({"timelapse_lineage_min_division_h": value})


def test_the_run_step_applies_the_default_and_the_field_shape(tmp_path):
    src = tmp_path / "merged"
    src.mkdir()
    (tmp_path / "tracks").mkdir()
    _division(x=5.0).to_csv(tmp_path / "tracks" / "trackpy_tracks_cell_p.csv", index=False)
    run = _run_lineage_step(str(src), "p", "cell", "iou",
                            {"frame_interval_s": 156.0, "save": False})
    assert _n_divisions(run["segments"]) == 0
    off = _run_lineage_step(str(src), "p", "cell", "iou",
                            {"frame_interval_s": 156.0, "save": False,
                             "timelapse_lineage_min_division_h": ""})
    assert _n_divisions(off["segments"]) == 1
    assert np.isfinite(off["segments"]["n_frames"]).all()
