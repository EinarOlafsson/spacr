"""Scratch-wound closure on the runs and curves at its edges: a folder with
nothing to measure, a field whose first frame is missing, a well with one
frame, curves that start below half open or have one time point, and the
helpers asked about wells and sources they do not know."""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import measure as m
from tests.test_scratch_wound_closure import _band, _brightfield, _labels


def _settings(**over):
    settings = {"channels": [0], "cell_mask_dim": 1, "wound_source":
                "texture", "wound_hours_per_frame": None, "plot": False}
    settings.update(over)
    return settings


def test_the_texture_map_without_a_percentile_cut_is_the_raw_variance(
        monkeypatch):
    plane = _brightfield(_band(60)).astype(float)
    capped = m._wound_signal(plane, "texture", 15)
    monkeypatch.setattr(m, "_WOUND_TEXTURE_PERCENTILE", 0)
    raw = m._wound_signal(plane, "texture", 15)
    assert np.allclose(capped, raw / np.percentile(raw, 50))
    with pytest.raises(ValueError, match="unknown wound source 'phase'"):
        m._wound_open(plane, source="phase")


def test_a_wound_of_two_pixels_is_read_along_the_image_rows():
    wound = np.zeros((20, 20), bool)
    wound[10, 10:12] = True
    axis = m._wound_axis(wound, 2)
    assert tuple(axis.direction) == (1.0, 0.0)


def test_closure_metrics_on_empty_short_and_already_half_closed_curves():
    empty = m._closure_metrics([], [])
    assert empty["n_timepoints"] == 0 and np.isnan(empty["closure_rate"])
    early = m._closure_metrics([0.0, 4.0], [0.4, 0.3])
    assert early["half_closure_time"] == 0.0
    assert early["half_closure_reached"] == 1
    flat = m._closure_metrics([2.0, 2.0], [1.0, 0.9])
    assert np.isnan(flat["closure_rate"])


def test_wells_and_conditions_of_what_is_not_there():
    assert m._wound_condition_of("??", "x", {}) == "??x"
    assert m._wound_by_well(None).empty
    failed = pd.DataFrame({"status": ["no_wound"], "plateID": ["p"]})
    assert m._wound_by_well(failed).empty
    curves, conditions = m._wound_by_condition(None, pd.DataFrame())
    assert curves.empty and conditions.empty
    with pytest.raises(ValueError, match="the wound plane is 3"):
        m._wound_plane(np.zeros((8, 8, 2)), _settings(wound_channel=3))


def test_a_folder_with_nothing_to_measure_says_so(tmp_path, capsys):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "notes.txt").write_text("not a frame")
    np.save(merged / "junk.npy", np.zeros((4, 4, 2)))
    summary = m._run_wound_closure(str(merged), _settings())
    assert summary.empty
    printed = capsys.readouterr().out
    assert "Wound closure: skipping junk.npy" in printed
    assert "no merged frames to measure" in printed


def test_a_late_field_is_not_pooled_and_a_single_frame_draws_no_curve(
        tmp_path, capsys):
    merged = tmp_path / "merged"
    merged.mkdir()
    (tmp_path / "measurements").mkdir()
    open_mask = _band(120)
    stack = np.stack([_brightfield(open_mask), _labels(open_mask)], axis=-1)
    np.save(merged / "plate1_A01_1_0.npy", stack)
    for time, width in ((1, 120), (2, 90), (3, 60)):
        band = _band(width)
        np.save(merged / f"plate1_B02_1_{time}.npy",
                np.stack([_brightfield(band, seed=time), _labels(band)],
                         axis=-1))
    summary = m._run_wound_closure(str(merged), _settings())
    assert "2 well(s), 0 with a closure curve" in capsys.readouterr().out
    assert summary["wound_ok"].tolist() == [0, 0]
    db = tmp_path / "measurements" / "measurements.db"
    with sqlite3.connect(db) as conn:
        fields = pd.read_sql_query(f"SELECT * FROM {m._WOUND_TABLE}", conn)
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    late = fields[fields["columnID"] == "c2"]
    assert set(late["status"]) == {"missing_start"}
    assert m._WOUND_CONDITION_TABLE not in names
    out = tmp_path / "results" / "wound_closure"
    assert (out / "wound_closure_per_well.csv").is_file()
    assert not (out / "wound_closure_per_condition.csv").exists()
    assert not any(n.startswith("closure_curves") for n in os.listdir(out))
