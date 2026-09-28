"""Time to event on the settings, tables and groupings at its edges: typed
lists, origins and frame lengths that cannot work, a group column read from
the object table, grouping by plate, row, column or field, one condition
only, a Cox model that cannot be fitted, and the step's own report."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import measure as m
from tests.test_time_to_event import (SIMULATED, _config, _simulated_database,
                                      _tracks)


def test_settings_typed_as_text_and_values_that_cannot_work():
    assert m._tte_list("a, b,,c") == ["a", "b", "c"]
    with pytest.raises(ValueError, match="time_to_event_origin"):
        m._tte_settings({"time_to_event_origin": "birth"})
    with pytest.raises(ValueError, match="hours_per_frame must be positive"):
        m._tte_settings({"time_to_event_hours_per_frame": -1})


def _objects():
    return pd.DataFrame({
        "plateID": ["p1", "p1", "p2", "p2"], "rowID": ["r1", "r2"] * 2,
        "columnID": ["c1", "c2"] * 2, "fieldID": ["f1", "f2"] * 2,
        "object_label": [1, 2, 3, 4], "duration": [1.0, 2.0, 3.0, 4.0],
        "event": [1, 1, 0, 1], "cell_line": ["a", "b", "a", "b"]})


@pytest.mark.parametrize("group,expected", (
    ("plate", ["p1", "p1", "p2", "p2"]),
    ("row", ["r1", "r2", "r1", "r2"]),
    ("column", ["c1", "c2", "c1", "c2"]),
    ("field", ["p1_r1_c1_f1", "p1_r2_c2_f2", "p2_r1_c1_f1",
               "p2_r2_c2_f2"]),
    ("cell_line", ["a", "b", "a", "b"]),
))
def test_objects_are_grouped_by_what_the_setting_names(group, expected):
    objects = _objects()
    objects["well"] = ["A01", "B02", "A01", "B02"]
    grouped, order = m._time_to_event_groups(objects, _config(group=group))
    assert grouped["condition"].tolist() == expected
    assert sorted(order) == sorted(set(expected))


def test_a_group_column_on_the_object_table_is_read_and_kept(tmp_path):
    db = _simulated_database(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute("ALTER TABLE cell ADD COLUMN cell_line TEXT")
        conn.execute("UPDATE cell SET cell_line = CASE WHEN columnID IN "
                     "('c1', 'c2') THEN 'wt' ELSE 'ko' END")
    settings = {**SIMULATED, "time_to_event_conditions": [],
                "time_to_event_group": "cell_line",
                "time_to_event_reference": "wt",
                "time_to_event_column": "cell_area",
                "time_to_event_mode": "above",
                "time_to_event_threshold": 1e9}
    result = m._time_to_event(db, settings, plot=False, engine="builtin")
    assert set(result["objects"]["condition"]) == {"wt", "ko"}
    assert int(result["objects"]["event"].sum()) == 0


def test_a_column_nowhere_in_the_database_is_refused(tmp_path):
    db = _simulated_database(tmp_path)
    with pytest.raises(ValueError, match="no column 'dose' to group"):
        m._read_time_to_event_inputs(db, _config(covariates=["dose"]))
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE png_list")
    with pytest.raises(ValueError, match="neither a column"):
        m._read_time_to_event_inputs(db, _config(mode="annotated",
                                                 column="dead"))


def test_movies_are_read_from_the_tracks_when_not_given():
    frame, _movies = _tracks({1: (0, 4, None), 2: (0, 9, None)})
    objects, _dropped = m._time_to_event_objects(frame, _config(), None)
    assert objects.set_index("object_label").loc[2, "censored_at"] == \
        "movie_end"


def test_no_followable_object_or_no_named_well_is_refused(tmp_path):
    db = _simulated_database(tmp_path)
    with pytest.raises(ValueError, match="could be followed"):
        m._time_to_event(db, {**SIMULATED, "time_to_event_min_frames": 999},
                         plot=False, engine="builtin")
    with pytest.raises(ValueError, match="no tracked object is in a well"):
        m._time_to_event(db, {**SIMULATED,
                              "time_to_event_conditions": ["x=c9"],
                              "time_to_event_reference": ""},
                         plot=False, engine="builtin")


def test_one_condition_draws_one_curve_without_tests(tmp_path):
    db = _simulated_database(tmp_path)
    settings = {**SIMULATED, "time_to_event_conditions": ["all=c1,c2,c3,c4"],
                "time_to_event_reference": "all",
                "time_to_event_covariates": []}
    result = m._time_to_event(db, settings, plot=True, engine="builtin")
    assert result["tests"].empty and result["cox"].empty
    assert [p.rsplit("/", 1)[-1] for p in result["figures"]] == [
        "kaplan_meier.pdf"]


def test_a_cox_model_that_cannot_be_fitted_is_reported(tmp_path, capsys,
                                                       monkeypatch):
    db = _simulated_database(tmp_path)

    def refuse(*_a, **_k):
        raise np.linalg.LinAlgError("singular matrix")

    import spacr.sp_stats as stats

    monkeypatch.setattr(stats, "_cox_regression", refuse)
    result = m._run_time_to_event_step(db, {**SIMULATED})
    assert result["cox"].empty
    printed = capsys.readouterr().out
    assert "Time to event (" in printed and "100 tracked objects" in printed
    assert "no Cox model (LinAlgError: singular matrix)" in printed


def test_many_conditions_show_the_first_few(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(m, "_TTE_FIGURE_GROUPS", 1)
    curves = pd.DataFrame({"condition": ["a", "a", "b", "b"],
                           "time": [0.0, 1.0, 0.0, 1.0],
                           "survival": [1.0, 0.5, 1.0, 0.8],
                           "ci_lower": [1.0, 0.3, 1.0, 0.6],
                           "ci_upper": [1.0, 0.7, 1.0, 0.9],
                           "censored": [0, 1, 0, 0], "time_unit": "h"})
    summary = pd.DataFrame({"level": ["condition"] * 2, "group": ["a", "b"],
                            "n": [4, 4], "events": [2, 1]})
    figure = m._time_to_event_figure(curves, summary, pd.DataFrame(), "run")
    try:
        assert figure.axes[0].get_title() == "run; first 1 shown"
    finally:
        plt.close(figure)
