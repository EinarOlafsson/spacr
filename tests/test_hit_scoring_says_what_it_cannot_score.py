"""Arrayed-screen hit scoring on the screens a user gets wrong: unknown
options, missing columns, controls named twice, a plate without a usable
negative control, a plate too sparse for a B-score, and a heatmap that has
nothing to draw. Each is refused with a sentence naming the problem, or
scored with a note saying what could not be scored and why."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import sp_stats as S


def _plate(plate="P1", *, neg=None, rows=8, cols=12, seed=0, treatment=False):
    """One 96-well plate: negative control in column 1, positive in 12."""
    rng = np.random.default_rng(seed)
    records = []
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            kind = "neg" if c == 1 else ("pos" if c == cols else "sample")
            value = 100.0 + rng.normal(0, 2.0) + (40.0 if kind == "pos" else 0)
            if neg is not None and kind == "neg":
                value = neg
            record = {"prc": f"{plate}_r{r}_c{c}", "signal": value,
                      "well_type": kind}
            if treatment:
                record["gene"] = f"g{c}"
            records.append(record)
    return pd.DataFrame(records)


def _wells(frame, **kw):
    kw.setdefault("control_column", "well_type")
    kw.setdefault("negative_levels", "neg")
    kw.setdefault("positive_levels", "pos")
    return S.screen_wells(frame, "signal", **kw)


def test_the_basic_statistics_answer_empty_and_flat_inputs_with_nan():
    assert np.isnan(S.mad([np.nan, np.inf]))
    assert S.mad([1, 2, 3, 4, 100], scale=False) == 1.0
    assert np.isnan(S.ssmd_unreplicated([5.0], [1.0, 2.0], "umvue")).all()
    assert np.isnan(S.ssmd_unreplicated([5.0], [3.0, 3.0, 3.0], "mm")).all()
    assert np.isnan(S.ssmd_replicated([2.0, 2.0, 2.0], "robust"))
    assert np.isnan(S.ssmd_replicated([2.0, 3.0], "umvue"))
    assert np.isnan(S.ssmd_replicated([2.0, 2.0, 2.0], "mm"))
    with pytest.raises(S.HitScoringError, match="two-dimensional"):
        S.median_polish([1.0, 2.0, 3.0])
    noisy = np.random.default_rng(0).normal(0, 1, (5, 6))
    polish = S.median_polish(noisy, max_iter=1, eps=0.0)
    assert polish.iterations == 1 and not polish.converged
    scores, _polish, scale = S.b_scores(np.full((4, 6), 5.0))
    assert np.isnan(scores).all() and not scale > 0


@pytest.mark.parametrize("kwargs,message", (
    ({"grouping": "mode"}, "grouping must be 'mean' or 'median'"),
    ({"control_column": "nope"}, "control column 'nope' is not in the table"),
    ({"negative_levels": "neg", "positive_levels": "neg"},
     "neg is named both negative and positive control"),
    ({"control_column": None}, "control levels are named but no control"),
    ({"min_count": 5}, "no well has 5 or more objects"),
    ({"control_column": None, "negative_levels": (), "positive_levels": (),
      "negative_wells": "c99"}, ""),
))
def test_screen_wells_refuses_what_it_cannot_score(kwargs, message):
    with pytest.raises(S.HitScoringError, match=message):
        _wells(_plate(), **kwargs)


def test_screen_wells_refuses_an_empty_table_and_a_missing_value():
    with pytest.raises(S.HitScoringError, match="the table is empty"):
        S.screen_wells(pd.DataFrame(), "signal")
    with pytest.raises(S.HitScoringError, match="no column 'area'"):
        S.screen_wells(_plate(), "area")


def test_screen_wells_needs_a_well_it_can_read():
    frame = pd.DataFrame({"rowID": ["??", "!!"], "columnID": ["x", "y"],
                          "plateID": ["P1", "P1"], "signal": [1.0, 2.0]})
    with pytest.raises(S.HitScoringError):
        S.screen_wells(frame, "signal")
    unreadable = pd.DataFrame({"signal": [1.0, 2.0]})
    with pytest.raises(S.HitScoringError):
        S.screen_wells(unreadable, "signal")


def test_a_plate_column_names_the_plates_and_a_level_list_is_accepted():
    frame = _plate()
    frame["plate_name"] = "screen_A"
    wells = _wells(frame, plate_column="plate_name",
                   negative_levels=["neg", " "], positive_levels=None)
    assert set(wells["plateID"]) == {"screen_A"}
    assert (wells["role"] == S.ROLE_NEGATIVE).sum() == 8
    assert (wells["role"] == S.ROLE_POSITIVE).sum() == 0


@pytest.mark.parametrize("kwargs,message", (
    ({"rank_by": "fold"}, "unknown method 'fold'"),
    ({"direction": "sideways"}, "unknown direction 'sideways'"),
))
def test_score_screen_refuses_an_unknown_method_or_direction(kwargs, message):
    with pytest.raises(S.HitScoringError, match=message):
        S.score_screen(_wells(_plate()), **kwargs)


def test_a_plate_without_a_usable_negative_control_is_scored_with_a_note():
    flat = _plate("P2", neg=100.0, seed=2)
    frame = pd.concat([_plate("P1"), flat], ignore_index=True)
    single = _plate("P3", seed=3)
    single = single[~((single["well_type"] == "neg")
                      & (single["prc"] != "P3_r1_c1"))]
    frame = pd.concat([frame, single], ignore_index=True)
    result = S.score_screen(_wells(frame), direction="down",
                            rank_by="robust_z")
    report = result.report()
    assert "plate P3: fewer than two negative-control wells" in report
    assert "plate P2: the negative control cannot support" in report
    p3 = result.wells[result.wells["plateID"] == "P3"]
    assert p3["robust_z"].isna().all() and p3["ssmd"].isna().all()
    assert result.wells.loc[result.wells["plateID"] == "P1", "robust_z"
                            ].notna().any()


def test_a_plate_too_sparse_for_a_b_score_says_so_and_draws_nothing(
        tmp_path):
    frame = _plate(rows=2, cols=3)
    result = S.score_screen(_wells(frame))
    assert "too few sample wells for a B-score" in result.report()
    written = S.write_hit_report(result, tmp_path / "hits",
                                 methods=("b_score",), target="print")
    assert "hit_heatmap_b_score" not in written
    assert "hit_table" in written
    figure, panel = S.hit_heatmap(result, "b_score", target="print")
    assert not panel.drawn
    with pytest.raises(S.HitScoringError, match="unknown method 'fold'"):
        S.hit_heatmap(result, "fold")


def test_treatments_that_score_nothing_give_an_empty_table_with_its_columns():
    wells = _wells(_plate())
    scored = S.score_screen(wells).wells
    scored["treatment"] = np.nan
    table = S.treatment_ssmd(scored)
    assert table.empty and "ssmd" in table.columns and "rank" in table.columns


def test_the_hit_table_can_list_every_sample_well():
    result = S.score_screen(_wells(_plate()))
    every = S.hit_table(result.wells, hits_only=False)
    assert len(every) == int((result.wells["role"] == S.ROLE_SAMPLE).sum())
    assert list(every.columns[:4]) == ["rank", "plateID", "well", "prc"]


def test_the_plate_summary_keeps_going_when_zprime_cannot_be_computed(
        monkeypatch):
    from spacr.qt.widgets import control_chart

    def broken(*_a, **_k):
        raise ValueError("no controls")

    monkeypatch.setattr(control_chart, "zprime_frame", broken)
    result = S.score_screen(_wells(_plate()))
    assert result.plates["zprime"].isna().all()
    assert "Z' =" not in result.report()


def test_a_well_without_a_control_label_is_a_sample_and_the_report_counts_treatments():
    frame = _plate(treatment=True)
    frame.loc[frame["prc"] == "P1_r2_c5", "well_type"] = np.nan
    result = S.score_screen(_wells(frame, treatment_column="gene"))
    well = result.wells.set_index("prc").loc["P1_r2_c5"]
    assert well["role"] == S.ROLE_SAMPLE
    assert "Replicate SSMD (umvue) over 10 treatment(s)" in result.report()


def test_a_plate_larger_than_any_layout_is_read_against_the_largest():
    rows = [{"prc": f"P1_r{r}_c{c}", "signal": float(r + c)}
            for r in (1, 40) for c in (1, 2)]
    wells = S.screen_wells(pd.DataFrame(rows), "signal", negative_wells="c1")
    negative = wells[wells["role"] == S.ROLE_NEGATIVE]
    assert negative["well"].tolist() == ["A01"], (
        "column 1 of a 1536-well plate stops at row 32")
