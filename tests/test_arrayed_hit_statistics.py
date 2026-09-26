"""Arrayed-screen hit statistics (item 570): SSMD, robust z, B-score.

Every statistic is checked against a value worked out by hand on a plate
small enough to do it on paper, against a published reference where one
exists (R's ``medpolish`` example, scipy's MAD, pycytominer's RobustMAD when
it is installed), and on synthetic plates with planted hits and a planted
row/column gradient.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from spacr import sp_stats as S


def test_mad_scale_is_one_over_the_normal_quartile():
    assert S.MAD_SCALE == pytest.approx(1.482602218505602, rel=1e-12)


def test_robust_z_by_hand():
    """Reference 1, 2, 3, 4, 100: median 3, absolute deviations 2, 1, 0, 1,
    97, raw MAD 1. A well at 9 sits (9 - 3) / 1.4826 robust sigma above."""
    z = S.robust_z_scores([9.0, 3.0, 0.0], [1, 2, 3, 4, 100])
    assert z[0] == pytest.approx(6 / S.MAD_SCALE)
    assert z[1] == 0.0
    assert z[2] == pytest.approx(-3 / S.MAD_SCALE)


def test_robust_z_matches_scipy_mad():
    from scipy.stats import median_abs_deviation

    rng = np.random.default_rng(3)
    neg = rng.normal(5, 2, 50)
    values = rng.normal(6, 3, 30)
    expected = (values - np.median(neg)) / median_abs_deviation(
        neg, scale="normal")
    assert np.allclose(S.robust_z_scores(values, neg), expected, rtol=1e-12)


def test_robust_z_matches_pycytominer_robust_mad():
    """pycytominer's mad_robustize, fitted on the negative control, is the
    robust z. It uses 1.4826 rounded, so agreement is to 1e-5."""
    transform = pytest.importorskip("pycytominer.operations.transform")
    rng = np.random.default_rng(1)
    neg = rng.normal(10, 2, 40)
    values = rng.normal(12, 3, 200)
    model = transform.RobustMAD().fit(pd.DataFrame({"x": neg}))
    expected = model.transform(pd.DataFrame({"x": values}))["x"].to_numpy()
    assert np.allclose(S.robust_z_scores(values, neg), expected, rtol=1e-5)


def test_robust_z_without_a_usable_reference_is_nan():
    assert np.isnan(S.robust_z_scores([1.0, 2.0], [5.0])).all()
    assert np.isnan(S.robust_z_scores([1.0, 2.0], [5.0, 5.0, 5.0])).all()


def test_unreplicated_ssmd_by_hand():
    """Reference 1..5: mean 3, s = sqrt(2.5), median 3, raw MAD 1. A well at
    8 is five above the mean.

    MM: 5 / (sqrt(2) sqrt(2.5)) = sqrt(5).
    UMVUE: K / (n - 1) = 2 / (4 Gamma(1.5)^2) = 2 / pi, so sqrt(5) sqrt(2/pi)
    = sqrt(10 / pi).
    Robust: 5 / (1.4826 sqrt(2)).
    """
    ref = [1, 2, 3, 4, 5]
    assert S.ssmd_unreplicated([8.0], ref, "mm")[0] == pytest.approx(
        math.sqrt(5))
    assert S.ssmd_unreplicated([8.0], ref, "umvue")[0] == pytest.approx(
        math.sqrt(10 / math.pi))
    assert S.ssmd_unreplicated([8.0], ref, "robust")[0] == pytest.approx(
        5 / (S.MAD_SCALE * math.sqrt(2)))


def test_robust_ssmd_is_robust_z_over_root_two():
    rng = np.random.default_rng(5)
    neg = rng.normal(0, 1, 30)
    values = rng.normal(1, 2, 20)
    assert np.allclose(S.ssmd_unreplicated(values, neg, "robust"),
                       S.robust_z_scores(values, neg) / math.sqrt(2))


def test_replicated_ssmd_by_hand():
    """Differences 2, 3, 4: mean 3, s 1, median 3, raw MAD 1.

    MM: 3. UMVUE at n = 3: sqrt(2/2) Gamma(1) / Gamma(1/2) = 1/sqrt(pi), so
    3 / sqrt(pi). Robust: 3 / 1.4826.
    """
    d = [2.0, 3.0, 4.0]
    assert S.ssmd_replicated(d, "mm") == pytest.approx(3.0)
    assert S.ssmd_replicated(d, "umvue") == pytest.approx(3 / math.sqrt(math.pi))
    assert S.ssmd_replicated(d, "robust") == pytest.approx(3 / S.MAD_SCALE)
    assert math.isnan(S.ssmd_replicated([1.0, 2.0], "umvue"))
    assert math.isnan(S.ssmd_replicated([1.0], "mm"))


def test_replicated_umvue_is_unbiased():
    """The UMVUE's defining property, by simulation: its mean over many
    draws is the true SSMD, where the method-of-moments estimate is biased
    upward at small n."""
    rng = np.random.default_rng(11)
    n, beta = 4, 1.5
    draws = rng.normal(beta, 1.0, size=(40000, n))
    umvue = np.array([S.ssmd_replicated(row, "umvue") for row in draws])
    mm = np.array([S.ssmd_replicated(row, "mm") for row in draws])
    assert umvue.mean() == pytest.approx(beta, abs=0.03)
    assert mm.mean() > beta + 0.3


def test_unreplicated_umvue_is_unbiased():
    """Same property for a single well against n_N = 6 negative wells: the
    true SSMD is (mu_i - mu_N) / (sqrt(2) sigma)."""
    rng = np.random.default_rng(12)
    reps, n_neg, shift = 40000, 6, 3.0
    neg = rng.normal(0.0, 1.0, size=(reps, n_neg))
    wells = rng.normal(shift, 1.0, size=reps)
    est = np.array([S.ssmd_unreplicated([w], r, "umvue")[0]
                    for w, r in zip(wells, neg)])
    assert est.mean() == pytest.approx(shift / math.sqrt(2), abs=0.03)


def test_unknown_estimator_is_refused():
    with pytest.raises(S.HitScoringError):
        S.ssmd_unreplicated([1.0], [1, 2, 3], "t")
    with pytest.raises(S.HitScoringError):
        S.ssmd_replicated([1.0, 2.0, 3.0], "t")


def test_median_polish_reproduces_the_r_medpolish_example():
    """The ``deaths`` table of R's ``?medpolish`` and its printed result:
    overall 8, row effects 6 -1 0 2 -8, column effects 0 -1 0."""
    deaths = np.array([[14, 15, 14], [7, 4, 7], [8, 2, 10], [15, 9, 10],
                       [0, 2, 0]], dtype=float)
    fit = S.median_polish(deaths)
    assert fit.overall == 8.0
    assert np.array_equal(fit.row, [6, -1, 0, 2, -8])
    assert np.array_equal(fit.column, [0, -1, 0])
    assert np.array_equal(fit.residuals, [[0, 2, 0], [0, -2, 0], [0, -5, 2],
                                          [5, 0, 0], [0, 3, 0]])
    assert fit.converged


def test_median_polish_recovers_an_additive_plate_exactly():
    rows = np.array([0.0, 2.0, -1.0, 5.0])
    cols = np.array([1.0, -3.0, 0.0, 4.0, 2.0, -2.0])
    plate = 10.0 + rows[:, None] + cols[None, :]
    fit = S.median_polish(plate)
    assert np.allclose(fit.residuals, 0.0)
    assert np.allclose(fit.fitted(), plate)


def test_median_polish_leaves_an_empty_row_without_an_effect():
    plate = np.arange(12, dtype=float).reshape(3, 4)
    plate[1] = np.nan
    fit = S.median_polish(plate)
    assert np.isnan(fit.row[1])
    assert np.isfinite(fit.row[[0, 2]]).all()


def _gradient_plate(rng, n_rows=16, n_cols=24):
    """A 384 plate with noise, a planted column and row gradient, and one
    true hit in the middle."""
    r = np.arange(1, n_rows + 1)[:, None]
    c = np.arange(1, n_cols + 1)[None, :]
    plate = 100.0 + 1.5 * c + 0.8 * r + rng.normal(0, 1.0, (n_rows, n_cols))
    plate[7, 11] += 12.0
    return plate


def test_b_score_removes_a_planted_row_and_column_gradient():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(21)
    plate = _gradient_plate(rng)
    cols = np.tile(np.arange(24), (16, 1))
    rows = np.tile(np.arange(16)[:, None], (1, 24))
    assert spearmanr(plate.ravel(), cols.ravel())[0] > 0.8

    scores, polish, scale = S.b_scores(plate)
    assert abs(spearmanr(scores.ravel(), cols.ravel())[0]) < 0.1
    assert abs(spearmanr(scores.ravel(), rows.ravel())[0]) < 0.1
    assert np.polyfit(np.arange(24), polish.column, 1)[0] == pytest.approx(
        1.5, abs=0.1)
    assert np.polyfit(np.arange(16), polish.row, 1)[0] == pytest.approx(
        0.8, abs=0.1)
    assert np.unravel_index(np.nanargmax(scores), scores.shape) == (7, 11)
    assert scores[7, 11] > 6
    assert scale == pytest.approx(1.0, abs=0.25)


def test_b_score_scales_by_the_scaled_mad_of_the_fitted_residuals():
    rng = np.random.default_rng(2)
    plate = rng.normal(0, 1, (8, 12))
    scores, polish, scale = S.b_scores(plate)
    residuals = plate - polish.fitted()
    assert scale == pytest.approx(S.mad(residuals))
    assert np.allclose(scores, residuals / scale)


def test_b_score_fitted_on_samples_scores_the_control_columns_too():
    rng = np.random.default_rng(4)
    plate = rng.normal(50, 1, (8, 12))
    plate[:, 0] = 10.0
    fit = np.ones_like(plate, dtype=bool)
    fit[:, 0] = False
    scores, polish, _scale = S.b_scores(plate, fit_mask=fit)
    assert np.isnan(polish.column[0])
    assert np.isfinite(scores[:, 0]).all()
    assert (scores[:, 0] < -10).all()


def test_call_hits_directions_and_nan():
    scores = np.array([3.5, -4.0, 1.0, np.nan, 3.0])
    assert S.call_hits(scores, 3).tolist() == [True, True, False, False, True]
    assert S.call_hits(scores, 3, "up").tolist() == [True, False, False,
                                                     False, True]
    assert S.call_hits(scores, 3, "down").tolist() == [False, True, False,
                                                       False, False]
    with pytest.raises(S.HitScoringError):
        S.call_hits(scores, 3, "sideways")


UP_HITS = {("P1", 3, 5), ("P1", 10, 14), ("P2", 6, 8)}
DOWN_HITS = {("P2", 12, 20), ("P1", 15, 3)}


def _screen(rng, *, plates=("P1", "P2"), objects=3, gradient=0.0,
            plate_shift=None):
    """A per-object table of 384-well plates: negative control in columns 1
    and 2, positive control in 23 and 24, and the planted hits."""
    rows = []
    for plate in plates:
        shift = (plate_shift or {}).get(plate, 0.0)
        for r in range(1, 17):
            for c in range(1, 25):
                value = 100.0 + gradient * c + shift + rng.normal(0, 2.0)
                kind = "neg" if c <= 2 else ("pos" if c >= 23 else "sample")
                if kind == "pos":
                    value += 40.0
                if (plate, r, c) in UP_HITS:
                    value += 30.0
                if (plate, r, c) in DOWN_HITS:
                    value -= 30.0
                for _ in range(objects):
                    rows.append({"prc": f"{plate}_r{r}_c{c}",
                                 "signal": value + rng.normal(0, 0.2),
                                 "well_type": kind,
                                 "gene": f"gene_{r}_{c}"})
    return pd.DataFrame(rows)


def _called(result, column="hit"):
    wells = result.wells
    rows = wells[wells[column]]
    return {(p, int(r), int(c)) for p, r, c in
            zip(rows["plateID"], rows["row_index"], rows["column_index"])}


def test_screen_wells_aggregates_objects_and_reads_the_roles():
    rng = np.random.default_rng(0)
    frame = _screen(rng)
    wells = S.screen_wells(frame, "signal", control_column="well_type",
                           negative_levels="neg", positive_levels=["pos"])
    assert len(wells) == 2 * 384
    assert (wells["n"] == 3).all()
    assert (wells["role"] == "negative").sum() == 2 * 32
    assert (wells["role"] == "positive").sum() == 2 * 32
    first = frame[frame["prc"] == "P1_r1_c1"]["signal"].mean()
    row = wells[(wells["plateID"] == "P1") & (wells["well"] == "A01")]
    assert row["value"].iloc[0] == pytest.approx(first)
    assert row["prc"].iloc[0] == "P1_r1_c1"


def test_control_wells_by_position_agree_with_control_labels():
    rng = np.random.default_rng(0)
    frame = _screen(rng)
    by_label = S.screen_wells(frame, "signal", control_column="well_type",
                              negative_levels="neg", positive_levels="pos")
    by_well = S.screen_wells(frame, "signal", negative_wells="c1,c2",
                             positive_wells=["c23", "c24"])
    assert by_label["role"].tolist() == by_well["role"].tolist()


def test_screen_wells_refuses_a_well_named_both_controls():
    rng = np.random.default_rng(0)
    frame = _screen(rng, plates=("P1",), objects=1)
    with pytest.raises(S.HitScoringError):
        S.screen_wells(frame, "signal", negative_wells="c1",
                       positive_wells="A01")


def test_screen_wells_min_count_drops_sparse_wells():
    rng = np.random.default_rng(0)
    frame = _screen(rng, plates=("P1",), objects=2)
    frame = frame[~((frame["prc"] == "P1_r1_c5")
                    & frame.duplicated("prc"))]
    wells = S.screen_wells(frame, "signal", negative_wells="c1",
                           min_count=2)
    assert "A05" not in set(wells["well"])
    assert len(wells) == 383


@pytest.mark.parametrize("rank_by", ["ssmd", "robust_z", "b_score"])
def test_planted_hits_are_called_and_ranked_first(rank_by):
    rng = np.random.default_rng(7)
    frame = _screen(rng)
    result = S.score_arrayed_screen(
        frame, "signal", control_column="well_type", negative_levels="neg",
        positive_levels="pos", rank_by=rank_by,
        thresholds={"ssmd": 3, "robust_z": 5, "b_score": 5})
    planted = UP_HITS | DOWN_HITS
    assert _called(result) == planted
    top = result.hits().head(len(planted))
    assert {(p, int(r), int(c)) for p, r, c in zip(
        top["plateID"],
        result.wells.set_index("prc").loc[top["prc"], "row_index"],
        result.wells.set_index("prc").loc[top["prc"], "column_index"])} \
        == planted
    assert list(result.hits()["rank"]) == list(range(1, len(planted) + 1))
    assert not result.wells.loc[result.wells["role"] != "sample",
                                "hit"].any()


def test_direction_up_calls_only_the_up_hits():
    rng = np.random.default_rng(7)
    frame = _screen(rng)
    result = S.score_arrayed_screen(
        frame, "signal", control_column="well_type", negative_levels="neg",
        positive_levels="pos",
        direction="up", thresholds={"ssmd": 3})
    assert _called(result) == UP_HITS


def test_a_gradient_fools_robust_z_and_not_the_b_score():
    """A column gradient against controls in the left-most columns makes
    every right-hand sample look like a hit to a control-based score; the
    B-score removes the column effect and keeps only the planted hits."""
    rng = np.random.default_rng(9)
    frame = _screen(rng, gradient=1.0)
    result = S.score_arrayed_screen(
        frame, "signal", control_column="well_type", negative_levels="neg",
        positive_levels="pos",
        rank_by="b_score", thresholds={"robust_z": 5, "b_score": 5})
    assert _called(result, "hit_b_score") == UP_HITS | DOWN_HITS
    assert len(_called(result, "hit_robust_z")) > 100


def _sample_median(result, plate, column="robust_z"):
    wells = result.wells
    on = wells[(wells["plateID"] == plate) & (wells["role"] == "sample")]
    return float(on[column].median())


def test_plate_scope_absorbs_a_plate_offset_and_pooled_does_not():
    """P2 reads 25 higher throughout. Against its own negative control it is
    an ordinary plate; against the pooled negative control every sample on it
    is shifted up and every sample on P1 down."""
    rng = np.random.default_rng(13)
    frame = _screen(rng, plate_shift={"P2": 25.0})
    common = dict(control_column="well_type", negative_levels="neg",
                  positive_levels="pos", thresholds={"robust_z": 5})
    plate = S.score_arrayed_screen(frame, "signal", scope="plate",
                                   rank_by="robust_z", **common)
    pooled = S.score_arrayed_screen(frame, "signal", scope="pooled",
                                    rank_by="robust_z", **common)
    assert _called(plate) == UP_HITS | DOWN_HITS
    assert abs(_sample_median(plate, "P2") - _sample_median(plate, "P1")) \
        < 0.5
    assert _sample_median(pooled, "P2") - _sample_median(pooled, "P1") > 1.0
    assert _called(pooled) != _called(plate)
    neg = frame[frame["well_type"] == "neg"].groupby("prc")["signal"].mean()
    assert pooled.wells["difference"].iloc[0] == pytest.approx(
        pooled.wells["value"].iloc[0] - neg.median())


def test_pooled_b_score_uses_one_screen_wide_scale():
    rng = np.random.default_rng(14)
    frame = _screen(rng)
    pooled = S.score_arrayed_screen(frame, "signal", scope="pooled",
                                    control_column="well_type",
                                    negative_levels="neg",
                                    positive_levels="pos")
    scales = pooled.plates["b_score_scale"].to_numpy()
    assert scales[0] == pytest.approx(scales[1])


def test_replicate_ssmd_finds_the_planted_treatment():
    """Three plates, each carrying the same library; one gene is up on all
    three, one gene is up on a single plate only. Replicate SSMD calls the
    consistent one and not the one-off."""
    rng = np.random.default_rng(17)
    frame = _screen(rng, plates=("P1", "P2", "P3"), objects=1)
    consistent = frame["gene"] == "gene_4_9"
    one_off = (frame["gene"] == "gene_9_4") & frame["prc"].str.startswith(
        "P3")
    frame.loc[consistent, "signal"] += 20.0
    frame.loc[one_off, "signal"] += 20.0
    result = S.score_arrayed_screen(
        frame, "signal", control_column="well_type", negative_levels="neg",
        positive_levels="pos",
        treatment_column="gene", replicate_estimator="umvue")
    table = result.treatments.set_index("treatment")
    assert table.loc["gene_4_9", "n_replicates"] == 3
    assert table.loc["gene_4_9", "hit"]
    assert not table.loc["gene_9_4", "hit"]
    assert table.loc["gene_4_9", "mean_difference"] == pytest.approx(
        20.0, abs=3.0)
    assert "gene_12_23" not in table.index
    d = result.wells.loc[result.wells["treatment"] == "gene_4_9",
                         "difference"].to_numpy()
    assert table.loc["gene_4_9", "ssmd"] == pytest.approx(
        S.ssmd_replicated(d, "umvue"))


def test_plate_summary_carries_the_control_chart_zprime():
    from spacr.qt.widgets.control_chart import ControlChartSpec, zprime_frame

    rng = np.random.default_rng(19)
    frame = _screen(rng)
    result = S.score_arrayed_screen(
        frame, "signal", control_column="well_type", negative_levels="neg",
        positive_levels="pos")
    wells = result.wells
    for plate in ("P1", "P2"):
        on = wells[wells["plateID"] == plate]
        pos = on.loc[on["role"] == "positive", "value"]
        neg = on.loc[on["role"] == "negative", "value"]
        by_hand = 1 - 3 * (pos.std() + neg.std()) / abs(pos.mean() - neg.mean())
        row = result.plates.set_index("plateID").loc[plate]
        assert row["zprime"] == pytest.approx(by_hand)
    spec = ControlChartSpec(value="value", plate="plateID",
                            control_column="role",
                            control_levels=("positive", "negative"),
                            positive_levels=("positive",),
                            negative_levels=("negative",))
    direct = zprime_frame(wells, spec).set_index("plate")["zprime"]
    assert result.plates.set_index("plateID")["zprime"].to_dict() == \
        pytest.approx(direct.to_dict())


def test_scoring_without_a_negative_control_is_refused():
    rng = np.random.default_rng(0)
    frame = _screen(rng, plates=("P1",), objects=1)
    with pytest.raises(S.HitScoringError):
        S.score_arrayed_screen(frame, "signal")
    with pytest.raises(S.HitScoringError):
        S.score_arrayed_screen(frame, "signal", control_column="well_type",
                               negative_levels="neg", scope="global")


def test_report_names_the_settings_and_counts():
    rng = np.random.default_rng(7)
    frame = _screen(rng)
    result = S.score_arrayed_screen(
        frame, "signal", control_column="well_type", negative_levels="neg",
        positive_levels="pos")
    text = result.report()
    assert "SSMD" in text and "Z'" in text and "plate scope" in text


def test_heatmap_outlines_exactly_the_called_wells():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.patches import Rectangle

    rng = np.random.default_rng(7)
    frame = _screen(rng)
    result = S.score_arrayed_screen(
        frame, "signal", control_column="well_type", negative_levels="neg",
        positive_levels="pos",
        thresholds={"ssmd": 3})
    figure, panel = S.hit_heatmap(result, "ssmd", target="print")
    assert panel.drawn
    outlined = sum(
        1 for ax in figure.axes for patch in ax.patches
        if isinstance(patch, Rectangle) and not patch.get_fill())
    assert outlined == int(result.wells["hit_ssmd"].sum())
    low, high = figure.axes[0].images[0].get_clim()
    assert low == pytest.approx(-high)
    import matplotlib.pyplot as plt
    plt.close(figure)


def test_write_hit_report_writes_the_tables_and_one_figure_per_method(tmp_path):
    rng = np.random.default_rng(7)
    frame = _screen(rng, objects=1)
    result = S.score_arrayed_screen(
        frame, "signal", control_column="well_type", negative_levels="neg",
        positive_levels="pos", treatment_column="gene")
    written = S.write_hit_report(result, tmp_path / "hits", target="print")
    for key in ("hit_table", "hit_scores_wells", "hit_plates",
                "hit_treatments", "hit_heatmap_ssmd",
                "hit_heatmap_robust_z", "hit_heatmap_b_score"):
        assert key in written
        assert (tmp_path / "hits").joinpath(
            written[key].split("/")[-1]).exists()
    table = pd.read_csv(written["hit_table"])
    assert len(table) == int(result.wells["hit"].sum())
    assert table["rank"].tolist() == sorted(table["rank"].tolist())
