"""Regression QC: every panel is checked against a fixture whose answer is known.

The rule this file is written to: a QC figure that cannot fail is worth nothing.
So none of these tests assert that a PDF exists and stop there. Each one plants
a defect the corresponding panel is supposed to reveal -- a high-leverage
outlier, an aliased predictor pair, a miscalibrated logistic fit, a plate whose
outer rows are systematically dim -- and asserts that the panel's own numbers
name it, plus a matching control fixture without the defect where the same
statistic stays quiet. Where a panel labels a well, the label is asserted to be
the *right* well, because a diagnostic that sends you to the wrong plate is
worse than none.

The other half of the file is the degradation contract: a sklearn ``Lasso`` has
no p-values and no covariance matrix, a ``MixedLM`` has no hat matrix, a
Gaussian fit has no calibration curve. Those panels must be recorded as skipped
*with a reason*, never quietly dropped and never faked.
"""
from __future__ import annotations

import functools
import os

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from sklearn.linear_model import Lasso

from spacr import schema
from spacr import regression_qc as rq


# ---------------------------------------------------------------------------
# Fixture builders. Deterministic seeds: a failure is reproducible from the
# git hash alone.
# ---------------------------------------------------------------------------


def _axes():
    """A bare Axes on a pyplot-free Figure, so nothing leaks into pyplot.

    Nothing here calls ``fig.clf()`` afterwards: these figures are never
    registered with pyplot, so dropping the reference is the whole clean-up --
    and clearing the figure would wipe the titles and annotations that the
    assertions below read back off the axes.
    """
    fig = Figure(figsize=(5, 4))
    return fig, fig.subplots()


def _stream(seed, key):
    """An independent RNG stream.

    Two ``default_rng(seed)`` calls in one fixture return the SAME numbers, so
    a "noise" term drawn that way is identical to the first predictor and the
    fit comes out exact -- every residual zero, every residual panel
    meaningless. Keying the stream keeps design and noise independent.
    """
    return np.random.default_rng([seed, key])


def _plate_metadata(n_rows=8, n_cols=12, plates=("plate1",), seed=0):
    """Per-well metadata in spaCR's canonical column names."""
    rng = _stream(seed, 3)
    records = []
    for plate in plates:
        for r in range(1, n_rows + 1):
            for c in range(1, n_cols + 1):
                records.append({
                    schema.PLATE_KEY: plate,
                    schema.ROW_KEY: f"r{r}",
                    schema.COLUMN_KEY: f"c{c}",
                    schema.PRC_KEY: f"{plate}_r{r}_c{c}",
                })
    meta = pd.DataFrame.from_records(records)
    meta["cell_count"] = rng.integers(30, 800, len(meta)).astype(float)
    return meta


def _design(n, seed=0, n_predictors=2):
    """Intercept + orthogonal-ish continuous predictors."""
    rng = _stream(seed, 1)
    data = {"Intercept": np.ones(n)}
    for i in range(n_predictors):
        data[f"x{i + 1}"] = rng.normal(size=n)
    return pd.DataFrame(data)


def _wells_for(n, seed):
    """Metadata for exactly ``n`` wells, on as many 12-column rows as it takes."""
    plate = _plate_metadata(n_rows=int(np.ceil(n / 12)), seed=seed)
    assert len(plate) >= n, "plate layout too small for the requested fit"
    return plate.iloc[:n].reset_index(drop=True)


def _ols_case(n=96, seed=0, noise=0.4):
    """A clean, well-specified OLS fit with plate metadata attached."""
    rng = _stream(seed, 2)
    X = _design(n, seed=seed)
    y = 1.0 + 2.0 * X["x1"] - 0.5 * X["x2"] + rng.normal(size=n) * noise
    model = sm.OLS(y, X).fit()
    return model, X, y, _wells_for(n, seed)


def _context(n=96, seed=0, **kwargs):
    model, X, y, meta = _ols_case(n=n, seed=seed)
    return rq.build_context(model, X, y, metadata=meta, regression_type="ols",
                            **kwargs)


def _logit_case(n=400, seed=3, effect=1.5, bias=0.0):
    """A binary logistic fit; ``bias`` shifts the world the model is TRAINED on.

    With ``bias != 0`` the model is fitted to labels drawn from a different
    intercept than the labels it is later scored against, which is exactly a
    miscalibrated model: the ranking survives, the probabilities do not.
    """
    rng = _stream(seed, 2)
    X = _design(n, seed=seed)
    eta = -0.2 + effect * X["x1"]
    p_true = 1.0 / (1.0 + np.exp(-eta))
    p_train = 1.0 / (1.0 + np.exp(-(eta + bias)))
    y_true = (rng.uniform(size=n) < p_true).astype(float)
    y_train = (rng.uniform(size=n) < p_train).astype(float)
    model = sm.GLM(y_train, X, family=sm.families.Binomial()).fit()
    return model, X, y_true


# ---------------------------------------------------------------------------
# Model fit panels
# ---------------------------------------------------------------------------


def test_residual_panel_draws_exactly_one_point_per_well():
    """The scatter is the wells; if it is not, the panel is showing something else."""
    ctx = _context(n=97, seed=1)
    fig, ax = _axes()
    stats = rq.draw_panel("residuals_vs_fitted", ctx, ax)

    scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(scatters) == 1
    assert scatters[0].get_offsets().shape[0] == 97
    assert stats["n_points"] == 97
    # ...and the panel says so on its own face, so the figure is readable
    # alone. The n used to be bolted onto a two-line sentence title; the house
    # style has no sentence titles, so it moved into the panel's own note.
    assert "n = 97 wells" in " ".join(t.get_text() for t in ax.texts)
    assert ax.get_xlabel() and ax.get_ylabel()

    offsets = np.asarray(scatters[0].get_offsets())
    np.testing.assert_allclose(np.sort(offsets[:, 0]), np.sort(ctx.fitted))
    np.testing.assert_allclose(np.sort(offsets[:, 1]), np.sort(ctx.resid))


def test_observed_vs_predicted_reports_the_r_squared_it_draws():
    """R² and RMSE are computed here, so they are checked against the definition."""
    ctx = _context(n=120, seed=2)
    fig, ax = _axes()
    stats = rq.draw_panel("observed_vs_predicted", ctx, ax)

    rss = float(np.sum((ctx.y - ctx.fitted) ** 2))
    tss = float(np.sum((ctx.y - ctx.y.mean()) ** 2))
    assert stats["r2"] == pytest.approx(1 - rss / tss, rel=1e-12)
    assert stats["rmse"] == pytest.approx(np.sqrt(rss / ctx.n), rel=1e-12)
    # statsmodels agrees, which is the real cross-check.
    assert stats["r2"] == pytest.approx(ctx.model.rsquared, rel=1e-9)

    annotations = " ".join(t.get_text() for t in ax.texts)
    assert f"R² (response scale) = {stats['r2']:.3f}" in annotations
    assert "RMSE" in annotations


def _scale_location(y, X):
    fig, ax = _axes()
    return rq.draw_panel("scale_location",
                         rq.build_context(sm.OLS(y, X).fit(), X, y), ax)


def test_scale_location_separates_heteroscedastic_from_clean():
    """A fan-shaped residual cloud must move the statistic the panel prints."""
    rng = _stream(7, 2)
    n = 300
    X = _design(n, seed=7, n_predictors=1)
    signal = 3.0 * X["x1"]
    # The textbook megaphone: noise SD rises monotonically with the fit.
    megaphone = signal + rng.normal(size=n) * (0.2 + 0.3 * (signal - signal.min()))
    clean = signal + rng.normal(size=n) * 0.5

    bad = _scale_location(megaphone, X)
    good = _scale_location(clean, X)

    assert bad["spearman_rho"] > 0.3
    assert bad["spearman_p"] < 0.01
    assert bad["levene_p"] < 0.01
    assert bad["quartile_sd_ratio"] > 2
    assert bad["verdict"] == "variance grows with the fit"

    assert abs(good["spearman_rho"]) < 0.2
    assert good["levene_p"] > 0.05
    assert good["verdict"] == "no detectable trend in spread"


def test_scale_location_sees_a_symmetric_funnel_that_spearman_cannot():
    """Spread large at both ends and small in the middle: rho is blind to it.

    This is the case a rank correlation alone reports as "no trend in spread"
    -- a confident wrong answer -- which is why the panel also runs
    Brown-Forsythe across quartiles of the fitted value.
    """
    rng = _stream(8, 2)
    n = 400
    X = _design(n, seed=8, n_predictors=1)
    signal = 3.0 * X["x1"]
    funnel = signal + rng.normal(size=n) * (0.1 + 0.9 * np.abs(signal))
    stats = _scale_location(funnel, X)

    assert abs(stats["spearman_rho"]) < 0.25, "the fixture is not symmetric"
    assert stats["levene_p"] < 0.01
    assert stats["quartile_sd_ratio"] > 2
    assert stats["verdict"] == "spread differs across the fit, but not monotonically"


def test_qq_reference_line_passes_through_both_quartiles():
    """The reference line is the R-style quartile line, not a least-squares fit."""
    from scipy import stats as sps

    ctx = _context(n=200, seed=4)
    fig, ax = _axes()
    stats = rq.draw_panel("qq_residuals", ctx, ax)

    sample = np.sort(ctx.std_resid)
    q1_t, q3_t = sps.norm.ppf([0.25, 0.75])
    q1_s, q3_s = np.quantile(sample, [0.25, 0.75])
    expected_slope = (q3_s - q1_s) / (q3_t - q1_t)
    assert stats["slope"] == pytest.approx(expected_slope, rel=1e-12)
    assert stats["intercept"] == pytest.approx(q1_s - expected_slope * q1_t,
                                               rel=1e-9, abs=1e-12)
    # Gaussian residuals: the line is near the 45 degree line of a standardised
    # residual and the quantile correlation is very high.
    assert 0.85 < stats["slope"] < 1.15
    assert stats["quantile_correlation"] > 0.99
    assert stats["n_points"] == 200


def test_qq_notices_heavy_tails():
    """The control for the test above: t(3) residuals must NOT look normal."""
    rng = _stream(11, 2)
    n = 400
    X = _design(n, seed=11, n_predictors=1)
    y = 1.0 + X["x1"] + rng.standard_t(3, size=n)
    ctx = rq.build_context(sm.OLS(y, X).fit(), X, y)
    fig, ax = _axes()
    stats = rq.draw_panel("qq_residuals", ctx, ax)
    assert stats["quantile_correlation"] < 0.99


def test_residual_distribution_reports_the_planted_skew():
    """A deliberately skewed residual is reported as skewed, not smoothed over."""
    rng = _stream(12, 2)
    n = 300
    X = _design(n, seed=12, n_predictors=1)
    y = 1.0 + X["x1"] + rng.exponential(1.0, size=n)
    ctx = rq.build_context(sm.OLS(y, X).fit(), X, y)
    fig, ax = _axes()
    stats = rq.draw_panel("residual_distribution", ctx, ax)
    assert stats["skew"] > 1.0
    assert stats["normality_p"] < 1e-6
    assert stats["n_points"] == n


# ---------------------------------------------------------------------------
# Influence and leverage
# ---------------------------------------------------------------------------


def test_leverage_from_design_matches_statsmodels():
    """The fallback used for MixedLM and sklearn is the same hat matrix."""
    model, X, y, _ = _ols_case(n=80, seed=5)
    ours = rq.leverage_from_design(X.to_numpy(dtype=float))
    theirs = model.get_influence().hat_matrix_diag
    np.testing.assert_allclose(ours, theirs, rtol=1e-10, atol=1e-12)
    # trace(H) == p for a full-rank design.
    assert ours.sum() == pytest.approx(X.shape[1], rel=1e-9)


def _outlier_case(n=96, seed=6, shift=9.0):
    """A clean fit with ONE well moved far off the line at high leverage."""
    rng = _stream(seed, 2)
    X = _design(n, seed=seed, n_predictors=2)
    y = 1.0 + 2.0 * X["x1"] - 0.5 * X["x2"] + rng.normal(size=n) * 0.3
    meta = _wells_for(n, seed)
    planted = 40
    X.loc[planted, "x1"] = 6.0          # far out in x -> high leverage
    y.iloc[planted] += shift            # ...and off the line -> big residual
    model = sm.OLS(y, X).fit()
    ctx = rq.build_context(model, X, y, metadata=meta, regression_type="ols")
    return ctx, planted, meta.loc[planted, schema.PRC_KEY]


def test_cooks_distance_flags_the_planted_outlier_by_name():
    """The panel must both flag the well and name the right one."""
    ctx, planted, prc = _outlier_case()
    fig, ax = _axes()
    stats = rq.draw_panel("cooks_distance", ctx, ax)

    assert stats["max_index"] == planted
    assert stats["max_label"] == prc
    assert stats["max_cooks"] > stats["threshold"]
    assert prc in stats["flagged"]
    # The label is drawn on the figure, not just returned.
    assert prc in [t.get_text() for t in ax.texts]

    # Control: the same generator with no planted well leaves nothing like it.
    clean = _context(n=96, seed=6)
    fig, ax = _axes()
    clean_stats = rq.draw_panel("cooks_distance", clean, ax)
    assert clean_stats["max_cooks"] < 0.2 * stats["max_cooks"]


def test_cooks_distance_matches_its_textbook_definition():
    """D_i computed from studentised residuals equals the e_i^2 form for OLS."""
    model, X, y, _ = _ols_case(n=60, seed=8)
    ctx = rq.build_context(model, X, y)
    ours = rq.cooks_distance(ctx.std_resid, ctx.leverage, ctx.p)
    theirs = model.get_influence().cooks_distance[0]
    np.testing.assert_allclose(ours, theirs, rtol=1e-8, atol=1e-12)


def test_influence_panel_labels_the_outlier_and_counts_high_leverage():
    ctx, planted, prc = _outlier_case()
    fig, ax = _axes()
    stats = rq.draw_panel("influence", ctx, ax)

    assert prc in stats["labelled"]
    assert prc in [t.get_text() for t in ax.texts]
    assert stats["max_leverage"] == pytest.approx(ctx.leverage.max())
    assert stats["max_leverage"] > stats["leverage_guides"][0]
    assert stats["n_points"] == ctx.n
    # The 2p/n and 3p/n guides are drawn.
    guides = [line.get_xdata()[0] for line in ax.lines
              if line.get_linestyle() == "--" and len(set(line.get_xdata())) == 1]
    for expected in stats["leverage_guides"]:
        assert any(abs(g - expected) < 1e-9 for g in guides)


def test_dffits_flags_the_planted_outlier_above_its_threshold():
    ctx, planted, prc = _outlier_case()
    fig, ax = _axes()
    stats = rq.draw_panel("dffits", ctx, ax)
    assert prc in stats["flagged"]
    assert stats["threshold"] == pytest.approx(2 * np.sqrt(ctx.p / ctx.n))
    assert stats["max_abs_dffits"] > stats["threshold"]

    values, _ = rq.dffits(ctx.std_resid, ctx.leverage, ctx.n, ctx.p)
    theirs = ctx.model.get_influence().dffits[0]
    np.testing.assert_allclose(values, theirs, rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# Design and collinearity
# ---------------------------------------------------------------------------


def test_vif_is_huge_for_a_deliberately_collinear_pair():
    rng = _stream(20, 2)
    n = 300
    a = rng.normal(size=n)
    frame = pd.DataFrame({
        "Intercept": np.ones(n),
        "a": a,
        "b": a + 0.01 * rng.normal(size=n),   # ~99.995% shared variance
        "c": rng.normal(size=n),
    })
    vif = rq.variance_inflation_factors(frame)
    assert vif["a"] > 100 and vif["b"] > 100
    assert vif["c"] < 2
    # The intercept is constant: it has no variance to inflate, and saying "1"
    # would be a made-up number.
    assert np.isnan(vif["Intercept"])


def test_vif_matches_statsmodels_on_a_well_conditioned_design():
    """The correlation-matrix identity is the same number, computed cheaper."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    rng = _stream(21, 2)
    n = 200
    base = rng.normal(size=n)
    frame = pd.DataFrame({
        "Intercept": np.ones(n),
        "a": base,
        "b": 0.7 * base + rng.normal(size=n),
        "c": rng.normal(size=n),
    })
    ours = rq.variance_inflation_factors(frame)
    matrix = frame.to_numpy(dtype=float)
    for i, name in enumerate(frame.columns):
        if name == "Intercept":
            continue
        assert ours[name] == pytest.approx(variance_inflation_factor(matrix, i),
                                           rel=1e-8)


def test_vif_is_infinite_for_exactly_aliased_columns_only():
    n = 120
    rng = _stream(22, 2)
    a = rng.normal(size=n)
    frame = pd.DataFrame({"a": a, "b": 2.0 * a, "c": rng.normal(size=n)})
    vif = rq.variance_inflation_factors(frame)
    assert np.isinf(vif["a"]) and np.isinf(vif["b"])
    assert np.isfinite(vif["c"]) and vif["c"] < 2


def test_vif_panel_counts_and_draws_the_guides():
    rng = _stream(23, 2)
    n = 200
    a = rng.normal(size=n)
    X = pd.DataFrame({"Intercept": np.ones(n), "a": a,
                      "b": a + 0.02 * rng.normal(size=n),
                      "c": rng.normal(size=n)})
    y = 1 + a + rng.normal(size=n) * 0.5
    ctx = rq.build_context(sm.OLS(y, X).fit(), X, y)
    fig, ax = _axes()
    stats = rq.draw_panel("vif", ctx, ax)

    assert stats["n_above_10"] == 2          # a and b, not c
    assert stats["n_constant"] == 1          # the intercept
    assert stats["n_aliased"] == 0
    guides = sorted({round(float(line.get_xdata()[0]), 6) for line in ax.lines
                     if len(set(line.get_xdata())) == 1})
    assert 5.0 in guides and 10.0 in guides


def test_condition_number_separates_orthogonal_from_collinear_from_singular():
    n = 200
    rng = _stream(24, 2)
    a = rng.normal(size=n)

    orthogonal = np.column_stack([np.ones(n), a, rng.normal(size=n)])
    scaled, unscaled, singular = rq.condition_number(orthogonal)
    assert scaled < 10
    assert rq.condition_verdict(scaled) == "no collinearity problem"
    assert singular.size == 3

    collinear = np.column_stack([np.ones(n), a, a + 1e-3 * rng.normal(size=n)])
    scaled_bad, _, _ = rq.condition_number(collinear)
    assert scaled_bad > 30
    assert "collinear" in rq.condition_verdict(scaled_bad)

    aliased = np.column_stack([np.ones(n), a, 2.0 * a])
    scaled_singular, _, _ = rq.condition_number(aliased)
    assert scaled_singular > 1e8
    # And the units of a column cannot change the scaled number, which is the
    # entire reason it is the one on the figure.
    rescaled = collinear * np.array([1.0, 1000.0, 1.0])
    assert rq.condition_number(rescaled)[0] == pytest.approx(scaled_bad, rel=1e-6)
    assert rq.condition_number(rescaled)[1] != pytest.approx(unscaled, rel=1e-3)


def test_condition_number_panel_states_a_verdict():
    rng = _stream(25, 2)
    n = 150
    a = rng.normal(size=n)
    X = pd.DataFrame({"Intercept": np.ones(n), "a": a,
                      "b": a + 1e-3 * rng.normal(size=n)})
    y = a + rng.normal(size=n) * 0.4
    ctx = rq.build_context(sm.OLS(y, X).fit(), X, y)
    fig, ax = _axes()
    stats = rq.draw_panel("condition_number", ctx, ax)
    text = " ".join(t.get_text() for t in ax.texts)
    assert stats["condition_number"] > 30
    assert "collinear" in stats["verdict"]
    assert "scaled condition number" in text
    assert stats["verdict"][:20] in text


def test_predictor_correlation_names_the_correlated_pair():
    rng = _stream(26, 2)
    n = 200
    a = rng.normal(size=n)
    X = pd.DataFrame({"Intercept": np.ones(n), "a": a, "b": rng.normal(size=n),
                      "twin": a + 0.05 * rng.normal(size=n)})
    y = a + rng.normal(size=n) * 0.4
    ctx = rq.build_context(sm.OLS(y, X).fit(), X, y)
    fig, ax = _axes()
    stats = rq.draw_panel("predictor_correlation", ctx, ax)
    assert stats["max_abs_offdiagonal"] > 0.95
    assert set(stats["max_pair"]) == {"a", "twin"}
    assert stats["n_predictors"] == 3        # the intercept is constant


# ---------------------------------------------------------------------------
# Response and coefficients
# ---------------------------------------------------------------------------


def test_calibration_curve_is_off_the_diagonal_for_a_miscalibrated_model():
    """The planted defect: a model trained on a shifted world."""
    good_model, X, y = _logit_case(bias=0.0)
    good = rq.build_context(good_model, X, y, regression_type="logit")
    fig, ax = _axes()
    good_stats = rq.draw_panel("calibration", good, ax)

    bad_model, Xb, yb = _logit_case(bias=2.5)
    bad = rq.build_context(bad_model, Xb, yb, regression_type="logit")
    fig, ax = _axes()
    bad_stats = rq.draw_panel("calibration", bad, ax)

    assert good_stats["ece"] < 0.06
    assert bad_stats["ece"] > 0.15
    assert bad_stats["max_gap"] > 0.15
    assert bad_stats["brier"] > good_stats["brier"]
    # The miscalibrated curve sits ABOVE nothing and BELOW the diagonal:
    # every bin over-predicts.
    over = [p > o for p, o in zip(bad_stats["pred_mean"], bad_stats["obs_mean"])]
    assert sum(over) >= len(over) - 1


def test_calibration_curve_maths_on_a_known_answer():
    """A model that predicts 0.5 for a population that is 90% positive."""
    y = np.concatenate([np.ones(90), np.zeros(10)])
    p = np.full(100, 0.5)
    curve = rq.calibration_curve(y, p, n_bins=4)
    assert curve["obs_mean"][0] == pytest.approx(0.9)
    assert curve["pred_mean"][0] == pytest.approx(0.5)
    assert curve["ece"] == pytest.approx(0.4)
    assert curve["brier"] == pytest.approx(0.25)


def test_calibration_weights_let_a_big_well_outvote_a_small_one():
    """Cell counts are the whole reason spaCR's logit path is weighted."""
    y = np.array([1.0, 0.0, 0.0])
    p = np.array([0.5, 0.5, 0.5])
    weights = np.array([1000.0, 1.0, 1.0])
    unweighted = rq.calibration_curve(y, p, n_bins=2)
    weighted = rq.calibration_curve(y, p, n_bins=2, weights=weights)
    assert unweighted["obs_mean"][0] == pytest.approx(1 / 3)
    assert weighted["obs_mean"][0] == pytest.approx(1000 / 1002)


def test_calibration_is_skipped_for_a_gaussian_fit_with_a_reason():
    ctx = _context()
    fig, ax = _axes()
    with pytest.raises(rq.PanelUnavailable) as excinfo:
        rq.draw_panel("calibration", ctx, ax)
    assert "probability response" in str(excinfo.value)
    assert "Gaussian" in str(excinfo.value)


def test_roc_and_pr_separate_signal_from_noise():
    strong_model, X, y = _logit_case(n=600, seed=31, effect=3.0)
    strong = rq.build_context(strong_model, X, y, regression_type="logit")
    null_model, Xn, yn = _logit_case(n=600, seed=32, effect=0.0)
    null = rq.build_context(null_model, Xn, yn, regression_type="logit")

    out = {}
    for name, ctx in (("strong", strong), ("null", null)):
        for panel in ("roc", "precision_recall"):
            fig, ax = _axes()
            out[(name, panel)] = rq.draw_panel(panel, ctx, ax)

    assert out[("strong", "roc")]["auc"] > 0.9
    assert out[("null", "roc")]["auc"] < 0.7
    assert (out[("strong", "roc")]["auc"]
            > out[("null", "roc")]["auc"] + 0.2)
    assert out[("strong", "roc")]["n_positive"] + out[("strong", "roc")]["n_negative"] == 600
    assert (out[("strong", "precision_recall")]["average_precision"]
            > out[("null", "precision_recall")]["average_precision"])
    assert out[("null", "precision_recall")]["prevalence"] == pytest.approx(
        float(np.mean(yn == 1)))


def test_roc_is_skipped_for_a_continuous_fraction_response():
    """spaCR fits logit on per-well FRACTIONS; ROC has no labels to rank."""
    rng = _stream(33, 2)
    n = 200
    X = _design(n, seed=33)
    counts = rng.integers(30, 400, n).astype(float)
    p = 1 / (1 + np.exp(-(0.3 + X["x1"])))
    fraction = rng.binomial(counts.astype(int), p) / counts
    model = sm.GLM(fraction, X, family=sm.families.Binomial(),
                   var_weights=counts).fit()
    ctx = rq.build_context(model, X, fraction, weights=counts,
                           regression_type="logit")
    assert ctx.is_binomial and not ctx.is_binary_response

    fig, ax = _axes()
    with pytest.raises(rq.PanelUnavailable) as excinfo:
        rq.draw_panel("roc", ctx, ax)
    assert "fraction" in str(excinfo.value)

    # ...but calibration IS defined for a fraction, and it uses the weights.
    fig, ax = _axes()
    stats = rq.draw_panel("calibration", ctx, ax)
    assert stats["weighted"] is True
    assert stats["ece"] < 0.1


def test_p_value_histogram_diagnoses_each_broken_shape():
    rng = _stream(40, 2)
    uniform = rng.uniform(size=2000)
    assert rq.diagnose_p_value_histogram(uniform)["verdict"] == "uniform"

    with_hits = np.concatenate([rng.uniform(0, 0.01, 300), rng.uniform(size=1700)])
    assert rq.diagnose_p_value_histogram(with_hits)["verdict"] == "uniform-with-spike"

    conservative = np.concatenate([rng.uniform(0.9, 1.0, 900),
                                   rng.uniform(size=1100)])
    diag = rq.diagnose_p_value_histogram(conservative)
    assert diag["verdict"] == "excess-large"
    assert "conservative" in diag["message"]

    u_shaped = np.concatenate([rng.uniform(0, 0.05, 500),
                               rng.uniform(0.95, 1.0, 500),
                               rng.uniform(size=1000)])
    u_diag = rq.diagnose_p_value_histogram(u_shaped)
    assert u_diag["verdict"] == "u-shaped"
    assert "mis-specified" in u_diag["message"]

    assert rq.diagnose_p_value_histogram([0.1, 0.2])["verdict"] == "too-few"


def test_p_value_panel_prints_the_diagnosis_on_the_figure():
    """A verdict nobody can see on the figure is a verdict nobody acts on."""
    rng = _stream(41, 2)
    coef_df = pd.DataFrame({
        "feature": [f"grna[g{i}]" for i in range(1200)],
        "coefficient": rng.normal(size=1200),
        "p_value": np.concatenate([rng.uniform(0.9, 1.0, 600),
                                   rng.uniform(size=600)]),
    })
    model, X, y, meta = _ols_case()
    ctx = rq.build_context(model, X, y, metadata=meta, coef_df=coef_df,
                           regression_type="ols")
    fig, ax = _axes()
    stats = rq.draw_panel("p_value_histogram", ctx, ax)
    drawn = " ".join(t.get_text() for t in ax.texts)

    assert stats["verdict"] == "excess-large"
    assert stats["source"] == "coefficient table"
    assert stats["n"] == 1200
    assert "conservative" in drawn
    # The n used to be a second line of the title. The house style has no
    # sentence titles, so it moved to the annotation -- still on the panel's
    # own face, which is the thing this test is actually protecting.
    assert "n = 1,200 coefficients" in drawn
    assert ax.get_title() == "p-value distribution"


def test_coefficient_forest_sorts_by_effect_size_and_carries_intervals():
    rng = _stream(42, 2)
    n = 300
    X = _design(n, seed=42, n_predictors=4)
    y = (1.0 + 5.0 * X["x1"] + 0.2 * X["x2"] - 3.0 * X["x3"]
         + 0.01 * X["x4"] + rng.normal(size=n) * 0.5)
    model = sm.OLS(y, X).fit()
    ctx = rq.build_context(model, X, y, regression_type="ols")
    fig, ax = _axes()
    stats = rq.draw_panel("coefficient_forest", ctx, ax, )
    labels = [t.get_text() for t in ax.get_yticklabels()]

    assert stats["has_intervals"] is True
    assert stats["limitation"] is None
    assert stats["largest_term"] == "x1"
    assert stats["largest_coefficient"] == pytest.approx(model.params["x1"])
    # Drawn bottom-to-top by increasing |effect|, so the biggest is last.
    assert labels[-1] == "x1"
    assert labels[0] == "x4"


def test_overdispersion_statistic_separates_poisson_from_negative_binomial():
    rng = _stream(43, 2)
    n = 500
    mu = np.full(n, 6.0)
    poisson = rng.poisson(6.0, size=n).astype(float)
    negbin = rng.negative_binomial(2, 2 / (2 + 6.0), size=n).astype(float)

    clean = rq.overdispersion_statistic(poisson, mu, n - 2)
    over = rq.overdispersion_statistic(negbin, mu, n - 2)
    assert 0.8 < clean["dispersion"] < 1.25
    assert "consistent" in clean["verdict"]
    assert over["dispersion"] > 2.0
    assert "negative binomial" in over["verdict"]
    assert over["pearson_chi2"] == pytest.approx(
        float(np.sum((negbin - mu) ** 2 / mu)))


def test_count_panel_reports_the_dispersion_of_an_overdispersed_poisson_fit():
    rng = _stream(44, 2)
    n = 300
    X = _design(n, seed=44, n_predictors=1)
    mu = np.exp(1.5 + 0.4 * X["x1"])
    counts = rng.negative_binomial(1.5, 1.5 / (1.5 + mu)).astype(float)
    model = sm.GLM(counts, X, family=sm.families.Poisson()).fit()
    ctx = rq.build_context(model, X, counts, regression_type="poisson")
    fig, ax = _axes()
    stats = rq.draw_panel("count_fit", ctx, ax)
    note = " ".join(t.get_text() for t in ax.texts)

    assert stats["dispersion"] > 1.5
    assert stats["df_resid"] == pytest.approx(model.df_resid)
    assert "dispersed" in stats["verdict"]
    assert "Pearson dispersion" in note
    assert stats["n_outside_2sd"] > 0

    # And a genuine Poisson fit is not accused of over-dispersion.
    good_counts = rng.poisson(mu).astype(float)
    good_model = sm.GLM(good_counts, X, family=sm.families.Poisson()).fit()
    good_ctx = rq.build_context(good_model, X, good_counts,
                                regression_type="poisson")
    fig, ax = _axes()
    good_stats = rq.draw_panel("count_fit", good_ctx, ax)
    assert good_stats["dispersion"] < 1.4


# ---------------------------------------------------------------------------
# Screen-level structure
# ---------------------------------------------------------------------------


def _edge_effect_case(delta, seed=50, n_rows=8, n_cols=12):
    """A plate whose outer ROWS are shifted by ``delta``, fitted without a row term."""
    rng = _stream(seed, 2)
    meta = _plate_metadata(n_rows=n_rows, n_cols=n_cols, seed=seed)
    n = len(meta)
    X = _design(n, seed=seed, n_predictors=1)
    row_index = meta[schema.ROW_KEY].str.extract(r"(\d+)")[0].astype(int)
    edge = (row_index == 1) | (row_index == n_rows)
    y = 1.0 + 2.0 * X["x1"] + rng.normal(size=n) * 0.3 + edge * delta
    model = sm.OLS(y, X).fit()
    return rq.build_context(model, X, y, metadata=meta, regression_type="ols")


def test_row_panel_sees_a_planted_edge_artefact_and_stays_quiet_without_one():
    planted = _edge_effect_case(delta=1.5)
    fig, ax = _axes()
    bad = rq.draw_panel("row_effects", planted, ax)

    control = _edge_effect_case(delta=0.0)
    fig, ax = _axes()
    good = rq.draw_panel("row_effects", control, ax)

    assert bad["n_groups"] == 8
    assert bad["edge_minus_interior_median"] > 1.0
    assert bad["kruskal_p"] < 1e-6
    assert bad["worst_group"] in ("r1", "r8")
    assert abs(good["edge_minus_interior_median"]) < 0.2
    assert good["kruskal_p"] > 0.01


def test_column_panel_sees_a_planted_column_gradient():
    rng = _stream(51, 2)
    meta = _plate_metadata(seed=51)
    n = len(meta)
    col_index = meta[schema.COLUMN_KEY].str.extract(r"(\d+)")[0].astype(int)
    X = _design(n, seed=51, n_predictors=1)
    y = 1.0 + X["x1"] + 0.25 * col_index + rng.normal(size=n) * 0.2
    ctx = rq.build_context(sm.OLS(y, X).fit(), X, y, metadata=meta)
    fig, ax = _axes()
    stats = rq.draw_panel("column_effects", ctx, ax)
    labels = [t.get_text() for t in ax.get_xticklabels()]

    assert stats["n_groups"] == 12
    assert stats["kruskal_p"] < 1e-10
    assert stats["worst_group"] in ("c1", "c12")
    # Natural order, not lexicographic: c2 comes before c10.
    assert labels == [f"c{i}" for i in range(1, 13)]


def test_plate_panel_sees_a_batch_effect_between_plates():
    rng = _stream(52, 2)
    meta = _plate_metadata(n_rows=4, n_cols=6, plates=("plate1", "plate2"),
                           seed=52)
    n = len(meta)
    X = _design(n, seed=52, n_predictors=1)
    offset = np.where(meta[schema.PLATE_KEY] == "plate2", 1.2, 0.0)
    y = 1.0 + X["x1"] + offset + rng.normal(size=n) * 0.25
    ctx = rq.build_context(sm.OLS(y, X).fit(), X, y, metadata=meta)
    fig, ax = _axes()
    stats = rq.draw_panel("plate_effects", ctx, ax)
    assert stats["n_groups"] == 2
    assert stats["kruskal_p"] < 1e-8
    assert abs(stats["worst_median"]) > 0.4


def test_positional_panels_state_why_they_cannot_run():
    ctx = _context()                       # metadata has a single plate
    for panel, expected in (("plate_effects", "every well has plateID"),):
        fig, ax = _axes()
        with pytest.raises(rq.PanelUnavailable) as excinfo:
            rq.draw_panel(panel, ctx, ax)
        assert expected in str(excinfo.value)

    model, X, y, _ = _ols_case()
    bare = rq.build_context(model, X, y)    # no metadata at all
    fig, ax = _axes()
    with pytest.raises(rq.PanelUnavailable) as excinfo:
        rq.draw_panel("row_effects", bare, ax)
    assert "no per-well metadata" in str(excinfo.value)


def test_cell_count_panel_sees_low_n_wells_driving_the_tails():
    """Wells with few cells are noisier; the panel must show that they are."""
    rng = _stream(53, 2)
    n = 400
    meta = pd.DataFrame({
        schema.PRC_KEY: [f"plate1_r{i // 24 + 1}_c{i % 24 + 1}" for i in range(n)],
        "cell_count": np.exp(rng.uniform(np.log(10), np.log(2000), n)),
    })
    X = _design(n, seed=53, n_predictors=1)
    # Binomial-style noise: the SD of a per-well mean scales as 1/sqrt(cells).
    y = 1.0 + X["x1"] + rng.normal(size=n) * (4.0 / np.sqrt(meta["cell_count"]))
    ctx = rq.build_context(sm.OLS(y, X).fit(), X, y, metadata=meta)
    fig, ax = _axes()
    stats = rq.draw_panel("cell_count_vs_effect", ctx, ax)

    assert stats["spearman_rho"] < -0.3
    assert stats["spearman_p"] < 1e-6
    assert stats["n_points"] == n
    assert stats["min_cell_count"] == pytest.approx(meta["cell_count"].min())
    assert stats["n_extreme"] > 0
    # Most of the |z| > 2 wells live in the smallest decile of cell counts.
    assert stats["frac_extreme_in_low_decile"] > 0.3

    # Control: constant noise, no relationship with cell count.
    flat = 1.0 + X["x1"] + rng.normal(size=n) * 0.3
    flat_ctx = rq.build_context(sm.OLS(flat, X).fit(), X, flat, metadata=meta)
    fig, ax = _axes()
    flat_stats = rq.draw_panel("cell_count_vs_effect", flat_ctx, ax)
    assert abs(flat_stats["spearman_rho"]) < 0.15


# ---------------------------------------------------------------------------
# Context construction: alignment is the one thing that must never be guessed
# ---------------------------------------------------------------------------


def test_labels_follow_the_index_when_rows_were_dropped():
    """patsy drops rows; positional alignment would then name the wrong well."""
    model, X, y, meta = _ols_case(n=96, seed=60)
    kept = [i for i in range(96) if i not in (0, 5, 40)]
    X_kept, y_kept = X.loc[kept], y.loc[kept]
    model = sm.OLS(y_kept, X_kept).fit()
    ctx = rq.build_context(model, X_kept, y_kept, metadata=meta)

    assert list(ctx.labels) == list(meta.loc[kept, schema.PRC_KEY])
    # The specific bug this guards: naive positional alignment would put the
    # metadata of well 0 onto the first fitted row, which is well 1.
    assert ctx.labels[0] == meta.loc[1, schema.PRC_KEY]
    assert ctx.labels[0] != meta.loc[0, schema.PRC_KEY]
    assert len(ctx.metadata) == len(kept)


def test_metadata_that_cannot_be_aligned_raises_instead_of_guessing():
    model, X, y, meta = _ols_case(n=96, seed=61)
    truncated = meta.iloc[:50].reset_index(drop=True).drop(index=[0, 1])
    with pytest.raises(ValueError, match="does not cover"):
        rq.build_context(model, X, y, metadata=truncated)


def test_mismatched_x_and_y_raise():
    model, X, y, _ = _ols_case(n=40, seed=62)
    with pytest.raises(ValueError, match="observations"):
        rq.build_context(model, X, np.asarray(y)[:-1])


def test_leverage_source_is_recorded_and_falls_back_for_a_mixed_model():
    rng = _stream(63, 2)
    n = 120
    X = _design(n, seed=63, n_predictors=1)
    groups = np.repeat(np.arange(12), 10)
    y = (1.0 + 2.0 * X["x1"] + np.repeat(rng.normal(size=12), 10)
         + rng.normal(size=n) * 0.3)
    model = sm.MixedLM(y, X, groups=groups).fit()
    ctx = rq.build_context(model, X, y, regression_type="mixed")
    assert ctx.leverage_source.startswith("design matrix")
    assert any("no hat matrix" in note for note in ctx.notes)
    np.testing.assert_allclose(
        ctx.leverage, rq.leverage_from_design(X.to_numpy(dtype=float)))

    # ...and an OLS fit uses statsmodels' own hat matrix.
    ols_ctx = _context()
    assert "influence" in ols_ctx.leverage_source


# ---------------------------------------------------------------------------
# The report driver
# ---------------------------------------------------------------------------


def test_report_writes_every_drawn_panel_and_a_combined_page(tmp_path):
    model, X, y, meta = _ols_case(n=96, seed=70)
    meta = meta.copy()
    meta[schema.PLATE_KEY] = np.where(np.arange(96) < 48, "plate1", "plate2")
    volcano = tmp_path / "ols_volcano_plot.pdf"
    manifest = rq.regression_qc_report(
        model, X, y, str(tmp_path), metadata=meta, regression_type="ols",
        volcano_path=str(volcano), verbose=False)

    drawn = [p for p in manifest["panels"] if p.status in ("written", "partial")]
    assert len(drawn) == len(manifest["written"]) >= 15
    for panel in drawn:
        assert os.path.isfile(panel.path)
        assert os.path.getsize(panel.path) > 1000
        assert os.path.basename(panel.path) == f"{panel.name}.pdf"
    assert os.path.isfile(manifest["combined"])
    assert os.path.isfile(manifest["report"])
    assert manifest["n_observations"] == 96
    assert manifest["n_predictors"] == X.shape[1]
    assert {p.name for p in manifest["panels"]} == set(rq.PANEL_ORDER)

    # The volcano is referenced, never redrawn.
    volcano_panel = next(p for p in manifest["panels"]
                         if p.name == "volcano_reference")
    assert volcano_panel.stats["state"] == "referenced"
    assert volcano_panel.stats["volcano_path"] == str(volcano)
    assert not any("volcano_plot" in name
                   for name in os.listdir(manifest["directory"]))


def test_every_skipped_panel_carries_a_reason_in_the_text_report(tmp_path):
    model, X, y, meta = _ols_case(n=96, seed=71)
    manifest = rq.regression_qc_report(model, X, y, str(tmp_path),
                                       metadata=meta, regression_type="ols",
                                       verbose=False)
    skipped = [p for p in manifest["panels"] if p.status == "skipped"]
    assert skipped, "a Gaussian fit must skip the binomial and count panels"
    report = open(manifest["report"], encoding="utf-8").read()
    for panel in skipped:
        assert panel.reason and len(panel.reason) > 20
        assert panel.path is None
        assert panel.name in report
        # The first few words of the reason survive into the report text.
        assert panel.reason.split(";")[0][:40] in report
    assert "SKIPPED" in report
    for name, _ in manifest["skipped"]:
        assert not os.path.exists(os.path.join(manifest["directory"],
                                               f"{name}.pdf"))


def test_a_lasso_degrades_to_the_panels_it_can_actually_support(tmp_path):
    """sklearn has no p-values and no covariance matrix. Say so; do not invent."""
    rng = _stream(72, 2)
    n = 150
    X = _design(n, seed=72, n_predictors=3)
    y = 1.0 + 2.0 * X["x1"] + rng.normal(size=n) * 0.5
    model = Lasso(alpha=0.05).fit(X, np.asarray(y))
    manifest = rq.regression_qc_report(model, X, y, str(tmp_path),
                                       regression_type="lasso", verbose=False)

    assert manifest["failed"] == []
    by_name = {p.name: p for p in manifest["panels"]}
    p_panel = by_name["p_value_histogram"]
    assert p_panel.status == "skipped"
    assert "no p-values" in p_panel.reason and "penalised" in p_panel.reason

    forest = by_name["coefficient_forest"]
    assert forest.status == "partial"
    assert "no covariance matrix" in forest.reason
    assert forest.stats["has_intervals"] is False
    assert os.path.isfile(forest.path)

    # The panels that only need residuals still work: a Lasso fit is still a fit.
    for name in ("residuals_vs_fitted", "observed_vs_predicted",
                 "cooks_distance", "influence", "vif", "condition_number"):
        assert by_name[name].status == "written", name
    assert by_name["cooks_distance"].stats["max_cooks"] > 0

    report = open(manifest["report"], encoding="utf-8").read()
    assert "PARTIAL" in report and "SKIPPED" in report


def test_a_rank_deficient_screen_design_degrades_instead_of_crashing(tmp_path):
    """More gRNAs than wells is routine here, and it makes leverage exactly 1.

    Every residual is then zero and every influence statistic is undefined. The
    report must say that, panel by panel, rather than dividing by zero or --
    far worse -- printing a Cook's distance of 0 for every well, which reads as
    "no influential wells" when the truth is "this fit interpolates the data".
    """
    rng = _stream(80, 2)
    n, p = 12, 18
    X = pd.DataFrame(
        np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))]),
        columns=["Intercept"] + [f"grna{i}" for i in range(p - 1)])
    y = pd.Series(rng.normal(size=n))
    model = sm.OLS(y, X).fit()
    manifest = rq.regression_qc_report(model, X, y, str(tmp_path),
                                       regression_type="ols", verbose=False)

    assert manifest["failed"] == []
    by_name = {panel.name: panel for panel in manifest["panels"]}
    assert by_name["cooks_distance"].status == "skipped"
    assert "leverage == 1" in by_name["cooks_distance"].reason
    assert by_name["dffits"].status == "skipped"
    assert "n > p + 1" in by_name["dffits"].reason
    # The design panels still work, and they are the ones that explain why.
    assert by_name["condition_number"].status == "written"
    assert by_name["vif"].stats["n_aliased"] >= 0
    assert len(manifest["written"]) >= 10


def test_the_report_leaks_no_pyplot_figures(tmp_path):
    """This repo has been bitten by figure leaks; the module never touches pyplot."""
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    model, X, y, meta = _ols_case(n=96, seed=73)
    rq.regression_qc_report(model, X, y, str(tmp_path), metadata=meta,
                            regression_type="ols", verbose=False)
    assert plt.get_fignums() == before


def test_a_broken_panel_is_reported_not_swallowed_and_not_fatal(tmp_path, monkeypatch):
    def boom(ctx, ax):
        raise RuntimeError("synthetic panel failure")

    title, group, _ = rq._PANEL_BY_NAME["qq_residuals"]
    monkeypatch.setitem(rq._PANEL_BY_NAME, "qq_residuals", (title, group, boom))

    model, X, y, _ = _ols_case(n=60, seed=74)
    manifest = rq.regression_qc_report(
        model, X, y, str(tmp_path),
        panels=["residuals_vs_fitted", "qq_residuals"], verbose=False)
    failed = {name: reason for name, reason in manifest["failed"]}
    assert "qq_residuals" in failed
    assert "synthetic panel failure" in failed["qq_residuals"]
    # The rest of the report still exists: a broken diagnostic must not destroy
    # an hour-long fit that already succeeded.
    assert len(manifest["written"]) == 1
    assert "FAILED" in open(manifest["report"], encoding="utf-8").read()

    with pytest.raises(RuntimeError, match="synthetic panel failure"):
        rq.regression_qc_report(model, X, y, str(tmp_path),
                                panels=["qq_residuals"], strict=True,
                                verbose=False)


def test_the_combined_page_shows_the_skip_reason_on_the_page(tmp_path, monkeypatch):
    """The reason has to be ON the report, not only in the manifest."""
    captured = {}
    real_save = rq._save

    def spy(fig, path):
        if os.path.basename(path) == "regression_qc_report.pdf":
            captured["texts"] = [t.get_text() for ax in fig.axes for t in ax.texts]
        return real_save(fig, path)

    monkeypatch.setattr(rq, "_save", spy)
    model, X, y, meta = _ols_case(n=96, seed=75)
    manifest = rq.regression_qc_report(model, X, y, str(tmp_path),
                                       metadata=meta, regression_type="ols",
                                       verbose=False)
    # The tile wraps its reason, so compare on collapsed whitespace: wrapping
    # only ever turns a space into a newline.
    joined = " ".join(" ".join(captured["texts"]).split())
    assert "SKIPPED" in joined
    for _, reason in manifest["skipped"]:
        head = " ".join(reason.split(";")[0].split())[:40]
        assert head in joined
    # Every panel got a tile, drawn or skipped.
    assert joined.count("SKIPPED") == len(manifest["skipped"])


def test_report_refuses_to_write_nowhere():
    model, X, y, _ = _ols_case(n=40, seed=76)
    with pytest.raises(ValueError, match="destination folder"):
        rq.regression_qc_report(model, X, y, None)


def test_unknown_panel_names_are_rejected(tmp_path):
    model, X, y, _ = _ols_case(n=40, seed=77)
    with pytest.raises(ValueError, match="unknown QC panel"):
        rq.regression_qc_report(model, X, y, str(tmp_path), panels=["nope"])
    fig, ax = _axes()
    with pytest.raises(KeyError):
        rq.draw_panel("nope", rq.build_context(model, X, y), ax)


def test_panel_names_cover_every_group_and_match_the_order():
    grouped = []
    for group in ("fit", "influence", "design", "response", "screen"):
        names = rq.panel_names(group)
        assert names, group
        grouped.extend(names)
    assert sorted(grouped) == sorted(rq.PANEL_ORDER)
    assert rq.panel_names() == rq.PANEL_ORDER
    with pytest.raises(ValueError, match="unknown panel group"):
        rq.panel_names("nonsense")


def test_format_qc_report_names_every_panel_and_the_headline_numbers(tmp_path):
    model, X, y, meta = _ols_case(n=96, seed=78)
    manifest = rq.regression_qc_report(model, X, y, str(tmp_path),
                                       metadata=meta, regression_type="ols",
                                       verbose=False)
    text = rq.format_qc_report(manifest)
    for panel in manifest["panels"]:
        assert panel.title in text
    assert "observations     : 96 wells" in text
    assert "Model fit" in text and "Screen-level structure" in text
    r2 = next(p for p in manifest["panels"]
              if p.name == "observed_vs_predicted").stats["r2"]
    assert f"{r2:.6g}" in text


# ---------------------------------------------------------------------------
# The caller: the objects spacr.ml.regression actually produces
# ---------------------------------------------------------------------------


def _screen_frame(n_plates=1, n_rows=8, n_cols=12, n_genes=4, seed=80):
    """A long-format screen table shaped like the one perform_regression builds."""
    rng = _stream(seed, 2)
    meta = _plate_metadata(n_rows=n_rows, n_cols=n_cols,
                           plates=tuple(f"plate{i + 1}" for i in range(n_plates)),
                           seed=seed)
    n = len(meta)
    genes = [f"gene{i + 1}" for i in range(n_genes)]
    grnas = [f"{g}_sg{j + 1}" for g in genes for j in range(2)]
    frame = meta.copy()
    frame["grna"] = rng.choice(grnas, size=n)
    frame["gene"] = frame["grna"].str.split("_").str[0]
    frame["fraction"] = rng.uniform(0.2, 1.0, n)
    frame["gene_fraction"] = frame["fraction"]
    effect = frame["gene"].map({g: v for g, v in
                                zip(genes, rng.normal(scale=1.5, size=n_genes))})
    frame["predictions"] = (0.5 + effect * frame["fraction"]
                            + rng.normal(scale=0.2, size=n))
    return frame


def test_the_report_runs_on_what_spacr_ml_regression_actually_builds(tmp_path):
    """Check the caller, not the callee: patsy design + spacr.ml.regression_model.

    This mirrors the body of :func:`spacr.ml.regression` exactly -- the same
    formula, the same ``dmatrices`` call, the same model factory -- so the hook
    requested in the report is proved against the real objects rather than
    against a tidy fixture.
    """
    from patsy import dmatrices

    from spacr.ml import prepare_formula, regression_model

    frame = _screen_frame()
    formula = prepare_formula("predictions", random_row_column_effects=False)
    y, X = dmatrices(formula, data=frame, return_type="dataframe")
    model = regression_model(X, y, regression_type="ols")

    manifest = rq.regression_qc_report(
        model, X, y, str(tmp_path),
        metadata=frame.loc[y.index, [schema.PLATE_KEY, schema.ROW_KEY,
                                     schema.COLUMN_KEY, schema.PRC_KEY,
                                     "cell_count"]],
        regression_type="ols", verbose=False)

    assert manifest["failed"] == []
    assert manifest["n_observations"] == len(frame)
    assert manifest["n_predictors"] == X.shape[1] > 20
    by_name = {p.name: p for p in manifest["panels"]}
    assert by_name["residuals_vs_fitted"].stats["n_points"] == len(frame)
    # The one-hot row/column terms are in the design, so the collinearity
    # panels have something real to say about them.
    assert by_name["condition_number"].stats["n_singular_values"] == X.shape[1]
    assert by_name["vif"].status == "written"
    # Wells are labelled by prc, which is what sends a user to a microscope.
    worst = by_name["cooks_distance"].stats["max_label"]
    assert worst in set(frame[schema.PRC_KEY])


# ---------------------------------------------------------------------------
# Residual standardisation, per model class
#
# The defect this section exists for: `model.scale` is present on every
# statsmodels results object and means a different thing on each of them.
# Treating it as "the error variance of y - fitted" is right for OLS, GLM and
# MixedLM and WRONG for WLS, RLM/HuberT, QuantReg and BetaModel -- five of
# spaCR's seventeen regression types. On an RLM fit the error is a factor of
# sqrt(scale), so it is unit-dependent and changes sign: a fraction response
# under-flags, a per-well count over-flags. The suite that shipped exercised
# only sm.OLS and friends, so nothing failed.
#
# Every test below either pins the standardisation against an INDEPENDENT
# statsmodels quantity (get_influence().resid_studentized, RLMResults.sresid,
# RegressionResults.resid_pearson) or pins the skip and its reason.
# ---------------------------------------------------------------------------


def _naive_std_resid(model, ctx):
    """The pre-fix formula: (y - fitted) / sqrt(model.scale * (1 - h)).

    Kept in the tests, not in the module, so the size and the direction of the
    old error can be asserted rather than described.
    """
    return ctx.resid / np.sqrt(float(model.scale)
                               * np.clip(1.0 - ctx.leverage, 1e-12, None))


_STANDARDISATION_PANELS = ("scale_location", "qq_residuals", "cooks_distance",
                           "influence", "dffits", "cell_count_vs_effect")


def _panels_skip_without_a_scale(ctx, fragment):
    """Assert the six standardised-residual panels skip, naming the reason."""
    for name in _STANDARDISATION_PANELS:
        fig, ax = _axes()
        with pytest.raises(rq.PanelUnavailable) as excinfo:
            rq.draw_panel(name, ctx, ax)
        assert fragment in str(excinfo.value), name
        assert "needs a standardised residual" in str(excinfo.value), name


def _wls_case(n=300, seed=100, outlier_well=7, shift=6.0):
    """A WLS fit whose weights are cell counts, with one planted bad well.

    The noise SD falls as 1/sqrt(cell count), which is the reason to weight the
    fit at all, and the planted well is a HIGH-count well that sits far from the
    line -- the well a weighted fit cares most about and an unweighted
    standardisation cannot see.
    """
    rng = _stream(seed, 2)
    X = _design(n, seed=seed, n_predictors=2)
    counts = rng.integers(30, 900, size=n).astype(float)
    y = (2.0 + 1.5 * X["x1"] - 0.5 * X["x2"]
         + rng.normal(size=n) * 0.7 * np.sqrt(400.0 / counts))
    counts[outlier_well] = 900.0
    y.iloc[outlier_well] += shift * 0.7 * np.sqrt(400.0 / 900.0)
    model = sm.WLS(y, X, weights=counts).fit()
    return model, X, y, counts


def test_wls_standardises_the_weighted_residual_not_the_raw_one():
    """WLS's model.scale is the variance of sqrt(w)*(y-fitted), not of (y-fitted).

    With per-well cell counts for w that is hundreds of times too large, so the
    pre-fix formula divided every residual by a number ~15x too big and the
    influence panels reported that nothing was wrong.
    """
    model, X, y, counts = _wls_case()
    ctx = rq.build_context(model, X, y, weights=counts, regression_type="wls")

    assert ctx.standardisation_available
    # Pinned against statsmodels' own weighted Pearson residual, which is
    # wresid / sqrt(scale) -- an independent implementation of the same idea.
    expected = np.asarray(model.resid_pearson) / np.sqrt(1.0 - ctx.leverage)
    np.testing.assert_allclose(ctx.std_resid, expected, rtol=1e-10, atol=1e-12)
    assert np.std(ctx.std_resid, ddof=1) == pytest.approx(1.0, abs=0.2)

    # ...and the defect, measured. The old formula is smaller by ~sqrt(mean w).
    naive = _naive_std_resid(model, ctx)
    ratio = np.abs(ctx.std_resid) / np.abs(naive)
    assert ratio.min() > 5.0, "the pre-fix formula was not materially different"
    assert np.max(np.abs(naive)) < 2.0
    assert np.sum(np.abs(naive) > 2.0) == 0, "the old formula flagged nothing"
    assert np.sum(np.abs(ctx.std_resid) > 2.0) >= 1

    # The planted high-count outlier is the worst well, and it is named.
    fig, ax = _axes()
    stats = rq.draw_panel("cooks_distance", ctx, ax)
    assert stats["max_index"] == 7
    assert stats["n_above"] >= 1


def test_wls_leverage_uses_the_weights_the_fit_was_given():
    """A caller's `weights` cannot silently redefine the fitted hat matrix.

    ``model.scale``, the residual and the hat diagonal have to be formed with
    the SAME weights or none of them agree; the fit's own weights are the only
    ones that satisfy that, so they win over anything passed in.
    """
    model, X, y, counts = _wls_case(seed=101)
    wrong = np.ones_like(counts)
    ctx = rq.build_context(model, X, y, weights=wrong, regression_type="wls")

    np.testing.assert_allclose(
        ctx.leverage,
        rq.leverage_from_design(X.to_numpy(dtype=float), weights=counts),
        rtol=1e-10)
    # trace(H) == p is the identity that says the hat matrix belongs to this fit.
    assert ctx.leverage.sum() == pytest.approx(X.shape[1], rel=1e-8)
    assert "fit weights" in ctx.leverage_source
    assert any("not the 'weights' argument" in note for note in ctx.notes)
    # The caller's weights are still what the cell-count panel sees; only the
    # influence maths is taken from the fit.
    np.testing.assert_allclose(ctx.weights, wrong)


def test_wls_without_recoverable_weights_skips_instead_of_guessing():
    """No weights, no weighted metric: say so rather than pick a scale."""
    model, X, y, counts = _wls_case(seed=102)

    class _WeightlessWLS:
        """A WLS results object that has lost its weights (a pickled/proxied fit)."""

        def __init__(self, inner):
            self._inner = inner
            self.model = type("WLS", (), {"weights": None})()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    ctx = rq.build_context(_WeightlessWLS(model), X, y, regression_type="wls")
    assert not ctx.standardisation_available
    assert "does not expose the per-observation weights" in ctx.standardisation.reason
    assert np.isnan(ctx.scale)
    assert np.all(np.isnan(ctx.std_resid))
    _panels_skip_without_a_scale(ctx, "does not expose the per-observation weights")


def _rlm_case(n=240, seed=110, unit=1.0, outlier_well=11, shift=8.0):
    """The same fit twice over, once per response unit. ``unit`` scales y."""
    rng = _stream(seed, 2)
    X = _design(n, seed=seed, n_predictors=2)
    y = 0.5 + 0.10 * X["x1"] - 0.04 * X["x2"] + rng.normal(size=n) * 0.05
    y.iloc[outlier_well] += shift * 0.05
    y = y * unit
    model = sm.RLM(y, X, M=sm.robust.norms.HuberT(t=1.345)).fit()
    return model, X, y


def test_rlm_scale_is_a_standard_deviation_not_a_variance():
    """RLMResults.scale is sigma. Dividing by sqrt(sigma) is wrong by sqrt(sigma)."""
    model, X, y = _rlm_case()
    ctx = rq.build_context(model, X, y, regression_type="rlm")

    assert ctx.standardisation_available
    assert ctx.scale == pytest.approx(float(model.scale) ** 2, rel=1e-12)
    # statsmodels' own robust standardised residual is resid / scale.
    expected = np.asarray(model.sresid) / np.sqrt(1.0 - ctx.leverage)
    np.testing.assert_allclose(ctx.std_resid, expected, rtol=1e-10, atol=1e-12)
    assert np.std(ctx.std_resid, ddof=1) == pytest.approx(1.0, abs=0.25)

    naive = _naive_std_resid(model, ctx)
    # The exact shape of the error: the old formula is the right one divided
    # by sqrt(scale), which is a number in the units of the response.
    np.testing.assert_allclose(naive / np.sqrt(float(model.scale)),
                               ctx.std_resid, rtol=1e-10)
    # y is a fraction here (SD well below 1), so the old formula SHRANK
    # every |z| -- real influential wells went unreported.
    assert float(model.scale) < 1.0
    assert np.max(np.abs(naive)) < 0.5 * np.max(np.abs(ctx.std_resid))


def test_rlm_flags_the_same_wells_whatever_the_response_units():
    """The bug's signature: the old |z| moved with the units of y, the fix does not.

    A per-well fraction and the same response expressed per 100 cells are the
    same science. Under the pre-fix formula the count-scaled fit inflated every
    |z| by sqrt(scale) ~ 2.2 and invented outliers; the fraction-scaled fit
    deflated them by ~4x and hid the real one.
    """
    fraction_model, X, y_fraction = _rlm_case(unit=1.0)
    count_model, _, y_count = _rlm_case(unit=100.0)
    frac = rq.build_context(fraction_model, X, y_fraction, regression_type="rlm")
    count = rq.build_context(count_model, X, y_count, regression_type="rlm")

    # Same wells, same z-scores, whatever the units.
    np.testing.assert_allclose(frac.std_resid, count.std_resid, rtol=1e-6)
    flagged = lambda ctx: set(np.flatnonzero(np.abs(ctx.std_resid) > 2.0))
    assert flagged(frac) == flagged(count)
    assert 11 in flagged(frac), "the planted well must be flagged at all"

    # ...and the same statement about the formula that shipped.
    naive_frac = _naive_std_resid(fraction_model, frac)
    naive_count = _naive_std_resid(count_model, count)
    assert not np.allclose(naive_frac, naive_count, rtol=0.1)
    assert np.max(np.abs(naive_count)) > 2.0 * np.max(np.abs(naive_frac))
    assert set(np.flatnonzero(np.abs(naive_frac) > 2.0)) == set(), (
        "the pre-fix formula hid the planted well on a fraction response")
    assert len(set(np.flatnonzero(np.abs(naive_count) > 2.0))) > len(flagged(count)), (
        "the pre-fix formula invented outliers on a count response")


def test_huber_is_the_same_backend_and_gets_the_same_treatment():
    """spaCR's 'huber' is RLM with a caller-set t; it must not diverge from 'rlm'."""
    rng = _stream(112, 2)
    n = 200
    X = _design(n, seed=112, n_predictors=2)
    y = 1.0 + 0.8 * X["x1"] + rng.normal(size=n) * 0.4
    model = sm.RLM(y, X, M=sm.robust.norms.HuberT(t=1.2)).fit()
    ctx = rq.build_context(model, X, y, regression_type="huber")
    assert ctx.scale == pytest.approx(float(model.scale) ** 2, rel=1e-12)
    assert "standard deviation" in ctx.standardisation.source


def _quantile_case(n=240, seed=120, q=0.5):
    rng = _stream(seed, 2)
    X = _design(n, seed=seed, n_predictors=2)
    y = 2.0 + 1.5 * X["x1"] - 0.4 * X["x2"] + rng.normal(size=n) * 0.6
    return sm.QuantReg(y, X).fit(q=q), X, y


@pytest.mark.parametrize("q", [0.5, 0.9])
def test_quantile_regression_has_no_error_scale_and_says_so(q):
    """QuantRegResults.scale is hard-coded to 1.0. It is not a variance.

    Standardising by it makes |z| equal the raw residual in whatever units y
    happens to be in: on a per-well count response every well is an outlier and
    on a fraction response none is. There is no correct number here, so the
    panels that need one are skipped with the reason on the report.
    """
    model, X, y = _quantile_case(q=q)
    assert float(model.scale) == 1.0, "the placeholder this test is about"
    ctx = rq.build_context(model, X, y, regression_type="quantile")

    assert not ctx.standardisation_available
    assert np.isnan(ctx.scale)
    assert np.all(np.isnan(ctx.std_resid))
    reason = ctx.standardisation.reason
    assert f"{q:g} quantile" in reason
    assert "hard-coded to 1.0" in reason and "not a variance" in reason
    _panels_skip_without_a_scale(ctx, "quantile regression estimates")


def test_quantile_report_skips_five_panels_and_draws_the_rest(tmp_path):
    model, X, y = _quantile_case(seed=121)
    meta = _wells_for(len(y), seed=121)
    manifest = rq.regression_qc_report(model, X, y, str(tmp_path), metadata=meta,
                                       regression_type="quantile", verbose=False)

    assert manifest["failed"] == []
    assert manifest["residual_scale_available"] is False
    by_name = {p.name: p for p in manifest["panels"]}
    for name in _STANDARDISATION_PANELS:
        assert by_name[name].status == "skipped", name
        assert by_name[name].path is None
        assert "quantile regression" in by_name[name].reason
    # The panels that do not need a scale are unaffected and still drawn.
    for name in ("residuals_vs_fitted", "residual_distribution",
                 "observed_vs_predicted", "vif", "condition_number",
                 "coefficient_forest", "row_effects"):
        assert by_name[name].status in ("written", "partial"), name

    report = open(manifest["report"], encoding="utf-8").read()
    assert "standardised by  : not available" in report
    assert "quantile regression estimates" in report


def _beta_case(n=260, seed=130, precision=30.0):
    rng = _stream(seed, 2)
    X = _design(n, seed=seed, n_predictors=2)
    mu = 1.0 / (1.0 + np.exp(-(0.3 + 0.9 * X["x1"] - 0.4 * X["x2"])))
    y = pd.Series(rng.beta(mu * precision, (1.0 - mu) * precision), index=X.index)
    from statsmodels.othermod.betareg import BetaModel
    return BetaModel(endog=y, exog=X).fit(disp=0), X, y


def test_beta_regression_standardises_the_pearson_residual():
    """BetaResults.scale is 1.0 -- correct, but only against the Pearson residual.

    Applied to (y - mu) on a fraction response it understates every |z| by an
    order of magnitude, because mu(1-mu)/(1+phi) is far below 1.
    """
    model, X, y = _beta_case()
    ctx = rq.build_context(model, X, y, regression_type="beta")

    assert ctx.standardisation_available
    assert ctx.scale == 1.0
    # Pinned against statsmodels' own MLEInfluence for this fit.
    np.testing.assert_allclose(ctx.std_resid,
                               model.get_influence().resid_studentized,
                               rtol=1e-10, atol=1e-12)
    assert np.std(ctx.std_resid, ddof=1) == pytest.approx(1.0, abs=0.25)

    naive = _naive_std_resid(model, ctx)
    assert np.std(naive, ddof=1) < 0.25, "the pre-fix formula was not the bug claimed"
    assert np.max(np.abs(ctx.std_resid)) > 5.0 * np.max(np.abs(naive))


def test_a_beta_fit_is_not_reported_as_gaussian_least_squares():
    """The wrapper class is BetaResultsWrapper, so a startswith('BetaModel') missed it."""
    model, X, y = _beta_case(seed=131)
    ctx = rq.build_context(model, X, y, regression_type="beta")
    assert ctx.family == "Beta"
    assert ctx.link == "Logit"
    assert "least squares" not in ctx.family


def test_beta_without_a_pearson_residual_skips_rather_than_using_scale_1():
    """A statsmodels build with no BetaResults.resid_pearson must not fall back."""
    model, X, y = _beta_case(seed=132)

    class _NoPearson:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "resid_pearson":
                raise AttributeError(name)
            return getattr(self._inner, name)

    ctx = rq.build_context(_NoPearson(model), X, y, regression_type="beta")
    assert not ctx.standardisation_available
    assert "resid_pearson" in ctx.standardisation.reason
    assert "generic likelihood default of 1.0" in ctx.standardisation.reason
    _panels_skip_without_a_scale(ctx, "resid_pearson")


def _hinge_case(n=200, seed=140):
    """A LinearSVC fitted on a binarised response, handed the CONTINUOUS one.

    That is what ``spacr.ml.regression_model`` produces: the binarisation
    happens inside the backend, so the caller still holds the fraction.
    """
    from sklearn.svm import LinearSVC

    rng = _stream(seed, 2)
    X = _design(n, seed=seed, n_predictors=2)
    y = 0.5 + 0.3 * X["x1"] + rng.normal(size=n) * 0.15
    labels = (y > y.median()).astype(float)
    model = LinearSVC(C=1.0, loss="hinge", dual=True, max_iter=20000,
                      random_state=0).fit(X, labels)
    return model, X, y


def test_a_hinge_classifier_has_no_standardised_residual():
    """A hinge loss has no error variance; RSS/(n-p) over class labels is not one."""
    model, X, y = _hinge_case()
    ctx = rq.build_context(model, X, y, regression_type="hinge")

    assert not ctx.standardisation_available
    assert "is a classifier" in ctx.standardisation.reason
    assert "no error variance" in ctx.standardisation.reason
    assert np.isnan(ctx.scale)
    _panels_skip_without_a_scale(ctx, "is a classifier")


def test_a_hinge_fit_states_that_its_predictions_are_class_labels(tmp_path):
    """R² of 0/1 predictions against a continuous response is not fit quality."""
    model, X, y = _hinge_case(seed=141)
    manifest = rq.regression_qc_report(model, X, y, str(tmp_path),
                                       regression_type="hinge", verbose=False)

    assert manifest["failed"] == []
    by_name = {p.name: p for p in manifest["panels"]}
    for name in ("residuals_vs_fitted", "residual_distribution",
                 "observed_vs_predicted"):
        assert by_name[name].status == "partial", name
        assert "class label" in by_name[name].reason, name
    assert by_name["p_value_histogram"].status == "skipped"
    assert by_name["coefficient_forest"].status == "partial"
    assert "no covariance matrix" in by_name["coefficient_forest"].reason
    report = open(manifest["report"], encoding="utf-8").read()
    assert "hinge loss (linear classifier)" in report
    assert "class label" in report


def test_mixedlm_residuals_are_conditional_and_the_scale_matches_them():
    """MixedLM.fittedvalues includes the BLUPs, so scale is the right variance.

    The number is fine and the wording is the risk: a reader who takes these as
    MARGINAL residuals reads a spread that is far too small. The report says
    which one it is.
    """
    rng = _stream(150, 2)
    n = 300
    X = _design(n, seed=150, n_predictors=1)
    groups = np.repeat(np.arange(30), 10)
    y = (1.0 + 0.8 * X["x1"] + np.repeat(rng.normal(size=30), 10)
         + rng.normal(size=n) * 0.5)
    model = sm.MixedLM(y, X, groups=groups).fit()
    ctx = rq.build_context(model, X, y, regression_type="mixed")

    assert ctx.standardisation_available
    assert ctx.scale == pytest.approx(float(model.scale), rel=1e-12)
    marginal = y - X.to_numpy(dtype=float) @ np.asarray(model.fe_params)
    assert np.std(ctx.resid, ddof=1) < 0.7 * np.std(marginal, ddof=1), (
        "the fixture has no random-effect signal, so it proves nothing")
    assert np.std(ctx.std_resid, ddof=1) == pytest.approx(1.0, abs=0.2)
    assert "CONDITIONAL" in ctx.standardisation.source
    assert "conditional residual" in ctx.standardisation.metric


@pytest.mark.parametrize("family,response", [
    ("gaussian", "continuous"),
    ("poisson", "count"),
    ("binomial_binary", "binary"),
    ("binomial_weighted", "fraction"),
    ("quasi_binomial", "fraction"),
])
def test_every_glm_family_standardises_its_pearson_residual(family, response):
    """A GLM's scale IS its dispersion -- against the Pearson residual, not y-mu."""
    rng = _stream(160, 2)
    n = 300
    X = _design(n, seed=160, n_predictors=2)
    eta = 0.3 + 0.8 * X["x1"] - 0.3 * X["x2"]
    weights = None
    if response == "continuous":
        y = eta + rng.normal(size=n) * 0.5
        model = sm.GLM(y, X, family=sm.families.Gaussian()).fit()
    elif response == "count":
        y = pd.Series(rng.poisson(np.exp(eta + 1.5)).astype(float), index=X.index)
        model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    else:
        mu = 1.0 / (1.0 + np.exp(-eta))
        if response == "binary":
            y = pd.Series((rng.uniform(size=n) < mu).astype(float), index=X.index)
            model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        else:
            weights = rng.integers(30, 600, size=n).astype(float)
            y = pd.Series(rng.binomial(weights.astype(int), mu) / weights,
                          index=X.index)
            fit_kwargs = {"scale": "X2"} if family == "quasi_binomial" else {}
            model = sm.GLM(y, X, family=sm.families.Binomial(),
                           var_weights=weights).fit(**fit_kwargs)

    ctx = rq.build_context(model, X, y, weights=weights)
    assert ctx.standardisation_available
    assert ctx.scale == pytest.approx(float(model.scale), rel=1e-12)
    assert "Pearson" in ctx.standardisation.metric
    # statsmodels' own studentised residual for a GLM is the cross-check.
    np.testing.assert_allclose(ctx.std_resid,
                               model.get_influence().resid_studentized,
                               rtol=1e-9, atol=1e-12)
    assert np.std(ctx.std_resid, ddof=1) == pytest.approx(1.0, abs=0.3)


def test_the_count_panel_uses_pearson_and_not_deviance_residuals():
    """Deviance/df and Pearson chi2/df are different numbers; only one is dispersion."""
    rng = _stream(161, 2)
    n = 400
    X = _design(n, seed=161, n_predictors=1)
    mu = np.exp(1.4 + 0.5 * X["x1"])
    y = pd.Series(rng.poisson(mu).astype(float), index=X.index)
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    ctx = rq.build_context(model, X, y, regression_type="poisson")

    fig, ax = _axes()
    stats = rq.draw_panel("count_fit", ctx, ax)
    assert stats["dispersion"] == pytest.approx(
        float(model.pearson_chi2) / float(model.df_resid), rel=1e-9)
    assert stats["dispersion"] != pytest.approx(
        float(model.deviance) / float(model.df_resid), rel=1e-3)


@pytest.mark.parametrize("estimator", ["lasso", "ridge", "elasticnet"])
def test_a_penalised_sklearn_fit_estimates_its_scale_and_names_the_estimate(estimator):
    """sklearn reports no dispersion; RSS/(n-p) is used and disclosed as such."""
    from sklearn.linear_model import ElasticNet, Lasso, Ridge

    rng = _stream(170, 2)
    n = 200
    X = _design(n, seed=170, n_predictors=3)
    y = 1.0 + 2.0 * X["x1"] + rng.normal(size=n) * 0.5
    model = {"lasso": Lasso(alpha=0.01), "ridge": Ridge(alpha=1.0),
             "elasticnet": ElasticNet(alpha=0.01)}[estimator].fit(X, np.asarray(y))
    ctx = rq.build_context(model, X, y, regression_type=estimator)

    assert ctx.standardisation_available
    assert not hasattr(model, "scale"), "sklearn grew a scale; revisit the registry"
    assert ctx.scale == pytest.approx(
        float(np.sum(ctx.resid ** 2)) / (ctx.n - ctx.p), rel=1e-12)
    assert "RSS / (n - p) estimated here" in ctx.standardisation.source
    assert type(model).__name__ in ctx.standardisation.source
    assert ctx.family == "Gaussian (penalised least squares)"


@pytest.mark.parametrize("estimator", ["ridge", "elasticnet", "hinge"])
def test_no_sklearn_backend_is_given_p_values_or_intervals(estimator, tmp_path):
    """The Lasso contract, held for every other sklearn backend spaCR can fit."""
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.svm import LinearSVC

    rng = _stream(171, 2)
    n = 180
    X = _design(n, seed=171, n_predictors=3)
    y = 1.0 + 2.0 * X["x1"] + rng.normal(size=n) * 0.5
    if estimator == "hinge":
        model = LinearSVC(C=1.0, loss="hinge", dual=True, max_iter=20000,
                          random_state=0).fit(X, (y > y.median()).astype(float))
    else:
        model = ({"ridge": Ridge(alpha=1.0),
                  "elasticnet": ElasticNet(alpha=0.01)}[estimator]
                 .fit(X, np.asarray(y)))

    manifest = rq.regression_qc_report(model, X, y, str(tmp_path),
                                       regression_type=estimator, verbose=False)
    assert manifest["failed"] == []
    by_name = {p.name: p for p in manifest["panels"]}
    assert by_name["p_value_histogram"].status == "skipped"
    assert "no p-values" in by_name["p_value_histogram"].reason
    assert by_name["coefficient_forest"].status == "partial"
    assert by_name["coefficient_forest"].stats["has_intervals"] is False


def test_a_gls_fit_is_not_standardised_as_though_it_were_ols():
    """GLS's scale is in a whitened metric. spaCR refuses 'gls'; so does the QC."""
    rng = _stream(180, 2)
    n = 120
    X = _design(n, seed=180, n_predictors=1)
    y = 1.0 + 0.7 * X["x1"] + rng.normal(size=n) * 0.4
    sigma = np.eye(n) + 0.3 * np.eye(n, k=1) + 0.3 * np.eye(n, k=-1)
    model = sm.GLS(y, X, sigma=sigma).fit()
    ctx = rq.build_context(model, X, y, regression_type="gls")

    assert not ctx.standardisation_available
    assert "WHITENED" in ctx.standardisation.reason
    assert ctx.family == "Gaussian (generalised least squares)"
    _panels_skip_without_a_scale(ctx, "WHITENED")


def test_a_results_object_with_no_fitted_values_is_refused_loudly():
    """spaCR's horseshoe adapter reports params and p-values and nothing else.

    It has no fittedvalues and no predict(), so there is no residual and no QC
    to do. That has to stop here, by name, rather than reach a panel.
    """
    X = _design(30, seed=190, n_predictors=2)
    y = pd.Series(np.linspace(0.0, 1.0, 30), index=X.index)

    class _HorseshoeLike:
        params = pd.Series([0.1, 0.2], index=["Intercept", "x1"])
        pvalues = pd.Series([0.5, 0.01], index=["Intercept", "x1"])

    with pytest.raises(ValueError, match="neither fittedvalues nor predict"):
        rq.build_context(_HorseshoeLike(), X, y, regression_type="horseshoe")


def test_the_model_kind_registry_tells_ols_and_wls_apart():
    """Both fits return a RegressionResultsWrapper; only results.model separates them.

    This is why the registry is keyed on the MODEL class and not on the results
    class -- and OLS subclasses WLS in statsmodels, so the MRO order matters too.
    """
    rng = _stream(191, 2)
    n = 60
    X = _design(n, seed=191, n_predictors=1)
    y = 1.0 + X["x1"] + rng.normal(size=n) * 0.3
    weights = rng.uniform(1.0, 5.0, size=n)
    ols = sm.OLS(y, X).fit()
    wls = sm.WLS(y, X, weights=weights).fit()

    assert type(ols).__name__ == type(wls).__name__
    assert rq._model_kind(ols)[0] == "OLS"
    assert rq._model_kind(wls)[0] == "WLS"
    # ...and an unrecognised object falls through to the estimated scale, which
    # is never `model.scale` taken on trust.
    class _Odd:
        scale = 12345.0
        fittedvalues = np.zeros(n)

    assert rq._model_kind(_Odd())[0] is None
    std = rq.resolve_residual_standardisation(_Odd(), np.ones(n), n, 2)
    assert std.available and std.variance != 12345.0
    assert "no dispersion this module recognises" in std.source


# ---------------------------------------------------------------------------
# The sweep: every regression type spacr.ml can hand this module
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=2)
def _all_fitted_models(n=240, seed=200):
    """One fit per spaCR regression type, built by spacr.ml.regression_model.

    Cached: sixteen fits (two of them iterative) is a second of work, and the
    sweeps below ask for the set once per parametrised case.

    Built through the real dispatcher rather than by hand: the point of this
    sweep is that no model class reaches the QC report without a rule, and a
    hand-rolled fixture would only ever cover the classes someone remembered.
    """
    from spacr.ml import REGRESSION_TYPES, regression_model

    rng = _stream(seed, 2)
    X = _design(n, seed=seed, n_predictors=2)
    eta = 0.3 + 0.9 * X["x1"] - 0.4 * X["x2"]
    counts = rng.integers(40, 600, size=n).astype(float)
    mu = 1.0 / (1.0 + np.exp(-eta))
    continuous = pd.Series(eta + rng.normal(size=n) * 0.5, index=X.index)
    # A per-well fraction drawn the way each family assumes it was: binomial
    # counting for the binomial backends, a beta draw for beta regression. A
    # single shared fraction would leave the binomial fits over-dispersed by
    # construction, which is a property of the fixture and not of the QC.
    binomial_fraction = pd.Series(
        rng.binomial(counts.astype(int), mu) / counts, index=X.index)
    beta_fraction = pd.Series(np.clip(rng.beta(mu * 40.0, (1.0 - mu) * 40.0),
                                      1e-4, 1 - 1e-4), index=X.index)
    integers = pd.Series(rng.poisson(np.exp(eta + 1.5)).astype(float), index=X.index)
    groups = np.repeat(np.arange(n // 10), 10)
    # A real random intercept, so the mixed fit has a variance component to
    # find. Without one the MLE sits on the boundary, the optimiser warns, and
    # the "scale is the residual variance" claim is tested against a fit that
    # is not a mixed model in anything but name.
    clustered = pd.Series(
        eta + np.repeat(rng.normal(scale=1.0, size=n // 10), 10)
        + rng.normal(size=n) * 0.5, index=X.index)

    plan = {
        "ols": (continuous, {}),
        "wls": (continuous, {"weights": counts}),
        "rlm": (continuous, {}),
        "huber": (continuous, {"huber_t": 1.2}),
        "glm": (binomial_fraction, {"weights": counts}),
        "poisson": (integers, {}),
        "quasi_binomial": (binomial_fraction, {"weights": counts}),
        "beta": (beta_fraction, {}),
        "logit": (binomial_fraction, {"weights": counts}),
        "probit": (binomial_fraction, {"weights": counts}),
        "quantile": (continuous, {"quantile": 0.5}),
        "mixed": (clustered, {"groups": groups}),
        "lasso": (continuous, {"alpha": 0.01}),
        "ridge": (continuous, {"alpha": 1.0}),
        "elasticnet": (continuous, {"alpha": 0.01, "l1_ratio": 0.5}),
        "hinge": (continuous, {"hinge_threshold": float(continuous.median())}),
    }
    # horseshoe is the one type with no fitted values at all; it has its own
    # test above and cannot be built without spacr.power_model.
    assert set(plan) | {"horseshoe"} == set(REGRESSION_TYPES), (
        "a regression type appeared or vanished; the QC sweep must follow it")

    fits = {}
    for name, (y, kwargs) in plan.items():
        model = regression_model(X, y, regression_type=name, **kwargs)
        fits[name] = (model, X, y, kwargs.get("weights"))
    return fits


#: The types for which no correct error scale exists, with the word that must
#: appear in the stated reason. Everything else must standardise correctly.
_NO_SCALE_TYPES = {
    "quantile": "quantile regression estimates",
    "hinge": "is a classifier",
}


@pytest.mark.parametrize("regression_type", [
    "ols", "wls", "rlm", "huber", "glm", "poisson", "quasi_binomial", "beta",
    "logit", "probit", "quantile", "mixed", "lasso", "ridge", "elasticnet",
    "hinge",
])
def test_every_regression_type_standardises_correctly_or_states_why(regression_type):
    """The contract, for all sixteen fittable backends. No silent third option.

    Either the standardised residual is a z-score -- spread ~1, so a |z| > 2 on
    the influence panels means the same thing whatever was fitted and whatever
    the response is measured in -- or there is none and the reason is on the
    report. What must never happen is a number in between, which is what
    dividing by an unexamined `model.scale` produced.
    """
    fits = _all_fitted_models()
    model, X, y, weights = fits[regression_type]
    ctx = rq.build_context(model, X, y, weights=weights,
                           regression_type=regression_type)

    if regression_type in _NO_SCALE_TYPES:
        assert not ctx.standardisation_available
        assert _NO_SCALE_TYPES[regression_type] in ctx.standardisation.reason
        assert np.isnan(ctx.scale)
        assert np.all(np.isnan(ctx.std_resid))
        _panels_skip_without_a_scale(ctx, _NO_SCALE_TYPES[regression_type])
        return

    assert ctx.standardisation_available, ctx.standardisation.reason
    assert np.isfinite(ctx.scale) and ctx.scale > 0
    # The definition, restated from the parts the registry published.
    np.testing.assert_allclose(
        ctx.std_resid,
        ctx.standardisation.base / np.sqrt(ctx.scale * (1.0 - ctx.leverage)),
        rtol=1e-10, atol=1e-12)
    # ...and it is on the scale the panels' +/-2 guides assume.
    spread = float(np.std(ctx.std_resid[np.isfinite(ctx.std_resid)], ddof=1))
    assert 0.7 < spread < 1.4, f"{regression_type}: standardised spread {spread}"
    assert ctx.standardisation.source and ctx.standardisation.metric


@pytest.mark.parametrize("regression_type", [
    "ols", "wls", "rlm", "huber", "glm", "poisson", "quasi_binomial", "beta",
    "logit", "probit", "quantile", "mixed", "lasso", "ridge", "elasticnet",
    "hinge",
])
def test_every_regression_type_produces_a_report_with_no_failed_panel(
        regression_type, tmp_path):
    """No backend may crash a panel, and every panel is written or explained."""
    fits = _all_fitted_models()
    model, X, y, weights = fits[regression_type]
    meta = _wells_for(len(y), seed=201)
    manifest = rq.regression_qc_report(
        model, X, y, str(tmp_path), weights=weights, metadata=meta,
        regression_type=regression_type, verbose=False)

    assert manifest["failed"] == [], manifest["failed"]
    for panel in manifest["panels"]:
        assert panel.status in ("written", "partial", "skipped")
        if panel.status == "skipped":
            assert panel.reason and len(panel.reason) > 20, panel.name
        else:
            assert os.path.isfile(panel.path), panel.name
    # The scale that every |z| on the report depends on is disclosed on it.
    report = open(manifest["report"], encoding="utf-8").read()
    assert "standardised by  :" in report
    assert manifest["residual_scale"] in report
    if not manifest["residual_scale_available"]:
        assert manifest["residual_scale_reason"][:40] in report


@pytest.mark.parametrize("regression_type", [
    "ols", "wls", "rlm", "glm", "poisson", "beta", "logit", "quantile",
    "mixed", "lasso", "hinge",
])
def test_the_family_on_the_report_is_the_family_that_was_fitted(regression_type):
    """A caption that contradicts the fit is a wrong answer with a figure attached."""
    expected = {
        "ols": "Gaussian (least squares)",
        "wls": "Gaussian (weighted least squares)",
        "rlm": "Huber M-estimate (robust regression)",
        "glm": "Binomial",
        "poisson": "Poisson",
        "beta": "Beta",
        "logit": "Binomial",
        "quantile": "quantile regression (no error distribution)",
        "mixed": "Gaussian (linear mixed effects)",
        "lasso": "Gaussian (penalised least squares)",
        "hinge": "hinge loss (linear classifier)",
    }[regression_type]
    fits = _all_fitted_models()
    model, X, y, weights = fits[regression_type]
    ctx = rq.build_context(model, X, y, weights=weights,
                           regression_type=regression_type)
    assert ctx.family == expected
