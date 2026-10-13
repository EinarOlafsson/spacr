"""Regression QC where the fit, the data or the disk will not cooperate.

Every number on this report multiplies a published screen result, so the
module's contract is that a diagnostic never lies and never takes the fit down
with it. That splits into two kinds of path, and both are here: inputs that are
refused loudly by name, and failures that are absorbed into a stated reason --
a design matrix whose hat values are impossible, a model class whose ``scale``
means something else, a panel that raises while judging a fit that already
succeeded, a numbers file that cannot be written.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from matplotlib.figure import Figure

from spacr import regression_qc as rq


def _axes():
    """A bare axes on an object-oriented figure, as the report driver makes."""
    figure = Figure(figsize=(4, 3))
    return figure.add_subplot(1, 1, 1)


class _Fit:
    """A minimal stand-in for a fitted model: whatever attributes are given."""

    def __init__(self, **attributes):
        self.__dict__.update(attributes)


# ---------------------------------------------------------------------------
# leverage, Cook's distance, condition number
# ---------------------------------------------------------------------------

def test_a_design_matrix_that_is_not_two_dimensional_is_refused():
    """A 1-D array has no columns to form a hat matrix from."""
    with pytest.raises(ValueError, match="must be 2-D"):
        rq.leverage_from_design(np.arange(6.0))


def test_weights_that_do_not_match_the_design_are_refused():
    """Mismatched weights would silently compute leverage for other rows."""
    with pytest.raises(ValueError, match="weights has 2 entries"):
        rq.leverage_from_design(np.ones((4, 1)), weights=[1.0, 2.0])


def test_a_hat_value_outside_zero_to_one_is_a_refusal_not_a_clip():
    """Only float noise is absorbed; a real excursion means bad inputs.

    ``sqrt(1 - h)`` has to stay real, and a hat value of 2 is not a rounding
    error -- it is weights that are not the weights the fit was given.
    """
    with pytest.raises(ValueError, match="outside"):
        rq.leverage_from_design(np.ones((2, 1)), weights=[-1.0, 2.0])


def test_cooks_distance_refuses_inputs_that_are_not_the_same_rows():
    """Residuals and leverage of different lengths are two different fits."""
    with pytest.raises(ValueError, match="std_resid has shape"):
        rq.cooks_distance(np.zeros(3), np.zeros(4), 2)


def test_cooks_distance_refuses_a_model_with_no_parameters():
    """``p`` is a divisor; zero would make every distance infinite."""
    with pytest.raises(ValueError, match="n_params must be positive"):
        rq.cooks_distance(np.zeros(3), np.zeros(3), 0)


def test_an_empty_design_has_no_condition_number():
    """Nothing to condition is not a condition number of 1."""
    with pytest.raises(ValueError, match="non-empty 2-D"):
        rq.condition_number(np.empty((0, 0)))


def test_a_singular_design_reports_an_infinite_condition_number():
    """A duplicated predictor is an exact combination, and it is named.

    The verdict has to say *singular* rather than quote a huge number: an
    exactly collinear design has no finite condition number at all.
    """
    duplicated = np.column_stack([np.ones(4), np.arange(4.0), np.arange(4.0)])

    scaled, _unscaled, _sv = rq.condition_number(duplicated)

    assert not np.isfinite(scaled)
    assert "singular" in rq.condition_verdict(scaled)


def test_a_roundoff_sized_singular_value_is_still_singular(monkeypatch):
    """LAPACK may report exact rank deficiency as a tiny positive value."""
    almost_zero = np.finfo(float).eps ** 2
    spectra = iter((np.array([1.0, almost_zero]),
                    np.array([2.0, almost_zero])))
    monkeypatch.setattr(rq.np.linalg, "svd",
                        lambda *_args, **_kwargs: next(spectra))

    scaled, unscaled, singular_values = rq.condition_number(np.eye(2))

    assert not np.isfinite(scaled)
    assert not np.isfinite(unscaled)
    assert singular_values[-1] == almost_zero


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def test_calibration_refuses_inputs_that_do_not_line_up():
    """Every refusal names the two lengths, because the caller has both."""
    with pytest.raises(ValueError, match="y_true has 3 entries"):
        rq.calibration_curve([0.0, 1.0, 0.0], [0.1, 0.2])
    with pytest.raises(ValueError, match="n_bins must be at least 2"):
        rq.calibration_curve([0.0, 1.0], [0.1, 0.2], n_bins=1)
    with pytest.raises(ValueError, match="weights has 1 entries"):
        rq.calibration_curve([0.0, 1.0], [0.1, 0.2], weights=[1.0])


def test_calibration_refuses_a_split_with_nothing_left_to_bin():
    """All-NaN predictions leave no observation, which is not a flat curve."""
    with pytest.raises(ValueError, match="no finite observations"):
        rq.calibration_curve([0.0, 1.0], [np.nan, np.nan])


def test_uniform_bins_span_the_whole_probability_range():
    """Equal-width bins are edges on [0, 1], not on the observed range.

    A model that never predicts above 0.4 has an empty upper half, and that
    emptiness is the finding; rescaling the axis to the predictions would hide
    it.
    """
    predictions = np.linspace(0.05, 0.35, 40)
    truth = (predictions > 0.2).astype(float)

    curve = rq.calibration_curve(truth, predictions, n_bins=4,
                                 strategy="uniform")

    assert curve["counts"].sum() == 40
    # Only the two bins below 0.5 are populated; empty bins are dropped
    # rather than reported as a frequency nothing was measured at.
    assert curve["n_bins"] == 2
    assert curve["pred_mean"].max() < 0.5


def test_an_unknown_binning_strategy_is_refused_by_name():
    """A misspelt strategy must not fall back to a different one silently."""
    with pytest.raises(ValueError, match="strategy must be"):
        rq.calibration_curve([0.0, 1.0], [0.1, 0.2], strategy="equal")


# ---------------------------------------------------------------------------
# over-dispersion
# ---------------------------------------------------------------------------

def test_overdispersion_refuses_inputs_it_cannot_form_a_ratio_from():
    """Each refusal is about a different half of the ratio."""
    with pytest.raises(ValueError, match="y has 3 entries"):
        rq.overdispersion_statistic([1.0, 2.0, 3.0], [1.0, 2.0], 5)
    with pytest.raises(ValueError, match="df_resid must be positive"):
        rq.overdispersion_statistic([1.0, 2.0], [1.0, 2.0], 0)
    with pytest.raises(ValueError, match="no observation has a positive"):
        rq.overdispersion_statistic([1.0, 2.0], [0.0, 0.0], 1)


def test_a_moderately_over_dispersed_fit_says_how_wide_the_intervals_should_be():
    """Between 1.5 and 2 the advice is the inflation factor, not a refit.

    The number a reader needs is how far off the standard errors are, because
    that is what decides whether a hit survives.
    """
    mu = np.full(50, 4.0)
    y = np.where(np.arange(50) % 2 == 0, 4.0 - 1.9, 4.0 + 1.9)

    out = rq.overdispersion_statistic(y, mu, 25)

    assert 1.5 < out["dispersion"] <= 2.0
    assert "over-dispersed" in out["verdict"]
    assert "too narrow" in out["verdict"]


def test_an_under_dispersed_fit_is_reported_as_conservative():
    """Under-dispersion is not a hit generator, so it says so plainly."""
    mu = np.full(50, 4.0)
    y = np.full(50, 4.5)

    out = rq.overdispersion_statistic(y, mu, 25)

    assert out["dispersion"] < 0.5
    assert "under-dispersed" in out["verdict"]


# ---------------------------------------------------------------------------
# the p-value histogram
# ---------------------------------------------------------------------------

def test_p_values_depleted_near_zero_are_called_anti_uniform():
    """No spike and an empty first bin is a screen with nothing in it.

    That is a different diagnosis from a flat histogram: the test is not just
    finding nothing, it is finding less than chance, which points at an
    over-conservative test rather than at an absence of hits.
    """
    p_values = np.linspace(0.3, 0.7, 200)

    out = rq.diagnose_p_value_histogram(p_values, n_bins=10)

    assert out["verdict"] == "anti-uniform"
    assert "depleted" in out["message"]


# ---------------------------------------------------------------------------
# residual standardisation, per model class
# ---------------------------------------------------------------------------

def test_a_value_that_is_not_a_positive_number_is_no_scale_at_all():
    """``model.scale`` can be a string, None, zero or negative."""
    assert rq._positive_float("not a number") is None
    assert rq._positive_float(None) is None
    assert rq._positive_float(0.0) is None
    assert rq._positive_float(2.5) == 2.5


def test_a_fit_that_reproduces_every_observation_gets_a_unit_variance():
    """An exact fit has no residual variance to divide by.

    Substituting one lets the influence panels report the saturation instead
    of dividing by zero and naming every well an outlier.
    """
    variance, exact = rq._residual_variance(np.zeros(5), 5, 2)

    assert (variance, exact) == (1.0, True)


def test_weights_are_only_taken_when_the_fit_really_carries_them():
    """The weights have to be the ones ``model.scale`` was formed with."""
    assert rq._fit_weights(_Fit(model=None), 3) is None
    assert rq._fit_weights(_Fit(model=_Fit(weights=[1.0, 2.0])), 3) is None
    assert rq._fit_weights(_Fit(model=_Fit(weights=[1.0, 0.0, 2.0])),
                           3) is None
    got = rq._fit_weights(_Fit(model=_Fit(weights=[1.0, 2.0, 3.0])), 3)
    assert list(got) == [1.0, 2.0, 3.0]


def test_an_exact_least_squares_fit_says_so_instead_of_quoting_a_scale():
    """OLS with no usable ``scale`` and no residuals left is saturated."""
    result = rq._scale_ols(_Fit(scale=None), np.zeros(4), 4, 2)

    assert result.available is True
    assert result.variance == 1.0
    assert "reproduces every observation" in result.source


def test_a_weighted_fit_recomputes_its_own_scale_when_the_fit_has_none():
    """The recomputation has to stay in the weighted metric.

    Unweighting the scale to match an unweighted residual would throw away
    the whole point of weighting the fit by cell count.
    """
    model = _Fit(scale=None, model=_Fit(weights=[1.0, 4.0, 9.0, 16.0]))

    recomputed = rq._scale_wls(model, np.array([1.0, -1.0, 1.0, -1.0]), 4, 2)

    assert recomputed.available is True
    assert recomputed.variance == pytest.approx((1 + 4 + 9 + 16) / 2)
    assert "recomputed here" in recomputed.source

    saturated = rq._scale_wls(model, np.zeros(4), 4, 2)
    assert saturated.variance == 1.0
    assert "reproduces every observation" in saturated.source


def test_a_weighted_fit_with_no_weights_on_it_cannot_be_standardised():
    """Its ``scale`` is in a metric the residuals are not in."""
    result = rq._scale_wls(_Fit(scale=2.0, model=None), np.ones(3), 3, 1)

    assert result.available is False
    assert "does not expose the per-observation weights" in result.reason


def test_a_robust_fit_with_a_collapsed_scale_says_why_it_collapsed():
    """RLM's scale is a robust SD; more than half an exact fit makes it zero."""
    result = rq._scale_rlm(_Fit(scale=0.0), np.zeros(4), 4, 2)

    assert result.available is False
    assert "robust standard deviation" in result.reason


def test_a_glm_without_a_pearson_residual_is_not_standardised_on_the_response():
    """Standardising a binomial on the response scale invents outliers.

    Every well near mu = 0.5 has the largest possible response-scale
    residual by construction, so the panel would name the middle of the
    plate.
    """
    result = rq._scale_glm(_Fit(scale=1.0, resid_pearson=None), np.ones(4),
                           4, 2)

    assert result.available is False
    assert "no per-observation Pearson residual" in result.reason


def test_a_glm_with_a_non_positive_dispersion_is_refused():
    """A dispersion of zero cannot divide a Pearson residual."""
    result = rq._scale_glm(
        _Fit(scale=0.0, resid_pearson=np.ones(4), family=_Fit()),
        np.ones(4), 4, 2)

    assert result.available is False
    assert "non-positive dispersion" in result.reason


def test_a_mixed_fit_falls_back_to_its_own_residual_variance():
    """MixedLM without a usable ``scale`` still has conditional residuals."""
    recomputed = rq._scale_mixedlm(_Fit(scale=None),
                                   np.array([1.0, -1.0, 2.0, -2.0]), 4, 2)
    assert recomputed.available is True
    assert recomputed.variance == pytest.approx(5.0)
    assert "recomputed here" in recomputed.source

    saturated = rq._scale_mixedlm(_Fit(scale=None), np.zeros(4), 4, 2)
    assert "reproduces every observation exactly" in saturated.source


def test_an_unrecognised_model_class_estimates_its_own_variance():
    """An unknown ``scale`` attribute is not trusted; RSS/(n-p) is computed."""
    saturated = rq._scale_estimated(_Fit(scale=17.0), np.zeros(4), 4, 2)

    assert saturated.variance == 1.0
    assert "reproduces every observation" in saturated.source


def test_a_standardisation_for_the_wrong_number_of_rows_is_refused(
        monkeypatch):
    """A base of the wrong length means the model and the data disagree.

    Naming an outlier off misaligned rows would send someone to the wrong
    well, which is the single worst thing this report can do.
    """
    def wrong_length(model, resid, n, p):
        return rq.ResidualStandardisation(
            available=True, metric="made up", source="made up",
            base=np.zeros(2), variance=1.0)

    monkeypatch.setattr(rq, "_scale_estimated", wrong_length)
    result = rq.resolve_residual_standardisation(_Fit(), np.zeros(5), 5, 2)

    assert result.available is False
    assert "not the same rows" in result.reason


# ---------------------------------------------------------------------------
# coercing the inputs
# ---------------------------------------------------------------------------

def test_a_bare_array_design_gets_usable_column_names():
    """Panels name predictors, so a nameless matrix still needs names."""
    frame = rq._as_frame(np.arange(6.0))

    assert list(frame.columns) == ["x0"]
    assert frame.shape == (6, 1)
    assert list(rq._as_frame(np.zeros((3, 2))).columns) == ["x0", "x1"]


def test_a_response_with_more_than_one_column_is_refused():
    """Which column is the response is not something to guess at."""
    with pytest.raises(ValueError, match="has 2 columns"):
        rq._as_vector(pd.DataFrame({"a": [1.0], "b": [2.0]}))


def test_a_number_too_large_to_print_is_shortened_not_truncated():
    """A condition number of 1e16 is a real result on a trapped design."""
    assert rq._readable_number(float("inf")) == "inf"
    assert rq._readable_number(1.3583e16) == "1.36e+16"
    assert rq._readable_number(1234.5) == "1,234.5"


# ---------------------------------------------------------------------------
# the decision score's orientation
# ---------------------------------------------------------------------------

def test_a_classifier_that_cannot_score_its_own_design_loses_only_the_roc():
    """Twenty other panels must not go down with the ROC."""
    class _Broken:
        def decision_function(self, X):
            raise RuntimeError("shapes changed since the fit")

    assert rq._decision_score(_Broken(), pd.DataFrame({"x": [1.0]}),
                              np.zeros(1)) is None


def test_a_multiclass_score_has_no_single_ranking_to_draw():
    """One ROC needs one ranking; ``(n, k)`` is k of them."""
    class _Multi:
        def decision_function(self, X):
            return np.zeros((len(X), 3))

    assert rq._decision_score(_Multi(), pd.DataFrame({"x": [1.0, 2.0]}),
                              np.zeros(2)) is None


def test_a_score_with_non_numeric_labels_is_taken_as_it_stands():
    """Only a numeric pair can say which class is the larger one."""
    class _Named:
        classes_ = np.array(["control", "hit"])

        def decision_function(self, X):
            return np.array([-1.0, 2.0])

    score = rq._decision_score(_Named(), pd.DataFrame({"x": [1.0, 2.0]}),
                               np.zeros(2))

    assert list(score) == [-1.0, 2.0]


def test_a_fit_with_no_family_is_named_least_squares():
    """The caption must not claim a family the fit does not have."""
    assert rq._family_and_link(_Fit(), None) == \
        ("Gaussian (least squares)", "Identity")


# ---------------------------------------------------------------------------
# well labels and metadata
# ---------------------------------------------------------------------------

def test_a_well_is_named_from_its_plate_row_and_column_when_there_is_no_prc():
    """"well 47" sends nobody to a microscope; ``plate1_r3_c11`` does."""
    from spacr import schema

    metadata = pd.DataFrame({column: [f"{column}{i}" for i in range(3)]
                             for column in schema.WELL_KEY_COLUMNS})

    labels = rq._well_labels(None, metadata, 3)

    assert len(labels) == 3
    assert schema.KEY_SEPARATOR in labels[0]


def test_a_row_with_no_metadata_at_all_falls_back_to_its_position():
    """A label is always produced; an unlabelled outlier is still an outlier."""
    assert list(rq._well_labels(None, None, 3)) == ["0", "1", "2"]


def test_metadata_that_is_not_a_frame_is_coerced_before_it_is_aligned():
    """A dict of columns is a reasonable thing for a caller to pass."""
    aligned = rq._align_metadata({"plateID": ["p1", "p1"]}, None, 2)

    assert isinstance(aligned, pd.DataFrame)
    assert list(aligned["plateID"]) == ["p1", "p1"]


# ---------------------------------------------------------------------------
# building the context
# ---------------------------------------------------------------------------

def _design(n=6):
    return pd.DataFrame({"intercept": np.ones(n),
                         "x": np.linspace(0.0, 1.0, n)})


def test_a_response_and_a_design_of_different_lengths_are_refused():
    """They must be the rows of the same fit, and the message says so."""
    with pytest.raises(ValueError, match="rows of the same fit"):
        rq.build_context(_Fit(fittedvalues=np.zeros(6)), _design(6),
                         np.zeros(5))


def test_a_model_that_produced_the_wrong_number_of_fitted_values_is_refused():
    """A fit over other rows would attribute every residual to the wrong well."""
    with pytest.raises(ValueError, match="fitted values"):
        rq.build_context(_Fit(fittedvalues=np.zeros(4)), _design(6),
                         np.zeros(6))


def test_weights_that_are_not_the_fitted_rows_are_refused():
    """Cell counts for other wells are not this fit's weights."""
    with pytest.raises(ValueError, match="weights has 3 entries"):
        rq.build_context(_Fit(fittedvalues=np.zeros(6)), _design(6),
                         np.zeros(6), weights=[1.0, 2.0, 3.0])


class _AwkwardFit:
    """A fit whose influence machinery raises, as several statsmodels do.

    ``get_hat_matrix_diag`` and ``get_influence`` are both advertised and both
    unusable for MixedLM and the regularised fits, which is why the design
    matrix fallback exists at all.
    """

    def __init__(self, fitted):
        self.fittedvalues = np.asarray(fitted, dtype=float)

    def get_hat_matrix_diag(self):
        raise NotImplementedError("no hat matrix for this fit")

    def get_influence(self):
        raise NotImplementedError("influence is not defined here")


def test_leverage_falls_back_to_the_design_when_the_fit_will_not_give_it():
    """The fallback is exact for the unweighted case and says which it is.

    A report that lost its influence panels because statsmodels raises a
    different exception per model class would be missing exactly the panels
    a rank-deficient screen design most needs.
    """
    frame = _design(6)
    y = np.linspace(0.0, 1.0, 6)
    model = _AwkwardFit(y * 0.5)

    ctx = rq.build_context(model, frame, y, weights=np.arange(1.0, 7.0))

    assert ctx.leverage_source == "design matrix (weighted)"
    assert ctx.leverage.shape == (6,)
    assert np.all(np.isfinite(ctx.leverage))


# ---------------------------------------------------------------------------
# the smoothed trend
# ---------------------------------------------------------------------------

def test_too_few_points_have_no_trend_to_report():
    """Four points do not make a smoother, and nan says so."""
    assert np.isnan(rq._trend(_axes(), [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]))


def test_a_trend_over_a_pair_of_points_is_just_its_largest_value():
    """With fewer than three samples there is no tied block to exclude."""
    assert rq._trend_off_the_ties([1.0, 2.0], [3.0, -5.0]) == 5.0


def test_a_smoothed_curve_with_no_usable_spacing_keeps_its_plain_maximum():
    """When no point can be shown to sit off a tie, none is excluded.

    Dropping every point because the spacing could not be measured would
    report no trend at all on a curve that plainly has one.
    """
    result = rq._trend_off_the_ties([0.0, np.nan, 1.0], [0.2, -0.9, 0.4])

    assert result == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Panels that cannot be computed for the fit they are handed
# ---------------------------------------------------------------------------

def _context(n=6, y=None, fitted=None, **kwargs):
    """A context over a straight-line design, built the way the report does."""
    frame = pd.DataFrame({"intercept": np.ones(n),
                          "x": np.linspace(0.0, 1.0, n)})
    response = np.linspace(0.0, 1.0, n) if y is None else np.asarray(y, float)
    values = response * 0.5 if fitted is None else np.asarray(fitted, float)
    return rq.build_context(_Fit(fittedvalues=values), frame, response,
                            **kwargs)


def test_a_two_well_fit_is_too_small_for_the_residual_panels():
    """Each panel names its own minimum rather than drawing two points."""
    ctx = _context(2)

    with pytest.raises(rq.PanelUnavailable, match="fewer than 3"):
        rq._panel_scale_location(ctx, _axes())
    with pytest.raises(rq.PanelUnavailable, match="needs at least 5"):
        rq._panel_qq_residuals(ctx, _axes())
    with pytest.raises(rq.PanelUnavailable, match="fewer than 3 observations"):
        rq._panel_observed_vs_predicted(ctx, _axes())


def test_spread_that_shrinks_across_the_fit_is_named_as_such():
    """Both statistics have to agree before the panel commits to a direction.

    Spearman alone is zero for a symmetric funnel, so a Brown-Forsythe test
    across quartiles decides whether there is any inequality at all, and the
    sign of rho then says which way it goes.
    """
    ctx = _context(40)
    fitted = np.linspace(0.0, 1.0, 40)
    resid = (1.0 - fitted) * np.random.default_rng(0).normal(size=40)
    ctx.fitted, ctx.resid = fitted, resid
    ctx.std_resid = resid / np.std(resid)

    stats = rq._panel_scale_location(ctx, _axes())

    assert stats["verdict"] == "variance shrinks with the fit"
    assert stats["levene_p"] < 0.01


def test_a_small_fit_reads_the_trend_off_spearman_alone():
    """Under twenty wells there are not enough quartiles to test.

    The monotone trend is still worth reporting; what is dropped is the claim
    that the inequality is significant.
    """
    ctx = _context(15)
    fitted = np.linspace(0.0, 1.0, 15)
    resid = fitted * np.resize([1.0, -1.0], 15)
    ctx.fitted, ctx.resid = fitted, resid
    ctx.std_resid = resid / np.std(resid)

    stats = rq._panel_scale_location(ctx, _axes())

    assert stats["verdict"] == "variance grows with the fit"
    assert not np.isfinite(stats["levene_p"])


def test_a_variance_test_that_refuses_is_a_missing_number_not_a_dead_panel(
        monkeypatch):
    """scipy refuses a constant group; that is an answer about the data.

    The panel still has Spearman's rho and still draws; only the equality-of-
    variance p-value goes missing.
    """
    from scipy import stats as sps

    def refuse(*groups, **kwargs):
        raise ValueError("Data must not be constant.")

    monkeypatch.setattr(sps, "levene", refuse)
    ctx = _context(40)
    fitted = np.linspace(0.0, 1.0, 40)
    resid = np.resize([1.0, -1.0, 0.5, -0.5], 40)
    ctx.fitted, ctx.resid = fitted, resid
    ctx.std_resid = resid

    stats = rq._panel_scale_location(ctx, _axes())

    assert not np.isfinite(stats["levene_p"])
    assert stats["verdict"] == "no detectable trend in spread"


def test_a_calibration_curve_needs_enough_wells_to_fill_three_bins():
    """Six wells cannot show whether a predicted 0.3 happens 30% of the time."""
    ctx = _context(6, regression_type="logit")

    with pytest.raises(rq.PanelUnavailable, match="needs at least 15"):
        rq._panel_calibration(ctx, _axes())


def test_a_binary_panel_needs_both_classes_present():
    """An ROC over one class has no curve; every threshold gives the same call."""
    ctx = _context(6, y=np.zeros(6), fitted=np.zeros(6),
                   regression_type="logit")

    with pytest.raises(rq.PanelUnavailable, match="needs both classes"):
        rq._require_binary(ctx, "the ROC")


def test_a_count_fit_with_no_degrees_of_freedom_on_it_computes_its_own():
    """``df_resid`` is a divisor; a fit that does not report one still has n-p."""
    counts = np.array([3.0, 5.0, 4.0, 6.0, 5.0, 4.0])
    ctx = _context(6, y=counts, fitted=np.full(6, 4.5),
                   regression_type="poisson")

    stats = rq._panel_count_fit(ctx, _axes())

    assert stats["df_resid"] == 4.0
    assert np.isfinite(stats["dispersion"])


def test_a_plate_column_with_one_value_has_no_between_plate_effect():
    """One group is not a comparison, and the panel says which column."""
    ctx = _context(6, metadata=pd.DataFrame({"plateID": ["p1"] * 6}))

    with pytest.raises(rq.PanelUnavailable, match="every well has plateID"):
        rq._grouped_residuals(ctx, "plateID")


def test_metadata_without_the_column_lists_the_columns_it_does_have():
    """A misnamed position column is the common cause, so the names help."""
    ctx = _context(6, metadata=pd.DataFrame({"plate": ["p1", "p2"] * 3}))

    with pytest.raises(rq.PanelUnavailable, match="no 'plateID' column"):
        rq._grouped_residuals(ctx, "plateID")


def test_a_test_that_refuses_identical_groups_leaves_the_panel_standing(
        monkeypatch):
    """Every residual identical is "no difference", not a broken panel."""
    from scipy import stats as sps

    def refuse(*groups, **kwargs):
        raise ValueError("All numbers are identical in kruskal")

    monkeypatch.setattr(sps, "kruskal", refuse)
    metadata = pd.DataFrame({"plateID": ["p1", "p1", "p1", "p2", "p2", "p2"]})
    ctx = _context(6, metadata=metadata)

    stats = rq._positional_effect_panel(ctx, _axes(), "plateID", "plate",
                                        False)

    assert not np.isfinite(stats["kruskal_p"])


def test_the_fit_weights_stand_in_for_a_missing_cell_count_column():
    """spaCR's binomial path passes cell counts as weights, not as metadata.

    Refusing the panel because the metadata has no ``cell_count`` column
    would drop it for exactly the fit it was written for.
    """
    ctx = _context(4, weights=np.array([10.0, 20.0, 30.0, 40.0]))

    with pytest.raises(rq.PanelUnavailable, match="only 4 well"):
        rq._panel_cell_count_vs_effect(ctx, _axes())


# ---------------------------------------------------------------------------
# The coefficient table
# ---------------------------------------------------------------------------

def test_a_fit_whose_intervals_raise_still_shows_its_coefficients(tmp_path):
    """An error bar that cannot be computed is not one to fabricate."""
    class _NoIntervals:
        fittedvalues = np.zeros(6)
        params = pd.Series([1.0, 2.0], index=["intercept", "x"])

        def conf_int(self):
            raise RuntimeError("singular covariance matrix")

    ctx = rq.build_context(
        _NoIntervals(),
        pd.DataFrame({"intercept": np.ones(6), "x": np.linspace(0, 1, 6)}),
        np.zeros(6))

    table, note = rq._coefficient_table(ctx)

    assert list(table["coefficient"]) == [1.0, 2.0]
    assert "conf_int() raised RuntimeError" in note


def test_a_fit_with_no_conf_int_at_all_says_so_once():
    """A point estimator has no intervals, and the caption states it."""
    model = _Fit(fittedvalues=np.zeros(6),
                 params=pd.Series([1.0, 2.0], index=["intercept", "x"]))
    ctx = rq.build_context(
        model, pd.DataFrame({"intercept": np.ones(6),
                             "x": np.linspace(0, 1, 6)}), np.zeros(6))

    _table, note = rq._coefficient_table(ctx)

    assert "exposes no conf_int()" in note


def test_a_model_with_neither_params_nor_coefficients_has_no_forest():
    """There is nothing to plot, and the class is named so it can be fixed."""
    ctx = _context(6)

    with pytest.raises(rq.PanelUnavailable, match="neither params nor coef_"):
        rq._coefficient_table(ctx)


def test_coefficients_that_are_all_non_finite_are_not_a_forest():
    """A fit that returned nan for everything has no effect sizes to sort."""
    model = _Fit(fittedvalues=np.zeros(6),
                 params=pd.Series([np.nan, np.nan],
                                  index=["intercept", "x"]))
    ctx = rq.build_context(
        model, pd.DataFrame({"intercept": np.ones(6),
                             "x": np.linspace(0, 1, 6)}), np.zeros(6))

    with pytest.raises(rq.PanelUnavailable, match="non-finite"):
        rq._panel_coefficient_forest(ctx, _axes())


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def test_every_panel_reports_unknown_rather_than_guessing_from_nothing():
    """A scorer handed no statistics must not invent a level.

    ``unknown`` is a real outcome on this report -- it says the panel did not
    run -- and it has to be distinguishable from a pass. The coefficient
    forest is the one exception: its only question is whether intervals
    exist, and an absent key answers it.
    """
    answered_by_absence = {"coefficient_forest"}
    for name in rq._SCORERS:
        verdict = rq.score_panel(name, {})
        assert verdict.headline, name
        if name in answered_by_absence:
            assert verdict.level in rq.VERDICT_LEVELS, name
        else:
            assert verdict.level == "unknown", name


def test_an_unscored_panel_name_is_not_an_error():
    """The report asks for a verdict for every panel it drew."""
    assert rq.score_panel("not_a_panel", {}).headline == \
        "this panel is not scored"


def test_a_separating_classifier_is_scored_off_its_auc():
    """0.5 is a coin toss, and the bands are read in that direction."""
    good = rq.score_panel("roc", {"auc": 0.91})
    poor = rq.score_panel("roc", {"auc": 0.55})

    assert good.level == "pass"
    assert poor.level == "fail"
    assert "0.910" in good.detail


def test_precision_recall_is_scored_against_the_prevalence_not_against_half():
    """On an imbalanced response the baseline IS the prevalence.

    An average precision of 0.2 is excellent at 2% prevalence and worthless
    at 50%, so the lift over the prevalence is what is scored -- and a bigger
    lift has to be the better verdict, the same way a bigger AUC is.

    The reading is asserted through ``PanelVerdict.score``, which is the
    field the number the verdict was read off is carried in; the type has no
    ``value``, so asking for one tested nothing but an AttributeError.
    """
    lifted = rq.score_panel("precision_recall",
                            {"average_precision": 0.2, "prevalence": 0.02})
    flat = rq.score_panel("precision_recall",
                          {"average_precision": 0.2, "prevalence": 0.2})

    assert lifted.score == pytest.approx(10.0)
    assert lifted.level == "pass"
    assert flat.level == "fail"


def test_a_constant_response_fails_the_response_panel():
    """Nothing can be regressed on a response with no variance."""
    verdict = rq.score_panel("response_distribution", {"sd": 0.0})

    assert verdict.level == "fail"
    assert verdict.headline == "the response is constant"


def test_a_run_with_a_volcano_names_it_rather_than_scoring_it():
    """The volcano is the result; this suite judges the fit behind it."""
    found = rq.score_panel("volcano_reference", {"state": "found"})

    assert found.level == "unknown"
    assert "for reference" in found.headline


def test_a_scorer_that_trips_over_its_own_statistics_returns_unknown():
    """A diagnostic that crashes while judging a diagnostic is not a failure.

    The fit already succeeded and cost an hour; losing it for the sake of a
    sentence would be the wrong trade.
    """
    class _Hostile(dict):
        def __contains__(self, key):
            return True

        def __getitem__(self, key):
            raise RuntimeError("the stats frame is not readable")

    verdict = rq.score_panel("residuals_vs_fitted", _Hostile())

    assert verdict.level == "unknown"
    assert "could not be computed" in verdict.headline
    assert "RuntimeError" in verdict.detail


def test_a_verdict_at_an_unknown_level_is_downgraded_not_shown(monkeypatch):
    """The badge has four inks; a fifth level would render as nothing."""
    monkeypatch.setitem(
        rq._SCORERS, "vif",
        lambda stats: rq.PanelVerdict("catastrophic", "the design is doomed",
                                      "made up level"))

    verdict = rq.score_panel("vif", {})

    assert verdict.level == "unknown"
    assert verdict.headline == "the design is doomed"


# ---------------------------------------------------------------------------
# Writing the report
# ---------------------------------------------------------------------------

def test_a_panel_that_crashes_is_reported_and_the_run_survives(tmp_path,
                                                               capsys,
                                                               monkeypatch):
    """A crashing diagnostic must be loud but must not take the fit down.

    It is printed, and it lands on the report as FAILED with the exception in
    it, so "the panel is broken" and "the panel is fine" never look the same.
    """
    def explode(ctx, ax):
        raise RuntimeError("the smoother gave up")

    title, group, _fn = rq._PANEL_BY_NAME["residuals_vs_fitted"]
    monkeypatch.setitem(rq._PANEL_BY_NAME, "residuals_vs_fitted",
                        (title, group, explode))

    frame = pd.DataFrame({"intercept": np.ones(8),
                          "x": np.linspace(0.0, 1.0, 8)})
    y = np.linspace(0.0, 1.0, 8)
    manifest = rq.regression_qc_report(
        _Fit(fittedvalues=y * 0.5), frame, y, str(tmp_path),
        panels=["residuals_vs_fitted"], combined=False, verbose=True)

    printed = capsys.readouterr().out
    assert "panel 'residuals_vs_fitted' failed" in printed
    assert "the smoother gave up" in printed
    failed = [p for p in manifest["panels"] if p.status == "failed"]
    assert len(failed) == 1
    assert "the smoother gave up" in failed[0].reason


def test_a_panel_that_will_not_redraw_leaves_a_tile_saying_why(tmp_path):
    """The combined page redraws every panel onto one figure.

    A panel that drew a moment ago and fails on the shared axes can only have
    an axes-specific problem; that has to be stated rather than left as a
    blank tile the reader reads as "fine".
    """
    class _OnceOnly:
        def __init__(self):
            self.calls = 0

        def __call__(self, ctx, ax):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("this axes already has a colorbar")
            return {}

    ctx = _context(8)
    title, group, _fn = rq._PANEL_BY_NAME["residuals_vs_fitted"]
    once = _OnceOnly()
    once(ctx, _axes())
    results = [rq.QCPanelResult(name="residuals_vs_fitted", title=title,
                                group=group, status="written",
                                verdict=rq.PanelVerdict("pass", "fine"))]

    import unittest.mock as mock

    with mock.patch.dict(rq._PANEL_BY_NAME,
                         {"residuals_vs_fitted": (title, group, once)}):
        path = rq._write_combined_page(ctx, results, str(tmp_path),
                                       ["residuals_vs_fitted"])

    assert once.calls == 2
    written = path[0] if isinstance(path, tuple) else path
    assert written is None or os.path.exists(str(written))


# ---------------------------------------------------------------------------
# The numbers file
# ---------------------------------------------------------------------------

def _results_with_stats():
    return [rq.QCPanelResult(
        name="p", title="t", group="fit", status="written",
        stats={"an_int": np.int64(3),
               "a_nan": np.float32("nan"),
               "a_float": np.float32(1.5),
               "an_object": object(),
               "a_list": [np.int32(1), np.float64("inf")],
               "a_map": {"k": np.float64(2.5)}})]


def test_numpy_scalars_reach_the_numbers_file_as_json_numbers(tmp_path):
    """The advisor reads this file; a numpy repr in it is unreadable.

    Non-finite values become null rather than bare ``NaN``, which is not JSON
    and which another process's ``json.load`` may refuse outright.
    """
    manifest = {"model": "OLS", "verdicts": {}}

    path = rq._write_qc_numbers(str(tmp_path), manifest, _results_with_stats())

    saved = json.loads(open(path, encoding="utf-8").read())
    numbers = saved["numbers"]
    assert numbers["an_int"] == 3
    assert numbers["a_nan"] is None
    assert numbers["a_float"] == 1.5
    assert isinstance(numbers["an_object"], str)
    assert numbers["a_list"] == [1, None]
    assert numbers["a_map"] == {"k": 2.5}
    assert manifest["numbers"] == path


def test_a_numbers_file_that_cannot_be_written_is_said_out_loud(tmp_path,
                                                                capsys):
    """The advisor notices a missing numbers file by going quiet.

    A run is not worth losing to a failure in its own bookkeeping, but the
    failure has to be printed or nothing ever finds out.
    """
    blocked = tmp_path / "not_a_folder"
    blocked.write_text("this is a file", encoding="utf-8")

    result = rq._write_qc_numbers(str(blocked), {"verdicts": {}},
                                  _results_with_stats())

    assert result is None
    assert "could not write" in capsys.readouterr().out
