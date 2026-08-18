"""Every regression backend, on data whose answer is known before the fit.

``tests/test_regression_types.py`` already drives each ``regression_type``
through ``perform_regression`` and requires the planted gene to come out on
top.  That catches a backend that does not run and a backend that ranks
backwards.  It does not catch a backend that runs, ranks correctly and reports
a coefficient that is not an effect on anything -- which is what a Poisson fit
with no exposure offset does, and what a penalty that is never actually applied
does.

So every claim here is made against a planted coefficient with a known
MAGNITUDE and a known SIGN, on a design built for the link the model uses:

* :func:`test_signal_is_recovered_with_the_right_magnitude_and_sign` plants
  ``+1.2`` on one covariate, ``-0.8`` on a second and ``0`` on a third, and
  requires all three back within tolerance.  A model that recovers the ranking
  but not the scale fails here; so does one whose sign convention is inverted,
  twice over.
* :func:`test_a_null_design_gives_a_null_answer` removes every effect and
  requires the coefficients to collapse and the p-values to stop being
  significant.  This is the test that catches a fit which reports structure in
  noise -- the one failure mode that looks exactly like a discovery.
* :func:`test_classifier_orientation_is_not_inverted` and
  :func:`test_the_inverted_score_scores_one_minus_auroc` pin the classifier's
  direction from both ends, so a suite that would pass at both AUROC 0.9 and
  AUROC 0.1 cannot exist here.  R's ``yardstick`` treats the FIRST factor level
  as the event; a score built with that convention and read with sklearn's
  yields ``1 - AUROC`` and looks entirely plausible.
* :func:`test_every_setting_a_type_declares_actually_changes_the_fit` walks
  ``REGRESSION_SETTINGS_USED`` and requires each declared setting to move the
  numbers.  ``regression_model`` refuses a setting a backend cannot read; this
  is the other half -- a setting it claims to read and quietly does not.

The pattern is :mod:`tests.test_power_model`'s, applied to the ordinary
backends.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.stats as st

import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# One design, several response scales
# ---------------------------------------------------------------------------

#: The planted truth.  Three covariates: one clearly positive, one clearly
#: negative, one inert.  Two signs rather than one because a globally inverted
#: convention swaps them and a partially inverted one does not.
B_POS, B_NEG, B_NULL = 1.2, -0.8, 0.0

N_WELLS = 600


def design(seed=0, n=N_WELLS):
    """Return ``(X, x_pos, x_neg, x_null)`` -- a centred, well-conditioned design.

    Centred covariates so the intercept keeps its meaning under every link, and
    independent columns so a recovered coefficient is that covariate's effect
    rather than a share of another's.
    """
    rng = np.random.default_rng(seed)
    x_pos = rng.uniform(-1.0, 1.0, n)
    x_neg = rng.uniform(-1.0, 1.0, n)
    x_null = rng.uniform(-1.0, 1.0, n)
    X = pd.DataFrame({"Intercept": 1.0, "x_pos": x_pos, "x_neg": x_neg,
                      "x_null": x_null})
    return X, x_pos, x_neg, x_null


def linear_predictor(x_pos, x_neg, intercept=0.0, scale=1.0, null=False):
    """``intercept + scale * (B_POS * x_pos + B_NEG * x_neg)``, or a flat line."""
    if null:
        return np.full_like(x_pos, intercept)
    return intercept + scale * (B_POS * x_pos + B_NEG * x_neg)


def make_response(kind, seed=0, null=False, n=N_WELLS):
    """Build the response for one backend, on the scale that backend fits.

    :param kind: ``'gaussian'``, ``'fraction_logit'``, ``'fraction_probit'``,
        ``'beta'``, ``'count'`` or ``'binary'``.
    :param null: When True the planted coefficients are all zero and everything
        else -- sample size, noise, well sizes, prevalence -- is unchanged, so
        the only difference between the two calls is the thing being tested.
    :returns: ``(X, y, extras)`` where ``extras`` carries the per-well weights
        or exposure the backend needs, and the truth on this scale.
    """
    X, x_pos, x_neg, _ = design(seed=seed, n=n)
    rng = np.random.default_rng(seed + 1000)
    truth = {"x_pos": 0.0 if null else B_POS, "x_neg": 0.0 if null else B_NEG,
             "x_null": 0.0}
    extras = {"truth": truth}

    if kind == "gaussian":
        eta = linear_predictor(x_pos, x_neg, intercept=0.5, null=null)
        y = pd.Series(eta + rng.normal(0, 0.35, n))

    elif kind in ("fraction_logit", "fraction_probit"):
        eta = linear_predictor(x_pos, x_neg, intercept=0.0, null=null)
        link = st.norm.cdf if kind == "fraction_probit" else \
            (lambda v: 1.0 / (1.0 + np.exp(-v)))
        p = link(eta)
        cells = rng.integers(80, 400, n).astype(float)
        y = pd.Series(rng.binomial(cells.astype(int), p) / cells)
        extras["weights"] = cells

    elif kind == "beta":
        eta = linear_predictor(x_pos, x_neg, intercept=0.0, null=null)
        mu = 1.0 / (1.0 + np.exp(-eta))
        phi = 60.0
        y = pd.Series(np.clip(rng.beta(mu * phi, (1 - mu) * phi), 1e-4,
                              1 - 1e-4))

    elif kind == "count":
        eta = linear_predictor(x_pos, x_neg, intercept=-2.0, null=null)
        # Well size varies ten-fold, exactly as it does on a real plate. It is
        # deliberately CORRELATED with x_null, which has no effect on the rate:
        # a fit with no exposure offset has to explain the well's headcount
        # with the covariates, and x_null is the one that can.
        _, _, _, x_null_col = design(seed=seed, n=n)
        n_total = np.round(60.0 + 900.0 * (x_null_col + 1.0) / 2.0)
        y = pd.Series(rng.poisson(n_total * np.exp(eta)).astype(float))
        extras["exposure"] = n_total
        extras["weights"] = n_total

    elif kind == "binary":
        eta = linear_predictor(x_pos, x_neg, intercept=0.0, scale=2.0,
                               null=null)
        p = 1.0 / (1.0 + np.exp(-eta))
        y = pd.Series((rng.uniform(size=n) < p).astype(float))

    else:                                              # pragma: no cover
        raise AssertionError(f"unknown response kind {kind!r}")

    return X, y, extras


#: ``(regression_type, response kind, kwargs, relative tolerance)``.
#:
#: The tolerance is on the recovered coefficient against the planted one. It is
#: loose (25-40%) on purpose: this is not a test of statistical efficiency, it
#: is a test that the number reported is an effect on the response at all. A
#: backend that reports something else misses by far more than 40%.
RECOVERY = [
    ("ols",            "gaussian",         {}, 0.10),
    ("wls",            "gaussian",         {}, 0.10),
    ("rlm",            "gaussian",         {}, 0.10),
    ("huber",          "gaussian",         {}, 0.10),
    ("glm",            "gaussian",         {}, 0.15),
    ("quantile",       "gaussian",         {}, 0.15),
    ("ridge",          "gaussian",         {"alpha": 0.01}, 0.15),
    ("lasso",          "gaussian",         {"alpha": 0.01}, 0.15),
    ("elasticnet",     "gaussian",         {"alpha": 0.01, "l1_ratio": 0.5}, 0.20),
    ("logit",          "fraction_logit",   {}, 0.15),
    ("probit",         "fraction_probit",  {}, 0.15),
    ("quasi_binomial", "fraction_logit",   {}, 0.15),
    ("beta",           "beta",             {}, 0.20),
    ("poisson",        "count",            {}, 0.15),
]


def _fit(regression_type, kind, seed=0, null=False, **over):
    """Fit one backend on a freshly generated response and return everything."""
    from spacr.ml import regression_model

    X, y, extras = make_response(kind, seed=seed, null=null)
    kwargs = dict(over)
    if regression_type == "wls":
        kwargs.setdefault("weights", extras.get("weights",
                                                np.ones(len(y)) * 100.0))
    if regression_type in ("logit", "probit", "quasi_binomial"):
        kwargs.setdefault("weights", extras["weights"])
    if regression_type in ("poisson", "glm", "horseshoe"):
        kwargs.setdefault("exposure", extras.get("exposure"))
    model = regression_model(X, y, regression_type=regression_type, **kwargs)
    return model, X, y, extras


def coefficients(model, X):
    """Coefficients as a name -> value dict, for statsmodels or sklearn."""
    params = getattr(model, "params", None)
    if params is not None:
        return {str(k): float(v) for k, v in params.items()}
    return {str(k): float(v)
            for k, v in zip(X.columns, np.asarray(model.coef_).ravel())}


# ---------------------------------------------------------------------------
# Signal: the planted coefficient comes back, with its size and its sign
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("regression_type", "kind", "kwargs", "tol"), RECOVERY,
                         ids=[c[0] for c in RECOVERY])
def test_signal_is_recovered_with_the_right_magnitude_and_sign(
        regression_type, kind, kwargs, tol):
    """+1.2, -0.8 and 0.0 come back as +1.2, -0.8 and 0.0.

    Both non-zero signs are asserted, so neither a global sign flip nor a
    per-link one can pass: a global flip turns +1.2 into -1.2 and -0.8 into
    +0.8, and each of the two assertions catches it on its own.
    """
    model, X, y, extras = _fit(regression_type, kind, **kwargs)
    coefs = coefficients(model, X)

    for name, expected in extras["truth"].items():
        got = coefs[name]
        assert np.isfinite(got), f"{regression_type}: {name} came back {got}"
        if expected == 0.0:
            assert abs(got) < 0.25, (
                f"{regression_type}: the inert covariate {name} came back at "
                f"{got:.4f}; nothing in the data drives it")
            continue
        assert np.sign(got) == np.sign(expected), (
            f"{regression_type}: {name} was planted at {expected:+.2f} and "
            f"came back {got:+.4f} -- the SIGN is wrong, which is the failure "
            f"that still ranks and still plots")
        assert abs(got - expected) <= tol * abs(expected) + 0.05, (
            f"{regression_type}: {name} planted at {expected:+.2f}, recovered "
            f"{got:+.4f} (tolerance {tol:.0%})")


@pytest.mark.parametrize(("regression_type", "kind", "kwargs", "tol"), RECOVERY,
                         ids=[c[0] for c in RECOVERY])
def test_a_null_design_gives_a_null_answer(regression_type, kind, kwargs, tol):
    """No effect planted, no effect reported.

    Everything about the null run matches the signal run except the planted
    coefficients: same n, same noise, same well sizes, same prevalence. So a
    backend that reports an effect here is reporting its own noise, and that is
    indistinguishable from a discovery on real data.
    """
    model, X, y, extras = _fit(regression_type, kind, null=True, **kwargs)
    coefs = coefficients(model, X)

    for name in ("x_pos", "x_neg", "x_null"):
        assert abs(coefs[name]) < 0.25, (
            f"{regression_type}: {name} came back at {coefs[name]:+.4f} on a "
            f"design with no effect in it")

    pvalues = getattr(model, "pvalues", None)
    if pvalues is not None:
        for name in ("x_pos", "x_neg", "x_null"):
            assert float(pvalues[name]) > 0.01, (
                f"{regression_type}: {name} is significant (p = "
                f"{float(pvalues[name]):.2g}) on a null design")


def test_ridge_p_values_err_conservatively_not_the_other_way():
    """Ridge's p-values are mis-specified; the direction of the error matters.

    ``calculate_p_values`` divides a SHRUNK coefficient by an UNPENALISED
    standard error, so the statistic is too small and the p-value too large.
    That is the safe direction -- it costs power. The version of this bug that
    would matter is the reverse, and the null design is where it would show:
    forty inert features and a response made of noise must not produce a pile
    of significant coefficients.
    """
    from spacr.ml import process_model_coefficients, regression_model

    rng = np.random.default_rng(21)
    n, p = 300, 40
    X = pd.DataFrame(rng.normal(0, 1, (n, p)),
                     columns=[f"f{i}" for i in range(p)])
    X.insert(0, "Intercept", 1.0)
    y = pd.Series(rng.normal(0, 1, n))

    model = regression_model(X, y, regression_type="ridge", alpha=1.0)
    coefs = process_model_coefficients(model, "ridge", X, y, "nc", "pc", [])
    n_significant = int((coefs["p_value"] <= 0.05).sum())

    # 5% of 41 is ~2. Anything near that or below is the conservative side;
    # a mis-signed or mis-scaled statistic lights up most of the table.
    assert n_significant <= 4, (
        f"{n_significant} of {len(coefs)} coefficients are significant on pure "
        f"noise; ridge's p-values have become anticonservative")


def test_lasso_reports_a_null_screen_as_a_null_screen_not_as_a_penalty_bug():
    """alpha='auto' on noise cross-validates to the empty model, and says so.

    The old message told this user to "set alpha to 'auto' to choose it by
    cross-validation", which is what they had already done. Cross-validation
    choosing the empty model is not a misconfiguration; it is the answer.
    """
    from spacr.ml import regression_model

    X, y, _ = make_response("gaussian", null=True)
    with pytest.raises(ValueError, match="cross-validated its way to the empty"):
        regression_model(X, y, regression_type="lasso", alpha="auto")


# ---------------------------------------------------------------------------
# The mixed model, which needs a grouping to be a mixed model at all
# ---------------------------------------------------------------------------

def _grouped(seed=0, null=False, n_groups=8, per_group=90):
    """A design with a real between-group intercept shift to be modelled."""
    rng = np.random.default_rng(seed)
    n = n_groups * per_group
    X, x_pos, x_neg, _ = design(seed=seed, n=n)
    groups = np.repeat(np.arange(n_groups), per_group)
    shifts = rng.normal(0, 0.8, n_groups)              # the batch effect
    eta = linear_predictor(x_pos, x_neg, intercept=0.5, null=null)
    y = pd.Series(eta + shifts[groups] + rng.normal(0, 0.35, n))
    return X, y, pd.Series(groups)


def test_mixed_recovers_the_fixed_effect_over_a_real_random_intercept():
    """A mixed model must estimate the fixed effects, not absorb them.

    The group shifts here are twice the residual noise, so a fit that puts the
    covariates in the random part returns nothing for them.
    """
    from spacr.ml import regression_model

    X, y, groups = _grouped()
    model = regression_model(X, y, regression_type="mixed", groups=groups)
    coefs = coefficients(model, X)

    assert abs(coefs["x_pos"] - B_POS) < 0.15
    assert abs(coefs["x_neg"] - B_NEG) < 0.15
    assert abs(coefs["x_null"]) < 0.15
    # The random intercept found the between-group variance it was given.
    assert float(model.cov_re.iloc[0, 0]) > 0.1, (
        "the random intercept collapsed to zero variance despite an "
        "0.8-SD group shift; the grouping is not reaching the fit")


def test_mixed_on_a_null_design_reports_nothing():
    """Same grouping, same group shifts, no fixed effect to find."""
    from spacr.ml import regression_model

    X, y, groups = _grouped(null=True)
    model = regression_model(X, y, regression_type="mixed", groups=groups)
    coefs = coefficients(model, X)

    for name in ("x_pos", "x_neg", "x_null"):
        assert abs(coefs[name]) < 0.2, f"{name} = {coefs[name]:+.4f}"
        assert float(model.pvalues[name]) > 0.01


def test_mixed_refuses_to_run_without_a_grouping():
    """A mixed model with no random effect specified is a linear model."""
    from spacr.ml import regression_model

    X, y, _ = _grouped()
    with pytest.raises(ValueError, match="Groups must be defined"):
        regression_model(X, y, regression_type="mixed")


# ---------------------------------------------------------------------------
# The classifier: direction, pinned from both ends
# ---------------------------------------------------------------------------

def _hinge_fit(seed=0, null=False, **over):
    from spacr.ml import regression_model

    X, y, extras = make_response("binary", seed=seed, null=null)
    model = regression_model(X, y, regression_type="hinge", **over)
    return model, X, y


def test_hinge_recovers_the_sign_of_both_planted_effects():
    """An SVM's coefficients are not log-odds, but their signs are the biology."""
    model, X, y = _hinge_fit()
    coefs = coefficients(model, X)

    assert coefs["x_pos"] > 0.2, coefs
    assert coefs["x_neg"] < -0.2, coefs
    assert abs(coefs["x_null"]) < abs(coefs["x_neg"]), coefs


def test_classifier_orientation_is_not_inverted():
    """AUROC on the decision function must be well above chance, not below.

    ``1 - AUROC`` is the shape an inverted convention takes, and 0.13 is as
    plausible-looking a number as 0.87 on a report nobody cross-checks. sklearn
    scores ``classes_[1]`` -- the LARGER label -- as the event; R's yardstick
    scores the FIRST factor level, which is the smaller one here.
    """
    from sklearn.metrics import roc_auc_score

    model, X, y = _hinge_fit()
    auroc = float(roc_auc_score(y, model.decision_function(X)))
    assert auroc > 0.75, (
        f"AUROC {auroc:.3f}: the classifier is not finding the planted effect "
        f"(or is finding it upside down -- 1 - AUROC = {1 - auroc:.3f})")


def test_the_inverted_score_scores_one_minus_auroc():
    """The control that stops this file passing under either convention.

    If both this and :func:`test_classifier_orientation_is_not_inverted` can be
    true at once, one of them is not testing direction.
    """
    from sklearn.metrics import roc_auc_score

    model, X, y = _hinge_fit()
    score = model.decision_function(X)
    forward = float(roc_auc_score(y, score))
    inverted = float(roc_auc_score(y, -score))
    assert np.isclose(forward + inverted, 1.0, atol=1e-9)
    assert inverted < 0.5 < forward


def test_hinge_on_a_null_design_is_at_chance():
    """Labels with nothing behind them must not separate.

    Averaged over four independent draws: a single AUROC on 600 wells has a
    null SD near 0.025, but the in-sample AUROC of a fitted classifier is
    biased upward, and one draw at 0.57 is noise where four averaging to 0.57
    is a model reading itself.
    """
    from sklearn.metrics import roc_auc_score

    aurocs = []
    for seed in range(4):
        model, X, y = _hinge_fit(seed=100 + seed, null=True)
        aurocs.append(float(roc_auc_score(y, model.decision_function(X))))

    mean_auroc = float(np.mean(aurocs))
    assert 0.40 <= mean_auroc <= 0.62, (
        f"null screens averaged AUROC {mean_auroc:.3f} over "
        f"{[round(a, 3) for a in aurocs]}")
    assert max(aurocs) < 0.72, (
        f"one null screen scored {max(aurocs):.3f}, which is the failure this "
        f"test exists to catch")


def test_the_qc_roc_panel_ranks_a_classifier_by_its_decision_function():
    """The one model whose entire output is a discrimination gets an ROC.

    ``_require_binary`` asked only ``is_binomial``, which is False for a hinge
    fit (an SVM has no likelihood and therefore no family), so the classifier
    was the single type with no ROC and no precision-recall panel. And the
    context's ``fitted`` is the hard 0/1 label, so an ROC drawn from it has two
    operating points and understates the model -- 0.76 where the decision
    function gives 0.85, on the same fit.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score

    from spacr.regression_qc import build_context, draw_panel

    model, X, y = _hinge_fit()
    ctx = build_context(model, X, y, regression_type="hinge")

    assert ctx.is_classifier
    assert ctx.decision_score is not None
    assert np.allclose(ctx.ranking_score, model.decision_function(X))
    # fitted is still the hard label; the panel just does not rank by it.
    assert set(np.unique(ctx.fitted)) <= {0.0, 1.0}

    fig, ax = plt.subplots()
    try:
        stats = draw_panel("roc", ctx, ax)
        pr = draw_panel("precision_recall", ctx, plt.subplots()[1])
    finally:
        plt.close("all")

    assert stats["ranked_by"] == "decision function"
    assert np.isclose(stats["auc"],
                      roc_auc_score(y, model.decision_function(X)))
    assert stats["auc"] > 0.75, stats
    assert pr["average_precision"] > pr["prevalence"]


def test_the_qc_roc_panel_follows_classes_not_the_data():
    """An estimator whose ``classes_`` is descending is flipped, not guessed at.

    The orientation is read off the estimator's own class order, never chosen
    to make the AUC look good -- picking the sign that beats 0.5 would report a
    flattering number on pure noise, which is the one thing this panel exists
    to rule out.
    """
    from spacr.regression_qc import _decision_score

    class _Descending:
        """classes_[1] is the SMALLER label: higher decision => more negative."""
        classes_ = np.array([1.0, 0.0])

        def decision_function(self, X):
            return np.asarray(X["x_pos"], dtype=float)

    X, y, _ = make_response("binary")
    score = _decision_score(_Descending(), X, y)
    assert np.allclose(score, -np.asarray(X["x_pos"], dtype=float))

    class _Ascending(_Descending):
        classes_ = np.array([0.0, 1.0])

    assert np.allclose(_decision_score(_Ascending(), X, y),
                       np.asarray(X["x_pos"], dtype=float))


def test_a_model_with_no_decision_function_has_no_decision_score():
    """Every non-classifier keeps ranking by its fitted values."""
    from spacr.regression_qc import build_context

    model, X, y, _ = _fit("ols", "gaussian")
    ctx = build_context(model, X, y, regression_type="ols")
    assert ctx.decision_score is None
    assert np.allclose(ctx.ranking_score, ctx.fitted)
    assert not ctx.is_classifier


# ---------------------------------------------------------------------------
# Poisson: an exposure offset, or the covariates explain the well's headcount
# ---------------------------------------------------------------------------

def test_poisson_models_the_rate_not_the_headcount():
    """Without offset(log(cell_count)) an inert covariate becomes the top hit.

    ``process_scores`` SUMS the response for the count models, so the Poisson
    response is the well's positive-object count. Wells here run 60 to 960
    cells and ``x_null`` is what drives the size -- it has no effect whatever on
    the per-cell rate. A fit with no offset must explain the headcount with the
    covariates, and ``x_null`` is the only one that can.
    """
    from spacr.ml import regression_model
    import statsmodels.api as sm

    X, y, extras = make_response("count")
    n_total = extras["exposure"]

    fitted = regression_model(X, y, regression_type="poisson",
                              exposure=n_total)
    with_offset = coefficients(fitted, X)
    no_offset = coefficients(
        sm.GLM(y, X, family=sm.families.Poisson()).fit(), X)

    # The offset fit is right about all three.
    assert abs(with_offset["x_pos"] - B_POS) < 0.15
    assert abs(with_offset["x_neg"] - B_NEG) < 0.15
    assert abs(with_offset["x_null"]) < 0.10
    assert float(fitted.pvalues["x_null"]) > 0.01

    # The offset-free fit invents an effect on the inert covariate, larger than
    # the real negative effect it is sitting next to.
    assert abs(no_offset["x_null"]) > abs(B_NEG), (
        "the offset-free control did not misbehave, so this test is no longer "
        f"testing anything: {no_offset}")
    assert abs(no_offset["x_null"]) > abs(with_offset["x_null"]) * 5
    no_offset_fit = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    assert float(no_offset_fit.pvalues["x_null"]) < 1e-6, (
        "the fabricated effect is not even significant, so it would not reach "
        "a hit list; this test is about the one that does")


def test_glm_auto_uses_the_same_exposure_as_poisson_when_it_picks_poisson():
    """A family chosen from the data is fitted the way the named one is."""
    from spacr.ml import regression_model

    X, y, extras = make_response("count")
    auto = regression_model(X, y, regression_type="glm",
                            exposure=extras["exposure"])
    named = regression_model(X, y, regression_type="poisson",
                             exposure=extras["exposure"])
    assert isinstance(auto.family, type(named.family))
    assert np.allclose(auto.params.values, named.params.values, atol=1e-8)


def test_poisson_refuses_an_exposure_that_is_not_a_cell_count():
    """log(0) would be -inf and every number downstream would follow it."""
    from spacr.ml import regression_model

    X, y, extras = make_response("count")
    bad = np.asarray(extras["exposure"], dtype=float).copy()
    bad[3] = 0.0
    with pytest.raises(ValueError, match="finite and strictly positive"):
        regression_model(X, y, regression_type="poisson", exposure=bad)

    with pytest.raises(ValueError, match="must carry its own cell count"):
        regression_model(X, y, regression_type="poisson",
                         exposure=bad[:10])


def test_poisson_without_an_exposure_still_fits_and_says_what_it_lost(capsys):
    """A count with no denominator is fittable; it is just not a rate."""
    from spacr.ml import regression_model

    X, y, _ = make_response("count")
    model = regression_model(X, y, regression_type="poisson")
    capture = capsys.readouterr().out
    assert "no offset(log(cell_count))" in capture
    assert model.params is not None


# ---------------------------------------------------------------------------
# Settings: declared means applied
# ---------------------------------------------------------------------------

#: For each policed setting, a value that must visibly change the fit, and the
#: response scale to test it on.
SETTING_PROBE = {
    # 0.05 rather than something bigger: a lasso penalty on this scale zeroes
    # the design outright above ~0.3, and regression_model refuses that fit, so
    # a larger probe would test the refusal rather than the setting.
    "alpha": (0.05, "gaussian"),
    "l1_ratio": (0.05, "gaussian"),
    "cov_type": ("HC3", "gaussian"),
    "quantile": (0.9, "gaussian"),
    "huber_t": (0.3, "gaussian"),
    "hinge_threshold": (0.5, "gaussian"),
}


def _settings_cases():
    from spacr.ml import REGRESSION_SETTINGS_USED

    return [(t, s) for t, names in REGRESSION_SETTINGS_USED.items()
            for s in names if s in SETTING_PROBE]


@pytest.mark.parametrize(("regression_type", "setting"), _settings_cases(),
                         ids=lambda v: str(v))
def test_every_setting_a_type_declares_actually_changes_the_fit(
        regression_type, setting):
    """A setting a backend advertises must reach the estimator.

    ``regression_model`` refuses a setting the chosen backend cannot read.
    This is the other direction: a setting it claims to read and quietly does
    not, which is the same silent failure wearing the opposite label.
    ``alpha='auto'`` for ``hinge`` was exactly this -- it meant "cross-validate
    the penalty" for the other three penalised backends and ``C = 1`` here, and
    the coefficients for ``alpha='auto'`` and ``alpha=1.0`` were bit-identical.
    """
    from spacr.ml import regression_model

    value, kind = SETTING_PROBE[setting]
    X, y, extras = make_response(kind)
    base_kwargs = {}
    if regression_type == "wls":
        base_kwargs["weights"] = np.linspace(1.0, 50.0, len(y))
    if regression_type in ("logit", "probit", "quasi_binomial"):
        X, y, extras = make_response("fraction_logit")
        base_kwargs["weights"] = extras["weights"]
    if regression_type == "poisson":
        X, y, extras = make_response("count")
        base_kwargs["exposure"] = extras["exposure"]
    if regression_type in ("lasso", "ridge", "elasticnet"):
        # The stock alpha=1 zeroes a design on this scale outright, which
        # regression_model refuses; the comparison has to be between two
        # penalties that both leave a fit behind.
        base_kwargs.setdefault("alpha", 0.01)
    if regression_type == "hinge":
        if setting == "alpha":
            X, y, _ = make_response("binary")
        else:
            # A continuous response is the only one a threshold can move.
            base_kwargs["hinge_threshold"] = 0.0

    changed_kwargs = dict(base_kwargs)
    changed_kwargs[setting] = value
    baseline = regression_model(X, y, regression_type=regression_type,
                                **base_kwargs)
    changed = regression_model(X, y, regression_type=regression_type,
                               **changed_kwargs)

    if setting == "cov_type":
        # A sandwich estimator changes the standard errors, not the point
        # estimates -- asserting on the coefficients would pass a cov_type that
        # was thrown away.
        assert not np.allclose(np.asarray(baseline.bse, dtype=float),
                               np.asarray(changed.bse, dtype=float)), (
            f"{regression_type}: cov_type={value!r} left every standard error "
            f"untouched, so it was not applied")
        assert np.allclose(np.asarray(baseline.params, dtype=float),
                           np.asarray(changed.params, dtype=float), atol=1e-8)
        return

    before = np.asarray(list(coefficients(baseline, X).values()))
    after = np.asarray(list(coefficients(changed, X).values()))
    assert not np.allclose(before, after, atol=1e-9), (
        f"{regression_type}: {setting}={value!r} produced exactly the same "
        f"coefficients as the default, so it was not applied")


def test_hinge_alpha_auto_cross_validates_instead_of_meaning_c_equals_one(
        capsys):
    """'auto' means the same thing for hinge as it does for lasso and ridge.

    It used to mean C = 1, which is what alpha=1.0 means -- so a user who asked
    for a cross-validated margin got an arbitrary fixed one, and the two fits
    were bit-identical.
    """
    from spacr.ml import regression_model

    X, y, _ = make_response("binary")
    auto = regression_model(X, y, regression_type="hinge", alpha="auto")
    fixed = regression_model(X, y, regression_type="hinge", alpha=1.0)

    assert "Optimal alpha for hinge" in capsys.readouterr().out
    assert not np.allclose(auto.coef_, fixed.coef_), (
        "alpha='auto' and alpha=1.0 gave identical coefficients, so nothing "
        "was cross-validated")


def test_hinge_balances_the_classes_instead_of_predicting_the_majority():
    """A 95/5 screen must still produce a decision boundary.

    An unweighted hinge minimises its loss on an imbalanced screen by calling
    every well negative: the coefficients collapse to ~0 and the run reports no
    hits, which is indistinguishable from a screen that has none.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.svm import LinearSVC

    from spacr.ml import regression_model

    # A different stream from the one `design` drew the covariates with: the
    # same seed would make the Bernoulli draw a deterministic function of
    # x_pos, and the "classifier" would be reading an identity.
    rng = np.random.default_rng(9_007)
    n = 800
    X, x_pos, x_neg, _ = design(seed=7, n=n)
    # Prevalence ~6%: the intercept, not the covariates, makes it rare.
    eta = -3.2 + 2.5 * x_pos
    y = pd.Series((rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float))
    assert 0.02 < y.mean() < 0.15, y.mean()

    balanced = regression_model(X, y, regression_type="hinge", alpha=1.0)
    unweighted = LinearSVC(C=1.0, loss="hinge", dual=True, max_iter=20000,
                           random_state=0).fit(X, y)

    assert float(balanced.coef_.ravel()[1]) > 0.2, balanced.coef_
    assert roc_auc_score(y, balanced.decision_function(X)) > 0.8
    # The control: the same fit without balancing predicts one class outright.
    assert set(np.unique(unweighted.predict(X))) == {0.0}, (
        "the unweighted control no longer collapses, so this test is no "
        "longer testing anything")


@pytest.mark.parametrize("regression_type", ["logit", "probit",
                                             "quasi_binomial"])
def test_the_binomial_links_weight_a_well_by_its_cell_count(regression_type):
    """A fraction from 30 cells is not the same evidence as one from 400.

    The binomial variance function only knows how much information a per-well
    fraction carries if it is told the denominator. spaCR's answer is
    ``var_weights = cell_count``; this is the assertion that it reaches the
    fit, because a fit that drops it produces the same plausible coefficients
    with standard errors that are wrong by the square root of the cell count.
    """
    from spacr.ml import regression_model

    X, y, extras = make_response("fraction_logit")
    weighted = regression_model(X, y, regression_type=regression_type,
                                weights=extras["weights"])
    unweighted = regression_model(X, y, regression_type=regression_type)

    assert not np.allclose(np.asarray(weighted.bse, dtype=float),
                           np.asarray(unweighted.bse, dtype=float)), (
        f"{regression_type}: the cell counts changed nothing, so they are not "
        f"reaching the fit")
    # And the weighted fit is the sharper one: it is using more information.
    assert float(weighted.bse["x_pos"]) < float(unweighted.bse["x_pos"])
    # Weighting changes the precision, not the effect. (The magnitude itself is
    # link-dependent -- a probit coefficient is ~1/1.6 of the logit one that
    # generated this response -- and is pinned per link by RECOVERY.)
    assert float(weighted.params["x_pos"]) > 0
    assert float(weighted.params["x_neg"]) < 0
    assert np.isclose(float(weighted.params["x_pos"]),
                      float(unweighted.params["x_pos"]), rtol=0.25)


def test_quasi_binomial_widens_the_errors_a_plain_binomial_understates():
    """The one reason to offer quasi-binomial is the free dispersion.

    Same coefficients, wider standard errors. A quasi-binomial that reports the
    binomial's standard errors is a binomial with a longer name, and every
    p-value it produces on an overdispersed screen is too small.
    """
    from spacr.ml import regression_model

    X, y, extras = make_response("fraction_logit")
    # Overdispersion: push each well's fraction away from its binomial mean.
    rng = np.random.default_rng(11)
    y = pd.Series(np.clip(y + rng.normal(0, 0.08, len(y)), 1e-3, 1 - 1e-3))

    binomial = regression_model(X, y, regression_type="logit",
                                weights=extras["weights"])
    quasi = regression_model(X, y, regression_type="quasi_binomial",
                             weights=extras["weights"])

    assert np.allclose(np.asarray(binomial.params, dtype=float),
                       np.asarray(quasi.params, dtype=float), atol=1e-6)
    assert float(quasi.bse["x_pos"]) > float(binomial.bse["x_pos"]) * 1.5, (
        f"quasi-binomial se {float(quasi.bse['x_pos']):.4g} vs binomial "
        f"{float(binomial.bse['x_pos']):.4g}: the dispersion is not being "
        f"estimated")


def test_wls_weights_reach_the_fit_and_the_standard_errors():
    """WLS on cell counts must differ from OLS in estimate AND in error."""
    from spacr.ml import regression_model

    X, y, _ = make_response("gaussian")
    # Noise that shrinks with the well's cell count, which is what the weights
    # are for: the big wells are the precise ones. The spread is large on
    # purpose -- a 20-cell well carries ~6x the noise of an 800-cell one, which
    # is the regime WLS exists for and where ignoring it costs something.
    rng = np.random.default_rng(13)
    cells = rng.integers(20, 800, len(y)).astype(float)
    y = pd.Series(np.asarray(y) + rng.normal(0, 12.0, len(y)) / np.sqrt(cells))

    ols = regression_model(X, y, regression_type="ols")
    wls = regression_model(X, y, regression_type="wls", weights=cells)

    assert float(wls.bse["x_pos"]) < float(ols.bse["x_pos"])
    assert abs(float(wls.params["x_pos"]) - B_POS) < 0.1


# ---------------------------------------------------------------------------
# random_row_column_effects: the flag that silently replaced the model
# ---------------------------------------------------------------------------

def test_random_row_column_effects_refuses_to_replace_a_chosen_model():
    """The flag fits a MixedLM whatever regression_type says. Now it says so.

    It used to win in silence: the run fitted a mixed model, ignored every
    penalty setting on the way (the mixed branch never reaches
    regression_model, so _reject_unused_settings never runs) and wrote the
    result into results/<screen>/lasso/.
    """
    from spacr.ml import _reconcile_random_row_column_effects

    with pytest.raises(ValueError, match="cannot also fit regression_type"):
        _reconcile_random_row_column_effects(
            {"random_row_column_effects": True, "regression_type": "lasso"})

    with pytest.raises(ValueError, match="cannot also fit regression_type"):
        _reconcile_random_row_column_effects(
            {"random_row_column_effects": True, "regression_type": "quantile"})


@pytest.mark.parametrize("reg_type", [None, "ols", "mixed"])
def test_random_row_column_effects_renames_the_run_after_what_it_fits(reg_type):
    """The results folder is named from settings['regression_type']."""
    from spacr.ml import _reconcile_random_row_column_effects

    settings = {"random_row_column_effects": True, "regression_type": reg_type}
    _reconcile_random_row_column_effects(settings)
    assert settings["regression_type"] == "mixed"


@pytest.mark.parametrize(("setting", "value"), [
    ("lasso_n_boot", 25),
    ("lasso_selection_threshold", 0.9),
    ("hinge_n_boot", 25),
])
@pytest.mark.parametrize("reg_type", ["ols", "ridge", "mixed", None])
def test_a_post_fit_setting_the_model_never_reads_is_refused(setting, value,
                                                             reg_type):
    """The three knobs that configure the hit list, not the fit.

    ``regression_model`` never sees these -- they configure what
    ``perform_regression`` does with the coefficients afterwards -- so nothing
    was policing them, and ``lasso_selection_threshold=0.9`` on an OLS run went
    through in silence for fifteen of the seventeen types.

    ``regression_type=None`` is included because "it might auto-select lasso"
    is not true: ``check_distribution`` only ever returns logit, beta,
    quasi_binomial, ols or glm.
    """
    from spacr.ml import _reject_unused_run_settings

    with pytest.raises(ValueError, match=f"does not read {setting}"):
        _reject_unused_run_settings({"regression_type": reg_type,
                                     setting: value})


@pytest.mark.parametrize(("reg_type", "setting"), [
    ("lasso", "lasso_n_boot"),
    ("lasso", "lasso_selection_threshold"),
    ("elasticnet", "lasso_n_boot"),
    ("elasticnet", "lasso_selection_threshold"),
    ("hinge", "hinge_n_boot"),
])
def test_the_type_that_does_read_a_post_fit_setting_accepts_it(reg_type,
                                                               setting):
    """Accepted, and handed back unchanged.

    ``_reject_unused_run_settings`` returns the dict it was given, so the
    assertion is that the value survives rather than that the call did not
    raise: a version that quietly dropped the knob it had just approved
    would pass "it did not raise" and lose the setting.
    """
    from spacr.ml import _reject_unused_run_settings

    settings = {"regression_type": reg_type, setting: 0.42}
    returned = _reject_unused_run_settings(settings)

    assert returned is settings
    assert returned == {"regression_type": reg_type, setting: 0.42}


def test_a_post_fit_setting_left_at_its_default_is_not_refused():
    """The GUI posts every widget on the panel, touched or not.

    All three post-fit knobs, at their defaults, on a type that reads none
    of them: accepted, and every one of them still there afterwards.
    """
    from spacr.ml import _reject_unused_run_settings

    at_defaults = {
        "regression_type": "ols", "lasso_n_boot": 200,
        "lasso_selection_threshold": 0.6, "hinge_n_boot": 200}
    returned = _reject_unused_run_settings(dict(at_defaults))
    assert returned == at_defaults

    bare = _reject_unused_run_settings({"regression_type": "ols"})
    assert bare == {"regression_type": "ols"}
    _reject_unused_run_settings({"regression_type": "ols"})


def test_random_row_column_effects_off_changes_nothing():
    from spacr.ml import _reconcile_random_row_column_effects

    settings = {"random_row_column_effects": False, "regression_type": "lasso",
                "alpha": 0.2}
    _reconcile_random_row_column_effects(settings)
    assert settings["regression_type"] == "lasso"


@pytest.mark.slow
def test_random_row_column_effects_names_the_results_folder_end_to_end(
        tmp_path, monkeypatch):
    """The run that was fitted is the run the folder is named for.

    ``_perform_regression_set_paths`` names ``results/<screen>/<type>/`` from
    ``settings['regression_type']``, and the mixed override happened later and
    somewhere else -- so a run configured as 'ols' with the flag on wrote a
    MixedLM fit into ``results/<screen>/ols/`` with nothing anywhere
    disagreeing.
    """
    import os

    import spacr.ml as ML
    import spacr.plot as P
    from tests.test_regression_types import settings_for, write_screen

    monkeypatch.setattr(P, "plot_plates", lambda df, **kw: None)
    monkeypatch.setattr(P, "plot_histogram", lambda df, column, dst=None: None)
    monkeypatch.setattr(P, "plot_data_from_csv", lambda settings: (None, None))
    monkeypatch.setattr(ML, "minimum_cell_simulation", lambda s, **kw: 3)

    # Four cells per well: this test is about which folder the run writes to,
    # not about statistical power, and a MixedLM with row and column variance
    # components over two full plates is the slowest fit in the module.
    score, count = write_screen(tmp_path, plates=("plate1", "plate2"),
                                n_cells=4)
    settings = settings_for(score, count, regression_type="ols",
                            random_row_column_effects=True)
    out = ML.perform_regression(settings)

    assert settings["regression_type"] == "mixed"
    assert os.path.isfile(os.path.join(
        os.path.dirname(count), "results", "screen_scores", "mixed", "list",
        "results.csv"))
    assert not os.path.isdir(os.path.join(
        os.path.dirname(count), "results", "screen_scores", "ols"))
    assert len(out["results"]) > 0


def test_random_row_column_effects_refuses_a_setting_the_mixed_fit_ignores():
    """The mixed branch reads none of the per-model knobs, and now says so."""
    from spacr.ml import _reconcile_random_row_column_effects

    with pytest.raises(ValueError, match="does not read quantile"):
        _reconcile_random_row_column_effects(
            {"random_row_column_effects": True, "regression_type": "ols",
             "quantile": 0.9})

    with pytest.raises(ValueError, match="does not read cov_type"):
        _reconcile_random_row_column_effects(
            {"random_row_column_effects": True, "regression_type": None,
             "cov_type": "HC3"})


# ---------------------------------------------------------------------------
# Quantile regression fits a quantile, which is the only reason to offer it
# ---------------------------------------------------------------------------

def test_quantile_regression_fits_the_quantile_it_was_given():
    """q=0.1, 0.5 and 0.9 must be three different fits of the same slope.

    Under symmetric noise all three recover the same slope and three ordered
    intercepts. A backend that ignores ``quantile`` returns one fit three
    times, which is what makes ``quantile`` worth its own test rather than a
    row in the recovery table.
    """
    from spacr.ml import regression_model

    X, y, _ = make_response("gaussian")
    fits = {q: regression_model(X, y, regression_type="quantile", quantile=q)
            for q in (0.1, 0.5, 0.9)}
    intercepts = [float(fits[q].params["Intercept"]) for q in (0.1, 0.5, 0.9)]

    assert intercepts[0] < intercepts[1] < intercepts[2], intercepts
    for q, model in fits.items():
        assert abs(float(model.params["x_pos"]) - B_POS) < 0.2, (q, model.params)
        assert abs(float(model.params["x_neg"]) - B_NEG) < 0.2, (q, model.params)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
def test_quantile_outside_the_open_unit_interval_is_refused(bad):
    from spacr.ml import regression_model

    X, y, _ = make_response("gaussian")
    with pytest.raises(ValueError, match="strictly inside"):
        regression_model(X, y, regression_type="quantile", quantile=bad)


# ---------------------------------------------------------------------------
# Coverage of the offering itself
# ---------------------------------------------------------------------------

#: The types whose planted-effect test is not the RECOVERY table above,
#: mapped to where it is. Named rather than merely excluded so adding a type
#: and forgetting to test it cannot be waved through by adding it here without
#: noticing what the entry is claiming.
COVERED_ELSEWHERE = {
    "mixed": "test_mixed_recovers_the_fixed_effect_over_a_real_random_intercept",
    "hinge": "test_hinge_recovers_the_sign_of_both_planted_effects",
    # ADVI: too slow for this file. The signal case and the hit/non-hit
    # separation are pinned by test_regression_types.py (marked slow) and the
    # null case by test_power_model.test_null_screen_auroc_is_near_chance.
    "horseshoe": "tests/test_regression_types.py + tests/test_power_model.py",
    # Instruction 133. Both need the DESIGN COLUMN NAMES to find the gene
    # behind each guide, and the fixtures in this file build a design whose
    # columns are 'a', 'b', 'c' -- there is no gene in them to group by, so
    # the planted-effect test has to be run on a screen-shaped design.
    "group_lasso":
        "tests/test_group_lasso_and_rra_backends.py"
        "::test_group_lasso_selects_a_genes_guides_as_a_block",
    "rra":
        "tests/test_group_lasso_and_rra_backends.py"
        "::test_rra_recovers_the_planted_gene_as_the_top_call",
}


def test_every_offered_type_is_covered_by_a_ground_truth_test():
    """No regression_type may be offered without an answer known in advance.

    'It ran' is what every bug this module has had also did.
    """
    from spacr.ml import REGRESSION_TYPES

    covered = {case[0] for case in RECOVERY} | set(COVERED_ELSEWHERE)
    assert set(REGRESSION_TYPES) == covered, (
        "a regression type is offered with no planted-effect test: "
        f"{sorted(set(REGRESSION_TYPES) ^ covered)}")


def test_the_tests_named_as_covering_a_type_exist():
    """COVERED_ELSEWHERE must point at something real."""
    here = set(globals())
    for reg_type, where in COVERED_ELSEWHERE.items():
        if where.startswith("tests/"):
            continue
        assert where in here, (
            f"{reg_type} claims to be covered by {where}(), which does not "
            f"exist in this module")
