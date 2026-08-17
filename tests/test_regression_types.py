"""Every regression_type spaCR advertises, driven end to end on a screen.

``perform_regression`` is the only way a user reaches a regression, so every
claim in here is made through it - not against ``regression_model`` alone. A
backend that fits in isolation and dies in the pipeline is the bug this file
exists to catch, and it is not hypothetical: ``'quantile'`` passed the entry
point's whitelist, was given its own ``agg_type`` handling and its own volcano
filename, and then raised "Unsupported regression type quantile" from the last
statement of the run; ``'beta'`` and ``'quasi_binomial'`` were fittable by
``regression_model`` and auto-selected by ``check_distribution`` while the
entry point refused them by name.

The screen is synthetic but shaped like a real one: 96 wells per plate, four
gRNAs per well at Dirichlet-drawn library fractions, one gene with a planted
positive effect proportional to its share of the well. Recovering that effect
- the planted gene carrying the largest positive gene coefficient - is the
assertion, because "it ran" is what every one of the bugs above also did.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic screen
# ---------------------------------------------------------------------------

GENES = ("000000", "233460", "239740", "111111", "222222")
N_GRNA_PER_GENE = 3
ROWS = tuple(f"r{i}" for i in range(1, 9))
COLS = tuple(f"c{i}" for i in range(1, 13))
HIT_GENE = "239740"
EFFECT = 0.45
GRNAS_PER_WELL = 4


def _grnas():
    return [f"TGGT1_{g}_{i}"
            for g in GENES for i in range(1, N_GRNA_PER_GENE + 1)]


def write_screen(tmp_path, response="fraction", seed=0, n_cells=12,
                 plates=("plate1",)):
    """Write a score CSV and a count CSV for a screen with a planted hit.

    :param tmp_path: Directory to write into.
    :param response: ``'fraction'`` for a per-cell score in (0, 1),
        ``'binary'`` for a 0/1 class call (what the count models need),
        ``'unbounded'`` for a continuous score off the unit interval.
    :param seed: RNG seed; the whole screen is deterministic from it.
    :param n_cells: Objects scored per well.
    :param plates: Plate ids; more than one is what a mixed model needs.
    :returns: ``(score_csv, count_csv)``.
    """
    rng = np.random.default_rng(seed)
    grnas = _grnas()
    counts, scores = [], []
    for plate in plates:
        plate_shift = rng.normal(0, 0.03)      # the batch effect 'mixed' models
        for row in ROWS:
            for column in COLS:
                chosen = rng.choice(len(grnas), size=GRNAS_PER_WELL,
                                    replace=False)
                fracs = rng.dirichlet(np.ones(GRNAS_PER_WELL) * 3)
                total = int(rng.integers(500, 2000))
                effect = 0.0
                for index, fraction in zip(chosen, fracs):
                    grna = grnas[index]
                    counts.append({"plateID": plate, "rowID": row,
                                   "columnID": column, "grna": grna,
                                   "count": int(round(fraction * total))})
                    if grna.split("_")[1] == HIT_GENE:
                        effect += EFFECT * fraction
                mean = 0.30 + effect + plate_shift
                for _ in range(n_cells):
                    if response == "fraction":
                        value = float(np.clip(mean + rng.normal(0, 0.05),
                                              0.02, 0.98))
                    elif response == "binary":
                        value = float(rng.uniform() < mean)
                    else:
                        value = float(10.0 * mean + rng.normal(0, 1.0))
                    scores.append({"plateID": plate, "rowID": row,
                                   "columnID": column, "fieldID": "f1",
                                   "pred": value})
    sdir = tmp_path / "scores"
    cdir = tmp_path / "counts"
    sdir.mkdir(exist_ok=True)
    cdir.mkdir(exist_ok=True)
    score = str(sdir / "screen_scores.csv")
    count = str(cdir / "counts.csv")
    pd.DataFrame(scores).to_csv(score, index=False)
    pd.DataFrame(counts).to_csv(count, index=False)
    return score, count


def settings_for(score, count, **over):
    """Settings exactly as a dispatcher builds them, plus this suite's choices.

    Finished by the SAME defaults builder all three entry points use, so a key
    the builder fails to supply is missing here too and these tests fail the
    way a user's run fails.
    """
    from spacr.settings import get_perform_regression_default_settings

    settings = {
        "score_data": [score],
        "count_data": [count],
        "dependent_variable": "pred",
        "min_cell_count": 3,
        "fraction_threshold": 0.01,
        "metadata_files": [],
        "toxo": False,
        "controls": None,
        "outlier_detection": False,
        "filter_value": [],
        "control_wells": [],
    }
    settings.update(over)
    return get_perform_regression_default_settings(settings)


@pytest.fixture(autouse=True)
def _no_figure_leak():
    yield
    plt.close("all")


@pytest.fixture
def stubs(monkeypatch):
    """Stub only the visual helpers and the Monte-Carlo cell-count sweep."""
    import spacr.plot as P
    import spacr.ml as ML
    import spacr.toxo   # noqa: F401 - warm the lazy imports
    import spacr.sequencing  # noqa: F401

    monkeypatch.setattr(P, "plot_plates", lambda df, **kwargs: None)
    monkeypatch.setattr(P, "plot_histogram",
                        lambda df, column, dst=None: None)
    monkeypatch.setattr(P, "plot_data_from_csv",
                        lambda settings: (None, None))
    monkeypatch.setattr(ML, "minimum_cell_simulation",
                        lambda settings, **kwargs: 3)


def gene_coefficients(results):
    """Per-gene coefficient, keyed by the bare gene id."""
    genes = results[results["feature"].str.startswith("gene_fraction:gene[")]
    return {feature.split("[")[1].rstrip("]"): coefficient
            for feature, coefficient
            in zip(genes["feature"], genes["coefficient"])}


def assert_recovers_the_planted_gene(results, regression_type):
    """The planted gene must carry the largest positive gene coefficient."""
    coefficients = gene_coefficients(results)
    assert set(coefficients) == set(GENES), (
        f"{regression_type}: expected one coefficient per gene, got "
        f"{sorted(coefficients)}")
    assert not np.isnan(list(coefficients.values())).any(), (
        f"{regression_type}: NaN gene coefficients {coefficients}")
    ranked = sorted(coefficients, key=coefficients.get, reverse=True)
    assert ranked[0] == HIT_GENE, (
        f"{regression_type}: the planted gene {HIT_GENE} is not the top "
        f"coefficient; ranking was {[(g, round(coefficients[g], 4)) for g in ranked]}")
    assert coefficients[HIT_GENE] > 0, (
        f"{regression_type}: planted effect came back with the wrong sign "
        f"({coefficients[HIT_GENE]})")


# ---------------------------------------------------------------------------
# Job A + B - every advertised type, end to end
# ---------------------------------------------------------------------------

#: (regression_type, response kind, extra settings). One entry per name in
#: spacr.ml.REGRESSION_TYPES that can be fitted with what spaCR ships;
#: 'horseshoe' needs spacr.power_model and has its own test below.
END_TO_END = [
    ("ols", "fraction", {}),
    ("wls", "fraction", {}),
    ("rlm", "fraction", {}),
    ("huber", "fraction", {}),
    ("glm", "fraction", {}),
    ("poisson", "binary", {}),
    ("quasi_binomial", "fraction", {}),
    ("beta", "fraction", {}),
    ("logit", "fraction", {}),
    ("probit", "fraction", {}),
    ("quantile", "fraction", {"quantile": 0.5}),
    ("mixed", "fraction", {"plates": ("plate1", "plate2", "plate3")}),
    ("lasso", "fraction", {"alpha": "auto", "lasso_n_boot": 5,
                           "lasso_selection_threshold": 0.5}),
    ("ridge", "fraction", {"alpha": 1.0}),
    ("elasticnet", "fraction", {"alpha": "auto", "l1_ratio": 0.5,
                                "lasso_n_boot": 5,
                                "lasso_selection_threshold": 0.5}),
    ("hinge", "binary", {"hinge_threshold": 0.5, "hinge_n_boot": 40}),
]


@pytest.mark.parametrize(("regression_type", "response", "extra"), END_TO_END,
                         ids=[case[0] for case in END_TO_END])
def test_every_regression_type_runs_and_recovers_the_planted_effect(
        tmp_path, stubs, regression_type, response, extra):
    """Each backend fits through perform_regression and finds the planted gene."""
    from spacr.ml import perform_regression

    extra = dict(extra)
    plates = extra.pop("plates", ("plate1",))
    score, count = write_screen(tmp_path, response=response, plates=plates)
    settings = settings_for(score, count, regression_type=regression_type,
                            **extra)

    out = perform_regression(settings)

    results = out["results"]
    assert len(results) > 0
    assert results["coefficient"].notna().all()
    assert results["p_value"].notna().all()
    assert (results["p_value"].between(0.0, 1.0)).all()
    # One coefficient per gRNA that survived the fraction threshold, plus one
    # per gene, plus the intercept.
    assert results["feature"].str.contains(r"grna\[").sum() > 0
    assert_recovers_the_planted_gene(results, regression_type)
    # THE RESULTS FOLDER IS NAMED FOR THE MODEL THAT WAS ACTUALLY FITTED,
    # and it is asked for rather than spelled out here.
    #
    # This assertion spelled out `results/screen_scores/<type>/list/` --
    # the four-level path from before the output rule changed to
    # `<count folder>/results/<type>`, with `_1`, `_2` for a repeat. It went
    # stale the day the rule changed and stayed red for every one of the
    # seventeen types, which is a lot of noise for a suite to carry and is
    # exactly how a real failure gets skipped over.
    folder = out["res_folder"]
    assert os.path.isfile(os.path.join(folder, "results.csv"))
    # The repeat suffix is `_1`, `_2` -- stripped by matching DIGITS at the
    # end, never by splitting on "_": `quasi_binomial` and
    # `guide_permutation` contain the separator, so a left-to-right split
    # reads the type as "quasi". The same trap as the plate keys.
    assert re.sub(r"_\d+$", "", os.path.basename(folder)) == regression_type, \
        folder
    assert os.path.dirname(folder) == os.path.join(
        os.path.dirname(count), "results"), folder


def test_every_advertised_type_has_a_coefficient_branch():
    """A model that fits must be turnable into a coefficient table.

    The two halves used to be separate hand-written lists that disagreed; this
    is the round trip that stops them drifting again.
    """
    from spacr.ml import (REGRESSION_TYPES, _SKLEARN_COEF_TYPES,
                          _STATSMODELS_COEF_TYPES)

    covered = set(_STATSMODELS_COEF_TYPES) | set(_SKLEARN_COEF_TYPES) | {
        "beta", "hinge"}
    assert set(REGRESSION_TYPES) == covered, (
        "process_model_coefficients and REGRESSION_TYPES disagree: "
        f"{sorted(set(REGRESSION_TYPES) ^ covered)}")


def test_the_entry_point_whitelist_is_the_dispatcher_list(tmp_path, stubs):
    """perform_regression accepts exactly what regression_model can fit.

    The hand-written copy this replaces refused 'beta' and 'quasi_binomial' -
    both fittable, both auto-selected by check_distribution - and accepted
    'gls', 'wls', 'rlm' and 'quantile', none of which had a backend.
    """
    import inspect

    from spacr.ml import REGRESSION_TYPES, perform_regression

    source = inspect.getsource(perform_regression)
    assert "reg_types = [" not in source, (
        "perform_regression is carrying its own whitelist again")
    assert "REGRESSION_TYPES" in source

    score, count = write_screen(tmp_path)
    with pytest.raises(ValueError, match="Unsupported regression type banana"):
        perform_regression(settings_for(score, count, regression_type="banana"))
    assert "beta" in REGRESSION_TYPES
    assert "quasi_binomial" in REGRESSION_TYPES


def test_gls_is_refused_with_the_reason_and_the_alternative(tmp_path, stubs):
    """'gls' was advertised and had no backend; the refusal names the fix."""
    from spacr.ml import perform_regression

    score, count = write_screen(tmp_path)
    with pytest.raises(ValueError, match=r"identical to 'ols'"):
        perform_regression(settings_for(score, count, regression_type="gls"))


# ---------------------------------------------------------------------------
# Job B - hinge
# ---------------------------------------------------------------------------

def test_hinge_refuses_a_continuous_response_rather_than_thresholding_it(
        tmp_path, stubs):
    """A cut chosen by the software would decide the hypothesis, not the biology."""
    from spacr.ml import perform_regression

    score, count = write_screen(tmp_path, response="fraction")
    settings = settings_for(score, count, regression_type="hinge")
    with pytest.raises(ValueError, match="needs a binary"):
        perform_regression(settings)


def test_hinge_accepts_an_already_binary_response_without_a_threshold(
        tmp_path, stubs):
    """Two distinct values ARE the two classes; no threshold is needed."""
    from spacr.ml import perform_regression

    score, count = write_screen(tmp_path, response="binary")
    settings = settings_for(score, count, regression_type="hinge",
                            agg_type=None, hinge_n_boot=20)
    out = perform_regression(settings)
    assert_recovers_the_planted_gene(out["results"], "hinge")


def test_binarise_response_rules():
    """The documented binarisation rule, on its own."""
    from spacr.ml import binarise_response

    # already binary: the higher value is the positive class
    assert list(binarise_response([0, 1, 1, 0])) == [0.0, 1.0, 1.0, 0.0]
    assert list(binarise_response([2, 5, 5, 2])) == [0.0, 1.0, 1.0, 0.0]
    # explicit threshold: strictly greater is positive
    assert list(binarise_response([0.2, 0.4, 0.6], threshold=0.4)) == \
        [0.0, 0.0, 1.0]
    # a threshold that leaves one class is refused, not fitted
    with pytest.raises(ValueError, match="one class"):
        binarise_response([0.2, 0.4, 0.6], threshold=0.9)
    with pytest.raises(ValueError, match="two classes"):
        binarise_response([0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="needs a binary"):
        binarise_response([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="finite"):
        binarise_response([0.0, 1.0, np.nan])


# ---------------------------------------------------------------------------
# Job C - a setting a model cannot read is refused, not ignored
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("regression_type", "setting", "value"), [
    ("lasso", "cov_type", "HC3"),
    ("ols", "alpha", 0.25),
    ("ridge", "l1_ratio", 0.9),
    ("ols", "quantile", 0.9),
    ("beta", "hinge_threshold", 0.5),
    ("ols", "huber_t", 2.0),
])
def test_a_setting_the_model_cannot_read_is_refused(regression_type, setting,
                                                    value):
    """A silently ignored setting is this project's most expensive failure mode.

    cov_type='HC3' with lasso is the archetype: sklearn has no covariance
    estimator at all, so the run would have reported ordinary p-values under a
    robust-sounding label and nothing anywhere would have disagreed.
    """
    from spacr.ml import regression_model

    rng = np.random.default_rng(0)
    n = 60
    x = rng.normal(0, 1, n)
    X = pd.DataFrame({"Intercept": 1.0, "x": x})
    y = pd.Series(0.5 + 2.0 * x + rng.normal(0, 0.3, n))

    with pytest.raises(ValueError, match=f"does not read {setting}"):
        regression_model(X, y, regression_type=regression_type,
                         **{setting: value})


def test_a_setting_left_at_its_default_is_not_refused():
    """The GUI posts every widget on the panel, touched or not."""
    from spacr.ml import regression_model

    rng = np.random.default_rng(1)
    n = 40
    X = pd.DataFrame({"Intercept": 1.0, "x": rng.normal(0, 1, n)})
    y = pd.Series(rng.normal(0, 1, n))

    model = regression_model(X, y, regression_type="ols", alpha=1.0,
                             l1_ratio=0.5, cov_type=None, quantile=0.5,
                             hinge_threshold=None, huber_t=1.345)
    assert "x" in model.params.index


@pytest.mark.parametrize("key", [
    "regression_type", "alpha", "random_row_column_effects", "l1_ratio",
    "quantile", "hinge_threshold", "hinge_n_boot", "huber_t", "lasso_n_boot",
    "lasso_selection_threshold", "cov_type",
])
def test_every_model_setting_is_typed_tooltipped_categorised_and_defaulted(key):
    """Any of the four missing makes the setting unreachable from a GUI.

    ``regression_type`` itself was missing from ``expected_types``, and
    ``check_settings`` DROPS a key it cannot type - so the Tk panel discarded
    whichever model the user picked and the defaults builder restored 'ols'.
    A run configured as 'mixed' fitted OLS and wrote it to results/.../ols/.
    """
    import spacr.settings as S

    assert key in S.expected_types, f"{key} has no expected_types entry"
    assert key in S.tooltips, f"{key} has no tooltip"
    categorised = {k for keys in S.categories.values() for k in keys}
    assert key in categorised, f"{key} is in no settings category"
    assert key in S.get_perform_regression_default_settings({}), \
        f"{key} has no default"


def test_check_settings_keeps_the_chosen_regression_type():
    """The Tk panel's own coercion must not drop the model choice."""
    from spacr.settings import check_settings, expected_types

    class _Var:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    vars_dict = {
        "regression_type": ("l", None, _Var("mixed"), None),
        "alpha": ("l", None, _Var("auto"), None),
        "l1_ratio": ("l", None, _Var("0.25"), None),
        "quantile": ("l", None, _Var("0.9"), None),
        "random_row_column_effects": ("l", None, _Var("True"), None),
    }
    settings, errors = check_settings(vars_dict, expected_types)

    assert not errors, errors
    assert settings["regression_type"] == "mixed"
    assert settings["alpha"] == "auto"
    assert settings["l1_ratio"] == 0.25
    assert settings["quantile"] == 0.9
    assert settings["random_row_column_effects"] is True


# ---------------------------------------------------------------------------
# The fixes the sweep above depends on
# ---------------------------------------------------------------------------

def test_scale_variables_keeps_a_constant_column(tmp_path):
    """MinMaxScaler maps the intercept to zeros; that deleted it silently."""
    from spacr.ml import scale_variables

    X = pd.DataFrame({"Intercept": 1.0, "a": [1.0, 2.0, 3.0, 4.0]})
    y = np.array([[0.0], [1.0], [2.0], [3.0]])
    X_scaled, y_scaled = scale_variables(X, y)

    assert np.allclose(X_scaled["Intercept"], 1.0)
    assert np.isclose(X_scaled["a"].min(), 0.0)
    assert np.isclose(X_scaled["a"].max(), 1.0)
    assert y_scaled.max() <= 1.0 + 1e-9


def test_gene_fraction_counts_each_grna_once_in_the_cross_join():
    """agg_type=None joins gRNAs against CELLS; the gene share must not inflate.

    With one row per (well, gRNA, cell), summing 'fraction' over the frame
    multiplies every gene's share by that well's cell count - and wells do not
    all hold the same number of cells, so the inflation differs per well.
    """
    from spacr.ml import check_and_clean_data

    rows = []
    for well, n_cells in (("p1_r1_c1", 3), ("p1_r1_c2", 7)):
        for grna, gene, fraction in (("gA_1", "gA", 0.2), ("gA_2", "gA", 0.3),
                                     ("gB_1", "gB", 0.5)):
            for _ in range(n_cells):
                rows.append({"prc": well, "grna": grna, "gene": gene,
                             "fraction": fraction, "prediction": 0.4,
                             "plateID": "p1",
                             "rowID": well.split("_")[1],
                             "columnID": well.split("_")[2]})
    out = check_and_clean_data(pd.DataFrame(rows), "prediction")

    shares = out.groupby(["prc", "gene"], observed=True)["gene_fraction"].unique()
    for (well, gene), values in shares.items():
        assert len(values) == 1
        expected = 0.5 if gene == "gA" else 0.5
        assert np.isclose(values[0], expected), (well, gene, values)


def test_mixed_model_refuses_a_single_plate_instead_of_returning_zeros(
        tmp_path, stubs):
    """One plate leaves nothing for a random intercept to describe.

    Grouping on the well - which is what this used to do - fitted, wrote
    results.csv, and put ~1e-11 with p ~ 1 in every coefficient, because the
    random intercept sat at the same level as the covariates.
    """
    from spacr.ml import perform_regression

    score, count = write_screen(tmp_path, plates=("plate1",))
    settings = settings_for(score, count, regression_type="mixed")
    with pytest.raises(ValueError, match="needs at least two clusters"):
        perform_regression(settings)


def test_a_penalty_that_zeroes_every_coefficient_is_an_error(tmp_path, stubs):
    """An all-zero lasso reaches the user as "no hits", which is a real result.

    The default alpha=1 does exactly this to a fraction-scale design, so the
    stock settings used to produce an empty hit list rather than a complaint.
    """
    from spacr.ml import perform_regression

    score, count = write_screen(tmp_path)
    settings = settings_for(score, count, regression_type="lasso", alpha=1.0)
    with pytest.raises(ValueError, match="shrank all .* to exactly zero"):
        perform_regression(settings)


def test_wls_actually_weights_by_the_cell_count():
    """The weights have to reach the fit, not just the function signature."""
    from spacr.ml import regression_model

    # Two wells disagree about the slope; the one with 100x the cells wins.
    X = pd.DataFrame({"Intercept": 1.0, "x": [0.0, 1.0, 0.0, 1.0]})
    y = pd.Series([0.0, 1.0, 0.0, -1.0])
    heavy = np.array([1.0, 1.0, 100.0, 100.0])

    ols = regression_model(X, y, regression_type="ols")
    wls = regression_model(X, y, regression_type="wls", weights=heavy)

    assert np.isclose(float(ols.params["x"]), 0.0)          # the two cancel
    assert float(wls.params["x"]) < -0.9                    # the heavy well wins


def test_rlm_resists_an_outlier_that_drags_ols():
    """Robust regression is only worth offering if it is actually robust."""
    from spacr.ml import regression_model

    rng = np.random.default_rng(3)
    n = 60
    x = rng.uniform(0, 1, n)
    y = 2.0 * x + rng.normal(0, 0.05, n)
    y[0] = 50.0                                             # one runaway well
    X = pd.DataFrame({"Intercept": 1.0, "x": x})

    ols = regression_model(X, pd.Series(y), regression_type="ols")
    rlm = regression_model(X, pd.Series(y), regression_type="rlm")

    assert abs(float(rlm.params["x"]) - 2.0) < 0.2
    assert abs(float(ols.params["x"]) - 2.0) > abs(float(rlm.params["x"]) - 2.0)
    # 'huber' is the same estimator under the name the request used.
    huber = regression_model(X, pd.Series(y), regression_type="huber")
    assert np.allclose(huber.params.values, rlm.params.values)


def test_wls_without_cell_counts_refuses_to_masquerade_as_ols():
    """WLS with unit weights IS OLS, and would be labelled 'wls' everywhere."""
    from spacr.ml import regression_model

    rng = np.random.default_rng(2)
    n = 40
    X = pd.DataFrame({"Intercept": 1.0, "x": rng.normal(0, 1, n)})
    y = pd.Series(rng.normal(0, 1, n))
    with pytest.raises(ValueError, match="needs per-well weights"):
        regression_model(X, y, regression_type="wls")


# ---------------------------------------------------------------------------
# Job D - the spaCRPower horseshoe model
# ---------------------------------------------------------------------------

def test_horseshoe_is_reachable_as_a_regression_type():
    """'horseshoe' is offered by the dispatcher and by the settings surface."""
    import spacr.settings as S
    from spacr.ml import REGRESSION_TYPES

    assert "horseshoe" in REGRESSION_TYPES
    assert "horseshoe" in S.tooltips["regression_type"]


def test_horseshoe_says_which_module_it_needs_when_it_is_missing(
        tmp_path, stubs, monkeypatch):
    """Absent spacr.power_model, the branch names the module, not a KeyError."""
    import sys

    from spacr.ml import perform_regression

    monkeypatch.setitem(sys.modules, "spacr.power_model", None)
    score, count = write_screen(tmp_path, response="binary")
    settings = settings_for(score, count, regression_type="horseshoe")
    with pytest.raises(ImportError, match="spacr.power_model"):
        perform_regression(settings)


@pytest.mark.slow
def test_horseshoe_fits_through_power_model_and_finds_the_planted_gene(
        tmp_path, stubs):
    """The real spaCRPower model, reached as an ordinary regression choice.

    The response reaching it is the per-well POSITIVE COUNT and the exposure
    is the well's imaged cell count, i.e. spaCRPower's
    ``Npositive ~ ... + offset(log(Ntotal))``. The horseshoe is doing variable
    selection, so the non-hit genes come back shrunk to ~0 rather than merely
    small - that separation is the thing worth testing.
    """
    pytest.importorskip("spacr.power_model")
    from spacr.ml import perform_regression

    score, count = write_screen(tmp_path, response="binary")
    settings = settings_for(score, count, regression_type="horseshoe")
    out = perform_regression(settings)

    assert_recovers_the_planted_gene(out["results"], "horseshoe")
    coefficients = gene_coefficients(out["results"])
    others = [abs(v) for g, v in coefficients.items() if g != HIT_GENE]
    assert abs(coefficients[HIT_GENE]) > 5 * max(others), (
        "the horseshoe did not separate the hit from the nulls: "
        f"{coefficients}")
    # p_value here is a posterior tail mass, so it must still be a probability.
    assert out["results"]["p_value"].between(0.0, 1.0).all()


def test_horseshoe_drops_terms_it_could_not_identify(tmp_path, stubs):
    """A constant column is confounded with the model's own intercept.

    Reporting it as a shrunk-to-zero coefficient would read as "tested, null";
    it was never estimable, so it is not in the table at all.
    """
    pytest.importorskip("spacr.power_model")
    from spacr.ml import perform_regression

    score, count = write_screen(tmp_path, response="binary")
    out = perform_regression(settings_for(score, count,
                                          regression_type="horseshoe"))
    assert "Intercept" not in set(out["results"]["feature"])
    assert out["results"]["coefficient"].notna().all()


def test_horseshoe_refuses_a_run_with_no_cell_counts():
    """offset(log(Ntotal)) has no meaning without Ntotal."""
    from spacr.ml import regression_model

    X = pd.DataFrame({"Intercept": 1.0, "x": [0.1, 0.5, 0.9, 0.3]})
    y = pd.Series([1.0, 4.0, 9.0, 2.0])
    with pytest.raises(ValueError, match="needs the per-well cell count"):
        regression_model(X, y, regression_type="horseshoe")


# ---------------------------------------------------------------------------
# The count families and the default transform
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("regression_type", ["poisson", "horseshoe"])
def test_a_count_model_does_not_get_its_response_logged(regression_type):
    """``log(count)`` is not a count, and every entry point defaulted to it.

    ``transform`` defaults to ``'log'`` because screen responses are fractions
    and skew hard. ``ml.process_scores`` already knows the count families are
    different -- it overrides ``agg_type`` to take the well's SUM for exactly
    these two -- and then transforms that sum like any other response. The
    integer becomes a float and ``_validate_poisson_response`` refuses it with
    "Poisson regression requires integer count data", at the very end of a run
    that had already read both CSVs.

    Since all three dispatchers build their settings from this one function,
    the effect was that neither count family could be started from Tk, Qt or
    the CLI without knowing to turn ``transform`` off by hand.
    """
    from spacr.settings import get_perform_regression_default_settings

    settings = get_perform_regression_default_settings(
        {"regression_type": regression_type})

    assert settings["transform"] is None, (
        f"{regression_type} is fitted as Npositive ~ ... + "
        f"offset(log(Ntotal)); it already has a log link, so transforming the "
        f"response logs it twice and stops it being integral")


@pytest.mark.parametrize("regression_type", ["poisson", "horseshoe"])
def test_an_explicit_log_is_overridden_for_a_count_model_too(regression_type):
    """Asking for it directly cannot produce a response the model refuses.

    The same shape as the quantile rule above it, which forces ``agg_type`` to
    None however the caller left it: a model choice that decides how the
    response must be prepared wins over the preparation setting, because the
    alternative is a run that dies at the end with a message about the data
    rather than about the combination.
    """
    from spacr.settings import get_perform_regression_default_settings

    settings = get_perform_regression_default_settings(
        {"regression_type": regression_type, "transform": "log"})

    assert settings["transform"] is None


def test_a_continuous_model_keeps_the_log_default():
    """The rule is narrow: only the two count families lose the transform."""
    from spacr.settings import get_perform_regression_default_settings

    assert get_perform_regression_default_settings(
        {"regression_type": "ols"})["transform"] == "log"
    assert get_perform_regression_default_settings(
        {"regression_type": "glm", "transform": "sqrt"})["transform"] == "sqrt"
