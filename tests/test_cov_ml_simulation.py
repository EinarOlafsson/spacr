"""Coverage for the simulation / coefficient / distribution block of spacr.ml.

Targets ``spacr/ml.py`` lines 308-575:

    * ``minimum_cell_simulation``    -- sampling curve + elbow detection + PDF
    * ``process_model_coefficients`` -- coefficient tables per regression type
    * ``check_distribution``         -- model recommendation from y
    * ``pick_glm_family_and_link``   -- GLM family/link selection from y

Everything here is CPU-only, offline and deterministic (fixed seeds). The
statistical branches are reached with real samples drawn from the relevant
distributions, not by monkeypatching scipy.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _no_leaked_figures():
    """Never let Agg figures accumulate between tests."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_scores(path, wells, n_cells, seed, prc_value=None):
    """Write a per-object score CSV shaped like a spacr 'score_data' table.

    ``wells`` is a list of (rowID, columnID) tuples; every well gets
    ``n_cells`` rows with a uniform score in ``pred``.
    """
    rng = np.random.default_rng(seed)
    records = []
    for row_id, col_id in wells:
        for score in rng.uniform(0.1, 0.9, n_cells):
            records.append({"rowID": row_id, "columnID": col_id,
                            "pred": float(score)})
    df = pd.DataFrame(records)
    if prc_value is not None:
        df["prc"] = prc_value
    df.to_csv(path, index=False)
    return df


def _settings(tmp_path, score_data, tolerance, min_cells_per_well=None,
              count_name="counts.csv"):
    return {
        "score_data": score_data,
        "dependent_variable": "pred",
        "tolerance": tolerance,
        "min_cells_per_well": min_cells_per_well,
        "count_data": [str(tmp_path / count_name)],
    }


def _coef_dataframe(seed=0, n_per_group=15):
    """A tidy frame whose patsy design gives bracketed gRNA feature names."""
    rng = np.random.default_rng(seed)
    levels = ["aaa_base", "nc_g1", "pc_g1", "ctl_g1", "oth_g1"]
    grna = np.repeat(levels, n_per_group)
    rows = np.array(["A", "B"] * (len(grna) // 2 + 1))[: len(grna)]
    offsets = {"aaa_base": 0.0, "nc_g1": 0.1, "pc_g1": 0.6,
               "ctl_g1": 0.05, "oth_g1": 0.3}
    score = np.array([offsets[g] for g in grna]) + rng.normal(0, 0.05, len(grna))
    return pd.DataFrame({"score": score, "grna": grna, "rowID": rows})


def _fit_formula_model(df):
    import statsmodels.formula.api as smf
    return smf.ols("score ~ grna + rowID", data=df).fit()


# ===========================================================================
# minimum_cell_simulation
# ===========================================================================

def case_minimum_cell_simulation_returns_elbow_and_writes_pdf(tmp_path):
    """Large tolerance -> the very first sample size clears the threshold."""
    _seen = []
    from spacr.ml import minimum_cell_simulation

    csv = tmp_path / "scores.csv"
    _write_scores(csv, [("A", 1), ("A", 2), ("B", 1)], n_cells=40, seed=1)

    # A bare string must be promoted to a one-element list in-place.
    settings = _settings(tmp_path, str(csv), tolerance=100)
    out = minimum_cell_simulation(settings, num_repeats=3, increment=10)

    _seen.append(bool(
        settings["score_data"] == [str(csv)]))
    # tolerance=100 -> threshold is 100% of the well mean, so sample size 2
    # (the first point of np.arange(2, max_cells + 1, increment)) qualifies.
    _seen.append(keep(int(out)))

    pdf = tmp_path / "results" / "cell_min_threshold.pdf"
    _seen.append(bool(
        pdf.is_file() and pdf.stat().st_size > 0))

    # DRAWN IN PYQTGRAPH, so there is no pyplot figure to read the axis
    # labels off -- and that absence is itself the assertion. The plot is
    # deleted as soon as it has been written, which is the point of a
    # renderer a run drives rather than a figure it leaves lying about, so
    # what the chart CONTAINS is asserted where it can be: directly, on the
    # drawing methods, in test_the_sweep_chart_marks_where_the_threshold_is.
    _seen.append(keep(plt.get_fignums()))
    return _seen


def case_minimum_cell_simulation_falls_back_to_last_point(tmp_path):
    """An unreachable tolerance leaves elbow_df empty -> last sample size."""
    _seen = []
    from spacr.ml import minimum_cell_simulation

    csv = tmp_path / "scores.csv"
    _write_scores(csv, [("A", 1), ("A", 2)], n_cells=40, seed=2)

    settings = _settings(tmp_path, [str(csv)], tolerance=1e-9)  # float branch
    out = minimum_cell_simulation(settings, num_repeats=3, increment=10)

    # np.arange(2, 41, 10) -> [2, 12, 22, 32]; nothing beats a ~0 threshold,
    # so the fallback is the last point of the summary curve.
    _seen.append(keep(int(out)))
    return _seen


def case_minimum_cell_simulation_min_cells_per_well_overrides_marker(tmp_path,
                                                                 monkeypatch):
    """min_cells_per_well moves the plotted marker but not the returned elbow."""
    _seen = []
    from spacr.ml import minimum_cell_simulation

    csv = tmp_path / "scores.csv"
    _write_scores(csv, [("A", 1), ("B", 2)], n_cells=30, seed=3)

    # count_data with no directory component -> dst == '' -> 'results' is
    # created relative to the CWD.
    monkeypatch.chdir(tmp_path)
    settings = _settings(tmp_path, str(csv), tolerance=0.9, min_cells_per_well=15)
    settings["count_data"] = ["counts.csv"]

    from spacr import ml as ml_module

    marks = []
    real = ml_module._draw_the_cell_count_sweep
    monkeypatch.setattr(
        ml_module, "_draw_the_cell_count_sweep",
        lambda summary, mark, path: (marks.append(float(mark))
                                     or real(summary, mark, path)))

    out = minimum_cell_simulation(settings, num_repeats=3, increment=10)

    _seen.append(keep(int(out)))
    _seen.append(bool(
        (tmp_path / "results" / "cell_min_threshold.pdf").is_file()))

    # The marker is at min_cells_per_well rather than at the elbow -- asserted
    # on the value handed to the drawing helper, which is the decision this
    # test is about; the drawing itself is covered directly below.
    _seen.append(keep(plt.get_fignums()))
    _seen.append(keep(marks))
    return _seen


def case_minimum_cell_simulation_returns_elbow_when_renderer_refuses(
        tmp_path, monkeypatch):
    """Numerical output does not disappear when no Qt renderer is usable."""
    _seen = []
    from spacr import ml as ml_module

    csv = tmp_path / "scores.csv"
    _write_scores(csv, [("A", 1), ("B", 2)], n_cells=30, seed=31)
    settings = _settings(tmp_path, str(csv), tolerance=100)

    monkeypatch.setattr(
        ml_module, "_draw_the_cell_count_sweep",
        lambda summary, mark, path: None,
    )

    out = ml_module.minimum_cell_simulation(
        settings, num_repeats=3, increment=10)

    _seen.append(keep(int(out)))
    _seen.append(bool(
        not (tmp_path / "results" / "cell_min_threshold.pdf").exists()))
    return _seen


def case_minimum_cell_simulation_concatenates_multiple_score_files(tmp_path):
    """Every score_data file is read and tagged with its own plateID."""
    _seen = []
    from spacr.ml import minimum_cell_simulation

    small = tmp_path / "plate1.csv"
    big = tmp_path / "plate2.csv"
    _write_scores(small, [("A", 1), ("A", 2)], n_cells=30, seed=4)
    _write_scores(big, [("C", 1), ("C", 2)], n_cells=120, seed=5)

    settings = _settings(tmp_path, [str(small), str(big)], tolerance=1e-9)
    out = minimum_cell_simulation(settings, num_repeats=2, increment=20)

    # Fallback = largest sample size in the pooled curve. np.arange(2, 121, 20)
    # ends at 102, which can only come from the second (plate2) file.
    _seen.append(keep(int(out)))
    return _seen


def case_minimum_cell_simulation_uses_existing_prc_column(tmp_path):
    """A pre-existing prc column is respected instead of being rebuilt."""
    _seen = []
    from spacr.ml import minimum_cell_simulation

    csv = tmp_path / "scores.csv"
    # 3 nominal wells, but every row shares one prc -> a single 60-cell well.
    _write_scores(csv, [("A", 1), ("A", 2), ("A", 3)], n_cells=20, seed=6,
                  prc_value="plate1_A_1")

    settings = _settings(tmp_path, str(csv), tolerance=1e-9)
    out = minimum_cell_simulation(settings, num_repeats=2, increment=20)

    # One 60-cell well -> np.arange(2, 61, 20) = [2, 22, 42] -> fallback 42.
    # Had prc been rebuilt from rowID/columnID the wells would hold 20 cells
    # each and the answer would be 2.
    _seen.append(keep(int(out)))
    return _seen


def case_minimum_cell_simulation_rejects_non_numeric_tolerance(tmp_path):
    _seen = []
    from spacr.ml import minimum_cell_simulation

    csv = tmp_path / "scores.csv"
    _write_scores(csv, [("A", 1)], n_cells=12, seed=7)

    settings = _settings(tmp_path, str(csv), tolerance="2")
    with pytest.raises(ValueError, match="Tolerance must be an integer") as _exc1:
        minimum_cell_simulation(settings, num_repeats=2, increment=10)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


# ===========================================================================
# process_model_coefficients
# ===========================================================================

def case_process_model_coefficients_ols_labels_conditions_and_drops_row():
    _seen = []
    from spacr.ml import process_model_coefficients

    df = _coef_dataframe()
    model = _fit_formula_model(df)

    coef_df = process_model_coefficients(
        model, "ols", X=None, y=None,
        nc="nc_g1", pc="pc_g1", controls=["ctl_g1"],
    )

    features = list(coef_df["feature"])
    _seen.append((("absent", 'rowID[T.B]') if 'rowID[T.B]' not in (features) else ("present", 'rowID[T.B]')))
    _seen.append(keep(set(features)))

    by_feature = coef_df.set_index("feature")
    _seen.append(keep(by_feature.loc["grna[T.nc_g1]", "condition"]))
    _seen.append(keep(by_feature.loc["grna[T.pc_g1]", "condition"]))
    _seen.append(keep(by_feature.loc["grna[T.ctl_g1]", "condition"]))
    _seen.append(keep(by_feature.loc["grna[T.oth_g1]", "condition"]))
    _seen.append(keep(by_feature.loc["Intercept", "condition"]))

    # The 'T.' prefix is stripped off the extracted gRNA name.
    _seen.append(keep(by_feature.loc["grna[T.oth_g1]", "grna"]))
    _seen.append(bool(
        pd.isna(by_feature.loc["Intercept", "grna"])))

    # Coefficients/p-values are taken straight off the fitted model.
    _seen.append(bool(
        by_feature.loc["grna[T.pc_g1]", "coefficient"] == pytest.approx(
        model.params["grna[T.pc_g1]"]
    )))
    np.testing.assert_allclose(
        coef_df["-log10(p_value)"], -np.log10(coef_df["p_value"])
    )
    return _seen


@pytest.mark.parametrize("regression_type",
                         ["glm", "logit", "probit", "quasi_binomial"])
def test_process_model_coefficients_shared_statsmodels_branch(regression_type):
    """All statsmodels-style types produce the same 3-column core table."""
    from spacr.ml import process_model_coefficients

    model = _fit_formula_model(_coef_dataframe(seed=1))
    coef_df = process_model_coefficients(
        model, regression_type, X=None, y=None,
        nc="nc_g1", pc="pc_g1", controls=[],
    )
    assert list(coef_df.columns) == [
        "feature", "coefficient", "p_value", "-log10(p_value)", "grna",
        "condition",
    ]
    assert len(coef_df) == 5
    assert not coef_df["feature"].str.contains("row").any()


def case_process_model_coefficients_beta_adds_wald_statistics():
    _seen = []
    from spacr.ml import process_model_coefficients
    import scipy.stats as st

    model = _fit_formula_model(_coef_dataframe(seed=2))
    coef_df = process_model_coefficients(
        model, "beta", X=None, y=None,
        nc="nc_g1", pc="pc_g1", controls=["ctl_g1"],
    )

    _seen.append(bool(
        {"std_err", "wald_stat"} <= set(coef_df.columns)))
    kept = coef_df.set_index("feature")
    for feature in kept.index:
        expected_wald = model.params[feature] / model.bse[feature]
        assert kept.loc[feature, "wald_stat"] == pytest.approx(expected_wald)
        assert kept.loc[feature, "std_err"] == pytest.approx(model.bse[feature])
        assert kept.loc[feature, "p_value"] == pytest.approx(
            2 * (1 - st.norm.cdf(abs(expected_wald)))
        )
    _seen.append((("absent", 'rowID[T.B]') if 'rowID[T.B]' not in (kept.index) else ("present", 'rowID[T.B]')))
    return _seen


@pytest.mark.parametrize("estimator", ["ridge", "lasso"])
def test_process_model_coefficients_sklearn_branch(estimator):
    from sklearn.linear_model import Lasso, Ridge

    from spacr.ml import process_model_coefficients

    rng = np.random.default_rng(3)
    columns = ["grna[T.nc_g1]", "grna[T.pc_g1]", "grna[T.ctl_g1]", "rowID[T.B]"]
    X = pd.DataFrame(rng.integers(0, 2, size=(60, 4)).astype(float),
                     columns=columns)
    y = 0.3 * X["grna[T.pc_g1]"] + rng.normal(0, 0.05, 60)

    model = (Ridge(alpha=0.1) if estimator == "ridge" else Lasso(alpha=0.001))
    model.fit(X, y)

    coef_df = process_model_coefficients(
        model, estimator, X=X, y=y,
        nc="nc_g1", pc="pc_g1", controls=["ctl_g1"],
    )

    assert list(coef_df["feature"]) == columns[:3]   # rowID column filtered out
    np.testing.assert_allclose(coef_df["coefficient"].to_numpy(),
                               np.asarray(model.coef_).ravel()[:3])
    assert list(coef_df["condition"]) == ["nc", "pc", "control"]
    assert coef_df["p_value"].between(0.0, 1.0).all()


def case_process_model_coefficients_rejects_unknown_regression_type():
    _seen = []
    from spacr.ml import process_model_coefficients

    model = _fit_formula_model(_coef_dataframe(seed=4))
    # 'quantile' used to be the example of an unsupported type here; it is a
    # real backend now, so the rejection is tested with a name that is not.
    with pytest.raises(ValueError, match="Unsupported regression type: banana") as _exc1:
        process_model_coefficients(model, "banana", X=None, y=None,
                                   nc="nc_g1", pc="pc_g1", controls=[])
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


# ===========================================================================
# check_distribution
# ===========================================================================

def case_check_distribution_binary_returns_logit(capsys):
    _seen = []
    from spacr.ml import check_distribution

    _seen.append(keep(check_distribution(np.array([0, 1, 1, 0, 1]))))
    _seen.append(('binary' if 'binary' in (capsys.readouterr().out.lower()) else ("missing", 'binary')))
    return _seen


def case_check_distribution_strict_interior_returns_beta():
    _seen = []
    from spacr.ml import check_distribution

    y = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
    _seen.append(keep(check_distribution(y)))
    return _seen


def case_check_distribution_strict_but_near_zero_returns_quasi_binomial():
    _seen = []
    from spacr.ml import check_distribution

    y = np.array([1e-9, 0.4, 0.5, 0.6])       # inside (0, 1) but within epsilon
    _seen.append(keep(check_distribution(y)))
    return _seen


def case_check_distribution_inclusive_bounds_returns_quasi_binomial():
    _seen = []
    from spacr.ml import check_distribution

    _seen.append(keep(check_distribution(np.array([0.0, 0.4, 0.7, 1.0]))))
    return _seen


def case_check_distribution_normal_returns_ols():
    _seen = []
    from spacr.ml import check_distribution

    y = np.random.default_rng(11).normal(50.0, 5.0, 300)
    _seen.append(keep(check_distribution(y)))
    return _seen


def case_check_distribution_beta_shaped_out_of_range_returns_quasi_binomial():
    """Beta(2,2)-shaped data with one out-of-range point hits the KS branch."""
    _seen = []
    from spacr.ml import check_distribution

    rng = np.random.default_rng(8)
    y = np.append(rng.beta(2, 2, 400), 1.05)

    # Sanity: the sample must genuinely bypass the three bounded shortcuts and
    # fail the normality test, otherwise the KS branch is never reached.
    _seen.append(bool(
        not ((y >= 0).all() and (y <= 1).all())))
    from scipy import stats as _stats
    _seen.append(bool(
        _stats.normaltest(y).pvalue < 0.05))
    _seen.append(bool(
        _stats.kstest(y, "beta", args=(2, 2)).pvalue > 0.05))

    _seen.append(keep(check_distribution(y)))
    return _seen


def case_check_distribution_beta_shaped_with_wide_epsilon_returns_beta():
    """The KS-beta branch returns 'beta' only when epsilon widens the window.

    Reaching the KS test requires a value outside [0, 1], which the default
    epsilon always flags as a boundary value; a negative epsilon is the only
    way to exercise the non-boundary side of that branch.
    """
    _seen = []
    from spacr.ml import check_distribution

    rng = np.random.default_rng(8)
    y = np.append(rng.beta(2, 2, 400), 1.05)

    _seen.append(keep(check_distribution(y, epsilon=-0.5)))
    return _seen


def case_check_distribution_heavy_tailed_returns_glm():
    _seen = []
    from spacr.ml import check_distribution

    y = np.random.default_rng(12).lognormal(6.0, 1.0, 200)
    _seen.append(keep(check_distribution(y)))
    return _seen


# ===========================================================================
# pick_glm_family_and_link
# ===========================================================================

def case_pick_glm_family_binary_uses_binomial_logit():
    _seen = []
    import statsmodels.api as sm

    from spacr.ml import pick_glm_family_and_link

    fam = pick_glm_family_and_link(np.array([0, 1, 1, 0]))
    _seen.append(bool(
        isinstance(fam, sm.families.Binomial)))
    _seen.append(bool(
        isinstance(fam.link, sm.families.links.Logit)))
    return _seen


def case_pick_glm_family_strict_proportions_use_binomial_logit(capsys):
    """A proportion inside (0, 1) gets the same family as one that touches 0.

    This used to raise "Use BetaModel for this data; GLM is not applicable",
    which made regression_type='glm' unusable on spaCR's most common response
    -- a per-well mean score -- while the very next branch fitted exactly this
    family as soon as one well landed on 0.0 or 1.0. Beta regression is still
    recommended, in the printed message, and regression_type='beta' is how the
    recommendation is taken.
    """
    _seen = []
    import statsmodels.api as sm

    from spacr.ml import pick_glm_family_and_link

    fam = pick_glm_family_and_link(np.array([0.1, 0.4, 0.9]))
    _seen.append(bool(
        isinstance(fam, sm.families.Binomial)))
    _seen.append(bool(
        isinstance(fam.link, sm.families.links.Logit)))
    _seen.append(("consider regression_type='beta'" if "consider regression_type='beta'" in (capsys.readouterr().out) else ("missing", "consider regression_type='beta'")))
    return _seen


def case_pick_glm_family_bounded_with_zero_uses_binomial_logit():
    _seen = []
    import statsmodels.api as sm

    from spacr.ml import pick_glm_family_and_link

    fam = pick_glm_family_and_link(np.array([0.0, 0.25, 0.5, 1.0]))
    _seen.append(bool(
        isinstance(fam, sm.families.Binomial)))
    _seen.append(bool(
        isinstance(fam.link, sm.families.links.Logit)))
    return _seen


def case_pick_glm_family_normal_uses_gaussian_identity():
    _seen = []
    import statsmodels.api as sm

    from spacr.ml import pick_glm_family_and_link

    y = np.random.default_rng(13).normal(20.0, 3.0, 300)
    fam = pick_glm_family_and_link(y)
    _seen.append(bool(
        isinstance(fam, sm.families.Gaussian)))
    _seen.append(bool(
        isinstance(fam.link, sm.families.links.Identity)))
    return _seen


def case_pick_glm_family_counts_use_poisson_log():
    _seen = []
    import statsmodels.api as sm

    from spacr.ml import pick_glm_family_and_link

    y = np.random.default_rng(14).poisson(1.5, 200).astype(float)
    fam = pick_glm_family_and_link(y)
    _seen.append(bool(
        isinstance(fam, sm.families.Poisson)))
    _seen.append(bool(
        isinstance(fam.link, sm.families.links.Log)))
    return _seen


def case_pick_glm_family_inverse_gaussian_sample():
    _seen = []
    import statsmodels.api as sm
    from scipy.stats import invgauss

    from spacr.ml import pick_glm_family_and_link

    y = invgauss.rvs(mu=1.0, size=300, random_state=6)
    _seen.append(bool(
        (y > 0).all() and not np.all(y.astype(int) == y)))

    fam = pick_glm_family_and_link(y)
    _seen.append(bool(
        isinstance(fam, sm.families.InverseGaussian)))
    _seen.append(bool(
        isinstance(fam.link, sm.families.links.Log)))
    return _seen


def case_pick_glm_family_overdispersed_uses_negative_binomial():
    _seen = []
    import statsmodels.api as sm

    from spacr.ml import pick_glm_family_and_link

    # Positive, continuous, strongly skewed and nowhere near invgauss(mu=1).
    y = np.random.default_rng(15).lognormal(6.0, 1.0, 200)
    fam = pick_glm_family_and_link(y)
    _seen.append(bool(
        isinstance(fam, sm.families.NegativeBinomial)))
    _seen.append(bool(
        isinstance(fam.link, sm.families.links.Log)))
    return _seen


def case_pick_glm_family_negative_values_fall_back_to_gaussian():
    _seen = []
    import statsmodels.api as sm

    from spacr.ml import pick_glm_family_and_link

    rng = np.random.default_rng(16)
    y = np.concatenate([rng.lognormal(0.0, 1.0, 199), [-5.0]])
    fam = pick_glm_family_and_link(y)
    _seen.append(bool(
        isinstance(fam, sm.families.Gaussian)))
    _seen.append(bool(
        isinstance(fam.link, sm.families.links.Identity)))
    return _seen


def case_the_sweep_chart_marks_where_the_threshold_is(qapp):
    """The chart itself: a curve inside its spread band, and one line.

    Asserted on the drawing rather than through the whole simulation,
    because the plot a run builds is written and deleted in the same
    breath -- which is what a renderer a pipeline drives should do, and
    leaves nothing for a test to reach into afterwards.
    """
    _seen = []
    import numpy as np
    import pyqtgraph as pg

    from spacr.qt.widgets.fast_plots import FastPlot

    sizes = np.arange(2, 40, 4, dtype=float)
    middle = 1.0 / sizes
    spread = middle * 0.2
    plot = FastPlot(title="sweep", x_label="Sample size",
                    y_label="Mean absolute difference")
    try:
        assert plot.add_curve(sizes, middle, low=middle - spread,
                              high=middle + spread) == len(sizes)
        plot.add_line(x=15.0, label="minimum cell count")
        marks = [item for item in plot.plot.items()
                 if isinstance(item, pg.InfiniteLine)]
        assert len(marks) == 1
        assert float(marks[0].value()) == pytest.approx(15.0)
        bands = [item for item in plot.plot.items()
                 if isinstance(item, pg.FillBetweenItem)]
        assert len(bands) == 1, "the spread band was not drawn"
        assert plot.curve_frame().shape == (len(sizes), 2)
    finally:
        plot.deleteLater()
    return _seen


def case_half_a_band_is_not_drawn(qapp):
    """Either edge alone is a second line pretending to be a band."""
    _seen = []
    import numpy as np
    import pyqtgraph as pg

    from spacr.qt.widgets.fast_plots import FastPlot

    xs = np.arange(6, dtype=float)
    plot = FastPlot(title="t", x_label="x", y_label="y")
    try:
        plot.add_curve(xs, xs, low=xs - 1)
        assert not [item for item in plot.plot.items()
                    if isinstance(item, pg.FillBetweenItem)]
    finally:
        plot.deleteLater()
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param(case_check_distribution_binary_returns_logit, ['logit', 'binary'], id="check_distribution_binary_returns_logit"),
    pytest.param(case_check_distribution_strict_interior_returns_beta, ['beta'], id="check_distribution_strict_interior_returns_beta"),
    pytest.param(case_check_distribution_strict_but_near_zero_returns_quasi_binomial, ['quasi_binomial'], id="check_distribution_strict_but_near_zero_returns_quasi_binomial"),
    pytest.param(case_check_distribution_inclusive_bounds_returns_quasi_binomial, ['quasi_binomial'], id="check_distribution_inclusive_bounds_returns_quasi_binomial"),
    pytest.param(case_check_distribution_normal_returns_ols, ['ols'], id="check_distribution_normal_returns_ols"),
    pytest.param(case_check_distribution_beta_shaped_out_of_range_returns_quasi_binomial, [True, True, True, 'quasi_binomial'], id="check_distribution_beta_shaped_out_of_range_returns_quasi_binomial"),
    pytest.param(case_check_distribution_beta_shaped_with_wide_epsilon_returns_beta, ['beta'], id="check_distribution_beta_shaped_with_wide_epsilon_returns_beta"),
    pytest.param(case_check_distribution_heavy_tailed_returns_glm, ['glm'], id="check_distribution_heavy_tailed_returns_glm"),
    pytest.param(case_pick_glm_family_binary_uses_binomial_logit, [True, True], id="pick_glm_family_binary_uses_binomial_logit"),
    pytest.param(case_pick_glm_family_strict_proportions_use_binomial_logit, [True, True, "consider regression_type='beta'"], id="pick_glm_family_strict_proportions_use_binomial_logit"),
    pytest.param(case_pick_glm_family_bounded_with_zero_uses_binomial_logit, [True, True], id="pick_glm_family_bounded_with_zero_uses_binomial_logit"),
    pytest.param(case_pick_glm_family_normal_uses_gaussian_identity, [True, True], id="pick_glm_family_normal_uses_gaussian_identity"),
    pytest.param(case_pick_glm_family_counts_use_poisson_log, [True, True], id="pick_glm_family_counts_use_poisson_log"),
    pytest.param(case_pick_glm_family_inverse_gaussian_sample, [True, True, True], id="pick_glm_family_inverse_gaussian_sample"),
    pytest.param(case_pick_glm_family_overdispersed_uses_negative_binomial, [True, True], id="pick_glm_family_overdispersed_uses_negative_binomial"),
    pytest.param(case_pick_glm_family_negative_values_fall_back_to_gaussian, [True, True], id="pick_glm_family_negative_values_fall_back_to_gaussian"),
    pytest.param(case_minimum_cell_simulation_returns_elbow_and_writes_pdf, [True, 2, True, []], id="minimum_cell_simulation_returns_elbow_and_writes_pdf"),
    pytest.param(case_minimum_cell_simulation_falls_back_to_last_point, [32], id="minimum_cell_simulation_falls_back_to_last_point"),
    pytest.param(case_minimum_cell_simulation_min_cells_per_well_overrides_marker, [2, True, [], [15.0]], id="minimum_cell_simulation_min_cells_per_well_overrides_marker"),
    pytest.param(case_minimum_cell_simulation_concatenates_multiple_score_files, [102], id="minimum_cell_simulation_concatenates_multiple_score_files"),
    pytest.param(case_minimum_cell_simulation_uses_existing_prc_column, [42], id="minimum_cell_simulation_uses_existing_prc_column"),
    pytest.param(case_minimum_cell_simulation_rejects_non_numeric_tolerance, ['ValueError'], id="minimum_cell_simulation_rejects_non_numeric_tolerance"),
    pytest.param(case_minimum_cell_simulation_returns_elbow_when_renderer_refuses, [2, True], id="minimum_cell_simulation_returns_elbow_when_renderer_refuses"),
    pytest.param(case_process_model_coefficients_ols_labels_conditions_and_drops_row, [("absent", 'rowID[T.B]'), {'Intercept', 'grna[T.ctl_g1]', 'grna[T.nc_g1]', 'grna[T.oth_g1]', 'grna[T.pc_g1]'}, 'nc', 'pc', 'control', 'other', 'other', 'oth_g1', True, True], id="process_model_coefficients_ols_labels_conditions_and_drops_row"),
    pytest.param(case_process_model_coefficients_beta_adds_wald_statistics, [True, ("absent", 'rowID[T.B]')], id="process_model_coefficients_beta_adds_wald_statistics"),
    pytest.param(case_process_model_coefficients_rejects_unknown_regression_type, ['ValueError'], id="process_model_coefficients_rejects_unknown_regression_type"),
    pytest.param(case_the_sweep_chart_marks_where_the_threshold_is, [], id="the_sweep_chart_marks_where_the_threshold_is"),
    pytest.param(case_half_a_band_is_not_drawn, [], id="half_a_band_is_not_drawn"),
])
def test_ml_simulation_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected



