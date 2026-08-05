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


def _settings(tmp_path, score_data, tolerance, min_cell_count=None,
              count_name="counts.csv"):
    return {
        "score_data": score_data,
        "score_column": "pred",
        "tolerance": tolerance,
        "min_cell_count": min_cell_count,
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

def test_minimum_cell_simulation_returns_elbow_and_writes_pdf(tmp_path):
    """Large tolerance -> the very first sample size clears the threshold."""
    from spacr.ml import minimum_cell_simulation

    csv = tmp_path / "scores.csv"
    _write_scores(csv, [("A", 1), ("A", 2), ("B", 1)], n_cells=40, seed=1)

    # A bare string must be promoted to a one-element list in-place.
    settings = _settings(tmp_path, str(csv), tolerance=100)
    out = minimum_cell_simulation(settings, num_repeats=3, increment=10)

    assert settings["score_data"] == [str(csv)]
    # tolerance=100 -> threshold is 100% of the well mean, so sample size 2
    # (the first point of np.arange(2, max_cells + 1, increment)) qualifies.
    assert int(out) == 2

    pdf = tmp_path / "results" / "cell_min_threshold.pdf"
    assert pdf.is_file() and pdf.stat().st_size > 0

    ax = plt.gcf().axes[0]
    assert ax.get_xlabel() == "Sample Size"
    assert ax.get_ylabel() == "Mean Absolute Difference"
    # One curve (smoothed mean) + one dashed elbow marker.
    vlines = [ln for ln in ax.lines if len(set(ln.get_xdata())) == 1]
    assert len(vlines) == 1
    assert float(vlines[0].get_xdata()[0]) == pytest.approx(2.0)


def test_minimum_cell_simulation_falls_back_to_last_point(tmp_path):
    """An unreachable tolerance leaves elbow_df empty -> last sample size."""
    from spacr.ml import minimum_cell_simulation

    csv = tmp_path / "scores.csv"
    _write_scores(csv, [("A", 1), ("A", 2)], n_cells=40, seed=2)

    settings = _settings(tmp_path, [str(csv)], tolerance=1e-9)  # float branch
    out = minimum_cell_simulation(settings, num_repeats=3, increment=10)

    # np.arange(2, 41, 10) -> [2, 12, 22, 32]; nothing beats a ~0 threshold,
    # so the fallback is the last point of the summary curve.
    assert int(out) == 32


def test_minimum_cell_simulation_min_cell_count_overrides_marker(tmp_path,
                                                                 monkeypatch):
    """min_cell_count moves the plotted marker but not the returned elbow."""
    from spacr.ml import minimum_cell_simulation

    csv = tmp_path / "scores.csv"
    _write_scores(csv, [("A", 1), ("B", 2)], n_cells=30, seed=3)

    # count_data with no directory component -> dst == '' -> 'results' is
    # created relative to the CWD.
    monkeypatch.chdir(tmp_path)
    settings = _settings(tmp_path, str(csv), tolerance=0.9, min_cell_count=15)
    settings["count_data"] = ["counts.csv"]

    out = minimum_cell_simulation(settings, num_repeats=3, increment=10)

    assert int(out) == 2                     # elbow itself is unchanged
    assert (tmp_path / "results" / "cell_min_threshold.pdf").is_file()

    ax = plt.gcf().axes[0]
    vlines = [ln for ln in ax.lines if len(set(ln.get_xdata())) == 1]
    assert [float(ln.get_xdata()[0]) for ln in vlines] == [15.0]


def test_minimum_cell_simulation_concatenates_multiple_score_files(tmp_path):
    """Every score_data file is read and tagged with its own plateID."""
    from spacr.ml import minimum_cell_simulation

    small = tmp_path / "plate1.csv"
    big = tmp_path / "plate2.csv"
    _write_scores(small, [("A", 1), ("A", 2)], n_cells=30, seed=4)
    _write_scores(big, [("C", 1), ("C", 2)], n_cells=120, seed=5)

    settings = _settings(tmp_path, [str(small), str(big)], tolerance=1e-9)
    out = minimum_cell_simulation(settings, num_repeats=2, increment=20)

    # Fallback = largest sample size in the pooled curve. np.arange(2, 121, 20)
    # ends at 102, which can only come from the second (plate2) file.
    assert int(out) == 102


def test_minimum_cell_simulation_uses_existing_prc_column(tmp_path):
    """A pre-existing prc column is respected instead of being rebuilt."""
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
    assert int(out) == 42


def test_minimum_cell_simulation_rejects_non_numeric_tolerance(tmp_path):
    from spacr.ml import minimum_cell_simulation

    csv = tmp_path / "scores.csv"
    _write_scores(csv, [("A", 1)], n_cells=12, seed=7)

    settings = _settings(tmp_path, str(csv), tolerance="2")
    with pytest.raises(ValueError, match="Tolerance must be an integer"):
        minimum_cell_simulation(settings, num_repeats=2, increment=10)


# ===========================================================================
# process_model_coefficients
# ===========================================================================

def test_process_model_coefficients_ols_labels_conditions_and_drops_row():
    from spacr.ml import process_model_coefficients

    df = _coef_dataframe()
    model = _fit_formula_model(df)

    coef_df = process_model_coefficients(
        model, "ols", X=None, y=None,
        nc="nc_g1", pc="pc_g1", controls=["ctl_g1"],
    )

    features = list(coef_df["feature"])
    assert "rowID[T.B]" not in features          # dropped by the row|column filter
    assert set(features) == {
        "Intercept", "grna[T.ctl_g1]", "grna[T.nc_g1]",
        "grna[T.oth_g1]", "grna[T.pc_g1]",
    }

    by_feature = coef_df.set_index("feature")
    assert by_feature.loc["grna[T.nc_g1]", "condition"] == "nc"
    assert by_feature.loc["grna[T.pc_g1]", "condition"] == "pc"
    assert by_feature.loc["grna[T.ctl_g1]", "condition"] == "control"
    assert by_feature.loc["grna[T.oth_g1]", "condition"] == "other"
    assert by_feature.loc["Intercept", "condition"] == "other"

    # The 'T.' prefix is stripped off the extracted gRNA name.
    assert by_feature.loc["grna[T.oth_g1]", "grna"] == "oth_g1"
    assert pd.isna(by_feature.loc["Intercept", "grna"])

    # Coefficients/p-values are taken straight off the fitted model.
    assert by_feature.loc["grna[T.pc_g1]", "coefficient"] == pytest.approx(
        model.params["grna[T.pc_g1]"]
    )
    np.testing.assert_allclose(
        coef_df["-log10(p_value)"], -np.log10(coef_df["p_value"])
    )


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


def test_process_model_coefficients_beta_adds_wald_statistics():
    from spacr.ml import process_model_coefficients
    import scipy.stats as st

    model = _fit_formula_model(_coef_dataframe(seed=2))
    coef_df = process_model_coefficients(
        model, "beta", X=None, y=None,
        nc="nc_g1", pc="pc_g1", controls=["ctl_g1"],
    )

    assert {"std_err", "wald_stat"} <= set(coef_df.columns)
    kept = coef_df.set_index("feature")
    for feature in kept.index:
        expected_wald = model.params[feature] / model.bse[feature]
        assert kept.loc[feature, "wald_stat"] == pytest.approx(expected_wald)
        assert kept.loc[feature, "std_err"] == pytest.approx(model.bse[feature])
        assert kept.loc[feature, "p_value"] == pytest.approx(
            2 * (1 - st.norm.cdf(abs(expected_wald)))
        )
    assert "rowID[T.B]" not in kept.index


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


def test_process_model_coefficients_rejects_unknown_regression_type():
    from spacr.ml import process_model_coefficients

    model = _fit_formula_model(_coef_dataframe(seed=4))
    # 'quantile' used to be the example of an unsupported type here; it is a
    # real backend now, so the rejection is tested with a name that is not.
    with pytest.raises(ValueError, match="Unsupported regression type: banana"):
        process_model_coefficients(model, "banana", X=None, y=None,
                                   nc="nc_g1", pc="pc_g1", controls=[])


# ===========================================================================
# check_distribution
# ===========================================================================

def test_check_distribution_binary_returns_logit(capsys):
    from spacr.ml import check_distribution

    assert check_distribution(np.array([0, 1, 1, 0, 1])) == "logit"
    assert "binary" in capsys.readouterr().out.lower()


def test_check_distribution_strict_interior_returns_beta():
    from spacr.ml import check_distribution

    y = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
    assert check_distribution(y) == "beta"


def test_check_distribution_strict_but_near_zero_returns_quasi_binomial():
    from spacr.ml import check_distribution

    y = np.array([1e-9, 0.4, 0.5, 0.6])       # inside (0, 1) but within epsilon
    assert check_distribution(y) == "quasi_binomial"


def test_check_distribution_inclusive_bounds_returns_quasi_binomial():
    from spacr.ml import check_distribution

    assert check_distribution(np.array([0.0, 0.4, 0.7, 1.0])) == "quasi_binomial"


def test_check_distribution_normal_returns_ols():
    from spacr.ml import check_distribution

    y = np.random.default_rng(11).normal(50.0, 5.0, 300)
    assert check_distribution(y) == "ols"


def test_check_distribution_beta_shaped_out_of_range_returns_quasi_binomial():
    """Beta(2,2)-shaped data with one out-of-range point hits the KS branch."""
    from spacr.ml import check_distribution

    rng = np.random.default_rng(8)
    y = np.append(rng.beta(2, 2, 400), 1.05)

    # Sanity: the sample must genuinely bypass the three bounded shortcuts and
    # fail the normality test, otherwise the KS branch is never reached.
    assert not ((y >= 0).all() and (y <= 1).all())
    from scipy import stats as _stats
    assert _stats.normaltest(y).pvalue < 0.05
    assert _stats.kstest(y, "beta", args=(2, 2)).pvalue > 0.05

    assert check_distribution(y) == "quasi_binomial"


def test_check_distribution_beta_shaped_with_wide_epsilon_returns_beta():
    """The KS-beta branch returns 'beta' only when epsilon widens the window.

    Reaching the KS test requires a value outside [0, 1], which the default
    epsilon always flags as a boundary value; a negative epsilon is the only
    way to exercise the non-boundary side of that branch.
    """
    from spacr.ml import check_distribution

    rng = np.random.default_rng(8)
    y = np.append(rng.beta(2, 2, 400), 1.05)

    assert check_distribution(y, epsilon=-0.5) == "beta"


def test_check_distribution_heavy_tailed_returns_glm():
    from spacr.ml import check_distribution

    y = np.random.default_rng(12).lognormal(6.0, 1.0, 200)
    assert check_distribution(y) == "glm"


# ===========================================================================
# pick_glm_family_and_link
# ===========================================================================

def test_pick_glm_family_binary_uses_binomial_logit():
    import statsmodels.api as sm

    from spacr.ml import pick_glm_family_and_link

    fam = pick_glm_family_and_link(np.array([0, 1, 1, 0]))
    assert isinstance(fam, sm.families.Binomial)
    assert isinstance(fam.link, sm.families.links.Logit)


def test_pick_glm_family_strict_proportions_use_binomial_logit(capsys):
    """A proportion inside (0, 1) gets the same family as one that touches 0.

    This used to raise "Use BetaModel for this data; GLM is not applicable",
    which made regression_type='glm' unusable on spaCR's most common response
    -- a per-well mean score -- while the very next branch fitted exactly this
    family as soon as one well landed on 0.0 or 1.0. Beta regression is still
    recommended, in the printed message, and regression_type='beta' is how the
    recommendation is taken.
    """
    import statsmodels.api as sm

    from spacr.ml import pick_glm_family_and_link

    fam = pick_glm_family_and_link(np.array([0.1, 0.4, 0.9]))
    assert isinstance(fam, sm.families.Binomial)
    assert isinstance(fam.link, sm.families.links.Logit)
    assert "consider regression_type='beta'" in capsys.readouterr().out


def test_pick_glm_family_bounded_with_zero_uses_binomial_logit():
    import statsmodels.api as sm

    from spacr.ml import pick_glm_family_and_link

    fam = pick_glm_family_and_link(np.array([0.0, 0.25, 0.5, 1.0]))
    assert isinstance(fam, sm.families.Binomial)
    assert isinstance(fam.link, sm.families.links.Logit)


def test_pick_glm_family_normal_uses_gaussian_identity():
    import statsmodels.api as sm

    from spacr.ml import pick_glm_family_and_link

    y = np.random.default_rng(13).normal(20.0, 3.0, 300)
    fam = pick_glm_family_and_link(y)
    assert isinstance(fam, sm.families.Gaussian)
    assert isinstance(fam.link, sm.families.links.Identity)


def test_pick_glm_family_counts_use_poisson_log():
    import statsmodels.api as sm

    from spacr.ml import pick_glm_family_and_link

    y = np.random.default_rng(14).poisson(1.5, 200).astype(float)
    fam = pick_glm_family_and_link(y)
    assert isinstance(fam, sm.families.Poisson)
    assert isinstance(fam.link, sm.families.links.Log)


def test_pick_glm_family_inverse_gaussian_sample():
    import statsmodels.api as sm
    from scipy.stats import invgauss

    from spacr.ml import pick_glm_family_and_link

    y = invgauss.rvs(mu=1.0, size=300, random_state=6)
    assert (y > 0).all() and not np.all(y.astype(int) == y)

    fam = pick_glm_family_and_link(y)
    assert isinstance(fam, sm.families.InverseGaussian)
    assert isinstance(fam.link, sm.families.links.Log)


def test_pick_glm_family_overdispersed_uses_negative_binomial():
    import statsmodels.api as sm

    from spacr.ml import pick_glm_family_and_link

    # Positive, continuous, strongly skewed and nowhere near invgauss(mu=1).
    y = np.random.default_rng(15).lognormal(6.0, 1.0, 200)
    fam = pick_glm_family_and_link(y)
    assert isinstance(fam, sm.families.NegativeBinomial)
    assert isinstance(fam.link, sm.families.links.Log)


def test_pick_glm_family_negative_values_fall_back_to_gaussian():
    import statsmodels.api as sm

    from spacr.ml import pick_glm_family_and_link

    rng = np.random.default_rng(16)
    y = np.concatenate([rng.lognormal(0.0, 1.0, 199), [-5.0]])
    fam = pick_glm_family_and_link(y)
    assert isinstance(fam, sm.families.Gaussian)
    assert isinstance(fam.link, sm.families.links.Identity)
