"""CPU coverage for the statistics-prep block at the top of ``spacr.ml``.

Covers the branches that the rest of the suite never reaches:

  * ``QuasiBinomial.variance`` (shadowed by statsmodels' instance attribute)
  * the ``pinv`` fallback in ``calculate_p_values`` when ``X.T @ X`` is singular
  * every branch of ``perform_mixed_model`` (no groups / low VIF / high VIF)
  * ``fit_mixed_model`` end to end, including the residual-histogram PDF
  * the defensive branches of ``check_and_clean_data``: dropped NaN rows,
    rank deficiency, a ``LinAlgError`` while computing VIF, and the
    high-VIF column drop.

Everything runs on tiny synthetic well-level frames so a full pass is
well under a second per test; MixedLM convergence warnings are expected
and harmless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _well_frame(n=40, seed=0):
    """A minimal screen-style frame with everything check_and_clean_data wants.

    ONE row per (well, gRNA), which is the shape ``perform_regression``'s own
    merge produces: ``process_reads`` emits one ``fraction`` per (well, gRNA)
    and the score table contributes one response per well. The earlier version
    of this fixture drew ``grna`` at random from two names across 40 rows, so
    the same gRNA appeared in the same well a dozen times carrying a dozen
    different fractions - a frame the pipeline cannot produce, and one in
    which "the gene's share of this well" has no defined value.
    """
    rng = np.random.default_rng(seed)
    wells = ("plate1_r1_c1", "plate1_r2_c2")
    genes = ("geneA", "geneB")
    rows = []
    for i in range(n):
        well = wells[i % len(wells)]
        gene = genes[(i // len(wells)) % len(genes)]
        _, row_id, column_id = well.split("_")
        rows.append({
            "fraction": float(rng.uniform(0.05, 0.9)),
            "prediction": float(rng.uniform(0.05, 0.9)),
            "grna": f"{gene}_g{i}",
            "gene": gene,
            "plateID": "plate1",
            "rowID": row_id,
            "columnID": column_id,
            "prc": well,
        })
    return pd.DataFrame(rows)


def _mixed_model_frame(n_plates=4, seed=0):
    """Balanced plate x row x column x gene x grna design for fit_mixed_model."""
    rng = np.random.default_rng(seed)
    recs = []
    for p in range(n_plates):
        for row in ("r1", "r2"):
            for col in ("c1", "c2"):
                for gene in ("geneA", "geneB"):
                    for gi in (1, 2):
                        recs.append({
                            "plateID": f"plate{p + 1}",
                            "rowID": row,
                            "columnID": col,
                            "prc": f"plate{p + 1}_{row}_{col}",
                            "gene": gene,
                            "grna": f"{gene}_g{gi}",
                            "fraction": float(rng.uniform(0.05, 0.6)),
                        })
    df = pd.DataFrame(recs)
    df["gene_fraction"] = df.groupby(["prc", "gene"])["fraction"].transform("sum")
    df["score"] = (0.4 * df["fraction"]
                   + 0.1 * (df["gene"] == "geneA").astype(float)
                   + rng.normal(0, 0.05, len(df)))
    return df


# ---------------------------------------------------------------------------
# QuasiBinomial
# ---------------------------------------------------------------------------

def test_quasibinomial_variance_method_scales_binomial_variance():
    """The overridden method itself multiplies mu*(1-mu) by the dispersion.

    It has to be invoked unbound because statsmodels' ``Binomial.__init__``
    puts a ``variance`` *callable object* in the instance ``__dict__``,
    which shadows the subclass method for ordinary attribute access.
    """
    from spacr.ml import QuasiBinomial

    fam = QuasiBinomial(dispersion=2.5)
    mu = np.array([0.2, 0.5, 0.8])

    # Precondition: the instance attribute is what shadows the method.
    assert "variance" in fam.__dict__

    got = QuasiBinomial.variance(fam, mu)
    assert np.allclose(got, 2.5 * mu * (1.0 - mu))
    # Dispersion 1.0 must degenerate to the plain binomial variance.
    plain = QuasiBinomial.variance(QuasiBinomial(dispersion=1.0), mu)
    assert np.allclose(plain, mu * (1.0 - mu))


def test_quasibinomial_family_applies_dispersion_when_called_normally():
    """A dispersion of 3.0 must triple the binomial variance."""
    from spacr.ml import QuasiBinomial

    fam = QuasiBinomial(dispersion=3.0)
    mu = np.array([0.25, 0.5, 0.75])
    assert np.allclose(fam.variance(mu), 3.0 * mu * (1.0 - mu))


def test_quasibinomial_variance_wrapper_scales_deriv_and_delegates_attributes():
    """statsmodels' GLM calls family.variance.deriv and reads varfunc state,
    so the dispersion wrapper has to scale the derivative and pass the rest
    of the varfunc's attribute surface straight through."""
    from spacr.ml import QuasiBinomial

    mu = np.array([0.25, 0.5, 0.75])
    plain = QuasiBinomial(dispersion=1.0)
    fam = QuasiBinomial(dispersion=4.0)

    # d/dmu [mu(1-mu)] = 1 - 2mu, scaled by the dispersion.
    assert np.allclose(fam.variance.deriv(mu), 4.0 * (1.0 - 2.0 * mu))
    assert np.allclose(fam.variance.deriv(mu), 4.0 * plain.variance.deriv(mu))
    # 'n' lives on the wrapped varfuncs.Binomial instance, not on the wrapper.
    assert 'n' not in fam.variance.__dict__
    assert fam.variance.n == plain.variance.n
    with pytest.raises(AttributeError):
        fam.variance.not_a_varfunc_attribute


def test_quasibinomial_survives_pickle_and_deepcopy():
    """statsmodels results hold on to the family, and those get pickled, so
    the variance wrapper must not turn copy/pickle's __setstate__ probe into
    a KeyError on a half-built instance."""
    import copy
    import pickle

    from spacr.ml import QuasiBinomial

    mu = np.array([0.25, 0.5, 0.75])
    fam = QuasiBinomial(dispersion=2.0)

    restored = pickle.loads(pickle.dumps(fam))
    assert np.allclose(restored.variance(mu), 2.0 * mu * (1.0 - mu))
    assert np.allclose(restored.variance.deriv(mu), 2.0 * (1.0 - 2.0 * mu))

    cloned = copy.deepcopy(fam)
    assert np.allclose(cloned.variance(mu), 2.0 * mu * (1.0 - mu))


def test_quasibinomial_dispersion_one_matches_plain_binomial_glm():
    """The only dispersion spacr constructs is 1.0, which must fit exactly
    like statsmodels' own Binomial family."""
    import statsmodels.api as sm

    from spacr.ml import QuasiBinomial

    rng = np.random.default_rng(0)
    n = 200
    X = sm.add_constant(rng.normal(size=(n, 2)))
    p = 1.0 / (1.0 + np.exp(-(X @ np.array([0.2, 0.5, -0.3]))))
    y = rng.binomial(1, p).astype(float)

    quasi = sm.GLM(y, X, family=QuasiBinomial()).fit()
    plain = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    assert np.allclose(quasi.params, plain.params)
    assert np.allclose(quasi.bse, plain.bse)

    # Dispersion 3 leaves the point estimates alone and inflates the standard
    # errors by sqrt(3) -- the whole point of a quasi-binomial fit.
    over = sm.GLM(y, X, family=QuasiBinomial(dispersion=3.0)).fit()
    assert np.allclose(over.params, plain.params)
    assert np.allclose(over.bse / plain.bse, np.sqrt(3.0))


# ---------------------------------------------------------------------------
# calculate_p_values — pinv fallback
# ---------------------------------------------------------------------------

def test_calculate_p_values_uses_pinv_when_xtx_is_singular():
    """Duplicated design columns make X.T @ X exactly singular; the helper
    must fall back to the pseudo-inverse instead of raising."""
    from sklearn.linear_model import LinearRegression
    from spacr.ml import calculate_p_values

    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 12)
    X = np.column_stack([base, base])          # rank 1, 2 columns
    y = 2.0 * base + rng.normal(0, 0.05, 12)

    # Precondition: plain inversion really does blow up here.
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.inv(X.T @ X)

    model = LinearRegression().fit(X, y)
    p = calculate_p_values(X, y, model)

    assert p.shape == (2,)
    assert np.all(np.isfinite(p))
    assert np.all((p >= 0.0) & (p <= 1.0))
    # dof = 12 - 2 - 1 = 9 > 0, so this is NOT the NaN short-circuit.
    assert not np.any(np.isnan(p))


# ---------------------------------------------------------------------------
# perform_mixed_model
# ---------------------------------------------------------------------------

def test_perform_mixed_model_requires_groups():
    from spacr.ml import perform_mixed_model

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.5, 0.1, 0.9]})
    y = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Groups must be defined"):
        perform_mixed_model(y, X, None)


def test_perform_mixed_model_low_vif_fits_plain_mixedlm(capsys):
    """Uncorrelated fixed effects -> no ridge adjustment, and the recovered
    slope on the informative column is close to the simulated 1.5."""
    from spacr.ml import perform_mixed_model

    rng = np.random.default_rng(1)
    n = 60
    X = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    groups = np.repeat(np.arange(6), 10)
    y = 1.5 * X["a"] + rng.normal(0, 0.3, n)

    result = perform_mixed_model(y, X, groups)

    out = capsys.readouterr().out
    assert "VIF:" in out
    assert "Multicollinearity detected" not in out

    assert list(result.params.index) == ["a", "b", "Group Var"]
    assert result.fe_params.shape == (2,)
    assert abs(float(result.fe_params["a"]) - 1.5) < 0.25
    assert abs(float(result.fe_params["b"])) < 0.25


def test_perform_mixed_model_high_vif_reports_and_fits_the_given_design(capsys):
    """Near-duplicate columns are REPORTED; the design is fitted unaltered.

    This test used to assert the opposite - that the fixed effects come back
    somewhere other than the simulated 1.5 / 0.0 - and called that "the
    ridge-rescaled design is a different model". It was: `X_ridge =
    ridge.coef_ * X` multiplies every column by that column's ridge
    coefficient, which is not ridge regression, and the coefficients it
    produces are effects on rescaled columns while everything downstream
    reads them as effects on the response. On a screen-scale one-hot design
    it also zeroes any column whose ridge coefficient is 0, which is where
    `regression_type='mixed'` died with a bare LinAlgError.

    The sum a + b is what the simulation moves, so with a and b near-identical
    the split between them is arbitrary - hence the assertion on the sum.
    """
    from spacr.ml import perform_mixed_model

    rng = np.random.default_rng(2)
    n = 60
    shared = rng.normal(0, 1, n)
    X = pd.DataFrame({"a": shared, "b": shared + rng.normal(0, 0.01, n)})
    groups = np.repeat(np.arange(6), 10)
    y = 1.5 * X["a"] + rng.normal(0, 0.3, n)

    result = perform_mixed_model(y, X, groups)

    out = capsys.readouterr().out
    assert "Multicollinearity detected with VIF > 10 for: ['a', 'b']" in out
    assert "fitted on the design as given" in out

    assert list(result.params.index) == ["a", "b", "Group Var"]
    assert np.all(np.isfinite(result.fe_params.values))
    assert abs(float(result.fe_params.sum()) - 1.5) < 0.25


def test_perform_mixed_model_refuses_a_penalty(capsys):
    """alpha named a "ridge fallback" that was not ridge; it is refused by name."""
    from spacr.ml import perform_mixed_model

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.5, 0.1, 0.9, 0.2]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="takes no penalty"):
        perform_mixed_model(y, X, np.array([0, 0, 1, 1]), alpha=1.0)


def test_perform_mixed_model_refuses_misaligned_groups():
    """A groups vector shorter than the design would relabel every later row."""
    from spacr.ml import perform_mixed_model

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.5, 0.1, 0.9, 0.2]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="groups has 3 entries but the design"):
        perform_mixed_model(y, X, np.array([0, 0, 1]))


def test_perform_mixed_model_refuses_a_rank_deficient_design():
    """An aliased column is named, instead of LinAlgError from inside statsmodels."""
    from spacr.ml import perform_mixed_model

    rng = np.random.default_rng(5)
    n = 40
    a = rng.normal(0, 1, n)
    X = pd.DataFrame({"a": a, "b": 2.0 * a})      # exactly aliased
    y = pd.Series(a + rng.normal(0, 0.1, n))
    with pytest.raises(ValueError, match="rank 1 with 2 columns"):
        perform_mixed_model(y, X, np.repeat(np.arange(4), 10))


# ---------------------------------------------------------------------------
# fit_mixed_model
# ---------------------------------------------------------------------------

def test_fit_mixed_model_returns_coefficients_and_writes_histogram(tmp_path):
    from spacr.ml import fit_mixed_model, prepare_formula

    df = _mixed_model_frame()
    formula = prepare_formula("score", random_row_column_effects=True)

    mixed_model, coef_df = fit_mixed_model(df, formula, dst=str(tmp_path))

    # The residual histogram PDF is the only file the function emits.
    assert (tmp_path / "residuals_histogram.pdf").is_file()
    assert (tmp_path / "residuals_histogram.pdf").stat().st_size > 0

    # Residuals are written back onto the caller's frame.
    assert "residuals" in df.columns
    assert len(df["residuals"]) == len(df)
    assert np.allclose(df["residuals"].values, mixed_model.resid.values)

    assert list(coef_df.columns) == ["feature", "coefficient", "p_value"]
    assert len(coef_df) == len(mixed_model.params)
    assert coef_df["coefficient"].dtype.kind == "f"
    assert coef_df["p_value"].dtype.kind == "f"
    # Variance components can sit on the boundary (NaN p-value); every
    # p-value that IS defined has to be a probability.
    pv = coef_df["p_value"].to_numpy()
    finite = np.isfinite(pv)
    assert finite.sum() >= 6
    assert np.all((pv[finite] >= 0.0) & (pv[finite] <= 1.0))

    feats = set(coef_df["feature"])
    assert "Intercept" in feats
    assert "fraction:grna[geneA_g1]" in feats
    assert "gene_fraction:gene[geneA]" in feats
    # score was simulated as 0.4*fraction, so the per-grna slopes are
    # significantly positive.
    grna_rows = coef_df[coef_df["feature"].str.startswith("fraction:grna")]
    assert len(grna_rows) == 4
    assert (grna_rows["coefficient"] > 0).all()
    assert (grna_rows["p_value"] < 0.01).all()

    # random_row_column_effects=True keeps row/column OUT of the fixed
    # effects; they only survive as the vc_formula variance components.
    assert "rowID[T.r2]" not in set(mixed_model.fe_params.index)
    assert "columnID[T.c2]" not in set(mixed_model.fe_params.index)
    assert {"rowID Var", "columnID Var", "Group Var"} <= feats


def test_fit_mixed_model_full_formula_adds_row_and_column_fixed_effects():
    """The default formula puts rowID/columnID in the fixed effects too."""
    from spacr.ml import fit_mixed_model, prepare_formula

    df = _mixed_model_frame(seed=5)
    formula = prepare_formula("score", random_row_column_effects=False)

    mixed_model, coef_df = fit_mixed_model(df, formula, dst=None)

    feats = set(coef_df["feature"])
    assert "rowID[T.r2]" in feats
    assert "columnID[T.c2]" in feats
    # ...and this time they really are FIXED effects.
    assert "rowID[T.r2]" in set(mixed_model.fe_params.index)
    assert "columnID[T.c2]" in set(mixed_model.fe_params.index)
    assert len(coef_df) == len(mixed_model.params) > 0
    # The simulated gene effect is positive for geneA.
    a = float(coef_df.loc[coef_df["feature"] == "gene_fraction:gene[geneA]",
                          "coefficient"].iloc[0])
    assert np.isfinite(a)


# ---------------------------------------------------------------------------
# check_and_clean_data
# ---------------------------------------------------------------------------

def test_check_and_clean_data_drops_rows_with_missing_values(capsys):
    from spacr.ml import check_and_clean_data

    df = _well_frame()
    df.loc[0:2, "fraction"] = np.nan       # 3 rows
    df.loc[10, "prediction"] = np.nan      # 1 more row
    n_in = len(df)

    out = check_and_clean_data(df, "prediction")

    printed = capsys.readouterr().out
    assert "Dropped 4 rows with missing values" in printed
    assert len(out) == n_in - 4
    assert out[["fraction", "prediction"]].isna().sum().sum() == 0
    assert "gene_fraction" in out.columns
    # gene_fraction is the per-(well, gene) sum of fraction.
    expect = out.groupby(["prc", "gene"], observed=True)["fraction"].transform("sum")
    assert np.allclose(out["gene_fraction"].values, expect.values)


def test_check_and_clean_data_flags_rank_deficiency(capsys):
    """A constant (all-zero) dependent variable makes the 2-column matrix
    rank deficient; the warning fires and the frame still comes back whole."""
    from spacr.ml import check_and_clean_data

    df = _well_frame(seed=3)
    df["prediction"] = 0.0
    assert np.linalg.matrix_rank(df[["fraction", "prediction"]].values) == 1

    out = check_and_clean_data(df, "prediction")

    printed = capsys.readouterr().out
    assert "Perfect multicollinearity detected" in printed
    # Neither column exceeds the VIF threshold, so no collinearity warning.
    assert "high collinearity (VIF > 10)" not in printed
    assert list(out.columns) == ["fraction", "prediction", "gene", "grna",
                                 "prc", "plateID", "rowID", "columnID",
                                 "gene_fraction"]
    assert len(out) == len(df)


def test_check_and_clean_data_survives_linalgerror_from_vif(monkeypatch, capsys):
    """If statsmodels cannot compute the VIF the helper bails out of the
    collinearity check and still returns a usable frame."""
    from spacr import ml as ml_mod
    from spacr.ml import check_and_clean_data

    def _boom(*args, **kwargs):
        raise np.linalg.LinAlgError("singular matrix (injected)")

    monkeypatch.setattr(ml_mod, "variance_inflation_factor", _boom)

    df = _well_frame(seed=4)
    out = check_and_clean_data(df, "prediction")

    printed = capsys.readouterr().out
    assert "LinAlgError: Unable to compute VIF" in printed
    assert "Variance Inflation Factor (VIF) for each feature" not in printed
    assert len(out) == len(df)
    assert {"fraction", "prediction", "gene_fraction"} <= set(out.columns)
    assert np.allclose(out["fraction"].values, df["fraction"].values)


def test_check_and_clean_data_high_vif_keeps_fraction_column(capsys):
    """A dependent variable proportional to fraction is perfectly collinear
    (VIF = inf). The collinearity is reported, but neither column may be
    dropped: the regression formula needs both, and dropping 'fraction' used
    to make the gene_fraction line raise KeyError."""
    from spacr.ml import check_and_clean_data

    df = _well_frame(seed=6)
    df["prediction"] = 2.0 * df["fraction"]

    out = check_and_clean_data(df, "prediction")

    printed = capsys.readouterr().out
    assert "high collinearity (VIF > 10)" in printed
    assert "fraction" in out.columns
    assert "prediction" in out.columns
    assert "gene_fraction" in out.columns
    assert len(out) == len(df)
    assert np.allclose(out["fraction"].values, df["fraction"].values)


def test_check_and_clean_data_near_collinear_dependent_variable(capsys):
    """Near- (but not exactly-) collinear columns give a huge finite VIF and
    take the same keep-and-warn path without the rank-deficiency branch."""
    from spacr.ml import check_and_clean_data

    rng = np.random.default_rng(7)
    df = _well_frame(seed=8)
    df["prediction"] = df["fraction"] + rng.normal(0, 1e-3, len(df))
    # Full rank, so the rank-deficiency branch is NOT what fires here.
    assert np.linalg.matrix_rank(df[["fraction", "prediction"]].values) == 2

    out = check_and_clean_data(df, "prediction")

    printed = capsys.readouterr().out
    assert "Perfect multicollinearity detected" not in printed
    assert "high collinearity (VIF > 10)" in printed
    assert "fraction" in out.columns
    assert "prediction" in out.columns
    assert "gene_fraction" in out.columns
    assert len(out) == len(df)
