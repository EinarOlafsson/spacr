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
    """A minimal screen-style frame with everything check_and_clean_data wants."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "fraction": rng.uniform(0.05, 0.9, n),
        "prediction": rng.uniform(0.05, 0.9, n),
        "grna": rng.choice(["g1", "g2"], n),
        "gene": rng.choice(["geneA", "geneB"], n),
        "plateID": ["plate1"] * n,
        "rowID": rng.choice(["r1", "r2"], n),
        "columnID": rng.choice(["c1", "c2"], n),
        "prc": rng.choice(["plate1_r1_c1", "plate1_r2_c2"], n),
    })


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


@pytest.mark.xfail(strict=True, reason=(
    "BUG: QuasiBinomial.variance is shadowed by the varfuncs instance "
    "attribute statsmodels sets in Binomial.__init__, so the dispersion "
    "factor is silently ignored by every GLM fit that uses this family"))
def test_quasibinomial_family_applies_dispersion_when_called_normally():
    """A dispersion of 3.0 must triple the binomial variance."""
    from spacr.ml import QuasiBinomial

    fam = QuasiBinomial(dispersion=3.0)
    mu = np.array([0.25, 0.5, 0.75])
    assert np.allclose(fam.variance(mu), 3.0 * mu * (1.0 - mu))


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


def test_perform_mixed_model_high_vif_applies_ridge(capsys):
    """Near-duplicate columns trip the VIF>10 guard and the fixed-effects
    design is rescaled by the ridge coefficients before fitting."""
    from spacr.ml import perform_mixed_model

    rng = np.random.default_rng(2)
    n = 60
    shared = rng.normal(0, 1, n)
    X = pd.DataFrame({"a": shared, "b": shared + rng.normal(0, 0.01, n)})
    groups = np.repeat(np.arange(6), 10)
    y = 1.5 * X["a"] + rng.normal(0, 0.3, n)

    result = perform_mixed_model(y, X, groups, alpha=1.0)

    out = capsys.readouterr().out
    assert "Multicollinearity detected" in out
    assert "Applying Ridge regression to the fixed effects" in out

    assert list(result.params.index) == ["a", "b", "Group Var"]
    assert np.all(np.isfinite(result.fe_params.values))
    # The ridge-rescaled design is a different model from the plain one:
    # its fixed effects no longer sit near the simulated 1.5 / 0.0.
    assert not np.allclose(result.fe_params.values, [1.5, 0.0], atol=0.25)


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
    # Neither column has a finite VIF > 10, so nothing is dropped.
    assert "Dropping columns with high VIF" not in printed
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


@pytest.mark.xfail(strict=True, reason=(
    "BUG: check_and_clean_data drops EVERY column whose VIF > 10 - including "
    "'fraction' - and then unconditionally computes gene_fraction from "
    "df_cleaned['fraction'], raising KeyError: 'Column not found: fraction'"))
def test_check_and_clean_data_high_vif_keeps_fraction_column():
    """A dependent variable proportional to fraction is perfectly collinear
    (VIF = inf). Cleaning should still return a usable frame."""
    from spacr.ml import check_and_clean_data

    df = _well_frame(seed=6)
    df["prediction"] = 2.0 * df["fraction"]

    out = check_and_clean_data(df, "prediction")

    assert "fraction" in out.columns
    assert "gene_fraction" in out.columns
    assert len(out) == len(df)


@pytest.mark.xfail(strict=True, reason=(
    "BUG: same high-VIF drop as above, reached without rank deficiency - "
    "check_and_clean_data raises KeyError instead of returning the frame"))
def test_check_and_clean_data_near_collinear_dependent_variable():
    """Near- (but not exactly-) collinear columns give a huge finite VIF."""
    from spacr.ml import check_and_clean_data

    rng = np.random.default_rng(7)
    df = _well_frame(seed=8)
    df["prediction"] = df["fraction"] + rng.normal(0, 1e-3, len(df))
    # Full rank, so the rank-deficiency branch is NOT what fires here.
    assert np.linalg.matrix_rank(df[["fraction", "prediction"]].values) == 2

    out = check_and_clean_data(df, "prediction")

    assert "fraction" in out.columns
    assert "gene_fraction" in out.columns
