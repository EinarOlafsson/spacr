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

def case_quasibinomial_variance_method_scales_binomial_variance():
    """The overridden method itself multiplies mu*(1-mu) by the dispersion.

    It has to be invoked unbound because statsmodels' ``Binomial.__init__``
    puts a ``variance`` *callable object* in the instance ``__dict__``,
    which shadows the subclass method for ordinary attribute access.
    """
    _seen = []
    from spacr.ml import QuasiBinomial

    fam = QuasiBinomial(dispersion=2.5)
    mu = np.array([0.2, 0.5, 0.8])

    # Precondition: the instance attribute is what shadows the method.
    _seen.append(('variance' if 'variance' in (fam.__dict__) else ("missing", 'variance')))

    got = QuasiBinomial.variance(fam, mu)
    _seen.append(bool(
        np.allclose(got, 2.5 * mu * (1.0 - mu))))
    # Dispersion 1.0 must degenerate to the plain binomial variance.
    plain = QuasiBinomial.variance(QuasiBinomial(dispersion=1.0), mu)
    _seen.append(bool(
        np.allclose(plain, mu * (1.0 - mu))))
    return _seen


def case_quasibinomial_family_applies_dispersion_when_called_normally():
    """A dispersion of 3.0 must triple the binomial variance."""
    _seen = []
    from spacr.ml import QuasiBinomial

    fam = QuasiBinomial(dispersion=3.0)
    mu = np.array([0.25, 0.5, 0.75])
    _seen.append(bool(
        np.allclose(fam.variance(mu), 3.0 * mu * (1.0 - mu))))
    return _seen


def case_quasibinomial_variance_wrapper_scales_deriv_and_delegates_attributes():
    """statsmodels' GLM calls family.variance.deriv and reads varfunc state,
    so the dispersion wrapper has to scale the derivative and pass the rest
    of the varfunc's attribute surface straight through."""
    _seen = []
    from spacr.ml import QuasiBinomial

    mu = np.array([0.25, 0.5, 0.75])
    plain = QuasiBinomial(dispersion=1.0)
    fam = QuasiBinomial(dispersion=4.0)

    # d/dmu [mu(1-mu)] = 1 - 2mu, scaled by the dispersion.
    _seen.append(bool(
        np.allclose(fam.variance.deriv(mu), 4.0 * (1.0 - 2.0 * mu))))
    _seen.append(bool(
        np.allclose(fam.variance.deriv(mu), 4.0 * plain.variance.deriv(mu))))
    # 'n' lives on the wrapped varfuncs.Binomial instance, not on the wrapper.
    _seen.append((("absent", 'n') if 'n' not in (fam.variance.__dict__) else ("present", 'n')))
    _seen.append(bool(
        fam.variance.n == plain.variance.n))
    with pytest.raises(AttributeError) as _exc1:
        fam.variance.not_a_varfunc_attribute
    _seen.append(('AttributeError' if isinstance(_exc1.value, AttributeError) else type(_exc1.value).__name__))
    return _seen


def case_quasibinomial_survives_pickle_and_deepcopy():
    """statsmodels results hold on to the family, and those get pickled, so
    the variance wrapper must not turn copy/pickle's __setstate__ probe into
    a KeyError on a half-built instance."""
    _seen = []
    import copy
    import pickle

    from spacr.ml import QuasiBinomial

    mu = np.array([0.25, 0.5, 0.75])
    fam = QuasiBinomial(dispersion=2.0)

    restored = pickle.loads(pickle.dumps(fam))
    _seen.append(bool(
        np.allclose(restored.variance(mu), 2.0 * mu * (1.0 - mu))))
    _seen.append(bool(
        np.allclose(restored.variance.deriv(mu), 2.0 * (1.0 - 2.0 * mu))))

    cloned = copy.deepcopy(fam)
    _seen.append(bool(
        np.allclose(cloned.variance(mu), 2.0 * mu * (1.0 - mu))))
    return _seen


def case_quasibinomial_dispersion_one_matches_plain_binomial_glm():
    """The only dispersion spacr constructs is 1.0, which must fit exactly
    like statsmodels' own Binomial family."""
    _seen = []
    import statsmodels.api as sm

    from spacr.ml import QuasiBinomial

    rng = np.random.default_rng(0)
    n = 200
    X = sm.add_constant(rng.normal(size=(n, 2)))
    p = 1.0 / (1.0 + np.exp(-(X @ np.array([0.2, 0.5, -0.3]))))
    y = rng.binomial(1, p).astype(float)

    quasi = sm.GLM(y, X, family=QuasiBinomial()).fit()
    plain = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    _seen.append(bool(
        np.allclose(quasi.params, plain.params)))
    _seen.append(bool(
        np.allclose(quasi.bse, plain.bse)))

    # Dispersion 3 leaves the point estimates alone and inflates the standard
    # errors by sqrt(3) -- the whole point of a quasi-binomial fit.
    over = sm.GLM(y, X, family=QuasiBinomial(dispersion=3.0)).fit()
    _seen.append(bool(
        np.allclose(over.params, plain.params)))
    _seen.append(bool(
        np.allclose(over.bse / plain.bse, np.sqrt(3.0))))
    return _seen


# ---------------------------------------------------------------------------
# calculate_p_values — pinv fallback
# ---------------------------------------------------------------------------

def case_calculate_p_values_uses_pinv_when_xtx_is_singular():
    """Duplicated design columns make X.T @ X exactly singular; the helper
    must fall back to the pseudo-inverse instead of raising."""
    _seen = []
    from sklearn.linear_model import LinearRegression
    from spacr.ml import calculate_p_values

    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 12)
    X = np.column_stack([base, base])          # rank 1, 2 columns
    y = 2.0 * base + rng.normal(0, 0.05, 12)

    # Precondition: plain inversion really does blow up here.
    with pytest.raises(np.linalg.LinAlgError) as _exc1:
        np.linalg.inv(X.T @ X)
    _seen.append(('LinAlgError' if isinstance(_exc1.value, np.linalg.LinAlgError) else type(_exc1.value).__name__))

    model = LinearRegression().fit(X, y)
    p = calculate_p_values(X, y, model)

    _seen.append(keep(p.shape))
    _seen.append(bool(
        np.all(np.isfinite(p))))
    _seen.append(bool(
        np.all((p >= 0.0) & (p <= 1.0))))
    # dof = 12 - 2 - 1 = 9 > 0, so this is NOT the NaN short-circuit.
    _seen.append(bool(
        not np.any(np.isnan(p))))
    return _seen


# ---------------------------------------------------------------------------
# perform_mixed_model
# ---------------------------------------------------------------------------

def case_perform_mixed_model_requires_groups():
    _seen = []
    from spacr.ml import perform_mixed_model

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.5, 0.1, 0.9]})
    y = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Groups must be defined") as _exc1:
        perform_mixed_model(y, X, None)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_perform_mixed_model_low_vif_fits_plain_mixedlm(capsys):
    """Uncorrelated fixed effects -> no ridge adjustment, and the recovered
    slope on the informative column is close to the simulated 1.5."""
    _seen = []
    from spacr.ml import perform_mixed_model

    rng = np.random.default_rng(1)
    n = 60
    X = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    groups = np.repeat(np.arange(6), 10)
    y = 1.5 * X["a"] + rng.normal(0, 0.3, n)

    result = perform_mixed_model(y, X, groups)

    out = capsys.readouterr().out
    _seen.append(('VIF:' if 'VIF:' in (out) else ("missing", 'VIF:')))
    _seen.append((("absent", 'Multicollinearity detected') if 'Multicollinearity detected' not in (out) else ("present", 'Multicollinearity detected')))

    _seen.append(keep(list(result.params.index)))
    _seen.append(keep(result.fe_params.shape))
    _seen.append(bool(
        abs(float(result.fe_params["a"]) - 1.5) < 0.25))
    _seen.append(bool(
        abs(float(result.fe_params["b"])) < 0.25))
    return _seen


def case_perform_mixed_model_high_vif_reports_and_fits_the_given_design(capsys):
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
    _seen = []
    from spacr.ml import perform_mixed_model

    rng = np.random.default_rng(2)
    n = 60
    shared = rng.normal(0, 1, n)
    X = pd.DataFrame({"a": shared, "b": shared + rng.normal(0, 0.01, n)})
    groups = np.repeat(np.arange(6), 10)
    y = 1.5 * X["a"] + rng.normal(0, 0.3, n)

    result = perform_mixed_model(y, X, groups)

    out = capsys.readouterr().out
    _seen.append(("Multicollinearity detected with VIF > 10 for: ['a', 'b']" if "Multicollinearity detected with VIF > 10 for: ['a', 'b']" in (out) else ("missing", "Multicollinearity detected with VIF > 10 for: ['a', 'b']")))
    _seen.append(('fitted on the design as given' if 'fitted on the design as given' in (out) else ("missing", 'fitted on the design as given')))

    _seen.append(keep(list(result.params.index)))
    _seen.append(bool(
        np.all(np.isfinite(result.fe_params.values))))
    _seen.append(bool(
        abs(float(result.fe_params.sum()) - 1.5) < 0.25))
    return _seen


def case_perform_mixed_model_refuses_a_penalty(capsys):
    """alpha named a "ridge fallback" that was not ridge; it is refused by name."""
    _seen = []
    from spacr.ml import perform_mixed_model

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.5, 0.1, 0.9, 0.2]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="takes no penalty") as _exc1:
        perform_mixed_model(y, X, np.array([0, 0, 1, 1]), alpha=1.0)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_perform_mixed_model_refuses_misaligned_groups():
    """A groups vector shorter than the design would relabel every later row."""
    _seen = []
    from spacr.ml import perform_mixed_model

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.5, 0.1, 0.9, 0.2]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="groups has 3 entries but the design") as _exc1:
        perform_mixed_model(y, X, np.array([0, 0, 1]))
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_perform_mixed_model_refuses_a_rank_deficient_design():
    """An aliased column is named, instead of LinAlgError from inside statsmodels."""
    _seen = []
    from spacr.ml import perform_mixed_model

    rng = np.random.default_rng(5)
    n = 40
    a = rng.normal(0, 1, n)
    X = pd.DataFrame({"a": a, "b": 2.0 * a})      # exactly aliased
    y = pd.Series(a + rng.normal(0, 0.1, n))
    with pytest.raises(ValueError, match="rank 1 with 2 columns") as _exc1:
        perform_mixed_model(y, X, np.repeat(np.arange(4), 10))
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


# ---------------------------------------------------------------------------
# fit_mixed_model
# ---------------------------------------------------------------------------

def case_fit_mixed_model_returns_coefficients_and_writes_histogram(tmp_path):
    """CONTRACT CHANGED BY INSTRUCTION 132 (maintainer, 2026-08-17).

    This test used to assert that ONE fitted design carried both
    ``fraction:grna[geneA_g1]`` and ``gene_fraction:gene[geneA]``. That design
    is unfittable: ``check_and_clean_data`` builds ``gene_fraction`` as the SUM
    of the gene's gRNA fractions, so the gene block is an exact linear
    combination of the guide block. Measured on the maintainer's TSG101 screen
    (spacr.ml.COLLINEAR_FORMULA_FRAGMENT): 1248 parameters at rank 862, a
    386-dimensional null space that annihilates the design EXACTLY.

    ``mixed`` is now the nested model the maintainer specified:
    ``y ~ gene_fraction:gene + (1 | gene/grna) + rowID + columnID``. The gene
    is a fixed effect; the guide is a random effect nested inside it, and its
    output is a BLUP, not a coefficient with a p-value.
    """
    _seen = []
    from spacr.ml import (TERM_BLUP, TERM_FIXED, TERM_VARIANCE,
                          fit_mixed_model, prepare_formula)

    df = _mixed_model_frame()
    formula = prepare_formula("score", random_row_column_effects=True,
                              level="gene")
    _seen.append((("absent", 'fraction:grna') if 'fraction:grna' not in (formula) else ("present", 'fraction:grna')))

    mixed_model, coef_df = fit_mixed_model(
        df, formula, dst=str(tmp_path), random_row_column_effects=True)

    # The residual histogram PDF is the only file the function emits.
    _seen.append(bool(
        (tmp_path / "residuals_histogram.pdf").is_file()))
    _seen.append(bool(
        (tmp_path / "residuals_histogram.pdf").stat().st_size > 0))

    # Residuals are written back onto the caller's frame.
    _seen.append(('residuals' if 'residuals' in (df.columns) else ("missing", 'residuals')))
    _seen.append(bool(
        len(df["residuals"]) == len(df)))
    _seen.append(bool(
        np.allclose(df["residuals"].values, mixed_model.resid.values)))

    _seen.append(keep(list(coef_df.columns)))
    _seen.append(keep(coef_df["coefficient"].dtype.kind))
    _seen.append(keep(coef_df["p_value"].dtype.kind))
    # Every fitted parameter is present, plus one row per guide BLUP.
    _seen.append(bool(
        (coef_df["term_type"] != TERM_BLUP).sum() == len(mixed_model.params)))

    feats = set(coef_df["feature"])
    _seen.append(('Intercept' if 'Intercept' in (feats) else ("missing", 'Intercept')))
    _seen.append(('gene_fraction:gene[geneA]' if 'gene_fraction:gene[geneA]' in (feats) else ("missing", 'gene_fraction:gene[geneA]')))
    # THE GUIDE IS NOT A FIXED EFFECT ANY MORE. Its old term must be gone from
    # the fit entirely -- that is the collinearity this instruction removed.
    _seen.append(bool(
        not any(f.startswith("fraction:grna") for f in feats)))

    # ...it comes back as a BLUP instead, one per guide, WITHOUT a p-value.
    blups = coef_df[coef_df["term_type"] == TERM_BLUP]
    _seen.append(keep(set(blups["feature"])))
    _seen.append(bool(
        blups["p_value"].isna().all()))
    _seen.append(bool(
        np.isfinite(blups["coefficient"].to_numpy()).all()))

    # The nesting itself: gene is the group, guide is a variance component
    # INSIDE it, so guide effects are never shared across genes.
    _seen.append(('grna Var' if 'grna Var' in (feats) else ("missing", 'grna Var')))
    _seen.append(('Group Var' if 'Group Var' in (feats) else ("missing", 'Group Var')))
    # A variance component's Wald p-value tests sigma^2 = 0, which is on the
    # BOUNDARY of the parameter space, so the normal reference distribution
    # does not hold and the number statsmodels prints is not a probability.
    # Reported as NaN rather than passed on.
    variances = coef_df[coef_df["term_type"] == TERM_VARIANCE]
    _seen.append(bool(
        len(variances) >= 3))
    _seen.append(bool(
        variances["p_value"].isna().all()))
    _seen.append(bool(
        (coef_df["term_type"] == TERM_FIXED).sum() >= 2))

    # Every p-value that IS defined has to be a probability.
    pv = coef_df.loc[coef_df["term_type"] == TERM_FIXED, "p_value"].to_numpy()
    finite = np.isfinite(pv)
    _seen.append(bool(
        finite.sum() >= 2))
    _seen.append(bool(
        np.all((pv[finite] >= 0.0) & (pv[finite] <= 1.0))))

    # random_row_column_effects=True keeps row/column OUT of the fixed
    # effects; they only survive as the vc_formula variance components.
    _seen.append((("absent", 'rowID[T.r2]') if 'rowID[T.r2]' not in (set(mixed_model.fe_params.index)) else ("present", 'rowID[T.r2]')))
    _seen.append((("absent", 'columnID[T.c2]') if 'columnID[T.c2]' not in (set(mixed_model.fe_params.index)) else ("present", 'columnID[T.c2]')))
    _seen.append(bool(
        {"rowID Var", "columnID Var", "Group Var"} <= feats))
    return _seen


def case_fit_mixed_model_full_formula_adds_row_and_column_fixed_effects():
    """The default formula puts rowID/columnID in the fixed effects too."""
    _seen = []
    from spacr.ml import TERM_BLUP, fit_mixed_model, prepare_formula

    df = _mixed_model_frame(seed=5)
    formula = prepare_formula("score", random_row_column_effects=False,
                              level="gene")

    mixed_model, coef_df = fit_mixed_model(df, formula, dst=None)

    feats = set(coef_df["feature"])
    _seen.append(('rowID[T.r2]' if 'rowID[T.r2]' in (feats) else ("missing", 'rowID[T.r2]')))
    _seen.append(('columnID[T.c2]' if 'columnID[T.c2]' in (feats) else ("missing", 'columnID[T.c2]')))
    # ...and this time they really are FIXED effects.
    _seen.append(('rowID[T.r2]' if 'rowID[T.r2]' in (set(mixed_model.fe_params.index)) else ("missing", 'rowID[T.r2]')))
    _seen.append(('columnID[T.c2]' if 'columnID[T.c2]' in (set(mixed_model.fe_params.index)) else ("missing", 'columnID[T.c2]')))
    _seen.append(bool(
        (coef_df["term_type"] != TERM_BLUP).sum() == len(mixed_model.params) > 0))
    # The simulated gene effect is positive for geneA.
    a = float(coef_df.loc[coef_df["feature"] == "gene_fraction:gene[geneA]",
                          "coefficient"].iloc[0])
    _seen.append(bool(
        np.isfinite(a)))
    return _seen


def case_fit_mixed_model_refuses_a_frame_that_cannot_nest():
    """One guide per gene leaves the guide variance confounded with the residual.

    Instruction 132: "If MixedLM cannot express the nesting on this data, say
    so with the error rather than faking it." A boundary variance of zero
    looks exactly like an answer.
    """
    _seen = []
    import pytest

    from spacr.ml import fit_mixed_model, prepare_formula

    df = _mixed_model_frame(seed=7)
    df = df[df["grna"].str.endswith("_g1")].copy()      # one guide per gene
    formula = prepare_formula("score", level="gene")

    with pytest.raises(ValueError, match="no gene in this frame has more") as _exc1:
        fit_mixed_model(df, formula, dst=None)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_fit_mixed_model_refuses_a_frame_with_no_gene_column():
    _seen = []
    import pytest

    from spacr.ml import fit_mixed_model, prepare_formula

    df = _mixed_model_frame(seed=8).drop(columns=["gene"])
    formula = prepare_formula("score", level="gene")

    with pytest.raises(ValueError, match="no 'gene' column") as _exc1:
        fit_mixed_model(df, formula, dst=None)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


# ---------------------------------------------------------------------------
# check_and_clean_data
# ---------------------------------------------------------------------------

def case_check_and_clean_data_drops_rows_with_missing_values(capsys):
    _seen = []
    from spacr.ml import check_and_clean_data

    df = _well_frame()
    df.loc[0:2, "fraction"] = np.nan       # 3 rows
    df.loc[10, "prediction"] = np.nan      # 1 more row
    n_in = len(df)

    out = check_and_clean_data(df, "prediction")

    printed = capsys.readouterr().out
    _seen.append(('Dropped 4 rows with missing values' if 'Dropped 4 rows with missing values' in (printed) else ("missing", 'Dropped 4 rows with missing values')))
    _seen.append(bool(
        len(out) == n_in - 4))
    _seen.append(keep(out[["fraction", "prediction"]].isna().sum().sum()))
    _seen.append(('gene_fraction' if 'gene_fraction' in (out.columns) else ("missing", 'gene_fraction')))
    # gene_fraction is the per-(well, gene) sum of fraction.
    expect = out.groupby(["prc", "gene"], observed=True)["fraction"].transform("sum")
    _seen.append(bool(
        np.allclose(out["gene_fraction"].values, expect.values)))
    return _seen


def case_check_and_clean_data_flags_rank_deficiency(capsys):
    """A constant (all-zero) dependent variable makes the 2-column matrix
    rank deficient; the warning fires and the frame still comes back whole."""
    _seen = []
    from spacr.ml import check_and_clean_data

    df = _well_frame(seed=3)
    df["prediction"] = 0.0
    _seen.append(keep(np.linalg.matrix_rank(df[["fraction", "prediction"]].values)))

    out = check_and_clean_data(df, "prediction")

    printed = capsys.readouterr().out
    _seen.append(('Perfect multicollinearity detected' if 'Perfect multicollinearity detected' in (printed) else ("missing", 'Perfect multicollinearity detected')))
    # Neither column exceeds the VIF threshold, so no collinearity warning.
    _seen.append((("absent", 'high collinearity (VIF > 10)') if 'high collinearity (VIF > 10)' not in (printed) else ("present", 'high collinearity (VIF > 10)')))
    _seen.append(keep(list(out.columns)))
    _seen.append(bool(
        len(out) == len(df)))
    return _seen


def case_check_and_clean_data_survives_linalgerror_from_vif(monkeypatch, capsys):
    """If statsmodels cannot compute the VIF the helper bails out of the
    collinearity check and still returns a usable frame."""
    _seen = []
    from spacr import ml as ml_mod
    from spacr.ml import check_and_clean_data

    def _boom(*args, **kwargs):
        raise np.linalg.LinAlgError("singular matrix (injected)")

    monkeypatch.setattr(ml_mod, "variance_inflation_factor", _boom)

    df = _well_frame(seed=4)
    out = check_and_clean_data(df, "prediction")

    printed = capsys.readouterr().out
    _seen.append(('LinAlgError: Unable to compute VIF' if 'LinAlgError: Unable to compute VIF' in (printed) else ("missing", 'LinAlgError: Unable to compute VIF')))
    _seen.append((("absent", 'Variance Inflation Factor (VIF) for each feature') if 'Variance Inflation Factor (VIF) for each feature' not in (printed) else ("present", 'Variance Inflation Factor (VIF) for each feature')))
    _seen.append(bool(
        len(out) == len(df)))
    _seen.append(bool(
        {"fraction", "prediction", "gene_fraction"} <= set(out.columns)))
    _seen.append(bool(
        np.allclose(out["fraction"].values, df["fraction"].values)))
    return _seen


def case_check_and_clean_data_high_vif_keeps_fraction_column(capsys):
    """A dependent variable proportional to fraction is perfectly collinear
    (VIF = inf). The collinearity is reported, but neither column may be
    dropped: the regression formula needs both, and dropping 'fraction' used
    to make the gene_fraction line raise KeyError."""
    _seen = []
    from spacr.ml import check_and_clean_data

    df = _well_frame(seed=6)
    df["prediction"] = 2.0 * df["fraction"]

    out = check_and_clean_data(df, "prediction")

    printed = capsys.readouterr().out
    _seen.append(('high collinearity (VIF > 10)' if 'high collinearity (VIF > 10)' in (printed) else ("missing", 'high collinearity (VIF > 10)')))
    _seen.append(('fraction' if 'fraction' in (out.columns) else ("missing", 'fraction')))
    _seen.append(('prediction' if 'prediction' in (out.columns) else ("missing", 'prediction')))
    _seen.append(('gene_fraction' if 'gene_fraction' in (out.columns) else ("missing", 'gene_fraction')))
    _seen.append(bool(
        len(out) == len(df)))
    _seen.append(bool(
        np.allclose(out["fraction"].values, df["fraction"].values)))
    return _seen


def case_check_and_clean_data_near_collinear_dependent_variable(capsys):
    """Near- (but not exactly-) collinear columns give a huge finite VIF and
    take the same keep-and-warn path without the rank-deficiency branch."""
    _seen = []
    from spacr.ml import check_and_clean_data

    rng = np.random.default_rng(7)
    df = _well_frame(seed=8)
    df["prediction"] = df["fraction"] + rng.normal(0, 1e-3, len(df))
    # Full rank, so the rank-deficiency branch is NOT what fires here.
    _seen.append(keep(np.linalg.matrix_rank(df[["fraction", "prediction"]].values)))

    out = check_and_clean_data(df, "prediction")

    printed = capsys.readouterr().out
    _seen.append((("absent", 'Perfect multicollinearity detected') if 'Perfect multicollinearity detected' not in (printed) else ("present", 'Perfect multicollinearity detected')))
    _seen.append(('high collinearity (VIF > 10)' if 'high collinearity (VIF > 10)' in (printed) else ("missing", 'high collinearity (VIF > 10)')))
    _seen.append(('fraction' if 'fraction' in (out.columns) else ("missing", 'fraction')))
    _seen.append(('prediction' if 'prediction' in (out.columns) else ("missing", 'prediction')))
    _seen.append(('gene_fraction' if 'gene_fraction' in (out.columns) else ("missing", 'gene_fraction')))
    _seen.append(bool(
        len(out) == len(df)))
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param(case_quasibinomial_variance_method_scales_binomial_variance, ['variance', True, True], id="quasibinomial_variance_method_scales_binomial_variance"),
    pytest.param(case_quasibinomial_family_applies_dispersion_when_called_normally, [True], id="quasibinomial_family_applies_dispersion_when_called_normally"),
    pytest.param(case_quasibinomial_survives_pickle_and_deepcopy, [True, True, True], id="quasibinomial_survives_pickle_and_deepcopy"),
    pytest.param(case_calculate_p_values_uses_pinv_when_xtx_is_singular, ['LinAlgError', (2,), True, True, True], id="calculate_p_values_uses_pinv_when_xtx_is_singular"),
    pytest.param(case_perform_mixed_model_requires_groups, ['ValueError'], id="perform_mixed_model_requires_groups"),
    pytest.param(case_perform_mixed_model_low_vif_fits_plain_mixedlm, ['VIF:', ("absent", 'Multicollinearity detected'), ['a', 'b', 'Group Var'], (2,), True, True], id="perform_mixed_model_low_vif_fits_plain_mixedlm"),
    pytest.param(case_perform_mixed_model_high_vif_reports_and_fits_the_given_design, ["Multicollinearity detected with VIF > 10 for: ['a', 'b']", 'fitted on the design as given', ['a', 'b', 'Group Var'], True, True], id="perform_mixed_model_high_vif_reports_and_fits_the_given_design"),
    pytest.param(case_perform_mixed_model_refuses_a_penalty, ['ValueError'], id="perform_mixed_model_refuses_a_penalty"),
    pytest.param(case_perform_mixed_model_refuses_misaligned_groups, ['ValueError'], id="perform_mixed_model_refuses_misaligned_groups"),
    pytest.param(case_perform_mixed_model_refuses_a_rank_deficient_design, ['ValueError'], id="perform_mixed_model_refuses_a_rank_deficient_design"),
    pytest.param(case_fit_mixed_model_returns_coefficients_and_writes_histogram, [("absent", 'fraction:grna'), True, True, 'residuals', True, True, ['feature', 'coefficient', 'p_value', 'term_type'], 'f', 'f', True, 'Intercept', 'gene_fraction:gene[geneA]', True, {'blup:grna[geneA_g1]', 'blup:grna[geneA_g2]', 'blup:grna[geneB_g1]', 'blup:grna[geneB_g2]'}, True, True, 'grna Var', 'Group Var', True, True, True, True, True, ("absent", 'rowID[T.r2]'), ("absent", 'columnID[T.c2]'), True], id="fit_mixed_model_returns_coefficients_and_writes_histogram"),
    pytest.param(case_fit_mixed_model_full_formula_adds_row_and_column_fixed_effects, ['rowID[T.r2]', 'columnID[T.c2]', 'rowID[T.r2]', 'columnID[T.c2]', True, True], id="fit_mixed_model_full_formula_adds_row_and_column_fixed_effects"),
    pytest.param(case_fit_mixed_model_refuses_a_frame_that_cannot_nest, ['ValueError'], id="fit_mixed_model_refuses_a_frame_that_cannot_nest"),
    pytest.param(case_fit_mixed_model_refuses_a_frame_with_no_gene_column, ['ValueError'], id="fit_mixed_model_refuses_a_frame_with_no_gene_column"),
    pytest.param(case_check_and_clean_data_drops_rows_with_missing_values, ['Dropped 4 rows with missing values', True, 0, 'gene_fraction', True], id="check_and_clean_data_drops_rows_with_missing_values"),
    pytest.param(case_check_and_clean_data_flags_rank_deficiency, [1, 'Perfect multicollinearity detected', ("absent", 'high collinearity (VIF > 10)'), ['fraction', 'prediction', 'gene', 'grna', 'prc', 'plateID', 'rowID', 'columnID', 'gene_fraction'], True], id="check_and_clean_data_flags_rank_deficiency"),
    pytest.param(case_check_and_clean_data_survives_linalgerror_from_vif, ['LinAlgError: Unable to compute VIF', ("absent", 'Variance Inflation Factor (VIF) for each feature'), True, True, True], id="check_and_clean_data_survives_linalgerror_from_vif"),
    pytest.param(case_check_and_clean_data_high_vif_keeps_fraction_column, ['high collinearity (VIF > 10)', 'fraction', 'prediction', 'gene_fraction', True, True], id="check_and_clean_data_high_vif_keeps_fraction_column"),
    pytest.param(case_check_and_clean_data_near_collinear_dependent_variable, [2, ("absent", 'Perfect multicollinearity detected'), 'high collinearity (VIF > 10)', 'fraction', 'prediction', 'gene_fraction', True], id="check_and_clean_data_near_collinear_dependent_variable"),
    pytest.param(case_quasibinomial_variance_wrapper_scales_deriv_and_delegates_attributes, [True, True, ("absent", 'n'), True, 'AttributeError'], id="quasibinomial_variance_wrapper_scales_deriv_and_delegates_attributes"),
    pytest.param(case_quasibinomial_dispersion_one_matches_plain_binomial_glm, [True, True, True, True], id="quasibinomial_dispersion_one_matches_plain_binomial_glm"),
])
def test_ml_stats_prep_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected

