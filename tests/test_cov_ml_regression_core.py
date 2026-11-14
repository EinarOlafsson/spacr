"""Coverage for the core regression entry points of :mod:`spacr.ml`.

Targets ``regression_model`` (ridge auto-alpha CV, GLM-binomial
``var_weights``, the mixed-effects dispatch and the internal
``_find_best_alpha`` guard) plus the full ``regression`` pipeline
(auto model selection, scaling / no-scaling branches, the mixed
random row/column path and the volcano-plot hand-off) and
``save_summary_to_file``.

Everything runs on a small synthetic pooled-CRISPR table: 24 wells x 8
gRNAs (4 genes x 2 gRNAs) with a per-well phenotype score, which is the
shape ``spacr.ml.perform_regression`` feeds in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NC = "233460"   # spacr default negative control
PC = "220950"   # spacr default positive control
GENES = [NC, PC, "gene3", "gene4"]

# Coefficients patsy produces for ONE LEVEL's formula, minus the row / column
# terms that process_model_coefficients filters out.
#
# CHANGED BY INSTRUCTION 132 (maintainer, 2026-08-17). This used to be the
# UNION of both -- guides and genes in one design -- and that design cannot be
# fitted: `gene_fraction` is the sum of the gene's guide fractions
# (`check_and_clean_data`), so the gene block is an exact linear combination of
# the guide block. Measured on the maintainer's TSG101 screen: 1248 parameters
# at rank 862, a 386-dimensional null space, and a fit whose residual sum of
# squares is unchanged when the coefficients are moved along it. `regression()`
# fits ONE level now; `regression_levels()` fits both, separately.
EXPECTED_GRNA_FEATURES = (
    {"Intercept"}
    | {f"fraction:grna[{g}_{s}]" for g in GENES for s in ("a", "b")}
)
EXPECTED_GENE_FEATURES = (
    {"Intercept"}
    | {f"gene_fraction:gene[{g}]" for g in GENES}
)
EXPECTED_FEATURES = EXPECTED_GRNA_FEATURES


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _wells_df(seed=0, n_plates=1, n_rows=4, n_cols=6, dep="predictions",
              dep_kind="normal"):
    """Long-format score/count table: one row per (well, gRNA)."""
    rng = np.random.default_rng(seed)
    grnas = {g: [f"{g}_a", f"{g}_b"] for g in GENES}
    recs = []
    for p in range(n_plates):
        plate = f"plate{p + 1}"
        for r in range(n_rows):
            for c in range(n_cols):
                row_id = f"r{r + 1:02d}"
                col_id = f"c{c + 1:02d}"
                prc = f"{plate}_{row_id}_{col_id}"
                raw = rng.random(len(GENES) * 2) + 0.2
                frac = raw / raw.sum()          # gRNA fractions sum to 1 per well
                if dep_kind == "binary":
                    score = float(rng.integers(0, 2))
                else:
                    score = float(rng.normal(0.0, 1.0))
                k = 0
                for gene in GENES:
                    for grna in grnas[gene]:
                        recs.append({
                            "plateID": plate, "rowID": row_id, "columnID": col_id,
                            "prc": prc, "gene": gene, "grna": grna,
                            "fraction": float(frac[k]),
                            "cell_count": int(rng.integers(30, 200)),
                            dep: score,
                        })
                        k += 1
    return pd.DataFrame(recs)


def _fraction_xy(n=120, seed=0):
    """Design matrix + a continuous fraction response in (0, 1) for GLM-binomial."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    X = pd.DataFrame({"const": 1.0, "x": x})
    p = 1.0 / (1.0 + np.exp(-(0.8 * x)))
    y = pd.Series(np.clip(p + rng.normal(0, 0.03, n), 0.01, 0.99), name="frac")
    return X, y


# ---------------------------------------------------------------------------
# regression_model — ridge cross-validated alpha (lines 601-602)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", ["auto", None])
def test_regression_model_ridge_auto_alpha_uses_ridgecv(alpha, capsys):
    from sklearn.linear_model import RidgeCV
    from spacr.ml import regression_model

    rng = np.random.default_rng(11)
    X = rng.normal(0, 1, (120, 4))
    beta = np.array([2.0, -1.5, 0.0, 0.75])
    y = X @ beta + rng.normal(0, 0.2, 120)

    model = regression_model(X, y, regression_type="ridge", alpha=alpha)

    assert isinstance(model, RidgeCV)
    # alpha_ must be one of the 100 log-spaced candidates the code builds.
    assert np.any(np.isclose(model.alpha_, np.logspace(-5, 5, 100)))
    assert model.coef_.shape == (4,)
    # Lightly penalised ridge recovers the planted signs / magnitudes.
    assert model.coef_[0] > 1.5 and model.coef_[1] < -1.0
    assert abs(model.coef_[2]) < 0.2
    out = capsys.readouterr().out
    assert "Optimal alpha for ridge" in out
    assert "Ridge regression MSE" in out
    assert "non-zero coefficients: 4 of 4" in out


def case_regression_model_find_best_alpha_rejects_unknown_model(monkeypatch):
    """The ``_find_best_alpha`` guard clause raises on an unknown backend.

    The helper is a closure, so it is recovered from the traceback frame of
    ``regression_model`` after injecting a failure into ``LassoCV.fit``.
    """
    _seen = []
    from spacr import ml as ML

    class _BoomCV:
        def __init__(self, *a, **k):
            pass

        def fit(self, X, y):
            raise RuntimeError("boom-injected")

    monkeypatch.setattr(ML, "LassoCV", _BoomCV)

    rng = np.random.default_rng(12)
    X = rng.normal(0, 1, (40, 3))
    y = X @ np.array([1.0, 0.5, -0.2]) + rng.normal(0, 0.1, 40)

    finder = None
    with pytest.raises(RuntimeError, match="boom-injected") as excinfo:
        ML.regression_model(X, y, regression_type="lasso", alpha="auto")
    _seen.append(('RuntimeError' if isinstance(excinfo.value, RuntimeError) else type(excinfo.value).__name__))
    tb = excinfo.value.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_name == "regression_model":
            finder = tb.tb_frame.f_locals.get("_find_best_alpha")
        tb = tb.tb_next

    _seen.append(bool(
        callable(finder)))
    with pytest.raises(ValueError, match="unknown model_cls='bogus'") as _exc1:
        finder("bogus")
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


# ---------------------------------------------------------------------------
# regression_model — GLM-binomial var_weights (line 613)
# ---------------------------------------------------------------------------

def case_regression_model_logit_applies_var_weights():
    _seen = []
    from spacr.ml import regression_model
    from statsmodels.genmod.families.links import Logit

    X, y = _fraction_xy(seed=13)
    rng = np.random.default_rng(14)
    weights = pd.Series(rng.integers(20, 200, len(y)).astype(float))

    weighted = regression_model(X, y, regression_type="logit", weights=weights)
    plain = regression_model(X, y, regression_type="logit")

    _seen.append(bool(
        np.allclose(weighted.model.var_weights, weights.to_numpy())))
    _seen.append(bool(
        np.allclose(plain.model.var_weights, 1.0)))
    _seen.append(bool(
        isinstance(weighted.model.family.link, Logit)))
    # Weighting by cell count inflates the effective sample size, so the
    # standard errors must shrink relative to the unweighted fit.
    _seen.append(bool(
        weighted.bse["x"] < plain.bse["x"] / 5))
    # Both fits still recover the planted positive slope.
    _seen.append(bool(
        weighted.params["x"] > 0.3 and plain.params["x"] > 0.3))
    return _seen


def case_regression_model_probit_uses_probit_link_and_weights():
    _seen = []
    from spacr.ml import regression_model

    X, y = _fraction_xy(seed=15)
    weights = np.full(len(y), 50.0)

    model = regression_model(X, y, regression_type="probit", weights=weights)

    _seen.append(keep(type(model.model.family.link).__name__))
    _seen.append(bool(
        np.allclose(model.model.var_weights, 50.0)))
    _seen.append(bool(
        np.isfinite(model.params).all()))
    _seen.append(bool(
        model.params["x"] > 0.1))
    return _seen


# ---------------------------------------------------------------------------
# regression_model — mixed dispatch (line 635)
# ---------------------------------------------------------------------------

def case_regression_model_mixed_dispatches_to_perform_mixed_model(capsys):
    _seen = []
    from spacr.ml import regression_model

    rng = np.random.default_rng(16)
    n_groups, per_group = 12, 10
    n = n_groups * per_group
    groups = np.repeat(np.arange(n_groups), per_group)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    y = (1.5 * x1 - 0.8 * x2
         + np.repeat(rng.normal(0, 0.5, n_groups), per_group)
         + rng.normal(0, 0.3, n))

    model = regression_model(X, pd.Series(y), regression_type="mixed",
                             groups=groups, alpha=1.0)

    _seen.append(keep(list(model.fe_params.index)))
    _seen.append(bool(
        model.fe_params["x1"] == pytest.approx(1.5, abs=0.3)))
    _seen.append(bool(
        model.fe_params["x2"] == pytest.approx(-0.8, abs=0.3)))
    _seen.append(('VIF:' if 'VIF:' in (capsys.readouterr().out) else ("missing", 'VIF:')))
    return _seen


def case_regression_model_glm_reports_mcfadden_pseudo_r2(capsys):
    """Count response -> Poisson/Log GLM, plus the McFadden R2 report."""
    _seen = []
    from spacr.ml import regression_model

    rng = np.random.default_rng(18)
    n = 150
    x = rng.normal(0, 1, n)
    X = pd.DataFrame({"const": 1.0, "x": x})
    y = pd.Series(rng.poisson(np.exp(1.2 + 0.6 * x)).astype(float), name="counts")

    model = regression_model(X, y, regression_type="glm")

    _seen.append(keep(type(model.model.family).__name__))
    _seen.append(keep(type(model.model.family.link).__name__))
    _seen.append(bool(
        model.params["x"] == pytest.approx(0.6, abs=0.15)))
    out = capsys.readouterr().out
    # The sentence names the SCALE it examined (instruction 182 A), so it
    # reads "the response looks like counts" rather than "Count data
    # detected" -- the old wording said nothing about which column.
    _seen.append(('looks like counts' if 'looks like counts' in (out) else ("missing", 'looks like counts')))
    _seen.append(("McFadden's R" if "McFadden's R" in (out) else ("missing", "McFadden's R")))
    _seen.append(('Generalized Linear Model Regression Results' if 'Generalized Linear Model Regression Results' in (out) else ("missing", 'Generalized Linear Model Regression Results')))
    # THE PRINTED PSEUDO-R2 MUST MATCH THE FORMULA THE FUNCTION USES, and
    # this pinned the wrong one. It was `1 - llf / (null_deviance / -2)`,
    # which equals McFadden only when the SATURATED log-likelihood is zero.
    # That is true of 0/1 binomial data and false of counts and of the
    # per-well proportions the pipeline actually fits, so the expression
    # mixed a likelihood with a deviance and reported negatives for
    # correctly specified models -- -11.53 for one, measured while wiring
    # instruction 182. `llnull` is the null model statsmodels fits.
    mcfadden = 1 - (model.llf / model.llnull)
    printed = float([l for l in out.splitlines() if "McFadden" in l][0].split(":")[-1])
    _seen.append(bool(
        printed == pytest.approx(mcfadden, abs=1e-4)))
    return _seen


def case_regression_model_mixed_without_groups_raises():
    _seen = []
    from spacr.ml import regression_model

    X, y = _fraction_xy(n=40, seed=17)
    with pytest.raises(ValueError, match="Groups must be defined") as _exc1:
        regression_model(X, y, regression_type="mixed", groups=None)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


# ---------------------------------------------------------------------------
# regression — full pipeline
# ---------------------------------------------------------------------------

def case_regression_ols_keeps_the_design_as_measured(tmp_path, capsys):
    """OLS is fitted on the raw design, intercept intact.

    This test used to assert the opposite -- "OLS goes through MinMax
    scaling" -- and that scaling did two things nothing downstream could see.
    MinMaxScaler maps a zero-range column to all zeros, so patsy's Intercept
    became a column of zeros and statsmodels fitted through the origin while
    still printing an Intercept row of 0.000; and it divided each
    `fraction:grna` column by that gRNA's own maximum, so the coefficients the
    volcano plot ranks against each other had each been rescaled by a
    different constant.
    """
    _seen = []
    from spacr.ml import regression

    df = _wells_df(seed=0)
    csv_path = str(tmp_path / "scores.csv")
    dst = str(tmp_path)

    model, coef_df, reg_type = regression(
        df, csv_path, dependent_variable="predictions",
        regression_type="ols", nc=NC, pc=PC, dst=dst,
    )

    _seen.append(keep(reg_type))
    # `level` says which of the two fits produced the row, so a stacked
    # results.csv can be read back apart. Instruction 132.
    _seen.append(keep(list(coef_df.columns)))
    _seen.append(keep(set(coef_df["level"])))
    # Row / column fixed effects are stripped from the reported coefficients.
    _seen.append(bool(
        not coef_df["feature"].str.contains("row|column").any()))
    # One coefficient per gRNA plus the intercept -- and NO gene terms, which
    # is the collinearity instruction 132 removed.
    _seen.append(bool(
        set(coef_df["feature"]) == EXPECTED_GRNA_FEATURES))
    _seen.append(bool(
        len(coef_df) == 8 + 1))

    # ...and the gene level is a SEPARATE fit of the same wells.
    _model, gene_coef, _kind = regression(
        df, csv_path, dependent_variable="predictions",
        regression_type="ols", nc=NC, pc=PC, dst=dst, level="gene",
    )
    _seen.append(bool(
        set(gene_coef["feature"]) == EXPECTED_GENE_FEATURES))
    _seen.append(keep(set(gene_coef["level"])))
    _seen.append(bool(
        np.allclose(coef_df["-log10(p_value)"], -np.log10(coef_df["p_value"]),
                       equal_nan=True)))
    # Control annotation is driven by the nc / pc identifiers.
    # Two guide rows each, not three: the third was the gene term, and the
    # gene term is a different fit now (instruction 132).
    nc_rows = coef_df[coef_df["feature"].str.contains(NC)]
    pc_rows = coef_df[coef_df["feature"].str.contains(PC)]
    _seen.append(bool(
        set(nc_rows["condition"]) == {"nc"} and len(nc_rows) == 2))
    _seen.append(bool(
        set(pc_rows["condition"]) == {"pc"} and len(pc_rows) == 2))
    gene_nc = gene_coef[gene_coef["feature"].str.contains(NC)]
    _seen.append(bool(
        set(gene_nc["condition"]) == {"nc"} and len(gene_nc) == 1))
    _seen.append(bool(
        hasattr(model, "params") and "Intercept" in model.params.index))

    out = capsys.readouterr().out
    _seen.append(('Using regression type: ols' if 'Using regression type: ols' in (out) else ("missing", 'Using regression type: ols')))
    # The line names the LEVEL too, because a run fits two models now.
    _seen.append(('Performing ols grna-level regression' if 'Performing ols grna-level regression' in (out) else ("missing", 'Performing ols grna-level regression')))
    _seen.append(('Data will not be scaled' if 'Data will not be scaled' in (out) else ("missing", 'Data will not be scaled')))
    # The proof that matters: the intercept column reached the fit as ones.
    intercept = model.model.exog[:, list(model.model.exog_names).index("Intercept")]
    _seen.append(bool(
        np.allclose(intercept, 1.0)))
    # ... and the response was not squeezed into [0, 1] on the way in.
    _seen.append(bool(
        np.allclose(np.sort(np.asarray(model.model.endog).ravel()),
                       np.sort(df["predictions"].to_numpy()))))

    # Both histograms were written; no volcano plot because plot=False.
    _seen.append(bool(
        (tmp_path / "predictions_histogram.pdf").stat().st_size > 0))
    _seen.append(bool(
        (tmp_path / "fraction_histogram.pdf").stat().st_size > 0))
    _seen.append(bool(
        not (tmp_path / "ols_scores_volcano_plot.pdf").exists()))
    return _seen


def case_regression_auto_selects_logit_for_binary_response(tmp_path, capsys):
    _seen = []
    from spacr.ml import regression
    from statsmodels.genmod.families.links import Logit

    df = _wells_df(seed=1, dep_kind="binary")
    _seen.append(keep(set(df["predictions"].unique())))

    model, coef_df, reg_type = regression(
        df, str(tmp_path / "scores.csv"), dependent_variable="predictions",
        regression_type=None, nc=NC, pc=PC, dst=None,
    )

    _seen.append(keep(reg_type))
    out = capsys.readouterr().out
    _seen.append(('Detected binary data.' if 'Detected binary data.' in (out) else ("missing", 'Detected binary data.')))
    _seen.append(('Data will not be scaled' if 'Data will not be scaled' in (out) else ("missing", 'Data will not be scaled')))
    _seen.append(keep(type(model.model.family).__name__))
    _seen.append(bool(
        isinstance(model.model.family.link, Logit)))
    _seen.append(keep(len(coef_df)))
    _seen.append(bool(
        coef_df["p_value"].between(0, 1).all()))
    # dst=None -> nothing is written to disk.
    _seen.append(keep(list(tmp_path.iterdir())))
    return _seen


def case_regression_marks_explicit_controls(tmp_path):
    _seen = []
    from spacr.ml import regression

    df = _wells_df(seed=2)
    _, coef_df, _ = regression(
        df, str(tmp_path / "scores.csv"), dependent_variable="predictions",
        regression_type="ols", nc=NC, pc=PC, controls=["gene3_a"], dst=None,
    )

    by_grna = coef_df.set_index("grna")["condition"]
    _seen.append(keep(by_grna["gene3_a"]))
    _seen.append(keep(by_grna["gene3_b"]))
    _seen.append(keep(by_grna[f"{NC}_a"]))
    _seen.append(keep(by_grna[f"{PC}_b"]))
    return _seen


def case_regression_lasso_skips_scaling_and_reports_pvalues(tmp_path, capsys):
    _seen = []
    from spacr.ml import regression

    df = _wells_df(seed=5)
    model, coef_df, reg_type = regression(
        df, str(tmp_path / "scores.csv"), dependent_variable="predictions",
        regression_type="lasso", alpha=0.01, nc=NC, pc=PC, dst=None,
    )

    _seen.append(keep(reg_type))
    _seen.append(bool(
        hasattr(model, "coef_")))
    _seen.append(keep(len(coef_df)))
    _seen.append(('Data will not be scaled' if 'Data will not be scaled' in (capsys.readouterr().out) else ("missing", 'Data will not be scaled')))
    # lasso/ridge coefficients come straight off the sklearn estimator, so the
    # feature column is the design-matrix column names.
    _seen.append(bool(
        set(coef_df["feature"]) == EXPECTED_FEATURES))
    _seen.append(bool(
        len(model.coef_) >= len(coef_df)))
    return _seen


def case_regression_random_row_column_effects_fits_mixed_model(tmp_path):
    _seen = []
    from spacr.ml import regression

    df = _wells_df(seed=2, n_plates=2, n_rows=2, n_cols=3)
    dst = str(tmp_path)

    model, coef_df, reg_type = regression(
        df, str(tmp_path / "scores.csv"), dependent_variable="predictions",
        random_row_column_effects=True, regression_type="ols", nc=NC, pc=PC, dst=dst,
    )

    # The mixed branch overrides whatever regression_type was requested.
    _seen.append(keep(reg_type))
    # CHANGED BY INSTRUCTION 132 (maintainer, 2026-08-17): mixed is now
    # y ~ gene_fraction:gene + (1 | gene/grna) + rowID + columnID. The gene is
    # a FIXED effect and the guide is a RANDOM effect nested inside it, so
    # `fraction:grna` is not a term of this model at all -- it was half of the
    # collinear design that instruction removed. Each row says what it is, and
    # the guide rows are BLUPs with no p-value.
    from spacr.ml import TERM_BLUP, TERM_FIXED

    _seen.append(keep(list(coef_df.columns)))
    _seen.append(keep(set(coef_df["level"])))
    _seen.append(('Intercept' if 'Intercept' in (set(coef_df["feature"])) else ("missing", 'Intercept')))
    _seen.append(bool(
        not any(f.startswith("fraction:grna[") for f in coef_df["feature"])))
    _seen.append(bool(
        any(f.startswith("gene_fraction:gene[") for f in coef_df["feature"])))
    blups = coef_df[coef_df["term_type"] == TERM_BLUP]
    _seen.append(bool(
        len(blups) == df["grna"].nunique()))
    _seen.append(bool(
        blups["p_value"].isna().all()))
    _seen.append(bool(
        coef_df.loc[coef_df["term_type"] == TERM_FIXED,
                       "p_value"].notna().any()))
    _seen.append(bool(
        hasattr(model, "fe_params")))
    _seen.append(bool(
        len(model.resid) == len(df)))
    # The mixed branch produces fit_mixed_model's residual histogram; the
    # dependent / fraction distributions belong to the fixed-effects branch
    # and are not drawn here.
    written = sorted(p.name for p in tmp_path.iterdir())
    _seen.append(('residuals_histogram.pdf' if 'residuals_histogram.pdf' in (written) else ("missing", 'residuals_histogram.pdf')))
    _seen.append(bool(
        not any(name.startswith("fraction_histogram") for name in written)))

    # AND THE HOUSE-STYLE OUTPUT, which every branch now produces: the panel
    # sheet and its generated legend. A mixed fit is still a fit and the
    # reader still needs the figure.
    _seen.append(('regression_figure.pdf' if 'regression_figure.pdf' in (written) else ("missing", 'regression_figure.pdf')))
    _seen.append(('regression_figure_legend.txt' if 'regression_figure_legend.txt' in (written) else ("missing", 'regression_figure_legend.txt')))
    return _seen


def case_the_legacy_volcano_is_written_only_when_it_is_asked_for(tmp_path):
    """"hide my old version behid a boolean that defaults to off".

    The original matplotlib volcano is not deleted -- it is what published
    figures were made with -- but a run does not draw it unless asked, and a
    run that draws both puts two volcanoes in two visual idioms on the same
    grid.
    """
    _seen = []
    from spacr.ml import regression

    df = _wells_df(seed=3)
    csv_path = str(tmp_path / "scores.csv")
    regression(
        df, csv_path, dependent_variable="predictions",
        regression_type="ols", nc=NC, pc=PC, dst=str(tmp_path), plot=True,
    )
    _seen.append(bool(
        not (tmp_path / "ols_scores_volcano_plot.pdf").exists()))

    regression(
        df, csv_path, dependent_variable="predictions",
        regression_type="ols", nc=NC, pc=PC, dst=str(tmp_path), plot=True,
        legacy_volcano=True,
    )
    _seen.append(bool(
        (tmp_path / "ols_scores_volcano_plot.pdf").stat().st_size > 0))
    return _seen


def case_regression_passes_cell_count_weights_to_logit(tmp_path, monkeypatch):
    _seen = []
    from spacr import ml as ML

    seen = {}
    real_regression_model = ML.regression_model

    def _spy(X, y, **kwargs):
        seen.update(kwargs)
        return real_regression_model(X, y, **kwargs)

    monkeypatch.setattr(ML, "regression_model", _spy)

    df = _wells_df(seed=6, dep_kind="binary")
    ML.regression(df, str(tmp_path / "scores.csv"), dependent_variable="predictions",
                  regression_type="logit", nc=NC, pc=PC, dst=None)

    _seen.append(keep(seen["regression_type"]))
    _seen.append(bool(
        seen["weights"] is not None))
    _seen.append(bool(
        np.allclose(np.asarray(seen["weights"], dtype=float),
                       df["cell_count"].to_numpy(dtype=float))))
    return _seen


def case_regression_rejects_unsupported_type(tmp_path):
    _seen = []
    from spacr.ml import regression

    df = _wells_df(seed=7)
    with pytest.raises(ValueError, match="Unsupported regression type") as _exc1:
        regression(df, str(tmp_path / "scores.csv"),
                   dependent_variable="predictions",
                   regression_type="not_a_model", dst=None)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


# ---------------------------------------------------------------------------
# save_summary_to_file
# ---------------------------------------------------------------------------

def case_save_summary_to_file_writes_model_summary(tmp_path):
    _seen = []
    import statsmodels.api as sm
    from spacr.ml import save_summary_to_file

    rng = np.random.default_rng(21)
    X = sm.add_constant(rng.normal(0, 1, (60, 2)))
    y = X @ np.array([0.5, 2.0, -1.0]) + rng.normal(0, 0.2, 60)
    model = sm.OLS(y, X).fit()

    out = tmp_path / "summary.csv"
    save_summary_to_file(model, file_path=str(out))

    text = out.read_text(encoding="utf-8")
    _seen.append(('OLS Regression Results' if 'OLS Regression Results' in (text) else ("missing", 'OLS Regression Results')))
    _seen.append(('R-squared' if 'R-squared' in (text) else ("missing", 'R-squared')))

    # statsmodels stamps a second-resolution "Date:"/"Time:" row into every
    # summary, so comparing the whole text against a freshly rendered one is a
    # race with the wall clock: it fails whenever a second ticks between
    # save_summary_to_file's render and this one. Compare the rest.
    def _stable(summary_text):
        return [line for line in summary_text.splitlines()
                if not line.lstrip().startswith(("Date:", "Time:"))]

    _seen.append(bool(
        _stable(text) == _stable(model.summary().as_text())))
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param(case_regression_model_find_best_alpha_rejects_unknown_model, ['RuntimeError', True, 'ValueError'], id="regression_model_find_best_alpha_rejects_unknown_model"),
    pytest.param(case_regression_model_logit_applies_var_weights, [True, True, True, True, True], id="regression_model_logit_applies_var_weights"),
    pytest.param(case_regression_model_probit_uses_probit_link_and_weights, ['probit', True, True, True], id="regression_model_probit_uses_probit_link_and_weights"),
    pytest.param(case_regression_model_mixed_dispatches_to_perform_mixed_model, [['x1', 'x2'], True, True, 'VIF:'], id="regression_model_mixed_dispatches_to_perform_mixed_model"),
    pytest.param(case_regression_model_glm_reports_mcfadden_pseudo_r2, ['Poisson', 'Log', True, 'looks like counts', "McFadden's R", 'Generalized Linear Model Regression Results', True], id="regression_model_glm_reports_mcfadden_pseudo_r2"),
    pytest.param(case_regression_model_mixed_without_groups_raises, ['ValueError'], id="regression_model_mixed_without_groups_raises"),
    pytest.param(case_regression_passes_cell_count_weights_to_logit, ['logit', True, True], id="regression_passes_cell_count_weights_to_logit"),
    pytest.param(case_regression_ols_keeps_the_design_as_measured, ['ols', ['feature', 'coefficient', 'p_value', '-log10(p_value)', 'grna', 'condition', 'level'], {'grna'}, True, True, True, True, {'gene'}, True, True, True, True, True, 'Using regression type: ols', 'Performing ols grna-level regression', 'Data will not be scaled', True, True, True, True, True], id="regression_ols_keeps_the_design_as_measured"),
    pytest.param(case_regression_auto_selects_logit_for_binary_response, [{0.0, 1.0}, 'logit', 'Detected binary data.', 'Data will not be scaled', 'Binomial', True, 9, True, []], id="regression_auto_selects_logit_for_binary_response"),
    pytest.param(case_regression_marks_explicit_controls, ['control', 'other', 'nc', 'pc'], id="regression_marks_explicit_controls"),
    pytest.param(case_regression_lasso_skips_scaling_and_reports_pvalues, ['lasso', True, 9, 'Data will not be scaled', True, True], id="regression_lasso_skips_scaling_and_reports_pvalues"),
    pytest.param(case_regression_random_row_column_effects_fits_mixed_model, ['mixed', ['feature', 'coefficient', 'p_value', 'term_type', 'level'], {'gene'}, 'Intercept', True, True, True, True, True, True, True, 'residuals_histogram.pdf', True, 'regression_figure.pdf', 'regression_figure_legend.txt'], id="regression_random_row_column_effects_fits_mixed_model"),
    pytest.param(case_the_legacy_volcano_is_written_only_when_it_is_asked_for, [True, True], id="the_legacy_volcano_is_written_only_when_it_is_asked_for"),
    pytest.param(case_regression_rejects_unsupported_type, ['ValueError'], id="regression_rejects_unsupported_type"),
    pytest.param(case_save_summary_to_file_writes_model_summary, ['OLS Regression Results', 'R-squared', True], id="save_summary_to_file_writes_model_summary"),
])
def test_ml_regression_core_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected

