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

# Coefficients patsy produces for the fixed-effects formula, minus the
# row / column terms that process_model_coefficients filters out.
EXPECTED_FEATURES = (
    {"Intercept"}
    | {f"fraction:grna[{g}_{s}]" for g in GENES for s in ("a", "b")}
    | {f"gene_fraction:gene[{g}]" for g in GENES}
)


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


def test_regression_model_find_best_alpha_rejects_unknown_model(monkeypatch):
    """The ``_find_best_alpha`` guard clause raises on an unknown backend.

    The helper is a closure, so it is recovered from the traceback frame of
    ``regression_model`` after injecting a failure into ``LassoCV.fit``.
    """
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
    tb = excinfo.value.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_name == "regression_model":
            finder = tb.tb_frame.f_locals.get("_find_best_alpha")
        tb = tb.tb_next

    assert callable(finder), "could not recover _find_best_alpha from the frame"
    with pytest.raises(ValueError, match="unknown model_cls='bogus'"):
        finder("bogus")


# ---------------------------------------------------------------------------
# regression_model — GLM-binomial var_weights (line 613)
# ---------------------------------------------------------------------------

def test_regression_model_logit_applies_var_weights():
    from spacr.ml import regression_model

    X, y = _fraction_xy(seed=13)
    rng = np.random.default_rng(14)
    weights = pd.Series(rng.integers(20, 200, len(y)).astype(float))

    weighted = regression_model(X, y, regression_type="logit", weights=weights)
    plain = regression_model(X, y, regression_type="logit")

    assert np.allclose(weighted.model.var_weights, weights.to_numpy())
    assert np.allclose(plain.model.var_weights, 1.0)
    assert type(weighted.model.family.link).__name__ == "logit"
    # Weighting by cell count inflates the effective sample size, so the
    # standard errors must shrink relative to the unweighted fit.
    assert weighted.bse["x"] < plain.bse["x"] / 5
    # Both fits still recover the planted positive slope.
    assert weighted.params["x"] > 0.3 and plain.params["x"] > 0.3


def test_regression_model_probit_uses_probit_link_and_weights():
    from spacr.ml import regression_model

    X, y = _fraction_xy(seed=15)
    weights = np.full(len(y), 50.0)

    model = regression_model(X, y, regression_type="probit", weights=weights)

    assert type(model.model.family.link).__name__ == "probit"
    assert np.allclose(model.model.var_weights, 50.0)
    assert np.isfinite(model.params).all()
    assert model.params["x"] > 0.1


# ---------------------------------------------------------------------------
# regression_model — mixed dispatch (line 635)
# ---------------------------------------------------------------------------

def test_regression_model_mixed_dispatches_to_perform_mixed_model(capsys):
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

    assert list(model.fe_params.index) == ["x1", "x2"]
    assert model.fe_params["x1"] == pytest.approx(1.5, abs=0.3)
    assert model.fe_params["x2"] == pytest.approx(-0.8, abs=0.3)
    assert "VIF:" in capsys.readouterr().out


def test_regression_model_glm_reports_mcfadden_pseudo_r2(capsys):
    """Count response -> Poisson/Log GLM, plus the McFadden R2 report."""
    from spacr.ml import regression_model

    rng = np.random.default_rng(18)
    n = 150
    x = rng.normal(0, 1, n)
    X = pd.DataFrame({"const": 1.0, "x": x})
    y = pd.Series(rng.poisson(np.exp(1.2 + 0.6 * x)).astype(float), name="counts")

    model = regression_model(X, y, regression_type="glm")

    assert type(model.model.family).__name__ == "Poisson"
    assert type(model.model.family.link).__name__ == "Log"
    assert model.params["x"] == pytest.approx(0.6, abs=0.15)
    out = capsys.readouterr().out
    assert "Count data detected" in out
    assert "McFadden's R" in out
    assert "Generalized Linear Model Regression Results" in out
    # The printed pseudo-R2 must match the formula the function uses.
    mcfadden = 1 - (model.llf / (model.null_deviance / -2))
    printed = float([l for l in out.splitlines() if "McFadden" in l][0].split(":")[-1])
    assert printed == pytest.approx(mcfadden, abs=1e-4)


def test_regression_model_mixed_without_groups_raises():
    from spacr.ml import regression_model

    X, y = _fraction_xy(n=40, seed=17)
    with pytest.raises(ValueError, match="Groups must be defined"):
        regression_model(X, y, regression_type="mixed", groups=None)


# ---------------------------------------------------------------------------
# regression — full pipeline
# ---------------------------------------------------------------------------

def test_regression_ols_scales_and_returns_coefficients(tmp_path, capsys):
    from spacr.ml import regression

    df = _wells_df(seed=0)
    csv_path = str(tmp_path / "scores.csv")
    dst = str(tmp_path)

    model, coef_df, reg_type = regression(
        df, csv_path, dependent_variable="predictions",
        regression_type="ols", nc=NC, pc=PC, dst=dst,
    )

    assert reg_type == "ols"
    assert list(coef_df.columns) == [
        "feature", "coefficient", "p_value", "-log10(p_value)", "grna", "condition",
    ]
    # Row / column fixed effects are stripped from the reported coefficients.
    assert not coef_df["feature"].str.contains("row|column").any()
    # One coefficient per gRNA plus one per gene plus the intercept.
    assert set(coef_df["feature"]) == EXPECTED_FEATURES
    assert len(coef_df) == 8 + 4 + 1
    assert np.allclose(coef_df["-log10(p_value)"], -np.log10(coef_df["p_value"]),
                       equal_nan=True)
    # Control annotation is driven by the nc / pc identifiers.
    nc_rows = coef_df[coef_df["feature"].str.contains(NC)]
    pc_rows = coef_df[coef_df["feature"].str.contains(PC)]
    assert set(nc_rows["condition"]) == {"nc"} and len(nc_rows) == 3
    assert set(pc_rows["condition"]) == {"pc"} and len(pc_rows) == 3
    assert hasattr(model, "params") and "Intercept" in model.params.index

    out = capsys.readouterr().out
    assert "Using regression type: ols" in out
    assert "Performing ols regression" in out
    assert "Data will not be scaled" not in out      # OLS goes through MinMax scaling

    # Both histograms were written; no volcano plot because plot=False.
    assert (tmp_path / "predictions_histogram.pdf").stat().st_size > 0
    assert (tmp_path / "fraction_histogram.pdf").stat().st_size > 0
    assert not (tmp_path / "ols_scores_volcano_plot.pdf").exists()


def test_regression_auto_selects_logit_for_binary_response(tmp_path, capsys):
    from spacr.ml import regression

    df = _wells_df(seed=1, dep_kind="binary")
    assert set(df["predictions"].unique()) == {0.0, 1.0}

    model, coef_df, reg_type = regression(
        df, str(tmp_path / "scores.csv"), dependent_variable="predictions",
        regression_type=None, nc=NC, pc=PC, dst=None,
    )

    assert reg_type == "logit"
    out = capsys.readouterr().out
    assert "Detected binary data." in out
    assert "Data will not be scaled" in out          # bounded response is left unscaled
    assert type(model.model.family).__name__ == "Binomial"
    assert type(model.model.family.link).__name__ == "logit"
    assert len(coef_df) == 13
    assert coef_df["p_value"].between(0, 1).all()
    # dst=None -> nothing is written to disk.
    assert list(tmp_path.iterdir()) == []


def test_regression_marks_explicit_controls(tmp_path):
    from spacr.ml import regression

    df = _wells_df(seed=2)
    _, coef_df, _ = regression(
        df, str(tmp_path / "scores.csv"), dependent_variable="predictions",
        regression_type="ols", nc=NC, pc=PC, controls=["gene3_a"], dst=None,
    )

    by_grna = coef_df.set_index("grna")["condition"]
    assert by_grna["gene3_a"] == "control"
    assert by_grna["gene3_b"] == "other"
    assert by_grna[f"{NC}_a"] == "nc"
    assert by_grna[f"{PC}_b"] == "pc"


def test_regression_lasso_skips_scaling_and_reports_pvalues(tmp_path, capsys):
    from spacr.ml import regression

    df = _wells_df(seed=5)
    model, coef_df, reg_type = regression(
        df, str(tmp_path / "scores.csv"), dependent_variable="predictions",
        regression_type="lasso", alpha=0.01, nc=NC, pc=PC, dst=None,
    )

    assert reg_type == "lasso"
    assert hasattr(model, "coef_")
    assert len(coef_df) == 13
    assert "Data will not be scaled" in capsys.readouterr().out
    # lasso/ridge coefficients come straight off the sklearn estimator, so the
    # feature column is the design-matrix column names.
    assert set(coef_df["feature"]) == EXPECTED_FEATURES
    assert len(model.coef_) >= len(coef_df)


def test_regression_random_row_column_effects_fits_mixed_model(tmp_path):
    from spacr.ml import regression

    df = _wells_df(seed=2, n_plates=2, n_rows=2, n_cols=3)
    dst = str(tmp_path)

    model, coef_df, reg_type = regression(
        df, str(tmp_path / "scores.csv"), dependent_variable="predictions",
        random_row_column_effects=True, regression_type="ols", nc=NC, pc=PC, dst=dst,
    )

    # The mixed branch overrides whatever regression_type was requested.
    assert reg_type == "mixed"
    assert list(coef_df.columns) == ["feature", "coefficient", "p_value"]
    assert "Intercept" in set(coef_df["feature"])
    assert any(f.startswith("fraction:grna[") for f in coef_df["feature"])
    assert hasattr(model, "fe_params")
    assert len(model.resid) == len(df)
    # Only fit_mixed_model's residual histogram is produced — the dependent /
    # fraction histograms belong to the fixed-effects branch.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["residuals_histogram.pdf"]


def test_regression_plot_true_writes_volcano_pdf(tmp_path):
    from spacr.ml import regression

    df = _wells_df(seed=3)
    csv_path = str(tmp_path / "scores.csv")
    regression(
        df, csv_path, dependent_variable="predictions",
        regression_type="ols", nc=NC, pc=PC, dst=str(tmp_path), plot=True,
    )
    assert (tmp_path / "ols_scores_volcano_plot.pdf").stat().st_size > 0


def test_regression_passes_cell_count_weights_to_logit(tmp_path, monkeypatch):
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

    assert seen["regression_type"] == "logit"
    assert seen["weights"] is not None
    assert np.allclose(np.asarray(seen["weights"], dtype=float),
                       df["cell_count"].to_numpy(dtype=float))


def test_regression_rejects_unsupported_type(tmp_path):
    from spacr.ml import regression

    df = _wells_df(seed=7)
    with pytest.raises(ValueError, match="Unsupported regression type"):
        regression(df, str(tmp_path / "scores.csv"),
                   dependent_variable="predictions",
                   regression_type="not_a_model", dst=None)


# ---------------------------------------------------------------------------
# save_summary_to_file
# ---------------------------------------------------------------------------

def test_save_summary_to_file_writes_model_summary(tmp_path):
    import statsmodels.api as sm
    from spacr.ml import save_summary_to_file

    rng = np.random.default_rng(21)
    X = sm.add_constant(rng.normal(0, 1, (60, 2)))
    y = X @ np.array([0.5, 2.0, -1.0]) + rng.normal(0, 0.2, 60)
    model = sm.OLS(y, X).fit()

    out = tmp_path / "summary.csv"
    save_summary_to_file(model, file_path=str(out))

    text = out.read_text()
    assert "OLS Regression Results" in text
    assert "R-squared" in text

    # statsmodels stamps a second-resolution "Date:"/"Time:" row into every
    # summary, so comparing the whole text against a freshly rendered one is a
    # race with the wall clock: it fails whenever a second ticks between
    # save_summary_to_file's render and this one. Compare the rest.
    def _stable(summary_text):
        return [line for line in summary_text.splitlines()
                if not line.lstrip().startswith(("Date:", "Time:"))]

    assert _stable(text) == _stable(model.summary().as_text())
