"""Everything a sweep row needs to be worth reading.

A row that says "10 hits" and nothing else cannot be used to CHOOSE a
configuration, which is the only reason to run a sweep. The nightly run over
the TSG101 screen made that concrete: ``min_cell_count=50`` reported MORE hits
than 100 while quietly losing GRA14, and no column in the table said so.

So a row carries four kinds of thing, and they answer different questions:

    fit quality     is the model describing the data at all
    residuals       are its standard errors, and therefore its p-values, real
    design          is the thing even identifiable, and how much data reached it
    controls        does it recover what is known to be true

The controls matter most. When the user has named a positive control, it is
the yardstick a configuration should be read against -- a setting that buries
a gene known to be real has told you something no goodness-of-fit statistic
will.

Every function here returns a flat dict of scalars, because a sweep row is a
row: nothing nested, nothing that needs unpacking to sort on.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd


def _first_column(frame: pd.DataFrame, *names: str) -> Optional[str]:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _numeric(frame: pd.DataFrame, column: Optional[str]) -> np.ndarray:
    if not column or column not in frame.columns:
        return np.array([], dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")


def fit_quality(model) -> dict:
    """R-squared and the information criteria, when the model reports them.

    Read defensively: the thirteen families spaCR fits do not agree on which
    of these exist. A robust fit has no R-squared, a penalised one's is not
    comparable to OLS's, and a permutation test has no model object at all.
    Reporting NaN for an absent statistic is honest; inventing one is not.
    """
    out: dict[str, Any] = {}
    if model is None:
        return out
    for key, attribute in (("r_squared", "rsquared"),
                           ("r_squared_adj", "rsquared_adj"),
                           ("aic", "aic"), ("bic", "bic"),
                           ("log_likelihood", "llf"),
                           ("f_pvalue", "f_pvalue"),
                           ("condition_number", "condition_number")):
        try:
            value = getattr(model, attribute, None)
            if value is not None and np.isfinite(float(value)):
                out[key] = float(value)
        except (TypeError, ValueError):
            pass
    try:
        residuals = np.asarray(getattr(model, "resid", []), dtype="float64")
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size:
            df = getattr(model, "df_resid", None) or max(residuals.size - 1, 1)
            out["residual_se"] = float(np.sqrt(np.sum(residuals ** 2) / df))
            out["n_observations"] = int(residuals.size)
    except (TypeError, ValueError):
        pass
    return out


def residual_diagnostics(model) -> dict:
    """Homoscedasticity, autocorrelation and normality of the residuals.

    These are what decide whether the p-values mean anything. A funnel in the
    residuals inflates or deflates every standard error in the fit, so a
    heteroscedastic model can rank genes plausibly and still be wrong about
    all of them.
    """
    out: dict[str, Any] = {}
    if model is None:
        return out
    try:
        residuals = np.asarray(getattr(model, "resid", []), dtype="float64")
        fitted = np.asarray(getattr(model, "fittedvalues", []), dtype="float64")
    except (TypeError, ValueError):
        return out
    good = np.isfinite(residuals) & np.isfinite(fitted) \
        if residuals.size == fitted.size else np.isfinite(residuals)
    residuals = residuals[good] if residuals.size else residuals
    if residuals.size < 8:
        return out

    try:
        from statsmodels.stats.stattools import durbin_watson, jarque_bera

        out["durbin_watson"] = float(durbin_watson(residuals))
        jb, jb_p, skew, kurtosis = jarque_bera(residuals)
        out["jarque_bera_p"] = float(jb_p)
        out["residual_skew"] = float(skew)
        out["residual_kurtosis"] = float(kurtosis)
    except Exception:  # pragma: no cover - statsmodels shape varies
        pass

    exog = getattr(model, "model", None)
    exog = getattr(exog, "exog", None) if exog is not None else None
    if exog is not None:
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan, het_white

            exog = np.asarray(exog, dtype="float64")[good] \
                if len(exog) == len(good) else np.asarray(exog, dtype="float64")
            if len(exog) == residuals.size:
                out["breusch_pagan_p"] = float(
                    het_breuschpagan(residuals, exog)[1])
                # White's test squares every column pair; on a wide screen
                # design that is thousands of terms and minutes of work, so it
                # is only worth attempting on a narrow one.
                if exog.shape[1] <= 30:
                    out["white_p"] = float(het_white(residuals, exog)[1])
        except Exception:  # pragma: no cover - singular or too wide
            pass

    if fitted.size == residuals.size and residuals.size > 2:
        try:
            slope, _intercept = np.polyfit(fitted[good], residuals, 1)
            out["residual_trend_slope"] = float(slope)
        except (np.linalg.LinAlgError, ValueError):
            pass
    return out


def control_recovery(results: pd.DataFrame, settings: Mapping[str, Any]) -> dict:
    """Where the named controls landed. The yardstick, when there is one.

    A configuration that buries a gene known to be real has said something no
    goodness-of-fit statistic will, and one that promotes a NEGATIVE control
    has said something worse.
    """
    out: dict[str, Any] = {}
    if results is None or not len(results):
        return out
    feature = _first_column(results, "feature")
    if not feature:
        return out

    p_column = _first_column(results, "p_value")
    q_column = _first_column(results, "q_value", "adjusted_p_value")
    frame = results.copy()
    frame["_p"] = _numeric(frame, p_column)
    # Rank among real coefficients: the intercept is not a candidate hit and
    # counting it shifts every rank by one.
    frame = frame[~frame[feature].astype(str).str.lower()
                  .str.contains("intercept", na=False)]
    frame = frame.sort_values("_p").reset_index(drop=True)
    frame["_rank"] = np.arange(1, len(frame) + 1)

    for key, label in (("positive_control", "positive"),
                       ("negative_control", "negative")):
        identifier = settings.get(key)
        if identifier in (None, ""):
            continue
        hit = frame[frame[feature].astype(str).str.contains(
            str(identifier), na=False, regex=False)]
        if not len(hit):
            out[f"{label}_control_found"] = False
            continue
        best = hit.iloc[0]
        out[f"{label}_control_found"] = True
        out[f"{label}_control_rank"] = int(best["_rank"])
        out[f"{label}_control_p"] = float(best["_p"]) \
            if np.isfinite(best["_p"]) else None
        if q_column:
            try:
                out[f"{label}_control_q"] = float(
                    pd.to_numeric(best[q_column], errors="coerce"))
            except (TypeError, ValueError):
                pass
    # One number for "did the assay work": how far the positive control sits
    # above the negative one in rank.
    if out.get("positive_control_rank") and out.get("negative_control_rank"):
        out["control_rank_separation"] = int(
            out["negative_control_rank"] - out["positive_control_rank"])
    return out


def calibration(results: pd.DataFrame) -> dict:
    """Is the null flat? Inflation above 1 says the hits are partly artefact."""
    out: dict[str, Any] = {}
    if results is None or not len(results):
        return out
    p = _numeric(results, _first_column(results, "p_value"))
    p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    if p.size < 10:
        return out
    observed = -np.log10(np.sort(p))
    expected = -np.log10((np.arange(1, p.size + 1) - 0.5) / p.size)
    median_expected = float(np.median(expected))
    if median_expected:
        out["genomic_inflation"] = float(np.median(observed) / median_expected)
    # A screen with signal has a spike in the first bin; a flat histogram with
    # no spike means nothing was found, and a SLOPING one means the model is
    # misspecified regardless of how many hits it reports.
    counts, _edges = np.histogram(p, bins=20, range=(0.0, 1.0))
    out["p_first_bin_excess"] = int(max(counts[0] - p.size / 20, 0))
    out["n_tests"] = int(p.size)
    return out


def design_summary(output: Mapping[str, Any]) -> dict:
    """How much data reached the fit, and whether it was identifiable."""
    out: dict[str, Any] = {}
    frame = output.get("model_data") if isinstance(output, Mapping) else None
    if isinstance(frame, pd.DataFrame):
        out["n_rows_fitted"] = int(len(frame))
        for key, column in (("n_wells", "prc"), ("n_guides", "grna")):
            if column in frame.columns:
                out[key] = int(frame[column].nunique())
    return out


def guide_support_summary(results: pd.DataFrame, alpha: float = 0.05) -> dict:
    """How many of the hits rest on a single guide.

    A gene with one surviving guide has a gene-level p identical to that
    guide's, so it is not independent evidence -- and on this screen the top
    of the list is exactly that. A count of them belongs in the row.
    """
    out: dict[str, Any] = {}
    try:
        from .guide_concordance import guide_support
    except Exception:  # pragma: no cover
        return out
    try:
        support = guide_support(results, alpha=alpha)
    except Exception:  # pragma: no cover - odd table
        return out
    if support is None or not len(support) or "gene_p" not in support:
        return out
    hits = support[pd.to_numeric(support["gene_p"], errors="coerce") <= alpha]
    out["n_genes_tested"] = int(len(support))
    out["n_gene_hits"] = int(len(hits))
    if len(hits):
        out["n_single_guide_hits"] = int(hits["single_guide"].sum())
        out["n_discordant_hits"] = int((hits["concordance"] < 0.6).sum())
        out["median_guides_per_hit"] = float(hits["n_guides"].median())
    return out


def hit_counts(output: Mapping[str, Any], alpha: float = 0.05) -> dict:
    """How many things the trial called, raw and corrected."""
    out: dict[str, Any] = {}
    if not isinstance(output, Mapping):
        return out
    for key in ("results", "significant", "primary"):
        frame = output.get(key)
        if isinstance(frame, pd.DataFrame):
            out[f"n_{key}"] = int(len(frame))
    results = output.get("results")
    if isinstance(results, pd.DataFrame) and len(results):
        p = _numeric(results, _first_column(results, "p_value"))
        if p.size:
            out["n_raw_below_alpha"] = int(np.nansum(p <= alpha))
        q = _numeric(results, _first_column(results, "q_value",
                                            "adjusted_p_value"))
        if q.size:
            out["n_below_alpha"] = int(np.nansum(q <= alpha))
    return out


def summarise_trial(output: Mapping[str, Any],
                    settings: Mapping[str, Any]) -> dict:
    """Every metric, flat, for one trial's row.

    Each block is guarded on its own: a family with no R-squared must still
    contribute its control recovery, and a design too wide for White's test
    must still contribute its residual trend. One missing statistic is a NaN
    in one column, never a lost row.
    """
    alpha = float(settings.get("fdr_alpha", 0.05) or 0.05)
    results = output.get("results") if isinstance(output, Mapping) else None
    model = output.get("model") if isinstance(output, Mapping) else None

    row: dict[str, Any] = {}
    for block in (
        lambda: hit_counts(output, alpha),
        lambda: design_summary(output),
        lambda: fit_quality(model),
        lambda: residual_diagnostics(model),
        lambda: control_recovery(results, settings),
        lambda: calibration(results),
        lambda: guide_support_summary(results, alpha),
    ):
        try:
            row.update(block())
        except Exception:  # pragma: no cover - a metric must not sink a trial
            pass
    return row
