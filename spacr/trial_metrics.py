"""Everything a sweep row needs to be worth reading.

A row that reports only a hit count is insufficient for choosing a sweep
configuration: a larger count can coincide with losing a known positive
control. Each row therefore records model quality, design diagnostics, and
control recovery alongside its hit count.

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

EVERYTHING HERE IS CHEAP, AND THAT IS THE DESIGN CONSTRAINT. A sweep runs
hundreds of trials, so a per-trial diagnostic that costs seconds costs hours
across the run. The full 23-panel suite in :mod:`spacr.regression_qc` was
measured at ~5.8 s per fit -- ten minutes per hundred trials, for pictures
nobody opens while a sweep is running. These scalars were measured at 73-230 ms
on screen-shaped fits (606-2400 rows, 200-400 parameters), which is under half
a percent of the ~60 s a real trial takes. So a sweep gets the NUMBERS on every
row and the PICTURES only for the rows worth reopening.

Nothing here refits anything. Every statistic is read off the model the trial
already fitted, which is what keeps it cheap.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd


def _first_column(frame: pd.DataFrame, *names: str) -> Optional[str]:
    """Return the first candidate present in ``frame``, or ``None``."""
    for name in names:
        if name in frame.columns:
            return name
    return None


def _numeric(frame: pd.DataFrame, column: Optional[str]) -> np.ndarray:
    """Coerce a present column to a float array, or return an empty array."""
    if not column or column not in frame.columns:
        return np.array([], dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")


def fit_quality(model) -> dict:
    """R-squared and the information criteria, when the model reports them.

    :param model: fitted model result object, or ``None``.

    The supported model families do not agree on which statistics exist. An
    RLM fit has no R-squared, a penalised model's value is not
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

    :param model: fitted model result object, or ``None``.

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
    except Exception:
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
                if exog.shape[1] <= 30:
                    out["white_p"] = float(het_white(residuals, exog)[1])
        except Exception:
            pass

    if fitted.size == good.size and residuals.size > 2:
        try:
            slope, _intercept = np.polyfit(fitted[good], residuals, 1)
            out["residual_trend_slope"] = float(slope)
        except (np.linalg.LinAlgError, ValueError):
            pass
    return out


#: The values :func:`spacr.ml.regression` writes into the ``condition`` column
#: of every coefficient table, one per control role. Reading this column is
#: preferable to matching the identifier against the feature name: it is the
#: SAME verdict the volcano plot colours by, so a row and its figure cannot
#: disagree about which coefficient the positive control is.
_CONDITION_LABELS = {"positive": "pc", "negative": "nc"}


def control_recovery(results: pd.DataFrame, settings: Mapping[str, Any]) -> dict:
    """Summarize where configured controls rank in a result table.

    Both rank and percentile are returned because result tables can contain
    different numbers of coefficients. Missing controls are reported with a
    false ``*_control_found`` value and no invented rank.

    Parameters
    ----------
    results : pandas.DataFrame
        Coefficient table produced by a regression trial.
    settings : mapping
        Trial settings containing optional positive and negative controls.

    Returns
    -------
    dict
        Control presence, rank, percentile, effect, and significance fields.
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
    frame = frame[~frame[feature].astype(str).str.lower()
                  .str.contains("intercept", na=False)]
    if not len(frame):
        return out
    frame = frame.sort_values("_p").reset_index(drop=True)
    frame["_rank"] = np.arange(1, len(frame) + 1)
    n_ranked = int(len(frame))
    condition = _first_column(frame, "condition")

    for key, label in (("positive_control_id", "positive"),
                       ("negative_control_id", "negative")):
        identifier = settings.get(key)
        if identifier in (None, ""):
            continue
        out["n_ranked"] = n_ranked
        hit = pd.DataFrame()
        if condition:
            hit = frame[frame[condition].astype(str).str.strip().str.lower()
                        == _CONDITION_LABELS[label]]
        if not len(hit):
            hit = frame[frame[feature].astype(str).str.contains(
                str(identifier), na=False, regex=False)]
        if not len(hit):
            out[f"{label}_control_found"] = False
            continue
        best = hit.iloc[0]
        out[f"{label}_control_found"] = True
        out[f"{label}_control_n_coefficients"] = int(len(hit))
        out[f"{label}_control_rank"] = int(best["_rank"])
        out[f"{label}_control_percentile"] = float(
            (best["_rank"] - 1) / n_ranked) if n_ranked else float("nan")
        out[f"{label}_control_p"] = float(best["_p"]) \
            if np.isfinite(best["_p"]) else None
        if q_column:
            try:
                out[f"{label}_control_q"] = float(
                    pd.to_numeric(best[q_column], errors="coerce"))
            except (TypeError, ValueError):
                pass
    if out.get("positive_control_rank") and out.get("negative_control_rank"):
        out["control_rank_separation"] = int(
            out["negative_control_rank"] - out["positive_control_rank"])
    return out


def calibration(results: pd.DataFrame) -> dict:
    """Is the null flat? Inflation above 1 says the hits are partly artefact.

    :param results: result table carrying raw ``p_value`` values.
    """
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
    counts, _edges = np.histogram(p, bins=20, range=(0.0, 1.0))
    out["p_first_bin_excess"] = int(max(counts[0] - p.size / 20, 0))
    out["n_tests"] = int(p.size)
    return out


def design_summary(output: Mapping[str, Any]) -> dict:
    """How much data reached the fit, and whether it was identifiable.

    :param output: trial output mapping optionally carrying ``model_data``.
    """
    out: dict[str, Any] = {}
    frame = output.get("model_data") if isinstance(output, Mapping) else None
    if isinstance(frame, pd.DataFrame):
        out["n_rows_fitted"] = int(len(frame))
        for key, column in (("n_wells", "prc"), ("n_guides", "grna")):
            if column in frame.columns:
                out[key] = int(frame[column].nunique())
    return out


def design_diagnostics(model) -> dict:
    """Rank, identifiability and collinearity -- read off the existing fit.

    :param model: fitted model result object, or ``None``.

    NOTHING IS RECOMPUTED HERE THAT THE FIT ALREADY KNOWS.
    :func:`spacr.regression_diagnostics.design_report` answers the same
    questions, but it takes the well-by-guide matrix and pays for a fresh
    ``matrix_rank`` and a full SVD. A sweep does not have that matrix -- it has
    a fitted model -- and statsmodels computed the rank at fit time and kept it
    on ``model.model.rank``. The field names below deliberately match
    ``design_report``'s so the two read the same in a trial folder.

    ``identifiable`` is the one that decides whether the rest of the row means
    anything. A rank-deficient design does not have unique coefficients, so its
    per-guide effects are one of infinitely many solutions and its p-values
    rank an arbitrary choice among them. spaCR fits these designs happily --
    statsmodels falls back to a pseudo-inverse rather than refusing -- so
    nothing else in the output says it happened.
    """
    out: dict[str, Any] = {}
    if model is None:
        return out
    inner = getattr(model, "model", None)
    exog = getattr(inner, "exog", None) if inner is not None else None
    if exog is None:
        return out
    try:
        exog = np.asarray(exog, dtype="float64")
    except (TypeError, ValueError):
        return out
    if exog.ndim != 2 or not exog.size:
        return out

    n_parameters = int(exog.shape[1])
    rank = getattr(inner, "rank", None)
    if rank is None:
        try:
            rank = int(np.linalg.matrix_rank(exog))
        except np.linalg.LinAlgError:
            return out
    rank = int(rank)
    residual_df = getattr(model, "df_resid", None)
    residual_df = int(residual_df) if residual_df is not None \
        else int(exog.shape[0] - rank)

    out["n_parameters"] = n_parameters
    out["design_rank"] = rank
    out["non_identifiable_directions"] = int(max(n_parameters - rank, 0))
    out["residual_degrees_of_freedom"] = residual_df
    out["design_identifiable"] = bool(rank >= n_parameters and residual_df > 0)
    out["wells_per_parameter"] = float(exog.shape[0] / n_parameters) \
        if n_parameters else float("nan")

    variance = exog.var(axis=0, ddof=1) if exog.shape[0] > 1 else \
        np.zeros(n_parameters)
    varying = variance > 0

    if out["design_identifiable"] and getattr(inner, "k_constant", 0):
        try:
            standard_errors = np.asarray(getattr(model, "bse"), dtype="float64")
            sigma_squared = float(getattr(model, "mse_resid"))
            if standard_errors.size == n_parameters and sigma_squared > 0:
                vif = (standard_errors ** 2) / sigma_squared \
                    * (exog.shape[0] - 1) * variance
                vif = vif[varying & np.isfinite(vif)]
                if vif.size:
                    out["max_vif"] = float(vif.max())
                    out["n_vif_above_10"] = int((vif > 10).sum())
        except (AttributeError, TypeError, ValueError):
            pass

    if 2 <= int(varying.sum()) <= _MAX_PREDICTORS_FOR_PAIRWISE:
        try:
            correlation = np.corrcoef(exog[:, varying], rowvar=False)
            upper = np.triu_indices_from(correlation, k=1)
            values = np.abs(correlation[upper])
            values = values[np.isfinite(values)]
            if values.size:
                out["n_collinear_pairs"] = int(
                    (values >= _COLLINEAR_THRESHOLD).sum())
                out["max_abs_predictor_correlation"] = float(values.max())
        except (np.linalg.LinAlgError, ValueError, MemoryError):
            pass
    return out


#: Above this many varying predictors the correlation matrix itself is the
#: expense (it is quadratic in the count), so the pair statistics are skipped
#: rather than allowed to dominate a trial. 4,000 predictors is a 128 MB matrix
#: and about a second; a real screen design is a quarter of that.
_MAX_PREDICTORS_FOR_PAIRWISE = 4000

#: Absolute Pearson correlation at which two predictors are counted as
#: collinear. Matches regression_diagnostics.collinear_guide_pairs' default, so
#: the count and the named table agree.
_COLLINEAR_THRESHOLD = 0.95


def guide_support_summary(results: pd.DataFrame, alpha: float = 0.05) -> dict:
    """How many of the hits rest on a single guide.

    :param results: guide-level result table used to compute gene support.

    A gene with one surviving guide has a gene-level p identical to that
    guide's, so it is not independent evidence -- and on this screen the top
    of the list is exactly that. A count of them belongs in the row.
    """
    out: dict[str, Any] = {}
    try:
        from .guide_concordance import guide_support
    except Exception:
        return out
    try:
        support = guide_support(results, alpha=alpha)
    except Exception:
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
    """How many things the trial called, raw and corrected.

    :param output: trial output carrying result and significant-hit frames.
    """
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


def qc_verdicts(row: Mapping[str, Any]) -> dict:
    """Summarize design and inference diagnostics for one sweep trial.

    Parameters
    ----------
    row : mapping
        Trial metrics produced by :func:`summarise_trial`. Design scoring uses
        well, parameter, rank, and condition-number fields. Inference scoring
        uses the result count and genomic-inflation estimate.

    Returns
    -------
    dict
        Available ``qc_design`` and ``qc_inference`` levels, plus
        ``qc_verdict`` containing the more severe available level. A diagnostic
        is omitted when its required metrics are unavailable.

    Notes
    -----
    The overall verdict is intentionally the worst available diagnostic rather
    than a pass rate. This prevents a severe identifiability or calibration
    problem from being hidden by otherwise acceptable checks.
    """
    from .regression_diagnostics import score_design, score_inference

    out: dict[str, Any] = {}
    scored: list = []
    design = {
        "wells": row.get("n_wells"),
        "parameters": row.get("n_parameters"),
        "design_rank": row.get("design_rank"),
        "wells_per_parameter": row.get("wells_per_parameter"),
        "condition_number": row.get("condition_number"),
        "identifiable": not (row.get("non_identifiable_directions") or 0),
    }
    if design["wells"]:
        try:
            verdict = score_design(design)
            out["qc_design"] = str(getattr(verdict, "level", ""))
            scored.append(verdict)
        except Exception:
            pass

    inference = {
        "genomic_inflation": row.get("genomic_inflation"),
        "tests": row.get("n_results"),
    }
    if inference["tests"] and inference["genomic_inflation"] is not None:
        try:
            verdict = score_inference(inference)
            out["qc_inference"] = str(getattr(verdict, "level", ""))
            scored.append(verdict)
        except Exception:                                    # noqa: BLE001
            pass

    if scored:
        from .regression_qc import worst_verdict

        try:
            out["qc_verdict"] = str(
                getattr(worst_verdict(scored), "level", "") or "")
        except Exception:                                    # noqa: BLE001
            pass
    return out


def summarise_trial(output: Mapping[str, Any],
                    settings: Mapping[str, Any]) -> dict:
    """Every metric, flat, for one trial's row.

    :param output: completed trial output with results, model, and fitted data.
    :param settings: trial settings used for thresholds and control recovery.

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
        lambda: design_diagnostics(model),
        lambda: fit_quality(model),
        lambda: residual_diagnostics(model),
        lambda: control_recovery(results, settings),
        lambda: calibration(results),
        lambda: guide_support_summary(results, alpha),
    ):
        try:
            row.update(block())
        except Exception:
            pass
    try:
        row.update(qc_verdicts(row))
    except Exception:
        pass
    return row


#: Columns that :func:`summarise_trial` may add to a sweep row.
#:
#: :func:`spacr.parameter_sweep.settings_for_trial` excludes these bookkeeping
#: fields when reconstructing a trial's settings. Keeping the complete metric
#: vocabulary here prevents diagnostic results such as ``r_squared`` or
#: ``qc_verdict`` from being passed back to the regression API as settings.
METRIC_COLUMNS: frozenset = frozenset({
    "qc_design", "qc_inference", "qc_verdict",
    "n_results", "n_significant", "n_primary", "n_below_alpha",
    "n_raw_below_alpha",
    "n_rows_fitted", "n_wells", "n_guides", "n_cells", "n_parameters",
    "design_rank", "non_identifiable_directions",
    "residual_degrees_of_freedom", "design_identifiable",
    "wells_per_parameter", "max_vif", "n_vif_above_10", "n_collinear_pairs",
    "max_abs_predictor_correlation",
    "r_squared", "r_squared_adj", "aic", "bic", "log_likelihood", "f_pvalue",
    "condition_number", "residual_se", "n_observations",
    "durbin_watson", "jarque_bera_p", "residual_skew", "residual_kurtosis",
    "breusch_pagan_p", "white_p", "residual_trend_slope",
    "n_ranked", "control_rank_separation",
    *(f"{role}_control_{suffix}"
      for role in ("positive", "negative")
      for suffix in ("found", "rank", "percentile", "p", "q",
                     "n_coefficients")),
    "genomic_inflation", "p_first_bin_excess", "n_tests",
    "n_genes_tested", "n_gene_hits", "n_single_guide_hits",
    "n_discordant_hits", "median_guides_per_hit",
})
