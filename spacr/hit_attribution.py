"""Resolve a well-level CRISPR hit back to *candidate* single cells.

Sequencing says which guide reads were present in a well. It does not say
which cell carried a guide, and a read fraction is not automatically an
infection fraction. This module keeps that distinction explicit:

* :func:`build_hit_cell_frame` makes the honest review queue: target-well
  cells ranked in the hit's phenotype direction.
* :func:`fit_hit_attribution` adds a cross-fitted two-component hierarchical
  mixture. Guide fraction changes a learned well-level prior; it is never
  imposed as the mean cell probability.
* :func:`write_hit_attribution` records versioned probabilities in their own
  tables. Hand annotations are untouched until an explicit promotion call.

The output is named ``hit_like_probability``, not guide identity. Only a
cell-resolved barcode or arrayed perturbation turns that inference into ground
truth.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "HitAttributionError",
    "InsufficientDesignError",
    "HitAttributionResult",
    "HitRunContext",
    "HitInvestigationResult",
    "build_hit_cell_frame",
    "fit_hit_attribution",
    "quantify_hit_enrichment",
    "crossfit_candidate_probabilities",
    "quantify_candidate_enrichment",
    "write_hit_attribution",
    "store_attribution",
    "promote_hit_calls",
    "promote_calls",
    "undo_hit_promotion",
    "revert_promotion",
]


class HitAttributionError(ValueError):
    """The requested cell attribution is ambiguous or not identifiable."""


class InsufficientDesignError(HitAttributionError):
    """There are too few independent wells/plates to cross-fit honestly."""


WELL_COLUMNS: Tuple[str, ...] = ("plateID", "rowID", "columnID")
OBJECT_COLUMNS: Tuple[str, ...] = (
    "prcfo", "plateID", "rowID", "columnID", "fieldID", "object_label")
_IDENTIFIER_HINTS = (
    "plate", "rowid", "columnid", "fieldid", "well", "object", "prcfo",
    "guide", "grna", "gene", "fraction", "annotation", "label", "class",
    "prediction", "probability", "posterior", "score",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise HitAttributionError(
            f"{label} is missing {missing}; it has {list(frame.columns)[:15]}")


def _well_columns(cells: pd.DataFrame, fractions: pd.DataFrame,
                  requested: Sequence[str]) -> List[str]:
    columns = [column for column in requested
               if column in cells.columns and column in fractions.columns]
    if not columns:
        raise HitAttributionError(
            "cells and guide fractions share no well key. Expected plateID, "
            "rowID and columnID (or pass well_columns explicitly).")
    if "rowID" in requested and "columnID" in requested:
        if "rowID" not in columns or "columnID" not in columns:
            raise HitAttributionError(
                "a well needs both rowID and columnID; refusing a partial key")
    return columns


def _object_columns(cells: pd.DataFrame,
                    requested: Sequence[str]) -> List[str]:
    if "prcfo" in cells.columns:
        columns = ["prcfo"]
    else:
        columns = [column for column in requested if column in cells.columns]
    if not columns:
        raise HitAttributionError(
            "cells have no stable object key (prcfo or plate/row/column/field/label)")
    if cells.duplicated(columns).any():
        examples = cells.loc[cells.duplicated(columns, keep=False), columns].head(4)
        raise HitAttributionError(
            "object keys are not unique; attribution would overwrite or "
            f"multiply cells. Examples: {examples.to_dict('records')}")
    return columns


def build_hit_cell_frame(
    cells: pd.DataFrame,
    guide_fractions: pd.DataFrame,
    *,
    target_guides: Sequence[str],
    score_column: str,
    direction: str = "positive",
    guide_column: str = "grna",
    fraction_column: str = "fraction",
    well_columns: Sequence[str] = WELL_COLUMNS,
    object_columns: Sequence[str] = OBJECT_COLUMNS,
) -> pd.DataFrame:
    """Join exact cells to well guide fractions and build a review ranking.

    The returned ``candidate_percentile`` ranks cells only within their well.
    It is deliberately not named a probability or infection call.
    """
    if not target_guides:
        raise HitAttributionError("choose at least one target guide")
    _require_columns(cells, [score_column], "cell frame")
    _require_columns(guide_fractions, [guide_column, fraction_column],
                     "guide-fraction frame")
    wells = _well_columns(cells, guide_fractions, well_columns)
    objects = _object_columns(cells, object_columns)

    fractions = guide_fractions.copy()
    numeric = pd.to_numeric(fractions[fraction_column], errors="coerce")
    if numeric.isna().any() or (~np.isfinite(numeric)).any():
        raise HitAttributionError("guide fractions must all be finite numbers")
    if ((numeric < 0) | (numeric > 1)).any():
        raise HitAttributionError("guide fractions must lie between 0 and 1")
    fractions[fraction_column] = numeric.astype(float)
    duplicate = fractions.duplicated(wells + [guide_column], keep=False)
    if duplicate.any():
        examples = fractions.loc[
            duplicate, wells + [guide_column, fraction_column]].head(4)
        raise HitAttributionError(
            "guide fractions contain more than one row per well/guide; "
            f"aggregate sequencing runs explicitly. Examples: {examples.to_dict('records')}")

    target = fractions[fractions[guide_column].astype(str).isin(
        {str(value) for value in target_guides})].copy()
    if target.empty:
        raise HitAttributionError(
            f"none of target guides {list(target_guides)} occur in guide fractions")
    target_by_well = (
        target.groupby(wells, dropna=False, as_index=False)[fraction_column]
        .sum().rename(columns={fraction_column: "target_guide_fraction"})
    )
    if (target_by_well["target_guide_fraction"] > 1 + 1e-9).any():
        raise HitAttributionError(
            "target guide fractions sum above 1 in at least one well")

    all_wells = fractions[wells].drop_duplicates()
    target_by_well = all_wells.merge(target_by_well, on=wells, how="left")
    target_by_well["target_guide_fraction"] = (
        target_by_well["target_guide_fraction"].fillna(0.0))
    target_wide = target.pivot(
        index=wells, columns=guide_column, values=fraction_column).fillna(0.0)
    guide_columns: Dict[str, str] = {}
    for guide in target_wide.columns:
        digest = hashlib.sha1(str(guide).encode("utf-8")).hexdigest()[:10]
        guide_columns[str(guide)] = f"target_guide_fraction__{digest}"
    target_wide = target_wide.rename(columns=guide_columns).reset_index()
    target_by_well = target_by_well.merge(
        target_wide, on=wells, how="left", validate="one_to_one")
    frame = cells.merge(target_by_well, on=wells, how="left",
                        validate="many_to_one")
    frame["target_guide_fraction"] = frame["target_guide_fraction"].fillna(0.0)
    frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
    if frame[score_column].isna().any():
        raise HitAttributionError(f"{score_column} contains missing/non-numeric values")

    direction_key = str(direction).strip().lower()
    if direction_key not in {"positive", "negative"}:
        raise HitAttributionError("direction must be 'positive' or 'negative'")
    ascending = direction_key == "negative"
    frame["candidate_rank"] = (
        frame.groupby(wells, dropna=False)[score_column]
        .rank(method="first", ascending=ascending).astype(int))
    frame["candidate_percentile"] = (
        frame.groupby(wells, dropna=False)[score_column]
        .rank(method="average", pct=True, ascending=not ascending))
    # High is always more hit-like. For a negative hit the descending=False
    # rank above gives the lowest raw score the highest percentile.
    frame["candidate_for_review"] = frame["target_guide_fraction"] > 0
    frame.attrs.update({
        "well_columns": wells,
        "object_columns": objects,
        "score_column": score_column,
        "direction": direction_key,
        "target_guides": [str(value) for value in target_guides],
        "target_guide_columns": guide_columns,
    })
    return frame


@dataclass
class _Mixture:
    median: np.ndarray
    scale: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    var0: np.ndarray
    var1: np.ndarray
    prior_intercept: float
    prior_slope: float
    iterations: int

    @staticmethod
    def _log_density(x: np.ndarray, mean: np.ndarray,
                     variance: np.ndarray) -> np.ndarray:
        return -0.5 * np.sum(
            np.log(2.0 * np.pi * variance) + ((x - mean) ** 2 / variance),
            axis=1,
        )

    def predict(self, values: np.ndarray, fractions: np.ndarray) -> np.ndarray:
        x = (values - self.median) / self.scale
        covariate = np.log(
            np.clip(fractions, 1e-3, 1 - 1e-3) /
            (1 - np.clip(fractions, 1e-3, 1 - 1e-3)))
        log_prior = self.prior_intercept + self.prior_slope * covariate
        log0 = self._log_density(x, self.mu0, self.var0) - np.logaddexp(0, log_prior)
        log1 = self._log_density(x, self.mu1, self.var1) - np.logaddexp(0, -log_prior)
        return 1.0 / (1.0 + np.exp(np.clip(log0 - log1, -40, 40)))


def _fit_fractional_logistic(covariate: np.ndarray, response: np.ndarray,
                             initial: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(covariate)), covariate])
    beta = np.asarray(initial, dtype=float).copy()
    for _ in range(40):
        linear = design @ beta
        probability = 1.0 / (1.0 + np.exp(np.clip(-linear, -40, 40)))
        weights = np.clip(probability * (1 - probability), 1e-6, None)
        hessian = design.T @ (weights[:, None] * design) + np.eye(2) * 1e-4
        gradient = design.T @ (response - probability) - beta * 1e-4
        step = np.linalg.solve(hessian, gradient)
        beta += step
        beta[1] = max(0.0, beta[1])
        if np.max(np.abs(step)) < 1e-6:
            break
    return beta


def _fit_mixture(values: np.ndarray, fractions: np.ndarray,
                 max_iter: int = 150) -> _Mixture:
    if not np.any(fractions <= 0) or not np.any(fractions > 0):
        raise HitAttributionError(
            "each training fold needs target-free and target-containing wells")
    median = np.nanmedian(values, axis=0)
    filled = np.where(np.isfinite(values), values, median)
    q25, q75 = np.nanpercentile(filled, [25, 75], axis=0)
    scale = np.where((q75 - q25) > 1e-8, q75 - q25,
                     np.nanstd(filled, axis=0))
    scale = np.where(scale > 1e-8, scale, 1.0)
    x = (filled - median) / scale

    control = fractions <= 0
    positive = fractions > 0
    delta = x[positive].mean(axis=0) - x[control].mean(axis=0)
    if not np.isfinite(delta).all() or np.linalg.norm(delta) < 1e-8:
        delta = np.zeros(x.shape[1]); delta[0] = 1.0
    projection = x @ (delta / max(np.linalg.norm(delta), 1e-8))
    projection = (projection - np.median(projection)) / (
        np.std(projection) + 1e-8)
    responsibility = 1.0 / (1.0 + np.exp(-projection))
    responsibility[control] *= 0.25

    covariate = np.log(
        np.clip(fractions, 1e-3, 1 - 1e-3) /
        (1 - np.clip(fractions, 1e-3, 1 - 1e-3)))
    beta = np.array([-1.0, 0.5])
    variance_floor = 1e-3
    previous = responsibility.copy()
    for iteration in range(1, max_iter + 1):
        w1 = np.clip(responsibility.sum(), 1e-6, None)
        w0 = np.clip((1 - responsibility).sum(), 1e-6, None)
        mu1 = (responsibility[:, None] * x).sum(axis=0) / w1
        mu0 = ((1 - responsibility)[:, None] * x).sum(axis=0) / w0
        var1 = (responsibility[:, None] * (x - mu1) ** 2).sum(axis=0) / w1
        var0 = ((1 - responsibility)[:, None] * (x - mu0) ** 2).sum(axis=0) / w0
        var1 = np.maximum(var1, variance_floor)
        var0 = np.maximum(var0, variance_floor)
        beta = _fit_fractional_logistic(covariate, responsibility, beta)
        log_prior = beta[0] + beta[1] * covariate
        log0 = _Mixture._log_density(x, mu0, var0) - np.logaddexp(0, log_prior)
        log1 = _Mixture._log_density(x, mu1, var1) - np.logaddexp(0, -log_prior)
        responsibility = 1.0 / (1.0 + np.exp(np.clip(log0 - log1, -40, 40)))
        if np.max(np.abs(responsibility - previous)) < 1e-5:
            break
        previous = responsibility.copy()

    return _Mixture(
        median=median, scale=scale, mu0=mu0, mu1=mu1,
        var0=var0, var1=var1, prior_intercept=float(beta[0]),
        prior_slope=float(beta[1]), iterations=iteration,
    )


@dataclass
class HitAttributionResult:
    """Cross-fitted hit-like probabilities and independent-unit evidence."""

    cells: pd.DataFrame
    wells: pd.DataFrame
    guide_evidence: pd.DataFrame
    threshold_sensitivity: pd.DataFrame
    validation: Dict[str, Any]
    feature_columns: List[str]
    well_columns: List[str]
    object_columns: List[str]
    target_gene: str
    target_guides: List[str]
    score_column: str
    direction: str
    threshold: float
    split_level: str
    random_seed: int
    source_regression_run: str = ""
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        effect = self.validation.get("prevalence_difference", float("nan"))
        low = self.validation.get("bootstrap_ci_low", float("nan"))
        high = self.validation.get("bootstrap_ci_high", float("nan"))
        p_value = self.validation.get("permutation_p_value", float("nan"))
        guide_refit = self.validation.get(
            "guide_fraction_refit_p_value", float("nan"))
        well_refit = self.validation.get(
            "well_label_refit_p_value", float("nan"))
        return "\n".join([
            f"Hit attribution: {self.target_gene}",
            "=" * (17 + len(self.target_gene)),
            f"cells: {len(self.cells):,}; wells: {len(self.wells):,}",
            f"cross-fit level: {self.split_level}",
            f"features: {', '.join(self.feature_columns)}",
            f"target-control prevalence difference: {effect:+.3f}",
            f"well bootstrap 95% CI: [{low:+.3f}, {high:+.3f}]",
            f"blocked permutation p: {p_value:.4g}",
            f"guide-fraction refit-null p: {guide_refit:.4g}",
            f"well-label refit-null p: {well_refit:.4g}",
            "Probabilities mean target-hit-like morphology, not observed guide identity.",
        ])


@dataclass(frozen=True)
class HitRunContext:
    """The exact regression result a cell investigation came from."""

    regression_results_folder: str
    regression_run_sha256: str
    gene: str
    phenotype: str
    effect: float
    guides: Tuple[str, ...] = ()
    fdr: float = float("nan")
    direction: str = "positive"


@dataclass
class HitInvestigationResult:
    """Portable result bundle used by the GUI and database persistence."""

    attribution_run_id: str
    context: HitRunContext
    cells: pd.DataFrame
    wells: pd.DataFrame
    enrichment: Dict[str, Any]
    feature_columns: List[str]
    split_level: str
    warnings: List[str] = field(default_factory=list)


def crossfit_candidate_probabilities(
    frame: pd.DataFrame,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    target_column: str = "target_well",
    prefer_plate: bool = True,
    random_seed: int = 0,
    n_splits: int = 5,
    threshold: float = 0.5,
) -> Tuple[pd.DataFrame, List[str], str, List[str]]:
    """Cross-fit a conservative morphology classifier from bag labels.

    This is the non-parametric alternative to :func:`fit_hit_attribution`'s
    hierarchical mixture. It predicts ``candidate_probability`` and never
    calls it infection probability. Model outputs, guide fractions, object
    identifiers and annotations are excluded from the default feature set.
    """
    _require_columns(frame, [target_column, "plateID", "rowID", "columnID"],
                     "candidate cell frame")
    well_columns = ["plateID", "rowID", "columnID"]
    well_table = frame[well_columns + [target_column]].drop_duplicates()
    if well_table.duplicated(well_columns).any():
        raise HitAttributionError("target-well status disagrees within a well")
    target_wells = int(well_table[target_column].astype(bool).sum())
    control_wells = int((~well_table[target_column].astype(bool)).sum())
    if target_wells < 4 or control_wells < 4:
        raise InsufficientDesignError(
            "at least four independent target and four independent control "
            f"wells are required (have {target_wells} and {control_wells})")

    features = list(feature_columns or _default_features(
        frame, score_column="__no_score_column__", include_score=False))
    _require_columns(frame, features, "candidate cell frame")
    forbidden = [column for column in features if any(
        hint in str(column).lower() for hint in _IDENTIFIER_HINTS)]
    if forbidden:
        raise HitAttributionError(
            f"candidate features leak identifiers/model outputs: {forbidden}")
    values = frame[features].apply(pd.to_numeric, errors="coerce")
    if values.isna().all(axis=0).any():
        bad = list(values.columns[values.isna().all(axis=0)])
        raise HitAttributionError(f"candidate features are entirely missing: {bad}")
    values = values.fillna(values.median())
    labels = frame[target_column].astype(bool).to_numpy()

    plate_count = frame["plateID"].nunique()
    if prefer_plate and plate_count >= 4:
        split_level = "plate"
        groups = frame["plateID"].astype(str)
    else:
        split_level = "well"
        groups = _group_series(frame, well_columns)
    group_count = groups.nunique()
    folds = min(max(2, int(n_splits)), group_count)
    if folds < 2:
        raise InsufficientDesignError("cross-fitting needs at least two groups")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    probability = np.full(len(frame), np.nan)
    assignment = np.full(len(frame), -1, dtype=int)
    warnings: List[str] = []
    splitter = GroupKFold(n_splits=folds)
    for fold, (train, test) in enumerate(
            splitter.split(values, labels, groups=groups)):
        if len(np.unique(labels[train])) < 2:
            raise InsufficientDesignError(
                f"{split_level} fold {fold} has only one bag class; add "
                "independent target/control groups or use well cross-fitting")
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=random_seed + fold)
        sample_weight = np.ones(len(train), dtype=float)
        if "target_guide_fraction" in frame.columns:
            fraction = frame.iloc[train]["target_guide_fraction"].to_numpy(float)
            # Fractions modulate evidence among positive bags but are not
            # treated as known cell-label proportions.
            sample_weight[labels[train]] = 0.5 + np.sqrt(
                np.clip(fraction[labels[train]], 0, 1))
        model.fit(values.iloc[train], labels[train], sample_weight=sample_weight)
        probability[test] = model.predict_proba(values.iloc[test])[:, 1]
        assignment[test] = fold
    if not np.isfinite(probability).all():
        raise HitAttributionError("cross-fitting left candidate cells unscored")
    scored = frame.copy()
    scored["candidate_probability"] = probability
    scored["candidate_uncertainty"] = 1 - np.abs(2 * probability - 1)
    scored["candidate_call"] = probability >= float(threshold)
    scored["attribution_fold"] = assignment
    return scored, features, split_level, warnings


def quantify_candidate_enrichment(
    scored: pd.DataFrame,
    *,
    target_column: str = "target_well",
    bootstrap_iterations: int = 1000,
    permutation_iterations: int = 1000,
    random_seed: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Quantify candidate prevalence at the well experimental unit."""
    well_columns = [column for column in WELL_COLUMNS if column in scored.columns]
    _require_columns(scored, well_columns + [target_column, "candidate_probability"],
                     "scored candidate frame")
    if "candidate_call" not in scored.columns:
        scored = scored.copy()
        scored["candidate_call"] = scored["candidate_probability"] >= 0.5
    wells = scored.groupby(well_columns, dropna=False).agg(
        target_well=(target_column, "first"),
        candidate_prevalence=("candidate_call", "mean"),
        mean_candidate_probability=("candidate_probability", "mean"),
        n_cells=("candidate_probability", "size"),
    ).reset_index()
    bridge = wells.rename(columns={
        "target_well": "_target_well",
        "candidate_prevalence": "hit_like_prevalence",
        "mean_candidate_probability": "mean_hit_like_probability",
    }).copy()
    bridge["target_guide_fraction"] = bridge["_target_well"].astype(float)
    summary = quantify_hit_enrichment(
        bridge, random_seed=random_seed, n_bootstrap=bootstrap_iterations,
        n_permutations=permutation_iterations)
    summary.update({
        "target_wells": summary.pop("n_target_wells"),
        "control_wells": summary.pop("n_control_wells"),
        "plate_blocked_permutation_p_value": summary.pop("permutation_p_value"),
    })
    return wells, summary


def _default_features(frame: pd.DataFrame, score_column: str,
                      include_score: bool) -> List[str]:
    numeric = list(frame.select_dtypes(include=[np.number]).columns)
    features = []
    for column in numeric:
        low = str(column).lower()
        if column == score_column:
            if include_score:
                features.append(column)
            continue
        if any(hint in low for hint in _IDENTIFIER_HINTS):
            continue
        features.append(column)
    return features


def _group_series(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    return frame[list(columns)].astype(str).agg("|".join, axis=1)


def _crossfit_mixture(values: np.ndarray, fractions: np.ndarray,
                      groups: pd.Series) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Score every cell with a mixture fitted without its group."""
    from sklearn.model_selection import GroupKFold
    unique_groups = pd.unique(groups)
    if len(unique_groups) < 3:
        raise HitAttributionError("cross-fitting needs at least three independent groups")
    n_splits = len(unique_groups) if len(unique_groups) <= 10 else 5
    splitter = GroupKFold(n_splits=n_splits)
    posterior = np.full(len(values), np.nan)
    folds = np.full(len(values), -1, dtype=int)
    iterations: List[int] = []
    for fold, (train, test) in enumerate(splitter.split(values, groups=groups)):
        model = _fit_mixture(values[train], fractions[train])
        posterior[test] = model.predict(values[test], fractions[test])
        folds[test] = fold
        iterations.append(model.iterations)
    if not np.isfinite(posterior).all():
        raise HitAttributionError("cross-fitting left cells without probabilities")
    return posterior, folds, iterations


def _permuted_well_fractions(frame: pd.DataFrame, fractions: np.ndarray,
                             well_columns: Sequence[str], rng,
                             *, binary_only: bool) -> np.ndarray:
    """Permute one fraction per well within plate and return cell-aligned values."""
    wells = frame[list(well_columns)].copy()
    wells["_fraction"] = fractions
    well_frame = wells.drop_duplicates(list(well_columns), keep="first")
    output = well_frame["_fraction"].to_numpy(float).copy()
    blocks = (well_frame["plateID"].astype(str).to_numpy()
              if "plateID" in well_frame else np.repeat("all", len(well_frame)))
    for block in pd.unique(blocks):
        indices = np.flatnonzero(blocks == block)
        source = output[indices].copy()
        if binary_only:
            positive = source[source > 0].copy()
            labels = rng.permutation(source > 0)
            rng.shuffle(positive)
            shuffled = np.zeros(len(source), dtype=float)
            shuffled[labels] = positive
            output[indices] = shuffled
        else:
            output[indices] = rng.permutation(source)
    mapping = dict(zip(_group_series(well_frame, well_columns), output))
    return _group_series(frame, well_columns).map(mapping).to_numpy(float)


def _refitted_permutation_p_values(
    frame: pd.DataFrame,
    values: np.ndarray,
    fractions: np.ndarray,
    groups: pd.Series,
    well_columns: Sequence[str],
    observed: float,
    *,
    iterations: int,
    random_seed: int,
    threshold: float,
) -> Dict[str, Any]:
    """Repeat cross-fitting under guide-fraction and well-label nulls."""
    count = max(0, int(iterations))
    if count == 0:
        return {
            "refitted_permutations": 0,
            "guide_fraction_refit_p_value": float("nan"),
            "well_label_refit_p_value": float("nan"),
        }
    rng = np.random.default_rng(random_seed)
    well_keys = _group_series(frame, well_columns)
    nulls = {"guide": [], "well": []}
    for label, binary_only in (("guide", False), ("well", True)):
        for _ in range(count):
            permuted = _permuted_well_fractions(
                frame, fractions, well_columns, rng,
                binary_only=binary_only)
            try:
                posterior, _folds, _fit_iterations = _crossfit_mixture(
                    values, permuted, groups)
            except HitAttributionError:
                # A sparse permutation can leave one training fold with only
                # one bag class. It is an unidentified null draw, not evidence;
                # omit it and report the completed count explicitly.
                continue
            temporary = pd.DataFrame({
                "_well": well_keys,
                "_positive": permuted > 0,
                "_call": posterior >= threshold,
            }).groupby("_well", sort=False).agg(
                positive=("_positive", "first"), prevalence=("_call", "mean"))
            positive = temporary["positive"].to_numpy(bool)
            if not positive.any() or positive.all():
                continue
            nulls[label].append(float(
                temporary.loc[positive, "prevalence"].mean() -
                temporary.loc[~positive, "prevalence"].mean()))
    output: Dict[str, Any] = {"refitted_permutations": count}
    for label, values_null in nulls.items():
        array = np.asarray(values_null, dtype=float)
        output[f"{label}_refitted_permutations_completed"] = int(len(array))
        output[f"{label}_fraction_refit_p_value" if label == "guide" else
               "well_label_refit_p_value"] = (
            float((1 + np.sum(np.abs(array) >= abs(observed))) /
                  (len(array) + 1)) if len(array) else float("nan"))
    return output


def fit_hit_attribution(
    frame: pd.DataFrame,
    *,
    target_gene: str,
    feature_columns: Optional[Sequence[str]] = None,
    include_original_score: bool = False,
    threshold: float = 0.8,
    split_by: str = "auto",
    random_seed: int = 0,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    n_pipeline_permutations: int = 0,
    source_regression_run: str = "",
) -> HitAttributionResult:
    """Estimate cross-fitted target-hit-like probabilities.

    ``frame`` must be the output of :func:`build_hit_cell_frame`. Every cell
    is predicted by a mixture fitted without its well, or without its plate
    when at least three plates make that split identifiable.
    """
    well_columns = list(frame.attrs.get("well_columns", WELL_COLUMNS))
    well_columns = [column for column in well_columns if column in frame.columns]
    object_columns = list(frame.attrs.get("object_columns", ("prcfo",)))
    object_columns = [column for column in object_columns if column in frame.columns]
    score_column = str(frame.attrs.get("score_column", "prediction"))
    direction = str(frame.attrs.get("direction", "positive"))
    target_guides = list(frame.attrs.get("target_guides", ()))
    target_guide_columns = dict(frame.attrs.get("target_guide_columns", {}))
    _require_columns(frame, well_columns + object_columns +
                     ["target_guide_fraction", score_column], "hit cell frame")
    if not (0 < float(threshold) < 1):
        raise HitAttributionError("threshold must lie strictly between 0 and 1")

    features = list(feature_columns or _default_features(
        frame, score_column, include_original_score))
    if include_original_score and score_column not in features:
        features.append(score_column)
    if not features:
        raise HitAttributionError(
            "no independent numeric morphology features remain; choose "
            "feature_columns or explicitly include the original score")
    _require_columns(frame, features, "hit cell frame")
    leaked = [column for column in features
              if column != score_column and any(
                  hint in column.lower() for hint in _IDENTIFIER_HINTS)]
    if leaked:
        raise HitAttributionError(
            f"feature columns contain identifiers/outcomes that leak the bag label: {leaked}")

    fractions = frame["target_guide_fraction"].to_numpy(dtype=float)
    if not np.any(fractions == 0) or not np.any(fractions > 0):
        raise HitAttributionError(
            "attribution needs both target-free and target-containing wells")
    values = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    plate_count = frame["plateID"].nunique() if "plateID" in frame.columns else 0
    requested = str(split_by).strip().lower()
    if requested == "auto":
        split_level = "plate" if plate_count >= 3 else "well"
    elif requested in {"plate", "well"}:
        split_level = requested
    else:
        raise HitAttributionError("split_by must be auto, plate or well")
    if split_level == "plate":
        if plate_count < 3:
            raise HitAttributionError("plate cross-fitting needs at least three plates")
        groups = frame["plateID"].astype(str)
    else:
        groups = _group_series(frame, well_columns)
    posterior, folds, iterations = _crossfit_mixture(values, fractions, groups)

    cells = frame.copy()
    cells["hit_like_probability"] = posterior
    cells["hit_like_uncertainty"] = 1.0 - np.abs(2.0 * posterior - 1.0)
    cells["hit_like_call"] = posterior >= float(threshold)
    cells["attribution_fold"] = folds
    cells["target_gene"] = str(target_gene)
    well_agg = {
        "target_guide_fraction": "first",
        "hit_like_probability": "mean",
        "hit_like_call": "mean",
        score_column: "mean",
    }
    for column in target_guide_columns.values():
        if column in cells.columns:
            well_agg[column] = "first"
    wells = cells.groupby(well_columns, dropna=False).agg(well_agg).reset_index()
    wells = wells.rename(columns={
        "hit_like_probability": "mean_hit_like_probability",
        "hit_like_call": "hit_like_prevalence",
        score_column: "mean_original_score",
    })
    sizes = cells.groupby(well_columns, dropna=False).size().rename("n_cells")
    wells = wells.merge(sizes.reset_index(), on=well_columns, validate="one_to_one")

    validation = quantify_hit_enrichment(
        wells, random_seed=random_seed, n_bootstrap=n_bootstrap,
        n_permutations=n_permutations)
    validation.update(_refitted_permutation_p_values(
        frame, values, fractions, groups, well_columns,
        float(validation["prevalence_difference"]),
        iterations=n_pipeline_permutations,
        random_seed=random_seed + 9173, threshold=float(threshold)))
    guide_rows = []
    for guide in target_guides:
        column = target_guide_columns.get(str(guide))
        if not column or column not in wells.columns:
            continue
        guide_fraction = wells[column].to_numpy(float)
        probability = wells["mean_hit_like_probability"].to_numpy(float)
        present = guide_fraction > 0
        correlation = pd.Series(guide_fraction).corr(
            pd.Series(probability), method="spearman")
        difference = (float(probability[present].mean() - probability[~present].mean())
                      if present.any() and (~present).any() else float("nan"))
        guide_rows.append({
            "guide": str(guide), "target_gene": str(target_gene),
            "wells_with_guide": int(present.sum()),
            "dose_response_spearman": float(correlation),
            "mean_probability_difference": difference,
        })
    guide_evidence = pd.DataFrame(guide_rows)
    sensitivity_rows = []
    target_well = cells[well_columns + ["target_guide_fraction"]].drop_duplicates(
        well_columns).set_index(well_columns)["target_guide_fraction"] > 0
    for candidate_threshold in sorted({0.5, 0.6, 0.7, 0.8, 0.9,
                                       float(threshold)}):
        temporary = cells[well_columns].copy()
        temporary["prevalence"] = (
            cells["hit_like_probability"].to_numpy() >= candidate_threshold)
        per_well = temporary.groupby(well_columns, dropna=False)[
            "prevalence"].mean()
        aligned_target = target_well.reindex(per_well.index).to_numpy(bool)
        sensitivity_rows.append({
            "threshold": candidate_threshold,
            "target_mean_prevalence": float(per_well[aligned_target].mean()),
            "control_mean_prevalence": float(per_well[~aligned_target].mean()),
            "prevalence_difference": float(
                per_well[aligned_target].mean() - per_well[~aligned_target].mean()),
            "target_wells": int(aligned_target.sum()),
            "control_wells": int((~aligned_target).sum()),
        })
    threshold_sensitivity = pd.DataFrame(sensitivity_rows)

    warnings = [
        "The original CV score was included in attribution; use the default "
        "score-excluded model as the less circular morphology check."
    ] if include_original_score else []
    validation["mean_em_iterations"] = float(np.mean(iterations))
    return HitAttributionResult(
        cells=cells, wells=wells, guide_evidence=guide_evidence,
        threshold_sensitivity=threshold_sensitivity,
        validation=validation, feature_columns=features,
        well_columns=well_columns, object_columns=object_columns,
        target_gene=str(target_gene), target_guides=target_guides,
        score_column=score_column, direction=direction,
        threshold=float(threshold), split_level=split_level,
        random_seed=int(random_seed), source_regression_run=str(source_regression_run),
        warnings=warnings,
    )


def quantify_hit_enrichment(wells: pd.DataFrame, *, random_seed: int = 0,
                            n_bootstrap: int = 1000,
                            n_permutations: int = 1000) -> Dict[str, Any]:
    """Per-well enrichment, well bootstrap CI, and plate-blocked null."""
    _require_columns(wells, ["target_guide_fraction", "hit_like_prevalence"],
                     "well summary")
    positive = wells["target_guide_fraction"].to_numpy(float) > 0
    if positive.sum() < 2 or (~positive).sum() < 2:
        raise HitAttributionError(
            "enrichment needs at least two target and two target-free wells")
    prevalence = wells["hit_like_prevalence"].to_numpy(float)
    observed = float(prevalence[positive].mean() - prevalence[~positive].mean())
    rng = np.random.default_rng(random_seed)
    boot = np.empty(max(1, int(n_bootstrap)), dtype=float)
    target_values = prevalence[positive]
    control_values = prevalence[~positive]
    for index in range(len(boot)):
        boot[index] = (
            rng.choice(target_values, len(target_values), replace=True).mean() -
            rng.choice(control_values, len(control_values), replace=True).mean())

    blocks = (wells["plateID"].astype(str).to_numpy()
              if "plateID" in wells.columns else np.repeat("all", len(wells)))
    null = np.empty(max(1, int(n_permutations)), dtype=float)
    for index in range(len(null)):
        shuffled = positive.copy()
        for block in pd.unique(blocks):
            mask = blocks == block
            shuffled[mask] = rng.permutation(shuffled[mask])
        null[index] = (
            prevalence[shuffled].mean() - prevalence[~shuffled].mean())
    p_value = float((1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1))
    dose = wells[["target_guide_fraction", "mean_hit_like_probability"]].corr(
        method="spearman").iloc[0, 1]
    return {
        "n_target_wells": int(positive.sum()),
        "n_control_wells": int((~positive).sum()),
        "prevalence_difference": observed,
        "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
        "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
        "permutation_p_value": p_value,
        "dose_response_spearman": float(dose),
        "independent_unit": "well",
    }


def _object_key(row: pd.Series, columns: Sequence[str]) -> str:
    payload = {column: row[column] for column in columns}
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def write_hit_attribution(db_path: str, result: HitAttributionResult,
                          *, run_id: Optional[str] = None) -> str:
    """Persist a versioned attribution without touching annotation columns."""
    path = os.path.abspath(os.path.expanduser(db_path))
    if not os.path.isfile(path):
        raise HitAttributionError(f"no database at {path}")
    run = str(run_id or uuid.uuid4())
    created = _now()
    manifest = {
        "target_gene": result.target_gene,
        "target_guides": result.target_guides,
        "feature_columns": result.feature_columns,
        "well_columns": result.well_columns,
        "object_columns": result.object_columns,
        "score_column": result.score_column,
        "direction": result.direction,
        "threshold": result.threshold,
        "split_level": result.split_level,
        "random_seed": result.random_seed,
        "source_regression_run": result.source_regression_run,
        "validation": result.validation,
        "warnings": result.warnings,
    }
    with sqlite3.connect(path, timeout=30) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS hit_attribution_runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                target_gene TEXT NOT NULL,
                manifest_json TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS object_hit_attribution (
                run_id TEXT NOT NULL,
                object_key TEXT NOT NULL,
                probability REAL NOT NULL,
                uncertainty REAL NOT NULL,
                threshold REAL NOT NULL,
                hit_like_call INTEGER NOT NULL,
                attribution_fold INTEGER NOT NULL,
                target_guide_fraction REAL NOT NULL,
                PRIMARY KEY (run_id, object_key),
                FOREIGN KEY (run_id) REFERENCES hit_attribution_runs(run_id)
                    ON DELETE CASCADE
            )
        """)
        connection.execute(
            "INSERT INTO hit_attribution_runs VALUES (?, ?, ?, ?)",
            (run, created, result.target_gene,
             json.dumps(manifest, sort_keys=True, default=str)))
        rows = []
        for _, row in result.cells.iterrows():
            rows.append((
                run, _object_key(row, result.object_columns),
                float(row["hit_like_probability"]),
                float(row["hit_like_uncertainty"]), result.threshold,
                int(bool(row["hit_like_call"])), int(row["attribution_fold"]),
                float(row["target_guide_fraction"]),
            ))
        connection.executemany(
            "INSERT INTO object_hit_attribution VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows)
    return run


_SQL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def promote_hit_calls(db_path: str, result: HitAttributionResult, *,
                      run_id: str, annotation_column: str,
                      positive_value: Any = 1) -> str:
    """Explicitly promote positive calls to a fresh, reversible annotation."""
    column = str(annotation_column)
    if not _SQL_NAME.fullmatch(column):
        raise HitAttributionError(
            "annotation column must contain only letters, numbers and underscores")
    if result.object_columns != ["prcfo"]:
        raise HitAttributionError(
            "promotion currently requires prcfo as the exact png_list object key")
    promotion_id = str(uuid.uuid4())
    with sqlite3.connect(db_path, timeout=30) as connection:
        columns = {row[1] for row in connection.execute(
            'PRAGMA table_info("png_list")')}
        if "prcfo" not in columns:
            raise HitAttributionError("png_list has no prcfo object key")
        if column in columns:
            raise HitAttributionError(
                f"annotation column {column!r} already exists; choose a fresh "
                "name so hand annotations cannot be overwritten")
        connection.execute(f'ALTER TABLE "png_list" ADD COLUMN "{column}"')
        connection.execute("""
            CREATE TABLE IF NOT EXISTS hit_attribution_promotions (
                promotion_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                annotation_column TEXT NOT NULL,
                prcfo TEXT NOT NULL,
                previous_value TEXT,
                promoted_value TEXT,
                created_at TEXT NOT NULL,
                undone_at TEXT,
                PRIMARY KEY (promotion_id, prcfo)
            )
        """)
        selected = result.cells[result.cells["hit_like_call"]]
        for value in selected["prcfo"].astype(str):
            connection.execute(
                f'UPDATE "png_list" SET "{column}"=? WHERE prcfo=?',
                (positive_value, value))
            connection.execute(
                "INSERT INTO hit_attribution_promotions VALUES "
                "(?, ?, ?, ?, NULL, ?, ?, NULL)",
                (promotion_id, run_id, column, value,
                 json.dumps(positive_value, default=str), _now()))
    return promotion_id


def undo_hit_promotion(db_path: str, promotion_id: str) -> int:
    """Clear values written by one promotion; preserve the audit record."""
    with sqlite3.connect(db_path, timeout=30) as connection:
        rows = connection.execute(
            "SELECT annotation_column, prcfo FROM hit_attribution_promotions "
            "WHERE promotion_id=? AND undone_at IS NULL", (promotion_id,)
        ).fetchall()
        if not rows:
            return 0
        columns = {row[0] for row in rows}
        if len(columns) != 1:
            raise HitAttributionError("promotion audit contains multiple columns")
        column = next(iter(columns))
        if not _SQL_NAME.fullmatch(column):
            raise HitAttributionError("stored annotation column is unsafe")
        connection.executemany(
            f'UPDATE "png_list" SET "{column}"=NULL WHERE prcfo=?',
            [(row[1],) for row in rows])
        connection.execute(
            "UPDATE hit_attribution_promotions SET undone_at=? "
            "WHERE promotion_id=? AND undone_at IS NULL", (_now(), promotion_id))
        return len(rows)


def store_attribution(db_path: str, result: HitInvestigationResult) -> int:
    """Store the GUI investigation bundle under its immutable run context."""
    path = os.path.abspath(os.path.expanduser(db_path))
    if not os.path.isfile(path):
        raise HitAttributionError(f"no database at {path}")
    _require_columns(result.cells, ["prcfo", "candidate_probability"],
                     "investigation cells")
    if result.cells["prcfo"].duplicated().any():
        raise HitAttributionError("investigation contains duplicate prcfo object keys")
    context = {
        "regression_results_folder": result.context.regression_results_folder,
        "regression_run_sha256": result.context.regression_run_sha256,
        "gene": result.context.gene,
        "phenotype": result.context.phenotype,
        "effect": result.context.effect,
        "guides": list(result.context.guides),
        "fdr": result.context.fdr,
        "direction": result.context.direction,
        "feature_columns": result.feature_columns,
        "split_level": result.split_level,
        "warnings": result.warnings,
        "enrichment": result.enrichment,
    }
    with sqlite3.connect(path, timeout=30) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS hit_investigation_runs (
                attribution_run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                context_json TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS hit_investigation_cells (
                attribution_run_id TEXT NOT NULL,
                prcfo TEXT NOT NULL,
                candidate_probability REAL NOT NULL,
                candidate_uncertainty REAL NOT NULL,
                candidate_call INTEGER NOT NULL,
                attribution_fold INTEGER NOT NULL,
                PRIMARY KEY (attribution_run_id, prcfo),
                FOREIGN KEY (attribution_run_id)
                    REFERENCES hit_investigation_runs(attribution_run_id)
                    ON DELETE CASCADE
            )
        """)
        connection.execute(
            "INSERT INTO hit_investigation_runs VALUES (?, ?, ?)",
            (result.attribution_run_id, _now(),
             json.dumps(context, sort_keys=True, default=str)))
        rows = []
        for row in result.cells.itertuples(index=False):
            probability = float(getattr(row, "candidate_probability"))
            uncertainty = float(getattr(
                row, "candidate_uncertainty", 1 - abs(2 * probability - 1)))
            call = bool(getattr(row, "candidate_call", probability >= 0.5))
            fold = int(getattr(row, "attribution_fold", -1))
            rows.append((result.attribution_run_id, str(row.prcfo),
                         probability, uncertainty, int(call), fold))
        connection.executemany(
            "INSERT INTO hit_investigation_cells VALUES (?, ?, ?, ?, ?, ?)",
            rows)
    return len(rows)


def promote_calls(db_path: str, attribution_run_id: str,
                  annotation_column: str) -> str:
    """Promote stored calls while recording every previous annotation value."""
    column = str(annotation_column)
    if not _SQL_NAME.fullmatch(column):
        raise HitAttributionError(
            "annotation column must contain only letters, numbers and underscores")
    promotion_id = str(uuid.uuid4())
    with sqlite3.connect(db_path, timeout=30) as connection:
        table_columns = {row[1] for row in connection.execute(
            'PRAGMA table_info("png_list")')}
        if "prcfo" not in table_columns:
            raise HitAttributionError("png_list has no prcfo object key")
        if column not in table_columns:
            connection.execute(f'ALTER TABLE "png_list" ADD COLUMN "{column}"')
        calls = connection.execute(
            "SELECT prcfo, candidate_call FROM hit_investigation_cells "
            "WHERE attribution_run_id=?", (attribution_run_id,)).fetchall()
        if not calls:
            raise HitAttributionError(
                f"no stored cells for attribution run {attribution_run_id!r}")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS hit_promotion_audit (
                promotion_id TEXT NOT NULL,
                attribution_run_id TEXT NOT NULL,
                annotation_column TEXT NOT NULL,
                prcfo TEXT NOT NULL,
                previous_json TEXT,
                promoted_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                undone_at TEXT,
                PRIMARY KEY (promotion_id, prcfo)
            )
        """)
        for prcfo, call in calls:
            current = connection.execute(
                f'SELECT "{column}" FROM "png_list" WHERE prcfo=?',
                (prcfo,)).fetchone()
            if current is None:
                raise HitAttributionError(
                    f"attributed object {prcfo!r} is absent from png_list")
            previous = current[0]
            connection.execute(
                f'UPDATE "png_list" SET "{column}"=? WHERE prcfo=?',
                (int(call), prcfo))
            connection.execute(
                "INSERT INTO hit_promotion_audit VALUES "
                "(?, ?, ?, ?, ?, ?, ?, NULL)",
                (promotion_id, attribution_run_id, column, prcfo,
                 json.dumps(previous, default=str), json.dumps(int(call)), _now()))
    return promotion_id


def revert_promotion(db_path: str, promotion_id: str) -> int:
    """Restore exactly the values replaced by :func:`promote_calls`.

    Reverting something that was never promoted is a NO-OP returning ``0``,
    not an error -- including on a database where nothing has ever been
    promoted at all. The audit table is created by :func:`promote_calls`, so
    until one has run it does not exist, and reaching for it raised a raw
    ``sqlite3.OperationalError: no such table`` out of a spaCR API. An undo
    that crashes when there is nothing to undo makes a stray click look like
    a corrupt database.
    """
    with sqlite3.connect(db_path, timeout=30) as connection:
        audited = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='hit_promotion_audit'").fetchone()
        if not audited:
            return 0
        rows = connection.execute(
            "SELECT annotation_column, prcfo, previous_json "
            "FROM hit_promotion_audit WHERE promotion_id=? AND undone_at IS NULL",
            (promotion_id,)).fetchall()
        if not rows:
            return 0
        columns = {row[0] for row in rows}
        if len(columns) != 1:
            raise HitAttributionError("promotion audit contains multiple columns")
        column = next(iter(columns))
        if not _SQL_NAME.fullmatch(column):
            raise HitAttributionError("stored annotation column is unsafe")
        for _, prcfo, previous_json in rows:
            previous = json.loads(previous_json)
            connection.execute(
                f'UPDATE "png_list" SET "{column}"=? WHERE prcfo=?',
                (previous, prcfo))
        connection.execute(
            "UPDATE hit_promotion_audit SET undone_at=? "
            "WHERE promotion_id=? AND undone_at IS NULL", (_now(), promotion_id))
        return len(rows)
