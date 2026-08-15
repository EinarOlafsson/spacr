"""File-driven, provenance-bound application around hit attribution.

The statistical engine lives in :mod:`spacr.hit_attribution`.  This module is
the application seam: it names exact files, hashes the selected regression
run, joins predictions to measured objects, writes review artifacts, and
registers settings for GUI and headless use.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from .hit_attribution import (
    HitAttributionError, HitAttributionResult, build_hit_cell_frame,
    fit_hit_attribution, write_hit_attribution,
)

APP_KEY = "investigate_hit"
__all__ = [
    "hit_investigation_default_settings", "investigate_hit",
    "control_fitted_embedding", "review_gallery_manifest",
    "review_gallery_key", "evaluate_blinded_reviews",
]


def _hash_run(folder: str) -> str:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise HitAttributionError(f"no regression results folder at {root}")
    files = sorted(path for path in root.iterdir()
                   if path.is_file() and path.suffix.lower() in {".csv", ".json"})
    if not files:
        raise HitAttributionError(f"{root} has no CSV/JSON regression outputs")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _read_cells(db_path: str, predictions_file: str,
                score_column: str, path_column: str) -> pd.DataFrame:
    from .io import _read_and_join_tables
    cells = _read_and_join_tables(db_path)
    if cells.index.name == "prcfo" and "prcfo" not in cells:
        cells = cells.reset_index()
    predictions = pd.read_csv(predictions_file)
    if score_column not in predictions:
        raise HitAttributionError(
            f"prediction file has no {score_column!r} column")
    if score_column in cells:
        return cells
    if "prcfo" in predictions and "prcfo" in cells:
        if predictions["prcfo"].duplicated().any():
            raise HitAttributionError("prediction file repeats prcfo")
        return cells.merge(
            predictions[["prcfo", score_column]], on="prcfo", how="inner",
            validate="one_to_one")
    if path_column not in predictions:
        raise HitAttributionError(
            "predictions need prcfo or the configured crop-path column")
    with sqlite3.connect(db_path, timeout=30) as connection:
        png = pd.read_sql_query(
            'SELECT "prcfo", "png_path" FROM "png_list"', connection)
    png["_crop"] = png["png_path"].astype(str).map(os.path.basename)
    predictions["_crop"] = predictions[path_column].astype(str).map(os.path.basename)
    if png["_crop"].duplicated().any() or predictions["_crop"].duplicated().any():
        raise HitAttributionError(
            "crop basenames are not unique; export prcfo with predictions")
    scores = predictions.merge(png, on="_crop", how="inner",
                               validate="one_to_one")
    return cells.merge(
        scores[["prcfo", "png_path", score_column]], on="prcfo", how="inner",
        validate="one_to_one")


def _read_fractions(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"plateID", "rowID", "columnID"}
    if not required.issubset(frame) and "prc" in frame:
        pieces = frame["prc"].astype(str).str.rsplit("_", n=2, expand=True)
        if pieces.shape[1] == 3:
            frame[["plateID", "rowID", "columnID"]] = pieces.to_numpy()
    return frame


def control_fitted_embedding(result: HitAttributionResult) -> pd.DataFrame:
    """Fit PCA on target-free morphology and transform all cells."""
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    features = list(result.feature_columns)
    if len(features) < 2:
        raise HitAttributionError("embedding requires at least two allowed features")
    cells = result.cells
    control = cells["target_guide_fraction"].to_numpy(float) <= 0
    if control.sum() < 3:
        raise HitAttributionError("embedding requires target-free training cells")
    values = cells[features].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median").fit(values.loc[control])
    controls = imputer.transform(values.loc[control])
    scaler = StandardScaler().fit(controls)
    pca = PCA(n_components=2, random_state=result.random_seed).fit(
        scaler.transform(controls))
    coordinates = pca.transform(scaler.transform(imputer.transform(values)))
    keys = list(dict.fromkeys(result.object_columns + result.well_columns))
    output = cells[[key for key in keys if key in cells]].copy()
    output["embedding_1"] = coordinates[:, 0]
    output["embedding_2"] = coordinates[:, 1]
    output["hit_like_probability"] = cells["hit_like_probability"].to_numpy()
    output["target_guide_fraction"] = cells["target_guide_fraction"].to_numpy()
    output["embedding_contract"] = (
        "PCA fit on target-free-well morphology only; target cells transformed")
    return output


def _review_gallery_selection(result: HitAttributionResult,
                              per_stratum: int) -> pd.DataFrame:
    cells = result.cells.copy()
    if "png_path" not in cells:
        return pd.DataFrame()
    cells["_distance"] = np.abs(
        cells["hit_like_probability"] - float(result.threshold))
    target = cells["target_guide_fraction"] > 0
    strata = (
        ("high", cells.loc[target].nlargest(per_stratum, "hit_like_probability")),
        ("borderline", cells.loc[target].nsmallest(per_stratum, "_distance")),
        ("low", cells.loc[target].nsmallest(per_stratum, "hit_like_probability")),
        ("control_false_looking", cells.loc[~target].nlargest(
            per_stratum, "hit_like_probability")),
    )
    frames = []
    for name, selection in strata:
        selection = selection.copy()
        selection["review_stratum"] = name
        frames.append(selection)
    output = pd.concat(frames, ignore_index=True).drop_duplicates(
        result.object_columns, keep="first")
    object_key = output[result.object_columns].astype(str).agg("|".join, axis=1)
    output["review_id"] = object_key.map(
        lambda value: hashlib.sha256(
            f"{result.random_seed}|{value}".encode("utf-8")).hexdigest()[:16])
    columns = list(dict.fromkeys([
        "review_id", *result.object_columns, *result.well_columns, "png_path",
        "review_stratum", "hit_like_probability", "hit_like_uncertainty",
        "target_guide_fraction", "attribution_fold"]))
    return output[[column for column in columns if column in output]]


def review_gallery_manifest(result: HitAttributionResult,
                            per_stratum: int = 24) -> pd.DataFrame:
    """Return a shuffled reviewer sheet that does not disclose group or score."""
    selection = _review_gallery_selection(result, per_stratum)
    if selection.empty:
        return selection
    blinded = selection[["review_id", "png_path"]].copy()
    blinded["reviewer_id"] = ""
    blinded["reviewer_label"] = ""
    blinded["reviewer_confidence"] = ""
    return blinded.sample(frac=1, random_state=result.random_seed).reset_index(drop=True)


def review_gallery_key(result: HitAttributionResult,
                       per_stratum: int = 24) -> pd.DataFrame:
    """Return the analyst-only key for a blinded gallery manifest."""
    return _review_gallery_selection(result, per_stratum)


def evaluate_blinded_reviews(reviews: pd.DataFrame,
                             key: pd.DataFrame) -> Dict[str, Any]:
    """Compare one or more blinded binary reviewers with held-back scores."""
    required_review = {"review_id", "reviewer_id", "reviewer_label"}
    required_key = {"review_id", "hit_like_probability"}
    if not required_review.issubset(reviews):
        raise HitAttributionError(
            f"review sheet needs {sorted(required_review)}")
    if not required_key.issubset(key):
        raise HitAttributionError(f"review key needs {sorted(required_key)}")
    if key["review_id"].duplicated().any():
        raise HitAttributionError("review key repeats review_id")
    frame = reviews.merge(
        key[["review_id", "hit_like_probability"]], on="review_id",
        how="inner", validate="many_to_one")
    frame["reviewer_label"] = pd.to_numeric(
        frame["reviewer_label"], errors="coerce")
    frame = frame[frame["reviewer_label"].isin([0, 1])].copy()
    if frame.empty:
        raise HitAttributionError("review sheet contains no binary 0/1 labels")
    from sklearn.metrics import (cohen_kappa_score, precision_score,
                                 recall_score, roc_auc_score)
    consensus = frame.groupby("review_id")["reviewer_label"].mean()
    probabilities = key.set_index("review_id").loc[consensus.index,
                                                    "hit_like_probability"]
    binary = consensus >= 0.5
    predicted = probabilities >= 0.5
    metrics: Dict[str, Any] = {
        "n_reviewed_objects": int(len(consensus)),
        "n_reviewers": int(frame["reviewer_id"].astype(str).nunique()),
        "precision": float(precision_score(binary, predicted, zero_division=0)),
        "recall": float(recall_score(binary, predicted, zero_division=0)),
        # Squared calibration error supports a soft human consensus (for
        # example, one of two blinded reviewers calling the object positive).
        "brier_score": float(np.mean((consensus - probabilities) ** 2)),
        "roc_auc": (float(roc_auc_score(binary, probabilities))
                    if binary.nunique() == 2 else float("nan")),
    }
    reviewers = sorted(frame["reviewer_id"].astype(str).unique())
    kappas = []
    for left_index, left in enumerate(reviewers):
        left_values = frame[frame["reviewer_id"].astype(str) == left][
            ["review_id", "reviewer_label"]]
        for right in reviewers[left_index + 1:]:
            right_values = frame[frame["reviewer_id"].astype(str) == right][
                ["review_id", "reviewer_label"]]
            paired = left_values.merge(right_values, on="review_id")
            if len(paired) >= 2:
                kappas.append(cohen_kappa_score(
                    paired["reviewer_label_x"], paired["reviewer_label_y"]))
    metrics["mean_pairwise_cohen_kappa"] = (
        float(np.nanmean(kappas)) if kappas else float("nan"))
    return metrics


def hit_investigation_default_settings(settings=None) -> Dict[str, Any]:
    """Return settings for the Investigate Hit application."""
    configured = dict(settings or {})
    for key in ("db_path", "predictions_file", "guide_fractions_file",
                "results_folder", "target_gene", "score_column",
                "hit_phenotype", "dst"):
        configured.setdefault(key, "")
    configured.setdefault("path_column", "path")
    configured.setdefault("target_guides", [])
    configured.setdefault("hit_effect", 0.0)
    configured.setdefault("hit_fdr", 1.0)
    configured.setdefault("hit_guide_agreement", float("nan"))
    configured.setdefault("hit_n_guides", 0)
    configured.setdefault("hit_well_support", 0)
    configured.setdefault("hit_direction", "positive")
    configured.setdefault("hit_feature_columns", [])
    configured.setdefault("hit_include_original_score", False)
    configured.setdefault("hit_probability_threshold", 0.8)
    configured.setdefault("hit_split_by", "auto")
    configured.setdefault("hit_random_seed", 0)
    configured.setdefault("hit_bootstrap", 5000)
    configured.setdefault("hit_permutations", 10000)
    configured.setdefault("hit_pipeline_permutations", 100)
    configured.setdefault("hit_gallery_per_stratum", 24)
    configured.setdefault("hit_store_database", True)
    configured.setdefault("verbose", True)
    return configured


def investigate_hit(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Run one exact regression hit through the cell-attribution workflow."""
    configured = hit_investigation_default_settings(settings)
    for key in ("db_path", "predictions_file", "guide_fractions_file"):
        path = os.path.abspath(os.path.expanduser(str(configured[key])))
        if not os.path.isfile(path):
            raise HitAttributionError(f"{key} does not identify a file: {path}")
        configured[key] = path
    target_guides = list(configured["target_guides"])
    if not configured["target_gene"] or not target_guides:
        raise HitAttributionError("target_gene and target_guides are required")
    run_hash = _hash_run(str(configured["results_folder"]))
    source = f"{Path(configured['results_folder']).resolve()}#sha256={run_hash}"
    cells = _read_cells(
        configured["db_path"], configured["predictions_file"],
        str(configured["score_column"]), str(configured["path_column"]))
    frame = build_hit_cell_frame(
        cells, _read_fractions(configured["guide_fractions_file"]),
        target_guides=target_guides,
        score_column=str(configured["score_column"]),
        direction=str(configured["hit_direction"]))
    result = fit_hit_attribution(
        frame, target_gene=str(configured["target_gene"]),
        feature_columns=configured["hit_feature_columns"] or None,
        include_original_score=bool(configured["hit_include_original_score"]),
        threshold=float(configured["hit_probability_threshold"]),
        split_by=str(configured["hit_split_by"]),
        random_seed=int(configured["hit_random_seed"]),
        n_bootstrap=int(configured["hit_bootstrap"]),
        n_permutations=int(configured["hit_permutations"]),
        n_pipeline_permutations=int(configured["hit_pipeline_permutations"]),
        source_regression_run=source)
    embedding = control_fitted_embedding(result)
    gallery_size = int(configured["hit_gallery_per_stratum"])
    gallery = review_gallery_manifest(result, gallery_size)
    gallery_key = review_gallery_key(result, gallery_size)
    root = Path(configured["dst"] or configured["results_folder"]) / \
        "hit_investigation" / str(configured["target_gene"])
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "cells": root / "candidate_cells.csv",
        "wells": root / "well_level_evidence.csv",
        "guides": root / "independent_guide_evidence.csv",
        "thresholds": root / "threshold_sensitivity.csv",
        "embedding": root / "control_fitted_embedding.csv",
        "gallery": root / "blinded_review_gallery_manifest.csv",
        "gallery_key": root / "blinded_review_gallery_key.csv",
        "manifest": root / "manifest.json",
    }
    result.cells.to_csv(paths["cells"], index=False)
    result.wells.to_csv(paths["wells"], index=False)
    result.guide_evidence.to_csv(paths["guides"], index=False)
    result.threshold_sensitivity.to_csv(paths["thresholds"], index=False)
    embedding.to_csv(paths["embedding"], index=False)
    gallery.to_csv(paths["gallery"], index=False)
    gallery_key.to_csv(paths["gallery_key"], index=False)
    attribution_run_id = ""
    if configured["hit_store_database"]:
        attribution_run_id = write_hit_attribution(configured["db_path"], result)
    manifest = {
        "target_gene": result.target_gene, "target_guides": result.target_guides,
        "phenotype": configured["hit_phenotype"],
        "regression_effect": float(configured["hit_effect"]),
        "regression_fdr": float(configured["hit_fdr"]),
        "regression_guide_agreement": float(configured["hit_guide_agreement"]),
        "regression_n_guides": int(configured["hit_n_guides"]),
        "regression_well_support": int(configured["hit_well_support"]),
        "source_regression_run": source,
        "attribution_run_id": attribution_run_id,
        "probability_semantics": (
            "cross-fitted target-hit-like phenotype under a well-level mixture; "
            "not observed cell guide identity"),
        "validation": result.validation, "feature_columns": result.feature_columns,
        "original_score_in_model": bool(configured["hit_include_original_score"]),
        "split_level": result.split_level, "random_seed": result.random_seed,
        "warnings": result.warnings,
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8")
    if configured["verbose"]:
        print(result.summary())
    return {"result": result, "embedding": embedding, "gallery": gallery,
            "gallery_key": gallery_key,
            "paths": {key: str(value) for key, value in paths.items()},
            "attribution_run_id": attribution_run_id}


def register_settings(replace: bool = False) -> bool:
    """Register this app's settings through spaCR's extension seam."""
    from .settings import (has_registered_defaults, register_defaults,
                           tooltips as shared_tooltips)
    if has_registered_defaults(APP_KEY) and not replace:
        return False
    types = {
        "db_path": str, "predictions_file": str,
        "guide_fractions_file": str, "results_folder": str,
        "target_gene": str, "target_guides": list, "score_column": str,
        "hit_phenotype": str, "hit_effect": (int, float),
        "hit_fdr": (int, float),
        "hit_guide_agreement": (int, float), "hit_n_guides": int,
        "hit_well_support": int,
        "path_column": str, "hit_direction": str,
        "hit_feature_columns": list, "hit_include_original_score": bool,
        "hit_probability_threshold": (int, float), "hit_split_by": str,
        "hit_random_seed": int, "hit_bootstrap": int,
        "hit_permutations": int, "hit_pipeline_permutations": int,
        "hit_gallery_per_stratum": int,
        "hit_store_database": bool, "dst": str,
    }
    tips = {
        "db_path": "(str) - Exact measurements.db whose objects the prediction file scored. Choosing another database is refused when crop identities do not match, preventing cross-experiment explanations.",
        "predictions_file": "(str) - Existing per-object CV prediction CSV joined to measured objects. The module never reruns the vision model or substitutes scores from another run.",
        "guide_fractions_file": "(str) - Sequencing-derived table with one fraction per well and guide. Duplicate well-guide rows are refused because they would reweight target evidence ambiguously.",
        "results_folder": "(str) - Exact regression output folder that produced the selected hit. Its CSV and JSON bytes are hashed so later edits create different provenance.",
        "target_gene": "(str) - Gene identifier carried from the selected Hit List row. It labels the attribution run and output folder without being inferred again.",
        "target_guides": "(list) - Exact guides supporting the selected gene in the source result. Each remains separate evidence so discordance cannot be averaged away.",
        "hit_phenotype": "(str) - Human-readable phenotype copied from the source regression. It records what direction and score the selected effect actually described.",
        "hit_effect": "(float) - Effect estimate copied from the exact source result for provenance and display. It does not get re-estimated from candidate cells.",
        "hit_fdr": "(float) - Adjusted P value copied from the selected regression hit. It records source evidence and never becomes a cell-level confidence value.",
        "path_column": "(str) - Column in the prediction CSV containing crop paths used for the one-to-one object join. Change it only when the exporter used another name. Default path.",
        "hit_direction": "(str) - Positive ranks larger phenotype scores first; negative ranks smaller scores first. It must match the sign and interpretation of the selected regression effect.",
        "hit_feature_columns": "(list) - Independent measured morphology features used by the weak-supervision model. Leave blank for guarded numeric selection excluding identifiers, labels, fractions and scores.",
        "hit_include_original_score": "(bool) - Add the original CV phenotype score to model features. Default False preserves an independent check; enabling it makes validation partly circular and records a warning.",
        "hit_probability_threshold": "(float) - Probability boundary for review calls and optional promotion. It is never a genotype threshold; changing it alters calls but not fitted probabilities. Default 0.8.",
        "hit_split_by": "(str) - Cross-fitting unit: auto prefers held-out plates when the design supports them and otherwise holds wells out. Cell-level splitting is never allowed.",
        "hit_random_seed": "(int) - Seed for grouped cross-fitting, bootstrap intervals and plate-aware permutations. Keep it fixed for reproducible candidate probabilities and uncertainty. Default 0.",
        "hit_bootstrap": "(int) - Number of well-level bootstrap resamples used for the prevalence-difference confidence interval. More resamples improve tail stability but increase runtime. Default 5000.",
        "hit_permutations": "(int) - Number of plate-aware well-label permutations used for the enrichment P value. More permutations improve resolution without changing fitted candidate probabilities. Default 10000.",
        "hit_pipeline_permutations": "(int) - Guide-fraction and well-label null iterations that repeat grouped cross-fitting inside every permutation. Increase it for stronger pipeline-null resolution at substantial runtime cost. Default 100.",
        "hit_gallery_per_stratum": "(int) - Maximum blinded-review examples drawn from high, borderline, low and control-false-looking strata. Raising it expands manual review without changing attribution. Default 24.",
        "hit_store_database": "(bool) - Store this attribution as a new versioned database run. Disabling it writes portable files only; neither choice overwrites hand annotations. Default True.",
        "dst": "(str) - Folder receiving versioned tables, manifests and figures. Leaving it blank uses a module-specific folder beside the primary input, keeping different analyses separated.",
    }
    # Shared settings keep the canonical cross-module help. Import order must
    # not decide whether this app or Barcode QC owns ``dst``/``db_path``.
    tips = {key: value for key, value in tips.items()
            if key not in shared_tooltips}
    register_defaults(
        APP_KEY, hit_investigation_default_settings, replace=replace,
        expected_types=types, tooltips=tips,
        description=(
            "Trace an exact regression hit to candidate cells with grouped "
            "cross-fitting, well-level enrichment, a control-fitted embedding "
            "and reversible versioned storage."))
    return True


register_settings()
