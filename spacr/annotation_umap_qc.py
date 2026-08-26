"""Evaluate guide annotations against positive and negative control cells.

The workflow tunes an embedding on one control subset, evaluates separation
on held-out controls, and summarizes each annotated cell by the fraction of
nearby controls that are positive. Guide-level purity can then be compared
with independently estimated effect signs by permutation testing.

This check is circular when the annotation method selected cells using the
same phenotype score, or when that score is included among the embedding
features. :func:`circularity_warning` reports those cases so apparent control
agreement is not presented as independent validation.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "POSITIVE",
    "NEGATIVE",
    "fit_on_controls",
    "neighbour_purity",
    "purity_by_guide",
    "effect_agreement",
    "circularity_warning",
]

POSITIVE = "PC"
NEGATIVE = "NC"

#: Methods whose choice of cells uses the phenotype score directly, so this
#: check cannot say anything about them. Names match the pickers in
#: `cell_montage.PICKING_MODES` plus the sudoku entries.
SCORE_PICKED = frozenset({"rank", "top_by_score"})


def circularity_warning(method: str, *,
                        score_in_features: bool = False) -> str:
    """Return a warning when control agreement is not independent.

    An empty string means that neither the method name nor the supplied
    feature flag identifies a known circularity.
    """
    if str(method) in SCORE_PICKED:
        return (f"{method} chooses cells by the phenotype score, so its "
                f"cells sitting near the positive controls restates how it "
                f"chose them and validates nothing")
    if score_in_features:
        return (f"{method} was run with the classification score among its "
                f"features, so part of any agreement here is that score "
                f"appearing on both sides")
    return ""


def fit_on_controls(features: np.ndarray,
                    labels: Sequence[str], *,
                    recipes: Sequence[Mapping[str, object]],
                    seed: int = 0,
                    holdout: float = 0.5,
                    neighbours: int = 15,
                    groups: Optional[Sequence[object]] = None,
                    group_by: str = "well") -> Dict[str, object]:
    """Select an embedding recipe using held-out control separation.

    Parameters
    ----------
    features : numpy.ndarray
        Control-cell feature matrix with shape ``(n_cells, n_features)``.
    labels : sequence of str
        :data:`POSITIVE` or :data:`NEGATIVE` for each control cell.
    recipes : sequence of mappings
        Candidate UMAP parameter dictionaries.
    seed : int, default=0
        Random seed for splitting and embedding.
    holdout : float, default=0.5
        Fraction of controls reserved for evaluation.
    neighbours : int, default=15
        Neighbour count passed to the embedding score.
    groups : sequence, optional
        One group identity per control cell, usually the well. Control cells
        from one well share illumination, seeding and fixation, so a split
        that puts siblings on both sides reports a held-out silhouette the
        embedding did not earn. When groups are given no group appears on
        both sides; when they are omitted the split is per object and the
        result says so in ``split_level``.
    group_by : str, default='well'
        The level ``groups`` names, one of the shared split ladder.

    Returns
    -------
    dict
        Winning recipe, tuned and held-out silhouette scores, overfit gap,
        trustworthiness flag, the split level used, and all attempted
        trials. Error dictionaries are returned when the controls cannot be
        split or scored.
    """
    from .classifier_evaluation import grouped_split

    values = np.asarray(features, dtype=float)
    marks = np.asarray([str(l) for l in labels])
    if values.shape[0] != marks.size or values.shape[0] < 8:
        return {"error": "too few control cells to split"}
    if len(set(marks.tolist())) < 2:
        return {"error": "the controls carry only one label"}

    # ONE SPLITTER FOR THE WHOLE PACKAGE. The grouped splitter refuses a
    # design it cannot hold apart rather than falling back to a random one,
    # which is the difference between an honest error and an optimistic
    # score.
    level = str(group_by) if groups is not None else "cell"
    identities = (np.asarray(list(groups), dtype=object) if groups is not None
                  else np.arange(values.shape[0], dtype=object))
    if identities.size != marks.size:
        return {"error": "one group is needed per control cell"}
    try:
        fit_index, test_index, report = grouped_split(
            identities, marks, float(holdout), int(seed), group_by=level)
    except ValueError as exc:
        return {"error": str(exc), "split_level": level}

    from .hyperparam import _default_umap_embed, _umap_scores

    trials: List[Dict[str, object]] = []
    for number, recipe in enumerate(recipes):
        params = dict(recipe)
        try:
            fitted = _default_umap_embed(values[fit_index], params,
                                         int(seed) + number)
            tuned = _umap_scores(values[fit_index], fitted,
                                 marks[fit_index], int(neighbours))
            held = _default_umap_embed(values[test_index], params,
                                       int(seed) + number)
            outside = _umap_scores(values[test_index], held,
                                   marks[test_index], int(neighbours))
        except Exception as exc:                              # noqa: BLE001
            trials.append({"recipe": params,
                           "error": f"{type(exc).__name__}: {exc}"})
            continue
        trials.append({
            "recipe": params,
            "tuned_silhouette": float(tuned.get("silhouette", float("nan"))),
            "holdout_silhouette": float(
                outside.get("silhouette", float("nan"))),
        })

    usable = [t for t in trials if np.isfinite(
        float(t.get("holdout_silhouette", float("nan"))))]
    if not usable:
        return {"error": "no recipe produced a scorable embedding",
                "split_level": report.group_by, "trials": trials}
    best = max(usable, key=lambda t: float(t["holdout_silhouette"]))
    gap = float(best["tuned_silhouette"]) - float(best["holdout_silhouette"])
    return {
        "recipe": best["recipe"],
        "tuned_silhouette": float(best["tuned_silhouette"]),
        "holdout_silhouette": float(best["holdout_silhouette"]),
        "overfit_gap": gap,
        "trustworthy": bool(float(best["holdout_silhouette"]) > 0.0
                            and gap < 0.25),
        "split_level": report.group_by,
        "split_rule": report.rule,
        "trials": trials,
    }


def neighbour_purity(embedding: np.ndarray,
                     control_labels: Sequence[Optional[str]], *,
                     k: int = 25) -> np.ndarray:
    """Compute the positive-control share among each cell's neighbours.

    ``embedding`` contains controls and annotated cells together.
    ``control_labels`` uses :data:`POSITIVE`, :data:`NEGATIVE`, or ``None``
    for non-control cells. Each control cell is excluded from its own
    neighbourhood. The result contains one value in ``[0, 1]`` per cell, or
    ``nan`` when no eligible control neighbour is available.
    """
    from sklearn.neighbors import NearestNeighbors

    points = np.asarray(embedding, dtype=float)
    marks = np.asarray([None if l is None else str(l)
                        for l in control_labels], dtype=object)
    control_rows = np.flatnonzero(np.array([m is not None for m in marks]))
    if control_rows.size == 0 or points.shape[0] != marks.size:
        return np.full(points.shape[0], np.nan)

    wanted = int(max(1, min(int(k), control_rows.size - 1)))
    finder = NearestNeighbors(n_neighbors=min(wanted + 1,
                                              control_rows.size)).fit(
        points[control_rows])
    distances, indices = finder.kneighbors(points)

    out = np.full(points.shape[0], np.nan)
    for row in range(points.shape[0]):
        neighbours = control_rows[indices[row]]
        # Drop self, which is only present for a control cell.
        neighbours = neighbours[neighbours != row][:wanted]
        if neighbours.size == 0:
            continue
        out[row] = float(np.mean(marks[neighbours] == POSITIVE))
    return out


def purity_by_guide(purity: np.ndarray,
                    guides: Sequence[str], *,
                    abstain: str = "Non_annotated",
                    minimum_cells: int = 10) -> Dict[str, Dict[str, float]]:
    """Summarize neighbour purity for each sufficiently represented guide.

    The abstention label is excluded. Guides with fewer than
    ``minimum_cells`` finite values are omitted; retained rows report mean
    purity, standard deviation, and cell count.
    """
    values = np.asarray(purity, dtype=float)
    names = np.asarray([str(g) for g in guides])
    out: Dict[str, Dict[str, float]] = {}
    for guide in sorted(set(names.tolist())):
        if guide == str(abstain):
            continue
        rows = np.flatnonzero(names == guide)
        here = values[rows]
        here = here[np.isfinite(here)]
        if here.size < int(minimum_cells):
            continue
        out[guide] = {
            "purity": float(np.mean(here)),
            "spread": float(np.std(here)),
            "cells": float(here.size),
        }
    return out


def effect_agreement(purity: Mapping[str, Mapping[str, float]],
                     effects: Mapping[str, float], *,
                     permutations: int = 999,
                     seed: int = 0) -> Dict[str, object]:
    """Test whether guide effects agree with positive-control proximity.

    The observed statistic is Spearman correlation between guide effect and
    mean neighbour purity. The permutation null shuffles effects between
    guides while preserving the purity values and guide counts. The result
    includes the correlation, two-sided permutation p-value, group means for
    positive and negative effects, and a boolean separation call.
    """
    from scipy.stats import spearmanr

    shared = [g for g in purity if g in effects
              and np.isfinite(float(effects[g]))]
    if len(shared) < 4:
        return {"error": f"only {len(shared)} guide(s) have both a purity "
                         f"and an effect; a correlation needs more"}

    values = np.array([float(purity[g]["purity"]) for g in shared])
    weights = np.array([float(effects[g]) for g in shared])
    observed = float(spearmanr(weights, values).statistic)

    rng = np.random.default_rng(int(seed))
    null = np.empty(int(permutations), dtype=float)
    for index in range(int(permutations)):
        null[index] = float(
            spearmanr(rng.permutation(weights), values).statistic)
    p = float((1 + np.sum(np.abs(null) >= abs(observed)))
              / (1 + int(permutations)))

    positive = [values[i] for i, g in enumerate(shared) if weights[i] > 0]
    negative = [values[i] for i, g in enumerate(shared) if weights[i] < 0]
    return {
        "guides": len(shared),
        "correlation": observed,
        "p_value": p,
        "positive_effect_purity": float(np.mean(positive)) if positive
        else float("nan"),
        "negative_effect_purity": float(np.mean(negative)) if negative
        else float("nan"),
        "separated": bool(p < 0.05 and observed > 0),
        "permutations": int(permutations),
    }
