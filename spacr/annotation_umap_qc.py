"""Do annotated cells land where their guide's effect says they should?

Proposed 2026-08-21: "generate a umap, perform a hyperparamiter search find
stucture that separates the positive and negative controlls and see where the
annotated cells land, do they cluster with PC or NC ... this would be a
quality controll and a proof of sorts that the annotation is working".

IT IS A GOOD CHECK BECAUSE IT IS INDEPENDENT. The annotation is made from
sequencing fractions plus a phenotype call; this asks a different question of
a different space -- where does the cell sit among the controls, using every
measurement at once -- and the controls supply a frame of reference that owes
nothing to the annotation.

TURNING IT INTO A TEST RATHER THAN A PICTURE takes three things, and without
them it will agree with whatever it is shown:

  1. THE HYPERPARAMETERS MUST NOT BE CHOSEN ON THE CELLS BEING JUDGED.
     Searching until PC and NC separate is fitting the projection to the
     labels, and with enough measurements some setting always separates two
     groups. :func:`fit_on_controls` splits the CONTROL wells, tunes on one
     half, and reports the separation achieved on the other. A search that
     only separates the half it was tuned on has found nothing.

  2. THE READOUT MUST BE A NUMBER. "Do they cluster with PC" is answered by
     :func:`neighbour_purity`: the share of a cell's nearest CONTROL
     neighbours that are positive controls. Eyeballing a UMAP is not a
     measurement -- cluster sizes and the distances between them are
     artefacts of the projection, and only the neighbourhoods mean anything.

  3. THERE MUST BE A NULL. Cells land somewhere whatever the annotation
     says. :func:`effect_agreement` asks whether purity tracks the guides'
     EFFECT SIGNS, and compares that against the same statistic with the
     effects shuffled between guides.

THE TRAP THIS CANNOT ESCAPE, and it decides which methods may be judged this
way at all. A method that PICKS CELLS BY THE PHENOTYPE SCORE will produce
cells that sit near the positive controls whatever else is true -- that is
what it selected for. `rank` takes the top-scoring cells in the well, so its
annotated cells clustering with PC is a restatement of how it chose them and
proves nothing.

The check is informative exactly for methods whose cell choice did not use
the score: `sudoku` with the score out of the graph, `assigned`, and
`attributed` where the effect comes from the regression rather than the
score. :func:`circularity_warning` refuses to be quiet about it.
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
    """Why this check may not be informative for ``method``, or ``""``.

    A method that selected cells BY the score will place them near the
    positive controls by construction. Reporting that as validation would be
    reporting the selection rule back to itself.
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
                    neighbours: int = 15) -> Dict[str, object]:
    """Choose an embedding on half the controls, score it on the other half.

    :param features: ``(n_cells, n_features)`` for the CONTROL cells only.
    :param labels: :data:`POSITIVE` or :data:`NEGATIVE` per control cell.
    :param recipes: candidate UMAP parameter sets to try.
    :returns: the winning recipe, its held-out separation, and every trial.

    THE HELD-OUT NUMBER IS THE ONLY ONE WORTH READING. Tuning until two
    labelled groups separate always succeeds given enough parameters to try;
    the question is whether the separation survives on cells the search
    never saw. A large gap between the tuned and held-out scores means the
    search fitted the split rather than the biology, and the embedding
    should not then be used to judge anything.
    """
    from sklearn.model_selection import train_test_split

    values = np.asarray(features, dtype=float)
    marks = np.asarray([str(l) for l in labels])
    if values.shape[0] != marks.size or values.shape[0] < 8:
        return {"error": "too few control cells to split"}
    if len(set(marks.tolist())) < 2:
        return {"error": "the controls carry only one label"}

    fit_index, test_index = train_test_split(
        np.arange(values.shape[0]), test_size=float(holdout),
        random_state=int(seed), stratify=marks)

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
                "trials": trials}
    best = max(usable, key=lambda t: float(t["holdout_silhouette"]))
    gap = float(best["tuned_silhouette"]) - float(best["holdout_silhouette"])
    return {
        "recipe": best["recipe"],
        "tuned_silhouette": float(best["tuned_silhouette"]),
        "holdout_silhouette": float(best["holdout_silhouette"]),
        "overfit_gap": gap,
        "trustworthy": bool(float(best["holdout_silhouette"]) > 0.0
                            and gap < 0.25),
        "trials": trials,
    }


def neighbour_purity(embedding: np.ndarray,
                     control_labels: Sequence[Optional[str]], *,
                     k: int = 25) -> np.ndarray:
    """Share of each cell's nearest CONTROL neighbours that are positive.

    :param embedding: ``(n_cells, 2)`` with controls and annotated cells
        embedded together.
    :param control_labels: :data:`POSITIVE`, :data:`NEGATIVE`, or ``None``
        for a cell that is not a control.
    :returns: one value per cell in ``[0, 1]``; ``nan`` where no control
        neighbour was found.

    NEIGHBOURHOODS, NOT CLUSTERS, and not distances. A UMAP's cluster sizes
    and the gaps between them are artefacts of the layout; what it does
    preserve, and all it preserves, is who is near whom. A statistic built
    on anything else is reading the projection rather than the data.

    A control cell's own purity counts its neighbours and NOT ITSELF, so
    the controls can be scored on the same footing as everything else --
    which is what makes them usable as a reference for what a "pure" score
    even looks like here.
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
    """Average purity per annotated guide, with the count behind it.

    :param minimum_cells: guides with fewer annotated cells are left out
        rather than reported noisily. A guide with three cells has a purity
        that can only be 0, 1/3, 2/3 or 1.
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
    """Does purity track the guides' effect signs, beyond chance?

    THE CLAIM BEING TESTED, in the terms it was proposed: "the cells with a
    positive effect size cluster with the positive controll and the cells
    that do not have a positive coeffisient do not".

    :returns: the rank correlation between effect and purity, a permutation
        p-value, and the group means for guides with positive, negative and
        near-zero effects.

    THE NULL SHUFFLES THE EFFECTS BETWEEN GUIDES and keeps everything else,
    so it preserves how many guides there are, how many cells each has, and
    the purity distribution -- and destroys only the correspondence being
    claimed. A correlation that survives that is about the annotation.
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
