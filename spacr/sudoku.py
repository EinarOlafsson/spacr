"""Infer cell-level guide assignments across wells.

Pooled screens provide guide counts for each well, not a guide identity for
each cell. This module combines those count constraints with similarity in
measurement space. It selects high-confidence anchor cells, propagates their
labels over a k-nearest-neighbour graph, and projects the result onto the
per-well guide fractions reported by sequencing.

Assignments can therefore borrow evidence for the same perturbation across
wells while retaining an explicit abstention state. The propagated mass,
competing-label evidence, and total anchor reach remain available separately
so ambiguous or unsupported calls can be inspected instead of forced.

Notes
-----
The propagation and class-mass normalization follow the label-propagation
framework described by Zhu and Ghahramani (2003) and Zhou et al. (2004).
The implementation has no trained graph-model weights; each result can be
traced to its anchors and the sequencing constraint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ABSTAIN",
    "SudokuResult",
    "anchors_for",
    "similarity_graph",
    "propagate",
    "constrain_to_fractions",
    "sudoku",
    "sudoku_all",
]

#: What a cell is annotated when the method declines to name a guide.
#: The same spelling `guide_attribution` uses, so a consumer that already
#: handles one handles the other.
ABSTAIN = "Non_annotated"

#: Below this share of the anchor mass a cell is "unlike everything" -- the
#: fourth quadrant of the two-score table, which a single ratio hides. It is
#: a share of the median reach, not an absolute, because the mass a graph
#: delivers depends on its size and its edge weights.
DEFAULT_REACH_FLOOR = 0.15

#: The coin flip, as a number. The sequential form stops when the best
#: remaining cell cannot beat this posterior. 0.55 is `0.5 + delta` at
#: `delta = 0.05`, matching `guide_attribution.DEFAULT_THRESHOLD`.
DEFAULT_DECISION = 0.55


@dataclass(frozen=True)
class SudokuResult:
    """What sudoku concluded, with both scores kept apart.

    :ivar guides: one name per cell, in the order the cells came, or
        :data:`ABSTAIN`.
    :ivar affirm: ``(n_cells, n_guides)`` -- how much each cell looks like
        each guide's anchors.
    :ivar eliminate: ``(n_cells, n_guides)`` -- how much it looks like
        anything ELSE. Kept separate on purpose: "probably not this guide"
        and "probably this guide" are different claims, and a cell can be
        low on both, which is the cell worth looking at. A single ratio
        maps that onto the same value as a cell that is high on both.
    :ivar reach: ``(n_cells,)`` -- total anchor mass that arrived. Low reach
        IS the fourth quadrant: unlike every anchor, of any guide.
    :ivar posterior: ``(n_cells, n_guides)`` after the well constraint.
    :ivar names: the guide names, in the column order of the matrices.
    :ivar report: counts and settings, for the record.
    """

    guides: Tuple[str, ...]
    affirm: np.ndarray
    eliminate: np.ndarray
    reach: np.ndarray
    posterior: np.ndarray
    names: Tuple[str, ...]
    report: Dict[str, object] = field(default_factory=dict)

    @property
    def abstained(self) -> np.ndarray:
        """Boolean mask of the cells no guide was named for."""
        return np.array([g == ABSTAIN for g in self.guides], dtype=bool)

    def called(self) -> int:
        """How many cells were annotated."""
        return int((~self.abstained).sum())


# ---------------------------------------------------------------------------
# 1. anchors
# ---------------------------------------------------------------------------

def anchors_for(guide: str,
                wells: Sequence[str],
                fractions: Mapping[str, Mapping[str, float]],
                scores: np.ndarray, *,
                quantile: float = 0.9,
                min_fraction: float = 0.5,
                max_per_well: int = 50) -> np.ndarray:
    """Indices of the cells taken as near-certain examples of ``guide``.

    :param wells: one well label per cell.
    :param fractions: ``{well: {guide: fraction}}``.
    :param scores: the classification score per cell.
    :param quantile: within an anchor well, the score quantile above which a
        cell is taken.
    :param min_fraction: only wells where the guide is at least this share
        are used. THE POINT OF THE WHOLE THING: in a well where the guide is
        70% of the reads, a top-scoring cell is very probably that guide. In
        a well where it is 5%, it is probably not, and an anchor set built
        from such wells teaches the graph the wrong shape.
    :param max_per_well: a cap, so one enormous well cannot define the
        guide's appearance on its own.
    :returns: cell indices, possibly empty.

    THE ANCHORS ARE WHERE CIRCULARITY ENTERS, and it is entered knowingly.
    They are chosen BY SCORE, so a profile built from the score alone would
    say "a `g` cell has a high score" -- true by construction and worthless.
    :func:`sudoku` therefore excludes the score from the graph features by
    default. The score picks the anchors; the morphology decides everyone
    else.
    """
    labels = np.asarray(wells)
    values = np.asarray(scores, dtype=float)
    picked: list = []
    for well in sorted(set(labels.tolist())):
        share = float(fractions.get(well, {}).get(guide, 0.0))
        if share < float(min_fraction):
            continue
        here = np.flatnonzero(labels == well)
        if here.size == 0:
            continue
        mine = values[here]
        usable = here[np.isfinite(mine)]
        if usable.size == 0:
            continue
        cut = float(np.quantile(values[usable], float(quantile)))
        chosen = usable[values[usable] >= cut]
        if chosen.size > int(max_per_well):
            order = np.argsort(-values[chosen])
            chosen = chosen[order[:int(max_per_well)]]
        picked.extend(chosen.tolist())
    return np.array(sorted(set(picked)), dtype=int)


# ---------------------------------------------------------------------------
# 2. the graph
# ---------------------------------------------------------------------------

def similarity_graph(features: np.ndarray, *,
                     neighbours: int = 15,
                     mutual: bool = True):
    """A symmetric k-nearest-neighbour affinity over cells.

    :param features: ``(n_cells, n_features)``, standardised here.
    :param neighbours: k.
    :param mutual: keep an edge only where BOTH cells list the other. A
        plain kNN graph gives an outlier k edges it did not earn -- it has
        to have SOME nearest neighbours -- and label mass then flows into
        cells that resemble nothing. Mutual kNN lets an outlier end up with
        no edges at all, which is the honest answer and is what
        :attr:`SudokuResult.reach` reports.
    :returns: a sparse CSR affinity with a zero diagonal.

    Weights are the heat kernel with a SELF-TUNING scale: each cell's own
    distance to its kth neighbour. One global sigma assumes the cloud has
    one density, and a screen's cells do not -- a dense healthy population
    beside a sparse dying one gets either no edges in the sparse region or
    an over-connected dense one.
    """
    from scipy import sparse
    from sklearn.neighbors import NearestNeighbors

    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        return sparse.csr_matrix((0, 0))
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    centre = values.mean(axis=0)
    spread = values.std(axis=0)
    spread[spread <= 0] = 1.0
    values = (values - centre) / spread

    n = values.shape[0]
    k = int(max(1, min(int(neighbours), n - 1))) if n > 1 else 0
    if k == 0:
        return sparse.csr_matrix((n, n))

    finder = NearestNeighbors(n_neighbors=k + 1).fit(values)
    distances, indices = finder.kneighbors(values)
    # Column 0 is the cell itself.
    distances, indices = distances[:, 1:], indices[:, 1:]
    sigma = distances[:, -1].copy()
    sigma[sigma <= 0] = float(np.median(sigma[sigma > 0])) if np.any(
        sigma > 0) else 1.0

    rows = np.repeat(np.arange(n), k)
    cols = indices.reshape(-1)
    scale = sigma[rows] * sigma[cols]
    scale[scale <= 0] = 1.0
    weights = np.exp(-(distances.reshape(-1) ** 2) / scale)
    graph = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n))
    if mutual:
        # Both directions present: the elementwise minimum of W and W.T is
        # zero wherever one direction is missing.
        graph = graph.minimum(graph.T)
    else:
        graph = graph.maximum(graph.T)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph.tocsr()


# ---------------------------------------------------------------------------
# 3. propagation
# ---------------------------------------------------------------------------

def propagate(graph, seeds: np.ndarray, *,
              alpha: float = 0.9,
              iterations: int = 100,
              tolerance: float = 1e-6) -> np.ndarray:
    """Spread the anchor mass along the graph.

    :param graph: the affinity from :func:`similarity_graph`.
    :param seeds: ``(n_cells, n_guides)``, non-zero on anchor rows only.
    :param alpha: how much a cell listens to its neighbours against its own
        seed. 0.9 is the usual value; at 1.0 the seeds are forgotten and the
        answer is the graph's leading eigenvector, which says nothing about
        guides.
    :returns: ``(n_cells, n_guides)`` propagated mass, NOT normalised.

    Zhou et al.'s local-and-global-consistency iteration, ``F <- a S F +
    (1-a) Y`` with ``S = D^-1/2 W D^-1/2``. It converges to
    ``(1-a)(I - aS)^-1 Y`` and is run as an iteration rather than a solve
    because the matrix is sparse and the fixed point is reached in tens of
    steps.

    THE MASS IS RETURNED UNNORMALISED and that is deliberate. Row-normalising
    here would turn a cell that received almost nothing -- a cell unlike
    every anchor of every guide -- into a confident-looking row summing to
    one. That cell is the interesting one, and :func:`sudoku` finds it by
    looking at the row's TOTAL before anything is normalised.
    """
    from scipy import sparse

    values = np.asarray(seeds, dtype=float)
    n = values.shape[0]
    if n == 0 or graph.shape[0] != n:
        return np.zeros_like(values)
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(degree)
    good = degree > 0
    inverse[good] = 1.0 / np.sqrt(degree[good])
    scaler = sparse.diags(inverse)
    normalised = scaler @ graph @ scaler

    a = float(alpha)
    field_ = values.copy()
    for _ in range(int(iterations)):
        updated = a * (normalised @ field_) + (1.0 - a) * values
        moved = float(np.abs(updated - field_).max()) if updated.size else 0.0
        field_ = updated
        if moved <= float(tolerance):
            break
    return field_


# ---------------------------------------------------------------------------
# 4. the well constraint -- the sudoku step
# ---------------------------------------------------------------------------

def constrain_to_fractions(mass: np.ndarray,
                           wells: Sequence[str],
                           names: Sequence[str],
                           fractions: Mapping[str, Mapping[str, float]], *,
                           iterations: int = 200,
                           tolerance: float = 1e-9) -> np.ndarray:
    """Project the propagated mass onto the counts sequencing implies.

    Within each well, scale the guide columns so each sums to ``pi_g * N_w``
    and renormalise the rows to 1, alternately. This is iterative
    proportional fitting -- the same fixed point
    :func:`spacr.guide_attribution.posterior` uses, applied here to
    graph-propagated evidence instead of a one-dimensional likelihood.

    IT IS THE ROW CONSTRAINT OF THE PUZZLE. Without it the graph would
    happily give every cell in a well to whichever guide had the most
    anchors, because nothing would say a well has only so many of each.
    """
    values = np.asarray(mass, dtype=float).copy()
    labels = np.asarray(wells)
    order = {str(name): i for i, name in enumerate(names)}
    out = np.zeros_like(values)
    for well in sorted(set(labels.tolist())):
        rows = np.flatnonzero(labels == well)
        if rows.size == 0:
            continue
        block = values[rows, :].copy()
        here = fractions.get(well, {}) or {}
        total = float(sum(float(v) for v in here.values() if np.isfinite(v)))
        if total <= 0:
            share = np.full(len(order), 1.0 / max(len(order), 1))
        else:
            share = np.zeros(len(order), dtype=float)
            for guide, value in here.items():
                index = order.get(str(guide))
                if index is not None and np.isfinite(value):
                    share[index] = float(value) / total
        if share.sum() <= 0:
            share = np.full(len(order), 1.0 / max(len(order), 1))
        target = share * rows.size
        # A cell no anchor reached gets the well's prior rather than zero:
        # it is a cell, it carries something, and "no idea" is the answer.
        empty = block.sum(axis=1) <= 0
        if empty.any():
            block[empty, :] = share
        for _ in range(int(iterations)):
            sums = block.sum(axis=1, keepdims=True)
            sums[sums <= 0] = 1.0
            block = block / sums
            columns = block.sum(axis=0)
            moved = float(np.abs(columns - target).max())
            if moved <= float(tolerance) * max(rows.size, 1):
                break
            factor = np.divide(target, columns,
                               out=np.ones_like(columns), where=columns > 0)
            block = block * factor
        sums = block.sum(axis=1, keepdims=True)
        sums[sums <= 0] = 1.0
        out[rows, :] = block / sums
    return out


# ---------------------------------------------------------------------------
# 5. the method
# ---------------------------------------------------------------------------

def sudoku(features: np.ndarray,
           scores: np.ndarray,
           wells: Sequence[str],
           fractions: Mapping[str, Mapping[str, float]],
           guides: Sequence[str], *,
           anchors: Optional[Mapping[str, Sequence[int]]] = None,
           neighbours: int = 15,
           alpha: float = 0.9,
           decision: float = DEFAULT_DECISION,
           reach_floor: float = DEFAULT_REACH_FLOOR,
           anchor_quantile: float = 0.9,
           anchor_min_fraction: float = 0.5,
           use_score_as_feature: bool = False,
           mutual: bool = True) -> SudokuResult:
    """Annotate every cell, across wells, with two scores kept apart.

    :param features: ``(n_cells, n_features)`` cell measurements.
    :param scores: the classification score per cell -- used to CHOOSE
        anchors, and by default not used as a graph feature.
    :param wells: one well label per cell.
    :param fractions: ``{well: {guide: fraction}}``.
    :param guides: the guides to annotate. The user's choice, and several is
        the point: "the user may choose several high scoring guides".
    :param anchors: optional explicit anchor indices per guide, overriding
        :func:`anchors_for`.
    :param use_score_as_feature: put the classification score in the graph
        as well. **Off by default and that is a correctness decision, not a
        preference**: the anchors are chosen by score, so a graph built on
        the score would place every high-scoring cell near every guide's
        anchors and affirm all of them. Turning this on makes the method
        partly tautological, and the report says so when it is on.
    :returns: the :class:`SudokuResult`.

    THE FOUR OUTCOMES, which is why two scores are carried rather than one:

      affirm high, eliminate low   -- confidently this guide
      affirm low,  eliminate high  -- confidently NOT this guide
      affirm high, eliminate high  -- the guides do not separate here
      affirm low,  eliminate low   -- unlike every anchor: the odd cell,
                                      and the one a single ratio hides by
                                      mapping it to the same value as the
                                      row above.
    """
    values = np.asarray(features, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    score_values = np.asarray(scores, dtype=float).reshape(-1)
    labels = [str(w) for w in wells]
    names = tuple(str(g) for g in guides)
    n = values.shape[0]
    empty = np.zeros((n, len(names)), dtype=float)
    if n == 0 or not names:
        return SudokuResult((), empty, empty, np.zeros(n), empty, names,
                            {"reason": "no cells or no guides"})

    graph_features = (np.column_stack([values, score_values])
                      if use_score_as_feature else values)
    graph = similarity_graph(graph_features, neighbours=neighbours,
                             mutual=mutual)

    seeds = np.zeros((n, len(names)), dtype=float)
    counts: Dict[str, int] = {}
    for column, guide in enumerate(names):
        if anchors is not None and guide in anchors:
            picked = np.asarray(list(anchors[guide]), dtype=int)
        else:
            picked = anchors_for(guide, labels, fractions, score_values,
                                 quantile=anchor_quantile,
                                 min_fraction=anchor_min_fraction)
        picked = picked[(picked >= 0) & (picked < n)]
        counts[guide] = int(picked.size)
        if picked.size:
            seeds[picked, column] = 1.0

    mass = propagate(graph, seeds, alpha=alpha)
    mass = np.clip(mass, 0.0, None)

    # THE TWO SCORES, BEFORE ANY NORMALISATION.
    reach = mass.sum(axis=1)
    total = mass.sum(axis=1, keepdims=True)
    safe = np.where(total > 0, total, 1.0)
    affirm = mass / safe
    eliminate = 1.0 - affirm
    # Reach relative to the typical cell: an absolute cut-off would depend
    # on the graph's size and its edge weights, which are not the user's to
    # reason about.
    typical = float(np.median(reach[reach > 0])) if np.any(reach > 0) else 0.0
    relative = reach / typical if typical > 0 else np.zeros_like(reach)

    posterior = constrain_to_fractions(mass, labels, names, fractions)

    called: list = []
    for row in range(n):
        if relative[row] < float(reach_floor):
            called.append(ABSTAIN)                # unlike every anchor
            continue
        best = int(np.argmax(posterior[row]))
        if float(posterior[row, best]) < float(decision):
            called.append(ABSTAIN)                # a coin flip
            continue
        called.append(names[best])

    report: Dict[str, object] = {
        "cells": n,
        "guides": len(names),
        "anchors": counts,
        "edges": int(graph.nnz // 2),
        "isolated": int((np.asarray(graph.sum(axis=1)).reshape(-1) <= 0).sum()),
        "abstained": int(sum(1 for g in called if g == ABSTAIN)),
        "abstained_for_reach": int((relative < float(reach_floor)).sum()),
        "decision": float(decision),
        "reach_floor": float(reach_floor),
        "score_in_graph": bool(use_score_as_feature),
    }
    if use_score_as_feature:
        report["warning"] = (
            "the classification score is a graph feature AND chooses the "
            "anchors, so affirmation is partly circular")
    if not any(counts.values()):
        report["warning"] = (
            "no guide reached the anchor threshold: no well gives any of "
            "them a large enough share, so nothing was annotated")

    return SudokuResult(tuple(called), affirm, eliminate, relative,
                        posterior, names, report)


def sudoku_all(features: np.ndarray,
               scores: np.ndarray,
               wells: Sequence[str],
               fractions: Mapping[str, Mapping[str, float]],
               ranking: Sequence[Tuple[str, float]], *,
               decision: float = DEFAULT_DECISION,
               max_guides: int = 50,
               **kwargs) -> SudokuResult:
    """Sudoku over every guide, in confidence order, claiming as it goes.

    :param ranking: ``[(guide, confidence)]`` in descending processing order.
        The caller defines confidence, for example by combining effect size
        and statistical significance.
    :param max_guides: a stop, so a screen with 1,500 guides does not run
        1,500 graph builds by accident.
    :returns: one :class:`SudokuResult` over all cells.

    Each round runs :func:`sudoku` for one guide over the cells still
    unclaimed, keeps the cells it decides, and removes them. It stops when a
    round claims nothing -- which is the coin flip, reached rather than
    counted.

    THE ORDER IS THE RISK, and the report carries what is needed to judge
    it: `claimed_by_round`. A greedy method commits early, the first guide's
    mistakes are inherited by every later one, and cells it takes are never
    reconsidered. A caller that wants to know how much that mattered runs
    this twice with the order perturbed and compares -- which is what
    :mod:`spacr.annotation_validation` does.
    """
    values = np.asarray(features, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    n = values.shape[0]
    labels = [str(w) for w in wells]
    order = [(str(g), float(c)) for g, c in ranking][: int(max_guides)]
    order.sort(key=lambda item: -item[1])
    names = tuple(g for g, _ in order)

    called = [ABSTAIN] * n
    affirm = np.zeros((n, len(names)), dtype=float)
    eliminate = np.ones((n, len(names)), dtype=float)
    posterior = np.zeros((n, len(names)), dtype=float)
    reach = np.zeros(n, dtype=float)
    unclaimed = np.ones(n, dtype=bool)
    rounds: list = []

    for column, (guide, confidence) in enumerate(order):
        live = np.flatnonzero(unclaimed)
        if live.size < 2:
            break
        # EVERY GUIDE IN THE RUN, ONE GUIDE COMMITTED. Running this with
        # `[guide]` alone is degenerate and was, briefly, the bug the
        # benchmark caught: with ONE column, `constrain_to_fractions`
        # normalises each row over a single guide, so every posterior is
        # exactly 1.0, every cell clears the decision bar, and the first
        # guide claims the entire screen. It scored at the null.
        #
        # A posterior is a COMPARISON. Comparing a guide against nothing
        # returns the prior, which is the same lesson `attribute_well`
        # records for its own per-well call.
        here = sudoku(values[live], np.asarray(scores)[live],
                      [labels[i] for i in live], fractions, names,
                      decision=decision, **kwargs)
        mine = here.names.index(guide) if guide in here.names else None
        taken = 0
        for position, index in enumerate(live):
            if mine is not None:
                affirm[index, column] = here.affirm[position, mine]
                eliminate[index, column] = here.eliminate[position, mine]
                posterior[index, column] = here.posterior[position, mine]
            reach[index] = max(reach[index], here.reach[position])
            # Only THIS guide's cells are claimed this round. The others
            # were computed to make the comparison honest and are left for
            # their own round, when the pool they compete over is smaller.
            if here.guides[position] == guide:
                called[index] = guide
                unclaimed[index] = False
                taken += 1
        rounds.append({"guide": guide, "confidence": confidence,
                       "claimed": taken, "left": int(unclaimed.sum())})
        if taken == 0:
            # THE STOPPING RULE, REACHED NOT COUNTED. Nothing this round
            # cleared the decision bar, so nothing later will either -- the
            # pool only shrinks and the guides only get less confident.
            break

    report = {
        "cells": n,
        "guides_considered": len(rounds),
        "guides_offered": len(order),
        "claimed": int((~unclaimed).sum()),
        "abstained": int(unclaimed.sum()),
        "claimed_by_round": rounds,
        "stopped_early": bool(rounds and rounds[-1]["claimed"] == 0),
        "decision": float(decision),
    }
    return SudokuResult(tuple(called), affirm, eliminate, reach, posterior,
                        names, report)
