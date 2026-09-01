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

#: Annotation assigned when the method cannot identify a guide. The value is
#: shared with :mod:`spacr.guide_attribution` for downstream compatibility.
ABSTAIN = "Non_annotated"

#: Minimum anchor reach, expressed as a fraction of median reach, required to
#: assign a guide. Relative scaling accommodates graph size and edge weights.
DEFAULT_REACH_FLOOR = 0.15

#: Minimum posterior probability required for assignment. The value matches
#: ``guide_attribution.DEFAULT_THRESHOLD`` for ``delta = 0.05``.
DEFAULT_DECISION = 0.55


@dataclass(frozen=True)
class SudokuResult:
    """Cell-level guide assignments and their separate evidence components.

    :ivar guides: one guide name per input cell, or
        :data:`ABSTAIN`.
    :ivar affirm: ``(n_cells, n_guides)`` support from each guide's anchors.
    :ivar eliminate: ``(n_cells, n_guides)`` support from competing anchors.
        Keeping this separate from ``affirm`` distinguishes ambiguous cells
        from cells unsupported by any anchor population.
    :ivar reach: ``(n_cells,)`` total propagated anchor mass.
    :ivar posterior: ``(n_cells, n_guides)`` after the well constraint.
    :ivar names: the guide names, in the column order of the matrices.
    :ivar report: assignment counts, settings, and diagnostic warnings.
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

    :param guide: guide whose high-fraction wells and high-scoring cells are
        being selected as anchors.
    :param wells: one well label per cell.
    :param fractions: ``{well: {guide: fraction}}``.
    :param scores: the classification score per cell.
    :param quantile: within an anchor well, the score quantile above which a
        cell is taken.
    :param min_fraction: minimum sequencing fraction for a well to contribute
        anchors. This limits anchor selection to wells in which a high-scoring
        cell is plausibly associated with the guide.
    :param max_per_well: maximum anchors contributed by one well.
    :returns: cell indices, possibly empty.

    Anchor selection uses the classifier score. To avoid circular inference,
    :func:`sudoku` excludes that score from graph features by default; cell
    morphology then determines propagation beyond the anchors.
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
    """Construct a symmetric k-nearest-neighbour cell-affinity graph.

    :param features: ``(n_cells, n_features)``, standardised here.
    :param neighbours: number of nearest neighbours considered per cell.
    :param mutual: retain an edge only when both cells identify each other as
        neighbours. This allows isolated outliers to have zero reach rather
        than receiving forced connections.
    :returns: sparse CSR affinity matrix with a zero diagonal.

    Edge weights use a heat kernel with a local scale defined by each cell's
    distance to its kth neighbour. Local scaling supports populations with
    different sampling densities.
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
              tolerance: float = 1e-4,
              dtype=np.float32) -> np.ndarray:
    """Propagate guide-anchor mass through a cell-similarity graph.

    :param graph: Sparse affinity matrix returned by
        :func:`similarity_graph`, with shape ``(n_cells, n_cells)``.
    :param seeds: Anchor weights with shape ``(n_cells, n_guides)``. Rows for
        unanchored cells should contain zeros.
    :param alpha: Relative weight assigned to neighbouring cells. Values near
        one favour graph propagation; ``alpha=1`` removes the seed term and
        is therefore unsuitable for guide assignment.
    :param iterations: Maximum number of propagation updates.
    :param tolerance: Stop when the largest element-wise update is no greater
        than this value.
    :param dtype: Floating-point type used for the seed and normalized graph
        matrices. The ``float32`` default reduces memory and runtime for large
        screens; use ``float64`` when additional numerical precision is
        required.
    :returns: Unnormalized propagated mass with shape
        ``(n_cells, n_guides)``.

    The update follows local-and-global consistency,
    ``F <- alpha S F + (1 - alpha) Y``, where
    ``S = D^-1/2 W D^-1/2``. The result is intentionally not row-normalized:
    :func:`sudoku` uses the total received mass to distinguish cells with
    weak support from confident assignments.
    """
    from scipy import sparse

    values = np.asarray(seeds, dtype=dtype)
    n = values.shape[0]
    if n == 0 or graph.shape[0] != n:
        return np.zeros_like(values)
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(degree)
    good = degree > 0
    inverse[good] = 1.0 / np.sqrt(degree[good])
    scaler = sparse.diags(inverse)
    normalised = (scaler @ graph @ scaler).astype(dtype)

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

    :param mass: graph-propagated cell-by-guide evidence matrix.
    :param wells: one well identifier for each row of ``mass``.
    :param names: guide identifiers in the column order of ``mass``.
    :param fractions: sequencing fractions mapped by well and then guide.

    Within each well, scale the guide columns so each sums to ``pi_g * N_w``
    and renormalise the rows to 1, alternately. This is iterative
    proportional fitting -- the same fixed point
    :func:`spacr.guide_attribution.posterior` uses, applied here to
    graph-propagated evidence instead of a one-dimensional likelihood.

    The row constraint prevents the graph from assigning every cell in a well
    to the guide with the greatest anchor mass when sequencing supports only
    a limited fraction for that guide.
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
    """Assign guides to cells while retaining competing evidence separately.

    :param features: ``(n_cells, n_features)`` cell measurements.
    :param scores: classification score per cell, used to choose
        anchors, and by default not used as a graph feature.
    :param wells: one well label per cell.
    :param fractions: ``{well: {guide: fraction}}``.
    :param guides: guide identifiers to consider for assignment.
    :param anchors: optional explicit anchor indices per guide, overriding
        :func:`anchors_for`.
    :param use_score_as_feature: include the classifier score in graph
        features. Disabled by default because the score also selects anchors;
        enabling it introduces circular evidence and is recorded in the
        result report.
    :returns: the :class:`SudokuResult`.

    Separate support and competing-label evidence distinguish four outcomes:

    * high support, low competition: confident assignment;
    * low support, high competition: confident exclusion;
    * high support, high competition: ambiguous between guides;
    * low support, low competition: unsupported by the anchor populations.
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
    """Assign guides sequentially in descending confidence order.

    :param features: cell-by-feature matrix used to propagate anchor support.
    :param scores: classification score for every cell, aligned to ``features``.
    :param wells: well identifier for every cell.
    :param fractions: sequencing fractions as ``{well: {guide: fraction}}``.
    :param ranking: ``[(guide, confidence)]`` in descending processing order.
        The caller defines confidence, for example by combining effect size
        and statistical significance.
    :param max_guides: maximum number of ranked guides to process.
    :returns: one :class:`SudokuResult` over all cells.

    Each round applies :func:`sudoku` to unclaimed cells and removes accepted
    assignments. Processing stops when a round assigns no cells. Because this
    greedy procedure is order-sensitive, ``claimed_by_round`` is retained in
    the report; :mod:`spacr.annotation_validation` evaluates sensitivity to
    ranking order.
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
        mine = here.names.index(guide)
        taken = 0
        for position, index in enumerate(live):
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
