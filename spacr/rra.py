"""Aggregate guide-level scores into gene-level calls using alpha-RRA.

RRA ranks guides and tests whether a gene's guides occupy unusually strong
positions. It therefore remains usable when a gene-fraction regression design
is singular.

Statistic
---------
Within each direction, rank every guide by score. A gene with ``k`` guides
occupies normalized ranks ``r_1 <= ... <= r_k`` in ``(0, 1]``. Under the
global null, ``r_i`` follows ``Beta(i, k - i + 1)``. The statistic is

    rho = min_i  Beta(i, k - i + 1).cdf(r_i)

where the minimum is restricted to ranks in the top ``alpha`` fraction.
Permutation p values are estimated by reassigning guides among genes with the
same guide count. Depletion and enrichment are evaluated separately.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: Top fraction of ranked guides used by alpha-RRA. This is the MAGeCK default.
DEFAULT_ALPHA = 0.25

#: Permutations per distinct guide count. Ten thousand gives a minimum
#: empirical p-value resolution of ``1e-4``.
DEFAULT_PERMUTATIONS = 10000

#: Supported tails: depletion, enrichment, or separate results for both.
DIRECTIONS = ("neg", "pos", "both")


def _beta_cdf(ranks: np.ndarray, k: int) -> np.ndarray:
    """``Beta(i, k - i + 1).cdf(r_i)`` for every column of ``ranks``.

    ``ranks`` is ``(n, k)``, sorted ascending along the last axis. Vectorised
    over rows because the permutation null draws tens of thousands of them.

    :param ranks: sorted normalized guide ranks, one gene per row.
    :param k: number of guide-rank columns in each row.
    :returns: beta-order-statistic cumulative probabilities with the same
        shape as ``ranks``.
    """
    from scipy.stats import beta

    order = np.arange(1, k + 1)
    return beta.cdf(ranks, order, k - order + 1)


def _rho(sorted_ranks: np.ndarray, k: int, alpha: float) -> np.ndarray:
    """Return the alpha-RRA statistic for each row of ``sorted_ranks``.

    :param sorted_ranks: ascending normalized guide ranks, one gene per row.
    :param k: number of guides represented by each row.
    :param alpha: largest normalized rank eligible for the row minimum.
    :returns: minimum eligible beta probability for every row; rows with no
        eligible rank receive ``1.0``.
    """
    scores = _beta_cdf(sorted_ranks, k)
    # OUTSIDE THE TOP alpha IS NOT CONSIDERED, and 1.0 is how a minimum
    # ignores it: a probability can never exceed 1, so a masked position can
    # never be the minimum unless every position is masked -- which happens
    # only when a gene has no guide in the top alpha at all, and rho = 1 is
    # then the right answer rather than a missing value.
    scores = np.where(sorted_ranks <= alpha, scores, 1.0)
    return scores.min(axis=-1)


def _null(k: int, n_guides: int, alpha: float, n_permutations: int,
          rng) -> np.ndarray:
    """``n_permutations`` draws of rho for a gene with ``k`` guides.

    The null is "these k guides are an arbitrary k of the library", so a draw
    is k normalised ranks sampled WITHOUT replacement from the n_guides
    positions -- without, because a gene never targets the same guide twice
    and sampling with replacement would let one very good rank appear k times.

    :param k: number of distinct guides assigned to the simulated gene.
    :param n_guides: number of ranked guide positions in the library.
    :param alpha: largest normalized rank eligible for each rho minimum.
    :param n_permutations: number of null gene assignments to draw.
    :param rng: NumPy random generator used for reproducible sampling.
    :returns: one null rho statistic per permutation.
    """
    positions = np.empty((n_permutations, k), dtype=np.int64)
    for row in range(n_permutations):
        positions[row] = rng.choice(n_guides, size=k, replace=False)
    ranks = np.sort((positions + 1) / n_guides, axis=1)
    return _rho(ranks, k, alpha)


def rank_aggregate(scores, groups, *, alpha: float = DEFAULT_ALPHA,
                   direction: str = "both",
                   n_permutations: int = DEFAULT_PERMUTATIONS,
                   seed: int = 0,
                   correction: str = "fdr_bh",
                   fdr_alpha: float = 0.05):
    """Aggregate per-guide scores to per-gene calls by rank.

    :param scores: one score per guide -- a coefficient, a log fold change,
        anything where more negative means more depleted.
    :param groups: the gene each guide belongs to, same length.
    :param alpha: the top fraction of the ranking to aggregate over.
    :param direction: ``"neg"``, ``"pos"`` or ``"both"``.
    :param n_permutations: draws per distinct guide count.
    :param seed: the permutation seed, so a re-run of the same screen gives
        the same P values. A screen whose hit list moves between runs of the
        same data is a screen nobody can act on.
    :param correction: any method :mod:`spacr.multiple_testing` accepts.
    :param fdr_alpha: the level the correction targets.
    :returns: a DataFrame, one row per gene, sorted by the strongest
        direction's P value.

    :raises ValueError: ``scores`` and ``groups`` differ in length, ``alpha``
        is outside (0, 1], or ``direction`` is not one of :data:`DIRECTIONS`.

    Guides with non-finite scores are omitted rather than ranked. This keeps an
    unestimated guide from being interpreted as strongly depleted.
    """
    import pandas as pd

    from .multiple_testing import adjust_p_values

    score = np.asarray(scores, dtype=float)
    gene = np.asarray(groups, dtype=object)
    if score.shape[0] != gene.shape[0]:
        raise ValueError(
            f"scores and groups must be the same length; got {score.shape[0]} "
            f"and {gene.shape[0]}")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1]; got {alpha!r}")
    if direction not in DIRECTIONS:
        raise ValueError(
            f"direction must be one of {DIRECTIONS}; got {direction!r}")

    keep = np.isfinite(score) & (gene != None)          # noqa: E711
    score, gene = score[keep], gene[keep]
    if not score.size:
        return pd.DataFrame(columns=["gene", "n_guides"])

    n_guides = score.size
    rng = np.random.default_rng(seed)
    wanted = ("neg", "pos") if direction == "both" else (direction,)

    genes, inverse = np.unique(gene, return_inverse=True)
    sizes = np.bincount(inverse)
    out: Dict[str, np.ndarray] = {"gene": genes, "n_guides": sizes}

    for tail in wanted:
        # A HIGH SCORE IS THE WORST RANK FOR DEPLETION and the best for
        # enrichment; `argsort` of the negated score is the whole difference
        # between the two tails.
        order = np.argsort(score if tail == "neg" else -score, kind="stable")
        rank = np.empty(n_guides, dtype=float)
        rank[order] = (np.arange(n_guides) + 1) / n_guides

        rhos = np.ones(genes.size, dtype=float)
        p = np.ones(genes.size, dtype=float)
        nulls: Dict[int, np.ndarray] = {}
        for index in range(genes.size):
            k = int(sizes[index])
            member = rank[inverse == index]
            rhos[index] = float(_rho(np.sort(member)[None, :], k, alpha)[0])
            if k not in nulls:
                nulls[k] = _null(k, n_guides, alpha, n_permutations, rng)
            null = nulls[k]
            # +1 IN BOTH PLACES. A permutation P value of exactly zero claims
            # a precision the permutation count does not have, and it is the
            # value that survives an FDR correction to become a "finding".
            p[index] = (1.0 + np.sum(null <= rhos[index])) / (null.size + 1.0)

        adjusted, _rejected = adjust_p_values(p, method=correction,
                                              alpha=fdr_alpha)
        out[f"rho_{tail}"] = rhos
        out[f"p_{tail}"] = p
        out[f"p_adj_{tail}"] = adjusted

    frame = pd.DataFrame(out)
    by = "p_neg" if "p_neg" in frame.columns else "p_pos"
    return frame.sort_values(by, kind="stable").reset_index(drop=True)


def describe(alpha: float = DEFAULT_ALPHA) -> str:
    """Return the formula and model guidance for the model tab's text box.

    :param alpha: top-rank cutoff to interpolate into the explanation; this
        formatter does not validate the supplied value.
    :returns: alpha-RRA formula, interpretation, and CRISPR-screen guidance.
    """
    return (
        "Robust rank aggregation (MAGeCK alpha-RRA). Guides are ranked "
        "against every other guide in the screen; a gene with k guides "
        "occupies normalised ranks r_1 <= ... <= r_k, and the statistic is\n\n"
        "    rho = min_i Beta(i, k - i + 1).cdf(r_i),  r_i <= "
        f"{alpha:g}\n\n"
        "the smallest probability of seeing an i-th-best guide at least that "
        "good if the gene's guides were an arbitrary set. Restricting the "
        "minimum to the top "
        f"{alpha:g} means a gene with one strong guide and three that did not "
        "cut is still findable. P values are permuted at the guide level and "
        "the two directions -- depleted and enriched -- are reported "
        "separately.\n\n"
        "Recommended for CRISPR screens. It aggregates to the gene BY RANK "
        "rather than by summing the gene's guide fractions, so it cannot "
        "suffer the collinearity that makes a guide-and-gene design matrix "
        "singular, and it is the field standard for pooled screens."
    )


__all__ = ["DEFAULT_ALPHA", "DEFAULT_PERMUTATIONS", "DIRECTIONS", "describe",
           "rank_aggregate"]
