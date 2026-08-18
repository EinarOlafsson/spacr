"""Robust rank aggregation: guides ranked, then aggregated to a gene BY RANK.

Instruction 133, asked for on 2026-08-17 as one of the "modeling tools ...
where we could plug in the same dependent and independent variables and get an
estimate of which gRNAs/genes are involved".

WHY THIS ONE IS RECOMMENDED FOR A CRISPR SCREEN, and it is not a preference.
Every regression backend spaCR has answers the gene question by SUMMING the
gene's guide fractions into a `gene_fraction` term -- which is exactly the
collinearity instruction 132 exists to fix, because the sum of a gene's guides
is a linear combination of those guides by construction. RRA never forms that
sum. It ranks guides against each other and asks a question about the RANKS a
gene's guides occupy, so a design matrix that is singular for OLS is not a
problem it can have.

It is also the field standard: this is MAGeCK's alpha-RRA, which is what a
reviewer of a pooled screen expects to see, and running it makes this screen
directly comparable to the published ones its phenotype scores come from.

THE STATISTIC. Rank every guide in the screen by its score, worst first for
the depleted direction. A gene with k guides occupies normalised ranks
r_1 <= ... <= r_k in (0, 1]. Under the null -- this gene is no different from
any other -- those are the order statistics of k draws from Uniform(0, 1), so
r_i is Beta(i, k - i + 1). The statistic is

    rho = min_i  Beta(i, k - i + 1).cdf(r_i)

the smallest probability of seeing an i-th-best guide at least that good.
alpha-RRA restricts the minimum to ranks in the top `alpha` fraction: a gene
with one strong guide and three dead ones is a real hit in a library where a
third of guides do not cut, and taking the minimum over all k lets the dead
guides pull rho back toward 1.

THE P VALUE IS PERMUTED, NOT READ OFF A TABLE. `rho` is a minimum over k
dependent order statistics, so its null distribution is not Beta and there is
no closed form. Genes are permuted at the GUIDE level -- guides reassigned to
genes of the same size -- which is the null "this gene's guides are an
arbitrary set of k guides", i.e. exactly what a screen hit is a departure
from. Genes with the same k share a null, so the cost is one permutation set
per distinct guide count, not one per gene.

BOTH DIRECTIONS, SEPARATELY. Depletion and enrichment are two questions and
MAGeCK reports them apart; a two-sided rank statistic would call a gene whose
guides split half up and half down, which is the signature of a bad guide set
rather than of a phenotype.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: The top fraction of the ranking a gene's guides are aggregated over.
#: MAGeCK's default, and the reason a gene with one real guide out of four is
#: still findable in a library where guides fail.
DEFAULT_ALPHA = 0.25

#: Permutations per distinct guide count. Ten thousand puts the resolution of
#: the smallest reportable P value at 1e-4, which is finer than any FDR
#: threshold a screen is read at and cheap because it is shared across every
#: gene with that many guides.
DEFAULT_PERMUTATIONS = 10000

#: Which tail. "neg" is depletion (a low score is a hit), "pos" is
#: enrichment, "both" reports each in its own columns.
DIRECTIONS = ("neg", "pos", "both")


def _beta_cdf(ranks: np.ndarray, k: int) -> np.ndarray:
    """``Beta(i, k - i + 1).cdf(r_i)`` for every column of ``ranks``.

    ``ranks`` is ``(n, k)``, sorted ascending along the last axis. Vectorised
    over rows because the permutation null draws tens of thousands of them.
    """
    from scipy.stats import beta

    order = np.arange(1, k + 1)
    return beta.cdf(ranks, order, k - order + 1)


def _rho(sorted_ranks: np.ndarray, k: int, alpha: float) -> np.ndarray:
    """The alpha-RRA statistic for each row of ``sorted_ranks``."""
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

    Guides whose score is not finite are DROPPED, not ranked last. A NaN
    coefficient means the fit could not estimate that guide; ranking it worst
    would turn "not measured" into "strongly depleted", which is the most
    consequential silent error this function could make.
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
    """The formula and what is modelled, for the model tab's text box."""
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
