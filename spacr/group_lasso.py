"""Guides grouped by gene, selected or dropped as a set.

Instruction 133, asked for on 2026-08-17 alongside RRA as one of the "modeling
tools ... where we could plug in the same dependent and independent variables
and get an estimate of which gRNAs/genes are involved".

WHAT IT IS. Ordinary lasso penalises each guide on its own, so from a gene's
four correlated guides it keeps whichever one happens to fit best and drops
the rest -- which reads as "one guide works and three do not" when the truth
is "the gene matters and the four guides are four measurements of it". Group
lasso penalises the gene's guides as a BLOCK:

    minimise  ||y - Xb||^2 / (2n)  +  lambda * sum_g sqrt(p_g) * ||b_g||_2

The L2 norm inside the sum is not squared, so its subgradient at zero is a
ball rather than a point, and the whole block goes to exactly zero or none of
it does. sqrt(p_g) is the standard correction that stops a gene with six
guides being penalised more than one with three purely for having more.

WHY IT IS RECOMMENDED FOR A CRISPR SCREEN. It is the penalised analogue of
what the mixed model says with random effects -- "which GENES are involved,
where each gene is measured by several guides" -- and it reaches the same
question with no random effects, no REML fit, and no `gene_fraction` column.
That last part matters: `gene_fraction` is the sum of a gene's guide
fractions, so a design carrying both blocks is singular by construction
(instruction 132). Here the gene enters as a GROUPING of the guide columns,
never as a column of its own, so the design stays full rank.

It also fits where OLS is undefined. This screen is 610 wells and 823 guides;
the penalty is what makes the answer well posed at p > n, and it is honest
about that -- a penalised fit is a statement about the assumption that most
guides do nothing, not new information.

WHAT IT DOES NOT GIVE YOU IS A P VALUE. A penalised coefficient's sampling
distribution is not the OLS one and a naive P value from it is wrong in a
direction that flatters the fit. :func:`stability_selection` is the answer:
re-fit on many subsamples and report the FRACTION of them in which a gene
survives, which is an error-controlled statement (Meinshausen & Buhlmann)
that needs no distributional assumption at all.

THE SOLVER is block coordinate descent with the exact group-soft-threshold
proximal step, in numpy. No new dependency: scikit-learn has no group lasso,
and the two packages that do are unmaintained.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Iterations before block coordinate descent gives up. Reached only on a
#: badly scaled design; the screen-sized problems here converge in tens.
MAX_ITERATIONS = 1000

#: Convergence: the largest coefficient change in a sweep, relative to the
#: largest coefficient.
TOLERANCE = 1e-6


def _groups(labels) -> Tuple[np.ndarray, List[np.ndarray]]:
    """``(unique labels, column indices per label)``, labels in sorted order."""
    labels = np.asarray(labels, dtype=object)
    unique = np.unique(labels)
    return unique, [np.flatnonzero(labels == one) for one in unique]


def _soft_threshold_block(vector: np.ndarray, amount: float) -> np.ndarray:
    """The proximal operator of ``amount * ||.||_2``.

    Shrinks the block toward the origin by ``amount`` and sets it to exactly
    zero once it is shorter than that -- which is the whole point: this is
    what makes a gene drop out as a unit rather than one guide at a time.
    """
    norm = float(np.linalg.norm(vector))
    if norm <= amount or norm == 0.0:
        return np.zeros_like(vector)
    return vector * (1.0 - amount / norm)


def max_lambda(X, y, labels) -> float:
    """The smallest penalty that zeroes every group.

    Useful as the top of a path: any larger value gives the same all-zero fit,
    and starting a path above it wastes every iteration spent there.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    _unique, blocks = _groups(labels)
    n = X.shape[0]
    correlation = X.T @ y / n
    return max(
        (float(np.linalg.norm(correlation[block]) / np.sqrt(block.size))
         for block in blocks), default=0.0)


def fit(X, y, labels, *, lam: float = 0.05, max_iterations: int = MAX_ITERATIONS,
        tolerance: float = TOLERANCE, fit_intercept: bool = True):
    """Fit the group lasso and return ``(coefficients, intercept, converged)``.

    :param X: the design, one column per guide.
    :param y: the response, one entry per well.
    :param labels: the gene each COLUMN belongs to, length ``X.shape[1]``.
    :param lam: the penalty. Zero is ordinary least squares and is allowed --
        it is how a caller checks the solver against a known answer.
    :param fit_intercept: centre both sides, so the intercept is never
        penalised. Penalising it would shrink the response's mean toward zero,
        which is not a claim anybody wants to make.
    :returns: ``coefficients`` (one per column), ``intercept``, and whether
        the sweep converged inside ``max_iterations``.

    :raises ValueError: the shapes disagree, or ``lam`` is negative.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be two-dimensional; got shape {X.shape}")
    if X.shape[0] != y.size:
        raise ValueError(
            f"X has {X.shape[0]} row(s) and y has {y.size}; they must match")
    if len(labels) != X.shape[1]:
        raise ValueError(
            f"labels has {len(labels)} entr(ies) and X has {X.shape[1]} "
            f"column(s); they must match")
    if lam < 0:
        raise ValueError(f"lam must not be negative; got {lam!r}")

    n = X.shape[0]
    if fit_intercept:
        x_mean, y_mean = X.mean(axis=0), float(y.mean())
        X, y = X - x_mean, y - y_mean
    else:
        x_mean, y_mean = np.zeros(X.shape[1]), 0.0

    _unique, blocks = _groups(labels)
    # THE STEP SIZE PER BLOCK is 1 / L where L is the block's largest squared
    # singular value -- the exact Lipschitz constant of that block's gradient,
    # so the proximal step is a descent step without any line search.
    steps = []
    for block in blocks:
        sub = X[:, block]
        largest = float(np.linalg.norm(sub, 2)) ** 2 / n
        steps.append(1.0 / largest if largest > 0 else 0.0)

    beta = np.zeros(X.shape[1])
    residual = y.copy()
    converged = False
    for _sweep in range(max_iterations):
        largest_change = 0.0
        for block, step in zip(blocks, steps):
            if step == 0.0:
                continue
            sub = X[:, block]
            current = beta[block]
            # Add this block back before its own gradient step, which is what
            # makes this coordinate descent rather than one global step.
            partial = residual + sub @ current
            gradient = sub.T @ partial / n
            updated = _soft_threshold_block(
                current + step * (gradient - sub.T @ (sub @ current) / n),
                step * lam * np.sqrt(block.size))
            change = float(np.max(np.abs(updated - current))) if block.size \
                else 0.0
            if change:
                residual = partial - sub @ updated
                beta[block] = updated
            largest_change = max(largest_change, change)
        scale = max(1.0, float(np.max(np.abs(beta))) if beta.size else 1.0)
        if largest_change <= tolerance * scale:
            converged = True
            break

    intercept = y_mean - float(x_mean @ beta) if fit_intercept else 0.0
    return beta, intercept, converged


def gene_effects(X, y, labels, **kwargs):
    """One number per gene: the L2 norm of its guide block.

    The block is zero or it is not, so the norm is the natural per-gene effect
    size -- and unlike a coefficient it is never negative, which is honest:
    group lasso says a gene's guides move the response TOGETHER, not in which
    direction each one does.
    """
    import pandas as pd

    beta, _intercept, converged = fit(X, y, labels, **kwargs)
    unique, blocks = _groups(labels)
    return pd.DataFrame({
        "gene": unique,
        "n_guides": [block.size for block in blocks],
        "effect": [float(np.linalg.norm(beta[block])) for block in blocks],
        "selected": [bool(np.any(beta[block])) for block in blocks],
        "converged": converged,
    }).sort_values("effect", ascending=False,
                   kind="stable").reset_index(drop=True)


def stability_selection(X, y, labels, *, lam: float = 0.05,
                        n_boot: int = 100, fraction: float = 0.5,
                        seed: int = 0, **kwargs):
    """How often each gene survives a re-fit on half the wells.

    :param n_boot: subsamples.
    :param fraction: the share of ROWS in each subsample. Half is
        Meinshausen & Buhlmann's choice and the one their error bound is
        stated for.
    :returns: a DataFrame with a ``selection_frequency`` per gene, sorted.

    THIS IS THE HONEST ANSWER TO "WHAT IS THE P VALUE". There is not one --
    the sampling distribution of a penalised coefficient is not the OLS one,
    and a P value computed as though it were is wrong in the direction that
    flatters the fit. A selection frequency is a statement about how
    reproducible the selection is, which is what a screen actually wants to
    know, and it needs no distributional assumption at all.
    """
    import pandas as pd

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1]; got {fraction!r}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1; got {n_boot!r}")

    unique, blocks = _groups(labels)
    counts = np.zeros(unique.size)
    rng = np.random.default_rng(seed)
    size = max(2, int(round(X.shape[0] * fraction)))
    for _draw in range(n_boot):
        rows = rng.choice(X.shape[0], size=size, replace=False)
        beta, _intercept, _converged = fit(X[rows], y[rows], labels, lam=lam,
                                           **kwargs)
        for index, block in enumerate(blocks):
            if np.any(beta[block]):
                counts[index] += 1

    return pd.DataFrame({
        "gene": unique,
        "n_guides": [block.size for block in blocks],
        "selection_frequency": counts / n_boot,
    }).sort_values("selection_frequency", ascending=False,
                   kind="stable").reset_index(drop=True)


def describe(lam: float = 0.05) -> str:
    """The formula and what is modelled, for the model tab's text box."""
    return (
        "Group lasso. The gene's guides are penalised as a block:\n\n"
        "    minimise  ||y - Xb||^2 / (2n)  +  lambda * sum_g sqrt(p_g) * "
        "||b_g||_2\n\n"
        f"with lambda = {lam:g}. The inner norm is not squared, so a gene's "
        "whole block of guides goes to exactly zero or none of it does -- "
        "which is the question a screen asks. Ordinary lasso instead keeps "
        "whichever one of a gene's correlated guides fits best and drops the "
        "rest, reading as 'one guide works' when the truth is 'the gene "
        "matters'. sqrt(p_g) stops a gene with six guides being penalised "
        "more than one with three for having more.\n\n"
        "Recommended for CRISPR screens. It is the penalised analogue of the "
        "mixed model's nesting -- which GENES are involved, each measured by "
        "several guides -- and the gene enters as a GROUPING of guide "
        "columns rather than as a summed `gene_fraction` column, so the "
        "design cannot be singular. It fits where OLS is undefined (823 "
        "guides, 610 wells).\n\n"
        "It gives no P value, and that is deliberate: a penalised "
        "coefficient's sampling distribution is not the OLS one. Use "
        "stability selection, which reports how often each gene survives a "
        "re-fit on half the wells."
    )


__all__ = ["MAX_ITERATIONS", "TOLERANCE", "describe", "fit", "gene_effects",
           "max_lambda", "stability_selection"]
