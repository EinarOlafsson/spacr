"""A guide and a probability for every cell.

Instruction 173. A pooled screen never observes which cell carried which
guide: sequencing gives a FRACTION per well. But a guide with a measurable
effect on the classification score leaves a trace in the scores themselves,
and that is enough to attribute cells probabilistically.

    prior       pi_g        the guide's read fraction in the well, normalised
    likelihood  f_g(s)      how likely this score is if the cell carries g
    posterior   r_ig    ∝   pi_g * f_g(s_i)

AND THE CONSTRAINT THAT MAKES IT WORK:

    sum_i r_ig  =  N * pi_g       for every guide

Without it, two guides pointing the same way both claim the same top cells --
"if i have 2 grnas both in the same direction and want to attribute cells to
both my arithmatic wont work well because both pick the top hits which would
be wrong". With it, the total mass attributed to each guide is pinned to its
read fraction, so they SPLIT the top cells in proportion instead. It is
enforced by iterative proportional fitting, which is the same fixed point as
fitting a mixture whose mixing proportions are known and fixed.

WHAT THIS IS NOT. It is not a genotype. Every name in this module says
"attributed" for that reason: a reader who takes the answer for an observation
has been misled by the column name alone.

TWO GUIDES WITH THE SAME EFFECT ARE UNIDENTIFIABLE, and the method says so
rather than inventing a split: their posteriors come back at exactly
pi_1 : pi_2 for every cell, which is flat, ambiguous and correct.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "AMBIGUOUS",
    "assign_well",
    "Assignment",
    "DEFAULT_THRESHOLD",
    "LIKELIHOODS",
    "Attribution",
    "attributable",
    "attribute_well",
    "normalise_fractions",
    "posterior",
]

#: The tag a cell gets when no guide reached the threshold.
AMBIGUOUS = "ambiguous"

#: "if a cell only has grna attributions that are below 0.55 or something then
#: it gets a ambiguous tag". Settable; this is the default that was asked for.
DEFAULT_THRESHOLD = 0.55

#: The likelihood families. `lognormal` first because it is on the same scale
#: the regression works in, so an effect and a score are commensurate; `beta`
#: kept because a classification score actually lives in [0, 1] and a log does
#: not (instruction 173, and 174 for the regression's own beta transform).
LIKELIHOODS: Tuple[str, ...] = ("lognormal", "beta")


def normalise_fractions(fractions: Mapping[str, float]) -> Dict[str, float]:
    """The guides' fractions, scaled to sum to 1.

    "first calculate what all the fractions in that well add up to if not 1
    then normalize to 1". A well whose fractions sum to nothing usable comes
    back empty rather than uniform: inventing a prior where there is no
    sequencing is the one thing this must not do.
    """
    usable = {str(g): float(v) for g, v in (fractions or {}).items()
              if v is not None and math.isfinite(float(v)) and float(v) > 0}
    total = float(sum(usable.values()))
    if total <= 0:
        return {}
    return {g: v / total for g, v in usable.items()}


def _lognormal_density(scores: np.ndarray, effect: float,
                       centre: float, scale: float) -> np.ndarray:
    """Density of the score for a cell carrying a guide of size ``effect``.

    Normal on the scale the caller is already working in -- the regression's
    own transformed scale -- so ``effect`` and ``scores`` are commensurate.
    The name says lognormal because that scale is the log one by default.
    """
    sigma = float(scale) if scale and math.isfinite(scale) and scale > 0 else 1.0
    z = (scores - (centre + float(effect))) / sigma
    return np.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def _beta_density(scores: np.ndarray, effect: float,
                  centre: float, scale: float) -> np.ndarray:
    """Beta density for a score in [0, 1], centred by ``effect``.

    The mean is moved by the effect on the LOGIT scale, which is where a
    proportion's effects are additive, and the concentration comes from the
    spread the caller measured. Scores are squeezed off 0 and 1 because a
    beta density is not finite there and a screen produces both.
    """
    from scipy.stats import beta as _beta

    eps = 1e-6
    x = np.clip(np.asarray(scores, dtype=float), eps, 1.0 - eps)
    base = min(max(float(centre), eps), 1.0 - eps)
    mean = 1.0 / (1.0 + math.exp(-(math.log(base / (1.0 - base)) + float(effect))))
    mean = min(max(mean, eps), 1.0 - eps)
    spread = float(scale) if scale and scale > 0 else 0.1
    # Concentration from the spread: var = mean(1-mean)/(1+nu) inverted.
    variance = min(max(spread * spread, 1e-9), mean * (1.0 - mean) * 0.999)
    nu = max(mean * (1.0 - mean) / variance - 1.0, 1e-3)
    return _beta.pdf(x, mean * nu, (1.0 - mean) * nu)


def _density(kind: str, scores, effect, centre, scale) -> np.ndarray:
    if str(kind).lower() == "beta":
        return _beta_density(scores, effect, centre, scale)
    return _lognormal_density(scores, effect, centre, scale)


def posterior(scores: Sequence[float], priors: Mapping[str, float],
              effects: Mapping[str, float], *,
              centre: float = 0.0, scale: float = 1.0,
              likelihood: str = "lognormal",
              iterations: int = 200,
              tolerance: float = 1e-9) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """``(r, guides)`` -- each cell's probability of carrying each guide.

    :returns: ``r`` with one row per cell and one column per guide, rows
        summing to 1, and COLUMNS summing to ``n_cells * prior`` -- the
        constraint that stops two same-direction guides both claiming the
        same cells.

    Iterative proportional fitting: scale the columns to the target masses,
    renormalise the rows to 1, repeat. Both operations are monotone in the
    right sense and the fixed point satisfies both constraints at once. It is
    the same answer as a mixture fitted with its mixing proportions held at
    the sequencing fractions.
    """
    guides = tuple(priors.keys())
    values = np.asarray(list(scores), dtype=float)
    n = int(values.size)
    if not guides or not n:
        return np.zeros((n, len(guides)), dtype=float), guides

    weights = np.array([float(priors[g]) for g in guides], dtype=float)
    density = np.column_stack([
        _density(likelihood, values, float(effects.get(g, 0.0)), centre, scale)
        for g in guides])
    density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    # A cell no guide can explain is given the prior rather than dropped: it
    # is a cell, it carries something, and the honest answer is "no idea".
    dead = density.sum(axis=1) <= 0
    if dead.any():
        density[dead, :] = weights

    target = weights * n
    r = density * weights                       # start from plain Bayes
    for _ in range(int(iterations)):
        rows = r.sum(axis=1, keepdims=True)
        rows[rows <= 0] = 1.0
        r = r / rows                            # every cell carries one guide
        columns = r.sum(axis=0)
        moved = np.abs(columns - target).max()
        if moved <= tolerance * max(n, 1):
            break
        factor = np.divide(target, columns, out=np.ones_like(columns),
                           where=columns > 0)
        r = r * factor                          # pin each guide to its reads
    rows = r.sum(axis=1, keepdims=True)
    rows[rows <= 0] = 1.0
    return r / rows, guides


@dataclass(frozen=True)
class Attribution:
    """What one cell was attributed, and how sure that is."""

    guide: str
    probability: float
    ambiguous: bool
    entropy: float
    runner_up: str = ""

    @property
    def called(self) -> bool:
        return not self.ambiguous


def _entropy(row: np.ndarray) -> float:
    """Shannon entropy in bits -- how arbitrary this cell's call was."""
    p = row[row > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0


def attribute_well(scores: Sequence[float], fractions: Mapping[str, float],
                   effects: Mapping[str, float], *,
                   threshold: float = DEFAULT_THRESHOLD,
                   centre: float = 0.0, scale: float = 1.0,
                   likelihood: str = "lognormal",
                   seed: int = 0) -> Tuple[Attribution, ...]:
    """One :class:`Attribution` per cell, in the order the scores came.

    :param threshold: "if it gets a grna above 0.55 then it gets annotated
        with that gene and the probability score", and below it the cell is
        tagged :data:`AMBIGUOUS` -- carrying the highest probability any guide
        reached anyway, because that number is the useful part of a refusal.
    :param seed: an exact tie is broken AT RANDOM, and the generator is
        seeded: spaCR runs are reproducible, and a coin flip that landed
        differently on a re-run would annotate the same screen two ways.
    """
    priors = normalise_fractions(fractions)
    r, guides = posterior(scores, priors, effects, centre=centre, scale=scale,
                          likelihood=likelihood)
    rng = np.random.default_rng(int(seed))
    out = []
    for row in r:
        if not guides or not row.size:
            out.append(Attribution(AMBIGUOUS, 0.0, True, 0.0))
            continue
        best = float(row.max())
        # EXACT ties only. `np.isclose` here would make an arbitrary choice on
        # values that are merely similar, which is a different claim.
        tied = np.flatnonzero(row == best)
        pick = int(tied[0] if tied.size == 1 else rng.choice(tied))
        order = np.argsort(row)[::-1]
        runner = str(guides[order[1]]) if row.size > 1 else ""
        ambiguous = best < float(threshold)
        out.append(Attribution(
            guide=AMBIGUOUS if ambiguous else str(guides[pick]),
            probability=best, ambiguous=ambiguous, entropy=_entropy(row),
            runner_up=runner))
    return tuple(out)


def attributable(effect: float, scale: float, prior: float, *,
                 threshold: float = DEFAULT_THRESHOLD) -> Tuple[bool, float]:
    """``(can_it, best_possible)`` -- can this guide ever be called?

    A guide whose effect is small against the spread of scores can NEVER reach
    the threshold: not with more cells, not with a better fit. It is
    arithmetic and not sample size, and a user deserves to know which of their
    hits support cell-level work BEFORE any assignment runs.

    The bound is the posterior at the best possible score -- the one furthest
    into this guide's own tail -- against a background of everything else at
    its own centre. Anything below the threshold there is below it everywhere.
    """
    sigma = float(scale) if scale and scale > 0 else 1.0
    p = min(max(float(prior), 0.0), 1.0)
    if p <= 0:
        return False, 0.0
    if p >= 1:
        return True, 1.0
    # Likelihood ratio at the guide's own centre, against the rest at theirs.
    separation = abs(float(effect)) / sigma
    ratio = math.exp(0.5 * separation * separation)
    best = (p * ratio) / (p * ratio + (1.0 - p))
    return best >= float(threshold), float(best)


# --------------------------------------------------------------------------- #
#  The constrained assignment -- the Sudoku one
# --------------------------------------------------------------------------- #
#
# "my mind always goes to suduko where you have rules and conditions that must
# be met and you use the little information you have within the confines of
# the rules to do your inference."
#
# That is the better framing, and the soft posterior above throws away its
# central mechanism. Sudoku's power is not probability, it is EXCLUSION: if
# this cell takes that value, no other cell in the region can. `posterior`
# gives each cell an independent marginal and then notices that none of them
# is confident. The constraints here are exactly Sudoku's shape:
#
#     every cell carries EXACTLY ONE guide
#     guide g occupies EXACTLY round(N * pi_g) cells of the well
#     a guide absent from the well occupies NONE of it
#
# Solved as a minimum-cost assignment: expand each guide into its own integer
# number of slots and match cells to slots so the total -log likelihood is
# smallest. Every count is then exactly right BY CONSTRUCTION, and every cell
# has a definite guide -- which the marginal posterior can never deliver when
# the priors are small.
#
# WHAT IT DOES NOT DO, and this is the honest half. An assignment being
# OPTIMAL does not make it CERTAIN. When the evidence is weak, many
# assignments are nearly as good, and swapping two cells costs almost
# nothing. `Assignment.degeneracy` reports exactly that, so a reader can tell
# a solved grid from one that merely satisfies the rules.


@dataclass(frozen=True)
class Assignment:
    """Every cell's guide, with the counts exactly as sequenced."""

    guides: Tuple[str, ...]
    #: -log likelihood of this assignment; lower is better.
    cost: float
    #: Mean cost of swapping two cells' guides, in nats. Near zero means many
    #: assignments are equally good and this one is arbitrary.
    degeneracy: float
    #: How many cells each guide was given.
    counts: Dict[str, int] = field(default_factory=dict)

    @property
    def decisive(self) -> bool:
        """Whether the constraints actually pinned the answer down."""
        return self.degeneracy > 1.0


def assign_well(scores: Sequence[float], fractions: Mapping[str, float],
                effects: Mapping[str, float], *,
                centre: float = 0.0, scale: float = 1.0,
                likelihood: str = "lognormal",
                seed: int = 0) -> Assignment:
    """Give every cell exactly one guide, with the counts sequencing implies.

    :returns: an :class:`Assignment`. ``guides`` is one name per cell, in the
        order the scores came.

    The counts are ``round(N * pi_g)`` adjusted to sum to N exactly -- a
    rounding that left a cell unassigned would break the one rule that makes
    this an assignment at all. The largest remainders take the slack, which is
    the standard apportionment and is deterministic.
    """
    from scipy.optimize import linear_sum_assignment

    priors = normalise_fractions(fractions)
    values = np.asarray(list(scores), dtype=float)
    n = int(values.size)
    names = tuple(priors)
    if not names or not n:
        return Assignment(guides=(AMBIGUOUS,) * n, cost=float("inf"),
                          degeneracy=0.0, counts={})

    # HOW MANY SLOTS EACH GUIDE GETS, summing to n exactly.
    exact = np.array([priors[g] * n for g in names], dtype=float)
    slots = np.floor(exact).astype(int)
    short = n - int(slots.sum())
    if short > 0:
        order = np.argsort(-(exact - np.floor(exact)))
        for index in order[:short]:
            slots[index] += 1
    elif short < 0:                                   # pragma: no cover - rare
        order = np.argsort(exact - np.floor(exact))
        for index in order[: -short]:
            slots[index] = max(slots[index] - 1, 0)

    density = np.column_stack([
        _density(likelihood, values, float(effects.get(g, 0.0)), centre, scale)
        for g in names])
    density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    cost_per_guide = -np.log(np.clip(density, 1e-300, None))

    # One column per SLOT, so the counts are a property of the matrix rather
    # than something checked afterwards.
    columns = np.repeat(np.arange(len(names)), slots)
    if columns.size != n:                             # pragma: no cover
        columns = columns[:n]
    cost = cost_per_guide[:, columns]
    rows, picks = linear_sum_assignment(cost)
    chosen = columns[picks]

    total = float(cost[rows, picks].sum())
    # HOW ARBITRARY IS IT: what this cell would cost under its best ALTERNATIVE
    # GUIDE, not its second-cheapest slot. Slots of the same guide have
    # identical cost, so the second-cheapest slot is almost always another
    # slot of the guide already chosen and the difference is exactly zero --
    # which made this read "arbitrary" for a perfectly decided grid. Caught by
    # the test that asks a decided assignment to score above an undecided one.
    order = np.empty(n, dtype=int)
    order[rows] = chosen
    best = cost_per_guide[np.arange(n), order]
    others = cost_per_guide.copy()
    others[np.arange(n), order] = np.inf
    alternative = others.min(axis=1) if len(names) > 1 else best
    gap = alternative - best
    gap = gap[np.isfinite(gap)]
    degeneracy = float(np.mean(gap)) if gap.size else 0.0

    rng = np.random.default_rng(int(seed))
    del rng                                            # ties are broken by the
    # solver deterministically; the seed is accepted so the signature matches
    # `attribute_well` and a caller can pass one without thinking about it.
    assigned = tuple(str(names[c]) for c in chosen)
    counts = {g: int((chosen == i).sum()) for i, g in enumerate(names)}
    return Assignment(guides=assigned, cost=total, degeneracy=degeneracy,
                      counts=counts)
