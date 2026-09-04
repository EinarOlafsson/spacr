"""A guide and a probability for every cell.

NOT `spacr.attribution`, which is the Grad-CAM and saliency library -- "what
a trained classifier attends to". This is "which guide is in this cell", a
different question with an unfortunately similar name.

A pooled screen never observes which cell carried which guide: sequencing
gives a FRACTION per well. But a guide with a measurable
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
    "Preflight",
    "preflight",
    "normalise_fractions",
    "posterior",
]

#: The tag a cell gets when no guide reached the threshold.
AMBIGUOUS = "ambiguous"

#: Default minimum posterior probability for assigning a guide to a cell.
#: Cells below the threshold receive :data:`AMBIGUOUS`.
DEFAULT_THRESHOLD = 0.55

#: The likelihood families. `lognormal` first because it is on the same scale
#: the regression works in, so an effect and a score are commensurate; `beta`
#: kept because a classification score actually lives in [0, 1] and a log does
#: not.
LIKELIHOODS: Tuple[str, ...] = ("lognormal", "beta")


def normalise_fractions(fractions: Mapping[str, float]) -> Dict[str, float]:
    """The guides' fractions, scaled to sum to 1.

    :param fractions: guide names mapped to their measured fraction in one
        well. Missing, non-finite, and non-positive values are discarded.

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
    shifted = math.log(base / (1.0 - base)) + float(effect)
    # `1 / (1 + exp(-shifted))` RAISES OverflowError once shifted drops below
    # about -709, so a guide with a large negative effect killed the whole
    # well instead of being attributed. Saturating there is not an
    # approximation: the clamp on the next line already pins the mean at
    # `eps` for every shift below about -14, so the value returned is
    # identical to what the unclamped expression produced.
    mean = 0.0 if shifted < -700.0 else 1.0 / (1.0 + math.exp(-shifted))
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

    :param scores: one finite classification or measurement score per cell.
    :param priors: normalized guide fractions; insertion order defines the
        returned matrix columns.
    :param effects: fitted score shifts keyed by guide. A missing guide uses a
        zero shift.

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
    """What one cell was attributed, and how sure that is.

    :ivar guide: assigned guide name, or :data:`AMBIGUOUS` when the posterior
        did not reach the call threshold.
    :ivar probability: largest posterior probability reached for the cell.
    :ivar ambiguous: whether ``probability`` was below the call threshold.
    :ivar entropy: Shannon entropy of the cell's guide posterior, in bits.
    :ivar runner_up: second-highest guide name, or an empty string when no
        second guide exists.
    """

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

    :param scores: one classification or measurement score per cell, in the
        output order.
    :param fractions: sequencing fractions keyed by guide for this well. The
        usable positive fractions are normalised before attribution.
    :param effects: fitted score effect keyed by guide; a missing guide is
        treated as having zero effect.
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
                 threshold: float = DEFAULT_THRESHOLD,
                 others: Optional[Sequence[Tuple[float, float]]] = None,
                 span: float = 4.0,
                 centre: float = 0.0,
                 likelihood: str = "lognormal",
                 grid: int = 513) -> Tuple[bool, float]:
    """``(can_it, best_possible)`` -- can this guide ever be called?

    Determine whether a guide can reach the assignment threshold within the
    supplied score range.

    :param effect: the guide's fitted shift on the selected likelihood scale.
    :param scale: the positive spread of scores used to express the plausible
        range and likelihood width; non-positive values resolve to one.
    :param prior: the guide's sequencing fraction in the well, clipped to the
        closed interval from zero to one.
    :param others: the competition, as ``(effect, prior)`` pairs. Omit it and
        the rest of the well is treated as one competitor with no effect,
        which provides the most permissive comparison.
    :param span: how far out, in units of ``scale``, a score may plausibly
        lie. The posterior ceiling is maximized across this finite range.

    The calculation evaluates competing guides at the same candidate score.
    Opposite-signed effects can therefore separate more strongly than a flat
    competitor. Without a finite ``span`` the plain-Bayes posterior may tend
    to one, so no useful finite ceiling would exist.
    """
    p = min(max(float(prior), 0.0), 1.0)
    if p <= 0:
        return False, 0.0
    if p >= 1:
        return True, 1.0
    sigma = float(scale) if scale and scale > 0 else 1.0
    mine = float(effect)
    # EACH WEIGHT IS READ ONCE. Converting in the filter and again in the
    # stored tuple let a weight whose conversion is not pure -- a lazily
    # fetched count, a mutable proxy, a value re-read from a stream -- pass
    # the positivity test and then be stored non-positive. The weight the
    # filter approved would not be the weight used, and the well could be
    # reported "this guide can never be called" on data that looked positive
    # when it was checked. Reading once makes the two agree by construction.
    rest = [pair for pair in
            ((float(e), float(w)) for e, w in (others or ()))
            if pair[1] > 0.0] or [(0.0, 1.0 - p)]
    total = sum(w for _, w in rest)
    if total <= 0:
        return False, 0.0
    rest = [(e, w * (1.0 - p) / total) for e, w in rest]

    # The range of scores a cell could plausibly take: `span` sigmas either
    # side of the centre, widened to cover every component's own centre.
    effects = [mine] + [e for e, _ in rest]
    low = centre + min(effects) - span * sigma
    high = centre + max(effects) + span * sigma
    scores = np.linspace(low, high, max(int(grid), 3))

    numerator = p * _density(likelihood, scores, mine, centre, sigma)
    denominator = numerator.copy()
    for other_effect, weight in rest:
        denominator = denominator + weight * _density(
            likelihood, scores, other_effect, centre, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(numerator, denominator,
                          out=np.zeros_like(numerator),
                          where=denominator > 0)
    best = float(np.nanmax(ratio)) if ratio.size else 0.0
    best = min(max(best, 0.0), 1.0)
    return best >= float(threshold), best


@dataclass(frozen=True)
class Preflight:
    """Whether a guide can produce a real assignment, BEFORE any is made.

    :ivar guide: guide whose achievable posterior was evaluated.
    :ivar wells: number of wells containing a positive fraction of the guide.
    :ivar callable_wells: wells in which the guide can reach ``threshold``.
    :ivar best: highest posterior ceiling reached across the guide's wells.
    :ivar threshold: posterior probability required for a guide call.
    """

    guide: str
    wells: int
    callable_wells: int
    best: float
    threshold: float

    @property
    def hopeless(self) -> bool:
        """True when no well in the screen could ever call this guide."""
        return self.wells > 0 and self.callable_wells == 0

    def note(self) -> str:
        """One line for a caption or a console."""
        if not self.wells:
            return f"{self.guide}: no well carries it."
        wells = f"{self.wells} well" + ("" if self.wells == 1 else "s")
        if self.hopeless:
            return (
                f"{self.guide} CANNOT BE ATTRIBUTED to a cell in any of its "
                f"{wells}: the best posterior it could reach "
                f"anywhere is {self.best:.3f}, under the {self.threshold:.2f} "
                f"threshold. Its effect is too small against the spread of "
                f"scores, which is arithmetic and not sample size -- more "
                f"cells will not change it. Cells shown for it are picked by "
                f"rank, not attributed.")
        return (
            f"{self.guide} can be attributed in {self.callable_wells} of its "
            f"{wells} (best possible posterior {self.best:.3f} "
            f"against a {self.threshold:.2f} threshold).")


def preflight(guide, fractions_by_well, effects, *, scale, centre=0.0,
              threshold=DEFAULT_THRESHOLD, likelihood="lognormal",
              span=4.0) -> Preflight:
    """Can ``guide`` ever be called, and in how many of its wells?

    "WHICH GUIDES CAN BE ATTRIBUTED AT ALL, reported BEFORE anything is
    assigned. A guide whose effect is small against the spread of scores can
    never reach the threshold -- not with more cells, not with a better fit."
    This is that report, and until it had a caller it was a library function
    nobody saw.

    :param guide: the guide being asked about.
    :param fractions_by_well: ``{well: {guide: fraction}}`` -- every guide in
        each well, because a ceiling is a comparison and comparing a guide
        against nothing returns its prior.
    :param effects: each guide's fitted effect.
    :param scale: the spread of the scores, per plate.
    :returns: a :class:`Preflight`.

    A guide is counted callable in a well when :func:`attributable` says so
    THERE -- the answer differs between wells because the prior does, and a
    guide at 0.6 of one well and 0.02 of another is not the same question.
    """
    name = str(guide)
    wells = 0
    callable_wells = 0
    best = 0.0
    for fractions in (fractions_by_well or {}).values():
        priors = normalise_fractions(fractions or {})
        if name not in priors:
            continue
        wells += 1
        can, ceiling = attributable(
            float(effects.get(name, 0.0)), scale, priors[name],
            threshold=threshold,
            others=[(float(effects.get(other, 0.0)), weight)
                    for other, weight in priors.items() if other != name],
            centre=centre, likelihood=likelihood, span=span)
        callable_wells += int(bool(can))
        best = max(best, float(ceiling))
    return Preflight(guide=name, wells=wells, callable_wells=callable_wells,
                     best=best, threshold=float(threshold))


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
    """Guide assignment for each cell with its observed sequencing counts.

    :ivar guides: assigned guide name for each input cell, in input order.
    :ivar cost: total negative log likelihood of the assignment; lower is
        better.
    :ivar degeneracy: mean cost of changing a cell to its best alternative
        guide, in nats; values near zero indicate interchangeable solutions.
    :ivar counts: number of cells assigned to each guide.
    """

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

    :param scores: one classification or measurement score per cell, in the
        desired output order.
    :param fractions: measured guide fractions for the well; usable positive
        values are normalized before integer cell counts are apportioned.
    :param effects: fitted score effect keyed by guide; absent guides use a
        zero effect.

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
    # ONLY THE SHORT-BY-SOME CASE. There was an `elif short < 0` arm
    # undoing an overshoot, marked "rare"; it is not rare, it is
    # impossible. `priors` sums to 1, so `exact` sums to n, and
    # `floor(x) <= x` gives `slots.sum() <= n` -- `short` cannot be
    # negative. Argued, then checked over 30,000 random fraction sets
    # with magnitudes spanning twelve orders and well sizes to 400: the
    # most negative value seen was 0.
    if short > 0:
        order = np.argsort(-(exact - np.floor(exact)))
        for index in order[:short]:
            slots[index] += 1

    density = np.column_stack([
        _density(likelihood, values, float(effects.get(g, 0.0)), centre, scale)
        for g in names])
    density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    cost_per_guide = -np.log(np.clip(density, 1e-300, None))

    # One column per SLOT, so the counts are a property of the matrix rather
    # than something checked afterwards.
    # `slots` sums to exactly n after the correction above, so this has
    # exactly n entries. The truncation that used to follow could not
    # fire for the same reason the removed arm could not: the only way
    # to get more than n columns is `slots.sum() > n`, which requires
    # the negative `short` that cannot happen.
    columns = np.repeat(np.arange(len(names)), slots)
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


# --------------------------------------------------------------------------- #
# Option C -- every measurement, not just the score                            #
# --------------------------------------------------------------------------- #
#
# "best case i can use all the fraction information and all the measurement
# and classefication data to estimate which grna is linked to which cell ...
# eaven if it only holds a timy little bit of information it still might
# work, right?"
#
# Right in principle, and the arithmetic below is what makes the "might"
# honest. Two things have to be got correct or this produces confident
# nonsense.
#
# 1. LOG SPACE. A product of 785 densities underflows to exactly zero in
#    double precision long before it reaches the end, and every cell then
#    looks equally impossible -- which the code above answers by handing back
#    the prior. The bug would present as "option C always says ambiguous".
#
# 2. THE MEASUREMENTS ARE NOT INDEPENDENT, and pretending otherwise is the
#    difference between a method and a fiction. `cell_area` and
#    `cell_perimeter` are one measurement wearing two names; multiplying
#    their likelihoods counts the same evidence twice. Measured on the
#    maintainer's own screen, 785 measurement columns carry an effective
#    dimension in the low tens. So the summed log-likelihood is SCALED by
#    n_eff / n_measured, which is the standard design-effect correction.
#    Without it the posterior saturates at 0 or 1 for every cell and the
#    0.55 threshold becomes decorative.


def effective_dimension(matrix: np.ndarray) -> float:
    """How many INDEPENDENT measurements ``matrix``'s columns amount to.

    The participation ratio of the correlation matrix's eigenvalues,
    ``(sum lambda)^2 / sum lambda^2`` -- 785 for 785 orthogonal columns, and
    1 for 785 copies of one column. The same statistic the sweep uses to
    count a guide's effective wells, applied to the other axis.

    :param matrix: cells x measurements, already finite.
    :returns: the effective count, between 1 and the number of columns.
    """
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        return 0.0
    if values.shape[1] == 1:
        return 1.0
    centred = values - values.mean(axis=0, keepdims=True)
    spread = centred.std(axis=0, ddof=0)
    # A column that does not vary carries no information and must not be
    # allowed to divide by zero; it is dropped rather than kept at scale 1,
    # which would have made it look like an independent measurement.
    alive = spread > 0
    if not alive.any():
        return 0.0
    centred = centred[:, alive] / spread[alive]
    correlation = (centred.T @ centred) / max(centred.shape[0], 1)
    eigenvalues = np.linalg.eigvalsh(correlation)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    squared = float((eigenvalues * eigenvalues).sum())
    if squared <= 0:
        return 0.0
    return float(total * total / squared)


def posterior_multivariate(measurements: np.ndarray,
                           priors: Mapping[str, float],
                           effects: Mapping[str, Sequence[float]], *,
                           centres: Optional[Sequence[float]] = None,
                           scales: Optional[Sequence[float]] = None,
                           likelihood: str = "lognormal",
                           correct_for_correlation: bool = True,
                           iterations: int = 200,
                           tolerance: float = 1e-9
                           ) -> Tuple[np.ndarray, Tuple[str, ...], Dict[str, float]]:
    """:func:`posterior`, but reading EVERY measurement instead of one score.

    Option C. Each guide carries a vector of effects -- one per measurement --
    and a cell's evidence is the summed log-density across them, scaled by
    the design effect (see the note above). The same iterative proportional
    fitting then applies, so the guide masses still match the sequencing.

    :param measurements: cells x measurements.
    :param priors: normalised sequencing fraction keyed by guide. Its key
        order defines the columns in the returned posterior matrix.
    :param effects: ``{guide: [effect per measurement]}``, in the columns'
        order. A guide with no entry is flat, which is the honest prior for
        a guide nothing was fitted for.
    :param correct_for_correlation: scale the evidence by
        ``effective_dimension / n_measurements``. Leave it on unless the
        columns really are independent; see the note above for why.
    :returns: ``(r, guides, diagnostics)``. The diagnostics carry
        ``n_measurements``, ``effective_dimension`` and the ``scale_factor``
        applied, because a reader has to be able to see how much the
        correction did.
    """
    guides = tuple(priors.keys())
    values = np.asarray(measurements, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    n_cells, n_measured = values.shape
    empty = np.zeros((n_cells, len(guides)), dtype=float)
    report = {"n_measurements": float(n_measured),
              "effective_dimension": 0.0, "scale_factor": 1.0}
    if not guides or not n_cells or not n_measured:
        return empty, guides, report

    finite = np.isfinite(values)
    values = np.where(finite, values, 0.0)

    if centres is None:
        centres = values.mean(axis=0)
    if scales is None:
        scales = values.std(axis=0, ddof=0)
    centres = np.asarray(centres, dtype=float).ravel()
    scales = np.asarray(scales, dtype=float).ravel()
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)

    n_eff = effective_dimension(values) if correct_for_correlation else float(n_measured)
    report["effective_dimension"] = float(n_eff)
    factor = 1.0
    if correct_for_correlation and n_measured > 0 and n_eff > 0:
        factor = float(min(n_eff / n_measured, 1.0))
    report["scale_factor"] = factor

    # LOG DENSITY, SUMMED. `_density` is per-measurement, so this is a loop
    # over columns rather than one vectorised call -- 785 columns is nothing
    # beside the per-cell work, and reusing the same densities is what keeps
    # option C's answer commensurable with option A's.
    log_density = np.zeros((n_cells, len(guides)), dtype=float)
    for column, (centre, scale) in enumerate(zip(centres, scales)):
        # A measurement missing for a cell contributes NOTHING for that cell
        # rather than a zero score, which would be a real and usually
        # extreme value.
        present = finite[:, column]
        if not present.any():
            continue
        scores = values[present, column]
        for index, guide in enumerate(guides):
            effect = effects.get(guide)
            size = 0.0 if effect is None else float(
                np.asarray(effect, dtype=float).ravel()[column]
                if np.asarray(effect).ravel().size > column else 0.0)
            density = _density(likelihood, scores, size, float(centre),
                               float(scale))
            density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
            log_density[present, index] += np.log(np.maximum(density, 1e-300))

    log_density *= factor
    # Subtract the per-cell maximum before exponentiating: the shift cancels
    # in the normalisation and is the difference between a usable number and
    # exp(-4000).
    log_density -= log_density.max(axis=1, keepdims=True)
    density = np.exp(log_density)

    weights = np.array([float(priors[g]) for g in guides], dtype=float)
    dead = density.sum(axis=1) <= 0
    if dead.any():
        density[dead, :] = weights

    target = weights * n_cells
    r = density * weights
    for _ in range(int(iterations)):
        rows = r.sum(axis=1, keepdims=True)
        rows[rows <= 0] = 1.0
        r = r / rows
        columns = r.sum(axis=0)
        if np.abs(columns - target).max() <= tolerance * max(n_cells, 1):
            break
        step = np.divide(target, columns, out=np.ones_like(columns),
                         where=columns > 0)
        r = r * step
    rows = r.sum(axis=1, keepdims=True)
    rows[rows <= 0] = 1.0
    return r / rows, guides, report
