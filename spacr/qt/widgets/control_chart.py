"""Control charts across a campaign — the statistics, with the decisions argued.

A screening campaign is dozens of plates run over weeks. The positive and
negative controls are supposed to be *the same thing every time*: that is what
makes hit calling, plate normalisation and Z' mean anything at all. When the
controls stop being the same — the cell line drifts, a reagent lot changes, the
incubator door is left open on a Friday — everything computed downstream is
already wrong, and nothing in the analysis says so, because the analysis
normalises to the controls it is given.

This module tracks one control value per plate over run order and puts limits
round it, so drift is visible **before** it ruins a screen rather than after.
It is pure numpy/pandas — no Qt anywhere in it, the same rule
:mod:`spacr.qt.widgets.graph_spec` and :mod:`spacr.qt.widgets.pca_model`
follow — so it is usable from a notebook, testable without a display, and
free for a report or a QC gate to call.

The decision the whole module turns on
--------------------------------------

**Sigma comes from the average moving range divided by d2 = 1.128
(:data:`D2_MOVING_RANGE`), never from the standard deviation of the series.**

This is not a stylistic preference and it is not a small effect. The standard
deviation of the plate-to-plate series is inflated by *exactly the drift the
chart exists to detect*. Take a control that slides linearly by half a unit per
plate over thirty plates with no other noise at all:

* the SD of those thirty values is 4.40, so ``mean ± 3 SD`` is ``[4.0, 30.5]``
  and every single point is comfortably inside. The chart says "in control"
  while the campaign slides from 10 to 24.5.
* the moving range only ever sees plate-to-plate *change*, which for a slow
  drift is 0.5 and stays 0.5. ``MR-bar / 1.128 = 0.443``, so the limits are
  ``[13.42, 16.08]`` around a baseline centre of 14.75 — and 24 of the 30
  plates fall outside them.

That asymmetry is the reason short-term (within-subgroup) variation is the only
admissible sigma for a control chart, and it is pinned by a test. What the SD
route *would* have said is computed anyway and reported — see
:func:`sd_reference_limits` and :attr:`ControlChartResult.sd_would_flag` —
because the argument is more convincing as two numbers side by side than as a
paragraph.

Three estimators, and which one ran is always named
---------------------------------------------------

:data:`ESTIMATOR_MOVING_RANGE`
    Individuals / moving range (I-MR). One control value per plate is the
    normal case in a screen, so this is what :data:`ESTIMATOR_AUTO` picks when
    every plate contributes a single control well. Sigma is
    ``mean(|x_i - x_{i-1}|) / 1.128``; 1.128 is d2 for a subgroup of two,
    which is what a moving range of consecutive points is.

:data:`ESTIMATOR_SUBGROUP_S`
    X-bar / S, for a plate with several control wells (subgroup ``n > 1``).
    The plotted point is the plate's mean and sigma is ``S-bar / c4(n)``,
    where :func:`c4` is computed from the gamma function rather than looked up
    in a table. The table in the back of a textbook stops at n = 25, does not
    interpolate, and is the sort of thing that gets transcribed with a typo
    into a module nobody re-derives. ``c4(n) = sqrt(2/(n-1)) · Γ(n/2) /
    Γ((n-1)/2)`` is three lines, is exact for every n, and can be checked
    against the published constant — ``c4(5) = 0.9400`` — which is what the
    test does.

    Sigma **of the plotted statistic** is ``sigma_within / sqrt(n)``, and that
    is what the zones and the rules are stated in. When subgroup sizes differ
    between plates the limits differ per plate; they are computed per point
    rather than averaged, and the result says so.

:data:`ESTIMATOR_ROBUST` / :data:`ESTIMATOR_MAD`
    Explicit, never implicit, and named in the result. spaCR measurements are
    skewed and one catastrophically bad plate inside the baseline drags the
    classical centre and MR-bar out far enough to swallow everything after it.
    The robust variant takes the **median** as the centre and either the median
    moving range (:data:`ESTIMATOR_ROBUST`) or the MAD (:data:`ESTIMATOR_MAD`)
    as the spread.

    The two robust constants are *not* interchangeable and the module refuses
    to pretend they are. 1.4826 (:data:`MAD_SCALE`) is ``1/Φ⁻¹(0.75)``, the
    factor that makes the MAD of a normal sample an unbiased estimate of its
    SD. A moving range is ``|x_i - x_{i-1}|``, the absolute value of a
    *difference* of two observations, so its scale is ``sqrt(2)·sigma`` and its
    median is ``sqrt(2)·Φ⁻¹(0.75)·sigma = 0.9539·sigma``
    (:data:`MEDIAN_MR_CONSTANT`). Multiplying a median moving range by 1.4826
    overestimates sigma by exactly ``sqrt(2)`` — 41% wider limits, which is the
    difference between catching a drift and not. Both constants are computed
    from ``Φ⁻¹(0.75)`` here for the same reason c4 is computed from the gamma
    function.

Phase I and Phase II — a parameter, not an afterthought
-------------------------------------------------------

Limits are estimated from a **stated baseline** set of plates and then applied
forward. Computing them from all the data including the drift is the standard
way a control chart is made useless, and it is easy to do by accident because
"just chart everything" is the obvious call.

The baseline is the first :data:`DEFAULT_BASELINE` plates in run order by
default (twenty, the textbook Phase I minimum for an individuals chart), or an
explicit list of plate ids (:attr:`ControlChartSpec.baseline_plates`), or
everything before a cut-off in the order column
(:attr:`ControlChartSpec.baseline_before`). Whichever it was, the plates that
formed it are in :attr:`ControlChartResult.baseline_plates` and named in
:meth:`ControlChartResult.report`.

:data:`MIN_BASELINE` is eight, because MR-bar over seven moving ranges is
already an opinion about one or two plate-to-plate jumps and below that it is
an opinion about one. Fewer than that is refused rather than answered.

**Limits estimated from an out-of-control baseline are not limits.** When the
baseline itself contains a violation the result says so, loudly, in
:meth:`ControlChartResult.caveats`. Re-estimating without the flagged plates is
*offered* (:attr:`ControlChartSpec.reestimate`) and is off by default, because
silently deleting the inconvenient plates is how a baseline gets talked into
agreeing with itself. It runs one pass, not iterations to convergence:
iterating is a ratchet that shrinks the limits onto the tightest subset of the
baseline and eventually flags everything.

The rules, and saying which one fired
--------------------------------------

The Nelson / Western Electric set, all eight, each a named constant with what
it physically detects and its approximate false-alarm rate:
:data:`RULE_BEYOND_3_SIGMA`, :data:`RULE_NINE_ONE_SIDE`,
:data:`RULE_SIX_TRENDING`, :data:`RULE_FOURTEEN_ALTERNATING`,
:data:`RULE_TWO_OF_THREE_BEYOND_2`, :data:`RULE_FOUR_OF_FIVE_BEYOND_1`,
:data:`RULE_FIFTEEN_WITHIN_1`, :data:`RULE_EIGHT_BEYOND_1`.

Which points a rule flags is a convention, and an unstated one makes two
implementations disagree about the same chart, so it is stated: **every point
that is part of the evidence is flagged.** For rule 1 that is the point itself.
For the run rules (2, 3, 4, 7, 8) it is every point of the run, reported as one
maximal span rather than as a sliding window's worth of overlapping alarms —
which is what lets :meth:`ControlChartResult.report` say "plates P22–P30 are
nine in a row above the centre line" instead of listing three violations. For
the "k of m" rules (5, 6) it is the qualifying points only: a window can
trigger on two points beyond 2 sigma with a third sitting on the centre line,
and flagging that third point would be an accusation against a plate that did
nothing.

The rule set is selectable, and it has to be, because **running all eight is
not eight times the sensitivity**. Each rule has its own in-control false-alarm
rate; :data:`RULE_ALARM_RATE` carries them, :meth:`ControlChartResult.caveats`
does the arithmetic for the set that actually ran, and over a 200-plate
campaign the answer is a number of expected false alarms rather than a
reassurance. :data:`RULES_DEFAULT` is the Western Electric four (1, 2, 5, 6),
which is the set the classical ARL figure was worked out for; :data:`RULES_ALL`
is all eight, and :data:`RULES_LIMITS_ONLY` is rule 1 alone for a user who
wants limits and nothing else.

Ordering
--------

"Plate by plate over time" means the x axis is **run order**, and rules 2, 3,
4, 6, 7 and 8 are statements about a sequence — get the order wrong and they
are statements about nothing. An explicit order or date column
(:attr:`ControlChartSpec.order`) is used when given. Without one the plate id
is sorted with a natural, digit-aware key so plate 2 comes before plate 10, and
:attr:`ControlChartResult.order_inferred` is set, the note is in
:attr:`ControlChartResult.notes`, and :meth:`ControlChartResult.report` says it
in capitals. An inferred order is a guess about the experiment, not a detail.

The key is a deliberate extension of
:func:`spacr.qt.widgets.graph_spec._sort_key`, not a copy of it: that one tries
``float(text)`` and falls back to a plain string compare, which puts ``P10``
before ``P2`` — fine for facet labels, wrong for a run order. Here the string
is split into digit and non-digit runs and the digit runs compare as integers.

Z-prime
-------

:func:`zprime_frame` computes the per-plate Z-factor when both a positive and a
negative control are named, and :func:`zprime_chart` charts it with everything
above — same estimator, same rules, same baseline. It is the number a screener
actually watches, and a Z' that slides from 0.7 to 0.3 over a campaign is the
same failure this module exists for, one level up.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import gamma as _gamma_fn, ndtri as _ndtri

from .graph_spec import CATEGORICAL, CONTINUOUS, column_kinds

__all__ = [
    "ControlChartError",
    "D2_MOVING_RANGE", "MEDIAN_MR_CONSTANT", "MAD_SCALE", "c4",
    "ESTIMATOR_AUTO", "ESTIMATOR_MOVING_RANGE", "ESTIMATOR_SUBGROUP_S",
    "ESTIMATOR_ROBUST", "ESTIMATOR_MAD", "ESTIMATORS", "ESTIMATOR_LABELS",
    "RULE_BEYOND_3_SIGMA", "RULE_NINE_ONE_SIDE", "RULE_SIX_TRENDING",
    "RULE_FOURTEEN_ALTERNATING", "RULE_TWO_OF_THREE_BEYOND_2",
    "RULE_FOUR_OF_FIVE_BEYOND_1", "RULE_FIFTEEN_WITHIN_1",
    "RULE_EIGHT_BEYOND_1",
    "RULES_ALL", "RULES_DEFAULT", "RULES_LIMITS_ONLY",
    "RULE_NAMES", "RULE_DETECTS", "RULE_ALARM_RATE", "RULE_WINDOW",
    "DEFAULT_BASELINE", "MIN_BASELINE", "SIGMA_TOLERANCE", "MISSING_PLATE",
    "ZPRIME_VALUE", "ZPRIME_PLATE", "ZPRIME_ORDER",
    "ABOVE", "BELOW", "EITHER",
    "ControlChartSpec", "Violation", "ControlChartResult", "control_chart",
    "sd_reference_limits", "candidate_value_columns", "candidate_key_columns",
    "zprime_frame", "zprime_chart",
]


class ControlChartError(ValueError):
    """A chart that cannot mean anything, with the way out in the message.

    Raised rather than returned as an empty result. Every one of these is a
    sentence a user can act on — "18 plates carry a control value and the
    baseline needs 20" — and a caller that swallowed it would draw an empty
    axis with no explanation. The screen catches it and shows the text.
    """


# ---------------------------------------------------------------------------
# The constants, and where they come from
# ---------------------------------------------------------------------------

#: d2 for a subgroup of two — the unbiasing constant that turns the mean
#: absolute difference of consecutive normal observations into an estimate of
#: their SD. A moving range *is* a subgroup of two, which is why this and not
#: some other d2. Its exact value is ``2/sqrt(pi)`` = 1.12838; 1.128 is the
#: published figure every control-chart text and every other implementation
#: uses, and matching them is worth more here than the fifth digit.
D2_MOVING_RANGE = 1.128

#: ``Φ⁻¹(0.75)`` = 0.67449. The quartile of the standard normal, and the root
#: of both robust constants below. Computed rather than typed.
_NORMAL_QUARTILE = float(_ndtri(0.75))

#: ``sqrt(2)·Φ⁻¹(0.75)`` = 0.95387 — the median of the moving range of a
#: normal series, in units of its sigma. Divide a median moving range by this
#: to get sigma. It is *not* 1/1.4826: see the module docstring.
MEDIAN_MR_CONSTANT = math.sqrt(2.0) * _NORMAL_QUARTILE

#: ``1/Φ⁻¹(0.75)`` = 1.48260 — the factor that makes 1.4826·MAD an unbiased
#: estimate of a normal sample's SD. For a MAD, and only for a MAD.
MAD_SCALE = 1.0 / _NORMAL_QUARTILE

#: Plates in the default Phase I baseline. Twenty is the textbook minimum for
#: estimating individuals-chart limits; fewer and the limits carry the noise of
#: the baseline into every judgement made against them.
DEFAULT_BASELINE = 20

#: The fewest baseline plates this module will estimate limits from. Eight
#: plates is seven moving ranges, and MR-bar over seven values is already an
#: opinion about one or two plate-to-plate jumps. Below it, it is an opinion
#: about one, and a limit built on that is decoration.
MIN_BASELINE = 8

#: Sigma at or below this, relative to the centre's own magnitude, is treated
#: as zero — see :attr:`ControlChartResult.degenerate`. Relative, so a control
#: measured in units of 1e-9 is not called constant for being small.
SIGMA_TOLERANCE = 1e-12

#: A point (or a run) on the high side of the centre line.
ABOVE = "above"
#: On the low side.
BELOW = "below"
#: Either side — rule 8 counts distance from the centre, not direction.
EITHER = "either"


def c4(n: int) -> float:
    """The unbiasing constant for the SD of a subgroup of ``n``.

    ``c4(n) = sqrt(2/(n-1)) · Γ(n/2) / Γ((n-1)/2)``, which is
    ``E[s] / sigma`` for a normal sample of size ``n``: the sample SD is a
    biased estimate of sigma, badly so for the small subgroups a plate's
    control wells make, and ``S-bar / c4(n)`` is the correction.

    Computed from the gamma function rather than looked up, because the
    published table stops at 25, does not interpolate, and is a transcription
    error waiting to happen in a module nobody re-derives. The values agree
    with the published ones to every digit they print: ``c4(2) = 0.7979``,
    ``c4(5) = 0.9400``, ``c4(10) = 0.9727``.

    :raises ControlChartError: for ``n < 2``. A subgroup of one has no spread
        to unbias, which is the whole reason the I-MR chart exists.
    """
    size = int(n)
    if size < 2:
        raise ControlChartError(
            f"c4 is the unbiasing constant for the SD of a subgroup and needs "
            f"at least two observations, not {n}. A plate with one control "
            f"well has no within-plate spread — that is the case the "
            f"individuals / moving-range chart ({ESTIMATOR_MOVING_RANGE!r}) "
            f"exists for.")
    if size > 342:
        # Γ(171) overflows a float; c4 is within 1e-3 of 1 long before then.
        return 1.0 - 0.75 / size
    return math.sqrt(2.0 / (size - 1)) * float(
        _gamma_fn(size / 2.0) / _gamma_fn((size - 1) / 2.0))


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

#: Choose from the subgroup size: I-MR when every plate has one control well,
#: X-bar/S when any has more. The default, and the choice is named in the
#: result rather than left to be inferred from the numbers.
ESTIMATOR_AUTO = "auto"
#: Individuals / moving range. Sigma = MR-bar / d2.
ESTIMATOR_MOVING_RANGE = "moving_range"
#: X-bar / S. Sigma = S-bar / c4(n), divided by sqrt(n) for the plotted mean.
ESTIMATOR_SUBGROUP_S = "subgroup_s"
#: Robust: median centre, median moving range / 0.9539 for sigma.
ESTIMATOR_ROBUST = "robust"
#: Robust: median centre, 1.4826 · MAD for sigma.
ESTIMATOR_MAD = "mad"

ESTIMATORS: Tuple[str, ...] = (
    ESTIMATOR_AUTO, ESTIMATOR_MOVING_RANGE, ESTIMATOR_SUBGROUP_S,
    ESTIMATOR_ROBUST, ESTIMATOR_MAD)

#: What each estimator says in a caption, in the user's terms.
ESTIMATOR_LABELS: Dict[str, str] = {
    ESTIMATOR_AUTO: "chosen from the subgroup size",
    ESTIMATOR_MOVING_RANGE:
        "individuals / moving range — sigma = MR-bar / 1.128",
    ESTIMATOR_SUBGROUP_S:
        "X-bar / S — sigma = S-bar / c4(n), per plate over sqrt(n)",
    ESTIMATOR_ROBUST:
        "robust — median centre, sigma = median(MR) / 0.9539",
    ESTIMATOR_MAD:
        "robust — median centre, sigma = 1.4826 · MAD",
}


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------

#: **Rule 1 — one point beyond 3 sigma.** A spike: one plate that was not like
#: the others. A dispensing failure, a plate read on the wrong gain, a control
#: well that got the wrong compound. It is the only rule that needs no
#: sequence, so it is the only one that survives a wrong run order.
#: False alarm rate 0.0027 per point — one in 370.
RULE_BEYOND_3_SIGMA = 1

#: **Rule 2 — nine points in a row on the same side of the centre line.**
#: A shift, not a spike: the control moved and stayed moved. A new reagent lot,
#: a re-thawed cell line, a recalibrated reader. This is the rule that catches
#: the failure that costs a campaign, because a shift produces no single
#: alarming plate. 0.0039 per point — one in 256.
RULE_NINE_ONE_SIDE = 2

#: **Rule 3 — six points in a row steadily increasing or decreasing.**
#: A trend: something is degrading monotonically. Cells passaged too far,
#: a lamp ageing, a stock solution evaporating. Six *points*, so five
#: consecutive differences in the same direction; a repeated value breaks the
#: run, because equal is neither up nor down. 0.0028 per point.
RULE_SIX_TRENDING = 3

#: **Rule 4 — fourteen points in a row alternating up and down.**
#: Systematic sawtooth. In a screen it is almost always two things being
#: alternated: two readers, two operators, two batches of plates processed on
#: odd and even days. 0.0046 per point.
RULE_FOURTEEN_ALTERNATING = 4

#: **Rule 5 — two of three consecutive beyond 2 sigma on the same side.**
#: A large shift caught early — it fires several plates before rule 2 would,
#: at the cost of firing on chance pairs. 0.0031 per point.
RULE_TWO_OF_THREE_BEYOND_2 = 5

#: **Rule 6 — four of five consecutive beyond 1 sigma on the same side.**
#: A moderate shift, about one sigma, that is too small to break the 3-sigma
#: limit and too gentle to make nine in a row quickly. 0.0055 per point — the
#: noisiest rule in the set.
RULE_FOUR_OF_FIVE_BEYOND_1 = 6

#: **Rule 7 — fifteen points in a row within 1 sigma.**
#: Stratification: the chart is *too good*. Real data does not sit inside one
#: sigma fifteen times running. It nearly always means the sigma is wrong —
#: estimated from a between-plate SD that includes variation the points do not
#: have, or the control column is mislabelled and the "control" is a computed
#: constant. A rule about the analysis rather than about the process.
#: 0.0033 per point.
RULE_FIFTEEN_WITHIN_1 = 7

#: **Rule 8 — eight points in a row all beyond 1 sigma, either side.**
#: A mixture: two populations with no middle. Two plate types, two cell
#: densities, controls from two different stocks alternating. Distinguished
#: from rule 2 by not caring which side. 0.0001 per point — the quietest rule.
RULE_EIGHT_BEYOND_1 = 8

#: Every rule, in Nelson's order.
RULES_ALL: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)

#: The default set: the Western Electric four, with Nelson's nine-in-a-row in
#: place of the original eight. It is the set the classical in-control ARL of
#: 91.75 was computed for, it covers spike / shift / early-shift / moderate-
#: shift, and it leaves out the four rules whose failure modes are rarer in a
#: screening campaign than the false alarms they generate.
RULES_DEFAULT: Tuple[int, ...] = (
    RULE_BEYOND_3_SIGMA, RULE_NINE_ONE_SIDE, RULE_TWO_OF_THREE_BEYOND_2,
    RULE_FOUR_OF_FIVE_BEYOND_1)

#: Rule 1 alone — limits and nothing else, for a user who wants the chart to
#: say only what a single plate did.
RULES_LIMITS_ONLY: Tuple[int, ...] = (RULE_BEYOND_3_SIGMA,)

#: Rule number → the rule in words, for a report a reader has not memorised
#: Nelson's numbering for.
RULE_NAMES: Dict[int, str] = {
    RULE_BEYOND_3_SIGMA: "one point beyond 3 sigma",
    RULE_NINE_ONE_SIDE: "nine points in a row on the same side of the centre",
    RULE_SIX_TRENDING: "six points in a row steadily increasing or decreasing",
    RULE_FOURTEEN_ALTERNATING: "fourteen points in a row alternating up and down",
    RULE_TWO_OF_THREE_BEYOND_2:
        "two of three consecutive beyond 2 sigma on the same side",
    RULE_FOUR_OF_FIVE_BEYOND_1:
        "four of five consecutive beyond 1 sigma on the same side",
    RULE_FIFTEEN_WITHIN_1: "fifteen points in a row within 1 sigma",
    RULE_EIGHT_BEYOND_1: "eight points in a row beyond 1 sigma on either side",
}

#: Rule number → the physical failure it detects, in a screener's terms.
RULE_DETECTS: Dict[int, str] = {
    RULE_BEYOND_3_SIGMA: "a spike — one plate that was not like the others",
    RULE_NINE_ONE_SIDE: "a shift — the control moved and stayed moved",
    RULE_SIX_TRENDING: "a trend — something degrading plate after plate",
    RULE_FOURTEEN_ALTERNATING:
        "a sawtooth — two things being alternated (readers, operators, batches)",
    RULE_TWO_OF_THREE_BEYOND_2: "a large shift, caught early",
    RULE_FOUR_OF_FIVE_BEYOND_1: "a moderate shift, under the 3-sigma limit",
    RULE_FIFTEEN_WITHIN_1:
        "stratification — the chart is too good, so the sigma is probably wrong",
    RULE_EIGHT_BEYOND_1: "a mixture — two populations with no middle",
}

#: Rule number → its approximate probability of firing at any one point of an
#: in-control normal process.
#:
#: Computed under normality and independence: rule 1 is ``2Φ(-3)``, rule 2 is
#: ``2·2⁻⁹``, rule 3 is ``2/6!``, rule 4 is ``2·A₁₄/14!`` from the Euler
#: zigzag number, rules 5 and 6 are binomial tail sums over their windows, and
#: rules 7 and 8 are powers of the 1-sigma probabilities.
#:
#: They are approximations and the direction of the error is known: treating
#: overlapping windows as independent overstates the alarm rate, because in
#: truth alarms cluster and a signal is only counted once. Combining these for
#: the *classical* Western Electric set — 3 sigma, 2 of 3 beyond 2 sigma, 4 of
#: 5 beyond 1 sigma, and eight in a row on one side — gives one alarm per 52.7
#: points where the exact Markov-chain answer (Champ & Woodall 1987) is 91.75.
#: So this arithmetic is a *floor* on the in-control run length, which is the
#: correct direction for a caveat to err in.
RULE_ALARM_RATE: Dict[int, float] = {
    RULE_BEYOND_3_SIGMA: 0.0026998,
    RULE_NINE_ONE_SIDE: 0.0039063,
    RULE_SIX_TRENDING: 0.0027778,
    RULE_FOURTEEN_ALTERNATING: 0.0045736,
    RULE_TWO_OF_THREE_BEYOND_2: 0.0030584,
    RULE_FOUR_OF_FIVE_BEYOND_1: 0.0055318,
    RULE_FIFTEEN_WITHIN_1: 0.0032573,
    RULE_EIGHT_BEYOND_1: 0.0001030,
}

#: Points a rule needs before it can fire at all — the window or run length.
#: A campaign shorter than this cannot trip the rule, which is worth saying
#: rather than leaving a user to conclude the process is in control.
RULE_WINDOW: Dict[int, int] = {
    RULE_BEYOND_3_SIGMA: 1,
    RULE_NINE_ONE_SIDE: 9,
    RULE_SIX_TRENDING: 6,
    RULE_FOURTEEN_ALTERNATING: 14,
    RULE_TWO_OF_THREE_BEYOND_2: 3,
    RULE_FOUR_OF_FIVE_BEYOND_1: 5,
    RULE_FIFTEEN_WITHIN_1: 15,
    RULE_EIGHT_BEYOND_1: 8,
}

#: What the plate id of a row with no plate is called, so it is visible rather
#: than silently dropped. Matches ``graph_spec.MISSING_LEVEL`` on purpose.
MISSING_PLATE = "(missing)"

#: Column names :func:`zprime_frame` writes. Fixed, because
#: :func:`zprime_chart` charts its own output and the two must agree.
ZPRIME_VALUE = "zprime"
ZPRIME_PLATE = "plate"
ZPRIME_ORDER = "order_index"


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

_DIGITS = re.compile(r"(\d+)")


def _natural_key(text: Any) -> Tuple[Tuple[int, int, str], ...]:
    """Digit-aware ordering for a plate id: ``P2`` before ``P10``.

    A deliberate extension of :func:`spacr.qt.widgets.graph_spec._sort_key`
    rather than a copy. That one tries ``float(text)`` and falls back to a
    plain string compare, which is right for a facet label and wrong for a run
    order: ``float("P10")`` raises, so ``P10`` sorts before ``P2`` and every
    run-based rule is then a statement about the wrong sequence. Splitting on
    digit runs and comparing those as integers fixes it, and still orders bare
    numbers, ``plate_3`` and ``2026-03-04`` the way a reader expects.
    """
    parts = _DIGITS.split(str(text))
    return tuple((1, int(part), "") if part.isdigit() else (0, 0, part)
                 for part in parts)


def _order_keys(values: Sequence[Any], series: Optional[pd.Series]
                ) -> Tuple[List[Any], str]:
    """``(sort key per value, kind)`` for the column the x axis is in.

    ``kind`` is ``"datetime"``, ``"numeric"`` or ``"text"``. Detection is by
    what the column *is*, not by what it is called: a datetime64 column sorts
    by its nanoseconds, a column every value of which is numeric sorts as
    numbers, and anything else gets :func:`_natural_key` — which sorts an
    ISO-8601 date string correctly as a side effect, since ISO-8601 was
    designed to.
    """
    if series is not None and pd.api.types.is_datetime64_any_dtype(series):
        stamps = series.to_numpy(dtype="datetime64[ns]").astype("int64")
        return [int(v) for v in stamps], "datetime"
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    if bool(numeric.notna().all()) and len(numeric):
        return [float(v) for v in numeric], "numeric"
    return [_natural_key(v) for v in values], "text"


def _cutoff_key(value: Any, kind: str) -> Any:
    """``value`` in the same currency :func:`_order_keys` produced.

    The baseline cut-off has to be comparable with the order keys or the
    comparison is a ``TypeError`` at the worst possible moment, so it is
    converted by the *detected* kind rather than by its own type.
    """
    if kind == "datetime":
        return int(pd.Timestamp(value).value)
    if kind == "numeric":
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ControlChartError(
                f"the order column is numeric, so the baseline cut-off has to "
                f"be a number; {value!r} is not one") from exc
    return _natural_key(value)


# ---------------------------------------------------------------------------
# Column offers
# ---------------------------------------------------------------------------

def candidate_value_columns(frame: pd.DataFrame) -> Tuple[str, ...]:
    """The columns worth offering as the charted measurement, sorted.

    Continuous by :func:`spacr.qt.widgets.graph_spec.column_kinds`, which is
    the one column classifier in this codebase — the Local Data Filter's rule,
    re-read. Reused rather than re-derived: the Graph Builder, the PCA screen
    and this one must agree about what ``cell_count`` is, or the same table
    presents three different mental models depending on which screen is open.
    """
    return tuple(sorted(name for name, kind in column_kinds(frame).items()
                        if kind == CONTINUOUS))


def candidate_key_columns(frame: pd.DataFrame) -> Tuple[str, ...]:
    """The columns worth offering as the plate id or the control label.

    Categorical by the same classifier, plus anything named like a plate or a
    date, because ``plateID`` is high-cardinality enough in a big campaign that
    the classifier calls it a key and skips it — and a key is exactly what is
    wanted here. This is the one place in spaCR where "identifies rather than
    describes" is a recommendation rather than a disqualification.
    """
    kinds = column_kinds(frame)
    wanted = {name for name, kind in kinds.items() if kind == CATEGORICAL}
    for name in frame.columns:
        lowered = str(name).lower()
        if any(token in lowered for token in
               ("plate", "date", "time", "batch", "run", "order")):
            wanted.add(str(name))
    return tuple(sorted(wanted))


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ControlChartSpec:
    """What to chart, in what order, from which baseline, under which rules.

    Frozen and JSON round-tripping, like
    :class:`~spacr.qt.widgets.graph_spec.GraphSpec` and
    :class:`~spacr.qt.widgets.pca_model.PCASpec`, so a QC configuration is
    something a settings file or a report can carry and re-run against the next
    campaign — which is the whole point of a chart whose limits were estimated
    once.

    :param value: the measured column to chart. Required at chart time.
    :param plate: the column identifying the plate. Required at chart time.
        One point per distinct value of it.
    :param order: an explicit run-order or date column. Strongly preferred:
        without it the order is inferred from ``plate`` and every run-based
        rule rests on that inference.
    :param control_column: the column saying what each row is
        (``"condition"``, ``"gene"``, ``"well_type"``). ``None`` means the
        table is already only the control.
    :param control_levels: which level(s) of ``control_column`` are the
        control being charted. Required whenever ``control_column`` is set —
        a control column with no level named is a filter with no predicate.
    :param positive_levels: the positive control, for :func:`zprime_frame`.
    :param negative_levels: the negative control, likewise.
    :param estimator: one of :data:`ESTIMATORS`.
    :param rules: which of :data:`RULES_ALL` to run.
    :param baseline_n: Phase I length when neither ``baseline_plates`` nor
        ``baseline_before`` is given.
    :param baseline_plates: an explicit Phase I set, by plate id.
    :param baseline_before: everything strictly before this value of the order
        column is Phase I. A date, for a campaign whose baseline is "the first
        week".
    :param reestimate: when the baseline itself trips a rule, drop the flagged
        plates and estimate once more. Off by default; see the module
        docstring for why silently deleting the inconvenient plates is not the
        default behaviour of anything here.
    :raises ControlChartError: on an unknown estimator or rule id, a baseline
        shorter than :data:`MIN_BASELINE`, or a control column with no level
        named — at the point the spec is built, not at render time.
    """

    value: str = ""
    plate: str = ""
    order: Optional[str] = None
    control_column: Optional[str] = None
    control_levels: Tuple[str, ...] = ()
    positive_levels: Tuple[str, ...] = ()
    negative_levels: Tuple[str, ...] = ()
    estimator: str = ESTIMATOR_AUTO
    rules: Tuple[int, ...] = RULES_DEFAULT
    baseline_n: int = DEFAULT_BASELINE
    baseline_plates: Tuple[str, ...] = ()
    baseline_before: Optional[str] = None
    reestimate: bool = False

    def __post_init__(self) -> None:
        for name in ("value", "plate"):
            object.__setattr__(self, name, str(getattr(self, name) or ""))
        for name in ("order", "control_column", "baseline_before"):
            raw = getattr(self, name)
            object.__setattr__(self, name, str(raw) if raw else None)
        for name in ("control_levels", "positive_levels", "negative_levels",
                     "baseline_plates"):
            values = getattr(self, name) or ()
            seen: Dict[str, None] = {}
            for item in values:
                seen.setdefault(str(item), None)
            object.__setattr__(self, name, tuple(seen))

        if self.estimator not in ESTIMATORS:
            raise ControlChartError(
                f"unknown estimator {self.estimator!r}; choose one of "
                f"{', '.join(ESTIMATORS)}. {ESTIMATOR_AUTO!r} picks "
                f"{ESTIMATOR_MOVING_RANGE!r} for one control well per plate "
                f"and {ESTIMATOR_SUBGROUP_S!r} for several.")

        rules: List[int] = []
        for rule in self.rules or ():
            try:
                number = int(rule)
            except (TypeError, ValueError) as exc:
                raise ControlChartError(
                    f"rule {rule!r} is not a rule number; the rules are "
                    f"{', '.join(str(r) for r in RULES_ALL)}") from exc
            if number not in RULE_NAMES:
                raise ControlChartError(
                    f"there is no rule {number}; the rules are 1-8 — "
                    + "; ".join(f"{r}: {RULE_NAMES[r]}" for r in RULES_ALL))
            if number not in rules:
                rules.append(number)
        object.__setattr__(self, "rules", tuple(sorted(rules)))

        count = int(self.baseline_n)
        if count < MIN_BASELINE:
            raise ControlChartError(
                f"a baseline of {count} plate(s) is not a baseline; "
                f"{MIN_BASELINE} is the fewest this module will estimate "
                f"limits from, because MR-bar over {MIN_BASELINE - 1} moving "
                f"ranges is already an opinion about one or two plates.")
        object.__setattr__(self, "baseline_n", count)

        if self.control_column and not self.control_levels:
            raise ControlChartError(
                f"control_column={self.control_column!r} says where to look "
                f"for the control but not which level it is. Name the "
                f"level(s) in control_levels, or leave control_column empty "
                f"if the table is already only the control.")
        if self.control_levels and not self.control_column:
            raise ControlChartError(
                f"control_levels={list(self.control_levels)} names a control "
                f"but no column to find it in; set control_column.")
        object.__setattr__(self, "reestimate", bool(self.reestimate))

    # -- edits -----------------------------------------------------------
    def with_columns(self, *, value: Optional[str] = None,
                     plate: Optional[str] = None,
                     order: Optional[str] = None) -> "ControlChartSpec":
        """A copy pointed at different columns."""
        return replace(self,
                       value=self.value if value is None else value,
                       plate=self.plate if plate is None else plate,
                       order=self.order if order is None else order)

    def with_control(self, column: Optional[str],
                     levels: Sequence[str]) -> "ControlChartSpec":
        """A copy charting a different control."""
        return replace(self, control_column=column, control_levels=tuple(levels))

    def with_rules(self, rules: Sequence[int]) -> "ControlChartSpec":
        """A copy running a different rule set."""
        return replace(self, rules=tuple(rules))

    def with_estimator(self, estimator: str) -> "ControlChartSpec":
        """A copy using a different sigma estimator."""
        return replace(self, estimator=estimator)

    def with_baseline(self, *, n: Optional[int] = None,
                      plates: Optional[Sequence[str]] = None,
                      before: Optional[str] = None) -> "ControlChartSpec":
        """A copy with a different Phase I."""
        return replace(
            self,
            baseline_n=self.baseline_n if n is None else n,
            baseline_plates=(self.baseline_plates if plates is None
                             else tuple(plates)),
            baseline_before=(self.baseline_before if before is None
                             else before))

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """A plain JSON-able dict. Every field, always — a stable schema beats
        a compact one for something a QC configuration is stored as."""
        return {
            "value": self.value,
            "plate": self.plate,
            "order": self.order,
            "control_column": self.control_column,
            "control_levels": list(self.control_levels),
            "positive_levels": list(self.positive_levels),
            "negative_levels": list(self.negative_levels),
            "estimator": self.estimator,
            "rules": list(self.rules),
            "baseline_n": self.baseline_n,
            "baseline_plates": list(self.baseline_plates),
            "baseline_before": self.baseline_before,
            "reestimate": self.reestimate,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControlChartSpec":
        """Rebuild from :meth:`to_dict`.

        Unknown keys ignored, missing keys defaulted, so a configuration
        written by another build still opens: a QC chart nobody can reload is
        a QC chart nobody re-runs.
        """
        fields = {"value", "plate", "order", "control_column",
                  "control_levels", "positive_levels", "negative_levels",
                  "estimator", "rules", "baseline_n", "baseline_plates",
                  "baseline_before", "reestimate"}
        known = {k: v for k, v in dict(payload).items() if k in fields}
        for key in ("control_levels", "positive_levels", "negative_levels",
                    "baseline_plates", "rules"):
            if key in known:
                known[key] = tuple(known[key] or ())
        return cls(**known)

    def to_json(self) -> str:
        """:meth:`to_dict` as sorted JSON."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ControlChartSpec":
        """Rebuild from :meth:`to_json`."""
        return cls.from_dict(json.loads(text))

    def describe(self) -> str:
        """One line, for a caption."""
        control = (f"{self.control_column}={'/'.join(self.control_levels)}"
                   if self.control_column else "every row")
        order = (f"ordered by {self.order}" if self.order
                 else "order inferred from the plate id")
        baseline = (f"baseline: {len(self.baseline_plates)} named plate(s)"
                    if self.baseline_plates
                    else f"baseline: before {self.baseline_before}"
                    if self.baseline_before
                    else f"baseline: first {self.baseline_n}")
        return (f"{self.value or '(no column)'} of {control} · {order} · "
                f"{baseline} · rules "
                f"{','.join(str(r) for r in self.rules) or 'none'}")


# ---------------------------------------------------------------------------
# One violation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Violation:
    """One rule firing, over one span of plates.

    A *maximal* span, not a sliding window's worth of overlapping alarms:
    twelve points in a row above the centre line is one violation of rule 2
    covering twelve plates, which is what a reader needs, rather than four
    violations covering nine plates each, which is an artefact of the
    implementation.

    :param rule: the rule number, 1-8.
    :param start: first index of the span, into
        :attr:`ControlChartResult.plates`.
    :param end: last index of the span, inclusive.
    :param points: the indices actually flagged. Equal to the whole span for
        the run rules; only the qualifying points for rules 5 and 6, where a
        window can trigger with an innocent point inside it.
    :param plates: the plate ids of :attr:`points`.
    :param side: :data:`ABOVE`, :data:`BELOW` or :data:`EITHER`. For rule 3
        it is the *direction of travel* (:data:`ABOVE` for a rising trend)
        rather than a side of the centre line, because a trend can run
        entirely on one side of the centre or cross it and is a violation
        either way.
    :param in_baseline: whether any flagged point is inside Phase I. Limits
        estimated from an out-of-control baseline are not limits, so this is
        carried per violation rather than recomputed by every reader.
    """

    rule: int
    start: int
    end: int
    points: Tuple[int, ...]
    plates: Tuple[str, ...]
    side: str = EITHER
    in_baseline: bool = False

    @property
    def span(self) -> int:
        """Plates from :attr:`start` to :attr:`end`, inclusive."""
        return self.end - self.start + 1

    def where(self) -> str:
        """The plates in words — ``"plate P17"`` or ``"plates P22-P30"``."""
        if not self.plates:  # pragma: no cover - a violation always has points
            return "no plates"
        if len(self.plates) == 1:
            return f"plate {self.plates[0]}"
        if self.span == len(self.plates):
            return f"plates {self.plates[0]}–{self.plates[-1]}"
        return "plates " + ", ".join(self.plates)

    def describe(self) -> str:
        """The sentence :meth:`ControlChartResult.report` prints.

        Names the rule by number *and* in words, because a report that says
        "rule 6" to a reader who has not memorised Nelson's numbering has said
        nothing, and one that says only "four of five beyond 1 sigma" cannot be
        cross-referenced against anyone else's chart.
        """
        side = ("above the centre line" if self.side == ABOVE
                else "below the centre line" if self.side == BELOW
                else "on either side of the centre line")
        detail = {
            RULE_BEYOND_3_SIGMA:
                f"{self.where()} is beyond the "
                f"{'upper' if self.side == ABOVE else 'lower'} limit",
            RULE_NINE_ONE_SIDE:
                f"{self.where()} are {self.span} in a row {side} — "
                f"a shift, not a spike",
            RULE_SIX_TRENDING:
                f"{self.where()} are {self.span} in a row moving steadily one "
                f"way — something is degrading plate after plate",
            RULE_FOURTEEN_ALTERNATING:
                f"{self.where()} alternate up and down {self.span} times "
                f"running — two things are being alternated",
            RULE_TWO_OF_THREE_BEYOND_2:
                f"two of three consecutive plates land beyond 2 sigma "
                f"{side} — {self.where()}, {len(self.points)} of them",
            RULE_FOUR_OF_FIVE_BEYOND_1:
                f"four of five consecutive plates land beyond 1 sigma "
                f"{side} — {self.where()}, {len(self.points)} of them",
            RULE_FIFTEEN_WITHIN_1:
                f"{self.where()} are {self.span} in a row inside 1 sigma "
                f"— too good; the sigma is probably wrong",
            RULE_EIGHT_BEYOND_1:
                f"{self.where()} are {self.span} in a row beyond 1 sigma on "
                f"either side — a mixture of two populations",
        }[self.rule]
        baseline = " (inside the baseline)" if self.in_baseline else ""
        return f"rule {self.rule} — {RULE_NAMES[self.rule]}: {detail}{baseline}."


# ---------------------------------------------------------------------------
# The reference nobody should use as limits
# ---------------------------------------------------------------------------

def sd_reference_limits(values: Sequence[float]
                        ) -> Tuple[float, float, float, float]:
    """``(centre, sigma, lower, upper)`` the *wrong* way — mean ± 3 SD.

    This is not an estimator option and never will be. It exists so a result
    can print what the SD route would have said next to what the moving-range
    route did say, because the central argument of this module is far more
    convincing as two intervals side by side than as a paragraph: on a drifting
    series the SD limits are wide enough to contain the drift, and a chart drawn
    with them reports "in control" all the way down.

    Uses the sample SD (``ddof=1``) over every value given, which is exactly
    the mistake being illustrated — limits computed from all the data,
    including the excursion they are supposed to detect.
    """
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size < 2:
        return (float(finite[0]) if finite.size else 0.0), 0.0, 0.0, 0.0
    centre = float(finite.mean())
    sigma = float(finite.std(ddof=1))
    return centre, sigma, centre - 3.0 * sigma, centre + 3.0 * sigma


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ControlChartResult:
    """A chart, its limits, what fired, and everything needed to read it.

    :param plates: one entry per point, in run order. The x axis.
    :param values: the plotted statistic per plate — the control value, or the
        plate's mean control value when there is more than one control well.
    :param order_labels: what the order column said for each plate, as text.
        Equal to :attr:`plates` when the order was inferred.
    :param subgroup_sizes: control wells behind each point.
    :param subgroup_sd: within-plate SD per point; NaN where ``n < 2``.
    :param centre: the centre line, estimated from Phase I only.
    :param sigma: sigma of the *plotted statistic*, which is what the zones and
        every rule are stated in. For X-bar/S that is
        ``sigma_within / sqrt(n)``; when subgroup sizes differ it is the median
        of :attr:`sigma_at` and the per-point values are the truth.
    :param sigma_within: the within-plate sigma, before the ``sqrt(n)``. Equal
        to :attr:`sigma` for an individuals chart.
    :param sigma_at: sigma per point.
    :param lower: per-point lower limit (centre - 3 sigma).
    :param upper: per-point upper limit.
    :param z: ``(value - centre) / sigma_at`` per point — the currency every
        rule is written in, so that varying subgroup sizes cost the rules
        nothing.
    :param estimator: which estimator actually ran. Never ``"auto"``.
    :param baseline: positional indices of the Phase I plates.
    :param baseline_excluded: plates dropped from Phase I by a re-estimation.
    :param violations: every rule firing, in run order then rule order.
    :param rules: the rule set that was run.
    :param order_inferred: the order was guessed from the plate id.
    :param degenerate: sigma came out zero — see :meth:`caveats`.
    :param sd_reference: ``(centre, sigma, lower, upper)`` from
        :func:`sd_reference_limits` over every point.
    :param sd_would_flag: how many points fall outside those SD limits.
    """

    plates: Tuple[str, ...]
    values: np.ndarray
    order_labels: Tuple[str, ...]
    subgroup_sizes: np.ndarray
    subgroup_sd: np.ndarray
    centre: float
    sigma: float
    sigma_within: float
    sigma_at: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    z: np.ndarray
    estimator: str
    baseline: np.ndarray
    rules: Tuple[int, ...]
    value_column: str
    plate_column: str
    order_column: Optional[str] = None
    control_column: Optional[str] = None
    control_levels: Tuple[str, ...] = ()
    baseline_excluded: Tuple[str, ...] = ()
    violations: Tuple[Violation, ...] = ()
    order_inferred: bool = False
    degenerate: bool = False
    sd_reference: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    sd_would_flag: int = 0
    n_rows_in: int = 0
    n_control_rows: int = 0
    notes: Tuple[str, ...] = ()

    # -- shape -----------------------------------------------------------
    def __len__(self) -> int:
        """Plates on the chart."""
        return len(self.plates)

    @property
    def baseline_plates(self) -> Tuple[str, ...]:
        """The plate ids Phase I was estimated from, in run order."""
        return tuple(self.plates[i] for i in self.baseline)

    @property
    def flagged(self) -> np.ndarray:
        """Boolean per point: did any rule fire on it."""
        out = np.zeros(len(self), dtype=bool)
        for violation in self.violations:
            for index in violation.points:
                out[index] = True
        return out

    def rules_at(self, index: int) -> Tuple[int, ...]:
        """Which rules fired on point ``index``, ascending.

        A point can trip several — a plate beyond 3 sigma is usually also the
        end of a two-of-three-beyond-2-sigma window — and reporting only the
        first would hide that the same plate is evidence for two different
        failures.
        """
        if not 0 <= int(index) < len(self):
            raise ControlChartError(
                f"there is no point {index}; this chart has {len(self)} "
                f"plate(s)")
        return tuple(sorted({v.rule for v in self.violations
                             if int(index) in v.points}))

    def zone(self, index: int) -> int:
        """Which sigma band point ``index`` sits in: 0, 1, 2 or 3.

        0 is inside 1 sigma, 3 is beyond 3 sigma. Signed nowhere — the side is
        the sign of ``z``. What a renderer colours a marker by.
        """
        if not 0 <= int(index) < len(self):
            raise ControlChartError(
                f"there is no point {index}; this chart has {len(self)} "
                f"plate(s)")
        magnitude = abs(float(self.z[int(index)]))
        if not np.isfinite(magnitude):  # pragma: no cover - guarded upstream
            return 0
        return min(3, int(magnitude))

    # -- frames ----------------------------------------------------------
    def points_frame(self) -> pd.DataFrame:
        """One row per plate — what the chart draws, and its CSV.

        Everything a reader would want to re-plot elsewhere: the order label,
        the value, the subgroup, the limits that applied to *that* point, the
        z, whether it was Phase I and which rules fired on it.
        """
        return pd.DataFrame({
            "plate": list(self.plates),
            "order": list(self.order_labels),
            "value": self.values,
            "n": self.subgroup_sizes,
            "subgroup_sd": self.subgroup_sd,
            "centre": np.full(len(self), self.centre),
            "lower": self.lower,
            "upper": self.upper,
            "sigma": self.sigma_at,
            "z": self.z,
            "in_baseline": np.isin(np.arange(len(self)), self.baseline),
            "flagged": self.flagged,
            "rules": [",".join(str(r) for r in self.rules_at(i))
                      for i in range(len(self))],
        })

    def violations_frame(self) -> pd.DataFrame:
        """One row per violation — the table the screen shows."""
        return pd.DataFrame({
            "rule": [v.rule for v in self.violations],
            "name": [RULE_NAMES[v.rule] for v in self.violations],
            "detects": [RULE_DETECTS[v.rule] for v in self.violations],
            "plates": [", ".join(v.plates) for v in self.violations],
            "first_plate": [v.plates[0] if v.plates else ""
                            for v in self.violations],
            "n_points": [len(v.points) for v in self.violations],
            "side": [v.side for v in self.violations],
            "in_baseline": [v.in_baseline for v in self.violations],
            "description": [v.describe() for v in self.violations],
        })

    # -- saying it in words -----------------------------------------------
    @property
    def baseline_violations(self) -> Tuple[Violation, ...]:
        """The violations that fell inside Phase I."""
        return tuple(v for v in self.violations if v.in_baseline)

    def headline(self) -> str:
        """One sentence about the campaign, including the bad news."""
        if not len(self):  # pragma: no cover - refused before a result exists
            return "no plates."
        if self.degenerate:
            return (f"every one of the {len(self)} plates reported the same "
                    f"{self.value_column} ({self.centre:.6g}), so there is no "
                    f"variation to put limits around and no rule was run.")
        if not self.violations:
            return (f"{len(self)} plates, all in control against limits "
                    f"[{float(self.lower.min()):.6g}, "
                    f"{float(self.upper.max()):.6g}] "
                    f"estimated from {len(self.baseline)} baseline plates.")
        flagged = int(self.flagged.sum())
        first = min(v.start for v in self.violations)
        kinds = sorted({v.rule for v in self.violations})
        return (f"{len(self.violations)} rule violation(s) over {flagged} of "
                f"{len(self)} plates, first at {self.plates[first]}; "
                f"rule(s) {', '.join(str(k) for k in kinds)} fired.")

    def false_alarm_rate(self) -> float:
        """Approximate probability that *some* selected rule fires on an
        in-control point.

        ``1 - Π(1 - p_i)`` over the rules that ran. The independence it assumes
        is not true and the error has a known direction — see
        :data:`RULE_ALARM_RATE` — so this overstates the alarm rate, which is
        the safe direction for a warning.
        """
        survive = 1.0
        for rule in self.rules:
            survive *= (1.0 - RULE_ALARM_RATE.get(rule, 0.0))
        return 1.0 - survive

    def caveats(self) -> Tuple[str, ...]:
        """Everything a reader needs before believing the chart."""
        out: List[str] = []
        if self.degenerate:
            out.append(
                f"Sigma came out zero: every plate reported "
                f"{self.centre:.6g}. The limits collapse onto the centre line, "
                f"which would put all {len(self)} plates 'beyond' them, so no "
                f"rule was run at all — flagging a whole campaign because a "
                f"column does not vary is not a QC finding. Either the "
                f"control genuinely never moved (a copied number, a value "
                f"rounded to death, an integer readout) or the wrong column "
                f"is charted.")
        if self.order_inferred:
            out.append(
                f"The run order was INFERRED by sorting the plate id "
                f"({self.plate_column}) with a digit-aware key, because no "
                f"order column was given. Rules 2, 3, 4, 6, 7 and 8 are "
                f"statements about a sequence, so if the plates were not run "
                f"in that order those rules are statements about nothing. "
                f"Point this chart at a date or run-order column.")
        if len(self) and self.baseline.size == len(self):
            out.append(
                f"Every one of the {len(self)} plates is in the baseline, so "
                f"the limits were estimated from the same points they are "
                f"judging. That is a description of the campaign, not a test "
                f"of it: a plate cannot be out of limits that it helped set. "
                f"Set a baseline (baseline_n, baseline_plates or "
                f"baseline_before) covering the part of the campaign you "
                f"believe was in control.")
        if self.baseline_violations:
            names = ", ".join(f"rule {v.rule} at {v.where()}"
                              for v in self.baseline_violations[:4])
            out.append(
                f"The baseline itself is out of control "
                f"({len(self.baseline_violations)} violation(s): {names}). "
                f"Limits estimated from an out-of-control baseline are not "
                f"limits — they are the spread of a process that was already "
                f"misbehaving, widened to make the misbehaviour look normal. "
                f"Re-estimate without those plates (reestimate=True), or pick "
                f"an explicit baseline that excludes them.")
        if self.baseline_excluded:
            out.append(
                f"{len(self.baseline_excluded)} plate(s) were dropped from "
                f"the baseline and the limits re-estimated without them: "
                f"{', '.join(self.baseline_excluded)}. One pass only, "
                f"deliberately: iterating to convergence shrinks the limits "
                f"onto the best-behaved subset of the baseline and eventually "
                f"flags everything.")
        rate = self.false_alarm_rate()
        if self.rules and rate > 0:
            every = 1.0 / rate
            expected = rate * len(self)
            out.append(
                f"{len(self.rules)} rule(s) on {len(self)} plates: each rule "
                f"has its own false-alarm rate on a process that is behaving, "
                f"and running several multiplies the alarms rather than the "
                f"evidence. Together they fire on about one in {every:.0f} "
                f"in-control points, so expect roughly {expected:.1f} false "
                f"alarm(s) across this campaign even if nothing is wrong. "
                f"The arithmetic treats the rules as independent, which they "
                f"are not, so it is a floor: on the classical Western "
                f"Electric set it gives one alarm per 52.7 points where the "
                f"exact answer is 91.75.")
        short = [r for r in self.rules if RULE_WINDOW[r] > len(self)]
        if short:
            out.append(
                f"Rule(s) {', '.join(str(r) for r in short)} need more points "
                f"than this campaign has ({len(self)}), so they cannot fire. "
                f"Their silence is not evidence of control.")
        if self.sd_would_flag < int(self.flagged.sum()) and not self.degenerate:
            centre, sigma, low, high = self.sd_reference
            out.append(
                f"For comparison: limits taken from the standard deviation of "
                f"the whole series ({centre:.6g} ± 3×{sigma:.6g}) "
                f"would be [{low:.6g}, {high:.6g}] and would put "
                f"{self.sd_would_flag} point(s) outside, against "
                f"{int(self.flagged.sum())} flagged here. That is the failure "
                f"this chart is built to avoid: the series SD is inflated by "
                f"the very excursion it is meant to detect, so SD-based "
                f"limits widen to swallow it.")
        varied = np.unique(self.subgroup_sizes)
        if varied.size > 1:
            out.append(
                f"Subgroup sizes vary between plates "
                f"({int(varied.min())}–{int(varied.max())} control "
                f"wells), so the limits are not one pair of lines: a plate "
                f"with fewer wells has a noisier mean and wider limits. They "
                f"are computed per plate rather than averaged.")
        return tuple(out)

    def report(self) -> str:
        """The whole story, as the screen shows it and a report file writes it."""
        control = (f"{self.control_column}="
                   f"{'/'.join(self.control_levels)}"
                   if self.control_column else "every row")
        lines = [
            f"Control chart of {self.value_column} for {control} — "
            f"{len(self)} plate(s), {ESTIMATOR_LABELS[self.estimator]}.",
            f"Centre {self.centre:.6g}; sigma {self.sigma:.6g}; limits "
            f"[{float(self.lower.min()):.6g}, {float(self.upper.max()):.6g}].",
        ]
        baseline = self.baseline_plates
        if baseline:
            span = (f"{baseline[0]}–{baseline[-1]}" if len(baseline) > 1
                    else baseline[0])
            lines.append(
                f"Baseline (Phase I): {len(baseline)} plate(s), {span}. "
                f"Limits estimated from those and applied forward.")
        if self.order_column:
            lines.append(f"Run order: the {self.order_column} column.")
        else:
            lines.append(
                f"Run order: INFERRED from {self.plate_column} — rules 2, "
                f"3, 4, 6, 7 and 8 are statements about a sequence.")
        lines.append("")
        lines.append("  " + self.headline())
        if self.violations:
            lines.append("")
            for violation in self.violations:
                lines.append("  " + violation.describe())
        caveats = self.caveats()
        if caveats:
            lines.append("")
            lines.extend("  ! " + c for c in caveats)
        if self.notes:
            lines.append("")
            lines.extend("  · " + n for n in self.notes)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The rule detectors
# ---------------------------------------------------------------------------

def _maximal_runs(mask: np.ndarray, length: int) -> List[Tuple[int, int]]:
    """``(start, end)`` of every maximal run of ``True`` at least ``length`` long.

    Maximal, so a run of twelve is one violation covering twelve plates rather
    than four overlapping ones covering nine each — the difference between a
    report a reader can act on and a report they have to de-duplicate.
    """
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= length:
                runs.append((start, i - 1))
            start = None
    if start is not None and len(mask) - start >= length:
        runs.append((start, len(mask) - 1))
    return runs


def _rule_1(z: np.ndarray) -> List[Tuple[int, int, Tuple[int, ...], str]]:
    """One point beyond 3 sigma. Strictly beyond: a point *on* the limit is
    not outside it, and the alternative is an alarm that depends on rounding."""
    out = []
    for i in np.flatnonzero(np.abs(z) > 3.0):
        out.append((int(i), int(i), (int(i),), ABOVE if z[i] > 0 else BELOW))
    return out


def _rule_run_side(z: np.ndarray, length: int
                   ) -> List[Tuple[int, int, Tuple[int, ...], str]]:
    """Runs of ``length`` consecutive points on one side of the centre.

    A point exactly on the centre line (``z == 0``) is on neither side and
    breaks the run. It has to: counting it for whichever side is convenient
    would let a perfectly centred process trip a shift rule.
    """
    out = []
    for side, mask in ((ABOVE, z > 0), (BELOW, z < 0)):
        for start, end in _maximal_runs(mask, length):
            out.append((start, end, tuple(range(start, end + 1)), side))
    return out


def _rule_3(z: np.ndarray, points: int
            ) -> List[Tuple[int, int, Tuple[int, ...], str]]:
    """``points`` in a row steadily increasing or decreasing.

    ``points`` points is ``points - 1`` differences. An exactly repeated value
    breaks the trend, because equal is neither up nor down — the alternative
    treats a flat stretch as a trend in whichever direction the run started.
    """
    if z.size < points:
        return []
    diff = np.diff(z)
    out = []
    for side, mask in ((ABOVE, diff > 0), (BELOW, diff < 0)):
        for start, end in _maximal_runs(mask, points - 1):
            out.append((start, end + 1, tuple(range(start, end + 2)), side))
    return out


def _rule_4(z: np.ndarray, points: int
            ) -> List[Tuple[int, int, Tuple[int, ...], str]]:
    """``points`` in a row alternating up and down.

    A difference of exactly zero breaks the alternation for the same reason it
    breaks a trend.
    """
    if z.size < points:
        return []
    diff = np.diff(z)
    turns = np.zeros(max(diff.size - 1, 0), dtype=bool)
    for i in range(turns.size):
        turns[i] = (diff[i] > 0 > diff[i + 1]) or (diff[i] < 0 < diff[i + 1])
    out = []
    for start, end in _maximal_runs(turns, points - 2):
        out.append((start, end + 2, tuple(range(start, end + 3)), EITHER))
    return out


def _rule_k_of_m(z: np.ndarray, k: int, m: int, threshold: float
                 ) -> List[Tuple[int, int, Tuple[int, ...], str]]:
    """``k`` of ``m`` consecutive beyond ``threshold`` sigma on one side.

    Only the qualifying points are flagged, not the whole window: a window can
    trigger on two points beyond 2 sigma with a third sitting on the centre
    line, and flagging that third one is an accusation against a plate that did
    nothing.

    Windows that overlap are merged into one violation, so a long excursion is
    reported once with every offending plate in it rather than once per
    position of the window.
    """
    if z.size < m:
        return []
    out = []
    for side, qualifying in ((ABOVE, z > threshold), (BELOW, z < -threshold)):
        windows: List[Tuple[int, int]] = []
        for start in range(z.size - m + 1):
            if int(qualifying[start:start + m].sum()) >= k:
                windows.append((start, start + m - 1))
        if not windows:
            continue
        merged: List[List[int]] = []
        for start, end in windows:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        for start, end in merged:
            points = tuple(int(i) for i in
                           np.flatnonzero(qualifying[start:end + 1]) + start)
            out.append((int(points[0]), int(points[-1]), points, side))
    return out


def _rule_7(z: np.ndarray, length: int
            ) -> List[Tuple[int, int, Tuple[int, ...], str]]:
    """``length`` in a row within 1 sigma — the stratification signal."""
    out = []
    for start, end in _maximal_runs(np.abs(z) < 1.0, length):
        out.append((start, end, tuple(range(start, end + 1)), EITHER))
    return out


def _rule_8(z: np.ndarray, length: int
            ) -> List[Tuple[int, int, Tuple[int, ...], str]]:
    """``length`` in a row beyond 1 sigma, either side — the mixture signal."""
    out = []
    for start, end in _maximal_runs(np.abs(z) > 1.0, length):
        out.append((start, end, tuple(range(start, end + 1)), EITHER))
    return out


def _detect(z: np.ndarray, rules: Sequence[int], plates: Sequence[str],
            baseline: np.ndarray) -> Tuple[Violation, ...]:
    """Run the selected rules and return the violations in run order."""
    detectors = {
        RULE_BEYOND_3_SIGMA: lambda: _rule_1(z),
        RULE_NINE_ONE_SIDE: lambda: _rule_run_side(z, 9),
        RULE_SIX_TRENDING: lambda: _rule_3(z, 6),
        RULE_FOURTEEN_ALTERNATING: lambda: _rule_4(z, 14),
        RULE_TWO_OF_THREE_BEYOND_2: lambda: _rule_k_of_m(z, 2, 3, 2.0),
        RULE_FOUR_OF_FIVE_BEYOND_1: lambda: _rule_k_of_m(z, 4, 5, 1.0),
        RULE_FIFTEEN_WITHIN_1: lambda: _rule_7(z, 15),
        RULE_EIGHT_BEYOND_1: lambda: _rule_8(z, 8),
    }
    inside = set(int(i) for i in baseline)
    found: List[Violation] = []
    for rule in rules:
        for start, end, points, side in detectors[rule]():
            found.append(Violation(
                rule=rule, start=int(start), end=int(end),
                points=tuple(int(p) for p in points),
                plates=tuple(plates[p] for p in points),
                side=side,
                in_baseline=any(p in inside for p in points)))
    found.sort(key=lambda v: (v.start, v.rule))
    return tuple(found)


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def _moving_ranges(values: np.ndarray) -> np.ndarray:
    """``|x_i - x_{i-1}|`` over consecutive points. The whole point of the
    module: this sees plate-to-plate change and nothing else, so a slow drift
    never inflates it."""
    return np.abs(np.diff(values)) if values.size >= 2 else np.zeros(0)


def _estimate(values: np.ndarray, sizes: np.ndarray, sds: np.ndarray,
              estimator: str, notes: List[str]) -> Tuple[float, float]:
    """``(centre, sigma_within)`` from the baseline points under ``estimator``."""
    if estimator == ESTIMATOR_MOVING_RANGE:
        ranges = _moving_ranges(values)
        return float(values.mean()), float(ranges.mean()) / D2_MOVING_RANGE

    if estimator == ESTIMATOR_SUBGROUP_S:
        usable = np.isfinite(sds) & (sizes >= 2)
        if not usable.any():
            raise ControlChartError(
                f"the X-bar/S estimator ({ESTIMATOR_SUBGROUP_S!r}) needs at "
                f"least one baseline plate with two or more control wells, "
                f"and every baseline plate here has one. Use "
                f"{ESTIMATOR_MOVING_RANGE!r} (or {ESTIMATOR_AUTO!r}, which "
                f"would have picked it).")
        if not usable.all():
            notes.append(
                f"{int((~usable).sum())} baseline plate(s) have a single "
                f"control well and contribute no within-plate SD; S-bar is "
                f"the mean over the {int(usable.sum())} that do")
        s_bar = float(sds[usable].mean())
        sizes_used = sizes[usable]
        typical = int(round(float(sizes_used.mean())))
        if np.unique(sizes_used).size > 1:
            notes.append(
                f"baseline subgroup sizes differ "
                f"({int(sizes_used.min())}–{int(sizes_used.max())}); "
                f"S-bar is unbiased with c4(n) at the mean size {typical}, "
                f"and each plate's own n sets its own limits")
        return float(values.mean()), s_bar / c4(max(typical, 2))

    if estimator == ESTIMATOR_ROBUST:
        ranges = _moving_ranges(values)
        return (float(np.median(values)),
                float(np.median(ranges)) / MEDIAN_MR_CONSTANT)

    if estimator == ESTIMATOR_MAD:
        centre = float(np.median(values))
        mad = float(np.median(np.abs(values - centre)))
        return centre, MAD_SCALE * mad

    raise ControlChartError(  # pragma: no cover - the spec validates first
        f"unknown estimator {estimator!r}")


def _resolve_estimator(estimator: str, sizes: np.ndarray,
                       notes: List[str]) -> str:
    """Turn :data:`ESTIMATOR_AUTO` into the estimator that will actually run."""
    if estimator != ESTIMATOR_AUTO:
        return estimator
    if sizes.size and int(sizes.max()) > 1:
        notes.append(
            f"estimator chosen automatically: plates carry up to "
            f"{int(sizes.max())} control wells, so this is an X-bar/S chart "
            f"and sigma comes from S-bar / c4(n)")
        return ESTIMATOR_SUBGROUP_S
    notes.append(
        "estimator chosen automatically: one control value per plate, so this "
        "is an individuals / moving-range chart and sigma comes from "
        "MR-bar / 1.128")
    return ESTIMATOR_MOVING_RANGE


def _baseline_indices(spec: ControlChartSpec, plates: Sequence[str],
                      keys: Sequence[Any], kind: str,
                      notes: List[str]) -> np.ndarray:
    """Which points form Phase I, as positional indices into the ordered series."""
    total = len(plates)
    if spec.baseline_plates:
        wanted = set(spec.baseline_plates)
        picked = [i for i, plate in enumerate(plates) if plate in wanted]
        missing = wanted - {plates[i] for i in picked}
        if missing:
            notes.append(
                f"{len(missing)} named baseline plate(s) are not in this "
                f"table and were ignored: {', '.join(sorted(missing))}")
        if not picked:
            raise ControlChartError(
                f"none of the {len(wanted)} plate(s) named as the baseline "
                f"are in this table. The plates here are "
                f"{', '.join(plates[:6])}"
                + (" ..." if total > 6 else "") + ".")
        return np.asarray(picked, dtype=int)

    if spec.baseline_before is not None:
        cutoff = _cutoff_key(spec.baseline_before, kind)
        picked = [i for i, key in enumerate(keys) if key < cutoff]
        if not picked:
            raise ControlChartError(
                f"no plate falls before the baseline cut-off "
                f"{spec.baseline_before!r}; the earliest is "
                f"{plates[0]!r}. Phase I would be empty.")
        return np.asarray(picked, dtype=int)

    count = min(spec.baseline_n, total)
    if count < spec.baseline_n:
        notes.append(
            f"the baseline asked for {spec.baseline_n} plates and the "
            f"campaign has {total}")
    return np.arange(count, dtype=int)


# ---------------------------------------------------------------------------
# The computation
# ---------------------------------------------------------------------------

def _control_rows(frame: pd.DataFrame, spec: ControlChartSpec
                  ) -> Tuple[pd.DataFrame, int]:
    """The rows this chart is about, and how many rows were given."""
    n_rows_in = int(len(frame))
    for label, column in (("value", spec.value), ("plate", spec.plate)):
        if not column:
            raise ControlChartError(
                f"no {label} column chosen. A control chart needs the column "
                f"holding the measurement and the column holding the plate.")
        if column not in frame.columns:
            raise ControlChartError(
                f"the {label} column {column!r} is not in this table; it has "
                f"{len(frame.columns)} columns and none of them is that one.")
    if spec.order and spec.order not in frame.columns:
        raise ControlChartError(
            f"the order column {spec.order!r} is not in this table. Leave it "
            f"empty to sort by the plate id instead, which is a guess the "
            f"result will say it made.")
    if spec.control_column and spec.control_column not in frame.columns:
        raise ControlChartError(
            f"the control column {spec.control_column!r} is not in this table.")

    rows = frame
    if spec.control_column:
        labels = frame[spec.control_column].astype(str)
        keep = labels.isin(list(spec.control_levels)).to_numpy()
        rows = frame.iloc[np.flatnonzero(keep)]
        if not len(rows):
            present = sorted({str(v) for v in
                              frame[spec.control_column].unique()})[:8]
            raise ControlChartError(
                f"no row has {spec.control_column} in "
                f"{list(spec.control_levels)}. That column holds "
                f"{', '.join(present)}"
                + (" ..." if len(present) == 8 else "") + ".")
    return rows, n_rows_in


def _plate_points(rows: pd.DataFrame, spec: ControlChartSpec,
                  notes: List[str]
                  ) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray,
                             List[Any], List[str], str, bool]:
    """Collapse control rows to one point per plate, in run order.

    Grouping is done with a plain dict over the string plate ids rather than
    ``DataFrame.groupby``: groupby on a categorical plate column changes its
    behaviour between pandas versions (the ``observed`` default), and this is
    two loops over a few thousand rows.
    """
    values = pd.to_numeric(rows[spec.value], errors="coerce").to_numpy(float)
    plate_labels = rows[spec.plate].astype(str).to_numpy()
    missing_plate = rows[spec.plate].isna().to_numpy()
    plate_labels = np.where(missing_plate, MISSING_PLATE, plate_labels)

    order_series = rows[spec.order] if spec.order else None
    order_raw = (order_series.astype(str).to_numpy() if order_series is not None
                 else plate_labels)

    groups: Dict[str, List[int]] = {}
    for position, plate in enumerate(plate_labels):
        groups.setdefault(str(plate), []).append(position)

    plates: List[str] = []
    means: List[float] = []
    sizes: List[int] = []
    sds: List[float] = []
    labels: List[str] = []
    order_source: List[Any] = []
    empty: List[str] = []
    for plate, positions in groups.items():
        finite = np.asarray([values[p] for p in positions], dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            empty.append(plate)
            continue
        plates.append(plate)
        means.append(float(finite.mean()))
        sizes.append(int(finite.size))
        sds.append(float(finite.std(ddof=1)) if finite.size >= 2 else np.nan)
        labels.append(str(order_raw[positions[0]]))
        if order_series is not None:
            order_source.append(order_series.iloc[positions[0]])
        else:
            order_source.append(plate)
    if empty:
        notes.append(
            f"{len(empty)} plate(s) had no finite {spec.value} for the "
            f"control and are not on the chart: {', '.join(sorted(empty)[:6])}")

    if not plates:
        raise ControlChartError(
            f"no plate has a finite {spec.value!r} for the control. Either "
            f"that column is empty for these rows or it is not numeric.")

    keys, kind = _order_keys(
        order_source,
        pd.Series(order_source) if order_series is not None else None)
    inferred = order_series is None
    if inferred:
        keys = [_natural_key(plate) for plate in plates]
        kind = "text"

    position = sorted(range(len(plates)), key=lambda i: (keys[i], plates[i]))
    return ([plates[i] for i in position],
            np.asarray([means[i] for i in position], dtype=float),
            np.asarray([sizes[i] for i in position], dtype=float),
            np.asarray([sds[i] for i in position], dtype=float),
            [keys[i] for i in position],
            [labels[i] for i in position],
            kind, inferred)


def control_chart(frame: pd.DataFrame,
                  spec: Optional[ControlChartSpec] = None
                  ) -> ControlChartResult:
    """Chart ``spec``'s control value plate by plate over run order.

    The whole policy is in the module docstring; the short version is that one
    point is drawn per plate in run order, sigma comes from short-term
    variation and never from the SD of the series, the limits are estimated
    from a stated baseline and applied forward, and every rule that fires is
    reported by number, in words, and with the plates it fired on.

    :raises ControlChartError: whenever the chart would be meaningless — with
        the reason and the way out in the message.
    """
    spec = spec or ControlChartSpec()
    notes: List[str] = []
    rows, n_rows_in = _control_rows(frame, spec)
    (plates, values, sizes, sds, keys, labels, kind,
     inferred) = _plate_points(rows, spec, notes)

    total = len(plates)
    if total < MIN_BASELINE:
        raise ControlChartError(
            f"{total} plate(s) carry a control value and the smallest "
            f"defensible baseline is {MIN_BASELINE}. Limits estimated from "
            f"fewer are an opinion about one or two plate-to-plate jumps, not "
            f"a limit. Chart this when the campaign is longer.")

    baseline = _baseline_indices(spec, plates, keys, kind, notes)
    if baseline.size < MIN_BASELINE:
        raise ControlChartError(
            f"the baseline selects {baseline.size} of {total} plate(s) and "
            f"{MIN_BASELINE} is the minimum. Widen it — baseline_n, an "
            f"explicit baseline_plates list, or a later baseline_before "
            f"cut-off.")

    if baseline.size == total:
        # Not a refusal — a chart of a campaign that is still short is worth
        # drawing. But Phase I and Phase II being the same plates means the
        # limits and the points they judge are the same data, and a rule
        # firing there is a statement about the estimate rather than a test
        # of anything, so it is said rather than left to be noticed.
        notes.append(
            f"all {total} plates are Phase I, so the limits and the points "
            f"they judge are the same data — this describes the campaign "
            f"rather than testing it")

    estimator = _resolve_estimator(spec.estimator, sizes, notes)
    centre, sigma_within = _estimate(
        values[baseline], sizes[baseline], sds[baseline], estimator, notes)

    def _finish(centre: float, sigma_within: float, baseline: np.ndarray,
                excluded: Tuple[str, ...]) -> ControlChartResult:
        degenerate = sigma_within <= max(abs(centre), 1.0) * SIGMA_TOLERANCE
        if degenerate:
            sigma_at = np.zeros(total)
            z = np.zeros(total)
            lower = np.full(total, centre)
            upper = np.full(total, centre)
            violations: Tuple[Violation, ...] = ()
            notes.append(
                "no rule was run: with sigma zero every point is nominally "
                "beyond the limits, and flagging the whole campaign would be "
                "a statement about the arithmetic rather than the process")
        else:
            if estimator == ESTIMATOR_SUBGROUP_S:
                sigma_at = sigma_within / np.sqrt(np.maximum(sizes, 1.0))
            else:
                sigma_at = np.full(total, sigma_within)
            z = (values - centre) / sigma_at
            lower = centre - 3.0 * sigma_at
            upper = centre + 3.0 * sigma_at
            violations = _detect(z, spec.rules, plates, baseline)

        sd_reference = sd_reference_limits(values)
        _, _, sd_low, sd_high = sd_reference
        sd_flag = int(((values < sd_low) | (values > sd_high)).sum())

        return ControlChartResult(
            plates=tuple(plates), values=values, order_labels=tuple(labels),
            subgroup_sizes=sizes, subgroup_sd=sds,
            centre=float(centre),
            sigma=float(np.median(sigma_at)) if not degenerate else 0.0,
            sigma_within=float(sigma_within),
            sigma_at=sigma_at, lower=lower, upper=upper, z=z,
            estimator=estimator, baseline=baseline, rules=tuple(spec.rules),
            value_column=spec.value, plate_column=spec.plate,
            order_column=spec.order, control_column=spec.control_column,
            control_levels=tuple(spec.control_levels),
            baseline_excluded=excluded, violations=violations,
            order_inferred=inferred, degenerate=degenerate,
            sd_reference=sd_reference, sd_would_flag=sd_flag,
            n_rows_in=n_rows_in, n_control_rows=int(len(rows)),
            notes=tuple(notes))

    result = _finish(centre, sigma_within, baseline, ())
    if not spec.reestimate or not result.baseline_violations:
        return result

    # One pass, not iterations to convergence — see the module docstring.
    offending = {p for v in result.baseline_violations for p in v.points}
    kept = np.asarray([i for i in baseline if i not in offending], dtype=int)
    if kept.size < MIN_BASELINE:
        notes.append(
            f"re-estimation was asked for but dropping the "
            f"{len(offending)} flagged baseline plate(s) would leave "
            f"{kept.size}, under the {MIN_BASELINE}-plate minimum, so the "
            f"limits are the original ones and the baseline is still out of "
            f"control")
        return _finish(centre, sigma_within, baseline, ())
    excluded = tuple(plates[i] for i in sorted(offending))
    notes.append(
        f"limits re-estimated after dropping {len(excluded)} out-of-control "
        f"baseline plate(s): {', '.join(excluded)}")
    centre, sigma_within = _estimate(
        values[kept], sizes[kept], sds[kept], estimator, notes)
    return _finish(centre, sigma_within, kept, excluded)


# ---------------------------------------------------------------------------
# Z-prime
# ---------------------------------------------------------------------------

def zprime_frame(frame: pd.DataFrame, spec: ControlChartSpec) -> pd.DataFrame:
    """Per-plate Z-factor, one row per plate, in run order.

    ``Z' = 1 - 3(sd_pos + sd_neg) / |mean_pos - mean_neg|`` — the number a
    screener actually watches, and the one that says whether a plate could have
    detected anything at all. Charting it with the same limits and the same
    rules is the same question one level up: a Z' that slides from 0.7 to 0.3
    over a campaign is a screen that stopped working, plate by plate, with no
    single plate looking wrong.

    Both controls need at least two wells on a plate for the SD to exist, so a
    plate with a single positive well produces no Z' and is left out rather
    than given a zero.

    :returns: columns ``plate``, ``order``, ``order_index``, ``zprime``,
        ``separation``, and the per-control means, SDs and n.
    :raises ControlChartError: when the spec does not name both controls.
    """
    if not spec.positive_levels or not spec.negative_levels:
        raise ControlChartError(
            "Z' needs both controls named: positive_levels and "
            "negative_levels on the spec. With one control there is a value "
            "to chart but no assay window to compute.")
    if not spec.control_column:
        raise ControlChartError(
            "Z' needs control_column set, so the positive and negative levels "
            "have a column to be levels of.")

    probe = replace(spec, control_levels=tuple(spec.positive_levels)
                    + tuple(spec.negative_levels))
    rows, _ = _control_rows(frame, probe)

    values = pd.to_numeric(rows[spec.value], errors="coerce").to_numpy(float)
    plate_labels = rows[spec.plate].astype(str).to_numpy()
    labels = rows[spec.control_column].astype(str).to_numpy()
    order_series = rows[spec.order] if spec.order else None
    order_raw = (order_series.astype(str).to_numpy() if order_series is not None
                 else plate_labels)

    groups: Dict[str, List[int]] = {}
    for position, plate in enumerate(plate_labels):
        groups.setdefault(str(plate), []).append(position)

    positive = set(spec.positive_levels)
    negative = set(spec.negative_levels)
    records: List[Dict[str, Any]] = []
    order_source: List[Any] = []
    for plate, positions in groups.items():
        pos = np.asarray([values[p] for p in positions
                          if labels[p] in positive], dtype=float)
        neg = np.asarray([values[p] for p in positions
                          if labels[p] in negative], dtype=float)
        pos = pos[np.isfinite(pos)]
        neg = neg[np.isfinite(neg)]
        if pos.size < 2 or neg.size < 2:
            continue
        separation = abs(float(pos.mean()) - float(neg.mean()))
        spread = float(pos.std(ddof=1)) + float(neg.std(ddof=1))
        zprime = 1.0 - 3.0 * spread / separation if separation > 0 else -np.inf
        records.append({
            ZPRIME_PLATE: plate,
            "order": str(order_raw[positions[0]]),
            ZPRIME_VALUE: zprime,
            "separation": separation,
            "mean_positive": float(pos.mean()),
            "mean_negative": float(neg.mean()),
            "sd_positive": float(pos.std(ddof=1)),
            "sd_negative": float(neg.std(ddof=1)),
            "n_positive": int(pos.size),
            "n_negative": int(neg.size),
        })
        order_source.append(
            order_series.iloc[positions[0]] if order_series is not None
            else plate)

    if not records:
        raise ControlChartError(
            "no plate carries at least two positive and two negative control "
            "wells with a finite value, so no plate has a Z'. A single well "
            "per control has no SD, and Z' is a statement about spread.")

    keys, _kind = _order_keys(
        order_source,
        pd.Series(order_source) if order_series is not None else None)
    if order_series is None:
        keys = [_natural_key(r[ZPRIME_PLATE]) for r in records]
    position = sorted(range(len(records)),
                      key=lambda i: (keys[i], records[i][ZPRIME_PLATE]))
    ordered = [records[i] for i in position]
    for index, record in enumerate(ordered):
        # The run order is already resolved, so the chart of this frame must
        # not have to guess it again: a plain 0..k-1 index is the one order
        # column that cannot be mis-sorted.
        record[ZPRIME_ORDER] = index
    return pd.DataFrame(ordered)


def zprime_chart(frame: pd.DataFrame,
                 spec: ControlChartSpec) -> ControlChartResult:
    """:func:`zprime_frame`, charted with the same estimator, rules and baseline.

    A thin composition rather than a second engine: the Z' series is one value
    per plate, which is an individuals chart, which is what
    :func:`control_chart` already is. The order is carried as an explicit
    integer column, so the Z' chart never reports an inferred order — the
    ordering decision was taken once, upstream.
    """
    series = zprime_frame(frame, spec)
    return control_chart(series, replace(
        spec, value=ZPRIME_VALUE, plate=ZPRIME_PLATE, order=ZPRIME_ORDER,
        control_column=None, control_levels=(),
        positive_levels=(), negative_levels=()))
