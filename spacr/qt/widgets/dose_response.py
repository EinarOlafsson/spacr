"""Dose–response — a four-parameter logistic that will refuse to name an EC50.

spaCR runs concentration series and has, until this module, had no curve
fitter at all: a screener with eight doses and three replicates exports a CSV
and opens Prism. The point of writing one is not that "it fits" — every
optimiser fits — but that **the answer is often "you cannot tell from this
experiment", and no fitter in common use says so.** ``curve_fit`` will return
four numbers and a covariance matrix for a dilution series that never reached
a plateau, for a bell-shaped cytotoxicity curve, and for eight points of pure
noise. All three come back looking like an EC50 with a confidence interval.

So the decisions here are almost all about refusing, and each one is written
down.

Why log10(EC50) is the parameter, and EC50 is derived
-----------------------------------------------------
The model is fitted in ``log10_ec50``, never in ``EC50``:

.. math::

    y = \\mathrm{bottom} + \\frac{\\mathrm{top} - \\mathrm{bottom}}
        {1 + 10^{(\\log_{10}\\mathrm{EC}_{50} - \\log_{10}x)\\,h}}

Three reasons, all of which show up in real data:

* **The doses are geometric.** A 2- or 3-fold dilution series is evenly
  spaced in log concentration and violently uneven in linear concentration.
  The information the experiment carries about the midpoint is information
  about *which dilution step* it sits on, which is a statement in log space.
* **The likelihood surface is close to symmetric in log10(EC50) and badly
  skewed in EC50.** That is what makes a quadratic (Wald) approximation
  defensible on ``log10_ec50`` and indefensible on ``EC50``.
* **A linear-space interval routinely reaches below zero**, and a negative
  concentration is not a thing. An EC50 of ``1.0 ± 1.4 µM`` is not a wide
  interval, it is a broken one.

Reporting therefore back-transforms the *interval*: ``10 ** [lo, hi]``. The
result is multiplicative (``0.62 – 1.61 µM`` around 1.0, a 1.6-fold factor
either way), always positive, and reads the way a potency actually behaves.

Which way the curve goes
------------------------
The parameterisation has an exact symmetry: ``(bottom, top, L, h)`` and
``(top, bottom, L, -h)`` are the *same function*, so the optimiser may return
either. Left alone, two runs on the same data can report Hill slopes of
opposite sign. Every fit is therefore canonicalised the way pharmacology
already writes it (and GraphPad's variable-slope equations do):

* ``top >= bottom`` always — they are the larger and smaller plateau, not the
  right-hand and left-hand one;
* **the sign of the Hill slope carries the direction**: ``hill < 0`` is
  inhibition (response falls as dose rises, and ``top`` is the low-dose
  control plateau), ``hill > 0`` is activation.

The direction is *inferred* from the data (Spearman rho of response against
log dose) and never assumed, so an activation series is not fitted upside
down. :data:`DIRECTION_INHIBITION` / :data:`DIRECTION_ACTIVATION` pin it when
the user knows better than the correlation does.

Two confidence intervals, and why the default is the slow one
-------------------------------------------------------------
Both are offered on ``log10_ec50`` and both back-transform:

* :data:`CI_WALD` — asymptotic, from the covariance matrix ``curve_fit``
  returns, with a **t quantile on ``n - 4`` degrees of freedom**. Not a
  normal quantile: eight concentrations with no replicates leaves 4 df, where
  ``t = 2.776`` against ``z = 1.960``. That is a 42% difference in the width
  of the published interval, which is not cosmetic.
* :data:`CI_PROFILE` — profile likelihood, **the default**. For a grid of
  candidate ``log10_ec50`` values the other three parameters are re-optimised
  and the residual sum of squares is compared against
  ``SSE_min * (1 + t²/(n - 4))``, the standard F-based profile region for one
  parameter out of four. (``F(1, ν) = t(ν)²``, so the two intervals use
  literally the same quantile and differ only in the shape of the surface
  they walk.)

The profile is the default for one reason that outweighs its cost: **the Wald
interval is finite by construction.** It is ``L ± t·SE``, so it returns a
tidy, symmetric, entirely fictional interval for a curve whose midpoint the
data does not locate at all — which is exactly the case this module exists
to catch. The profile interval can *fail to close*: the residual sum of
squares stays under the threshold all the way out to a concentration ten
thousand times the highest dose tested, and that failure is the diagnostic.
It is reported as an open bound, not smoothed into a number.

A residual bootstrap was the other candidate and was rejected. It shares the
profile's homoscedastic-normal assumption so it buys no robustness; it needs
a seed and gives a slightly different published number per seed; and — the
deciding objection — percentiles of a bootstrap distribution are *always*
finite, so it launders an unidentified parameter into a confident interval in
precisely the situation that matters. Nothing here resamples and nothing here
is random: the same data gives the same interval, every time.

``bottom``, ``top`` and the Hill slope get Wald intervals only. Profiling
each of them would quadruple the work to improve numbers nobody quotes; the
EC50 is the number that leaves the building.

The three things that actually go wrong
---------------------------------------
1. **The curve is incomplete** — one plateau was never reached, or the
   midpoint sits outside the tested range. Then the EC50 is an extrapolation.
   :attr:`DoseResponseResult.ec50` is ``None`` in that case and the number
   lives in :attr:`DoseResponseResult.ec50_unconstrained`, under a name that
   cannot be mistaken for a result; :attr:`DoseResponseResult.ec50_bounded`
   is ``False`` and :meth:`DoseResponseResult.bound_statement` gives the
   one-sided fact the experiment does support ("EC50 > 30 µM, the highest
   concentration tested"). Three independent detectors have to agree it is
   fine before a number is released: the fitted midpoint inside the tested
   dose range, the observed responses bracketing the fitted half-maximum, and
   neither fitted plateau more than :data:`PLATEAU_SLACK` of the observed
   response span outside the observed responses.
2. **The data is not monotone.** A bell shape — the classic being
   cytotoxicity killing the signal at the top dose — is not a 4PL, and a 4PL
   fitted to it returns a confident EC50 for a curve of the wrong shape.
   Detected from the concentration-ordered per-dose medians and **refused**,
   with the concentrations where it turns named in the message. See
   :func:`monotonicity`.
3. **There is not enough experiment.** Four parameters need at least four
   distinct concentrations (:data:`MIN_DOSES`) and, to say anything about
   uncertainty, more observations than parameters (:data:`MIN_OBSERVATIONS`).
   A constant response has no curve in it. A **zero concentration is not an
   error** — a vehicle control is normal and belongs in the file — so it is
   excluded from the fit deliberately, counted, and reported as a reference
   response, never fed to ``log10``. A *negative* concentration has no such
   reading and is refused.

Fit quality, and the reason R² is printed with a warning attached
-----------------------------------------------------------------
Every result carries the residual standard error, R², and — when the design
has replicates — a **lack-of-fit F test against pure error**. That last one
is the statistic that answers the question people think R² answers.

R² on a sigmoid is nearly useless. Any monotone curve through a well-sampled
dose–response scores above 0.95, because the total sum of squares is
dominated by the difference between the two plateaus and *any* S-shaped line
captures that. :meth:`DoseResponseResult.caveats` says so next to the number,
every time.

The lack-of-fit test does the real work when replicates exist: pure error
(within-concentration scatter, ``n - m`` df) is a model-free estimate of
noise, and the residual variance in excess of it (``m - 4`` df) is
model-misspecification. A small p-value means a 4PL is the wrong shape for
this data whatever the R² says. With no replicates there is no pure-error
estimate and the test does not exist — which is itself reported, because
"cannot be tested" and "passed" are different states.

No Qt in here
-------------
numpy, pandas and scipy only, like :mod:`spacr.qt.widgets.pca_model` and
:mod:`spacr.selection`: usable from a notebook, testable without a display,
and with nothing in the fitting path that knows a widget exists. There is not
a single ``PySide6`` import in this file, and the two seams that *would* need
one — :func:`candidate_concentration_columns` and
:func:`candidate_response_columns`, which re-use the Local Data Filter's
column classifier through :func:`spacr.qt.widgets.graph_spec.column_kinds`
rather than inventing a second one — do it inside the function body, so
importing this module and fitting a curve pulls in no Qt of its own.

(``spacr/qt/widgets/__init__.py`` eagerly imports the widget modules, so
*reaching* this module through the package still costs a PySide6 import
today. That is a property of the package's ``__init__``, not of this file:
nothing here would have to change for the fitter to run in an environment
without PySide6.)
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit, minimize_scalar

__all__ = [
    "DoseResponseError",
    "CI_PROFILE", "CI_WALD", "CI_METHODS",
    "DIRECTION_AUTO", "DIRECTION_INHIBITION", "DIRECTION_ACTIVATION",
    "DIRECTIONS",
    "BOUND_OK", "BOUND_ABOVE", "BOUND_BELOW", "BOUND_OPEN",
    "STATUS_FITTED", "STATUS_UNBOUNDED", "STATUS_REFUSED",
    "MIN_DOSES", "MIN_OBSERVATIONS", "DEFAULT_CONFIDENCE",
    "MAX_REVERSAL", "FLAT_FRACTION", "PLATEAU_SLACK",
    "STEEP_HILL", "SHALLOW_HILL", "PROFILE_REACH", "PROFILE_TOLERANCE",
    "four_parameter_logistic",
    "MonotonicityCheck", "monotonicity",
    "DoseResponseSpec", "DoseResponseResult",
    "GroupFit", "DoseResponseSet",
    "fit_dose_response", "fit_frame",
    "candidate_concentration_columns", "candidate_response_columns",
]


class DoseResponseError(ValueError):
    """A dose–response that cannot mean anything, with the way out in the text.

    Raised rather than returned as an empty result, for the same reason
    :class:`spacr.qt.widgets.pca_model.PCAError` is: every one of these is a
    sentence a screener can act on — "the response turns around between 10
    and 30 µM, which is usually cytotoxicity; drop the top dose" — and a
    caller that swallowed it would draw an empty axis with no explanation.

    :func:`fit_frame` catches it per group and keeps the message beside that
    group's row, so one bad compound does not take the plate down.
    """


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

#: Profile-likelihood interval on ``log10_ec50``. The default; it is the one
#: that can decline to close. See the module docstring.
CI_PROFILE = "profile"
#: Asymptotic (Wald) interval from the covariance matrix, ``L ± t·SE`` with
#: ``t`` on ``n - 4`` df. Symmetric in log space, finite always — including
#: when the data does not determine the parameter.
CI_WALD = "wald"
CI_METHODS: Tuple[str, ...] = (CI_PROFILE, CI_WALD)

#: Infer the direction from the data. The default.
DIRECTION_AUTO = "auto"
#: Response falls as concentration rises — ``hill < 0``.
DIRECTION_INHIBITION = "inhibition"
#: Response rises as concentration rises — ``hill > 0``.
DIRECTION_ACTIVATION = "activation"
DIRECTIONS: Tuple[str, ...] = (DIRECTION_AUTO, DIRECTION_INHIBITION,
                               DIRECTION_ACTIVATION)

#: The EC50 is determined by the experiment: quote it.
BOUND_OK = "bounded"
#: The EC50 is above every concentration tested. One-sided statement only.
BOUND_ABOVE = "above"
#: The EC50 is below every concentration tested. One-sided statement only.
BOUND_BELOW = "below"
#: The midpoint is inside the tested range but the interval does not close —
#: the data is consistent with an EC50 far outside it in both directions.
BOUND_OPEN = "open"

#: A group whose fit produced a quotable EC50.
STATUS_FITTED = "fitted"
#: A group that fitted but whose EC50 is an extrapolation.
STATUS_UNBOUNDED = "unbounded"
#: A group the engine declined to fit at all.
STATUS_REFUSED = "refused"

#: Distinct positive concentrations a 4PL needs. Four parameters; four points
#: is an interpolation, not a fit, and anything fewer is under-determined.
MIN_DOSES = 4

#: Observations a 4PL needs before any uncertainty can be reported. With
#: exactly four the residual df is zero: the curve passes through the points
#: and the residual variance is 0/0.
MIN_OBSERVATIONS = 5

#: Nominal coverage of every reported interval.
DEFAULT_CONFIDENCE = 0.95

#: Reversal against the dominant trend, as a fraction of the response span,
#: at or above which :func:`monotonicity` calls the data non-monotone and the
#: fit is refused. 0.30 is a judgement call and is a parameter for that
#: reason: a real bell shape returns most of the span (a symmetric one
#: returns all of it), while a monotone series with 10% noise reverses by
#: well under a fifth of it.
MAX_REVERSAL = 0.30

#: Successive median differences smaller than this fraction of the response
#: span are treated as flat when counting sign changes — otherwise every
#: measurement error on a plateau counts as a turn.
FLAT_FRACTION = 0.05

#: How far outside the observed response range a fitted plateau may sit, as a
#: fraction of the observed span, before the plateau counts as never reached
#: and the EC50 as an extrapolation.
PLATEAU_SLACK = 0.25

#: |Hill slope| at or above which the curve is flagged as absurdly steep — an
#: all-or-nothing step between two adjacent dilutions, usually one dose doing
#: all the work or a threshold artefact rather than a binding curve.
STEEP_HILL = 10.0

#: |Hill slope| at or below which the curve barely bends across the whole
#: tested range. The EC50 of a nearly straight line is wherever you put it.
SHALLOW_HILL = 0.2

#: How far past the tested range, in log10 concentration, the profile search
#: walks before declaring that side open. 4.0 is a factor of ten thousand
#: beyond the highest dose tested; nothing quotable lives out there.
PROFILE_REACH = 4.0

#: Bisection tolerance for a profile bound, in log10 concentration. 1e-3 is a
#: 0.23% change in the reported EC50 — far below anything meaningful.
PROFILE_TOLERANCE = 1e-3

#: Relative tolerance for calling a response column constant.
CONSTANT_TOLERANCE = 1e-12

#: ``curve_fit`` function evaluations before it gives up.
_MAX_FUNCTION_EVALUATIONS = 20_000

#: ``10 ** x`` overflows past ~308; the exponent is clipped here so the model
#: returns the correct *limit* (a plateau) instead of ``inf`` or ``nan``, and
#: so ``x = 0`` evaluates to the plateau rather than raising.
_EXPONENT_LIMIT = 250.0

#: Hill magnitudes scanned by the profile's inner optimisation before it
#: refines. Log-spaced across every slope anyone has ever published.
_HILL_GRID = np.logspace(np.log10(0.02), np.log10(40.0), 40)

#: A first fit leaving less than this share of the total sum of squares is
#: accepted without trying the restart ladder (R² >= 0.9).
_GOOD_FIT_FRACTION = 0.10


def four_parameter_logistic(x, bottom, top, log10_ec50, hill):
    """The 4PL curve, parameterised in ``log10(EC50)``.

    ``y = bottom + (top - bottom) / (1 + 10 ** ((log10_ec50 - log10 x) *
    hill))``.

    At ``x == EC50`` the exponent is zero and the response is exactly halfway
    between the plateaus, which is the definition the EC50 is quoted under.

    :param x: concentration(s), in the user's units. ``0`` is evaluated at
        its limit (the low-dose plateau) rather than raising, because the
        clipped exponent below makes ``log10(0) = -inf`` well behaved; the
        fit itself never sees a zero — see :func:`fit_dose_response`.
    :param bottom: the smaller plateau, after canonicalisation.
    :param top: the larger plateau.
    :param log10_ec50: base-10 log of the half-maximal concentration.
    :param hill: slope. Negative is inhibition, positive is activation.
    :returns: the modelled response, same shape as ``x``.
    """
    values = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        exponent = (log10_ec50 - np.log10(values)) * hill
    exponent = np.clip(exponent, -_EXPONENT_LIMIT, _EXPONENT_LIMIT)
    return bottom + (top - bottom) / (1.0 + 10.0 ** exponent)


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MonotonicityCheck:
    """Whether the concentration-ordered response only ever goes one way.

    Computed on the **per-concentration medians**, not the raw points: with
    replicates the median is the robust summary of what happened at that
    dose, and a single outlier well should not be able to veto a fit.

    :param doses: the distinct positive concentrations, ascending.
    :param medians: the median response at each of them.
    :param span: ``max(medians) - min(medians)``, the yardstick everything
        else is measured against.
    :param reversal: the excursion that **no** monotone trend explains, in
        response units. It is the smaller of the largest fall after a rise
        and the largest rise after a fall, so a clean increasing series
        scores ~0 (its falls are noise), a clean decreasing series scores ~0,
        and a bell scores most of the span whichever way you read it.
    :param reversal_fraction: ``reversal / span``.
    :param sign_changes: how many times the direction of the successive
        median differences flips, ignoring steps flatter than
        :data:`FLAT_FRACTION` of the span.
    :param turning_points: the concentrations at which those flips happen.
    :param spearman_rho: rank correlation of median response against log
        concentration. Near zero with a large reversal is the signature of a
        symmetric bell.
    :param threshold: the ``reversal`` at which this check would have failed.
    :param is_monotone: the verdict.
    """

    doses: np.ndarray
    medians: np.ndarray
    span: float
    reversal: float
    reversal_fraction: float
    sign_changes: int
    turning_points: Tuple[float, ...]
    spearman_rho: float
    threshold: float
    is_monotone: bool

    def describe(self) -> str:
        """One line, for a caption or an error message."""
        if self.is_monotone:
            return (f"monotone (largest unexplained reversal "
                    f"{self.reversal_fraction:.0%} of the response span, "
                    f"Spearman rho {self.spearman_rho:+.2f})")
        turns = ", ".join(f"{d:.3g}" for d in self.turning_points) or "n/a"
        return (f"not monotone: the response turns around at {turns} and the "
                f"reversal is {self.reversal_fraction:.0%} of the response "
                f"span (Spearman rho {self.spearman_rho:+.2f})")


def monotonicity(doses: Sequence[float], responses: Sequence[float], *,
                 max_reversal: float = MAX_REVERSAL) -> MonotonicityCheck:
    """Is this concentration series consistent with a single sigmoid?

    A 4PL is monotone by construction. Data that is not monotone is not
    described by one, and fitting it anyway returns a confident EC50 for a
    curve of the wrong shape — the specific failure this module exists to
    prevent.

    The test is an excursion test rather than a sign-change count, because a
    sign-change count cannot tell a 5% wobble on a plateau from a collapse at
    the top dose, and on a ten-point series with replicates the wobbles are
    guaranteed. Both numbers are reported; only the excursion decides.

    :param doses: positive concentrations, one per observation. Replicates
        allowed and expected.
    :param responses: the matching responses.
    :param max_reversal: the fraction of the response span an excursion
        against the trend may reach. See :data:`MAX_REVERSAL`.
    :returns: a :class:`MonotonicityCheck`; read :attr:`~MonotonicityCheck.
        is_monotone`.
    """
    dose = np.asarray(doses, dtype=float)
    response = np.asarray(responses, dtype=float)
    distinct, medians, _counts = _per_dose(dose, response)
    span = float(medians.max() - medians.min()) if medians.size else 0.0
    threshold = float(max_reversal) * span

    if medians.size < 2 or span <= 0:
        return MonotonicityCheck(
            doses=distinct, medians=medians, span=span, reversal=0.0,
            reversal_fraction=0.0, sign_changes=0, turning_points=(),
            spearman_rho=float("nan"), threshold=threshold, is_monotone=True)

    running_max = np.maximum.accumulate(medians)
    running_min = np.minimum.accumulate(medians)
    fall_after_rise = float(np.max(running_max - medians))
    rise_after_fall = float(np.max(medians - running_min))
    reversal = float(min(fall_after_rise, rise_after_fall))

    steps = np.diff(medians)
    material = np.abs(steps) > FLAT_FRACTION * span
    signs = np.sign(steps)[material]
    where = np.flatnonzero(material)
    changes = 0
    turns: List[float] = []
    for i in range(1, signs.size):
        if signs[i] != signs[i - 1]:
            changes += 1
            turns.append(float(distinct[where[i]]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = float(stats.spearmanr(np.log10(distinct), medians)[0])

    return MonotonicityCheck(
        doses=distinct, medians=medians, span=span, reversal=reversal,
        reversal_fraction=reversal / span, sign_changes=int(changes),
        turning_points=tuple(turns), spearman_rho=rho, threshold=threshold,
        is_monotone=bool(reversal < threshold))


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoseResponseSpec:
    """Which columns to fit and under what policy.

    Frozen and JSON round-tripping, like
    :class:`spacr.qt.widgets.pca_model.PCASpec`, so the analysis behind a
    figure is something a settings file or a methods section can carry
    verbatim.

    :param concentration: column holding the dose. Only the positive values
        are fitted; zeros are the vehicle control and are reported separately.
    :param response: column holding the measured response.
    :param group: optional column giving one curve per level — per gene, per
        compound. ``None`` fits the table as a single series.
    :param ci_method: :data:`CI_PROFILE` (default) or :data:`CI_WALD`.
    :param confidence: nominal coverage, strictly between 0 and 1.
    :param unit: concentration unit, for the sentences only. It never enters
        the arithmetic; an EC50 is reported in whatever the column is in.
    :param direction: :data:`DIRECTION_AUTO` (default) or a pinned direction.
    :param allow_non_monotone: fit a bell-shaped series anyway. Offered
        because a user may know that the reversal is one bad well; never the
        default, and the result keeps the check so the caveat survives.
    :param max_reversal: the :func:`monotonicity` threshold.
    :raises DoseResponseError: on an unknown method or direction, or a
        confidence outside (0, 1) — at the point the spec is built, not
        halfway through a plate.
    """

    concentration: str = ""
    response: str = ""
    group: Optional[str] = None
    ci_method: str = CI_PROFILE
    confidence: float = DEFAULT_CONFIDENCE
    unit: str = ""
    direction: str = DIRECTION_AUTO
    allow_non_monotone: bool = False
    max_reversal: float = MAX_REVERSAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "concentration", str(self.concentration or ""))
        object.__setattr__(self, "response", str(self.response or ""))
        object.__setattr__(self, "unit", str(self.unit or "").strip())
        group = str(self.group).strip() if self.group else ""
        object.__setattr__(self, "group", group or None)
        if self.ci_method not in CI_METHODS:
            raise DoseResponseError(
                f"unknown ci_method {self.ci_method!r}; it is "
                f"{CI_PROFILE!r} (profile likelihood, the default — it can "
                f"report that the interval does not close) or {CI_WALD!r} "
                f"(asymptotic, always finite whether or not the data "
                f"supports it)")
        if self.direction not in DIRECTIONS:
            raise DoseResponseError(
                f"unknown direction {self.direction!r}; choose one of "
                f"{', '.join(DIRECTIONS)}")
        level = float(self.confidence)
        if not 0.0 < level < 1.0:
            raise DoseResponseError(
                f"confidence is a coverage probability and must be strictly "
                f"between 0 and 1, not {self.confidence}. For a 95% interval "
                f"pass 0.95, not 95.")
        object.__setattr__(self, "confidence", level)
        reversal = float(self.max_reversal)
        if not 0.0 < reversal <= 1.0:
            raise DoseResponseError(
                f"max_reversal is a fraction of the response span and must be "
                f"in (0, 1], not {self.max_reversal}")
        object.__setattr__(self, "max_reversal", reversal)

    # -- edits ------------------------------------------------------------
    def with_columns(self, concentration: str, response: str,
                     group: Optional[str] = None) -> "DoseResponseSpec":
        """A copy pointed at different columns."""
        return replace(self, concentration=concentration, response=response,
                       group=group)

    def with_ci_method(self, method: str) -> "DoseResponseSpec":
        """A copy using a different interval."""
        return replace(self, ci_method=method)

    def with_unit(self, unit: str) -> "DoseResponseSpec":
        """A copy that says the concentrations are in ``unit``."""
        return replace(self, unit=unit)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """A plain dict, for JSON or a settings file."""
        return {
            "concentration": self.concentration,
            "response": self.response,
            "group": self.group,
            "ci_method": self.ci_method,
            "confidence": self.confidence,
            "unit": self.unit,
            "direction": self.direction,
            "allow_non_monotone": self.allow_non_monotone,
            "max_reversal": self.max_reversal,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoseResponseSpec":
        """Rebuild from :meth:`to_dict`.

        Unknown keys are ignored and missing keys defaulted, so an analysis
        written by another build of spaCR still opens.
        """
        fields = {"concentration", "response", "group", "ci_method",
                  "confidence", "unit", "direction", "allow_non_monotone",
                  "max_reversal"}
        known = {k: v for k, v in dict(payload).items() if k in fields}
        return cls(**known)

    def to_json(self) -> str:
        """:meth:`to_dict` as sorted JSON."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "DoseResponseSpec":
        """Inverse of :meth:`to_json`."""
        return cls.from_dict(json.loads(text))

    def describe(self) -> str:
        """One line, for a figure caption."""
        columns = (f"{self.response or '?'} vs {self.concentration or '?'}"
                   + (f" per {self.group}" if self.group else ""))
        method = ("profile-likelihood" if self.ci_method == CI_PROFILE
                  else "Wald")
        unit = f" ({self.unit})" if self.unit else ""
        return (f"4PL · {columns}{unit} · {self.confidence:.0%} {method} CI "
                f"on log10(EC50) · direction {self.direction}")


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoseResponseResult:
    """One fitted curve, plus everything needed to know whether to quote it.

    The important field is :attr:`ec50_bounded`. When it is ``False``,
    :attr:`ec50` is ``None`` — there is no way to read a point estimate out of
    this object without passing the flag, which is the whole design. The
    fitted number is still there, under :attr:`ec50_unconstrained`, because
    hiding it would only make people re-derive it; the name says what it is.

    :param group: the level this curve belongs to, or ``""``.
    :param bottom: smaller plateau. :param top: larger plateau.
    :param log10_ec50: the fitted parameter. Always finite when the fit
        converged, whether or not it is inside the tested range.
    :param hill: slope; negative for inhibition, positive for activation.
    :param ec50: the quotable half-maximal concentration, or ``None`` when
        the experiment does not bound it.
    :param ec50_unconstrained: ``10 ** log10_ec50``, always. An extrapolation
        when :attr:`ec50_bounded` is ``False``.
    :param ec50_low: back-transformed lower end of the interval, or ``None``
        for an open side.
    :param ec50_high: back-transformed upper end of the interval, or
        ``None`` for an open side.
    :param log10_ec50_ci: the same interval before back-transformation.
    :param hill_ci: Wald interval for the Hill slope.
    :param top_ci: Wald interval for the top asymptote.
    :param bottom_ci: Wald interval for the bottom asymptote.
    :param bound_direction: one of :data:`BOUND_OK`, :data:`BOUND_ABOVE`,
        :data:`BOUND_BELOW`, :data:`BOUND_OPEN`.
    :param dose: the concentrations actually fitted, in input order.
    :param response: the responses actually fitted, aligned with ``dose``
        (positive
        concentrations only), in input order.
    :param n_obs: observations fitted.
    :param n_doses: distinct
        concentrations.
    :param dof: residual degrees of freedom, ``n_obs - 4``.
    :param rse: residual standard error, in response units.
    :param r_squared: with the health warning in :meth:`caveats` attached.
    :param lack_of_fit_f: F statistic of the test against pure error.
    :param lack_of_fit_p: p value of the test against pure
        error, or ``None`` when the design cannot support it.
    :param lack_of_fit_df: ``(numerator, denominator)`` df of that test.
    :param covariance: the 4×4 matrix, or ``None`` when it could not be
        estimated. :param covariance_ok: whether it is finite and usable.
    :param optimizer_notes: every warning scipy raised during the fit,
        captured rather than allowed to escape — an
        ``OptimizeWarning: Covariance of the parameters could not be
        estimated`` is a *result*, not console noise.
    :param vehicle_response: mean response at concentration 0, or ``None``.
    :param n_vehicle: how many vehicle observations there were.
    :param n_excluded: rows dropped for a missing or non-finite value.
    :param check: the :class:`MonotonicityCheck` this fit passed (or was
        forced past).
    """

    group: str
    bottom: float
    top: float
    log10_ec50: float
    hill: float
    ec50: Optional[float]
    ec50_unconstrained: float
    ec50_bounded: bool
    bound_direction: str
    ec50_low: Optional[float]
    ec50_high: Optional[float]
    log10_ec50_ci: Tuple[Optional[float], Optional[float]]
    hill_ci: Tuple[Optional[float], Optional[float]]
    top_ci: Tuple[Optional[float], Optional[float]]
    bottom_ci: Tuple[Optional[float], Optional[float]]
    dose: np.ndarray
    response: np.ndarray
    n_obs: int
    n_doses: int
    dof: int
    dose_min: float
    dose_max: float
    sse: float
    rse: float
    r_squared: float
    lack_of_fit_f: Optional[float]
    lack_of_fit_p: Optional[float]
    lack_of_fit_df: Optional[Tuple[int, int]]
    covariance: Optional[np.ndarray]
    covariance_ok: bool
    check: MonotonicityCheck
    ci_method: str = CI_PROFILE
    confidence: float = DEFAULT_CONFIDENCE
    unit: str = ""
    direction: str = DIRECTION_INHIBITION
    vehicle_response: Optional[float] = None
    n_vehicle: int = 0
    n_excluded: int = 0
    optimizer_notes: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()

    # -- shape -------------------------------------------------------------
    @property
    def parameters(self) -> Tuple[float, float, float, float]:
        """``(bottom, top, log10_ec50, hill)`` — the vector the model takes."""
        return (self.bottom, self.top, self.log10_ec50, self.hill)

    @property
    def status(self) -> str:
        """:data:`STATUS_FITTED` or :data:`STATUS_UNBOUNDED`."""
        return STATUS_FITTED if self.ec50_bounded else STATUS_UNBOUNDED

    @property
    def span(self) -> float:
        """``top - bottom``: how much of a window the curve moves through."""
        return float(self.top - self.bottom)

    @property
    def has_replicates(self) -> bool:
        """Whether any concentration was measured more than once."""
        return self.n_obs > self.n_doses

    @property
    def ec50_fold_uncertainty(self) -> Optional[float]:
        """``sqrt(high / low)`` — the interval as a multiplicative factor.

        The natural way to state a potency's uncertainty: "1.0 µM, within a
        factor of 1.6". ``None`` when either side is open.
        """
        if self.ec50_low is None or self.ec50_high is None:
            return None
        if self.ec50_low <= 0:  # pragma: no cover - a log-space CI is positive
            return None
        return float(np.sqrt(self.ec50_high / self.ec50_low))

    def is_steep(self) -> bool:
        """Whether the Hill slope is implausibly steep."""
        return bool(abs(self.hill) >= STEEP_HILL)

    def is_shallow(self) -> bool:
        """Whether the curve barely bends across the tested range."""
        return bool(abs(self.hill) <= SHALLOW_HILL)

    # -- prediction --------------------------------------------------------
    def predict(self, x) -> np.ndarray:
        """The fitted response at ``x``."""
        return four_parameter_logistic(x, *self.parameters)

    def curve(self, points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """``(x, y)`` for drawing, geometrically spaced across the tested
        range with half a decade of margin at each end.

        Geometric because the axis a dose–response is read on is logarithmic;
        an evenly spaced grid would put nine tenths of its points in the top
        dilution and draw the interesting part as three line segments.
        """
        low = np.log10(self.dose_min) - 0.5
        high = np.log10(self.dose_max) + 0.5
        grid = np.logspace(low, high, max(2, int(points)))
        return grid, self.predict(grid)

    # -- frames ------------------------------------------------------------
    def points_frame(self) -> pd.DataFrame:
        """The fitted observations with their fitted values and residuals."""
        fitted = self.predict(self.dose)
        return pd.DataFrame({
            "group": self.group,
            "concentration": self.dose,
            "response": self.response,
            "fitted": fitted,
            "residual": self.response - fitted,
        })

    def curve_frame(self, points: int = 200) -> pd.DataFrame:
        """:meth:`curve` as a two-column frame, for export."""
        x, y = self.curve(points)
        return pd.DataFrame({"group": self.group, "concentration": x,
                             "fitted": y})

    def parameter_frame(self) -> pd.DataFrame:
        """One row per parameter: estimate and interval.

        ``EC50`` appears as its own row, back-transformed, and its estimate is
        ``NaN`` when the experiment does not bound it — the frame carries the
        same refusal the object does, so an exported CSV cannot quietly
        disagree with the screen.
        """
        rows = [
            ("bottom", self.bottom, self.bottom_ci),
            ("top", self.top, self.top_ci),
            ("log10_ec50", self.log10_ec50, self.log10_ec50_ci),
            ("hill", self.hill, self.hill_ci),
            ("ec50", self.ec50 if self.ec50_bounded else float("nan"),
             (self.ec50_low, self.ec50_high)),
        ]
        return pd.DataFrame({
            "group": [self.group] * len(rows),
            "parameter": [name for name, _v, _ci in rows],
            "estimate": [float(v) if v is not None else float("nan")
                         for _n, v, _ci in rows],
            "ci_low": [float(ci[0]) if ci[0] is not None else float("nan")
                       for _n, _v, ci in rows],
            "ci_high": [float(ci[1]) if ci[1] is not None else float("nan")
                        for _n, _v, ci in rows],
        })

    def summary_row(self) -> Dict[str, Any]:
        """One flat record — the row this curve gets in a results table."""
        return {
            "group": self.group,
            "status": self.status,
            "n": self.n_obs,
            "concentrations": self.n_doses,
            "ec50": self.ec50 if self.ec50 is not None else float("nan"),
            "ec50_low": (self.ec50_low if self.ec50_low is not None
                         else float("nan")),
            "ec50_high": (self.ec50_high if self.ec50_high is not None
                          else float("nan")),
            "ec50_unconstrained": self.ec50_unconstrained,
            "hill": self.hill,
            "top": self.top,
            "bottom": self.bottom,
            "r_squared": self.r_squared,
            "rse": self.rse,
            "lack_of_fit_p": (self.lack_of_fit_p
                              if self.lack_of_fit_p is not None
                              else float("nan")),
            "note": ("" if self.ec50_bounded else self.bound_statement()),
        }

    # -- saying it in words ------------------------------------------------
    def _dose(self, value: Optional[float]) -> str:
        if value is None or not np.isfinite(value):
            return "n/a"
        return f"{value:.3g}" + (f" {self.unit}" if self.unit else "")

    def bound_statement(self) -> str:
        """The one-sided fact the experiment supports, when it supports no
        two-sided one.

        Returns ``""`` for a bounded fit, so a caller can print it
        unconditionally.
        """
        if self.ec50_bounded:
            return ""
        if self.bound_direction == BOUND_ABOVE:
            return (f"EC50 > {self._dose(self.dose_max)}, the highest "
                    f"concentration tested")
        if self.bound_direction == BOUND_BELOW:
            return (f"EC50 < {self._dose(self.dose_min)}, the lowest "
                    f"concentration tested")
        return (f"EC50 is not bounded in either direction by concentrations "
                f"from {self._dose(self.dose_min)} to "
                f"{self._dose(self.dose_max)}")

    def headline(self) -> str:
        """One sentence — the number, or the reason there is no number."""
        shape = ("inhibition" if self.hill < 0 else "activation")
        where = (f"{self.n_obs} observations at {self.n_doses} "
                 f"concentrations")
        if not self.ec50_bounded:
            return (
                f"This experiment does not bound the EC50: "
                f"{self.bound_statement()}. The unconstrained fit puts it at "
                f"{self._dose(self.ec50_unconstrained)}, but that number is "
                f"set by the shape of the model where the measurements ran "
                f"out rather than by the measurements, so it must not be "
                f"quoted as an EC50 ({shape}, Hill {self.hill:+.2f}, "
                f"{where}).")
        interval = (f"{self.confidence:.0%} "
                    f"{'profile' if self.ci_method == CI_PROFILE else 'Wald'} "
                    f"CI {self._dose(self.ec50_low)} – "
                    f"{self._dose(self.ec50_high)}")
        fold = self.ec50_fold_uncertainty
        factor = (f", a factor of {fold:.2g} either way" if fold else "")
        return (f"EC50 = {self._dose(self.ec50)} ({interval}{factor}); "
                f"Hill slope {self.hill:+.2f} ({shape}); plateaus "
                f"{self.bottom:.4g} to {self.top:.4g}; {where}.")

    def caveats(self) -> Tuple[str, ...]:
        """Everything a reader needs before believing the number."""
        out: List[str] = []
        if not self.ec50_bounded:
            out.append(
                "The point estimate is deliberately withheld (`ec50` is "
                "None): the fitted midpoint is outside what the experiment "
                "measured, so its value is set by the shape of the model "
                "rather than by data. Extend the dilution series past the "
                "midpoint before quoting a potency.")
        if not self.covariance_ok:
            out.append(
                "The covariance matrix could not be estimated, so there is "
                "no Wald interval on any parameter. That happens when a "
                "parameter is not identified by the data — most often the "
                "plateau that was never reached.")
        out.append(
            f"R² is {self.r_squared:.4f}, and it is nearly useless here: "
            f"almost any monotone curve through a dose–response scores above "
            f"0.95, because the total sum of squares is dominated by the gap "
            f"between the plateaus. Read the residual standard error "
            f"({self.rse:.4g}, in response units) and the lack-of-fit test "
            f"instead.")
        if self.lack_of_fit_p is None:
            if not self.has_replicates:
                out.append(
                    "No concentration was measured twice, so there is no "
                    "pure-error estimate and no lack-of-fit test. Whether a "
                    "4PL is the right shape for this data is untested, not "
                    "confirmed.")
            else:
                out.append(
                    f"The lack-of-fit test needs more distinct "
                    f"concentrations than parameters; {self.n_doses} "
                    f"concentrations against 4 parameters leaves it no "
                    f"degrees of freedom.")
        elif self.lack_of_fit_p < 0.05:
            out.append(
                f"Lack-of-fit F = {self.lack_of_fit_f:.3g} on "
                f"{self.lack_of_fit_df[0]} and {self.lack_of_fit_df[1]} df, "
                f"p = {self.lack_of_fit_p:.3g}: the scatter around the curve "
                f"is bigger than the scatter between replicates, so a 4PL is "
                f"the wrong shape for this data. Anything below is the EC50 "
                f"of a curve that does not describe the experiment. (The "
                f"test is approximate for a nonlinear model and rejects a "
                f"little above nominal, so read a p just under 0.05 as a "
                f"hint and a p of 1e-10 as a verdict.)")
        if self.is_steep():
            out.append(
                f"The Hill slope is {self.hill:+.2f}, which is an "
                f"all-or-nothing step between two adjacent dilutions rather "
                f"than a binding curve. It usually means one concentration is "
                f"doing all the work, or that the response saturates the "
                f"assay.")
        elif self.is_shallow():
            out.append(
                f"The Hill slope is {self.hill:+.2f}: the curve barely bends "
                f"across the whole tested range, so the midpoint is poorly "
                f"located wherever the interval happens to fall.")
        if self.dof < 4:
            out.append(
                f"{self.dof} residual degree(s) of freedom. Every interval "
                f"here uses a t quantile on that df, which is why they are "
                f"wide; they are not wide by mistake.")
        if not self.check.is_monotone:
            out.append(
                f"Fitted against the monotonicity check, which failed: "
                f"{self.check.describe()}.")
        if self.n_vehicle:
            out.append(
                f"{self.n_vehicle} vehicle observation(s) at concentration 0 "
                f"were excluded from the fit (log10(0) has no value) and "
                f"averaged {self.vehicle_response:.4g}. Compare that with the "
                f"fitted low-dose plateau: they should agree, and a gap means "
                f"the series never got back to control.")
        if self.n_excluded:
            out.append(
                f"{self.n_excluded} row(s) had a missing or non-finite "
                f"concentration or response and were dropped.")
        for note in self.optimizer_notes:
            out.append(f"The optimiser said: {note}")
        return tuple(out)

    def report(self) -> str:
        """The whole story, as the panel prints it and a report file writes it."""
        lines = [
            f"4PL dose–response{f' · {self.group}' if self.group else ''} "
            f"({self.n_obs} observations, {self.n_doses} concentrations from "
            f"{self._dose(self.dose_min)} to {self._dose(self.dose_max)}).",
            "",
            "  " + self.headline(),
        ]
        if not self.ec50_bounded:
            lines.append(f"  ec50_bounded = False · {self.bound_statement()}.")
        lines.append("")
        lines.append(
            f"  residual SE {self.rse:.4g} on {self.dof} df; R² "
            f"{self.r_squared:.4f}")
        if self.lack_of_fit_p is not None:
            verdict = ("a 4PL does not fit" if self.lack_of_fit_p < 0.05
                       else "consistent with a 4PL")
            lines.append(
                f"  lack of fit vs pure error: F = {self.lack_of_fit_f:.4g} "
                f"on {self.lack_of_fit_df[0]}, {self.lack_of_fit_df[1]} df, "
                f"p = {self.lack_of_fit_p:.4g} — {verdict}")
        else:
            lines.append("  lack of fit vs pure error: not testable")
        caveats = self.caveats()
        if caveats:
            lines.append("")
            lines.extend("  ! " + c for c in caveats)
        if self.notes:
            lines.append("")
            lines.extend("  · " + n for n in self.notes)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Many curves at once
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroupFit:
    """One level of the grouping column: a result, or the refusal.

    Both are first-class. A plate where three compounds fit and one is
    bell-shaped has four rows in its table, and the fourth says why — hiding
    it would turn a refusal into a missing row, which reads as "no data".

    :param group: the level.
    :param result: its :class:`DoseResponseResult`, or ``None``.
    :param error: the refusal message, or ``None``.
    :param n_rows: rows the group had before anything was dropped.
    """

    group: str
    result: Optional[DoseResponseResult]
    error: Optional[str]
    n_rows: int

    @property
    def status(self) -> str:
        """:data:`STATUS_REFUSED`, :data:`STATUS_UNBOUNDED` or
        :data:`STATUS_FITTED`."""
        if self.result is None:
            return STATUS_REFUSED
        return self.result.status

    def summary_row(self) -> Dict[str, Any]:
        """The row this level gets in the results table, refusal included."""
        if self.result is not None:
            row = self.result.summary_row()
            row["group"] = self.group
            return row
        blank = float("nan")
        return {
            "group": self.group, "status": STATUS_REFUSED, "n": self.n_rows,
            "concentrations": 0, "ec50": blank, "ec50_low": blank,
            "ec50_high": blank, "ec50_unconstrained": blank, "hill": blank,
            "top": blank, "bottom": blank, "r_squared": blank, "rse": blank,
            "lack_of_fit_p": blank, "note": self.error or "refused",
        }


@dataclass(frozen=True)
class DoseResponseSet:
    """Every curve on a plate, in the order the levels were seen.

    :param fits: one :class:`GroupFit` per level.
    :param spec: the spec they were all fitted under.
    """

    fits: Tuple[GroupFit, ...]
    spec: DoseResponseSpec

    def __len__(self) -> int:
        """How many levels were attempted."""
        return len(self.fits)

    def __iter__(self):
        """Iterate the :class:`GroupFit` records."""
        return iter(self.fits)

    @property
    def groups(self) -> Tuple[str, ...]:
        """The level names, in order."""
        return tuple(fit.group for fit in self.fits)

    def get(self, group: str) -> Optional[GroupFit]:
        """The fit for one level, or ``None``."""
        for fit in self.fits:
            if fit.group == group:
                return fit
        return None

    def results(self) -> Tuple[DoseResponseResult, ...]:
        """Every level that produced a curve, bounded or not."""
        return tuple(f.result for f in self.fits if f.result is not None)

    def refusals(self) -> Tuple[GroupFit, ...]:
        """Every level the engine declined to fit."""
        return tuple(f for f in self.fits if f.result is None)

    def table(self) -> pd.DataFrame:
        """One row per level — the results grid, with refusals in it.

        Columns: ``group``, ``status``, ``n``, ``concentrations``, ``ec50``
        and its interval, ``ec50_unconstrained``, ``hill``, ``top``,
        ``bottom``, ``r_squared``, ``rse``, ``lack_of_fit_p``, ``note``.
        ``ec50`` is ``NaN`` for anything not :data:`STATUS_FITTED`, and
        ``note`` says which of the two reasons it is.
        """
        rows = [fit.summary_row() for fit in self.fits]
        if not rows:
            return pd.DataFrame(columns=[
                "group", "status", "n", "concentrations", "ec50", "ec50_low",
                "ec50_high", "ec50_unconstrained", "hill", "top", "bottom",
                "r_squared", "rse", "lack_of_fit_p", "note"])
        return pd.DataFrame(rows)

    def headline(self) -> str:
        """One sentence about the whole plate."""
        fitted = sum(1 for f in self.fits if f.status == STATUS_FITTED)
        unbounded = sum(1 for f in self.fits if f.status == STATUS_UNBOUNDED)
        refused = len(self.refusals())
        parts = [f"{fitted} of {len(self)} curve(s) give a bounded EC50"]
        if unbounded:
            parts.append(f"{unbounded} are one-sided (the midpoint is outside "
                         f"the tested range)")
        if refused:
            parts.append(f"{refused} were refused")
        return "; ".join(parts) + "."

    def report(self) -> str:
        """Every curve's report, one after another, under a summary line."""
        lines = [self.headline(), f"  {self.spec.describe()}", ""]
        for fit in self.fits:
            if fit.result is not None:
                lines.append(fit.result.report())
            else:
                lines.append(f"4PL dose–response · {fit.group}: REFUSED — "
                             f"{fit.error}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# The computation
# ---------------------------------------------------------------------------

def _back_transform(log10_value: Optional[float]) -> Optional[float]:
    """``10 ** x`` as a concentration, without an overflow on the way.

    Python's ``float.__pow__`` raises ``OverflowError`` rather than returning
    ``inf``, and a Wald interval on an unidentified parameter really does
    produce an upper bound of ``10 ** 400``: the asymptotic formula is
    ``L ± t·SE`` and nothing in it is bounded by the tested range. Clipping to
    the same limit the model uses keeps the arithmetic finite; the number is
    meaningless either way, which is why the boundedness rules withhold it.
    """
    if log10_value is None or not np.isfinite(log10_value):
        return None
    clipped = float(np.clip(log10_value, -_EXPONENT_LIMIT, _EXPONENT_LIMIT))
    return float(10.0 ** clipped)


def _per_dose(dose: np.ndarray, response: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(distinct doses, median response, count)`` — the replicate summary."""
    if dose.size == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty, np.zeros(0, dtype=int)
    distinct, inverse = np.unique(dose, return_inverse=True)
    medians = np.array([float(np.median(response[inverse == i]))
                        for i in range(distinct.size)])
    return distinct, medians, np.bincount(inverse, minlength=distinct.size)


def _clean(doses: Sequence[float], responses: Sequence[float]
           ) -> Tuple[np.ndarray, np.ndarray, Optional[float], int, int]:
    """Split the input into fittable points, the vehicle, and the rubbish.

    Returns ``(dose, response, vehicle_mean, n_vehicle, n_excluded)``.

    A zero concentration is a vehicle control, which is *normal* and belongs
    in the file: it is removed from the fit (``log10(0)`` has no value and a
    silent ``-inf`` would poison the whole design matrix) and reported as a
    reference response. A negative concentration has no such reading and is
    refused.
    """
    dose = np.asarray(doses, dtype=float).ravel()
    response = np.asarray(responses, dtype=float).ravel()
    if dose.size != response.size:
        raise DoseResponseError(
            f"there are {dose.size} concentration(s) and {response.size} "
            f"response(s); they have to pair up one to one")
    finite = np.isfinite(dose) & np.isfinite(response)
    n_excluded = int((~finite).sum())
    dose, response = dose[finite], response[finite]

    negative = dose < 0
    if negative.any():
        worst = ", ".join(f"{v:.3g}" for v in np.unique(dose[negative])[:4])
        raise DoseResponseError(
            f"{int(negative.sum())} concentration(s) are negative ({worst}). "
            f"A concentration of zero is a vehicle control and is handled as "
            f"one; a negative concentration is a data error — check whether "
            f"the column holds a log dose rather than a dose.")

    zero = dose == 0
    n_vehicle = int(zero.sum())
    vehicle = float(np.mean(response[zero])) if n_vehicle else None
    return dose[~zero], response[~zero], vehicle, n_vehicle, n_excluded


def _guard(dose: np.ndarray, response: np.ndarray, n_vehicle: int) -> None:
    """Refuse everything that cannot carry a 4PL, with the count in the text."""
    distinct = np.unique(dose)
    if distinct.size < MIN_DOSES:
        vehicle = (f" (the {n_vehicle} vehicle observation(s) at "
                   f"concentration 0 cannot count towards this — a 4PL is "
                   f"fitted in log concentration)" if n_vehicle else "")
        raise DoseResponseError(
            f"a four-parameter logistic has four parameters and this series "
            f"has {distinct.size} distinct positive concentration(s){vehicle}. "
            f"At least {MIN_DOSES} are needed to fit one at all, and it takes "
            f"6–10 spanning the midpoint to fit one worth quoting.")
    if dose.size < MIN_OBSERVATIONS:
        raise DoseResponseError(
            f"{dose.size} observations against 4 parameters leaves "
            f"{dose.size - 4} residual degrees of freedom, so the curve "
            f"passes through the points and there is nothing left to estimate "
            f"the uncertainty from. At least {MIN_OBSERVATIONS} are needed.")
    spread = float(np.max(response) - np.min(response))
    scale = max(abs(float(np.mean(response))), 1.0)
    if spread <= scale * CONSTANT_TOLERANCE:
        raise DoseResponseError(
            f"every response is {float(response[0]):.6g}. There is no curve "
            f"in a flat line: the plateaus, the midpoint and the slope are "
            f"all unidentified, and any EC50 reported from it would be an "
            f"artefact of the starting guess.")


def _direction_sign(dose: np.ndarray, response: np.ndarray,
                    direction: str) -> float:
    """``-1`` for inhibition, ``+1`` for activation.

    Inferred from the Spearman rank correlation of response against log
    concentration — rank-based so one saturated well cannot flip it — with the
    difference between the plateau ends as the tie-break for the case where
    the ranks are exactly balanced.
    """
    if direction == DIRECTION_INHIBITION:
        return -1.0
    if direction == DIRECTION_ACTIVATION:
        return 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = float(stats.spearmanr(np.log10(dose), response)[0])
    if np.isfinite(rho) and rho != 0.0:
        return -1.0 if rho < 0 else 1.0
    distinct, medians, _ = _per_dose(dose, response)
    return -1.0 if medians[-1] < medians[0] else 1.0


def _initial_guesses(dose: np.ndarray, response: np.ndarray,
                     sign: float) -> List[Tuple[float, float, float, float]]:
    """Data-derived starting points, best first.

    The plateaus come from the extreme quarter of the *distinct*
    concentrations (so a dose with six replicates does not outvote three
    doses with one), the midpoint from where the per-dose medians cross the
    half-maximal response, and the slope sign from the inferred direction.
    The rest of the list is a ladder of slope magnitudes, used only when the
    first fit comes back poor — see :func:`_best_fit`.
    """
    distinct, medians, _ = _per_dose(dose, response)
    k = max(1, distinct.size // 4)
    low = float(np.mean(medians[:k]))
    high = float(np.mean(medians[-k:]))
    top = max(low, high)
    bottom = min(low, high)
    middle = (top + bottom) / 2.0

    log_doses = np.log10(distinct)
    centred = medians - middle
    crossing = None
    for i in range(centred.size - 1):
        a, b = centred[i], centred[i + 1]
        if a == 0.0:
            crossing = log_doses[i]
            break
        if a * b < 0:
            weight = abs(a) / (abs(a) + abs(b))
            crossing = log_doses[i] + weight * (log_doses[i + 1] - log_doses[i])
            break
    if crossing is None:
        crossing = float(log_doses[int(np.argmin(np.abs(centred)))])

    guesses = [(bottom, top, float(crossing), sign * 1.0)]
    middle_log = float((log_doses[0] + log_doses[-1]) / 2.0)
    for magnitude in (2.0, 0.5, 4.0, 8.0, 0.25):
        guesses.append((bottom, top, float(crossing), sign * magnitude))
    guesses.append((bottom, top, middle_log, sign * 1.0))
    guesses.append((float(np.min(response)), float(np.max(response)),
                    middle_log, sign * 1.0))
    return guesses


def _fit_once(dose: np.ndarray, response: np.ndarray,
              p0: Tuple[float, float, float, float]):
    """One ``curve_fit``, with every warning it raises captured as data.

    ``curve_fit`` signals "I could not estimate the covariance" by *warning*
    (``scipy.optimize.OptimizeWarning``) and returning a matrix of ``inf``. A
    warning that escapes is console noise a user will not connect to the
    number in front of them — and under a strict warning filter it is a test
    failure in an unrelated module. It is caught here, deliberately, and
    becomes :attr:`DoseResponseResult.optimizer_notes`.
    """
    popt = pcov = None
    messages: List[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            popt, pcov = curve_fit(
                four_parameter_logistic, dose, response, p0=list(p0),
                maxfev=_MAX_FUNCTION_EVALUATIONS)
        except (RuntimeError, ValueError, TypeError) as exc:
            messages.append(f"{type(exc).__name__}: {exc}")
        for entry in caught:
            messages.append(f"{entry.category.__name__}: {entry.message}")
    ok = popt is not None and bool(np.all(np.isfinite(popt)))
    return popt, pcov, ok, messages


def _best_fit(dose: np.ndarray, response: np.ndarray,
              guesses: Sequence[Tuple[float, float, float, float]]):
    """The best of a ladder of starts. Returns ``(sse, popt, pcov, notes)``.

    The ladder exists because a 4PL's residual surface has a long flat valley
    when a plateau is under-sampled, and Levenberg–Marquardt from a single
    start can stop in it. It is walked lazily: a first fit that already
    explains 90% of the total sum of squares is accepted, so the common case
    costs exactly one ``curve_fit``.
    """
    total = float(np.sum((response - response.mean()) ** 2))
    good_enough = _GOOD_FIT_FRACTION * total
    best = None
    notes: List[str] = []
    for p0 in guesses:
        popt, pcov, ok, messages = _fit_once(dose, response, p0)
        notes.extend(messages)
        if not ok:
            continue
        residual = response - four_parameter_logistic(dose, *popt)
        sse = float(np.sum(residual ** 2))
        if not np.isfinite(sse):
            continue
        if best is None or sse < best[0]:
            best = (sse, popt, pcov, tuple(notes))
        if sse <= good_enough:
            break
    if best is None:
        raise DoseResponseError(
            "the optimiser did not converge on a four-parameter logistic for "
            "this series from any of the starting points tried. That is "
            "almost always a shape problem rather than a numerical one: check "
            "that the response really is sigmoid in log concentration, and "
            "that the concentration column is a concentration and not "
            "already a log dose. "
            + ("; ".join(notes[:3]) if notes else ""))
    return best


def _canonicalise(popt: np.ndarray, pcov: Optional[np.ndarray]):
    """Force ``top >= bottom`` so the Hill slope carries the direction.

    ``(bottom, top, L, h)`` and ``(top, bottom, L, -h)`` are the same curve —
    an exact symmetry of this parameterisation — so the optimiser returns
    whichever it wandered into and two runs on the same data can disagree
    about the sign of the slope. The swap is linear, so the covariance
    transforms exactly under its Jacobian.
    """
    bottom, top, log10_ec50, hill = (float(v) for v in popt)
    if top >= bottom:
        return np.array([bottom, top, log10_ec50, hill]), pcov
    jacobian = np.array([[0.0, 1.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, -1.0]])
    flipped = np.array([top, bottom, log10_ec50, -hill])
    if pcov is None:
        return flipped, None
    return flipped, jacobian @ np.asarray(pcov, dtype=float) @ jacobian.T


def _plateau_sse(log_dose: np.ndarray, response: np.ndarray,
                 log10_ec50: float, hill) -> np.ndarray:
    """Residual sum of squares with ``(bottom, top)`` profiled out exactly.

    For a fixed midpoint and slope the model is **linear** in the two
    plateaus: ``y = bottom·(1 - w) + top·w`` with ``w = 1/(1 + 10^((L - u)h))``.
    So the inner two parameters have a closed form and the profile's inner
    optimisation is one-dimensional (over the slope) instead of
    three-dimensional. That is what makes a deterministic profile interval
    cheap enough to be the default.

    ``hill`` may be an array, in which case one SSE per element comes back and
    the whole slope grid is evaluated in a handful of numpy operations.
    """
    slopes = np.atleast_1d(np.asarray(hill, dtype=float))
    exponent = np.clip((log10_ec50 - log_dose)[:, None] * slopes[None, :],
                       -_EXPONENT_LIMIT, _EXPONENT_LIMIT)
    high = 1.0 / (1.0 + 10.0 ** exponent)
    low = 1.0 - high
    y = response[:, None]
    s_ll = (low * low).sum(axis=0)
    s_lh = (low * high).sum(axis=0)
    s_hh = (high * high).sum(axis=0)
    s_ly = (low * y).sum(axis=0)
    s_hy = (high * y).sum(axis=0)
    determinant = s_ll * s_hh - s_lh * s_lh
    scale = np.maximum(s_ll * s_hh, 1e-300)
    usable = np.abs(determinant) > 1e-10 * scale
    safe = np.where(usable, determinant, 1.0)
    bottom = np.where(usable, (s_ly * s_hh - s_hy * s_lh) / safe,
                      float(response.mean()))
    top = np.where(usable, (s_hy * s_ll - s_ly * s_lh) / safe,
                   float(response.mean()))
    residual = y - (low * bottom[None, :] + high * top[None, :])
    return (residual ** 2).sum(axis=0)


def _profile_sse(log_dose: np.ndarray, response: np.ndarray,
                 log10_ec50: float, sign: float) -> float:
    """``min SSE`` over ``(bottom, top, hill)`` with the midpoint held fixed.

    The slope is searched on :data:`_HILL_GRID` — restricted to the direction
    the data already showed, because a slope of the opposite sign would be a
    different experiment, not a wider interval — and then refined by bounded
    Brent between the grid's neighbours.
    """
    grid = sign * _HILL_GRID
    values = _plateau_sse(log_dose, response, log10_ec50, grid)
    j = int(np.argmin(values))
    lo = float(_HILL_GRID[max(0, j - 1)])
    hi = float(_HILL_GRID[min(_HILL_GRID.size - 1, j + 1)])
    best = float(values[j])
    if hi > lo:
        outcome = minimize_scalar(
            lambda magnitude: float(_plateau_sse(
                log_dose, response, log10_ec50, sign * magnitude)[0]),
            bounds=(lo, hi), method="bounded",
            options={"xatol": 1e-4})
        if outcome.success and float(outcome.fun) < best:
            best = float(outcome.fun)
    return best


def _profile_bound(log_dose: np.ndarray, response: np.ndarray,
                   centre: float, sign: float, target: float,
                   step: float, limit: float,
                   upward: bool) -> Optional[float]:
    """Walk one side of the profile until it crosses ``target``.

    Steps outward from the estimate, doubling the step each time the residual
    sum of squares is still under the threshold, and stops at ``limit`` —
    :data:`PROFILE_REACH` decades past the tested range. Returning ``None``
    means the walk reached the limit without the data ever ruling that side
    out, which is the whole point of preferring this interval: the answer
    "this experiment does not bound the EC50 from above" exists here and does
    not exist in a Wald interval.
    """
    direction = 1.0 if upward else -1.0
    inside = centre
    reach = step
    for _ in range(60):
        candidate = centre + direction * reach
        if (upward and candidate > limit) or (not upward and candidate < limit):
            candidate = limit
        if _profile_sse(log_dose, response, candidate, sign) > target:
            outside = candidate
            break
        inside = candidate
        if candidate == limit:
            return None
        reach *= 2.0
    else:  # pragma: no cover - 60 doublings passes any finite limit
        return None
    while abs(outside - inside) > PROFILE_TOLERANCE:
        middle = 0.5 * (inside + outside)
        if _profile_sse(log_dose, response, middle, sign) > target:
            outside = middle
        else:
            inside = middle
    return float(0.5 * (inside + outside))


def _lack_of_fit(dose: np.ndarray, response: np.ndarray, sse: float
                 ) -> Tuple[Optional[float], Optional[float],
                            Optional[Tuple[int, int]]]:
    """The F test of model misspecification against pure error.

    Pure error is the within-concentration scatter — ``n - m`` df, and free
    of any assumption about the shape of the curve. The residual variance in
    excess of it carries ``m - 4`` df and is model misspecification. This is
    the statistic that answers "is a 4PL the right shape here", which is the
    question R² is usually misread as answering.

    Two honest caveats, both stated rather than buried.

    First, **the classical derivation is for a linear model.** Charging the
    fit exactly four degrees of freedom is exact only when the model is linear
    in its parameters, and a 4PL is not, so the null distribution is an
    approximation. It is not a harmless one: over 120 seeded datasets drawn
    from a genuine 4PL (10 concentrations × 3 replicates) this test rejects at
    nominal 5% about 11% of the time. Read it as a screen for *gross*
    misspecification — where it is emphatic, returning p ~ 1e-15 for a
    response that ramps linearly between its plateaus — rather than as a
    calibrated 5% test.

    Second, it needs both replicates (``n > m``) and more concentrations than
    parameters (``m > 4``). When either is missing it returns ``None`` —
    "cannot be tested", which is not "passed".
    """
    distinct, _medians, counts = _per_dose(dose, response)
    m = int(distinct.size)
    n = int(dose.size)
    df_pure = n - m
    df_lof = m - 4
    if df_pure < 1 or df_lof < 1:
        return None, None, None
    _values, inverse = np.unique(dose, return_inverse=True)
    means = np.zeros(m, dtype=float)
    for i in range(m):
        means[i] = float(np.mean(response[inverse == i]))
    ss_pure = float(np.sum((response - means[inverse]) ** 2))
    ss_lof = float(sse - ss_pure)
    if ss_pure <= 0:
        return None, None, None
    ss_lof = max(ss_lof, 0.0)
    f_statistic = (ss_lof / df_lof) / (ss_pure / df_pure)
    p_value = float(stats.f.sf(f_statistic, df_lof, df_pure))
    return float(f_statistic), p_value, (int(df_lof), int(df_pure))


def _bound_direction(log10_ec50: float, top: float, bottom: float,
                     hill: float, response: np.ndarray,
                     log_min: float, log_max: float,
                     open_low: bool, open_high: bool) -> str:
    """Which side the EC50 has escaped to, in order of reliability.

    The first rule is the robust one and does not use the fitted midpoint at
    all: when the observed responses never reach the fitted half-maximum, the
    experiment stopped short, and *which* end it stopped at follows from the
    direction of the curve alone. That matters because a truncated series is
    exactly the case where the fitted midpoint itself is unreliable.
    """
    middle = 0.5 * (top + bottom)
    low, high = float(np.min(response)), float(np.max(response))
    if middle < low or middle > high:
        rising = hill > 0
        below_middle = high < middle
        if rising:
            return BOUND_ABOVE if below_middle else BOUND_BELOW
        return BOUND_ABOVE if low > middle else BOUND_BELOW
    if log10_ec50 > log_max:
        return BOUND_ABOVE
    if log10_ec50 < log_min:
        return BOUND_BELOW
    if open_high and not open_low:
        return BOUND_ABOVE
    if open_low and not open_high:
        return BOUND_BELOW
    return BOUND_OPEN


def fit_dose_response(doses: Sequence[float], responses: Sequence[float],
                      spec: Optional[DoseResponseSpec] = None, *,
                      group: str = "") -> DoseResponseResult:
    """Fit one concentration series, or refuse it.

    The whole policy is in the module docstring. The short version: zeros are
    vehicle controls and are reported rather than logged; a series that is not
    monotone is refused rather than fitted; the fit is canonicalised so the
    Hill slope carries the direction; the EC50 is reported only when the
    experiment actually locates it, and otherwise as a one-sided bound with
    :attr:`DoseResponseResult.ec50` set to ``None``.

    :param doses: concentrations, one per observation, replicates included.
    :param responses: the matching responses.
    :param spec: a :class:`DoseResponseSpec`. Only its policy fields matter
        here; the column names are for :func:`fit_frame`.
    :param group: a label carried through onto the result.
    :raises DoseResponseError: for every series that cannot carry a 4PL —
        too few concentrations, a flat response, a negative concentration, a
        bell shape, or an optimiser that never converged. The message says
        which and what to do about it.
    """
    spec = spec or DoseResponseSpec()
    dose, response, vehicle, n_vehicle, n_excluded = _clean(doses, responses)
    _guard(dose, response, n_vehicle)

    check = monotonicity(dose, response, max_reversal=spec.max_reversal)
    if not check.is_monotone and not spec.allow_non_monotone:
        turns = ", ".join(f"{d:.3g}" for d in check.turning_points) or \
            "inside the tested range"
        unit = f" {spec.unit}" if spec.unit else ""
        raise DoseResponseError(
            f"this series is not monotone, so a four-parameter logistic is "
            f"the wrong model for it and no EC50 fitted to it would mean "
            f"anything. The response turns around at {turns}{unit}: the "
            f"reversal against the trend is {check.reversal_fraction:.0%} of "
            f"the response span (Spearman rho {check.spearman_rho:+.2f}, "
            f"{check.sign_changes} sign change(s)). A bell shape in a "
            f"concentration series is almost always cytotoxicity at the top "
            f"dose killing the signal that was being measured. Re-fit without "
            f"the top dose, or use a biphasic model — spaCR does not have "
            f"one. Set allow_non_monotone=True to force a fit, and expect the "
            f"EC50 to be an artefact of where the curve turns.")

    sign = _direction_sign(dose, response, spec.direction)
    sse, popt, pcov, optimizer_notes = _best_fit(
        dose, response, _initial_guesses(dose, response, sign))
    popt, pcov = _canonicalise(popt, pcov)
    bottom, top, log10_ec50, hill = (float(v) for v in popt)

    n_obs = int(dose.size)
    distinct = np.unique(dose)
    n_doses = int(distinct.size)
    dof = n_obs - 4
    fitted = four_parameter_logistic(dose, bottom, top, log10_ec50, hill)
    total = float(np.sum((response - response.mean()) ** 2))
    r_squared = float(1.0 - sse / total) if total > 0 else float("nan")
    rse = float(np.sqrt(sse / dof)) if dof > 0 else float("nan")

    covariance = None if pcov is None else np.asarray(pcov, dtype=float)
    covariance_ok = bool(
        covariance is not None and np.all(np.isfinite(covariance))
        and np.all(np.diag(covariance) >= 0))
    quantile = (float(stats.t.ppf(0.5 + spec.confidence / 2.0, dof))
                if dof > 0 else float("nan"))

    def wald(index: int) -> Tuple[Optional[float], Optional[float]]:
        if not covariance_ok or not np.isfinite(quantile):
            return (None, None)
        error = float(np.sqrt(covariance[index, index]))
        centre = float(popt[index])
        return (centre - quantile * error, centre + quantile * error)

    log_dose = np.log10(dose)
    log_min, log_max = float(log_dose.min()), float(log_dose.max())
    wald_log_ci = wald(2)

    notes: List[str] = []
    if spec.ci_method == CI_PROFILE and dof > 0 and sse > 0:
        target = sse * (1.0 + quantile ** 2 / dof)
        step = max(0.05, (log_max - log_min) / 8.0)
        lower = _profile_bound(log_dose, response, log10_ec50, sign, target,
                               step, log_min - PROFILE_REACH, upward=False)
        upper = _profile_bound(log_dose, response, log10_ec50, sign, target,
                               step, log_max + PROFILE_REACH, upward=True)
        log_ci: Tuple[Optional[float], Optional[float]] = (lower, upper)
    elif spec.ci_method == CI_PROFILE:
        log_ci = (None, None)
        notes.append(
            "the profile interval needs a positive residual sum of squares "
            "and at least one residual degree of freedom; this fit has "
            f"sse={sse:.3g} on {dof} df, so no interval was computed")
    else:
        log_ci = wald_log_ci

    # One reach for both methods. The profile stops walking at
    # PROFILE_REACH decades past the tested range and calls that side open;
    # the Wald formula has no such stopping rule and will happily return
    # 10 ** 400 as an upper bound on a parameter the data does not identify.
    # Applying the same limit to both is what makes the two methods
    # comparable: past a factor of 10**PROFILE_REACH beyond the highest dose
    # tested, "the interval ends here" and "the interval does not close" are
    # the same statement about the experiment.
    reach = (log_min - PROFILE_REACH, log_max + PROFILE_REACH)
    log_ci = (log_ci[0] if log_ci[0] is not None and log_ci[0] >= reach[0]
              else None,
              log_ci[1] if log_ci[1] is not None and log_ci[1] <= reach[1]
              else None)

    open_low = log_ci[0] is None
    open_high = log_ci[1] is None

    span = float(np.max(response) - np.min(response))
    slack = PLATEAU_SLACK * span
    plateaus_reached = (bottom >= float(np.min(response)) - slack
                        and top <= float(np.max(response)) + slack)
    middle = 0.5 * (top + bottom)
    bracketed = (float(np.min(response)) <= middle <= float(np.max(response)))
    in_range = log_min <= log10_ec50 <= log_max
    bounded = bool(in_range and bracketed and plateaus_reached
                   and not open_low and not open_high)
    direction = (BOUND_OK if bounded else
                 _bound_direction(log10_ec50, top, bottom, hill, response,
                                  log_min, log_max, open_low, open_high))

    ec50_unconstrained = _back_transform(log10_ec50)
    ec50 = ec50_unconstrained if bounded else None
    ec50_low = _back_transform(log_ci[0])
    ec50_high = _back_transform(log_ci[1])
    if not bounded:
        # An interval around a midpoint the data does not locate is a picture
        # of the model, not of the experiment. It is dropped with the point
        # estimate rather than drawn.
        ec50_low = ec50_high = None
        log_ci = (None, None)

    f_statistic, p_value, lof_df = _lack_of_fit(dose, response, sse)
    if spec.ci_method == CI_PROFILE and not covariance_ok:
        notes.append(
            "the covariance matrix was not estimable, so the Hill slope and "
            "the plateaus have no interval; the EC50's profile interval does "
            "not depend on it and is still reported")

    return DoseResponseResult(
        group=str(group), bottom=bottom, top=top, log10_ec50=log10_ec50,
        hill=hill, ec50=ec50, ec50_unconstrained=ec50_unconstrained,
        ec50_bounded=bounded, bound_direction=direction, ec50_low=ec50_low,
        ec50_high=ec50_high, log10_ec50_ci=log_ci, hill_ci=wald(3),
        top_ci=wald(1), bottom_ci=wald(0), dose=dose, response=response,
        n_obs=n_obs, n_doses=n_doses, dof=dof,
        dose_min=float(distinct.min()), dose_max=float(distinct.max()),
        sse=sse, rse=rse, r_squared=r_squared, lack_of_fit_f=f_statistic,
        lack_of_fit_p=p_value, lack_of_fit_df=lof_df, covariance=covariance,
        covariance_ok=covariance_ok, check=check, ci_method=spec.ci_method,
        confidence=spec.confidence, unit=spec.unit,
        direction=(DIRECTION_INHIBITION if hill < 0 else DIRECTION_ACTIVATION),
        vehicle_response=vehicle, n_vehicle=n_vehicle, n_excluded=n_excluded,
        optimizer_notes=tuple(dict.fromkeys(optimizer_notes)),
        notes=tuple(notes))


def fit_frame(frame: pd.DataFrame,
              spec: DoseResponseSpec) -> DoseResponseSet:
    """Fit one curve per level of ``spec.group`` — the whole plate at once.

    A refusal in one group is kept beside that group and does not stop the
    others: a plate where one compound is cytotoxic at the top dose should
    still report the other twenty-three, with the cytotoxic one visibly
    labelled rather than missing.

    :raises DoseResponseError: only for something wrong with the *table* —
        a column that is not there, or a group column with no levels. Per-curve
        failures land in :attr:`GroupFit.error`.
    """
    for column in (spec.concentration, spec.response):
        if not column:
            raise DoseResponseError(
                "both a concentration column and a response column have to be "
                "chosen before anything can be fitted.")
        if column not in frame.columns:
            raise DoseResponseError(
                f"{column!r} is not a column of this table. It has "
                f"{len(frame.columns)} columns; the first few are "
                f"{', '.join(map(str, list(frame.columns)[:6]))}.")
    if spec.group and spec.group not in frame.columns:
        raise DoseResponseError(
            f"the grouping column {spec.group!r} is not a column of this "
            f"table.")

    if spec.group:
        levels = [(str(level), part) for level, part
                  in frame.groupby(spec.group, sort=True, dropna=False,
                                   observed=True)]
        if not levels:
            raise DoseResponseError(
                f"the grouping column {spec.group!r} has no values, so there "
                f"is nothing to fit one curve per level of.")
    else:
        levels = [("", frame)]

    fits: List[GroupFit] = []
    for name, part in levels:
        try:
            result = fit_dose_response(
                part[spec.concentration].to_numpy(),
                part[spec.response].to_numpy(), spec, group=name)
        except DoseResponseError as exc:
            fits.append(GroupFit(group=name, result=None, error=str(exc),
                                 n_rows=int(len(part))))
        else:
            fits.append(GroupFit(group=name, result=result, error=None,
                                 n_rows=int(len(part))))
    return DoseResponseSet(fits=tuple(fits), spec=spec)


# ---------------------------------------------------------------------------
# Column suggestions — the only seam that reaches into the Qt tree
# ---------------------------------------------------------------------------

def _kinds(frame: pd.DataFrame) -> Mapping[str, str]:
    """The Local Data Filter's column classification, by name.

    Imported inside the function on purpose. The classifier lives in
    :mod:`spacr.qt.widgets.graph_spec`, which reaches
    :mod:`spacr.qt.widgets.data_filter_panel` and therefore PySide6; reusing
    it is right — two column classifiers in one codebase would give a user two
    mental models of the same table — but the *fitting* path must stay
    importable with no Qt installed, so the dependency is paid only by the
    caller that asks for column suggestions.
    """
    from .graph_spec import column_kinds
    return column_kinds(frame)


def candidate_concentration_columns(frame: pd.DataFrame) -> Tuple[str, ...]:
    """Columns worth offering as the concentration axis.

    Numeric, not a key or free text, and carrying at least :data:`MIN_DOSES`
    distinct positive values — a column that never takes four different
    positive values cannot be a dilution series whatever it is called, and
    offering it only produces a refusal one click later.

    Note what is *not* required: :data:`~spacr.qt.widgets.graph_spec.
    CONTINUOUS`. The shared classifier calls a low-cardinality numeric column
    categorical, which is the right call for ``cell_count`` and the wrong one
    here — an eight-point dilution series has exactly eight levels *by
    design*, so the classifier's own rule would hide every concentration
    column in the project. The classifier is still what excludes object keys
    and free text (:data:`~spacr.qt.widgets.graph_spec.UNPLOTTABLE`), which is
    the part of its judgement that transfers; the continuous/categorical split
    is simply not the cut a dose column falls on.
    """
    from .graph_spec import UNPLOTTABLE
    kinds = _kinds(frame)
    out: List[str] = []
    for name in sorted(kinds):
        if kinds[name] == UNPLOTTABLE:
            continue
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        positive = values[np.isfinite(values) & (values > 0)]
        if np.unique(positive).size >= MIN_DOSES:
            out.append(name)
    return tuple(out)


def candidate_response_columns(frame: pd.DataFrame) -> Tuple[str, ...]:
    """Columns worth offering as the response axis: every continuous one.

    Here the classifier's continuous/categorical split *is* the right cut: a
    response is a measured quantity, and a column with four levels is a label
    or a count rather than something a sigmoid passes through.
    """
    from .graph_spec import CONTINUOUS
    kinds = _kinds(frame)
    return tuple(sorted(name for name, kind in kinds.items()
                        if kind == CONTINUOUS))
