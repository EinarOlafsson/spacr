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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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
    "SelectivityIndex", "selectivity_index",
    "SYNERGY_BLISS", "SYNERGY_LOEWE", "SYNERGY_MODELS",
    "InteractionSurface", "bliss_surface", "loewe_surface",
    "Checkerboard", "checkerboard_from_frame",
    "NORMALISE_NONE", "NORMALISE_PERCENT", "NORMALISATIONS",
    "PERCENT_COLUMN", "ZPRIME_MARGINAL",
    "PlateSpec", "PlateReport",
    "normalise_to_controls", "plate_reports",
    "MAX_HETEROGENEITY", "MIN_PLATES",
    "PooledFit", "pool_across_plates", "pool_frame",
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

    Computed on the **per-concentration medians**, not the raw points. The
    median limits the influence of one outlier well on the fit decision.

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
        """Normalise the column names and validate the fit settings.

        :raises DoseResponseError: if ``ci_method`` or ``direction`` is not one
            this module offers; if ``confidence`` is not strictly between 0 and
            1 -- it is a coverage probability, so a 95% interval is 0.95 and not
            95; or if ``max_reversal`` is not a fraction of the response span in
            ``(0, 1]``.
        """
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
    :param bottom: smaller plateau.
    :param top: larger plateau.
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
        estimated.
    :param covariance_ok: whether it is finite and usable.
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
        if self.ec50_low <= 0:
            # A bound from `fit_dose_response` IS positive -- both ends
            # are back-transformed out of log space. But this dataclass
            # is public, frozen and validates nothing, so one
            # `dataclasses.replace` away is a bound of zero, and the
            # alternative to declining is a division that yields `inf`
            # and a panel reporting "within a factor of inf".
            #
            # `<= 0` rather than `== 0`: a negative bound would otherwise
            # take the square root of a negative number.
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
        """Render one dose for the report, with its unit.

        :param value: the dose; ``None`` or non-finite renders as ``"n/a"``,
            which is what an EC50 the data does not determine looks like.
        :returns: the formatted dose.
        """
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
    else:
        # THE WALK NEVER ARRIVED. Sixty doublings covers `step * 2**59`,
        # which is not the same as "any finite limit" -- a step small
        # enough against a large enough limit exhausts the loop, and an
        # infinite limit exhausts it outright.
        #
        # Either way the answer is the same one the limit case gives:
        # this experiment does not bound the EC50 on that side.
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
        """One parameter's Wald interval, or ``None`` when it cannot be formed.

        Returns None rather than an interval when the covariance is unusable:
        an interval computed from a bad covariance looks like a result and is
        not one.
        """
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
    does not identify dose columns reliably.
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


@dataclass(frozen=True)
class SelectivityIndex:
    """Host toxicity over parasite killing, with the interval it deserves.

    THE NUMBER THAT DECIDES WHETHER ANYBODY CARES about an anti-parasitic
    compound is not the EC50, it is this ratio. A compound that kills the
    parasite at 1 uM and the host monolayer at 1.2 uM is not a hit, and an
    EC50 quoted with a clean confidence interval says nothing about that.

    QUOTED WITH ITS INTERVAL OR NOT AT ALL, for the same reason this module
    already refuses a naked EC50: a ratio of two uncertain numbers is more
    uncertain than either of them, and a selectivity index without its
    interval invites a reader to treat 1.2 and 12 as the same kind of claim.

    :param status: :data:`STATUS_FITTED`, :data:`STATUS_UNBOUNDED` or
        :data:`STATUS_REFUSED`, reusing the vocabulary the single-curve fits
        already speak rather than inventing a second one.
    :param index: the quotable ratio, or ``None`` when it is not quotable.
    :param index_low: lower end of the interval, or ``None`` for an open side.
    :param index_high: upper end, or ``None`` for an open side.
    :param log10_index: the difference of the two log10 midpoints, always
        present when both curves fitted. An extrapolation when either EC50 is
        unbounded, exactly as :attr:`DoseResponseResult.ec50_unconstrained`
        is.
    :param host: the host-viability fit.
    :param pathogen: the parasite fit.
    :param note: why, when the index is refused or one-sided.
    """

    status: str
    index: Optional[float]
    index_low: Optional[float]
    index_high: Optional[float]
    log10_index: Optional[float]
    host: Optional[DoseResponseResult]
    pathogen: Optional[DoseResponseResult]
    confidence: float = DEFAULT_CONFIDENCE
    note: str = ""

    def summary_row(self) -> Dict[str, Any]:
        """One row for the results table, refusal included."""
        blank = float("nan")
        return {
            "metric": "selectivity_index",
            "status": self.status,
            "selectivity_index": blank if self.index is None else self.index,
            "si_low": blank if self.index_low is None else self.index_low,
            "si_high": blank if self.index_high is None else self.index_high,
            "host_ec50": blank if (self.host is None or self.host.ec50 is None)
                         else self.host.ec50,
            "pathogen_ec50": (blank if (self.pathogen is None
                                        or self.pathogen.ec50 is None)
                              else self.pathogen.ec50),
            "note": self.note,
        }


def _log10_standard_error(result: DoseResponseResult) -> Optional[float]:
    """A log10 standard error read back off the interval the fit reported.

    The engine already chose between a profile-likelihood and a Wald interval
    and applied the right quantile; re-deriving a standard error from the
    covariance matrix here would silently use a different one and disagree
    with the interval printed beside it. So the half-width IS the source of
    truth, divided by the normal quantile at the same confidence.

    :param result: a fit with a closed log10 interval.
    :returns: the standard error, or ``None`` when either side is open.
    """
    low, high = result.log10_ec50_ci
    if low is None or high is None:
        return None
    quantile = float(stats.norm.ppf(0.5 + result.confidence / 2.0))
    if not np.isfinite(quantile) or quantile <= 0:
        return None
    width = float(high) - float(low)
    if not np.isfinite(width) or width <= 0:
        return None
    return width / (2.0 * quantile)


def selectivity_index(pathogen: Optional[DoseResponseResult],
                      host: Optional[DoseResponseResult], *,
                      confidence: Optional[float] = None
                      ) -> SelectivityIndex:
    """Divide a host EC50 by a parasite EC50, and carry the uncertainty.

    THE TWO FITS COME OFF ONE PLATE, which is the argument for computing this
    here rather than in a spreadsheet. spaCR segments host cell and pathogen
    as separate object types from the same image, so host viability and
    parasite burden are measured at the same doses in the same run. Most
    selectivity indices divide two numbers from two experiments and hope the
    conditions matched; these cannot fail to match.

    THE INTERVAL IS PROPAGATED IN LOG SPACE, where the fit lives and where a
    ratio is a difference: ``log10 SI = log10 CC50 - log10 EC50``, and the
    two variances add. Back-transforming at the end gives an interval that is
    asymmetric in linear space, which is the honest shape for a ratio.

    REFUSAL PROPAGATES TOO. If either curve was refused the index is refused;
    if either EC50 is unbounded the index is unbounded, and whichever side of
    the interval the data still supports is reported rather than dropped --
    "at least 8-fold" is a useful sentence and this returns it.

    :param pathogen: the parasite-burden fit, or ``None`` if it was refused.
    :param host: the host-viability fit, or ``None`` if it was refused.
    :param confidence: overrides the level carried by the fits.
    :returns: a :class:`SelectivityIndex`, never an exception, because a
        refusal is a result the caller has to show.
    """
    level = (confidence if confidence is not None
             else (host.confidence if host is not None
                   else pathogen.confidence if pathogen is not None
                   else DEFAULT_CONFIDENCE))
    if pathogen is None or host is None:
        missing = "parasite" if pathogen is None else "host"
        return SelectivityIndex(
            status=STATUS_REFUSED, index=None, index_low=None,
            index_high=None, log10_index=None, host=host, pathogen=pathogen,
            confidence=level,
            note=f"the {missing} curve was refused, so the ratio has no "
                 f"numerator or denominator to be a ratio of")

    log10_index = float(host.log10_ec50) - float(pathogen.log10_ec50)

    # ONE-SIDED RATHER THAN DROPPED. Interval arithmetic on whichever ends
    # survive: the smallest possible index divides the host's lower bound by
    # the parasite's upper one, and vice versa. Conservative, and it is the
    # only form available when a fit is open on one side.
    def _ratio(numerator, denominator):
        """One end of the interval, or ``None`` when that end is open.

        Returns ``None`` rather than raising or substituting a sentinel: an
        open side of a one-sided index is a fact about the experiment, and a
        number here would make it look bounded.

        :param numerator: a host EC50 bound, or ``None``.
        :param denominator: a parasite EC50 bound, or ``None``.
        """
        if numerator is None or denominator is None:
            return None
        if not np.isfinite(numerator) or not np.isfinite(denominator):
            return None
        if denominator <= 0:
            return None
        return float(numerator) / float(denominator)

    if not (host.ec50_bounded and pathogen.ec50_bounded):
        unbounded = ("host" if not host.ec50_bounded else "parasite")
        both = not host.ec50_bounded and not pathogen.ec50_bounded
        return SelectivityIndex(
            status=STATUS_UNBOUNDED, index=None,
            index_low=_ratio(host.ec50_low, pathogen.ec50_high),
            index_high=_ratio(host.ec50_high, pathogen.ec50_low),
            log10_index=log10_index, host=host, pathogen=pathogen,
            confidence=level,
            note=("neither EC50 is bounded by the concentrations tested"
                  if both else
                  f"the {unbounded} EC50 is not bounded by the "
                  f"concentrations tested, so the ratio is one-sided"))

    host_se = _log10_standard_error(host)
    pathogen_se = _log10_standard_error(pathogen)
    if host_se is None or pathogen_se is None:
        return SelectivityIndex(
            status=STATUS_UNBOUNDED, index=10.0 ** log10_index,
            index_low=_ratio(host.ec50_low, pathogen.ec50_high),
            index_high=_ratio(host.ec50_high, pathogen.ec50_low),
            log10_index=log10_index, host=host, pathogen=pathogen,
            confidence=level,
            note="one of the curves reported an open interval, so the ratio "
                 "carries interval arithmetic rather than a propagated one")

    combined = float(np.hypot(host_se, pathogen_se))
    quantile = float(stats.norm.ppf(0.5 + level / 2.0))
    half = quantile * combined
    return SelectivityIndex(
        status=STATUS_FITTED, index=10.0 ** log10_index,
        index_low=10.0 ** (log10_index - half),
        index_high=10.0 ** (log10_index + half),
        log10_index=log10_index, host=host, pathogen=pathogen,
        confidence=level, note="")


# ---------------------------------------------------------------------------
# Two-compound checkerboards: Bliss and Loewe
# ---------------------------------------------------------------------------

#: Bliss independence. Expects the two agents to act on independent targets,
#: so their surviving fractions multiply.
SYNERGY_BLISS = "bliss"

#: Loewe additivity. Expects the two agents to behave as dilutions of the
#: same agent, so a fixed effect costs a constant total dose.
SYNERGY_LOEWE = "loewe"

#: Every model :func:`interaction_surface` accepts.
SYNERGY_MODELS: Tuple[str, ...] = (SYNERGY_BLISS, SYNERGY_LOEWE)


@dataclass(frozen=True)
class InteractionSurface:
    """A checkerboard's interaction, per cell, with its own axes.

    THE SURFACE IS THE RESULT AND A SINGLE INDEX IS NOT. One number for a
    whole checkerboard hides exactly the concentration-dependent structure
    that makes synergy interesting. Real combinations are frequently
    synergistic
    in one corner of the grid and additive or antagonistic in another, and a
    mean over the grid reports neither.

    SIGN CONVENTION, stated because every paper states a different one:
    POSITIVE MEANS MORE EFFECT THAN EXPECTED -- synergy for an inhibition
    assay. The expected surface is what the model predicts from the two
    single-agent curves; ``excess`` is observed minus expected.

    :param model: :data:`SYNERGY_BLISS` or :data:`SYNERGY_LOEWE`.
    :param dose_a: the unique concentrations of agent A, ascending.
    :param dose_b: the unique concentrations of agent B, ascending.
    :param observed: ``(len(dose_a), len(dose_b))`` effect, 0 to 1.
    :param expected: what ``model`` predicts for each cell.
    :param excess: ``observed - expected``. NaN where a cell was not tested.
    :param n_cells: cells with an observation.
    :param note: what could not be computed, and why.
    """

    model: str
    dose_a: np.ndarray
    dose_b: np.ndarray
    observed: np.ndarray
    expected: np.ndarray
    excess: np.ndarray
    n_cells: int
    note: str = ""

    def summary(self) -> Dict[str, Any]:
        """Headline numbers, each said to be over the grid rather than of it.

        Deliberately NOT a synergy index. The strongest cell and where it sits
        are reportable; a mean over the whole grid is the number this class
        exists to avoid, so it is absent rather than provided-with-a-warning.
        """
        finite = np.isfinite(self.excess)
        if not finite.any():
            return {"model": self.model, "n_cells": 0, "note": self.note}
        values = self.excess[finite]
        flat = np.argmax(np.where(finite, self.excess, -np.inf))
        row, col = np.unravel_index(flat, self.excess.shape)
        return {
            "model": self.model,
            "n_cells": int(finite.sum()),
            "max_excess": float(values.max()),
            "max_at_dose_a": float(self.dose_a[row]),
            "max_at_dose_b": float(self.dose_b[col]),
            "min_excess": float(values.min()),
            "synergistic_cells": int((values > 0).sum()),
            "antagonistic_cells": int((values < 0).sum()),
            "note": self.note,
        }


def _effect_curve(result: DoseResponseResult) -> Callable[[np.ndarray], np.ndarray]:
    """A fitted curve as EFFECT in [0, 1], whichever way the response runs.

    A 4PL is fitted in response units and may rise or fall. Both synergy
    models are defined on the fraction affected, so the curve is rescaled
    against its own plateaus rather than against the data's extremes -- the
    plateaus are what the fit actually estimated.
    """
    bottom, top = float(result.bottom), float(result.top)
    span = top - bottom
    log10_ec50, hill = float(result.log10_ec50), float(result.hill)

    def effect(dose: np.ndarray) -> np.ndarray:
        """The affected fraction at each dose, in [0, 1].

        Zero and negative doses are clamped to a tiny positive number rather
        than refused: a checkerboard's first row IS zero, and a 4PL has no
        value there because log10(0) is undefined. The clamp puts them at the
        curve's own baseline, which is what an untreated well measures.

        :param dose: concentrations, any shape.
        :returns: affected fraction, same shape.
        """
        safe = np.where(np.asarray(dose, dtype=float) <= 0, 1e-12, dose)
        value = four_parameter_logistic(safe, bottom, top, log10_ec50, hill)
        if not np.isfinite(span) or span == 0:
            return np.zeros_like(safe, dtype=float)
        fraction = (value - bottom) / span
        # THE HILL SIGN CARRIES THE DIRECTION, and the two cases are not
        # symmetric. A negative Hill is inhibition: the response FALLS with
        # dose, so at a high dose `fraction` approaches 0 while the affected
        # fraction approaches 1, and the affected fraction is its complement.
        # A positive Hill is activation, where `fraction` already IS the
        # affected fraction and inverting it would report every activator as
        # its own antagonist.
        affected = (1.0 - fraction) if hill < 0 else fraction
        return np.clip(affected, 0.0, 1.0)

    return effect


def _grid(dose_a, dose_b, response) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold three parallel columns into a matrix, averaging repeats.

    :returns: ``(unique a, unique b, observed)``; unobserved cells are NaN,
        which is the honest value for a checkerboard corner nobody plated.
    """
    a = np.asarray(dose_a, dtype=float)
    b = np.asarray(dose_b, dtype=float)
    y = np.asarray(response, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b) & np.isfinite(y)
    a, b, y = a[keep], b[keep], y[keep]
    ua, ub = np.unique(a), np.unique(b)
    total = np.zeros((ua.size, ub.size))
    count = np.zeros((ua.size, ub.size))
    ia = np.searchsorted(ua, a)
    ib = np.searchsorted(ub, b)
    np.add.at(total, (ia, ib), y)
    np.add.at(count, (ia, ib), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        observed = np.where(count > 0, total / count, np.nan)
    return ua, ub, observed


@dataclass(frozen=True)
class Checkerboard:
    """A two-agent dose grid pulled out of a well table, with its two axes.

    The three surface functions take parallel arrays; a plate reader hands
    you a table. This is the join between them, and it keeps the single-agent
    rows -- the row where B is zero and the column where A is zero -- because
    :func:`bliss_surface` and :func:`loewe_surface` are both calibrated
    against those axes and a caller who filtered them out would silently get
    a surface with no reference.

    :param dose_a: agent A concentration per well, combination wells included.
    :param dose_b: agent B concentration per well.
    :param response: the measured response per well.
    :param a_alone: ``(dose, response)`` for the wells where B is zero.
    :param b_alone: ``(dose, response)`` for the wells where A is zero.
    """

    dose_a: np.ndarray
    dose_b: np.ndarray
    response: np.ndarray
    a_alone: Tuple[np.ndarray, np.ndarray]
    b_alone: Tuple[np.ndarray, np.ndarray]

    @property
    def shape(self) -> Tuple[int, int]:
        """How many distinct A doses by how many distinct B doses."""
        return (int(np.unique(self.dose_a).size),
                int(np.unique(self.dose_b).size))


def checkerboard_from_frame(frame: pd.DataFrame, *, dose_a: str, dose_b: str,
                            response: str) -> Checkerboard:
    """Read a checkerboard off a well table, single-agent axes and all.

    :param frame: one row per well.
    :param dose_a: column holding agent A's concentration.
    :param dose_b: column holding agent B's.
    :param response: column holding the measurement.
    :returns: a :class:`Checkerboard` ready for :func:`bliss_surface`,
        :func:`loewe_surface` and the single-agent fits they need.
    :raises DoseResponseError: when a column is missing, when either agent
        has no single-agent wells -- without them there is no curve to
        predict the combination from, and a surface computed against the
        combination wells themselves would be comparing the data to itself --
        or when no well has both agents present, which is a pair of dose
        series and not a checkerboard.
    """
    for column in (dose_a, dose_b, response):
        if column not in frame.columns:
            raise DoseResponseError(
                f"column {column!r} is not in the table; it has "
                f"{', '.join(map(str, frame.columns[:12]))}"
                f"{' ...' if len(frame.columns) > 12 else ''}")

    a = pd.to_numeric(frame[dose_a], errors="coerce").to_numpy(float)
    b = pd.to_numeric(frame[dose_b], errors="coerce").to_numpy(float)
    y = pd.to_numeric(frame[response], errors="coerce").to_numpy(float)
    keep = np.isfinite(a) & np.isfinite(b) & np.isfinite(y)
    a, b, y = a[keep], b[keep], y[keep]

    alone_a = (b == 0) & (a > 0)
    alone_b = (a == 0) & (b > 0)
    both = (a > 0) & (b > 0)
    for present, name, other in ((alone_a, dose_a, dose_b),
                                 (alone_b, dose_b, dose_a)):
        if not present.any():
            raise DoseResponseError(
                f"no well has {name} alone (with {other} at zero), so there "
                f"is no single-agent curve for it. Both surfaces predict the "
                f"combination FROM the single agents; without that row the "
                f"surface would be comparing the data to itself.")
    if not both.any():
        raise DoseResponseError(
            f"no well has both {dose_a} and {dose_b} above zero, so this is "
            f"two dose series rather than a checkerboard and there is no "
            f"interaction to measure")

    return Checkerboard(dose_a=a, dose_b=b, response=y,
                        a_alone=(a[alone_a], y[alone_a]),
                        b_alone=(b[alone_b], y[alone_b]))


def bliss_surface(dose_a, dose_b, response, *,
                  fit_a: DoseResponseResult,
                  fit_b: DoseResponseResult) -> InteractionSurface:
    """Bliss independence over a checkerboard.

    THE MODEL IN ONE LINE: if two agents act independently, the fraction
    surviving both is the product of the fractions surviving each, so the
    expected effect is ``Ea + Eb - Ea*Eb``. Excess over that is synergy.

    BLISS NEEDS NOTHING FROM THE COMBINATION FITS, which is why it is the
    cheaper of the two and the one to reach for first: both single-agent
    curves already give an effect at every concentration on the grid.

    :param dose_a: agent A concentration per well.
    :param dose_b: agent B concentration per well.
    :param response: measured response per well, same length.
    :param fit_a: the single-agent fit for A (B held at zero).
    :param fit_b: the single-agent fit for B.
    :returns: an :class:`InteractionSurface`.
    """
    ua, ub, observed = _grid(dose_a, dose_b, response)
    ea = _effect_curve(fit_a)(ua)[:, None]
    eb = _effect_curve(fit_b)(ub)[None, :]
    expected = ea + eb - ea * eb
    obs_effect = _observed_effect(observed, fit_a, fit_b)
    return InteractionSurface(
        model=SYNERGY_BLISS, dose_a=ua, dose_b=ub, observed=obs_effect,
        expected=expected, excess=obs_effect - expected,
        n_cells=int(np.isfinite(obs_effect).sum()))


def _observed_effect(observed: np.ndarray, fit_a: DoseResponseResult,
                     fit_b: DoseResponseResult) -> np.ndarray:
    """Measured response rescaled to effect, using the fits' own plateaus.

    Rescaled against the FITTED plateaus rather than the grid's own min and
    max, because a checkerboard's extremes are themselves measurements with
    noise in them -- normalising to them makes the strongest observed cell
    exactly 1.0 by construction and quietly caps the synergy it can report.
    """
    bottom = float(min(fit_a.bottom, fit_b.bottom))
    top = float(max(fit_a.top, fit_b.top))
    span = top - bottom
    if not np.isfinite(span) or span == 0:
        return np.full_like(observed, np.nan)
    fraction = (observed - bottom) / span
    inhibiting = float(fit_a.hill) < 0
    affected = (1.0 - fraction) if inhibiting else fraction
    return np.clip(affected, 0.0, 1.0)


def loewe_surface(dose_a, dose_b, response, *,
                  fit_a: DoseResponseResult,
                  fit_b: DoseResponseResult) -> InteractionSurface:
    """Loewe additivity over a checkerboard, as a combination index.

    THE MODEL IN ONE LINE: if two agents are dilutions of one another, then
    reaching an effect costs a constant total dose, so
    ``a/Da + b/Db = 1`` where ``Da`` and ``Db`` are the single-agent doses
    giving that same effect. Below 1 is synergy.

    REPORTED AS EXCESS RATHER THAN AS THE INDEX ITSELF, so that the sign
    convention matches Bliss and a reader comparing the two surfaces is not
    also flipping a comparison in their head: ``excess = 1 - CI``, positive
    for synergy.

    WHY IT CAN BE NaN WHERE BLISS IS NOT: Loewe needs the INVERSE curve --
    the dose achieving an observed effect -- and that dose does not exist
    when the observed effect lies outside a single agent's own plateaus. A
    combination that kills more than either agent can alone has no Loewe
    answer, and NaN is the honest one.

    :param dose_a: agent A concentration per well.
    :param dose_b: agent B concentration per well.
    :param response: measured response per well.
    :param fit_a: the single-agent fit for A.
    :param fit_b: the single-agent fit for B.
    """
    ua, ub, observed = _grid(dose_a, dose_b, response)
    effect = _observed_effect(observed, fit_a, fit_b)
    da = _dose_for_effect(fit_a, effect)
    db = _dose_for_effect(fit_b, effect)
    grid_a = ua[:, None] * np.ones_like(effect)
    grid_b = ub[None, :] * np.ones_like(effect)
    with np.errstate(invalid="ignore", divide="ignore"):
        index = np.where(da > 0, grid_a / da, np.nan) + \
                np.where(db > 0, grid_b / db, np.nan)
    # THE UNTREATED WELL IS NOT INFINITELY SYNERGISTIC. With both doses at
    # zero the index is 0 and the excess reads +1.0, the strongest possible
    # synergy, from the one well where nothing was combined. Measured on a
    # simulated board it was the maximum of the whole surface. Loewe is
    # undefined without a combination, so that cell is NaN.
    index = np.where((grid_a > 0) | (grid_b > 0), index, np.nan)
    excess = 1.0 - index
    note = ("cells where the observed effect lies outside a single agent's "
            "plateaus have no Loewe answer and are NaN")
    return InteractionSurface(
        model=SYNERGY_LOEWE, dose_a=ua, dose_b=ub, observed=effect,
        expected=np.ones_like(index), excess=excess,
        n_cells=int(np.isfinite(excess).sum()), note=note)


def _dose_for_effect(result: DoseResponseResult,
                     effect: np.ndarray) -> np.ndarray:
    """Invert a 4PL: the dose giving each effect, or NaN outside its range.

    NaN RATHER THAN AN EXTRAPOLATION. The inverse of a logistic runs to
    infinity at its plateaus, so an effect at or past one of them has no
    finite dose. Returning a very large number instead would make a
    combination index look enormous and finite when the truth is that the
    question has no answer for that cell.
    """
    e = np.clip(np.asarray(effect, dtype=float), 0.0, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = e / (1.0 - e)
        log10_dose = float(result.log10_ec50) + \
            np.log10(ratio) / abs(float(result.hill))
    dose = np.where(np.isfinite(log10_dose), 10.0 ** log10_dose, np.nan)
    return np.where((e > 0) & (e < 1), dose, np.nan)


# ---------------------------------------------------------------------------
# The plate: normalisation to its controls, and the Z' it already has
# ---------------------------------------------------------------------------

#: Leave the response column alone. The default, because a table that is
#: already percent inhibition must not be normalised twice.
NORMALISE_NONE = "none"

#: Percent inhibition against each plate's own controls: the negative control
#: reads 0 and the positive control reads 100, whatever the raw units were.
NORMALISE_PERCENT = "percent_inhibition"

#: Every normalisation this module offers.
NORMALISATIONS = (NORMALISE_NONE, NORMALISE_PERCENT)

#: Column :func:`normalise_to_controls` writes when the caller names no other.
PERCENT_COLUMN = "percent_inhibition"

#: The Z' below which a plate is conventionally called unusable. Offered as a
#: default for :attr:`PlateSpec.min_zprime`, never applied unless the caller
#: asks for it -- a threshold nobody chose is a threshold nobody can defend.
ZPRIME_MARGINAL = 0.5


@dataclass(frozen=True)
class PlateSpec:
    """Which column is the plate, and which wells on it are the controls.

    THE ENGINE HAD NO NOTION OF A PLATE, which is why it could fit a clean
    EC50 on a plate the rest of the package already knew was bad. This is
    that notion: the plate column, the control column, and the levels in it
    that mean "full effect" and "no effect".

    Frozen and JSON round-tripping like :class:`DoseResponseSpec`, so the
    normalisation behind a figure travels with the fit that used it.

    :param plate: column identifying the plate. Required.
    :param control: column naming each well's control role. Required.
    :param positive: levels of ``control`` that are the positive control --
        the full-effect wells, which normalise to 100.
    :param negative: levels that are the negative control -- vehicle or
        untreated, which normalise to 0.
    :param min_zprime: refuse every plate whose Z' falls below this, and say
        the Z' in the refusal. ``None`` (default) gates nothing and still
        reports the Z' beside each plate. :data:`ZPRIME_MARGINAL` is the
        conventional 0.5 if you want one.
    :raises DoseResponseError: when a column or a control level is missing,
        at the point the spec is built rather than halfway through a plate.
    """

    plate: str = ""
    control: str = ""
    positive: Tuple[str, ...] = ()
    negative: Tuple[str, ...] = ()
    min_zprime: Optional[float] = None

    def __post_init__(self) -> None:
        """Normalise the names and insist both controls exist.

        :raises DoseResponseError: when ``plate`` or ``control`` is blank, or
            when either control has no levels -- percent inhibition is a
            two-point scale and one control cannot define it.
        """
        object.__setattr__(self, "plate", str(self.plate or "").strip())
        object.__setattr__(self, "control", str(self.control or "").strip())
        object.__setattr__(self, "positive",
                           tuple(str(level) for level in self.positive))
        object.__setattr__(self, "negative",
                           tuple(str(level) for level in self.negative))
        if not self.plate:
            raise DoseResponseError(
                "plate normalisation needs the column that identifies the "
                "plate; without it every well on every plate would be scaled "
                "by one pooled pair of controls, which is the error "
                "normalising per plate exists to prevent")
        if not self.control:
            raise DoseResponseError(
                "plate normalisation needs control_column set, so the "
                "positive and negative levels have a column to be levels of")
        if not self.positive or not self.negative:
            raise DoseResponseError(
                "percent inhibition is a two-point scale and needs BOTH "
                "controls named: positive (full effect, reads 100) and "
                "negative (vehicle, reads 0). With one of them there is an "
                "offset but no assay window to divide by.")
        if self.min_zprime is not None:
            gate = float(self.min_zprime)
            if not np.isfinite(gate):
                raise DoseResponseError(
                    f"min_zprime must be a finite number, not {self.min_zprime!r}")
            object.__setattr__(self, "min_zprime", gate)

    def to_json(self) -> Dict[str, Any]:
        """A plain dict, for a settings file or a methods section."""
        return {
            "plate": self.plate,
            "control": self.control,
            "positive": list(self.positive),
            "negative": list(self.negative),
            "min_zprime": self.min_zprime,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "PlateSpec":
        """Rebuild from :meth:`to_json`, validating on the way in.

        :param payload: the mapping :meth:`to_json` produced. Missing keys
            fall back to the field defaults rather than raising, so a spec
            written by an older version still loads.
        """
        gate = payload.get("min_zprime")
        return cls(plate=str(payload.get("plate", "")),
                   control=str(payload.get("control", "")),
                   positive=tuple(payload.get("positive", ()) or ()),
                   negative=tuple(payload.get("negative", ()) or ()),
                   min_zprime=None if gate is None else float(gate))


@dataclass(frozen=True)
class PlateReport:
    """What one plate's controls said, and whether the plate may be fitted.

    ONE ROW PER PLATE, REFUSALS INCLUDED. A plate that is dropped leaves a
    report saying why it was dropped, so a user comparing eight plates and
    seeing six curves can find the other two without re-running anything.

    :param status: :data:`STATUS_FITTED` when the plate normalised, or
        :data:`STATUS_REFUSED` -- the same vocabulary the curve fits speak.
    :param zprime: the plate's Z-factor, or ``None`` when it has no Z'
        because a control had fewer than two wells. ``None`` is not a
        failure; it is the absence of a number, and is reported as such.
    :param note: the sentence to show the user. Empty when nothing is wrong.
    :param plate: the plate identifier this report is about, as it appears in
        the plate column of the frame.
    :param n_positive: how many positive-control wells were found on it.
        Reported even when the plate is refused, because "two" and "none"
        are different problems and the note alone does not distinguish them.
    :param n_negative: the same for negative controls.
    """

    plate: str
    status: str
    n_positive: int
    n_negative: int
    mean_positive: Optional[float] = None
    mean_negative: Optional[float] = None
    separation: Optional[float] = None
    zprime: Optional[float] = None
    note: str = ""

    @property
    def usable(self) -> bool:
        """Whether wells on this plate carry a normalised response."""
        return self.status == STATUS_FITTED

    def summary_row(self) -> Dict[str, Any]:
        """One row for the plate table, refusal included."""
        blank = float("nan")
        return {
            "plate": self.plate,
            "status": self.status,
            "zprime": blank if self.zprime is None else self.zprime,
            "mean_positive": (blank if self.mean_positive is None
                              else self.mean_positive),
            "mean_negative": (blank if self.mean_negative is None
                              else self.mean_negative),
            "separation": blank if self.separation is None else self.separation,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "note": self.note,
        }


def _zprime_by_plate(frame: pd.DataFrame, spec: PlateSpec,
                     response: str) -> Dict[str, float]:
    """Per-plate Z', computed by the module that already computes it.

    CALLS :func:`spacr.qt.widgets.control_chart.zprime_frame` RATHER THAN
    REPEATING THE FORMULA. Two screens computing Z' two ways would be worse
    than the gap this closes: the Control Chart screen would show 0.62 and
    Dose-Response would refuse the same plate at 0.48, and no user could tell
    which one to believe. The import is local because the only thing
    dose-response needs from that module is this one function.

    :returns: plate label to Z'. Plates whose controls have fewer than two
        wells are absent -- ``zprime_frame`` leaves them out rather than
        giving them a zero, and inventing one here would undo that. A table
        where NO plate has a Z' comes back empty rather than raising: that is
        an error for a Z' chart, which would have nothing to draw, but it is
        an ordinary state for normalisation, which never needed a Z' to scale
        a plate to its own controls.
    """
    from .control_chart import ControlChartError, ControlChartSpec
    from .control_chart import ZPRIME_PLATE, ZPRIME_VALUE, zprime_frame

    chart = ControlChartSpec(
        value=response,
        plate=spec.plate,
        control_column=spec.control,
        control_levels=tuple(spec.positive) + tuple(spec.negative),
        positive_levels=tuple(spec.positive),
        negative_levels=tuple(spec.negative),
    )
    try:
        table = zprime_frame(frame, chart)
    except ControlChartError:
        return {}
    return {str(row[ZPRIME_PLATE]): float(row[ZPRIME_VALUE])
            for _, row in table.iterrows()}


def normalise_to_controls(frame: pd.DataFrame, spec: PlateSpec, *,
                          response: str,
                          out: str = PERCENT_COLUMN,
                          ) -> Tuple[pd.DataFrame, Tuple[PlateReport, ...]]:
    """Percent inhibition against each plate's own controls.

    RAW RESPONSES ARE NOT COMPARABLE ACROSS PLATES. Two plates read on
    different days differ in absolute signal by more than most compounds
    move it, so three replicate plates fitted raw produce three EC50s whose
    spread is mostly instrument drift. Scaling each plate to its own controls
    -- negative reads 0, positive reads 100 -- removes exactly that and
    leaves the biology.

    ``percent = 100 * (value - mean_negative) / (mean_positive - mean_negative)``

    The formula is signed and direction-agnostic on purpose: whichever way
    the raw readout runs, the positive control reads 100 by construction, so
    a viability readout and a burden readout normalise the same way and the
    fit downstream does not need to be told which it got.

    REFUSED RATHER THAN SCALED BY NOISE. When a plate's two controls do not
    separate there is no assay window, and dividing by that near-zero
    difference would turn well-to-well noise into hundreds of percent
    inhibition and a confident EC50 on a plate that measured nothing. Such a
    plate's rows come back with a NaN response and a :class:`PlateReport`
    saying so.

    :param frame: one row per well, with the plate, control and response
        columns the spec and this call name.
    :param spec: the plate, its control column and its two control levels.
    :param response: the raw column to normalise.
    :param out: column to write the percent into. Defaults to
        :data:`PERCENT_COLUMN`; pass another name to keep several readouts
        (host viability and parasite burden, say) side by side.
    :returns: ``(frame, reports)`` -- a copy of the frame carrying ``out``,
        and one :class:`PlateReport` per plate in the order the plates first
        appear. Rows on a refused plate carry NaN, so a caller that fits the
        whole table drops those plates without having to filter first.
    :raises DoseResponseError: when a named column is missing, or when no
        plate has both controls -- there is nothing to normalise against and
        a frame of NaN would be a worse answer than a sentence.
    """
    percent, reports = _scan_plates(frame, spec, response)
    if not any(report.usable for report in reports):
        raise DoseResponseError(
            "no plate in this table has a usable pair of controls, so there "
            "is nothing to normalise against. " +
            (reports[0].note if reports else
             f"no plate was found in column {spec.plate!r}."))
    normalised = frame.copy()
    normalised[out] = percent
    return normalised, reports


def _scan_plates(frame: pd.DataFrame, spec: PlateSpec, response: str
                 ) -> Tuple[np.ndarray, Tuple[PlateReport, ...]]:
    """Walk the plates once: the percent column and the verdict per plate.

    THE ONE PLACE THE RULES LIVE, so :func:`normalise_to_controls` and
    :func:`plate_reports` cannot disagree about whether a plate is usable --
    which they would, sooner or later, if each carried its own copy.

    :returns: ``(percent, reports)``, the percent array aligned to the frame's
        rows and NaN wherever its plate was refused.
    """
    for column in (spec.plate, spec.control, response):
        if column not in frame.columns:
            raise DoseResponseError(
                f"column {column!r} is not in the table; it has "
                f"{', '.join(map(str, frame.columns[:12]))}"
                f"{' ...' if len(frame.columns) > 12 else ''}")

    values = pd.to_numeric(frame[response], errors="coerce").to_numpy(float)
    plates = frame[spec.plate].astype(str).to_numpy()
    roles = frame[spec.control].astype(str).to_numpy()
    positive = set(spec.positive)
    negative = set(spec.negative)

    zprimes = _zprime_by_plate(frame, spec, response)

    percent = np.full(values.shape, np.nan, dtype=float)
    reports: List[PlateReport] = []
    seen: List[str] = []
    for plate in plates:
        if plate not in seen:
            seen.append(plate)

    for plate in seen:
        on_plate = plates == plate
        pos = values[on_plate & np.isin(roles, list(positive))]
        neg = values[on_plate & np.isin(roles, list(negative))]
        pos = pos[np.isfinite(pos)]
        neg = neg[np.isfinite(neg)]
        zprime = zprimes.get(plate)
        common = dict(plate=plate, n_positive=int(pos.size),
                      n_negative=int(neg.size), zprime=zprime)

        if pos.size == 0 or neg.size == 0:
            missing = ("positive" if pos.size == 0 else "negative")
            if pos.size == 0 and neg.size == 0:
                missing = "positive and negative"
            reports.append(PlateReport(
                status=STATUS_REFUSED,
                note=(f"plate {plate} has no {missing} control well with a "
                      f"finite {response}; percent inhibition is measured "
                      f"against this plate's own controls and there are none "
                      f"to measure against"),
                **common))
            continue

        mean_pos = float(pos.mean())
        mean_neg = float(neg.mean())
        separation = mean_pos - mean_neg
        common.update(mean_positive=mean_pos, mean_negative=mean_neg,
                      separation=abs(separation))

        if not np.isfinite(separation) or separation == 0.0:
            reports.append(PlateReport(
                status=STATUS_REFUSED,
                note=(f"plate {plate} has no assay window: its positive and "
                      f"negative controls both read {mean_pos:.4g}, so there "
                      f"is nothing to scale by. Dividing by that difference "
                      f"would report noise as percent inhibition."),
                **common))
            continue

        gate = spec.min_zprime
        if gate is not None and zprime is None:
            reports.append(PlateReport(
                status=STATUS_REFUSED,
                note=(f"plate {plate} cannot be gated on Z': a Z-factor needs "
                      f"at least two wells of each control for the SDs to "
                      f"exist and this plate has {pos.size} positive and "
                      f"{neg.size} negative. Drop min_zprime to normalise it "
                      f"ungated, or fill the control wells."),
                **common))
            continue
        if gate is not None and zprime < gate:
            shown = "-inf" if not np.isfinite(zprime) else f"{zprime:.3g}"
            reports.append(PlateReport(
                status=STATUS_REFUSED,
                note=(f"plate {plate} fails Z': {shown}, below the {gate:g} "
                      f"asked for. A plate this noisy could not have detected "
                      f"the effect, so an EC50 fitted on it would be a number "
                      f"about the instrument, not the compound."),
                **common))
            continue

        percent[on_plate] = 100.0 * (values[on_plate] - mean_neg) / separation
        note = ""
        if zprime is None:
            note = (f"plate {plate} normalised, but has no Z': a Z-factor "
                    f"needs two wells of each control and this plate has "
                    f"{pos.size} positive and {neg.size} negative.")
        elif np.isfinite(zprime) and zprime < ZPRIME_MARGINAL:
            note = (f"plate {plate} normalised with Z' {zprime:.3g}, below "
                    f"the conventional {ZPRIME_MARGINAL:g}. Nothing was "
                    f"gated -- set min_zprime if it should have been.")
        reports.append(PlateReport(status=STATUS_FITTED, note=note, **common))

    return percent, tuple(reports)


def plate_reports(frame: pd.DataFrame, spec: PlateSpec, *,
                  response: str) -> Tuple[PlateReport, ...]:
    """The per-plate verdict alone, without normalising anything.

    For the screen that wants to show the plate table before the user has
    chosen a readout to fit, and for a caller that only wants to know which
    plates would be dropped. A table where every plate is refused returns its
    refusals rather than raising -- here the refusals ARE the answer.

    :param frame: the long-format table, one row per well.
    :param spec: which column names the plate, which the control, and which
        control values are the two ends.
    :param response: the readout column whose control means decide the
        plate's Z'. Keyword-only, because a plate's verdict is about a
        PARTICULAR readout and passing it positionally invites reading the
        report as a property of the plate alone.
    :raises DoseResponseError: when a named column is missing.
    """
    _, reports = _scan_plates(frame, spec, response)
    return reports


# ---------------------------------------------------------------------------
# Replicate plates: one EC50, with plate as a random effect
# ---------------------------------------------------------------------------

#: Above this share of the spread being real rather than sampling noise, a
#: pooled EC50 is refused. I-squared is the fraction of the between-plate
#: variance that the plates' own uncertainty does NOT explain, so 0.9 means
#: nine tenths of the disagreement is the plates genuinely disagreeing.
MAX_HETEROGENEITY = 0.9

#: Fewest plates a pooled fit will accept. Two plates give a pooled estimate
#: whose between-plate variance is estimated from one degree of freedom, which
#: is a number but not a measurement of anything.
MIN_PLATES = 2


@dataclass(frozen=True)
class PooledFit:
    """One EC50 across replicate plates, with plate as a random effect.

    THREE EC50s AND AN EYEBALL is what this replaces. A user with three
    replicate plates today fits three curves and averages the numbers by
    hand, which throws away how well each one was determined and says
    nothing about whether the three agreed.

    TWO-STAGE, NOT ONE. Each plate is fitted on its own -- by the same
    :func:`fit_dose_response` that fits everything else, with the same
    refusals -- and the per-plate log10 EC50s are then combined with a
    random-effects weight. One joint nonlinear mixed model would be the other
    way to do it; it would also mean a second fitting path with a second set
    of failure modes, and a plate that :func:`fit_dose_response` refuses would
    have to be refused again, differently, inside it. This way a refusal on
    one plate stays exactly the refusal this module already speaks.

    RANDOM, NOT FIXED. Fixed-effect pooling assumes every plate measures the
    same value and differs only by noise. Three tight plates that disagree
    then give a narrow interval around a value none of them support. The
    DerSimonian-Laird estimate of the between-plate variance is added to each
    plate's own, so real variation between plates widens the answer instead
    of being weighted away.

    :param status: :data:`STATUS_FITTED`, or :data:`STATUS_REFUSED` when too
        few plates fitted or the plates disagree beyond what their own
        uncertainty explains.
    :param ec50: the pooled EC50, or ``None`` when refused.
    :param tau: the between-plate SD on the log10 scale -- it shows how far
        the plates agree about this compound, which no average of three EC50
        values can report.
    :param i_squared: the share of the observed spread that is real rather
        than sampling noise, in ``[0, 1]``.
    :param q: Cochran's Q against the null that every plate measured the
        same EC50.
    :param q_p: the p-value of that Q.
    :param per_plate: the individual fits, kept so the pooled number can
        always be taken apart again.
    :param note: why, when refused or when the plates sit uneasily together.
    :param log10_ec50: the pooled estimate on the log10 scale, which is where
        the pooling is actually done -- EC50s are log-normal, so averaging
        them in linear units weights the high plates more than the data
        warrants.
    :param log10_se: the standard error of that estimate, on the same scale.
    :param ec50_low: the low end of the confidence interval, back on the
        linear scale the user reads.
    :param ec50_high: its high end.
    """

    status: str
    ec50: Optional[float]
    ec50_low: Optional[float]
    ec50_high: Optional[float]
    log10_ec50: Optional[float]
    log10_se: Optional[float]
    tau: Optional[float]
    i_squared: Optional[float]
    q: Optional[float]
    q_p: Optional[float]
    per_plate: Tuple[Tuple[str, DoseResponseResult], ...] = ()
    n_plates: int = 0
    n_used: int = 0
    confidence: float = DEFAULT_CONFIDENCE
    unit: str = ""
    note: str = ""

    @property
    def reproducible(self) -> bool:
        """Whether the plates agreed well enough for the pooled number."""
        return self.status == STATUS_FITTED

    def summary_row(self) -> Dict[str, Any]:
        """One row for the results table, refusal included."""
        blank = float("nan")

        def num(value):
            """``None`` as NaN, so a refused fit still fills its columns.

            A refusal has no EC50, and leaving the cell empty would make the
            row a different shape from a fitted one -- which is what a table
            cannot have. NaN is the value that says "not a number here"
            without changing the columns.
            """
            return blank if value is None else float(value)
        return {
            "metric": "pooled_ec50",
            "status": self.status,
            "ec50": num(self.ec50),
            "ec50_low": num(self.ec50_low),
            "ec50_high": num(self.ec50_high),
            "tau_log10": num(self.tau),
            "i_squared": num(self.i_squared),
            "q": num(self.q),
            "q_p": num(self.q_p),
            "n_plates": self.n_plates,
            "n_used": self.n_used,
            "unit": self.unit,
            "note": self.note,
        }


def _refused_pool(note: str, per_plate, n_plates, n_used,
                  confidence, unit, **extra) -> "PooledFit":
    """A :class:`PooledFit` that carries only the reason it is not one."""
    fields = dict(ec50=None, ec50_low=None, ec50_high=None, log10_ec50=None,
                  log10_se=None, tau=None, i_squared=None, q=None, q_p=None)
    fields.update(extra)
    return PooledFit(status=STATUS_REFUSED, per_plate=tuple(per_plate),
                     n_plates=n_plates, n_used=n_used,
                     confidence=confidence, unit=unit, note=note, **fields)


def pool_across_plates(fits: Mapping[str, DoseResponseResult], *,
                       confidence: float = DEFAULT_CONFIDENCE,
                       max_heterogeneity: float = MAX_HETEROGENEITY,
                       ) -> PooledFit:
    """Combine per-plate fits into one EC50 with plate as a random effect.

    POOLED ON THE LOG10 SCALE, because that is the scale the EC50 is
    estimated on and the scale its interval is symmetric on. Averaging three
    EC50s of 1, 10 and 100 uM arithmetically gives 37 uM; pooling their
    logarithms gives 10, which is the middle of the three in the only sense
    that matters for a concentration.

    :param fits: plate label to that plate's fit. Only fits that are
        :data:`STATUS_FITTED` with a closed interval can carry a weight; the
        rest are counted, named in the note and left out of the arithmetic,
        because a plate whose EC50 is unbounded has no variance to weight by
        and dropping it silently would make the pooled interval look better
        than the experiment was.
    :param confidence: coverage for the pooled interval.
    :param max_heterogeneity: refuse above this I-squared.
    :returns: a :class:`PooledFit`, refused rather than empty when the plates
        cannot support a single number.
    :raises DoseResponseError: when ``max_heterogeneity`` is not in ``(0, 1]``.
    """
    if not 0.0 < float(max_heterogeneity) <= 1.0:
        raise DoseResponseError(
            "max_heterogeneity is a share of the spread and must be in "
            f"(0, 1], not {max_heterogeneity}")

    ordered = tuple((str(plate), result) for plate, result in fits.items())
    n_plates = len(ordered)
    unit = next((r.unit for _, r in ordered if r.unit), "")

    usable, dropped = [], []
    for plate, result in ordered:
        se = _log10_standard_error(result)
        if result.status != STATUS_FITTED or se is None or se <= 0:
            dropped.append(plate)
            continue
        usable.append((plate, float(result.log10_ec50), float(se)))

    if len(usable) < MIN_PLATES:
        missing = (f" ({', '.join(dropped)} did not fit to a closed interval)"
                   if dropped else "")
        return _refused_pool(
            f"pooling needs at least {MIN_PLATES} plates with a bounded EC50 "
            f"and this has {len(usable)} of {n_plates}{missing}. One plate is "
            f"not a replicate; report its own fit instead.",
            ordered, n_plates, len(usable), confidence, unit)

    effects = np.asarray([value for _, value, _ in usable], dtype=float)
    variances = np.asarray([se ** 2 for _, _, se in usable], dtype=float)

    # Stage one: fixed-effect weights, only to measure the disagreement.
    fixed_w = 1.0 / variances
    fixed_mean = float(np.sum(fixed_w * effects) / np.sum(fixed_w))
    q = float(np.sum(fixed_w * (effects - fixed_mean) ** 2))
    dof = len(usable) - 1
    q_p = float(stats.chi2.sf(q, dof)) if dof > 0 else float("nan")

    # DerSimonian--Laird: the spread the plates' own uncertainty cannot explain.
    c = float(np.sum(fixed_w) - np.sum(fixed_w ** 2) / np.sum(fixed_w))
    tau_squared = max(0.0, (q - dof) / c) if c > 0 else 0.0
    tau = float(np.sqrt(tau_squared))
    i_squared = float(max(0.0, (q - dof) / q)) if q > 0 else 0.0

    if i_squared > float(max_heterogeneity):
        spread = 10.0 ** (float(effects.max()) - float(effects.min()))
        return _refused_pool(
            f"the {len(usable)} plates disagree beyond what their own "
            f"uncertainty explains (I-squared {i_squared:.0%}, Q={q:.3g} on "
            f"{dof} df, p={q_p:.3g}): their EC50s span a factor of "
            f"{spread:.3g}. One pooled number would hide that, and the "
            f"disagreement is the finding -- look for a plate effect before "
            f"averaging it away.",
            ordered, n_plates, len(usable), confidence, unit,
            tau=tau, i_squared=i_squared, q=q, q_p=q_p)

    weights = 1.0 / (variances + tau_squared)
    pooled = float(np.sum(weights * effects) / np.sum(weights))
    pooled_se = float(np.sqrt(1.0 / np.sum(weights)))
    quantile = float(stats.norm.ppf(0.5 + float(confidence) / 2.0))
    low = pooled - quantile * pooled_se
    high = pooled + quantile * pooled_se

    note = ""
    if dropped:
        note = (f"{len(dropped)} of {n_plates} plates carried no weight "
                f"({', '.join(dropped)}): an EC50 the plate does not bound "
                f"has no variance to weight by.")
    if tau > 0.0:
        spacing = "; " if note else ""
        note += (f"{spacing}plate-to-plate SD is {tau:.3g} on log10, a factor "
                 f"of {10.0 ** tau:.3g} in EC50, and is included in the "
                 f"interval rather than weighted away.")

    return PooledFit(
        status=STATUS_FITTED,
        ec50=10.0 ** pooled, ec50_low=10.0 ** low, ec50_high=10.0 ** high,
        log10_ec50=pooled, log10_se=pooled_se,
        tau=tau, i_squared=i_squared, q=q, q_p=q_p,
        per_plate=ordered, n_plates=n_plates, n_used=len(usable),
        confidence=confidence, unit=unit, note=note)


def pool_frame(frame: pd.DataFrame, spec: DoseResponseSpec, *,
               plate: str,
               max_heterogeneity: float = MAX_HETEROGENEITY,
               ) -> PooledFit:
    """Fit each plate in ``frame`` on its own, then pool them.

    The convenience over :func:`pool_across_plates` for the common case: one
    table, one compound, a plate column. A plate that raises
    :class:`DoseResponseError` is kept out of the pool and named in the note,
    exactly as :func:`fit_frame` keeps one bad compound from taking a plate
    down.

    :param frame: the long-format table, one row per well, with every
        replicate plate in it.
    :param spec: the fit specification, applied unchanged to every plate --
        which is what makes the per-plate EC50s comparable in the first
        place.
    :param plate: the column identifying the replicate.
    :raises DoseResponseError: when ``plate`` is not a column, or when no
        plate produced a fit at all -- there is nothing to pool and a refusal
        with no plates in it would say nothing about why.
    """
    if plate not in frame.columns:
        raise DoseResponseError(
            f"column {plate!r} is not in the table, so there are no "
            f"replicates to pool across")

    fits: Dict[str, DoseResponseResult] = {}
    failures: List[str] = []
    for label, rows in frame.groupby(frame[plate].astype(str), sort=False):
        try:
            fits[str(label)] = fit_dose_response(
                rows[spec.concentration], rows[spec.response], spec,
                group=str(label))
        except DoseResponseError as failure:
            failures.append(f"{label}: {failure}")

    if not fits:
        raise DoseResponseError(
            "no plate in this table produced a fit, so there is nothing to "
            "pool. " + (" | ".join(failures) if failures else
                        f"column {plate!r} held no groups."))

    pooled = pool_across_plates(fits, confidence=spec.confidence,
                                max_heterogeneity=max_heterogeneity)
    if failures:
        extra = (f"{len(failures)} plate(s) did not fit at all: "
                 f"{' | '.join(failures)}")
        pooled = replace(pooled,
                         note=f"{pooled.note}; {extra}" if pooled.note else extra)
    return pooled
