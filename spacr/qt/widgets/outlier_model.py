"""Robust outlier detection over a measurement table — per object *and* per well.

Not a z-score
=============

The obvious thing to write here is ``|x - mean| / sd > 3``, and it is the one
thing this module refuses to do. spaCR object measurements are skewed and
heavy-tailed by construction: ``cell_area`` is a positive quantity with a long
right tail, ``pathogen_area`` is zero-inflated, an intensity ratio has a
denominator that can approach zero. The mean and the SD of such a column are
themselves dragged by the very objects they are supposed to find — one debris
particle measured at 40× the median area moves the mean, inflates the SD, and
so raises the threshold that was meant to catch it. The failure is silent and
it gets worse the worse the contamination is: this is the *masking* effect,
and it is the reason robust statistics exist.

Every method here therefore estimates the centre and the spread from
statistics that a minority of arbitrary values cannot move: the median, the
quartiles, and — in several dimensions — the Minimum Covariance Determinant.

Three methods, one idea
=======================

1. MAD — :data:`METHOD_MAD`
---------------------------
Flag when ``|x - median(x)| / (1.4826 * MAD) > k``, default ``k = 3.5``
(:data:`DEFAULT_MAD_K`), where ``MAD = median(|x - median(x)|)``.

**The 1.4826.** For a Gaussian, ``MAD = Phi^-1(0.75) * sigma = 0.6745 * sigma``,
so ``sigma = MAD / 0.6745 = 1.4826 * MAD``. Multiplying by
:data:`MAD_TO_SIGMA` makes the MAD a *consistent estimator of sigma* under
normality, which is what lets the threshold be read on the familiar scale —
"3.5 SD" — while never being computed from an SD that the outliers could
inflate. The constant is the whole reason ``k`` is comparable to a z-score
cut-off at all; without it, ``k`` would be in units of "MADs" and 3.5 would
mean something different for every column.

**MAD == 0 is handled, not divided by.** When more than half the objects share
one value — a zero-inflated ``pathogen_area``, an integer count, a saturated
intensity — the MAD is exactly zero. Naively, every value that differs at all
then scores infinity and *the entire tail is flagged*, which is the opposite
of robust. So :func:`robust_scale` falls back to the **mean absolute deviation
from the median**, scaled by :data:`MEAN_AD_TO_SIGMA` = ``sqrt(pi/2)`` =
1.2533 (for a Gaussian ``E|X - mu| = sigma * sqrt(2/pi)``), and says so in
:attr:`OutlierResult.notes` and in :meth:`OutlierResult.caveats`.

The alternative considered and not taken is the Rousseeuw–Croux ``Sn``
(``1.1926 * med_i med_j |x_i - x_j|``), which is the statistically better
answer: it has a 50% breakdown point, needs no symmetry assumption, and does
not collapse on tied data. It is not used because the naive form is O(n^2) —
unusable on the 10^5–10^6 object rows spaCR routinely produces — and the
O(n log n) algorithm is a page of intricate code to maintain for a path that
is only ever reached when the MAD has *already* collapsed. The mean absolute
deviation has a 0% breakdown point and that is stated plainly rather than
glossed: it is a stopgap that keeps a degenerate column from flagging its
whole tail, not a robust estimator. If the fallback fires, the honest reading
is that the column is not continuous enough for this test.

A column with **no** variation at all (mean absolute deviation zero too) is
scored 0 everywhere and flagged nowhere. One value cannot be an outlier from
itself.

2. IQR / Tukey — :data:`METHOD_IQR`
-----------------------------------
Flag outside ``[Q1 - c*IQR, Q3 + c*IQR]``, default ``c = 1.5``
(:data:`DEFAULT_IQR_C`).

**The asymmetry problem is real and is not fixed by choosing a bigger c.**
Tukey's fence is symmetric in the quartiles, and a right-skewed distribution
therefore has far more mass beyond the upper fence than beyond the lower one
*by construction*, whatever the data. For a lognormal with sigma = 1 the
standard fence flags about 8% of the sample on the right and essentially
nothing on the left — and it does so for a perfectly clean sample. Read
uncorrected, "8% of my cells are outliers" is a statement about the shape of
the distribution, not about the cells.

The offered remedy is a **log10 transform**, :data:`TRANSFORM_LOG10`, which
turns a lognormal into a Gaussian and makes the fence mean what it appears to
mean. It is **never implicit**: a transform silently applied would change what
``outlier_score`` is in units of without the user asking. And it **refuses**
rather than fudges on non-positive values — ``log10`` of a zero
``pathogen_area`` is ``-inf``, and an implicit ``+1`` or a dropped row would
be an invented measurement or a self-selected population. The refusal names
the feature and the count.

:meth:`OutlierResult.caveats` says the asymmetry out loud whenever the IQR
rule ran untransformed and the flags landed lopsidedly.

3. Robust Mahalanobis on an MCD covariance — :data:`METHOD_MAHALANOBIS`
-----------------------------------------------------------------------
Fit :class:`sklearn.covariance.MinCovDet` — the Minimum Covariance Determinant
— to the selected features, take each object's squared Mahalanobis distance to
the robust centre in the robust metric, and flag it against a chi-square
quantile.

**Why MCD and not an isolation forest.** Both are "multivariate outlier
detectors" and they are not interchangeable:

* **The threshold is a stated false-positive rate, not a tuned fraction.**
  Under multivariate normality the squared robust Mahalanobis distance is
  distributed chi-square with ``p`` degrees of freedom, so the cut point is
  ``chi2.ppf(1 - alpha, p)`` for a chosen ``alpha`` — default
  :data:`DEFAULT_ALPHA` = 0.001, meaning *one clean object in a thousand is
  expected to be flagged*. Over 10^5 objects that is about 100 false flags,
  and over the 200,000 objects of a typical two-plate screen it is about 200.
  That number is knowable in advance and is printed in
  :meth:`OutlierResult.caveats`. An isolation forest's score has no such null
  distribution and therefore no calibrated cut point at all.
* **Isolation forest requires ``contamination``** — the share of the data that
  is bad — declared *in advance*. That share is precisely the unknown the
  analysis is trying to estimate. Setting ``contamination=0.01`` guarantees
  1% of the objects come back flagged whether the plate is pristine or ruined,
  which makes the output uninterpretable in exactly the case it matters.
* **MCD is affine-equivariant.** Rescale ``area`` from px^2 to um^2, or rotate
  to any linear combination of the features, and the fitted centre and
  covariance transform with it: every distance, and so every flag, is
  unchanged. This is the same invariance argument
  :mod:`spacr.qt.widgets.pca_model` makes for standardising before a PCA, and
  it is why no scaling option is needed here — the metric *is* the covariance.
  A tree ensemble splits on raw coordinates and is not equivariant; its answer
  depends on the units.
* **MCD is the multivariate generalisation of the median and the MAD.** The
  MCD centre is a multivariate median (the mean of the h-subset of minimum
  covariance determinant) and the MCD scatter is its spread. So all three
  methods offered here are the same idea at one, one, and p dimensions, and a
  user who understands the MAD already understands this.
* **MCD is deterministic given a seed.** The subset search is randomised, so
  ``random_state`` is set from :attr:`OutlierSpec.seed` and the same table
  gives the same flags every time — a figure in a report matches the screen it
  came from. An isolation forest is a randomised ensemble whose per-object
  score wobbles between fits.

**The cost, stated.** MCD needs ``n > p`` to have a covariance at all, and it
is unstable until roughly ``n >= 2p``
(:data:`MCD_MIN_OBJECTS_PER_FEATURE` * p): the estimate is fitted on a
*subset* of about ``n/2`` rows, so ``n = 2p`` already means the subset is only
barely larger than the dimension. Its breakdown point is set by
``support_fraction``: the sklearn default of ``(n + p + 1) / 2n`` gives the
maximum ~50%, and raising it trades robustness for efficiency. Too few objects
for the number of features is **refused with a message that names the way
out**: run a PCA first (:func:`spacr.qt.widgets.pca_model.pca`) and flag on
the first few components, which are fewer, uncorrelated, and carry the same
information.

Multiple testing is not optional here. Flagging N objects at level alpha is N
hypothesis tests; :meth:`OutlierResult.caveats` states both the expected count
of false flags at the chosen alpha and the Bonferroni-corrected alpha
(``alpha / N``) that would hold the *family-wise* rate at the nominal level,
so the user can pick which of the two questions they are asking.

Per object AND per well
=======================

The whole-plate failure that actually happens is a **bad well**: one well
pipetted wrong, one well out of focus, one well where the cells were seeded at
twice the density. Per-object flags are close to useless for finding it, and
the reason is arithmetic rather than opinion.

Take 60 objects per well, a feature whose within-well spread is sigma, and a
well whose whole population is shifted by 0.9 sigma. The **object** rule at
k = 3.5 asks each object to exceed 3.5 sigma; a 0.9 sigma shift moves that
requirement to 2.6 sigma, which about one object in 200 clears — so the bad
well contributes a fraction of one extra flagged object and disappears into
the ordinary tail. The **well** rule compares well *medians*, whose sampling
spread is only ``1.2533 * sigma / sqrt(60) = 0.16 sigma``; the same shift is
then 5.6 robust SDs and the well is unmistakable. The two tests have wildly
different power against the same defect, which is why both are run and both
are reported.

So the well pass is emphatically **not** "the well contains many flagged
objects". That statistic exists — it is reported as ``flagged_share`` — and it
answers a different question: it finds a well containing a few *catastrophic*
objects (a segmentation blow-up, a piece of dust measured as a cell) while
being blind to a uniform shift. The well-level robust score finds the uniform
shift while being blind to the isolated catastrophe. Both are on the well
frame because neither subsumes the other.

The well pass reduces each well to a robust summary — the **median** of each
feature over the well's objects, plus ``n`` — and then runs *the same* rule
(:data:`METHOD_MAD`, :data:`METHOD_IQR` or :data:`METHOD_MAHALANOBIS`) across
wells. The median rather than the mean for the same reason as everywhere else
here: a single blown-up object must not become a bad well.

**A well is not scored on three objects.** Below
:data:`DEFAULT_MIN_WELL_OBJECTS` objects the median is too noisy for a
comparison across wells to mean anything, so the well appears in the well
frame with ``well_scored = False`` and a reason saying how many objects it
had. It is **never silently dropped and never quietly scored** — a low-n well
excluded without a word is how an empty well becomes an unremarkable row.

**Well identity** comes from the columns spaCR already writes, in
:data:`WELL_KEY_SETS` order: the canonical
``(plateID, rowID, columnID)`` of :data:`spacr.schema.WELL_KEY_COLUMNS`, then
the composed ``prc`` key, then ``(plateID, well)``, then the legacy spellings
:mod:`spacr.schema` renames (``plate`` / ``row_name`` / ``column_name``), then
a bare ``well`` for a single-plate table. Any list of columns may be given
explicitly instead. When none can be found the error names the columns the
table actually has, and points at ``per_well=False`` for a table that has no
wells at all.

Flagging is never deletion
==========================

:func:`detect_outliers` **adds columns**; it never drops a row and never
changes one. :meth:`OutlierResult.object_frame` returns the input frame with
``outlier`` / ``outlier_score`` / ``outlier_reason`` / ``outlier_method``
appended, and ``len()`` of it always equals ``len()`` of the input. A name
that would collide with an existing column is suffixed rather than
overwritten, and the substitution is reported in
:attr:`OutlierResult.column_names` and in the notes — a QC pass that silently
overwrote a user's own ``outlier`` column would be a data-loss bug.

:meth:`OutlierResult.filtered` exists and drops the flagged rows. It is an
explicit call whose docstring says, in those words, that the caller is
choosing to delete objects from their analysis.

No Qt in here
=============
Pure numpy, pandas, scipy and scikit-learn, like
:mod:`spacr.qt.widgets.pca_model` and :mod:`spacr.qt.widgets.graph_spec`:
usable from a notebook, testable without a display, and free for the screen —
and for any later QC report — to reuse without inheriting a widget.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .graph_spec import CONTINUOUS, column_kinds

__all__ = [
    "OutlierError",
    "METHOD_MAD", "METHOD_IQR", "METHOD_MAHALANOBIS", "METHODS",
    "TRANSFORM_NONE", "TRANSFORM_LOG10", "TRANSFORMS",
    "MAD_TO_SIGMA", "MEAN_AD_TO_SIGMA", "NORMAL_IQR_SIGMAS",
    "NORMAL_QUARTILE_SIGMAS",
    "DEFAULT_MAD_K", "DEFAULT_IQR_C", "DEFAULT_ALPHA",
    "DEFAULT_MIN_WELL_OBJECTS", "MIN_WELLS_TO_SCORE",
    "MCD_MIN_OBJECTS_PER_FEATURE",
    "OBJECT_COLUMNS", "WELL_KEY_SETS",
    "OutlierSpec", "OutlierResult", "detect_outliers",
    "candidate_features", "well_key_columns",
    "median_absolute_deviation", "robust_scale", "tukey_fences",
]


class OutlierError(ValueError):
    """An outlier test that cannot mean anything, with the reason in the message.

    Raised rather than returned as an empty result. Every one of these is a
    sentence the user can act on — "``cell_area`` has 412 non-positive values
    and log10 of them is not a number", "3 features and 4 complete objects is
    fewer than the 6 the MCD needs" — and a caller that swallowed it would show
    an empty table with nothing to explain it. Screens catch it and print the
    message.
    """


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

#: Modified z-score against the median and the scaled MAD. Per feature, and
#: the default: it is the univariate test with the highest breakdown point on
#: offer and it needs no distributional assumption beyond a meaningful median.
METHOD_MAD = "mad"
#: Tukey's fence, ``Q1 - c*IQR`` / ``Q3 + c*IQR``. Per feature. Familiar from
#: every box plot ever drawn, and asymmetric on skewed data — see the module
#: docstring.
METHOD_IQR = "iqr"
#: Squared Mahalanobis distance in an MCD-robust metric, cut at a chi-square
#: quantile. The only method here that can see a *combination* of features
#: going wrong while every feature on its own looks ordinary.
METHOD_MAHALANOBIS = "mahalanobis"
METHODS: Tuple[str, ...] = (METHOD_MAD, METHOD_IQR, METHOD_MAHALANOBIS)

#: Values are used as measured.
TRANSFORM_NONE = "none"
#: ``log10(x)`` before anything else. Explicit only, and refused on
#: non-positive values.
TRANSFORM_LOG10 = "log10"
TRANSFORMS: Tuple[str, ...] = (TRANSFORM_NONE, TRANSFORM_LOG10)

# ---------------------------------------------------------------------------
# The constants of the robust scale estimators
# ---------------------------------------------------------------------------

#: ``1 / Phi^-1(0.75)``. The factor that makes ``MAD * MAD_TO_SIGMA`` a
#: consistent estimator of sigma for a Gaussian, so a threshold of 3.5 reads on
#: the same scale as "3.5 SD" without an SD ever being computed. See the module
#: docstring.
MAD_TO_SIGMA = 1.4826022185056018

#: ``sqrt(pi / 2)``. The same job for the mean absolute deviation from the
#: median (``E|X - mu| = sigma * sqrt(2/pi)`` for a Gaussian), which is the
#: stated fallback when the MAD is exactly zero.
MEAN_AD_TO_SIGMA = 1.2533141373155003

#: ``IQR / sigma`` for a Gaussian, ``2 * Phi^-1(0.75)``. Used only to
#: reconstruct a fence when the IQR itself is zero.
NORMAL_IQR_SIGMAS = 1.3489795003921634

#: ``Phi^-1(0.75)`` — how far a Gaussian's quartile sits from its median, in
#: sigmas. Same use.
NORMAL_QUARTILE_SIGMAS = 0.6744897501960817

#: Modified z above which :data:`METHOD_MAD` flags. 3.5 is Iglewicz & Hoaglin's
#: recommendation and corresponds to a two-sided Gaussian tail of about 5 in
#: 10,000 — deliberately stricter than the 3 that a mean/SD rule would use,
#: because a robust scale does not get inflated by the very points being
#: tested and so does not need the slack.
DEFAULT_MAD_K = 3.5

#: Tukey's multiplier. 1.5 is the box plot's whisker, roughly a 0.7% two-sided
#: Gaussian rate; 3.0 is the conventional "far out" fence.
DEFAULT_IQR_C = 1.5

#: Per-object false-positive rate for :data:`METHOD_MAHALANOBIS`. One clean
#: object in a thousand — about 100 false flags over 10^5 objects, about 200
#: over 200,000. Said out loud in :meth:`OutlierResult.caveats` rather than
#: left for the reader to multiply.
DEFAULT_ALPHA = 0.001

#: Objects a well needs before its median is compared with other wells'. Below
#: it the median's own sampling spread swamps any shift worth finding, so the
#: well is reported "not scored" instead.
DEFAULT_MIN_WELL_OBJECTS = 20

#: Wells needed before the across-well rule runs at all. A median and a MAD
#: over four points are arbitrary; below this the well pass reports every well
#: as not scored and says why.
MIN_WELLS_TO_SCORE = 5

#: MCD is undefined at ``n <= p`` and unstable until about ``n >= 2p``,
#: because it is fitted on a subset of roughly half the rows.
MCD_MIN_OBJECTS_PER_FEATURE = 2

#: The columns :meth:`OutlierResult.object_frame` adds, as
#: ``logical name -> preferred column name``. Collisions are suffixed, never
#: overwritten.
OBJECT_COLUMNS: Dict[str, str] = {
    "outlier": "outlier",
    "score": "outlier_score",
    "reason": "outlier_reason",
    "method": "outlier_method",
}

#: Column sets that identify a well, tried in this order by
#: :func:`well_key_columns`.
#:
#: The first is :data:`spacr.schema.WELL_KEY_COLUMNS`, the canonical trio every
#: spaCR measurement table carries. The second is ``prc`` — the same well,
#: pre-joined, which is what an aggregated table usually keeps. Then a
#: plate plus a well name, then the legacy spellings
#: :data:`spacr.schema.LEGACY_COLUMN_NAMES` renames on read (a database written
#: before the migration still has them), then a bare well for a single-plate
#: export.
WELL_KEY_SETS: Tuple[Tuple[str, ...], ...] = (
    ("plateID", "rowID", "columnID"),
    ("prc",),
    ("plateID", "well"),
    ("plate", "row_name", "column_name"),
    ("plate", "row", "column"),
    ("well",),
)


# ---------------------------------------------------------------------------
# The robust estimators, on their own so they can be tested on a hand vector
# ---------------------------------------------------------------------------

def median_absolute_deviation(values: Sequence[float]) -> float:
    """``median(|x - median(x)|)`` over the finite values. **Unscaled.**

    Returned raw rather than pre-multiplied by :data:`MAD_TO_SIGMA` so that a
    caller can see the zero when it is zero — which is the case the whole
    fallback in :func:`robust_scale` exists for.

    :returns: the MAD, or ``nan`` when nothing is finite.
    """
    finite = _finite(values)
    if finite.size == 0:
        return float("nan")
    return float(np.median(np.abs(finite - np.median(finite))))


def robust_scale(values: Sequence[float]) -> Tuple[float, float, str]:
    """``(median, sigma estimate, note)`` for one feature.

    The sigma estimate is :data:`MAD_TO_SIGMA` times the MAD whenever the MAD
    is non-zero. When it is zero — more than half the objects share one value —
    dividing by it would score every other value infinity and flag the entire
    tail, so the estimate falls back to :data:`MEAN_AD_TO_SIGMA` times the mean
    absolute deviation from the median and ``note`` says ``"mad-zero"``. See
    the module docstring for why that fallback and not Rousseeuw–Croux ``Sn``,
    and for the honest reading of it.

    :returns: ``(centre, scale, note)``. ``note`` is ``""`` when the MAD did
        the work, ``"mad-zero"`` when the fallback fired, ``"constant"`` when
        the feature has no variation at all (scale 0, nothing can be an
        outlier), and ``"empty"`` when there is nothing finite to measure.
    """
    finite = _finite(values)
    if finite.size == 0:
        return float("nan"), 0.0, "empty"
    centre = float(np.median(finite))
    deviations = np.abs(finite - centre)
    scale = MAD_TO_SIGMA * float(np.median(deviations))
    if scale > 0:
        return centre, scale, ""
    fallback = MEAN_AD_TO_SIGMA * float(np.mean(deviations))
    if fallback > 0:
        return centre, fallback, "mad-zero"
    return centre, 0.0, "constant"


def tukey_fences(values: Sequence[float], c: float = DEFAULT_IQR_C
                 ) -> Tuple[float, float, float, float, str]:
    """``(q1, q3, lower fence, upper fence, note)`` for one feature.

    Quartiles by ``numpy.percentile``'s default linear interpolation, fences at
    ``Q1 - c*IQR`` and ``Q3 + c*IQR``.

    When the IQR is zero — half the objects tied at one value — the fences
    collapse onto that value and everything else is "outside" them. Rather than
    flag the whole tail, the quartiles are reconstructed from the robust sigma
    of :func:`robust_scale` at their Gaussian positions
    (``median +/- 0.6745 sigma``, ``IQR = 1.349 sigma``), which reproduces the
    ordinary fence exactly for Gaussian data and is reported as ``"iqr-zero"``.

    :returns: ``(q1, q3, low, high, note)``; ``note`` is ``""``,
        ``"iqr-zero"``, ``"constant"`` or ``"empty"``.
    """
    finite = _finite(values)
    if finite.size == 0:
        nan = float("nan")
        return nan, nan, nan, nan, "empty"
    q1 = float(np.percentile(finite, 25))
    q3 = float(np.percentile(finite, 75))
    spread = q3 - q1
    note = ""
    if spread <= 0:
        centre, sigma, scale_note = robust_scale(finite)
        if sigma <= 0:
            return q1, q3, q1, q3, "constant"
        q1 = centre - NORMAL_QUARTILE_SIGMAS * sigma
        q3 = centre + NORMAL_QUARTILE_SIGMAS * sigma
        spread = NORMAL_IQR_SIGMAS * sigma
        note = "iqr-zero" if scale_note != "constant" else "constant"
    return q1, q3, q1 - c * spread, q3 + c * spread, note


def _finite(values: Sequence[float]) -> np.ndarray:
    """``values`` as a float array with every non-finite entry removed."""
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def candidate_features(frame: pd.DataFrame) -> Tuple[str, ...]:
    """The columns worth offering as outlier features, sorted.

    Continuous by :func:`spacr.qt.widgets.graph_spec.column_kinds`, exactly as
    :func:`spacr.qt.widgets.pca_model.candidate_features` picks its features —
    the one column classifier in this codebase, re-read rather than re-derived,
    so PCA and this screen cannot disagree about what ``cell_count`` is.

    That rule already excludes the object keys (they identify rather than
    describe, and the modified z-score of a ``prcfo`` is nonsense) and the
    small-cardinality numeric codes — a class label, a plate number — which are
    labels rather than measured quantities.

    A user who wants a particular column anyway names it in
    :attr:`OutlierSpec.features`; nothing here refuses it.
    """
    return tuple(sorted(name for name, kind in column_kinds(frame).items()
                        if kind == CONTINUOUS))


def well_key_columns(frame: pd.DataFrame) -> Tuple[str, ...]:
    """The columns that identify a well in ``frame``, auto-detected.

    Tries :data:`WELL_KEY_SETS` in order and returns the first set whose
    columns are all present.

    :raises OutlierError: when none of them is, with the table's own columns in
        the message — the user can then either name the right ones explicitly
        or turn the well pass off.
    """
    for keys in WELL_KEY_SETS:
        if all(name in frame.columns for name in keys):
            return keys
    present = ", ".join(str(c) for c in list(frame.columns)[:12])
    more = f" (+{len(frame.columns) - 12} more)" if len(frame.columns) > 12 \
        else ""
    wanted = "; ".join("+".join(keys) for keys in WELL_KEY_SETS)
    raise OutlierError(
        f"no well key columns in this table, so objects cannot be grouped "
        f"into wells. Looked for {wanted}. This table has: {present}{more}. "
        f"Name the columns yourself with OutlierSpec(well_keys=(...)), or set "
        f"per_well=False if these measurements did not come off a plate.")


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OutlierSpec:
    """What to test, how, and where the line is.

    Frozen and JSON round-tripping, like
    :class:`~spacr.qt.widgets.pca_model.PCASpec` and
    :class:`~spacr.qt.widgets.graph_spec.GraphSpec`, so a QC pass is something
    a settings file or a methods section can carry verbatim.

    :param features: the columns to test. Empty means
        :func:`candidate_features` of whatever frame it runs against — a
        *default*, not a promise; the result records what it actually used.
    :param method: one of :data:`METHODS`.
    :param k: modified-z threshold for :data:`METHOD_MAD`.
    :param c: fence multiplier for :data:`METHOD_IQR`.
    :param alpha: per-object false-positive rate for
        :data:`METHOD_MAHALANOBIS`. The chi-square cut point is
        ``chi2.ppf(1 - alpha, p)``.
    :param transform: :data:`TRANSFORM_NONE` or :data:`TRANSFORM_LOG10`.
        Applied before anything else and never chosen for the user.
    :param well_keys: columns identifying a well. Empty means
        :func:`well_key_columns` auto-detects them.
    :param min_well_objects: a well with fewer objects than this is reported
        "not scored" rather than scored or dropped.
    :param per_well: run the across-well pass. ``False`` for measurements that
        did not come off a plate — and the only thing that makes a table with
        no well columns analysable at all.
    :param support_fraction: MCD's ``support_fraction``. ``None`` is sklearn's
        default ``(n + p + 1) / 2n``, the maximum-breakdown choice.
    :param seed: MCD's ``random_state``, so the same table gives the same
        flags every time.
    :raises OutlierError: on an unknown method or transform, a non-positive
        threshold, an alpha outside ``(0, 1)`` or a support fraction outside
        ``(0, 1]`` — at the point the spec is built, not half way through a
        200,000-row fit.
    """

    features: Tuple[str, ...] = ()
    method: str = METHOD_MAD
    k: float = DEFAULT_MAD_K
    c: float = DEFAULT_IQR_C
    alpha: float = DEFAULT_ALPHA
    transform: str = TRANSFORM_NONE
    well_keys: Tuple[str, ...] = ()
    min_well_objects: int = DEFAULT_MIN_WELL_OBJECTS
    per_well: bool = True
    support_fraction: Optional[float] = None
    seed: int = 0

    def __post_init__(self) -> None:
        seen: Dict[str, None] = {}
        for name in self.features or ():
            if name:
                seen.setdefault(str(name), None)
        object.__setattr__(self, "features", tuple(seen))
        keys: Dict[str, None] = {}
        for name in self.well_keys or ():
            if name:
                keys.setdefault(str(name), None)
        object.__setattr__(self, "well_keys", tuple(keys))
        if self.method not in METHODS:
            raise OutlierError(
                f"unknown method {self.method!r}; choose one of "
                f"{', '.join(METHODS)} — {METHOD_MAD!r} is the modified "
                f"z-score against the median and the scaled MAD, "
                f"{METHOD_IQR!r} is Tukey's fence, and "
                f"{METHOD_MAHALANOBIS!r} is the robust multivariate distance")
        if self.transform not in TRANSFORMS:
            raise OutlierError(
                f"unknown transform {self.transform!r}; it is "
                f"{TRANSFORM_NONE!r} (the default, values as measured) or "
                f"{TRANSFORM_LOG10!r}")
        k = float(self.k)
        if not k > 0:
            raise OutlierError(
                f"k is how many robust SDs from the median count as an "
                f"outlier and must be positive, not {self.k}")
        object.__setattr__(self, "k", k)
        c = float(self.c)
        if not c > 0:
            raise OutlierError(
                f"c multiplies the IQR to place Tukey's fence and must be "
                f"positive, not {self.c}")
        object.__setattr__(self, "c", c)
        alpha = float(self.alpha)
        if not 0.0 < alpha < 1.0:
            raise OutlierError(
                f"alpha is the per-object false-positive rate and must be in "
                f"(0, 1), not {self.alpha}. {DEFAULT_ALPHA} means one clean "
                f"object in a thousand is expected to be flagged.")
        object.__setattr__(self, "alpha", alpha)
        minimum = int(self.min_well_objects)
        if minimum < 1:
            raise OutlierError(
                f"min_well_objects must be at least 1, not "
                f"{self.min_well_objects}; a well with no objects has no "
                f"median")
        object.__setattr__(self, "min_well_objects", minimum)
        object.__setattr__(self, "per_well", bool(self.per_well))
        if self.support_fraction is not None:
            fraction = float(self.support_fraction)
            if not 0.0 < fraction <= 1.0:
                raise OutlierError(
                    f"support_fraction is the share of objects the MCD fits "
                    f"on and must be in (0, 1], not {self.support_fraction}; "
                    f"None picks sklearn's maximum-breakdown default")
            object.__setattr__(self, "support_fraction", fraction)
        object.__setattr__(self, "seed", int(self.seed))

    # -- edits ------------------------------------------------------------
    def with_features(self, features: Sequence[str]) -> "OutlierSpec":
        """A copy testing ``features``."""
        return replace(self, features=tuple(features))

    def with_method(self, method: str) -> "OutlierSpec":
        """A copy using ``method``."""
        return replace(self, method=method)

    def with_transform(self, transform: str) -> "OutlierSpec":
        """A copy applying ``transform`` first."""
        return replace(self, transform=transform)

    def with_well_keys(self, keys: Sequence[str]) -> "OutlierSpec":
        """A copy grouping wells by ``keys``."""
        return replace(self, well_keys=tuple(keys))

    # -- the threshold ----------------------------------------------------
    def threshold(self, n_features: int = 1) -> float:
        """The number :attr:`OutlierResult.scores` is compared against.

        ``k`` for :data:`METHOD_MAD`, ``c`` for :data:`METHOD_IQR`, and the
        chi-square quantile ``chi2.ppf(1 - alpha, p)`` for
        :data:`METHOD_MAHALANOBIS` — which is the one that depends on how many
        features are in play, because the null distribution does.
        """
        if self.method == METHOD_MAD:
            return self.k
        if self.method == METHOD_IQR:
            return self.c
        from scipy.stats import chi2
        return float(chi2.ppf(1.0 - self.alpha, max(1, int(n_features))))

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """A plain JSON-able dict. Every field, always — a stable schema beats
        a compact one for something a report reads back."""
        return {
            "features": list(self.features),
            "method": self.method,
            "k": self.k,
            "c": self.c,
            "alpha": self.alpha,
            "transform": self.transform,
            "well_keys": list(self.well_keys),
            "min_well_objects": self.min_well_objects,
            "per_well": self.per_well,
            "support_fraction": self.support_fraction,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OutlierSpec":
        """Rebuild from :meth:`to_dict`; unknown keys ignored, missing keys
        defaulted, so a QC pass written by another build still opens."""
        fields = {"features", "method", "k", "c", "alpha", "transform",
                  "well_keys", "min_well_objects", "per_well",
                  "support_fraction", "seed"}
        known = {key: value for key, value in dict(payload).items()
                 if key in fields}
        for name in ("features", "well_keys"):
            if name in known:
                known[name] = tuple(known[name] or ())
        return cls(**known)

    def to_json(self) -> str:
        """:meth:`to_dict` as sorted JSON."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "OutlierSpec":
        """Inverse of :meth:`to_json`."""
        return cls.from_dict(json.loads(text))

    def describe(self) -> str:
        """One line, for a caption."""
        if self.method == METHOD_MAD:
            rule = f"modified z > {self.k:g}"
        elif self.method == METHOD_IQR:
            rule = f"outside Q1/Q3 ± {self.c:g}·IQR"
        else:
            rule = f"robust Mahalanobis, α = {self.alpha:g} per object"
        features = (f"{len(self.features)} features" if self.features
                    else "every continuous column")
        transform = ("" if self.transform == TRANSFORM_NONE
                     else f" · {self.transform}")
        wells = (f" · wells of ≥ {self.min_well_objects} objects"
                 if self.per_well else " · objects only")
        return f"{self.method} · {features} · {rule}{transform}{wells}"


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OutlierResult:
    """Flags, scores and reasons — per object and per well — plus the caveats.

    :param method: the :data:`METHODS` member that produced this.
    :param features: the columns actually tested, in matrix order.
    :param scores: one per **input row**, positionally aligned with the frame
        :func:`detect_outliers` was given. ``nan`` for a row that could not be
        scored. Positional because a measurement frame carries a duplicated or
        reset index often enough that positions are the only safe currency —
        the same reason :class:`~spacr.qt.widgets.pca_model.PCAResult` uses
        them.
    :param flags: one per input row. ``True`` is "flagged", never "deleted".
    :param reasons: one per input row; ``""`` for an unflagged, scored row.
    :param threshold: the number :attr:`scores` was compared against.
    :param centres: per feature, the robust centre used (the median, or the
        MCD centre's coordinate).
    :param scales: per feature, the robust sigma used. Empty for
        :data:`METHOD_MAHALANOBIS`, whose scale is a matrix.
    :param fences: per feature, ``(low, high)`` for :data:`METHOD_IQR`.
    :param wells: one row per well, always including the wells that were not
        scored. See :meth:`well_frame`.
    :param well_keys: the columns ``wells`` is keyed on. Empty when the well
        pass did not run.
    :param column_names: logical name -> the column
        :meth:`object_frame` will actually add, which differs from
        :data:`OBJECT_COLUMNS` only when the input already had that name.
    """

    method: str
    features: Tuple[str, ...]
    scores: np.ndarray
    flags: np.ndarray
    reasons: Tuple[str, ...]
    threshold: float
    n_rows_in: int
    n_scored: int
    centres: Mapping[str, float] = field(default_factory=dict)
    scales: Mapping[str, float] = field(default_factory=dict)
    fences: Mapping[str, Tuple[float, float]] = field(default_factory=dict)
    wells: pd.DataFrame = field(default_factory=pd.DataFrame)
    well_keys: Tuple[str, ...] = ()
    n_wells_scored: int = 0
    min_well_objects: int = DEFAULT_MIN_WELL_OBJECTS
    alpha: float = DEFAULT_ALPHA
    transform: str = TRANSFORM_NONE
    n_non_finite: int = 0
    column_names: Mapping[str, str] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()

    # -- shape ------------------------------------------------------------
    def __len__(self) -> int:
        """Objects the test was given — **not** the objects it flagged.

        Deliberately the input count: a length that shrank with the flags
        would make ``len(result)`` mean something different for a clean plate
        and a ruined one.
        """
        return int(self.n_rows_in)

    @property
    def n_flagged(self) -> int:
        """Objects flagged."""
        return int(np.count_nonzero(self.flags))

    @property
    def flagged_share(self) -> float:
        """Flagged objects over **scored** objects, in ``[0, 1]``."""
        return (self.n_flagged / self.n_scored) if self.n_scored else 0.0

    @property
    def n_not_scored(self) -> int:
        """Objects with no usable value, which are never flagged."""
        return int(self.n_rows_in - self.n_scored)

    @property
    def n_features(self) -> int:
        """How many columns were tested."""
        return len(self.features)

    @property
    def n_wells(self) -> int:
        """Wells found, scored or not. 0 when the well pass did not run."""
        return int(len(self.wells))

    @property
    def has_wells(self) -> bool:
        """Whether the across-well pass produced anything."""
        return bool(self.well_keys) and not self.wells.empty

    # -- reading the wells -------------------------------------------------
    def flagged_wells(self) -> Tuple[Any, ...]:
        """The wells the across-well rule flagged, as key tuples.

        A one-column key yields plain scalars rather than one-tuples, because
        ``prc`` is already the whole name of the well and wrapping it would
        make every caller unwrap it.
        """
        if not self.has_wells or "well_outlier" not in self.wells.columns:
            return ()
        hit = self.wells.loc[
            self.wells["well_outlier"].to_numpy(dtype=bool)]
        keys = [k for k in self.well_keys if k in hit.columns]
        if not keys:
            return ()
        if len(keys) == 1:
            return tuple(hit[keys[0]].tolist())
        return tuple(map(tuple, hit[keys].itertuples(index=False, name=None)))

    def unscored_wells(self) -> Tuple[Any, ...]:
        """The wells reported but not scored — too few objects, in the same
        key form as :meth:`flagged_wells`."""
        if not self.has_wells or "well_scored" not in self.wells.columns:
            return ()
        rest = self.wells.loc[
            ~self.wells["well_scored"].to_numpy(dtype=bool)]
        keys = [k for k in self.well_keys if k in rest.columns]
        if not keys:
            return ()
        if len(keys) == 1:
            return tuple(rest[keys[0]].tolist())
        return tuple(map(tuple, rest[keys].itertuples(index=False, name=None)))

    # -- frames ------------------------------------------------------------
    def object_frame(self, source: pd.DataFrame) -> pd.DataFrame:
        """``source`` with the flag columns **added**. Nothing is dropped.

        ``len()`` of the returned frame equals ``len(source)``, always, and no
        input column is touched. A name that would collide with one of
        ``source``'s own columns is suffixed — see :attr:`column_names` for
        what was actually written.

        :raises OutlierError: when ``source`` is not the frame that was
            analysed. The flags are positional, so silently aligning them to a
            frame of a different length would attribute one object's badness to
            another.
        """
        if len(source) != self.n_rows_in:
            raise OutlierError(
                f"these flags were computed on {self.n_rows_in:,} objects and "
                f"this frame has {len(source):,}. The flags are positional, so "
                f"they can only be written back onto the frame they were "
                f"computed from — re-run detect_outliers on this one.")
        frame = source.copy()
        names = dict(self.column_names) or dict(OBJECT_COLUMNS)
        frame[names["outlier"]] = self.flags
        frame[names["score"]] = self.scores
        frame[names["reason"]] = list(self.reasons)
        frame[names["method"]] = self.method_label()
        return frame

    def well_frame(self) -> pd.DataFrame:
        """One row per well: its key, its ``n``, its medians, and both signals.

        Both, on purpose. ``flagged_share`` is the share of the well's objects
        the per-object rule flagged — it finds a well with a few catastrophic
        objects in it. ``well_outlier_score`` is the same robust rule applied
        *across* wells to their medians — it finds a well whose whole
        population is shifted, which the object rule is nearly blind to. See
        the module docstring for the arithmetic of why they differ.

        Wells with fewer than :attr:`min_well_objects` objects are present with
        ``well_scored = False`` and a reason. They are never dropped.
        """
        return self.wells.copy()

    def filtered(self, source: pd.DataFrame) -> pd.DataFrame:
        """``source`` without the flagged rows.

        **Calling this is choosing to delete objects from your analysis.**
        Nothing else in this module removes a row; :func:`detect_outliers` only
        ever adds columns, so that a flag can be reviewed, argued with and
        reversed. This method is the explicit escape hatch for a caller who has
        looked at the flags and decided.

        Objects that could not be scored are **kept** — they were not flagged,
        and dropping them here would delete rows for missingness under the name
        of outlier removal.
        """
        if len(source) != self.n_rows_in:
            raise OutlierError(
                f"these flags were computed on {self.n_rows_in:,} objects and "
                f"this frame has {len(source):,}; positional flags cannot be "
                f"applied to a different frame.")
        return source.iloc[np.flatnonzero(~self.flags)]

    # -- saying it in words -------------------------------------------------
    def method_label(self) -> str:
        """The method and its threshold in one token, for the added column."""
        if self.method == METHOD_MAD:
            return f"mad(k={self.threshold:g})"
        if self.method == METHOD_IQR:
            return f"iqr(c={self.threshold:g})"
        return f"mahalanobis(mcd, alpha={self.alpha:g})"

    def headline(self) -> str:
        """One sentence with the counts, including the bad news."""
        parts = [
            f"{self.method_label()} over {self.n_features} feature(s) flagged "
            f"{self.n_flagged:,} of {self.n_scored:,} scored objects "
            f"({self.flagged_share:.2%})"]
        if self.has_wells:
            flagged = len(self.flagged_wells())
            parts.append(
                f"and {flagged} of {self.n_wells_scored:,} scored wells")
            if flagged and self.flagged_share < 0.01:
                parts.append(
                    "— a well shifted as a whole barely moves its individual "
                    "objects, which is exactly why the well pass exists")
        elif self.well_keys:
            parts.append("and no well was scored")
        return " ".join(parts) + "."

    def caveats(self) -> Tuple[str, ...]:
        """Everything a reader needs before believing the flags."""
        out: List[str] = []
        if self.method == METHOD_MAHALANOBIS:
            expected = self.n_scored * self.alpha
            out.append(
                f"This is {self.n_scored:,} hypothesis tests at α = "
                f"{self.alpha:g}, so about {expected:,.0f} clean object(s) are "
                f"expected to be flagged even if nothing is wrong — at this α, "
                f"200,000 objects would produce roughly 200 false flags. To "
                f"hold the family-wise rate at {self.alpha:g} instead of the "
                f"per-object rate, Bonferroni says use α = "
                f"{self.alpha / max(1, self.n_scored):.3g} "
                f"(threshold {_chi2_quantile(self.alpha / max(1, self.n_scored), self.n_features):.1f} "
                f"rather than {self.threshold:.1f}). Which of the two you want "
                f"depends on whether you are asking 'is this object bad' or "
                f"'is anything on this plate bad'.")
            out.append(
                "The χ² cut point assumes the clean objects are multivariate "
                "normal in these features. spaCR measurements usually are not "
                "without a transform, so read the rate as an order of "
                "magnitude rather than a guarantee.")
        if self.method == METHOD_IQR and self.transform == TRANSFORM_NONE:
            out.append(
                "Tukey's fence is symmetric in the quartiles, so on a "
                "right-skewed measurement — which most spaCR features are — it "
                "flags a large share of the right tail by construction, on "
                "clean data. A log10 transform (transform='log10') makes the "
                "fence mean what it looks like; without one, compare the share "
                "flagged between plates rather than reading it as a rate.")
        if any("the MAD of" in note and "is zero" in note
               for note in self.notes):
            out.append(
                "At least one feature had a MAD of exactly zero — more than "
                "half its objects share one value — so its scale came from the "
                "mean absolute deviation instead. That fallback has no "
                "breakdown point of its own: it keeps the test from flagging "
                "the whole tail, but a column that degenerate is better read "
                "as a category than tested as a measurement.")
        if self.n_not_scored:
            out.append(
                f"{self.n_not_scored:,} of {self.n_rows_in:,} objects "
                f"({self.n_not_scored / max(1, self.n_rows_in):.1%}) could not "
                f"be scored — no finite value for the feature(s) they needed — "
                f"so they are neither flagged nor cleared. They are still in "
                f"the frame.")
        if self.n_non_finite:
            out.append(
                f"{self.n_non_finite:,} cell(s) were ±inf or NaN and were "
                f"treated as missing. An infinity from a ratio feature would "
                f"otherwise be the outlier and the scale at once.")
        if self.has_wells:
            unscored = len(self.unscored_wells())
            if unscored:
                out.append(
                    f"{unscored} well(s) had fewer than "
                    f"{self.min_well_objects} objects and are reported "
                    f"'not scored' rather than compared: a median over a "
                    f"handful of objects moves more than any plate effect "
                    f"worth finding. They are in the well frame, not deleted "
                    f"from it.")
        else:
            out.append(
                "No well was scored, so every conclusion here is per object. "
                "A whole bad well is the more common failure and it is the one "
                "per-object flags are nearly blind to — it is not visible in "
                "this run.")
        out.append(
            "Flagged is not deleted. Every object is still in the frame with "
            "an added column; filtered() is the separate, explicit call that "
            "removes them.")
        return tuple(out)

    def report(self) -> str:
        """The whole story, as the panel shows it and a report file writes it."""
        lines = [
            f"Outlier scan of {self.n_rows_in:,} objects × "
            f"{self.n_features} feature(s)"
            + (", log10-transformed" if self.transform == TRANSFORM_LOG10
               else "") + ".",
            "  " + self.headline(),
        ]
        if self.method != METHOD_MAHALANOBIS:
            lines.append("")
            for name in self.features:
                centre = self.centres.get(name, float("nan"))
                if self.method == METHOD_IQR:
                    low, high = self.fences.get(name, (float("nan"),) * 2)
                    lines.append(
                        f"  {name}: median {centre:.6g}, fence "
                        f"[{low:.6g}, {high:.6g}]")
                else:
                    scale = self.scales.get(name, float("nan"))
                    lines.append(
                        f"  {name}: median {centre:.6g}, robust SD "
                        f"{scale:.6g}, flag beyond "
                        f"{centre - self.threshold * scale:.6g} / "
                        f"{centre + self.threshold * scale:.6g}")
        else:
            lines.append(
                f"  threshold: squared robust Mahalanobis distance > "
                f"{self.threshold:.3f} (χ² with {self.n_features} df at "
                f"α = {self.alpha:g}).")
        if self.has_wells:
            flagged = self.flagged_wells()
            lines.append("")
            lines.append(
                f"  wells: {self.n_wells:,} found, {self.n_wells_scored:,} "
                f"scored (≥ {self.min_well_objects} objects), "
                f"{len(flagged)} flagged.")
            if flagged:
                shown = ", ".join(
                    "+".join(str(p) for p in k) if isinstance(k, tuple)
                    else str(k) for k in flagged[:8])
                more = f" (+{len(flagged) - 8} more)" if len(flagged) > 8 \
                    else ""
                lines.append(f"    {shown}{more}")
        caveats = self.caveats()
        if caveats:
            lines.append("")
            lines.extend("  ! " + c for c in caveats)
        if self.notes:
            lines.append("")
            lines.extend("  · " + n for n in self.notes)
        return "\n".join(lines)


def _chi2_quantile(alpha: float, df: int) -> float:
    """``chi2.ppf(1 - alpha, df)``, guarded for a degenerate alpha."""
    from scipy.stats import chi2
    alpha = min(max(float(alpha), 1e-300), 1.0 - 1e-15)
    return float(chi2.ppf(1.0 - alpha, max(1, int(df))))


# ---------------------------------------------------------------------------
# The computation
# ---------------------------------------------------------------------------

def _select_features(frame: pd.DataFrame, spec: OutlierSpec) -> List[str]:
    """The requested features that exist, refusing when there is no test left."""
    wanted = list(spec.features) or list(candidate_features(frame))
    if not wanted:
        raise OutlierError(
            "this table has no continuous columns to test. Outlier detection "
            "needs measured quantities; plate/row/well identifiers and small "
            "integer codes are deliberately not offered.")
    missing = [name for name in wanted if name not in frame.columns]
    kept = [name for name in wanted if name in frame.columns]
    if not kept:
        raise OutlierError(
            f"none of the requested features is a column of this table: "
            f"{', '.join(missing)}. It has {len(frame.columns)} columns.")
    if spec.method == METHOD_MAHALANOBIS and len(kept) < 2:
        raise OutlierError(
            f"{METHOD_MAHALANOBIS!r} is the multivariate test and needs at "
            f"least two features; it was given {len(kept)}. On one column the "
            f"robust Mahalanobis distance is the modified z-score, so use "
            f"method={METHOD_MAD!r}.")
    return kept


def _matrix(frame: pd.DataFrame, features: List[str], spec: OutlierSpec,
            notes: List[str]) -> Tuple[np.ndarray, int]:
    """``frame[features]`` as floats, non-finite cells NaN, transform applied."""
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    matrix = numeric.to_numpy(dtype=float, copy=True)
    non_finite = ~np.isfinite(matrix)
    n_non_finite = int((non_finite & ~np.isnan(matrix)).sum())
    if n_non_finite:
        matrix[non_finite] = np.nan
        notes.append(
            f"{n_non_finite:,} non-finite value(s) were treated as missing "
            f"before anything else")
    if spec.transform == TRANSFORM_LOG10:
        with np.errstate(invalid="ignore"):
            bad = np.isfinite(matrix) & (matrix <= 0)
        if bad.any():
            offenders = ", ".join(
                f"{features[i]} ({int(bad[:, i].sum()):,})"
                for i in np.flatnonzero(bad.any(axis=0))[:4])
            raise OutlierError(
                f"transform={TRANSFORM_LOG10!r} was asked for and "
                f"{int(bad.sum()):,} value(s) are zero or negative, which "
                f"log10 cannot take: {offenders}. Adding a pseudocount would "
                f"invent measurements and dropping those objects would "
                f"self-select the population, so neither is done for you — "
                f"drop the feature, filter the objects yourself, or run "
                f"untransformed.")
        with np.errstate(divide="ignore", invalid="ignore"):
            matrix = np.log10(matrix)
        notes.append(
            f"values were log10-transformed first, so every centre, scale and "
            f"score below is in log10 units")
    return matrix, n_non_finite


def _unique_names(preferred: Mapping[str, str], taken: Sequence[str],
                  notes: List[str]) -> Dict[str, str]:
    """Column names that do not collide with ``taken``, suffixed if they would.

    Overwriting a user's own ``outlier`` column would be data loss disguised as
    a QC pass, so a collision is renamed and said out loud.
    """
    used = set(str(c) for c in taken)
    out: Dict[str, str] = {}
    for logical, name in preferred.items():
        candidate = name
        suffix = 2
        while candidate in used:
            candidate = f"{name}_{suffix}"
            suffix += 1
        if candidate != name:
            notes.append(
                f"this table already has a {name!r} column, so the flag was "
                f"written to {candidate!r} rather than over it")
        used.add(candidate)
        out[logical] = candidate
    return out


def _score_per_feature(matrix: np.ndarray, features: List[str],
                       spec: OutlierSpec, notes: List[str], where: str
                       ) -> Tuple[np.ndarray, Dict[str, float],
                                  Dict[str, float],
                                  Dict[str, Tuple[float, float]]]:
    """Per-feature robust scores for :data:`METHOD_MAD` / :data:`METHOD_IQR`.

    :returns: ``(scores, centres, scales, fences)`` where ``scores`` is
        ``(n, p)`` with NaN where a value was missing. A score is comparable
        with :meth:`OutlierSpec.threshold` in both methods: modified z for the
        MAD rule, and "IQRs beyond the nearer quartile" for the Tukey rule, so
        that in each case ``score > threshold`` is the flag.
    """
    n, p = matrix.shape
    scores = np.full((n, p), np.nan, dtype=float)
    centres: Dict[str, float] = {}
    scales: Dict[str, float] = {}
    fences: Dict[str, Tuple[float, float]] = {}
    for i, name in enumerate(features):
        column = matrix[:, i]
        present = np.isfinite(column)
        if spec.method == METHOD_MAD:
            centre, scale, note = robust_scale(column)
            centres[name] = centre
            scales[name] = scale
            if scale > 0:
                scores[present, i] = np.abs(column[present] - centre) / scale
            else:
                scores[present, i] = 0.0
            fences[name] = ((centre - spec.k * scale, centre + spec.k * scale)
                            if scale > 0 else (centre, centre))
        else:
            q1, q3, low, high, note = tukey_fences(column, spec.c)
            spread = q3 - q1
            centres[name] = float((q1 + q3) / 2.0)
            scales[name] = float(spread)
            fences[name] = (low, high)
            if spread > 0:
                scores[present, i] = np.maximum(
                    q1 - column[present], column[present] - q3) / spread
            else:
                scores[present, i] = 0.0
        if note == "mad-zero":
            notes.append(
                f"{where}: the MAD of {name!r} is zero — more than half the "
                f"values are identical — so its scale came from the mean "
                f"absolute deviation from the median instead of dividing by "
                f"zero and flagging every other value")
        elif note == "iqr-zero":
            notes.append(
                f"{where}: the IQR of {name!r} is zero — more than half the "
                f"values are identical — so the fence was rebuilt from the "
                f"robust SD at its Gaussian quartiles instead of collapsing "
                f"onto the tied value")
        elif note == "constant":
            notes.append(
                f"{where}: {name!r} has one value over these objects, so "
                f"nothing in it can be an outlier")
        elif note == "empty":
            notes.append(f"{where}: {name!r} has no finite value at all")
    return scores, centres, scales, fences


def _reasons_per_feature(scores: np.ndarray, matrix: np.ndarray,
                         features: List[str], flags: np.ndarray,
                         unscored: np.ndarray, spec: OutlierSpec,
                         fences: Mapping[str, Tuple[float, float]]
                         ) -> List[str]:
    """One sentence per flagged (or unscorable) row; ``""`` for the rest.

    Built only for the rows that need one — a 200,000-element list of formatted
    strings nobody reads is a second of wall clock for nothing.
    """
    reasons = [""] * scores.shape[0]
    for row in np.flatnonzero(flags):
        offending = np.flatnonzero(scores[row] > spec.threshold())
        if offending.size == 0:  # pragma: no cover - flags come from scores
            continue
        order = offending[np.argsort(scores[row][offending])[::-1]][:3]
        bits = []
        for i in order:
            name = features[i]
            value = matrix[row, i]
            low, high = fences.get(name, (float("nan"), float("nan")))
            side = "high" if value > high else "low"
            if spec.method == METHOD_MAD:
                bits.append(f"{name} {side} (z={scores[row, i]:.1f})")
            else:
                bits.append(f"{name} {side} ({scores[row, i]:.2f}·IQR past "
                            f"the quartile)")
        reasons[row] = ", ".join(bits)
    for row in np.flatnonzero(unscored):
        reasons[row] = "not scored: no finite value for any tested feature"
    return reasons


def _mahalanobis(matrix: np.ndarray, features: List[str], spec: OutlierSpec,
                 notes: List[str], where: str
                 ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], float]:
    """Squared robust Mahalanobis distance on an MCD covariance.

    :returns: ``(scores, complete, centres, threshold)``; ``scores`` is NaN for
        any row missing a feature, ``complete`` says which rows were used.
    :raises OutlierError: when there are too few complete objects for the
        number of features, naming the way out.
    """
    n, p = matrix.shape
    complete = np.isfinite(matrix).all(axis=1)
    usable = int(complete.sum())
    if usable <= p:
        raise OutlierError(
            f"{where}: the robust Mahalanobis test needs more {where} than "
            f"features and has {usable:,} complete {where[:-1]}(s) for {p} "
            f"features. Below n > p there is no covariance matrix to invert. "
            f"Either test fewer features, or run a PCA first "
            f"(spacr.qt.widgets.pca_model.pca) and flag on the first few "
            f"components — they are fewer, uncorrelated, and carry the same "
            f"information.")
    needed = MCD_MIN_OBJECTS_PER_FEATURE * p
    if usable < needed:
        raise OutlierError(
            f"{where}: {usable:,} complete {where[:-1]}(s) for {p} features is "
            f"too few for a Minimum Covariance Determinant. It is fitted on a "
            f"subset of about half the rows, so it is unstable until roughly "
            f"n = {MCD_MIN_OBJECTS_PER_FEATURE}p = {needed}. Either test fewer "
            f"features, or run a PCA first (spacr.qt.widgets.pca_model.pca) "
            f"and flag on the first few components.")
    from sklearn.covariance import MinCovDet
    try:
        estimator = MinCovDet(random_state=spec.seed,
                              support_fraction=spec.support_fraction)
        estimator.fit(matrix[complete])
        distances = estimator.mahalanobis(matrix[complete])
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise OutlierError(
            f"{where}: the MCD could not be fitted ({exc}). This is almost "
            f"always collinear features — two columns that are the same "
            f"measurement in different units leave a singular covariance with "
            f"no inverse and so no distance. Drop one of them, or run a PCA "
            f"first (spacr.qt.widgets.pca_model.pca) and flag on the "
            f"components, which are orthogonal by construction.") from exc
    scores = np.full(n, np.nan, dtype=float)
    scores[complete] = np.asarray(distances, dtype=float)
    centres = {name: float(value)
               for name, value in zip(features, estimator.location_)}
    threshold = _chi2_quantile(spec.alpha, p)
    support = float(np.mean(estimator.support_)) if hasattr(
        estimator, "support_") else float("nan")
    notes.append(
        f"{where}: MCD fitted on {support:.0%} of the {usable:,} complete "
        f"{where} (support_fraction="
        f"{'default' if spec.support_fraction is None else spec.support_fraction}"
        f", random_state={spec.seed}); threshold χ²({p}) at α={spec.alpha:g} "
        f"= {threshold:.3f}")
    return scores, complete, centres, threshold


def _well_summary(frame: pd.DataFrame, matrix: np.ndarray,
                  features: List[str], flags: np.ndarray, scored: np.ndarray,
                  keys: List[str]) -> pd.DataFrame:
    """Reduce the objects to one row per well: n, medians, flagged share.

    ``dropna=False`` on the groupby, so a well whose key is partly missing gets
    its own row instead of vanishing — an object with no ``rowID`` is a fact
    about the run, not a row to discard.
    """
    work = frame[keys].copy()
    median_names = []
    for i, name in enumerate(features):
        column = f"{name}__median"
        work[column] = matrix[:, i]
        median_names.append(column)
    work["__flagged"] = np.asarray(flags, dtype=bool)
    work["__scored"] = np.asarray(scored, dtype=bool)
    grouped = work.groupby(keys, dropna=False, sort=True, observed=True)
    summary = grouped[median_names].median()
    counts = grouped.size()
    flagged = grouped["__flagged"].sum()
    scored_count = grouped["__scored"].sum()
    wells = summary.reset_index()
    wells["n_objects"] = counts.to_numpy(dtype=int)
    wells["n_scored_objects"] = scored_count.to_numpy(dtype=int)
    wells["n_flagged_objects"] = flagged.to_numpy(dtype=int)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(wells["n_scored_objects"].to_numpy(dtype=float) > 0,
                         wells["n_flagged_objects"].to_numpy(dtype=float)
                         / np.maximum(
                             wells["n_scored_objects"].to_numpy(dtype=float),
                             1.0),
                         np.nan)
    wells["flagged_share"] = share
    return wells


def _score_wells(wells: pd.DataFrame, features: List[str], spec: OutlierSpec,
                 notes: List[str]) -> pd.DataFrame:
    """Run the same robust rule across wells, on their medians.

    Wells below :attr:`OutlierSpec.min_well_objects` are carried through
    unscored rather than dropped or scored, which is the difference between
    "this well is fine" and "nobody looked".
    """
    n_wells = len(wells)
    wells = wells.copy()
    wells["well_scored"] = wells["n_objects"].to_numpy(dtype=int) \
        >= spec.min_well_objects
    wells["well_outlier"] = False
    wells["well_outlier_score"] = np.nan
    wells["well_outlier_reason"] = [
        "" if ok else
        f"not scored: {int(n)} object(s) < the {spec.min_well_objects} "
        f"required for a stable median"
        for ok, n in zip(wells["well_scored"], wells["n_objects"])]

    eligible = np.flatnonzero(wells["well_scored"].to_numpy(dtype=bool))
    if eligible.size < MIN_WELLS_TO_SCORE:
        notes.append(
            f"the across-well rule did not run: {eligible.size} well(s) have "
            f"at least {spec.min_well_objects} objects and a median with a MAD "
            f"over fewer than {MIN_WELLS_TO_SCORE} points is arbitrary. Every "
            f"well is reported, none is scored.")
        wells["well_scored"] = False
        wells["well_outlier_reason"] = [
            reason or (f"not scored: only {eligible.size} well(s) qualify, "
                       f"fewer than the {MIN_WELLS_TO_SCORE} the across-well "
                       f"comparison needs")
            for reason in wells["well_outlier_reason"]]
        return wells

    median_columns = [f"{name}__median" for name in features]
    matrix = wells.loc[wells.index[eligible], median_columns] \
        .to_numpy(dtype=float)
    reasons = list(wells["well_outlier_reason"])
    scores = np.full(n_wells, np.nan, dtype=float)
    flags = np.zeros(n_wells, dtype=bool)

    if spec.method == METHOD_MAHALANOBIS:
        p = len(features)
        if eligible.size < MCD_MIN_OBJECTS_PER_FEATURE * p or eligible.size <= p:
            notes.append(
                f"the across-well MCD did not run: {eligible.size} scored "
                f"well(s) for {p} features is below the "
                f"n ≥ {MCD_MIN_OBJECTS_PER_FEATURE}p = "
                f"{MCD_MIN_OBJECTS_PER_FEATURE * p} an MCD needs. Reduce the "
                f"features (a PCA first, then its components) or read the "
                f"per-feature well medians directly.")
            wells["well_scored"] = False
            wells["well_outlier_reason"] = [
                reason or (f"not scored: too few wells "
                           f"({eligible.size}) for an MCD on {p} features")
                for reason in reasons]
            return wells
        well_scores, complete, _centres, threshold = _mahalanobis(
            matrix, features, spec, notes, "wells")
        scores[eligible] = well_scores
        hit = np.isfinite(well_scores) & (well_scores > threshold)
        flags[eligible[hit]] = True
        for local in np.flatnonzero(hit):
            reasons[eligible[local]] = (
                f"robust Mahalanobis d² = {well_scores[local]:.1f} > "
                f"{threshold:.1f} across wells")
        for local in np.flatnonzero(~complete):
            reasons[eligible[local]] = (
                "not scored: this well has no median for every feature")
    else:
        per_feature, centres, scales, fences = _score_per_feature(
            matrix, features, spec, notes, "wells")
        with np.errstate(invalid="ignore"):
            worst = np.where(np.isfinite(per_feature).any(axis=1),
                             np.nanmax(np.where(np.isfinite(per_feature),
                                                per_feature, -np.inf), axis=1),
                             np.nan)
        scores[eligible] = worst
        threshold = spec.threshold()
        hit = np.isfinite(worst) & (worst > threshold)
        flags[eligible[hit]] = True
        for local in np.flatnonzero(hit):
            i = int(np.nanargmax(np.where(np.isfinite(per_feature[local]),
                                          per_feature[local], -np.inf)))
            name = features[i]
            value = matrix[local, i]
            low, high = fences.get(name, (float("nan"), float("nan")))
            side = "high" if value > high else "low"
            unit = "z" if spec.method == METHOD_MAD else "IQR"
            reasons[eligible[local]] = (
                f"well median of {name} {side} across wells "
                f"({per_feature[local, i]:.1f} {unit})")

    wells["well_outlier"] = flags
    wells["well_outlier_score"] = scores
    wells["well_outlier_reason"] = reasons
    return wells


def detect_outliers(frame: pd.DataFrame,
                    spec: Optional[OutlierSpec] = None) -> OutlierResult:
    """Flag outlying objects and outlying wells in ``frame``.

    The policy is in the module docstring; the short version is that nothing
    is estimated from a mean or an SD, that a zero MAD falls back to a stated
    alternative rather than dividing by zero, that a log transform is never
    applied unless asked for, that the multivariate test is an MCD Mahalanobis
    distance cut at a chi-square quantile so its threshold is a stated
    false-positive rate, and that **no row is ever dropped or altered** — the
    result adds columns.

    The well pass reduces each well to the median of each feature and runs the
    same rule across wells, because a well shifted as a whole flags almost none
    of its individual objects and is nevertheless unmistakable as a point among
    wells.

    :param frame: an object-level measurement table.
    :param spec: what to test and how. ``None`` is :class:`OutlierSpec`'s
        defaults: the MAD rule at k = 3.5 over every continuous column, with
        the well pass on.
    :returns: an :class:`OutlierResult`. Write the flags back with
        :meth:`OutlierResult.object_frame`.
    :raises OutlierError: whenever the answer would be meaningless — with the
        reason and the way out in the message.
    """
    spec = spec or OutlierSpec()
    notes: List[str] = []
    n_rows_in = int(len(frame))
    if n_rows_in < 1:
        raise OutlierError(
            "this table has no objects, so there is nothing to test.")
    features = _select_features(frame, spec)
    matrix, n_non_finite = _matrix(frame, features, spec, notes)

    centres: Dict[str, float] = {}
    scales: Dict[str, float] = {}
    fences: Dict[str, Tuple[float, float]] = {}
    if spec.method == METHOD_MAHALANOBIS:
        scores, complete, centres, threshold = _mahalanobis(
            matrix, features, spec, notes, "objects")
        scored = complete
        flags = np.isfinite(scores) & (scores > threshold)
        reasons = [""] * n_rows_in
        for row in np.flatnonzero(flags):
            reasons[row] = (
                f"robust Mahalanobis d² = {scores[row]:.1f} > "
                f"{threshold:.1f} (χ²({len(features)}) at α={spec.alpha:g})")
        for row in np.flatnonzero(~scored):
            reasons[row] = (
                "not scored: the multivariate distance needs a finite value "
                "for every tested feature")
    else:
        per_feature, centres, scales, fences = _score_per_feature(
            matrix, features, spec, notes, "objects")
        usable = np.isfinite(per_feature)
        scored = usable.any(axis=1)
        with np.errstate(invalid="ignore"):
            scores = np.where(
                scored,
                np.max(np.where(usable, per_feature, -np.inf), axis=1),
                np.nan)
        threshold = spec.threshold()
        flags = scored & (scores > threshold)
        reasons = _reasons_per_feature(per_feature, matrix, features, flags,
                                       ~scored, spec, fences)
        partial = int((usable.sum(axis=1) < len(features)).sum()
                      - (~scored).sum())
        if partial > 0:
            notes.append(
                f"{partial:,} object(s) were scored on the features they have "
                f"a value for rather than on all {len(features)}")

    n_scored = int(np.count_nonzero(scored))
    column_names = _unique_names(OBJECT_COLUMNS, list(frame.columns), notes)

    wells = pd.DataFrame()
    well_keys: Tuple[str, ...] = ()
    n_wells_scored = 0
    if spec.per_well:
        keys = list(spec.well_keys) if spec.well_keys \
            else list(well_key_columns(frame))
        missing = [name for name in keys if name not in frame.columns]
        if missing:
            raise OutlierError(
                f"well_keys names {', '.join(missing)}, which this table does "
                f"not have. Its columns include: "
                f"{', '.join(str(c) for c in list(frame.columns)[:12])}.")
        well_keys = tuple(keys)
        wells = _well_summary(frame, matrix, features, flags, scored, keys)
        wells = _score_wells(wells, features, spec, notes)
        n_wells_scored = int(np.count_nonzero(
            wells["well_scored"].to_numpy(dtype=bool)))
        wells = wells.rename(columns={f"{name}__median": f"{name}_median"
                                      for name in features})

    return OutlierResult(
        method=spec.method, features=tuple(features), scores=scores,
        flags=np.asarray(flags, dtype=bool), reasons=tuple(reasons),
        threshold=float(threshold), n_rows_in=n_rows_in, n_scored=n_scored,
        centres=centres, scales=scales, fences=fences, wells=wells,
        well_keys=well_keys, n_wells_scored=n_wells_scored,
        min_well_objects=spec.min_well_objects, alpha=spec.alpha,
        transform=spec.transform, n_non_finite=n_non_finite,
        column_names=column_names, notes=tuple(notes))
