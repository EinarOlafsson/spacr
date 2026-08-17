"""Which cells to show behind one coefficient on the regression plot.

Instruction 131, headless half. Given a coefficient (a gene or a guide), the
per-object rows out of the imported measurement databases, and the per-well
guide fractions the regression was fitted on, this module decides **which
objects to draw and says exactly how it decided**. It draws nothing: the
pixels come from :mod:`spacr.crops`, and the tab that shows them is wired
separately.

THE TRAP, AND WHY IT DECIDES THE OUTPUT SHAPE
---------------------------------------------
**This is a pooled screen.** Which individual cell carries which guide is not
known and cannot be looked up -- that is the whole reason the analysis is a
regression on well-level fractions instead of a per-cell comparison. The
sequencing says a well is 15% GRA14; it does not say WHICH 15%.

So a montage cannot show "the cells with GRA14 knocked out". It can show the
cells most consistent with the effect, and that distinction has to survive
onto the figure. Every :class:`MontagePlan` therefore carries
:data:`INFERENCE_NOTICE` in its :meth:`~MontagePlan.caption`, and there is no
way to get a plan without it -- a montage captioned "GRA14" that a reader
takes for genotyped cells is the worst thing this feature can produce.

THE SELECTION, AND THE THREE NUMBERS IT TURNS ON
------------------------------------------------
1. **The wells.** Those whose count data reports the guide/gene present, with
   a non-zero fraction (:func:`wells_for_coefficient`). A gene with several
   guides sums their fractions by default; :func:`select_montage_per_guide`
   is the other question, asked separately, because they are different
   questions.

2. **The score window.** ``target = baseline + effect``; keep objects whose
   classification score lies within ``half_widths`` robust scales of that
   target, and among them take the closest (:func:`score_window`).

   *"Closest" means smallest ``|score - target|``.* The baseline and the
   scale are computed **once, over every object supplied**, not per gene:
   a window recomputed per gene is a window that can be tuned until the
   pictures look right, and nothing in the output would show that it had
   been. :data:`WINDOW_HALF_WIDTHS` is the one width, and a caller that
   overrides it has the override written into the caption.

3. **How many.** ``round(n_objects_in_well * guide_fraction_in_well)``, per
   well (:func:`objects_to_show`). That is the *expected* number of objects
   in that well carrying the guide, and it is the right count precisely
   because it is the only number the pooled design supports. A well of 200
   cells at 15% contributes 30. A well that rounds to zero is reported as a
   zero-contribution well rather than silently vanishing.

WHERE THE PIXELS COME FROM -- ALREADY BUILT, NOT REBUILT HERE
--------------------------------------------------------------
:func:`spacr.crops.resolve_crop_source` picks between
:class:`~spacr.crops.PngCropSource` (the exported PNGs, via ``png_list``) and
:class:`~spacr.crops.MergedCropSource` (cut on demand out of
``merged/<fov>.npy``) **and says which it picked**.
:func:`resolve_montage_crop_source` is a thin wrapper that turns the "no
source at all" case into an answer the tab can display instead of an
exception, because a tab that cannot be filled has to say why.

WHICH CSV ACTUALLY CARRIES THE WELL FRACTIONS
----------------------------------------------
``grna_well.csv`` and ``well_grna.csv`` do **not**, despite being the obvious
candidates. Measured against ``spacr.ml.grna_metricks`` and the contract test
``test_qc_block_writes_the_three_well_level_tables`` in
``tests/test_cov_ml_perform_regression.py``:

    ``grna_well.csv``   ``grna, plateID, grna_well_count, gene_well_count``
                        -- how MANY wells a guide was seen in, never which.
    ``well_grna.csv``   ``prc, gene_count`` (+ the split ``prc`` parts)
                        -- how many distinct genes per well, and it does not
                        name a guide at all.
    ``regression_data.csv``
                        ``prc, grna, gene, fraction, pred, cell_count,
                        plateID, rowID, columnID`` -- the well, the guide,
                        the fraction and the cell count, which is exactly and
                        only what steps 1 and 3 need.

:func:`read_well_guide_fractions` reads the third and refuses the first two by
name, saying what each holds, so the mistake is made once.

Dependencies
------------
numpy, pandas and the standard library. :mod:`spacr.crops` and
:mod:`spacr.io` are imported lazily inside the functions that need them, so a
caller that only wants the selection never pays for torch.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "INFERENCE_NOTICE",
    "SCORE_TARGET_RULE",
    "WINDOW_HALF_WIDTHS",
    "MAX_OBJECTS",
    "WELL_KEY_COLUMNS",
    "GUIDE_AGGREGATIONS",
    "MontageError",
    "CoefficientNotFound",
    "MissingScores",
    "Coefficient",
    "ScoreWindow",
    "WellSelection",
    "MontagePlan",
    "CropSourceChoice",
    "round_half_up",
    "objects_to_show",
    "coefficient_level",
    "guides_for_coefficient",
    "wells_for_coefficient",
    "score_window",
    "select_montage",
    "select_montage_per_guide",
    "read_well_guide_fractions",
    "load_montage_objects",
    "resolve_montage_crop_source",
]


#: The sentence that must ride on every montage. It is not decoration and it
#: is not a tooltip: the montage is a picture of cells that a reader will
#: otherwise read as genotyped, and the pooled design cannot support that
#: reading. :meth:`MontagePlan.caption` always ends with it, filled in with
#: the coefficient's own name.
INFERENCE_NOTICE = (
    "Guide membership is INFERRED, not observed. This is a pooled screen: the "
    "sequencing reports what FRACTION of each well carried {name}, never which "
    "cells did. The objects shown are those whose classification score sits "
    "closest to what the coefficient implies, drawn from wells where {name} "
    "was detected. They are candidates consistent with the effect, not "
    "genotyped cells, and the per-well count is the expected number the "
    "fraction supports rather than a set of identified objects."
)

#: How the per-object score a coefficient implies is computed. Named so the
#: caption can state the rule instead of a bare number.
SCORE_TARGET_RULE = "baseline + effect"

#: Half-width of the score window, in robust scales (MAD-sigma) of the
#: per-object score computed over EVERY object supplied.
#:
#: One number, for the whole screen, for every coefficient. Choosing it per
#: gene is choosing the pictures, and a figure produced that way carries no
#: evidence that it was. 1.0 is a full robust standard deviation either side
#: of the target -- wide enough that a well with a normal object count can
#: usually fill its quota, narrow enough that "closest" still means something.
WINDOW_HALF_WIDTHS = 1.0

#: Default cap on the number of objects one montage may contain.
#:
#: A montage for one gene can span dozens of wells, and each merged array is a
#: whole field, so the cost had to be measured before a default was chosen
#: rather than after. 300 crops of 224x224, cut through the real sources:
#:
#: ===========================  ==============  ==============
#: fixture                      merged/*.npy    exported PNGs
#: ===========================  ==============  ==============
#: 6 fields of 1024x1024x7       2.58 ms/crop    0.49 ms/crop
#: 6 fields of 2048x2048x7       5.01 ms/crop    0.64 ms/crop
#: 30 fields of 2048x2048x7     11.43 ms/crop    0.62 ms/crop
#: ===========================  ==============  ==============
#:
#: The finding is in the third row: the same 300 crops cost more than twice
#: as much spread over 30 fields as over 6. **The merged source is priced by
#: how many FIELDS a montage touches, not by how many crops it cuts** -- so a
#: cap on objects is only half a bound, and a montage that spans many wells is
#: the expensive one however few objects it draws from each. The PNG source is
#: ~10x cheaper and flat in both, which is a second reason
#: :func:`spacr.crops.resolve_crop_source` is right to prefer it when it
#: exists.
#:
#: 300 is therefore ~3.4 s of cutting in the worst measured case and ~0.2 s in
#: the common one. Whatever the cap trims is named in the caption rather than
#: dropped quietly.
MAX_OBJECTS = 300

#: The well key, in the spelling every spaCR table uses.
WELL_KEY_COLUMNS: Tuple[str, ...] = ("plateID", "rowID", "columnID")

#: How a gene's several guides are combined. They are different questions and
#: the plan says which was asked.
GUIDE_AGGREGATIONS: Tuple[str, ...] = ("sum", "separate")


class MontageError(ValueError):
    """The requested montage cannot be selected from what was supplied."""


class CoefficientNotFound(MontageError):
    """The gene or guide named by the coefficient is not in the count data."""


class MissingScores(MontageError):
    """The object frame carries no usable per-object classification score."""


# ---------------------------------------------------------------------------
# The count rule
# ---------------------------------------------------------------------------

def round_half_up(value: float) -> int:
    """Round ``value`` to the nearest integer, halves away from zero.

    Deliberately **not** :func:`round` or :func:`numpy.round`, both of which
    round halves to even: a well whose ``n * fraction`` is 2.5 would then
    contribute 2 while an otherwise identical well at 3.5 contributed 4. That
    is a montage whose per-well counts depend on the parity of a number the
    caption calls "round(n x fraction)", which no reader could reconstruct.

    :param value: the number to round. Non-finite input raises rather than
        producing a count.
    :returns: the rounded integer.
    """
    number = float(value)
    if not math.isfinite(number):
        raise MontageError(f"cannot round a non-finite count: {value!r}")
    if number < 0:
        return -int(math.floor(-number + 0.5))
    return int(math.floor(number + 0.5))


def objects_to_show(n_objects: int, fraction: float) -> int:
    """Return ``round(n_objects * fraction)`` -- the montage's count rule.

    This is the EXPECTED number of objects in the well carrying the guide,
    and it is the right count precisely because it is the only number a
    pooled design supports: the sequencing gives a fraction, so the design
    gives a count and never an identity.

    :param n_objects: how many objects the well actually has to draw from.
        Negative counts are refused; zero is legal and yields zero.
    :param fraction: the guide's fraction in that well, in ``[0, 1]``. A
        fraction outside that range is refused rather than clipped -- it means
        the count data and the well key did not line up, and a clipped
        montage would hide that.
    :returns: how many objects that well contributes.
    """
    count = int(n_objects)
    if count < 0:
        raise MontageError(f"a well cannot hold {count} objects")
    share = float(fraction)
    if not math.isfinite(share) or not 0.0 <= share <= 1.0:
        raise MontageError(
            f"guide fraction {fraction!r} is not a fraction in [0, 1]; the "
            "count data and the object frame do not describe the same wells")
    return round_half_up(count * share)


# ---------------------------------------------------------------------------
# The coefficient
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Coefficient:
    """One point on the regression plot: a gene or a guide, and its effect.

    :param name: the gene or guide name as the count data spells it.
    :param effect: the fitted coefficient.
    :param level: ``'gene'`` or ``'grna'``.
    :param guides: the guides the coefficient covers -- one for a guide-level
        coefficient, all of the gene's for a gene-level one.
    """

    name: str
    effect: float
    level: str = "gene"
    guides: Tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "effect", float(self.effect))
        object.__setattr__(self, "guides", tuple(str(g) for g in self.guides))
        if self.level not in ("gene", "grna"):
            raise MontageError(
                f"level must be 'gene' or 'grna', got {self.level!r}")
        if not math.isfinite(self.effect):
            raise MontageError(
                f"coefficient {self.name!r} has a non-finite effect "
                f"({self.effect!r}); there is no score it implies")

    def describe(self) -> str:
        """Return ``'GRA14 (gene, effect +0.700)'`` for a caption or a log."""
        return f"{self.name} ({self.level}, effect {self.effect:+.3f})"


# ---------------------------------------------------------------------------
# The score window
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoreWindow:
    """The band of classification scores a coefficient implies, and its rule.

    :param target: the per-object score the coefficient implies.
    :param low: the window's lower bound, inclusive.
    :param high: the window's upper bound, inclusive.
    :param baseline: the score of an object carrying no guide of interest.
    :param baseline_source: ``'screen_median'`` when derived from every
        object supplied, ``'given'`` when the caller passed one.
    :param scale: the robust scale (MAD-sigma) the half-width is measured in.
    :param half_widths: how many scales wide each side is.
    :param n_scored: how many objects the baseline and scale were computed
        from -- the whole supplied frame, not the selected wells.
    :param observed_low: smallest score seen across all objects.
    :param observed_high: largest score seen across all objects.
    :param degenerate: True when every score is identical, so the window has
        no width to speak of and admits everything.
    """

    target: float
    low: float
    high: float
    baseline: float
    baseline_source: str
    scale: float
    half_widths: float
    n_scored: int
    observed_low: float
    observed_high: float
    degenerate: bool = False

    @property
    def target_is_observable(self) -> bool:
        """True when the implied score lies inside the observed score range.

        False is a real finding, not an error: it says no object anywhere in
        the screen scores anything like what the coefficient implies, so the
        montage is showing the least-far objects rather than close ones.
        """
        return self.observed_low <= self.target <= self.observed_high

    def contains(self, scores) -> np.ndarray:
        """Return the boolean mask of ``scores`` that fall inside the window.

        :param scores: any array-like of per-object scores. Non-finite
            entries are outside the window whatever the bounds are -- a NaN
            score is a missing measurement, not a near miss.
        """
        values = np.asarray(scores, dtype=float)
        finite = np.isfinite(values)
        return finite & (values >= self.low) & (values <= self.high)

    def describe(self) -> str:
        """Return the one-line statement of the rule and the width."""
        if self.degenerate:
            return (
                f"score window: every object scores {self.baseline:.4g}, so "
                f"the window has no width and admits all of them "
                f"(target {self.target:.4g} = {SCORE_TARGET_RULE})")
        return (
            f"score window: {self.low:.4g} to {self.high:.4g} "
            f"(target {self.target:.4g} = {SCORE_TARGET_RULE}, "
            f"baseline {self.baseline:.4g} from {self.baseline_source}, "
            f"+/-{self.half_widths:g} robust scales of {self.scale:.4g} "
            f"measured once over all {self.n_scored:,} objects, not per gene)")


def _finite_scores(objects: pd.DataFrame, score_column: str) -> np.ndarray:
    if score_column not in objects.columns:
        raise MissingScores(
            f"the object frame has no {score_column!r} column; it has "
            f"{list(objects.columns)[:15]}. A montage needs a per-object "
            "classification score -- 'pred' is the column "
            "spacr.predictions.merge_cv_predictions writes into png_list.")
    values = pd.to_numeric(objects[score_column], errors="coerce").to_numpy(
        dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise MissingScores(
            f"no object carries a finite {score_column!r}; the classification "
            "step has not been merged into this database")
    return finite


def score_window(objects: pd.DataFrame, effect: float, *,
                 score_column: str = "pred",
                 half_widths: float = WINDOW_HALF_WIDTHS,
                 baseline: Optional[float] = None) -> ScoreWindow:
    """Return the band of scores the coefficient ``effect`` implies.

    THE RULE, stated once and applied identically to every coefficient:

    * the **target** is ``baseline + effect``. Under the well-level model a
      well's score is ``baseline + fraction * effect``, so a well made
      entirely of cells carrying the guide would score ``baseline + effect``
      -- which is the score one such cell implies.
    * the **baseline** is the median per-object score over *every* object
      supplied. One number for the screen, so it does not move when the gene
      does.
    * "**closest**" is smallest ``|score - target|``, and the window is
      ``target +/- half_widths * scale`` where ``scale`` is
      ``1.4826 * MAD`` of the same screen-wide score distribution.

    :param objects: every object available to the montage -- all wells, not
        just the coefficient's. Passing only the selected wells is what makes
        a window per-gene, so the whole frame is the argument.
    :param effect: the fitted coefficient.
    :param score_column: the per-object classification score.
    :param half_widths: the window's half-width in robust scales. Overriding
        :data:`WINDOW_HALF_WIDTHS` is allowed and is written into the caption,
        because a widened window changes which cells a reader is looking at.
    :param baseline: use this baseline instead of the screen median -- the
        fitted intercept, say. Recorded as ``'given'``.
    :raises MissingScores: no usable score column, or no finite score in it.
    :raises MontageError: a non-positive or non-finite ``half_widths``.
    """
    width = float(half_widths)
    if not math.isfinite(width) or width <= 0:
        raise MontageError(
            f"half_widths must be a positive number of robust scales, got "
            f"{half_widths!r}")
    values = _finite_scores(objects, score_column)
    if baseline is None:
        base = float(np.median(values))
        base_source = "screen_median"
    else:
        base = float(baseline)
        if not math.isfinite(base):
            raise MontageError(f"baseline {baseline!r} is not a finite score")
        base_source = "given"

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale <= 0:
        # Every score identical, or a distribution so concentrated the MAD
        # underflows. Fall back to the plain standard deviation rather than
        # inventing a width; if that is zero too the window is degenerate and
        # says so instead of admitting nothing.
        scale = float(np.std(values))
    degenerate = not (scale > 0)

    target = base + float(effect)
    if degenerate:
        low, high = -math.inf, math.inf
    else:
        low, high = target - width * scale, target + width * scale
    return ScoreWindow(
        target=target, low=low, high=high, baseline=base,
        baseline_source=base_source, scale=float(scale), half_widths=width,
        n_scored=int(values.size), observed_low=float(values.min()),
        observed_high=float(values.max()), degenerate=degenerate)


# ---------------------------------------------------------------------------
# Wells
# ---------------------------------------------------------------------------

def _require(frame: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise MontageError(
            f"{what} is missing {missing}; it has {list(frame.columns)[:15]}")


def _well_key(frame: pd.DataFrame, what: str) -> List[str]:
    """Return the well key columns present on ``frame``, preferring ``prc``."""
    if "prc" in frame.columns:
        return ["prc"]
    if all(c in frame.columns for c in WELL_KEY_COLUMNS):
        return list(WELL_KEY_COLUMNS)
    raise MontageError(
        f"{what} names no well: it needs a 'prc' column or all of "
        f"{list(WELL_KEY_COLUMNS)}, and it has {list(frame.columns)[:15]}")


def coefficient_level(counts: pd.DataFrame, name: str, *,
                      guide_column: str = "grna",
                      gene_column: str = "gene") -> str:
    """Return ``'grna'`` or ``'gene'`` for ``name``, read off the count data.

    A guide wins a tie. A library where one string is both a guide name and a
    gene name is pathological, and resolving it as the gene would silently
    widen the montage to that gene's other guides.

    :param counts: the per-well count data.
    :param name: the coefficient's name.
    :param guide_column: the guide column, default ``'grna'``.
    :param gene_column: the gene column, default ``'gene'``.
    :raises CoefficientNotFound: the name is in neither column.
    """
    text = str(name)
    if guide_column in counts.columns:
        if (counts[guide_column].astype(str) == text).any():
            return "grna"
    if gene_column in counts.columns:
        if (counts[gene_column].astype(str) == text).any():
            return "gene"
    raise CoefficientNotFound(
        f"{name!r} is neither a {guide_column!r} nor a {gene_column!r} in the "
        f"count data ({len(counts):,} rows). The montage cannot name wells "
        "for a coefficient the count data has never seen.")


def guides_for_coefficient(counts: pd.DataFrame, name: str, *,
                           level: Optional[str] = None,
                           guide_column: str = "grna",
                           gene_column: str = "gene") -> List[str]:
    """Return the guides a coefficient covers, in sorted order.

    :param counts: the per-well count data.
    :param name: the coefficient's name.
    :param level: ``'gene'`` / ``'grna'``; ``None`` reads it off the data
        with :func:`coefficient_level`.
    :param guide_column: the guide column.
    :param gene_column: the gene column.
    :returns: one guide for a guide-level coefficient, the gene's guides for
        a gene-level one.
    """
    resolved = level or coefficient_level(
        counts, name, guide_column=guide_column, gene_column=gene_column)
    text = str(name)
    if resolved == "grna":
        return [text]
    _require(counts, [guide_column, gene_column], "count data")
    hit = counts[counts[gene_column].astype(str) == text]
    return sorted({str(v) for v in hit[guide_column]})


def wells_for_coefficient(counts: pd.DataFrame, name: str, *,
                          level: Optional[str] = None,
                          guide_aggregation: str = "sum",
                          guide_column: str = "grna",
                          gene_column: str = "gene",
                          fraction_column: str = "fraction") -> pd.DataFrame:
    """Return the wells whose count data reports ``name`` present.

    Step 1 of the selection. A gene's guides are summed by default -- the
    gene-level coefficient describes the fraction of the well carrying ANY
    guide against the gene -- and the sum is refused if it exceeds 1, because
    that means the same well was counted twice.

    :param counts: per-well count data: a well key (``prc``, or all of
        ``plateID``/``rowID``/``columnID``), ``grna``, ``gene`` and
        ``fraction``. ``regression_data.csv`` is that frame;
        ``grna_well.csv`` and ``well_grna.csv`` are NOT -- see
        :func:`read_well_guide_fractions`.
    :param name: the coefficient's name.
    :param level: ``'gene'`` / ``'grna'``; ``None`` reads it off the data.
    :param guide_aggregation: ``'sum'`` (one fraction per well, the default)
        or ``'separate'`` (a ``grna`` column is kept and one row per
        well/guide comes back, for :func:`select_montage_per_guide`).
    :param guide_column: the guide column.
    :param gene_column: the gene column.
    :param fraction_column: the per-well guide fraction column.
    :returns: one row per well (or per well/guide) with the well key,
        ``fraction``, and ``cell_count`` when the count data carries it.
    :raises CoefficientNotFound: no well reports the coefficient present.
    """
    if guide_aggregation not in GUIDE_AGGREGATIONS:
        raise MontageError(
            f"guide_aggregation must be one of {list(GUIDE_AGGREGATIONS)}, "
            f"got {guide_aggregation!r}")
    _require(counts, [guide_column, fraction_column], "count data")
    keys = _well_key(counts, "count data")

    guides = guides_for_coefficient(
        counts, name, level=level, guide_column=guide_column,
        gene_column=gene_column)
    frame = counts[counts[guide_column].astype(str).isin(set(guides))].copy()
    if frame.empty:
        raise CoefficientNotFound(
            f"no well in the count data reports {name!r} (guides "
            f"{guides}) present")

    fractions = pd.to_numeric(frame[fraction_column], errors="coerce")
    if not np.isfinite(fractions.to_numpy(dtype=float)).all():
        raise MontageError(
            f"{fraction_column!r} holds missing or non-numeric values for "
            f"{name!r}; a montage count computed from one would be a guess")
    if ((fractions < 0) | (fractions > 1)).any():
        raise MontageError(
            f"{fraction_column!r} must lie in [0, 1]; {name!r} has values "
            f"from {fractions.min():.4g} to {fractions.max():.4g}. A "
            "percentage is not a fraction.")
    frame[fraction_column] = fractions.astype(float)

    if guide_aggregation == "separate":
        group = keys + [guide_column]
        out = frame.groupby(group, dropna=False, as_index=False)[
            fraction_column].sum()
    else:
        out = frame.groupby(keys, dropna=False, as_index=False)[
            fraction_column].sum()
        if (out[fraction_column] > 1.0 + 1e-9).any():
            worst = out.loc[out[fraction_column].idxmax()]
            raise MontageError(
                f"the guides of {name!r} sum to "
                f"{worst[fraction_column]:.4g} in one well, which is more "
                "than the whole well. The count data holds the same well "
                "twice, or two sequencing runs were concatenated without "
                "being aggregated.")

    if "cell_count" in frame.columns:
        counted = frame.groupby(keys, dropna=False, as_index=False)[
            "cell_count"].max()
        out = out.merge(counted, on=keys, how="left")

    out = out.rename(columns={fraction_column: "fraction"})
    present = out[out["fraction"] > 0].copy()
    if present.empty:
        raise CoefficientNotFound(
            f"every well that lists {name!r} reports a fraction of zero, so "
            "the count data says the guide is present nowhere")
    sort_key = keys + ([guide_column] if guide_aggregation == "separate" else [])
    return present.sort_values(sort_key).reset_index(drop=True)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WellSelection:
    """What one well contributed to the montage, and why that number.

    :param well: the well key as a display string.
    :param fraction: the guide's fraction in this well, from the count data.
    :param n_objects: how many objects the object frame holds for this well
        -- the population ``round(n x fraction)`` is computed from.
    :param n_reported: ``cell_count`` as the count data recorded it, or
        ``None``. Kept beside ``n_objects`` rather than instead of it,
        because when they disagree the montage has to say so.
    :param n_expected: ``round(n_objects x fraction)`` -- the count rule.
    :param n_in_window: how many of this well's objects fall inside the
        score window at all.
    :param n_selected: how many were actually taken, after the window and
        after any cap.
    :param note: why ``n_selected`` is not ``n_expected``, or ``''``.
    """

    well: str
    fraction: float
    n_objects: int
    n_reported: Optional[int]
    n_expected: int
    n_in_window: int
    n_selected: int
    note: str = ""

    @property
    def contributed(self) -> bool:
        """True when this well put at least one object into the montage."""
        return self.n_selected > 0

    def describe(self) -> str:
        """Return the one-line account of this well's contribution."""
        line = (f"{self.well}: {self.n_selected} of "
                f"round({self.n_objects} x {self.fraction:.4g}) = "
                f"{self.n_expected}")
        return f"{line} -- {self.note}" if self.note else line


@dataclass(frozen=True)
class MontagePlan:
    """The objects to show behind one coefficient, and the whole reason why.

    The plan is the module's product. It carries the selected rows *and*
    every number the caption needs, so nothing downstream has to re-derive
    (or re-invent) the selection in order to describe it.

    :param coefficient: the point that was clicked.
    :param window: the score window that was applied.
    :param wells: one :class:`WellSelection` per well the count data reported
        the coefficient present in -- **including the wells that contributed
        nothing**, which is the whole reason they are a list of records and
        not a filtered frame.
    :param objects: the selected object rows, in well order then by distance
        to the target, with ``montage_distance``, ``montage_well`` and
        ``montage_rank`` added.
    :param score_column: the per-object score the selection used.
    :param guide_aggregation: ``'sum'`` or ``'separate'``.
    :param guides: the guides the coefficient covers.
    :param source_kind: ``'png'`` / ``'merged'`` / ``''`` -- which crop source
        will draw these, when one was resolved.
    :param source_reason: why that source was picked.
    :param cap: the cap that was applied.
    :param n_before_cap: how many objects the rule selected before the cap.
    :param notes: everything the caption has to disclose that is not a
        per-well count.
    """

    coefficient: Coefficient
    window: ScoreWindow
    wells: Tuple[WellSelection, ...]
    objects: pd.DataFrame
    score_column: str = "pred"
    guide_aggregation: str = "sum"
    guides: Tuple[str, ...] = ()
    source_kind: str = ""
    source_reason: str = ""
    cap: int = MAX_OBJECTS
    n_before_cap: int = 0
    notes: Tuple[str, ...] = ()

    @property
    def n_objects(self) -> int:
        """How many objects the montage holds."""
        return int(len(self.objects))

    @property
    def is_empty(self) -> bool:
        """True when no object survived the selection.

        The tab shows the caption anyway: "no objects" with the wells and the
        window that produced none is an answer, and an empty tab is not.
        """
        return self.n_objects == 0

    @property
    def capped(self) -> bool:
        """True when the cap trimmed the montage."""
        return self.n_before_cap > self.n_objects

    @property
    def zero_wells(self) -> Tuple[WellSelection, ...]:
        """The wells that reported the guide but contributed no object."""
        return tuple(w for w in self.wells if not w.contributed)

    def rows(self) -> List[Dict[str, Any]]:
        """Return the selected objects as plain dicts for a crop source.

        :class:`spacr.crops.PngCropSource` and
        :class:`~spacr.crops.MergedCropSource` both take a row mapping, so
        this is what the tab hands to ``source.get_many(...)``.
        """
        return self.objects.to_dict("records")

    def summary(self) -> str:
        """Return the one-line status-bar sentence."""
        drawn = f"{self.n_objects:,} objects from {len(self.wells)} wells"
        if self.capped:
            drawn += f" (capped from {self.n_before_cap:,})"
        source = f" via the {self.source_kind} crop source" if self.source_kind else ""
        return f"{self.coefficient.describe()}: {drawn}{source}"

    def caption(self) -> str:
        """Return the caption the montage must carry.

        States the wells, the score window, the count rule, which crop source
        drew the pixels, every well that contributed nothing, and -- last, so
        it is the sentence a reader leaves with -- that guide membership is
        inferred rather than observed.
        """
        lines: List[str] = []
        lines.append(f"Cells behind {self.coefficient.describe()}")
        if self.guides and self.coefficient.level == "gene":
            joined = ", ".join(self.guides)
            how = ("fractions summed across the gene's guides"
                   if self.guide_aggregation == "sum"
                   else "one guide at a time")
            lines.append(f"guides: {joined} ({how})")
        contributing = [w for w in self.wells if w.contributed]
        lines.append(
            f"wells: {len(contributing)} of {len(self.wells)} wells reporting "
            f"the guide contributed an object "
            f"({', '.join(w.well for w in contributing) or 'none'})")
        lines.append(self.window.describe())
        if not self.window.target_is_observable:
            lines.append(
                f"the implied score {self.window.target:.4g} lies OUTSIDE the "
                f"observed range [{self.window.observed_low:.4g}, "
                f"{self.window.observed_high:.4g}]: no object in this screen "
                "scores anything like it")
        lines.append(
            "count per well: round(objects in well x guide fraction in well) "
            "-- the expected number the pooled design supports")
        if self.source_kind:
            lines.append(f"images: {self.source_kind} crop source "
                         f"({self.source_reason})" if self.source_reason
                         else f"images: {self.source_kind} crop source")
        if self.capped:
            lines.append(
                f"capped at {self.cap:,} of {self.n_before_cap:,} objects; "
                "the ones closest to the implied score were kept")
        for well in self.zero_wells:
            lines.append(f"contributed nothing -- {well.describe()}")
        lines.extend(self.notes)
        lines.append(INFERENCE_NOTICE.format(name=self.coefficient.name))
        return "\n".join(lines)


def _well_labels(frame: pd.DataFrame, keys: Sequence[str]) -> pd.Series:
    if list(keys) == ["prc"]:
        return frame["prc"].astype(str)
    parts = [frame[k].astype(str) for k in keys]
    joined = parts[0]
    for part in parts[1:]:
        joined = joined + "_" + part
    return joined


def _shared_well_key(objects: pd.DataFrame, wells: pd.DataFrame) -> List[str]:
    """Return a well key both frames carry, preferring ``prc``."""
    if "prc" in objects.columns and "prc" in wells.columns:
        return ["prc"]
    if (all(c in objects.columns for c in WELL_KEY_COLUMNS)
            and all(c in wells.columns for c in WELL_KEY_COLUMNS)):
        return list(WELL_KEY_COLUMNS)
    raise MontageError(
        "the object frame and the count data share no well key. Both need "
        f"'prc', or both need all of {list(WELL_KEY_COLUMNS)}. Objects have "
        f"{list(objects.columns)[:12]}; counts have {list(wells.columns)[:12]}.")


def _object_sort_key(objects: pd.DataFrame) -> pd.Series:
    """A stable per-object tie-break, so the same montage comes back twice."""
    for column in ("prcfo", "png_path", "path", "object_label"):
        if column in objects.columns:
            return objects[column].astype(str)
    return pd.Series(objects.index.astype(str), index=objects.index)


def select_montage(objects: pd.DataFrame, counts: pd.DataFrame,
                   name: str, effect: float, *,
                   level: Optional[str] = None,
                   score_column: str = "pred",
                   half_widths: float = WINDOW_HALF_WIDTHS,
                   baseline: Optional[float] = None,
                   cap: int = MAX_OBJECTS,
                   guide_aggregation: str = "sum",
                   guide_column: str = "grna",
                   gene_column: str = "gene",
                   fraction_column: str = "fraction",
                   crop_source: Optional["CropSourceChoice"] = None,
                   guides: Optional[Sequence[str]] = None,
                   ) -> MontagePlan:
    """Return the objects to show behind one coefficient.

    The whole selection, in the order instruction 131 states it: the wells
    the count data reports the guide present in, the objects in those wells
    whose classification score is closest to what the coefficient implies,
    and ``round(n x fraction)`` of them per well.

    :param objects: **every** object available -- all wells, not just the
        coefficient's. The window's baseline and scale are computed from this
        whole frame precisely so that they do not move when the gene does, so
        pre-filtering it to the coefficient's wells is the one thing that
        makes the window tunable.
    :param counts: the per-well count data (``regression_data.csv``).
    :param name: the coefficient's gene or guide name.
    :param effect: its fitted coefficient.
    :param level: ``'gene'`` / ``'grna'``; ``None`` reads it off the counts.
    :param score_column: the per-object classification score.
    :param half_widths: the score window's half-width in robust scales.
    :param baseline: an explicit baseline instead of the screen median.
    :param cap: the largest montage to return; the objects closest to the
        implied score survive, and the caption says what was trimmed.
    :param guide_aggregation: ``'sum'`` or ``'separate'``. ``'separate'`` is
        served by :func:`select_montage_per_guide`; passing it here is an
        error rather than a silent sum.
    :param guide_column: the guide column in ``counts``.
    :param gene_column: the gene column in ``counts``.
    :param fraction_column: the per-well guide fraction column in ``counts``.
    :param crop_source: an already-resolved :class:`CropSourceChoice`, so the
        plan can say which source will draw it.
    :param guides: override the guides the coefficient covers -- used by
        :func:`select_montage_per_guide` to plan one guide at a time.
    :returns: the :class:`MontagePlan`.
    :raises CoefficientNotFound: no well reports the coefficient present.
    :raises MissingScores: the object frame carries no usable score.
    """
    if guide_aggregation == "separate" and guides is None:
        raise MontageError(
            "guide_aggregation='separate' asks a different question -- one "
            "montage per guide. Call select_montage_per_guide(), which "
            "returns one plan for each.")
    if guide_aggregation not in GUIDE_AGGREGATIONS:
        raise MontageError(
            f"guide_aggregation must be one of {list(GUIDE_AGGREGATIONS)}, "
            f"got {guide_aggregation!r}")
    limit = int(cap)
    if limit <= 0:
        raise MontageError(f"cap must be a positive object count, got {cap!r}")
    if objects.empty:
        raise MontageError(
            "the object frame is empty; there is nothing to draw a montage "
            "from. Import a measurement database first.")

    resolved_level = level or coefficient_level(
        counts, name, guide_column=guide_column, gene_column=gene_column)
    if guides is None:
        covered = guides_for_coefficient(
            counts, name, level=resolved_level, guide_column=guide_column,
            gene_column=gene_column)
        selected_counts = counts
    else:
        covered = [str(g) for g in guides]
        selected_counts = counts[
            counts[guide_column].astype(str).isin(set(covered))]
        if selected_counts.empty:
            raise CoefficientNotFound(
                f"no well in the count data reports guides {covered} present")

    coefficient = Coefficient(name=name, effect=effect, level=resolved_level,
                              guides=tuple(covered))
    window = score_window(objects, coefficient.effect,
                          score_column=score_column, half_widths=half_widths,
                          baseline=baseline)

    well_frame = wells_for_coefficient(
        selected_counts, name, level=resolved_level,
        guide_aggregation="sum", guide_column=guide_column,
        gene_column=gene_column, fraction_column=fraction_column)
    keys = _shared_well_key(objects, well_frame)

    work = objects.copy()
    work["montage_well"] = _well_labels(work, keys)
    scores = pd.to_numeric(work[score_column], errors="coerce").to_numpy(
        dtype=float)
    work["montage_distance"] = np.abs(scores - window.target)
    work["_montage_in_window"] = window.contains(scores)
    work["_montage_tiebreak"] = _object_sort_key(work)

    # Always key on a tuple. ``groupby(['prc'])`` yields a bare string on
    # pandas 1.x and a 1-tuple on 2.2+, so a lookup written for either one
    # silently matches nothing on the other -- and "nothing matched" here is
    # an empty montage with a caption that still reads as if it worked.
    grouped: Dict[Tuple[Any, ...], pd.DataFrame] = {}
    for label, frame in work.groupby(keys, dropna=False):
        grouped[label if isinstance(label, tuple) else (label,)] = frame
    notes: List[str] = []
    selections: List[WellSelection] = []
    chosen: List[pd.DataFrame] = []
    mismatched: List[str] = []

    for _, row in well_frame.iterrows():
        here = grouped.get(tuple(row[k] for k in keys))
        label = "_".join(str(row[k]) for k in keys)
        fraction = float(row["fraction"])
        reported = row.get("cell_count")
        n_reported = (int(reported)
                      if reported is not None and pd.notna(reported) else None)
        if here is None or here.empty:
            selections.append(WellSelection(
                well=label, fraction=fraction, n_objects=0,
                n_reported=n_reported, n_expected=0, n_in_window=0,
                n_selected=0,
                note="no object in the imported databases comes from this well"))
            continue

        n_objects = int(len(here))
        if n_reported is not None and n_reported != n_objects:
            mismatched.append(f"{label} ({n_reported} reported, {n_objects} present)")
        expected = objects_to_show(n_objects, fraction)
        admissible = here[here["_montage_in_window"]]
        n_in_window = int(len(admissible))
        if expected == 0:
            note = (f"round({n_objects} x {fraction:.4g}) rounds to zero, so "
                    "the design expects no object from this well")
            selections.append(WellSelection(
                well=label, fraction=fraction, n_objects=n_objects,
                n_reported=n_reported, n_expected=0, n_in_window=n_in_window,
                n_selected=0, note=note))
            continue

        take = admissible.sort_values(
            ["montage_distance", "_montage_tiebreak"]).head(expected)
        n_taken = int(len(take))
        note = ""
        if n_taken < expected:
            note = (f"only {n_taken} of {expected} objects fall inside the "
                    "score window")
        selections.append(WellSelection(
            well=label, fraction=fraction, n_objects=n_objects,
            n_reported=n_reported, n_expected=expected,
            n_in_window=n_in_window, n_selected=n_taken, note=note))
        if n_taken:
            chosen.append(take)

    if chosen:
        picked = pd.concat(chosen, ignore_index=True)
    else:
        picked = work.iloc[0:0].copy()
    n_before_cap = int(len(picked))

    if n_before_cap > limit:
        keep = picked.sort_values(
            ["montage_distance", "_montage_tiebreak"]).head(limit)
        kept_per_well = keep["montage_well"].value_counts().to_dict()
        trimmed: List[WellSelection] = []
        for well in selections:
            kept = int(kept_per_well.get(well.well, 0))
            note = well.note
            if kept != well.n_selected:
                trim = f"trimmed by the montage cap from {well.n_selected} to {kept}"
                note = f"{note}; {trim}" if note else trim
            trimmed.append(WellSelection(
                well=well.well, fraction=well.fraction,
                n_objects=well.n_objects, n_reported=well.n_reported,
                n_expected=well.n_expected, n_in_window=well.n_in_window,
                n_selected=kept, note=note))
        selections = trimmed
        picked = keep

    picked = picked.sort_values(
        ["montage_well", "montage_distance", "_montage_tiebreak"])
    picked = picked.drop(columns=["_montage_in_window", "_montage_tiebreak"])
    picked = picked.reset_index(drop=True)
    picked["montage_rank"] = np.arange(1, len(picked) + 1)

    if mismatched:
        notes.append(
            "cell_count in the count data disagrees with the objects present "
            "for " + "; ".join(mismatched[:6]) +
            (" and others" if len(mismatched) > 6 else "") +
            ". The count rule used the objects actually present, because a "
            "montage cannot show an object it does not have.")
    if n_before_cap == 0:
        notes.append(
            "no object was selected: every well reporting this coefficient "
            "gave none, for the reason listed against it above")
    if window.degenerate:
        notes.append(
            "every object scores the same, so the score window admits all of "
            "them and 'closest' does not distinguish anything")

    choice = crop_source
    return MontagePlan(
        coefficient=coefficient, window=window, wells=tuple(selections),
        objects=picked, score_column=score_column,
        guide_aggregation=guide_aggregation, guides=tuple(covered),
        source_kind=choice.kind if choice else "",
        source_reason=choice.reason if choice else "",
        cap=limit, n_before_cap=n_before_cap, notes=tuple(notes))


def select_montage_per_guide(objects: pd.DataFrame, counts: pd.DataFrame,
                             name: str, effect: float,
                             **kwargs) -> List[MontagePlan]:
    """Return one :class:`MontagePlan` per guide of a gene-level coefficient.

    The other question. Summing a gene's guide fractions asks "which cells
    are consistent with losing this gene"; keeping them apart asks "does each
    guide pick out the same cells", which is how a real effect is told from
    one guide's off-target. Neither answer is a substitute for the other, so
    both are here and each plan says which it is.

    :param objects: every object available, as for :func:`select_montage`.
    :param counts: the per-well count data.
    :param name: the gene (or guide) name.
    :param effect: the fitted coefficient -- the SAME effect for every guide,
        because a gene-level coefficient is one number; the guides differ in
        which wells and which cells they select, not in what they imply.
    :param kwargs: forwarded to :func:`select_montage`.
    :returns: one plan per guide, in guide order. A guide whose wells report
        it nowhere is skipped, and the guides that produced a plan are on
        each plan's ``guides``.
    """
    kwargs.pop("guide_aggregation", None)
    kwargs.pop("guides", None)
    guide_column = kwargs.get("guide_column", "grna")
    gene_column = kwargs.get("gene_column", "gene")
    level = kwargs.get("level") or coefficient_level(
        counts, name, guide_column=guide_column, gene_column=gene_column)
    covered = guides_for_coefficient(
        counts, name, level=level, guide_column=guide_column,
        gene_column=gene_column)
    plans: List[MontagePlan] = []
    for guide in covered:
        try:
            plans.append(select_montage(
                objects, counts, name, effect, guide_aggregation="separate",
                guides=[guide], **kwargs))
        except CoefficientNotFound:
            continue
    return plans


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

#: What the two obvious-looking QC CSVs actually hold. Measured against
#: ``spacr.ml.grna_metricks`` and its contract test -- neither names a well
#: AND a guide AND a fraction, so neither can drive step 1 or step 3.
_NOT_THE_FRACTION_CSV: Dict[str, str] = {
    "grna_well.csv": (
        "holds grna, plateID, grna_well_count, gene_well_count -- HOW MANY "
        "wells a guide was seen in, never which wells and never a fraction"),
    "well_grna.csv": (
        "holds prc and gene_count -- how many distinct genes are in a well, "
        "and it does not name a guide at all"),
}

#: The QC CSV that does carry well x guide x fraction x cell count.
FRACTION_CSV = "regression_data.csv"


def read_well_guide_fractions(path: str) -> pd.DataFrame:
    """Read the per-well guide fractions a montage needs.

    :param path: a regression results folder, or the CSV itself. A folder is
        resolved to ``regression_data.csv``.
    :returns: the frame, validated to carry a well key, ``grna``, ``gene``
        and ``fraction``.
    :raises MontageError: the path names ``grna_well.csv`` or
        ``well_grna.csv`` -- refused by name, with what each actually holds,
        because they are the obvious guess and neither can answer the
        question; or the file is missing, or short of a needed column.
    """
    target = os.fspath(path)
    if os.path.isdir(target):
        target = os.path.join(target, FRACTION_CSV)
    base = os.path.basename(target)
    if base in _NOT_THE_FRACTION_CSV:
        raise MontageError(
            f"{base} {_NOT_THE_FRACTION_CSV[base]}. The montage needs the "
            f"well, the guide and the guide's fraction in that well, which "
            f"is {FRACTION_CSV} -- written to the same results folder.")
    if not os.path.isfile(target):
        raise MontageError(
            f"{target} does not exist. A montage needs {FRACTION_CSV} from "
            "the regression results folder.")
    frame = pd.read_csv(target)
    _well_key(frame, target)
    _require(frame, ["grna", "gene", "fraction"], target)
    return frame


def load_montage_objects(db_path: str, *, object_type: str = "cell",
                         score_column: str = "pred",
                         table: str = "png_list",
                         verbose: bool = False) -> pd.DataFrame:
    """Return the per-object rows a montage selects from, out of one database.

    Reads ``png_list`` -- which is where
    :func:`spacr.predictions.merge_cv_predictions` writes the per-object
    classification score -- and hands it to
    :func:`spacr.io.crop_rows_from_png_list`, which is the join that recovers
    ``path_name`` and the integer ``object_label`` a merged crop is cut by.
    So the frame that comes back serves **both** crop sources: ``png_path``
    for :class:`~spacr.crops.PngCropSource`, ``path_name`` plus
    ``object_label`` for :class:`~spacr.crops.MergedCropSource`.

    :param db_path: the ``measurements.db``.
    :param object_type: which crop mode's rows to read.
    :param score_column: the per-object classification score column.
    :param table: the table holding the crops; ``'png_list'``.
    :param verbose: let the io join report the rows it could not place.
    :returns: the object frame, with ``prc`` composed when the well keys are
        there.
    :raises MontageError: the database is missing or has no such table.
    :raises MissingScores: the table has no score column, or no finite score
        in it -- i.e. classification has not been merged into this database.
    """
    if not os.path.isfile(db_path):
        raise MontageError(f"measurements database not found: {db_path}")
    from .database_concurrency import connect as _connect

    conn = _connect(db_path)
    try:
        frame = pd.read_sql(f'SELECT * FROM "{table}"', conn)
    except Exception as exc:
        raise MontageError(
            f"{db_path} has no readable {table!r} table ({exc}). Without one "
            "there are no per-object crops to show.") from exc
    finally:
        conn.close()
    if score_column not in frame.columns:
        raise MissingScores(
            f"{table!r} in {db_path} has no {score_column!r} column, so no "
            "object carries a classification score. Run Classify and merge "
            "its predictions into the database first.")
    _finite_scores(frame, score_column)

    from .io import crop_rows_from_png_list

    joined = crop_rows_from_png_list(db_path, frame, object_type=object_type,
                                     verbose=verbose)
    if joined.empty:
        # The join keeps only rows that can be cut from merged/. A PNG folder
        # alone is still a montage, so fall back rather than returning none.
        joined = frame.copy()
        joined["object_type"] = object_type
    if "prc" not in joined.columns and all(
            c in joined.columns for c in WELL_KEY_COLUMNS):
        from . import schema

        joined["prc"] = [
            schema.compose_prc(p, r, c) for p, r, c in zip(
                joined["plateID"], joined["rowID"], joined["columnID"])]
    return joined


# ---------------------------------------------------------------------------
# The crop source
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CropSourceChoice:
    """Which crop source will draw a montage, or why none can.

    The tab needs both answers in the same shape: instruction 131 says a tab
    that cannot be filled must say why rather than be absent, so "there is no
    source" is a value here and not an exception.

    :param source: the :class:`spacr.crops.CropSource`, or ``None``.
    :param kind: ``'png'``, ``'merged'``, or ``''`` when unavailable.
    :param reason: why that source was picked, or why none could be.
    :param available: whether a montage can be drawn at all.
    """

    source: Any
    kind: str
    reason: str
    available: bool

    def describe(self) -> str:
        """Return the one-line sentence for the tab's status line."""
        if not self.available:
            return f"no crop source: {self.reason}"
        return f"{self.kind} crop source ({self.reason})"


def resolve_montage_crop_source(src, *, object_type: str = "cell",
                                prefer: Optional[str] = None
                                ) -> CropSourceChoice:
    """Pick the crop source for a montage, and never raise for "none".

    A thin wrapper on :func:`spacr.crops.resolve_crop_source`, which is the
    module that already knows how to choose between the exported PNGs and
    ``merged/*.npy`` and already says which it chose. The wrapper exists for
    exactly one reason: a missing source has to arrive as a sentence the tab
    can display, not as an exception it has to catch to stay on screen.

    :param src: a settings mapping (with ``src``, optionally
        ``crop_source``) or the experiment root / its ``merged`` folder.
    :param object_type: which object the crops are cut by.
    :param prefer: force ``'png'`` or ``'merged'``.
    :returns: a :class:`CropSourceChoice`; ``available`` is False, with the
        reason, when neither source exists.
    """
    from .crops import CropError, resolve_crop_source

    try:
        source = resolve_crop_source(src, object_type=object_type,
                                     prefer=prefer)
    except CropError as exc:
        return CropSourceChoice(source=None, kind="", reason=str(exc),
                                available=False)
    return CropSourceChoice(source=source, kind=source.kind,
                            reason=source.reason, available=True)
