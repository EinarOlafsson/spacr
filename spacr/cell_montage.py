"""Select cells to show for a regression coefficient.

Given a coefficient (a gene or a guide), the
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

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import portable_paths


LOG = logging.getLogger(__name__)

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
    "CROP_SHAPES",
    "DEFAULT_SCORE_COLUMN",
    "RouteRequirements",
    "montage_route_requirements",
    "round_half_up",
    "objects_to_show",
    "coefficient_level",
    "guides_for_coefficient",
    "wells_for_coefficient",
    "score_window",
    "select_montage",
    "select_montage_per_guide",
    "read_well_guide_fractions",
    "fractions_from_counts",
    "load_montage_objects",
    "resolve_montage_crop_source",
    "effects_grid_from_results",
    "write_effects_grid",
    "EFFECTS_GRID_FILE",
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
#: 2,000 is therefore ~23 s of cutting in the worst measured case and ~1.4 s
#: in the common one. Whatever the cap trims is named in the caption rather
#: than dropped quietly.
#:
#: WHY 2,000 AND NOT 10,000. 10,000 was asked for and the cost of it was
#: measured before it was taken: twelve open well tabs would hold 16.8 GB of
#: thumbnails, and at 24 to 60 objects a page the last page sits 167 to 417
#: clicks away behind a pager that only steps. 2,000 is ~3.4 GB across twelve
#: tabs and ~33 to ~83 pages -- most of the reach, none of the cliff.
MAX_OBJECTS = 2000

#: The per-object classification score a montage selects on, by default.
#: Named because a screen with more than one classifier output has more than
#: one candidate, and the caption has to say which one produced the picture.
DEFAULT_SCORE_COLUMN = "pred"

#: The two shapes a crop can take.
#:
#: ``'object'`` follows the object's own mask -- the better picture, and the
#: one worth defaulting to. ``'bbox'`` is its padded bounding box. THE TWO
#: ROUTES TO PIXELS DO NOT BOTH OFFER BOTH: a crop cut live from ``merged/``
#: has the mask plane and can do either, while a route that has only a
#: coordinate table has no mask and can do bounding boxes ONLY. That is why
#: :func:`montage_route_requirements` exists -- an object-shaped crop must not
#: appear as a choice that silently does something else.
CROP_SHAPES: Tuple[str, ...] = ("object", "bbox")

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




#: How the montage decides which cells belong to a coefficient.
#:
#: ``rank``       selects the highest-scoring cells using the read fractions;
#:                this is the default and requires no attribution model.
#: ``attributed`` each cell's posterior probability of carrying the guide,
#:                with the well's read fractions as the prior and the fitted
#:                effects as the likelihood. Shows the cells whose
#:                probability clears the threshold.
#: ``assigned``   the constrained assignment -- every cell in the well gets
#:                exactly one guide and each guide gets exactly the number of
#:                cells its reads imply. Shows the cells assigned to this one.
#: ``sudoku``     label propagation across WELLS. The others decide a well
#:                from its own cells; this one learns what a guide's cells
#:                look like from every well the guide is in, and applies it
#:                here. `spacr.sudoku`.
#: What an object shown but NOT picked is called (207 B).
#:
#: THE SAME SPELLING `spacr.sudoku` ABSTAINS WITH, so the two annotation
#: routes put the same word in the same column and a consumer does not have
#: to know which produced the frame.
NOT_ANNOTATED = "Non_annotated"

#: The column carrying that name.
ANNOTATION_COLUMN = "montage_annotation"


PICKING_MODES: Tuple[str, ...] = ("rank", "attributed", "assigned",
                                 "multivariate", "sudoku")

#: What each mode does, in one line, for anyone reading the code rather
#: than the panel.
#:
#: SUDOKU IS DOCUMENTED HERE AND NOT IN THE `cell_picking` TOOLTIP, which
#: names the other four. Adding a fifth sentence to that tooltip changed
#: its text, which invalidated nine hand-built translations of it -- and
#: the Chinese one could not be rebuilt: every rewording still failed the
#: target-script gate, so the panel would have shown English there. The
#: dropdown names all five (see `picture_settings.offered_values`) and each
#: entry carries its own hover help (`PICKING_HELP`), so nothing is
#: unreachable; only the summary tooltip is one short.
PICKING_NOTES = {
    "rank": "the highest scoring cells, as many as the read fraction implies",
    "attributed": "each cell's posterior probability of carrying the guide, "
                  "above a threshold",
    "assigned": "one guide per cell, so each guide gets exactly the cells "
                "its reads imply",
    "multivariate": "the same from every measurement rather than the score "
                    "alone; needs the gene x measurement sweep",
    "sudoku": "learns a guide's cells from every well the guide is in, and "
              "abstains rather than guessing when it cannot tell",
}


#: What a sweep writes its per-measurement effects to, inside the run folder.
#:
#: ONE NAME, TWO MODULES. `gene_measurement_sweep` writes it and this reads
#: it, and the montage finds it the same way it finds the results table --
#: off `results_path`. A grid passed in memory between two panels would work
#: exactly once, in the session that ran the sweep.
EFFECTS_GRID_FILE = "gene_measurement_effects.csv"


def effects_grid_from_results(path) -> Optional["pd.DataFrame"]:
    """Load the gene-by-measurement effects grid stored beside a run.

    Parameters
    ----------
    path : path-like
        Results CSV or directory containing :data:`EFFECTS_GRID_FILE`.

    Returns
    -------
    pandas.DataFrame or None
        Effects indexed by guide, or ``None`` when the grid is absent, empty,
        or unreadable.

    Notes
    -----
    The persisted grid lets multivariate montage selection work across
    application sessions. Callers may fall back to single-score attribution
    when no valid grid is available.
    """
    import os

    text = str(path or "")
    if not text:
        return None
    folder = text if os.path.isdir(text) else os.path.dirname(text)
    target = os.path.join(folder, EFFECTS_GRID_FILE)
    if not os.path.isfile(target):
        return None
    try:
        grid = pd.read_csv(target, index_col=0)
    except Exception:                                            # noqa: BLE001
        # Silent, like `effects_from_results` beside it: an unreadable grid
        # means "no sweep to use", and the caller's own message says what
        # that costs. Raising here would take the montage down over a file
        # it can do without.
        return None
    if not len(grid) or not len(grid.columns):
        return None
    grid.index = [_guide_of_term(str(i)) or str(i) for i in grid.index]
    return grid


def write_effects_grid(effects, folder) -> str:
    """Write a sweep's effects grid beside the run. Returns the path.

    Written by the sweep so the montage can read it later -- and in another
    session, which is the half a panel-to-panel handover cannot do.
    """
    import os

    if effects is None or not len(effects) or not str(folder or ""):
        return ""
    target = os.path.join(str(folder), EFFECTS_GRID_FILE)
    try:
        os.makedirs(str(folder), exist_ok=True)
        effects.to_csv(target)
    except Exception:                                            # noqa: BLE001
        # A sweep that produced its answer has not failed because the grid
        # could not be filed beside it.
        return ""
    return target


def effects_from_results(path) -> Dict[str, float]:
    """``{guide: effect}`` from a run's results table.

    The attribution needs every guide's effect in the well, not just the one
    being drawn -- a posterior is a comparison, and comparing a guide against
    nothing returns the prior. This reads them from the run the montage is
    already showing.
    """
    import os

    text = str(path or "")
    if not text or not os.path.isfile(text):
        return {}
    try:
        frame = pd.read_csv(text)
    except Exception:                                            # noqa: BLE001
        return {}
    name = next((c for c in ("feature", "guide", "grna", "coefficient", "name")
                 if c in frame.columns), None)
    value = next((c for c in ("coefficient", "effect", "estimate", "beta",
                              "standardized_marginal_effect")
                  if c in frame.columns and c != name), None)
    if name is None or value is None:
        return {}
    out: Dict[str, float] = {}
    for key, number in zip(frame[name], pd.to_numeric(frame[value],
                                                      errors="coerce")):
        text_key = _guide_of_term(str(key))
        if text_key and pd.notna(number):
            out.setdefault(text_key, float(number))
    return out


def _guide_of_term(term: str) -> str:
    """The guide inside a design term, or the term when it is already one.

    A results table names its rows as the DESIGN did -- `fraction:grna[g1]`,
    not `g1` -- so a lookup written for the bare name matches nothing at all,
    and "nothing matched" here is an attribution that silently falls back to
    the prior for every cell.
    """
    text = str(term or "").strip()
    if not text or text.lower() == "intercept":
        return ""
    if "[" in text and text.endswith("]"):
        inside = text[text.rindex("[") + 1:-1].strip()
        # statsmodels writes `C(rowID)[T.r2]` for a factor level.
        if inside.startswith("T."):
            inside = inside[2:]
        return inside
    return text


def _well_guide_fractions(counts, label, keys, guide_column, fraction_column):
    """``{guide: fraction}`` for one well, from the full count table.

    Every guide in the well, not only the one being drawn: a posterior is a
    comparison and there is nothing to compare a lone guide against.

    THE EXCLUSION IS NOT DONE HERE. `select_montage` drops excluded guides
    from the count table before anything reads it, so this function sees a
    table they have already left -- one exclusion point rather than one per
    caller, which is what keeps `well_totals` and these fractions agreeing
    about what the well contains.
    """
    try:
        frame = counts.copy()
        frame["_montage_well"] = _well_labels(frame, keys)
        here = frame[frame["_montage_well"].astype(str) == str(label)]
        if guide_column not in here.columns:
            return {}
        return {str(g): float(v) for g, v in
                zip(here[guide_column], here[fraction_column])
                if v is not None}
    except Exception:                                            # noqa: BLE001
        return {}

def normalised_share(well_fractions, fraction: float) -> Tuple[float, float]:
    """``(share, factor)`` -- the guide's fraction of what is left in the well.

    THE FACTOR IS NOT A NO-OP, and that is why this exists. In a representative
    screen, the raw count tables give every well a fraction sum
    of exactly 1.000000, but `fraction_threshold` defaults to 0.02 and the
    filtered table's sums fall to a median of 0.5526 with a minimum of
    0.1515. The filtered table is the ordinary case, so the un-normalised
    share understates the count by roughly half.

    :param well_fractions: every guide's fraction in this well.
    :param fraction: the chosen guide's fraction.
    :returns: the normalised share and the factor applied. A well whose
        fractions sum to zero -- or to nothing usable -- keeps its fraction
        and a factor of 1: there is nothing to normalise against, and
        inventing one would be arithmetic on no evidence.
    """
    try:
        total = float(sum(float(v) for v in well_fractions
                          if math.isfinite(float(v))))
    except (TypeError, ValueError):
        total = 0.0
    share = float(fraction)
    if not math.isfinite(share):
        return 0.0, 1.0
    if total <= 0.0 or not math.isfinite(total):
        return share, 1.0
    factor = 1.0 / total
    # Capped at 1: a share above 1 would mean this guide is more than all of
    # the well, which is a join that did not line up rather than a number.
    return min(share * factor, 1.0), factor


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
                 score_column: str = DEFAULT_SCORE_COLUMN,
                 half_widths: float = WINDOW_HALF_WIDTHS,
                 baseline: Optional[float] = None,
                 baseline_label: Optional[str] = None) -> ScoreWindow:
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
    :param baseline_label: what to record as the baseline's SOURCE when one
        is given, e.g. ``'the model\'s fitted intercept'``. The caption reads
        this back, so a montage centred on the intercept says so rather than
        saying ``given``, which names no source at all. Ignored when
        ``baseline`` is None -- the screen median has one source and it is
        already named.
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
        base_source = str(baseline_label) if baseline_label else "given"

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


def _read_table(path, **kwargs):
    """Read a table while deferring the canonical reader import.

    Deferring the import preserves the module's lightweight GUI import path.
    """
    from .tabular import read_table as _read

    return _read(path, **kwargs)


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
    #: The fraction that was actually multiplied to reach ``n_expected``.
    #:
    #: `fraction` is the guide's RAW share of the well; when the fractions
    #: are normalised, the count is reached with the normalised one, and a
    #: line printing the raw fraction beside the normalised count states an
    #: equation that is false -- "round(105 x 0.0205) = 27" was on screen,
    #: where the arithmetic gives 2. None means the two are the same.
    share: Optional[float] = None

    @property
    def contributed(self) -> bool:
        """True when this well put at least one object into the montage."""
        return self.n_selected > 0

    def describe(self) -> str:
        """Return the one-line account of this well's contribution.

        The fraction shown is the one the count was reached with, so the
        equation on screen is one the reader can check.
        """
        used = self.fraction if self.share is None else self.share
        line = (f"{self.well}: {self.n_selected} of "
                f"round({self.n_objects} x {used:.4g}) = "
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
    score_column: str = DEFAULT_SCORE_COLUMN
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

    #: How many per-well lines the arithmetic prints in full. A gene can span
    #: dozens of wells, and a caption that is 200 lines long is one nobody
    #: reads -- but the point of printing them is that a reader can add them
    #: up, so the elision says how many it hid rather than trailing off.
    WELL_LINES = 40

    def settings_line(self) -> str:
        """Every setting that decided WHICH CELLS this is, defaults included.

        The design: each of these changes which cells a reader is
        looking at, so each is written onto the montage and a non-default
        value says that it is one. THE SETTINGS ARE PER SCREEN, NEVER PER
        GENE -- a width chosen per gene is a width that can be tuned until
        the pictures look right, and nothing in the output would show that it
        had been. This sentence is what makes that visible either way.

        :returns: one or two lines -- what was in force, and (only when
            something was) what was changed from the default.
        """
        baseline = ("the screen median"
                    if self.window.baseline_source == "screen_median"
                    else self.window.baseline_source)
        guides = ("summed across the gene's guides"
                  if self.guide_aggregation == "sum" else "one at a time")
        lines = [
            "settings (per screen, not per gene): window half-width "
            f"{self.window.half_widths:g} robust scale(s); baseline "
            f"{baseline}; score column {self.score_column!r}; cap "
            f"{self.cap:,} objects; guides {guides}"]
        changed: List[str] = []
        if self.window.half_widths != WINDOW_HALF_WIDTHS:
            changed.append(
                f"window half-width {self.window.half_widths:g} instead of "
                f"the default {WINDOW_HALF_WIDTHS:g}")
        if self.window.baseline_source != "screen_median":
            changed.append(
                f"baseline taken from {baseline} instead of the default "
                "screen median")
        if self.score_column != DEFAULT_SCORE_COLUMN:
            changed.append(
                f"score column {self.score_column!r} instead of the default "
                f"{DEFAULT_SCORE_COLUMN!r}")
        if self.cap != MAX_OBJECTS:
            changed.append(
                f"cap {self.cap:,} objects instead of the default "
                f"{MAX_OBJECTS:,}")
        if self.guide_aggregation != "sum":
            changed.append(
                "guides kept apart instead of the default summed")
        if changed:
            lines.append(
                "NON-DEFAULT SETTINGS ON THIS MONTAGE: " + "; ".join(changed)
                + ". Each of these changes which cells are shown, so this "
                  "montage is not comparable to one made with the defaults.")
        return "\n".join(lines)

    def arithmetic(self) -> str:
        """The whole sum, in words a reader can check.

        Everything a reader would need to reproduce the
        selection by hand: where the baseline came from, what the target is
        and why, how wide the window is and in what units, and
        ``round(n x fraction)`` for each well with the total they add to.
        """
        window = self.window
        effect = self.coefficient.effect
        lines = ["how these cells were chosen -- the arithmetic:"]
        if window.baseline_source == "screen_median":
            lines.append(
                f"  baseline = median per-object {self.score_column} over "
                f"EVERY object supplied ({window.n_scored:,} objects, the "
                f"whole screen and not this coefficient's wells) = "
                f"{window.baseline:.6g}")
        else:
            lines.append(
                f"  baseline = {window.baseline_source} = "
                f"{window.baseline:.6g} (not the screen median; the median of "
                f"the {window.n_scored:,} supplied objects was not used)")
        lines.append(
            f"  target   = baseline + effect = {window.baseline:.6g} + "
            f"{effect:.6g} = {window.target:.6g}   <- {SCORE_TARGET_RULE}")
        if window.degenerate:
            lines.append(
                "  scale    = 1.4826 x MAD of those same scores = 0, and the "
                "standard deviation is 0 too: every object scores the same")
            lines.append(
                "  window   = degenerate, so it admits every object and "
                "'closest' does not distinguish anything")
        else:
            lines.append(
                f"  scale    = 1.4826 x MAD of those same scores = "
                f"{window.scale:.6g}")
            lines.append(
                f"  window   = target +/- {window.half_widths:g} x scale = "
                f"[{window.low:.6g}, {window.high:.6g}]")
        lines.append(
            "  kept     = the objects inside that window, closest first "
            f"(smallest |{self.score_column} - target|)")
        lines.append(
            "per-well count = round(objects in well x guide fraction in "
            "well), halves away from zero:")
        shown = self.wells[:self.WELL_LINES]
        for well in shown:
            line = (f"  {well.well}: round({well.n_objects} x "
                    f"{well.fraction:.4g}) = {well.n_expected} -> "
                    f"{well.n_selected} shown")
            if well.note:
                line += f" ({well.note})"
            lines.append(line)
        if len(self.wells) > len(shown):
            hidden = self.wells[len(shown):]
            lines.append(
                f"  ... and {len(hidden)} more wells contributing "
                f"{sum(w.n_selected for w in hidden)} objects between them, "
                "by the same rule")
        total = sum(w.n_selected for w in self.wells)
        if len(self.wells) <= 12:
            summed = " + ".join(str(w.n_selected) for w in self.wells) or "0"
            lines.append(f"  total: {summed} = {total} objects shown")
        else:
            lines.append(
                f"  total: {total} objects shown across "
                f"{len(self.wells)} wells")
        if total != self.n_objects:
            lines.append(
                f"  (the grid holds {self.n_objects}; the difference is the "
                "cap, named above)")
        # THE GUIDES SHARE CELLS, AND A READER SHOULD NOT FIND THAT OUT BY
        # ARITHMETIC (172, last open item). Every guide in a well is given
        # `round(n x share)` of the SAME top-ranked cells, so the counts of a
        # well's guides can add to more than the well holds -- measured on
        # the maintainer's four plates, 190 of 1,366 wells. It is correct for
        # this heuristic and it is exactly the flaw instruction 173 exists to
        # fix, but unsaid it reads as a bug in the count.
        crowded = sum(w.n_selected for w in self.wells
                      if w.n_selected > 0 and w.n_objects
                      and w.n_selected > w.n_objects)
        lines.append(
            "  note: each guide in a well is counted against that well's own "
            "cells, and the guides are not given DIFFERENT cells -- so the "
            "counts of a well's guides can add to more than the well holds. "
            "That is this heuristic, not a miscount; the per-cell "
            "attribution is what separates them."
            + (f" ({crowded} object(s) here come from a well already at its "
               f"own count.)" if crowded else ""))
        return "\n".join(lines)

    def caption(self) -> str:
        """Return the caption the montage must carry.

        States the wells, the score window, EVERY SETTING THAT DECIDED WHICH
        CELLS THESE ARE (:meth:`settings_line`), THE WHOLE ARITHMETIC A
        READER CAN CHECK THE SUM FROM (:meth:`arithmetic`), which crop source
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
        lines.append(self.settings_line())
        lines.append(self.arithmetic())
        if not self.window.target_is_observable:
            lines.append(
                f"the implied score {self.window.target:.4g} lies OUTSIDE the "
                f"observed range [{self.window.observed_low:.4g}, "
                f"{self.window.observed_high:.4g}]: no object in this screen "
                "scores anything like it")
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


#: How large a guide must be in a well before that well is worth pulling in
#: to anchor it. Matches `sudoku.anchors_for`'s own `min_fraction` default:
#: in a well where a guide is half the reads, a top-scoring cell is probably
#: that guide, and below that an anchor set teaches the graph the wrong
#: shape.
_SUDOKU_ANCHOR_SHARE = 0.5


def _sudoku_calls(work, counts, keys, guide_column, fraction_column,
                  score_column, name, notes, guides=()):
    """Assign one guide name per cell across the available wells.

    :returns: ``{well label: [guide per cell, in that well's row order]}``,
        or ``None`` when the screen cannot support it.

    Returned assignments preserve each well's input row order because callers
    align them positionally with their per-well frames.

    :param guides: Guide identifiers represented by the coefficient. For a
        gene-level montage, ``name`` is a gene while fractions and Sudoku
        assignments are guide keyed; this parameter supplies the corresponding
        guide identifiers for filtering and highlighting.
    """
    try:
        from .sudoku import sudoku as _sudoku

        frame = work.copy()
        frame["_montage_well"] = _well_labels(frame, keys)
        scores = pd.to_numeric(frame[score_column], errors="coerce")
        frame = frame[scores.notna()]
        if not len(frame):
            notes.append("sudoku: no cell carries a classification score")
            return None

        # THE FEATURES ARE THE MEASUREMENTS, NOT THE SCORE. The anchors are
        # chosen BY the score, so a graph built on it would place every
        # high-scoring cell beside every guide's anchors and affirm all of
        # them. `spacr.sudoku` leaves the score out for the same reason.
        numeric = frame.select_dtypes(include=[np.number])
        features = numeric.drop(
            columns=[c for c in numeric.columns
                     if str(c) == str(score_column)
                     or str(c).startswith("_montage")],
            errors="ignore")
        if features.shape[1] == 0:
            notes.append("sudoku: the objects carry no numeric measurement "
                         "to build a graph on")
            return None

        wells = [str(w) for w in frame["_montage_well"]]
        fractions = {}
        for label in sorted(set(wells)):
            here = _well_guide_fractions(counts, label, keys, guide_column,
                                         fraction_column)
            if here:
                fractions[label] = here
        if not fractions:
            notes.append("sudoku: no well has a guide fraction to constrain "
                         "the propagation")
            return None

        # SCOPE: THIS GUIDE'S WELLS, PLUS THE WELLS THAT ANCHOR ITS RIVALS.
        #
        # THE FIRST VERSION OF THIS TRIMMED TO THIS GUIDE'S WELLS ALONE AND
        # THAT WAS WRONG, on an argument that conflated two different parts
        # of the method. The WELL CONSTRAINT is per well, so a guide absent
        # from a well genuinely cannot claim cells there -- that much held.
        # The GRAPH is not per well at all: it links cells by how they look,
        # across the screen, and that is where a guide's appearance is
        # learned.
        #
        # So trimming the cells also trimmed the ANCHORS. `anchors_for`
        # takes a guide's examples from wells where that guide DOMINATES,
        # and a rival's best wells are usually not this guide's wells -- so
        # the rivals were being characterised from whatever share they
        # happened to have here, which is the weakest sample available
        # rather than the strongest. Raised by the maintainer: "other guides
        # in the chosen guides wells do [share wells], so that information
        # can be used, no?" -- yes, and it was being thrown away.
        #
        # The scope is therefore the union of this guide's wells and, for
        # every guide that appears in them, the wells where that guide is
        # large enough to anchor. Those extra wells inform the graph and the
        # anchors; only this guide's wells are drawn.
        # THE COEFFICIENT'S GUIDES, not its name. A gene is never a key in
        # `here`, which is guide -> fraction.
        wanted = {str(g) for g in (guides or ())} or {str(name)}
        mine = [label for label, here in fractions.items()
                if any(g in here for g in wanted)]
        if not mine:
            notes.append(
                f"sudoku: {name} is in none of these wells "
                f"(looked for {', '.join(sorted(wanted))})")
            return None
        rivals = {g for label in mine for g in fractions[label]}
        anchoring = {
            label for label, here in fractions.items()
            if any(float(here.get(g, 0.0)) >= _SUDOKU_ANCHOR_SHARE
                   for g in rivals)
        }
        keep = set(mine) | anchoring
        fractions = {label: here for label, here in fractions.items()
                     if label in keep}
        notes.append(
            f"sudoku: {len(mine)} well(s) hold {name}; "
            f"{len(keep) - len(mine)} more well(s) joined to anchor the "
            f"{len(rivals)} guide(s) it is compared against")
        rows = [i for i, w in enumerate(wells) if w in keep]
        if not rows:
            notes.append("sudoku: no cell sits in a well holding this guide")
            return None
        frame = frame.iloc[rows]
        features = features.iloc[rows]
        wells = [wells[i] for i in rows]

        guides = sorted({g for here in fractions.values() for g in here})
        result = _sudoku(
            features.to_numpy(dtype=float),
            pd.to_numeric(frame[score_column],
                          errors="coerce").to_numpy(dtype=float),
            wells, fractions, guides)
        notes.append(
            f"sudoku: {result.called():,} of {len(frame):,} cell(s) "
            f"annotated across {len(fractions)} well(s); "
            f"{result.report.get('abstained', 0):,} abstained")
        # KEYED BY THE FRAME'S INDEX, NOT BY POSITION.
        #
        # THIS WAS THE BUG, and it was silent: the caller matches these
        # against `ranked`, which is the well's rows SORTED BY SCORE, while
        # this list was in the order the rows arrived. Same length, different
        # cells -- so the calls landed on the wrong objects, and where the
        # lengths disagreed nothing was marked at all. A real run reported
        # "59 of 1,076 cells annotated" and then highlighted zero in every
        # well.
        #
        # An index cannot be misaligned by a sort.
        return dict(zip(frame.index, result.guides))
    except Exception as exc:                                 # noqa: BLE001
        # A PICKER THAT CANNOT RUN SAYS SO AND THE MONTAGE STILL DRAWS.
        notes.append(f"sudoku could not run ({type(exc).__name__}: {exc}); "
                     f"the montage fell back to rank")
        return None


def select_montage(objects: pd.DataFrame, counts: pd.DataFrame,
                   name: str, effect: float, *,
                   level: Optional[str] = None,
                   score_column: str = DEFAULT_SCORE_COLUMN,
                   half_widths: float = WINDOW_HALF_WIDTHS,
                   baseline: Optional[float] = None,
                   baseline_label: Optional[str] = None,
                   cap: int = MAX_OBJECTS,
                   guide_aggregation: str = "sum",
                   guide_column: str = "grna",
                   gene_column: str = "gene",
                   fraction_column: str = "fraction",
                   crop_source: Optional["CropSourceChoice"] = None,
                   guides: Optional[Sequence[str]] = None,
                   show_all: bool = False,
                   picking: str = "rank",
                   effects: Optional[Mapping[str, float]] = None,
                   effects_grid: Optional["pd.DataFrame"] = None,
                   exclude_grnas: Optional[Sequence[str]] = None,
                   normalise_fraction: bool = True,
                   threshold: float = 0.55) -> MontagePlan:
    """Return the objects to show behind one coefficient.

    The whole selection, in the order the design states it: the wells
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
    :param baseline_label: what to record as that baseline's source, so the
        caption names it -- ``'the fitted intercept'`` rather than ``given``.
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
                          baseline=baseline, baseline_label=baseline_label)

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
    # THE EXCLUSION HAPPENS FIRST, ONCE, AND ON THE COUNT TABLE ITSELF.
    # Asked for 2026-08-21: "make sure it is removed first. right?" -- and
    # right, for a reason that a later per-call filter would not have fixed.
    # `well_totals` below sums the fraction column over the FULL table, and
    # `normalised_share` divides by that sum. A contaminant removed further
    # downstream would still be sitting in that denominator, holding every
    # real guide's share down by its own -- which on one real plate was a
    # fifth of all the reads.
    #
    # So there is exactly one exclusion point and it is above every path:
    # the ranking, the per-well fractions, the posteriors and the totals all
    # read a table the contaminant has already left.
    #
    # GUIDES AND GENES, several of each, resolved by `control_names` -- the
    # same resolver `controls` and `positive_control` use, so an exclusion
    # is typed in the spelling those already accept.
    excluded_note = ""
    if exclude_grnas:
        try:
            from .read_background import (resolve_exclusions,
                                          unmatched_exclusions)

            names = counts[guide_column].astype(str) \
                if guide_column in counts.columns else pd.Series([], dtype=str)
            genes = counts[gene_column].astype(str) \
                if gene_column in counts.columns else None
            drop = resolve_exclusions(exclude_grnas, names, genes)
            if drop:
                counts = counts[~names.isin(drop)].copy()
                excluded_note = (f"excluded {len(drop)} guide(s) named by "
                                 f"exclude_grnas before any fraction was "
                                 f"formed")
            # A MISSPELLED EXCLUSION EXCLUDES NOTHING AND LOOKS LIKE IT
            # WORKED, which is how a known contaminant survives the filter
            # that was meant to remove it.
            missing = unmatched_exclusions(exclude_grnas, names, genes)
            if missing:
                excluded_note = ((excluded_note + " · ") if excluded_note
                                 else "") + (
                    f"exclude_grnas named {len(missing)} thing(s) that match "
                    f"nothing here: {', '.join(map(str, missing[:5]))}")
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not apply exclude_grnas", exc_info=True)

    work["_montage_in_window"] = window.contains(scores)
    work["_montage_tiebreak"] = _object_sort_key(work)

    # Always key on a tuple. ``groupby(['prc'])`` yields a bare string on
    # pandas 1.x and a 1-tuple on 2.2+, so a lookup written for either one
    # silently matches nothing on the other -- and "nothing matched" here is
    # an empty montage with a caption that still reads as if it worked.
    grouped: Dict[Tuple[Any, ...], pd.DataFrame] = {}
    for label, frame in work.groupby(keys, dropna=False):
        grouped[label if isinstance(label, tuple) else (label,)] = frame
    # EVERY GUIDE'S FRACTION IN EACH WELL, which is what the normalisation
    # divides by (instruction 172). It comes from the FULL count table and
    # not from `selected_counts`: the latter holds only the chosen
    # coefficient, so summing it would always give that guide's own fraction
    # and a factor of exactly 1 -- a normalisation that never normalised.
    well_totals: Dict[str, float] = {}
    try:
        totals = counts.copy()
        totals["_montage_well"] = _well_labels(totals, keys)
        well_totals = {
            str(k): float(v) for k, v in
            totals.groupby("_montage_well")[fraction_column].sum().items()}
    except Exception:                                        # noqa: BLE001
        well_totals = {}

    notes: List[str] = []
    if excluded_note:
        notes.append(excluded_note)
    selections: List[WellSelection] = []
    chosen: List[pd.DataFrame] = []
    mismatched: List[str] = []

    # CAN THIS GUIDE BE ATTRIBUTED AT ALL -- asked BEFORE any cell is
    # attributed, which is instruction 173's own wording. Until now this was
    # a library function with no caller, so the answer existed and nobody
    # saw it.
    #
    # IT IS THE DIFFERENCE BETWEEN AN EMPTY MONTAGE AND AN EXPLAINED ONE. A
    # guide whose effect is too small against the spread of scores can never
    # reach the threshold in any well, so the attributed picker selects
    # nothing and the montage comes back blank -- which reads as a bug in the
    # viewer rather than as arithmetic about the guide.
    #
    # The scale and centre are taken over ALL the objects, not per well:
    # centring per well destroys the between-well signal that identifies the
    # effect at all.
    if str(picking or "rank") in ("attributed", "assigned", "multivariate") \
            and effects:
        try:
            from .guide_attribution import preflight

            fractions_by_well = {}
            for _, row in well_frame.iterrows():
                label = "_".join(str(row[k]) for k in keys)
                here = _well_guide_fractions(
                    counts, label, keys, guide_column, fraction_column)
                if here:
                    fractions_by_well[label] = here
            finite = scores[np.isfinite(scores)]
            if fractions_by_well and finite.size:
                verdict = preflight(
                    name, fractions_by_well, effects,
                    scale=float(np.std(finite)) or 1.0,
                    centre=float(np.median(finite)),
                    threshold=float(threshold))
                notes.append(verdict.note())
        except Exception:                                    # noqa: BLE001
            # A pre-flight is a courtesy, not a precondition. It must never
            # be the reason a montage does not draw.
            pass

    # SUDOKU RUNS ONCE, OVER THE WHOLE SCREEN, BEFORE THE WELL LOOP.
    # Every other picker decides a well from that well's own cells, so it
    # can be computed inside the loop. Sudoku cannot: what a guide's cells
    # look like is learned from every well the guide is in, and running it
    # per well would throw away the one thing it is for.
    sudoku_calls = None
    if str(picking or "rank") == "sudoku":
        sudoku_calls = _sudoku_calls(
            work, counts, keys, guide_column, fraction_column,
            score_column, name, notes, guides=covered)

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

        # HOW MANY (instruction 172). The guide's share of what the count
        # table still holds for this well, times the number of cells that
        # actually carry a classification score -- not the number of rows.
        # An object with no score cannot be ranked, so counting it would
        # promise cells the ranking cannot deliver.
        total_here = float(well_totals.get(label, 0.0))
        if normalise_fraction:
            share, factor = normalised_share(
                [total_here] if total_here else [], fraction)
        else:
            # THE RAW FRACTION, WHICH IS THE CONSERVATIVE ONE (207 D).
            # Normalising divides by what SURVIVED the threshold, so the
            # reads of every filtered-out guide are redistributed onto the
            # ones that remain -- measured on a real screen, the filtered
            # sums fall to a median of 0.5526, so each survivor is inflated
            # by about 1.8x. A guide with few reads in a well can come out
            # of that with a high share, and the ranking then takes cells on
            # the strength of a number normalisation created.
            #
            # Raw keeps the discarded reads in the denominator, where they
            # dilute a marginal guide instead of being handed to it. Neither
            # is right for every screen, which is why it is a choice.
            share, factor = float(fraction), 1.0
        here_scores = pd.to_numeric(here[score_column], errors="coerce")
        n_classified = int(here_scores.notna().sum())
        expected = objects_to_show(n_classified, share)
        admissible = here[here["_montage_in_window"]]
        n_in_window = int(len(admissible))
        if expected == 0:
            # THE SAME FOUR NUMBERS AS THE NON-ZERO CASE. This said
            # round(n_objects x fraction), and the count is
            # round(n_classified x share) -- a different count off a different
            # base. It named the UN-NORMALISED fraction, which is the one
            # number instruction 172 exists to stop anyone reading: on the
            # maintainer's four plates the factor runs to 6.6x, so a reader
            # checking this line would work out a share up to six times too
            # small and conclude the montage had dropped their well. It
            # covers 153 of 5,615 guide-well pairs that hold cells.
            note = (f"round({share:.4g} x {n_classified}) rounds to zero, so "
                    f"the design expects no object from this well "
                    f"({share:.4g} is this guide's {fraction:.4g} normalised "
                    f"by {factor:.4g}, the well's fractions summing to "
                    f"{total_here:.4g}; {n_classified} of the well's "
                    f"{n_objects} objects carry a score)")
            selections.append(WellSelection(
                well=label, fraction=fraction, n_objects=n_objects,
                n_reported=n_reported, n_expected=0, n_in_window=n_in_window,
                n_selected=0, note=note))
            continue

        # WHICH ONES (instruction 172). "rank all cells by classefication
        # score and take the top x cells".
        #
        # THE DIRECTION FOLLOWS THE COEFFICIENT. Highest scores for a positive
        # effect, lowest for a negative one, because the cells a coefficient
        # points at are the ones whose phenotype moved the way it says.
        # Always-descending would show a negative coefficient the cells LEAST
        # consistent with it.
        ranked = here.assign(_montage_score=here_scores)
        ranked = ranked.dropna(subset=["_montage_score"]).sort_values(
            ["_montage_score", "_montage_tiebreak"],
            ascending=[coefficient.effect < 0, True])
        top = ranked.head(expected)

        # THE OTHER TWO PICKERS (instruction 173). Both need every guide's
        # fraction AND effect in this well, because a posterior is a
        # comparison: comparing a guide against nothing returns the prior.
        picked_by = "rank"
        wanted = str(picking or "rank")
        if wanted == "multivariate" and effects_grid is None:
            # SAID, NOT SUBSTITUTED SILENTLY. Option C needs one effect per
            # MEASUREMENT per guide, which is the gene x measurement sweep's
            # grid; a run that has not swept has nothing to read. Falling
            # back to the single-score attribution is the right answer, and
            # a montage that quietly changed how it chose its cells is not.
            wanted = "attributed"
            picked_by = "attributed (no sweep grid for multivariate)"
        if wanted == "multivariate" and effects_grid is not None:
            here_fractions = _well_guide_fractions(
                counts, label, keys, guide_column, fraction_column)
            columns = [c for c in effects_grid.columns if c in ranked.columns]
            if len(here_fractions) > 1 and columns:
                from .guide_attribution import (normalise_fractions,
                                                posterior_multivariate)

                priors = normalise_fractions(here_fractions)
                grid = {g: effects_grid.loc[g, columns].to_numpy(dtype=float)
                        for g in priors if g in effects_grid.index}
                values = ranked[columns].to_numpy(dtype=float)
                r, order, report = posterior_multivariate(values, priors, grid)
                if name in order:
                    mine = r[:, order.index(name)]
                    top = ranked[mine >= float(threshold)]
                    picked_by = (
                        f"multivariate over {len(columns)} measurement(s), "
                        f"worth {report['effective_dimension']:.1f} "
                        f"independent one(s)")
            else:
                wanted = "attributed"
                picked_by = ("attributed (this well has one guide, or none of "
                             "the swept measurements are on the objects)")
        if wanted == "sudoku":
            # ACROSS WELLS, WHICH IS THE WHOLE POINT (209), and also the
            # reason it cannot be computed inside this per-well loop the way
            # the others are: a guide's appearance is learned from every
            # well it is in. `_sudoku_calls` runs once for the whole screen
            # before the loop and this reads its answer for these rows.
            #
            # BY INDEX. `ranked` is sorted by score, so a positional lookup
            # would put each cell's call on a different cell.
            if sudoku_calls:
                # AGAINST THE COEFFICIENT'S GUIDES. `sudoku` calls a cell
                # for a GUIDE; a gene-level montage is named for the gene,
                # and comparing the two matched nothing -- which is a
                # montage of unringed cells and no error anywhere.
                _wanted = {str(g) for g in (covered or ())} or {str(name)}
                mask = np.array([sudoku_calls.get(i) in _wanted
                                 for i in ranked.index], dtype=bool)
                top = ranked[mask]
                picked_by = "sudoku (propagated across wells)"
            else:
                picked_by = ("sudoku could not run on this screen; fell "
                             "back to rank")
        if wanted in ("attributed", "assigned") and effects:
            here_fractions = _well_guide_fractions(
                counts, label, keys, guide_column, fraction_column)
            if len(here_fractions) > 1:
                from .guide_attribution import (assign_well, attribute_well,
                                          normalise_fractions)

                spread = float(ranked["_montage_score"].std()) or 1.0
                middle = float(ranked["_montage_score"].median())
                values = ranked["_montage_score"].to_numpy()
                if wanted == "assigned":
                    # EVERY cell in the well gets exactly one guide and each
                    # guide gets exactly the cells its reads imply, so this
                    # picker's count is x by construction rather than by
                    # rounding.
                    outcome = assign_well(values, here_fractions, effects,
                                          centre=middle, scale=spread)
                    mask = np.array([g == name for g in outcome.guides])
                    top = ranked[mask]
                    picked_by = "assigned"
                else:
                    calls = attribute_well(values, here_fractions, effects,
                                           threshold=threshold, centre=middle,
                                           scale=spread)
                    mask = np.array([c.guide == name for c in calls])
                    top = ranked[mask]
                    picked_by = "attributed"
        direction = "lowest" if coefficient.effect < 0 else "highest"
        # THE NOTE MUST DESCRIBE THE PICKER THAT RAN. The fraction
        # arithmetic below is how `rank` decides; every other picker decides
        # some other way and does not consult it. Printing it regardless
        # reported a calculation that did not happen -- observed on a sudoku
        # montage that highlighted nothing and still said "round(0.1267 x
        # 187) = 24", which is a number the run never used.
        #
        # `expected` stays computed either way: it is the count the
        # SEQUENCING supports, which is worth stating next to what the
        # picker actually chose, because the gap between them is the
        # interesting part.
        _by_rank = str(picked_by).startswith("rank")
        if not _by_rank:
            chosen_here = int(len(top))
            arithmetic = (
                f"{picked_by} chose {chosen_here} of {n_classified} "
                f"classified cell(s); the fraction would support "
                f"{expected}")
            if chosen_here == 0:
                arithmetic += (" -- this picker annotated none of this "
                               "well's cells to this guide")
        elif normalise_fraction:
            arithmetic = (f"round({share:.4g} x {n_classified}) = {expected}, "
                          f"where {share:.4g} is this guide's {fraction:.4g} "
                          f"normalised by {factor:.4g} (the well's fractions "
                          f"sum to {total_here:.4g})")
        else:
            # SAY WHICH FRACTION IT USED. The two differ by the
            # normalisation factor, and a count that cannot be traced to a
            # fraction is a count nobody can check.
            arithmetic = (f"round({share:.4g} x {n_classified}) = {expected}, "
                          f"where {share:.4g} is this guide's RAW fraction "
                          f"(un-normalised; the well's kept fractions sum to "
                          f"{total_here:.4g})")
        if show_all:
            # EVERY CELL IN THE WELL, with the chosen ones marked rather than
            # the rest removed. "show all the images from each well and
            # highlight the cells most likely to be whatever gene is picked".
            take = ranked.copy()
            take["montage_candidate"] = take.index.isin(top.index)
            _n_marked = int(take["montage_candidate"].sum())
            # AND THE REST ARE NAMED, NOT MERELY UNMARKED (207 B). Asked for
            # 2026-08-21: "i an the non annotated datapoints to be annotated
            # as Non_annotated and shown".
            #
            # A CELL THAT IS SHOWN AND CARRIES NO LABEL IS COUNTED BY THE
            # EYE AND BY NOTHING ELSE. Giving it a name puts it in the
            # legend, in the group-by, and in the denominator -- which is
            # where it has to be, because a fraction computed over only the
            # annotated cells is the fraction that came out as 1.
            take[ANNOTATION_COLUMN] = np.where(
                take["montage_candidate"], str(name), NOT_ANNOTATED)
            n_taken = int(len(take))
            if _n_marked and _by_rank:
                marked = (f"the {_n_marked} with the {direction} scores are "
                          f"highlighted and the rest are {NOT_ANNOTATED}")
            elif _n_marked:
                # NOT "the N with the highest scores" -- a picker that is not
                # `rank` did not choose by score, and saying it did explains
                # the picture with the wrong rule.
                marked = (f"{_n_marked} are highlighted and the rest are "
                          f"{NOT_ANNOTATED}")
            else:
                marked = (f"none is highlighted; every cell here is "
                          f"{NOT_ANNOTATED}")
            note = (f"showing all {n_taken} classified cells in the well; "
                    f"{marked} -- {arithmetic}")
        else:
            take = top.copy()
            take["montage_candidate"] = True
            # The same column either way, so a consumer does not have to
            # know which view produced the frame.
            take[ANNOTATION_COLUMN] = str(name)
            n_taken = int(len(take))
            note = arithmetic
            if n_taken < expected:
                note = (f"only {n_taken} of {expected} cells in this well "
                        f"carry a classification score -- {arithmetic}")
        take = take.drop(columns=["_montage_score"], errors="ignore")
        selections.append(WellSelection(
            well=label, fraction=fraction, n_objects=n_objects,
            n_reported=n_reported, n_expected=expected,
            n_in_window=n_in_window, n_selected=n_taken, note=note,
            share=float(share) if _by_rank else None))
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
                n_selected=kept, note=note, share=well.share))
        selections = trimmed
        picked = keep

    picked = picked.sort_values(
        ["montage_well", "montage_distance", "_montage_tiebreak"])
    # WHICH OF THEM THE COEFFICIENT POINTS AT, kept as a column so the panel
    # can mark them. In the filtered view every row is a candidate by
    # construction; in the show-all view it is the distinction the whole
    # option exists for.
    if "montage_candidate" not in picked.columns:
        picked["montage_candidate"] = True
    picked["montage_candidate"] = picked["montage_candidate"].fillna(False).astype(bool)
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

    Keeping guides separate shows whether each guide selects comparable cells,
    complementing the gene-level montage built from summed guide fractions.

    :param objects: every object available, as for :func:`select_montage`.
    :param counts: the per-well count data.
    :param name: the gene (or guide) name.
    :param effect: fitted gene-level coefficient shared by every guide plan;
        guides differ only in the wells and cells they select.
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
    # THROUGH THE FUNNEL (145). `_well_key` composes prc from plateID,
    # rowID and columnID, and a results folder written by an older spaCR --
    # or by a plate whose png_list spells them row_name / column_name -- gave
    # it none of the three. It raised nothing; it produced a frame with no
    # well key, and the montage then drew nothing for wells holding 244
    # objects.
    frame = _read_table(target, report=None)
    _well_key(frame, target)
    _require(frame, ["grna", "gene", "fraction"], target)
    return frame


def fractions_from_counts(paths: Sequence[str]) -> pd.DataFrame:
    """Per-well guide fractions, built from the COUNT CSVs directly.

    WHY THIS EXISTS. :func:`read_well_guide_fractions` reads
    ``regression_data.csv`` out of a run folder, and the montage refused to
    draw anything without one -- so a user who had loaded their scores and
    their counts, run a regression, and was looking at its coefficients was
    told "the loaded coefficient table was not read from a run folder". The
    sentence was true and the requirement was not: nothing in the fraction
    frame comes from the run.

    A guide's fraction in a well is its share of that well's reads:

        fraction = count / (sum of count over the well)

    which is the same arithmetic :func:`spacr.ml.process_reads` does at
    ``ml.py`` before the fit -- so this is the SAME NUMBER, computed from the
    same input, not an approximation of it. ``regression_data.csv`` is that
    join persisted, not the source of it.

    :param paths: the count CSVs from the regression input table -- the
        ``count`` column of the rows the database provider already returns, so
        no new plumbing reaches them.
    :returns: one row per (well, guide) with ``prc``, ``grna`` and
        ``fraction``, and ``gene`` when the counts carry it.
    :raises MontageError: no readable counts, or a file short of a column.

    THE WELL TOTAL IS A ONE-ROW-PER-WELL FRAME BY CONSTRUCTION, and the merge
    says so with ``validate='many_to_one'``. `ml.process_reads` carries the
    same guard and the same comment: a duplicated well total would duplicate
    every guide row of that well, and the fractions would still sum to 1 per
    copy -- so the corruption would be invisible in every downstream check.
    """
    frames: List[pd.DataFrame] = []
    problems: List[str] = []
    # ENUMERATED, and the index is used even for the files that are skipped.
    # A count CSV names its plate in a column or not at all -- the real ones
    # carry `row_name, column_name, grna_name, count` and nothing else -- so
    # the plate comes from WHICH FILE it is, exactly as
    # `ml.load_regression_input_pairs` resolves it: own column, then pair-row
    # order. Letting an unreadable file collapse the numbering would shift
    # every later plate's label by one and silently mislabel the wells.
    for index, path in enumerate(paths):
        text = os.fspath(path)
        if not text or not os.path.isfile(text):
            continue
        try:
            # The count CSVs are the case 145 measured: `row_name`,
            # `column_name`, `grna_name` and NO plate column at all, so four
            # plates' r1/c1 pooled into one well -- 384 wells instead of
            # 1,536 -- and the fractions still summed to 1.
            frame = _read_table(text, report=None)
        except Exception as error:                               # noqa: BLE001
            problems.append(f"{os.path.basename(text)}: {error}")
            continue

        # CANONICALISE FIRST, the way every other reader in spaCR does.
        # Reported 2026-08-18: "the cell montage failed because the column grna
        # was not found in any of the count tables" -- and the tables had the
        # identifier under one of the spellings `correct_metadata_column_names`
        # exists to absorb (`grna_name`, and whatever `schema.canonicalise_frame`
        # maps). Reading the CSV raw made this function the ONE reader that did
        # not, which is exactly the "one vocabulary" failure instruction 145 is
        # about, introduced while fixing something else.
        try:
            from .schema import correct_metadata_column_names
            frame = correct_metadata_column_names(frame)
        except Exception:                                        # noqa: BLE001
            pass

        # AND THE ALIASES THE REST OF THE PROJECT ALREADY ACCEPTS. `utils`
        # looks for a gRNA identifier under seven spellings when reading
        # metadata; a count table is the same identifier in the same shape, so
        # refusing it here for its header would be this module inventing a
        # stricter rule than the code around it.
        if "grna" not in frame.columns:
            for alias in ("grna_name", "name", "sgrna", "sgRNA", "guide",
                          "sequence"):
                if alias in frame.columns:
                    frame = frame.rename(columns={alias: "grna"})
                    break
        if "count" not in frame.columns:
            for alias in ("read_count", "reads", "counts", "n", "count_sum"):
                if alias in frame.columns:
                    frame = frame.rename(columns={alias: "count"})
                    break

        # THE PLATE, from the pair row, and only when the file does not say.
        # Without this the four plates' wells pool: `prc` composed from row
        # and column alone makes plate1 r1/c1 and plate2 r1/c1 ONE well, and
        # every fraction below is then a share of four plates' reads. That is
        # a wrong number that looks right -- the fractions still sum to 1.
        if "plateID" not in frame.columns or frame["plateID"].isna().all():
            frame["plateID"] = f"plate{index + 1}"

        missing = [c for c in ("grna", "count") if c not in frame.columns]
        if missing:
            # NAME WHAT THE FILE ACTUALLY HAS. "column grna was not found" is
            # true and unactionable: the user cannot tell whether they picked
            # the wrong file or whether their header is spelled differently,
            # and those have different answers.
            problems.append(
                f"{os.path.basename(text)} has no {' or '.join(missing)} "
                f"column; it has {list(frame.columns)[:12]}")
            continue
        frames.append(frame)

    if not frames:
        detail = ("; ".join(problems) if problems
                  else "no count CSV was attached to the input table")
        raise MontageError(
            f"The per-well guide fractions could not be built from the count "
            f"CSVs: {detail}. They are the regression's own input -- the same "
            f"files the fit read -- so a run that produced coefficients has "
            f"them.")

    counts = pd.concat(frames, ignore_index=True, sort=False)
    if "prc" not in counts.columns:
        try:
            from .ml import _compose_prc_column
            counts["prc"] = _compose_prc_column(counts)
        except Exception as error:                               # noqa: BLE001
            raise MontageError(
                f"The count CSVs name no well: they need a 'prc' column, or "
                f"plateID / rowID / columnID to compose one ({error}).")

    totals = counts.groupby("prc")["count"].sum().reset_index()
    totals = totals.rename(columns={"count": "_well_total"})
    merged = pd.merge(counts, totals, on="prc", validate="many_to_one")
    merged["fraction"] = merged["count"] / merged["_well_total"]
    keep = [c for c in ("prc", "grna", "gene", "fraction", "count")
            if c in merged.columns]
    return merged[keep].copy()



def _read_scores(scores: Any):
    """Whatever a caller offered as scores, as one frame or ``None``.

    A frame, a path, or several paths -- a regression is paired per plate, so
    the score side is a list as often as it is one file, and a caller should
    not have to concatenate before asking.
    """
    if scores is None:
        return None
    if hasattr(scores, "columns"):
        return scores if len(scores) else None
    paths = [scores] if isinstance(scores, (str, os.PathLike)) else list(scores)
    frames = []
    for path in paths:
        try:
            text = os.fspath(path)
        except TypeError:
            continue
        if not text or not os.path.isfile(text):
            continue
        try:
            frames.append(_read_table(text, report=None))
        except Exception:                                        # noqa: BLE001
            continue
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True, sort=False)


def load_montage_objects(db_path: str, *, object_type: str = "cell",
                         score_column: str = "pred",
                         table: str = "png_list",
                         src: Optional[str] = None,
                         scores: Any = None,
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
    :param src: the folder the screen lives in NOW. Defaults to the plate
        folder the database sits in. Crop paths are recorded absolute at crop
        time, so a screen that has moved computer -- or a NAS mounted
        somewhere else -- carries 60,000 paths that no longer exist while
        every file is present; see :mod:`spacr.portable_paths`. Only paths
        that resolve to a file that EXISTS are rewritten.
    :param scores: where the per-object classification scores are, when the
        database has none: a frame, a path, or several paths -- the score CSVs
        the run was fitted on. Used ONLY when `png_list` carries no score
        column, and never written back.
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
    # CANONICALISE FIRST, the way every other reader in spaCR does -- and the
    # way `fractions_from_counts` was taught to earlier the same day, for the
    # same reason and by the same failure. Instruction 145.
    #
    # Measured on the maintainer's four plates, `png_list`:
    #
    #     plate1  rowID / columnID   and plateID = 'pplate1'
    #     plate2  row_name / column_name
    #     plate3  row_name / column_name
    #     plate4  row_name / column_name
    #
    # So plates 2-4 could not compose a `prc` at all, and plate1 composed one
    # against a doubled plate name that matches nothing in the counts. Every
    # well then reported "no object in the imported databases comes from this
    # well" and the montage drew nothing, while the baseline was happily
    # computed over all 226,467 objects -- which is what made it look like a
    # selection problem rather than a join one.
    try:
        from .schema import correct_metadata_column_names

        frame = correct_metadata_column_names(frame)
    except Exception:                                            # noqa: BLE001
        pass
    try:
        from .multi_database import normalise_plate_ids

        frame = normalise_plate_ids(frame)
    except Exception:                                            # noqa: BLE001
        pass

    if score_column not in frame.columns and scores is not None:
        # THE SCORES THE RUN ALREADY HAS (instruction 167). A screen whose
        # png_list has no `pred` is not a screen without scores: the score
        # CSVs the regression module is holding carry one row per cell, and
        # the fit was run on exactly those numbers. Joined through
        # `predictions.attach_predictions`, which is the SAME key choice
        # `merge_prediction_results` makes, so a montage reading them here and
        # a database that had them merged in cannot disagree.
        #
        # NOTHING IS WRITTEN. A montage is a read, which is the same rule the
        # crop-path re-rooting follows.
        from .predictions import attach_predictions

        table_of_scores = _read_scores(scores)
        if table_of_scores is not None:
            frame, matched = attach_predictions(frame, table_of_scores)
            if matched and verbose:
                print(f"Took {matched:,} score(s) from the loaded score "
                      f"table; {db_path} was not modified.")

    if score_column not in frame.columns:
        looked = (" The loaded score table has no column that joins to it "
                  "either." if scores is not None else
                  " No score table was offered alongside it.")
        raise MissingScores(
            f"{table!r} in {db_path} has no {score_column!r} column, so no "
            f"object carries a classification score.{looked} Either load the "
            "score CSVs this screen was fitted on, or run Classify and merge "
            "its predictions into the database.")
    _finite_scores(frame, score_column)

    from .png_list import crop_rows_from_png_list

    joined = crop_rows_from_png_list(db_path, frame, object_type=object_type,
                                     verbose=verbose)
    if joined.empty:
        # The join keeps only rows that can be cut from merged/. A PNG folder
        # alone is still a montage, so fall back rather than returning none.
        joined = frame.copy()
        joined["object_type"] = object_type
    # RE-ROOT BEFORE ANYTHING READS A PATH. The crop source is resolved from
    # the folder the user is looking at, so it is found correctly; it was the
    # per-object rows that still pointed at the machine the screen was
    # measured on, and a montage over 60,000 dead paths draws nothing and
    # blames the crops.
    root = src or portable_paths.source_root_for_database(db_path)
    for column in ("png_path", "path_name"):
        report = portable_paths.reroot_column(joined, column, root)
        # SAID WHEN IT CANNOT, not only when it can -- a crop that could not
        # be placed is returned unchanged and fails later as a missing file,
        # somewhere with less context (instruction 155 F). But a column where
        # NOTHING resolved is a route that is not on this machine, not 60,816
        # failures: a screen with PNG crops and no `merged/` folder is
        # healthy, and saying otherwise is the false alarm that teaches a
        # reader to ignore the true one.
        if report.partial or (report.moved and verbose) or (
                report.absent and verbose):
            print(report.describe())

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

#: Columns a row can carry an object's integer label in, in the order
#: :meth:`spacr.crops.MergedCropSource.spec_for` reads them.
_LABEL_COLUMNS: Tuple[str, ...] = (
    "object_label", "label", "cell_id", "nucleus_id", "pathogen_id",
    "cytoplasm_id")

#: The four bounding-box columns, in either spelling. All four or none:
#: three of them describe no box.
_BBOX_COLUMNS: Tuple[Tuple[str, str], ...] = tuple(
    (f"bbox-{i}", f"bbox_{i}") for i in range(4))


@dataclass(frozen=True)
class RouteRequirements:
    """What one route to pixels needs, and what it can therefore offer.

    THE TWO ROUTES NEED DIFFERENT THINGS and they are
    checked UP FRONT, because the alternative is a montage that fails halfway
    through cutting, or -- worse -- one that quietly cuts a different picture
    from the one the user chose:

    ROUTE 1, ``'merged-mask'`` -- live crop out of ``merged/<fov>.npy``.
    Needs THE IMAGE CHANNELS, THE MASK ARRAY and A BOUNDING BOX OR AN OBJECT
    ID. With an object id the crop follows the object's own outline, which is
    the better picture and the one worth defaulting to.

    ROUTE 2, ``'merged-bbox'`` -- coordinates from a table, then cut from the
    merged arrays. Needs THE TABLE AND THE CHANNELS, and offers BOUNDING-BOX
    CROPS ONLY: there is no mask in this route, so an object-shaped crop
    cannot be produced and MUST NOT be offered as a choice that silently does
    something else.

    ``'png'`` is the third answer and is not one of the two: the exported
    crops were cut when the run wrote them, so their shape is whatever
    ``measure_crop`` used and nothing here can change it.

    :param route: ``'png'`` / ``'merged-mask'`` / ``'merged-bbox'`` /
        ``'none'``.
    :param shapes: the crop shapes this route can actually produce, out of
        :data:`CROP_SHAPES`. Empty for the PNG route, which produces what it
        already produced.
    :param missing: what is needed and absent, each as a sentence naming the
        thing rather than the failure it would cause later.
    :param assumed: what was not stated and has been defaulted -- a channel
        list nobody recorded, say. Not a refusal: the montage draws, and the
        caption says which planes it drew from.
    :param detail: how the route was identified, for the status line.
    """

    route: str
    shapes: Tuple[str, ...] = ()
    missing: Tuple[str, ...] = ()
    assumed: Tuple[str, ...] = ()
    detail: str = ""

    @property
    def satisfied(self) -> bool:
        """True when this route has everything it needs to cut a crop."""
        return not self.missing and self.route != "none"

    def offers(self, shape: str) -> bool:
        """True when ``shape`` is a crop this route can really produce."""
        return str(shape) in self.shapes

    def why_not(self, shape: str) -> str:
        """Why ``shape`` is unavailable, or ``''`` when it is available.

        The sentence a disabled control carries. An object-shaped crop that
        cannot be cut is greyed out WITH THIS, never left clickable and
        quietly served a bounding box.
        """
        if self.offers(shape):
            return ""
        if self.route == "png":
            return ("The exported PNGs were cut when the run wrote them, so "
                    "their shape is whatever measure_crop used and cannot be "
                    "changed here. Force 'cut from merged/*.npy' to choose.")
        if str(shape) == "object":
            return ("This route has no mask array, only coordinates, so it "
                    "can cut bounding boxes and nothing else. An "
                    "object-shaped crop needs the mask plane the merged "
                    "array carries. " + (self.detail or ""))
        return f"{shape!r} is not one of {list(CROP_SHAPES)}"

    def describe(self) -> str:
        """The one-line answer for the tab's status line."""
        if self.route == "none":
            return "no route to pixels: " + "; ".join(self.missing)
        head = {
            "png": "exported PNG crops",
            "merged-mask": "live crop from merged/ (object-shaped available)",
            "merged-bbox": "coordinates from the table, bounding-box crops "
                           "only -- there is no mask in this route",
        }.get(self.route, self.route)
        if self.missing:
            return f"{head}; MISSING: " + "; ".join(self.missing)
        return head


def _has_column(frame, name: str) -> bool:
    return frame is not None and name in getattr(frame, "columns", ())


def montage_route_requirements(source, objects=None, *,
                               object_type: str = "cell",
                               channels: Optional[Sequence[int]] = None,
                               channels_declared: Optional[bool] = None
                               ) -> RouteRequirements:
    """Check what the chosen route needs BEFORE anything is cut.

    :param source: the :class:`spacr.crops.CropSource` (or the
        :class:`CropSourceChoice` holding one) that will draw the montage.
        ``None`` is the "no source" answer and comes back as route
        ``'none'``.
    :param objects: the per-object frame the crops will be cut from. Its
        COLUMNS are what decide between the two merged routes -- an object id
        means the mask can be followed, only a bounding box means it cannot.
        ``None`` skips the row checks and reports the route the source alone
        implies.
    :param object_type: which mask plane the crop is cut by.
    :param channels: the channels the user asked for, if any.
    :param channels_declared: whether a channel list exists at all -- typed
        by the user or recorded by the run. ``None`` infers it from
        ``channels``. False is the case the request names specifically: A
        USER MISSING A CHANNEL LIST IS TOLD THAT, not told there is no
        source.
    :returns: the :class:`RouteRequirements`.
    """
    choice = getattr(source, "source", source)
    if choice is None:
        reason = getattr(source, "reason", "") or "no crop source was resolved"
        return RouteRequirements(route="none", missing=(reason,))

    kind = str(getattr(choice, "kind", "") or "")
    assumed: List[str] = []
    missing: List[str] = []

    if kind == "png":
        if not _has_column(objects, "png_path") and objects is not None:
            missing.append(
                "the 'png_path' column: the exported-PNG route needs the path "
                "of each crop, and this object table carries none")
        return RouteRequirements(
            route="png", shapes=(), missing=tuple(missing),
            detail="the run's exported crops, read as written")

    # -- the merged routes ---------------------------------------------------
    spec = getattr(choice, "spec", None)
    declared = bool(channels) if channels_declared is None else bool(channels_declared)
    spec_channels = tuple(getattr(spec, "channels", ()) or ())
    if not declared:
        assumed.append(
            "no channel list: this run's measurements.db records no png_dims "
            "and none was typed, so the picture is made from the default "
            f"planes {list(channels or spec_channels) or [0, 1, 2]}. Set the "
            "channels if that is not what you meant.")

    mask_dims = dict(getattr(spec, "mask_dims", None) or {})
    if object_type == "cytoplasm":
        # Derived as cell minus nucleus/pathogen/organelle, so it needs the
        # cell plane and at least one to subtract.
        has_mask = "cell" in mask_dims and bool(
            {"nucleus", "pathogen"} & set(mask_dims))
        mask_detail = ("cytoplasm is derived as cell minus "
                       "nucleus/pathogen, so it needs both planes")
    else:
        has_mask = object_type in mask_dims
        mask_detail = (f"the merged arrays record no {object_type} mask "
                       f"plane; the planes they name are "
                       f"{sorted(mask_dims) or 'none'}")

    has_id = any(_has_column(objects, c) for c in _LABEL_COLUMNS)
    has_bbox = all(any(_has_column(objects, n) for n in pair)
                   for pair in _BBOX_COLUMNS)
    if objects is None:
        has_id = has_bbox = True

    if not has_id and not has_bbox:
        missing.append(
            "an object id or a bounding box: a crop cannot be located in the "
            "merged array without one. The object table needs "
            f"{list(_LABEL_COLUMNS[:3])} or all four of bbox-0..bbox-3.")

    if has_mask and has_id:
        route = "merged-mask"
        shapes: Tuple[str, ...] = ("object", "bbox")
        detail = (f"the merged arrays carry the {object_type} mask plane and "
                  "the objects carry their labels")
    else:
        route = "merged-bbox"
        shapes = ("bbox",)
        if not has_mask:
            detail = mask_detail
        else:
            detail = ("the object table carries bounding boxes but no object "
                      "label, so the mask cannot be followed")
        if not has_bbox and not missing:
            missing.append(
                "a bounding box: without a mask plane the crop can only be "
                "the recorded box, and this object table has no "
                "bbox-0..bbox-3 columns")
    return RouteRequirements(route=route, shapes=shapes,
                             missing=tuple(missing), assumed=tuple(assumed),
                             detail=detail)


@dataclass(frozen=True)
class CropSourceChoice:
    """Which crop source will draw a montage, or why none can.

    The tab needs both answers in the same shape: the design says a tab
    that cannot be filled must say why rather than be absent, so "there is no
    source" is a value here and not an exception.

    :param source: the :class:`spacr.crops.CropSource`, or ``None``.
    :param kind: ``'png'``, ``'merged'``, or ``''`` when unavailable.
    :param reason: why that source was picked, or why none could be.
    :param available: whether a montage can be drawn at all.
    :param requirements: what that route needs and what it can offer --
        :class:`RouteRequirements`, checked up front so a missing channel
        list is reported as a missing channel list rather than surfacing
        later as a crop that will not cut. ``None`` when nothing asked.
    """

    source: Any
    kind: str
    reason: str
    available: bool
    requirements: Optional[RouteRequirements] = None

    def describe(self) -> str:
        """Return the one-line sentence for the tab's status line."""
        if not self.available:
            return f"no crop source: {self.reason}"
        return f"{self.kind} crop source ({self.reason})"

    def requirement_notes(self) -> Tuple[str, ...]:
        """The caption lines this route's requirements oblige, if any."""
        req = self.requirements
        if req is None:
            return ()
        notes = [f"crop route: {req.describe()}"]
        notes.extend(f"ASSUMED -- {a}" for a in req.assumed)
        notes.extend(f"MISSING -- {m}" for m in req.missing)
        return tuple(notes)


def resolve_montage_crop_source(src, *, object_type: str = "cell",
                                prefer: Optional[str] = None,
                                objects: Optional[pd.DataFrame] = None,
                                channels: Optional[Sequence[int]] = None
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
    :param objects: the per-object frame, so the ROUTE's own requirements can
        be checked up front -- an object id means the mask can be followed, a
        bounding box alone means it cannot. Left out, the requirements are
        those the source alone implies.
    :param channels: the channels the caller asked for, so "no channel list
        anywhere" can be reported as exactly that.
    :returns: a :class:`CropSourceChoice`; ``available`` is False, with the
        reason, when neither source exists.
    """
    from .crops import CropError, crop_settings_from_db, resolve_crop_source

    try:
        source = resolve_crop_source(src, object_type=object_type,
                                     prefer=prefer)
    except CropError as exc:
        return CropSourceChoice(source=None, kind="", reason=str(exc),
                                available=False,
                                requirements=RouteRequirements(
                                    route="none", missing=(str(exc),)))
    declared = bool(channels)
    if not declared:
        # WHETHER A CHANNEL LIST EXISTS AT ALL, which is a different question
        # from whether the spec has channels: `crop_spec_from_settings`
        # always produces some, so an unrecorded run silently draws planes
        # 0,1,2 and looks like a deliberate choice.
        root = src.get("src") if isinstance(src, Mapping) else src
        if isinstance(root, (list, tuple)):
            root = root[0] if root else None
        if root:
            root = str(root)
            if os.path.basename(os.path.abspath(root)) == "merged":
                root = os.path.dirname(os.path.abspath(root))
            db = os.path.join(root, "measurements", "measurements.db")
            if os.path.isfile(db):
                try:
                    saved = crop_settings_from_db(db)
                except Exception:                               # noqa: BLE001
                    saved = {}
                declared = bool(saved.get("png_channel_mapping")
                                or saved.get("png_dims"))
    requirements = montage_route_requirements(
        source, objects, object_type=object_type, channels=channels,
        channels_declared=declared)
    return CropSourceChoice(source=source, kind=source.kind,
                            reason=source.reason, available=True,
                            requirements=requirements)
