"""What a guide's reads look like in wells it is not in.

Proposed 2026-08-21: "another way to calculate the gRNA fraction threshold
is measure the total fraction and the individual guide fractions in that are
detected in the first 3 columns. as the first should be 100% 233460, the
second column 100% 249730 and the thired a mix but no other guides. this
might also be used somehow as a single calue grna correction factor or a
individual grna correction factor."

IT IS THE RIGHT IDEA AND THE ANSWER IS THE INDIVIDUAL FACTOR, NOT THE SINGLE
VALUE. A column of known composition is a direct measurement of the spurious
read rate: every read that is not the intended guide is, by construction,
noise. `fraction_threshold` currently guesses at that quantity with a
constant, and it can be measured instead.

WHAT THE MEASUREMENT SHOWS, on the screen this was written against and
recomputable on any other: the background is not one number wearing
different hats. Across the control wells the median guide's background is
0.013% and the 99th percentile is 0.15% -- a diffuse haze that any small
threshold removes -- while ONE guide sat at 9.2% and appeared in every
single control well. A global threshold cannot serve both. Set low enough to
catch the outlier it deletes real biology; set where it is now it deletes
the haze it did not need to and admits the outlier anyway.

SO THE BACKGROUND IS PER GUIDE, and a guide that shows up where it cannot be
is telling you a fact about itself -- its barcode, its amplification, its
neighbours on the index -- that no other guide's behaviour predicts.

WHAT THIS CANNOT DECIDE, and it must not pretend to. A guide abundant in the
control wells is EITHER a sequencing artefact -- index hopping, barcode
collision -- OR genuinely over-represented in the library and really present
in those cells. The read counts cannot tell those apart, and the correction
is opposite in the two cases: subtract it, or keep it. The imaging can tell
them apart, because a guide really present will move its cells' phenotype
and an artefact will not. :func:`suspicious` reports the candidates rather
than resolving them.

INDEX HOPPING SCALES WITH THE SOURCE'S ABUNDANCE, which is the reason these
columns give an UPPER bound rather than an estimate. A control well with one
guide at 70% sheds more reads onto its neighbours than an ordinary well
whose largest guide is 5%. The per-guide ranking transfers; the absolute
level does not, and :func:`subtract_background` says so when asked to apply
it to ordinary wells.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np

__all__ = [
    "drop_guides",
    "background_from_controls",
    "suggest_threshold",
    "subtract_background",
    "suspicious",
]


def drop_guides(fractions: Mapping[str, float],
                exclude: Iterable[str], *,
                renormalise: bool = True) -> Dict[str, float]:
    """Remove guides that are not in any cell, and renormalise what is left.

    EXCLUSION IS NOT BACKGROUND SUBTRACTION and the difference is the whole
    point. Background subtraction says "some of this guide's reads here are
    spurious". Exclusion says "this sequence is not a guide in a cell at
    all" -- primer or plasmid carry-over, which is in the library prep and
    never in a nucleus.

    So it is removed BEFORE the fractions are formed, and the survivors are
    renormalised over what remains. Leaving it in and subtracting later
    would leave its reads in every other guide's denominator, holding every
    real fraction down by the contaminant's share.

    On the screen this was written against, one plasmid contaminant was
    19.9% of ALL reads on a plate: excluding it raises every other guide's
    fraction by a quarter, which moves guides across the annotatability
    floor. It is not a tidying step.
    """
    drop = {str(g) for g in (exclude or ())}
    kept = {str(g): float(v) for g, v in fractions.items()
            if str(g) not in drop and np.isfinite(float(v))}
    total = sum(kept.values())
    if renormalise and total > 0:
        return {g: v / total for g, v in kept.items()}
    return kept


def background_from_controls(
        fractions: Mapping[str, Mapping[str, float]],
        intended: Mapping[str, Iterable[str]], *,
        exclude: Optional[Iterable[str]] = None,
        statistic: str = "median") -> Dict[str, object]:
    """Measure each guide's read fraction in wells it does not belong to.

    :param fractions: ``{well: {guide: fraction}}`` for the CONTROL wells
        only.
    :param intended: ``{well: the guides that well really contains}``. A
        well absent from this mapping is skipped rather than assumed empty,
        because assuming would turn a real guide into background.
    :param exclude: guides that are not in any cell -- primer or plasmid
        carry-over. Dropped and the well renormalised BEFORE anything is
        measured, because a contaminant left in the denominator holds every
        real guide's background down by its own share and makes the
        threshold look tighter than it is.
    :param statistic: ``'median'`` or ``'mean'`` across the control wells.
        Median by default: one contaminated well should not set a guide's
        background for the whole plate.
    :returns: the per-guide background, the total spurious mass per well,
        and the counts behind both.

    THE SPURIOUS MASS IS REPORTED PER WELL AND NOT ONLY AVERAGED, because
    its SPREAD is the thing that says whether these wells are describing one
    phenomenon. A tight spread is a sequencing property; a wide one means
    some wells were contaminated and the rest were not.
    """
    per_guide: Dict[str, List[float]] = {}
    seen_in: Dict[str, int] = {}
    spurious_mass: Dict[str, float] = {}
    wells_used = 0

    for well, raw in fractions.items():
        belongs = {str(g) for g in intended.get(well, ())}
        if not belongs:
            continue
        here = drop_guides(raw, exclude or ()) if exclude else raw
        wells_used += 1
        mass = 0.0
        for guide, share in here.items():
            name = str(guide)
            value = float(share)
            if not np.isfinite(value) or name in belongs:
                continue
            per_guide.setdefault(name, []).append(value)
            seen_in[name] = seen_in.get(name, 0) + 1
            mass += value
        spurious_mass[str(well)] = mass

    pick = np.median if str(statistic) == "median" else np.mean
    background = {name: float(pick(values))
                  for name, values in per_guide.items()}
    masses = np.asarray(list(spurious_mass.values()), dtype=float)

    return {
        "background": background,
        "seen_in_wells": seen_in,
        "control_wells": int(wells_used),
        "spurious_mass_per_well": spurious_mass,
        "spurious_mass_median": float(np.median(masses)) if masses.size
        else float("nan"),
        "spurious_mass_min": float(masses.min()) if masses.size
        else float("nan"),
        "spurious_mass_max": float(masses.max()) if masses.size
        else float("nan"),
        "statistic": str(statistic),
    }


def suggest_threshold(measurement: Mapping[str, object], *,
                      quantile: float = 0.99,
                      outlier_factor: float = 20.0) -> Dict[str, float]:
    """A `fraction_threshold` taken from the data instead of a constant.

    :param quantile: the share of the background distribution to clear.
    :returns: the suggested threshold and what it would and would not
        remove.

    IT IS ONLY HONEST FOR THE DIFFUSE PART. The quantile describes the haze
    of guides that appear at trace level everywhere. A guide whose
    background is an outlier is not described by any quantile of the others
    and needs :func:`subtract_background`; the count of those is returned
    beside the number so it cannot be read as a complete answer.
    """
    background = dict(measurement.get("background") or {})
    if not background:
        return {"threshold": float("nan"), "guides": 0.0}
    values = np.asarray(list(background.values()), dtype=float)
    middle = float(np.median(values))

    # THE OUTLIERS ARE REMOVED BEFORE THE QUANTILE IS TAKEN, and leaving
    # them in was a real fault rather than a rounding one: the quantile is
    # contaminated by exactly the guides it is supposed to exclude. Caught
    # on a 42-guide fixture where one outlier at 9% dragged the 99th
    # percentile to 6.6% -- a threshold that would delete most of a real
    # library. On a 1,325-guide screen the same outlier hid inside the
    # quantile instead, which is worse, because nothing looked wrong.
    #
    # `factor` matches `suspicious`, so the guides excluded here are the
    # guides that function reports. One rule, applied twice.
    keep = values[values < max(middle, 1e-12) * float(outlier_factor)]
    if keep.size == 0:                       # every guide is an outlier
        keep = values
    suggested = float(np.quantile(keep, float(quantile)))

    return {
        "threshold": suggested,
        "quantile": float(quantile),
        "guides": float(values.size),
        "guides_used": float(keep.size),
        "median_background": middle,
        "guides_above": float((values >= suggested).sum()),
        # The ones a single threshold cannot serve, whatever it is set to:
        # they were left out of the estimate and clear it anyway.
        "guides_needing_their_own": float(values.size - keep.size),
    }


def subtract_background(fractions: Mapping[str, float],
                        background: Mapping[str, float], *,
                        scale: float = 1.0,
                        renormalise: bool = True) -> Dict[str, float]:
    """Remove each guide's own background from a well, then renormalise.

    :param scale: how much of the measured background to subtract. THE
        CONTROL COLUMNS OVERSTATE IT for an ordinary well -- index hopping
        scales with the abundance of the source, and a control well has one
        guide at seventy per cent where an ordinary well's largest is five.
        1.0 is the conservative choice and removes the most; a caller who
        has estimated the ratio between the two regimes passes it here.
    :param renormalise: rescale the survivors to sum to what they summed to
        before, so the correction moves the SHARES between guides without
        also moving the well's total.
    """
    out: Dict[str, float] = {}
    before = 0.0
    for guide, share in fractions.items():
        value = float(share)
        if not np.isfinite(value):
            continue
        before += value
        out[str(guide)] = max(
            0.0, value - float(scale) * float(background.get(str(guide), 0.0)))
    after = sum(out.values())
    if renormalise and after > 0 and before > 0:
        factor = before / after
        out = {guide: value * factor for guide, value in out.items()}
    return out


def suspicious(measurement: Mapping[str, object], *,
               factor: float = 20.0,
               everywhere: float = 0.9) -> List[Dict[str, object]]:
    """Guides whose background is unlike every other guide's.

    :param factor: how many times the median background counts as an
        outlier.
    :param everywhere: appearing in at least this share of control wells.
    :returns: one row per candidate, worst first.

    THESE ARE CANDIDATES AND NOT A VERDICT. A guide abundant in wells it
    cannot be in is either a sequencing artefact or genuinely
    over-represented in the library and really in those cells -- and the
    correction is opposite in the two cases. Read counts cannot separate
    them. The imaging can: a guide really present moves its cells'
    phenotype, an artefact does not.

    APPEARING EVERYWHERE IS THE STRONGER SIGNAL, more than the level. A
    guide at one per cent in every control well is behaving systematically;
    a guide at five per cent in one well is a contaminated well.
    """
    background = dict(measurement.get("background") or {})
    seen = dict(measurement.get("seen_in_wells") or {})
    wells = int(measurement.get("control_wells") or 0)
    if not background or not wells:
        return []
    values = np.asarray(list(background.values()), dtype=float)
    middle = float(np.median(values))
    out: List[Dict[str, object]] = []
    for guide, level in background.items():
        share = seen.get(guide, 0) / wells
        if level >= middle * float(factor) and share >= float(everywhere):
            out.append({
                "guide": str(guide),
                "background": float(level),
                "times_median": float(level / middle) if middle > 0
                else float("inf"),
                "in_wells": int(seen.get(guide, 0)),
                "of_wells": wells,
                "verdict": "artefact or genuinely over-represented -- the "
                           "reads cannot say which, the imaging can",
            })
    out.sort(key=lambda row: -float(row["background"]))
    return out
