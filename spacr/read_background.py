"""Measure and correct guide-read background using control wells.

Control wells with known guide composition provide direct observations of
reads assigned to guides that should be absent. Background is estimated per
guide because barcode-specific effects can differ substantially. The module
also separates two operations with different interpretations:

* exclusion removes sequences that cannot be present in cells, such as primer
  or plasmid carry-over, before fractions are calculated;
* background subtraction removes an estimated spurious component from a real
  guide and then optionally renormalizes the remaining fractions.

Control-well measurements provide an upper bound for ordinary wells when
cross-sample contamination scales with source abundance. Candidate outliers
are reported for imaging-based review rather than automatically classified as
sequencing artefacts.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np

__all__ = [
    "drop_guides",
    "resolve_exclusions",
    "unmatched_exclusions",
    "background_from_controls",
    "suggest_threshold",
    "subtract_background",
    "suspicious",
]


def drop_guides(fractions: Mapping[str, float],
                exclude: Iterable[str], *,
                renormalise: bool = True) -> Dict[str, float]:
    """Remove excluded sequences from a guide-fraction mapping.

    Parameters
    ----------
    fractions : mapping of str to float
        Guide fractions for one well.
    exclude : iterable of str
        Guide or gene identifiers to remove.
    renormalise : bool, default=True
        Rescale retained finite fractions to sum to one when their total is
        positive.

    Returns
    -------
    dict
        Retained guide fractions. Exclusion is applied before downstream
        background correction so removed sequences do not remain in the
        denominator.
    """
    names = [str(g) for g in fractions]
    drop = resolve_exclusions(exclude, names)
    kept = {str(g): float(v) for g, v in fractions.items()
            if str(g) not in drop and np.isfinite(float(v))}
    total = sum(kept.values())
    if renormalise and total > 0:
        return {g: v / total for g, v in kept.items()}
    return kept


def resolve_exclusions(exclude: Optional[Iterable[str]],
                       guides: Sequence[str],
                       genes: Optional[Sequence[str]] = None) -> Set[str]:
    """Resolve guide and gene exclusions to guide identifiers.

    Gene names select every associated guide. Matching uses the same
    organism-prefix handling as the control settings. If the shared resolver
    cannot run, exact guide-name matches are returned as a conservative
    fallback; unmatched inputs are omitted and can be reported with
    :func:`unmatched_exclusions`.
    """
    wanted = [e for e in (exclude or ()) if str(e).strip()]
    if not wanted:
        return set()
    names = [str(g) for g in guides]
    try:
        import pandas as pd

        from .control_names import rows_for

        series = pd.Series(names, dtype=object)
        out: Set[str] = set()
        for entry in wanted:
            mask, _note = rows_for(entry, series, genes, names=names)
            out.update(series[np.asarray(mask, dtype=bool)].tolist())
        return out
    except Exception:                                        # noqa: BLE001
        # A resolver that cannot run must not silently exclude NOTHING --
        # that would leave a known contaminant in the denominator. Fall back
        # to the exact names, which is the subset everybody agrees on.
        return {str(e) for e in wanted}


def unmatched_exclusions(exclude: Optional[Iterable[str]],
                         guides: Sequence[str],
                         genes: Optional[Sequence[str]] = None) -> List[str]:
    """Return exclusion entries that match no guide in the screen."""
    wanted = [str(e) for e in (exclude or ()) if str(e).strip()]
    missing: List[str] = []
    for entry in wanted:
        if not resolve_exclusions([entry], guides, genes):
            missing.append(entry)
    return missing


def background_from_controls(
        fractions: Mapping[str, Mapping[str, float]],
        intended: Mapping[str, Iterable[str]], *,
        exclude: Optional[Iterable[str]] = None,
        statistic: str = "median") -> Dict[str, object]:
    """Measure each guide's fraction where it should be absent.

    Parameters
    ----------
    fractions : mapping
        Nested mapping ``{well: {guide: fraction}}`` for control wells.
    intended : mapping
        Guides known to be present in each well. Wells missing from this
        mapping are skipped rather than treated as empty.
    exclude : iterable of str, optional
        Sequences to remove and renormalize before measuring background.
    statistic : {'median', 'mean'}, default='median'
        Summary applied across eligible control wells for each guide.

    Returns
    -------
    dict
        Per-guide background, occurrence counts, per-well spurious mass,
        aggregate mass statistics, and the number of controls used.
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
    """Estimate a global fraction threshold from diffuse background.

    Guides at least ``outlier_factor`` times the median are excluded from the
    quantile calculation and counted separately because a single threshold
    does not describe them. The result reports the threshold, sample counts,
    and the number of guides that require guide-specific review or correction.
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
    """Subtract guide-specific background from one well.

    ``scale`` multiplies the control-derived background before subtraction;
    values are clipped at zero. When ``renormalise`` is true, corrected values
    are rescaled to preserve the original finite total. Use a scale below one
    when control-well abundance is known to overstate contamination in
    ordinary wells.
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
    """Return guides with high, recurrent control-well background.

    Candidates must reach ``factor`` times the median background and appear
    in at least ``everywhere`` of eligible control wells. Results are sorted
    by decreasing background. Read counts alone cannot distinguish a
    sequencing artefact from a genuinely over-represented guide, so the
    returned verdict explicitly recommends imaging-based review.
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
