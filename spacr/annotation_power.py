"""Estimate which guide assignments a pooled screen can support.

The calculations combine each guide's within-well fraction with classifier
sensitivity and specificity. They report the minimum fraction required for a
phenotype-positive call to reach a requested posterior probability, identify
guides that never reach that fraction, and estimate the screen design needed
to improve coverage.

Increasing the number of cells in a well increases the number of possible
assignments but does not change the posterior probability for an individual
cell. That probability changes when guide fractions or classifier performance
change. The functions in this module therefore keep annotation coverage and
assignment confidence as separate quantities.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "posterior_for_prior",
    "required_fraction",
    "annotatable",
    "screen_size_for",
    "quality_report",
]


def posterior_for_prior(prior: float, sensitivity: float,
                        specificity: float) -> float:
    """Return guide probability after a phenotype-positive classifier call.

    Parameters
    ----------
    prior : float
        Guide fraction before observing the classifier call.
    sensitivity : float
        Probability of a positive call for a guide-carrying cell.
    specificity : float
        Probability of a negative call for a cell without the guide.

    Returns
    -------
    float
        Posterior guide probability, or ``nan`` when the call has zero total
        probability under the supplied rates.
    """
    pi = float(prior)
    hit = pi * float(sensitivity)
    miss = (1.0 - pi) * (1.0 - float(specificity))
    total = hit + miss
    return float(hit / total) if total > 0 else float("nan")


def required_fraction(sensitivity: float, specificity: float, *,
                      decision: float = 0.55) -> float:
    """Return the minimum guide fraction for a requested posterior.

    ``decision`` is the minimum probability that a phenotype-positive cell
    carries the guide. The calculation accounts for false positives among
    cells that do not carry the guide.
    """
    se, sp = float(sensitivity), float(specificity)
    t = float(decision)
    numerator = t * (1.0 - sp)
    denominator = se * (1.0 - t) + numerator
    return float(numerator / denominator) if denominator > 0 else float("nan")


def annotatable(fractions: Mapping[str, Mapping[str, float]], *,
                sensitivity: float,
                specificity: float,
                decision: float = 0.55,
                cells_per_well: Optional[Mapping[str, int]] = None,
                ) -> Dict[str, object]:
    """Summarize the annotatable portion of a screen.

    Parameters
    ----------
    fractions : mapping
        Nested mapping ``{well: {guide: fraction}}``.
    sensitivity, specificity : float
        Classifier performance used to compute the minimum guide fraction.
    decision : float, default=0.55
        Required posterior probability for a guide assignment.
    cells_per_well : mapping, optional
        Cell count for each well. When supplied, the result includes an
        upper bound on the number of reachable cells.

    Returns
    -------
    dict
        Minimum fraction, reachable well-guide pairs and guides, and optional
        cell-count bounds. A guide is unreachable when it fails to meet the
        minimum fraction in every well.
    """
    floor = required_fraction(sensitivity, specificity, decision=decision)
    pairs = 0
    clearing = 0
    reachable: set = set()
    every: set = set()
    cells_total = 0
    cells_reachable = 0
    for well, here in fractions.items():
        size = int((cells_per_well or {}).get(well, 0))
        cells_total += size
        best = 0.0
        for guide, share in here.items():
            value = float(share)
            if not np.isfinite(value):
                continue
            pairs += 1
            every.add(str(guide))
            if value >= floor:
                clearing += 1
                reachable.add(str(guide))
                best = max(best, value)
        if best > 0:
            # An upper bound on the cells this well can yield: the largest
            # clearing guide's share of it. Generous on purpose -- it is a
            # CEILING, and a ceiling that flattered would be worthless.
            cells_reachable += int(round(size * best))

    return {
        "floor": float(floor),
        "pairs": int(pairs),
        "pairs_clearing": int(clearing),
        "pairs_clearing_share": float(clearing / pairs) if pairs else 0.0,
        "guides": len(every),
        "guides_reachable": len(reachable),
        "guides_unreachable": len(every) - len(reachable),
        "guides_reachable_share": (len(reachable) / len(every)) if every
        else 0.0,
        "cells": int(cells_total),
        "cells_reachable_ceiling": int(cells_reachable),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "decision": float(decision),
    }


def screen_size_for(fractions: Mapping[str, Mapping[str, float]], *,
                    sensitivity: float,
                    specificity: float,
                    decision: float = 0.55,
                    target: float = 0.80) -> Dict[str, object]:
    """Estimate the screen design needed to improve guide reachability.

    Parameters
    ----------
    fractions : mapping
        Nested mapping ``{well: {guide: fraction}}``.
    sensitivity, specificity : float
        Classifier performance used to compute the minimum guide fraction.
    decision : float, default=0.55
        Required posterior probability for a guide assignment.
    target : float, default=0.80
        Target share recorded in the result for reporting.

    Returns
    -------
    dict
        Current screen shape, estimated guides per well and wells required,
        size multiplier, and the specificity required at the current shape.

    Notes
    -----
    The estimate holds library size and mean wells per guide fixed. It raises
    typical guide fractions by placing fewer guides in each well and therefore
    increasing the number of wells proportionally.
    """
    floor = required_fraction(sensitivity, specificity, decision=decision)
    wells = len(fractions)
    per_well = [len(v) for v in fractions.values()]
    if not wells or not per_well:
        return {"error": "no wells"}

    guides: set = set()
    appearances: Dict[str, int] = {}
    for here in fractions.values():
        for guide in here:
            guides.add(str(guide))
            appearances[str(guide)] = appearances.get(str(guide), 0) + 1
    library = len(guides)
    coverage = float(np.mean(list(appearances.values()))) if appearances else 0.0
    now = float(np.median(per_well))

    # For a typical guide to clear the floor its share must be at least
    # `floor`, and in a well of `k` guides the typical share is about 1/k.
    needed_per_well = float(1.0 / floor) if floor > 0 else float("inf")
    # Same library, same wells-per-guide, fewer guides in each well.
    needed_wells = (library * coverage / needed_per_well
                    if needed_per_well > 0 else float("inf"))

    return {
        "floor": float(floor),
        "wells_now": int(wells),
        "library": int(library),
        "guides_per_well_now": now,
        "wells_per_guide_now": coverage,
        "guides_per_well_needed": needed_per_well,
        "wells_needed": float(needed_wells),
        "wells_multiplier": float(needed_wells / wells) if wells else
        float("inf"),
        "library_if_wells_fixed": float(wells * needed_per_well / coverage)
        if coverage > 0 else float("nan"),
        "target": float(target),
        "specificity_needed_at_current_shape": _specificity_for(
            now, sensitivity, decision),
    }


def _specificity_for(guides_per_well: float, sensitivity: float,
                     decision: float) -> float:
    """The specificity that would make a typical well's guide reachable.

    The other lever, and usually the cheaper one to reach for -- though
    these numbers show how far it has to move.
    """
    pi = 1.0 / float(guides_per_well) if guides_per_well > 0 else 0.0
    t, se = float(decision), float(sensitivity)
    if pi <= 0 or t >= 1.0:
        return float("nan")
    # Solve P(g|+) = t for (1 - sp).
    false_positive = pi * se * (1.0 - t) / (t * (1.0 - pi))
    return float(np.clip(1.0 - false_positive, 0.0, 1.0))


def quality_report(verdicts: Mapping[str, object], *,
                   power: Optional[Mapping[str, object]] = None,
                   size: Optional[Mapping[str, object]] = None,
                   width: int = 78) -> str:
    """Format annotation quality and power metrics as a text report.

    Parameters
    ----------
    verdicts : mapping
        Mapping from method name to
        :class:`spacr.annotation_validation.Verdict`.
    power : mapping, optional
        Result from :func:`annotatable`.
    size : mapping, optional
        Result from :func:`screen_size_for`.
    width : int, default=78
        Rule width used in the text layout.

    Returns
    -------
    str
        Report that presents coverage, precision, and recall separately,
        followed by optional reachability and screen-size estimates.
    """
    lines: List[str] = []
    rows = []
    for name, verdict in verdicts.items():
        rows.append((str(name), float(getattr(verdict, "coverage", 0.0)),
                     float(getattr(verdict, "precision", 0.0)),
                     float(getattr(verdict, "recall", 0.0)),
                     int(getattr(verdict, "n", 0))))
    rows.sort(key=lambda r: -r[3])

    lines.append("Annotation quality")
    lines.append("-" * width)
    lines.append(f"{'method':<22}{'annotated':>11}{'of those':>11}"
                 f"{'right, all':>12}")
    lines.append(f"{'':<22}{'':>11}{'correct':>11}{'cells':>12}")
    for name, coverage, precision, recall, _n in rows:
        lines.append(f"{name:<22}{coverage:>10.1%}{precision:>11.1%}"
                     f"{recall:>12.1%}")
    lines.append("")
    lines.append("Coverage, precision, and recall are reported separately.")
    lines.append("Methods are ranked by the share of all cells called correctly.")

    if power:
        lines.append("")
        lines.append("Screen reachability")
        lines.append("-" * width)
        floor = float(power.get("floor", float("nan")))
        lines.append(
            f"A guide must be {floor:.1%} of a well before a positive call "
            f"is more")
        lines.append(
            f"likely right than wrong (classifier se "
            f"{float(power.get('sensitivity', 0)):.3f}, sp "
            f"{float(power.get('specificity', 0)):.3f}).")
        lines.append("")
        lines.append(
            f"  well-guide pairs clearing it : "
            f"{int(power.get('pairs_clearing', 0)):,} of "
            f"{int(power.get('pairs', 0)):,} "
            f"({float(power.get('pairs_clearing_share', 0)):.1%})")
        lines.append(
            f"  guides reachable anywhere    : "
            f"{int(power.get('guides_reachable', 0)):,} of "
            f"{int(power.get('guides', 0)):,} "
            f"({float(power.get('guides_reachable_share', 0)):.1%})")
        unreachable = int(power.get("guides_unreachable", 0))
        if unreachable:
            lines.append("")
            lines.append(
                f"  {unreachable:,} guides never reach it in any well. No "
                f"method can")
            lines.append(
                "  annotate cells for those guides from this screen design.")

    if size:
        lines.append("")
        lines.append("Estimated screen size")
        lines.append("-" * width)
        lines.append(
            f"  now    : {int(size.get('wells_now', 0)):,} wells, "
            f"{float(size.get('guides_per_well_now', 0)):,.0f} guides in a "
            f"typical well")
        lines.append(
            f"  needed : {float(size.get('guides_per_well_needed', 0)):,.0f} "
            f"guides per well, so "
            f"{float(size.get('wells_needed', 0)):,.0f} wells")
        lines.append(
            f"  that is {float(size.get('wells_multiplier', 0)):.1f}x the "
            f"plate count, for the same library and coverage.")
        lines.append("")
        lines.append(
            f"  Or hold the wells and cut the library to "
            f"{float(size.get('library_if_wells_fixed', 0)):,.0f} guides.")
        needed_sp = float(size.get("specificity_needed_at_current_shape",
                                   float("nan")))
        if np.isfinite(needed_sp):
            lines.append(
                f"  Or raise specificity to {needed_sp:.5f} at the current "
                f"shape.")
        lines.append("")
        lines.append("  More cells per well increase the number of possible")
        lines.append("  assignments but do not change the guide fraction or")
        lines.append("  per-cell assignment probability.")

    return "\n".join(lines)
