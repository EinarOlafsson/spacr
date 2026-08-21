"""How many cells CAN be annotated, and how big the screen would have to be.

Asked 2026-08-21: "i dont need all cells annotated. for each method i want as
many cells as the method can reasonably annotate annotated ... with no method
can we annotate every cell, for that we would need a much largesr screen.
that is something that can be printed in the textbox under the graph, all the
quality metricks that can be caluclated for each method and how large the
screen would have to be based on these data".

COVERAGE IS NOT A GOAL, which reverses how the benchmark should be read. A
method that annotates every cell has not done well; it has declined to
abstain. The question each method answers is "which cells can I speak about",
and the honest ones leave most of a pooled screen alone.

THE LIMIT IS ARITHMETIC, NOT EFFORT. Take the simplest method's model: a
guide `g` is a fraction `pi` of a well's reads, only `g` produces the
phenotype, and the classifier has sensitivity `se` and specificity `sp`. A
cell called phenotype-positive carries `g` with probability

    P(g | +)  =  pi * se  /  ( pi * se  +  (1 - pi) * (1 - sp) )

and the second term in that denominator is the trap. FALSE POSITIVES ARE A
SHARE OF THE NEGATIVES. When `pi` is small nearly every cell in the well is a
negative, so a small false-positive rate applied to almost everything
outnumbers a large true-positive rate applied to almost nothing. Below a
certain `pi` a positive call is more likely to be a mistake than a hit, and
NO METHOD USING THAT EVIDENCE CAN DO BETTER -- it is not a limitation of the
algorithm but of what the well contains.

Rearranged, the fraction a guide must reach for a decision at confidence
``t``:

    pi*  =  t (1 - sp)  /  ( se (1 - t)  +  t (1 - sp) )

MORE CELLS PER WELL DO NOT HELP, and this is the counterintuitive part worth
stating before anyone images more fields. `pi` is a fraction: doubling the
cells doubles the true positives and the false positives together, so the
per-cell confidence is unchanged. It buys more annotated cells at the same
rate, never a higher rate. The rate moves only when the guide's SHARE moves
-- fewer guides per well -- or when the classifier separates better.
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
    """P(the cell carries the guide | it was called phenotype-positive)."""
    pi = float(prior)
    hit = pi * float(sensitivity)
    miss = (1.0 - pi) * (1.0 - float(specificity))
    total = hit + miss
    return float(hit / total) if total > 0 else float("nan")


def required_fraction(sensitivity: float, specificity: float, *,
                      decision: float = 0.55) -> float:
    """The smallest guide fraction that can support a call at ``decision``.

    Below this the arithmetic is against you whatever the method: a
    phenotype-positive cell in such a well is more likely to be one of the
    many negatives misread than one of the few positives.
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
    """How much of a screen is reachable at all, given its own fractions.

    :param fractions: ``{well: {guide: fraction}}`` -- the screen as it is.
    :param cells_per_well: optional, to turn the share of well-guide pairs
        into a share of CELLS.
    :returns: the floor, what clears it, and what that leaves unreachable.

    THE UNREACHABLE GUIDES ARE THE HEADLINE. A guide that never reaches
    `pi*` in any well cannot be annotated anywhere in the screen, so no
    amount of method development will produce a single cell for it. That is
    a fact about the experiment, available before any method is run, and it
    is the number that should decide whether to run one.
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
    """What the screen would have to be for ``target`` of guides to be
    reachable.

    THE ONE LEVER IS GUIDES PER WELL. A guide's fraction is its share of the
    well, so the way to raise it is to put fewer guides in each well -- and
    holding the library and the coverage per guide fixed, that means more
    wells, in direct proportion.

    :param target: the share of guides that should reach the floor.
    :returns: the current shape, the shape needed, and the ratio between --
        plus what the same result would cost through the classifier
        instead, since a better model is the other way to move the floor.
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
    """The textbox under the graph: every method's metrics, and the ceiling.

    :param verdicts: ``{method: Verdict}`` from
        :func:`spacr.annotation_validation.score_annotation`.
    :param power: the :func:`annotatable` result for this screen.
    :param size: the :func:`screen_size_for` result.

    COVERAGE IS PRINTED BESIDE PRECISION AND NEVER BLENDED WITH IT. The
    methods are ordered by the cells they got RIGHT, which is the only
    ranking that does not reward either annotating everything badly or
    annotating nothing safely.
    """
    lines: List[str] = []
    rows = []
    for name, verdict in verdicts.items():
        rows.append((str(name), float(getattr(verdict, "coverage", 0.0)),
                     float(getattr(verdict, "precision", 0.0)),
                     float(getattr(verdict, "recall", 0.0)),
                     int(getattr(verdict, "n", 0))))
    rows.sort(key=lambda r: -r[3])

    lines.append("ANNOTATION QUALITY")
    lines.append("-" * width)
    lines.append(f"{'method':<22}{'annotated':>11}{'of those':>11}"
                 f"{'right, all':>12}")
    lines.append(f"{'':<22}{'':>11}{'correct':>11}{'cells':>12}")
    for name, coverage, precision, recall, _n in rows:
        lines.append(f"{name:<22}{coverage:>10.1%}{precision:>11.1%}"
                     f"{recall:>12.1%}")
    lines.append("")
    lines.append("A method that annotates everything has not done well, it")
    lines.append("has declined to abstain. Ranked by cells got right.")

    if power:
        lines.append("")
        lines.append("WHAT THIS SCREEN CAN SUPPORT AT ALL")
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
                f"  {unreachable:,} guides never reach it in ANY well. No "
                f"method can")
            lines.append(
                "  annotate a single cell for them -- that is the "
                "experiment,")
            lines.append("  not the algorithm.")

    if size:
        lines.append("")
        lines.append("HOW MUCH BIGGER THE SCREEN WOULD HAVE TO BE")
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
        lines.append("  MORE CELLS PER WELL DO NOT MOVE ANY OF THIS. The")
        lines.append("  fraction is a share, so more cells give more")
        lines.append("  annotated cells at the same rate, never a better")
        lines.append("  rate.")

    return "\n".join(lines)
