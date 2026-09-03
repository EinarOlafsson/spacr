"""Evaluate residual exchangeability for blocked permutation tests.

Permutation inference assumes that residuals can be exchanged within each
block. This module reports serial autocorrelation, positional gradients, and
block-level diagnostics that may invalidate that assumption. These checks
complement residual-versus-fitted and Q-Q plots: normality and constant
variance do not establish exchangeability across plate positions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "block_residual_report",
    "autocorrelation",
    "position_effect",
    "exchangeability_verdict",
]


def autocorrelation(residuals: Sequence[float]) -> float:
    """Durbin-Watson on ``residuals`` in the order given.

    :param residuals: ordered residual values to test for serial dependence.
    :returns: the Durbin-Watson statistic, or ``nan`` with fewer than two
        finite residuals or a nonpositive sum of squared residuals.

    2 is no autocorrelation, 0 is perfect positive, 4 perfect negative.
    Written out rather than imported so this module does not pull
    statsmodels for one line -- and so the ORDER is explicit: it is the
    order the rows arrive in, which for a well table is plate reading order.
    """
    values = np.asarray(list(residuals), dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    diff = np.diff(values)
    denominator = float(np.sum(values ** 2))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(diff ** 2) / denominator)


def position_effect(residuals: Sequence[float],
                    positions: Sequence[Any]) -> Dict[str, float]:
    """How much of the residual one position column explains.

    :param residuals: residual values whose positional structure is measured.
    :param positions: position level aligned to each residual.

    :returns: a mapping containing ``eta_squared`` and ``levels``. With at
        least three finite observations it also contains ``p_value``; when
        two or more levels vary, it includes ``omega_squared``,
        ``worst_level``, and ``worst_departure``, the signed departure of
        that level's mean from the grand residual mean.
    :raises ValueError: if ``residuals`` and ``positions`` do not contain
        exactly one value for each other.

    THE NUMBER THAT MATTERS FOR EXCHANGEABILITY. If rows differ from each
    other more than cells within a row do, then shuffling across rows inside
    a plate is shuffling things that are not alike, and the permutation null
    is wider than the truth.
    """
    residual_values = list(residuals)
    position_values = list(positions)
    if len(residual_values) != len(position_values):
        raise ValueError(
            "residuals and positions must have the same length; got "
            f"{len(residual_values)} and {len(position_values)}")
    values = np.asarray(residual_values, dtype=float)
    labels = np.asarray([str(p) for p in position_values])
    keep = np.isfinite(values)
    values, labels = values[keep], labels[keep]
    if values.size < 3:
        return {"eta_squared": float("nan"), "levels": 0.0}
    grand = float(values.mean())
    total = float(np.sum((values - grand) ** 2))
    levels = sorted(set(labels.tolist()))
    if total <= 0 or len(levels) < 2:
        return {"eta_squared": 0.0, "p_value": 1.0,
                "levels": float(len(levels))}
    between, worst, worst_level = 0.0, 0.0, ""
    for level in levels:
        here = values[labels == level]
        between += here.size * (float(here.mean()) - grand) ** 2
        if abs(float(here.mean()) - grand) > abs(worst):
            worst, worst_level = float(here.mean()) - grand, level

    # ETA-SQUARED ALONE CANNOT BE COMPARED TO A FIXED THRESHOLD, and doing
    # so was this module's first bug: under the null it has an expected
    # value of about (k-1)/(n-1), so with twelve levels pure noise scores
    # 0.046 and any tolerance near 0.05 flags it. Caught immediately, on the
    # control case that was supposed to pass.
    #
    # THE F TEST IS THE COMPARISON THAT KNOWS THIS. It divides the between
    # -level variance by the within-level variance with the right degrees of
    # freedom, so "how many levels" is already accounted for and the answer
    # is a p-value that means the same thing at any k.
    within = total - between
    k, n = len(levels), int(values.size)
    p_value = 1.0
    if within > 0 and n > k:
        from scipy.stats import f as _f

        statistic = (between / (k - 1)) / (within / (n - k))
        p_value = float(_f.sf(statistic, k - 1, n - k))
    return {
        "eta_squared": float(between / total),
        # UNBIASED, so it can be reported beside the p-value without
        # contradicting it: omega-squared subtracts the variance the null
        # would have produced anyway.
        "omega_squared": float(max(
            0.0, (between - (k - 1) * (within / (n - k))) /
            (total + (within / (n - k))))) if within > 0 and n > k else 0.0,
        "p_value": p_value,
        "levels": float(k),
        "worst_level": worst_level,
        "worst_departure": float(worst),
    }


def block_residual_report(residuals: Sequence[float],
                          blocks: Sequence[Any],
                          positions: Optional[Mapping[str, Sequence[Any]]]
                          = None) -> Dict[str, Any]:
    """Everything the permutation's assumption needs, per block and overall.

    :param residuals: the phenotype residuals that WILL BE PERMUTED -- not a
        model's residuals, which is the distinction this whole module exists
        for.
    :param blocks: the block each residual belongs to; the shuffle happens
        inside these.
    :param positions: ``{'rowID': [...], 'columnID': [...]}`` or similar.
    :returns: pooled sample and block counts, pooled and per-block
        Durbin-Watson diagnostics, per-block means and standard deviations,
        and one :func:`position_effect` result per named position column.
    :raises ValueError: if ``blocks`` or any named position column does not
        contain exactly one value for each residual.

    PER BLOCK AND NOT ONLY POOLED. The shuffle is within-block, so a pooled
    statistic can look healthy while one plate is badly structured -- and
    that one plate is where the false positives come from.
    """
    residual_values = list(residuals)
    block_values = list(blocks)
    if len(residual_values) != len(block_values):
        raise ValueError(
            "residuals and blocks must have the same length; got "
            f"{len(residual_values)} and {len(block_values)}")
    position_values = {
        str(name): list(column) for name, column in (positions or {}).items()
    }
    for name, column in position_values.items():
        if len(column) != len(residual_values):
            raise ValueError(
                f"position column {name!r} must have the same length as "
                f"residuals; got {len(column)} and {len(residual_values)}")

    values = np.asarray(residual_values, dtype=float)
    labels = np.asarray([str(b) for b in block_values])
    out: Dict[str, Any] = {
        "n": int(values.size),
        "blocks": int(len(set(labels.tolist()))),
        "durbin_watson": autocorrelation(values),
        "per_block": {},
        "position": {},
    }
    for block in sorted(set(labels.tolist())):
        here = values[labels == block]
        out["per_block"][block] = {
            "n": int(here.size),
            "durbin_watson": autocorrelation(here),
            "mean": float(here.mean()) if here.size else float("nan"),
            "sd": float(here.std(ddof=1)) if here.size > 1 else float("nan"),
        }
    for name, column in position_values.items():
        out["position"][name] = position_effect(values, column)
    return out


#: How far from 2 the Durbin-Watson may sit before it is worth acting on.
#: 0.4 puts the boundary at 1.6/2.4, which is roughly where the standard
#: tables reject at n in the hundreds.
DW_TOLERANCE = 0.4

#: How unlikely a position column's structure must be, under the null of no
#: position effect, before it is worth acting on. A p-value rather than an
#: effect size, because eta-squared's null value depends on the number of
#: levels and a fixed cut flags noise wherever there are many.
POSITION_ALPHA = 0.01


def exchangeability_verdict(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Is the within-block shuffle defensible, and if not, what to change?

    :param report: diagnostics returned by :func:`block_residual_report`.

    :returns: ``{'ok': bool, 'findings': [...], 'remedy': str}``.

    THE REMEDY IS NAMED, NOT IMPLIED. "Durbin-Watson 1.22" is a number;
    "add rowID to guide_nuisance_columns" is something the reader can do,
    and it is the same setting the run already has.
    """
    findings: List[str] = []
    dw = float(report.get("durbin_watson", float("nan")))
    if np.isfinite(dw) and abs(dw - 2.0) > DW_TOLERANCE:
        direction = "positive" if dw < 2.0 else "negative"
        findings.append(
            f"Durbin-Watson {dw:.2f} against 2 for none -- {direction} "
            f"autocorrelation in row order. Neighbouring wells are not "
            f"independent, so shuffling them treats structure as noise.")
    for block, stats in (report.get("per_block") or {}).items():
        block_dw = float(stats.get("durbin_watson", float("nan")))
        if np.isfinite(block_dw) and abs(block_dw - 2.0) > DW_TOLERANCE:
            findings.append(
                f"block {block!r}: Durbin-Watson {block_dw:.2f} over "
                f"{stats['n']} well(s). The shuffle is WITHIN blocks, so one "
                f"structured block is where false positives come from even "
                f"when the pooled figure looks healthy.")
    culprits: List[str] = []
    for name, stats in (report.get("position") or {}).items():
        p_value = float(stats.get("p_value", 1.0))
        omega = float(stats.get("omega_squared", 0.0))
        if np.isfinite(p_value) and p_value < POSITION_ALPHA:
            culprits.append(name)
            findings.append(
                f"{name} explains {omega:.1%} of the residual variance "
                f"across {int(stats.get('levels', 0))} level(s) "
                f"(p = {p_value:.2g}). Wells at different {name} values are "
                f"not alike, so they are not swappable.")
    remedy = ""
    if culprits:
        remedy = (f"Add {', '.join(sorted(culprits))} to "
                  f"guide_nuisance_columns. They are then removed before the "
                  f"residualisation and the same statistic is evaluated on "
                  f"residuals that have a better claim to exchangeability.")
    elif findings:
        remedy = ("The structure is not explained by the position columns "
                  "measured here. Check whether the block column names the "
                  "right grouping -- a batch that spans plates is not "
                  "removed by blocking on plate.")
    return {"ok": not findings, "findings": findings, "remedy": remedy}
