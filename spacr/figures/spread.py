"""Compute and label uncertainty or dispersion for bar charts.

The standard deviation (SD) describes variation among observations, whereas
the standard error of the mean (SEM) describes precision of the estimated
mean and decreases with sample size. Variance is SD squared and therefore has
squared units. Shared helpers keep these definitions and their plot labels
consistent across graph-building surfaces.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "SPREAD_CHOICES",
    "SPREAD_NONE",
    "SPREAD_SD",
    "SPREAD_SEM",
    "SPREAD_VAR",
    "spread_of",
    "spread_label",
    "summarise",
]

SPREAD_NONE = "none"
SPREAD_SD = "sd"
SPREAD_SEM = "sem"
SPREAD_VAR = "var"

#: ``(value, label)`` for a settings control. The label says what the
#: quantity DESCRIBES, because that is the thing a reader gets wrong.
SPREAD_CHOICES: Tuple[Tuple[str, str], ...] = (
    (SPREAD_NONE, "none — the bar alone"),
    (SPREAD_SD, "SD — how spread out the cells are"),
    (SPREAD_SEM, "SEM — how precisely the mean is located"),
    (SPREAD_VAR, "variance — SD squared, in squared units"),
)


def spread_of(values: Sequence[float], kind: str) -> float:
    """Return the whisker half-length for a sequence of observations.

    :param values: the observations behind one bar.
    :param kind: one of :data:`SPREAD_CHOICES`' values.
    :returns: the spread, or ``nan`` when it is not defined.

    SD and variance use the sample definition (``ddof=1``). Fewer than two
    finite observations return ``nan`` because their spread is not
    measurable; a zero would incorrectly claim that no variation was found.
    """
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n < 2:
        return float("nan")
    if kind == SPREAD_SD:
        return float(np.std(array, ddof=1))
    if kind == SPREAD_VAR:
        return float(np.var(array, ddof=1))
    if kind == SPREAD_SEM:
        return float(np.std(array, ddof=1) / np.sqrt(n))
    if kind == SPREAD_NONE:
        return float("nan")
    raise ValueError(
        f"spread {kind!r} is not one of "
        f"{[value for value, _ in SPREAD_CHOICES]}")


def spread_label(kind: str, *, unit: str = "") -> str:
    """Return an axis label that identifies the whisker statistic.

    :param kind: one of :data:`SPREAD_CHOICES`; ``none`` produces an empty
        label and any unrecognised value raises :class:`ValueError`.
    :param unit: measurement unit. Variance labels append a squared unit.
    """
    if kind == SPREAD_NONE:
        return ""
    if kind == SPREAD_SD:
        return f"mean ± SD ({unit})" if unit else "mean ± SD"
    if kind == SPREAD_SEM:
        return f"mean ± SEM ({unit})" if unit else "mean ± SEM"
    if kind == SPREAD_VAR:
        return f"mean ± variance ({unit}²)" if unit else "mean ± variance"
    raise ValueError(f"spread {kind!r} has no label")


def summarise(groups: Dict[str, Sequence[float]],
              kind: str) -> Dict[str, Dict[str, float]]:
    """Summarize grouped observations for a bar chart.

    :returns: ``{level: {"mean", "spread", "n"}}``. Groups without finite
        observations are omitted rather than drawn as zero-valued bars.
    """
    out: Dict[str, Dict[str, float]] = {}
    for level, values in groups.items():
        array = np.asarray(list(values), dtype=float)
        array = array[np.isfinite(array)]
        if not array.size:
            continue
        out[str(level)] = {
            "mean": float(array.mean()),
            "spread": spread_of(array, kind),
            "n": float(array.size),
        }
    return out
