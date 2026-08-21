"""What a bar's whisker means, and how long it is.

Instruction 204: "for the cell table graphs if bar is chosen the user should
be able to choose SD, Var, or SEM error bars", and instruction 200 asks for
the same choice in the figure settings.

ONE IMPLEMENTATION, TWO SCREENS. 204 says so directly -- "one
implementation, used by both, or the two screens will disagree about what
SEM means on the same data" -- and that is not a hypothetical: SD and SEM
differ by a factor of sqrt(n), so two screens using different definitions of
the same word would draw whiskers fifty-five times apart at n=3000 and both
label them the same.

THEY ARE NOT INTERCHANGEABLE AND THE PLOT MUST SAY WHICH IT DREW.

    SD   describes the CELLS: how spread out the population is. It does not
         shrink as you measure more of them.
    SEM  describes the MEAN: how precisely the centre is located. It shrinks
         as sqrt(n), so it is small whenever n is large, whatever the
         biology does.
    VAR  is SD squared, so it is in squared units and does not share the
         axis with the bar it sits on. Offered because it was asked for;
         drawn, but the label has to carry the units or the picture is
         wrong in a way that looks fine.

A reader who assumes the wrong one reads a real effect as noise, or noise as
a real effect. The choice belongs in the axis label or the legend -- not
only in the settings dialog the reader never opens.
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
    """The half-length of the whisker for ``values``.

    :param values: the observations behind one bar.
    :param kind: one of :data:`SPREAD_CHOICES`' values.
    :returns: the spread, or ``nan`` when it is not defined.

    THE SAMPLE STANDARD DEVIATION, ``ddof=1``, and that is a choice with a
    reason: these are cells drawn from a population, not the population, and
    the population formula understates the spread by ``sqrt((n-1)/n)`` --
    ten per cent at n=5, which is exactly the well that needed the whisker.

    ONE OBSERVATION HAS NO SPREAD. It returns ``nan`` rather than zero: a
    zero-length whisker on a bar of one cell says "no variation measured"
    where the truth is "not measurable", and those look identical on a plot.
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
    """What to write on the axis so the whisker is not ambiguous.

    :param unit: the measurement's unit, if there is one. VARIANCE SQUARES
        IT, which is the whole reason this takes the argument -- a variance
        whisker on an axis labelled with the plain unit is wrong in a way
        that looks fine.
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
    """``{level: {"mean", "spread", "n"}}`` for a bar chart.

    A level with no finite observation is omitted rather than drawn at
    zero -- an empty bar and a bar of zeros are different claims.
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
