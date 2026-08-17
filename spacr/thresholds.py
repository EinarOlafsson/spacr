"""How wide a coefficient has to be before it counts as a hit.

A p-value says an effect is distinguishable from zero. With a thousand wells
that includes effects far too small to follow up, so the EFFECT-SIZE cut is
what separates "detectable" from "worth doing an experiment about". This
module is the one place that decides how wide the cut is.

THE SPREAD IS MEASURED ON THE CONTROLS wherever there are any: guides that
target nothing ARE the null, and that is the sentence a methods section
wants. Measured on the tsg101 screen the control-based cut and the
all-guide cut agree closely (0.83 against 0.84) -- but a screen with a
strong signal pulls the all-guide spread up while leaving the controls where
they are, and then only one of the two is still a null.

Added 2026-08-17 at the maintainer's request: "coefficient threshold mode
(none, var, std, also add several other methods that make sense at least 4
more)", reachable from the plot's right-click menu.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: The consistent scale estimator for a normal distribution. MAD x this is
#: an estimate of sigma that, unlike the standard deviation, is not inflated
#: by the outliers a screen exists to find.
MAD_TO_SIGMA = 1.4826


def _finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def _std(values) -> float:
    return float(np.std(_finite(values), ddof=1))


def _var(values) -> float:
    return float(np.var(_finite(values), ddof=1))


def _mad(values) -> float:
    array = _finite(values)
    return float(np.median(np.abs(array - np.median(array))) * MAD_TO_SIGMA)


def _iqr(values) -> float:
    array = _finite(values)
    return float(np.percentile(array, 75) - np.percentile(array, 25))


def _abs_percentile(values) -> float:
    """The 95th percentile of |value|, as a width."""
    return float(np.percentile(np.abs(_finite(values)), 95))


def _range(values) -> float:
    array = _finite(values)
    return float(array.max() - array.min())


#: ``{name: (spread function, one-line description)}``.
#:
#: Each is a WIDTH in the units of the coefficient, so `centre + k * width`
#: is a coefficient -- with one deliberate exception, noted below.
METHODS: Dict[str, Tuple[Optional[object], str]] = {
    "none": (None,
             "no effect-size cut; significance alone decides"),
    "std": (_std,
            "standard deviation of the control coefficients"),
    "var": (_var,
            "VARIANCE of the control coefficients -- squared units, see "
            "below"),
    "mad": (_mad,
            "median absolute deviation x 1.4826, the robust sigma"),
    "iqr": (_iqr,
            "interquartile range of the control coefficients"),
    "percentile": (_abs_percentile,
                   "95th percentile of |control coefficient|"),
    "range": (_range,
              "full range of the control coefficients, max - min"),
}

#: Spellings accepted for the same method, so an old settings CSV still loads.
ALIASES = {
    "standard_deveation": "std",     # spaCR's own historical misspelling
    "standard_deviation": "std",
    "variance": "var",
    "median_absolute_deviation": "mad",
    "interquartile_range": "iqr",
    "quantile": "percentile",
    "": "none",
}

#: `var` returns a width in SQUARED units, so `mean + k * var` adds a variance
#: to a coefficient and is dimensionally wrong. It is kept because it is what
#: spaCR shipped and what a saved settings file may carry -- and because the
#: maintainer asked for it by name -- but it is the one method whose number
#: cannot be read as "k spreads away from the centre". Below a spread of 1 it
#: is NARROWER than std and above it much wider, which is why a screen can
#: change character when nothing but the units moved.
DIMENSIONALLY_ODD = ("var",)


def canonical(method) -> str:
    """The canonical name for ``method``.

    :raises ValueError: naming every method, rather than falling back to a
        default the caller did not ask for.
    """
    key = str(method or "none").strip().lower().replace(" ", "_")
    key = ALIASES.get(key, key)
    if key not in METHODS:
        raise ValueError(
            f"Unsupported threshold method {method!r}. Choose one of: "
            f"{', '.join(METHODS)}.")
    return key


def describe(method) -> str:
    """One line saying what a method measures."""
    key = canonical(method)
    text = METHODS[key][1]
    if key in DIMENSIONALLY_ODD:
        text += (" -- k x variance is not k spreads from the centre, so this "
                 "cut is narrower than 'std' below a spread of 1 and much "
                 "wider above it")
    return text


def coefficient_threshold(values: Sequence[float], method="mad",
                          multiplier: float = 3.0,
                          centre: Optional[float] = None) -> Tuple[Optional[float], str]:
    """``(threshold, sentence)`` for a set of control coefficients.

    :param values: the control coefficients -- the null.
    :param method: one of :data:`METHODS`, or an alias.
    :param multiplier: how many spreads wide the cut is.
    :param centre: what to measure from; the MEDIAN of ``values`` by default,
        which is not moved by one control guide with a real phenotype. This
        screen has one -- `000000_22` is a non-targeting control and the
        strongest effect in the run at +4.37.
    :returns: ``(None, reason)`` when no cut can be made, never a silent 0.

    The sentence is not decoration: a threshold a reader cannot attribute is
    a threshold they cannot report, and it goes on the panel beside the line.
    """
    key = canonical(method)
    if key == "none":
        return None, "no effect-size cut"

    array = _finite(values)
    if array.size < 2:
        return None, (f"{array.size} control coefficient(s) is not enough to "
                      f"measure a spread")

    spread = METHODS[key][0](array)
    if not np.isfinite(spread) or spread <= 0:
        return None, (f"the control coefficients have no {key} spread "
                      f"(every one is the same value)")

    origin = float(np.median(array)) if centre is None else float(centre)
    threshold = abs(origin) + float(multiplier) * spread
    return threshold, (f"{multiplier:g}x {key} of {array.size} controls "
                       f"= {threshold:.3g}")


__all__ = ["ALIASES", "DIMENSIONALLY_ODD", "MAD_TO_SIGMA", "METHODS",
           "canonical", "coefficient_threshold", "describe"]
