"""Define the reference used to report regression effect sizes.

For a model of the form

    y ~ fraction:grna + gene_fraction:gene + rowID + columnID

Patsy assigns coefficients to every guide and gene because both occur only in
interactions with a continuous fraction. The first plate row and column define
the intercept reference, but changing those categorical references shifts the
intercept without changing guide or gene coefficients.

Guide coefficients are slopes with respect to guide fraction and therefore
use zero guide fraction (no dose-response) as their fitted reference. This
module records that reference explicitly and can re-express effects relative
to non-targeting controls, a named gene or guide, or a supplied numeric value.
Control-based baselines use the median coefficient to limit sensitivity to a
control guide with a genuine phenotype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

#: The baseline a fit already uses: no dose-response. Every guide coefficient
#: is a slope, and zero means the guide's abundance does not move the
#: response.
ZERO = "zero"

#: The non-targeting controls. What a screen paper reports, and what a reader
#: of the figure already believes they are looking at.
CONTROLS = "controls"

#: A named gene or guide, for a screen with a positive control worth
#: normalising to.
NAMED = "named"

#: A user-supplied numeric baseline.
#:
#: Use this when zero comes from another experiment, a published effect, or
#: an assay-specific floor rather than from the fitted data.
VALUE = "value"

#: How the controls are spelled in a screen's `condition` column. Matched
#: case-insensitively. `nc` is spaCR's own spelling; the others are what real
#: annotation files in this project use.
CONTROL_LABELS = ("nc", "negative", "non-targeting", "nontargeting",
                  "control", "neg")


@dataclass
class Baseline:
    """Store a selected baseline and its reporting sentence."""

    kind: str
    #: What to subtract from every effect. 0.0 for :data:`ZERO`.
    shift: float
    #: How many coefficients the shift was estimated from.
    n: int
    #: The sentence for the caption. Never empty.
    sentence: str
    #: Why the request could not be honoured, if it could not.
    reason: Optional[str] = None

    @property
    def moves(self) -> bool:
        """Return whether applying the baseline changes effect values."""
        return bool(self.shift)


def _control_rows(frame, labels: Sequence[str] = CONTROL_LABELS):
    """Return non-targeting-control rows, or ``None`` if no type column exists.

    ``None`` distinguishes an unidentifiable control population from an empty
    population in a table that does provide control annotations.
    """
    for column in ("condition", "control", "guide_type"):
        if column in getattr(frame, "columns", ()):
            values = frame[column].astype("string").str.strip().str.lower()
            wanted = {label.lower() for label in labels}
            return frame.loc[values.isin(wanted)]
    return None


def describe_intercept(frame=None) -> str:
    """Return a one-sentence description of the fitted intercept.

    The intercept is the response at zero guide fraction in the categorical
    plate row and column selected by Patsy as references.
    """
    return ("The intercept is the response at zero guide fraction in the "
            "first row and column of the plate, which patsy drops as the "
            "reference; moving it shifts the intercept and no guide or gene "
            "effect.")


def resolve(frame, kind: str = ZERO, *, column: str = "coefficient",
            name: Optional[str] = None, value=None,
            key_column: str = "feature") -> Baseline:
    """Resolve the baseline used to report effects.

    :param frame: the coefficient table.
    :param kind: :data:`ZERO`, :data:`CONTROLS`, :data:`NAMED` or
        :data:`VALUE`.
    :param column: the effect column.
    :param name: for :data:`NAMED`, the gene or guide to normalise to.
    :param value: for :data:`VALUE`, the number to measure from.
    :returns: a :class:`Baseline`. An unavailable requested baseline returns
        the zero baseline with an explanatory ``reason``.

    :raises ValueError: never. See above.
    """
    if kind == ZERO or not kind:
        return Baseline(ZERO, 0.0, 0,
                        "Effects are slopes on guide fraction, measured from "
                        "zero: no dose-response. " + describe_intercept())

    if column not in getattr(frame, "columns", ()):
        return Baseline(ZERO, 0.0, 0,
                        "Effects are measured from zero (no dose-response).",
                        reason=f"the table has no {column!r} column")

    if kind == CONTROLS:
        rows = _control_rows(frame)
        if rows is None:
            return Baseline(
                ZERO, 0.0, 0,
                "Effects are measured from zero (no dose-response).",
                reason="the table names no condition column, so which rows "
                       "are non-targeting controls is not knowable from it")
        values = rows[column].dropna()
        if len(values) < 2:
            return Baseline(
                ZERO, 0.0, len(values),
                "Effects are measured from zero (no dose-response).",
                reason=f"{len(values)} non-targeting control coefficient(s) "
                       f"is not enough to place a baseline on")
        shift = float(values.median())
        return Baseline(
            CONTROLS, shift, len(values),
            f"Effects are measured from the non-targeting controls "
            f"(median of {len(values)} control coefficients, "
            f"{shift:+.3g}). " + describe_intercept())

    if kind == VALUE:
        try:
            shift = float(value)
        except (TypeError, ValueError):
            return Baseline(ZERO, 0.0, 0,
                            "Effects are measured from zero (no "
                            "dose-response).",
                            reason=f"{value!r} is not a number")
        if shift != shift or shift in (float("inf"), float("-inf")):
            return Baseline(ZERO, 0.0, 0,
                            "Effects are measured from zero (no "
                            "dose-response).",
                            reason=f"{value!r} is not a finite number")
        return Baseline(
            VALUE, shift, 0,
            f"Effects are measured from {shift:+.6g}, a value chosen by "
            f"hand rather than estimated from this screen. "
            + describe_intercept())

    if kind == NAMED:
        if not name:
            return Baseline(ZERO, 0.0, 0,
                            "Effects are measured from zero (no "
                            "dose-response).",
                            reason="no gene or guide was named")
        keys = frame[key_column].astype("string")
        rows = frame.loc[keys.str.contains(str(name), regex=False, na=False)]
        values = rows[column].dropna()
        if not len(values):
            return Baseline(ZERO, 0.0, 0,
                            "Effects are measured from zero (no "
                            "dose-response).",
                            reason=f"nothing in {key_column!r} matches "
                                   f"{name!r}")
        shift = float(values.median())
        return Baseline(
            NAMED, shift, len(values),
            f"Effects are measured from {name} "
            f"(median of {len(values)} coefficient(s), {shift:+.3g}). "
            + describe_intercept())

    return Baseline(ZERO, 0.0, 0,
                    "Effects are measured from zero (no dose-response).",
                    reason=f"unknown baseline {kind!r}; choose one of "
                           f"{ZERO}, {CONTROLS}, {NAMED}, {VALUE}")


def apply(frame, baseline: Baseline, *, column: str = "coefficient"):
    """``frame`` with every effect re-expressed against ``baseline``.

    The function returns a copy because the input may also supply coefficient
    tables, exports and other panels that must retain their original values.

    Standard errors and p-values are unchanged: a
    location shift of every coefficient by one constant changes what each
    effect is measured from, not how precisely it was estimated. The
    p-values still test the coefficient against zero, which is why the
    sentence goes in the caption -- a reader who sees shifted effects and
    unshifted stars must be told the two answer different questions.
    """
    if baseline is None or not baseline.moves:
        return frame
    out = frame.copy()
    if column in out.columns:
        out[column] = out[column] - baseline.shift
    return out


__all__ = ["Baseline", "CONTROLS", "CONTROL_LABELS", "NAMED", "VALUE", "ZERO",
           "apply", "describe_intercept", "resolve"]
