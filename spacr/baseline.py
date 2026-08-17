"""What an effect size is measured FROM, said out loud and chosen by the user.

Asked for on 2026-08-16: "the user should be able to determine the intercept".

READ LITERALLY, THAT FEATURE DOES NOTHING, and it is worth writing down why
rather than building it. The formula is

    y ~ fraction:grna + gene_fraction:gene + rowID + columnID

and patsy gives EVERY guide and EVERY gene a coefficient -- none is dropped as
a reference, because they appear only in interaction with a continuous term,
so full coding applies. The only levels dropped are the first row and the
first column. Re-fitting a planted effect with the row reference moved from
r01 to r03 shifts the intercept by 0.806 and changes not one guide or gene
coefficient. A control for it would be a control a user could set, watch the
volcano not move, and reasonably conclude was broken.

WHAT THE REQUEST IS ABOUT is the sentence beside it: an effect size with an
unstated reference is not interpretable. The guide coefficients are SLOPES on
the guide's own fraction, referenced to zero -- "no dose-response". A reader
of a screen figure assumes they are looking at differences from the
non-targeting controls, and today nothing on the figure says otherwise.

So this module does the two things that DO change what the reader sees: it
says what the baseline is, and it lets the user move it -- to the controls,
to a named gene, or (2026-08-17) to any number they choose.

THE MEDIAN, NOT THE MEAN. One control guide with a real phenotype drags a
mean baseline and shifts every effect in the screen -- and this screen has
one: `000000_22`, a non-targeting control, is the strongest effect in the
whole run at +4.37.
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

#: A NUMBER THE USER CHOSE. Asked for on 2026-08-17 -- "the user should be
#: able to also set the intercept at what ever they want".
#:
#: The three above are all "let the data tell me where zero is". This one is
#: not, and that is the point of having it: a reader may be normalising to a
#: value from another experiment, a published effect, or an assay-specific
#: floor that this table has no way to know about. It is the only kind whose
#: sentence cannot say where the number came from, so it says that plainly
#: rather than implying the data supports it.
VALUE = "value"

#: How the controls are spelled in a screen's `condition` column. Matched
#: case-insensitively. `nc` is spaCR's own spelling; the others are what real
#: annotation files in this project use.
CONTROL_LABELS = ("nc", "negative", "non-targeting", "nontargeting",
                  "control", "neg")


@dataclass
class Baseline:
    """A chosen baseline and the sentence that states it."""

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
        """True when applying this baseline actually changes the numbers."""
        return bool(self.shift)


def _control_rows(frame, labels: Sequence[str] = CONTROL_LABELS):
    """The rows that are non-targeting controls, or None if unknowable.

    NONE IS NOT AN EMPTY SELECTION. A table with no `condition` column and a
    table whose condition column contains no controls are different failures
    -- the first cannot answer the question, the second answers it with
    "there are none" -- and a caller that treated both as "no controls found"
    would offer a control-based baseline on a table that has no idea.
    """
    for column in ("condition", "control", "guide_type"):
        if column in getattr(frame, "columns", ()):
            values = frame[column].astype("string").str.strip().str.lower()
            wanted = {label.lower() for label in labels}
            return frame.loc[values.isin(wanted)]
    return None


def describe_intercept(frame=None) -> str:
    """What the intercept of this fit is, in one sentence.

    Part of stating the baseline: the number every coefficient is added to is
    the response at zero guide fraction, in whichever row and column patsy
    dropped -- an arbitrary corner of the plate, not a condition anybody
    chose.
    """
    return ("The intercept is the response at zero guide fraction in the "
            "first row and column of the plate, which patsy drops as the "
            "reference; moving it shifts the intercept and no guide or gene "
            "effect.")


def resolve(frame, kind: str = ZERO, *, column: str = "coefficient",
            name: Optional[str] = None, value=None,
            key_column: str = "feature") -> Baseline:
    """The baseline to measure effects from.

    :param frame: the coefficient table.
    :param kind: :data:`ZERO`, :data:`CONTROLS`, :data:`NAMED` or
        :data:`VALUE`.
    :param column: the effect column.
    :param name: for :data:`NAMED`, the gene or guide to normalise to.
    :param value: for :data:`VALUE`, the number to measure from.
    :returns: a :class:`Baseline`, ALWAYS -- a request that cannot be honoured
        comes back as the zero baseline carrying the reason, because a figure
        with no baseline sentence at all is the state this module exists to
        end.

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

    A COPY. The caller is a figure; the table it was handed is the run's own
    results, and shifting it in place would move the numbers under the
    coefficient table, the export and every other panel -- each of which
    would then disagree with its own caption about which baseline it used.

    The standard errors and p-values are NOT touched, and that is correct: a
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
