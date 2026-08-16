"""What a finished regression found, in prose, for the console.

Asked for on 2026-08-16: "id also like a little written summary at the end in
the console saying what is significant and so on".

PROSE, NOT A TABLE DUMP. A table of a thousand coefficients is already on
screen and already in a CSV; what is missing is the paragraph a person would
write after looking at it. So this says how many hypotheses were tested, how
many survived and under which rule, which genes they were, whether the assay
worked, and -- because the calibration on real screens is routinely off in
one direction or the other -- whether the test was conservative or
anti-conservative.

IT COMES FROM THE SAME NUMBERS THE PANELS DO. Every figure in
:mod:`spacr.figures.panels` computes its own statistic; a summary that
recomputed them separately would be a second implementation that can disagree
with the pictures beside it, and the reader would have no way to tell which
one was wrong.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .panels import (control_threshold, effect_column, label_series, p_column,
                     q_column, tested)


def _fmt(value, digits=3) -> str:
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def summarise(frame, *, alpha: float = 0.05,
              effect_threshold: Optional[float] = "auto",
              max_named: int = 8) -> str:
    """A paragraph describing what this coefficient table found.

    :param frame: the coefficient table a run returned.
    :param alpha: the level the calls were made at.
    :param effect_threshold: ``'auto'`` to use the control-based cut, a
        number to force one, or ``None`` for significance only.
    :param max_named: how many hits to name before saying "and N more".
    :returns: the summary, ready to print. Empty string when there is
        nothing to summarise -- a caller should say so itself rather than
        print a paragraph about an absence.
    """
    if frame is None or not len(frame):
        return ""
    effect, p = effect_column(frame), p_column(frame)
    if effect is None:
        return ""

    keep = tested(frame)
    sub = frame.loc[keep]
    dropped = int((~keep).sum())
    values = np.asarray(sub[effect], dtype="float64")

    q = q_column(frame)
    if q is not None:
        called = np.asarray(sub[q], dtype="float64") <= alpha
        rule = f"BH q ≤ {alpha:g}"
    elif p is not None:
        called = np.asarray(sub[p], dtype="float64") <= alpha
        rule = f"uncorrected p ≤ {alpha:g} (NO correction was applied)"
    else:
        called = np.zeros(len(sub), bool)
        rule = "no p-value: this backend ranks by selection frequency"
    called = np.nan_to_num(called, nan=False).astype(bool)

    cut_rule, cut = "", None
    if effect_threshold == "auto":
        cut_rule, cut = control_threshold(frame)
    elif effect_threshold:
        cut_rule, cut = "the value you set", float(effect_threshold)
    if cut:
        big = np.abs(values) >= cut
        called_and_big = called & big
    else:
        big = called_and_big = called

    lines = [
        f"{int(keep.sum())} coefficients were tested"
        + (f"; {dropped} nuisance terms (intercept, row and column) are not "
           f"hypotheses and were excluded." if dropped else "."),
    ]

    if cut:
        lines.append(
            f"{int(called.sum())} pass {rule}. Of those, "
            f"{int(called_and_big.sum())} also clear the effect-size cut of "
            f"{_fmt(cut)} ({cut_rule}) — the rest are detectable but smaller "
            f"than the spread of the guides that target nothing.")
    else:
        lines.append(f"{int(called.sum())} pass {rule}.")

    # WHICH ONES. A count without names is not something anyone can act on.
    if called_and_big.any():
        names = label_series(sub)
        order = np.argsort(-np.abs(np.where(called_and_big, values, 0.0)))
        picked = [i for i in order if called_and_big[i]][:max_named]
        listed = ", ".join(
            f"{names.iloc[i]} ({values[i]:+.3g})" for i in picked)
        more = int(called_and_big.sum()) - len(picked)
        lines.append(f"Strongest: {listed}"
                     + (f", and {more} more." if more > 0 else "."))

    # DID THE ASSAY WORK. A screen whose controls do not separate has not
    # measured anything, however many hits the correction reports.
    condition = None
    for name in ("condition", "control", "class"):
        if name in frame.columns:
            condition = name
            break
    if condition is not None:
        names = sub[condition].astype(str).str.lower()
        positive = values[names.isin(("pc", "positive")).to_numpy()]
        negative = values[names.isin(("nc", "control",
                                      "negative")).to_numpy()]
        positive = positive[np.isfinite(positive)]
        negative = negative[np.isfinite(negative)]
        if positive.size and negative.size:
            gap = float(np.median(positive) - np.median(negative))
            spread = float(np.median(np.abs(negative - np.median(negative)))
                           * 1.4826) or float("nan")
            lines.append(
                f"Assay window: the {positive.size} positive control(s) sit "
                f"{_fmt(gap)} from the {negative.size} negative(s), "
                f"{_fmt(abs(gap) / spread)}σ of the negative spread."
                if np.isfinite(spread) and spread else
                f"Assay window: {positive.size} positive and "
                f"{negative.size} negative controls.")
        elif negative.size:
            lines.append(
                f"{negative.size} negative controls and no positive ones, so "
                f"there is nothing to check the assay window against.")

    # CALIBRATION. On a real screen this is routinely off, and which way it
    # is off changes whether the hit count is an over- or an undercount.
    if p is not None:
        raw = np.asarray(sub[p], dtype="float64")
        raw = raw[np.isfinite(raw) & (raw > 0)]
        if raw.size >= 20:
            from scipy.stats import chi2

            lam = float(np.median(chi2.isf(raw, 1)) / chi2.ppf(0.5, 1))
            if lam > 1.15:
                verdict = ("inflated — more small p-values than the null "
                           "predicts, which is either real signal or a "
                           "mis-specified model, so treat the count as an "
                           "upper bound")
            elif lam < 0.85:
                verdict = ("deflated — the test is CONSERVATIVE here, so the "
                           "hits above are more likely an undercount than an "
                           "overcount")
            else:
                verdict = "close to calibrated"
            lines.append(f"Calibration: λ = {lam:.2f}, {verdict}.")

    return " ".join(lines)


__all__ = ["summarise"]
