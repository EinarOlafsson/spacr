"""Diagnostics that say whether a screen regression can be believed.

A volcano plot shows what the model concluded. It cannot show whether the
model was entitled to conclude it. These are the checks that can, and they are
the ones that would have caught the failure this module was written after: a
design with 824 guides in 587 wells, fitted simultaneously, returning a
confident coefficient and P value for every guide from a rank-deficient matrix.

The suite is deliberately split in two:

``design_report`` / ``plot_design_diagnostics``
    Properties of the *design*, computable before any model is fitted, and
    the ones that decide whether the fit means anything: how many wells per
    parameter, the rank of the design matrix, its condition number, how many
    wells each guide appears in, and which guides are so collinear that no
    method can separate them.

``residual_report`` / ``plot_residual_diagnostics``
    Properties of a *fitted* model: the residual-versus-fitted, scale-location,
    QQ and leverage panels, plus Cook's distance.

``plot_inference_diagnostics``
    Properties of the *test*: the P-value histogram, whose shape tells you
    whether the null is calibrated, and the observed-versus-expected quantile
    plot with its genomic inflation factor.

Every function takes plain arrays or frames and returns plain numbers, so they
can be asserted on in tests rather than eyeballed.
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

# THE HOUSE PALETTE, BY ROLE. This module drew in seaborn's `deep` -- nine
# hardcoded hexes with no rule behind which one meant what -- so a run wrote
# its design and inference panels in a third visual idiom beside the seven
# house-style panels and the nineteen QC ones.
#
# Mapped rather than renamed: `#4C72B0` was doing the job of "the data" in one
# panel and "the highlight" in another, so a find-and-replace would have kept
# the inconsistency and only changed the hues. See
# `.claude/skills/apicomplexan-figures`: everything is grey except what the
# sentence is about.
from .figures.style import ROLES, TYPE_SCALE, WEIGHTS, figure_style

#: What each old hex was actually being used FOR, decided per call site.
_DATA = ROLES["data"]              # bars, clouds, anything not the claim
_BAD = ROLES["down"]               # a failed check, a threshold crossed
_GOOD = ROLES["up"]                # a passed check
_MARK = ROLES["highlight"]         # the one series a panel is about
_REFERENCE = ROLES["reference"]    # thresholds and guides

__all__ = [
    "design_report",
    "score_design",
    "score_residuals",
    "score_inference",
    "residual_report",
    "collinear_guide_pairs",
    "variance_inflation_factors",
    "plot_design_diagnostics",
    "plot_residual_diagnostics",
    "plot_inference_diagnostics",
    "write_diagnostic_suite",
]


# --------------------------------------------------------------------- design


def design_report(fractions: pd.DataFrame, *, block: pd.Series | None = None,
                  presence_threshold: float = 0.0) -> dict:
    """Measure whether a simultaneous fit on this design is identifiable.

    :param fractions: well-by-guide matrix, one row per analysed well.
    :param block: optional per-well block labels (normally the plate), which
        cost one parameter each beyond the first.
    :param presence_threshold: a guide counts as present in a well when its
        value exceeds this.
    :returns: a dict of scalars. ``identifiable`` is the one that matters: it
        is False when the design has fewer wells than parameters, which is the
        state in which per-guide coefficients are not unique.
    """
    matrix = np.asarray(fractions, dtype=float)
    n_wells, n_guides = matrix.shape
    blocks = 0
    if block is not None:
        blocks = max(int(pd.Series(block).nunique()) - 1, 0)
    parameters = 1 + blocks + n_guides

    design = np.column_stack([np.ones((n_wells, 1)), matrix])
    rank = int(np.linalg.matrix_rank(design)) if n_wells and n_guides else 0
    residual_df = n_wells - rank
    with np.errstate(divide="ignore", invalid="ignore"):
        singular = np.linalg.svd(design, compute_uv=False) if design.size else np.array([0.0])
        positive = singular[singular > 0]
        condition = float(positive[0] / positive[-1]) if positive.size else np.inf

    support = (matrix > float(presence_threshold)).sum(axis=0)
    return {
        "wells": int(n_wells),
        "guides": int(n_guides),
        "block_terms": int(blocks),
        "parameters": int(parameters),
        "design_rank": rank,
        "residual_degrees_of_freedom": int(residual_df),
        "non_identifiable_directions": int(max(parameters - rank, 0)),
        "condition_number": condition,
        "wells_per_parameter": (
            float(n_wells / parameters) if parameters else float("nan")),
        # The single verdict. Rank deficiency is not a warning: a coefficient
        # in a rank-deficient fit is one of infinitely many solutions.
        "identifiable": bool(rank >= parameters and residual_df > 0),
        "guide_support_min": int(support.min()) if support.size else 0,
        "guide_support_median": float(np.median(support)) if support.size else 0.0,
        "guide_support_max": int(support.max()) if support.size else 0,
        "guides_in_one_well": int((support <= 1).sum()),
        "mean_guides_per_well": float(
            (matrix > float(presence_threshold)).sum(axis=1).mean())
        if n_wells else 0.0,
    }


def collinear_guide_pairs(fractions: pd.DataFrame, *,
                          threshold: float = 0.95,
                          limit: int = 500) -> pd.DataFrame:
    """Guide pairs whose well patterns are so alike they cannot be separated.

    This is the mechanism behind the screen's false positives: a guide that
    appears in nearly the same wells as a true hit inherits its signal, and no
    amount of correction distinguishes them, because the data contain no
    contrast between them. Reported as a table so the offenders can be named.

    :param threshold: absolute Pearson correlation at or above which a pair is
        listed.
    :param limit: stop after this many pairs; a wide screen has millions and
        the first few hundred are the ones worth reading.
    """
    frame = fractions.loc[:, fractions.std(axis=0) > 0]
    if frame.shape[1] < 2:
        return pd.DataFrame(columns=["guide_a", "guide_b", "correlation",
                                     "shared_wells"])
    correlation = np.corrcoef(frame.to_numpy(dtype=float), rowvar=False)
    names = list(frame.columns)
    presence = (frame.to_numpy(dtype=float) > 0)
    rows = []
    upper = np.triu_indices_from(correlation, k=1)
    for i, j in zip(*upper):
        value = correlation[i, j]
        if not np.isfinite(value) or abs(value) < threshold:
            continue
        rows.append({
            "guide_a": names[i],
            "guide_b": names[j],
            "correlation": float(value),
            "shared_wells": int(np.sum(presence[:, i] & presence[:, j])),
        })
        if len(rows) >= limit:
            break
    columns = ["guide_a", "guide_b", "correlation", "shared_wells"]
    if not rows:
        # A well-designed screen legitimately has no collinear pair. Building
        # the frame from an empty list gives it no columns at all, so sorting
        # raised KeyError('correlation') on exactly the healthy case.
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        "correlation", ascending=False, key=abs, kind="stable"
    ).reset_index(drop=True)


def variance_inflation_factors(fractions: pd.DataFrame, *,
                               max_guides: int = 200) -> pd.DataFrame:
    """Per-guide VIF: how much collinearity inflates that guide's variance.

    VIF is only defined when the design is full rank, so this refuses rather
    than returning infinities for a design that cannot support the simultaneous
    model at all -- in that case :func:`design_report` is the answer.

    ONE VIF, COMPUTED ONCE. finding 4: this package computed
    variance inflation factors in two modules, and "the duplicated VIF is a
    small correctness risk of the same shape as finding 2". The numbers came
    from :func:`spacr.regression_qc.variance_inflation_factors` now; what stays
    here is the SCREEN's contract around them, which is a different thing from
    the statistic and is why the two functions both exist.

    MEASURED BEFORE CONSOLIDATING, on 200x20, 400x60 and 96x30 designs each
    carrying a near-collinear pair: the two implementations agreed to 2.3e-10
    relative, and named the same columns ``inf`` under exact collinearity. So
    unlike finding 2 there was no statistical disagreement to resolve -- which
    is exactly why the duplication was worth removing rather than arbitrating.
    Two routes to one number that agree today are two routes that can stop
    agreeing, and nothing was comparing them.

    THE SURVIVOR IS THE CORRELATION ROUTE, on cost. The auxiliary-regression
    form this used ran one least-squares fit per guide, ``O(p^4)``; the
    identity ``VIF_j = (R^-1)_jj`` is one ``O(p^3)`` decomposition. Timed here:
    130.8 ms against 13.0 ms at 60 guides, and the gap widens with the fourth
    power. A pooled screen has hundreds.

    :param max_guides: how many guides to REPORT, widest support first. It was
        never a limit on the computation -- each guide's VIF was always taken
        against every other guide -- and it still is not.
    """
    frame = fractions.loc[:, fractions.std(axis=0) > 0]
    n_wells, n_guides = frame.shape
    if n_guides == 0:
        return pd.DataFrame(columns=["guide", "vif"])
    if n_wells <= n_guides:
        # The engine would answer `inf` for every guide here, which is true and
        # useless. A rank-deficient design has no VIF to report, and saying so
        # is what sends the caller to the panel that can describe it.
        raise ValueError(
            f"Variance inflation factors need more wells ({n_wells}) than "
            f"guides ({n_guides}); this design is rank deficient, so use "
            f"design_report() and collinear_guide_pairs() instead.")

    from .regression_qc import variance_inflation_factors as _engine

    values = _engine(frame)
    support = (frame > 0).sum(axis=0).sort_values(ascending=False)
    selected = list(support.index[:max_guides])
    rows = [{"guide": name, "vif": float(values[name]),
             "wells_with_guide": int(support[name])}
            for name in selected]
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(
        drop=True)


# ------------------------------------------------------------------ residuals


def residual_report(observed, fitted, *, design: np.ndarray | None = None) -> dict:
    """Summarise the residuals of a fitted model.

    :param design: the model matrix. When given, leverage and Cook's distance
        are computed from its hat matrix; without it those keys are omitted
        rather than guessed.
    """
    y = np.asarray(observed, dtype=float)
    yhat = np.asarray(fitted, dtype=float)
    residual = y - yhat
    n = residual.size
    total = float(np.sum((y - y.mean()) ** 2))
    sse = float(np.sum(residual ** 2))
    report = {
        "n": int(n),
        "residual_mean": float(residual.mean()) if n else float("nan"),
        "residual_sd": float(residual.std(ddof=1)) if n > 1 else float("nan"),
        "sse": sse,
        "r_squared": float(1.0 - sse / total) if total > 0 else float("nan"),
    }
    if n > 2:
        from scipy import stats
        # Shapiro-Wilk is exact but degrades above a few thousand points, where
        # D'Agostino's K^2 is the right test -- the same rule the manuscript's
        # own statistics section uses.
        if n < 5000:
            statistic, p_value = stats.shapiro(residual)
            report["normality_test"] = "shapiro"
        else:
            statistic, p_value = stats.normaltest(residual)
            report["normality_test"] = "dagostino_pearson"
        report["normality_statistic"] = float(statistic)
        report["normality_p_value"] = float(p_value)
        report["skew"] = float(stats.skew(residual))
        report["kurtosis"] = float(stats.kurtosis(residual))
        # Breusch-Pagan against the fitted values: is the spread constant?
        if np.std(yhat) > 0:
            slope_design = np.column_stack([np.ones(n), yhat])
            squared = residual ** 2
            coefficients, *_ = np.linalg.lstsq(slope_design, squared, rcond=None)
            explained = slope_design @ coefficients
            ss_total = float(np.sum((squared - squared.mean()) ** 2))
            ss_res = float(np.sum((squared - explained) ** 2))
            r2 = 1.0 - ss_res / ss_total if ss_total > 0 else 0.0
            statistic = n * r2
            report["heteroscedasticity_statistic"] = float(statistic)
            report["heteroscedasticity_p_value"] = float(
                stats.chi2.sf(statistic, df=1))
    if design is not None:
        matrix = np.asarray(design, dtype=float)
        pinv = np.linalg.pinv(matrix)
        leverage = np.einsum("ij,ji->i", matrix, pinv)
        rank = int(np.linalg.matrix_rank(matrix))
        report["max_leverage"] = float(np.max(leverage)) if leverage.size else float("nan")
        report["mean_leverage"] = float(np.mean(leverage)) if leverage.size else float("nan")
        residual_df = max(n - rank, 1)
        mse = sse / residual_df
        with np.errstate(divide="ignore", invalid="ignore"):
            cooks = (residual ** 2 / (rank * mse)) * (
                leverage / (1.0 - leverage) ** 2)
        cooks = np.nan_to_num(cooks, nan=0.0, posinf=np.inf)
        report["max_cooks_distance"] = float(np.max(cooks)) if cooks.size else float("nan")
        # 4/n is the usual screening rule for "look at this point".
        report["high_influence_points"] = int(np.sum(cooks > 4.0 / max(n, 1)))
    return report


# ------------------------------------------------------------------- verdicts
#
# EVERY DIAGNOSTIC REACHES A VERDICT, INCLUDING THESE THREE.
#
# `spacr.regression_qc` scores each of its twenty-three panels and stamps the
# judgement on the panel, so a reader is told whether a number is fine instead
# of being expected to know. These three sheets did not, and they are the ones
# a permutation run gets INSTEAD of that suite -- so on the analysis mode where
# there is no fitted design matrix, nothing in the whole output said whether
# the design was usable. That is the reading this section exists to prevent.
#
# THE VOCABULARY IS BORROWED, NOT REBUILT. `PanelVerdict`, the four levels and
# the badge come from `regression_qc`; a second set of words for the same four
# states is how a run comes to say CHECK on one page and WARN on another about
# the same fit.
#
# A SHEET GETS ONE VERDICT because a sheet answers one question -- is this
# design usable, are these residuals behaved, is this null calibrated -- and
# its panels are the evidence for that one answer.


def _verdict(level, headline, detail="", score=None, statistic=""):
    """A :class:`spacr.regression_qc.PanelVerdict`, imported lazily.

    Lazily because `regression_qc` imports statsmodels-shaped things and this
    module is deliberately importable for its plain-array functions alone.
    """
    from .regression_qc import PanelVerdict

    return PanelVerdict(level=level, headline=headline, detail=detail,
                        score=score, statistic=statistic)


def _worst(verdicts):
    """The verdict a summary should report: the worst one."""
    from .regression_qc import worst_verdict

    return worst_verdict([v for v in verdicts if v is not None])


def score_design(report: Mapping) -> "object":
    """Is a simultaneous fit on this design entitled to its coefficients?

    :param report: the dict :func:`design_report` returns.
    :returns: a ``PanelVerdict``.

    RANK DEFICIENCY IS THE ONE FAIL AND IT IS NOT A MATTER OF DEGREE. A design
    with fewer independent directions than parameters returns a coefficient per
    guide that is ONE OF INFINITELY MANY solutions, and statsmodels answers
    with a pseudo-inverse rather than refusing -- so the numbers look like any
    other fit. Nothing else in the output says so.

    The other two rules are the conventional ones and are cited in ``detail``:
    a design needs more wells than parameters to have any residual degrees of
    freedom at all, and a guide seen in one well or none has no contrast to be
    estimated from.
    """
    if not report or not report.get("wells"):
        return _verdict("unknown", "no design was supplied")
    per_parameter = float(report.get("wells_per_parameter") or 0.0)
    alone = int(report.get("guides_in_one_well") or 0)
    condition = float(report.get("condition_number") or float("nan"))
    evidence = (f"{report.get('wells')} wells, {report.get('parameters')} "
                f"parameters, rank {report.get('design_rank')}, "
                f"condition number {condition:.3g}")
    if not report.get("identifiable"):
        return _verdict(
            "fail", "the design cannot identify one coefficient per guide",
            "The rank is below the number of parameters, so every "
            "per-guide coefficient is one of infinitely many solutions "
            f"({evidence}). Use the permutation test.",
            score=per_parameter, statistic="wells per parameter")
    if per_parameter < 2.0:
        return _verdict(
            "check", "there is very little data per parameter",
            f"{per_parameter:.2f} wells per parameter ({evidence}). The fit "
            "is identifiable but every coefficient rests on a handful of "
            "wells.", score=per_parameter, statistic="wells per parameter")
    if alone:
        return _verdict(
            "check", f"{alone} guide(s) appear in one well or none",
            "A guide seen in a single well has no contrast to be estimated "
            f"from; its coefficient is that well ({evidence}).",
            score=float(alone), statistic="guides in <=1 well")
    return _verdict("pass", "the design supports a simultaneous fit",
                    f"Identifiable, {per_parameter:.1f} wells per parameter "
                    f"({evidence}).", score=per_parameter,
                    statistic="wells per parameter")


def score_residuals(report: Mapping) -> "object":
    """Are these residuals behaved enough for the inference drawn from them?

    :param report: the dict :func:`residual_report` returns.

    A DIAGNOSTIC TEST'S p IS BACKWARDS AND THAT IS THE COMMONEST WAY TO READ
    ONE WRONG: the null is "the assumption holds", so a LARGE p is the good
    outcome. Every rule below is written in that direction and says so.

    Thresholds are the conventional ones: p >= 0.05 for a diagnostic test,
    Cook's distance of 0.5 and 1.0. A threshold invented here would be a number
    a reviewer cannot check.
    """
    if not report or not report.get("n"):
        return _verdict("unknown", "no residuals were supplied")
    findings = []
    cooks = report.get("max_cooks_distance")
    if cooks is not None and np.isfinite(cooks):
        if cooks > 1.0:
            findings.append(_verdict(
                "fail", "one observation dominates the fit",
                f"The largest Cook's distance is {cooks:.2f}; above 1.0 is the "
                "conventional line for a single point that moves the "
                "coefficients on its own.", cooks, "max Cook's D"))
        elif cooks > 0.5:
            findings.append(_verdict(
                "check", "one observation has a lot of influence",
                f"The largest Cook's distance is {cooks:.2f}; 0.5 is the "
                "conventional line for a point worth looking at.",
                cooks, "max Cook's D"))
    spread_p = report.get("heteroscedasticity_p_value")
    if spread_p is not None and np.isfinite(spread_p):
        if spread_p < 0.01:
            findings.append(_verdict(
                "fail", "the residual spread changes with the fitted value",
                f"Breusch-Pagan p = {spread_p:.3g}. The null is constant "
                "variance, so a SMALL p is the bad outcome, and the standard "
                "errors this fit reports are not the right ones.",
                spread_p, "Breusch-Pagan p"))
        elif spread_p < 0.05:
            findings.append(_verdict(
                "check", "the residual spread may change with the fit",
                f"Breusch-Pagan p = {spread_p:.3g}, under the conventional "
                "0.05. The null is constant variance, so a small p is the bad "
                "outcome.", spread_p, "Breusch-Pagan p"))
    normal_p = report.get("normality_p_value")
    if normal_p is not None and np.isfinite(normal_p) and normal_p < 0.05:
        findings.append(_verdict(
            "check", "the residuals are not normal",
            f"{report.get('normality_test', 'normality')} p = {normal_p:.3g}. "
            "The null is normality, so a small p is the bad outcome; with "
            "enough wells this is common and matters most in the tails.",
            normal_p, "normality p"))
    if findings:
        return _worst(findings)
    return _verdict("pass", "the residuals are well behaved",
                    f"n = {report['n']}, no diagnostic test under 0.05 and no "
                    "single observation dominating the fit.",
                    report.get("r_squared"), "r-squared")


def score_inference(report: Mapping) -> "object":
    """Is the null calibrated, or is the whole family shifted?

    Read off the genomic inflation factor: the median observed chi-square over
    its null median, which is 1.0 when the null behaves. The bands are the ones
    the GWAS literature uses -- 1.1 is where a report starts explaining itself
    and 1.2 is where the p-values are not believed.

    A SPIKE OF SMALL p IS NOT A FAULT. It is what a screen with real hits looks
    like, and lambda is deliberately a MEDIAN so a handful of true hits barely
    move it. Scoring the spike itself would flag every successful screen.
    """
    if not report or not report.get("tests"):
        return _verdict("unknown", "no p-values were supplied")
    inflation = report.get("genomic_inflation")
    if inflation is None or not np.isfinite(inflation):
        return _verdict("unknown", "the inflation factor could not be computed")
    distance = abs(inflation - 1.0)
    detail = (f"lambda = {inflation:.3f} over {report['tests']} tests; "
              f"pi0 = {report.get('pi0', float('nan')):.2f}, so about "
              f"{report.get('estimated_non_null', 0):.0f} are estimated "
              "non-null.")
    if distance > 0.2:
        direction = "inflated" if inflation > 1 else "deflated"
        return _verdict("fail", f"the null is {direction}", detail,
                        inflation, "genomic inflation")
    if distance > 0.1:
        return _verdict("check", "the null is a little off centre", detail,
                        inflation, "genomic inflation")
    return _verdict("pass", "the null is calibrated", detail, inflation,
                    "genomic inflation")


def _stamp(axis, verdict) -> None:
    """Put a sheet's verdict on its first panel, in the suite's own badge."""
    try:
        from .regression_qc import draw_verdict

        draw_verdict(axis, verdict)
    except Exception:                       # pragma: no cover - advisory only
        pass


# ------------------------------------------------------------------- plotting


def _house(axis, title="", xlabel="", ylabel=""):
    """Put one axis into the house style.

    The figures here are built by `plt.subplots` OUTSIDE a style context in
    some callers, and rcParams only reach an artist when it is CREATED -- so
    the ink, the type sizes and the spines are set on the axis by hand rather
    than trusted to the context. `grid(False)` is explicit for the same
    reason: the rule is no gridlines ever, and a caller with a grid-on global
    style would otherwise put one here.
    """
    ink = ROLES["reference"]
    try:
        from .figures.style import resolve_ink, theme_target

        ink = resolve_ink(theme_target())
    except Exception:                       # pragma: no cover - style absent
        pass
    if title:
        axis.set_title(title, fontsize=TYPE_SCALE["label"], color=ink, pad=3.0)
    if xlabel:
        axis.set_xlabel(xlabel, fontsize=TYPE_SCALE["label"], color=ink)
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=TYPE_SCALE["label"], color=ink)
    axis.tick_params(color=ink, labelcolor=ink,
                     labelsize=TYPE_SCALE["tick"], which="both")
    for spine in axis.spines.values():
        spine.set_edgecolor(ink)
        spine.set_linewidth(WEIGHTS["spine"])
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(False, which="both")
    return ink


def _finish(fig, save_path, dpi=None, fmt=None, renderer=None, title=None):
    """Lay the figure out, PUBLISH it when there is somewhere to write it, close it.

    Published, not saved. Instruction 139 C: "several graphs are saved but I
    cannot see them in the software". This module wrote its panels with a bare
    ``fig.savefig`` and never called ``show``, and a figure reaches the spaCR
    gallery by exactly one route -- so the design, residual and inference
    panels were on disk and none of them was in the application, which is the
    reported bug precisely. :func:`spacr.figure_sink.publish` saves and
    announces as ONE event, so the file and the tile can no longer disagree.

    Two things the raw ``savefig`` also got wrong and ``publish`` fixes, both
    through :func:`spacr.plot.save_figure`:

    * the FORMAT is the user's preference rather than whatever extension the
      caller happened to type, and the file NAME follows the format -- a PNG
      written to ``design.pdf`` is a file no viewer opens;
    * the DPI is the preference's, and it reaches a PDF as well, which matters
      because these panels are full of scatter clouds.

    ``fmt`` forces one particular format, for a caller that genuinely needs
    one (``write_diagnostic_suite`` when it was asked for named formats), and
    ``dpi`` forces a resolution. Both default to None, which is what lets the
    preference through; the old hard-coded 200 applied to PNG only and was the
    reason the resolution preference reached none of these panels.

    THE RENDERER IS THE SCREEN'S WHERE THAT IS POSSIBLE.
    :func:`spacr.figures.scene.write_figure` translates the finished figure --
    all six panels of it -- into a pyqtgraph scene and exports that, so the
    generated file is drawn by the library the tabs are drawn by. The numbers
    are still computed exactly once, here. A figure it cannot carry completely
    is published by matplotlib exactly as it was.

    The figure is closed on BOTH paths. These figures come from ``plt.subplots``,
    so pyplot holds a reference to every one of them until it is closed;
    returning early on ``save_path=None`` -- the default of all three public
    ``plot_*_diagnostics`` functions, and the way a caller asks for the report
    dict without keeping the picture -- leaked one figure per call. Across a
    parameter sweep that is hundreds of live figures and matplotlib's
    "More than 20 figures have been opened" warning.

    :returns: the path actually written, which is not necessarily the path
        asked for. A caller that records the requested name records a file
        that is not there.
    """
    import matplotlib.pyplot as plt

    from .figures.scene import write_figure

    fig.tight_layout()
    try:
        if save_path is None:
            return None
        path = os.fspath(save_path)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        written, _drew, _why = write_figure(fig, path, fmt=fmt, dpi=dpi,
                                            renderer=renderer, title=title,
                                            bbox_inches="tight")
        return written or path
    finally:
        plt.close(fig)


def plot_design_diagnostics(fractions: pd.DataFrame, *,
                            block: pd.Series | None = None,
                            save_path=None, save_format=None,
                            presence_threshold: float = 0.0):
    """Six panels describing the design, before any model is fitted."""
    import matplotlib.pyplot as plt

    report = design_report(fractions, block=block,
                           presence_threshold=presence_threshold)
    matrix = np.asarray(fractions, dtype=float)
    presence = matrix > float(presence_threshold)
    support = presence.sum(axis=0)
    per_well = presence.sum(axis=1)

    # THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS.
    # rcParams colour an artist when it is CREATED, so a
    # context opened after plt.subplots would leave the
    # spines, ticks and text at whatever the caller's global
    # style happened to be.
    with figure_style():
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))

        axis = axes[0, 0]
        axis.hist(support, bins=min(40, max(int(support.max()), 1)),
                  color=_DATA, edgecolor="white")
        axis.set_xlabel("Wells containing the guide")
        axis.set_ylabel("Guides")
        axis.set_title("Guide support")
        axis.axvline(1.5, color=_BAD, linestyle="--", linewidth=1)
        axis.text(0.98, 0.95, f"{report['guides_in_one_well']} guides in ≤1 well",
                  transform=axis.transAxes, ha="right", va="top", fontsize=8,
                  color=_BAD)

        axis = axes[0, 1]
        axis.hist(per_well, bins=min(30, max(int(per_well.max()), 1)),
                  color=_GOOD, edgecolor="white")
        axis.set_xlabel("Guides retained in the well")
        axis.set_ylabel("Wells")
        axis.set_title("Guides per well")
        axis.axvline(float(np.mean(per_well)), color=_REFERENCE, linestyle="--",
                     linewidth=1)

        # Identifiability, stated rather than implied.
        axis = axes[0, 2]
        axis.axis("off")
        verdict = "IDENTIFIABLE" if report["identifiable"] else "NOT IDENTIFIABLE"
        colour = _GOOD if report["identifiable"] else _BAD
        lines = [
            f"{report['wells']} wells × {report['guides']} guides",
            f"{report['parameters']} parameters (incl. {report['block_terms']} block terms)",
            f"design rank {report['design_rank']}",
            f"{report['residual_degrees_of_freedom']} residual df",
            f"{report['non_identifiable_directions']} non-identifiable directions",
            f"condition number {report['condition_number']:.3g}",
            f"{report['wells_per_parameter']:.2f} wells per parameter",
        ]
        axis.text(0.5, 0.92, verdict, transform=axis.transAxes, ha="center",
                  va="top", fontsize=14, fontweight="bold", color=colour)
        axis.text(0.5, 0.72, "\n".join(lines), transform=axis.transAxes,
                  ha="center", va="top", fontsize=9, family="monospace")
        if not report["identifiable"]:
            axis.text(0.5, 0.1,
                      "A simultaneous fit cannot return a unique\n"
                      "coefficient per guide on this design.\n"
                      "Use the permutation test.",
                      transform=axis.transAxes, ha="center", va="bottom",
                      fontsize=8, color=_BAD)

        # Cumulative singular-value spectrum: where the rank runs out.
        axis = axes[1, 0]
        design = np.column_stack([np.ones((matrix.shape[0], 1)), matrix])
        singular = np.linalg.svd(design, compute_uv=False)
        axis.semilogy(np.arange(1, singular.size + 1),
                      np.maximum(singular, np.finfo(float).tiny),
                      color=_MARK)
        axis.axvline(report["design_rank"], color=_BAD, linestyle="--",
                     linewidth=1, label=f"rank {report['design_rank']}")
        axis.set_xlabel("Component")
        axis.set_ylabel("Singular value (log)")
        axis.set_title("Design spectrum")
        axis.legend(frameon=False, fontsize=8)

        axis = axes[1, 1]
        pairs = collinear_guide_pairs(fractions, threshold=0.5, limit=20000)
        if pairs.empty:
            axis.text(0.5, 0.5, "No guide pair correlates above 0.5",
                      transform=axis.transAxes, ha="center", va="center",
                      fontsize=9)
            axis.set_axis_off()
        else:
            axis.hist(pairs["correlation"].abs(), bins=30, color=_DATA,
                      edgecolor="white")
            severe = int((pairs["correlation"].abs() >= 0.95).sum())
            axis.set_xlabel("|correlation| between guide well patterns")
            axis.set_ylabel("Guide pairs")
            axis.set_title("Guide co-occurrence")
            axis.text(0.98, 0.95, f"{severe} pairs ≥ 0.95",
                      transform=axis.transAxes, ha="right", va="top",
                      fontsize=8, color=_BAD)

        # Occupancy map: which wells hold which guides, sorted so structure shows.
        axis = axes[1, 2]
        order = np.argsort(-support)
        shown = presence[:, order[:min(200, presence.shape[1])]]
        axis.imshow(shown, aspect="auto", cmap="Greys", interpolation="nearest")
        axis.set_xlabel(f"Guide (top {shown.shape[1]} by support)")
        axis.set_ylabel("Well")
        axis.set_title("Occupancy")

        fig.suptitle("Screen design diagnostics", fontsize=13, fontweight="bold")
        # SCORED AFTER IT DREW AND STAMPED BEFORE IT IS WRITTEN, exactly as
        # the QC suite does it: the verdict is read off the report the panels
        # were drawn from, so it cannot disagree with the numbers printed
        # beside it.
        verdict = score_design(report)
        _stamp(axes[0, 0], verdict)
        report = dict(report, verdict=verdict.headline,
                      verdict_level=verdict.level, verdict_detail=verdict.detail)
        return _finish(fig, save_path, fmt=save_format,
                       title="screen design diagnostics"), report


def plot_residual_diagnostics(observed, fitted, *,
                              design: np.ndarray | None = None,
                              save_path=None, save_format=None,
                              label: str = ""):
    """The four classical residual panels, plus Cook's distance."""
    import matplotlib.pyplot as plt
    from scipy import stats

    y = np.asarray(observed, dtype=float)
    yhat = np.asarray(fitted, dtype=float)
    residual = y - yhat
    scale = residual.std(ddof=1) if residual.size > 1 else 1.0
    standardized = residual / scale if scale > 0 else residual

    panels = 6 if design is not None else 4
    rows = 2
    columns = 3 if panels == 6 else 2
    # THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS.
    # rcParams colour an artist when it is CREATED, so a
    # context opened after plt.subplots would leave the
    # spines, ticks and text at whatever the caller's global
    # style happened to be.
    with figure_style():
        fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 8))
        flat = axes.ravel()

        axis = flat[0]
        axis.scatter(yhat, residual, s=14, alpha=0.6, color=_DATA,
                     edgecolor="none")
        axis.axhline(0, color=_BAD, linestyle="--", linewidth=1)
        if yhat.size > 10 and np.std(yhat) > 0:
            order = np.argsort(yhat)
            window = max(int(len(yhat) * 0.2), 3)
            smooth = pd.Series(residual[order]).rolling(
                window, center=True, min_periods=1).mean()
            axis.plot(yhat[order], smooth, color=_MARK, linewidth=1.4)
        axis.set_xlabel("Fitted")
        axis.set_ylabel("Residual")
        axis.set_title("Residuals vs fitted")

        axis = flat[1]
        stats.probplot(standardized, dist="norm", plot=axis)
        axis.get_lines()[0].set(markersize=3, alpha=0.6, color=_DATA)
        axis.get_lines()[1].set(color=_BAD, linewidth=1)
        axis.set_title("Normal Q-Q")

        axis = flat[2]
        axis.scatter(yhat, np.sqrt(np.abs(standardized)), s=14, alpha=0.6,
                     color=_GOOD, edgecolor="none")
        axis.set_xlabel("Fitted")
        axis.set_ylabel("√|standardized residual|")
        axis.set_title("Scale-location")

        axis = flat[3]
        axis.hist(residual, bins=min(40, max(int(np.sqrt(residual.size)), 5)),
                  color=_MARK, edgecolor="white", density=True)
        if scale > 0:
            grid = np.linspace(residual.min(), residual.max(), 200)
            axis.plot(grid, stats.norm.pdf(grid, residual.mean(), scale),
                      color=_BAD, linewidth=1.2)
        axis.set_xlabel("Residual")
        axis.set_ylabel("Density")
        axis.set_title("Residual distribution")

        if design is not None:
            matrix = np.asarray(design, dtype=float)
            pinv = np.linalg.pinv(matrix)
            leverage = np.einsum("ij,ji->i", matrix, pinv)
            rank = int(np.linalg.matrix_rank(matrix))
            residual_df = max(len(y) - rank, 1)
            mse = float(np.sum(residual ** 2)) / residual_df
            with np.errstate(divide="ignore", invalid="ignore"):
                cooks = (residual ** 2 / (rank * mse)) * (
                    leverage / (1.0 - leverage) ** 2)
            cooks = np.nan_to_num(cooks, nan=0.0)

            axis = flat[4]
            axis.scatter(leverage, standardized, s=14, alpha=0.6,
                         color=_DATA, edgecolor="none")
            axis.axhline(0, color=_REFERENCE, linewidth=0.8)
            axis.axvline(2.0 * rank / max(len(y), 1), color=_BAD,
                         linestyle="--", linewidth=1, label="2p/n")
            axis.set_xlabel("Leverage")
            axis.set_ylabel("Standardized residual")
            axis.set_title("Residuals vs leverage")
            axis.legend(frameon=False, fontsize=8)

            axis = flat[5]
            axis.stem(np.arange(len(cooks)), cooks, markerfmt=" ", basefmt=" ",
                      linefmt=_DATA)
            cutoff = 4.0 / max(len(cooks), 1)
            axis.axhline(cutoff, color=_BAD, linestyle="--", linewidth=1,
                         label="4/n")
            axis.set_xlabel("Observation")
            axis.set_ylabel("Cook's distance")
            axis.set_title("Influence")
            axis.legend(frameon=False, fontsize=8)

        title = "Residual diagnostics"
        if label:
            title = f"{title} — {label}"
        fig.suptitle(title, fontsize=13, fontweight="bold")
        report = residual_report(y, yhat, design=design)
        verdict = score_residuals(report)
        _stamp(flat[0], verdict)
        report = dict(report, verdict=verdict.headline,
                      verdict_level=verdict.level, verdict_detail=verdict.detail)
        return _finish(fig, save_path, fmt=save_format, title=title), report


def plot_inference_diagnostics(p_values, *, adjusted=None, alpha: float = 0.05,
                               save_path=None, save_format=None,
                               label: str = ""):
    """Is the null calibrated, and how much of the family is non-null?

    A well-behaved screen gives a flat P-value histogram with a spike at zero.
    A histogram that slopes or humps in the middle means the test is
    mis-calibrated, and no correction repairs that.
    """
    import matplotlib.pyplot as plt

    from .multiple_testing import estimate_pi0

    values = np.asarray(p_values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size

    # THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS.
    # rcParams colour an artist when it is CREATED, so a
    # context opened after plt.subplots would leave the
    # spines, ticks and text at whatever the caller's global
    # style happened to be.
    with figure_style():
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

        axis = axes[0]
        axis.hist(values, bins=20, range=(0, 1), color=_DATA,
                  edgecolor="white")
        axis.axhline(n / 20.0, color=_BAD, linestyle="--", linewidth=1,
                     label="uniform null")
        axis.set_xlabel("P value")
        axis.set_ylabel("Tests")
        axis.set_title("P-value histogram")
        axis.legend(frameon=False, fontsize=8)
        pi0 = estimate_pi0(values)
        axis.text(0.98, 0.80, f"π₀ ≈ {pi0:.2f}\n({(1 - pi0) * 100:.0f}% non-null)",
                  transform=axis.transAxes, ha="right", va="top", fontsize=8)

        # QQ against the uniform null, on -log10 so the tail is readable.
        axis = axes[1]
        observed = -np.log10(np.sort(np.clip(values, np.finfo(float).tiny, 1.0)))
        expected = -np.log10((np.arange(1, n + 1) - 0.5) / n)
        axis.scatter(expected, observed, s=12, alpha=0.7, color=_GOOD,
                     edgecolor="none")
        limit = float(max(expected.max(), observed.max())) if n else 1.0
        axis.plot([0, limit], [0, limit], color=_BAD, linestyle="--",
                  linewidth=1)
        axis.set_xlabel("Expected −log₁₀(P)")
        axis.set_ylabel("Observed −log₁₀(P)")
        axis.set_title("P-value Q-Q")
        # Genomic inflation: median observed chi-square over its null median.
        inflation = float("nan")
        if n:
            from scipy import stats
            chi2 = stats.chi2.isf(np.clip(values, np.finfo(float).tiny, 1.0), df=1)
            inflation = float(np.median(chi2) / stats.chi2.ppf(0.5, df=1))
            axis.text(0.03, 0.95, f"λ = {inflation:.3f}", transform=axis.transAxes,
                      ha="left", va="top", fontsize=9,
                      color=_BAD if abs(inflation - 1) > 0.2 else _REFERENCE)

        axis = axes[2]
        if adjusted is None:
            axis.text(0.5, 0.5, "No adjusted values supplied",
                      transform=axis.transAxes, ha="center", va="center")
            axis.set_axis_off()
        else:
            q = np.asarray(adjusted, dtype=float)
            q = q[np.isfinite(q)]
            grid = np.linspace(0, min(1.0, max(float(alpha) * 4, 0.2)), 200)
            discoveries = [(q <= level).sum() for level in grid]
            axis.plot(grid, discoveries, color=_MARK, linewidth=1.6)
            axis.axvline(alpha, color=_BAD, linestyle="--", linewidth=1,
                         label=f"α = {alpha:g}")
            called = int((q < alpha).sum())
            axis.scatter([alpha], [called], color=_BAD, zorder=5, s=30)
            axis.set_xlabel("Adjusted-value threshold")
            axis.set_ylabel("Discoveries")
            axis.set_title(f"Discoveries vs threshold ({called} at α)")
            axis.legend(frameon=False, fontsize=8)

        title = "Inference diagnostics"
        if label:
            title = f"{title} — {label}"
        fig.suptitle(title, fontsize=13, fontweight="bold")
        report = {
            "tests": int(n),
            "pi0": float(pi0),
            "estimated_non_null": float((1.0 - pi0) * n),
            # ON THE REPORT, NOT ONLY ON THE PICTURE. lambda was computed here
            # to colour one annotation and then thrown away, so the one number
            # that says whether the whole family of p-values can be believed
            # was readable by eye and by nothing else -- not by the summary
            # CSV, not by a sweep row, not by a verdict.
            "genomic_inflation": inflation,
        }
        if adjusted is not None:
            q = np.asarray(adjusted, dtype=float)
            report["discoveries"] = int(np.sum(q[np.isfinite(q)] < alpha))
        verdict = score_inference(report)
        _stamp(axes[0], verdict)
        report = dict(report, verdict=verdict.headline,
                      verdict_level=verdict.level, verdict_detail=verdict.detail)
        return _finish(fig, save_path, fmt=save_format, title=title), report


def write_diagnostic_suite(destination, *, fractions=None, block=None,
                           observed=None, fitted=None, design=None,
                           p_values=None, adjusted=None, alpha: float = 0.05,
                           label: str = "",
                           presence_threshold: float = 0.0,
                           formats: Sequence[str] | None = None
                           ) -> Mapping[str, str]:
    """Write every diagnostic the supplied inputs can support.

    Nothing is required: pass what a given analysis mode has. The permutation
    test has a design and P values but no fitted values; a simultaneous fit has
    all of them. Each block is skipped silently when its inputs are absent, and
    a block that raises is recorded as an error rather than aborting the run --
    a diagnostic that fails must never take the analysis down with it.

    Every sheet also SCORES itself -- see :func:`score_design`,
    :func:`score_residuals` and :func:`score_inference` -- and wears its own
    badge. The verdicts are rows in ``diagnostic_summary.csv``, per sheet and
    for the suite (its worst), rather than entries in the returned mapping:
    that mapping's contract is "key -> a file that exists". A reader is told
    whether the numbers are fine rather than expected to know.

    :param formats: force particular formats, one file per format. ``None``,
        the default, writes ONE file per panel in the format the user's figure
        preference asks for.

        IT USED TO DEFAULT TO ``("pdf", "png")``, and that was wrong twice
        over. Each entry re-runs the whole panel -- the same numbers computed
        and the same picture drawn a second time -- and the pair ignored the
        figure-format preference completely, so a user who had chosen PNG got
        a PDF anyway. It also broke the rule the design sets, that a
        saved figure and a visible one are the same event: two files for one
        panel is two tiles in the gallery for one picture.
    """
    destination = os.path.abspath(os.path.expanduser(os.fspath(destination)))
    os.makedirs(destination, exist_ok=True)
    stem = f"_{label}" if label else ""
    written: dict[str, str] = {}
    reports: dict[str, dict] = {}
    # None is the sentinel for "the preference decides"; it has to survive as
    # far as `save_figure`, so it is a one-element list rather than a resolved
    # format string.
    requested = [None] if formats is None else [str(f) for f in formats]

    def _emit(name, function, **kwargs):
        for fmt in requested:
            path = os.path.join(destination, f"{name}{stem}")
            try:
                _written, report = function(save_path=path, save_format=fmt,
                                            **kwargs)
            except Exception as error:  # noqa: BLE001 - diagnostics are advisory
                suffix = fmt or "figure"
                written[f"{name}_{suffix}_error"] = f"{type(error).__name__}: {error}"
                continue
            # KEYED BY WHAT WAS WRITTEN, not by what was asked for. The
            # extension `save_figure` chose is the only one that names a file
            # that exists, and a manifest entry pointing at a file that is not
            # there is worse than no entry.
            suffix = (os.path.splitext(str(_written))[1].lstrip(".").lower()
                      or (fmt or "figure"))
            written[f"{name}_{suffix}"] = str(_written)
            reports[name] = report

    if fractions is not None:
        _emit("design_diagnostics", plot_design_diagnostics,
              fractions=fractions, block=block,
              presence_threshold=presence_threshold)
        try:
            pairs = collinear_guide_pairs(fractions)
            path = os.path.join(destination, f"collinear_guide_pairs{stem}.csv")
            pairs.to_csv(path, index=False)
            written["collinear_guide_pairs"] = path
        except Exception as error:  # noqa: BLE001
            written["collinear_guide_pairs_error"] = str(error)

    if observed is not None and fitted is not None:
        _emit("residual_diagnostics", plot_residual_diagnostics,
              observed=observed, fitted=fitted, design=design, label=label)

    if p_values is not None:
        _emit("inference_diagnostics", plot_inference_diagnostics,
              p_values=p_values, adjusted=adjusted, alpha=alpha, label=label)

    if reports:
        summary = os.path.join(destination, f"diagnostic_summary{stem}.csv")
        rows = [
            {"section": section, "metric": key, "value": value}
            for section, report in reports.items()
            for key, value in report.items()
        ]
        # THE SUITE'S VERDICT IS ITS WORST SHEET, and it goes in the SUMMARY
        # rather than in the returned mapping. That mapping's contract is
        # "key -> a file that exists", and a caller iterating it to check its
        # own output is entitled to that; a verdict string in it is a path
        # that is not there. Per-sheet verdicts are already rows here, because
        # each report carries `verdict`, `verdict_level` and `verdict_detail`.
        #
        # A design that cannot identify its own coefficients beside two clean
        # sheets is not "two out of three": it is a run whose numbers are one
        # of infinitely many answers, and a summary that averages that away
        # loses exactly the thing worth reporting.
        levels = [report.get("verdict_level") for report in reports.values()
                  if report.get("verdict_level")]
        if levels:
            from .regression_qc import VERDICT_LEVELS

            worst = max(levels, key=lambda level: VERDICT_LEVELS.index(level)
                        if level in VERDICT_LEVELS else 0)
            rows.append({"section": "suite", "metric": "verdict_level",
                         "value": worst})
            rows.append({"section": "suite", "metric": "verdict",
                         "value": "; ".join(
                             f"{section}: {report.get('verdict_level', 'unknown')}"
                             f" - {report.get('verdict', '')}"
                             for section, report in reports.items())})
        pd.DataFrame(rows).to_csv(summary, index=False)
        written["diagnostic_summary"] = summary
    return written
