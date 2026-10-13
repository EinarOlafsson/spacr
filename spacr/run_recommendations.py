"""Generate reproducible recommendations from regression diagnostics.

Each recommendation is a fixed rule triggered by a measured threshold. The
result names the setting to change, the proposed action, the evidence that
triggered it, and whether the issue blocks interpretation. Rules that do not
fire are omitted, so the summary contains only actions supported by the
current run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

__all__ = ["Recommendation", "recommend", "format_recommendations"]


@dataclass(frozen=True)
class Recommendation:
    """One thing to change, why, and how strongly.

    :param setting: spaCR setting key the recommendation proposes changing.
    :param action: human-readable proposed value or action; it is display
        prose and is not applied automatically.
    :param because: measured evidence explaining why the recommendation
        fired.
    :param severity: ``"blocking"`` for a result that should not be trusted
        until addressed; ``"consider"`` by default, with every other value
        rendered as advisory.
    """

    setting: str
    action: str
    because: str
    severity: str = "consider"

    def line(self) -> str:
        """Return the two-line display block for this recommendation."""
        mark = "!" if self.severity == "blocking" else "-"
        return f"  {mark} {self.setting}: {self.action}\n      because {self.because}"


#: Below this, D'Agostino has rejected normality hard enough that the
#: parametric p-values should not be relied on.
NORMALITY_ALPHA = 0.01

#: Excess kurtosis above this is heavy enough that a least-squares statistic
#: is materially moved by its tails.
KURTOSIS_HEAVY = 3.0

#: How far from 2 Durbin-Watson may sit before the independence assumption
#: is worth acting on.
DW_TOLERANCE = 0.4

#: Cook's-distance threshold for an observation with substantial individual
#: influence on the fitted model.
COOKS_INFLUENTIAL = 1.0

#: The usual collinearity bar.
VIF_HIGH = 10.0


def _number(source: Mapping[str, Any], *names: str) -> Optional[float]:
    """First finite value among ``names``, or ``None``.

    Several spellings because the diagnostics dict is assembled by more than
    one writer, and a recommendation that silently does not fire because it
    looked for the wrong key is worse than one that is absent by design.
    """
    for name in names:
        if name not in source:
            continue
        try:
            value = float(source[name])
        except (TypeError, ValueError):
            continue
        if value == value and abs(value) != float("inf"):
            return value
    return None


def recommend(diagnostics: Mapping[str, Any], *,
              settings: Optional[Mapping[str, Any]] = None
              ) -> List[Recommendation]:
    """Derive recommendations from a run's measured diagnostics.

    :param diagnostics: the numbers the QC computed.
    :param settings: settings used for the run. Recommendations already
        satisfied by these values are omitted.
    :returns: the recommendations that fired, blocking ones first.
    """
    settings = dict(settings or {})
    out: List[Recommendation] = []

    inference = str(settings.get("inference", "")).lower()
    statistic = str(settings.get("grna_statistic", "pearson")).lower()

    normality_p = _number(diagnostics, "normality_p", "dagostino_p",
                          "normality_p_value")
    if normality_p is not None and normality_p < NORMALITY_ALPHA:
        if inference not in ("nonparametric", "permutation"):
            out.append(Recommendation(
                "inference", "'nonparametric'",
                f"the residuals are not normal (p = {normality_p:.2g}), and "
                f"a parametric p-value assumes they are. The coefficients "
                f"are unaffected; only their significance is.",
                severity="blocking"))
        kurtosis = _number(diagnostics, "excess_kurtosis", "kurtosis")
        if kurtosis is not None and kurtosis > KURTOSIS_HEAVY \
                and statistic != "rank":
            out.append(Recommendation(
                "grna_statistic", "'rank'",
                f"excess kurtosis is {kurtosis:+.2f}, so the tails are heavy "
                f"enough to move a correlation. A rank responds to order "
                f"rather than magnitude."))

    dw = _number(diagnostics, "durbin_watson")
    if dw is not None and abs(dw - 2.0) > DW_TOLERANCE:
        already = [str(c) for c in
                   (settings.get("guide_nuisance_columns") or [])]
        if not already:
            out.append(Recommendation(
                "guide_nuisance_columns", "['rowID', 'columnID']",
                f"Durbin-Watson is {dw:.2f} against 2 for none, so "
                f"neighbouring wells are not independent and the shuffle "
                f"treats that structure as noise.",
                severity="blocking"))
        else:
            out.append(Recommendation(
                "guide_permutation_block", "a grouping that spans the "
                                           "structure",
                f"Durbin-Watson is {dw:.2f} with {', '.join(already)} "
                f"already removed, so the remaining structure is not plate "
                f"position."))

    cooks = _number(diagnostics, "max_cooks_distance", "cooks_max")
    if cooks is not None and cooks >= COOKS_INFLUENTIAL:
        kind = str(settings.get("regression_type", "")).lower()
        if kind not in ("rlm", "huber", "quantile"):
            out.append(Recommendation(
                "regression_type", "'rlm', 'huber' or 'quantile'",
                f"one observation has Cook's distance {cooks:.2f}, above the "
                f"1.0 rule -- it moves the fit on its own, and least squares "
                f"is the estimator most sensitive to that."))

    vif = _number(diagnostics, "max_vif")
    if vif is not None and vif > VIF_HIGH:
        out.append(Recommendation(
            "regression_type", "'ridge', 'lasso' or 'elasticnet'",
            f"the largest VIF is {vif:.1f}, above 10, so the predictors "
            f"carry overlapping information and their individual "
            f"coefficients are not separable."))

    wells = _number(diagnostics, "median_wells_per_guide")
    if wells is not None and wells < 3:
        out.append(Recommendation(
            "fraction_threshold", "None, to sweep it",
            f"the median gRNA is in {wells:.0f} well(s) after the threshold, "
            f"which is too few for the support families to resolve. The "
            f"sweep shows what other thresholds would keep.",
            severity="blocking"))

    out.sort(key=lambda r: 0 if r.severity == "blocking" else 1)
    return out


def format_recommendations(items: List[Recommendation]) -> str:
    """Format recommendations for the end of a run summary.

    :param items: ordered run recommendations to render.

    An empty input is reported explicitly so a completed check with no
    recommendations is distinguishable from a missing result.
    """
    if not items:
        return ("RECOMMENDATIONS\n"
                "  Nothing to change: every check this run made either "
                "passed or\n  had nothing to act on.")
    lines = ["RECOMMENDATIONS"]
    blocking = [r for r in items if r.severity == "blocking"]
    if blocking:
        lines.append(f"  {len(blocking)} of these should be settled before "
                     f"the result is relied on (marked !).")
    lines.extend(item.line() for item in items)
    return "\n".join(lines)
