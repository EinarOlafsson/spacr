"""Multiple-testing corrections for spaCR, with one name for every method.

Every correction spaCR offers lives here, so the GUI dropdown, the CLI, the
settings validator and :mod:`spacr.guide_permutation` cannot drift apart: the
dropdown is built from :data:`METHODS`, the validator checks against the same
table, and the analysis calls :func:`adjust_p_values`.

Two properties matter for a pooled screen and are easy to get wrong:

* **Missing P values must not join the family.** A guide that could not be
  tested is not a test. Counting it inflates the family size and makes every
  real discovery less significant. NaNs pass through untouched here.
* **The family is the displayed set.** The caller decides what a family is
  (spaCR corrects separately per classifier outcome and per minimum-support
  threshold); this module only corrects whatever vector it is handed.

Beyond the statsmodels inventory this module implements Storey's q-value,
which is standard in screening work and absent from statsmodels. It estimates
the proportion of true nulls (pi0) rather than assuming it is 1, so it is
uniformly less conservative than Benjamini-Hochberg while targeting the same
false-discovery rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "METHODS",
    "MethodSpec",
    "canonical_method",
    "method_choices",
    "method_label",
    "adjust_p_values",
    "storey_qvalue",
    "estimate_pi0",
]


@dataclass(frozen=True)
class MethodSpec:
    """One correction: its canonical key, label, family and controlled rate."""

    key: str
    label: str
    controls: str
    #: ``statsmodels`` name, or ``None`` when spaCR implements it itself.
    statsmodels_name: str | None
    summary: str


#: Canonical key -> :class:`MethodSpec`, in the order the dropdown shows them.
#: Ordered least-to-most permissive within each family so that scrolling down
#: the list moves monotonically from "fewest discoveries" to "most".
METHODS: dict[str, MethodSpec] = {
    "none": MethodSpec(
        "none", "None (raw P values)", "nothing", None,
        "No correction. Every P value is reported as tested. Use only when "
        "the family really is a single prespecified test.",
    ),
    "bonferroni": MethodSpec(
        "bonferroni", "Bonferroni", "FWER", "bonferroni",
        "Multiply each P value by the number of tests. Valid under any "
        "dependence, and the most conservative option offered.",
    ),
    "sidak": MethodSpec(
        "sidak", "Sidak", "FWER", "sidak",
        "Slightly less conservative than Bonferroni, but assumes the tests "
        "are independent.",
    ),
    "holm": MethodSpec(
        "holm", "Holm-Bonferroni (step-down)", "FWER", "holm",
        "Uniformly more powerful than Bonferroni with the same guarantee "
        "under any dependence. The default choice when FWER is wanted.",
    ),
    "holm_sidak": MethodSpec(
        "holm_sidak", "Holm-Sidak (step-down)", "FWER", "holm-sidak",
        "Step-down Sidak. A little more powerful than Holm when the tests "
        "are independent.",
    ),
    "simes_hochberg": MethodSpec(
        "simes_hochberg", "Simes-Hochberg (step-up)", "FWER",
        "simes-hochberg",
        "Step-up FWER control. More powerful than Holm, but requires "
        "independent or positively dependent tests.",
    ),
    "hommel": MethodSpec(
        "hommel", "Hommel", "FWER", "hommel",
        "The most powerful FWER method here under positive dependence, at a "
        "higher computational cost. Slow for very large families.",
    ),
    "fdr_bh": MethodSpec(
        "fdr_bh", "Benjamini-Hochberg FDR", "FDR", "fdr_bh",
        "The standard screening default. Controls the expected proportion of "
        "false discoveries under independence or positive dependence.",
    ),
    "fdr_by": MethodSpec(
        "fdr_by", "Benjamini-Yekutieli FDR", "FDR", "fdr_by",
        "Benjamini-Hochberg made valid under arbitrary dependence, at the "
        "cost of a log(m) penalty. Use when guides co-occur unpredictably.",
    ),
    "fdr_tsbh": MethodSpec(
        "fdr_tsbh", "Two-stage Benjamini-Hochberg FDR", "FDR", "fdr_tsbh",
        "Estimates the number of true nulls in a first pass, then applies "
        "Benjamini-Hochberg to that smaller family. More powerful when a "
        "large share of tests are non-null.",
    ),
    "fdr_tsbky": MethodSpec(
        "fdr_tsbky", "Two-stage Benjamini-Krieger-Yekutieli FDR", "FDR",
        "fdr_tsbky",
        "A different two-stage adaptive estimator of the true-null count. "
        "Behaves like fdr_tsbh and is usually within rounding of it.",
    ),
    "fdr_gbs": MethodSpec(
        "fdr_gbs", "Adaptive Gavrilov-Benjamini-Sarkar FDR", "FDR",
        "fdr_gbs",
        "Adaptive step-down FDR. Powerful on sparse families, which is the "
        "usual shape of a pooled screen.",
    ),
    "storey": MethodSpec(
        "storey", "Storey q-value (pi0-adaptive FDR)", "FDR", None,
        "Estimates the proportion of true nulls from the P-value histogram "
        "and rescales Benjamini-Hochberg by it. Never more conservative than "
        "Benjamini-Hochberg. Needs a reasonably large family to estimate pi0.",
    ),
}


#: Spellings accepted from settings CSVs, the CLI and older runs.
_ALIASES: dict[str, str] = {
    "": "none",
    "raw": "none",
    "uncorrected": "none",
    "nan": "none",
    "b": "bonferroni",
    "bonf": "bonferroni",
    "s": "sidak",
    "h": "holm",
    "holm-bonferroni": "holm",
    "holm_bonferroni": "holm",
    "hs": "holm_sidak",
    "holm-sidak": "holm_sidak",
    "sh": "simes_hochberg",
    "simes-hochberg": "simes_hochberg",
    "hochberg": "simes_hochberg",
    "ho": "hommel",
    "bh": "fdr_bh",
    "benjamini-hochberg": "fdr_bh",
    "benjamini_hochberg": "fdr_bh",
    "fdr": "fdr_bh",
    "by": "fdr_by",
    "benjamini-yekutieli": "fdr_by",
    "benjamini_yekutieli": "fdr_by",
    "tsbh": "fdr_tsbh",
    "fdr_2sbh": "fdr_tsbh",
    "tsbky": "fdr_tsbky",
    "gbs": "fdr_gbs",
    "qvalue": "storey",
    "q_value": "storey",
    "storey_qvalue": "storey",
    "storey-tibshirani": "storey",
}


def canonical_method(method) -> str:
    """Return the canonical key for ``method``.

    Accepts the canonical keys, the statsmodels spellings and the common
    aliases. ``None`` maps to ``'none'``. Raises :class:`ValueError` with the
    full inventory for anything else, rather than silently falling back to a
    correction the user did not ask for.
    """
    if method is None:
        return "none"
    key = str(method).strip().lower().replace(" ", "_")
    key = _ALIASES.get(key, key)
    if key in METHODS:
        return key
    # statsmodels spellings that are not already canonical keys.
    for spec in METHODS.values():
        if spec.statsmodels_name and key == spec.statsmodels_name.lower():
            return spec.key
    raise ValueError(
        f"Unsupported multiple-testing method {method!r}. Choose one of: "
        f"{', '.join(METHODS)}."
    )


def method_choices() -> list[str]:
    """Canonical keys in dropdown order."""
    return list(METHODS)


def method_label(method) -> str:
    """Human-readable label for ``method``."""
    return METHODS[canonical_method(method)].label


def estimate_pi0(p_values, *, lambdas: Sequence[float] | None = None) -> float:
    """Estimate the proportion of true null hypotheses (Storey's pi0).

    Uses the smoothed bootstrap-free spline-free estimator: pi0(lambda) is
    computed over a grid and the estimate is taken at the largest lambda whose
    value is stable, then clipped to (0, 1]. With few tests the grid collapses
    and the estimator returns 1.0, which makes the q-values fall back to
    Benjamini-Hochberg -- the conservative answer, not an error.
    """
    values = np.asarray(p_values, dtype=float)
    values = values[np.isfinite(values)]
    m = values.size
    if m == 0:
        return 1.0
    if lambdas is None:
        lambdas = np.arange(0.05, 0.96, 0.05)
    grid = np.asarray([lam for lam in lambdas if 0.0 <= lam < 1.0], dtype=float)
    if grid.size == 0 or m < 20:
        # Too few tests to read a null plateau off the histogram.
        return 1.0
    counts = np.asarray([(values > lam).sum() for lam in grid], dtype=float)
    pi0_grid = counts / (m * (1.0 - grid))
    # Take the minimum of the tail estimates: stable, and never above 1.
    tail = pi0_grid[grid >= float(np.median(grid))]
    pi0 = float(np.min(tail)) if tail.size else float(pi0_grid[-1])
    if not np.isfinite(pi0) or pi0 <= 0:
        return 1.0
    return float(min(pi0, 1.0))


def storey_qvalue(p_values, *, pi0: float | None = None):
    """Return Storey q-values for a vector of P values.

    ``q[i]`` is the minimum positive-FDR at which test ``i`` is called
    significant. The result is monotone in the P value, as the definition
    requires. NaNs are preserved.
    """
    values = np.asarray(p_values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    observed = values[finite]
    m = observed.size
    if m == 0:
        return out
    if pi0 is None:
        pi0 = estimate_pi0(observed)
    order = np.argsort(observed, kind="stable")
    ranked = observed[order]
    ranks = np.arange(1, m + 1, dtype=float)
    raw = pi0 * m * ranked / ranks
    # Enforce monotonicity from the largest P value downwards.
    q_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty(m, dtype=float)
    q[order] = q_sorted
    out[finite] = q
    return out


def adjust_p_values(p_values, method="fdr_bh", alpha=0.05):
    """Return ``(adjusted, rejected)`` for one multiple-testing family.

    :param p_values: P values for every test in the family. Non-finite entries
        stay non-finite, are never rejected, and do not count toward the family
        size.
    :param method: any key, statsmodels name or alias accepted by
        :func:`canonical_method`.
    :param alpha: the level the correction targets, strictly inside (0, 1).
    :returns: ``adjusted`` (same shape as the input) and ``rejected``, a
        boolean array.
    """
    values = np.asarray(p_values, dtype=float)
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1; got {alpha!r}")
    key = canonical_method(method)
    spec = METHODS[key]

    adjusted = np.full(values.shape, np.nan, dtype=float)
    rejected = np.zeros(values.shape, dtype=bool)
    finite = np.isfinite(values)
    if not finite.any():
        return adjusted, rejected
    observed = values[finite]
    if (observed < 0).any() or (observed > 1).any():
        raise ValueError("P values must lie in [0, 1]")

    if key == "none":
        adjusted[finite] = observed
        rejected[finite] = observed < alpha
        return adjusted, rejected

    if key == "storey":
        q = storey_qvalue(observed)
        adjusted[finite] = q
        rejected[finite] = q < alpha
        return adjusted, rejected

    from statsmodels.stats.multitest import multipletests

    call, corrected, _, _ = multipletests(
        observed, alpha=alpha, method=spec.statsmodels_name
    )
    # fdr_tsbh returns adjusted values already scaled by the estimated null
    # count; statsmodels' own rejection call is authoritative for every method.
    adjusted[finite] = corrected
    rejected[finite] = call
    return adjusted, rejected
