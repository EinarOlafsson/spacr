"""What each regression family assumes, and which of three kinds it is.

Nineteen families in one flat alphabetical list is a menu that hides its own
contents. The quantile fit, the two robust losses and the rank aggregation
were all present and none of them could be found, so the screen looked as
though it offered least squares and nothing else.

THE THREE KINDS, and the distinction between them is the one a reviewer
checks:

``parametric``
    the response is given a distribution and the p-values come from that
    distribution's theory. Least squares, the GLM links, the penalised fits
    and the mixed model.
``robust_semiparametric``
    still a linear model, still parametric IN THE COEFFICIENTS, but the loss
    is chosen so that a handful of extreme wells cannot set a coefficient on
    their own. ``rlm``, ``huber`` and ``quantile`` are these -- and calling
    them nonparametric, as the request that produced this table did, is wrong
    in a way that matters: only the ERROR term is left unspecified, and for
    ``quantile`` alone.
``rank_based``
    the fit reads only the ORDER of the wells, so no distribution is assumed
    anywhere. ``rra`` is the one genuinely nonparametric family here.

DISTRIBUTION-FREE INFERENCE IS A SEPARATE AXIS and is not in this table. Any
family that produces a coefficient can be paired with a permutation null,
which is what makes the p-value assumption-free; the kind recorded here is
about how the EFFECT is estimated, not how it is tested.

NOTHING IS RENAMED. These are the stored values, so a settings file written
before the grouping existed asks for exactly the fit it always asked for.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .regression_spec import (NO_P_VALUE_TYPES, REGRESSION_TYPES,
                              UNSUPPORTED_REGRESSION_TYPES)

__all__ = [
    "REGRESSION_FAMILY_ASSUMPTIONS",
    "REGRESSION_FAMILY_GROUPS",
    "GROUP_TITLES",
    "family_group",
    "family_label",
    "regression_family_choices",
]

#: The three kinds, in the order a panel lists them, and the families in each.
#:
#: ``mixed`` leads the parametric group because it is the default and answers
#: the most central question; the rest of each group is alphabetical, so a
#: family added to the inventory lands somewhere predictable.
REGRESSION_FAMILY_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("parametric", (
        "mixed",
        "beta", "elasticnet", "glm", "group_lasso", "hinge", "horseshoe",
        "lasso", "logit", "ols", "poisson", "probit", "quasi_binomial",
        "ridge", "wls",
    )),
    ("robust_semiparametric", ("huber", "quantile", "rlm")),
    ("rank_based", ("rra",)),
)

#: What a panel calls each kind.
GROUP_TITLES: Dict[str, str] = {
    "parametric": "parametric",
    "robust_semiparametric": "robust/semiparametric",
    "rank_based": "rank-based",
}

#: One sentence per family: what it assumes, in the terms the choice is made
#: in. Not what it computes -- the API links say that -- but what has to be
#: true of the data for its answer to mean anything.
REGRESSION_FAMILY_ASSUMPTIONS: Dict[str, str] = {
    "mixed": ("guides nested in genes as random effects, those effects "
              "normal; guides that disagree widen their gene's interval"),
    "ols": "least squares: normal errors of constant variance, every well equal",
    "wls": ("least squares weighted by cell count: normal errors whose "
            "variance falls as a well holds more cells"),
    "glm": ("a family chosen from the response, whose mean-variance "
            "relationship is then assumed to hold"),
    "poisson": ("counts, variance equal to the mean, log cell count as "
                "exposure so the effect is on a per-cell rate"),
    "quasi_binomial": ("a fraction with binomial mean-variance, freely "
                       "rescaled, so overdispersion is absorbed not modelled"),
    "beta": ("a fraction strictly inside 0 and 1, beta distributed; exact "
             "zeros and ones have to be moved off the boundary first"),
    "logit": ("a fraction as a binomial proportion, log-odds linear in the "
              "guides, weighted by the cells behind the well"),
    "probit": ("the same binomial proportion as logit through a normal "
               "link, so the tails are lighter"),
    "lasso": ("normal errors plus an L1 penalty: the truth is sparse and one "
              "guide of a correlated group carries the effect"),
    "ridge": ("normal errors plus an L2 penalty: the effect is shared across "
              "correlated guides rather than assigned to one"),
    "elasticnet": ("both penalties mixed by l1_ratio: sparse, but correlated "
                   "guides of a gene are kept or dropped together"),
    "horseshoe": ("Bayesian shrinkage assuming most guides do nothing and a "
                  "few do a lot; the interval is a posterior, not a p-value"),
    "group_lasso": ("a gene's guides penalised as one block, so a gene is in "
                    "or out as a whole rather than by its best guide"),
    "hinge": ("a linear boundary between a positive and a negative class: "
              "the response is thresholded, so the question is which class a "
              "well is in, not by how much"),
    "rlm": ("a linear model fitted by an M-estimator, so extreme wells are "
            "downweighted rather than assumed absent"),
    "huber": ("squared loss near the centre and linear in the tails, so a "
              "handful of bright wells cannot set a coefficient"),
    "quantile": ("a quantile of the response rather than its mean, which "
                 "assumes nothing about the shape of the errors; tau=0.9 "
                 "asks whether the guide moves the TOP of the distribution"),
    "rra": ("only the ORDER of the wells, so no distribution at all; its p "
            "value is a permutation p value over guide ranks"),
}

#: Appended to the families whose coefficient has no usable p-value, from the
#: inventory that decides it, so the two cannot disagree.
_NO_P_VALUE_NOTE = ("no p value from the fit -- ranked by bootstrap "
                    "selection frequency")


def _fittable() -> Tuple[str, ...]:
    """Every family that can actually be fitted, in inventory order."""
    return tuple(name for name in REGRESSION_TYPES
                 if name not in UNSUPPORTED_REGRESSION_TYPES)


def family_group(name: str) -> str:
    """Which of the three kinds a family is.

    :param name: a stored ``regression_type`` value.
    :returns: ``'parametric'``, ``'robust_semiparametric'`` or
        ``'rank_based'``.
    :raises KeyError: a family this table does not place, which is the
        signal that the inventory grew and this did not.
    """
    key = str(name).strip().lower()
    for group, families in REGRESSION_FAMILY_GROUPS:
        if key in families:
            return group
    raise KeyError(f"{name!r} is in no regression family group")


def family_label(name: str) -> str:
    """The one line a dropdown shows for a family: its kind and its assumption.

    The stored value leads, because that is what a user is looking for and
    what every settings file and results folder is named after.
    """
    key = str(name).strip().lower()
    assumption = REGRESSION_FAMILY_ASSUMPTIONS[key]
    if key in NO_P_VALUE_TYPES:
        assumption = f"{assumption}; {_NO_P_VALUE_NOTE}"
    return f"{key} — {GROUP_TITLES[family_group(key)]}: {assumption}"


def regression_family_choices() -> List[Tuple[str, str]]:
    """Every fittable family as ``(stored value, label)``, grouped.

    Parametric first, then robust/semiparametric, then rank-based; the stored
    value is unchanged, so a panel that renders these writes exactly what it
    wrote before and every settings file already on disk still means what it
    meant.
    """
    fittable = set(_fittable())
    out: List[Tuple[str, str]] = []
    for _group, families in REGRESSION_FAMILY_GROUPS:
        for name in families:
            if name in fittable:
                out.append((name, family_label(name)))
    return out
