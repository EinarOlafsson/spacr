"""One panel showing the response before and after its transform.

Instruction 218: "i want to see a histogram of the dependent variable
distribution after and before transformation (same graph) in the graphs
clearly stating what type of distribution each is, normal bets, etcetera".

THE TRANSFORM IS THE MOST CONSEQUENTIAL SETTING NOBODY LOOKS AT. It decides
which model family is even appropriate, it is chosen from a dropdown, and
its effect is invisible until the coefficients come out the other end -- so
a user who picks the wrong one gets a fit that runs, plots, exports and is
wrong in a way no error can catch.

ONE GRAPH, NOT TWO. The question is what CHANGED, and two panels side by
side is a comparison the reader has to do themselves.

THE NAME COMES FROM `spacr.ml.check_distribution`, which is the function
that picks the regression family. Writing a second classifier for the
picture would let the panel and the fitted model disagree about the same
data, and the panel is the one the user would believe.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np

LOG = logging.getLogger("spacr.response_distribution")

#: What each `check_distribution` answer is called on the panel. Its return
#: values are REGRESSION FAMILIES, and a histogram labelled "quasi_binomial"
#: tells a reader what spaCR would fit rather than what they are looking at.
FAMILY_NAMES: Dict[str, str] = {
    "logit": "binary",
    "quasi_binomial": "bounded, touching 0 or 1",
    "beta": "beta — bounded, strictly inside (0, 1)",
    "ols": "normal",
    "glm": "non-negative and skewed",
}

#: Transforms whose output is not comparable on the input's axis. A log of a
#: proportion and the proportion itself share no scale, so the two histograms
#: get their own x axes rather than being forced onto one that flatters
#: neither.
RESCALING = ("log", "sqrt", "square", "beta", "logit")


def describe(values: Sequence[float]) -> Dict[str, Any]:
    """Name the distribution of ``values``, and give the numbers behind it.

    THE STATISTIC TRAVELS WITH THE NAME so the label can be disagreed with.
    A panel that says "normal" and shows nothing else is asking to be taken
    on faith about the one thing the reader came to check.

    :param values: the response, before or after its transform.
    :returns: ``family``, ``name``, ``normality_p``, ``skew``, ``n``,
        ``low``, ``high``. ``family`` is ``""`` when there is too little
        data to classify, which is an answer rather than a failure.
    """
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    out: Dict[str, Any] = {
        "n": int(data.size), "family": "", "name": "too few values to name",
        "normality_p": float("nan"), "skew": float("nan"),
        "low": float("nan"), "high": float("nan"),
    }
    if data.size < 8:
        return out
    out["low"], out["high"] = float(data.min()), float(data.max())
    try:
        from scipy import stats

        out["skew"] = float(stats.skew(data))
        # D'Agostino, which is what `check_distribution` itself uses -- so
        # the number on the panel is the number that chose the family.
        out["normality_p"] = float(stats.normaltest(data).pvalue)
    except Exception:                                            # noqa: BLE001
        LOG.debug("could not measure the response's shape", exc_info=True)
    try:
        from .ml import check_distribution
        import contextlib
        import io

        # `check_distribution` PRINTS its reasoning, which is useful in a run
        # log and is noise when a plot asks it a question. Swallowed here
        # rather than removed there: the printing is somebody's diagnostic.
        with contextlib.redirect_stdout(io.StringIO()):
            family = str(check_distribution(data))
    except Exception:                                            # noqa: BLE001
        LOG.debug("could not classify the response", exc_info=True)
        return out
    out["family"] = family
    out["name"] = FAMILY_NAMES.get(family, family)
    return out


def transformed(values: Sequence[float], transform: str) -> np.ndarray:
    """Apply ``transform`` the way a run would. Returns the new values.

    THROUGH `spacr.ml.apply_transformation`, never a second implementation:
    a panel showing a log the fit did not take is worse than no panel.
    ``'none'`` and anything unrecognised come back unchanged, which is what
    the run does with them too.
    """
    data = np.asarray(values, dtype=float)
    name = str(transform or "").strip().lower()
    if not name or name == "none":
        return data
    try:
        from .ml import apply_transformation

        transformer = apply_transformation(data, name)
    except Exception:                                            # noqa: BLE001
        LOG.debug("could not build the transformer", exc_info=True)
        return data
    if transformer is None:
        return data
    try:
        return np.asarray(
            transformer.fit_transform(data.reshape(-1, 1)), dtype=float
        ).ravel()
    except Exception:                                            # noqa: BLE001
        LOG.debug("the transform failed on this response", exc_info=True)
        return data


def compare(values: Sequence[float], transform: str) -> Dict[str, Any]:
    """Both distributions and both names, for the panel and for a test.

    :returns: ``before``, ``after`` (each a :func:`describe` dict),
        ``transform``, ``changed`` and ``rescaled``.
    """
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    after = transformed(data, transform)
    name = str(transform or "none").strip().lower() or "none"
    changed = not (after.shape == data.shape
                   and np.allclose(after, data, equal_nan=True))
    return {
        "transform": name,
        "before": describe(data),
        "after": describe(after),
        "values_before": data,
        "values_after": after,
        # A TRANSFORM THAT CHANGED NOTHING IS VISIBLE AS SUCH. An absent
        # panel reads as a missing feature rather than as an answer.
        "changed": bool(changed),
        "rescaled": name in RESCALING,
    }


def caption(result: Dict[str, Any]) -> str:
    """The sentence under the panel. Names both, with their statistics."""
    before, after = result["before"], result["after"]

    def one(part: Dict[str, Any]) -> str:
        if not part["family"]:
            return part["name"]
        bits = [part["name"]]
        if np.isfinite(part["normality_p"]):
            bits.append(f"D'Agostino p = {part['normality_p']:.3g}")
        if np.isfinite(part["skew"]):
            bits.append(f"skew {part['skew']:+.2f}")
        return f"{bits[0]} ({', '.join(bits[1:])})" if len(bits) > 1 \
            else bits[0]

    if not result["changed"]:
        return (f"transform {result['transform']!r} changed nothing: "
                f"{one(before)}. The two histograms are the same data, drawn "
                f"twice, which is what this setting is currently doing.")
    return (f"before: {one(before)}   →   after {result['transform']}: "
            f"{one(after)}")


def panel(values: Sequence[float], transform: str, ax=None,
          dependent_variable: str = ""):
    """Draw both histograms on one axes. Returns the :func:`compare` dict.

    :param ax: draw here; a new figure is made when omitted. Through
        ``matplotlib.figure.Figure`` rather than pyplot in that case -- a
        Figure pyplot never sees cannot be leaked into its global registry,
        which this repo has been bitten by more than once.
    :param dependent_variable: named on the axis, so a panel pasted into a
        notebook still says what it is about.
    """
    result = compare(values, transform)
    if ax is None:
        from matplotlib.figure import Figure

        figure = Figure(figsize=(7.0, 4.0))
        ax = figure.add_subplot(111)
    before = result["values_before"]
    after = result["values_after"]

    if result["rescaled"] and result["changed"]:
        # SEPARATE AXES, SHARED FIGURE. A log of a proportion and the
        # proportion itself have no common scale, and forcing them onto one
        # puts every point of the smaller into a single bar -- which looks
        # like a finding and is an artefact of the axis.
        twin = ax.twiny()
        ax.hist(before, bins=40, alpha=0.55, label="before",
                color="#4C72B0")
        twin.hist(after, bins=40, alpha=0.55, label="after",
                  color="#DD8452")
        twin.set_xlabel(f"after {result['transform']}")
        handles = [ax.patches[0]] if ax.patches else []
        handles += [twin.patches[0]] if twin.patches else []
        if handles:
            ax.legend(handles, ["before", f"after {result['transform']}"],
                      loc="upper right", fontsize=8)
    else:
        # ONE AXIS, which is what makes "what changed" readable at a glance.
        both = np.concatenate([before, after]) if after.size else before
        edges = np.histogram_bin_edges(both[np.isfinite(both)], bins=40)
        ax.hist(before, bins=edges, alpha=0.55, label="before",
                color="#4C72B0")
        ax.hist(after, bins=edges, alpha=0.55,
                label=f"after {result['transform']}", color="#DD8452")
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel(dependent_variable or "response")
    ax.set_ylabel("wells")
    ax.set_title("Response before and after the transform", fontsize=10)
    # THE NAMES GO ON THE PANEL, which is the substance of the request --
    # not left for the reader to judge by eye.
    ax.text(0.01, -0.22, caption(result), transform=ax.transAxes,
            fontsize=8, va="top", wrap=True)
    result["axes"] = ax
    return result
