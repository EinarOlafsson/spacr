"""Compare a response distribution before and after transformation.

The module applies the same transformation and distribution classifier used
by the regression pipeline. Its combined histogram therefore provides a
diagnostic of how the selected transformation changes the response and the
candidate model family.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np

LOG = logging.getLogger("spacr.response_distribution")

#: Human-readable distribution labels for ``check_distribution`` results.
FAMILY_NAMES: Dict[str, str] = {
    "logit": "binary",
    "quasi_binomial": "bounded, touching 0 or 1",
    "beta": "beta — bounded, strictly inside (0, 1)",
    "ols": "normal",
    "glm": "non-negative and skewed",
}

#: Transformations whose output requires a separate horizontal axis.
RESCALING = ("log", "sqrt", "square", "beta", "logit")


def describe(values: Sequence[float]) -> Dict[str, Any]:
    """Summarize and classify a response distribution.

    Parameters
    ----------
    values : sequence of float
        Response values before or after transformation. Non-finite values
        are excluded.

    Returns
    -------
    dict
        Sample size, range, skewness, D'Agostino normality-test p-value,
        regression-family identifier, and display label. The family is empty
        when fewer than eight finite observations are available or
        classification fails.
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
    """Apply the regression pipeline's response transformation.

    Parameters
    ----------
    values : sequence of float
        Response values to transform.
    transform : str
        Transformation accepted by :func:`spacr.ml.apply_transformation`.

    Returns
    -------
    numpy.ndarray
        Transformed values. Values are returned unchanged for ``"none"``, an
        unsupported transformation, or a transformation that cannot be
        applied.
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
    """Compare response distributions before and after transformation.

    Parameters
    ----------
    values : sequence of float
        Response values. Non-finite values are excluded.
    transform : str
        Transformation to evaluate.

    Returns
    -------
    dict
        The original and transformed arrays, their summaries, the normalized
        transformation name, and flags indicating whether values changed and
        whether the transformed values require a separate axis.
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
    """Format a statistical caption for a :func:`compare` result."""
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


def fast_panel(values: Sequence[float], transform: str, plot=None,
               dependent_variable: str = ""):
    """Draw the before/after comparison on a pyqtgraph plot.

    The same picture :func:`panel` draws, on the screen's own renderer, so
    the figure a run writes and the figure a tab shows are one scene. Both
    distributions go on ONE pair of axes as outlines rather than on two
    stacked panels: the question is whether the transform moved the shape,
    and two shapes on separate axes with separate scales is the one layout
    that cannot answer it.

    Parameters
    ----------
    values : sequence of float
        Response values. Non-finite values are excluded.
    transform : str
        Transformation name, as :func:`transformed` accepts.
    plot : FastPlot or None, default=None
        Where to draw. One is created when omitted.
    dependent_variable : str, default=""
        The response's name, for the axis label.

    Returns
    -------
    FastPlot or None
        The plot drawn on, or None when there is nothing finite to draw.
    """
    result = compare(values, transform)
    before = np.asarray(result["values_before"], dtype=float)
    after = np.asarray(result["values_after"], dtype=float)
    if not before.size:
        return None

    if plot is None:
        from .qt.widgets.fast_plots import FastPlot

        plot = FastPlot(title="Response distribution",
                        x_label=str(dependent_variable or "response"),
                        y_label="wells")

    from .qt.widgets.fast_plots import colour_for

    bins = max(10, min(60, int(np.sqrt(before.size)) * 2))
    for index, (series, name) in enumerate(
            ((before, result["before"]["name"]),
             (after, result["after"]["name"]))):
        if not series.size:
            continue
        counts, edges = np.histogram(series, bins=bins)
        # A STEP OUTLINE, NOT FILLED BARS. Two filled histograms on one axis
        # hide each other whichever order they are drawn in; two outlines
        # overlay and stay readable, which is the comparison being asked
        # for.
        xs = np.repeat(edges, 2)[1:-1]
        ys = np.repeat(counts, 2)
        plot.plot.plot(xs, ys, pen=_pen(colour_for(index), name))
    plot.set_status(caption(result))
    return plot


def _pen(colour, name):
    """A 2 px pen in ``colour``, named for the legend."""
    import pyqtgraph as pg

    return pg.mkPen(colour, width=2.0)


def panel(values: Sequence[float], transform: str, ax=None,
          dependent_variable: str = ""):
    """Plot response distributions before and after transformation.

    Parameters
    ----------
    values : sequence of float
        Response values to compare.
    transform : str
        Transformation to apply.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw. A standalone figure and axes are created when
        omitted.
    dependent_variable : str, optional
        Response name shown on the horizontal axis.

    Returns
    -------
    dict
        Result from :func:`compare`, augmented with the primary axes under
        ``"axes"``.
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
