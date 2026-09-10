"""Match graph types and defaults to the structure of plotted data.

The module distinguishes categorical, continuous, and ordered axes. It lists
compatible graph types first and supplies a reason for each incompatible
option, allowing interfaces to disable unsupported choices without hiding
them.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

LOG = logging.getLogger("spacr.graph_types")

#: Available graph types represented as ``(value, display description)``.
GRAPH_TYPES: Tuple[Tuple[str, str], ...] = (
    ("bar", "one value per group"),
    ("bar_jitter", "the summary AND the observations"),
    ("box_jitter", "the five numbers AND the observations"),
    ("jitter", "every observation"),
    ("box", "the distribution's shape, five numbers"),
    ("violin", "the distribution's shape, continuous"),
    ("line", "ordered x only, with or without a spread band"),
    ("scatter", "two continuous axes"),
)

#: Concise menu labels for the graph types described by :data:`GRAPH_TYPES`.
#:
#: Descriptions remain available for explanatory tooltips, while these names
#: keep graph-selection menus compact and scannable.
GRAPH_NAMES: Dict[str, str] = {
    "bar": "Bar",
    "bar_jitter": "Bar with jitter",
    "box_jitter": "Box with jitter",
    "jitter": "Jitter",
    "box": "Box",
    "violin": "Violin",
    "line": "Line",
    "scatter": "Scatter",
}


#: Supported combinations of axis data types.
DATA_SHAPES: Tuple[Tuple[str, str], ...] = (
    ("categorical_continuous", "groups against a measurement"),
    ("continuous_continuous", "a measurement against a measurement"),
    ("ordered_continuous", "an ordered x against a measurement"),
    ("continuous_only", "one measurement, no grouping"),
)

#: Which types FIT each shape.
FITS: Dict[str, Tuple[str, ...]] = {
    "categorical_continuous": ("box_jitter", "bar_jitter", "bar", "jitter",
                               "box", "violin"),
    # NO BAR AND NO BOX HERE. Both need groups to summarise, and forming
    # groups out of a continuous x means binning it -- which is a different
    # graph of different data, not this one drawn another way.
    "continuous_continuous": ("scatter",),
    # A LINE NEEDS AN ORDER. Through unordered categories it is a row of
    # markers joined for no reason, which is why it is here and not above.
    "ordered_continuous": ("line", "scatter", "jitter"),
    "continuous_only": ("jitter", "box", "violin"),
}

#: Default graph type for each supported data shape.
#:
#: A BOX, NOT A BAR, for groups against a measurement. A bar drawn at a
#: mean shows one number and hides the shape of the data: two groups with
#: the same mean and completely different spreads draw the same bar. The
#: box shows the median, the quartiles and the whiskers, and the jitter
#: stays because the box summarises and the points are the evidence.
#: `spacr.plot` and `spacr.settings` already default to `jitter_box` for
#: the same reason; this table disagreeing with them was a second answer
#: to one question.
DEFAULTS: Dict[str, str] = {
    "categorical_continuous": "box_jitter",
    "continuous_continuous": "scatter",
    "ordered_continuous": "line",
    "continuous_only": "jitter",
}

#: Explanations for incompatible data-shape and graph-type pairs.
WHY_NOT: Dict[Tuple[str, str], str] = {
    ("continuous_continuous", "bar"):
        "a bar needs groups to summarise, and grouping a continuous x means "
        "binning it -- a different graph of different data",
    ("continuous_continuous", "bar_jitter"):
        "a bar needs groups to summarise, and grouping a continuous x means "
        "binning it -- a different graph of different data",
    ("continuous_continuous", "box_jitter"):
        "a box is five numbers per group, and a continuous x has no groups",
    ("continuous_continuous", "box"):
        "a box is five numbers per group, and a continuous x has no groups",
    ("continuous_continuous", "violin"):
        "a violin is a distribution per group, and a continuous x has no "
        "groups",
    ("continuous_continuous", "line"):
        "a line needs the x to be ordered; use the ordered shape if it is",
    ("categorical_continuous", "scatter"):
        "a scatter of one categorical against one continuous is a jitter "
        "under another name",
    ("categorical_continuous", "line"):
        "a line through unordered categories is a row of markers joined for "
        "no reason",
    ("continuous_only", "bar"):
        "a bar needs groups, and there is only one measurement here",
    ("continuous_only", "box_jitter"):
        "a box per group needs groups, and there is only one measurement "
        "here -- the plain box draws the same five numbers",
    ("continuous_only", "line"):
        "a line needs an ordered x, and there is only one measurement here",
    ("continuous_only", "scatter"):
        "a scatter needs two axes, and there is only one measurement here",
}


def shape_of(frame, x: str = "", y: str = "") -> str:
    """Classify the axis structure of a data frame.

    Parameters
    ----------
    frame : pandas.DataFrame
        Source data.
    x, y : str, optional
        Column names assigned to the horizontal and vertical axes. An empty
        ``x`` represents an ungrouped continuous distribution.

    Returns
    -------
    str
        Key from :data:`DATA_SHAPES`. A numeric, unique, monotonically
        increasing x-axis is classified as ordered continuous data.
    """
    import pandas as pd

    def kind(name: str) -> str:
        """Classify one candidate axis against the captured data frame.

        :param name: column name assigned to the axis, or an empty string for
            an unassigned axis.
        :returns: ``'absent'`` when the column is unavailable,
            ``'continuous'`` for a numeric column, or ``'categorical'`` for
            every other present column.
        """
        if not name or frame is None or name not in getattr(
                frame, "columns", ()):
            return "absent"
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series):
            return "continuous"
        return "categorical"

    left, right = kind(x), kind(y)
    if left == "absent":
        return "continuous_only"
    if left == "continuous" and right == "continuous":
        # ORDERED IS A PROPERTY OF THE VALUES, not of the dtype. An x that
        # is already sorted and unique is a series; one that is neither is a
        # cloud, and joining a cloud with a line is 178 A's bug.
        try:
            values = frame[x].dropna()
            if values.is_monotonic_increasing and values.is_unique:
                return "ordered_continuous"
        except Exception:                                    # noqa: BLE001
            pass
        return "continuous_continuous"
    if left == "categorical":
        return "categorical_continuous"
    return "continuous_only"


def types_for(shape: str) -> Tuple[str, ...]:
    """Return graph types compatible with a data shape.

    :param shape: one of the keys in :data:`DATA_SHAPES`.

    Raises
    ------
    KeyError
        If ``shape`` is unsupported.
    """
    return FITS[str(shape)]


def default_for(shape: str) -> str:
    """The graph type drawn FIRST for ``shape``.

    :param shape: one of the keys of :data:`DATA_SHAPES`.
    :returns: a graph type from :data:`GRAPH_TYPES`.
    :raises KeyError: if ``shape`` is unsupported.

    THE USER'S CHOICE COMES FIRST. The preference decides which compatible
    graph is drawn first, and right-click can still change it afterwards.
    Every graph in spaCR reaches its starting form through this function, so
    the preference applies consistently across screens.

    A SAVED CHOICE THAT DOES NOT FIT THE DATA IS IGNORED. Someone who
    prefers bars has not asked for a bar of a continuous x against a
    continuous y -- that is a different graph of different data, and
    :data:`WHY_NOT` says so. The table's own default is used instead, which
    is what a user who never expressed a preference gets.
    """
    shape = str(shape)
    fallback = DEFAULTS[shape]                    # KeyError for a bad shape
    try:
        from .qt.preferences import get_default_graph_type

        chosen = str(get_default_graph_type(shape) or "")
    except Exception:
        # No Qt, no stored preferences, or a preference file that cannot be
        # read: a figure still has to be drawn.
        return fallback
    return chosen if chosen and fits(shape, chosen) else fallback


def fits(shape: str, graph_type: str) -> bool:
    """Return whether a graph type supports a data shape.

    :param shape: inferred data-shape key.
    :param graph_type: graph implementation key to test.
    """
    return str(graph_type) in FITS.get(str(shape), ())


def why_not(shape: str, graph_type: str) -> str:
    """Explain why a graph type is incompatible with a data shape.

    :param shape: inferred data-shape key.
    :param graph_type: graph implementation key whose mismatch is explained.

    An empty string is returned for compatible pairs.
    """
    if fits(shape, graph_type):
        return ""
    said = WHY_NOT.get((str(shape), str(graph_type)))
    if said:
        return said
    return (f"{graph_type} does not fit "
            f"{dict(DATA_SHAPES).get(str(shape), shape)}")


def offer(frame, x: str = "", y: str = "") -> List[Tuple[str, str, str]]:
    """List graph choices for a frame, with compatible choices first.

    :param frame: data frame whose selected columns determine the shape.
    :param x: horizontal-axis column, or an empty string for an ungrouped
        distribution.
    :param y: vertical-axis measurement column.

    Returns
    -------
    list of tuple
        ``(graph_type, description, incompatibility_reason)`` for every graph
        type. Compatible entries have an empty reason.
    """
    shape = shape_of(frame, x, y)
    captions = dict(GRAPH_TYPES)
    good = [(name, captions[name], "") for name, _ in GRAPH_TYPES
            if fits(shape, name)]
    bad = [(name, captions[name], why_not(shape, name))
           for name, _ in GRAPH_TYPES if not fits(shape, name)]
    return good + bad
