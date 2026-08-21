"""Which graph types fit which data, and what each shape defaults to.

Instruction 200 A and F, built together because they are the same question
asked twice:

    A: "not all graph types fit all types of data"
    F: "which should set the default graph type for each data type"

NOT ALL GRAPH TYPES FIT ALL DATA. A scatter of one categorical against one
continuous is a jitter under another name; a line through unordered
categories is a row of markers joined for no reason -- 178 A already found
that one and fixed it by joining only when there are two points to join.

SO THE LIST IS OFFERED FROM THE DATA. A type the current frame cannot
support is ABSENT or greyed with the reason, and never silently accepted and
drawn as something else.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

LOG = logging.getLogger("spacr.graph_types")

#: Every graph type, with what it shows. The maintainer's own list, in the
#: order they named them.
GRAPH_TYPES: Tuple[Tuple[str, str], ...] = (
    ("bar", "one value per group"),
    ("bar_jitter", "the summary AND the observations"),
    ("jitter", "every observation"),
    ("box", "the distribution's shape, five numbers"),
    ("violin", "the distribution's shape, continuous"),
    ("line", "ordered x only, with or without a spread band"),
    ("scatter", "two continuous axes"),
)

#: The shapes of data a graph can be asked to draw.
#:
#: KEYED ON WHAT THE AXES ARE, not on what the module is. The same frame
#: drawn by two modules is the same question, and a table keyed by module
#: would answer it twice and eventually differently.
DATA_SHAPES: Tuple[Tuple[str, str], ...] = (
    ("categorical_continuous", "groups against a measurement"),
    ("continuous_continuous", "a measurement against a measurement"),
    ("ordered_continuous", "an ordered x against a measurement"),
    ("continuous_only", "one measurement, no grouping"),
)

#: Which types FIT each shape.
FITS: Dict[str, Tuple[str, ...]] = {
    "categorical_continuous": ("bar_jitter", "bar", "jitter", "box",
                               "violin"),
    # NO BAR AND NO BOX HERE. Both need groups to summarise, and forming
    # groups out of a continuous x means binning it -- which is a different
    # graph of different data, not this one drawn another way.
    "continuous_continuous": ("scatter",),
    # A LINE NEEDS AN ORDER. Through unordered categories it is a row of
    # markers joined for no reason, which is why it is here and not above.
    "ordered_continuous": ("line", "scatter", "jitter"),
    "continuous_only": ("jitter", "box", "violin"),
}

#: What each shape is BORN as -- instruction 200 F.
#:
#: `bar_jitter` for groups, because 139 B argues for showing the summary AND
#: the observations: a bar alone hides the spread it was computed from, and
#: a reader cannot tell three points from three hundred.
DEFAULTS: Dict[str, str] = {
    "categorical_continuous": "bar_jitter",
    "continuous_continuous": "scatter",
    "ordered_continuous": "line",
    "continuous_only": "jitter",
}

#: Why a type does not fit, so a greyed entry can say so (instruction 106).
WHY_NOT: Dict[Tuple[str, str], str] = {
    ("continuous_continuous", "bar"):
        "a bar needs groups to summarise, and grouping a continuous x means "
        "binning it -- a different graph of different data",
    ("continuous_continuous", "bar_jitter"):
        "a bar needs groups to summarise, and grouping a continuous x means "
        "binning it -- a different graph of different data",
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
    ("continuous_only", "line"):
        "a line needs an ordered x, and there is only one measurement here",
    ("continuous_only", "scatter"):
        "a scatter needs two axes, and there is only one measurement here",
}


def shape_of(frame, x: str = "", y: str = "") -> str:
    """The data shape of ``frame``'s two axes.

    :param x: the column on the x axis, or ``""`` for none.
    :param y: the column on the y axis.
    :returns: one of :data:`DATA_SHAPES`' keys.
    """
    import pandas as pd

    def kind(name: str) -> str:
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
    """The graph types that fit ``shape``.

    :raises KeyError: for a shape that does not exist. Returning everything
        would offer types that cannot draw the data, which is the failure
        this module exists to prevent.
    """
    return FITS[str(shape)]


def default_for(shape: str) -> str:
    """What ``shape`` is born as (instruction 200 F)."""
    return DEFAULTS[str(shape)]


def fits(shape: str, graph_type: str) -> bool:
    """Whether ``graph_type`` can draw ``shape``."""
    return str(graph_type) in FITS.get(str(shape), ())


def why_not(shape: str, graph_type: str) -> str:
    """Why ``graph_type`` does not fit ``shape``, or ``""`` if it does.

    A REASON, NOT A SILENCE. Instruction 106: a control that is unavailable
    without saying why is a control the user will keep trying.
    """
    if fits(shape, graph_type):
        return ""
    said = WHY_NOT.get((str(shape), str(graph_type)))
    if said:
        return said
    return (f"{graph_type} does not fit "
            f"{dict(DATA_SHAPES).get(str(shape), shape)}")


def offer(frame, x: str = "", y: str = "") -> List[Tuple[str, str, str]]:
    """``[(type, caption, why_not)]`` for this frame, fitting ones first.

    EVERY TYPE IS RETURNED, with the ones that do not fit carrying their
    reason -- greyed rather than absent, because a list that silently
    shortens leaves the user wondering whether they misremembered.
    """
    shape = shape_of(frame, x, y)
    captions = dict(GRAPH_TYPES)
    good = [(name, captions[name], "") for name, _ in GRAPH_TYPES
            if fits(shape, name)]
    bad = [(name, captions[name], why_not(shape, name))
           for name, _ in GRAPH_TYPES if not fits(shape, name)]
    return good + bad
