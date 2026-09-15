"""Match graph types and defaults to the structure of plotted data.

The module distinguishes categorical, continuous, and ordered axes. It lists
compatible graph types first and supplies a reason for each incompatible
option, allowing interfaces to disable unsupported choices without hiding
them.

IT IS ALSO WHERE EVERY GRAPH IN spaCR GETS ITS STARTING FORM. The DEFAULT
GRAPH TYPE setting is stored per data shape (`spacr.qt.preferences`), read
back through :func:`chosen_for`, and turned into "what is drawn before the
first right-click" by :func:`default_for` and :func:`start_for`. A widget
asks one of those two and gets an answer that already accounts for the
user's choice, whether that choice fits the data, and whether the data is
thick enough to support it -- so the setting reaches a new graph by the
graph doing the ordinary thing rather than by remembering to.
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


#: Which MARK on a live pyqtgraph plot draws each graph type.
#:
#: THE TWO VOCABULARIES USE DIFFERENT WORDS FOR THE SAME PICTURE. This module
#: names a graph after the question it answers (``bar_jitter``); the live plot
#: names the mark that draws it (``jitter_bar``), and a plot's menu, its
#: starting form and its right-click all speak the second. The preference is
#: stored in the first, so the translation has to live somewhere, and here --
#: beside the table that decides what fits -- is the one place every consumer
#: already imports.
#:
#: ``spacr.qt.widgets.grouped_plot.MARKS`` is the SAME TABLE, written first
#: and owned by that module; the two agreeing is asserted by
#: `tests/qt/test_every_graph_starts_where_the_setting_says.py`
#: (TestTheTablesAgree) so the copy cannot drift in silence. Collapsing it
#: into this one is left to whoever owns that file.
MARKS: Dict[str, str] = {
    "bar": "bar",
    "bar_jitter": "jitter_bar",
    "box_jitter": "jitter_box",
    "jitter": "jitter",
    "box": "box",
    "violin": "violin",
    "line": "line",
    "scatter": "points",
}

#: Graph types that REPLACE the observations with a summary of them.
#:
#: A bar is one height, a box is five numbers, a violin is a smoothed
#: density and a line is one point per group: none of them show a reader how
#: many observations are behind the mark, so each of them can be drawn over
#: data too thin to support it. The composites (``bar_jitter``,
#: ``box_jitter``) and the point marks keep the observations on the picture,
#: which is why they are not here -- and why they are what a summary falls
#: back TO.
HIDES_THE_OBSERVATIONS: Tuple[str, ...] = ("bar", "box", "violin", "line")

#: At or below this many observations in a group, a summary is a claim the
#: data cannot support.
#:
#: The house rule, stated as a number: with eight or fewer points per group
#: the individual points ARE the figure, a box's quartiles come from a
#: handful of values, and a violin draws a smooth density through points that
#: never described one.
#:
#: ``spacr.qt.widgets.fast_plots.MIN_N_FOR_DISTRIBUTION`` is the same number,
#: where the rule was written down first. It is repeated here because this
#: module is the Qt-free one and the fallback has to work in a headless
#: render; the two are asserted equal by
#: `tests/qt/test_every_graph_starts_where_the_setting_says.py`
#: (TestTheTablesAgree).
MIN_N_FOR_DISTRIBUTION: int = 8


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
    "continuous_continuous": ("scatter",),
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


def chosen_for(shape: str) -> str:
    """The graph type the user asked to see FIRST for ``shape``, or ``""``.

    :param shape: one of the keys of :data:`DATA_SHAPES`.
    :returns: the saved graph type, or an empty string when the user has
        expressed no preference for this shape -- or expressed one this shape
        cannot take.

    EMPTY MEANS "NOTHING WAS CHOSEN", which is not the same answer as the
    table's default: a widget with a starting form of its own keeps it when
    nothing was chosen, and the preference overrides it when something was.
    :func:`default_for` is the caller that turns "nothing" into the table's
    own answer.

    A SAVED CHOICE THAT DOES NOT FIT THE DATA IS NOT A CHOICE FOR IT.
    Someone who prefers bars has not asked for a bar of a continuous x
    against a continuous y -- that is a different graph of different data,
    and :data:`WHY_NOT` says so.

    A MISSING PREFERENCE STORE IS NOT AN ERROR. A headless render has no Qt
    and no QSettings, and a figure still has to be drawn.
    """
    shape = str(shape)
    try:
        from .qt.preferences import get_default_graph_type

        chosen = str(get_default_graph_type(shape) or "")
    except Exception:                                        # noqa: BLE001
        return ""
    return chosen if chosen and fits(shape, chosen) else ""


def default_for(shape: str) -> str:
    """The graph type drawn FIRST for ``shape``.

    :param shape: one of the keys of :data:`DATA_SHAPES`.
    :returns: a graph type from :data:`GRAPH_TYPES`.
    :raises KeyError: if ``shape`` is unsupported.

    THE USER'S CHOICE COMES FIRST. The preference decides which compatible
    graph is drawn first, and right-click can still change it afterwards.
    Every graph in spaCR reaches its starting form through this function or
    through :func:`start_for`, so the preference applies consistently across
    screens.
    """
    shape = str(shape)
    fallback = DEFAULTS[shape]
    return chosen_for(shape) or fallback


def mark_to_start_on(shape: str, fallback_mark: str) -> Tuple[str, str]:
    """What to draw FIRST, in the vocabulary the DRAWING code speaks.

    :param shape: one of the keys of :data:`DATA_SHAPES`.
    :param fallback_mark: what the caller drew BEFORE the setting existed, in
        mark vocabulary -- ``'jitter_box'``, ``'jitter_bar'`` and so on.
    :returns: ``(mark, note)``, the mark to draw and the sentence explaining
        any fallback, empty when the choice was honoured.

    TWO VOCABULARIES MEET HERE AND THEY ARE NOT THE SAME. This module stores
    ``bar_jitter`` and ``box_jitter``; :class:`spacr.plot.spacrGraph` and the
    live panels draw ``jitter_bar`` and ``jitter_box``. A caller that reads
    the setting and forgets :func:`mark_for` hands matplotlib a name that
    ``plot.py``'s own error message lists as unknown -- which is the mistake
    this function exists to stop anyone making twice. Instruction 293 named
    four call sites that had to do this two-step; they should all do it here.

    THE FALLBACK IS THE CALLER'S OWN FORM, NOT THE TABLE'S DEFAULT, and it is
    taken in MARK vocabulary because that is what a caller already has
    written down. A preference nobody expressed must not move an existing
    view, so a caller passing what it used to hardcode gets exactly that back
    until someone chooses otherwise.

    THE FALLBACK IS HANDED TO :func:`start_for` UNTRANSLATED, on purpose.
    It looks like it needs mapping back to graph-type vocabulary first, and
    the first version of this function did that with a reverse lookup over
    :data:`MARKS`. A mutation test could not tell the two versions apart, and
    reading `start_for` says why: with no preference stored it answers the
    fallback AS GIVEN, without validating it, because "a widget knows what it
    can draw". The reverse lookup was dead code.

    IT WOULD STOP BEING DEAD the day a caller passes ``counts``. `start_for`
    uses the fallback a second time on the too-thin-for-a-distribution path,
    where `fits(shape, fallback)` decides between the caller's form and the
    table default -- and a MARK spelling fails `fits`. No caller here has
    counts: these are settings defaults, built before any data is read. If
    one ever does, this function needs the mapping back and a test that
    reaches that path.
    """
    chosen, note = start_for(str(shape), fallback=str(fallback_mark))
    return mark_for(chosen, fallback=str(fallback_mark)), note


def mark_for(graph_type: str, fallback: str = "") -> str:
    """The live-plot mark that draws ``graph_type``.

    :param graph_type: a graph type from :data:`GRAPH_TYPES`.
    :param fallback: what to answer for a name :data:`MARKS` has no entry
        for.
    :returns: a mark key from ``spacr.qt.widgets.fast_plots.MARK_TYPES``.
    """
    return MARKS.get(str(graph_type), str(fallback))


def type_of_mark(mark: str, fallback: str = "") -> str:
    """The graph type a live plot's ``mark`` draws.

    :param mark: a mark key from ``spacr.qt.widgets.fast_plots.MARK_TYPES``.
    :param fallback: what to answer for a mark no graph type draws.
    :returns: a graph type from :data:`GRAPH_TYPES`.

    The inverse of :func:`mark_for`, so a widget whose own starting form is
    written in marks can state it once and still be compared against a
    preference written in graph types.
    """
    wanted = str(mark)
    for graph_type, drawn in MARKS.items():
        if drawn == wanted:
            return graph_type
    return str(fallback)


def too_thin_for(graph_type: str, counts) -> str:
    """Why ``graph_type`` cannot summarise groups this small, or ``""``.

    :param graph_type: the graph type about to be drawn.
    :param counts: observations per group. Unknown sizes -- an empty
        iterable -- are not an objection: a graph that has not been handed
        its data yet is not yet drawing a claim.
    :returns: a sentence naming the smallest group, or an empty string when
        the graph type is supported by the data.

    Only the types that REPLACE the observations can be too thin; see
    :data:`HIDES_THE_OBSERVATIONS`. A jitter of two points is two points,
    which is the truth; a violin of two points is a density that was never
    measured.
    """
    if str(graph_type) not in HIDES_THE_OBSERVATIONS:
        return ""
    try:
        sizes = [int(n) for n in counts if int(n) > 0]
    except Exception:                                        # noqa: BLE001
        return ""
    if not sizes:
        return ""
    smallest = min(sizes)
    if smallest > MIN_N_FOR_DISTRIBUTION:
        return ""
    name = GRAPH_NAMES.get(str(graph_type), str(graph_type))
    return (f"{name} needs more than {MIN_N_FOR_DISTRIBUTION} observations "
            f"in a group and the smallest here has {smallest}")


def start_for(shape: str, counts=(), fallback: str = "") -> Tuple[str, str]:
    """What to draw FIRST, and why that is not what was asked for.

    :param shape: one of the keys of :data:`DATA_SHAPES`.
    :param counts: observations per group, when they are known.
    :param fallback: the caller's OWN starting form, for when the user has
        expressed no preference. It is answered as given -- a widget knows
        what it can draw -- and an empty one means :data:`DEFAULTS`.
    :returns: ``(graph_type, note)``. The note is empty whenever the graph
        being drawn is the one that was asked for.
    :raises KeyError: if ``shape`` is unsupported and no usable fallback was
        given.

    THE FALLBACK IS SAID OUT LOUD. A preference is a blanket statement --
    "draw violins" -- and a blanket statement meets data it cannot describe
    eventually. Drawing the points instead and saying nothing would leave a
    user who asked for violins looking at a jitter with no explanation, so
    the sentence comes back with the answer and the caller puts it on the
    plot.

    THE TABLE'S OWN DEFAULT IS NEVER OVERRIDDEN THIS WAY. Only a saved
    choice is re-examined against the data: :data:`DEFAULTS` already picks
    types that keep the observations, and a line of one point per x -- the
    ordered default -- would otherwise be talked out of itself every time.
    """
    shape = str(shape)
    chosen = chosen_for(shape)
    if not chosen:
        return (str(fallback) or DEFAULTS[shape]), ""
    floor = str(fallback) if fits(shape, str(fallback)) else DEFAULTS[shape]
    thin = too_thin_for(chosen, counts)
    if not thin:
        return chosen, ""
    instead = "jitter" if fits(shape, "jitter") else floor
    if instead == chosen:
        return chosen, ""
    return instead, (f"{thin}, so {GRAPH_NAMES.get(instead, instead)} is "
                     f"drawn instead. Right-click to draw the "
                     f"{GRAPH_NAMES.get(chosen, chosen)} anyway.")


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
