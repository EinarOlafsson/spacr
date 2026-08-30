"""Provide interactive Qt regression plots backed by pyqtgraph.

Pyqtgraph keeps marks in a ``QGraphicsScene`` so Qt can composite pan, zoom,
selection, and hover updates without asking Python to redraw every artist. In
the recorded 1,215-point volcano benchmark, a log-axis update took 4.7 ms and
full point recoloring took 45 ms, compared with about 115 ms for a Matplotlib
redraw.

The application uses these widgets for on-screen interaction and retains
Matplotlib for publication-oriented vector exports. Each plot accepts a
``pandas.DataFrame`` and returns a widget without importing other spaCR modules,
which keeps the components independently testable and reusable.
"""

from __future__ import annotations

import logging
from collections import namedtuple
from contextlib import contextmanager
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

LOG = logging.getLogger(__name__)

_MIN_PDF_RESOLUTION = 72
_MAX_PDF_RESOLUTION = 2400

try:  # exercised by the import guard test
    import pyqtgraph as pg
    from pyqtgraph import ScatterPlotItem
    HAVE_PYQTGRAPH = True
except Exception:  # pragma: no cover - pyqtgraph is optional
    pg = None
    ScatterPlotItem = object
    HAVE_PYQTGRAPH = False


class _Absorbs:
    """Answers any call, any attribute, with something harmless.

    Stands in for BOTH the pyqtgraph module and a PlotWidget when pyqtgraph
    is not installed, so the thirty ``pg.`` calls and the forty
    ``self.plot.`` calls in this file do not each need a guard. A guard per
    call site is how one gets missed on the path nobody tested, and the
    crash comes back somewhere new.

    Every attribute is a callable returning an empty list, because the two
    things callers do with a result are ignore it and iterate it
    (``listDataItems``, ``actions``). Returning None satisfies the first and
    raises on the second.
    """

    def __getattr__(self, _name):
        return self

    def __call__(self, *args, **kwargs):
        return self

    # Chains have to survive whole, not one link at a time. The subclasses
    # run `self.plot.scene().sigMouseClicked.connect(...)` in their own
    # __init__ -- three links -- so returning a plain [] from the first call
    # only moves the AttributeError one step along.
    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __bool__(self):
        # `if self._highlight:` must read as "nothing is drawn", which is
        # true, rather than as a live artist to remove.
        return False

    def __repr__(self):
        return "<pyqtgraph absent>"


if not HAVE_PYQTGRAPH:  # pragma: no cover - exercised by the absence test
    # THE MODULE TOO, not only the widget. Thirty call sites in this file go
    # through `pg.` -- mkBrush, mkPen, ScatterPlotItem, InfiniteLine -- and
    # `pg = None` turns every one into an AttributeError the moment a table
    # arrives. The panel would then BUILD and die on its first redraw, which
    # is a worse failure than the original: the app looks fine until the user
    # loads data.
    pg = _Absorbs()


#: What the user is told, and what they can do about it. Names the EXTRA the
#: way NAPARI_MISSING_MESSAGE does, rather than the bare distribution: a
#: `pip install pyqtgraph` into an environment installed from an extra is how
#: people end up with a package the next upgrade quietly removes again.
PYQTGRAPH_MISSING_MESSAGE = (
    "Interactive plots need pyqtgraph.\n\n"
    "Install it with  pip install 'spacr[qt]'  and reopen this module.\n\n"
    "Everything else works without it: the run still produces every figure, "
    "and they appear on the grid above the console.")

from PySide6.QtCore import QSizeF, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)

#: Colour-blind-safe qualitative order. A screen's categories are nominal, so a
#: sequential map would imply a ranking that is not there.
PALETTE = (
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
    "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD", "#4878CF", "#EE854A",
    "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979",
    "#D5BB67", "#82C6E2",
)

#: The two colours a "one thing against grey" plot needs. THE SAME OBJECTS
#: the saved figure uses, not a second pair chosen to look similar: a run
#: must not draw in two idioms, and a compartment that is blue on screen and
#: amber in the exported PDF is exactly that failure in miniature.
#:
#: `spacr.figures.style` is hex strings and imports no matplotlib -- measured
#: at 76 ms with matplotlib still absent from sys.modules -- so this costs a
#: GUI module nothing.
from ...figures.style import ROLES as _ROLES
from .sortable_table import install_sorting, table_item

HIGHLIGHT = _ROLES["highlight"]
MUTED = _ROLES["data"]
#: A called effect, by direction. The same two hues the saved volcano and the
#: saved effect-rank panel use, for the reason above: one run, one idiom.
UP = _ROLES["up"]
DOWN = _ROLES["down"]
#: Thresholds, limits and 1:1 lines. Darker than :data:`MUTED`, so a
#: reference line is tellable from the data it is drawn over.
REFERENCE = _ROLES["reference"]
#: Histogram and density fills, again the saved figure's own.
FILL = _ROLES["fill"]

#: The standard error, however this backend spelled it. spaCR's own writer and
#: a statsmodels summary disagree (``std_err`` against ``std err``), and the
#: penalised backends report none at all -- which is not a failure but a fact
#: about the fit, and is said rather than drawn as a zero-width interval.
ERROR_COLUMNS = ("std_err", "std err", "bse", "se", "standard_error")

#: What decides whether a coefficient is CALLED. Corrected p-values only, and
#: deliberately: :func:`spacr.figures.panels.effect_rank` colours on a q and
#: nothing else, because calling hits off an uncorrected p across a thousand
#: tests is the multiple-testing error a screen panel exists to make visible.
#: A plot that coloured on the raw p would disagree with the figure the same
#: run writes to disk.
CORRECTED_P_COLUMNS = ("q_value", "adjusted_p_value", "fdr", "qval")

#: Passed as ``significance_column`` to say "this table HAS none", which is
#: not the same request as "go and look for one".
#:
#: ``spacr.ml`` writes an OLS-style p-value into a lasso ``results.csv`` --
#: computed as though there were no penalty, which is why
#: :data:`spacr.hits.NO_P_VALUE_TYPES` exists -- so a plot left to search for
#: a significance column on a penalised fit would colour its dots by a number
#: nobody tested. The caller knows which backend it has; the plot does not.
NO_SIGNIFICANCE = "\0no significance"

#: Points beyond this many stop getting individual hover hit-boxes, which is
#: what makes a large scatter slow to move over rather than slow to draw.
HOVER_LIMIT = 20000

#: How wide a plot is rendered for :meth:`FastPlot.snapshot`. Big enough that
#: the axes and the point cloud survive being scaled down into a grid cell; a
#: tile is read, not merely recognised. The height follows the plot's aspect.
SNAPSHOT_PX = (520, 380)

#: The marks a plot with a categorical x-axis can be drawn with, and what each
#: says on the menu.
#:
#: ORDERED BY HOW MUCH THEY HIDE, honest first. ``points`` shows every
#: observation against a mean line; ``jitter`` is the same thing spread
#: sideways so overlapping values stay countable; ``box`` replaces the
#: observations with five numbers; ``violin`` replaces them with a smoothed
#: density; ``bar`` replaces them with one number and a rectangle whose area
#: means nothing.
#: Composite marks retain the observations alongside their summary. For
#: example, ``jitter_box`` overlays individual points on the box plot so users
#: can inspect both the distribution and its compact statistical summary.
MARK_TYPES = (
    ("points", "Points with a mean line"),
    ("line", "Means joined by a line"),
    ("bar", "Bar chart"),
    ("jitter_bar", "Bar chart with jittered points"),
    ("jitter_box", "Box plot with jittered points"),
    ("jitter", "Jittered points"),
    ("box", "Box plot"),
    ("violin", "Violin plot"),
)

#: At or below this many observations in a group, a summarising mark is a
#: claim the data does not support. The house rule, stated as a number: with
#: eight or fewer points per group the individual points ARE the figure, a box
#: plot's quartiles come from a handful of values, and a violin draws a smooth
#: density through points that never described one.
MIN_N_FOR_DISTRIBUTION = 8

#: The colour scales offered for "colour the points by a numeric column".
#:
#: PERCEPTUALLY UNIFORM ONLY, and pyqtgraph ships all five itself -- no
#: matplotlib import, which this module is careful never to make. A jet or a
#: rainbow puts bright bands where the data has none and reads as structure;
#: these five do not, which is the whole reason a cmap is allowed on a
#: continuous quantity while the categorical palette stays nominal.
COLORMAPS = ("viridis", "plasma", "inferno", "magma", "cividis")

#: How many steps a colour scale is quantised to before brushes are built.
#:
#: 256 is the colormap's OWN resolution -- pyqtgraph builds them as
#: ``ColorMap(256)`` -- so this loses nothing and caps the brush count at 256
#: instead of one per point. Measured on the volcano when this module was
#: written: a brush per point costs 39.5 ms against 3.5 ms for a reused set.
COLORMAP_STEPS = 256

#: What a row with no value in the mapped column is drawn as. Grey, and it is
#: SAID in the status line rather than left to look like a low value -- a NaN
#: painted at the bottom of a viridis scale is a made-up measurement.
MISSING_COLOUR = MUTED

#: ``(pyqtgraph symbol, what it is called)`` for "shape the points by a
#: column". Ordered by how easily one is told from the ones before it at
#: scatter-plot size; a ninth shape would be a circle a reader has to squint
#: at, which is why the list ends and a column with more values is refused.
SHAPE_SYMBOLS = (
    ("o", "circle"), ("s", "square"), ("t", "triangle"), ("d", "diamond"),
    ("+", "plus"), ("t1", "triangle up"), ("p", "pentagon"), ("star", "star"),
)

#: The most distinct values a column can have and still be drawn as shapes.
MAX_SHAPE_VALUES = len(SHAPE_SYMBOLS)

#: How wide a saved page is, in millimetres: a journal's double-column width.
EXPORT_WIDTH_MM = 180.0

#: The shape of the CANVAS -- the thing that gets exported -- as
#: ``(name, height / width)``. ``None`` is "whatever the box it sits in is".
#:
#: NOT THE ASPECT LOCK. That one ties one y unit to n x units and is a
#: statement about the DATA, which is what a Q-Q wants and is not what "save
#: it as a square" means. Canvas shape and data aspect are controlled
#: independently.
CANVAS_SHAPES = (
    ("square", 1.0),
    ("wide", 2.0 / 3.0),
    ("tall", 1.5),
    ("free", None),
)

#: What each stored shape is CALLED where a user meets it. A ratio is a
#: number and this is a choice of three, so the menu says which shape it is
#: drawing rather than asking for a figure the reader has to compute. The
#: stored names are unchanged: they are what `set_canvas_shape` takes, what a
#: saved preference holds, and renaming them would break every one of those.
CANVAS_SHAPE_LABELS = {
    "square": "square",
    "wide": "horizontal rectangle",
    "tall": "vertical rectangle",
    "free": "free",
}

#: What the group holding them is called. "Graph shape", not "aspect ratio":
#: the two are different quantities and only one of them is a ratio -- the
#: axis LOCK, which lives under Axes and says so in its own name.
GRAPH_SHAPE_MENU = "Graph shape"

#: Qt's own "no maximum", which PySide6 does not re-export from QtWidgets --
#: checked: ``from PySide6.QtWidgets import QWIDGETSIZE_MAX`` raises
#: ImportError on 6.11.1. Needed to give a widget its stretch back after a
#: fixed size has been imposed on it.
QWIDGET_SIZE_MAX = (1 << 24) - 1


# One positionable thing on a plot, with the DATA coordinates it was handed.
# Private on purpose: it is bookkeeping for the log transform, not a contract
# anyone outside this module holds.
#
#   x, y     float arrays -- or None where the item does not move on that
#            axis, as a horizontal reference line does not move in x.
#   blocks   {axis: why this item can NEVER be logged on that axis}
#   kind     "points", "bar", or "line" -- so a refusal can name what is in
#            the way rather than quoting a count of anonymous "values".
#   counts   {axis: (at or below zero, finite in total, the lowest of them)},
#            measured once at registration rather than re-measured every time
#            a menu opens.
_Drawn = namedtuple("_Drawn", "item x y blocks kind counts")


def menu_entries(menu) -> list:
    """Every action a user can actually TRIGGER on ``menu``, submenus included.

    THE MIGRATION THIS EXISTS FOR. ``QMenu.actions()`` returns a SUBMENU'S OWN
    action and not what is inside it, so sixty-odd assertions written against
    a flat menu read every grouped entry as a removed feature -- which is what
    reverted the first attempt at the design. Asserting REACHABILITY
    instead of depth makes those assertions true of the flat menu and of the
    grouped one alike, so the restructure stops being a rewrite of the suite.

    Two things are skipped because a user cannot do them:

    * separators, which are not entries at all;
    * ``addSection`` labels, which Qt implements as a separator that HAS text
      -- so ``isSeparator()`` catches both, and a naive text filter would
      count a heading as a feature.

    :param menu: a ``QMenu``.
    :returns: the ``QAction`` objects, in the order a reader meets them, with each
        submenu's contents spliced in where the submenu sits.
    """
    found = []
    for action in menu.actions():
        submenu = action.menu()
        if submenu is not None:
            found.extend(menu_entries(submenu))
        elif not action.isSeparator():
            found.append(action)
    return found


def menu_groups(menu) -> list:
    """The names a reader sees dividing ``menu`` into parts, in order.

    An ``addSection`` heading and a submenu title are the same idea to a
    reader and two different objects to Qt -- which is exactly why a test
    that names one cannot be written against the other, and why moving a
    heading into a submenu title read as a deleted feature the first time
    the design was attempted.

    :param menu: a ``QMenu``.
    """
    names = []
    for action in menu.actions():
        submenu = action.menu()
        if submenu is not None:
            names.append(action.text())
            names.extend(menu_groups(submenu))
        elif action.isSeparator() and action.text():
            names.append(action.text())
    return names


def menu_reading_order(menu) -> list:
    """``menu`` as a reader meets it: entry texts, ``"|"`` at every boundary.

    A separator, a section heading and the edge of a submenu are all one
    thing to the person reading the menu -- a break -- and are three
    different things to Qt. This flattens them into the same mark, so
    "these two entries are kept apart" is a claim that survives being
    reorganised into submenus.

    :param menu: a ``QMenu``.
    """
    order = []
    for action in menu.actions():
        submenu = action.menu()
        if submenu is not None:
            order.append("|")
            order.extend(menu_reading_order(submenu))
            order.append("|")
        elif action.isSeparator():
            order.append("|")
        else:
            order.append(action.text())
    return order


# Which group of the right-click menu a style field belongs on, as
# ``(group, name fragments)`` tried in order; anything unmatched is
# Appearance. A plain module constant rather than a documented one, because
# it is a layout table for this module's menu and not a contract anyone
# outside it holds.
#
# ORDER IS THE ORDER OF USE, the same as `build_style_menu`: what changes what
# the figure CLAIMS, then its axes, then how it looks, then how big it is.
_STYLE_GROUPS = (
    ("Data", ("column", "threshold", "alpha", "control", "annotat",
              "label_top", "significant")),
    ("Axes", ("scale", "_lim", "invert", "split", "x_label", "y_label",
              "title", "grid", "spine", "log")),
    ("Size", ("figure_", "dpi")),
    ("Appearance", ()),
)

#: The file dialog filter for a saved style. JSON, because the serialisation
#: is `dataclasses.asdict` and a style a user can open in a text editor is one
#: they can fix when a field name changes under them.
STYLE_FILE_FILTER = "spaCR figure style (*.json);;All files (*)"

# What a style field can be edited WITH. A field whose kind is "unsupported"
# still gets an entry, greyed and saying so: a setting silently absent from
# the menu is one the user is told exists and cannot find.
STYLE_FIELD_KINDS = ("flag", "colour", "choice", "multi", "number", "pair",
                     "text", "unsupported")


def style_field_kind(name: str, value, choices=None, declared: str = "") -> str:
    """Return the editor type for a figure-style field.

    Parameters
    ----------
    name : str
        Field name. Colour-related suffixes select the colour editor.
    value : Any
        Current value. Its runtime type takes precedence when it is not
        ``None``.
    choices : collection, optional
        Allowed values for a closed selection.
    declared : str, optional
        Type annotation used when ``value`` is ``None``.

    Returns
    -------
    str
        One of :data:`STYLE_FIELD_KINDS`.
    """
    written = str(declared or "").replace(" ", "").lower()
    if choices:
        # A CLOSED SET HELD IN A CONTAINER IS TICKED, NOT PICKED. "Dense
        # granules and rhoptries 1" is one question rather than two, so a
        # field whose value is a tuple of members of the offered list gets a
        # submenu where any number of them can be on at once.
        if (isinstance(value, (tuple, list))
                or "tuple[str" in written or "list[str" in written
                or "sequence[str" in written):
            return "multi"
        return "choice"
    if isinstance(value, bool):
        return "flag"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "colour" if str(name).endswith(("color", "colour")) else "text"
    if isinstance(value, (tuple, list)):
        return ("pair" if len(value) == 2
                and all(isinstance(part, (int, float)) for part in value)
                else "unsupported")
    if value is not None:
        return "unsupported"
    if "tuple[tuple" in written or "dict" in written or "list[" in written:
        # A NESTED SHAPE HAS NO DIALOG. Offering the split axis's pair of
        # pairs as one pair of numbers would write a value the renderer
        # cannot read -- worse than saying it is not editable here.
        return "unsupported"
    if written.startswith("bool") or "bool|" in written:
        return "flag"
    if "tuple" in written:
        return "pair"
    if "float" in written or written.startswith("int") or "int|" in written:
        return "number"
    if "str" in written:
        return "colour" if str(name).endswith(("color", "colour")) else "text"
    # Nothing declared and nothing held: the name is the only clue left.
    if str(name).endswith(("_lim", "_lims")):
        return "pair"
    if str(name).endswith(("color", "colour")):
        return "colour"
    return "text"


def style_field_choices(style, name: str, choices=None):
    """The closed set ``name`` may take, or ``()``.

    Looked for in three places, most specific first: the argument, the
    field's own ``metadata["choices"]``, and a ``CHOICES`` mapping on the
    style's class. Three, because the styles in this package declare their
    sets in all three ways and a mechanism that read only one would silently
    turn a closed set into a free-text box.
    """
    import dataclasses

    if choices and name in choices:
        return tuple(choices[name])
    for entry in dataclasses.fields(style):
        if entry.name == name:
            declared = entry.metadata.get("choices")
            if declared:
                return tuple(declared)
            break
    declared = getattr(type(style), "CHOICES", {})
    if isinstance(declared, dict) and name in declared:
        return tuple(declared[name])
    return ()


def style_field_group(name: str) -> str:
    """Which menu group ``name`` belongs on."""
    lowered = str(name).lower()
    for group, fragments in _STYLE_GROUPS:
        if any(fragment in lowered for fragment in fragments):
            return group
    return "Appearance"


def style_field_label(name: str, value, kind: str) -> str:
    """What the entry for ``name`` reads.

    The CURRENT VALUE is in the label for everything but a flag, which shows
    its state as a tick. A menu of settings that does not say what they are
    set to is one the user has to open each entry to read.
    """
    pretty = str(name).replace("_", " ").strip().capitalize()
    if kind == "flag":
        return pretty
    if kind == "unsupported":
        return pretty
    if kind == "multi":
        # WHAT IS TICKED, spelled out. A multi-select whose label said only
        # its name would make "is anything selected?" a question the reader
        # has to open the submenu to answer.
        chosen = [str(item) for item in (value or ())]
        return f"{pretty}: {', '.join(chosen)}" if chosen else f"{pretty}: none"
    if value is None:
        return f"{pretty}: automatic…"
    if kind == "pair":
        return f"{pretty}: {value[0]:g} to {value[1]:g}…"
    if kind == "number":
        return f"{pretty}: {value:g}…"
    return f"{pretty}: {value}…"


def add_style_entries(menu, style, on_change=None, *, choices=None,
                      labels=None) -> list:
    """Put EVERY field of ``style`` onto ``menu``, grouped and editable.

    The point 3: the menu is built FROM THE STYLE OBJECT'S
    FIELDS rather than from a hand-written list per figure, so "as many
    settings as possible, depending on the graph" is automatic -- a style
    gains a field, the menu gains an entry, and the two cannot fall out of
    step. The acceptance test compares what this produces against
    ``dataclasses.fields(style)``, which is only a meaningful check because
    NOTHING IS SKIPPED: a field this cannot edit is still listed, greyed, and
    says why, exactly as the design requires of every other control here.

    :param menu: a ``QMenu`` to add groups to.
    :param style: any dataclass instance describing how a figure looks.
    :param on_change: called ``(name, value)`` when the user changes one.
        Without it the entries are built and inert, which is what a test
        reading the menu wants and what a caller with nothing to redraw gets.
    :param choices: ``{field: values}`` for fields that are a closed set,
        overriding the field's own metadata and the style class's ``CHOICES``.
    :param labels: ``{field: {value: what to call it}}``, for a closed set
        whose stored values are not what a reader should be shown -- a marker
        is stored as ``"o"`` and read as "Circle". Anything unnamed shows its
        stored value, so a partial map is fine.
    :returns: the ``QAction`` objects added, in menu order.

    The groups match :func:`build_style_menu` and use the same order --
    Data, Axes, Appearance, Size -- so a figure's own settings and the plot's
    read as one menu rather than two conventions side by side.

    ``menu`` MUST OUTLIVE THE CALL. The groups are parented to it, but a
    ``QMenu()`` built with no parent of its own is Python-owned and takes
    every action here with it when the local holding it goes out of scope.
    :meth:`FastPlot.build_style_menu` parents its menu to the widget, which
    is why the application does not meet this.
    """
    import dataclasses
    from PySide6.QtWidgets import QMenu

    entries = dataclasses.fields(style)
    groups: dict = {}
    added = []
    for name in [group for group, _ in _STYLE_GROUPS]:
        wanted = [entry for entry in entries
                  if style_field_group(entry.name) == name]
        if not wanted:
            continue
        submenu = QMenu(name, menu)
        submenu.setToolTipsVisible(True)
        menu.addMenu(submenu)
        groups[name] = submenu
        for entry in wanted:
            added.append(_add_style_entry(submenu, style, entry.name,
                                          on_change, choices, labels))
    return added


def style_kind(style) -> str:
    """A stable name for the KIND of style ``style`` is.

    ``VolcanoStyle`` -> ``"volcano"``. It is what a saved default is keyed on,
    so it has to be derived from the class rather than passed in: a caller
    that had to name its own kind would eventually name two of them the same
    and one lab's house style would land on another figure type.
    """
    name = type(style).__name__
    if name.endswith("Style"):
        name = name[:-len("Style")]
    out = []
    for index, character in enumerate(name):
        if character.isupper() and index:
            out.append("_")
        out.append(character.lower())
    return "".join(out) or "figure"


def style_as_dict(style) -> dict:
    """``style`` as plain JSON-able data.

    ``dataclasses.asdict`` recurses and turns tuples into lists, which is what
    JSON does anyway -- so a pair field round-trips as a list and
    :func:`apply_style_dict` puts it back as a tuple rather than leaving two
    representations of one value in the store.
    """
    import dataclasses

    return dataclasses.asdict(style)


def apply_style_dict(style, values, on_change=None) -> list:
    """Write ``values`` into ``style``. Returns the field names that changed.

    FORWARDS-COMPATIBLE LOADING, which is the half a bare ``**values`` would
    lose: a style file written by a later spaCR carries fields this one has
    never heard of, and a file written by an earlier one is missing some. An
    unknown field is skipped rather than raising, and a missing one keeps
    whatever the style already had -- so a house style saved today is still
    loadable after the dataclass grows.

    ``on_change`` is called ONCE, with ``(None, style)``, rather than per
    field: a host that redraws per field would redraw sixty times for one
    load, and the interesting question for the host is "the whole style
    changed", not "line_width did".
    """
    import dataclasses

    known = {entry.name for entry in dataclasses.fields(style)}
    changed = []
    for name, value in dict(values or {}).items():
        if name not in known:
            continue
        current = getattr(style, name, None)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        if value != current:
            setattr(style, name, value)
            changed.append(name)
    if changed and on_change is not None:
        on_change(None, style)
    return changed


def save_style(style, path) -> str:
    """Write ``style`` to ``path`` as JSON. Returns the path written.

    The serialisation already existed
    (``VolcanoStyle.from_dict`` / ``asdict``); what did not was any way for a
    user to reach it, which made a restyle something they redid every time
    they needed the picture.
    """
    import json
    from pathlib import Path

    target = Path(str(path))
    if not target.suffix:
        target = target.with_suffix(".json")
    payload = {"spacr_style_kind": style_kind(style),
               "fields": style_as_dict(style)}
    target.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                 default=str))
    return str(target)


def load_style(style, path, on_change=None) -> list:
    """Read a saved style into ``style``. Returns the fields that changed.

    :raises ValueError: for a file that is not a style, and for a style of
        the WRONG KIND. Refused rather than partially applied: a volcano's
        style loaded into a heatmap would set the four fields whose names
        happen to match and leave the rest, which looks like a corrupted
        figure rather than like a mistake.
    """
    import json
    from pathlib import Path

    try:
        payload = json.loads(Path(str(path)).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path} is not a readable style file: {exc}") from exc
    if not isinstance(payload, dict) or "fields" not in payload:
        raise ValueError(f"{path} is not a spaCR style file")
    saved = str(payload.get("spacr_style_kind", ""))
    mine = style_kind(style)
    if saved and saved != mine:
        raise ValueError(
            f"that is a {saved} style and this figure is a {mine} one")
    return apply_style_dict(style, payload.get("fields"), on_change)


def add_style_file_entries(menu, style, on_change=None, *, parent=None,
                           note=None, ask_path=None) -> list:
    """Save, load and default a whole style -- the design.

    :param menu: the "Figure style" group to add to.
    :param style: the style dataclass the figure is drawn from.
    :param on_change: called ``(None, style)`` when a load or a default
        changes it, i.e. where the host redraws.
    :param parent: the widget file dialogs are parented to.
    :param note: called with one sentence saying what happened, for the
        plot's status line. A save that says nothing is a save the user
        repeats because they cannot tell whether it worked.
    :param ask_path: ``(mode, suggested) -> path`` for a test to answer
        instead of a modal. ``None`` uses ``QFileDialog``.
    :returns: the actions added, in menu order.

    THE DEFAULT IS PER STYLE KIND, not per figure. That is what makes it a
    house style: every volcano this project draws from now on starts from it,
    which is the difference between saving a style and re-picking one.
    """
    from PySide6.QtWidgets import QFileDialog

    kind = style_kind(style)
    added = []

    def _say(message: str) -> None:
        if note is not None:
            note(message)

    def _ask(mode: str, suggested: str) -> str:
        if ask_path is not None:
            return str(ask_path(mode, suggested) or "")
        if mode == "save":
            path, _ = QFileDialog.getSaveFileName(
                parent, "Save this figure's style", suggested,
                STYLE_FILE_FILTER, options=QFileDialog.DontUseNativeDialog)
        else:
            path, _ = QFileDialog.getOpenFileName(
                parent, "Load a figure style", suggested,
                STYLE_FILE_FILTER, options=QFileDialog.DontUseNativeDialog)
        return str(path or "")

    def _save() -> None:
        path = _ask("save", f"{kind}_style.json")
        if not path:
            return
        try:
            written = save_style(style, path)
        except OSError as exc:
            _say(f"Could not write the style: {exc}")
            return
        _say(f"Saved this {kind} style to {written}.")

    def _load() -> None:
        path = _ask("load", f"{kind}_style.json")
        if not path:
            return
        try:
            changed = load_style(style, path, on_change)
        except ValueError as exc:
            # NAMED, not swallowed. The two ways this fails -- a file that is
            # not a style and a style of another kind -- are both things the
            # user can act on, and a menu entry that silently does nothing is
            # the one failure this module keeps being written to avoid.
            _say(f"Could not load that style: {exc}")
            return
        _say(f"Loaded {len(changed)} setting{'s' if len(changed) != 1 else ''}"
             f" from {path}." if changed else
             f"That style is identical to this figure's; nothing changed.")

    def _make_default() -> None:
        try:
            from ..preferences import set_figure_style_default

            set_figure_style_default(kind, style_as_dict(style))
        except Exception:       # no settings store
            _say("There is no settings store to save a default into.")
            return
        _say(f"Every {kind} figure from now on starts from this style. "
             f"'Clear the default' puts it back.")

    def _clear_default() -> None:
        try:
            from ..preferences import clear_figure_style_default

            cleared = clear_figure_style_default(kind)
        except Exception:       # no settings store
            return
        _say(f"The saved {kind} default is gone; new figures use the "
             f"package's own." if cleared else
             f"There was no saved {kind} default.")

    added.append(menu.addAction("Save style…", _save))
    added[-1].setToolTip(
        "Write every setting on this figure to a file, so the same look can "
        "be put back on another figure of the same kind.")
    added.append(menu.addAction("Load style…", _load))
    added[-1].setToolTip(
        "Read a saved style into this figure. A style of another kind is "
        "refused rather than half applied.")
    added.append(menu.addAction(f"Use as the default for every {kind}",
                                _make_default))
    added[-1].setToolTip(
        "This project's house style for this kind of figure. Every one drawn "
        "from now on starts from it.")
    has_default = False
    try:
        from ..preferences import get_figure_style_default

        has_default = bool(get_figure_style_default(kind))
    except Exception:           # no settings store
        pass
    action = menu.addAction("Clear the default", _clear_default)
    # Greyed when there is nothing to clear (106), and it doubles as the
    # readout for "is a house style in force here?", which is the question a
    # user arrives with when a figure does not look like the package's.
    action.setEnabled(has_default)
    if not has_default:
        action.setToolTip(f"No {kind} default is saved.")
    added.append(action)
    return added


def apply_default_style(style, on_change=None) -> list:
    """Start ``style`` from this project's saved default. Returns the fields
    it changed, or ``[]`` when there is no default for this kind.

    Called by the HOST before it draws, not by the menu: only the host knows
    when a figure is new, and applying a default to a style the user has
    already edited would undo their edits at the next redraw -- the same
    mistake as a host that re-asserts an axis choice, which cost this module
    a day.
    """
    try:
        from ..preferences import get_figure_style_default

        saved = get_figure_style_default(style_kind(style))
    except Exception:           # no settings store
        return []
    return apply_style_dict(style, saved, on_change) if saved else []


def _add_style_entry(menu, style, name: str, on_change, choices, labels=None):
    """One field of ``style`` as one entry on ``menu``."""
    from PySide6.QtWidgets import QMenu

    import dataclasses

    value = getattr(style, name, None)
    options = style_field_choices(style, name, choices)
    declared = next((str(entry.type) for entry in dataclasses.fields(style)
                     if entry.name == name), "")
    kind = style_field_kind(name, value, options, declared)
    # WHAT A VALUE IS CALLED, where that is not what it is. A marker stored
    # as "o" and a line style stored as "--" are what a saved style file
    # says and are not what a reader picks off a menu.
    named = dict((labels or {}).get(name) or {})

    def _shown(option):
        if option in named:
            return str(named[option])
        return "automatic" if option is None else str(option)

    if kind == "multi":
        label = style_field_label(name, [_shown(item) for item in
                                         (value or ())], kind)
    elif kind == "choice" and value in named:
        label = style_field_label(name, named[value], kind)
    else:
        label = style_field_label(name, value, kind)
    if kind == "unsupported":
        action = menu.addAction(
            f"{label}  —  this setting is not one the menu can edit")
        action.setEnabled(False)
        action.setToolTip("It is kept in the style and written to a saved "
                          "style file; it is changed from the plot itself.")
        action.setObjectName(name)
        return action
    if kind in ("choice", "multi"):
        submenu = QMenu(label.rstrip("…"), menu)
        submenu.setToolTipsVisible(True)
        menu.addMenu(submenu)
        chosen = tuple(value or ()) if kind == "multi" else ()
        for option in options:
            # `None` is a real option -- it is how a colour-by column is taken
            # back off -- and "None" is not what that reads as on a menu.
            entry = submenu.addAction(_shown(option))
            entry.setCheckable(True)
            if kind == "multi":
                entry.setChecked(option in chosen)
                entry.toggled.connect(
                    lambda on, picked=option:
                    _toggle_style_member(style, name, picked, on, options,
                                         on_change))
            else:
                entry.setChecked(option == value)
                entry.triggered.connect(
                    lambda _checked=False, picked=option:
                    _apply_style(style, name, picked, on_change))
        # The GROUP's action is what a reader meets, so it carries the name.
        action = submenu.menuAction()
        action.setObjectName(name)
        return action
    action = menu.addAction(label)
    action.setObjectName(name)
    if kind == "flag":
        action.setCheckable(True)
        action.setChecked(bool(value))
        action.toggled.connect(
            lambda on: _apply_style(style, name, bool(on), on_change))
        return action
    action.triggered.connect(
        lambda _checked=False: _ask_style_value(menu, style, name, value,
                                                kind, on_change))
    return action


def style_menu_fields(menu) -> set:
    """The style FIELDS a built menu offers, by name.

    What a reader meets is prose -- "Marker size: 26…" -- so the field's own
    name travels on its action as the object name. That is what lets "the
    right-click menu and the side panel offer the same settings" be asserted
    as a comparison of two sets rather than by reading two lists of words and
    hoping they mean the same thing.

    :param menu: a ``QMenu`` that :func:`add_style_entries` has been called on.
    """
    found = set()
    for action in menu.actions():
        name = action.objectName()
        if name:
            found.add(name)
        submenu = action.menu()
        if submenu is not None:
            found |= style_menu_fields(submenu)
    return found


def _toggle_style_member(style, name, option, on, options, on_change) -> None:
    """Tick or untick one member of a multi-select field.

    KEPT IN THE OFFERED ORDER rather than the order they were ticked, so one
    combination is one value however a reader arrived at it -- which is what
    makes a saved style redraw the figure it was saved from. A member the
    menu does not offer (a compartment this screen has none of, loaded from a
    style file) is not silently dropped; it sorts to the end and stays.
    """
    kept = list(getattr(style, name, ()) or ())
    if on and option not in kept:
        kept.append(option)
    elif not on and option in kept:
        kept.remove(option)
    order = list(options)
    rank = {value: index for index, value in enumerate(order)}
    _apply_style(style, name,
                 tuple(sorted(kept, key=lambda item: rank.get(item,
                                                              len(order)))),
                 on_change)


def _apply_style(style, name: str, value, on_change) -> None:
    """Write one field and tell whoever is drawing from it."""
    setattr(style, name, value)
    if on_change is not None:
        on_change(name, value)


def _ask_style_value(parent, style, name, value, kind, on_change) -> None:
    """Ask for one field's new value with the dialog its kind wants."""
    from PySide6.QtWidgets import QInputDialog

    pretty = str(name).replace("_", " ").strip().capitalize()
    if kind == "colour":
        # A SEVENTH CALL SITE, which is the reason `pick_colour` exists.
        # Instruction 151 counted six unguarded `QColorDialog.getColor` calls
        # in the tree; this one -- the figure style's own colour fields -- was
        # not among them, so the count was low and the flag would have been
        # forgotten here even after the six were fixed.
        colour = pick_colour(parent, value or "#000000", pretty)
        if colour.isValid():
            _apply_style(style, name, colour.name(), on_change)
        return
    if kind == "number":
        decimals = 0 if isinstance(value, int) else 3
        new, ok = QInputDialog.getDouble(parent, pretty, f"{pretty}:",
                                         float(value or 0.0), -1e12, 1e12,
                                         decimals)
        if ok:
            _apply_style(style, name,
                         int(round(new)) if isinstance(value, int) else new,
                         on_change)
        return
    if kind == "pair":
        low, high = (value if value is not None else (0.0, 1.0))
        first, ok = QInputDialog.getDouble(parent, pretty, f"{pretty} from:",
                                           float(low), -1e12, 1e12, 4)
        if not ok:
            return
        second, ok = QInputDialog.getDouble(parent, pretty, f"{pretty} to:",
                                            float(high), -1e12, 1e12, 4)
        if ok:
            # CANCELLING THE SECOND ABANDONS THE FIRST, the same rule the
            # axis-limit dialog follows: a half-set pair is a range nobody
            # chose.
            _apply_style(style, name, (first, second), on_change)
        return
    text, ok = QInputDialog.getText(parent, pretty, f"{pretty}:",
                                    text="" if value is None else str(value))
    if ok:
        _apply_style(style, name, text or None, on_change)


def mark_advice(kind: str, counts) -> str:
    """Explain when a summary mark is poorly supported by group size.

    Parameters
    ----------
    kind : str
        Mark key from :data:`MARK_TYPES`.
    counts : iterable of int
        Number of observations in each group.

    Returns
    -------
    str
        Decision-ready warning, or an empty string when no warning is needed.
    """
    sizes = [int(n) for n in counts if int(n) > 0]
    if not sizes or kind in ("points", "jitter", "jitter_box", "jitter_bar"):
        return ""
    smallest = min(sizes)
    if smallest > MIN_N_FOR_DISTRIBUTION:
        return ""
    thin = sum(1 for n in sizes if n <= MIN_N_FOR_DISTRIBUTION)
    where = (f"the smallest group has {smallest}" if thin == 1 else
             f"{thin} of {len(sizes)} groups have "
             f"{MIN_N_FOR_DISTRIBUTION} or fewer, the smallest {smallest}")
    if kind == "bar":
        return (f"A bar hides every observation behind one height, and "
                f"{where} -- the points themselves are the honest mark here.")
    if kind == "box":
        return (f"A box plot hides n, and {where}: these quartiles are "
                f"computed from a handful of values.")
    if kind == "line":
        return (f"A line shows one summary per group and hides the underlying "
                f"observations; {where}.")
    return (f"A violin draws a density that is not there -- {where}, which "
            f"is too few to have a shape.")


def _require_pyqtgraph() -> None:
    """Raise when a caller requires the optional pyqtgraph dependency.

    General plot construction uses an unavailable-state widget instead;
    callers should use this guard only when no graceful fallback exists.
    """
    if not HAVE_PYQTGRAPH:
        raise RuntimeError(PYQTGRAPH_MISSING_MESSAGE)


#: Ink to fall back to when the preference store cannot be read.
#:
#: White, because every spaCR theme but one is dark and a plot with invisible
#: axes is worse than a plot with slightly wrong ones. Only reached in a bare
#: process with no QSettings -- a headless render or a unit test.
_FALLBACK_FOREGROUND = "#ffffff"


def _figure_colors() -> tuple:
    """``(background, foreground)`` for a plot, from the figure preferences.

    The same source the matplotlib renderer uses
    (:func:`spacr.qt.preferences.get_figure_colors`), so the two cannot
    disagree about what a figure looks like and a theme switch moves both.
    """
    try:
        from ..preferences import get_figure_colors

        return get_figure_colors()
    except Exception:      # no settings store available
        return "none", _FALLBACK_FOREGROUND


def pick_colour(parent, initial=None, title: str = "Colour"):
    """Ask for a colour with QT'S OWN dialog. Re-exported, not implemented.

    The implementation is
    :func:`spacr.qt.widgets.colour_picker.pick_colour`, and this module goes
    through it rather than keeping a second copy. ``QColorDialog.getColor``
    defaults to the PLATFORM's chooser, and on a GNOME session that request
    is brokered through ``xdg-desktop-portal`` -- the tens-of-seconds stall
    behind slow native colour dialogs. One helper, because an option that has to be
    remembered at each call site is one that gets forgotten at the seventh --
    and there WERE seven here: the six the design counted plus
    :func:`_ask_style_value`, which its count missed.

    Imported inside the function because this module is imported at GUI start
    on installs that may not have every sibling widget module, and a colour
    picker is not worth an import-time dependency at the top of the file.

    :returns: a :class:`QColor`. Check ``isValid()`` -- an invalid one is the
        user cancelling, which is an answer and not a failure.
    """
    from .colour_picker import pick_colour as shared

    return shared(parent, initial, title)


def colour_for(index: int, alpha: int = 255) -> QColor:
    """Stable colour for category ``index``."""
    colour = QColor(PALETTE[index % len(PALETTE)])
    colour.setAlpha(alpha)
    return colour


def _first_column(frame, names) -> Optional[str]:
    """The first of ``names`` this frame carries, or ``None``.

    ``None`` is an answer and not a failure: a penalised fit has no standard
    error and never will, and a table with no corrected p-value has nothing to
    call a hit with. Both are said out loud by the plots below rather than
    being papered over with a default.
    """
    columns = getattr(frame, "columns", ())
    for name in names:
        if name in columns:
            return name
    return None


def _finite(values) -> np.ndarray:
    """Coerce to float and replace anything unplottable with NaN.

    A p-value column arrives with blanks, strings and the occasional inf from
    a log of zero. Left alone, one of those silently rescales the whole axis
    and the plot looks empty.
    """
    array = np.asarray(values, dtype="float64")
    return np.where(np.isfinite(array), array, np.nan)


def _violin_profile(values, half_width: float):
    """``(centres, half-widths)`` tracing one side of a violin.

    A histogram rather than a kernel density estimate, deliberately. A KDE
    needs a bandwidth, and a bandwidth chosen for the user is a smoothing
    decision made on their behalf that shows up as structure in the picture --
    on the handful of points per group these plots often hold, the bandwidth
    decides the shape entirely. Counting into bins invents nothing; the bins
    are visible as steps, which is the honest tell that the shape is coarse.

    Returns ``(None, None)`` when every value is identical: a density with no
    width is a vertical line, and drawing one as a violin claims a spread that
    is not there.
    """
    v = np.asarray(values, dtype=float)
    low, high = float(np.min(v)), float(np.max(v))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None, None
    bins = int(np.clip(np.sqrt(len(v)) * 2, 6, 24))
    counts, edges = np.histogram(v, bins=bins, range=(low, high))
    centres = (edges[:-1] + edges[1:]) / 2.0
    peak = float(counts.max())
    if peak <= 0:
        return None, None
    density = counts.astype(float) / peak * float(half_width)
    # Pinned shut at both ends, so the outline closes on the data's range
    # instead of stopping mid-air at the first and last bin's width.
    centres = np.concatenate([[low], centres, [high]])
    density = np.concatenate([[0.0], density, [0.0]])
    return centres, density


class FastPlot(QWidget):
    """A pyqtgraph plot with the controls every plot here wants.

    :ivar point_clicked: emitted with the position of a clicked point IN THIS
        PLOT'S OWN FRAME. It is not an index into anyone else's table; see
        :attr:`key_selected` for the link that survives sorting and filtering.
    :ivar key_selected: emitted with the identifier of a clicked point.
    :ivar keys_selected: emitted with the identifiers of EVERY row behind the
        thing that was clicked. A scatter point is one row and emits both; a
        histogram bar is a hundred rows and can only honestly emit this one.
    """

    point_clicked = Signal(int)
    key_selected = Signal(str)
    keys_selected = Signal(list)

    def __init__(self, title: str = "", x_label: str = "", y_label: str = "",
                 parent=None):
        super().__init__(parent)
        #: False when pyqtgraph is absent. The widget still constructs, still
        #: lays out, and says why it is empty rather than raising.
        self.plots_available = HAVE_PYQTGRAPH
        # BEFORE THE BRANCH, so both constructors have it: the hooks below
        # fire on the very first setLabel, and the pyqtgraph-absent path
        # still has to answer log_axes() without raising.
        self._init_axis_state()
        #: The level control that lives ON the plot rather than three clicks
        #: into a menu, and the sentence beside it. Born as None: a plot
        #: whose host never offers levels never grows either.
        self._header = None
        self._level_label = None
        self._level_box = None
        self._level_note_label = None
        self._level_note = ""
        if not HAVE_PYQTGRAPH:
            self._build_without_pyqtgraph(title)
            return
        # BACKGROUND None IS TRANSPARENT, WHICH WAS ALREADY RIGHT. The ink was
        # not: `foreground="k"` hardcoded BLACK axes, ticks and labels, so on
        # a dark theme the plot drew black-on-transparent over a dark surface
        # and the axes were invisible. The matplotlib path has resolved this
        # correctly for a while via preferences.get_figure_colors(), which
        # returns TRANSPARENT_FIGURE_BG plus theme-correct ink and honours an
        # explicit colour the user has chosen; pyqtgraph simply never asked
        # it. Same source for both renderers, so a theme switch cannot move
        # one and not the other.
        self._background, self._foreground = _figure_colors()
        pg.setConfigOptions(antialias=True, background=None,
                            foreground=self._foreground)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        # ABOVE THE PLOT, BESIDE THE TITLE. Empty it costs no height, and it
        # stays empty on every plot whose host offers no levels -- which is
        # every plot but the volcano.
        self._header = QHBoxLayout()
        self._header.setContentsMargins(0, 0, 0, 0)
        self._header.setSpacing(6)
        layout.addLayout(self._header)

        self.plot = pg.PlotWidget(title=title or None)
        # EVERY ITEM AND EVERY LABEL, CAUGHT ON THE WAY IN -- see
        # :meth:`_install_axis_hooks`. It goes in before the first label is
        # set, because the label is one of the two things it catches.
        self._install_axis_hooks()
        # Modified left-drag selects a region (instruction 206). In here
        # rather than in a subclass because the funnel every plot emits
        # through is on this class, and a band that only some plots had
        # would be a gesture the user cannot rely on.
        self._install_rubber_band()
        self.plot.setLabel("bottom", x_label)
        self.plot.setLabel("left", y_label)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # A transparent pyqtgraph background is not enough on its own: the
        # QWidget it lives in still paints the theme's `bg` under the blanket
        # QWidget rule, so the plot sits on an opaque slab regardless. The
        # theme's own helper is what every other transparent surface here
        # uses -- see the hyperparam screen, which does exactly this.
        self.plot.setBackground(None)
        try:
            from ..theme import make_transparent

            make_transparent(self, self.plot, self.plot.viewport())
        except Exception:                   # theme absent
            pass
        layout.addWidget(self.plot, 1)

        # THE STRIP CARRIES WHAT IS PRESSED, NOT WHAT IS SET. Log x, log y
        # and grid were checkboxes here and are entries on the right-click
        # menu now (instruction 148 C): they are set once and then read off
        # the axis, so a permanent row of them under every plot spent screen
        # on three states nobody looks at twice. The legend stays, because it
        # is the one a reader flicks on and off while looking at the figure.
        controls = QHBoxLayout()
        self._legend_box = QCheckBox("legend")
        self._legend_box.setEnabled(False)
        self._legend_box.setToolTip(
            "Name the categories. Off by default: a 27-entry legend costs "
            "~40 ms of every redraw, against 3 ms for the plot itself.")
        self._legend_box.toggled.connect(self._toggle_legend)
        controls.addWidget(self._legend_box)
        controls.addStretch(1)

        reset = QPushButton("Reset view")
        # `auto_range_axes`, NOT `plot.autoRange`: the bare call freezes the
        # axes on today's points, so the NEXT redraw opens inside this run's
        # window. See that method.
        reset.clicked.connect(self.auto_range_axes)
        controls.addWidget(reset)
        # NO EXPORT BUTTON (187 D). Reported 2026-08-20: "the export button
        # dosnt cause any errors but the exported figure is broken, with
        # massive text and so on ... actually remove the export button, save
        # styled is enough."
        #
        # TWO DOORS AND ONE OF THEM WROTE BLIND. `export` writes with no
        # preview and no styling pass, so a page sized in millimetres got
        # text scaled for the screen -- which is the massive text. Save
        # styled shows what it is about to write, which is the difference,
        # and it is the door that stays.
        save = QPushButton("Save figure…")
        save.setToolTip(
            "Preview the figure's colors, background, and dimensions before "
            "saving it.")
        save.clicked.connect(lambda: self.save_styled())
        controls.addWidget(save)
        layout.addLayout(controls)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        #: The plot's own sentence, kept apart from whatever was last clicked.
        self._headline = ""
        #: Whatever was last clicked, kept apart from the plot's own sentence.
        self._note = ""
        #: What a RESTYLE has to say -- the colour scale's range, which shape
        #: is which. Its own slot because it belongs to neither of the other
        #: two: a click must not wipe the legend of a colour scale, and a
        #: redraw's headline must not wipe it either.
        self._style_note = ""

        self._restyle_state()

        self._labels: Sequence[str] = ()
        self._legend_colours: dict = {}
        # The other two layered channels' keys, set by `set_results` and
        # read by the legend. Declared here so a plot that has never drawn
        # still answers what its legend would say: nothing.
        self._shape_legend = None
        self._opacity_legend = None
        self._items: list = []

        # THE KEY JOIN. Row-to-point highlighting is joined on the identifier
        # the row carries, never on a position -- a table sorted by effect and
        # a scatter drawn in input order are the same points in two orders,
        # and joining them by index lights up the WRONG guide silently, in
        # exactly the direction nobody questions, because something lit up.
        self._keys: Sequence[str] = ()
        self._key_rows: dict = {}
        self._row_xy: dict = {}
        self._selected_key: Optional[str] = None
        self._highlight = None
        #: EVERY selected identifier, in the order they were picked.
        #: `_selected_key` stays the most recent one so the single-select
        #: consumers keep working unchanged; this is the list the ones that
        #: can show more than one read. Two names for one state would drift,
        #: so `_selected_key` is derived from this and never set beside it.
        self._selected_keys: List[str] = []
        #: The rings for members 2..n. The first stays `_highlight` so that
        #: every existing test and every subclass override still finds it
        #: where it has always been.
        self._extra_highlights: List[Any] = []

        #: ``[(label, callback, checked)]`` for raw vs adjusted p-values.
        self._p_values = []

        #: ``[(label, callback, checked)]`` for the multiple-testing
        #: correction the plot draws. Empty on a plot that corrects nothing.
        self._corrections = []

        #: ``[(label, callback, checked)]`` for HOW the corrected p is shown
        #: -- as a call, as a ramp, as a size. Empty on a plot with no q.
        self._encodings = []

        #: ``([(label, callback, checked)], multiplier, on_multiplier)`` for
        #: the effect-size cut, or an empty triple.
        self._thresholds = ([], None, None)

        #: ``[(label, callback, checked)]`` for gene / guide / both.
        self._levels = []

        #: ``[(label, callback, checked)]`` for the TAGM/LOPIT compartments
        #: this screen actually has. ONE at a time against grey; 27 hues is
        #: what the house style forbids and also what cost 40 ms of a 49 ms
        #: redraw.
        self._compartments = []

        #: ``[(label, callback, checked)]`` for the baselines this plot can
        #: measure its effects from. Empty unless the host offers them.
        self._baselines = []
        self._smoothers = None
        self._smoother_chosen = ""

        #: ``[(label, callback, checked)]`` for the MARK this plot's groups
        #: are drawn with. Empty on a plot whose x-axis is continuous, where
        #: "draw it as a violin" is not a question that has an answer.
        self._marks = []

        #: ``(style, on_change, choices)`` for a figure style whose fields
        #: become menu entries, or None.
        self._style = None

        #: ``(callback, label)`` for an action that re-runs the analysis, or
        #: None. BORN HERE, not on first use: a filter control connected in
        #: __init__ to a handler that reads an attribute created later is the
        #: `_significance` crash, and it took the whole panel down at launch.
        self._refit = None

        # Right-click to restyle, the same gesture the matplotlib figures use.
        self.plot.setContextMenuPolicy(Qt.CustomContextMenu)
        self.plot.customContextMenuRequested.connect(self._style_menu)

    def _build_without_pyqtgraph(self, title: str) -> None:
        """A usable, honest empty box instead of a traceback.

        Every attribute the rest of this class and its callers touch is set
        HERE. A half-built widget that raises on its third method is worse
        than one that raises on its first: the traceback then names a symptom
        instead of the cause -- which is exactly how the original report read.
        """
        from PySide6.QtWidgets import QLabel

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)
        if title:
            layout.addWidget(QLabel(f"<b>{title}</b>"))
        notice = QLabel(PYQTGRAPH_MISSING_MESSAGE)
        notice.setWordWrap(True)
        notice.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(notice)
        layout.addStretch(1)

        self.plot = _Absorbs()
        self._background, self._foreground = _figure_colors()
        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._headline = ""
        self._note = ""
        self._labels = ()
        self._legend_colours = {}
        self._shape_legend = None
        self._opacity_legend = None
        self._items = []
        self._keys = ()
        self._key_rows = {}
        self._row_xy = {}
        self._selected_key = None
        self._selected_keys = []
        self._extra_highlights = []
        self._highlight = None
        self._refit = None
        self._baselines = []
        self._smoothers = None
        self._smoother_chosen = ""
        self._compartments = []
        self._corrections = []
        self._encodings = []
        self._p_values = []
        self._thresholds = ([], None, None)
        self._marks = []
        self._frame = None
        self._style_note = ""
        self._restyle_state()
        self._legend_box = QCheckBox("Legend")
        self._legend_box.setEnabled(False)

    # ----------------------------------------------------------------- state

    def _reset_scene(self) -> None:
        """Empty the plot AND the bookkeeping that describes what was on it.

        ``plot.clear()`` takes the artists off the scene and leaves every
        dictionary that pointed at them behind. That is not tidiness: after a
        redraw ``_row_xy`` still holds the PREVIOUS table's coordinates, so
        :meth:`highlight_key` rings the place where a point used to be, and
        ``_highlight`` still names an item that is no longer in the scene, so
        removing it later raises. The volcano did this by hand; every plot
        here needs it, which is why it is one method.

        ``_keys`` is deliberately NOT cleared here -- a redraw of the same
        table must keep its identifiers -- but every ``set_*`` below re-sets
        them, so a NEW table cannot inherit the old ones either.
        """
        self.plot.clear()
        self._row_xy = {}
        self._highlight = None
        self._labels = ()
        # The log transform's bookkeeping goes with the artists it described.
        # THE SCALE ITSELF STAYS: a user who asked for a log axis has not
        # unasked for it by filtering the table, and every item the redraw
        # puts back is transformed as it arrives.
        self._drawn = []

    # ------------------------------------------------------------- restyling

    def offer_refit(self, callback, label: str = "Re-fit with another model…"):
        """Add an action that CHANGES THE NUMBERS to the right-click menu.

        :param callback: called with no arguments when the user picks it.
        :param label: what the action says.

        Everything else on that menu changes how the figure looks and nothing
        else. This one re-runs the regression, so it is put under its own
        heading rather than in the list -- a user reaching for "Point size"
        must not be one slip away from starting a fit.

        Offered by the host rather than built in, because the plot knows
        nothing about settings, count data or where a run writes, and should
        not learn: the same widget draws a simulation and a sweep trial.
        """
        self._refit = (callback, label)

    def offer_compartments(self, options) -> None:
        """Offer "colour by localisation" as a submenu.

        :param options: ``[(label, callback, checked)]``.
        """
        self._compartments = list(options or ())

    def offer_p_values(self, options) -> None:
        """Offer raw vs adjusted p-values for the y-axis.

        :param options: ``[(label, callback, checked)]``, or empty when there
            is no correction to switch to -- an entry promising "adjusted" on
            an uncorrected run offers a number that is not there.
        """
        self._p_values = list(options or ())

    def offer_corrections(self, options) -> None:
        """Offer the multiple-testing correction, ON the graph.

        :param options: ``[(label, callback, checked)]``, or empty when this
            plot has no correction to redo.

        :data:`spacr.multiple_testing.METHODS` holds thirteen methods
        and the run picks one; comparing BH against Bonferroni against Storey
        on the screen in front of you is a two-second question that otherwise
        costs a re-run.
        """
        self._corrections = list(options or ())

    def offer_encodings(self, options) -> None:
        """Offer every field-acceptable way of SHOWING the adjusted p.

        :param options: ``[(label, callback, checked)]``, or empty when this
            plot has no corrected p to encode.

        The design: "id like the user to have access to visualizing
        adjusted P in all the ways that are acceptable to the field. showing
        as color, showing the descrete P on the axis buy showing the line
        where the adjusted p threshold lands, etc."

        BESIDE THE CORRECTION AND ABOVE THE RESTYLING, because these are
        statements about what the picture MEANS -- which channel carries the
        FDR -- and not about how it looks. Two of them compete for the colour
        channel and one composes with whatever is on it; the entries say so
        and the caption says which is in force.
        """
        self._encodings = list(options or ())

    def offer_thresholds(self, options, *, multiplier=None,
                         on_multiplier=None) -> None:
        """Offer the effect-size cut: how it is measured and how wide.

        :param options: ``[(label, callback, checked)]`` -- the modes.
        :param multiplier: the current width, shown on its own entry.
        :param on_multiplier: called with the new number when it is changed.

        On the PLOT because the settings-panel controls for these grey out
        under `inference='nonparametric'` -- correctly, since the permutation
        path uses no control-spread cut -- and users may otherwise not
        being able to find them.
        """
        self._thresholds = (list(options or ()), multiplier, on_multiplier)

    def offer_levels(self, options, *, note: str = "") -> None:
        """Configure result-level choices on the plot and context menu.

        Level selection changes which rows are displayed rather than their
        styling, so it receives a separate menu section and an on-plot control.
        The control remains visible when a run contains both gene- and
        guide-level fits, making the active subset explicit.

        :param options: Sequence of ``(label, callback, checked)`` entries.
        :param note: Host-supplied description of the displayed and excluded
            levels. It is shown beside the control and in the status line and
            is retained across redraws.
        """
        self._levels = list(options or ())
        self._level_note = str(note or "")
        self._refresh_level_control()

    def offer_baselines(self, options) -> None:
        """Offer "measure the effects from ..." on the right-click menu.

        :param options: ``[(label, callback, checked)]``.

        Separate from :meth:`offer_refit` because the two are different kinds
        of thing and a user must be able to tell them apart: a baseline moves
        where zero is drawn on a fit that has already happened, a re-fit
        replaces the fit.
        """
        self._baselines = list(options or ())

    def offer_smoothers(self, on_change, *, chosen: str = "lowess") -> None:
        """Offer the four diagnostic smoothers on the right-click menu.

        :param on_change: called ``(method_name)`` when one is picked, or
            with ``""`` for none. The host redraws from it.
        :param chosen: which one is currently drawn.

        SEPARATE FROM :meth:`offer_refit` DELIBERATELY, and the menu says so.
        These curves are laid over a fit that has already happened; none of
        them replaces it, and none of them decides a hit. They therefore
        belong to a different category from the inferential entries in
        ``regression_type``.
        """
        self._smoothers = on_change
        self._smoother_chosen = str(chosen or "")

    def _smoother_options(self) -> list:
        """Build the menu entries, one per diagnostic, plus "none"."""
        from spacr.nonparametric_fits import CATEGORY_DIAGNOSTIC, METHODS

        entries = [("None", lambda: self._smoothers(""),
                    not self._smoother_chosen)]
        for name, spec in METHODS.items():
            if spec["category"] != CATEGORY_DIAGNOSTIC:
                continue
            entries.append((
                spec["label"],
                (lambda n=name: self._smoothers(n)),
                self._smoother_chosen == name,
            ))
        return entries

    def add_smoother(self, x, y, *, method: str = "lowess",
                     colour: str = "#55A868") -> str:
        """Lay one diagnostic curve over the points already drawn.

        :returns: what to say about it -- the method, its note, and the band
            when it reports one -- or the refusal, which is a sentence the
            caller shows rather than an exception it swallows. Returns ``""``
            when ``method`` is empty, so "none" is not a special case at
            every call site.

        The curve carries no p-value and cannot acquire one: it comes back as
        a :class:`spacr.nonparametric_fits.Curve`, which has no such
        attribute. A smoother that bends where the straight trend line is
        flat is the finding -- it means the mean model is missing a term --
        and that is a statement about the FIT, not a test of a guide.
        """
        if not method:
            return ""
        from spacr.nonparametric_fits import smooth

        try:
            curve = smooth(x, y, method=method)
        except ValueError as refusal:
            # A refusal is the answer, not a failure: a Gaussian process
            # asked for more rows than it can take says the number.
            return str(refusal)
        except Exception as problem:                          # noqa: BLE001
            return f"{method} could not be drawn: {problem}"

        if curve.has_band:
            band = pg.FillBetweenItem(
                pg.PlotDataItem(curve.x, curve.lower),
                pg.PlotDataItem(curve.x, curve.upper),
                brush=pg.mkBrush(85, 168, 104, 60))
            self.plot.addItem(band)
        self.plot.plot(curve.x, curve.y,
                       pen=pg.mkPen(colour, width=2.0))
        said = f"{curve.method} curve laid over the points"
        if curve.note:
            said += f" ({curve.note})"
        return said + ". It is a diagnostic: it decides no hit."

    def offer_style(self, style, on_change=None, *, choices=None,
                    use_default: bool = True) -> None:
        """Put a figure's OWN style object onto this plot's right-click menu.

        :param style: any dataclass describing how the figure looks --
            :class:`spacr.volcano_style.VolcanoStyle` and whatever joins it.
        :param on_change: called ``(name, value)`` when the user changes one,
            which is where the host redraws.
        :param choices: ``{field: values}`` for fields that are a closed set.
        :param use_default: start ``style`` from this project's saved default
            for its kind, if there is one. This is where a house style
            actually reaches a figure -- point 5 is otherwise a preference
            nothing reads.

            APPLIED ONCE PER STYLE OBJECT, not on every call. A host that
            re-offers the SAME object after the user has edited it gets
            nothing done to it; a host that builds a fresh one per redraw had
            no edits to lose. Re-asserting it unconditionally is the mistake
            that cost this module a day on the p-axis: the host redraws on
            every level, baseline and compartment change, and any one of them
            would silently undo a choice the user had made.

        The design: the entries come from ``dataclasses.fields(style)``,
        so a style that gains a field gains a menu entry and the two cannot
        fall out of step. Offered by the host for the same reason
        :meth:`offer_refit` is -- only the host knows which style object is
        driving the picture, and the same widget draws a simulation and a
        sweep trial.
        """
        first_time = self._style is None or self._style[0] is not style
        self._style = (style, on_change, dict(choices or {}))
        if use_default and first_time:
            changed = apply_default_style(style, on_change)
            if changed:
                self.set_style_note(
                    f"Started from this project's saved "
                    f"{style_kind(style)} style ({len(changed)} settings). "
                    f"'Clear the default' on the Figure style menu undoes it.")

    def offer_marks(self, options) -> None:
        """Offer "draw the groups as ..." on the right-click menu.

        :param options: ``[(label, callback, checked)]``, the same shape as
            :meth:`offer_baselines` -- one entry per :data:`MARK_TYPES` the
            host can draw.

        Offered by the host rather than built in, and for the same reason
        :meth:`offer_refit` is: only the plot that owns the arrays knows
        whether its x-axis is a set of GROUPS at all. A volcano's x is an
        effect size, and "show it as a violin" is not a question that has an
        answer there.
        """
        self._marks = list(options or ())

    def _refresh_level_control(self) -> None:
        """Build, fill or hide the level control above the plot."""
        if self._header is None:
            return
        from PySide6.QtWidgets import QComboBox, QLabel

        if self._level_box is None:
            self._level_label = QLabel("Level:")
            self._level_box = QComboBox()
            self._level_box.setToolTip(
                "Which family of coefficients is drawn. A run fitted at both "
                "levels holds two of them.")
            self._level_box.activated.connect(self._on_level_chosen)
            self._level_note_label = QLabel("")
            self._level_note_label.setWordWrap(True)
            self._header.addWidget(self._level_label)
            self._header.addWidget(self._level_box)
            self._header.addWidget(self._level_note_label, 1)
        showing = bool(self._levels)
        for widget in (self._level_label, self._level_box,
                       self._level_note_label):
            widget.setVisible(showing)
        if showing:
            # `activated` fires on a USER's choice only, so refilling cannot
            # re-enter the callback -- blocked anyway, because a future
            # currentIndexChanged here would, and silently.
            blocked = self._level_box.blockSignals(True)
            self._level_box.clear()
            current = 0
            for index, (label, _callback, checked) in enumerate(self._levels):
                self._level_box.addItem(str(label))
                if checked:
                    current = index
            self._level_box.setCurrentIndex(current)
            self._level_box.blockSignals(blocked)
        self._level_note_label.setText(self._level_note)
        self._level_note_label.setVisible(showing and bool(self._level_note))
        self._refresh_status()

    def _on_level_chosen(self, index: int) -> None:
        """The user picked a level off the control above the plot."""
        if 0 <= int(index) < len(self._levels):
            self._levels[int(index)][1]()

    def level_note(self) -> str:
        """The sentence naming what is drawn and what is not, or ``""``."""
        return self._level_note

    def _restyle_state(self) -> None:
        """Every field the restyle menu reads, born before anything can ask.

        BOTH constructors call this, for the reason written on
        :meth:`_build_without_pyqtgraph`: a widget whose third method raises
        because its second never ran is a trap, and the pyqtgraph-absent path
        is exactly the one nobody exercises by hand.
        """
        #: Points, and how big, from :meth:`set_font_size`. ``None`` is
        #: "whatever pyqtgraph chose", which is not the same as any number.
        self._font_size: Optional[int] = None
        #: The ink for EVERY piece of text -- title, axis labels, tick
        #: LABELS, legend, the caption on a threshold line -- or None while
        #: it follows the theme. The corresponding line control is
        #: :attr:`_line_colour`.
        self._font_colour: Optional[str] = None
        #: The ink for EVERY line -- the data's own, the reference and
        #: threshold lines, the Q-Q diagonal, the trends, AND the axis spines
        #: and the tick MARKS -- or None while each keeps its own.
        #:
        #: TICK MARKS ARE LINES AND TICK LABELS ARE TEXT. That is the one
        #: place the two controls meet and it is the one a reader could take
        #: either way, so it is written down rather than left to the code.
        self._line_colour: Optional[str] = None
        #: ``(column, colormap)`` while a colour scale is mapped, else None.
        self._colour_column: Optional[tuple] = None
        #: The column mapped to point shapes, or None.
        self._shape_column: Optional[str] = None
        #: The width and height of a SAVED page, in millimetres. The height
        #: is None until asked for, meaning "follow the plot's own aspect".
        self._export_width_mm: float = EXPORT_WIDTH_MM
        self._export_height_mm: Optional[float] = None
        #: Dots per inch a RASTER export is written at, or None to keep the
        #: scene's own pixel count. The page is measured in millimetres, so
        #: this is the number that decides how many pixels a PNG has:
        #: ``page width in inches x dpi``. Vector output ignores it, which is
        #: why a PDF or an SVG never asks.
        self._export_dpi: Optional[int] = None
        #: The size floors this widget was given by whoever placed it, kept
        #: so :meth:`clear_screen_size` puts them back rather than releasing
        #: the widget to nothing. `RegressionResultsPanel` sets
        #: `volcano.setMinimumHeight(240)`, and a restyle that silently
        #: dropped that floor would let the splitter collapse the plot.
        self._size_bounds: Optional[tuple] = None
        #: A callable that writes the correction the plot is drawing, or
        #: None on a plot that recorrects nothing.
        self._correction_writer = None

    # ------------------------------------------------------- what is stylable

    def frame(self):
        """The table this plot was drawn from, or ``None``.

        The two column-mapping controls need it -- "cmap (choose any column)"
        and "point shape (choose any column)" both name a column of THIS
        plot's own table -- and a plot handed bare arrays honestly has none,
        which is why those entries grey out rather than offering a list of
        nothing.
        """
        return getattr(self, "_frame", None)

    def numeric_columns(self) -> list:
        """Columns a COLOUR SCALE could read, in table order.

        A cmap belongs only on a continuous quantity: mapping one onto a
        nominal category is the mistake the house style warns about, because
        it puts an order into the picture that the data does not have.

        So: numeric dtype, not boolean -- ``True``/``False`` is two
        categories wearing a number's dtype -- and at least two distinct
        finite values, because a column with one value maps every point to
        the same colour and a scale with no range is not a scale.
        """
        frame = self.frame()
        if frame is None or not len(frame):
            return []
        from pandas.api.types import is_bool_dtype, is_numeric_dtype

        found = []
        for name in frame.columns:
            column = frame[name]
            if not is_numeric_dtype(column) or is_bool_dtype(column):
                continue
            values = _finite(column.to_numpy())
            usable = values[~np.isnan(values)]
            if len(usable) and float(usable.min()) < float(usable.max()):
                found.append(str(name))
        return found

    def shape_columns(self) -> list:
        """Columns a POINT SHAPE could read, in table order.

        Low cardinality and nothing else: two to :data:`MAX_SHAPE_VALUES`
        distinct values. Both ends are real limits rather than tidiness --
        one value gives every point the same shape and says nothing, and past
        eight the shapes stop being tellable apart at scatter-plot size, so
        the reader is decoding a key instead of reading a figure.

        DTYPE IS NOT THE TEST. ``n_guides`` is an integer column with four
        values and is exactly what a reader wants shapes for, while
        ``feature`` is a string column with 1,215 and is exactly what they do
        not. Counting the values answers both; asking the dtype answers
        neither.
        """
        frame = self.frame()
        if frame is None or not len(frame):
            return []
        found = []
        for name in frame.columns:
            try:
                distinct = frame[name].astype(str).nunique(dropna=False)
            except Exception:       # an unhashable cell
                continue
            if 2 <= int(distinct) <= MAX_SHAPE_VALUES:
                found.append(str(name))
        return found

    def colour_map_reason(self) -> str:
        """Why "colour by a column" cannot act here, or ``""``.

        The rule, applied to a menu: an entry that cannot do
        anything is greyed out AND SAYS WHY. Silently absent leaves the user
        hunting for a control they were told about; present-but-inert leaves
        them clicking it and concluding the application is broken.
        """
        if not self._scatter_items():
            return "nothing on this plot is drawn as points"
        if self.frame() is None:
            return "this plot holds no table, so there is no column to read"
        if not self.numeric_columns():
            return "no column here is a number a colour scale could read"
        return ""

    def shape_reason(self) -> str:
        """Why "shape by a column" cannot act here, or ``""``."""
        if not self._scatter_items():
            return "nothing on this plot is drawn as points"
        if self.frame() is None:
            return "this plot holds no table, so there is no column to read"
        if not self.shape_columns():
            return (f"no column has between 2 and {MAX_SHAPE_VALUES} values, "
                    f"and more shapes than that cannot be told apart")
        return ""

    def line_reason(self) -> str:
        """Return why line styling is unavailable, or an empty string.

        Axis spines are excluded because the control applies to plot marks
        and reference lines.
        """
        return "" if self.line_items() else "this plot has no lines on it"

    def line_colour_reason(self) -> str:
        """Return why line-colour styling is unavailable, or an empty string.

        The control applies to data lines and axes, so it is available for
        any rendered plot that contains either.
        """
        if not self.plots_available:
            return "this build has no pyqtgraph, so nothing is drawn"
        return ("" if self.line_items() or self.axis_items()
                else "this plot has nothing drawn as a line")

    def follow_the_theme(self) -> None:
        """Put BOTH colour controls back to automatic.

        The way out of a colour, and it has to exist: the design is a
        preference that froze because a resolved default was written back
        over the word "auto", and a control a user can only set is the same
        freeze performed by hand.
        """
        self.set_line_colour(None)
        self.set_font_colour(None)
        self.restyle()

    def point_reason(self) -> str:
        """Why the point controls cannot act here, or ``""``.

        A p-value histogram is bars. "Point size" on it is the plainest case
        of a control that looks live and does nothing.
        """
        return ("" if self._scatter_items()
                else "nothing on this plot is drawn as points")

    # ----------------------------------------------------- the log transform

    def _init_axis_state(self) -> None:
        """The axis bookkeeping, born before the first item can be drawn.

        Called from BOTH constructors, and before the first ``setLabel``,
        because :meth:`_install_axis_hooks` writes into these the moment an
        axis is named.
        """
        #: Which axes are drawn on a logarithmic scale.
        self._log = {"x": False, "y": False}
        #: Every positionable item on the plot, with the DATA coordinates it
        #: was handed. The drawn coordinates are always derived from these
        #: and never the other way round, so logging and unlogging an axis is
        #: exactly the identity rather than a round trip through log10.
        self._drawn: list = []
        #: The axis labels as the plot asked for them, WITHOUT the note that
        #: says an axis is logged -- so the note is added once however many
        #: times the scale is switched.
        self._base_labels: dict = {}
        #: Which of :data:`CANVAS_SHAPES` the canvas is held at.
        self._canvas_shape: str = "free"
        #: Guards the re-entrant resize the shape imposes.
        self._shaping: bool = False
        #: Whether the grid is drawn. State rather than a widget since the
        #: control moved onto the menu: a menu entry is built fresh on every
        #: right-click and cannot be the thing that remembers.
        self._grid_on: bool = True
        #: ``(low, high)`` IN DATA UNITS of the band the y axis HIDES, or
        #: None while the axis is unbroken. Data units for the same reason
        #: :attr:`_pinned` is: it is what the user typed, and re-deriving it
        #: from the drawn range would put a rounding error into it.
        self._split: dict = {"y": None}
        #: How tall the break itself is drawn, in DRAWN units. Computed from
        #: the data that is kept when the split is set, so the break reads as
        #: a break whatever the axis spans.
        self._split_gap: float = 0.0
        #: The left axis' own tick methods, kept so a split can be undone.
        self._axis_ticks: Optional[tuple] = None
        #: The limits the user typed, IN DATA UNITS, per axis -- or None
        #: where the axis still follows its data. Kept in data units because
        #: that is what was typed: the drawn range is log10 of this while the
        #: axis is logged, and re-deriving it from the drawn range would put
        #: a rounding error into a number the user chose exactly.
        self._pinned: dict = {"x": None, "y": None}

    def _install_axis_hooks(self) -> None:
        """Install central hooks that transform every item and axis label.

        Both ``PlotItem`` and the bound methods copied onto ``PlotWidget``
        are wrapped. This keeps data marks, annotations, thresholds, and
        labels consistent when an axis uses a transformed scale.
        """
        plot_item = self.plot.plotItem
        add_item = plot_item.addItem
        set_label = plot_item.setLabel

        def _add(item, *args, **kwargs):
            add_item(item, *args, **kwargs)
            self._register_drawn(item)

        def _label(axis, text=None, units=None, unitPrefix=None, **kwargs):
            self._base_labels[str(axis)] = "" if text is None else str(text)
            set_label(axis, self._axis_label_text(str(axis)), units,
                      unitPrefix, **kwargs)

        plot_item.addItem = _add
        plot_item.setLabel = _label
        self.plot.addItem = _add
        self.plot.setLabel = _label
        # A MOUSE DRAG FORGETS A TYPED LIMIT. Panning or zooming by hand is a
        # user saying "not that window any more"; remembering the old one and
        # snapping back to it the next time the scale changes would make the
        # plot argue with the person driving it.
        try:
            plot_item.vb.sigRangeChangedManually.connect(self._forget_pins)
        except Exception:               # no such signal
            pass

    def _forget_pins(self, *_args) -> None:
        self._pinned = {"x": None, "y": None}

    # -- what is drawn, and what it says about being logged ------------------

    def _describe_item(self, item):
        """A :data:`_Drawn` record for ``item``, or ``None`` if it does not move.

        Read at ADD TIME, when the item still holds the coordinates the
        caller handed it. Everything after this reads the record.
        """
        blocks: dict = {}
        if isinstance(item, ScatterPlotItem):
            x = np.array(item.data["x"], dtype="float64")
            y = np.array(item.data["y"], dtype="float64")
            return _Drawn(item, x, y, blocks, "points",
                          self._counts_of(x, y))
        if isinstance(item, pg.InfiniteLine):
            angle = float(getattr(item, "angle", 90.0)) % 180.0
            # THE ANGLE IS ASKED FIRST, AND THAT ORDER IS THE POINT.
            # `InfiniteLine.value()` answers with a scalar only for the two
            # orthogonal angles; an oblique line answers with the whole
            # ``[x, y]`` position, and `float()` of a list is a TypeError --
            # so reading the value before the branch made adding a diagonal
            # line to any plot in this module raise out of `addItem` instead
            # of reaching the branch written to ignore it.
            if angle == 90.0:
                x, y = np.array([float(item.value())]), None
            elif angle == 0.0:
                x, y = None, np.array([float(item.value())])
            else:                       # an oblique line moves on neither axis
                return None
            return _Drawn(item, x, y, blocks, "line", self._counts_of(x, y))
        if isinstance(item, pg.BarGraphItem):
            try:
                edges = [np.atleast_1d(np.asarray(v, dtype="float64"))
                         for v in item._getNormalizedCoords()]
            except Exception:           # an odd bar spec
                # A BAR WE CANNOT READ IS A BAR WE CANNOT MOVE, and a scale
                # that leaves one item behind is the bug this file is fixing.
                blocks = {"x": "one of the bars cannot be re-measured",
                          "y": "one of the bars cannot be re-measured"}
                return _Drawn(item, None, None, blocks, "bar",
                              self._counts_of(None, None))
            width = max(len(part) for part in edges)
            x0, y0, x1, y1 = (np.resize(part, width) for part in edges)
            x = np.vstack([x0, x1])
            y = np.vstack([y0, y1])
            if not np.any(y0 > 0):
                # THE BASELINE IS THE POINT. A histogram's bars are measured
                # from zero, and zero has no logarithm; saying that is a
                # better answer than "50 of 100 values are at or below zero",
                # which is true and tells the reader nothing they can act on.
                blocks["y"] = ("the bars are measured from zero, which has "
                               "no logarithm")
                y = None
            return _Drawn(item, x, y, blocks, "bar", self._counts_of(x, y))
        getter = getattr(item, "getData", None)
        if callable(getter):
            try:
                data = getter()
            except Exception:           # an empty curve
                return None
            if not data or data[0] is None or data[1] is None:
                return None
            x = np.array(data[0], dtype="float64")
            y = np.array(data[1], dtype="float64")
            return _Drawn(item, x, y, blocks, "line", self._counts_of(x, y))
        return None

    @staticmethod
    def _counts_of(x, y) -> dict:
        """``{axis: (at or below zero, finite in total, the lowest)}``."""
        counts = {}
        for axis, values in (("x", x), ("y", y)):
            if values is None:
                counts[axis] = (0, 0, None)
                continue
            flat = np.asarray(values, dtype="float64").ravel()
            finite = flat[np.isfinite(flat)]
            bad = finite[finite <= 0]
            counts[axis] = (int(bad.size), int(finite.size),
                            float(bad.min()) if bad.size else None)
        return counts

    def _register_drawn(self, item) -> None:
        """Remember ``item``'s data coordinates and draw it at the current scale."""
        entry = self._describe_item(item)
        if entry is None:
            return
        self._drawn.append(entry)
        # A REDRAW CAN MAKE A LIVE LOG SCALE IMPOSSIBLE. The level filter
        # admits a coefficient of zero, a compartment's p-values reach 1 --
        # and log10 of those is not a number, so the points would silently
        # leave the plot. The scale comes off instead, and says so.
        refused = [axis for axis in ("x", "y")
                   if self._log[axis] and (axis in entry.blocks
                                           or entry.counts[axis][0])]
        for axis in refused:
            self._log[axis] = False
        if refused:
            names = " and ".join(f"log {axis}" for axis in refused)
            self._apply_log()
            self.set_style_note(
                f"{names} came off: {self.log_reason(refused[0])}")
            return
        if self._log["x"] or self._log["y"]:
            self._place(entry)

    def log_reason(self, axis: str) -> str:
        """Why ``axis`` cannot be drawn on a log scale, or ``""``.

        The rule, and the decision: a value at or
        below zero has no logarithm, and the answer is to REFUSE the axis
        rather than drop the points. Dropping them would make a volcano whose
        visible point count is a number nobody can account for.

        :param axis: ``"x"`` or ``"y"``.
        """
        axis = "y" if str(axis).lower().startswith("y") else "x"
        if not self.plots_available:
            return "this build has no pyqtgraph, so nothing is drawn"
        edge = "left" if axis == "y" else "bottom"
        try:
            named = getattr(self.plot.getAxis(edge), "_tickLevels", None)
        except Exception:               # absent axis
            named = None
        if named:
            # THE AXIS IS A LIST OF GROUPS. A control panel's x and an effect
            # ranking's y carry names at hand-placed positions; a logarithm
            # of a position that stands for "nc" is not a quantity.
            return (f"log {axis}: this axis names its groups rather than "
                    f"measuring a quantity")
        point_bad = point_total = 0
        scenery = None
        for entry in self._drawn:
            if axis in entry.blocks:
                return f"log {axis}: {entry.blocks[axis]}"
            bad, total, worst = entry.counts[axis]
            if entry.kind == "points":
                point_bad += bad
                point_total += total
            elif bad and (scenery is None or worst < scenery[1]):
                scenery = (entry.kind, worst)
        if point_bad:
            # THE SENTENCE INSTRUCTION 148 ASKED FOR, to the comma. A count
            # and a total, because "some points cannot be logged" leaves the
            # reader unable to judge whether the scale is worth having.
            return (f"log {axis}: {point_bad:,} of {point_total:,} points "
                    f"are at or below zero and have no logarithm")
        if scenery is not None:
            noun = "a bar" if scenery[0] == "bar" else "a line"
            return (f"log {axis}: {noun} on this plot reaches "
                    f"{scenery[1]:g}, which has no logarithm")
        return ""

    # -- the scale itself ----------------------------------------------------

    def log_axes(self) -> tuple:
        """``(log x, log y)`` -- whether each axis is drawn logarithmically."""
        return bool(self._log["x"]), bool(self._log["y"])

    def set_log_axes(self, x=None, y=None) -> tuple:
        """Put an axis onto a log scale, or take it off one.

        THE DATA IS TRANSFORMED, NOT ONLY THE AXIS. ``PlotItem.setLogMode``
        relabels the axis and then walks its items asking each to re-transform
        itself -- and ``ScatterPlotItem`` HAS NO ``setLogMode``, while every
        point on every plot in this module is one. The axis therefore claimed
        a log scale and the dots did not move: a reader took p-values off a
        ruler that did not describe the marks beside it, with nothing on
        screen to say so. That is a wrong figure, not an inert control.

        :param x: True, False, or None to leave the bottom axis alone.
        :param y: the same for the left axis.
        :returns: :meth:`log_axes` afterwards, because a request can be
            REFUSED -- see :meth:`log_reason` -- and a caller that assumed it
            was honoured would be making the original mistake again.
        """
        for axis, wanted in (("x", x), ("y", y)):
            if wanted is None:
                continue
            wanted = bool(wanted)
            if wanted == self._log[axis] or (wanted and self.log_reason(axis)):
                continue
            self._log[axis] = wanted
        self._apply_log()
        return self.log_axes()

    def _to_drawn(self, values, axis: str):
        """Data coordinates as they are DRAWN on ``axis``.

        Two transforms in one place and always in this order: the log scale
        first, then the y-axis split -- which is defined against the drawn
        quantity, so a split on a logged axis hides a band of DECADES and
        still lands where the user pointed.
        """
        if values is None:
            return None
        values = np.asarray(values, dtype="float64")
        if self._log[axis]:
            with np.errstate(divide="ignore", invalid="ignore"):
                values = np.log10(values)
        if axis == "y" and self._split.get("y") is not None:
            values = self._compress(values)
        return values

    def _to_data(self, value, axis: str) -> float:
        """A drawn coordinate back in DATA units. The tooltip's inverse."""
        value = float(value)
        if axis == "y" and self._split.get("y") is not None:
            value = float(self._expand(np.asarray([value], dtype="float64"))[0])
        return 10.0 ** value if self._log[axis] else value

    # -- the y-axis split ----------------------------------------------------

    def y_split(self) -> Optional[tuple]:
        """``(low, high)`` of the hidden band IN DATA UNITS, or ``None``."""
        return self._split.get("y")

    def _split_drawn(self) -> tuple:
        """The hidden band in DRAWN units -- after the log, before the break."""
        low, high = self._split["y"]
        if self._log["y"]:
            with np.errstate(divide="ignore", invalid="ignore"):
                return float(np.log10(low)), float(np.log10(high))
        return float(low), float(high)

    def _compress(self, values):
        """Drawn coordinates with the hidden band taken out."""
        low, high = self._split_drawn()
        gap = self._split_gap
        values = np.asarray(values, dtype="float64")
        return np.where(values <= low, values,
                        values - (high - low) + gap)

    def _expand(self, values):
        """The inverse of :meth:`_compress`. What a tick label has to read."""
        low, high = self._split_drawn()
        gap = self._split_gap
        values = np.asarray(values, dtype="float64")
        return np.where(values <= low, values,
                        values + (high - low) - gap)

    def y_split_reason(self, low, high) -> str:
        """Why ``(low, high)`` cannot be hidden on the y axis, or ``""``.

        A SPLIT THAT SWALLOWS POINTS IS REFUSED, and this is the one place
        that rule is stated. A broken axis is a piecewise-linear ruler whose
        tick labels still read the data's own values, so every mark stays at
        its own number -- but a mark INSIDE the hidden band has no number left
        on the ruler to sit at, and drawing it in the break would put a point
        somewhere its value is not. That is the same thing y-axis jitter does,
        and the design forbids it for the same reason.

        So the answer is the count, and the user moves the band.
        """
        if not self.plots_available:
            return "this build has no pyqtgraph, so nothing is drawn"
        low, high = float(low), float(high)
        if not np.isfinite(low) or not np.isfinite(high):
            return "a split needs two finite numbers"
        if high <= low:
            return "the top of the hidden band has to be above its bottom"
        if self._log["y"] and low <= 0:
            return "a logged axis has no coordinate at or below zero"
        inside = 0
        for entry in self._drawn:
            if entry.y is None or entry.kind != "points":
                continue
            values = np.asarray(entry.y, dtype="float64").ravel()
            values = values[np.isfinite(values)]
            inside += int(np.sum((values > low) & (values < high)))
        if inside:
            return (f"{inside:,} point{'s' if inside != 1 else ''} sit inside "
                    f"{low:g} to {high:g}, and a split cannot hide a band the "
                    f"data is in")
        return ""

    def set_y_split(self, low, high) -> str:
        """Hide an empty interval on the y-axis.

        Parameters
        ----------
        low : float
            Lower boundary of the hidden interval, in data units.
        high : float
            Upper boundary of the hidden interval, in data units.

        Returns
        -------
        str
            Empty string when the split is applied; otherwise, a user-facing
            explanation of why it was refused.

        Notes
        -----
        The split uses a piecewise-linear display transform while retaining
        the original data values in tick labels. It is refused when a plotted
        point lies inside the proposed interval, either boundary is non-finite,
        ``high <= low``, or a logarithmic y-axis would receive a non-positive
        lower boundary.

        Removing an empty interval can make the remaining dynamic range easier
        to inspect. It does not separate tied p-values or adjusted p-values;
        use :meth:`VolcanoPlot.set_p_axis` to choose which statistic is drawn.
        """
        reason = self.y_split_reason(low, high)
        if reason:
            self.set_style_note(f"y split: {reason}")
            return reason
        self._split["y"] = (float(low), float(high))
        self._split_gap = self._gap_for_split()
        self._install_split_ticks()
        self._apply_log()
        self._reframe_y()
        self.set_style_note(
            f"y axis split: {float(low):g} to {float(high):g} is not drawn. "
            f"A split gives the rest of the screen its height back; it does "
            f"not make a stepped adjusted P continuous.")
        return ""

    def _gap_for_split(self) -> float:
        """How tall to draw the break, in DRAWN units.

        A FIXED FRACTION OF WHAT IS KEPT, not of what is hidden. A band 40
        decades tall between two segments three decades tall is the case this
        control exists for, and a gap sized from the hidden band would then
        be thirteen times the data.
        """
        low, high = self._split_drawn()
        span = 0.0
        for entry in self._drawn:
            if entry.y is None:
                continue
            values = np.asarray(entry.y, dtype="float64").ravel()
            if self._log["y"]:
                with np.errstate(divide="ignore", invalid="ignore"):
                    values = np.log10(values)
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            below = values[values <= low]
            above = values[values >= high]
            if below.size:
                span += float(below.max() - below.min())
            if above.size:
                span += float(above.max() - above.min())
        return max(span * 0.06, 1e-9)

    def _install_split_ticks(self) -> None:
        """Make the left axis print DATA values across the break.

        Without this the axis would number the compressed coordinate, which
        is a ruler that reads one thing and measures another -- and a figure
        whose y-axis numbers are wrong is worse than one with no split.
        """
        if self._axis_ticks is not None:
            return
        try:
            axis = self.plot.getAxis("left")
        except Exception:                   # absent axis
            return
        original_values = axis.tickValues
        original_strings = axis.tickStrings

        def tick_values(minVal, maxVal, size):
            if self._split.get("y") is None:
                return original_values(minVal, maxVal, size)
            low, high = self._split_drawn()
            lo = float(self._expand(np.asarray([minVal]))[0])
            hi = float(self._expand(np.asarray([maxVal]))[0])
            out = []
            for spacing, values in original_values(lo, hi, size):
                kept = [v for v in values if not (low < v < high)]
                out.append((spacing,
                            [float(self._compress(np.asarray([v]))[0])
                             for v in kept]))
            return out

        def tick_strings(values, scale, spacing):
            if self._split.get("y") is None:
                return original_strings(values, scale, spacing)
            return [f"{self._to_data(v, 'y'):g}" for v in values]

        axis.tickValues = tick_values
        axis.tickStrings = tick_strings
        self._axis_ticks = (axis, original_values, original_strings)

    def clear_y_split(self) -> None:
        """Put the y axis back in one piece. The way out of a split."""
        if self._split.get("y") is None:
            return
        self._split["y"] = None
        self._split_gap = 0.0
        if self._axis_ticks is not None:
            axis, values, strings = self._axis_ticks
            axis.tickValues = values
            axis.tickStrings = strings
            self._axis_ticks = None
        self._apply_log()
        self._reframe_y()
        self.set_style_note("y axis split removed.")

    def _reframe_y(self) -> None:
        """Give the y axis back to its data -- UNLESS a limit was typed.

        The window the user was looking at is meaningless once the ruler
        underneath it changes, so the axis re-fits. A limit they TYPED is a
        different thing: :meth:`_apply_log` has just put it back in the new
        units, and releasing it here would throw away a number they chose.
        """
        box = self.plot.getViewBox()
        if self._pinned.get("y") is not None:
            return
        box.enableAutoRange(axis=box.YAxis, enable=True)
        self._sync_auto_range()

    def _place(self, entry) -> None:
        """Move one item to where the current scale says it belongs."""
        item = entry.item
        xs = self._to_drawn(entry.x, "x")
        ys = self._to_drawn(entry.y, "y")
        if isinstance(item, pg.InfiniteLine):
            position = xs if xs is not None else ys
            if position is not None and len(position):
                item.setValue(float(position[0]))
            return
        if isinstance(item, pg.BarGraphItem):
            options = {}
            if xs is not None:
                options.update(x=None, width=None, x0=xs[0], x1=xs[1])
            if ys is not None:
                options.update(y=None, height=None, y0=ys[0], y1=ys[1])
            if options:
                item.setOpts(**options)
            return
        if isinstance(item, ScatterPlotItem):
            # IN PLACE, not through setData. `ScatterPlotItem.setData` clears
            # the item and re-adds the points, which drops the per-point row
            # index every click on this plot is resolved through -- so the
            # dots would move and stop identifying anything.
            if xs is not None:
                item.data["x"] = xs
            if ys is not None:
                item.data["y"] = ys
            item.bounds = [None, None]
            item.prepareGeometryChange()
            item.informViewBoundsChanged()
            item.invalidate()
            return
        setter = getattr(item, "setData", None)
        if callable(setter):
            setter(x=xs if xs is not None else entry.x,
                   y=ys if ys is not None else entry.y)

    def _apply_log(self) -> None:
        """Re-place every item, relabel the axes, and re-gate the controls."""
        if not self.plots_available:
            return
        live = {id(item) for item in self.plot.plotItem.items}
        kept = []
        for entry in self._drawn:
            if id(entry.item) not in live:
                # Removed since it was registered -- the previous selection
                # ring, a bar that was outlined and then was not.
                continue
            kept.append(entry)
            self._place(entry)
        self._drawn = kept
        self._relabel_axes()
        self._reapply_pinned()

    def grid_shown(self) -> bool:
        """Whether the grid is drawn behind the marks."""
        return bool(self._grid_on)

    def set_grid(self, on: bool) -> None:
        """Draw the grid, or stop. Reachable from the Appearance group."""
        self._grid_on = bool(on)
        self.plot.showGrid(x=self._grid_on, y=self._grid_on, alpha=0.25)

    def _axis_label_text(self, edge: str) -> str:
        """What the label on ``edge`` reads, given the scale it is drawn at."""
        base = self._base_labels.get(edge, "")
        axis = "x" if edge in ("bottom", "top") else "y"
        # SAY IT WHERE THE AXIS IS. A tick on a menu nobody has open is not
        # notice; "-log10(p)" was already this idea, and a logged axis owes
        # the reader the same sentence. A SPLIT axis owes it more: its ruler
        # is piecewise linear, and a reader measuring a distance on it
        # without being told is measuring the wrong thing.
        notes = []
        if self._log.get(axis):
            notes.append("log scale")
        if axis == "y" and self._split.get("y") is not None:
            low, high = self._split["y"]
            notes.append(f"split, {low:g}-{high:g} not drawn")
        if not notes:
            return base
        joined = ", ".join(notes)
        return f"{base} ({joined})" if base else joined

    def _relabel_axes(self) -> None:
        for edge, axis in (("bottom", "x"), ("left", "y")):
            base = self._base_labels.get(edge, "")
            split = axis == "y" and self._split.get("y") is not None
            if not base and not self._log[axis] and not split:
                # AN UNLABELLED AXIS STAYS UNLABELLED. `setLabel` calls
                # `showLabel()`, so writing the empty string onto the control
                # panel's deliberately bare x-axis grows a blank strip there.
                continue
            self.plot.setLabel(edge, base)
        if self._font_colour is not None or self._font_size is not None:
            # `setLabel` takes the style with the text, so relabelling drops
            # a font the user chose off the menu unless it is put back.
            self.apply_text_style()

    def _point_tip(self, x, y, data) -> str:
        """The hover text for one point, IN DATA UNITS whatever the scale.

        pyqtgraph's own tip reads the point's DRAWN position, so on a logged
        axis it reports the logarithm -- a volcano saying "y: 1.30" where the
        p-value is 0.05. Undone here, which is the same inversion the axis
        ticks already make.
        """
        return (f"x: {self._to_data(x, 'x'):.3g}\n"
                f"y: {self._to_data(y, 'y'):.3g}\n"
                f"data={data}")

    # --------------------------------------------------------- axes and shape

    def _sync_auto_range(self) -> None:
        """Apply a pending pyqtgraph auto-range update before using the view.

        pyqtgraph 0.14 changed item-bound updates from an immediate
        ``updateAutoRange()`` call to ``queueUpdateAutoRange()``. That is a
        useful paint-time optimisation, but a public operation that reads or
        renders the plot in the same event-loop turn would otherwise see the
        previous table's range.

        ``updateAutoRange`` honours the ViewBox's per-axis flags, so draining
        the update cannot release an axis the user pinned. The method exists
        on the older pyqtgraph releases spaCR still supports as well.
        """
        if not self.plots_available:
            return
        update = getattr(self.plot.getViewBox(), "updateAutoRange", None)
        if callable(update):
            update()

    def axis_limits(self) -> tuple:
        """``((x from, x to), (y from, y to))`` as shown, IN DATA UNITS.

        Data units whatever the scale, because that is the only answer that
        means one thing. pyqtgraph's view range is in DRAWN units, so on a
        logged axis it reads ``(-6, 0)`` where the plot shows a millionth to
        one -- and a caller pre-filling a dialog from it would offer the user
        a logarithm to edit.
        """
        self._sync_auto_range()
        ranges = self.plot.getViewBox().viewRange()
        return ((self._to_data(ranges[0][0], "x"),
                 self._to_data(ranges[0][1], "x")),
                (self._to_data(ranges[1][0], "y"),
                 self._to_data(ranges[1][1], "y")))

    def set_axis_limits(self, x=None, y=None) -> None:
        """Pin an axis to ``(from, to)`` IN DATA UNITS. ``None`` skips it.

        DATA UNITS, and converted here, because the transform is this class's
        own: a user who types ``1e-6, 1`` on a logged axis means a millionth
        to one, not log10 of those, and pyqtgraph's ranges are in the drawn
        units it knows nothing about the meaning of.

        AUTO-RANGE IS TURNED OFF ON THE AXIS THAT IS PINNED, and only that
        one. pyqtgraph re-fits the view to the data on the next redraw
        otherwise, so a limit the user typed would survive until the first
        recolour and then silently spring back -- which reads as the control
        not working rather than as a redraw. The pin is REMEMBERED as well as
        applied, so a scale change re-imposes the same data window rather
        than leaving the view where the old units put it.

        A NON-POSITIVE LIMIT ON A LOGGED AXIS IS REFUSED, with the reason in
        the status line: it has no logarithm, and quietly substituting a
        bound the user did not type is how a figure comes to show a range
        nobody chose.

        :param x: ``(from, to)`` for the bottom axis, or None.
        :param y: ``(from, to)`` for the left axis, or None.
        """
        box = self.plot.getViewBox()
        refused = []
        for axis, limits, setter, which in (
                ("x", x, box.setXRange, box.XAxis),
                ("y", y, box.setYRange, box.YAxis)):
            if limits is None:
                continue
            low, high = float(limits[0]), float(limits[1])
            if self._log[axis] and min(low, high) <= 0:
                refused.append(f"{axis} from {min(low, high):g}")
                continue
            self._pinned[axis] = (low, high)
            setter(self._to_drawn_value(low, axis),
                   self._to_drawn_value(high, axis), padding=0)
            box.enableAutoRange(axis=which, enable=False)
        if refused:
            self.set_style_note(
                f"{' and '.join(refused)} has no logarithm, so that limit "
                f"was not applied while the axis is on a log scale.")

    def _to_drawn_value(self, value, axis: str) -> float:
        """One data coordinate as it is DRAWN. The scalar of :meth:`_to_drawn`."""
        drawn = self._to_drawn(np.asarray([float(value)]), axis)
        return float(drawn[0])

    def pinned_limits(self) -> dict:
        """Return the axis limits the user entered for this plot.

        Each value is a ``(low, high)`` pair in data coordinates. An axis that
        still follows automatic ranging has the value ``None``; its current
        visible range is deliberately not returned, because restoring that
        transient range would turn automatic ranging off. The returned
        dictionary is a copy and can be stored or changed safely.

        :returns: ``{"x": limits_or_none, "y": limits_or_none}``.
        """
        return {axis: self._pinned.get(axis) for axis in ("x", "y")}

    def _reapply_pinned(self) -> None:
        """Put the typed limits back, in whatever units are drawn now."""
        box = self.plot.getViewBox()
        for axis, setter, which in (("x", box.setXRange, box.XAxis),
                                    ("y", box.setYRange, box.YAxis)):
            limits = self._pinned.get(axis)
            if limits is None:
                continue
            if self._log[axis] and min(limits) <= 0:
                # The scale just changed under a limit that cannot survive
                # it. Forgetting it is the honest answer -- the axis goes
                # back to its data rather than to a bound nobody typed.
                self._pinned[axis] = None
                continue
            setter(self._to_drawn_value(limits[0], axis),
                   self._to_drawn_value(limits[1], axis), padding=0)
            box.enableAutoRange(axis=which, enable=False)

    def auto_range_axes(self) -> None:
        """Give both axes back to the data. The way out of a typed limit.

        A control that can only be set is a trap: a user who pins x to the
        wrong decade has no way back to the picture they started from except
        reloading the run.
        """
        self._pinned = {"x": None, "y": None}
        box = self.plot.getViewBox()
        box.autoRange()
        # ENABLED LAST, AND THAT ORDER IS THE WHOLE FIX. pyqtgraph's
        # `autoRange()` ends in `setRange(..., disableAutoRange=True)`, so
        # calling it turns auto-ranging OFF -- and this method, whose entire
        # job is to give the axes back to the data, was leaving them frozen
        # on whatever happened to be drawn at the moment it ran.
        #
        # The panel calls this between runs, BEFORE drawing the new table.
        # Measured: run A spanning +-13 followed by run B spanning +-0.6
        # opened run B inside run A's window, twenty times too wide, and
        # "Reset view" -- which autoranges the points that are now there --
        # put it right. That is exactly what was reported.
        box.enableAutoRange(x=True, y=True)
        self._sync_auto_range()

    def aspect_ratio(self) -> Optional[float]:
        """The locked ratio of y units to x units, or ``None`` if unlocked."""
        locked = self.plot.getViewBox().state.get("aspectLocked", False)
        return None if not locked else float(locked)

    def set_aspect_ratio(self, ratio: Optional[float]) -> None:
        """Lock one y unit to ``ratio`` x units. ``None`` unlocks it.

        :param ratio: how many x units one y unit is drawn as wide. 1.0 is
            the square-units lock a Q-Q wants, where the 45-degree diagonal
            is only meaningful if the axes share a scale.
        """
        box = self.plot.getViewBox()
        if ratio is None or float(ratio) <= 0:
            box.setAspectLocked(False)
            return
        box.setAspectLocked(True, ratio=float(ratio))

    # ------------------------------------------------------------------ text

    def font_size(self) -> Optional[int]:
        """The point size the axes are drawn at, or None for the default."""
        return self._font_size

    def set_font_size(self, points: int) -> None:
        """Draw the labels, TICKS and title at ``points``.

        THE TICKS ARE THE HALF THAT WAS MISSING. The old handler passed
        ``tickFont=None`` -- which asks for pyqtgraph's default rather than
        for a size -- so "Font size: 20" enlarged the two axis labels and
        left every tick number at its original size. Measured on the volcano
        before this change: the bottom axis' tick font came back as None at
        every setting, i.e. the control moved two strings out of about
        twenty.
        """
        self._font_size = int(points)
        self.apply_text_style()

    def font_colour(self) -> Optional[str]:
        """The ink chosen for text, or None while it follows the theme."""
        return self._font_colour

    def set_font_colour(self, colour) -> None:
        """Draw every piece of text on the plot in ``colour``.

        Separate from :meth:`restyle`, which resolves the THEME's ink. This
        is the user overriding it for one figure, so it is re-applied after a
        theme switch rather than being quietly reverted by one.
        """
        self._font_colour = None if colour is None else QColor(colour).name()
        self.apply_text_style()

    def apply_text_style(self) -> None:
        """Put the chosen size and ink onto both axes and the title.

        One place, because the size and the colour are set from two different
        menu entries and each has to leave the other's choice standing --
        applying them separately is how "font size" quietly reverts "font
        colour" and the user concludes one of the two is broken.
        """
        colour = self._font_colour or self._foreground
        size = self._font_size
        pen = pg.mkPen(QColor(colour))
        for name in ("bottom", "left"):
            try:
                axis = self.plot.getAxis(name)
            except Exception:           # absent axis
                continue
            axis.setTextPen(pen)
            if size is not None:
                from PySide6.QtGui import QFont

                font = QFont()
                font.setPointSize(int(size))
                axis.setStyle(tickFont=font)
            # AN AXIS WITH NO LABEL IS LEFT ALONE. `setLabel` calls
            # `showLabel()`, so restyling the empty string would make the
            # control panel -- whose x-axis is deliberately unlabelled,
            # because its ticks already name the groups -- grow a blank strip
            # under it the first time anyone changed the font.
            if axis.labelText:
                style = {"color": colour}
                if size is not None:
                    style["font-size"] = f"{int(size)}pt"
                axis.setLabel(axis.labelText, **style)
        title = getattr(self.plot.plotItem, "titleLabel", None)
        if title is not None and title.text:
            if size is not None:
                self.plot.setTitle(title.text, color=colour,
                                   size=f"{int(size) + 2}pt")
            else:
                self.plot.setTitle(title.text, color=colour)
        # A THRESHOLD LINE'S CAPTION IS TEXT, NOT PART OF THE LINE. "p=0.05"
        # and "FDR 5%" used to be recoloured by `set_line_style`, on the
        # reasoning that a red word beside a green line looks wrong. The
        # maintainer's decision (instruction 152 B) is the other way and it
        # is the one that can be stated in a sentence: "a font color that
        # controls the color of all font in the graph". A caption that
        # followed the line would make "all font" untrue, and would be the
        # one string on the figure the font control could not reach.
        for item in self.line_items():
            label = getattr(item, "label", None)
            if label is None:
                continue
            try:
                label.setColor(QColor(colour))
            except Exception:       # not a labelled line
                pass
        legend = getattr(self.plot.plotItem, "legend", None)
        if legend is not None:
            for entry in getattr(legend, "items", ()):
                text = entry[1] if isinstance(entry, (tuple, list)) else None
                if text is None:
                    continue
                try:
                    text.setText(text.text, color=colour)
                except Exception:   # an odd legend item
                    pass

    # ----------------------------------------------------------------- lines

    def line_items(self) -> list:
        """Every LINE on this plot, for a restyle to reach.

        The reference lines and threshold lines added by :meth:`add_line`,
        the Q-Q's diagonal, the residual and scale-location trends, and the
        summary line across a points/jitter group -- all of them, because
        each is a visible line with colour and width controls, and a
        control that reached three of five kinds would be worse than none.

        The scatters are excluded because they have their own controls, and
        the selection ring is excluded because it is a cursor rather than
        data: recolouring it to match the threshold lines would make the
        selection invisible against them.
        """
        if not HAVE_PYQTGRAPH:
            return []
        kinds = (pg.InfiniteLine, pg.PlotDataItem, pg.PlotCurveItem)
        return [item for item in self.plot.plotItem.items
                if isinstance(item, kinds) and item is not self._highlight
                and not isinstance(item, ScatterPlotItem)]

    def axis_items(self) -> list:
        """The four axes, for the line control to reach.

        THE SPINES AND THE TICK MARKS ARE LINES. They were unreachable: the
        axis takes ``foreground`` at CONSTRUCTION and nothing changed it
        afterwards, so the first report on the design -- "doesnt look
        like there is an option to change the axis color for the volcano
        plot" -- was exactly right, and :meth:`line_items` deliberately
        excludes anything that is not a plot item, so no existing call could
        have reached them.
        """
        if not HAVE_PYQTGRAPH:
            return []
        found = []
        for edge in ("bottom", "left", "top", "right"):
            try:
                axis = self.plot.getAxis(edge)
            except Exception:           # absent axis
                continue
            if axis is not None:
                found.append(axis)
        return found

    def line_colour(self) -> Optional[str]:
        """The ink chosen for lines, or None while each keeps its own."""
        return self._line_colour

    def set_line_colour(self, colour) -> int:
        """Colour EVERY line, axis spines and tick marks included.

        ``None`` puts every line back to the colour it was drawn with and the
        axes back to the theme's -- the "Follow the theme" half, which a user
        who has set a colour needs or the freeze the API is intended to fix
        just happens by hand instead of by accident.
        """
        self._line_colour = None if colour is None else QColor(colour).name()
        return self.set_line_style(colour=self._line_colour or "\0theme")

    def set_line_style(self, colour=None,
                       width: Optional[float] = None) -> int:
        """Apply colour and width changes to plot lines.

        Existing dash patterns are preserved. Colour changes also reach axis
        spines and tick marks, while width changes apply only to plot marks
        and reference lines.

        Parameters
        ----------
        colour : QColor-compatible, optional
            New colour, ``None`` to retain each line's colour, or the internal
            ``"\0theme"`` sentinel to restore theme colours.
        width : float, optional
            Pen width in pixels. ``None`` retains current widths.

        Returns
        -------
        int
            Number of line items updated.
        """
        from PySide6.QtGui import QPen

        if not self.plots_available:
            # NOTHING WAS DRAWN, so there is nothing to restyle. Answering
            # zero is the honest reply; reaching into a plot that was never
            # built is the half-built-widget trap `_build_without_pyqtgraph`
            # exists to close.
            return 0
        theme = colour == "\0theme"
        if theme:
            colour = None
        touched = 0
        for item in self.line_items():
            existing = self._pen_of(item)
            pen = QPen(existing) if existing is not None else pg.mkPen(MUTED)
            # THE COLOUR IT WAS DRAWN WITH, REMEMBERED ONCE. Without it
            # "Follow the theme" has nothing to go back to and would have to
            # invent a colour, which is the same class of mistake as
            # persisting a resolved default.
            if not hasattr(item, "_spacr_base_colour"):
                item._spacr_base_colour = QColor(pen.color()).name()
            if theme:
                pen.setColor(QColor(item._spacr_base_colour))
            elif colour is not None:
                pen.setColor(QColor(colour))
            if width is not None:
                pen.setWidthF(float(width))
            item.setPen(pen)
            touched += 1
        if colour is not None or theme:
            ink = self._foreground if theme else colour
            axis_pen = pg.mkPen(QColor(ink))
            for axis in self.axis_items():
                axis.setPen(axis_pen)
                # THE TICK MARKS, SEPARATELY. `setPen` paints the spine;
                # pyqtgraph draws the little dashes with `tickPen` and falls
                # back to the spine's pen only while none is set -- so an
                # axis that has ever been given one keeps drawing its ticks
                # in the old ink unless this line is here.
                try:
                    axis.setTickPen(axis_pen)
                except Exception:   # older pyqtgraph
                    pass
        # THE CAPTIONS ARE NOT TOUCHED HERE. "p=0.05" is text and follows the
        # font control -- see :meth:`apply_text_style`.
        return touched

    @staticmethod
    def _pen_of(item):
        """The pen an item is currently drawn with, whatever kind it is."""
        pen = getattr(item, "pen", None)
        if pen is not None and not callable(pen):
            return pen
        options = getattr(item, "opts", None)
        if isinstance(options, dict):
            return options.get("pen")
        return None

    # ------------------------------------------------- a column onto a channel

    def colour_by_column(self, column: str, colormap: str = "viridis") -> int:
        """Colour every point by ``column`` through ``colormap``. Returns n.

        This maps a continuous data column to a visual channel rather than
        assigning one fixed point colour. The status line reports the numeric
        range and the number of missing values; missing values are drawn grey
        instead of being placed at the bottom of the scale.

        :raises ValueError: for a column that is not there or not continuous,
            and for a colormap this build does not provide.
        """
        frame = self.frame()
        if frame is None:
            raise ValueError("this plot holds no table to colour by")
        if column not in frame.columns:
            raise ValueError(
                f"no column {column!r}; this table has "
                f"{', '.join(map(str, frame.columns))}")
        if column not in self.numeric_columns():
            raise ValueError(
                f"{column!r} is not a continuous column, and a colour scale "
                f"on a category invents an order the data does not have")
        # ASKED BEFORE pyqtgraph IS. `pg.colormap.get` resolves a name by
        # opening a file in its own package directory, so an unknown one
        # raises FileNotFoundError naming a path inside site-packages -- a
        # traceback about the library's install layout, in answer to a user
        # picking a colour scale. Measured on 'jet'.
        if colormap not in COLORMAPS:
            raise ValueError(
                f"unknown colormap {colormap!r}; this build offers "
                f"{', '.join(COLORMAPS)}")
        table = pg.colormap.get(colormap)

        values = _finite(frame[column].to_numpy())
        usable = values[~np.isnan(values)]
        low, high = float(usable.min()), float(usable.max())
        lookup = table.getLookupTable(nPts=COLORMAP_STEPS, alpha=True)
        cache: dict = {}
        missing_brush = pg.mkBrush(QColor(MISSING_COLOUR))

        painted, blank = 0, 0
        for item in self._scatter_items():
            rows = self._rows_of(item)
            if rows is None:
                continue
            self._remember_point_style(item)
            picked = values[rows]
            steps = np.clip(
                np.round((picked - low) / (high - low) * (COLORMAP_STEPS - 1)),
                0, COLORMAP_STEPS - 1)
            brushes = []
            for value, step in zip(picked, steps):
                if np.isnan(value):
                    brushes.append(missing_brush)
                    blank += 1
                    continue
                index = int(step)
                brush = cache.get(index)
                if brush is None:
                    r, g, b, a = (int(c) for c in lookup[index])
                    brush = cache[index] = pg.mkBrush(QColor(r, g, b, a))
                brushes.append(brush)
            item.setBrush(brushes)
            painted += len(brushes)

        self._colour_column = (column, colormap)
        note = (f"Coloured by {column} ({colormap}): {low:.3g} at the dark "
                f"end to {high:.3g} at the bright end.")
        if blank:
            note += (f" {blank} point{'s' if blank != 1 else ''} have no "
                     f"{column} and are grey.")
        self.set_style_note(note)
        return painted

    def shape_by_column(self, column: str) -> int:
        """Draw each value of ``column`` as its own marker. Returns n shaped.

        :raises ValueError: for a column that is not there, or one with more
            values than there are shapes a reader can tell apart. Refused
            rather than truncated: reusing a circle for the ninth and the
            first value would draw two different things identically, which is
            worse than not offering the column at all.
        """
        frame = self.frame()
        if frame is None:
            raise ValueError("this plot holds no table to take shapes from")
        if column not in frame.columns:
            raise ValueError(
                f"no column {column!r}; this table has "
                f"{', '.join(map(str, frame.columns))}")
        text = frame[column].astype(str)
        names = sorted(set(text))
        if len(names) > MAX_SHAPE_VALUES:
            raise ValueError(
                f"{column!r} has {len(names)} values and only "
                f"{MAX_SHAPE_VALUES} shapes are distinguishable")
        if len(names) < 2:
            raise ValueError(
                f"{column!r} has one value, so every point would be the same "
                f"shape and the column would say nothing")
        order = {name: i for i, name in enumerate(names)}
        codes = text.map(order).to_numpy()

        shaped = 0
        for item in self._scatter_items():
            rows = self._rows_of(item)
            if rows is None:
                continue
            self._remember_point_style(item)
            symbols = [SHAPE_SYMBOLS[int(codes[row])][0] for row in rows]
            item.setSymbol(symbols)
            shaped += len(symbols)

        self._shape_column = column
        legend = ", ".join(f"{name} is a {SHAPE_SYMBOLS[i][1]}"
                           for i, name in enumerate(names))
        self.set_style_note(f"Shaped by {column}: {legend}.")
        return shaped

    def clear_column_mapping(self) -> int:
        """Put the original colours and shapes back. Returns items restored.

        The brushes and symbols each scatter was BUILT with are kept the
        first time a mapping touches it, because they are the only record of
        what the plot's own colouring said -- the compartment split, the
        single-guide genes, the influential wells. Recomputing them here
        would need this class to know every subclass's rule, and a "restore"
        that guessed would quietly replace one sentence with another.
        """
        restored = 0
        for item in self._scatter_items():
            saved = getattr(item, "_spacr_point_style", None)
            if saved is None:
                continue
            brushes, symbols = saved
            item.setBrush(list(brushes))
            item.setSymbol(list(symbols))
            item._spacr_point_style = None
            restored += 1
        self._colour_column = None
        self._shape_column = None
        self.set_style_note("")
        return restored

    @staticmethod
    def _rows_of(item):
        """The FRAME ROWS behind a scatter's points, as an integer array.

        ``add_scatter`` puts them on the item as its per-point ``data``, and
        that is the only honest source: a Q-Q is sorted and a control panel
        is split into groups, so the nth point of a scatter is not the nth
        row of the table -- see :meth:`add_scatter`, where the same trap is
        written out in full.
        """
        data = getattr(item, "data", None)
        if data is None or not len(data):
            return None
        try:
            rows = np.asarray(data["data"])
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        if rows.dtype == object:
            if any(row is None for row in rows):
                return None
            rows = rows.astype("int64")
        return rows

    @staticmethod
    def _remember_point_style(item) -> None:
        """Keep what a scatter looked like before a mapping touched it."""
        if getattr(item, "_spacr_point_style", None) is not None:
            return
        item._spacr_point_style = (list(item.data["brush"]),
                                   list(item.data["symbol"]))

    # ------------------------------------------------------------ dimensions

    def set_screen_size(self, width: int, height: int) -> None:
        """Set the plot's fixed on-screen size in pixels.

        This does not change export dimensions; use :meth:`set_export_size`
        to configure the exported page.
        """
        if self._size_bounds is None:
            self._size_bounds = (self.minimumWidth(), self.minimumHeight(),
                                 self.maximumWidth(), self.maximumHeight())
        self.setFixedSize(int(width), int(height))

    def clear_screen_size(self) -> None:
        """Let the layout size the plot again, keeping its original floors.

        NOT ``setMinimumSize(0, 0)``. `RegressionResultsPanel` gives the
        volcano ``setMinimumHeight(240)`` so a splitter cannot collapse it to
        a sliver; releasing the widget to nothing would silently drop that
        floor, and the plot would then vanish the first time the user dragged
        the divider.
        """
        if self._size_bounds is None:
            return
        min_w, min_h, max_w, max_h = self._size_bounds
        self.setMinimumSize(min_w, min_h)
        self.setMaximumSize(max_w if max_w else QWIDGET_SIZE_MAX,
                            max_h if max_h else QWIDGET_SIZE_MAX)
        self._size_bounds = None

    def canvas_shape(self) -> str:
        """Which of :data:`CANVAS_SHAPES` the FIGURE is held at."""
        return self._canvas_shape

    def canvas_ratio(self) -> Optional[float]:
        """Height over width for the current shape, or None when free."""
        return dict(CANVAS_SHAPES).get(self._canvas_shape)

    def set_canvas_shape(self, name: str) -> None:
        """Set the canvas shape to square, wide, tall, or free.

        The shape constrains both the on-screen canvas and exported page; it
        does not alter the data-unit aspect ratio configured by
        :meth:`set_aspect_ratio`.
        """
        name = str(name)
        if name not in dict(CANVAS_SHAPES):
            raise ValueError(
                f"unknown shape {name!r}; known shapes: "
                f"{', '.join(shape for shape, _ in CANVAS_SHAPES)}")
        self._canvas_shape = name
        self._apply_canvas_shape()

    def _chrome_height(self) -> int:
        """How much of this widget is NOT the plot: controls, status, margins.

        Measured from the layout's OTHER items rather than from the plot's
        current height, which is the number about to be changed -- reading it
        would make the next resize depend on the last one and the shape would
        creep a few pixels every pass.
        """
        layout = self.layout()
        if layout is None:                  # pragma: no cover - always laid out
            return 0
        margins = layout.contentsMargins()
        total = margins.top() + margins.bottom()
        total += max(0, layout.count() - 1) * max(0, layout.spacing())
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item is None or item.widget() is self.plot:
                continue
            total += int(item.sizeHint().height())
        return int(total)

    def _apply_canvas_shape(self) -> None:
        """Hold the plot at the shape's proportion on screen.

        CLAMPED TO THE ROOM THERE IS. A square imposed on a panel that is
        twice as wide as it is tall would push the controls and the status
        line off the bottom, so the shape gives way on the long side: the
        height stops at the room available and the width is capped to keep
        the proportion exact. That is a stable fixed point rather than a
        negotiation -- the next pass computes the same two numbers.
        """
        if not self.plots_available or self._shaping:
            return
        ratio = self.canvas_ratio()
        self._shaping = True
        try:
            if ratio is None:
                self.plot.setMinimumHeight(0)
                self.plot.setMaximumHeight(QWIDGET_SIZE_MAX)
                self.plot.setMaximumWidth(QWIDGET_SIZE_MAX)
                return
            room = max(1, int(self.height()) - self._chrome_height())
            width = max(1, int(self.plot.width()))
            height = max(1, min(int(round(width * ratio)), room))
            self.plot.setFixedHeight(height)
            self.plot.setMaximumWidth(max(1, int(round(height / ratio))))
        finally:
            self._shaping = False

    def resizeEvent(self, event) -> None:      # noqa: N802 - Qt's own name
        """Re-impose the canvas shape whenever the box around it changes."""
        super().resizeEvent(event)
        if self._canvas_shape != "free":
            self._apply_canvas_shape()

    @contextmanager
    def _held_at_the_page_shape(self):
        """Give the SCENE the canvas shape's proportions for one render.

        WHY THE PAGE ALONE IS NOT ENOUGH. Both raster and vector export map
        the scene onto the page with ``Qt::KeepAspectRatio``, so asking for a
        square page while the scene is still 900x433 does not make a square
        FIGURE -- it makes a square file with the plot pressed into the top
        half of it. Measured on the volcano before this existed: a 900x900
        PNG whose ink stopped at row 428.

        So the plot item is resized to the shape first and the exporters
        follow it, which is the same thing the on-screen shape does through
        :meth:`_apply_canvas_shape`. The geometry is put back in ``finally``,
        including when the render raises.

        A free canvas yields without touching anything.
        """
        from PySide6.QtCore import QRectF

        ratio = self.canvas_ratio()
        item = getattr(self.plot, "plotItem", None)
        if ratio is None or item is None:
            yield False
            return
        before = QRectF(item.geometry())
        width = float(before.width())
        if not np.isfinite(width) or width <= 0:
            yield False
            return
        item.setGeometry(QRectF(before.x(), before.y(), width, width * ratio))
        try:
            yield True
        finally:
            item.setGeometry(before)

    def export_size(self) -> tuple:
        """``(width mm, height mm)`` of a saved page.

        A height of None means "follow the plot's own aspect" -- which is
        what a FREE canvas does. A shape set by :meth:`set_canvas_shape` is
        an answer to that question, so it is honoured here: this is the one
        place every export path reads, so it is the one place the shape has
        to land to reach all three of them.
        """
        ratio = self.canvas_ratio()
        if self._export_height_mm is None and ratio is not None:
            return (float(self._export_width_mm),
                    float(self._export_width_mm) * ratio)
        return (float(self._export_width_mm), self._export_height_mm)

    def set_export_size(self, width_mm: float,
                        height_mm: Optional[float] = None) -> None:
        """Set the PAGE a PDF or SVG is written onto, in millimetres.

        THIS DOES NOT MOVE THE PLOT ON SCREEN. See :meth:`set_screen_size`.

        :param width_mm: page width. :data:`EXPORT_WIDTH_MM` is a journal's
            double-column width and is the default.
        :param height_mm: page height, or None to follow the plot's own
            aspect so nothing is stretched.
        """
        self._export_width_mm = float(width_mm)
        self._export_height_mm = (None if height_mm is None
                                  else float(height_mm))

    def _style_menu(self, position) -> None:
        """Right-click: build the menu and show it."""
        self.build_style_menu().exec(self.plot.mapToGlobal(position))

    @staticmethod
    def _gated(menu, label: str, callback, reason: str):
        """Add an enabled menu action or a disabled action with its reason.

        Disabled actions include the reason in both the visible label and the
        tooltip so unavailable plot controls remain discoverable.
        """
        if not reason:
            if callback is None:
                # A CHECKABLE ENTRY WIRES ITSELF. `addAction(text, callable)`
                # connects `triggered`, whose bool is dropped for a slot that
                # does not ask for one -- so a checkable entry connected that
                # way reports the state it had BEFORE the press, i.e. never
                # turns anything on. Those connect `toggled` themselves.
                return menu.addAction(label)
            return menu.addAction(label, callback)
        action = menu.addAction(f"{label}  —  {reason}")
        action.setEnabled(False)
        action.setToolTip(reason)
        return action

    @staticmethod
    def _group(menu, title: str):
        """Create a tooltip-enabled submenu owned by ``menu``.

        Explicit parent ownership keeps the submenu and its actions alive
        after this helper returns.
        """
        from PySide6.QtWidgets import QMenu

        submenu = QMenu(title, menu)
        submenu.setToolTipsVisible(True)
        menu.addMenu(submenu)
        return submenu

    @staticmethod
    def _checkable(menu, options) -> None:
        """Add checkable menu options, including disabled options with reasons.

        Each option is ``(label, callback, checked)`` with an optional fourth
        item containing the reason the option is unavailable.
        """
        for option in options:
            label, callback, checked = option[0], option[1], option[2]
            reason = option[3] if len(option) > 3 else ""
            if reason:
                entry = menu.addAction(f"{label}  —  {reason}")
                entry.setEnabled(False)
                entry.setToolTip(reason)
                continue
            action = menu.addAction(label, callback)
            action.setCheckable(True)
            action.setChecked(bool(checked))

    def build_style_menu(self):
        """The right-click menu, built from what the plot actually has on it.

        SEPARATE FROM SHOWING IT so the menu can be inspected without a modal
        event loop. `QMenu.exec` blocks until the user picks something and is
        not patchable from a test -- it is a C++ slot, and assigning over it
        leaves the real one dispatching -- so a test that reached in to read
        the entries hung the suite instead of failing it.

        THE ORDER IS THE ORDER OF USE, not alphabetical, and it is the whole
        design of the design:

          * the two entries every user reaches for stay at the TOP LEVEL and
            one click away. A menu reorganised until nothing is one click
            away is worse than the flat list it replaced.
          * then what changes the CLAIM -- which rows are drawn, which
            p-value the axis means, where the cut is, what zero is measured
            from. These come first because they change what the figure says.
          * then what changes the LOOK -- the mark, the colour, the axes, the
            appearance, the size.
          * then, alone under its own heading, the one entry that re-runs the
            analysis. A user reaching for "Point size" must not be one slip
            away from starting a fit.

        Groups appear only when this plot HAS the thing they hold, so a Q-Q
        is not offered a p-value axis it does not draw and a volcano is not
        offered a violin.
        """
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        # A DISABLED ENTRY'S REASON HAS TO BE READABLE. Qt hides action
        # tooltips unless a menu asks for them, so without this the greyed
        # entries would be exactly the "present but inert" control that
        # instruction 106 forbids. Each group repeats it -- see `_group`.
        menu.setToolTipsVisible(True)
        menu.addAction("Reset view", self.auto_range_axes)
        # ONE DOOR (187 D). There were two -- "Export…", which wrote the plot
        # as it looked, and "Save styled", which opened the preview where
        # ink, background, grid and text size are chosen FOR THE FILE. Export
        # wrote with no preview and no styling pass, which is where the
        # reported "massive text and a tiny misaligned graph" came from: the
        # page is sized in millimetres and nothing scaled the text to it.
        #
        # A second door that produces a worse file is not a shortcut, so the
        # remaining one shows what it will write. `export` itself stays as
        # the API both paths and every test use.
        menu.addAction("Save figure…", lambda: self.save_styled())
        # AND THE WHOLE THING (instruction 223). Beside "Save figure…" rather
        # than replacing it: a user who wants a png for Slack should not have
        # to take a folder of five files, and one who wants the figure
        # checkable six months from now should not have to assemble it by
        # hand. Two doors here is not the 187 D duplication -- they produce
        # DIFFERENT things, and each says which.
        # AND CHANGE WHAT KIND OF GRAPH IT IS. On the base class, so every
        # plot that holds a spec gets it and no plot that does not shows an
        # empty submenu. `_offer_graph_kinds` returns without adding
        # anything when there is nothing to offer.
        self._offer_graph_kinds(menu)

        bundle = menu.addAction("Save",
                                lambda: self.export_bundle())
        bundle.setToolTip(
            "Writes a FOLDER: the figure as pdf and png, the rows it was "
            "drawn from, and the test that was run on them with its "
            "assumptions. A pdf on its own cannot be checked -- six months "
            "later the question is what the numbers were and whether the "
            "difference was tested, and a figure file answers neither.")

        # ------------------------------------------------ what the plot CLAIMS
        claims = False
        if self._levels:
            # WHICH ROWS, not how they look. First, because a filtered plot
            # that looks like a restyled one is read as the whole screen.
            self._checkable(self._group(menu, "Show"), self._levels)
            claims = True
        if self._p_values:
            # THE Y-AXIS ITSELF. Above the effect-size cut because it changes
            # what the axis MEANS, while the cut changes where a line is
            # drawn on it.
            self._checkable(self._group(menu, "p-value"), self._p_values)
            claims = True
        if self._corrections:
            # WHICH MULTIPLE-TESTING CORRECTION IS DRAWN. Beside the p-value
            # axis because the two are one question -- what the height and
            # the colour MEAN -- and above the effect-size cut for the same
            # reason the axis is.
            group = self._group(menu, "Correction")
            self._checkable(group, self._corrections)
            if self._correction_writer is not None:
                # NOT LEFT AMBIGUOUS. A plot recorrected to something other
                # than the run's is showing a different analysis from the
                # results.csv beside it; the status line says so, and this is
                # the other half of the answer -- the numbers on screen,
                # written out, so the table and the figure can be made to
                # agree rather than merely be known to differ.
                group.addAction("Write this correction as a table…",
                                self._correction_writer)
            claims = True
        if self._encodings:
            # WHICH CHANNEL CARRIES THE FDR. Under the correction rather than
            # beside "Colour by", because it changes what the picture MEANS
            # and not how it looks -- and because two of its entries take the
            # colour channel away from the colouring below.
            self._checkable(self._group(menu, "Show the FDR as"),
                            self._encodings)
            claims = True
        options, multiplier, on_multiplier = self._thresholds
        if options:
            # It changes which points count as hits, so it belongs neither
            # with the restyling below nor with the re-fit at the end -- it
            # re-reads a fit that has already happened, like the baseline.
            cut = self._group(menu, "Effect-size cut")
            if multiplier is not None and on_multiplier is not None:
                cut.addAction(
                    f"Multiplier: {multiplier:g}…",
                    lambda: self._ask_threshold_multiplier(multiplier,
                                                           on_multiplier))
            self._checkable(cut, options)
            claims = True
        if self._baselines:
            # WHAT THE EFFECTS ARE MEASURED FROM. It moves the points and
            # does NOT change the fit: it changes where zero is drawn on a
            # fit that has already happened.
            self._checkable(self._group(menu, "Measured from"),
                            self._baselines)
            claims = True
        if self._smoothers is not None:
            # NOT under a heading that could read as a choice of fit. These
            # are drawn on top of one; the heading says which it is.
            self._checkable(
                self._group(menu, "Diagnostic curve (decides no hit)"),
                self._smoother_options())
            claims = True
        if claims:
            menu.addSeparator()

        # -------------------------------------------------- how it is DRAWN
        if self._marks:
            # Every option is offered -- including the ones that mislead for
            # the data on screen, because a menu that hides them cannot
            # explain why -- and the plot says so in its status line once the
            # choice is made.
            self._checkable(self._group(menu, "Draw as"), self._marks)
        points = self.point_reason()
        colour = self._group(menu, "Colour by")
        self._gated(colour, "Point colour…", self._ask_point_colour, points)
        self._gated(colour, "Colour by a column…", self._ask_colour_column,
                    self.colour_map_reason())
        if self._compartments:
            # ITS OWN LIST, because this is the one that can be long -- and
            # it holds only what this screen actually has, so a choice that
            # would colour nothing is not offered at all.
            self._checkable(self._group(colour, "Colour by localisation"),
                            self._compartments)
        if self._colour_column or self._shape_column:
            colour.addAction("Back to this plot's own colouring",
                             self.clear_column_mapping)

        axes = self._group(menu, "Axes")
        axes.addAction("Axis labels…", self._ask_labels)
        axes.addAction("Axis limits…", self._ask_axis_limits)
        axes.addAction("Axis limits: back to automatic", self.auto_range_axes)
        # NAMED FOR WHAT IT DOES. It was "Aspect ratio", which everybody
        # read as "make the figure square" -- and it is a statement about the
        # DATA: one y unit drawn as n x units, which is what a Q-Q's
        # 45-degree diagonal needs and is nothing to do with the page. The
        # page is "Shape", under Appearance.
        axes.addAction("Lock axis scales (1 y unit = n x units)…",
                       self._ask_aspect_ratio)
        # THE SPLIT, ASKED FOR BY NAME. Under Axes because it is a statement
        # about the RULER: it takes an empty stretch out so the rest of the
        # screen gets its height back. It does NOT make a stepped adjusted P
        # continuous, and the status line it writes says so.
        axes.addAction("Split the y axis…", self._ask_y_split)
        if self.y_split() is not None:
            low, high = self.y_split()
            axes.addAction(f"Y axis split ({low:g}-{high:g}): remove",
                           self.clear_y_split)
        for axis in ("x", "y"):
            # CHECKABLE, AND GATED. The tick is the state, and an axis that
            # cannot be logged says why in the entry itself rather than
            # sitting there live and doing nothing.
            reason = self.log_reason(axis)
            entry = self._gated(axes, f"Log {axis} axis", None, reason)
            entry.setCheckable(True)
            entry.setChecked(self._log[axis])
            if not reason:
                entry.toggled.connect(
                    lambda on, which=axis: self.set_log_axes(**{which: on}))

        look = self._group(menu, "Appearance")
        self._gated(look, "Point size…", self._ask_point_size, points)
        self._gated(look, "Opacity…", self._ask_opacity, points)
        self._gated(look, "Shape by a column…", self._ask_shape_column,
                    self.shape_reason())
        shape = self._group(look, GRAPH_SHAPE_MENU)
        for name, _ratio in CANVAS_SHAPES:
            entry = shape.addAction(
                CANVAS_SHAPE_LABELS.get(name, name),
                lambda _checked=False, which=name:
                self.set_canvas_shape(which))
            entry.setCheckable(True)
            entry.setChecked(self._canvas_shape == name)
            # The stored name travels with the entry, so a caller reading the
            # menu back does not have to un-translate the label.
            entry.setData(name)
        look.addAction("Font size…", self._ask_font_size)
        # EXACTLY TWO COLOUR CONTROLS, split by what a mark IS rather than by
        # which part of the code draws it (instruction 152 B). Font colour is
        # every piece of text, tick LABELS included; Line colour is every
        # line, the axis spines and tick MARKS included.
        look.addAction("Font colour…", self._ask_font_colour)
        self._gated(look, "Line colour…", self._ask_line_colour,
                    self.line_colour_reason())
        self._gated(look, "Line width…", self._ask_line_width,
                    self.line_reason())
        if self._font_colour is not None or self._line_colour is not None:
            look.addAction("Follow the theme (colours)", self.follow_the_theme)
        grid = look.addAction("Grid")
        grid.setCheckable(True)
        grid.setChecked(self.grid_shown())
        grid.toggled.connect(self.set_grid)
        if self._legend_box.isEnabled():
            legend = look.addAction("Legend")
            legend.setCheckable(True)
            legend.setChecked(self._legend_box.isChecked())
            legend.toggled.connect(self._legend_box.setChecked)

        # NAMED SEPARATELY BECAUSE THEY ARE DIFFERENT QUANTITIES. "Dimensions"
        # as one entry is the misleading version: on the live plot it is the
        # widget's size, on a saved figure it is the page, and a user who sets
        # one and inspects the other finds nothing changed.
        size = self._group(menu, "Size")
        size.addAction("Size on screen…", self._ask_screen_size)
        size.addAction("Exported page size…", self._ask_export_size)
        size.addAction("Size on screen: back to automatic",
                       self.clear_screen_size)

        if self._style is not None:
            # THE FIGURE'S OWN SETTINGS, under one heading and below the
            # plot's, because they belong to whoever supplied them and a
            # reader has to be able to tell the two apart.
            style, on_change, choices = self._style
            group = self._group(menu, "Figure style")
            add_style_entries(group, style, on_change, choices=choices)
            # SAVABLE, not only editable -- the half the maintainer restated
            # on 2026-08-16 ("each figure should be editable and savable").
            # A restyle the user cannot keep is a restyle they redo every
            # time they need the picture.
            group.addSeparator()
            add_style_file_entries(group, style, on_change, parent=self,
                                   note=self.set_style_note)
        if self._refit is not None:
            callback, label = self._refit
            # A SECTION, not another line in the list. Everything above
            # restyles; below here the numbers change.
            menu.addSection("Re-runs the analysis")
            menu.addAction(label, callback)
        return menu

    @staticmethod
    def _paint_items(item) -> list:
        """Every item under ``item`` with an opinion about being exported."""
        found, stack = [], [item]
        while stack:
            current = stack.pop()
            if hasattr(current, "setExportMode"):
                found.append(current)
            stack.extend(current.childItems())
        return found

    @classmethod
    def _paint_scene(cls, item, painter, target, source) -> None:
        """Render the scene onto ``painter`` AS A FIGURE, not as a screenshot.

        THE EXPORT MODE IS THE WHOLE POINT. A ScatterPlotItem draws its
        markers from a cached pixmap atlas -- that cache is why 1,215 points
        pan at no cost -- and ``scene.render`` into a vector device copies
        those pixmaps straight through. Measured on the volcano: a plain
        render gave an SVG with 50 ``<image>`` elements and ONE ``<path>``,
        i.e. fifty little bitmaps of a dot in a file that claims to be
        vector. With pyqtgraph's export mode on, the same plot gives 51
        ``<path>`` elements and no ``<image>`` at all, because the scatter
        redraws its symbols through the painter instead of blitting them.

        The PDF was written before this was understood and had the identical
        defect, so both paths go through here now: "true vector, not a bitmap
        in a PDF wrapper" was only true of the axes and the text.
        """
        marks = cls._paint_items(item)
        for mark in marks:
            mark.setExportMode(True, {"painter": painter, "antialias": True})
        try:
            item.scene().render(painter, target, source)
        finally:
            for mark in marks:
                mark.setExportMode(False)

    @staticmethod
    def _all_compartments():
        """The "colour every compartment" sentinel, or ``None``."""
        try:
            from ...localisation import ALL

            return ALL
        except Exception:                                # noqa: BLE001
            return None

    @staticmethod
    def _categorical_brushes(values):
        """``(brushes, legend)`` for one colour-per-value, counted.

        SHARED BY TWO CALLERS, which is why it is a method: the condition
        colouring and "Colour by localisation -> all" want exactly the same
        thing, and the second was written as a Python loop over brushes
        before this existed -- which is precisely the 45 ms of a 48 ms redraw
        the first one was rewritten to avoid.

        Categorical codes are computed in C. The counts come from
        `np.bincount` over the CODES rather than a `value_counts` on the
        frame, because this path exists to keep pandas out of the per-point
        work and a group-by here would put a chunk of it straight back.
        """
        import pandas as _pd

        categorical = _pd.Categorical(_pd.Series(values).astype(str))
        names = list(categorical.categories)
        palette = [pg.mkBrush(colour_for(i)) for i, _ in enumerate(names)]
        unknown = pg.mkBrush(colour_for(0))
        brushes = [palette[c] if c >= 0 else unknown
                   for c in categorical.codes]
        counts = np.bincount(categorical.codes[categorical.codes >= 0],
                             minlength=len(names))
        legend = {f"{name} ({int(counts[i])})": colour_for(i)
                  for i, name in enumerate(names)}
        return brushes, legend

    #: The marker shapes a second colour-by column cycles through, in the
    #: order they are used. FOUR, and they are the four pyqtgraph symbols a
    #: reader can actually tell apart at eight pixels -- a star and a
    #: pentagon are the same blob at that size, and a legend that
    #: distinguishes them is a legend that lies about what is on screen.
    SHAPES = ("o", "s", "t", "d")

    @classmethod
    def _categorical_symbols(cls, values):
        """Return point symbols and the level-to-symbol legend mapping.

        Shapes repeat after :data:`SHAPES` is exhausted. The returned mapping
        records repeated symbols explicitly so the legend represents the
        displayed encoding.
        """
        import pandas as _pd

        categorical = _pd.Categorical(_pd.Series(values).astype(str))
        names = list(categorical.categories)
        shapes = {name: cls.SHAPES[i % len(cls.SHAPES)]
                  for i, name in enumerate(names)}
        order = [cls.SHAPES[i % len(cls.SHAPES)] for i in range(len(names))]
        out = [order[c] if c >= 0 else cls.SHAPES[0]
               for c in categorical.codes]
        return out, shapes

    @staticmethod
    def _levels_of(values):
        """The level names of a categorical column, in the drawn order."""
        import pandas as _pd

        return list(_pd.Categorical(_pd.Series(values).astype(str))
                    .categories)

    @staticmethod
    def _opacity_alphas(count: int):
        """The alpha each of ``count`` levels is drawn at, faintest first.

        One source of truth for the ramp, because the legend has to name the
        same fades the points were drawn with; two ramps that drift are a key
        that describes a different picture.
        """
        floor, top = 70, 255
        if count <= 0:
            return []
        step = 0 if count < 2 else (top - floor) / (count - 1)
        return [int(round(floor + step * i)) for i in range(count)]

    @classmethod
    def _categorical_opacity(cls, brush_list, values, count):
        """Fade each point by its level in ``values``. THE THIRD CHANNEL.

        The levels are spread over a floor and full opacity rather than
        [0, 1]: a fully transparent point is an absent point, and a channel
        that can delete a datapoint is not an encoding.
        """
        names = cls._levels_of(values)
        if not names:
            return brush_list
        import pandas as _pd

        categorical = _pd.Categorical(_pd.Series(values).astype(str))
        alphas = cls._opacity_alphas(len(names))
        base = list(brush_list) if brush_list is not None else \
            [pg.mkBrush(colour_for(0))] * count
        out = []
        for index, code in enumerate(categorical.codes[:len(base)]):
            brush = base[index]
            colour = QColor(brush.color()) if hasattr(brush, "color") \
                else QColor(colour_for(0))
            colour.setAlpha(alphas[code] if code >= 0 else 255)
            out.append(pg.mkBrush(colour))
        # A frame shorter than the drawn points leaves the rest untouched
        # rather than dropping them: a missing level is not a reason for a
        # point to vanish.
        out.extend(base[len(out):])
        return out

    def _export_ground(self):
        """Return the page colour for the current export as a ``QColor``.

        A background chosen in the save dialog takes precedence for the
        duration of that export. Otherwise the saved-figure appearance is
        used; transparent is the fallback when no ground is configured or
        the preference cannot be read.
        """
        chosen = getattr(self, "_chosen_ground", "")
        if chosen:
            return QColor(str(chosen))
        try:
            from ...figure_style import saved_figure_appearance

            look = saved_figure_appearance()
            if look is not None and getattr(look, "ground", None):
                return QColor(str(look.ground))
        except Exception:                                # noqa: BLE001
            LOG.debug("could not resolve the export ground", exc_info=True)
        return QColor(0, 0, 0, 0)

    @classmethod
    def _wear_the_print_look(cls, item) -> list:
        """Repaint the CHROME for the page. Returns the undo callables.

        THE CHROME FLIPS, THE DATA DOES NOT, and that is the whole design
        (150 A). A blanket white-to-black would turn a white data point black,
        which on a volcano is the colour of "not a hit" -- it would change
        what the figure SAYS. So the axes, their text and the title go through
        `export_colour(..., kind='chrome')` and the marks are not touched at
        all: not passed with a different kind, not passed.

        `export_colour` returns None for anything it should leave alone, and
        it decides on LEGIBILITY against the page rather than on the theme --
        so a light-mode save changes nothing, which is what makes 'print' safe
        as the default.
        """
        try:
            from ...figure_style import export_colour, saved_figure_appearance
        except Exception:                                # noqa: BLE001
            LOG.debug("the saved-figure look is unavailable", exc_info=True)
            return []
        try:
            look = saved_figure_appearance()
        except Exception:                                # noqa: BLE001
            return []
        if look is None or getattr(look, "mode", "") == "screen":
            return []

        undo = []

        def repaint(getter, setter, current):
            replacement = export_colour(current, "chrome", look)
            if replacement is None:
                return
            setter(replacement)
            undo.append(lambda old=current, put=setter: put(old))

        plot_item = getattr(item, "plotItem", item)
        for edge in ("bottom", "left", "top", "right"):
            try:
                axis = plot_item.getAxis(edge)
            except Exception:
                continue
            if axis is None:
                continue
            pen = axis.pen()
            repaint(None, lambda c, a=axis: a.setPen(pg.mkPen(c)),
                    pen.color().name() if pen is not None else "")
            text_pen = getattr(axis, "textPen", None)
            if callable(text_pen):
                current = text_pen()
                repaint(None,
                        lambda c, a=axis: a.setTextPen(pg.mkPen(c)),
                        current.color().name() if current is not None else "")
        title = getattr(plot_item, "titleLabel", None)
        if title is not None and getattr(title, "text", ""):
            colour = str((title.opts or {}).get("color") or "")
            if colour:
                text = title.text
                repaint(None,
                        lambda c, t=text, p=plot_item: p.setTitle(t, color=c),
                        colour)
        return undo

    @staticmethod
    def _page_source(item):
        """Return the visible item bounds and aspect ratio for vector export."""
        scene = item.scene()
        if scene is not None and hasattr(item, "sceneBoundingRect"):
            source = item.sceneBoundingRect()
        else:
            source = item.boundingRect()
        width = float(source.width())
        height = float(source.height())
        if (not np.isfinite(width) or not np.isfinite(height)
                or width <= 0 or height <= 0):
            return None, 0.0
        return source, height / width

    @staticmethod
    def _pdf_resolution(source_width: float, page_width_mm: float) -> int:
        """Choose a safe PDF coordinate scale for the visible plot width."""
        try:
            resolution = float(source_width) * 25.4 / float(page_width_mm)
        except (TypeError, ValueError, ZeroDivisionError):
            return _MIN_PDF_RESOLUTION
        if not np.isfinite(resolution) or resolution <= 0:
            return _MIN_PDF_RESOLUTION
        return min(max(int(round(resolution)), _MIN_PDF_RESOLUTION),
                   _MAX_PDF_RESOLUTION)

    @classmethod
    def _export_pdf(cls, item, path, width_mm: float = EXPORT_WIDTH_MM,
                    height_mm: Optional[float] = None) -> None:
        """Render a plot item into a vector PDF.

        pyqtgraph has no PDF exporter, so the scene is painted into a
        QPdfWriter with the same QPainter it draws itself with -- which keeps
        the text as text and the lines as lines. A raster PNG dropped into a
        PDF would satisfy the file extension and nothing else.

        :param width_mm: page width; :data:`EXPORT_WIDTH_MM` is a journal's
            double-column width.
        :param height_mm: page height, or None to follow the plot's own
            aspect so nothing is stretched.
        """
        from PySide6.QtCore import QMarginsF, QRectF
        from PySide6.QtGui import QPageLayout, QPageSize, QPainter, QPdfWriter

        source, aspect = cls._page_source(item)
        if source is None:
            return
        height = float(height_mm) if height_mm else width_mm * aspect

        writer = QPdfWriter(str(path))
        # THE DEVICE SCALE MUST MATCH THE SCENE, and 600 was the bug.
        #
        # Reported 2026-08-20: "the png saving seems to work but the pdf still
        # has ginormous text and on a tiny missaligned graph." Reproduced by
        # rasterising the PDF beside the PNG: the tick labels came out several
        # times the height of the plot and the axes sat off the page.
        #
        # WHY IT ONLY HITS THE PDF. `scene.render` maps the scene rect onto
        # the page geometrically -- but pyqtgraph draws its tick labels with
        # `ItemIgnoresTransformations`, which is what keeps them upright and
        # legible while a user zooms. Those items render at the DEVICE's own
        # scale, untouched by the mapping. At 600 dpi a 180 mm page is ~4250
        # device units wide while the scene is ~900, so everything geometric
        # was scaled 4.7x and the text was not -- it stayed device-sized and
        # so came out 4.7x too big relative to everything around it. The PNG
        # path never had this because ImageExporter renders at the item's own
        # pixel size, 1:1.
        #
        # Resolution chosen so one scene unit is one device unit. A PDF is
        # vector at any resolution -- this sets the coordinate scale, not the
        # fidelity -- so the text stays text and the lines stay lines.
        writer.setResolution(cls._pdf_resolution(source.width(), width_mm))
        size = QPageSize(QSizeF(width_mm, height), QPageSize.Millimeter)
        writer.setPageSize(size)
        writer.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Millimeter)

        painter = QPainter(writer)
        try:
            target = QRectF(0, 0, writer.width(), writer.height())
            cls._paint_scene(item, painter, target, source)
        finally:
            painter.end()

    #: Dots per inch a QSvgGenerator assumes when it converts its pixel size
    #: into the physical width it writes into the file.
    SVG_RESOLUTION = 72

    @classmethod
    def _export_svg(cls, item, path, width_mm: float = EXPORT_WIDTH_MM,
                    height_mm: Optional[float] = None) -> None:
        """Render a plot item directly into a vector SVG with Qt.

        Direct ``QSvgGenerator`` rendering keeps paths and text as vector
        elements and works across supported pyqtgraph/Qt combinations. Some
        combinations of pyqtgraph's ``SVGExporter`` fail while normalizing
        closed paths, so the stable Qt route is used for every installation.

        Parameters
        ----------
        item : pyqtgraph.GraphicsItem
            Plot item or graphics item to render.
        path : str or path-like
            Destination SVG path.
        width_mm : float, default=EXPORT_WIDTH_MM
            Physical page width in millimeters.
        height_mm : float or None, default=None
            Physical page height. ``None`` preserves the plot aspect ratio.
        """
        from PySide6.QtCore import QRectF, QSize
        from PySide6.QtGui import QPainter
        from PySide6.QtSvg import QSvgGenerator

        source, aspect = cls._page_source(item)
        if source is None:
            return
        height = float(height_mm) if height_mm else width_mm * aspect
        per_mm = cls.SVG_RESOLUTION / 25.4
        width_px = max(1, int(round(width_mm * per_mm)))
        height_px = max(1, int(round(height * per_mm)))

        generator = QSvgGenerator()
        generator.setFileName(str(path))
        generator.setResolution(cls.SVG_RESOLUTION)
        generator.setSize(QSize(width_px, height_px))
        generator.setViewBox(QRectF(0, 0, width_px, height_px))
        generator.setTitle(str(path))

        painter = QPainter(generator)
        try:
            cls._paint_scene(item, painter, QRectF(0, 0, width_px, height_px),
                             source)
        finally:
            painter.end()

    def _scatter_items(self):
        """Every scatter on the plot, for a restyle to reach.

        The selection marker is deliberately not one of them: it is a cursor,
        not data, and a restyle that shrank it to the point size would make
        the selection invisible.
        """
        return [i for i in self.plot.listDataItems()
                if hasattr(i, "setSize") and hasattr(i, "setBrush")
                and i is not self._highlight]

    def _ask_threshold_multiplier(self, current, callback) -> None:
        from PySide6.QtWidgets import QInputDialog

        value, ok = QInputDialog.getDouble(
            self, "Effect-size cut",
            "How many spreads wide is the cut?", float(current), 0.0, 20.0, 2)
        if ok:
            callback(value)

    def _ask_point_size(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        value, ok = QInputDialog.getDouble(
            self, "Point size", "Size in pixels:", 8.0, 1.0, 60.0, 1)
        if ok:
            for item in self._scatter_items():
                item.setSize(value)

    def _ask_point_colour(self) -> None:
        colour = pick_colour(self, PALETTE[0], "Point colour")
        if colour.isValid():
            # One brush for everything: this is the deliberate override of a
            # category colouring, and it is also the fastest path there is.
            brush = pg.mkBrush(colour)
            for item in self._scatter_items():
                item.setBrush(brush)

    def _ask_opacity(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        value, ok = QInputDialog.getDouble(
            self, "Opacity", "0 is invisible, 1 is solid:", 1.0, 0.05, 1.0, 2)
        if ok:
            for item in self._scatter_items():
                item.setOpacity(value)

    def _ask_labels(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        current_x = self.plot.getAxis("bottom").labelText
        current_y = self.plot.getAxis("left").labelText
        x, ok = QInputDialog.getText(self, "X axis label", "X:", text=current_x)
        if not ok:
            return
        y, ok = QInputDialog.getText(self, "Y axis label", "Y:", text=current_y)
        if not ok:
            return
        self.plot.setLabel("bottom", x)
        self.plot.setLabel("left", y)

    def _ask_font_size(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        value, ok = QInputDialog.getInt(
            self, "Font size", "Points:", self._font_size or 10, 5, 40)
        if ok:
            self.set_font_size(value)

    def _ask_font_colour(self) -> None:
        colour = pick_colour(self, self._font_colour or self._foreground,
                             "Font colour")
        if colour.isValid():
            self.set_font_colour(colour)

    def _ask_axis_limits(self) -> None:
        """Four numbers, each pre-filled with what is on screen now.

        FOUR DIALOGS RATHER THAN A FORM, deliberately, for the reason
        :meth:`build_style_menu` is separate from showing itself: the modal
        ones Qt ships are drivable from a test, and a hand-built form on this
        menu would be the one control here that no test can reach. Cancelling
        any of the four abandons the whole change, so a user who gets three
        numbers in and changes their mind is not left with a half-pinned
        axis.
        """
        from PySide6.QtWidgets import QInputDialog

        (x_from, x_to), (y_from, y_to) = self.axis_limits()
        # THE DIALOG SAYS WHICH UNITS IT WANTS. They are always the data's
        # own -- a logged axis is drawn in log10 and typed in the quantity --
        # and a prompt that does not say so is a number the user has to guess
        # the meaning of on the one axis where the two differ.
        units = {axis: (" in data units, not log10" if self._log[axis] else "")
                 for axis in ("x", "y")}
        asked = []
        for title, prompt, current in (
                ("X axis limits", f"X from{units['x']}:", x_from),
                ("X axis limits", f"X to{units['x']}:", x_to),
                ("Y axis limits", f"Y from{units['y']}:", y_from),
                ("Y axis limits", f"Y to{units['y']}:", y_to)):
            value, ok = QInputDialog.getDouble(
                self, title, prompt, float(current), -1e12, 1e12, 4)
            if not ok:
                return
            asked.append(value)
        self.set_axis_limits(x=(asked[0], asked[1]), y=(asked[2], asked[3]))

    def _ask_aspect_ratio(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        current = self.aspect_ratio()
        value, ok = QInputDialog.getDouble(
            self, "Lock axis scales",
            "X units per Y unit; 0 lets the data fill its box. This is the "
            "DATA's scale, not the figure's shape:",
            float(current or 0.0), 0.0, 1000.0, 3)
        if ok:
            self.set_aspect_ratio(None if value <= 0 else value)

    def _ask_line_width(self) -> None:
        """Prompt for a line width and apply the accepted value.

        Colour selection is a separate action, so this method never opens a
        platform colour dialog.
        """
        from PySide6.QtWidgets import QInputDialog

        lines = self.line_items()
        first = self._pen_of(lines[0]) if lines else None
        width, ok = QInputDialog.getDouble(
            self, "Line width", "Width in pixels:",
            float(first.widthF()) if first is not None else 1.5, 0.1, 20.0, 1)
        if ok:
            self.set_line_style(width=width)

    def _ask_line_colour(self) -> None:
        """The other half. Every line, the axes and the tick marks."""
        lines = self.line_items()
        first = self._pen_of(lines[0]) if lines else None
        start = (self._line_colour
                 or (first.color().name() if first is not None
                     else self._foreground))
        colour = pick_colour(self, start, "Line colour")
        if colour.isValid():
            self.set_line_colour(colour)

    def _ask_y_split(self) -> None:
        """The band of the y axis to leave out. Two numbers, in data units."""
        from PySide6.QtWidgets import QInputDialog

        (_, _), (y_from, y_to) = self.axis_limits()
        current = self.y_split() or (y_from, y_to)
        units = " in data units, not log10" if self._log["y"] else ""
        low, ok = QInputDialog.getDouble(
            self, "Split the y axis",
            f"Hide from{units} (the bottom of the empty stretch):",
            float(current[0]), -1e12, 1e12, 4)
        if not ok:
            return
        high, ok = QInputDialog.getDouble(
            self, "Split the y axis", f"Hide to{units}:",
            float(current[1]), -1e12, 1e12, 4)
        if ok:
            self.set_y_split(low, high)

    def _ask_colour_column(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        columns = self.numeric_columns()
        column, ok = QInputDialog.getItem(
            self, "Colour by a column", "Column:", columns, 0, False)
        if not ok or not column:
            return
        name, _ = QInputDialog.getItem(
            self, "Colour by a column", "Colour scale:", list(COLORMAPS), 0,
            False)
        self.colour_by_column(column, name or COLORMAPS[0])

    def _ask_shape_column(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        columns = self.shape_columns()
        column, ok = QInputDialog.getItem(
            self, "Shape by a column", "Column:", columns, 0, False)
        if ok and column:
            self.shape_by_column(column)

    def _ask_screen_size(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        width, ok = QInputDialog.getInt(
            self, "Size on screen",
            "Width in pixels (this moves the widget, not the saved page):",
            self.width(), 120, 8000)
        if not ok:
            return
        height, ok = QInputDialog.getInt(
            self, "Size on screen", "Height in pixels:", self.height(), 90,
            8000)
        if ok:
            self.set_screen_size(width, height)

    def _ask_export_size(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        width, height = self.export_size()
        new_width, ok = QInputDialog.getDouble(
            self, "Exported page size",
            "Page width in mm (this moves the saved page, not the screen):",
            float(width), 20.0, 1000.0, 1)
        if not ok:
            return
        new_height, ok = QInputDialog.getDouble(
            self, "Exported page size",
            "Page height in mm; 0 follows the plot's own shape:",
            float(height or 0.0), 0.0, 1000.0, 1)
        if ok:
            self.set_export_size(new_width,
                                 None if new_height <= 0 else new_height)


    #: Colour the shape and opacity keys are drawn in, so neither borrows a
    #: hue that means something else on the same plot.
    NEUTRAL_KEY = "#9E9E9E"

    def _legend_entries(self) -> list:
        """``[(label, ScatterPlotItem kwargs)]`` for every channel in force.

        THE LEGEND NAMES ITS CHANNEL, not only its level. Layering puts up to
        three columns on one point -- hue, then shape, then opacity -- and a
        key saying "fail" without saying which column that came from leaves
        the reader guessing which encoding they are reading.
        """
        out: list = []
        for name, colour in (getattr(self, "_legend_colours", None)
                             or {}).items():
            out.append((str(name), {"brush": pg.mkBrush(colour)}))
        shape = getattr(self, "_shape_legend", None)
        if shape:
            column, shapes = shape
            for name, symbol in shapes.items():
                out.append((f"{column} (shape): {name}",
                            {"brush": pg.mkBrush(self.NEUTRAL_KEY),
                             "symbol": symbol}))
        fade = getattr(self, "_opacity_legend", None)
        if fade:
            column, levels = fade
            alphas = self._opacity_alphas(len(levels))
            for name, alpha in zip(levels, alphas):
                colour = QColor(self.NEUTRAL_KEY)
                colour.setAlpha(int(alpha))
                out.append((f"{column} (opacity): {name}",
                            {"brush": pg.mkBrush(colour)}))
        return out

    def _build_legend(self) -> None:
        """Add the legend. Only ever called when it is actually wanted."""
        entries = self._legend_entries()
        if not entries:
            return
        self.plot.addLegend(offset=(-10, 10), labelTextSize="8pt")
        for label, style in entries:
            marker = pg.ScatterPlotItem([], [], pen=None, size=8, **style)
            self.plot.plotItem.legend.addItem(marker, label)

    def _toggle_legend(self, on: bool) -> None:
        if on:
            self._build_legend()
            return
        legend = getattr(self.plot.plotItem, "legend", None)
        if legend is not None:
            self.plot.plotItem.legend = None
            try:
                self.plot.plotItem.scene().removeItem(legend)
            except Exception:  # already detached
                pass

    def set_status(self, text: str) -> None:
        """What this plot has to say about ITSELF. Survives a selection."""
        self._headline = text
        self._status.setText(self._compose(text, self._level_note,
                                           self._style_note))

    def set_style_note(self, note: str) -> None:
        """What a RESTYLE has to say. Survives a redraw and a selection.

        A colour scale is unreadable without its range and a shape mapping is
        unreadable without its key, so those sentences are not decoration --
        they are the legend. They cannot live in the headline, which every
        redraw rewrites, nor in the click note, which every click rewrites;
        either would leave the reader looking at a picture whose key had been
        overwritten by something unrelated.
        """
        self._style_note = note
        self._status.setText(self._compose(self._headline, self._level_note,
                                           note, self._note))

    @staticmethod
    def _compose(*parts) -> str:
        """The status line: whichever of the three sentences exist."""
        return "   ".join(part for part in parts if part)

    def set_status_note(self, note: str) -> None:
        """Add a sentence about the CLICKED thing, keeping the headline.

        The diagnostics' status lines carry the numbers they exist for -- the
        inflation factor, the control medians, how many genes rest on one
        guide -- and overwriting those with the name of whatever was just
        clicked trades the panel's whole content for a string the user can
        already read in the table. Both fit.
        """
        self._note = note
        self._status.setText(self._compose(getattr(self, "_headline", ""),
                                           getattr(self, "_level_note", ""),
                                           getattr(self, "_style_note", ""),
                                           note))

    def _refresh_status(self) -> None:
        """Rewrite the status line from the four sentences it can hold."""
        status = getattr(self, "_status", None)
        if status is None:              # pragma: no cover - before the layout
            return
        status.setText(self._compose(getattr(self, "_headline", ""),
                                     getattr(self, "_level_note", ""),
                                     getattr(self, "_style_note", ""),
                                     getattr(self, "_note", "")))

    def note_selection(self, key, found: bool) -> None:
        """Say a row was picked -- unless this plot already said MORE about it.

        A click travels: the dot announces its key, the table selects that
        row, and every other plot then marks it. That last step arrives AFTER
        the clicked plot has written its own answer, and the clicked plot
        knows the most -- which control group a dot is in and what its effect
        was, how many of a gene's guides agree, what the p-value is. The plots
        that merely received the key know only the key, so letting them write
        last replaces the answer with the question.

        :param key: the shared row identifier to name in the status line. It
            is compared as text with the plot's existing detail so a richer
            click report for the same row is not overwritten.
        :param found: whether this plot actually drew that row. ``False`` is a
            real answer and is said out loud: a coefficient with an unusable
            p-value is on no plot, a nuisance term is off the volcano on
            purpose, and a guide is not a point on a per-gene plot at all.
        """
        if found and str(key) in getattr(self, "_note", ""):
            return
        self.set_status_note(
            f"{key}" if found
            else f"{key} is in the table but not on this plot.")

    def add_scatter(self, x, y, *, colours=None, brush_list=None,
                    size: float = 8.0, size_list=None,
                    labels: Sequence[str] = (),
                    symbol: str = "o", symbol_list=None, name: str = "",
                    rows=None) -> ScatterPlotItem:
        """Add points and wire up clicking them.

        :param x: x coordinates in frame order. An entry is omitted when it
            or its paired y coordinate is non-finite.
        :param y: y coordinates aligned one-for-one with ``x``.
        :param colours: one QColor per point, or None for a single colour.
        :param symbol_list: one pyqtgraph symbol per point, or None for
            ``symbol`` everywhere. This provides an independent categorical
            encoding without constructing a combined colour-by-shape legend.
        :param size_list: one diameter per point, or None for ``size``
            everywhere. A plain float array -- pyqtgraph takes it straight
            into its own arrays, so unlike a brush per point this costs
            nothing per point.
        :param labels: per-point text, shown on hover and on click.
        :param rows: the FRAME ROW each element of ``x``/``y`` came from.
            Default ``None`` means the arrays are already in frame order.

            THIS IS THE WHOLE OF THE Q-Q TRAP. A Q-Q plot is SORTED by
            p-value, so its nth drawn point is not its nth table row; a
            control panel is split into groups, so its nth point is not the
            nth row either. Left to assume otherwise, every one of those
            plots would carry an index that looks like a row, joins like a
            row, and names a different guide -- silently, and in the
            direction nobody questions, because something did light up.
        """
        x = _finite(x)
        y = _finite(y)
        keep = ~(np.isnan(x) | np.isnan(y))
        # Positions in the arrays as handed in: what indexes `colours` and
        # `brush_list`, which are drawn up alongside x and y.
        drawn = np.nonzero(keep)[0]
        # Indices into the ORIGINAL frame, so a click still identifies the
        # right row after unplottable points have been dropped -- and, when
        # `rows` says the arrays were reordered, after that too.
        original = drawn if rows is None else np.asarray(rows)[drawn]
        # ONE numpy->Python conversion, not a loop of them. `.tolist()` is a
        # C-level bulk convert; `[int(row) for row in original]` is 1,215
        # interpreter round trips and measurably re-slowed the volcano the
        # first time this was written that way.
        rows_drawn = original.tolist()

        brushes = None
        if brush_list is not None:
            # Already one reusable brush per point; nothing to build.
            brushes = [brush_list[i] for i in drawn]
        elif colours is not None:
            # ONE BRUSH PER DISTINCT COLOUR, REUSED -- not one per point.
            #
            # pg.mkBrush() per point builds 1,215 QBrush objects and defeats
            # pyqtgraph's fast path completely. Measured on the real volcano:
            #
            #     a brush constructed per point      39.5 ms
            #     27 brushes, indexed per point       3.5 ms
            #     a single brush for everything       1.6 ms
            #
            # The colours themselves were never the problem; allocating them
            # was. This is the whole of the lag on the last graph.
            colours = list(colours)
            cache: dict = {}
            brushes = []
            for i in drawn:
                colour = colours[i]
                key = colour.rgba() if hasattr(colour, "rgba") else str(colour)
                brush = cache.get(key)
                if brush is None:
                    brush = cache[key] = pg.mkBrush(colour)
                brushes.append(brush)

        # `data` must go in with the points: calling setData afterwards ADDS
        # points rather than annotating the ones already there.
        sizes = size
        if size_list is not None:
            sizes = np.asarray(size_list, dtype=float)[drawn]
        symbols = symbol
        if symbol_list is not None:
            # SUBSET BY `drawn`, exactly as the sizes and the brushes are.
            # A per-point array indexed in frame order against points that
            # have been filtered is the same misalignment `rows` exists to
            # prevent, and it would put the wrong shape on the right dot --
            # silently, because something is drawn.
            symbols = [symbol_list[int(i)] for i in drawn]
        item = pg.ScatterPlotItem(
            x=x[keep], y=y[keep], size=sizes, symbol=symbols,
            pen=pg.mkPen(None),
            brush=brushes if brushes is not None else pg.mkBrush(colour_for(0)),
            hoverable=len(drawn) <= HOVER_LIMIT, tip=self._point_tip,
            data=rows_drawn, name=name or None,
        )
        item.sigClicked.connect(self._on_points_clicked)
        self.plot.addItem(item)
        # Where each row ended up, so a selection arriving later can be drawn
        # without re-deriving the transform that put it there.
        self._row_xy.update(zip(rows_drawn,
                                zip(x[keep].tolist(), y[keep].tolist())))
        if labels is not None and len(labels):
            self._labels = labels
        return item

    # ------------------------------------------------------------ selection

    def set_keys(self, keys) -> int:
        """Give each frame row its identifier. Returns the number of rows.

        Duplicates are kept as the FIRST row carrying the key, and counted:
        an identifier that names two rows cannot select one of them, and
        picking silently is how the wrong point gets highlighted.

        ``None`` -- either as the whole argument or as one entry -- means
        "this row has no identifier". Such a row still draws and can still be
        clicked; it reports no identifier to other components, which is the
        truthful answer and is not the same as reporting the empty string,
        which would collide with every other unidentified row.
        """
        if keys is None:
            keys = ()
        # A MISSING KEY IS None, NOT THE STRING "nan". A frame column carries
        # its blanks as float NaN, and str() turns every one of them into the
        # same four characters -- which would make one bogus identifier that
        # several unrelated rows answer to, i.e. exactly the collision this
        # method's duplicate rule exists to prevent.
        self._keys = [None if key is None or key != key else str(key)
                      for key in keys]
        self._key_rows = {}
        for row, key in enumerate(self._keys):
            if key is not None:
                self._key_rows.setdefault(key, row)
        return len(self._keys)

    def _has_usable_keys(self) -> bool:
        """Whether ANY row on this plot can be identified to anyone else.

        NOT ``bool(self._keys)``. A caller can hand over a full-length column
        of blanks -- a fit with no gene-level terms gives the agreement plot
        exactly that, one ``None`` per gene -- and a list of ``None`` is
        truthy. Read that way, the plot appends "Click a point for its
        coefficient" to its status line while every click resolves to no key
        and selects nothing, which is an invitation it cannot honour.
        """
        return any(key is not None for key in self._keys)

    def key_for_row(self, row: int) -> Optional[str]:
        """The identifier at frame position ``row``, if this plot has keys."""
        if self._keys and 0 <= int(row) < len(self._keys):
            return self._keys[int(row)]
        return None

    def highlight_key(self, key) -> bool:
        """Ring the point identified by ``key``. Returns whether one was found.

        ``False`` is a real answer, not a failure: a key can be absent because
        its point was not plotted (an unusable p-value) or because it is a
        nuisance term this plot deliberately leaves off. Saying so beats
        ringing something near it.
        """
        key = None if key is None else str(key)
        self._selected_key = key
        # Single-select REPLACES. Leaving the multi list alone here is how the
        # count on screen and the list the consumers read drift apart.
        self._selected_keys = [] if key is None else [key]
        self._clear_extra_highlights()
        if self._highlight is not None:
            try:
                self.plot.removeItem(self._highlight)
            except Exception:               # already gone
                pass
            self._highlight = None
        if key is None:
            return False
        row = self._key_rows.get(key)
        if row is None:
            return False
        return self._draw_marker(row)

    def _draw_marker(self, row: int) -> bool:
        """Mark the row at frame position ``row``. False if it is not drawn.

        Split out so a plot whose marks are not points can say so its own way
        -- a histogram bar cannot be ringed like a dot -- while the key
        lookup, the clearing and the "was it found" answer stay in one place.
        """
        position = self._row_xy.get(row)
        if position is None:
            return False
        x, y = position
        # An open ring, not a filled dot: filling it would hide the point it
        # is meant to identify, including its category colour.
        self._highlight = pg.ScatterPlotItem(
            x=[x], y=[y], symbol="o", size=20, brush=pg.mkBrush(None),
            pen=pg.mkPen(QColor(self._foreground), width=2.0))
        self._highlight.setZValue(50)
        self.plot.addItem(self._highlight)
        return True

    def clear_highlight(self) -> None:
        self.highlight_key(None)

    # ------------------------------------------------------- multi-select

    def selected_keys(self) -> List[str]:
        """Return selected identifiers in pick order.

        Linked gene, image, and cell-table views read this shared selection.
        """
        return list(self._selected_keys)

    def highlight_keys(self, keys) -> int:
        """Ring every key in ``keys``. Returns how many were found and drawn.

        A KEY THAT IS NOT ON THE PLOT IS STILL SELECTED. It can be missing
        because its point was not plotted -- an unusable p-value, a nuisance
        term this plot leaves off -- and dropping it from the selection would
        make the count on screen disagree with what the consumers receive.
        So the ring is what is conditional here, never the membership.
        """
        wanted = []
        for key in keys or ():
            key = str(key)
            if key not in wanted:
                wanted.append(key)
        self._clear_extra_highlights()
        if self._highlight is not None:
            try:
                self.plot.removeItem(self._highlight)
            except Exception:               # already gone
                pass
            self._highlight = None
        self._selected_keys = wanted
        self._selected_key = wanted[-1] if wanted else None
        if not wanted:
            return 0
        drawn = 0
        # The LAST picked one keeps `_highlight`, because that is the one a
        # single-select consumer means by "the selection".
        for key in wanted[:-1]:
            row = self._key_rows.get(key)
            if row is not None and self._draw_extra_marker(row):
                drawn += 1
        row = self._key_rows.get(wanted[-1])
        if row is not None and self._draw_marker(row):
            drawn += 1
        return drawn

    def toggle_key(self, key) -> List[str]:
        """Add ``key`` to the selection, or remove it if it is already in.

        The modifier-click half of the platform gesture.
        """
        key = None if key is None else str(key)
        if key is None:
            return self.selected_keys()
        keys = list(self._selected_keys)
        if key in keys:
            keys.remove(key)
        else:
            keys.append(key)
        self.highlight_keys(keys)
        return self.selected_keys()

    def _draw_extra_marker(self, row: int) -> bool:
        """A ring for a selection member that is not the most recent one."""
        position = self._row_xy.get(row)
        if position is None:
            return False
        x, y = position
        ring = pg.ScatterPlotItem(
            x=[x], y=[y], symbol="o", size=20, brush=pg.mkBrush(None),
            pen=pg.mkPen(QColor(self._foreground), width=2.0))
        ring.setZValue(50)
        self.plot.addItem(ring)
        self._extra_highlights.append(ring)
        return True

    def select_in_rect(self, x0, y0, x1, y1, *, add: bool = False) -> List[str]:
        """Select every plotted point inside the data-coordinate rectangle.

        Points without a plotted position cannot be selected. Rectangle
        selection therefore contains exactly the points visibly enclosed by
        the supplied bounds.

        :param x0: x coordinate of the first rectangle corner; it need not be
            the lower bound.
        :param y0: y coordinate of the first rectangle corner; it need not be
            the lower bound.
        :param x1: x coordinate of the opposite rectangle corner.
        :param y1: y coordinate of the opposite rectangle corner.
        :param add: extend the current selection rather than replacing it,
            as used when a modifier key is held during selection.
        :returns: selected keys in pick order.
        """
        low_x, high_x = sorted((float(x0), float(x1)))
        low_y, high_y = sorted((float(y0), float(y1)))
        inside = []
        for row, position in self._row_xy.items():
            try:
                x, y = float(position[0]), float(position[1])
            except (TypeError, ValueError, IndexError):
                continue
            if low_x <= x <= high_x and low_y <= y <= high_y:
                key = self.key_for_row(int(row))
                if key is not None and key not in inside:
                    inside.append(key)
        keys = (self.selected_keys() if add else [])
        for key in inside:
            if key not in keys:
                keys.append(key)
        self.highlight_keys(keys)
        self._announce_selection(keys)
        return self.selected_keys()

    def _install_rubber_band(self) -> None:
        """Make a modified left-drag select instead of pan.

        WRAPPED, NOT REPLACED. pyqtgraph's own ``mouseDragEvent`` is what
        gives the plot its pan and its rectangle zoom, and a plot that lost
        those to gain a selection would be a worse plot. The wrapper takes
        the event only while the modifier is down and hands every other drag
        straight back.
        """
        box = self.plot.getViewBox()
        if box is None or getattr(box, "_spacr_band", False):
            return
        original = box.mouseDragEvent
        plot = self

        def drag(event, axis=None):
            modifiers = event.modifiers() if hasattr(event, "modifiers") \
                else Qt.NoModifier
            wanted = bool(modifiers & (Qt.ControlModifier | Qt.ShiftModifier))
            if not wanted or event.button() != Qt.LeftButton:
                return original(event, axis)
            event.accept()
            # The band is pyqtgraph's own scale box, so it looks exactly like
            # the rectangle zoom the user already knows.
            box.updateScaleBox(event.buttonDownPos(), event.pos())
            if not event.isFinish():
                return None
            box.rbScaleBox.hide()
            start = box.mapToView(event.buttonDownPos())
            end = box.mapToView(event.pos())
            plot.select_in_rect(start.x(), start.y(), end.x(), end.y(),
                                add=True)
            return None

        box.mouseDragEvent = drag
        box._spacr_band = True

    def _clear_extra_highlights(self) -> None:
        for ring in self._extra_highlights:
            try:
                self.plot.removeItem(ring)
            except Exception:               # already gone
                pass
        self._extra_highlights = []

    # ------------------------------------------------------- ranked bars

    #: Longest tick label a ranked-bar chart shows before it is elided.
    RANK_LABEL = 34

    def add_ranked_bars(self, labels, values, *, colour=None,
                        highlight: int = 0, thickness: float = 0.72,
                        descending: bool = True) -> int:
        """Draw a HORIZONTAL bar per label, longest at the top.

        The shape a feature-importance chart is: twenty names against one
        number each. It is horizontal because twenty names on a vertical
        axis are unreadable at any font size that fits them -- which is why
        matplotlib's ``barh`` was reached for, and why this plot could not
        replace it until now.

        NOT ``add_group_mark``. That method draws at categorical positions
        in x, one group at a time, and every one of its eight marks assumes
        the measurement is y. A named orientation flag through all of them
        would be eight branches doubled to serve one chart; this is the one
        chart, drawn directly.

        Parameters
        ----------
        labels : sequence of str
            One name per bar, in the caller's own order.
        values : array-like
            One number per label.
        colour : color-like or None, default=None
            Bar colour. The first categorical colour is used by default.
        highlight : int, default=0
            How many of the leading bars carry the accent. The house rule is
            that everything is grey except what the sentence is about, so 0
            means "no claim is being made about any particular one".
        thickness : float, default=0.72
            Bar thickness as a fraction of the row spacing.
        descending : bool, default=True
            Sort by value, largest first. False keeps the caller's order,
            which is what a chart of an already-ranked table wants.

        Returns
        -------
        int
            Number of bars drawn.
        """
        names = [str(name) for name in labels]
        heights = _finite(values)
        if len(names) != len(heights) or not len(names):
            return 0
        keep = ~np.isnan(heights)
        if not keep.any():
            return 0
        names = [name for name, ok in zip(names, keep) if ok]
        heights = heights[keep]

        order = (np.argsort(heights)[::-1] if descending
                 else np.arange(len(heights)))
        heights = heights[order]
        names = [names[int(i)] for i in order]

        # LARGEST AT THE TOP, so `rank` counts DOWN the screen. pyqtgraph's
        # y grows upward, so the first bar takes the highest number and the
        # axis is inverted rather than the data being reversed -- the same
        # picture, but the values stay in the order the caller can read off
        # the frame they handed over.
        rows = np.arange(len(heights), dtype=float)
        accent = QColor(colour) if colour is not None else colour_for(0)
        grey = QColor(self._foreground)
        grey.setAlpha(110)

        for index in range(len(heights)):
            ink = accent if index < int(highlight) else grey
            fill = QColor(ink)
            fill.setAlpha(150)
            self.plot.addItem(pg.BarGraphItem(
                y=[float(rows[index])], x0=[0.0],
                x1=[float(heights[index])], height=float(thickness),
                brush=fill, pen=pg.mkPen(ink)))

        axis = self.plot.getAxis("left")
        axis.setTicks([[(float(rows[i]), names[i][:self.RANK_LABEL])
                        for i in range(len(names))]])
        self.plot.getViewBox().invertY(True)
        self.plot.setYRange(-0.6, len(heights) - 0.4, padding=0.02)
        span = float(np.max(heights)) if len(heights) else 1.0
        self.plot.setXRange(min(0.0, float(np.min(heights))),
                            span * 1.06 if span > 0 else 1.0, padding=0.0)
        self._ranked = (names, heights)
        return int(len(heights))

    def ranked_frame(self):
        """The last ranked-bar chart as a frame, or ``None``.

        WHAT THE BUNDLE WRITES BESIDE THE PICTURE. A bar chart's data is its
        labels and its numbers, and without this the folder beside the
        figure would hold an empty ``data.csv``.
        """
        ranked = getattr(self, "_ranked", None)
        if ranked is None:
            return None
        import pandas as pd

        names, heights = ranked
        return pd.DataFrame({"name": list(names),
                             "value": [float(v) for v in heights]})

    # ------------------------------------------------------ curve + band

    def add_curve(self, x, y, *, colour=None, width: float = 2.0,
                  low=None, high=None, name: str = "") -> int:
        """A line through ordered points, optionally inside a spread band.

        The shape of a convergence or sweep chart: one series against an
        ordered x, with the spread it was summarised from drawn behind it.
        A line with no band claims a precision the data does not have, and
        it is the band that tells a reader where the curve stops meaning
        anything.

        Parameters
        ----------
        x, y : array-like
            Ordered point coordinates. Entries for which either coordinate is
            non-finite are omitted.
        colour : color-like or None, default=None
            Line colour. ``None`` uses the first categorical colour.
        width : float, default=2.0
            Line width in pixels.
        low, high : array-like or None, default=None
            Lower and upper band boundaries aligned with ``x`` and ``y``. The
            band is drawn only when both arrays are provided.
        name : str, default=""
            Legend label. An empty string adds no label.

        Returns
        -------
        int
            Number of points drawn.
        """
        xs = _finite(x)
        ys = _finite(y)
        keep = ~(np.isnan(xs) | np.isnan(ys))
        if not keep.any():
            return 0
        xs, ys = xs[keep], ys[keep]
        ink = QColor(colour) if colour is not None else colour_for(0)

        if low is not None and high is not None:
            bottom = _finite(low)[keep]
            top = _finite(high)[keep]
            shade = QColor(ink)
            shade.setAlpha(64)
            lower = pg.PlotDataItem(x=xs, y=bottom)
            upper = pg.PlotDataItem(x=xs, y=top)
            band = pg.FillBetweenItem(lower, upper, brush=pg.mkBrush(shade))
            band.setZValue(-10)
            self.plot.addItem(band)

        self.plot.plot(xs, ys, pen=pg.mkPen(ink, width=width),
                       name=name or None)
        self._curve = (xs, ys)
        return int(len(xs))

    def curve_frame(self):
        """The last curve as a frame, or ``None``. What the bundle writes."""
        curve = getattr(self, "_curve", None)
        if curve is None:
            return None
        import pandas as pd

        xs, ys = curve
        return pd.DataFrame({"x": [float(v) for v in xs],
                             "y": [float(v) for v in ys]})

    # --------------------------------------------------------- beeswarm

    #: Colour scale a beeswarm encodes the FEATURE VALUE on.
    BEESWARM_SCALE = "viridis"

    #: Vertical room one feature's points may use, as a fraction of a row.
    BEESWARM_SPREAD = 0.62

    def add_beeswarm(self, labels, contributions, feature_values=None, *,
                     rings: int = 0, size: float = 5.0,
                     colormap: str = "") -> int:
        """One row per feature, every observation's contribution as a point.

        The shape of a SHAP summary: for each of the top features, every
        sample's contribution as a dot on that feature's row, spread by how
        many share a value, and coloured by how large that sample's own
        feature value was. Reading it is the point of the chart -- a wide
        row matters more than a narrow one, and the colour split along it
        says which direction the feature pushes.

        DRAWN HERE RATHER THAN BY THE LIBRARY. ``shap.summary_plot`` draws
        into a matplotlib figure it makes itself, so it cannot be handed a
        pyqtgraph scene; it was the last thing on this path that kept the
        second renderer alive. It is not much of a chart to reproduce, and
        reproducing it is what makes the saved file and the tab the same
        picture.

        :param labels: one feature name per row, in the order they should
            appear.
        :param contributions: each sample's contribution for each feature,
            as ``(n_samples, n_features)``.
        :param feature_values: the same shape, holding the feature's own
            value per sample; it becomes the point colour. Without it every
            point is one colour and the direction of the effect is not shown.
        :param rings: unused; accepted so the three chart methods share a
            signature.
        :param size: point diameter.
        :param colormap: colour scale for ``feature_values``;
            :data:`BEESWARM_SCALE` when empty.
        :returns: the number of points drawn.
        """
        names = [str(name) for name in labels]
        matrix = np.asarray(contributions, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(names) or not len(names):
            return 0

        scale = str(colormap or self.BEESWARM_SCALE)
        lookup = None
        if feature_values is not None and scale in COLORMAPS:
            try:
                lookup = pg.colormap.get(scale).getLookupTable(
                    nPts=COLORMAP_STEPS, alpha=True)
            except Exception:                                # noqa: BLE001
                lookup = None
        colours_source = (np.asarray(feature_values, dtype=float)
                          if feature_values is not None else None)
        if colours_source is not None and colours_source.shape != matrix.shape:
            colours_source = None

        drawn = 0
        for column, name in enumerate(names):
            values = _finite(matrix[:, column])
            keep = ~np.isnan(values)
            if not keep.any():
                continue
            values = values[keep]
            offsets = self._beeswarm_offsets(values)
            row = float(column)
            if colours_source is not None and lookup is not None:
                shades = _finite(colours_source[:, column])[keep]
                inks = self._shade(shades, lookup)
            else:
                inks = [colour_for(0)] * len(values)
            self.add_scatter(values, row + offsets, size=size,
                             rows=None, colours=inks)
            drawn += int(len(values))

        self.add_line(x=0.0, colour=REFERENCE, width=1.0)
        axis = self.plot.getAxis("left")
        axis.setTicks([[(float(i), names[i][:self.RANK_LABEL])
                        for i in range(len(names))]])
        self.plot.getViewBox().invertY(True)
        self.plot.setYRange(-0.7, len(names) - 0.3, padding=0.02)
        self._beeswarm = (names, matrix)
        return drawn

    @staticmethod
    def _beeswarm_offsets(values) -> np.ndarray:
        """Vertical offsets that spread a row by local DENSITY.

        NOT PLAIN JITTER. Random offsets scatter a row evenly whatever its
        distribution, so a bimodal feature and a single tight cluster draw
        the same band of noise. Stacking outward from the centre within each
        narrow slice makes the row BULGE where observations pile up and go
        thin along the tails, which is the shape the reader is looking for.

        The row's full height is used either way -- the vertical extent
        carries no meaning and is not a second encoding of anything.
        """
        values = np.asarray(values, dtype=float)
        if len(values) < 2:
            return np.zeros(len(values))
        bins = max(8, min(64, int(np.sqrt(len(values)) * 3)))
        edges = np.linspace(float(np.min(values)), float(np.max(values)),
                            bins + 1)
        which = np.clip(np.digitize(values, edges) - 1, 0, bins - 1)
        counts = np.bincount(which, minlength=bins)
        busiest = max(1, int(counts.max()))
        seen: dict = {}
        offsets = np.zeros(len(values))
        for index, slot in enumerate(which):
            rank = seen.get(int(slot), 0)
            seen[int(slot)] = rank + 1
            here = max(1, int(counts[int(slot)]))
            # Centred: rank 0 in the middle, then alternating out.
            step = ((rank + 1) // 2) * (1 if rank % 2 else -1)
            offsets[index] = (step / max(1.0, here)) * \
                (here / busiest) * FastPlot.BEESWARM_SPREAD
        return offsets

    @staticmethod
    def _shade(values, lookup) -> list:
        """One QColor per value, through ``lookup``. NaN takes the grey."""
        values = np.asarray(values, dtype=float)
        usable = values[~np.isnan(values)]
        if not len(usable):
            return [QColor(MISSING_COLOUR)] * len(values)
        low, high = float(usable.min()), float(usable.max())
        span = (high - low) or 1.0
        out = []
        for value in values:
            if np.isnan(value):
                out.append(QColor(MISSING_COLOUR))
                continue
            step = int(np.clip(round((value - low) / span
                                     * (COLORMAP_STEPS - 1)),
                               0, COLORMAP_STEPS - 1))
            r, g, b, a = (int(c) for c in lookup[step])
            out.append(QColor(r, g, b, a))
        return out

    def beeswarm_frame(self):
        """The last beeswarm as a long frame, or ``None``."""
        swarm = getattr(self, "_beeswarm", None)
        if swarm is None:
            return None
        import pandas as pd

        names, matrix = swarm
        rows = []
        for column, name in enumerate(names):
            for value in matrix[:, column]:
                rows.append({"feature": name, "contribution": float(value)})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------- radar

    #: Grid rings drawn behind a radar polygon.
    RADAR_RINGS = 4

    def add_radar(self, labels, values, *, colour=None,
                  rings: int = 0, label_pad: float = 1.16) -> int:
        """Draw a closed radar polygon, one spoke per label.

        A RADAR IS A POLYGON, NOT AN AXIS. pyqtgraph has no polar view, and
        matplotlib's ``subplot_kw=dict(polar=True)`` was the only reason two
        of this module's figures could not move to the screen's renderer.
        The chart itself is elementary once that is said out loud: each
        label takes an angle, each value a radius, and the whole thing is a
        line through the resulting points with the axes hidden.

        THE GRID IS DRAWN, not inherited. A radar read against a square grid
        is unreadable -- the reference a reader needs is the concentric
        rings, which say what a radius is worth.

        Parameters
        ----------
        labels : sequence of str
            One name per spoke, clockwise from the top.
        values : array-like
            One radius per label. Negative values are clipped to zero: a
            radar has no inside-out.
        colour : color-like or None, default=None
            Polygon colour. The first categorical colour is used by default.
        rings : int, default=0
            Grid rings; :data:`RADAR_RINGS` when 0.
        label_pad : float, default=1.16
            Where the names sit, as a multiple of the outer ring.

        Returns
        -------
        int
            Number of spokes drawn.
        """
        import math

        names = [str(name) for name in labels]
        radii = _finite(values)
        if len(names) != len(radii) or len(names) < 3:
            return 0
        keep = ~np.isnan(radii)
        if not keep.all():
            names = [n for n, ok in zip(names, keep) if ok]
            radii = radii[keep]
        if len(names) < 3:
            return 0
        radii = np.clip(radii, 0.0, None)
        outer = float(np.max(radii)) or 1.0

        # CLOCKWISE FROM THE TOP, which is how every radar chart anybody has
        # seen is laid out; counter-clockwise from the right is the maths
        # convention and reads as a different chart.
        angles = np.array([math.pi / 2.0 - 2.0 * math.pi * i / len(names)
                           for i in range(len(names))])

        grid = QColor(self._foreground)
        grid.setAlpha(48)
        pen = pg.mkPen(grid, width=1.0)
        steps = int(rings or self.RADAR_RINGS)
        circle = np.linspace(0.0, 2.0 * np.pi, 90)
        for step in range(1, steps + 1):
            ring = outer * step / steps
            self.plot.plot(ring * np.cos(circle), ring * np.sin(circle),
                           pen=pen)
        for angle in angles:
            self.plot.plot([0.0, outer * math.cos(angle)],
                           [0.0, outer * math.sin(angle)], pen=pen)

        ink = QColor(colour) if colour is not None else colour_for(0)
        closed_x = np.append(radii * np.cos(angles),
                             radii[0] * math.cos(angles[0]))
        closed_y = np.append(radii * np.sin(angles),
                             radii[0] * math.sin(angles[0]))
        fill = QColor(ink)
        fill.setAlpha(64)
        self.plot.addItem(pg.PlotCurveItem(
            x=closed_x, y=closed_y, pen=pg.mkPen(ink, width=2.0),
            fillLevel=None, brush=None))
        # THE FILL IS ITS OWN ITEM. `fillLevel` fills to a horizontal line,
        # which on a closed polygon shades a half-moon rather than the
        # inside -- so the interior is a second, brushed curve.
        area = pg.PlotDataItem(x=closed_x, y=closed_y,
                               pen=pg.mkPen(None), fillLevel=0,
                               brush=fill, connect="all")
        area.setZValue(-5)
        self.plot.addItem(area)

        ink_text = QColor(self._font_colour or self._foreground)
        for name, angle in zip(names, angles):
            text = pg.TextItem(name, color=ink_text, anchor=(0.5, 0.5))
            text.setPos(outer * label_pad * math.cos(angle),
                        outer * label_pad * math.sin(angle))
            self.plot.addItem(text)

        # A RADAR HAS NO AXES: the numbers live on the rings, and a pair of
        # cartesian scales beside a polar chart is two coordinate systems on
        # one picture.
        for side in ("left", "bottom"):
            self.plot.getAxis(side).setTicks([[]])
        self.plot.showGrid(x=False, y=False)
        self.plot.getViewBox().setAspectLocked(True)
        reach = outer * (label_pad + 0.16)
        self.plot.setXRange(-reach, reach, padding=0.0)
        self.plot.setYRange(-reach, reach, padding=0.0)
        self._radar = (names, radii)
        return int(len(names))

    def radar_frame(self):
        """The last radar as a frame, or ``None``. What the bundle writes."""
        radar = getattr(self, "_radar", None)
        if radar is None:
            return None
        import pandas as pd

        names, radii = radar
        return pd.DataFrame({"name": list(names),
                             "value": [float(v) for v in radii]})

    def add_line(self, *, x=None, y=None, colour: str = "#C44E52",
                 style=Qt.DashLine, width: float = 1.5, label: str = ""):
        """A threshold line. ``x`` for vertical, ``y`` for horizontal."""
        if self._line_colour is not None:
            # A LINE ADDED AFTER THE CONTROL WAS USED STILL OBEYS IT. A
            # redraw puts new threshold lines on the plot, and without this
            # they would arrive in the default red beside the ones the user
            # recoloured.
            colour = self._line_colour
        pen = pg.mkPen(QColor(colour), width=width, style=style)
        # THE CAPTION FOLLOWS THE FONT, NOT THE LINE (instruction 152 B).
        ink = self._font_colour or self._foreground
        line = pg.InfiniteLine(
            pos=(x if x is not None else y),
            angle=90 if x is not None else 0,
            pen=pen, label=label or None,
            labelOpts={"position": 0.92, "color": ink, "movable": False},
        )
        self.plot.addItem(line)
        return line

    # ------------------------------------------------------------ group marks

    def add_group_mark(self, position: float, values, kind: str = "points", *,
                       colour=None, rows=None, width: float = 0.6,
                       size: float = 7.0, seed: int = 0,
                       centre: str = "mean", spread: str = "sem") -> int:
        """Draw one group's observations or summary at ``position``.

        Parameters
        ----------
        position : float
            Group position on the categorical axis.
        values : array-like
            Observations in the group.
        kind : str, default="points"
            Mark key from :data:`MARK_TYPES`.
        colour : color-like or None, default=None
            Mark color. The first categorical color is used by default.
        rows : array-like or None, default=None
            Source-frame row for each observation. Individual points remain
            clickable when these identifiers are supplied.
        width : float, default=0.6
            Mark width in x-axis units.
        size : float, default=7.0
            Point-marker size.
        seed : int, default=0
            Random seed for reproducible horizontal jitter.
        centre : {"mean", "median"}, default="mean"
            Summary used by point, jitter, and line marks.
        spread : {"sd", "sem", "var", "none"}, default="sem"
            What the bar's whisker MEANS, from
            :data:`spacr.figures.spread.SPREAD_CHOICES`. The three are not
            interchangeable -- SD describes the observations, SEM the
            confidence in their mean, and at n=3000 they differ by a factor
            of fifty-five -- so a caller that draws one has to say which, and
            ``none`` draws no whisker at all.

        Returns
        -------
        int
            Number of finite observations represented by the mark.
        """
        v = _finite(values)
        keep = ~np.isnan(v)
        if not keep.any():
            return 0
        finite = np.nonzero(keep)[0]
        picked = finite if rows is None else np.asarray(rows)[finite]
        v = v[keep]
        ink = QColor(colour) if colour is not None else colour_for(0, 200)
        half = float(width) / 2.0

        if kind in ("points", "jitter"):
            if kind == "jitter":
                rng = np.random.default_rng(seed)
                x = position + (rng.random(len(v)) - 0.5) * width
            else:
                x = np.full(len(v), float(position))
            self.add_scatter(x, v, size=size, rows=picked,
                             colours=[ink] * len(v))
            # THE SUMMARY LINE IS THE POINT OF "points". Bare points with no
            # summary answer nothing; the rule this menu follows is
            # "individual points WITH a mean line", and the line is the half
            # that carries the comparison between the groups.
            level = float(np.median(v) if centre == "median" else np.mean(v))
            self.plot.plot([position - half, position + half], [level, level],
                           pen=pg.mkPen(QColor(self._foreground), width=2))
            return int(len(v))

        if kind in ("jitter_box", "jitter_bar"):
            # THE SUMMARY FIRST, THE POINTS ON TOP. Drawn in that order so the
            # observations are not hidden behind the shape that summarises
            # them -- which is the whole reason a composite is a different
            # request from either half.
            #
            # AND THE POINTS KEEP THEIR ROWS, so a composite stays CLICKABLE
            # where the bare box and bar do not. That is what makes this the
            # honest default: the reader sees the distribution and can still
            # name any observation in it.
            self.add_group_mark(position, values,
                                "box" if kind == "jitter_box" else "bar",
                                colour=colour, rows=None, width=width,
                                size=size, seed=seed, centre=centre,
                                spread=spread)
            return self.add_group_mark(position, values, "jitter",
                                       colour=colour, rows=rows, width=width,
                                       size=size, seed=seed, centre=centre)

        if kind == "line":
            # ONE POINT PER GROUP, and the JOINING is the caller's: this
            # method draws one group at a time and cannot see its neighbours.
            # The marker is what a line chart is made of, and `GroupedPlot`
            # connects them once every group has been drawn.
            level = float(np.median(v) if centre == "median" else np.mean(v))
            self.add_scatter([float(position)], [level], size=max(6, size),
                             rows=None, colours=[ink])
            if len(v) > 1:
                if centre == "median":
                    low, high = (float(np.percentile(v, q)) for q in (25, 75))
                else:
                    err = float(np.std(v, ddof=1)) / np.sqrt(len(v))
                    low, high = level - err, level + err
                self.plot.plot([position, position], [low, high],
                               pen=pg.mkPen(QColor(self._foreground), width=1))
            return int(len(v))

        if kind == "bar":
            mean = float(np.mean(v))
            fill = QColor(ink)
            fill.setAlpha(150)
            self.plot.addItem(pg.BarGraphItem(
                x=[position], height=[mean], width=width, brush=fill,
                pen=pg.mkPen(ink)))
            # THE SPREAD, ON THE BAR, AND THE USER SAYS WHICH ONE. A bar
            # already hides every observation; one with no interval at all
            # hides that there was any spread to hide, which is the version
            # of this chart that gets published and then argued about.
            #
            # THROUGH `spacr.figures.spread`, which is the one vocabulary --
            # a second definition of SEM here would let two screens draw
            # whiskers sqrt(n) apart and label them identically.
            from ...figures.spread import SPREAD_NONE, spread_of

            if len(v) > 1 and str(spread or SPREAD_NONE) != SPREAD_NONE:
                err = spread_of(v, str(spread))
                if np.isfinite(err):
                    self.plot.plot([position, position],
                                   [mean - err, mean + err],
                                   pen=pg.mkPen(QColor(self._foreground),
                                                width=2))
            return int(len(v))

        if kind == "box":
            low, q1, median, q3, high = (float(np.percentile(v, p))
                                         for p in (0, 25, 50, 75, 100))
            span = q3 - q1
            top = float(np.max(v[v <= q3 + 1.5 * span])) if span else high
            bottom = float(np.min(v[v >= q1 - 1.5 * span])) if span else low
            pen = pg.mkPen(QColor(self._foreground), width=1.5)
            fill = QColor(ink)
            fill.setAlpha(110)
            self.plot.addItem(pg.BarGraphItem(
                x=[position], y0=[q1], y1=[q3], width=width, brush=fill,
                pen=pg.mkPen(ink)))
            self.plot.plot([position - half, position + half],
                           [median, median], pen=pen)
            self.plot.plot([position, position], [q3, top], pen=pen)
            self.plot.plot([position, position], [bottom, q1], pen=pen)
            # OUTLIERS STAY POINTS, and stay clickable. They are the rows a
            # reader of a box plot actually wants to name, and they are
            # individual observations, so the rule above lets them keep their
            # rows.
            beyond = (v > top) | (v < bottom)
            if beyond.any():
                self.add_scatter(np.full(int(beyond.sum()), float(position)),
                                 v[beyond], size=size, rows=picked[beyond],
                                 colours=[ink] * int(beyond.sum()))
            return int(len(v))

        if kind == "violin":
            centres, density = _violin_profile(v, half)
            if centres is None:
                # Every value identical: a density has no width and the
                # outline would be a vertical line pretending to be a shape.
                # Fall back to the honest mark rather than drawing that.
                return self.add_group_mark(position, values, "points",
                                           colour=colour, rows=rows,
                                           width=width, size=size, seed=seed,
                                           centre=centre)
            fill = QColor(ink)
            fill.setAlpha(110)
            xs = np.concatenate([position + density, (position - density)[::-1]])
            ys = np.concatenate([centres, centres[::-1]])
            self.plot.addItem(pg.PlotCurveItem(
                x=xs, y=ys, pen=pg.mkPen(ink, width=1.5), brush=fill,
                fillLevel=None, connect="all"))
            median = float(np.median(v))
            self.plot.plot([position - half * 0.5, position + half * 0.5],
                           [median, median],
                           pen=pg.mkPen(QColor(self._foreground), width=2))
            return int(len(v))

        raise ValueError(
            f"unknown mark {kind!r}; known marks: "
            f"{', '.join(name for name, _ in MARK_TYPES)}")

    def _on_points_clicked(self, _item, points) -> None:
        if not len(points):
            return
        index = points[0].data()
        if index is None:
            return
        index = int(index)
        text = "   ".join(part for part in (self._describe(index),
                                            self._detail(index)) if part)
        if text:
            self.set_status_note(text)
        key = self.key_for_row(index)
        if key is not None:
            # THE PLATFORM GESTURE, NOT A BESPOKE MODE (instruction 206).
            # Ctrl or Shift adds and removes; a plain click replaces. Read
            # from the application rather than the event because pyqtgraph's
            # click carries a scene event whose modifiers are not always
            # populated on every platform.
            if self._adding_to_selection():
                keys = self.toggle_key(key)
            else:
                self.highlight_key(key)
                keys = [key]
            self._announce_selection(keys)
        self.point_clicked.emit(index)

    @staticmethod
    def _adding_to_selection() -> bool:
        """Whether the modifier for add-and-remove is down right now."""
        from PySide6.QtWidgets import QApplication

        modifiers = QApplication.keyboardModifiers()
        return bool(modifiers & (Qt.ControlModifier | Qt.ShiftModifier))

    def _announce_selection(self, keys) -> None:
        """Tell the consumers, and say how many are selected.

        A SELECTION YOU CANNOT COUNT IS ONE YOU CANNOT TRUST, so the count
        goes on the plot whenever it is more than one. `key_selected` still
        carries the most recent member: it is what the single-select
        consumers mean, and re-pointing them at a list would break every one
        of them at once.
        """
        keys = [str(k) for k in (keys or ())]
        if len(keys) > 1:
            self.set_status_note(
                f"{len(keys)} selected: {', '.join(keys[:4])}"
                + (f" and {len(keys) - 4} more" if len(keys) > 4 else ""))
        if keys:
            self.key_selected.emit(keys[-1])
        self.keys_selected.emit(list(keys))

    def _describe(self, index: int) -> str:
        """Describe ONE point, on demand.

        Formatting every point up front is what made the plot slow to appear;
        formatting the clicked one costs nothing and reads the same.
        """
        if self._labels is not None and index < len(self._labels or ()):
            return str(self._labels[index])
        frame = getattr(self, "_frame", None)
        if frame is not None and index < len(frame):
            parts = []
            for column in (getattr(self, "_label_column", None),
                           getattr(self, "_effect_column", None),
                           getattr(self, "_p_column", None)):
                if column and column in frame.columns:
                    value = frame[column].iloc[index]
                    parts.append(f"{column}={value}"
                                 if not isinstance(value, str) else str(value))
            if parts:
                return "   ".join(parts)
        # THE IDENTIFIER IS ALREADY THE ANSWER. A diagnostic plot holds no
        # frame -- it is handed an array of p-values -- so without this a
        # click on the Q-Q reported an empty status line while quietly
        # selecting the right row somewhere else. The key IS the guide's
        # name; saying it costs one lookup and is what the user clicked for.
        key = self.key_for_row(index)
        return key or ""

    def _detail(self, index: int) -> str:
        """Whatever THIS plot knows about the row that the key does not.

        A hook, not a table read: the point of formatting on click is that no
        per-point work happens before one. Subclasses that already hold the
        plotted arrays answer from them in O(1).
        """
        return ""

    # ---------------------------------------------------------------- export

    def export(self, path: Optional[str] = None) -> Optional[str]:
        """Write the plot out: PDF, SVG or PNG, by the name given.

        PDF and SVG use Qt's vector painters, producing vector output rather
        than embedding a bitmap. PNG uses pyqtgraph's image exporter.

        The page is :meth:`export_size`, which the right-click menu sets. It
        is independent of the widget's on-screen size.
        """
        # `clicked` AND `triggered` BOTH CARRY A BOOL, and this method takes
        # an optional first argument, so Qt hands the checked state straight
        # into `path`. `False is None` is False, the dialog never opened, and
        # `False` travelled all the way to QImage.save -- which is where the
        # user saw it:
        #
        #     TypeError: 'QImage.save' called with wrong argument types:
        #       QImage.save(bool)
        #
        # The connections below now pass no argument, but this stays: `export`
        # is public, and the next person to wire a button to it should not
        # have to know that Qt's signal has an argument it does not want.
        if isinstance(path, bool):
            path = None
        if path is None:
            from PySide6.QtWidgets import QFileDialog
            path, _ = QFileDialog.getSaveFileName(
                self, "Export plot", "plot.pdf",
                "PDF (*.pdf);;Vector (*.svg);;Image (*.png)")
            if not path:
                return None
        self._sync_auto_range()
        from pyqtgraph import exporters

        item = self.plot.plotItem
        width_mm, height_mm = self.export_size()
        # INSTRUCTION 150. Around the WHOLE export and not around
        # `_paint_scene`, which was the first attempt and reached only two of
        # the three formats: PDF and SVG paint the scene themselves, and PNG
        # goes through pyqtgraph's ImageExporter, which does not. A rule that
        # covers two formats out of three is worse than none, because the one
        # it misses is the default.
        #
        # AND NOT AROUND `snapshot()`, deliberately. That render is the tile
        # in the gallery, which is the SCREEN version -- 139 C: the tile and
        # the file differ on purpose and the difference is the point.
        restore = self._wear_the_print_look(item)
        try:
            # THE SCENE TAKES THE SHAPE FIRST. `export_size` above has
            # already put the shape on the PAGE; without this the page and
            # the scene disagree and Qt letterboxes the difference.
            with self._held_at_the_page_shape():
                self._write_export(item, path, width_mm, height_mm, exporters)
        finally:
            for undo in restore:
                undo()
        return path

    def _write_export(self, item, path, width_mm, height_mm, exporters) -> None:
        """Put the plot on disk in the format the name asks for."""
        if str(path).lower().endswith(".pdf"):
            self._export_pdf(item, path, width_mm, height_mm)
        elif str(path).lower().endswith(".svg"):
            self._export_svg(item, path, width_mm, height_mm)
        else:
            exporter = exporters.ImageExporter(item)
            # THE PAGE THE SAVE ASKS FOR (150 B). Transparent was the old
            # answer and it is only right for the `transparent` mode: dark ink
            # on no background is still unreadable on a dark slide, and a
            # figure going into a manuscript is going onto white. `print` --
            # the default -- writes an explicit light page, `screen` keeps
            # what is on screen, and `transparent` keeps the old behaviour for
            # anyone compositing onto their own colour.
            try:
                exporter.parameters()["background"] = self._export_ground()
            except (KeyError, TypeError):   # older pyqtgraph
                pass
            self._shape_the_image(exporter, width_mm, height_mm)
            exporter.export(path)

    @contextmanager
    def _dressed_for_the_file(self, ink: str = "", background: str = "",
                              grid: Optional[bool] = None, *,
                              font_size: Optional[int] = None,
                              line_width: Optional[float] = None,
                              aspect: Optional[float] = None,
                              x_title: Optional[str] = None,
                              y_title: Optional[str] = None,
                              text_colour: str = "",
                              line_colour: str = "",
                              canvas_shape: str = "",
                              dpi: Optional[int] = None):
        """Apply export styling for one synchronous render, then restore it.

        Pyqtgraph scenes cannot be copied safely, so the live scene is styled
        only while an offscreen exporter or painter renders it. The original
        colors, grid, text size, line width, aspect, axis titles, canvas
        shape and export resolution are restored in ``finally``, including
        when the export raises.

        EVERY KNOB THE SAVE DIALOG OFFERS COMES THROUGH HERE, so the preview
        and the file are styled by one path. A second styling path is how a
        preview comes to show something the file does not.

        ``ink`` colours text AND lines together, which is what a paper/slide
        preset means. ``text_colour`` and ``line_colour`` are the halves,
        applied after it so a caller that names one of them wins over the
        preset for that half.

        ``canvas_shape`` names one of :data:`CANVAS_SHAPES` and reaches the
        page through :meth:`export_size` and the scene through
        :meth:`_held_at_the_page_shape`. An unknown name is ignored rather
        than raised: this is a render, and a render that refuses to draw is
        the failure the caller is trying to avoid.
        """
        before_bg, before_fg = self._background, self._foreground
        before_grid = self._grid_on
        before_font = self._font_size
        before_labels = dict(getattr(self, "_base_labels", {}) or {})
        before_text_colour = self._font_colour
        before_line_colour = self._line_colour
        before_shape = self._canvas_shape
        before_dpi = self._export_dpi
        before_width = None
        if line_width is not None:
            lines = self.line_items()
            first = self._pen_of(lines[0]) if lines else None
            before_width = float(first.widthF()) if first is not None else None
        before_aspect = None
        if aspect is not None:
            box = self.plot.getViewBox()
            state = box.state.get("aspectLocked", False)
            before_aspect = float(state) if state else None
        # THE PAGE, not just the scene. Restyling the scene colours what is
        # DRAWN; the raster exporter fills the page behind it separately, and
        # it reads `_export_ground`. Recording the choice here is what lets
        # that method prefer it over the global look. Empty means the dialog
        # said "transparent", which is not a colour and must not become one.
        before_ground = getattr(self, "_chosen_ground", "")
        try:
            if canvas_shape and canvas_shape in dict(CANVAS_SHAPES):
                # THE STATE, NOT `set_canvas_shape`. That one re-lays the
                # widget out on screen, and this styling is for the file:
                # the scene is given the proportion by
                # `_held_at_the_page_shape` around the render itself.
                self._canvas_shape = str(canvas_shape)
            if dpi:
                self._export_dpi = int(dpi)
            if ink or background:
                self.restyle(background=background or before_bg,
                             foreground=ink or before_fg)
            if text_colour:
                self.set_font_colour(text_colour)
            if line_colour:
                self.set_line_colour(line_colour)
            self._chosen_ground = str(background or "")
            if grid is not None:
                self.set_grid(grid)
            if font_size:
                self.set_font_size(int(font_size))
            if line_width:
                self.set_line_style(width=float(line_width))
            if aspect is not None:
                self.set_aspect_ratio(aspect or None)
            for edge, title in (("bottom", x_title), ("left", y_title)):
                if title is not None:
                    self.plot.setLabel(edge, str(title))
            if x_title is not None or y_title is not None:
                self.apply_text_style()
            yield self
        finally:
            self._chosen_ground = before_ground
            self._export_dpi = before_dpi
            self._canvas_shape = before_shape
            if line_colour:
                self.set_line_colour(before_line_colour)
            if text_colour:
                self.set_font_colour(before_text_colour)
            for edge, title in (("bottom", x_title), ("left", y_title)):
                if title is not None:
                    self.plot.setLabel(edge, before_labels.get(edge, ""))
            if aspect is not None:
                self.set_aspect_ratio(before_aspect)
            if line_width and before_width:
                self.set_line_style(width=before_width)
            if font_size:
                # BACK TO WHAT IT WAS, INCLUDING None. `set_font_size` takes
                # an int, so "no size of its own" -- the default state of
                # every plot nobody has resized -- can only be restored by
                # putting the attribute back and re-applying.
                self._font_size = before_font
                self.apply_text_style()
            if grid is not None:
                self.set_grid(before_grid)
            if ink or background:
                self.restyle(background=before_bg, foreground=before_fg)

    def styled_snapshot(self, width: int = SNAPSHOT_PX[0], *, ink: str = "",
                        background: str = "", grid: Optional[bool] = None,
                        **styling):
        """Render a preview with the styling used for file export.

        Parameters
        ----------
        width : int, default=SNAPSHOT_PX[0]
            Preview width in pixels. Height follows the plot's aspect ratio.
        ink : str, optional
            Temporary foreground color. An empty string keeps the current
            foreground.
        background : str, optional
            Temporary background color. An empty string keeps the current
            background.
        grid : bool or None, optional
            Temporary grid visibility. ``None`` keeps the current setting.

        Returns
        -------
        QPixmap or None
            Styled plot image, or ``None`` when the plot has no content.

        Notes
        -----
        The same temporary styling context is used by :meth:`export_styled`.
        The on-screen plot is restored after rendering, including when export
        raises an exception.
        """
        with self._dressed_for_the_file(ink, background, grid, **styling):
            # THE PAGE THE FILE WOULD GET. `_export_ground` is what the
            # raster exporter fills behind the scene when the file is
            # written, so a preview that left it out showed a transparent
            # page for a file that will not have one.
            return self.snapshot(width, ground=self._export_ground())

    def export_bundle(self, folder: Optional[str] = None,
                      name: str = "") -> Optional[str]:
        """Export the graph with its data, statistics, and settings.

        Parameters
        ----------
        folder : str, optional
            Parent directory for the bundle. A directory chooser opens when
            omitted.
        name : str, optional
            Bundle and graph name. The plot title is used when omitted.

        Returns
        -------
        str or None
            Created bundle directory, or ``None`` when directory selection is
            cancelled.

        Notes
        -----
        PDF and PNG files are exported from the same plot state. The bundle
        also records the displayed data, statistical comparison, and relevant
        plot settings.
        """
        if isinstance(folder, bool):        # a signal's checked state
            folder = None
        if folder is None:
            from PySide6.QtWidgets import QFileDialog

            folder = QFileDialog.getExistingDirectory(
                self, "Save the graph, its data and its statistics")
            if not folder:
                return None
        from ...figures.bundle import save

        title = name or self.plot.plotItem.titleLabel.text or "graph"
        return save(folder, str(title), render=self.export,
                    data=self.frame(), groups=self.comparison_groups(),
                    unit=self.comparison_unit(),
                    settings=self.export_settings())

    def graph_spec(self):
        """Return the data specification used to draw this plot, if present.

        Plots that retain a specification can be redrawn using another
        compatible graph type. Rendering-only plots return ``None``.
        """
        return getattr(self, "spec", None)

    def _offer_graph_kinds(self, menu) -> None:
        """Add a ``Show as`` submenu for graph types compatible with the data.

        Incompatible types remain visible but disabled, with a tooltip that
        explains the data requirement they do not meet.
        """
        spec = self.graph_spec()
        if spec is None:
            return
        try:
            from ...graph_types import GRAPH_NAMES, GRAPH_TYPES, offer
        except Exception:                                    # noqa: BLE001
            return
        frame = getattr(spec, "frame", None)
        if frame is None or not len(frame):
            return
        rows = offer(frame, getattr(spec, "group", ""),
                     getattr(spec, "value", ""))
        if not rows:
            return
        from PySide6.QtWidgets import QMenu

        show_as = QMenu("Show as", menu)
        show_as.setToolTipsVisible(True)
        menu.addMenu(show_as)
        described = dict(GRAPH_TYPES)
        current = str(getattr(spec, "kind", "") or "")
        for kind, caption, reason in rows:
            # THE NAME ON THE ENTRY, THE DESCRIPTION IN THE TOOLTIP. A menu
            # reading "one value per group" instead of "Bar" cannot be
            # scanned, which is what a menu is for.
            action = show_as.addAction(str(GRAPH_NAMES.get(kind, kind)))
            action.setToolTip(reason or str(described.get(kind, caption)))
            action.setCheckable(True)
            action.setChecked(kind == current)
            if reason:
                # GREYED WITH THE REASON, never absent: a list that silently
                # shortens leaves the user wondering whether they
                # misremembered (instruction 106).
                action.setEnabled(False)
            else:
                action.triggered.connect(
                    lambda _checked=False, k=kind: self._show_as_kind(k))

        # AND A WAY TO MAKE IT THE STARTING POINT. Asked for 2026-08-28: the
        # right-click menu was right, but it only ever changed the graph in
        # front of you -- the next one of the same shape was drawn the old
        # way again. This is the same choice, remembered.
        #
        # HERE, where the user is already choosing a graph type, rather than
        # only in Preferences: the moment somebody decides they prefer a
        # violin is the moment they are looking at one.
        if current:
            show_as.addSeparator()
            from ...graph_types import shape_of

            try:
                shape = shape_of(frame, getattr(spec, "group", ""),
                                 getattr(spec, "value", ""))
            except Exception:                                # noqa: BLE001
                shape = ""
            if shape:
                remember = show_as.addAction(
                    f"Always start with {GRAPH_NAMES.get(current, current)}")
                remember.setToolTip(
                    "Draw this kind first for every graph of this shape. "
                    "Right-click still changes any individual graph.")
                remember.triggered.connect(
                    lambda _checked=False, k=current, sh=shape:
                    self._remember_default_kind(sh, k))

    @staticmethod
    def _remember_default_kind(shape: str, kind: str) -> None:
        """Persist ``kind`` as what ``shape`` is drawn as first."""
        try:
            from ..preferences import set_default_graph_type

            set_default_graph_type(shape, kind)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not remember %r for %r", kind, shape,
                      exc_info=True)

    def _show_as_kind(self, kind: str) -> None:
        try:
            self.show_as(kind)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not redraw as %r", kind, exc_info=True)

    def comparison_groups(self) -> Optional[dict]:
        """Return labeled values for export-time statistical comparison.

        Returns
        -------
        dict or None
            Mapping of group labels to values. The base plot has no defined
            comparison; grouped subclasses override this method.
        """
        return None

    def comparison_unit(self) -> str:
        """Return the experimental unit used by exported statistics."""
        return "observation"

    def export_settings(self) -> dict:
        """Return plot metadata written to the bundle settings file."""
        out = {"plot": type(self).__name__}
        title = getattr(self.plot.plotItem.titleLabel, "text", "")
        if title:
            out["title"] = str(title)
        frame = self.frame()
        if frame is not None:
            out["columns"] = [str(c) for c in frame.columns]
        return out

    def export_styled(self, path: str, *, ink: str = "", background: str = "",
                      grid: Optional[bool] = None,
                      **styling) -> Optional[str]:
        """Write the plot with the FILE's styling, leaving the screen alone.

        ``styling`` takes the same keywords :meth:`styled_snapshot` does --
        ``font_size``, ``line_width``, ``aspect``, ``x_title``, ``y_title``
        -- so the preview and the file go through one styling path and
        cannot disagree.
        """
        with self._dressed_for_the_file(ink, background, grid, **styling):
            return self.export(path)

    def save_styled(self):
        """Open the shared export styling and preview dialog.

        Returns
        -------
        int
            Qt dialog result code: accepted or rejected.
        """
        from .save_figure_dialog import SaveFigureDialog

        dialog = SaveFigureDialog(self, parent=self)
        return dialog.exec()

    def raster_pixels(self, source_width: float, source_height: float,
                      width_mm: Optional[float] = None,
                      height_mm: Optional[float] = None) -> tuple:
        """Return raster-export dimensions as ``(width, height)`` pixels.

        When export DPI is configured, width is derived from the page width;
        otherwise, source pixel width is retained. Height follows the selected
        canvas ratio, explicit page ratio, or source aspect ratio, in that
        order.
        """
        source_width = max(1.0, float(source_width))
        source_height = max(1.0, float(source_height))
        dpi = self._export_dpi
        if dpi:
            width = max(1, int(round(float(width_mm or self._export_width_mm)
                                     / 25.4 * float(dpi))))
        else:
            width = max(1, int(round(source_width)))
        ratio = self.canvas_ratio()
        if ratio is not None:
            height = max(1, int(round(width * float(ratio))))
        elif height_mm and width_mm:
            height = max(1, int(round(width * float(height_mm)
                                      / float(width_mm))))
        else:
            height = max(1, int(round(width * source_height / source_width)))
        return width, height

    def _shape_the_image(self, exporter, width_mm: Optional[float] = None,
                         height_mm: Optional[float] = None) -> None:
        """Size a raster exporter's output in pixels.

        The width and height parameters of pyqtgraph's ImageExporter are
        linked, so each is written with the other's handler blocked and both
        end up as :meth:`raster_pixels` asked for.
        """
        try:
            source = exporter.getSourceRect()
            width, height = self.raster_pixels(source.width(), source.height(),
                                               width_mm, height_mm)
            parameters = exporter.parameters()
            parameters.param("width").setValue(
                width, blockSignal=exporter.widthChanged)
            parameters.param("height").setValue(
                height, blockSignal=exporter.heightChanged)
        except Exception:       # a different exporter API
            pass

    def snapshot(self, width: int = SNAPSHOT_PX[0], *, ground=None):
        """A picture of this plot, even on a page nobody has opened.

        :param width: pixels across. The height follows from the plot's own
            aspect, exactly as :meth:`export` leaves it.
        :returns: a ``QPixmap``, or ``None`` when there is nothing to show.

        WHY THIS IS NOT ``grab()``. A live plot on a stacked page the user has
        never raised has never been through a layout pass, so its size is
        whatever its parent last guessed. Measured on the real regression
        screen, the volcano inside the collapsed gene splitter of an unshown
        page: ``volcano.size()`` is 100x9 and ``grab()`` returns a 100x9
        pixmap of ONE colour. That is the "blank box with a caption under it"
        that got the live tile deleted from the figure grid instead of fixed.

        Resizing the widget first does not fix it either, and that is worth
        writing down because it is the obvious repair: ``resize`` is honoured
        by ``size()`` and ignored by ``grab()``, because the splitter the
        widget sits in owns its geometry and re-imposes it. Measured, on a
        freshly built screen: ``resize(520, 380)`` then ``grab()`` still
        returns 100x9, with or without ``layout.activate()``, ``setGeometry``,
        ``processEvents`` or an explicit grab rectangle.

        So this renders THE SCENE rather than the widget, through the same
        pyqtgraph exporter :meth:`export` writes files with. The scene has no
        opinion about how big the widget on screen happens to be: 520x390 and
        236 distinct colours from the very widget that grabs blank.

        ``None`` for an empty plot is the other half. A tile showing an empty
        plot invites a click that opens an empty plot, and a run that has
        fitted nothing yet should have no tile at all rather than a misleading
        one.
        """
        if not self.plots_available or not len(self.plot.listDataItems()):
            return None
        self._sync_auto_range()
        from pyqtgraph import exporters

        try:
            with self._held_at_the_page_shape():
                return self._render_snapshot(exporters, width, ground)
        except Exception:
            # A picture is never worth taking the screen down for. The caller
            # pins nothing, which is the same thing that happens before a run.
            return None

    def _render_snapshot(self, exporters, width: int, ground):
        """Render the scene to a ``QPixmap`` ``width`` pixels across.

        Split out of :meth:`snapshot` so the canvas shape can be held around
        the render without the ``try`` that swallows a failed one also
        swallowing a failure to restore the geometry.
        """
        from PySide6.QtGui import QPixmap

        exporter = exporters.ImageExporter(self.plot.plotItem)
        exporter.parameters()["width"] = int(width)
        ratio = self.canvas_ratio()
        if ratio is not None:
            # THE SHAPE, IN THE PIXELS TOO. The scene is already held at the
            # proportion, but the exporter's linked height is recomputed from
            # the source rect it saw when the width was written -- so a
            # preview could come back one pixel out of square, which is the
            # difference between "the shape worked" and "nearly".
            try:
                exporter.parameters().param("height").setValue(
                    max(1, int(round(int(width) * float(ratio)))),
                    blockSignal=exporter.heightChanged)
            except Exception:           # other exporter
                pass
        try:
            # TRANSPARENT BY DEFAULT, like the tile behind it. The
            # exporter otherwise uses pyqtgraph's configured background,
            # and a tile painted onto an opaque slab is the "the graphs
            # still have a black background" report all over again.
            #
            # A CALLER MAY ASK FOR THE PAGE, and the save dialog does:
            # its preview is meant to be the file, and a file written
            # onto white while its preview showed transparent is a
            # preview of something else.
            exporter.parameters()["background"] = (
                QColor(0, 0, 0, 0) if ground is None else ground)
        except (KeyError, TypeError):   # old pyqtgraph
            pass
        image = exporter.export(toBytes=True)
        if image is None or image.isNull():
            return None
        pixmap = QPixmap.fromImage(image)
        return None if pixmap.isNull() else pixmap

    def restyle(self, background: Optional[str] = None,
                foreground: Optional[str] = None) -> None:
        """Re-read the figure colours, or take the ones given.

        Needed because pyqtgraph resolves ``foreground`` at construction:
        without this a theme switch leaves every open plot drawing its old
        ink, and on a dark-to-light switch that ink is invisible.
        """
        if background is None or foreground is None:
            resolved_bg, resolved_fg = _figure_colors()
            background = resolved_bg if background is None else background
            foreground = resolved_fg if foreground is None else foreground
        self._background, self._foreground = background, foreground
        pg.setConfigOptions(foreground=foreground)
        axis_pen = pg.mkPen(foreground)
        for edge in ("bottom", "left", "top", "right"):
            try:
                axis = self.plot.getAxis(edge)
            except Exception:               # absent axis
                continue
            axis.setPen(axis_pen)
            axis.setTextPen(axis_pen)
        title = getattr(self.plot.plotItem, "titleLabel", None)
        if title is not None and title.text:
            self.plot.setTitle(title.text, color=foreground)
        # A THEME SWITCH MUST NOT UNDO A CHOICE THE USER MADE. The loop above
        # has just painted the theme's ink over every axis; if the user set a
        # font or a line colour off the menu, that is what they asked this
        # plot to look like and it goes back on top.
        if self._line_colour is not None:
            self.set_line_style(colour=self._line_colour)
        if (self._font_colour is not None or self._font_size is not None
                or self._line_colour is None):
            # The captions are text and were just repainted with the axes'
            # ink by nothing at all -- they are LabelItems, which the loop
            # above does not reach. apply_text_style is what carries the
            # theme to them, so it runs on a plain theme switch too.
            self.apply_text_style()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Retire the parentless menus that belong to this plot."""
        try:
            from ..widget_cleanup import retire_pyqtgraph_menus

            retire_pyqtgraph_menus(self)
        except (ImportError, RuntimeError):
            pass
        super().closeEvent(event)


class VolcanoPlot(FastPlot):
    """Effect against -log10(p), with the FDR carried by colour and a line.

    The y axis is the raw P value and is continuous. The reasoning is worth
    keeping beside the code because the
    observation that produced it looked like a bug and was not.

    Benjamini-Hochberg's adjusted P is a cumulative minimum taken from the
    largest P downwards -- ``q_(i) = min over j >= i of (n * p_(j) / j)`` --
    and the ``min`` is both what enforces monotonicity and what creates ties:
    the moment a later rank produces a smaller value, EVERY earlier rank is
    pulled down onto it. Measured on reference runs: 823 q values
    with 31 distinct, 19 tied levels covering 811 coefficients, GRA14 and
    225160 both at 4.1150e-03. A volcano drawn against that has a staircase
    for a y-axis, and no transform can separate two tests that hold the same
    number.

    So the height is the RAW P, which is continuous and is the evidence per
    test, and the correction decides the COLOUR and the LINE, which is the
    discrete thing it actually is. The same two numbers are on the plot, each
    doing the job it can do, and nothing is invented -- which is also why the
    one thing this plot will not do is jitter the y axis to separate ties.
    That moves a point away from its own value, and this class exists because
    a plot was showing something the data did not say.
    """

    #: What the y axis measures, in the order the menu offers them.
    P_AXES = ("raw", "adjusted", "lfdr")

    #: How the corrected p is carried on the colour channel. One at a time, by
    #: construction rather than by
    #: policy: a dot cannot be coloured for its direction and for its q at
    #: once, so whichever is chosen the other is not shown and the legend
    #: and the caption say which is in force.
    Q_COLOURS = ("call", "ramp")

    #: The encoding that COMPOSES instead of competing (F.6): the colour
    #: stays wherever it is and the FDR goes into the mark itself. This is
    #: the only one of the six that can be on at the same time as another.
    Q_MARKS = ("none", "size", "opacity")

    #: Stops the ramp and the opacity ladder are quantised to.
    #:
    #: NOT ONE BRUSH PER POINT. `add_scatter` records the measurement: a
    #: brush constructed per point costs 39.5 ms on the real screen's 1,215
    #: coefficients against 3.5 ms for a reused set, and it is the whole of
    #: the lag this module was written to remove. 32 stops is more than a
    #: reader can distinguish on a scatter and caps the brush count at 32.
    Q_STOPS = 32

    #: The diameters "size by q" draws between. The weakest evidence is
    #: SMALLER than the plot's own 8.0 and the strongest is larger, so the
    #: encoding reads as a range rather than as "everything got bigger".
    Q_SIZE_RANGE = (3.0, 13.0)

    #: The alphas "opacity by q" draws between. The floor is well clear of
    #: invisible: a point faded to nothing is a point removed from the plot,
    #: which would imply data were absent rather than weakly supported.
    Q_ALPHA_RANGE = (55, 255)

    #: How many stops of the ramp the legend names. Five, because a legend
    #: is a key and not a colour bar -- and because 27 entries cost 40 ms of
    #: a 49 ms redraw, which is why the legend is opt-in at all.
    Q_LEGEND_STOPS = 5

    #: What the horizontal axis is called on each path. A fitted run puts a
    #: regression coefficient there; the permutation path puts a PARTIAL
    #: CORRELATION -- `standardized_marginal_effect`, copied into
    #: `coefficient` so the rest of the screen can read it by one name. The
    #: two are different quantities: one is on the response's scale and
    #: unbounded, the other is bounded in [-1, 1]. Calling both "coefficient"
    #: is what makes a reader ask whether these are actually coefficients.
    EFFECT_LABELS = {
        "fitted": "coefficient",
        "permutation": "standardized marginal effect (partial correlation)",
    }

    def __init__(self, parent=None):
        super().__init__(title="Volcano",
                         x_label=self.EFFECT_LABELS["fitted"],
                         y_label="-log10(p)", parent=parent)
        # NO SI PREFIX ON AN EFFECT SIZE. pyqtgraph factors a common power of
        # ten out of the tick labels and states it once in the axis title, so
        # a partial correlation running -0.06 to 0.50 was drawn as an axis
        # reading -100 to 400 titled "coefficient (x0.001)". That is correct
        # and unreadable: the number a reader wants to quote is 0.50, and
        # they should not have to multiply it back themselves.
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)
        #: Which of :data:`P_AXES` the height is. Raw by default.
        self._p_axis = "raw"
        #: Whether that was a person's choice rather than a default. A host
        #: may seed the axis; only a person may overrule one.
        self._p_axis_chosen = False
        #: The correction the user picked ON the plot, or None for the run's.
        self._correction: Optional[str] = None
        #: The correction the RUN used, as its table records it.
        self._run_method = ""
        self._alpha = 0.05
        #: ``(frame, kwargs)`` of the last draw, so changing the axis or the
        #: correction can redraw without the host being involved.
        self._results_call: Optional[tuple] = None
        #: Per family: ``(critical raw p or None, called, tested)``.
        self._families: dict = {}

        #: The recomputed values, aligned with the plotted frame.
        self._q_values = None
        self._lfdr_values = None
        self._called = None
        self._raw_values = None
        self._family_of = None
        self._raw_resolution = None
        #: The sentence that says which number is which. Never empty once
        #: anything is drawn: a volcano whose height is the raw P and whose
        #: colour is the FDR, with nothing saying so, is read in the
        #: direction of over-confidence.
        self._caption = ""
        #: ``(distinct, tested, finest)`` when the raw P is quantised, else
        #: None -- the permutation path, which is discrete before BH runs.
        self._raw_resolution = None
        self._y_floor = 1e-300
        #: Which of :data:`Q_COLOURS` has the colour channel. The binary
        #: call by default, which is F.1 and is the field's own volcano.
        self._q_colour = "call"
        #: Which of :data:`Q_MARKS` is composed on top of it. Neither by
        #: default: both are OFFERS, and nobody asked for one as a default.
        self._q_mark = "none"
        self._correction_writer = self._write_corrected_table

    # -------------------------------------------------------- what is drawn

    def name_the_effect(self, path: str) -> str:
        """Title the horizontal axis for the analysis that produced it.

        :param path: ``'fitted'`` or ``'permutation'``; anything else is
            treated as fitted, because a run whose path cannot be read is
            far more likely to be an ordinary fit than a permutation.
        :returns: the label applied.
        """
        label = self.EFFECT_LABELS.get(str(path).strip().lower(),
                                       self.EFFECT_LABELS["fitted"])
        self.plot.setLabel("bottom", label)
        return label

    def p_axis(self) -> str:
        """Which of :data:`P_AXES` the y axis measures."""
        return self._p_axis

    def set_p_axis(self, kind: str) -> None:
        """Draw the height against the raw P, the adjusted P, or the lfdr.

        ``"adjusted"`` IS KEPT AND IS NOT THE DEFAULT. It is honest and
        stepped, because that is what BH is, and a user reproducing a
        published figure drawn that way needs to be able to.
        """
        kind = str(kind)
        if kind not in self.P_AXES:
            raise ValueError(f"p axis must be one of {self.P_AXES}; got {kind!r}")
        # THE RUN-TIME HALF OF THE GREYING RULE. The menu entry for the
        # adjusted axis is DISABLED when no correction is in force -- because
        # the adjusted p then IS the raw p, and the axis is the one above with
        # a label that reads "adjusted p (None (raw P values))". But the menu
        # is not the only way in: the host drives this axis directly and
        # redraws on every level, baseline and compartment change, so a user
        # who chose the adjusted axis on a CORRECTED run and then moved to an
        # uncorrected one got exactly the label this instruction removed.
        # Same shape as `ml._require_backend`: a rule that lives only in the
        # widget that greys it is a rule with one entry point unguarded.
        if kind == "adjusted" and self.correction() in ("none", "", None):
            kind = "raw"
        self._p_axis = kind
        #: True once a person has picked an axis off this plot's own menu.
        self._p_axis_chosen = True
        self.redraw()

    def q_colour(self) -> str:
        """Which of :data:`Q_COLOURS` has the colour channel."""
        return self._q_colour

    def set_q_colour(self, mode: str) -> None:
        """Carry the FDR as a binary call, or as a continuous ramp over q.

        THEY CANNOT BOTH BE ON, and not because this refuses: a dot has one
        colour. The design says so and says what follows from it --
        whichever is chosen, the other is not shown, and the LEGEND says
        which is in force. That is the localisation rule again: one sentence
        per figure.

        THE RAMP DOES NOT REPLACE THE CALL AS THE DEFAULT. F.1 is the
        field's own volcano -- continuous height, binary colour, and the
        colour carrying the claim -- and a ramp answers a different question:
        it shows how far each test is from the threshold, which is a reading
        aid and not a call. It is an OFFER.
        """
        mode = str(mode)
        if mode not in self.Q_COLOURS:
            raise ValueError(
                f"q colour must be one of {self.Q_COLOURS}; got {mode!r}")
        self._q_colour = mode
        self.redraw()

    def q_mark(self) -> str:
        """Which of :data:`Q_MARKS` is composed on top of the colour."""
        return self._q_mark

    def set_q_mark(self, mode: str) -> None:
        """Put the FDR into the SIZE or the OPACITY of the mark, or neither.

        THE ONE ENCODING THAT COMPOSES. Size and
        opacity are channels the colour is not using, so this can be on at
        the same time as either colouring -- a volcano coloured by condition
        with the marks sized by q says both things at once, which is what the
        colour ramp cannot do.

        Both are offers, not defaults. Neither is on until it is asked for.
        """
        mode = str(mode)
        if mode not in self.Q_MARKS:
            raise ValueError(
                f"q mark must be one of {self.Q_MARKS}; got {mode!r}")
        self._q_mark = mode
        self.redraw()

    def _q_strength(self, q):
        """``q`` as 0 (weakest evidence on the plot) to 1 (strongest).

        On -log10(q) rather than on q, for the same reason the height is:
        q values pile up near 1 and the interesting ones are decades apart,
        so a linear ramp over q spends nearly all its colour on the tests
        nobody is looking at. NaN stays NaN -- a row with no q is not a weak
        result, and painting it at the bottom of the scale would be a made-up
        measurement (the rule `colour_by_column` already states for a
        missing value).
        """
        q = np.asarray(q, dtype=float)
        strength = np.full(q.shape, np.nan)
        usable = np.isfinite(q) & (q > 0)
        if not usable.any():
            return strength
        logged = -np.log10(q[usable])
        low, high = float(logged.min()), float(logged.max())
        if high <= low:
            # Every test on the plot holds the same q. Half way up, so the
            # ramp is one colour and is honest about being one colour.
            strength[usable] = 0.5
        else:
            strength[usable] = (logged - low) / (high - low)
        # A q of exactly zero underflowed; it is the strongest evidence
        # there is, not a missing value.
        strength[np.isfinite(q) & (q <= 0)] = 1.0
        return strength

    def _q_ramp(self, q):
        """``(brush_list, legend, missing)`` for the continuous ramp (F.5).

        Perceptually uniform, from the same five this module already allows
        on a continuous quantity: a rainbow puts bright bands where the data
        has none and reads as structure.
        """
        strength = self._q_strength(q)
        table = pg.colormap.get("viridis")
        lookup = table.getLookupTable(nPts=self.Q_STOPS, alpha=True)
        missing_brush = pg.mkBrush(QColor(MISSING_COLOUR))
        cache: dict = {}

        def _brush(step: int):
            brush = cache.get(step)
            if brush is None:
                r, g, b, a = (int(c) for c in lookup[step])
                brush = cache[step] = pg.mkBrush(QColor(r, g, b, a))
            return brush

        steps = np.where(np.isnan(strength), 0.0, strength)
        steps = np.clip(np.round(steps * (self.Q_STOPS - 1)),
                        0, self.Q_STOPS - 1).astype(int)
        brushes = [missing_brush if np.isnan(value) else _brush(step)
                   for value, step in zip(strength, steps)]
        missing = int(np.isnan(strength).sum())

        legend = {}
        finite = np.asarray(q, dtype=float)[np.isfinite(q)]
        if finite.size:
            # The stops are named by the q they stand for, strongest first,
            # so the key reads in the direction a volcano is read.
            for fraction in np.linspace(1.0, 0.0, self.Q_LEGEND_STOPS):
                step = int(round(fraction * (self.Q_STOPS - 1)))
                r, g, b, _a = (int(c) for c in lookup[step])
                # The q at this stop, inverted back out of the log scale.
                logged = -np.log10(np.clip(finite, 1e-300, 1.0))
                low, high = float(logged.min()), float(logged.max())
                value = 10 ** -(low + fraction * (high - low))
                legend[f"q {value:.2g}"] = QColor(r, g, b).name()
        if missing:
            legend[f"no q ({missing})"] = MISSING_COLOUR
        return brushes, legend, missing

    def _q_sizes(self, q):
        """One diameter per point, from the evidence against it (F.6)."""
        strength = self._q_strength(q)
        smallest, largest = self.Q_SIZE_RANGE
        filled = np.where(np.isnan(strength), 0.0, strength)
        return smallest + filled * (largest - smallest)

    def _q_opacity(self, brush_list, q, count: int):
        """``brush_list`` again, faded by the evidence against each point.

        COMPOSES WITH WHATEVER THE COLOUR IS, which is the point of F.6: the
        base colour is read off each point's own brush and only the alpha is
        replaced, so a volcano coloured by condition keeps its conditions.
        Quantised to :data:`Q_STOPS` alphas and cached by
        ``(base colour, step)``, so the brush count stays a few dozen rather
        than one per point.
        """
        strength = self._q_strength(q)
        floor, ceiling = self.Q_ALPHA_RANGE
        cache: dict = {}
        default = QColor(colour_for(0))
        out = []
        for index in range(count):
            base = default
            if brush_list is not None:
                try:
                    base = QColor(brush_list[index].color())
                except Exception:       # an odd brush
                    base = default
            value = strength[index] if index < len(strength) else np.nan
            fraction = 0.0 if np.isnan(value) else float(value)
            step = int(round(fraction * (self.Q_STOPS - 1)))
            key = (base.rgb(), step)
            brush = cache.get(key)
            if brush is None:
                alpha = floor + (ceiling - floor) * step / (self.Q_STOPS - 1)
                faded = QColor(base)
                faded.setAlpha(int(round(alpha)))
                brush = cache[key] = pg.mkBrush(faded)
            out.append(brush)
        return out

    def correction(self) -> str:
        """The correction being DRAWN, canonical, whoever chose it."""
        from ...multiple_testing import canonical_method

        chosen = self._correction or self._run_method or "fdr_bh"
        try:
            return canonical_method(chosen)
        except ValueError:
            return "fdr_bh"

    def run_correction(self) -> str:
        """The correction the RUN used, or ``""`` if the table does not say."""
        return self._run_method

    def set_correction(self, method) -> None:
        """Recompute the correction on the spot. ``None`` goes back to the run's."""
        from ...multiple_testing import canonical_method

        self._correction = None if method is None else canonical_method(method)
        if self._correction == "none" and self._p_axis == "adjusted":
            # Choosing "no correction" while the height IS the adjusted p
            # would leave the axis labelled "-log10(adjusted p, none)" over
            # numbers that are the raw p. The menu greys the adjusted entry
            # for the same reason; this is the one path that can reach the
            # state anyway, so it is unwound rather than left.
            self._p_axis = "raw"
        self.redraw()

    def caption(self) -> str:
        """The sentence naming which quantity is the height and which the call."""
        return self._caption

    def families(self) -> dict:
        """``{family: (critical raw p or None, called, tested)}`` as drawn."""
        return dict(self._families)

    def local_fdr_values(self):
        """The local FDR per row, computed once and per FAMILY.

        LAZY ON PURPOSE. The beta-uniform fit is 25 ms on the real screen's
        1,215 coefficients -- more than drawing the whole plot -- and the
        default axis does not use it. Computing it on every redraw would put
        the lag back that this module's whole first half exists to remove.
        """
        if self._lfdr_values is not None:
            return self._lfdr_values
        from ...multiple_testing import local_fdr

        if self._raw_values is None:
            return None
        values = np.full(self._raw_values.shape, np.nan)
        for name in self._families:
            mask = self._family_of == name
            values[mask] = local_fdr(self._raw_values[mask])
        self._lfdr_values = values
        return values

    def _forget_statistics(self) -> None:
        """Drop every number the last draw computed.

        A DRAW THAT PUT NOTHING ON THE PLOT MUST LEAVE NOTHING BEHIND. The
        caption, the q values and the per-family counts are read by the
        status line, the click handler and the table writer; left standing
        after an empty redraw they would describe the PREVIOUS screen, which
        is the worst kind of wrong number -- one that used to be right.
        """
        self._caption = ""
        self._families = {}
        self._q_values = None
        self._lfdr_values = None
        self._called = None
        self._raw_values = None
        self._family_of = None
        self._raw_resolution = None

    def redraw(self) -> int:
        """Draw the last table again with the current axis and correction."""
        if self._results_call is None:
            return 0
        frame, kwargs = self._results_call
        return self.set_results(frame, **kwargs)

    def set_results(self, frame, *, effect: str = "coefficient",
                    p_column: str = "p_value", label_column: str = "feature",
                    category_column: Optional[str] = None,
                    symbol_column: Optional[str] = None,
                    opacity_column: Optional[str] = None,
                    alpha: float = 0.05,
                    effect_threshold: Optional[float] = None,
                    key_column: Optional[str] = None,
                    drop_untested: bool = True,
                    compartment: Optional[str] = None,
                    q_column: Optional[str] = None,
                    run_method: Optional[str] = None):
        """Draw ``frame``. Returns the number of points actually plotted.

        :param frame: the coefficient table. Rows without a finite effect and
            raw p-value are left off the plot, while their original positions
            remain the identifiers used for linked selections.
        :param symbol_column: optional categorical column encoded by marker
            shape, independently of colour.
        :param opacity_column: optional categorical column encoded by opacity.
            Encodings are applied in the stable order colour, shape, opacity.
            The caller rejects requests for additional visual channels.

        :param p_column: the RAW P value. Not the corrected one -- the height
            is the raw P by default and the correction is recomputed here.
        :param q_column: the run's own corrected column, for the plot to
            check itself against. Found in the table when omitted.
        :param run_method: which correction the run used. Read off the
            table's ``multiple_testing_method`` column when omitted.
        :param compartment: one TAGM/LOPIT compartment to pick out against
            grey. ONE, not all 27 -- see :mod:`spacr.localisation`. It
            REPLACES any category colouring rather than combining with it: a
            volcano where a coloured dot might be coloured for its condition
            or for its compartment has no sentence.
        """
        from ...multiple_testing import (LOCAL_FDR_MIN_TESTS, METHODS,
                                         adjust_p_values, canonical_method,
                                         method_label)

        self._reset_scene()
        if frame is None or not len(frame):
            self._results_call = None
            self._forget_statistics()
            self.set_status("No coefficients to plot.")
            return 0

        # THE CALL IS REMEMBERED, NOT THE PICTURE. Switching the axis or the
        # correction redraws from the table the host handed over, so neither
        # costs a round trip through the host and neither can drift from what
        # is on screen.
        self._results_call = (frame, dict(
            effect=effect, p_column=p_column, label_column=label_column,
            category_column=category_column, alpha=alpha,
            effect_threshold=effect_threshold, key_column=key_column,
            drop_untested=drop_untested, compartment=compartment,
            q_column=q_column, run_method=run_method))
        self._alpha = float(alpha)

        columns = getattr(frame, "columns", ())
        if q_column is None:
            q_column = _first_column(frame, ("q_value", "adjusted_p_value"))
        if run_method is None:
            run_method = ""
            if "multiple_testing_method" in columns:
                seen = frame["multiple_testing_method"].dropna().unique()
                run_method = str(seen[0]) if len(seen) else ""
        # CANONICAL, so "bh" in an older table and "fdr_bh" in a new one are
        # the same run method and the menu ticks the same entry for both.
        try:
            self._run_method = canonical_method(run_method) if run_method else ""
        except ValueError:
            self._run_method = str(run_method)
        # A HOST STILL DRIVING THE AXIS THROUGH `p_column` IS HONOURED UNTIL
        # THE USER SAYS OTHERWISE. `RegressionResultsPanel` switches
        # raw/adjusted by handing over a different column, and reading that
        # as "the host asked for the adjusted axis" keeps its control working
        # while this plot owns the choice.
        #
        # A CHOICE MADE ON THE PLOT WINS FROM THEN ON, and that is not a
        # nicety: the host redraws on every level change, baseline change and
        # compartment change, so without this any one of them would silently
        # put the axis back and the user would watch their choice undo itself.
        raw_column = _first_column(frame, ("p_value", "p", "pvalue"))
        if q_column and p_column == q_column:
            if not self._p_axis_chosen:
                self._p_axis = "adjusted"
            p_column = raw_column or p_column
        elif not self._p_axis_chosen and raw_column and p_column == raw_column:
            self._p_axis = "raw"

        # NUISANCE TERMS ARE NOT HYPOTHESES, AND THEY OWN THE AXIS.
        #
        # The intercept and the plate row/column effects are covariates: they
        # are fitted so the guide effects come out clean, not so anyone can
        # ask whether they differ from zero. spacr.ml already draws that line
        # -- it leaves them out of the multiple-testing family, which is why
        # they leave a fit with q_value = NaN -- and plotting them draws a
        # different experiment from the one the q-values describe.
        #
        # It is not a rounding error. On plate1_dv the intercept sits at
        # -log10(p) = 45.5 against 12.5 for the strongest real hit and 2.3 at
        # the 99th percentile, so ONE untestable row makes the y-axis 3.6x
        # taller than the data and flattens the whole screen into the bottom
        # of it. A fit carrying row and column terms has ~25 of them.
        untested = 0
        if drop_untested and "feature" in columns:
            from ...hits import tested_family

            keep_rows = tested_family(frame["feature"])
            if not keep_rows.all():
                untested = int((~keep_rows).sum())
                frame = frame.loc[keep_rows].reset_index(drop=True)
                if not len(frame):
                    self._forget_statistics()
                    self.set_status(
                        f"No testable coefficients: all {untested} rows are "
                        "nuisance terms.")
                    return 0
                columns = frame.columns

        effects = _finite(frame[effect]) if effect in frame else np.zeros(len(frame))
        raw = _finite(frame[p_column]) if p_column in frame \
            else np.full(len(frame), np.nan)

        # ------------------------------------------------ the correction
        #
        # RECOMPUTED, AND WITHIN THE RIGHT FAMILY. The correction applies
        # within a LEVEL -- a run at level='both' fits twice and each fit is
        # its own family (instruction 128 R) -- so pooling whatever happens
        # to be on screen would change n and with it every q value, quietly
        # and in the direction that makes the run look weaker than it is.
        # `hits.family_labels` is the single statement of that split.
        method = self.correction()
        if "feature" in columns:
            from ...hits import family_labels

            families = family_labels(frame["feature"])
        else:
            families = np.full(len(frame), "all", dtype=object)
        q = np.full(len(frame), np.nan)
        called = np.zeros(len(frame), dtype=bool)
        self._families = {}
        self._family_of = families
        self._raw_values = raw
        # THE LOCAL FDR IS NOT COMPUTED UNLESS IT IS WANTED. Measured on the
        # real screen's 1,215 coefficients: the mixture fit is 25 ms of a 40
        # ms redraw -- more than drawing the plot -- and the default axis is
        # the raw P, which does not use it. It is computed when the axis
        # asks for it and when a click asks for it, once, and cached.
        self._lfdr_values = None
        for name in sorted({str(f) for f in families if f}):
            mask = families == name
            fam_q, fam_called = adjust_p_values(raw[mask], method=method,
                                               alpha=self._alpha)
            q[mask] = fam_q
            called[mask] = fam_called
            # THE CRITICAL RAW P, WHICH IS EXACT AND IS NOT ALPHA.
            #
            # Every correction here is monotone in the raw P within a family,
            # so the set it calls is a lower set: there is a rank k with
            # every p <= p_(k) called and everything above it not. For BH
            # that is the textbook identity q_(i) <= alpha iff
            # p_(i) <= alpha*i/n at the largest such i. One horizontal line
            # at -log10(p_(k)) therefore divides this plot EXACTLY as the FDR
            # does, on a continuous axis, with no steps anywhere.
            #
            # Drawing it at -log10(alpha) instead is the mistake this
            # replaces: that is the UNCORRECTED threshold and it calls far
            # too much of the screen.
            critical = (float(np.max(raw[mask][fam_called]))
                        if fam_called.any() else None)
            self._families[name] = (critical, int(fam_called.sum()),
                                    int(np.isfinite(raw[mask]).sum()))
        self._q_values, self._called = q, called
        # IS THE RAW P ITSELF QUANTISED? The permutation path is discrete
        # TWICE OVER and that is why it looks worst: a permutation p is
        # (1 + #{null <= observed}) / (n + 1), so 1,000 permutations admit
        # only 1,001 possible values and many guides already share one before
        # BH ever runs. Saying so is the difference between a user raising
        # `guide_permutations` and a user concluding the plot is broken.
        finite_raw = raw[np.isfinite(raw)]
        self._raw_resolution = None
        if finite_raw.size:
            distinct = int(np.unique(finite_raw).size)
            if distinct < 0.95 * finite_raw.size:
                positive = finite_raw[finite_raw > 0]
                self._raw_resolution = (
                    distinct, int(finite_raw.size),
                    float(positive.min()) if positive.size else 0.0)

        # DOES THE PLOT AGREE WITH THE TABLE BESIDE IT? results.csv carries
        # the run's own q values; recomputing the run's own method on the
        # run's own family has to reproduce them, and if it does not, the
        # user is looking at two analyses and is entitled to know.
        agrees = None
        if q_column and q_column in columns:
            try:
                same = canonical_method(self._run_method) == method
            except ValueError:
                same = False
            if same:
                stored = _finite(frame[q_column])
                both = np.isfinite(stored) & np.isfinite(q)
                agrees = bool(both.any() and np.allclose(
                    stored[both], q[both], rtol=1e-6, atol=1e-12))

        # ------------------------------------------------------ the height
        if self._p_axis == "adjusted":
            values = q
            axis_label = f"-log10(adjusted p, {method})"
        elif self._p_axis == "lfdr":
            values = self.local_fdr_values()
            axis_label = "-log10(local FDR)"
        else:
            values = raw
            axis_label = "-log10(p)"
        # A p of exactly zero is a real result underflowing, not a mistake;
        # clamping keeps it on the plot instead of sending it to infinity.
        smallest = np.nanmin(values[values > 0]) if np.any(values > 0) \
            else 1e-300
        #: The floor a zero is drawn at, shared with the threshold line so
        #: the two cannot end up on different scales.
        self._y_floor = float(smallest * 1e-3)
        neglog = -np.log10(np.clip(values, self._y_floor, 1.0))
        self.plot.setLabel("left", axis_label)

        brush_list, legend = None, {}
        # F.5 -- THE RAMP TAKES THE COLOUR CHANNEL, and what it takes it from
        # is named. A dot cannot be coloured for its condition and for its q
        # at once, so the colouring that would otherwise be in force is not
        # drawn and the caption says which one it was. Silently dropping it
        # is how a reader ends up reading a q ramp as a condition.
        ramp_on = self._q_colour == "ramp" and bool(np.isfinite(q).any())
        displaced = ""
        if ramp_on:
            brush_list, legend, _missing = self._q_ramp(q)
            if compartment:
                displaced = f"the {compartment} colouring"
            elif category_column:
                displaced = f"the {category_column} colouring"
            else:
                displaced = "the called/not-called colouring"
        elif compartment and compartment == self._all_compartments():
            # EVERY COMPARTMENT AT ONCE, asked for on 2026-08-20: "Colour by
            # lets me color by a single location, all should be an option."
            #
            # ONE AT A TIME WAS A DECISION, NOT AN OVERSIGHT -- "everything is
            # grey except what the sentence is about", and the 27-colour
            # legend measured 40 ms of a 49 ms redraw. So this is offered
            # BESIDE that, never instead of it, and it is built on the
            # CATEGORICAL path below rather than a Python loop over brushes,
            # which is where that 40 ms actually went.
            from ...localisation import of as compartment_of

            names = compartment_of(frame)
            if len(names):
                labelled = names.replace("", "elsewhere")
                brush_list, legend = self._categorical_brushes(labelled)
        elif compartment:
            # ONE COMPARTMENT AGAINST GREY. Two brushes and a two-entry
            # legend: the 27-colour version is what the house style forbids
            # and, measured, its legend cost 40 ms of a 49 ms redraw.
            from ...localisation import mask as compartment_mask

            inside = compartment_mask(frame, compartment).to_numpy()
            if inside.any():
                here = pg.mkBrush(HIGHLIGHT)
                elsewhere = pg.mkBrush(MUTED)
                brush_list = [here if flag else elsewhere for flag in inside]
                legend = {f"{compartment} ({int(inside.sum())})": HIGHLIGHT,
                          f"{int((~inside).sum())} elsewhere": MUTED}
        elif category_column and category_column in frame:
            # Categorical codes are computed in C; the alternative is a Python
            # loop over 1,215 pandas values plus a QColor.rgba() per point,
            # which cost 45 ms of the 48 ms this used to take.
            import pandas as _pd

            brush_list, legend = self._categorical_brushes(
                frame[category_column])

            # THE COUNT BESIDE EACH LABEL. Asked for 2026-08-17: "beside the
            # label on the graph should be the count of each label".
            #
            # It is not decoration on a screen. `nc` and `pc` are three and
            # twenty-four points among twelve hundred, and a legend that
            # names them without saying so invites reading a two-point
            # cluster as a group -- the same reason the compartment legend
            # and the gene/guide menu already carry theirs.
            #
            # Counted with np.bincount over the CODES, not by grouping the
            # frame -- see `_categorical_brushes`.
        elif called.any() or np.isfinite(q).any():
            # THE FIELD'S OWN VOLCANO: continuous height, binary colour, and
            # the colour carrying the claim. Only when nothing else has
            # claimed the colour channel -- a dot cannot be coloured for its
            # condition and for its q at once, and the legend says which is
            # in force.
            here, elsewhere = pg.mkBrush(HIGHLIGHT), pg.mkBrush(MUTED)
            brush_list = [here if flag else elsewhere for flag in called]
            n_called = int(called.sum())
            legend = {f"called ({n_called})": HIGHLIGHT,
                      f"not called ({int(len(called) - n_called)})": MUTED}

        # NO PER-POINT WORK BEFORE DRAWING.
        #
        # This used to build a label string for all 1,215 rows up front, three
        # `frame[col].iloc[i]` lookups each. Pandas scalar indexing in a Python
        # loop is ~3,600 lookups to draw a scatter plot, and it cost more than
        # the drawing did. The frame is kept instead and a label is formatted
        # for the ONE point that gets clicked -- which is the only one anybody
        # ever reads.
        self._frame = frame
        self._label_column = label_column
        self._effect_column = effect
        self._p_column = p_column
        self._labels = ()

        # `feature` is the design-matrix term name and is one-to-one with the
        # row -- checked on the real screen: 1,213 rows, 1,213 distinct. `gene`
        # and `grna` are NOT keys, because a gene has several guides and
        # several rows, so joining on either highlights an arbitrary one.
        key = key_column or ("feature" if "feature" in frame.columns
                             else label_column)
        self.set_keys(frame[key] if key in frame.columns else frame.index)

        # F.6 -- THE ENCODING THAT COMPOSES. Size and opacity are channels
        # the colour is not using, so either can be on at the same time as
        # any of the colourings above: a volcano coloured by condition with
        # its marks sized by q says both things at once, which is exactly
        # what the ramp cannot do.
        size_list = None
        if self._q_mark == "size":
            size_list = self._q_sizes(q)
        elif self._q_mark == "opacity":
            brush_list = self._q_opacity(brush_list, q, len(neglog))
        # THE SECOND AND THIRD CHANNELS. Only when the FIRST is in force:
        # a shape that means one thing beside a colour that means the q
        # value is two claims on one dot with nothing saying which.
        symbol_list = None
        layered = bool(category_column) and not compartment
        if layered and symbol_column and symbol_column in frame:
            symbol_list, shapes = self._categorical_symbols(
                frame[symbol_column])
            self._shape_legend = (symbol_column, shapes)
        else:
            self._shape_legend = None
        if layered and opacity_column and opacity_column in frame:
            brush_list = self._categorical_opacity(
                brush_list, frame[opacity_column], len(neglog))
            # THE COLUMN AND ITS LEVELS, because a fade nobody can name is
            # not an encoding. The legend draws them at the alphas the
            # points got, from the same ramp.
            self._opacity_legend = (opacity_column,
                                    self._levels_of(frame[opacity_column]))
        else:
            self._opacity_legend = None
        self.add_scatter(effects, neglog, brush_list=brush_list,
                         symbol_list=symbol_list,
                         size_list=size_list)
        controls = METHODS[method].controls
        level = f"{controls} {self._alpha:g}" if controls != "nothing" \
            else f"p {self._alpha:g}"
        drawn_lines = self._add_significance_lines(level)
        if effect_threshold:
            for sign in (-1, 1):
                self.add_line(x=sign * abs(effect_threshold), colour="#8C8C8C")

        # THE LEGEND IS OPT-IN, AND IT IS THE REASON WHY.
        #
        # Twenty-seven entries cost 40 ms of a 49 ms redraw -- each one builds
        # a ScatterPlotItem and a LabelItem. It is the identical cost that made
        # matplotlib's version 63 ms, so bringing it across unchanged would
        # have carried the lag over to the new library and wasted the switch.
        #
        #     scatter alone, 1,215 points        3.4 ms
        #     the same plus a 27-entry legend   43.7 ms
        #
        # So the plot draws without one and offers a checkbox. Colour still
        # identifies the compartments; the legend only names them, and naming
        # them is worth 40 ms when asked for and not before.
        self._legend_colours = legend
        # THE OTHER TWO CHANNELS COUNT TOO. A volcano whose shape carries a
        # column and whose legend can only be switched on when a COLOUR
        # column is chosen is a picture with an unreachable key.
        entries = len(self._legend_entries())
        if entries:
            self._legend_box.setEnabled(True)
            self._legend_box.setText(f"legend ({entries})")
            if self._legend_box.isChecked():
                self._build_legend()
        else:
            self._legend_box.setEnabled(False)

        # A SELECTION SURVIVES A REDRAW. plot.clear() took the marker with it,
        # so it goes back on -- otherwise changing the colouring, or any other
        # setting, silently deselects whatever the user was looking at.
        if self._selected_key is not None:
            self.highlight_key(self._selected_key)

        plotted = int(np.sum(~(np.isnan(effects) | np.isnan(neglog))))
        self._caption = self._build_caption(
            method, level, agrees, drawn_lines,
            bool(brush_list) and not (compartment or category_column
                                      or ramp_on),
            displaced=displaced)
        note = f"{plotted} coefficients."
        if untested:
            # Reported, not silently removed: the difference between a filter
            # and a lie.
            note += (f" {untested} nuisance "
                     f"term{'s' if untested != 1 else ''} not shown (not "
                     "tested, so no q-value).")
        self._offer_p_axes(method, int(np.isfinite(raw).sum()),
                           LOCAL_FDR_MIN_TESTS, method_label)
        self._offer_corrections(method)
        self._offer_encodings(method, method_label)
        self.set_status(f"{note} {self._caption} Click a point for detail.")
        return plotted

    # ------------------------------------------------------- the line and
    #                                                          the sentence

    def _add_significance_lines(self, level: str) -> int:
        """Draw the threshold. Returns how many lines went on.

        ZERO IS AN ANSWER. When nothing is called there is no rank k, there
        is no critical P value, and there is no line -- saying so is a
        finding, and drawing one at alpha instead would claim a threshold the
        procedure never reached.
        """
        if self._p_axis != "raw":
            # The corrected axis is the one place alpha IS the threshold.
            name = "q" if self._p_axis == "adjusted" else "lfdr"
            self.add_line(y=-np.log10(self._alpha),
                          label=f"{name}={self._alpha:g}")
            return 1
        criticals = {name: value[0] for name, value in self._families.items()
                     if value[0] is not None}
        if not criticals:
            return 0
        drawn = 0
        # DEDUPED ON THE EXACT VALUE, not on a rounded one. `round(p, 15)`
        # was here and it sent every P value below 1e-15 to zero -- so a
        # screen whose critical P is 1.7e-20 drew its line at -log10(0), and
        # pyqtgraph died on the infinity ("cannot convert float NaN to
        # integer") rather than on anything a reader could act on. Rounding
        # to DECIMAL places is never right for a P value; the two families'
        # criticals are computed from the same array and compare exactly.
        distinct = sorted(set(criticals.values()))
        for value in distinct:
            if len(distinct) > 1:
                who = ", ".join(sorted(name for name, v in criticals.items()
                                       if v == value))
                text = f"{level} ({who}): p<={value:.3g}"
            else:
                text = f"{level}: p<={value:.3g}"
            # THE LINE SITS ON THE SAME FLOOR THE POINTS DO. A P value that
            # underflowed to zero is a real result, and the scatter already
            # clamps it rather than sending it to infinity; a threshold line
            # that did not use the same floor would leave the plot.
            self.add_line(y=-np.log10(max(value, self._y_floor)), label=text)
            drawn += 1
        return drawn

    def _build_caption(self, method: str, level: str, agrees,
                       drawn_lines: int, colour_is_the_call: bool,
                       displaced: str = "") -> str:
        """The sentence that says which number is which.

        THE CAPTION IS NOT OPTIONAL. A volcano whose height is the raw P and
        whose colour is the FDR, with nothing saying so, is a figure a reader
        misreads in the direction of over-confidence -- and it is the reason
        this default is safe to ship at all.
        """
        from ...multiple_testing import method_label

        label = method_label(method)
        parts = []
        height = {"raw": "the raw p", "adjusted": f"the adjusted p ({label})",
                  "lfdr": "the local FDR"}[self._p_axis]
        if self._p_axis == "raw":
            parts.append(f"Height is {height}; {label} decides the colour "
                         f"and the line.")
        else:
            parts.append(f"Height is {height}.")
        if colour_is_the_call:
            parts.append(f"Colour is the call at {level}.")
        # ONE SENTENCE PER FIGURE saying which encoding has which channel.
        # A ramp that silently replaced a condition colouring, or a mark
        # sized by q with nothing saying so, is a figure that shows one thing
        # and is read as another.
        if self._q_colour == "ramp" and displaced:
            parts.append(f"Colour is a continuous ramp over q, brightest at "
                         f"the smallest q; {displaced} is off while it is.")
        if self._q_mark == "size":
            parts.append("Point SIZE is the evidence against the null: "
                         "largest at the smallest q. It composes with the "
                         "colour rather than replacing it.")
        elif self._q_mark == "opacity":
            parts.append("Point OPACITY is the evidence against the null: "
                         "most solid at the smallest q. It composes with the "
                         "colour rather than replacing it.")
        called = sum(value[1] for value in self._families.values())
        tested = sum(value[2] for value in self._families.values())
        if self._p_axis == "raw":
            if drawn_lines:
                thresholds = ", ".join(
                    f"{name} p<={value[0]:.3g}"
                    for name, value in sorted(self._families.items())
                    if value[0] is not None)
                parts.append(f"{level} ({thresholds}): {called} of {tested} "
                             f"called.")
                # WHY THE LINE ALWAYS TOUCHES A POINT, said once, because it
                # reads as a coincidence and is not one. The threshold is
                # `max(p over the called tests)` -- an OBSERVED value, not a
                # formula evaluated in the abstract -- so it is necessarily
                # some test's own p, and the line therefore lands exactly on
                # the last test it called. A reader who notices two genes
                # sitting on the line is seeing the borderline, and asked
                # whether those are in or out; `p<=` in the label answers it
                # and this says which points those are.
                parts.append(
                    "The threshold is itself an observed p, so the line lands "
                    "on the last test it called; points on the line are "
                    "called.")
            else:
                parts.append(f"Nothing is called at {level}, so there is no "
                             f"threshold line to draw.")
        else:
            parts.append(f"{called} of {tested} called at {level}.")
        if len(self._families) > 1:
            parts.append(f"Corrected within each level separately "
                         f"({', '.join(sorted(self._families))}).")
        if self._correction and self._run_method and \
                self._correction != self._run_method:
            parts.append(f"The run used {self._run_method}; this is a VIEW "
                         f"and the exported table still holds the run's q.")
        elif not self._run_method:
            parts.append("The table does not record which correction the run "
                         "used, so this one is the plot's own.")
        if agrees is False:
            parts.append("WARNING: recomputing the run's own method does not "
                         "reproduce the table's q values, so the two "
                         "disagree about the family.")
        if self._p_axis == "lfdr":
            parts.append("The local FDR is continuous by construction and "
                         "models the alternatives as Beta(a, 1).")
        if self._raw_resolution is not None:
            distinct, total, finest = self._raw_resolution
            parts.append(
                f"The RAW p is itself quantised -- {distinct:,} distinct "
                f"values among {total:,}, the smallest {finest:.3g} -- so a "
                f"permutation p can be no finer than 1/(permutations + 1) "
                f"and raising that count is what buys resolution here.")
        return " ".join(parts)

    # ------------------------------------------------------------- the menu

    def _offer_p_axes(self, method: str, tested: int, minimum: int,
                      method_label) -> None:
        """Put the three honest y-axes on the plot's own menu."""
        too_small = ("" if tested >= minimum else
                     f"a family of {tested} is too small to read a density "
                     f"off; {minimum} are needed")
        # NO CORRECTION MEANS THERE IS NO ADJUSTED AXIS. With
        # `multiple_testing_method='none'` the q value written for every row
        # EQUALS its raw p, so this entry offered a second copy of the axis
        # above it under a name -- "adjusted p — None (raw P values)
        # (stepped)" -- that promises a number the run never computed. A user
        # who picks it sees an identical plot and concludes the correction
        # made no difference, which is the reading this module exists to
        # prevent.
        #
        # GREYED, NOT REMOVED (INVARIANTS 6), and it says why: the answer to
        # "where is the adjusted axis" is "this run applied no correction",
        # which a missing entry does not give. The condition is the
        # correction IN FORCE and not the run's, because a plot recorrected
        # from the menu has real q values whatever the run did.
        uncorrected = ("" if str(method) != "none" else
                       "this run applied no correction, so the adjusted p "
                       "IS the raw p and the axis would be the one above")
        adjusted_label = ("adjusted p — no correction applied" if uncorrected
                          else f"adjusted p — {method_label(method)} (stepped)")
        self.offer_p_values([
            ("raw p (continuous)", lambda: self.set_p_axis("raw"),
             self._p_axis == "raw"),
            (adjusted_label,
             lambda: self.set_p_axis("adjusted"),
             self._p_axis == "adjusted", uncorrected),
            ("local FDR (continuous)", lambda: self.set_p_axis("lfdr"),
             self._p_axis == "lfdr", too_small),
        ])

    def _offer_corrections(self, method: str) -> None:
        """Every correction spaCR knows, recomputed on the spot."""
        from ...multiple_testing import METHODS

        options = []
        for key, spec in METHODS.items():
            label = spec.label
            if self._run_method and key == self._run_method:
                label = f"{label} (the run's)"
            options.append((label,
                            lambda k=key: self.set_correction(k),
                            key == method))
        self.offer_corrections(options)

    def _offer_encodings(self, method: str, method_label) -> None:
        """Every field-acceptable way of showing the adjusted p (F).

        THE TWO GROUPS ARE DIFFERENT KINDS OF THING and the labels say so.
        The colour entries are exclusive of each other because a dot has one
        colour; the mark entries compose with whichever colour is in force,
        which is the one property F.6 was singled out for.

        Both are OFFERS. Nothing here is a default and nothing here is
        chosen for the user -- F.1's binary call keeps the colour channel
        until somebody says otherwise.
        """
        q = self._q_values
        label = method_label(method)
        has_q = q is not None and bool(np.isfinite(q).any())
        no_q = "" if has_q else "this fit has no corrected p to encode"
        flat = ""
        if has_q and np.unique(q[np.isfinite(q)]).size < 2:
            # The staircase at its limit: one q for the whole screen. A ramp
            # over it is one colour, and offering it live would be the
            # present-but-inert control instruction 106 forbids.
            flat = ("every q on this plot is the same number, so a ramp "
                    "would be one colour")
        self.offer_encodings([
            (f"colour: called or not, at the {label} threshold",
             lambda: self.set_q_colour("call"), self._q_colour == "call",
             no_q),
            ("colour: a continuous ramp over q",
             lambda: self.set_q_colour("ramp"), self._q_colour == "ramp",
             no_q or flat),
            ("mark: nothing (colour only)",
             lambda: self.set_q_mark("none"), self._q_mark == "none"),
            ("mark: size by q — composes with the colour",
             lambda: self.set_q_mark("size"), self._q_mark == "size", no_q),
            ("mark: opacity by q — composes with the colour",
             lambda: self.set_q_mark("opacity"), self._q_mark == "opacity",
             no_q),
        ])

    def _write_corrected_table(self) -> Optional[str]:
        """Write the displayed multiple-testing correction to a CSV file.

        Return the selected path, or ``None`` when no corrected values exist
        or the save dialog is canceled.
        """
        from PySide6.QtWidgets import QFileDialog

        if self._frame is None or self._q_values is None:
            return None
        path, _ = QFileDialog.getSaveFileName(
            self, "Write this correction as a table", "results_recorrected.csv",
            "CSV (*.csv)", options=QFileDialog.DontUseNativeDialog)
        if not path:
            return None
        frame = self._frame.copy()
        method = self.correction()
        frame["multiple_testing_method"] = method
        frame["q_value"] = self._q_values
        frame["local_fdr"] = self.local_fdr_values()
        frame[f"called_at_{self._alpha:g}"] = self._called
        frame.to_csv(path, index=False)
        self.set_style_note(f"Wrote {method} q values for "
                            f"{len(frame)} coefficients to {path}.")
        return path

    def _detail(self, index: int) -> str:
        """The three numbers behind one dot, in O(1).

        A user who clicks a point on a raw-P axis has to be able to see the
        q that decided its colour, or the colour is an assertion they cannot
        check.
        """
        if self._q_values is None or index >= len(self._q_values):
            return ""
        q = self._q_values[index]
        if not np.isfinite(q):
            return "not in the tested family, so no q"
        parts = [f"q={q:.3g} ({self.correction()})"]
        lfdr = self.local_fdr_values()
        if lfdr is not None and index < len(lfdr) and np.isfinite(lfdr[index]):
            parts.append(f"local FDR={lfdr[index]:.3g}")
        if self._called is not None and index < len(self._called):
            parts.append("called" if self._called[index] else "not called")
        return "   ".join(parts)


class EffectRankPlot(FastPlot):
    """Every coefficient ranked by effect, as a dot with its interval.

    The interactive twin of :func:`spacr.figures.panels.effect_rank`, and the
    panel that answers what a volcano structurally cannot: HOW BIG, and how
    sure. A volcano ranks by significance, so an effect of 0.02 measured on
    six hundred wells outranks one of 2.0 measured on four; ranking by the
    effect itself puts them the other way round, and the interval drawn
    through each dot is what says which of the two to believe.

    A BAR CHART OF COEFFICIENTS IS THE WRONG PICTURE, which is why this is
    dots and lines. A bar replaces every observation with one height and hides
    the uncertainty that decides whether to believe any of them -- and on a
    ranked list, that uncertainty is the only question worth asking.

    THE SAVED PANEL DRAWS THE STRONGEST FOURTEEN AND THIS ONE DRAWS THEM ALL.
    That is the difference a zoomable plot is FOR: a sheet has one cell and
    has to choose, a screen does not. The opening view is the strongest
    :data:`LABELLED`, because that is as many names as a y-axis can carry and
    still be read, and "Reset view" reaches the rest -- the same rule
    :class:`GuideAgreementPlot` follows for its over-represented gene, and for
    the same reason: a point outside the opening view is still a point, and a
    point that was dropped is gone.
    """

    #: What multiplies a standard error into half an interval. 1.96 is the
    #: normal 95%, which is what the saved panel draws and what a reader of a
    #: regression table assumes unless they are told otherwise.
    INTERVAL_Z = 1.96

    #: How many names the y-axis carries, and how many rows the opening view
    #: shows. Past this the labels stop being legible at any window size and
    #: the reader is decoding a wall of text instead of reading a figure.
    LABELLED = 40

    #: Ink for a coefficient that was not called -- the house rule's default.
    GREY = MUTED
    #: The two directions a called coefficient can point.
    UP_INK = UP
    DOWN_INK = DOWN

    def __init__(self, parent=None):
        super().__init__(title="Effect rank", x_label="effect size",
                         y_label="", parent=parent)
        #: Every array below is in FRAME ORDER, not drawing order, because
        #: `_detail` is handed a frame row. The plot is sorted; the record of
        #: what it drew must not be, or a click would report its neighbour.
        self._effects: np.ndarray = np.empty(0)
        self._half: np.ndarray = np.empty(0)
        self._significance: np.ndarray = np.empty(0)
        self._significance_name = ""
        self._names: Sequence[str] = ()
        # RANK 1 AT THE TOP. A ranked list is read downwards, and pyqtgraph's
        # y-axis grows upwards, so without this the strongest effect sits at
        # the bottom of the panel and the reader starts at the weakest.
        self.plot.getViewBox().invertY(True)

    def set_results(self, frame, *, effect: str = "coefficient",
                    error_column: Optional[str] = None,
                    significance_column: Optional[str] = None,
                    label_column: str = "feature",
                    key_column: Optional[str] = None,
                    alpha: float = 0.05,
                    drop_untested: bool = True) -> int:
        """Draw ``frame`` ranked by absolute effect. Return the number of dots drawn.

        :param frame: the coefficient table.
        :param effect: the fitted-effect column.
        :param error_column: the standard error, so an interval can be drawn.
            ``None`` looks for :data:`ERROR_COLUMNS`; a table carrying none
            gets dots and no bars, and the status line SAYS the effects are
            drawn without their uncertainty rather than leaving a reader to
            assume they are exact.
        :param significance_column: what decides the colour. ``None`` looks
            for :data:`CORRECTED_P_COLUMNS`; :data:`NO_SIGNIFICANCE` says this
            table has none, which is a different statement from "go and look".
        :param label_column: the column a dot is named by when the frame
            carries no gene or guide of its own.
        :param key_column: the identifier every other view joins on.
        :param alpha: the cut a coefficient is coloured at.
        :param drop_untested: leave the nuisance terms off, as the volcano and
            :func:`spacr.figures.panels.effect_rank` both do.

            NOT FOR THE AXIS, and that is worth writing down because it is the
            obvious reason and it is wrong here. The volcano drops the
            intercept because it OWNS the p-axis -- 3.6x the tallest real hit
            on plate1_dv. Measured on the TSG101 screen, its COEFFICIENT is
            0.190 against a tested maximum of 4.37, so by effect it ranks 547
            of 1,213 and stretches nothing at all.

            It is dropped because it is not a hypothesis. Its ``q_value`` is
            NaN -- ``perform_regression`` leaves the covariates out of the
            multiple-testing family -- so it would sit halfway down a ranked
            list of hypotheses, permanently grey, with no verdict available
            for it and nothing on the picture saying why. A fit carrying plate
            row and column terms has ~25 more of them.

        THE SORT IS THE TRAP, and here it is the plot's whole shape: drawn dot
        n is the nth LARGEST effect and almost never row n of the table. The
        frame rows are therefore carried through the sort explicitly
        (``rows=``) rather than re-derived from the drawing order -- see
        :meth:`FastPlot.add_scatter`, where the same trap is written out for
        the Q-Q.
        """
        self._reset_scene()
        self._frame = None
        self._names = ()
        self._effects = self._half = self._significance = np.empty(0)
        self._significance_name = ""
        if frame is None or not len(frame):
            self.set_keys(())
            self.set_status("No coefficients to rank.")
            return 0

        untested = 0
        if drop_untested and "feature" in getattr(frame, "columns", ()):
            from ...hits import tested_family

            keep = tested_family(frame["feature"])
            if not keep.all():
                untested = int((~keep).sum())
                frame = frame.loc[keep]
                if not len(frame):
                    self.set_keys(())
                    self.set_status(
                        f"No testable coefficients: all {untested} rows are "
                        f"nuisance terms, which are fitted so the guide "
                        f"effects come out clean rather than to be ranked.")
                    return 0
        # POSITIONAL FROM HERE ON. Every row index this method hands to
        # `add_scatter`, and every index `_detail` is later asked about, is a
        # position in THIS frame; a caller's filtered frame arrives with holes
        # in its index and `.iloc` would then disagree with `.loc`.
        frame = frame.reset_index(drop=True)
        self._frame = frame

        effects = (_finite(frame[effect]) if effect in frame.columns
                   else np.full(len(frame), np.nan))
        error = error_column or _first_column(frame, ERROR_COLUMNS)
        if error is not None and error not in frame.columns:
            error = None
        half = (self.INTERVAL_Z * np.abs(_finite(frame[error]))
                if error else np.full(len(frame), np.nan))
        if significance_column == NO_SIGNIFICANCE:
            significance = None
        elif significance_column:
            significance = (significance_column
                            if significance_column in frame.columns else None)
        else:
            significance = _first_column(frame, CORRECTED_P_COLUMNS)
        cut = (_finite(frame[significance]) if significance
               else np.full(len(frame), np.nan))
        self._effects, self._half = effects, half
        self._significance, self._significance_name = cut, significance or ""

        key = key_column or ("feature" if "feature" in frame.columns else
                             label_column)
        self.set_keys(frame[key] if key in frame.columns else None)

        # numpy puts NaN last under an ascending sort whatever its sign, so a
        # coefficient that did not converge ranks below every one that did
        # rather than at the top of the list.
        order = np.argsort(-np.abs(effects), kind="stable")
        ranks = np.arange(len(order), dtype="float64")
        x = effects[order]
        widths = half[order]
        called = cut[order] <= alpha
        # 0 grey, 1 up, 2 down -- as a code array rather than a list of
        # colours, so the intervals can be grouped by ink in three passes
        # instead of one PlotCurveItem per coefficient.
        code = np.where(called, np.where(x > 0, 1, 2), 0)
        inks = (QColor(self.GREY), QColor(self.UP_INK), QColor(self.DOWN_INK))

        # THE INTERVALS FIRST, so the dots sit on top of them. One curve per
        # ink with `connect="pairs"` -- 1,213 disconnected segments in three
        # items rather than 1,213 items, which is the difference between a
        # plot that opens and one that hangs.
        usable = np.isfinite(x) & np.isfinite(widths) & (widths > 0)
        for value, ink in enumerate(inks):
            picked = np.nonzero(usable & (code == value))[0]
            if not len(picked):
                continue
            xs = np.empty(len(picked) * 2)
            xs[0::2] = x[picked] - widths[picked]
            xs[1::2] = x[picked] + widths[picked]
            self.plot.addItem(pg.PlotCurveItem(
                x=xs, y=np.repeat(ranks[picked], 2), connect="pairs",
                pen=pg.mkPen(ink, width=1.0)))

        self.add_scatter(x, ranks, size=8.0, rows=order,
                         colours=[inks[int(c)] for c in code])
        self.add_line(x=0.0, colour=REFERENCE, width=1.0)

        # THE NAMES ARE Y-TICKS HERE AND ANNOTATIONS IN THE SAVED PANEL, and
        # the difference is deliberate. A tick label is drawn outside the axes,
        # so on a sheet a long gene id reaches into the cell to its left --
        # which is why the static panel puts them inside. A tab has no
        # neighbouring cell, and a tick is the axis a reader can then zoom.
        names = self._label_series(frame, label_column)
        self._names = names
        shown = min(len(order), self.LABELLED)
        self.plot.getAxis("left").setTicks(
            [[(float(row), str(names[int(order[row])])[:28])
              for row in range(shown)]])
        if shown:
            self.plot.setYRange(-0.6, shown - 0.4, padding=0.02)

        plotted = int(np.sum(np.isfinite(x)))
        # COUNTED OVER WHAT IS ON THE PICTURE. `called` is computed over every
        # row, and a coefficient that did not come out can still carry a
        # q-value -- so counting one would put a number in the status line
        # that no reader can reach by counting the coloured dots.
        self.set_status(self._sentence(plotted, len(order), shown, error,
                                       significance,
                                       int(np.sum(called & np.isfinite(x))),
                                       alpha, untested))
        if self._selected_key is not None:
            self.highlight_key(self._selected_key)
        return plotted

    @staticmethod
    def _label_series(frame, label_column: str):
        """A readable name per row, from the ONE place that knows the rule.

        :func:`spacr.figures.panels.label_series` coalesces ``gene`` and
        ``grna`` -- each of which is blank on the other's rows -- and strips
        the design-matrix boilerplate off ``feature``. Re-deriving that here
        would give the tab and the saved panel two different names for the
        same coefficient, and the reader would have no way to tell which.
        """
        try:
            from ...figures.panels import label_series

            return label_series(frame).to_numpy()
        except Exception:              # figures unavailable
            if label_column in getattr(frame, "columns", ()):
                return frame[label_column].astype(str).to_numpy()
            return np.array([str(i) for i in range(len(frame))])

    def _sentence(self, plotted, total, named, error, significance, called,
                  alpha, untested) -> str:
        """What this plot has to say about itself, in one place.

        Every branch is a real fact about the TABLE rather than a fallback:
        no standard error, no corrected p, more rows than names, nuisance
        terms removed. Each one changes how the picture should be read, so
        each is said rather than left for the reader to notice.
        """
        note = f"{plotted} coefficients, ranked by the size of the effect."
        note += (f" The bar through each dot is a "
                 f"{self.INTERVAL_Z:g}-standard-error interval from "
                 f"“{error}”." if error else
                 " This table carries no standard error, so there are no "
                 "intervals: the dots are point estimates drawn without the "
                 "uncertainty that decides whether to believe them.")
        if significance:
            note += (f" {called} called at {significance} ≤ {alpha:g}; "
                     f"everything else is grey.")
        else:
            note += (" Nothing is coloured: this table has no corrected "
                     "p-value, and calling hits off an uncorrected p across "
                     f"{total} tests is the error this panel exists to make "
                     "visible.")
        if plotted > named:
            note += (f" The strongest {named} are named on the axis; all "
                     f"{plotted} finite coefficients are drawn, and Reset "
                     f"view reaches them.")
        missing = total - plotted
        if missing:
            note += (f" {missing} coefficient"
                     f"{'s' if missing != 1 else ''} with a blank or "
                     f"non-finite effect {'are' if missing != 1 else 'is'} "
                     f"not drawn.")
        if untested:
            note += (f" {untested} nuisance "
                     f"term{'s' if untested != 1 else ''} not ranked (fitted "
                     f"as covariates, not as hypotheses).")
        if self._has_usable_keys():
            note += " Click a dot for its coefficient."
        return note

    def _detail(self, index: int) -> str:
        parts = []
        if index < len(self._effects) and np.isfinite(self._effects[index]):
            value = float(self._effects[index])
            if index < len(self._half) and np.isfinite(self._half[index]):
                half = float(self._half[index])
                parts.append(f"effect = {value:.3g} "
                             f"[{value - half:.3g}, {value + half:.3g}]")
            else:
                parts.append(f"effect = {value:.3g}")
        if (self._significance_name and index < len(self._significance)
                and np.isfinite(self._significance[index])):
            parts.append(f"{self._significance_name} = "
                         f"{self._significance[index]:.3g}")
        return "   ".join(parts)


class BinnedPlot(FastPlot):
    """A histogram whose bars remember which rows they were built from.

    Two panels here are histograms of a coefficient table -- the p-values and
    the effects -- and both need the three things a scatter gets for free:
    which rows are in which bar, a click that lands in the bar under the
    cursor, and an outline marking the bar a row selected elsewhere falls in.
    That machinery is subtle -- half-open bins with the last one closed, a
    row-to-bar index built without a per-coefficient Python loop -- and it is
    exactly the kind of thing that drifts when it is written twice.

    A BAR IS NOT A POINT, which is the rule this whole class is shaped by. A
    bar holding a hundred coefficients cannot select one of them, and picking
    the first, the strongest or the nearest would be a guess dressed up as an
    answer -- the same mistake as joining on a position. So a bar of many
    hands the whole set over for the table to narrow to, and only a bar
    holding exactly one row selects it like any other mark.
    """

    #: What one observation IS, for the sentence a clicked bar writes. Named
    #: rather than hardcoded because "p 0.02 to 0.04" and "effect -1.2 to
    #: -0.9" are the same sentence about two different quantities.
    QUANTITY = "value"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._edges: Optional[np.ndarray] = None
        self._counts: Optional[np.ndarray] = None
        self._bin_rows: list = []
        self._row_bin: np.ndarray = np.empty(0, dtype="int64")
        self._values: np.ndarray = np.empty(0)
        # A BAR IS NOT A POINT, so there is no sigClicked to connect to. The
        # scene reports where the user pressed and the bin is worked out from
        # the x coordinate, which is also the only definition of "which bar"
        # that stays right when the axis is zoomed.
        self.plot.scene().sigMouseClicked.connect(self._on_scene_clicked)

    def _fill_bins(self, values, bins: int, span=None) -> np.ndarray:
        """Histogram ``values`` and record which rows each bar holds.

        :param values: one observation per FRAME ROW, blanks included, so a
            row's position here is its position in the caller's table.
        :param bins: how many bars.
        :param span: ``(low, high)`` to bin over, or None for the data's own
            range. The p-value histogram pins it to ``(0, 1)`` because the
            axis means something there; a distribution of effects has no such
            fixed range and pinning one would invent it.
        :returns: the finite values, or an empty array.

        WHICH ROWS ARE IN WHICH BAR, worked out the same way np.histogram
        decided the counts -- half-open bins with the last one closed at the
        top edge -- so the number a bar draws and the number of rows it hands
        back are the same number. A value outside the span is in no bar at
        all, which is exactly what ``np.histogram``'s ``range`` did with it.

        Vectorised, because a bar chart must not pay a per-COEFFICIENT Python
        loop to be drawn: that is the cost this whole module exists to avoid,
        and a screen has as many observations as it has coefficients.
        """
        held_all = _finite(values)
        self._values = held_all
        self._edges = self._counts = None
        self._bin_rows = []
        self._row_bin = np.empty(0, dtype="int64")
        rows = np.nonzero(~np.isnan(held_all))[0]
        if not len(rows):
            return np.empty(0)
        held = held_all[rows]
        counts, edges = np.histogram(held, bins=bins, range=span)
        inside = (held >= edges[0]) & (held <= edges[-1])
        placed = np.clip(np.searchsorted(edges, held, side="right") - 1,
                         0, bins - 1)
        self._edges, self._counts = edges, counts
        members, in_bin = rows[inside], placed[inside]
        order = np.argsort(in_bin, kind="stable")
        cuts = np.cumsum(np.bincount(in_bin, minlength=bins))[:-1]
        self._bin_rows = list(np.split(members[order], cuts))
        # Row -> its bar, as a dense array rather than a dict of thousands.
        self._row_bin = np.full(len(held_all), -1, dtype="int64")
        self._row_bin[members] = in_bin
        return held

    def add_bars(self, brush=None):
        """Put the bars from the last :meth:`_fill_bins` onto the plot."""
        bars = pg.BarGraphItem(
            x0=self._edges[:-1], x1=self._edges[1:], height=self._counts,
            brush=brush if brush is not None else pg.mkBrush(colour_for(0, 190)),
            pen=pg.mkPen(None))
        self.plot.addItem(bars)
        return bars

    # ------------------------------------------------------------- clicking

    def bin_at(self, x) -> Optional[int]:
        """The bar under data coordinate ``x``, or ``None`` beyond the axis."""
        if self._edges is None or not len(self._bin_rows):
            return None
        if x < self._edges[0] or x > self._edges[-1]:
            return None
        index = int(np.searchsorted(self._edges, x, side="right") - 1)
        return int(np.clip(index, 0, len(self._bin_rows) - 1))

    def keys_in_bin(self, index: int) -> list:
        """Every identifier the bar at ``index`` was built from."""
        if not 0 <= int(index) < len(self._bin_rows):
            return []
        found = (self.key_for_row(int(row)) for row in self._bin_rows[index])
        return [key for key in found if key is not None]

    def select_bin(self, index: int) -> list:
        """Answer a click on one bar. Returns the identifiers inside it.

        A BAR HOLDING A HUNDRED COEFFICIENTS CANNOT SELECT ONE OF THEM, and
        picking the first, the strongest or the nearest would be a guess
        dressed up as an answer -- the same mistake as joining on a position.
        So the honest split: a bar that holds exactly one row selects it like
        any other point, and a bar that holds more says what it holds and
        hands the whole set over for the table to narrow to.
        """
        keys = self.keys_in_bin(index)
        if self._edges is None or not 0 <= int(index) < len(self._bin_rows):
            return []
        low, high = float(self._edges[index]), float(self._edges[index + 1])
        count = len(self._bin_rows[index])
        span = f"{self.QUANTITY} {low:.3g} to {high:.3g}"
        if not count:
            self.set_status_note(f"{span}: empty.")
            return []
        if len(keys) == 1:
            self.highlight_key(keys[0])
            self.set_status_note(f"{span}: {keys[0]}")
            self.key_selected.emit(keys[0])
        else:
            self.highlight_bin(index)
            named = f", {len(keys)} of them named" if 0 < len(keys) < count \
                else ""
            self.set_status_note(
                f"{span}: {count} coefficient{'s' if count != 1 else ''}"
                f"{named}. A bar is not one point, so this selects the set "
                f"rather than guessing which of them you meant.")
        if keys:
            self.keys_selected.emit(list(keys))
        return keys

    def _on_scene_clicked(self, event) -> None:
        """A press anywhere on the plot, mapped to the bar under it."""
        try:
            if event.button() != Qt.LeftButton:
                return
            position = event.scenePos()
            item = self.plot.plotItem
            if not item.sceneBoundingRect().contains(position):
                return
            point = item.vb.mapSceneToView(position)
        except Exception:          # no viewbox to map into
            return
        # THE BAR IS FOUND IN DATA UNITS. `mapSceneToView` answers in DRAWN
        # units, which are log10 of the data while the x axis is logged, and
        # a bin looked up with those lands in the wrong bar or in none.
        index = self.bin_at(self._to_data(point.x(), "x"))
        if index is not None:
            self.select_bin(index)

    def highlight_bin(self, index: int) -> bool:
        """Outline one bar. The histogram's answer to ringing a point."""
        if self._edges is None or self._counts is None:
            return False
        if not 0 <= int(index) < len(self._counts):
            return False
        if self._highlight is not None:
            try:
                self.plot.removeItem(self._highlight)
            except Exception:           # already gone
                pass
        # An OUTLINE, not a refill: the same reason the scatter marker is an
        # open ring. A solid bar in the highlight colour would hide how tall
        # it is against its neighbours, which is the only thing the panel is
        # for.
        self._highlight = pg.BarGraphItem(
            x0=[self._edges[index]], x1=[self._edges[index + 1]],
            height=[self._counts[index]], brush=pg.mkBrush(None),
            pen=pg.mkPen(QColor(self._foreground), width=2.0))
        self._highlight.setZValue(50)
        self.plot.addItem(self._highlight)
        return True

    def _draw_marker(self, row: int) -> bool:
        """A row selected elsewhere marks THE BAR IT FALLS IN.

        Not a ring floating over the bars: this plot never drew that row as a
        mark of its own, and inventing one would put a point on a histogram
        where there is no point. The bar is where the coefficient actually is.
        """
        row = int(row)
        if not 0 <= row < len(self._row_bin):
            return False
        index = int(self._row_bin[row])
        if index < 0:
            return False
        return self.highlight_bin(index)

    def _detail(self, index: int) -> str:
        if index < len(self._values) and np.isfinite(self._values[index]):
            return f"{self.QUANTITY} = {self._values[index]:.3g}"
        return ""


class PValueHistogram(BinnedPlot):
    """The single most informative check that a correction means anything.

    Under the null, p-values are uniform. A histogram that is flat with a spike
    at zero is a screen with real hits in it; one that slopes, or piles up near
    one, says the model is misspecified and every q-value downstream of it is
    decoration.
    """

    QUANTITY = "p"

    def __init__(self, parent=None):
        super().__init__(title="p-value distribution", x_label="p",
                         y_label="count", parent=parent)

    def set_p_values(self, values, bins: int = 50, *, keys=None):
        """Draw the histogram. Returns the number of usable p-values.

        :param values: one p-value per frame row, blanks included.
        :param bins: how many bars across ``[0, 1]``.
        :param keys: one identifier per element of ``values``, in frame order.
            Given them, clicking a bar names the coefficients inside it.
        """
        self._reset_scene()
        self.set_keys(keys)
        # PINNED TO [0, 1], because that is what a p-value's axis MEANS. A
        # histogram of p over the observed range would put its left edge at
        # the smallest p in the screen, and the spike at zero -- the whole
        # signal this panel exists to show -- would then be the first bar of
        # every screen, calibrated or not.
        held = self._fill_bins(values, bins, span=(0.0, 1.0))
        if not len(held):
            self.set_status("No p-values.")
            return 0
        self.add_bars()
        expected = len(held) / bins
        self.add_line(y=expected, colour="#C44E52", label="uniform")

        excess = max(int(self._counts[0] - expected), 0)
        note = (f"{len(held)} p-values. The flat line is what a screen with "
                f"no signal would give; the first bin holds {excess} more "
                f"than that.")
        if self._has_usable_keys():
            note += " Click a bar for what is in it."
        self.set_status(note)
        if self._selected_key is not None:
            self.highlight_key(self._selected_key)
        return len(held)


class EffectDistribution(BinnedPlot):
    """Where the screen's effects sit, and how wide the null under them is.

    The interactive twin of
    :func:`spacr.figures.panels.effect_distribution`. The volcano says which
    coefficients are extreme; this says what "extreme" is worth on THIS
    screen, which is the number a reader needs before they believe any of
    them. A screen whose effects are a tight bell with nothing in the tails
    has no hits however small its p-values are.

    σ IS A MAD, NOT A STANDARD DEVIATION, and that is the point of the panel
    rather than a detail of it: a standard deviation is inflated by exactly
    the outliers a screen exists to find, so a cut measured from one is pulled
    outwards by the hits and then fails to call them. The median absolute
    deviation is not, and ×1.4826 makes it the consistent estimator for a
    normal -- the same statistic
    :func:`spacr.figures.panels.control_threshold` measures the effect-size
    cut from, so the dashed lines here and the lines on the volcano cannot
    disagree about where three sigmas is.
    """

    QUANTITY = "effect"

    #: How many MAD-sigmas out the dashed lines sit. Three, matching the saved
    #: panel and :func:`spacr.thresholds`' own default multiplier.
    SIGMAS = 3.0

    #: MAD x this is the consistent estimate of sigma for a normal.
    MAD_TO_SIGMA = 1.4826

    def __init__(self, parent=None):
        super().__init__(title="Effect distribution", x_label="effect size",
                         y_label="coefficients", parent=parent)

    def set_effects(self, values, bins: int = 50, *, keys=None,
                    untested: int = 0):
        """Draw the histogram. Returns the number of usable effects.

        :param values: one fitted effect per frame row, blanks included.
        :param bins: how many bars across the data's own range.
        :param keys: one identifier per element of ``values``, in frame order.
            Given them, clicking a bar names the coefficients inside it.
        :param untested: how many nuisance terms the CALLER left out, so the
            plot can say so. It is the caller that knows spaCR's term grammar,
            which is why the drop is not done here.

            IT IS THE FAMILY, NOT THE AXIS, and the measurement says so: on
            the TSG101 screen σ (MAD) is 0.229228 over the tested family and
            0.229036 with the intercept added, a difference of 0.08%. Dropping
            it does not visibly move this picture -- it makes the picture be
            OF something, namely the 1,212 coefficients the q-values describe,
            which is the same family
            :func:`spacr.figures.panels.effect_distribution` draws and the
            same one the effect-size cut is measured from.

        THE RANGE IS THE DATA'S OWN, unlike the p-value histogram's. An effect
        size has no fixed domain, and pinning one would invent a scale the fit
        never produced.
        """
        self._reset_scene()
        self.set_keys(keys)
        held = self._fill_bins(values, bins)
        if not len(held):
            self.set_status(
                "No fitted effects to plot: every coefficient in this table "
                "is blank or non-finite.")
            return 0
        self.add_bars(brush=pg.mkBrush(QColor(FILL)))
        self.add_line(x=0.0, colour=REFERENCE, width=1.0)

        sigma = float(np.median(np.abs(held - np.median(held)))
                      * self.MAD_TO_SIGMA)
        note = f"{len(held)} coefficients."
        if sigma > 0:
            cut = self.SIGMAS * sigma
            for sign in (-1, 1):
                self.add_line(x=sign * cut, colour=REFERENCE,
                              label=f"{self.SIGMAS:g}σ" if sign > 0 else "")
            beyond = int(np.sum(np.abs(held - np.median(held)) > cut))
            note += (f" σ (MAD) = {sigma:.3g}, which the outliers a screen "
                     f"exists to find do not inflate the way a standard "
                     f"deviation would. The dashed lines are ±{self.SIGMAS:g}σ "
                     f"and {beyond} coefficient{'s' if beyond != 1 else ''} "
                     f"lie outside them.")
        else:
            note += (" Every finite effect here is the same value, so there "
                     "is no spread to measure a σ from and no ±σ lines are "
                     "drawn.")
        if untested:
            note += (f" {untested} nuisance "
                     f"term{'s' if untested != 1 else ''} not counted: they "
                     f"are covariates, so they are outside the family the "
                     f"q-values and the effect-size cut describe.")
        if self._has_usable_keys():
            note += " Click a bar for what is in it."
        self.set_status(note)
        if self._selected_key is not None:
            self.highlight_key(self._selected_key)
        return len(held)


class QQPlot(FastPlot):
    """Observed against expected quantiles -- is the null calibrated?

    Points on the diagonal mean the test is behaving. A curve that lifts off it
    early means inflation: the design is confounded, and the hits at the top of
    the volcano are partly an artefact of that rather than biology.
    """

    def __init__(self, parent=None):
        super().__init__(title="p-value Q-Q", x_label="expected -log10(p)",
                         y_label="observed -log10(p)", parent=parent)
        self._p: np.ndarray = np.empty(0)

    def set_p_values(self, values, *, keys=None):
        """Draw the Q-Q. Returns the number of usable tests.

        :param values: one p-value per frame row, including missing entries;
            only finite positive values are ranked and drawn.
        :param keys: one identifier per element of ``values``, IN THE ORDER
            THEY WERE HANDED IN -- i.e. in frame order, including the ones
            with no usable p-value. Given them, every point is clickable and
            selects the coefficient it was computed from.

        THE SORT IS THE TRAP. A Q-Q is ranked by p, so the nth drawn point is
        the nth SMALLEST p-value and almost never the nth row of the table.
        The rows are therefore carried through the sort explicitly (``rows=``)
        rather than re-derived from the drawing order, which is the mistake
        that lights up the wrong guide and looks entirely correct doing it.
        """
        self._reset_scene()
        self.set_keys(keys)
        p = _finite(values)
        self._p = p
        # Frame rows, kept alongside their p-values through the sort.
        rows = np.nonzero(~np.isnan(p) & (p > 0))[0]
        if not len(rows):
            self.set_status("No usable p-values.")
            return 0
        rows = rows[np.argsort(p[rows], kind="stable")]
        n = len(rows)
        expected = -np.log10((np.arange(1, n + 1) - 0.5) / n)
        observed = -np.log10(p[rows])
        self.add_scatter(expected, observed, size=6, rows=rows)
        top = float(max(expected.max(), observed.max()))
        self.plot.plot([0, top], [0, top],
                       pen=pg.mkPen("#C44E52", width=1.5, style=Qt.DashLine))
        # Genomic inflation: the ratio at the median. 1.0 is calibrated.
        chi = np.median(observed) / np.median(expected) if np.median(expected) else float("nan")
        note = (f"{n} tests. Inflation at the median is {chi:.2f} "
                f"(1.00 is calibrated; well above it means the null is not "
                f"flat).")
        if self._has_usable_keys():
            note += " Click a point for its coefficient."
        self.set_status(note)
        # A SELECTION SURVIVES A REDRAW, here for the same reason it does on
        # the volcano: the user picked a guide, and reloading or recolouring
        # is not them un-picking it.
        if self._selected_key is not None:
            self.highlight_key(self._selected_key)
        return n

    def _detail(self, index: int) -> str:
        if index < len(self._p) and np.isfinite(self._p[index]):
            return f"p = {self._p[index]:.3g}"
        return ""


class ResidualPlot(FastPlot):
    """Residual against fitted -- the check for a mis-specified mean.

    A horizontal band is what a well-specified model gives. A funnel means the
    variance grows with the fit and the standard errors are wrong, which is a
    p-value problem rather than a cosmetic one.
    """

    def __init__(self, parent=None):
        super().__init__(title="Residuals vs fitted", x_label="fitted",
                         y_label="residual", parent=parent)
        self._residual_data = None
        self.offer_smoothers(self._choose_smoother, chosen="lowess")

    def _choose_smoother(self, method: str) -> None:
        """Redraw with a different diagnostic curve, or with none."""
        self._smoother_chosen = str(method or "")
        if self._residual_data is not None:
            self.set_residuals(*self._residual_data)

    def set_residuals(self, fitted, residuals, labels: Sequence[str] = ()):
        self._residual_data = (fitted, residuals, labels)
        self._reset_scene()
        f, r = _finite(fitted), _finite(residuals)
        if not len(f):
            self.set_status("No residuals.")
            return 0
        self.add_scatter(f, r, size=6, labels=labels)
        self.add_line(y=0.0, colour="#C44E52")
        good = ~(np.isnan(f) | np.isnan(r))
        if good.sum() > 2:
            # A crude trend line: if this is not flat, the mean is wrong.
            slope, intercept = np.polyfit(f[good], r[good], 1)
            xs = np.array([np.nanmin(f), np.nanmax(f)])
            self.plot.plot(xs, slope * xs + intercept,
                           pen=pg.mkPen("#DD8452", width=1.5))
            said = (f"{int(good.sum())} residuals. Trend slope "
                    f"{slope:+.3g} -- far from zero means the mean model is "
                    f"missing something.")
            # THE STRAIGHT LINE CAN BE FLAT WHILE THE RESIDUALS BEND, which
            # is the case a slope cannot report and a smoother can.
            curve = self.add_smoother(f[good], r[good],
                                      method=self._smoother_chosen)
            self.set_status(f"{said} {curve}" if curve else said)
        return int(good.sum())


class ScaleLocationPlot(FastPlot):
    """Plot the square root of absolute standardised residual against fitted.

    The interactive twin of :func:`spacr.regression_qc._panel_scale_location`,
    used as the variance-homogeneity panel.
    A residual-vs-fitted plot shows the mean and the variance at once and a
    reader has to separate them by eye; taking the square root of the absolute
    standardised residual removes the sign, so what is left is only the
    spread. A rising trend means the standard errors -- and therefore every
    p-value on the volcano -- are wrong in a direction that depends on the
    fitted value.

    Drawn on the STANDARDISED residual, so it is empty for a model class that
    has no error scale (quantile regression, a hinge classifier). That is a
    real answer and is said out loud rather than drawn from ``y - fitted`` and
    labelled as though it were the same quantity.
    """

    def __init__(self, parent=None):
        super().__init__(title="Scale-location", x_label="fitted",
                         y_label="sqrt(|standardised residual|)",
                         parent=parent)
        self._scale_data = None
        self.offer_smoothers(self._choose_smoother, chosen="lowess")

    def _choose_smoother(self, method: str) -> None:
        """Redraw with a different diagnostic curve, or with none."""
        self._smoother_chosen = str(method or "")
        if self._scale_data is not None:
            self.set_scale_location(*self._scale_data[:3],
                                    reason=self._scale_data[3])

    def set_scale_location(self, fitted, std_resid,
                           labels: Sequence[str] = (), reason: str = ""):
        """Draw it. Returns the number of wells plotted.

        :param fitted: one fitted response per well, aligned with
            ``std_resid`` and ``labels``; pairs containing a non-finite value
            are not drawn.
        :param std_resid: ``RegressionQCContext.std_resid``. All-NaN when the
            model class has no error scale -- see
            :func:`spacr.regression_qc.resolve_residual_standardisation`.
        :param reason: what to say when there is no standardised residual;
            pass ``ctx.standardisation.reason``.
        """
        self._scale_data = (fitted, std_resid, labels, reason)
        self._reset_scene()
        f, s = _finite(fitted), _finite(std_resid)
        good = ~(np.isnan(f) | np.isnan(s))
        if not good.any():
            self.set_status(
                f"No standardised residual for this fit, so the variance "
                f"cannot be checked: {reason}" if reason else
                "No standardised residuals.")
            return 0
        root = np.sqrt(np.abs(s))
        self.add_scatter(f, root, size=6, labels=labels)
        slope, intercept = np.polyfit(f[good], root[good], 1)
        xs = np.array([float(np.nanmin(f)), float(np.nanmax(f))])
        self.plot.plot(xs, slope * xs + intercept,
                       pen=pg.mkPen("#DD8452", width=1.5))
        said = (f"{int(good.sum())} wells. Trend slope {slope:+.3g} -- a "
                f"flat line is constant variance; a rising one means the "
                f"standard errors, and so every p-value on the volcano, "
                f"depend on the fitted value.")
        curve = self.add_smoother(f[good], root[good],
                                  method=self._smoother_chosen)
        self.set_status(f"{said} {curve}" if curve else said)
        return int(good.sum())


class InfluencePlot(FastPlot):
    """Leverage against standardised residual, with Cook's distance on top.

    The interactive twin of :func:`spacr.regression_qc._panel_influence`. The
    question it answers is the one a screen cannot answer from the volcano: is
    a hit the shape of the data, or the shape of ONE WELL? A well far to the
    right has an unusual combination of guides; a well far up or down is
    poorly predicted; a well that is both is one whose removal moves the
    coefficients, and Cook's distance is the product that says so.

    The wells past the 4/n screening rule are the only ones coloured, which is
    the house rule -- everything else is grey, because the sentence here is
    "these ones are worth going back to the microscope for".
    """

    #: Genes and wells the fit is not resting on.
    GREY = MUTED
    #: The argument: this well is moving the coefficients on its own.
    INFLUENTIAL = HIGHLIGHT

    def __init__(self, parent=None):
        super().__init__(title="Leverage vs standardised residual",
                         x_label="leverage", y_label="standardised residual",
                         parent=parent)
        self._cooks: np.ndarray = np.empty(0)

    def set_influence(self, leverage, std_resid, cooks,
                      labels: Sequence[str] = (), n_params: int = 0,
                      reason: str = ""):
        """Draw it. Returns the number of wells plotted.

        Every array comes from :mod:`spacr.regression_qc` -- ``ctx.leverage``,
        ``ctx.std_resid`` and :func:`spacr.regression_qc.cooks_distance` --
        rather than being recomputed here, so the live panel and the saved
        report cannot name different wells as influential.
        """
        self._reset_scene()
        h, s, d = _finite(leverage), _finite(std_resid), _finite(cooks)
        self._cooks = d
        good = ~(np.isnan(h) | np.isnan(s))
        if not good.any():
            self.set_status(
                f"No standardised residual for this fit, so influence cannot "
                f"be measured: {reason}" if reason else
                "No influence measures.")
            return 0
        n = int(good.sum())
        # 4/n, the conventional screening rule and the one the saved report
        # draws. The stricter D > 1 almost never fires on a few hundred wells,
        # which makes it a rule that separates nothing.
        cut = 4.0 / n
        flagged = good & (d > cut)
        rows = np.arange(len(h))
        for mask, colour, size in ((good & ~flagged, self.GREY, 6),
                                   (flagged, self.INFLUENTIAL, 9)):
            if not mask.any():
                continue
            picked = rows[mask]
            self.add_scatter(h[picked], s[picked], size=size, rows=picked,
                             labels=labels,
                             colours=[QColor(colour)] * len(picked))
        self.add_line(y=0.0, colour=self.GREY)
        if n_params:
            # 2p/n: the standard "this row has an unusual design" rule.
            self.add_line(x=2.0 * int(n_params) / n, colour="#DD8452",
                          label="2p/n")
        count = int(flagged.sum())
        if not count:
            self.set_status(
                f"{n} wells, none past Cook's D > 4/n ({cut:.3g}): no single "
                f"well is carrying the fit.")
        elif count == 1:
            self.set_status(
                f"{n} wells; 1 past Cook's D > 4/n ({cut:.3g}), so that well "
                f"is moving the coefficients on its own.")
        else:
            self.set_status(
                f"{n} wells; {count} past Cook's D > 4/n ({cut:.3g}), so "
                f"those wells are moving the coefficients on their own.")
        return n

    def _detail(self, index: int) -> str:
        if index < len(self._cooks) and np.isfinite(self._cooks[index]):
            return f"Cook's D = {self._cooks[index]:.3g}"
        return ""


class GroupedPlot(FastPlot):
    """Base class for live plots with a categorical x-axis.

    The right-click menu exposes every mark in :data:`MARK_TYPES`. Subclasses
    retain their source groups so switching marks redraws the same observations
    and uses :func:`mark_advice` to explain unsuitable small-sample summaries.

    Attributes
    ----------
    mark_changed : PySide6.QtCore.Signal
        Emitted with the new mark key after a successful change.
    """

    mark_changed = Signal(str)

    #: Initial group mark. Jitter preserves individual observations and keeps
    #: existing control and guide-support views unchanged.
    DEFAULT_MARK = "jitter"
    #: How wide one group's mark is, in x units.
    MARK_WIDTH = 0.6
    #: Which summary the line across a points/jitter group is; see
    #: :meth:`FastPlot.add_group_mark`.
    MARK_CENTRE = "mean"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mark = self.DEFAULT_MARK
        self._offer_marks()

    def _offer_marks(self) -> None:
        """(Re)build the menu so the tick sits on the current mark."""
        self.offer_marks([
            (label, (lambda _checked=False, key=name: self.set_mark(key)),
             name == self._mark)
            for name, label in MARK_TYPES])

    def mark(self) -> str:
        """Which mark the groups are currently drawn with."""
        return self._mark

    def set_mark(self, kind: str) -> bool:
        """Draw the groups as ``kind``. Returns True if the mark changed.

        :raises ValueError: on a mark this module cannot draw. Loudly, because
            the only callers are this class's own menu and a test -- a silent
            fallback would make a typo look like a working option.
        """
        known = dict(MARK_TYPES)
        if kind not in known:
            raise ValueError(
                f"unknown mark {kind!r}; known marks: {', '.join(known)}")
        changed = kind != self._mark
        self._mark = kind
        # The menu is rebuilt from scratch on every right-click, so the tick
        # only moves if the stored list moves with it.
        self._offer_marks()
        self.redraw()
        if changed:
            self.mark_changed.emit(kind)
        return changed

    def redraw(self) -> None:
        """Draw the last data handed in, with whatever the mark now is.

        Subclasses re-run their own ``set_*`` from what they stored. Nothing
        is recovered from the picture -- the arrays are kept -- so switching
        marks cannot show a different set of observations than the mark before
        it showed.
        """
        raise NotImplementedError

    def group_sizes(self) -> list:
        """Observations per group, for :func:`mark_advice`."""
        return []

    def mark_note(self) -> str:
        """The sentence about the CURRENT mark, or ``""``.

        Two things, because a user who picks "bar" has done two at once: they
        have chosen a mark that may misrepresent the spread, and they have
        given up the ability to click a guide -- one rectangle stands for
        forty-one rows and cannot honestly select one of them.
        """
        parts = []
        advice = mark_advice(self._mark, self.group_sizes())
        if advice:
            parts.append(advice)
        if self._mark in ("box", "violin", "bar", "line") and self._has_usable_keys():
            parts.append(
                "Only the outliers are still individual points, so only they "
                "can be clicked; switch back to points or jitter to pick any "
                "of the rest." if self._mark == "box" else
                f"A {self._mark} stands for many rows at once, so nothing on "
                f"it can be clicked; switch back to points or jitter to pick "
                f"a row.")
        return " ".join(parts)

    def _join_group_line(self, points) -> None:
        """Connect ordered ``(x, centre)`` pairs for the line mark."""
        if self._mark != "line" or len(points) < 2:
            return
        self.plot.plot(
            [x for x, _centre in points],
            [centre for _x, centre in points],
            pen=pg.mkPen(QColor(self._foreground), width=2),
        )


class ControlSeparation(GroupedPlot):
    """How far apart the positive and negative controls sit.

    This is the assay window. If the controls do not separate, nothing further
    down the pipeline can be trusted, and it is worth seeing before the volcano
    rather than after arguing about a hit list.
    """

    #: The medians are what the status line quotes and what the reader
    #: compares, so the line drawn across a points/jitter group has to be the
    #: same statistic -- see :meth:`FastPlot.add_group_mark`.
    MARK_CENTRE = "median"
    #: Narrower than the default: three groups a unit apart, and a 0.6-wide
    #: jitter puts a negative control close enough to the positives to be read
    #: as one of them.
    MARK_WIDTH = 0.35

    def __init__(self, parent=None):
        super().__init__(title="Control separation", x_label="",
                         y_label="effect", parent=parent)
        self._effects: np.ndarray = np.empty(0)
        #: ``(start, stop, group name)`` per group over the flat row space.
        self._spans: list = []
        #: The last groups and keys, so a change of mark redraws the SAME
        #: observations rather than whatever the caller happens to hand in
        #: next. See :meth:`GroupedPlot.redraw`.
        self._groups: dict = {}
        self._group_keys: Optional[dict] = None

    def redraw(self) -> None:
        """Draw the stored groups again with the current mark."""
        if self._groups:
            self.set_groups(self._groups, keys=self._group_keys)

    def group_sizes(self) -> list:
        return [int(np.sum(~np.isnan(_finite(values))))
                for values in self._groups.values()]

    def set_groups(self, groups: dict, *, keys: Optional[dict] = None):
        """Draw the groups. Returns the number of points plotted.

        :param groups: ``{'negative': array, 'positive': array, ...}``
        :param keys: ``{'negative': identifiers, ...}``, one identifier per
            value of the SAME group, in the same order. Given them, every dot
            is clickable and selects the coefficient behind it.

        THE GROUPS ARE THE SECOND FORM OF THE SORT TRAP. These arrays are
        slices of the table taken by condition, so a dot's position within
        its own group is not its row -- and the negative controls are drawn
        before the screen, so it is not its position on the plot either. Rows
        are therefore laid out in one flat sequence up front and carried into
        each scatter, rather than being inferred from the drawing order.
        """
        self._reset_scene()
        self._spans = []
        self._groups = dict(groups or {})
        self._group_keys = keys
        if not groups:
            self.set_keys(())
            self._effects = np.empty(0)
            self.set_status("No controls identified.")
            return 0

        # One flat row space over every group, so a key means the same thing
        # whichever group it came from.
        flat_keys: list = []
        columns: list = []
        base: dict = {}
        for name, values in groups.items():
            v = _finite(values)
            base[name] = len(flat_keys)
            given = None if keys is None else keys.get(name)
            if given is None:
                flat_keys.extend([None] * len(v))
            else:
                given = list(given)
                # A short or long key list is a caller bug that would silently
                # shift every row after it; pad rather than mis-join.
                if len(given) != len(v):
                    given = [given[i] if i < len(given) else None
                             for i in range(len(v))]
                flat_keys.extend(given)
            columns.append(v)
            # WHICH GROUP A ROW IS IN, AS A SPAN RATHER THAN AS 1,186 DICT
            # ENTRIES. Only the clicked point is ever asked, and the module's
            # whole performance argument is that nothing is computed per point
            # before a click. Three tuples answer it in a scan of three.
            self._spans.append((base[name], base[name] + len(v), name))
        self.set_keys(flat_keys if keys is not None else ())
        self._effects = np.concatenate(columns) if columns else np.empty(0)

        summary, total = [], 0
        line_points: list = []
        for position, (name, values) in enumerate(groups.items()):
            v = _finite(values)
            finite = np.nonzero(~np.isnan(v))[0]
            if not len(finite):
                continue
            total += len(finite)
            # THE MEDIAN IS THE SENTENCE OF THIS PANEL -- whether the classes
            # separate is read off those lines -- so `MARK_CENTRE` keeps the
            # summary line on the median whatever mark the user picks, and the
            # line is drawn in the plot's own ink. It was hardcoded BLACK,
            # which on every dark spaCR theme but one is a line nobody can
            # see, and it is the one mark here that must be visible.
            self.add_group_mark(position, v[finite], self._mark, size=7,
                                rows=base[name] + finite,
                                colour=colour_for(position, 200),
                                width=self.MARK_WIDTH,
                                centre=self.MARK_CENTRE)
            median = float(np.median(v[finite]))
            summary.append(f"{name} n={len(finite)} median={median:.3g}")
            if self._mark == "line":
                centre = (float(np.median(v[finite]))
                          if self.MARK_CENTRE == "median"
                          else float(np.mean(v[finite])))
                line_points.append((float(position), centre))
        self._join_group_line(line_points)
        axis = self.plot.getAxis("bottom")
        # THE COUNT BESIDE THE LABEL, not only in the note below the plot.
        # Asked for 2026-08-17. "pc" and "nc" are three and twenty-four
        # points, and a label that does not say so lets a three-point group
        # be read as a group -- which is the same reason the mark advice
        # exists. Taken from the SAME `sizes` the note and the advice use, so
        # the axis cannot disagree with the sentence under it.
        axis.setTicks([[(i, f"{name}\n(n={size})")
                        for i, (name, size) in enumerate(
                            zip(groups, self.group_sizes()))]])
        note = "   ".join(summary) if summary else "No control values."
        if (self._has_usable_keys() and summary
                and self._mark in ("points", "jitter", "jitter_box",
                                   "jitter_bar")):
            note += "   Click a point for its coefficient."
        mark_note = self.mark_note()
        if mark_note:
            note += "   " + mark_note
        self.set_status(note)
        if self._selected_key is not None:
            self.highlight_key(self._selected_key)
        return total

    def group_of(self, row: int) -> Optional[str]:
        """Which group the flat row ``row`` belongs to."""
        for start, stop, name in self._spans:
            if start <= int(row) < stop:
                return name
        return None

    def _detail(self, index: int) -> str:
        parts = []
        name = self.group_of(index)
        if name:
            parts.append(str(name))
        if index < len(self._effects) and np.isfinite(self._effects[index]):
            parts.append(f"effect = {self._effects[index]:.3g}")
        return "   ".join(parts)


class GuideAgreementPlot(GroupedPlot):
    """Per gene: do its own guides push the same way?

    The interactive twin of :func:`spacr.figures.panels.guide_agreement`, and
    the one thing a volcano structurally cannot show. A gene called by one
    guide out of six and a gene whose six guides agree are the same dot on a
    volcano, ranked by the same number, and only one of them is corroborated
    evidence.

    Measured on the TSG101 screen: 389 genes, of which 102 rest on a single
    surviving guide -- including 244480, whose gene-level p of 2.9e-13 ranks
    it above everything else in the screen and IS that one guide's p-value.

    THE HOUSE RULE DECIDES THE COLOURING. Everything is grey except what the
    sentence is about, and the sentence here is "these ones rest on a single
    guide", so those are the only points that get colour.
    """

    #: Default ink for a gene whose guides corroborate each other.
    GREY = "#B4B4B4"
    #: The argument: a gene with nothing to corroborate it.
    SINGLE = "#C44E52"

    #: Genes with the same guide count sit on the same integer x, so a mark a
    #: whole unit wide would touch its neighbour.
    MARK_WIDTH = 0.7

    def __init__(self, parent=None):
        super().__init__(title="Guide agreement", x_label="guides per gene",
                         y_label="fraction agreeing in sign", parent=parent)
        self._support = None
        self._rows_shown: np.ndarray = np.empty(0, dtype="int64")
        #: The last call's arguments, so changing the mark redraws THESE
        #: genes. See :meth:`GroupedPlot.redraw`.
        self._support_keys = None
        self._support_key_column = "feature"

    def redraw(self) -> None:
        """Draw the stored support table again with the current mark."""
        if self._support is not None:
            self.set_support(self._support, keys=self._support_keys,
                             key_column=self._support_key_column)

    def group_sizes(self) -> list:
        """Genes per distinct guide count -- the groups a box would draw."""
        counts, agree = self._guide_counts_and_agreement()
        if counts is None:
            return []
        usable = ~(np.isnan(counts) | np.isnan(agree))
        return [int(np.sum(usable & (counts == value)))
                for value in np.unique(counts[~np.isnan(counts)])]

    def _guide_counts_and_agreement(self):
        """``(n_guides, concordance)`` from the stored table, or ``(None, None)``."""
        frame = self._support
        if frame is None or not len(frame):
            return None, None
        frame = frame.reset_index() if frame.index.name else frame
        counts = _finite(frame["n_guides"]) if "n_guides" in frame \
            else np.full(len(frame), np.nan)
        agree = _finite(frame["concordance"]) if "concordance" in frame \
            else np.full(len(frame), np.nan)
        return counts, agree

    def set_support(self, support, *, keys=None, key_column: str = "feature"):
        """Draw one point per gene. Returns the number of genes plotted.

        :param support: the frame :func:`spacr.guide_concordance.guide_support`
            returns -- ``n_guides``, ``concordance``, ``single_guide`` per
            gene -- indexed by gene or carrying a ``gene`` column.
        :param keys: one identifier per row of ``support``. Default is
            ``support[key_column]`` when that column is there.

        THE KEY IS THE GENE-LEVEL TERM, NOT THE GENE ID. A gene appears in
        the coefficient table as ``gene_fraction:gene[244480]``, and that is
        what the volcano, the table and the gene tile all join on. Handing
        this plot the bare ``244480`` would make a second key space that
        nothing else can resolve, so the caller passes the term and clicking
        a gene here selects exactly the row clicking its dot on the volcano
        would.
        """
        self._reset_scene()
        self._support = support
        self._support_keys = keys
        self._support_key_column = key_column
        self._rows_shown = np.empty(0, dtype="int64")
        self._frame = None
        if support is None or not len(support):
            self.set_keys(())
            self.set_status("No guide-level terms were fitted, so guide "
                            "support is unknown.")
            return 0

        frame = support.reset_index() if support.index.name else support
        # THE TABLE THE RESTYLE MENU READS. It is the RESET frame, not the
        # argument: the row indices carried into every scatter below are
        # positions in this one, so a column mapped onto a colour or a shape
        # has to be indexed the same way or it would shade the wrong genes.
        self._frame = frame
        if keys is None and key_column in getattr(frame, "columns", ()):
            keys = frame[key_column]
        self.set_keys(keys)

        counts = _finite(frame["n_guides"]) if "n_guides" in frame \
            else np.full(len(frame), np.nan)
        agree = _finite(frame["concordance"]) if "concordance" in frame \
            else np.full(len(frame), np.nan)
        single = (np.asarray(frame["single_guide"], dtype=bool)
                  if "single_guide" in frame else counts <= 1)

        # JITTERED, for the same reason the static panel is: guides per gene
        # is a small integer and agreement is a handful of fractions, so
        # several hundred genes stack into a dozen dots and the panel looks
        # like it holds no data. Seeded, so the picture is the same every
        # time, and recorded per row -- the ring a selection draws reads its
        # coordinates back out of `_row_xy`, so it lands on the dot the user
        # actually sees rather than on the un-jittered lattice point.
        rows = np.arange(len(frame))
        if self._mark in ("points", "jitter"):
            # THE HOUSE-RULE COLOURING SURVIVES ONLY WHILE THE MARKS ARE
            # POINTS. Grey for a gene its own guides corroborate, colour for
            # one that rests on a single guide, which is the sentence this
            # panel exists to make -- and a sentence about individual genes
            # that a box plot cannot carry, because a box holds both kinds at
            # once. So the point marks keep this path and the summarising
            # marks take the grouped one below, rather than one path drawing
            # a compromise neither picture wanted.
            spread = 0.22 if self._mark == "jitter" else 0.0
            rng = np.random.default_rng(0)
            x = counts + (rng.uniform(-spread, spread, len(frame))
                          if spread else 0.0)
            for mask, colour, size in ((~single, self.GREY, 7),
                                       (single, self.SINGLE, 9)):
                if not mask.any():
                    continue
                picked = rows[mask]
                self.add_scatter(x[picked], agree[picked], size=size,
                                 rows=picked,
                                 colours=[QColor(colour)] * len(picked))
        else:
            # ONE MARK PER GUIDE COUNT. The x-axis is already a small integer,
            # so the groups are the counts themselves and no tick remapping is
            # needed -- "3" on the axis still means three guides.
            x = counts
            line_points = []
            usable = ~(np.isnan(counts) | np.isnan(agree))
            for position in np.unique(counts[~np.isnan(counts)]):
                picked = rows[usable & (counts == position)]
                if not len(picked):
                    continue
                self.add_group_mark(float(position), agree[picked], self._mark,
                                    rows=picked, colour=QColor(self.GREY),
                                    width=self.MARK_WIDTH, size=7,
                                    centre=self.MARK_CENTRE)
                if self._mark == "line":
                    values = agree[picked]
                    centre = (float(np.median(values))
                              if self.MARK_CENTRE == "median"
                              else float(np.mean(values)))
                    line_points.append((float(position), centre))
            self._join_group_line(line_points)
        self._rows_shown = rows

        self.add_line(y=0.5, colour=self.GREY, label="chance")

        # ONE GENE MUST NOT OWN THE AXIS. The library gives a gene two to four
        # guides; the non-targeting control block parses as a single "gene"
        # carrying all 24 of them, and on autorange that ONE point stretches
        # the x-axis six times wider than the data and squashes all 388 real
        # genes into the left fifth of the panel. It is the identical failure
        # the intercept caused on the volcano, measured the same way.
        #
        # The volcano's answer was to DROP the offender, because a nuisance
        # term is not a hypothesis. This one is different: an over-represented
        # gene is still a gene, and dropping it would lose a real point. So it
        # is drawn and merely left outside the OPENING view -- "Reset view"
        # reaches it, and the status says it is out there rather than letting
        # it disappear silently.
        beyond = 0
        finite = counts[np.isfinite(counts)]
        if len(finite):
            bound = max(4.0, float(np.ceil(np.percentile(finite, 99))))
            beyond = int(np.sum(finite > bound))
            self.plot.setXRange(0.5, bound + 0.5, padding=0.02)

        drawn = int(np.sum(~(np.isnan(x) | np.isnan(agree))))
        alone = int(np.sum(single & ~np.isnan(agree)))
        note = (f"{drawn} genes; {alone} rest on a single guide, so nothing "
                f"corroborates them and they are indistinguishable from "
                f"agreement on a volcano.")
        if beyond:
            plural = beyond != 1
            note += (f" {beyond} gene{'s' if plural else ''} with far more "
                     f"guides than the rest of the library "
                     f"{'are' if plural else 'is'} drawn beyond the opening "
                     f"view; Reset view reaches {'them' if plural else 'it'}.")
        if (self._has_usable_keys()
                and self._mark in ("points", "jitter", "jitter_box",
                                   "jitter_bar")):
            note += " Click a gene for its coefficient."
        mark_note = self.mark_note()
        if mark_note:
            note += " " + mark_note
        self.set_status(note)
        if self._selected_key is not None:
            self.highlight_key(self._selected_key)
        return drawn

    def _detail(self, index: int) -> str:
        frame = self._support
        if frame is None or not len(frame):
            return ""
        frame = frame.reset_index() if frame.index.name else frame
        if not 0 <= int(index) < len(frame):
            return ""
        row = frame.iloc[int(index)]
        parts = []
        if "n_same_direction" in frame and "n_guides" in frame:
            parts.append(f"{int(row['n_same_direction'])} of "
                         f"{int(row['n_guides'])} guides agree")
        if "gene_p" in frame and np.isfinite(row["gene_p"]):
            parts.append(f"gene p = {row['gene_p']:.3g}")
        if "single_guide" in frame and bool(row["single_guide"]):
            parts.append("SINGLE GUIDE -- gene p IS that guide's p")
        return "   ".join(parts)


class ResultsTable(QWidget):
    """The coefficient table, sortable and searchable, wired to a plot.

    Use the table to inspect exact values that cannot be read reliably from a
    scatter plot. Selecting a row can synchronize a linked plot.

    :ivar row_selected: emitted with the frame row index of the selected row.
    :ivar key_selected: emitted with the selected row's identifier. This is
        the one to connect a plot to: an index only means anything to the
        frame it came from, and the table's frame and the plot's frame are
        not required to be the same one.
    """

    row_selected = Signal(int)
    key_selected = Signal(str)
    #: Emitted with the complete selection. ``key_selected`` remains available
    #: to consumers that can display only one row; both signals are emitted by
    #: :meth:`select_keys` from the same live selection.
    keys_selected = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import (QAbstractItemView, QLineEdit,
                                       QTableWidget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        top = QHBoxLayout()
        self._filter = QLineEdit()
        self._filter.setPlaceholderText(
            "Filter rows — type a gene, a guide, anything in the table")
        self._filter.textChanged.connect(self._on_filter_text)
        top.addWidget(self._filter, 1)
        self._only_hits = QCheckBox("significant only")
        self._only_hits.toggled.connect(self._apply_filter)
        top.addWidget(self._only_hits)
        self._copy = QPushButton("Copy")
        self._copy.setToolTip("Copy the visible rows as TSV.")
        self._copy.clicked.connect(self.copy_visible)
        top.addWidget(self._copy)
        layout.addLayout(top)

        self.table = QTableWidget(0, 0)
        # The application's one sorting contract: descending on the first
        # click, ascending on the second, the frame's own order on the third.
        install_sorting(self.table)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        # EXTENDED, NOT SINGLE (instruction 206). A band over the volcano
        # selects several guides, and a table that can only hold one of them
        # would show a different guide from the plot that fed it.
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._on_selection)
        layout.addWidget(self.table, 1)

        self._count = QLabel("")
        layout.addWidget(self._count)

        self._frame = None
        self._alpha = 0.05
        self._key_column: Optional[str] = None
        # EVERY PIECE OF STATE _apply_filter READS IS BORN HERE.
        #
        # `_significance` was created only in set_frame, and the filter
        # controls are connected in this constructor -- so any path that
        # touched a control before the first frame arrived crashed the
        # application on startup with AttributeError. configure() is one such
        # path: it can uncheck "significant only", which emits toggled.
        #
        # A widget must be fully usable the moment it exists. Half-built state
        # that only becomes valid after some other method has been called is
        # how a constructor turns into a trap.
        self._significance: Optional[str] = None
        #: Identifiers the table has been narrowed to from a plot, or None.
        self._key_restriction: Optional[set] = None

    def set_frame(self, frame, *, alpha: float = 0.05,
                  significance_column: Optional[str] = None,
                  key_column: Optional[str] = None) -> int:
        """Fill the table. Returns the row count."""
        from PySide6.QtWidgets import QTableWidgetItem

        self._frame = frame
        self._alpha = alpha
        # A new table is a new experiment: a set of keys chosen off the last
        # one names nothing here, and leaving it on would hide every row.
        self._key_restriction = None
        self._key_column = key_column or (
            "feature" if frame is not None and "feature" in frame.columns
            else None)
        self._significance = significance_column or self._guess_significance(frame)
        if frame is None or not len(frame):
            self.table.setRowCount(0)
            self._count.setText("Nothing to show.")
            return 0

        columns = list(frame.columns)
        # Sorting must be off while filling: with it on, Qt re-sorts after
        # every insert and the rows end up interleaved.
        self.table.setSortingEnabled(False)
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(len(frame))
        for row in range(len(frame)):
            for column, name in enumerate(columns):
                value = frame.iloc[row][name]
                item = table_item(value)
                # The frame row, so a click still maps home after sorting.
                item.setData(Qt.UserRole, row)
                self.table.setItem(row, column, item)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self._apply_filter()
        return len(frame)

    @staticmethod
    def _guess_significance(frame) -> Optional[str]:
        """Prefer a corrected column: filtering on raw p would mislead."""
        if frame is None:
            return None
        for name in ("q_value", "adjusted_p_value", "p_value"):
            if name in frame.columns:
                return name
        return None

    def show_keys(self, keys) -> int:
        """Narrow the table to a set of identifiers. ``None`` clears it.

        The other end of :attr:`FastPlot.keys_selected`: a histogram bar is a
        hundred coefficients and cannot select one of them, but "show me the
        hundred" is a question the table can answer exactly. Returns how many
        rows are visible afterwards.
        """
        self._key_restriction = None if keys is None else {
            str(key) for key in keys}
        self._apply_filter()
        return sum(not self.table.isRowHidden(row)
                   for row in range(self.table.rowCount()))

    def _on_filter_text(self) -> None:
        """Typing is a new intent, so it drops a set chosen on a plot.

        Otherwise the two filters AND together and the user types a gene they
        can see in the plot, gets nothing, and has no way to find out why.
        """
        self._key_restriction = None
        self._apply_filter()

    def _apply_filter(self) -> None:
        text = self._filter.text().strip().lower()
        hits_only = self._only_hits.isChecked()
        # The significance cut needs the frame to find its column in. Without
        # one there is nothing to cut on, and asking for the column would be
        # the same crash one line further down.
        significance = self._significance if self._frame is not None else None
        shown = 0
        for row in range(self.table.rowCount()):
            visible = True
            if text:
                visible = any(
                    text in (self.table.item(row, c).text() or "").lower()
                    for c in range(self.table.columnCount())
                    if self.table.item(row, c) is not None)
            if visible and hits_only and significance:
                if significance not in self._frame.columns:
                    continue
                column = list(self._frame.columns).index(significance)
                item = self.table.item(row, column)
                try:
                    visible = float(item.text()) <= self._alpha
                except (TypeError, ValueError):
                    visible = False
            if visible and self._key_restriction is not None:
                item = self.table.item(row, 0)
                index = None if item is None else item.data(Qt.UserRole)
                key = None if index is None else self.key_for_row(int(index))
                visible = key is not None and key in self._key_restriction
            self.table.setRowHidden(row, not visible)
            shown += int(visible)
        total = self.table.rowCount()
        note = f"{shown} of {total} rows"
        if hits_only and significance:
            note += f" ({significance} <= {self._alpha:g})"
        if self._key_restriction is not None:
            # Said out loud, because a table that has silently narrowed itself
            # is indistinguishable from a table that has lost its rows.
            note += (f" — narrowed to {len(self._key_restriction)} chosen on "
                     f"a plot; type here to clear")
        self._count.setText(note)

    def _on_selection(self) -> None:
        items = self.table.selectedItems()
        if not items:
            return
        index = items[0].data(Qt.UserRole)
        if index is None:
            return
        index = int(index)
        self.row_selected.emit(index)
        key = self.key_for_row(index)
        if key is not None:
            self.key_selected.emit(key)

    def configure(self, *, placeholder: Optional[str] = None,
                  significance_filter: Optional[bool] = None) -> None:
        """Adapt the controls to what the table actually holds.

        This widget is reused for tables that are not coefficient tables --
        the sweep's runs, for one -- and a filter offering "significant only"
        over a list of trials is a control that cannot do anything, sitting
        next to a placeholder telling the user to type a gene into it.
        """
        if placeholder is not None:
            self._filter.setPlaceholderText(placeholder)
        if significance_filter is not None:
            self._only_hits.setVisible(bool(significance_filter))
            if not significance_filter:
                self._only_hits.setChecked(False)

    def key_for_row(self, index: int) -> Optional[str]:
        """The identifier at frame position ``index``, or ``None``."""
        if self._frame is None or not self._key_column:
            return None
        if self._key_column not in self._frame.columns:
            return None
        if not 0 <= int(index) < len(self._frame):
            return None
        return str(self._frame[self._key_column].iloc[int(index)])

    def select_frame_row(self, index: int) -> bool:
        """Scroll to and select the row for frame position ``index``.

        This is the other half of clicking a point on the volcano: the dot and
        the numbers behind it should be two views of one thing.
        """
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.data(Qt.UserRole) == index:
                self.table.selectRow(row)
                self.table.scrollToItem(item)
                return True
        return False

    def select_key(self, key) -> bool:
        """Select the row whose identifier is ``key``. The safe direction.

        A plot has no business knowing where a row sits in this table -- the
        user sorts it, filters it, and after the redesign it may not even
        be drawn from the same frame. It knows the key, and the key is enough.

        A hidden row is unhidden to select it: silently doing nothing because
        the filter box excludes the point the user just clicked reads as a
        broken click.
        """
        if self._frame is None or not self._key_column:
            return False
        if self._key_column not in self._frame.columns:
            return False
        wanted = str(key)
        column = list(self._frame.columns).index(self._key_column)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, column)
            if item is not None and item.text() == wanted:
                self.table.setRowHidden(row, False)
                self.table.selectRow(row)
                self.table.scrollToItem(item)
                return True
        return False

    def select_keys(self, keys) -> int:
        """Select table rows whose keys occur in ``keys``.

        Both selection signals are emitted here so every consumer receives
        the same ordered selection. Rows remain visible; :meth:`show_keys`
        performs filtering when a plot interaction requires it.

        :param keys: row keys to select, in the desired selection order.
        :returns: number of matching rows selected.
        """
        wanted = [str(k) for k in (keys or ())]
        if self._frame is None or not self._key_column:
            return 0
        if self._key_column not in self._frame.columns:
            return 0
        from PySide6.QtCore import QItemSelectionModel

        column = list(self._frame.columns).index(self._key_column)
        self.table.clearSelection()
        model = self.table.selectionModel()
        flags = (QItemSelectionModel.Select | QItemSelectionModel.Rows)
        found = 0
        first = None
        for row in range(self.table.rowCount()):
            item = self.table.item(row, column)
            if item is None or item.text() not in wanted:
                continue
            # A hidden row is unhidden to select it, for the same reason
            # `select_key` does: silently dropping the point the user just
            # dragged over reads as a broken gesture.
            self.table.setRowHidden(row, False)
            if model is not None:
                model.select(self.table.model().index(row, column), flags)
            first = first if first is not None else item
            found += 1
        if first is not None:
            self.table.scrollToItem(first)
        if wanted:
            self.key_selected.emit(wanted[-1])
        self.keys_selected.emit(list(wanted))
        return found

    def selected_keys(self) -> list:
        """Return selected row identifiers in visual order.

        The method reads the live widget selection so Ctrl-click changes are
        represented immediately.
        """
        if self._frame is None or not self._key_column:
            return []
        if self._key_column not in self._frame.columns:
            return []
        column = list(self._frame.columns).index(self._key_column)
        rows = sorted({index.row()
                       for index in self.table.selectedIndexes()})
        out = []
        for row in rows:
            item = self.table.item(row, column)
            if item is not None and item.text():
                out.append(item.text())
        return out

    def copy_visible(self) -> str:
        """Put the visible rows on the clipboard as TSV, and return them."""
        from PySide6.QtWidgets import QApplication

        lines = ["\t".join(
            self.table.horizontalHeaderItem(c).text()
            for c in range(self.table.columnCount()))]
        for row in range(self.table.rowCount()):
            if self.table.isRowHidden(row):
                continue
            lines.append("\t".join(
                (self.table.item(row, c).text() if self.table.item(row, c)
                 else "")
                for c in range(self.table.columnCount())))
        text = "\n".join(lines)
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        return text
