"""Shared publication and on-screen styling for spaCR figures.

The module provides a fixed data palette, theme-aware foreground colors, and
scoped Matplotlib style contexts. Data colors retain the same meaning across
panels, while text and axes adapt to screen or print backgrounds. Styles are
applied without mutating process-wide ``rcParams``.

Text and lines are two colours, not one. Titles, axis labels, tick labels,
annotations and legend entries are TEXT; axis spines and tick marks are
LINES. Each follows the matching user preference when one has been chosen and
the measured house ink otherwise, so an untouched settings store draws the
published look unchanged.
"""

from __future__ import annotations

import contextlib
from typing import Iterable, Optional, Sequence

#: Print ink: the near-black the published figures use for text and axes.
INK_PRINT = "#231F20"
#: Screen ink: what the dark GUI needs instead. Not pure white — pure white
#: on a dark ground reads as glare, and the published figures never use a
#: maximal ink either.
INK_SCREEN = "#E8EDEE"

#: Transparent, spelled the way matplotlib understands it.
TRANSPARENT = "none"


class Palette:
    """Fixed data colors shared by all house-style figure panels."""

    GREY = "#B4B4B4"          # default data, non-significant, comparisons
    GREY_DARK = "#7F7F7F"     # secondary series, mean bars
    BLUE = "#2E77BC"          # the primary highlight / the gene of interest
    BLUE_LIGHT = "#7FB3E0"    # a second strain
    GREEN = "#2E7D4F"         # wild type, upregulated
    RUST = "#C4441C"          # downregulated, the other highlight
    CORAL = "#E8A88C"         # density and histogram fills
    GOLD = "#E8C33A"
    OCHRE = "#C87A28"
    PURPLE = "#8B4A82"
    NAVY = "#1F3F6E"

    #: Single-hue ramp for a p-value or a score. Diverging maps are for
    #: genuinely signed quantities only.
    SEQUENTIAL = "Blues"


#: What each role means in a spaCR regression figure, fixed once so a colour
#: cannot drift between panels. This is the whole of the colour vocabulary.
ROLES = {
    "data": Palette.GREY,             # every guide that is not the point
    "up": Palette.GREEN,              # positive effect, called
    "down": Palette.RUST,             # negative effect, called
    "highlight": Palette.BLUE,        # the selected gene
    "control_negative": Palette.GREY_DARK,
    "control_positive": Palette.PURPLE,
    "fill": Palette.CORAL,            # histogram and density fills
    "reference": Palette.GREY_DARK,   # thresholds, limits, 1:1 lines
}

#: Absolute type sizes that reproduce the published hierarchy at 300 dpi.
#: Centralising them prevents individual panels from drifting apart.
TYPE_SCALE = {
    "tick": 6.2,
    "label": 7.0,        # the 1.0x reference
    "annotation": 6.0,
    "panel_letter": 13.0,
    "legend": 5.6,
}

#: Line weights, likewise measured rather than chosen.
WEIGHTS = {"spine": 0.65, "data": 1.2, "reference": 0.6}

#: The rcParams that are TEXT: everything a reader reads. Tick LABELS are in
#: here and tick MARKS are not.
TEXT_KEYS = ("text.color", "axes.labelcolor",
             "xtick.labelcolor", "ytick.labelcolor")
#: The rcParams that are LINES: the axis spines and the tick marks. The
#: figure's chrome, not its data — :func:`resolve_line_ink` says why the
#: data's own series are left alone.
LINE_KEYS = ("axes.edgecolor", "xtick.color", "ytick.color")


def chosen_ink() -> Optional[str]:
    """The TEXT colour the user picked, or ``None`` while it follows the theme.

    Reads the stored TOKEN, never the resolved pair. A resolved pair has
    already lost the one bit that matters here — whether the user chose the
    colour or the theme produced it — so seeding the house style from
    ``get_figure_colors()`` would replace the measured publication ink with
    whatever the current theme happens to answer, for every user who has
    never opened the dialog.

    ``None`` when there is no settings store at all: a headless render or a
    bare unit run, where the house style is the only answer there is.
    """
    try:
        from ..qt.preferences import (figure_color_is_auto,
                                      get_figure_color_tokens)

        _ground, text = get_figure_color_tokens()
    except Exception:                                          # noqa: BLE001
        return None
    if figure_color_is_auto(text):
        return None
    return str(text).strip() or None


def chosen_line_ink() -> Optional[str]:
    """The LINE colour the user picked, or ``None`` while it follows the text.

    The user's second colour control: "line color which should change the
    color of all lines including axis lines and ticks". Stored as a token
    like the text half, and read as a token for the same reason —
    :func:`chosen_ink` says which.
    """
    try:
        from ..qt.preferences import (figure_color_is_auto,
                                      get_figure_line_token)

        token = get_figure_line_token()
    except Exception:                                          # noqa: BLE001
        return None
    if figure_color_is_auto(token):
        return None
    return str(token).strip() or None


def resolve_ink(target: str = "screen", ink: Optional[str] = None) -> str:
    """The TEXT colour for where this figure is going.

    Titles, axis labels, tick LABELS, annotations and legend entries. The
    axis spines and the tick MARKS are lines, and they have their own
    resolver — see :func:`resolve_line_ink`.

    :param target: ``'screen'`` for the GUI, ``'print'`` for a file that will
        be looked at on paper or in a white-page viewer.
    :param ink: an explicit override, which always wins.
    """
    if ink:
        return ink
    return chosen_ink() or (INK_PRINT if target == "print" else INK_SCREEN)


def resolve_line_ink(target: str = "screen", ink: Optional[str] = None,
                     line: Optional[str] = None) -> str:
    """Resolve the colour used for axis spines and tick marks.

    Textual elements, including tick labels, use :func:`resolve_ink`; the
    separate resolvers allow line and text colours to be configured
    independently. If no line colour is configured, the resolved text colour
    preserves the earlier single-ink behavior. Data-series colours remain
    governed by :data:`ROLES` and per-figure styling.

    :param target: Output target passed to :func:`resolve_ink` when a fallback
        is required.
    :param ink: Explicit text colour used as the fallback for line work.
    :param line: Explicit line colour, which takes precedence over all other
        values.
    :returns: Resolved line colour.
    """
    if line:
        return line
    return chosen_line_ink() or resolve_ink(target, ink)


#: The page a LABEL sits on, per target. Not the figure's ground -- that is
#: transparent by design (118) -- but what goes behind a text box that has to
#: stay readable over the data underneath it.
#:
#: THIS EXISTS BECAUSE A BOX WAS HARD-CODED WHITE while its text followed the
#: theme, so on the dark theme it was white ink on a white box: a label that
#: was there, was drawn, and could not be read. `resolve_ink`'s opposite
#: number, and used the same way.
LABEL_GROUND_PRINT = "#FFFFFF"
LABEL_GROUND_SCREEN = "#1B1E20"


def resolve_label_ground(target: str = "screen",
                         ground: Optional[str] = None) -> str:
    """The colour behind a text label, for where this figure is going.

    :param target: ``'screen'`` for the GUI, ``'print'`` for a file.
    :param ground: an explicit override, which always wins.
    """
    if ground:
        return ground
    return LABEL_GROUND_PRINT if target == "print" else LABEL_GROUND_SCREEN


def user_overrides(kind: Optional[str] = None) -> dict:
    """Return explicit preference changes that override the house style.

    Only settings that differ from spaCR's figure defaults are returned. This
    preserves the house style for untouched preferences while allowing general
    and graph-specific choices to take precedence.

    Parameters
    ----------
    kind : str or None, default=None
        Graph kind from :data:`spacr.figure_style.GRAPH_KINDS`. ``None`` uses
        only the general preference layer.

    Returns
    -------
    dict
        Matplotlib ``rcParams`` overrides. An empty dictionary is returned
        when no preference differs or preferences cannot be read safely.
    """
    try:
        from ..qt.preferences import (get_figure_style,
                                      get_figure_style_per_graph)

        general = get_figure_style()
        per_graph = get_figure_style_per_graph()
    except Exception:                                          # noqa: BLE001
        return {}
    # Both stores are EMPTY until the user changes something -- they hold the
    # deltas, not the defaults, on purpose -- so this is the common case and
    # it costs nothing.
    if not general and not per_graph:
        return {}
    try:
        from ..figure_style import rc_params, resolve

        chosen = resolve(kind, general, per_graph)
        untouched = resolve(kind)
        after, before = rc_params(chosen), rc_params(untouched)
        return {key: value for key, value in after.items()
                if before.get(key) != value}
    except Exception:                                          # noqa: BLE001
        return {}


def rc(target: str = "screen", *, frame: str = "L",
       ink: Optional[str] = None,
       line: Optional[str] = None,
       ground: Optional[str] = None,
       kind: Optional[str] = None) -> dict:
    """The rcParams for the house style, as a plain dict.

    Returned rather than applied, so a caller can hand it to
    :func:`figure_style` or to ``plt.rc_context`` directly. Nothing in this
    module ever mutates the global rcParams.

    :param frame: ``'L'`` draws the left and bottom spines only (the Cell
        figures); ``'box'`` draws all four (Nature Microbiology). Pick one
        per figure and hold it — box reads better when panels are small and
        dense, L when they are sparse.
    :param ink: the TEXT colour — titles, labels, tick labels, annotations.
    :param line: the LINE colour — the axis spines and the tick marks. Falls
        back to ``ink`` when nobody has said otherwise, which is what the
        figures did before the two were separable.
    :param ground: the figure and axes background. Defaults to transparent,
        which lets the GUI theme
        show through.
    :param kind: which graph kind this is, so the user's PER-GRAPH preference
        for it can be applied on top. See :func:`user_overrides`.
    """
    box = frame == "box"
    # Read each control ONCE. `picked_*` is None while that half follows the
    # theme, and it is also the flag that decides whether the choice outranks
    # the house-style panel's own colours further down.
    picked_ink = ink or chosen_ink()
    picked_line = line or chosen_line_ink()
    colour = resolve_ink(target, picked_ink)
    from ..figure_font import FAMILY as _FIGURE_FAMILY
    from ..figure_font import use_open_sans_for_figures
    use_open_sans_for_figures()
    line_colour = resolve_line_ink(target, ink=picked_ink, line=picked_line)
    ground = TRANSPARENT if ground is None else ground
    params = {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        # OPEN SANS SHIPS WITH SPACR, so it is always there to resolve.
        # Naming "Helvetica" first meant a Linux machine without it fell
        # silently back to DejaVu Sans, and figures came out in a different
        # face from the interface around them -- and in a different face on
        # each contributor's machine. `use_open_sans_for_figures` registers
        # the bundled files with the font manager, which is what makes the
        # name resolve at all; the rest of the list stays as a fallback.
        "font.sans-serif": [_FIGURE_FAMILY, "Helvetica", "Arial",
                            "DejaVu Sans"],
        "font.size": TYPE_SCALE["tick"],
        "axes.labelsize": TYPE_SCALE["label"],
        "axes.titlesize": TYPE_SCALE["label"],
        "axes.titleweight": "regular",
        "axes.titlelocation": "center",
        "axes.edgecolor": line_colour,
        "axes.labelcolor": colour,
        "axes.linewidth": WEIGHTS["spine"],
        # NO GRIDLINES. EVER. The published figures have none, and a grid is
        # the fastest way to make a panel look like a spreadsheet.
        "axes.grid": False,
        "axes.spines.top": box,
        "axes.spines.right": box,
        # THE MARK IS A LINE, THE LABEL IS TEXT. `xtick.color` is the little
        # dash beside the axis; `xtick.labelcolor` is the number printed next
        # to it. Matplotlib's default for the second is "inherit", so both
        # are named here or the two can never be told apart.
        "xtick.color": line_colour, "ytick.color": line_colour,
        "xtick.labelcolor": colour, "ytick.labelcolor": colour,
        "text.color": colour,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "xtick.major.width": WEIGHTS["spine"],
        "ytick.major.width": WEIGHTS["spine"],
        "xtick.labelsize": TYPE_SCALE["tick"],
        "ytick.labelsize": TYPE_SCALE["tick"],
        "legend.frameon": False,
        "legend.fontsize": TYPE_SCALE["legend"],
        "legend.handlelength": 0.9,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 0.8,
        "figure.facecolor": ground,
        "axes.facecolor": ground,
        "savefig.facecolor": ground,
        "savefig.transparent": ground == TRANSPARENT,
        "lines.linewidth": WEIGHTS["data"],
        "patch.linewidth": WEIGHTS["spine"],
    }
    # LAST, so the user wins. Everything above is the published look; this is
    # the handful of settings they went into Preferences and changed.
    params.update(user_overrides(kind))
    # LATER STILL, and only for a colour the user actually named. The two
    # figure-colour controls are the dedicated ones for these roles, so they
    # outrank the graph-style panel's general `foreground` — which resolves
    # to `xtick.color` and would otherwise repaint the tick marks the line
    # control was just told to own. Nothing is written here while both halves
    # follow the theme, so an untouched store keeps the house style exactly.
    if picked_ink:
        params.update(dict.fromkeys(TEXT_KEYS, colour))
    if picked_ink or picked_line:
        params.update(dict.fromkeys(LINE_KEYS, line_colour))
    return params


@contextlib.contextmanager
def figure_style(target: str = "screen", **kwargs):
    """Draw inside the house style, and put the globals back afterwards.

    THE ONLY SUPPORTED WAY TO APPLY THIS STYLE.

    ::

        with figure_style("print"):
            figure = build_volcano(results)
        figure.savefig(path)          # already styled; the globals are back

    A plain ``rcParams.update`` would leak: spaCR draws figures from a
    long-lived GUI, so a style applied once applies to every figure drawn
    afterwards, in every other module, until the process exits. That failure
    mode has already cost this repository a day.
    """
    import matplotlib.pyplot as plt

    with plt.rc_context(rc(target, **kwargs)):
        yield


def theme_target() -> str:
    """``'screen'`` or ``'print'``, from the user's own figure preferences.

    The GROUND decides this, and only the ground: it is the question "what is
    this figure going to sit on", which is what picks between the two
    measured house inks. The colours the user may have CHOSEN are a different
    question and are read where they are used —
    :func:`chosen_ink` for the text, :func:`chosen_line_ink` for the lines —
    because a chosen colour outranks whichever house ink this returns.

    Falls back to ``'screen'`` when there is no settings store — a headless
    render or a bare unit test — because spaCR's themes are dark and ink that
    is slightly wrong is better than ink that is invisible.
    """
    try:
        from ..qt.preferences import get_figure_colors

        background, _foreground = get_figure_colors()
    except Exception:
        return "screen"
    text = str(background).strip().lower()
    # A white or very light ground means the figure is destined for paper,
    # whatever the GUI theme is doing.
    return "print" if text in ("white", "#ffffff", "#fff") else "screen"


# --------------------------------------------------------------------------- #
#  The small vocabulary every panel shares
# --------------------------------------------------------------------------- #

def panel_letter(ax, letter: str, dx: float = -0.16, dy: float = 1.06) -> None:
    """A bold upper-case letter at the panel's top left. No period.

    Sized from the measured 1.9-2.2x of the axis-label tier.
    """
    ax.text(dx, dy, letter.upper(), transform=ax.transAxes,
            fontsize=TYPE_SCALE["panel_letter"], fontweight="bold",
            va="bottom", ha="left")


def descriptor(ax, text: str) -> None:
    """Two to four lower-case words above the axes.

    NOT a sentence title. The axis labels carry the content; a descriptor
    only says which condition this panel is.
    """
    ax.set_title(text, fontsize=TYPE_SCALE["label"], pad=3.0)


def reference_line(ax, *, x=None, y=None, label: str = "",
                   colour: Optional[str] = None) -> None:
    """A threshold, a limit of detection, a 1:1 diagonal.

    Thin, dashed and grey — never bold, never coloured. A reference is not a
    result and must not compete with one.
    """
    colour = colour or ROLES["reference"]
    drawn = (ax.axvline(x, color=colour, lw=WEIGHTS["reference"], ls=(0, (4, 3)),
                        zorder=0) if x is not None else
             ax.axhline(y, color=colour, lw=WEIGHTS["reference"], ls=(0, (4, 3)),
                        zorder=0))
    if label:
        if x is not None:
            ax.annotate(label, (x, 0.98), xycoords=("data", "axes fraction"),
                        fontsize=TYPE_SCALE["annotation"], color=colour,
                        ha="left", va="top", rotation=90,
                        xytext=(2, -2), textcoords="offset points")
        else:
            ax.annotate(label, (0.99, y), xycoords=("axes fraction", "data"),
                        fontsize=TYPE_SCALE["annotation"], color=colour,
                        ha="right", va="bottom",
                        xytext=(0, 2), textcoords="offset points")
    return drawn


def text_legend(ax, entries: Sequence, x: float = 0.02, y: float = 0.97,
                dy: float = 0.075) -> None:
    """A legend as coloured TEXT, with no marker and no frame.

    What the published figures do. A framed legend with sample markers costs
    a corner of the axes and adds a box the style has no other boxes to
    match.

    :param entries: ``[(label, colour), ...]``.
    """
    for index, (label, colour) in enumerate(entries):
        ax.text(x, y - index * dy, label, transform=ax.transAxes,
                fontsize=TYPE_SCALE["annotation"], color=colour,
                ha="left", va="top")


def rotate_ticks(ax, degrees: int = 45) -> None:
    """Long categorical labels rotate 45 degrees, right-aligned."""
    for label in ax.get_xticklabels():
        label.set_rotation(degrees)
        label.set_ha("right")
        label.set_rotation_mode("anchor")


def annotate(ax, text: str, *, x: float = 0.02, y: float = 0.97,
             colour: Optional[str] = None, ha: str = "left",
             va: str = "top") -> None:
    """An in-panel note: an n, a correlation coefficient, a count.

    No frame, no box. The published figures never draw one.
    """
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=TYPE_SCALE["annotation"],
            color=colour or ax.xaxis.label.get_color(), ha=ha, va=va)


def hide_unused(axes: Iterable) -> None:
    """Turn off axes a grid allocated and no panel filled.

    An empty framed box in a figure sheet reads as a panel that failed to
    draw, which is worse than a gap.
    """
    for ax in axes:
        ax.set_axis_off()


__all__ = [
    "INK_PRINT", "INK_SCREEN", "LINE_KEYS", "TEXT_KEYS", "TRANSPARENT",
    "Palette", "ROLES", "TYPE_SCALE", "WEIGHTS", "annotate", "chosen_ink",
    "chosen_line_ink", "descriptor", "figure_style", "hide_unused",
    "panel_letter", "rc", "reference_line", "resolve_ink",
    "resolve_line_ink", "rotate_ticks", "text_legend", "theme_target",
    "user_overrides",
]
