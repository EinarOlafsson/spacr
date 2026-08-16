"""The house style for every figure spaCR draws.

Asked for on 2026-08-16: "i would like a more polished modern look ... i
generally like figures in the style of Sebastian Lourido (papers he is last
author on)", and "the all figures section should look like a publication
ready figure".

The visual system is the one in ``.claude/skills/apicomplexan-figures``,
derived by direct inspection of published figures rather than from general
design taste — Waldman *et al.* Cell 2020 and Giuliano *et al.* Nature
Microbiology 2024. Read that skill before adding a panel; the rules it states
are the ones this module implements.

THE ONE RULE THAT MATTERS MOST, restated here because every panel has to obey
it: **everything is grey except what the sentence is about.** Grey is the
default ink for data; colour is an argument. A highlight set is a small
minority of the marks — if half the points are coloured, the figure has no
claim.

TWO THINGS THIS MODULE DOES THAT THE SKILL'S OWN HELPER DOES NOT, both
because spaCR is an application and not a figure script:

1. **IT NEVER WRITES rcParams GLOBALLY.** The skill's ``use()`` calls
   ``mpl.rcParams.update`` at module level. In a long-running GUI that is a
   process-wide mutation: draw one figure and every later figure in the
   session inherits its font, palette and ground. That is not hypothetical —
   five spaCR functions were doing exactly that and were fixed the same day
   this module was written. Everything here is a **context manager**.

2. **THE INK FOLLOWS THE THEME.** The published palette is for paper: near
   black on white. On spaCR's dark theme that is invisible axes on a
   transparent page. The hues are fixed and never re-mapped; only the ink and
   the ground resolve against where the figure is going — screen or file.
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

#: Transparent, spelled the way matplotlib understands it. The maintainer's
#: standing preference (instruction 118): "not black not white just
#: transparent".
TRANSPARENT = "none"


class Palette:
    """The fixed hues. Sampled from the published figures; do not invent more.

    Strain and condition colours are **fixed across every panel of a figure**.
    Assign once and never re-map: a reader who has learned that blue is the
    knockout in panel B must not find it means something else in panel E.
    """

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

#: Absolute type sizes that reproduce the published look at 300 dpi. The
#: skill states these as a ratio; they are pinned here so a panel cannot
#: drift.
TYPE_SCALE = {
    "tick": 6.2,
    "label": 7.0,        # the 1.0x reference
    "annotation": 6.0,
    "panel_letter": 13.0,
    "legend": 5.6,
}

#: Line weights, likewise measured rather than chosen.
WEIGHTS = {"spine": 0.65, "data": 1.2, "reference": 0.6}


def resolve_ink(target: str = "screen", ink: Optional[str] = None) -> str:
    """The text and axis colour for where this figure is going.

    :param target: ``'screen'`` for the GUI, ``'print'`` for a file that will
        be looked at on paper or in a white-page viewer.
    :param ink: an explicit override, which always wins.
    """
    if ink:
        return ink
    return INK_PRINT if target == "print" else INK_SCREEN


def rc(target: str = "screen", *, frame: str = "L",
       ink: Optional[str] = None,
       ground: Optional[str] = None) -> dict:
    """The rcParams for the house style, as a plain dict.

    Returned rather than applied, so a caller can hand it to
    :func:`figure_style` or to ``plt.rc_context`` directly. Nothing in this
    module ever mutates the global rcParams.

    :param frame: ``'L'`` draws the left and bottom spines only (the Cell
        figures); ``'box'`` draws all four (Nature Microbiology). Pick one
        per figure and hold it — box reads better when panels are small and
        dense, L when they are sparse.
    :param ground: the figure and axes background. Defaults to transparent,
        which is the maintainer's standing preference and lets the GUI theme
        show through.
    """
    box = frame == "box"
    colour = resolve_ink(target, ink)
    ground = TRANSPARENT if ground is None else ground
    return {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": TYPE_SCALE["tick"],
        "axes.labelsize": TYPE_SCALE["label"],
        "axes.titlesize": TYPE_SCALE["label"],
        "axes.titleweight": "regular",
        "axes.titlelocation": "center",
        "axes.edgecolor": colour,
        "axes.labelcolor": colour,
        "axes.linewidth": WEIGHTS["spine"],
        # NO GRIDLINES. EVER. The published figures have none, and a grid is
        # the fastest way to make a panel look like a spreadsheet.
        "axes.grid": False,
        "axes.spines.top": box,
        "axes.spines.right": box,
        "xtick.color": colour, "ytick.color": colour,
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
    "INK_PRINT", "INK_SCREEN", "TRANSPARENT", "Palette", "ROLES",
    "TYPE_SCALE", "WEIGHTS", "annotate", "descriptor", "figure_style",
    "hide_unused", "panel_letter", "rc", "reference_line", "resolve_ink",
    "rotate_ticks", "text_legend", "theme_target",
]
