"""How spaCR's figures look: one general style, plus per-graph overrides.

WHY THIS EXISTS

Every plot inherited matplotlib's defaults, which is what "the graphs look
pretty ugly" means in practice. Restyling each figure by hand after a run is
work the application should do once -- and a per-figure restyle is lost the
next time the analysis is re-run, which during a revision is constantly.

TWO LEVELS, and the split matters. A GENERAL style covers what every figure
shares: font, palette, grid, spines, marker and line size. A PER-GRAPH style
overrides it for one kind of plot, because the settings that make a volcano
readable are not the ones that make a plate heatmap readable. Changing the
volcano's point size must not touch the heatmaps.

Nothing here imports Qt or matplotlib at module level, so the style can be
read, merged and tested without a display.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Optional

#: Applied to every figure unless a graph kind overrides it.
GENERAL_DEFAULTS: dict[str, Any] = {
    "font_family": "DejaVu Sans",
    "font_size": 11.0,
    "title_size": 13.0,
    "label_size": 11.0,
    "tick_size": 9.0,
    # Colour-blind-safe and print-safe. A screen's categories are nominal, so
    # a sequential map would imply an order that is not there.
    "palette": "colorblind",
    "background": "#FFFFFF",
    "foreground": "#222222",
    "grid": True,
    "grid_colour": "#DDDDDD",
    "grid_width": 0.6,
    "grid_style": "-",
    "spines": "left_bottom",      # all | left_bottom | none
    "spine_width": 1.0,
    "marker_size": 28.0,
    "line_width": 1.4,
    "dpi": 150,
    "format": "pdf",
    "tight_layout": True,
}

#: Per-graph overrides. A key absent here falls through to GENERAL_DEFAULTS,
#: so a graph kind only states what it needs to differ on.
GRAPH_DEFAULTS: dict[str, dict[str, Any]] = {
    "volcano": {
        "marker_size": 22.0,
        "point_alpha": 0.85,
        "threshold_style": "--",
        "threshold_colour": "#C44E52",
        "threshold_width": 1.2,
        "label_top_n": 10,
        "annotate": True,
        "split_axis": False,
        # 27 LOPIT compartments is a legend taller than the plot, and it
        # costs 40 ms of every redraw. Colour identifies them; the legend
        # only names them.
        "legend": False,
    },
    "plate_heatmap": {
        "colormap": "viridis",
        "centred": False,
        "annotate_cells": False,
        "per_row": 2,
        # A plate is 24x16 wells. Forcing it square stops the wells being
        # square, which is the whole point of looking at one.
        "aspect": "equal",
        "grid": False,
    },
    "histogram": {
        "bins": "auto",
        "fill_colour": "#4C72B0",
        "edge_colour": "#FFFFFF",
        "edge_width": 0.6,
        "log_y": False,
    },
    "scatter": {
        "marker_size": 20.0,
        "point_alpha": 0.7,
        "trend_line": True,
        "trend_colour": "#DD8452",
    },
    "residuals": {
        "marker_size": 16.0,
        "point_alpha": 0.6,
        "reference_style": "--",
        "reference_colour": "#C44E52",
        "trend_line": True,
    },
    "qq": {
        "marker_size": 14.0,
        "reference_style": "--",
        "reference_colour": "#C44E52",
    },
    "jitter_bar": {
        "jitter_width": 0.28,
        "marker_size": 18.0,
        "point_alpha": 0.7,
        "error_bars": "sem",       # sem | sd | ci95 | none
        "bar_alpha": 0.35,
    },
}

#: The graph kinds a user can style, in the order a preferences page shows
#: them: the ones they look at most, first.
GRAPH_KINDS = ("volcano", "plate_heatmap", "histogram", "scatter",
               "residuals", "qq", "jitter_bar")

#: Spine presets, as (top, right, bottom, left) visibility.
SPINE_PRESETS = {
    "all": (True, True, True, True),
    "left_bottom": (False, False, True, True),
    "none": (False, False, False, False),
}


def resolve(kind: Optional[str] = None,
            general: Optional[Mapping[str, Any]] = None,
            overrides: Optional[Mapping[str, Any]] = None) -> dict:
    """The effective style for ``kind``.

    Three layers, each beating the one before: the general defaults, the
    user's general settings, then the per-graph settings for this kind. A
    graph kind states only what it differs on, so a change to the general
    font reaches every plot that has not overridden it.

    :param kind: a member of :data:`GRAPH_KINDS`, or None for the general
        style alone.
    :param general: the user's general settings, if any.
    :param overrides: the user's per-graph settings, keyed by graph kind.
    """
    style = dict(GENERAL_DEFAULTS)
    if general:
        style.update({k: v for k, v in general.items() if v is not None})
    if kind:
        style.update(GRAPH_DEFAULTS.get(kind, {}))
        if overrides:
            per_kind = overrides.get(kind) or {}
            style.update({k: v for k, v in per_kind.items() if v is not None})
    return style


def rc_params(style: Mapping[str, Any]) -> dict:
    """``style`` as matplotlib rcParams.

    Only the keys matplotlib actually has: a style carries settings that no
    rcParam expresses (``per_row``, ``label_top_n``), and passing those to
    ``rcParams.update`` raises rather than being ignored.
    """
    spines = SPINE_PRESETS.get(str(style.get("spines", "all")),
                               SPINE_PRESETS["all"])
    params = {
        "font.family": style.get("font_family", "DejaVu Sans"),
        "font.size": float(style.get("font_size", 11.0)),
        "axes.titlesize": float(style.get("title_size", 13.0)),
        "axes.labelsize": float(style.get("label_size", 11.0)),
        "xtick.labelsize": float(style.get("tick_size", 9.0)),
        "ytick.labelsize": float(style.get("tick_size", 9.0)),
        "figure.facecolor": style.get("background", "#FFFFFF"),
        "axes.facecolor": style.get("background", "#FFFFFF"),
        "text.color": style.get("foreground", "#222222"),
        "axes.labelcolor": style.get("foreground", "#222222"),
        "xtick.color": style.get("foreground", "#222222"),
        "ytick.color": style.get("foreground", "#222222"),
        "axes.grid": bool(style.get("grid", True)),
        "grid.color": style.get("grid_colour", "#DDDDDD"),
        "grid.linewidth": float(style.get("grid_width", 0.6)),
        "grid.linestyle": style.get("grid_style", "-"),
        "axes.linewidth": float(style.get("spine_width", 1.0)),
        "axes.spines.top": spines[0],
        "axes.spines.right": spines[1],
        "axes.spines.bottom": spines[2],
        "axes.spines.left": spines[3],
        "lines.linewidth": float(style.get("line_width", 1.4)),
        "lines.markersize": float(style.get("marker_size", 28.0)) ** 0.5,
        "savefig.dpi": int(style.get("dpi", 150)),
        "figure.dpi": int(style.get("dpi", 150)),
    }
    if style.get("tight_layout"):
        params["figure.autolayout"] = True
    return params


def apply(kind: Optional[str] = None,
          general: Optional[Mapping[str, Any]] = None,
          overrides: Optional[Mapping[str, Any]] = None) -> dict:
    """Push the resolved style into matplotlib and return it.

    Returns the style as well, because the settings matplotlib has no rcParam
    for -- how many heatmaps per row, how many volcano labels -- still have to
    reach the code that draws them.
    """
    style = resolve(kind, general, overrides)
    try:
        import matplotlib as mpl

        mpl.rcParams.update(rc_params(style))
        palette = style.get("palette")
        if palette:
            _apply_palette(palette)
    except Exception:  # pragma: no cover - never fail a run over styling
        pass
    return style


def _apply_palette(name: str) -> None:
    """Set the colour cycle. seaborn's names when it is installed, else ours."""
    import matplotlib as mpl
    from cycler import cycler

    colours = None
    try:
        import seaborn as sns
        colours = sns.color_palette(name).as_hex()
    except Exception:
        from .qt.widgets.fast_plots import PALETTE
        colours = list(PALETTE)
    if colours:
        mpl.rcParams["axes.prop_cycle"] = cycler(color=colours)


# ---------------------------------------------------------------------------
# A SAVED FIGURE IS FOR PAPER, NOT FOR THE SCREEN.
#
# Instruction 150, reported 2026-08-18: "when a graph is saved and the user is
# on a dark theme white elements are changed to black for saving (text lines,
# etc)". On a dark theme `spacr.qt.preferences.get_figure_colors()` hands both
# renderers a WHITE foreground, so the axes, ticks, labels, title and legend
# are white -- and nothing anywhere inverted them at export time. A PNG saved
# with a transparent ground even looks right in a dark file manager and
# disappears when it is pasted into a manuscript, which means the user finds
# out at the point of writing the paper.
#
# THE DECISION LIVES HERE AND THE APPLICATION DOES NOT. This half is
# matplotlib-free and Qt-free, like the rest of the module, so the pyqtgraph
# exporter (`FastPlot._paint_scene`, instruction 150 C) can import
# `saved_figure_appearance` and get the same answer as `spacr.plot.print_ready`
# without either of them owning the rule. Two renderers deciding separately
# what "print" means is the same defect as two engines deciding which
# statistical test applies.
# ---------------------------------------------------------------------------

#: The three states, which are genuinely different jobs (instruction 150 B).
#:
#: ``print``        light ground, dark chrome. The default, because a figure
#:                  saved from spaCR is going into a manuscript.
#: ``screen``       exactly what the user is looking at. The old behaviour,
#:                  kept reachable rather than removed.
#: ``transparent``  no ground, chrome still follows the print rule -- dark ink
#:                  on a transparent ground is still unreadable on a dark
#:                  slide, but the compositing is then the user's choice.
SAVE_MODES = ("print", "screen", "transparent")

#: The page a printed figure is going onto, and the ink it is printed in.
PRINT_GROUND = "#FFFFFF"
PRINT_INK = "#222222"

#: Gridlines are chrome, and they are chrome that is MEANT to be faint. A grid
#: repainted in the ink is a cage over the data, so an illegible grid becomes
#: this rather than :data:`PRINT_INK` -- which is also the light-mode default,
#: so a light-mode save is unchanged.
PRINT_GRID = "#DDDDDD"

#: Contrast below which chrome is repainted, WCAG 2.x ratio against the ground.
#:
#: 2.0 rather than the 3.0 the guideline gives for graphical objects: this
#: number decides whether to CHANGE A USER'S FIGURE, so it is set where the
#: change is unarguable. White on white is 1.0 and #DDDDDD on white is 1.27;
#: the palest thing anyone deliberately draws chrome in sits above 2.
CHROME_CONTRAST_FLOOR = 2.0

#: Contrast below which a DATA colour is NAMED rather than changed (150 D).
#:
#: SET FROM THE HOUSE STYLE RATHER THAN FROM THE GUIDELINE, because a warning
#: that fires on every figure is a warning nobody reads. Measured on
#: `figures.style.ROLES` against white: the palest colour the house style
#: deliberately puts on the page is ``fill`` #E8A88C at 2.02, with ``data``
#: #B4B4B4 at 2.07 -- so WCAG's 3.0 for graphical objects would name the house
#: style's own greys on every save.
#:
#: 1.8 sits under both and over the colours that are genuinely chosen for a
#: dark ground: white 1.0, the pale yellow the instruction describes 1.03,
#: Okabe-Ito yellow #F0E442 1.32. `guide_permutation`'s "not a hit" grey
#: #B8BDC5 is 1.89 and stays quiet, which is the point -- it is de-emphasis,
#: not a mark the reader has to find.
DATA_CONTRAST_FLOOR = 1.8

_NAMED_COLOURS = {
    "white": (1.0, 1.0, 1.0), "black": (0.0, 0.0, 0.0),
    "none": None, "transparent": None,
}


def to_rgb(colour) -> Optional[tuple]:
    """``colour`` as an ``(r, g, b)`` triple on 0-1, or None if it has none.

    None means "no colour to judge" -- ``'none'``, a fully transparent RGBA, a
    colour map, anything this cannot read. Every caller below treats that as
    "leave it alone", which is the safe direction: the failure to avoid is
    repainting something that was never white.

    matplotlib is consulted only if it is already importable, and only for the
    spellings this cannot parse itself, so the rule stays testable headless.
    """
    if colour is None:
        return None
    if isinstance(colour, str):
        text = colour.strip().lower()
        if text in _NAMED_COLOURS:
            return _NAMED_COLOURS[text]
        if text.startswith("#"):
            digits = text[1:]
            if len(digits) in (3, 4):
                digits = "".join(character * 2 for character in digits)
            if len(digits) in (6, 8):
                try:
                    values = [int(digits[i:i + 2], 16) / 255.0
                              for i in range(0, 6, 2)]
                except ValueError:
                    return None
                if len(digits) == 8 and int(digits[6:8], 16) == 0:
                    return None          # fully transparent is not a colour
                return tuple(values)
            return None
    else:
        try:
            values = tuple(float(component) for component in colour)
        except (TypeError, ValueError):
            values = ()
        if len(values) == 4 and values[3] == 0:
            return None
        if len(values) in (3, 4):
            return tuple(values[:3])
    try:                                   # pragma: no cover - optional path
        from matplotlib.colors import to_rgba

        red, green, blue, alpha = to_rgba(colour)
    except Exception:                      # pragma: no cover
        return None
    return None if alpha == 0 else (red, green, blue)


def relative_luminance(colour) -> Optional[float]:
    """WCAG relative luminance of ``colour``, or None when it has no colour."""
    rgb = to_rgb(colour)
    if rgb is None:
        return None
    channels = []
    for value in rgb:
        value = min(max(float(value), 0.0), 1.0)
        channels.append(value / 12.92 if value <= 0.04045
                        else ((value + 0.055) / 1.055) ** 2.4)
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(colour, other) -> Optional[float]:
    """WCAG contrast ratio between two colours, 1.0 to 21.0.

    None when either has no colour -- which is not "no contrast", and the
    distinction matters: a caller that read None as 1.0 would repaint every
    transparent artist in the figure.
    """
    first = relative_luminance(colour)
    second = relative_luminance(other)
    if first is None or second is None:
        return None
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def is_legible_on(colour, ground, floor: float = CHROME_CONTRAST_FLOOR) -> bool:
    """Whether ``colour`` can be seen on ``ground``. Unreadable colours are
    the only ones the save is allowed to change."""
    ratio = contrast_ratio(colour, ground)
    return True if ratio is None else ratio >= float(floor)


class SavedFigureAppearance(NamedTuple):
    """What a save should look like. The answer both renderers act on.

    :param mode: one of :data:`SAVE_MODES`.
    :param ground: the background to paint, or None to leave it as it is.
        None is what ``screen`` and ``transparent`` both want, for opposite
        reasons -- one keeps the ground, the other has none.
    :param ink: the colour illegible CHROME is repainted in, or None to leave
        every colour alone.
    :param grid: the colour illegible GRIDLINES are repainted in.
    :param transparent: pass ``transparent=True`` to the writer.
    :param flip: whether anything is repainted at all. False for ``screen``,
        which is the whole meaning of that mode.
    """

    mode: str
    ground: Optional[str]
    ink: Optional[str]
    grid: Optional[str]
    transparent: bool
    flip: bool


#: What a piece of a figure IS, for the purposes of a save. The whole of
#: instruction 150 A is the difference between the last one and the rest.
#:
#: ``ground``   the page behind everything -- a figure patch, an axes patch,
#:              a legend's fill.
#: ``grid``     gridlines, which are chrome that is MEANT to be faint.
#: ``chrome``   spines, ticks, tick labels, axis labels, title, legend text
#:              and frame, annotation text, the significance line, the zero
#:              line, arrows and leader lines.
#: ``data``     everything that carries the CLAIM. Never repainted.
ARTIST_KINDS = ("ground", "grid", "chrome", "data")


def export_colour(current, kind: str, look=None) -> Optional[str]:
    """The colour one artist should be painted for the save, or None.

    THE PER-ARTIST HALF OF THE SHARED DECISION, and the reason it is here
    rather than in ``spacr.plot``: the design says the rule has to
    reach BOTH renderers, and a rule that lives in the matplotlib application
    forces the pyqtgraph exporter to write a second one. Two renderers
    deciding separately what "print" means is the same defect as two engines
    deciding which statistical test applies.

    :param current: the colour the artist is painted in now. Anything
        :func:`to_rgb` cannot read -- ``'none'``, a fully transparent RGBA, a
        colour map -- is left alone, which is the safe direction: the failure
        to avoid is repainting something that was never white.
    :param kind: one of :data:`ARTIST_KINDS`.
    :param look: a :class:`SavedFigureAppearance`; None asks
        :func:`saved_figure_appearance`.
    :returns: the replacement colour, or None for "leave this one as it is".

    WHAT DECIDES WHETHER A PIECE OF CHROME MOVES IS LEGIBILITY, NOT THE THEME.
    An artist is repainted only when it has less than
    :data:`CHROME_CONTRAST_FLOOR` contrast against the page, so a LIGHT-MODE
    save changes nothing at all -- which is the property that makes ``print``
    safe as the default -- and nothing here reads the theme. It reads the
    FIGURE.

    Example:
        >>> look = saved_figure_appearance("print")
        >>> export_colour("#FFFFFF", "chrome", look)
        '#222222'
        >>> export_colour("#222222", "chrome", look) is None
        True
        >>> export_colour("#FFFFFF", "data", look) is None
        True
    """
    look = saved_figure_appearance() if look is None else look
    if not look.flip or kind == "data":
        # THE DATA NEVER MOVES. A white data point turned black is, on a
        # volcano, the colour of "not a hit" -- section A exists to prevent
        # exactly that, and it is the one line of this function that must
        # never grow a special case.
        return None
    page = look.ground or PRINT_GROUND
    if kind == "ground":
        # Only a DARK ground is repainted. A deliberately tinted light
        # background is somebody's choice, and 'transparent' has no ground to
        # argue about -- the writer owns that.
        luminance = relative_luminance(current)
        if look.ground is None or luminance is None or luminance >= 0.5:
            return None
        return look.ground
    if is_legible_on(current, page):
        return None
    # A grid repainted in the ink is a cage over the data, so an illegible
    # grid becomes the faint print grey instead.
    replacement = look.grid if kind == "grid" else look.ink
    # AND NOTHING IS "REPAINTED" IN THE COLOUR IT ALREADY IS. The light-mode
    # grid default IS `PRINT_GRID`, and #DDDDDD on white is 1.27 contrast --
    # deliberately faint, correctly below the chrome floor, and already the
    # colour it would be changed to. Saying None here is what makes "a
    # light-mode save changes nothing at all" true of the artists as well as
    # of the pixels, and it leaves the caller nothing to restore.
    if to_rgb(replacement) == to_rgb(current):
        return None
    return replacement


def illegible_colours(colours, ground=PRINT_GROUND,
                      floor: Optional[float] = None) -> list:
    """The DATA colours a reader will not find on ``ground``, as hex (150 D).

    The data deliberately does not flip, so a palette chosen against near-black
    can be illegible on paper -- and the honest answer is to NAME the colour,
    because a substitution the user did not ask for changes what the picture
    says. Renderer-free, so the pyqtgraph exporter can hand it a list of pen
    colours and get the same sentence as the matplotlib one.

    :param colours: any iterable of colours, in any spelling :func:`to_rgb`
        reads. Unreadable entries are skipped rather than guessed at.
    :param ground: the page they are going onto.
    :param floor: contrast below which a colour is named; defaults to
        :data:`DATA_CONTRAST_FLOOR`.
    :returns: sorted, deduplicated ``#RRGGBB`` strings, so the same figure
        produces the same sentence twice.

    A WASH IS NOT A MARK. An artist drawn at less than half opacity is
    de-emphasis by construction -- `figures.plates` lays its "never measured"
    colour down at 9% -- and judging its base hue would name a colour nobody
    is being asked to find, on every plate figure.
    """
    floor = DATA_CONTRAST_FLOOR if floor is None else float(floor)
    named = set()
    for colour in colours or ():
        try:
            components = tuple(float(value) for value in colour)
        except (TypeError, ValueError):
            components = ()
        if len(components) == 4 and components[3] < 0.5:
            continue
        rgb = to_rgb(colour)
        if rgb is None or is_legible_on(rgb, ground, floor):
            continue
        named.add("#%02X%02X%02X" % tuple(
            int(round(min(max(channel, 0.0), 1.0) * 255)) for channel in rgb))
    return sorted(named)


def illegible_colour_warning(names) -> str:
    """The sentence for :func:`illegible_colours`, or '' when there is none.

    One sentence, in one place, because both renderers say it and a user who
    saw it once from a tab and once from a run should not have to work out
    whether they are the same warning.
    """
    if not names:
        return ""
    return ("Saved-figure warning: these data colours have almost no contrast "
            "on the light page and are NOT being changed, because the colour "
            f"is the claim: {', '.join(names)}. Pick an accessible palette in "
            "Preferences > Figures if the marks are meant to be read.")


def figure_save_mode() -> str:
    """The save mode in force: the user's preference, else ``'print'``.

    Read defensively for the same reason as
    :func:`spacr.plot.figure_output_preferences`: the preference store is
    Qt's, and the pipelines that save figures run headless from the CLI and
    from notebooks, where importing PySide6 to decide an ink colour would be
    absurd. ``SPACR_FIGURE_SAVE_MODE`` overrides, which is how a headless run
    or a test asks for one without a store at all.

    The Qt getter is looked up by NAME rather than imported, so this works
    before ``spacr.qt.preferences`` grows one -- that module belongs to
    another session and adding the control there is its call, not this one's.
    """
    import os

    requested = os.environ.get("SPACR_FIGURE_SAVE_MODE", "").strip().lower()
    if requested in SAVE_MODES:
        return requested
    try:                                   # pragma: no cover - needs Qt
        from .qt import preferences

        getter = getattr(preferences, "get_figure_save_mode", None)
        if getter is not None:
            stored = str(getter()).strip().lower()
            if stored in SAVE_MODES:
                return stored
    except Exception:                      # pragma: no cover
        pass
    return "print"



#: The ink a transparent figure takes in each theme, and the grid beside it.
#: The light theme is the print pair exactly, so switching a light-theme user to
#: transparent changes nothing about the marks -- only the ground goes.
DARK_INK = "#EDEDED"
DARK_GRID = "#4A4A4A"


def theme_ink() -> Tuple[str, str]:
    """``(ink, grid)`` for the theme the user is working in.

    Matplotlib-free and Qt-free like the rest of this module: the preference
    is read through a late import so a headless run that never built a GUI
    still answers, and answers `print`'s pair -- there is no dark theme to
    follow when there is no application.
    """
    try:
        from .qt.preferences import resolve_effective_theme
    except Exception:                                            # noqa: BLE001
        return PRINT_INK, PRINT_GRID
    try:
        theme = str(resolve_effective_theme() or "").strip().lower()
    except Exception:                                            # noqa: BLE001
        return PRINT_INK, PRINT_GRID
    # `resolve_effective_theme` says so itself: compare against "light" and
    # treat everything else as dark, because Space and Cell are dark themes
    # and "system" has already been resolved by the time it answers.
    return (PRINT_INK, PRINT_GRID) if theme == "light" else (DARK_INK, DARK_GRID)

def saved_figure_appearance(mode: Optional[str] = None
                            ) -> SavedFigureAppearance:
    """THE shared decision. Both renderers ask this and neither decides.

    :param mode: force one of :data:`SAVE_MODES`; None asks
        :func:`figure_save_mode`. An unrecognised mode falls back to
        ``'print'`` rather than raising -- a run must not lose its figures
        over a misspelt preference.

    NOTE WHAT THIS DOES NOT SAY. It names a ground and an ink for the
    FURNITURE. It says nothing about point colours, colour maps or the up/down
    colouring on a volcano, and that omission is the design: those carry the
    claim, and a white data point turned black is, on a volcano, the colour of
    "not a hit".
    """
    chosen = str(mode).strip().lower() if mode is not None else figure_save_mode()
    if chosen not in SAVE_MODES:
        chosen = "print"
    if chosen == "screen":
        return SavedFigureAppearance("screen", None, None, None, False, False)
    if chosen == "transparent":
        # THE INK FOLLOWS THE THEME HERE, and only here.
        #
        # Asked for twice: "the background of the figures should be
        # transparent and the lines should be white on a dark theme and black on
        # a light one", and then reported as a fault -- "a lot of the text in
        # the figures is black on the dark theme and the axes as well".
        #
        # This mode used to keep the PRINT ink on the ground that it removes,
        # on the argument that dark ink on a transparent ground is still
        # unreadable on a dark slide. That argument is right about `print` and
        # wrong about this: transparent MEANS the ground is whatever the
        # figure is pasted onto, and the only thing that knows what that is,
        # is the user -- who says so by the theme they are working in.
        #
        # `print` is unchanged and is still the default, so a figure going
        # into a manuscript is untouched by this.
        ink, grid = theme_ink()
        return SavedFigureAppearance("transparent", None, ink, grid,
                                     True, True)
    return SavedFigureAppearance("print", PRINT_GROUND, PRINT_INK,
                                 PRINT_GRID, False, True)
