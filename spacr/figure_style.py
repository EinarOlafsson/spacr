"""Resolve spaCR figure styles for display and export.

General settings define the shared appearance of every figure. Per-graph
settings override only the values needed by a specific graph type. The style
tables and export-colour helpers can be used without starting Qt or importing
Matplotlib at module import time.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Optional, Tuple

#: Default settings applied to every figure before per-graph overrides.
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

#: Default overrides for each graph type. Missing keys inherit
#: :data:`GENERAL_DEFAULTS`.
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
        # The set this may take is STYLE_CHOICES["error_bars"], not the
        # comment that used to be on this line.
        "error_bars": "sem",
        "bar_alpha": 0.35,
    },
}

#: Graph types available in the figure-style preferences, in display order.
GRAPH_KINDS = ("volcano", "plate_heatmap", "histogram", "scatter",
               "residuals", "qq", "jitter_bar")

#: Allowed values for style settings represented by closed selections.
#: Spine and line-style choices are derived separately by
#: :func:`style_choices`; keys absent from these mappings are free-form.
STYLE_CHOICES = {
    "palette": ("colorblind", "deep", "muted", "pastel", "bright", "dark"),
    "format": ("pdf", "png", "svg"),
    "colormap": ("viridis", "plasma", "inferno", "magma", "cividis",
                 "coolwarm", "RdBu_r"),
    "bins": ("auto", "sturges", "fd", "scott", "sqrt"),
    "error_bars": ("sem", "sd", "ci95", "none"),
    "aspect": ("equal", "auto"),
}

#: Style keys that accept a Matplotlib line-style value.
LINE_STYLE_KEYS = ("grid_style", "threshold_style", "reference_style")

#: Matplotlib line-style values accepted by :data:`LINE_STYLE_KEYS`.
LINE_STYLE_CHOICES = ("-", "--", "-.", ":")


def style_choices(name: str) -> tuple:
    """Return the allowed values for a closed-choice style setting.

    Parameters
    ----------
    name : str
        Style key from :data:`GENERAL_DEFAULTS` or
        :data:`GRAPH_DEFAULTS`.

    Returns
    -------
    tuple
        Allowed values. An empty tuple indicates that the setting is
        free-form or unknown.
    """
    if name == "spines":
        return tuple(SPINE_PRESETS)
    if name in LINE_STYLE_KEYS:
        return LINE_STYLE_CHOICES
    return tuple(STYLE_CHOICES.get(name, ()))

#: Spine presets as ``(top, right, bottom, left)`` visibility flags.
SPINE_PRESETS = {
    "all": (True, True, True, True),
    "left_bottom": (False, False, True, True),
    "none": (False, False, False, False),
}


def resolve(kind: Optional[str] = None,
            general: Optional[Mapping[str, Any]] = None,
            overrides: Optional[Mapping[str, Any]] = None) -> dict:
    """Resolve the effective style for a graph type.

    Settings are merged in this order: general defaults, user-defined general
    settings, graph-type defaults, and user-defined graph-type overrides.
    Entries whose value is ``None`` do not replace an earlier value.

    Parameters
    ----------
    kind : str, optional
        Graph type from :data:`GRAPH_KINDS`. If ``None``, only the general
        layers are applied.
    general : mapping of str to Any, optional
        User-defined general settings.
    overrides : mapping of str to mapping, optional
        User-defined settings keyed by graph type.

    Returns
    -------
    dict
        Merged style settings. Unknown graph types inherit only the general
        layers and any matching entry in ``overrides``.
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
    """Convert spaCR style settings to Matplotlib ``rcParams``.

    Settings without an equivalent Matplotlib parameter, such as ``per_row``
    and ``label_top_n``, are omitted.

    Parameters
    ----------
    style : mapping of str to Any
        Resolved or partial spaCR figure style.

    Returns
    -------
    dict
        Matplotlib parameter names and values derived from ``style``.
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
    """Apply a resolved spaCR style to Matplotlib.

    Matplotlib parameters and the colour cycle are updated when their optional
    dependencies are available. Styling failures are ignored so that figure
    generation can continue.

    Parameters
    ----------
    kind : str, optional
        Graph type from :data:`GRAPH_KINDS`.
    general : mapping of str to Any, optional
        User-defined general settings.
    overrides : mapping of str to mapping, optional
        User-defined settings keyed by graph type.

    Returns
    -------
    dict
        Fully resolved spaCR style, including settings that have no
        Matplotlib ``rcParam`` equivalent.
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

#: Supported export modes: ``print`` uses a light background and dark figure
#: elements; ``screen`` preserves the displayed appearance; ``transparent``
#: removes the background and chooses figure-element colours from the theme.
SAVE_MODES = ("print", "screen", "transparent")

#: Background and figure-element colours used for print-mode exports.
PRINT_GROUND = "#FFFFFF"
PRINT_INK = "#222222"

#: Gridline colour used when a print-mode grid needs additional contrast.
PRINT_GRID = "#DDDDDD"

#: WCAG contrast threshold used to identify non-data elements for recolouring.
CHROME_CONTRAST_FLOOR = 2.0

#: Contrast ratio below which unchanged data colours are reported to the user.
#: The threshold is below the lightest colours in spaCR's default figure style
#: so standard palettes do not produce routine warnings.
DATA_CONTRAST_FLOOR = 1.8

_NAMED_COLOURS = {
    "white": (1.0, 1.0, 1.0), "black": (0.0, 0.0, 0.0),
    "none": None, "transparent": None,
}


def to_rgb(colour) -> Optional[tuple]:
    """Convert a colour specification to RGB components.

    Parameters
    ----------
    colour : Any
        Hexadecimal, named, RGB, or RGBA colour specification.

    Returns
    -------
    tuple of float or None
        Three RGB components. Hexadecimal and Matplotlib colour inputs are
        normalized to the interval ``[0, 1]``; numeric sequences are returned
        as floats. ``None`` is returned for transparent or unrecognized
        colours.
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
    """Calculate the WCAG relative luminance of a colour.

    Parameters
    ----------
    colour : Any
        Colour specification accepted by :func:`to_rgb`.

    Returns
    -------
    float or None
        Relative luminance in the interval ``[0, 1]``, or ``None`` when the
        colour is transparent or cannot be parsed.
    """
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
    """Calculate the WCAG contrast ratio between two colours.

    Parameters
    ----------
    colour : Any
        First colour specification.
    other : Any
        Second colour specification.

    Returns
    -------
    float or None
        Contrast ratio from ``1.0`` to ``21.0``. ``None`` is returned when
        either colour is transparent or cannot be parsed.
    """
    first = relative_luminance(colour)
    second = relative_luminance(other)
    if first is None or second is None:
        return None
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def is_legible_on(colour, ground, floor: float = CHROME_CONTRAST_FLOOR) -> bool:
    """Determine whether a colour meets a contrast threshold.

    Parameters
    ----------
    colour : Any
        Foreground colour specification.
    ground : Any
        Background colour specification.
    floor : float, default=CHROME_CONTRAST_FLOOR
        Minimum accepted WCAG contrast ratio.

    Returns
    -------
    bool
        ``True`` when the ratio meets ``floor``. Unrecognized and transparent
        colours are treated as legible so they are not recoloured.
    """
    ratio = contrast_ratio(colour, ground)
    return True if ratio is None else ratio >= float(floor)


class SavedFigureAppearance(NamedTuple):
    """Describe how a figure should be rendered during export.

    Parameters
    ----------
    mode : str
        Active mode from :data:`SAVE_MODES`.
    ground : str or None
        Export background, or ``None`` to retain or remove the current
        background according to ``mode``.
    ink : str or None
        Replacement colour for low-contrast non-data elements, or ``None``
        to preserve their colours.
    grid : str or None
        Replacement colour for low-contrast gridlines.
    transparent : bool
        Whether the figure writer should request a transparent background.
    flip : bool
        Whether low-contrast figure elements may be recoloured.
    """

    mode: str
    ground: Optional[str]
    ink: Optional[str]
    grid: Optional[str]
    transparent: bool
    flip: bool


#: Figure-element categories used during export. ``ground`` covers figure,
#: axes, and legend backgrounds; ``grid`` covers gridlines; ``chrome`` covers
#: labels, ticks, spines, annotations, and reference lines; ``data`` covers
#: marks that encode results and are therefore never recoloured automatically.
ARTIST_KINDS = ("ground", "grid", "chrome", "data")


def export_colour(current, kind: str, look=None) -> Optional[str]:
    """Choose an export replacement colour for a figure element.

    Data colours are always preserved. A dark background may be replaced in
    print mode; gridlines and other figure elements are replaced only when the
    active export mode allows it and their contrast is below the configured
    threshold.

    Parameters
    ----------
    current : Any
        Current artist colour. Transparent and unrecognized values are left
        unchanged.
    kind : {'ground', 'grid', 'chrome', 'data'}
        Role of the artist in the figure.
    look : SavedFigureAppearance, optional
        Export appearance. If ``None``, use
        :func:`saved_figure_appearance`.

    Returns
    -------
    str or None
        Replacement colour, or ``None`` when the current colour should be
        preserved.

    Examples
    --------
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
    """Find data colours with insufficient contrast against a background.

    Parameters
    ----------
    colours : iterable of colour specifications
        Colours accepted by :func:`to_rgb`. Unrecognized values and numeric
        RGBA entries with alpha below ``0.5`` are ignored.
    ground : Any, default=PRINT_GROUND
        Background colour used for the contrast calculation.
    floor : float, optional
        Minimum accepted contrast ratio. If ``None``, use
        :data:`DATA_CONTRAST_FLOOR`.

    Returns
    -------
    list of str
        Sorted, deduplicated colours in ``#RRGGBB`` format.

    Notes
    -----
    This function reports low-contrast data colours but does not replace
    them, because colour may encode a result or category.
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
    """Format a warning for low-contrast data colours.

    Parameters
    ----------
    names : iterable of str
        Colour names returned by :func:`illegible_colours`.

    Returns
    -------
    str
        Warning text, or an empty string when ``names`` is empty.
    """
    if not names:
        return ""
    return ("Saved-figure warning: these data colours have almost no contrast "
            "on the light page and are NOT being changed, because the colour "
            f"is the claim: {', '.join(names)}. Pick an accessible palette in "
            "Preferences > Figures if the marks are meant to be read.")


def figure_save_mode() -> str:
    """Return the configured figure export mode.

    A valid ``SPACR_FIGURE_SAVE_MODE`` value takes precedence over the Qt
    preference store, which allows command-line and notebook workflows to
    choose a mode without starting the GUI. Missing or invalid environment
    values fall through to the stored preference; if no valid preference is
    available, the mode is ``'print'``.

    Returns
    -------
    {'print', 'screen', 'transparent'}
        Active export mode.
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



#: Figure-element and grid colours used for transparent exports in dark themes.
DARK_INK = "#EDEDED"
DARK_GRID = "#4A4A4A"


def theme_ink() -> Tuple[str, str]:
    """Return figure-element colours for the active application theme.

    Returns
    -------
    ink : str
        Colour for labels, ticks, spines, and annotations.
    grid : str
        Colour for gridlines.

    Notes
    -----
    Light themes use :data:`PRINT_INK` and :data:`PRINT_GRID`. Dark themes use
    :data:`DARK_INK` and :data:`DARK_GRID`. The light-theme pair is returned
    when the Qt preference store is unavailable.
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
    """Resolve the background and figure-element colours for export.

    Parameters
    ----------
    mode : {'print', 'screen', 'transparent'}, optional
        Export mode. If ``None``, use :func:`figure_save_mode`. Invalid values
        fall back to ``'print'``.

    Returns
    -------
    SavedFigureAppearance
        Rendering instructions shared by the Matplotlib and pyqtgraph export
        paths. Data colours are outside this appearance and remain unchanged.
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
