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
    "font_family": "Open Sans",
    "font_size": 11.0,
    "title_size": 13.0,
    "label_size": 11.0,
    "tick_size": 9.0,
    "palette": "colorblind",
    "background": "none",
    "foreground": "#222222",
    "grid": True,
    "grid_colour": "#DDDDDD",
    "grid_width": 0.6,
    "grid_style": "-",
    "spines": "left_bottom",
    "spine_width": 1.0,
    "marker_size": 28.0,
    "line_width": 1.4,
    "dpi": 150,
    "format": "pdf",
    "tight_layout": True,

    "chrome_colour": "",

    "mark_colouring": "group",
    "marker_style": "o",
    "page_shape": "landscape",
}


def chrome_of(style, element: str = "grid") -> str:
    """Resolve the color of a grid, spine, or box element.

    :param style: figure-style mapping containing chrome color overrides.

    A nonempty element-specific value takes precedence over
    ``chrome_colour``. An empty result indicates that the resolved figure ink
    color should be used.
    """
    style = dict(style or {})
    specific = str(style.get(f"{element}_colour", "") or "").strip()
    if specific:
        return specific
    return str(style.get("chrome_colour", "") or "").strip()

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
        "legend": False,
    },
    "plate_heatmap": {
        "colormap": "viridis",
        "centred": False,
        "annotate_cells": False,
        "per_row": 2,
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

    "mark_colouring": ("group", "uniform", "random"),
    "marker_style": ("o", "s", "^", "D", "v", "P", "X", "*"),

    "page_shape": ("square", "portrait", "landscape", "wide", "custom"),
}

#: width:height for each named page shape. `custom` has none -- it means
#: "use the inches", which is the escape hatch the ratio does not remove.
PAGE_SHAPES: dict = {
    "square": 1.0,
    "portrait": 3.0 / 4.0,
    "landscape": 4.0 / 3.0,
    "wide": 16.0 / 9.0,
}

#: Figure width in inches that a named page shape is measured against when no
#: caller supplies its own. It is Matplotlib's own default width, so the
#: default shape resolves to the size a figure already had and only a CHANGED
#: shape moves it.
PAGE_WIDTH_IN: float = 6.4


def page_size(shape: str, width: float) -> tuple:
    """Calculate figure dimensions for a named page shape.

    Parameters
    ----------
    shape : {"square", "portrait", "landscape", "wide"}
        Aspect-ratio preset.
    width : float
        Figure width in inches.

    Returns
    -------
    tuple of float
        ``(width, height)`` in inches.

    Raises
    ------
    KeyError
        If ``shape`` has no fixed ratio, including ``"custom"``.
    """
    return (float(width), float(width) / PAGE_SHAPES[str(shape)])

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
    from .figure_font import use_open_sans_for_figures
    use_open_sans_for_figures()

    spines = SPINE_PRESETS.get(str(style.get("spines", "all")),
                               SPINE_PRESETS["all"])
    params = {
        "font.family": style.get("font_family", "Open Sans"),
        "font.size": float(style.get("font_size", 11.0)),
        "axes.titlesize": float(style.get("title_size", 13.0)),
        "axes.labelsize": float(style.get("label_size", 11.0)),
        "xtick.labelsize": float(style.get("tick_size", 9.0)),
        "ytick.labelsize": float(style.get("tick_size", 9.0)),
        "figure.facecolor": style.get("background", "none"),
        "axes.facecolor": style.get("background", "none"),
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

    spine_ink = chrome_of(style, "spine")
    if spine_ink:
        params["axes.edgecolor"] = spine_ink
    tick_ink = chrome_of(style, "tick")
    if tick_ink:
        params["xtick.color"] = tick_ink
        params["ytick.color"] = tick_ink
    frame_ink = str(style.get("chrome_colour", "") or "").strip()
    grid_ink = str(style.get("grid_colour", "") or "").strip()
    if frame_ink and grid_ink in ("", GENERAL_DEFAULTS["grid_colour"]):
        params["grid.color"] = frame_ink

    marker = str(style.get("marker_style", "") or "").strip()
    if marker and marker != GENERAL_DEFAULTS["marker_style"]:
        params["lines.marker"] = marker

    shape = str(style.get("page_shape", "") or "").strip()
    if shape in PAGE_SHAPES and shape != GENERAL_DEFAULTS["page_shape"]:
        params["figure.figsize"] = list(page_size(shape, PAGE_WIDTH_IN))
    colours = _marks_coloured_by(palette_colours(style.get("palette")),
                                style.get("mark_colouring"))
    if colours:
        from cycler import cycler

        params["axes.prop_cycle"] = cycler(color=colours)
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
            _apply_palette(palette, style.get("mark_colouring"))
    except Exception:
        pass
    return style


def palette_colours(name: Optional[str]) -> list:
    """Return a palette as hexadecimal colour strings.

    Named seaborn palettes are used when they resolve successfully; otherwise,
    spaCR's built-in palette is used. An empty name, or failure to load either
    source, returns an empty list so callers can preserve the current colour
    cycle.

    :param name: Palette name from ``STYLE_CHOICES["palette"]``, or ``None``.
    :returns: The resolved sequence of hexadecimal colours.
    """
    if not name:
        return []
    try:
        import seaborn as sns
        return list(sns.color_palette(str(name)).as_hex())
    except Exception:
        try:
            from .qt.widgets.fast_plots import PALETTE
            return list(PALETTE)
        except Exception:
            return []


RANDOM_MARK_SEED = 20260925


def _marks_coloured_by(colours, rule: Optional[str] = "group") -> list:
    """The colour cycle a figure's groups take under ``mark_colouring``.

    The cycle is what every grouped renderer draws from when it does not
    name a colour itself -- one artist per group, each taking the next
    colour -- so the rule reaches a figure through ``axes.prop_cycle`` the
    same way the palette does.

    * ``group``: the palette in its order, one colour per group.
    * ``uniform``: the palette's first colour and nothing else, so every
      group is drawn in the same ink.
    * ``random``: the palette reordered by a FIXED seed. Neighbouring groups
      stop taking adjacent palette colours, which is what telling them apart
      by eye in a working view needs; the seed is fixed so that redrawing
      the same figure gives the same colours.

    :param colours: the resolved palette, as :func:`palette_colours`
        returns it.
    :param rule: one of ``STYLE_CHOICES["mark_colouring"]``; anything else
        is read as ``group``.
    :returns: the colours for the cycle; empty when ``colours`` is empty, so
        an empty palette still leaves the current cycle alone whatever the
        rule.
    """
    colours = list(colours or [])
    chosen = str(rule or "group").strip().lower()
    if chosen == "uniform":
        return colours[:1]
    if chosen == "random" and len(colours) > 1:
        import random

        shuffled = list(colours)
        random.Random(RANDOM_MARK_SEED).shuffle(shuffled)
        if shuffled == colours:
            shuffled = shuffled[1:] + shuffled[:1]
        return shuffled
    return colours


def _apply_palette(name: str, mark_colouring: Optional[str] = "group") -> None:
    """Set the colour cycle. seaborn's names when it is installed, else ours.

    :param name: the palette name.
    :param mark_colouring: the ``mark_colouring`` rule the cycle follows; see
        :func:`_marks_coloured_by`.
    """
    import matplotlib as mpl
    from cycler import cycler

    colours = _marks_coloured_by(palette_colours(name), mark_colouring)
    if colours:
        mpl.rcParams["axes.prop_cycle"] = cycler(color=colours)



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
                    return None
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
    try:
        from matplotlib.colors import to_rgba

        red, green, blue, alpha = to_rgba(colour)
    except Exception:
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
        return None
    page = look.ground or PRINT_GROUND
    if kind == "ground":
        luminance = relative_luminance(current)
        if look.ground is None or luminance is None or luminance >= 0.5:
            return None
        return look.ground
    if is_legible_on(current, page):
        return None
    replacement = look.grid if kind == "grid" else look.ink
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
    try:
        from .qt import preferences

        stored = str(preferences.get_figure_save_mode()).strip().lower()
        if stored in SAVE_MODES:
            return stored
    except Exception:
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
        ink, grid = theme_ink()
        return SavedFigureAppearance("transparent", None, ink, grid,
                                     True, True)
    return SavedFigureAppearance("print", PRINT_GROUND, PRINT_INK,
                                 PRINT_GRID, False, True)
