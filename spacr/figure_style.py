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

from typing import Any, Mapping, Optional

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
