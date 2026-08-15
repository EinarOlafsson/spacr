"""The complete appearance of a volcano plot, as data rather than arguments.

:class:`VolcanoStyle` holds every setting the interactive explorer exposes, so
one object can be handed to the renderer, saved beside a figure, reloaded, and
replayed to reproduce a plot exactly. The renderer is a plain function of
``(results, style)`` with no Qt in it, which is what lets the same code draw
the headless PDF written by a pipeline run and the live canvas the user clicks
on -- they cannot drift, because there is only one of them.

Splitting the style out this way is also what makes "export exactly what I am
looking at" true: the export path re-renders from the same object at a
different size and dpi rather than screenshotting the widget.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "VolcanoStyle",
    "MARKER_SHAPES",
    "COLORMAPS",
    "FONT_FAMILIES",
    "LINE_STYLES",
    "SCALES",
    "render_volcano",
    "point_details",
]

#: Marker shapes offered, as ``(matplotlib code, label)``. Restricted to the
#: filled shapes, because an unfilled marker cannot carry a colour mapping.
MARKER_SHAPES: tuple[tuple[str, str], ...] = (
    ("o", "Circle"),
    ("s", "Square"),
    ("^", "Triangle up"),
    ("v", "Triangle down"),
    ("<", "Triangle left"),
    (">", "Triangle right"),
    ("D", "Diamond"),
    ("d", "Thin diamond"),
    ("p", "Pentagon"),
    ("h", "Hexagon"),
    ("H", "Hexagon (rotated)"),
    ("8", "Octagon"),
    ("*", "Star"),
    ("P", "Plus (filled)"),
    ("X", "Cross (filled)"),
)

#: Colormaps, grouped by what they are for. A categorical mapping must not be
#: drawn with a sequential map, and a continuous one must not be drawn with a
#: qualitative map, so the explorer picks the default from the column's dtype.
COLORMAPS: dict[str, tuple[str, ...]] = {
    "sequential": ("viridis", "plasma", "inferno", "magma", "cividis",
                   "Blues", "Greens", "Oranges", "Purples", "Reds",
                   "YlGnBu", "YlOrRd"),
    "diverging": ("coolwarm", "RdBu_r", "RdYlBu_r", "BrBG", "PiYG",
                  "PuOr", "Spectral_r", "bwr", "seismic"),
    "qualitative": ("tab10", "tab20", "Set1", "Set2", "Set3", "Dark2",
                    "Paired", "Accent"),
}

FONT_FAMILIES: tuple[str, ...] = (
    "sans-serif", "serif", "monospace", "DejaVu Sans", "DejaVu Serif",
    "DejaVu Sans Mono", "Arial", "Helvetica", "Times New Roman", "Courier New",
)

LINE_STYLES: tuple[tuple[str, str], ...] = (
    ("-", "Solid"), ("--", "Dashed"), ("-.", "Dash-dot"), (":", "Dotted"),
    ("none", "None"),
)

SCALES: tuple[str, ...] = ("linear", "log", "symlog", "logit")


@dataclass
class VolcanoStyle:
    """Every knob the volcano exposes. Defaults reproduce the pipeline plot."""

    # ---- what is plotted -------------------------------------------------
    x_column: str = "standardized_marginal_effect"
    y_column: str = "adjusted_p_value"
    #: -log10 the y column. Off means "plot the value as it is", which is what
    #: you want if the column already holds a -log10 value.
    y_neg_log10: bool = True
    label_column: str = "guide"

    # ---- axes ------------------------------------------------------------
    x_label: str = "Standardized marginal effect"
    y_label: str = ""            # blank means "derive it from the columns"
    title: str = ""
    x_scale: str = "linear"      # any of SCALES
    y_scale: str = "linear"
    x_lim: tuple[float, float] | None = None
    y_lim: tuple[float, float] | None = None
    #: Broken y axis: ``[(lo1, hi1), (lo2, hi2)]`` draws two stacked panels
    #: with a break, for a screen whose hits sit far above the null cloud.
    split_axis: bool = False
    split_y_lims: tuple[tuple[float, float], tuple[float, float]] | None = None
    split_height_ratio: float = 0.35
    invert_x: bool = False
    invert_y: bool = False

    # ---- thresholds ------------------------------------------------------
    alpha: float = 0.05
    #: Effect-size cut. ``None`` draws none. ``threshold_multiplier`` scales
    #: whichever rule ``threshold_method`` names.
    effect_threshold: float | None = None
    threshold_method: str = "value"   # 'value' | 'std' | 'mad' | 'quantile'
    threshold_multiplier: float = 3.0
    show_alpha_line: bool = True
    show_effect_lines: bool = True
    show_zero_line: bool = True

    # ---- marks -----------------------------------------------------------
    marker: str = "o"
    marker_size: float = 26.0
    significant_marker_size: float = 52.0
    marker_alpha: float = 0.85
    marker_edge_width: float = 0.35
    marker_edge_color: str = "#FFFFFF"
    base_color: str = "#B8BDC5"
    significant_color: str = "#D55E00"
    #: Column whose values choose each point's colour. ``None`` uses the
    #: two-tone significant / not-significant scheme.
    color_by: str | None = None
    colormap: str = "viridis"
    color_vmin: float | None = None
    color_vmax: float | None = None
    show_colorbar: bool = True
    #: Column whose values choose each point's SHAPE. Categorical only --
    #: a shape cannot encode a continuous value.
    shape_by: str | None = None

    # ---- lines -----------------------------------------------------------
    line_width: float = 1.0
    line_color: str = "#404040"
    line_style: str = "--"
    zero_line_color: str = "#777777"
    zero_line_width: float = 0.7

    # ---- text ------------------------------------------------------------
    font_family: str = "sans-serif"
    font_size: float = 10.0
    title_font_size: float = 12.0
    label_font_size: float = 8.0
    tick_font_size: float = 9.0
    font_weight: str = "normal"
    #: Guides to annotate: ``{guide id: printed label}``.
    annotations: dict = field(default_factory=dict)
    #: Annotate everything called significant, in addition to `annotations`.
    annotate_significant: bool = False

    # ---- frame -----------------------------------------------------------
    figure_width: float = 6.2
    figure_height: float = 4.8
    dpi: int = 200
    grid: bool = True
    grid_axis: str = "y"          # 'x' | 'y' | 'both' | 'none'
    grid_color: str = "#E6E6E6"
    grid_width: float = 0.6
    hide_top_right_spines: bool = True
    legend: bool = True
    legend_location: str = "best"
    background_color: str = "none"
    transparent: bool = False

    # ------------------------------------------------------------------ i/o

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "VolcanoStyle":
        """Build from a dict, ignoring keys this version does not know.

        Forwards-compatible on purpose: a style saved by a newer spaCR still
        loads here, minus the settings that did not exist yet.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (values or {}).items() if k in known})

    def save(self, path) -> str:
        path = os.fspath(path)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
        return path

    @classmethod
    def load(cls, path) -> "VolcanoStyle":
        with open(os.fspath(path), encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def _resolve_effect_threshold(values: np.ndarray, style: VolcanoStyle):
    """Turn ``threshold_method`` + multiplier into a symmetric cut, or None."""
    method = str(style.threshold_method or "value").lower()
    multiplier = float(style.threshold_multiplier)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    if method == "value":
        if style.effect_threshold is None:
            return None
        return abs(float(style.effect_threshold)) * multiplier
    if method == "std":
        return float(np.std(finite, ddof=1)) * multiplier
    if method == "mad":
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        # 1.4826 makes the MAD a consistent estimator of sigma under normality.
        return mad * 1.4826 * multiplier
    if method == "quantile":
        # The multiplier is the quantile itself here, e.g. 0.99.
        quantile = min(max(multiplier, 0.5), 0.999999)
        return float(np.quantile(np.abs(finite), quantile))
    raise ValueError(
        f"threshold_method={style.threshold_method!r} must be one of "
        f"'value', 'std', 'mad' or 'quantile'.")


def _prepare(results: pd.DataFrame, style: VolcanoStyle):
    """Extract x, y, significance and the label series the plot needs."""
    frame = results.copy()
    if style.x_column not in frame.columns:
        raise ValueError(
            f"x_column={style.x_column!r} is not a column of the results "
            f"({sorted(frame.columns)[:15]}).")
    if style.y_column not in frame.columns:
        raise ValueError(
            f"y_column={style.y_column!r} is not a column of the results "
            f"({sorted(frame.columns)[:15]}).")
    x = pd.to_numeric(frame[style.x_column], errors="coerce").to_numpy(float)
    raw_y = pd.to_numeric(frame[style.y_column], errors="coerce").to_numpy(float)
    if style.y_neg_log10:
        y = -np.log10(np.clip(raw_y, np.finfo(float).tiny, None))
    else:
        y = raw_y
    if "significant" in frame.columns:
        significant = frame["significant"].astype(bool).to_numpy()
    else:
        significant = raw_y < float(style.alpha)
    effect_cut = _resolve_effect_threshold(x, style)
    if effect_cut is not None:
        significant = significant & (np.abs(x) >= effect_cut)
    return frame, x, y, raw_y, significant, effect_cut


def render_volcano(results: pd.DataFrame, style: VolcanoStyle, *,
                   figure=None, save_path=None):
    """Draw the volcano described by ``style`` and return ``(figure, axes)``.

    :param figure: draw into this figure instead of creating one, so a live
        canvas can be redrawn in place.
    :param save_path: also write the figure here. The extension chooses the
        format; ``.pdf`` stays vector.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    frame, x, y, raw_y, significant, effect_cut = _prepare(results, style)

    if figure is None:
        figure = plt.figure(figsize=(style.figure_width, style.figure_height),
                            dpi=style.dpi)
    else:
        figure.clear()

    with mpl.rc_context({
        "font.family": style.font_family,
        "font.size": style.font_size,
        "font.weight": style.font_weight,
    }):
        if style.split_axis and style.split_y_lims:
            lower, upper = style.split_y_lims
            ratio = max(min(float(style.split_height_ratio), 0.9), 0.1)
            axes = figure.subplots(
                2, 1, sharex=True,
                gridspec_kw={"height_ratios": [ratio, 1 - ratio],
                             "hspace": 0.08})
            panels = [axes[0], axes[1]]
            panels[0].set_ylim(*upper)
            panels[1].set_ylim(*lower)
        else:
            panels = [figure.add_subplot(111)]

        mappable = _draw_points(panels, frame, x, y, significant, style)
        _draw_reference_lines(panels, style, effect_cut)
        _annotate(panels, frame, x, y, significant, style)
        _finish_axes(figure, panels, style, mappable)

    if save_path is not None:
        path = os.fspath(save_path)
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        figure.savefig(
            path,
            dpi=style.dpi if path.lower().endswith((".png", ".jpg", ".jpeg",
                                                    ".tif", ".tiff")) else None,
            transparent=style.transparent,
            bbox_inches="tight",
        )
    return figure, panels


def _colour_values(frame: pd.DataFrame, style: VolcanoStyle):
    """Return ``(values, is_categorical)`` for the colour mapping, or None."""
    if not style.color_by or style.color_by not in frame.columns:
        return None
    column = frame[style.color_by]
    numeric = pd.to_numeric(column, errors="coerce")
    # A column that is mostly unparseable is a category, whatever its dtype.
    if numeric.notna().mean() > 0.9:
        return numeric.to_numpy(float), False
    return column.astype(str).to_numpy(), True


def _draw_points(panels, frame, x, y, significant, style):
    """Scatter the points onto every panel; returns a mappable or None."""
    colours = _colour_values(frame, style)
    shapes = {}
    if style.shape_by and style.shape_by in frame.columns:
        categories = list(pd.unique(frame[style.shape_by].astype(str)))
        codes = [code for code, _label in MARKER_SHAPES]
        shapes = {name: codes[index % len(codes)]
                  for index, name in enumerate(categories)}

    mappable = None
    for axis in panels:
        if colours is None:
            groups = [
                (~significant, style.base_color, style.marker_size,
                 "not significant"),
                (significant, style.significant_color,
                 style.significant_marker_size, "significant"),
            ]
            for mask, colour, size, label in groups:
                if not mask.any():
                    continue
                _scatter_by_shape(axis, frame, x, y, mask, style, shapes,
                                  color=colour, size=size, label=label)
        else:
            values, categorical = colours
            if categorical:
                categories = list(pd.unique(values))
                import matplotlib as mpl
                cmap = mpl.colormaps[style.colormap]
                for index, name in enumerate(categories):
                    mask = values == name
                    if not mask.any():
                        continue
                    colour = cmap(index % getattr(cmap, "N", 256)
                                  if cmap.N <= 32 else index / max(len(categories) - 1, 1))
                    _scatter_by_shape(axis, frame, x, y, mask, style, shapes,
                                      color=colour, size=style.marker_size,
                                      label=str(name))
            else:
                sizes = np.where(significant, style.significant_marker_size,
                                 style.marker_size)
                mappable = axis.scatter(
                    x, y, c=values, cmap=style.colormap,
                    vmin=style.color_vmin, vmax=style.color_vmax,
                    s=sizes, marker=style.marker, alpha=style.marker_alpha,
                    edgecolor=style.marker_edge_color,
                    linewidth=style.marker_edge_width)
    return mappable


def _scatter_by_shape(axis, frame, x, y, mask, style, shapes, *, color, size,
                      label):
    """One scatter per shape category, so each can carry its own marker."""
    if not shapes:
        axis.scatter(x[mask], y[mask], s=size, marker=style.marker,
                     color=color, alpha=style.marker_alpha,
                     edgecolor=style.marker_edge_color,
                     linewidth=style.marker_edge_width, label=label)
        return
    values = frame[style.shape_by].astype(str).to_numpy()
    # When colour and shape encode the SAME column, "GRA · GRA" is noise.
    same_source = style.shape_by == style.color_by
    for name, code in shapes.items():
        combined = mask & (values == name)
        if not combined.any():
            continue
        axis.scatter(x[combined], y[combined], s=size, marker=code,
                     color=color, alpha=style.marker_alpha,
                     edgecolor=style.marker_edge_color,
                     linewidth=style.marker_edge_width,
                     label=name if same_source else f"{label} · {name}")


def _draw_reference_lines(panels, style, effect_cut):
    for axis in panels:
        if style.show_alpha_line and style.line_style != "none":
            level = (-np.log10(max(float(style.alpha), np.finfo(float).tiny))
                     if style.y_neg_log10 else float(style.alpha))
            axis.axhline(level, color=style.line_color,
                         linestyle=style.line_style,
                         linewidth=style.line_width)
        if style.show_zero_line:
            axis.axvline(0, color=style.zero_line_color,
                         linewidth=style.zero_line_width)
        if style.show_effect_lines and effect_cut:
            for sign in (-1.0, 1.0):
                axis.axvline(sign * effect_cut, color=style.line_color,
                             linestyle=style.line_style,
                             linewidth=style.line_width)


def _annotate(panels, frame, x, y, significant, style):
    """Print the requested labels, alternating sides so they do not collide."""
    if style.label_column not in frame.columns:
        return
    labels = frame[style.label_column].astype(str).to_numpy()
    wanted: dict[str, str] = {str(k): str(v)
                              for k, v in (style.annotations or {}).items()}
    if style.annotate_significant:
        for index in np.flatnonzero(significant):
            wanted.setdefault(labels[index], labels[index])
    if not wanted:
        return
    rows = [(text, index) for index, name in enumerate(labels)
            if (text := wanted.get(name)) is not None]
    rows.sort(key=lambda item: x[item[1]])
    axis = panels[0]
    for order, (text, index) in enumerate(rows):
        offset, align = ((-5, 8), "right") if order % 2 == 0 else ((5, -15), "left")
        target = axis
        for panel in panels:
            low, high = panel.get_ylim()
            if low <= y[index] <= high:
                target = panel
                break
        target.annotate(text, (x[index], y[index]), xytext=offset,
                        textcoords="offset points",
                        fontsize=style.label_font_size,
                        fontweight="bold", ha=align)


def _finish_axes(figure, panels, style, mappable):
    y_label = style.y_label
    if not y_label:
        y_label = (f"-log10({style.y_column})" if style.y_neg_log10
                   else style.y_column)
    bottom = panels[-1]
    for axis in panels:
        if style.x_scale != "linear":
            axis.set_xscale(style.x_scale)
        if style.y_scale != "linear" and not style.split_axis:
            axis.set_yscale(style.y_scale)
        if style.x_lim:
            axis.set_xlim(*style.x_lim)
        if style.y_lim and not style.split_axis:
            axis.set_ylim(*style.y_lim)
        if style.invert_x:
            axis.invert_xaxis()
        if style.invert_y:
            axis.invert_yaxis()
        if style.grid and style.grid_axis != "none":
            axis.grid(axis=style.grid_axis, color=style.grid_color,
                      linewidth=style.grid_width)
        else:
            axis.grid(False)
        if style.hide_top_right_spines:
            axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=style.tick_font_size)
        if style.background_color and style.background_color != "none":
            axis.set_facecolor(style.background_color)

    if len(panels) == 2:
        # The break marks. Hide the shared edge, then draw the diagonal ticks.
        panels[0].spines["bottom"].set_visible(False)
        panels[1].spines["top"].set_visible(False)
        panels[0].tick_params(bottom=False, labelbottom=False)
        kwargs = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=7,
                      linestyle="none", color="#404040", mec="#404040",
                      mew=1, clip_on=False)
        panels[0].plot([0, 1], [0, 0], transform=panels[0].transAxes, **kwargs)
        panels[1].plot([0, 1], [1, 1], transform=panels[1].transAxes, **kwargs)

    bottom.set_xlabel(style.x_label, fontsize=style.font_size)
    if len(panels) == 2:
        figure.supylabel(y_label, fontsize=style.font_size)
    else:
        bottom.set_ylabel(y_label, fontsize=style.font_size)
    if style.title:
        panels[0].set_title(style.title, fontsize=style.title_font_size,
                            fontweight="bold")
    if style.legend:
        handles, labels = panels[0].get_legend_handles_labels()
        if handles:
            panels[0].legend(handles, labels, frameon=False,
                             loc=style.legend_location,
                             fontsize=style.label_font_size)
    if mappable is not None and style.show_colorbar:
        bar = figure.colorbar(mappable, ax=panels, fraction=0.046, pad=0.02)
        bar.set_label(style.color_by, fontsize=style.label_font_size)
        bar.ax.tick_params(labelsize=style.tick_font_size)
    # tight_layout cannot lay out a broken axis or a figure-level colorbar and
    # warns instead of doing nothing, so it is only run when it applies.
    if not style.split_axis and not (mappable is not None and style.show_colorbar):
        figure.tight_layout()


def point_details(results: pd.DataFrame, index: int, style: VolcanoStyle
                  ) -> dict:
    """Everything known about one plotted point, for the click handler.

    Returns every column of the row, plus the derived plotted coordinates, so
    the panel can show both what was plotted and what it came from.
    """
    row = results.iloc[int(index)]
    detail: dict[str, Any] = {str(k): row[k] for k in results.columns}
    raw_y = pd.to_numeric(pd.Series([row[style.y_column]]),
                          errors="coerce").iloc[0]
    detail["_plotted_x"] = pd.to_numeric(
        pd.Series([row[style.x_column]]), errors="coerce").iloc[0]
    detail["_plotted_y"] = (
        -np.log10(max(float(raw_y), np.finfo(float).tiny))
        if style.y_neg_log10 and pd.notna(raw_y) else raw_y)
    return detail
