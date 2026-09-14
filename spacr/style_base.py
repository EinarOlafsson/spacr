"""Shared figure-style values and rendering conventions.

Figure-specific style dataclasses inherit :class:`FigureStyle` so common
controls have the same name and can be copied between plots. Renderers use
the signature ``render(data, style, *, figure=None, save_path=None)`` and
return ``(figure, axes)``. Passing ``figure`` redraws an existing canvas;
passing ``save_path`` also writes the result with spaCR's export settings.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from .figure_font import FAMILY as FONT_FAMILY
from .figure_font import use_open_sans_for_figures

#: Axis scales any figure may use.
SCALES: Tuple[str, ...] = ("linear", "log", "symlog", "logit")

#: Where a grid may be drawn.
GRID_AXES: Tuple[str, ...] = ("x", "y", "both", "none")

#: Families matplotlib resolves through its own rcParam lists rather than by
#: name, so they are passed through untouched.
GENERIC_FAMILIES: Tuple[str, ...] = ("serif", "monospace", "cursive",
                                     "fantasy")



@dataclass
class FigureStyle:
    """Appearance settings shared by every spaCR figure.

    Plot-specific options, such as an effect-size threshold or a colour-by
    column, belong on the corresponding subclass. Keeping only portable
    values here allows a saved house style to be applied across plot types.

    :param x_label: horizontal-axis label supplied to the renderer; empty
        permits renderer-specific or default behavior.
    :param y_label: vertical-axis label supplied to the renderer; empty
        permits renderer-specific or default behavior.
    :param title: figure title; empty suppresses the shared title operation.
    :param x_scale: Matplotlib horizontal scale from :data:`SCALES`.
    :param y_scale: Matplotlib vertical scale from :data:`SCALES`.
    :param x_lim: explicit horizontal ``(minimum, maximum)`` limits, or
        ``None`` for data-derived limits.
    :param y_lim: explicit vertical ``(minimum, maximum)`` limits, or ``None``.
    :param invert_x: whether to reverse the horizontal axis after limits apply.
    :param invert_y: whether to reverse the vertical axis after limits apply.
    :param font_family: font-family preference available to renderers that
        support a figure-wide family. Defaults to the face spaCR ships; see
        :func:`font_names`.
    :param font_size: base text size available for plot-specific prose.
    :param title_font_size: title size in points.
    :param label_font_size: axis-label size in points.
    :param tick_font_size: tick-label size in points.
    :param font_weight: shared Matplotlib text weight.
    :param figure_width: figure-canvas width in inches for live and saved use.
    :param figure_height: figure-canvas height in inches for live and saved use.
    :param dpi: dots per inch used for raster output.
    :param grid: whether the selected grid is visible.
    :param grid_axis: axes receiving grid lines: ``"x"``, ``"y"``, ``"both"``,
        or ``"none"``.
    :param grid_color: Matplotlib-compatible grid-line color.
    :param grid_width: grid-line width in points.
    :param hide_top_right_spines: whether to remove the top and right frame
        lines.
    :param legend: whether renderers with keyed marks should draw a legend.
    :param legend_location: Matplotlib legend placement from
        :data:`SHARED_CHOICES`.
    :param background_color: named page and axes color; ``"none"`` delegates
        the background choice to the renderer.
    :param transparent: whether exported figure backgrounds are transparent.
    """

    x_label: str = ""
    y_label: str = ""
    title: str = ""
    x_scale: str = "linear"
    y_scale: str = "linear"
    x_lim: Optional[Tuple[float, float]] = None
    y_lim: Optional[Tuple[float, float]] = None
    invert_x: bool = False
    invert_y: bool = False

    font_family: str = FONT_FAMILY
    font_size: float = 10.0
    title_font_size: float = 12.0
    label_font_size: float = 8.0
    tick_font_size: float = 9.0
    font_weight: str = "normal"

    figure_width: float = 6.2
    figure_height: float = 4.8
    dpi: int = 200
    grid: bool = True
    grid_axis: str = "y"
    grid_color: str = "#E6E6E6"
    grid_width: float = 0.6
    hide_top_right_spines: bool = True
    legend: bool = True
    legend_location: str = "best"
    background_color: str = "none"
    transparent: bool = False

    #: Closed sets, for the menu. Read by `add_style_entries`, which offers a
    #: submenu of these rather than a text box a user can put `lgo` in.
    #:
    #: `ClassVar`, and that is not decoration: annotated as an ordinary type
    #: it becomes a dataclass FIELD, and the restyle menu -- which is built
    #: from `dataclasses.fields(style)` and skips nothing -- would grow an
    #: entry offering to edit the list of choices itself.
    CHOICES: ClassVar[Dict[str, Tuple[str, ...]]] = {}

    def as_dict(self) -> Dict[str, Any]:
        """Every field by name. Portable between styles that share them."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def shared_with(self, other: "FigureStyle") -> Dict[str, Any]:
        """The fields BOTH styles have, so one can be applied to the other.

        :param other: figure-style dataclass whose shared field names define
            the returned values.

        What makes a house style a house style: a font size and a grid
        chosen on a volcano should reach the comparison figure beside it,
        while the volcano's effect-size threshold must not follow it there.
        """
        theirs = {f.name for f in fields(other)}
        return {k: v for k, v in self.as_dict().items() if k in theirs}


#: The closed sets every figure shares. A subclass extends rather than
#: replaces it -- `dict(SHARED_CHOICES, **{...})` -- so a figure type cannot
#: silently lose the general ones.
SHARED_CHOICES: Dict[str, Tuple[str, ...]] = {
    "x_scale": SCALES,
    "y_scale": SCALES,
    "grid_axis": GRID_AXES,
    "font_weight": ("normal", "bold", "light"),
    "legend_location": ("best", "upper right", "upper left", "lower left",
                        "lower right", "right", "center left",
                        "center right", "lower center", "upper center",
                        "center"),
}

FigureStyle.CHOICES = dict(SHARED_CHOICES)


def style_kind(style: Any) -> str:
    """Return a stable figure kind derived from a style class name.

    :param style: style instance whose class name identifies the figure kind.

    For example, ``VolcanoStyle`` becomes ``"volcano"``. Deriving the value
    prevents independently declared names from colliding and keeps this
    headless helper independent of the Qt plotting widgets.
    """
    name = type(style).__name__
    if name.endswith("Style"):
        name = name[:-len("Style")]
    return name.lower() or "figure"


def font_names(style: FigureStyle) -> List[str]:
    """The families to ask matplotlib for, best first, face REGISTERED.

    :param style: figure style whose ``font_family`` is the first choice.
    :returns: a family list suitable for ``font.family``, a ``FontProperties``
        or :meth:`matplotlib.text.Text.set_fontfamily`. A generic family is
        returned alone, because matplotlib resolves it through its own list.

    Naming a family is not the same as having it. Matplotlib answers a name it
    cannot find by falling back -- silently, bar a ``findfont`` warning -- to
    DejaVu Sans, so a style saying "Open Sans" on a machine where Open Sans
    was never installed drew in DejaVu Sans and looked nothing like the
    interface around it. This REGISTERS the faces spaCR ships before naming
    them, which is what makes the name resolve on that machine, and puts
    :data:`FONT_FAMILY` last so a chosen family the machine lacks lands on the
    house face instead of DejaVu Sans.

    Nothing is listed after :data:`FONT_FAMILY`: it is registered from a file
    in the package, so it always resolves, and an unreachable name in the list
    would only make matplotlib warn once per drawn string about a font it was
    never going to use.
    """
    use_open_sans_for_figures()
    requested = str(getattr(style, "font_family", "") or "").strip()
    if requested in GENERIC_FAMILIES:
        return [requested]
    names = [requested] if requested and requested != "sans-serif" else []
    if FONT_FAMILY not in names:
        names.append(FONT_FAMILY)
    return names


def font_rc(style: FigureStyle) -> Dict[str, Any]:
    """The Matplotlib font parameters a style asks for.

    :param style: figure style supplying family, size and weight.
    :returns: ``rcParams`` entries to draw inside, as
        ``matplotlib.rc_context`` takes them.

    Hand this to ``rc_context`` rather than assembling ``font.*`` by hand: the
    family arrives as the resolvable list :func:`font_names` builds, with the
    bundled face already registered.
    """
    names = font_names(style)
    params: Dict[str, Any] = {
        "font.family": list(names),
        "font.size": float(style.font_size),
        "font.weight": str(style.font_weight),
    }
    if names and names[0] not in GENERIC_FAMILIES:
        params["font.sans-serif"] = list(names)
    return params


def apply_page(figure, axes, style: FigureStyle) -> None:
    """Apply shared axes, typography, grid, spine, and page settings.

    :param figure: Matplotlib figure whose page appearance is updated.
    :param axes: Matplotlib axes whose presentation is updated.
    :param style: shared figure-style settings to apply.

    Call this after drawing plot-specific marks. It changes figure and axes
    presentation only; it does not add or remove data marks. The title, the
    axis labels and the tick labels are put into the style's family --
    :func:`font_names` -- so a renderer that draws outside a font
    ``rc_context`` still gets the face spaCR ships rather than DejaVu Sans.
    """
    figure.set_size_inches(float(style.figure_width),
                           float(style.figure_height))
    if style.title:
        axes.set_title(style.title, fontsize=style.title_font_size,
                       fontweight=style.font_weight)
    if style.x_label:
        axes.set_xlabel(style.x_label, fontsize=style.label_font_size)
    if style.y_label:
        axes.set_ylabel(style.y_label, fontsize=style.label_font_size)
    for name, scale in (("x", style.x_scale), ("y", style.y_scale)):
        if scale and scale != "linear":
            try:
                getattr(axes, f"set_{name}scale")(scale)
            except Exception:                                # noqa: BLE001
                continue
    if style.x_lim:
        axes.set_xlim(*style.x_lim)
    if style.y_lim:
        axes.set_ylim(*style.y_lim)
    if style.invert_x:
        axes.invert_xaxis()
    if style.invert_y:
        axes.invert_yaxis()
    axes.tick_params(labelsize=style.tick_font_size)

    # The face, on the text objects themselves. A renderer that drew outside
    # a font `rc_context` would otherwise keep whatever the process-wide
    # default was -- DejaVu Sans, unless something else had changed it.
    names = font_names(style)
    for text in (axes.title, axes.xaxis.label, axes.yaxis.label,
                 *axes.get_xticklabels(), *axes.get_yticklabels()):
        text.set_fontfamily(list(names))

    wanted = bool(style.grid) and str(style.grid_axis) != "none"
    if wanted:
        axes.grid(True, axis=str(style.grid_axis or "y"),
                  color=style.grid_color, linewidth=style.grid_width)
        axes.set_axisbelow(True)
    else:
        axes.grid(False)
    for side in ("top", "right"):
        axes.spines[side].set_visible(not style.hide_top_right_spines)
    if str(style.background_color or "none") != "none":
        figure.patch.set_facecolor(style.background_color)
        axes.set_facecolor(style.background_color)


def write(figure, save_path, style: FigureStyle) -> str:
    """Write a styled figure with spaCR's standard export pipeline.

    :param figure: drawn Matplotlib figure to export.
    :param save_path: destination path; its suffix selects the format.
    :param style: figure-style export settings, including DPI and
        transparency.

    The extension in ``save_path`` selects the format. Raster outputs use the
    style's DPI; font embedding, paper repainting, transparency, and bounding
    box behavior are delegated to :func:`spacr.plot.save_figure`.
    """
    import os

    from .plot import save_figure

    path = os.fspath(save_path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    suffix = os.path.splitext(path)[1].lstrip(".").lower() or None
    raster = suffix in ("png", "jpg", "jpeg", "tif", "tiff")
    return save_figure(figure, path, fmt=suffix,
                       dpi=style.dpi if raster else None,
                       transparent=style.transparent, bbox_inches="tight")
