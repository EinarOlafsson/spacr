"""Shared figure-style values and rendering conventions.

Figure-specific style dataclasses inherit :class:`FigureStyle` so common
controls have the same name and can be copied between plots. Renderers use
the signature ``render(data, style, *, figure=None, save_path=None)`` and
return ``(figure, axes)``. Passing ``figure`` redraws an existing canvas;
passing ``save_path`` also writes the result with spaCR's export settings.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar, Dict, Optional, Tuple

#: Axis scales any figure may use.
SCALES: Tuple[str, ...] = ("linear", "log", "symlog", "logit")

#: Where a grid may be drawn.
GRID_AXES: Tuple[str, ...] = ("x", "y", "both", "none")


@dataclass
class FigureStyle:
    """Appearance settings shared by every spaCR figure.

    Plot-specific options, such as an effect-size threshold or a colour-by
    column, belong on the corresponding subclass. Keeping only portable
    values here allows a saved house style to be applied across plot types.
    """

    # ---- axes ---------------------------------------------------------
    x_label: str = ""
    y_label: str = ""
    title: str = ""
    x_scale: str = "linear"
    y_scale: str = "linear"
    x_lim: Optional[Tuple[float, float]] = None
    y_lim: Optional[Tuple[float, float]] = None
    invert_x: bool = False
    invert_y: bool = False

    # ---- type ---------------------------------------------------------
    font_family: str = "sans-serif"
    font_size: float = 10.0
    title_font_size: float = 12.0
    label_font_size: float = 8.0
    tick_font_size: float = 9.0
    font_weight: str = "normal"

    # ---- the page -----------------------------------------------------
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
    """``VolcanoStyle`` -> ``'volcano'``, from the CLASS.

    Derived rather than declared, because a caller naming its own kind would
    eventually name two the same and one lab's house style would land on
    another figure type. Duplicated from `fast_plots.style_kind` deliberately
    -- this module imports no Qt, so a headless caller can ask.
    """
    name = type(style).__name__
    if name.endswith("Style"):
        name = name[:-len("Style")]
    return name.lower() or "figure"


def apply_page(figure, axes, style: FigureStyle) -> None:
    """Put the shared half of ``style`` onto a drawn figure.

    THE SHARED HALF ONLY. A renderer draws its own marks and then calls this
    for the furniture, so the grid, the type scale, the spines and the page
    are done once rather than once per figure type -- which is the actual
    payoff of the base class, and the reason two figures cannot drift on what
    "grid off" means.
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

    # THE GRID IS OFF WHEN IT IS OFF. matplotlib warns -- "First parameter to
    # grid() is false, but line properties are supplied. The grid will be
    # enabled." -- and then enables it, which is how a "draw a grid" tick box
    # drew one whichever way it was set. The same fault was found and fixed
    # in the save dialog; it is spelled once here so a third renderer cannot
    # meet it again.
    wanted = bool(style.grid) and str(style.grid_axis) != "none"
    if wanted:
        axes.grid(True, axis=str(style.grid_axis or "y"),
                  color=style.grid_color, linewidth=style.grid_width)
        axes.set_axisbelow(True)
    else:
        axes.grid(False)
    if style.hide_top_right_spines:
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
    if str(style.background_color or "none") != "none":
        figure.patch.set_facecolor(style.background_color)
        axes.set_facecolor(style.background_color)


def write(figure, save_path, style: FigureStyle) -> str:
    """Write ``figure`` to ``save_path`` through the one writer (108 point 6).

    The CALLER'S extension wins -- `save_path` is a filename someone chose --
    and everything else `save_figure` decides is gained: the resolution rule,
    the TrueType embedding, and the repaint for paper.
    """
    import os

    from .plot import save_figure

    path = os.fspath(save_path)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    suffix = os.path.splitext(path)[1].lstrip(".").lower() or None
    raster = suffix in ("png", "jpg", "jpeg", "tif", "tiff")
    return save_figure(figure, path, fmt=suffix,
                       dpi=style.dpi if raster else None,
                       transparent=style.transparent, bbox_inches="tight")
