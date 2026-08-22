"""Build publication-ready spaCR panels and multi-panel figure sheets.

    from spacr.figures import build_sheet, build_panel, figure_style

    sheet = build_sheet(results)          # every panel, as one figure
    sheet.figure.savefig(path)
    print(sheet.legend())                 # the legend, generated from it

The shared palette and layout are derived from published apicomplexan figures.
Use :mod:`spacr.figures.style` for the style contract and
:mod:`spacr.figures.panels` for the panel catalog.
"""

from .fast_render import (FAST_PANELS, RenderedPanel, render_panel,
                          renderer_for, write_panels)
from .headless import render_bundle, render_offscreen
from .panels import (REGISTRY, SHEET_ORDER, Panel, available, effect_column,
                     p_column, q_column, statistic_column)
from .scene import (SceneReport, render_figure, scene_renderer,
                    write_figure)
from .sheet import Sheet, build_panel, build_sheet
from .style import (ROLES, Palette, TYPE_SCALE, figure_style, panel_letter,
                    rc, theme_target)

__all__ = [
    "FAST_PANELS", "Palette", "Panel", "REGISTRY", "ROLES", "RenderedPanel",
    "SHEET_ORDER", "Sheet", "SceneReport", "TYPE_SCALE", "available",
    "build_panel", "build_sheet", "effect_column", "figure_style", "p_column",
    "render_bundle", "render_offscreen",
    "panel_letter", "q_column", "rc", "render_figure", "render_panel",
    "renderer_for", "scene_renderer", "statistic_column", "theme_target",
    "write_figure", "write_panels",
]
