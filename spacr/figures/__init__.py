"""spaCR's figure system: one house style, one panel catalog, one sheet.

Asked for on 2026-08-16 -- "the figures themselves need to be remade from
scratch ... i want the graphs presented in a beautifull organized way where
the user has controll over appearence. the all figures section should look
like a publication ready figure".

    from spacr.figures import build_sheet, build_panel, figure_style

    sheet = build_sheet(results)          # every panel, as one figure
    sheet.figure.savefig(path)
    print(sheet.legend())                 # the legend, generated from it

The visual system is documented in `.claude/skills/apicomplexan-figures`,
derived from published Lourido-lab figures rather than from design taste.
Read it before adding a panel.
"""

from .panels import (REGISTRY, SHEET_ORDER, Panel, available, effect_column,
                     p_column, q_column, statistic_column)
from .sheet import Sheet, build_panel, build_sheet
from .style import (ROLES, Palette, TYPE_SCALE, figure_style, panel_letter,
                    rc, theme_target)

__all__ = [
    "Palette", "Panel", "REGISTRY", "ROLES", "SHEET_ORDER", "Sheet",
    "TYPE_SCALE", "available", "build_panel", "build_sheet", "effect_column",
    "figure_style", "p_column", "panel_letter", "q_column", "rc",
    "statistic_column", "theme_target",
]
