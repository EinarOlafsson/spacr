'Build publication-ready spaCR panels and multi-panel figure sheets.\n\n    from spacr.figures import build_sheet, build_panel, figure_style\n\n    sheet = build_sheet(results)          # every panel, as one figure\n    sheet.figure.savefig(path)\n    print(sheet.legend())                 # the legend, generated from it\n\nThe shared palette and layout are derived from published apicomplexan figures.\nUse :mod:`spacr.figures.style` for the style contract and\n:mod:`spacr.figures.panels` for the panel catalog.'

from importlib import import_module

_EXPORTS = {'FAST_PANELS': ('fast_render', 'FAST_PANELS'), 'RenderedPanel': ('fast_render', 'RenderedPanel'), 'render_panel': ('fast_render', 'render_panel'), 'renderer_for': ('fast_render', 'renderer_for'), 'write_panels': ('fast_render', 'write_panels'), 'render_bundle': ('headless', 'render_bundle'), 'render_offscreen': ('headless', 'render_offscreen'), 'REGISTRY': ('panels', 'REGISTRY'), 'SHEET_ORDER': ('panels', 'SHEET_ORDER'), 'Panel': ('panels', 'Panel'), 'available': ('panels', 'available'), 'effect_column': ('panels', 'effect_column'), 'p_column': ('panels', 'p_column'), 'q_column': ('panels', 'q_column'), 'statistic_column': ('panels', 'statistic_column'), 'SceneReport': ('scene', 'SceneReport'), 'render_figure': ('scene', 'render_figure'), 'scene_renderer': ('scene', 'scene_renderer'), 'write_figure': ('scene', 'write_figure'), 'Sheet': ('sheet', 'Sheet'), 'build_panel': ('sheet', 'build_panel'), 'build_sheet': ('sheet', 'build_sheet'), 'ROLES': ('style', 'ROLES'), 'Palette': ('style', 'Palette'), 'TYPE_SCALE': ('style', 'TYPE_SCALE'), 'figure_style': ('style', 'figure_style'), 'panel_letter': ('style', 'panel_letter'), 'rc': ('style', 'rc'), 'theme_target': ('style', 'theme_target')}

__all__ = ['FAST_PANELS', 'Palette', 'Panel', 'REGISTRY', 'ROLES', 'RenderedPanel', 'SHEET_ORDER', 'Sheet', 'SceneReport', 'TYPE_SCALE', 'available', 'build_panel', 'build_sheet', 'effect_column', 'figure_style', 'p_column', 'render_bundle', 'render_offscreen', 'panel_letter', 'q_column', 'rc', 'render_figure', 'render_panel', 'renderer_for', 'scene_renderer', 'statistic_column', 'theme_target', 'write_figure', 'write_panels']


def __getattr__(name):
    """Load public figure components on demand so shared style tables stay engine independent."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module("." + module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    """Expose the established public exports to introspection without eagerly loading renderers."""
    return sorted(set(globals()) | set(_EXPORTS))
