"""Reusable custom widgets for the spacr Qt GUI.

`DataFilterPanel` is imported LAZILY, through the `__getattr__` below.
It is the one widget here that imports pandas, and this package's eager
imports meant every other widget paid for it. Measured with
`python -X importtime -c "import spacr.qt.app"`:

    670 ms  spacr.qt.app
    396 ms    spacr.qt.widgets           (via one 40-line label module)
    333 ms      spacr.qt.widgets.data_filter_panel
    238 ms        pandas

All of that lands before the first window is drawn, which is the startup
stutter in instruction 55. `from spacr.qt.widgets import DataFilterPanel`
still works and still returns the same class; it just costs pandas at the
moment something actually asks for it, which is when a data screen opens
and pandas is needed anyway.
"""
from .ai_chat_panel import AIChatPanel
from .ai_toggle_label import AiToggleLabel
from .card import Card
from .column_picker import (ColumnPickerButton, ColumnPickerDialog,
                             attach_column_picker)
from .console_panel import ConsolePanel
from .divider import Divider
from .eliding import ElidingLabel, ElidingPushButton
from .empty_state import EmptyState
# Imported here, not lazily: it registers its QSS block through
# `theme.register_widget_qss`, and `launch()` builds the stylesheet before the
# first window. A block registered after that call is missing from the
# stylesheet the application is actually given.
from .hover_tooltip import HoverTooltip
from .info_link import InfoLink
from .section import Section
from .tile import HTile, Tile
from .toggle import Toggle
from .usage_bar import UsageBar

#: The names that cost real time to import, and the module behind each.
#: Everything else in this package stays eager on purpose: deferring a
#: cheap module trades no measurable time for a class of bug that is much
#: harder to see -- an ImportError that stops happening at import time and
#: starts happening when a user clicks something.
#:
#: These five modules pull pandas and matplotlib between them, which is
#: ~240 ms of a ~670 ms startup, none of it needed before a window exists.
_LAZY = {
    "DataFilterPanel": "data_filter_panel",
    "FigureQueue": "figure_queue",
    "ColumnWell": "graph_builder",
    "DropZone": "graph_builder",
    "GraphBuilderPanel": "graph_builder",
    "GraphCanvas": "graph_builder",
    "ImageUmapExplorer": "umap_explorer",
    "LivePreviewPanel": "live_preview",
}


def __getattr__(name):
    """Import a heavy widget on first use (PEP 562).

    `from spacr.qt.widgets import GraphCanvas` behaves exactly as it did
    and returns the same class; it just pays for pandas and matplotlib at
    the moment something asks, which is when a data or figure screen opens
    and needs them anyway.
    """
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value      # cached; this runs once per name
    return value


def __dir__():
    """Keep the deferred names visible to `dir()` and tab-completion."""
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "AIChatPanel", "AiToggleLabel", "Card",
    "attach_column_picker",
    "ColumnPickerDialog",
    "ColumnPickerButton", "ColumnWell", "ConsolePanel", "DataFilterPanel",
    "Divider", "DropZone",
    "ElidingLabel", "ElidingPushButton",
    "EmptyState", "FigureQueue", "GraphBuilderPanel", "GraphCanvas",
    "HTile", "HoverTooltip", "InfoLink",
    "ImageUmapExplorer", "LivePreviewPanel", "Section",
    "Tile", "Toggle",
    "UsageBar",
]
