"""Reusable custom widgets for the spacr Qt GUI."""
from .ai_chat_panel import AIChatPanel
from .ai_toggle_label import AiToggleLabel
from .card import Card
from .column_picker import (ColumnPickerButton, ColumnPickerDialog,
                             attach_column_picker)
from .console_panel import ConsolePanel
from .data_filter_panel import DataFilterPanel
from .divider import Divider
from .eliding import ElidingLabel, ElidingPushButton
from .empty_state import EmptyState
from .figure_queue import FigureQueue
# Imported here, not lazily: it registers its QSS block through
# `theme.register_widget_qss`, and `launch()` builds the stylesheet before the
# first window. A block registered after that call is missing from the
# stylesheet the application is actually given.
from .graph_builder import ColumnWell, DropZone, GraphBuilderPanel, GraphCanvas
from .hover_tooltip import HoverTooltip
from .info_link import InfoLink
from .umap_explorer import ImageUmapExplorer
from .live_preview import LivePreviewPanel
from .section import Section
from .tile import HTile, Tile
from .toggle import Toggle
from .usage_bar import UsageBar

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
