"""
Central icon lookup for the spacr Qt GUI.

Wraps `qtawesome` so callers stay decoupled from Font Awesome glyph
names, and returns a placeholder QIcon() if qtawesome isn't installed
so the UI still renders (with text-only buttons).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from PySide6.QtGui import QIcon

from .theme import PALETTE


@lru_cache(maxsize=128)
def _try_qta():
    try:
        import qtawesome as qta
        return qta
    except Exception:
        return None


def icon(name: str, color: Optional[str] = None, size: int = 16) -> QIcon:
    """Return a QIcon for the named glyph, or an empty QIcon fallback.

    `name` is a semantic key (e.g. "open", "run", "brush") mapped to a
    Font Awesome glyph. Unknown names fall back to a puzzle piece.
    """
    qta = _try_qta()
    if qta is None:
        return QIcon()
    glyph = _NAME_TO_GLYPH.get(name, "fa5s.puzzle-piece")
    fill = color or PALETTE["fg_muted"]
    try:
        return qta.icon(glyph, color=fill)
    except Exception:
        return QIcon()


def accent_icon(name: str) -> QIcon:
    """Icon painted in the accent color (used for primary buttons)."""
    return icon(name, color=PALETTE["accent"])


def contrast_icon(name: str) -> QIcon:
    """Icon painted in the background color — used inside filled
    (PrimaryButton) buttons where the button bg IS the accent."""
    return icon(name, color=PALETTE["bg"])


# Semantic name → Font Awesome glyph. Keep names short + generic so
# callers don't have to think about the icon library.
_NAME_TO_GLYPH = {
    # File / source
    "open":            "fa5s.folder-open",
    "folder":          "fa5s.folder",
    "file":            "fa5s.file",
    "save":            "fa5s.save",
    "import":          "fa5s.file-import",
    "export":          "fa5s.file-export",
    # Navigation
    "prev":            "fa5s.chevron-left",
    "next":            "fa5s.chevron-right",
    "up":              "fa5s.chevron-up",
    "down":            "fa5s.chevron-down",
    "home":            "fa5s.home",
    "skip":            "fa5s.forward",
    # Editing
    "brush":           "fa5s.paint-brush",
    "erase":           "fa5s.eraser",
    "erase_object":    "fa5s.trash-alt",
    "wand":            "fa5s.magic",
    "wand_add":        "fa5s.plus-circle",
    "wand_erase":      "fa5s.minus-circle",
    "zoom":            "fa5s.search-plus",
    "zoom_reset":      "fa5s.compress-arrows-alt",
    "undo":            "fa5s.undo",
    "redo":            "fa5s.redo",
    "fill":            "fa5s.fill",
    "invert":          "fa5s.adjust",
    "relabel":         "fa5s.tags",
    "remove":          "fa5s.filter",
    "clear":           "fa5s.times-circle",
    # Actions
    "run":             "fa5s.play",
    "stop":            "fa5s.stop",
    "settings":        "fa5s.cog",
    "info":            "fa5s.info-circle",
    "check":           "fa5s.check",
    "warning":         "fa5s.exclamation-triangle",
    "chart":           "fa5s.chart-bar",
    "tag":             "fa5s.tag",
    "search":          "fa5s.search",
    # App keys mirrored from app.py for the sidebar / tiles.
    "mask":            "fa5s.mask",
    "measure":         "fa5s.ruler",
    "annotate":        "fa5s.tag",
    "make_masks":      "fa5s.paint-brush",
    "classify":        "fa5s.layer-group",
    "umap":            "fa5s.project-diagram",
    "ml_analyze":      "fa5s.chart-line",
    "regression":      "fa5s.wave-square",
    "recruitment":     "fa5s.crosshairs",
    "activation":      "fa5s.bolt",
    "analyze_plaques": "fa5s.microscope",
    "train_cellpose":  "fa5s.dumbbell",
    "cellpose_masks":  "fa5s.shapes",
    "cellpose_all":    "fa5s.th",
    "map_barcodes":    "fa5s.barcode",
}
