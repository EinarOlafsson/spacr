"""Widgets that used to freeze the dark palette now follow Preferences."""
from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pytest


def test_qt_app_import_has_no_frozen_palette_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        import spacr.qt.app  # noqa: F401


def test_no_qt_module_imports_the_frozen_dark_palette():
    import spacr.qt

    root = Path(spacr.qt.__file__).resolve().parent
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "theme.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(
                    alias.name == "PALETTE" for alias in node.names):
                offenders.append(str(path.relative_to(root)))
    assert offenders == []


def test_toggle_paints_with_current_palette(qtbot, monkeypatch):
    from PySide6.QtGui import QImage
    from spacr.qt.widgets import toggle

    colors = {
        "accent": "#123456",
        "surface_alt": "#654321",
        "border": "#abcdef",
        "fg": "#fedcba",
    }
    monkeypatch.setattr(toggle, "active_palette", lambda: colors)
    widget = toggle.Toggle("test")
    qtbot.addWidget(widget)
    widget.resize(100, 30)

    image = QImage(widget.size(), QImage.Format_ARGB32)
    image.fill(0)
    widget.render(image)

    assert image.pixelColor(20, 15).name() == colors["surface_alt"]


def test_hover_tooltip_refreshes_after_theme_change(qtbot, monkeypatch):
    from spacr.qt.widgets import hover_tooltip

    current = {
        "surface_alt": "#112233",
        "border": "#223344",
        "fg": "#eeeeee",
    }
    monkeypatch.setattr(
        hover_tooltip, "active_palette", lambda: current.copy())
    tip = hover_tooltip.HoverTooltip()
    qtbot.addWidget(tip)
    assert "#112233" in tip.styleSheet()

    current["surface_alt"] = "#445566"
    anchor = hover_tooltip.QWidget()
    qtbot.addWidget(anchor)
    tip.show_for(anchor, "Help")
    assert "#445566" in tip.styleSheet()
