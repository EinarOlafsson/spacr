"""Linux launch surfaces have an identity and an opaque first frame."""

from __future__ import annotations

import ast
from pathlib import Path

from PySide6.QtCore import Qt


ROOT = Path(__file__).resolve().parents[2]
DESKTOP = ROOT / "packaging/linux/io.github.olafssonlab.spacr.desktop"
APP_ID = "io.github.olafssonlab.spacr"


def test_desktop_entry_matches_qt_application_id():
    text = DESKTOP.read_text(encoding="utf-8")
    assert "Icon=io.github.olafssonlab.spacr" in text
    assert "StartupWMClass=spaCR" in text
    source = (ROOT / "spacr/qt/app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "setDesktopFileName"]
    assert len(calls) == 1
    assert ast.literal_eval(calls[0].args[0]) == APP_ID


def test_main_window_and_loading_cover_are_opaque(qapp, monkeypatch):
    import spacr.qt.app as app_module
    from spacr.qt.widgets.loading_screen import LoadingScreen

    monkeypatch.setattr(app_module.MainWindow, "_install_loading_screen",
                        lambda self: None)
    window = app_module.MainWindow()
    assert window.testAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
    assert window.autoFillBackground()

    cover = LoadingScreen(parent=window)
    assert cover.testAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
    assert cover.autoFillBackground()
