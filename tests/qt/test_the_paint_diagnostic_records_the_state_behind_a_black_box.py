"""Item 408: the paint diagnostic records the state behind a black box.

The AI chat input and the per-object settings table paint opaque black in the
maintainer's running session and in no probe run outside it, so the state that
produces the box has to be RECORDED where it happens rather than reconstructed.
Launched with ``SPACR_PAINT_DIAG=1``, ``Ctrl+Alt+Shift+D`` writes the paint
state of every visible widget on the current screen, plus the window as the
display holds it, and says where both files went.

These tests pin the recording, not the bug: the key is bound only when asked
for, the dump names the two 408 widgets with their tags exactly as the live
widgets carry them, a staged untagged opaque viewport comes out as a suspect,
and a part that fails does not take the rest of the dump with it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

KEYS = "Ctrl+Alt+Shift+D"


def _settle(qapp, rounds: int = 30) -> None:
    for _ in range(rounds):
        qapp.processEvents()


def _diagnostic_shortcuts(window):
    from PySide6.QtGui import QKeySequence, QShortcut

    return [shortcut for shortcut in window.findChildren(QShortcut)
            if shortcut.key() == QKeySequence(KEYS)]


def _chat_input(screen):
    from PySide6.QtWidgets import QWidget

    return screen.findChild(QWidget, "ConsoleChatInput")


def _window_rect(widget, window):
    from PySide6.QtCore import QPoint

    origin = widget.mapTo(window, QPoint(0, 0))
    return [origin.x(), origin.y(), widget.width(), widget.height()]


def _read(report) -> dict:
    return json.loads(Path(report["files"]["json"]).read_text(encoding="utf-8"))


@pytest.fixture
def mask_window(qapp, qtbot):
    """A real MainWindow on Mask, themed the way `launch` themes it, with the
    per-object table on screen.

    The grid is switched on AFTER Mask is open. Switched on before, a fresh
    Mask builds the table inside a section that is hidden until the form
    asks for it, so it is not visible and a dump of visible widgets would
    not name it. The live switch is the preference path the maintainer uses.
    """
    from spacr.qt import preferences as prefs
    from spacr.qt.app import MainWindow

    grid_was = prefs.get_object_grid_enabled()
    prefs.set_object_grid_enabled(False)
    prefs.apply_preferences_to_app(qapp)
    window = MainWindow()
    qtbot.addWidget(window)
    try:
        window.resize(1400, 900)
        window.show()
        window._on_nav_selected("mask")
        _settle(qapp)
        prefs.set_object_grid_enabled(True)
        window._stack.currentWidget().apply_object_grid_preference()
        _settle(qapp)
        yield window
    finally:
        prefs.set_object_grid_enabled(grid_was)
        window.close()


def test_without_the_variable_no_key_is_bound(qapp, qtbot, monkeypatch):
    """A normal launch carries no trace of the diagnostic."""
    from spacr.qt.app import MainWindow

    monkeypatch.delenv("SPACR_PAINT_DIAG", raising=False)
    window = MainWindow()
    qtbot.addWidget(window)

    assert _diagnostic_shortcuts(window) == []
    assert window._paint_diagnostic_shortcut is None


def test_with_the_variable_one_application_wide_key_is_bound(
        qapp, qtbot, monkeypatch):
    """``SPACR_PAINT_DIAG=1`` binds exactly one key, and it works from any
    window -- a popup can hold focus when the box appears."""
    from PySide6.QtCore import Qt

    from spacr.qt.app import MainWindow

    monkeypatch.setenv("SPACR_PAINT_DIAG", "1")
    window = MainWindow()
    qtbot.addWidget(window)

    bound = _diagnostic_shortcuts(window)
    assert len(bound) == 1
    assert bound[0].context() == Qt.ShortcutContext.ApplicationShortcut
    assert window._paint_diagnostic_shortcut is bound[0]


def test_the_dump_names_the_chat_input_and_the_table_with_their_tags(
        mask_window, qapp, tmp_path, monkeypatch):
    """Both 408 widgets are in the dump, carrying the tags they really carry."""
    from spacr.qt.app import _PAINT_DIAG_KEYS_IN_ORDER, _dump_paint_diagnostics
    from spacr.qt.theme import TRANSPARENT_PROPERTY
    from spacr.qt.widgets.console_panel import ConsolePanel

    said = []
    monkeypatch.setattr(ConsolePanel, "append_stdout",
                        lambda self, text: said.append(text))
    window = mask_window
    screen = window._stack.currentWidget()
    chat = _chat_input(screen)
    table = screen._object_grid._table
    assert chat.isVisible() and table.isVisible(), \
        "both 408 widgets have to be on screen for this to mean anything"

    report = _dump_paint_diagnostics(window, _out_dir=tmp_path)

    written = _read(report)
    assert list(written) == list(_PAINT_DIAG_KEYS_IN_ORDER)
    assert written["errors"] == []
    assert Path(written["files"]["json"]).parent == tmp_path
    png = Path(written["files"]["png"])
    assert png.parent == tmp_path and png.stat().st_size > 0
    assert written["screenshot"]["method"] == \
        "QScreen.grabWindow(window.winId())"
    assert written["qt_platform"] == qapp.platformName()
    assert written["screen"]["app_key"] == "mask"
    assert written["window_backdrop"]["present"] == \
        (window.window_backdrop() is not None)
    assert set(written["environment"]) == {"SPACR_NO_BACKDROP", "SPACR_NO_GL"}
    assert {"ambient_enabled", "ambient_theme", "ambient_palette",
            "pane_opacity", "theme", "tooltips_box_enabled",
            "tooltips_bottom_enabled"} <= set(written["preferences"])
    assert written["stylesheets"]["screen"]["length"] == \
        len(screen.styleSheet())

    found = {}
    for record in written["widgets"]:
        if record["objectName"] == "ConsoleChatInput":
            found["chat"] = record
        elif (record["class"] == "QTableView"
              and "ObjectSettingsGrid" in record["path"]):
            found["table"] = record
    assert set(found) == {"chat", "table"}, "a 408 widget is missing"

    for key, widget in (("chat", chat), ("table", table)):
        record = found[key]
        assert record["geometry"] == _window_rect(widget, window)
        assert record[TRANSPARENT_PROPERTY] == \
            widget.property(TRANSPARENT_PROPERTY)
        assert record["autoFillBackground"] == widget.autoFillBackground()
        for field in ("WA_TranslucentBackground", "WA_OpaquePaintEvent",
                      "WA_NoSystemBackground", "nearest_sheet_ancestor"):
            assert field in record
        assert set(record["styleSheet"]) >= {"length", "sha1"}
        assert set(record["palette"]) == {"Base", "Window"}
        viewport = widget.viewport()
        assert record["viewport"][TRANSPARENT_PROPERTY] == \
            viewport.property(TRANSPARENT_PROPERTY)
        assert record["viewport"]["autoFillBackground"] == \
            viewport.autoFillBackground()
        assert record["viewport"]["palette"]["Base"] == {
            "name": viewport.palette().color(
                viewport.palette().ColorGroup.Active,
                viewport.palette().ColorRole.Base).name(),
            "alpha": viewport.palette().color(
                viewport.palette().ColorGroup.Active,
                viewport.palette().ColorRole.Base).alpha(),
        }

    assert said, "nothing was written to the console"
    assert written["files"]["json"] in said[-1]
    assert written["files"]["png"] in said[-1]


def test_an_untagged_opaque_viewport_is_named_a_suspect(mask_window, tmp_path):
    """Staged: the chat input's viewport loses its tag and fills itself.

    That is the state 408's screenshot implies, and the dump has to put it
    in `suspects` with a measured share of near-black pixels.
    """
    from spacr.qt.app import _dump_paint_diagnostics
    from spacr.qt.theme import TRANSPARENT_PROPERTY

    window = mask_window
    chat = _chat_input(window._stack.currentWidget())

    def reasons(written):
        return [reason for suspect in written["suspects"]
                if suspect["objectName"] == "ConsoleChatInput"
                for reason in suspect["reasons"]]

    before = _read(_dump_paint_diagnostics(window, _out_dir=tmp_path / "a"))
    assert "viewport untagged" not in reasons(before)

    viewport = chat.viewport()
    viewport.setProperty(TRANSPARENT_PROPERTY, False)
    viewport.setAutoFillBackground(True)
    after = _read(_dump_paint_diagnostics(window, _out_dir=tmp_path / "b"))

    assert "viewport untagged" in reasons(after)
    assert "viewport autoFillBackground" in reasons(after)
    suspect = next(s for s in after["suspects"]
                   if s["objectName"] == "ConsoleChatInput")
    assert suspect["reasons"][0].startswith("viewport")
    assert 0.0 <= suspect["near_black_fraction"] <= 1.0


def test_a_part_that_fails_is_named_and_the_rest_is_still_written(
        mask_window, tmp_path, monkeypatch):
    """The diagnostic runs inside the app; it must never raise into it."""
    from spacr.qt.app import _dump_paint_diagnostics

    window = mask_window

    def broken():
        raise RuntimeError("staged")

    monkeypatch.setattr(window, "window_backdrop", broken)
    report = _dump_paint_diagnostics(window, _out_dir=tmp_path)

    written = _read(report)
    assert any("window backdrop" in error for error in written["errors"])
    assert written["widgets"], "one broken part took the widget list with it"
    assert written["screen"]["app_key"] == "mask"
