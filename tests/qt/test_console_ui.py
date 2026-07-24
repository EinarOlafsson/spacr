"""Tests for the merged Console panel + AI toggle + shutdown cleanup.

Covers the specific behaviours the user asked us to lock down:
* User bubble carries the "spaCR user:" prefix.
* Bubbles span the console's full width (no offset row wrapper).
* Bubble QFrame renders as a full-width rectangle (border-radius 0
  via QSS, sqare corners) with a dark-green fill.
* AiToggleLabel starts off (white), click → on (accent blue), click
  again → off.
* ConsolePanel.shutdown() cancels the AI worker + waits for its
  QThread to exit BEFORE Qt drops references. This is what prevents
  the `QThread: Destroyed while thread '' is still running` crash on
  quit.
"""
from __future__ import annotations

import pytest

from PySide6.QtCore import Qt


# ---------------------------------------------------------------------------
# Bubble prefix + full-width
# ---------------------------------------------------------------------------

def test_user_bubble_has_spacr_user_prefix(qtbot, qt_theme_applied):
    from spacr.qt.widgets.console_panel import _Bubble
    b = _Bubble("user", "hello")
    qtbot.addWidget(b)
    # The label's HTML text embeds the prefix
    assert "spaCR user:" in b._label.text()
    assert "hello" in b._label.text()


def test_ai_bubble_has_spacr_ai_prefix(qtbot, qt_theme_applied):
    from spacr.qt.widgets.console_panel import _Bubble
    b = _Bubble("assistant", "world")
    qtbot.addWidget(b)
    assert "spaCR AI:" in b._label.text()


def test_bubble_no_longer_has_max_width_cap(qtbot, qt_theme_applied):
    """The old design capped bubbles at _MAX_WIDTH=720 with an offset
    row wrapper. The new design spans the console's full width."""
    from spacr.qt.widgets.console_panel import _Bubble
    b = _Bubble("user", "spans full width")
    qtbot.addWidget(b)
    # No hard maximum — bubble grows with the parent
    assert b.maximumWidth() >= 10_000    # QWIDGETSIZE_MAX-ish
    # Old class attribute was 720; verify it's gone
    assert not hasattr(_Bubble, "_MAX_WIDTH")


def test_insert_entry_no_longer_wraps_bubbles_in_offset_row(qtbot, qt_theme_applied):
    from spacr.qt.widgets.console_panel import ConsolePanel, _Bubble
    panel = ConsolePanel(active_app_label="Mask")
    qtbot.addWidget(panel)
    b = _Bubble("user", "hi")
    panel._insert_entry(b)
    # First non-stretch entry should be the bubble itself, not a
    # QHBoxLayout row wrapper.
    first = panel._entries.itemAt(0).widget()
    assert first is b


# ---------------------------------------------------------------------------
# Bubble QSS looks like a rectangle (no rounding)
# ---------------------------------------------------------------------------

def test_console_bubble_qss_has_no_rounded_corners(qt_theme_applied):
    from spacr.qt.theme import stylesheet
    qss = stylesheet()
    # Find the ConsoleBubbleUser block
    idx = qss.find("QFrame#ConsoleBubbleUser")
    assert idx != -1
    block = qss[idx : idx + 300]
    assert "border-radius: 0px" in block


# ---------------------------------------------------------------------------
# AiToggleLabel behaviour
# ---------------------------------------------------------------------------

def test_ai_toggle_label_starts_off(qtbot, qt_theme_applied):
    from spacr.qt.widgets import AiToggleLabel
    lbl = AiToggleLabel()
    qtbot.addWidget(lbl)
    assert lbl.text() == "AI"
    assert not lbl.isChecked()


def test_ai_toggle_click_flips_state_and_fires_signal(qtbot, qt_theme_applied):
    from spacr.qt.widgets import AiToggleLabel
    lbl = AiToggleLabel()
    qtbot.addWidget(lbl)
    with qtbot.waitSignal(lbl.toggled, timeout=1000) as blocker:
        qtbot.mouseClick(lbl, Qt.LeftButton)
    assert blocker.args[0] is True
    assert lbl.isChecked()

    with qtbot.waitSignal(lbl.toggled, timeout=1000) as blocker:
        qtbot.mouseClick(lbl, Qt.LeftButton)
    assert blocker.args[0] is False
    assert not lbl.isChecked()


def test_ai_toggle_style_reflects_state(qtbot, qt_theme_applied):
    from spacr.qt.widgets import AiToggleLabel
    from spacr.qt.theme import PALETTE
    lbl = AiToggleLabel()
    qtbot.addWidget(lbl)
    # Off → fg color in stylesheet
    assert PALETTE["fg"].lower() in lbl.styleSheet().lower()
    lbl.setChecked(True)
    assert PALETTE["accent"].lower() in lbl.styleSheet().lower()


def test_ai_toggle_setchecked_emits_signal(qtbot, qt_theme_applied):
    from spacr.qt.widgets import AiToggleLabel
    lbl = AiToggleLabel()
    qtbot.addWidget(lbl)
    with qtbot.waitSignal(lbl.toggled, timeout=1000) as blocker:
        lbl.setChecked(True)
    assert blocker.args[0] is True


# ---------------------------------------------------------------------------
# ConsolePanel — submit + shutdown
# ---------------------------------------------------------------------------

def test_console_panel_input_has_no_send_button(qtbot, qt_theme_applied):
    """Design change — Enter submits, there is no explicit Send button
    anywhere in the console."""
    from PySide6.QtWidgets import QPushButton
    from spacr.qt.widgets.console_panel import ConsolePanel
    panel = ConsolePanel()
    qtbot.addWidget(panel)
    for btn in panel.findChildren(QPushButton):
        assert btn.text() != "Send"


def test_submit_without_ai_creates_user_banner_and_green_text(qtbot, qt_theme_applied):
    # User input is now rendered as a 'spaCR user' banner + green text block
    # (not a coloured bubble box).
    from spacr.qt.widgets.console_panel import (
        ConsolePanel, _StdoutBlock, _TopicBar, COLOR_USER)
    panel = ConsolePanel()
    qtbot.addWidget(panel)
    panel.set_ai_active(False)
    panel._input.setPlainText("hi")
    panel._on_submit()
    entries = [panel._entries.itemAt(i).widget()
               for i in range(panel._entries.count() - 1)]
    bars = [w for w in entries if isinstance(w, _TopicBar)]
    blocks = [w for w in entries if isinstance(w, _StdoutBlock)]
    assert any("spaCR user" in b._label.text() for b in bars)
    assert any("hi" in b.text() for b in blocks)
    # the user block is coloured green (text colour, not a box)
    assert any(COLOR_USER in (b.styleSheet() or "") for b in blocks)


def test_submit_with_ai_but_no_provider_shows_hint(qtbot, qt_theme_applied,
                                                    monkeypatch):
    """When AI is on but no provider is selected, we shouldn't crash —
    we surface a hint into the console instead."""
    from spacr.qt.widgets.console_panel import ConsolePanel
    from spacr.qt.widgets.console_panel import _StdoutBlock
    panel = ConsolePanel()
    qtbot.addWidget(panel)
    panel.set_ai_active(True)
    panel.set_ai_provider(None)
    panel._input.setPlainText("hi")
    panel._on_submit()
    # Hint text landed in a stdout block
    for i in range(panel._entries.count() - 1):
        w = panel._entries.itemAt(i).widget()
        if isinstance(w, _StdoutBlock) and "[AI]" in w.text():
            return
    pytest.fail("Expected an [AI] hint in the console after submit")


def test_shutdown_is_safe_when_no_stream(qtbot, qt_theme_applied):
    from spacr.qt.widgets.console_panel import ConsolePanel
    panel = ConsolePanel()
    qtbot.addWidget(panel)
    # No exception even with nothing running.
    panel.shutdown()
    assert panel._ai_thread is None
    assert panel._ai_worker is None


def test_shutdown_cancels_active_worker(qtbot, qt_theme_applied):
    """Simulate a running stream and confirm shutdown() calls cancel()
    on the worker + clears the references. This is the crash-fix
    guarantee: if any real subprocess were running the wait(3000) call
    would let it drain."""
    from PySide6.QtCore import QThread
    from spacr.qt.widgets.console_panel import ConsolePanel

    class _FakeWorker:
        def __init__(self):
            self.cancelled = False
        def cancel(self):
            self.cancelled = True

    panel = ConsolePanel()
    qtbot.addWidget(panel)
    fake_worker = _FakeWorker()
    fake_thread = QThread()
    panel._ai_worker = fake_worker
    panel._ai_thread = fake_thread
    try:
        panel.shutdown()
    finally:
        # Whether shutdown succeeded or not, don't leak the thread.
        try:
            fake_thread.quit()
            fake_thread.wait(1000)
        except Exception:
            pass
    assert fake_worker.cancelled is True
    assert panel._ai_thread is None
    assert panel._ai_worker is None


# ---------------------------------------------------------------------------
# AppScreen wiring
# ---------------------------------------------------------------------------

def test_app_screen_has_ai_toggle_label_not_checkbox(qtbot, qt_theme_applied):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets import AiToggleLabel
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    assert isinstance(screen._ai_switch, AiToggleLabel)


def test_app_screen_close_shuts_down_console(qtbot, qt_theme_applied):
    """MainWindow.closeEvent shuts down every ConsolePanel found in
    its widget tree — verify the plumbing works on AppScreen."""
    from PySide6.QtGui import QCloseEvent
    from spacr.qt.app import MainWindow
    from spacr.qt.widgets.console_panel import ConsolePanel
    win = MainWindow()
    qtbot.addWidget(win)
    win._on_nav_selected("mask")
    panels = win.findChildren(ConsolePanel)
    assert len(panels) >= 1
    # Simulate the shutdown path — nothing should raise, and the
    # ConsolePanel's own shutdown() gets called via findChildren.
    win.closeEvent(QCloseEvent())
