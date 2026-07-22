"""Stress + edge-case tests for the merged Console AI flow.

Every test here uses a FakeProvider whose stream_chat spawns a real
Python subprocess. That's important because the crashes we're
regression-testing all involve subprocess I/O and QThread ownership
— pure mocks would paper over them.
"""
from __future__ import annotations

import subprocess
import sys
import time
from typing import List, Optional

import pytest


# ---------------------------------------------------------------------------
# Fixtures — a real-subprocess-backed fake provider
# ---------------------------------------------------------------------------

def _make_short_provider(reply: str = "ok"):
    """Build a ClaudeCliProvider subclass that streams `reply` from a
    fresh subprocess for each call."""
    from spacr.qt.ai.providers import ClaudeCliProvider, _stream_process
    escaped = reply.replace("'", "\\'")

    class _ShortProvider(ClaudeCliProvider):
        def stream_chat(self, messages, system="", model=None):
            code = (
                f"import sys; "
                f"sys.stdout.write('{escaped}\\n'); "
                f"sys.stdout.flush()"
            )
            argv = [sys.executable, "-c", code]
            yield from _stream_process(argv, provider=self)

    return _ShortProvider()


def _make_streaming_provider(lines: int = 20, per_line: str = "chunk"):
    """Provider that emits many lines with a small delay so we can
    catch mid-stream cancellation."""
    from spacr.qt.ai.providers import ClaudeCliProvider, _stream_process

    class _StreamProvider(ClaudeCliProvider):
        def stream_chat(self, messages, system="", model=None):
            code = (
                "import sys, time\n"
                f"for i in range({lines}):\n"
                f"    sys.stdout.write('{per_line} ' + str(i) + '\\n')\n"
                "    sys.stdout.flush()\n"
                "    time.sleep(0.05)\n"
            )
            argv = [sys.executable, "-c", code]
            yield from _stream_process(argv, provider=self)

    return _StreamProvider()


def _swap_provider(monkeypatch, fake_provider, name: str = "claude"):
    """Route ai_module.get_provider(name) → fake_provider."""
    from spacr.qt import ai as ai_module
    from spacr.qt.ai import providers as pmod
    orig = ai_module.get_provider

    def _get(n):
        if n == name:
            return fake_provider
        return orig(n)

    monkeypatch.setattr(ai_module, "get_provider", _get)
    monkeypatch.setattr(pmod, "get_provider", _get)


def _wait_stream_done(panel, qtbot, timeout_ms: int = 8000):
    """Poll until the panel is idle (no active AI thread)."""
    ticks = timeout_ms // 20
    for _ in range(ticks):
        qtbot.wait(20)
        if panel._ai_thread is None:
            return True
    return False


# ---------------------------------------------------------------------------
# 1. Five consecutive streams — no crash, no leaked threads
# ---------------------------------------------------------------------------

def test_five_consecutive_streams_dont_crash(qtbot, qt_theme_applied,
                                              monkeypatch):
    from spacr.qt.widgets.console_panel import ConsolePanel

    fake = _make_short_provider("hello")
    _swap_provider(monkeypatch, fake)

    panel = ConsolePanel()
    qtbot.addWidget(panel)
    panel.set_ai_provider("claude")
    panel.set_ai_active(True)

    try:
        for i in range(5):
            panel._input.setPlainText(f"turn {i}")
            panel._on_submit()
            assert _wait_stream_done(panel, qtbot), \
                f"stream {i} never completed"

        # After a beat the pruner (called on next submit + on shutdown)
        # should have cleared retired entries whose threads have exited.
        for _ in range(20):
            qtbot.wait(20)
        panel._prune_retired()
        # Retired list should be empty — every stream we ran is done.
        assert len(panel._retired) == 0, \
            f"expected no retired threads left, got {len(panel._retired)}"
    finally:
        panel.shutdown()


# ---------------------------------------------------------------------------
# 2. Cancel mid-stream via the public cancel_ai() API
# ---------------------------------------------------------------------------

def test_cancel_mid_stream_unblocks_worker(qtbot, qt_theme_applied,
                                             monkeypatch):
    from spacr.qt.widgets.console_panel import ConsolePanel

    fake = _make_streaming_provider(lines=100, per_line="line")
    _swap_provider(monkeypatch, fake)

    panel = ConsolePanel()
    qtbot.addWidget(panel)
    panel.set_ai_provider("claude")
    panel.set_ai_active(True)

    try:
        panel._input.setPlainText("stream please")
        panel._on_submit()
        # Let a few chunks arrive
        for _ in range(10):
            qtbot.wait(20)
        assert panel.is_ai_streaming()

        started = time.monotonic()
        panel.cancel_ai()
        # Should promptly resolve
        assert _wait_stream_done(panel, qtbot, timeout_ms=4000), \
            "cancel_ai() did not stop the stream"
        elapsed = time.monotonic() - started
        assert elapsed < 4, f"cancel took {elapsed:.1f}s"
    finally:
        panel.shutdown()


# ---------------------------------------------------------------------------
# 3. Rapid successive submits while a stream is running
# ---------------------------------------------------------------------------

def test_second_submit_while_streaming_is_noop_not_crash(qtbot, qt_theme_applied,
                                                          monkeypatch):
    from spacr.qt.widgets.console_panel import ConsolePanel

    fake = _make_streaming_provider(lines=30, per_line="x")
    _swap_provider(monkeypatch, fake)

    panel = ConsolePanel()
    qtbot.addWidget(panel)
    panel.set_ai_provider("claude")
    panel.set_ai_active(True)

    try:
        panel._input.setPlainText("first")
        panel._on_submit()
        for _ in range(5):
            qtbot.wait(20)
        # Second submit — must not crash and must not spawn a rival thread
        first_thread = panel._ai_thread
        panel._input.setPlainText("second — while first still running")
        panel._on_submit()
        # We're still on the first thread; the second submit is a no-op.
        assert panel._ai_thread is first_thread
        # Let the first one finish naturally.
        assert _wait_stream_done(panel, qtbot, timeout_ms=10000)
    finally:
        panel.shutdown()


# ---------------------------------------------------------------------------
# 4. Provider switch between messages
# ---------------------------------------------------------------------------

def test_provider_switch_between_messages(qtbot, qt_theme_applied,
                                            monkeypatch):
    from spacr.qt.widgets.console_panel import ConsolePanel
    from spacr.qt import ai as ai_module

    claude_fake = _make_short_provider("A")
    codex_fake = _make_short_provider("B")
    real_get = ai_module.get_provider

    def _get(n):
        if n == "claude":
            return claude_fake
        if n == "codex":
            return codex_fake
        return real_get(n)

    from spacr.qt.ai import providers as pmod
    monkeypatch.setattr(ai_module, "get_provider", _get)
    monkeypatch.setattr(pmod, "get_provider", _get)

    panel = ConsolePanel()
    qtbot.addWidget(panel)
    panel.set_ai_active(True)

    try:
        panel.set_ai_provider("claude")
        panel._input.setPlainText("via claude")
        panel._on_submit()
        assert _wait_stream_done(panel, qtbot)

        panel.set_ai_provider("codex")
        panel._input.setPlainText("via codex")
        panel._on_submit()
        assert _wait_stream_done(panel, qtbot)
    finally:
        panel.shutdown()


# ---------------------------------------------------------------------------
# 5. Explain-error flow end-to-end (fake provider replies)
# ---------------------------------------------------------------------------

def test_open_error_flow_streams_a_reply(qtbot, qt_theme_applied, monkeypatch):
    from spacr.qt.widgets.console_panel import ConsolePanel

    fake = _make_short_provider("You forgot to pass channels=[0,1].")
    _swap_provider(monkeypatch, fake)

    panel = ConsolePanel(active_app_label="Mask")
    qtbot.addWidget(panel)
    panel.set_ai_provider("claude")

    try:
        tb = "Traceback (most recent call last):\n  File 'x'\nTypeError: foo"
        panel.open_error_flow(tb, active_app="Mask")
        assert _wait_stream_done(panel, qtbot)
        # The reply should show up in a stdout block containing the fake text
        from spacr.qt.widgets.console_panel import _StdoutBlock
        found = False
        for i in range(panel._entries.count() - 1):
            w = panel._entries.itemAt(i).widget()
            if isinstance(w, _StdoutBlock) and "channels=[0,1]" in w.text():
                found = True
                break
        assert found, "AI reply didn't land in a stdout block"
    finally:
        panel.shutdown()


# ---------------------------------------------------------------------------
# 6. Two AppScreens each with their own console — no cross-talk
# ---------------------------------------------------------------------------

def test_two_consoles_do_not_share_ai_state(qtbot, qt_theme_applied,
                                              monkeypatch):
    from spacr.qt.widgets.console_panel import ConsolePanel

    fake = _make_short_provider("done")
    _swap_provider(monkeypatch, fake)

    panel_a = ConsolePanel(active_app_label="Mask")
    panel_b = ConsolePanel(active_app_label="Measure")
    qtbot.addWidget(panel_a)
    qtbot.addWidget(panel_b)
    for p in (panel_a, panel_b):
        p.set_ai_provider("claude")
        p.set_ai_active(True)

    try:
        panel_a._input.setPlainText("A")
        panel_a._on_submit()
        assert _wait_stream_done(panel_a, qtbot)
        # panel_b did not run
        assert panel_b._ai_thread is None
        assert not panel_b._ai_messages

        panel_b._input.setPlainText("B")
        panel_b._on_submit()
        assert _wait_stream_done(panel_b, qtbot)
        assert panel_a._ai_messages != panel_b._ai_messages
    finally:
        panel_a.shutdown()
        panel_b.shutdown()


# ---------------------------------------------------------------------------
# 7. Multi-line + longer streaming reply renders correctly
# ---------------------------------------------------------------------------

def test_streaming_reply_appends_every_line(qtbot, qt_theme_applied,
                                              monkeypatch):
    from spacr.qt.widgets.console_panel import ConsolePanel, _StdoutBlock

    N = 12
    fake = _make_streaming_provider(lines=N, per_line="row")
    _swap_provider(monkeypatch, fake)

    panel = ConsolePanel()
    qtbot.addWidget(panel)
    panel.set_ai_provider("claude")
    panel.set_ai_active(True)

    try:
        panel._input.setPlainText("stream 12 lines")
        panel._on_submit()
        assert _wait_stream_done(panel, qtbot)

        text = ""
        for i in range(panel._entries.count() - 1):
            w = panel._entries.itemAt(i).widget()
            if isinstance(w, _StdoutBlock):
                text += w.text()
        # Every line arrived
        for i in range(N):
            assert f"row {i}" in text, f"missing row {i} in reply"
    finally:
        panel.shutdown()


# ---------------------------------------------------------------------------
# 8. Shutdown during an active stream — no crash, thread joined
# ---------------------------------------------------------------------------

def test_shutdown_during_active_stream(qtbot, qt_theme_applied, monkeypatch):
    from spacr.qt.widgets.console_panel import ConsolePanel

    fake = _make_streaming_provider(lines=200, per_line="a")
    _swap_provider(monkeypatch, fake)

    panel = ConsolePanel()
    qtbot.addWidget(panel)
    panel.set_ai_provider("claude")
    panel.set_ai_active(True)

    try:
        panel._input.setPlainText("please stream a lot")
        panel._on_submit()
        # Let it start
        for _ in range(10):
            qtbot.wait(20)
        assert panel.is_ai_streaming()

        # Now shut down — panel refs should clear and the thread should exit
        started = time.monotonic()
        panel.shutdown()
        elapsed = time.monotonic() - started
        assert panel._ai_thread is None
        assert panel._ai_worker is None
        assert elapsed < 6, f"shutdown took {elapsed:.1f}s during active stream"
    finally:
        # Idempotent
        panel.shutdown()


# ---------------------------------------------------------------------------
# 9. Submit while AI is off — a green bubble, no thread
# ---------------------------------------------------------------------------

def test_submit_with_ai_off_never_starts_a_thread(qtbot, qt_theme_applied):
    from spacr.qt.widgets.console_panel import ConsolePanel

    panel = ConsolePanel()
    qtbot.addWidget(panel)
    panel.set_ai_active(False)

    for _ in range(3):
        panel._input.setPlainText("note")
        panel._on_submit()
    assert panel._ai_thread is None
    assert panel._ai_worker is None
    assert not panel.is_ai_streaming()


# ---------------------------------------------------------------------------
# 10. AppScreen: AI toggle drives ConsolePanel state
# ---------------------------------------------------------------------------

def test_app_screen_ai_toggle_drives_console_panel(qtbot, qt_theme_applied,
                                                    monkeypatch):
    from spacr.qt.screens.app_screen import AppScreen

    # Pretend claude is installed
    from spacr.qt.ai import providers as pmod
    monkeypatch.setattr(pmod.shutil, "which",
                          lambda name: "/opt/bin/claude" if name == "claude" else None)

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    console = screen._console
    assert console._ai_active is False
    screen._ai_switch.setChecked(True)
    assert console._ai_active is True
    assert console._current_provider_name == "claude"
    screen._ai_switch.setChecked(False)
    assert console._ai_active is False
