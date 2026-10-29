"""Every AI summary of a failure reaches the spaCR console under "spaCR AI".

The maintainer's case: a Mask run refused for a missing channel, with
Preferences -> AI -> AI Error report set to "Summarize and send all errors",
filed its issue with an AI summary but showed no "spaCR AI:" line. The
console's notice template names its placeholder ``{text}``, which collided
with :func:`spacr.qt.i18n.tr`'s own first parameter, so the notice raised on
the GUI thread and the background runner only logged it.

These tests drive the real flow end to end: the real Mask screen, its real
background runner and real console, and only the relay's transport faked.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr import crashreport as cr  # noqa: E402
from tests.test_700_anonymous_error_reports import FakeRelay  # noqa: E402
from tests.qt.test_700_anonymous_reports_qt import _ReportSession  # noqa: E402
from tests.qt.test_the_console_keeps_the_ai_error_analysis import (  # noqa: E402
    manual_error_batches,
)

REFUSAL = ("At least one registered *_channel (for example cell_channel or "
           "organelle_channel) must be set; no masks can be generated.")
ADVICE = ("**Set** `cell_channel` to the channel {index} of your cells, "
          "then run Mask again.")
RELAY_SUMMARY = "Likely cause: no {channel} is set for Mask."


@pytest.fixture
def relay(monkeypatch, manual_error_batches):
    """Allow the network for this test and install a fake relay."""
    monkeypatch.delenv(cr.NO_NETWORK_ENV, raising=False)
    monkeypatch.setattr(cr, "_ERROR_DISPATCH_SEEN", {})
    monkeypatch.setattr(cr, "_ERROR_OCCURRENCE_POLICIES", {})
    fake = FakeRelay()
    fake.finish_quiet_window = manual_error_batches
    fake.sessions = []
    monkeypatch.setattr(cr, "_open_url", fake)
    yield fake
    fake.finish_quiet_window()
    for session in fake.sessions:
        session.wait()


def _console_text(screen) -> str:
    """Everything the screen's console shows, block by block."""
    from PySide6.QtWidgets import QPlainTextEdit, QTextEdit

    widgets = (screen._console.findChildren(QPlainTextEdit)
               + screen._console.findChildren(QTextEdit))
    return "\n".join(w.toPlainText() for w in widgets)


def _ai_text(screen) -> str:
    """What the console's "spaCR AI" entries show."""
    from PySide6.QtWidgets import QPlainTextEdit

    return "\n".join(
        w.toPlainText() for w in screen._console.findChildren(QPlainTextEdit)
        if w.property("consoleContextKind") == "spacr_ai")


def _refuse_a_mask_run(qtbot, relay, monkeypatch, mode, report_answer,
                       diagnose=(200, {"advice": ADVICE, "model": "big",
                                       "tier": "best"})):
    """Refuse a Mask run for a missing channel and let the runner settle."""
    from spacr.errors import raise_if_strict
    from spacr.qt.screens.app_screen import AppScreen

    cr.set_error_report_mode(mode)
    relay.routes["/diagnose"] = [diagnose]
    relay.routes["/report"] = [(200, report_answer)]
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    session = _ReportSession(screen, qtbot, monkeypatch)
    relay.sessions.append(session)
    screen._watch_for_refusals()
    raise_if_strict(REFUSAL, settings={})
    screen._on_finished(True)
    relay.finish_quiet_window()
    session.wait()
    return screen


def test_the_refused_run_shows_the_ai_explanation(qtbot, qt_theme_applied,
                                                  relay, monkeypatch,
                                                  console_info_enabled):
    screen = _refuse_a_mask_run(qtbot, relay, monkeypatch, "all", {
        "issue_number": 149, "issue_url": "u", "deduplicated": False,
        "ai_summary_source": "client"})
    text = _console_text(screen)
    assert _ai_text(screen) == (
        "Set cell_channel to the channel {index} of your cells, "
        "then run Mask again.")
    assert "Reported as issue #149 (with AI summary)" in text
    assert relay.calls_to("/report")[0]["body"]["ai_summary"] == ADVICE


def test_a_summary_the_relay_wrote_is_shown(qtbot, qt_theme_applied, relay,
                                            monkeypatch):
    screen = _refuse_a_mask_run(qtbot, relay, monkeypatch, "all", {
        "issue_number": 150, "issue_url": "u", "deduplicated": False,
        "ai_summary_source": "relay", "ai_summary": RELAY_SUMMARY},
        diagnose=(429, {"refused": "quota"}))
    assert _ai_text(screen).startswith(RELAY_SUMMARY + "\n")


@pytest.mark.parametrize("mode", ["send_all", "send_faults", "off"])
def test_no_ai_summary_modes_print_nothing_from_ai(qtbot, qt_theme_applied,
                                                   relay, monkeypatch, mode):
    screen = _refuse_a_mask_run(qtbot, relay, monkeypatch, mode, {
        "issue_number": 151, "issue_url": "u", "deduplicated": False,
        "ai_summary_source": "none"})
    assert relay.calls_to("/diagnose") == []
    assert "spaCR AI" not in _console_text(screen)


def test_tr_accepts_a_placeholder_named_text():
    from spacr.qt.i18n import tr

    assert tr("spaCR AI: {text}{light}", text="{x} **y**",
              light="") == "spaCR AI: {x} **y**"
