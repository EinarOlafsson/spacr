"""spaCR AI answers a failure in a console entry of its own, in spaCR teal.

The maintainer's run: Mask with no channel set and a source folder that does
not exist, AI Error report "Summarize and send all errors". The answer sat
inside the "spaCR output" block as one run-on line, named the wrong screen
("Make Masks"), hid the docs link as "<URL>", and the report named only the
missing channel. These tests drive the real Mask screen, console, bridge and
background runner with only the relay transport faked.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr import crashreport as cr  # noqa: E402
from spacr.qt.ai import providers  # noqa: E402
from tests.test_700_anonymous_error_reports import FakeRelay  # noqa: E402

REFUSAL = ("At least one registered *_channel (for example cell_channel or "
           "organelle_channel) must be set; no masks can be generated.")
ADVICE = ("**What happened** The Mask run did not start. **How to fix it** "
          "1. Set src to your image folder. 2. Set cell_channel. "
          "See https://einarolafsson.github.io/spacr/ for the Mask screen.")
MISSING = "/nonexistent-700/plate1"


@pytest.fixture
def relay(monkeypatch):
    """Allow the network for this test and install a fake relay."""
    monkeypatch.delenv(cr.NO_NETWORK_ENV, raising=False)
    fake = FakeRelay()
    monkeypatch.setattr(cr, "_open_url", fake)
    return fake


def _entries(screen):
    """The console's entries, top to bottom."""
    lay = screen._console._entries
    return [lay.itemAt(i).widget() for i in range(lay.count())
            if lay.itemAt(i).widget() is not None]


def _ai_blocks(screen):
    from PySide6.QtWidgets import QPlainTextEdit

    return [w for w in _entries(screen) if isinstance(w, QPlainTextEdit)
            and w.property("consoleContextKind") == "spacr_ai"]


def _mask_settings():
    from spacr.settings import set_default_settings_preprocess_generate_masks

    settings = set_default_settings_preprocess_generate_masks({"src": MISSING})
    for key in list(settings):
        if key.endswith("_channel"):
            settings[key] = None
    return settings


def _refused_mask_run(qtbot, monkeypatch, relay, hold_ai=False):
    """Run the bridge-wrapped refusal on the real Mask screen."""
    from spacr.errors import raise_if_strict
    from spacr.qt.bridge import _say_what_is_wrong_with_the_settings
    from spacr.qt.screens.app_screen import AppScreen

    cr.set_error_report_mode("all")
    relay.routes["/diagnose"] = [(200, {"advice": ADVICE, "model": "big",
                                        "tier": "best"})]
    relay.routes["/report"] = [(200, {"issue_number": 152, "issue_url": "u",
                                      "deduplicated": False,
                                      "ai_summary_source": "client"})]
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    monkeypatch.setattr(screen, "_report_summary_provider",
                        providers.hosted_provider)
    jobs = []
    if hold_ai:
        monkeypatch.setattr(screen._jobs, "submit",
                            lambda fn, done: jobs.append((fn, done)))

    def pipeline(settings):
        if all(settings.get(k) is None for k in settings
               if k.endswith("_channel")):
            raise_if_strict(REFUSAL, settings=settings)

    screen._watch_for_refusals()
    _say_what_is_wrong_with_the_settings("mask", pipeline)(_mask_settings())
    screen._on_finished(True)
    if not hold_ai:
        qtbot.waitUntil(lambda: not screen._jobs.is_busy(), timeout=15000)
        qtbot.wait(50)
    return screen, jobs


def test_the_answer_is_its_own_teal_entry_after_the_error(
        qtbot, qt_theme_applied, relay, monkeypatch):
    from spacr.qt.theme import active_palette
    from spacr.qt.widgets.console_panel import _TopicBar, _WorkingDots

    screen, jobs = _refused_mask_run(qtbot, monkeypatch, relay, hold_ai=True)
    blocks = _ai_blocks(screen)
    assert len(blocks) == 1
    block = blocks[0]
    entries = _entries(screen)
    at = entries.index(block)
    bar = entries[at - 1]
    assert isinstance(bar, _TopicBar) and bar.text() == "spaCR AI"
    teal = active_palette()["chip_class"]
    assert teal.lower() in bar._label.styleSheet().lower()
    assert block._text_color == teal
    dots = bar.findChild(_WorkingDots)
    assert dots is not None and dots._timer.isActive()
    assert not block.isVisibleTo(screen)
    before = [w for w in entries[:at - 1]
              if hasattr(w, "toPlainText") and w.toPlainText().strip()]
    assert "The run did not start" in before[-1].toPlainText()
    output_bars = [w.text() for w in entries[:at - 1]
                   if isinstance(w, _TopicBar)]
    assert all("spaCR AI" not in t for t in output_bars)

    fn, done = jobs[0]
    done(fn())
    assert not dots._timer.isActive()
    assert block.isVisibleTo(screen)
    lines = block.toPlainText().split("\n")
    assert lines[0] == "What happened"
    assert "How to fix it" in lines
    assert any(line.startswith("1. Set src") for line in lines)
    assert any(line.startswith("2. Set cell_channel") for line in lines)
    assert "https://einarolafsson.github.io/spacr/" in block.toPlainText()
    first = block.document().firstBlock().begin().fragment()
    assert first.charFormat().fontWeight() >= 700
    screen._console.append_stdout("later output\n")
    assert _entries(screen).index(block) == at


def test_every_settings_problem_is_listed_and_sent(qtbot, qt_theme_applied,
                                                   relay, monkeypatch):
    screen, _jobs = _refused_mask_run(qtbot, monkeypatch, relay)
    shown = "\n".join(w.toPlainText() for w in _entries(screen)
                      if hasattr(w, "toPlainText"))
    assert "The run did not start: \n  1. src does not exist: " + MISSING in shown
    assert "2. cell_channel: no segmentation channel is set" in shown
    body = relay.calls_to("/report")[0]["body"]
    assert body["kind"] == "user_error"
    assert body["screen"] == "Mask"
    lines = body["message"].split("\n")
    assert lines[0] == "The run did not start: 2 settings problems."
    assert lines[1].startswith("- src does not exist (its parent folder")
    assert lines[2].startswith("- cell_channel: no segmentation channel")
    assert "nonexistent" not in body["message"]
    assert "nonexistent" not in body["error_text"]
    assert "The run did not start" in body["error_text"]
    assert any(f["function"] == "pipeline" for f in body["frames"])
    assert body["frames"][-1]["function"] != "raise_if_strict" or len(
        body["frames"]) > 1
    diagnose = relay.calls_to("/diagnose")[0]["body"]
    assert diagnose["screen"] == "Mask"
    assert "error_text" not in diagnose


def test_the_entry_opens_again_when_the_console_was_cleared(
        qtbot, qt_theme_applied):
    from spacr.qt.widgets.console_panel import ConsolePanel

    panel = ConsolePanel("Mask")
    qtbot.addWidget(panel)
    block = panel.begin_ai_entry()
    panel.clear()
    qtbot.wait(10)
    again = panel.fill_ai_entry(block, "Done.")
    assert again.toPlainText() == "Done."
    assert again.property("consoleContextKind") == "spacr_ai"


def test_a_repeated_error_looks_like_a_fresh_answer_without_thinking(
        qtbot, qt_theme_applied, relay, monkeypatch):
    from spacr.qt.widgets.console_panel import _TopicBar, _WorkingDots

    first = _refused_mask_run(qtbot, monkeypatch, relay)
    if isinstance(first, tuple):
        first = first[0]
    fresh_block = _ai_blocks(first)[0]
    fresh_entries = _entries(first)
    fresh_bar = fresh_entries[fresh_entries.index(fresh_block) - 1]
    assert len(relay.calls_to("/diagnose")) == 1
    screen, jobs = _refused_mask_run(qtbot, monkeypatch, relay, hold_ai=True)
    block = _ai_blocks(screen)[0]
    entries = _entries(screen)
    bar = entries[entries.index(block) - 1]
    assert bar.text() == fresh_bar.text()
    assert isinstance(bar, _TopicBar)
    assert bar.text() == "spaCR AI"
    assert "saved" not in bar.text().lower()
    assert not bar.findChild(_WorkingDots)._timer.isActive()
    assert block.isVisibleTo(screen)
    assert block.toPlainText().startswith("What happened")
    fn, done = jobs[0]
    done(fn())
    assert len(relay.calls_to("/diagnose")) == 1, "no second AI call"
    assert relay.calls_to("/report")[-1]["body"]["ai_summary"].startswith(
        "**What happened**")
    assert len(_ai_blocks(screen)) == 1
