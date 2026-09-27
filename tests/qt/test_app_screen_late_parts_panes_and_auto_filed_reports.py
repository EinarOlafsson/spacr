"""Late parts, runtime panes, the hint strip and automatically filed reports.

Pins what an AppScreen does at the edges of its deferred and optional
machinery:

* a deferred part's waiting callbacks all run even when one raises, and a
  callback queued before any other creates the queue;
* a late part whose surfaces cannot be cleared, whose children die while
  polished, or whose language pass fails, is still left usable;
* a card with no title/body cannot fold; one that already folds keeps its
  folder; a runtime column without a collapsible splitter stacks its
  cards; a card outside the splitter is not adopted; a body that is not
  collapsible counts as open;
* the hint strip's writers do nothing without a strip, and the hint link
  does nothing without a hovered setting;
* showing and hiding a screen with no focus rule works, and the first-show
  surface sweep failing is survived;
* the terms/preference probes answer "no" when they cannot be read, and the
  automatic issue report: nothing without a traceback, no wait for an AI
  answering a different error, a wait that is re-armed without a second
  timer or listener, a waiting report filed once, a report that fails to
  be filed said to have failed, a job queue that refuses clears the in-flight
  mark, and a filed report that cannot be remembered is still announced.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QHideEvent, QShowEvent  # noqa: E402
from PySide6.QtWidgets import QLabel, QSplitter, QVBoxLayout, QWidget  # noqa: E402

from spacr.qt.screens import app_screen as aps  # noqa: E402
from spacr.qt.screens.app_screen import AppScreen  # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus  # noqa: E402

pytestmark = pytest.mark.qt


def _console_text(console) -> str:
    from spacr.qt.widgets.console_panel import _StdoutBlock

    return "\n".join(block.text()
                     for block in console.findChildren(_StdoutBlock))


@pytest.fixture
def screen(qtbot):
    widget = AppScreen("regression")
    try:
        yield widget
    finally:
        retire_pyqtgraph_menus(widget)
        widget.close()
        widget.deleteLater()


def _raise(error=ValueError):
    def go(*_args, **_kwargs):
        raise error("refused")
    return go


# --------------------------------------------------------------------------
# deferred parts


def test_every_waiting_callback_runs_even_when_one_raises(screen):
    ran = []
    screen.__dict__["_parts_owed"] = {"late": lambda: None}
    screen.__dict__["_after_parts"] = {
        "late": [_raise(RuntimeError), lambda: ran.append("second")]}
    assert screen._build_owed_part("late") is True
    assert ran == ["second"]
    assert screen._build_owed_part("late") is False


def test_the_first_callback_for_a_part_starts_its_queue(screen):
    screen.__dict__["_parts_owed"] = {"late": lambda: None}
    screen.__dict__.pop("_after_parts", None)
    try:
        assert screen._after_part_is_built("late", print) is True
        assert screen.__dict__["_after_parts"] == {"late": [print]}
        assert screen._after_part_is_built("late", repr) is True
        assert screen.__dict__["_after_parts"] == {"late": [print, repr]}
    finally:
        screen.__dict__["_parts_owed"].pop("late", None)
        screen.__dict__.pop("_after_parts", None)


def test_a_part_whose_surfaces_cannot_be_cleared_is_kept(screen, monkeypatch,
                                                         qtbot):
    arrows = []
    monkeypatch.setattr("spacr.qt.theme.clear_container_surfaces",
                        _raise(TypeError))
    monkeypatch.setattr("spacr.qt.theme.take_the_scroll_arrows_off",
                        arrows.append)
    part = QWidget()
    qtbot.addWidget(part)
    screen._clear_a_late_parts_surfaces(part)
    assert arrows == []
    assert part.isEnabled()


class _DiesWhenPolished(QWidget):
    def ensurePolished(self):                                # noqa: N802
        raise RuntimeError("Internal C++ object already deleted.")


def test_a_part_that_has_gone_is_not_translated(screen, monkeypatch):
    translated = []
    monkeypatch.setattr("spacr.qt.i18n.retranslate_widget_tree",
                        lambda *a, **k: translated.append(a))
    screen._translate_a_late_part(types.SimpleNamespace(
        children=_raise(RuntimeError)))
    assert translated == []


def test_a_root_that_dies_while_polished_is_not_translated(screen, qtbot,
                                                           monkeypatch):
    translated = []
    monkeypatch.setattr("spacr.qt.i18n.retranslate_widget_tree",
                        lambda *a, **k: translated.append(a))
    root = _DiesWhenPolished()
    qtbot.addWidget(root)
    child = _DiesWhenPolished(root)
    screen._translate_a_late_part(root)
    assert translated == []
    assert child.parent() is root


def test_a_failing_language_pass_leaves_the_part_in_place(screen, qtbot,
                                                          monkeypatch):
    calls = []

    def retranslate(widget, only_new=False):
        calls.append((widget, only_new))
        if not only_new:
            raise RuntimeError("child went away")

    monkeypatch.setattr("spacr.qt.i18n.retranslate_widget_tree", retranslate)
    monkeypatch.setattr(
        "spacr.qt.screens.settings_model.retarget_field_tooltips",
        _raise(ValueError))
    root = QWidget()
    qtbot.addWidget(root)
    child = _DiesWhenPolished(root)
    screen._translate_a_late_part(root)
    assert calls == [(child, False), (root, True)]


# --------------------------------------------------------------------------
# runtime panes


def test_folding_a_card_needs_a_title_and_a_body():
    assert AppScreen._fold_a_card(None, "x") is None
    folder = object()
    assert AppScreen._fold_a_card(types.SimpleNamespace(folder=folder),
                                  "x") is folder
    assert AppScreen._fold_a_card(
        types.SimpleNamespace(title_label=QLabel("t")), "x") is None


def test_without_a_collapsible_splitter_the_cards_just_stack(screen, qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    usage, section = QWidget(), QWidget()
    was_split = screen._runtime_splitter
    was_focus = screen._shell_focus
    screen._runtime_splitter = QSplitter()
    try:
        screen._install_the_shell_panes(layout, usage, section)
        assert layout.count() == 2
        assert layout.itemAt(0).widget() is usage
        assert layout.itemAt(1).widget() is section
    finally:
        screen._runtime_splitter = was_split
        screen._shell_focus = was_focus


def test_a_card_outside_the_splitter_is_not_adopted(screen, qtbot):
    stray = QWidget()
    qtbot.addWidget(stray)
    assert screen.adopt_runtime_pane(stray) is None
    assert screen.adopt_runtime_pane(None) is None
    was = screen._runtime_splitter
    screen._runtime_splitter = QSplitter()
    try:
        assert screen.adopt_runtime_pane(stray) is None
    finally:
        screen._runtime_splitter = was


def test_a_card_is_adopted_without_a_focus_rule(screen):
    card = QWidget()
    card.setObjectName("Declared preview")
    split = screen._runtime_splitter
    split.insertWidget(0, card)
    was = screen._shell_focus
    screen._shell_focus = None
    try:
        pane = screen.adopt_runtime_pane(card)
    finally:
        screen._shell_focus = was
    assert pane is not None
    assert split.indexOf(card) >= 0


def test_a_settings_column_that_cannot_fold_counts_as_open(screen):
    was = screen._body_splitter
    screen._body_splitter = QSplitter()
    try:
        assert screen.reveal_settings() is True
    finally:
        screen._body_splitter = was


# --------------------------------------------------------------------------
# the hint strip


def test_without_a_strip_the_hint_writers_do_nothing(screen, monkeypatch):
    was = screen.__dict__.get("_hint_strip")
    screen._hint_strip = None
    fitted = []
    monkeypatch.setattr(aps, "_fit_to_lines",
                        lambda *a: fitted.append(a) or "")
    try:
        screen._write_hint("anything", "https://example.org")
        screen._release_the_hint()
    finally:
        screen._hint_strip = was
    assert fitted == []


def test_the_animation_link_needs_a_hovered_setting(screen, monkeypatch):
    shown = []
    from spacr.qt.widgets.hover_tooltip import HoverTooltip

    monkeypatch.setattr(HoverTooltip, "instance",
                        classmethod(lambda cls: shown.append(True)))
    screen._hinted_widget = None
    screen._hinted_html = "<b>help</b>"
    screen._on_hint_link(aps._HINT_ANIMATION_HREF)
    screen._hinted_widget = QLabel()
    screen._hinted_html = ""
    screen._on_hint_link(aps._HINT_ANIMATION_HREF)
    assert shown == []


# --------------------------------------------------------------------------
# show and hide


def test_show_and_hide_without_a_focus_rule_and_a_failing_sweep(
        screen, monkeypatch):
    swept = []

    def sweep():
        swept.append(True)
        raise ValueError("sweep refused")

    monkeypatch.setattr(screen, "_clear_page_surfaces", sweep)
    was = screen._shell_focus
    screen._shell_focus = None
    screen._surfaces_cleared_on_show = False
    try:
        screen.showEvent(QShowEvent())
        screen.hideEvent(QHideEvent())
    finally:
        screen._shell_focus = was
    assert swept == [True]
    assert screen._surfaces_cleared_on_show is True


# --------------------------------------------------------------------------
# probes


def test_terms_that_cannot_be_read_forbid_automatic_filing(monkeypatch):
    monkeypatch.setattr("spacr.qt.terms.needs_agreement", _raise(OSError))
    assert AppScreen._the_terms_allow_automatic_filing() is False


def test_an_unreadable_reporting_mode_is_neither_always_nor_allowed(
        monkeypatch):
    monkeypatch.setattr("spacr.qt.preferences.get_issue_prompt_mode",
                        _raise(OSError))
    assert AppScreen._reporting_is_set_to_always() is False
    assert AppScreen._reporting_is_not_set_to_never() is False


# --------------------------------------------------------------------------
# the automatic report


TB = "Traceback (most recent call last):\n  ValueError: bad plate\n"


@pytest.fixture
def filing(screen):
    aps._REPORTS_BEING_FILED.clear()
    yield screen
    aps._REPORTS_BEING_FILED.clear()


def test_no_traceback_files_nothing(filing, monkeypatch):
    posted = []
    monkeypatch.setattr(filing, "_post_the_report",
                        lambda *a: posted.append(a))
    filing._last_error_text = ""
    filing._file_the_report_automatically()
    assert posted == []


def test_an_ai_answering_another_error_is_not_waited_for(filing):
    was = filing._console
    filing._console = types.SimpleNamespace(
        _ai_worker=object(), _ai_error_traceback="a different error",
        ai_explanation_of=lambda tb: "")
    try:
        assert filing._the_ai_is_still_explaining(TB) is False
        filing._console._ai_error_traceback = TB
        filing._console.ai_explanation_of = _raise(RuntimeError)
        assert filing._the_ai_is_still_explaining(TB) is False
    finally:
        filing._console = was


def test_a_rearmed_wait_keeps_one_timer_and_one_listener(filing,
                                                         monkeypatch):
    posted = []
    monkeypatch.setattr(filing, "_post_the_report",
                        lambda tb, fp: posted.append((tb, fp)))
    filing._wait_for_the_ai_then_file(TB, "fp1")
    timer = filing._ai_answer_timer
    filing._wait_for_the_ai_then_file(TB, "fp1")
    assert filing._ai_answer_timer is timer
    assert timer.isActive()
    assert "fp1" in aps._REPORTS_BEING_FILED
    filing._file_the_waiting_report()
    assert posted == [(TB, "fp1")]
    assert not timer.isActive()
    assert filing._listening_for_the_ai is False
    filing._file_the_waiting_report()
    assert posted == [(TB, "fp1")], "a wait files once"


def test_a_signal_already_gone_does_not_stop_the_waiting_report(
        filing, monkeypatch):
    posted = []
    monkeypatch.setattr(filing, "_post_the_report",
                        lambda tb, fp: posted.append(fp))

    class _Gone:
        def disconnect(self, slot):
            raise RuntimeError("already disconnected")

    was = filing._console
    filing._console = types.SimpleNamespace(ai_stream_finished=_Gone())
    filing._ai_answer_timer = None
    filing._listening_for_the_ai = True
    filing._report_waiting_for_the_ai = (TB, "fp2")
    try:
        filing._file_the_waiting_report()
    finally:
        filing._console = was
    assert posted == ["fp2"]
    assert filing._listening_for_the_ai is False


class _Jobs:
    def __init__(self, accept=True):
        self.accept = accept
        self.outcomes = []

    def submit(self, fn, done):
        if not self.accept:
            return False
        outcome = fn()
        self.outcomes.append(outcome)
        done(outcome)
        return True


def test_a_report_that_cannot_be_filed_is_said_to_have_failed(
        filing, monkeypatch):
    from spacr.qt.ai import issue_report

    monkeypatch.setattr(filing._console, "ai_explanation_of",
                        _raise(RuntimeError))
    monkeypatch.setattr("spacr.qt.preferences.get_share_diagnostic_logs",
                        _raise(OSError))
    built = []
    monkeypatch.setattr(issue_report, "build_report",
                        lambda tb, **kw: built.append(kw) or {"title": "t"})
    monkeypatch.setattr(issue_report, "public_report", lambda report: report)
    monkeypatch.setattr(issue_report, "file_without_review",
                        _raise(ValueError))
    jobs = _Jobs()
    was = filing._jobs
    filing._jobs = jobs
    try:
        filing._post_the_report(TB, "fp3")
    finally:
        filing._jobs = was
    assert built[0]["ai_response"] == ""
    assert built[0]["include_log_tail"] is False
    assert jobs.outcomes == [{"status": issue_report.FAILED,
                              "detail": "ValueError: refused",
                              "fingerprint": "fp3"}]
    assert "fp3" not in aps._REPORTS_BEING_FILED


def test_a_job_queue_that_refuses_clears_the_in_flight_mark(filing,
                                                            monkeypatch):
    monkeypatch.setattr(filing._console, "ai_explanation_of", lambda tb: "")
    was = filing._jobs
    filing._jobs = _Jobs(accept=False)
    try:
        filing._post_the_report(TB, "fp4")
    finally:
        filing._jobs = was
    assert "fp4" not in aps._REPORTS_BEING_FILED


def test_a_filed_report_that_cannot_be_remembered_is_still_announced(
        filing, monkeypatch):
    from spacr.qt.ai import issue_report

    monkeypatch.setattr("spacr.qt.ai.settings.remember_auto_filed",
                        _raise(OSError))
    filing._on_report_filed_automatically({
        "status": issue_report.FILED, "fingerprint": "fp5",
        "url": "https://github.com/x/y/issues/1"})
    assert "https://github.com/x/y/issues/1" in _console_text(
        filing._console)
