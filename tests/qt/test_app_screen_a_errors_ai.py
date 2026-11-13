"""Pipeline errors, issue filing, the AI switch and hover hints on ``AppScreen``.

A pipeline traceback is always printed raw unless routing through the AI is
on and works; consent to file issues reveals a button but a crash never
submits anything by itself; the issue snapshot survives any settings model.
The AI switch picks a provider without overriding a user's choice. Hover
help goes to the fixed bottom strip, which keeps the last hint (and its API
link) after the pointer leaves.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent                                  # noqa: E402
from PySide6.QtWidgets import QCheckBox, QDialog, QLabel           # noqa: E402

from spacr.qt.screens.app_screen import AppScreen                  # noqa: E402

from tests.qt.test_app_screen_a_construction import (               # noqa: E402,F401
    _reset_hover_tooltip_around_each_test,
    built,
    console_text,
    settle,
)
from tests.qt.test_the_console_keeps_the_ai_error_analysis import (  # noqa: E402,F401
    manual_error_batches,
)

pytestmark = pytest.mark.qt

ROUTE = "spacr.qt.ai.settings.get_route_errors_through_ai"
AUTO_FILE = "spacr.qt.ai.settings.get_auto_file_issues"


class _FakeProvider:
    def __init__(self, name, label):
        self.name = name
        self.label = label


def _raises(message):
    return lambda *a, **k: (_ for _ in ()).throw(RuntimeError(message))


def test_a_pipeline_error_is_printed_raw_unless_ai_routing_works(
        qtbot, subtests):
    """The traceback reaches the console verbatim when AI is off, when routing
    is disabled, or when the AI lookup itself raises; the file-issue button
    appears only with consent (and then no report is submitted), and an
    unreadable consent preference keeps it hidden.
    """
    submitted = []

    def ai_on(mp, scr):
        scr._console.set_ai_active(True)
        scr._console.set_ai_provider("claude")
        scr._console.open_error_flow = lambda *a, **k: pytest.fail(
            "must not route when routing is unavailable")

    def consent(mp, scr):
        mp.setattr("spacr.qt.ai.issue_report.submit_report",
                   lambda report: submitted.append(report))
        widgets = scr._settings_model._widgets
        widgets["src"].setText("/data/for_issue")
        widgets["batch_size"].setValue(11)
        widgets["plot"].setChecked(True)

    off, on = (lambda: False), (lambda: True)
    cases = [
        # name, route pref, auto-file pref, setup, button enabled
        ("ai off", off, off, None, False),
        ("routing disabled", off, off, ai_on, False),
        ("routing lookup explodes", _raises("prefs gone"), off, ai_on, False),
        ("consent given", off, on, consent, True),
        ("consent unreadable", off, _raises("registry gone"), None, False),
    ]
    for name, route, auto_file, setup, enabled in cases:
        with subtests.test(case=name), pytest.MonkeyPatch.context() as mp:
            mp.setattr("spacr.qt.ai.get_provider",
                       lambda n: _FakeProvider(n, n.title()))
            # A preference that raises is broken only after construction,
            # as it would be when the registry goes away mid-session.
            mp.setattr(ROUTE, route if route in (off, on) else off)
            mp.setattr(AUTO_FILE, auto_file if auto_file in (off, on) else off)
            if name == "ai off":
                mp.setattr("spacr.qt.preferences.get_ai_on_by_default",
                           lambda: False)
            with built("mask") as scr:
                mp.setattr(ROUTE, route)
                mp.setattr(AUTO_FILE, auto_file)
                if setup is not None:
                    setup(mp, scr)
                tb = f"Traceback ...\nValueError: {name}"
                scr._on_pipeline_error(tb)
                assert tb in console_text(scr._console)
                assert scr._last_error_text.endswith(name)
                assert scr._btn_file_issue.isEnabled() is enabled
                assert not scr._btn_file_issue.isVisible() or enabled
                assert submitted == []


def test_routed_errors_get_ai_advice_and_the_raw_traceback_stays_local(
        qtbot, monkeypatch, manual_error_batches):
    """With hosted routing agreed to, the console shows the relay's advice
    beside the full traceback, exactly one /diagnose call is made, no /report
    is filed, and the outgoing body carries no private path fragment.
    """
    from spacr import crashreport
    from spacr.qt import preferences, terms
    from spacr.qt.ai import providers, settings
    from spacr.qt.ai.issue_report import _sync_automatic_failure_policy
    from tests.test_700_anonymous_error_reports import FakeRelay

    relay = FakeRelay()
    relay.routes["/diagnose"] = [(200, {"advice": "Check the image path."})]
    monkeypatch.setattr(crashreport, "_open_url", relay)
    with built("mask") as scr:
        preferences.set_preferred_provider(providers.HOSTED_PROVIDER_NAME)
        terms.record_agreement()
        settings.set_route_errors_through_ai(True)
        settings.set_auto_file_issues(False)
        crashreport.set_error_report_mode("summarize")
        _sync_automatic_failure_policy()
        scr.show()
        qtbot.waitExposed(scr)
        workers = []
        dispatch = crashreport._dispatch_error_now

        def observe(*args, **kwargs):
            worker = dispatch(*args, **kwargs)
            if worker is not None:
                workers.append(worker)
            return worker

        monkeypatch.setattr(crashreport, "_dispatch_error_now", observe)
        traceback = ("RuntimeError: cannot read "
                     "/home/PRIVATE_PERSON/SECRET-TRACEBACK.tif")
        try:
            scr._on_pipeline_error(traceback)
            manual_error_batches()
            assert workers
            qtbot.waitUntil(lambda: all(not w.is_alive() for w in workers),
                            timeout=20000)
            qtbot.waitUntil(lambda: "Check the image path." in console_text(
                scr._console), timeout=20000)
            assert traceback in console_text(scr._console)
            assert len(relay.calls_to("/diagnose")) == 1
            assert relay.calls_to("/report") == []
            outgoing = str(relay.calls_to("/diagnose")[0]["body"])
            assert "PRIVATE_PERSON" not in outgoing
            assert "SECRET-TRACEBACK" not in outgoing
        finally:
            qtbot.waitUntil(lambda: all(not w.is_alive() for w in workers),
                            timeout=20000)
            scr._console.shutdown()


def _approve_and_capture(mp, seen, submit=None):
    """Accept the preview dialog and record what ``build_report`` receives."""
    from spacr.qt.ai.issue_preview import IssuePreviewDialog

    def _build(tb, active_app="", settings=None, include_log_tail=True,
               ai_response=""):
        # ai_response is NAMED: a stub swallowing **kwargs could not notice
        # the screen dropping an argument of the public bug report.
        seen.update(tb=tb, app=active_app, settings=settings,
                    include_log_tail=include_log_tail, ai_response=ai_response)
        return {"title": "t", "body": "b", "fingerprint": "f"}

    mp.setattr("spacr.qt.ai.issue_report.build_report", _build)
    mp.setattr(IssuePreviewDialog, "exec", lambda _self: QDialog.Accepted)
    mp.setattr("spacr.qt.ai.issue_report.submit_report", submit or (
        lambda _r: "https://github.com/EinarOlafsson/spacr/issues/1"))


class _HostileWidgets:
    def items(self):
        raise RuntimeError("widget map corrupted")

    def get(self, key, default=None):
        return default


@pytest.mark.usefixtures("console_info_enabled")
def test_filing_an_issue_snapshots_what_it_can_and_reports_the_outcome(
        qtbot, subtests):
    """With no traceback the button is inert. Otherwise the report is built
    with a settings snapshot of readable widgets only ({} when the model is
    gone or its widget map raises), the handoff is logged, and a submit that
    raises is reported in the console rather than swallowed. Explain-error
    is a no-op without a traceback and otherwise opens the flow and emits.
    """
    def readable_only(scr):
        box = QCheckBox()
        box.setChecked(True)
        scr._settings_model._widgets = {"a_flag": box, "a_label": QLabel("x")}

    def no_model(scr):
        scr._settings_model = None

    def hostile(scr):
        scr._settings_model._widgets = _HostileWidgets()

    cases = [
        ("hostile widget map", hostile, {},
         "[issue] report handoff completed"),
        ("unreadable widget types skipped", readable_only, {"a_flag": True},
         None),
        ("no settings model", no_model, {}, None),
    ]
    for name, spoil, snapshot, logged in cases:
        with subtests.test(case=name), pytest.MonkeyPatch.context() as mp:
            seen = {}
            _approve_and_capture(mp, seen)
            with built("mask") as scr:
                scr._last_error_text = "TB"
                spoil(scr)
                scr._on_file_issue()
                settle(qtbot, scr)
                assert seen["settings"] == snapshot
                assert seen["tb"] == "TB" and seen["app"] == "mask"
                if logged:
                    assert logged in console_text(scr._console)

    with subtests.test(case="submit raises"), \
            pytest.MonkeyPatch.context() as mp:
        mp.setattr(AUTO_FILE, lambda: True)
        mp.setattr(ROUTE, lambda: False)
        # The REAL build_report runs here, so the fingerprint is taken over a
        # traceback whose exception line is neither indented nor "Traceback".
        from spacr.qt.ai import issue_report
        from spacr.qt.ai.issue_preview import IssuePreviewDialog
        reports = []

        def failing_submit(report):
            reports.append(report)
            raise RuntimeError("github down")

        mp.setattr(IssuePreviewDialog, "exec", lambda _self: QDialog.Accepted)
        mp.setattr("spacr.qt.ai.issue_report.submit_report", failing_submit)
        tb = ('Traceback (most recent call last):\n'
              '  File "pipeline.py", line 3, in run\n'
              'ValueError: SUBMIT-FAILS')
        with built("mask") as scr:
            scr._on_pipeline_error(tb)
            scr._on_file_issue()
            settle(qtbot, scr)
            assert "[issue] auto-file failed: github down" in console_text(
                scr._console)
            assert len(reports) == 1
            fingerprint = issue_report._traceback_hash(tb)
            assert fingerprint != issue_report._traceback_hash(
                tb.replace("ValueError", "KeyError"))
            assert fingerprint == issue_report._traceback_hash(
                tb.replace("SUBMIT-FAILS", "other message"))

    with subtests.test(case="no traceback"), \
            pytest.MonkeyPatch.context() as mp:
        mp.setattr("spacr.qt.ai.issue_report.file_issue",
                   lambda *a, **k: pytest.fail("must not file"))
        with built("mask") as scr:
            before = console_text(scr._console)
            scr._on_file_issue()
            assert console_text(scr._console) == before

            emitted, opened = [], []
            scr.error_explain_requested.connect(
                lambda tb, app: emitted.append((tb, app)))
            scr._on_explain_error()
            assert emitted == []
            scr._console.open_error_flow = lambda *a, **k: opened.append((a, k))
            scr._last_error_text = "TB-TEXT"
            scr._on_explain_error()
            assert opened == [(("TB-TEXT", "mask"), {})]
            assert emitted == [("TB-TEXT", "mask")]


@pytest.mark.usefixtures("console_info_enabled")
def test_the_ai_switch_picks_a_provider_without_overriding_the_users(
        qtbot, subtests):
    """``_wanted_provider`` honours the preference only among configured
    providers; switching AI on activates the console with the user's chosen
    provider or else the first configured one, refuses to stay on with none
    installed, and switching off deactivates but keeps the provider. The
    old provider menu/dialog hooks are gone from the screen.
    """
    from spacr.qt import preferences as prefs

    claude, codex = (_FakeProvider("claude", "Claude Code"),
                     _FakeProvider("codex", "Codex"))

    for name, configured, preferred, wanted in (
            ("preference honoured", [claude, codex], "codex", "codex"),
            ("nothing chosen", [claude, codex], "", ""),
            ("none installed", [], "claude", "")):
        with subtests.test(case=name), pytest.MonkeyPatch.context() as mp:
            mp.setattr("spacr.qt.ai.configured_providers",
                       lambda _c=configured: _c)
            was = prefs.get_preferred_provider()
            try:
                with built("mask") as scr:
                    prefs.set_preferred_provider(preferred)
                    assert scr._wanted_provider() == wanted
            finally:
                prefs.set_preferred_provider(was)

    # name, configured, ai on by default, preset provider, toggles,
    # expected active, expected provider
    cases = [
        ("autoselect first", [codex], False, None, [True], True, "codex"),
        ("keeps the chosen one", [claude, codex], None, "codex", [True], True,
         "codex"),
        ("off keeps provider", [codex], None, None, [True, False], False,
         "codex"),
    ]
    for name, configured, default_on, preset, toggles, active, provider in cases:
        with subtests.test(case=name), pytest.MonkeyPatch.context() as mp:
            mp.setattr("spacr.qt.ai.configured_providers",
                       lambda _c=configured: _c)
            mp.setattr("spacr.qt.ai.get_provider", lambda n, _c=configured: next(
                (p for p in _c if p.name == n), None))
            if default_on is not None:
                mp.setattr("spacr.qt.preferences.get_ai_on_by_default",
                           lambda: default_on)
            with built("mask") as scr:
                if default_on is False:
                    assert scr._console._current_provider_name is None
                if preset:
                    scr._console.set_ai_provider(preset)
                for state in toggles:
                    scr._ai_switch.setChecked(state)
                assert scr._console._ai_active is active
                assert scr._console._current_provider_name == provider

    with subtests.test(case="no provider refuses to stay on"), \
            pytest.MonkeyPatch.context() as mp:
        mp.setattr("spacr.qt.ai.configured_providers", lambda: [])
        with built("mask") as scr:
            scr._ai_switch.setChecked(True)
            assert "No vendor CLI installed" in console_text(scr._console)
            assert scr._ai_switch.isChecked() is False

    with subtests.test(case="menu hooks retired"):
        assert not hasattr(AppScreen, "_on_open_providers_dialog")
        assert not hasattr(AppScreen, "_refresh_ai_menu")


def test_hover_hints_go_to_a_fixed_strip_that_keeps_the_last_hint(
        qtbot, subtests):
    """The strip has a fixed height that fits its text so Run/Stop never move;
    hovering a label puts its hint and a clickable API link there without the
    popup, Leave keeps it (a hold timer clears it later); unregistered widgets
    and other events are ignored; before the strip exists the popup is used;
    the popup survives its anchor being deleted.
    """
    import shiboken6
    from spacr.qt.widgets.hover_tooltip import HoverTooltip

    with built("mask") as scr:
        scr.resize(1200, 900)
        scr.show()
        qtbot.wait(1)
        tip = HoverTooltip.instance()
        label, hint = next(iter(scr._hint_map.items()))

        with subtests.test(case="strip height is fixed"):
            reserved = scr._hint_strip.minimumHeight()
            assert scr._hint_strip.maximumHeight() == reserved
            assert reserved >= scr._hint_strip.heightForWidth(
                scr._hint_strip.width())
            before = (scr._btn_run.pos(), scr._btn_stop.pos())
            scr._hint_strip.setText(
                "A deliberately long setting description. " * 30)
            qtbot.wait(1)
            assert (scr._btn_run.pos(), scr._btn_stop.pos()) == before
            scr._release_the_hint()

        with subtests.test(case="enter, leave, release"):
            assert scr._hint_strip.text() == scr._default_hint()
            scr._show_hover_hint(label)
            assert tip._anchor is not label
            shown = scr._hint_strip.text()
            assert shown != scr._default_hint()
            opening = shown.split("<br>")[0].rstrip("…")[:30]
            assert opening and opening.split()[0] in hint
            assert "<a href=" in shown
            assert scr._hint_strip.openExternalLinks()
            scr.eventFilter(label, QEvent(QEvent.Leave))
            assert scr._hint_strip.text() == shown
            assert scr._hint_hold_timer.isActive()
            scr._release_the_hint()
            assert scr._hint_strip.text() == scr._default_hint()
            tip.cancel_hide()

        with subtests.test(case="unregistered widget and other events"):
            scr._hint_strip.setText("untouched")
            scr.eventFilter(QLabel("stray"), QEvent(QEvent.Enter))
            assert scr._hint_strip.text() == "untouched"
            assert scr.eventFilter(label, QEvent(QEvent.Show)) is False
            scr._release_the_hint()

        with subtests.test(case="no strip yet uses the popup"):
            strip = scr._hint_strip
            del scr._hint_strip
            try:
                scr._show_hover_hint(label)
                assert tip._anchor is label
                scr.eventFilter(label, QEvent(QEvent.Leave))
                assert tip._hide_timer.isActive()
            finally:
                scr._hint_strip = strip
                tip.cancel_hide()
            assert strip.text() == scr._default_hint()

    with subtests.test(case="deleted anchor"):
        scr = AppScreen("mask")
        label = next(iter(scr._hint_map))
        tip = HoverTooltip.instance()
        tip.show_for(label, "<b>anything</b>", immediate=True)
        assert tip._anchor is label
        shiboken6.delete(scr)
        assert not shiboken6.isValid(label)
        tip._maybe_hide()
        assert not shiboken6.isValid(label)
