"""What the screen does with a failed run: classify it, explain it, report it.

The anonymous report goes through ``crashreport``'s real dispatch thread with
the network edge (``_enqueue``, ``_send_queued``) and the AI provider stood
in, so what is asserted is what would have left the machine. Every case owns
its own batcher, report folder and dispatch thread, and joins them before the
next case starts. The issue filer is driven on a real screen with the consent
dialog stood in.
"""
from __future__ import annotations

import contextlib
import logging
import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr.qt.screens.app_screen import AppScreen                # noqa: E402
from spacr.qt.widgets.console_panel import ConsolePanel          # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus       # noqa: E402

pytestmark = pytest.mark.qt


def _boom(*_args, **_kwargs):
    raise RuntimeError("the panel is half built")


class _Console:
    def __init__(self):
        self.notices = []
        self.stdout = []

    def append_notice(self, text, **kw):
        self.notices.append(text.format(**kw) if kw else text)

    def append_stdout(self, text, **_kw):
        self.stdout.append(text)


class _Outcome(_Console):
    _show_error_report_result = ConsolePanel._show_error_report_result

    def __init__(self):
        super().__init__()
        self.entries = []
        self.usage_refreshed = 0
        self._pending_report_entries = {}

    def fill_ai_entry(self, entry, text):
        self.entries.append(text)

    def refresh_ai_usage(self):
        self.usage_refreshed += 1


def test_each_report_outcome_gets_its_own_sentence(subtests):
    """The console says what became of an anonymous report, whatever it was.

    Invariant: the AI entry is filled with the explanation (noting a lighter
    tier), the used-up allowance, the failure to explain, or the relay's own
    words; the usage meter is refreshed after an explanation; each status
    (deduplicated, with/without AI summary, queued, dropped with or without
    a reason) gets its own notice; a meter or entry that raises never hides
    the answer or the issue number.
    """
    sent = {"explained": True, "status": "sent", "issue_number": 7}
    # (case, outcome, entry starts with / contains, notice contains, meter)
    explained = [
        ("explanation and lighter tier",
         dict(sent, explanation={"text": "Because.", "tier": "light"}),
         ("Because.", "lighter model"), None, 1),
        ("allowance used up", dict(sent, explanation={"refused": "quota"}),
         ("", "used up"), "daily AI allowance used", None),
        ("explanation failed", dict(sent, explanation={}),
         ("", "could not explain"), "without AI summary)", None),
        ("relay text fills the entry",
         {"ai_summary": "relay words", "status": "queued", "quota": True},
         ("relay words", ""), "daily limit resets", None),
    ]
    for name, outcome, (prefix, words), notice, meter in explained:
        with subtests.test(case=name):
            ns = types.SimpleNamespace(_console=_Outcome(),
                                       _ai_report_entry="entry")
            AppScreen._on_anonymous_report_sent(ns, outcome)
            entry = ns._console.entries[0]
            assert entry.startswith(prefix) and words in entry
            if notice:
                assert notice in ns._console.notices[-1]
            if meter is not None:
                assert ns._console.usage_refreshed == meter

    statuses = [
        ({"status": "sent", "issue_number": 5, "deduplicated": True},
         "already has issue #5"),
        ({"status": "sent", "issue_number": 5, "ai_summary_attached": True},
         "with AI summary"),
        ({"status": "sent", "issue_number": 5}, "Reported anonymously as"),
        ({"status": "queued"}, "will retry later"),
        ({"status": "dropped", "detail": "bad"}, "did not accept"),
        ({"status": "dropped"}, "unknown"),
    ]
    for outcome, notice in statuses:
        with subtests.test(case=f"status {outcome}"):
            ns = types.SimpleNamespace(_console=_Outcome(),
                                       _ai_report_entry=None)
            AppScreen._on_anonymous_report_sent(ns, outcome)
            assert notice in ns._console.notices[-1]

    with subtests.test(case="usage meter raises"):
        console = _Outcome()
        console.refresh_ai_usage = _boom
        AppScreen._on_anonymous_report_sent(
            types.SimpleNamespace(_console=console, _ai_report_entry="e"),
            {"explained": True, "explanation": {"text": "ok"}})
        assert console.entries == ["ok"]

    with subtests.test(case="entry cannot be filled"):
        console = _Outcome()
        console.fill_ai_entry = _boom
        AppScreen._on_anonymous_report_sent(
            types.SimpleNamespace(_console=console, _ai_report_entry="e"),
            {"explained": True, "explanation": {"text": "ok"},
             "status": "sent", "issue_number": 3})
        assert "issue #3" in console.notices[-1]


class _Jobs:
    def __init__(self, outcomes):
        self.thread = None
        self.outcomes = outcomes

    def job(self):
        assert self.thread is not None
        self.thread.join(timeout=5)
        assert not self.thread.is_alive()
        return next((outcome for outcome in reversed(self.outcomes)
                     if not outcome.get("pending")), None)


class _ReportConsole(_Outcome):
    _run_function = "mask"

    def __init__(self, as_text_fails=False):
        super().__init__()
        self.begun = []
        self.filled = []
        self.as_text_fails = as_text_fails

    def as_text(self):
        if self.as_text_fails:
            raise RuntimeError("closed")
        return "log"

    def begin_ai_entry(self, thinking=True):
        self.begun.append(thinking)
        return "entry"

    def fill_ai_entry(self, entry, text):
        self.filled.append(text)


@contextlib.contextmanager
def _owned_dispatch(m, folder):
    """A private batcher, report folder and dispatch cache for one case.

    On leaving, the dispatch thread and the batcher timer are joined, so no
    case's thread can deliver into the next one.
    """
    from spacr import crashreport

    state = types.SimpleNamespace(outcomes=[])
    state.jobs = _Jobs(state.outcomes)
    state.batcher = crashreport._ErrorBatcher()
    m.setattr(crashreport, "_reports_dir", lambda: folder / "reports")
    m.setattr(crashreport, "_ERROR_BATCHER", state.batcher)
    m.setattr(crashreport, "_ERROR_DISPATCH_SEEN", {})
    m.setattr(crashreport, "_ERROR_OCCURRENCE_POLICIES", {})
    m.setattr(crashreport, "_ERROR_POLICY_CACHE", {})
    try:
        yield state
    finally:
        if state.jobs.thread is not None:
            state.jobs.thread.join(timeout=5)
            assert not state.jobs.thread.is_alive()
        if state.batcher.timer is not None:
            state.batcher.timer.cancel()
            state.batcher.timer.join(timeout=5)
            assert not state.batcher.timer.is_alive()


def _report_screen(state, console=None, started=None):
    return types.SimpleNamespace(
        _last_error_text="Traceback\nValueError: x",
        _last_error_console_text="", _last_error_occurrence="",
        _last_error_problems=["a", "b"],
        _run_settings={"dry_run": True}, _run_started_at=started,
        _console=console or _ReportConsole(), _jobs=state.jobs,
        app_key="mask", _kind_of_failure=lambda tb: "fault",
        _on_anonymous_report_sent=lambda outcome: None,
        _screen_name=lambda: "Mask",
        _report_summary_provider=lambda: object())


def _consenting(m, state, console, **overrides):
    """Every consent given, the network edge and the AI stood in."""
    from spacr import crashreport
    from spacr.qt import preferences, terms
    from spacr.qt.ai import providers, settings

    calls = {"queued": [], "provider_requests": []}
    m.setattr(crashreport, "network_allowed", lambda: True)
    terms.record_agreement()
    preferences.set_preferred_provider("claude")
    preferences.set_issue_prompt_mode("always")
    settings.set_route_errors_through_ai(True)
    settings.set_auto_file_issues(True)
    crashreport.set_error_report_mode("all")
    original_summary = crashreport.public_summary
    original_dispatch = crashreport._dispatch_error

    def public_summary(*args, **kwargs):
        calls["summary"] = dict(kwargs)
        return original_summary(*args, **kwargs)

    def dispatch(*args, **kwargs):
        state.jobs.thread = original_dispatch(*args, **kwargs)
        return state.jobs.thread

    def deliver(outcome):
        state.outcomes.append(dict(outcome))
        console._show_error_report_result(outcome)

    def enqueue(summary, **kwargs):
        calls["queued"].append(dict(summary))

    def send_queued(queued, summary, timeout, *, allowed):
        assert allowed()
        calls["submitted"] = {"status": "sent", "summary": dict(summary)}
        return {"status": "sent"}

    table = {"public_summary": public_summary, "_dispatch_error": dispatch,
             "_deliver_error_result": deliver, "_enqueue": enqueue,
             "_send_queued": send_queued}
    table.update(overrides)
    for name, fn in table.items():
        m.setattr(crashreport, name, fn)

    def report_summary_result(summary, provider, timeout):
        calls["provider_requests"].append(dict(summary))
        return "why", ""

    m.setattr(providers, "_report_summary_result", report_summary_result)
    return calls


def test_an_anonymous_report_is_not_sent_when_it_cannot_be(
        monkeypatch, subtests, tmp_path):
    """Nothing leaves the machine unless there is a report and leave to send.

    Invariant: no error text, unreadable settings, a mode that neither
    explains nor sends, and a summary that cannot be built each answer False
    or start nothing -- no dispatch thread, no outcome, no queue claim, no
    notice -- and leave the error text in place for a later attempt.
    """
    from spacr import crashreport

    def refuse_summary(**_kw):
        raise OSError("no disk")

    cases = [
        ("no error text", None, False),
        ("settings cannot be read", {"error_report_mode": _boom}, False),
        ("neither explains nor sends",
         {"mode_explains": lambda mode=None: False,
          "mode_sends": lambda kind, mode: False}, False),
        ("summary cannot be built", {"public_summary": refuse_summary}, None),
    ]
    for index, (name, overrides, returns) in enumerate(cases):
        with subtests.test(case=name), monkeypatch.context() as m, \
                _owned_dispatch(m, tmp_path / str(index)) as state:
            console = _ReportConsole()
            ns = _report_screen(state, console)
            if overrides is None:
                ns._last_error_text = ""
                assert AppScreen._send_the_anonymous_report(ns) is False
                assert state.jobs.thread is None
                continue
            calls = _consenting(m, state, console, **overrides)
            answer = AppScreen._send_the_anonymous_report(ns)
            if returns is not None:
                assert answer is returns
                assert state.jobs.thread is None
            else:
                assert state.jobs.job() is None
            assert state.outcomes == [] and console.notices == []
            assert calls["queued"] == [] and "submitted" not in calls
            assert ns._last_error_text == "Traceback\nValueError: x"
            if name == "settings cannot be read":
                assert crashreport._ERROR_POLICY_CACHE[
                    "affirmed_policy"] == {}
                assert crashreport._automatic_error_policy(
                    "fault")[1:] == (False, False)


def test_a_sent_report_carries_the_explanation_it_was_given(
        monkeypatch, subtests, tmp_path):
    """The explanation shown is the one that travels; the rest stays local.

    Invariant: a saved explanation is shown (not thinking), attached and
    sent with its cause, without asking the AI again, the summary leaves out
    the run, its settings, log and timing, and the screen's own state is
    untouched; a saved lookup that raises asks the AI afresh (thinking) and
    sends its answer; an AI that raises is recorded as refused and the
    report still goes without an AI summary.
    """
    from spacr.qt.ai import providers

    def refuse(*_args):
        raise RuntimeError("disk")

    with subtests.test(case="saved explanation travels"), \
            monkeypatch.context() as m, \
            _owned_dispatch(m, tmp_path / "saved") as state:
        console = _ReportConsole(as_text_fails=True)
        calls = _consenting(m, state, console, saved_explanation=lambda key: {
            "text": "kept", "tier": "full", "cause": "root"})
        ns = _report_screen(state, console, started=1.0)
        assert AppScreen._send_the_anonymous_report(ns) is True
        outcome = state.jobs.job()
        assert console.begun == [False] and console.filled == ["kept"]
        assert outcome["explained"] is True and outcome["kind"] == "fault"
        assert outcome["ai_summary_attached"] is True
        sent = calls["submitted"]["summary"]
        assert sent["ai_summary"] == "kept" and sent["cause"] == "root"
        assert calls["provider_requests"] == []
        assert calls["summary"]["message"].startswith(
            "The run did not start: 2 settings problems.")
        assert not {"run", "run_settings", "log_text"} & \
            calls["summary"].keys()
        assert not {"run", "run_settings", "log_text", "elapsed_s"} & \
            sent.keys()
        assert ns._run_settings == {"dry_run": True}
        assert ns._run_started_at == 1.0
        assert ns._last_error_text == "Traceback\nValueError: x"

    with subtests.test(case="saved lookup raises, AI asked afresh"), \
            monkeypatch.context() as m, \
            _owned_dispatch(m, tmp_path / "afresh") as state:
        console = _ReportConsole()
        calls = _consenting(m, state, console, saved_explanation=refuse)
        ns = _report_screen(state, console)
        assert AppScreen._send_the_anonymous_report(ns) is True
        outcome = state.jobs.job()
        assert console.begun == [True]
        assert outcome["explanation"]["text"] == "why"
        assert calls["submitted"]["summary"]["ai_summary"] == "why"
        assert len(calls["provider_requests"]) == 1

    with subtests.test(case="AI raises, report still goes"), \
            monkeypatch.context() as m, \
            _owned_dispatch(m, tmp_path / "offline") as state:
        console = _ReportConsole()
        calls = _consenting(m, state, console)
        m.setattr(providers, "explain_failure", lambda summary, provider:
                  refuse())
        ns = _report_screen(state, console)
        AppScreen._send_the_anonymous_report(ns)
        outcome = state.jobs.job()
        assert outcome["explanation"]["refused"] == "error"
        assert outcome["ai_summary_attached"] is False
        assert "submitted" in calls


def test_a_failure_is_classified_by_what_caused_it(monkeypatch, subtests):
    """A wrong path in the settings is the user's; everything else a fault.

    Invariant: an empty traceback is no settings error; a
    ConfigurationError always is; a missing file is one only when its path
    lies inside a folder the settings name; a classifier that raises makes
    the failure a fault; an explanation check that raises answers no; the
    console's own provider is the report's, and a missing console gives none.
    """
    from spacr import crashreport

    inside = ("Traceback\nFileNotFoundError: [Errno 2] No such file or "
              "directory: '/data/exp/plate1/img.tif'")
    elsewhere = ("Traceback\nFileNotFoundError: [Errno 2] No such file or "
                 "directory: '/usr/lib/thing.so'")
    paths = [
        ("empty traceback", "", {}, False),
        ("blank traceback", "\n \n", {}, False),
        ("configuration error", "Traceback\nConfigurationError: src is wrong",
         {}, True),
        ("other error", "Traceback\nValueError: no", {}, False),
        ("missing file without a path", "Traceback\nFileNotFoundError: gone",
         {}, False),
        ("missing file in a settings folder", inside,
         {"src": ["/data/exp", 3], "n": 4, "note": "x"}, True),
        ("missing file elsewhere", elsewhere, {"src": "/data/exp"}, False),
    ]
    for name, traceback_text, settings, is_settings in paths:
        with subtests.test(case=name):
            assert AppScreen._is_a_settings_path_error(
                traceback_text, settings) is is_settings

    def refuse(*_a):
        raise RuntimeError("no")

    with subtests.test(case="classifier raises"), monkeypatch.context() as m:
        m.setattr(crashreport, "classify_failure", refuse)
        ns = types.SimpleNamespace(
            _is_a_settings_path_error=AppScreen._is_a_settings_path_error,
            _settings_snapshot=lambda: {})
        assert AppScreen._kind_of_failure(ns, "tb") == crashreport.KIND_FAULT

    with subtests.test(case="settings path error is the user's"):
        ns = types.SimpleNamespace(
            _is_a_settings_path_error=lambda tb, s: True,
            _settings_snapshot=lambda: {})
        assert AppScreen._kind_of_failure(ns, "tb") == \
            crashreport.KIND_USER_ERROR

    with subtests.test(case="explanation check raises"), \
            monkeypatch.context() as m:
        m.setattr(crashreport, "mode_explains", refuse)
        assert AppScreen._the_failure_is_explained_after_the_run(
            types.SimpleNamespace()) is False

    with subtests.test(case="report provider"):
        chosen = object()
        assert AppScreen._report_summary_provider(types.SimpleNamespace(
            _console=types.SimpleNamespace(
                _current_provider=lambda: chosen))) is chosen
        assert AppScreen._report_summary_provider(
            types.SimpleNamespace(_console=None)) is None


def test_run_start_notices_say_what_will_happen(subtests):
    """The refusal watch and the permutation banner are read once, honestly.

    Invariant: stopping the refusal watch returns the readable refusal
    messages (skipping records that cannot be read), keeps the first call
    site and the settings problems, and a second stop returns nothing; the
    permutation banner reports a non-number or a count below one as an
    invalid configuration (also in the console) and a valid count with its
    smallest attainable p value.
    """
    class Bad:
        def getMessage(self):
            raise ValueError("bad")

    class Good:
        spacr_call_site = ["a.py", 3]

        def getMessage(self):
            return "refused"

    class Problem:
        spacr_problems = ["p1"]

    with subtests.test(case="refusal watch"):
        ns = types.SimpleNamespace(
            _refusal_watch=(logging.NullHandler(), [Bad(), Good()],
                            logging.NullHandler(), [Problem()]),
            _refusal_call_site=[], _settings_problems=[])
        assert AppScreen._stop_watching_refusals(ns) == ["refused"]
        assert ns._refusal_call_site == ["a.py", 3]
        assert ns._settings_problems == ["p1"]
        assert ns._refusal_watch is None
        assert AppScreen._stop_watching_refusals(ns) == []

    banners = [
        ("not a number", "many", ["configuration is invalid", "'many'"], True),
        ("below one", 0, ["guide_permutations=0"], False),
        ("valid count", 99, ["99 within-plateID", "0.01"], False),
    ]
    for name, count, words, invalid in banners:
        with subtests.test(case=f"permutations: {name}"):
            ns = types.SimpleNamespace(_console=_Console())
            note = AppScreen._say_what_the_permutation_will_do(
                ns, {"guide_permutations": count})
            for word in words:
                assert word in note
            if invalid:
                assert "configuration is invalid" in ns._console.notices[-1]


@pytest.fixture
def screen(qtbot):
    widget = AppScreen("regression")
    try:
        yield widget
    finally:
        retire_pyqtgraph_menus(widget)
        widget.close()
        widget.deleteLater()


def test_filing_an_issue_goes_through_the_consent_preview(
        screen, monkeypatch, subtests, qtbot):
    """The preview IS the consent boundary.

    Invariant: a cancelled preview sends nothing and says so; an approved
    report is built with every setting that was running (list settings
    included) and sent as approved; a send that raises on the worker comes
    back as a displayed failure exactly once.
    """
    from PySide6.QtWidgets import QDialog, QLineEdit
    from spacr.qt import preferences
    from spacr.qt.ai import issue_preview, issue_report

    def preview(m, accepted):
        class Preview:
            # `console` and `traceback_text` are NAMED: the Diagnose button
            # is inert without them, and a stub taking **kwargs could not
            # notice the screen stopping passing them.
            def __init__(self, built, parent=None, console=None,
                         traceback_text=""):
                self.built = built

            def exec(self):
                return QDialog.Accepted if accepted else QDialog.Rejected

            def approved_report(self):
                return self.built

        m.setattr(issue_preview, "IssuePreviewDialog", Preview)

    def arrange(m, said, accepted, **console):
        screen._console = types.SimpleNamespace(
            append_notice=lambda text, **kw: said.append(text), **console)
        m.setattr(screen, "_last_error_text", "Traceback: cellpose exploded",
                  raising=False)
        m.setattr(preferences, "get_issue_prompt_mode", lambda: "ask")
        preview(m, accepted)
        m.setattr(screen._jobs, "submit", lambda work, done: done(work()))

    with subtests.test(case="cancelled preview"), monkeypatch.context() as m:
        said = []
        arrange(m, said, accepted=False)
        m.setattr(issue_report, "submit_report",
                  lambda _r: pytest.fail("a report was sent"))
        screen._on_file_issue()
        assert any("cancelled" in line for line in said)

    with subtests.test(case="approved report carries the settings"), \
            monkeypatch.context() as m:
        plain = QLineEdit()
        qtbot.addWidget(plain)
        plain.setText("/data/plate1")
        m.setitem(screen._settings_model._widgets, "a_plain_text_setting",
                  plain)
        arrange(m, [], accepted=True)
        built = {}
        m.setattr(issue_report, "build_report", lambda text, **kwargs:
                  built.update(kwargs) or {"body": text})
        sent = []
        m.setattr(issue_report, "submit_report",
                  lambda report: sent.append(report) or "https://x/1")
        screen._on_file_issue()
        assert built["settings"]["a_plain_text_setting"] == "/data/plate1"
        assert sent == [{"body": "Traceback: cellpose exploded"}]

    with subtests.test(case="a send that raises comes back as a failure"), \
            monkeypatch.context() as m:
        said, failures = [], []
        arrange(m, said, accepted=True,
                _display_error=lambda text: (failures.append(text),
                                             said.append(text)),
                append_error=lambda text: pytest.fail(
                    "report failure was captured again"))
        m.setattr(issue_report, "build_report",
                  lambda text, **kwargs: {"body": text})
        m.setattr(issue_report, "submit_report", _boom)
        screen._on_file_issue()
        assert any("auto-file failed" in text for text in said)
        assert failures == [
            "[issue] auto-file failed: the panel is half built\n"]
