"""With issue reporting set to 'always', a failed run files its own report.

The maintainer's decision, 2026-09-19, after issue #117: "Do real
auto-filing, and make this the default, and add the user agreeing to this in
the user agreement, if set to always."

Until then 'always' was inert. 807ba9e0a (instruction 45) removed automatic
filing, and nothing read the choice afterwards, so it behaved like 'ask'.

These tests hold the decision down:

* a profile that never chose files automatically, and one that chose keeps
  its choice;
* the report filed is the one "File as issue" would build, with the
  redaction its preview applies by default, and with this computer's login
  and host names removed;
* one crash is filed once: never twice from one profile, and as a comment
  when an open issue already carries its fingerprint;
* without a GitHub sign-in nothing is sent and no browser opens;
* an AI provider's error is never attached as its analysis;
* the terms of use say all of this, and point at the setting that
  changes it.

The network is faked at ``github_auth._HTTP_OPEN``, the one seam every
GitHub request goes through, and the sign-in at ``github_auth.resolve_token``.
Nothing here can reach github.com, and ``HOME`` is a temporary folder.
"""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

pytest.importorskip("PySide6")

from spacr.qt.ai import github_auth, issue_report  # noqa: E402

USER = "jakobk"
HOST = "jaks-mbp.lab.example.org"
TOKEN = "ghp_" + "Z" * 36
FILED_URL = "https://github.com/EinarOlafsson/spacr/issues/501"
OPEN_URL = "https://github.com/EinarOlafsson/spacr/issues/88"


class _Resp(io.BytesIO):
    """What ``urlopen`` hands back, as a context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeGitHub:
    """api.github.com, in-process.

    :param open_issue: what the fingerprint search finds, or ``None``.
    :param create_fails: answer the new-issue POST with HTTP 500.
    """

    def __init__(self, open_issue=None, create_fails=False):
        self.open_issue = open_issue
        self.create_fails = create_fails
        self.requests = []

    def __call__(self, req, timeout=None):
        payload = json.loads(req.data.decode("utf-8")) if req.data else None
        self.requests.append({"method": req.get_method(),
                              "url": req.full_url, "json": payload})
        if "/search/issues" in req.full_url:
            items = [self.open_issue] if self.open_issue else []
            return _Resp(json.dumps({"items": items}).encode())
        if req.full_url.endswith("/comments"):
            return _Resp(json.dumps(
                {"html_url": OPEN_URL + "#issuecomment-1"}).encode())
        if req.full_url.endswith("/repos/EinarOlafsson/spacr/issues"):
            if self.create_fails:
                raise urllib.error.HTTPError(
                    req.full_url, 500, "Server Error", {},
                    io.BytesIO(json.dumps(
                        {"message": f"broken, token {TOKEN}"}).encode()))
            return _Resp(json.dumps({"html_url": FILED_URL}).encode())
        raise AssertionError(f"unexpected request {req.full_url}")

    def posted(self, suffix):
        """The JSON bodies POSTed to URLs ending in ``suffix``."""
        return [r["json"] for r in self.requests
                if r["method"] == "POST" and r["url"].endswith(suffix)]


@pytest.fixture
def machine(monkeypatch, tmp_path):
    """A computer whose login name and host name are known, and a log.

    :returns: the home folder.
    """
    home = tmp_path / "home" / USER
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr("getpass.getuser", lambda: USER)
    monkeypatch.setattr("socket.gethostname", lambda: HOST)
    monkeypatch.setattr("platform.node", lambda: HOST)
    log = home / "spacr.log"
    log.write_text(f"opened {home}/plates/plate_01 as {USER} on {HOST}\n"
                   "SECRET-LOG-LINE\n", encoding="utf-8")
    monkeypatch.setattr("spacr.qt.logging_util.log_path", lambda: log)
    return home


@pytest.fixture
def signed_in(monkeypatch):
    """A GitHub sign-in, without the ``gh`` CLI."""
    monkeypatch.setattr(github_auth, "resolve_token", lambda: (TOKEN, "env"))


def _traceback(home) -> str:
    """A traceback whose message and frames name this computer."""
    return (
        "Traceback (most recent call last):\n"
        f'  File "{home}/miniforge3/envs/spacr/lib/python3.12/site-packages/'
        'spacr/qt/bridge.py", line 1290, in run\n'
        "    payload = self._fn(self._settings)\n"
        '  File "/opt/homebrew/lib/python3.12/site-packages/spacr/core.py", '
        "line 790, in generate_cellpose_masks_sam\n"
        "    with np.load(path) as data:\n"
        f"ValueError: cannot read {home}/plates/plate_01/masks.npz for {USER} "
        f"on {HOST}; see /srv/lab/{USER}/experiments.db "
        f"GITHUB_TOKEN={TOKEN}\n")


# ---------------------------------------------------------------------------
# Who files automatically
# ---------------------------------------------------------------------------


class TestTheDefault:

    @pytest.fixture
    def fresh(self):
        """A profile that has never answered either question."""
        from spacr.qt import preferences
        from spacr.qt.ai import settings as ai_settings

        preferences._settings().remove(preferences._KEY_ISSUE_PROMPT)
        ai_settings._settings().remove(ai_settings._KEY_AUTO_ISSUE)
        return preferences, ai_settings

    def test_a_profile_that_never_chose_files_automatically(self, fresh):
        preferences, ai_settings = fresh

        assert preferences.get_issue_prompt_mode() == "always"
        assert ai_settings.get_auto_file_issues() is True

    @pytest.mark.parametrize("chosen", ["ask", "never", "always"])
    def test_a_choice_that_was_made_is_kept(self, fresh, chosen):
        preferences, _ai = fresh

        preferences.set_issue_prompt_mode(chosen)

        assert preferences.get_issue_prompt_mode() == chosen

    def test_switching_reporting_off_is_kept(self, fresh):
        _prefs, ai_settings = fresh

        ai_settings.set_auto_file_issues(False)

        assert ai_settings.get_auto_file_issues() is False

    def test_a_value_this_build_cannot_read_does_not_publish(self, fresh):
        """Somebody chose it. Reading it as 'always' would publish without
        a preview on a choice nobody can see."""
        preferences, _ai = fresh

        preferences._settings().setValue(preferences._KEY_ISSUE_PROMPT,
                                         "only-on-tuesdays")

        assert preferences.get_issue_prompt_mode() == "ask"

    def test_first_run_setup_shows_always_to_a_new_profile(self, fresh,
                                                           qtbot):
        """The slide is where the choice is made, so it must show the
        default the profile will get, and say what it does."""
        from PySide6.QtWidgets import QComboBox

        from spacr.qt.widgets.setup_slides import SLIDES, SetupSlides

        slides = SetupSlides()
        qtbot.addWidget(slides)
        editor = slides._editors["issue_prompt"]
        assert isinstance(editor, QComboBox)
        assert editor.currentData() == "always"
        assert slides.answers()["issue_prompt"] == "always"

        blurb = dict((title, text) for title, text, _k in SLIDES)[
            "When something breaks"]
        assert "automatically" in blurb
        assert "public spaCR GitHub repository" in blurb
        assert "Nothing is ever sent" not in blurb

    def test_the_installer_s_explicit_answers_are_kept(self, fresh):
        """The desktop installer asks, and says a report is sent only on
        Send. Its answers are explicit choices, so they are kept."""
        from spacr.qt.install_consent import apply_choices

        preferences, ai_settings = fresh
        apply_choices({"report_issues": False})
        assert preferences.get_issue_prompt_mode() == "never"
        assert ai_settings.get_auto_file_issues() is False

        apply_choices({"report_issues": True})
        assert preferences.get_issue_prompt_mode() == "ask"


# ---------------------------------------------------------------------------
# What is filed
# ---------------------------------------------------------------------------


class TestTheRedaction:

    def test_this_computer_s_names_come_out(self, machine):
        words = dict(issue_report._identity_words())
        assert words[USER] == issue_report.USER_PLACEHOLDER
        assert words[HOST] == issue_report.HOST_PLACEHOLDER
        assert words["jaks-mbp"] == issue_report.HOST_PLACEHOLDER

        text = issue_report.redact_identity(
            f"owner {USER.upper()} at {HOST}, {USER}s and x{USER}")

        assert text == "owner <USER> at <HOST>, jakobks and xjakobk"

    def test_generic_and_short_names_stay(self, monkeypatch):
        monkeypatch.setattr("getpass.getuser", lambda: "root")
        monkeypatch.setattr("socket.gethostname", lambda: "localhost")
        monkeypatch.setattr("platform.node", lambda: "ab")
        monkeypatch.setenv("HOME", "/root")

        assert issue_report._identity_words() == []

    def test_the_automatic_report_leaves_nothing_identifying(self, machine):
        report = issue_report.build_report(
            _traceback(machine), active_app="mask",
            settings={"src": f"{machine}/plates/plate_01",
                      "pathogen_model_name": f"/srv/models/{USER}_toxo.cp",
                      "custom_regex": f"owner={USER}",
                      "api_key": "sk-ant-" + "q" * 20},
            include_log_tail=True)

        public = issue_report.public_report(report)
        text = public["title"] + "\n" + public["body"]

        for leak in (str(machine), USER, HOST, "jaks-mbp", "/srv/", "/opt/",
                     "experiments.db", TOKEN, "sk-ant-", "SECRET-LOG-LINE",
                     "plate_01"):
            assert leak not in text, leak
        for mark in ("<PATH>", "<USER>", "<HOST>", "<REDACTED>"):
            assert mark in text, mark
        assert public["fingerprint"] == report["fingerprint"]
        assert f"`{report['fingerprint']}`" in public["body"]

    def test_the_log_stays_on_this_computer_and_is_redacted(self, machine):
        issue_report.build_report(_traceback(machine), include_log_tail=True)

        saved = list((machine / ".spacr" / "reports").glob("log-*.txt"))
        assert len(saved) == 1
        kept = saved[0].read_text(encoding="utf-8")
        assert "SECRET-LOG-LINE" in kept
        assert USER not in kept and HOST not in kept

    def test_it_is_what_the_preview_sends_by_default(self, machine, qtbot):
        """'the SAME REDACTION the manual path uses': the body posted
        automatically is, character for character, the body a reviewer sends
        by pressing Send without editing."""
        from spacr.qt.ai.issue_preview import IssuePreviewDialog

        report = issue_report.build_report(
            _traceback(machine), active_app="mask",
            settings={"src": f"{machine}/plates/plate_01"})
        preview = IssuePreviewDialog(report)
        qtbot.addWidget(preview)

        sent_by_hand = preview.approved_report()
        sent_by_itself = issue_report.public_report(report)

        assert sent_by_itself["body"] == sent_by_hand["body"]
        assert sent_by_itself["fingerprint"] == sent_by_hand["fingerprint"]

    def test_the_sections_still_close(self, machine):
        """Issue #121 was filed with ``</summary>`` and ``</details>`` turned
        into ``<<PATH>``, so everything after the first section stayed
        folded inside it."""
        report = issue_report.build_report(
            _traceback(machine), settings={"src": f"{machine}/x"},
            include_log_tail=True,
            ai_response="The file is a macOS sidecar.")

        body = issue_report.public_report(report)["body"]

        assert "<<PATH>" not in body
        assert body.count("<details>") == body.count("</details>") == 3
        assert body.count("<summary>") == body.count("</summary>") == 3


class TestOneCrashOneFingerprint:

    def test_the_same_crash_on_two_computers(self):
        """The fingerprint used to hash every frame's full path, so the same
        crash on a Mac and on Windows never found each other's issue."""
        mac = (
            "Traceback (most recent call last):\n"
            '  File "/Users/jak/miniforge3/envs/spacr/lib/python3.12/'
            'site-packages/spacr/core.py", line 790, in generate\n'
            "    x()\n"
            "ValueError: pickled data in /Users/jak/a.npz\n")
        windows = (
            "Traceback (most recent call last):\n"
            '  File "C:\\ProgramData\\spacr\\Lib\\site-packages\\spacr\\'
            'core.py", line 802, in generate\n'
            "    x()\n"
            "ValueError: pickled data in D:\\b.npz\n")

        assert (issue_report.fingerprint_of(mac)
                == issue_report.fingerprint_of(windows))

    def test_a_different_module_is_a_different_crash(self):
        one = ('Traceback (most recent call last):\n'
               '  File "/x/spacr/core.py", line 1, in run\n'
               'ValueError: a\n')
        other = one.replace("spacr/core.py", "spacr/io.py")
        assert (issue_report.fingerprint_of(one)
                != issue_report.fingerprint_of(other))

    def test_it_is_the_one_the_report_carries(self, machine):
        tb = _traceback(machine)
        assert (issue_report.fingerprint_of(tb)
                == issue_report.build_report(tb)["fingerprint"])


# ---------------------------------------------------------------------------
# Filing, without a screen
# ---------------------------------------------------------------------------


class TestFilingWithoutReview:

    @pytest.fixture
    def report(self, machine):
        return issue_report.public_report(issue_report.build_report(
            _traceback(machine), active_app="mask"))

    def test_a_new_crash_opens_an_issue(self, monkeypatch, signed_in,
                                        report):
        github = FakeGitHub()
        monkeypatch.setattr(github_auth, "_HTTP_OPEN", github)

        outcome = issue_report.file_without_review(report)

        assert outcome == {"status": issue_report.FILED, "url": FILED_URL}
        [created] = github.posted("/repos/EinarOlafsson/spacr/issues")
        assert created["title"] == report["title"]
        assert created["body"] == report["body"]
        assert created["labels"] == ["auto-filed"]

    def test_an_open_issue_with_the_fingerprint_gets_a_comment(
            self, monkeypatch, signed_in, report):
        github = FakeGitHub(open_issue={"number": 88, "html_url": OPEN_URL})
        monkeypatch.setattr(github_auth, "_HTTP_OPEN", github)

        outcome = issue_report.file_without_review(report)

        assert outcome == {"status": issue_report.SEEN_AGAIN, "url": OPEN_URL}
        assert github.posted("/repos/EinarOlafsson/spacr/issues") == []
        [comment] = github.posted("/issues/88/comments")
        assert comment["body"].startswith("Seen again.")
        search = github.requests[0]["url"]
        assert report["fingerprint"] in search and "is%3Aopen" in search

    def test_signed_out_sends_nothing_and_opens_no_browser(
            self, monkeypatch, report):
        github = FakeGitHub()
        monkeypatch.setattr(github_auth, "_HTTP_OPEN", github)
        monkeypatch.setattr(github_auth, "resolve_token", lambda: ("", None))
        opened = []
        monkeypatch.setattr(issue_report, "open_issue_in_browser",
                            lambda url: opened.append(url) or True)

        outcome = issue_report.file_without_review(report)

        assert outcome == {"status": issue_report.SIGNED_OUT}
        assert github.requests == [] and opened == []

    def test_a_failure_says_why_without_the_token(self, monkeypatch,
                                                  signed_in, report):
        monkeypatch.setattr(github_auth, "_HTTP_OPEN",
                            FakeGitHub(create_fails=True))

        outcome = issue_report.file_without_review(report)

        assert outcome["status"] == issue_report.FAILED
        assert "500" in outcome["detail"]
        assert TOKEN not in outcome["detail"]

    def test_inside_a_test_run_the_real_transport_is_refused(self, report):
        """No monkeypatched transport: the session guard answers."""
        outcome = issue_report.file_without_review(report)
        assert outcome["status"] == issue_report.REFUSED

    def test_the_reviewed_path_still_falls_back_to_the_browser(
            self, monkeypatch, report):
        """'ask' is unchanged: without a sign-in, Send opens the form."""
        monkeypatch.setattr(github_auth, "_HTTP_OPEN", FakeGitHub())
        monkeypatch.setattr(github_auth, "resolve_token", lambda: ("", None))
        opened = []
        monkeypatch.setattr(issue_report, "open_issue_in_browser",
                            lambda url: opened.append(url) or True)

        url = issue_report.submit_report(report)

        assert opened == [url]
        assert url.startswith("https://github.com/EinarOlafsson/spacr/"
                              "issues/new?")


# ---------------------------------------------------------------------------
# The whole path: a Mask run that fails, on a profile left at the defaults
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_desktop_notifications(monkeypatch):
    """A failed run announces itself on the desktop; not from a test."""
    monkeypatch.setattr("spacr.qt.notify.announce_pipeline_finished",
                        lambda *args, **kwargs: None)


@pytest.fixture(autouse=True)
def _nothing_is_being_filed():
    from spacr.qt.screens import app_screen

    app_screen._REPORTS_BEING_FILED.clear()
    yield
    app_screen._REPORTS_BEING_FILED.clear()


def _failing_entry(home):
    """A Mask entry whose error names this computer."""
    def preprocess_generate_masks(settings):
        raise ValueError(
            f"cannot read {home}/plates/plate_01/masks.npz for {USER} on "
            f"{HOST}")

    return preprocess_generate_masks


@pytest.fixture
def mask_screen(qtbot, monkeypatch, tmp_path, machine):
    """A Mask screen on a profile that never chose, whose Run fails.

    Both issue-reporting preferences are REMOVED, so what runs is the
    default. The AI is not asked: routing errors through it is switched
    off, so the report cannot wait on a provider.
    """
    from spacr import run_journal
    from spacr.qt import preferences
    from spacr.qt.ai import settings as ai_settings
    from spacr.qt.ai.issue_preview import IssuePreviewDialog
    from spacr.qt.screens import app_screen

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: runs)
    preferences._settings().remove(preferences._KEY_ISSUE_PROMPT)
    ai_settings._settings().remove(ai_settings._KEY_AUTO_ISSUE)
    ai_settings.set_route_errors_through_ai(False)
    monkeypatch.setattr(app_screen, "resolve_pipeline_entry",
                        lambda key: _failing_entry(machine))
    previews, opened = [], []
    monkeypatch.setattr(IssuePreviewDialog, "exec",
                        lambda dialog: previews.append(dialog) or 0)
    monkeypatch.setattr(issue_report, "open_issue_in_browser",
                        lambda url: opened.append(url) or True)

    screen = app_screen.AppScreen("mask")
    qtbot.addWidget(screen)
    screen.previews, screen.opened = previews, opened
    yield screen
    screen._console.shutdown()


def _run_and_wait_for(qtbot, screen, needle: str) -> str:
    """Press Run and wait until the console says ``needle``."""
    screen._on_run(override={"src": "/Volumes/drive/Projection",
                             "test_mode": True, "hash_inputs": False})
    qtbot.waitUntil(lambda: needle in screen._console.as_text(),
                    timeout=30000)
    qtbot.waitUntil(lambda: not screen._jobs.is_busy(), timeout=30000)
    return screen._console.as_text()


class TestAFailedRunFilesItself:

    def test_the_report_is_filed_with_no_preview(
            self, qtbot, monkeypatch, signed_in, mask_screen, machine):
        github = FakeGitHub()
        monkeypatch.setattr(github_auth, "_HTTP_OPEN", github)

        text = _run_and_wait_for(qtbot, mask_screen, "[issue] Filed on")

        failed = text.index("✗ Failed")
        filing = text.index("[issue] Filing a redacted report")
        filed = text.index(f"[issue] Filed on GitHub: {FILED_URL}")
        assert failed < filing < filed
        assert text[failed:filing].count("\n") == 1, (
            "the line belongs directly under the failure")
        assert "Nothing was sent to GitHub" not in text
        assert mask_screen.previews == [] and mask_screen.opened == []

        [created] = github.posted("/repos/EinarOlafsson/spacr/issues")
        posted = created["title"] + "\n" + created["body"]
        assert created["title"].startswith("[auto ")
        assert "[mask] ValueError" in created["title"]
        for leak in (str(machine), USER, HOST, "plate_01"):
            assert leak not in posted, leak
        assert "<USER>" in posted and "<HOST>" in posted

    def test_the_same_crash_is_not_filed_twice(
            self, qtbot, monkeypatch, signed_in, mask_screen):
        github = FakeGitHub()
        monkeypatch.setattr(github_auth, "_HTTP_OPEN", github)
        _run_and_wait_for(qtbot, mask_screen, "[issue] Filed on")
        requests_after_the_first = len(github.requests)

        text = _run_and_wait_for(qtbot, mask_screen,
                                 "was reported from this computer before")

        assert len(github.requests) == requests_after_the_first
        assert f"not filed again: {FILED_URL}" in text
        from spacr.qt.ai import settings as ai_settings
        fingerprint = issue_report.fingerprint_of(
            mask_screen._last_error_text)
        assert ai_settings.auto_filed_url(fingerprint) == FILED_URL

    def test_an_open_issue_gets_this_occurrence(
            self, qtbot, monkeypatch, signed_in, mask_screen):
        github = FakeGitHub(open_issue={"number": 88, "html_url": OPEN_URL})
        monkeypatch.setattr(github_auth, "_HTTP_OPEN", github)

        text = _run_and_wait_for(qtbot, mask_screen, "already has an open")

        assert f"This occurrence was added to it: {OPEN_URL}" in text
        assert github.posted("/repos/EinarOlafsson/spacr/issues") == []
        assert len(github.posted("/issues/88/comments")) == 1

    def test_signed_out_it_says_how_to_sign_in(
            self, qtbot, monkeypatch, mask_screen):
        github = FakeGitHub()
        monkeypatch.setattr(github_auth, "_HTTP_OPEN", github)
        monkeypatch.setattr(github_auth, "resolve_token", lambda: ("", None))

        text = _run_and_wait_for(qtbot, mask_screen, "not signed in")

        assert "`gh auth login`" in text
        assert github.requests == [] and mask_screen.opened == []
        assert mask_screen._btn_file_issue.isEnabled()

        monkeypatch.setattr(github_auth, "resolve_token",
                            lambda: (TOKEN, "env"))
        text = _run_and_wait_for(qtbot, mask_screen, "[issue] Filed on")

        assert "reported from this computer before" not in text, (
            "an unsent report was remembered, so it would never be sent")
        assert len(github.posted("/repos/EinarOlafsson/spacr/issues")) == 1

    def test_a_failure_is_said_and_can_be_retried(
            self, qtbot, monkeypatch, signed_in, mask_screen):
        monkeypatch.setattr(github_auth, "_HTTP_OPEN",
                            FakeGitHub(create_fails=True))

        text = _run_and_wait_for(qtbot, mask_screen, "[issue] Not filed")

        assert "GitHub API error 500" in text
        assert TOKEN not in text
        assert "Press File as issue to try again" in text

    def test_ask_files_nothing(self, qtbot, monkeypatch, signed_in,
                               mask_screen):
        from spacr.qt import preferences

        github = FakeGitHub()
        monkeypatch.setattr(github_auth, "_HTTP_OPEN", github)
        preferences.set_issue_prompt_mode("ask")

        text = _run_and_wait_for(qtbot, mask_screen,
                                 "Nothing was sent to GitHub")

        assert github.requests == []
        assert "[issue] Filing" not in text

    def test_reporting_switched_off_files_nothing(
            self, qtbot, monkeypatch, signed_in, mask_screen):
        from spacr.qt.ai import settings as ai_settings

        github = FakeGitHub()
        monkeypatch.setattr(github_auth, "_HTTP_OPEN", github)
        ai_settings.set_auto_file_issues(False)

        text = _run_and_wait_for(qtbot, mask_screen, "✗ Failed")

        assert github.requests == []
        assert "[issue]" not in text

    def test_two_failures_before_github_answers_file_once(
            self, qtbot, monkeypatch, signed_in, mask_screen):
        """Two screens, or two quick runs, fail on one crash before the
        first report has come back."""
        github = FakeGitHub()
        monkeypatch.setattr(github_auth, "_HTTP_OPEN", github)
        mask_screen._last_error_text = _traceback("/home/" + USER)

        mask_screen._file_the_report_automatically()
        mask_screen._file_the_report_automatically()
        qtbot.waitUntil(
            lambda: "[issue] Filed on" in mask_screen._console.as_text(),
            timeout=30000)
        qtbot.waitUntil(lambda: not mask_screen._jobs.is_busy(),
                        timeout=30000)

        assert len(github.posted("/repos/EinarOlafsson/spacr/issues")) == 1
        assert "being reported already" in mask_screen._console.as_text()


# ---------------------------------------------------------------------------
# An AI error is never the analysis (432)
# ---------------------------------------------------------------------------


AUTH_LINE = ("Failed to authenticate: OAuth session expired and could not "
             "be refreshed")


class TestTheAIsErrorIsNotItsAnalysis:

    def _filed_body(self, qtbot, screen, github) -> str:
        screen._file_the_report_automatically()
        qtbot.waitUntil(lambda: bool(github.posted(
            "/repos/EinarOlafsson/spacr/issues")), timeout=30000)
        qtbot.waitUntil(lambda: not screen._jobs.is_busy(), timeout=30000)
        return github.posted("/repos/EinarOlafsson/spacr/issues")[0]["body"]

    def _asked(self, screen, tb, monkeypatch):
        console = screen._console
        monkeypatch.setattr(console, "_current_provider", lambda: object())
        monkeypatch.setattr(console, "_start_stream", lambda **kw: None)
        screen._last_error_text = tb
        console.open_error_flow(tb, active_app="mask", show_raw=False)

    def test_a_provider_that_failed_leaves_no_analysis(
            self, qtbot, monkeypatch, signed_in, mask_screen):
        github = FakeGitHub()
        monkeypatch.setattr(github_auth, "_HTTP_OPEN", github)
        tb = _traceback("/home/" + USER)
        self._asked(mask_screen, tb, monkeypatch)
        mask_screen._console._on_stream_finished(False, AUTH_LINE)

        body = self._filed_body(qtbot, mask_screen, github)

        assert AUTH_LINE not in body
        assert "spaCR AI's analysis" not in body

    def test_an_answer_to_this_error_is_attached(
            self, qtbot, monkeypatch, signed_in, mask_screen):
        """The control: the same path does attach a real answer."""
        github = FakeGitHub()
        monkeypatch.setattr(github_auth, "_HTTP_OPEN", github)
        tb = _traceback("/home/" + USER)
        self._asked(mask_screen, tb, monkeypatch)
        mask_screen._console._on_stream_finished(
            True, "The file holds pickled objects.")

        body = self._filed_body(qtbot, mask_screen, github)

        assert "spaCR AI's analysis" in body
        assert "The file holds pickled objects." in body


# ---------------------------------------------------------------------------
# The agreement says so
# ---------------------------------------------------------------------------


class TestTheAgreementSaysSo:

    @pytest.fixture
    def clause(self):
        from spacr.qt import terms

        return {c.split(" ", 1)[0]: c for c in terms.TERMS}

    def test_it_states_automatic_public_filing_by_default(self, clause):
        automatic = clause["5.6"]
        for said in ("AUTOMATIC ERROR REPORTS", "\u201calways\u201d",
                     "WHICH IS THE DEFAULT", "without showing it to You first",
                     "YOU AGREE TO THE AUTOMATIC FILING"):
            assert said in automatic, said
        published = clause["5.5"]
        for said in ("PUBLISHED", "github.com/EinarOlafsson/spacr",
                     "traceback", "settings", "login and host names"):
            assert said in published, said

    def test_it_no_longer_says_nothing_is_sent(self, clause):
        from spacr.qt import terms

        text = "\n".join(terms.TERMS)
        assert "does not collect or transmit Diagnostic Data" not in text
        assert "Diagnostic Data is not published" not in text
        assert "Section 5.6" in clause["5.1"]

    def test_the_way_to_change_it_is_where_it_says(self, clause):
        """The clause names a menu item, a page, a question and a switch.
        Each has to exist under that name, or the clause is a promise the
        screen does not keep."""
        from pathlib import Path

        import spacr
        from spacr.qt import setup_screen
        from spacr.qt.widgets.setup_slides import SLIDES

        automatic = clause["5.6"]
        root = Path(spacr.__file__).parent / "qt"
        assert "Set spaCR up again\u2026" in automatic
        assert '"Set spaCR up again…"' in (root / "app.py").read_text(
            encoding="utf-8")
        assert "When something breaks" in automatic
        assert "When something breaks" in [t for t, _b, _k in SLIDES]
        caption = {q[0]: q[1] for q in setup_screen.questions()}[
            "issue_prompt"]
        assert f"\u201c{caption}\u201d" in automatic
        assert "Report errors as GitHub issues" in automatic
        assert "Report errors as GitHub issues" in (
            root / "widgets" / "ai_chat_panel.py").read_text(encoding="utf-8")

    def test_a_profile_that_accepted_the_old_terms_is_asked_again(self):
        from spacr.qt import terms

        terms.record_agreement("4.1")

        assert terms.TERMS_VERSION == "4.2"
        assert terms.needs_agreement() is True


def test_the_route_errors_tooltip_states_the_real_default():
    """It said "Default False" while the getter defaults to True. The
    report switch's own tooltip states its new default too."""
    import re

    from spacr.qt.ai import settings as ai_settings
    from spacr.settings import tooltips

    for key, key_name in (("route_errors_through_ai", "_KEY_ROUTE_ERRORS"),
                          ("auto_file_issues", "_KEY_AUTO_ISSUE"),
                          ("console_aware", "_KEY_CONSOLE_AWARE")):
        ai_settings._settings().remove(getattr(ai_settings, key_name))
        stated = re.search(r"Default (True|False)\.\s*$", tooltips[key])
        assert stated, key
        actual = getattr(ai_settings, f"get_{key}")()
        assert stated.group(1) == str(actual), key
