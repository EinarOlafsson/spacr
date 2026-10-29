"""Consented errors use the anonymous relay; explicit GitHub sends are reviewed.

The automatic route needs no GitHub sign-in and preserves bounded retry,
per-occurrence delivery and relay issue deduplication. Local raw diagnostics
stay visible while outgoing payloads are scrubbed. Explicit File as issue
uses its real preview and Send action; absent credentials or an API failure
can hand the approved report to the browser instead of publishing via API.

Retain the saved reporting choices, terms, planted-identity redaction,
analysis pairing and bounded worker lifetime contracts. All relay and GitHub
transports are fakes, and HOME is a temporary folder.
"""
from __future__ import annotations

import io
import json
import time
import urllib.error

import pytest

pytest.importorskip("PySide6")

from spacr.qt.ai import github_auth, issue_report  # noqa: E402
from PySide6.QtWidgets import QDialog as _QDialog  # noqa: E402

_REAL_DIALOG_EXEC = _QDialog.exec

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

    def test_first_run_setup_preserves_automatic_reporting_without_a_question(
            self, fresh, qtbot):
        """Setup no longer asks about reports or writes over their defaults."""
        from spacr.qt import setup_screen
        from spacr.qt.widgets.setup_slides import SLIDES, SetupSlides

        preferences, ai_settings = fresh
        slides = SetupSlides()
        qtbot.addWidget(slides)
        assert "issue_prompt" not in slides._editors
        answers = slides.answers()
        assert "issue_prompt" not in answers
        assert all("issue_prompt" not in keys for _title, _text, keys in SLIDES)
        assert setup_screen.apply(answers) == []
        assert preferences.get_issue_prompt_mode() == "always"
        assert ai_settings.get_auto_file_issues() is True

    def test_a_profile_that_only_ever_dismissed_setup_gets_the_default(
            self, fresh):
        """THE DECISION HAS TO REACH THE INSTALLED BASE.

        `SetupSlides.accept()` and `reject()` both apply their answers, and
        `issue_prompt` has been one of them since 6c57da8d6 (1.5.0.5), so
        every profile that so much as opened first-run setup has 'ask'
        written into it by the default of the day. Read as a choice, the new
        default would have reached new profiles only -- and the reporter of
        issue #117, on 1.5.0.8, would still file nothing after upgrading.
        """
        preferences, _ai = fresh
        store = preferences._settings()
        store.setValue(preferences._KEY_ISSUE_PROMPT, "ask")
        store.remove(preferences._KEY_ISSUE_PROMPT_CHOSEN)

        assert preferences.get_issue_prompt_mode() == "always"

    def test_an_ask_this_build_wrote_is_an_answer_and_is_kept(self, fresh):
        """The other half of it: once a build that marks the choice has
        written 'ask', 'ask' is what the user meant."""
        preferences, _ai = fresh

        preferences.set_issue_prompt_mode("ask")

        assert preferences._settings().contains(
            preferences._KEY_ISSUE_PROMPT_CHOSEN)
        assert preferences.get_issue_prompt_mode() == "ask"

    @pytest.mark.parametrize("stored", ["never", "always"])
    def test_an_older_profile_that_said_never_or_always_is_left_alone(
            self, fresh, stored):
        """Neither was ever written on a user's behalf: the old default was
        'ask', so both took an answer."""
        preferences, _ai = fresh
        store = preferences._settings()
        store.setValue(preferences._KEY_ISSUE_PROMPT, stored)
        store.remove(preferences._KEY_ISSUE_PROMPT_CHOSEN)

        assert preferences.get_issue_prompt_mode() == stored

    def test_an_upgraded_profile_keeps_the_default_through_setup(self, fresh, qtbot):
        """Setup preserves the migrated default without a retired report question."""
        from spacr.qt import setup_screen
        from spacr.qt.widgets.setup_slides import SetupSlides

        preferences, _ai = fresh
        store = preferences._settings()
        store.setValue(preferences._KEY_ISSUE_PROMPT, "ask")
        store.remove(preferences._KEY_ISSUE_PROMPT_CHOSEN)

        slides = SetupSlides()
        qtbot.addWidget(slides)

        assert "issue_prompt" not in slides._editors
        assert "issue_prompt" not in slides.answers()
        assert setup_screen.apply(slides.answers()) == []
        assert preferences.get_issue_prompt_mode() == "always"
        assert _ai.get_auto_file_issues() is True

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

    def test_a_machine_named_after_a_word_does_not_rewrite_the_report(
            self, monkeypatch):
        """A host called `gpu` turned `use_gpu = True` in the settings block
        into `use_<HOST> = True`, because `_` is a word boundary here. The
        name says nothing about who the user is, so it stays."""
        monkeypatch.setenv("HOME", "/home/bio")
        monkeypatch.setattr("getpass.getuser", lambda: "bio")
        monkeypatch.setattr("socket.gethostname", lambda: "gpu")
        monkeypatch.setattr("platform.node", lambda: "gpu")

        assert issue_report._identity_words() == []
        assert issue_report.redact_identity(
            "use_gpu = True, bio-replicate 2") == (
            "use_gpu = True, bio-replicate 2")

    def test_a_dotted_address_is_not_a_name(self, monkeypatch):
        """`192.168.0.10` contributes the label `192`, which would then be
        replaced wherever it stood -- including inside a traceback line the
        fingerprint is taken over."""
        monkeypatch.setenv("HOME", "/home/anna")
        monkeypatch.setattr("socket.gethostname", lambda: "192.168.0.10")
        monkeypatch.setattr("platform.node", lambda: "192.168.0.10")
        monkeypatch.setattr("getpass.getuser", lambda: "anna")

        names = dict(issue_report._identity_words())

        assert "192" not in names
        assert issue_report.redact_identity(
            "line 192, in run") == "line 192, in run"

    def test_a_workstation_named_after_its_user_is_one_name(self,
                                                            monkeypatch):
        """Login and host the same string: the host entry won, so the login
        name was reported as <HOST>."""
        monkeypatch.setenv("HOME", "/home/annali")
        monkeypatch.setattr("getpass.getuser", lambda: "annali")
        monkeypatch.setattr("socket.gethostname", lambda: "annali")
        monkeypatch.setattr("platform.node", lambda: "annali")

        assert issue_report._identity_words() == [
            ("annali", issue_report.USER_PLACEHOLDER)]

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

    @pytest.mark.parametrize("line,leaks", [
        ("src = '/Volumes/Lab Drive/Patient 042/plate A.tif'",
         ("Lab Drive", "Patient 042", "plate A.tif")),
        (r"src = 'C:\Users\anna\OneDrive - Karolinska Institutet\Screens'",
         ("Karolinska", "Screens", "anna")),
        ("PermissionError: [Errno 13] Permission denied: "
         "'/mnt/lab share/Smith lab/raw'",
         ("lab share", "Smith lab")),
        ("opening C:\\Program Files\\spaCR bin\\plate 1\\img.tif failed",
         ("Program Files", "spaCR bin", "plate 1")),
        (r"could not reach \\LAB-NAS\screens\plate 1\img.tif",
         ("LAB-NAS", "screens", "img.tif")),
        ("[auto ab12cd] [mask] FileNotFoundError: [Errno 2] No such file "
         "or directory: '/Volumes/Lab Drive/Patient 042/plate A.t",
         ("Lab Drive", "Patient 042")),
    ])
    def test_a_path_with_spaces_is_replaced_whole(self, line, leaks):
        """A PATH CAN CONTAIN A SPACE, and stopping at the first whitespace
        published the rest of it: '/Volumes/Lab Drive/Patient 042' came out
        as '<PATH> Drive/Patient 042'. macOS volumes, Windows 'OneDrive -
        <institution>' and any lab share are the normal case, not the odd
        one, and with 'always' nobody sees the report before it is public.
        The last case is the title, which build_report cuts to 80
        characters -- so the opening quote can be there with its closing one
        cut off."""
        stripped = issue_report.strip_report_paths(
            issue_report.sanitize_path(line))

        for leak in leaks:
            assert leak not in stripped, f"{leak!r} survived in {stripped!r}"
        assert "<PATH>" in stripped

    def test_what_is_around_a_path_is_still_readable(self):
        """Over-redaction has a cost too: a report whose sentences have been
        eaten is a report nobody can act on. Only the path goes."""
        cases = {
            "opening /mnt/data/x.tif failed": "opening <PATH> failed",
            "</summary> and </details> stay": "</summary> and </details> stay",
            "3 / 4 of the wells and/or the plate":
                "3 / 4 of the wells and/or the plate",
            "raise ValueError('mask failed')":
                "raise ValueError('mask failed')",
            "the well is 'plate 1' here": "the well is 'plate 1' here",
        }
        for line, expected in cases.items():
            assert issue_report.strip_report_paths(line) == expected, line

    def test_a_spaced_path_in_the_settings_is_not_published(self, machine):
        """The settings block is `key = repr(value)`, so every path in it is
        quoted -- and a quoted path is taken whole."""
        report = issue_report.build_report(
            _traceback(machine), active_app="mask",
            settings={"src": "/Volumes/Lab Drive/Patient 042",
                      "dst": r"C:\Users\anna\OneDrive - KI\Screens\plate 1"})

        text = "\n".join(issue_report.public_report(report).values())

        for leak in ("Lab Drive", "Patient 042", "OneDrive", "Screens",
                     "plate 1", "anna"):
            assert leak not in text, leak

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
    default. The terms of use are accepted, as the setup slide records them.
    The AI is not asked: routing errors through it is switched off, so the
    report cannot wait on a provider.
    """
    from spacr import run_journal
    from spacr.qt import preferences, terms
    from spacr.qt.ai import settings as ai_settings
    from spacr.qt.ai.issue_preview import IssuePreviewDialog
    from spacr.qt.screens import app_screen

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: runs)
    preferences._settings().remove(preferences._KEY_ISSUE_PROMPT)
    ai_settings._settings().remove(ai_settings._KEY_AUTO_ISSUE)
    terms.record_agreement(terms.TERMS_VERSION)
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


@pytest.fixture
def automatic_mask_screen(qtbot, monkeypatch, mask_screen, manual_error_batches):
    """Observe consented automatic relay work and retire only its real workers."""
    from PySide6.QtCore import QCoreApplication
    from spacr import crashreport
    from spacr.qt import preferences
    from spacr.qt.ai import providers
    from spacr.qt.ai.issue_report import _sync_automatic_failure_policy
    from tests.test_700_anonymous_error_reports import FakeRelay

    relay = FakeRelay((200, {"issue_number": 501, "issue_url": FILED_URL,
                             "deduplicated": False}))
    monkeypatch.setattr(crashreport, "_open_url", relay)
    monkeypatch.setattr(crashreport, "_ERROR_DISPATCH_SEEN", {})
    monkeypatch.setattr(crashreport, "_ERROR_OCCURRENCE_POLICIES", {})
    preferences.set_preferred_provider(providers.HOSTED_PROVIDER_NAME)
    crashreport.set_error_report_mode("send_all")
    _sync_automatic_failure_policy()
    policy, explain, send = crashreport._automatic_error_policy(crashreport.KIND_FAULT)
    assert not explain and send
    mask_screen.show()
    qtbot.waitExposed(mask_screen)
    workers, releases = [], []
    dispatch = crashreport._dispatch_error_now

    def observe(*args, **kwargs):
        worker = dispatch(*args, **kwargs)
        if worker is not None:
            workers.append(worker)
        return worker

    def finish():
        manual_error_batches()
        qtbot.waitUntil(lambda: all(not worker.is_alive() for worker in workers),
                        timeout=30000)
        QCoreApplication.processEvents()

    monkeypatch.setattr(crashreport, "_dispatch_error_now", observe)
    mask_screen.relay = relay
    mask_screen.report_releases = releases
    mask_screen.advance_reports = manual_error_batches
    mask_screen.finish_reports = finish
    yield mask_screen
    for release in releases:
        release.set()
    finish()


def _run_automatic_report(qtbot, screen, **override):
    """Await the real failed pipeline before delivering its consented batch."""
    from spacr.qt.bridge import thread_has_stopped

    settings = {"src": "/Volumes/drive/Projection", "test_mode": True,
                "hash_inputs": False}
    settings.update(override)
    screen._on_run(override=settings)
    qtbot.waitUntil(lambda: "✗ Failed" in screen._console.as_text(), timeout=30000)
    qtbot.waitUntil(lambda: thread_has_stopped(screen._thread), timeout=30000)
    screen.finish_reports()
    return screen._console.as_text()


def _send_approved_issue(qtbot, screen, monkeypatch):
    """Press the real preview's Send action and consume the owned report job."""
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtWidgets import QDialog, QDialogButtonBox
    from spacr.qt.ai.issue_preview import IssuePreviewDialog

    previews = []

    def approve(dialog):
        previews.append(dialog)
        failures = []

        def send_once():
            try:
                buttons = dialog.findChild(QDialogButtonBox)
                send = next(button for button in buttons.buttons()
                            if buttons.buttonRole(button) == QDialogButtonBox.AcceptRole)
                qtbot.mouseClick(send, Qt.LeftButton)
                assert dialog.result() == QDialog.Accepted
            except BaseException as error:
                failures.append(error)
                dialog.reject()

        timer = QTimer(dialog)
        timer.setSingleShot(True)
        timer.timeout.connect(send_once)
        timer.start(0)
        result = _REAL_DIALOG_EXEC(dialog)
        assert failures == []
        assert result == QDialog.Accepted
        return result

    monkeypatch.setattr(IssuePreviewDialog, "exec", approve)
    screen.show()
    qtbot.waitExposed(screen)
    assert screen._btn_file_issue.isVisible() and screen._btn_file_issue.isEnabled()
    qtbot.mouseClick(screen._btn_file_issue, Qt.LeftButton)
    qtbot.waitUntil(lambda: not screen._jobs.is_busy(), timeout=30000)
    qtbot.waitUntil(lambda: screen._jobs.active_jobs() == 0, timeout=30000)
    assert len(previews) == 1
    return screen._console.as_text()


class TestAFailedRunFilesItself:

    @pytest.mark.usefixtures("console_info_enabled")
    def test_the_report_is_filed_with_no_preview(
            self, qtbot, monkeypatch, automatic_mask_screen, machine):
        screen = automatic_mask_screen
        monkeypatch.setattr(github_auth, "resolve_token",
                            lambda: pytest.fail("automatic relay must not query GitHub credentials"))
        text = _run_automatic_report(qtbot, screen)
        assert "Reported anonymously as issue #501" in text
        assert "✗ Failed" in text
        assert text.index("ValueError: cannot read") < text.index("Reported anonymously")
        assert screen.previews == [] and screen.opened == []
        [created] = screen.relay.calls_to("/report")
        posted = json.dumps(created["body"])
        for leak in (str(machine), USER, HOST, "plate_01", TOKEN):
            assert leak not in posted, leak
        assert "ValueError" in posted
        assert screen.relay.calls_to("/diagnose") == []

    @pytest.mark.usefixtures("console_info_enabled")
    def test_the_same_crash_is_not_filed_twice(
            self, qtbot, automatic_mask_screen):
        screen = automatic_mask_screen
        screen.relay.routes["/report"] = [
            (200, {"issue_number": 501, "issue_url": FILED_URL, "deduplicated": False}),
            (200, {"issue_number": 501, "issue_url": FILED_URL, "deduplicated": True})]
        _run_automatic_report(qtbot, screen)
        _run_automatic_report(qtbot, screen)
        reports = screen.relay.calls_to("/report")
        assert len(reports) == 2
        from spacr import crashreport
        assert crashreport.explanation_signature(reports[0]["body"]) == \
            crashreport.explanation_signature(reports[1]["body"])
        assert "already has issue #501" in screen._console.as_text()
        assert FILED_URL in screen._console.as_text()
        assert screen.previews == [] and screen.opened == []

    @pytest.mark.usefixtures("console_info_enabled")
    def test_an_open_issue_gets_this_occurrence(self, qtbot, automatic_mask_screen):
        screen = automatic_mask_screen
        screen.relay.routes["/report"] = [(200, {
            "issue_number": 88, "issue_url": OPEN_URL, "deduplicated": True})]
        text = _run_automatic_report(qtbot, screen)
        assert "already has issue #88, which now counts it again" in text
        assert OPEN_URL in text
        assert len(screen.relay.calls_to("/report")) == 1
        assert screen.previews == [] and screen.opened == []

    @pytest.mark.usefixtures("console_info_enabled")
    def test_signed_out_approved_report_uses_browser_fallback_without_api_request(
            self, qtbot, monkeypatch, mask_screen, machine, manual_error_batches):
        github = FakeGitHub()
        monkeypatch.setattr(github_auth, "_HTTP_OPEN", github)
        monkeypatch.setattr(github_auth, "resolve_token", lambda: ("", None))
        mask_screen._on_pipeline_error(_traceback(machine))
        text = _send_approved_issue(qtbot, mask_screen, monkeypatch)
        assert github.requests == []
        assert len(mask_screen.opened) == 1
        assert mask_screen.opened[0].startswith(
            "https://github.com/EinarOlafsson/spacr/issues/new?")
        assert "report handoff completed" in text
        assert mask_screen._btn_file_issue.isEnabled()
        monkeypatch.setattr(github_auth, "resolve_token", lambda: (TOKEN, "env"))
        text = _send_approved_issue(qtbot, mask_screen, monkeypatch)
        assert len(github.posted("/repos/EinarOlafsson/spacr/issues")) == 1
        assert FILED_URL in text

    @pytest.mark.usefixtures("console_info_enabled")
    def test_an_approved_api_failure_falls_back_and_can_be_retried(
            self, qtbot, monkeypatch, signed_in, mask_screen, machine, manual_error_batches):
        github = FakeGitHub(create_fails=True)
        monkeypatch.setattr(github_auth, "_HTTP_OPEN", github)
        mask_screen._on_pipeline_error(_traceback(machine))
        first = _send_approved_issue(qtbot, mask_screen, monkeypatch)
        assert len(github.posted("/repos/EinarOlafsson/spacr/issues")) == 1
        assert len(mask_screen.opened) == 1
        assert mask_screen.opened[0].startswith(
            "https://github.com/EinarOlafsson/spacr/issues/new?")
        assert TOKEN in first
        notices = first[first.index("[issue] sending the approved report"):]
        assert TOKEN not in notices
        assert TOKEN not in json.dumps(github.posted("/repos/EinarOlafsson/spacr/issues"))
        assert TOKEN not in mask_screen.opened[0]
        github.create_fails = False
        second = _send_approved_issue(qtbot, mask_screen, monkeypatch)
        assert len(github.posted("/repos/EinarOlafsson/spacr/issues")) == 2
        assert FILED_URL in second

    @pytest.mark.usefixtures("console_info_enabled")
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

    @pytest.mark.parametrize("accepted", ["", "4.1"])
    @pytest.mark.usefixtures("console_info_enabled")
    def test_nothing_is_filed_before_the_terms_are_accepted(
            self, qtbot, monkeypatch, signed_in, mask_screen, accepted):
        """Consent is Section 5.6. A profile launched with --no-setup, on a
        server, or that closed the terms slide has not given it, and neither
        has one that accepted 4.1, which said nothing was sent."""
        from spacr.qt import terms

        github = FakeGitHub()
        monkeypatch.setattr(github_auth, "_HTTP_OPEN", github)
        terms._settings().setValue(terms._KEY_VERSION, accepted)

        text = _run_and_wait_for(qtbot, mask_screen,
                                 "Automatic filing starts once you accept")

        assert github.requests == []
        assert "[issue] Filing" not in text
        failed = text.index("✗ Failed")
        said = text.index("[issue] Nothing was sent to GitHub.")
        assert text[failed:said].count("\n") == 1
        assert "Set spaCR up again" in text[said:]
        assert mask_screen._btn_file_issue.isEnabled()

    @pytest.mark.usefixtures("console_info_enabled")
    def test_reporting_switched_off_files_nothing(
            self, qtbot, monkeypatch, signed_in, mask_screen):
        from spacr.qt.ai import settings as ai_settings

        github = FakeGitHub()
        monkeypatch.setattr(github_auth, "_HTTP_OPEN", github)
        ai_settings.set_auto_file_issues(False)

        text = _run_and_wait_for(qtbot, mask_screen, "✗ Failed")

        assert github.requests == []
        assert "[issue]" not in text

    @pytest.mark.usefixtures("console_info_enabled")
    def test_a_run_on_a_folder_with_spaces_publishes_none_of_it(
            self, qtbot, monkeypatch, automatic_mask_screen):
        from spacr.qt.screens import app_screen

        def entry(_key):
            def preprocess_generate_masks(settings):
                raise FileNotFoundError(
                    2, "No such file or directory", "/Volumes/Lab Drive/Patient 042/plate A.tif")
            return preprocess_generate_masks

        screen = automatic_mask_screen
        monkeypatch.setattr(app_screen, "resolve_pipeline_entry", entry)
        _run_automatic_report(qtbot, screen, src="/Volumes/Lab Drive/Patient 042")
        [created] = screen.relay.calls_to("/report")
        posted = json.dumps(created["body"])
        for leak in ("Lab Drive", "Patient 042", "plate A.tif", "Volumes"):
            assert leak not in posted, f"{leak!r} was published: {posted!r}"
        assert created["body"]["exception_type"] == "FileNotFoundError"

    @pytest.mark.usefixtures("console_info_enabled")
    def test_a_report_that_never_came_back_does_not_block_the_next_one(
            self, qtbot, automatic_mask_screen):
        from spacr import crashreport

        screen = automatic_mask_screen
        screen.relay.routes["/report"] = [OSError("offline"), (200, {
            "issue_number": 501, "issue_url": FILED_URL, "deduplicated": False})]
        first = _run_automatic_report(qtbot, screen)
        assert "will retry later" in first
        [queued] = crashreport.queued_summaries()
        second = _run_automatic_report(qtbot, screen)
        assert "Reported anonymously as issue #501" in second
        assert crashreport.queued_summaries() == []
        reports = screen.relay.calls_to("/report")
        assert len(reports) == 3
        signature = crashreport.explanation_signature(queued)
        assert all(crashreport.explanation_signature(report["body"]) == signature
                   for report in reports)

    def test_two_deliveries_before_the_relay_answers_file_once(
            self, qtbot, monkeypatch, automatic_mask_screen, machine):
        import threading

        screen = automatic_mask_screen
        entered, release = threading.Event(), threading.Event()
        screen.report_releases.append(release)
        transport = screen.relay

        def gated(request, timeout):
            if request.full_url.endswith("/report"):
                entered.set()
                assert release.wait(5)
            return transport(request, timeout)

        from spacr import crashreport
        monkeypatch.setattr(crashreport, "_open_url", gated)
        tb = _traceback(machine)
        try:
            screen._on_pipeline_error(tb)
            occurrence = screen._last_error_occurrence
            screen.advance_reports()
            assert entered.wait(5)
            screen._on_pipeline_error(tb, event_id=occurrence)
            screen.advance_reports()
            assert screen.relay.calls_to("/report") == []
        finally:
            release.set()
        screen.finish_reports()
        assert len(screen.relay.calls_to("/report")) == 1


# ---------------------------------------------------------------------------
# An AI error is never the analysis (432)
# ---------------------------------------------------------------------------


from tests.qt.test_the_console_keeps_the_ai_error_analysis import manual_error_batches


AUTH_LINE = ("Failed to authenticate: OAuth session expired and could not "
             "be refreshed")


class TestTheAIsErrorIsNotItsAnalysis:

    def _filed_body(self, qtbot, screen, github, monkeypatch) -> str:
        from PySide6.QtCore import QTimer, Qt
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        from spacr.qt.ai.issue_preview import IssuePreviewDialog

        def approve(dialog):
            buttons = dialog.findChild(QDialogButtonBox)
            send = next(button for button in buttons.buttons()
                        if buttons.buttonRole(button) == QDialogButtonBox.AcceptRole)
            failures = []

            def send_once():
                try:
                    qtbot.mouseClick(send, Qt.LeftButton)
                    assert dialog.result() == QDialog.Accepted
                except BaseException as error:
                    failures.append(error)
                    dialog.reject()

            timer = QTimer(dialog)
            timer.setSingleShot(True)
            timer.timeout.connect(send_once)
            timer.start(0)
            result = _REAL_DIALOG_EXEC(dialog)
            assert failures == []
            return result

        monkeypatch.setattr(IssuePreviewDialog, "exec", approve)
        screen.show()
        qtbot.waitExposed(screen)
        assert screen._btn_file_issue.isVisible() and screen._btn_file_issue.isEnabled()
        qtbot.mouseClick(screen._btn_file_issue, Qt.LeftButton)
        qtbot.waitUntil(lambda: bool(github.posted(
            "/repos/EinarOlafsson/spacr/issues")), timeout=30000)
        qtbot.waitUntil(lambda: not screen._jobs.is_busy(), timeout=30000)
        qtbot.waitUntil(lambda: screen._jobs.active_jobs() == 0, timeout=30000)
        return github.posted("/repos/EinarOlafsson/spacr/issues")[0]["body"]

    def _asked(self, screen, tb, monkeypatch, manual_error_batches):
        from types import SimpleNamespace

        console = screen._console
        provider = SimpleNamespace(name="claude")
        monkeypatch.setattr(console, "_current_provider", lambda: provider)
        monkeypatch.setattr(console, "_start_stream", lambda **kw: None)
        screen._on_pipeline_error(tb)
        console.open_error_flow(tb, active_app="mask", show_raw=False)
        manual_error_batches()

    def test_a_provider_that_failed_leaves_no_analysis(
            self, qtbot, monkeypatch, signed_in, mask_screen, manual_error_batches):
        github = FakeGitHub()
        monkeypatch.setattr(github_auth, "_HTTP_OPEN", github)
        tb = _traceback("/home/" + USER)
        self._asked(mask_screen, tb, monkeypatch, manual_error_batches)
        mask_screen._console._on_stream_finished(False, AUTH_LINE)

        body = self._filed_body(qtbot, mask_screen, github, monkeypatch)

        assert AUTH_LINE not in body
        assert "spaCR AI's analysis" not in body

    def test_an_answer_to_this_error_is_attached(
            self, qtbot, monkeypatch, signed_in, mask_screen, manual_error_batches):
        """The control: the same path does attach a real answer."""
        github = FakeGitHub()
        monkeypatch.setattr(github_auth, "_HTTP_OPEN", github)
        tb = _traceback("/home/" + USER)
        self._asked(mask_screen, tb, monkeypatch, manual_error_batches)
        mask_screen._console._on_stream_finished(
            True, "The file holds pickled objects.")

        body = self._filed_body(qtbot, mask_screen, github, monkeypatch)

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
        for said in ("ANONYMISED ERROR REPORTS ARE ON BY DEFAULT",
                     "PUBLIC issue", "EinarOlafsson/spacr",
                     "You need no GitHub account", "NEVER sent",
                     "file and folder paths", "login and host names",
                     "tokens and other credentials", "logs, settings values",
                     "cannot be reliably unpublished"):
            assert said in automatic, said
        assert "YOU AGREE TO THE SENDING" in clause["5.7"]
        published = clause["5.5"]
        for said in ("PUBLISHED", "github.com/EinarOlafsson/spacr",
                     "credential-shaped values", "database file names",
                     "login and host names", "replaced with placeholders",
                     "The log is never published"):
            assert said in published, said

    def test_it_no_longer_says_nothing_is_sent(self, clause):
        from spacr.qt import terms

        text = "\n".join(terms.TERMS)
        assert "does not collect or transmit Diagnostic Data" not in text
        assert "Diagnostic Data is not published" not in text
        assert "Section 5.6" in clause["5.1"]

    def test_the_way_to_change_it_is_where_it_says(self, clause):
        """The agreement's opt-out labels exist in Preferences and the notice."""
        from pathlib import Path

        import spacr
        from spacr import crashreport

        choices = clause["5.7"]
        root = Path(spacr.__file__).parent / "qt"
        preferences = (root / "preferences.py").read_text(encoding="utf-8")
        notice = (root / "install_consent.py").read_text(encoding="utf-8")
        assert "Preferences → AI" in choices
        for caption in ("AI Error report", "What is sent…"):
            assert f"“{caption}”" in choices
            assert f'tr("{caption}")' in preferences
        assert "“Send anonymised error reports”" in choices
        assert 'tr("Send anonymised error reports")' in notice
        modes = dict(crashreport.ERROR_REPORT_MODES)
        for key in ("off", "summarize"):
            assert f"“{modes[key]}”" in choices
            for kind in (crashreport.KIND_FAULT, crashreport.KIND_USER_ERROR):
                assert crashreport.mode_sends(kind, key) is False

    def test_a_profile_that_accepted_the_old_terms_is_asked_again(self):
        from spacr.qt import terms

        terms.record_agreement("4.3")

        assert terms.TERMS_VERSION == "4.5"
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


# ---------------------------------------------------------------------------
# The automatic report waits for spaCR AI's answer
# ---------------------------------------------------------------------------

ANALYSIS = "The masks file is missing because the plate folder moved."


def _enable_hosted_report_analysis(screen):
    """Choose consented hosted explanations on the actual automatic route."""
    from spacr import crashreport
    from spacr.qt.ai import settings
    from spacr.qt.ai.issue_report import _sync_automatic_failure_policy

    settings.set_route_errors_through_ai(True)
    crashreport.set_error_report_mode("all")
    _sync_automatic_failure_policy()
    _, explain, send = crashreport._automatic_error_policy(crashreport.KIND_FAULT)
    assert explain and send
    screen.relay.routes["/diagnose"] = [(200, {"advice": ANALYSIS})]


class TestTheReportWaitsForTheAI:
    """ITEM 432: filing did not wait for the AI, so an automatic report
    rarely carried the analysis it exists to carry."""

    def test_the_report_carries_the_answer_that_arrived_after_the_run(
            self, qtbot, monkeypatch, automatic_mask_screen, machine):
        import threading
        from spacr import crashreport

        screen = automatic_mask_screen
        _enable_hosted_report_analysis(screen)
        entered, release = threading.Event(), threading.Event()
        screen.report_releases.append(release)
        transport = screen.relay

        def gated(request, timeout):
            if request.full_url.endswith("/diagnose"):
                entered.set()
                assert release.wait(5)
            return transport(request, timeout)

        monkeypatch.setattr(crashreport, "_open_url", gated)
        try:
            screen._on_pipeline_error(_traceback(machine))
            screen.advance_reports()
            assert entered.wait(5)
            qtbot.waitUntil(lambda: "asking spaCR AI" in screen._console.as_text(), timeout=5000)
            assert screen.relay.calls_to("/report") == [], "filed before the AI had answered"
        finally:
            release.set()
        screen.finish_reports()
        [created] = screen.relay.calls_to("/report")
        assert ANALYSIS in created["body"]["ai_summary"]
        assert ANALYSIS in screen._console.as_text()

    @pytest.mark.usefixtures("console_info_enabled")
    def test_an_ai_that_never_answers_does_not_hold_the_report(
            self, qtbot, automatic_mask_screen, machine):
        from spacr import crashreport

        screen = automatic_mask_screen
        _enable_hosted_report_analysis(screen)
        screen.relay.routes["/diagnose"] = [TimeoutError("diagnosis deadline")]
        screen._on_pipeline_error(_traceback(machine))
        screen.finish_reports()
        [diagnosis] = screen.relay.calls_to("/diagnose")
        assert diagnosis["timeout"] == crashreport.AI_SUMMARY_TIMEOUT_S
        [report] = screen.relay.calls_to("/report")
        assert "ai_summary" not in report["body"]
        assert "diagnosis deadline" not in json.dumps(report["body"])
        assert "without AI summary" in screen._console.as_text()
        screen._console.ai_stream_finished.emit()
        assert len(screen.relay.calls_to("/report")) == 1, "one failure files once"

    def test_a_second_delivery_during_the_wait_is_not_filed_twice(
            self, qtbot, monkeypatch, automatic_mask_screen, machine):
        import threading
        from spacr import crashreport

        screen = automatic_mask_screen
        _enable_hosted_report_analysis(screen)
        entered, release = threading.Event(), threading.Event()
        screen.report_releases.append(release)
        transport = screen.relay

        def gated(request, timeout):
            if request.full_url.endswith("/diagnose"):
                entered.set()
                assert release.wait(5)
            return transport(request, timeout)

        monkeypatch.setattr(crashreport, "_open_url", gated)
        tb = _traceback(machine)
        try:
            screen._on_pipeline_error(tb)
            occurrence = screen._last_error_occurrence
            screen.advance_reports()
            assert entered.wait(5)
            screen._on_pipeline_error(tb, event_id=occurrence)
            screen.advance_reports()
            assert screen.relay.calls_to("/report") == []
        finally:
            release.set()
        screen.finish_reports()
        assert len(screen.relay.calls_to("/diagnose")) == 1
        [report] = screen.relay.calls_to("/report")
        assert report["body"]["ai_summary"] == ANALYSIS

    def test_no_wait_when_the_ai_is_not_answering_this_error(
            self, qtbot, automatic_mask_screen, machine):
        from spacr.qt import preferences
        from spacr.qt.ai.issue_report import _sync_automatic_failure_policy

        screen = automatic_mask_screen
        preferences.set_preferred_provider("none")
        _sync_automatic_failure_policy()
        screen._on_pipeline_error(_traceback(machine))
        screen.finish_reports()
        assert screen.relay.calls_to("/diagnose") == []
        [report] = screen.relay.calls_to("/report")
        assert report["body"]["summarize"] is False
        assert "ai_summary" not in report["body"]
        assert "Waiting up to" not in screen._console.as_text()
        assert "asking spaCR AI" not in screen._console.as_text()


@pytest.mark.parametrize("line,leaks", [
    (r"could not reach \\LAB-NAS\screens\plate 1", ("plate 1", " 1")),
    ("mask /Volumes/Lab Drive failed", ("Drive",)),
    ("wrote /mnt/lab share/Patient 042 and stopped", ("042",)),
    ("wrote /mnt/data/plate a.tif then stopped", ("a.tif",)),
])
def test_the_last_component_of_an_unquoted_path_keeps_no_tail(line, leaks):
    """ITEM 432: an unquoted path's last component stopped at its first
    space, so ``\\\\LAB-NAS\\screens\\plate 1`` published `` 1``. The words
    that follow are taken when they look like a name -- a number, a
    capital, a file name -- and the prose after the path stays readable."""
    stripped = issue_report.strip_report_paths(line)

    for leak in leaks:
        assert leak not in stripped, f"{leak!r} survived in {stripped!r}"
    assert "<PATH>" in stripped
    for word in ("failed", "and stopped", "then stopped"):
        if word in line:
            assert word in stripped, f"{word!r} was eaten: {stripped!r}"
