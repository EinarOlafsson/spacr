"""The auto-issue URL, and the preference that governs the prompt.

The reported symptom was "page not found" on triggering a report. GitHub
answers an over-long `issues/new` with a page that reads exactly that, and
the URL was over-long even after the code had "truncated" it.
"""

import urllib.parse

import pytest

from spacr.qt.ai.issue_report import ISSUE_LABEL, MAX_URL_LEN, REPO, issue_url


TRACEBACK = 'Traceback (most recent call last):\n  File "x.py", line 1\n' * 400


class TestTheUrlFits:

    @pytest.mark.parametrize("name, body", [
        ("newline-dense", TRACEBACK),
        ("plain ascii", "x" * 30_000),
        ("unicode", "café — μm\n" * 2000),
        ("short", "it broke"),
        ("empty", ""),
    ])
    def test_every_body_produces_a_url_within_the_limit(self, name, body):
        """The regression. Truncation measured RAW characters against a
        budget denominated in URL characters, and quoting expands -- a
        newline is `%0A`, three characters. The newline-dense case came out
        at 11,924 against a 7,500 limit AFTER truncating.
        """
        assert len(issue_url("spaCR crashed", body)) <= MAX_URL_LEN

    def test_a_long_report_says_it_was_truncated(self):
        url = issue_url("spaCR crashed", TRACEBACK)
        assert "truncated" in urllib.parse.unquote(url)

    def test_a_short_report_is_not_truncated(self):
        url = issue_url("spaCR crashed", "it broke")
        assert "truncated" not in urllib.parse.unquote(url)

    def test_the_traceback_survives_truncation(self):
        """Trimming keeps the head, which is where the exception is."""
        assert "Traceback (most recent call last)" in urllib.parse.unquote(
            issue_url("spaCR crashed", TRACEBACK))


class TestTheUrlShape:
    """A substring check would not catch a wrong owner. These name parts."""

    def test_it_points_at_the_real_repository(self):
        parsed = urllib.parse.urlparse(issue_url("t", "b"))
        assert parsed.netloc == "github.com"
        assert parsed.path == f"/{REPO}/issues/new"

    def test_the_label_is_one_the_repository_actually_has(self):
        """GitHub 404s on `labels=` naming a label that does not exist --
        it does NOT create it lazily, whatever the old docstring said."""
        query = urllib.parse.parse_qs(
            urllib.parse.urlparse(issue_url("t", "b")).query)
        assert query["labels"] == [ISSUE_LABEL]

    def test_the_title_and_body_survive_the_round_trip(self):
        query = urllib.parse.parse_qs(
            urllib.parse.urlparse(issue_url("a title", "a body")).query)
        assert query["title"] == ["a title"]
        assert query["body"] == ["a body"]


class TestThePromptPreference:

    def test_the_default_is_to_ask(self):
        """On a store that has never been written, the answer is `ask`.

        Read from a fresh QSettings rather than the ambient one: the qt
        conftest sets `always` for every test so the reporter cannot open a
        modal, and that fixture would otherwise be what this measures.
        """
        from spacr.qt import preferences as P
        assert P.get_issue_prompt_mode.__doc__  # the default is documented
        settings = P._settings()
        saved = settings.value("ai/issue_prompt", None)
        settings.remove("ai/issue_prompt")
        try:
            assert P.get_issue_prompt_mode() == P.ISSUE_PROMPT_ASK
        finally:
            if saved is not None:
                settings.setValue("ai/issue_prompt", saved)

    def test_all_three_modes_round_trip(self):
        from spacr.qt import preferences as P
        for mode in P.ISSUE_PROMPT_MODES:
            P.set_issue_prompt_mode(mode)
            assert P.get_issue_prompt_mode() == mode
        P.set_issue_prompt_mode(P.ISSUE_PROMPT_ASK)

    def test_there_are_three_modes_not_two(self):
        """'ask' and 'never' leave out the user who wants it filed
        silently."""
        from spacr.qt.preferences import ISSUE_PROMPT_MODES
        assert set(ISSUE_PROMPT_MODES) == {"ask", "never", "always"}

    def test_an_unknown_mode_is_refused_rather_than_stored(self):
        from spacr.qt import preferences as P
        with pytest.raises(ValueError, match="not one of"):
            P.set_issue_prompt_mode("sometimes")

    def test_an_unrecognised_stored_value_reads_back_as_ask(self):
        """A preference file from a newer build must not silence the
        reporter on an older one."""
        from spacr.qt import preferences as P
        P._settings().setValue("ai/issue_prompt", "a-future-mode")
        assert P.get_issue_prompt_mode() == "ask"
        P.set_issue_prompt_mode(P.ISSUE_PROMPT_ASK)


class TestThePromptPathHonoursThePreference:
    """`never` must actually prevent it, not merely change a dialog."""

    @pytest.fixture
    def screen(self, qt_theme_applied, qtbot):
        from spacr.qt.screens.app_screen import AppScreen
        widget = AppScreen("mask")
        qtbot.addWidget(widget)
        widget._last_error_text = "Traceback: boom"
        return widget

    def _no_dialogs(self, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        opened = []

        def _boom(*a, **k):
            opened.append(1)
            raise AssertionError("a modal was opened")

        for name in ("exec", "exec_"):
            monkeypatch.setattr(QMessageBox, name, _boom, raising=False)
        return opened

    def test_never_files_nothing_and_opens_nothing(self, screen, monkeypatch):
        from spacr.qt import preferences as P
        filed = []
        monkeypatch.setattr("spacr.qt.ai.issue_report.file_issue",
                            lambda *a, **k: filed.append(1) or "url")
        self._no_dialogs(monkeypatch)
        P.set_issue_prompt_mode(P.ISSUE_PROMPT_NEVER)
        try:
            screen._on_file_issue()          # must not raise, must not ask
        finally:
            P.set_issue_prompt_mode(P.ISSUE_PROMPT_ASK)
        assert filed == []

    def test_never_says_why_rather_than_doing_nothing_silently(
            self, screen, monkeypatch):
        """Silence would read as the button being broken."""
        from spacr.qt import preferences as P
        said = []
        monkeypatch.setattr(screen._console, "append_notice",
                            lambda text: said.append(text))
        self._no_dialogs(monkeypatch)
        P.set_issue_prompt_mode(P.ISSUE_PROMPT_NEVER)
        try:
            screen._on_file_issue()
        finally:
            P.set_issue_prompt_mode(P.ISSUE_PROMPT_ASK)
        assert any("never" in t for t in said)

    def test_with_no_error_captured_nothing_happens_at_all(self, screen,
                                                           monkeypatch):
        self._no_dialogs(monkeypatch)
        screen._last_error_text = ""
        screen._on_file_issue()
