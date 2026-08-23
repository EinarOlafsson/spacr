"""Both sign-ins on the setup screen said things that were not happening.

GITHUB. The button ran ``gh auth login --web`` as a detached process and
set the status to "waiting for GitHub in your browser…". `gh` opens a
browser only AFTER printing a one-time code and waiting for Enter on a
terminal -- and a GUI child process has no terminal, so it sat on that
prompt forever while the dialog described a browser that never opened.
spaCR now reads the code out of `gh`'s output, shows it, opens GitHub's
device page itself, and answers the prompt so `gh` goes on to poll.

THE AI PROVIDERS. Clicking one only ticked it. The colour of a mark meant
"the CLI is on PATH", so a provider that was installed and signed out
looked exactly like one ready to answer. The colour is the SIGN-IN state
now, and choosing a provider starts its login -- in a terminal, because
these logins are conversations and starting them without one is the same
mistake the GitHub button was making.

GitHub gets a mark of its own for the same reason, monochrome because its
logo is: grey signed out, GitHub's black signed in.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QRectF

from spacr.qt.widgets.provider_marks import BRAND, MARKS, ProviderMark, mark_for
from spacr.qt.widgets.setup_slides import PROVIDERS, SetupSlides


class _FakeProcess:
    """Stands in for `gh`, which cannot be run in a test."""

    def __init__(self, text: str):
        self._text = text.encode()
        self.written = b""

    def readAllStandardOutput(self):
        text, self._text = self._text, b""
        return text

    def write(self, data):
        self.written += data


@pytest.fixture
def slides(qapp):
    dialog = SetupSlides()
    try:
        yield dialog
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------

GH_OUTPUT = ("! First copy your one-time code: A1B2-C3D4\n"
             "Press Enter to open github.com in your browser...")


def test_the_browser_is_opened_by_spacr(slides):
    opened = []
    slides._open_in_the_browser = lambda url: (opened.append(url), True)[1]

    slides._read_github_output(_FakeProcess(GH_OUTPUT))

    assert opened == [SetupSlides.GITHUB_DEVICE_PAGE]


def test_the_one_time_code_is_shown(slides):
    slides._open_in_the_browser = lambda url: True

    code = slides._read_github_output(_FakeProcess(GH_OUTPUT))

    assert code == "A1B2-C3D4"
    assert "A1B2-C3D4" in slides._gh_status.text()


def test_the_prompt_gh_is_waiting_on_is_answered(slides):
    """Without this `gh` never polls and the sign-in cannot complete."""
    slides._open_in_the_browser = lambda url: True
    process = _FakeProcess(GH_OUTPUT)

    slides._read_github_output(process)

    assert process.written == b"\n"


def test_a_second_line_does_not_open_a_second_tab(slides):
    """`gh` reprints the code as it polls."""
    opened = []
    slides._open_in_the_browser = lambda url: (opened.append(url), True)[1]

    slides._read_github_output(_FakeProcess(GH_OUTPUT))
    slides._read_github_output(_FakeProcess("one-time code: A1B2-C3D4"))

    assert len(opened) == 1


def test_output_with_no_code_changes_nothing(slides):
    opened = []
    slides._open_in_the_browser = lambda url: (opened.append(url), True)[1]

    assert slides._read_github_output(_FakeProcess("Logging in...")) == ""
    assert opened == []


@pytest.mark.parametrize("source,coloured", [("gh", True), ("env", True),
                                             (None, False)])
def test_the_github_mark_follows_the_sign_in(slides, monkeypatch, source,
                                             coloured):
    from spacr.qt.ai import github_auth

    monkeypatch.setattr(github_auth, "auth_source", lambda: source)
    slides._refresh_github()

    assert slides._gh_mark.available is coloured


def test_github_has_a_mark_to_draw():
    assert "github" in MARKS
    path = mark_for("github", QRectF(0, 0, 60, 60))
    assert path is not None and not path.isEmpty()
    # Monochrome, because the logo is: "go from grey to black and white".
    assert BRAND["github"].lower() in ("#181717", "#000000")


# ---------------------------------------------------------------------------
# the AI providers
# ---------------------------------------------------------------------------

class _StubProvider:
    label = "GPT"
    cli_name = "codex"
    login_command = "codex login"

    def __init__(self, installed=True, logged_in=False):
        self._installed = installed
        self._logged_in = logged_in

    def is_installed(self):
        return self._installed

    def is_logged_in(self):
        return self._logged_in

    def is_configured(self):
        return self._installed and self._logged_in


@pytest.fixture
def stub_gpt(monkeypatch):
    from spacr.qt.ai import providers

    holder = {"provider": _StubProvider()}
    real = providers.get_provider
    monkeypatch.setattr(
        providers, "get_provider",
        lambda name: holder["provider"] if name == "gpt" else real(name))
    return holder


def test_choosing_a_signed_out_provider_starts_its_login(slides, stub_gpt):
    launched = []
    slides._run_in_a_terminal = lambda cmd: (launched.append(cmd), True)[1]

    note = slides._start_provider_login("gpt")

    assert launched == ["codex login"]
    assert "GPT" in note


def test_a_signed_in_provider_is_not_asked_to_log_in_again(slides, stub_gpt):
    stub_gpt["provider"] = _StubProvider(installed=True, logged_in=True)
    launched = []
    slides._run_in_a_terminal = lambda cmd: (launched.append(cmd), True)[1]

    assert slides._start_provider_login("gpt") == ""
    assert launched == []


def test_an_uninstalled_provider_says_so_instead(slides, stub_gpt):
    stub_gpt["provider"] = _StubProvider(installed=False)
    launched = []
    slides._run_in_a_terminal = lambda cmd: (launched.append(cmd), True)[1]

    note = slides._start_provider_login("gpt")

    assert "codex" in note and "not installed" in note
    assert launched == []


def test_with_no_terminal_the_command_is_named(slides, stub_gpt):
    """Naming it beats starting it where its prompts cannot be answered."""
    slides._run_in_a_terminal = lambda cmd: False

    note = slides._start_provider_login("gpt")

    assert "codex login" in note


def test_the_mark_colour_means_signed_in_not_installed(slides, stub_gpt):
    """An installed-but-signed-out provider must not look ready."""
    assert SetupSlides._provider_is_signed_in("gpt", "codex") is False

    stub_gpt["provider"] = _StubProvider(installed=True, logged_in=True)
    assert SetupSlides._provider_is_signed_in("gpt", "codex") is True


def test_clicking_a_mark_selects_it_and_recolours_the_strip(slides, stub_gpt):
    slides._run_in_a_terminal = lambda cmd: True
    holder = slides._provider_buttons("")

    slides._choose_provider(holder, "gpt")

    assert holder._chosen == "gpt"
    assert holder._buttons["gpt"].is_chosen()
    assert holder._buttons["gpt"].available is False   # started, not finished
    assert holder._note.text()


def test_every_provider_has_a_mark():
    for code, _label, _command in PROVIDERS:
        assert code in MARKS, f"{code} has no mark to draw"
