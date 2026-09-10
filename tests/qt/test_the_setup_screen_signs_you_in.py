"""Both sign-ins on the setup screen said things that were not happening.

GITHUB. The button ran ``gh auth login --web`` as a detached process and
set the status to "waiting for GitHub in your browser…". `gh` opens a
browser only AFTER printing a one-time code and waiting for Enter on a
terminal -- and a GUI child process has no terminal, so it sat on that
prompt forever while the dialog described a browser that never opened.
spaCR now reads the code out of `gh`'s output, shows it, opens GitHub's
device page itself, and answers the prompt so `gh` goes on to poll.

THE AI PROVIDERS. Clicking one only ticked it, and the colour of a mark
meant "the CLI is on PATH" -- so installed-and-signed-out looked exactly
like ready. There are three states now (ready / signed out / not
installed), the brand FILL is what ready looks like, an unready mark shows
its brand colour only on hover, and choosing one opens a dialog that says
what it needs and offers to do it: open the install page, or run the sign-in
in a terminal. A note under the row was not enough -- "there is no popup no
prompt for installing or any guidance".

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
    """GPT, in whatever state a test wants it.

    Patched on `SetupSlides._provider_object` rather than on the registry:
    the screen calls OpenAI's provider `gpt` and `ai.providers` files it
    under `codex`, and the resolver that bridges the two is what every code
    path here goes through.
    """
    holder = {"provider": _StubProvider()}
    monkeypatch.setattr(
        SetupSlides, "_provider_object",
        staticmethod(lambda code, command="": holder["provider"]
                     if str(code) in ("gpt", "codex") else None))
    return holder


@pytest.fixture
def answer_the_dialog(monkeypatch):
    """Press the dialog's first accepting button, whatever it is."""
    from PySide6.QtWidgets import QMessageBox

    def press(self):
        for button in self.buttons():
            if self.buttonRole(button) == QMessageBox.ButtonRole.AcceptRole:
                self.setProperty("_pressed", button)
                return 0
        return 0

    monkeypatch.setattr(QMessageBox, "exec", press)
    monkeypatch.setattr(QMessageBox, "clickedButton",
                        lambda self: self.property("_pressed"))


def test_choosing_a_signed_out_provider_offers_to_sign_it_in(
        slides, stub_gpt, answer_the_dialog):
    stub_gpt["provider"] = _StubProvider(installed=True, logged_in=False)
    launched = []
    slides._run_in_a_terminal = lambda cmd: (launched.append(cmd), True)[1]

    note = slides._start_provider_login("gpt")

    assert launched == ["codex login"]
    assert "GPT" in note


def test_a_signed_in_provider_is_not_asked_to_log_in_again(
        slides, stub_gpt, answer_the_dialog):
    stub_gpt["provider"] = _StubProvider(installed=True, logged_in=True)
    launched = []
    slides._run_in_a_terminal = lambda cmd: (launched.append(cmd), True)[1]

    assert slides._start_provider_login("gpt") == ""
    assert launched == []


def test_an_uninstalled_provider_is_offered_its_install_page(
        slides, stub_gpt, answer_the_dialog):
    """"Install" is a thing a click can do, on every operating system."""
    stub_gpt["provider"] = _StubProvider(installed=False)
    opened, launched = [], []
    slides._open_in_the_browser = lambda url: (opened.append(url), True)[1]
    slides._run_in_a_terminal = lambda cmd: (launched.append(cmd), True)[1]

    note = slides._start_provider_login("gpt")

    assert opened == [SetupSlides.PROVIDER_PAGES["gpt"]]
    assert launched == [], "an uninstalled CLI cannot be signed in to"
    assert "browser" in note


def test_with_no_terminal_the_command_is_named(slides, stub_gpt,
                                               answer_the_dialog):
    """Naming it beats starting it where its prompts cannot be answered."""
    stub_gpt["provider"] = _StubProvider(installed=True, logged_in=False)
    slides._run_in_a_terminal = lambda cmd: False

    note = slides._start_provider_login("gpt")

    assert "codex login" in note


@pytest.mark.parametrize("installed,logged_in,expected", [
    (True, True, "ready"),
    (True, False, "signed out"),
    (False, False, "not installed"),
])
def test_the_three_states_are_told_apart(slides, stub_gpt, installed,
                                         logged_in, expected):
    """They need three different things from the user, so they are three."""
    stub_gpt["provider"] = _StubProvider(installed, logged_in)

    assert SetupSlides.provider_status("gpt", "codex") == expected


def test_only_a_ready_provider_is_filled_with_its_brand_colour(qapp):
    """"GPT and Gemini should only get their color fill when installed"."""
    ready = ProviderMark("gpt", "GPT", True, None, status="ready")
    unready = ProviderMark("gpt", "GPT", False, None, status="not installed")

    assert BRAND["gpt"].lower() in ready._colours()[0].name().lower()
    assert BRAND["gpt"].lower() not in unready._colours()[0].name().lower()

    # And hovering an unready one shows the brand colour behind it.
    unready._hovered = True
    assert BRAND["gpt"].lower() in unready._colours()[1].name().lower()


def test_clicking_a_mark_selects_it_and_recolours_the_strip(
        slides, stub_gpt, answer_the_dialog):
    slides._run_in_a_terminal = lambda cmd: True
    slides._open_in_the_browser = lambda url: True
    holder = slides._provider_buttons("")

    slides._choose_provider(holder, "gpt")

    assert holder._chosen == "gpt"
    assert holder._buttons["gpt"].is_chosen()
    assert holder._buttons["gpt"].available is False   # started, not finished
    assert holder._note.text()


def test_every_provider_has_a_mark():
    for code, _label, _command in PROVIDERS:
        assert code in MARKS, f"{code} has no mark to draw"
