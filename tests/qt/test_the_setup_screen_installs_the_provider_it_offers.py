"""The setup screen installs the AI provider, and the GitHub CLI, it offers.

Item 420. Asked for on 2026-09-16, in the maintainer's words: "in the startup
spacr when the use clicks an AI provider there should be an aditional button,
install which automatically downloads and installs the chosen ai provider and
asks the user for the needed information and sets up the AI provider . so
instead of giving the user a curl link it uses the curl link to downloade e.g.
claude and then installs it." And the same day: "and same for the github cli,
add that to and a button that links to generating a github account".

These tests PRESS THE BUTTONS: a click on the mark, Install in the prompt,
Cancel, Try again, Copy, the account button. The installer process is faked
at :data:`spacr.qt.ai.cli_install._spawn` -- an autouse guard fails any test
that would start a real one -- and the install runs on the panel's real
worker thread, which is what keeps the screen responsive while it runs.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import types

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from spacr.qt.ai import cli_install, providers
from spacr.qt.widgets import cli_setup_panel as panel_module
from spacr.qt.widgets.cli_setup_panel import (CONFIRM, FAILED, IDLE,
                                              INSTALLING, READY,
                                              SIGN_IN_PENDING, SIGNING_IN,
                                              CliSetupPanel)
from spacr.qt.widgets.provider_marks import ProviderMark
from spacr.qt.widgets.setup_slides import SLIDES, SetupSlides

WAIT_MS = 5000

REAL_RUN_QUIETLY = providers._run_quietly
REAL_LIKELY_FOLDERS = cli_install.likely_folders


@pytest.fixture(autouse=True)
def _nothing_is_really_installed(monkeypatch):
    """No test here may start a real installer, npm query or status check."""
    def refuse(*args, **kwargs):
        raise AssertionError(f"a real process was about to start: {args!r}")

    monkeypatch.setattr(cli_install, "_spawn", refuse)
    monkeypatch.setattr(cli_install, "_query", refuse)
    monkeypatch.setattr(cli_install, "_signal_group", refuse)
    monkeypatch.setattr(providers, "_run_quietly", refuse)
    monkeypatch.setattr(cli_install, "likely_folders",
                        lambda platform=None: [])
    monkeypatch.setattr(cli_install, "WATCH_INTERVAL_S", 0.01)
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))


@pytest.fixture
def present(monkeypatch):
    """The programs this machine has, as far as the code can tell."""
    found = {}

    def which(name, mode=os.F_OK | os.X_OK, path=None):
        if path is not None:
            return None
        return found.get(name)

    import shutil

    monkeypatch.setattr(shutil, "which", which)
    return found


@pytest.fixture
def signed_out_of_github(monkeypatch):
    monkeypatch.setattr("spacr.qt.ai.github_auth.auth_source", lambda: None)


@pytest.fixture
def slides(qapp, qtbot, present, signed_out_of_github):
    dialog = SetupSlides()
    qtbot.addWidget(dialog)
    dialog.show()
    try:
        yield dialog
    finally:
        dialog._stop_the_installs()
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


@pytest.fixture
def press(monkeypatch):
    """Answer the next prompt by pressing the button captioned ``text``."""
    seen = {}

    def choose(text):
        def exec_(box):
            seen["text"] = box.text()
            seen["informative"] = box.informativeText()
            seen["buttons"] = [b.text() for b in box.buttons()]
            seen["pressed"] = next(
                (b for b in box.buttons() if b.text() == text), None)
            return 0

        monkeypatch.setattr(QMessageBox, "exec", exec_)
        monkeypatch.setattr(QMessageBox, "clickedButton",
                            lambda box: seen.get("pressed"))
        return seen

    return choose


@pytest.fixture
def opened(slides):
    pages = []
    slides._open_in_the_browser = lambda url: pages.append(url) or True
    return pages


@pytest.fixture
def clipboard(monkeypatch):
    copied = []
    monkeypatch.setattr(QApplication, "clipboard", staticmethod(
        lambda: types.SimpleNamespace(setText=copied.append)))
    return copied


class FakeStdout:
    """The installer's output, one chunk per read, then end of file."""

    def __init__(self, chunks, gate):
        self._chunks = list(chunks)
        self._gate = gate

    def read1(self, _size):
        if self._chunks:
            return self._chunks.pop(0)
        if self._gate is not None:
            self._gate.wait(10)
        return b""

    def close(self):
        pass


class FakeInstaller:
    """An installer that prints, optionally waits to be released, and exits.

    ``on_exit`` runs when it exits 0, which is where a test puts the CLI on
    the pretend ``PATH``.
    """

    pid = 424242

    def __init__(self, chunks=(), status=0, gate=None, on_exit=None):
        self.stdout = FakeStdout(chunks, gate)
        self._status = status
        self._gate = gate
        self._on_exit = on_exit
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self._gate is not None and not self._gate.wait(
                timeout if timeout is not None else 10):
            raise subprocess.TimeoutExpired("fake", timeout)
        if self.returncode is None:
            self.returncode = self._status
            if self._status == 0 and self._on_exit is not None:
                self._on_exit()
        return self.returncode

    def end(self, status):
        self._status = status
        if self._gate is not None:
            self._gate.set()


@pytest.fixture
def spawned(monkeypatch):
    """Record every installer started, and on which thread."""
    calls = []
    queue = []

    def spawn(command, **kwargs):
        calls.append({"command": command, "kwargs": kwargs,
                      "gui_thread": threading.current_thread()
                      is threading.main_thread()})
        return queue.pop(0)

    monkeypatch.setattr(cli_install, "_spawn", spawn)
    return types.SimpleNamespace(calls=calls, queue=queue)


def _go_to(slides, title):
    """Show the slide called ``title``, as Next would."""
    slides._show_slide(next(i for i, (name, _b, _k) in enumerate(SLIDES)
                            if name == title))


def _assistant(slides):
    """Go to the assistant slide and hand back its row of marks."""
    _go_to(slides, "The assistant")
    return slides._editors["ai_provider"]


def _issues(slides):
    """Go to the slide with the GitHub row."""
    _go_to(slides, "When something breaks")


def _click(qtbot, widget):
    qtbot.mouseClick(widget, Qt.LeftButton)


# ------------------------------------------------------------ an AI provider


def test_install_runs_claudes_own_command_off_the_gui_thread_then_signs_in(
        slides, qtbot, present, press, spawned, monkeypatch, tmp_path):
    """The whole request, pressed through: mark, Install, sign-in, READY."""
    present.update(curl="/usr/bin/curl", bash="/usr/bin/bash")
    claude = providers.get_provider("claude")
    spawned.queue.append(FakeInstaller(
        [b"Downloading Claude Code...\n", b"\x1b[1mInstalled\x1b[0m\n"],
        on_exit=lambda: present.update(claude=str(tmp_path / "claude"))))
    launched = []
    slides._run_in_a_terminal = lambda command: launched.append(command) or True
    monkeypatch.setattr(claude, "check_signed_in", lambda: True)
    holder = _assistant(slides)
    panel = slides._ai_setup
    panel._timer.setInterval(20)
    assert holder._buttons["claude"].status == ProviderMark.NOT_INSTALLED

    seen = press("Install")
    _click(qtbot, holder._buttons["claude"])

    assert claude.install_hint in seen["informative"], \
        "the command is shown before it runs"
    assert "Install" in seen["buttons"]
    qtbot.waitUntil(lambda: panel.state == READY, timeout=WAIT_MS)
    call = spawned.calls[0]
    assert call["command"] == ["/usr/bin/bash", "-o", "pipefail", "-c",
                               claude.install_hint]
    assert call["gui_thread"] is False, "the installer ran on the GUI thread"
    assert launched == ["claude auth login"]
    assert "Claude" in panel.message.text()
    assert holder._buttons["claude"].status == ProviderMark.READY
    assert holder._buttons["claude"].available
    assert str(tmp_path) in os.environ["PATH"].split(os.pathsep)[0]


def test_the_installer_output_streams_and_cancel_stops_it(
        slides, qtbot, present, press, spawned, monkeypatch):
    present.update(curl="/usr/bin/curl", bash="/usr/bin/bash")
    gate = threading.Event()
    installer = FakeInstaller([b"Downloading 40%\n"], gate=gate)
    spawned.queue.append(installer)
    sent = []

    def signal_group(pid, sig):
        sent.append((pid, sig))
        installer.end(-sig)

    monkeypatch.setattr(cli_install, "_signal_group", signal_group)
    holder = _assistant(slides)
    panel = slides._ai_setup

    press("Install")
    _click(qtbot, holder._buttons["claude"])
    qtbot.waitUntil(lambda: panel.latest.text() == "Downloading 40%",
                    timeout=WAIT_MS)
    assert panel.state == INSTALLING
    assert panel.progress.isVisible() and panel.cancel_button.isVisible()
    assert panel.command.isVisible()
    assert panel.command.text() == providers.get_provider(
        "claude").install_hint
    assert slides._next.isEnabled(), "the screen is still usable"

    _click(qtbot, panel.cancel_button)
    qtbot.waitUntil(lambda: panel.state == FAILED, timeout=WAIT_MS)
    assert sent == [(installer.pid, signal.SIGTERM)]
    assert "cancelled" in panel.message.text().lower()
    assert panel.install_button.isVisible()
    assert panel.install_button.text() == "Try again"
    assert panel.command.isVisible()


@pytest.mark.parametrize("output,status,words", [
    (b"curl: (6) Could not resolve host: claude.ai\n", 6,
     "internet connection"),
    (b"curl: (22) The requested URL returned error: 403\n", 22, "refused"),
    (b"bash: line 1: /usr/lib: Permission denied\n", 1, "not allowed"),
    (b"Unsupported architecture\n", 1, "exit status 1: Unsupported"),
])
def test_every_failure_says_what_to_do_and_leaves_the_screen_usable(
        slides, qtbot, present, press, spawned, opened, clipboard, output,
        status, words):
    present.update(curl="/usr/bin/curl", bash="/usr/bin/bash")
    spawned.queue.append(FakeInstaller([output], status=status))
    holder = _assistant(slides)
    panel = slides._ai_setup

    press("Install")
    _click(qtbot, holder._buttons["claude"])
    qtbot.waitUntil(lambda: panel.state == FAILED, timeout=WAIT_MS)
    assert words in panel.message.text()
    assert holder._buttons["claude"].status == ProviderMark.NOT_INSTALLED

    _click(qtbot, panel.copy_button)
    assert clipboard == [providers.get_provider("claude").install_hint]
    _click(qtbot, panel.page_button)
    assert opened == [SetupSlides.PROVIDER_PAGES["claude"]]

    spawned.queue.append(FakeInstaller([output], status=status))
    _click(qtbot, panel.install_button)
    qtbot.waitUntil(lambda: len(spawned.calls) == 2 and panel.state == FAILED,
                    timeout=WAIT_MS)


def test_a_permission_failure_from_npm_names_the_npm_fix(
        slides, qtbot, present, press, spawned):
    """What this machine would say: its npm is the system's, in /usr."""
    present.update(npm="/usr/bin/npm")
    spawned.queue.append(FakeInstaller(
        [b"npm ERR! code EACCES\n",
         b"npm ERR! Error: EACCES: permission denied, mkdir "
         b"'/usr/lib/node_modules/@openai'\n"], status=243))
    holder = _assistant(slides)
    panel = slides._ai_setup

    seen = press("Install")
    _click(qtbot, holder._buttons["gpt"])
    assert "npm install -g @openai/codex" in seen["informative"]
    qtbot.waitUntil(lambda: panel.state == FAILED, timeout=WAIT_MS)
    assert spawned.calls[0]["command"] == [
        "/usr/bin/npm", "install", "-g", "@openai/codex"]
    assert "npm config set prefix ~/.local" in panel.message.text()


def test_with_no_npm_and_no_homebrew_the_prompt_says_so_and_runs_nothing(
        slides, qtbot, present, press, spawned, opened):
    holder = _assistant(slides)
    seen = press("Open the page")
    _click(qtbot, holder._buttons["gpt"])
    assert "Install" not in seen["buttons"]
    assert "npm (Node.js) or Homebrew" in seen["informative"]
    assert opened == [SetupSlides.PROVIDER_PAGES["gpt"]]
    assert spawned.calls == []


def test_a_second_install_waits_for_the_first(slides, qtbot, present, press,
                                              spawned, monkeypatch):
    present.update(curl="/usr/bin/curl", bash="/usr/bin/bash",
                   npm="/usr/bin/npm")
    gate = threading.Event()
    installer = FakeInstaller([b"working\n"], gate=gate)
    spawned.queue.append(installer)
    monkeypatch.setattr(cli_install, "_signal_group",
                        lambda pid, sig: installer.end(-sig))
    holder = _assistant(slides)
    press("Install")
    _click(qtbot, holder._buttons["claude"])
    qtbot.waitUntil(lambda: slides._ai_setup.latest.text() == "working",
                    timeout=WAIT_MS)
    assert slides._ai_setup.state == INSTALLING

    note = slides._start_provider_login("gpt")

    assert "still running" in note
    assert len(spawned.calls) == 1
    gate.set()


def _start_a_long_install(slides, qtbot, present, press, spawned,
                          monkeypatch):
    """Press Install on Claude with an installer that runs until stopped.

    :returns: the signals the installer's group was sent, as they arrive.
    """
    present.update(curl="/usr/bin/curl", bash="/usr/bin/bash")
    gate = threading.Event()
    installer = FakeInstaller([b"Downloading\n"], gate=gate)
    spawned.queue.append(installer)
    sent = []
    monkeypatch.setattr(cli_install, "_signal_group",
                        lambda pid, sig: (sent.append(sig),
                                          installer.end(-sig)))
    holder = _assistant(slides)
    press("Install")
    _click(qtbot, holder._buttons["claude"])
    qtbot.waitUntil(lambda: slides._ai_setup.latest.text() == "Downloading",
                    timeout=WAIT_MS)
    return sent


@pytest.mark.parametrize("leave", ["reject", "accept"])
def test_closing_the_screen_mid_install_asks_and_can_keep_it_running(
        slides, qtbot, present, press, spawned, monkeypatch, leave):
    """Review of 420, 2026-09-19: Start spaCR stopped the install silently.

    Pressing Install, then Next to the end and Start spaCR (accept), or
    Escape (reject), used to close the screen and SIGTERM the installer
    with nothing on screen to say so. Now the user is asked, and keeping
    it running is the default.
    """
    sent = _start_a_long_install(slides, qtbot, present, press, spawned,
                                 monkeypatch)
    answered = []
    monkeypatch.setattr("spacr.qt.setup_screen.mark_answered",
                        lambda version: answered.append(version))

    seen = press("Keep installing")
    getattr(slides, leave)()

    assert "Claude" in seen["text"]
    assert "Stop it and close" in seen["buttons"]
    assert slides.isVisible(), "the screen closed over a running install"
    assert sent == [], "the installer was stopped anyway"
    assert slides._ai_setup.state == INSTALLING
    assert answered == [], "the screen was recorded as done"

    press("Stop it and close")
    getattr(slides, leave)()

    assert not slides.isVisible()
    assert sent == [signal.SIGTERM]
    assert slides._ai_setup._runner.active_jobs() == 0
    assert slides._ai_setup.state == FAILED
    assert "cancelled" in slides._ai_setup.message.text().lower()
    assert len(answered) == 1


def test_keeping_the_install_is_the_default_and_what_escape_does(
        slides, qtbot, present, spawned, monkeypatch, press):
    _start_a_long_install(slides, qtbot, present, press, spawned,
                          monkeypatch)
    seen = {}

    def exec_(box):
        seen["default"] = box.defaultButton().text()
        seen["escape"] = box.escapeButton().text()
        seen["title"] = box.windowTitle()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", exec_)
    monkeypatch.setattr(QMessageBox, "clickedButton",
                        lambda box: box.escapeButton())

    assert slides._may_close() is False
    assert seen == {"default": "Keep installing",
                    "escape": "Keep installing",
                    "title": "An install is still running"}


def test_with_nothing_installing_the_screen_closes_without_asking(
        slides, press):
    seen = press("Keep installing")
    slides.reject()
    assert not slides.isVisible()
    assert "buttons" not in seen, "it asked with nothing running"


def test_a_github_install_is_named_when_closing(slides, press):
    slides._refresh_github()
    slides._gh_setup._tool = providers.github_cli()
    slides._gh_setup.state = INSTALLING
    try:
        assert slides._running_installs() == ["GitHub CLI"]
        seen = press("Keep installing")
        assert slides._may_close() is False
        assert "GitHub CLI" in seen["text"]
    finally:
        slides._gh_setup.state = IDLE


def test_signing_in_to_an_installed_provider_is_watched_to_the_end(
        slides, qtbot, press, monkeypatch):
    """"Sign in now" opens the terminal; Done re-reads the marks."""
    stub = types.SimpleNamespace(
        label="GPT", cli_name="codex", login_command="codex login",
        install_hint="npm install -g @openai/codex",
        is_installed=lambda: True, is_logged_in=lambda: False,
        is_configured=lambda: False)
    monkeypatch.setattr(SetupSlides, "_provider_object",
                        staticmethod(lambda code, command="": stub
                                     if code in ("gpt", "codex") else None))
    launched = []
    slides._run_in_a_terminal = lambda command: launched.append(command) or True
    holder = _assistant(slides)
    panel = slides._ai_setup
    refreshed = []
    monkeypatch.setattr(slides, "_refresh_provider_marks",
                        lambda h: refreshed.append(h))

    press("Sign in now")
    _click(qtbot, holder._buttons["gpt"])

    assert launched == ["codex login"]
    assert panel.state == CONFIRM and panel.done_button.isVisible()
    assert panel.command.text() == "codex login"
    refreshed.clear()
    _click(qtbot, panel.done_button)
    assert panel.state == IDLE and not panel.isVisible()
    assert refreshed, "Done did not re-read the marks"


def test_with_no_terminal_the_sign_in_command_stays_to_copy(
        slides, qtbot, press, monkeypatch, clipboard):
    stub = types.SimpleNamespace(
        label="GPT", cli_name="codex", login_command="codex login",
        is_installed=lambda: True, is_logged_in=lambda: False,
        is_configured=lambda: False)
    monkeypatch.setattr(SetupSlides, "_provider_object",
                        staticmethod(lambda code, command="": stub))
    slides._run_in_a_terminal = lambda command: False
    holder = _assistant(slides)
    panel = slides._ai_setup

    press("Sign in now")
    _click(qtbot, holder._buttons["gpt"])

    assert panel.state == SIGN_IN_PENDING
    assert "terminal" in panel.message.text()
    _click(qtbot, panel.copy_button)
    assert clipboard == ["codex login"]


# ------------------------------------------------------------- the GitHub CLI


def test_the_github_cli_installs_and_goes_straight_to_its_sign_in(
        slides, qtbot, present, press, spawned, monkeypatch):
    present.update(conda="/opt/conda/bin/conda")
    spawned.queue.append(FakeInstaller(
        [b"Collecting package metadata: done\n"],
        on_exit=lambda: present.update(gh="/opt/conda/bin/gh")))
    started = []
    slides._sign_in_to_github = lambda: started.append(True) or True
    monkeypatch.setattr(providers, "_run_quietly", lambda argv, timeout: 0)
    _issues(slides)
    slides._refresh_github()
    assert slides._gh_action == "install"
    assert slides._gh_mark.status == ProviderMark.NOT_INSTALLED
    panel = slides._gh_setup
    panel._timer.setInterval(20)

    seen = press("Install")
    _click(qtbot, slides._gh_mark)

    tool = providers.github_cli()
    row = providers.gh_conda_row(sys.prefix, sys.platform)
    assert row.command in seen["informative"]
    qtbot.waitUntil(lambda: panel.state == READY, timeout=WAIT_MS)
    assert spawned.calls[0]["command"] == (
        ["/opt/conda/bin/conda"]
        + cli_install.split_command(row.command, sys.platform)[1:])
    assert started == [True]
    assert slides._gh_action == "login", "the row re-read the installed CLI"
    assert tool.label in panel.message.text()


def test_a_github_sign_in_that_ends_unfinished_says_so(slides, qtbot,
                                                        monkeypatch):
    monkeypatch.setattr(providers, "_run_quietly", lambda argv, timeout: 1)
    _issues(slides)
    panel = slides._gh_setup
    panel.sign_in(providers.github_cli(), lambda: True)
    assert panel.state == SIGNING_IN

    slides._github_sign_in_ended(1)

    assert panel.state == SIGN_IN_PENDING
    assert "Sign in" in panel.message.text()
    assert panel.sign_in_button.isVisible()
    slides._github_sign_in_ended(0)
    assert panel.state == SIGN_IN_PENDING


def test_with_no_way_to_install_gh_the_prompt_offers_the_page(
        slides, qtbot, press, opened, spawned):
    slides._refresh_github()
    seen = press("Open the page")
    assert slides._on_github_mark() is True
    assert "Install" not in seen["buttons"]
    assert "Homebrew or conda" in seen["informative"]
    assert opened == [SetupSlides.GITHUB_CLI_PAGE]
    assert spawned.calls == []


def test_the_github_prompt_copies_or_declines(slides, press, present,
                                              clipboard, monkeypatch):
    present.update(brew="/opt/homebrew/bin/brew")
    slides._refresh_github()
    press("Copy the command")
    assert slides._offer_github_install() is True
    assert clipboard == ["brew install gh"]
    press("Later")
    assert slides._offer_github_install() is False

    def refuse():
        raise RuntimeError("no clipboard")

    monkeypatch.setattr(QApplication, "clipboard", staticmethod(refuse))
    press("Copy the command")
    assert slides._offer_github_install() is True


def test_the_account_button_opens_githubs_sign_up_page(slides, qtbot, opened,
                                                       monkeypatch):
    _issues(slides)
    slides._refresh_github()
    assert slides._gh_signup.isVisible()

    _click(qtbot, slides._gh_signup)

    assert opened == ["https://github.com/signup"]
    monkeypatch.setattr("spacr.qt.ai.github_auth.auth_source", lambda: "gh")
    slides._refresh_github()
    assert not slides._gh_signup.isVisible(), "a signed-in user has one"


# ------------------------------------------------- the screen's bookkeeping


def test_without_its_panels_the_screen_still_signs_in(slides, monkeypatch):
    from shiboken6 import delete

    launched = []
    slides._run_in_a_terminal = lambda command: launched.append(command) or True
    delete(slides._ai_setup)
    assert slides._live_panel("_ai_setup") is None
    assert slides._install_provider(providers.get_provider("claude"),
                                    "claude") is False
    assert slides._sign_in_to_provider(None, "claude auth login") is True
    assert launched == ["claude auth login"]
    del slides._gh_setup
    assert slides._live_panel("_gh_setup") is None
    slides._stop_the_installs()


def test_the_github_mark_does_not_ask_again_while_its_install_runs(
        slides, press):
    slides._refresh_github()
    seen = press("Install")
    slides._gh_setup.state = INSTALLING
    assert slides._on_github_mark() is False
    assert "buttons" not in seen, "a second prompt opened over the first"
    slides._gh_setup.state = IDLE


def test_with_gh_installed_the_mark_signs_in_without_asking(slides, press,
                                                             present):
    present.update(gh="/usr/bin/gh")
    slides._refresh_github()
    started = []
    slides._sign_in_to_github = lambda: started.append(True) or True
    seen = press("Install")
    assert slides._on_github_mark() is True
    assert started == [True] and "buttons" not in seen


def test_without_its_panel_the_github_row_still_signs_in(slides):
    started = []
    slides._sign_in_to_github = lambda: started.append(True) or True
    del slides._gh_setup
    assert slides._sign_in_after_github_install() is True
    assert started == [True]


def test_the_row_is_read_even_without_its_account_button(slides):
    from shiboken6 import delete

    delete(slides._gh_signup)
    slides._refresh_github()
    assert slides._gh_status.text()


def test_a_panel_that_will_not_stop_does_not_stop_the_screen(slides,
                                                             monkeypatch):
    def refuse():
        raise RuntimeError("stuck")

    stopped = []
    monkeypatch.setattr(slides._ai_setup, "shutdown", refuse)
    monkeypatch.setattr(slides._gh_setup, "shutdown",
                        lambda: stopped.append("github"))
    slides._stop_the_installs()
    assert stopped == ["github"], "the next panel is still stopped"


def test_a_mark_that_is_not_installed_says_choosing_it_offers_the_install(
        slides):
    holder = _assistant(slides)
    tip = holder._buttons["claude"].toolTip()
    assert holder._buttons["claude"].status == ProviderMark.NOT_INSTALLED
    assert "offers to install" in tip
    slides._refresh_provider_marks(holder)
    assert holder._buttons["claude"].toolTip() == tip
    assert "starts the sign-in" in SetupSlides._mark_tip(
        "GPT", ProviderMark.SIGNED_OUT)
    assert "offers" not in SetupSlides._mark_tip("GPT",
                                                  ProviderMark.SIGNED_OUT)
    assert SetupSlides._mark_tip("GPT", ProviderMark.READY) == (
        "Use GPT. You are signed in.")


def test_a_runnable_row_is_not_reported_as_a_missing_program(slides, press,
                                                             present):
    """With curl and bash present, nothing is missing even without a panel."""
    from shiboken6 import delete

    present.update(curl="/usr/bin/curl", bash="/usr/bin/bash")
    delete(slides._ai_setup)
    seen = press("Later")
    slides._start_provider_login("claude")
    assert "Install" not in seen["buttons"]
    assert "none was found" not in seen["informative"]
    assert providers.get_provider("claude").install_hint in \
        seen["informative"]


def test_an_install_that_did_not_start_is_not_blamed_on_another(
        slides, press, present, monkeypatch):
    present.update(curl="/usr/bin/curl", bash="/usr/bin/bash")
    monkeypatch.setattr(slides, "_install_provider",
                        lambda provider, code: False)
    press("Install")
    note = slides._start_provider_login("claude")
    assert "still running" not in note
    assert note == (f"{providers.get_provider('claude').label} is not set "
                    f"up yet.")


def test_with_its_panel_gone_the_github_prompt_shows_the_command_only(
        slides, press, present):
    present.update(brew="/opt/homebrew/bin/brew")
    del slides._gh_setup
    seen = press("Later")
    assert slides._offer_github_install() is False
    assert "Install" not in seen["buttons"]
    assert "none was found" not in seen["informative"]
    assert "brew install gh" in seen["informative"]


def test_a_sign_in_ending_after_the_screen_is_gone_touches_nothing(
        slides, monkeypatch):
    monkeypatch.setattr(slides, "_still_on_screen", lambda: False)
    slides._gh_setup.sign_in(providers.github_cli(), lambda: True)
    slides._github_sign_in_ended(1)
    assert slides._gh_setup.state == SIGNING_IN
    slides._gh_setup._stop_watching()


# ------------------------------------------------------- the panel on its own


class Tool(providers.CommandLineTool):
    """A tool with one install row and no status command."""

    name = "fake"
    label = "Fake"
    cli_name = "fakecli"
    install_methods = (providers.InstallMethod(("npm",), "npm i -g fake"),)
    install_hint = "npm i -g fake"
    login_command = "fake login"


@pytest.fixture
def panel(qtbot, present):
    made = CliSetupPanel(threaded=False, open_page=lambda url: True)
    qtbot.addWidget(made)
    made.show()
    return made


def test_a_tool_that_cannot_be_installed_here_offers_the_page(panel,
                                                              present):
    assert panel.install(Tool(), page="https://example.invalid") is False
    assert panel.state == FAILED
    assert "npm (Node.js)" in panel.message.text()
    assert not panel.install_button.isVisible()
    assert panel.page_button.isVisible() and panel._open_the_page() is True


@pytest.mark.parametrize("kind,words", [
    (cli_install.TIMED_OUT, "20 minutes"),
    (cli_install.NOT_FOUND, "fakecli --version"),
    (cli_install.NOT_STARTED, "could not be started (no bash)"),
    (cli_install.PERMISSION, "not allowed to write"),
    ("something new", "exit status None"),
])
def test_each_outcome_has_its_sentence(panel, present, kind, words):
    present.update(npm="/usr/bin/npm")
    panel._tool = Tool()
    panel._plan = cli_install.InstallPlan(
        Tool(), providers.InstallMethod(("curl",), "curl x | sh", "shell"),
        "linux")
    text = panel._explain(cli_install.InstallOutcome(kind, tail="no bash"))
    assert words in text


def test_an_install_that_raises_is_reported(panel, present, monkeypatch):
    present.update(npm="/usr/bin/npm")

    def explode(*_a, **_k):
        raise ValueError("the worker broke")

    monkeypatch.setattr(cli_install, "run_install", explode)
    assert panel.install(Tool()) is True
    assert panel.state == FAILED
    assert "the worker broke" in panel.message.text()
    panel._job_failed("a check broke")
    assert panel.state == FAILED


def test_an_install_with_nothing_to_follow_it_just_says_so(
        panel, present, monkeypatch):
    present.update(npm="/usr/bin/npm")
    changed = []
    panel.changed.connect(lambda: changed.append(True))
    monkeypatch.setattr(
        cli_install, "run_install",
        lambda plan, on_output, stop: cli_install.InstallOutcome(
            cli_install.INSTALLED, 0, "", ""))
    assert panel.install(Tool()) is True
    assert panel.message.text() == "Fake is installed."
    assert changed == [True]
    panel.cancel()
    assert panel.message.text() == "Fake is installed.", \
        "Cancel with nothing running changes nothing"


def test_a_busy_panel_refuses_a_second_install(panel, present):
    panel.state = INSTALLING
    assert panel.install(Tool()) is False
    panel.cancel()
    assert panel._stop.is_set()
    assert not panel.cancel_button.isEnabled()


def test_a_sign_in_that_takes_too_long_is_given_up_on(panel, monkeypatch):
    starts = []
    tool = providers.github_cli()
    monkeypatch.setattr(providers, "_run_quietly", lambda argv, timeout: 1)
    assert panel.sign_in(tool, lambda: starts.append(1) or True) is True
    panel._poll()
    assert panel.state == SIGNING_IN
    panel._checking = True
    panel._poll()
    panel._checking = False
    panel._deadline = 0
    panel._poll()
    assert panel.state == SIGN_IN_PENDING
    assert "10 minutes" in panel.message.text()
    _click_button = panel.sign_in_button
    assert _click_button.isVisible()
    panel._sign_in_again()
    assert starts == [1, 1]
    panel.cancel()
    assert panel.state == SIGN_IN_PENDING
    assert "Stopped waiting" in panel.message.text()
    panel._checked(True)
    assert panel.state == SIGN_IN_PENDING, "a late answer changes nothing"


def test_a_sign_in_that_will_not_start_says_so(panel):
    def refuse():
        raise OSError("no terminal")

    assert panel.sign_in(Tool(), refuse) is False
    assert panel.state == SIGN_IN_PENDING


def test_long_output_is_shortened_and_a_gone_panel_drops_it(panel):
    from shiboken6 import delete

    panel.state = INSTALLING
    panel._show_output("x" * 500)
    assert len(panel.latest.text()) == panel_module.LATEST_CHARS
    gone = CliSetupPanel(threaded=False)
    delete(gone)
    gone._relay_output("after the panel is gone")


def test_copy_and_the_browser_forgive_a_desktop_that_refuses(
        panel, monkeypatch):
    from PySide6.QtGui import QDesktopServices

    def refuse(*_a):
        raise RuntimeError("no desktop")

    panel.command.setText("npm i -g fake")
    monkeypatch.setattr(QApplication, "clipboard", staticmethod(refuse))
    assert panel.copy_command() == "npm i -g fake"
    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(refuse))
    assert panel_module._open_url("https://example.invalid") is False
    monkeypatch.setattr(QDesktopServices, "openUrl",
                        staticmethod(lambda url: True))
    assert panel_module._open_url("https://example.invalid") is True


def test_captions_fall_back_to_english_without_a_catalog(monkeypatch):
    monkeypatch.setitem(sys.modules, "spacr.qt.i18n", None)
    assert panel_module._say("{label} is installed.", label="X") == \
        "X is installed."
    assert panel_module._say("Install") == "Install"


def test_shutdown_stops_everything(panel):
    panel.shutdown()
    assert panel._stop.is_set()
    assert not panel._timer.isActive()


# ------------------------------------------------- with a real process


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX script")
def test_a_real_installer_process_takes_gpt_to_ready(
        qapp, qtbot, press, monkeypatch, tmp_path, signed_out_of_github):
    """Real pipes, a real worker thread, a real PATH: nothing real installed.

    A shell script named ``npm`` stands in for npm. It prints like an
    installer and writes a ``codex`` script into ``~/.local/bin`` under a
    temporary HOME that is not on PATH -- the folder a desktop session's
    PATH typically leaves out -- and that ``codex`` answers ``login status``
    with 0, as a signed-in Codex does.
    """
    monkeypatch.setattr(cli_install, "_spawn", subprocess.Popen)
    monkeypatch.setattr(providers, "_run_quietly", REAL_RUN_QUIETLY)
    monkeypatch.setattr(cli_install, "likely_folders", REAL_LIKELY_FOLDERS)
    home = tmp_path / "home"
    target = home / ".local" / "bin"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    npm = bin_dir / "npm"
    npm.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = prefix ]; then echo /nonexistent; exit 0; fi\n"
        "echo 'added 1 package in 2s'\n"
        f"mkdir -p '{target}'\n"
        f"printf '#!/bin/sh\\nexit 0\\n' > '{target}/codex'\n"
        f"chmod +x '{target}/codex'\n")
    npm.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin"
                               f"{os.pathsep}/bin")
    monkeypatch.setattr(cli_install, "_query", subprocess.run)
    dialog = SetupSlides()
    qtbot.addWidget(dialog)
    dialog.show()
    try:
        launched = []
        dialog._run_in_a_terminal = lambda c: launched.append(c) or True
        holder = _assistant(dialog)
        panel = dialog._ai_setup
        panel._timer.setInterval(50)
        assert holder._buttons["gpt"].status == ProviderMark.NOT_INSTALLED

        press("Install")
        _click(qtbot, holder._buttons["gpt"])
        qtbot.waitUntil(lambda: panel.state == READY, timeout=20000)

        assert launched == ["codex login"]
        assert os.environ["PATH"].split(os.pathsep)[0] == str(target)
        assert holder._buttons["gpt"].status == ProviderMark.READY
    finally:
        dialog._stop_the_installs()
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()

