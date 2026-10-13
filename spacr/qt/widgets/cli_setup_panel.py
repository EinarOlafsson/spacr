"""Install a vendor command-line tool, then sign in to it, from a screen row.

The setup screen shows one of these under its AI provider marks and one under
its GitHub mark. Pressing Install there runs the tool's own installer --
:func:`spacr.qt.ai.cli_install.run_install`, on a
:class:`spacr.qt.job_runner.JobRunner` worker thread, never on the GUI thread
-- and this panel shows what is happening: the command that is running, the
installer's latest line of output, a Cancel button, and at the end either the
next step or a sentence the user can act on.

After a successful install the owner's ``after_install`` starts the tool's
sign-in, and :meth:`CliSetupPanel.sign_in` watches for it to finish by asking
the tool's own status command every few seconds, again off the GUI thread.

The command is on screen from the first moment to the last, with a Copy
button beside it, for the user who would rather run it themselves.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QProgressBar,
                               QPushButton, QVBoxLayout, QWidget)

from ..ai import cli_install

LOG = logging.getLogger("spacr.qt.cli_setup_panel")

#: Nothing is happening; the panel is hidden.
IDLE = "idle"
#: The installer is running.
INSTALLING = "installing"
#: The installer finished and the CLI was found.
INSTALLED = "installed"
#: The install failed, was cancelled, or cannot run here.
FAILED = "failed"
#: The sign-in was started and the panel is asking the tool whether it is done.
SIGNING_IN = "signing in"
#: The sign-in was started and the tool has no way to say it is done.
CONFIRM = "confirm"
#: The sign-in could not be started, or it was not finished in time.
SIGN_IN_PENDING = "sign-in pending"
#: The tool is installed and signed in.
READY = "ready"

#: Milliseconds between two questions to the tool about its sign-in.
POLL_MS = 3000

#: Minutes the panel waits for a sign-in to finish before it says so.
SIGN_IN_WAIT_MIN = 10

#: Milliseconds :meth:`CliSetupPanel.shutdown` waits for the worker thread.
SHUTDOWN_WAIT_MS = 5000

#: The longest installer line shown, in characters.
LATEST_CHARS = 120

#: What each way an install can end says, with ``{label}`` for the tool.
#:
#: Every sentence names something the user can do next, because a failure
#: that only says it failed leaves them where they started.
OUTCOME_TEXT = {
    cli_install.CANCELLED:
        "Install cancelled. Press Try again to start over, or run the "
        "command below yourself.",
    cli_install.TIMED_OUT:
        "The installer ran for {minutes} minutes without finishing, so spaCR "
        "stopped it. Press Try again, or run the command below in a terminal.",
    cli_install.NO_NETWORK:
        "The installer could not reach its download server. Check the "
        "internet connection and any proxy, then press Try again.",
    cli_install.REFUSED:
        "The download server refused the request. Try again later, or open "
        "the install page for another way to install {label}.",
    cli_install.PERMISSION:
        "The installer was not allowed to write where it installs. Run the "
        "command below in a terminal with the rights it needs, or open the "
        "install page.",
    cli_install.NOT_FOUND:
        "The installer finished, but spaCR cannot find {cli}. Open a new "
        "terminal and run `{cli} --version`; if that works, restart spaCR.",
    cli_install.NOT_STARTED:
        "The installer could not be started ({detail}). Run the command below "
        "in a terminal instead.",
    cli_install.NOT_AUTOMATIC:
        "spaCR cannot install {label} on this computer by itself: it needs "
        "{needs}, and none was found. Install one of them, or open the "
        "install page.",
    cli_install.FAILED:
        "The installer stopped with exit status {status}{said}. Run the "
        "command below in a terminal to see the whole message, or open the "
        "install page.",
}

#: What a permission failure says when the installer was npm.
#:
#: npm's global folder belongs to the system when Node.js came from the
#: distribution, and a user-owned prefix is the fix npm's own documentation
#: gives; ``~/.local/bin`` is one of the folders spaCR looks in afterwards.
NPM_PERMISSION_TEXT = (
    "npm was not allowed to write to its global folder. Run `npm config set "
    "prefix ~/.local` once in a terminal so npm installs into a folder you "
    "own, then press Try again.")


def _say(text: str, **values) -> str:
    """Translate one caption, falling back to the English it was given.

    :param text: the English source, with ``{name}`` fields.
    :param values: the fields' values.
    :returns: the caption in the current language.
    """
    try:
        from ..i18n import tr

        return tr(text, **values)
    except Exception:                                        # noqa: BLE001
        LOG.debug("no translation available", exc_info=True)
        return text.format(**values) if values else text


def _open_url(url: str) -> bool:
    """Open ``url`` in the default browser.

    :param url: the address.
    :returns: whether the desktop accepted it.
    """
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices

    try:
        return bool(QDesktopServices.openUrl(QUrl(str(url))))
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not open %r", url, exc_info=True)
        return False


class CliSetupPanel(QWidget):
    """Install one command-line tool and see its sign-in through.

    :param parent: the owning widget.
    :param threaded: ``False`` runs the installer and the sign-in checks
        inline, for a test that drives the panel synchronously.
    :param open_page: opens an install page; the default browser by default.
    :ivar state: what the panel is doing, one of this module's states.
    """

    #: The tool's install or sign-in state may have changed.
    changed = Signal()

    #: Internal relay: one line of installer output, emitted from the worker
    #: thread and received on the GUI thread.
    _output_arrived = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None, *,
                 threaded: bool = True,
                 open_page: Optional[Callable[[str], bool]] = None):
        """Build the panel, hidden until there is something to show.

        :param parent: the owning widget.
        :param threaded: run jobs on a worker thread.
        :param open_page: opens an install page.
        """
        super().__init__(parent)
        from ..job_runner import JobRunner

        self.setProperty("spacrClearContainer", True)
        self._open_page = open_page or _open_url
        self.state = IDLE
        self._tool: Any = None
        self._plan: Optional[cli_install.InstallPlan] = None
        self._after_install: Optional[Callable[[], Any]] = None
        self._start: Optional[Callable[[], bool]] = None
        self._page = ""
        self._stop = threading.Event()
        self._deadline = 0.0
        self._checking = False

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 4, 0, 0)
        column.setSpacing(6)

        self.message = QLabel("")
        self.message.setObjectName("Muted")
        self.message.setWordWrap(True)
        column.addWidget(self.message)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(6)
        column.addWidget(self.progress)

        self.latest = QLabel("")
        self.latest.setObjectName("Muted")
        self.latest.setProperty("i18nSkipText", True)
        column.addWidget(self.latest)

        self.command = QLineEdit("")
        self.command.setReadOnly(True)
        self.command.setToolTip(
            "The command spaCR runs. Copy it to run it in a terminal "
            "yourself instead.")
        column.addWidget(self.command)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.install_button = QPushButton("Install")
        self.install_button.clicked.connect(self._install_again)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel)
        self.sign_in_button = QPushButton("Sign in")
        self.sign_in_button.clicked.connect(self._sign_in_again)
        self.done_button = QPushButton("Done")
        self.done_button.clicked.connect(self.done)
        self.copy_button = QPushButton("Copy the command")
        self.copy_button.clicked.connect(self.copy_command)
        self.page_button = QPushButton("Open the page")
        self.page_button.clicked.connect(self._open_the_page)
        for button in (self.install_button, self.cancel_button,
                       self.sign_in_button, self.done_button,
                       self.copy_button, self.page_button):
            row.addWidget(button)
        row.addStretch(1)
        column.addLayout(row)

        self._runner = JobRunner(self, threaded=threaded, app_key="setup",
                                 user_visible=False)
        self._runner.job_failed.connect(self._job_failed)
        self._output_arrived.connect(self._show_output)
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._show(IDLE, "")

    def is_busy(self) -> bool:
        """Whether an installer is running."""
        return self.state == INSTALLING

    def _label(self) -> str:
        """The tool's name as the user reads it."""
        return str(getattr(self._tool, "label", "") or "")

    def tool_label(self) -> str:
        """The name of the tool this panel is installing or signing in to.

        :returns: its label, such as ``"Claude"``, or ``""`` before the
            panel has been given a tool.
        """
        return self._label()

    def _show(self, state: str, message: str) -> None:
        """Enter ``state``: say ``message`` and show the controls it needs.

        :param state: one of this module's states.
        :param message: the sentence to show, already translated.
        """
        self.state = state
        self.message.setText(message)
        installing = state == INSTALLING
        automatic = self._plan is not None and self._plan.automatic
        self.progress.setVisible(installing)
        self.latest.setVisible(installing and bool(self.latest.text()))
        self.command.setVisible(state not in (IDLE, READY)
                                and bool(self.command.text()))
        self.cancel_button.setVisible(state in (INSTALLING, SIGNING_IN))
        self.cancel_button.setEnabled(True)
        self.install_button.setVisible(state == FAILED and automatic)
        self.sign_in_button.setVisible(
            state == SIGN_IN_PENDING and self._start is not None)
        self.done_button.setVisible(state in (CONFIRM, SIGN_IN_PENDING))
        self.copy_button.setVisible(state in (FAILED, CONFIRM,
                                              SIGN_IN_PENDING)
                                    and bool(self.command.text()))
        self.page_button.setVisible(state == FAILED and bool(self._page))
        self.setVisible(state != IDLE)

    def install(self, tool: Any, *,
                after_install: Optional[Callable[[], Any]] = None,
                page: str = "") -> bool:
        """Run ``tool``'s installer off the GUI thread.

        Shows the command before anything runs and keeps it on screen. When
        none of the tool's install methods can run on this computer, says
        which program is missing instead and offers the command and the
        install page.

        :param tool: a :class:`spacr.qt.ai.providers.CommandLineTool`.
        :param after_install: called on the GUI thread once the tool is
            installed, to start its sign-in.
        :param page: the tool's install page, offered when an install fails.
        :returns: ``True`` when an installer was started.
        """
        if self.is_busy():
            return False
        self._stop_watching()
        self._tool = tool
        self._after_install = after_install
        self._page = str(page or "")
        self._plan = cli_install.plan_install(tool)
        self.command.setText(self._plan.command)
        self.latest.setText("")
        self.install_button.setText(_say("Try again"))
        if not self._plan.automatic:
            self._show(FAILED, self._explain(
                cli_install.InstallOutcome(cli_install.NOT_AUTOMATIC)))
            return False
        self._stop = threading.Event()
        self._show(INSTALLING, _say("Installing {label}…",
                                    label=self._label()))
        plan, stop = self._plan, self._stop
        self._runner.submit(
            lambda: cli_install.run_install(plan, self._relay_output, stop),
            self._install_ended)
        return True

    def _relay_output(self, line: str) -> None:
        """Pass one installer line to the GUI thread.

        Runs on the worker thread, where emitting a signal is the only safe
        thing to do. A panel already deleted by then drops the line.

        :param line: the line.
        """
        try:
            self._output_arrived.emit(str(line))
        except RuntimeError:
            pass

    def _show_output(self, line: str) -> None:
        """Show the installer's latest line under the progress bar.

        :param line: the line.
        """
        text = str(line)
        if len(text) > LATEST_CHARS:
            text = text[:LATEST_CHARS - 1] + "…"
        self.latest.setText(text)
        self.latest.setVisible(self.state == INSTALLING)

    def _install_ended(self, outcome: cli_install.InstallOutcome) -> None:
        """Say how the install went, and start the sign-in when it worked.

        :param outcome: from :func:`spacr.qt.ai.cli_install.run_install`.
        """
        if not outcome.ok:
            self._show(FAILED, self._explain(outcome))
            return
        cli_install.put_on_path(outcome.location)
        self._show(INSTALLED, _say("{label} is installed.",
                                   label=self._label()))
        self.changed.emit()
        if self._after_install is not None:
            self._after_install()

    def _explain(self, outcome: cli_install.InstallOutcome) -> str:
        """The sentence for a failed install, in the current language.

        :param outcome: how the install ended.
        :returns: what went wrong and what to do about it.
        """
        method = self._plan.method if self._plan is not None else None
        if (outcome.kind == cli_install.PERMISSION and method is not None
                and method.needs[:1] == ("npm",)):
            return _say(NPM_PERMISSION_TEXT)
        template = OUTCOME_TEXT.get(outcome.kind,
                                    OUTCOME_TEXT[cli_install.FAILED])
        said = f": {outcome.tail}" if outcome.tail else ""
        return _say(template, label=self._label(),
                    cli=str(getattr(self._tool, "cli_name", "") or ""),
                    needs=self._plan.needs if self._plan is not None else "",
                    minutes=cli_install.INSTALL_TIMEOUT_S // 60,
                    status=outcome.exit_status, said=said,
                    detail=outcome.tail)

    def _job_failed(self, text: str) -> None:
        """Report a worker job that raised instead of returning.

        :param text: the job's error, one line.
        """
        self._checking = False
        if self.state == INSTALLING:
            self._show(FAILED, _say(
                "The installer could not be run: {error}", error=text))
            return
        LOG.debug("a sign-in check failed: %s", text)

    def _install_again(self) -> bool:
        """What Install and Try again do: run the same install once more."""
        return self.install(self._tool, after_install=self._after_install,
                            page=self._page)

    def sign_in(self, tool: Any, start: Callable[[], bool]) -> bool:
        """Start ``tool``'s sign-in and watch for it to finish.

        :param tool: a :class:`spacr.qt.ai.providers.CommandLineTool`.
        :param start: starts the sign-in -- in a terminal for an AI CLI, in
            spaCR itself for the GitHub CLI -- and says whether it started.
        :returns: whether the sign-in started.
        """
        self._stop_watching()
        self._tool = tool
        self._start = start
        self.command.setText(str(getattr(tool, "login_command", "") or ""))
        try:
            started = bool(start())
        except Exception:                                    # noqa: BLE001
            LOG.debug("the sign-in would not start", exc_info=True)
            started = False
        if not started:
            self._show(SIGN_IN_PENDING, _say(
                "The sign-in could not be started from here. Run the command "
                "below in a terminal, then press Done."))
            return False
        if getattr(tool, "status_command", ()):
            self._deadline = time.monotonic() + SIGN_IN_WAIT_MIN * 60
            self._show(SIGNING_IN, _say(
                "Sign in to {label} in the window that opened. spaCR notices "
                "when you are done.", label=self._label()))
            self._timer.start()
        else:
            self._show(CONFIRM, _say(
                "Finish signing in to {label} in the window that opened, then "
                "press Done.", label=self._label()))
        return True

    def _sign_in_again(self) -> bool:
        """What the Sign in button does: start the same sign-in again."""
        return self.sign_in(self._tool, self._start)

    def _poll(self) -> None:
        """Ask the tool, off the GUI thread, whether it is signed in yet."""
        if self.state != SIGNING_IN or self._checking:
            return
        if time.monotonic() >= self._deadline:
            self._stop_watching()
            self._show(SIGN_IN_PENDING, _say(
                "Signing in to {label} did not finish within {minutes} "
                "minutes. Press Sign in to try again.",
                label=self._label(), minutes=SIGN_IN_WAIT_MIN))
            return
        self._checking = True
        self._runner.submit(self._tool.check_signed_in, self._checked)

    def _checked(self, signed_in: Optional[bool]) -> None:
        """Take the tool's answer about its sign-in.

        :param signed_in: ``True`` once the tool says it is signed in.
        """
        self._checking = False
        if self.state != SIGNING_IN or signed_in is not True:
            return
        self._stop_watching()
        self._show(READY, _say("{label} is signed in and ready.",
                               label=self._label()))
        self.changed.emit()

    def sign_in_failed(self, message: str) -> None:
        """Say that a sign-in the owner was running has ended unfinished.

        :param message: what happened and what to do, already translated.
        """
        self._stop_watching()
        self._show(SIGN_IN_PENDING, message)

    def _stop_watching(self) -> None:
        """Stop asking the tool about its sign-in."""
        self._timer.stop()
        self._checking = False

    def cancel(self) -> None:
        """Stop the installer, or stop waiting for the sign-in."""
        if self.state == INSTALLING:
            self._stop.set()
            self.message.setText(_say("Stopping the installer…"))
            self.cancel_button.setEnabled(False)
            return
        if self.state == SIGNING_IN:
            self._stop_watching()
            self._show(SIGN_IN_PENDING, _say(
                "Stopped waiting for the sign-in. Press Sign in to start it "
                "again, or Done once you have signed in."))

    def done(self) -> None:
        """Close the panel and have the owner look at the tool again."""
        self._stop_watching()
        self._show(IDLE, "")
        self.changed.emit()

    def copy_command(self) -> str:
        """Put the command on the clipboard.

        :returns: the command copied, which stays on screen either way.
        """
        text = self.command.text()
        try:
            from PySide6.QtWidgets import QApplication

            QApplication.clipboard().setText(text)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not copy the command", exc_info=True)
        return text

    def _open_the_page(self) -> bool:
        """Open the tool's install page in the browser."""
        return bool(self._open_page(self._page))

    def shutdown(self) -> None:
        """Stop the installer and the sign-in watch, and retire the thread.

        Called when the screen closes: an installer left running with no
        Cancel button on screen is one the user can no longer stop. The
        worker's own answer is dropped once the thread is retired, so an
        install that was running is shown as cancelled here, and the panel
        no longer counts as busy.
        """
        self._stop.set()
        self._stop_watching()
        self._runner.shutdown(SHUTDOWN_WAIT_MS)
        if self.state == INSTALLING:
            self._show(FAILED, self._explain(
                cli_install.InstallOutcome(cli_install.CANCELLED)))
