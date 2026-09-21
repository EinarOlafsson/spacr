"""Sign in to an AI command-line tool without leaving spaCR.

Item 420: "installed, logged in and marked READY, with no terminal used".
The Claude, GPT and Gemini command-line tools sign in by conversation -- a web
page to confirm in, sometimes a code to paste back -- and they only hold that
conversation with a TERMINAL, which is why spaCR used to open one. A
pseudo-terminal is a terminal as far as the tool can tell, so the sign-in is
run on one here, inside a small window: the web page the tool prints is
opened, what it asks is shown, and the answer is typed into a box and sent.

POSIX only (Linux, macOS), where a pseudo-terminal is part of the standard
library. On Windows the caller falls back to opening a terminal, as before.
"""
from __future__ import annotations

import os
import queue
import re
import subprocess
import threading
from typing import Dict, List, Optional, Sequence

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
    QVBoxLayout,
)

from ..i18n import tr

__all__ = [
    "pty_available",
    "strip_terminal_codes",
    "find_urls",
    "PtySession",
    "SignInDialog",
]

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
                   r"|\x1b[@-Z\\-_]")
_URL = re.compile(r"https?://[^\s\"'<>\x1b]+")


def pty_available() -> bool:
    """Whether a sign-in can run on a pseudo-terminal here.

    :returns: True on Linux and macOS.
    """
    if os.name != "posix":
        return False
    try:
        import pty

        del pty
    except ImportError:
        return False
    return True


def strip_terminal_codes(text: str) -> str:
    """``text`` without colour, cursor and title escape sequences.

    :param text: raw terminal output.
    :returns: what a reader would see.
    """
    return _ANSI.sub("", text).replace("\r\n", "\n").replace("\r", "\n")


def find_urls(text: str) -> List[str]:
    """Every web address in ``text``, in order, without trailing punctuation.

    :param text: terminal output, escape sequences removed.
    :returns: the addresses.
    """
    return [url.rstrip(").,;]") for url in _URL.findall(text)]


class PtySession:
    """One command running on a pseudo-terminal, read off its own thread.

    :param argv: the command.
    :param env: its environment; the current one when None.
    """

    def __init__(self, argv: Sequence[str], env: Optional[Dict[str, str]] = None):
        """Start ``argv``; output collects in :meth:`read`."""
        master, slave = os.openpty()
        environ = dict(os.environ if env is None else env)
        environ.setdefault("TERM", "xterm-256color")
        self._master = master
        self._queue: "queue.Queue[str]" = queue.Queue()
        self.proc = subprocess.Popen(
            list(argv), stdin=slave, stdout=slave, stderr=slave, env=environ,
            start_new_session=True, close_fds=True)
        os.close(slave)
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        """Move the terminal's output to the queue until it closes."""
        while True:
            try:
                chunk = os.read(self._master, 4096)
            except OSError:
                break
            if not chunk:
                break
            self._queue.put(chunk.decode("utf-8", "replace"))

    def read(self) -> str:
        """Everything the command has written since the last call.

        :returns: the text, escape sequences removed.
        """
        parts = []
        while True:
            try:
                parts.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return strip_terminal_codes("".join(parts))

    def send(self, text: str) -> None:
        """Type ``text`` and press Enter.

        :param text: the answer.
        """
        os.write(self._master, (str(text) + "\r").encode("utf-8"))

    def poll(self) -> Optional[int]:
        """The exit code, or None while it runs."""
        return self.proc.poll()

    def stop(self) -> None:
        """End the command and close the terminal."""
        if self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
        try:
            os.close(self._master)
        except OSError:
            pass


class SignInDialog(QDialog):
    """A tool's sign-in, held inside spaCR.

    :param title: what is being signed in to, e.g. ``"Claude"``.
    :param argv: the tool's own sign-in command.
    :param parent: the window it belongs to.
    :param session_factory: ``fn(argv) -> PtySession``, for tests.
    :param open_url: ``fn(url)``; opens the system browser when None.
    """

    finished_signing_in = Signal(bool)

    def __init__(self, title: str, argv: Sequence[str], parent=None, *,
                 session_factory=None, open_url=None):
        """Start the sign-in and show what it says."""
        super().__init__(parent)
        self.setWindowTitle(tr("Sign in to {name}", name=title))
        self._open_url = open_url or (lambda url: QDesktopServices.openUrl(QUrl(url)))
        self._urls: List[str] = []
        self.succeeded: Optional[bool] = None
        layout = QVBoxLayout(self)
        self._intro = QLabel(tr(
            "{name} is signing in. Its sign-in page opens in your browser; "
            "if it asks for a code, paste it below and press Send.",
            name=title))
        self._intro.setWordWrap(True)
        layout.addWidget(self._intro)
        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(180)
        layout.addWidget(self.log)
        link_row = QHBoxLayout()
        self._link = QLabel("", self)
        self._link.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._link.setWordWrap(True)
        link_row.addWidget(self._link, 1)
        self.open_btn = QPushButton(tr("Open sign-in page"), self)
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_latest)
        link_row.addWidget(self.open_btn)
        layout.addLayout(link_row)
        answer_row = QHBoxLayout()
        self.answer = QLineEdit(self)
        self.answer.setPlaceholderText(tr("Code or answer the tool asks for"))
        self.answer.returnPressed.connect(self._send)
        answer_row.addWidget(self.answer, 1)
        self.send_btn = QPushButton(tr("Send"), self)
        self.send_btn.clicked.connect(self._send)
        answer_row.addWidget(self.send_btn)
        layout.addLayout(answer_row)
        self.close_btn = QPushButton(tr("Cancel"), self)
        self.close_btn.clicked.connect(self.reject)
        layout.addWidget(self.close_btn)
        factory = session_factory or PtySession
        self.session = factory(list(argv))
        self._timer = QTimer(self)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self) -> None:
        """Show new output, open a new sign-in page, notice the end."""
        text = self.session.read()
        if text:
            self.log.moveCursor(QTextCursor.End)
            self.log.insertPlainText(text)
            for url in find_urls(text):
                if url not in self._urls:
                    self._urls.append(url)
                    self._link.setText(url)
                    self.open_btn.setEnabled(True)
                    if len(self._urls) == 1:
                        self._open_url(url)
        code = self.session.poll()
        if code is not None:
            self._timer.stop()
            self.succeeded = code == 0
            self._intro.setText(
                tr("Signed in.") if self.succeeded else
                tr("The sign-in ended without success (exit code {code}). "
                   "What it said is above.", code=code))
            self.close_btn.setText(tr("Close"))
            self.send_btn.setEnabled(False)
            self.answer.setEnabled(False)
            self.finished_signing_in.emit(bool(self.succeeded))

    def _open_latest(self) -> None:
        """Open the most recent sign-in page again."""
        if self._urls:
            self._open_url(self._urls[-1])

    def _send(self) -> None:
        """Send what is in the answer box."""
        text = self.answer.text()
        if not text:
            return
        self.session.send(text)
        self.log.appendPlainText(tr("[sent]"))
        self.answer.clear()

    def done(self, result: int) -> None:
        """Stop the sign-in when the window closes."""
        self._timer.stop()
        try:
            self.session.stop()
        except Exception:
            pass
        super().done(result)
