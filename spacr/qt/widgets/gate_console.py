"""A console and a chat box beside the gating surface.

Two panes with one job between them: let the user ask a question about the
table they are gating without leaving the screen to do it.

``Console``
    Runs an expression against the CURRENT frame and prints the answer. Not a
    Python shell -- a Python shell in a GUI is a way to hang the GUI, and the
    questions that come up while gating are all of one shape: "how many
    objects satisfy this", "what is the median of that". So it evaluates
    pandas expressions with the frame in scope and nothing else.

``Chat``
    The same box, addressed in English. It is wired to whatever assistant the
    host provides and is EMPTY when none is configured -- it says so rather
    than pretending to think, because a chat box that silently ignores you is
    worse than one that is honestly unavailable.

Both write into one transcript, so the record of what was asked and what came
back reads in order regardless of which pane asked.
"""
from __future__ import annotations

import logging
import traceback
from typing import Any, Callable, Optional

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QVBoxLayout,
    QWidget,
)

from ..theme import SPACING, register_widget_qss

LOG = logging.getLogger("spacr.qt.gate_console")

QSS_NAME = "GateConsole"

#: Names an expression may use, beyond the frame itself. Deliberately short:
#: this is a question box, not a scripting environment, and every name added
#: here is one more thing that can be typed by accident.
SAFE_NAMES = ("pd", "np", "len", "abs", "min", "max", "sum", "round",
              "sorted", "True", "False", "None")


def _console_qss(palette, opacity=None) -> str:
    return f"""
    QTextEdit#GateConsoleLog {{
        background: transparent;
        color: {palette['fg']};
        border: none;
        font-family: monospace;
    }}
    QLineEdit#GateConsoleInput, QLineEdit#GateChatInput {{
        background: {palette['surface_alt']};
        color: {palette['fg']};
        border: 1px solid {palette['border']};
        border-radius: 4px;
        padding: 3px 6px;
    }}
    QLabel#GateConsoleHint {{
        color: {palette['fg_muted']};
        background: transparent;
    }}
    QWidget#GateConsole {{
        background: transparent;
    }}
    """


register_widget_qss(QSS_NAME, _console_qss, replace=True)


def evaluate(expression: str, frame: Optional[pd.DataFrame]) -> str:
    """Answer one question about ``frame``.

    The frame is in scope as ``df``, and every column as itself, so
    ``area.mean()`` and ``df['area'].mean()`` both work -- the first is what
    people type.

    Errors come back as text rather than exceptions: this is a question box,
    and a typo is a normal thing to do in one.
    """
    text = str(expression or "").strip()
    if not text:
        return ""
    if frame is None or frame.empty:
        return "no table loaded"

    import numpy as np

    scope = {"df": frame, "pd": pd, "np": np}
    for column in frame.columns:
        name = str(column)
        if name.isidentifier() and name not in scope:
            scope[name] = frame[column]

    try:
        # eval, not exec: an expression has a VALUE, which is the thing being
        # asked for. Statements would let the user rebind `df` and then
        # wonder why the plot disagrees with the console.
        value = eval(text, {"__builtins__": _builtins()}, scope)  # noqa: S307
    except Exception as exc:
        LOG.debug("console expression failed", exc_info=True)
        return f"{type(exc).__name__}: {exc}"

    if isinstance(value, pd.DataFrame):
        return f"{len(value):,} rows × {len(value.columns)} columns"
    if isinstance(value, pd.Series):
        if value.dtype == bool:
            return f"{int(value.sum()):,} of {len(value):,} objects"
        return str(value.describe())
    return str(value)


def _builtins() -> dict:
    """The handful of builtins an expression may use.

    An allowlist rather than the real builtins: `__import__` and `open` in a
    box the user types into is a way to lose a dataset by typo, and none of
    the questions this box exists for need them.
    """
    import builtins

    return {name: getattr(builtins, name)
            for name in ("len", "abs", "min", "max", "sum", "round", "sorted",
                         "list", "dict", "set", "tuple", "float", "int", "str",
                         "bool", "range", "zip", "enumerate", "any", "all")
            if hasattr(builtins, name)}


class GateConsole(QWidget):
    """The console and the chat box, sharing one transcript."""

    #: A question was asked of the assistant. The host answers by calling
    #: :meth:`reply`; nothing here talks to a network.
    asked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("GateConsole")
        self._frame: Optional[pd.DataFrame] = None
        self._responder: Optional[Callable[[str], str]] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACING["xs"])

        self.log = QTextEdit(self)
        self.log.setObjectName("GateConsoleLog")
        self.log.setReadOnly(True)
        outer.addWidget(self.log, 1)

        hint = QLabel("Ask with an expression — area.mean(), (area > 500).sum()",
                      self)
        hint.setObjectName("GateConsoleHint")
        hint.setWordWrap(True)
        outer.addWidget(hint)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.input = QLineEdit(self)
        self.input.setObjectName("GateConsoleInput")
        self.input.setPlaceholderText("expression")
        self.input.returnPressed.connect(self.run_input)
        row.addWidget(self.input, 1)
        run = QPushButton("Run", self)
        run.clicked.connect(self.run_input)
        row.addWidget(run)
        outer.addLayout(row)

        chat_row = QHBoxLayout()
        chat_row.setContentsMargins(0, 0, 0, 0)
        self.chat = QLineEdit(self)
        self.chat.setObjectName("GateChatInput")
        self.chat.setPlaceholderText("ask in words")
        self.chat.returnPressed.connect(self.send_chat)
        chat_row.addWidget(self.chat, 1)
        send = QPushButton("Ask", self)
        send.clicked.connect(self.send_chat)
        chat_row.addWidget(send)
        outer.addLayout(chat_row)

    # -- state ------------------------------------------------------------
    def set_frame(self, frame: Optional[pd.DataFrame]) -> None:
        self._frame = frame

    def set_responder(self, responder: Optional[Callable[[str], str]]) -> None:
        """Give the chat box something to answer with.

        Without one it says it is not configured rather than staying silent:
        a chat box that ignores you is worse than one that is honestly
        unavailable.
        """
        self._responder = responder

    def transcript(self) -> str:
        return self.log.toPlainText()

    # -- asking -----------------------------------------------------------
    def write(self, line: str, *, prefix: str = "") -> None:
        self.log.append(f"{prefix}{line}" if prefix else line)

    def run(self, expression: str) -> str:
        """Evaluate ``expression`` and record both halves."""
        text = str(expression or "").strip()
        if not text:
            return ""
        self.write(text, prefix="› ")
        answer = evaluate(text, self._frame)
        self.write(answer)
        return answer

    def run_input(self) -> None:
        if self.run(self.input.text()):
            self.input.clear()

    def ask(self, question: str) -> str:
        """Put a question to the assistant, or say there is not one."""
        text = str(question or "").strip()
        if not text:
            return ""
        self.write(text, prefix="? ")
        self.asked.emit(text)
        if self._responder is None:
            answer = ("no assistant is configured for this build — the "
                      "expression box above works without one")
        else:
            try:
                answer = str(self._responder(text))
            except Exception as exc:
                LOG.debug("responder failed", exc_info=True)
                answer = f"the assistant could not answer: {exc}"
        self.write(answer)
        return answer

    def send_chat(self) -> None:
        if self.ask(self.chat.text()):
            self.chat.clear()

    def reply(self, answer: str) -> None:
        """Record an answer that arrived later, from an async host."""
        self.write(str(answer))
