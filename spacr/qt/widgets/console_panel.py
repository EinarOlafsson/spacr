"""
ConsolePanel — merged pipeline console + AI chat panel.

One vertical scrolling area shows both pipeline stdout AND AI chat
messages, separated by dark-gray "topic" bars ("Mask", "Measure",
"spaCR AI", …). Below the scroll sits an input row where the user
can type at any time; a switch on the left decides whether the
message goes to the AI or is ignored.

Public API
----------
* begin_topic(label)          — insert a dark-gray divider bar
                                (used at the start of every pipeline
                                run and every time we switch to/from
                                the AI)
* append_stdout(text)         — append pipeline output; if the last
                                entry isn't already a stdout block it
                                starts a new one
* append_error(traceback)     — same as stdout but red-tinted
* open_error_flow(tb, app)    — inject the AI-explainer prompt for a
                                traceback and stream the reply into
                                a fresh spaCR-AI section
* clear()                     — wipe every entry

Streaming state
---------------
The panel owns the AI thread+worker itself so state stays coherent
even as the user switches between pipeline apps.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import QRect, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFontDatabase, QKeyEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import ai as ai_module
from .. import iconset
from ..ai.providers import ChatProvider
from ..ai.worker import StreamWorker, make_stream_thread
from ..theme import FONT_SIZE, PALETTE, SPACING


# ---------------------------------------------------------------------------
# Divider bar with a topic label
# ---------------------------------------------------------------------------

class _TopicBar(QFrame):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("ConsoleTopicBar")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(SPACING["md"], SPACING["xs"],
                                SPACING["md"], SPACING["xs"])
        self._label = QLabel(label)
        self._label.setObjectName("ConsoleTopicLabel")
        lay.addWidget(self._label)
        lay.addStretch(1)


# ---------------------------------------------------------------------------
# Stdout block (grows in place while pipeline is running)
# ---------------------------------------------------------------------------

class _StdoutBlock(QLabel):
    def __init__(self, text: str = "", error: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("ConsoleStdoutBlockError"
                            if error else "ConsoleStdoutBlock")
        self.setTextFormat(Qt.PlainText)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setWordWrap(True)
        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        self.setFont(mono)
        self._buf: List[str] = []
        if text:
            self.append(text)

    def append(self, text: str) -> None:
        self._buf.append(text)
        # Cap the buffer to keep the UI snappy for very long runs.
        joined = "".join(self._buf)
        if len(joined) > 200_000:
            joined = joined[-200_000:]
            self._buf = [joined]
        self.setText(joined)


# ---------------------------------------------------------------------------
# Chat bubble
# ---------------------------------------------------------------------------

class _Bubble(QFrame):
    """Chat bubble — a coloured QFrame that renders wrapped rich text.

    Manual sizing: on every resizeEvent we clamp the inner label's
    width to our own width minus padding, then set the label's fixed
    height from QFontMetrics.boundingRect for that wrap width. The
    frame's height is set to match. Simple, works reliably even
    inside a QScrollArea.
    """

    _H_PAD = 24     # inner horizontal padding
    _V_PAD = 12     # inner vertical padding

    def __init__(self, role: str, text: str = "", parent=None):
        super().__init__(parent)
        self.role = role
        self.setObjectName(
            "ConsoleBubbleUser" if role == "user" else "ConsoleBubbleAI"
        )
        self._recalc_guard = False
        self._label = QLabel(self)
        self._label.setObjectName("ConsoleBubbleText")
        self._label.setTextFormat(Qt.RichText)
        self._label.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse
        )
        self._label.setOpenExternalLinks(True)
        self._label.setWordWrap(True)
        self._label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self._label.setStyleSheet(
            "QLabel#ConsoleBubbleText {"
            f"  color: {PALETTE['fg']};"
            f"  font-size: {FONT_SIZE['body']}px;"
            "  background: transparent;"
            "  border: none;"
            "}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(SPACING["md"], SPACING["sm"],
                                SPACING["md"], SPACING["sm"])
        lay.setSpacing(0)
        lay.addWidget(self._label)
        self._raw_text = ""
        self._prefix = "spaCR user: " if role == "user" else "spaCR AI: "
        if text:
            self.set_text(text)

    def set_text(self, text: str) -> None:
        self._raw_text = text or ""
        safe = self._raw_text.replace("<", "&lt;").replace(">", "&gt;")
        safe = safe.replace("\n", "<br>")
        html = f'<span style="opacity:0.7;">{self._prefix}</span>{safe}'
        self._label.setText(html)
        self._recalc()

    def _recalc(self) -> None:
        """Fit the label + frame to the wrapped text at our current
        width. Uses QLabel.heightForWidth which — for a word-wrap
        enabled label — returns the correct line-broken height."""
        if self._recalc_guard:
            return   # setFixedHeight below triggers a resizeEvent → guard
        w = self.width()
        if w <= 0:
            return
        text_width = max(120, w - self._H_PAD)
        self._recalc_guard = True
        try:
            self._label.setMaximumWidth(text_width)
            self._label.setMinimumWidth(text_width)
            h = self._label.heightForWidth(text_width)
            if h <= 0:
                h = self._label.sizeHint().height()
            self._label.setFixedHeight(h)
            self.setFixedHeight(h + self._V_PAD)
        finally:
            self._recalc_guard = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._recalc()

    def showEvent(self, event):
        super().showEvent(event)
        self._recalc()


# ---------------------------------------------------------------------------
# Chat input — Enter sends, Shift+Enter newline
# ---------------------------------------------------------------------------

class _ChatInput(QTextEdit):
    submitted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(48)
        self.setMaximumHeight(120)
        self.setAcceptRichText(False)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
                return
            self.submitted.emit()
            return
        super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

class ConsolePanel(QWidget):
    # Fires when an AI stream ends (ok or error) so the AppScreen
    # actions row can flip a Cancel button back to something else.
    ai_stream_finished = Signal()

    def __init__(self, active_app_label: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("ConsolePanel")
        self._active_app_label = active_app_label or ""
        self._last_entry_kind: str = ""   # "stdout" | "ai" | ""
        self._current_stdout: Optional[_StdoutBlock] = None
        self._ai_messages: List[Dict] = []
        self._ai_buf: List[str] = []
        self._ai_thread: Optional[QThread] = None
        self._ai_worker: Optional[StreamWorker] = None
        # Retired stream (thread, worker) pairs — we hold these until
        # thread.finished actually emits so Python doesn't GC the
        # QThread while its OS thread is still winding down (which is
        # what causes `QThread: Destroyed while thread '' is still
        # running / Aborted` on the second consecutive AI request).
        self._retired: List = []

        self._build_ui()
        # Pipe records from the global logger into this console. Every
        # ConsolePanel subscribes to the same shared signal handler,
        # so log records fanned out across screens all see them.
        try:
            from ..logging_util import get_signal_handler
            get_signal_handler().record_ready.connect(self._on_log_record)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Scroll area of entries
        self._scroll = QScrollArea()
        self._scroll.setObjectName("ConsoleScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        # Never show a horizontal scrollbar — content that doesn't fit
        # must wrap. This is what prevents the runaway-width crash.
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._holder = QWidget()
        self._holder.setObjectName("ConsoleHolder")
        self._entries = QVBoxLayout(self._holder)
        self._entries.setContentsMargins(0, 0, 0, 0)
        self._entries.setSpacing(0)
        self._entries.addStretch(1)
        self._scroll.setWidget(self._holder)
        outer.addWidget(self._scroll, 1)

        # Input row — just a text field, no Send button. Users press
        # Enter to submit. AI on/off toggle + provider selector live
        # in the AppScreen actions row (bottom-right of the screen),
        # not here.
        input_bar = QFrame()
        input_bar.setObjectName("ConsoleInputBar")
        row = QHBoxLayout(input_bar)
        row.setContentsMargins(SPACING["md"], SPACING["sm"],
                                SPACING["md"], SPACING["sm"])
        row.setSpacing(SPACING["sm"])

        self._input = _ChatInput()
        self._input.setPlaceholderText(
            "Type here and hit Enter…  (toggle AI at the bottom-right "
            "to route through your chat subscription)"
        )
        self._input.submitted.connect(self._on_submit)
        row.addWidget(self._input, 1)
        outer.addWidget(input_bar)

        # AppScreen creates + owns the AI toggle/provider menu and calls
        # our setters when they change. Panel-internal state stays here
        # so we always know what to do on Enter.
        self._ai_active: bool = False
        self._current_provider_name: Optional[str] = None

    # ------------------------------------------------------------------
    # Entry-management helpers
    # ------------------------------------------------------------------
    def _insert_entry(self, w: QWidget) -> None:
        """Every entry — topic bar, stdout block, chat bubble — spans
        the full width of the console. Bubbles no longer get a
        horizontal offset row."""
        self._entries.insertWidget(self._entries.count() - 1, w)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        sb = self._scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _needs_topic(self, kind: str) -> bool:
        return self._last_entry_kind != kind

    # ------------------------------------------------------------------
    # Public: pipeline hooks
    # ------------------------------------------------------------------
    def set_active_app(self, label: str) -> None:
        self._active_app_label = label

    def begin_topic(self, label: str) -> None:
        """Insert a divider bar labeled `label` (e.g. 'Mask'). Callers
        can force a new section this way. AI content NEVER uses this
        — AI replies flow inline in the same stdout block."""
        self._insert_entry(_TopicBar(label))
        self._last_entry_kind = ""    # force next append to open a block
        self._current_stdout = None

    def append_stdout(self, text: str) -> None:
        """Append pipeline output. Opens a fresh stdout block (with a
        topic divider) at the very first stdout of a session; opens a
        divider-less block after a bubble breaks the flow."""
        if not text:
            return
        if self._current_stdout is None:
            # First-ever stdout: show a topic divider once. Subsequent
            # bubble-broken flows just get a fresh block without a
            # divider, so the AI reply feels like inline console output.
            if self._last_entry_kind == "":
                self.begin_topic(self._active_app_label or "Pipeline")
            self._current_stdout = _StdoutBlock()
            self._insert_entry(self._current_stdout)
            self._last_entry_kind = "stdout"
        self._current_stdout.append(text)
        self._scroll_to_bottom()

    def _on_log_record(self, text: str, level: int) -> None:
        """Slot for QtLogHandler.record_ready. WARNING/ERROR/CRITICAL
        records go through append_error so they're visually distinct."""
        import logging as _logging
        if level >= _logging.WARNING:
            self.append_error(text)
        else:
            self.append_stdout(text)

    def append_error(self, tb: str) -> None:
        if not tb:
            return
        self.begin_topic(f"{self._active_app_label or 'Pipeline'} — ERROR")
        block = _StdoutBlock(tb, error=True)
        self._insert_entry(block)
        self._last_entry_kind = "stdout"

    def clear(self) -> None:
        # Remove every entry (but keep the trailing stretch)
        while self._entries.count() > 1:
            item = self._entries.takeAt(0)
            w = item.widget() if item else None
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._last_entry_kind = ""
        self._current_stdout = None
        self._ai_messages.clear()

    # ------------------------------------------------------------------
    # AI toggle + provider — external setters called by AppScreen.
    # ------------------------------------------------------------------
    def set_ai_active(self, on: bool) -> None:
        self._ai_active = bool(on)

    def set_ai_provider(self, provider_name: Optional[str]) -> None:
        self._current_provider_name = provider_name

    def _current_provider(self) -> Optional[ChatProvider]:
        if not self._current_provider_name:
            return None
        return ai_module.get_provider(self._current_provider_name)

    # ------------------------------------------------------------------
    # Submit — Enter in the input
    # ------------------------------------------------------------------
    def _on_submit(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            return
        self._input.clear()
        if self._ai_active:
            self._send_to_ai(text)
        else:
            # Local note — a green "user:" bubble, no AI reply.
            self._insert_entry(_Bubble("user", text))
            # Bubble breaks the current stdout block — next stdout
            # opens a fresh one below the bubble.
            self._current_stdout = None
            self._last_entry_kind = "bubble"

    def _send_to_ai(self, text: str) -> None:
        provider = self._current_provider()
        if provider is None:
            self.append_stdout(
                "[AI] No provider configured. Open Providers…\n"
            )
            return
        if self._ai_thread is not None:
            # Silent no-op: another stream is running. The AppScreen
            # actions row exposes the Cancel button, not us.
            return
        self._ai_messages.append({"role": "user", "content": text})
        # User message as a green bubble
        self._insert_entry(_Bubble("user", text))
        # Bubble breaks the current stdout block — force a fresh one
        # below the bubble for the AI's reply.
        self._current_stdout = None
        self._last_entry_kind = "bubble"
        self._ensure_stdout_block()
        self._current_stdout.append("spaCR AI: ")
        self._start_stream(system=ai_module.default_system_prompt())

    def _ensure_stdout_block(self) -> None:
        """Open a new plain stdout block if the last entry was not one."""
        if self._current_stdout is None or self._needs_topic("stdout"):
            block = _StdoutBlock()
            self._insert_entry(block)
            self._current_stdout = block
            self._last_entry_kind = "stdout"

    def _start_stream(self, system: str) -> None:
        provider = self._current_provider()
        if provider is None:
            return
        self._ai_buf = []
        # Parent the thread to this panel so its C++ lifetime is tied
        # to the panel, not to our Python refcount. Without this the
        # QThread can be GC'd between worker.run returning and
        # thread.finished firing → Qt aborts.
        thread, worker = make_stream_thread(
            provider, list(self._ai_messages), system=system,
            parent=self,
        )
        worker.stage_changed.connect(self._on_stage)
        worker.chunk_ready.connect(self._on_chunk)
        worker.finished.connect(self._on_stream_finished)
        self._ai_thread = thread
        self._ai_worker = worker
        thread.start()

    def cancel_ai(self) -> None:
        """Public — AppScreen calls this if the user cancels a stream."""
        if self._ai_worker is not None:
            self._ai_worker.cancel()

    def _prune_retired(self) -> None:
        """Drop entries whose QThread has already exited (isRunning
        returns False) OR whose C++ was already deleted by Qt's
        deferred-delete queue. Both are safe to forget."""
        alive = []
        for thread, worker in self._retired:
            try:
                if thread.isRunning():
                    alive.append((thread, worker))
            except RuntimeError:
                # C++ already deleted — safe to drop
                pass
        self._retired = alive

    def is_ai_streaming(self) -> bool:
        return self._ai_thread is not None

    def shutdown(self) -> None:
        """Cancel any active stream and block until its QThread has
        exited. Must be called before the panel (or its parent window)
        is destroyed — otherwise Python drops the last reference to
        the running QThread and Qt aborts with:
        `QThread: Destroyed while thread '' is still running`.

        The cancel path kills the CLI subprocess directly so the
        stream reader unblocks immediately; we then wait for the
        worker's run() to return and the QThread to quit normally.
        """
        worker = self._ai_worker
        thread = self._ai_thread
        # Defensively belt-and-suspender: also try every provider's
        # cancel_stream() in case the worker itself is somehow lost.
        try:
            for p in ai_module.list_providers():
                p.cancel_stream()
        except Exception:
            pass
        if worker is not None:
            try:
                worker.cancel()
            except Exception:
                pass
        if thread is not None and thread.isRunning():
            try:
                thread.quit()
                thread.wait(3000)
                if thread.isRunning():
                    # Last resort — Qt itself asks the thread to stop.
                    thread.terminate()
                    thread.wait(1000)
            except Exception:
                pass
        self._ai_thread = None
        self._ai_worker = None
        # Also drain any retired (post-finished) threads that haven't
        # been fully cleaned up yet.
        for t, _w in list(self._retired):
            try:
                if t.isRunning():
                    t.quit()
                    t.wait(1000)
                    if t.isRunning():
                        t.terminate()
                        t.wait(500)
            except Exception:
                pass
        self._retired.clear()

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)

    def _on_stage(self, _stage: str) -> None:
        # Could show a spinner; keeping this quiet for now.
        pass

    def _on_chunk(self, chunk: str) -> None:
        self._ai_buf.append(chunk)
        # AI reply flows into the same stdout block as pipeline output
        # (no separate section — user asked for this).
        self._ensure_stdout_block()
        self._current_stdout.append(chunk)
        self._scroll_to_bottom()

    def _on_stream_finished(self, ok: bool, final_text: str) -> None:
        # Retire the current (thread, worker) pair — hold both refs
        # in a list so Python can't GC the QThread before its OS
        # thread has fully exited AND Qt's deleteLater has run.
        # Prune already-dead entries on the way in so the list can't
        # grow unbounded across a long session.
        self._prune_retired()
        thread, worker = self._ai_thread, self._ai_worker
        self._ai_thread = None
        self._ai_worker = None
        if thread is not None:
            self._retired.append((thread, worker))
        if ok:
            self._ai_messages.append(
                {"role": "assistant", "content": final_text}
            )
            if not self._ai_buf:
                self.append_stdout(
                    "(empty response — try again or switch provider)\n"
                )
        else:
            self.append_stdout(f"[AI error] {final_text}\n")
        # Terminate the AI reply block with a newline so pipeline
        # stdout that arrives next visually separates from the reply.
        if self._current_stdout is not None:
            self._current_stdout.append("\n")
        self._ai_buf = []
        # Notify AppScreen so it can flip Cancel→AI on the toggle button.
        self.ai_stream_finished.emit()

    # ------------------------------------------------------------------
    # Public: Explain-error entry point (called from AppScreen)
    # ------------------------------------------------------------------
    def open_error_flow(self, traceback_text: str, active_app: str = "") -> None:
        from ..ai.prompts import wrap_error_for_prompt, error_explainer_prompt
        if self._current_provider() is None:
            self.append_stdout(
                "[AI] Enable AI in the actions row + pick a provider first.\n"
            )
            return
        prompt = wrap_error_for_prompt(
            traceback_text, active_app or self._active_app_label
        )
        self._ai_messages.append({"role": "user", "content": prompt})
        self._insert_entry(_Bubble("user", prompt))
        self._current_stdout = None
        self._last_entry_kind = "bubble"
        self._ensure_stdout_block()
        self._current_stdout.append("spaCR AI: ")
        self._start_stream(system=error_explainer_prompt())
