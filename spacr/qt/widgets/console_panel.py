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
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
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
        self._label = QLabel(self)
        self._label.setObjectName("ConsoleBubbleText")
        self._label.setTextFormat(Qt.RichText)
        self._label.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse
        )
        self._label.setOpenExternalLinks(True)
        self._label.setWordWrap(True)
        self._label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
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
        self._prefix = "user: " if role == "user" else "spaCR AI: "
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
        w = self.width()
        if w <= 0:
            return
        text_width = max(120, w - self._H_PAD)
        self._label.setFixedWidth(text_width)
        h = self._label.heightForWidth(text_width)
        if h <= 0:
            h = self._label.sizeHint().height()
        self._label.setFixedHeight(h)
        self.setFixedHeight(h + self._V_PAD)

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
    def __init__(self, active_app_label: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("ConsolePanel")
        self._active_app_label = active_app_label or ""
        self._last_entry_kind: str = ""   # "stdout" | "ai" | ""
        self._current_stdout: Optional[_StdoutBlock] = None
        self._current_ai_bubble: Optional[_Bubble] = None
        self._ai_messages: List[Dict] = []
        self._ai_buf: List[str] = []
        self._ai_thread: Optional[QThread] = None
        self._ai_worker: Optional[StreamWorker] = None

        self._build_ui()
        self._refresh_provider_combo()

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
        self._holder = QWidget()
        self._holder.setObjectName("ConsoleHolder")
        self._entries = QVBoxLayout(self._holder)
        self._entries.setContentsMargins(0, 0, 0, 0)
        self._entries.setSpacing(0)
        self._entries.addStretch(1)
        self._scroll.setWidget(self._holder)
        outer.addWidget(self._scroll, 1)

        # Input row
        input_bar = QFrame()
        input_bar.setObjectName("ConsoleInputBar")
        row = QHBoxLayout(input_bar)
        row.setContentsMargins(SPACING["md"], SPACING["sm"],
                                SPACING["md"], SPACING["sm"])
        row.setSpacing(SPACING["sm"])

        # AI toggle
        self._ai_toggle = QCheckBox("Ask AI")
        self._ai_toggle.setCursor(Qt.PointingHandCursor)
        self._ai_toggle.setToolTip(
            "When on, Enter sends your message to the selected AI "
            "provider and appends its reply in a spaCR AI section."
        )
        self._ai_toggle.toggled.connect(self._on_ai_toggle)
        row.addWidget(self._ai_toggle)

        # Provider selector (only visible when a CLI is configured)
        self._provider_combo = QComboBox()
        self._provider_combo.setMinimumWidth(160)
        self._provider_combo.setToolTip(
            "Which vendor coding-agent CLI to use for the AI."
        )
        row.addWidget(self._provider_combo)

        # Providers button (jumps to install/login dialog)
        self._btn_providers = QPushButton("Providers")
        self._btn_providers.setIcon(iconset.icon("settings"))
        self._btn_providers.setCursor(Qt.PointingHandCursor)
        self._btn_providers.setToolTip(
            "Install + login instructions for the vendor CLIs."
        )
        self._btn_providers.clicked.connect(self._on_open_providers_dialog)
        row.addWidget(self._btn_providers)

        # Input
        self._input = _ChatInput()
        self._input.setPlaceholderText(
            "Type a message… (Ask AI to route it through your chat "
            "subscription, or leave off to jot notes)"
        )
        self._input.submitted.connect(self._on_submit)
        row.addWidget(self._input, 1)

        # Send / Cancel button
        self._btn_send = QPushButton("Send")
        self._btn_send.setObjectName("PrimaryButton")
        self._btn_send.setIcon(iconset.contrast_icon("run"))
        self._btn_send.setCursor(Qt.PointingHandCursor)
        self._btn_send.clicked.connect(self._on_submit)
        row.addWidget(self._btn_send)

        outer.addWidget(input_bar)

    # ------------------------------------------------------------------
    # Entry-management helpers
    # ------------------------------------------------------------------
    def _insert_entry(self, w: QWidget) -> None:
        # Bubbles get a horizontal offset row so user (green) hugs right
        # and spaCR AI (blue) hugs left. Everything else spans full width.
        if isinstance(w, _Bubble):
            row = QWidget()
            sp = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
            sp.setHeightForWidth(True)
            row.setSizePolicy(sp)
            row_lay = QHBoxLayout(row)
            row_lay.setContentsMargins(SPACING["md"], SPACING["xs"],
                                        SPACING["md"], SPACING["xs"])
            row_lay.setSpacing(0)
            if w.role == "user":
                row_lay.addSpacing(SPACING["xxl"] * 2)
                row_lay.addWidget(w, 1)
            else:
                row_lay.addWidget(w, 1)
                row_lay.addSpacing(SPACING["xxl"] * 2)
            self._entries.insertWidget(self._entries.count() - 1, row)
        else:
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
        """Insert a divider bar labeled `label` (e.g. 'Mask' or
        'spaCR AI'). Callers can force a new section this way."""
        self._insert_entry(_TopicBar(label))
        self._last_entry_kind = ""    # force next append to open a block
        self._current_stdout = None
        self._current_ai_bubble = None

    def append_stdout(self, text: str) -> None:
        """Append pipeline output. Opens a new stdout block preceded by
        an app-name divider if the previous entry wasn't stdout."""
        if not text:
            return
        if self._needs_topic("stdout") or self._current_stdout is None:
            self.begin_topic(self._active_app_label or "Pipeline")
            self._current_stdout = _StdoutBlock()
            self._insert_entry(self._current_stdout)
            self._last_entry_kind = "stdout"
        self._current_stdout.append(text)
        self._scroll_to_bottom()

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
        self._current_ai_bubble = None
        self._ai_messages.clear()

    # ------------------------------------------------------------------
    # Provider / AI
    # ------------------------------------------------------------------
    def _refresh_provider_combo(self) -> None:
        self._provider_combo.blockSignals(True)
        self._provider_combo.clear()
        configured = ai_module.configured_providers()
        for p in configured:
            self._provider_combo.addItem(p.label, userData=p.name)
        self._provider_combo.blockSignals(False)
        self._ai_toggle.setEnabled(bool(configured))
        if not configured:
            self._ai_toggle.setChecked(False)
            self._ai_toggle.setToolTip(
                "No vendor CLI installed yet — open Providers… to "
                "install `claude`, `codex`, or `gemini`."
            )

    def _current_provider(self) -> Optional[ChatProvider]:
        name = self._provider_combo.currentData()
        return ai_module.get_provider(name) if name else None

    def _on_ai_toggle(self, on: bool) -> None:
        # When the user turns AI on we already show the provider combo;
        # nothing else to do here — the actual routing happens in
        # _on_submit.
        if on and not self._current_provider():
            self._on_open_providers_dialog()

    def _on_open_providers_dialog(self) -> None:
        from .ai_chat_panel import _ProvidersDialog
        dlg = _ProvidersDialog(self)
        if dlg.exec() == QDialog.Accepted:
            self._refresh_provider_combo()

    # ------------------------------------------------------------------
    # Send / cancel
    # ------------------------------------------------------------------
    def _set_send_mode(self, mode: str) -> None:
        if mode == "cancel":
            self._btn_send.setText("Cancel")
            self._btn_send.setObjectName("DangerButton")
        else:
            self._btn_send.setText("Send")
            self._btn_send.setObjectName("PrimaryButton")
        try:
            self._btn_send.clicked.disconnect()
        except Exception:
            pass
        if mode == "cancel":
            self._btn_send.clicked.connect(self._cancel_ai)
        else:
            self._btn_send.clicked.connect(self._on_submit)
        self._btn_send.style().unpolish(self._btn_send)
        self._btn_send.style().polish(self._btn_send)

    def _on_submit(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            return
        self._input.clear()
        if self._ai_toggle.isChecked():
            self._send_to_ai(text)
        else:
            # Local note echo — same "user:" bubble but no AI reply
            if self._needs_topic("ai"):
                self.begin_topic("Notes")
            bubble = _Bubble("user", text)
            self._insert_entry(bubble)
            self._last_entry_kind = "ai"
            self._current_ai_bubble = None

    def _send_to_ai(self, text: str) -> None:
        provider = self._current_provider()
        if provider is None:
            self.append_stdout(
                "[AI] No provider configured. Open Providers…\n"
            )
            return
        if self._ai_thread is not None:
            self.append_stdout(
                "[AI] A stream is already running — hit Cancel first.\n"
            )
            return
        if self._needs_topic("ai"):
            self.begin_topic("spaCR AI")
        self._ai_messages.append({"role": "user", "content": text})
        self._insert_entry(_Bubble("user", text))
        self._current_ai_bubble = _Bubble("assistant", "…")
        self._insert_entry(self._current_ai_bubble)
        self._last_entry_kind = "ai"
        self._start_stream(system=ai_module.default_system_prompt())

    def _start_stream(self, system: str) -> None:
        provider = self._current_provider()
        if provider is None:
            return
        self._ai_buf = []
        thread, worker = make_stream_thread(
            provider, list(self._ai_messages), system=system,
        )
        worker.stage_changed.connect(self._on_stage)
        worker.chunk_ready.connect(self._on_chunk)
        worker.finished.connect(self._on_stream_finished)
        self._ai_thread = thread
        self._ai_worker = worker
        self._set_send_mode("cancel")
        thread.start()

    def _cancel_ai(self) -> None:
        if self._ai_worker is not None:
            self._ai_worker.cancel()

    def _on_stage(self, _stage: str) -> None:
        # Could show a spinner; keeping this quiet for now.
        pass

    def _on_chunk(self, chunk: str) -> None:
        self._ai_buf.append(chunk)
        if self._current_ai_bubble is not None:
            self._current_ai_bubble.set_text("".join(self._ai_buf))
            self._scroll_to_bottom()

    def _on_stream_finished(self, ok: bool, final_text: str) -> None:
        self._ai_thread = None
        self._ai_worker = None
        self._set_send_mode("send")
        if ok:
            self._ai_messages.append(
                {"role": "assistant", "content": final_text}
            )
            if self._current_ai_bubble is not None and not self._ai_buf:
                self._current_ai_bubble.set_text(
                    "(empty response — try again or switch provider)"
                )
        else:
            if self._current_ai_bubble is not None:
                self._current_ai_bubble.set_text(f"[error] {final_text}")
        self._current_ai_bubble = None
        self._ai_buf = []

    # ------------------------------------------------------------------
    # Public: Explain-error entry point (called from AppScreen)
    # ------------------------------------------------------------------
    def open_error_flow(self, traceback_text: str, active_app: str = "") -> None:
        from ..ai.prompts import wrap_error_for_prompt, error_explainer_prompt
        if not ai_module.configured_providers():
            self._on_open_providers_dialog()
            if not ai_module.configured_providers():
                return
            self._refresh_provider_combo()
        self._ai_toggle.setChecked(True)
        prompt = wrap_error_for_prompt(traceback_text, active_app or self._active_app_label)
        self.begin_topic("spaCR AI — Explain error")
        self._ai_messages.append({"role": "user", "content": prompt})
        self._insert_entry(_Bubble("user", prompt))
        self._current_ai_bubble = _Bubble("assistant", "…")
        self._insert_entry(self._current_ai_bubble)
        self._last_entry_kind = "ai"
        self._start_stream(system=error_explainer_prompt())
