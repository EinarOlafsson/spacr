"""
AIChatPanel — reusable chat widget hosting the spacr AI Console.

Meant to be embedded next to the pipeline Console (via a QDockWidget on
the main window) so users never leave the app they're running. Exposes
`open_error_flow(traceback, active_app)` for AppScreen's "Explain
error" button.

Design notes
------------
* One instance is shared across the whole main window; the user's
  chat context persists as they switch between Mask/Measure/etc.
* Streaming is done in a QThread via `spacr.qt.ai.worker.StreamWorker`.
  The worker reference is stored on `self` — if we let Python drop it,
  Qt may never deliver the finished signal.
* There is NO "already streaming" guard. If a stream is in flight, the
  Send button turns into Cancel so the user can always recover.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import ai as ai_module
from .. import iconset
from ..ai import keys as ai_keys
from ..ai.providers import ChatProvider
from ..ai.worker import StreamWorker, make_stream_thread
from ..theme import PALETTE, SPACING
from .divider import Divider
from .empty_state import EmptyState


# ---------------------------------------------------------------------------
# Message bubble
# ---------------------------------------------------------------------------

class _MessageBubble(QWidget):
    def __init__(self, role: str, text: str = "", parent=None):
        super().__init__(parent)
        self.role = role
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING["sm"])
        self._text = QLabel(text)
        self._text.setWordWrap(True)
        self._text.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse
        )
        self._text.setOpenExternalLinks(True)
        self._text.setObjectName(
            "ChatBubbleUser" if role == "user" else "ChatBubbleAssistant"
        )
        self._text.setMaximumWidth(720)
        if role == "user":
            layout.addStretch(1)
            layout.addWidget(self._text)
        else:
            layout.addWidget(self._text)
            layout.addStretch(1)

    def set_text(self, text: str) -> None:
        self._text.setText(text)


# ---------------------------------------------------------------------------
# Manage keys dialog
# ---------------------------------------------------------------------------

class _KeysDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Console — provider keys")
        self.setMinimumWidth(520)
        outer = QVBoxLayout(self)

        intro = QLabel(
            "Set an API key for any provider you want to use. Keys are "
            "read from env vars first "
            "(<code>ANTHROPIC_API_KEY</code> / <code>OPENAI_API_KEY</code> "
            "/ <code>GOOGLE_API_KEY</code>), otherwise from your OS "
            "keyring under service <code>spacr-qt-ai</code>."
        )
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        outer.addWidget(intro)

        form = QFormLayout()
        self._inputs: Dict[str, QLineEdit] = {}
        for p in ai_module.list_providers():
            row = QHBoxLayout()
            edit = QLineEdit()
            edit.setEchoMode(QLineEdit.Password)
            edit.setPlaceholderText("<not set>")
            self._inputs[p.name] = edit
            row.addWidget(edit, 1)
            status = QLabel(self._status_text(p))
            status.setObjectName("SubtitleSmall")
            row.addWidget(status)
            wrap = QWidget(); wrap.setLayout(row)
            form.addRow(p.label, wrap)
        outer.addLayout(form)

        note = QLabel(
            "Blank field = keep the current key. Delete stored keys via "
            "your OS keyring app."
        )
        note.setObjectName("SubtitleSmall")
        note.setWordWrap(True)
        outer.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _status_text(self, provider: ChatProvider) -> str:
        parts = []
        parts.append("SDK OK" if provider.is_sdk_available()
                     else f"SDK missing — {provider.install_hint}")
        parts.append(f"key: {provider.source_of_key()}")
        return " · ".join(parts)

    def _save(self):
        for name, edit in self._inputs.items():
            text = edit.text().strip()
            if text:
                if not ai_keys.set_key(name, text):
                    QMessageBox.warning(
                        self, "Keyring unavailable",
                        f"Could not persist the {name} key to the OS "
                        f"keyring. Set the matching env var instead.",
                    )
        self.accept()


# ---------------------------------------------------------------------------
# Chat input — Enter = send, Shift+Enter = newline
# ---------------------------------------------------------------------------

class _ChatInput(QTextEdit):
    submitted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(56)
        self.setMaximumHeight(140)
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
# AIChatPanel
# ---------------------------------------------------------------------------

class AIChatPanel(QWidget):
    """Full chat panel — embed inside a QDockWidget or any container."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._messages: List[Dict] = []
        # Keep BOTH thread AND worker references — Qt's signal delivery
        # relies on the worker still being reachable.
        self._thread: Optional[QThread] = None
        self._worker: Optional[StreamWorker] = None
        self._pending_bubble: Optional[_MessageBubble] = None
        self._pending_buf: List[str] = []

        self._build_ui()
        self.refresh_provider_combo()

    # ------------------------------------------------------------------
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["md"], SPACING["md"],
                                  SPACING["md"], SPACING["md"])
        outer.setSpacing(SPACING["sm"])

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(SPACING["sm"])
        toolbar.addWidget(QLabel("Provider"))
        self._provider_combo = QComboBox()
        self._provider_combo.setMinimumWidth(160)
        toolbar.addWidget(self._provider_combo)
        self._btn_keys = QPushButton("Keys")
        self._btn_keys.setIcon(iconset.icon("settings"))
        self._btn_keys.setCursor(Qt.PointingHandCursor)
        self._btn_keys.clicked.connect(self._on_open_keys_dialog)
        toolbar.addWidget(self._btn_keys)
        toolbar.addStretch(1)
        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setObjectName("GhostButton")
        self._btn_clear.setIcon(iconset.icon("clear"))
        self._btn_clear.setCursor(Qt.PointingHandCursor)
        self._btn_clear.clicked.connect(self.clear_chat)
        toolbar.addWidget(self._btn_clear)
        tb_wrap = QWidget(); tb_wrap.setLayout(toolbar)
        outer.addWidget(tb_wrap)

        outer.addWidget(Divider())

        # Empty state ↔ chat stack
        self._empty_state = EmptyState(
            title="Configure a provider to chat",
            subtitle=(
                "Set an API key for Anthropic, OpenAI or Google — via env "
                "var or the Keys button — and this panel is where the "
                "SpaCR assistant lives. It also handles Explain-error "
                "requests from every pipeline app."
            ),
            icon=iconset.accent_icon("info"),
            cta_label="Keys…",
            on_action=self._on_open_keys_dialog,
        )
        self._chat_scroll = QScrollArea()
        self._chat_scroll.setWidgetResizable(True)
        self._chat_scroll.setFrameShape(QScrollArea.NoFrame)
        self._chat_holder = QWidget()
        self._chat_layout = QVBoxLayout(self._chat_holder)
        self._chat_layout.setContentsMargins(0, 0, 0, 0)
        self._chat_layout.setSpacing(SPACING["sm"])
        self._chat_layout.addStretch(1)
        self._chat_scroll.setWidget(self._chat_holder)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._empty_state)
        self._stack.addWidget(self._chat_scroll)
        outer.addWidget(self._stack, 1)

        # Input area
        input_row = QHBoxLayout()
        input_row.setSpacing(SPACING["sm"])
        self._input = _ChatInput()
        self._input.setPlaceholderText(
            "Ask a question (Enter to send · Shift+Enter for newline)"
        )
        self._input.submitted.connect(self._send_from_input)
        input_row.addWidget(self._input, 1)
        self._btn_send = QPushButton("Send")
        self._btn_send.setObjectName("PrimaryButton")
        self._btn_send.setIcon(iconset.contrast_icon("run"))
        self._btn_send.setCursor(Qt.PointingHandCursor)
        self._btn_send.clicked.connect(self._send_from_input)
        input_row.addWidget(self._btn_send)
        input_wrap = QWidget(); input_wrap.setLayout(input_row)
        outer.addWidget(input_wrap)

        # Status
        self._status = QLabel("")
        self._status.setObjectName("SubtitleSmall")
        outer.addWidget(self._status)

    # ------------------------------------------------------------------
    # Provider
    # ------------------------------------------------------------------
    def refresh_provider_combo(self):
        self._provider_combo.blockSignals(True)
        self._provider_combo.clear()
        configured = ai_module.configured_providers()
        for p in configured:
            self._provider_combo.addItem(p.label, userData=p.name)
        self._provider_combo.blockSignals(False)
        if configured:
            self._stack.setCurrentWidget(self._chat_scroll)
            self._input.setEnabled(True)
            self._set_send_mode("send")
        else:
            self._stack.setCurrentWidget(self._empty_state)
            self._input.setEnabled(False)
            self._set_send_mode("send")
            self._btn_send.setEnabled(False)

    def _current_provider(self) -> Optional[ChatProvider]:
        name = self._provider_combo.currentData()
        return ai_module.get_provider(name) if name else None

    def _on_open_keys_dialog(self):
        dlg = _KeysDialog(self)
        if dlg.exec() == QDialog.Accepted:
            self.refresh_provider_combo()

    # ------------------------------------------------------------------
    # Send / cancel
    # ------------------------------------------------------------------
    def _set_send_mode(self, mode: str):
        if mode == "cancel":
            self._btn_send.setText("Cancel")
            self._btn_send.setObjectName("DangerButton")
            try:
                self._btn_send.clicked.disconnect()
            except Exception:
                pass
            self._btn_send.clicked.connect(self._cancel_stream)
        else:
            self._btn_send.setText("Send")
            self._btn_send.setObjectName("PrimaryButton")
            try:
                self._btn_send.clicked.disconnect()
            except Exception:
                pass
            self._btn_send.clicked.connect(self._send_from_input)
        # Re-polish so QSS picks up the new objectName
        self._btn_send.style().unpolish(self._btn_send)
        self._btn_send.style().polish(self._btn_send)

    def _send_from_input(self):
        text = self._input.toPlainText().strip()
        if not text:
            return
        if self._thread is not None:
            self._status.setText("A response is already streaming — hit "
                                  "Cancel to interrupt.")
            return
        self._input.clear()
        self._append_user(text)
        self._start_stream(system=ai_module.default_system_prompt())

    def _cancel_stream(self):
        if self._worker is not None:
            self._worker.cancel()
            self._status.setText("Cancelling…")

    def _append_user(self, text: str):
        self._messages.append({"role": "user", "content": text})
        bubble = _MessageBubble("user", text)
        self._chat_layout.insertWidget(self._chat_layout.count() - 1, bubble)
        self._scroll_to_bottom()

    def _start_stream(self, system: str):
        provider = self._current_provider()
        if provider is None:
            self._status.setText("No provider configured.")
            return
        self._pending_buf = []
        self._pending_bubble = _MessageBubble("assistant", "…")
        self._chat_layout.insertWidget(self._chat_layout.count() - 1,
                                        self._pending_bubble)
        self._scroll_to_bottom()

        thread, worker = make_stream_thread(
            provider, list(self._messages), system=system
        )
        worker.stage_changed.connect(self._on_stage_changed)
        worker.chunk_ready.connect(self._on_chunk)
        worker.finished.connect(self._on_stream_finished)
        # Hold references so nothing is GC'd mid-flight.
        self._thread = thread
        self._worker = worker
        self._set_send_mode("cancel")
        self._status.setText(f"Connecting to {provider.label}…")
        thread.start()

    def _on_stage_changed(self, stage: str):
        provider = self._current_provider()
        label = provider.label if provider else ""
        if stage == "connecting":
            self._status.setText(f"Connecting to {label}…")
        elif stage == "streaming":
            self._status.setText(f"Streaming from {label}…")

    def _on_chunk(self, text: str):
        self._pending_buf.append(text)
        if self._pending_bubble is not None:
            self._pending_bubble.set_text("".join(self._pending_buf))
            self._scroll_to_bottom()

    def _on_stream_finished(self, ok: bool, final_text: str):
        # Reset streaming state FIRST so a fast follow-up send works.
        self._thread = None
        self._worker = None
        self._set_send_mode("send")
        if ok and self._pending_bubble is not None:
            self._messages.append({"role": "assistant", "content": final_text})
            if not self._pending_buf:
                # Provider returned no chunks — surface an obvious message
                self._pending_bubble.set_text(
                    "(empty response — try again or switch provider)"
                )
            self._status.setText("Ready.")
        else:
            if self._pending_bubble is not None:
                self._pending_bubble.set_text(f"[error] {final_text}")
            self._status.setText(f"Failed: {final_text}")
        self._pending_bubble = None
        self._pending_buf = []

    def _scroll_to_bottom(self):
        sb = self._chat_scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear_chat(self):
        self._messages.clear()
        while self._chat_layout.count() > 1:
            item = self._chat_layout.takeAt(0)
            w = item.widget() if item else None
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    # ------------------------------------------------------------------
    # Public API used by AppScreen's Explain-error
    # ------------------------------------------------------------------
    def open_error_flow(self, traceback_text: str, active_app: str = "") -> None:
        from ..ai.prompts import wrap_error_for_prompt, error_explainer_prompt
        provider = self._current_provider()
        if provider is None:
            self._status.setText("Configure a provider first (Keys…).")
            return
        prompt = wrap_error_for_prompt(traceback_text, active_app)
        self._append_user(prompt)
        self._start_stream(system=error_explainer_prompt())
