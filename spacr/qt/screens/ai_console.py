"""
AIConsoleScreen — chat panel that talks to Anthropic Claude, OpenAI, or
Google Gemini via the shared `spacr.qt.ai` abstraction.

Also exposes:
    open_error_flow(traceback_text, active_app)
        prefills the input with an "Explain error" request and sends it
        immediately, so AppScreen can wire an "Explain error" button
        that lands here.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QKeyEvent, QTextCursor
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import ai as ai_module
from .. import iconset
from ..ai import keys as ai_keys
from ..ai.providers import ChatProvider
from ..ai.worker import make_stream_thread
from ..theme import PALETTE, SPACING
from ..widgets import Divider, EmptyState


# ---------------------------------------------------------------------------
# Message bubble
# ---------------------------------------------------------------------------

class _MessageBubble(QWidget):
    def __init__(self, role: str, text: str = "", parent=None):
        super().__init__(parent)
        self.role = role
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(SPACING["sm"])

        self._text = QLabel(text)
        self._text.setWordWrap(True)
        self._text.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        self._text.setOpenExternalLinks(True)
        self._text.setObjectName(
            "ChatBubbleUser" if role == "user" else "ChatBubbleAssistant"
        )
        self._text.setMaximumWidth(760)
        if role == "user":
            self._layout.addStretch(1)
            self._layout.addWidget(self._text)
        else:
            self._layout.addWidget(self._text)
            self._layout.addStretch(1)

    def append_chunk(self, chunk: str) -> None:
        self._text.setText(self._text.text() + chunk)

    def set_text(self, text: str) -> None:
        self._text.setText(text)


# ---------------------------------------------------------------------------
# Settings dialog — add / remove API keys
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
            "(<code>ANTHROPIC_API_KEY</code> / <code>OPENAI_API_KEY</code> / "
            "<code>GOOGLE_API_KEY</code>), otherwise fetched from your "
            "OS keyring under service <code>spacr-qt-ai</code>."
        )
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        outer.addWidget(intro)

        form = QFormLayout()
        self._inputs: Dict[str, QLineEdit] = {}
        self._status: Dict[str, QLabel] = {}
        for p in ai_module.list_providers():
            row = QHBoxLayout()
            edit = QLineEdit()
            edit.setEchoMode(QLineEdit.Password)
            edit.setPlaceholderText("<not set>")
            self._inputs[p.name] = edit
            row.addWidget(edit, 1)
            status = QLabel(self._status_text(p))
            status.setObjectName("SubtitleSmall")
            self._status[p.name] = status
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
        if provider.is_sdk_available():
            parts.append("SDK OK")
        else:
            parts.append(f"SDK missing — {provider.install_hint}")
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
# The screen
# ---------------------------------------------------------------------------

class AIConsoleScreen(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._messages: List[Dict] = []
        self._thread: Optional[QThread] = None
        self._pending_bubble: Optional[_MessageBubble] = None
        self._pending_buf: List[str] = []

        self._build_ui()
        self._refresh_provider_combo()

    # ------------------------------------------------------------------
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                  SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        # Header
        header_col = QVBoxLayout()
        header_col.setContentsMargins(0, 0, 0, 0)
        header_col.setSpacing(4)
        title = QLabel("AI Console")
        title.setObjectName("TitleHeading")
        header_col.addWidget(title)
        self._src_label = QLabel(
            "Ask questions about SpaCR, or paste a traceback for a fix."
        )
        self._src_label.setObjectName("SubtitleSmall")
        header_col.addWidget(self._src_label)
        header_wrap = QWidget(); header_wrap.setLayout(header_col)
        outer.addWidget(header_wrap)
        outer.addWidget(Divider())

        # Toolbar: provider selector + keys button + clear
        toolbar = QHBoxLayout()
        toolbar.setSpacing(SPACING["sm"])
        toolbar.addWidget(QLabel("Provider"))
        self._provider_combo = QComboBox()
        self._provider_combo.setMinimumWidth(220)
        toolbar.addWidget(self._provider_combo)
        self._btn_keys = QPushButton("Manage keys…")
        self._btn_keys.setIcon(iconset.icon("settings"))
        self._btn_keys.setCursor(Qt.PointingHandCursor)
        self._btn_keys.clicked.connect(self._on_open_keys_dialog)
        toolbar.addWidget(self._btn_keys)
        toolbar.addStretch(1)
        self._btn_clear = QPushButton("Clear chat")
        self._btn_clear.setObjectName("GhostButton")
        self._btn_clear.setIcon(iconset.icon("clear"))
        self._btn_clear.setCursor(Qt.PointingHandCursor)
        self._btn_clear.clicked.connect(self._clear_chat)
        toolbar.addWidget(self._btn_clear)
        tb_wrap = QWidget(); tb_wrap.setLayout(toolbar)
        outer.addWidget(tb_wrap)

        # Body: scrolling message list (or empty state)
        self._empty_state = EmptyState(
            title="No provider configured yet",
            subtitle=(
                "Set an API key for Anthropic, OpenAI, or Google — either "
                "in an env var or via the Manage keys… button — and this "
                "panel becomes an in-app chat assistant. It also handles "
                "the Explain-error button on each pipeline screen."
            ),
            icon=iconset.accent_icon("info"),
            cta_label="Manage keys…",
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

        # Stack via a simple QWidget swap
        from PySide6.QtWidgets import QStackedWidget
        self._stack = QStackedWidget()
        self._stack.addWidget(self._empty_state)
        self._stack.addWidget(self._chat_scroll)
        outer.addWidget(self._stack, 1)

        # Input area
        input_row = QHBoxLayout()
        input_row.setSpacing(SPACING["sm"])
        self._input = _ChatInput()
        self._input.setPlaceholderText(
            "Ask a question about SpaCR (Enter to send · Shift+Enter for newline)"
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

        # Status label
        self._status = QLabel("")
        self._status.setObjectName("SubtitleSmall")
        outer.addWidget(self._status)

    # ------------------------------------------------------------------
    # Provider combo
    # ------------------------------------------------------------------
    def _refresh_provider_combo(self):
        self._provider_combo.blockSignals(True)
        self._provider_combo.clear()
        configured = ai_module.configured_providers()
        for p in configured:
            self._provider_combo.addItem(p.label, userData=p.name)
        self._provider_combo.blockSignals(False)
        if configured:
            self._stack.setCurrentWidget(self._chat_scroll)
            self._btn_send.setEnabled(True)
            self._input.setEnabled(True)
        else:
            self._stack.setCurrentWidget(self._empty_state)
            self._btn_send.setEnabled(False)
            self._input.setEnabled(False)

    def _current_provider(self) -> Optional[ChatProvider]:
        name = self._provider_combo.currentData()
        if not name:
            return None
        return ai_module.get_provider(name)

    # ------------------------------------------------------------------
    # Keys dialog
    # ------------------------------------------------------------------
    def _on_open_keys_dialog(self):
        dlg = _KeysDialog(self)
        if dlg.exec() == QDialog.Accepted:
            self._refresh_provider_combo()

    # ------------------------------------------------------------------
    # Chat send / stream
    # ------------------------------------------------------------------
    def _send_from_input(self):
        text = self._input.toPlainText().strip()
        if not text:
            return
        self._input.clear()
        self._append_user(text)
        self._start_stream()

    def _append_user(self, text: str):
        self._messages.append({"role": "user", "content": text})
        bubble = _MessageBubble("user", text)
        self._chat_layout.insertWidget(self._chat_layout.count() - 1, bubble)
        self._scroll_to_bottom()

    def _start_stream(self):
        if self._thread is not None:
            self._status.setText("Already streaming — wait for the previous "
                                  "response to finish.")
            return
        provider = self._current_provider()
        if provider is None:
            self._status.setText("No provider configured.")
            return
        # Assistant placeholder bubble that we grow as chunks arrive
        self._pending_buf = []
        self._pending_bubble = _MessageBubble("assistant", "")
        self._chat_layout.insertWidget(self._chat_layout.count() - 1,
                                        self._pending_bubble)
        self._scroll_to_bottom()

        system = ai_module.default_system_prompt()
        thread, worker = make_stream_thread(provider, list(self._messages),
                                              system=system)
        worker.chunk_ready.connect(self._on_chunk)
        worker.finished.connect(self._on_stream_finished)
        self._thread = thread
        self._btn_send.setEnabled(False)
        self._status.setText(f"Streaming from {provider.label}…")
        thread.start()

    def _on_chunk(self, text: str):
        self._pending_buf.append(text)
        if self._pending_bubble is not None:
            self._pending_bubble.set_text("".join(self._pending_buf))
            self._scroll_to_bottom()

    def _on_stream_finished(self, ok: bool, final_text: str):
        self._btn_send.setEnabled(True)
        self._thread = None
        if ok:
            self._messages.append(
                {"role": "assistant", "content": final_text}
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

    def _clear_chat(self):
        self._messages.clear()
        while self._chat_layout.count() > 1:
            item = self._chat_layout.takeAt(0)
            w = item.widget() if item else None
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    # ------------------------------------------------------------------
    # Public: opened from the AppScreen error-explainer button
    # ------------------------------------------------------------------
    def open_error_flow(self, traceback_text: str, active_app: str = "") -> None:
        """Send an 'Explain error' request straight away."""
        from ..ai.prompts import wrap_error_for_prompt, error_explainer_prompt
        provider = self._current_provider()
        if provider is None:
            self._status.setText(
                "Configure a provider first (Manage keys…)."
            )
            return
        prompt = wrap_error_for_prompt(traceback_text, active_app)
        self._append_user(prompt)
        # For error explanation we use the tighter explainer system prompt
        self._pending_buf = []
        self._pending_bubble = _MessageBubble("assistant", "")
        self._chat_layout.insertWidget(self._chat_layout.count() - 1,
                                        self._pending_bubble)
        thread, worker = make_stream_thread(provider, list(self._messages),
                                              system=error_explainer_prompt())
        worker.chunk_ready.connect(self._on_chunk)
        worker.finished.connect(self._on_stream_finished)
        self._thread = thread
        self._btn_send.setEnabled(False)
        self._status.setText(f"Explaining error via {provider.label}…")
        thread.start()


# ---------------------------------------------------------------------------
# Chat input — Enter = send, Shift+Enter = newline
# ---------------------------------------------------------------------------

class _ChatInput(QTextEdit):
    submitted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(64)
        self.setMaximumHeight(160)
        self.setAcceptRichText(False)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
                return
            self.submitted.emit()
            return
        super().keyPressEvent(event)
