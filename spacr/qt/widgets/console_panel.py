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

Threading
---------
Every entry in this console is a QWidget, so every method that appends
one has to run on the GUI thread. Log records do not: Python's logging
module calls handlers inline on whatever thread logged, and a pipeline
worker logging a warning used to reach :meth:`ConsolePanel.append_stdout`
directly. :meth:`ConsolePanel.append_stdout` and
:meth:`ConsolePanel.append_error` therefore bounce off-thread calls back
through a queued signal instead of building the widget where they stand.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import QSize, Qt, QThread, Signal
from PySide6.QtGui import (
    QFont,
    QFontDatabase,
    QKeyEvent,
    QTextBlockFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import ai as ai_module
from ..ai import settings as ai_settings
from ..ai.providers import ChatProvider
from ..ai.worker import StreamWorker, make_stream_thread
from ..i18n import retranslate_widget_tree, tr
from ..theme import FONT_SIZE, SPACING, active_palette


# ---------------------------------------------------------------------------
# Console text colours (per the output type) — we colour the *text*, not the
# background, so there are no coloured boxes.
#
# Resolved through `active_palette()` on every call rather than captured
# at import time. They used to be three module-level constants read off
# `theme.PALETTE`, which is the frozen DARK palette, so the console
# painted the same dark chrome on every theme. Measured on light:
# `_StdoutBlock` filled itself `#161719` inside a `#fafafa` page (a black
# rectangle in a white one), and `_Bubble` inked `#ffffff` text on the
# `#dbe8fb` bubble the app stylesheet paints — 1.24:1. Now 15.59:1.
#
# `COLOR_OUTPUT` / `COLOR_USER` / `COLOR_ERROR` are still importable —
# module __getattr__ below serves them live — because they read well at
# the call sites and existing callers spell them that way.
# ---------------------------------------------------------------------------

#: Palette role behind each of the three legacy ``COLOR_*`` names.
_TEXT_ROLES = {
    "COLOR_OUTPUT": "accent",     # spaCR output  → blue
    "COLOR_USER":   "success",    # user input    → green
    "COLOR_ERROR":  "error",      # errors        → red
}


def color_output() -> str:
    """Pipeline stdout colour for the theme on screen right now."""
    return active_palette()["accent"]


def color_user() -> str:
    """User-input colour for the theme on screen right now."""
    return active_palette()["success"]


def color_error() -> str:
    """Error colour for the theme on screen right now."""
    return active_palette()["error"]


def __getattr__(name: str) -> str:
    """Serve ``COLOR_OUTPUT`` / ``COLOR_USER`` / ``COLOR_ERROR`` live.

    PEP 562. Reading one of the three resolves it against the current
    theme, so ``from ...console_panel import COLOR_USER`` can no longer
    freeze a dark-theme hex into a caller at import time.
    """
    role = _TEXT_ROLES.get(name)
    if role is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return active_palette()[role]


# spaCR AI text colour depends on the backing provider.
AI_COLOR_CLAUDE = "#DE7356"         # Anthropic terracotta / peach
AI_COLOR_OPENAI = "#74AA9C"         # OpenAI signature green
AI_COLOR_GEMINI = "#74AA9C"         # Gemini — same green as requested
AI_COLOR_DEFAULT = "#74AA9C"


def ai_color_for_provider(provider_name: Optional[str]) -> str:
    """Return the spaCR-AI text colour for a provider name."""
    p = (provider_name or "").lower()
    if "claude" in p or "anthropic" in p:
        return AI_COLOR_CLAUDE
    if "gpt" in p or "openai" in p or "chatgpt" in p:
        return AI_COLOR_OPENAI
    if "gemini" in p or "google" in p:
        return AI_COLOR_GEMINI
    return AI_COLOR_DEFAULT


# ---------------------------------------------------------------------------
# Divider bar with a topic label
# ---------------------------------------------------------------------------

class _TopicBar(QFrame):
    """Dark-gray divider bar with a topic label ("spaCR output — …", …).

    An optional ``accent`` colour tints the label so each banner reads in
    the same colour as the text that follows it. A trailing ``widget`` (e.g.
    an animated working indicator) can be pinned to the right.
    """

    def __init__(self, label: str, parent=None, accent: Optional[str] = None,
                 trailing: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ConsoleTopicBar")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(SPACING["md"], SPACING["xs"],
                                SPACING["md"], SPACING["xs"])
        self._label = QLabel(label)
        self._label.setObjectName("ConsoleTopicLabel")
        # Topic history is presentation generated in the language active when
        # it was appended. Do not reinterpret composite module/function text
        # during a later whole-window language switch.
        self._label.setProperty("i18nSkipText", True)
        if accent:
            self._label.setStyleSheet(
                f"QLabel#ConsoleTopicLabel {{ color: {accent}; "
                "background: transparent; }")
        lay.addWidget(self._label)
        if trailing is not None:
            lay.addWidget(trailing)
        lay.addStretch(1)


# ---------------------------------------------------------------------------
# Animated "working" indicator — three dots cycling in AI colour
# ---------------------------------------------------------------------------

class _WorkingDots(QLabel):
    """Three dots that cycle (. → .. → ...) to show work is in progress."""

    def __init__(self, color: str = AI_COLOR_DEFAULT, parent=None):
        super().__init__(parent)
        self.setObjectName("ConsoleWorkingDots")
        self.setProperty("i18nSkipText", True)
        self._color = color
        self._n = 0
        self.setStyleSheet(
            f"QLabel#ConsoleWorkingDots {{ color: {color}; "
            f"font-size: {max(6, FONT_SIZE['xs'] - 2)}px; font-weight: 400; "
            "background: transparent; }")
        from PySide6.QtCore import QTimer
        self._timer = QTimer(self)
        self._timer.setInterval(350)
        self._timer.timeout.connect(self._tick)
        self._render()

    def set_color(self, color: str) -> None:
        self._color = color
        self.setStyleSheet(
            f"QLabel#ConsoleWorkingDots {{ color: {color}; "
            f"font-size: {max(6, FONT_SIZE['xs'] - 2)}px; font-weight: 400; "
            "background: transparent; }")

    def _render(self) -> None:
        # Fixed-width so the row doesn't jitter as the count changes.
        dots = "●" * (self._n + 1)
        pad = " " * (2 - self._n)   # keep three glyph-slots wide
        self.setText(dots + pad)

    def _tick(self) -> None:
        self._n = (self._n + 1) % 3
        self._render()

    def start(self) -> None:
        self._n = 0
        self._render()
        self._timer.start()
        self.show()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()


# ---------------------------------------------------------------------------
# Stdout block (grows in place while pipeline is running)
# ---------------------------------------------------------------------------

class _StdoutBlock(QPlainTextEdit):
    """Readable text block that grows in place as pipeline output arrives.

    A single block is reused for a whole stdout run so line breaks do
    not fragment the console into one widget per line.  A read-only
    ``QPlainTextEdit`` gives us selectable plain text while also exposing
    ``QTextBlockFormat``—the reliable Qt API for real line spacing.  QSS
    does not implement CSS ``line-height`` for a ``QLabel``.
    """

    LINE_HEIGHT_PERCENT = 145

    def __init__(self, text: str = "", error: bool = False, parent=None,
                 text_color: Optional[str] = None):
        super().__init__(parent)
        self.setObjectName("ConsoleStdoutBlockError"
                            if error else "ConsoleStdoutBlock")
        self.setReadOnly(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._font_pt = 10
        self._font = QFont("Open Sans", self._font_pt, QFont.Light)
        self._font.setStyleName("Light")
        self.setFont(self._font)
        self.document().setDefaultFont(self._font)
        self.document().setDocumentMargin(0)
        # Colour the TEXT (not a coloured box): each output type gets its own
        # foreground colour while the block background stays neutral.
        if text_color is None:
            text_color = color_error() if error else color_output()
        self._text_color = text_color
        self._refresh_style()
        self._buf: List[str] = []
        self._user_height: Optional[int] = None
        self._height_handle = _BlockHeightHandle(self)
        self._height_handle.show()
        if text:
            self.append(text)

    def _refresh_style(self) -> None:
        """Keep inline, theme-aware ink and the user-selected point size."""
        self.setStyleSheet(
            "QPlainTextEdit#%s { color: %s; background-color: transparent; "
            "border: none; "
            "font-family: 'Open Sans','Segoe UI','Helvetica Neue',sans-serif; "
            "font-weight: 300; font-size: %dpt; "
            "padding: %dpx %dpx; }" % (
                self.objectName(), self._text_color, self._font_pt,
                SPACING["sm"], SPACING["md"]))

    def _apply_line_spacing(self) -> None:
        """Apply 145% leading to every paragraph in the plain-text document."""
        cursor = QTextCursor(self.document())
        cursor.select(QTextCursor.Document)
        block_format = QTextBlockFormat()
        block_format.setLineHeight(
            float(self.LINE_HEIGHT_PERCENT),
            QTextBlockFormat.ProportionalHeight.value,
        )
        cursor.mergeBlockFormat(block_format)

    def set_console_font_pt(self, pt: int) -> None:
        """Apply the console size while retaining Open Sans Light."""
        self._font_pt = int(pt)
        self._font.setPointSize(self._font_pt)
        self._font.setWeight(QFont.Light)
        self._font.setStyleName("Light")
        self.setFont(self._font)
        self.document().setDefaultFont(self._font)
        self._refresh_style()
        self._apply_line_spacing()
        self.updateGeometry()

    def text(self) -> str:
        """Compatibility with the former QLabel-backed output block."""
        return self.toPlainText()

    def append(self, text: str) -> None:
        """Append ``text`` to the block, capping the buffer at 200k chars."""
        self._buf.append(text)
        # Cap the buffer to keep the UI snappy for very long runs.
        joined = "".join(self._buf)
        if len(joined) > 200_000:
            joined = joined[-200_000:]
            self._buf = [joined]
        self.setPlainText(joined)
        self._apply_line_spacing()
        self.updateGeometry()

    def sizeHint(self) -> QSize:
        """Report the full document height; the outer console owns scrolling."""
        if self._user_height is not None:
            return QSize(
                max(120, super().sizeHint().width()), self._user_height)
        layout = self.document().documentLayout()
        height = 0.0
        block = self.document().begin()
        while block.isValid():
            height += layout.blockBoundingRect(block).height()
            block = block.next()
        chrome = (2 * SPACING["sm"]) + 2
        return QSize(max(120, super().sizeHint().width()),
                     max(32, int(round(height)) + chrome))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.document().setTextWidth(max(1, self.viewport().width()))
        handle_height = self._height_handle.sizeHint().height()
        self._height_handle.setGeometry(
            0, max(0, self.height() - handle_height),
            self.width(), handle_height,
        )
        self._height_handle.raise_()
        self.updateGeometry()

    def set_user_height(self, height: int) -> None:
        """Pin this section to a user-selected height."""
        self._user_height = max(48, min(4000, int(height)))
        self.setFixedHeight(self._user_height)
        self.updateGeometry()

    def reset_user_height(self) -> None:
        """Return to automatic document-height sizing."""
        self._user_height = None
        self.setMinimumHeight(0)
        self.setMaximumHeight(16_777_215)
        self.updateGeometry()


class _BlockHeightHandle(QFrame):
    """Thin drag handle along a console section's lower edge."""

    HEIGHT = 7

    def __init__(self, block: _StdoutBlock):
        super().__init__(block)
        self._block = block
        self._press_y: Optional[float] = None
        self._start_height = 0
        self.setObjectName("ConsoleSectionResizeHandle")
        self.setCursor(Qt.SizeVerCursor)
        self.setFixedHeight(self.HEIGHT)
        source = (
            "Drag to resize this console section. Double-click for auto height."
        )
        self.setProperty("_spacr_i18n_tooltip", source)
        self.setToolTip(tr(source))

    def sizeHint(self) -> QSize:
        return QSize(80, self.HEIGHT)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._press_y = event.globalPosition().y()
            self._start_height = self._block.height()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._press_y is not None and event.buttons() & Qt.LeftButton:
            delta = event.globalPosition().y() - self._press_y
            self._block.set_user_height(self._start_height + int(delta))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._press_y = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._block.reset_user_height()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


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
        self._label.setProperty("i18nSkipText", True)
        self._label.setTextFormat(Qt.RichText)
        self._label.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse
        )
        self._label.setOpenExternalLinks(True)
        self._label.setWordWrap(True)
        self._label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self._label.setStyleSheet(
            "QLabel#ConsoleBubbleText {"
            f"  color: {active_palette()['fg']};"
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
        self._prefix_source = "spaCR user" if role == "user" else "spaCR AI"
        if text:
            self.set_text(text)

    def set_text(self, text: str) -> None:
        """Replace the bubble's body with ``text`` (HTML-escaped, wrapped)."""
        self._raw_text = text or ""
        safe = self._raw_text.replace("<", "&lt;").replace(">", "&gt;")
        safe = safe.replace("\n", "<br>")
        prefix = tr(self._prefix_source)
        html = f'<span style="opacity:0.7;">{prefix}: </span>{safe}'
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
        """Re-fit label height to the new wrap width."""
        super().resizeEvent(event)
        self._recalc()

    def showEvent(self, event):
        """Re-fit label height once the bubble becomes visible."""
        super().showEvent(event)
        self._recalc()


# ---------------------------------------------------------------------------
# Chat input — Enter sends, Shift+Enter newline
# ---------------------------------------------------------------------------

class _ChatInput(QTextEdit):
    """Multi-line chat input: Enter sends, Shift+Enter inserts a newline."""

    submitted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(48)
        self.setMaximumHeight(120)
        self.setAcceptRichText(False)

    def keyPressEvent(self, event: QKeyEvent):
        """Emit ``submitted`` on plain Enter; forward Shift+Enter as newline."""
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
                return
            self.submitted.emit()
            return
        super().keyPressEvent(event)

    def canInsertFromMimeData(self, source) -> bool:
        """Reject file/URL drops. A plain QTextEdit answers yes to a dropped file and
        then tries to read it into the text buffer — which freezes the whole app when
        the file (or folder) is large. Datasets belong on the app's dropzone, not the
        chat box, so only real text is insertable here."""
        if source.hasUrls():
            return False
        return super().canInsertFromMimeData(source)

    def insertFromMimeData(self, source) -> None:
        if source.hasUrls():
            return                       # ignore dropped files; never read them in here
        super().insertFromMimeData(source)


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

class ConsolePanel(QWidget):
    """Merged pipeline stdout + AI chat panel.

    Owns the AI stream thread so provider switches and app changes
    do not orphan a running subprocess. See the module docstring for
    the full public surface.

    :ivar ai_stream_finished: emitted when an AI stream ends (ok or
        error) so the parent screen can flip its Cancel button back.
    """

    # Fires when an AI stream ends (ok or error) so the AppScreen
    # actions row can flip a Cancel button back to something else.
    ai_stream_finished = Signal()

    #: Internal relays used by :meth:`append_stdout` / :meth:`append_error`
    #: to hop a call made on a worker thread onto the thread that owns
    #: this widget. Both are connected to the very method that emits
    #: them — a bound method of this QObject, so Qt queues the delivery
    #: rather than running it inline — and the second entry finds itself
    #: on the GUI thread and falls through to the real body.
    _relay_stdout = Signal(str)
    _relay_error = Signal(str)
    _relay_notice = Signal(str, object)

    def __init__(self, active_app_label: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("ConsolePanel")
        # QWidget (unlike QFrame) doesn't paint a QSS background/border/radius
        # unless told to — without this the ConsolePanel's rounded surface box
        # never draws and the console area shows straight through to the black
        # app background. WA_StyledBackground makes the rounded box appear.
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._active_app_label = active_app_label or ""
        # Module + function the current pipeline output is coming from, shown
        # in the "spaCR output — <module> — <function>" banner.
        self._run_module: str = ""
        self._run_function: str = ""
        self._last_entry_kind: str = ""   # "stdout" | "ai" | "user" | ""
        self._current_stdout: Optional[_StdoutBlock] = None
        self._working_dots: Optional[_WorkingDots] = None
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

        # Wired before anything can append: a log record can arrive the
        # instant this panel is registered as the console target.
        self._relay_stdout.connect(self.append_stdout)
        self._relay_error.connect(self.append_error)
        self._relay_notice.connect(self._append_notice_on_gui_thread)

        self._build_ui()
        # Pipe records from the global logger into this console. Every
        # ConsolePanel subscribes to the same shared signal handler,
        # so log records fanned out across screens all see them.
        try:
            from ..logging_util import get_signal_handler
            get_signal_handler().record_ready.connect(self._on_log_record)
        except Exception:
            pass
        retranslate_widget_tree(self)

    # ------------------------------------------------------------------
    def _build_ui(self):
        outer = QVBoxLayout(self)
        # The panel itself is transparent (see theme QSS) — the rounded box is
        # the ConsoleBox frame below, so the AI chat input can sit UNDER it as
        # a separate, edge-aligned row rather than inside the box.
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACING["sm"])

        # Console box — a rounded surface frame that wraps ONLY the scrolling
        # output. QFrame paints its QSS background/border/radius natively.
        self._console_box = QFrame()
        self._console_box.setObjectName("ConsoleBox")
        box_lay = QVBoxLayout(self._console_box)
        inset = SPACING["sm"]
        box_lay.setContentsMargins(inset, inset, inset, inset)
        box_lay.setSpacing(0)

        # Scroll area of entries
        self._scroll = QScrollArea()
        self._scroll.setObjectName("ConsoleScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        # The viewport paints its own background — make it transparent too so
        # the box's rounded surface shows through at the corners.
        self._scroll.viewport().setStyleSheet("background: transparent;")
        self._scroll.setStyleSheet("background: transparent;")
        # Never show a horizontal scrollbar — content that doesn't fit
        # must wrap. This is what prevents the runaway-width crash.
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._holder = QWidget()
        self._holder.setObjectName("ConsoleHolder")
        self._holder.setStyleSheet("background: transparent;")
        self._entries = QVBoxLayout(self._holder)
        self._entries.setContentsMargins(0, 0, 0, 0)
        # A little breathing room between console entries.
        self._entries.setSpacing(SPACING["xs"])
        self._entries.addStretch(1)
        self._scroll.setWidget(self._holder)
        box_lay.addWidget(self._scroll, 1)
        outer.addWidget(self._console_box, 1)

        # AI chat input — a separate row UNDER the console box (not inside it),
        # borderless wrapper so only the text field's own box shows, edges
        # flush with the console + system boxes. AI on/off toggle + provider
        # selector live in the AppScreen actions row.
        input_row = QWidget()
        row = QHBoxLayout(input_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["sm"])

        self._input = _ChatInput()
        self._input.setObjectName("ConsoleChatInput")
        self._input.setPlaceholderText(
            "Type here and hit Enter…  (toggle AI at the bottom-right "
            "to route through your chat subscription)"
        )
        self._input.submitted.connect(self._on_submit)
        row.addWidget(self._input, 1)
        outer.addWidget(input_row)

        # Console font-size control — its own right-aligned row below the input
        # so the text box stays full width, flush with the console + system
        # boxes. Adjusts every stdout/AI entry live.
        self._font_pt = int(QFontDatabase.systemFont(
            QFontDatabase.FixedFont).pointSize()) or 10
        font_row = QWidget()
        frow = QHBoxLayout(font_row)
        frow.setContentsMargins(0, 0, 0, 0)
        frow.addStretch(1)
        font_lbl = QLabel("Font size")
        font_lbl.setObjectName("Muted")
        frow.addWidget(font_lbl)
        self._font_spin = QSpinBox()
        self._font_spin.setRange(7, 22)
        self._font_spin.setValue(self._font_pt)
        self._font_spin.setToolTip("Console font size")
        self._font_spin.setFixedWidth(56)
        self._font_spin.valueChanged.connect(self.set_console_font_pt)
        frow.addWidget(self._font_spin)
        outer.addWidget(font_row)

        # AppScreen creates + owns the AI toggle/provider menu and calls
        # our setters when they change. Panel-internal state stays here
        # so we always know what to do on Enter.
        self._ai_active: bool = False
        self._current_provider_name: Optional[str] = None

    # ------------------------------------------------------------------
    # Font size
    # ------------------------------------------------------------------
    def set_console_font_pt(self, pt: int) -> None:
        """Set the console font size and apply it to every existing entry."""
        self._font_pt = int(pt)
        for block in self._holder.findChildren(_StdoutBlock):
            block.set_console_font_pt(self._font_pt)
        for lbl in self._holder.findChildren(QLabel):
            f = lbl.font()
            f.setPointSize(self._font_pt)
            lbl.setFont(f)

    def _apply_font(self, w: QWidget) -> None:
        """Apply the current console font size to a newly-created entry."""
        if isinstance(w, _StdoutBlock):
            w.set_console_font_pt(getattr(self, "_font_pt", 10))
        for lbl in ([w] if isinstance(w, QLabel) else w.findChildren(QLabel)):
            f = lbl.font()
            f.setPointSize(getattr(self, "_font_pt", 10))
            lbl.setFont(f)

    # ------------------------------------------------------------------
    # Entry-management helpers
    # ------------------------------------------------------------------
    def _insert_entry(self, w: QWidget) -> None:
        """Every entry — topic bar, stdout block, chat bubble — spans
        the full width of the console. Bubbles no longer get a
        horizontal offset row."""
        self._apply_font(w)
        self._entries.insertWidget(self._entries.count() - 1, w)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        sb = self._scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _needs_topic(self, kind: str) -> bool:
        return self._last_entry_kind != kind

    def _on_gui_thread(self) -> bool:
        """True when the caller is on the thread that owns this widget.

        Everything this panel appends is a QWidget, and Qt only allows a
        QWidget to be built on the GUI thread. Python's logging module
        does not care: it runs handlers inline on whatever thread logged
        the record, so a pipeline worker's ``LOG.warning`` used to land
        in :meth:`append_stdout` on the worker thread and construct a
        ``_TopicBar`` there.
        """
        return QThread.currentThread() is self.thread()

    # ------------------------------------------------------------------
    # Public: pipeline hooks
    # ------------------------------------------------------------------
    def set_active_app(self, label: str) -> None:
        """Set the label used in the next auto-inserted topic divider."""
        self._active_app_label = label

    def set_run_context(self, module: str = "", function: str = "") -> None:
        """Record the module/function the pipeline output comes from.

        Shown in the "spaCR output — <module> — <function>" banner so users
        can see the source of the output at a glance.
        """
        self._run_module = module or ""
        self._run_function = function or ""

    def _output_banner(self, head: str) -> str:
        """Build a banner like 'spaCR output — mask — preprocess_generate_masks'."""
        parts = [tr(head)]
        mod = self._run_module or self._active_app_label
        if mod:
            parts.append(str(mod))
        if self._run_function:
            parts.append(str(self._run_function))
        return "  —  ".join(parts)

    def begin_topic(self, label: str, accent: Optional[str] = None,
                    trailing: Optional[QWidget] = None) -> None:
        """Insert a divider bar labeled `label` (e.g. 'spaCR output — …')."""
        self._insert_entry(_TopicBar(label, accent=accent, trailing=trailing))
        self._last_entry_kind = ""    # force next append to open a block
        self._current_stdout = None

    def append_stdout(self, text: str) -> None:
        """Append pipeline output as blue text under a 'spaCR output' banner.

        Safe to call from any thread: an off-thread call is re-posted to
        the GUI thread through :attr:`_relay_stdout` and returns without
        touching a widget. See :meth:`_on_gui_thread`.
        """
        if not text:
            return
        if not self._on_gui_thread():
            self._relay_stdout.emit(text)
            return
        if self._current_stdout is None:
            # Open a "spaCR output — <module> — <function>" banner + a fresh
            # blue-text block. Reused until a different entry type breaks it.
            accent = color_output()
            self.begin_topic(self._output_banner("spaCR output"),
                             accent=accent)
            self._current_stdout = _StdoutBlock(text_color=accent)
            self._insert_entry(self._current_stdout)
            self._last_entry_kind = "stdout"
        self._current_stdout.append(text)
        self._scroll_to_bottom()

    def append_notice(self, source: str, **values: object) -> None:
        """Append one localized spaCR-authored UI notice.

        This is intentionally separate from :meth:`append_stdout`: arbitrary
        worker stdout, logs, tracebacks, paths and AI responses must remain
        byte-for-byte English/canonical. Off-thread notices carry their stable
        English template to the GUI thread and are translated only there.
        """
        if not source:
            return
        if not self._on_gui_thread():
            self._relay_notice.emit(str(source), dict(values))
            return
        self._append_notice_on_gui_thread(str(source), dict(values))

    def _append_notice_on_gui_thread(
        self, source: str, values: object = None,
    ) -> None:
        mapping = values if isinstance(values, dict) else {}
        # Call sites may add line breaks for console layout. Translation keys
        # deliberately omit incidental leading/trailing whitespace, so retain
        # that framing around the translated semantic template.
        core = source.strip()
        if not core:
            return
        leading = source[:len(source) - len(source.lstrip())]
        trailing = source[len(source.rstrip()):]
        self.append_stdout(leading + tr(core, **mapping) + trailing)

    def _on_log_record(self, text: str, level: int) -> None:
        """Slot for QtLogHandler.record_ready. WARNING/ERROR/CRITICAL
        records go through append_error so they're visually distinct."""
        import logging as _logging
        if level >= _logging.WARNING:
            self.append_error(text)
        else:
            self.append_stdout(text)

    def append_error(self, tb: str) -> None:
        """Append red error text under a 'spaCR ERROR — <module> — <function>'
        banner.

        :param tb: traceback text; empty strings are ignored.

        Thread-safe in the same way as :meth:`append_stdout`.
        """
        if not tb:
            return
        if not self._on_gui_thread():
            self._relay_error.emit(tb)
            return
        red = color_error()
        self.begin_topic(self._output_banner("spaCR ERROR"), accent=red)
        block = _StdoutBlock(tb, error=True, text_color=red)
        self._insert_entry(block)
        self._last_entry_kind = "stdout"

    def clear(self) -> None:
        """Wipe every entry (topic bars, stdout blocks, chat bubbles)."""
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
        """Enable/disable AI routing for Enter-submits from the input."""
        self._ai_active = bool(on)

    def set_ai_provider(self, provider_name: Optional[str]) -> None:
        """Select the provider used for AI submissions, or None to unset."""
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
            # Local note — green "spaCR user" text under its own banner.
            self._append_user(text)

    def _append_user(self, text: str) -> None:
        """Insert a 'spaCR user' banner + green user text."""
        green = color_user()
        self.begin_topic(tr("spaCR user"), accent=green)
        block = _StdoutBlock(text, text_color=green)
        self._insert_entry(block)
        self._current_stdout = None
        self._last_entry_kind = "user"

    def _send_to_ai(self, text: str) -> None:
        provider = self._current_provider()
        if provider is None:
            self.append_notice(
                "[AI] No provider configured. Open Providers…\n"
            )
            return
        if self._ai_thread is not None:
            # Silent no-op: another stream is running. The AppScreen
            # actions row exposes the Cancel button, not us.
            return
        self._ai_messages.append({"role": "user", "content": text})
        # User message — green "spaCR user" text.
        self._append_user(text)
        # AI reply — a "spaCR AI" banner tinted in the provider colour, with a
        # three-dot working indicator that cycles until the stream finishes,
        # followed by the reply text in the same provider colour.
        ai_color = ai_color_for_provider(self._current_provider_name)
        self._working_dots = _WorkingDots(color=ai_color)
        self.begin_topic(tr("spaCR AI"), accent=ai_color,
                         trailing=self._working_dots)
        self._working_dots.start()
        self._current_stdout = _StdoutBlock(text_color=ai_color)
        self._insert_entry(self._current_stdout)
        self._last_entry_kind = "ai"
        self._start_stream(system=ai_settings.get_system_prompt())

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
        """Return True while an AI response is being streamed."""
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
        """Ensure the AI thread is drained before Qt destroys the panel."""
        self.shutdown()
        super().closeEvent(event)

    def _on_stage(self, _stage: str) -> None:
        # Could show a spinner; keeping this quiet for now.
        pass

    def _on_chunk(self, chunk: str) -> None:
        self._ai_buf.append(chunk)
        # Stream into the provider-coloured AI block created in _send_to_ai.
        # Guard in case it was cleared (open_error_flow uses its own path).
        if self._current_stdout is None or self._last_entry_kind != "ai":
            ai_color = ai_color_for_provider(self._current_provider_name)
            self._current_stdout = _StdoutBlock(text_color=ai_color)
            self._insert_entry(self._current_stdout)
            self._last_entry_kind = "ai"
        self._current_stdout.append(chunk)
        self._scroll_to_bottom()

    def _on_stream_finished(self, ok: bool, final_text: str) -> None:
        # Retire the current (thread, worker) pair — hold both refs
        # in a list so Python can't GC the QThread before its OS
        # thread has fully exited AND Qt's deleteLater has run.
        # Prune already-dead entries on the way in so the list can't
        # grow unbounded across a long session.
        self._prune_retired()
        # Stop the cycling working dots — the stream is done.
        if self._working_dots is not None:
            self._working_dots.stop()
            self._working_dots = None
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
                self.append_notice(
                    "(empty response — try again or switch provider)\n"
                )
        else:
            self.append_notice(
                "[AI error] {detail}\n", detail=final_text)
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
    def open_error_flow(self, traceback_text: str, active_app: str = "",
                        show_raw: bool = True) -> None:
        """Send a traceback to the AI explainer and stream the reply inline.

        :param traceback_text: raw traceback captured from the pipeline.
        :param active_app: optional app label used in the framing prompt.
        :param show_raw: when False, the raw traceback is NOT printed to the
            console (only a short note); the AI still receives it in its
            prompt, so the user can ask the AI to show the error.
        """
        from ..ai.prompts import wrap_error_for_prompt, error_explainer_prompt
        if self._current_provider() is None:
            self.append_notice(
                "[AI] Enable AI in the actions row + pick a provider first.\n"
            )
            return
        prompt = wrap_error_for_prompt(
            traceback_text, active_app or self._active_app_label
        )
        # The AI always receives the full error; the console only echoes the
        # raw traceback when show_raw is True.
        self._ai_messages.append({"role": "user", "content": prompt})
        self._append_user(
            prompt if show_raw
            else tr(
                "An error occurred — asking spaCR AI to explain it. "
                "(Ask the AI to \"show the raw error\" to see the traceback.)"
            ))
        # AI reply with provider colour + cycling working dots.
        ai_color = ai_color_for_provider(self._current_provider_name)
        self._working_dots = _WorkingDots(color=ai_color)
        self.begin_topic(tr("spaCR AI"), accent=ai_color,
                         trailing=self._working_dots)
        self._working_dots.start()
        self._current_stdout = _StdoutBlock(text_color=ai_color)
        self._insert_entry(self._current_stdout)
        self._last_entry_kind = "ai"
        self._start_stream(system=error_explainer_prompt())
