"""Combine pipeline output, errors, and AI chat in one console panel.

Topic bars separate pipeline stages and AI conversations in a shared scrolling
view. The input area and transcript form a vertical ``QSplitter``; when a
``persist_key`` is supplied, its state is stored under ``console/split/<key>``
and restored per screen.

:class:`ConsolePanel` can start a topic, append ordinary or error output,
launch the AI error-explanation flow, and clear the transcript. It owns its AI
worker so streamed state remains coherent while the active pipeline screen
changes.

Widget creation always occurs on the GUI thread. Calls to
:meth:`ConsolePanel.append_stdout` and :meth:`ConsolePanel.append_error` from
logging or pipeline workers are relayed through queued Qt signals before they
modify the transcript.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import (QByteArray, QSize, Qt, QThread, QTimer, Slot,
                            Signal)
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QKeyEvent,
    QTextBlockFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
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
from ..verbose_logger import console_write, console_write_in_progress
from .flash import Flash


#: Soft budget for pipeline scrollback attached to one AI turn. A complete
#: traceback is never cut to satisfy it; ordinary stdout yields first.
AI_CONSOLE_CONTEXT_CHARS = 24_000

#: A deliberately small, visible heuristic for the Auto context mode. The
#: adjacent dropdown is the override when wording falls outside this set.
_CONSOLE_QUESTION_TERMS = (
    "console", "traceback", "error", "exception", "failed", "failure",
    "went wrong", "what happened", "why did", "debug", "log output",
)



#: Palette role behind each of the three legacy ``COLOR_*`` names.
_TEXT_ROLES = {
    "COLOR_OUTPUT": "accent",
    "COLOR_USER":   "success",
    "COLOR_ERROR":  "error",
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


def color_warning() -> str:
    """Warning colour for the theme on screen right now.

    Distinct from :func:`color_error` on purpose -- see
    :meth:`ConsolePanel.append_warning`. Every theme already defines the
    ``warning`` role; nothing here needed inventing.
    """
    return active_palette()["warning"]


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


AI_COLOR_CLAUDE = "#DE7356"
AI_COLOR_OPENAI = "#74AA9C"
AI_COLOR_GEMINI = "#74AA9C"
AI_COLOR_DEFAULT = "#74AA9C"


def ai_color_for_provider(provider_name: Optional[str]) -> str:
    """Return the spaCR-AI text colour for a provider name.

    :param provider_name: the provider's name, matched case-insensitively by
        substring (``claude``, ``gpt``, ``gemini`` and the like); None or
        unknown gives the default colour.
    """
    p = (provider_name or "").lower()
    if "claude" in p or "anthropic" in p:
        return AI_COLOR_CLAUDE
    if "gpt" in p or "openai" in p or "chatgpt" in p:
        return AI_COLOR_OPENAI
    if "gemini" in p or "google" in p:
        return AI_COLOR_GEMINI
    return AI_COLOR_DEFAULT



#: Default height, in pixels, of the AI chat box — the height the old
#: hard-coded ``setMaximumHeight(120)`` pinned it at.
DEFAULT_CHAT_HEIGHT = 120

#: Floor for the chat box. Also the splitter's stop, since
#: ``setChildrenCollapsible(False)`` honours a child's minimum.
CHAT_MIN_HEIGHT = 48

#: Floor for the console box. Without one, a QScrollArea's minimum size hint
#: is a couple of pixels and "not collapsible" would still let the console be
#: dragged down to a sliver.
CONSOLE_MIN_HEIGHT = 80

#: QSettings key prefix for the persisted splitter state, one entry per
#: screen (``console/split/mask``, ``console/split/measure``, …). Screens are
#: used for different things and a user who wants a tall chat box on one does
#: not necessarily want it on all of them.
_SPLIT_KEY_PREFIX = "console/split"


def _settings():
    """The app's ``QSettings``, borrowed from :mod:`spacr.qt.preferences`.

    Going through preferences rather than constructing ``QSettings`` here
    keeps the org/app pair single-sourced — and, just as important, keeps this
    module inside the sandbox the test suite installs, which redirects by
    *path* and so only catches settings opened the same way everything else
    opens them.
    """
    from ..preferences import _settings as _prefs_settings
    return _prefs_settings()


def get_split_state(screen_key: str):
    """Return the saved ``QSplitter.saveState()`` blob for ``screen_key``.

    :param screen_key: the screen's app key, e.g. ``"mask"``.
    :returns: the stored ``QByteArray``, or ``None`` when the user has never
        dragged this screen's handle (or the stored value is unusable).
    """
    key = str(screen_key or "").strip()
    if not key:
        return None
    try:
        raw = _settings().value(f"{_SPLIT_KEY_PREFIX}/{key}")
    except Exception:
        return None
    if isinstance(raw, (bytes, bytearray, QByteArray)) and len(raw):
        return QByteArray(raw)
    return None


def set_split_state(screen_key: str, state) -> None:
    """Persist a ``QSplitter.saveState()`` blob against ``screen_key``.

    Stored as the splitter's own state rather than as a pixel pair on
    purpose: a saved ``[572, 120]`` means something different on a laptop
    panel than on the 4K display the same user docks into, whereas
    ``restoreState`` is the mechanism Qt itself defines for this.

    :param screen_key: the screen the state belongs to; stripped, and a blank
        key stores nothing.
    :param state: the ``QSplitter.saveState()`` bytes, stored as a
        ``QByteArray``.
    """
    key = str(screen_key or "").strip()
    if not key:
        return
    try:
        _settings().setValue(f"{_SPLIT_KEY_PREFIX}/{key}", QByteArray(state))
    except Exception:
        pass



class _CopyGlyphButton(QAbstractButton):
    """The two-offset-squares copy mark, drawn rather than shipped.

    An icon file would need a light and a dark variant and would have to be
    kept in step with the theme; two rounded rectangles in the current
    foreground colour follow it for free, at any DPI.

    :param parent: parent widget; ownership only.
    """

    #: Side of the front square, in px. The back one is drawn behind it,
    #: offset by :data:`_OFFSET`, which is what reads as "a copy".
    _SIDE = 9
    _OFFSET = 3

    def __init__(self, parent=None):
        """Build the copy mark, drawn rather than shipped as an icon."""
        super().__init__(parent)
        self.setObjectName("ConsoleCopyGlyph")
        self.setFocusPolicy(Qt.NoFocus)
        edge = self._SIDE + self._OFFSET + 5
        self.setFixedSize(edge, edge)
        self._flash = Flash(self)

    def flash_copied(self) -> None:
        """Briefly mark the glyph, so a silent clipboard write is visible."""
        self._flash.trigger()

    def paintEvent(self, _event) -> None:      # noqa: N802 (Qt naming)
        """Draw the two offset squares that read as one sheet over another.

        The accent colour while the copy flash is active, the dim foreground
        otherwise, and lighter under the pointer -- so the button says it is
        hoverable before it is pressed and says it worked after.

        :param _event: the paint event; unused.
        """
        from PySide6.QtGui import QPainter, QPen
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        try:
            palette = active_palette()
            colour = QColor(palette["button_accent"] if self._flash.active
                            else palette["fg_dim"])
        except Exception:
            colour = QColor("#888888")
        if self.underMouse() and not self._flash.active:
            colour = colour.lighter(150)
        pen = QPen(colour)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        side, off = self._SIDE, self._OFFSET
        painter.drawRoundedRect(off + 1, 1, side, side, 2, 2)
        painter.drawRoundedRect(1, off + 1, side, side, 2, 2)
        painter.end()


class _TopicBar(QFrame):
    """Dark-gray divider bar with a topic label ("spaCR output — …", …).

    An optional ``accent`` colour tints the label so each banner reads in
    the same colour as the text that follows it. A trailing ``widget`` (e.g.
    an animated working indicator) can be pinned to the right.
    """

    def __init__(self, label: str, parent=None, accent: Optional[str] = None,
                 trailing: Optional[QWidget] = None):
        """Build one console section heading.

        :param label: the heading text.
        :param parent: parent widget.
        :param accent: colour for the heading, or ``None`` for the theme's.
        :param trailing: an optional widget pinned to the right of the
            heading, for a count or a control belonging to the section.

        The heading is a CONTROL, not a caption: clicking it brings its
        section to the top and expands it, so it takes strong focus while
        preserving the operating system's cursor. A control only a mouse can
        reach is one some users cannot reach at all.
        """
        super().__init__(parent)
        self.setObjectName("ConsoleTopicBar")
        self.setFocusPolicy(Qt.StrongFocus)
        self._expanded = True
        lay = QHBoxLayout(self)
        lay.setContentsMargins(SPACING["md"], SPACING["xs"],
                                SPACING["md"], SPACING["xs"])
        self._chevron = QLabel("▾")
        self._chevron.setObjectName("ConsoleTopicChevron")
        self._chevron.setProperty("i18nSkipText", True)
        lay.addWidget(self._chevron)
        self._label = QLabel(label)
        self._label.setObjectName("ConsoleTopicLabel")
        self._label.setProperty("i18nSkipText", True)
        if accent:
            self._label.setStyleSheet(
                f"QLabel#ConsoleTopicLabel {{ color: {accent}; "
                "background: transparent; }")
        lay.addWidget(self._label)
        if trailing is not None:
            lay.addWidget(trailing)
        self._copy_btn = _CopyGlyphButton(self)
        copy_tip = "Copy this section, header and all"
        self._copy_btn.setProperty("_spacr_i18n_tooltip", copy_tip)
        self._copy_btn.setToolTip(tr(copy_tip))
        self._copy_btn.clicked.connect(self._copy_section)
        lay.addWidget(self._copy_btn)
        lay.addStretch(1)

    def text(self) -> str:
        """The header text, for the plain-text export."""
        return self._label.text()

    def is_expanded(self) -> bool:
        """Whether this section's body is showing."""
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        """Record the state and turn the chevron to match."""
        self._expanded = bool(expanded)
        self._chevron.setText("▾" if self._expanded else "▸")

    def _panel(self):
        """The owning :class:`ConsolePanel`, or None."""
        node = self.parentWidget()
        while node is not None and not hasattr(node, "toggle_section"):
            node = node.parentWidget()
        return node

    def _activate(self) -> None:
        """Fold or unfold this section, if the panel is still there."""
        panel = self._panel()
        if panel is not None:
            panel.toggle_section(self)

    def mouseReleaseEvent(self, event):        # noqa: N802 (Qt naming)
        """Raise this section on a click inside the bar.

        On RELEASE rather than press, so dragging off cancels. The copy button
        and any trailing widget are children with their own handlers, so a click
        on them never reaches here -- which is what keeps "copy this section"
        from also moving the viewport.

        :param event: the mouse event.
        """
        if (event.button() == Qt.LeftButton
                and self.rect().contains(event.pos())):
            self._activate()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):            # noqa: N802 (Qt naming)
        """Raise this section on Return, Enter or Space.

        :param event: the key event.
        """
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self._activate()
            return
        super().keyPressEvent(event)

    def _copy_section(self) -> None:
        """Put this section on the clipboard, asking the panel for its span."""
        panel = self.parent()
        for _ in range(6):
            if panel is None:
                return
            if hasattr(panel, "section_text"):
                break
            panel = panel.parent()
        else:
            return
        if panel is None:
            return
        text = panel.section_text(self)
        if not text.strip():
            return
        try:
            from PySide6.QtWidgets import QApplication
            QApplication.clipboard().setText(text)
        except Exception:
            return
        self._copy_btn.flash_copied()



class _WorkingDots(QLabel):
    """Three dots that cycle (. → .. → ...) to show work is in progress.

    :param color: the dots' colour, as a CSS string. Baked into a stylesheet
        at construction, so a theme change needs a new widget rather than a
        setter.
    :param parent: parent widget; ownership only.
    """

    def __init__(self, color: str = AI_COLOR_DEFAULT, parent=None):
        """Build the dots at the AI colour, baked into a stylesheet."""
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
        """Re-ink the working dots.

        :param color: the provider's colour, so the indicator matches the reply
            it belongs to.
        """
        self._color = color
        self.setStyleSheet(
            f"QLabel#ConsoleWorkingDots {{ color: {color}; "
            f"font-size: {max(6, FONT_SIZE['xs'] - 2)}px; font-weight: 400; "
            "background: transparent; }")

    def _render(self) -> None:
        """Draw the current dot count, padded to a fixed width.

        Padded so the row does not JITTER as the count cycles: three glyph-slots
        either way, and the text beside it stays where it is.
        """
        dots = "●" * (self._n + 1)
        pad = " " * (2 - self._n)
        self.setText(dots + pad)

    def _tick(self) -> None:
        """Advance one step through . / .. / ... and redraw."""
        self._n = (self._n + 1) % 3
        self._render()

    @Slot()
    def _noop_slot(self) -> None:                # pragma: no cover - anchor
        """Present so the Slot import is used even if the others change."""

    def _on_gui_thread(self) -> bool:
        """Whether the caller is the thread this widget lives on."""
        from PySide6.QtCore import QThread

        return QThread.currentThread() is self.thread()

    @Slot()
    def start(self) -> None:
        """Start the activity animation on the widget's Qt thread.

        This method may be called from any thread. Calls from a worker are
        queued onto the widget's owning thread before the timer is started.
        """
        if not self._on_gui_thread():
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(self, "start", Qt.QueuedConnection)
            return
        self._n = 0
        self._render()
        self._timer.start()
        self.show()

    @Slot()
    def stop(self) -> None:
        """Stop the animation and hide the dots.

        Marshalled onto the GUI thread when called from another: a stream
        finishes on a worker, and touching a widget from there is undefined.
        """
        if not self._on_gui_thread():
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(self, "stop", Qt.QueuedConnection)
            return
        self._timer.stop()
        self.hide()



class _StdoutBlock(QPlainTextEdit):
    """Readable text block that grows in place as pipeline output arrives.

    A single block is reused for a whole stdout run so line breaks do
    not fragment the console into one widget per line.  A read-only
    ``QPlainTextEdit`` gives us selectable plain text while also exposing
    ``QTextBlockFormat``—the reliable Qt API for real line spacing.  QSS
    does not implement CSS ``line-height`` for a ``QLabel``.

    The viewport never fills its own background (item 515). A scroll area's
    viewport fills with ``QPalette.Base`` by default and only the stylesheet
    turns that off; a polish nested in another stylesheet style's call lets
    the unpolish through (item 408), the fill comes back in the window
    colour, and the console text sat on black until the module was left and
    reopened. Switched off before any sheet sees it, there is nothing for an
    unpolish to restore.
    """

    LINE_HEIGHT_PERCENT = 145

    #: Characters kept in one block. Older text is dropped from the head.
    MAX_CHARS = 200_000

    def __init__(self, text: str = "", error: bool = False, parent=None,
                 text_color: Optional[str] = None):
        """Build the block one stdout run grows into.

        :param text: the initial contents.
        :param error: whether this is the error stream. Chooses the object
            name, so the theme colours it without this class deciding what
            "error" looks like.
        :param parent: parent widget.
        :param text_color: an explicit colour, or ``None`` for the theme's.

        The block is reused for a whole run rather than made per line: a
        widget per line fragments the console, and :data:`MAX_CHARS` drops
        text from the HEAD so a long run cannot grow without bound.
        """
        super().__init__(parent)
        self.viewport().setAutoFillBackground(False)
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
        if text_color is None:
            text_color = color_error() if error else color_output()
        self._text_color = text_color
        self._refresh_style()
        #: Characters currently in the document. Tracked rather than derived
        #: from ``toPlainText()``, which copies the whole document.
        self._chars = 0
        #: ``sizeHint`` cache — see :meth:`sizeHint`.
        self._size_key: tuple = ()
        self._size_value = 32
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

    def _block_format(self) -> QTextBlockFormat:
        """The 145% leading every paragraph in this block carries."""
        block_format = QTextBlockFormat()
        block_format.setLineHeight(
            float(self.LINE_HEIGHT_PERCENT),
            QTextBlockFormat.ProportionalHeight.value,
        )
        return block_format

    def _apply_line_spacing(self) -> None:
        """Apply the leading to every paragraph in the document.

        Whole-document, therefore O(document): only for the rare events that
        genuinely change every paragraph, such as a font-size change.
        :meth:`append` formats the paragraphs it creates and nothing else.
        """
        cursor = QTextCursor(self.document())
        cursor.select(QTextCursor.Document)
        cursor.mergeBlockFormat(self._block_format())

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
        """Append ``text`` in place, trimming the head past :attr:`MAX_CHARS`.

        Costs what the new text costs, not what the console already holds.
        This used to rebuild the entire document on every line —
        ``setPlainText("".join(buf))`` plus a document-wide
        ``mergeBlockFormat`` — which made a run's own output quadratic in
        its length. Measured on this tree: 0.56 ms per line for the first
        500, 6.64 ms per line by line 3000, and level there only because
        the 200k cap had been reached.

        That is not merely slow. With Verbose logging on,
        ``spacr.logging_util``'s profile hook emits a record on entry to
        every spaCR function — including the ones inside Qt event delivery
        — and each record lands here. At 7 ms a line the GUI thread cannot
        drain its own queue: the process sits at 100% CPU making no forward
        progress, which is exactly how the Qt shard "live-lock" presented.
        """
        if not text:
            return
        doc = self.document()
        cursor = QTextCursor(doc)
        cursor.movePosition(QTextCursor.End)
        first_touched = cursor.blockNumber()
        cursor.insertText(text)
        self._chars += len(text)
        fmt_cursor = QTextCursor(doc.findBlockByNumber(first_touched))
        fmt_cursor.setPosition(cursor.position(), QTextCursor.KeepAnchor)
        fmt_cursor.mergeBlockFormat(self._block_format())
        self._trim_to_cap()
        self.updateGeometry()

    def _trim_to_cap(self) -> None:
        """Drop whole paragraphs off the head until back under the cap.

        Removing from the front costs what is removed. Re-setting the
        document to its own tail — the previous approach — costs what is
        kept, on every single line once the cap is reached.
        """
        doc = self.document()
        while self._chars > self.MAX_CHARS and doc.blockCount() > 1:
            block = doc.begin()
            removed = block.length()
            cursor = QTextCursor(block)
            cursor.movePosition(
                QTextCursor.NextBlock, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            self._chars = max(0, self._chars - removed)

    def sizeHint(self) -> QSize:
        """Report the full document height; the outer console owns scrolling.

        Cached: Qt asks for a size hint several times per layout pass, and
        the answer can only change when the text, the width or the font
        does. Without the cache each of those calls walks every paragraph.
        """
        if self._user_height is not None:
            return QSize(
                max(120, super().sizeHint().width()), self._user_height)
        key = (self._chars, self.viewport().width(), self._font_pt)
        if key != self._size_key:
            layout = self.document().documentLayout()
            height = 0.0
            block = self.document().begin()
            while block.isValid():
                height += layout.blockBoundingRect(block).height()
                block = block.next()
            chrome = (2 * SPACING["sm"]) + 2
            self._size_key = key
            self._size_value = max(32, int(round(height)) + chrome)
        return QSize(max(120, super().sizeHint().width()), self._size_value)

    def resizeEvent(self, event) -> None:
        """Re-wrap the text and keep the drag handle pinned to the bottom edge.

        :param event: the resize event.
        """
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
    """Thin drag handle along a console section's lower edge.

    :param block: the section this handle resizes. ALSO ITS QWIDGET PARENT,
        so the handle is laid out inside the block it drags and cannot
        outlive it; there is no separate ``parent``.
    """

    HEIGHT = 7

    def __init__(self, block: _StdoutBlock):
        """Build the handle with a vertical-resize cursor and its tooltip."""
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
        """Return the handle's preferred size.

        :returns: a strip as tall as the handle and nominally 80 wide -- the
            width comes from the block it spans, not from this hint.
        """
        return QSize(80, self.HEIGHT)

    def mousePressEvent(self, event) -> None:
        """Begin a drag, recording where it started and how tall the block was.

        Both are needed: the new height is the starting height plus the total
        movement, so a drag that reverses returns to where it began rather than
        accumulating.

        :param event: the mouse event.
        """
        if event.button() == Qt.LeftButton:
            self._press_y = event.globalPosition().y()
            self._start_height = self._block.height()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """Resize the block to follow the drag.

        :param event: the mouse event.
        """
        if self._press_y is not None and event.buttons() & Qt.LeftButton:
            delta = event.globalPosition().y() - self._press_y
            self._block.set_user_height(self._start_height + int(delta))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """End the drag.

        :param event: the mouse event.
        """
        self._press_y = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        """Return the block to its automatic height.

        A double click is the undo for the drag: without it a block dragged
        short can only be restored by guessing its original size.

        :param event: the mouse event.
        """
        if event.button() == Qt.LeftButton:
            self._block.reset_user_height()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)



class _Bubble(QFrame):
    """Chat bubble — a coloured QFrame that renders wrapped rich text.

    Manual sizing: on every resizeEvent we clamp the inner label's
    width to our own width minus padding, then set the label's fixed
    height from QFontMetrics.boundingRect for that wrap width. The
    frame's height is set to match. Simple, works reliably even
    inside a QScrollArea.

    :param role: ``"user"`` or anything else, which is read as the AI. It
        picks the object name and so the whole appearance -- there is no
        third style, and an unrecognised role is drawn as the AI rather
        than refused.
    :param text: the initial message; may be set later instead.
    :param parent: parent widget; ownership only.
    """

    _H_PAD = 24
    _V_PAD = 12

    def __init__(self, role: str, text: str = "", parent=None):
        """Build the bubble, styled by role, sized to its text."""
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
            return
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



class _ChatInput(QTextEdit):
    """Multi-line chat input: Enter sends, Shift+Enter inserts a newline.

    :param parent: parent widget; ownership only.
    """

    submitted = Signal()

    def __init__(self, parent=None):
        """Build the input, floored at one line and capped so it cannot take the pane."""
        super().__init__(parent)
        self.setMinimumHeight(CHAT_MIN_HEIGHT)
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
        """Paste text, ignoring dropped files.

        A file dropped on the chat box is never read here: the console is a
        place to type a question, and silently pasting a path -- or worse, a
        file's contents -- is not what the gesture meant.

        :param source: the mime data being inserted.
        """
        if source.hasUrls():
            return
        super().insertFromMimeData(source)



class ConsolePanel(QWidget):
    """Merged pipeline stdout + AI chat panel.

    Owns the AI stream thread so provider switches and app changes
    do not orphan a running subprocess. See the module docstring for
    the full public surface.

    The console box and the AI chat box are the two halves of a vertical
    :class:`~PySide6.QtWidgets.QSplitter`, so the user can drag the handle
    between them to trade height — a taller chat box is a shorter console.
    Pass ``persist_key`` and the position is remembered per screen.

    :ivar ai_stream_finished: emitted when an AI stream ends (ok or
        error) so the parent screen can flip its Cancel button back.
    """

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

    def __init__(self, active_app_label: str = "", parent=None,
                 persist_key: str = "", *, follow_log: bool = True,
                 chat: bool = True):
        """
        :param active_app_label: the app name shown in the output banner.
        :param parent: parent widget.
        :param persist_key: screen key the console/chat split is remembered
            against (usually the screen's ``app_key``). Empty means the split
            is not persisted, which is what a bare panel in a test wants.
        :param follow_log: False keeps the application-wide log out of this
            panel, for a screen whose console carries only its own messages
            (Make Masks); the log still reaches the file and the shell's
            console.
        :param chat: False hides the chat row, for a console that only
            reports.
        """
        super().__init__(parent)
        self.setObjectName("ConsolePanel")
        self._persist_key = str(persist_key or "").strip()
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._active_app_label = active_app_label or ""
        self._run_module: str = ""
        self._run_function: str = ""
        self._last_entry_kind: str = ""
        self._current_stdout: Optional[_StdoutBlock] = None
        #: Label of the topic bar currently showing, so an
        #: identical one is not drawn again. See begin_topic.
        self._current_topic_label: Optional[str] = None
        self._working_dots: Optional[_WorkingDots] = None
        #: The "spaCR AI" heading opened when a reply was asked for, and the
        #: empty block under it, until the first chunk arrives. See
        #: :meth:`_take_down_the_waiting_heading`.
        self._pending_ai_topic: Optional[_TopicBar] = None
        self._pending_ai_block: Optional[_StdoutBlock] = None
        #: The traceback the AI is currently explaining, and its answer once
        #: the stream finishes. Read by the bug reporter -- see
        #: :meth:`ai_explanation_of`.
        self._ai_error_traceback: str = ""
        self._ai_error_explanation: str = ""
        self._ai_messages: List[Dict] = []
        self._ai_buf: List[str] = []
        self._ai_thread: Optional[QThread] = None
        self._ai_worker: Optional[StreamWorker] = None
        self._console_sent_lengths: Dict[int, int] = {}
        self._retired: List = []

        self._relay_stdout.connect(self.append_stdout)
        self._relay_error.connect(self.append_error)
        self._relay_notice.connect(self._append_notice_on_gui_thread)

        self._build_ui()
        if not chat:
            self._chat_row.setVisible(False)
        if follow_log:
            try:
                from ..logging_util import get_signal_handler
                get_signal_handler().record_ready.connect(self._on_log_record)
            except Exception:
                pass
        retranslate_widget_tree(self)

    def _build_ui(self):
        """Lay out the console box and the chat row as the two halves of a splitter.

        The handle between them trades height: drag it up and the chat grows
        while the console shrinks by the same amount. Only the console carries a
        stretch factor, so a taller window grows the scrollback and leaves the
        chat box at whatever height the user gave it.
        """
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACING["sm"])

        self._split = QSplitter(Qt.Vertical)
        self._split.setObjectName("ConsoleSplit")
        self._split.setChildrenCollapsible(False)
        self._split.setHandleWidth(SPACING["sm"])
        try:
            from ..theme import make_transparent
            make_transparent(self._split)
        except Exception:
            pass
        try:
            accent = active_palette()["button_accent"]
        except Exception:
            accent = "#4A9EFF"
        self._split.setStyleSheet(
            self._split.styleSheet()
            + f"""
QSplitter#ConsoleSplit::handle:vertical {{
    background: transparent;
    border: none;
}}
QSplitter#ConsoleSplit::handle:vertical:hover {{
    background: transparent;
    border-top: 1px solid {accent};
}}
""")
        outer.addWidget(self._split, 1)

        self._console_box = QFrame()
        self._console_box.setObjectName("ConsoleBox")
        self._console_box.setMinimumHeight(CONSOLE_MIN_HEIGHT)
        box_lay = QVBoxLayout(self._console_box)
        inset = SPACING["sm"]
        box_lay.setContentsMargins(inset, inset, inset, inset)
        box_lay.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("ConsoleScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        self._scroll.viewport().setAutoFillBackground(False)
        self._scroll.viewport().setStyleSheet("background: transparent;")
        self._scroll.setStyleSheet("background: transparent;")
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._holder = QWidget()
        self._holder.setObjectName("ConsoleHolder")
        self._holder.setStyleSheet("background: transparent;")
        self._entries = QVBoxLayout(self._holder)
        self._entries.setContentsMargins(0, 0, 0, 0)
        self._entries.setSpacing(SPACING["xs"])
        self._entries.addStretch(1)
        self._scroll.setWidget(self._holder)
        #: Whether the view follows new output. Cleared by raising a section
        #: and restored by scrolling back to the bottom.
        self._follow_output = True
        self._scroll.verticalScrollBar().valueChanged.connect(
            self._on_console_scrolled)
        self._scroll.verticalScrollBar().valueChanged.connect(
            self._refresh_jump_button)
        box_lay.addWidget(self._scroll, 1)

        from PySide6.QtGui import QKeySequence, QShortcut
        from PySide6.QtWidgets import QPushButton

        self._jump = QPushButton("↓ jump to the end", self._console_box)
        self._jump.setToolTip(
            "Go to the newest line. Ctrl+End does the same from anywhere in "
            "the scrollback.")
        self._jump.clicked.connect(self.jump_to_the_end)
        self._jump.setVisible(False)
        box_lay.addWidget(self._jump)

        self._end_shortcut = QShortcut(QKeySequence("Ctrl+End"), self)
        self._end_shortcut.activated.connect(self.jump_to_the_end)
        self._split.addWidget(self._console_box)

        input_row = QWidget()
        self._chat_row = input_row
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
        self._split.addWidget(input_row)

        self._split.setStretchFactor(0, 1)
        self._split.setStretchFactor(1, 0)
        self._apply_default_split()
        self._restore_split()
        self._split.splitterMoved.connect(self._on_split_moved)

        self._font_pt = self._zoomed_font_pt()

        self._ai_active: bool = False
        self._current_provider_name: Optional[str] = None

    def _apply_default_split(self) -> None:
        """Seat the handle where the panel used to draw it with no splitter.

        ``setSizes`` totals are advisory — Qt rescales them to the height the
        splitter actually has, distributing the difference by stretch factor.
        With the chat box on stretch 0 the whole difference lands on the
        console, so the chat box comes out at exactly
        :data:`DEFAULT_CHAT_HEIGHT` at any window size, which is what the old
        ``setMaximumHeight(120)`` produced.
        """
        self._split.setSizes([max(CONSOLE_MIN_HEIGHT, 400), DEFAULT_CHAT_HEIGHT])

    def _restore_split(self) -> None:
        """Re-seat the handle where this screen's user last dragged it."""
        if not self._persist_key:
            return
        state = get_split_state(self._persist_key)
        if state is None:
            return
        try:
            self._split.restoreState(state)
        except Exception:
            self._apply_default_split()
        self._split.setChildrenCollapsible(False)

    def _on_split_moved(self, _pos: int = 0, _index: int = 0) -> None:
        """Persist the split as the user drags the handle."""
        if not self._persist_key:
            return
        set_split_state(self._persist_key, self._split.saveState())

    def split_sizes(self) -> List[int]:
        """Current ``[console_height, chat_height]`` in pixels.

        Public because it is the honest thing for a test — or a caller
        arranging the screen — to read, rather than reaching into ``_split``.
        """
        return list(self._split.sizes())

    def set_split_sizes(self, console_px: int, chat_px: int) -> None:
        """Move the handle programmatically and persist the result.

        Same end state as a user drag, so a caller restoring a layout and a
        user dragging leave the panel in the same place.

        :param console_px: height for the console box.
        :param chat_px: height for the AI chat box.
        """
        self._split.setSizes([int(console_px), int(chat_px)])
        self._on_split_moved()

    @staticmethod
    def _zoomed_font_pt() -> int:
        """The console point size, from the platform's fixed font x Zoom.

        The base is the system's monospace size so the console still looks
        native, and the Zoom preference multiplies it so the console tracks
        the rest of the interface. Falls back to the unscaled base if
        preferences cannot be read at all, which is what a first run
        mid-generation gets.
        """
        base = int(QFontDatabase.systemFont(
            QFontDatabase.FixedFont).pointSize()) or 10
        try:
            from ..preferences import get_font_scale
            return max(1, int(round(base * get_font_scale())))
        except Exception:
            return base

    def apply_zoom(self) -> None:
        """Re-read Zoom and restyle every entry. Called on a preferences save."""
        self.set_console_font_pt(self._zoomed_font_pt())

    def set_console_font_pt(self, pt: int) -> None:
        """Set the console font size and apply it to every existing entry.

        :param pt: the font size in points, converted to int.
        """
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

    def _insert_entry(self, w: QWidget) -> None:
        """Every entry — topic bar, stdout block, chat bubble — spans
        the full width of the console. Bubbles no longer get a
        horizontal offset row."""
        self._apply_font(w)
        self._entries.insertWidget(self._entries.count() - 1, w)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        """Follow the newest line, unless the user has scrolled away.

        Raising a section is a statement that the user is reading there, and a
        log that scrolls away from what is being read cannot be read at all.
        """
        if not getattr(self, "_follow_output", True):
            return
        sb = self._scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_console_scrolled(self, value: int) -> None:
        """Follow the tail while the view is at the bottom, and only then.

        The convention every log viewer uses: scrolling up means "let me
        read", scrolling back down means "keep going". A few pixels of
        tolerance because a scrollbar dragged to the end does not always land
        exactly on maximum().

        BOTH DIRECTIONS, because only one of them was ever wired. Raising a
        section cleared the follow, so a reader who got to the middle of the
        log by clicking a heading stayed there; a reader who got to the same
        place by dragging the scrollbar was thrown back to the bottom by the
        next line written, which is the thing that makes a live log
        unreadable. Where the viewport is answers that question, and it
        answers it the same way whichever gesture put it there.
        """
        scrollbar = self._scroll.verticalScrollBar()
        self._follow_output = value >= scrollbar.maximum() - 4

    def jump_to_the_end(self) -> None:
        """Show the newest line, and follow the tail again.

        BOTH HALVES, because they are one decision. A console that jumped
        without resuming the follow would slide back off the end on the very
        next line written, and the user would press it again.
        """
        bar = self._scroll.verticalScrollBar()
        bar.setValue(bar.maximum())
        self._follow_output = True
        self._refresh_jump_button()

    def at_the_end(self) -> bool:
        """Whether the view is showing the newest line.

        A few pixels of tolerance, because a scrollbar dragged to the end
        does not always land exactly on maximum() -- the same tolerance
        `_on_console_scrolled` uses, and for the same reason.
        """
        bar = self._scroll.verticalScrollBar()
        return bar.value() >= bar.maximum() - 4

    def _refresh_jump_button(self, *_args) -> None:
        """Show the control only when it would do something."""
        button = getattr(self, "_jump", None)
        if button is not None:
            button.setVisible(not self.at_the_end())

    def _needs_topic(self, kind: str) -> bool:
        """Report whether a new banner is needed before writing this kind of entry.

        :param kind: the entry kind about to be written.
        :returns: ``True`` when it differs from the last one written, so
            consecutive entries of one kind share a single banner.
        """
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

    def set_active_app(self, label: str) -> None:
        """Set the label used in the next auto-inserted topic divider.

        :param label: the text shown in the next automatic topic divider.
        """
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
            parts.append(tr(str(mod)))
        if self._run_function:
            parts.append(str(self._run_function))
        return "  —  ".join(parts)

    def begin_topic(self, label: str, accent: Optional[str] = None,
                    trailing: Optional[QWidget] = None
                    ) -> Optional["_TopicBar"]:
        """Insert a divider bar labeled `label` (e.g. 'spaCR output — …').

        A BAR IS NOT REDRAWN WHEN IT WOULD SAY THE SAME THING. Three bands now
        write under the "spaCR output" heading -- stdout, warnings, and the
        notice path -- and each opens its topic. A run that alternates between
        them therefore drew the identical banner before EVERY line:

            === spaCR output — Mask Generation ===
            Source directory (src): ...
            === spaCR output — Mask Generation ===
            12:09:37 [WARNING] cellpose.vit: Could not import CPDINO...
            === spaCR output — Mask Generation ===
            12:09:47 [INFO] spacr.qt.resource_cleanup: memory budget...

        which is what the console looked like when this was reported. The
        divider exists to say the subject CHANGED; repeating it says nothing
        and costs three lines of a panel people read during a run.

        The accent is deliberately not part of the comparison. It rides on the
        TEXT below the bar -- amber for a warning, blue for output -- so a
        warning still reads differently without a second identical heading
        above it.

        :param label: the heading text.
        :param accent: colour for the heading, or ``None`` for the theme's.
        :param trailing: a widget pinned to the right of the heading, such as
            a working indicator.
        :returns: the bar that was drawn, or ``None`` when the one already
            showing says the same thing and was kept. The caller needs the
            widget to be able to take an empty heading down again -- see
            :meth:`_take_down_the_waiting_heading`.
        """
        if label and label == getattr(self, "_current_topic_label", None):
            self._last_entry_kind = ""
            self._current_stdout = None
            return None
        bar = _TopicBar(label, accent=accent, trailing=trailing)
        self._insert_entry(bar)
        self._current_topic_label = label
        self._last_entry_kind = ""
        self._current_stdout = None
        return bar

    def append_stdout(self, text: str) -> None:
        """Append pipeline output as blue text under a 'spaCR output' banner.

        Safe to call from any thread: an off-thread call is re-posted to
        the GUI thread through :attr:`_relay_stdout` and returns without
        touching a widget. See :meth:`_on_gui_thread`.

        Re-entrant calls on the same thread are refused. Drawing a line
        runs Python inside a QWidget, and with verbose logging on the
        function-trace profile hook logs on entry to every spaCR function
        it passes through — including this one. Both console log sinks feed
        that record straight back here, and ``_StdoutBlock.append`` answers
        it with a nested ``setPlainText`` whose first act is to destroy the
        QTextDocument's frames — the ones the outer call is still inside.
        gdb: ``QTextFrame::~QTextFrame -> QTextDocumentPrivate::clear``,
        ``#0`` in freed memory. Reproduced as ``pytest
        tests/qt/test_all_module_smoke.py
        tests/qt/test_batch_f_diagnostics.py`` (exit 139).

        :param text: the pipeline output to append; empty does nothing.
        """
        if not text:
            return
        if not self._on_gui_thread():
            self._relay_stdout.emit(text)
            return
        if console_write_in_progress():
            return
        with console_write():
            if (self._current_stdout is None
                    or self._needs_topic("stdout")
                    or self._current_stdout.property(
                        "consoleContextKind") != "stdout"):
                accent = color_output()
                self.begin_topic(self._output_banner("spaCR output"),
                                 accent=accent)
                self._current_stdout = _StdoutBlock(text_color=accent)
                self._current_stdout.setProperty(
                    "consoleContextKind", "stdout")
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

        :param source: the notice's untranslated English template; empty does
            nothing. It is translated on the GUI thread and filled with the
            keyword values.
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
        """Translate a notice and append it, keeping its surrounding whitespace.

        Call sites add line breaks for console layout, but translation keys omit
        incidental leading and trailing whitespace -- so the framing is stripped
        off, the core translated, and the framing put back.

        :param source: the untranslated notice, with whatever framing it
            carries; a blank one is dropped.
        :param values: substitutions for the translated template; anything that
            is not a mapping is treated as none.
        """
        mapping = values if isinstance(values, dict) else {}
        core = source.strip()
        if not core:
            return
        leading = source[:len(source) - len(source.lstrip())]
        trailing = source[len(source.rstrip()):]
        self.append_stdout(leading + tr(core, **mapping) + trailing)

    def _on_log_record(self, text: str, level: int) -> None:
        """Slot for QtLogHandler.record_ready, routed by level.

        A WARNING IS NOT AN ERROR. This used to send everything at or above
        WARNING through :meth:`append_error`, which draws the red "spaCR
        ERROR" banner -- so a routine Qt warning ("libpyside: addMetaMethod
        ...") was presented to the user as a failure, a dozen times, on
        merely opening a module. The cost is not the wrong colour:
        it is that a pane which cries error over routine noise is a pane
        people stop reading, and the next line in it might be the one that
        matters.

        Three bands now, not two.
        """
        import logging as _logging
        if level >= _logging.ERROR:
            self.append_error(text)
        elif level >= _logging.WARNING:
            self.append_warning(text)
        else:
            self.append_stdout(text)

    def append_warning(self, text: str) -> None:
        """Append warning text in the theme's amber, under the output banner.

        DELIBERATELY NOT ITS OWN BANNER, and the reason is a coordination one
        rather than a design one: a "spaCR warning" heading would need a new
        row in ``spacr.qt.i18n._ROWS`` with nine translations, which moves the
        COMPACT caption ratchet. Amber under the existing translated "spaCR
        output" heading already achieves the point -- warnings out of the
        ERROR pane, errors still in it -- without reaching into that
        machinery. A dedicated banner would be the nicer end state.

        :param text: the formatted record; empty strings are ignored.
        """
        if not text:
            return
        with console_write():
            amber = color_warning()
            self.begin_topic(self._output_banner("spaCR output"), accent=amber)
            block = _StdoutBlock(text, text_color=amber)
            block.setProperty("consoleContextKind", "warning")
            self._insert_entry(block)
            self._last_entry_kind = "stdout"

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
        if console_write_in_progress():
            return
        with console_write():
            red = color_error()
            self.begin_topic(self._output_banner("spaCR ERROR"), accent=red)
            block = _StdoutBlock(tb, error=True, text_color=red)
            block.setProperty("consoleContextKind", "traceback")
            self._insert_entry(block)
            self._last_entry_kind = "stdout"

    def as_text(self, start: int = 0, stop: Optional[int] = None) -> str:
        """The console as plain text, section headers included.

        :param start: first entry index to include.
        :param stop: one past the last, or ``None`` for the rest.
        :returns: the text a person would have selected by hand.
        """
        parts = []
        last = self._entries.count() - 1
        stop = last if stop is None else min(stop, last)
        for index in range(max(0, start), stop):
            item = self._entries.itemAt(index)
            widget = item.widget() if item is not None else None
            if widget is None:
                continue
            if isinstance(widget, _TopicBar):
                parts.append(f"\n=== {widget.text()} ===")
            elif hasattr(widget, "toPlainText"):
                text = widget.toPlainText().rstrip()
                if text:
                    parts.append(text)
            elif hasattr(widget, "text"):
                text = (widget.text() or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip() + "\n"

    def _section_span(self, bar: "_TopicBar"):
        """Entry indices ``(start, stop)`` spanning ``bar`` and its content.

        ``stop`` is exclusive and may be ``None``, meaning "to the end".
        ONE definition of where a section ends, so copying, raising and
        collapsing cannot disagree about it.

        The span runs to the next header that actually has something under
        it. ``append_stdout`` inserts its own "spaCR output" bar, so a module
        banner is followed immediately by another banner: stopping at the
        first boundary copies a title and nothing else, and folds a section
        that hides nothing.
        """
        start = None
        last = self._entries.count() - 1
        boundaries = []
        for index in range(last):
            item = self._entries.itemAt(index)
            widget = item.widget() if item is not None else None
            if widget is bar:
                start = index
                continue
            if start is not None and isinstance(widget, _TopicBar):
                boundaries.append(index)
        if start is None:
            return None, None
        for boundary in boundaries:
            text = self.as_text(start, boundary)
            if len(text.strip().splitlines()) > 1:
                return start, boundary
        return start, None

    def section_text(self, bar: "_TopicBar") -> str:
        """Return a topic bar and its content up to the next topic bar.

        :param bar: the topic bar (section heading) whose section is meant. A
            bar not in the console gives ``""``.
        """
        start, stop = self._section_span(bar)
        if start is None:
            return ""
        return self.as_text(start, stop)

    def section_body(self, bar: "_TopicBar"):
        """The widgets under ``bar``, up to the next topic bar.

        The same span :meth:`section_text` copies, as widgets rather than as
        text, so raising, collapsing and copying a section cannot disagree
        about where it ends. A nested heading inside the span is part of the
        body: folding a module banner folds the "spaCR output" banner under
        it too, because that banner is the section's own content.

        :param bar: the topic bar (section heading) whose section is meant. A
            bar not in the console gives an empty list.
        """
        start, stop = self._section_span(bar)
        if start is None:
            return []
        end = self._entries.count() - 1 if stop is None else stop
        body = []
        for index in range(start + 1, end):
            item = self._entries.itemAt(index)
            widget = item.widget() if item is not None else None
            if widget is not None:
                body.append(widget)
        return body

    def raise_section(self, bar: "_TopicBar") -> None:
        """Bring ``bar``'s section to the top of the view and expand it.

        The console is a transcript, so the order of its sections is the one
        property a log has: this SCROLLS, it does not reorder.

        Raising a section also stops the view following new output. A user
        who clicked a heading is reading THERE, and appending output that
        yanks the viewport away is what makes a live log unreadable.
        Following resumes when they scroll back to the bottom, which is the
        convention every log viewer uses.

        :param bar: the topic bar (section heading) whose section is meant.
        """
        bar.set_expanded(True)
        folded = False
        for widget in self.section_body(bar):
            if isinstance(widget, _TopicBar):
                folded = not widget.is_expanded()
                widget.setVisible(True)
                continue
            widget.setVisible(not folded)
        self._follow_output = False
        QTimer.singleShot(0, lambda: self._scroll_widget_to_top(bar))

    def _scroll_widget_to_top(self, bar) -> None:
        """Scroll the console so a widget sits at the top of the viewport.

        :param bar: the widget to bring to the top -- typically a section
            heading that was just clicked. A section torn down between the click
            and the layout is ignored rather than raising.
        """
        try:
            top = bar.mapTo(self._holder, bar.rect().topLeft()).y()
        except RuntimeError:
            return
        scrollbar = self._scroll.verticalScrollBar()
        scrollbar.setValue(min(top, scrollbar.maximum()))

    def collapse_section(self, bar: "_TopicBar") -> None:
        """Hide ``bar``'s body, leaving its heading in place.

        :param bar: the topic bar (section heading) whose section is meant.
        """
        bar.set_expanded(False)
        for widget in self.section_body(bar):
            widget.setVisible(False)

    def _is_raised(self, bar: "_TopicBar") -> bool:
        """Whether ``bar`` is already sitting at the top of the viewport."""
        try:
            top = bar.mapTo(self._holder, bar.rect().topLeft()).y()
        except RuntimeError:
            return False
        scrollbar = self._scroll.verticalScrollBar()
        return abs(scrollbar.value() - min(top, scrollbar.maximum())) <= 4

    def toggle_section(self, bar: "_TopicBar") -> None:
        """Reach the section first; fold it away second.

        Sections are created EXPANDED, so a plain expanded/collapsed toggle
        spent the user's first click hiding the very section they were
        reaching for, and the viewport never moved -- the opposite of "click
        a console section heading to bring it to the top of the console".

        A heading that is not already at the top of the viewport is therefore
        a request to GO THERE, whatever its state. Only a heading already
        sitting at the top has nowhere left to navigate to, and there
        collapsing is the one thing the gesture can still mean -- reachable
        on a second click, exactly where the user's hand already is.

        :param bar: the topic bar (section heading) whose section is meant.
        """
        if bar.is_expanded() and self._is_raised(bar):
            self.collapse_section(bar)
        else:
            self.raise_section(bar)

    def copy_all(self) -> str:
        """Put the whole console on the clipboard; return what was copied."""
        text = self.as_text()
        try:
            from PySide6.QtWidgets import QApplication
            QApplication.clipboard().setText(text)
        except Exception:
            pass
        return text

    def clear(self) -> None:
        """Wipe every entry (topic bars, stdout blocks, chat bubbles)."""
        while self._entries.count() > 1:
            item = self._entries.takeAt(0)
            w = item.widget() if item else None
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._last_entry_kind = ""
        self._current_stdout = None
        self._current_topic_label = None
        self._pending_ai_topic = None
        self._pending_ai_block = None
        self._ai_messages.clear()
        self._console_sent_lengths.clear()

    def set_ai_active(self, on: bool) -> None:
        """Enable/disable AI routing for Enter-submits from the input.

        :param on: whether Enter in the input goes to the AI; converted to
            bool.
        """
        self._ai_active = bool(on)

    def set_ai_provider(self, provider_name: Optional[str]) -> None:
        """Select the provider used for AI submissions, or None to unset.

        :param provider_name: the provider's name, or None to unset it.
        """
        self._current_provider_name = provider_name

    def _current_provider(self) -> Optional[ChatProvider]:
        """Resolve the selected AI provider.

        :returns: the provider, or ``None`` when none is selected.
        """
        if not self._current_provider_name:
            return None
        return ai_module.get_provider(self._current_provider_name)

    def _on_submit(self) -> None:
        """Send the chat box's contents, to the AI or to the console.

        An empty box does nothing. With AI off the text is written as a local
        note under its own banner rather than dropped, so a typed thought stays
        in the transcript beside the run it was about.
        """
        text = self._input.toPlainText().strip()
        if not text:
            return
        self._input.clear()
        if self._ai_active:
            self._send_to_ai(text)
        else:
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
        """Send one question to the provider and open a reply block for it.

        Console context is attached according to the AI preference and reported
        on the message it went with, rather than as furniture that goes stale
        between asks. Asking a question of the user's own also ends the error
        pairing: whatever comes back answers this, not the crash, so it must not
        later be filed as an analysis of the crash.

        :param text: the user's question. With no provider configured this says
            so; with a stream already running it is silently dropped, since the
            Cancel button lives on the actions row rather than here.
        """
        provider = self._current_provider()
        if provider is None:
            self.append_notice(
                "[AI] No provider configured. Open Providers…\n"
            )
            return
        if self._ai_thread is not None:
            return
        context, status = self._console_context_for_question(text)
        prompt = text
        if context:
            prompt += (
                "\n\n<spacr_console_context>\n" + context
                + "\n</spacr_console_context>")
        self._ai_messages.append({"role": "user", "content": prompt})
        self._ai_error_traceback = ""
        self._append_user(text + f"\n\n[{status}]")
        ai_color = ai_color_for_provider(self._current_provider_name)
        self._working_dots = _WorkingDots(color=ai_color)
        self._pending_ai_topic = self.begin_topic(
            tr("spaCR AI"), accent=ai_color, trailing=self._working_dots)
        self._working_dots.start()
        self._current_stdout = _StdoutBlock(text_color=ai_color)
        self._insert_entry(self._current_stdout)
        self._pending_ai_block = self._current_stdout
        self._last_entry_kind = "ai"
        self._start_stream(system=ai_settings.get_system_prompt())

    def _pipeline_console_blocks(self):
        """Yield rendered pipeline blocks that are eligible as AI context."""
        last = self._entries.count() - 1
        for index in range(last):
            item = self._entries.itemAt(index)
            widget = item.widget() if item is not None else None
            kind = widget.property("consoleContextKind") if widget else None
            if kind in {"stdout", "traceback"} and hasattr(
                    widget, "toPlainText"):
                yield widget, str(kind), widget.toPlainText()

    def _console_context_for_question(self, question: str):
        """Package unsent console context when an AI question is submitted.

        Return ``(context, visible_status)``. Complete traceback blocks take
        priority over ordinary output and may exceed the soft context budget.
        When the default-on console-aware preference is disabled, no context
        is attached. Text is marked sent only after it is included.
        """
        from ..ai import settings as ai_settings

        if not ai_settings.get_console_aware():
            label = tr("Console context off")
            return "", label

        pieces = []
        current_lengths = {}
        for block, kind, full_text in self._pipeline_console_blocks():
            key = id(block)
            sent = min(self._console_sent_lengths.get(key, 0), len(full_text))
            fresh = full_text[sent:]
            if not fresh:
                continue
            anchor = ""
            if kind == "stdout" and sent:
                anchor = full_text[max(0, sent - 400):sent]
            pieces.append((kind, anchor + fresh, len(fresh)))
            current_lengths[key] = len(full_text)

        if not pieces:
            label = tr("Console context: no new output")
            return "", label

        tracebacks = [text for kind, text, _fresh in pieces
                      if kind == "traceback"]
        stdout = "\n".join(text for kind, text, _fresh in pieces
                           if kind == "stdout")
        traceback_text = "\n\n".join(
            f"--- complete traceback ---\n{text}" for text in tracebacks)
        remaining = max(0, AI_CONSOLE_CONTEXT_CHARS - len(traceback_text))
        kept_stdout = stdout[-remaining:] if remaining else ""
        dropped = max(0, len(stdout) - len(kept_stdout))
        sections = []
        if dropped:
            sections.append(
                f"[Console tail; {dropped:,} earlier characters dropped]")
        elif stdout:
            sections.append("[New console output]")
        if kept_stdout:
            sections.append(kept_stdout)
        if traceback_text:
            sections.append(traceback_text)
        context = "\n".join(sections)
        self._console_sent_lengths.update(current_lengths)
        label = tr("Console context: {n} chars sent", n=f"{len(context):,}")
        if dropped:
            label += tr(", {n} dropped", n=f"{dropped:,}")
        return context, label

    def _ensure_stdout_block(self) -> None:
        """Open a new plain stdout block if the last entry was not one."""
        if self._current_stdout is None or self._needs_topic("stdout"):
            block = _StdoutBlock()
            self._insert_entry(block)
            self._current_stdout = block
            self._last_entry_kind = "stdout"

    def _start_stream(self, system: str) -> None:
        """Start the streaming worker for the pending conversation.

        The thread is parented to the panel so its C++ lifetime is tied to the
        panel rather than to a Python refcount -- an unparented ``QThread`` can
        be collected between the worker returning and ``finished`` firing, which
        aborts Qt.

        :param system: system prompt for the request.
        """
        provider = self._current_provider()
        if provider is None:
            return
        self._ai_buf = []
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

        There is deliberately no ``QThread.terminate()`` fallback. It used
        to be here, described as a last resort, and it was reached far more
        often than "last resort" suggests: `spacr.qt.ai.worker` queued
        `worker.finished -> thread.quit` to the GUI-affine QThread object,
        so the event that stops the thread sat behind this method's own
        `wait()` and the wait timed out on streams that had already
        finished. Terminating a thread that is running Python is
        `pthread_cancel`: if it dies holding the GIL the process stops
        making progress with every thread still alive, and if it dies
        inside Qt or PySide the heap is corrupt and the crash lands
        somewhere unrelated later. `bridge.drain_thread` parks a thread
        that will not stop instead, which keeps the "never destroy a
        running QThread" rule without buying it with undefined behaviour.
        """
        from ..bridge import drain_thread

        worker = self._ai_worker
        thread = self._ai_thread
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
        drain_thread(thread, worker, timeout_ms=3000)
        self._ai_thread = None
        self._ai_worker = None
        for pair in list(self._retired):
            drain_thread(pair[0], pair[1], timeout_ms=1000)
        self._retired.clear()

    def closeEvent(self, event) -> None:
        """Ensure the AI thread is drained before Qt destroys the panel.

        :param event: the close event, passed to the base class after
            :meth:`shutdown` drains the AI thread.
        """
        self.shutdown()
        super().closeEvent(event)

    def _on_stage(self, _stage: str) -> None:
        """Ignore a stage change from the streaming worker.

        :param _stage: the stage name. Nothing is shown for it yet -- this is
            where a spinner would go.
        """
        pass

    def _take_down_the_waiting_heading(self) -> Optional[QWidget]:
        """Remove the "spaCR AI" heading opened before the reply, if it is empty.

        A reply's heading is drawn when the question is asked, so the working
        dots have somewhere to sit while the provider thinks. Anything written
        in the meantime opens a heading of its own underneath it -- a failing
        run writes its manifest, "run closed" and "✗ Failed" under
        "spaCR output" -- and the reply then arrives below THAT and opens a
        second "spaCR AI" heading. GitHub #117's console is the result, and
        read top to bottom the empty first heading is the part that says the
        AI was never asked.

        So the heading MOVES rather than repeating: the empty one is taken
        down, and its working indicator is handed back to be pinned to the
        heading that replaces it, which keeps the indicator running from the
        moment the question was asked. A reply that is not interrupted keeps
        the heading it started under and nothing here runs.

        :returns: the working indicator, detached and still running, to pass
            as the replacement heading's ``trailing``; or ``None`` when there
            was no empty heading to take down, or nothing was pinned to it.
        """
        bar = self._pending_ai_topic
        block = self._pending_ai_block
        self._pending_ai_topic = None
        self._pending_ai_block = None
        if bar is None:
            return None
        try:
            label = bar.text()
            if block is not None and block.toPlainText().strip():
                return None
        except RuntimeError:
            return None
        dots = self._working_dots
        if dots is not None:
            dots.setParent(self)
        for widget in (block, bar):
            if widget is None:
                continue
            try:
                self._entries.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
            except RuntimeError:
                pass
        if self._current_stdout is block:
            self._current_stdout = None
        if self._current_topic_label == label:
            self._current_topic_label = None
            self._last_entry_kind = ""
        return dots

    def _on_chunk(self, chunk: str) -> None:
        """Append one streamed chunk to the AI reply block.

        The block is recreated if it went away -- the error flow writes through
        its own path and can clear it mid-stream.

        A RECREATED BLOCK TAKES THE "spaCR AI" HEADING WITH IT. When a run
        fails, the error flow opens the heading and its reply block at once,
        and the run then writes its closing lines -- the manifest, "run
        closed", "✗ Failed" -- under "spaCR output" before the provider's
        first line arrives. Without a heading of its own the whole reply sat
        under "spaCR output", in the AI's colour, and the "spaCR AI" heading
        above it stayed empty: issue 117's console, which read as "the AI was
        never asked". Drawing a second heading fixed the colour and left the
        empty one, so the empty one is now taken down and its working
        indicator moves to the heading the reply really starts under -- see
        :meth:`_take_down_the_waiting_heading`.

        :param chunk: the text just received.
        """
        self._ai_buf.append(chunk)
        if self._current_stdout is None or self._last_entry_kind != "ai":
            ai_color = ai_color_for_provider(self._current_provider_name)
            dots = self._take_down_the_waiting_heading()
            self.begin_topic(tr("spaCR AI"), accent=ai_color, trailing=dots)
            self._current_stdout = _StdoutBlock(text_color=ai_color)
            self._insert_entry(self._current_stdout)
            self._last_entry_kind = "ai"
        else:
            self._pending_ai_topic = None
            self._pending_ai_block = None
        self._current_stdout.append(chunk)
        self._scroll_to_bottom()

    def _on_stream_finished(self, ok: bool, final_text: str) -> None:
        """Close the reply block and retire the streaming thread.

        The finished ``(thread, worker)`` pair is held in a list rather than
        dropped, so Python cannot collect the ``QThread`` before its OS thread
        has exited and Qt's ``deleteLater`` has run; already-dead entries are
        pruned on the way in so the list cannot grow across a long session.

        :param ok: whether the stream completed.
        :param final_text: the assembled reply, or the error detail when
            ``ok`` is ``False``.
        """
        self._prune_retired()
        if self._working_dots is not None:
            self._working_dots.stop()
            self._working_dots = None
        self._take_down_the_waiting_heading()
        thread, worker = self._ai_thread, self._ai_worker
        self._ai_thread = None
        self._ai_worker = None
        if thread is not None:
            self._retired.append((thread, worker))
        if ok:
            self._ai_messages.append(
                {"role": "assistant", "content": final_text}
            )
            if getattr(self, "_ai_error_traceback", ""):
                self._ai_error_explanation = final_text or ""
            if not self._ai_buf:
                self.append_notice(
                    "(empty response — try again or switch provider)\n"
                )
        else:
            self.append_notice(
                "[AI error] {detail}\n", detail=final_text)
        if self._current_stdout is not None:
            self._current_stdout.append("\n")
        self._ai_buf = []
        self.ai_stream_finished.emit()

    def ai_explanation_of(self, traceback_text: str) -> str:
        """spaCR AI's answer about ``traceback_text``, or ``""``.

        Used by the bug reporter: when the AI is switched on it has usually
        already diagnosed the crash by the time the user files, and that
        analysis is the most useful thing in the report -- it is what whoever
        picks the report up would otherwise spend the first hour
        reproducing.

        Empty unless there IS an answer AND it is an answer to THIS error.
        The console holds one conversation across a whole session, so without
        the second condition a report about one crash would carry an
        explanation of an earlier one, stated with equal confidence.

        :param traceback_text: the traceback the report is about; compared,
            whitespace-stripped, with the traceback the AI last explained.
        """
        mine = getattr(self, "_ai_error_traceback", "") or ""
        answer = getattr(self, "_ai_error_explanation", "") or ""
        if not mine or not answer:
            return ""
        return answer if mine.strip() == (traceback_text or "").strip() else ""

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
        self._ai_error_traceback = traceback_text
        self._ai_error_explanation = ""
        self._ai_messages.append({"role": "user", "content": prompt})
        self._append_user(
            prompt if show_raw
            else tr(
                "An error occurred — asking spaCR AI to explain it. "
                "(Ask the AI to \"show the raw error\" to see the traceback.)"
            ))
        ai_color = ai_color_for_provider(self._current_provider_name)
        self._working_dots = _WorkingDots(color=ai_color)
        self._pending_ai_topic = self.begin_topic(
            tr("spaCR AI"), accent=ai_color, trailing=self._working_dots)
        self._working_dots.start()
        self._current_stdout = _StdoutBlock(text_color=ai_color)
        self._insert_entry(self._current_stdout)
        self._pending_ai_block = self._current_stdout
        self._last_entry_kind = "ai"
        self._start_stream(system=error_explainer_prompt())
