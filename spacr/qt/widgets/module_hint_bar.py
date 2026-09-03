"""The strip along the bottom that explains the module under the pointer.

Asked for on 2026-09-03: "remove the popup window tooltip on the moduals. the
tooltip is shown at the botom of the screen. these should also have an API
link and a tutorial link and the list onw hovered should be shown at the
botom for 30 seconds. ... the hover over the doc should function the same."

THE HOLD IS THE WHOLE POINT, and it is the same argument instruction 371 made
for the per-setting strip: a link that appears only while the pointer is on
the tile is a link that cannot be clicked, because moving toward it removes
it. So the strip keeps the LAST module hovered for thirty seconds -- long
enough to notice it, cross the window and press a word.

Thirty rather than 371's ten because these two links leave the application.
Ten seconds is a budget for reaching a word; a reader deciding whether to
open documentation or a lesson in a browser is making a larger decision, and
the maintainer named the number.

TWO SURFACES, ONE BAR. Home's tiles and the dock's rows both write here, and
so does a module screen's own strip when the dock is hovered over it -- see
:meth:`spacr.qt.app.MainWindow._show_module_hint`, which routes to whichever
bar is on screen. A module explained differently depending on where you
pointed at it would be two explanations to maintain.
"""
from __future__ import annotations

from html import escape
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QLabel, QWidget

#: What the strip says when nothing has been hovered yet.
DEFAULT_HINT = "Hover a tile to see what it does."

#: The objectName the stylesheet selects on. Shared with the plain bar this
#: replaces so the two are one thing wearing one style.
BAR_NAME = "HintBar"


class ModuleHintBar(QLabel):
    """A module's summary, its API link and its Tutorial link, held 30 s.

    :param default: what the strip says with nothing hovered. Restored when
        the hold expires, so it should read as a prompt rather than a blank.
    :param parent: parent widget.
    """

    #: Emitted with the app key when a link is followed, so a caller can
    #: log or intercept. The bar opens the URL itself either way.
    link_followed = Signal(str, str)

    #: How long the strip keeps the last module hovered, in milliseconds.
    #:
    #: THE NUMBER WAS ASKED FOR: "shown at the botom for 30 seconds". It is
    #: the budget for noticing the strip, deciding, crossing the window and
    #: pressing a word that opens a browser -- not a value to tune down
    #: because it reads as long in source.
    HOLD_MS = 30_000

    def __init__(self, default: str = DEFAULT_HINT,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(default, parent)
        self._default = default
        self._key = ""
        self._timer: Optional[QTimer] = None
        self.setObjectName(BAR_NAME)
        self.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.setTextFormat(Qt.RichText)
        # The links leave the application, so Qt opens them. `setOpenExternal
        # Links` also makes the label focusable for a keyboard user, which is
        # what makes the words reachable without a pointer at all.
        self.setOpenExternalLinks(True)
        self.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.linkActivated.connect(self._on_link)
        # A FIXED HEIGHT, and it is load-bearing rather than tidy.
        #
        # A strip that takes whatever height its text needs GROWS when a long
        # summary wraps and shrinks when a short one does not -- so moving
        # the pointer between two modules relayouts the page under it. The
        # dock is in that layout. Reported 2026-09-03: "if i hover quickly an
        # element in the dock it blinks blue a bunch of times then stays blue
        # after a while" -- the row moving out from under the pointer and
        # back, delivering an Enter and a Leave each time.
        #
        # The same lesson twice already: `HintBar` caps itself at three lines
        # ("moving the pointer between two controls whose help differs in
        # length made the whole dialog jump") and the status bar is pinned
        # for it too ("without this the dock flickered on Linux each time one
        # arrived"). This is the third surface and the first to forget.
        #
        # TWO LINES: the summary and the row of links under it. Measured from
        # the font rather than pinned at a number, so it stays right at any
        # font scale -- a hard number is a promise about text metrics that
        # breaks the moment the scale or the theme's font stack changes.
        line = max(1, self.fontMetrics().lineSpacing())
        self.setFixedHeight(line * 2 + 10)
        # And the text is ELIDED into it rather than wrapping past it.
        self.setWordWrap(False)

    # -- what it shows ---------------------------------------------------

    @property
    def module_key(self) -> str:
        """The module the strip is currently explaining, or ``""``."""
        return self._key

    def show_module(self, key: str, summary: str, stage: str = "") -> str:
        """Explain ``key``, with its links, and start the hold.

        :param key: the module's app key. Both links are derived from it.
        :param summary: the sentence to show, already in the UI language.
        :param stage: an optional maturity word appended to the summary.
            It rides here because a tile's hover HUE cannot carry it alone
            -- colour by itself fails WCAG 1.4.1, and a colour-blind sighted
            reader reads neither the hue nor the accessibility tree.
        :returns: the rich text written, so a test can read it back.
        """
        text = str(summary or "").strip()
        if stage:
            text = f"{text} — {stage}" if text else str(stage)
        self._key = str(key or "")
        # ELIDED TO ONE LINE, because the strip is two lines tall and the
        # second is the links. A module blurb runs to several hundred
        # characters; letting it wrap is what made the strip resize and the
        # page relayout under the pointer. The whole sentence stays in the
        # accessible description below, which is what a screen reader reads.
        html = escape(self._fit(text))
        links = self._links_html(self._key)
        if links:
            html = f"{html}<br>{links}" if html else links
        self.setText(html)
        # The plain sentence stays reachable for a screen reader, which reads
        # the accessible description rather than parsed rich text.
        self.setAccessibleDescription(text)
        self._hold(True)
        return html

    def _fit(self, text: str) -> str:
        """``text`` shortened to the one line the strip has for it.

        Measured against the font Qt is actually painting and the width the
        strip actually has, so it stays correct at any font scale rather than
        at the one this was written on. A strip with no width yet -- asked
        before it is laid out -- gets the text back untouched, because
        eliding to nothing would be worse than a first paint that is long.
        """
        from PySide6.QtCore import Qt as _Qt

        room = self.width() - 16
        if room <= 0:
            return text
        return self.fontMetrics().elidedText(text, _Qt.ElideRight, room)

    def _links_html(self, key: str) -> str:
        """``API`` and ``Tutorial`` as anchors, whichever of them resolve.

        Short words on purpose, and the same shortening instruction 371
        argued for the per-setting strip: "which should also just say API".
        The long forms repeat on every module and the strip is a few lines.

        A word is drawn only where its target exists. Every registry module
        has a lesson today -- measured, 36 of 36 -- but a new module lands in
        the registry before it lands in the lesson catalog, and a Tutorial
        word that goes to an index of seventy-three lessons is worse than no
        word at all.
        """
        from ..i18n import tr
        from ..tutorials import tutorial_url

        parts = []
        api = self._api_url(key)
        if api:
            parts.append(f'<a href="{escape(api, quote=True)}">'
                         f'{escape(tr("API"))}</a>')
        lesson = tutorial_url(key)
        if lesson:
            parts.append(f'<a href="{escape(lesson, quote=True)}">'
                         f'{escape(tr("Tutorial"))}</a>')
        return "&nbsp;&nbsp;".join(parts)

    @staticmethod
    def _api_url(key: str) -> str:
        """``key``'s module page in the API documentation, or ``""``.

        Imported inside the call because `settings_model` is a large module
        and this one is reached from a hover handler on the startup page.
        """
        if not key:
            return ""
        try:
            from ..screens.settings_model import api_docs_url
            return api_docs_url(key)
        except Exception:                                        # noqa: BLE001
            return ""

    # -- the hold --------------------------------------------------------

    def release(self) -> None:
        """Put the default prompt back and stop holding."""
        self._key = ""
        self._hold(False)
        from ..i18n import tr
        self.setText(escape(tr(self._default)))
        self.setAccessibleDescription(tr(self._default))

    def is_holding(self) -> bool:
        """Whether the strip is keeping a module. For tests."""
        return bool(self._timer is not None and self._timer.isActive())

    def _hold(self, holding: bool) -> None:
        """Start, restart or stop the thirty-second hold.

        RESTARTED ON EACH NEW MODULE, so reading across a row of tiles is
        not a race against a clock started by the first one. Stopped outright
        when the strip is being put back to its default, or the timer would
        blank a strip that is already the prompt.
        """
        if self._timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self.release)
            self._timer = timer
        self._timer.stop()
        if holding:
            self._timer.start(self.HOLD_MS)

    def _on_link(self, href: str) -> None:
        self.link_followed.emit(self._key, str(href))
