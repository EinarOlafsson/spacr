"""Labels with hover-accessible links to spaCR API documentation."""
from __future__ import annotations

from html import escape
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QLabel, QWidget


class ApiHelpLabel(QLabel):
    """Display descriptive text with an API link in its hover help.

    The link follows the active module and interface language. Labels without
    an application key retain their description but omit the documentation
    link. Hover content is also exposed through the Qt accessibility tree.

    IT ELIDES RATHER THAN CLIPS, which is the difference between a sentence
    the reader can SEE is unfinished and one that simply stops. This label is
    the module masthead's blurb, and the masthead deliberately keeps it to
    one line that "may shrink below its ideal width rather than force the
    window wider" -- so being cut short is designed, and the hover help is
    where the rest is meant to live.

    A `wordWrap(False)` QLabel gives none of that away: Qt paints as many
    characters as fit and stops, with no ellipsis and no hint there is more.
    Measured by instruction 350's sweep on Classify in German, where the
    blurb is 1,354 px of text on a 1,281 px line -- 73 px, about six
    characters, gone without a mark. English fits, which is why it was
    invisible until a second locale was measured.
    """

    def __init__(self, text: str = "", app_key: str = "",
                 parent: Optional[QWidget] = None):
        super().__init__(str(text), parent)
        self._description = str(text)
        self._app_key = str(app_key or "")
        self._url_override = ""
        self._language: Optional[str] = None
        self._help_filter = None
        if self._app_key:
            self.setProperty("moduleApiAppKey", self._app_key)
        self._full_description = str(text)
        self._refresh_help()

    # -- painting as much as fits ------------------------------------------

    def setText(self, text: str) -> None:          # noqa: N802 (Qt casing)
        """Remember the whole sentence, then paint as much of it as fits."""
        self._full_description = str(text or "")
        QLabel.setText(self, self._full_description)
        self._elide_to_fit()

    def full_text(self) -> str:
        """The complete description, however much is being painted."""
        return getattr(self, "_full_description", QLabel.text(self))

    def resizeEvent(self, event):                  # noqa: N802 (Qt naming)
        """Re-elide for the width just granted."""
        super().resizeEvent(event)
        self._elide_to_fit()

    def _elide_to_fit(self) -> None:
        """Paint the full text when it fits, an elided copy when it does not.

        ONLY WHEN THE LABEL DOES NOT WRAP. A wrapping label uses its HEIGHT
        for the overflow, and eliding one would throw away a line it had room
        to draw.
        """
        full = getattr(self, "_full_description", "")
        if not full or self.wordWrap():
            return
        margins = self.contentsMargins()
        room = self.width() - margins.left() - margins.right()
        if room <= 0:
            return
        metrics = QFontMetrics(self.font())
        shown = (full if metrics.horizontalAdvance(full) <= room
                 else metrics.elidedText(full, Qt.ElideRight, room))
        if shown != QLabel.text(self):
            QLabel.setText(self, shown)

    # -- what the label speaks for -----------------------------------------

    def set_api_app_key(self, app_key: str) -> None:
        """Set the module whose API documentation is linked."""
        self._app_key = str(app_key or "")
        self._url_override = ""
        self.setProperty("moduleApiAppKey", self._app_key or None)
        self._refresh_help()

    # -- the link ----------------------------------------------------------

    def url(self) -> str:
        """Return the documentation URL in the current hover content."""
        from .hover_tooltip import split_api_link
        return split_api_link(self.help_html())[1]

    def set_url(self, url: str) -> None:
        """Override the documentation URL while preserving the description."""
        self._url_override = str(url or "")
        self._refresh_help()

    def help_html(self) -> str:
        """Return the rich-text help used for hover and accessibility."""
        return str(self.property("apiTooltipHtml") or "")

    def retranslate_dynamic_content(self, language: object) -> None:
        """Rebuild translated hover content and its documentation link."""
        self._language = str(language) if language else None
        self._url_override = ""
        self._refresh_help()

    # -- internals ---------------------------------------------------------

    def _compose_help(self) -> str:
        description = self._description.strip()
        if not self._app_key:
            # Nothing to link to. `format_tooltip` would fall back to the
            # documentation index, which is a link that answers no question
            # the reader asked.
            return escape(description)

        from ..screens.settings_model import format_tooltip
        from .hover_tooltip import split_api_link

        html = format_tooltip(description, self._app_key, "", self._language)
        if not self._url_override:
            return html
        body, current = split_api_link(html)
        if current == self._url_override:
            return html

        from ..i18n import tr
        caption = escape(tr("Open spaCR API documentation", self._language))
        href = escape(self._url_override, quote=True)
        link = f'<a href="{href}">{caption}</a>'
        return f"{body}<br>{link}" if body else link

    def _refresh_help(self) -> None:
        html = self._compose_help()
        self.setProperty("apiTooltipHtml", html)
        # Kept on the widget as well as in the popup: this string is what the
        # accessibility tree reads out.
        self.setToolTip(html)
        self.setToolTipDuration(-1)
        # The cursor is the affordance the dot used to be: it says there is
        # something here to read before the popup appears.
        self.setCursor(Qt.WhatsThisCursor)
        self._install_help_filter()

    def _install_help_filter(self) -> None:
        """Show the clickable sticky popup on hover, once.

        Qt keeps a LIST of event filters and calls each installation
        separately, so a second filter object would pop two tooltips for one
        hover. One is built per label and reinstalled idempotently.
        """
        if self._help_filter is None:
            from ..screens.settings_model import _ApiTooltipFilter
            self._help_filter = _ApiTooltipFilter(self)
        self.removeEventFilter(self._help_filter)
        self.installEventFilter(self._help_filter)
