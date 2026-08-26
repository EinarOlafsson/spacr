"""A label whose hover help carries a module's spaCR API link."""
from __future__ import annotations

from html import escape
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QWidget


class ApiHelpLabel(QLabel):
    """Prose that keeps a documentation link inside its hover help.

    NOTHING IS DRAWN BESIDE THE TEXT. The link is the last line of the
    sticky popup, which is where every setting's API link already lives —
    see :func:`spacr.qt.screens.settings_model.format_tooltip`, which
    builds the same help this label shows. Hovering the prose is the
    affordance; the help cursor says so.

    The destination is not fixed when the label is built:

    * a workbench whose two tabs speak for two modules repoints it as the
      tabs change, through :meth:`set_api_app_key`;
    * a language change repoints it at the translated documentation pages,
      through :meth:`set_url` — the call the language pass makes on any
      widget carrying a ``moduleApiAppKey`` property, which this label
      does — and rebuilds the prose through
      :meth:`retranslate_dynamic_content`.

    Given no app key there is no module page to reach, so the help is the
    description alone: an empty link is worse than none.
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
        self._refresh_help()

    # -- what the label speaks for -----------------------------------------

    def set_api_app_key(self, app_key: str) -> None:
        """Point the help at another module's documentation.

        Used by a screen serving more than one module from a single
        masthead. The property moves with it, so a later language change
        repoints the help at the module now on screen rather than at the
        one the masthead was built for.
        """
        self._app_key = str(app_key or "")
        self._url_override = ""
        self.setProperty("moduleApiAppKey", self._app_key or None)
        self._refresh_help()

    # -- the link ----------------------------------------------------------

    def url(self) -> str:
        """The documentation URL the help currently offers, or ``""``.

        Read back out of the composed help rather than from a field beside
        it: what the reader can reach is the only honest answer.
        """
        from .hover_tooltip import split_api_link
        return split_api_link(self.help_html())[1]

    def set_url(self, url: str) -> None:
        """Send the help's link to ``url`` without rebuilding the prose."""
        self._url_override = str(url or "")
        self._refresh_help()

    def help_html(self) -> str:
        """The rich-text help shown on hover and read by screen readers."""
        return str(self.property("apiTooltipHtml") or "")

    def retranslate_dynamic_content(self, language: object) -> None:
        """Rebuild the help — prose, link caption and page — in ``language``.

        Called by :func:`spacr.qt.i18n.retranslate_widget_tree` after it has
        already repointed the URL, so the language pass and this agree on
        the destination.
        """
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
