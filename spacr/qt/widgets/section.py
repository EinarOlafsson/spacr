"""
Section — a collapsible group with a header row (chevron + title) and
a QFormLayout body that expands/collapses on click. Used by settings
screens to group related fields; every section is collapsed by
default so users see one row per category instead of a wall of
controls.

A HEADING CAN ALSO SAY WHOSE SETTINGS THESE ARE. A module folded into
another one keeps its settings and loses its tile, and where the fold
made those settings a category rather than a page there is no button
left to carry the picture the user learned the module by. It goes on
the heading instead — :meth:`Section.set_source_app`, which draws the
module's own icon at the trailing end of the header row.
"""
from __future__ import annotations

import re
from typing import Optional, Union

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr
from ..theme import SPACING, STAGE_LABEL, STAGE_NOTE

#: Edge of the module mark drawn beside a category heading, in logical px.
#:
#: Smaller than the 20 px a fold button carries, because this one sits
#: inside a 34 px header next to text rather than alone on a 34 px plate.
SOURCE_ICON_PX = 18

#: The objectName the mark carries, so a screen (or a test) can find it.
SOURCE_ICON_NAME = "SectionSourceIcon"


def module_mark(key: str):
    """Return the specific icon for a folded module, if available.

    Generic fallback artwork is not returned because it does not identify the
    source module.

    :param key: Folded module registry key.
    """
    from .. import iconset

    try:
        has_art = iconset.bundled_icon_path(key) is not None
        # The glyph table is the second place a key can have a mark of
        # its own. Read defensively: without it this degrades to "bundled
        # artwork only", which is still a real mark rather than a guess.
        glyphs = getattr(iconset, "_NAME_TO_GLYPH", {}) or {}
        if not has_art and key not in glyphs:
            return None
        mark = iconset.app_icon(key)
    except Exception:                                   # noqa: BLE001
        return None
    if mark is None or mark.isNull():
        return None
    return mark


class Section(QFrame):
    """Collapsible section with an animated chevron header + form body."""

    toggled = Signal(bool)

    def __init__(self, title: str, parent=None, expanded: bool = False):
        super().__init__(parent)
        self.setObjectName("SectionCard")
        self._expanded = False
        self._maturity = "stable"
        self._hint = ""
        self._row_widgets = []
        #: The folded module these settings came from, once one claims
        #: them, and the mark that says so. Empty on a category the host
        #: module wrote itself, which is most of them.
        self._source_app = ""
        self._source_mark = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header
        self._header = QToolButton(self)
        self._header.setObjectName("SectionHeader")
        # A QToolButton reads '&' as a mnemonic marker, so the categories
        # named "Plate Layout & Controls" and "Embedding & Clustering"
        # rendered as "PLATE LAYOUT _CONTROLS" -- the ampersand swallowed and
        # the following letter underlined as an accelerator that goes
        # nowhere. Section headers are not keyboard shortcuts, so the '&' is
        # escaped for display. `title()` still answers with the real text.
        #
        # THE CATALOG IS KEYED ON THE WRITTEN CATEGORY NAME, so the source is
        # kept as the caller wrote it and uppercased only on the way to the
        # button. Looking up a caption that has already been uppercased finds
        # nothing and leaves the header in English.
        self._title_source = str(title)
        self._title = self._title_source.upper()
        # The caption is composed -- the category, and for beta or alpha a
        # maturity badge -- so the generic language pass would ask for the
        # finished line as one key and never find it. Keep that pass off the
        # button and rebuild the caption from the translated parts whenever
        # the language changes.
        self._header.setProperty("i18nSkipText", True)
        self._header.retranslate_dynamic_content = self._refresh_header_text
        self._refresh_header_text()
        self._header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._header.setArrowType(Qt.RightArrow)
        self._header.setCheckable(True)
        self._header.setChecked(False)
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._header.setMinimumHeight(34)
        self._header.clicked.connect(self._on_toggle)
        outer.addWidget(self._header)

        # Body
        self._body = QWidget(self)
        self._body.setObjectName("SectionBody")
        self._form = QFormLayout(self._body)
        self._form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._form.setFormAlignment(Qt.AlignTop)
        self._form.setContentsMargins(SPACING["md"], SPACING["md"],
                                       SPACING["md"], SPACING["md"])
        self._form.setHorizontalSpacing(SPACING["md"])
        self._form.setVerticalSpacing(SPACING["sm"])
        self._body.setVisible(False)
        outer.addWidget(self._body)

        if expanded:
            self.set_expanded(True)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def add_row(
        self,
        label: Union[str, QWidget],
        widget: QWidget,
        info_widget: Optional[QWidget] = None,
        wrap_label: bool = False,
    ) -> None:
        """Add a labeled row, optionally with an information-link icon.

        :param label: text or widget shown on the row's label side.
        :param widget: setting control shown on the row's field side.
        :param wrap_label: build the ``SettingLabelWithInfo`` host even with
            no info widget to put in it. The host is what right-aligns the
            label against its field; it used to arrive only as a side effect
            of there being a dot, so removing the dot from the settings form
            left every label left-aligned and turned half of each row's
            width into page showing through rather than category surface.
            ``_row_widgets`` still records the caller's label, not the host,
            so anything reading rows back gets the ``QLabel`` it passed in.
        """
        form_label = label
        if info_widget is not None or wrap_label:
            form_label = QWidget(self._body)
            form_label.setObjectName("SettingLabelWithInfo")
            label_row = QHBoxLayout(form_label)
            label_row.setContentsMargins(0, 0, 0, 0)
            label_row.setSpacing(SPACING["xs"])
            label_row.addStretch(1)
            if isinstance(label, QWidget):
                label_row.addWidget(label)
            else:
                label_row.addWidget(QLabel(str(label), form_label))
            if info_widget is not None:
                label_row.addWidget(info_widget)
        self._form.addRow(form_label, widget)
        self._row_widgets.append((label, widget))
        self._apply_maturity(label, setting=True)
        self._apply_maturity(widget, setting=True)

    def add_prose_row(self, label: Union[str, QWidget],
                      widget: QWidget, *, at_top: bool = False) -> None:
        """A labelled row that is NOT a setting.

        The difference from :meth:`add_row` is `_row_widgets`, for the reason
        :meth:`add_prose` gives: every entry there is taken to BE a labelled
        setting by the module smoke test, which asserts each field carries a
        ``settingKey`` and that its label holds linked API help. A row of
        Download buttons is neither.

        It still goes through the section's own QFormLayout, so its label sits
        in the same column and at the same right-aligned edge as every setting
        above and below it -- which is the point: a row of buttons floating in
        the middle of the section reads as unrelated to the settings it acts
        on, and one aligned with them reads as part of the same form.
        """
        form_label = QWidget(self._body)
        form_label.setObjectName("SettingLabelWithInfo")
        label_row = QHBoxLayout(form_label)
        label_row.setContentsMargins(0, 0, 0, 0)
        label_row.setSpacing(SPACING["xs"])
        label_row.addStretch(1)
        label_row.addWidget(label if isinstance(label, QWidget)
                            else QLabel(str(label), form_label))
        if at_top:
            # `insertRow(0, ...)`, the same mechanism :meth:`add_prose`
            # uses. Asked for on 2026-09-02 for Regression's Input Tables:
            # "the input tables sould start with download buttons not end
            # wit them" -- a row of buttons that fills the fields below it
            # reads as a footer when it sits under them.
            self._form.insertRow(0, form_label, widget)
        else:
            self._form.addRow(form_label, widget)

    def add_widget(self, widget: QWidget) -> None:
        """Add a full-width (label-less) widget to the section's form body."""
        self._form.addRow(widget)
        self._row_widgets.append((None, widget))
        self._apply_maturity(widget, setting=True)

    def add_prose(self, widget: QWidget, *, at_top: bool = False) -> None:
        """Add full-width content that is NOT a setting row.

        :param widget: prose or other non-setting content to add.
        :param at_top: put it above the section's controls rather than below.

        THE DIFFERENCE FROM :meth:`add_widget` IS `_row_widgets`, and it is
        the whole reason this exists. Every entry there is taken to BE a
        labelled setting row by
        ``tests/qt/test_all_module_smoke.py::_setting_row_contract``, which
        asserts each field carries a ``settingKey`` and that its label is a
        QLabel holding linked API help. A prose box is neither a setting nor
        labelled, so registering it there would either fail that contract or
        force a fake ``settingKey`` onto a non-setting -- which would then be
        pushed into the tooltip and API-documentation machinery.

        `add_widget` has the same signature and the opposite bookkeeping, and
        had no caller in the GUI, so nothing had yet exposed the conflation.
        """
        if at_top:
            self._form.insertRow(0, widget)
        else:
            self._form.addRow(widget)
        self._apply_maturity(widget, setting=True)

    def title(self) -> str:
        """Return the section's header text, un-escaped."""
        return self._title

    def header(self) -> QToolButton:
        """Return the clickable header button (chevron + category title).

        Public because a screen needs a precise hover target for the
        *category* it represents: the section itself covers the whole form
        once expanded, so filtering events on it would report the category
        while the pointer is over one of its settings.
        """
        return self._header

    def set_source_app(self, key: str, name: str = "") -> bool:
        """Associate this category with a folded module and display its icon.

        The icon is drawn separately from the header button so the collapse
        chevron and pointer target remain unchanged.

        :param key: Folded module registry key.
        :param name: Accessible module name. If empty, ``key`` is used.
        :returns: ``True`` if a module-specific icon was displayed.
        """
        key = str(key or "")
        self._source_app = key
        for target in (self, self._header):
            target.setProperty("settingsSourceApp", key)
        mark = module_mark(key) if key else None
        if mark is None:
            if self._source_mark is not None:
                self._source_mark.setVisible(False)
            return False
        badge = self._source_mark
        if badge is None:
            badge = QLabel(self._header)
            badge.setObjectName(SOURCE_ICON_NAME)
            badge.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            badge.setFixedSize(SOURCE_ICON_PX, SOURCE_ICON_PX)
            self._source_row().addWidget(badge, 0, Qt.AlignVCenter)
            self._source_mark = badge
        badge.setPixmap(mark.pixmap(QSize(SOURCE_ICON_PX, SOURCE_ICON_PX),
                                    badge.devicePixelRatioF()))
        badge.setAccessibleName(str(name or key))
        badge.setVisible(True)
        return True

    def source_app(self) -> str:
        """Return the folded module associated with this category."""
        return self._source_app

    def source_mark(self) -> Optional[QWidget]:
        """Return the visible source-icon label, or ``None``."""
        mark = self._source_mark
        if mark is None or not mark.isVisibleTo(self._header):
            return None
        return mark

    def _source_row(self):
        """The header's own layout, made on demand for the module mark.

        Built only when a mark actually arrives, so every category that
        has none is the widget it always was. The right margin matches
        the header's stylesheet padding, so the mark lines up with the
        text that starts on the other side of it.
        """
        row = self._header.layout()
        if row is None:
            row = QHBoxLayout(self._header)
            row.setContentsMargins(0, 0, SPACING["md"], 0)
            row.setSpacing(0)
            # The heading's own text is painted by the button, under this
            # layout; the stretch keeps the mark off it and against the
            # trailing edge.
            row.addStretch(1)
        return row

    def set_hint(self, text: str) -> None:
        """Attach a hover tooltip to the section's header.

        The tooltip appears when the user hovers the header, whether
        the section is currently expanded or collapsed — same UX as
        every other Qt tooltip.

        :param text: tooltip text (plain or HTML; empty clears it).
        """
        self._hint = text or ""
        self._refresh_tooltip()

    def set_maturity(self, stage: str) -> None:
        """Colour this section and every setting row by maturity stage.

        ``stable``/``beta``/``alpha`` use the exact hues shown in Home's
        maturity legend. Unknown values deliberately fall back to stable.
        """
        stage = str(stage or "stable").lower()
        if stage not in STAGE_LABEL:
            stage = "stable"
        self._maturity = stage
        for target in (self, self._header, self._body):
            self._apply_maturity(target)
        for label, widget in self._row_widgets:
            self._apply_maturity(label, setting=True)
            self._apply_maturity(widget, setting=True)
        self.setAccessibleDescription(
            f"{STAGE_LABEL[stage]} maturity settings section"
        )
        self._refresh_header_text()
        self._refresh_tooltip()

    def maturity(self) -> str:
        """Return ``stable``, ``beta`` or ``alpha`` for this section."""
        return self._maturity

    def set_expanded(self, on: bool) -> None:
        """Expand or collapse the section body programmatically."""
        self._header.setChecked(on)
        self._on_toggle(on)

    def is_expanded(self) -> bool:
        """Return True when the section body is currently visible."""
        return self._expanded

    def _apply_maturity(self, widget, *, setting: bool = False) -> None:
        if not isinstance(widget, QWidget):
            return
        prop = "settingMaturity" if setting else "maturity"
        widget.setProperty(prop, self._maturity)
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)

    def _refresh_header_text(self, language: Optional[str] = None) -> None:
        text = tr(self._title_source, language).upper()
        if self._maturity != "stable":
            stage = tr(STAGE_LABEL[self._maturity], language).upper()
            # Category names historically carried their own ``(BETA)`` or
            # were simply named ``Beta``. Maturity styling then appended a
            # second ``· BETA`` badge, producing ``BETA · BETA``. Keep the
            # original title for configuration lookups, but render one badge.
            # Both spellings are stripped: a translated caption can still
            # carry the English badge if the catalog left that word alone.
            for badge in dict.fromkeys(
                    (stage, STAGE_LABEL[self._maturity].upper())):
                text = re.sub(
                    rf"\s*(?:\(\s*{re.escape(badge)}\s*\)|{re.escape(badge)})\s*$",
                    "",
                    text,
                    flags=re.IGNORECASE,
                ).strip()
            text = f"{text}   ·   {stage}" if text else f"·   {stage}"
        text = text.replace("&", "&&")
        self._header.setText(text)

    def _refresh_tooltip(self) -> None:
        # Stable is the normal case, so preserve existing curated tooltips
        # byte-for-byte. Beta/alpha need the caution text because their colour
        # carries information the old tooltip did not.
        note = (
            STAGE_NOTE.get(self._maturity, "")
            if self._maturity != "stable"
            else ""
        )
        parts = [part for part in (self._hint, note) if part]
        self._header.setToolTip("\n\n".join(parts))

    # ------------------------------------------------------------------
    def _on_toggle(self, checked: bool) -> None:
        self._expanded = bool(checked)
        self._header.setArrowType(Qt.DownArrow if self._expanded else Qt.RightArrow)
        self._body.setVisible(self._expanded)
        self.toggled.emit(self._expanded)
