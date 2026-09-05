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

from PySide6.QtCore import QEvent, QSize, Qt, Signal
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

#: Floor for a category header's height, in logical px.
#:
#: A FLOOR AND NOT A HEIGHT. It is what a header gets when its own size hint
#: is smaller than a comfortable pointer target; a header whose font asks for
#: more keeps what it asks for. See :meth:`Section._sync_header_minimum` for
#: what happened while this was the header's flat minimum.
SECTION_HEADER_MIN_PX = 34


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
        # See the note in `fold_strip.py`: `iconset.app_icon` knows
        # nothing of `_ICON_OVERRIDES`, so a module that borrows
        # another's picture gets the wrong file. This heading marks a
        # folded module's settings and must match its button.
        from ..app import _icon_for_app
        mark = _icon_for_app(key)
    except Exception:                                   # noqa: BLE001
        return None
    if mark is None or mark.isNull():
        return None
    return mark


class Section(QFrame):
    """Collapsible section with an animated chevron header + form body.

    :param title: the heading text, which is also what the hover strip looks
        this section's help up by.
    :param parent: parent widget.
    :param expanded: whether it opens unfolded. Sections start CLOSED by
        default because a screen that opens every one of them is a wall of
        settings before the user has chosen anything.
    """

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
        # NO POPUP OVER A CATEGORY. The blurb is already shown in the strip
        # under the actions row whenever a header is hovered, so Qt's own
        # tooltip put the same words in a second place -- a tall window that
        # follows the pointer and covers the settings underneath the one
        # being read. The TEXT stays on the widget: assistive technology
        # reads `toolTip()`, and so do the checks that assert a category
        # explains itself. Only the popup is refused.
        self._header.installEventFilter(self)
        self._header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._sync_header_minimum()
        self._header.clicked.connect(self._on_toggle)
        outer.addWidget(self._header)

        # Body
        self._body = QWidget(self)
        self._body.setObjectName("SectionBody")
        self._form = QFormLayout(self._body)
        # SET EXPLICITLY, because the default is the STYLE's answer and not
        # every style answers the same way. Issue 115 reported "field and
        # setting do not expand with container" on macOS. Measured: with
        # Fusion, a 1,178 px section gives its QLineEdit 1,115 px; under a
        # style whose SH_FormLayoutFieldGrowthPolicy is FieldsStayAtSizeHint
        # -- hostile, but a valid Qt answer, and the shape the reporter's
        # platform style chose -- the same section gives the field 108 px.
        # Naming the policy here takes the decision away from the platform.
        self._form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._form.setFormAlignment(Qt.AlignTop)
        self._form.setContentsMargins(SPACING["md"], SPACING["md"],
                                       SPACING["md"], SPACING["md"])
        self._form.setHorizontalSpacing(SPACING["md"])
        self._form.setVerticalSpacing(SPACING["sm"])
        self._body.setVisible(False)
        outer.addWidget(self._body)
        self._seal_body()

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
        if isinstance(label, QWidget):
            label_row.addWidget(label)
        else:
            # AN ELIDING LABEL, NOT A BARE ONE, and the difference only shows
            # in a translation. The label column is as wide as its widest
            # label's hint, and every settings label in it elides -- so the
            # column is capped, and a plain QLabel wider than the cap is cut
            # off mid-glyph rather than shortened. Measured on Regression in
            # German: "Herunterladen" wants 83 px, the column grants 58, and
            # English "Download" needs 57 and fits exactly, which is why it
            # was invisible until the sweep of instruction 350 ran in a
            # second locale.
            #
            # `ElidingLabel.sizeHint` still asks for the FULL width, so where
            # the column can afford it nothing is elided at all; the tooltip
            # carries the whole word for when it cannot.
            from .eliding import ElidingLabel

            text = str(label)
            prose = ElidingLabel(text, form_label)
            prose.setToolTip(text)
            label_row.addWidget(prose)
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

    def eventFilter(self, watched, event):                   # noqa: N802
        """Swallow the header's tooltip request; pass everything else on.

        The header's own minimum is re-taken here on the two events that can
        change what it needs -- a new font, and a new style -- because both
        arrive after the button is built. See :meth:`_sync_header_minimum`.

        :param watched: the object the event is for.
        :param event: the event.
        :returns: True to stop a tooltip from being shown.
        """
        if watched is self._header:
            if event.type() == QEvent.ToolTip:
                return True
            if event.type() in (QEvent.FontChange, QEvent.StyleChange,
                                QEvent.Polish):
                self._sync_header_minimum()
        return super().eventFilter(watched, event)

    def _seal_body(self) -> None:
        """Let Qt know the body is opaque, when the card really is opaque.

        MEASURED on the mask screen, 3840x2160, font_scale 2, blobs backdrop
        at 12 fps: the animation reaches 34.3% of the window and 0.0% of a
        section body -- the card's own fill covers every pixel of it. The
        body and its sixty-odd fields were repainting 12 times a second
        anyway, because the card's fill comes from the stylesheet and a QSS
        background is not something ``QWidgetRepaintManager`` can subtract
        damage against. See :func:`spacr.qt.theme.seal_surface` for the pair
        of measurements that establishes that.

        Sealing the BODY rather than the card on purpose: the card's rect
        includes its ``margin-bottom``, which is the gutter between two
        categories and is where the backdrop legitimately shows through, so
        the card cannot promise its whole rect. The body can.

        The seal undoes itself when ``surface`` is translucent at the user's
        page-opacity setting, which is the case the maintainer asked for on
        the dock and the panels.
        """
        try:
            from ..theme import seal_surface
            seal_surface(self._body, role="surface")
        except Exception:                                    # noqa: BLE001
            pass

    def changeEvent(self, event):
        """Re-seal the body when the stylesheet or the palette is swapped.

        A theme change or a move of the page-opacity slider re-renders
        ``surface``; the seal is a palette brush this widget owns, so nothing
        else re-computes it. Only ``StyleChange`` is answered -- setting the
        body's palette posts ``PaletteChange`` back here, and answering that
        one would be a loop.
        """
        super().changeEvent(event)
        if event.type() == QEvent.StyleChange:
            self._seal_body()

    def _sync_header_minimum(self) -> None:
        """Keep the header from being squeezed below the height it needs.

        `QSizePolicy.Fixed` is not the last word on how short a widget may
        be made: `qSmartMinSize` computes a minimum from the policy and the
        hints, and then an explicitly set `minimumHeight` REPLACES it,
        downwards as readily as upwards. So the flat `setMinimumHeight(34)`
        this used to carry did not raise a floor, it granted the layout
        permission to shrink the header to 34 px.

        That permission is taken up on every expand. Showing a section body
        makes the settings column taller than the widget the scroll area has
        given it -- the scroll area only resizes that widget on the NEXT
        pass -- so the first pass distributes too little height and shrinks
        every category header to its stated minimum. MEASURED on the mask
        screen: at font_scale 1 the headers want 36 px and drop to 34, which
        is why this went unseen for so long; at font_scale 2 they want 52 and
        drop to 34, and 11 of the 19 headers on the panel took an 18 px dip
        and came back one layout pass later. That is 22 of the 175 resize
        events an expand costs, and a window repaint landing inside the
        ~2 ms both passes take draws every heading short.

        34 stays as a floor for the opposite case -- a hint smaller than a
        comfortable pointer target, which is what the constant was for.
        Re-taken on font and style changes because the hint is the polished
        button's, and at construction the stylesheet has not been applied.
        """
        wanted = max(SECTION_HEADER_MIN_PX,
                     self._header.sizeHint().height())
        # Guarded: `setMinimumHeight` invalidates the layout, and this runs
        # from inside style and font delivery, where an unconditional write
        # would post a layout request on every polish.
        if self._header.minimumHeight() != wanted:
            self._header.setMinimumHeight(wanted)

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
