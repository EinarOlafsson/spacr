"""HomePage — the Home screen.

    ┌────────────────────────────────────────────────────────────────┐
    │ 🖼 spaCR   End-to-end microscopy → single-cell measurements …   │
    │ ┌ Mask · running ────── 41 of 96 ──── [Open] [Pause] ────────┐ │
    │ │ Home │ Core │ Data │ Segmentation models │ Results │ Toxo  │ │  QUEUED
    │ │ CORE 9 ────────────────────────────────────────────────── │ │  RECENT
    │ │  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐                       │ │  SYSTEM
    │ │  │ ▧  │ │ ▧  │ │ ▧  │ │ ▧  │ │ ▧  │                       │ │  NEWS
    │ │  │Mask│ │Time│ │Moti│ │Meas│ │Anno│                       │ │  TOTALS
    │ │  └────┘ └────┘ └────┘ └────┘ └────┘                       │ │  ────────
    │ │ DATA 6 ───────────────────────────────────────────────────│ │  ● Alpha
    │ │  ┌────┐ ┌────┐ …                                          │ │  ● Beta
    │ └────────────────────────────────────────────────────────────┘ │  ● Stable
    │  Hover a tile to see what it does.                             │
    └────────────────────────────────────────────────────────────────┘

The first tab displays every registered app grouped into the same sections as
the category-filter tabs. :class:`AppTile` supplies one consistent icon-and-
name tile in every view; tooltips and the hint bar provide descriptions.
``categories`` and ``bands`` remain separate inputs so navigation and the
all-app layout can evolve independently while both derive from
:data:`spacr.qt.app.APPS`.

Maturity is represented by each tile's ``stage`` property and the legend,
using :data:`spacr.qt.app.APP_STAGE` and
:data:`spacr.qt.theme.STAGE_HOVER`. The right column presents state rather
than navigation: queued work, recent runs, system information, release news,
totals, and maturity labels.

Home subscribes to :data:`spacr.qt.bridge.registry` to display jobs started by
any screen. :class:`RunningBanner` exposes only the controls supported by the
worker; cooperative Pause remains disabled when the pipeline has no
:class:`spacr.qt.bridge.PauseGate` checkpoints. Widget colors are resolved
from :func:`spacr.qt.theme.active_palette` so runtime theme changes remain
consistent.
"""
from __future__ import annotations

import os
import re
import warnings
from html import escape
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import QEvent, QObject, QRect, QRectF, QSize, Qt, QTimer, Signal, Slot
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QIcon,
    QPainter,
    QPixmap,
    QTextDocumentFragment,
)
from PySide6.QtGui import QSyntaxHighlighter as _SyntaxHighlighter
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..hidpi import device_ratio, follow_device_ratio, scaled_for
from ..theme import (
    QSS_FONT_FAMILY,
    SPACING,
    TILE_H,
    TILE_ICON_PX,
    TILE_MAX_W,
    TILE_W,
    font_px,
    make_transparent,
    palette_for,
)
from .divider import Divider
from .height_grip import HeightGrip
from .workflow_diagram import WorkflowView as _WorkflowView
from .workflow_diagram import _Edge as _WorkflowEdge
from .workflow_diagram import _Node as _WorkflowNode

#: Hero brand sizes. The mark and wordmark are the first thing on the first
#: screen, so they are sized to read as a masthead rather than as a row of
#: labels. Kept as named constants because the logo's pixmap scale and the
#: label's font size have to move together to stay optically balanced.
HERO_LOGO_PX = 72
HERO_TITLE_PX = 52


class _AsideFloorFit(QObject):
    """Refresh the aside floor after its real styled content changes."""

    def __init__(self, panels, fit):
        """Own one coalesced post-layout callback on the panels being measured."""
        super().__init__(panels)
        self._fit_floor = fit
        self._busy = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._apply)
        panels._floor_timer = self._timer
        panels.installEventFilter(self)
        self._timer.start(0)

    def eventFilter(self, obj, event):
        """Observe only local polish/font/style/layout events, never resize polling."""
        if not self._busy and event.type() in (
                QEvent.Show, QEvent.Polish, QEvent.FontChange,
                QEvent.StyleChange, QEvent.LayoutRequest):
            self._timer.start(0)
        return False

    def _apply(self):
        """Measure once after queued polish while guarding synchronous feedback."""
        self._busy = True
        try:
            self._fit_floor()
        finally:
            self._busy = False


class _HomeBodyFit(QObject):
    """Keep tile access and the masthead inside Home's local scroll bounds."""

    def __init__(self, body, scroll, tabs, reflow):
        """Own coalesced fresh style/layout measurements with this body."""
        super().__init__(body)
        self._tabs = tabs
        self._hero = body.findChild(QWidget, 'Hero')
        self._outer_viewport = scroll.viewport()
        self._viewports = {scroll.viewport(),
                           *(tabs.widget(index).viewport() for index in range(tabs.count()))}
        self._reflow = reflow
        self._busy = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._apply)
        self._outer_viewport.destroyed.connect(self._timer.stop)
        for widget in (body, *self._viewports, tabs, tabs.tabBar(),
                       *(tabs.widget(index).widget() for index in range(tabs.count()))):
            widget.installEventFilter(self)
        self._timer.start(0)

    def eventFilter(self, widget, event):
        """Observe this subtree's native layout/style and viewport resizing."""
        if not self._busy and (event.type() in (
                QEvent.Show, QEvent.Polish, QEvent.FontChange, QEvent.StyleChange,
                QEvent.LayoutRequest) or
                (widget in self._viewports and event.type() == QEvent.Resize)):
            from shiboken6 import isValid

            if isValid(self._outer_viewport):
                self._timer.start(0)
        return False

    def _apply(self):
        """Use fresh row minima and cap only the masthead to its real viewport."""
        from shiboken6 import isValid
        from ..gui_scale import _set_actual_dim

        if not isValid(self._outer_viewport):
            return
        self._busy = True
        try:
            cap = max(1, self._outer_viewport.property('size').height())
            if self._hero.property('maximumSize').height() != cap:
                _set_actual_dim(self._hero, 'maxh', cap)
            self._reflow()
            width = height = 0
            for index in range(self._tabs.count()):
                scroll = self._tabs.widget(index)
                tiles = scroll.findChildren(AppTile)
                if not tiles:
                    continue
                left, top, right, bottom = scroll.widget().layout().getContentsMargins()
                row_width = max(max(tile.property('minimumSize').width(),
                                    tile.layout().totalMinimumSize().width()) for tile in tiles)
                row_height = max(tile.layout().totalMinimumSize().height() for tile in tiles)
                frame = scroll.frameWidth() * 2
                width = max(width, row_width + left + right + frame
                            + scroll.verticalScrollBar().sizeHint().width())
                height = max(height, row_height + top + bottom + frame)
            width = max(width, self._tabs.tabBar().minimumSizeHint().width())
            height += self._tabs.tabBar().sizeHint().height()
            height += max(0, self._tabs.rect().height()
                          - self._tabs.currentWidget().rect().height()
                          - self._tabs.tabBar().sizeHint().height())
            current = self._tabs.property('minimumSize')
            for key, wanted, actual in (('minw', width, current.width()),
                                        ('minh', height, current.height())):
                if wanted != actual:
                    _set_actual_dim(self._tabs, key, wanted)
        finally:
            self._busy = False


class _HeroMark(QLabel):
    """Keep the source mark while permitting the masthead to compress it."""

    def __init__(self, source, preferred_side):
        """Retain the original pixels and a logical preferred branding size."""
        super().__init__()
        self._source = QPixmap(source)
        self._preferred_side = preferred_side
        self._drawn = None
        follow_device_ratio(self, self._redraw)

    def sizeHint(self):
        """Offer the selected font/GUI size without making it a hard floor."""
        from ..gui_scale import scale_int

        side = scale_int(font_px(self._preferred_side))
        return QSize(side, side)

    def minimumSizeHint(self):
        """Allow the actual header allocation to be smaller than its preference."""
        return QSize(0, 0)

    def _redraw(self):
        """Fit original source pixels to actual bounds, preserving ratio and DPR."""
        bounds = self.contentsRect().size().boundedTo(self.sizeHint())
        signature = (bounds.width(), bounds.height(), device_ratio(self))
        if not bounds.isEmpty() and signature != self._drawn:
            self.setPixmap(scaled_for(self._source, self, bounds))
            self._drawn = signature

    def resizeEvent(self, event):
        """Redraw after the label receives its actual allocation."""
        super().resizeEvent(event)
        self._redraw()


class _HeroLabel(QLabel):
    """Retain selected-font preferred space without imposing a minimum."""

    def sizeHint(self):
        """Let a larger hero restore the font instead of retaining fitted hints."""
        return getattr(self, '_hero_preferred_hint', None) or super().sizeHint()

    def heightForWidth(self, width):
        """Prefer selected-font wrapped height, not the last fitted allocation."""
        selected = getattr(self, '_hero_selected_font', None)
        if not self.wordWrap() or selected is None:
            return super().heightForWidth(width)
        contents = self.contentsRect()
        horizontal = self.rect().width() - contents.width()
        vertical = self.rect().height() - contents.height()
        bounds = QRect(0, 0, max(1, width - horizontal), 1 << 20)
        flags = int(self.alignment() | Qt.TextWordWrap)
        return QFontMetrics(selected).boundingRect(bounds, flags, self.text()).height() + vertical


class _HeroTextFit(QObject):
    """Fit only the masthead text, retaining its selected styles for growth."""

    def __init__(self, hero, labels):
        """Coalesce actual slot and selected-font changes on the owned hero."""
        super().__init__(hero)
        self._labels = labels
        self._selected = {}
        self._busy = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fit)
        hero.installEventFilter(self)
        for label in labels:
            label.installEventFilter(self)
        self._timer.start(0)

    def eventFilter(self, obj, event):
        """Schedule one fit without recursing through our own style changes."""
        if not self._busy and event.type() in (
                QEvent.Resize, QEvent.FontChange, QEvent.StyleChange,
                QEvent.Polish, QEvent.LayoutRequest):
            self._timer.start(0)
        return False

    @staticmethod
    def _restyle(label, sheet, original_set):
        """Give ``label`` its own ``sheet`` and the font that sheet selects.

        PySide6 6.10 keeps the font an earlier sheet set when a polished
        label's own sheet is cleared, so a fitted ``font-size`` outlived its
        rule and the masthead never grew back. Polishing again resolves the
        font from the sheets that remain, as 6.11 does on its own.
        """
        original_set(label, sheet)
        if not sheet and label.testAttribute(Qt.WA_WState_Polished):
            style = label.style()
            style.unpolish(label)
            style.polish(label)

    def _fit(self):
        """Choose the largest fitting actual pixel size from the selected font."""
        from .. import gui_scale, preferences, theme

        if self._busy:
            return
        original_get = gui_scale._ORIGINAL.get((QWidget, "styleSheet"), QWidget.styleSheet)
        original_set = gui_scale._ORIGINAL.get((QWidget, "setStyleSheet"), QWidget.setStyleSheet)
        app = QApplication.instance()
        window = self.parent().window()
        context = (gui_scale.current_scale(), preferences.get_font_scale(),
                   app.styleSheet(), app.font().toString(),
                   window.property(theme._WINDOW_SHEET_SERIAL),
                   window.property(theme._WINDOW_SHEET_DIGEST))
        self._busy = True
        try:
            for label in self._labels:
                label.ensurePolished()
                sheet = original_get(label)
                previous = self._selected.get(label)
                if (previous is None or previous['context'] != context
                        or sheet != previous['applied']
                        or label.font().toString() != previous.get('applied_font')):
                    base = previous['base'] if previous is not None and sheet == previous['applied'] else sheet
                    if sheet != base:
                        self._restyle(label, base, original_set)
                        label.ensurePolished()
                    previous = dict(base=base, font=QFont(label.font()),
                                    context=context, applied=base)
                    font_changed = getattr(label, '_hero_selected_font', None) != previous['font']
                    label._hero_selected_font = QFont(previous['font'])
                    hint = QLabel.sizeHint(label)
                    if font_changed or getattr(label, '_hero_preferred_hint', None) != hint:
                        label._hero_preferred_hint = hint
                        label.updateGeometry()
                    self._selected[label] = previous
                selected = previous['font']
                maximum = selected.pixelSize()
                if maximum <= 0:
                    maximum = QFontMetrics(selected).height()
                bounds = label.contentsRect()
                if bounds.isEmpty():
                    continue
                flags = int(label.alignment() | (
                    Qt.TextWordWrap if label.wordWrap() else Qt.TextSingleLine))
                signature = (bounds.getRect(), label.text(), flags,
                             label.isEnabled(), selected.toString(), sheet)
                if previous.get('signature') == signature:
                    continue

                def fits(size):
                    """Return whether this font pixel size fits the label content rectangle."""
                    font = QFont(selected)
                    font.setPixelSize(size)
                    text = label.style().itemTextRect(
                        QFontMetrics(font), bounds, flags, label.isEnabled(), label.text())
                    return bounds.contains(text)

                size = maximum
                if not fits(maximum):
                    low, high, size = 1, maximum - 1, 1
                    while low <= high:
                        middle = (low + high) // 2
                        if fits(middle):
                            size, low = middle, middle + 1
                        else:
                            high = middle - 1
                target = previous['base']
                if size < maximum:
                    if '{' in target:
                        target += f'\nQLabel#{label.objectName()} {{ font-size: {size}px; }}'
                    else:
                        target = (target.rstrip().rstrip(';') + '; ' if target else '') + f'font-size: {size}px;'
                if original_get(label) != target:
                    self._restyle(label, target, original_set)
                previous['applied'] = target
                previous['applied_font'] = label.font().toString()
                previous['signature'] = (*signature[:-1], target)
        finally:
            self._busy = False


def active_palette() -> dict:
    """The palette for the theme that is on screen *right now*.

    Not ``theme.PALETTE``: that module-level dict is the dark palette and
    nothing ever updates it, so a widget that inlines colours from it
    renders dark on every theme. On the light theme that produced black
    panels with black text in the right-hand column — unreadable, and
    invisible to any test that only checks widget structure.

    Home is rebuilt from scratch on a theme change
    (``MainWindow._rebuild_startup_page``), so resolving once per widget
    construction is enough.
    """
    try:
        from ..preferences import resolve_effective_theme
        return palette_for(resolve_effective_theme())
    except Exception:
        return palette_for("dark")

_DEFAULT_HINT = "Hover a tile to see what it does."

#: Why the Pause control is disabled. Shown as its tooltip, and asserted
#: by the test suite so it cannot quietly become a lie.
PAUSE_UNAVAILABLE = (
    "Pause is not available for this module.\n\n"
    "Pausing means holding the pipeline at a point where nothing is "
    "half-written — between fields, not mid-write. spaCR's pipelines do "
    "not yet offer such a checkpoint, so a Pause button here could only "
    "freeze the thread wherever it happened to be, which can truncate a "
    "mask file or leave a field measured into some tables and not "
    "others.\n\n"
    "Use Stop, or queue plates so the run can be halted between them."
)

PAUSE_AVAILABLE = "Hold this run at its next safe checkpoint."

#: Appended to the header of an aside panel whose numbers are not yet
#: trusted. Lower-case on purpose: the header is upper-cased and
#: letter-spaced, so a lower-case marker reads as an annotation ON the
#: heading rather than as another word IN it.
BETA_SUFFIX = " (beta)"

#: Why those panels carry it. Shown as the header's tooltip, so the mark
#: is an explanation rather than a shrug.
BETA_PANEL_TOOLTIP = (
    "Beta: this panel is still being worked on and its numbers may be "
    "incomplete or wrong. Nothing else on this page depends on it.")


def _escape_amp(text: str) -> str:
    """Double any ``&`` so Qt draws it instead of eating it.

    ``QTabBar`` (like ``QToolButton``) reads a lone ``&`` as a mnemonic:
    "Results & QC" rendered as "Results  QC" with an underlined Q. No
    tab label carries an ampersand today — that section was retired —
    which is exactly when this stops being applied and the next name
    with one in it ships broken. It stays, and it stays tested.
    """
    return text.replace("&", "&&")


def _find_logo_pixmap() -> Optional[QPixmap]:
    """The bundled spaCR logo, re-inked for the theme, or ``None``."""
    from ..iconset import themed_pixmap

    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in ("logo_spacr.png", "logo_spacr_v1.png"):
        path = os.path.normpath(
            os.path.join(here, "..", "..", "resources", "icons", candidate))
        if os.path.isfile(path):
            pix = themed_pixmap(path) or QPixmap(path)
            if not pix.isNull():
                return pix
    return None




def _fmt_elapsed(seconds: float) -> str:
    """Render an elapsed time compactly.

    :param seconds: the duration; negatives read as zero.
    :returns: seconds, then minutes and seconds, then hours and minutes --
        each with the smaller unit zero-padded so a column of them lines up.
    """
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"



class _AppTileIcon(QLabel):
    """Render the original icon into this label's actual bounds and DPR."""

    def __init__(self, icon, side):
        """Keep the source icon and follow this label's size and device ratio."""
        super().__init__()
        self._icon = QIcon(icon)
        self.setFixedSize(side, side)
        self.setAlignment(Qt.AlignCenter)
        self._redraw()
        follow_device_ratio(self, self._redraw)

    def _redraw(self):
        """Render the source icon at the current content size and device ratio."""
        bounds = self.contentsRect().size()
        if not bounds.isEmpty():
            rendered = self._icon.pixmap(bounds, device_ratio(self))
            self.setPixmap(scaled_for(rendered, self, bounds))

    def resizeEvent(self, event):
        """Refresh the icon after the label handles its resize event."""
        super().resizeEvent(event)
        self._redraw()


class AppTile(QPushButton):
    """**The** tile: a large square-ish button, icon over module name.

    One class for every tab including Home. There used to be two — a
    dense icon-beside-name row for Home and a tall card carrying the
    app's one-line description everywhere else — and the difference
    made the first tab read as a list and the rest as a launcher. The
    description is gone with them: it was a third copy of a sentence
    already on the tooltip and in the hint bar at the foot of the page,
    and three lines of 11 px grey under every tile is what made the
    tiles small enough to need two sizes in the first place.

    Deliberately **not** an :class:`HTile` subclass. ``HTile`` is a
    horizontal row whose name and description live in a ``QLabel``
    stack beside the button's own icon; this is a vertical stack with
    the icon drawn as a child label. It has its own object name,
    ``AppTile``, so the stylesheet can give it a height floor without
    giving one to every horizontal tile in the app.

    **Its height floor is in the QSS, not here.** ``setFixedSize`` does not survive polish, and
    neither does answering through ``sizeHint`` /
    ``minimumSizeHint``: the app stylesheet's blanket
    ``QPushButton { min-height: 22px }`` becomes a real
    ``setMinimumHeight(22)``, and ``qSmartMinSize`` lets an explicit
    minimum override the hints. On a page that does not fit, every tile
    then collapses to 22 px and paints its name over its icon. See
    :data:`spacr.qt.theme.TILE_H`. The hints below are still worth
    having — they are what the layout *prefers* — and
    ``heightForWidth`` is overridden with them because
    ``QWidgetItem::sizeHint`` reads it in preference to
    ``sizeHint().height()`` whenever it is available.

    :param stage: ``stable`` / ``beta`` / ``alpha``. Set as a Qt
        property, which is what the stylesheet's
        ``QPushButton#AppTile[stage="alpha"]:hover`` rule selects on. Set
        *before* the widget is first polished, or the rule does not
        apply until something else forces a repolish.
    """

    def __init__(self, text: str, description: str = "",
                 icon: Optional[QIcon] = None, *, width: int, height: int,
                 icon_px: int = 52, stage: str = "stable", parent=None):
        """Build one tile: an icon over a module name.

        :param text: the module name drawn on the tile, and what the tile is
            identified by.
        :param description: accepted and not drawn: the same sentence is
            already on the tooltip and in the hint bar. Kept in the
            signature so callers that pass it still work.
        :param icon: the module's mark, drawn above the name.
        :param width: tile width in pixels.
        :param height: tile height in pixels.
        :param icon_px: the icon's edge in pixels.
        :param stage: maturity -- ``"stable"``, ``"beta"`` or ``"alpha"``.
            Set as a Qt property, so the theme paints the badge and the
            maturity filter can find the tile without reading its text.
        :param parent: parent widget.
        """
        super().__init__(parent)
        P = active_palette()
        self._text = text
        self._stage = str(stage or "stable")
        self._size = QSize(int(width), int(height))
        self.setObjectName("AppTile")
        self.setProperty("stage", self._stage)
        self.setCursor(Qt.PointingHandCursor)
        self.setAccessibleName(text)
        from ..theme import STAGE_LABEL
        mark = STAGE_LABEL.get(self._stage, "")
        self.setAccessibleDescription(
            f"{mark} — {description}" if description and mark else
            (description or mark))

        col = QVBoxLayout(self)
        col.setContentsMargins(10, 10, 10, 10)
        col.setSpacing(8)
        col.addStretch(1)

        if icon is not None:
            glyph = _AppTileIcon(icon, icon_px)
            glyph.setStyleSheet("background: transparent;")
            col.addWidget(glyph, 0, Qt.AlignHCenter)

        from .eliding import ElidingLabel
        name = ElidingLabel(text)
        name.setAlignment(Qt.AlignHCenter)
        name.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        name.setStyleSheet(
            f"color: {P['fg']}; font-size: {font_px(14)}px; font-weight: 500;"
            "background: transparent;")
        col.addWidget(name)
        self._name_lbl = name
        col.addStretch(1)

    @property
    def text_label(self) -> str:
        """The tile's app name, matching ``HTile.text_label``."""
        return self._text

    @property
    def stage(self) -> str:
        """``stable`` / ``beta`` / ``alpha`` — what the hover colour says."""
        return self._stage

    @property
    def name_label(self):
        """The label showing the module's name.

        Exposed so the text-fits sweep can measure it: a tile that elides
        its own name is a module the user cannot identify.

        :returns: the label.
        """
        return self._name_lbl

    def is_name_elided(self) -> bool:
        """Whether the module name is being cut short to fit.

        :returns: True when the label is eliding.
        """
        return self._name_lbl.is_elided()

    def heightForWidth(self, width: int) -> int:   # noqa: N802
        """At least the tile height, more if a child somehow needs it.

        :param width: proposed tile width in pixels, passed to the base
            implementation.
        """
        natural = super().heightForWidth(width)
        return max(self._size.height(), natural)

    def sizeHint(self) -> QSize:               # noqa: N802
        """The tile's preferred size, height derived from its width.

        Height follows width because the tile is icon-over-name in a fixed
        proportion; asking for a free height would let the grid stretch one
        tile and not its neighbours.

        :returns: the preferred size.
        """
        return QSize(self._size.width(),
                     self.heightForWidth(self._size.width()))

    def minimumSizeHint(self) -> QSize:        # noqa: N802
        """The same as :meth:`sizeHint`: a tile does not shrink below its shape.

        :returns: the minimum size.
        """
        return self.sizeHint()



def panel_box_qss(P: dict) -> str:
    """The rule of a Home panel's box, at the page opacity now in force.

    :param P: the active palette.
    :returns: the box's own QSS.
    """
    from ..theme import pane_surface

    return ("QFrame#HomePanelBox {"
            f"background: {pane_surface('surface_alt')};"
            f"border: 1px solid {P['border_soft']};"
            "border-radius: 8px; }")


def hero_qss() -> str:
    """The masthead's rule, at the page opacity now in force.

    :returns: the masthead's own QSS.
    """
    from ..preferences import resolve_effective_theme
    from ..theme import RADIUS, pane_surface

    return (f"QWidget#Hero {{"
            f" background-color:"
            f" {pane_surface('surface_alt', resolve_effective_theme())};"
            f" border-radius: {RADIUS['lg']}px;"
            f" }}")


class Panel(QWidget):
    """Captioned box for the right-hand column.

    The border lives on a ``QFrame`` with its own object name and the
    rule is scoped to it. An unscoped border rule cascades into every
    child row and outlines each one — a mistake this codebase has
    already made once on the Home dashboard.

    :param beta: mark the header with :data:`BETA_SUFFIX`. The suffix is
        appended *after* the upper-casing, so it stays lower case and
        reads as a mark on the heading rather than part of it.

    :param title: the caption above the panel. Drawn OUTSIDE the panel's
        frame, beside any action word, rather than inside it.
    :param parent: parent widget; ownership only.
    """

    #: Which palette colour an action word turns when the pointer is on it.
    #: `Clear` and `Reset` throw something away and are red; `Refresh` only
    #: re-reads and is blue. Specified as a clear button -- the word "clear" alone, turning red on hover or click)" and "a refresh
    #: button (like clear button but blue)".
    ACTION_INKS = {"danger": "error", "safe": "accent"}

    def __init__(self, title: str, parent=None, *, beta: bool = False):
        """Build a captioned box for the right-hand column.

        :param title: the caption.
        :param parent: parent widget.
        """
        super().__init__(parent)
        P = active_palette()
        self.is_beta = bool(beta)
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(SPACING["xs"])

        head = QWidget()
        head_row = QHBoxLayout(head)
        head_row.setContentsMargins(0, 0, 0, 0)
        head_row.setSpacing(SPACING["sm"])
        self.header = QLabel(title.upper()
                             + (BETA_SUFFIX if self.is_beta else ""))
        self.header.setObjectName("HomePanelHeader")
        self.header.setStyleSheet(
            f"font-family: {QSS_FONT_FAMILY}; font-weight: 600;"
            f"font-size: {font_px(10)}px; letter-spacing: 2px;"
            "background: transparent;"
            f"color: {P['fg_muted']};")
        if self.is_beta:
            self.header.setToolTip(BETA_PANEL_TOOLTIP)
        head_row.addWidget(self.header)
        head_row.addStretch(1)
        self._head_row = head_row
        self._actions: Dict[str, QPushButton] = {}
        col.addWidget(head)
        self._head = head

        box = QFrame()
        box.setObjectName("HomePanelBox")
        box.setStyleSheet(panel_box_qss(P))
        self.body_layout = QVBoxLayout(box)
        self.body_layout.setContentsMargins(SPACING["md"], SPACING["sm"],
                                            SPACING["md"], SPACING["sm"])
        self.body_layout.setSpacing(SPACING["xs"])
        col.addWidget(box)
        self._box = box
        make_transparent(self)
        make_transparent(self._head)

    def add(self, widget: QWidget) -> QWidget:
        """Add a widget to the panel body, transparent so the box shows through.

        The border and fill live on the frame around the body, so a child
        painting its own background would draw a square inside the rounded
        box rather than sitting in it.

        :param widget: the widget to add.
        :returns: the same widget, for chaining.
        """
        widget.setStyleSheet("background: transparent;")
        self.body_layout.addWidget(widget)
        return widget

    def add_action(self, text: str, *, kind: str = "danger",
                   tip: str = "") -> QPushButton:
        """Put an action word on the right of the panel's caption.

        Not a button in the styled sense -- no frame, no fill, no padding
        that would make it look pressable. It is the word alone, in the
        muted caption ink, and it takes its colour when the pointer is on
        it or while it is held down.

        Rendered through a QPushButton rather than a QLabel because the
        word is a control: a button is what Tab reaches, what Space
        activates, what a screen reader announces as pressable, and what
        already has a `:pressed` state to hang the click colour on. A
        clickable QLabel has none of that.

        :param text: the word to draw. Not upper-cased -- the caption
            beside it is, and matching it would make the word read as a
            second heading.
        :param kind: ``danger`` for a word that throws something away, drawn
            red; ``safe`` for one that only re-reads, drawn in the accent.
        :param tip: optional hover help. Also becomes the accessible
            description, because that is what a screen reader reads.
        :returns: the button, so the caller can connect it.
        """
        P = active_palette()
        ink = P[self.ACTION_INKS.get(kind, "error")]
        word = QPushButton(text)
        word.setObjectName("HomePanelAction")
        word.setCursor(Qt.PointingHandCursor)
        word.setFlat(True)
        word.setStyleSheet(
            "QPushButton#HomePanelAction {"
            " background: transparent; border: none; padding: 0px;"
            f" font-family: {QSS_FONT_FAMILY}; font-weight: 600;"
            f" font-size: {font_px(10)}px; letter-spacing: 1px;"
            f" color: {P['fg_dim']}; }}"
            f"QPushButton#HomePanelAction:hover {{ color: {ink}; }}"
            f"QPushButton#HomePanelAction:pressed {{ color: {ink}; }}"
            f"QPushButton#HomePanelAction:disabled {{"
            f" color: {P['fg_dim']}; }}")
        if tip:
            word.setToolTip(tip)
            word.setAccessibleDescription(tip)
        self._head_row.addWidget(word)
        self._actions[text.lower()] = word
        return word

    def action(self, text: str) -> Optional[QPushButton]:
        """The action word named ``text``, or ``None``. For tests.

        :param text: the action's label, matched case-insensitively.
        """
        return self._actions.get(text.lower())

    def _clear_body(self) -> None:
        """Take every row out of the panel's box and delete it.

        The same six lines were written out in five panels; they are here
        because a panel that forgets the `deleteLater` leaks a widget on
        every Home revisit, and Home is revisited constantly.

        Each row is hidden as it leaves, not only queued for deletion: a
        queued row stays a child of the box, painted where it last stood,
        until the event loop next empties its deletion queue. Above 100%
        text size the new rows sit at new heights, so the old ones showed
        through and the panel read as drawn twice.
        """
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()


def _row(label: str, value: str, value_colour: Optional[str] = None,
         mono: bool = False) -> QWidget:
    """One ``label   value`` line inside a :class:`Panel`."""
    P = active_palette()
    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(SPACING["sm"])
    left = QLabel(label)
    left.setStyleSheet(f"color: {P['fg_muted']}; font-size: {font_px(11)}px;"
                       "font-weight: 500; background: transparent;")
    left.setMinimumWidth(48)
    right = QLabel(value)
    right.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    right.setStyleSheet(
        f"color: {value_colour or P['fg']}; font-size: {font_px(12)}px;"
        "font-weight: 500; background: transparent;"
        + ("font-family: 'JetBrains Mono', monospace;" if mono else ""))
    lay.addWidget(left)
    lay.addStretch(1)
    lay.addWidget(right)
    return row



class RunningBanner(QFrame):
    """Banner for the running job, with only the controls it supports.

    Reads :func:`spacr.qt.bridge.registry`, so it reflects a job started
    from *any* screen. Hidden entirely when nothing is running, which is
    the common case and should cost the page nothing.

    **On the Pause button.** It is enabled if and only if the running
    job's entry point declares itself ``bridge.pausable`` — i.e. it
    actually polls ``bridge.checkpoint()``. No shipped pipeline does, so
    today it renders disabled with :data:`PAUSE_UNAVAILABLE` as its
    tooltip: a Pause button that stopped the thread wherever it happened
    to be would not be a safe pause, so the control is shown disabled with
    the reason.

    :param icon_provider: called with a module key for that module's icon.
        A callable rather than a dict of icons, so the banner does not build
        artwork for modules that never run.
    :param names: module key to the name a human reads, for the banner's
        text and for :attr:`open_requested`'s meaning.
    :param parent: parent widget.
    """

    open_requested = Signal(str)

    def __init__(self, icon_provider: Callable[[str], Optional[QIcon]],
                 names: Dict[str, str], parent=None):
        """Build the banner for one running module.

        :param icon_provider: how to get a module's icon.
        :param names: display names, keyed by module.
        :param parent: parent widget.
        """
        super().__init__(parent)
        P = active_palette()
        self.setObjectName("HomeRunningBanner")
        self.setStyleSheet(
            "QFrame#HomeRunningBanner {"
            f"background: {P['accent_soft']};"
            f"border: 1px solid {P['accent_lo']};"
            "border-radius: 8px; }")
        self._icon_provider = icon_provider
        self._names = names
        self._handle = None

        row = QHBoxLayout(self)
        row.setContentsMargins(SPACING["md"], SPACING["sm"],
                               SPACING["md"], SPACING["sm"])
        row.setSpacing(SPACING["md"])

        self._icon = QLabel()
        self._icon.setFixedSize(32, 32)
        self._icon.setStyleSheet("background: transparent;")
        row.addWidget(self._icon)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(1)
        self._title = QLabel("")
        self._title.setStyleSheet(
            f"color: {P['fg']}; font-size: {font_px(14)}px; font-weight: 600;"
            "background: transparent;")
        self._sub = QLabel("")
        self._sub.setObjectName("HomeRunningSub")
        self._sub.setStyleSheet(
            f"color: {P['fg_muted']}; font-size: {font_px(11)}px;"
            "background: transparent;")
        text_col.addWidget(self._title)
        text_col.addWidget(self._sub)
        row.addLayout(text_col, 1)

        self._bar = QProgressBar()
        self._bar.setFixedWidth(180)
        self._bar.setTextVisible(False)
        row.addWidget(self._bar)

        self._btn_open = QPushButton("Open")
        self._btn_open.setObjectName("GhostButton")
        self._btn_open.setCursor(Qt.PointingHandCursor)
        self._btn_open.setToolTip("Go to the screen this run belongs to.")
        self._btn_open.clicked.connect(self._on_open)
        row.addWidget(self._btn_open)

        self._btn_pause = QPushButton("Pause")
        self._btn_pause.setObjectName("GhostButton")
        self._btn_pause.setCursor(Qt.PointingHandCursor)
        self._btn_pause.setStyleSheet(
            "QPushButton#GhostButton:disabled {"
            f"color: {P['fg_dim']};"
            f"border-color: {P['border_soft']};"
            "background: transparent; }")
        self._btn_pause.clicked.connect(self._on_pause)
        row.addWidget(self._btn_pause)

        self._btn_quit = QPushButton("Quit")
        self._btn_quit.setCursor(Qt.PointingHandCursor)
        self._btn_quit.setToolTip(
            "Stop this run. You are asked whether to let it finish the "
            "step it is on, or to stop it immediately.")
        from ..shutdown import style_as_danger
        style_as_danger(self._btn_quit, P)
        self._btn_quit.clicked.connect(self._on_quit)
        row.addWidget(self._btn_quit)

        self.hide()

    def _on_quit(self) -> None:
        """Stop the run this banner is showing.

        Quits the RUN, not the application: this button is attached to one
        job, and a user who wants the app gone has the one in Preferences.
        Force here means `QThread.terminate()`, which `cancel_all` refuses
        to do on its own and documents why -- mid-write is exactly when it
        is unsafe. The difference is that here somebody has been told that
        and asked for it anyway.
        """
        from ..shutdown import (CANCEL, FORCE, GracefulQuitWatcher,
                                ask_how_to_quit, describe_active)

        handle = self._handle
        if handle is None:
            return
        name = self._names.get(handle.app_key, handle.app_key)
        choice = ask_how_to_quit(self, what=name,
                                 detail=describe_active([handle]))
        if choice == CANCEL:
            return

        if choice == FORCE:
            self._terminate(handle)
            return

        handle.request_cancel("quit from the Home screen")
        self._quit_watcher = GracefulQuitWatcher(
            self,
            lambda h=handle: bool(h.is_running()),
            what=name,
            describe=lambda h=handle: describe_active([h]),
            on_force=lambda h=handle: self._terminate(h),
        )
        self._quit_watcher.start()

    @staticmethod
    def _terminate(handle) -> None:
        """Stop a job's thread outright.

        Never reached without the user having been shown what it costs.
        `request_cancel` first regardless, so a worker that IS still
        checking gets the chance to stop on its own terms in the moment
        before the thread is taken away from it.
        """
        import logging

        logging.getLogger(__name__).warning(
            "Force-stopping %s at the user's request", handle.app_key)
        try:
            handle.request_cancel("force quit from the Home screen")
        except Exception:
            pass
        thread = getattr(handle, "thread", None)
        if thread is None:
            return
        try:
            from ..bridge import drain_thread
            drain_thread(thread, getattr(handle, "worker", None),
                         timeout_ms=2000)
        except RuntimeError:
            pass

    def bind(self, handle) -> None:
        """Show ``handle``'s job, or hide the banner when it is ``None``.

        :param handle: the running job's handle, whose ``app_key`` sets the
            icon and title; ``None`` hides the banner.
        """
        self._handle = handle
        if handle is None:
            self.hide()
            return
        key = handle.app_key
        icon = self._icon_provider(key) if self._icon_provider else None
        if icon is not None:
            self._icon.setPixmap(icon.pixmap(32, 32))
        self._title.setText(f"{self._names.get(key, key)} · running")
        self._sync_pause_control()
        self.refresh()
        self.show()

    def _sync_pause_control(self) -> None:
        """Make the pause button say what the run is actually doing.

        READ FROM THE RUN, not from the last press: a run can pause itself at
        a checkpoint, and a button showing what the user last clicked would
        then be wrong.
        """
        handle = self._handle
        pausable = bool(handle is not None and handle.supports_pause)
        self._btn_pause.setEnabled(pausable)
        self._btn_pause.setToolTip(
            PAUSE_AVAILABLE if pausable else PAUSE_UNAVAILABLE)
        self._btn_pause.setAccessibleDescription(
            PAUSE_AVAILABLE if pausable else PAUSE_UNAVAILABLE)
        if handle is not None and handle.gate.is_paused():
            self._btn_pause.setText("Resume")
        else:
            self._btn_pause.setText("Pause")

    def refresh(self) -> None:
        """Re-read elapsed time + progress from the handle."""
        handle = self._handle
        if handle is None:
            return
        fraction = handle.fraction()
        if fraction is None:
            self._bar.setRange(0, 0)
        else:
            self._bar.setRange(0, 100)
            self._bar.setValue(int(round(fraction * 100)))
        bits = [_fmt_elapsed(handle.elapsed())]
        if handle.progress:
            bits.append(f"{handle.progress[0]} of {handle.progress[1]}")
        if handle.gate.is_paused():
            bits.append("paused")
        tail = handle.last_line
        if tail and not tail.lstrip().startswith("Progress:"):
            bits.append(tail[:70])
        self._sub.setText(" · ".join(bits))
        self._sync_pause_control()

    def _on_open(self) -> None:
        """Go to the module this banner is about."""
        if self._handle is not None:
            self.open_requested.emit(self._handle.app_key)

    def _on_pause(self) -> None:
        """Only reachable when the job declared itself pausable."""
        handle = self._handle
        if handle is None or not handle.supports_pause:
            return
        gate = handle.gate
        gate.resume() if gate.is_paused() else gate.pause()
        self._sync_pause_control()

    @property
    def pause_button(self) -> QPushButton:
        """The pause control, exposed so a test can drive it.

        :returns: the button.
        """
        return self._btn_pause



class QueuedPanel(Panel):
    """The plate queue, when there is one.

    Reads ``~/.spacr/queue.json`` through :class:`spacr.qt.plate_queue.
    PlateQueue` — read-only; the Queue screen owns writes. The panel
    hides itself when the queue is empty rather than drawing an empty
    box, which is the difference between "nothing queued" and "queue
    broken".

    :param parent: parent widget.
    """

    #: Rows drawn before the rest collapse into a "+N more" line.
    MAX_ROWS = 4

    #: Emitted after **Clear** has emptied the queue, so anything else
    #: showing it -- the Queue screen, if it is built -- can re-read.
    queue_cleared = Signal()

    def __init__(self, parent=None):
        """Build the panel and its caption.

        :param parent: parent widget.
        """
        super().__init__("Queued", parent)
        self._clear = self.add_action(
            "Clear", kind="danger",
            tip="Empty the plate queue and hide this panel.")
        self._clear.clicked.connect(self.clear_queue)
        self.refresh()

    def queue_items(self) -> List:
        """Whatever is in the saved plate queue right now.

        Read from disk on each call rather than cached: the queue is written
        by other screens, and a Home page showing a stale count is worse
        than one that costs a file read when it is looked at.

        :returns: the queued items, empty when there is no queue.
        """
        try:
            from ..plate_queue import PlateQueue
            return list(PlateQueue().items())
        except Exception:
            return []

    def clear_queue(self) -> int:
        """Drop every item from the plate queue. Returns how many went.

        This method writes, which no other part of this panel does -- the class docstring
        says the Queue screen owns writes, and this is the one exception:
        **Clear** empties the queue from here rather than sending the reader
        to another screen to do it.

        Not confirmed, and deliberately. A queue entry is a plate waiting to
        be processed -- settings and a source path, no results -- so clearing
        one throws away a few seconds of setting up, not any data. A
        confirmation on a two-word action that costs that little is a
        dialog people learn to dismiss without reading.

        After any failure the panel shows what is actually in the queue,
        because it re-reads from disk afterwards either way.
        """
        removed = 0
        try:
            from ..plate_queue import PlateQueue
            queue = PlateQueue()
            removed = len(list(queue.items()))
            queue.clear()
        except Exception:                                        # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(
                "could not clear the plate queue", exc_info=True)
        self.refresh()
        if removed:
            self.queue_cleared.emit()
        return removed

    def refresh(self) -> None:
        """Re-read the queue and redraw the panel."""
        P = active_palette()
        self._clear_body()
        items = self.queue_items()
        pending = [i for i in items
                   if str(getattr(i.status, "value", i.status)) in
                   ("queued", "running")]
        if not pending:
            self.hide()
            return
        for item in pending[:self.MAX_ROWS]:
            state = str(getattr(item.status, "value", item.status))
            label = item.label or item.app_key
            self.add(_row(label, state,
                          P["accent"] if state == "running"
                          else P["fg_muted"]))
        if len(pending) > self.MAX_ROWS:
            from ..i18n import tr

            more = QLabel(tr("+{n} more", n=len(pending) - self.MAX_ROWS))
            more.setStyleSheet(
                f"color: {P['fg_dim']}; font-size: {font_px(11)}px;"
                "background: transparent;")
            self.add(more)
        self.show()


class RecentRunsPanel(Panel):
    """Last few journalled runs of an existing module; each row navigates.

    Only runs of a module Home knows about are listed. A run journal can
    hold entries under keys that are not modules, such as the ``_job`` or
    ``job`` key a test fixture once wrote into ``~/.spacr/runs``; a row for
    one of those would ask the window to open a module that does not exist.

    Because each row opens its module when pressed, any entry whose key is
    not a module Home knows about is dropped.

    :param limit: how many journalled runs to show.
    :param known_keys: callable returning the module keys that exist, or a
        mapping of key to display name (or either, directly). ``None``
        filters nothing, which is what a standalone panel with no registry
        to consult has to do.
    :param parent: parent widget.
    """

    run_clicked = Signal(str)
    #: Emitted after **Clear** moved the watermark, so Home can re-read.
    cleared = Signal()

    def __init__(self, limit: int = 4, known_keys=None, parent=None,
                 read_now: bool = True):
        """Build the panel and its caption.

        :param parent: parent widget.
        :param read_now: read the journal on the calling thread now. Home
            passes ``False`` and fills the panel from its worker read, so
            the window is not held up by the journal.
        """
        super().__init__("Recent runs", parent)
        self._limit = limit
        self._known_keys = known_keys
        self._clear = self.add_action(
            "Clear", kind="danger",
            tip="Hide the runs listed here. The run journal and Run "
                "History keep them.")
        self._clear.clicked.connect(self.clear_list)
        self.refresh(None if read_now else [])

    def _registry(self):
        """Whatever ``known_keys`` resolves to right now, or ``None``."""
        source = self._known_keys
        if source is None:
            return None
        try:
            return source() if callable(source) else source
        except Exception:                                        # noqa: BLE001
            return None

    def known(self) -> Optional[set]:
        """The module keys that exist, or ``None`` for "do not filter"."""
        value = self._registry()
        return None if value is None else {str(k) for k in value}

    def name_for(self, key: str) -> str:
        """``key``'s display name when Home knows one, else ``key`` itself.

        Without this a row would be captioned with the raw app key, such as
        ``mask``. The names come from Home's own registry.

        :param key: application key to name; returned unchanged when the name
            registry does not know it.
        """
        value = self._registry()
        if isinstance(value, dict):
            return str(value.get(key, key) or key)
        return key

    def clear_list(self) -> None:
        """Hide every run listed, by moving the watermark to now.

        Nothing is deleted. See
        :func:`spacr.qt.preferences.get_dashboard_watermark` for why: this
        panel reads the run journal, and the journal is the record Run
        History searches and a run's manifest documents.
        """
        from ..preferences import set_dashboard_watermark

        set_dashboard_watermark("runs")
        self.refresh([])
        self.cleared.emit()

    def read(self) -> list:
        """The journal entries this panel would show. **Worker-thread safe.**

        Split out of :meth:`refresh` so :class:`HomePage` can call it off the
        GUI thread: it touches no widget, only the run journal.
        ``recent_runs`` opens and JSON-parses every manifest in a bounded
        window of the newest run folders before it sorts and truncates --
        measured at 540 ms over 4,865 of them -- so it is not something the
        GUI thread should be doing on the way back to Home.

        The filtering happens here, not in `refresh`, for two reasons. It is
        the half that runs off the GUI thread, and it changes how much has to
        be read: dropping four rows in five after asking for five leaves one,
        so this asks for a multiple of the limit and truncates afterwards.
        """
        try:
            from spacr.run_journal import recent_runs
            from ..preferences import get_dashboard_watermark
        except Exception:                                        # noqa: BLE001
            return []
        try:
            entries = recent_runs(limit=max(self._limit * 8, 32))
        except Exception:                                        # noqa: BLE001
            return []
        keys = self.known()
        since = get_dashboard_watermark("runs")
        kept = []
        for entry in entries:
            key = str(entry.get("app_key") or "")
            if keys is not None and key not in keys:
                continue
            if since and str(entry.get("start_utc") or "") <= since:
                continue
            kept.append(entry)
            if len(kept) >= self._limit:
                break
        return kept

    def refresh(self, runs: Optional[list] = None) -> None:
        """Redraw the panel.

        :param runs: entries a worker has already read. ``None`` reads them
            here, on the calling thread — which is what a standalone panel
            and the tests do, and what :class:`HomePage` deliberately does
            not.
        """
        P = active_palette()
        self._clear_body()
        if runs is None:
            runs = self.read()
        self._clear.setEnabled(bool(runs))
        if not runs:
            hint = QLabel("No runs yet.")
            hint.setStyleSheet(
                f"color: {P['fg_dim']}; font-size: {font_px(11)}px;"
                "font-style: italic; background: transparent;")
            hint.setWordWrap(True)
            self.add(hint)
            return
        for entry in runs:
            self.add(self._run_row(entry))

    def _run_row(self, entry: dict) -> QWidget:
        """One row describing a finished run.

        :param entry: the run to describe.
        :returns: the row widget.
        """
        P = active_palette()
        ok = entry.get("status") == "success"
        key = entry.get("app_key", "?")
        elapsed = int(entry.get("elapsed_s") or 0)
        btn = QPushButton()
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFlat(True)
        btn.setStyleSheet(
            "QPushButton { background: transparent; border: none;"
            " text-align: left; padding: 1px; }"
            f"QPushButton:hover {{ background: {P['surface_hi']}; }}")
        lay = QHBoxLayout(btn)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(SPACING["sm"])
        dot = QLabel("●" if ok else "○")
        dot.setFixedWidth(12)
        dot.setStyleSheet(
            f"color: {P['success'] if ok else P['error']};"
            f"font-size: {font_px(11)}px; background: transparent;")
        name = QLabel(self.name_for(key))
        name.setStyleSheet(f"color: {P['fg']}; font-size: {font_px(12)}px;"
                           "background: transparent;")
        when = QLabel(_fmt_elapsed(elapsed))
        when.setStyleSheet(f"color: {P['fg_dim']}; font-size: {font_px(11)}px;"
                           "background: transparent;")
        lay.addWidget(dot)
        lay.addWidget(name, 1)
        lay.addWidget(when)
        btn.clicked.connect(lambda _=False, k=key: self.run_clicked.emit(k))
        return btn


class SystemPanel(Panel):
    """GPU / VRAM / Disk, read on build and on every Home revisit.

    Every reading degrades to a string rather than vanishing — a blank
    row reads as "broken", while ``n/a`` says the lightweight system probe
    could not measure a device.  Home must not import a model
    runtime merely to decorate the dashboard.

    :param parent: parent widget.
    """

    def __init__(self, parent=None):
        """Build the panel and its caption.

        :param parent: parent widget.
        """
        super().__init__("System", parent)
        self._refresh = self.add_action(
            "Refresh", kind="safe", tip="Re-read GPU, VRAM and disk now.")
        self._refresh.clicked.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        """Re-read GPU, VRAM and disk, and redraw the panel."""
        self._clear_body()
        self.add(_row("GPU", self.gpu_util()))
        self.add(_row("VRAM", self.gpu_vram()))
        self.add(_row("Disk", self.disk_used()))

    @staticmethod
    def gpu_util() -> str:
        """Current GPU utilisation, as text for display.

        Never raises. A missing NVML, a machine with no GPU and a driver
        mismatch are all ordinary here, and none of them is a reason for the
        Home page to fail to build.

        :returns: the reading, or a dash when it cannot be taken.
        """
        try:
            nvml = _nvml()
            if nvml is None:
                raise RuntimeError("no NVML")
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            return f"{nvml.nvmlDeviceGetUtilizationRates(handle).gpu}%"
        except Exception:
            return "n/a"

    @staticmethod
    def gpu_vram() -> str:
        """Current VRAM use, as text for display.

        Never raises, for the same reason as :meth:`gpu_util`.

        :returns: the reading, or a dash when it cannot be taken.
        """
        try:
            nvml = _nvml()
            if nvml is None:
                raise RuntimeError("no NVML")
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            info = nvml.nvmlDeviceGetMemoryInfo(handle)
            return f"{info.used / 1e9:.1f} / {info.total / 1e9:.0f} GB"
        except Exception:
            return "n/a"

    @staticmethod
    def disk_used() -> str:
        """Disk use for the working volume, as text for display.

        :returns: the reading, or a dash when it cannot be taken.
        """
        try:
            import shutil
            usage = shutil.disk_usage(os.path.expanduser("~"))
            return f"{int(100 * usage.used / usage.total)}%"
        except Exception:
            return "n/a"


#: Sentinel so a machine with no GPU is probed once, not once per refresh.
_UNSET = object()
_NVML = _UNSET


def _nvml():
    """Return an initialised NVML module, or None when there is no NVIDIA GPU.

    ``nvidia-ml-py`` is the maintained package and ``pynvml`` is the retired
    one, and BOTH install a module called ``pynvml`` -- so the import line is
    the same either way and only the installed distribution differs. The
    retired one warns on import, which reached the console at every start.

    The suppression is local rather than a global filter because a global
    one was already in place and did not hold: the warning is raised while
    the module body executes, and anything that has reset the warning
    filters by then -- a Qt plugin, a library imported earlier -- lets it
    through. ``catch_warnings`` around the import itself cannot be reset by
    somebody else.

    :returns: the NVML module with ``nvmlInit`` already called, or None.
    """
    global _NVML
    if _NVML is _UNSET:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                warnings.simplefilter("ignore", DeprecationWarning)
                import pynvml
            pynvml.nvmlInit()
            _NVML = pynvml
        except Exception:                                        # noqa: BLE001
            _NVML = None
    return _NVML


class TotalsPanel(Panel):
    """Aggregate counts from the automatically complete run journal.

    :param parent: parent widget.
    """

    #: Emitted after **Reset** moved the watermark.
    reset_requested = Signal()

    def __init__(self, parent=None, read_now: bool = True):
        """Build the panel and its caption.

        :param parent: parent widget.
        :param read_now: read the journal on the calling thread now. Home
            passes ``False`` and fills the panel from its worker read.
        """
        super().__init__("Totals", parent)
        self._reset = self.add_action(
            "Reset", kind="danger",
            tip="Start these counts from now. The run journal and Run "
                "History keep every run.")
        self._reset.clicked.connect(self.reset_counts)
        self.refresh(None if read_now else {})

    def reset_counts(self) -> None:
        """Count from now on, by moving the watermark.

        Nothing is deleted, for the same reason **Clear** on Recent runs
        deletes nothing — see
        :func:`spacr.qt.preferences.get_dashboard_watermark`. The counts
        show how much this installation has done, and a reset that removed
        the manifests would also remove Run History and every run's
        provenance.
        """
        from ..preferences import set_dashboard_watermark

        set_dashboard_watermark("totals")
        self.refresh()
        self.reset_requested.emit()

    def read(self) -> dict:
        """The journal totals. **Worker-thread safe** — see
        :meth:`RecentRunsPanel.read`; ``journal_totals`` walks the same
        thousands of manifests, measured at 247 ms.

        Counted from the **Reset** watermark when one is set, which is the
        one case this cannot answer out of the cached totals file: that file
        holds LIFETIME counts, so a windowed count has to walk the runs the
        window covers. It is bounded by the window, which is the property
        that makes the feature affordable — somebody who reset yesterday
        is counting yesterday's runs, not eleven thousand of them.
        """
        try:
            from ..preferences import get_dashboard_watermark
            since = get_dashboard_watermark("totals")
        except Exception:                                        # noqa: BLE001
            since = ""
        try:
            if since:
                return self._totals_since(since)
            from spacr.run_journal import journal_totals
            return journal_totals()
        except Exception:
            return {"total_runs": 0, "mask_runs": 0, "measure_runs": 0,
                    "models_recorded": 0}

    @staticmethod
    def _totals_since(since: str) -> dict:
        """Counts over the runs that started after ``since``.

        ISO-8601 UTC strings compare lexicographically in time order. Run
        folders are named by their UTC start second, so only folders named
        from an hour before ``since`` onwards are opened: the cost follows the
        window, not the size of the journal.
        """
        import json
        from datetime import datetime, timedelta, timezone

        from spacr.run_journal import _run_dir_names, runs_root

        counts = {"total_runs": 0, "mask_runs": 0, "measure_runs": 0,
                  "classify_runs": 0, "models_recorded": 0}
        root = runs_root()
        names = sorted(_run_dir_names(root)) if root.exists() else []
        try:
            moment = datetime.fromisoformat(since.replace("Z", "+00:00"))
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            floor = (moment.astimezone(timezone.utc) - timedelta(hours=1)
                     ).strftime("%Y-%m-%d_%H%M%S")
            names = [n for n in names if n[:17] >= floor]
        except ValueError:
            pass
        for name in names:
            try:
                entry = json.loads((root / name / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if str(entry.get("start_utc") or "") <= since:
                continue
            counts["total_runs"] += 1
            bucket = f"{str(entry.get('app_key') or '')}_runs"
            if bucket in counts:
                counts[bucket] += 1
        return counts

    def refresh(self, totals: Optional[dict] = None) -> None:
        """Redraw the panel.

        :param totals: counts a worker has already read; ``None`` reads them
            on the calling thread.
        """
        self._clear_body()
        if totals is None:
            totals = self.read()
        self._reset.setEnabled(bool(totals.get("total_runs", 0)))
        self.add(_row("Runs", str(totals.get("total_runs", 0))))
        self.add(_row("Mask", str(totals.get("mask_runs", 0))))
        self.add(_row("Meas.", str(totals.get("measure_runs", 0))))
        self.add(_row("Models", str(totals.get("models_recorded", 0))))


class StageLegend(Panel):
    """What the three hover colours mean. One row per stage.

    The legend follows the numeric status panels in the right-hand column
    because it explains the module tiles rather than reporting machine state.

    Each row draws the stage's hue as a filled swatch *and* names the
    stage in words. Colour alone would fail WCAG 1.4.1 and would be
    invisible to the colour-blind mode this app already ships — the
    words are what make it a legend rather than a palette.

    The rows are built from :data:`spacr.qt.theme.STAGE_HOVER`, which is
    the same table the stylesheet builds the hover rules from, so the
    swatch and the tile it explains cannot drift apart.

    :param parent: parent widget.
    """

    #: Side of the colour chip in px, at 100 % font scale.
    SWATCH = 12

    def __init__(self, parent=None):
        """Build the panel and its caption.

        :param parent: parent widget.
        """
        super().__init__("Module state", parent)
        from ..theme import STAGE_LABEL, STAGE_NOTE
        self.header.setToolTip(
            "Hover any module tile and it lights up in the colour of "
            "how finished it is.")
        self._rows: Dict[str, QWidget] = {}
        for stage in ("alpha", "beta", "stable"):
            row = self._legend_row(stage, self.swatch_colour(stage),
                                   STAGE_LABEL[stage], STAGE_NOTE[stage])
            self._rows[stage] = self.add(row)

    def _legend_row(self, stage: str, colour: str, label: str,
                    note: str) -> QWidget:
        """One row explaining what a maturity colour means.

        :param stage: the stage's name.
        :returns: the row widget.
        """
        from ..preferences import scaled_px
        P = active_palette()
        row = QWidget()
        row.setToolTip(note)
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(SPACING["sm"])
        side = scaled_px(self.SWATCH)
        chip = QLabel()
        name_id = f"StageSwatch_{stage}"
        chip.setObjectName(name_id)
        chip.setFixedSize(side, side)
        chip.setStyleSheet(
            f"QLabel#{name_id} {{ background: {colour};"
            f" border: 1px solid {P['border']}; border-radius: 3px; }}")
        name = QLabel(label)
        name.setStyleSheet(f"color: {P['fg']}; font-size: {font_px(12)}px;"
                           "font-weight: 500; background: transparent;")
        lay.addWidget(chip)
        lay.addWidget(name, 1)
        return row

    @staticmethod
    def swatch_colour(stage: str) -> str:
        """The hex this legend draws for ``stage``.

        The same function the stylesheet builds the hover rules from, so
        the swatch and the tile it explains cannot come apart.

        :param stage: maturity stage name; an unknown stage gets the
            ``'stable'`` colour.
        """
        from ..theme import stage_hover
        return stage_hover(stage)

    def row_for(self, stage: str) -> Optional[QWidget]:
        """The legend row explaining one maturity stage.

        :param stage: the stage's name.
        :returns: the row widget, or None when that stage has no row.
        """
        return self._rows.get(stage)


class _NewsTitleFit(QObject):
    """Bound release headings to the real viewport, not wide body hints."""

    def __init__(self, panel):
        """Own the coalesced observer under its News panel.

        :param panel: actual panel whose headline labels may be rebuilt.
        """
        self._busy = True
        super().__init__(panel)
        self._panel = panel
        self._viewport = panel._notes.viewport()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fit)
        self._viewport.destroyed.connect(self._timer.stop)
        panel.installEventFilter(self)
        self._viewport.installEventFilter(self)
        self._busy = False
        self._timer.start(0)

    def eventFilter(self, watched, event):
        """Observe allocation and source-layout changes without consuming them.

        :param watched: the panel or its actual notes viewport.
        :param event: native event; only local allocation/style work is queued.
        :returns: False, preserving normal Qt delivery.
        """
        if not getattr(self, "_busy", True) and ((watched is self._viewport and event.type() == QEvent.Resize)
                or (watched is self._panel and event.type() in
                    (QEvent.Show, QEvent.FontChange, QEvent.StyleChange, QEvent.LayoutRequest))):
            from shiboken6 import isValid

            if isValid(self._viewport):
                self._timer.start(0)
        return False

    def _fit(self):
        """Fit fresh wrapped-title documents; leave wide plain dates scrollable."""
        from math import ceil
        from PySide6.QtGui import QTextDocument, QTextOption
        from shiboken6 import isValid
        from ..gui_scale import _set_actual_dim

        if not isValid(self._viewport):
            return
        width = self._viewport.property('size').width()
        if width <= 0:
            return
        self._busy = True
        try:
            for label in self._panel._headline_labels:
                limit = width
                if label.wordWrap():
                    if (label.property('minimumSize').width() != width
                            or getattr(label, '_gs_geo_set', {}).get('minw') != width):
                        _set_actual_dim(label, 'minw', width)
                else:
                    size, contents = label.property('size'), label.contentsRect()
                    text = label.style().itemTextRect(label.fontMetrics(), contents,
                        int(label.alignment() | Qt.TextSingleLine), label.isEnabled(), label.text())
                    required = max(text.width(), label.fontMetrics().horizontalAdvance(label.text()))
                    if required+size.width()-contents.width() > width:
                        limit = 16777215
                if (label.property('maximumSize').width() != limit
                        or getattr(label, '_gs_geo_set', {}).get('maxw') != limit):
                    _set_actual_dim(label, 'maxw', limit)
                if label.wordWrap():
                    size, contents = label.property('size'), label.contentsRect()
                    margin = label.margin()
                    indent = label.indent()
                    if indent < 0:
                        indent = max(0, label.fontMetrics().horizontalAdvance('x') // 2-margin) if label.frameWidth() else 0
                    alignment = QStyle.visualAlignment(label.layoutDirection(), label.alignment())
                    horizontal = size.width()-contents.width()+2*margin
                    vertical = size.height()-contents.height()+2*margin
                    if alignment & (Qt.AlignLeft | Qt.AlignRight):
                        horizontal += indent
                    if alignment & (Qt.AlignTop | Qt.AlignBottom):
                        vertical += indent
                    document = QTextDocument()
                    document.setDocumentMargin(0)
                    document.setDefaultFont(label.font())
                    option = QTextOption()
                    option.setWrapMode(QTextOption.WordWrap)
                    option.setAlignment(label.alignment())
                    document.setDefaultTextOption(option)
                    document.setHtml(label.text())
                    document.setTextWidth(max(1, width-horizontal))
                    height = ceil(document.size().height()+vertical)
                    if (label.property('minimumSize').height() != height
                            or getattr(label, '_gs_geo_set', {}).get('minh') != height):
                        _set_actual_dim(label, 'minh', height)
        finally:
            self._busy = False


class _NewsFirstPaint(QObject):
    """Queues a News panel's remaining releases after its first paint.

    An event filter rather than a ``paintEvent`` override, so the panel adds
    no API symbol and stops calling into Python on its paints once nothing
    is left to append.
    """

    def eventFilter(self, watched, event):
        """On a paint, queue the releases not drawn yet; never consume it.

        :param watched: the News panel.
        :param event: any of its events; only ``Paint`` acts.
        :returns: ``False`` always.
        """
        try:
            if event.type() == QEvent.Paint:
                if getattr(watched, "_pending_releases", None):
                    watched._queue_the_rest(0)
                else:
                    watched.removeEventFilter(self)
        except RuntimeError:
            pass
        return False


class NewsPanel(Panel):
    """Every spaCR release, with links, in a box the reader can resize.

    The notes are bundled, and the bundle is what is drawn. They come from
    ``spacr/resources/release_notes.json``, which
    ``tools/build_release_notes.py`` writes from the GitHub releases and
    which ``.github/workflows/release.yml`` refreshes on every release.
    This panel is on the first screen the application shows; content read
    from api.github.com would be empty offline, rate-limited behind a shared
    NAT, and slower to draw than the window. The bundled file therefore
    remains the offline source and the panel is complete before any network
    access.

    Newer releases are then added. The wheel for a release cannot contain
    its own release note -- the note is written when the GitHub release is
    published, which is after that wheel is on PyPI -- so a bundled file is
    always one release behind the build carrying it. After the page is shown,
    :attr:`refresh_requested` asks the window to read the public releases
    list on a worker thread, and :meth:`apply_releases` merges whatever
    comes back in front of the bundled list. Nothing here opens a socket:
    the panel only asks, and a fetch that fails, is rate-limited, is
    switched off in Preferences, or simply finds nothing newer leaves the
    bundled list exactly as it was drawn.

    The panel shows the newest release, its body rendered with its links
    live, and the older ones newest-first behind a "Show older releases"
    switch under the box, all inside a scroll area whose height is capped, with
    a grip along the bottom edge that drags the box taller or shorter. The
    height is remembered between sessions. One release's notes can be long
    -- 1.5.1.5's wrap to about 21,000 px at the aside's width -- so the
    scroll area, not the page, carries them.

    The update check stays a button: installing an update is a decision,
    so it is made only when pressed.

    :param version: the build to name in the heading. Empty leaves the
        heading as the translated word alone -- the two are kept separate
        because the catalog is keyed on "News", so composing the release into
        the caption first would leave the only aside panel that names a build
        in English.
    :param parent: parent widget.
    """

    check_requested = Signal()

    #: Emitted once, after the panel has been shown, to ask the window for
    #: a newer release list than the one in this wheel. It carries nothing
    #: and it opens nothing: the window answers it on a worker thread and
    #: hands the result back through :meth:`apply_releases`. A panel built
    #: in a test, or on a window that does not connect it, simply never
    #: gets an answer and keeps drawing the bundled list.
    refresh_requested = Signal()

    #: Height of the scrolling list in px at 100 % font scale: the default,
    #: and how far the grip may drag it. The floor has to show a heading and
    #: a line under it or dragging to it looks like a bug; the ceiling is
    #: about the height of the window's content area, past which the panel
    #: pushes everything below it off the page.
    #:
    #: 100 IS A BUDGET, NOT A TASTE. `test_no_variant_clips_elides_or_
    #: overflows` measures the shipped layout at 1440x900 and today it fits
    #: exactly; at 190 the aside needed 989 px of a 900 px canvas, so the
    #: dashboard would have arrived needing a scrollbar on the smallest
    #: common laptop. The reader drags it taller when they want to read a
    #: release, and that height is remembered -- which is the whole point of
    #: the grip, and why the default does not have to be generous.
    NOTES_H = 100
    NOTES_H_MIN = 90
    NOTES_H_MAX = 720

    #: Releases built with the panel; the rest follow its first paint,
    #: hidden until the reader switches on the older releases. Each wrapped
    #: rich-text body costs several document layouts to size, and all of
    #: them together were about 0.4 s of Home's first paint on an Intel Mac,
    #: for a box that shows the newest release at its default height.
    FIRST_RELEASES = 1

    #: When a shown panel that has not painted -- scrolled below the fold of
    #: a short aside -- appends the rest anyway, so every release is there
    #: for assistive technology without the reader scrolling to it first.
    REST_AFTER_SHOW_MS = 300

    def __init__(self, version: str = "", parent=None):
        """Build the panel and its caption.

        :param parent: parent widget.
        """
        from ..i18n import tr

        heading = tr("News")
        super().__init__(f"{heading} \u00b7 spaCR {version}" if version
                         else heading, parent)
        P = active_palette()
        self.content: Optional[QWidget] = None
        self._releases = self.read_releases()
        self._refresh_asked = False

        self._notes = QScrollArea()
        self._notes.setObjectName("HomeNewsScroll")
        self._notes.setFocusPolicy(Qt.NoFocus)
        self._notes.setWidgetResizable(True)
        self._notes.setFrameShape(QFrame.NoFrame)
        self._notes.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._notes.setStyleSheet(
            "QScrollArea#HomeNewsScroll { background: transparent;"
            " border: none; }")
        inner = QWidget()
        make_transparent(inner)
        self._notes_column = QVBoxLayout(inner)
        self._notes_column.setContentsMargins(0, 0, 0, 0)
        self._notes_column.setSpacing(SPACING["sm"])
        self._notes.setWidget(inner)
        self.body_layout.addWidget(self._notes)

        from .toggle import Toggle

        self._older = Toggle(tr("Show older releases"))
        self._older.setObjectName("HomeNewsOlderReleases")
        self._older.setChecked(False)
        self._older_blocks = []
        self._older.toggled.connect(self._show_older_releases)
        self.body_layout.addWidget(self._older)

        self._placeholder = QLabel(
            "No release notes bundled with this build.")
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(
            f"color: {P['fg_muted']}; font-size: {font_px(11)}px;"
            "font-style: italic; background: transparent;")
        self._fill(self.FIRST_RELEASES)
        self._first_paint = _NewsFirstPaint(self)
        self.installEventFilter(self._first_paint)

        self._grip = _HeightGrip(self._notes, self.NOTES_H_MIN,
                                 self.NOTES_H_MAX,
                                 name="Resize the release notes")
        self._grip.height_changed.connect(self._remember_height)
        self.body_layout.addWidget(self._grip)
        self._notes.setFixedHeight(self._stored_height())
        self._news_title_fit = _NewsTitleFit(self)

        check = QPushButton("Check for updates\u2026")
        check.setObjectName("GhostButton")
        check.setCursor(Qt.PointingHandCursor)
        check.clicked.connect(self.check_requested)
        self.add(check)
        self._check = check

    @staticmethod
    def read_releases() -> list:
        """The bundled release records, newest first, or ``[]``.

        Degrades to an empty list on any failure, and the panel then draws
        the placeholder it drew before there was a feed. A dashboard must
        not fail to appear because a resource file is missing from a wheel.
        """
        try:
            import json
            from importlib.resources import files

            raw = files("spacr") / "resources" / "release_notes.json"
            data = json.loads(raw.read_text(encoding="utf-8"))
            releases = data.get("releases") or []
            return [r for r in releases if isinstance(r, dict)]
        except Exception:                                        # noqa: BLE001
            return []

    def _fill(self, count: Optional[int] = None) -> None:
        """Draw :attr:`_releases` into the scrolling column.

        Called once while the panel is built and again whenever a fetched
        list arrives, so the two paths cannot diverge. The placeholder is
        kept rather than rebuilt: it is what a build with no bundled
        resource shows, and :meth:`set_content` holds a reference to it.

        :param count: draw only the newest ``count`` releases now and the
            rest one event-loop turn after the panel first paints; ``None``
            draws every release at once.
        """
        self._pending_releases = []
        self._headline_labels = []
        self._older_blocks = []
        while self._notes_column.count():
            item = self._notes_column.takeAt(0)
            widget = item.widget()
            if widget is not None and widget is not self._placeholder:
                widget.setParent(None)
                widget.deleteLater()
        self._notes_column.addWidget(self._placeholder)
        self._placeholder.setVisible(not self._releases
                                     and self.content is None)
        drawn = self._releases if count is None else self._releases[:count]
        for index, entry in enumerate(drawn):
            block = self._release_block(entry)
            self._notes_column.addWidget(block)
            if index:
                self._older_blocks.append(block)
                block.setVisible(self._older.isChecked())
        self._older.setVisible(len(self._releases) > 1)
        self._notes_column.addStretch(1)
        self._pending_releases = list(self._releases[len(drawn):])
        fitter = getattr(self, '_news_title_fit', None)
        if fitter is not None:
            fitter._timer.start(0)

    def _queue_the_rest(self, delay_ms: int) -> None:
        """Start, or bring forward, the append of the releases not drawn yet.

        :param delay_ms: how long to wait; a running timer due sooner is kept.
        """
        if not self._pending_releases:
            return
        timer = getattr(self, '_rest_timer', None)
        if timer is None:
            timer = self._rest_timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._fill_the_rest)
        if not timer.isActive() or timer.remainingTime() > delay_ms:
            timer.start(delay_ms)

    def _show_older_releases(self, shown: bool) -> None:
        """Build any releases still waiting, then show or hide the older ones.

        A bound method rather than a lambda: a lambda closing over ``self`` and
        connected to a child's signal kept the panel's Python wrapper alive
        after the panel was closed, so it was never collected.

        :param shown: whether the "Show older releases" switch is on.
        """
        self._fill_the_rest()
        for block in self._older_blocks:
            block.setVisible(shown)

    def _fill_the_rest(self) -> None:
        """Append the releases :meth:`_fill` left for after the first paint."""
        pending, self._pending_releases = self._pending_releases, []
        if not pending:
            return
        stretch = self._notes_column.count() - 1
        for offset, entry in enumerate(pending):
            block = self._release_block(entry)
            self._notes_column.insertWidget(stretch + offset, block)
            block.setVisible(self._older.isChecked())
            self._older_blocks.append(block)
        fitter = getattr(self, '_news_title_fit', None)
        if fitter is not None:
            fitter._timer.start(0)

    def showEvent(self, event):                                  # noqa: N802
        """Ask for a refresh the first time the panel is shown.

        After the page exists and on the event loop, not during
        construction: the single-shot timer means the emit lands on a later
        turn than this show, so Home's first paint is never waiting on it.
        Once per panel, because a page that is shown again -- a tab
        revisited, a font-scale rebuild -- is not news.

        :param event: the Qt show event.
        """
        super().showEvent(event)
        self._queue_the_rest(self.REST_AFTER_SHOW_MS)
        if self._refresh_asked:
            return
        self._refresh_asked = True
        QTimer.singleShot(0, self._ask_for_newer_releases)

    def _ask_for_newer_releases(self) -> None:
        """Emit :attr:`refresh_requested`, unless the panel is already gone."""
        try:
            self.refresh_requested.emit()
        except RuntimeError:                                     # noqa: BLE001
            pass

    def apply_releases(self, fetched) -> None:
        """Merge a fetched release list into the list on screen.

        Failures are silent. An empty list, an unreadable list, or a list
        that adds nothing to the bundled file leaves the panel unchanged
        and shows no message, because the reader did not ask for the
        background fetch.

        :param fetched: release records from
            :func:`spacr.updater.fetch_release_notes`, or anything at all.
        """
        try:
            merged = self.merge_releases(self._releases, fetched)
        except Exception:                                        # noqa: BLE001
            return
        if merged == self._releases:
            return
        self._releases = merged
        self._fill()

    @staticmethod
    def merge_releases(bundled, fetched) -> list:
        """The bundled and fetched lists as one, newest first.

        One record per tag, and a fetched record wins: the same release can
        have its notes edited on GitHub after it ships, and the live copy
        is then the true one. Ordering is by publication date and then by
        the version in the tag, so a release published on the same day as
        the one before it still lands above it.

        :param bundled: the records read from the wheel.
        :param fetched: the records read from GitHub, or ``None``.
        :returns: a new list; neither argument is modified.
        """
        by_tag = {}
        for entry in list(bundled or []) + list(fetched or []):
            if not isinstance(entry, dict):
                continue
            tag = str(entry.get("tag") or entry.get("name") or "").strip()
            if not tag:
                continue
            by_tag[tag] = entry
        return sorted(by_tag.values(), key=NewsPanel._newest_first,
                      reverse=True)

    @staticmethod
    def _newest_first(entry: dict) -> tuple:
        """Sort key: publication date, then the version the tag names."""
        digits = re.findall(r"\d+", str(entry.get("tag") or ""))[:4]
        version = tuple(int(d) for d in digits)
        return (str(entry.get("published") or ""),
                version + (0,) * (4 - len(version)))

    def _release_block(self, entry: dict) -> QWidget:
        """One release: its name, its date, and its notes with links live."""
        P = active_palette()
        block = QWidget()
        make_transparent(block)
        column = QVBoxLayout(block)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)

        tag = str(entry.get("tag") or "")
        url = str(entry.get("url") or "")
        name = str(entry.get("name") or tag or "spaCR")
        when = str(entry.get("published") or "")
        title = QLabel(
            f'<a href="{escape(url, quote=True)}"'
            f' style="color: {P["accent"]}; text-decoration: none;">'
            f'{escape(name)}</a>' if url else escape(name))
        title.setOpenExternalLinks(True)
        title.setTextInteractionFlags(Qt.TextBrowserInteraction)
        title.setAccessibleName(name)
        if not url:
            title.setFocusPolicy(Qt.NoFocus)
        title.setWordWrap(True)
        title.setStyleSheet(
            f"color: {P['fg']}; font-size: {font_px(12)}px;"
            "font-weight: 600; background: transparent;")
        column.addWidget(title, 0, Qt.AlignLeft)
        self._headline_labels.append(title)
        if when:
            stamp = QLabel(when)
            stamp.setStyleSheet(
                f"color: {P['fg_muted']}; font-size: {font_px(10)}px;"
                "background: transparent;")
            column.addWidget(stamp, 0, Qt.AlignLeft)
            self._headline_labels.append(stamp)

        body = self.render_body(str(entry.get("body") or ""), P["accent"])
        if body:
            notes = QLabel(body)
            notes.setWordWrap(True)
            notes.setOpenExternalLinks(True)
            notes.setTextInteractionFlags(Qt.TextBrowserInteraction)
            notes.setAccessibleName(name)
            notes.setAccessibleDescription(QTextDocumentFragment.fromHtml(
                body).toPlainText())
            if "<a " not in body:
                notes.setFocusPolicy(Qt.NoFocus)
            notes.setStyleSheet(
                f"color: {P['fg_muted']}; font-size: {font_px(11)}px;"
                "background: transparent;")
            column.addWidget(notes)
        return block

    @staticmethod
    def render_body(body: str, link_colour: str) -> str:
        """Turn a release body into the small HTML subset a QLabel draws.

        This is not a Markdown renderer. Release bodies are
        GitHub-flavoured markdown; what they contain in practice is bullet
        lists, bare URLs and ``**bold**``, and a QLabel understands a
        handful of tags. So: escape everything first -- a body is text from
        a web page and must never reach a rich-text widget as markup --
        then put back the three constructs that are worth having.

        Escaping first is what makes this safe. Linkifying first and
        escaping after would escape the anchors too and show the reader
        their own tags; escaping after building the HTML is the mistake that
        turns a release note into an injection.

        Markdown links ``[text](url)`` become anchors on their own text in
        the same pass that links bare URLs; linking bare URLs alone took the
        closing parenthesis into the address and showed the reader
        ``[releases page](https://...releases)`` with a broken link, as the
        1.5.1.5 notes did on Home. Headings of any level are drawn bold, and
        inline code loses its backticks.

        :param body: release-note text in GitHub-flavoured markdown; it is
            HTML-escaped before links, ``**bold**``, inline code, headings
            and bullet lines are turned into tags.
        :param link_colour: CSS colour for the generated links.
        """
        text = escape(body or "").strip()
        if not text:
            return ""

        link = (f'<a href="{{0}}" style="color: {link_colour};'
                f' text-decoration: none;">{{1}}</a>')
        text = re.sub(
            r"\[([^\]\n]+)\]\((https?://[^\s()<>`]+)\)"
            r"|(https?://[^\s<>()`]+?)([.,;:]?)(?=[\s)]|$)",
            lambda m: (link.format(m.group(2), m.group(1)) if m.group(1)
                       else link.format(m.group(3), m.group(3)) + m.group(4)),
            text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            heading = re.match(r"#{1,6} (.+)", stripped)
            if heading:
                lines.append(f"<b>{heading.group(1)}</b>")
            elif stripped.startswith(("* ", "- ")):
                lines.append(f"\u2022 {stripped[2:]}")
            else:
                lines.append(stripped)
        return "<br>".join(lines)

    def _stored_height(self) -> int:
        """The reader's remembered height, clamped, in device px."""
        from ..preferences import get_news_height, scaled_px

        wanted = get_news_height() or self.NOTES_H
        return scaled_px(max(self.NOTES_H_MIN,
                             min(self.NOTES_H_MAX, int(wanted))))

    def _remember_height(self, px: int) -> None:
        """Store a dragged height, back in font-scale-independent units."""
        from ..preferences import scaled_px, set_news_height

        scale = max(1, scaled_px(100)) / 100.0
        set_news_height(int(round(px / scale)))

    @property
    def releases(self) -> list:
        """The release records currently drawn, newest first."""
        return list(self._releases)

    @property
    def notes_view(self) -> QScrollArea:
        """The scrolling list of releases. For tests."""
        return self._notes

    @property
    def grip(self) -> "_HeightGrip":
        """The drag handle under the list. For tests."""
        return self._grip

    def set_content(self, widget: QWidget) -> None:
        """Replace the release list with real content.

        The escape hatch :meth:`HomePage.set_reserved_content` exposes. It
        hides the bundled notes rather than deleting them, so a caller that
        drops content in has not thrown the feed away.

        :param widget: the widget to insert at the top of the panel body; any
            previous content widget is deleted and the bundled notes are
            hidden.
        """
        self._placeholder.hide()
        self._notes.hide()
        self._grip.hide()
        if self.content is not None:
            self.content.setParent(None)
            self.content.deleteLater()
        self.content = widget
        self.body_layout.insertWidget(0, widget)


#: THE DRAG HANDLE UNDER THE NEWS LIST, and it is no longer Home's own.
#: It was written here, and the nested Regression containers need the same affordance on
#: every nested container in Regression's Measurements tab -- so it moved to
#: :mod:`spacr.qt.widgets.height_grip` and this name is kept pointing at it.
#: A second implementation would be a second set of bugs.
_HeightGrip = HeightGrip



#: Bounds of the Home right-hand column's text slider, as factors of the
#: designed size.
_ASIDE_TEXT_RANGE = (0.7, 1.6)

#: The right-hand column's pane in Home's splitter: its name, where its
#: dragged width is stored, and where its folded state is.
_ASIDE_PANE = "Widgets"
_ASIDE_PERSIST_KEY = "home::body"
_ASIDE_FOLD_KEY = "home/Widgets"

#: Dynamic properties holding a label's own sheet as it was built, and the
#: sheet last written from it, so a rebuilt sheet is taken as the new base.
_BASE_SHEET = "_spacrAsideBaseSheet"
_SCALED_SHEET = "_spacrAsideScaledSheet"


def _scale_text_under(root: QWidget, ratio: float,
                      exempt: Optional[QWidget] = None) -> None:
    """Size the text inside ``root`` by ``ratio`` of what it was built with.

    :param root: the container; its descendants follow.
    :param ratio: 1.0 for the built size.
    :param exempt: a descendant whose own text keeps its built size.
    """
    from ..live_zoom import _FONT_SIZE, _MIN_PX, _MIN_PT, ColumnTextScale
    from ..live_zoom import scaled_font_sheet

    def scaled(sheet: str) -> str:
        """``sheet`` with every font-size in it multiplied by ``ratio``."""
        def one(match):
            """One ``font-size`` declaration, scaled and floored."""
            size = float(match.group(1)) * ratio
            if match.group(2).lower() == "px":
                return f"font-size: {max(_MIN_PX, int(round(size)))}px"
            return f"font-size: {max(_MIN_PT, round(size, 2))}pt"
        return _FONT_SIZE.sub(one, sheet)

    host = scaled_font_sheet(ColumnTextScale._inherited_sheet(root), ratio)
    if str(root.styleSheet() or "") != host:
        root.setStyleSheet(host)
    for widget in root.findChildren(QWidget):
        own = str(widget.styleSheet() or "")
        if not own or (exempt is not None and (
                widget is exempt or exempt.isAncestorOf(widget))):
            continue
        base = widget.property(_BASE_SHEET)
        if base is None or own != widget.property(_SCALED_SHEET):
            base = own
        new = scaled(str(base))
        widget.setProperty(_BASE_SHEET, str(base))
        widget.setProperty(_SCALED_SHEET, new)
        if new != own:
            widget.setStyleSheet(new)


class _TileHintWatcher(QObject):
    """Hands Home the Enter and Leave events of its tiles.

    Home used to be an event filter on each of its 18 tiles, and Qt asks
    every filter on a widget about every event it gets; over the window's
    moving field every tile repaints on every frame, so that was a Python
    call per tile per frame for a filter that acts on two event types. The
    watcher hears those two through the application event hub
    (:func:`spacr.qt.gil_priority._watch_application_events`), which is
    asked about every event anyway, and passes Home only its tiles' events.
    """

    def __init__(self, page: QWidget):
        """Watch the application on behalf of ``page``."""
        from ..gil_priority import _watch_application_events

        super().__init__(page)
        _watch_application_events(QApplication.instance(), self,
                                  frozenset((QEvent.Enter, QEvent.Leave)))

    def eventFilter(self, watched, event) -> bool:
        """Home's answer for its own tiles; never stops an event."""
        page = self.parent()
        if watched in page._tile_hints:
            page.eventFilter(watched, event)
        return False


class HomePage(QWidget):
    """Home. ``tile_clicked(str key)`` fires when a tile is pressed.

    Drop-in for the page it replaces: same constructor, same signal,
    same ``set_reserved_content`` escape hatch.

    :param apps: ``(key, name, description, section)`` per app.
    :param icon_provider: app key → QIcon (or ``None``).
    :param section_notes: optional section → one line, drawn under that
        category's heading on its own tab. A category with two apps in
        it looks broken until it says why; passed in rather than
        imported so this widget still knows nothing about
        :mod:`spacr.qt.app`.
    :param categories: optional ordered ``(title, [app key])`` — one
        entry per tab after Home. Defaults to grouping ``apps`` by their
        section in first-appearance order, which is what every test that
        builds a HomePage out of a handful of tuples wants.
    :param bands: optional ordered ``(title, [app key])`` for the Home
        tab. Same default. Kept separate from ``categories`` because the
        two answer different questions, even when — as today — they
        return the same list. See the module docstring.
    :param stages: optional app key → ``stable`` / ``beta`` / ``alpha``.
        Becomes each tile's ``stage`` property, which is what the app
        stylesheet turns into its hover colour, and what the legend at
        the foot of the aside is drawn from. Anything missing is stable.
    :param parent: parent widget; ownership only.
    """

    tile_clicked = Signal(str)
    #: Emitted when the person asks for a sample project of their own kind
    #: (GitHub #130). The window answers it with
    #: :func:`spacr.qt.widgets.sample_project.offer_a_sample_project`.
    sample_project_requested = Signal()
    #: Emitted when the page wants the window to run its update check.
    update_check_requested = Signal()
    #: Emitted once, after the News panel has been shown, to ask the window
    #: for a release list newer than the one bundled in this wheel. The
    #: window answers it on a worker thread; see
    #: :meth:`spacr.qt.app.MainWindow._refresh_news`.
    news_refresh_requested = Signal()

    #: Declared on the class so a paint that arrives mid-construction —
    #: a nested layout activation delivers one on some styles — finds an
    #: answer instead of an ``AttributeError``. ``AppScreen`` learned the
    #: same lesson; see its backdrop-state block.
    _ambient = None

    #: The masthead's logo label, or ``None`` on a build where the bundled
    #: artwork could not be read. Declared here so anything asking for the
    #: mark gets an answer rather than an ``AttributeError``.
    _hero_mark = None

    #: The tile, at 100 % font scale. One size for every tab, and read
    #: from :mod:`spacr.qt.theme` rather than written here, because the
    #: stylesheet needs the same numbers — see
    #: :data:`spacr.qt.theme.TILE_H` for why the height floor has to be
    #: expressible in QSS.
    #:
    #: ``TILE_MAX_W`` is how far a tile may stretch to reach the
    #: right-hand edge of its row. Without a cap, a band with two apps in
    #: it draws two tiles half a metre wide; without any stretch at all
    #: every row stops short of the pane's edge and the page reads as
    #: sparse, which is the complaint that started this redesign.
    TILE_MIN_W = TILE_W
    TILE_MAX_W = TILE_MAX_W
    TILE_H = TILE_H
    TILE_ICON_PX = TILE_ICON_PX

    #: Right-hand column width. Fixed: it holds numbers, and a column of
    #: numbers that reflows on every window resize is unreadable.
    ASIDE_W = 300

    def __init__(
        self,
        apps: List[Tuple[str, str, str, str]],
        icon_provider: Callable[[str], Optional[QIcon]],
        parent=None,
        *,
        section_notes: Optional[Dict[str, str]] = None,
        categories: Optional[Sequence[Tuple[str, Sequence[str]]]] = None,
        bands: Optional[Sequence[Tuple[str, Sequence[str]]]] = None,
        stages: Optional[Dict[str, str]] = None,
    ):
        """Build Home: the hero, the module tabs and the right-hand column.

        :param parent: parent widget.
        """
        super().__init__(parent)
        self._P = active_palette()
        from ..job_runner import JobRunner
        self._journal_jobs = JobRunner(self, app_key="home journal",
                                       user_visible=False)
        self._apps = list(apps)
        self._icon_provider = icon_provider
        self._section_notes = dict(section_notes or {})
        self._stages = dict(stages or {})
        self._by_key = {k: (k, n, d) for k, n, d, _s in self._apps}
        self._categories = self._grouping(categories)
        self._bands = self._grouping(bands)
        self._names = {k: n for k, n, _d, _s in self._apps}
        self._tile_hints: dict = {}
        #: (holder, grid, tiles, tile_width) per grid, so a resize can
        #: rewrap each one at its own column width.
        self._grids: List[Tuple[QWidget, QGridLayout, list, int]] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(SPACING["xl"], SPACING["lg"],
                               SPACING["xl"], SPACING["md"])
        col.setSpacing(SPACING["md"])

        col.addWidget(self._build_hero())

        self._running_host = QWidget()
        self._running_layout = QVBoxLayout(self._running_host)
        self._running_layout.setContentsMargins(0, 0, 0, 0)
        self._running_layout.setSpacing(SPACING["xs"])
        self._banners: List[RunningBanner] = []
        self._banner = self._new_running_banner()
        col.addWidget(self._running_host)

        from .collapsible_splitter import EDGE, CollapsibleSplitter
        from ..preferences import scaled_px
        split = CollapsibleSplitter(Qt.Horizontal,
                                    persist_key=_ASIDE_PERSIST_KEY)
        split.setChildrenCollapsible(False)
        split.add_pane(self._build_tabs(), "Apps", stretch=1)
        split.add_pane(self._build_aside(), _ASIDE_PANE, mode=EDGE,
                       fold_key=_ASIDE_FOLD_KEY, stretch=0,
                       extent=scaled_px(self.ASIDE_W))
        split.splitterMoved.connect(lambda *_a: self._rewrap_grids())
        split.pane_toggled.connect(lambda *_a: self._rewrap_grids())
        self._aside_split = split
        col.addWidget(split, 1)
        self._read_the_journal()

        from ..i18n import tr
        self._body_scroll = QScrollArea()
        self._body_scroll.setAccessibleName(tr("Home"))
        self._body_scroll.setFrameShape(QScrollArea.NoFrame)
        self._body_scroll.setWidgetResizable(True)
        self._body_scroll.setWidget(body)
        make_transparent(body, self._body_scroll)
        self._body_scroll.viewport().setAutoFillBackground(False)
        self._body_fit = _HomeBodyFit(body, self._body_scroll, self._tabs,
                                     self._rewrap_grids)
        outer.addWidget(self._body_scroll, 1)

        from .module_hint_bar import ModuleHintBar
        self._hint_bar = ModuleHintBar(_DEFAULT_HINT)
        outer.addWidget(self._hint_bar)

        from .. import bridge
        self._closing = False
        self._registry = bridge.registry()
        self._registry.changed.connect(self._on_runs_changed)

        self._ticker = QTimer(self)
        self._ticker.setInterval(1000)
        self._ticker.timeout.connect(self._refresh_run_banners)

        self._on_runs_changed()

        #: The drifting backdrop, or ``None``. Home takes the same animation
        #: the module screens do, so the page the user lands on is not the
        #: one page in the app that is flat.
        self._ambient = None
        self._install_ambient()

        self._clear_page_surfaces()
        self._apply_aside_text()

    def page_fill(self):
        """The flat colour Home paints itself, or ``None``.

        The same rule, and the same reasoning, as
        :meth:`spacr.qt.screens.app_screen.AppScreen.page_fill`: with an
        animation installed the animation is the page, with an image
        theme the window's wallpaper is, and otherwise it is this — a
        real colour rather than the ``bg`` slab that no page-opacity
        setting can reach.

        Never raises.
        """
        from ..preferences import resolve_effective_theme

        return self._page_fill_under(resolve_effective_theme)

    def _page_fill_for_a_frame(self):
        """:meth:`page_fill` for :meth:`paintEvent`, without a store read.

        The theme comes from :func:`spacr.qt.theme._painting_theme`,
        resolved once per application palette, because a paint runs on
        every frame of the window's field.
        """
        from ..theme import _painting_theme

        return self._page_fill_under(_painting_theme)

    def _page_fill_under(self, theme_of):
        """The page colour with the theme ``theme_of()`` names, or ``None``.

        :param theme_of: returns the effective theme name; anything it
            raises reads as no colour.
        """
        if (self._ambient is not None
                or getattr(self, "_uses_window_backdrop", False)):
            return None
        try:
            from ..theme import IMAGE_THEMES, page_colour
            theme = theme_of()
            if theme in IMAGE_THEMES:
                return None
            return QColor(page_colour(theme))
        except Exception:
            return None

    def paintEvent(self, event) -> None:
        """Paint the page under everything Home lays out.

        Does not chain to ``super()`` when it fills: the base
        implementation is what draws the stylesheet background, and that
        background is the slab being replaced.

        :param event: the paint event; passed to the base class only when no
            page fill colour is set.
        """
        colour = self._page_fill_for_a_frame()
        if colour is None:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), colour)
        finally:
            painter.end()

    def _install_ambient(self) -> None:
        """Put the ambient animation behind Home. Never raises.

        Home needs none of :class:`spacr.qt.screens.app_screen.AppScreen`'s
        ``_ambient_applied`` bookkeeping and no ``changeEvent`` handling:
        this page is rebuilt from scratch on every theme change (see the
        registry comment in ``__init__``), so a stale flat fill cannot
        survive one, and there is no second attempt to guard against.

        Ordered exactly as the module screens are: the preference is read
        *before* anything is constructed, because not building it is the
        cost the toggle exists to avoid, and the surfaces are cleared here
        only *after* a successful install so that the animation is already
        behind them when they go transparent.

        That ordering is no longer what decides whether a failed install
        leaves Home opaque, and the docstring used to claim it was.
        ``__init__`` clears the surfaces again unconditionally after
        calling this, because there is never nothing behind them —
        :meth:`paintEvent` paints :meth:`page_fill`. See the comment on
        that second call for why the old "stay opaque on failure" rule was
        the black-slab bug rather than a safety net.
        """
        if self._ambient is not None:
            return
        widget = None
        try:
            from ..preferences import (get_ambient_enabled,
                                       get_ambient_palette,
                                       get_ambient_theme)
            if not get_ambient_enabled():
                return
            from .ambient import (install_ambient,
                                  _the_heavy_import_lock_is_free)
            if not _the_heavy_import_lock_is_free():
                from PySide6.QtCore import QTimer

                QTimer.singleShot(120, self._install_ambient)
                return
            widget = install_ambient(
                self, None,
                theme=get_ambient_theme(),
                palette=get_ambient_palette(),
                backdrop=self._ambient_backdrop())
            self._clear_page_surfaces()
            self._ambient = widget
        except Exception as error:
            self._ambient = None
            self._discard_ambient(widget)
            try:
                from .ambient import _the_backdrop_wants_a_retry
            except Exception:                                # noqa: BLE001
                return
            if _the_backdrop_wants_a_retry(error):
                from PySide6.QtCore import QTimer

                QTimer.singleShot(120, self._install_ambient)

    @staticmethod
    def _ambient_backdrop():
        """The wallpaper the animation composites over, or ``None``.

        Only the image themes have one; every other theme paints over its
        own flat window colour.
        """
        try:
            from ..preferences import (resolve_effective_theme,
                                       theme_background_path)
            return theme_background_path(resolve_effective_theme())
        except Exception:
            return None

    def _discard_ambient(self, widget=None) -> None:
        """Unparent an ambient widget an aborted install left behind.

        ``install_ambient`` parents the widget before it finishes wiring it
        up, so an installer that raises part way through hands nothing back
        to unparent — and an invisible leftover is still a child with a live
        timer.
        """
        try:
            from .ambient import AmbientWidget
        except Exception:
            return
        seen = []
        if widget is not None:
            seen.append(widget)
        seen += [c for c in list(self.children())
                 if isinstance(c, AmbientWidget)]
        for child in seen:
            try:
                child.set_animating(False)
            except Exception:
                pass
            try:
                child.setParent(None)
                child.deleteLater()
            except Exception:
                pass

    def _clear_page_surfaces(self) -> None:
        """Stop Home's layout containers painting over the backdrop.

        The same layering rule the module screens use: containers that only
        *position* things go transparent, while the cards that carry text —
        the hero, the aside panels, the tile pane — keep painting a surface.
        Without this the animation runs, costs its frames, and reaches the
        eye through nothing but the gaps between widgets.

        Everything that only POSITIONS things is tagged. That is most of the
        page: every plain ``QWidget`` container inherits the blanket
        ``QWidget {{ background-color: bg }}`` rule, so an untagged one paints
        an opaque slab whatever the opacity preference says — it is the window
        colour, not a surface, which is why no amount of dialling reached it.

        Not tagged, on purpose: the rounded panel boxes and the tiles. Those
        are the things the user is meant to SEE, and they carry the page
        opacity themselves.
        """
        from ..theme import clear_container_surfaces, make_transparent
        from PySide6.QtWidgets import (QLabel, QScrollArea, QStackedWidget,
                                       QTabBar, QTabWidget)

        clear_container_surfaces(self)

        for bar in self.findChildren(QTabBar):
            make_transparent(bar)

        hero = self.findChild(QWidget, "Hero")
        if hero is not None:
            make_transparent(*hero.findChildren(QLabel))

        make_transparent(*(w for w in (
            getattr(self, "_running_host", None),
            getattr(self, "_hint_bar", None),
            getattr(self, "_tabs", None),
        ) if w is not None))

        tabs = getattr(self, "_tabs", None)
        if tabs is not None:
            for area in tabs.findChildren(QScrollArea):
                make_transparent(area, area.viewport())
            make_transparent(*tabs.findChildren(QStackedWidget))
            for i in range(tabs.count()):
                page = tabs.widget(i)
                if page is not None:
                    make_transparent(page)
        layout = self.layout()
        if layout is not None and layout.count():
            item = layout.itemAt(0)
            body = item.widget() if item is not None else None
            if body is not None:
                make_transparent(body)

    def _new_running_banner(self) -> RunningBanner:
        """Build a banner for a run that has just started.

        :returns: the banner.
        """
        banner = RunningBanner(self._icon_provider, self._names)
        banner.open_requested.connect(self.tile_clicked)
        self._running_layout.addWidget(banner)
        self._banners.append(banner)
        return banner

    def _refresh_run_banners(self) -> None:
        """Add and remove banners so they match the runs actually going.

        RECONCILED RATHER THAN APPENDED: a run that ends while Home is not
        visible leaves a banner claiming it is still running otherwise.
        """
        for banner in self._banners:
            if banner.isVisible():
                banner.refresh()

    def _grouping(
        self,
        given: Optional[Sequence[Tuple[str, Sequence[str]]]],
    ) -> List[Tuple[str, List[Tuple[str, str, str]]]]:
        """Normalise a caller's grouping into ``(title, [entry])``.

        ``None`` means "group ``apps`` by their own section, in the
        order the sections first appear" — the pre-#16i behaviour, and
        the only sensible default for the many tests that hand this
        widget four hand-written tuples.

        Unknown keys are dropped rather than raised on: a grouping is a
        *view* of the registry, and a view that names an app that no
        longer exists should lose the tile, not the page.
        """
        if given is None:
            grouped: Dict[str, List[Tuple[str, str, str]]] = {}
            for key, name, desc, section in self._apps:
                grouped.setdefault(section, []).append((key, name, desc))
            return list(grouped.items())
        out = []
        for title, keys in given:
            entries = [self._by_key[k] for k in keys if k in self._by_key]
            if entries:
                out.append((title, entries))
        return out

    def _build_hero(self) -> QWidget:
        """Build the masthead over the tiles."""
        P = self._P
        hero = QWidget()
        hero.setObjectName("Hero")
        try:
            hero.setStyleSheet(hero_qss())
        except Exception:                                # noqa: BLE001
            import logging

            logging.getLogger(__name__).debug(
                "the masthead would not take its surface", exc_info=True)
        row = QHBoxLayout(hero)
        row.setContentsMargins(SPACING["md"], SPACING["sm"],
                               SPACING["md"], SPACING["sm"])
        row.setSpacing(SPACING["md"])

        logo = _find_logo_pixmap()
        if logo is not None:
            label = _HeroMark(logo, HERO_LOGO_PX)
            label.setObjectName("HeroMark")
            label.setStyleSheet("background: transparent;")
            self._hero_mark = label
            row.addWidget(label)

        title = _HeroLabel("spaCR")
        title.setObjectName("HeroTitle")
        title.setStyleSheet(
            f"font-family: {QSS_FONT_FAMILY}; font-weight: 300;"
            f"font-size: {font_px(HERO_TITLE_PX)}px; color: {P['accent']};"
            "letter-spacing: -0.6px; background: transparent;")
        row.addWidget(title)

        from .loading_screen import strap_line
        subtitle = _HeroLabel(strap_line())
        subtitle.setObjectName("HeroSubtitle")
        subtitle.setWordWrap(True)
        row.addWidget(subtitle, 1)
        hero._text_fit = _HeroTextFit(hero, (title, subtitle))

        return hero

    def _build_tabs(self) -> QWidget:
        """Home (everything), then one tab per category.

        Home is not a summary of the categories, it *is* every app —
        which is what makes the categories optional rather than a
        hierarchy you have to navigate. It bands them by the same
        categories the tabs use, because thirty unlabelled tiles is a
        wall and because two groupings of the same thirty apps is one
        grouping too many.

        The tab list is *derived*: a category with no members gets no
        tab. That is not a special case to maintain, it is the reason
        there is no empty pane to open when a category's last app moves
        elsewhere. What counts as a member is the registry's business —
        see ``categories``.
        """
        self._tabs = QTabWidget()
        self._tabs.setObjectName("HomeTabs")
        self._tabs.setUsesScrollButtons(True)
        self._tabs.setDocumentMode(False)
        self._tabs.setStyleSheet(
            _tab_qss(self._P, self._pane_alpha()))
        from PySide6.QtWidgets import QStackedWidget
        from ..theme import make_transparent
        make_transparent(*self._tabs.findChildren(QStackedWidget))

        self._section_names = [title for title, _e in self._categories]

        from ..i18n import tr

        self._tabs.addTab(self._build_home_tab(),
                          f"{tr('Home')}  ({len(self._apps)})")
        for section, entries in self._categories:
            self._tabs.addTab(
                self._build_category_tab(section, entries),
                _escape_amp(f"{tr(section)}  ({len(entries)})"))
        return self._tabs


    def _build_home_tab(self) -> QWidget:
        """Build the first tab: every module, grouped by band."""
        from ..preferences import scaled_px
        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(SPACING["md"], SPACING["sm"],
                               SPACING["md"], SPACING["sm"])
        col.setSpacing(SPACING["xs"])

        width = scaled_px(self.TILE_MIN_W)
        for band, entries in self._bands:
            col.addWidget(self._band_header(band, len(entries)))
            holder = QWidget()
            grid = QGridLayout(holder)
            grid.setContentsMargins(0, 0, 0, SPACING["xs"])
            grid.setHorizontalSpacing(SPACING["xs"])
            grid.setVerticalSpacing(SPACING["xs"])
            tiles = [self._make_tile(k, n, d) for k, n, d in entries]
            self._grids.append((holder, grid, tiles, width))
            self._fill_grid(grid, tiles, 1)
            col.addWidget(holder)
        col.addStretch(1)
        return self._scrolled(page)

    def _band_header(self, title: str, count: int) -> QWidget:
        """One band's heading and its one-line description.

        :param title: the band's name.
        :param count: how many modules it holds.
        :returns: the header widget.
        """
        P = self._P
        wrap = QWidget()
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 2)
        col.setSpacing(3)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["sm"])
        label = QLabel(title.upper())
        label.setStyleSheet(
            f"font-family: {QSS_FONT_FAMILY}; font-weight: 600;"
            f"font-size: {font_px(11)}px; letter-spacing: 2px;"
            "background: transparent;"
            f"color: {P['fg_muted']};")
        note = QLabel(str(count))
        note.setStyleSheet(f"color: {P['fg_dim']}; font-size: {font_px(11)}px;"
                           "background: transparent;")
        row.addWidget(label)
        row.addWidget(note)
        row.addStretch(1)
        col.addLayout(row)
        col.addWidget(Divider())
        return wrap

    def _build_category_tab(self, section: str,
                            entries: List[Tuple[str, str, str]]) -> QWidget:
        """Build one band's own tab.

        :param section: the band to build.
        :param entries: its modules.
        :returns: the tab widget.
        """
        from ..preferences import scaled_px
        P = self._P
        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(SPACING["md"], SPACING["sm"],
                               SPACING["md"], SPACING["sm"])
        col.setSpacing(SPACING["xs"])

        head = QWidget()
        head_col = QVBoxLayout(head)
        head_col.setContentsMargins(0, 0, 0, 0)
        head_col.setSpacing(2)
        from ..i18n import tr
        heading = QLabel(tr(section).upper())
        heading.setStyleSheet(
            f"font-family: {QSS_FONT_FAMILY}; font-weight: 600;"
            f"font-size: {font_px(11)}px; letter-spacing: 2px;"
            "background: transparent;"
            f"color: {P['fg_muted']};")
        head_col.addWidget(heading)

        note = self._section_notes.get(section)
        if note:
            caption = QLabel(tr(note))
            caption.setObjectName("HomeSectionNote")
            caption.setWordWrap(True)
            caption.setStyleSheet(
                f"color: {P['fg_dim']}; font-size: {font_px(12)}px;"
                "background: transparent;")
            head_col.addWidget(caption)
        col.addWidget(head)

        col.addWidget(Divider())

        holder = QWidget()
        grid = QGridLayout(holder)
        grid.setContentsMargins(0, SPACING["xs"], 0, 0)
        grid.setHorizontalSpacing(SPACING["xs"])
        grid.setVerticalSpacing(SPACING["xs"])
        width = scaled_px(self.TILE_MIN_W)
        tiles = [self._make_tile(k, n, d) for k, n, d in entries]
        self._grids.append((holder, grid, tiles, width))
        self._fill_grid(grid, tiles, 1)
        col.addWidget(holder)
        col.addStretch(1)
        return self._scrolled(page)

    def _make_tile(self, key: str, name: str, desc: str) -> AppTile:
        """One tile. Same class, same size, on every tab."""
        from ..preferences import scaled_px
        icon = self._icon_provider(key) if self._icon_provider else None
        tile = AppTile(name, desc, icon,
                       width=scaled_px(self.TILE_MIN_W),
                       height=scaled_px(self.TILE_H),
                       icon_px=scaled_px(self.TILE_ICON_PX),
                       stage=self._stages.get(key, "stable"))
        tile.setMaximumWidth(scaled_px(self.TILE_MAX_W))
        tile.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        return self._wire_tile(tile, key, desc)

    def _wire_tile(self, tile, key: str, desc: str):
        """Connect one tile so pressing it opens its module, and so its
        Enter and Leave reach :meth:`eventFilter` (through
        :class:`_TileHintWatcher`, so its paint events never reach Python).

        :param tile: the tile.
        :param key: the module it opens.
        """
        self._tile_hints[tile] = (key, desc)
        from ..theme import STAGE_LABEL
        tile.setProperty("moduleAppKey", key)
        tile.setProperty("moduleNameSource", tile.text_label)
        tile.setProperty("moduleSummarySource", desc)
        tile.setProperty("moduleTooltipStyle", "tile")
        tile.setProperty("moduleStageSource", STAGE_LABEL.get(tile.stage, ""))
        if getattr(self, "_tile_watcher", None) is None:
            self._tile_watcher = _TileHintWatcher(self)
        tile.clicked.connect(lambda _=False, k=key: self.tile_clicked.emit(k))
        return tile

    def _scrolled(self, page: QWidget) -> QScrollArea:
        """Wrap a page in a scroll area.

        :param page: the widget to wrap.
        :returns: the scroll area.
        """
        scroll = QScrollArea()
        scroll.setFocusPolicy(Qt.NoFocus)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        from ..theme import make_transparent
        make_transparent(page, scroll)
        scroll.viewport().setAutoFillBackground(False)
        scroll.setWidget(page)
        return scroll

    @staticmethod
    def _fill_grid(grid: QGridLayout, tiles: list, columns: int) -> None:
        """(Re)place ``tiles`` into ``columns`` columns, packed to the top.

        Rows get zero stretch and one extra row takes all of it,
        otherwise QGridLayout shares the leftover height between the rows
        and the tiles drift apart down the page.

        The tiles widen to their column, up to ``AppTile``'s own maximum,
        so a row reaches both edges instead of leaving a ragged gap after
        each tile — the difference between a page that reads as full and
        one that reads as sparse. There used to be a second mode that
        packed fixed-size cards to the left; every tile is the same class
        now and every grid wants the same behaviour, so the flag went
        with the second tile size.
        """
        for tile in tiles:
            grid.removeWidget(tile)
        grid._home_columns = columns
        rows = 0
        for index, tile in enumerate(tiles):
            rows = index // columns
            grid.addWidget(tile, rows, index % columns)
        for row in range(grid.rowCount()):
            grid.setRowStretch(row, 0)
        grid.setRowStretch(rows + 1, 1)
        span = max(grid.columnCount(), columns)
        for column in range(span):
            grid.setColumnStretch(column, 1 if column < columns else 0)

    def refresh_pane_surfaces(self) -> int:
        """Give the masthead, the panels and the tabs the page opacity now set.

        They took it when Home was built, and a new opacity used to rebuild
        Home: 2.1-2.6 s of the GUI thread on a loaded box, for three kinds
        of rule. Each is now set again in place, one widget per pass of the
        event loop, and only where the rule changed, since setting a rule
        restyles everything under it.

        THE TABS ARE TOLD ON THEIR BAR. The tab widget's rule holds the
        whole of Home under it, and setting it again restyled every tile
        (1.1 s loaded) to change one tab's fill; the bar takes the selected
        tab's fill as a rule of its own, which wins over the one it
        inherits.

        :returns: how many widgets will take a new rule.
        """
        from PySide6.QtWidgets import QFrame

        wanted = []
        hero = self.findChild(QWidget, "Hero")
        if hero is not None:
            try:
                wanted.append((hero, hero_qss()))
            except Exception:                            # noqa: BLE001
                pass
        box_rule = panel_box_qss(self._P)
        wanted.extend((box, box_rule)
                      for box in self.findChildren(QFrame, "HomePanelBox"))
        tabs = getattr(self, "_tabs", None)
        if tabs is not None:
            wanted.append((tabs.tabBar(),
                           _selected_tab_qss(self._P, self._pane_alpha())))
        owed = []
        for widget, rule in wanted:
            try:
                if widget.styleSheet() != rule:
                    owed.append((widget, rule))
            except RuntimeError:
                continue

        def _next():
            """Set one owed rule, then wait for the next pass."""
            while owed:
                widget, rule = owed.pop(0)
                try:
                    widget.setStyleSheet(rule)
                except RuntimeError:
                    continue
                break
            if owed:
                QTimer.singleShot(0, _next)

        if owed:
            QTimer.singleShot(0, _next)
        return len(owed)

    @staticmethod
    def _pane_alpha() -> float:
        """Opacity of the rounded box behind the tiles.

        The user's ``pane_opacity`` preference, already clamped up to the
        theme's legibility floor by
        :func:`spacr.qt.preferences.effective_pane_alpha`. Falls back to
        fully opaque — what the page looked like before the preference
        existed — if preferences cannot be read at all.
        """
        try:
            from ..preferences import effective_pane_alpha
            return effective_pane_alpha()
        except Exception:
            return 1.0

    def _build_aside(self) -> QWidget:
        """Build the right-hand column: status panels, then the text slider.

        The column is an EDGE pane of Home's splitter, the same handle the
        module screens give their settings column: drag it to resize, click
        its arrow to hide or show the column; both are remembered. The panels
        and the slider under the lowest one sit in a scroll area, so large
        text scrolls instead of squeezing a card.
        """
        from ..preferences import _KEY_HOME_ASIDE_TEXT, _home_aside_scale
        from ..theme import make_transparent
        aside = QWidget()
        aside.setObjectName("HomeAside")
        make_transparent(aside)
        self._aside = aside
        outer = QVBoxLayout(aside)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        panels = QWidget()
        panels.setObjectName("HomeAsidePanels")
        self._aside_panels = panels
        col = QVBoxLayout(panels)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(SPACING["md"])

        from ..i18n import tr

        start = QPushButton(tr("Pipeline overviews"), panels)
        start.setObjectName("PrimaryButton")
        start.setToolTip(tr(
            "Explore pipeline flowcharts, their modules, inputs and outputs, "
            "and optionally start with example data."))
        start.clicked.connect(
            lambda _checked=False: self.sample_project_requested.emit())
        self._sample_project_button = start
        col.addWidget(start)

        self._queued = QueuedPanel()
        self._recent = RecentRunsPanel(known_keys=lambda: self._names,
                                       read_now=False)
        self._recent.run_clicked.connect(self._on_run_clicked)
        self._recent.cleared.connect(self.refresh)
        self._news = NewsPanel(self._version())
        self._news.check_requested.connect(self.update_check_requested)
        self._news.refresh_requested.connect(self.news_refresh_requested)
        self._totals = TotalsPanel(read_now=False)
        self._system = SystemPanel()
        self._system._refresh.clicked.connect(self._apply_aside_text)
        self._legend = StageLegend(self)
        self._legend.hide()

        for panel in (self._queued, self._recent, self._news,
                      self._totals, self._system):
            col.addWidget(panel)
        self._aside_text = _home_aside_scale(
            _KEY_HOME_ASIDE_TEXT, *_ASIDE_TEXT_RANGE)
        col.addWidget(self._build_aside_slider())
        col.addStretch(1)

        scroll = self._scrolled(panels)
        scroll.setObjectName("HomeAsideScroll")
        self._aside_scroll = scroll
        outer.addWidget(scroll, 1)
        self._fit_aside_minimum()
        panels._floor_fit = _AsideFloorFit(panels, self._fit_aside_minimum)
        return aside

    def _build_aside_slider(self) -> QWidget:
        """The compact Text size slider under the lowest panel."""
        from PySide6.QtWidgets import QSlider

        from ..i18n import tr

        box = QWidget()
        box.setObjectName("HomeAsideScaleControls")
        make_transparent(box)
        self._aside_controls = box
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["sm"])
        tip = tr("Make the text in this column's panels larger or smaller. "
                 "The setting is remembered. Default 100%.")
        P = self._P
        label = QLabel(tr("Text size"), box)
        label.setStyleSheet(f"color: {P['fg_muted']};"
                            f" font-size: {font_px(11)}px;"
                            " background: transparent;")
        slider = QSlider(Qt.Horizontal, box)
        slider.setObjectName("HomeAsideTextSlider")
        slider.setRange(int(round(_ASIDE_TEXT_RANGE[0] * 100)),
                        int(round(_ASIDE_TEXT_RANGE[1] * 100)))
        slider.setSingleStep(5)
        slider.setPageStep(10)
        slider.setValue(int(round(self._aside_text * 100)))
        readout = QLabel(f"{slider.value()}%", box)
        readout.setProperty("i18nSkipText", True)
        readout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        readout.setStyleSheet(label.styleSheet())
        readout.setMinimumWidth(QLabel("200%").sizeHint().width())
        for widget in (label, slider, readout):
            widget.setToolTip(tip)
        slider.setAccessibleName(label.text())
        label.setBuddy(slider)
        slider.valueChanged.connect(lambda v: readout.setText(f"{v}%"))
        slider.valueChanged.connect(self._on_aside_text)
        row.addWidget(label)
        row.addWidget(slider, 1)
        row.addWidget(readout)
        self._homeAsideTextSlider = slider
        return box

    def _aside_collapsed(self) -> bool:
        """Whether the right-hand column is folded away by its handle."""
        split = getattr(self, "_aside_split", None)
        return bool(split is not None and split.is_collapsed(_ASIDE_PANE))

    def _on_aside_text(self, percent: int) -> None:
        """Resize the right-hand column's text and remember the size."""
        from ..preferences import _KEY_HOME_ASIDE_TEXT, _set_home_aside_scale
        self._aside_text = _set_home_aside_scale(
            _KEY_HOME_ASIDE_TEXT, percent / 100.0, *_ASIDE_TEXT_RANGE)
        self._apply_aside_text()

    def _apply_aside_text(self) -> None:
        """Give the panels' text the stored size, then refit the width.

        Two passes, because the panels take their sizes from two places:
        the window's sheet, re-declared on the panel host at the new size by
        :func:`spacr.qt.live_zoom.scaled_font_sheet` (the same pass the
        module screens' Ctrl + wheel uses), and each label's own sheet, whose
        sizes are rewritten from the size it was built with. Panels rebuild
        rows on refresh, so this runs again after every refresh.
        """
        panels = self._aside_panels
        if self._aside_text == 1.0 and not getattr(
                self, "_aside_text_applied", False):
            return
        self._aside_text_applied = True
        _scale_text_under(panels, self._aside_text,
                          exempt=getattr(self, "_aside_controls", None))
        self._fit_aside_minimum()

    def _fit_aside_minimum(self) -> None:
        """Keep the column at least as wide as its content needs.

        Large text raises the floor, so a row widens the column instead of
        clipping; dragging the handle past the floor folds the column away,
        as it does the module screens' settings column.
        """
        aside = self._aside
        panels = self._aside_panels
        layout = panels.layout()
        layout.activate()
        floor = layout.totalMinimumSize().width()
        scroll = getattr(self, "_aside_scroll", None)
        if scroll is not None:
            floor += scroll.verticalScrollBar().sizeHint().width()
        from ..gui_scale import _actual_minimum_size, _set_actual_dim

        changed = _actual_minimum_size(aside).width() != floor
        if changed:
            _set_actual_dim(aside, 'minw', floor)
        split = getattr(self, "_aside_split", None)
        pane = split.pane(_ASIDE_PANE) if split is not None else None
        if pane is not None:
            pane.minimum = getattr(aside, '_gs_geo', {}).get('minw', aside.minimumWidth())
        if changed:
            self._rewrap_grids()

    def _rewrap_grids(self) -> None:
        """Re-flow every grid against its own actual tab viewport."""
        from ..gui_scale import _ORIGINAL

        for holder, grid, tiles, _tile_w in self._grids:
            ancestor = holder
            margins = 0
            while not isinstance(ancestor, QScrollArea):
                layout = ancestor.layout()
                if layout is not None:
                    left, _top, right, _bottom = layout.getContentsMargins()
                    margins += left + right
                ancestor = ancestor.parentWidget()
            available = max(0, ancestor.viewport().property('size').width() - margins)
            spacing = _ORIGINAL.get((QGridLayout, 'horizontalSpacing'),
                                    QGridLayout.horizontalSpacing)(grid)
            spacing = max(0, spacing)
            tile_width = max(max(tile.property('minimumSize').width(),
                                 tile.layout().totalMinimumSize().width()) for tile in tiles)
            columns = max(1, (available + spacing) // max(1, tile_width + spacing))
            if columns != getattr(grid, '_home_columns', None):
                self._fill_grid(grid, tiles, columns)

    def _on_run_clicked(self, key: str) -> None:
        """Open the module a Recent runs row names, if it still exists.

        The panel already filters to modules Home knows about, so this is
        the second of two guards rather than the only one. It is here
        because the filter depends on `known_keys` resolving, and a signal
        that navigates has to be safe when it does not: a row for a module
        that is not registered used to ask the window to open `_job`.
        """
        if key and key in self._names:
            self.tile_clicked.emit(key)

    @property
    def legend(self) -> "StageLegend":
        """The colour-to-maturity key at the foot of the right column."""
        return self._legend

    @staticmethod
    def _version() -> str:
        """The version string shown on the masthead.

        :returns: the version.
        """
        try:
            import spacr
            version = str(getattr(spacr, "__version__", "") or "").strip()
        except Exception:
            return ""
        return "" if version.lower() in ("", "dev", "unknown") else version

    def _on_runs_changed(self) -> None:
        """Show every active job across the top, oldest first."""
        if self._closing:
            return
        active = [h for h in self._registry.active()
                  if h.app_key and getattr(h, "user_visible", True)]
        while len(self._banners) < len(active):
            self._new_running_banner()
        for index, banner in enumerate(self._banners):
            banner.bind(active[index] if index < len(active) else None)
        if not active:
            self._ticker.stop()
        elif not self._ticker.isActive():
            self._ticker.start()

    def refresh(self) -> None:
        """Re-read everything that can change while Home is off screen.

        The two run-journal panels are read on a worker thread. Together
        ``recent_runs`` + ``journal_totals`` walk every manifest under the
        runs root twice — 774 ms on a machine with 4 865 journalled runs,
        measured, and it grows with the journal — and this used to run inline
        on every single return to Home, which is the most-travelled
        navigation in the application.

        The panels keep whatever they are already showing until the worker
        delivers; a stale count for half a second beats a frozen window, and
        on the first ever call they are showing their empty state anyway.
        Everything else here is cheap (a JSON read and three stat calls) and
        stays inline.
        """
        self._queued.refresh()
        self._system.refresh()
        self._on_runs_changed()
        self._apply_aside_text()
        self._read_the_journal()

    def _read_the_journal(self) -> None:
        """Read Recent runs and Totals on a worker; :meth:`_apply_journal`
        paints them. Home is built with both panels empty and filled from
        here, so the window never waits for the run journal."""
        recent, totals = self._recent, self._totals
        self._journal_jobs.cancel()
        self._journal_jobs.submit(
            lambda r=recent, t=totals: (r.read(), t.read()),
            self._apply_journal)

    def _apply_journal(self, payload) -> None:
        """Paint the worker's journal read. GUI thread only."""
        runs, totals = payload
        self._recent.refresh(runs)
        self._totals.refresh(totals)
        self._apply_aside_text()

    def active_jobs(self) -> int:
        """How many journal-reading threads are still winding down."""
        return self._journal_jobs.active_jobs()

    def set_reserved_content(self, widget: QWidget) -> None:
        """Fill the featured/news surface with real content.

        :param widget: the widget to show in the featured/news panel in place
            of the release notes.
        """
        self._news.set_content(widget)
        self._apply_aside_text()

    def apply_release_news(self, releases) -> None:
        """Hand a fetched release list to the News panel.

        The answer to :attr:`news_refresh_requested`, and the only way in:
        the window never reaches into the panel, so a page rebuilt at a new
        font scale simply asks again.

        :param releases: records from
            :func:`spacr.updater.fetch_release_notes`, or anything at all.
        """
        self._news.apply_releases(releases)
        self._apply_aside_text()

    @property
    def news_panel(self) -> "NewsPanel":
        """The News panel. For tests and for the window's own wiring."""
        return self._news

    @property
    def _reserved_content(self) -> Optional[QWidget]:
        """The widget currently filling the news surface, if any."""
        return self._news.content

    def resizeEvent(self, event):               # noqa: N802
        """Re-flow the tile grid for the new width.

        :param event: the Qt resize event.
        """
        super().resizeEvent(event)
        self._rewrap_grids()

    def show_module_hint(self, key: str, summary: str = "") -> bool:
        """Explain ``key`` in the strip. Called by the DOCK as well as Home.

        The dock's rows and Home's tiles name the same modules, so they say
        the same thing in the same place -- `MainWindow._show_module_hint`
        routes a dock hover here whenever Home is the page on screen.

        :param key: the module to explain.
        :param summary: the sentence, already resolved and translated by the
            caller. Empty falls back to Home's own registry, which is what a
            tile hover uses -- it has the description in hand and has no
            reason to ask the window for it.
        :returns: whether anything was written. A key Home does not know is
            not an error: the dock lists Help modules that have no tile.
        """
        key = str(key or "")
        from ..theme import STAGE_LABEL

        if not summary:
            entry = self._by_key.get(key)
            if entry is None:
                return False
            from ..i18n_module_summaries import module_summary
            summary = module_summary(key, entry[2])
        if not summary:
            return False
        stage = STAGE_LABEL.get(str(self._stages.get(key, "stable")), "")
        self._hint_bar.show_module(key, summary, stage)
        return True

    def eventFilter(self, obj, event):          # noqa: N802
        """Show a tile's summary on Enter and release it on Leave.

        :class:`_TileHintWatcher` hands it the tiles' events; it is not
        installed on any widget.

        :param obj: the object the event is for.
        :param event: the event.
        :returns: True to stop the event going further.
        """
        if event.type() == QEvent.Enter:
            hint = self._tile_hints.get(obj)
            if hint:
                from ..i18n_module_summaries import module_summary
                from ..theme import STAGE_LABEL
                key, source = hint
                summary = module_summary(key, source)
                mark = STAGE_LABEL.get(
                    str(obj.property("stage") or "stable"), "")
                self._hint_bar.show_module(key, summary, mark)
        elif event.type() == QEvent.Leave:
            if not self._hint_bar.is_holding():
                self._hint_bar.release()
        return super().eventFilter(obj, event)

    def closeEvent(self, event):                # noqa: N802
        """Stop Home-page background activity before closing.

        Shut down the journal reader, disconnect run-registry notifications,
        and stop the refresh ticker before delegating to the base close
        handler. This prevents pending work from invoking a page that Qt is
        destroying.

        :param event: the close event; it is passed on to the base class after
            background activity stops.
        """
        self._closing = True
        self._journal_jobs.shutdown()
        try:
            self._registry.changed.disconnect(self._on_runs_changed)
        except (RuntimeError, TypeError):
            pass
        self._ticker.stop()
        super().closeEvent(event)


def _selected_tab_qss(P: dict, pane_alpha: float = 1.0) -> str:
    """The selected Home tab's fill alone, for the tab bar to wear.

    :param P: the active palette.
    :param pane_alpha: the page opacity.
    :returns: the rule, as :func:`_tab_qss` writes it for the same fill.
    """
    from ..theme import css_color
    selected_fill = ("transparent" if pane_alpha <= 0.0
                     else css_color(P["surface"], pane_alpha))
    return (f"QTabWidget#HomeTabs > QTabBar::tab:selected {{"
            f" background: {selected_fill};"
            f" border-bottom-color: {selected_fill}; }}")


def _tab_qss(P: dict, pane_alpha: float = 1.0) -> str:
    """Return styling for the transparent Home tab container.

    ``pane_alpha`` controls only the selected tab's surface fill; the pane and
    tab-bar backgrounds remain transparent. The pane has no decorative rim;
    the selected tab's own edge is the meaningful state indicator.
    """
    from ..theme import css_color
    selected_fill = ("transparent" if pane_alpha <= 0.0
                     else css_color(P["surface"], pane_alpha))
    return f"""
QTabWidget#HomeTabs::pane {{
    border: none;
    background: transparent;
    top: -1px;
}}
/* The BAR, not the tabs on it. Qt builds `qt_tabwidget_tabbar` itself, and
   with no rule of its own it takes the blanket window fill — measured as the
   last opaque strip on the page after everything else was cleared. Tagging
   the widget is not enough: the stylesheet wins over the property for this
   one, so it needs saying here. */
QTabWidget#HomeTabs > QTabBar {{
    background: transparent;
}}
QTabWidget#HomeTabs > QTabBar::tab {{
    background: transparent;
    color: {P['fg_muted']};
    border: 1px solid transparent;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 7px 14px;
    margin-right: 2px;
    font-size: {font_px(13)}px;
}}
QTabWidget#HomeTabs > QTabBar::tab:hover {{
    color: {P['fg']};
    background: {P['surface_alt']};
}}
/* The selected tab takes the page opacity like everything else. Its own edge
   carries selection without drawing a decorative rim around the empty pane. */
QTabWidget#HomeTabs > QTabBar::tab:selected {{
    color: {P['accent']};
    background: {selected_fill};
    border: 1px solid {P['border_soft']};
    border-bottom-color: {selected_fill};
}}
"""


_EASY_DIRECTORY_LIMIT = 24
_EASY_FILE_LIMIT = 1032
_EASY_ENTRY_LIMIT = 8192
_EASY_HEADER_LIMIT = 8192
_EASY_METADATA_LIMIT = 16384
_EASY_METADATA_NAMES = ('metadata.json', 'acquisition.json', 'experiment.json',
                        'MeasurementProtocol.xml', 'ImagingResult.xml', 'MeasurementDetail.mrf')


def _easy_read_header(path, limit, offset=0):
    """Read bounded regular-file bytes without following a symbolic link or opening a device."""
    import stat

    original = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(original.st_mode):
        raise ValueError('Not a regular file')
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0))
    try:
        current = os.fstat(descriptor)
        if not stat.S_ISREG(current.st_mode) or (original.st_dev, original.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError('Not a regular file')
        if offset < 0 or offset > current.st_size:
            raise ValueError('Invalid header offset')
        os.lseek(descriptor, offset, os.SEEK_SET)
        return os.read(descriptor, limit)
    finally:
        os.close(descriptor)


def _easy_axes_metadata(raw):
    """Retain explicit T/Z sizes from bounded TIFF descriptions without guessing axes."""
    import json

    text = raw.rstrip(b'\x00').decode('utf-8', errors='replace')
    try:
        value = json.loads(text)
    except ValueError:
        value = {}
    if isinstance(value, dict) and isinstance(value.get('axes'), str) and isinstance(value.get('shape'), list):
        axes, shape = value['axes'], value['shape']
        if len(axes) == len(shape) and all(isinstance(size, int) and size > 0 for size in shape):
            dimensions = dict(zip(axes.upper(), shape))
            return dict(axes=axes, shape=shape, t=dimensions.get('T', 1), z=dimensions.get('Z', 1))
    dimensions = {axis: int(value) for axis, value in re.findall(r'\bSize([TZC])=["\'](\d+)["\']', text)}
    if dimensions:
        return dict(t=dimensions.get('T', 1), z=dimensions.get('Z', 1), axes='OME')
    return {}


def _easy_external_format(path, raw=b''):
    """Recognize requested external document families without opening their readers."""
    name = path.name.lower()
    families = {
        'flowjo': ('.wsp', '.wst', '.acs', '.fcs'),
        'prism': ('.pzf', '.pzfx', '.prism'),
        'chemdraw': ('.cdx', '.cdxml'),
        'structure': ('.pdb', '.cif', '.mmcif', '.pdb.gz', '.cif.gz', '.mmcif.gz'),
    }
    for family, suffixes in families.items():
        if name.endswith(suffixes):
            return family
    if path.suffix.lower() in ('.mrc', '.rec', '.map', '.npy', '.npz'):
        return None
    text = raw[:4096].decode('utf-8', errors='replace').strip()
    if raw.startswith((b'FCS2.', b'FCS3.')) or re.match(r'(?is)^(?:<\?xml[^>]*>\s*)?<(?:Workspace|FlowJo)\b', text):
        return 'flowjo'
    if re.match(r'(?is)^(?:<\?xml[^>]*>\s*)?<GraphPadPrismFile\b', text):
        return 'prism'
    if re.match(r'(?is)^(?:<\?xml[^>]*>\s*)?<CDXML\b', text):
        return 'chemdraw'
    if re.match(r'(?i)^(?:AlphaFold\b|PDB\s+(?:model|structure)\b)', text):
        return 'structure'
    return None


def _easy_external_guidance(family):
    """Offer original-software guidance without inventing scientific readers."""
    from ..i18n import tr

    return {
        'flowjo': tr('Open this FlowJo file in FlowJo. To use its measurements in spaCR, export a numeric CSV for Graph Builder or Gate Editor.'),
        'prism': tr('Open this GraphPad Prism file in GraphPad Prism. To use its measurements in spaCR, export a numeric CSV for Graph Builder or Gate Editor.'),
        'chemdraw': tr('Open this ChemDraw structure in ChemDraw or compatible chemistry software. spaCR does not read chemical structure documents.'),
        'structure': tr('Open this PDB/mmCIF or AlphaFold model in ChimeraX or PyMOL. spaCR does not analyze molecular structures.'),
    }[family]


def _easy_external_reference(value):
    """Recognize explicit non-path document references without network access."""
    from pathlib import Path

    text = str(value).strip()[:4096]
    if Path(text).suffix.lower() in ('.mrc', '.rec', '.map', '.npy', '.npz'):
        return None
    for family, pattern in (('flowjo', r'\bflowjo\b'),
                            ('prism', r'\b(?:graphpad(?:\s+prism)?|prism\s+(?:file|project|workspace))\b'),
                            ('chemdraw', r'\bchemdraw\b'),
                            ('structure', r'\b(?:pdb|mmcif|alphafold)\b')):
        if re.search(pattern, text, re.I):
            return family
    return None


def _easy_file_kind(path, metadata=None):
    """Inspect bounded local headers without decoding image pixels or loading arrays."""
    import csv
    import io
    from pathlib import Path

    from PIL import Image, UnidentifiedImageError

    header = _easy_read_header(path, 16)
    if header[:4] in (b'II*\x00', b'MM\x00*', b'II+\x00', b'MM\x00+'):
        from ...channel_sorting import IMAGE_EXTS

        endian = 'little' if header[:2] == b'II' else 'big'
        big = int.from_bytes(header[2:4], endian) == 43
        if big and header[4:8] != (b'\x08\x00\x00\x00' if endian == 'little' else b'\x00\x08\x00\x00'):
            return 'unsupported'
        offset = int.from_bytes(header[8:16] if big else header[4:8], endian)
        count_size, entry_size = (8, 20) if big else (2, 12)
        raw_count = _easy_read_header(path, count_size, offset)
        if len(raw_count) != count_size:
            return 'unsupported'
        count = int.from_bytes(raw_count, endian)
        if not count or count > (_EASY_HEADER_LIMIT - 16 - count_size) // entry_size:
            return 'unsupported'
        entries = _easy_read_header(path, count * entry_size, offset + count_size)
        if len(entries) != count * entry_size:
            return 'unsupported'
        dimensions = {}
        description = None
        organized = False
        for index in range(count):
            entry = entries[index * entry_size:(index + 1) * entry_size]
            tag = int.from_bytes(entry[:2], endian)
            kind = int.from_bytes(entry[2:4], endian)
            values = int.from_bytes(entry[4:12] if big else entry[4:8], endian)
            if tag in (273, 324) and values > 0 and kind in (3, 4, 16):
                width = {3: 2, 4: 4, 16: 8}[kind]
                inline = entry[12:] if big else entry[8:]
                pointer = int.from_bytes(inline[:width] if values * width <= len(inline) else inline, endian)
                organized = 0 < pointer < os.stat(path, follow_symlinks=False).st_size
            if tag in (256, 257) and values == 1 and kind in (3, 4, 16):
                width = {3: 2, 4: 4, 16: 8}[kind]
                inline = entry[12:] if big else entry[8:]
                if width <= len(inline):
                    dimensions[tag] = int.from_bytes(inline[:width], endian)
            if tag == 270 and kind == 2 and 0 < values <= 4096:
                inline = entry[12:] if big else entry[8:]
                description = (values, inline)
        if organized and dimensions.get(256, 0) > 0 and dimensions.get(257, 0) > 0:
            try:
                Image._decompression_bomb_check((dimensions[256], dimensions[257]))
            except Image.DecompressionBombError:
                return 'unsupported'
            if metadata is not None:
                metadata.update(shape=[dimensions[257], dimensions[256]], axes='YX', axes_unresolved=True)
                if description:
                    size, inline = description
                    remaining = _EASY_HEADER_LIMIT - 16 - count_size - len(entries)
                    if size <= remaining:
                        raw = inline[:size] if size <= len(inline) else _easy_read_header(
                            path, size, int.from_bytes(inline, endian))
                        explicit = _easy_axes_metadata(raw)
                        metadata.update(explicit)
                        if explicit:
                            metadata['axes_unresolved'] = False
            return 'image' if Path(path).suffix.lower() in IMAGE_EXTS else 'filename_format'
        return 'unsupported'
    header += _easy_read_header(path, _EASY_HEADER_LIMIT - 16, len(header))
    if header.startswith(b'\x1f\x8b'):
        import zlib

        try:
            sequence_header = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(header, _EASY_HEADER_LIMIT)
        except zlib.error:
            sequence_header = b''
    else:
        sequence_header = header
    lines = sequence_header.splitlines()
    external = _easy_external_format(path, sequence_header)
    if external:
        return 'external_' + external
    if header.startswith(b'%PDF-'):
        return 'pdf'
    if (len(lines) >= 4 and lines[0].startswith(b'@') and lines[2].startswith(b'+')
            and lines[1] and len(lines[1]) == len(lines[3])
            and set(lines[1].upper()) <= set(b'ACGTNRYKMSWBDHV.')):
        return 'fastq'
    if header.startswith(b'SQLite format 3\x00'):
        return 'database'
    if header.startswith(b'\x93NUMPY'):
        from numpy.lib.format import read_array_header_1_0, read_array_header_2_0, read_magic

        stream = io.BytesIO(header)
        version = read_magic(stream)
        reader = read_array_header_1_0 if version == (1, 0) else read_array_header_2_0
        shape, _, dtype = reader(stream, max_header_size=_EASY_HEADER_LIMIT)
        import math

        valid = (2 <= len(shape) <= 5 and all(size > 0 for size in shape)
                 and not dtype.hasobject and dtype.kind in 'buifc'
                 and stream.tell() + math.prod(shape) * dtype.itemsize
                 <= os.stat(path, follow_symlinks=False).st_size)
        if metadata is not None:
            metadata.update(shape=list(shape), dtype=str(dtype), dtype_kind=dtype.kind,
                            axes='', axes_unresolved=True)
        return 'array' if valid else 'unsupported'
    try:
        with Image.open(io.BytesIO(header)) as image:
            if image.width > 0 and image.height > 0:
                from ...channel_sorting import IMAGE_EXTS
                from ...convert import _image_extensions

                suffix = Path(path).suffix.lower()
                if suffix in IMAGE_EXTS:
                    return 'image'
                return 'import_image' if suffix in _image_extensions() else 'filename_format'
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        pass
    try:
        text = header.decode('utf-8-sig')
        dialect = csv.Sniffer().sniff(text, delimiters=',\t;')
        rows = list(csv.reader(io.StringIO(text), dialect))[:8]
        if len(rows) >= 2 and len(rows[0]) >= 2:
            return 'table' if Path(path).suffix.lower() in ('.csv', '.tsv', '.txt') else 'filename_format'
    except (UnicodeError, csv.Error):
        pass
    from ...convert import _image_extensions
    from ..dnd_handlers import _MASK_FILE_EXTS

    extensions = _image_extensions()
    if Path(path).suffix.lower() in set(_MASK_FILE_EXTS) - set(extensions) and header.startswith(b'PK\x03\x04'):
        return 'native_container'
    if Path(path).suffix.lower() in extensions and Path(path).suffix.lower() not in ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'):
        return 'container'
    return 'unsupported'


def _easy_table_evidence(path):
    """Inspect a bounded delimited sample for numeric features and sequencing identities."""
    import csv
    import io
    import math

    raw = _easy_read_header(path, _EASY_HEADER_LIMIT).decode('utf-8-sig')
    dialect = csv.Sniffer().sniff(raw, delimiters=',\t;')
    rows = list(csv.reader(io.StringIO(raw), dialect))[:8]
    if len(rows) < 2:
        return {}
    columns = rows[0]
    numeric = False
    for row in rows[1:]:
        for value in row:
            try:
                numeric = numeric or bool(value.strip()) and math.isfinite(float(value))
            except ValueError:
                pass
    sequencing = ({'rowID', 'columnID'} <= set(columns)
                  and bool({'grna', 'grna_name'} & set(columns))
                  and bool({'count', 'counts', 'reads'} & set(columns)))
    return dict(numeric=numeric, sequencing=sequencing, columns=columns[:64])


def _easy_analysis_inputs(root, details=None):
    """Reserve bounded discovery for scientific databases beneath immediate plate folders."""
    from itertools import islice

    folders = [root]
    try:
        with os.scandir(root) as entries:
            folders.extend(type(root)(entry.path) for entry in islice(entries, 128)
                           if not entry.name.startswith('.') and entry.is_dir(follow_symlinks=False))
    except OSError:
        return []
    found = []
    for folder in folders[:17]:
        for candidate in (folder / 'measurements' / 'measurements.db', folder / 'measurements.db'):
            if candidate.is_file() and not any(part.is_symlink() for part in (candidate, candidate.parent)):
                found.append(candidate)
    if details is not None:
        details.update(sampled_plate_candidates=min(len(folders), 17),
                       truncated=len(folders) > 17 or len(found) > 8)
    return found[:8]


def _easy_database_evidence(path):
    """Read only bounded SQLite schema metadata, never measurement rows."""
    import sqlite3
    from ...schema import IMAGE_MEASUREMENT_TABLES, OBJECT_TABLES, ORGANELLE_SUMMARY_TABLES

    connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=.25)
    try:
        connection.execute('PRAGMA query_only=ON')
        connection.set_progress_handler(lambda: 1, 100000)
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' LIMIT 32")]
        numeric = False
        objects = False
        for name in tables[:16]:
            if name not in OBJECT_TABLES + ORGANELLE_SUMMARY_TABLES + IMAGE_MEASUREMENT_TABLES:
                continue
            columns = connection.execute('PRAGMA table_info("' + name.replace('"', '""') + '")').fetchmany(256)
            names = {row[1] for row in columns}
            numeric = numeric or any(any(kind in str(row[2]).upper() for kind in ('INT', 'REAL', 'FLOAT', 'DOUBLE', 'NUM'))
                                     for row in columns if row[1] not in
                                     {'plateID', 'rowID', 'columnID', 'fieldID', 'timeID', 'object_label', 'label', 'prc', 'prcf', 'prcfo'})
            objects = objects or (name in OBJECT_TABLES and {'plateID', 'rowID', 'columnID'} <= names)
        annotation_only = bool(tables) and all(name.startswith(('image_annotations', 'sqlite_')) for name in tables)
        return dict(tables=tables, numeric=numeric, objects=objects, crops='png_list' in tables,
                    annotation_only=annotation_only)
    finally:
        connection.close()


def _easy_saved_settings(path):
    """Recognize bounded saved settings without applying values or evaluating expressions."""
    import csv
    import io
    import json
    import sys

    from ..app_catalog import DECLARED_APPS
    from ..settings_pack import PACK_FILES, _HEADER_ROWS, _coerce

    names = {name.rsplit('.', 1)[0]: key for key, files in PACK_FILES.items() for name in files}
    keys = {app.key for app in DECLARED_APPS} | set(PACK_FILES)
    registry = sys.modules.get('spacr.qt.app')
    keys.update(getattr(registry, 'APP_META', {}))
    folded = getattr(registry, 'folded_children', None)
    if callable(folded):
        keys.update(key for children in folded().values() for key in children)
    names.update({key + '_settings': key for key in keys})
    module = names.get(path.stem)
    if path.stem == 'settings':
        manifest = path.parent / 'manifest.json'
        if manifest.is_file() and not manifest.is_symlink():
            raw = _easy_read_header(manifest, _EASY_METADATA_LIMIT + 1)
            value = json.loads(raw) if len(raw) <= _EASY_METADATA_LIMIT else None
            if (isinstance(value, dict) and isinstance(value.get('app_key'), str)
                    and value['app_key'] in keys):
                module = value['app_key']
    if module is None or path.suffix.lower() not in ('.json', '.csv'):
        return None
    raw = _easy_read_header(path, _EASY_METADATA_LIMIT + 1)
    if len(raw) > _EASY_METADATA_LIMIT:
        return None
    if path.suffix.lower() == '.json':
        values = json.loads(raw)
    else:
        rows = list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))))
        if not rows or tuple(cell.lower() for cell in rows[0]) not in _HEADER_ROWS:
            return None
        values = {row[0]: _coerce(row[1]) for row in rows[1:] if len(row) == 2}
    if not isinstance(values, dict) or not values or not all(isinstance(key, str) for key in values):
        return None
    return dict(path=str(path), module=module, settings=values)


def _easy_paired_layout(images, root):
    """Pair unambiguous sampled names and verify merged headers before scientific routes."""
    from itertools import islice

    from ...drop_classification import _has_mask_word, _image_stem, _mask_stem

    local = [str(path) for path in images if path.parent == root or path.parent.parent == root]
    masks = [path for path in local if _has_mask_word(path)]
    originals = [path for path in local if path not in masks]
    pairs = {}
    for image in originals:
        matches = [mask for mask in masks if _image_stem(image) in (_image_stem(mask), _mask_stem(mask))]
        peers = [other for other in originals if _image_stem(other) == _image_stem(image)]
        if len(matches) == len(peers) == 1:
            pairs[image] = matches[0]
    if not pairs:
        return None
    merged = None
    for folder in (root,):
        candidate = folder / 'merged'
        if not candidate.is_dir() or candidate.is_symlink():
            continue
        try:
            with os.scandir(candidate) as entries:
                sampled = list(islice(entries, 32))
            for entry in sampled:
                if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith('.npy'):
                    path = type(root)(entry.path)
                    metadata = {}
                    if _easy_file_kind(path, metadata) == 'array' and len(metadata.get('shape', [])) == 3:
                        merged = folder
                        break
        except (OSError, ValueError, EOFError):
            continue
        if merged:
            break
    return dict(kind='spacr' if merged else 'unstructured', pairs=pairs,
                project=str(merged or root), filename_evidence=True)


def _easy_run_evidence(root, files, include_runs=True):
    """Read bounded run manifests and coefficient-table headers without model execution."""
    import json
    from itertools import islice

    candidates = [root / 'manifest.json'] if include_runs else []
    runs = root / 'runs'
    if include_runs and runs.is_dir() and not runs.is_symlink():
        with os.scandir(runs) as entries:
            candidates.extend(type(root)(entry.path) / 'manifest.json' for entry in islice(entries, 16)
                              if entry.is_dir(follow_symlinks=False))
    candidates.extend(path for path in files if path.name == 'manifest.json')
    manifests = []
    run_folders = []
    for path in dict.fromkeys(candidates):
        if len(manifests) >= 8:
            break
        if not path.is_file() or path.is_symlink():
            continue
        try:
            raw = _easy_read_header(path, _EASY_METADATA_LIMIT + 1)
            value = json.loads(raw) if len(raw) <= _EASY_METADATA_LIMIT else None
        except (OSError, ValueError, RecursionError):
            continue
        if (not isinstance(value, dict) or not isinstance(value.get('app_key'), str)
                or not value['app_key'] or not value.get('schema_version')):
            continue
        run_folders.append(path.parent)
        status = value.get('status')
        if status in ('failed', 'partial', 'running', 'cancelled', 'incomplete'):
            expected = {'mask': ('merged',), 'measure': ('measurements/measurements.db',)}.get(value['app_key'], ())
            missing = [name for name in expected if not (root / name).exists()]
            manifests.append(dict(path=str(path), status=status, app_key=str(value['app_key']),
                                  missing_outputs=missing))
    results = []
    result_candidates = list(files)
    for folder in run_folders[:8]:
        for name in ('results.csv', 'results_gene.csv', 'results_grna.csv'):
            path = folder / name
            if path.is_file() and not path.is_symlink() and path not in result_candidates:
                result_candidates.append(path)
    for path in result_candidates:
        if len(results) >= 4:
            break
        if path.name not in ('results.csv', 'results_gene.csv', 'results_grna.csv'):
            continue
        try:
            evidence = _easy_table_evidence(path)
        except (OSError, ValueError, UnicodeError):
            continue
        if {'feature', 'coefficient'} <= set(evidence.get('columns', [])):
            diagnostics = path.parent / 'diagnostics'
            diagnostics_folder = None
            if diagnostics.is_dir() and not diagnostics.is_symlink():
                try:
                    with os.scandir(diagnostics) as entries:
                        if any(entry.is_file(follow_symlinks=False) for entry in islice(entries, 16)):
                            diagnostics_folder = str(diagnostics)
                except OSError:
                    pass
            results.append(dict(path=str(path), folder=str(path.parent),
                                diagnostics=bool(diagnostics_folder), diagnostics_folder=diagnostics_folder))
    return dict(manifests=manifests, results=results)


def _easy_small_json(path):
    """Read a small regular metadata object without following links or opening arrays."""
    import json

    if any(part.is_symlink() for part in (path, *path.parents)):
        return {}
    try:
        raw = _easy_read_header(path, _EASY_METADATA_LIMIT + 1)
        value = json.loads(raw) if len(raw) <= _EASY_METADATA_LIMIT else None
    except (OSError, ValueError, UnicodeError, RecursionError):
        return {}
    return value if isinstance(value, dict) else {}


def _easy_merged_evidence(project):
    """Verify a bounded numeric merged-image header before suggesting spatial mapping."""
    from itertools import islice

    folder = project / 'merged'
    if not folder.is_dir() or folder.is_symlink():
        return False
    try:
        with os.scandir(folder) as entries:
            candidates = [type(project)(entry.path) for entry in islice(entries, 32)
                          if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith('.npy')]
        for path in candidates[:3]:
            metadata = {}
            if (_easy_file_kind(path, metadata) == 'array' and len(metadata['shape']) == 3
                    and metadata['shape'][-1] > 1):
                return True
    except (OSError, ValueError, EOFError):
        pass
    return False


def _easy_specialized_inputs(root, files, images, schemas):
    """Seed only verified acquisition paths and identity settings for real OPS/Spatial pages."""
    import math
    import sqlite3
    from itertools import islice

    from ...ops_engine import _match_tile, _st_platform
    from ..i18n import tr

    hints = {}
    cycled, uncycled = [], []
    for image in images:
        if _match_tile(image.name, 'cycled'):
            cycled.append(image.parent)
        elif _match_tile(image.name, 'uncycled'):
            uncycled.append(image.parent)
    genotype = next((folder for folder in cycled if folder == root or folder.parent == root), None)
    if genotype:
        settings = dict(genotype_source=str(genotype))
        phenotype = next((folder for folder in uncycled if folder != genotype
                          and folder.parent == root), None)
        if phenotype:
            settings['phenotype_source'] = str(phenotype)
        for path in files[:32]:
            if path.suffix.lower() != '.csv' or (path.parent != root and path.parent.parent != root):
                continue
            try:
                columns = _easy_table_evidence(path).get('columns', [])
            except (OSError, ValueError, UnicodeError):
                continue
            if {'prefix', 'barcode', 'sequence'} & set(columns):
                settings['ops_library'] = str(path)
                break
        hints['ops'] = dict(settings=settings, source=str(genotype),
                            evidence=tr('OPS tile filenames were found; review acquisition sources before continuing.'))
    folders = list(dict.fromkeys([root] + [path.parent for path in files
                                                     if path.parent.parent == root]))[:16]
    for folder in folders:
        if folder.is_symlink():
            continue
        try:
            platform = _st_platform(str(folder))
        except (OSError, ValueError):
            continue
        slide_candidates, bins = [], []
        if platform == 'xenium':
            slide_candidates = [folder / name for name in ('morphology_focus.ome.tif',
                                'morphology_mip.ome.tif', 'morphology.ome.tif')]
            focus = folder / 'morphology_focus'
            if focus.is_dir() and not focus.is_symlink():
                try:
                    with os.scandir(focus) as entries:
                        slide_candidates.extend(type(root)(entry.path) for entry in islice(entries, 16)
                                                if entry.is_file(follow_symlinks=False)
                                                and entry.name.lower().endswith(('.ome.tif', '.ome.tiff')))
                except OSError:
                    pass
        else:
            slide_candidates = [path for path in files if path.parent == folder
                                and path.name.lower().endswith(('.tif', '.tiff', '.btf'))]
            scale_paths = [folder / 'spatial' / 'scalefactors_json.json']
            binned = folder / 'binned_outputs'
            if binned.is_dir() and not binned.is_symlink():
                try:
                    with os.scandir(binned) as entries:
                        for entry in islice(entries, 16):
                            match = re.fullmatch(r'square_(\d{3})um', entry.name)
                            if match and int(match[1]) > 0 and entry.is_dir(follow_symlinks=False):
                                bins.append(int(match[1]))
                                scale_paths.append(type(root)(entry.path) / 'spatial' / 'scalefactors_json.json')
                except OSError:
                    pass
            calibrated = False
            for scale in scale_paths[:17]:
                values = _easy_small_json(scale)
                value = values.get('microns_per_pixel') or values.get('spot_diameter_fullres')
                try:
                    calibrated = calibrated or math.isfinite(float(value)) and float(value) > 0
                except (TypeError, ValueError):
                    pass
            if not calibrated:
                continue
        slide = None
        for path in slide_candidates[:4]:
            try:
                metadata = {}
                kind = _easy_file_kind(path, metadata)
                if kind in ('image', 'import_image') or (kind == 'filename_format' and metadata.get('shape')):
                    slide = path
                    break
            except (OSError, ValueError, EOFError):
                continue
        if slide is None:
            continue
        settings = dict(spatial_source=str(folder), spatial_platform=platform)
        if bins:
            settings['spatial_bin_um'] = 2 if 2 in bins else min(bins)
        hints.setdefault('spatial_masks', dict(settings=settings, source=str(folder), image=str(slide),
                          evidence=tr('Spatial acquisition metadata and a morphology image were found.')))
        if platform == 'xenium':
            counts_ready = (folder / 'transcripts.parquet').is_file() and not (folder / 'transcripts.parquet').is_symlink()
        else:
            if platform == 'visium_hd' and not bins:
                continue
            count_folder = (folder / 'binned_outputs' / f"square_{settings['spatial_bin_um']:03d}um"
                            if platform == 'visium_hd' and bins else folder)
            positions = any((count_folder / 'spatial' / name).is_file() for name in
                            ('tissue_positions.parquet', 'tissue_positions.csv', 'tissue_positions_list.csv'))
            values = _easy_small_json(count_folder / 'spatial' / 'scalefactors_json.json')
            value = values.get('microns_per_pixel') or values.get('spot_diameter_fullres')
            try:
                calibrated = math.isfinite(float(value)) and float(value) > 0
            except (TypeError, ValueError):
                calibrated = False
            try:
                with os.scandir(count_folder) as entries:
                    counts_ready = positions and calibrated and any(entry.is_file(follow_symlinks=False)
                        and entry.name.endswith('filtered_feature_bc_matrix.h5') for entry in islice(entries, 32))
            except OSError:
                counts_ready = False
        if not counts_ready:
            continue
        for schema in schemas:
            if 'spatial_fields' not in schema['tables']:
                continue
            database = type(root)(schema['path'])
            project = database.parent.parent if database.parent.name == 'measurements' else database.parent
            if not _easy_merged_evidence(project):
                continue
            try:
                connection = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=.25)
                try:
                    connection.execute('PRAGMA query_only=ON')
                    connection.set_progress_handler(lambda: 1, 100000)
                    sources = connection.execute('SELECT substr(source, 1, 4096) FROM spatial_fields LIMIT 8').fetchall()
                finally:
                    connection.close()
            except (OSError, sqlite3.Error):
                continue
            if not any(isinstance(row[0], str) and os.path.abspath(os.path.expanduser(row[0])) == str(folder)
                       for row in sources):
                continue
            map_settings = dict(settings, spatial_masks_dir=str(project))
            hints.setdefault('spatial_map', dict(settings=map_settings, source=str(folder),
                             evidence=tr('Spatial measurements match this acquisition; review masks before continuing.')))
    return hints


def _easy_local_sample(path, sample_limit=100):
    """Sample only the source and one child level within global budgets."""
    from itertools import islice
    from pathlib import Path
    import random

    root = path if path.is_dir() else path.parent
    queue = [(root, True)]
    files = [] if path.is_dir() else [path]
    seen_files, seen_dirs = set(files), set()
    rng = random.Random(733)
    count = len(files)
    entries_left, unreadable, limited = _EASY_ENTRY_LIMIT, 0, False
    while queue and len(seen_dirs) < _EASY_DIRECTORY_LIMIT:
        folder, descend = queue.pop(0)
        if folder in seen_dirs or folder.is_symlink():
            continue
        seen_dirs.add(folder)
        try:
            with os.scandir(folder) as entries:
                allowance = min(entries_left, 4096 if folder == root else 128)
                sample = list(islice(entries, allowance + 1))
            if len(sample) > allowance:
                limited = True
                sample = sample[:allowance]
            entries_left -= len(sample)
            for entry in sample:
                if entry.name.startswith('.') or entry.is_symlink():
                    continue
                candidate = Path(entry.path)
                if entry.is_file(follow_symlinks=False) and candidate not in seen_files:
                    count += 1
                    if len(files) < sample_limit:
                        files.append(candidate)
                    else:
                        limited = True
                        position = rng.randrange(count)
                        if 8 <= position < sample_limit and (path.is_dir() or files[position] != path):
                            files[position] = candidate
                    seen_files.add(candidate)
                elif descend and entry.is_dir(follow_symlinks=False):
                    queue.append((candidate, False))
        except OSError:
            unreadable += 1
        if not entries_left:
            limited = True
            break
    return files, tuple(sorted(seen_dirs)), unreadable, limited or bool(queue)


def _easy_filename_hints(images):
    """Rank existing conventions per image folder without applying a regex or renaming."""
    from ...regex_infer import _metadata_autodetect, _metadata_convention

    hints = []
    for folder in dict.fromkeys(path.parent for path in images):
        own = [path for path in images if path.parent == folder]
        for suffix in sorted({path.suffix.lower() for path in own}):
            names = [path.name for path in own if path.suffix.lower() == suffix]
            for key, matched, total in _metadata_autodetect(names, suffix.lstrip('.'))[:3]:
                record = _metadata_convention(key)
                hints.append(dict(folder=str(folder), convention=key, matched=matched,
                                  sampled=total, status=record.get('status', 'provisional')))
    return hints


def _easy_metadata_hints(folders):
    """Read at most four named sidecars and retain only bounded structural hints."""
    import json

    hints = []
    for folder in folders:
        for name in _EASY_METADATA_NAMES:
            candidate = folder / name
            if candidate.is_symlink() or not candidate.is_file():
                continue
            if len(hints) >= 4:
                return hints
            hint = dict(name=name, folder=str(folder), fields=[], truncated=False)
            try:
                raw = _easy_read_header(candidate, _EASY_METADATA_LIMIT + 1)
                hint['truncated'] = len(raw) > _EASY_METADATA_LIMIT
                if name.endswith('.json') and not hint['truncated']:
                    value = json.loads(raw)
                    if isinstance(value, dict):
                        hint['fields'] = sorted(set(value) & {'microscope', 'metadata_type', 'channels',
                            'pixel_size_um', 'plate_id', 'well_id'})
            except (OSError, ValueError, UnicodeError, RecursionError):
                hint['unreadable'] = True
            hints.append(hint)
    return hints


def _easy_validate_regex_inline(result, pattern):
    """Validate an explicit filename pattern through the same parser the Mask run will use."""
    import re

    from ..i18n import tr
    from ..regex_detect import KNOWN_FIELDS, apply_regex, validate_records
    from ...import_plan import for_get_regex
    from ...regex_infer import _metadata_parse_report, _metadata_pattern

    proposal = dict(result.get('regex') or {})
    names = proposal.get('sample_filenames') or []
    updated = dict(result)
    try:
        if not isinstance(pattern, str) or not pattern.strip() or len(pattern) > 4096 or not 0 < len(names) <= 1000:
            raise ValueError('Unbounded filename mapping')
        compiled = re.compile(pattern)
        records, missed = apply_regex(names, pattern)
        warnings = validate_records(records, multichannel='chanID' in compiled.groupindex)
        required = [warning for warning in warnings if not warning.startswith('Optional:')]
        required.extend(warning for record in records
                        for warning in validate_records([record], multichannel='chanID' in compiled.groupindex)
                        if not warning.startswith('Optional:'))
        stored = for_get_regex(pattern)
        extension = proposal['img_format']
        report = _metadata_parse_report(names, 'custom', extension, stored)
        downstream = re.compile(_metadata_pattern('custom', extension, stored))
        if not names or missed or required or report[:2] != (len(names), len(names)) or not all(
                compiled.fullmatch(name) and downstream.fullmatch(name) for name in names):
            raise ValueError('Incomplete filename mapping')
        example = proposal['example_path']
        from pathlib import Path

        match = compiled.fullmatch(Path(example).name)
        if match is None:
            raise ValueError('The example does not match')
        offset = len(example) - len(Path(example).name)
        groups = [dict(name=name, start=offset + match.start(name), end=offset + match.end(name), color_index=index)
                  for index, name in enumerate(KNOWN_FIELDS)
                  if name in compiled.groupindex and match.start(name) >= 0]
        proposal.update(pattern=pattern, required=False, groups=groups)
        proposal.pop('validation_error', None)
        updated.update(regex=proposal, settings_overrides={'mask': dict(
            metadata_type='custom', custom_regex=stored)})
        updated.pop('error', None)
    except (re.error, ValueError, KeyError, TypeError):
        proposal.update(pattern=pattern, required=True, validation_error=tr(
            'The pattern must match every sampled filename and capture the field and channel identities required by the import parser.'))
        updated['regex'] = proposal
        updated.pop('settings_overrides', None)
    return updated


def _easy_regex_worker(proposal, pattern, connection):
    """Return primitive parser results from an isolated explicit-validation process."""
    import json

    try:
        result = _easy_validate_regex_inline({'regex': json.loads(proposal)}, pattern)
        payload = json.dumps(result, ensure_ascii=False).encode('utf-8')
    except Exception:
        payload = b'null'
    try:
        connection.send_bytes(payload)
    except (OSError, EOFError):
        pass
    finally:
        connection.close()


def _easy_validate_regex(result, pattern, cancelled=None):
    """Bound arbitrary explicit patterns outside the GUI process and retire the owned child."""
    import json
    import multiprocessing
    import time

    from ..i18n import tr

    updated = dict(result)
    proposal = dict(result.get('regex') or {})
    proposal.update(pattern=pattern, required=True)
    updated['regex'] = proposal
    updated.pop('settings_overrides', None)
    if callable(cancelled) and cancelled():
        return updated
    names = proposal.get('sample_filenames') or []
    if not isinstance(pattern, str) or len(pattern) > 4096 or not 0 < len(names) <= 1000:
        proposal['validation_error'] = tr('This pattern could not be validated within the local limits. Simplify it and try again.')
        return updated
    payload = json.dumps({key: proposal[key] for key in (
        'pattern', 'example_path', 'source_folder', 'sample_filenames', 'img_format', 'required')
        if key in proposal}, ensure_ascii=False)
    if len(payload.encode('utf-8')) > 1048576:
        proposal['validation_error'] = tr('This pattern could not be validated within the local limits. Simplify it and try again.')
        return updated
    context = multiprocessing.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_easy_regex_worker, args=(payload, pattern, sender))
    started = False
    try:
        process.start()
        started = True
        sender.close()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if callable(cancelled) and cancelled():
                return updated
            if receiver.poll(0.025):
                reply = json.loads(receiver.recv_bytes(1048576))
                if isinstance(reply, dict) and isinstance(reply.get('regex'), dict):
                    updated.update(reply)
                    return updated
                break
            if not process.is_alive():
                break
    except (OSError, EOFError, ValueError):
        pass
    finally:
        receiver.close()
        sender.close()
        if started:
            if process.is_alive():
                process.terminate()
            process.join(timeout=0.25)
            if process.is_alive():
                process.kill()
                process.join()
        process.close()
    proposal['validation_error'] = tr('This pattern could not be validated within the local limits. Simplify it and try again.')
    return updated


def _easy_regex_evidence(images, chosen, hints):
    """Propose a local editable pattern without applying it or claiming acquisition metadata."""
    import re

    from ..regex_detect import KNOWN_FIELDS, auto_detect_regex
    from ...regex_infer import _metadata_parse_report

    own = [image for image in images if image.parent == chosen]
    if not own:
        return None
    example = own[0]
    extension = example.suffix.lstrip('.')
    names = [image.name for image in own if image.suffix == example.suffix]
    if any(hint['folder'] == str(chosen)
           and _metadata_parse_report(names, hint['convention'], extension)[:2] == (len(names), len(names))
           for hint in hints):
        return None
    pattern, _, _ = auto_detect_regex(names)
    pattern = pattern or ''
    match = re.match(pattern, example.name) if pattern else None
    groups = []
    if match:
        for index, name in enumerate(KNOWN_FIELDS):
            if name in match.re.groupindex and match.start(name) >= 0:
                offset = len(str(example)) - len(example.name)
                groups.append(dict(name=name, start=offset + match.start(name),
                                   end=offset + match.end(name), color_index=index))
    return dict(pattern=pattern, example_path=str(example), source_folder=str(chosen),
                sample_filenames=names, img_format=extension, groups=groups, required=True)


def _easy_generated_sample(folder, depth=4):
    """Inspect a fixed directory/entry budget inside a canonical output folder."""
    from itertools import islice

    pending, files, directories, entries_left = [(folder, 0)], [], 0, 1024
    while pending and directories < 24 and entries_left > 0:
        current, level = pending.pop(0)
        if current.is_symlink():
            continue
        directories += 1
        try:
            with os.scandir(current) as entries:
                sampled = list(islice(entries, min(entries_left, 128)))
            entries_left -= len(sampled)
            for entry in sampled:
                if entry.name.startswith('.') or entry.is_symlink():
                    continue
                candidate = type(folder)(entry.path)
                if entry.is_file(follow_symlinks=False):
                    files.append(candidate)
                elif level < depth and entry.is_dir(follow_symlinks=False):
                    pending.append((candidate, level + 1))
        except OSError:
            continue
    return files


def _easy_generated_outputs(path, files, schemas, run_evidence, calibrated=False):
    """Recognize bounded canonical outputs and enumerate their shared workflow consumers."""
    import sqlite3
    from pathlib import Path

    from ...ports import CROPS, MASKS, MERGED_ARRAYS, MODEL_WEIGHTS, PORTS
    from ..i18n import tr

    empty = dict(project='', artifacts=[], routes={})
    canonical = {Path(port.path).parts[0] for spec in PORTS.values()
                 for port in spec.produces if port.path and '/' not in port.path.strip('/')}
    canonical.update(('measurements',))
    folder = path if path.is_dir() else path.parent
    owner = next((parent for parent in (folder, *tuple(folder.parents)[:4])
                  if parent.name in canonical), None)
    if path.is_file() and (owner is None or
            not (path.suffix.lower() == '.npy' and owner.name in ('merged', 'masks')
                 or path.suffix.lower() == '.pth' and owner.name == 'model'
                 or path.suffix.lower() == '.png' and owner.name == 'data'
                 and path.parent.name.endswith('_png')
                 or path.name == 'measurements.db' and owner.name == 'measurements'
                 or any(item['path'] == str(path) for item in run_evidence['results']))):
        return empty
    project = owner.parent if owner is not None else folder
    if owner is None:
        discovered = next((item for item in schemas if item['objects'] and item['numeric']), None)
        if discovered is not None:
            discovered_database = Path(discovered['path'])
            discovered_project = (discovered_database.parent.parent
                                  if discovered_database.parent.name == 'measurements'
                                  else discovered_database.parent)
            if discovered_project.is_relative_to(folder):
                project = discovered_project
    if calibrated or os.path.lexists(project / 'plate_batch.json') or os.path.lexists(
            project / 'psf' / 'segmentation_application.json'):
        return empty
    artifacts = []
    port_kinds = {MERGED_ARRAYS: 'merged', MASKS: 'masks', CROPS: 'crops',
                  MODEL_WEIGHTS: 'classifier'}
    samples = {}
    for spec in PORTS.values():
        for port in spec.produces:
            if port.kind not in port_kinds or not port.path or '/' in port.path:
                continue
            target = project / port.path
            if target in samples or not target.is_dir() or target.is_symlink():
                continue
            if owner is not None and owner != target:
                continue
            if path.is_file():
                candidates = [path]
            elif port.kind in (CROPS, MODEL_WEIGHTS):
                candidates = _easy_generated_sample(target)
            else:
                candidates = [item for item in files if item.is_relative_to(target)]
                if not candidates:
                    candidates = _easy_generated_sample(target, depth=1)
            samples[target] = candidates
            checked = 0
            for candidate in candidates:
                if checked >= 4:
                    break
                suffix = candidate.suffix.lower()
                if port.kind in (MERGED_ARRAYS, MASKS) and suffix != '.npy':
                    continue
                if port.kind == CROPS and (suffix != '.png' or not any(
                        parent.name.endswith('_png') for parent in candidate.parents
                        if parent.is_relative_to(target))):
                    continue
                if port.kind == MODEL_WEIGHTS and suffix != '.pth':
                    continue
                checked += 1
                try:
                    metadata = {}
                    if port.kind == MODEL_WEIGHTS:
                        header = _easy_read_header(candidate, 16)
                        valid = header.startswith((b'PK\x03\x04', b'\x80\x02', b'\x80\x03', b'\x80\x04', b'\x80\x05'))
                    else:
                        kind = _easy_file_kind(candidate, metadata)
                        shape = metadata.get('shape', ())
                        valid = (kind == 'image' if port.kind == CROPS else
                                 kind == 'array' and metadata.get('dtype_kind') in
                                 ('buif' if port.kind == MERGED_ARRAYS else 'bui')
                                 and ((len(shape) in (3, 4) and shape[-1] >= 2)
                                      if port.kind == MERGED_ARRAYS else len(shape) in (2, 3)))
                except (OSError, ValueError, EOFError):
                    continue
                if valid:
                    producer = 'measure' if port.kind == CROPS else 'classify_merged' if port.kind == MODEL_WEIGHTS else 'mask'
                    artifacts.append(dict(kind=port_kinds[port.kind], path=str(candidate), producer=producer))
                    break
    scientific = next((item for item in schemas if item['objects'] and item['numeric']
                       and Path(item['path']).parent in (project, project / 'measurements')), None)
    database = project / 'measurements' / 'measurements.db'
    if scientific is None and artifacts and database.is_file() and not database.is_symlink():
        try:
            evidence = _easy_database_evidence(database)
            if evidence['objects'] and evidence['numeric']:
                scientific = dict(path=str(database), **evidence)
        except (OSError, sqlite3.Error):
            pass
    if scientific:
        artifacts.append(dict(kind='measurements', path=scientific['path'], producer='measure'))
        if scientific['crops']:
            artifacts.append(dict(kind='crops', path=scientific['path'], producer='measure'))
    for result in run_evidence['results']:
        if Path(result['path']).is_relative_to(project):
            artifacts.append(dict(kind='regression', path=result['path'], producer='regression'))
    if not artifacts:
        return empty
    from ..app import APPS, folded_children
    from ..dnd_handlers import NO_DROP_TARGET
    from .fold_strip import fold_label
    from .workflow_diagram import workflow_map

    data = workflow_map()
    modules = data['modules']
    rows = {row[0]: row for row in APPS}
    for children in folded_children().values():
        for key in children:
            title, guidance, _stage = fold_label(key)
            rows.setdefault(key, (key, title, guidance, ''))
    available = {item['kind'] for item in artifacts}
    future = set(available)
    selected = []
    while True:
        children = [key for key, module in modules.items() if key in rows and key not in selected
                    and future.intersection(module.get('inputs', ()))
                    and not set(module.get('inputs', ())) == {'raw_images'}]
        if not children:
            break
        selected.extend(children)
        for key in children:
            future.update(modules[key].get('outputs', ()))
    for producer in dict.fromkeys(item['producer'] for item in artifacts):
        if producer in rows and producer not in selected:
            selected.append(producer)
    if 'merged' in available and 'measure' in selected:
        selected.remove('measure')
        selected.insert(0, 'measure')
    routes = {}
    for key in selected:
        if key == 'mask':
            continue
        inputs = set(modules.get(key, {}).get('inputs', ()))
        ready = False
        source = project
        producers = {item['producer'] for item in artifacts}
        prerequisite = (key if key in producers else 'measure'
                        if 'measurements' in future or 'crops' in future else artifacts[0]['producer'])
        if key == 'measure' and 'merged' not in available and 'measure' not in producers:
            prerequisite = 'mask'
        if key == 'measure' and 'merged' in available:
            ready = True
        elif (scientific and inputs.intersection(('measurements', 'crops', 'labels', 'scores'))
              and not (key in ('annotate', 'umap') and not scientific['crops'])):
            ready = True
            source = project if key in ('annotate', 'classify_merged', 'qc_dashboard', 'run_compare') else Path(scientific['path'])
        elif key in ('hit_list', 'profiler', 'volcano_explorer') and 'regression' in available:
            result = next(item for item in artifacts if item['kind'] == 'regression')
            source = Path(result['path']).parent if key == 'hit_list' else Path(result['path'])
            ready = True
        elif key == 'regression_diagnostics':
            diagnostics = next((item['diagnostics_folder'] for item in run_evidence['results']
                                if item.get('diagnostics_folder')), None)
            if diagnostics:
                source = Path(diagnostics)
                ready = True
        if key in NO_DROP_TARGET:
            ready = False
            source = project
            prerequisite = key
        specification = PORTS.get(key)
        required_port_kinds = dict(port_kinds, **{'measurements-db': 'measurements'})
        if specification:
            required = {required_port_kinds.get(port.kind, port.kind)
                        for port in specification.consumes if port.required}
        else:
            required = {'measurements'} if 'measurements' in inputs else inputs.copy()
        required.update(inputs.intersection(('classifier', 'feature_model', 'segmentation_model', 'evaluation')))
        if 'measurements' not in inputs:
            required.update(inputs.intersection(('labels', 'scores')))
        if key in ('annotate', 'umap'):
            required.add('crops')
        missing = required - available
        if ready and missing:
            ready = False
            source = project
            prerequisite = next((producer for producer, module in modules.items()
                                 if producer in rows and producer != key
                                 and missing.intersection(module.get('outputs', ()))), prerequisite)
        title = tr(rows[key][1])
        prerequisite_title = tr(rows.get(prerequisite, (prerequisite, prerequisite))[1])
        reason = (tr('Review merged arrays and mask channels.') if key == 'measure' and ready else
                  tr(rows[key][2]) if ready or key == prerequisite else
                  tr('Run {module} first, then review this module’s inputs before processing.', module=prerequisite_title))
        routes[key] = dict(source=str(source), project=str(project), prerequisite=prerequisite,
                           ready=ready, title=title, reason=reason, missing_inputs=sorted(missing))
        if ready and scientific:
            routes[key]['database'] = scientific['path']
    return dict(project=str(project), artifacts=artifacts, routes=routes)


def _easy_inspect_path_impl(value):
    """Infer local next steps from bounded headers, filename conventions and nearby folder structure."""
    from pathlib import Path
    from importlib.util import find_spec
    import sqlite3

    from ..i18n import tr

    try:
        path = Path(os.path.expandvars(value.strip())).expanduser().absolute()
    except RuntimeError:
        return {'path': value, 'message': tr('Choose an existing local file or folder.'), 'actions': []}
    try:
        exists = path.exists()
    except OSError:
        external = _easy_external_reference(value)
        if external:
            return dict(path=str(path), message=_easy_external_guidance(external), actions=[],
                        context=dict(unsupported_external=external, source=value), confidence='external_reference')
        raise
    if value.strip() and not exists:
        external = _easy_external_format(path) or _easy_external_reference(value)
        if external:
            return dict(path=str(path), message=_easy_external_guidance(external), actions=[],
                        context=dict(unsupported_external=external, source=value), confidence='external_reference')
    if not value.strip() or not exists:
        return {'path': value, 'message': tr('Choose an existing local file or folder.'), 'actions': []}
    if any(part.is_symlink() for part in (path, *path.parents)):
        return {'path': str(path), 'message': tr('Choose the original local file or folder rather than a symbolic link.'), 'actions': []}
    path = Path(os.path.normpath(path))
    if not path.is_file() and not path.is_dir():
        return {'path': str(path), 'message': tr('Choose a regular file or folder, rather than a device or pipe.'), 'actions': []}
    external = _easy_external_format(path) if path.is_file() else None
    if external:
        return dict(path=str(path), message=_easy_external_guidance(external), actions=[],
                    context=dict(unsupported_external=external, source=str(path)), confidence='external_format')
    from ...convert import _image_extensions, _READERS
    from ..dnd_handlers import _MASK_FILE_EXTS

    image_extensions = _image_extensions()
    sample_limit = 100
    files, folders, unreadable, limited = _easy_local_sample(path, sample_limit)
    names = [candidate for candidate in files if candidate.suffix.lower() in image_extensions]
    hints = _easy_filename_hints(names)
    if limited and not any(hint['matched'] == hint['sampled'] and hint['sampled'] for hint in hints):
        sample_limit = 1000
        files, folders, unreadable, limited = _easy_local_sample(path, sample_limit)
        names = [candidate for candidate in files if candidate.suffix.lower() in image_extensions]
        hints = _easy_filename_hints(names)
    kinds = {}
    representatives = [] if path.is_dir() else [path]
    seen = {(item.parent, item.suffix.lower()) for item in representatives}
    root = path if path.is_dir() else path.parent
    for candidate in files:
        identity = (candidate.parent, candidate.suffix.lower())
        own = candidate.parent == root or candidate.parent.parent == root
        maximum = 8 if own else 16
        if identity not in seen and len(representatives) < maximum:
            representatives.append(candidate)
            seen.add(identity)
    classified = {}
    image_dimensions = []
    for candidate in representatives:
        try:
            dimensions = {}
            kind = _easy_file_kind(candidate, dimensions)
        except (OSError, ValueError, EOFError):
            unreadable += 1
            continue
        kinds.setdefault(kind, []).append(candidate)
        classified[(candidate.parent, candidate.suffix.lower())] = kind
        if dimensions:
            image_dimensions.append(dict(path=str(candidate), **dimensions))
    for candidate in names:
        kind = classified.get((candidate.parent, candidate.suffix.lower()))
        if kind in ('image', 'container', 'import_image') and candidate not in kinds.get(kind, []):
            kinds.setdefault(kind, []).append(candidate)
    if path.is_file():
        selected = next((kind for kind, candidates in kinds.items() if path in candidates), 'unsupported')
        if selected == 'image':
            kinds = {'image': [item for item in kinds['image'] if item.parent == path.parent]}
            files = list(kinds['image'])
        else:
            kinds = {selected: [path]}
            files = [path]
        folders = [path.parent]
        names = [item for item in files if item in kinds.get('image', [])]
        hints = _easy_filename_hints(names)
        image_dimensions = [item for item in image_dimensions
                            if Path(item['path']) in kinds.get('image', []) or item['path'] == str(path)]
    images = kinds.get('image', [])
    external_documents = [dict(kind=kind.removeprefix('external_'), path=str(candidate))
                          for kind, candidates in kinds.items() if kind.startswith('external_')
                          for candidate in candidates[:4]]
    root = path if path.is_dir() else path.parent
    local_images = [image for image in images if image.parent == root]
    own_images = [image for image in images if image.parent == root or image.parent.parent == root]
    own_data = any(candidate.parent == root or candidate.parent.parent == root
                   for kind in ('array', 'table', 'database', 'container', 'native_container', 'import_image') for candidate in kinds.get(kind, []))
    candidates = own_images if own_images else ([] if own_data else images)
    candidate_folders = sorted({image.parent for image in candidates})
    image_folders = sorted({image.parent for image in images})
    chosen = root if local_images else (candidate_folders[0] if candidate_folders else root)
    actions = []
    input_map = {}
    saved_settings = None
    if path.is_file():
        try:
            saved_settings = _easy_saved_settings(path)
        except (OSError, ValueError, UnicodeError, RecursionError, SyntaxError):
            pass
    if saved_settings:
        actions.append((saved_settings['module'], tr('Open saved settings'),
                        tr('Preview these settings before applying them.'), str(path)))
    pdf = next((item for item in kinds.get('pdf', [])
                if item.parent == root or item.parent.parent == root), None)
    if pdf is not None:
        actions.append(('analyze_plaques', tr('Plaque Assay'),
                        tr('Review this PDF in Plaque Assay before running.'), str(pdf)))
        input_map['analyze_plaques'] = dict(source=str(pdf), pdf=str(pdf))
    native_container = (path.is_file() and path.suffix.lower() in _MASK_FILE_EXTS
                        and path in kinds.get('array', []) + kinds.get('native_container', []))
    if native_container:
        actions.append(('mask', tr('Preview a container dataset'),
                        tr('Review the native extraction preview before generating masks.'), str(path)))
    array_imports = []
    from ...convert import _describe_npy
    from ...errors import ConfigurationError

    for array in kinds.get('array', [])[:4]:
        if array.parent != root and array.parent.parent != root:
            continue
        try:
            description = _describe_npy(str(array))
        except (ConfigurationError, OSError, ValueError, EOFError) as exc:
            array_imports.append(dict(path=str(array), supported=False, reason=str(exc)))
            continue
        array_imports.append(dict(path=str(array), supported=True, axes=description['axes']))
        actions.append(('convert', tr('Import microscopy images'),
                        tr('Review the reader, embedded metadata and conversion plan before importing.'), str(array.parent)))
    npy_image_source = None
    if kinds.get('array'):
        from ..mask_engine import _npy_image_header

        for array in kinds['array'][:4]:
            if array.parent != root and array.parent.parent != root:
                continue
            try:
                _npy_image_header(array)
            except (OSError, ValueError, EOFError):
                continue
            npy_image_source = str(array if path.is_file() else array.parent)
            actions = [action for action in actions if action[0] != 'mask']
            if path.is_file():
                native_container = False
            actions.append(('make_masks', tr('View and edit images'),
                            tr('Inspect images and edit masks.'), npy_image_source))
            input_map['make_masks'] = dict(source=npy_image_source)
            break
    if candidates:
        exact = path if path.is_file() and path in candidates else chosen
        actions.extend([
            ('make_masks', tr('View and edit images'), tr('Inspect images and edit masks.'), str(exact)),
            ('mask', tr('Create segmentation masks'), tr('Create segmentation masks for images in {folder}.', folder=chosen.name), str(chosen)),
        ])
        if not local_images:
            for folder in candidate_folders[1:3]:
                actions.append(('mask', tr('Create masks: {folder}', folder=folder.name),
                                tr('Another candidate folder; review channels before Run.'), str(folder)))
    dimension_routes = {}
    for item in image_dimensions:
        image = Path(item['path'])
        if image.parent != root and image.parent.parent != root:
            continue
        if item.get('axes_unresolved', True):
            continue
        for key, dimension, title in (('3d', 'z', tr('3D')), ('timelapse', 't', tr('Time'))):
            size = item.get(dimension, 1)
            if size > 1 and key not in dimension_routes:
                route = dict(source=str(image), host='mask', dimension=dimension, size=size)
                dimension_routes[key] = route
                input_map[key] = route
                actions.append((key, title, tr('Review the native extraction preview before generating masks.'), str(image)))
    containers = [item for item in kinds.get('container', [])
                  if item.parent == root or item.parent.parent == root]
    import_images = [item for item in kinds.get('import_image', [])
                     if item.parent == root or item.parent.parent == root]
    if import_images:
        image = path if path in import_images else import_images[0]
        actions.append(('convert', tr('Import microscopy images'),
                        tr('Review the reader, embedded metadata and conversion plan before importing.'), str(image.parent)))
    if containers:
        container = path if path in containers else containers[0]
        reader, install = _READERS.get(container.suffix.lower(), ('spacr.convert', ''))
        description = tr('Review the reader, embedded metadata and conversion plan before importing.')
        if find_spec(reader) is None:
            description = tr('This import needs {reader}. Install it with {command}, then review the import plan.',
                             reader=reader, command=install)
        actions.append(('convert', tr('Import microscopy images'),
                        description, str(container.parent)))
    for kind, key, title, description in (
            ('database', 'db_browser', tr('Browse measurements'), tr('Browse the detected SQLite database.')),
            ('table', 'foreign', tr('Import a measurement table'), tr('Review identities, columns and units.'))):
        if saved_settings and kind == 'table':
            continue
        own = next((candidate for candidate in kinds.get(kind, [])
                    if candidate.parent == root or candidate.parent.parent == root), None)
        if kind == 'database' and own is not None and path.is_dir():
            try:
                if _easy_database_evidence(own)['annotation_only']:
                    own = None
            except (OSError, sqlite3.Error):
                own = None
        if own is not None:
            actions.append((key, title, description, str(own)))
    analysis_discovery = {}
    analysis_databases = _easy_analysis_inputs(root, analysis_discovery) if path.is_dir() else []
    limited = limited or analysis_discovery.get('truncated', False)
    if path.is_file() and path in kinds.get('table', []):
        analysis_databases = []
    if path.is_file() and path in kinds.get('database', []):
        analysis_databases = [path]
    schemas = []
    for database in analysis_databases:
        try:
            evidence = _easy_database_evidence(database)
        except (OSError, sqlite3.Error):
            continue
        schemas.append(dict(path=str(database), **evidence))
    scientific = next((item for item in schemas if item['objects'] and item['numeric']), None)
    if scientific:
        database = Path(scientific['path'])
        plate = database.parent.parent if database.parent.name == 'measurements' else database.parent
        actions = [action for action in actions if action[0] != 'db_browser']
        for key, title in (('db_browser', tr('Database Browser')), ('graph_builder', tr('Graph Builder')),
                           ('gate_editor', tr('Gate Editor')), ('umap', tr('Image UMAP')),
                           ('regression', tr('Regression')), ('dose_response', tr('Dose Response')),
                           ('qc_dashboard', tr('QC')), ('run_compare', tr('Run Compare')),
                           ('annotate', tr('Annotate')), ('classify_merged', tr('Classify'))):
            if key in ('annotate', 'umap') and not scientific['crops']:
                continue
            source = plate if key in ('qc_dashboard', 'run_compare', 'annotate', 'classify_merged') else database
            actions.append((key, title, tr('Scientific measurements were found in this database.'), str(source)))
            input_map[key] = dict(database=str(database), source=str(source), project=str(plate))
    table_candidates = [] if saved_settings else list(kinds.get('table', []))
    if path.is_dir():
        for candidate in files:
            if (classified.get((candidate.parent, candidate.suffix.lower())) == 'table'
                    and candidate not in table_candidates):
                table_candidates.append(candidate)
        table_candidates.sort(key=lambda item: item.name != 'unique_combinations.csv')
    elif path in table_candidates:
        table_candidates = [path]
    for table in table_candidates[:8]:
        if table.parent != root and table.parent.parent != root:
            continue
        try:
            evidence = _easy_table_evidence(table)
        except (OSError, ValueError, UnicodeError):
            continue
        if evidence.get('numeric'):
            for key, title in (('graph_builder', tr('Graph Builder')), ('gate_editor', tr('Gate Editor'))):
                actions.append((key, title, tr('Numeric columns were found in this table.'), str(table)))
                input_map.setdefault(key, dict(source=str(table), table=str(table)))
        if evidence.get('sequencing'):
            actions.append(('regression', tr('Regression'), tr('Sequencing counts were found in this table.'), str(table)))
            input_map.setdefault('regression', dict(source=str(table), count_data=str(table)))
    fastq = next((item for item in kinds.get('fastq', []) if item.parent == root or item.parent.parent == root), None)
    if fastq is not None:
        actions.append(('map_barcodes', tr('Map Barcodes'), tr('FASTQ reads were found in this folder.'), str(fastq.parent)))
        input_map['map_barcodes'] = dict(source=str(fastq.parent), fastq=str(fastq))
    calibrated = os.path.lexists(root / 'plate_batch.json') or os.path.lexists(root / 'psf' / 'segmentation_application.json')
    paired_layout = _easy_paired_layout(images, root)
    if paired_layout:
        if paired_layout['kind'] == 'unstructured':
            actions = [action for action in actions if action[0] not in
                       ('mask', '3d', 'timelapse', 'measure', 'qc_dashboard', 'annotate')]
            dimension_routes = {}
        else:
            for key, title in (('measure', tr('Measure')), ('qc_dashboard', tr('QC')), ('annotate', tr('Annotate'))):
                actions.append((key, title, tr('Review merged arrays and mask channels.'), paired_layout['project']))
                input_map.setdefault(key, dict(source=paired_layout['project'], project=paired_layout['project']))
    run_evidence = _easy_run_evidence(root, [item for item in files
                                            if item.parent == root or item.parent.parent == root],
                                      include_runs=path.is_dir())
    if run_evidence['manifests']:
        for key, title in (('run_history', tr('Run History')), ('qc_dashboard', tr('QC'))):
            actions.append((key, title, tr('A failed or incomplete run was found; review its outputs before continuing.'), str(root)))
            input_map.setdefault(key, dict(source=str(root), run_manifests=run_evidence['manifests']))
    if run_evidence['results']:
        result_file = run_evidence['results'][0]
        for key, title in (('hit_list', tr('Hit List')), ('profiler', tr('Prediction Profiler'))):
            source = result_file['folder'] if key == 'hit_list' else result_file['path']
            actions.append((key, title, tr('Fitted regression coefficients were found in this folder.'), source))
            input_map[key] = dict(source=source, coefficients=result_file['path'], project=result_file['folder'])
        if result_file['diagnostics']:
            actions.append(('regression_diagnostics', tr('Regression diagnostics'),
                            tr('Fitted regression coefficients were found in this folder.'), result_file['diagnostics_folder']))
            input_map['regression_diagnostics'] = dict(source=result_file['diagnostics_folder'],
                                                     diagnostics_folder=result_file['diagnostics_folder'],
                                                     coefficients=result_file['path'])
    generated_outputs = _easy_generated_outputs(path, files, schemas, run_evidence, calibrated)
    if generated_outputs['artifacts']:
        if path.is_file() and path.suffix.lower() == '.npy':
            actions = [action for action in actions if action[0] not in ('mask', 'convert')]
            native_container = False
        if npy_image_source:
            generated_outputs['routes']['make_masks'] = dict(
                source=npy_image_source, project=generated_outputs['project'], prerequisite='',
                ready=True, title=tr('View and edit images'), reason=tr('Inspect images and edit masks.'))
        generated_actions = []
        for key, route in generated_outputs['routes'].items():
            generated_actions.append((key, route['title'], route['reason'], route['source']))
            if route['ready']:
                input_map.setdefault(key, {name: route[name] for name in ('source', 'project', 'database')
                                           if name in route})
            else:
                input_map.pop(key, None)
        actions = generated_actions + actions
    submodule_settings = _easy_specialized_inputs(root, files, images, schemas) if path.is_dir() else {}
    for key, title in (('ops', tr('OPS')), ('spatial_masks', tr('Spatial Mask Generation')),
                       ('spatial_map', tr('Spatial Map'))):
        if key in submodule_settings:
            hint = submodule_settings[key]
            actions.append((key, title, hint['evidence'], hint['source']))
            input_map[key] = dict(source=hint['source'], settings=hint['settings'])
    metadata = _easy_metadata_hints(folders)
    calibrated_message = (tr('This folder has a calibrated-workflow marker. Use a completed child dataset and the supported calibrated measurement workflow; ordinary Measure is not suggested.')
                          if path.is_dir() and calibrated else '')
    if actions:
        message = tr('Suggestions come from local file headers. Review each module before running; nothing has been processed or sent to AI.')
        if len(kinds.keys() - {'unsupported'}) > 1:
            message += ' ' + tr('The sampled source and nearby folders contain several content types; choose the next step you need.')
        if len(image_folders) > 1 or (images and not local_images):
            message += ' ' + tr('Images were found in nearby folders. Suggestions use {folder}; choose a different child folder to inspect another group.', folder=str(chosen))
        if hints:
            message += ' ' + tr('Some filenames match existing microscope conventions. These are naming hints, not verified acquisition metadata; confirm the convention before processing.')
        if metadata:
            message += ' ' + tr('Nearby files with metadata-style names were found. Review their contents before choosing channels or calibration.')
    elif external_documents:
        message = _easy_external_guidance(external_documents[0]['kind'])
    elif calibrated_message:
        message = calibrated_message
    elif kinds.get('filename_format'):
        message = tr('The contents look supported, but the filename extension is not accepted by the matching module. Keep the original and make a correctly named copy before continuing.')
    else:
        message = tr('No supported image, data table or SQLite database was found in the bounded sample. Choose an image folder, a delimited table or a spaCR database.')
    if actions and calibrated_message:
        message += ' ' + calibrated_message
    if limited:
        message += ' ' + tr('A bounded sample of this folder was inspected; other files may differ.')
    if unreadable:
        message += ' ' + tr('Some files could not be read. Check their permissions and try again.')
    if actions and external_documents:
        message += ' ' + ' '.join(_easy_external_guidance(kind)
                                 for kind in dict.fromkeys(item['kind'] for item in external_documents))
    if path.is_file() and not generated_outputs['artifacts']:
        preferred = [action for action in actions if action[3] == str(path)
                     or (path in containers and action[0] == 'convert' and action[3] == str(path.parent))]
        actions = preferred + [action for action in actions if action not in preferred]
    context = dict(sampled_files=len(files), sampled_directories=len(folders),
                   image_folders=[str(folder) for folder in image_folders], filename_matches=hints,
                   metadata_files=metadata, truncated=limited, header_files=len(representatives),
                   filename_sample_limit=sample_limit, sampled_image_filenames=len(names),
                   analysis_inputs=input_map, database_schemas=schemas,
                   analysis_discovery=analysis_discovery,
                   image_dimensions=image_dimensions,
                   dimension_routes=dimension_routes,
                   array_imports=array_imports,
                   plate_inputs=[dict(identity=item['path'], database=item['path'],
                                      project=str(Path(item['path']).parent.parent
                                                  if Path(item['path']).parent.name == 'measurements'
                                                  else Path(item['path']).parent))
                                 for item in schemas if item['objects'] and item['numeric']],
                   run_evidence=run_evidence,
                   generated_outputs=generated_outputs,
                   submodule_settings=submodule_settings,
                   unsupported_documents=external_documents,
                   reader_candidates=[dict(path=str(item), format=item.suffix.lower(),
                       reader=_READERS.get(item.suffix.lower(), ('spacr.convert', ''))[0],
                       install_hint=_READERS.get(item.suffix.lower(), ('', ''))[1],
                       available=find_spec(_READERS.get(item.suffix.lower(), ('spacr.convert', ''))[0]) is not None,
                       metadata_inspected=False) for item in kinds.get('container', [])[:4]])
    if native_container:
        context.update(native_mask_container=str(path), native_metadata_inspected=False)
    if paired_layout:
        context['paired_layout'] = paired_layout
    context['plate_selection_required'] = len(context['plate_inputs']) > 1
    context['all_compatible_databases'] = [item['database'] for item in context['plate_inputs']]
    unique = []
    seen_actions = set()
    for action in actions:
        identity = (action[0], action[3] if action[0] == 'mask' else '')
        if identity not in seen_actions:
            seen_actions.add(identity)
            unique.append(action)
    actions = unique
    groups = {}
    for key, title, reason, source in actions:
        if saved_settings and source == str(path) and key == saved_settings['module']:
            kind, group = 'settings', tr('Settings')
        elif key == 'map_barcodes' or input_map.get(key, {}).get('count_data'):
            kind, group = 'sequencing', tr('Sequencing')
        elif (input_map.get(key, {}).get('database') or input_map.get(key, {}).get('coefficients')
              or key in ('db_browser', 'measure', 'run_history', 'qc_dashboard')):
            kind, group = 'measurements', tr('Measurements')
        elif key in ('foreign', 'graph_builder', 'gate_editor'):
            kind, group = 'tables', tr('Tables')
        else:
            kind, group = 'images', tr('Images')
        groups[key] = dict(group=group, reason=reason, source=source, kind=kind)
    context['action_groups'] = groups
    if saved_settings:
        context['saved_settings'] = saved_settings
    result = {'path': str(path), 'message': message, 'actions': actions, 'context': context,
              'confidence': 'filename_hint' if hints else 'content_only'}
    proposal = _easy_regex_evidence(images, chosen, hints) if not native_container else None
    if proposal:
        result['regex'] = proposal
    elif images and not native_container:
        from ...regex_infer import _metadata_parse_report

        example = next((image for image in images if image.parent == chosen), None)
        if example is not None:
            extension = example.suffix.lstrip('.')
            own_names = [image.name for image in images if image.parent == chosen and image.suffix == example.suffix]
            preset = next((hint['convention'] for hint in hints
                           if hint['folder'] == str(chosen)
                           and _metadata_parse_report(own_names, hint['convention'], extension)[:2]
                           == (len(own_names), len(own_names))), None)
            if preset:
                result['metadata_preset'] = preset
                result['settings_overrides'] = {'mask': dict(metadata_type=preset)}
    return result


def _easy_inspect_path(value):
    """Convert expected local access errors into guidance without requesting AI reporting."""
    from ..i18n import tr

    try:
        return _easy_inspect_path_impl(value)
    except (OSError, ValueError, EOFError):
        return {'path': value, 'message': tr('This input could not be inspected. Check permissions and choose another path.'), 'actions': []}

class _EasyDropWatcher(QObject):
    """Receive only local drag events belonging to the owned Easy mode window."""

    def __init__(self, window):
        """Register the three drag kinds without adding a Python paint filter."""
        from ..gil_priority import _watch_application_events

        super().__init__(window)
        self._window = window
        window.setAcceptDrops(True)
        _watch_application_events(QApplication.instance(), self,
                                  (QEvent.DragEnter, QEvent.DragMove, QEvent.Drop))

    def eventFilter(self, watched, event):
        """Route whole-window local drops while preserving every ordinary-mode event."""
        from ..preferences import get_easy_mode

        if event.type() not in (QEvent.DragEnter, QEvent.DragMove, QEvent.Drop):
            return False
        if not get_easy_mode() or not isinstance(watched, QWidget) or watched.window() is not self._window:
            return False
        stack = getattr(self._window, '_stack', None)
        home = getattr(self._window, '_startup', None)
        if stack is None or stack.currentWidget() is not home or not isinstance(home, _EasyHomePage):
            return False
        mime = event.mimeData()
        if not mime.hasUrls():
            return False
        if event.type() == QEvent.Drop:
            paths = [url.toLocalFile() for url in mime.urls() if url.isLocalFile()]
            self._window._easy_mode_drop(paths if len(paths) == len(mime.urls()) else [])
        event.acceptProposedAction()
        return True


class _EasyPathRim(QWidget):
    """Keep a static outline above the native search surface."""

    def __init__(self, parent=None):
        """Draw a passive rounded border without a timer or field callbacks."""
        super().__init__(parent)
        self._radius = 28
        self.setObjectName('EasyPathRim')
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        make_transparent(self)

    def paintEvent(self, event):
        """Paint the fixed border independently of pointer and idle time."""
        from PySide6.QtGui import QPen

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(palette_for('dark')['border']), 1))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1),
                                self._radius, self._radius)
        painter.end()


class _NavigatorAiReply(QObject):
    """Keep generation-tagged stream callbacks on their owning GUI thread."""

    def __init__(self, page, generation):
        """Retain the page and request token until the stream is retired."""
        super().__init__(page)
        self._page = page
        self._generation = generation

    @Slot(str)
    def chunk(self, text):
        """Deliver a worker chunk through an actual GUI-affine Qt receiver."""
        self._page._navigator_ai_chunk(self._generation, text)

    @Slot(bool, str)
    def finished(self, ok, text):
        """Complete only the request this receiver was created for."""
        self._page._navigator_ai_finished(self._generation, ok, text)


class _NavigatorAiButton(QPushButton):
    """A quiet provider-colored toggle with bounded typing and waiting motion."""

    _COLORS = {'spacr_hosted': '#FFFFFF', 'claude': '#DE7356',
               'codex': '#A8E6CF', 'gemini': '#4285F4'}

    def __init__(self, parent=None):
        """Keep a matte provider fill inside a native, keyboard-accessible toggle."""
        from PySide6.QtCore import QElapsedTimer

        from ..i18n import tr

        super().__init__(parent)
        self.setObjectName('NavigatorAiButton')
        self.setAccessibleName(tr('Navigator AI'))
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(36, 36)
        self._provider = 'spacr_hosted'
        self._busy = False
        self._marker_busy = False
        self._marker_phase = None
        self._marker_started = 0
        self._marker_origin = (0.0, 0.0)
        self._pulse = 0.0
        self._last_key = -1000
        self._clock = QElapsedTimer()
        self._clock.start()
        self._motion = QTimer(self)
        self._motion.setInterval(33)
        self._motion.timeout.connect(self._tick)
        self.toggled.connect(self._tick)

    def _typing(self, _text):
        """Give rapid native edits a larger pulse without ever submitting text."""
        now = self._clock.elapsed()
        speed = max(0.0, 1.0 - (now - self._last_key) / 500.0)
        self._last_key = now
        self._pulse = 0.35 + 0.65 * speed
        self._tick()

    def _hosted_marker(self, now, reduced=False):
        """Describe the marker in fill-radius units with a whiteward return tint."""
        import math

        if reduced:
            return (0.5, 0.0, 0) if self._busy else None
        elapsed = max(0, now - self._marker_started)
        if self._marker_phase == 'enter':
            fraction = min(1.0, elapsed / 250.0)
            distance = 0.5 * fraction * fraction * (3.0 - 2.0 * fraction)
            return distance, 0.0, 0
        if self._marker_phase == 'orbit':
            angle = elapsed * math.tau / 1800.0
            return 0.5 * math.cos(angle), 0.5 * math.sin(angle), 0
        if self._marker_phase == 'return' and elapsed < 300:
            fraction = elapsed / 300.0
            eased = fraction * fraction * (3.0 - 2.0 * fraction)
            return (self._marker_origin[0] * (1.0 - eased),
                    self._marker_origin[1] * (1.0 - eased), round(255 * eased))
        return None

    def _tick(self, *_args):
        """Run the existing timer only for visible typing, work or marker return."""
        from ..preferences import get_reduce_motion

        now = self._clock.elapsed()
        reduced = get_reduce_motion()
        hosted = self.isChecked() and self._provider == 'spacr_hosted'
        busy = hosted and self._busy
        if busy and not self._marker_busy:
            self._marker_phase = 'enter'
            self._marker_started = now
        elif not busy and self._marker_busy:
            marker = self._hosted_marker(now, reduced)
            self._marker_origin = marker[:2] if marker is not None else (0.0, 0.0)
            self._marker_phase = 'return'
            self._marker_started = now
        self._marker_busy = busy
        if not hosted or (reduced and not busy):
            self._marker_phase = None
        elif self._marker_phase == 'enter' and now - self._marker_started >= 250:
            self._marker_phase = 'orbit'
            self._marker_started += 250
        elif self._marker_phase == 'return' and now - self._marker_started >= 300:
            self._marker_phase = None
        self.update()
        if (self.isVisible() and self.isChecked() and not reduced
                and (self._busy or self._marker_phase == 'return'
                     or now - self._last_key < 850)):
            if not self._motion.isActive():
                self._motion.start()
        else:
            self._motion.stop()

    def showEvent(self, event):
        """Resume pending motion only when the icon becomes visible."""
        super().showEvent(event)
        self._tick()

    def hideEvent(self, event):
        """Release animation work while another module is on screen."""
        self._motion.stop()
        super().hideEvent(event)

    def paintEvent(self, event):
        """Keep the provider fill matte while a subtle busy glow circles outside."""
        import math

        from PySide6.QtGui import QPen

        from ..preferences import get_reduce_motion

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.translate(self.rect().center())
        color = QColor(self._COLORS.get(self._provider, self._COLORS['spacr_hosted']))
        if (self.isChecked() and self._busy and self._provider != 'spacr_hosted'
                and not get_reduce_motion()):
            painter.save()
            painter.rotate((self._clock.elapsed() * 0.18) % 360)
            painter.setBrush(Qt.NoBrush)
            for width, alpha in ((4.0, 18), (2.5, 35), (1.0, 90)):
                glow = QColor(color)
                glow.setAlpha(alpha)
                painter.setPen(QPen(glow, width, Qt.SolidLine, Qt.RoundCap))
                painter.drawArc(QRectF(-12, -12, 24, 24), 20 * 16, 100 * 16)
            painter.restore()
        painter.setPen(QPen(QColor('white'), 1.5 if not self.hasFocus() else 2.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QRectF(-14, -14, 28, 28))
        if self.isChecked():
            elapsed = max(0, self._clock.elapsed() - self._last_key)
            pulse = 0.0 if get_reduce_motion() else self._pulse * math.exp(-elapsed / 220.0)
            radius = (8.0 + 5.0 * pulse) * (1.0125 if self._provider == 'spacr_hosted' else 1.0)
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(QRectF(-radius, -radius, radius * 2, radius * 2))
            marker = self._hosted_marker(self._clock.elapsed(), get_reduce_motion())
            if self._provider == 'spacr_hosted' and marker is not None:
                x, y, shade = marker
                size = radius * 0.2
                painter.setBrush(QColor(shade, shade, shade))
                painter.drawEllipse(QRectF(x * radius - size, y * radius - size,
                                          size * 2, size * 2))
        painter.end()


class _EasyPathEdit(QLineEdit):
    """One rounded path field with an inset text-only native Browse control."""

    def __init__(self, parent=None):
        """Keep the browse affordance inside the editor's reserved text margin."""
        from ..i18n import tr
        from .field_fade import OPT_OUT_PROPERTY

        super().__init__(parent)
        self.setObjectName('EasyModePath')
        self.setProperty(OPT_OUT_PROPERTY, True)
        self.setFrame(False)
        self.setPlaceholderText(tr('Drop a file or folder | Type a Path | Ask a Question ◉'))
        self.setAccessibleName(tr('File or folder path'))
        self.setFixedHeight(64)
        self.setMouseTracking(True)
        self.browse = QPushButton(tr('Browse'), self)
        self.browse.setObjectName('EasyModeBrowse')
        self.browse.setCursor(Qt.PointingHandCursor)
        self.ai = _NavigatorAiButton(self)
        self.textEdited.connect(self.ai._typing)

    def resizeEvent(self, event):
        """Reserve native editor space for Browse without covering editable text."""
        super().resizeEvent(event)
        if not hasattr(self, 'browse'):
            return
        width = self.browse.fontMetrics().horizontalAdvance(self.browse.text()) + 24
        self.browse.setGeometry(self.width() - width - 60, 6, width, self.height() - 12)
        self.ai.move(self.width() - 48, (self.height() - self.ai.height()) // 2)
        self.setTextMargins(18, 0, width + 68, 0)



_EASY_WORKFLOW_PORT_RADIUS = 4


class _EasySuggestionCard(QPushButton):
    """A growing local next-step card with an explicit module handoff."""

    activated = Signal(str, str)

    def __init__(self, action, parent=None):
        """Make the whole module card a native keyboard-accessible action."""
        from ..app import APPS, _icon_for_app
        from ..i18n import tr
        from PySide6.QtWidgets import QGraphicsOpacityEffect, QLayout

        super().__init__(parent)
        self._key, title, description, self._path = action
        self._spaceout_movie = None
        if self._key == '__easy_command__:Spaceout':
            from PySide6.QtGui import QMovie

            from ..iconset import _spaceout_gif_path

            self._spaceout_movie = QMovie(str(_spaceout_gif_path()), parent=self)
            self._spaceout_movie.setCacheMode(QMovie.CacheNone)
            self._spaceout_movie.frameChanged.connect(self._spaceout_frame)
        self._growing = False
        self._final_rect = QRect()
        self.setObjectName('EasySuggestionCard')
        self.setProperty('spaceout', self._key == '__easy_command__:Spaceout')
        self._stage = self._stage_for(self._key)
        self.setProperty('stage', self._stage)
        self.setCursor(Qt.PointingHandCursor)
        self.setAccessibleName(title)
        self.clicked.connect(self._activate)
        outer = QVBoxLayout(self)
        outer.setSizeConstraint(QLayout.SetNoConstraint)
        outer.setContentsMargins(0, 0, 0, 0)
        self.content = QWidget(self)
        self.content.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        outer.addWidget(self.content)
        self._content_opacity = QGraphicsOpacityEffect(self)
        self._content_opacity.setOpacity(1.0)
        self.setGraphicsEffect(self._content_opacity)
        layout = QVBoxLayout(self.content)
        layout.setSizeConstraint(QLayout.SetNoConstraint)
        heading = QVBoxLayout()
        heading.setAlignment(Qt.AlignCenter)
        icon = QLabel(self)
        self.icon = icon
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet('background: transparent;')
        module_icon = _icon_for_app(self._key, theme="dark")
        if module_icon is None or module_icon.isNull():
            module_icon = self.style().standardIcon(QStyle.SP_FileDialogContentsView)
        self._module_icon = module_icon
        self._refresh_icon()
        follow_device_ratio(icon, self._refresh_icon)
        icon.setAccessibleName(title)
        heading.addWidget(icon)
        module_name = next((row[1] for row in APPS if row[0] == self._key), title)
        name = QLabel(tr(module_name), self)
        name.setWordWrap(True)
        name.setAlignment(Qt.AlignCenter)
        name.setStyleSheet(f'font-size: {font_px(18)}px; font-weight: 300;')
        self.title = name
        heading.addWidget(name)
        layout.addLayout(heading)
        label = QLabel(description, self)
        self.description = label
        label.setWordWrap(True)
        label.setVisible(self._key in ("regex", "warning"))
        layout.addWidget(label, 1)
        self._explanation = description
        self.setAccessibleDescription(description)
        from ..theme import STAGE_HOVER
        from ..theme import ALPHA_MARK
        self.alpha = QLabel(ALPHA_MARK, self)
        self.alpha.setObjectName('EasyAlphaMark')
        self.alpha.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.alpha.setStyleSheet('font-family: "Open Sans"; font-weight: 300; '
                                f'font-size: {font_px(28)}px; color: {STAGE_HOVER["alpha"]}; '
                                'background: transparent;')
        self.alpha.setAlignment(Qt.AlignCenter)
        self.alpha.resize(40, 40)
        self.alpha.setVisible(self._stage == 'alpha')
        if self._spaceout_movie is not None:
            self.content.hide()
            self.alpha.hide()

    def enterEvent(self, event):
        """Keep card explanations in the stationary lower description panel."""
        parent = self.parentWidget()
        if hasattr(parent, 'explain_card'):
            parent.explain_card(self)
        super().enterEvent(event)

    def focusInEvent(self, event):
        """Expose the same explanation to keyboard navigation."""
        parent = self.parentWidget()
        if hasattr(parent, 'explain_card'):
            parent.explain_card(self)
        super().focusInEvent(event)

    @staticmethod
    def _stage_for(key):
        """Resolve maturity from the registered app or its actual folded metadata."""
        from .fold_strip import fold_label
        if key == 'parascape':
            return 'alpha'
        return fold_label(key)[2]

    def resizeEvent(self, event):
        """Keep the alpha annotation at the rounded card's upper-right corner."""
        super().resizeEvent(event)
        if getattr(self, '_spaceout_movie', None) is not None:
            self._resize_spaceout()
        if hasattr(self, 'alpha'):
            self.alpha.move(max(0, self.width() - self.alpha.width() - 8), 8)
            self.alpha.raise_()

    def paintEvent(self, event):
        """Keep native card painting and add equal output ports only for its network."""
        super().paintEvent(event)
        picture = getattr(self, '_spaceout_display', None)
        if picture is not None and not picture.isNull():
            from PySide6.QtGui import QPainterPath

            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            target = QRectF(self.rect()).adjusted(1, 1, -1, -1)
            clip = QPainterPath()
            clip.addRoundedRect(target, 15, 15)
            painter.setClipPath(clip)
            source = QRectF(picture.rect())
            ratio = target.width() / max(1, target.height())
            if source.width() / max(1, source.height()) > ratio:
                wanted = source.height() * ratio
                source.setLeft((source.width() - wanted) / 2)
                source.setWidth(wanted)
            else:
                wanted = source.width() / ratio
                source.setTop((source.height() - wanted) / 2)
                source.setHeight(wanted)
            painter.drawPixmap(target, picture, source)
            if self.hasFocus():
                from PySide6.QtWidgets import QStyle, QStyleOptionFocusRect

                painter.setClipping(False)
                focus = QStyleOptionFocusRect()
                focus.initFrom(self)
                focus.rect = self.rect().adjusted(3, 3, -3, -3)
                self.style().drawPrimitive(QStyle.PE_FrameFocusRect, focus, painter, self)
            painter.end()
        points = getattr(self, '_workflow_ports', ())
        if points:
            color = QColor(palette_for('dark')['accent'])
            color.setAlpha(round(255 * getattr(self, '_port_opacity', 1.0)))
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            for point in points:
                painter.drawEllipse(point, _EASY_WORKFLOW_PORT_RADIUS, _EASY_WORKFLOW_PORT_RADIUS)
            painter.end()

    def _refresh_icon(self):
        """Rasterize the original module icon for its actual display density."""
        pixels = max(1, round(96 * device_ratio(self.icon)))
        picture = self._module_icon.pixmap(pixels, pixels)
        picture.setDevicePixelRatio(device_ratio(self.icon))
        self.icon.setPixmap(picture)
        if self._spaceout_movie is not None:
            self._resize_spaceout()
            self._sync_run_state()

    def _resize_spaceout(self):
        """Decode at bounded card resolution only when its size or density changes."""
        pixels = max(1, min(500, round(max(self.width(), self.height()) * device_ratio(self))))
        size = QSize(pixels, pixels)
        if self._spaceout_movie.scaledSize() == size:
            return
        self._spaceout_movie.setScaledSize(size)
        self._spaceout_picture = self._module_icon.pixmap(size)
        self._spaceout_picture.setDevicePixelRatio(device_ratio(self))
        self._spaceout_display = self._spaceout_picture
        self.update()

    def _sync_run_state(self):
        """Play the provided animation only while visible and motion is allowed."""
        from ..preferences import get_reduce_motion

        if self._spaceout_movie is None:
            return
        if self.isVisible() and not get_reduce_motion():
            self._spaceout_movie.start()
        else:
            self._spaceout_movie.stop()
            self._spaceout_display = self._spaceout_picture
            self.update()

    def _spaceout_frame(self, _frame):
        """Show one native decoded frame at the icon's current display density."""
        picture = self._spaceout_movie.currentPixmap()
        picture.setDevicePixelRatio(device_ratio(self))
        self._spaceout_display = picture
        self.update()

    def showEvent(self, event):
        """Resume the animation when the Spaceout card reappears."""
        super().showEvent(event)
        self._sync_run_state()

    def hideEvent(self, event):
        """Stop decoding the animation while its card is hidden."""
        if self._spaceout_movie is not None:
            self._spaceout_movie.stop()
        super().hideEvent(event)

    def keyPressEvent(self, event):
        """Keep the image-only command explicitly keyboard-activatable."""
        if self._spaceout_movie is not None and event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.click()
            event.accept()
        else:
            super().keyPressEvent(event)

    def _activate(self):
        """Emit the inspected input only after the user chooses this step."""
        self.activated.emit(self._key, self._path)

    def _growth_changed(self, value):
        """Use the actual expanding widget rectangle as the field footprint."""
        fraction = min(self.width() / max(1, self._final_rect.width()),
                       self.height() / max(1, self._final_rect.height()))
        self._content_opacity.setOpacity(max(0.0, min(1.0, fraction)))
        self.parentWidget()._publish_growth()

    def _growth_finished(self):
        """Release the parted field without an automatic card ripple."""
        self._growing = False
        self._content_opacity.setOpacity(1.0)
        self.setEnabled(True)
        self.setGeometry(self._final_rect)
        self.raise_()
        self.parentWidget()._publish_growth()


class _EasyRegexHighlight(_SyntaxHighlighter):
    """Color compiled named-group headers without interpreting another regex grammar."""

    def __init__(self, document):
        """Own only presentation formats on the native editable document."""
        self.colors = {}
        super().__init__(document)

    def highlightBlock(self, text):
        """Match the displayed group names to the example's capture colors."""
        from PySide6.QtGui import QTextCharFormat

        for name, color in self.colors.items():
            token = '(?P<' + name + '>'
            start = text.find(token)
            while start >= 0:
                previous = start - 1
                while previous >= 0 and text[previous] == '\\':
                    previous -= 1
                if (start - previous - 1) % 2 == 0:
                    format = QTextCharFormat()
                    format.setForeground(QColor(color))
                    self.setFormat(len(text[:start].encode('utf-16-le')) // 2,
                                   len(token.encode('utf-16-le')) // 2, format)
                start = text.find(token, start + len(token))


class _EasyRegexCard(_EasySuggestionCard):
    """Offer an editable filename pattern before applying any metadata settings."""

    regex_requested = Signal(str)

    def __init__(self, proposal, parent=None):
        """Show native editable regex text and a capture-colored example path."""
        from ..iconset import icon
        from ..i18n import tr

        self._proposal = dict(proposal)
        super().__init__(('regex', tr('Regex'), '', proposal['example_path']), parent)
        self._module_icon = icon('regex', size=96, theme='dark')
        self._refresh_icon()
        self.setAccessibleName(tr('Apply regex'))
        self.setToolTip(tr('Edit the pattern, then click this card to apply it.'))
        self.content.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.pattern_edit = QPlainTextEdit(self.content)
        self.pattern_edit.setObjectName('EasyRegexPattern')
        self.pattern_edit.setAccessibleName(tr('Filename regex'))
        self.pattern_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.pattern_edit.setFixedHeight(76)
        self.content.layout().insertWidget(1, self.pattern_edit)
        self._highlight = _EasyRegexHighlight(self.pattern_edit.document())
        self._preview_pattern = None
        self.description.setAccessibleName(tr('Example image path'))
        self.error = QLabel(self.content)
        self.error.setWordWrap(True)
        self.content.layout().addWidget(self.error)
        self.pattern_edit.textChanged.connect(self._refresh_preview)
        self.pattern_edit.setPlainText(proposal.get('pattern', ''))
        self._refresh_preview()

    def _refresh_preview(self):
        """Color corresponding compiled captures without changing stored settings."""
        from ..i18n import tr
        from ..regex_detect import KNOWN_FIELDS

        path = self._proposal['example_path']
        pattern = self.pattern_edit.toPlainText()
        if pattern == self._preview_pattern:
            return
        self._preview_pattern = pattern
        self._highlight.colors = {}
        parts = []
        try:
            compiled = re.compile(pattern) if len(pattern) <= 4096 else None
            for index, group in enumerate(compiled.groupindex if compiled is not None else ()):
                position = KNOWN_FIELDS.index(group) if group in KNOWN_FIELDS else len(KNOWN_FIELDS) + index
                color = QColor.fromHsvF((0.56 + position * 0.135) % 1.0, 0.55, 1.0).name()
                self._highlight.colors[group] = color
            if pattern == self._proposal.get('pattern'):
                for group in self._proposal.get('groups', ()):
                    color = self._highlight.colors.get(group['name'])
                    if color and 0 <= group['start'] <= group['end'] <= len(path):
                        parts.append((group['start'], group['end'], color))
            self.error.setText(self._proposal.get('validation_error', '')
                               if self.pattern_edit.toPlainText() == self._proposal.get('pattern') else '')
        except (re.error, RecursionError, OverflowError) as error:
            self.error.setText(tr('Invalid regex: {error}', error=str(error)))
        rendered = []
        end = 0
        for start, stop, color in sorted(parts):
            if start < end:
                continue
            rendered.append(escape(path[end:start]))
            rendered.append('<span style="color:' + color + '">' + escape(path[start:stop]) + '</span>')
            end = stop
        rendered.append(escape(path[end:]))
        self.description.setText(''.join(rendered))
        self._highlight.rehighlight()

    def _activate(self):
        """Validate only when the user explicitly chooses this regex card."""
        if not self._growing:
            self.regex_requested.emit(self.pattern_edit.toPlainText())


class _EasyGuidanceCard(_EasySuggestionCard):
    """Keep a failed input's explanation visible without prose under the search."""

    def __init__(self, message, parent=None):
        """Present the real local failure and offer focus back to the editor."""
        from ..i18n import tr

        super().__init__(('warning', tr('Input guidance'), message, ''), parent)

    def _activate(self):
        """Return to the search field without opening a module."""
        self.parentWidget().panel.setFocus()


class _NavigatorDetailsBrowser(QTextBrowser):
    """Keep bounded description history and controls on the shared native surface."""

    def __init__(self, field):
        """Build bounded history and navigation controls owned by the Navigator field.

        :param field: owning field that arranges the shared description surface.
        """
        from ..i18n import tr
        from ..preferences import _set_scaled_icon_size

        super().__init__(field)
        self.setOpenExternalLinks(False)
        self.setOpenLinks(False)
        self.anchorClicked.connect(self._open_link)
        self._field = field
        self._entries = []
        self._entry = -1
        self._stream_token = None
        self._stream_entry = -1
        self._minimized = False
        self.setViewportMargins(8, 36, 8, 8)
        self._back = QToolButton(self)
        self._forward = QToolButton(self)
        self._collapse = QToolButton(self)
        for button, name, direction in (
                (self._back, tr('Back'), 'back'),
                (self._forward, tr('Next'), 'forward'),
                (self._collapse, tr('Collapse or expand this container'), 'down')):
            button.setAutoRaise(True)
            button.setAccessibleName(name)
            button.setToolTip(name)
            button.setIcon(self._arrow_icon(direction))
            _set_scaled_icon_size(button, 20)
            button.setFixedSize(28, 28)
            button.setCursor(Qt.PointingHandCursor)
        self._back.clicked.connect(lambda: self._navigate(-1))
        self._forward.clicked.connect(lambda: self._navigate(1))
        self._collapse.clicked.connect(self._toggle_minimized)
        self._update_controls()

    def _open_link(self, url):
        """Navigate an explicitly clicked assistant or documentation link."""
        from .ai_chat_panel import _activate_ai_link

        _activate_ai_link(self, url)

    def _arrow_icon(self, direction, color='white'):
        """Draw the requested arrow with normal and disabled device-scale variants.

        :param direction: back, forward, up or down arrow.
        :param color: arrow stroke color.
        :returns: the native icon containing both button-state variants.
        """
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QPen, QPolygonF

        points = {
            'back': ((13, 4), (7, 10), (13, 16)),
            'forward': ((7, 4), (13, 10), (7, 16)),
            'up': ((4, 13), (10, 7), (16, 13)),
            'down': ((4, 7), (10, 13), (16, 7)),
        }
        icon = QIcon()
        ratio = self.devicePixelRatioF()
        for mode, alpha in ((QIcon.Normal, 255), (QIcon.Disabled, 80)):
            image = QPixmap(round(20 * ratio), round(20 * ratio))
            image.setDevicePixelRatio(ratio)
            image.fill(Qt.transparent)
            painter = QPainter(image)
            painter.setRenderHint(QPainter.Antialiasing, True)
            ink = QColor(color)
            ink.setAlpha(alpha)
            painter.setPen(QPen(ink, 1.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPolyline(QPolygonF([QPointF(x, y) for x, y in points[direction]]))
            painter.end()
            icon.addPixmap(image, mode)
        return icon

    def _save_scroll(self):
        """Remember the visible vertical scroll position for the current history entry."""
        if 0 <= self._entry < len(self._entries):
            self._entries[self._entry][2] = self.verticalScrollBar().value()

    def _display_entry(self):
        """Replay the selected HTML or plain entry and restore its scroll position."""
        kind, text, scroll = self._entries[self._entry]
        if kind == 'html':
            super().setHtml(text)
        else:
            super().setPlainText(text)
        self.verticalScrollBar().setValue(scroll)

    def _record(self, kind, text, stream=None):
        """Record bounded content history or update the active streamed entry.

        :param kind: HTML or plain content representation.
        :param text: content to display; blank text clears only the visible document.
        :param stream: optional reply identity that updates its existing history entry.
        """
        if not text.strip():
            super().clear()
            self._update_controls()
            return
        self._save_scroll()
        if stream is not None and stream == self._stream_token and self._stream_entry >= 0:
            index = self._stream_entry
            self._entries[index][:2] = [kind, text]
            if self._entry == index:
                self._display_entry()
        elif (self._entry >= 0 and self._entries[self._entry][:2] == [kind, text]):
            self._display_entry()
        else:
            del self._entries[self._entry + 1:]
            if self._stream_entry > self._entry:
                self._stream_entry = -1
            self._entries.append([kind, text, 0])
            self._entry = len(self._entries) - 1
            if stream is not None:
                self._stream_token, self._stream_entry = stream, self._entry
            while len(self._entries) > 40 or (len(self._entries) > 1 and
                    sum(len(entry[1]) for entry in self._entries) > 2_000_000):
                self._entries.pop(0)
                self._entry -= 1
                self._stream_entry -= 1
            self._display_entry()
        self._update_controls()

    def setHtml(self, text, *, stream=None):
        """Display HTML through the bounded history, optionally updating one reply.

        :param text: HTML content converted to a string for storage.
        :param stream: optional active reply identity for replacing its history entry.
        """
        self._record('html', str(text), stream)

    def setPlainText(self, text):
        """Display plain text through the same bounded back-and-forward history.

        :param text: plain content converted to a string for storage.
        """
        self._record('plain', str(text))

    def _start_reply(self, token):
        """Associate streaming updates with the currently selected history entry.

        :param token: identity supplied by the new asynchronous reply.
        """
        self._stream_token, self._stream_entry = token, self._entry

    def _navigate(self, direction):
        """Move within available history while preserving each entry's scroll offset.

        :param direction: signed relative history offset.
        """
        target = self._entry + direction
        if 0 <= target < len(self._entries):
            self._save_scroll()
            self._entry = target
            self._display_entry()
            self._update_controls()

    def _update_controls(self):
        """Synchronize history actions, collapse appearance and viewport scroll policies."""
        self._back.setEnabled(self._entry > 0)
        self._forward.setEnabled(self._entry + 1 < len(self._entries))
        self._back.setVisible(not self._minimized)
        self._forward.setVisible(not self._minimized)
        self._collapse.setIcon(self._arrow_icon(
            'up' if self._minimized else 'down',
            'white' if self._minimized else palette_for('dark')['button_accent']))
        policy = Qt.ScrollBarAlwaysOff if self._minimized else Qt.ScrollBarAsNeeded
        self.setVerticalScrollBarPolicy(policy)
        self.setHorizontalScrollBarPolicy(policy)
        self.viewport().setVisible(not self._minimized)
        self._place_controls()

    def _place_controls(self):
        """Place navigation controls on the shared surface and clear stale clipping."""
        for offset, button in enumerate((self._collapse, self._forward, self._back)):
            button.move(max(0, self.width() - 34 - offset * 30), 4)
            button.raise_()
        self.clearMask()

    def _toggle_minimized(self):
        """Toggle the description viewport and ask its owning field to arrange it."""
        self._minimized = not self._minimized
        self._update_controls()
        self._field._arrange()

    def _reset_history(self):
        """Clear reply history, restore the expanded viewport and arrange its field."""
        self._entries.clear()
        self._entry = self._stream_entry = -1
        self._stream_token = None
        self._minimized = False
        super().clear()
        self._update_controls()
        self._field._arrange()

    def resizeEvent(self, event):
        """Forward native resizing before repositioning the history controls.

        :param event: Qt resize event received by this browser.
        """
        super().resizeEvent(event)
        self._place_controls()


class _NavigatorDetailsGrip(HeightGrip):
    """Resize the fixed-bottom description from its upper edge."""

    def __init__(self, field):
        """Carry the handle with the description when the workflow reparents it."""
        from ..i18n import tr

        self._field = field
        super().__init__(field.details, 96, 4096, parent=field.details,
                         name=tr('Resize this panel'))
        self.setObjectName('NavigatorDetailsHeightGrip')
        self.height_changed.connect(self._persist)

    def resize_target(self, height):
        """Clamp the requested height without imposing a fixed widget size."""
        from ..preferences import get_font_scale

        wanted = max(self._min, min(self._max, int(height)))
        self._field._details_height_base = wanted / max(0.01, get_font_scale())
        self._field._arrange()
        return self._target.height()

    def _persist(self, _height):
        """Remember an explicit resize in font-independent pixels."""
        from ..preferences import _settings

        _settings().setValue('home/navigator_details_height', self._field._details_height_base)

    def reset(self):
        """Restore the responsive default through the existing reset menu."""
        from ..preferences import _settings

        self._field._details_height_base = None
        _settings().remove('home/navigator_details_height')
        self._field._arrange()
        return self._target.height()

    def mouseMoveEvent(self, event):
        """Dragging upward grows the box while its lower edge stays fixed."""
        if self._from_y is None:
            return super().mouseMoveEvent(event)
        self.resize_target(self._from_h + int(self._from_y - event.globalPosition().y()))
        event.accept()

    def mouseReleaseEvent(self, event):
        """Persist the drag without adding an ambient-field gesture."""
        if self._from_y is None:
            return super().mouseReleaseEvent(event)
        self._from_y = None
        self.height_changed.emit(self._target.height())
        event.accept()

    def keyPressEvent(self, event):
        """Up grows and Down shrinks this upper-edge handle."""
        from ..preferences import scaled_px

        moves = {Qt.Key_Up: 1, Qt.Key_Down: -1,
                 Qt.Key_PageUp: 5, Qt.Key_PageDown: -5}
        if event.key() in moves:
            self.nudge(scaled_px(self.STEP) * moves[event.key()])
            event.accept()
        else:
            super().keyPressEvent(event)


class _EasyHomeField(QWidget):
    """Place a single centered editor over the window's unchanged ambient field."""

    def __init__(self, parent=None):
        """Own the path, feedback and suggestion widgets without another frame timer."""
        super().__init__(parent)
        self.cards = []
        self._suggestions_active = True
        self._animations = []
        self.feedback = None
        self.setObjectName('EasyHomeField')
        self.setProperty('spacrAmbientPassiveOverlay', True)
        make_transparent(self)
        self.panel = _EasyPathEdit(self)
        self.rim = _EasyPathRim(self)
        self.panel.setProperty('spacrAmbientPassiveOverlay', True)
        self.rim.setProperty('spacrAmbientPassiveOverlay', True)
        self.setMinimumSize(340, 340)
        from PySide6.QtWidgets import QGraphicsOpacityEffect, QScrollBar
        from ..i18n import tr
        from .field_fade import OPT_OUT_PROPERTY
        self.details = _NavigatorDetailsBrowser(self)
        self.details.setObjectName('EasyWorkflowDetails')
        self.details.setProperty('spacrAmbientPassiveOverlay', True)
        self.details.setAccessibleName(tr('Module description'))
        self.details.setOpenExternalLinks(False)
        self.details.setOpenLinks(False)
        self.details.setProperty(OPT_OUT_PROPERTY, True)
        from ..preferences import _settings

        try:
            stored = float(_settings().value('home/navigator_details_height', 0))
            self._details_height_base = stored if 0 < stored < 100000 else None
        except (TypeError, ValueError):
            self._details_height_base = None
        self._details_grip = _NavigatorDetailsGrip(self)
        self._scrollbar = QScrollBar(Qt.Horizontal, self)
        self._scrollbar.setObjectName('EasySuggestionsScroll')
        self._scrollbar.setAccessibleName(tr('Tools'))
        self._scrollbar._content_opacity = QGraphicsOpacityEffect(self._scrollbar)
        self._scrollbar.setGraphicsEffect(self._scrollbar._content_opacity)
        self._scrollbar.setProperty(OPT_OUT_PROPERTY, True)
        self._scrollbar.valueChanged.connect(self._scroll_cards)
        self._row_rect = QRect()

    def _motion_enabled(self):
        """Respect live backdrop and Reduce motion preferences for card growth."""
        from ..preferences import get_ambient_enabled, get_reduce_motion

        return get_ambient_enabled() and not get_reduce_motion()

    def _publish_growth(self, *, invalidate=False):
        """Track card animation locally without altering the shared field."""
        state = tuple((id(card), card._growing, card.isVisible())
                      for card in self.cards if card.parentWidget() is self)
        changed = state != getattr(self, '_growth_state', ())
        self._growth_state = state
        if changed or (invalidate and state):
            import sys

            ambient = sys.modules.get(f'{__package__}.ambient')
            if ambient is not None:
                ambient._field_growing_regions_for_widget(self, invalidate=True)

    def _finish_growth(self):
        """Retire geometry animations and release their field ownership."""
        for animation in self._animations:
            animation.stop()
            animation.deleteLater()
        self._animations.clear()
        for card in self.cards:
            if card.parentWidget() is not self or not self._suggestions_active:
                continue
            card._growing = False
            card._content_opacity.setOpacity(1.0)
            card.setEnabled(True)
            card.setGeometry(card._final_rect)
        self._publish_growth()

    def _sync_run_state(self):
        """Apply live motion changes without another timer or settings polling."""
        if not self._motion_enabled():
            self._finish_growth()

    def hideEvent(self, event):
        """Stop owned card animations when another module hides this page."""
        self._finish_growth()
        super().hideEvent(event)

    def set_cards(self, actions, leading=None, leading_index=0, groups=None):
        """Group all inspected choices while bounding animated field footprints."""
        from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QSequentialAnimationGroup

        had_cards = bool(self.cards)
        self._finish_growth()
        for card in self.cards:
            card.hide()
            card.deleteLater()
        groups = groups or {}
        ordering = {group: index for index, group in enumerate(dict.fromkeys(
            groups.get(action[0], {}).get('kind', '') for action in actions))}
        actions = sorted(actions, key=lambda action: ordering[groups.get(action[0], {}).get('kind', '')])
        self.cards = [_EasySuggestionCard(action, self) for action in actions]
        for card in self.cards:
            record = groups.get(card._key)
            if not record:
                continue
            group, reason = record.get('group', ''), record.get('reason', '')
            card.description.setText('<b>' + escape(group) + '</b><br>' + escape(reason))
            card.description.setAlignment(Qt.AlignCenter)
            card.description.setStyleSheet(f'font-size: {font_px(12)}px; font-weight: 300;')
            card.description.hide()
            card._explanation = '<b>' + escape(group) + '</b><br>' + escape(reason)
            card.setAccessibleDescription(group + ': ' + reason)
        if isinstance(leading, _EasyGuidanceCard):
            leading.hide()
            leading.deleteLater()
            leading = None
        if leading is not None:
            self.cards.insert(min(leading_index, len(self.cards)), leading)
        animate = self._motion_enabled() and self.isVisible()
        for card in self.cards:
            card._content_opacity.setOpacity(0.0 if animate else 1.0)
        self._arrange()
        for index, card in enumerate(self.cards):
            card.setVisible(self._suggestions_active and card._final_rect.intersects(self._row_rect))
            card.setMinimumSize(0, 0)
            card.raise_()
            if animate:
                target = QRect(card._final_rect)
                start = QRect(target.center().x(), target.center().y(), 1, 1)
                card._content_opacity.setOpacity(0.0)
                card.setGeometry(start)
                card._growing = True
                card.setEnabled(False)
                group = QSequentialAnimationGroup(self)
                group.addPause(index * 200)
                animation = QPropertyAnimation(card, b'geometry', group)
                animation.setStartValue(start)
                animation.setEndValue(target)
                animation.setDuration(500)
                animation.setEasingCurve(QEasingCurve.InOutCubic)
                animation.valueChanged.connect(card._growth_changed)
                group.addAnimation(animation)
                group.finished.connect(card._growth_finished)
                self._animations.append(group)
                group.start()
        self._publish_growth(invalidate=had_cards or bool(self.cards))

    def resizeEvent(self, event):
        """Keep suggestions in responsive slots while leaving the field centered."""
        super().resizeEvent(event)
        if hasattr(self, "panel"):
            self._finish_growth()
            self._arrange()

    def _place_fixed(self, widget, rect):
        """Preserve a control's global anchor while the workflow owns its native surface."""
        if widget.parentWidget() is not self:
            rect.moveTopLeft(widget.parentWidget().mapFromGlobal(self.mapToGlobal(rect.topLeft())))
        widget.setGeometry(rect)

    def explain_card(self, card):
        """Display the focused module's explanation below the fixed suggestion row."""
        self.details.setHtml(card._explanation)

    def _scroll_cards(self, value):
        """Move only the horizontal choice row, preserving input and description anchors."""
        self._finish_growth()
        self._arrange()

    def wheelEvent(self, event):
        """Scroll overflow choices horizontally with the native mouse or trackpad."""
        if self._row_rect.contains(event.position().toPoint()) and self._scrollbar.maximum():
            delta = event.pixelDelta().x() or event.pixelDelta().y()
            if not delta:
                delta = (event.angleDelta().x() or event.angleDelta().y()) // 2
            self._scrollbar.setValue(self._scrollbar.value() - delta)
            event.accept()
        else:
            super().wheelEvent(event)

    def _arrange(self):
        """Keep input and description fixed around one horizontally scrollable row."""
        from PySide6.QtGui import QBrush, QLinearGradient, QRegion

        width, height = self.width(), self.height()
        panel_width = min(720, max(180, width - 48))
        panel_height = self.panel.height()
        panel_y = max(16, (height - panel_height) // 2)
        self._place_fixed(self.panel, QRect((width - panel_width) // 2, panel_y, panel_width, panel_height))
        self._place_fixed(self.rim, QRect((width - panel_width) // 2, panel_y, panel_width, panel_height))
        self.rim.raise_()
        details_width = min(1560, max(180, width - 32))
        from ..preferences import scaled_px

        default_height = 176 + 2 * self.details.fontMetrics().lineSpacing()
        maximum_height = max(1, height - panel_y - panel_height - 32)
        wanted_height = (default_height if self._details_height_base is None
                         else scaled_px(self._details_height_base))
        details_height = (36 if self.details._minimized else
                          min(maximum_height, max(min(scaled_px(96), maximum_height), wanted_height)))
        self._place_fixed(self.details, QRect((width - details_width) // 2, height - details_height - 16,
                                              details_width, details_height))
        self._details_grip._min = min(scaled_px(96), maximum_height)
        self._details_grip._max = maximum_height
        self._details_grip.setGeometry(12, 0, max(1, details_width - 24), self._details_grip.height())
        self._details_grip.setVisible(not self.details._minimized)
        self._details_grip.raise_()
        self.details._place_controls()
        available = max(64, panel_y - 40)
        side = min(224, available, max(96, width - 64))
        card_height = max([side] + [min(available, max(card.sizeHint().height(), side))
                                  for card in self.cards if isinstance(card, _EasyRegexCard)])
        top = max(16, panel_y - card_height - 24)
        self._row_rect = QRect(16, top, max(1, width - 32), card_height)
        gap = 20
        total = len(self.cards) * side + max(0, len(self.cards) - 1) * gap
        maximum = max(0, total - self._row_rect.width())
        self._scrollbar.blockSignals(True)
        self._scrollbar.setRange(0, maximum)
        self._scrollbar.setPageStep(self._row_rect.width())
        self._scrollbar.setSingleStep(side + gap)
        self._scrollbar.blockSignals(False)
        self._scrollbar.setGeometry(24, self._row_rect.bottom() + 5, max(1, width - 48), 12)
        self._scrollbar.setVisible(self._suggestions_active and bool(maximum))
        start = self._row_rect.left() + max(0, (self._row_rect.width() - total) // 2)
        start -= self._scrollbar.value()
        for index, card in enumerate(self.cards):
            card._final_rect = QRect(start + index * (side + gap), top, side,
                                     card_height if isinstance(card, _EasyRegexCard) else side)
            if card.parentWidget() is self and not card._growing:
                card.setGeometry(card._final_rect)
            if card.parentWidget() is not self:
                continue
            if not self._suggestions_active:
                continue
            local = self._row_rect.translated(-card.x(), -card.y()).intersected(card.rect())
            if local.isEmpty():
                card.hide()
            else:
                card.show()
                card.setMask(QRegion(local))
            gradient = QLinearGradient(self._row_rect.left() - card.x(), 0,
                                       self._row_rect.right() - card.x(), 0)
            fade = min(0.15, 44 / max(1, self._row_rect.width()))
            gradient.setColorAt(0, QColor(0, 0, 0, 0 if self._scrollbar.value() else 255))
            gradient.setColorAt(fade, QColor(0, 0, 0, 255))
            gradient.setColorAt(1 - fade, QColor(0, 0, 0, 255))
            gradient.setColorAt(1, QColor(0, 0, 0, 0 if self._scrollbar.value() < maximum else 255))
            card._content_opacity.setOpacityMask(QBrush(gradient))
        self.details.raise_()



class _EasyWorkflowNode(_WorkflowNode):
    """Select a registered tool without opening it from a neighboring node."""

    def __init__(self, view, key):
        """Resolve one native icon and maturity once, outside repeated hover paints."""
        from ..app import _icon_for_app
        super().__init__(view, key)
        self._glyph = _icon_for_app(key, theme='dark')
        self._stage = _EasySuggestionCard._stage_for(key)
        self._hovered = False
        self._pressed = False

    def boundingRect(self):
        """Keep every neighboring module in the same rounded square footprint."""
        return QRectF(0, 0, self.view._node_side, self.view._node_side)

    def paint(self, painter, option, widget=None):
        """Draw the original icon sharply above a centered short module label."""
        from PySide6.QtGui import QPen
        from ..i18n import tr
        from ..theme import ALPHA_MARK, stage_hover, _STAGE_BUTTON_HOVER_ALPHA, _STAGE_BUTTON_PRESS_ALPHA

        palette = palette_for('dark')
        painter.setRenderHint(QPainter.Antialiasing)
        hue = QColor(stage_hover(self._stage))
        fill = QColor(25, 25, 25)
        if self._hovered or self._pressed:
            fill = QColor(hue)
            fill.setAlphaF(_STAGE_BUTTON_PRESS_ALPHA if self._pressed else _STAGE_BUTTON_HOVER_ALPHA)
        painter.setBrush(fill)
        painter.setPen(QPen(hue if self._hovered or self._pressed or self.highlighted
                            else QColor(palette['border']), 1.5))
        painter.drawRoundedRect(self.boundingRect().adjusted(1, 1, -1, -1), 18, 18)
        module = self.view.data['modules'][self.key]
        side = self.view._node_side
        icon_side = 80 if side >= 192 else 64
        icon_top = 28 if side >= 192 else 16
        icon_rect = QRect((side - icon_side) // 2, icon_top, icon_side, icon_side)
        if self._glyph is not None:
            self._glyph.paint(painter, icon_rect)
        font = QFont('Open Sans')
        font.setWeight(QFont.Light)
        font.setPixelSize(font_px(18, scale=1))
        painter.setFont(font)
        painter.setPen(QColor(palette['fg']))
        if self._stage == 'alpha':
            font.setPixelSize(font_px(28, scale=1))
            painter.setFont(font)
            painter.setPen(hue)
            painter.drawText(QRectF(side - 50, 8, 40, 40), Qt.AlignCenter, ALPHA_MARK)
            font.setPixelSize(font_px(18, scale=1))
            painter.setFont(font)
            painter.setPen(QColor(palette['fg']))
        title_top = icon_top + icon_side + 12
        painter.drawText(QRectF(12, title_top, side - 24, side - title_top - 12),
                         Qt.AlignCenter | Qt.TextWordWrap, tr(module['name']))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(palette['accent']))
        for port in (*getattr(self, '_input_ports', ()), *getattr(self, '_output_ports', ())):
            painter.drawEllipse(port, _EASY_WORKFLOW_PORT_RADIUS, _EASY_WORKFLOW_PORT_RADIUS)

    def hoverEnterEvent(self, event):
        """Describe a neighbor without moving the selected tool or changing its input."""
        self._hovered = True
        self.update()
        self.view.explain_node(self.key)
        event.accept()

    def hoverLeaveEvent(self, event):
        """Return candidate connections to the selected neighborhood after hovering."""
        self._hovered = False
        self.update()
        if self.view._selected_key is not None:
            self.view.explain_node(self.view._selected_key)
        event.accept()

    def mousePressEvent(self, event):
        """Center this tool first; only the central native card can launch it."""
        if event.button() == Qt.LeftButton:
            self._pressed = True
            self.update()
            self.setFocus()
        event.accept()

    def mouseReleaseEvent(self, event):
        """Paint native press feedback before selecting the released node."""
        chosen = self._pressed and event.button() == Qt.LeftButton and self.boundingRect().contains(event.pos())
        self._pressed = False
        self.update()
        if chosen:
            self.view.chosen.emit(self.key)
        event.accept()

    def keyPressEvent(self, event):
        """Offer the same explicit selection to keyboard navigation."""
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.view.chosen.emit(self.key)
            event.accept()
        else:
            super().keyPressEvent(event)


class _EasyWorkflowEdge(_WorkflowEdge):
    """Curve verified handoffs through shared clear gutters around square cards."""

    def __init__(self, view, edge, start, end):
        """Retain the real handoff and attach it to the fixed card's scene origin."""
        super().__init__(view, edge, start, end)
        self._set_origin(start)

    def _set_origin(self, start):
        """Route distinct ports around card bounds and center the direction arrow by length."""
        from math import cos, hypot, radians, sin
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QPainterPath, QPolygonF

        self.prepareGeometryChange()
        self.start = start
        node = self.view.nodes[self.edge['to']]
        spine = 8 if self.view._direction > 0 else self.view.viewport().width() - 8
        gutter = node.pos().y() - 46
        approach = node.pos().x() - 24
        points = [start]
        if (self.edge['from'] == getattr(self.view, '_root_key', None)
                and (self.view._direction < 0 or self.view._root_x is not None)):
            root = self.view.parentWidget().central
            root_top = self.view.mapToScene(self.view.viewport().mapFromGlobal(root.mapToGlobal(root.rect().topLeft())))
            outside = start.x() + 24
            over = min(root_top.y() - 38, self.view._first_y - 46)
            points.extend((QPointF(outside, start.y()), QPointF(outside, over), QPointF(spine, over)))
        else:
            points.append(QPointF(spine, start.y()))
        self._lead_path = QPainterPath(start)
        for point in points[1:]:
            self._lead_path.lineTo(point)
        points.extend((QPointF(spine, gutter), QPointF(approach, gutter),
                       QPointF(approach, self.end.y()), self.end))
        unique = [points[0]]
        for point in points[1:]:
            if point != unique[-1]:
                unique.append(point)
        path = QPainterPath(start)
        for previous, corner, following in zip(unique, unique[1:], unique[2:]):
            incoming, outgoing = corner - previous, following - corner
            before, after = hypot(incoming.x(), incoming.y()), hypot(outgoing.x(), outgoing.y())
            radius = min(14.0, before / 2, after / 2)
            path.lineTo(corner - incoming * (radius / before))
            path.quadTo(corner, corner + outgoing * (radius / after))
        path.lineTo(self.end)
        self.setPath(path)
        fraction = path.percentAtLength(path.length() / 2)
        self._arrow_center = path.pointAtPercent(fraction)
        angle = radians(path.angleAtPercent(fraction))
        direction = QPointF(cos(angle), -sin(angle))
        self._arrow_direction = direction
        normal = QPointF(-direction.y(), direction.x())
        self.arrowhead = QPolygonF([self._arrow_center + direction * 6,
                                   self._arrow_center - direction * 3 + normal * 4,
                                   self._arrow_center - direction * 3 - normal * 4])

    def _badge_rect(self):
        """Use the same badge clearance above every module and owned submodule."""
        node = self.view.nodes[self.edge['to']]
        return QRectF(node.pos().x(), node.pos().y() - 38, self.view._node_side, 24)

    def boundingRect(self):
        """Include connector labels in scene culling and repaint bounds."""
        bounds = super().boundingRect()
        if self.edge.get('mode_labels'):
            bounds = bounds.united(self._badge_rect())
        return bounds

    def paint(self, painter, option, widget=None):
        """Let the transparent owner draw complete curves beyond the child viewport."""
        return

    def draw_connection(self, painter):
        """Distinguish ownership while keeping midpoint arrows clear of mode badges."""
        from PySide6.QtGui import QBrush, QLinearGradient, QPainterPath, QPainterPathStroker, QPen
        from ..i18n import tr

        painter.save()
        painter.setOpacity(painter.opacity() * self.view.nodes[self.edge['to']].opacity())
        viewport = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
        clip = QPainterPath()
        clip.addRect(viewport)
        stroke = QPainterPathStroker()
        stroke.setWidth(40)
        clip = clip.united(stroke.createStroke(self._lead_path))
        painter.setClipPath(clip, Qt.IntersectClip)
        palette = palette_for('dark')
        utility = self.edge['kind'] == 'utility'
        color = QColor(palette['fg_muted'] if utility else palette['accent'])
        color.setAlpha(200 if self.highlighted else 125)
        gradient = QLinearGradient(self.start, self.end)
        faint = QColor(color)
        faint.setAlpha(45)
        gradient.setColorAt(0, faint)
        gradient.setColorAt(1, color)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Qt.NoBrush)
        if self.edge['kind'] not in ('containment', 'utility'):
            halo = QColor(color)
            halo.setAlpha(24)
            painter.setPen(QPen(halo, 5))
            painter.drawPath(self.path())
        pen = QPen(QBrush(gradient), 1.2 if self.edge['kind'] in ('containment', 'utility') else 1.6)
        if utility:
            pen.setStyle(Qt.DashLine)
        elif self.edge['kind'] == 'containment':
            pen.setStyle(Qt.DotLine)
        painter.setPen(pen)
        painter.drawPath(self.path())
        painter.setBrush(color)
        painter.drawPolygon(self.arrowhead)
        labels = self.edge.get('mode_labels', ())
        if labels:
            painter.save()
            painter.setClipRect(viewport, Qt.IntersectClip)
            font = QFont('Open Sans')
            font.setWeight(QFont.Light)
            font.setPixelSize(font_px(12))
            painter.setFont(font)
            painter.setBrush(QColor(palette['page']))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(self._badge_rect(), 8, 8)
            painter.setPen(QColor(palette['fg']))
            painter.drawText(self._badge_rect(), Qt.AlignCenter, ' / '.join(tr(label) for label in labels))
            painter.restore()
        painter.restore()


class _EasyWorkflowView(_WorkflowView):
    """Reveal only the current module's direct documented downstream branches."""

    chosen = Signal(str)
    viewport_changed = Signal()
    _PRIORITIES = {
        'mask': ('measure', 'overview', 'timelapse', 'ops'),
        'measure': ('annotate', 'classify_merged', 'umap', 'graph_builder'),
        'make_masks': ('train_cellpose', 'measure', 'analyze_plaques', 'model_compare'),
        'regression': ('hit_list', 'volcano_explorer', 'run_compare', 'regression_diagnostics'),
        'map_barcodes': ('barcode_qc', 'regression', 'spatial_map'),
    }

    def __init__(self, data, parent=None):
        """Retain existing nodes and descriptions, excluding the global edge network."""
        from shiboken6 import delete
        from PySide6.QtCore import QVariantAnimation, QEasingCurve

        self._selected_key = None
        self._columns = 1
        self._direction = 1
        self._root_y = 0
        self._root_x = None
        self._node_side = 192
        self._first_y = 64
        self.branch_records = []
        self.remaining_records = []
        self._expanded_keys = []
        self._visible_records = []
        self._node_order = []
        super().__init__(data, keys=(), parent=parent)
        self._growth_nodes = []
        self._growth = QVariantAnimation(self)
        self._growth.setStartValue(0.0)
        self._growth.setEndValue(1.0)
        self._growth.setDuration(350)
        self._growth.setEasingCurve(QEasingCurve.InOutCubic)
        self._growth.valueChanged.connect(self._grow_branches)
        self.keys = list(data['modules'])
        self._fitted = True
        self._auto_fit = False
        self.setObjectName('EasyWorkflowView')
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.verticalScrollBar().valueChanged.connect(self._position_connections)
        for edge in self.edges:
            self.scene().removeItem(edge)
            delete(edge)
        self.edges = []
        for key in self.keys:
            node = _EasyWorkflowNode(self, key)
            self.scene().addItem(node)
            self.nodes[key] = node
            node.hide()

    def explain_node(self, key):
        """Describe the inspected tool without changing the selected root or input."""
        from .workflow_diagram import node_description

        from ..i18n import tr

        if getattr(self, '_explanation_key', None) == key:
            return
        cache = self.__dict__.setdefault('_explanation_cache', {})
        if key not in cache:
            text = node_description(self.data, key)
            note = self.data['modules'][key].get('documentation_note')
            if note:
                text += '<p>' + escape(tr(note)) + '</p>'
            cache[key] = text
        self._explanation_key = key
        self.explanation.emit(cache[key])

    def _downstream_records(self, selected):
        """Keep scientific handoffs first and disclose actual Help tools as navigation."""
        from ..i18n import tr
        helpers = self.data.get('easy_helpers', ())
        known = {(record['from'], record['to']) for record in self.data['easy_connections']}
        if selected not in helpers:
            for key in helpers:
                if key != selected and (selected, key) not in known:
                    self.data['easy_connections'].append(dict(
                        **{'from': selected, 'to': key}, artifacts=[], handoff='',
                        kind='utility', strength=0, mode_labels=[tr('Tools')]))
        candidates = {}
        for record in self.data['easy_connections']:
            if (record['from'] == selected and record['to'] != selected
                    and record['kind'] in ('documented', 'containment', 'utility')):
                candidates.setdefault(record['to'], record)
        priority = self._PRIORITIES.get(selected, ())
        keys = [key for key in priority if key in candidates]
        keys.extend(key for key, record in candidates.items()
                    if key not in keys and record['kind'] != 'utility')
        keys.extend(key for key, record in candidates.items()
                    if key not in keys and record['kind'] == 'utility')
        return [candidates[key] for key in keys]

    def center_selection(self, key):
        """Append the selected tool's actual branches while retaining the explored path."""
        self._growth.stop()
        self._grow_branches(1.0)
        previous = set(self._node_order)
        self._selected_key = key
        if not self._expanded_keys:
            self._root_key = key
        records = self._downstream_records(key)
        self.branch_records = records[:4]
        self.remaining_records = records[4:]
        if key != self._root_key and key not in self._node_order:
            self._node_order.append(key)
        if key not in self._expanded_keys:
            self._expanded_keys.append(key)
            known = {(record['from'], record['to']) for record in self._visible_records}
            for record in self.branch_records:
                if record['to'] == self._root_key:
                    continue
                pair = (record['from'], record['to'])
                if pair not in known:
                    self._visible_records.append(record)
                    known.add(pair)
                if record['to'] not in self._node_order:
                    self._node_order.append(record['to'])
        self._arrange_branches()
        self.explain_node(key)
        self._growth_nodes = [self.nodes[child] for child in self._node_order if child not in previous]
        from ..preferences import get_ambient_enabled, get_reduce_motion
        if previous and self._growth_nodes and get_ambient_enabled() and not get_reduce_motion():
            self._grow_branches(0.0)
            self._growth.start()

    def _grow_branches(self, value):
        """Apply growth opacity to newly disclosed cards and update their surfaces.

        :param value: animation progress converted to each card's opacity.
        """
        for node in self._growth_nodes:
            node.setOpacity(float(value))
        self.viewport().update()
        self.parentWidget().update()

    def _arrange_branches(self):
        """Keep earlier cards in order and append newly disclosed children below them."""
        from PySide6.QtCore import QPointF
        from shiboken6 import delete

        width = max(224, self.viewport().width())
        self._columns = 2 if width >= 488 else 1
        for key, node in self.nodes.items():
            node.setVisible(key in self._node_order)
            node.highlighted = key == self._selected_key
        for index, key in enumerate(self._node_order):
            row, column = divmod(index, self._columns)
            x = 32 + column * (self._node_side + 40)
            if self._direction < 0:
                x = width - x - self._node_side
            node = self.nodes[key]
            node.setPos(x, self._first_y + row * (self._node_side + 64))
            node._input_ports = (QPointF(5, self._node_side * 0.75),)
            outgoing = [record for record in self._visible_records if record['from'] == key]
            node._output_ports = tuple(QPointF(self._node_side - 5,
                16 + index * 14) for index in range(len(outgoing)))
            node.update()
        old_edges, self.edges = self.edges, []
        for edge in old_edges:
            self.scene().removeItem(edge)
            delete(edge)
        for record in self._visible_records:
            node = self.nodes[record['to']]
            edge = _EasyWorkflowEdge(self, record, QPointF(0, self._root_y),
                                     node.pos() + node._input_ports[0])
            self.scene().addItem(edge)
            self.edges.append(edge)
        rows = (len(self._node_order) + self._columns - 1) // self._columns
        height = (self._first_y + max(0, rows - 1) * (self._node_side + 64)
                  + (self._node_side if rows else 0) + 16)
        self.setSceneRect(QRectF(0, 0, width, max(height, self.viewport().height())))
        self._position_connections()

    def _position_connections(self):
        """Attach every retained edge to its actual root or explored parent card."""
        parent = self.parentWidget()
        if parent.central is None:
            return
        root_edges = [edge for edge in self.edges if edge.edge['from'] == self._root_key]
        ports = getattr(parent.central, '_workflow_ports', ())
        for port, edge in zip(ports, root_edges):
            point = self.viewport().mapFromGlobal(parent.central.mapToGlobal(port))
            edge._set_origin(self.mapToScene(point))
        for key in self._node_order:
            source = self.nodes[key]
            edges = [edge for edge in self.edges if edge.edge['from'] == key]
            for port, edge in zip(getattr(source, '_output_ports', ()), edges):
                edge._set_origin(source.pos() + port)
        parent.update()

    def resizeEvent(self, event):
        """Reflow child rows without moving the original selected module card."""
        super().resizeEvent(event)
        if self._selected_key is not None:
            self._arrange_branches()


class _EasyWorkflowPage(QWidget):
    """Keep a prominent selected action above a progressively explored workflow."""

    back_requested = Signal()
    open_requested = Signal(str, str)

    def __init__(self, data, actions, selected, source_card, anchor, parent=None):
        """Separate available navigation from scientific compatibility and future output."""
        from PySide6.QtCore import QEasingCurve, QVariantAnimation
        from PySide6.QtWidgets import QGraphicsOpacityEffect, QTextBrowser
        from .field_fade import OPT_OUT_PROPERTY
        from ..i18n import tr

        self.central = None
        self._chosen_key = None
        self._source_card = source_card
        self._source_path = source_card._path
        self._source_parent = source_card.parentWidget()
        self._anchor = QRect(anchor)
        self._progress = 0.0
        super().__init__(parent)
        self.setObjectName('EasyWorkflowPage')
        self.setProperty('spacrAmbientPassiveOverlay', True)
        make_transparent(self)
        self.setProperty("spacrAmbientTransparentOverlay", True)
        self.data = data
        field = self._source_parent
        self._fixed_controls = (field.panel, field.rim, field.details)
        for control in self._fixed_controls:
            rect = QRect(self.mapFromGlobal(control.mapToGlobal(control.rect().topLeft())), control.size())
            control.setParent(self)
            control.setGeometry(rect)
            control.show()
        self._source_card.setParent(self)
        self._source_card.activated.connect(self.open_requested)
        self.routes = {action[0]: action for action in actions}
        artifacts = {'mask': {'raw_images'}, 'make_masks': {'raw_images'},
                     'measure': {'merged'}, 'db_browser': {'measurements'}, 'foreign': {'foreign_tables'}}
        self.native_routes = set()
        for key, module in data['modules'].items():
            if key in self.routes or key in data.get('easy_helpers', ()):
                continue
            for action in actions:
                if set(module.get('inputs', ())) & artifacts.get(action[0], set()):
                    self.routes[key] = (key, module['name'], module.get('guidance', ''), action[3])
                    self.native_routes.add(key)
                    break
            if module.get('route') == 'overview' and module['host_key'] in self.routes:
                host = self.routes[module['host_key']]
                self.routes[key] = (key, module['name'], module.get('guidance', ''), host[3])
        self.view = _EasyWorkflowView(data, self)
        make_transparent(self.view)
        for surface in (self.view, self.view.viewport()):
            surface.setProperty("spacrAmbientTransparentOverlay", True)
        self.view.chosen.connect(self.select)
        self.back = QPushButton(tr('Back'), self)
        self.back.setObjectName('EasyWorkflowBack')
        self.back.clicked.connect(self.back_requested)
        self.tools = QComboBox(self)
        self.tools.setObjectName('EasyWorkflowTools')
        self.tools.setProperty(OPT_OUT_PROPERTY, True)
        self.tools.setPlaceholderText(tr('All tools'))
        self.tools.setAccessibleName(tr('All tools'))
        titles = {key: tr(module['name']) for key, module in data['modules'].items()}
        ordering = {key: title.casefold() for key, title in titles.items()}
        for key in sorted(titles, key=ordering.__getitem__):
            self.tools.addItem(titles[key], key)
        self.tools.activated.connect(self._choose_tool)
        self.more = QComboBox(self)
        self.more.setObjectName('EasyWorkflowNextTools')
        self.more.setProperty(OPT_OUT_PROPERTY, True)
        self.more.setPlaceholderText(tr('More next steps'))
        self.more.setAccessibleName(tr('More next steps'))
        self.more.activated.connect(self._choose_next)
        self.details = parent.field.details
        self.details.setOpenExternalLinks(False)
        self.details.setOpenLinks(False)
        self.details.setAccessibleName(tr('Module description'))
        self.view.explanation.connect(self.details.setHtml)
        self.empty = QLabel(tr('No suggested next steps.'), self)
        self.empty.setAlignment(Qt.AlignCenter)
        make_transparent(self.empty)
        self.select(selected)
        self._departing = [card for card in self._source_parent.cards
                           if card is not source_card and card.isVisible()
                           and card._content_opacity.opacity() > 0.0]
        self._departing.extend((field.panel, field.rim))
        if self._source_parent._scrollbar.maximum():
            self._departing.append(self._source_parent._scrollbar)
        for card in self._departing:
            effect = getattr(card, '_content_opacity', None)
            card._navigator_departure_opacity = effect.opacity() if effect is not None else 1.0
        self._effects = []
        for widget in (self.view, self.tools, self.more, self.back, self.empty):
            effect = QGraphicsOpacityEffect(widget)
            effect.setOpacity(0.0)
            widget.setGraphicsEffect(effect)
            self._effects.append(effect)
        self._transition = QVariantAnimation(self)
        self._transition.setStartValue(0.0)
        self._transition.setEndValue(1.0)
        self._transition.setDuration(500)
        self._transition.setEasingCurve(QEasingCurve.InOutCubic)
        self._transition.valueChanged.connect(self._emerge)
        self._entered = False

    def _choose_tool(self, index):
        """Select an available tool without inventing an input or starting a run."""
        key = self.tools.itemData(index)
        if key:
            self.select(key)

    def _choose_next(self, index):
        """Disclose an additional real downstream option from the current module."""
        key = self.more.itemData(index)
        if key:
            self.select(key)

    def select(self, key):
        """Explore a child's branches first and open only on a repeated explicit choice."""
        from ..i18n import tr

        if key not in self.data['modules']:
            return
        if key == self._chosen_key:
            action = self.routes.get(key)
            source = action[3] if action and action[3] else self._source_path
            self.open_requested.emit(key, '' if key in self.data.get('easy_helpers', ()) else source)
            return
        self._chosen_key = key
        if self.central is None:
            self.central = self._source_card
            self.central.show()
        self.view.center_selection(key)
        self.tools.setCurrentIndex(-1)
        self.more.clear()
        for record in self.view.remaining_records:
            child = record['to']
            self.more.addItem(tr(self.data['modules'][child]['name']), child)
        self.more.setCurrentIndex(-1)
        self.more.setVisible(bool(self.view.remaining_records))
        self.empty.setVisible(not self.view._visible_records)
        self._arrange()
        if key != self._source_card._key:
            self.view.nodes[key].setFocus()

    def _arrange(self):
        """Place the branch on the free side without recentering the selected card."""
        if self.central is None:
            return
        self.central.clearMask()
        from PySide6.QtGui import QBrush
        self.central._content_opacity.setOpacityMask(QBrush())
        self.central._content_opacity.setOpacity(1.0)
        self.central.setGeometry(self._anchor)
        from PySide6.QtCore import QPoint

        count = sum(record['from'] == self.view._root_key for record in self.view._visible_records)
        first, last = 16, max(16, self.central.height() // 3)
        self.central._workflow_ports = tuple(QPoint(self.central.width() - 5,
            round((first + last) / 2 if count == 1 else first + index * (last - first) / (count - 1)))
            for index in range(count))
        self.central._port_opacity = self._progress
        self.central.update()
        header_height = max(32, self.back.sizeHint().height(),
                            self.tools.minimumSizeHint().height(), self.more.minimumSizeHint().height())
        stacked = self.width() < 572 and self.more.isVisible()
        header_bottom = 16 + header_height * (2 if stacked else 1) + (8 if stacked else 0)
        field = self._source_parent
        field._arrange()
        origin = self.mapFromGlobal(field.details.mapToGlobal(field.details.rect().topLeft()))
        bottom = QRect(origin, field.details.size())
        search_origin = self.mapFromGlobal(field.panel.mapToGlobal(field.panel.rect().topLeft()))
        search = QRect(search_origin, field.panel.size()).adjusted(-4, -4, 4, 4)
        top = header_bottom + 16
        limit = bottom.top() - 16
        self.view._direction = 1 if self._anchor.center().x() < self.width() / 2 else -1
        if self.view._direction > 0:
            x, width = self._anchor.right() + 1, self.width() - self._anchor.right() - 17
        else:
            x, width = 16, self._anchor.left() - 16
        self.view._node_side = 192
        self.view._first_y = 64
        self.view._root_x = None
        self.view._root_y = self._anchor.center().y() - top
        height = limit - top
        if width < 224:
            x, width = 16, self.width() - 32
            self.view._node_side = 144
            self.view._first_y = 46
            self.view._root_x = self._anchor.center().x() - x
            above = self._anchor.top() - top
            below = limit - self._anchor.bottom() - 1
            if above >= below:
                height = above
                self.view._root_y = height
            else:
                top = self._anchor.bottom() + 1
                height = below
                self.view._root_y = 0
        self.view.setGeometry(x, top, max(1, width), max(1, height))
        self.view._arrange_branches()
        self.empty.setGeometry(self.view.geometry())
        self.back.setGeometry(16, 16, max(70, self.back.sizeHint().width()), header_height)
        self.tools.setGeometry(max(104, self.width() - 236), 16, 220, header_height)
        self.more.setGeometry(16 if stacked else max(104, self.width() - 468),
                              24 + header_height if stacked else 16, 220, header_height)
        self.central.raise_()
        for control in self._fixed_controls:
            control.raise_()
        self.back.raise_()
        self.tools.raise_()
        self.more.raise_()

    def _emerge(self, value):
        """Dim the surrounding live field gradually as the branch becomes visible."""
        self._progress = float(value)
        if self.central is not None:
            self.central._port_opacity = self._progress
            self.central.update()
        for effect in self._effects:
            effect.setOpacity(self._progress)
        for card in self._departing:
            effect = getattr(card, '_content_opacity', None)
            if effect is None:
                from PySide6.QtWidgets import QGraphicsOpacityEffect
                effect = QGraphicsOpacityEffect(card)
                card.setGraphicsEffect(effect)
                card._content_opacity = effect
            effect.setOpacity(card._navigator_departure_opacity * (1.0 - self._progress))
            card.setEnabled(False)
            if self._progress >= 1.0:
                card.hide()
        self.update()

    def _sync_run_state(self):
        """Finish this single transition immediately when decorative motion is disabled."""
        from ..preferences import get_ambient_enabled, get_reduce_motion

        if not get_ambient_enabled() or get_reduce_motion():
            self._transition.stop()
            self._emerge(1.0)

    def hideEvent(self, event):
        """Retire the selected-node visit before another module owns the field."""
        super().hideEvent(event)

    def showEvent(self, event):
        """Animate the network once while its original native card remains unchanged."""
        from ..preferences import get_ambient_enabled, get_reduce_motion

        super().showEvent(event)
        if not self._entered:
            self._entered = True
            if get_ambient_enabled() and not get_reduce_motion():
                self._transition.start()
            else:
                self._emerge(1.0)

    def paintEvent(self, event):
        """Shade only the field gently, leaving the selected card's anchor undimmed."""
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QRadialGradient

        painter = QPainter(self)
        gradient = QRadialGradient(QPointF(self._anchor.center()), max(self.width(), self.height()))
        gradient.setColorAt(0.0, QColor(0, 0, 0, 0))
        gradient.setColorAt(0.25, QColor(0, 0, 0, round(28 * self._progress)))
        gradient.setColorAt(1.0, QColor(0, 0, 0, round(90 * self._progress)))
        painter.fillRect(self.rect(), gradient)
        painter.setOpacity(self._progress)
        painter.translate(self.view.viewport().mapTo(self, self.view.viewport().rect().topLeft()))
        painter.setWorldTransform(self.view.viewportTransform(), True)
        for edge in self.view.edges:
            node = self.view.nodes[edge.edge['to']]
            visible = self.view.mapFromScene(node.sceneBoundingRect()).boundingRect()
            if visible.intersects(self.view.viewport().rect()):
                edge.draw_connection(painter)
        painter.end()

    def release_source_card(self):
        """Restore the original card to its unchanged suggestion list before retiring."""
        self._transition.stop()
        self.view._growth.stop()
        self._source_card.activated.disconnect(self.open_requested)
        self._source_card._workflow_ports = ()
        self._source_card.setParent(self._source_parent)
        for control in self._fixed_controls:
            control.setParent(self._source_parent)
            control.show()
        self._source_card.setGeometry(self._source_card._final_rect)
        for card in (*self._source_parent.cards, *self._departing):
            effect = getattr(card, '_content_opacity', None)
            if effect is not None:
                effect.setOpacity(1.0)
            card.setEnabled(True)
            card.show()
        self._source_parent._arrange()
        self._source_card.show()
        return self._source_card

    def resizeEvent(self, event):
        """Follow only the original grid placement when the window itself resizes."""
        super().resizeEvent(event)
        if self.central is not None:
            source_rect = self._source_card._final_rect
            origin = self.mapFromGlobal(self._source_parent.mapToGlobal(source_rect.topLeft()))
            self._anchor = QRect(origin, source_rect.size())
            self._arrange()


class _EasyHomePage(QWidget):
    """A minimal content-guided Home that remains useful with every AI provider off."""

    tile_clicked = Signal(str)
    sample_project_requested = Signal()
    update_check_requested = Signal()
    news_refresh_requested = Signal()
    module_requested = Signal(str, str)
    command_requested = Signal(str)

    def __init__(self, parent=None):
        """Build only the path field initially; expose guidance after an input attempt."""
        from PySide6.QtGui import QKeySequence, QShortcut
        from ..i18n import tr
        from ..job_runner import JobRunner

        super().__init__(parent)
        self.setObjectName('EasyHomePage')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self.field = _EasyHomeField(scroll)
        scroll.setWidget(self.field)
        layout.addWidget(scroll)
        self._suggestions_scroll = scroll
        self._workflow = None
        self._workflow_escape = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self._workflow_escape.activated.connect(self._show_suggestions)
        self.path_edit = self.field.panel
        self.path_edit.browse.clicked.connect(self._browse)
        self._ai_provider = ''
        self._ai_thread = None
        self._ai_worker = None
        self._ai_relay = None
        self._ai_retired = []
        self._ai_generation = 0
        self._ai_text = ''
        self._ai_reply_timer = QTimer(self)
        self._ai_reply_timer.setSingleShot(True)
        self._ai_reply_timer.setInterval(50)
        self._ai_reply_timer.timeout.connect(self._render_ai_reply)
        self.path_edit.ai.clicked.connect(self._toggle_navigator_ai)
        self.path_edit.ai.setContextMenuPolicy(Qt.CustomContextMenu)
        self.path_edit.ai.customContextMenuRequested.connect(self._choose_navigator_ai)
        self._update_ai_tooltip()
        self.path_edit.setToolTip(tr('Local suggestions. No AI provider or network connection is required.'))
        self.status = QLabel(self.field)
        self.status.setObjectName('EasyInputFeedback')
        self.status.setWordWrap(True)
        self.status.setAccessibleName(tr('Input guidance'))
        self.status.hide()
        self.field.feedback = self.status
        self._runner = JobRunner(self, app_key='home', user_visible=False)
        self._runner.job_failed.connect(self._inspection_failed)
        self._runner.job_finished.connect(self._inspection_finished)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(600)
        self._debounce.timeout.connect(self._inspect)
        self.path_edit.textEdited.connect(self._schedule_inspection)
        self.path_edit.returnPressed.connect(self._submit)
        self._requested = ''
        self._queued_inspection = False
        self._input_generation = 0
        self._last_result = None
        self._settings_overrides = {}
        self._regex_cancel = None
        self._guide_visible = False
        self._guide_writing = False
        self._details_has_content = False
        self.field.details.textChanged.connect(self._guide_details_changed)
        self.path_edit.textChanged.connect(self._guide_input_changed)
        self.refresh()
        self._show_empty_guide()

    def _guide_details_changed(self):
        """Distinguish real descriptions and answers from the guide's own writes."""
        if not self._guide_writing:
            self._guide_visible = False
            self._details_has_content = bool(self.field.details.toPlainText().strip())

    def _guide_input_changed(self, text):
        """Retire introductory copy on the first edit, including a dropped path."""
        if text.strip():
            if self._guide_visible:
                self._guide_writing = True
                self.field.details.clear()
                self._guide_writing = False
                self._guide_visible = False
        else:
            self._show_empty_guide()

    def _show_empty_guide(self):
        """Use only an otherwise unoccupied lower description box for guidance."""
        from ..i18n import tr

        if (self.path_edit.text().strip() or self.field.cards or self._workflow is not None
                or self._ai_thread is not None or self._details_has_content):
            return
        folder = 'C:/Data/experiment/plate1' if os.name == 'nt' else '/data/experiment/plate1'
        image = folder + '/image.tif'
        lines = (
            tr('File/folder: Drop an image or a folder of images to see image based modules.'),
            tr('Path: Paste {folder} or {image} to see compatible modules. You can also type a file type, such as tif.',
               folder=folder, image=image),
            tr('Question: Try “segment images” to find segmentation tools.'),
            tr('Module: Type “Measure” to find the measurement module.'),
            tr('Explore: Click once to open the module network. Click twice to open the module.'),
            tr('spaCR AI: Ask for help using spaCR, e.g., ‘How do I generate object masks?’ or ‘How do i train a Cellpose or Torch model?’'),
        )
        paragraphs = []
        for line in lines:
            parts = re.split(r'[:：]', line, maxsplit=1)
            text = ('<b>' + escape(parts[0]) + ':</b>' + escape(parts[1])
                    if len(parts) == 2 else escape(line))
            paragraphs.append('<p>' + text + '</p>')
        self._guide_writing = True
        self.field.details.setHtml('<div style="color:#808080;">' + ''.join(paragraphs) + '</div>')
        self._guide_writing = False
        self._guide_visible = True
        self._details_has_content = False

    def _update_ai_tooltip(self):
        """Describe the selected provider and the explicit change-provider gesture."""
        from ..ai.providers import PROVIDER_CHOICES
        from ..i18n import tr

        button = self.path_edit.ai
        label = next((tr(label) for name, label, _needs in PROVIDER_CHOICES
                      if name == self._ai_provider), '')
        button.setToolTip(tr('AI: {provider}. Click to toggle; right-click to change provider.',
                             provider=label) if button.isChecked() else
                          tr('AI assistance is off. Click to choose a provider.'))

    def _toggle_navigator_ai(self, on):
        """Reuse consent after first selection and cancel immediately when disabled."""
        from ..ai.providers import PROVIDER_CHOICES
        from ..i18n import tr
        from ..preferences import _settings

        if not on:
            pending = self._ai_thread is not None
            self._cancel_navigator_ai()
            if pending:
                self._feedback(tr('Cancelled.'))
        elif not self._ai_provider:
            self._ai_provider = str(_settings().value('home/navigator_ai_provider', '') or '')
            if self._ai_provider not in {name for name, _label, _needs in PROVIDER_CHOICES if name != 'none'}:
                self._ai_provider = ''
                self.path_edit.ai.setChecked(False)
                self._choose_navigator_ai()
                return
        self.path_edit.ai._provider = self._ai_provider
        self.path_edit.ai._tick()
        self._update_ai_tooltip()

    def _choose_navigator_ai(self, *_args):
        """Explain sharing and assistance before accepting an explicit provider choice."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser

        from ..ai.providers import PROVIDER_CHOICES, apply_provider_choice
        from ..i18n import tr
        from ..preferences import _settings

        dialog = QDialog(self)
        dialog.setObjectName('NavigatorAiProviderDialog')
        dialog.setWindowTitle(tr('Navigator AI'))
        layout = QVBoxLayout(dialog)
        explanation = QTextBrowser(dialog)
        explanation.setMinimumSize(520, 230)
        explanation.setPlainText(tr(
            'AI explains tools, settings, workflows and errors throughout spaCR. '
            'In Navigator, unmatched tasks go to your chosen AI and answers appear below. '
            'It does not run an analysis from this field.') + '\n\n' + tr(
            'spaCR AI uses the hosted service and daily allowance. Claude, Codex and Gemini '
            'use your own CLI login. Submitted text is shared with the chosen provider. '
            'Check AI advice before using it.'))
        layout.addWidget(explanation)
        combo = QComboBox(dialog)
        combo.setObjectName('NavigatorAiProviderChoice')
        combo.setAccessibleName(tr('Choose AI provider'))
        for name, label, _needs in PROVIDER_CHOICES:
            if name != 'none':
                combo.addItem(tr(label), name)
        selected = combo.findData(self._ai_provider)
        if selected >= 0:
            combo.setCurrentIndex(selected)
        layout.addWidget(combo)
        providers = QPushButton(tr('Providers…'), dialog)

        def configure():
            """Reuse the existing install and sign-in dialog without sending a task."""
            from .ai_chat_panel import _ProvidersDialog

            _ProvidersDialog(dialog).exec()

        providers.clicked.connect(configure)
        layout.addWidget(providers)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        buttons.button(QDialogButtonBox.Ok).setText(tr('Use AI'))
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.Accepted:
            self._cancel_navigator_ai()
            self._ai_provider = str(combo.currentData())
            _settings().setValue('home/navigator_ai_provider', self._ai_provider)
            apply_provider_choice(self._ai_provider)
            self.path_edit.ai._provider = self._ai_provider
            self.path_edit.ai.setChecked(True)
        self.path_edit.ai._tick()
        self._update_ai_tooltip()
        dialog.deleteLater()

    @staticmethod
    def _navigator_ai_can_answer(value):
        """Keep submitted paths and filenames in local inspection even with AI on."""
        value = str(value).strip()
        return bool(value and not (
            value.startswith(('/', '\\', '~', '.')) or re.match(r'^[A-Za-z]:[\\/]', value)
            or re.match(r'^\w+://', value)
            or os.path.exists(os.path.expanduser(value))
            or (not re.search(r'\s', value) and (
                '/' in value or '\\' in value or re.search(r'\.[A-Za-z0-9]{1,10}$', value)))))

    @staticmethod
    def _navigator_local_reply(value):
        """Answer recognized literary prompts locally while leaving paths untouched.

        :param value: the submitted Navigator text.
        :returns: a translated reply for a recognized prompt, else an empty string.
        """
        from ..app import _easy_command_text
        from ..i18n import tr

        if not _EasyHomePage._navigator_ai_can_answer(value):
            return ''
        phrases = (
            '42', 'the answer to life', 'meaning of life', 'what is the meaning of life',
            'what is the meaning of 42', 'what does 42 mean', 'life the universe and everything',
            'svaret på livet', 'meningen med livet', 'vad är meningen med 42',
            'vad betyder 42', 'livet universum och allting',
            'die antwort auf das leben', 'der sinn des lebens', 'was ist die bedeutung von 42',
            'was bedeutet 42', 'das leben das universum und der ganze rest',
            'la respuesta a la vida', 'el sentido de la vida', 'cuál es el significado de 42',
            'qué significa 42', 'la vida el universo y todo lo demás',
            'a resposta para a vida', 'o sentido da vida', 'qual é o significado de 42',
            'o que significa 42', 'a vida o universo e tudo mais',
            'la réponse à la vie', 'le sens de la vie', 'quelle est la signification de 42',
            'que signifie 42', "la vie l'univers et le reste",
            '生命的答案', '生命的意义', '42的意义是什么', '42是什么意思', '生命宇宙以及一切',
            'जीवन का उत्तर', 'जीवन का अर्थ', '42 का अर्थ क्या है',
            '42 का क्या मतलब है', 'जीवन ब्रह्मांड और सब कुछ',
            '삶의 답', '삶의 의미', '42의 의미는 무엇인가요', '42는 무슨 뜻인가요',
            '삶 우주 그리고 모든 것', 'svarið við lífinu', 'tilgangur lífsins',
            'hver er merking 42', 'hvað þýðir 42', 'lífið alheimurinn og allt',
        )
        if _easy_command_text(value) not in {_easy_command_text(phrase) for phrase in phrases}:
            return ''
        return tr('42 — the answer to life, the universe and everything. — The Hitchhiker’s Guide to the Galaxy')

    def _ask_navigator_ai(self, value):
        """Stream one unmatched submission through the existing guarded transport."""
        from ..ai.providers import get_provider
        from ..ai.worker import make_stream_thread
        from ..i18n import tr
        from ..preferences import get_preferred_provider
        from .ai_chat_panel import _current_system_prompt

        if get_preferred_provider() == 'none':
            self._cancel_navigator_ai()
            self.path_edit.ai.setChecked(False)
            self._feedback(tr('No provider configured.'))
            return
        if self._ai_thread is not None:
            self.field.details.setPlainText(tr('Turn AI off to cancel the current answer.'))
            return
        provider = get_provider(self._ai_provider)
        if provider is None or not provider.is_configured():
            self._feedback(tr('No provider configured.'))
            return
        self._runner.cancel()
        self._input_generation += 1
        self._requested = value
        self._last_result = None
        self._settings_overrides = {}
        self.field.set_cards([])
        self._feedback(tr('Thinking…'))
        self._ai_text = ''
        self._ai_generation += 1
        generation = self._ai_generation
        self.field.details._start_reply(generation)
        alive = []
        for thread, worker, relay in self._ai_retired:
            try:
                if thread.isRunning():
                    alive.append((thread, worker, relay))
                    continue
            except RuntimeError:
                pass
            relay.deleteLater()
        self._ai_retired = alive
        system = _current_system_prompt() + (
            '\n\nNavigator is an advice interface. Answer the submitted task in text. '
            'Do not invoke tools, run commands or analyses, or change files from this interface.')
        thread, worker = make_stream_thread(provider, [{'role': 'user', 'content': value}],
                                            system=system, parent=self)
        self._ai_thread, self._ai_worker = thread, worker
        self._ai_relay = _NavigatorAiReply(self, generation)
        worker.chunk_ready.connect(self._ai_relay.chunk, Qt.QueuedConnection)
        worker.finished.connect(self._ai_relay.finished, Qt.QueuedConnection)
        self.path_edit.ai._busy = True
        self.path_edit.ai._tick()
        thread.start()

    def _navigator_ai_chunk(self, generation, text):
        """Ignore retired streams and coalesce partial replies into bounded repaints."""
        from ..preferences import get_preferred_provider

        if generation != self._ai_generation:
            return
        if get_preferred_provider() == 'none':
            self._cancel_navigator_ai()
            self.path_edit.ai.setChecked(False)
            return
        self._ai_text += text
        if not self._ai_reply_timer.isActive():
            self._ai_reply_timer.start()

    def _render_ai_reply(self):
        """Present a readable reply using the same passive Markdown renderer as chat."""
        from ..preferences import get_preferred_provider

        if get_preferred_provider() == 'none':
            self._cancel_navigator_ai()
            self.path_edit.ai.setChecked(False)
            return
        from .ai_chat_panel import _ai_reply_html

        self.field.details.setHtml(_ai_reply_html(self._ai_text), stream=self._ai_generation)

    def _navigator_ai_finished(self, generation, ok, text):
        """Stop the waiting texture and preserve worker ownership until thread exit."""
        from ..i18n import tr
        from ..preferences import get_preferred_provider

        if generation != self._ai_generation:
            return
        if get_preferred_provider() == 'none':
            self._cancel_navigator_ai()
            self.path_edit.ai.setChecked(False)
            return
        self._ai_reply_timer.stop()
        self._ai_retired.append((self._ai_thread, self._ai_worker, self._ai_relay))
        self._ai_thread = self._ai_worker = self._ai_relay = None
        self.path_edit.ai._busy = False
        self.path_edit.ai._tick()
        self._ai_text = (text or tr('(empty response — try again or switch provider)')) if ok else tr(
            'Failed: {detail}', detail=text)
        self._render_ai_reply()

    def _cancel_navigator_ai(self):
        """Invalidate queued replies and retire threads without blocking navigation."""
        from ..bridge import drain_thread

        self._ai_generation += 1
        self._ai_reply_timer.stop()
        worker, thread, relay = self._ai_worker, self._ai_thread, self._ai_relay
        self._ai_worker = self._ai_thread = self._ai_relay = None
        self.path_edit.ai._busy = False
        self.path_edit.ai._tick()
        if worker is not None:
            worker.cancel()
        pairs = list(self._ai_retired)
        self._ai_retired.clear()
        if thread is not None:
            pairs.append((thread, worker, relay))
        for retired_thread, retired_worker, retired_relay in pairs:
            try:
                retired_thread.setParent(None)
            except RuntimeError:
                pass
            drain_thread(retired_thread, retired_worker, timeout_ms=0)
            retired_relay.deleteLater()

    def hideEvent(self, event):
        """Restore ordinary modules without changing their saved theme."""
        self._cancel_navigator_ai()
        self._cancel_regex_validation()
        super().hideEvent(event)

    def refresh(self):
        """Keep existing dark field colors and apply Open Sans Light only to this Home."""
        from ..preferences import get_pane_opacity, resolve_effective_theme
        from ..theme import css_color, panel_alpha

        palette = palette_for('dark')
        background = 'rgb(25, 25, 25)'
        foreground = palette['fg']
        muted = palette['fg_muted']
        border = palette['border']
        details_background = css_color('#191919', panel_alpha(
            resolve_effective_theme(), 'surface_alt', get_pane_opacity()))
        sheet = (
            f'QWidget#EasyHomePage, QWidget#EasyHomePage QWidget {{ font-family: "Open Sans"; font-weight: 300; }} '
            f'QLineEdit#EasyModePath {{ background: {background}; color: {foreground}; border: none; border-radius: 28px; font-size: {font_px(18)}px; padding: 0px; placeholder-text-color: {muted}; }} '
            f'QPushButton#EasyModeBrowse {{ color: {foreground}; background: transparent; border: none; padding: 0px; font-size: {font_px(14)}px; }} '
            f'QPushButton#EasySuggestionCard {{ background: {background}; color: {foreground}; border: 1px solid {border}; border-radius: 16px; padding: 12px; min-width: 0px; min-height: 0px; text-align: left; }} '
            f'QPushButton#EasySuggestionCard QLabel {{ color: {foreground}; background: transparent; border: none; font-weight: 300; }} '
            f'QPlainTextEdit#EasyRegexPattern {{ color: {foreground}; background: transparent; border: 1px solid {border}; border-radius: 6px; font-weight: 300; }} '
            f'QPushButton#EasyWorkflowBack {{ color: {foreground}; background: transparent; border: none; padding: 0px; }} '
            f'QComboBox#EasyWorkflowTools, QComboBox#EasyWorkflowNextTools {{ color: {foreground}; background: {background}; placeholder-text-color: {muted}; border: 1px solid {border}; border-radius: 10px; }} '
            f'QComboBox#EasyWorkflowTools QAbstractItemView, QComboBox#EasyWorkflowNextTools QAbstractItemView {{ color: {foreground}; background: {palette["surface"]}; selection-background-color: {palette["accent"]}; selection-color: {palette["bg"]}; }} '
            f'QTextBrowser#EasyWorkflowDetails {{ color: {foreground}; background: {details_background}; border: 1px solid {border}; border-radius: 10px; padding: 8px; font-size: {2 * font_px("body")}px; }} '
            f'QLabel#EasyInputFeedback {{ color: {muted}; background: transparent; }}')
        from ..theme import _stage_button_rules
        sheet += _stage_button_rules('QPushButton#EasySuggestionCard')
        sheet += ('QPushButton#EasySuggestionCard[spaceout="true"], '
                  'QPushButton#EasySuggestionCard[spaceout="true"]:hover, '
                  'QPushButton#EasySuggestionCard[spaceout="true"]:pressed '
                  '{ background-color: rgb(32, 32, 32); }')
        if getattr(self, '_easy_stylesheet', None) != sheet:
            self._easy_stylesheet = sheet
            self.setStyleSheet(sheet)

    def _feedback(self, message, visible=False):
        """Keep normal guidance accessible without adding an explanatory panel."""
        self.status.setText(message)
        self.status.hide()
        self.path_edit.setAccessibleDescription(message)
        self.path_edit.setToolTip(message)
        if visible:
            self.field.set_cards([], _EasyGuidanceCard(message, self.field))
        self.field.details.setPlainText(message)
        self._details_has_content = False
        if not message:
            self._show_empty_guide()
        self.field._arrange()
        self.field._publish_growth(invalidate=True)

    def accept_paths(self, paths):
        """Accept exactly one local input and give useful feedback for ambiguous drops."""
        from ..i18n import tr

        self._show_suggestions()
        if len(paths) != 1:
            self._cancel_regex_validation()
            self._input_generation += 1
            self._last_result = None
            self._settings_overrides = {}
            self._debounce.stop()
            self._requested = self.path_edit.text()
            self.field.set_cards([])
            self._feedback(tr('Drop one local file or folder at a time.'), visible=True)
            return
        self.path_edit.setText(paths[0])
        self._inspect()

    def _browse(self):
        """Offer both native file and folder selection from the inset Browse text."""
        from PySide6.QtWidgets import QFileDialog, QMenu

        from ..i18n import tr

        menu = QMenu(self)
        file_action = menu.addAction(tr('Choose a file'))
        folder_action = menu.addAction(tr('Choose a folder'))
        selected = menu.exec(self.path_edit.browse.mapToGlobal(self.path_edit.browse.rect().bottomLeft()))
        if selected is file_action:
            path, _ = QFileDialog.getOpenFileName(self, tr('Choose a file'))
        elif selected is folder_action:
            path = QFileDialog.getExistingDirectory(self, tr('Choose a folder'))
        else:
            return
        if path:
            self.accept_paths([path])

    def _reset_for_home(self):
        """Return explicit Home navigation to a fresh input with no retained graph or job."""
        self._cancel_navigator_ai()
        self._show_suggestions()
        self._details_has_content = False
        self.field.details._reset_history()
        self.path_edit.clear()
        self._queued_inspection = False
        self._clear_empty_input()

    def _clear_empty_input(self):
        """Return an empty editor to the live field without keeping stale work."""
        self._cancel_navigator_ai()
        self._debounce.stop()
        self._cancel_regex_validation()
        self._input_generation += 1
        self._runner.cancel()
        self._requested = self.path_edit.text()
        self._last_result = None
        self._settings_overrides = {}
        self.field.set_cards([])
        if not self._details_has_content:
            self._feedback('')
        self._show_suggestions()

    def _schedule_inspection(self, text):
        """Invalidate unfinished work while leaving submitted choices unchanged until Enter."""
        if self._ai_thread is not None:
            self._cancel_navigator_ai()
        self._debounce.stop()
        self._queued_inspection = False
        self._cancel_regex_validation()
        self._input_generation += 1
        self._runner.cancel()
        if not text.strip():
            self._clear_empty_input()

    def _submit(self):
        """Submit through explicit commands, local suggestions or opted-in AI."""
        from ..app import (
            _easy_command_text,
            _easy_module_matches,
            _easy_module_search,
            _easy_shortcut_match,
            _easy_task_keys,
        )
        from ..organisms import ORGANISMS

        value = self.path_edit.text().strip()
        self._show_suggestions()
        if self._navigator_local_reply(value):
            self._inspect()
            return
        shortcut = _easy_shortcut_match(value)
        if shortcut and shortcut not in ('spaceout', 'running'):
            self._cancel_navigator_ai()
            self._debounce.stop()
            self._cancel_regex_validation()
            self._runner.cancel()
            self._input_generation += 1
            self.command_requested.emit(value)
            return
        organism = any(key in ORGANISMS for key in _easy_module_matches(value))
        if (organism or _easy_task_keys(value) or not _easy_command_text(value)
                or shortcut in ('spaceout', 'running') or value.endswith(':')
                or _easy_module_matches(value) or _easy_module_search(value)):
            self._inspect()
            return
        suggest = getattr(self.window(), '_easy_navigation_suggestions', _easy_module_search)
        if self._navigator_ai_can_answer(value) and not suggest(value):
            self._inspect()
            return
        self._debounce.stop()
        self._cancel_regex_validation()
        self._runner.cancel()
        self._input_generation += 1
        self.command_requested.emit(value)

    def _inspect(self):
        """Inspect local inputs or send an explicitly enabled unmatched task to AI."""
        from functools import partial

        from ..i18n import tr

        if not self.path_edit.text().strip():
            self._clear_empty_input()
            return
        self._debounce.stop()
        self._cancel_regex_validation()
        value = self.path_edit.text()
        from ..app import _easy_module_search

        suggest = getattr(self.window(), '_easy_navigation_suggestions', _easy_module_search)
        actions = suggest(value)
        if actions:
            self._cancel_navigator_ai()
            self._runner.cancel()
            self._requested = value
            self._input_generation += 1
            self._last_result = None
            self._settings_overrides = {}
            groups = {key: dict(group=tr('Help') if key.startswith('__easy_command__:') else tr('Tools'),
                                reason=description, kind='navigation')
                      for key, _title, description, _path in actions}
            self._show_result(self._input_generation, dict(path=value, actions=actions, message='',
                                                         context=dict(action_groups=groups)))
            return
        local_reply = self._navigator_local_reply(value)
        if local_reply or (not self.path_edit.ai.isChecked()
                           and self._navigator_ai_can_answer(value)
                           and not _easy_external_reference(value)):
            self._cancel_navigator_ai()
            self._runner.cancel()
            self._input_generation += 1
            self._requested = value
            self._last_result = None
            self._settings_overrides = {}
            self.field.set_cards([])
            self._feedback(local_reply or tr(
                'No matching tool found.\n\nRight-click ◯ to choose spaCR AI (free, hosted by Cloudflare), Claude, Codex or Gemini. Left-click ◯ to turn AI on, then ask your question.\n\nTry a task such as “segment images”, a module name such as “Measure”, or a file type such as “tif”, then press Enter. You can also drop a file/folder or paste a path.'))
            return
        if self.path_edit.ai.isChecked() and self._navigator_ai_can_answer(value):
            self._ask_navigator_ai(value)
            return
        self._cancel_navigator_ai()
        if self._runner.is_busy() or self._runner.active_jobs():
            self._queued_inspection = value != self._requested or not self._runner.is_busy()
            if value != self._requested:
                self._last_result = None
                self._settings_overrides = {}
                self.field.set_cards([])
            if self._queued_inspection:
                self._debounce.start()
            return
        self._requested = value
        self._input_generation += 1
        self._last_result = None
        self._settings_overrides = {}
        self.field.set_cards([])
        self._feedback(tr('Inspecting local file headers…'))
        self._runner.submit(partial(_easy_inspect_path, value),
                            on_done=partial(self._show_result, self._input_generation))

    def _show_result(self, generation, result):
        """Publish only the latest inspected path and wire explicit next-step actions."""
        if generation != self._input_generation or self.path_edit.text() != self._requested:
            return
        self._regex_cancel = None
        from ..app import app_is_visible
        from ..i18n import tr

        actions = [action for action in result['actions'] if app_is_visible(action[0])]
        visible_count = len(actions)
        self._last_result = result
        self._settings_overrides = dict(result.get('settings_overrides', {}))
        proposal = result.get('regex')
        leading = None
        if proposal:
            leading = _EasyRegexCard(proposal, self.field)
            leading.regex_requested.connect(self._apply_regex)
        elif result.get('metadata_preset'):
            actions = [(key, title,
                        description + ' ' + tr('Filename preset: {preset}', preset=result['metadata_preset'])
                        if key == 'mask' else description, path)
                       for key, title, description, path in actions]
        message = result['message']
        if visible_count < len(result['actions']):
            message += ' ' + tr('Additional matching modules are hidden by your maturity or alpha preferences.')
        self._feedback(message, visible=not actions and leading is None)
        if not actions and leading is None:
            return
        leading_index = int(bool(actions and actions[0][0] in ('foreign', 'db_browser', 'measure')
                                 and actions[0][3] == result['path']))
        self.field.set_cards(actions, leading, leading_index,
                             groups=result.get('context', {}).get('action_groups'))
        if self.field.cards:
            self.field.explain_card(self.field.cards[0])
        for card in self.field.cards:
            if not isinstance(card, _EasyRegexCard):
                card.activated.connect(self._select_suggestion)

    def _select_suggestion(self, key, path):
        """Center a suggested tool and reveal its real workflow before opening it."""
        if key.startswith('__easy_command__:'):
            from ..app import _easy_shortcut_match

            command = key.split(':', 1)[1]
            self.command_requested.emit(key if _easy_shortcut_match(command, explicit=True) else command)
            return
        if key == 'parascape':
            from ..app import app_is_visible
            from ..preferences import maturity_is_visible

            if maturity_is_visible('alpha') and app_is_visible(key):
                self.module_requested.emit(key, path)
            return
        from ..app import APPS, _HELP_MODULES, app_is_visible, folded_children
        from ..preferences import maturity_is_visible
        from .fold_strip import fold_label
        from .workflow_diagram import _easy_workflow_data
        from ..i18n import tr

        folds = folded_children()
        all_keys = {row[0] for row in APPS} | {child for children in folds.values() for child in children}
        labels = {candidate: fold_label(candidate) for children in folds.values() for candidate in children}
        visible = {candidate for candidate in all_keys if app_is_visible(candidate)
                   and (candidate not in labels or maturity_is_visible(labels[candidate][2]))}
        metadata = {candidate: values[:2] for candidate, values in labels.items()}
        data = _easy_workflow_data(APPS, visible, folds, folded_metadata=metadata,
                                  subtools=(dict(key='overview', name=tr('Overview'),
                                                 guidance=tr('Inspect images and field metadata.'),
                                                 host_key='mask', route='overview'),
                                            dict(key='3d', name=tr('3D'),
                                                 guidance=tr('Inspect images and field metadata.'),
                                                 host_key='mask', route='dimension')),
                                  mode_labels={('mask', 'measure'): (tr('Concatenate'), tr('3D'), tr('Time')),
                                               ('mask', 'ops'): (tr('OPS'),),
                                               ('mask', 'timelapse'): (tr('Time'),)})
        data['easy_helpers'] = tuple(candidate for candidate in
                                     (*[row[0] for row in _HELP_MODULES], 'feature_dict')
                                     if candidate in data['modules'])
        if key not in data['modules']:
            return
        actions = list((self._last_result or {}).get('actions', ()))
        actions = [action for action in actions if action[0] != key]
        actions.insert(0, (key, data['modules'][key]['name'], data['modules'][key].get('guidance', ''), path))
        source_card = next((card for card in self.field.cards
                            if card._key == key and card._path == path), None)
        if source_card is None:
            return
        departure = {card: (card.isVisible(), card._content_opacity.opacity())
                     for card in self.field.cards if card is not source_card}
        self.field._finish_growth()
        self.field._suggestions_active = False
        for card, (visible, opacity) in departure.items():
            card._content_opacity.setOpacity(opacity)
            card.setVisible(visible and opacity > 0.0)
        anchor = QRect(source_card.mapTo(self, source_card.rect().topLeft()), source_card.size())
        source_card.activated.disconnect(self._select_suggestion)
        self._workflow = _EasyWorkflowPage(data, actions, key, source_card, anchor, self)
        self._workflow.setGeometry(self.rect())
        self._workflow.back_requested.connect(self._show_suggestions)
        self._workflow.open_requested.connect(self.module_requested)
        self._workflow.show()
        self._workflow.raise_()
        self.field.details.raise_()

    def _show_suggestions(self):
        """Return to the unchanged path and suggestions without starting another inspection."""
        self.field._suggestions_active = True
        if self._workflow is not None:
            source_card = self._workflow.release_source_card()
            source_card.activated.connect(self._select_suggestion)
            self._workflow.hide()
            self._workflow.deleteLater()
            self._workflow = None
            self.field._arrange()
        self._suggestions_scroll.show()
        self.path_edit.show()
        self.field.rim.show()
        self.path_edit.setFocus()

    def _apply_regex(self, pattern):
        """Validate a chosen pattern off the GUI thread before storing any seed."""
        from functools import partial
        from threading import Event
        from ..i18n import tr

        if self._last_result is None or self._runner.is_busy():
            return
        self._input_generation += 1
        self._regex_cancel = Event()
        self._feedback(tr('Validating regex…'))
        for card in self.field.cards:
            card.setEnabled(False)
        self._runner.submit(partial(_easy_validate_regex, self._last_result, pattern,
                                    cancelled=self._regex_cancel.is_set),
                            on_done=partial(self._show_result, self._input_generation))

    def _cancel_regex_validation(self):
        """Retire the owned validation child before abandoning its completion."""
        if self._regex_cancel is not None:
            self._regex_cancel.set()
            self._regex_cancel = None
            self._input_generation += 1
            self._runner.cancel()
            for card in self.field.cards:
                card.setEnabled(True)

    def _inspection_finished(self, ok):
        """Inspect a changed input after the previous owned worker has retired."""
        if self._queued_inspection and self.path_edit.text().strip():
            self._queued_inspection = False
            self._debounce.start()

    def _inspection_failed(self, message):
        """Present a real worker failure without leaving stale actions available."""
        from ..i18n import tr

        if not self.path_edit.text().strip():
            return
        self.field.set_cards([])
        self._feedback(tr('This input could not be inspected. Check permissions and choose another path.'), visible=True)

    def resizeEvent(self, event):
        """Keep the transparent network over the same live field when resizing."""
        super().resizeEvent(event)
        if getattr(self, '_workflow', None) is not None:
            self._workflow.setGeometry(self.rect())

    def closeEvent(self, event):
        """Stop owned timers and retire the local inspection worker before closing."""
        self._debounce.stop()
        self._cancel_navigator_ai()
        self._cancel_regex_validation()
        self._show_suggestions()
        self.field._finish_growth()
        self._runner.shutdown(3000)
        super().closeEvent(event)
