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

from PySide6.QtCore import QEvent, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..hidpi import follow_device_ratio, scaled_for
from ..theme import (
    SPACING, TILE_H, TILE_ICON_PX, TILE_MAX_W, TILE_W, font_px,
    make_transparent, palette_for,
)
from .divider import Divider
from .height_grip import HeightGrip

#: Hero brand sizes. The mark and wordmark are the first thing on the first
#: screen, so they are sized to read as a masthead rather than as a row of
#: labels. Kept as named constants because the logo's pixmap scale and the
#: label's font size have to move together to stay optically balanced.
HERO_LOGO_PX = 72
HERO_TITLE_PX = 52


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

    **Its height floor is in the QSS, not here, and that is not a
    style choice.** ``setFixedSize`` does not survive polish, and
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
        :param description: accepted and NOT DRAWN. The tile stopped showing
            it because it was a third copy of a sentence already on the
            tooltip and in the hint bar, and three lines of grey under every
            tile is what forced two tile sizes in the first place. Kept in
            the signature so callers that pass it still work.
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
            glyph = QLabel()
            glyph.setPixmap(icon.pixmap(icon_px, icon_px))
            glyph.setFixedSize(icon_px, icon_px)
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
        """At least the tile height, more if a child somehow needs it."""
        natural = super().heightForWidth(width)
        return max(self._size.height(), natural)

    def sizeHint(self) -> QSize:               # noqa: N802
        """The tile's preferred size, height derived from its width.

        HEIGHT FOLLOWS WIDTH because the tile is icon-over-name in a fixed
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
            "font-family: 'Open Sans', sans-serif; font-weight: 600;"
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
        from ..theme import pane_surface
        box.setStyleSheet(
            "QFrame#HomePanelBox {"
            f"background: {pane_surface('surface_alt')};"
            f"border: 1px solid {P['border_soft']};"
            "border-radius: 8px; }")
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
        """Put an action WORD on the right of the panel's caption.

        Not a button in the styled sense -- no frame, no fill, no padding
        that would make it look pressable. It is the word alone, in the
        muted caption ink, and it takes its colour when the pointer is on
        it or while it is held down: "just the text clear which turns red
        upon hover or click" (2026-09-03).

        Rendered through a QPushButton rather than a QLabel because the
        word is a CONTROL: a button is what Tab reaches, what Space
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
            " font-family: 'Open Sans', sans-serif; font-weight: 600;"
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
        """The action word named ``text``, or ``None``. For tests."""
        return self._actions.get(text.lower())

    def _clear_body(self) -> None:
        """Take every row out of the panel's box and delete it.

        The same six lines were written out in five panels; they are here
        because a panel that forgets the `deleteLater` leaks a widget on
        every Home revisit, and Home is revisited constantly.
        """
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
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
    """"spaCR is doing something right now" — with honest controls.

    Reads :func:`spacr.qt.bridge.registry`, so it reflects a job started
    from *any* screen. Hidden entirely when nothing is running, which is
    the common case and should cost the page nothing.

    **On the Pause button.** It is enabled if and only if the running
    job's entry point declares itself ``bridge.pausable`` — i.e. it
    actually polls ``bridge.checkpoint()``. No shipped pipeline does, so
    today it renders disabled with :data:`PAUSE_UNAVAILABLE` as its
    tooltip. That is the whole point: a Pause button that stops the
    thread wherever it happens to be is not a pause, and the honest
    thing to draw is a control that says so.

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
        """Show ``handle``'s job, or hide the banner when it is ``None``."""
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

        WRITES, which no other part of this panel does -- the class docstring
        says the Queue screen owns writes, and this is the one exception:
        **Clear** empties the queue from here rather than sending the reader
        to another screen to do it.

        Not confirmed, and deliberately. A queue entry is a plate waiting to
        be processed -- settings and a source path, no results -- so clearing
        one throws away a few seconds of setting up, not any data. A
        confirmation on a two-word action that costs that little is a
        dialog people learn to dismiss without reading.

        Every failure mode ends with the panel telling the truth about what
        is in the queue, because it re-reads from disk afterwards either way.
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
    """Last few journalled runs of a REAL module; each row navigates.

    "Real module" is doing work here. The panel used to list whatever the
    run journal's newest manifests said, and those rows were clickable
    without representing runs -- they opened a ``_job`` module. On one
    machine the journal held 11,046 run folders of which 7,323 were written
    under the app_key ``_job`` or ``job``: a test fixture's key, written
    straight into the real `~/.spacr/runs` because nothing sandboxed it
    (fixed in ``tests/conftest.py``). So most rows named a module that does
    not exist, and clicking one asked the window to open it.

    The pollution is stopped at the source now, but a filter here is still
    the right thing: this panel NAVIGATES, so a row it draws is a promise
    that pressing it goes somewhere. Anything whose key is not a module Home
    knows about is dropped.

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

    def __init__(self, limit: int = 4, known_keys=None, parent=None):
        """Build the panel and its caption.

        :param parent: parent widget.
        """
        super().__init__("Recent runs", parent)
        self._limit = limit
        self._known_keys = known_keys
        self._clear = self.add_action(
            "Clear", kind="danger",
            tip="Hide the runs listed here. The run journal and Run "
                "History keep them.")
        self._clear.clicked.connect(self.clear_list)
        self.refresh()

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

        A row used to be captioned with the raw app_key, which is what put
        ``_job`` and ``mask`` on the dashboard in the same typeface as each
        other. The names are already in Home; the panel just had no way to
        ask for them.
        """
        value = self._registry()
        if isinstance(value, dict):
            return str(value.get(key, key) or key)
        return key

    def clear_list(self) -> None:
        """Hide every run listed, by moving the watermark to now.

        NOTHING IS DELETED. See
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

        THE FILTERING HAPPENS HERE, not in `refresh`, for two reasons. It is
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
    row reads as "broken", while ``n/a`` honestly says the lightweight
    system probe could not measure a device.  Home must not import a model
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

        NEVER RAISES. A missing NVML, a machine with no GPU and a driver
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

    def __init__(self, parent=None):
        """Build the panel and its caption.

        :param parent: parent widget.
        """
        super().__init__("Totals", parent)
        self._reset = self.add_action(
            "Reset", kind="danger",
            tip="Start these counts from now. The run journal and Run "
                "History keep every run.")
        self._reset.clicked.connect(self.reset_counts)
        self.refresh()

    def reset_counts(self) -> None:
        """Count from now on, by moving the watermark.

        NOTHING IS DELETED, for the same reason **Clear** on Recent runs
        deletes nothing — see
        :func:`spacr.qt.preferences.get_dashboard_watermark`. What these
        counts are FOR is telling the user how much this installation has
        done, and a reset that removed the manifests would take Run History
        and every run's provenance with it.
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

        ISO-8601 UTC strings compare lexicographically in time order, and
        ``recent_runs`` hands them back newest-first, so the walk stops at
        the first entry at or before the watermark.
        """
        from spacr.run_journal import recent_runs

        counts = {"total_runs": 0, "mask_runs": 0, "measure_runs": 0,
                  "classify_runs": 0, "models_recorded": 0}
        for entry in recent_runs(limit=None):
            if str(entry.get("start_utc") or "") <= since:
                break
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
        """
        from ..theme import stage_hover
        return stage_hover(stage)

    def row_for(self, stage: str) -> Optional[QWidget]:
        """The legend row explaining one maturity stage.

        :param stage: the stage's name.
        :returns: the row widget, or None when that stage has no row.
        """
        return self._rows.get(stage)


class NewsPanel(Panel):
    """Every spaCR release, with links, in a box the reader can resize.

    THE NOTES ARE BUNDLED, and the bundle is what draws. They come from
    ``spacr/resources/release_notes.json``, which
    ``tools/build_release_notes.py`` writes from the GitHub releases and
    which ``.github/workflows/release.yml`` refreshes on every release.
    This panel is on the first screen the application shows, so making its
    CONTENT depend on api.github.com would mean a dashboard that is empty
    offline, throttled behind a shared NAT, and slower to draw than the
    window it is in. The bundled file therefore remains the offline source
    of truth and the panel is complete before anything touches a socket.

    AND THEN IT CATCHES UP. The wheel for a release cannot contain its own
    release note -- the note is written when the GitHub release is
    published, which is after that wheel is on PyPI -- so a bundled file is
    always one release behind the build carrying it, and that is what went
    wrong: "im on 1.5.1.0 and the news only goes to 1.5.0.7. the news
    section should always automatically reflect the latest spacr release
    news." So after the page is shown,
    :attr:`refresh_requested` asks the window to read the public releases
    list on a worker thread, and :meth:`apply_releases` merges whatever
    comes back in front of the bundled list. Nothing here opens a socket:
    the panel only asks, and a fetch that fails, is rate-limited, is
    switched off in Preferences, or simply finds nothing newer leaves the
    bundled list exactly as it was drawn.

    There was previously no feed at all and this panel said so -- "No
    release notes bundled with this build" -- which was honest and useless.
    What it shows now is the real thing: "News should contain all the
    releas information with links. i nkow that there was release information
    for the current 1.5.0.4 version. this one as well as all of the other
    ones should be scrollable and the user should be able to controll the
    height."

    So: every release newest-first, each one's body rendered with its links
    live, the whole list inside a scroll area, and a grip along the bottom
    edge that drags the box taller or shorter. The height is remembered
    between sessions -- a reader who made it tall wants it tall next time.

    The update check stays a BUTTON. Offering to install something is a
    decision, so it is still made only when pressed.

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
        self._notes.setWidgetResizable(True)
        self._notes.setFrameShape(QFrame.NoFrame)
        self._notes.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
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

        self._placeholder = QLabel(
            "No release notes bundled with this build.")
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(
            f"color: {P['fg_muted']}; font-size: {font_px(11)}px;"
            "font-style: italic; background: transparent;")
        self._fill()

        self._grip = _HeightGrip(self._notes, self.NOTES_H_MIN,
                                 self.NOTES_H_MAX,
                                 name="Resize the release notes")
        self._grip.height_changed.connect(self._remember_height)
        self.body_layout.addWidget(self._grip)
        self._notes.setFixedHeight(self._stored_height())

        check = QPushButton("Check for updates\u2026")
        check.setObjectName("GhostButton")
        check.setCursor(Qt.PointingHandCursor)
        check.clicked.connect(self.check_requested)
        self.add(check)
        self._check = check

    @staticmethod
    def read_releases() -> list:
        """The bundled release records, newest first, or ``[]``.

        Degrades to an empty list on ANY failure, and the panel then draws
        the placeholder it drew before there was a feed. A dashboard must
        not fail to appear because a resource file is missing from a wheel.
        """
        try:
            import json
            from importlib.resources import files

            raw = (files("spacr.resources") / "release_notes.json")
            data = json.loads(raw.read_text(encoding="utf-8"))
            releases = data.get("releases") or []
            return [r for r in releases if isinstance(r, dict)]
        except Exception:                                        # noqa: BLE001
            return []

    def _fill(self) -> None:
        """Draw :attr:`_releases` into the scrolling column.

        Called once while the panel is built and again whenever a fetched
        list arrives, so the two paths cannot diverge. The placeholder is
        kept rather than rebuilt: it is what a build with no bundled
        resource shows, and :meth:`set_content` holds a reference to it.
        """
        while self._notes_column.count():
            item = self._notes_column.takeAt(0)
            widget = item.widget()
            if widget is not None and widget is not self._placeholder:
                widget.setParent(None)
                widget.deleteLater()
        self._notes_column.addWidget(self._placeholder)
        self._placeholder.setVisible(not self._releases
                                     and self.content is None)
        for entry in self._releases:
            self._notes_column.addWidget(self._release_block(entry))
        self._notes_column.addStretch(1)

    def showEvent(self, event):                                  # noqa: N802
        """Ask for a refresh the first time the panel is shown.

        AFTER the page exists and ON THE EVENT LOOP, not during
        construction: the single-shot timer means the emit lands on a later
        turn than this show, so Home's first paint is never waiting on it.
        Once per panel, because a page that is shown again -- a tab
        revisited, a font-scale rebuild -- is not news.

        :param event: the Qt show event.
        """
        super().showEvent(event)
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

        Silence is the contract. An empty list, a list of rubbish, or a
        list that says nothing the bundled file did not already say leaves
        the panel untouched and says nothing to the reader -- the failure
        of an unasked-for background fetch is not the reader's problem.

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
        title.setWordWrap(True)
        title.setStyleSheet(
            f"color: {P['fg']}; font-size: {font_px(12)}px;"
            "font-weight: 600; background: transparent;")
        column.addWidget(title)
        if when:
            stamp = QLabel(when)
            stamp.setStyleSheet(
                f"color: {P['fg_muted']}; font-size: {font_px(10)}px;"
                "background: transparent;")
            column.addWidget(stamp)

        body = self.render_body(str(entry.get("body") or ""), P["accent"])
        if body:
            notes = QLabel(body)
            notes.setWordWrap(True)
            notes.setOpenExternalLinks(True)
            notes.setTextInteractionFlags(Qt.TextBrowserInteraction)
            notes.setStyleSheet(
                f"color: {P['fg_muted']}; font-size: {font_px(11)}px;"
                "background: transparent;")
            column.addWidget(notes)
        return block

    @staticmethod
    def render_body(body: str, link_colour: str) -> str:
        """Turn a release body into the small HTML subset a QLabel draws.

        NOT A MARKDOWN RENDERER, and not trying to be. Release bodies are
        GitHub-flavoured markdown; what they actually contain is bullet
        lists, bare URLs and ``**bold**``, and a QLabel understands a
        handful of tags. So: escape everything first -- a body is text from
        a web page and must never reach a rich-text widget as markup --
        then put back the three constructs that are worth having.

        Escaping FIRST is what makes this safe. Linkifying first and
        escaping after would escape the anchors too and show the reader
        their own tags; escaping after building the HTML is the mistake that
        turns a release note into an injection.
        """
        text = escape(body or "").strip()
        if not text:
            return ""
        text = re.sub(
            r"(https?://[^\s<>\"']+?)([.,;:]?)(?=\s|$)",
            lambda m: (f'<a href="{m.group(1)}" style="color: {link_colour};'
                       f' text-decoration: none;">{m.group(1)}</a>'
                       f"{m.group(2)}"),
            text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                lines.append(f"<b>{stripped[3:]}</b>")
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

        split = QHBoxLayout()
        split.setContentsMargins(0, 0, 0, 0)
        split.setSpacing(SPACING["lg"])
        split.addWidget(self._build_tabs(), 1)
        split.addWidget(self._build_aside())
        col.addLayout(split, 1)

        outer.addWidget(body, 1)

        from .module_hint_bar import ModuleHintBar
        self._hint_bar = ModuleHintBar(_DEFAULT_HINT)
        outer.addWidget(self._hint_bar)

        from .. import bridge
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
        if (self._ambient is not None
                or getattr(self, "_uses_window_backdrop", False)):
            return None
        try:
            from ..preferences import resolve_effective_theme
            from ..theme import IMAGE_THEMES, page_colour
            theme = resolve_effective_theme()
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
        """
        colour = self.page_fill()
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
            from ..preferences import resolve_effective_theme
            from ..theme import RADIUS, pane_surface

            hero.setStyleSheet(
                f"QWidget#Hero {{"
                f" background-color:"
                f" {pane_surface('surface_alt', resolve_effective_theme())};"
                f" border-radius: {RADIUS['lg']}px;"
                f" }}")
        except Exception:                                # noqa: BLE001
            import logging

            logging.getLogger(__name__).debug(
                "the masthead would not take its surface", exc_info=True)
        row = QHBoxLayout(hero)
        row.setContentsMargins(SPACING["md"], SPACING["sm"],
                               SPACING["md"], SPACING["sm"])
        row.setSpacing(SPACING["md"])

        logo_px = font_px(HERO_LOGO_PX)
        logo = _find_logo_pixmap()
        if logo is not None:
            label = QLabel()
            label.setObjectName("HeroMark")
            label.setFixedSize(logo_px, logo_px)
            label.setStyleSheet("background: transparent;")

            def _draw_mark(label=label, source=logo, side=logo_px):
                """Redraw the mark at the ratio of the screen it is on.

                Always from the 3334 px master. Re-scaling whatever is
                already on the label would compound one resample for every
                screen the window has been dragged across.
                """
                label.setPixmap(scaled_for(source, label, side))

            _draw_mark()
            follow_device_ratio(label, _draw_mark)
            self._hero_mark = label
            row.addWidget(label)

        title = QLabel("spaCR")
        title.setStyleSheet(
            "font-family: 'Open Sans', sans-serif; font-weight: 300;"
            f"font-size: {font_px(HERO_TITLE_PX)}px; color: {P['accent']};"
            "letter-spacing: -0.6px; background: transparent;")
        row.addWidget(title)

        from .loading_screen import strap_line
        subtitle = QLabel(strap_line())
        subtitle.setObjectName("HeroSubtitle")
        subtitle.setWordWrap(True)
        row.addWidget(subtitle, 1)

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
            self._fill_grid(grid, tiles,
                            self._columns_for(self.width(), width))
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
            "font-family: 'Open Sans', sans-serif; font-weight: 600;"
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
        heading = QLabel(section.upper())
        heading.setStyleSheet(
            "font-family: 'Open Sans', sans-serif; font-weight: 600;"
            f"font-size: {font_px(11)}px; letter-spacing: 2px;"
            "background: transparent;"
            f"color: {P['fg_muted']};")
        head_col.addWidget(heading)

        note = self._section_notes.get(section)
        if note:
            caption = QLabel(note)
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
        self._fill_grid(grid, tiles, self._columns_for(self.width(), width))
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
        """Connect one tile so pressing it opens its module.

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
        tile.installEventFilter(self)
        tile.clicked.connect(lambda _=False, k=key: self.tile_clicked.emit(k))
        return tile

    def _scrolled(self, page: QWidget) -> QScrollArea:
        """Wrap a page in a scroll area.

        :param page: the widget to wrap.
        :returns: the scroll area.
        """
        scroll = QScrollArea()
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

    def _columns_for(self, width: int, tile_w: int) -> int:
        """How many ``tile_w``-wide tiles fit beside the aside.

        Recomputed on resize so a narrow window rewraps instead of
        growing a horizontal scrollbar.
        """
        from ..preferences import scaled_px
        available = max(1, width - scaled_px(self.ASIDE_W)
                        - SPACING["xl"] * 2 - SPACING["lg"]
                        - SPACING["md"] * 2 - 4)
        return max(1, available // (tile_w + SPACING["xs"]))

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
        """Build the right-hand column of status panels."""
        from ..preferences import scaled_px
        from ..theme import make_transparent
        aside = QWidget()
        make_transparent(aside)
        aside.setFixedWidth(scaled_px(self.ASIDE_W))
        col = QVBoxLayout(aside)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(SPACING["md"])

        from ..i18n import tr

        # GitHub #130: Home said what spaCR can do and nothing about where to
        # begin. First in the column, above the panels, because it is the
        # answer to the first question a new user has.
        start = QPushButton(tr("Pipeline overviews"), aside)
        start.setObjectName("PrimaryButton")
        start.setToolTip(tr(
            "Explore pipeline flowcharts, their modules, inputs and outputs, "
            "and optionally start with example data."))
        start.clicked.connect(
            lambda _checked=False: self.sample_project_requested.emit())
        self._sample_project_button = start
        col.addWidget(start)

        self._queued = QueuedPanel()
        self._recent = RecentRunsPanel(known_keys=lambda: self._names)
        self._recent.run_clicked.connect(self._on_run_clicked)
        self._recent.cleared.connect(self.refresh)
        self._news = NewsPanel(self._version())
        self._news.check_requested.connect(self.update_check_requested)
        self._news.refresh_requested.connect(self.news_refresh_requested)
        self._totals = TotalsPanel()
        self._system = SystemPanel()
        self._legend = StageLegend()

        for panel in (self._queued, self._recent, self._news,
                      self._totals, self._system):
            col.addWidget(panel)
        col.addStretch(1)
        return aside

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

    def active_jobs(self) -> int:
        """How many journal-reading threads are still winding down."""
        return self._journal_jobs.active_jobs()

    def set_reserved_content(self, widget: QWidget) -> None:
        """Fill the featured/news surface with real content."""
        self._news.set_content(widget)

    def apply_release_news(self, releases) -> None:
        """Hand a fetched release list to the News panel.

        The answer to :attr:`news_refresh_requested`, and the only way in:
        the window never reaches into the panel, so a page rebuilt at a new
        font scale simply asks again.

        :param releases: records from
            :func:`spacr.updater.fetch_release_notes`, or anything at all.
        """
        self._news.apply_releases(releases)

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
        for _holder, grid, tiles, tile_w in self._grids:
            self._fill_grid(grid, tiles,
                            self._columns_for(self.width(), tile_w))

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
        """Watch the widgets this filter is installed on.

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
        """
        self._journal_jobs.shutdown()
        try:
            self._registry.changed.disconnect(self._on_runs_changed)
        except (RuntimeError, TypeError):
            pass
        self._ticker.stop()
        super().closeEvent(event)


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
