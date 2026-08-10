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

Seven decisions worth knowing about before editing this file:

1. **The first tab is everything, in the same categories as the rest.**
   Home is not a summary of the other tabs — it holds every app, at a
   density that fits one screen, banded by the *same* sections the
   category tabs use. It used to band them into Prepare / Run / Review
   instead, a second grouping that existed nowhere else: an app read as
   "Prepare" on the first tab and "Data" on the second, and adding a
   section meant editing two tables that could disagree. There is one
   table now (:data:`spacr.qt.app.APPS`) and both the bands and the
   tabs are computed from it, by the *registry*, and handed in — this
   widget takes ``categories`` and ``bands`` already grouped and does
   not know what a section means.

   ``bands`` and ``categories`` are therefore the same list today, and
   the two arguments are kept apart anyway: they answer different
   questions ("what does Home list" and "what tabs are there"), and the
   version of this file that assumed they could not differ is the one
   that had to be undone.

   The categories are a *filter*, not a hierarchy you have to descend;
   category tabs with no "everything" view read as an empty page, which
   is the version this one replaced.
2. **One tile, everywhere.** :class:`AppTile` — icon over name, nothing
   else — on Home and on every category tab alike. There used to be two
   sizes: a dense icon-beside-name row on Home and a tall card carrying
   the one-line description on the category tabs. That made the first
   tab look like a list of links and the rest like a launcher, and the
   description was a third copy of text already in the tooltip and in
   the hint bar. The tiles are large, packed tight, and say the module's
   name; :attr:`HomePage._hint_bar` and the tooltip say the rest.
3. **Maturity is a colour, not a place.** Every tile carries a ``stage``
   property (``stable`` / ``beta`` / ``alpha``) which the app
   stylesheet turns into its hover colour, and the legend at the foot of
   the right-hand column says what each colour means. #16i made staging
   two extra TABS instead, which drained three of the five real
   categories and gave "where is the format converter" two answers.
   The classification lives in :data:`spacr.qt.app.APP_STAGE`; the
   colours in :data:`spacr.qt.theme.STAGE_HOVER`. This widget only
   passes them on.
4. **The right-hand column is state, not navigation.** Queue, recent
   runs, machine, release, then the legend. Putting it *beside* the apps
   rather than under them is what stops it pushing the tiles off the
   page. Three of those panels are marked ``(beta)`` — see
   :data:`BETA_SUFFIX`.
5. **A running job is shown here even though Home did not start it.**
   ``spacr.qt.bridge.registry`` knows, because every screen goes through
   ``make_thread``. Home subscribes; nothing had to report in.
6. **Pause is disabled, on purpose, and says why.** See
   :class:`RunningBanner` and :class:`spacr.qt.bridge.PauseGate`.
7. **Every colour is resolved per instance, not imported.** See
   :func:`active_palette` — ``theme.PALETTE`` is a frozen dark palette
   and inlining it renders black-on-black in the light theme.
"""
from __future__ import annotations

import os
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

from ..theme import (
    SPACING, TILE_H, TILE_ICON_PX, TILE_MAX_W, TILE_W, font_px, palette_for,
)
from .divider import Divider

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


# `elide_to_lines` used to live here. It shortened a tile's one-line
# description until it wrapped into at most three lines, because a
# word-wrapped QLabel in a fixed-height box does not elide — it just
# stops painting. #16j removed the descriptions from the tiles, so there
# is no fixed-height wrapped label left on this page and nothing to
# shorten. The identical helper in
# `spacr/resources/home/versions/_generators/parts.py` is still live:
# that renders the archived home-screen candidates, several of which do
# carry a blurb.


def _fmt_elapsed(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


# ---------------------------------------------------------------------------
# Tiles
# ---------------------------------------------------------------------------

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
        super().__init__(parent)
        P = active_palette()
        self._text = text
        self._stage = str(stage or "stable")
        self._size = QSize(int(width), int(height))
        self.setObjectName("AppTile")
        self.setProperty("stage", self._stage)
        self.setCursor(Qt.PointingHandCursor)
        self.setAccessibleName(text)
        # The stage goes in the accessible description as a WORD, not
        # only as a colour: a legend keyed on hue is no legend at all to
        # a screen reader, and colour alone fails WCAG 1.4.1.
        from ..theme import STAGE_LABEL
        mark = STAGE_LABEL.get(self._stage, "")
        self.setAccessibleDescription(
            f"{mark} — {description}" if description and mark else
            (description or mark))
        # NO TOOLTIP on the tile. The description already appears in the
        # hint bar at the bottom of the Home screen, updated by HomePage's
        # eventFilter on the same hover -- so a popup was a second copy of
        # the same sentence, drawn ON TOP of the grid the user is reading
        # to choose between. These blurbs run to several hundred
        # characters, which is fine in a fixed line the eye can skip and
        # wrong in a box covering the tiles.
        #
        # The accessible name and description above are set independently
        # and are what a screen reader reads, so removing the tooltip costs
        # no assistive text.

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
        # `Ignored` horizontally, and added with NO alignment flag, so
        # the layout hands it the tile's whole content width. Both halves
        # matter:
        #
        # * ``setFixedWidth`` — the obvious way to write this, and what
        #   this line used to be — leaves the label at that width with
        #   ``WA_Resized`` still FALSE (``setMaximumSize`` restores the
        #   flag after the resize it does internally). ``ElidingLabel``
        #   deliberately does not elide before its first real layout
        #   pass, so a fixed-width label never elides at all: a long name
        #   is painted and cut off, which is the exact bug this widget
        #   family exists to prevent.
        # * an alignment flag would make QGridLayout/QBoxLayout give the
        #   item only its ``sizeHint`` — which for an ElidingLabel is the
        #   width of the FULL text — and the tile would be dragged wider
        #   by its longest name.
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
        return self._name_lbl

    def is_name_elided(self) -> bool:
        return self._name_lbl.is_elided()

    # -- geometry ------------------------------------------------------
    def heightForWidth(self, width: int) -> int:   # noqa: N802
        """At least the tile height, more if a child somehow needs it."""
        natural = super().heightForWidth(width)
        return max(self._size.height(), natural)

    def sizeHint(self) -> QSize:               # noqa: N802
        return QSize(self._size.width(),
                     self.heightForWidth(self._size.width()))

    def minimumSizeHint(self) -> QSize:        # noqa: N802
        return self.sizeHint()


# ---------------------------------------------------------------------------
# Small panel helper
# ---------------------------------------------------------------------------

class Panel(QWidget):
    """Captioned box for the right-hand column.

    The border lives on a ``QFrame`` with its own object name and the
    rule is scoped to it. An unscoped border rule cascades into every
    child row and outlines each one — a mistake this codebase has
    already made once on the Home dashboard.

    :param beta: mark the header with :data:`BETA_SUFFIX`. The suffix is
        appended *after* the upper-casing, so it stays lower case and
        reads as a mark on the heading rather than part of it.
    """

    def __init__(self, title: str, parent=None, *, beta: bool = False):
        super().__init__(parent)
        P = active_palette()
        self.is_beta = bool(beta)
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(SPACING["xs"])

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
        col.addWidget(self.header)

        box = QFrame()
        box.setObjectName("HomePanelBox")
        # The rounded box KEEPS its dark-grey fill — at the page opacity.
        # `active_palette()` returns raw hex, which is why the preference never
        # reached these panels; `pane_surface` reads it. Making the box itself
        # transparent (tried, reverted) left nothing but a floating outline:
        # the fill is what makes it read as a panel. What has to go is the
        # CONTAINER behind it, which is handled in `_clear_page_surfaces`.
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
        # The Panel wrapper positions a header and the box; it paints nothing.
        # Untagged it takes the blanket `QWidget { background-color: bg }`
        # rule, and six of these stacked down the aside read as one large
        # black column behind every panel — which is exactly what they looked
        # like.
        from ..theme import make_transparent
        make_transparent(self)

    def add(self, widget: QWidget) -> QWidget:
        widget.setStyleSheet("background: transparent;")
        self.body_layout.addWidget(widget)
        return widget


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


# ---------------------------------------------------------------------------
# Running banner
# ---------------------------------------------------------------------------

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
    """

    open_requested = Signal(str)

    def __init__(self, icon_provider: Callable[[str], Optional[QIcon]],
                 names: Dict[str, str], parent=None):
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
        # The app stylesheet's `QPushButton:disabled` rule loses to its
        # own `QPushButton#GhostButton` rule (an ID selector outranks a
        # pseudo-state), so a disabled ghost button renders identically
        # to a live one. This scoped rule puts the difference back —
        # a control that cannot be used has to *look* like it.
        self._btn_pause.setStyleSheet(
            "QPushButton#GhostButton:disabled {"
            f"color: {P['fg_dim']};"
            f"border-color: {P['border_soft']};"
            "background: transparent; }")
        self._btn_pause.clicked.connect(self._on_pause)
        row.addWidget(self._btn_pause)

        # Quit, in red, because a run that will not stop is the state this
        # banner is most often being read in. `Pause` asks the gate and
        # `Open` navigates; this is the only control here that can end a
        # job whose worker has stopped checking whether it should.
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
        # Keep the watcher on the banner rather than on the handle: the
        # handle is retired the moment the job stops, and a timer parented
        # to a dead object is a crash rather than a missed prompt.
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
            thread.terminate()
            thread.wait(2000)
        except RuntimeError:
            # Already gone: the job finished between the prompt and here,
            # which is the good outcome and not an error.
            pass

    # -- state ---------------------------------------------------------
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
            self._bar.setRange(0, 0)          # indeterminate
        else:
            self._bar.setRange(0, 100)
            self._bar.setValue(int(round(fraction * 100)))
        bits = [_fmt_elapsed(handle.elapsed())]
        if handle.progress:
            bits.append(f"{handle.progress[0]} of {handle.progress[1]}")
        if handle.gate.is_paused():
            bits.append("paused")
        # The last line is the *other* thing the job said. When it is the
        # progress line the count above already came from, repeating it
        # is noise ("41 of 96 · Progress: 41/96, operation_type: …").
        tail = handle.last_line
        if tail and not tail.lstrip().startswith("Progress:"):
            bits.append(tail[:70])
        self._sub.setText(" · ".join(bits))
        self._sync_pause_control()

    # -- actions -------------------------------------------------------
    def _on_open(self) -> None:
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

    # -- introspection for tests ---------------------------------------
    @property
    def pause_button(self) -> QPushButton:
        return self._btn_pause


# ---------------------------------------------------------------------------
# Right-hand column
# ---------------------------------------------------------------------------

class QueuedPanel(Panel):
    """The plate queue, when there is one.

    Reads ``~/.spacr/queue.json`` through :class:`spacr.qt.plate_queue.
    PlateQueue` — read-only; the Queue screen owns writes. The panel
    hides itself when the queue is empty rather than drawing an empty
    box, which is the difference between "nothing queued" and "queue
    broken".
    """

    #: Rows drawn before the rest collapse into a "+N more" line.
    MAX_ROWS = 4

    def __init__(self, parent=None):
        super().__init__("Queued", parent)
        self.refresh()

    def queue_items(self) -> List:
        try:
            from ..plate_queue import PlateQueue
            return list(PlateQueue().items())
        except Exception:
            return []

    def refresh(self) -> None:
        P = active_palette()
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
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
            more = QLabel(f"+{len(pending) - self.MAX_ROWS} more")
            more.setStyleSheet(
                f"color: {P['fg_dim']}; font-size: {font_px(11)}px;"
                "background: transparent;")
            self.add(more)
        self.show()


class RecentRunsPanel(Panel):
    """Last few automatically journalled runs; each row navigates."""

    run_clicked = Signal(str)

    def __init__(self, limit: int = 4, parent=None):
        super().__init__("Recent runs", parent)
        self._limit = limit
        self.refresh()

    def read(self) -> list:
        """The journal entries this panel would show. **Worker-thread safe.**

        Split out of :meth:`refresh` so :class:`HomePage` can call it off the
        GUI thread: it touches no widget, only the run journal.
        ``recent_runs`` opens and JSON-parses *every* manifest under the runs
        root before it sorts and truncates — 4 865 of them on this developer's
        machine, measured at 540 ms — so it is not something the GUI thread
        should be doing on the way back to Home.
        """
        try:
            from spacr.run_journal import recent_runs
            return recent_runs(limit=self._limit)
        except Exception:
            return []

    def refresh(self, runs: Optional[list] = None) -> None:
        """Redraw the panel.

        :param runs: entries a worker has already read. ``None`` reads them
            here, on the calling thread — which is what a standalone panel
            and the tests do, and what :class:`HomePage` deliberately does
            not.
        """
        P = active_palette()
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if runs is None:
            runs = self.read()
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
        name = QLabel(key)
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
    row reads as "broken", "no CUDA" reads as an answer.
    """

    def __init__(self, parent=None):
        super().__init__("System", parent)
        self.refresh()

    def refresh(self) -> None:
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.add(_row("GPU", self.gpu_util()))
        self.add(_row("VRAM", self.gpu_vram()))
        self.add(_row("Disk", self.disk_used()))

    @staticmethod
    def gpu_util() -> str:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return f"{pynvml.nvmlDeviceGetUtilizationRates(handle).gpu}%"
        except Exception:
            try:
                import torch
                return "idle" if torch.cuda.is_available() else "no CUDA"
            except Exception:
                return "n/a"

    @staticmethod
    def gpu_vram() -> str:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return f"{info.used / 1e9:.1f} / {info.total / 1e9:.0f} GB"
        except Exception:
            try:
                import torch
                if torch.cuda.is_available():
                    return f"{torch.cuda.memory_allocated() / 1e9:.1f} GB"
            except Exception:
                pass
            return "n/a"

    @staticmethod
    def disk_used() -> str:
        try:
            import shutil
            usage = shutil.disk_usage(os.path.expanduser("~"))
            return f"{int(100 * usage.used / usage.total)}%"
        except Exception:
            return "n/a"


class TotalsPanel(Panel):
    """Aggregate counts from the automatically complete run journal."""

    def __init__(self, parent=None):
        super().__init__("Totals", parent)
        self.refresh()

    def read(self) -> dict:
        """The journal totals. **Worker-thread safe** — see
        :meth:`RecentRunsPanel.read`; ``journal_totals`` walks the same
        thousands of manifests, measured at 247 ms."""
        try:
            from spacr.run_journal import journal_totals
            return journal_totals()
        except Exception:
            return {"total_runs": 0, "mask_runs": 0, "measure_runs": 0,
                    "models_recorded": 0}

    def refresh(self, totals: Optional[dict] = None) -> None:
        """Redraw the panel.

        :param totals: counts a worker has already read; ``None`` reads them
            on the calling thread.
        """
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if totals is None:
            totals = self.read()
        self.add(_row("Runs", str(totals.get("total_runs", 0))))
        self.add(_row("Mask", str(totals.get("mask_runs", 0))))
        self.add(_row("Meas.", str(totals.get("measure_runs", 0))))
        self.add(_row("Models", str(totals.get("models_recorded", 0))))


class StageLegend(Panel):
    """What the three hover colours mean. One row per stage.

    Sits under the rest of the right-hand column because that is where
    the user asked for it, and because it is the only panel there that
    is not a number: it explains the tiles rather than reporting on the
    machine, so it belongs at the end of the column rather than at the
    top of it.

    Each row draws the stage's hue as a filled swatch *and* names the
    stage in words. Colour alone would fail WCAG 1.4.1 and would be
    invisible to the colour-blind mode this app already ships — the
    words are what make it a legend rather than a palette.

    The rows are built from :data:`spacr.qt.theme.STAGE_HOVER`, which is
    the same table the stylesheet builds the hover rules from, so the
    swatch and the tile it explains cannot drift apart.
    """

    #: Side of the colour chip in px, at 100 % font scale.
    SWATCH = 12

    def __init__(self, parent=None):
        super().__init__("Module state", parent)
        from ..theme import STAGE_LABEL, STAGE_NOTE
        self.header.setToolTip(
            "Hover any module tile and it lights up in the colour of "
            "how finished it is.")
        self._rows: Dict[str, QWidget] = {}
        # Least finished first: the row a user needs to have read before
        # they trust a number is the one they should meet first.
        for stage in ("alpha", "beta", "stable"):
            row = self._legend_row(stage, self.swatch_colour(stage),
                                   STAGE_LABEL[stage], STAGE_NOTE[stage])
            self._rows[stage] = self.add(row)

    def _legend_row(self, stage: str, colour: str, label: str,
                    note: str) -> QWidget:
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
        # Filled with the hue and rimmed like the tiles, so the swatch
        # is a small picture of the thing it stands for.
        #
        # The rule is scoped to the chip's own object name on purpose:
        # ``Panel.add`` sets an unscoped ``background: transparent`` on
        # the row this chip lives in, and an unscoped rule on a parent
        # is exactly what strips the fill off its children.
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
        return self._rows.get(stage)


class NewsPanel(Panel):
    """What changed in this release — and the slot for it.

    **There is no release-notes feed in spaCR.** Rather than invent one
    (a hardcoded list of bullets is a lie the moment it is committed),
    this panel names the installed version, offers the update check the
    Help menu already has, and exposes the surface itself through
    :meth:`HomePage.set_reserved_content` so a real feed can be dropped
    in without touching layout code. Nothing here contacts the network
    on its own — the check is a button, so a test and an offline user
    both get a page that just renders.
    """

    check_requested = Signal()

    def __init__(self, version: str = "", parent=None):
        super().__init__(f"News · spaCR {version}" if version else "News",
                         parent, beta=True)
        P = active_palette()
        self.content: Optional[QWidget] = None
        self._placeholder = QLabel(
            "No release notes bundled with this build. "
            "Reserved for featured content — news and what's new land here.")
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(
            f"color: {P['fg_dim']}; font-size: {font_px(11)}px;"
            "font-style: italic; background: transparent;")
        self.add(self._placeholder)

        check = QPushButton("Check for updates…")
        check.setObjectName("GhostButton")
        check.setCursor(Qt.PointingHandCursor)
        check.clicked.connect(self.check_requested)
        self.add(check)
        self._check = check

    def set_content(self, widget: QWidget) -> None:
        """Replace the placeholder with real content."""
        self._placeholder.hide()
        if self.content is not None:
            self.content.setParent(None)
            self.content.deleteLater()
        self.content = widget
        self.body_layout.insertWidget(0, widget)


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

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
    """

    tile_clicked = Signal(str)
    #: Emitted when the page wants the window to run its update check.
    update_check_requested = Signal()

    #: Declared on the class so a paint that arrives mid-construction —
    #: a nested layout activation delivers one on some styles — finds an
    #: answer instead of an ``AttributeError``. ``AppScreen`` learned the
    #: same lesson; see its backdrop-state block.
    _ambient = None

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
        super().__init__(parent)
        self._P = active_palette()
        # The run-journal walk behind Recent runs and Totals goes through
        # here, so returning to Home never blocks on it. journal=False:
        # reading the journal is not itself a run.
        from ..job_runner import JobRunner
        self._journal_jobs = JobRunner(self, app_key="home journal")
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

        # One row per active run, oldest first. Keep ``_banner`` as the first
        # row for compatibility with integrations that predate concurrent
        # module runs.
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

        self._hint_bar = QLabel(_DEFAULT_HINT)
        self._hint_bar.setObjectName("HintBar")
        self._hint_bar.setAlignment(Qt.AlignHCenter)
        # Derived from the font rather than pinned at 32. A hard number is a
        # promise about text metrics that no longer holds the moment the font
        # scale, the theme's font stack or a label's size role changes — the
        # hint text needed 35 px against a 32 px floor and clipped. Asking the
        # label for its own sizeHint keeps the bar correct across all of them.
        self._hint_bar.setMinimumHeight(
            max(32, self._hint_bar.sizeHint().height() + SPACING["xs"]))
        outer.addWidget(self._hint_bar)

        # Live job state. The registry is process-wide and outlives this
        # page (Home is rebuilt on every theme change), so the connection
        # is made with a bound method and dropped in closeEvent —
        # a lambda would keep a destroyed page alive as a receiver.
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

        # And unconditionally, whatever happened above — the same
        # correction the module screens already carry. This used to run
        # only inside the successful-install arm, on the reasoning that a
        # page with nothing behind it should stay opaque; that reasoning
        # is what left Home a solid `bg` slab for anyone with the ambient
        # preference off or the Animation preference set to `none`. There
        # is never nothing behind it: `paintEvent` paints the page, which
        # is what these containers are supposed to be showing.
        # `clear_container_surfaces` is idempotent, so the call inside
        # `_install_ambient` stays where it is for its own ordering
        # reasons and this one costs a second pass over the tree.
        self._clear_page_surfaces()

    # -- the page itself -----------------------------------------------
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
        if self._ambient is not None:
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

    # -- ambient backdrop ----------------------------------------------
    def _install_ambient(self) -> None:
        """Put the ambient animation behind Home. Never raises.

        Home needs none of :class:`spacr.qt.screens.app_screen.AppScreen`'s
        ``_ambient_applied`` bookkeeping and no ``changeEvent`` handling:
        this page is rebuilt from scratch on every theme change (see the
        registry comment in ``__init__``), so a stale flat fill cannot
        survive one, and there is no second attempt to guard against.

        Ordered exactly as the module screens are, for the same two reasons:
        the preference is read *before* anything is constructed, because not
        building it is the cost the toggle exists to avoid; and the page
        surfaces are cleared only *after* a successful install, so a failed
        one leaves Home opaque and normal rather than transparent with
        nothing behind it.
        """
        widget = None
        try:
            from ..preferences import (get_ambient_enabled,
                                       get_ambient_palette,
                                       get_ambient_theme)
            if not get_ambient_enabled():
                return
            from .ambient import install_ambient
            widget = install_ambient(
                self, None,
                theme=get_ambient_theme(),
                palette=get_ambient_palette(),
                backdrop=self._ambient_backdrop())
            self._clear_page_surfaces()
            self._ambient = widget
        except Exception:
            self._ambient = None
            self._discard_ambient(widget)

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
            # If the import is what failed, nothing was constructed.
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

        # The generic sweep FIRST. Home used to hand-list five widgets it
        # guessed were responsible, which is why measuring found three that
        # were not on it: the hero's own QLabels, Qt's internal
        # `qt_tabwidget_tabbar`, and the anonymous row hosts the tiles sit in.
        # Naming widgets one at a time cannot keep up with a layout; sweeping
        # by rule can.
        clear_container_surfaces(self)

        # Qt builds the tab bar itself, so it is neither anonymous nor ours to
        # name at construction — it has to be reached through the tab widget.
        for bar in self.findChildren(QTabBar):
            make_transparent(bar)

        # The hero's labels: the mark, the wordmark and the subtitle. They are
        # type on the page, and a QLabel with no rule of its own takes the
        # blanket window fill — which is what left a black band across the
        # masthead after the Hero FRAME was already transparent.
        hero = self.findChild(QWidget, "Hero")
        if hero is not None:
            make_transparent(*hero.findChildren(QLabel))

        make_transparent(*(w for w in (
            getattr(self, "_running_host", None),
            getattr(self, "_hint_bar", None),
            getattr(self, "_tabs", None),
        ) if w is not None))

        # Every scroll area, its viewport, and the stacked pages the tabs keep
        # their tab bodies in. These are the containers behind the tile rows
        # and their headings.
        tabs = getattr(self, "_tabs", None)
        if tabs is not None:
            for area in tabs.findChildren(QScrollArea):
                make_transparent(area, area.viewport())
            make_transparent(*tabs.findChildren(QStackedWidget))
            # The direct page widgets of each tab: one per category, each the
            # host for that category's rows and headings.
            for i in range(tabs.count()):
                page = tabs.widget(i)
                if page is not None:
                    make_transparent(page)
        # The body wrapper is the first child of `outer` and has no object
        # name of its own, so it is reached through the layout.
        layout = self.layout()
        if layout is not None and layout.count():
            item = layout.itemAt(0)
            body = item.widget() if item is not None else None
            if body is not None:
                make_transparent(body)

    # -- pieces --------------------------------------------------------
    def _new_running_banner(self) -> RunningBanner:
        banner = RunningBanner(self._icon_provider, self._names)
        banner.open_requested.connect(self.tile_clicked)
        self._running_layout.addWidget(banner)
        self._banners.append(banner)
        return banner

    def _refresh_run_banners(self) -> None:
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
        P = self._P
        hero = QWidget()
        row = QHBoxLayout(hero)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["md"])

        # The mark and the wordmark move together, so both take the Zoom
        # multiplier: scaling only the text leaves a 52 px logo beside
        # 78 px lettering, which is the one thing the two constants exist
        # to prevent.
        logo_px = font_px(HERO_LOGO_PX)
        logo = _find_logo_pixmap()
        if logo is not None:
            label = QLabel()
            label.setPixmap(logo.scaled(logo_px, logo_px,
                                        Qt.KeepAspectRatio,
                                        Qt.SmoothTransformation))
            label.setFixedSize(logo_px, logo_px)
            label.setStyleSheet("background: transparent;")
            row.addWidget(label)

        title = QLabel("spaCR")
        title.setStyleSheet(
            "font-family: 'Open Sans', sans-serif; font-weight: 300;"
            f"font-size: {font_px(HERO_TITLE_PX)}px; color: {P['accent']};"
            "letter-spacing: -0.6px; background: transparent;")
        row.addWidget(title)

        subtitle = QLabel(
            "End-to-end microscopy → single-cell measurements "
            "→ genotype-phenotype mapping.")
        subtitle.setObjectName("HeroSubtitle")
        subtitle.setWordWrap(True)
        row.addWidget(subtitle, 1)

        # No "All apps" button: the first tab IS all apps. The edge
        # drawer still exists for the screens that are not Home, and is
        # reachable there from the spaCR menu or Ctrl+B.
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
        # documentMode(True) suppresses the pane frame, and without the
        # frame the tab strip floats with nothing under it.
        self._tabs.setDocumentMode(False)
        try:
            from ..preferences import resolve_effective_theme
            glass = resolve_effective_theme() == "glass"
        except Exception:
            glass = False
        self._tabs.setStyleSheet(
            _tab_qss(self._P, self._pane_alpha(), glass=glass))
        # The QStackedWidget QTabWidget keeps its pages in is a plain
        # QWidget, and the blanket `QWidget { background-color: bg }`
        # rule makes it paint the window colour over the ::pane it sits
        # on. Every other layer between the pane and the tiles is tagged
        # in `_scrolled`; this is the one that is not ours to construct.
        from PySide6.QtWidgets import QStackedWidget
        from ..theme import make_transparent
        make_transparent(*self._tabs.findChildren(QStackedWidget))

        self._section_names = [title for title, _e in self._categories]

        self._tabs.addTab(self._build_home_tab(),
                          f"Home  ({len(self._apps)})")
        for section, entries in self._categories:
            self._tabs.addTab(self._build_category_tab(section, entries),
                              _escape_amp(f"{section}  ({len(entries)})"))
        return self._tabs

    # -- tab 1: everything ---------------------------------------------
    #
    # One band per category, in the order the app registry hands them
    # over, each app in exactly one. There is deliberately no membership
    # table here: the page this replaced had Prepare/Run/Review bands
    # with a section→band map and a three-app override list *in this
    # file*, so "which group is Plate Queue in" had two answers
    # depending on which tab you were looking at, and a renamed section
    # silently dropped its apps into a fallback band.

    def _build_home_tab(self) -> QWidget:
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
            # Tiles are packed tight — the rim is what separates them
            # now, not the gap, so the gap only has to stop two rims
            # touching and reading as one line.
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

    # -- tabs 2..6: one category each -----------------------------------
    def _build_category_tab(self, section: str,
                            entries: List[Tuple[str, str, str]]) -> QWidget:
        from ..preferences import scaled_px
        P = self._P
        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(SPACING["md"], SPACING["sm"],
                               SPACING["md"], SPACING["sm"])
        # `xs`, not `sm`: the heading block, the rule and the grid are one
        # unit, and every gap above the grid is a gap that decides
        # whether the biggest tab scrolls.
        col.setSpacing(SPACING["xs"])

        # Heading + note in one block on 2 px, so adding the note costs a
        # line rather than a line plus a layout gap.
        #
        # The heading is redundant with the tab label for a sighted user,
        # but it is what a screen reader lands on inside the page and
        # what the category-coverage test reads.
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
        # Packed tight, in both axes: each tile carries its own rim, so
        # the gap no longer has to do the work of telling two of them
        # apart — it only has to stop two rims reading as one line.
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
        # Preferred/Fixed + a maximum: the tile widens to reach the edge
        # of its column (see ``_fill_grid``) but stops at TILE_MAX_W, and
        # never changes height.
        tile.setMaximumWidth(scaled_px(self.TILE_MAX_W))
        tile.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        return self._wire_tile(tile, key, desc)

    # -- shared ---------------------------------------------------------
    def _wire_tile(self, tile, key: str, desc: str):
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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # No stylesheet on the scroll area: an unscoped
        # `background: transparent` on a QScrollArea cascades to every
        # descendant and strips the fill off the tiles inside it.
        # `make_transparent` tags each widget with a property instead,
        # and the QSS rule matches only widgets carrying it.
        #
        # This is what lets the pane's colour reach the eye. In the two
        # opaque themes the blanket `QWidget { background-color: bg }`
        # rule made this page — and its scroll viewport — paint the
        # window colour straight over the rounded box behind them, so
        # the box was a border around nothing and the opacity preference
        # could not have shown a difference at any setting.
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
        # No alignment flags at all: QGridLayout gives an *aligned* item
        # exactly its sizeHint and positions it in the cell, so even
        # Qt.AlignTop alone leaves a 172 px tile sitting in a 205 px
        # column with a gap after it. Unaligned, the item is handed the
        # whole cell; the tile's Fixed vertical policy keeps the height,
        # and its maximumWidth caps how far it stretches.
        rows = 0
        for index, tile in enumerate(tiles):
            rows = index // columns
            grid.addWidget(tile, rows, index % columns)
        for row in range(grid.rowCount()):
            grid.setRowStretch(row, 0)
        grid.setRowStretch(rows + 1, 1)
        # Stretch is set over `columns` columns even when fewer are
        # occupied. ``grid.columnCount()`` counts columns that HAVE an
        # item, so a band with a single app got one column, that column
        # took the whole width, and the unaligned tile floated to the
        # middle of the page under a left-aligned heading. Naming the
        # empty columns puts the tile back at the left edge where the
        # heading is.
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
        # No cap at six any more: the tiles are 172 px, and capping the
        # count is how a wide window ends up with a row that stops
        # two-thirds of the way across and a page that reads as empty.
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
        from ..preferences import scaled_px
        from ..theme import make_transparent
        aside = QWidget()
        # The column itself paints nothing. Untagged it runs the full height of
        # the window as one black slab behind every panel in it — the "one
        # large black box spanning all right side elements and going down to
        # the bottom".
        make_transparent(aside)
        aside.setFixedWidth(scaled_px(self.ASIDE_W))
        col = QVBoxLayout(aside)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(SPACING["md"])

        self._queued = QueuedPanel()
        self._recent = RecentRunsPanel()
        self._recent.run_clicked.connect(self.tile_clicked)
        self._system = SystemPanel()
        self._news = NewsPanel(self._version())
        self._news.check_requested.connect(self.update_check_requested)
        self._totals = TotalsPanel()
        self._legend = StageLegend()

        for panel in (self._queued, self._recent, self._system,
                      self._news, self._totals, self._legend):
            col.addWidget(panel)
        col.addStretch(1)
        return aside

    @property
    def legend(self) -> "StageLegend":
        """The colour-to-maturity key at the foot of the right column."""
        return self._legend

    @staticmethod
    def _version() -> str:
        try:
            import spacr
            version = str(getattr(spacr, "__version__", "") or "").strip()
        except Exception:
            return ""
        # "dev" / "" are not a release, and heading a panel "spaCR dev"
        # says less than heading it "News".
        return "" if version.lower() in ("", "dev", "unknown") else version

    # -- live state ----------------------------------------------------
    def _on_runs_changed(self) -> None:
        """Show every active job across the top, oldest first."""
        # `user_visible` is False for housekeeping the user did not start --
        # the two-second usage poll above all. Without this filter Home
        # flashed a blue "<module> usage - running" banner on and off
        # continuously while any module screen was open.
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

    def closeEvent(self, event):        # noqa: N802 - Qt override
        """Do not let a journal walk outlive the page that asked for it."""
        self._journal_jobs.shutdown()
        super().closeEvent(event)

    # -- API kept from the page this replaces --------------------------
    def set_reserved_content(self, widget: QWidget) -> None:
        """Fill the featured/news surface with real content."""
        self._news.set_content(widget)

    @property
    def _reserved_content(self) -> Optional[QWidget]:
        """The widget currently filling the news surface, if any."""
        return self._news.content

    # -- events --------------------------------------------------------
    def resizeEvent(self, event):               # noqa: N802
        super().resizeEvent(event)
        for _holder, grid, tiles, tile_w in self._grids:
            self._fill_grid(grid, tiles,
                            self._columns_for(self.width(), tile_w))

    def eventFilter(self, obj, event):          # noqa: N802
        if event.type() == QEvent.Enter:
            hint = self._tile_hints.get(obj)
            if hint:
                from ..i18n_module_summaries import module_summary
                from ..theme import STAGE_LABEL
                key, source = hint
                summary = module_summary(key, source)
                # The stage goes in the hint bar as a WORD. It used to ride
                # on the tile's tooltip, which is gone -- and the tile's
                # hover HUE cannot be the only carrier, because colour alone
                # fails WCAG 1.4.1. The accessible description covers screen
                # readers; this covers a sighted colour-blind user, who
                # reads neither the hue nor the accessibility tree.
                mark = STAGE_LABEL.get(
                    str(obj.property("stage") or "stable"), "")
                self._hint_bar.setText(
                    f"{summary} — {mark}" if mark else summary)
        elif event.type() == QEvent.Leave:
            from ..i18n import tr
            self._hint_bar.setText(tr(_DEFAULT_HINT))
        return super().eventFilter(obj, event)

    def closeEvent(self, event):                # noqa: N802
        try:
            self._registry.changed.disconnect(self._on_runs_changed)
        except (RuntimeError, TypeError):
            pass
        self._ticker.stop()
        super().closeEvent(event)


def _tab_qss(P: dict, pane_alpha: float = 1.0,
             glass: bool = False) -> str:
    """QSS for the Home tab widget.

    :param pane_alpha: accepted so the call sites and their tests keep one
        signature; the pane itself paints nothing.

    The box behind the tiles is GONE, not dialled. This went back and forth:
    it was a surface at the effective alpha, then transparent, then briefly
    painted at the preference again on the reading that opacity should "apply
    to the containers the tiles are in". The final instruction is the clearest
    of the three — remove the black boxes behind the tiles and make the TILES
    subject to opacity instead — so the container is transparent and the
    dialling moved to the tile fill, where it is actually visible.

    The 1px outline stays: it is what the selected tab joins onto, and without
    it the tab strip floats with nothing under it.
    """
    from ..theme import css_color
    pane_border = (css_color("#ffffff", 0.27)
                   if glass else P["border_soft"])
    radius = 14 if glass else 8
    selected_fill = ("transparent" if pane_alpha <= 0.0
                     else css_color(P["surface"], pane_alpha))
    return f"""
QTabWidget#HomeTabs::pane {{
    border: 1px solid {pane_border};
    border-radius: {radius}px;
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
/* The selected tab takes the page opacity like everything else. It paints
   `surface` so it joins onto the pane's edge — but with the pane transparent
   that made it the last solid black rectangle on the page, which is exactly
   what "the tabs still have black backgrounds" was pointing at. */
QTabWidget#HomeTabs > QTabBar::tab:selected {{
    color: {P['accent']};
    background: {selected_fill};
    border: 1px solid {P['border_soft']};
    border-bottom-color: {selected_fill};
}}
"""
