"""HomePage — the Home screen.

    ┌────────────────────────────────────────────────────────────────┐
    │ 🖼 spaCR   End-to-end microscopy → single-cell measurements …   │
    │ ┌ Mask · running ────── 41 of 96 ──── [Open] [Pause] ────────┐ │
    │ │ Home │ Core │ Data │ Segmentation models │ Results │ Toxo │ │  QUEUED
    │ │ PREPARE 8 ─────────────────────────────────────────────── │ │  RECENT
    │ │  ▢ Format Converter  ▢ Align & Stitch  ▢ Import Project … │ │  SYSTEM
    │ │ RUN 15 ─────────────────────────────────────────────────── │ │  NEWS
    │ │  ▢ Mask  ▢ Timelapse  ▢ Motility  ▢ Measure  ▢ Annotate … │ │  TOTALS
    │ │ REVIEW 7 ───────────────────────────────────────────────── │ │
    │ │  ▢ Plate Viewer  ▢ Annotator Agreement  ▢ Image UMAP …     │ │
    │ └────────────────────────────────────────────────────────────┘ │
    │  Hover a tile to see what it does.                             │
    └────────────────────────────────────────────────────────────────┘

Six decisions worth knowing about before editing this file:

1. **Six tabs, and the first one is everything.** Home is not a summary
   of the other five — it holds every app, in three broad bands, at a
   density that fits one screen. The categories are then a *filter*, not
   a hierarchy you have to descend. Five tabs of nine apps with no
   "everything" view read as an empty page, which is the version this
   one replaced.
2. **Two tile sizes, on purpose.** The Home tab uses :class:`DenseTile`
   (icon + name, five to a row) because thirty tiles have to fit. Each
   category tab uses :class:`TallTile` (icon over name over the one-line
   description) because nine tiles have room to explain themselves —
   and explaining themselves is why the categories are worth a tab.
3. **The right-hand column is state, not navigation.** Queue, recent
   runs, machine, release. Putting it *beside* the apps rather than
   under them is what stops it pushing the tiles off the page.
4. **A running job is shown here even though Home did not start it.**
   ``spacr.qt.bridge.registry`` knows, because every screen goes through
   ``make_thread``. Home subscribes; nothing had to report in.
5. **Pause is disabled, on purpose, and says why.** See
   :class:`RunningBanner` and :class:`spacr.qt.bridge.PauseGate`.
6. **Every colour is resolved per instance, not imported.** See
   :func:`active_palette` — ``theme.PALETTE`` is a frozen dark palette
   and inlining it renders black-on-black in the light theme.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import QEvent, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QPixmap
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

from ..theme import SPACING, palette_for
from .divider import Divider
from .tile import HTile


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


def _escape_amp(text: str) -> str:
    """Double any ``&`` so Qt draws it instead of eating it.

    ``QTabBar`` (like ``QToolButton``) reads a lone ``&`` as a mnemonic:
    "Results & QC" renders as "Results  QC" with an underlined Q. Only
    one of the six tab labels is affected today, which is exactly the
    kind of thing that ships unnoticed.
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


def elide_to_lines(text: str, font, width: int, lines: int) -> str:
    """Shorten ``text`` until it wraps into at most ``lines`` lines.

    A word-wrapped ``QLabel`` in a fixed-height box does not elide — it
    just stops painting, which is the silent clipping this whole layout
    effort exists to remove ("… invasion efficiency per we"). Shortening
    the string up front instead means the box can be a fixed height and
    still never cut a word in half; the full text stays on the tile's
    tooltip.
    """
    from PySide6.QtGui import QFontMetrics
    metrics = QFontMetrics(font)

    def wrapped_lines(value: str) -> int:
        rect = metrics.boundingRect(
            0, 0, max(1, width), 10000,
            int(Qt.TextWordWrap | Qt.AlignHCenter | Qt.AlignTop), value)
        return max(1, round(rect.height() / max(1, metrics.lineSpacing())))

    if wrapped_lines(text) <= lines:
        return text
    words = text.split()
    kept: list = []
    for word in words:
        candidate = " ".join(kept + [word]) + "…"
        if wrapped_lines(candidate) > lines:
            break
        kept.append(word)
    return (" ".join(kept) + "…") if kept else text[:1] + "…"


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

class DenseTile(HTile):
    """The Home tab's tile: icon, name, nothing else — packed tight.

    An :class:`HTile` whose height and name size it actually keeps.
    ``setFixedSize`` does not survive the app stylesheet:
    ``theme.stylesheet()`` carries ``QPushButton { min-height: 22px }``
    and ``QStyleSheetStyle`` re-applies that rule's geometry on polish,
    wiping the minimum. A tile then reports 40 px to its layout and gets
    squashed — painted, just cropped, with no warning. The supported
    route is to answer through ``sizeHint`` / ``minimumSizeHint``, which
    is what this does. The height is a literal rather than
    ``self.minimumHeight()`` for the same reason: that is exactly the
    value the stylesheet overwrites.

    ``name_px`` restyles the name label down from the 17 px "subtitle"
    size. That size is what forces a 255 px minimum tile width — with a
    300 px aside beside it, thirty tiles at 255 px would not fit the
    page without scrolling. At 13 px the longest name ("Annotator
    Agreement") needs about 205 px, which is what gets all thirty onto
    one screen.
    """

    def __init__(self, *args, tile_height: int = 66, name_px: int = 0,
                 ink: str = "", **kwargs):
        self._tile_height = int(tile_height)
        super().__init__(*args, **kwargs)
        if name_px and self._name_lbl is not None:
            self._name_lbl.setStyleSheet(
                f"font-size: {name_px}px; font-weight: 500;"
                f"color: {ink or active_palette()['fg']};"
                "background: transparent;")

    def sizeHint(self) -> QSize:               # noqa: N802 (Qt casing)
        base = super().sizeHint()
        return QSize(base.width(), max(base.height(), self._tile_height))

    def minimumSizeHint(self) -> QSize:        # noqa: N802
        base = super().minimumSizeHint()
        return QSize(base.width(), max(base.height(), self._tile_height))


class TallTile(QPushButton):
    """The category tabs' tile: icon over name over description.

    The rail-and-pane card. Deliberately **not** an ``HTile`` subclass:
    ``HTile`` is a horizontal row and its description lives in a
    ``QLabel#HTileDesc``, which the Home layout contract says a
    horizontal row must not carry. This is a different shape with a
    different contract — a launcher card big enough to read the
    one-line description off, which is the whole reason the categories
    got their own tabs.

    Sized through ``sizeHint``/``minimumSizeHint`` for the same reason
    as :class:`DenseTile`.
    """

    #: Lines the one-line description is allowed to wrap into. Anything
    #: longer is elided; the tooltip keeps the whole thing.
    BLURB_LINES = 3

    def __init__(self, text: str, description: str = "",
                 icon: Optional[QIcon] = None, *, width: int, height: int,
                 icon_px: int = 52, parent=None):
        super().__init__(parent)
        P = active_palette()
        self._text = text
        self._size = QSize(int(width), int(height))
        self.setObjectName("HTile")          # inherits the tile styling
        self.setCursor(Qt.PointingHandCursor)
        self.setAccessibleName(text)
        self.setAccessibleDescription(description)
        self.setToolTip(f"{text} — {description}" if description else text)

        col = QVBoxLayout(self)
        col.setContentsMargins(12, 12, 12, 12)
        col.setSpacing(6)
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
        name.setFixedWidth(width - 24)
        name.setStyleSheet(
            f"color: {P['fg']}; font-size: 14px; font-weight: 500;"
            "background: transparent;")
        col.addWidget(name, 0, Qt.AlignHCenter)
        self._name_lbl = name

        if description:
            blurb = QLabel()
            blurb.setWordWrap(True)
            blurb.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
            blurb.setFixedWidth(width - 24)
            blurb.setStyleSheet(
                f"color: {P['fg_muted']}; font-size: 11px;"
                "background: transparent;")
            blurb.ensurePolished()
            from PySide6.QtGui import QFontMetrics
            blurb.setText(elide_to_lines(description, blurb.font(),
                                         width - 24, self.BLURB_LINES))
            # A fixed height, so every card in a row is the same height
            # and none of them can grow into its neighbour's space. The
            # text was already shortened to fit it.
            blurb.setFixedHeight(
                QFontMetrics(blurb.font()).lineSpacing() * self.BLURB_LINES)
            col.addWidget(blurb, 0, Qt.AlignHCenter)
        col.addStretch(1)

    @property
    def text_label(self) -> str:
        """The tile's app name, matching ``HTile.text_label``."""
        return self._text

    @property
    def name_label(self):
        return self._name_lbl

    def is_name_elided(self) -> bool:
        return self._name_lbl.is_elided()

    # -- geometry ------------------------------------------------------
    #
    # The wrapped description makes this widget height-for-width, and
    # ``QWidgetItem::sizeHint`` *prefers* ``heightForWidth`` over
    # ``sizeHint().height()`` whenever it is available. Overriding
    # sizeHint alone therefore did nothing: the cards rendered 34 px
    # shorter than asked for, and a longer blurb would have been
    # silently clipped instead of given a taller card. So the floor goes
    # into heightForWidth, where the layout actually reads it.

    def heightForWidth(self, width: int) -> int:   # noqa: N802
        """At least the card height, more when the blurb needs more."""
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
    """

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        P = active_palette()
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(SPACING["xs"])

        self.header = QLabel(title.upper())
        self.header.setObjectName("HomePanelHeader")
        self.header.setStyleSheet(
            "font-family: 'Open Sans', sans-serif; font-weight: 600;"
            "font-size: 10px; letter-spacing: 2px; background: transparent;"
            f"color: {P['fg_muted']};")
        col.addWidget(self.header)

        box = QFrame()
        box.setObjectName("HomePanelBox")
        box.setStyleSheet(
            "QFrame#HomePanelBox {"
            f"background: {P['surface_alt']};"
            f"border: 1px solid {P['border_soft']};"
            "border-radius: 8px; }")
        self.body_layout = QVBoxLayout(box)
        self.body_layout.setContentsMargins(SPACING["md"], SPACING["sm"],
                                            SPACING["md"], SPACING["sm"])
        self.body_layout.setSpacing(SPACING["xs"])
        col.addWidget(box)
        self._box = box

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
    left.setStyleSheet(f"color: {P['fg_muted']}; font-size: 11px;"
                       "font-weight: 500; background: transparent;")
    left.setMinimumWidth(48)
    right = QLabel(value)
    right.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    right.setStyleSheet(
        f"color: {value_colour or P['fg']}; font-size: 12px;"
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
            f"color: {P['fg']}; font-size: 14px; font-weight: 600;"
            "background: transparent;")
        self._sub = QLabel("")
        self._sub.setObjectName("HomeRunningSub")
        self._sub.setStyleSheet(
            f"color: {P['fg_muted']}; font-size: 11px;"
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

        self.hide()

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
            more.setStyleSheet(f"color: {P['fg_dim']}; font-size: 11px;"
                               "background: transparent;")
            self.add(more)
        self.show()


class RecentRunsPanel(Panel):
    """Last few runs from the run journal; each row navigates."""

    run_clicked = Signal(str)

    def __init__(self, limit: int = 4, parent=None):
        super().__init__("Recent runs", parent)
        self._limit = limit
        self.refresh()

    def refresh(self) -> None:
        P = active_palette()
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        try:
            from spacr.run_journal import recent_runs
            runs = recent_runs(limit=self._limit)
        except Exception:
            runs = []
        if not runs:
            hint = QLabel("No runs yet.")
            hint.setStyleSheet(f"color: {P['fg_dim']}; font-size: 11px;"
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
            "font-size: 11px; background: transparent;")
        name = QLabel(key)
        name.setStyleSheet(f"color: {P['fg']}; font-size: 12px;"
                           "background: transparent;")
        when = QLabel(_fmt_elapsed(elapsed))
        when.setStyleSheet(f"color: {P['fg_dim']}; font-size: 11px;"
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
    """Aggregate journal counts."""

    def __init__(self, parent=None):
        super().__init__("Totals", parent)
        self.refresh()

    def refresh(self) -> None:
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        try:
            from spacr.run_journal import journal_totals
            totals = journal_totals()
        except Exception:
            totals = {"total_runs": 0, "mask_runs": 0, "measure_runs": 0,
                      "models_recorded": 0}
        self.add(_row("Runs", str(totals.get("total_runs", 0))))
        self.add(_row("Mask", str(totals.get("mask_runs", 0))))
        self.add(_row("Meas.", str(totals.get("measure_runs", 0))))
        self.add(_row("Models", str(totals.get("models_recorded", 0))))


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
                         parent)
        P = active_palette()
        self.content: Optional[QWidget] = None
        self._placeholder = QLabel(
            "No release notes bundled with this build. "
            "Reserved for featured content — news and what's new land here.")
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(
            f"color: {P['fg_dim']}; font-size: 11px;"
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
    """

    tile_clicked = Signal(str)
    #: Emitted when the page wants the window to run its update check.
    update_check_requested = Signal()

    #: Category-tab card, at 100 % font scale — the rail-and-pane size:
    #: big enough to carry the one-line description, which is the whole
    #: point of giving each category its own tab.
    TILE_MIN_W = 246
    TILE_H = 172
    TALL_ICON_PX = 52
    #: Home-tab tile. Small name, small icon, five to a row — the size
    #: that gets every app onto one screen next to the aside.
    DENSE_TILE_W = 205
    DENSE_TILE_MAX_W = 300
    DENSE_TILE_H = 62
    DENSE_ICON_PX = 40
    DENSE_NAME_PX = 13

    #: Right-hand column width. Fixed: it holds numbers, and a column of
    #: numbers that reflows on every window resize is unreadable.
    ASIDE_W = 300

    def __init__(
        self,
        apps: List[Tuple[str, str, str, str]],
        icon_provider: Callable[[str], Optional[QIcon]],
        parent=None,
    ):
        super().__init__(parent)
        self._P = active_palette()
        self._apps = list(apps)
        self._icon_provider = icon_provider
        self._names = {k: n for k, n, _d, _s in self._apps}
        self._tile_hints: dict = {}
        #: (holder, grid, tiles, tile_width, fill) per grid, so a resize
        #: can rewrap each one at its own column width.
        self._grids: List[Tuple[QWidget, QGridLayout, list, int,
                        bool]] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(SPACING["xl"], SPACING["lg"],
                               SPACING["xl"], SPACING["md"])
        col.setSpacing(SPACING["md"])

        col.addWidget(self._build_hero())

        self._banner = RunningBanner(icon_provider, self._names)
        self._banner.open_requested.connect(self.tile_clicked)
        col.addWidget(self._banner)

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
        self._hint_bar.setMinimumHeight(32)
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
        self._ticker.timeout.connect(self._banner.refresh)

        self._on_runs_changed()

    # -- pieces --------------------------------------------------------
    def _build_hero(self) -> QWidget:
        P = self._P
        hero = QWidget()
        row = QHBoxLayout(hero)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["md"])

        logo = _find_logo_pixmap()
        if logo is not None:
            label = QLabel()
            label.setPixmap(logo.scaled(44, 44, Qt.KeepAspectRatio,
                                        Qt.SmoothTransformation))
            label.setFixedSize(44, 44)
            label.setStyleSheet("background: transparent;")
            row.addWidget(label)

        title = QLabel("spaCR")
        title.setStyleSheet(
            "font-family: 'Open Sans', sans-serif; font-weight: 300;"
            f"font-size: 34px; color: {P['accent']};"
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
        """Six tabs: Home (everything), then one per category.

        Home is not a summary of the categories, it *is* every app —
        which is what makes the categories optional rather than a
        hierarchy you have to navigate. It groups them into three broad
        bands because thirty unlabelled tiles is a wall, and three
        headings is about as many as anyone holds in their head while
        scanning.
        """
        self._tabs = QTabWidget()
        self._tabs.setObjectName("HomeTabs")
        # documentMode(True) suppresses the pane frame, and without the
        # frame the tab strip floats with nothing under it.
        self._tabs.setDocumentMode(False)
        self._tabs.setStyleSheet(_tab_qss(self._P))

        sections: Dict[str, List[Tuple[str, str, str]]] = {}
        for key, name, desc, section in self._apps:
            sections.setdefault(section, []).append((key, name, desc))
        self._section_names = list(sections)

        self._tabs.addTab(self._build_home_tab(),
                          f"Home  ({len(self._apps)})")
        for section, entries in sections.items():
            self._tabs.addTab(self._build_category_tab(section, entries),
                              _escape_amp(f"{section}  ({len(entries)})"))
        return self._tabs

    # -- tab 1: everything ---------------------------------------------
    #
    # Three broad bands. Membership follows the five sections so a new
    # app lands somewhere sensible without anyone editing a second
    # table; the three exceptions below are the apps whose *section* and
    # whose *stage of work* genuinely disagree.
    _BAND_OVERRIDE = {
        "queue":      "Run",      # Data by kind, but it runs plates
        "batch":      "Run",      # ditto
        "db_browser": "Review",   # Data by kind, but you open it to read
    }
    _BAND_FOR_SECTION = {
        "Core": "Run",
        "Data": "Prepare",
        "Segmentation models": "Prepare",
        "Results & QC": "Review",
        "Toxoplasma": "Run",
    }
    BANDS = ("Prepare", "Run", "Review")

    def _band_of(self, key: str, section: str) -> str:
        return self._BAND_OVERRIDE.get(
            key, self._BAND_FOR_SECTION.get(section, "Run"))

    def _build_home_tab(self) -> QWidget:
        from ..preferences import scaled_px
        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(SPACING["md"], SPACING["sm"],
                               SPACING["md"], SPACING["sm"])
        col.setSpacing(SPACING["xs"])

        bands: Dict[str, List[Tuple[str, str, str]]] = {
            band: [] for band in self.BANDS}
        for key, name, desc, section in self._apps:
            bands[self._band_of(key, section)].append((key, name, desc))

        for band in self.BANDS:
            entries = bands[band]
            if not entries:
                continue
            col.addWidget(self._band_header(band, len(entries)))
            holder = QWidget()
            grid = QGridLayout(holder)
            grid.setContentsMargins(0, 0, 0, SPACING["sm"])
            grid.setHorizontalSpacing(SPACING["sm"])
            grid.setVerticalSpacing(SPACING["xs"])
            tiles = [self._make_dense_tile(k, n, d) for k, n, d in entries]
            self._grids.append((holder, grid, tiles,
                                scaled_px(self.DENSE_TILE_W), True))
            self._fill_grid(grid, tiles, self._columns_for(
                self.width(), scaled_px(self.DENSE_TILE_W)))
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
            "font-size: 11px; letter-spacing: 2px; background: transparent;"
            f"color: {P['fg_muted']};")
        note = QLabel(str(count))
        note.setStyleSheet(f"color: {P['fg_dim']}; font-size: 11px;"
                           "background: transparent;")
        row.addWidget(label)
        row.addWidget(note)
        row.addStretch(1)
        col.addLayout(row)
        col.addWidget(Divider())
        return wrap

    def _make_dense_tile(self, key: str, name: str, desc: str) -> DenseTile:
        from ..preferences import scaled_px
        icon = self._icon_provider(key) if self._icon_provider else None
        tile = DenseTile(text=name, description="", icon=icon,
                         icon_size=self.DENSE_ICON_PX,
                         tile_height=scaled_px(self.DENSE_TILE_H),
                         name_px=self.DENSE_NAME_PX, ink=self._P["fg"])
        width = scaled_px(self.DENSE_TILE_W)
        tile.setMinimumWidth(width)
        tile.setMaximumWidth(max(width, scaled_px(self.DENSE_TILE_MAX_W)))
        tile.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        return self._wire_tile(tile, key, desc)

    # -- tabs 2..6: one category each -----------------------------------
    def _build_category_tab(self, section: str,
                            entries: List[Tuple[str, str, str]]) -> QWidget:
        from ..preferences import scaled_px
        P = self._P
        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(SPACING["md"], SPACING["sm"],
                               SPACING["md"], SPACING["sm"])
        col.setSpacing(SPACING["sm"])

        # Redundant with the tab label for a sighted user, but it is what
        # a screen reader lands on inside the page and what the
        # category-coverage test reads.
        heading = QLabel(section.upper())
        heading.setStyleSheet(
            "font-family: 'Open Sans', sans-serif; font-weight: 600;"
            "font-size: 11px; letter-spacing: 2px; background: transparent;"
            f"color: {P['fg_muted']};")
        col.addWidget(heading)
        col.addWidget(Divider())

        holder = QWidget()
        grid = QGridLayout(holder)
        grid.setContentsMargins(0, SPACING["xs"], 0, 0)
        grid.setSpacing(SPACING["sm"])
        tiles = [self._make_tall_tile(k, n, d) for k, n, d in entries]
        self._grids.append((holder, grid, tiles, scaled_px(self.TILE_MIN_W),
                            False))
        self._fill_grid(grid, tiles,
                        self._columns_for(self.width(),
                                          scaled_px(self.TILE_MIN_W)),
                        fill=False)
        col.addWidget(holder)
        col.addStretch(1)
        return self._scrolled(page)

    def _make_tall_tile(self, key: str, name: str, desc: str) -> TallTile:
        from ..preferences import scaled_px
        icon = self._icon_provider(key) if self._icon_provider else None
        tile = TallTile(name, desc, icon,
                        width=scaled_px(self.TILE_MIN_W),
                        height=scaled_px(self.TILE_H),
                        icon_px=scaled_px(self.TALL_ICON_PX))
        return self._wire_tile(tile, key, desc)

    # -- shared ---------------------------------------------------------
    def _wire_tile(self, tile, key: str, desc: str):
        self._tile_hints[tile] = desc
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
        scroll.viewport().setAutoFillBackground(False)
        scroll.setWidget(page)
        return scroll

    @staticmethod
    def _fill_grid(grid: QGridLayout, tiles: list, columns: int,
                   fill: bool = True) -> None:
        """(Re)place ``tiles`` into ``columns`` columns, packed to the top.

        Rows get zero stretch and one extra row takes all of it,
        otherwise QGridLayout shares the leftover height between the rows
        and the tiles drift apart down the page.

        :param fill: when True the tiles widen to their column (up to
            their own maximum), so a row reaches both edges instead of
            leaving a ragged gap after each tile — the difference
            between a page that reads as full and one that reads as
            sparse. Fixed-size cards pass False and pack to the left.
        """
        for tile in tiles:
            grid.removeWidget(tile)
        # No alignment flags at all in fill mode: QGridLayout gives an
        # *aligned* item exactly its sizeHint and positions it in the
        # cell, so even Qt.AlignTop alone leaves a 205 px tile sitting in
        # a 262 px column with a gap after it. Unaligned, the item is
        # handed the whole cell; the tile's Fixed vertical policy keeps
        # the height, and its maximumWidth caps how far it stretches.
        align = Qt.Alignment() if fill else (Qt.AlignLeft | Qt.AlignTop)
        rows = 0
        for index, tile in enumerate(tiles):
            rows = index // columns
            grid.addWidget(tile, rows, index % columns, align)
        for row in range(grid.rowCount()):
            grid.setRowStretch(row, 0)
        grid.setRowStretch(rows + 1, 1)
        for column in range(grid.columnCount()):
            if fill:
                grid.setColumnStretch(column, 1 if column < columns else 0)
            else:
                grid.setColumnStretch(column, 0 if column < columns else 1)

    def _columns_for(self, width: int, tile_w: int) -> int:
        """How many ``tile_w``-wide tiles fit beside the aside.

        Recomputed on resize so a narrow window rewraps instead of
        growing a horizontal scrollbar.
        """
        from ..preferences import scaled_px
        available = max(1, width - scaled_px(self.ASIDE_W)
                        - SPACING["xl"] * 2 - SPACING["lg"]
                        - SPACING["md"] * 2 - 4)
        return max(1, min(6, available // (tile_w + SPACING["sm"])))

    def _build_aside(self) -> QWidget:
        from ..preferences import scaled_px
        aside = QWidget()
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

        for panel in (self._queued, self._recent, self._system,
                      self._news, self._totals):
            col.addWidget(panel)
        col.addStretch(1)
        return aside

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
        """The set of running jobs changed — reflect it, or clear it."""
        active = [h for h in self._registry.active() if h.app_key]
        handle = active[0] if active else None
        self._banner.bind(handle)
        if handle is None:
            self._ticker.stop()
        elif not self._ticker.isActive():
            self._ticker.start()

    def refresh(self) -> None:
        """Re-read everything that can change while Home is off screen."""
        self._queued.refresh()
        self._recent.refresh()
        self._system.refresh()
        self._totals.refresh()
        self._on_runs_changed()

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
        for _holder, grid, tiles, tile_w, fill in self._grids:
            self._fill_grid(grid, tiles,
                            self._columns_for(self.width(), tile_w),
                            fill=fill)

    def eventFilter(self, obj, event):          # noqa: N802
        if event.type() == QEvent.Enter:
            hint = self._tile_hints.get(obj)
            if hint:
                self._hint_bar.setText(hint)
        elif event.type() == QEvent.Leave:
            self._hint_bar.setText(_DEFAULT_HINT)
        return super().eventFilter(obj, event)

    def closeEvent(self, event):                # noqa: N802
        try:
            self._registry.changed.disconnect(self._on_runs_changed)
        except (RuntimeError, TypeError):
            pass
        self._ticker.stop()
        super().closeEvent(event)


def _tab_qss(P: dict) -> str:
    return f"""
/* The pane keeps the PAGE background, not a surface colour: the tiles
   are themselves drawn on `surface`, and a surface pane behind them
   would erase the only separation they have. All the pane contributes
   is an edge, so the tab strip has something to sit on. */
QTabWidget#HomeTabs::pane {{
    border: 1px solid {P['border_soft']};
    border-radius: 8px;
    background: {P['bg']};
    top: -1px;
}}
QTabWidget#HomeTabs > QTabBar::tab {{
    background: transparent;
    color: {P['fg_muted']};
    border: 1px solid transparent;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 7px 14px;
    margin-right: 2px;
    font-size: 13px;
}}
QTabWidget#HomeTabs > QTabBar::tab:hover {{
    color: {P['fg']};
    background: {P['surface_alt']};
}}
QTabWidget#HomeTabs > QTabBar::tab:selected {{
    color: {P['accent']};
    background: {P['surface']};
    border: 1px solid {P['border_soft']};
    border-bottom-color: {P['surface']};
}}
"""
