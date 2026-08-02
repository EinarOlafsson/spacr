"""Real Qt widgets the thirty Home-screen variants are built from.

Everything here is a genuine widget — the ones that already exist in
spaCR (``HTile``, ``Card``, ``Section``, ``Divider``, ``UsageBar``,
``ElidingLabel``, the real ``Sidebar``) are imported and used as-is; the
ones a variant proposes but the app does not have yet (a recent-runs
strip, a resume banner, a guided quick-start, a project status bar, a
what's-new panel, a big illustrated tile) are built here as minimal but
real widgets, so a layout that wins the review is known-buildable.

No mockups. No painted screenshots.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFont, QFontMetrics, QTextLayout
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import common
from common import MOCK, Ctx, blurb_of, name_of


# ---------------------------------------------------------------------------
# Text that never clips
# ---------------------------------------------------------------------------

def elide_to_lines(text: str, font: QFont, width: int, lines: int) -> str:
    """Shorten ``text`` so it lays out in at most ``lines`` at ``width`` px.

    Uses :class:`QTextLayout` — the same line breaker QLabel uses — so
    the answer is exact rather than a character-count guess. Text
    clipping is the exact defect the home-screen rework was raised to
    fix, so every wrapped blurb in these variants goes through here.
    """
    if width <= 0 or lines <= 0 or not text:
        return text
    layout = QTextLayout(text, font)
    layout.beginLayout()
    starts: List[int] = []
    while True:
        line = layout.createLine()
        if not line.isValid():
            break
        line.setLineWidth(width)
        starts.append(line.textStart())
        if len(starts) > lines:
            break
    layout.endLayout()
    if len(starts) <= lines:
        return text
    cut = starts[lines - 1]
    fm = QFontMetrics(font)
    return text[:cut] + fm.elidedText(text[cut:], Qt.ElideRight, width)


def line_count(text: str, font: QFont, width: int) -> int:
    """How many lines ``text`` occupies when wrapped at ``width`` px."""
    if width <= 0 or not text:
        return 1
    layout = QTextLayout(text, font)
    layout.beginLayout()
    n = 0
    while True:
        line = layout.createLine()
        if not line.isValid():
            break
        line.setLineWidth(width)
        n += 1
    layout.endLayout()
    return max(1, n)


def wrapped(ctx: Ctx, text: str, width: int, lines: int, *,
            color: Optional[str] = None, size: int = 12,
            weight: int = 300, reserve: bool = False) -> QLabel:
    """A word-wrapped label with a guaranteed line budget.

    ``lines`` is a *ceiling*: longer text is elided to fit it, and the
    label is then sized to the lines it actually uses, so a one-line
    string in a three-line budget does not leave a hole. Pass
    ``reserve=True`` where a fixed height matters (grid cells that must
    line up).
    """
    lbl = QLabel()
    lbl.setStyleSheet(
        f"color: {color or ctx.P['fg_muted']}; font-size: {size}px;"
        f"font-weight: {weight}; background: transparent;")
    lbl.setWordWrap(True)
    lbl.setFixedWidth(width)
    lbl.ensurePolished()
    shown = elide_to_lines(text, lbl.font(), width - 2, lines)
    lbl.setText(shown)
    fm = QFontMetrics(lbl.font())
    used = lines if reserve else min(lines,
                                     line_count(shown, lbl.font(), width - 2))
    lbl.setFixedHeight(fm.lineSpacing() * used + 2)
    lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    return lbl


def text_label(ctx: Ctx, text: str, *, size: int = 13, weight: int = 400,
               color: Optional[str] = None, tracking: str = "0px",
               upper: bool = False) -> QLabel:
    """A plain styled label (no wrapping, sized to its own text)."""
    lbl = QLabel(text.upper() if upper else text)
    lbl.setStyleSheet(
        f"color: {color or ctx.P['fg']}; font-size: {size}px;"
        f"font-weight: {weight}; letter-spacing: {tracking};"
        "background: transparent;")
    return lbl


# ---------------------------------------------------------------------------
# Page-level QSS for the widgets invented here
# ---------------------------------------------------------------------------

def extra_qss(ctx: Ctx) -> str:
    """Stylesheet for the object names this module introduces."""
    P = ctx.P
    # The Space theme paints its sky on QMainWindow, which these pages
    # are not. Reproduce the offline fallback sky (a deep-space
    # gradient) so a Space render is not a flat near-black rectangle.
    # These renders never load the generated star image — it is cached
    # per user and would make the output non-deterministic.
    page_bg = P["bg"] if ctx.theme != "space" else (
        "qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, "
        f"stop: 0 {P['surface']}, stop: 0.55 {P['bg']}, "
        f"stop: 1 {P['accent_soft']})")
    return f"""
QWidget#Page {{ background: {page_bg}; }}
QWidget#Transparent {{ background: transparent; }}

QFrame#Panel {{
    background: {P['surface_alt']};
    border: 1px solid {P['border_soft']};
    border-radius: 10px;
}}
QFrame#PanelAccent {{
    background: {P['accent_soft']};
    border: 1px solid {P['accent_lo']};
    border-radius: 10px;
}}
QFrame#PanelPlain {{
    background: {P['surface']};
    border: 1px solid {P['border_soft']};
    border-radius: 10px;
}}

QPushButton#BigTile {{
    background: {P['surface']};
    border: 1px solid {P['border_soft']};
    border-radius: 12px;
    padding: 0px;
    text-align: center;
}}
QPushButton#BigTile:hover {{
    background: {P['surface_hi']};
    border: 1px solid {P['accent']};
}}
QPushButton#BigTileAccent {{
    background: {P['accent_soft']};
    border: 1px solid {P['accent_lo']};
    border-radius: 12px;
    padding: 0px;
}}

QPushButton#DenseRow {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 2px 6px;
    text-align: left;
    min-height: 0px;
}}
QPushButton#DenseRow:hover {{
    background: {P['surface_alt']};
    border: 1px solid {P['border_soft']};
}}

QPushButton#Chip {{
    background: {P['surface_alt']};
    border: 1px solid {P['border_soft']};
    border-radius: 14px;
    padding: 4px 12px;
    color: {P['fg_muted']};
    font-size: 12px;
    font-weight: 500;
    min-height: 0px;
}}
QPushButton#Chip:hover {{ color: {P['fg']}; border-color: {P['border']}; }}
QPushButton#ChipOn {{
    background: {P['accent_soft']};
    border: 1px solid {P['accent']};
    border-radius: 14px;
    padding: 4px 12px;
    color: {P['accent']};
    font-size: 12px;
    font-weight: 600;
    min-height: 0px;
}}

QLabel#Kbd {{
    background: {P['surface_hi']};
    border: 1px solid {P['border']};
    border-radius: 4px;
    color: {P['fg_muted']};
    font-family: "JetBrains Mono", monospace;
    font-size: 10px;
    padding: 1px 5px;
}}

QLineEdit#Search {{
    background: {P['surface_alt']};
    border: 1px solid {P['border']};
    border-radius: 8px;
    padding: 8px 12px;
    color: {P['fg']};
    font-size: 14px;
}}
QLineEdit#SearchBig {{
    background: {P['surface_alt']};
    border: 1px solid {P['border']};
    border-radius: 12px;
    padding: 14px 18px;
    color: {P['fg']};
    font-size: 19px;
}}

QWidget#MenuStrip {{
    background: {P['surface']};
    border-bottom: 1px solid {P['border_soft']};
}}
QWidget#StatusStrip {{
    background: {P['surface']};
    border-top: 1px solid {P['border_soft']};
}}
QWidget#RailBg {{
    background: {P['surface']};
    border-right: 1px solid {P['border_soft']};
}}

QListWidget#CatRail {{
    background: transparent;
    border: none;
    outline: none;
    font-size: 14px;
}}
QListWidget#CatRail::item {{
    color: {P['fg_muted']};
    padding: 9px 12px;
    border-radius: 6px;
    margin: 1px 4px;
}}
QListWidget#CatRail::item:selected {{
    background: {P['accent_soft']};
    color: {P['accent']};
}}

QTabWidget::pane {{
    border: 1px solid {P['border_soft']};
    border-radius: 8px;
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {P['fg_muted']};
    padding: 8px 18px;
    margin-right: 2px;
    border: 1px solid transparent;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-size: 13px;
    font-weight: 500;
}}
QTabBar::tab:selected {{
    background: {P['surface_alt']};
    color: {P['accent']};
    border: 1px solid {P['border_soft']};
    border-bottom-color: {P['surface_alt']};
}}
"""


# ---------------------------------------------------------------------------
# Page scaffold
# ---------------------------------------------------------------------------

class Page(QWidget):
    """A 1440x900 home-screen canvas with optional window chrome.

    :param ctx: the theme context.
    :param chrome: draw the app's menu strip + status bar, so the space
        a variant actually gets on a 1440x900 laptop is what is rendered.
    :param margins: content margins of the body area.
    """

    def __init__(self, ctx: Ctx, *, chrome: bool = True,
                 margins: Tuple[int, int, int, int] = (28, 22, 28, 18),
                 spacing: int = 16):
        super().__init__()
        self.ctx = ctx
        self.setObjectName("Page")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(extra_qss(ctx))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        if chrome:
            outer.addWidget(menu_strip(ctx))

        self.middle = QHBoxLayout()
        self.middle.setContentsMargins(0, 0, 0, 0)
        self.middle.setSpacing(0)
        outer.addLayout(self.middle, 1)

        self._content = QWidget()
        self._content.setObjectName("Transparent")
        self.body = QVBoxLayout(self._content)
        self.body.setContentsMargins(*margins)
        self.body.setSpacing(spacing)
        self.middle.addWidget(self._content, 1)

        self._outer = outer
        self._chrome = chrome

    def add_rail(self, rail: QWidget) -> None:
        """Insert a navigation rail to the left of the content area."""
        self.middle.insertWidget(0, rail)

    def add_aside(self, aside: QWidget) -> None:
        """Insert a column to the right of the content area."""
        self.middle.addWidget(aside)

    def finish(self, footer: Optional[QWidget] = None,
               status: Optional[str] = None) -> "Page":
        """Attach the optional footer + status bar. Returns self."""
        if footer is not None:
            self._outer.addWidget(footer)
        if self._chrome:
            self._outer.addWidget(status_bar(self.ctx, status or "Ready"))
        return self


def menu_strip(ctx: Ctx) -> QWidget:
    """The app's menu bar, as a real (non-interactive) strip."""
    w = QWidget()
    w.setObjectName("MenuStrip")
    w.setAttribute(Qt.WA_StyledBackground, True)
    w.setFixedHeight(26)
    row = QHBoxLayout(w)
    row.setContentsMargins(10, 0, 10, 0)
    row.setSpacing(18)
    for item in ("spaCR", "Demos", "Help"):
        row.addWidget(text_label(ctx, item, size=12,
                                 color=ctx.P["fg_muted"], weight=500))
    row.addStretch(1)
    return w


def status_bar(ctx: Ctx, message: str = "Ready") -> QWidget:
    """The app's status bar: transient message left, app + version right."""
    w = QWidget()
    w.setObjectName("StatusStrip")
    w.setAttribute(Qt.WA_StyledBackground, True)
    w.setFixedHeight(24)
    row = QHBoxLayout(w)
    row.setContentsMargins(10, 0, 10, 0)
    row.setSpacing(14)
    row.addWidget(text_label(ctx, message, size=11, color=ctx.P["fg_muted"]))
    row.addStretch(1)
    row.addWidget(text_label(ctx, "Home", size=11, color=ctx.P["fg_muted"]))
    row.addWidget(text_label(ctx, f"spaCR {MOCK['version']}", size=11,
                             color=ctx.P["fg_dim"], tracking="0.6px"))
    return w


def hint_bar(ctx: Ctx, text: str = "Hover a tile to see what it does."
             ) -> QLabel:
    """The sticky bottom hint bar the current home screen ships with."""
    lbl = QLabel(text)
    lbl.setObjectName("HintBar")
    lbl.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
    lbl.setFixedHeight(34)
    return lbl


def transparent(spacing: int = 0, margins=(0, 0, 0, 0), horizontal=False):
    """A bare container widget plus its layout (background transparent)."""
    w = QWidget()
    w.setObjectName("Transparent")
    lay = QHBoxLayout(w) if horizontal else QVBoxLayout(w)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    return w, lay


def panel(ctx: Ctx, *, accent: bool = False, plain: bool = False,
          margins=(14, 12, 14, 12), spacing: int = 8, horizontal=False):
    """A rounded surface panel plus its layout."""
    frame = QFrame()
    frame.setObjectName("PanelAccent" if accent
                        else "PanelPlain" if plain else "Panel")
    lay = QHBoxLayout(frame) if horizontal else QVBoxLayout(frame)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    return frame, lay


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------

def hero(ctx: Ctx, *, compact: bool = False) -> QWidget:
    """The current home hero: logo + wordmark + one-line pitch."""
    w, row = transparent(horizontal=True, spacing=16)
    logo_px = 56 if compact else 96
    pix = ctx.logo(logo_px)
    if pix is not None:
        lbl = QLabel()
        lbl.setPixmap(pix)
        lbl.setFixedSize(logo_px, logo_px)
        lbl.setStyleSheet("background: transparent;")
        row.addWidget(lbl, 0, Qt.AlignVCenter)
    mark = QLabel("spaCR")
    mark.setStyleSheet(
        "font-family: 'Open Sans', sans-serif; font-weight: 300;"
        f"font-size: {36 if compact else 56}px; color: {ctx.P['accent']};"
        "letter-spacing: -1.2px; background: transparent;")
    row.addWidget(mark, 0, Qt.AlignVCenter)
    sub = wrapped(ctx,
                  "End-to-end microscopy → single-cell measurements → "
                  "genotype-phenotype mapping.",
                  520, 2, color=ctx.P["fg"], size=13)
    row.addWidget(sub, 0, Qt.AlignVCenter)
    row.addStretch(1)
    return w


def top_bar(ctx: Ctx, *, title: str = "spaCR",
            subtitle: str = "",
            actions: Sequence[Tuple[str, bool]] = ()) -> QWidget:
    """A slim brand bar: logo mark + title (+ subtitle) + right actions.

    :param actions: ``(label, primary)`` pairs rendered right-aligned.
    """
    w, row = transparent(horizontal=True, spacing=12)
    pix = ctx.logo(34)
    if pix is not None:
        lbl = QLabel()
        lbl.setPixmap(pix)
        lbl.setFixedSize(34, 34)
        lbl.setStyleSheet("background: transparent;")
        row.addWidget(lbl, 0, Qt.AlignVCenter)
    row.addWidget(text_label(ctx, title, size=25, weight=300,
                             color=ctx.P["accent"], tracking="-0.6px"),
                  0, Qt.AlignVCenter)
    if subtitle:
        row.addWidget(text_label(ctx, subtitle, size=13, weight=300,
                                 color=ctx.P["fg_muted"]), 0, Qt.AlignVCenter)
    row.addStretch(1)
    for label, primary in actions:
        btn = QPushButton(label)
        if primary:
            btn.setObjectName("PrimaryButton")
        btn.setCursor(Qt.PointingHandCursor)
        row.addWidget(btn, 0, Qt.AlignVCenter)
    return w


def cat_header(ctx: Ctx, text: str, *, rule: bool = True,
               note: str = "", size: int = 11) -> QWidget:
    """A category heading: tracked small caps, optional hairline rule."""
    from spacr.qt.widgets.divider import Divider
    w, col = transparent(spacing=5)
    line, row = transparent(horizontal=True, spacing=10)
    row.addWidget(text_label(ctx, text, size=size, weight=600,
                             color=ctx.P["fg_muted"], tracking="2px",
                             upper=True))
    if note:
        row.addWidget(text_label(ctx, note, size=11, weight=400,
                                 color=ctx.P["fg_dim"]))
    row.addStretch(1)
    col.addWidget(line)
    if rule:
        col.addWidget(Divider())
    return w


def plain_header(ctx: Ctx, text: str, note: str = "") -> QWidget:
    """A sentence-case heading (used where small caps read as shouting)."""
    w, row = transparent(horizontal=True, spacing=10)
    row.addWidget(text_label(ctx, text, size=17, weight=500,
                             color=ctx.P["fg"], tracking="-0.2px"))
    if note:
        row.addWidget(text_label(ctx, note, size=12, color=ctx.P["fg_dim"]))
    row.addStretch(1)
    return w


def search_box(ctx: Ctx, placeholder: str, *, big: bool = False,
               width: int = 0) -> QLineEdit:
    """A real search field (the search-first variants' entry point)."""
    edit = QLineEdit()
    edit.setObjectName("SearchBig" if big else "Search")
    edit.setPlaceholderText(placeholder)
    edit.setClearButtonEnabled(False)
    if width:
        edit.setFixedWidth(width)
    return edit


def kbd(ctx: Ctx, text: str) -> QLabel:
    """A keycap chip, e.g. ``Ctrl+1``."""
    lbl = QLabel(text)
    lbl.setObjectName("Kbd")
    lbl.setAlignment(Qt.AlignCenter)
    return lbl


def chip(ctx: Ctx, text: str, on: bool = False) -> QPushButton:
    """A pill-shaped filter chip."""
    btn = QPushButton(text)
    btn.setObjectName("ChipOn" if on else "Chip")
    btn.setCursor(Qt.PointingHandCursor)
    btn.setFlat(True)
    return btn


# ---------------------------------------------------------------------------
# App surfaces
# ---------------------------------------------------------------------------

#: Height an ``HTile`` needs for a given icon: the QSS pads it 12 px top
#: and bottom, and the icon is the tallest thing inside.
def htile_height(icon_px: int) -> int:
    """Minimum height an ``HTile`` needs so its icon is not cropped."""
    return max(72, icon_px + 28)


#: Narrowest an ``HTile`` may be drawn before "Annotator Agreement"
#: elides, measured by bisection (see the module notes in VARIANTS.md).
#: ``{name font px: {icon px: min width}}``; the shipped tile uses the
#: 17 px subtitle size and therefore needs 255-263 px.
HTILE_MIN_WIDTH = {
    0:  {32: 243, 36: 247, 40: 251, 44: 255, 52: 263},
    12: {32: 192, 36: 196, 40: 200},
    13: {32: 202, 36: 206, 40: 210},
    14: {32: 212, 36: 216, 40: 220},
}


def htile(ctx: Ctx, key: str, *, width: int, height: int = 0,
          icon_px: int = 44, name_px: int = 0) -> QWidget:
    """One of the app's real ``HTile`` cards, at a fixed width.

    :param name_px: override the tile name's font size. The shipped
        tile draws it at the 17 px "subtitle" size, which is what forces
        a 255 px minimum width and therefore at most five columns on a
        1440 px screen; the compact variants restyle that one label.
    """
    from spacr.qt.widgets.tile import HTile
    tile = HTile(text=name_of(key), description="", icon=ctx.icon(key),
                 icon_size=icon_px)
    if name_px:
        tile._name_lbl.setStyleSheet(
            f"font-size: {name_px}px; font-weight: 500;"
            f"color: {ctx.P['fg']}; background: transparent;")
        tile._name_lbl.setText(name_of(key))
    tile.setFixedWidth(width)
    tile.setFixedHeight(height or htile_height(icon_px))
    tile.setToolTip(f"{name_of(key)} — {blurb_of(key)}")
    return tile


def htile_grid(ctx: Ctx, keys: Sequence[str], *, cols: int,
               width: int, hspace: int = 8, vspace: int = 8,
               icon_px: int = 44, height: int = 0,
               name_px: int = 0) -> QWidget:
    """A wrapping grid of real ``HTile`` cards."""
    w = QWidget()
    w.setObjectName("Transparent")
    grid = QGridLayout(w)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(hspace)
    grid.setVerticalSpacing(vspace)
    for i, key in enumerate(keys):
        grid.addWidget(htile(ctx, key, width=width, icon_px=icon_px,
                             height=height, name_px=name_px),
                       i // cols, i % cols, Qt.AlignLeft | Qt.AlignTop)
    grid.setColumnStretch(cols, 1)
    return w


class FixedButton(QPushButton):
    """A QPushButton whose size the app stylesheet cannot take away.

    ``setFixedSize`` alone is not enough here: the app QSS carries
    ``QPushButton { min-height: 22px }``, and ``QStyleSheetStyle``
    re-applies that rule's geometry to the widget on polish, wiping the
    minimum a ``setFixedSize`` had set. The layout then reads a 40 px
    minimum and squashes a 116 px tile to 48 px — which is exactly the
    kind of silent clipping these renders exist to catch. Reporting the
    size through ``sizeHint``/``minimumSizeHint`` survives the restyle.
    """

    def __init__(self, width: int, height: int, parent=None):
        super().__init__(parent)
        self._fixed = QSize(int(width), int(height))
        self.setFixedSize(self._fixed)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def sizeHint(self) -> QSize:               # noqa: N802 (Qt casing)
        """The size asked for at construction, always."""
        return QSize(self._fixed)

    def minimumSizeHint(self) -> QSize:        # noqa: N802
        """Same as :meth:`sizeHint` — the tile does not shrink."""
        return QSize(self._fixed)


class BigTile(FixedButton):
    """A large illustrated launcher tile: icon above name above blurb.

    Does not exist in spaCR today — this is the widget the
    "large illustrated tiles" variants propose, built for real so the
    layout can be judged on how it actually renders.
    """

    def __init__(self, ctx: Ctx, key: str, *, width: int, height: int,
                 icon_px: int = 56, blurb_lines: int = 0,
                 accent: bool = False, badge: str = ""):
        super().__init__(width, height)
        self.setObjectName("BigTileAccent" if accent else "BigTile")
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(f"{name_of(key)} — {blurb_of(key)}")

        col = QVBoxLayout(self)
        col.setContentsMargins(10, 10, 10, 10)
        col.setSpacing(6)
        col.addStretch(1)

        icon = QLabel()
        icon.setPixmap(ctx.pixmap(key, icon_px))
        icon.setFixedSize(icon_px, icon_px)
        icon.setStyleSheet("background: transparent;")
        col.addWidget(icon, 0, Qt.AlignHCenter)

        from spacr.qt.widgets.eliding import ElidingLabel
        nm = ElidingLabel(name_of(key))
        nm.setAlignment(Qt.AlignHCenter)
        nm.setFixedWidth(width - 20)
        nm.setStyleSheet(
            f"color: {ctx.P['fg']}; font-size: 13px; font-weight: 500;"
            "background: transparent;")
        col.addWidget(nm, 0, Qt.AlignHCenter)
        self._name = nm

        if blurb_lines:
            body = wrapped(ctx, blurb_of(key), width - 24, blurb_lines,
                           size=11, reserve=True)
            body.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
            col.addWidget(body, 0, Qt.AlignHCenter)
        if badge:
            col.addWidget(text_label(ctx, badge, size=10, weight=600,
                                     color=ctx.P["fg_dim"], tracking="0.6px"),
                          0, Qt.AlignHCenter)
        col.addStretch(1)


def big_tile_grid(ctx: Ctx, keys: Sequence[str], *, cols: int, width: int,
                  height: int, icon_px: int = 56, blurb_lines: int = 0,
                  hspace: int = 10, vspace: int = 10,
                  badges: Optional[dict] = None) -> QWidget:
    """A grid of :class:`BigTile`."""
    w = QWidget()
    w.setObjectName("Transparent")
    grid = QGridLayout(w)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(hspace)
    grid.setVerticalSpacing(vspace)
    for i, key in enumerate(keys):
        grid.addWidget(
            BigTile(ctx, key, width=width, height=height, icon_px=icon_px,
                    blurb_lines=blurb_lines,
                    badge=(badges or {}).get(key, "")),
            i // cols, i % cols, Qt.AlignLeft | Qt.AlignTop)
    grid.setColumnStretch(cols, 1)
    return w


class DenseRow(QPushButton):
    """A compact one-line app row: small icon, name, blurb, optional badge.

    The "dense list" answer to the tile grid — many more apps above the
    fold, at the cost of the tile's visual weight.
    """

    def __init__(self, ctx: Ctx, key: str, *, width: int,
                 name_width: int = 136, icon_px: int = 20,
                 show_blurb: bool = True, badge: str = "",
                 shortcut: str = ""):
        super().__init__()
        self.setObjectName("DenseRow")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedWidth(width)
        self.setFixedHeight(30)
        self.setToolTip(f"{name_of(key)} — {blurb_of(key)}")

        row = QHBoxLayout(self)
        row.setContentsMargins(6, 0, 6, 0)
        row.setSpacing(9)

        icon = QLabel()
        icon.setPixmap(ctx.pixmap(key, icon_px))
        icon.setFixedSize(icon_px, icon_px)
        icon.setStyleSheet("background: transparent;")
        row.addWidget(icon, 0, Qt.AlignVCenter)

        from spacr.qt.widgets.eliding import ElidingLabel
        nm = ElidingLabel(name_of(key))
        nm.setFixedWidth(name_width)
        nm.setStyleSheet(
            f"color: {ctx.P['fg']}; font-size: 13px; font-weight: 500;"
            "background: transparent;")
        row.addWidget(nm, 0, Qt.AlignVCenter)
        self._name = nm

        if show_blurb:
            rest = width - 12 - icon_px - name_width - 9 * 3 - (
                60 if (badge or shortcut) else 0)
            if rest > 40:
                blurb = QLabel()
                blurb.setFixedWidth(rest)
                blurb.setStyleSheet(
                    f"color: {ctx.P['fg_muted']}; font-size: 11px;"
                    "font-weight: 300; background: transparent;")
                blurb.ensurePolished()
                fm = QFontMetrics(blurb.font())
                blurb.setText(fm.elidedText(blurb_of(key), Qt.ElideRight,
                                            rest - 2))
                row.addWidget(blurb, 0, Qt.AlignVCenter)
        row.addStretch(1)
        if badge:
            row.addWidget(text_label(ctx, badge, size=11, weight=500,
                                     color=ctx.P["fg_dim"]), 0,
                          Qt.AlignVCenter)
        if shortcut:
            row.addWidget(kbd(ctx, shortcut), 0, Qt.AlignVCenter)


def dense_list(ctx: Ctx, keys: Sequence[str], *, width: int,
               name_width: int = 136, show_blurb: bool = True,
               badges: Optional[dict] = None,
               shortcuts: Optional[dict] = None, spacing: int = 1
               ) -> QWidget:
    """A vertical stack of :class:`DenseRow`."""
    w = QWidget()
    w.setObjectName("Transparent")
    col = QVBoxLayout(w)
    col.setContentsMargins(0, 0, 0, 0)
    col.setSpacing(spacing)
    for key in keys:
        col.addWidget(DenseRow(ctx, key, width=width, name_width=name_width,
                               show_blurb=show_blurb,
                               badge=(badges or {}).get(key, ""),
                               shortcut=(shortcuts or {}).get(key, "")))
    return w


# ---------------------------------------------------------------------------
# Elements that do not exist on the home screen today
# ---------------------------------------------------------------------------

def resume_banner(ctx: Ctx, *, width: int = 0) -> QWidget:
    """ADDED: "pick up where you left off" — the single biggest button.

    Names the last run and offers to resume it, open its output, or
    start the next stage.
    """
    app_key, plate, when = MOCK["last_run"]
    frame, row = panel(ctx, accent=True, horizontal=True,
                       margins=(18, 14, 18, 14), spacing=16)
    if width:
        frame.setFixedWidth(width)
    icon = QLabel()
    icon.setPixmap(ctx.pixmap(app_key, 40))
    icon.setFixedSize(40, 40)
    icon.setStyleSheet("background: transparent;")
    row.addWidget(icon, 0, Qt.AlignVCenter)

    txt, col = transparent(spacing=2)
    col.addWidget(text_label(ctx, f"Resume {name_of(app_key)} on {plate}",
                             size=16, weight=600))
    col.addWidget(text_label(ctx, f"{when} · 4 of 12 plates done · "
                                  "next stage is Annotate",
                             size=12, weight=300, color=ctx.P["fg_muted"]))
    row.addWidget(txt, 1, Qt.AlignVCenter)

    resume = QPushButton("Resume run")
    resume.setObjectName("PrimaryButton")
    resume.setCursor(Qt.PointingHandCursor)
    row.addWidget(resume, 0, Qt.AlignVCenter)
    out = QPushButton("Open output")
    row.addWidget(out, 0, Qt.AlignVCenter)
    return frame


def recent_runs_strip(ctx: Ctx, *, count: int = 3, card_width: int = 300,
                      height: int = 92) -> QWidget:
    """ADDED: the last few runs as resumable cards."""
    w = QWidget()
    w.setObjectName("Transparent")
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(10)
    for key, plate, when, ok, elapsed in MOCK["recent"][:count]:
        frame, col = panel(ctx, margins=(12, 10, 12, 10), spacing=5)
        frame.setFixedSize(card_width, height)
        head, hrow = transparent(horizontal=True, spacing=8)
        icon = QLabel()
        icon.setPixmap(ctx.pixmap(key, 20))
        icon.setFixedSize(20, 20)
        icon.setStyleSheet("background: transparent;")
        hrow.addWidget(icon, 0, Qt.AlignVCenter)
        hrow.addWidget(text_label(ctx, f"{name_of(key)} · {plate}", size=13,
                                  weight=500), 0, Qt.AlignVCenter)
        hrow.addStretch(1)
        hrow.addWidget(text_label(ctx, "✓" if ok else "✗", size=13, weight=700,
                                  color=ctx.P["success"] if ok
                                  else ctx.P["error"]), 0, Qt.AlignVCenter)
        col.addWidget(head)
        col.addWidget(text_label(ctx, f"{when} · {elapsed}", size=11,
                                 weight=300, color=ctx.P["fg_muted"]))
        foot, frow = transparent(horizontal=True, spacing=6)
        frow.addWidget(chip(ctx, "Resume"))
        frow.addWidget(chip(ctx, "Settings"))
        frow.addStretch(1)
        col.addWidget(foot)
        row.addWidget(frame)
    row.addStretch(1)
    return w


def recent_runs_list(ctx: Ctx, *, count: int = 4, width: int = 320) -> QWidget:
    """ADDED: the same history as a narrow vertical list."""
    frame, col = panel(ctx, margins=(14, 12, 14, 12), spacing=7)
    frame.setFixedWidth(width)
    col.addWidget(text_label(ctx, "Recent runs", size=11, weight=600,
                             color=ctx.P["fg_muted"], tracking="2px",
                             upper=True))
    for key, plate, when, ok, elapsed in MOCK["recent"][:count]:
        row_w, row = transparent(horizontal=True, spacing=8)
        row.addWidget(text_label(ctx, "●" if ok else "○", size=12,
                                 color=ctx.P["success"] if ok
                                 else ctx.P["error"]))
        row.addWidget(text_label(ctx, name_of(key), size=12, weight=500))
        row.addWidget(text_label(ctx, plate, size=11, weight=300,
                                 color=ctx.P["fg_muted"]))
        row.addStretch(1)
        row.addWidget(text_label(ctx, when, size=11, weight=300,
                                 color=ctx.P["fg_dim"]))
        col.addWidget(row_w)
    col.addStretch(1)
    return frame


def project_status_strip(ctx: Ctx) -> QWidget:
    """ADDED: which dataset is open, how big it is, what state it is in."""
    frame, row = panel(ctx, plain=True, horizontal=True,
                       margins=(16, 10, 16, 10), spacing=22)
    frame.setFixedHeight(52)
    row.addWidget(text_label(ctx, "PROJECT", size=10, weight=600,
                             color=ctx.P["fg_dim"], tracking="1.6px"))
    row.addWidget(text_label(ctx, MOCK["project"], size=14, weight=600,
                             color=ctx.P["accent"]))
    for value in (MOCK["plates"], MOCK["images"], MOCK["objects"]):
        row.addWidget(_dot(ctx))
        row.addWidget(text_label(ctx, value, size=12, weight=400,
                                 color=ctx.P["fg_muted"]))
    row.addStretch(1)
    row.addWidget(text_label(ctx, "measurements.db 4.1 GB", size=11,
                             weight=300, color=ctx.P["fg_dim"]))
    switch = QPushButton("Switch project…")
    switch.setCursor(Qt.PointingHandCursor)
    row.addWidget(switch)
    return frame


def _dot(ctx: Ctx) -> QLabel:
    return text_label(ctx, "·", size=13, color=ctx.P["fg_dim"])


def system_panel(ctx: Ctx, *, width: int = 300, title: str = "System"
                 ) -> QWidget:
    """ADDED: disk + GPU state, built on the real ``UsageBar`` widget."""
    from spacr.qt.widgets.usage_bar import UsageBar
    frame, col = panel(ctx, margins=(14, 12, 14, 12), spacing=6)
    frame.setFixedWidth(width)
    col.addWidget(text_label(ctx, title, size=11, weight=600,
                             color=ctx.P["fg_muted"], tracking="2px",
                             upper=True))
    for label, pct, note in MOCK["system"]:
        bar = UsageBar(label)
        bar.set_value(pct)
        col.addWidget(bar)
    col.addWidget(text_label(ctx, "RTX 4090 · 1.2 TB free", size=11,
                             weight=300, color=ctx.P["fg_dim"]))
    col.addStretch(1)
    return frame


def whats_new_panel(ctx: Ctx, *, width: int = 320, items: int = 4) -> QWidget:
    """ADDED: what changed in this version, so an upgrade is discoverable."""
    frame, col = panel(ctx, margins=(14, 12, 14, 12), spacing=7)
    frame.setFixedWidth(width)
    head, hrow = transparent(horizontal=True, spacing=8)
    hrow.addWidget(text_label(ctx, "New in " + MOCK["version"], size=11,
                              weight=600, color=ctx.P["fg_muted"],
                              tracking="2px", upper=True))
    hrow.addStretch(1)
    col.addWidget(head)
    inner = width - 28
    for line in MOCK["whats_new"][:items]:
        row_w, row = transparent(horizontal=True, spacing=8)
        row.setAlignment(Qt.AlignTop)
        row.addWidget(text_label(ctx, "•", size=12, color=ctx.P["accent"]))
        row.addWidget(wrapped(ctx, line, inner - 18, 2, size=11))
        col.addWidget(row_w)
    col.addStretch(1)
    return frame


def quick_start(ctx: Ctx, *, width: int = 0, steps: Sequence[
        Tuple[str, str, str]] = ()) -> QWidget:
    """ADDED: a guided first-run path — three numbered steps with buttons."""
    steps = steps or (
        ("1", "Point spaCR at your images",
         "A Yokogawa/ND2/CZI folder, or import someone else's project."),
        ("2", "Segment and measure",
         "Mask makes the objects, Measure turns them into a table."),
        ("3", "Call your hits",
         "Annotate, classify, then map barcodes to scores."),
    )
    buttons = (("Choose folder…", True), ("Run Mask → Measure", True),
               ("Open Annotate", False))
    w = QWidget()
    w.setObjectName("Transparent")
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(12)
    card_w = 348
    for (num, title, body), (blabel, primary) in zip(steps, buttons):
        frame, col = panel(ctx, margins=(16, 14, 16, 14), spacing=8)
        frame.setFixedSize(card_w, 168)
        top, trow = transparent(horizontal=True, spacing=10)
        badge = QLabel(num)
        badge.setFixedSize(26, 26)
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(
            f"background: {ctx.P['accent_soft']}; color: {ctx.P['accent']};"
            f"border: 1px solid {ctx.P['accent_lo']}; border-radius: 13px;"
            "font-size: 13px; font-weight: 700;")
        trow.addWidget(badge, 0, Qt.AlignTop)
        trow.addWidget(wrapped(ctx, title, card_w - 32 - 36, 2, size=15,
                               weight=600, color=ctx.P["fg"]), 1)
        col.addWidget(top)
        col.addWidget(wrapped(ctx, body, card_w - 32, 3, size=12))
        col.addStretch(1)
        btn = QPushButton(blabel)
        if primary:
            btn.setObjectName("PrimaryButton")
        btn.setCursor(Qt.PointingHandCursor)
        col.addWidget(btn, 0, Qt.AlignLeft)
        row.addWidget(frame)
    row.addStretch(1)
    return w


def start_run_panel(ctx: Ctx, *, width: int = 0, height: int = 210
                    ) -> QWidget:
    """ADDED: one prominent "start a run" path — folder, pipeline, Run."""
    frame, col = panel(ctx, margins=(22, 18, 22, 18), spacing=12)
    if width:
        frame.setFixedWidth(width)
    frame.setFixedHeight(height)
    col.addWidget(text_label(ctx, "Start a run", size=22, weight=500,
                             tracking="-0.3px"))
    col.addWidget(text_label(
        ctx, "Everything else on this page is optional.", size=12,
        weight=300, color=ctx.P["fg_muted"]))

    form, frow = transparent(horizontal=True, spacing=10)
    src = search_box(ctx, "/data/toxo_mito_screen/plate_08")
    src.setFixedWidth(430)
    frow.addWidget(src)
    browse = QPushButton("Browse…")
    browse.setCursor(Qt.PointingHandCursor)
    frow.addWidget(browse)
    frow.addStretch(1)
    col.addWidget(form)

    pipe, prow = transparent(horizontal=True, spacing=8)
    prow.addWidget(text_label(ctx, "Pipeline", size=12, weight=500,
                              color=ctx.P["fg_muted"]))
    for label, on in (("Mask", True), ("Measure", True), ("Annotate", False),
                      ("Classify", False), ("Regression", False)):
        prow.addWidget(chip(ctx, label, on=on))
    prow.addStretch(1)
    run = QPushButton("Run")
    run.setObjectName("PrimaryButton")
    run.setCursor(Qt.PointingHandCursor)
    run.setFixedWidth(120)
    prow.addWidget(run)
    col.addWidget(pipe)
    col.addStretch(1)
    return frame


def pinned_row(ctx: Ctx, keys: Sequence[str] = (), *, tile_w: int = 150,
               tile_h: int = 108) -> QWidget:
    """ADDED: a pinned-favourites row the user curates themselves."""
    keys = list(keys or common.PINNED)
    w = QWidget()
    w.setObjectName("Transparent")
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(10)
    for key in keys:
        row.addWidget(BigTile(ctx, key, width=tile_w, height=tile_h,
                              icon_px=40))
    add = FixedButton(52, tile_h)
    add.setText("+")
    add.setObjectName("BigTile")
    add.setToolTip("Pin another app")
    add.setStyleSheet(f"color: {ctx.P['fg_dim']}; font-size: 22px;")
    row.addWidget(add)
    row.addStretch(1)
    return w


def queue_panel(ctx: Ctx, *, width: int = 320) -> QWidget:
    """ADDED: what is queued to run next, straight on the home screen."""
    frame, col = panel(ctx, margins=(14, 12, 14, 12), spacing=7)
    frame.setFixedWidth(width)
    col.addWidget(text_label(ctx, "Queued", size=11, weight=600,
                             color=ctx.P["fg_muted"], tracking="2px",
                             upper=True))
    for plate, pipeline, state in MOCK["queue"]:
        row_w, row = transparent(horizontal=True, spacing=8)
        row.addWidget(text_label(ctx, plate, size=12, weight=500))
        row.addWidget(text_label(ctx, pipeline, size=11, weight=300,
                                 color=ctx.P["fg_muted"]))
        row.addStretch(1)
        row.addWidget(text_label(ctx, state, size=11, weight=300,
                                 color=ctx.P["fg_dim"]))
        col.addWidget(row_w)
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(34)
    bar.setTextVisible(False)
    col.addWidget(bar)
    col.addWidget(text_label(ctx, "plate_07 · Measure · 34 %", size=11,
                             weight=300, color=ctx.P["fg_dim"]))
    col.addStretch(1)
    return frame


def stat_row(ctx: Ctx, stats: Sequence[Tuple[str, str]], *,
             height: int = 74) -> QWidget:
    """ADDED: a row of big-number tiles (runs, plates, objects, models)."""
    w = QWidget()
    w.setObjectName("Transparent")
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(10)
    for value, label in stats:
        frame, col = panel(ctx, margins=(14, 8, 14, 8), spacing=0)
        frame.setFixedHeight(height)
        col.addWidget(text_label(ctx, value, size=24, weight=300,
                                 tracking="-0.5px"))
        col.addWidget(text_label(ctx, label, size=10, weight=600,
                                 color=ctx.P["fg_muted"], tracking="1.4px",
                                 upper=True))
        row.addWidget(frame, 1)
    return w


# ---------------------------------------------------------------------------
# Navigation surfaces
# ---------------------------------------------------------------------------

def real_sidebar(ctx: Ctx) -> QWidget:
    """The app's actual ``Sidebar`` widget, unmodified.

    Used by the baseline variant so the render shows exactly what a
    1440x900 laptop gets today — including the fact that one row per
    registered app plus five section headings does not fit in 900 px,
    so the bottom of the list is simply not reachable.
    """
    from spacr.qt.app import Sidebar
    return Sidebar()


def cat_rail(ctx: Ctx, titles: Sequence[str], *, selected: int = 0,
             width: int = 232, header: str = "",
             counts: Optional[Sequence[int]] = None) -> QWidget:
    """A left rail of CATEGORIES (not apps) beside a content pane."""
    holder = QWidget()
    holder.setObjectName("RailBg")
    holder.setAttribute(Qt.WA_StyledBackground, True)
    holder.setFixedWidth(width)
    col = QVBoxLayout(holder)
    col.setContentsMargins(0, 14, 0, 14)
    col.setSpacing(8)
    if header:
        lbl = text_label(ctx, header, size=11, weight=600,
                         color=ctx.P["fg_dim"], tracking="2px", upper=True)
        lbl.setContentsMargins(16, 0, 16, 0)
        col.addWidget(lbl)
    lst = QListWidget()
    lst.setObjectName("CatRail")
    lst.setFrameShape(QFrame.NoFrame)
    lst.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    lst.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    for i, title in enumerate(titles):
        label = title
        if counts is not None:
            label = f"{title}    {counts[i]}"
        QListWidgetItem(label, lst)
    lst.setCurrentRow(selected)
    col.addWidget(lst, 1)
    return holder


def scroll_area(inner: QWidget, *, horizontal: bool = False) -> QScrollArea:
    """Wrap a widget in a frameless scroll area (vertical by default)."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.NoFrame)
    # Scoped to the scroll area itself. An unscoped `background:
    # transparent` here cascades to every descendant and silently strips
    # the fill off the buttons and panels inside it.
    area.setStyleSheet("QScrollArea { background: transparent; "
                       "border: none; }")
    area.viewport().setObjectName("ScrollViewport")
    area.viewport().setStyleSheet(
        "QWidget#ScrollViewport { background: transparent; }")
    inner.setObjectName("Transparent")
    area.setWidget(inner)
    if horizontal:
        area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    else:
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    return area
