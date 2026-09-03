"""The left navigation dock: an icon, a name, and a category heading.

Requested 2026-09-03, after four rounds of fixes to the old one had not
settled it: "start from scratch and write a simple dock no effects just icon
and text in categories. Scrap the sub categories, only effect should be hover
is blue and tooltip at bottom of screen".

WHAT WAS REMOVED, AND WHY EACH ONE WAS THE BUG. The dock this replaces was
1,116 lines across ``Sidebar`` and ``_DockRow`` in ``spacr.qt.app``, and
nearly all of it was machinery that existed to defeat itself:

* a translucent slab painted in ``paintEvent`` — the "black box" of four
  separate commits, which turned out to be the dock painting itself rather
  than any stylesheet;
* a per-row icon-size model (``resting_icon_px``, ``_place_icon``,
  ``_set_icon_px``, ``_forget_icon_sizes``, ``_rest_every_icon``) that grew
  and shrank icons under the pointer, which is what made hovering relayout
  the column and blink;
* the name painted only while hovered, so a resting dock was a column of
  unlabelled glyphs;
* a second, indented level of folded modules with its own expand state
  (``_fold_children``, ``_open_hosts``), which is the "sub categories".

None of that is here. A row is a button with an icon and its name, always
both. The only thing the pointer changes is the colour.

WHERE THE EXPLANATION WENT. Not into a popup tooltip — those are explicitly
unwanted — but into the strip along the bottom of the window, which already
exists as :mod:`spacr.qt.widgets.module_hint_bar` and already holds the last
hovered module for thirty seconds with its API and tutorial links. This dock
only says which module is under the pointer, via :attr:`Dock.module_hovered`;
the bar decides how to explain it.

CATEGORIES STILL COLLAPSE. The headings are clickable, because the module
list is longer than a short screen and a dock taller than its window is the
failure the sections were introduced to fix.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..theme import active_palette
from .eliding import ElidingPushButton

#: A dock row as the registry hands it over: key, name, description, section.
Row = Tuple[str, str, str, str]


class DockRow(ElidingPushButton):
    """One module: its icon, then its name, both always drawn.

    The row paints nothing of its own — the colour comes from the
    stylesheet :class:`Dock` installs, so there is one place that decides
    what hover looks like and no ``paintEvent`` to disagree with it.
    """

    hovered = Signal(str, bool)          #: key, and whether the pointer entered

    def __init__(self, key: str, name: str, desc: str = "", parent=None):
        super().__init__(name, parent)
        self.key = key
        self.desc = desc
        self.setObjectName("DockRow")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Fixed)
        # NO POPUP TOOLTIP. The bottom strip is the explanation surface; a
        # popup here would be a second one, in a place the pointer covers.
        self.setToolTip("")
        self._hovered = False

    def is_hovered(self) -> bool:
        """Whether the pointer is currently on this row."""
        return self._hovered

    def enterEvent(self, event):                # noqa: N802 - Qt naming
        self._hovered = True
        self.hovered.emit(self.key, True)
        super().enterEvent(event)

    def leaveEvent(self, event):                # noqa: N802 - Qt naming
        self._hovered = False
        self.hovered.emit(self.key, False)
        super().leaveEvent(event)


class SectionHeader(QLabel):
    """A category heading. A label rather than a button, because it is
    already styled as a heading and a button would have to be un-styled
    back into one; the click arrives through :meth:`mousePressEvent`."""

    clicked = Signal(str)

    def __init__(self, section: str, parent=None):
        super().__init__(section, parent)
        self.section = section
        self.setObjectName("DockSection")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):           # noqa: N802 - Qt naming
        self.clicked.emit(self.section)
        super().mousePressEvent(event)


class Dock(QWidget):
    """The navigation column: categories, each holding icon+name rows.

    :param rows: the modules to draw, in order, as ``(key, name, desc,
        section)``. Grouping IS ordering: a new heading starts whenever the
        section changes, so a row out of place draws its heading twice.
    :param icon_for: optional ``key -> QIcon | None`` used for the row icons.
        Injected rather than imported so this module does not depend on
        :mod:`spacr.qt.app`, which is what defines the registry.
    """

    nav_selected = Signal(str)           #: a row was clicked
    module_hovered = Signal(str)         #: a row is under the pointer

    def __init__(self, rows: Iterable[Row],
                 icon_for: Optional[Callable[[str], object]] = None,
                 parent=None):
        super().__init__(parent)
        self.setObjectName("Dock")
        self._icon_for = icon_for
        self._rows: List[DockRow] = []
        self._headers: Dict[str, SectionHeader] = {}
        self._section_rows: Dict[str, List[DockRow]] = {}
        self._open: set = set()

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)

        current = None
        for key, name, desc, section in rows:
            if section != current:
                header = SectionHeader(section)
                header.clicked.connect(self.toggle_section)
                column.addWidget(header)
                self._headers[section] = header
                self._section_rows.setdefault(section, [])
                self._open.add(section)
                current = section
            row = DockRow(key, name, desc)
            row.clicked.connect(lambda _checked=False, k=key:
                                self.nav_selected.emit(k))
            row.hovered.connect(self._on_row_hovered)
            column.addWidget(row)
            self._rows.append(row)
            self._section_rows.setdefault(section, []).append(row)
        column.addStretch(1)

        self.refresh_icons()
        self.apply_theme()

    # -- what the pointer does -------------------------------------------
    def _on_row_hovered(self, key: str, entered: bool) -> None:
        """Name the hovered module so the bottom strip can explain it.

        Only entering is reported. A leave that cleared the bar would empty
        it the moment the pointer set off toward the links it holds, which
        is the whole reason that bar keeps its last module.
        """
        if entered:
            self.module_hovered.emit(key)

    def hovered_row(self) -> Optional[DockRow]:
        """The row under the pointer, or ``None``."""
        for row in self._rows:
            if row.is_hovered():
                return row
        return None

    # -- categories -------------------------------------------------------
    def sections(self) -> List[str]:
        """Every category heading, in the order they are drawn."""
        return list(self._headers)

    def rows(self) -> List[DockRow]:
        """Every module row, in the order they are drawn."""
        return list(self._rows)

    def section_is_open(self, section: str) -> bool:
        """Whether ``section``'s rows are currently shown."""
        return section in self._open

    def toggle_section(self, section: str) -> bool:
        """Open a closed category or close an open one. Returns the new state."""
        if section in self._open:
            self._open.discard(section)
        else:
            self._open.add(section)
        self.refresh_visibility()
        return section in self._open

    def refresh_visibility(self) -> None:
        """Show each row if, and only if, its category is open."""
        for section, rows in self._section_rows.items():
            visible = section in self._open
            for row in rows:
                row.setVisible(visible)

    # -- appearance -------------------------------------------------------
    def refresh_icons(self) -> None:
        """Re-ask the provider for every row's icon.

        Every icon is set once, at one size, and never changed again — the
        old dock's growing/shrinking icons are what made hover relayout the
        column.
        """
        if self._icon_for is None:
            return
        for row in self._rows:
            icon = self._icon_for(row.key)
            if icon is not None:
                row.setIcon(icon)

    def apply_theme(self) -> None:
        """Install the one rule that decides what hover looks like."""
        accent = active_palette()["accent"]
        self.setStyleSheet(
            "QPushButton#DockRow {"
            "  background: transparent; border: none; text-align: left;"
            "  padding: 6px 10px;"
            "}"
            f"QPushButton#DockRow:hover {{ color: {accent}; }}"
            "QLabel#DockSection {"
            "  padding: 10px 10px 4px 10px; font-weight: 600;"
            "}"
        )

    def row_height(self) -> int:
        """The height of a row, or 0 if the dock is empty."""
        return self._rows[0].sizeHint().height() if self._rows else 0
