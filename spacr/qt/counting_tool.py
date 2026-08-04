"""``B13`` — counting by hand on the layer canvas.

:mod:`spacr.counting` is the session: the classes, the markers, the undo stack
and the export, all in plain numpy so the tally can be tested without a
display. This is the mouse and the panel — a click that lands on a *world*
coordinate, a live tally beside the image, and one button that writes the
clicks out rather than the total.

Why a click removes as well as adds
-----------------------------------

A counting session is thousands of clicks and a proportion of them are wrong.
The two ways of correcting a mistake are different actions: **undo** takes back
the last thing you did, and **clicking a marker again** takes back a specific
thing you did some time ago. Both are here, and clicking a marker removes it
whatever class it was scored as — a counter who has to re-select the class
before they can take a marker back will leave the wrong marker there.

The tally is derived, never typed
---------------------------------

Everything the panel shows is read from the marker layers on every model
event. There is no counter variable to drift out of step with the picture, so
a marker removed through the layer list rather than through this panel changes
the number too.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem,
                               QVBoxLayout, QWidget)

from ..counting import CountingSession
from ..layers import FieldKey, LayerError, LayerEvent
from .layer_viewer import CanvasTool, LayerCanvas
from .theme import register_widget_qss
from .widgets.preview_controls import FlatButton

LOG = logging.getLogger(__name__)

__all__ = [
    "CountingTool",
    "CountingPanel",
]


def _counting_qss(palette: Dict[str, Any], opacity) -> str:
    """This panel's QSS block, appended to every generated stylesheet."""
    return f"""
QWidget#CountingPanel {{
    background: transparent;
}}
QListWidget#CountingClasses {{
    background: {palette["surface_alt"]};
    border: 1px solid {palette["border_soft"]};
    border-radius: 10px;
    padding: 4px;
    color: {palette["fg"]};
}}
QListWidget#CountingClasses::item {{
    padding: 5px 6px;
    border-radius: 6px;
}}
QListWidget#CountingClasses::item:selected {{
    background: {palette["accent_soft"]};
    color: {palette["fg"]};
}}
QLabel#CountingTotal {{
    color: {palette["fg_muted"]};
}}
"""


register_widget_qss("CountingPanel", _counting_qss, replace=True)


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------

class CountingTool(CanvasTool):
    """Turns clicks on a :class:`~spacr.qt.layer_viewer.LayerCanvas` into counts.

    Left click adds a marker of the active class, or takes away the marker
    already under the cursor. Right click only ever removes. ``1``–``9`` choose
    the class and Backspace undoes, so a whole session is one hand on the mouse
    and one on the number row.

    :param session: the :class:`spacr.counting.CountingSession` to count into.
    """

    cursor = Qt.CrossCursor

    def __init__(self, session: CountingSession):
        if not isinstance(session, CountingSession):
            raise LayerError(
                f"a counting tool counts into a CountingSession, got "
                f"{session!r}")
        self.session = session

    def press(self, view: LayerCanvas, world: Dict[str, float],
              event: Any) -> bool:
        """Add or remove one marker."""
        button = event.button() if hasattr(event, "button") else Qt.LeftButton
        if button == Qt.RightButton:
            self.session.remove_at(world)
            return True
        if button != Qt.LeftButton:
            return False
        self.session.toggle(world)
        return True

    def key(self, view: LayerCanvas, event: Any) -> bool:
        """``1``–``9`` select a class; Backspace undoes the last click."""
        if event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
            self.session.undo()
            return True
        text = event.text()
        if text and text.isdigit():
            name = self.session.class_for_shortcut(text)
            if name is not None:
                self.session.active = name
                return True
        return False


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

class CountingPanel(QWidget):
    """The tally beside the image, and the button that writes it out.

    :param canvas: the :class:`~spacr.qt.layer_viewer.LayerCanvas` to count on.
    :param classes: what is being counted; see
        :class:`spacr.counting.CountingSession`.
    :param field: the :class:`~spacr.layers.FieldKey` being counted, carried
        into the export so a count can join the measurement tables.
    """

    #: The tally changed. Carries ``{class: count}``.
    counts_changed = Signal(object)
    #: A count was written. Carries the path.
    exported = Signal(str)

    def __init__(self, canvas: LayerCanvas, parent=None, *,
                 classes: Optional[List[Any]] = None,
                 field: Optional[FieldKey] = None,
                 session: Optional[CountingSession] = None):
        super().__init__(parent)
        self.setObjectName("CountingPanel")
        self._canvas = canvas
        self._session = session or CountingSession(
            canvas.stack, classes=classes, field=field)
        self._tool: Optional[CountingTool] = None
        self._syncing = False
        self._build()
        self._canvas.stack.subscribe(self._on_layers_changed)
        self.refresh()

    # -- construction -------------------------------------------------------
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self.count_button = FlatButton(
            "Count", self,
            tooltip="Click to add a marker, click a marker to take it away, "
                    "1–9 choose the class, Backspace undoes")
        self.count_button.setCheckable(True)
        self.count_button.toggled.connect(self._on_count_toggled)
        outer.addWidget(self.count_button)

        self.class_list = QListWidget(self)
        self.class_list.setObjectName("CountingClasses")
        self.class_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.class_list.currentRowChanged.connect(self._on_class_selected)
        outer.addWidget(self.class_list, 1)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.add_class_button = FlatButton("＋ Class", self,
                                           tooltip="Count something else too")
        self.add_class_button.clicked.connect(self.add_class)
        row.addWidget(self.add_class_button)
        self.undo_button = FlatButton("Undo", self,
                                      tooltip="Take back the last click")
        self.undo_button.clicked.connect(self.undo)
        row.addWidget(self.undo_button)
        self.clear_button = FlatButton("Clear", self,
                                       tooltip="Remove every marker")
        self.clear_button.clicked.connect(self.clear)
        row.addWidget(self.clear_button)
        outer.addLayout(row)

        self.total = QLabel("", self)
        self.total.setObjectName("CountingTotal")
        self.total.setWordWrap(True)
        outer.addWidget(self.total)

        exports = QHBoxLayout()
        exports.setSpacing(4)
        self.export_button = FlatButton(
            "Export clicks…", self,
            tooltip="One row per marker: class, world coordinates and units")
        self.export_button.clicked.connect(self.export_points)
        exports.addWidget(self.export_button, 1)
        self.summary_button = FlatButton(
            "Export tally…", self, tooltip="One row per class: count and share")
        self.summary_button.clicked.connect(self.export_summary)
        exports.addWidget(self.summary_button, 1)
        outer.addLayout(exports)

    # -- model --------------------------------------------------------------
    @property
    def session(self) -> CountingSession:
        """The counting session this panel drives."""
        return self._session

    @property
    def tool(self) -> Optional[CountingTool]:
        """The tool while counting is switched on, else ``None``."""
        return self._tool

    def start_counting(self) -> CountingTool:
        """Attach the counting tool to the canvas and return it."""
        self._tool = CountingTool(self._session)
        self._canvas.set_tool(self._tool)
        return self._tool

    def stop_counting(self) -> None:
        """Give the canvas its mouse back."""
        if self._canvas.tool is self._tool and self._tool is not None:
            self._canvas.set_tool(None)
        self._tool = None

    def _on_count_toggled(self, checked: bool) -> None:
        if checked:
            self.start_counting()
        else:
            self.stop_counting()

    def _on_layers_changed(self, event: LayerEvent) -> None:
        # Derived, not stored: a marker removed through the layer list changes
        # the number here too.
        if event.kind in ("data", "inserted", "removed"):
            self.refresh()

    # -- actions ------------------------------------------------------------
    def add_class(self, name: Optional[str] = None) -> str:
        """Count one more thing; returns the class name."""
        if not isinstance(name, str) or not name.strip():
            name = f"class {len(self._session.classes) + 1}"
        entry = self._session.add_class(name)
        self.refresh()
        return entry.name

    def undo(self) -> bool:
        """Take back the last click. ``True`` if there was one."""
        undone = self._session.undo()
        self.refresh()
        return undone is not None

    def clear(self) -> int:
        """Remove every marker of every class; returns how many went."""
        removed = self._session.clear()
        self.refresh()
        return removed

    def export_points(self) -> Optional[str]:
        """Write one row per marker, asking where."""
        return self._export(summary=False)

    def export_summary(self) -> Optional[str]:
        """Write one row per class, asking where."""
        return self._export(summary=True)

    def _export(self, *, summary: bool) -> Optional[str]:
        suggested = os.path.join(
            os.getcwd(), "counts_summary.csv" if summary else "counts.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the count", suggested, "CSV (*.csv)")
        if not path:
            return None
        return self.write(path, summary=summary)

    def write(self, path: str, *, summary: bool = False) -> Optional[str]:
        """Write the count to ``path`` without asking; returns the path.

        The seam the dialog goes through, so a screen (or a test) can save a
        count without a modal.
        """
        try:
            target = self._session.to_csv(path, summary=summary)
        except (OSError, LayerError) as exc:
            LOG.info("could not write the count", exc_info=True)
            self.total.setText(f"Could not write the count: {exc}")
            return None
        self.exported.emit(target)
        return target

    def _on_class_selected(self, row: int) -> None:
        if self._syncing or row < 0:
            return
        names = self._session.class_names
        if row < len(names):
            self._session.active = names[row]

    # -- the tally ----------------------------------------------------------
    def refresh(self) -> None:
        """Redraw the tally from the marker layers."""
        counts = self._session.counts()
        self._syncing = True
        try:
            self.class_list.clear()
            for entry in self._session.classes:
                n = counts[entry.name]
                share = self._session.fraction(entry.name)
                shortcut = f" [{entry.shortcut}]" if entry.shortcut else ""
                item = QListWidgetItem(
                    f"{entry.name}{shortcut}   {n}  ({share:.0%})")
                item.setData(Qt.UserRole, entry.name)
                item.setToolTip(
                    f"{n} marker(s) of {entry.name}"
                    + (f" — press {entry.shortcut} to count these"
                       if entry.shortcut else ""))
                self.class_list.addItem(item)
            names = self._session.class_names
            if self._session.active in names:
                self.class_list.setCurrentRow(names.index(self._session.active))
        finally:
            self._syncing = False
        self.total.setText(self._session.describe())
        self.counts_changed.emit(counts)

    def closeEvent(self, event) -> None:
        """Stop listening and give the canvas its mouse back."""
        self.stop_counting()
        self._canvas.stack.unsubscribe(self._on_layers_changed)
        super().closeEvent(event)
