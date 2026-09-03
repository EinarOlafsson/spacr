"""The repeated per-object settings, drawn as one row per question.

78 of Mask's 201 settings are the SAME twenty-odd questions asked once per
object type -- ``cell_diameter``, ``nucleus_diameter``,
``pathogen_diameter``, ``organelle_diameter`` -- and a form that lists them
flat asks 203 questions before anything is segmented. A table was chosen over
tabs and over leaving the names flat.

:mod:`spacr.object_settings_table` is the model and draws nothing; this is
the view over it. The split matters more than it looks: the stored keys never
change, so no settings file, notebook, tutorial or ``spacr-run`` invocation
migrates. What was wrong was the presentation, so only the presentation
changes.

WHY THIS SHAPE IS WHAT LETS AN ARBITRARY ORGANELLE COUNT LAND. The number of
organelles a run may declare is not fixed. In a flat vocabulary each new organelle is twenty new
settings that every tooltip table and translation catalog has to learn; here
it is one COLUMN, and the number of questions does not move. :meth:`
ObjectSettingsGrid.add_object` is that operation, and it starts a new
organelle from the first one's answers rather than from a global default
nobody chose.

TWO THINGS THIS VIEW IS CAREFUL ABOUT, both of which would corrupt a settings
file rather than merely look wrong:

* **A value keeps its type.** ``cell_diameter`` is an int, ``organelle_
  cellprob_threshold`` is a float, and a cell edited in a table arrives as a
  string. Writing ``"12"`` where ``12`` was is a settings file that has
  quietly changed meaning, and the pipeline reading it either coerces
  silently or fails a long way from here.
* **A question an object does not ask stays absent.** ``cytoplasm`` has no
  channel, no diameter and no detection method -- it is DERIVED, cell minus
  the rest, not found in a channel. Those cells are blank and not editable,
  because writing a value there invents a key nothing reads.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ...object_roles import setting_label
from ...object_settings_table import (OBJECT_ORDER, column_label, from_table,
                                      to_table, widen)
from ...organelle_types import MAX_ORGANELLES, organelle_role
from ..theme import SPACING
from .sortable_table import install_sorting

__all__ = ["AUTO_TEXT", "ObjectSettingsGrid", "ObjectSettingsModel"]

#: What an unset value reads as. ``None`` means "work it out" for most of
#: these -- a diameter of None is Cellpose estimating it -- and an empty cell
#: would read as "nobody has filled this in yet", which is a different claim.
AUTO_TEXT = "auto"


def _coerce(text: str, like: Any) -> Any:
    """Read ``text`` back as the type ``like`` already had.

    THE WHOLE REASON THIS FUNCTION EXISTS is that a table hands back strings.
    ``cell_diameter`` is an int and ``organelle_cellprob_threshold`` is a
    float; storing either as ``"12"`` is a settings file that has changed
    meaning without anyone saying so.

    :param text: what the user typed.
    :param like: the value being replaced, whose type is the target. When it
        is ``None`` there is no type to copy -- the caller passes a sibling
        answer from the same row instead, because the same question about a
        different object is the best evidence available about what this one
        is.
    """
    raw = str(text).strip()
    if raw == "" or raw.lower() == AUTO_TEXT:
        return None
    if isinstance(like, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    for kind in ((int, float) if isinstance(like, (int, float))
                 and not isinstance(like, bool) else ()):
        try:
            return kind(raw)
        except (TypeError, ValueError):
            continue
    if like is None or isinstance(like, str):
        # NO TYPE TO COPY. Read a number as a number so a diameter typed
        # into an empty cell is not stored as text, and leave anything else
        # as the string it is -- a model name is a string and always was.
        for kind in (int, float):
            try:
                return kind(raw)
            except (TypeError, ValueError):
                continue
        if raw.lower() in ("true", "false"):
            return raw.lower() == "true"
    return raw


class ObjectSettingsModel(QAbstractTableModel):
    """One row per question, one column per object type.

    A model rather than a widget full of cells because the table is 55 rows
    by as many objects as the run has, and every one of those cells would
    otherwise be a widget the form has to build, lay out and translate.
    """

    #: Emitted when a cell's value actually changed.
    edited = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._table: Dict[str, Dict[str, Any]] = {}
        self._questions: Tuple[str, ...] = ()
        self._objects: Tuple[str, ...] = ()

    # -- content -----------------------------------------------------------

    def set_table(self, table: Mapping[str, Mapping[str, Any]]) -> None:
        """Show ``table``, as :func:`spacr.object_settings_table.to_table`
        returns it."""
        self.beginResetModel()
        self._table = {q: dict(row) for q, row in (table or {}).items()}
        self._questions = tuple(self._table)
        order = {name: index for index, name in enumerate(OBJECT_ORDER)}
        present = {obj for row in self._table.values() for obj in row}
        self._objects = tuple(sorted(
            present, key=lambda o: order.get(o, len(order))))
        self.endResetModel()

    def table(self) -> Dict[str, Dict[str, Any]]:
        """The table as it now stands, including every edit."""
        return {q: dict(row) for q, row in self._table.items()}

    def objects(self) -> Tuple[str, ...]:
        """The object columns, in the order they are drawn."""
        return self._objects

    def question_at(self, row: int) -> str:
        """The settings question one row asks, or ``''``."""
        return self._questions[row] if 0 <= row < len(self._questions) else ""

    def value_at(self, question: str, obj: str) -> Any:
        """One cell's stored value. ``KeyError``-free: absent is ``None``."""
        return self._table.get(question, {}).get(obj)

    def asks(self, question: str, obj: str) -> bool:
        """Whether ``obj`` asks ``question`` at all.

        Absence is a fact about the object, not a value it has yet to be
        given: cytoplasm is derived and has no channel to be found in.
        """
        return obj in self._table.get(question, {})

    # -- QAbstractTableModel ----------------------------------------------

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._questions)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._objects)

    def flags(self, index):
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if not index.isValid():
            return base
        question = self.question_at(index.row())
        obj = self._objects[index.column()]
        if not self.asks(question, obj):
            # NOT EDITABLE AND NOT ENABLED: a cell that can be typed into
            # invents a settings key nothing reads.
            return Qt.ItemIsSelectable
        return base | Qt.ItemIsEditable

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        question = self.question_at(index.row())
        obj = self._objects[index.column()]
        if not self.asks(question, obj):
            if role == Qt.ToolTipRole:
                return (f"{column_label(obj)} does not ask this. It is not a "
                        f"value waiting to be filled in.")
            return None
        value = self.value_at(question, obj)
        if role == Qt.DisplayRole:
            return AUTO_TEXT if value is None else str(value)
        if role == Qt.EditRole:
            return "" if value is None else str(value)
        if role == Qt.ToolTipRole:
            return (f"{obj}_{question}  =  "
                    f"{AUTO_TEXT if value is None else value!r}")
        return None

    def setData(self, index, value, role=Qt.EditRole) -> bool:
        if not index.isValid() or role != Qt.EditRole:
            return False
        question = self.question_at(index.row())
        obj = self._objects[index.column()]
        if not self.asks(question, obj):
            return False
        row = self._table[question]
        current = row.get(obj)
        like = current
        if like is None:
            # THE SAME QUESTION ABOUT ANOTHER OBJECT is the best evidence
            # available about what this one is: a diameter is a diameter
            # whether it is a cell's or a nucleus's.
            like = next((v for o, v in row.items()
                         if o != obj and v is not None), None)
        new = _coerce(value, like)
        if new == current and type(new) is type(current):
            return False
        row[obj] = new
        self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
        self.edited.emit()
        return True

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return column_label(self._objects[section])
            return setting_label(self._questions[section])
        if role == Qt.ToolTipRole and orientation == Qt.Vertical:
            # THE STORED KEY, because the whole table is one settings file
            # rearranged and a user has to be able to find the key again.
            return f"<object>_{self._questions[section]}"
        return None


class ObjectSettingsGrid(QWidget):
    """The per-object settings table, and the button that widens it.

    :param parent: parent widget.
    """

    #: Emitted when any cell changed, or a column was added.
    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._base: Dict[str, Any] = {}
        self._model = ObjectSettingsModel(self)
        self._model.edited.connect(self.settings_changed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACING["sm"])

        self._table = QTableView(self)
        self._table.setModel(self._model)
        # AFTER setModel, as the contract requires: a QTableView is wrapped in
        # a proxy, so the selection model has to be taken afterwards. The
        # stored answers are unaffected -- `table()` reads the model, not the
        # view, so sorting the questions on screen reorders nothing on disk.
        install_sorting(self._table)
        self._table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self._table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        outer.addWidget(self._table, 1)

        row = QHBoxLayout()
        row.setSpacing(SPACING["sm"])
        self._status = QLabel("", self)
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        self._add = QPushButton("Add an organelle", self)
        self._add.setToolTip(
            "One more organelle is one more COLUMN. In the flat settings "
            "vocabulary it was twenty new settings, which is why the count "
            "could not be arbitrary before this table existed.")
        self._add.clicked.connect(self.add_organelle)
        row.addWidget(self._status, 1)
        row.addWidget(self._add)
        outer.addLayout(row)

    # -- content -----------------------------------------------------------

    def set_settings(self, settings: Mapping[str, Any]) -> None:
        """Show the per-object half of a flat settings dict.

        The rest is KEPT, not dropped: :meth:`settings` returns it unchanged
        beside the table's own keys, so this widget can edit a corner of a
        settings file without holding the whole of it hostage.
        """
        self._base = dict(settings or {})
        self._model.set_table(to_table(self._base))
        self._announce()

    def settings(self) -> Dict[str, Any]:
        """The whole settings dict, with the table's answers written back."""
        return from_table(self._model.table(), self._base)

    def table(self) -> Dict[str, Dict[str, Any]]:
        """The table itself, for a caller that wants the shape."""
        return self._model.table()

    def objects(self) -> Tuple[str, ...]:
        """Which object columns are on screen."""
        return self._model.objects()

    def questions(self) -> Tuple[str, ...]:
        """Which questions are on screen, in order."""
        return tuple(self._model.table())

    # -- widening ----------------------------------------------------------

    def next_organelle(self) -> str:
        """The role the next organelle column would take, or ``''``.

        Empty when the alphabet has run out: slot names are lettered because
        an object type is embedded in underscore-separated object keys, so
        twenty-six is where the naming scheme ends rather than where somebody
        chose to stop.
        """
        used = {obj for obj in self.objects() if obj.startswith("organelle")}
        for number in range(1, MAX_ORGANELLES + 1):
            role = organelle_role(number)
            if role not in used:
                return role
        return ""

    def add_organelle(self) -> bool:
        """Add the next organelle column, seeded from the first one.

        :returns: False when there is no slot left, with the reason on screen
            rather than as an exception into a GUI slot.
        """
        role = self.next_organelle()
        if not role:
            self._status.setText(
                f"{MAX_ORGANELLES} organelles is the ceiling: the slots are "
                f"lettered, so the alphabet is what runs out.")
            return False
        self._model.set_table(widen(self._model.table(), role))
        self._announce()
        self.settings_changed.emit()
        return True

    def _announce(self) -> None:
        """Say what the table is holding, in the numbers 364 is about."""
        questions = len(self.questions())
        objects = len(self.objects())
        self._status.setText(
            f"{questions} question(s) x {objects} object(s) = "
            f"{sum(len(row) for row in self._model.table().values())} "
            f"settings, asked once each.")

    def status_text(self) -> str:
        """What the line under the table says."""
        return self._status.text()

    def set_value(self, question: str, obj: str, text: str) -> bool:
        """Type into one cell the way the editor would.

        The screen's own edit path, exposed so a test drives the same code an
        item delegate does rather than reaching into the model.
        """
        objects = self.objects()
        rows = self.questions()
        if question not in rows or obj not in objects:
            return False
        index = self._model.index(rows.index(question), objects.index(obj))
        return self._model.setData(index, text, Qt.EditRole)
