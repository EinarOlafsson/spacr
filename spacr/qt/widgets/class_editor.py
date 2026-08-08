"""The Classes editor: pick a column, name what is in it.

The setting is a dict of ``name -> {column, value}``, and this is how it gets
filled in: choose a column, and every distinct value in it becomes a row the
user gives a name to. That is the whole gesture — "you set the column then the
keys of this dict get populated and the user fills in their names."

Two things it does that the old settings could not.

**More than one column.** Each row remembers which column its value came from,
so classes can be defined across several annotation columns at once. Adding a
second column appends its values rather than replacing the first's.

**The random complement.** One row can be "everything not claimed, chosen at
random", which is what the retired ``write_random_annotation_column`` used to
arrange. It is a KIND OF ROW here rather than a button pressed beforehand,
because it is a way of defining a class and belongs where the other class
definitions are.

Under the metadata basis the offered columns become the plate's own
coordinates — plate, row, column, field, well — which is why
``location_column``, ``positive_control`` and ``negative_control`` are no
longer needed: "positive control is column 3" is exactly a row in this table.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from ...classify_classes import (
    METADATA_COLUMNS, ClassDefinitionError, ClassRule, candidate_columns,
    values_in,
)
from ..theme import SPACING, register_widget_qss

LOG = logging.getLogger("spacr.qt.class_editor")

QSS_NAME = "ClassEditor"


def _class_editor_qss(palette, opacity=None) -> str:
    return f"""
    QTreeWidget#ClassTable {{
        background: transparent;
        color: {palette['fg']};
        border: 1px solid {palette['border']};
        border-radius: 4px;
    }}
    QTreeWidget#ClassTable::item {{
        color: {palette['fg']};
        padding: 2px 4px;
    }}
    QLabel#ClassEditorHint {{
        color: {palette['fg_muted']};
        background: transparent;
    }}
    QWidget#ClassEditor {{
        background: transparent;
    }}
    """


register_widget_qss(QSS_NAME, _class_editor_qss, replace=True)


class ClassEditorWidget(QWidget):
    """Edits the ``classes`` setting.

    :param value: the current setting -- a dict, or the old list of names.
    :param frame: the table whose columns and values are offered. Without one
        the widget still edits an existing dict; it simply cannot populate new
        rows, and says so rather than showing an empty column picker as though
        the table had no columns.
    """

    value_changed = Signal(object)

    def __init__(self, value: Any = None, parent=None, *,
                 frame: Optional[pd.DataFrame] = None,
                 basis: str = "annotation"):
        super().__init__(parent)
        self.setObjectName("ClassEditor")
        self._frame = frame
        self._basis = basis
        self._rules: List[ClassRule] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACING["xs"])

        picker = QHBoxLayout()
        picker.setContentsMargins(0, 0, 0, 0)
        picker.addWidget(QLabel("Column", self))
        self.column = QComboBox(self)
        self.column.setToolTip(
            "Choosing a column fills the table below with its values, one row "
            "per class. Choosing a SECOND column adds its values alongside — "
            "classes can be defined across more than one column.")
        picker.addWidget(self.column, 1)
        self._add = QPushButton("Add values", self)
        self._add.clicked.connect(self.populate_from_column)
        picker.addWidget(self._add)
        outer.addLayout(picker)

        self.table = QTreeWidget(self)
        self.table.setObjectName("ClassTable")
        self.table.setColumnCount(3)
        self.table.setHeaderLabels(["Class name", "Value", "From column"])
        header = self.table.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.itemChanged.connect(self._on_item_changed)
        outer.addWidget(self.table, 1)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self._remove = QPushButton("Remove", self)
        self._remove.clicked.connect(self.remove_selected)
        row.addWidget(self._remove)
        self._complement = QPushButton("Add random rest", self)
        self._complement.setToolTip(
            "One class made of the objects no other class claimed, chosen at "
            "random and sized to match the largest class. This is what to use "
            "when only one class is annotated — a comparison group ten times "
            "larger teaches the model the prior, not the difference.")
        self._complement.clicked.connect(self.add_random_complement)
        row.addWidget(self._complement)
        row.addStretch(1)
        outer.addLayout(row)

        self._hint = QLabel("", self)
        self._hint.setObjectName("ClassEditorHint")
        self._hint.setWordWrap(True)
        outer.addWidget(self._hint)

        self.set_frame(frame)
        self.set_value(value)

    # -- what is on offer --------------------------------------------------
    def set_frame(self, frame: Optional[pd.DataFrame]) -> None:
        """Offer this table's columns."""
        self._frame = frame
        columns = candidate_columns(
            {"dataset_mode": self._basis},
            available=list(frame.columns) if frame is not None else ())
        current = self.column.currentText()
        self.column.blockSignals(True)
        self.column.clear()
        self.column.addItems([str(c) for c in columns])
        if current in columns:
            self.column.setCurrentText(current)
        self.column.blockSignals(False)
        self._add.setEnabled(bool(columns) and frame is not None)
        if frame is None:
            self._say("Load a table to fill classes in from a column.")

    def set_basis(self, basis: str) -> None:
        """Metadata or annotation: it decides which columns are offered.

        Under metadata these become plate / row / column / field / well, which
        is what replaces location_column plus the two control settings.
        """
        self._basis = basis
        self.set_frame(self._frame)

    # -- the value ---------------------------------------------------------
    def set_value(self, value: Any) -> None:
        """Show ``value``, whether it is the dict or the old list of names."""
        self._rules = []
        if isinstance(value, Mapping):
            for name, spec in value.items():
                if not isinstance(spec, Mapping):
                    continue
                try:
                    self._rules.append(ClassRule(
                        name=str(name),
                        column=str(spec.get("column", "") or ""),
                        value=spec.get("value"),
                        random_complement=bool(
                            spec.get("random_complement", False))))
                except ClassDefinitionError:
                    LOG.debug("skipping malformed class %r", name)
        elif isinstance(value, (list, tuple)):
            # The old shape: names with nothing saying what they select. Shown
            # as named rows with no value, so the user can see what has to be
            # filled in rather than finding the table empty.
            for name in value:
                self._rules.append(ClassRule(name=str(name), column="?",
                                             value=None))
        self._rebuild()

    def value(self) -> Dict[str, Dict[str, Any]]:
        return {r.name: r.to_dict() for r in self._rules}

    #: The settings panel reads every custom widget through this name.
    def get_value(self) -> Dict[str, Dict[str, Any]]:
        return self.value()

    def rules(self) -> List[ClassRule]:
        return list(self._rules)

    # -- editing -----------------------------------------------------------
    def populate_from_column(self) -> None:
        """Fill the table from the chosen column's distinct values.

        Values already present are left alone, so adding a second column adds
        to the table rather than replacing what is in it -- and re-adding the
        same column does not duplicate or reset the names already typed.
        """
        column = self.column.currentText()
        if not column or self._frame is None:
            return
        try:
            values = values_in(self._frame, column)
        except ClassDefinitionError as exc:
            self._say(str(exc))
            return

        known = {(r.column, _key(r.value)) for r in self._rules}
        added = 0
        for value in values:
            if (column, _key(value)) in known:
                continue
            # `_key` for the label too, so a float 1.0 reads as "1" -- the
            # name is what the user sees in every report afterwards.
            self._rules.append(ClassRule(name=f"{column}={_key(value)}",
                                         column=column, value=value))
            added += 1
        self._rebuild()
        self._say(f"added {added} value(s) from {column}"
                  if added else f"{column} adds nothing new")

    def add_random_complement(self) -> None:
        if any(r.random_complement for r in self._rules):
            self._say("there is already a random-rest class; two classes both "
                      "meaning 'everything else' have no boundary between them")
            return
        self._rules.append(ClassRule(name="rest", random_complement=True))
        self._rebuild()

    def remove_selected(self) -> None:
        item = self.table.currentItem()
        if item is None:
            return
        index = self.table.indexOfTopLevelItem(item)
        if 0 <= index < len(self._rules):
            del self._rules[index]
            self._rebuild()

    # -- plumbing ----------------------------------------------------------
    def _rebuild(self) -> None:
        self.table.blockSignals(True)
        self.table.clear()
        for rule in self._rules:
            if rule.random_complement:
                labels = [rule.name, "the rest, at random", ""]
            else:
                labels = [rule.name, "" if rule.value is None else str(rule.value),
                          rule.column]
            item = QTreeWidgetItem(labels)
            # Only the NAME is editable. The value and its column are facts
            # about the table, and letting them be typed over would produce a
            # class that selects nothing with no sign of why.
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.table.addTopLevelItem(item)
        self.table.blockSignals(False)
        self._emit()

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0:
            return
        index = self.table.indexOfTopLevelItem(item)
        if not (0 <= index < len(self._rules)):
            return
        name = item.text(0).strip()
        if not name:
            # A class with no name cannot be trained on or reported, so the
            # old one is put back rather than accepted and failing later.
            self.table.blockSignals(True)
            item.setText(0, self._rules[index].name)
            self.table.blockSignals(False)
            return
        rule = self._rules[index]
        self._rules[index] = ClassRule(
            name=name, column=rule.column, value=rule.value,
            random_complement=rule.random_complement)
        self._emit()

    def _emit(self) -> None:
        self.value_changed.emit(self.value())

    def _say(self, message: str) -> None:
        self._hint.setText(message)


def _key(value: Any) -> str:
    """A value's identity for de-duplication, insensitive to 1 vs 1.0."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
