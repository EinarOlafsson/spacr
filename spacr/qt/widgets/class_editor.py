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

import ast
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QPushButton, QScrollArea, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
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


class ClassChip(QWidget):
    """One class, as the two bubbles the maintainer asked for.

    "class then value class generating a teal bubble and the value generating
    a green bubble ... i just thought it was a good idea to consolidate the
    information into one object."

    So the two halves are one object on screen: a TEAL pill carrying the class
    name and a GREEN pill carrying the value it selects, side by side with a
    single remove button for the pair. A class that selects nothing -- the
    random complement -- shows only the teal half, because there is no value
    to put in the green one.

    The colours are palette ROLES (`chip_class`, `chip_value`), not literals.
    A hard-coded teal survives exactly until someone switches to the light
    theme, where it fails contrast against a white surface.
    """

    removed = Signal(int)

    def __init__(self, index: int, rule: "ClassRule", palette, parent=None):
        super().__init__(parent)
        self.setObjectName("ClassChip")
        self._index = int(index)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["xs"])

        self.name_pill = QLabel(str(rule.name), self)
        self.name_pill.setObjectName("ClassChipName")
        self.name_pill.setStyleSheet(
            f"background:{palette['chip_class']}; color:{palette['bg']};"
            f"border-radius:8px; padding:2px 8px;")
        row.addWidget(self.name_pill)

        if rule.random_complement:
            text = "the rest, at random"
        elif rule.value is None:
            text = "\u2014"
        else:
            text = _key(rule.value)
        self.value_pill = QLabel(text, self)
        self.value_pill.setObjectName("ClassChipValue")
        self.value_pill.setStyleSheet(
            f"background:{palette['chip_value']}; color:{palette['bg']};"
            f"border-radius:8px; padding:2px 8px;")
        row.addWidget(self.value_pill)

        if rule.column:
            source = QLabel(f"({rule.column})", self)
            source.setObjectName("ClassChipSource")
            source.setStyleSheet(f"color:{palette['fg_muted']};")
            row.addWidget(source)

        self._close = QPushButton("\u00d7", self)
        self._close.setObjectName("ClassChipRemove")
        self._close.setFixedWidth(20)
        self._close.setToolTip(f"Remove the class {rule.name!r}")
        self._close.clicked.connect(self._on_removed)
        row.addWidget(self._close)
        row.addStretch(1)

    def _on_removed(self) -> None:
        self.removed.emit(self._index)


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

        # TWO FIELDS, SIDE BY SIDE -- the gesture the maintainer asked for:
        # "2 fields next to each other with class then value". Typing a class
        # and its value and pressing Enter in either field adds one chip, so
        # the whole interaction is two words and a keystroke, and it is the
        # SAME in metadata mode and annotation mode. Only the columns the
        # picker above offers differ between the two bases.
        entry = QHBoxLayout()
        entry.setContentsMargins(0, 0, 0, 0)
        entry.setSpacing(SPACING["xs"])
        self.class_field = QLineEdit(self)
        self.class_field.setPlaceholderText("Class")
        self.class_field.setToolTip(
            "The name of the class. It becomes the teal bubble, and it is the "
            "name that appears in every figure and results table afterwards.")
        self.class_field.returnPressed.connect(self.add_typed_class)
        entry.addWidget(self.class_field, 1)
        self.value_field = QLineEdit(self)
        self.value_field.setPlaceholderText("Value")
        self.value_field.setToolTip(
            "The value in the chosen column that makes an object a member of "
            "this class. It becomes the green bubble.")
        self.value_field.returnPressed.connect(self.add_typed_class)
        entry.addWidget(self.value_field, 1)
        self._add_typed = QPushButton("Add", self)
        self._add_typed.clicked.connect(self.add_typed_class)
        entry.addWidget(self._add_typed)
        outer.addLayout(entry)

        # The bubbles themselves.
        self.chips_host = QWidget(self)
        self.chips_host.setObjectName("ClassChips")
        self._chips_layout = QVBoxLayout(self.chips_host)
        self._chips_layout.setContentsMargins(0, 0, 0, 0)
        self._chips_layout.setSpacing(SPACING["xs"])
        outer.addWidget(self.chips_host)

        # The table stays, hidden, as the accessible/edit-a-name surface and
        # because every existing test and integration reads `self.table`.
        # Removing it would be a second change riding on this one.
        self.table = QTreeWidget(self)
        self.table.setVisible(False)
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
        """Show ``value``, whether it is the dict, the old list, or a string.

        A settings CSV stores ``repr(value)``, so ``classes`` comes back as
        the TEXT ``"['nc', 'pc']"``. Without the string branch below, that
        matched neither the Mapping nor the list arm, fell through to an empty
        table, and reported SUCCESS: ``apply_settings_dict`` returned
        ``applied=1`` while ``collect()['classes']`` was ``{}``. The class
        names were dropped without a word -- and because ``{}`` is a Mapping,
        ``classify_classes.normalize_settings`` then skipped its own
        legacy-translation branch too, so nothing downstream recovered them.
        Every other list-shaped key survived that round trip; this was the one
        that decides what gets trained.
        """
        self._rules = []
        if isinstance(value, str):
            text = value.strip()
            if text.startswith(("[", "(", "{")):
                try:
                    value = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    LOG.debug("classes is a string that does not parse: %r",
                              text)
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

    def add_typed_class(self) -> None:
        """Add one class from the two fields. The chip appears; the fields clear.

        A class with no name is refused rather than added blank -- `ClassRule`
        raises on it anyway, and the message a user needs is which field is
        empty, not a traceback.
        """
        name = self.class_field.text().strip()
        value = self.value_field.text().strip()
        if not name:
            self._say("give the class a name first")
            return
        column = self.column.currentText().strip()
        if not column:
            self._say("choose the column the value comes from")
            return
        if not value:
            self._say(f"give {name!r} a value in {column!r}, or use "
                      f"'Add random rest' for the objects nothing else claims")
            return
        if any((r.column, _key(r.value)) == (column, _key(value))
               for r in self._rules):
            self._say(f"{column}={value} is already a class")
            return
        try:
            self._rules.append(
                ClassRule(name=name, column=column, value=value))
        except ClassDefinitionError as exc:
            self._say(str(exc))
            return
        self.class_field.clear()
        self.value_field.clear()
        self.class_field.setFocus()
        self._rebuild()
        self._say(f"added {name}")

    def remove_at(self, index: int) -> None:
        """Remove the class a chip's \u00d7 belongs to."""
        if 0 <= int(index) < len(self._rules):
            del self._rules[int(index)]
            self._rebuild()

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
        self._rebuild_chips()
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

    def _rebuild_chips(self) -> None:
        """Redraw the bubbles from `self._rules`.

        Cleared and rebuilt rather than diffed: the list is a handful of
        classes, and a diff here would be the second place the order lives.
        """
        from ..theme import active_palette

        while self._chips_layout.count():
            item = self._chips_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        palette = active_palette()
        for index, rule in enumerate(self._rules):
            chip = ClassChip(index, rule, palette, self.chips_host)
            chip.removed.connect(self.remove_at)
            self._chips_layout.addWidget(chip)

    def _emit(self) -> None:
        self.value_changed.emit(self.value())

    def _say(self, message: str) -> None:
        self._hint.setText(message)


def _key(value: Any) -> str:
    """A value's identity for de-duplication, insensitive to 1 vs 1.0."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
