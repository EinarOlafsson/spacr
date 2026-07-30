"""Column/value editor for excluding rows from UMAP input data."""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...row_exclusions import normalize_row_exclusions


class _CheckableValueCombo(QComboBox):
    """A compact dropdown that keeps multiple checked values."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText("Choose values…")
        self.setModel(QStandardItemModel(self))
        self.view().pressed.connect(self._toggle_index)

    def _toggle_index(self, index) -> None:
        item = self.model().itemFromIndex(index)
        state = item.checkState()
        item.setCheckState(
            Qt.Unchecked if state == Qt.Checked else Qt.Checked)
        self._refresh_text()

    def set_options(self, options, selected=()) -> None:
        selected_text = {str(value) for value in selected}
        all_values = list(options)
        existing = {str(value) for value in all_values}
        all_values.extend(
            value for value in selected if str(value) not in existing)
        model = self.model()
        model.clear()
        for value in all_values:
            item = QStandardItem(str(value))
            item.setData(value, Qt.UserRole)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setCheckState(
                Qt.Checked if str(value) in selected_text else Qt.Unchecked)
            model.appendRow(item)
        self._refresh_text()

    def checked_values(self) -> list[Any]:
        model = self.model()
        return [
            model.item(row).data(Qt.UserRole)
            for row in range(model.rowCount())
            if model.item(row).checkState() == Qt.Checked
        ]

    def _refresh_text(self) -> None:
        self.lineEdit().setText(
            ", ".join(str(value) for value in self.checked_values()))


class _ExclusionRuleRow(QWidget):
    """One editable ``column is one of values`` rule."""

    column_changed = Signal(str)
    remove_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self.column = QComboBox(self)
        self.column.setEditable(True)
        self.column.setMinimumWidth(150)
        self.column.setToolTip("Database column whose matching rows are removed.")
        self.column.currentTextChanged.connect(self.column_changed.emit)
        row.addWidget(self.column, 2)

        self.values = _CheckableValueCombo(self)
        self.values.setMinimumWidth(180)
        self.values.setToolTip(
            "Check one or more values. A row matching any checked value is "
            "excluded.")
        row.addWidget(self.values, 3)

        remove = QToolButton(self)
        remove.setText("×")
        remove.setToolTip("Remove this exclusion rule")
        remove.clicked.connect(lambda: self.remove_requested.emit(self))
        row.addWidget(remove)

    def set_columns(self, columns) -> None:
        current = self.column.currentText()
        self.column.blockSignals(True)
        self.column.clear()
        self.column.addItems([str(column) for column in columns])
        if current:
            index = self.column.findText(current)
            if index < 0:
                self.column.addItem(current)
                index = self.column.count() - 1
            self.column.setCurrentIndex(index)
        self.column.blockSignals(False)


class RowExclusionEditor(QWidget):
    """Add one or more UMAP row exclusions by choosing columns and values."""

    def __init__(self, value=None, parent=None):
        super().__init__(parent)
        self._rows: list[_ExclusionRuleRow] = []
        self._column_sources: dict[str, list[tuple[Path, str]]] = {}
        self._value_cache: dict[str, list[Any]] = {}

        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(4)
        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(4)
        self._outer.addLayout(self._rows_layout)

        add = QToolButton(self)
        add.setText("+ Add exclusion")
        add.setToolTip("Exclude values from another database column")
        add.clicked.connect(lambda: self._add_row())
        self._outer.addWidget(add, 0, Qt.AlignLeft)
        self.set_value(value)

    def get_value(self) -> dict[str, list[Any]] | None:
        rules: dict[str, list[Any]] = {}
        for row in self._rows:
            column = row.column.currentText().strip()
            values = row.values.checked_values()
            if column and values:
                rules.setdefault(column, []).extend(values)
        return normalize_row_exclusions(rules) or None

    def set_value(self, value) -> None:
        rules = normalize_row_exclusions(value)
        self._clear_rows()
        if rules:
            for column, values in rules.items():
                self._add_row(column, values)
        else:
            self._add_row()

    def set_source(self, source, tables=None) -> None:
        """Populate column/value choices from a dropped measurements DB."""
        self._column_sources = self._discover_columns(source, tables)
        self._value_cache.clear()
        columns = list(self._column_sources)
        for row in self._rows:
            row.set_columns(columns)
            self._refresh_values(row)

    def _add_row(self, column: str = "", values=()) -> None:
        row = _ExclusionRuleRow(self)
        row.remove_requested.connect(self._remove_row)
        row.column_changed.connect(
            lambda _text, r=row: self._refresh_values(r, preserve=False))
        self._rows.append(row)
        self._rows_layout.addWidget(row)
        row.set_columns(self._column_sources)
        if column:
            index = row.column.findText(column)
            if index < 0:
                row.column.addItem(column)
                index = row.column.count() - 1
            row.column.setCurrentIndex(index)
        self._refresh_values(row, selected=values)

    def _remove_row(self, row) -> None:
        if row in self._rows:
            self._rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        if not self._rows:
            self._add_row()

    def _clear_rows(self) -> None:
        for row in self._rows:
            row.setParent(None)
            row.deleteLater()
        self._rows.clear()

    def _refresh_values(self, row, selected=(), preserve: bool = True) -> None:
        column = row.column.currentText().strip()
        if preserve and not selected:
            selected = row.values.checked_values()
        options = self._values_for_column(column)
        row.values.set_options(options, selected)

    def _values_for_column(self, column: str) -> list[Any]:
        if not column:
            return []
        if column in self._value_cache:
            return self._value_cache[column]
        values: list[Any] = []
        seen: set[str] = set()
        quoted_column = '"' + column.replace('"', '""') + '"'
        for db_path, table in self._column_sources.get(column, []):
            quoted_table = '"' + table.replace('"', '""') + '"'
            try:
                with sqlite3.connect(str(db_path)) as connection:
                    rows = connection.execute(
                        f"SELECT DISTINCT {quoted_column} FROM {quoted_table} "
                        f"WHERE {quoted_column} IS NOT NULL LIMIT 501"
                    )
                    for (value,) in rows:
                        key = str(value)
                        if key not in seen:
                            seen.add(key)
                            values.append(value)
                        if len(values) >= 500:
                            break
            except sqlite3.Error:
                continue
            if len(values) >= 500:
                break
        values.sort(key=lambda value: str(value))
        self._value_cache[column] = values
        return values

    @staticmethod
    def _source_paths(source) -> list[Path]:
        if isinstance(source, str):
            text = source.strip()
            if text.startswith(("[", "(")):
                try:
                    source = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    source = text
        values = source if isinstance(source, (list, tuple)) else [source]
        paths: list[Path] = []
        for value in values:
            if not value:
                continue
            path = Path(str(value)).expanduser()
            if path.is_file() and path.suffix.lower() == ".db":
                candidate = path
            elif path.name == "measurements":
                candidate = path / "measurements.db"
            else:
                candidate = path / "measurements" / "measurements.db"
            if candidate.is_file():
                paths.append(candidate)
        return paths

    @classmethod
    def _discover_columns(cls, source, tables=None):
        wanted = set(tables or ())
        found: dict[str, list[tuple[Path, str]]] = {}
        for db_path in cls._source_paths(source):
            try:
                with sqlite3.connect(str(db_path)) as connection:
                    available = [
                        row[0] for row in connection.execute(
                            "SELECT name FROM sqlite_master "
                            "WHERE type='table' ORDER BY name")
                    ]
                    for table in available:
                        if wanted and table not in wanted:
                            continue
                        quoted = '"' + table.replace('"', '""') + '"'
                        for info in connection.execute(
                                f"PRAGMA table_info({quoted})"):
                            found.setdefault(str(info[1]), []).append(
                                (db_path, table))
            except sqlite3.Error:
                continue
        return found
