"""Assign condition labels using metadata rules and stable multirow drag/drop."""
from __future__ import annotations

import copy
import csv
import io
import json
import re

import pandas as pd
from PySide6.QtCore import (
    QAbstractTableModel,
    QMimeData,
    QModelIndex,
    QSize,
    QSortFilterProxyModel,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ...condition_annotations import new_definition, preview, table_identity
from ..i18n import tr
from ..job_runner import JobRunner
from ..theme import RADIUS, SPACING, register_widget_qss
from .sortable_table import install_sorting


def _condition_qss(palette, opacity=None):
    """Keep condition drop boxes visible using the active theme's border."""
    return f"""
    QFrame#ConditionBox {{
        border: 1px solid {palette['border']};
        border-radius: {RADIUS['sm']}px;
        background: {palette['surface']};
    }}
    """


register_widget_qss("ConditionAnnotations", _condition_qss, replace=True)

ROWS_MIME = "application/x-spacr-condition-rows"
_COLUMN_MIME = "application/x-spacr-annotation-column"
_PART_MIME = "application/x-spacr-annotation-part"


class _CriterionRow(QWidget):
    """One readable predicate, with literal operators separated from regex."""

    changed = Signal()
    remove_requested = Signal(object)

    def __init__(self, columns, criterion=None, parent=None):
        """One criterion row: a column, an operator and a text value."""
        super().__init__(parent)
        criterion = criterion or {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.column = QComboBox(self)
        self.column.addItems(columns)
        requested = criterion.get("metadata_column", "")
        if requested and requested not in columns:
            self.column.addItem(requested)
        self.column.setCurrentText(criterion.get("metadata_column", self.column.currentText()))
        self.column.setToolTip(tr("Choose a source column or an earlier generated column."))
        self.operator = QComboBox(self)
        for caption, value in ((tr("contains"), "contains"), (tr("does not contain"), "not_contains"),
                               (tr("equals"), "equals"), (tr("does not equal"), "not_equals"),
                               (tr("starts with"), "starts_with"), (tr("ends with"), "ends_with"),
                               (tr("matches regex"), "regex"), (tr("does not match regex"), "not_regex")):
            self.operator.addItem(caption, value)
        self.operator.setCurrentIndex(max(0, self.operator.findData(criterion.get("operator", "contains"))))
        self.operator.setToolTip(tr("Only the regex operators interpret patterns. Other operators compare literal text and are case-sensitive."))
        self.value = QLineEdit(str(criterion.get("value", "")), self)
        self.value.setPlaceholderText(tr("Text to match"))
        self.value.setToolTip(tr("Enter the text to test. Equals can match an empty string; missing values never match."))
        self.remove = QPushButton(tr("Remove rule"), self)
        self.remove.setToolTip(tr("Remove this criterion from the condition box."))
        for widget in (self.column, self.operator, self.value, self.remove):
            layout.addWidget(widget)
        layout.setStretch(2, 1)
        self.column.currentTextChanged.connect(self.changed)
        self.operator.currentIndexChanged.connect(self.changed)
        self.value.textChanged.connect(self.changed)
        self.remove.clicked.connect(lambda: self.remove_requested.emit(self))

    def _definition(self):
        """The criterion this row describes, as a dict."""
        return {"metadata_column": self.column.currentText(), "operator": self.operator.currentData(),
                "value": self.value.text()}


class _ColumnPalette(QListWidget):
    """Drag available column tokens into the composition sequence."""

    def startDrag(self, supported_actions):  # noqa: N802
        """Copy the selected column token.

        :param supported_actions: Qt drag action flags.
        """
        if self.currentItem() is None:
            return
        mime = QMimeData()
        mime.setData(_COLUMN_MIME, self.currentItem().text().encode())
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)


class _TemplateParts(QListWidget):
    """Ordered editable text and column tokens with native internal moves."""

    changed = Signal()

    def __init__(self, parent=None):
        """An ordered strip of column and fixed-text parts, reorderable by drag."""
        super().__init__(parent)
        self.allowed_columns = set()
        self._next_token = 0
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setFlow(QListView.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(QListView.Adjust)
        self.setSpacing(SPACING["xs"])
        self.model().rowsMoved.connect(self.changed)
        self.model().rowsInserted.connect(self.changed)
        self.model().rowsRemoved.connect(self.changed)
        self.itemChanged.connect(self._token_edited)

    def _token_edited(self, item):
        """Keep token widths readable after editing fixed text."""
        size = QSize(max(48, self.fontMetrics().horizontalAdvance(item.text()) + 24),
                     self.fontMetrics().height() + 18)
        if item.sizeHint() != size:
            item.setSizeHint(size)
        self.changed.emit()

    def _append_part(self, part, row=None):
        """Add one column or fixed-text part, at ``row`` or at the end."""
        item = QListWidgetItem("{" + part.get("column", "") + "}" if part["kind"] == "column" else part.get("text", ""))
        item.setData(Qt.UserRole, part["kind"])
        self._next_token += 1
        item.setData(Qt.UserRole + 2, self._next_token)
        if part["kind"] == "column":
            item.setData(Qt.UserRole + 1, part["column"])
        if part["kind"] == "text":
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            item.setToolTip(tr("Fixed text. Double-click to edit; drag to reorder."))
        else:
            item.setToolTip(tr("Column value. Drag to reorder or select and remove this token."))
        item.setSizeHint(QSize(max(48, self.fontMetrics().horizontalAdvance(item.text()) + 24),
                               self.fontMetrics().height() + 18))
        self.insertItem(self.count() if row is None else row, item)
        self.changed.emit()
        return item

    def _parts(self):
        """The parts in their shown order, as dicts."""
        return [{"kind": self.item(i).data(Qt.UserRole),
                 "column" if self.item(i).data(Qt.UserRole) == "column" else "text":
                 self.item(i).data(Qt.UserRole + 1) if self.item(i).data(Qt.UserRole) == "column" else self.item(i).text()}
                for i in range(self.count())]

    def _drag_mime(self):
        """Bind an internal move to this widget and an immutable token id."""
        if self.currentItem() is None:
            return None
        mime = QMimeData()
        mime.setData(_PART_MIME, json.dumps({"owner": id(self),
                                          "token": self.currentItem().data(Qt.UserRole + 2)}).encode())
        return mime

    def startDrag(self, supported_actions):  # noqa: N802
        """Move the existing token through the explicit insertion handler.

        :param supported_actions: Qt drag action flags.
        """
        mime = self._drag_mime()
        if mime is not None:
            drag = QDrag(self)
            drag.setMimeData(mime)
            drag.exec(Qt.MoveAction)

    def _insertion_row(self, point):
        """Resolve wrapped horizontal token insertion before or after a token."""
        index = self.indexAt(point)
        if not index.isValid():
            return self.count()
        return index.row() + int(point.x() >= self.visualRect(index).center().x())

    def dragEnterEvent(self, event):  # noqa: N802
        """Accept palette columns or native moves.

        :param event: Qt drag entry event.
        """
        if event.mimeData().hasFormat(_COLUMN_MIME) or event.mimeData().hasFormat(_PART_MIME):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):  # noqa: N802
        """Keep the insertion target active for valid token drags.

        :param event: Qt drag move event.
        """
        if event.mimeData().hasFormat(_COLUMN_MIME) or event.mimeData().hasFormat(_PART_MIME):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):  # noqa: N802
        """Insert an available column or move an existing token.

        :param event: Qt drop event.
        """
        if event.mimeData().hasFormat(_COLUMN_MIME):
            column = bytes(event.mimeData().data(_COLUMN_MIME)).decode("utf-8", errors="replace")
            if column not in self.allowed_columns:
                event.ignore()
                return
            self._append_part({"kind": "column", "column": column}, self._insertion_row(event.position().toPoint()))
            event.setDropAction(Qt.CopyAction)
            event.accept()
        elif event.mimeData().hasFormat(_PART_MIME):
            try:
                payload = json.loads(bytes(event.mimeData().data(_PART_MIME)))
                if payload["owner"] != id(self):
                    raise ValueError("different editor")
                row = next(i for i in range(self.count()) if self.item(i).data(Qt.UserRole + 2) == payload["token"])
            except (ValueError, KeyError, TypeError, StopIteration):
                event.ignore()
                return
            target = self._insertion_row(event.position().toPoint())
            item = self.takeItem(row)
            self.insertItem(target - int(row < target), item)
            self.setCurrentItem(item)
            event.setDropAction(Qt.MoveAction)
            event.accept()
            self.changed.emit()
        else:
            super().dropEvent(event)


def _display(value):
    """Render metadata without assigning semantic meaning to missing values.

    :param value: One source cell.
    :returns: Human-readable source value.
    """
    try:
        if pd.isna(value):
            return ""
    except (ValueError, TypeError):
        pass
    return str(value)


class ConditionRowsModel(QAbstractTableModel):
    """Expose source rows in immutable positional order and optional preview labels.

    :param frame: Unmodified source frame.
    :param parent: Owning widget.
    """

    def __init__(self, frame, parent=None):
        """Retain the source frame and its duplicate-safe row tokens.

        :param frame: Source measurements and metadata.
        :param parent: Owning object.
        """
        super().__init__(parent)
        self.frame = frame
        _schema, self.fingerprint, self.tokens = table_identity(frame)
        self.locations = {token: position for position, token in enumerate(self.tokens)}
        self.preview_values = None
        self.preview_columns = {}

    def rowCount(self, parent=QModelIndex()):  # noqa: N802, B008
        """Count source rows; child indices have no rows.

        :param parent: Qt parent index.
        """
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent=QModelIndex()):  # noqa: N802, B008
        """Include all generated preview columns beside the source columns.

        :param parent: Qt parent index.
        """
        return 0 if parent.isValid() else len(self.frame.columns) + max(1, len(self.preview_columns))

    def data(self, index, role=Qt.DisplayRole):
        """Read by source position, never by the DataFrame index label.

        :param index: Source model cell.
        :param role: Qt display or sorting role.
        """
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.UserRole + 1):
            return None
        if index.column() >= len(self.frame.columns):
            offset = index.column() - len(self.frame.columns)
            values = list(self.preview_columns.values())
            value = values[offset].iloc[index.row()] if values else (
                self.preview_values.iloc[index.row()] if self.preview_values is not None else "")
        else:
            value = self.frame.iloc[index.row(), index.column()]
        if role == Qt.UserRole + 1 and pd.api.types.is_number(value):
            return float(value)
        return _display(value)

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        """Show all metadata column names and the working assignment preview.

        :param section: Row or column position.
        :param orientation: Header orientation.
        :param role: Qt role.
        """
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            if section < len(self.frame.columns):
                return str(self.frame.columns[section])
            names = list(self.preview_columns)
            return names[section - len(self.frame.columns)] if names else tr("Preview condition")
        return str(section + 1)

    def set_preview(self, values):
        """Refresh assignment diagnostics without reordering source rows.

        :param values: Ordered output-name/Series mapping, one label Series, or None.
        """
        self.beginResetModel()
        self.preview_columns = dict(values) if isinstance(values, dict) else {}
        self.preview_values = (next(reversed(self.preview_columns.values()))
                               if self.preview_columns else values)
        self.endResetModel()

    def row_summary(self, token):
        """Describe a manually assigned source row with its metadata values.

        :param token: Stable row token from this source model.
        :returns: Source row number and a bounded metadata summary.
        """
        row = self.locations[token]
        columns = list(self.frame.columns)
        preferred = [c for c in columns if str(c) in
                     ("plateID", "rowID", "columnID", "wellID", "fieldID", "filename", "file_name", "object_label")]
        shown = (preferred + [c for c in columns if c not in preferred])[:6]
        values = "; ".join(f"{column}: {_display(self.frame.iloc[row][column])}" for column in shown)
        return f"{row + 1}: {values}"


class ConditionSourceTable(QTableView):
    """Drag selected proxy rows using their original source identities."""

    def _show_sorted(self, model):
        """Show ``model`` with the shared three-state header sorting.

        :param model: the filter proxy over the condition rows.
        """
        self.setModel(model)
        install_sorting(self)

    def startDrag(self, supported_actions):  # noqa: N802
        """Create one token per selected row after sorting/filtering.

        :param supported_actions: Qt drag action flags.
        """
        mime = self.selection_mime()
        if mime is None:
            return
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)

    def selection_mime(self):
        """Encode the current multiselection with the full source fingerprint.

        :returns: Drag MIME data, or None when no rows are selected.
        """
        proxy = self.model()
        model = proxy.sourceModel()
        positions = sorted({proxy.mapToSource(index).row()
                            for index in self.selectionModel().selectedRows()})
        if not positions:
            return None
        mime = QMimeData()
        mime.setData(ROWS_MIME, json.dumps({"fingerprint": model.fingerprint,
                                          "tokens": [model.tokens[i] for i in positions]}).encode())
        return mime


class ConditionBox(QFrame):
    """One named condition with explicit metadata rules and dropped source rows.

    :param source_model: Immutable source rows and token mapping.
    :param condition: Condition definition copied into editable controls.
    :param parent: Owning annotation dialog.
    """

    changed = Signal()
    remove_requested = Signal(object)
    problem = Signal(str)

    def __init__(self, source_model, condition, parent=None):
        """Place metadata column, include and exclude controls at the top.

        :param source_model: Source data model.
        :param condition: Initial condition definition.
        :param parent: Owning widget.

        Old recipe and API editing is preserved: entering a legacy regex
        explicitly switches to its editor rather than silently ignoring the
        expression.
        """
        super().__init__(parent)
        self.setObjectName("ConditionBox")
        condition = copy.deepcopy(condition)
        if condition.get("match_mode") in ("contains", "not_contains", "equals") and "criteria" not in condition:
            condition["criteria"] = [{"metadata_column": condition.get("metadata_column", ""),
                                       "operator": condition["match_mode"], "value": condition.get("match_text", "")}]
        self.source_model = source_model
        self.manual_rows = list(dict.fromkeys(condition.get("manual_rows", [])))
        self.setFrameShape(QFrame.StyledPanel)
        self.setAcceptDrops(True)
        outer = QVBoxLayout(self)
        top = QHBoxLayout()
        self.name = QLineEdit(condition.get("name", ""), self)
        self.name.setPlaceholderText(tr("Value to assign"))
        self.name.setToolTip(tr("Label written to the output column for this condition. Use a nonempty label; several boxes may assign the same label."))
        top.addWidget(QLabel(tr("Value to assign"), self))
        top.addWidget(self.name)
        self.column = QComboBox(self)
        self.column.addItems([str(c) for c in source_model.frame.columns])
        self.column.setCurrentText(condition.get("metadata_column", self.column.currentText()))
        self.column.setToolTip(tr("Metadata column searched by this box's Include and Exclude expressions. Any available source column can be used."))
        self._column_label = QLabel(tr("Column"), self)
        top.addWidget(self._column_label)
        top.addWidget(self.column)
        self.include = QLineEdit(condition.get("include", ""), self)
        self.include.setPlaceholderText(tr("Include regex (blank: manual only)"))
        self.include.setToolTip(tr("Regular expression selecting rows from this box's chosen column. Matches anywhere unless anchored with ^ and $. Leave blank to use only manually dropped rows."))
        self._include_label = QLabel(tr("Include"), self)
        top.addWidget(self._include_label)
        top.addWidget(self.include, 1)
        self.exclude = QLineEdit(condition.get("exclude", ""), self)
        self.exclude.setPlaceholderText(tr("Exclude regex"))
        self.exclude.setToolTip(tr("Remove rows whose chosen metadata matches this expression, including manually dropped rows. Leave blank to exclude nothing."))
        self._exclude_label = QLabel(tr("Exclude"), self)
        top.addWidget(self._exclude_label)
        top.addWidget(self.exclude, 1)
        remove = QPushButton(tr("Remove condition"), self)
        remove.setToolTip(tr("Remove this entire condition from the draft. Source measurements and other condition boxes are preserved."))
        remove.clicked.connect(lambda: self.remove_requested.emit(self))
        top.addWidget(remove)
        outer.addLayout(top)
        matching = QHBoxLayout()
        self.match_mode = QComboBox(self)
        self.match_mode.addItem(tr("Regular expression"), "regex")
        self.match_mode.addItem(tr("Exact values"), "values")
        self.match_mode.addItem(tr("Readable rules"), "criteria")
        self.match_mode.addItem(tr("Manual rows only"), "manual")
        self.match_mode.setToolTip(" ".join((
            tr('Exact values must match the entire table value.'),
            tr('For example, c1 does not match c10.'),
            tr('Regular expression mode uses the Include and Exclude fields above.'),
        )))
        self.match_mode.setCurrentIndex(2 if "criteria" in condition or condition.get("_readable") else
                                       (1 if condition.get("match_mode") == "values" else 0))
        matching.addWidget(self.match_mode)
        self.include_values = QPlainTextEdit(self)
        self.exclude_values = QPlainTextEdit(self)
        for editor, key, caption in ((self.include_values, "include_values", tr("Include exact values")),
                                     (self.exclude_values, "exclude_values", tr("Exclude exact values"))):
            stream = io.StringIO()
            csv.writer(stream, lineterminator="\n").writerows([[value] for value in condition.get(key, [])])
            editor.setPlainText(stream.getvalue().rstrip("\n"))
            editor.setPlaceholderText(caption)
            editor.setToolTip(tr("Enter comma-separated values or one value per line. Whitespace and duplicates are ignored; quote a value containing a comma. Blank Include uses manual rows only."))
            editor.setMaximumHeight(76)
            editor.textChanged.connect(self.changed)
            matching.addWidget(editor, 1)
        outer.addLayout(matching)
        self.criteria_panel = QWidget(self)
        self._criteria_layout = QVBoxLayout(self.criteria_panel)
        self._criteria_layout.setContentsMargins(0, 0, 0, 0)
        criterion_bar = QHBoxLayout()
        criterion_bar.addWidget(QLabel(tr("Assign this value when"), self))
        self.criteria_match = QComboBox(self)
        self.criteria_match.addItem(tr("all rules match"), "all")
        self.criteria_match.addItem(tr("any rule matches"), "any")
        self.criteria_match.setCurrentIndex(1 if condition.get("match") == "any" else 0)
        self.criteria_match.setToolTip(tr("All requires every criterion; any requires at least one criterion."))
        criterion_bar.addWidget(self.criteria_match)
        self.add_criterion = QPushButton(tr("Add rule"), self)
        self.add_criterion.setToolTip(tr("Add another metadata comparison to this condition box."))
        criterion_bar.addWidget(self.add_criterion)
        criterion_bar.addStretch()
        self._criteria_layout.addLayout(criterion_bar)
        self.criteria = []
        for criterion in condition.get("criteria", [{}]):
            self._add_criterion(criterion)
        self.add_criterion.clicked.connect(lambda: self._add_criterion())
        self.criteria_match.currentIndexChanged.connect(self.changed)
        outer.addWidget(self.criteria_panel)
        self.match_mode.currentIndexChanged.connect(self._match_mode_changed)
        self._match_mode_changed()
        self.count = QLabel(tr("Drag table rows into this box, or enter a matching rule."), self)
        outer.addWidget(self.count)
        self.rows = QListWidget(self)
        self.rows.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.rows.setMaximumHeight(115)
        self.rows.setAcceptDrops(False)
        self.rows.setToolTip(tr("Rows assigned by dragging from the source table. Select rows here to remove manual assignments; regex matches are controlled by Include and Exclude."))
        outer.addWidget(self.rows)
        actions = QHBoxLayout()
        remove_rows = QPushButton(tr("Remove selected manual rows"), self)
        remove_rows.setToolTip(tr("Remove the selected manual assignments from this box. Rows can still match its Include expression."))
        remove_rows.clicked.connect(self.remove_selected_rows)
        actions.addWidget(remove_rows)
        clear_rows = QPushButton(tr("Clear manual rows"), self)
        clear_rows.setToolTip(tr("Clear every manually dropped row in this box while keeping its metadata expressions."))
        clear_rows.clicked.connect(self.clear_manual_rows)
        actions.addWidget(clear_rows)
        actions.addStretch()
        outer.addLayout(actions)
        self._show_manual_rows()
        for control in (self.name, self.include, self.exclude):
            control.textChanged.connect(self.changed)
        self.include.textEdited.connect(lambda: self.match_mode.setCurrentIndex(0))
        self.include.textChanged.connect(lambda text: self.match_mode.setCurrentIndex(0)
                                         if text and self.match_mode.currentData() == "criteria" else None)
        self.column.currentTextChanged.connect(self.changed)

    def definition(self):
        """Return the controls as a reproducible condition rule."""
        result = {"name": self.name.text().strip(), "metadata_column": self.column.currentText(),
                  "include": self.include.text(), "exclude": self.exclude.text(),
                  "manual_rows": list(self.manual_rows)}
        if self.match_mode.currentData() == "values":
            result.update(match_mode="values",
                          include_values=_exact_values(self.include_values.toPlainText()),
                          exclude_values=_exact_values(self.exclude_values.toPlainText()))
        elif self.match_mode.currentData() == "criteria":
            result.update(include="")
            criteria = [row._definition() for row in self.criteria]
            result.update(criteria=criteria, match=self.criteria_match.currentData())
            excluded = _exact_values(self.exclude_values.toPlainText())
            if excluded:
                result["match_mode"] = "values"
                result["exclude_values"] = excluded
        elif self.match_mode.currentData() == "manual":
            result.update(include="")
        return result

    def _add_criterion(self, criterion=None):
        """Append one readable metadata comparison."""
        row = _CriterionRow([self.column.itemText(i) for i in range(self.column.count())], criterion, self)
        row.changed.connect(self.changed)
        row.remove_requested.connect(self._remove_criterion)
        self.criteria.append(row)
        self._criteria_layout.addWidget(row)
        self.changed.emit()
        return row

    def _remove_criterion(self, row):
        """Remove an explicit comparison while retaining manual assignments."""
        self.criteria.remove(row)
        self._criteria_layout.removeWidget(row)
        row.hide()
        row.deleteLater()
        self.changed.emit()

    def _match_mode_changed(self, *_args):
        """Show literal list editors only while exact matching is selected."""
        exact = self.match_mode.currentData() == "values"
        readable = self.match_mode.currentData() == "criteria"
        manual = self.match_mode.currentData() == "manual"
        self.criteria_panel.setVisible(readable)
        for control in (self.column, self._column_label):
            control.setVisible(not readable and not manual)
        for control in (self.include, self.exclude, self._include_label, self._exclude_label):
            control.setVisible(not exact and not readable and not manual)
        if readable and (self.exclude.text() or self.exclude_values.toPlainText()):
            self.column.show()
            self._column_label.show()
            if self.exclude.text():
                self.exclude.show()
                self._exclude_label.show()
        self.include_values.setVisible(exact)
        self.exclude_values.setVisible(exact or (readable and bool(self.exclude_values.toPlainText())))
        self.changed.emit()

    def _show_manual_rows(self):
        """List manual row identities without allocating one widget per source row."""
        if self.manual_rows and self.match_mode.currentData() == "criteria" and all(
                not row.value.text() and row.operator.currentData() == "contains" for row in self.criteria):
            self.match_mode.setCurrentIndex(self.match_mode.findData("manual"))
        self.rows.clear()
        for token in self.manual_rows[:200]:
            text = self.source_model.row_summary(token) if token in self.source_model.locations else token
            item = QListWidgetItem(text, self.rows)
            item.setData(Qt.UserRole, token)
        if len(self.manual_rows) > 200:
            item = QListWidgetItem(tr("Showing 200 of {count:,} manual rows", count=len(self.manual_rows)), self.rows)
            item.setFlags(Qt.NoItemFlags)

    def add_mime_rows(self, mime):
        """Accept only tokens from this exact source and keep earlier memberships.

        :param mime: Drag MIME data generated by the source table.
        :returns: Whether the payload was accepted.
        """
        try:
            payload = json.loads(bytes(mime.data(ROWS_MIME)))
            tokens = payload["tokens"]
            if payload["fingerprint"] != self.source_model.fingerprint or any(
                    token not in self.source_model.locations for token in tokens):
                raise ValueError("different source")
        except (ValueError, TypeError, KeyError):
            self.problem.emit(tr("Those rows belong to another source snapshot."))
            return False
        self.manual_rows = list(dict.fromkeys(self.manual_rows + tokens))
        self._show_manual_rows()
        self.changed.emit()
        return True

    def dragEnterEvent(self, event):  # noqa: N802
        """Offer copy only for spaCR condition-row drags.

        :param event: Qt drag-entry event.
        """
        if event.mimeData().hasFormat(ROWS_MIME):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):  # noqa: N802
        """Allow moving a supported row drag within this box.

        :param event: Qt drag-move event.
        """
        if event.mimeData().hasFormat(ROWS_MIME):
            event.acceptProposedAction()

    def dropEvent(self, event):  # noqa: N802
        """Resolve dropped rows by their source tokens, independent of sorting.

        :param event: Qt drop event.
        """
        if self.add_mime_rows(event.mimeData()):
            event.acceptProposedAction()

    def remove_selected_rows(self):
        """Remove only selected manual memberships; regex rules remain visible."""
        selected = {item.data(Qt.UserRole) for item in self.rows.selectedItems()}
        self.manual_rows = [token for token in self.manual_rows if token not in selected]
        self._show_manual_rows()
        self.changed.emit()

    def clear_manual_rows(self):
        """Remove all manual memberships from this condition."""
        self.manual_rows = []
        self._show_manual_rows()
        self.changed.emit()


def _exact_values(text):
    """Read trimmed, duplicate-free CSV or newline-delimited exact values."""
    return list(dict.fromkeys(value.strip() for row in csv.reader(io.StringIO(text), skipinitialspace=True)
                              for value in row if value.strip()))


def _regex_examples(frame, column):
    """Build safe copyable regex examples from a bounded sample of one column.

    :param frame: Source metadata, never modified by example generation.
    :param column: Metadata column chosen for examples.
    :returns: Translatable labels, literal regex patterns and explanations.
    """
    values = []
    if column in frame:
        for value in frame[column].head(256):
            text = _display(value)
            if text and text not in values:
                values.append(text)
                if len(values) == 2:
                    break
    first = values[0] if values else "sample_value"
    second = values[1] if len(values) > 1 else "another_value"
    literal = re.escape(first)
    middle = len(first) // 2
    prefix, suffix = re.escape(first[:middle]), re.escape(first[middle + 1:])
    return [
        (tr("Contains this text"), literal,
         tr("Find this literal text anywhere in the selected column. Regex punctuation from the sample is escaped.")),
        (tr("Either of two values"), "^(?:" + literal + "|" + re.escape(second) + ")$",
         tr("The vertical bar means OR. Add more alternatives with | inside the parentheses; ^ and $ require a whole-value match.")),
        (tr("One variable character (.)"), "^" + prefix + "." + suffix + "$",
         tr("A dot matches one character. This example replaces one character near the middle of a sample value.")),
        (tr("Variable text (.*)"), "^" + prefix + ".*" + suffix + "$",
         tr("Dot-star matches zero or more characters between the fixed parts. Edit those parts to match your filenames or metadata.")),
        (tr("Exact value (^...$)"), "^" + literal + "$",
         tr("^ marks the start and $ marks the end, so extra text before or after the value does not match.")),
        (tr("Ignore case ((?i))"), "(?i)" + literal,
         tr("(?i) makes letter matching case-insensitive. Without it, Include and Exclude distinguish uppercase and lowercase.")),
    ]


class ConditionAnnotationDialog(QDialog):
    """Edit, preview and safely apply labels to any loaded table.

    :param frame: Original unannotated source frame.
    :param source: Source context binding the definition to a file/table/schema.
    :param parent: Owning screen.
    :param definition: Existing annotation to reopen, or None for a new one.
    :param threaded: Evaluate full-table patterns on a worker when true.
    """

    def __init__(self, frame, source, parent=None, *, definition=None, threaded=True):
        """Build the sortable source table and an unbounded collection of boxes.

        :param frame: Immutable source measurements and metadata.
        :param source: Source context.
        :param parent: Owning screen.
        :param definition: Existing annotation definition to copy.
        :param threaded: Use background preview evaluation.
        """
        super().__init__(parent)
        self.frame = frame
        self.source = source
        self._initial = copy.deepcopy(definition) if definition else new_definition(frame, source)
        self.definition = None
        self.result_frame = None
        self.boxes = []
        self._loading_column = True
        self._active_column = 0
        self._columns = copy.deepcopy(self._initial.get("columns") or [{
            "column": self._initial.get("column", "condition"), "kind": "rules",
            "conditions": self._initial.get("conditions", [])}])
        if definition is None:
            self._columns[0]["_new_editor"] = True
        self._jobs = JobRunner(self, threaded=threaded, app_key="graph_builder")
        self._jobs.job_failed.connect(self._failed)
        self._schema_saves = JobRunner(self, threaded=threaded, app_key="graph_builder")
        self._schema_saves.job_failed.connect(self._schema_save_failed)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(180)
        self._timer.timeout.connect(self.refresh_preview)
        self.setWindowTitle(tr("Annotate conditions"))
        self.setObjectName("ConditionAnnotationDialog")
        self.setSizeGripEnabled(True)
        self.resize(1200, 800)
        outer = QVBoxLayout(self)
        outer.setSpacing(SPACING["sm"])
        column_bar = QHBoxLayout()
        self.column_selector = QComboBox(self)
        self.column_selector.setToolTip(tr("Generated columns are evaluated in this order. Select a column to edit its rules or combination."))
        column_bar.addWidget(QLabel(tr("Generated columns"), self))
        column_bar.addWidget(self.column_selector, 1)
        for attribute, text, callback in (("add_column", tr("Add column"), self._add_column),
                               ("remove_column", tr("Remove column"), self._remove_column),
                               ("column_up", tr("Move up"), lambda: self._move_column(-1)),
                               ("column_down", tr("Move down"), lambda: self._move_column(1))):
            button = QPushButton(text, self)
            button.setToolTip(text)
            button.clicked.connect(callback)
            button.setObjectName(attribute)
            setattr(self, attribute, button)
            column_bar.addWidget(button)
        outer.addLayout(column_bar)
        header = QHBoxLayout()
        header.addWidget(QLabel(tr("Output column"), self))
        self.output_column = QLineEdit(self._columns[0]["column"], self)
        self.output_column.setToolTip(tr("Name of the new condition column. An existing source column cannot be overwritten; choose a distinct name."))
        header.addWidget(self.output_column)
        self.column_kind = QComboBox(self)
        self.column_kind.addItem(tr("Assign values"), "rules")
        self.column_kind.addItem(tr("Compose column"), "combine")
        self.column_kind.addItem(tr("Extract text"), "extract")
        self.column_kind.setToolTip(tr("Assign a fixed value, extract text using a capture group, or compose a value from ordered column and text tokens."))
        header.addWidget(self.column_kind)
        self.add_condition = QPushButton(tr("Add condition"), self)
        self.add_condition.setToolTip(tr("Add another named condition box with its own metadata rules and manual row assignments."))
        self.add_condition.clicked.connect(lambda: self.add_box())
        header.addWidget(self.add_condition)
        outer.addLayout(header)
        self.mode_help = QLabel(self)
        self.mode_help.setWordWrap(True)
        outer.addWidget(self.mode_help)
        self.extract_panel = self._build_extract_panel()
        outer.addWidget(self.extract_panel)
        self.template_panel = self._build_template_panel()
        outer.addWidget(self.template_panel)
        self.example = QLabel(self)
        self.example.setWordWrap(True)
        self.example.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self.example)
        self._composition_legacy = False
        self.combine_panel = QWidget(self)
        combination = QHBoxLayout(self.combine_panel)
        self.combine_available = QComboBox(self)
        self.combine_available.setToolTip(tr("Choose a source column or an earlier generated column to append to the combination."))
        combination.addWidget(self.combine_available, 1)
        append = QPushButton(tr("Add input"), self)
        append.setToolTip(tr("Append the selected column to the ordered combination inputs."))
        self.add_combine_input = append
        append.clicked.connect(self._append_combine_input)
        combination.addWidget(append)
        self.combine_inputs = QListWidget(self)
        self.combine_inputs.setMaximumHeight(82)
        self.combine_inputs.setToolTip(tr("Inputs are joined from top to bottom. A row stays blank if any component is missing or empty."))
        combination.addWidget(self.combine_inputs, 1)
        for attribute, text, callback in (("remove_combine_input", tr("Remove input"), self._remove_combine_input),
                               ("combine_input_up", tr("Input up"), lambda: self._move_combine_input(-1)),
                               ("combine_input_down", tr("Input down"), lambda: self._move_combine_input(1))):
            button = QPushButton(text, self)
            button.setToolTip(text)
            button.clicked.connect(callback)
            button.setObjectName(attribute)
            setattr(self, attribute, button)
            combination.addWidget(button)
        combination.addWidget(QLabel(tr("Separator"), self))
        self.combine_separator = QLineEdit("_", self)
        self.combine_separator.setMaximumWidth(80)
        self.combine_separator.setToolTip(tr("Text placed between combination components, such as an underscore. An empty separator joins them directly."))
        combination.addWidget(self.combine_separator)
        outer.addWidget(self.combine_panel)
        splitter = QSplitter(Qt.Vertical, self)
        source_panel = QWidget(self)
        source_layout = QVBoxLayout(source_panel)
        self.filter = QLineEdit(self)
        self.filter.setPlaceholderText(tr("Filter source metadata or preview conditions"))
        self.filter.setToolTip(tr("Case-insensitive literal search across the visible source table. This is not a regex field and only changes which rows you see; it does not change condition assignments."))
        source_layout.addWidget(self.filter)
        self.source_model = ConditionRowsModel(frame, self)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.source_model)
        self.proxy.setFilterKeyColumn(-1)
        self.proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.proxy.setSortRole(Qt.UserRole + 1)
        self.filter.textChanged.connect(self.proxy.setFilterFixedString)
        self.table = ConditionSourceTable(self)
        self.table._show_sorted(self.proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setDragEnabled(True)
        self.table.setToolTip(tr("Click column headers to sort. Select multiple rows with Ctrl or Shift, then drag them into a condition box. Sorting and filtering preserve row identity."))
        self.table.setDragDropMode(QAbstractItemView.DragOnly)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setDefaultSectionSize(150)
        self.table.horizontalHeader().setStretchLastSection(True)
        source_layout.addWidget(self.table)
        splitter.addWidget(source_panel)
        scroll = QScrollArea(self)
        self._rule_scroll = scroll
        scroll.setWidgetResizable(True)
        boxes_panel = QWidget(self)
        self.box_layout = QVBoxLayout(boxes_panel)
        self.regex_guide = self._build_regex_guide(boxes_panel)
        self.box_layout.addWidget(self.regex_guide)
        self.box_layout.addStretch()
        scroll.setWidget(boxes_panel)
        splitter.addWidget(scroll)
        splitter.setSizes([320, 380])
        outer.addWidget(splitter, 1)
        self.status = QLabel(self)
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        actions = QHBoxLayout()
        self.preview_button = QPushButton(tr("Preview assignments"), self)
        self.preview_button.setToolTip(tr("Check expressions and show each condition's matches, unmatched rows and overlaps without modifying the working table."))
        self.preview_button.clicked.connect(self.refresh_preview)
        actions.addWidget(self.preview_button)
        self.save_schema_button = QPushButton(tr("Save schema"), self)
        self.save_schema_button.setObjectName("save_annotation_schema")
        self.save_schema_button.setToolTip(tr("Save the validated rules, extraction patterns and column composition as reusable JSON. Manually dragged row assignments are omitted."))
        self.save_schema_button.setEnabled(False)
        self.save_schema_button.clicked.connect(self._save_schema)
        actions.addWidget(self.save_schema_button)
        self.load_schema_button = QPushButton(tr("Load schema"), self)
        self.load_schema_button.setObjectName("load_annotation_schema")
        self.load_schema_button.setToolTip(tr("Load a JSON schema for the current table and preview all generated columns. Loading does not apply changes or restore manual row assignments."))
        self.load_schema_button.clicked.connect(self._load_schema)
        actions.addWidget(self.load_schema_button)
        self.apply_button = QPushButton(tr("Apply conditions"), self)
        self.apply_button.setEnabled(False)
        self.apply_button.setToolTip(tr("Add all generated columns to the working table after validation. Invalid expressions and overlapping conditions must be resolved first."))
        self.apply_button.clicked.connect(self.accept)
        actions.addWidget(self.apply_button)
        cancel = QPushButton(tr("Cancel"), self)
        cancel.setToolTip(tr("Discard this draft and keep previously applied annotations unchanged."))
        cancel.clicked.connect(self.reject)
        actions.addWidget(cancel)
        outer.addLayout(actions)
        self.output_column.textChanged.connect(self._column_name_changed)
        self.column_kind.currentIndexChanged.connect(self._kind_changed)
        self.combine_separator.textChanged.connect(self._changed)
        self.column_selector.currentIndexChanged.connect(self._switch_column)
        self._refresh_column_selector()
        self._load_column()
        self.refresh_preview()

    def _save_schema(self):
        """Write a validated snapshot without carrying source-bound row tokens."""
        if not self.save_schema_button.isEnabled() or self.definition != self.configuration():
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, tr("Save schema"), "annotation.schema.json", tr("JSON files (*.json)"))
        if not path:
            return
        definition = copy.deepcopy(self.definition)
        self.save_schema_button.setEnabled(False)
        self.status.setText(tr("Saving annotation schema…"))

        def work():
            """Save the schema in a worker; returns the path, rows and any error."""
            from ...condition_annotations import _save_schema
            try:
                return path, _save_schema(path, self.frame, definition, self.source), None
            except (ValueError, OSError) as exc:
                return path, 0, str(exc)

        self._schema_saves.submit(work, self._schema_saved)

    def _schema_saved(self, payload):
        """Report the completed snapshot independently of subsequent draft edits."""
        path, omitted, error = payload
        if error:
            self.status.setText(error)
        elif omitted:
            self.status.setText(tr("Schema saved to {path}. {count:,} manual row assignments were omitted.",
                                   path=path, count=omitted))
        else:
            self.status.setText(tr("Schema saved to {path}.", path=path))
        self.save_schema_button.setEnabled(self.apply_button.isEnabled())

    def _schema_save_failed(self, message):
        """Restore the save action if an unexpected worker failure occurs."""
        self.status.setText(str(message))
        self.save_schema_button.setEnabled(self.apply_button.isEnabled())

    def _load_schema(self):
        """Validate a portable file on a worker before replacing the current draft."""
        path, _filter = QFileDialog.getOpenFileName(
            self, tr("Load schema"), "", tr("JSON files (*.json)"))
        if not path:
            return
        self._timer.stop()
        self._jobs.cancel()
        prior_valid = self.apply_button.isEnabled()
        self.apply_button.setEnabled(False)
        self.save_schema_button.setEnabled(False)
        self.status.setText(tr("Checking annotation schema for this table…"))

        def work():
            """Load a schema in a worker and apply it to a copy of the table."""
            from ...condition_annotations import _load_schema
            try:
                definition, report = _load_schema(path, self.frame, self.source)
                result = None
                if not len(report.overlaps):
                    result = self.frame.copy()
                    for column, values in report.column_values.items():
                        result[column] = values.array
                    result.attrs["condition_annotation"] = copy.deepcopy(definition)
                return definition, report, result, prior_valid
            except (ValueError, OSError) as exc:
                return None, str(exc), None, prior_valid

        self._jobs.submit(work, self._schema_loaded)

    def _schema_loaded(self, payload):
        """Install only a successfully validated schema; never accept the dialog.

        A hand-edited portable recipe can use a newer version for a simpler
        legacy shape, so the editor's normalized recipe is what gets validated.
        """
        definition, report, result, prior_valid = payload
        if definition is None:
            self.status.setText(tr("Schema was not loaded: {reason}", reason=report))
            self.apply_button.setEnabled(prior_valid)
            self.save_schema_button.setEnabled(prior_valid and not self._schema_saves.is_busy())
            return
        self._initial = copy.deepcopy(definition)
        self._columns = copy.deepcopy(definition.get("columns") or [{
            "column": definition["column"], "kind": "rules", "conditions": definition.get("conditions", [])}])
        self._active_column = 0
        self._refresh_column_selector()
        self._load_column()
        if self.configuration() != definition:
            self.refresh_preview()
            return
        self._previewed((definition, report, result))
        self.status.setText(tr("Schema loaded for preview. Review the generated columns, then apply when ready.")
                            + " " + self.status.text())

    def _build_extract_panel(self):
        """Build a source, capture-pattern and capture-group editor."""
        panel = QWidget(self)
        layout = QGridLayout(panel)
        self.extract_source = QComboBox(panel)
        self.extract_source.setObjectName("extract_source")
        self.extract_source.setToolTip(tr("Choose a filename or another source or earlier generated column."))
        self.extract_pattern = QLineEdit(panel)
        self.extract_pattern.setObjectName("extract_pattern")
        self.extract_pattern.setPlaceholderText(r"(?P<cell_type>[^_]+)_(?P<replicate>[^.]+)")
        self.extract_pattern.setToolTip(tr("Capture the text to keep in parentheses. Name captures with {example} to create several output columns at once.", example="(?P<name>...)"))
        self.extract_group = QComboBox(panel)
        self.extract_group.setEditable(True)
        self.extract_group.setObjectName("extract_group")
        self.extract_group.setToolTip(tr("Choose a named capture or its number. Group 0 keeps the whole regex match."))
        self.extract_group.addItems(["1", "0"])
        for column, (caption, widget) in enumerate(((tr("Source column"), self.extract_source),
                                                   (tr("Extraction regex"), self.extract_pattern),
                                                   (tr("Capture group"), self.extract_group))):
            layout.addWidget(QLabel(caption, panel), 0, column)
            layout.addWidget(widget, 1, column)
        layout.setColumnStretch(1, 2)
        self.named_groups_button = QPushButton(tr("Create columns from named groups"), panel)
        self.named_groups_button.setObjectName("create_named_groups")
        self.named_groups_button.setToolTip(tr("Create one output column for each named capture, in pattern order. Existing output and source column names are protected."))
        self.named_groups_button.clicked.connect(self._create_named_groups)
        layout.addWidget(self.named_groups_button, 2, 1)
        self.extract_example = QLineEdit(r"(?P<cell_type>[^_]+)_(?P<replicate>[^.]+)", panel)
        self.extract_example.setReadOnly(True)
        self.extract_example.setToolTip(tr("Copy this example and adapt the separators to your filenames."))
        layout.addWidget(self.extract_example, 3, 1)
        copy_example = QPushButton(tr("Use example"), panel)
        copy_example.setToolTip(tr("Copy the shown example into the extraction regex field."))
        copy_example.clicked.connect(lambda: self.extract_pattern.setText(self.extract_example.text()))
        layout.addWidget(copy_example, 3, 2)
        self.source_sample = QLabel(panel)
        self.source_sample.setWordWrap(True)
        self.source_sample.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.source_sample, 4, 0, 1, 3)
        self.extract_source.currentTextChanged.connect(self._changed)
        self.extract_source.currentTextChanged.connect(self._update_source_sample)
        self.extract_pattern.textChanged.connect(self._capture_groups_changed)
        self.extract_group.currentTextChanged.connect(self._changed)
        return panel

    def _update_source_sample(self, *_args):
        """Show one bounded source example without evaluating any regex."""
        column = self.extract_source.currentText()
        values = self.frame[column] if column in self.frame else getattr(
            getattr(self, "source_model", None), "preview_columns", {}).get(column)
        value = _display(values.iloc[0]) if values is not None and len(values) else ""
        self.source_sample.setText(tr("Source example: {value}", value=value[:240]))

    def _build_template_panel(self):
        """Build a draggable column palette and editable composition sequence."""
        panel = QWidget(self)
        layout = QGridLayout(panel)
        self.template_palette = _ColumnPalette(panel)
        self.template_palette.setObjectName("template_palette")
        self.template_palette.setDragEnabled(True)
        self.template_palette.setMaximumHeight(125)
        self.template_palette.setToolTip(tr("Drag a column into the composition, or double-click to append it. Only source and earlier generated columns are available."))
        self.template_parts = _TemplateParts(panel)
        self.template_parts.setObjectName("template_parts")
        self.template_parts.setMaximumHeight(125)
        self.template_parts.setToolTip(tr("Tokens are joined from left to right. Braces identify column values. Drag to reorder or double-click fixed text to edit it. Missing column values leave the result empty."))
        layout.addWidget(QLabel(tr("Available columns"), panel), 0, 0)
        layout.addWidget(QLabel(tr("Composition order"), panel), 0, 1)
        layout.addWidget(self.template_palette, 1, 0)
        layout.addWidget(self.template_parts, 1, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 3)
        controls = QHBoxLayout()
        self.template_add_column = QPushButton(tr("Add column token"), panel)
        self.template_add_column.setToolTip(tr("Append the selected available column to the composition."))
        self.template_add_column.clicked.connect(self._append_template_column)
        self.template_palette.itemDoubleClicked.connect(self._append_template_column)
        controls.addWidget(self.template_add_column)
        self.template_text = QLineEdit("_", panel)
        self.template_text.setObjectName("template_text")
        self.template_text.setPlaceholderText(tr("Fixed text or separator"))
        self.template_text.setToolTip(tr("Literal text to insert, such as an underscore, space, or treatment prefix."))
        controls.addWidget(self.template_text, 1)
        self.template_add_text = QPushButton(tr("Add text token"), panel)
        self.template_add_text.setToolTip(tr("Append this literal text. Text tokens can be edited or moved independently."))
        self.template_add_text.clicked.connect(lambda: self.template_parts._append_part(
            {"kind": "text", "text": self.template_text.text()}))
        controls.addWidget(self.template_add_text)
        for attribute, caption, callback in (("template_remove", tr("Remove token"), self._remove_template_part),
                                              ("template_up", tr("Move token up"), lambda: self._move_template_part(-1)),
                                              ("template_down", tr("Move token down"), lambda: self._move_template_part(1))):
            button = QPushButton(caption, panel)
            button.setToolTip(caption)
            button.clicked.connect(callback)
            setattr(self, attribute, button)
            controls.addWidget(button)
        layout.addLayout(controls, 2, 0, 1, 2)
        self.template_parts.changed.connect(self._template_changed)
        return panel

    def _capture_groups_changed(self, *_args):
        """List capture names without scanning table data on the GUI thread."""
        selected = self.extract_group.currentText()
        try:
            compiled = re.compile(self.extract_pattern.text())
            choices = list(compiled.groupindex) + [str(i) for i in range(1, compiled.groups + 1)] + ["0"]
        except re.error:
            choices = ["1", "0"]
        self.extract_group.blockSignals(True)
        self.extract_group.clear()
        self.extract_group.addItems(choices)
        self.extract_group.setCurrentText(selected or (choices[0] if choices else "1"))
        self.extract_group.blockSignals(False)
        self._changed()

    def _create_named_groups(self):
        """Create ordered extraction outputs atomically from named captures.

        The active extraction draft is replaced and every other output is
        preserved.
        """
        try:
            names = list(re.compile(self.extract_pattern.text()).groupindex)
        except re.error as exc:
            self._failed(str(exc))
            return
        if not names:
            self._failed(tr("Add named captures such as {example} before creating columns.", example="(?P<cell_type>...)"))
            return
        self._save_column()
        current = self._columns[self._active_column]
        occupied = set(map(str, self.frame.columns)) | {
            c["column"] for i, c in enumerate(self._columns) if i != self._active_column}
        duplicates = occupied.intersection(names)
        if duplicates:
            self._failed(tr("These column names already exist: {names}", names=", ".join(sorted(duplicates))))
            return
        additions = [{"column": name, "kind": "extract", "metadata_column": current["metadata_column"],
                      "pattern": current["pattern"], "group": name} for name in names]
        self._columns[self._active_column:self._active_column + 1] = additions
        self._refresh_column_selector()
        self._load_column()
        self._changed()

    def _append_template_column(self, *_args):
        """Append one selected column token; repeated tokens are intentional."""
        item = self.template_palette.currentItem()
        if item is not None:
            self.template_parts._append_part({"kind": "column", "column": item.text()})

    def _remove_template_part(self):
        """Remove only the selected token."""
        if self.template_parts.currentRow() >= 0:
            self.template_parts.takeItem(self.template_parts.currentRow())
            self._template_changed()

    def _move_template_part(self, direction):
        """Provide keyboard-accessible alternatives to token dragging."""
        row = self.template_parts.currentRow()
        target = row + direction
        if row >= 0 and 0 <= target < self.template_parts.count():
            item = self.template_parts.takeItem(row)
            self.template_parts.insertItem(target, item)
            self.template_parts.setCurrentRow(target)
            self._template_changed()

    def _template_changed(self, *_args):
        """Promote an edited composition to the explicit v3 token recipe."""
        if self._loading_column:
            return
        self._composition_legacy = False
        self.combine_panel.hide()
        self._changed()

    def _save_column(self):
        """Retain editor state before selecting another generated column."""
        item = self._columns[self._active_column]
        item.update(column=self.output_column.text().strip(), kind=self.column_kind.currentData(),
                    conditions=[box.definition() for box in self.boxes],
                    columns=[self.combine_inputs.item(i).text() for i in range(self.combine_inputs.count())],
                    separator=self.combine_separator.text())
        if item["kind"] == "extract":
            group = self.extract_group.currentText()
            item.update(metadata_column=self.extract_source.currentText(), pattern=self.extract_pattern.text(),
                        group=int(group) if group.isdigit() else group)
        elif item["kind"] == "combine" and not self._composition_legacy:
            item.update(kind="template", parts=self.template_parts._parts())

    def _refresh_column_selector(self):
        """Keep the ordered output selector synchronized without emitting edits."""
        self.column_selector.blockSignals(True)
        self.column_selector.clear()
        self.column_selector.addItems([item["column"] for item in self._columns])
        self.column_selector.setCurrentIndex(self._active_column)
        self.column_selector.blockSignals(False)

    def _load_column(self):
        """Load one output editor while preserving immutable source row tokens."""
        self._loading_column = True
        item = self._columns[self._active_column]
        self.output_column.setText(item["column"])
        kind = item.get("kind", "rules")
        self.column_kind.setCurrentIndex(self.column_kind.findData("combine" if kind == "template" else kind))
        for box in self.boxes:
            self.box_layout.removeWidget(box)
            box.hide()
            box.deleteLater()
        self.boxes = []
        self.combine_available.clear()
        available = [str(c) for c in self.frame.columns] + [c["column"] for c in self._columns[:self._active_column]]
        self.combine_available.addItems(available)
        self.combine_inputs.clear()
        self.combine_inputs.addItems(item.get("columns", []))
        self.combine_separator.setText(item.get("separator", "_"))
        self.extract_source.clear()
        self.extract_source.addItems(available)
        if item.get("metadata_column") and item["metadata_column"] not in available:
            self.extract_source.addItem(item["metadata_column"])
        preferred = next((c for c in ("original_filename", "filename", "file_name") if c in available),
                         available[0] if available else "")
        self.extract_source.setCurrentText(item.get("metadata_column", preferred))
        self.extract_pattern.setText(item.get("pattern", ""))
        self.extract_group.setCurrentText(str(item.get("group", 1)))
        self.template_palette.clear()
        self.template_palette.addItems(available)
        self.template_palette.setCurrentRow(0)
        self.template_parts.allowed_columns = set(available)
        self.template_parts.clear()
        parts = item.get("parts", [])
        if kind == "combine":
            parts = []
            for index, name in enumerate(item.get("columns", [])):
                if index:
                    parts.append({"kind": "text", "text": item.get("separator", "_")})
                parts.append({"kind": "column", "column": name})
        for part in parts:
            self.template_parts._append_part(part)
        self._composition_legacy = kind == "combine"
        for condition in item.get("conditions", []):
            self.add_box(condition)
        if not self.boxes and kind == "rules" and item.pop("_new_editor", False):
            self.add_box()
        self._show_kind()
        self._loading_column = False

    def _switch_column(self, index):
        """Select an output without evaluating the table on the GUI thread."""
        if index < 0 or self._loading_column:
            return
        self._save_column()
        self._active_column = index
        self._load_column()
        self._changed()

    def _column_name_changed(self, *_args):
        """Rename the selected output; dependency validation remains explicit."""
        if self._loading_column:
            return
        self._columns[self._active_column]["column"] = self.output_column.text().strip()
        self._refresh_column_selector()
        self._changed()

    def _show_kind(self):
        """Show a short contextual guide and only the active mode's controls."""
        rules = self.column_kind.currentData() == "rules"
        extract = self.column_kind.currentData() == "extract"
        compose = self.column_kind.currentData() == "combine"
        self._rule_scroll.setVisible(rules)
        self.add_condition.setVisible(rules)
        self.combine_panel.setVisible(compose and self._composition_legacy)
        self.template_panel.setVisible(compose)
        self.extract_panel.setVisible(extract)
        self.regex_guide.setVisible(rules and any(box.match_mode.currentData() == "regex" for box in self.boxes))
        if rules:
            self.mode_help.setText(tr("Name the value to assign, then choose which metadata rules must match. You can also drag source rows into a box."))
        elif extract:
            self.mode_help.setText(tr("Capture part of a filename or metadata value. Preview shows the captured text; the source value stays unchanged."))
        else:
            self.mode_help.setText(tr("Drag column tokens into the desired order and insert fixed text or separators. Earlier generated columns can be reused."))

    def _kind_changed(self, *_args):
        """Invalidate a draft when its output changes between rules and combine."""
        if self._loading_column:
            return
        self._composition_legacy = False
        self._show_kind()
        self._changed()

    def _add_column(self):
        """Append a uniquely named rules output and select its editor."""
        self._save_column()
        names = set(map(str, self.frame.columns)) | {c["column"] for c in self._columns}
        number = len(self._columns) + 1
        name = f"condition_{number}"
        while name in names:
            number += 1
            name = f"condition_{number}"
        self._columns.append({"column": name, "kind": "rules", "conditions": [], "_new_editor": True})
        self._active_column = len(self._columns) - 1
        self._refresh_column_selector()
        self._load_column()
        self._changed()

    def _remove_column(self):
        """Remove the selected output while retaining at least one editor."""
        if len(self._columns) <= 1:
            return
        self._columns.pop(self._active_column)
        self._active_column = min(self._active_column, len(self._columns) - 1)
        self._refresh_column_selector()
        self._load_column()
        self._changed()

    def _move_column(self, direction):
        """Move an output in evaluation order; invalid dependencies are reported."""
        target = self._active_column + direction
        if 0 <= target < len(self._columns):
            self._save_column()
            self._columns[self._active_column], self._columns[target] = self._columns[target], self._columns[self._active_column]
            self._active_column = target
            self._refresh_column_selector()
            self._load_column()
            self._changed()

    def _append_combine_input(self):
        """Append a selected available component once."""
        name = self.combine_available.currentText()
        if name and not self.combine_inputs.findItems(name, Qt.MatchExactly):
            self.combine_inputs.addItem(name)
            self._composition_legacy = True
            self._changed()

    def _remove_combine_input(self):
        """Remove the selected combination component."""
        row = self.combine_inputs.currentRow()
        if row >= 0:
            self.combine_inputs.takeItem(row)
            self._changed()

    def _move_combine_input(self, direction):
        """Reorder components independently of output evaluation order."""
        row = self.combine_inputs.currentRow()
        target = row + direction
        if row >= 0 and 0 <= target < self.combine_inputs.count():
            item = self.combine_inputs.takeItem(row)
            self.combine_inputs.insertItem(target, item)
            self.combine_inputs.setCurrentRow(target)
            self._changed()

    def _build_regex_guide(self, parent):
        """Place column-aware, copyable examples above the first condition box.

        :param parent: Scroll-panel owner.
        :returns: Compact guide that never changes an annotation rule itself.
        """
        guide = QFrame(parent)
        layout = QGridLayout(guide)
        layout.setContentsMargins(0, 0, 0, SPACING["sm"])
        explanation = QLabel(tr(
            "Regex help: Include selects matches; Exclude removes them, including dropped rows. "
            "Leave Include blank for a manual-only box. Patterns search within values: | means OR, "
            ". matches one character, .* matches any length, and ^...$ matches the whole value."), guide)
        explanation.setWordWrap(True)
        explanation.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(explanation, 0, 0, 1, 5)
        layout.addWidget(QLabel(tr("Examples for column"), guide), 1, 0)
        self._regex_column = QComboBox(guide)
        self._regex_column.addItems([str(column) for column in self.frame.columns])
        chosen = next((column for column in ("wellID", "columnID", "filename", "plateID")
                       if column in self.frame), self._regex_column.currentText())
        self._regex_column.setCurrentText(chosen)
        self._regex_column.setToolTip(tr("Choose metadata to sample for the examples. Changing a condition box's column updates this selection; examples do not alter the box."))
        layout.addWidget(self._regex_column, 1, 1)
        self._regex_kind = QComboBox(guide)
        self._regex_kind.setToolTip(tr("Choose a regex pattern to learn from. Examples use up to two values from the first 256 source rows; sample_value and another_value are placeholders when values are missing."))
        layout.addWidget(self._regex_kind, 1, 2)
        self._regex_pattern = QLineEdit(guide)
        self._regex_pattern.setToolTip(tr("Copy or edit this example, then paste it into a condition's Include or Exclude field. Copying does not change any condition."))
        layout.addWidget(self._regex_pattern, 1, 3)
        self._copy_regex = QPushButton(tr("Copy regex"), guide)
        self._copy_regex.setToolTip(tr("Copy the exact expression shown here to the clipboard for use in Include or Exclude."))
        self._copy_regex.clicked.connect(lambda: QApplication.clipboard().setText(self._regex_pattern.text()))
        layout.addWidget(self._copy_regex, 1, 4)
        self._regex_help = QLabel(guide)
        self._regex_help.setWordWrap(True)
        self._regex_help.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._regex_help, 2, 0, 1, 5)
        layout.setColumnStretch(3, 1)
        self._regex_column.currentTextChanged.connect(self._refresh_regex_examples)
        self._regex_kind.currentIndexChanged.connect(self._show_regex_example)
        self._refresh_regex_examples()
        return guide

    def _refresh_regex_examples(self, *_args):
        """Regenerate escaped sample patterns after the example column changes."""
        self._regex_examples = _regex_examples(self.frame, self._regex_column.currentText())
        selected = max(0, self._regex_kind.currentIndex())
        self._regex_kind.blockSignals(True)
        self._regex_kind.clear()
        self._regex_kind.addItems([row[0] for row in self._regex_examples])
        self._regex_kind.setCurrentIndex(selected)
        self._regex_kind.blockSignals(False)
        self._show_regex_example(selected)

    def _show_regex_example(self, index):
        """Show the selected pattern as editable, directly copyable plain text.

        :param index: Example type selected in the guide.
        """
        if 0 <= index < len(self._regex_examples):
            _label, pattern, explanation = self._regex_examples[index]
            self._regex_pattern.setText(pattern)
            self._regex_help.setText(explanation)

    def add_box(self, condition=None):
        """Append another named condition without imposing a fixed count.

        :param condition: Existing rule to edit, or None for a blank condition.
        :returns: Newly added ConditionBox.
        """
        if condition is None:
            names = {box.name.text() for box in self.boxes}
            number = len(self.boxes) + 1
            while tr("Condition {number}", number=number) in names:
                number += 1
            condition = {"name": tr("Condition {number}", number=number),
                         "metadata_column": next((c for c in ("wellID", "columnID", "filename", "plateID")
                                                   if c in self.frame), str(self.frame.columns[0]) if len(self.frame.columns) else ""),
                         "include": "", "exclude": "", "manual_rows": [], "_readable": True}
        box = ConditionBox(self.source_model, condition, self)
        available = [str(c) for c in self.frame.columns] + [c["column"] for c in self._columns[:self._active_column]]
        box.column.clear()
        if condition.get("metadata_column") and condition["metadata_column"] not in available:
            available.append(condition["metadata_column"])
        box.column.addItems(available)
        box.column.setCurrentText(condition.get("metadata_column", box.column.currentText()))
        for criterion in box.criteria:
            current = criterion.column.currentText()
            criterion.column.clear()
            criterion.column.addItems(available)
            if current and current not in available:
                criterion.column.addItem(current)
            criterion.column.setCurrentText(current)
        box.changed.connect(self._changed)
        box.match_mode.currentIndexChanged.connect(self._show_kind)
        box.column.currentTextChanged.connect(self._regex_column.setCurrentText)
        box.problem.connect(self._failed)
        box.remove_requested.connect(self.remove_box)
        if not self.boxes:
            self._regex_column.setCurrentText(box.column.currentText())
        self.boxes.append(box)
        self.box_layout.insertWidget(self.box_layout.count() - 1, box)
        self._changed()
        return box

    def remove_box(self, box):
        """Remove a condition from this unsaved draft and recompute diagnostics.

        :param box: ConditionBox being removed.
        """
        self.boxes.remove(box)
        self.box_layout.removeWidget(box)
        box.hide()
        box.deleteLater()
        self._changed()

    def configuration(self):
        """Return a private reproducible copy of the current editor state."""
        self._save_column()
        definition = copy.deepcopy(self._initial)
        columns = []
        for item in self._columns:
            if item.get("kind") == "combine":
                columns.append({key: copy.deepcopy(item[key]) for key in ("column", "kind", "columns", "separator")})
            elif item.get("kind") == "extract":
                columns.append({key: copy.deepcopy(item[key]) for key in ("column", "kind", "metadata_column", "pattern", "group")})
            elif item.get("kind") == "template":
                columns.append({key: copy.deepcopy(item[key]) for key in ("column", "kind", "parts")})
            else:
                columns.append({"column": item["column"], "kind": "rules",
                                "conditions": copy.deepcopy(item.get("conditions", []))})
        conditions = columns[0].get("conditions", [])
        names = [c.get("name", "") for c in conditions]
        v3 = any(c["kind"] in ("extract", "template") or any("criteria" in rule for rule in c.get("conditions", []))
                 for c in columns)
        legacy = not v3 and len(columns) == 1 and columns[0]["kind"] == "rules" and len(names) == len(set(names))
        definition.pop("columns", None)
        definition.pop("column", None)
        definition.pop("conditions", None)
        if legacy:
            definition.update(version=1, column=columns[0]["column"], conditions=conditions)
        else:
            definition.update(version=3 if v3 else 2, columns=columns)
        return definition

    def _changed(self, *_args):
        """Invalidate applied state immediately and debounce full-table previews."""
        if self._loading_column:
            return
        self._jobs.cancel()
        self.apply_button.setEnabled(False)
        self.save_schema_button.setEnabled(False)
        self.result_frame = None
        self._timer.start()
        if hasattr(self, "example"):
            self.example.setText(tr("Preview to see source and generated values."))

    def refresh_preview(self):
        """Check metadata rules and row labels in a background task.

        Enable the ``Apply`` button after validation succeeds without conflicts.
        """
        self._timer.stop()
        self._jobs.cancel()
        self.apply_button.setEnabled(False)
        self.save_schema_button.setEnabled(False)
        definition = self.configuration()
        self.status.setText(tr("Checking condition assignments…"))
        def work():
            """Preview the draft in a worker and apply it to a copy of the table.

            Failures travel through the generation-guarded result channel too,
            so a slow invalid draft cannot disable a newer valid preview.
            """
            try:
                report = preview(self.frame, definition, self.source)
                result = None
                if not len(report.overlaps):
                    result = self.frame.copy()
                    for column, values in report.column_values.items():
                        result[column] = values.array
                    result.attrs["condition_annotation"] = copy.deepcopy(definition)
                return definition, report, result
            except ValueError as exc:
                return definition, str(exc), None

        self._jobs.submit(work, self._previewed)

    def _previewed(self, payload):
        """Expose row counts and overlapping labels using the current source order.

        :param payload: Validated definition and assignment diagnostics.
        """
        definition, report, result = payload
        if isinstance(report, str):
            self._failed(report)
            return
        self.definition = definition
        self.result_frame = result
        self.source_model.set_preview(report.column_values)
        active_report = report.column_previews.get(self.output_column.text().strip(), report)
        source_name = (self.extract_source.currentText() if self.column_kind.currentData() == "extract"
                       else (str(self.frame.columns[0]) if len(self.frame.columns) else ""))
        examples = []
        source_values = self.frame[source_name] if source_name in self.frame else report.column_values.get(source_name)
        for index in range(min(2, len(self.frame))):
            before = _display(source_values.iloc[index]) if source_values is not None else ""
            after = _display(active_report.values.iloc[index])
            examples.append(f"{before[:160]} → {after[:160]}")
        self.example.setText(tr("Example values: {examples}", examples="; ".join(examples)))
        for index, box in enumerate(self.boxes):
            rule_counts = getattr(active_report, "rule_counts", [])
            count = rule_counts[index] if index < len(rule_counts) else active_report.counts.get(box.name.text().strip(), 0)
            box.count.setText(tr("{count:,} matching rows; {manual:,} manual rows",
                                 count=count,
                                 manual=len(box.manual_rows)))
        self.status.setText(tr("{assigned:,} assigned; {unmatched:,} unmatched; {overlaps:,} overlapping rows.",
                               assigned=len(self.frame) - active_report.unmatched,
                               unmatched=active_report.unmatched, overlaps=len(report.overlaps)))
        if len(report.overlaps):
            self.status.setText(self.status.text() + " " + tr("Resolve overlaps before applying."))
        self.apply_button.setEnabled(not len(report.overlaps))
        self.save_schema_button.setEnabled(not len(report.overlaps) and not self._schema_saves.is_busy())

    def _failed(self, message):
        """Preserve the source and applied annotations after invalid input.

        :param message: Validation or source mismatch explanation.
        """
        self.apply_button.setEnabled(False)
        self.save_schema_button.setEnabled(False)
        self.status.setText(str(message))

    def accept(self):
        """Apply only the current validated draft, leaving source rows untouched."""
        if not self.apply_button.isEnabled():
            return
        if self.result_frame is None or self.definition != self.configuration():
            self._failed(tr("Preview the current conditions before applying."))
            return
        self._timer.stop()
        self._jobs.shutdown()
        self._schema_saves.shutdown()
        super().accept()

    def reject(self):
        """Discard draft edits and stop delivery of pending preview results."""
        self._timer.stop()
        self._jobs.shutdown()
        self._schema_saves.shutdown()
        super().reject()
