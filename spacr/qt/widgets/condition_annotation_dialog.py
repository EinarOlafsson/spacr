"""Assign condition labels using metadata rules and stable multirow drag/drop."""
from __future__ import annotations

import copy
import json

import pandas as pd
from PySide6.QtCore import (
    QAbstractTableModel,
    QMimeData,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
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

    def rowCount(self, parent=QModelIndex()):  # noqa: N802, B008
        """Count source rows; child indices have no rows.

        :param parent: Qt parent index.
        """
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent=QModelIndex()):  # noqa: N802, B008
        """Include one diagnostic preview column beside every source column.

        :param parent: Qt parent index.
        """
        return 0 if parent.isValid() else len(self.frame.columns) + 1

    def data(self, index, role=Qt.DisplayRole):
        """Read by source position, never by the DataFrame index label.

        :param index: Source model cell.
        :param role: Qt display or sorting role.
        """
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.UserRole + 1):
            return None
        if index.column() == len(self.frame.columns):
            value = self.preview_values.iloc[index.row()] if self.preview_values is not None else ""
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
            return str(self.frame.columns[section]) if section < len(self.frame.columns) else tr("Preview condition")
        return str(section + 1)

    def set_preview(self, values):
        """Refresh assignment diagnostics without reordering source rows.

        :param values: Condition labels in original source order, or None.
        """
        self.preview_values = values
        if len(self.frame):
            column = len(self.frame.columns)
            self.dataChanged.emit(self.index(0, column), self.index(len(self.frame) - 1, column))

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
        """
        super().__init__(parent)
        self.setObjectName("ConditionBox")
        self.source_model = source_model
        self.manual_rows = list(dict.fromkeys(condition.get("manual_rows", [])))
        self.setFrameShape(QFrame.StyledPanel)
        self.setAcceptDrops(True)
        outer = QVBoxLayout(self)
        top = QHBoxLayout()
        self.name = QLineEdit(condition.get("name", ""), self)
        self.name.setPlaceholderText(tr("Condition name"))
        top.addWidget(self.name)
        self.column = QComboBox(self)
        self.column.addItems([str(c) for c in source_model.frame.columns])
        self.column.setCurrentText(condition.get("metadata_column", self.column.currentText()))
        top.addWidget(QLabel(tr("Column"), self))
        top.addWidget(self.column)
        self.include = QLineEdit(condition.get("include", ""), self)
        self.include.setPlaceholderText(tr("Include regex (blank: manual only)"))
        top.addWidget(QLabel(tr("Include"), self))
        top.addWidget(self.include, 1)
        self.exclude = QLineEdit(condition.get("exclude", ""), self)
        self.exclude.setPlaceholderText(tr("Exclude regex"))
        top.addWidget(QLabel(tr("Exclude"), self))
        top.addWidget(self.exclude, 1)
        remove = QPushButton(tr("Remove condition"), self)
        remove.clicked.connect(lambda: self.remove_requested.emit(self))
        top.addWidget(remove)
        outer.addLayout(top)
        self.count = QLabel(tr("Drop source rows here or enter an include expression."), self)
        outer.addWidget(self.count)
        self.rows = QListWidget(self)
        self.rows.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.rows.setMaximumHeight(115)
        self.rows.setAcceptDrops(False)
        outer.addWidget(self.rows)
        actions = QHBoxLayout()
        remove_rows = QPushButton(tr("Remove selected manual rows"), self)
        remove_rows.clicked.connect(self.remove_selected_rows)
        actions.addWidget(remove_rows)
        clear_rows = QPushButton(tr("Clear manual rows"), self)
        clear_rows.clicked.connect(self.clear_manual_rows)
        actions.addWidget(clear_rows)
        actions.addStretch()
        outer.addLayout(actions)
        self._show_manual_rows()
        for control in (self.name, self.include, self.exclude):
            control.textChanged.connect(self.changed)
        self.column.currentTextChanged.connect(self.changed)

    def definition(self):
        """Return the controls as a reproducible condition rule."""
        return {"name": self.name.text().strip(), "metadata_column": self.column.currentText(),
                "include": self.include.text(), "exclude": self.exclude.text(),
                "manual_rows": list(self.manual_rows)}

    def _show_manual_rows(self):
        """List manual row identities without allocating one widget per source row."""
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
        self._jobs = JobRunner(self, threaded=threaded, app_key="graph_builder")
        self._jobs.job_failed.connect(self._failed)
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
        header = QHBoxLayout()
        header.addWidget(QLabel(tr("Output column"), self))
        self.output_column = QLineEdit(self._initial["column"], self)
        header.addWidget(self.output_column)
        self.add_condition = QPushButton(tr("Add condition"), self)
        self.add_condition.clicked.connect(lambda: self.add_box())
        header.addWidget(self.add_condition)
        outer.addLayout(header)
        note = QLabel(tr("Include selects matching metadata values; a blank include uses manual rows only. "
                         "Exclude removes matching rows, including dropped rows. Drag selected source rows "
                         "into a condition box. Resolve overlapping conditions before Apply. "
                         "Unmatched rows remain blank; source files are preserved."), self)
        note.setWordWrap(True)
        outer.addWidget(note)
        splitter = QSplitter(Qt.Vertical, self)
        source_panel = QWidget(self)
        source_layout = QVBoxLayout(source_panel)
        self.filter = QLineEdit(self)
        self.filter.setPlaceholderText(tr("Filter source metadata or preview conditions"))
        source_layout.addWidget(self.filter)
        self.source_model = ConditionRowsModel(frame, self)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.source_model)
        self.proxy.setFilterKeyColumn(-1)
        self.proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.proxy.setSortRole(Qt.UserRole + 1)
        self.filter.textChanged.connect(self.proxy.setFilterFixedString)
        self.table = ConditionSourceTable(self)
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setDragEnabled(True)
        self.table.setDragDropMode(QAbstractItemView.DragOnly)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setDefaultSectionSize(150)
        self.table.horizontalHeader().setStretchLastSection(True)
        source_layout.addWidget(self.table)
        splitter.addWidget(source_panel)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        boxes_panel = QWidget(self)
        self.box_layout = QVBoxLayout(boxes_panel)
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
        self.preview_button.clicked.connect(self.refresh_preview)
        actions.addWidget(self.preview_button)
        self.apply_button = QPushButton(tr("Apply conditions"), self)
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.accept)
        actions.addWidget(self.apply_button)
        cancel = QPushButton(tr("Cancel"), self)
        cancel.clicked.connect(self.reject)
        actions.addWidget(cancel)
        outer.addLayout(actions)
        self.output_column.textChanged.connect(self._changed)
        for condition in self._initial["conditions"]:
            self.add_box(condition)
        if not self.boxes:
            self.add_box()
        self.refresh_preview()

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
                         "include": "", "exclude": "", "manual_rows": []}
        box = ConditionBox(self.source_model, condition, self)
        box.changed.connect(self._changed)
        box.problem.connect(self._failed)
        box.remove_requested.connect(self.remove_box)
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
        box.deleteLater()
        self._changed()

    def configuration(self):
        """Return a private reproducible copy of the current editor state."""
        definition = copy.deepcopy(self._initial)
        definition["column"] = self.output_column.text().strip()
        definition["conditions"] = [box.definition() for box in self.boxes]
        return definition

    def _changed(self, *_args):
        """Invalidate applied state immediately and debounce full-table previews."""
        self._jobs.cancel()
        self.apply_button.setEnabled(False)
        self.result_frame = None
        self._timer.start()

    def refresh_preview(self):
        """Validate patterns and assignments off-thread before enabling Apply."""
        self._timer.stop()
        self._jobs.cancel()
        self.apply_button.setEnabled(False)
        definition = self.configuration()
        self.status.setText(tr("Checking condition assignments…"))
        def work():
            # Carry failures through the generation-guarded result channel too;
            # a slow invalid draft must not disable a newer valid preview.
            try:
                report = preview(self.frame, definition, self.source)
                result = None
                if not len(report.overlaps):
                    result = self.frame.copy()
                    result[definition["column"]] = report.values.array
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
        self.source_model.set_preview(report.values)
        for box in self.boxes:
            box.count.setText(tr("{count:,} matching rows; {manual:,} manual rows",
                                 count=report.counts.get(box.name.text().strip(), 0),
                                 manual=len(box.manual_rows)))
        self.status.setText(tr("{assigned:,} assigned; {unmatched:,} unmatched; {overlaps:,} overlapping rows.",
                               assigned=len(self.frame) - report.unmatched,
                               unmatched=report.unmatched, overlaps=len(report.overlaps)))
        if len(report.overlaps):
            self.status.setText(self.status.text() + " " + tr("Resolve overlaps before applying."))
        self.apply_button.setEnabled(not len(report.overlaps))

    def _failed(self, message):
        """Preserve the source and applied annotations after invalid input.

        :param message: Validation or source mismatch explanation.
        """
        self.apply_button.setEnabled(False)
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
        super().accept()

    def reject(self):
        """Discard draft edits and stop delivery of pending preview results."""
        self._timer.stop()
        self._jobs.shutdown()
        super().reject()
