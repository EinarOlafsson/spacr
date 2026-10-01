"""Shared table merging workflow for Graph Builder and Gate Editor.

Users can preview standard spaCR aggregation or explicitly configure external
schemas. The custom editor works on a private copy, and applying requires both
acknowledgment and a successful full-data validation.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

from ...derived_tables import (
    allowed_methods,
    column_sample,
    default_definition,
    execute,
    schemas,
)
from ...merge_tables import MergeError, aggregation_for
from ..i18n import tr
from ..job_runner import JobRunner
from ..theme import SPACING

WARNING = (
    "Custom merging changes spaCR's standard table relationships and "
    "aggregation rules. It is intended for non-spaCR databases or deliberately "
    "different schemas. Incorrect settings can link unrelated objects, "
    "duplicate or omit observations, and change measurements used in plots "
    "and gates. Check the join keys, output level, aggregation, and preview "
    "before applying.")

def _combo(values, value, parent):
    """Build a compact choice widget with its current value selected.

    :param values: Available method or relationship names.
    :param value: Initially selected name.
    :param parent: Owning widget.
    """
    widget = QComboBox(parent)
    widget.addItems(values)
    widget.setCurrentText(value)
    return widget


def _columns(text):
    """Parse explicit composite keys, preserving the user's column order.

    :param text: Comma-separated source column names.
    """
    return [value.strip() for value in text.split(",") if value.strip()]


class CustomMergeDialog(QDialog):
    """Edit an isolated explicit key mapping; Cancel has no side effects.

    :param path: Source database path.
    :param definition: Initial merge configuration to copy.
    :param parent: Owning merge popup.
    """

    def __init__(self, path, definition, parent=None):
        """Build isolated mapping controls.

        :param path: Source SQLite database.
        :param definition: Configuration copied before editing.
        :param parent: Owning merge popup.
        """
        super().__init__(parent)
        self.setWindowTitle("Customize merging")
        self.setObjectName("CustomMergeDialog")
        self.setSizeGripEnabled(True)
        self.resize(940, 620)
        self.definition = copy.deepcopy(definition)
        self.path = path
        outer = QVBoxLayout(self)
        outer.setSpacing(SPACING["sm"])
        warning = QLabel(tr(WARNING), self)
        warning.setWordWrap(True)
        outer.addWidget(warning)
        schema_note = QLabel(" ".join((
            tr("Enter the column names for a composite key in matching order, separated by commas."),
            tr("Each output row represents one observation in the base table."),
            tr("Related rows with missing keys cannot match."),
            tr("A left join retains unmatched base observations with missing measurements."),
            tr("For a one-to-many relationship, each related table is aggregated independently before joining."),
        )), self)
        schema_note.setWordWrap(True)
        outer.addWidget(schema_note)
        form = QFormLayout()
        self.base_keys = QLineEdit(", ".join(definition["base_keys"]), self)
        form.addRow("Base observation keys", self.base_keys)
        outer.addLayout(form)
        self.joins = QTableWidget(len(definition["joins"]), 6, self)
        self.joins.setHorizontalHeaderLabels(
            ["Child table", "Base keys", "Child keys", "Relationship", "Join", "Identifiers"])
        self._rows = []
        for row, join in enumerate(definition["joins"]):
            item = QTableWidgetItem(join["table"])
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.joins.setItem(row, 0, item)
            left = QLineEdit(", ".join(join["left_keys"]), self)
            right = QLineEdit(", ".join(join["right_keys"]), self)
            relationship = _combo(["one-to-one", "one-to-many"], join["relationship"], self)
            how = _combo(["left", "inner"], join["how"], self)
            identifiers = QLineEdit(", ".join(join.get("identifiers", [])), self)
            for column, widget in enumerate((left, right, relationship, how, identifiers), 1):
                self.joins.setCellWidget(row, column, widget)
            self._rows.append((left, right, relationship, how, identifiers))
        self.joins.horizontalHeader().setStretchLastSection(True)
        outer.addWidget(self.joins, 1)
        available = schemas(path)
        schema_text = QTextEdit(self)
        schema_text.setReadOnly(True)
        schema_text.setPlainText(tr("Available source columns:") + "\n" + "\n".join(
            table + ": " + ", ".join(c[0] for c in available[table])
            for table in definition["schema"]))
        schema_text.setMaximumHeight(130)
        outer.addWidget(schema_text)
        self.acknowledge = QCheckBox("I understand the warning and will check the merge preview.", self)
        self.acknowledge.setChecked(bool(definition.get("acknowledged")))
        outer.addWidget(self.acknowledge)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(self.acknowledge.isChecked())
        self.acknowledge.toggled.connect(self.buttons.button(QDialogButtonBox.Ok).setEnabled)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        outer.addWidget(self.buttons)

    def accept(self):
        """Commit edited controls to the private copy after acknowledgment."""
        if not self.acknowledge.isChecked():
            return
        self.definition["mode"] = "custom"
        self.definition["acknowledged"] = True
        self.definition["base_keys"] = _columns(self.base_keys.text())
        for join, widgets in zip(self.definition["joins"], self._rows):
            left, right, relationship, how, identifiers = widgets
            join.update(left_keys=_columns(left.text()), right_keys=_columns(right.text()),
                        relationship=relationship.currentText(), how=how.currentText(),
                        identifiers=_columns(identifiers.text()))
        super().accept()


class MergeTablesDialog(QDialog):
    """Select tables, customize aggregation, preview, and create a derived table.

    :param path: SQLite source database.
    :param parent: Owning screen.
    :param selected: Initially checked source tables.
    :param threaded: Validate on a worker; false is intended for interaction tests.
    :param initial_definition: Existing merge to reopen with its relationships preserved.
    """

    def __init__(self, path, parent=None, *, selected=(), threaded=True, initial_definition=None):
        """Build the shared merge workflow.

        :param path: Source SQLite database.
        :param parent: Owning screen.
        :param selected: Initially checked source tables.
        :param threaded: Run full validation on a worker when true.
        :param initial_definition: Existing source-bound merge settings to reuse.
        """
        super().__init__(parent)
        self.path = path
        self.definition = None
        self.result_frame = None
        self._custom = None
        self._overrides = {}
        self._filename_map = copy.deepcopy((initial_definition or {}).get("original_filenames"))
        self._jobs = JobRunner(self, threaded=threaded)
        self._jobs.job_failed.connect(self._failed)
        self.setWindowTitle("Merge tables")
        self.setObjectName("MergeTablesDialog")
        self.setSizeGripEnabled(True)
        self.resize(920, 760)
        outer = QVBoxLayout(self)
        outer.setSpacing(SPACING["sm"])
        self.state = QLabel(self)
        self.state.setWordWrap(True)
        outer.addWidget(self.state)
        choices = QHBoxLayout()
        self.tables = QListWidget(self)
        available = schemas(path)
        if initial_definition:
            selected = [initial_definition["base"]] + [j["table"] for j in initial_definition["joins"]]
        selected = list(selected) or (["cell"] if "cell" in available else list(available)[:1])
        for name in available:
            item = QListWidgetItem(name, self.tables)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in selected else Qt.Unchecked)
        choices.addWidget(self.tables)
        form = QFormLayout()
        self.base = QComboBox(self)
        self.base.addItems(list(available))
        self.base.setCurrentText("cell" if "cell" in selected else selected[0] if selected else "")
        if initial_definition:
            self.base.setCurrentText(initial_definition["base"])
        self.name = QLineEdit("Merged measurements", self)
        form.addRow("Output observation table", self.base)
        form.addRow("Result name", self.name)
        self.customize = QPushButton("Customize merging", self)
        self.customize.clicked.connect(self._customize)
        form.addRow(self.customize)
        self.reset = QPushButton("Reset to spaCR defaults", self)
        self.reset.clicked.connect(self._reset)
        form.addRow(self.reset)
        self.original_filenames = QPushButton(tr("Merge original filenames…"), self)
        self.original_filenames.setObjectName("MergeOriginalFilenames")
        self.original_filenames.setToolTip(tr(
            "Read a conversion or rename manifest to recover filenames from before Yokogawa conversion. "
            "The result gains original_filename and original_path columns for condition annotation. "
            "Source measurements and image files are preserved."))
        self.original_filenames.clicked.connect(self._choose_original_filenames)
        form.addRow(self.original_filenames)
        self.filename_note = QLabel(self)
        self.filename_note.setWordWrap(True)
        form.addRow(self.filename_note)
        self.clear_filenames = QPushButton(tr("Remove filename mapping"), self)
        self.clear_filenames.setToolTip(tr("Remove the mapping from this merge without deleting its file."))
        self.clear_filenames.clicked.connect(self._clear_original_filenames)
        form.addRow(self.clear_filenames)
        choices.addLayout(form)
        outer.addLayout(choices)
        note = QLabel("Aggregation: numeric measurements use the shared spaCR rules "
                      "(area/total/count sum; minimum min; maximum max; median median; "
                      "other numeric measurements mean). Text and IDs use first nonmissing "
                      "value in source order. Override individual columns below. Missing "
                      "measurements remain missing; count reports nonmissing values.", self)
        note.setWordWrap(True)
        outer.addWidget(note)
        self.rules = QTableWidget(0, 3, self)
        self.rules.setHorizontalHeaderLabels(["Table", "Column", "Aggregation"])
        self.rules.horizontalHeader().setStretchLastSection(True)
        outer.addWidget(self.rules, 1)
        self.preview_text = QTextEdit(self)
        self.preview_text.setReadOnly(True)
        outer.addWidget(self.preview_text, 1)
        buttons = QHBoxLayout()
        self.preview = QPushButton("Validate and preview", self)
        self.preview.clicked.connect(self.validate_preview)
        buttons.addWidget(self.preview)
        self.create = QPushButton("Create merged table", self)
        self.create.setEnabled(False)
        self.create.clicked.connect(self.accept)
        buttons.addWidget(self.create)
        cancel = QPushButton("Cancel", self)
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        outer.addLayout(buttons)
        self.tables.itemChanged.connect(self._selection_changed)
        self.base.currentTextChanged.connect(self._selection_changed)
        self.name.textChanged.connect(self._invalidate)
        self._selection_changed()
        if initial_definition:
            self._custom = copy.deepcopy(initial_definition)
            self._overrides = {j["table"]: j.get("overrides", {}).copy()
                               for j in initial_definition["joins"]}
            self.name.setText(initial_definition["name"] + tr(" with original filenames"))
            self._fill_rules()
            self._show_state()

    def _selected(self):
        """Return checked tables plus the explicitly selected base table."""
        return list(dict.fromkeys([self.base.currentText()] + [self.tables.item(i).text()
            for i in range(self.tables.count()) if self.tables.item(i).checkState() == Qt.Checked]))

    def _invalidate(self, *_args):
        """Prevent applying a preview after any configuration edit."""
        self._jobs.cancel()
        self.create.setEnabled(False)
        self.result_frame = None

    def _selection_changed(self, *_args):
        """Rebuild standard defaults when the selected sources change."""
        self._custom = None
        self._overrides = {}
        self._invalidate()
        self._fill_rules()
        self._show_state()

    def configuration(self):
        """Read a reproducible configuration reflecting the current controls."""
        result = copy.deepcopy(self._custom) if self._custom else default_definition(
            self.path, self._selected(), base=self.base.currentText())
        result["name"] = self.name.text().strip()
        for join in result["joins"]:
            join["overrides"] = self._overrides.get(join["table"], {}).copy()
        if result["mode"] == "default":
            result["policy"]["overrides"] = {
                table + "." + column: method
                for table, values in self._overrides.items()
                for column, method in values.items()}
        result.pop("original_filenames", None)
        if self._filename_map:
            result["original_filenames"] = copy.deepcopy(self._filename_map)
            if not result["joins"] and (not self._custom or self._custom.get("mode") == "metadata"):
                result["mode"] = "metadata"
                result["base_keys"] = []
        return result

    def _show_state(self):
        """Display the active mechanism and actual output observation level."""
        filename_only = bool(self._filename_map and len(self._selected()) == 1
                             and (not self._custom or self._custom.get("mode") == "metadata"))
        if filename_only:
            text = " ".join((
                tr("Original filenames are added to {base}.", base=self.base.currentText()),
                tr("All input rows and columns are preserved."),
                tr("Adding names for several image channels or image layers keeps the same number of rows."),
            ))
        elif self._custom and self._custom.get("mode") == "custom":
            text = tr("Custom rules active — one row per {base}. "
                      "Explicit relationships and join types are shown in Customize merging.",
                      base=self.base.currentText())
        else:
            text = tr("spaCR defaults active — one row per {base}. "
                      "Children are aggregated before joining; unmatched measurements stay missing. "
                      "Nucleus uses an inner join; pathogen/organelle retain uninfected cells by default.",
                      base=self.base.currentText())
        self.state.setText(text)
        self.customize.setEnabled(not filename_only)
        self.filename_note.setText(
            tr("Mapping: {path}", path=self._filename_map["map_path"]) if self._filename_map else
            tr("Optional: recover original names before annotating conditions."))
        self.clear_filenames.setVisible(bool(self._filename_map))

    def _choose_original_filenames(self):
        """Choose a recorded source-to-converted filename mapping for preview."""
        from ...original_filenames import discover_maps

        found = discover_maps(self.path)
        start = str(found[0]) if found else str(Path(self.path).resolve().parent)
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Merge original filenames"), start,
            tr("Conversion maps (*.csv *.db *.sqlite *.sqlite3);;All files (*)"))
        if path:
            self._filename_map = {"map_path": str(Path(path).resolve()),
                                  "output_column": "original_filename"}
            self._invalidate()
            self._fill_rules()
            self._show_state()

    def _clear_original_filenames(self):
        """Clear this optional enrichment without touching source files."""
        self._filename_map = None
        if self._custom and self._custom.get("mode") == "metadata":
            self._custom = None
        self._invalidate()
        self._fill_rules()
        self._show_state()

    def _fill_rules(self):
        """Offer type-compatible rules for selected children, excluding join keys."""
        definition = self.configuration()
        self.rules.setRowCount(0)
        for join in definition["joins"]:
            # A small sample supplies UI choices; execution validates full data.
            frame = column_sample(self.path, join["table"])
            identifiers = set(join.get("identifiers", []))
            for column in frame:
                if column in join["right_keys"]:
                    continue
                row = self.rules.rowCount()
                self.rules.insertRow(row)
                for index, text in enumerate((join["table"], column)):
                    item = QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.rules.setItem(row, index, item)
                method = self._overrides.get(join["table"], {}).get(column, aggregation_for(
                    column, numeric=pd.api.types.is_numeric_dtype(frame[column])))
                methods = allowed_methods(frame[column], identifier=column in identifiers)
                combo = _combo(methods, method if method in methods else methods[0], self)
                if definition["mode"] == "default" and join["relationship"] == "one-to-one":
                    combo.setEnabled(False)
                    combo.setToolTip("Default one-to-one rows are attached intact. Customize merging to transform individual values.")
                combo.currentTextChanged.connect(lambda value, t=join["table"], c=column:
                                                 self._rule_changed(t, c, value))
                self.rules.setCellWidget(row, 2, combo)

    def _rule_changed(self, table, column, method):
        """Record a per-column choice and invalidate the previous preview.

        :param table: Source table name.
        :param column: Source measurement column.
        :param method: Compatible aggregation method.
        """
        self._overrides.setdefault(table, {})[column] = method
        self._invalidate()

    def _customize(self):
        """Open the warning and explicit mapping editor on a private copy."""
        dialog = CustomMergeDialog(self.path, self.configuration(), self)
        if dialog.exec() == QDialog.Accepted:
            self._custom = dialog.definition
            self._invalidate()
            self._fill_rules()
            self._show_state()

    def _reset(self):
        """Discard custom relationships and overrides, restoring shared defaults."""
        self._filename_map = None
        self._selection_changed()

    def validate_preview(self):
        """Materialize the full merge off-thread and expose counts and sample rows."""
        self._invalidate()
        try:
            definition = self.configuration()
            if not definition["name"] or definition["name"] in schemas(self.path):
                raise MergeError(tr("Choose a result name different from the source tables."))
        except (ValueError, OSError) as exc:
            self._failed(str(exc))
            return
        self.preview_text.setPlainText(tr("Validating all rows…"))
        self._jobs.submit(lambda: (definition, execute(self.path, definition)), self._validated)

    def _validated(self, payload):
        """Enable creation only for the current successfully validated preview.

        :param payload: Definition and the worker's frame/diagnostics result.
        """
        self.definition, (self.result_frame, report) = payload
        self.definition = self.result_frame.attrs.get("merge_definition", self.definition)
        if self.definition.get("original_filenames"):
            self._filename_map = copy.deepcopy(self.definition["original_filenames"])
        note = (tr("Image/object navigation is unavailable: this merge has no verified spaCR "
                   "image provenance. Plotting and tabular gating remain available.") + "\n"
                if not report["image_provenance"] else "")
        lines = [note + tr("Base {base}: {input_rows:,} rows → {output_rows:,} output rows",
                           base=report["base"], input_rows=report["base_rows"],
                           output_rows=report["output_rows"])]
        for joined in report["joins"]:
            lines.append(tr(
                "{table}: {input_rows:,} input rows, {groups:,} groups; {relationship}, {how} join",
                table=joined["table"], input_rows=joined["input_rows"], groups=joined["groups"],
                relationship=joined["relationship"], how=joined["how"]))
            lines.append("  " + tr("Keys: {left_keys} ← {right_keys}",
                left_keys=", ".join(joined["left_keys"]), right_keys=", ".join(joined["right_keys"])))
            lines.append("  " + tr(
                "Unmatched base rows: {unmatched_base:,}; unmatched related rows: {unmatched_child:,}.",
                unmatched_base=joined["unmatched_base"], unmatched_child=joined["unmatched_child"]))
            lines.append("  " + tr(
                "Related rows with missing keys: {missing_keys:,}.",
                missing_keys=joined["missing_key_rows"]) + " " + tr(
                "Related rows sharing the same join values: {count:,}.",
                count=joined["duplicate_key_rows"]))
        if report.get("original_filenames"):
            metadata = report["original_filenames"]
            lines.append(tr("Original filenames: {matched:,} matched rows; {unmatched:,} unmatched rows. "
                            "Unmatched original names remain blank.\nMapping: {path}",
                            matched=metadata["matched_rows"], unmatched=metadata["unmatched_rows"],
                            path=metadata["map_path"]))
        lines.append("\n" + tr("First 12 output rows:") + "\n" +
                     self.result_frame.head(12).to_string(index=False))
        self.preview_text.setPlainText("\n\n".join(lines))
        self.create.setEnabled(True)

    def _failed(self, message):
        """Keep the popup editable and show an actionable validation error.

        :param message: Failure details from configuration or full validation.
        """
        self.create.setEnabled(False)
        self.preview_text.setPlainText(str(message))

    def accept(self):
        """Apply only a validated current configuration."""
        if self.create.isEnabled() and self.result_frame is not None:
            self._jobs.shutdown()
            super().accept()

    def reject(self):
        """Cancel outstanding work before dismissing the popup."""
        self._jobs.shutdown()
        super().reject()
