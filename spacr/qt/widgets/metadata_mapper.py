"""One-dialog recovery for missing metadata columns."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...metadata_resolution import (
    MetadataDecision,
    MetadataRequest,
    MetadataResolutionRequired,
    ResolutionResult,
    resolve_metadata_columns,
)
from ...schema import WellParseError, parse_well
from ..dialogs import detach_from_window_manager


class MetadataColumnDialog(QDialog):
    """Resolve every missing canonical column in one editable table."""

    def __init__(self, request: MetadataRequest, parent: Optional[QWidget] = None):
        super().__init__(parent)
        detach_from_window_manager(self)
        self.request = request
        self.setWindowTitle("Match metadata columns")
        self.setMinimumWidth(720)
        self._selectors = {}

        outer = QVBoxLayout(self)
        intro = QLabel(
            "spaCR needs the metadata columns below. Choose the column in "
            "your table that means each one. All missing columns are applied "
            "together, so the run can continue without restarting.", self)
        intro.setWordWrap(True)
        outer.addWidget(intro)

        grid = QGridLayout()
        grid.addWidget(QLabel("spaCR column"), 0, 0)
        grid.addWidget(QLabel("Your column"), 0, 1)
        grid.addWidget(QLabel("Example values"), 0, 2)
        for row, target in enumerate(request.missing, start=1):
            grid.addWidget(QLabel(target, self), row, 0)
            selector = QComboBox(self)
            selector.setEditable(True)
            selector.addItem("")
            selector.addItems(request.available)
            guess = request.guesses.get(target)
            if guess:
                selector.setCurrentText(guess)
            self._selectors[target] = selector
            grid.addWidget(selector, row, 1)
            example = QLabel("Select a column to preview its values", self)
            example.setWordWrap(True)
            grid.addWidget(example, row, 2)
            selector.currentTextChanged.connect(
                lambda name, label=example: label.setText(
                    ", ".join(request.examples.get(name, ())) or "—"))
        outer.addLayout(grid)

        options = QFormLayout()
        self.well_selector = QComboBox(self)
        self.well_selector.addItem("Do not derive")
        self.well_selector.addItems(request.available)
        options.addRow("Derive rowID/columnID from well column", self.well_selector)
        self.well_preview = QLabel("Choose a well column to preview its mapping", self)
        self.well_preview.setWordWrap(True)
        options.addRow("Well mapping preview", self.well_preview)
        self.well_selector.currentTextChanged.connect(self._preview_wells)

        self.pseudo_selector = QComboBox(self)
        self.pseudo_selector.addItem("Do not generate pseudo wells")
        self.pseudo_selector.addItems(request.available)
        options.addRow("Pseudo wells from condition/folder column", self.pseudo_selector)
        outer.addLayout(options)

        self.remember = QCheckBox("Remember this answer for the current run", self)
        self.remember.setChecked(True)
        outer.addWidget(self.remember)

        save_row = QHBoxLayout()
        self.save_mapping = QCheckBox("Save mapping", self)
        self.save_path = QLineEdit(self)
        self.save_path.setPlaceholderText("metadata_column_map.json")
        browse = QPushButton("Browse…", self)
        browse.clicked.connect(self._browse)
        save_row.addWidget(self.save_mapping)
        save_row.addWidget(self.save_path, 1)
        save_row.addWidget(browse)
        outer.addLayout(save_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save metadata mapping", self.save_path.text(),
            "JSON (*.json);;All files (*)")
        if path:
            self.save_path.setText(path)
            self.save_mapping.setChecked(True)

    def decision(self) -> MetadataDecision:
        mapping = {
            target: selector.currentText().strip()
            for target, selector in self._selectors.items()
            if selector.currentText().strip()
        }
        well = self.well_selector.currentText().strip()
        if self.well_selector.currentIndex() == 0:
            well = None
        pseudo = self.pseudo_selector.currentText().strip()
        if self.pseudo_selector.currentIndex() == 0:
            pseudo = None
        path = self.save_path.text().strip() if self.save_mapping.isChecked() else None
        return MetadataDecision(
            column_map=mapping,
            well_column=well,
            pseudo_source=pseudo,
            allow_pseudo=bool(pseudo),
            save_path=path or None,
            remember=self.remember.isChecked(),
        )

    def _preview_wells(self, source: str) -> None:
        if self.well_selector.currentIndex() == 0:
            self.well_preview.setText("Choose a well column to preview its mapping")
            return
        preview = []
        for value in self.request.examples.get(source, ()):
            try:
                row, column = parse_well(value, strict=True)
                preview.append(f"{value} → {row}/{column}")
            except WellParseError:
                preview.append(f"{value} → not recognised")
        self.well_preview.setText(", ".join(preview) or "No non-empty values to preview")


def resolve_metadata_with_dialog(
        frame, required: Iterable[str], *, parent: Optional[QWidget] = None,
        cache_key: Optional[str] = None) -> ResolutionResult:
    """Resolve a frame interactively; cancellation remains an explicit stop."""
    def prompt(request: MetadataRequest) -> MetadataDecision:
        dialog = MetadataColumnDialog(request, parent)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            raise MetadataResolutionRequired(request.missing, request.available)
        return dialog.decision()

    return resolve_metadata_columns(
        frame, required, prompt=prompt, cache_key=cache_key)
