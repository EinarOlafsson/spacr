"""Editable image/mask detection table for the External Masks module."""
from __future__ import annotations

from typing import Any, Iterable, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...external_masks import (
    InputGroup,
    OBJECT_TYPES,
    ROLES,
    detect_inputs,
)


class ExternalMaskInputWidget(QWidget):
    """Group dropped files and let the user correct every inferred role."""

    value_changed = Signal()

    def __init__(self, value: Any = None, parent=None):
        super().__init__(parent)
        self._groups: List[InputGroup] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self._table = QTableWidget(0, 6, self)
        self._table.setHorizontalHeaderLabels([
            "Source", "Detected as", "Use as", "Mask type", "Files",
            "Confidence",
        ])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 6):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self._table.setMinimumHeight(190)
        outer.addWidget(self._table)

        buttons = QHBoxLayout()
        self._add_files = QPushButton("Add files…", self)
        self._add_folder = QPushButton("Add folder…", self)
        self._remove = QPushButton("Remove selected", self)
        self._add_files.clicked.connect(self._pick_files)
        self._add_folder.clicked.connect(self._pick_folder)
        self._remove.clicked.connect(self.remove_selected)
        buttons.addWidget(self._add_files)
        buttons.addWidget(self._add_folder)
        buttons.addWidget(self._remove)
        buttons.addStretch(1)
        outer.addLayout(buttons)

        self.set_value(value or [])

    def add_paths(self, paths: Iterable[Any]) -> int:
        """Detect and append one drag/drop or picker batch."""
        detected = detect_inputs(list(paths))
        existing = {group.key: group for group in self._groups}
        for group in detected:
            held = existing.get(group.key)
            if held is None:
                self._groups.append(group)
                existing[group.key] = group
            else:
                held.paths = sorted(set([*held.paths, *group.paths]))
                held.confidence = min(held.confidence, group.confidence)
        self._rebuild()
        if detected:
            self.value_changed.emit()
        return sum(len(group.paths) for group in detected)

    def set_value(self, value: Any) -> None:
        self._groups = []
        if isinstance(value, (str, bytes)):
            value = [value]
        values = list(value or [])
        if values and all(isinstance(item, (str, bytes)) for item in values):
            self._groups = detect_inputs(values)
        else:
            self._groups = [InputGroup.from_value(item) for item in values]
        self._rebuild()

    def get_value(self) -> List[dict]:
        return [group.to_dict() for group in self._groups
                if group.role != "ignore"]

    def groups(self) -> List[InputGroup]:
        return [InputGroup.from_value(group.to_dict())
                for group in self._groups]

    def group_count(self) -> int:
        return len(self._groups)

    def file_count(self) -> int:
        return sum(len(group.paths) for group in self._groups)

    def set_group_role(self, row: int, role: str) -> bool:
        if role not in ROLES or not 0 <= row < len(self._groups):
            return False
        self._groups[row].role = role
        self._rebuild()
        self.value_changed.emit()
        return True

    def set_group_object_type(self, row: int, object_type: Any) -> bool:
        if not 0 <= row < len(self._groups):
            return False
        value = None if object_type in (None, "", "unassigned") else str(object_type)
        if value is not None and value not in OBJECT_TYPES:
            return False
        self._groups[row].object_type = value
        self._rebuild()
        self.value_changed.emit()
        return True

    def remove_selected(self) -> None:
        rows = sorted({index.row() for index in self._table.selectedIndexes()},
                      reverse=True)
        for row in rows:
            if 0 <= row < len(self._groups):
                del self._groups[row]
        if rows:
            self._rebuild()
            self.value_changed.emit()

    def _rebuild(self) -> None:
        self._table.setRowCount(len(self._groups))
        for row, group in enumerate(self._groups):
            source = group.root
            if len(group.paths) == 1:
                source = group.paths[0]
            source_item = QTableWidgetItem(source)
            source_item.setToolTip("\n".join(group.paths[:20]))
            self._table.setItem(row, 0, source_item)
            detected = (
                f"mask · {group.object_type or 'unassigned'}"
                if group.role == "mask" else group.role
            )
            detected_item = QTableWidgetItem(detected)
            detected_item.setToolTip(group.reason)
            self._table.setItem(row, 1, detected_item)

            role_box = QComboBox(self._table)
            for role in ROLES:
                role_box.addItem(role.title(), role)
            role_box.setCurrentIndex(max(role_box.findData(group.role), 0))
            role_box.currentIndexChanged.connect(
                lambda _index, r=row, box=role_box:
                self._role_changed(r, str(box.currentData())))
            self._table.setCellWidget(row, 2, role_box)

            object_box = QComboBox(self._table)
            object_box.addItem("Choose…", None)
            for object_type in OBJECT_TYPES:
                object_box.addItem(object_type.title(), object_type)
            index = object_box.findData(group.object_type)
            object_box.setCurrentIndex(max(index, 0))
            object_box.setEnabled(group.role == "mask")
            object_box.currentIndexChanged.connect(
                lambda _index, r=row, box=object_box:
                self._object_changed(r, box.currentData()))
            self._table.setCellWidget(row, 3, object_box)

            count = QTableWidgetItem(str(len(group.paths)))
            count.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 4, count)
            confidence = QTableWidgetItem(f"{group.confidence:.0%}")
            confidence.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 5, confidence)

    def _role_changed(self, row: int, role: str) -> None:
        if 0 <= row < len(self._groups):
            self._groups[row].role = role
            object_box = self._table.cellWidget(row, 3)
            if object_box is not None:
                object_box.setEnabled(role == "mask")
            self.value_changed.emit()

    def _object_changed(self, row: int, value: Any) -> None:
        if 0 <= row < len(self._groups):
            self._groups[row].object_type = (
                str(value) if value in OBJECT_TYPES else None)
            self.value_changed.emit()

    def _pick_files(self) -> None:
        paths, _selected = QFileDialog.getOpenFileNames(
            self, "Choose intensity images and label masks", "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*)")
        if paths:
            self.add_paths(paths)

    def _pick_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose a folder containing images or masks")
        if path:
            self.add_paths([path])


__all__ = ["ExternalMaskInputWidget"]
