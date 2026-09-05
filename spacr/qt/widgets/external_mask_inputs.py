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
from .sortable_table import install_sorting, table_item


class ExternalMaskInputWidget(QWidget):
    """Group dropped files and let the user correct every inferred role.

    :param value: the groups already saved. ``None`` opens with none, which
        is the ordinary case: the groups are built by dropping files in.
    :param parent: parent widget.
    """

    value_changed = Signal()

    def __init__(self, value: Any = None, parent=None):
        """Build the external-mask input table.

        :param value: the input groups to start with.
        :param parent: parent widget, or ``None``.
        """
        super().__init__(parent)
        self._groups: List[InputGroup] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self._table = QTableWidget(0, 6, self)
        install_sorting(self._table)
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
        """Replace every group from a settings value.

        :param value: the groups, as the settings dict carries them.
        """
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
        """The groups worth saving, in the shape the settings dict wants.

        IGNORED GROUPS ARE DROPPED. "Ignore" is the user saying these files
        are not part of the run, so carrying them into the settings would
        make the next reader wonder what they are for.

        :returns: one dict per kept group.
        """
        return [group.to_dict() for group in self._groups
                if group.role != "ignore"]

    def groups(self) -> List[InputGroup]:
        """Every group, ignored ones included.

        :returns: the groups, in display order.
        """
        return [InputGroup.from_value(group.to_dict())
                for group in self._groups]

    def group_count(self) -> int:
        """How many groups the files fell into.

        :returns: the group count.
        """
        return len(self._groups)

    def file_count(self) -> int:
        """How many files were grouped, across all groups.

        :returns: the file count.
        """
        return sum(len(group.paths) for group in self._groups)

    def set_group_role(self, row: int, role: str) -> bool:
        """Correct one group's inferred role.

        REFUSES AN UNKNOWN ROLE rather than storing it: the role decides how
        the files are read, and an unrecognised one would fail later, in the
        run, rather than here where the user can see what they typed.

        :param row: which group.
        :param role: the role's name.
        :returns: True when the row and role were both valid.
        """
        if role not in ROLES or not 0 <= row < len(self._groups):
            return False
        self._groups[row].role = role
        self._rebuild()
        self.value_changed.emit()
        return True

    def set_group_object_type(self, row: int, object_type: Any) -> bool:
        """Correct one group's object type.

        An empty or unassigned value clears the type rather than storing a
        placeholder string, so the settings say "not chosen" rather than
        naming an object that does not exist.

        :param row: which group.
        :param object_type: the object type, or None to clear it.
        :returns: True when the row was valid.
        """
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
        """Drop the selected groups."""
        rows = {index.row() for index in self._table.selectedIndexes()}
        groups = sorted({self._group_of_row(row) for row in rows} - {-1},
                        reverse=True)
        for group in groups:
            del self._groups[group]
        if groups:
            self._rebuild()
            self.value_changed.emit()

    def _group_of_row(self, row: int) -> int:
        """The group a drawn row stands for, or -1."""
        item = self._table.item(row, 0)
        index = None if item is None else item.data(Qt.UserRole)
        if index is None or not 0 <= int(index) < len(self._groups):
            return -1
        return int(index)

    def _rebuild(self) -> None:
        """Redraw the table, one row per input group.

        Each row's first cell carries the index of the group it was built from:
        the table sorts, so the third row is not the third group after a header
        click, and the per-row combo boxes have to write back to the right one.

        The object-type box is enabled only for a group being used as a mask --
        an intensity image has no object type to name.
        """
        self._table.setRowCount(len(self._groups))
        for row, group in enumerate(self._groups):
            source = group.root
            if len(group.paths) == 1:
                source = group.paths[0]
            source_item = table_item(source)
            # Which group the row was built from. The table sorts, so the
            # third row is not the third group after a header click.
            source_item.setData(Qt.UserRole, row)
            source_item.setToolTip("\n".join(group.paths[:20]))
            self._table.setItem(row, 0, source_item)
            detected = (
                f"mask · {group.object_type or 'unassigned'}"
                if group.role == "mask" else group.role
            )
            detected_item = table_item(detected)
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

            count = table_item(str(len(group.paths)))
            count.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 4, count)
            confidence = table_item(f"{group.confidence:.0%}")
            confidence.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 5, confidence)

    def _role_changed(self, row: int, role: str) -> None:
        """Change what one group is used as, and re-gate its object type.

        :param row: the group's index.
        :param role: the new role.
        """
        if 0 <= row < len(self._groups):
            self._groups[row].role = role
            object_box = self._table.cellWidget(row, 3)
            if object_box is not None:
                object_box.setEnabled(role == "mask")
            self.value_changed.emit()

    def _object_changed(self, row: int, value: Any) -> None:
        """Change which object a mask group labels.

        :param row: the group's index.
        :param value: the new object type; anything not in the known set clears
            it rather than being stored as a name nothing will match.
        """
        if 0 <= row < len(self._groups):
            self._groups[row].object_type = (
                str(value) if value in OBJECT_TYPES else None)
            self.value_changed.emit()

    def _pick_files(self) -> None:
        """Ask for image and mask files and add them."""
        paths, _selected = QFileDialog.getOpenFileNames(
            self, "Choose intensity images and label masks", "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*)")
        if paths:
            self.add_paths(paths)

    def _pick_folder(self) -> None:
        """Ask for a folder of images or masks and add it."""
        path = QFileDialog.getExistingDirectory(
            self, "Choose a folder containing images or masks")
        if path:
            self.add_paths([path])


__all__ = ["ExternalMaskInputWidget"]
