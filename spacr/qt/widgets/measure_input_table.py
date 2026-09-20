"""The FEATURES table: rows are fields, columns are channels and mask types.

The one control the Measure module has no equivalent of. Measure reads a
``merged/`` folder a pipeline already built, and everything about which file
is which channel was decided by the plate's naming convention long before
Measure saw it. A user who drew masks with a mouse has no naming convention
and no pipeline, so the question has to be asked, and this is where it is
asked: one row per field, one column per channel and per mask type, and three
ways to fill a cell -- drop files on it, click it and browse, or write one
regex that fills the whole table at once.

The model behind it is :class:`spacr.measure.FieldTable`, which is Qt-free
and is what :func:`spacr.measure.measure_from_field_table` consumes. This
module holds no measurement knowledge of its own: it edits that table and
shows what it says.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...measure import (
    FieldRow,
    FieldTable,
    assign_paths_by_regex,
    mask_role_of,
)
from ...object_roles import ORGANELLE_ROLES, organelle_label

#: What the regex box starts with. It reads the shape hand-drawn exports
#: usually have -- ``fov001_C1.tif``, ``fov001_cell_mask.tif`` -- and is
#: shown rather than applied silently, because a regex nobody can see is a
#: rule nobody can correct.
DEFAULT_REGEX = (
    r'(?P<field>.+?)_(?:C(?P<channel>\d+)'
    r'|(?P<mask>cell|nucleus|pathogen|organelle\d*)(?:_mask)?)')

#: The mask columns the checkboxes offer. The organelle slots past the first
#: are reachable by regex and by raising the count; three fixed types and one
#: organelle is what a hand-drawn set almost always needs.
OFFERED_ROLES: Tuple[str, ...] = ('cell', 'nucleus', 'pathogen', 'organelle')

#: Columns that are the row's identity rather than one of its files.
_IDENTITY_COLUMNS = 3


def role_caption(role: str) -> str:
    """What a mask column is called on screen.

    An organelle slot reads as a number -- ``Organelle 2`` -- never as the
    lettered role it is stored under, the same caption the settings forms and
    the Import Project screen give it.

    :param role: a role from :data:`spacr.crops.MASK_PLANE_ORDER`.
    :returns: the column heading.
    """
    if role in ORGANELLE_ROLES:
        return organelle_label(role)
    return f"{str(role).title()} mask"


class MeasureInputTable(QWidget):
    """Edit a :class:`spacr.measure.FieldTable` by dropping, browsing or regex.

    :param parent: parent widget, or ``None``.

    :ivar table_changed: emitted whenever the table or its problems change,
        so the window around it can refresh the derived settings and the Run
        button together.
    """

    table_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None, *,
                 threaded: bool = True) -> None:
        """Build the table, its column controls and its regex box.

        :param parent: parent widget, or ``None``.
        :param threaded: run the walk over dropped folders on a worker.
            ``False`` runs it inline, so a test can drop a folder and read
            the table on the next line.
        """
        super().__init__(parent)
        self._table = FieldTable(rows=[], n_channels=2, roles=('cell',),
                                 plate='drawn')
        self._unassigned: List[Tuple[str, str]] = []
        self._picker = None
        self._known_paths: List[str] = []
        from ..job_runner import JobRunner

        self._scanner = JobRunner(self, threaded=bool(threaded),
                                  app_key="features table scan",
                                  user_visible=False)
        self._scanner.job_failed.connect(self._scan_failed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        shape = QHBoxLayout()
        shape.addWidget(QLabel("Plate name", self))
        self._plate = QLineEdit(self._table.plate, self)
        self._plate.setToolTip(
            "What the fields are filed under in the measurements database. "
            "These images did not come from a plate, so this names where "
            "they did come from rather than claiming a plate that was never "
            "run. Every row becomes one field of well A01 unless you change "
            "its well below.")
        self._plate.setMaximumWidth(140)
        self._plate.textChanged.connect(self._on_plate_changed)
        shape.addWidget(self._plate)

        shape.addWidget(QLabel("Channels", self))
        self._channels = QSpinBox(self)
        self._channels.setRange(1, 16)
        self._channels.setValue(self._table.n_channels)
        self._channels.setToolTip(
            "How many intensity channels each field has. One column is added "
            "per channel, and the channel numbering of the merged array "
            "follows the column order.")
        self._channels.valueChanged.connect(self._on_channels_changed)
        shape.addWidget(self._channels)

        self._role_boxes: Dict[str, QCheckBox] = {}
        for role in OFFERED_ROLES:
            box = QCheckBox(role_caption(role), self)
            box.setChecked(role in self._table.roles)
            box.setToolTip(
                f"Measure {role} objects. One mask column is added, and "
                f"{role}_mask_dim is set from the column's position in the "
                "merged array.")
            box.toggled.connect(self._on_roles_changed)
            self._role_boxes[role] = box
            shape.addWidget(box)
        shape.addStretch(1)
        outer.addLayout(shape)

        regex_row = QHBoxLayout()
        regex_row.addWidget(QLabel("Regex", self))
        self._regex = QLineEdit(DEFAULT_REGEX, self)
        self._regex.setToolTip(
            "One pattern that sorts dropped files into rows and columns. "
            "(?P<field>...) names the row, (?P<channel>...) puts the file in "
            "a channel column and (?P<mask>...) in a mask column. Files that "
            "do not match are listed below rather than dropped silently.")
        self._regex.textChanged.connect(self._on_regex_changed)
        regex_row.addWidget(self._regex, 1)
        self._reapply = QPushButton("Re-apply to all files", self)
        self._reapply.setToolTip(
            "Sort every file this table already holds again, with the regex "
            "as it reads now.")
        self._reapply.clicked.connect(self.reapply_regex)
        regex_row.addWidget(self._reapply)
        outer.addLayout(regex_row)

        self._regex_status = QLabel("", self)
        self._regex_status.setObjectName("CardSubtitle")
        self._regex_status.setWordWrap(True)
        outer.addWidget(self._regex_status)

        self._grid = QTableWidget(0, 0, self)
        self._grid.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._grid.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._grid.setMinimumHeight(180)
        self._grid.cellDoubleClicked.connect(self._on_cell_activated)
        self._grid.itemChanged.connect(self._on_item_edited)
        self._grid.setToolTip(
            "Rows are fields, columns are channels and mask types. Drop "
            "files anywhere on this table, or double-click a cell to browse "
            "for the one file it wants.")
        outer.addWidget(self._grid, 1)

        buttons = QHBoxLayout()
        self._add_row = QPushButton("Add field", self)
        self._add_row.clicked.connect(self.add_field)
        self._remove = QPushButton("Remove selected", self)
        self._remove.clicked.connect(self.remove_selected)
        self._clear = QPushButton("Clear", self)
        self._clear.clicked.connect(self.clear)
        for button in (self._add_row, self._remove, self._clear):
            buttons.addWidget(button)
        buttons.addStretch(1)
        outer.addLayout(buttons)

        self._left_over_caption = QLabel("", self)
        self._left_over_caption.setObjectName("CardSubtitle")
        outer.addWidget(self._left_over_caption)
        self._left_over = QListWidget(self)
        self._left_over.setMaximumHeight(90)
        self._left_over.setToolTip(
            "Files that were dropped but matched no row and column. Nothing "
            "here is measured.")
        outer.addWidget(self._left_over)

        self.setAcceptDrops(True)
        self._rebuild()

    def table(self) -> FieldTable:
        """The model this widget edits. Live, not a copy."""
        return self._table

    def problems(self) -> List[str]:
        """Everything stopping this table being measured, as sentences."""
        return self._table.problems()

    def unassigned(self) -> List[Tuple[str, str]]:
        """``(path, reason)`` for every dropped file that landed nowhere."""
        return list(self._unassigned)

    def regex(self) -> str:
        """The pattern in the regex box."""
        return self._regex.text()

    def set_regex(self, pattern: str) -> None:
        """Put ``pattern`` in the regex box and sort the known files again."""
        self._regex.setText(str(pattern))
        self.reapply_regex()

    def set_file_picker(self, picker) -> None:
        """Replace the modal file chooser with ``picker(caption) -> path``.

        Headless Qt refuses a static modal -- it runs its event loop in C++
        and hangs the run -- so a test drives the browse path through this
        rather than through :class:`QFileDialog`.
        """
        self._picker = picker

    def add_paths(self, paths: Sequence[Any]) -> int:
        """Sort ``paths`` into the table with the current regex.

        Synchronous, and safe to be: the regex is matched against each path's
        BASENAME, so sorting files into cells is string work that touches no
        filesystem. Anything that might be a FOLDER must come through
        :meth:`add_dropped` instead, which expands it on a worker first.

        :param paths: file paths, as strings or anything ``str`` accepts.
        :returns: how many of them landed in a cell.
        """
        texts = [str(path) for path in paths]
        self._known_paths.extend(
            path for path in texts if path not in self._known_paths)
        return self._apply(texts)

    def add_dropped(self, paths: Sequence[Any]) -> None:
        """Add every file under ``paths``, once a worker has expanded them.

        SPLIT FROM :meth:`add_paths`, and the split is the fix for a frozen
        application. Everything here is a list of strings; the ``isdir`` and
        ``listdir`` that turn a dropped folder into files run on this
        widget's own worker (:func:`files_under`), because a drop is a path
        the user chose and on a microscope rig that is the share the images
        live on -- one stat under a sleeping ``autofs`` mount was measured at
        over twenty seconds, on the thread that paints.

        NOTHING IS ADDED BY THE TIME THIS RETURNS, which is the point. Read
        the table from :attr:`table_changed`, not from the line after this.

        :param paths: files and folders from a drop or a file dialog.
        :returns: ``None``. The count does not exist yet.
        """
        wanted = [str(path) for path in paths or ()]
        if not wanted:
            return
        self._regex_status.setText("Looking at what was dropped...")
        if not self._scanner.submit(lambda: _walk(wanted), self._files_found):
            self.add_paths(wanted)

    def _files_found(self, answer: Any) -> None:
        """Sort what the walk found, on the GUI thread.

        Generation-guarded by ``JobRunner``, so a walk the user abandoned by
        pressing Clear cannot refill the table twenty seconds later.
        """
        found, trouble = answer if answer else ((), "")
        if trouble:
            self._regex_status.setText(
                f"Some of what was dropped could not be read: {trouble}")
        self.add_paths(list(found or ()))

    def _scan_failed(self, message: str) -> None:
        """Say so when the walk itself did not finish.

        :func:`_walk` carries an ordinary failure back through
        :meth:`_files_found`, so this is the case that cannot: the worker
        did not return at all. Without it the "Looking..." caption would
        stay on screen for the rest of the session. ``RuntimeError``: a
        worker parked by shutdown outlives this widget's C++ half.
        """
        try:
            self._regex_status.setText(
                f"What was dropped could not be read: {message}")
        except RuntimeError:
            pass

    def is_scanning(self) -> bool:
        """True while a walk started by :meth:`add_dropped` is still running."""
        return self._scanner.is_busy()

    def reapply_regex(self) -> int:
        """Sort every file this table has ever been given, again.

        The cells are emptied first, so a corrected regex REPLACES the
        previous assignment instead of adding a second one beside it.

        :returns: how many files landed in a cell.
        """
        remembered = list(self._known_paths)
        self._table.rows = []
        self._table.channel_tokens = ()
        self._unassigned = []
        return self._apply(remembered)

    def add_field(self) -> FieldRow:
        """Add one empty row, numbered after the last one."""
        row = FieldRow(label=f"field {len(self._table.rows) + 1}",
                       well='A01', field=len(self._table.rows) + 1)
        self._table.rows.append(row)
        self._rebuild()
        return row

    def remove_selected(self) -> int:
        """Remove every selected row. Returns how many went."""
        doomed = sorted({index.row()
                         for index in self._grid.selectedIndexes()},
                        reverse=True)
        for index in doomed:
            if 0 <= index < len(self._table.rows):
                del self._table.rows[index]
        if doomed:
            self._rebuild()
        return len(doomed)

    def clear(self) -> None:
        """Forget every row and every file that was ever dropped.

        The channel-token memory goes with them: an empty table's columns
        mean nothing yet, and keeping the old ranking would let a token the
        user has cleared decide where the next drop's files land.
        """
        self._scanner.cancel()
        self._table.rows = []
        self._table.channel_tokens = ()
        self._unassigned = []
        self._known_paths = []
        self._rebuild()

    def assign_file(self, row: int, column: str, path: str) -> bool:
        """Put one file in one cell, as double-clicking and browsing does.

        :param row: the row's position in the table.
        :param column: ``'channel:<index>'`` counting from zero, or a mask
            role name.
        :param path: the file to put there.
        :returns: whether the cell existed and took it.
        """
        if not 0 <= row < len(self._table.rows):
            return False
        target = self._table.rows[row]
        text = str(column)
        if text.startswith('channel:'):
            index = int(text.split(':', 1)[1])
            if not 0 <= index < int(self._table.n_channels):
                return False
            target.channels[index] = str(path)
        else:
            role = mask_role_of(text)
            if role is None or role not in self._table.ordered_roles():
                return False
            target.masks[role] = str(path)
        if str(path) not in self._known_paths:
            self._known_paths.append(str(path))
        self._rebuild()
        return True

    def _apply(self, paths: Sequence[str]) -> int:
        """Run the regex over ``paths`` and fold the result into the table."""
        if not paths:
            self._rebuild()
            return 0
        try:
            found = assign_paths_by_regex(
                paths, self._regex.text(), table=self._table,
                plate=self._table.plate)
        except Exception as exc:
            self._regex_status.setText(f"The regex will not compile: {exc}")
            self._rebuild()
            return 0
        self._table = found.table
        self._unassigned = list(found.unassigned)
        for role in found.table.ordered_roles():
            box = self._role_boxes.get(role)
            if box is not None and not box.isChecked():
                box.blockSignals(True)
                box.setChecked(True)
                box.blockSignals(False)
        self._channels.blockSignals(True)
        self._channels.setValue(max(1, int(self._table.n_channels)))
        self._channels.blockSignals(False)
        self._regex_status.setText(
            f"{len(found.assigned)} file(s) assigned, "
            f"{len(found.unassigned)} left over.")
        self._rebuild()
        return len(found.assigned)

    def _column_keys(self) -> List[str]:
        """The cell key of every file column, left to right."""
        keys = [f'channel:{index}'
                for index in range(int(self._table.n_channels))]
        keys.extend(self._table.ordered_roles())
        return keys

    def _rebuild(self) -> None:
        """Redraw the grid from the model and announce the change."""
        roles = self._table.ordered_roles()
        headings = ["Field", "Well", "Field #"]
        headings.extend(f"Channel {index + 1}"
                        for index in range(int(self._table.n_channels)))
        headings.extend(role_caption(role) for role in roles)

        self._grid.blockSignals(True)
        self._grid.clear()
        self._grid.setColumnCount(len(headings))
        self._grid.setHorizontalHeaderLabels(headings)
        self._grid.setRowCount(len(self._table.rows))
        for index, row in enumerate(self._table.rows):
            for column, text in enumerate(
                    (row.label, row.well, str(row.field))):
                item = QTableWidgetItem(str(text))
                if column == 0:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setToolTip(
                        f"Written and measured as "
                        f"{row.stem(self._table.plate)}")
                self._grid.setItem(index, column, item)
            for offset, key in enumerate(self._column_keys()):
                if key.startswith('channel:'):
                    path = row.channels.get(int(key.split(':', 1)[1]))
                else:
                    path = row.masks.get(key)
                item = QTableWidgetItem(
                    _basename(path) if path else "double-click to browse")
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setToolTip(path or "No file yet")
                if not path:
                    item.setForeground(Qt.gray)
                self._grid.setItem(index, _IDENTITY_COLUMNS + offset, item)
        header = self._grid.horizontalHeader()
        if header is not None and headings:
            header.setSectionResizeMode(0, QHeaderView.Stretch)
            for column in range(1, len(headings)):
                header.setSectionResizeMode(
                    column, QHeaderView.ResizeToContents)
        self._grid.blockSignals(False)

        self._left_over.clear()
        for path, reason in self._unassigned:
            self._left_over.addItem(f"{_basename(path)} -- {reason}")
        self._left_over_caption.setText(
            "Nothing was left over." if not self._unassigned
            else f"{len(self._unassigned)} file(s) went nowhere:")
        self.table_changed.emit()

    def _on_plate_changed(self, text: str) -> None:
        """Rename the plate every row's stem starts with."""
        self._table.plate = str(text) or 'drawn'
        self._rebuild()

    def _on_channels_changed(self, value: int) -> None:
        """Add or drop channel columns, keeping the files that still fit.

        A column that goes takes its token with it, or the memory would
        still claim a column the table no longer has and the next drop
        would silently bring it back.
        """
        self._table.n_channels = int(value)
        self._table.channel_tokens = tuple(
            self._table.channel_tokens[:int(value)])
        for row in self._table.rows:
            for index in list(row.channels):
                if index >= int(value):
                    del row.channels[index]
        self._rebuild()

    def _on_roles_changed(self, _checked: bool = False) -> None:
        """Add or drop mask columns to match the checkboxes."""
        wanted = tuple(role for role, box in self._role_boxes.items()
                       if box.isChecked())
        kept = tuple(role for role in self._table.roles
                     if role in ORGANELLE_ROLES and role not in OFFERED_ROLES)
        self._table.roles = wanted + kept
        for row in self._table.rows:
            for role in list(row.masks):
                if role not in self._table.ordered_roles():
                    del row.masks[role]
        self._rebuild()

    def _on_regex_changed(self, _text: str) -> None:
        """Say whether the pattern compiles, without touching the table."""
        import re

        try:
            re.compile(self._regex.text())
        except re.error as exc:
            self._regex_status.setText(f"The regex will not compile: {exc}")
        else:
            self._regex_status.setText(
                "Press Re-apply to sort the files with this pattern.")

    def _on_item_edited(self, item: QTableWidgetItem) -> None:
        """Take an edited well or field number back into the model."""
        row = item.row()
        if not 0 <= row < len(self._table.rows):
            return
        if item.column() == 1:
            self._table.rows[row].well = item.text().strip() or 'A01'
        elif item.column() == 2:
            try:
                self._table.rows[row].field = int(item.text().strip())
            except ValueError:
                pass
        else:
            return
        self._rebuild()

    def _on_cell_activated(self, row: int, column: int) -> None:
        """Browse for the one file the double-clicked cell wants."""
        keys = self._column_keys()
        offset = column - _IDENTITY_COLUMNS
        if not 0 <= offset < len(keys):
            return
        key = keys[offset]
        caption = (f"Choose the file for {self._table.rows[row].label}, "
                   f"{self._grid.horizontalHeaderItem(column).text()}")
        if self._picker is not None:
            chosen = self._picker(caption)
        else:
            chosen, _filter = QFileDialog.getOpenFileName(self, caption)
        if chosen:
            self.assign_file(row, key, str(chosen))

    def dragEnterEvent(self, event):  # noqa: N802 - Qt contract
        """Accept a drag that carries local files.

        Decided from the mime data alone. Asking the filesystem whether the
        drag is worth accepting would run a stat on the GUI thread for every
        drag-move event the pointer produces.

        :param event: the Qt drag event.
        """
        if _dropped_paths(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):  # noqa: N802 - Qt contract
        """Keep accepting while local files stay over the table.

        :param event: the Qt drag event.
        """
        self.dragEnterEvent(event)

    def dropEvent(self, event):  # noqa: N802 - Qt contract
        """Sort the dropped files into rows and columns.

        Returns before anything is assigned when a folder was dropped: the
        folder is expanded on a worker. Read the table from
        :attr:`table_changed`, not from the line after this one.

        :param event: the Qt drop event.
        """
        paths = _dropped_paths(event)
        if not paths:
            event.ignore()
            return
        self.add_dropped(paths)
        event.acceptProposedAction()


def _basename(path: Optional[str]) -> str:
    """The last component of ``path``, for a cell that has no room for more."""
    import os

    return os.path.basename(str(path)) if path else ""


def _dropped_paths(event) -> List[str]:
    """Every local path a drag carries, as strings and nothing more.

    TOUCHES NO FILESYSTEM, and that is the whole point of it. It used to
    expand a dropped folder here, which meant an ``isdir`` and a ``listdir``
    inside ``dropEvent`` -- and, because ``dragEnterEvent`` called this to
    decide whether to accept, inside every ``dragMoveEvent`` as well, so the
    stats ran repeatedly while the pointer simply moved over the table. On a
    microscope rig the dropped folder is on the share the images live on:
    measured on the maintainer's machine, ONE stat under a sleeping
    ``autofs`` mount had not returned after twenty seconds. Expanding the
    folder is :func:`files_under`'s job, on a worker.
    """
    mime = event.mimeData() if hasattr(event, 'mimeData') else None
    if mime is None or not mime.hasUrls():
        return []
    return [local for local in (url.toLocalFile() for url in mime.urls())
            if local]


def files_under(paths: Sequence[str]) -> List[str]:
    """Every file among ``paths``, with any folder expanded one level.

    A user who drew masks for six fields drops the folder, not the files, so
    a drop that refused a directory would refuse the ordinary case.

    NOT FOR THE GUI THREAD. Every path here is one the user chose, which on a
    microscope rig means a network share, and the ``isdir`` is what wakes the
    automount. :meth:`MeasureInputTable.add_dropped` is the only caller and
    it runs this on a worker. Looked up through the module global so a test
    can replace it and see which thread it ran on.

    :param paths: what was dropped or chosen.
    :returns: the files, folders expanded, in a stable order.
    """
    import os

    found: List[str] = []
    for raw in paths or ():
        path = str(raw)
        if os.path.isdir(path):
            found.extend(
                os.path.join(path, name)
                for name in sorted(os.listdir(path))
                if os.path.isfile(os.path.join(path, name)))
        else:
            found.append(path)
    return found


def _walk(paths: Sequence[str]) -> Tuple[List[str], str]:
    """Run :func:`files_under`, carrying any failure back as a string.

    On the worker thread, and the failure is RETURNED rather than raised on
    purpose: ``JobRunner`` hands a result to its ``on_done`` only for a job
    that succeeded, so a walk that raised would leave the "Looking..."
    caption on screen for the rest of the session.
    """
    try:
        return files_under(paths), ""
    except Exception as exc:                                     # noqa: BLE001
        return [], str(exc) or exc.__class__.__name__
