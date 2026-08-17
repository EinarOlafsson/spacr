"""A settings widget for "one or more files": browse, drop, repeat.

Several spaCR settings are lists of input paths -- ``score_data``,
``count_data``, ``metadata_files``, ``grna_csv`` and friends. They used to
render as a chip strip that the user typed absolute paths into, one character
at a time, which is unusable for the four plate CSVs a screen actually has and
silently accepted a path that did not exist.

:class:`FilePathListWidget` replaces that with the two gestures people expect:

* **Add files...** opens a file dialog with multi-selection enabled. Pressing
  it again *appends*, so several sources can be gathered from different
  folders in several trips rather than one impossible single selection.
* **Dropping** files or folders anywhere on the widget adds them. A dropped
  folder contributes its matching files, one directory level deep, sorted.

Everything else follows from those: duplicates are refused, order is editable
because a regression that reads ``plate1..plate4`` cares about it, missing
paths are marked instead of being discovered at run time, and the value is a
plain ``list[str]`` so the CLI, the settings CSV and the Qt panel all agree.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable, List, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def _pair_tokens(path: str) -> set[str]:
    stem = os.path.splitext(os.path.basename(os.fspath(path)))[0].casefold()
    stem = re.sub(r"plate[\s_-]*0*(\d+)", r"plate\1", stem)
    generic = {"score", "scores", "count", "counts", "result", "results",
               "unique", "combinations", "csv", "maxvit", "xgb"}
    return {token for token in re.findall(r"[a-z]+\d+|\d+|[a-z]+", stem)
            if token not in generic and len(token) > 1}


def suggest_file_pairs(scores: Sequence[str], counts: Sequence[str]) -> list[dict]:
    """Propose visible score/count pairs by filename tokens.

    A proposal is never authoritative: the editable table is the contract the
    user confirms. Unique best matches are used; ties remain unpaired.
    """
    unused = set(range(len(counts)))
    rows = []
    for score in scores:
        left = _pair_tokens(score)
        ranked = sorted(
            ((len(left & _pair_tokens(counts[index])), index)
             for index in unused), reverse=True)
        match = None
        if ranked and ranked[0][0] > 0 and (
                len(ranked) == 1 or ranked[0][0] > ranked[1][0]):
            match = ranked[0][1]
            unused.remove(match)
        common = left & (_pair_tokens(counts[match]) if match is not None else set())
        plate = sorted((token for token in common if token.startswith("plate")),
                       key=len, reverse=True)
        rows.append({"plate": plate[0] if plate else "",
                     "score": os.fspath(score),
                     "count": os.fspath(counts[match]) if match is not None else None})
    for index in sorted(unused):
        rows.append({"plate": "", "score": None,
                     "count": os.fspath(counts[index])})
    return rows


def side_for_header(path) -> str:
    """``'count'`` when the file's header names a gRNA and a count, else
    ``'score'``.

    Read from the header rather than the filename: a count export carries a
    gRNA name and a count, a score export carries neither, and that is true
    whatever the file is called.

    Module-level because two screens ask the same question of the same files.
    Regression asks it through :class:`PairedFileTableWidget`; Parameter Sweep
    holds its two sides in separate list widgets and asks it through
    ``spacr.qt.dnd_handlers.SweepInputsDropHandler``. A second copy of this
    rule would drift, and the direction it would drift in is silent: a count
    table filed as a score is not an error, it is a wrong regression.
    """
    try:
        import csv as _csv
        with open(path, newline="", encoding="utf-8",
                  errors="replace") as handle:
            header = {str(name).strip().lower()
                      for name in next(_csv.reader(handle), [])}
    except OSError:
        return "score"
    return ("count" if {"grna", "grna_name"} & header and "count" in header
            else "score")


class PairedFileTableWidget(QWidget):
    """Editable one-row-per-plate score/count input contract."""

    value_changed = Signal()

    def __init__(self, value=None, parent=None):
        super().__init__(parent)
        self._scores: list[str] = []
        self._counts: list[str] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(
            ["Plate / proposal", "Score CSV", "Count CSV", "Plate rule"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.itemChanged.connect(lambda *_: self.value_changed.emit())
        layout.addWidget(self.table)
        buttons = QHBoxLayout()
        add_scores = QPushButton("Add score CSVs…", self)
        add_counts = QPushButton("Add count CSVs…", self)
        add_row = QPushButton("Add empty pair", self)
        up = QPushButton("↑", self)
        down = QPushButton("↓", self)
        remove = QPushButton("Remove", self)
        add_scores.clicked.connect(lambda: self._pick("score"))
        add_counts.clicked.connect(lambda: self._pick("count"))
        add_row.clicked.connect(lambda: self._append_row({}))
        up.clicked.connect(lambda: self._move(-1))
        down.clicked.connect(lambda: self._move(1))
        remove.clicked.connect(self._remove)
        for button in (add_scores, add_counts, add_row, up, down, remove):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        # The table is the drop target the user aims at, so the widget takes
        # drops and the table does not swallow them first.
        self.setAcceptDrops(True)
        self.table.setAcceptDrops(False)
        self.set_value(value)

    #: Column index of each side in the table, so a drop lands where the user
    #: aimed it rather than in whichever input the router reached first.
    SIDE_COLUMNS = {"score": 1, "count": 2}

    def add_paths_for_side(self, paths, side: str = "score") -> int:
        """Add files to the score or count column and re-propose the pairing.

        The drop targets the user asked for -- "a dependent variable square I
        can drop into and an independent variable square I can drop into" --
        resolve to this. It is also what the drop router calls once it has
        decided which side a file belongs to from its header, so dropping a
        count table anywhere on the widget still fills the count column.

        Returns how many files were added.
        """
        if side not in self.SIDE_COLUMNS:
            raise ValueError(
                f"side must be 'score' or 'count'; got {side!r}.")
        incoming = [os.fspath(p) for p in (paths or [])]
        if not incoming:
            return 0
        current = self.get_value()
        self._scores = list(dict.fromkeys(
            row["score"] for row in current if row.get("score")))
        self._counts = list(dict.fromkeys(
            row["count"] for row in current if row.get("count")))
        target = self._scores if side == "score" else self._counts
        added = 0
        for path in incoming:
            if path not in target:
                target.append(path)
                added += 1
        if added:
            # Re-propose so a count dropped after its score lands on the same
            # row: the pairing is by filename token, not by drop order.
            self.set_value(suggest_file_pairs(self._scores, self._counts))
            self.value_changed.emit()
        return added

    def _side_for_header(self, path) -> str:
        """Which column ``path`` belongs in. See :func:`side_for_header`."""
        return side_for_header(path)

    # ---------------------------------------------------------- drag / drop

    @staticmethod
    def _dropped(event):
        mime = event.mimeData()
        if not mime.hasUrls():
            return []
        return [url.toLocalFile() for url in mime.urls() if url.isLocalFile()]

    def dragEnterEvent(self, event):  # noqa: N802 - Qt name
        if self._dropped(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):  # noqa: N802 - Qt name
        if self._dropped(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):  # noqa: N802 - Qt name
        """Route each dropped file to the column it belongs in.

        A drop over the score or count column goes there regardless of what
        the file looks like -- the user aimed it. A drop anywhere else is
        sorted by header, so dropping the whole set at once still fills both
        columns correctly.
        """
        paths = self._dropped(event)
        if not paths:
            event.ignore()
            return
        aimed = None
        position = event.position().toPoint() if hasattr(event, "position") \
            else event.pos()
        local = self.table.mapFrom(self, position)
        column = self.table.columnAt(local.x())
        for side, index in self.SIDE_COLUMNS.items():
            if column == index and self.table.rect().contains(local):
                aimed = side
                break
        for path in paths:
            self.add_paths_for_side([path], aimed or self._side_for_header(path))
        event.acceptProposedAction()

    def _pick(self, side: str) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, f"Add {side} CSVs", "", "Tables (*.csv *.tsv *.txt)")
        if not paths:
            return
        current = self.get_value()
        self._scores = list(dict.fromkeys(
            row["score"] for row in current if row.get("score")))
        self._counts = list(dict.fromkeys(
            row["count"] for row in current if row.get("count")))
        if side == "score":
            self._scores.extend(path for path in paths if path not in self._scores)
        else:
            self._counts.extend(path for path in paths if path not in self._counts)
        self.set_value(suggest_file_pairs(self._scores, self._counts))
        self.value_changed.emit()

    def _append_row(self, row: dict) -> None:
        index = self.table.rowCount()
        self.table.insertRow(index)
        values = (row.get("plate") or "", row.get("score") or "",
                  row.get("count") or "", row.get("rule") or "resolved at run")
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            if column == 3:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(index, column, item)

    def set_value(self, value: Any) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for row in value or []:
            if isinstance(row, dict):
                self._append_row(row)
        self.table.blockSignals(False)

    def get_value(self) -> list[dict]:
        rows = []
        for index in range(self.table.rowCount()):
            value = lambda column: (self.table.item(index, column).text().strip()
                                    if self.table.item(index, column) else "")
            score, count = value(1), value(2)
            if score or count:
                rows.append({"plate": value(0) or None,
                             "score": score or None, "count": count or None})
        return rows

    def _move(self, offset: int) -> None:
        row = self.table.currentRow()
        target = row + offset
        if row < 0 or not 0 <= target < self.table.rowCount():
            return
        values = self.get_value()
        values[row], values[target] = values[target], values[row]
        self.set_value(values)
        self.table.selectRow(target)
        self.value_changed.emit()

    def _remove(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()},
                      reverse=True)
        for row in rows:
            self.table.removeRow(row)
        if rows:
            self.value_changed.emit()

#: Extension groups offered in the dialog, by the kind of input a setting wants.
FILE_KIND_FILTERS: dict[str, str] = {
    "table": "Tables (*.csv *.tsv *.txt *.xlsx *.parquet);;CSV (*.csv);;"
             "Excel (*.xlsx);;All files (*)",
    "csv": "CSV (*.csv *.tsv *.txt);;All files (*)",
    "image": "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.czi *.lif *.nd2);;"
             "All files (*)",
    "model": "Models (*.pth *.pt *.ckpt *.h5 *.joblib *.pkl);;All files (*)",
    "sequencing": "Reads (*.fastq *.fq *.fastq.gz *.fq.gz);;All files (*)",
    "any": "All files (*)",
}

#: Extensions a dropped *folder* contributes, per kind. A folder dropped on a
#: CSV setting must not add its PNGs.
_KIND_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "table": (".csv", ".tsv", ".txt", ".xlsx", ".parquet"),
    "csv": (".csv", ".tsv", ".txt"),
    "image": (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".czi",
              ".lif", ".nd2"),
    "model": (".pth", ".pt", ".ckpt", ".h5", ".joblib", ".pkl"),
    "sequencing": (".fastq", ".fq", ".gz"),
    "any": (),
}


class FilePathListWidget(QWidget):
    """An ordered, de-duplicated list of input paths with picker and drop.

    ``single=True`` is the same control for a setting that names exactly ONE
    file. It keeps the file dialog and the drop target -- which is the whole
    reason these settings stopped being text boxes -- but the value it holds
    and returns is a plain ``str``, choosing again REPLACES rather than
    appends, and the reorder buttons are gone because one path has no order.

    That distinction is not cosmetic. ``grna_csv``, ``row_csv`` and
    ``column_csv`` are declared ``str`` and go straight to ``pd.read_csv``,
    so rendering them as a list turned a working default into
    ``['/path/to/barcodes_row.csv']``: the settings file was rewritten the
    moment the screen was opened and saved, and the run died on "Invalid file
    path or buffer object type: <class 'list'>" only once it had started
    reading FASTQs.
    """

    value_changed = Signal()

    def __init__(
        self,
        value: Any = None,
        *,
        kind: str = "table",
        title: str = "Choose input files",
        allow_folders: bool = True,
        single: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._kind = kind if kind in FILE_KIND_FILTERS else "any"
        self._title = title
        self._single = bool(single)
        # A folder is not a file, and expanding one into "all the CSVs in
        # here" cannot mean anything for a setting that names one of them.
        self._allow_folders = bool(allow_folders) and not self._single
        self._last_directory = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        self._list = QListWidget(self)
        self._list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._list.setAlternatingRowColors(True)
        self._list.setMinimumHeight(48 if self._single else 96)
        self._list.setUniformItemSizes(True)
        # The list itself must not swallow the drop before the widget sees it.
        self._list.setAcceptDrops(False)
        self._list.setDragDropMode(QAbstractItemView.NoDragDrop)
        outer.addWidget(self._list)

        self._hint = QLabel(self._empty_hint(), self)
        self._hint.setWordWrap(True)
        self._hint.setProperty("role", "hint")
        outer.addWidget(self._hint)

        row = QHBoxLayout()
        row.setSpacing(4)
        self._add_files_button = QPushButton(
            "Choose file…" if self._single else "Add files…", self)
        self._add_files_button.setToolTip(
            "Select the file this setting names. Choosing again replaces it."
            if self._single else
            "Select one or more files. Press again to add more from another "
            "folder — each press appends to the list.")
        self._add_files_button.clicked.connect(self.pick_files)
        row.addWidget(self._add_files_button)

        if self._allow_folders:
            self._add_folder_button = QPushButton("Add folder…", self)
            self._add_folder_button.setToolTip(
                "Add every matching file directly inside a folder.")
            self._add_folder_button.clicked.connect(self.pick_folder)
            row.addWidget(self._add_folder_button)
        else:
            self._add_folder_button = None

        # One path has no order, so the two buttons that reorder the list are
        # not built at all rather than built and left doing nothing.
        if self._single:
            self._up_button = self._down_button = None
        else:
            self._up_button = QPushButton("↑", self)
            self._up_button.setToolTip(
                "Move the selected file earlier in the list")
            self._up_button.setMaximumWidth(30)
            self._up_button.clicked.connect(lambda: self._move_selected(-1))
            row.addWidget(self._up_button)

            self._down_button = QPushButton("↓", self)
            self._down_button.setToolTip(
                "Move the selected file later in the list")
            self._down_button.setMaximumWidth(30)
            self._down_button.clicked.connect(lambda: self._move_selected(1))
            row.addWidget(self._down_button)

        self._remove_button = QPushButton("Remove", self)
        self._remove_button.clicked.connect(self.remove_selected)
        row.addWidget(self._remove_button)

        self._clear_button = QPushButton("Clear", self)
        self._clear_button.clicked.connect(self.clear)
        row.addWidget(self._clear_button)
        row.addStretch(1)
        outer.addLayout(row)

        self.setAcceptDrops(True)
        self.set_value(value)

    # ------------------------------------------------------------------ value

    def set_value(self, value: Any) -> None:
        """Replace the contents. Accepts None, a str, or any iterable.

        A single-file widget keeps only the last of whatever it is given.
        That is what loads a settings file written while these keys were
        wrongly rendered as lists: ``['/x/barcodes_row.csv']`` comes back as
        ``/x/barcodes_row.csv`` rather than carrying the wrong shape forward.
        """
        self._list.clear()
        paths = self._coerce(value)
        for path in (paths[-1:] if self._single else paths):
            self._append(path)
        self._refresh_hint()

    def paths(self) -> List[str]:
        """Every path currently listed, in order -- always a list."""
        return [self._list.item(row).data(Qt.UserRole)
                for row in range(self._list.count())]

    def get_value(self) -> Any:
        """The setting's value: a ``list[str]``, or a ``str`` when single.

        The shape is the SETTING's, not the widget's. A key declared ``str``
        that came back as a one-element list rewrote the user's settings file
        on open and reached ``pd.read_csv`` as a list.
        """
        listed = self.paths()
        if not self._single:
            return listed
        return listed[0] if listed else ""

    # A settings CSV written before this widget existed can hold the literal
    # placeholder 'list of paths'; it is not a path and must not become one.
    _PLACEHOLDERS = {"list of paths", "none", "", "[]"}

    @classmethod
    def _coerce(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (str, bytes, os.PathLike)):
            value = [value]
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            text = os.fspath(item) if isinstance(item, os.PathLike) else str(item)
            text = text.strip().strip('"').strip("'")
            if text.lower() in cls._PLACEHOLDERS:
                continue
            out.append(text)
        return out

    # ------------------------------------------------------------------- edit

    def add_paths(self, paths: Iterable[Any]) -> int:
        """Append ``paths``, expanding folders. Returns how many were added.

        When the setting names ONE file this REPLACES what is there. A second
        choice is a correction, and a control that appended left the run
        reading a file the user believed they had swapped out.
        """
        incoming = self._coerce(paths)
        if self._single:
            if not incoming:
                return 0
            chosen = os.path.abspath(os.path.expanduser(incoming[-1]))
            if chosen == (self.paths() or [None])[0]:
                return 0
            self._list.clear()
            self._append(chosen)
            self._refresh_hint()
            self.value_changed.emit()
            return 1
        added = 0
        for raw in incoming:
            expanded = os.path.abspath(os.path.expanduser(raw))
            if os.path.isdir(expanded):
                for member in self._folder_members(expanded):
                    added += int(self._append(member))
            else:
                added += int(self._append(expanded))
        if added:
            self._refresh_hint()
            self.value_changed.emit()
        return added

    def _folder_members(self, folder: str) -> List[str]:
        """Matching files one level inside ``folder``, sorted for stable order."""
        extensions = _KIND_EXTENSIONS.get(self._kind, ())
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            return []
        members = []
        for name in names:
            full = os.path.join(folder, name)
            if not os.path.isfile(full):
                continue
            if extensions and not name.lower().endswith(extensions):
                continue
            members.append(full)
        return members

    def _append(self, path: str) -> bool:
        """Add one path unless it is already listed. Returns True if added."""
        resolved = os.path.abspath(os.path.expanduser(str(path)))
        if resolved in set(self.paths()):
            return False
        item = QListWidgetItem(self._display_text(resolved))
        item.setData(Qt.UserRole, resolved)
        if os.path.exists(resolved):
            item.setToolTip(resolved)
        else:
            # Marked, not dropped: a settings file may legitimately be edited
            # on one machine and run on another, and silently discarding the
            # path would leave the user staring at an empty list.
            item.setToolTip(f"{resolved}\n\nThis path does not exist right now.")
            item.setForeground(Qt.red)
        self._list.addItem(item)
        return True

    @staticmethod
    def _display_text(path: str) -> str:
        """Basename plus enough parent to tell four plate CSVs apart."""
        parent = os.path.basename(os.path.dirname(path))
        name = os.path.basename(path)
        return f"{parent}/{name}" if parent else name

    def remove_selected(self) -> None:
        rows = sorted((self._list.row(item) for item in self._list.selectedItems()),
                      reverse=True)
        for row in rows:
            self._list.takeItem(row)
        if rows:
            self._refresh_hint()
            self.value_changed.emit()

    def clear(self) -> None:
        if self._list.count():
            self._list.clear()
            self._refresh_hint()
            self.value_changed.emit()

    def _move_selected(self, offset: int) -> None:
        """Move the single selected row by ``offset``, keeping it selected."""
        items = self._list.selectedItems()
        if len(items) != 1:
            return
        row = self._list.row(items[0])
        target = row + offset
        if not 0 <= target < self._list.count():
            return
        item = self._list.takeItem(row)
        self._list.insertItem(target, item)
        self._list.setCurrentRow(target)
        self.value_changed.emit()

    def _empty_hint(self) -> str:
        if self._single:
            return "Drop one file here, or use Choose file…"
        return "Drop files or folders here, or use Add files…"

    def _refresh_hint(self) -> None:
        count = self._list.count()
        missing = sum(
            1 for row in range(count)
            if not os.path.exists(self._list.item(row).data(Qt.UserRole))
        )
        if not count:
            self._hint.setText(self._empty_hint())
        elif missing:
            self._hint.setText(
                f"{count} file{'s' if count != 1 else ''} selected — "
                f"{missing} not found (shown in red)")
        else:
            self._hint.setText(
                f"{count} file{'s' if count != 1 else ''} selected")

    # -------------------------------------------------------------- drag/drop

    @staticmethod
    def _urls(event) -> List[str]:
        mime = event.mimeData()
        if not mime.hasUrls():
            return []
        return [url.toLocalFile() for url in mime.urls() if url.isLocalFile()]

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if self._urls(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # noqa: N802
        if self._urls(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        paths = self._urls(event)
        if not paths:
            event.ignore()
            return
        self.add_paths(paths)
        event.acceptProposedAction()

    # ----------------------------------------------------------------- picker

    def pick_files(self) -> int:
        """Open the file dialog and take what is chosen.

        Multi-select, except when the setting names one file -- a dialog that
        lets you pick four when three of them will be discarded is a control
        lying about what it does.
        """
        if self._single:
            path, _selected = QFileDialog.getOpenFileName(
                self, self._title, self._start_directory(),
                FILE_KIND_FILTERS[self._kind])
            paths = [path] if path else []
        else:
            paths, _selected = QFileDialog.getOpenFileNames(
                self, self._title, self._start_directory(),
                FILE_KIND_FILTERS[self._kind])
        if not paths:
            return 0
        self._last_directory = os.path.dirname(paths[0])
        return self.add_paths(paths)

    def pick_folder(self) -> int:
        folder = QFileDialog.getExistingDirectory(
            self, f"{self._title} — choose a folder", self._start_directory())
        if not folder:
            return 0
        self._last_directory = folder
        return self.add_paths([folder])

    def _start_directory(self) -> str:
        """Reopen where the user last was, or beside the last file added."""
        if self._last_directory and os.path.isdir(self._last_directory):
            return self._last_directory
        values = self.paths()
        if values:
            parent = os.path.dirname(values[-1])
            if os.path.isdir(parent):
                return parent
        return ""


__all__ = ["FilePathListWidget", "FILE_KIND_FILTERS", "PairedFileTableWidget",
           "side_for_header", "suggest_file_pairs"]
