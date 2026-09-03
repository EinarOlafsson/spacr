"""Preview how imported image names map to a spaCR folder structure.

Drop images into the table, review or edit the inferred filename pattern,
and assign a role to each captured group. The preview updates after every
edit and does not write or move files. Pattern inference is provided by
:mod:`spacr.regex_infer`; the displayed plan comes from
:mod:`spacr.import_plan`.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog,
                               QDialogButtonBox, QFileDialog, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QPlainTextEdit,
                               QPushButton, QSplitter, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ...import_plan import (CHANNEL_MEANINGS, ROLES, for_get_regex,
                            group_names, plan)
from .sortable_table import install_sorting, table_item

LOG = logging.getLogger("spacr.qt.import_workbench")

#: Image suffixes a drop is filtered to. A folder dropped whole is walked.
IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".nd2", ".czi",
                  ".lif", ".bmp")


def images_under(paths: Sequence[str], *, limit: int = 5000) -> List[str]:
    """Every image among ``paths``, walking any folder given.

    :param paths: files and directories from the drop or file picker. Supported
        files are kept directly and directories are walked recursively.
    :param limit: stop after this many. A plate is tens of thousands of
        files and the table is a PREVIEW -- the regex is inferred from an
        aligned set, and the set does not have to be all of it.
    """
    found: List[str] = []
    for raw in paths or ():
        path = str(raw)
        if os.path.isdir(path):
            for root, _dirs, names in os.walk(path):
                for name in sorted(names):
                    if name.lower().endswith(IMAGE_SUFFIXES):
                        found.append(os.path.join(root, name))
                        if len(found) >= limit:
                            return found
        elif path.lower().endswith(IMAGE_SUFFIXES):
            found.append(path)
            if len(found) >= limit:
                return found
    return found


class ImportWorkbench(QWidget):
    """The table, the regex, the roles and the preview, in one panel.

    :param filenames: the names to work the pattern out from. Copied to a
        list of strings, so a generator or a set of paths is accepted and the
        caller's sequence is not consumed.
    :param regex: the pattern to start from. Empty starts with none, which is
        the ordinary case -- working the pattern out is what this is for.
    :param parent: parent widget.
    """

    def __init__(self, filenames: Sequence[str] = (),
                 regex: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._files: List[str] = [str(f) for f in filenames or ()]
        self._roles: Dict[str, str] = {}
        self._plan = None
        self.setAcceptDrops(True)

        outer = QVBoxLayout(self)

        # ------------------------------------------------------------ A
        top = QHBoxLayout()
        self.dropped = QLabel("")
        self.dropped.setObjectName("Muted")
        self.dropped.setWordWrap(True)
        top.addWidget(self.dropped, 1)
        self.add_button = QPushButton("Add files…", self)
        self.add_button.clicked.connect(self.ask_for_files)
        top.addWidget(self.add_button)
        self.clear_button = QPushButton("Clear", self)
        self.clear_button.clicked.connect(lambda: self.set_files([]))
        top.addWidget(self.clear_button)
        outer.addLayout(top)

        # ------------------------------------------------------------ B/D
        row = QHBoxLayout()
        row.addWidget(QLabel("regex"))
        self.regex = QLineEdit(self)
        self.regex.setPlaceholderText(
            r"drop files and press Propose, or type (?P<wellID>...)")
        self.regex.textChanged.connect(lambda *_: self.refresh())
        row.addWidget(self.regex, 1)
        self.propose_button = QPushButton("Propose from the names", self)
        self.propose_button.setToolTip(
            "Work the pattern out from the filenames themselves: tokenise "
            "them, align the set, and the parts that VARY are the groups.")
        self.propose_button.clicked.connect(self.propose_from_the_names)
        row.addWidget(self.propose_button)
        outer.addLayout(row)

        self.evidence = QLabel("")
        self.evidence.setObjectName("Muted")
        self.evidence.setWordWrap(True)
        outer.addWidget(self.evidence)

        # ------------------------------------------------------------ C
        self.roles_row = QHBoxLayout()
        self.roles_holder = QWidget(self)
        self.roles_holder.setLayout(self.roles_row)
        outer.addWidget(self.roles_holder)
        self.role_trouble = QLabel("")
        self.role_trouble.setWordWrap(True)
        outer.addWidget(self.role_trouble)

        # ------------------------------------------- the table and the tree
        split = QSplitter(Qt.Horizontal, self)
        self.table = QTableWidget(0, 2, self)
        install_sorting(self.table)
        self.table.setHorizontalHeaderLabels(["file", "would become"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive)
        split.addWidget(self.table)
        self.tree = QPlainTextEdit(self)
        self.tree.setReadOnly(True)
        split.addWidget(self.tree)
        split.setSizes([560, 340])
        outer.addWidget(split, 1)

        if regex:
            self.regex.setText(regex)
        self.set_files(self._files)

    # ------------------------------------------------------------ A: drops

    def dragEnterEvent(self, event):                 # noqa: N802 - Qt
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):                  # noqa: N802 - Qt
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):                      # noqa: N802 - Qt
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        self.add_files(paths)
        event.acceptProposedAction()

    def ask_for_files(self) -> int:
        chosen, _filter = QFileDialog.getOpenFileNames(
            self, "Images to import", "",
            "Images (" + " ".join(f"*{s}" for s in IMAGE_SUFFIXES) + ")")
        return self.add_files(chosen)

    def add_files(self, paths: Sequence[str]) -> int:
        """Add every image under ``paths``. Returns how many are held now."""
        found = images_under(paths)
        seen = set(self._files)
        self._files.extend(p for p in found if p not in seen)
        self.set_files(self._files)
        return len(self._files)

    def set_files(self, paths: Sequence[str]) -> None:
        self._files = [str(p) for p in paths or ()]
        # A REGEX PROPOSED FOR THE OLD SET IS NOT PROPOSED FOR THIS ONE, so
        # the first drop offers one and a later drop does not overwrite what
        # the user has since edited.
        if self._files and not self.regex.text().strip():
            self.propose_from_the_names()
        self.refresh()

    def files(self) -> List[str]:
        return list(self._files)

    # ------------------------------------------------------------ B

    def propose_from_the_names(self) -> str:
        """Infer a regex from the dropped names. Returns what it set."""
        if not self._files:
            self.evidence.setText("Nothing to work from yet.")
            return ""
        names = [os.path.basename(p) for p in self._files]
        try:
            from ..regex_detect import auto_detect_regex

            pattern, label, hits = auto_detect_regex(names)
        except Exception as exc:                             # noqa: BLE001
            LOG.debug("could not propose a regex", exc_info=True)
            self.evidence.setText(f"Could not work one out: {exc}")
            return ""
        if not pattern:
            self.evidence.setText(
                "No pattern fits these names. Type one, and the table below "
                "will show what it does as you go.")
            return ""
        self.regex.setText(pattern)
        self.evidence.setText(
            f"{label}: matches {hits} of {len(names)} name(s).")
        return pattern

    # ------------------------------------------------------------ C: roles

    def _rebuild_roles(self, groups: Sequence[str]) -> None:
        """One dropdown per group, keeping any choice already made."""
        while self.roles_row.count():
            item = self.roles_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._boxes: Dict[str, QComboBox] = {}
        if not groups:
            self.roles_holder.setVisible(False)
            return
        self.roles_holder.setVisible(True)
        self.roles_row.addWidget(QLabel("what each group means:"))
        known = dict(ROLES)
        for group in groups:
            self.roles_row.addWidget(QLabel(str(group)))
            box = QComboBox(self)
            for value, why in ROLES:
                box.addItem(value or "(ignore)", value)
                box.setItemData(box.count() - 1, why, Qt.ToolTipRole)
            # THE GROUP'S OWN NAME IS THE DEFAULT when it is already a role:
            # a proposal that named its groups `wellID` should not make the
            # user say so again.
            chosen = self._roles.get(group, group if group in known else "")
            index = box.findData(chosen)
            box.setCurrentIndex(index if index >= 0 else box.count() - 1)
            box.currentIndexChanged.connect(
                lambda *_a, g=group, b=box: self._set_role(g, b.currentData()))
            self.roles_row.addWidget(box)
            self._boxes[group] = box
        self.roles_row.addStretch(1)

    def _set_role(self, group: str, role) -> None:
        self._roles[str(group)] = str(role or "")
        self.refresh()

    def roles(self) -> Dict[str, str]:
        """``{group: role}`` as the dropdowns currently stand."""
        out = {}
        for group in group_names(self.regex.text()):
            box = getattr(self, "_boxes", {}).get(group)
            if box is not None:
                out[group] = str(box.currentData() or "")
            else:
                out[group] = self._roles.get(group, "")
        return out

    # ------------------------------------------------------------ D

    def refresh(self):
        """Re-run the plan and redraw. Returns it."""
        names = [os.path.basename(p) for p in self._files]
        groups = group_names(self.regex.text())
        if set(groups) != set(getattr(self, "_boxes", {})):
            self._rebuild_roles(groups)
        self._plan = plan(names, self.regex.text(), self.roles(),
                          plate=self._plate_name())
        self.dropped.setText(self._plan.summary())
        self.role_trouble.setText(
            " · ".join(self._plan.trouble) if self._plan.trouble else "")
        self._fill_the_table()
        self.tree.setPlainText(
            "\n".join(self._plan.tree_lines())
            or "Nothing to organise yet.")
        return self._plan

    def _plate_name(self) -> str:
        """The folder the files came from, which is what the import uses."""
        if not self._files:
            return ""
        return os.path.basename(os.path.dirname(self._files[0]))

    def _fill_the_table(self) -> None:
        plan_ = self._plan
        rows = list(plan_.renamed) if plan_ else []
        missed = list(plan_.unmatched) if plan_ else []
        self.table.setRowCount(len(rows) + len(missed))
        for index, row in enumerate(rows):
            self.table.setItem(index, 0, table_item(row.before))
            self.table.setItem(index, 1, table_item(row.after))
        # UNMATCHED LAST AND NAMED, never dropped in silence: "412 of 480
        # matched" with the other 68 listed is an answer, and 412 files
        # appearing without comment is how half a plate goes missing.
        for offset, name in enumerate(missed):
            index = len(rows) + offset
            self.table.setItem(index, 0, table_item(name))
            cell = table_item("no match")
            cell.setForeground(Qt.red)
            cell.setToolTip(
                "The regex above does not match this name, so this file "
                "would not be imported at all.")
            self.table.setItem(index, 1, cell)
        self.table.resizeColumnsToContents()

    def the_plan(self):
        return self._plan


class ImportWorkbenchDialog(QDialog):
    """The workbench in a window, returning the accepted regex.

    :param filenames: the names to work the pattern out from.
    :param regex: the pattern to start from.
    :param parent: parent widget.

    Both are handed to the :class:`ImportWorkbench` this wraps, which
    documents what they mean.
    """

    def __init__(self, filenames: Sequence[str] = (), regex: str = "",
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Import images — work out the pattern")
        self.resize(1000, 680)
        outer = QVBoxLayout(self)
        self.workbench = ImportWorkbench(filenames, regex, parent=self)
        outer.addWidget(self.workbench, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def chosen_regex(self) -> str:
        r"""Return the accepted pattern for ``_get_regex`` custom mode.

        The workbench previews patterns against complete filenames, while
        ``_get_regex`` appends the selected image extension. This method
        removes that trailing extension with :func:`for_get_regex` to avoid
        duplicating it in the import pattern.
        """
        return for_get_regex(self.workbench.regex.text())
