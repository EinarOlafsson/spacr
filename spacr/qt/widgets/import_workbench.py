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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

    NOT FOR THE GUI THREAD. Every path here is one the user chose, which on
    a microscope rig means the share the images live on: the `isdir` is what
    wakes an `autofs` mount, and the walk is thousands more stats behind it.
    Call it from :func:`_walk` on a worker -- :meth:`ImportWorkbench.add_files`
    is the only caller in this module and that is what it does.
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


def _walk(paths: Sequence[str]) -> Tuple[List[str], str]:
    """Run :func:`images_under`, carrying any failure back as a string.

    ON THE WORKER THREAD, and the failure is RETURNED rather than raised on
    purpose. ``JobRunner`` hands a result to its ``on_done`` only for a job
    that succeeded, so a walk that raised would leave the "Working…" caption
    :meth:`ImportWorkbench.add_files` writes before it on screen for the rest
    of the session -- the panel would claim to still be looking at a folder
    it gave up on. Returned, the failure travels the same generation-guarded
    path as a result, so a walk the user has since abandoned cannot paint
    over the panel either.

    Looked up through the module global so a test may replace
    :func:`images_under`, which is how the freeze is reproduced.
    """
    try:
        return images_under(paths), ""
    except Exception as exc:                                     # noqa: BLE001
        LOG.info("could not walk what was dropped", exc_info=True)
        return [], str(exc) or exc.__class__.__name__


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
        #: What the last walk could not do, appended to the summary by
        #: :meth:`refresh`. Empty is the ordinary case.
        self._scan_trouble = ""
        self.setAcceptDrops(True)
        #: The worker that walks what was dropped. A DROP IS A PATH THE USER
        #: CHOSE, and a plate lives on the microscope's share: `images_under`
        #: stats it and then walks the whole tree. Measured on the
        #: maintainer's machine 2026-09-04, ONE `os.path.exists` under an
        #: `autofs` mount whose share was asleep had not returned after twenty
        #: seconds -- and a walk is thousands of those, on the thread that
        #: paints. Inline, the drop froze the application with no traceback,
        #: because a stalled event loop is not a crash; it was reported as
        #: hover flicker and glimpses of other screens.
        #:
        #: `user_visible=False`: the user dropped a folder, they did not
        #: start a run, so this must never claim a run banner on Home. Safe
        #: because the runner is ITS OWN and carries nothing but the walk --
        #: nothing else in this panel leaves the GUI thread, so the flag
        #: hides no work the user started.
        from ..job_runner import JobRunner
        self._scanner = JobRunner(self, threaded=True,
                                  app_key="import workbench scan",
                                  user_visible=False)
        self._scanner.job_failed.connect(self._scan_failed)

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
        """Accept a drag carrying images.

        :param event: the Qt drag event.
        """
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):                  # noqa: N802 - Qt
        """Keep accepting while images stay over the workbench.

        :param event: the Qt drag event.
        """
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):                      # noqa: N802 - Qt
        """Take the dropped images and work out their naming pattern.

        :param event: the Qt drop event.
        """
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        self.add_files(paths)
        event.acceptProposedAction()

    def ask_for_files(self) -> None:
        """Ask for images through a file dialog.

        RETURNS NOTHING, AND THAT IS THE CHANGE. It used to answer with how
        many files were taken, which it could only do by walking every
        dropped folder before returning -- on the GUI thread, which is the
        freeze this module was rewritten to remove. :meth:`add_files` hands
        the walk to a worker now, so the count does not exist yet when this
        returns; it arrives at :meth:`_files_found`.

        A caller that wants the number should watch the table, not this.
        """
        chosen, _filter = QFileDialog.getOpenFileNames(
            self, "Images to import", "",
            "Images (" + " ".join(f"*{s}" for s in IMAGE_SUFFIXES) + ")")
        self.add_files(chosen)

    def add_files(self, paths: Sequence[str]) -> None:
        """Add every image under ``paths``, once a worker has found them.

        SPLIT IN TWO, and the split is the fix for a frozen application.
        Everything here is a list of strings and a caption; the walk runs on
        this panel's own worker (:func:`_walk`, wrapping
        :func:`images_under`) because it is an `isdir` and a recursive
        `os.walk` over a path the user chose, which on one such workstation is
        an `autofs` share that took twenty seconds to answer a
        single stat. :meth:`_files_found` takes the answer back on the GUI
        thread.

        Nothing is added by the time this returns, which is the point. Read
        :meth:`files` from a redraw, not from the line after this one.

        Two drops in a row start two walks rather than one: unlike a
        refresh, they ask DIFFERENT questions, and coalescing them would
        lose a plate. Nothing here is shared mutable state -- both answers
        arrive on the GUI thread and both are kept.
        """
        wanted = [str(p) for p in paths or ()]
        if not wanted:
            return
        # SAID BEFORE THE WALK, not after: on a sleeping share the walk is
        # the part that takes seconds, and a panel that says nothing in the
        # meantime looks like a drop that was ignored. `refresh` replaces it
        # with the plan summary the moment there is one -- and `_walk` makes
        # sure there IS one even when the walk fails.
        #
        # THE CATALOGS ALREADY CARRY "Working…" in all nine languages. A
        # wordier caption invented here would be English-only until someone
        # noticed, and `tests/qt/test_i18n_caption_ratchet.py` fails on it.
        self._scan_trouble = ""
        self.dropped.setText("Working…")
        if not self._scanner.submit(lambda: _walk(wanted), self._files_found):
            # `submit` answers False only for a job that ran INLINE -- a
            # runner built `threaded=False`, which is how some tests drive
            # this panel -- and whose handler raised. Nothing is coming to
            # replace the caption above, so put the summary back rather than
            # leave the panel claiming to be working.
            self.refresh()

    def _files_found(self, answer: Optional[Any]) -> None:
        """Take what the walk returned, on the GUI thread.

        Generation-guarded by ``JobRunner``: a walk abandoned by
        :meth:`set_files` or :meth:`shutdown` never reaches here, so a folder
        the user cleared cannot reappear twenty seconds later.
        """
        found, trouble = answer if answer else ((), "")
        self._scan_trouble = str(trouble or "")
        seen = set(self._files)
        # `_show`, NOT `set_files`: `set_files` cancels, and a second drop
        # landing must not abandon the first drop's walk.
        self._show(self._files + [p for p in (found or ()) if p not in seen])

    def _scan_failed(self, message: str) -> None:
        """Repaint from what is held when a walk did not deliver at all.

        :func:`_walk` carries an ordinary failure back through
        :meth:`_files_found`, so this is the case that cannot reach: the
        worker itself did not finish. ``JobRunner`` calls ``on_done`` only
        for a job that succeeded, so without this the "Working…" caption
        would stay on screen for the rest of the session.

        NOT generation-guarded, because ``job_failed`` carries no job
        identity. Repainting from :attr:`_files` is the safe response for
        exactly that reason -- it can only ever show the truth as it now
        stands, never an abandoned walk's data. ``RuntimeError``: a worker
        parked by ``shutdown`` outlives this widget's C++ half.
        """
        try:
            LOG.info("a dropped folder was not walked: %s", message)
            self._scan_trouble = str(message or "")
            self.refresh()
        except RuntimeError:
            pass

    def is_scanning(self) -> bool:
        """True while a walk started by :meth:`add_files` is still running."""
        return self._scanner.is_busy()

    def set_files(self, paths: Sequence[str]) -> None:
        """Replace the file set and re-propose a pattern for it.

        A REGEX PROPOSED FOR THE OLD SET IS NOT PROPOSED FOR THIS ONE. The
        pattern is inferred from what varies across the names, so carrying it
        over would describe a set the user has replaced.

        ANY WALK STILL RUNNING IS ABANDONED.

        The Clear button lands here, and a walk of a share that is not
        answering is precisely the one the user gives up on. Without the
        cancel its result arrives half a minute later and quietly refills the
        table that was just emptied. ``JobRunner.cancel`` bumps a generation,
        so the result is dropped on arrival rather than handed to
        :meth:`_files_found`; the thread is left to retire itself, because
        joining it here would be the freeze this all exists to remove.

        :param paths: the image paths.
        """
        self._scanner.cancel()
        self._scan_trouble = ""
        self._show([str(p) for p in paths or ()])

    def _show(self, files: List[str]) -> None:
        """Hold ``files`` and redraw. The half of :meth:`set_files` a walk uses.

        Split off so that a landing walk does not cancel its siblings -- see
        :meth:`_files_found`.
        """
        self._files = files
        # A REGEX PROPOSED FOR THE OLD SET IS NOT PROPOSED FOR THIS ONE, so
        # the first drop offers one and a later drop does not overwrite what
        # the user has since edited.
        if self._files and not self.regex.text().strip():
            self.propose_from_the_names()
        self.refresh()

    def files(self) -> List[str]:
        """The files currently loaded.

        :returns: the paths, in load order.
        """
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
        said = self._plan.summary()
        if self._scan_trouble:
            # SAID, NOT SWALLOWED. A walk that failed leaves the table short
            # by a whole folder, and a summary that counts only what did
            # arrive reads as if that folder had held nothing.
            said += (f" · Could not read what you dropped: "
                     f"{self._scan_trouble}")
        self.dropped.setText(said)
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
        """The import plan the current pattern and roles produce.

        :returns: the plan.
        """
        return self._plan

    # ------------------------------------------------------------ shutdown

    def shutdown(self) -> None:
        """Abandon any walk in flight, briefly waiting for its thread.

        Qt ABORTS THE PROCESS if a running QThread is destroyed, and a drop
        on a share that is not answering is exactly the case where the user
        gives up and closes the window.
        """
        self._scanner.shutdown()

    def closeEvent(self, event):                     # noqa: N802 - Qt
        """Stop background work and unlink before going away.

        :param event: the Qt close event.
        """
        self.shutdown()
        super().closeEvent(event)


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

    def done(self, result: int) -> None:              # Qt override
        """Close, and let no walk outlive the dialog.

        ``done`` rather than ``closeEvent`` because it is the one funnel:
        Ok, Cancel and the window's close button all arrive here, and a
        dialog dismissed while the share is still being walked is the
        ordinary case -- it is why the user gave up on it.
        """
        self.workbench.shutdown()
        super().done(result)

    def chosen_regex(self) -> str:
        r"""Return the accepted pattern for ``_get_regex`` custom mode.

        The workbench previews patterns against complete filenames, while
        ``_get_regex`` appends the selected image extension. This method
        removes that trailing extension with :func:`for_get_regex` to avoid
        duplicating it in the import pattern.
        """
        return for_get_regex(self.workbench.regex.text())
