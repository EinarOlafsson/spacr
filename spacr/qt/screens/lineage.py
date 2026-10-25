"""Lineage screen: the containment tree of cells, nuclei and pathogens.

In ``measurements.db``, ``nucleus`` and ``pathogen`` rows carry a ``cell_id``
naming the cell they sit inside (:data:`spacr.schema.CHILD_OBJECT_TABLES`).
This screen shows those links as a tree, so questions such as which cells are
uninfected, which pathogen is implausibly large for the cell around it, and
which nucleus belongs to no cell at all can be answered by inspection.

:mod:`spacr.lineage` builds the tree in plain pandas. This is the tree widget
over it: a node selected here publishes through
:mod:`spacr.qt.linked_selection`, so picking a cell highlights the same cell
on the plate view and in the UMAP, and picking it with its contents highlights
the whole family. Double-clicking opens the crops.

Unattached children are listed separately
-----------------------------------------

A child whose ``cell_id`` names no cell is dropped by every join in the
codebase. It is not noise: it means the nucleus mask found an object the cell
mask did not, which is a segmentation disagreement worth looking at. It gets
its own list here, with the same double-click-to-open, so these objects can be
reviewed without writing SQL.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QPushButton, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from ... import lineage as lin
from .. import _after
from ..job_runner import JobRunner
from ...selection import match_keys
from ..linked_selection import DEFAULT_OPEN_KIND, LinkedView, has_object_opener
from ..theme import SPACING, active_palette, mark_surface
from ..widgets.collapsible_splitter import CollapsibleSplitter
from ..widgets.measurements_example import install_test_data_button
from ..widgets.sortable_table import (
    _keep_sorting_held, _prime_row_key, _reuse_list_items, _reuse_tree_items,
    _settle_sorting, _wipe, install_sorting)
from ..app_catalog import declared_app, register_declared

LOG = logging.getLogger(__name__)

#: Longest a turn of :meth:`LineageScreen._fill_tree_in_turns` writes rows
#: for, in seconds, before it lets the event loop run.
_FILL_TURN_S = 0.02

__all__ = [
    "LineageScreen",
    "register",
    "APP_KEY",
    "LINK_SOURCE",
]

APP_KEY = "lineage"

#: What this view calls itself on the shared selection.
LINK_SOURCE = "lineage"

#: How many root objects to put in the tree at once. A field of 4 000 cells
#: is 20 000 tree items, which Qt will build and nobody will scroll; the tree
#: is for looking at a family, and the search box is how you reach one.
TREE_LIMIT = 2000


class _BuiltLineage(NamedTuple):
    """One loaded database's tree, built off the GUI thread. Plain data."""

    frames: Dict[str, pd.DataFrame]
    forest: tuple
    orphans: pd.DataFrame
    orphan_rows: Tuple[Tuple[str, str], ...]
    collisions: Dict[str, Any]
    summary: str
    problem: Optional[str]


def _build_lineage(frames) -> _BuiltLineage:
    """Build the forest, the orphan rows and their texts from loaded tables.

    Every row's typed key goes through :func:`spacr.schema.object_id`, which
    is per-row string work; on a plate of objects it froze the window for
    1-2 s when it ran in the GUI thread's callback. The load job runs this on
    the worker, so the GUI thread only writes the capped tree.

    :param frames: ``{table name: rows}``; ``None`` counts as empty.
    :returns: the built tree, or the ``LineageError`` text in ``problem``.
    """
    frames = dict(frames or {})
    try:
        forest = lin.build_forest(frames)
        orphans = lin.orphans(frames)
    except lin.LineageError as exc:
        return _BuiltLineage(frames, (), pd.DataFrame(), (), {}, "", str(exc))
    rows = []
    for _index, row in orphans.iterrows():
        key = lin.node_key(row, str(row.get("table") or "") or None)
        claimed = str(row.get("parent_id") or "")
        text = (f"{row['table']} {row[lin.schema.OBJECT_LABEL_KEY]} → "
                + (f"cell {claimed} (missing)" if claimed
                   else "no cell_id at all"))
        rows.append((text, key))
    return _BuiltLineage(frames, forest, orphans, tuple(rows),
                         lin.forest_key_collisions(forest),
                         lin.describe_forest(forest), None)

#: Where the shared object key is stashed on a tree row — what gets
#: published and routed.
_KEY_ROLE = Qt.UserRole
#: Where the table-qualified identity is stashed. The two are different on
#: purpose: a nucleus 1 and a pathogen 1 inside one cell share the shared key
#: (see :mod:`spacr.lineage`), so anything that must address ONE row — a
#: de-duplication, a lookup — uses this one.
_ID_ROLE = Qt.UserRole + 1


class LineageScreen(LinkedView, QWidget):
    """The containment tree for one measurements database.

    :param threaded: ``False`` reads inline, so a test drives the screen
        without a worker thread and gets the same calls in the same order.
    :param parent: parent widget; ownership only.
    """

    #: A node was selected. Carries its object key.
    node_selected = Signal(str)

    def __init__(self, parent=None, *, threaded: bool = True):
        """Build the screen, join the shared selection and arm its drop zone.

        :param parent: parent widget, or ``None``.
        :param threaded: read the database on a worker thread. Set ``False`` in
            tests so ``load`` finishes before it returns.
        """
        super().__init__(parent)
        self.setObjectName("LineageScreen")
        self._jobs = JobRunner(self, threaded=bool(threaded),
                               app_key="lineage")
        self._jobs.job_failed.connect(self._on_job_failed)
        self._frames: Dict[str, pd.DataFrame] = {}
        self._forest: tuple = ()
        self._orphans = pd.DataFrame()
        self._orphan_rows: Tuple[Tuple[str, str], ...] = ()
        self._collisions: Dict[str, Any] = {}
        self._build()
        self.link_selection(LINK_SOURCE)
        from ..dnd import install_for
        install_for(self, "lineage")

    def _build(self) -> None:
        """Lay out the source row, the containment tree and the orphan list."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["sm"])

        title = QLabel("Lineage", self)
        title.setObjectName("DisplayHeading")
        outer.addWidget(title)
        intro = QLabel(
            "Every cell with the nuclei and pathogens it contains, read from "
            "the cell_id links in the measurements database.", self)
        intro.setObjectName("Muted")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        source = QHBoxLayout()
        source.addWidget(QLabel("measurements.db", self))
        self._db = QLineEdit(self)
        self._db.setPlaceholderText("…/measurements/measurements.db")
        self._db.returnPressed.connect(self.load)
        source.addWidget(self._db, 1)
        self._browse = QPushButton("Browse…", self)
        self._browse.clicked.connect(self._choose_db)
        source.addWidget(self._browse)
        self._reload = QPushButton("Build tree", self)
        self._reload.setObjectName("PrimaryButton")
        self._reload.clicked.connect(self.load)
        source.addWidget(self._reload)
        install_test_data_button(
            self, source, self._open_the_example,
            say=lambda message: self.status.setText(message))
        outer.addLayout(source)

        split = CollapsibleSplitter(Qt.Horizontal, self,
                                    persist_key=f"{APP_KEY}::body")
        self._body = split

        left = QWidget(self)
        left_column = QVBoxLayout(left)
        left_column.setContentsMargins(0, 0, 0, 0)
        left_column.setSpacing(4)
        self.tree = QTreeWidget(left)
        install_sorting(self.tree)
        self.tree.setHeaderLabels(["Object", "Inside it", "Key"])
        self.tree.setColumnHidden(2, True)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.itemSelectionChanged.connect(self._on_tree_selection)
        self.tree.itemDoubleClicked.connect(self._on_tree_activated)
        left_column.addWidget(self.tree, 1)
        buttons = QHBoxLayout()
        self._publish_family = QPushButton("Select with contents", left)
        self._publish_family.setToolTip(
            "Publish the selected object and everything inside it, so the "
            "whole family is highlighted in every other open view.")
        self._publish_family.clicked.connect(self.publish_family)
        buttons.addWidget(self._publish_family)
        self._open_button = QPushButton("Open crops", left)
        self._open_button.setToolTip(
            "Show the selected objects as crops, parents first.")
        self._open_button.clicked.connect(self.open_selected)
        buttons.addWidget(self._open_button)
        left_column.addLayout(buttons)
        self.tree_section = split.add_section(
            left, "Objects", persist_key=f"{APP_KEY}/Objects", stretch=1,
            extent=620)

        right = QWidget(self)
        right_column = QVBoxLayout(right)
        right_column.setContentsMargins(0, 0, 0, 0)
        right_column.setSpacing(4)
        self.summary = QLabel("", right)
        self.summary.setWordWrap(True)
        right_column.addWidget(self.summary)
        self.orphan_heading = QLabel("Unattached children", right)
        right_column.addWidget(self.orphan_heading)
        self.orphan_list = QListWidget(right)
        self.orphan_list.setToolTip(
            "Children whose cell_id names no cell in their field. The masks "
            "disagree — the child segmentation found an object the parent "
            "segmentation did not. Double-click to open one.")
        self.orphan_list.itemDoubleClicked.connect(self._on_orphan_activated)
        right_column.addWidget(self.orphan_list, 1)
        self.orphans_section = split.add_section(
            right, "Unattached children",
            persist_key=f"{APP_KEY}/Unattached children", stretch=0,
            extent=300)
        mark_surface(self.tree, self.orphan_list)
        outer.addWidget(split, 1)

        self.status = QLabel("", self)
        self.status.setObjectName("Muted")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

    def _choose_db(self) -> None:
        """Ask for a measurements database and build the tree from it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a measurements database", self._db.text().strip(),
            "SQLite (*.db *.sqlite);;All files (*)")
        if path:
            self._db.setText(path)
            self.load()

    def _open_the_example(self, _folder, database) -> None:
        """Fill the database box with the example plate's and build its tree.

        :param _folder: the example plate folder.
        :param database: its ``measurements/measurements.db``, whose nucleus
            and pathogen rows carry the ``cell_id`` the tree is built from.
        """
        self._db.setText(str(database))
        self.load()

    def load(self) -> None:
        """Read the object tables and build the forest, off the GUI thread."""
        db_path = self._db.text().strip()
        if not db_path or not os.path.isfile(db_path):
            self.status.setText("Choose a measurements database first.")
            return
        self.status.setText("Reading the object tables…")
        self._jobs.submit(
            lambda: _build_lineage(lin.read_object_tables(db_path)),
            self._show_built)

    def set_frames(self, frames: Dict[str, pd.DataFrame]) -> None:
        """Build and draw the tree from already-loaded tables.

        The seam a test — or another screen holding the same frames — goes
        through, so nothing here needs a database to be exercised.

        :param frames: ``{table name: rows}``, as
            :func:`spacr.lineage.build_forest` takes it; ``None`` counts as
            empty. A ``LineageError`` from building the tree is shown in the
            summary instead of raised.
        """
        self._show_built(_build_lineage(frames), in_turns=False)

    def _show_built(self, built: _BuiltLineage,
                    in_turns: Optional[bool] = None) -> None:
        """Draw a tree :func:`_build_lineage` has already built.

        :param built: the worker's result for the current load.
        :param in_turns: write the tree over several event-loop turns
            (:meth:`_fill_tree_in_turns`), the status line following when
            the last row is written. ``None`` does so on a screen that reads
            on a worker thread: rewriting a 7,000-node tree took 210-230 ms
            in one turn, the Build tree freeze of lesson 56.
        """
        if in_turns is None:
            in_turns = bool(getattr(self._jobs, "_threaded", True))
        self._fill_generation = getattr(self, "_fill_generation", 0) + 1
        self._frames = built.frames
        self._forest = built.forest
        self._orphans = built.orphans
        self._orphan_rows = built.orphan_rows
        self._collisions = built.collisions
        if built.problem is not None:
            self.tree.setUpdatesEnabled(True)
            _reuse_tree_items(self.tree, 0)
            _reuse_list_items(self.orphan_list, 0)
            self.summary.setText(built.problem)
            self.status.setText("Nothing to show.")
            return
        self._fill_orphans()
        note = self.collision_note()
        self.summary.setText(built.summary + (f"\n{note}" if note else ""))
        if in_turns:
            self._fill_tree_in_turns(self._say_what_is_shown)
        else:
            self._fill_tree()
            self._say_what_is_shown()

    def _say_what_is_shown(self) -> None:
        """Put how much of the forest the tree shows on the status line."""
        shown = min(len(self._forest), TREE_LIMIT)
        self.status.setText(
            f"{shown} of {len(self._forest)} parent object(s) in the tree"
            + (f" · {len(self._orphans)} unattached child(ren)"
               if len(self._orphans) else " · every child has a parent"))

    def _fill_tree(self) -> None:
        """Rebuild the containment tree from the loaded forest.

        Capped at ``TREE_LIMIT`` roots: a plate's worth of cells would take
        longer to build than to read.
        """
        roots = self._forest[:TREE_LIMIT]
        rows = _reuse_tree_items(self.tree, len(roots))
        for item, root in zip(rows, roots):
            self._item(root, item)
        if not self.tree.updatesEnabled():
            self.tree.setUpdatesEnabled(True)

    def _fill_tree_in_turns(self, done) -> None:
        """:meth:`_fill_tree`, at most :data:`_FILL_TURN_S` of rows a turn.

        The top-level rows are counted out first and each is wiped in the
        turn that rewrites it. The tree is not repainted while it is half
        written, and its sort stays held until the last row is in: each row's
        sort key is read in the turn that writes it, so the one sort at the
        end compares keys already read. A newer fill makes this one stop.

        :param done: called once the last row is written and sorted.
        """
        import time

        generation = self._fill_generation
        tree = self.tree
        roots = self._forest[:TREE_LIMIT]
        rows = _reuse_tree_items(tree, len(roots), wipe=False)
        columns = max(1, tree.columnCount())
        work = list(zip(rows, roots))
        position = [0]
        tree.setUpdatesEnabled(False)

        def turn() -> None:
            """Write rows for one turn, then hand the event loop back."""
            if generation != self._fill_generation:
                return
            started = time.perf_counter()
            while position[0] < len(work):
                item, root = work[position[0]]
                position[0] += 1
                _wipe(tree, item, columns)
                self._item(root, item, prime=True)
                if time.perf_counter() - started >= _FILL_TURN_S:
                    break
            _keep_sorting_held(tree)
            _after(0, self, turn if position[0] < len(work) else finish)

        def finish() -> None:
            """Sort once on the keys read, then show the tree again."""
            if generation != self._fill_generation:
                return
            _settle_sorting(tree)
            tree.setUpdatesEnabled(True)
            done()

        turn()

    def _item(self, node: "lin.LineageNode",
              item: QTreeWidgetItem, prime: bool = False) -> QTreeWidgetItem:
        """Write one node and its descendants into a tree item.

        The item and any children it already has are rewritten in place, so a
        reload deletes no item.

        The key is stashed on the item rather than re-derived from the label
        text: a label is for reading and a key is for matching, and the two
        have drifted apart in this codebase before.

        :param prime: read each written row's key for the held sort
            (:func:`spacr.qt.widgets.sortable_table._prime_row_key`).
        """
        inside = {t: n for t, n in node.counts().items() if t != node.table}
        texts = [
            f"{node.table} {node.label}",
            ", ".join(f"{n} {t}" for t, n in sorted(inside.items())),
            node.key,
        ]
        for column, text in enumerate(texts):
            item.setText(column, text)
        item.setData(0, _KEY_ROLE, node.key)
        item.setData(0, _ID_ROLE, node.node_id)
        item.setToolTip(0, node.describe())
        if prime:
            _prime_row_key(self.tree, item)
        kids = _reuse_tree_items(item, len(node.children))
        for kid, child in zip(kids, node.children):
            self._item(child, kid, prime)
        return item

    def _fill_orphans(self) -> None:
        """List the children whose ``cell_id`` names no cell in their field.

        Each row publishes a key stamped with the table it came from, so an
        unattached child identifies itself the way the tree does rather than
        naming every object with that label in the field. An empty list says so
        in words rather than being left blank.
        """
        if self._orphans.empty:
            only = _reuse_list_items(self.orphan_list, 1)[0]
            only.setData(Qt.DisplayRole,
                         "(none — every child names a parent that exists)")
            return
        rows = _reuse_list_items(self.orphan_list, len(self._orphan_rows))
        for item, (text, key) in zip(rows, self._orphan_rows):
            item.setText(text)
            item.setData(_KEY_ROLE, key)
            item.setToolTip(key)

    def _on_job_failed(self, message: str) -> None:
        """Show a failed load on the status line, in the error colour.

        :param message: the failure text from the job runner.
        """
        self.status.setText(message)
        self.status.setStyleSheet(f"color: {active_palette()['error']};")

    def selected_keys(self) -> List[str]:
        """The object keys of the selected tree rows, in tree order."""
        return [str(item.data(0, _KEY_ROLE))
                for item in self.tree.selectedItems()
                if item.data(0, _KEY_ROLE)]

    def family_keys(self) -> List[str]:
        """The selected rows and everything inside them, parents first.

        De-duplicated while keeping order: selecting a cell and one of its
        pathogens must not open that pathogen twice.

        The de-duplication is on the shared key, which is what the routing
        contract takes. That key carries the object type, so a cell's
        nucleus 1 and pathogen 1 stay two keys; if keys ever collide, fewer
        keys than objects are returned and :meth:`collision_note` reports
        it on screen.
        """
        out: List[str] = []
        seen = set()
        for item in self.tree.selectedItems():
            for key in self._subtree_keys(item):
                if key not in seen:
                    seen.add(key)
                    out.append(key)
        return out

    def family_ids(self) -> List[str]:
        """The same rows by table-qualified identity — one entry per object."""
        out: List[str] = []
        seen = set()
        for item in self.tree.selectedItems():
            for node_id in self._subtree_ids(item):
                if node_id not in seen:
                    seen.add(node_id)
                    out.append(node_id)
        return out

    def _subtree_keys(self, item: QTreeWidgetItem) -> List[str]:
        """Collect the object keys of an item and everything under it.

        :param item: the subtree root.
        :returns: its key first, then its descendants' in tree order; items
            carrying no key contribute nothing.
        """
        key = item.data(0, _KEY_ROLE)
        keys = [str(key)] if key else []
        for index in range(item.childCount()):
            keys.extend(self._subtree_keys(item.child(index)))
        return keys

    def _subtree_ids(self, item: QTreeWidgetItem) -> List[str]:
        """Collect the node ids of an item and everything under it.

        :param item: the subtree root.
        :returns: its id first, then its descendants' in tree order; items
            carrying no id contribute nothing.
        """
        node_id = item.data(0, _ID_ROLE)
        ids = [str(node_id)] if node_id else []
        for index in range(item.childCount()):
            ids.extend(self._subtree_ids(item.child(index)))
        return ids

    def collision_note(self) -> str:
        """One line naming any objects the shared key cannot tell apart.

        **Normally empty.** The object key carries the object type, so a
        cell's nucleus 1 and its pathogen 1 are two keys. The check is kept
        to report a regression in which keys collide again; without it,
        "opening four objects showed three crops" would have no explanation
        on screen.
        """
        collisions = self._collisions
        if not collisions:
            return ""
        example = sorted(collisions)[0]
        return (f"{len(collisions)} object key(s) name more than one object — "
                f"{example} is a "
                f"{' and a '.join(collisions[example])}. Every other view "
                f"treats them as one object, so opening this family will "
                f"show fewer crops than it has objects.")

    def _on_tree_selection(self) -> None:
        """A row was picked: highlight exactly it, everywhere.

        Exactly it, not its family — expanding a selection behind the user's
        back means the plate view lights up five objects for one click and
        there is no way to ask for the one. "Select with contents" is the
        other act, and it is a button.
        """
        keys = self.selected_keys()
        if not keys:
            return
        self.publish_selection(keys)
        self.node_selected.emit(keys[0])

    def publish_family(self) -> List[str]:
        """Publish the selected objects together with everything inside them."""
        keys = self.family_keys()
        if keys:
            self.publish_selection(keys)
        return keys

    def _on_tree_activated(self, item: QTreeWidgetItem, _column: int) -> None:
        """Open the crop for a double-clicked tree row.

        :param item: the activated row.
        :param _column: the column that was hit; unused, since the row is what
            identifies the object.
        """
        key = item.data(0, _KEY_ROLE)
        if key:
            self._open([str(key)], f"double-clicked in the lineage tree")

    def _on_orphan_activated(self, item: QListWidgetItem) -> None:
        """Open the crop for a double-clicked unattached child.

        :param item: the activated row.
        """
        key = item.data(_KEY_ROLE)
        if key:
            self._open([str(key)],
                       "unattached child — its cell_id names no cell")

    def open_selected(self) -> Any:
        """Open the selected objects as crops, parents before their contents."""
        keys = self.family_keys()
        if not keys:
            self.status.setText("Select something in the tree first.")
            return None
        return self._open(keys, "selected in the lineage tree, parents first")

    def _open(self, keys: List[str], reason: str) -> Any:
        """Open crops for a set of object keys, reporting whatever stops it.

        :param keys: the objects to show.
        :param reason: what prompted the request, passed through to the opener
            so the receiving screen can say where the selection came from.
        :returns: whatever :meth:`open_objects` returns, or ``None`` when no
            opener is registered or the open failed -- either way the status
            line says which.
        """
        if not has_object_opener(DEFAULT_OPEN_KIND):
            self.status.setText(
                "Open the Annotate screen first — it is what shows crops.")
            return None
        try:
            return self.open_objects(keys, reason=reason)
        except Exception as exc:
            LOG.exception("Could not open a lineage selection")
            self.status.setText(f"Could not open those objects: {exc}")
            return None

    def on_linked_selection_changed(self, selection) -> None:
        """Reveal and highlight what another view selected, if the tree contains it.

        :param selection: the shared selection published by another view;
            its ``keys`` are matched against the tree's object keys, and
            ``None`` keys clear the highlight.
        """
        if selection.keys is None:
            self.tree.clearSelection()
            return
        wanted = [str(key) for key in selection.keys]
        blocked = self.tree.blockSignals(True)
        try:
            self.tree.clearSelection()
            found = 0
            iterator = [self.tree.topLevelItem(i)
                        for i in range(self.tree.topLevelItemCount())]
            while iterator:
                item = iterator.pop()
                if match_keys([item.data(0, _KEY_ROLE)], wanted)[0]:
                    item.setSelected(True)
                    found += 1
                    parent = item.parent()
                    while parent is not None:
                        parent.setExpanded(True)
                        parent = parent.parent()
                for index in range(item.childCount()):
                    iterator.append(item.child(index))
        finally:
            self.tree.blockSignals(blocked)
        if found:
            self.status.setText(
                f"{found} of the {len(wanted)} selected object(s) are in this "
                f"tree.")

    def closeEvent(self, event) -> None:
        """Stop background work and unlink before going away.

        :param event: the Qt close event.
        """
        self.unlink_selection()
        self._jobs.cancel()
        super().closeEvent(event)



_ROW = declared_app(APP_KEY)
APP_NAME = _ROW.name
APP_DESCRIPTION = _ROW.desc
APP_INTRO = _ROW.intro
APP_CLI_NOTE = _ROW.cli_note


def make_lineage_screen(**_kwargs) -> LineageScreen:
    """Build the screen. The ``factory=`` for :func:`spacr.qt.app.register_app`."""
    return LineageScreen()


def register(*, section: Optional[str] = None, stage: Optional[str] = None,
             key: str = APP_KEY):
    """Put Lineage in the app registry. Idempotent.

    :returns: the registry row, or ``None`` when the key was already there.
    """
    return register_declared(
        __name__, key=key, section=section, stage=stage)
