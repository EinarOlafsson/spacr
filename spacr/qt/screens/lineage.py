"""``V9`` ``B20`` — the containment tree: cell → nucleus → pathogen.

The links have been in ``measurements.db`` since the first Measure run:
``nucleus`` and ``pathogen`` rows carry a ``cell_id`` naming the cell they sit
inside (:data:`spacr.schema.CHILD_OBJECT_TABLES`). Nothing has ever shown
them. "This cell holds one nucleus and four pathogens" was a fact you could
compute and could not see, and the questions that follow from it — which cells
are uninfected, which pathogen is impossibly large for the cell around it,
which nucleus belongs to no cell at all — had no view.

:mod:`spacr.lineage` builds the tree in plain pandas. This is the tree widget
over it: a node selected here publishes through
:mod:`spacr.qt.linked_selection`, so picking a cell rings the same cell on the
plate view and in the UMAP, and picking it with its contents rings the whole
family. Double-clicking opens the crops.

Orphans are a tab, not a footnote
---------------------------------

A child whose ``cell_id`` names no cell is dropped by every join in the
codebase. It is not noise: it means the nucleus mask found an object the cell
mask did not, which is a segmentation disagreement worth looking at. It gets
its own list here, with the same double-click-to-open, because a finding you
have to write SQL to see is a finding nobody sees.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QPushButton, QSplitter, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from ... import lineage as lin
from ..job_runner import JobRunner
from ...selection import match_keys
from ..linked_selection import DEFAULT_OPEN_KIND, LinkedView, has_object_opener
from ..theme import SPACING, active_palette

LOG = logging.getLogger(__name__)

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
    """

    #: A node was selected. Carries its object key.
    node_selected = Signal(str)

    def __init__(self, parent=None, *, threaded: bool = True):
        super().__init__(parent)
        self.setObjectName("LineageScreen")
        self._jobs = JobRunner(self, threaded=bool(threaded),
                               app_key="lineage")
        self._jobs.job_failed.connect(self._on_job_failed)
        self._frames: Dict[str, pd.DataFrame] = {}
        self._forest: tuple = ()
        self._orphans = pd.DataFrame()
        self._build()
        self.link_selection(LINK_SOURCE)
        # Drop anywhere on this screen: the path is resolved through spaCR's
        # project layout, so the plate folder finds what this screen reads.
        from ..dnd import install_for
        install_for(self, "lineage")

    # -- construction --------------------------------------------------------
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["sm"])

        title = QLabel("Lineage", self)
        title.setObjectName("DisplayHeading")
        outer.addWidget(title)
        intro = QLabel(
            "What is inside what: every cell with the nuclei and pathogens "
            "it contains. The links were always in the database; this is the "
            "first view of them.", self)
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
        outer.addLayout(source)

        split = QSplitter(Qt.Horizontal, self)

        left = QWidget(self)
        left_column = QVBoxLayout(left)
        left_column.setContentsMargins(0, 0, 0, 0)
        left_column.setSpacing(4)
        self.tree = QTreeWidget(left)
        self.tree.setHeaderLabels(["Object", "Inside it", "Key"])
        self.tree.setColumnHidden(2, True)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.itemSelectionChanged.connect(self._on_tree_selection)
        self.tree.itemDoubleClicked.connect(self._on_tree_activated)
        left_column.addWidget(self.tree, 1)
        buttons = QHBoxLayout()
        self._publish_family = QPushButton("Select with contents", left)
        self._publish_family.setToolTip(
            "Publish the selected object AND everything inside it, so the "
            "whole family lights up in every other open view.")
        self._publish_family.clicked.connect(self.publish_family)
        buttons.addWidget(self._publish_family)
        self._open_button = QPushButton("Open crops", left)
        self._open_button.setToolTip(
            "Show the selected objects as crops, parents first.")
        self._open_button.clicked.connect(self.open_selected)
        buttons.addWidget(self._open_button)
        left_column.addLayout(buttons)
        split.addWidget(left)

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
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([620, 300])
        outer.addWidget(split, 1)

        self.status = QLabel("", self)
        self.status.setObjectName("Muted")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

    # -- loading -------------------------------------------------------------
    def _choose_db(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a measurements database", self._db.text().strip(),
            "SQLite (*.db *.sqlite);;All files (*)")
        if path:
            self._db.setText(path)
            self.load()

    def load(self) -> None:
        """Read the object tables and build the forest, off the GUI thread."""
        db_path = self._db.text().strip()
        if not db_path or not os.path.isfile(db_path):
            self.status.setText("Choose a measurements database first.")
            return
        self.status.setText("Reading the object tables…")
        self._jobs.submit(lambda: lin.read_object_tables(db_path),
                          self.set_frames)

    def set_frames(self, frames: Dict[str, pd.DataFrame]) -> None:
        """Build and draw the tree from already-loaded tables.

        The seam a test — or another screen holding the same frames — goes
        through, so nothing here needs a database to be exercised.
        """
        self._frames = dict(frames or {})
        try:
            self._forest = lin.build_forest(self._frames)
            self._orphans = lin.orphans(self._frames)
        except lin.LineageError as exc:
            self._forest = ()
            self._orphans = pd.DataFrame()
            self.tree.clear()
            self.orphan_list.clear()
            self.summary.setText(str(exc))
            self.status.setText("Nothing to show.")
            return
        self._fill_tree()
        self._fill_orphans()
        note = self.collision_note()
        self.summary.setText(
            lin.describe_forest(self._forest) + (f"\n{note}" if note else ""))
        shown = min(len(self._forest), TREE_LIMIT)
        self.status.setText(
            f"{shown} of {len(self._forest)} parent object(s) in the tree"
            + (f" · {len(self._orphans)} unattached child(ren)"
               if len(self._orphans) else " · every child has a parent"))

    def _fill_tree(self) -> None:
        self.tree.clear()
        for root in self._forest[:TREE_LIMIT]:
            self.tree.addTopLevelItem(self._item(root))

    def _item(self, node: "lin.LineageNode") -> QTreeWidgetItem:
        """One node and its descendants as tree items.

        The key is stashed on the item rather than re-derived from the label
        text: a label is for reading and a key is for matching, and the two
        have drifted apart in this codebase before.
        """
        inside = {t: n for t, n in node.counts().items() if t != node.table}
        item = QTreeWidgetItem([
            f"{node.table} {node.label}",
            ", ".join(f"{n} {t}" for t, n in sorted(inside.items())),
            node.key,
        ])
        item.setData(0, _KEY_ROLE, node.key)
        item.setData(0, _ID_ROLE, node.node_id)
        item.setToolTip(0, node.describe())
        for child in node.children:
            item.addChild(self._item(child))
        return item

    def _fill_orphans(self) -> None:
        self.orphan_list.clear()
        if self._orphans.empty:
            self.orphan_list.addItem(QListWidgetItem(
                "(none — every child names a parent that exists)"))
            return
        for _index, row in self._orphans.iterrows():
            # `orphans` stamps each row with the table it came from, so the
            # key an unattached child publishes says which child it is —
            # the same identity the tree uses, rather than one that names
            # every object with that label in the field.
            key = lin.node_key(row, str(row.get("table") or "") or None)
            claimed = str(row.get("parent_id") or "")
            text = (f"{row['table']} {row[lin.schema.OBJECT_LABEL_KEY]} → "
                    + (f"cell {claimed} (missing)" if claimed
                       else "no cell_id at all"))
            item = QListWidgetItem(text)
            item.setData(_KEY_ROLE, key)
            item.setToolTip(key)
            self.orphan_list.addItem(item)

    def _on_job_failed(self, message: str) -> None:
        self.status.setText(message)
        self.status.setStyleSheet(f"color: {active_palette()['error']};")

    # -- reading the selection back out --------------------------------------
    def selected_keys(self) -> List[str]:
        """The object keys of the selected tree rows, in tree order."""
        return [str(item.data(0, _KEY_ROLE))
                for item in self.tree.selectedItems()
                if item.data(0, _KEY_ROLE)]

    def family_keys(self) -> List[str]:
        """The selected rows and everything inside them, parents first.

        De-duplicated while keeping order: selecting a cell and one of its
        pathogens must not open that pathogen twice.

        The de-duplication is on the SHARED key, which is what the routing
        contract takes — so a cell whose nucleus 1 and pathogen 1 collide
        yields three keys for four objects. That is not a bug here, it is the
        key having no object type in it; :meth:`collision_note` is what puts
        it on screen instead of leaving the arithmetic unexplained.
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
        key = item.data(0, _KEY_ROLE)
        keys = [str(key)] if key else []
        for index in range(item.childCount()):
            keys.extend(self._subtree_keys(item.child(index)))
        return keys

    def _subtree_ids(self, item: QTreeWidgetItem) -> List[str]:
        node_id = item.data(0, _ID_ROLE)
        ids = [str(node_id)] if node_id else []
        for index in range(item.childCount()):
            ids.extend(self._subtree_ids(item.child(index)))
        return ids

    def collision_note(self) -> str:
        """One line naming any objects the shared key cannot tell apart.

        **Empty, now and normally.** The object key carries the object type,
        so a cell's nucleus 1 and its pathogen 1 are two keys; this used to
        fire on every family that had both, and it is kept as the alarm for
        that ever being true again. It is cheap, and the failure it watches
        for — "opening four objects showed three crops" — is otherwise a
        mystery rather than a message.
        """
        collisions = lin.forest_key_collisions(self._forest)
        if not collisions:
            return ""
        example = sorted(collisions)[0]
        return (f"{len(collisions)} object key(s) name more than one object — "
                f"{example} is a "
                f"{' and a '.join(collisions[example])}. Every other view "
                f"treats them as one object, so opening this family will "
                f"show fewer crops than it has objects.")

    # -- publishing ----------------------------------------------------------
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
        key = item.data(0, _KEY_ROLE)
        if key:
            self._open([str(key)], f"double-clicked in the lineage tree")

    def _on_orphan_activated(self, item: QListWidgetItem) -> None:
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

    # -- the shared selection ------------------------------------------------
    def on_linked_selection_changed(self, selection) -> None:
        """Reveal and highlight what another view selected, when we hold it."""
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
        self.unlink_selection()
        self._jobs.cancel()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

APP_NAME = "Lineage"
APP_DESCRIPTION = "What is inside what: cell → nucleus → pathogen"
APP_INTRO = (
    "Every cell with the nuclei and pathogens it contains, read off the "
    "cell_id links Measure has always written. Selecting a node highlights "
    "the same object in every other open view; 'Select with contents' "
    "highlights the whole family. Children whose cell_id names no cell get "
    "their own list — that is the two masks disagreeing, and it is a finding "
    "rather than noise.")
APP_CLI_NOTE = (
    "Lineage is an interactive tree; run it in the GUI (spacr-qt). Headless, "
    "spacr.lineage.build_forest gives the same tree as data.")


def make_lineage_screen(**_kwargs) -> LineageScreen:
    """Build the screen. The ``factory=`` for :func:`spacr.qt.app.register_app`."""
    return LineageScreen()


def register(*, section: Optional[str] = None, stage: Optional[str] = None,
             key: str = APP_KEY):
    """Put Lineage in the app registry. Idempotent.

    :returns: the registry row, or ``None`` when the key was already there.
    """
    from ..app import APPS, SECTION_EXPLORE, STAGE_ALPHA, register_app
    if any(row[0] == key for row in APPS):
        return None
    return register_app(
        key, APP_NAME, APP_DESCRIPTION, section or SECTION_EXPLORE,
        factory=make_lineage_screen,
        stage=STAGE_ALPHA if stage is None else stage,
        intro=APP_INTRO, cli_note=APP_CLI_NOTE,
        api_module="qt/screens/lineage",
        translations=("Härstamning", "Abstammung", "Linaje", "谱系",
                      "Linhagem", "वंशावली", "계보", "Ætterni", "Lignée"))
