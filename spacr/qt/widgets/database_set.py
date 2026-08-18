"""The set of measurement databases a screen is working on.

Instruction 109, point 1: *"Wherever a module accepts one database, accept
several: add, remove, and see what is loaded."* And the idiom it names --
"the little rounded box with an x per member of a working set" -- which is
:class:`spacr.qt.widgets.table_chip.TableChip`, already used for the Gate
Editor's table working set. Databases get the same control rather than a
second one for the same idea.

WHAT THIS WIDGET IS FOR, beyond holding a list of paths:

    THE ANSWER HAS TO ARRIVE BEFORE THE USER COMMITS.

Point 4 of the instruction is that the column set a merge produces IS the
analysis about to be run, and finding out afterwards that half the
measurements were dropped is finding out too late. So every time the set
changes this asks :func:`spacr.multi_database.describe_merge` -- which reads
only sqlite metadata and the distinct plate ids, and is therefore cheap enough
to run while files are still being chosen -- and puts the answer on screen:
how many rows, how many columns are common, which measurements are in only
some, and whether two databases claim the same plate.

A COLLISION IS SHOWN, NOT RESOLVED. Two databases that each hold a ``plate1``
are two experiments, and the widget says so and names them. It deliberately
does NOT offer ``on_collision='qualify'``: rewriting ``plate1`` to
``runA-plate1`` makes the keys unique and hides which experiment a plate came
from inside its own id, where nothing can block on it, test for it or colour
by it. The resolutions offered are the ones that keep the experiment
analysable -- remove one of the databases, or rename the plates.
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
    QWidget,
)

from ..theme import SPACING
from .table_chip import TableChip

__all__ = ["DatabaseSetWidget", "database_for_source"]

#: What the Image UMAP and every other ``src``-taking module means by a source
#: root: the plate folder Measure wrote into.
#:
#: The same join :func:`spacr.utils.get_db_paths` performs, spelled out here
#: rather than imported: ``spacr.utils`` pulls torch, and a settings panel must
#: not pay seconds of import to redraw a summary line when a chip is added.
#: ``test_the_folder_join_matches_get_db_paths`` is what keeps the two equal.
MEASUREMENTS_SUFFIX = ("measurements", "measurements.db")


def database_for_source(source: str, mode: str = "database") -> str:
    """The database a chosen source names.

    :param source: what the user picked -- a database file in ``'database'``
        mode, a spaCR project/plate folder in ``'folder'`` mode.
    :param mode: ``'database'`` or ``'folder'``.
    :returns: the path to open. Nothing is checked for existence; a source
        whose database is missing is reported by the summary rather than
        removed from the set behind the user's back.
    """
    if mode == "folder":
        return os.path.join(str(source).rstrip(os.sep), *MEASUREMENTS_SUFFIX)
    return str(source)


class DatabaseSetWidget(QWidget):
    """Add, remove and see the databases a screen will merge.

    :param value: the initial sources. A bare string is accepted because
        every module's ``src`` has historically been one, and a settings CSV
        written before this widget existed still holds one.
    :param mode: ``'database'`` (the user picks .db files -- the Gate Editor)
        or ``'folder'`` (the user picks plate folders -- Image UMAP and every
        other module whose ``src`` is a project root).
    :param table: the table the merge is described on. ``'cell'`` is the
        anchor every object join is built on.
    :param min_items: how many sources may never be removed. The Gate Editor
        passes 1, because a gate editor with no table is a screen with
        nothing on it; a settings panel passes 0, because clearing the field
        is a legitimate thing to want.
    :param on_colour_by: called with :data:`spacr.multi_database.SOURCE_COLUMN`
        or ``None`` when the "colour by source" box is toggled. Given by a
        settings panel that owns a ``color_by`` field; omitted elsewhere, and
        the box is then not shown.
    """

    value_changed = Signal()

    def __init__(self, value=None, parent=None, *, mode: str = "database",
                 table: str = "cell", title: Optional[str] = None,
                 min_items: int = 0,
                 on_colour_by: Optional[Callable[[Optional[str]], None]] = None):
        super().__init__(parent)
        self.setObjectName("DatabaseSetWidget")
        self._mode = "folder" if mode == "folder" else "database"
        self._table = table
        self._min_items = max(0, int(min_items))
        self._on_colour_by = on_colour_by
        self._sources: List[str] = []
        self._plan = None
        self._title = title or (
            "Choose one or more spaCR project folders"
            if self._mode == "folder" else
            "Choose one or more measurement databases")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACING["xs"])

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["xs"])
        self.add_button = QPushButton(
            "Add project folders…" if self._mode == "folder"
            else "Add databases…", self)
        self.add_button.setObjectName("DatabaseSetAdd")
        self.add_button.setToolTip(
            "Adds to the set rather than replacing it, so three plates can "
            "be gathered in three trips. Every row of the merged frame "
            "carries the database it came from, so the map can be coloured "
            "by source.")
        self.add_button.clicked.connect(self.choose_sources)
        row.addWidget(self.add_button)
        row.addStretch(1)
        outer.addLayout(row)

        self._chips = QHBoxLayout()
        self._chips.setContentsMargins(0, 0, 0, 0)
        self._chips.setSpacing(SPACING["xs"])
        self._chips.addStretch(1)
        outer.addLayout(self._chips)

        #: What the merge WOULD cost, before it is performed.
        self.summary = QLabel("", self)
        self.summary.setObjectName("DatabaseSetSummary")
        self.summary.setWordWrap(True)
        self.summary.setProperty("role", "hint")
        outer.addWidget(self.summary)

        self.colour_by_source = QCheckBox(
            "Colour the map by source database", self)
        self.colour_by_source.setObjectName("DatabaseSetColourBySource")
        self.colour_by_source.setToolTip(
            "A merged embedding whose clusters turn out to be the plates "
            "rather than biology is the most important thing a multi-database "
            "map can show — and it can only show it if the points are "
            "coloured by where they came from.")
        self.colour_by_source.toggled.connect(self._on_colour_toggled)
        self.colour_by_source.setVisible(False)
        outer.addWidget(self.colour_by_source)

        self.set_value(value)

    # -- the value ---------------------------------------------------------
    def get_value(self):
        """The chosen sources, as a list -- the shape every consumer takes.

        Always a list, including for one source: ``generate_image_umap``
        wraps a bare string in a list on the first line it touches ``src``,
        so handing it the list it is going to build anyway removes the case
        where one database is a different code path from three.
        """
        return list(self._sources)

    def set_value(self, value) -> None:
        """Replace the set. Accepts a list, a bare string, or ``None``."""
        self._sources = self._clean(value)
        self._rebuild()

    def sources(self) -> List[str]:
        """The chosen sources."""
        return list(self._sources)

    def database_paths(self) -> List[str]:
        """The database each source names -- what a merge would open."""
        return [database_for_source(source, self._mode)
                for source in self._sources]

    def plan(self):
        """The last :class:`spacr.multi_database.MergePlan`, or ``None``."""
        return self._plan

    # -- editing -----------------------------------------------------------
    def choose_sources(self) -> None:
        """Open the picker and ADD what comes back."""
        if self._mode == "folder":
            folder = QFileDialog.getExistingDirectory(self, self._title, "")
            chosen = [folder] if folder else []
        else:
            chosen, _ = QFileDialog.getOpenFileNames(
                self, self._title, "",
                "Measurements (*.db *.sqlite *.sqlite3);;All files (*)")
        self.add_sources(chosen)

    def add_sources(self, paths: Sequence[str]) -> int:
        """Add sources, ignoring the ones already in the set.

        :returns: how many were actually added.
        """
        added = 0
        for path in paths or []:
            text = str(path).strip()
            if text and text not in self._sources:
                self._sources.append(text)
                added += 1
        if added:
            self._rebuild()
            self.value_changed.emit()
        return added

    def remove_source(self, name: str) -> bool:
        """Drop one member of the set, by its chip's label or its path.

        The chip carries the LABEL -- ``plate1``, not
        ``/data/plate1/measurements/measurements.db`` -- because that is what
        a legend and the provenance column say. Both are accepted so a caller
        with the path does not have to work out the label first.
        """
        target = None
        if name in self._sources:
            target = name
        else:
            for source, label in zip(self._sources, self._labels()):
                if label == name:
                    target = source
                    break
        if target is None or len(self._sources) <= self._min_items:
            return False
        self._sources.remove(target)
        self._rebuild()
        self.value_changed.emit()
        return True

    def clear(self) -> None:
        """Empty the set."""
        if not self._sources:
            return
        self._sources = []
        self._rebuild()
        self.value_changed.emit()

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _clean(value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            # 'path' is what spacr.settings ships as the "not chosen yet"
            # placeholder for src. Rendering it as a chip would offer to merge
            # a database called path.
            return [] if text in ("", "path", "/path", "/path/to/src") else [text]
        out: List[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in out:
                out.append(text)
        return out

    def _labels(self) -> List[str]:
        """The name each source will carry in the provenance column.

        Asked of :mod:`spacr.multi_database` rather than computed here, so a
        chip and the ``source_database`` value it stands for cannot disagree.
        """
        from ...multi_database import source_labels

        paths = self.database_paths()
        if not paths:
            return []
        try:
            return list(source_labels(paths))
        except Exception:
            return [os.path.basename(str(p).rstrip(os.sep)) or str(p)
                    for p in self._sources]

    def _rebuild(self) -> None:
        self._rebuild_chips()
        self._refresh_summary()
        self.colour_by_source.setVisible(
            self._on_colour_by is not None and len(self._sources) > 1)

    def _rebuild_chips(self) -> None:
        while self._chips.count() > 1:
            item = self._chips.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        removable = len(self._sources) > self._min_items
        for index, (source, label) in enumerate(
                zip(self._sources, self._labels())):
            chip = TableChip(label, self, removable=removable)
            chip.setToolTip(source)
            chip.removed.connect(self.remove_source)
            self._chips.insertWidget(index, chip)

    def _refresh_summary(self) -> None:
        self._plan = None
        if not self._sources:
            self.summary.setText("")
            return
        from ...multi_database import describe_merge

        paths = self.database_paths()
        missing = [path for path in paths if not os.path.isfile(path)]
        if missing:
            # Named, not swallowed. In folder mode the user picked a plate
            # folder and the database is two levels below it, so "nothing
            # happened" would be indistinguishable from "that plate was never
            # measured".
            self.summary.setText(
                f"{len(missing)} of {len(paths)} sources have no "
                f"measurements database yet: "
                + ", ".join(os.path.dirname(p) for p in missing[:3])
                + (" …" if len(missing) > 3 else ""))
            return
        try:
            plan = describe_merge(paths, self._table)
        except Exception as exc:
            self.summary.setText(
                f"could not read {len(paths)} database(s) as "
                f"'{self._table}': {exc}")
            return
        self._plan = plan
        self.summary.setText(self._summary_text(plan))

    def _summary_text(self, plan) -> str:
        """What the merge would cost, in the order a user needs it."""
        lines = [
            f"{len(plan.sources)} database(s) · {plan.total_rows:,} rows · "
            f"{len(plan.common_columns)} columns in all of them"
        ]
        dropped = plan.dropped_columns
        if dropped:
            shown = ", ".join(dropped[:6]) + (" …" if len(dropped) > 6 else "")
            lines.append(
                f"{len(dropped)} measurement(s) are in only some of them and "
                f"would be dropped: {shown}")
        if plan.colliding_plates:
            detail = "; ".join(
                f"{plate} in {', '.join(labels)}"
                for plate, labels in sorted(plan.colliding_plates.items()))
            lines.append(
                f"THE SAME PLATE ID IS IN MORE THAN ONE DATABASE: {detail}. "
                "Merging would compute every per-well number over two "
                "experiments at once. Remove one of them, or rename the "
                "plates.")
        return "\n".join(lines)

    def _on_colour_toggled(self, on: bool) -> None:
        if self._on_colour_by is None:
            return
        from ...multi_database import SOURCE_COLUMN

        self._on_colour_by(SOURCE_COLUMN if on else None)
