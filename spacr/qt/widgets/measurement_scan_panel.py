"""Which measurement has genes with clear effect sizes.

Instruction 122 part 3, as asked for:

    "doing a sweep on these screen data of which measurements have genes with
     an effect size. so instead of a parameter search a search for which
     measurement has genes with clear effect sizes (one or several)"

The pure logic is :mod:`spacr.measurement_scan`, which had no caller. This is
the thin renderer over it, and it lives beside the sweep's runs for the reason
the instruction gives: structurally this IS the parameter search with a
different thing varying -- the settings are held fixed and the DEPENDENT
VARIABLE is swept.

WHAT THIS PANEL EXISTS TO SHOW, and the reason a plain table would be wrong:

    A MEASUREMENT SCAN IS A MULTIPLE-TESTING PROBLEM ACROSS MEASUREMENTS.

spaCR measures hundreds of features per object. Scan 500 for "genes with a
clear effect" and some look clear by chance, and they look exactly as
convincing as the real ones -- because the per-measurement FDR was computed
WITHIN each measurement and knows nothing about the other 499. So both
numbers are on every row, and the across-scan one is what the verdict column
reads. A panel that showed only the within-run q-value would have rebuilt the
exact trap the module exists to close.

Measured on plate1 of the tsg101 screen with the gene labels permuted, so no
effect can exist: the within-run correction fired on 83.5% of those scans and
the across-scan correction on 5.0%. That gap is the feature.

Ranked by EFFECT SIZE, not by p-value: with two screens' worth of wells a
trivial effect is significant, and "clear effect sizes" is what was asked for.

THE MEASUREMENT DATABASES, AND WHY THEY LIVE HERE TOO
-----------------------------------------------------
Instruction 130 sections B and C. A regression row is one PLATE, and a plate
can now carry its measurements database beside its score and count CSVs. Those
databases surface in this tab, with the join offered:
:class:`DatabaseMergePanel` lists every attached database, its object tables
and its plates, and merges the chosen ones onto one anchor.

NOTHING ABOUT THAT MERGE IS DECIDED HERE. :mod:`spacr.multi_database` already
merges several databases without pooling two plates that share a name;
:mod:`spacr.merge_tables` already holds the per-measurement aggregation --
areas SUM, perimeters MEAN, a minimum takes MIN, a label takes FIRST -- and
``MergePolicy.how_for`` already decides the join PER TABLE from object
cardinality. :func:`merge_across_databases` is the composition of those two and
adds no arithmetic of its own, which is why there is no blanket "join type"
control anywhere in this file: one ``how`` for every table is exactly the thing
instruction 77 found wrong.

And it SAYS what the merge cost, because a merge that silently changed how a
measurement was combined produces a number that is wrong and looks fine.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping as _Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel,
    QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...merge_tables import (AGGREGATION_RULES, DEFAULT_AGGREGATION,
                             DEFAULT_PRIMARY, IDENTITY, OBJECT_COLUMN,
                             OBJECT_TABLES, MergeError, MergePolicy,
                             _align_keys, _apply_na_policy, aggregation_plan,
                             mergeable_tables, roll_up)
from ...multi_database import (SCREEN_COLUMN, SOURCE_COLUMN, MergeRefused,
                               describe_merge, read_merged)
from ...object_roles import ONE_ROW_PER_CELL, anchor_column, is_one_row_per_cell

#: Columns worth reading first, in this order. Effect size leads because it is
#: the primary sort and the thing that was asked for; the two corrections sit
#: beside each other so the gap between them is visible without scrolling.
PREFERRED_COLUMNS = (
    "measurement", "effect_size", "top_gene", "across_scan_q", "within_run_q",
    "verdict", "coefficient", "p_value", "within_run_hits", "n_wells",
    "n_genes", "measurement_p",
)

#: What a row's two corrections mean together, in words. The middle one is the
#: single most useful thing this feature can say and the easiest to hide.
VERDICT_SURVIVES = "clear effect"
VERDICT_WITHIN_ONLY = "would pass alone — not across the scan"
VERDICT_NEITHER = "no effect"


def verdict_for(row) -> str:
    """One phrase per measurement, from BOTH corrections."""
    if getattr(row, "survives_across_scan", False):
        return VERDICT_SURVIVES
    if getattr(row, "survives_within_run", False):
        return VERDICT_WITHIN_ONLY
    return VERDICT_NEITHER


def ordered_columns(frame) -> list:
    """:data:`PREFERRED_COLUMNS` this frame has, then everything else.

    Ordering, not filtering -- the columns nobody thought to list are still
    the user's own numbers.
    """
    if frame is None:
        return []
    have = list(frame.columns)
    first = [name for name in PREFERRED_COLUMNS if name in have]
    return first + [name for name in have if name not in first]


# --------------------------------------------------------------------------- #
#  Instruction 130 B: the databases attached to the input table
# --------------------------------------------------------------------------- #

#: What a merge anchors on unless the user says otherwise, from
#: :data:`spacr.merge_tables.DEFAULT_PRIMARY`. Named rather than typed so this
#: panel cannot drift from the module that performs the merge.
DEFAULT_ANCHOR = DEFAULT_PRIMARY


@dataclass(frozen=True)
class AttachedDatabase:
    """One plate of the regression input table, seen from this tab.

    :param plate: the plate the input-table row names.
    :param path: its measurements database, or ``""``. **A plate with no
        database is legal** -- the regression runs on the score and count CSVs,
        and the database is only what makes this tab possible for that plate --
        so an empty path is listed and disabled rather than refused.
    :param screen: the screen the plate belongs to, when the project has more
        than one. Carried through to :func:`spacr.multi_database.read_merged`
        as ``screens=``, which is what keeps two screens sharing ``plate1``
        apart as two identities instead of one collision.
    """

    plate: str
    path: str = ""
    screen: Optional[str] = None

    @property
    def attached(self) -> bool:
        """Whether this plate has a database at all."""
        return bool(str(self.path).strip())

    @property
    def present(self) -> bool:
        """Whether the attached database is on disk right now.

        Checked here rather than at run time: instruction 130 asks that a row
        whose database has gone missing says so BEFORE the run, not four
        minutes into it.
        """
        return self.attached and os.path.exists(str(self.path))

    @property
    def label(self) -> str:
        """A readable name for the file, disambiguated by its folder.

        Every plate's database is usually called ``measurements.db``, so the
        stem alone names all of them the same thing -- the reason
        :func:`spacr.multi_database.describe_merge` folds the parent directory
        into its own labels.
        """
        if not self.attached:
            return ""
        path = str(self.path)
        parent = os.path.basename(os.path.dirname(path))
        return f"{parent}/{os.path.basename(path)}" if parent \
            else os.path.basename(path)

    @property
    def status(self) -> str:
        """Why this plate is or is not in the merge, in words."""
        if not self.attached:
            return "no database — this plate is not in the merge"
        if not self.present:
            return "missing from disk — attach it again or remove the row"
        return "ready"


def attached_databases(rows: Any) -> Tuple[AttachedDatabase, ...]:
    """The input table's rows as :class:`AttachedDatabase` entries.

    :param rows: what the host's database provider returned. The shape the
        input table emits is a list of ``{"plate", "score", "count",
        "database"}`` dicts; a ``(plate, path)`` pair, a ``(plate, path,
        screen)`` triple and a bare path are accepted too, so a caller with a
        plainer list does not have to build dicts to be understood.
    :returns: one entry per row, IN THE ROW ORDER, including the rows with no
        database -- they are the plates this tab has to disable rather than
        drop, and dropping them here would make them invisible instead.
    """
    if rows is None:
        return ()
    out: List[AttachedDatabase] = []
    for index, row in enumerate(rows):
        if isinstance(row, AttachedDatabase):
            out.append(row)
            continue
        if isinstance(row, _Mapping):
            plate = row.get("plate", row.get("plateID", ""))
            path = row.get("database", row.get("db", ""))
            screen = row.get("screen", row.get(SCREEN_COLUMN))
        elif isinstance(row, (str, os.PathLike)):
            plate, path, screen = "", row, None
        else:
            values = list(row)
            plate = values[0] if values else ""
            path = values[1] if len(values) > 1 else ""
            screen = values[2] if len(values) > 2 else None
        label = str(plate or "").strip() or f"row {index + 1}"
        screen = str(screen).strip() if screen not in (None, "") else None
        out.append(AttachedDatabase(plate=label,
                                    path=str(path or "").strip(),
                                    screen=screen or None))
    return tuple(out)


def joinable_tables(paths: Sequence[str]) -> Tuple[str, ...]:
    """The object tables EVERY one of these databases has, in table order.

    The intersection, not the union, and this is not a nicety:
    :func:`spacr.multi_database.describe_merge` reads a row count from every
    path, so asking it for a table one database lacks raises a bare
    ``sqlite3.OperationalError`` naming the table and nothing else. Offering
    only what all of them have means the user never picks that.

    :param paths: measurement databases.
    :returns: names from :data:`spacr.merge_tables.OBJECT_TABLES` -- which is
        the object-role registry, cell/nucleus/pathogen/cytoplasm and every
        organelle slot, rather than four names typed here.
    :raises sqlite3.Error: a path that is not a readable database.
    """
    shared: Optional[set] = None
    for path in paths:
        present = set(mergeable_tables(str(path))) & set(OBJECT_TABLES)
        shared = present if shared is None else (shared & present)
    if not shared:
        return ()
    return tuple(name for name in OBJECT_TABLES if name in shared)


def anchor_tables(tables: Sequence[str]) -> Tuple[str, ...]:
    """The subset of ``tables`` that can be an anchor.

    One row per cell, from :data:`spacr.object_roles.ONE_ROW_PER_CELL`.
    Anchoring on a many-per-cell table would make a row of the merged frame
    mean one nucleus or one pathogen, with the cell's own measurements
    repeated across its children -- which is the fan-out the roll-up exists to
    prevent, arrived at from the other side.
    """
    return tuple(name for name in tables if is_one_row_per_cell(name))


def default_aggregation_columns(columns: Sequence[str], *,
                                overrides: Optional[Dict[str, str]] = None
                                ) -> Tuple[str, ...]:
    """The columns NO :data:`~spacr.merge_tables.AGGREGATION_RULES` rule names.

    Instruction 130 section C, third bullet: a measurement nobody thought
    about is exactly the one worth naming. These fall through to
    :data:`~spacr.merge_tables.DEFAULT_AGGREGATION`, which is MEAN -- right
    more often than not for an unrecognised number, and silently wrong for a
    total.

    Computed by re-walking the rule table, so it cannot drift from it: a list
    written here would go stale the first time a rule was added.

    :param columns: the column names about to be aggregated.
    :param overrides: the user's explicit choices, which win over every rule
        and are therefore not fall-throughs.
    """
    chosen = dict(overrides or {})
    out = []
    for column in columns:
        if column in chosen:
            continue
        name = str(column).lower()
        if any(re.search(pattern, name) for pattern, _how in AGGREGATION_RULES):
            continue
        out.append(str(column))
    return tuple(out)


def merge_across_databases(paths: Sequence[str], tables: Sequence[str], *,
                           policy: Optional[MergePolicy] = None,
                           screens: Any = None,
                           columns: str = "common",
                           report=None,
                           limit_per_source: Optional[int] = None):
    """Every chosen table of every chosen database, on one anchor.

    THE COMPOSITION OF THE TWO MERGES THAT ALREADY EXIST, and deliberately
    nothing else. :func:`spacr.multi_database.read_merged` is *many databases,
    one table*; :func:`spacr.merge_tables.merge_tables` is *one database, many
    tables* and takes a path, so it cannot be handed a frame that already spans
    databases. This runs the first per chosen table and then joins them with
    the second's own :func:`~spacr.merge_tables.roll_up` and
    :meth:`~spacr.merge_tables.MergePolicy.how_for`. There is no sum and no
    mean written here; every number comes from the rules.

    THE ROLL-UP KEYS CARRY THE SCREEN AND THE SOURCE. Omit them and two
    screens legitimately sharing ``plate1`` -- the case
    :func:`~spacr.multi_database.describe_merge` deliberately permits -- would
    collapse into one parent, reintroducing one layer up the exact pooling
    :mod:`spacr.multi_database` exists to prevent.

    :param paths: measurement databases. Repeats are read once.
    :param tables: the object tables to join. The anchor is added if absent.
    :param policy: how each measurement combines and what happens to a cell
        with no children. ``policy.primary`` IS the anchor and defaults to
        :data:`DEFAULT_ANCHOR`.
    :param screens: screen label per database -- a sequence parallel to
        ``paths`` or a mapping from path. Passed to both the plan and the read.
    :param columns: ``'common'`` (default) or ``'union'``, as ``read_merged``.
    :param report: called with one line per thing the merge cost.
    :param limit_per_source: row cap per database, for a preview.
    :returns: one row per anchor object, with ``frame.attrs`` carrying what the
        merge cost -- see :func:`merge_report`, which renders it.
    :raises MergeError: the anchor is not one row per cell, or carries no
        object label.
    :raises spacr.multi_database.MergeRefused: a plate id appears twice within
        one screen.
    """
    policy = policy or MergePolicy(primary=DEFAULT_ANCHOR)
    anchor = str(policy.primary)
    if not is_one_row_per_cell(anchor):
        raise MergeError(
            f"{anchor!r} is many rows per cell, so anchoring on it would make "
            f"a row mean one {anchor} and repeat the cell's own measurements "
            f"across its children; anchor on one of "
            f"{list(ONE_ROW_PER_CELL)}")

    paths = list(dict.fromkeys(str(path) for path in paths))
    wanted = list(dict.fromkeys([anchor] + [str(name) for name in tables]))

    plan = describe_merge(paths, anchor, screens=screens)
    base = read_merged(paths, anchor, plan=plan, columns=columns,
                       screens=screens, report=report,
                       limit_per_source=limit_per_source)
    if OBJECT_COLUMN not in base.columns:
        raise MergeError(
            f"the {anchor} table has no {OBJECT_COLUMN}, so nothing can be "
            f"merged onto it")

    keys = [name for name in IDENTITY if name in base.columns]
    # WHAT EVERY JOIN IS KEYED ON: the well identity, the screen and the file.
    # `source_database` is in here as well as `screenID` because two databases
    # of one screen are still two files, and a cell in one of them is not the
    # same cell as the identically numbered cell in the other.
    carried = [name for name in [*keys, SCREEN_COLUMN, SOURCE_COLUMN]
               if name in base.columns]
    rows_before = _rows_per_source(base)

    # The anchor's own measurements carry its name, exactly as `merge_tables`
    # prefixes its primary -- so `area` from cell and `area` from nucleus can
    # be told apart in the axis picker. A column that ALREADY starts with the
    # table's name is left alone: `cell_area` must not become `cell_cell_area`.
    reserved = set(carried) | {OBJECT_COLUMN}
    base = base.rename(columns={
        name: f"{anchor}_{name}" for name in base.columns
        if name not in reserved and not str(name).startswith(f"{anchor}_")})

    joins: List[Dict[str, Any]] = []
    skipped: Dict[str, str] = {}
    fell_through: Dict[str, Tuple[str, ...]] = {}
    dropped: Dict[str, Tuple[str, ...]] = {
        anchor: tuple(plan.dropped_columns) if columns == "common" else ()}

    for table in wanted:
        if table == anchor:
            continue
        child_plan = describe_merge(paths, table, screens=screens)
        child = read_merged(paths, table, plan=child_plan, columns=columns,
                            screens=screens, report=report,
                            limit_per_source=limit_per_source)
        dropped[table] = (tuple(child_plan.dropped_columns)
                          if columns == "common" else ())
        link = anchor_column(table)
        if link not in child.columns:
            # Measured without a parent mask: the roll-up is not empty, it is
            # UNDEFINED. Named and skipped, as merge_tables does -- one
            # unlinkable table must not cost the user the others.
            skipped[table] = (
                f"carries no {link}, so its rows cannot be matched to a "
                f"{anchor}; re-run Measure with the {anchor} mask set")
            continue

        child_keys = [name for name in carried if name in child.columns] + [link]
        if is_one_row_per_cell(table):
            # One row per cell already: nothing to aggregate, and putting it
            # through the roll-up rules would answer a question nobody asked.
            rolled = child.rename(columns={
                name: (name if str(name).startswith(f"{table}_")
                       else f"{table}_{name}")
                for name in child.columns if name not in set(child_keys)})
        else:
            plan_for_table = aggregation_plan(child, overrides=policy.overrides,
                                              skip=child_keys)
            numeric = [name for name in plan_for_table
                       if plan_for_table[name] == DEFAULT_AGGREGATION]
            fell_through[table] = default_aggregation_columns(
                numeric, overrides=policy.overrides)
            rolled = roll_up(child, child_keys, name=table, policy=policy)
        if link != OBJECT_COLUMN:
            rolled = rolled.rename(columns={link: OBJECT_COLUMN})

        # The object key is in both by construction: the anchor was checked for
        # it above, and a child that does not carry it was skipped as
        # unlinkable a few lines up.
        on = [name for name in carried + [OBJECT_COLUMN]
              if name in rolled.columns and name in base.columns]
        _align_keys(base, rolled, on)
        # PER TABLE, FROM CARDINALITY -- never one blanket `how`. A cell with
        # no nucleus is not a cell; a cell with no pathogen is an uninfected
        # cell and usually the control population.
        how = policy.how_for(table)
        before = len(base)
        base = base.merge(rolled, on=on, how=how)
        joins.append({"table": table, "how": how, "before": before,
                      "after": len(base)})

    base = _apply_na_policy(base, policy)
    base.attrs["anchor"] = anchor
    base.attrs["tables"] = tuple(wanted)
    base.attrs["joins"] = tuple(joins)
    base.attrs["skipped_tables"] = dict(skipped)
    base.attrs["rows_before"] = rows_before
    base.attrs["rows_after"] = _rows_per_source(base)
    base.attrs["default_aggregation"] = {name: values
                                         for name, values in fell_through.items()
                                         if values}
    base.attrs["dropped_columns"] = {name: values
                                     for name, values in dropped.items()
                                     if values}
    base.attrs["screens"] = plan.screens
    base.attrs["shared_plates_across_screens"] = dict(
        plan.shared_plates_across_screens)
    base.attrs["sources"] = tuple((source.label, source.path)
                                  for source in plan.sources)
    return base


def _rows_per_source(frame) -> Dict[str, int]:
    """How many rows each database has in ``frame`` right now.

    ``read_merged`` writes :data:`~spacr.multi_database.SOURCE_COLUMN` into
    every frame it returns, so this is always answerable.
    """
    counted = frame[SOURCE_COLUMN].value_counts()
    return {str(label): int(count) for label, count in counted.items()}


def merge_report(frame) -> str:
    """What the merge in ``frame.attrs`` cost, in the order 130 C asks for.

    The anchor and the row count it produced; what each source contributed and
    what was dropped; every column that fell through to the default
    aggregation; and every plate id that appeared in more than one database
    with what was done about it.

    :param frame: the output of :func:`merge_across_databases`.
    """
    attrs = getattr(frame, "attrs", {}) or {}
    anchor = attrs.get("anchor", DEFAULT_ANCHOR)
    lines = [f"Merged {len(frame):,} rows on {anchor}, "
             f"{len(frame.columns)} columns."]

    before = attrs.get("rows_before") or {}
    after = attrs.get("rows_after") or {}
    for label in before:
        kept = after.get(label, 0)
        lost = before[label] - kept
        lines.append(
            f"  {label}: {kept:,} of {before[label]:,} {anchor} rows"
            + (f" — {lost:,} dropped by the joins below" if lost else ""))

    for join in attrs.get("joins", ()):
        removed = join["before"] - join["after"]
        lines.append(
            f"  {join['table']}: {join['how']} join"
            + (f", removed {removed:,} of {join['before']:,} rows"
               if removed > 0 else
               (f", added {-removed:,} rows" if removed < 0 else
                ", removed nothing")))
    for table, why in (attrs.get("skipped_tables") or {}).items():
        lines.append(f"  {table}: left out — {why}")

    fell_through = attrs.get("default_aggregation") or {}
    if fell_through:
        for table, names in fell_through.items():
            lines.append(
                f"  {table}: {len(names)} column(s) matched no aggregation "
                f"rule and were combined with the default "
                f"({DEFAULT_AGGREGATION}) — " + ", ".join(names))
    else:
        lines.append("  Every aggregated column matched a rule; none fell "
                     f"through to the default ({DEFAULT_AGGREGATION}).")

    dropped = attrs.get("dropped_columns") or {}
    for table, names in dropped.items():
        lines.append(
            f"  {table}: {len(names)} measurement(s) present in only some "
            f"databases were dropped — " + ", ".join(names))

    shared = attrs.get("shared_plates_across_screens") or {}
    for plate, screens in shared.items():
        lines.append(
            f"  plate {plate} appears in screens {', '.join(screens)}: kept "
            f"apart by {SCREEN_COLUMN}, not renamed — a qualified plate id "
            f"hides the screen inside the plate name.")
    return "\n".join(lines)


class DatabaseMergePanel(QWidget):
    """The databases attached to the input table, and the join offered.

    Instruction 130 section B. One row per plate of the regression input
    table, whether or not it has a database -- a plate with none is listed and
    disabled here, because it still runs in the regression and the user needs
    to see why it is absent from this tab.

    WHAT IS NOT OFFERED IS AS DELIBERATE AS WHAT IS. There is no join-type
    control: the join follows object cardinality per table through
    :meth:`spacr.merge_tables.MergePolicy.how_for`, and a blanket ``how`` is
    the finding instruction 77 raised. The two checkboxes here are the two
    settings that policy actually reads.

    :ivar databases_changed: emitted with the number of readable databases
        whenever the list is re-read.
    :ivar merged: emitted with the merged frame.
    """

    databases_changed = Signal(int)
    merged = Signal(object)

    #: The list columns, in reading order.
    COLUMNS = ("Plate", "Database", "Screen", "Tables", "Plates in it",
               "Rows", "Status")

    def __init__(self, database_provider=None, parent=None):
        """
        :param database_provider: called with no arguments for the input
            table's rows. A callable rather than a stored list, for the same
            reason ``frame_provider`` is one: the tab must not go on showing
            the previous run's inputs.
        """
        super().__init__(parent)
        self._provider = database_provider
        self._databases: Tuple[AttachedDatabase, ...] = ()
        self._tables: Tuple[str, ...] = ()
        self._frame = None
        self._overrides: Dict[str, str] = {}
        self._rules_dialog = None
        self._filling = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.heading = QLabel("No measurement database attached yet.")
        self.heading.setWordWrap(True)
        layout.addWidget(self.heading)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.table.setMaximumHeight(170)
        layout.addWidget(self.table)

        chooser = QHBoxLayout()
        chooser.addWidget(QLabel("join"))
        self.tables_list = QListWidget()
        self.tables_list.setFlow(QListWidget.LeftToRight)
        self.tables_list.setWrapping(True)
        self.tables_list.setMaximumHeight(72)
        self.tables_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.tables_list.setToolTip(
            "The object tables every attached database has. A table only one "
            "database has is not offered — merging it would fail on the "
            "others rather than on this list.")
        self.tables_list.itemChanged.connect(self._on_choice)
        chooser.addWidget(self.tables_list, 1)
        layout.addLayout(chooser)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("onto"))
        self.anchor_box = QComboBox()
        self.anchor_box.setToolTip(
            f"The anchor: one row of the merged table is one of these. "
            f"{DEFAULT_ANCHOR} by default, and only tables with one row per "
            f"cell can be one — anchoring on a many-per-cell table repeats "
            f"the cell's own measurements across its children.")
        self.anchor_box.currentIndexChanged.connect(self._on_choice)
        controls.addWidget(self.anchor_box)
        self.anchor_note = QLabel(
            f"default {DEFAULT_ANCHOR} — one anchor, one copy of each column")
        self.anchor_note.setWordWrap(True)
        controls.addWidget(self.anchor_note, 1)
        layout.addLayout(controls)

        options = QHBoxLayout()
        self.consolidate = QCheckBox("only cells that have the child object")
        self.consolidate.setChecked(True)
        self.consolidate.setToolTip(
            "consolidate_on_cell. Off keeps every cell and leaves the child's "
            "columns empty. This is not a blanket join type: the join is "
            "decided per table by what that object IS.")
        self.consolidate.toggled.connect(self._on_choice)
        options.addWidget(self.consolidate)
        self.keep_uninfected = QCheckBox("keep uninfected cells")
        self.keep_uninfected.setChecked(True)
        self.keep_uninfected.setToolTip(
            "An uninfected cell is a cell, and in a screen it is usually the "
            "control population. Off restricts the analysis to cells that "
            "contain a pathogen or organelle.")
        self.keep_uninfected.toggled.connect(self._on_choice)
        options.addWidget(self.keep_uninfected)
        options.addStretch(1)
        self.rules_button = QPushButton("Aggregation rules…")
        self.rules_button.setToolTip(
            "How each measurement combines when several children roll up onto "
            "one cell, and a dropdown to change any of it.")
        self.rules_button.clicked.connect(self.show_aggregation_rules)
        options.addWidget(self.rules_button)
        self.merge_button = QPushButton("Merge")
        self.merge_button.clicked.connect(self.merge)
        options.addWidget(self.merge_button)
        layout.addLayout(options)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(190)
        layout.addWidget(self.report, 1)

        self.refresh()

    # ------------------------------------------------------------- the list

    def set_database_provider(self, provider) -> None:
        """Take a new source of input-table rows and re-read it."""
        self._provider = provider
        self.refresh()

    @property
    def databases(self) -> Tuple[AttachedDatabase, ...]:
        """Every plate row, attached or not, in the input table's order."""
        return self._databases

    def paths(self) -> Tuple[str, ...]:
        """The databases that are attached AND on disk, de-duplicated."""
        return tuple(dict.fromkeys(
            entry.path for entry in self._databases if entry.present))

    def screens(self) -> Optional[Dict[str, str]]:
        """``{path: screen}`` for the rows that named one, else ``None``.

        ``None`` rather than a dict of defaults: naming a screen for every
        database says the user is working in screens, and
        :attr:`~spacr.multi_database.MergePlan.screens_were_named` reads that
        to decide how a refusal is worded.
        """
        named = {entry.path: entry.screen for entry in self._databases
                 if entry.present and entry.screen}
        return named or None

    def refresh(self) -> int:
        """Re-read the provider and describe what is attached.

        :returns: the number of readable databases.
        """
        rows = None
        if callable(self._provider):
            try:
                rows = self._provider()
            except Exception as error:  # noqa: BLE001 - report, do not raise
                self._databases = ()
                self._fill_table()
                self.report.setPlainText(
                    f"Could not read the input table: {error}")
                self.heading.setText("No measurement database attached.")
                self.databases_changed.emit(0)
                return 0
        self._databases = attached_databases(rows)
        self._fill_table()
        self._offer_tables()
        self.describe()
        count = len(self.paths())
        self.databases_changed.emit(count)
        return count

    def _fill_table(self) -> None:
        entries = self._databases
        self.table.setRowCount(len(entries))
        attached = sum(1 for entry in entries if entry.present)
        missing = [entry.plate for entry in entries
                   if entry.attached and not entry.present]
        empty = [entry.plate for entry in entries if not entry.attached]
        text = (f"{attached} measurement database(s) attached to "
                f"{len(entries)} plate row(s).")
        if missing:
            text += (f" {len(missing)} named a database that is not on disk: "
                     + ", ".join(missing) + ".")
        if empty:
            text += (f" {len(empty)} plate(s) have none and are disabled here "
                     f"— they still run in the regression: "
                     + ", ".join(empty) + ".")
        self.heading.setText(text if entries else
                             "No measurement database attached yet. Drop a "
                             "database onto a plate row of the input table.")

        info = self._source_info()
        for row, entry in enumerate(entries):
            detail = info.get(entry.path, {})
            for column, value in enumerate(
                    [entry.plate, detail.get("label", entry.label),
                     entry.screen or "", detail.get("tables", ""),
                     detail.get("plates", ""), detail.get("rows", ""),
                     detail.get("status", entry.status)]):
                item = QTableWidgetItem(str(value))
                if not entry.present:
                    # Disabled, not removed: the user has to be able to see
                    # which plate is missing from this tab and why.
                    item.setFlags(Qt.ItemIsSelectable)
                self.table.setItem(row, column, item)

    def _source_info(self) -> Dict[str, Dict[str, str]]:
        """Each attached database's tables, plates and anchor row count.

        The NAME shown is the one :func:`~spacr.multi_database.describe_merge`
        gives it, which is the value that ends up in
        :data:`~spacr.multi_database.SOURCE_COLUMN`. Naming the file one way
        here and another way in the merged frame would leave the user unable to
        connect a row to the database it came from, which is the whole reason
        provenance is carried.
        """
        out: Dict[str, Dict[str, str]] = {}
        tables: Dict[str, List[str]] = {}
        for path in self.paths():
            out[path] = {"label": os.path.basename(path), "tables": "",
                         "plates": "", "rows": "", "status": "ready"}
            try:
                tables[path] = [name for name in mergeable_tables(path)
                                if name in OBJECT_TABLES]
            except Exception as error:  # noqa: BLE001 - one bad file, not all
                tables[path] = []
                out[path]["status"] = f"could not be read: {error}"
                continue
            out[path]["tables"] = ", ".join(tables[path]) or "no object table"

        anchor = self.anchor()
        readable = [path for path in out if anchor in tables.get(path, ())]
        if not readable:
            return out
        try:
            plan = describe_merge(readable, anchor, screens=self.screens())
        except Exception as error:  # noqa: BLE001 - report, do not raise
            for path in readable:
                out[path]["status"] = f"could not be read: {error}"
            return out
        by_plate = {entry.path: entry.plate for entry in self._databases}
        for source in plan.sources:
            # Keyed on the path the plan was given, so the row and the summary
            # cannot come apart.
            detail = out[source.path]
            detail["label"] = source.label
            detail["plates"] = ", ".join(source.plates)
            detail["rows"] = f"{source.rows:,} {anchor}"
            plate = by_plate.get(source.path, "")
            if plate and source.plates and plate not in source.plates:
                # The row says plate3 and the file holds plate7. Not refused --
                # the plate label in the input table is the user's own name for
                # the row -- but never silent either.
                detail["status"] = (f"holds {', '.join(source.plates)}, not "
                                    f"{plate}")
        return out

    # ------------------------------------------------------------ the choice

    def _offer_tables(self) -> None:
        """Fill the table chooser with what every attached database has."""
        paths = self.paths()
        try:
            names = joinable_tables(paths) if paths else ()
        except Exception as error:  # noqa: BLE001 - report, do not raise
            names = ()
            self.report.setPlainText(f"Could not read the databases: {error}")
        previous = set(self.selected_tables()) or set(self._tables)
        self._tables = names

        self._filling = True
        try:
            self.tables_list.clear()
            for name in names:
                item = QListWidgetItem(name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                # Everything present is checked: the user asked for the
                # measurements, and a table they did not want is one click.
                item.setCheckState(
                    Qt.Checked if (not previous or name in previous)
                    else Qt.Unchecked)
                self.tables_list.addItem(item)

            anchors = anchor_tables(names)
            chosen = self.anchor_box.currentText()
            self.anchor_box.clear()
            self.anchor_box.addItems(anchors)
            if chosen in anchors:
                self.anchor_box.setCurrentText(chosen)
            elif DEFAULT_ANCHOR in anchors:
                self.anchor_box.setCurrentText(DEFAULT_ANCHOR)
        finally:
            self._filling = False

    def selected_tables(self) -> Tuple[str, ...]:
        """The tables the user has ticked, in table order."""
        out = []
        for row in range(self.tables_list.count()):
            item = self.tables_list.item(row)
            if item.checkState() == Qt.Checked:
                out.append(item.text())
        return tuple(out)

    def set_selected_tables(self, names: Sequence[str]) -> None:
        """Tick exactly ``names``."""
        wanted = {str(name) for name in names}
        self._filling = True
        try:
            for row in range(self.tables_list.count()):
                item = self.tables_list.item(row)
                item.setCheckState(Qt.Checked if item.text() in wanted
                                   else Qt.Unchecked)
        finally:
            self._filling = False
        self.describe()

    def anchor(self) -> str:
        """The table a row of the merge means one of."""
        return self.anchor_box.currentText() or DEFAULT_ANCHOR

    def set_anchor(self, name: str) -> None:
        """Choose the anchor, if it is on offer."""
        self.anchor_box.setCurrentText(str(name))

    def policy(self) -> MergePolicy:
        """The merge policy the controls describe.

        Note what is NOT here: a join type. ``how_for`` derives it per table
        from cardinality, and these two checkboxes are the only settings that
        change it.
        """
        return MergePolicy(primary=self.anchor(),
                           overrides=dict(self._overrides),
                           consolidate_on_cell=self.consolidate.isChecked(),
                           keep_uninfected=self.keep_uninfected.isChecked())

    def _on_choice(self, *_args) -> None:
        if self._filling:
            return
        self.describe()

    # ----------------------------------------------------- what it will cost

    def describe(self) -> str:
        """State what the merge WOULD do, before it is done.

        Reads only sqlite metadata and the distinct plate ids, so it is cheap
        enough to run on every click -- which is the point, because the answer
        has to arrive before the user commits.
        """
        text = self.plan_text()
        self.report.setPlainText(text)
        return text

    def plan_text(self) -> str:
        """The pre-merge statement: anchor, rows, joins, drops, collisions."""
        paths = self.paths()
        if not paths:
            return ("No database to merge. A plate row with no database is "
                    "legal — it still runs in the regression; it just has no "
                    "measurements to show here.")
        anchor = self.anchor()
        tables = [name for name in self.selected_tables() if name != anchor]
        policy = self.policy()
        lines = [
            f"Anchor: {anchor}"
            + (" (the default)" if anchor == DEFAULT_ANCHOR else "")
            + " — one row per cell, one anchor, one copy of each column.",
        ]
        # BEFORE THE RUN, NOT FOUR MINUTES IN. A row that named a database
        # which is not there is left out of the merge, and left out silently
        # is how a result comes to describe fewer plates than the user thinks.
        gone = [entry.plate for entry in self._databases
                if entry.attached and not entry.present]
        if gone:
            lines.append(
                f"{len(gone)} plate(s) name a database that is not on disk and "
                f"are left out: " + ", ".join(gone) + ".")
        try:
            plan = describe_merge(paths, anchor, screens=self.screens())
        except Exception as error:  # noqa: BLE001 - report, do not raise
            return "\n".join(lines + [f"Could not read {anchor}: {error}"])

        lines.append(f"{len(plan.sources)} database(s), "
                     f"{plan.total_rows:,} {anchor} rows before any join:")
        for source in plan.sources:
            lines.append(f"  {source.label}: {source.rows:,} rows, plates "
                         + (", ".join(source.plates) or "none"))

        if tables:
            lines.append("Joined per table, by what the object IS — there is "
                         "no single join type to choose:")
            for table in tables:
                lines.append(f"  {table}: {policy.how_for(table)} join")
        else:
            lines.append("No other table chosen: the merge is the "
                         f"{anchor} table alone.")

        if plan.dropped_columns:
            lines.append(
                f"{len(plan.dropped_columns)} {anchor} measurement(s) are in "
                f"only some databases and would be dropped: "
                + ", ".join(plan.dropped_columns))
        for table in tables:
            lines.extend(self._table_notes(paths, table, policy))

        if plan.shared_plates_across_screens:
            for plate, screens in plan.shared_plates_across_screens.items():
                lines.append(
                    f"Plate {plate} appears in screens "
                    f"{', '.join(screens)} — two identities, kept apart by "
                    f"{SCREEN_COLUMN}. It is NOT renamed: a qualified plate "
                    f"id hides the screen inside the plate name.")
        if plan.colliding_plates:
            for plate, labels in plan.colliding_plates.items():
                lines.append(
                    f"Plate {plate} is in {', '.join(labels)} WITHIN one "
                    f"screen. The merge will be refused: pooling them would "
                    f"compute every per-well number over two experiments at "
                    f"once. Rename the plates, drop one database, or name "
                    f"the screens.")
        return "\n".join(lines)

    def _table_notes(self, paths, table, policy) -> List[str]:
        """The per-table lines: dropped columns and default aggregations."""
        try:
            plan = describe_merge(paths, table, screens=self.screens())
        except Exception as error:  # noqa: BLE001
            return [f"  {table}: could not be read: {error}"]
        lines = []
        if plan.dropped_columns:
            lines.append(
                f"  {table}: {len(plan.dropped_columns)} measurement(s) in "
                f"only some databases would be dropped: "
                + ", ".join(plan.dropped_columns))
        if not is_one_row_per_cell(table):
            keys = set(IDENTITY) | {anchor_column(table), OBJECT_COLUMN,
                                    SCREEN_COLUMN, SOURCE_COLUMN}
            fell = default_aggregation_columns(
                [name for name in plan.common_columns if name not in keys],
                overrides=policy.overrides)
            if fell:
                lines.append(
                    f"  {table}: {len(fell)} column(s) match no aggregation "
                    f"rule and would take the default "
                    f"({DEFAULT_AGGREGATION}) — " + ", ".join(fell))
        return lines

    # ---------------------------------------------------------- the merge

    def merge(self, **kwargs):
        """Merge the chosen tables and report what it cost.

        :returns: the merged frame, or ``None`` when nothing was merged. A
            refusal is shown in full rather than summarised: it is an ANSWER,
            and it says what to do about it.
        """
        # Re-read first: the input table may have gained a row, and a database
        # that was on disk when the tab was opened may not be now. Merging the
        # list the panel happens to be holding would merge the previous run's
        # inputs, which is the failure the provider is a callable to prevent.
        self.refresh()
        paths = self.paths()
        if not paths:
            self.report.setPlainText(self.plan_text())
            return None
        plan_text = self.plan_text()
        notes: List[str] = []
        try:
            frame = merge_across_databases(
                paths, self.selected_tables(), policy=self.policy(),
                screens=self.screens(), report=notes.append, **kwargs)
        except MergeRefused as refusal:
            self.report.setPlainText(
                plan_text + "\n\nRefused, and nothing was merged:\n"
                + str(refusal))
            return None
        except Exception as error:  # noqa: BLE001 - report, do not raise
            self.report.setPlainText(
                plan_text + f"\n\nThe merge did not finish: {error}")
            return None

        self._frame = frame
        self.report.setPlainText(plan_text + "\n\n" + merge_report(frame)
                                 + ("\n" + "\n".join(notes) if notes else ""))
        self.merged.emit(frame)
        return frame

    @property
    def frame(self):
        """The last merged frame, or ``None``."""
        return self._frame

    @property
    def overrides(self) -> Dict[str, str]:
        """The user's per-column aggregation choices, which beat every rule."""
        return dict(self._overrides)

    def show_aggregation_rules(self) -> None:
        """The per-column rules, for the columns actually about to be merged.

        Reuses the Gate Editor's dialog rather than growing a second one: the
        rules are the same rules, and two editors of one decision is how they
        come to disagree.
        """
        from PySide6.QtWidgets import QMessageBox
        from .aggregation_rules import AggregationRulesDialog

        frame = self._frame
        if frame is None:
            paths = self.paths()
            tables = [name for name in self.selected_tables()
                      if not is_one_row_per_cell(name)] or list(
                          self.selected_tables())
            if not paths or not tables:
                QMessageBox.information(
                    self, "Nothing to show",
                    "The rules are per measurement, so there is nothing to "
                    "show until a database is attached and a table chosen.")
                return
            try:
                # A preview, not the merge: enough rows to know each column's
                # type, which is all the rules need.
                frame = read_merged(paths, tables[0], screens=self.screens(),
                                    limit_per_source=200)
            except Exception as error:  # noqa: BLE001
                QMessageBox.information(self, "Could not read the tables",
                                        str(error))
                return
        dialog = AggregationRulesDialog(frame, self,
                                        overrides=self._overrides)
        dialog.rules_changed.connect(self._on_rules_changed)
        dialog.show()
        self._rules_dialog = dialog

    def _on_rules_changed(self, overrides: dict) -> None:
        self._overrides = dict(overrides or {})
        self.describe()


class MeasurementScanPanel(QWidget):
    """The scan's result table, and the two numbers behind every row.

    :ivar measurement_selected: emitted with the measurement name of the
        selected row, so a host can draw it.
    :ivar scanned: emitted with the number of measurements scanned.
    """

    measurement_selected = Signal(str)
    scanned = Signal(int)

    def __init__(self, frame_provider=None, parent=None,
                 database_provider=None):
        """
        :param frame_provider: called with no arguments for the well-level
            frame to scan. A callable rather than a stored frame, so the panel
            cannot go on scanning the previous run's data after a new one is
            loaded.
        :param database_provider: called with no arguments for the regression
            input table's rows, so the measurement databases attached to each
            plate appear here (instruction 130). A callable for the same
            reason: the tab must not go on showing the previous run's inputs.
        """
        super().__init__(parent)
        from .fast_plots import ResultsTable

        self._frame_provider = frame_provider
        self._result = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # THE DATABASES COME FIRST, because they are the input to everything
        # below them. Hidden entirely when no plate row has one, so a project
        # that never attached a database sees the tab it has always seen.
        self.databases = DatabaseMergePanel(database_provider, self)
        self.databases.databases_changed.connect(self._on_databases_changed)
        layout.addWidget(self.databases)
        self.databases.setVisible(bool(self.databases.databases))

        top = QHBoxLayout()
        self._run = QPushButton("Scan measurements")
        self._run.setToolTip(
            "Hold the model fixed and sweep the dependent variable: which "
            "measurement has genes with a clear effect. Corrected across the "
            "scan, not only within each measurement.")
        self._run.clicked.connect(self.run_scan)
        top.addWidget(self._run)

        top.addWidget(QLabel("rank by"))
        self._rank = QComboBox()
        # Effect size first, because that is what was asked for and because
        # with enough wells a trivial effect is significant.
        self._rank.addItem("effect size", "effect_size")
        self._rank.addItem("across-scan q", "across_scan_q")
        self._rank.addItem("within-run q", "within_run_q")
        self._rank.currentIndexChanged.connect(self._resort)
        top.addWidget(self._rank)
        top.addStretch(1)
        layout.addLayout(top)

        self._status = QLabel("No scan yet.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self.table = ResultsTable()
        self.table.configure(
            placeholder="Filter measurements — a channel, a shape, anything",
            significance_filter=False)
        self.table.table.itemSelectionChanged.connect(self._on_selection)
        layout.addWidget(self.table, 1)

    # -------------------------------------------------------------- running

    def set_frame_provider(self, provider) -> None:
        """Take a new source for the frame the scan runs on."""
        self._frame_provider = provider

    def set_database_provider(self, provider) -> None:
        """Take a new source for the input table's attached databases.

        The same shape as :meth:`set_frame_provider`, and for the same reason:
        the tab re-reads the rows rather than holding a copy of them.
        """
        self.databases.set_database_provider(provider)

    def refresh_databases(self) -> int:
        """Re-read the attached databases. Called when the tab is opened.

        :returns: how many readable databases are attached.
        """
        return self.databases.refresh()

    def _on_databases_changed(self, count: int) -> None:
        # Shown when there is anything to show -- including rows whose
        # database is missing or absent, because "this plate has none" is
        # exactly what a user opening this tab needs to be told.
        self.databases.setVisible(bool(self.databases.databases))

    def run_scan(self, **kwargs) -> bool:
        """Scan whatever the provider is holding. Returns whether it ran."""
        frame = None
        if callable(self._frame_provider):
            try:
                frame = self._frame_provider()
            except Exception as error:  # noqa: BLE001 - report, do not raise
                self._status.setText(f"Could not read the data: {error}")
                return False
        if frame is None or not len(frame):
            self._status.setText(
                "Nothing to scan. Load a run whose wells carry both the gene "
                "assignment and the measurements.")
            return False
        return self.scan(frame, **kwargs)

    def scan(self, frame, **kwargs) -> bool:
        """Scan ``frame`` and show the result."""
        from ...measurement_scan import ScanRefused, scan_measurements

        try:
            result = scan_measurements(frame, **kwargs)
        except ScanRefused as refusal:
            # A refusal is an ANSWER and it says what to do about it. Shown
            # in full rather than summarised: "the scan failed" would send the
            # user looking for a bug in the software.
            self._status.setText(str(refusal))
            self.table.set_frame(None)
            self._result = None
            return False
        except Exception as error:  # noqa: BLE001 - report, do not raise
            self._status.setText(f"The scan did not finish: {error}")
            self.table.set_frame(None)
            self._result = None
            return False
        return self.set_result(result)

    def set_result(self, result) -> bool:
        """Show an already-computed :class:`ScanResult`."""
        self._result = result
        table = result.frame()
        if not len(table):
            self._status.setText("No measurement could be scanned.\n"
                                 + result.describe())
            self.table.set_frame(None)
            return False

        # BOTH CORRECTIONS, IN WORDS, ON EVERY ROW. A measurement that passes
        # within its own run and fails across the scan is the single most
        # important thing this feature can tell a user, and it is invisible in
        # two columns of small numbers.
        table = table.copy()
        table["verdict"] = [verdict_for(row) for row in result.rows]
        table = table.loc[table.index]           # keep the frame's own order
        self.table.set_frame(table[ordered_columns(table)],
                             key_column="measurement")
        self._status.setText(self._summary(result))
        self.scanned.emit(len(result.rows))
        return True

    @staticmethod
    def _summary(result) -> str:
        """The header. Leads with the gap between the two corrections."""
        survivors = len(result.surviving())
        within = sum(1 for row in result.rows if row.survives_within_run)
        text = [
            f"{len(result.rows)} measurements scanned. "
            f"{survivors} show a clear gene effect across the scan; "
            f"{within} would have been reported by a single-measurement run."
        ]
        if within > survivors:
            text.append(
                f"The {within - survivors} in between are the ones a "
                f"per-measurement analysis would have shown you as hits.")
        dropped = getattr(result, "genes_dropped", None)
        if dropped:
            text.append(
                f"{len(dropped)} gene(s) left out for having fewer than two "
                f"wells — a gene in one well has nothing corroborating it: "
                + ", ".join(sorted(dropped)[:6])
                + ("…" if len(dropped) > 6 else ""))
        if result.skipped:
            text.append(f"{len(result.skipped)} column(s) not scanned.")
        return "  ".join(text)

    # ------------------------------------------------------------ selection

    @property
    def result(self):
        return self._result

    def _resort(self) -> None:
        if self._result is None:
            return
        column = self._rank.currentData()
        table = self._result.frame().copy()
        table["verdict"] = [verdict_for(row) for row in self._result.rows]
        if column in table.columns:
            ascending = column != "effect_size"
            key = (lambda s: s.abs()) if column == "effect_size" else None
            table = table.sort_values(column, ascending=ascending, key=key,
                                      kind="stable").reset_index(drop=True)
        self.table.set_frame(table[ordered_columns(table)],
                             key_column="measurement")

    def _on_selection(self) -> None:
        key = None
        items = self.table.table.selectedItems()
        if items:
            key = self.table.key_for_row(items[0].data(Qt.UserRole))
        if key:
            self.measurement_selected.emit(str(key))
