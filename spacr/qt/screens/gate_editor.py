"""V2 — Gate Editor: draw a region, name it, and it becomes a filter.

The flow-cytometry gesture, on spaCR measurement tables. Drag a threshold
across a histogram or a polygon round the cloud on a two-parameter scatter, name
it, and the shape becomes a :class:`spacr.selection.DataFilter` clause that
every open view honours — the UMAP, the plate map, the crop grid, the Graph
Builder, Small Multiples.

Assembles:

* :class:`spacr.qt.widgets.gate_editor.GateEditorPanel` — the canvas, the three
  tools and the hierarchy with its percentages;
* :class:`spacr.qt.widgets.data_filter_panel.DataFilterPanel` — the Local Data
  Filter, which the gate composes *onto* rather than replacing;
* :class:`spacr.qt.widgets.formula_editor.FormulaPanel` — so a gate can be
  drawn on a computed column;
* Save / Load, because a gating strategy is the reusable part: the whole point
  of a gate over a lasso is that it re-applies to the next plate.

:func:`register` is not called at import; read its docstring.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSizePolicy,
    QSplitter, QVBoxLayout, QWidget, QTabWidget,
)

from ..job_runner import JobRunner
from ..theme import SPACING, page_tabs_qss, register_widget_qss
from ..widgets.data_filter_panel import DataFilterPanel
from ..widgets.gate_search_panel import GateSearchPanel
from ..widgets.formula_editor import FormulaPanel
from ..widgets.gate_canvas import (
    AxisCutoffs, CutoffError, apply_cutoffs, axis_at, axis_menu_items,
    parse_cutoff, AXIS_NAMES,
)
from ..widgets.gate_editor import GateEditorPanel
from ..widgets.gate_spec import GateError, GateSet
from ..widgets.gate_settings import GateEditorSettings, GateSettingsDialog
from ..widgets.graph_spec import GraphSpec, plottable_columns
from ..widgets.gate_console import GateConsole
from ..widgets.table_chip import TableChip
from .graph_builder import read_table, table_names
from .app_screen import ModuleHeader

LOG = logging.getLogger("spacr.qt.screens.gate_editor")

__all__ = ["GateEditorScreen", "make_gate_editor_screen", "register",
           "APP_KEY", "APP_NAME", "APP_DESCRIPTION", "APP_INTRO",
           "APP_CLI_NOTE", "APP_NAME_TRANSLATIONS", "SIDE_TABS_NAME"]

APP_KEY = "gate_editor"

#: objectName of the Filter/Search tab strip, and the key its QSS block is
#: registered under. The two must stay the same string: the rule is matched by
#: objectName, and a tab strip with no rule falls through to the blanket
#: ``QWidget { background-color: bg }`` -- #000000 on dark -- which is a black
#: slab beside the plot rather than an unstyled one.
SIDE_TABS_NAME = "GateSidePanel"


def _side_tabs_qss(palette: dict, opacity) -> str:
    """QSS for the Filter/Search tab strip, through the theme seam.

    Registered HERE, at import, rather than from `GateEditorScreen.__init__`.
    The screen used to ask the theme for a `register_qss` that has never
    existed -- the name is `register_widget_qss` -- and hand it a one-argument
    lambda where the seam calls ``fn(palette, opacity)``. The ImportError
    landed in the except beside it, whose comment says the styling "is not
    worth taking the screen down for", so the block was never registered and
    nothing said so. Registering at import is also what puts the rule in the
    FIRST stylesheet of a session; see `theme.WIDGET_QSS_MODULES`, which this
    module is now listed in.

    ``replace=True``: this module owns the name, so a reimport re-registers
    rather than raising and leaving the tabs unstyled.
    """
    return page_tabs_qss(SIDE_TABS_NAME, palette, opacity)


register_widget_qss(SIDE_TABS_NAME, _side_tabs_qss, replace=True)


class _AxisCutoffDialog(QDialog):
    """Ask for the lowest and highest value one axis should show.

    Two boxes rather than one range, and a BLANK box is a value: it means
    "let the data decide this end". Cutting a long tail off the bottom while
    leaving the top alone is the common case, and demanding both ends would
    make the user invent a number for the end they did not care about.
    """

    def __init__(self, title: str, column: str, cutoff, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{title} cutoffs")
        form = QFormLayout(self)
        self._explain = QLabel(
            f"How much of {column} to draw. Leave a box empty to let the "
            f"data decide that end.\n"
            f"Cutoffs change the VIEW only -- a gate keeps the objects it "
            f"already holds.", self)
        self._explain.setWordWrap(True)
        form.addRow(self._explain)
        self._low = QLineEdit("" if cutoff.low is None else f"{cutoff.low:g}",
                              self)
        self._low.setPlaceholderText("the smallest value drawn")
        self._high = QLineEdit(
            "" if cutoff.high is None else f"{cutoff.high:g}", self)
        self._high.setPlaceholderText("the largest value drawn")
        form.addRow("Lowest", self._low)
        form.addRow("Highest", self._high)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                                   parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> Tuple[Optional[float], Optional[float]]:
        """``(low, high)`` as typed, with a blank box meaning ``None``.

        :raises spacr.qt.widgets.gate_canvas.CutoffError: for text that is
            neither blank nor a number.
        """
        return (parse_cutoff(self._low.text()),
                parse_cutoff(self._high.text()))


class GateEditorScreen(QWidget):
    """A table, two axis pickers, the gating surface, and save/load."""

    def __init__(self, parent=None, *, link=None, threaded: bool = True):
        super().__init__(parent)
        self.setObjectName("GateEditorScreen")
        self._frame: Optional[pd.DataFrame] = None
        self._path: Optional[str] = None
        self._table: Optional[str] = None
        #: The plan for a multi-database load, or None for a single file.
        #: Kept so the screen can say what the merge cost -- which columns
        #: were dropped and which plates were qualified.
        self._merge_plan = None
        #: The working set: every table whose measurements are on offer.
        self._tables: List[str] = []
        #: The OTHER working set: every database those tables are read from.
        #: A screen acquired as three plates is three databases, and the
        #: comparison uses all three in one gate. One database is not a special
        #: case -- it is a set of one.
        self._paths: List[str] = []
        #: What was decided about the last merge, as recorded. Held so the
        #: screen can say it and a test can read it.
        self._merge_decision = None
        self._settings = GateEditorSettings()
        self._settings_dialog: Optional[GateSettingsDialog] = None
        #: Per-measurement display cutoffs, set by right-clicking an axis.
        #: They narrow what is DRAWN and never which rows a gate holds, so a
        #: population cannot come to depend on how far the plot was cut down.
        self._cutoffs = AxisCutoffs()
        self._jobs = JobRunner(self, threaded=threaded, app_key=APP_KEY)
        self._jobs.job_failed.connect(self._on_load_failed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["md"], SPACING["md"],
                                 SPACING["md"], SPACING["md"])
        outer.setSpacing(SPACING["sm"])

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(SPACING["sm"])
        header = ModuleHeader(
            APP_NAME,
            description=APP_DESCRIPTION,
            instruction="Load a table, choose the axes, then drag a "
                        "threshold or draw a polygon.",
        )
        self._header = header
        head.addWidget(header)

        self._source = QLabel("no table loaded", self)
        self._source.setObjectName("GateSourceLabel")
        head.addWidget(self._source, 1)

        self._table_picker = QComboBox(self)
        self._table_picker.setVisible(False)
        self._table_picker.setToolTip(
            "Adds a table to the working set. Picking nucleus does not switch "
            "to nucleus — it merges nucleus measurements alongside the ones "
            "already loaded, so a gate can put a cell measurement on one axis "
            "and a nuclear one on another.")
        self._table_picker.activated.connect(self._on_table_added)
        head.addWidget(self._table_picker)

        load = QPushButton("Load table…", self)
        load.setObjectName("PrimaryButton")
        load.clicked.connect(self.choose_table)
        head.addWidget(load)

        self._save_gates = QPushButton("Save gates…", self)
        self._save_gates.setToolTip(
            "Write the gating strategy to a file. It re-applies to any table "
            "carrying the same measurements — which is the whole difference "
            "between a gate and a lasso.")
        self._save_gates.clicked.connect(self.choose_save_gates)
        head.addWidget(self._save_gates)

        self._load_gates = QPushButton("Load gates…", self)
        self._load_gates.clicked.connect(self.choose_load_gates)
        head.addWidget(self._load_gates)

        # Beside the gate buttons, because a filter set is the same kind of
        # decision: which rows this analysis is about.
        self._save_filters = QPushButton("Save filters…", self)
        self._save_filters.setToolTip(
            "Write the current filter set to a file, so the same rows can "
            "be selected again on another plate.")
        self._save_filters.clicked.connect(self.choose_save_filters)
        head.addWidget(self._save_filters)

        self._load_filters = QPushButton("Load filters…", self)
        self._load_filters.setToolTip(
            "Apply a saved filter set. Columns this table does not have are "
            "reported rather than skipped silently.")
        self._load_filters.clicked.connect(self.choose_load_filters)
        head.addWidget(self._load_filters)

        self._annotate = QPushButton("Annotate…", self)
        self._annotate.setToolTip(
            "Turn the SHOWN gates into one annotation column. Binary marks "
            "objects inside every one of them; multi-class gives each "
            "combination that actually occurs its own class.")
        self._annotate.clicked.connect(self.annotate_from_gates)
        head.addWidget(self._annotate)

        self._export = QPushButton("Export gates…", self)
        self._export.setToolTip(
            "Write every gate to the database as a column of the `filters` "
            "table — the gate's name, 1 inside and 0 outside. The gate is "
            "applied to EVERY object, whatever fraction of the table is "
            "loaded, so a gate drawn on a sample still labels everything.")
        self._export.clicked.connect(self.export_gates)
        head.addWidget(self._export)

        self._save_graph = QPushButton("Save graph\u2026", self)
        self._save_graph.setToolTip(
            "Write the graph as it appears, including the gates drawn on "
            "it. The format follows the figure format in Preferences \u2014 "
            "PNG at the resolution set there, or a real vector PDF whose "
            "text stays selectable and editable \u2014 and you can override "
            "it for one save in the file dialog.")
        self._save_graph.clicked.connect(self.save_graph)
        head.addWidget(self._save_graph)

        # The Settings button lives on the gates panel's tool row, left of
        # Cluster, where the rest of the gating controls are. Two buttons
        # opening one window is one too many.
        outer.addLayout(head)

        # The DATABASE working set, one removable chip per source. Same idiom
        # as the table chips below it, because it is the same idea: a
        # combination the user assembled, which has to be visible and has to
        # be editable a member at a time (instruction 109, point 1).
        self._db_chips = QHBoxLayout()
        self._db_chips.setContentsMargins(0, 0, 0, 0)
        self._db_chips.setSpacing(SPACING["xs"])
        self._db_chips_label = QLabel("Databases", self)
        self._db_chips_label.setObjectName("GateDatabaseChipsLabel")
        self._db_chips_label.setVisible(False)
        self._db_chips.addWidget(self._db_chips_label)
        self._db_chips.addStretch(1)
        outer.addLayout(self._db_chips)

        # The working set, one removable chip per table.
        self._chips = QHBoxLayout()
        self._chips.setContentsMargins(0, 0, 0, 0)
        self._chips.setSpacing(SPACING["xs"])
        self._chips.addStretch(1)
        outer.addLayout(self._chips)

        axes = QHBoxLayout()
        axes.setContentsMargins(0, 0, 0, 0)
        axes.setSpacing(SPACING["xs"])
        axes.addWidget(QLabel("X", self))
        self._x = QComboBox(self)
        self._x.setObjectName("GateXPicker")
        self._x.currentTextChanged.connect(self._on_axes_changed)
        axes.addWidget(self._x, 1)
        axes.addWidget(QLabel("Y", self))
        self._y = QComboBox(self)
        self._y.setObjectName("GateYPicker")
        self._y.setToolTip(
            "Leave empty for a one-parameter histogram, which is what a "
            "threshold gate is drawn on.")
        self._y.currentTextChanged.connect(self._on_axes_changed)
        axes.addWidget(self._y, 1)
        # Z, shown only in 3D/xD. Hidden rather than absent in 2D: the third
        # measurement is remembered while the user works in 2D, so switching
        # back does not lose it.
        self._z_label = QLabel("Z", self)
        axes.addWidget(self._z_label)
        self._z = QComboBox(self)
        self._z.setObjectName("GateZPicker")
        self._z.setToolTip(
            "The third measurement. Gates in 3D are drawn against it; in 2D "
            "it is remembered but not used.")
        self._z.currentTextChanged.connect(self._on_z_changed)
        axes.addWidget(self._z, 1)
        self._set_z_visible(False)
        axes.addStretch(2)
        outer.addLayout(axes)

        body = QSplitter(Qt.Horizontal, self)
        # Collapsible, deliberately. The console carries a width floor
        # (CONSOLE_MIN_WIDTH), so with collapsing DISABLED the splitter
        # would be forced to hand it 320px on every screen -- and the
        # console is meant to start out of the way. Allowing collapse gives
        # it two honest states, hidden or readable, instead of the third
        # one the user actually met: open but too narrow to read.
        body.setChildrenCollapsible(True)
        self.gates = GateEditorPanel(self, link=link)
        self.gates.gates_changed.connect(self._on_gates_changed)
        self.gates.axes_requested.connect(self._on_axes_requested)
        self.gates.settings_requested.connect(self.open_settings)
        self.gates.mode_requested.connect(self._on_mode_requested)
        self.gates.projection_requested.connect(self._on_projection_requested)
        self.gates.spin_axis_changed.connect(self.gates.canvas.set_spin_axis)
        self._install_graph_context_menu()
        body.addWidget(self.gates)

        self.console = GateConsole(self)
        self.console.setToolTip(
            "Ask a question about the table you are gating without leaving "
            "the screen.")

        # ONE section, not two tabs. Filter and Columns were separate tabs
        # inside a QTabWidget capped at 340px, and a panel whose content
        # needs more than that had nowhere to put it -- which is what read
        # as elements overlapping. They are also the same job: both narrow
        # what the scatter shows, so hiding one behind the other meant
        # neither could be checked while using the other.
        #
        # A scroll area rather than a taller widget: the content is
        # unbounded (a table can have hundreds of columns) and the panel is
        # not, so something has to scroll or something has to clip.
        side_body = QWidget(self)
        side_column = QVBoxLayout(side_body)
        side_column.setContentsMargins(0, 0, 0, 0)
        side_column.setSpacing(SPACING["md"])

        self.filters = DataFilterPanel(self, link=link)
        self.formulas = FormulaPanel(self)
        self.formulas.formulas_changed.connect(self._on_formulas_changed)
        for title, panel in (("Filter", self.filters),
                             ("Columns", self.formulas)):
            heading = QLabel(title, side_body)
            heading.setObjectName("SectionHeading")
            side_column.addWidget(heading)
            side_column.addWidget(panel)
        side_column.addStretch(1)

        filter_scroll = QScrollArea(self)
        filter_scroll.setWidget(side_body)
        filter_scroll.setWidgetResizable(True)
        filter_scroll.viewport().setAutoFillBackground(False)

        # FILTER AND SEARCH AS TABS, which is the last item of instruction 31.
        #
        # The search is a thing you ITERATE ON -- change a parameter, look,
        # change it again -- and it lived behind a modal, so looking meant
        # closing the dialog and reopening it to change anything. Beside the
        # filter it is one click away and the plot stays visible while it is
        # adjusted.
        #
        # Filter and COLUMNS are NOT the pair that becomes tabs, and they were
        # deliberately merged into one page earlier: they are the same job --
        # both narrow what the scatter shows -- so hiding one behind the other
        # meant neither could be checked while using the other. Search is a
        # different job, which is what makes it a different tab.
        self.side_tabs = QTabWidget(self)
        self.side_tabs.setObjectName(SIDE_TABS_NAME)
        self.side_tabs.addTab(filter_scroll, "Filter")
        self.search = GateSearchPanel(self)
        self.search.settings_changed.connect(self._on_search_settings)
        self.search.run_requested.connect(self.gates.run_cluster)
        search_scroll = QScrollArea(self)
        search_scroll.setWidget(self.search)
        search_scroll.setWidgetResizable(True)
        search_scroll.viewport().setAutoFillBackground(False)
        self.side_tabs.addTab(search_scroll, "Search")
        # The tab strip's QSS is registered at import -- see `_side_tabs_qss`.
        # It used to be registered from here, against a name the theme has
        # never exported, so it never was.

        side = self.side_tabs
        side.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Expanding)
        # The width is the SPLITTER's to decide now. A hard maximum is what
        # made the cap unescapable: the user could not widen the column even
        # when the content plainly needed it.
        side.setMinimumWidth(260)
        try:
            from ..theme import make_transparent
            make_transparent(side)
        except Exception:
            pass

        body.addWidget(side)

        # The console goes in the splitter too, so the user decides how much
        # room a transcript deserves. Collapsed by default: it is a thing you
        # reach for, not a thing you look past.
        body.addWidget(self.console)
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 0)
        body.setStretchFactor(2, 0)
        body.setSizes([700, 260, 0])
        outer.addWidget(body, 1)
        # Drop anywhere on this screen: the path is resolved through spaCR's
        # project layout, so the plate folder finds what this screen reads.
        from ..dnd import install_for
        install_for(self, "gate_editor")
        # Hover help belongs on a setting's NAME, not on the field the user
        # is about to type into (instruction 113). One post-pass rather than
        # a convention every hand-built row has to remember.
        from .settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

    # -- data -------------------------------------------------------------
    def set_frame(self, frame: pd.DataFrame, *, label: str = "") -> None:
        self._frame = frame
        self.formulas.set_frame(frame)
        self._push_frame()
        self._source.setText(
            label or f"{len(frame):,} rows × {len(frame.columns)} columns")

    def _push_frame(self) -> None:
        """Hand the table plus its computed columns to everything below."""
        frame = self.formulas.computed_frame()
        if frame is None:
            return
        self.gates.set_frame(frame)
        self.console.set_frame(frame)
        self.filters.set_frame(frame)
        self._refill_axis_pickers(frame)

    def _refill_axis_pickers(self, frame: pd.DataFrame) -> None:
        columns = list(plottable_columns(frame))
        current_z = self._z.currentText()
        self._z.blockSignals(True)
        self._z.clear()
        self._z.addItems([""] + columns)
        if current_z in columns:
            self._z.setCurrentText(current_z)
        self._z.blockSignals(False)
        for box, allow_blank in ((self._x, False), (self._y, True)):
            previous = box.currentText()
            box.blockSignals(True)
            box.clear()
            if allow_blank:
                box.addItem("")
            box.addItems(columns)
            if previous in columns:
                box.setCurrentText(previous)
            box.blockSignals(False)
        self._on_axes_changed()

    def _on_formulas_changed(self) -> None:
        self._push_frame()

    def _on_axes_changed(self, *_args) -> None:
        self.gates.set_spec(GraphSpec(x=self._x.currentText() or None,
                                      y=self._y.currentText() or None))

    def _on_axes_requested(self, x_column: str, y_column: str) -> None:
        """Show the measurements a newly selected gate was drawn on.

        Sets the pickers rather than the plot directly, so the change goes
        through the same path a user choosing the axes by hand would take --
        one route to the plot means one behaviour, and the pickers do not end
        up disagreeing with what is drawn.

        A column the current table does not have is ignored: a gate loaded
        from a saved strategy can name a measurement this project never
        produced, and silently blanking the axis would be worse than leaving
        it where it was.
        """
        for box, column in ((self._x, x_column), (self._y, y_column)):
            if not column:
                continue
            index = box.findText(column)
            if index >= 0 and box.currentIndex() != index:
                box.setCurrentIndex(index)

    def _on_gates_changed(self) -> None:
        self._source.setText(self._source.text().split(" · gates")[0]
                             + f" · gates: {len(self.gates.gates)}")

    # -- loading ----------------------------------------------------------
    def choose_table(self) -> None:
        # getOpenFileNames, plural: a screen acquired as three plates is three
        # databases, and comparing them used to mean three sessions
        # (instruction 109). One file behaves exactly as it did before.
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open one or more measurement tables", "",
            "Measurements (*.db *.sqlite *.csv *.tsv);;All files (*)")
        if len(paths) == 1:
            self.load_path(paths[0])
        elif paths:
            self.load_paths(paths)

    def load_paths(self, paths, table: Optional[str] = None) -> None:
        """Load several measurement databases as one frame.

        Every decision that can go quietly wrong -- plate-id collisions,
        mismatched column sets, provenance -- is delegated to
        :mod:`spacr.multi_database`, so this screen and Image UMAP cannot
        disagree about them.

        A collision is REPORTED, not resolved. Two databases that each hold a
        plate called ``plate1`` are two experiments, and pooling them would
        compute every per-well number over both at once with nothing on screen
        to say so. The user is told which plate ids clash, because they are
        the only one who can say whether they are the same plate.
        """
        from ...multi_database import (
            SOURCE_COLUMN, MergeRefused, describe_merge, read_merged)

        paths = [str(p) for p in paths]
        if not paths:
            return
        self._path = paths[0]
        names: List[str] = []
        chosen = table
        try:
            names = table_names(paths[0])
        except Exception as exc:
            self._source.setText(f"could not read {paths[0]}: {exc}")
            return
        if chosen is None:
            chosen = names[0] if names else None
        if not chosen:
            self._source.setText("no table to merge in the chosen files")
            return

        # The plan first and on its own, because it is the thing that has to
        # survive a refusal: a merge that is refused still has to be able to
        # say WHICH plates clashed and in which databases, and that answer
        # comes from the plan rather than from the read that refused.
        plan = None
        try:
            plan = describe_merge(paths, chosen)
            frame = read_merged(paths, chosen, plan=plan)
        except MergeRefused as exc:
            # REFUSED, AND WRITTEN DOWN. The refusal is the screen telling the
            # user; the record is what makes the decision they then take --
            # dropping one of the two databases, usually -- answerable six
            # months later, when the surviving frame can no longer say which
            # plate1 it is.
            self._source.setText(str(exc))
            LOG.info("merge refused for %s: %s", paths, exc)
            self._record_merge(plan, "refused", str(exc),
                               paths=paths, table=chosen)
            return
        except Exception as exc:
            self._source.setText(f"could not merge {len(paths)} files: {exc}")
            LOG.info("merge failed for %s", paths, exc_info=True)
            return

        self._merge_plan = plan
        self._paths = paths
        # The table pickers follow the merge. Without this a multi-database
        # session had no table working set at all: the picker was never
        # filled, so nucleus could not be added to a merged frame, and a
        # later reload would have read only the FIRST database.
        self._table_picker.blockSignals(True)
        self._table_picker.clear()
        self._table_picker.addItems(names)
        self._table_picker.setVisible(bool(names))
        if chosen in names:
            self._table_picker.setCurrentText(chosen)
        self._table_picker.blockSignals(False)
        self._table = chosen
        self._tables = [chosen]
        self._rebuild_chips()
        self._rebuild_database_chips()
        self.set_frame(
            frame,
            label=(f"{len(plan.sources)} databases · {chosen} · "
                   f"{len(frame):,} rows × {len(frame.columns)} columns · "
                   f"colour by {SOURCE_COLUMN}"))
        if plan.partial_columns:
            LOG.info("merge kept only shared columns; %d were present in some "
                     "sources only: %s", len(plan.partial_columns),
                     sorted(plan.partial_columns))
        self._record_merge(plan, "merged",
                           f"merged {len(plan.sources)} databases into the "
                           f"Gate Editor")

    def _record_merge(self, plan, outcome: str, resolution: str, *,
                      paths=None, table: str = "") -> None:
        """Persist a merge decision and retain it for the current view.

        The record preserves collision resolutions that cannot be recovered
        from the merged table itself.
        """
        from ...multi_database import (MergeDecision, decision_for,
                                       record_decision)

        try:
            if plan is not None:
                decision = decision_for(plan, outcome=outcome,
                                        resolution=resolution)
            else:
                decision = MergeDecision(
                    table=table, sources=tuple(paths or ()), labels=(),
                    rows={}, columns="common", dropped_columns=(),
                    colliding_plates={}, outcome=outcome,
                    resolution=resolution,
                    when="")
            self._merge_decision = decision
            record_decision(decision)
        except Exception:
            # An audit line must never be the reason a screen fails to load.
            LOG.info("could not record the merge decision", exc_info=True)

    def load_path(self, path: str, table: Optional[str] = None) -> None:
        """Read a CSV or one table of a measurement database, off the GUI
        thread."""
        self._path = path
        # One database is a working set of one, not a different mode. Keeping
        # the two in one list is what lets `_reload_working_set` stay a single
        # code path -- and what stopped a merged session silently reloading
        # only its first database when a table was added to it.
        if self._paths != [path]:
            self._paths = [path]
            self._rebuild_database_chips()
        names: List[str] = []
        if not str(path).lower().endswith((".csv", ".tsv", ".txt")):
            try:
                names = table_names(path)
            except Exception as exc:
                LOG.info("could not list tables in %s", path, exc_info=True)
                self._source.setText(
                    f"could not read {os.path.basename(path)}: {exc}")
                return
        self._table_picker.blockSignals(True)
        self._table_picker.clear()
        self._table_picker.addItems(names)
        self._table_picker.setVisible(bool(names))
        if table and table in names:
            self._table_picker.setCurrentText(table)
        self._table_picker.blockSignals(False)
        chosen = table or (self._table_picker.currentText() or None)
        self._jobs.cancel()
        self._source.setText(
            f"loading {os.path.basename(path)}"
            + (f" · {chosen}" if chosen else "") + "…")
        self._table = chosen
        if chosen and chosen not in self._tables:
            # A table the working set does not have means a NEW database or a
            # deliberate switch, so the set restarts. A table it already has
            # means a reload -- a settings change, say -- and the set has to
            # survive it, or every sampling change silently unmerges.
            self._tables = [chosen]
            self._rebuild_chips()
        fraction = self._settings.sample_fraction
        cap = self._settings.max_points or None
        self._jobs.submit(
            lambda p=path, t=chosen, f=fraction, c=cap: (t, self._read(p, t, f, c)),
            self._on_frame_loaded)

    @staticmethod
    def _read(path: str, table: Optional[str], fraction: float,
              cap: Optional[int]):
        """Read the table, taking only ``fraction`` of a database table.

        A CSV is read whole: the sampling is done in SQL, and reading the
        whole file only to throw four rows in five away would cost more than
        it saves. The row cap still applies to both.
        """
        if str(path).lower().endswith((".csv", ".tsv", ".txt")) or not table:
            return read_table(path, table, limit=cap)
        from ...filters import read_sampled
        return read_sampled(path, table, fraction=fraction, limit=cap)

    def _on_frame_loaded(self, payload) -> None:
        chosen, frame = payload
        path = self._path or ""
        suffix = f" · {chosen}" if chosen else ""
        # WHAT THE FRAME ACTUALLY IS. Naming one file after a merge of three
        # would be the screen saying something untrue about the numbers on it.
        head = (f"{len(self._paths)} databases" if len(self._paths) > 1
                else os.path.basename(path))
        self.set_frame(
            frame,
            label=f"{head}{suffix} · {len(frame):,} rows "
                  f"× {len(frame.columns)} columns")

    # -- settings ---------------------------------------------------------

    # -- context menu ------------------------------------------------------
    def _install_graph_context_menu(self) -> None:
        """Right-click the plot for the things you can do to it.

        Every action here already exists somewhere on the screen. This is
        discoverability, not capability -- right-clicking a plot is where
        people look for plot actions, and a feature nobody finds is a
        feature nobody has.

        Each item CALLS the existing method rather than reimplementing it,
        so the menu and the buttons cannot drift apart.
        """
        canvas = getattr(self.gates, "canvas", None)
        if canvas is None:
            return
        canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        canvas.customContextMenuRequested.connect(self._show_graph_menu)
        # Cutoffs are re-applied after EVERY render, because a render is what
        # undoes them: the limits are computed from the data each time, and a
        # render happens on every gate edit. Riding the canvas's own
        # `rendered` signal is what makes a cutoff a state of the view rather
        # than a gesture that survives until the next click.
        canvas.rendered.connect(self._narrow_to_cutoffs)

    def graph_menu_items(self):
        """The plot menu as data: ``[(label, enabled, callback, why)]``.

        Separated from the QMenu so the CONTENTS can be tested without a
        display. An offscreen Qt cannot grab for a popup, so a test that
        builds a real menu hangs -- which is how this method came to exist.

        Every callback is an EXISTING method. The menu is a second route to
        the same code, never a second implementation.
        """
        canvas = getattr(self.gates, "canvas", None)
        if canvas is None:
            return []
        try:
            has_figure = bool(canvas.figure().get_axes())
        except Exception:
            has_figure = False
        draw_first = "" if has_figure else "Draw a graph first."
        items = [
            ("Save graph\u2026", has_figure, lambda: self.save_graph(),
             draw_first),
            ("Copy image to clipboard", has_figure,
             self._copy_graph_to_clipboard, draw_first),
            (None, True, None, ""),                       # separator
            ("Reset view", hasattr(canvas, "reset_view"),
             getattr(canvas, "reset_view", None),
             "" if hasattr(canvas, "reset_view") else "Not available here."),
            ("Graph settings\u2026", True, self.open_settings, ""),
            (None, True, None, ""),
            ("Export gates to the database\u2026", True, self.export_gates, ""),
        ]
        return items

    # -- the axis gesture --------------------------------------------------
    def axis_column(self, axis: str) -> str:
        """The measurement drawn on ``"x"`` or ``"y"``, or ``""``."""
        box = {"x": self._x, "y": self._y}.get(str(axis))
        return "" if box is None else box.currentText()

    def axis_under(self, point) -> Optional[str]:
        """Which axis a right-click at ``point`` landed on, or ``None``.

        ``point`` is in the canvas widget's own coordinates, which is what
        Qt hands a custom context menu. Getting from there to the figure
        means two conversions and both are easy to get wrong: the matplotlib
        canvas is a CHILD of the gate canvas rather than the same widget, and
        matplotlib's display coordinates count upward from the BOTTOM while
        Qt counts downward from the top.

        Returns ``None`` for a click inside the plotting rectangle, which is
        where the plot's own menu belongs.
        """
        canvas = getattr(self.gates, "canvas", None)
        if canvas is None:
            return None
        try:
            figure = canvas.figure()
            axes = figure.get_axes()
            widget = figure.canvas
        except Exception:
            return None
        if not axes or widget is None:
            return None
        local = widget.mapFrom(canvas, point)
        ratio = float(getattr(widget, "device_pixel_ratio", 0)
                      or widget.devicePixelRatioF())
        box = axes[0].bbox
        return axis_at((local.x() * ratio,
                        figure.bbox.height - local.y() * ratio),
                       (box.x0, box.y0, box.x1, box.y1))

    def axis_menu_items(self, axis: str):
        """The axis menu as data, so its CONTENTS can be tested offscreen.

        Separated from the QMenu for the same reason
        :meth:`graph_menu_items` is: an offscreen Qt cannot grab for a popup,
        so a test that builds a real menu hangs.
        """
        column = self.axis_column(axis)
        return axis_menu_items(
            axis, column,
            scale=self._settings.scale_for(axis),
            cutoff=self._cutoffs.get(column),
            positive=self._axis_is_positive(column),
            on_scale=lambda value: self.set_axis_scale(axis, value),
            on_cutoffs=lambda: self.ask_axis_cutoffs(axis),
            on_clear=lambda: self.clear_axis_cutoffs(axis))

    def _axis_is_positive(self, column: str) -> bool:
        """Whether every finite value of ``column`` stays above zero.

        Asked of the canvas, which already answers it for the drawing code,
        so the menu cannot grey a scale the plot would have applied or offer
        one the plot would silently skip.
        """
        canvas = getattr(self.gates, "canvas", None)
        asked = getattr(canvas, "_column_is_positive", None)
        if not column or asked is None:
            return True
        try:
            return bool(asked(column))
        except Exception:
            return True

    def set_axis_scale(self, axis: str, scale: str) -> None:
        """Lay ``axis`` out on ``scale``, from the menu or from the window.

        The menu is a second ROUTE to the scale the settings window already
        holds, never a second copy of it: this writes the same field, so the
        two cannot come to disagree about how the plot is drawn.
        """
        field = f"{axis}_scale"
        if not hasattr(self._settings, field):
            return
        # `log_x` / `log_y` are the retired spelling of the same choice, and
        # `scale_for` prefers the scale only while the scale is linear. Left
        # set, an old log flag would put the axis back on log the moment the
        # menu chose linear.
        self.apply_settings(self._settings.replaced(
            **{field: scale, f"log_{axis}": False}))
        self._show_settings_dialog_scale(axis, scale)

    def _show_settings_dialog_scale(self, axis: str, scale: str) -> None:
        """Keep an open settings window from showing a scale nothing uses.

        The window and the axis menu are two editors of one value. When the
        window has no way of being told, it is rebuilt from the settings that
        are now in force rather than left displaying the old choice -- a
        control that disagrees with the plot is worse than one that blinked.
        """
        dialog = self._settings_dialog
        if dialog is None:
            return
        told = getattr(dialog, "set_scale", None)
        if callable(told):
            told(axis, scale)
            return
        visible = dialog.isVisible()
        dialog.settings_changed.disconnect(self.apply_settings)
        dialog.close()
        dialog.deleteLater()
        self._settings_dialog = None
        if visible:
            self.open_settings()

    def ask_axis_cutoffs(self, axis: str) -> Optional[Tuple]:
        """Ask for the lowest and highest value ``axis`` should show.

        Returns the pair that was applied, or ``None`` when the request was
        cancelled or could not be read.
        """
        column = self.axis_column(axis)
        if not column:
            return None
        dialog = _AxisCutoffDialog(AXIS_NAMES.get(axis, axis), column,
                                   self._cutoffs.get(column), self)
        if not dialog.exec():
            return None
        try:
            low, high = dialog.values()
        except CutoffError as exc:
            self.console.write(f"Cutoffs not applied: {exc}")
            return None
        try:
            self.set_axis_cutoffs(axis, low, high)
        except CutoffError as exc:
            self.console.write(f"Cutoffs not applied: {exc}")
            return None
        return (low, high)

    def set_axis_cutoffs(self, axis: str, low, high) -> None:
        """Show only ``low`` to ``high`` of the measurement on ``axis``.

        Either end may be ``None``, meaning the data decides it.

        :raises spacr.qt.widgets.gate_canvas.CutoffError: when the low end is
            not below the high one.
        """
        column = self.axis_column(axis)
        if not column:
            return
        cutoff = self._cutoffs.set(column, low, high)
        self.console.write(
            f"{column} shows {cutoff.describe()}." if cutoff.is_set
            else f"{column} follows the data again.")
        self._redraw_for_cutoffs()

    def clear_axis_cutoffs(self, axis: str) -> bool:
        """Let ``axis`` follow the data again. Returns whether it was cut."""
        column = self.axis_column(axis)
        if not column or not self._cutoffs.clear(column):
            return False
        self.console.write(f"{column} follows the data again.")
        self._redraw_for_cutoffs()
        return True

    def _redraw_for_cutoffs(self) -> None:
        """Redraw so the new cutoffs take effect."""
        canvas = getattr(self.gates, "canvas", None)
        render = getattr(canvas, "render_now", None)
        if callable(render):
            render()

    def _narrow_to_cutoffs(self, *_args) -> None:
        """Apply the cutoffs to every panel the canvas has just drawn."""
        canvas = getattr(self.gates, "canvas", None)
        if canvas is None or not self._cutoffs:
            return
        columns = (self.axis_column("x"), self.axis_column("y"))
        try:
            panels = canvas.panel_axes().values()
        except Exception:
            return
        narrowed = False
        for ax in panels:
            narrowed = bool(apply_cutoffs(ax, columns, self._cutoffs)) or narrowed
        if narrowed:
            try:
                canvas.figure().canvas.draw_idle()
            except Exception:
                LOG.debug("cutoff repaint skipped", exc_info=True)

    def _show_axis_menu(self, axis: str, point) -> None:
        """Build and show the menu for one axis at ``point``."""
        from PySide6.QtWidgets import QMenu

        canvas = getattr(self.gates, "canvas", None)
        if canvas is None:
            return
        menu = QMenu(self)
        for item in self.axis_menu_items(axis):
            if item.label is None:
                menu.addSeparator()
                continue
            action = menu.addAction(item.label)
            action.setEnabled(bool(item.enabled))
            if item.checked is not None:
                action.setCheckable(True)
                action.setChecked(bool(item.checked))
            if item.why:
                action.setToolTip(item.why)
            if item.callback is not None and item.enabled:
                action.triggered.connect(
                    lambda _c=False, cb=item.callback: cb())
        menu.exec(canvas.mapToGlobal(point))

    def _show_graph_menu(self, point) -> None:
        """Build and show the plot menu at ``point``.

        A right-click on an AXIS asks a different question from one on the
        plot -- how that measurement is laid out and how much of it to show
        -- so it gets its own menu.
        """
        from PySide6.QtWidgets import QMenu

        canvas = getattr(self.gates, "canvas", None)
        if canvas is None:
            return
        axis = self.axis_under(point)
        if axis is not None:
            self._show_axis_menu(axis, point)
            return
        menu = QMenu(self)
        for label, enabled, callback, why in self.graph_menu_items():
            if label is None:
                menu.addSeparator()
                continue
            action = menu.addAction(label)
            action.setEnabled(bool(enabled))
            # A greyed row with no reason is a dead end that looks like a
            # bug, so disabled items say why.
            if why:
                action.setToolTip(why)
            if callback is not None and enabled:
                action.triggered.connect(lambda _c=False, cb=callback: cb())
        menu.exec(canvas.mapToGlobal(point))

    def _copy_graph_to_clipboard(self) -> None:
        """Put the rendered plot on the clipboard."""
        canvas = getattr(self.gates, "canvas", None)
        if canvas is None:
            return
        try:
            from PySide6.QtWidgets import QApplication
            pixmap = canvas.grab()
            if not pixmap.isNull():
                QApplication.clipboard().setPixmap(pixmap)
                self.console.write("Graph copied to the clipboard.")
        except Exception as exc:
            LOG.debug("clipboard copy failed: %s", exc, exc_info=True)
            self.console.write(f"Could not copy the graph: {exc}")

    def open_settings(self) -> None:
        """Show the settings window.

        Not modal, and not re-created: a settings window you have to close to
        see what it did is a settings window you cannot tune anything with.
        The same dialog is raised again so its tab and scroll position
        survive, which is the difference between adjusting a value and
        hunting for it.
        """
        if self._settings_dialog is None:
            self._settings_dialog = GateSettingsDialog(
                self._settings, self, columns=tuple(self._x.itemText(i)
                                                    for i in range(self._x.count())))
            self._settings_dialog.settings_changed.connect(self.apply_settings)
            self._settings_dialog.aggregation_rules_requested.connect(
                self.show_aggregation_rules)
            from ..dialogs import detach_from_window_manager
            detach_from_window_manager(self._settings_dialog)
        self._settings_dialog.show()
        self._settings_dialog.raise_()

    def _set_z_visible(self, visible: bool) -> None:
        self._z_label.setVisible(visible)
        self._z.setVisible(visible)

    def _on_z_changed(self, column: str) -> None:
        self._settings = self._settings.replaced(z_axis=column or "")
        if self._settings.gate_mode == "3D":
            self.gates.canvas.set_mode(self._settings.gate_mode,
                                       z_column=column or "")

    def _on_mode_requested(self, mode: str) -> None:
        """2D or 3D, from the buttons beside Cluster.

        HOW MANY AXES ARE DRAWN, and nothing else. Whether those axes are
        components is `_on_projection_requested`, because the two are
        orthogonal: PC1 vs PC2 in 2D and PC1/PC2/PC3 in 3D are both things
        people want, and one exclusive button group could express neither.

        Routed through `apply_settings` rather than set directly, so the mode
        button and the settings dialog cannot end up disagreeing about which
        mode the editor is in.
        """
        self.apply_settings(self._settings.replaced(gate_mode=mode))
        self._set_z_visible(mode == "3D")
        self.gates.set_spin_controls_visible(mode == "3D")
        self.gates.canvas.set_mode(mode, z_column=self._z.currentText())
        if self._settings_dialog is not None:
            self._settings_dialog.set_mode(mode)

    def _on_projection_requested(self, on: bool) -> None:
        """Gate on components, or on the measurements themselves.

        Switching it ON projects now. Switching it OFF does NOT undo the
        projection: the components are ordinary columns by then, gates may
        already be drawn on them, and silently dropping the columns those
        gates name would break them. The user chooses different axes, which
        is the same gesture as any other axis change.
        """
        self.apply_settings(self._settings.replaced(xd_projection=bool(on)))
        if on:
            error = self.reduce_to_components()
            if error:
                # The button claimed something that did not happen.
                self.gates.set_projection_active(False)
                self.apply_settings(
                    self._settings.replaced(xd_projection=False))

    def _on_search_settings(self, changed: dict) -> None:
        """Fold a search-panel edit into the screen's settings.

        Through `apply_settings` like every other route, so the panel, the
        Cluster dialog and the settings dialog cannot end up holding three
        different opinions about the same number.
        """
        self.apply_settings(self._settings.replaced(**changed))

    def apply_settings(self, settings: GateEditorSettings) -> None:
        """Take new settings, re-reading the table only if one of them needs it.

        Two settings cost a read -- the sample fraction and the row cap. The
        rest are drawing, and re-reading a large table because the user
        nudged a colour map is the lag this dialog exists to remove.
        """
        previous, self._settings = self._settings, settings
        self.gates.apply_settings(settings)
        self.search.apply_settings(settings)
        if previous.costs_a_reload(settings) and self._path:
            # Through the working set, so a reload keeps every merged table.
            if len(self._tables) > 1:
                self._reload_working_set()
            else:
                self.load_path(self._path, self._table)

    def reduce_to_components(self) -> Optional[str]:
        """Project every measurement onto components, and gate on those.

        This is what xD MEANS here. More measurements than can be drawn is not
        a drawing problem to be solved with another axis -- past three there
        is no fourth to add -- so the measurements are projected and the
        projection is gated.

        The components come back as ORDINARY COLUMNS, so every existing tool
        works on them unchanged: the same rectangle, oval, polygon, wand and
        cluster, saved and exported the same way. A gate on PC1 vs PC2 is the
        same kind of object as a gate on area vs intensity.

        :returns: an error to show, or None on success.
        """
        from ...merge_tables import ReductionError, reduce_dimensions

        from ...column_groups import resolve

        frame = self._frame
        if frame is None or frame.empty:
            return "Load a table first."
        numeric = [c for c in plottable_columns(frame)
                   if not str(c).startswith(("PC", "UMAP", "tSNE"))]
        # Nothing picked means every numeric column -- what xD did before
        # there was a picker, so an existing session is unchanged.
        groups = getattr(self._settings, "reduction_groups", None) or {}
        explicit = getattr(self._settings, "reduction_columns", ()) or ()
        columns = resolve(numeric, groups, explicit=explicit) \
            if (groups or explicit) else numeric
        if len(columns) < 2:
            return ("The xD tab selects fewer than two measurements; a "
                    "projection needs two.")
        method = getattr(self._settings, "reduction", "pca")
        try:
            components = reduce_dimensions(
                frame, columns, method=method,
                components=int(getattr(self._settings, "components", 3)),
                n_neighbors=int(getattr(self._settings, "xd_n_neighbors", 15)),
                min_dist=float(getattr(self._settings, "xd_min_dist", 0.1)),
                perplexity=float(getattr(self._settings, "xd_perplexity", 30.0)))
        except ReductionError as exc:
            LOG.info("could not reduce: %s", exc)
            self._source.setText(str(exc))
            return str(exc)

        variance = components.attrs.get("explained_variance") or []
        combined = frame.drop(columns=[c for c in components.columns
                                       if c in frame.columns])
        label = self._variance_label(components, variance)
        warning = self._projection_warning(frame, components, columns, groups,
                                           explicit)
        self.set_frame(combined.join(components),
                       label=label + warning)
        names = list(components.columns)
        if len(names) >= 2:
            self._x.setCurrentText(names[0])
            self._y.setCurrentText(names[1])
        if len(names) >= 3:
            self._z.setCurrentText(names[2])
        return None

    @staticmethod
    def _projection_warning(frame, components, columns, groups, explicit) -> str:
        """What the projection is about, and whether it is an artefact.

        Two questions a picture cannot answer on its own, and neither is
        optional once a user is allowed to choose columns:

        WHICH GROUP DRIVES IT. A group can be ticked and carry almost
        nothing, and nobody notices, because a projection always produces a
        picture. Reported only when one group is doing nearly all the work
        or nearly none -- a balanced split is the expected case and saying
        so every time would train the user to ignore the line.

        WHETHER IT SPLIT ON MISSINGNESS. `reduce_dimensions` fills gaps with
        the column median rather than dropping the row, which is right --
        dropping loses every measurement the object DID have -- but it puts
        every uninfected cell on the same point of every pathogen column.
        The projection can then separate infected from uninfected on the
        FACT of measurement, which is real, reproducible, and not a
        phenotype. That is a split someone would otherwise write up.

        Never raises: a diagnostic that takes the projection down with it
        has cost more than it explained.
        """
        from ...column_groups import columns_in
        from ...merge_tables import group_variance_share, missingness_leak

        notes = []
        try:
            if groups or explicit:
                named = {f"{kind}:{name}": columns_in(columns, kind, name)
                         for kind, names in (groups or {}).items()
                         for name in names}
                if explicit:
                    named["picked by hand"] = list(explicit)
                share = group_variance_share(frame, named)
                if len(share) > 1 and not share.empty:
                    worst = share.iloc[-1]
                    if worst["share"] < 0.05 and worst["columns"]:
                        notes.append(
                            f"{share.index[-1]} carries "
                            f"{worst['share']:.0%} of the variance")
        except Exception:
            LOG.debug("variance share failed", exc_info=True)
        try:
            leak = missingness_leak(components, frame, columns)
            if not leak.empty and leak.iloc[0]["severity"] > 0.5:
                row = leak.iloc[0]
                notes.append(
                    f"the projection separates objects by whether "
                    f"{row['column']} was measured "
                    f"({row['missing_fraction']:.0%} missing) — that is not "
                    f"a phenotype")
        except Exception:
            LOG.debug("missingness leak failed", exc_info=True)
        return ("  ·  " + "; ".join(notes)) if notes else ""

    @staticmethod
    def _variance_label(components, variance) -> str:
        """Name each component with how much it explains.

        "PC1" alone says nothing about whether it is the data or the noise,
        and a projection read without that is the commonest way to see
        structure that is not there.
        """
        if not len(variance):
            return f"{len(components.columns)} component(s)"
        parts = [f"{name} {share:.0%}"
                 for name, share in zip(components.columns, variance)]
        return "projected onto " + ", ".join(parts)

    def show_aggregation_rules(self) -> None:
        """The per-column merge rules, for the columns actually loaded."""
        from PySide6.QtWidgets import QMessageBox
        from ..widgets.aggregation_rules import AggregationRulesDialog

        frame = self._frame
        if frame is None or frame.empty:
            QMessageBox.information(
                self, "No table",
                "Load a table first — the rules are per measurement, so there "
                "is nothing to show until there are measurements.")
            return
        dialog = AggregationRulesDialog(
            frame, self, overrides=self._settings.merge_overrides)
        dialog.rules_changed.connect(self._on_aggregation_rules_changed)
        dialog.show()
        self._rules_dialog = dialog

    def _on_aggregation_rules_changed(self, overrides: dict) -> None:
        """Take new rules and re-merge, but only when several tables are up.

        A single table is never aggregated, so re-reading it would be a
        visible pause in exchange for an identical result.
        """
        self._settings = self._settings.replaced(merge_overrides=overrides)
        if len(self._tables) > 1:
            self._reload_working_set()

    def settings(self) -> GateEditorSettings:
        return self._settings

    # -- export -----------------------------------------------------------

    def save_graph(self, path: str = "") -> str:
        """Write the current graph to a PNG or PDF.

        The format comes from the figure-format PREFERENCE rather than from
        a setting of this screen's own; the design is explicit that a
        second place to answer "am I making PDFs" is one too many. The file
        dialog still lets a single save differ, because "save as" is when a
        user thinks about format.

        Rendering goes through `render_figure_to_png`, the same helper the
        figure queue uses, rather than `savefig`: it applies the figure
        colour, line and text-size preferences, caps the display raster,
        and in PDF mode writes a genuine vector page beside the PNG with
        its fonts embedded as TrueType. Calling matplotlib directly would
        give none of that.

        WHAT IT DOES NOT DO IS RESTYLE FOR PRINT. The colours it applies
        are the ones the preferences resolve to, and the "auto" halves
        follow the app theme -- so under a dark theme the text is white on
        a transparent page, which disappears on paper. That is deliberate
        where it is decided (`_export_vector_pdf` explains why the export
        and the on-screen refinement have to agree), and the way to get a
        print-ready file is to set the figure background and text colours
        explicitly rather than leaving them on "follow the theme".

        :param path: destination. Empty opens a file dialog.
        :returns: the path written, or "" when cancelled or nothing is
            drawn.
        """
        canvas = getattr(self.gates, "canvas", None)
        figure = canvas.figure() if canvas is not None else None
        if figure is None or not figure.get_axes():
            self.console.write("No graph to save yet.")
            return ""

        from ..preferences import get_figure_format
        prefer_pdf = str(get_figure_format() or "png").lower() == "pdf"
        default_ext = ".pdf" if prefer_pdf else ".png"

        if not path:
            filters = ("PDF (*.pdf);;PNG (*.png)" if prefer_pdf
                       else "PNG (*.png);;PDF (*.pdf)")
            path, _chosen = QFileDialog.getSaveFileName(
                self, "Save graph", f"gate_graph{default_ext}", filters)
            if not path:
                return ""

        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(default_ext)

        # `render_figure_to_png` writes the PNG and, in PDF mode, a vector
        # PDF beside it. Asking it for a .png next to the chosen .pdf is how
        # the vector file gets made, so point it at the sibling and hand
        # back whichever one the user asked for.
        from ..widgets.figure_queue import render_figure_to_png
        png_path = target.with_suffix(".png")
        try:
            ok = render_figure_to_png(figure, str(png_path))
        except Exception as exc:                      # pragma: no cover
            LOG.info("saving the gate graph failed: %s", exc, exc_info=True)
            self.console.write(f"Could not save the graph: {exc}")
            return ""
        if not ok:
            self.console.write("Could not save the graph.")
            return ""

        written = target if target.exists() else png_path
        self.console.write(f"Saved the graph to {written}")
        return str(written)

    def export_gates(self) -> None:
        """Write every gate to the database as a column of ``filters``."""
        from PySide6.QtWidgets import QMessageBox

        gates = self.gates.gates
        if gates.is_empty:
            QMessageBox.information(self, "No gates",
                                    "Draw a gate before exporting.")
            return
        path = self._path or ""
        if not path or path.lower().endswith((".csv", ".tsv", ".txt")):
            QMessageBox.information(
                self, "Not a database",
                "Filters are written to the `filters` table of a measurement "
                "database. This table was loaded from a file, which has "
                "nowhere to put them.")
            return
        table = self._table
        if not table:
            QMessageBox.information(
                self, "No table",
                "Choose which table of the database the gates were drawn on.")
            return

        self._source.setText(f"exporting {len(gates)} gate(s)…")
        self._jobs.cancel()
        self._jobs.submit(
            lambda p=path, t=table, g=gates: self._write_gates(p, t, g),
            self._on_exported)

    @staticmethod
    def _write_gates(path: str, table: str, gates: GateSet):
        """Apply every gate to the FULL table and write the columns.

        Off the GUI thread, and reading only the columns each gate needs --
        a handful out of hundreds, which is what keeps this affordable on the
        table that made the module laggy in the first place.

        A gate that cannot be applied is reported by name rather than sinking
        the export: gates are drawn on computed columns too, and one gate on a
        formula the database does not have must not cost the user the other
        five.
        """
        from ...filters import FilterError, export_gate, gate_mask_over_table

        written, failed = [], []
        for gate in gates.gates:
            try:
                frame, mask = gate_mask_over_table(path, table, gates, gate.name)
                column, marked = export_gate(path, frame, mask, gate.name)
                written.append((column, marked))
            except (FilterError, Exception) as exc:
                LOG.info("could not export gate %r", gate.name, exc_info=True)
                failed.append((gate.name, str(exc)))
        return written, failed

    def annotate_from_gates(self) -> None:
        """Label every object from the gates currently shown.

        The SHOWN gates, not all of them: ticking a gate on and off is already
        how the user says which ones count, so asking again in a dialog would
        be asking a question they have already answered.
        """
        from PySide6.QtWidgets import QInputDialog, QMessageBox
        from ...filters import ANNOTATION_MODES, FilterError, annotate_from_gates

        names = list(self.gates.canvas.enabled_gates)
        if not names:
            QMessageBox.information(
                self, "No gates shown",
                "Tick the gates to annotate from. An annotation is built from "
                "the gates on screen.")
            return
        frame = self._frame
        if frame is None or frame.empty:
            QMessageBox.information(self, "No table", "Load a table first.")
            return

        mode, ok = QInputDialog.getItem(
            self, "Annotate from gates",
            f"Using {len(names)} gate(s): {', '.join(names)}",
            list(ANNOTATION_MODES), 0, False)
        if not ok:
            return
        column, ok = QInputDialog.getText(
            self, "Name the annotation",
            "The column this is written to in the filters table:")
        if not ok or not column.strip():
            return

        try:
            labels = annotate_from_gates(frame, self.gates.gates, names,
                                         mode=mode)
        except FilterError as exc:
            QMessageBox.warning(self, "Could not annotate", str(exc))
            return

        counts = labels.value_counts()
        summary = ", ".join(f"{value}: {count:,}"
                            for value, count in counts.head(6).items())
        path = self._path or ""
        if not path or path.lower().endswith((".csv", ".tsv", ".txt")):
            # Still useful without a database: the counts are the answer, and
            # refusing outright would hide them.
            self._source.setText(f"{mode} annotation — {summary} "
                                 f"(not written: this table came from a file)")
            return

        self._jobs.submit(
            lambda p=path, f=frame, l=labels, c=column.strip():
                self._write_annotation(p, f, l, c),
            lambda payload: self._source.setText(
                f"wrote {payload[0]} — {summary}"))

    @staticmethod
    def _write_annotation(path: str, frame, labels, column: str):
        from ...filters import export_annotation

        return export_annotation(path, frame, labels, column)

    def _on_exported(self, payload) -> None:
        written, failed = payload
        parts = [f"{column} ({marked:,} objects)" for column, marked in written]
        message = ("wrote " + ", ".join(parts)) if parts else "nothing written"
        if failed:
            message += " · could not export " + ", ".join(
                f"{name} ({why})" for name, why in failed)
        self._source.setText(message)

    def _on_load_failed(self, message: str) -> None:
        path = self._path or ""
        LOG.info("could not read %s: %s", path, message)
        self._source.setText(
            f"could not read {os.path.basename(path)}: {message}")

    def _on_table_added(self, _index: int) -> None:
        """Picking a table ADDS it to the working set."""
        name = self._table_picker.currentText()
        if not name or name in self._tables:
            return
        self._tables.append(name)
        self._rebuild_chips()
        self._reload_working_set()

    def remove_table(self, name: str) -> None:
        """Drop a table from the working set.

        The last one cannot be dropped: a gate editor with no table is a
        screen with nothing on it, and the user's next move would be to load
        the same table again.
        """
        if name not in self._tables or len(self._tables) == 1:
            return
        self._tables.remove(name)
        self._rebuild_chips()
        self._reload_working_set()

    def _rebuild_chips(self) -> None:
        while self._chips.count() > 1:
            item = self._chips.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for index, name in enumerate(self._tables):
            chip = TableChip(name, self, removable=len(self._tables) > 1)
            chip.removed.connect(self.remove_table)
            self._chips.insertWidget(index, chip)

    # -- the database working set (instruction 109) ------------------------
    def database_labels(self) -> List[str]:
        """The name each loaded database carries in the provenance column.

        Asked of :mod:`spacr.multi_database` rather than derived here, so a
        chip and the ``source_database`` value it stands for cannot disagree
        -- a chip reading ``plate1`` beside a legend reading
        ``measurements (2)`` is provenance the user cannot follow.
        """
        from ...multi_database import source_labels

        if not self._paths:
            return []
        try:
            return list(source_labels(self._paths))
        except Exception:
            return [os.path.splitext(os.path.basename(p))[0]
                    for p in self._paths]

    def _rebuild_database_chips(self) -> None:
        """One removable chip per source database.

        Shown only when there is more than one: a single-database session is
        every session this screen has ever had, and a chip strip naming the
        file the header already names is noise.
        """
        while self._db_chips.count() > 2:
            item = self._db_chips.takeAt(1)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        show = len(self._paths) > 1
        self._db_chips_label.setVisible(show)
        if not show:
            return
        for index, (path, label) in enumerate(
                zip(self._paths, self.database_labels())):
            chip = TableChip(label, self, removable=True)
            chip.setToolTip(path)
            chip.removed.connect(self.remove_database)
            self._db_chips.insertWidget(index + 1, chip)

    def remove_database(self, name: str) -> None:
        """Drop one database from the working set and re-merge the rest.

        By its CHIP's label or by its path, because the chip shows the label
        and a caller usually holds the path.

        This is also the resolution the screen offers for a plate-id
        collision, and the reason it does not offer ``on_collision='qualify'``
        instead: qualifying rewrites ``plate1`` to ``runA-plate1``, which
        makes the keys unique by hiding which experiment a plate belongs to
        inside its own id, where nothing can block on it or colour by it.
        Dropping one of the two databases keeps every remaining number
        meaning what it says.
        """
        labels = self.database_labels()
        target = None
        if name in self._paths:
            target = name
        else:
            for path, label in zip(self._paths, labels):
                if label == name:
                    target = path
                    break
        if target is None or len(self._paths) <= 1:
            return
        remaining = [path for path in self._paths if path != target]
        self._record_merge(self._merge_plan, "resolved",
                           f"removed {name} from the working set",
                           paths=self._paths, table=self._table or "")
        if len(remaining) == 1:
            self._paths = remaining
            self._rebuild_database_chips()
            self.load_path(remaining[0], self._table)
        else:
            self.load_paths(remaining, self._table)

    def _reload_working_set(self) -> None:
        """Re-read the working set: every chosen table, from every database.

        One table is read straight, because merging a table onto itself only
        renames its columns and would make every saved gate on a single-table
        session stop matching.

        Several DATABASES go through
        :func:`spacr.plate_measurements.merge_plate_databases`, which is the
        composition of the two things that already exist -- 41's per-table
        merge rules and 109's per-database stacking -- rather than a third
        set of rules that could disagree with either.
        """
        if not self._path or not self._tables:
            return
        self._jobs.cancel()
        self._source.setText(
            "merging " + ", ".join(self._tables) + "…"
            if len(self._tables) > 1 else f"loading {self._tables[0]}…")
        tables = list(self._tables)
        fraction = self._settings.sample_fraction
        cap = self._settings.max_points or None
        policy = self._merge_policy()
        if len(self._paths) > 1:
            paths = list(self._paths)
            labels = self.database_labels()
            self._jobs.submit(
                lambda p=paths, l=labels, t=tables, c=cap, m=policy:
                    (t[0], self._read_across_databases(p, l, t, c, m)),
                self._on_frame_loaded)
            return
        self._jobs.submit(
            lambda p=self._path, t=tables, f=fraction, c=cap, m=policy:
                (t[0], self._read_working_set(p, t, f, c, m)),
            self._on_frame_loaded)

    @staticmethod
    def _read_across_databases(paths: List[str], labels: List[str],
                               tables: List[str], cap: Optional[int], policy):
        """Every chosen table, from every chosen database, in one frame.

        Off the GUI thread. The frame keeps
        :data:`spacr.multi_database.SOURCE_COLUMN`, so the merged view can
        still be coloured by which database a point came from -- which is the
        single most valuable thing a multi-plate view can show.
        """
        from ...plate_measurements import merge_plate_databases

        merge = merge_plate_databases(
            dict(zip(labels, paths)), tables,
            anchor=tables[0], policy=policy)
        frame = merge.frame
        if cap and len(frame) > int(cap):
            step = max(1, len(frame) // int(cap))
            frame = frame.iloc[::step].head(int(cap)).reset_index(drop=True)
        return frame

    def _merge_policy(self):
        from ...merge_tables import MergePolicy

        primary = self._tables[0] if self._tables else "cell"
        return MergePolicy(
            primary=getattr(self._settings, "merge_primary", None) or primary,
            na=getattr(self._settings, "merge_na", "keep"),
            overrides=getattr(self._settings, "merge_overrides", None))

    @staticmethod
    def _read_working_set(path: str, tables: List[str], fraction: float,
                          cap: Optional[int], policy):
        """Read one table, or merge several, off the GUI thread."""
        from ...merge_tables import merge_tables

        if len(tables) == 1:
            return GateEditorScreen._read(path, tables[0], fraction, cap)
        merged = merge_tables(path, tables, policy=policy)
        if cap and len(merged) > cap:
            step = max(1, len(merged) // int(cap))
            merged = merged.iloc[::step].head(int(cap)).reset_index(drop=True)
        elif fraction < 1:
            step = max(2, int(round(1.0 / fraction)))
            merged = merged.iloc[::step].reset_index(drop=True)
        return merged

    def active_jobs(self) -> int:
        return self._jobs.active_jobs()

    def is_busy(self) -> bool:
        return self._jobs.is_busy()

    # -- the strategy -----------------------------------------------------
    # -- filter sets, saved the way gates already are -------------------

    def choose_save_filters(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the filter set", "filters.json",
            "Filter sets (*.json);;All files (*)")
        if path:
            self.save_filters(path)

    def save_filters(self, path: str) -> str:
        """Write the current filter set to ``path``."""
        self.filters.save(path)
        self._source.setText(f"filters saved to {os.path.basename(path)}")
        return path

    def choose_load_filters(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load a filter set", "",
            "Filter sets (*.json);;All files (*)")
        if path:
            self.load_filters(path)

    def load_filters(self, path: str) -> List[str]:
        """Apply a saved filter set. Reports columns this table does not have.

        Saying so matters more here than it looks. A filter set saved against
        one plate and loaded against another is an ordinary thing to do, and
        a set that half-applies selects the wrong rows while looking like it
        worked.
        """
        try:
            missing = self.filters.load(path)
        except Exception as exc:
            LOG.exception("could not load the filter set %s", path)
            self._source.setText(f"could not load that filter set: {exc}")
            return []
        name = os.path.basename(path)
        if missing:
            self._source.setText(
                f"{name} loaded; this table has no "
                f"{', '.join(sorted(missing))}")
        else:
            self._source.setText(f"filters loaded from {name}")
        return missing

    def choose_save_gates(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the gating strategy", "gates.json",
            "Gates (*.json);;All files (*)")
        if path:
            self.save_gates(path)

    def save_gates(self, path: str) -> str:
        """Write the gating strategy to ``path``."""
        self.gates.gates.save(path)
        self._source.setText(f"gates saved to {os.path.basename(path)}")
        return path

    def choose_load_gates(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load a gating strategy", "",
            "Gates (*.json);;All files (*)")
        if path:
            self.load_gates(path)

    def load_gates(self, path: str) -> bool:
        """Read a gating strategy and apply it to the loaded table.

        A strategy naming a measurement this table does not carry loads anyway
        and reports the problem: the gates are still there to look at and fix,
        which is more use than refusing the file.
        """
        try:
            self.gates.set_gates(GateSet.load(path))
        except (GateError, OSError, ValueError) as exc:
            LOG.info("could not load gates from %s: %s", path, exc)
            self._source.setText(f"could not load those gates: {exc}")
            return False
        self._source.setText(
            f"{len(self.gates.gates)} gate(s) from {os.path.basename(path)}")
        return True

    def closeEvent(self, event):  # noqa: N802 - Qt name
        self._jobs.shutdown()
        self.gates.close()
        super().closeEvent(event)


def make_gate_editor_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory handed to :func:`spacr.qt.app.register_app`."""
    return GateEditorScreen()


APP_NAME = "Gate Editor"
APP_DESCRIPTION = "Draw a threshold or a region on a plot; it becomes a filter"
APP_INTRO = (
    "The flow-cytometry gesture on measurement tables. Drag a threshold across "
    "a histogram or click a polygon round a cloud on a two-parameter scatter, "
    "name it, and the shape becomes a filter every open view honours. Gates "
    "nest — gate on gate on gate — and each one shows its n, its percentage of "
    "its parent and its percentage of the whole table. Save the strategy and "
    "re-apply it to the next plate.")
APP_CLI_NOTE = (
    "The Gate Editor is drawing on a plot; run it in the GUI (spacr-qt). "
    "Headless, spacr.qt.widgets.gate_spec.GateSet.load() reads a saved "
    "strategy and .population(frame, name) applies it — no Qt involved, so a "
    "gate drawn once can gate a whole campaign from a script.")
#: The display name in the nine non-English UI languages, in
#: `spacr.qt.i18n.LANGUAGES` order (sv, de, es, zh_CN, pt, hi, ko, is, fr).
APP_NAME_TRANSLATIONS = (
    "Gate-redigerare", "Gate-Editor", "Editor de compuertas",
    "门控编辑器", "Editor de gates", "गेट संपादक", "게이트 편집기",
    "Gate-ritill", "Éditeur de gates")


def register() -> bool:
    """Put the Gate Editor in the app registry. Idempotent.

    Called from :data:`spacr.qt.SELF_REGISTERING_MODULES`. Everything after
    ``SECTION_EXPLORE`` is a table this key would otherwise need a hand-edit
    in; :func:`spacr.qt.app.register_app` distributes them from this one call.

    :returns: ``True`` if this call is what registered it.
    """
    from ..app import APPS, SECTION_EXPLORE, STAGE_ALPHA, register_app
    if any(row[0] == APP_KEY for row in APPS):
        return False
    register_app(APP_KEY, APP_NAME, APP_DESCRIPTION, SECTION_EXPLORE,
                 factory=make_gate_editor_screen, stage=STAGE_ALPHA,
                 intro=APP_INTRO, cli_note=APP_CLI_NOTE,
                 api_module="qt/screens/gate_editor",
                 translations=APP_NAME_TRANSLATIONS)
    return True
