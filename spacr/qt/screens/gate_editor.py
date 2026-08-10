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
from typing import List, Optional

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

from ..job_runner import JobRunner
from ..theme import SPACING
from ..widgets.data_filter_panel import DataFilterPanel
from ..widgets.formula_editor import FormulaPanel
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
           "APP_CLI_NOTE", "APP_NAME_TRANSLATIONS"]

APP_KEY = "gate_editor"


class GateEditorScreen(QWidget):
    """A table, two axis pickers, the gating surface, and save/load."""

    def __init__(self, parent=None, *, link=None, threaded: bool = True):
        super().__init__(parent)
        self.setObjectName("GateEditorScreen")
        self._frame: Optional[pd.DataFrame] = None
        self._path: Optional[str] = None
        self._table: Optional[str] = None
        #: The working set: every table whose measurements are on offer.
        self._tables: List[str] = []
        self._settings = GateEditorSettings()
        self._settings_dialog: Optional[GateSettingsDialog] = None
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

        side = QScrollArea(self)
        side.setWidget(side_body)
        side.setWidgetResizable(True)
        side.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Expanding)
        # The width is the SPLITTER's to decide now. A hard maximum is what
        # made the cap unescapable: the user could not widen the column even
        # when the content plainly needed it.
        side.setMinimumWidth(260)
        # A QScrollArea's viewport auto-fills with the WINDOW colour, which
        # is #000000 on dark and would put a black slab beside the plot
        # (INVARIANTS 2/3).
        side.viewport().setAutoFillBackground(False)
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
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a measurement table", "",
            "Measurements (*.db *.sqlite *.csv *.tsv);;All files (*)")
        if path:
            self.load_path(path)

    def load_path(self, path: str, table: Optional[str] = None) -> None:
        """Read a CSV or one table of a measurement database, off the GUI
        thread."""
        self._path = path
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
        self.set_frame(
            frame,
            label=f"{os.path.basename(path)}{suffix} · {len(frame):,} rows "
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

    def _show_graph_menu(self, point) -> None:
        """Build and show the plot menu at ``point``."""
        from PySide6.QtWidgets import QMenu

        canvas = getattr(self.gates, "canvas", None)
        if canvas is None:
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
        if self._settings.gate_mode in ("3D", "xD"):
            self.gates.canvas.set_mode(self._settings.gate_mode,
                                       z_column=column or "")

    def _on_mode_requested(self, mode: str) -> None:
        """2D / 3D / xD, from the buttons beside Cluster.

        Routed through `apply_settings` rather than set directly, so the mode
        button and the 3D tab's dropdown cannot end up disagreeing about which
        mode the editor is in.
        """
        self.apply_settings(self._settings.replaced(gate_mode=mode))
        self._set_z_visible(mode in ("3D", "xD"))
        self.gates.set_spin_controls_visible(mode in ("3D", "xD"))
        if mode == "xD":
            self.reduce_to_components()
        self.gates.canvas.set_mode(mode, z_column=self._z.currentText())
        if self._settings_dialog is not None:
            self._settings_dialog.set_mode(mode)

    def apply_settings(self, settings: GateEditorSettings) -> None:
        """Take new settings, re-reading the table only if one of them needs it.

        Two settings cost a read -- the sample fraction and the row cap. The
        rest are drawing, and re-reading a large table because the user
        nudged a colour map is the lag this dialog exists to remove.
        """
        previous, self._settings = self._settings, settings
        self.gates.apply_settings(settings)
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

        frame = self._frame
        if frame is None or frame.empty:
            return "Load a table first."
        columns = [c for c in plottable_columns(frame)
                   if not str(c).startswith(("PC", "UMAP", "tSNE"))]
        method = getattr(self._settings, "reduction", "pca")
        try:
            components = reduce_dimensions(
                frame, columns, method=method,
                components=int(getattr(self._settings, "components", 3)))
        except ReductionError as exc:
            LOG.info("could not reduce: %s", exc)
            self._source.setText(str(exc))
            return str(exc)

        variance = components.attrs.get("explained_variance") or []
        combined = frame.drop(columns=[c for c in components.columns
                                       if c in frame.columns])
        self.set_frame(combined.join(components),
                       label=self._variance_label(components, variance))
        names = list(components.columns)
        if len(names) >= 2:
            self._x.setCurrentText(names[0])
            self._y.setCurrentText(names[1])
        if len(names) >= 3:
            self._z.setCurrentText(names[2])
        return None

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
        a setting of this screen's own; instruction 50 is explicit that a
        second place to answer "am I making PDFs" is one too many. The file
        dialog still lets a single save differ, because "save as" is when a
        user thinks about format.

        Rendering goes through `render_figure_to_png`, the same helper the
        figure queue uses, rather than `savefig` -- it restyles the figure
        for print first. A plot exported with the dark theme's colours is
        white text on black, which is unusable on paper, and that restyle
        is the whole reason not to call matplotlib directly.

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

    def _reload_working_set(self) -> None:
        """Re-read the tables in the working set, merged.

        One table is read straight, because merging a table onto itself only
        renames its columns and would make every saved gate on a single-table
        session stop matching.
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
        self._jobs.submit(
            lambda p=self._path, t=tables, f=fraction, c=cap, m=policy:
                (t[0], self._read_working_set(p, t, f, c, m)),
            self._on_frame_loaded)

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
    "Grindredigerare", "Gate-Editor", "Editor de compuertas",
    "门控编辑器", "Editor de gates", "गेट संपादक", "게이트 편집기",
    "Hliðaritill", "Éditeur de fenêtres")


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
