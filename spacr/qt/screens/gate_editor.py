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
from typing import List, Optional

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSplitter,
    QTabWidget, QVBoxLayout, QWidget,
)

from ..job_runner import JobRunner
from ..theme import SPACING
from ..widgets.data_filter_panel import DataFilterPanel
from ..widgets.formula_editor import FormulaPanel
from ..widgets.gate_editor import GateEditorPanel
from ..widgets.gate_spec import GateError, GateSet
from ..widgets.graph_spec import GraphSpec, plottable_columns
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
        self._table_picker.currentTextChanged.connect(self._on_table_picked)
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
        outer.addLayout(head)

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
        axes.addStretch(2)
        outer.addLayout(axes)

        body = QSplitter(Qt.Horizontal, self)
        body.setChildrenCollapsible(False)
        self.gates = GateEditorPanel(self, link=link)
        self.gates.gates_changed.connect(self._on_gates_changed)
        body.addWidget(self.gates)

        side = QTabWidget(self)
        side.setMaximumWidth(340)
        self.filters = DataFilterPanel(self, link=link)
        side.addTab(self.filters, "Filter")
        self.formulas = FormulaPanel(self)
        self.formulas.formulas_changed.connect(self._on_formulas_changed)
        side.addTab(self.formulas, "Columns")
        body.addWidget(side)
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 0)
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
        self.filters.set_frame(frame)
        self._refill_axis_pickers(frame)

    def _refill_axis_pickers(self, frame: pd.DataFrame) -> None:
        columns = list(plottable_columns(frame))
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
        self._jobs.submit(
            lambda p=path, t=chosen: (t, read_table(p, t)),
            self._on_frame_loaded)

    def _on_frame_loaded(self, payload) -> None:
        chosen, frame = payload
        path = self._path or ""
        suffix = f" · {chosen}" if chosen else ""
        self.set_frame(
            frame,
            label=f"{os.path.basename(path)}{suffix} · {len(frame):,} rows "
                  f"× {len(frame.columns)} columns")

    def _on_load_failed(self, message: str) -> None:
        path = self._path or ""
        LOG.info("could not read %s: %s", path, message)
        self._source.setText(
            f"could not read {os.path.basename(path)}: {message}")

    def _on_table_picked(self, name: str) -> None:
        if self._path and name:
            self.load_path(self._path, table=name)

    def active_jobs(self) -> int:
        return self._jobs.active_jobs()

    def is_busy(self) -> bool:
        return self._jobs.is_busy()

    # -- the strategy -----------------------------------------------------
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
