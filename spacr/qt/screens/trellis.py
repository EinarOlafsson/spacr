"""V5 — Small Multiples: one chart per group, in a grid, on shared axes.

Assembles four things that already exist into one surface:

* :class:`spacr.qt.widgets.trellis_view.TrellisPanelWidget` — the drop zones,
  the scale options and the grid;
* :class:`spacr.qt.widgets.data_filter_panel.DataFilterPanel` — the Local Data
  Filter, unchanged, so narrowing here narrows every open view;
* :class:`spacr.qt.widgets.formula_editor.FormulaPanel` — computed columns, so
  ``ratio = area / perimeter ** 2`` can be faceted the moment it is defined;
* :mod:`spacr.qt.linked_selection` — a brush on one panel highlights the same
  objects in the UMAP, on the plate map and in the crop grid.

Why it is a screen of its own and not a checkbox on the Graph Builder: the
question a trellis answers is "does this shift hold in every plate?", and the
options that make that answerable — which panels share a scale, how a long
strip of levels wraps, what n each panel is built on — are not decoration on a
single chart. They are the chart.

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
from ..widgets.trellis_view import TrellisPanelWidget
from ..widgets.trellis_spec import TrellisSpec
from .graph_builder import read_table, table_names

LOG = logging.getLogger("spacr.qt.screens.trellis")

__all__ = ["TrellisScreen", "make_trellis_screen", "register",
           "APP_KEY", "APP_NAME", "APP_DESCRIPTION", "APP_INTRO",
           "APP_CLI_NOTE", "APP_NAME_TRANSLATIONS"]

#: The registry key. Chosen once and never renamed.
APP_KEY = "trellis"


class TrellisScreen(QWidget):
    """A table, a filter, computed columns, and a grid of small multiples.

    :param link: a private :class:`~spacr.qt.linked_selection.LinkedSelection`
        for tests. ``None`` joins the process-wide one.
    """

    def __init__(self, parent=None, *, link=None, threaded: bool = True):
        super().__init__(parent)
        self.setObjectName("TrellisScreen")
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
        title = QLabel("Small Multiples", self)
        title.setObjectName("ScreenTitle")
        head.addWidget(title)

        self._source = QLabel("no table loaded", self)
        self._source.setObjectName("TrellisSourceLabel")
        head.addWidget(self._source, 1)

        self._table_picker = QComboBox(self)
        self._table_picker.setObjectName("TrellisTablePicker")
        self._table_picker.setToolTip("Which table of the database to plot")
        self._table_picker.setVisible(False)
        self._table_picker.currentTextChanged.connect(self._on_table_picked)
        head.addWidget(self._table_picker)

        load = QPushButton("Load table…", self)
        load.setObjectName("PrimaryButton")
        load.setToolTip("A measurements.db, or a CSV of measurements")
        load.clicked.connect(self.choose_table)
        head.addWidget(load)
        outer.addLayout(head)

        body = QSplitter(Qt.Horizontal, self)
        body.setChildrenCollapsible(False)
        self.panel = TrellisPanelWidget(self, link=link)
        body.addWidget(self.panel)

        side = QTabWidget(self)
        side.setMaximumWidth(360)
        self.filters = DataFilterPanel(self, link=link)
        side.addTab(self.filters, "Filter")
        self.formulas = FormulaPanel(self)
        self.formulas.formulas_changed.connect(self._on_formulas_changed)
        side.addTab(self.formulas, "Columns")
        body.addWidget(side)
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 0)
        outer.addWidget(body, 1)

    # -- data -------------------------------------------------------------
    def set_frame(self, frame: pd.DataFrame, *, label: str = "") -> None:
        """Plot ``frame``. The one call a host needs."""
        self._frame = frame
        self.formulas.set_frame(frame)
        self._push_frame()
        self._source.setText(
            label or f"{len(frame):,} rows × {len(frame.columns)} columns")

    def _push_frame(self) -> None:
        """Hand the table *plus its computed columns* to everything below.

        One place, so a formula added later reaches the grid, the filter picker
        and the column well by the same path the loaded table did — which is
        the whole of what "computed columns participate in everything else"
        means here.
        """
        frame = self.formulas.computed_frame()
        if frame is None:
            return
        self.panel.set_frame(frame)
        self.filters.set_frame(frame)

    def _on_formulas_changed(self) -> None:
        self._push_frame()

    def choose_table(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a measurement table", "",
            "Measurements (*.db *.sqlite *.csv *.tsv);;All files (*)")
        if path:
            self.load_path(path)

    def load_path(self, path: str, table: Optional[str] = None) -> None:
        """Load a CSV or one table of a SQLite measurement database.

        The read runs on a worker thread through
        :class:`spacr.qt.job_runner.JobRunner`; listing the table names stays
        inline because the picker has to be populated before the read is
        dispatched, to know which table to read.
        """
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
        # A second load supersedes the first, so switching table twice does
        # not deliver the frames in whatever order the reads happen to finish.
        self._jobs.cancel()
        self._source.setText(
            f"loading {os.path.basename(path)}"
            + (f" · {chosen}" if chosen else "") + "…")
        self._jobs.submit(
            lambda p=path, t=chosen: (t, read_table(p, t)),
            self._on_frame_loaded)

    def _on_frame_loaded(self, payload) -> None:
        """Hand a worker-read frame to the panel. GUI thread only."""
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

    # -- the grid ---------------------------------------------------------
    @property
    def spec(self) -> TrellisSpec:
        return self.panel.spec

    def set_spec(self, spec: TrellisSpec) -> None:
        self.panel.set_spec(spec)

    def closeEvent(self, event):  # noqa: N802 - Qt name
        # Abandon an in-flight read rather than let it outlive the screen:
        # Qt aborts the process if a running QThread is destroyed.
        self._jobs.shutdown()
        self.panel.close()
        super().closeEvent(event)


def make_trellis_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory handed to :func:`spacr.qt.app.register_app`."""
    return TrellisScreen()


APP_NAME = "Small Multiples"
APP_DESCRIPTION = "One chart per group, in a grid, on axes that really are shared"
APP_INTRO = (
    "Drop a column on X or Y to say what each panel shows, then a grouping "
    "column on Facet ↓ or Facet → to repeat it once per level. Axes are "
    "shared by default, so a shift between panels is a shift in the data; "
    "free, per-row and per-column scales are available and the grid says so "
    "when they are on. Every panel prints its n.")
APP_CLI_NOTE = (
    "Small Multiples is interactive: the drop zones, the scale options and "
    "the brush are the feature. Run it in the GUI (spacr-qt). Headless, "
    "spacr.qt.widgets.trellis_spec.trellis() computes the same grid — panels, "
    "scales and per-panel n — with no Qt involved.")
#: The display name in the nine non-English UI languages, in
#: `spacr.qt.i18n.LANGUAGES` order (sv, de, es, zh_CN, pt, hi, ko, is, fr).
APP_NAME_TRANSLATIONS = (
    "Smådiagram", "Kleine Vielfache", "Múltiplos pequeños",
    "小型多组图", "Pequenos múltiplos", "स्मॉल मल्टीपल्स", "스몰 멀티플",
    "Smámyndaröð", "Petits multiples")


def register() -> bool:
    """Put Small Multiples in the app registry. Idempotent.

    Called from :data:`spacr.qt.SELF_REGISTERING_MODULES`, which
    :func:`spacr.qt.run` runs after ``spacr.qt.app`` is fully executed and
    before ``MainWindow.__init__`` reads the registry — the position the
    docstring there explains.

    Everything after ``SECTION_EXPLORE`` is a table this key would otherwise
    need a hand-edit in: the screen header and blurb, the "no headless run"
    sentence, the API doc link and the nine translations of the display name.
    :func:`spacr.qt.app.register_app` distributes them from this one call.

    :returns: ``True`` if this call is what registered it.
    """
    from ..app import APPS, SECTION_EXPLORE, STAGE_ALPHA, register_app
    if any(row[0] == APP_KEY for row in APPS):
        return False
    register_app(APP_KEY, APP_NAME, APP_DESCRIPTION, SECTION_EXPLORE,
                 factory=make_trellis_screen, stage=STAGE_ALPHA,
                 intro=APP_INTRO, cli_note=APP_CLI_NOTE,
                 api_module="qt/screens/trellis",
                 translations=APP_NAME_TRANSLATIONS)
    return True
