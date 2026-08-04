"""The Tabulate screen — a pivot table, and the chart of it underneath.

A JMP-style *Tabulate*: drag ``plateID`` down the rows, ``gene`` across the
columns, tick the aggregations, read the numbers. It is the first thing most
people want from a measurement database and the last thing spaCR had.

Four parts, three of them already written:

* :class:`spacr.qt.widgets.pivot_builder.PivotPanel` — the wells and the grid;
* :class:`spacr.qt.widgets.graph_builder.GraphBuilderPanel` — the chart of the
  summary, in the lower half of the splitter;
* :class:`spacr.qt.widgets.data_filter_panel.DataFilterPanel` — the Local Data
  Filter, so narrowing the population narrows the table and every other open
  view at once;
* :func:`spacr.qt.screens.graph_builder.read_table` — the same CSV/SQLite
  reader the Graph Builder loads through.

The table is not a chart
------------------------
"Plot this table" hands the Graph Builder
:meth:`~spacr.qt.widgets.pivot_spec.PivotResult.to_long` — one row per
non-empty cell, one column per statistic — and the Graph Builder does the rest.
So ``x = plateID``, ``y = mean``, ``size = n`` is a drag, and there is no second
implementation of scales, facets or colour to keep in step with the first.

Two things follow from the summary being a real frame rather than a picture,
and both are correct rather than unfortunate:

* the chart's own status line reports **no object keys in this table**, because
  a summary row is a *group*, not an object. Brushing it cannot publish an
  object selection, and saying so beats publishing an empty one.
* the chart is on its own linked-selection source, so it does not answer to a
  lasso drawn over individual cells somewhere else.

Filter, then aggregate
----------------------
A filter change recomputes the pivot rather than restyling it, for the same
reason it recomputes a PCA: an aggregate is a property of the population, and a
mean of the unfiltered rows shown next to a filtered plot is the kind of
mismatch nobody catches by eye.

:func:`register` is **not** called at import; see
:func:`spacr.qt.screens.graph_builder.register` for the registration collateral
still owned by ``app.py``.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSplitter,
    QVBoxLayout, QWidget,
)

from ..job_runner import JobRunner
from ..theme import SPACING
from ..widgets.data_filter_panel import DataFilterPanel
from ..widgets.graph_builder import GraphBuilderPanel
from ..widgets.pivot_builder import PivotPanel
from .graph_builder import read_table, table_names

LOG = logging.getLogger("spacr.qt.screens.tabulate")

__all__ = ["TabulateScreen", "make_tabulate_screen", "register", "APP_KEY",
           "APP_NAME", "APP_DESCRIPTION", "APP_INTRO", "APP_CLI_NOTE"]

#: The registry key. Chosen once and never renamed.
APP_KEY = "tabulate"

#: A filter change re-aggregates the whole frame, so it is coalesced this long.
REFILTER_MS = 200

#: The linked-selection source the summary chart publishes under. Its own, not
#: ``graph_builder``: two views sharing a source would each ignore the other's
#: selections as their own echo.
GRAPH_SOURCE = "tabulate_graph"


class TabulateScreen(QWidget):
    """Load a measurement table, pivot it, and plot the summary.

    :param link: a private
        :class:`~spacr.qt.linked_selection.LinkedSelection` for tests. ``None``
        joins the process-wide one.
    """

    def __init__(self, parent=None, *, link=None, threaded: bool = True):
        super().__init__(parent)
        self.setObjectName("TabulateScreen")
        self._frame: Optional[pd.DataFrame] = None
        self._path: Optional[str] = None
        # Every table read goes through here, so it never runs on the GUI
        # thread and always shows up in the run registry (and so in the
        # background-activity spinner).
        self._jobs = JobRunner(self, threaded=threaded, app_key="tabulate")
        self._jobs.job_failed.connect(self._on_load_failed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["md"], SPACING["md"],
                                 SPACING["md"], SPACING["md"])
        outer.setSpacing(SPACING["sm"])

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(SPACING["sm"])
        title = QLabel("Tabulate", self)
        title.setObjectName("ScreenTitle")
        head.addWidget(title)

        self._source = QLabel("no table loaded", self)
        self._source.setObjectName("TabulateSourceLabel")
        head.addWidget(self._source, 1)

        self._table_picker = QComboBox(self)
        self._table_picker.setObjectName("TabulateTablePicker")
        self._table_picker.setToolTip("Which table of the database to pivot")
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

        stack = QSplitter(Qt.Vertical, body)
        stack.setChildrenCollapsible(False)
        self.pivot = PivotPanel(stack)
        stack.addWidget(self.pivot)
        self.graph = GraphBuilderPanel(stack, link=link, source=GRAPH_SOURCE)
        stack.addWidget(self.graph)
        stack.setStretchFactor(0, 1)
        stack.setStretchFactor(1, 1)
        body.addWidget(stack)

        self.filters = DataFilterPanel(self, link=link)
        self.filters.setMaximumWidth(320)
        body.addWidget(self.filters)
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 0)
        outer.addWidget(body, 1)

        self.pivot.plot_requested.connect(self.plot_summary)
        self.pivot.computed.connect(self._on_computed)

        self._refilter = QTimer(self)
        self._refilter.setSingleShot(True)
        self._refilter.setInterval(REFILTER_MS)
        self._refilter.timeout.connect(self._recompute_filtered)
        self._link = self.graph.canvas.link
        self._link.filter_changed.connect(self._on_filter_changed)
        # Drop anywhere on this screen: the path is resolved through spaCR's
        # project layout, so the plate folder finds what this screen reads.
        from ..dnd import install_for
        install_for(self, "tabulate")

    # -- data -------------------------------------------------------------
    def set_frame(self, frame: pd.DataFrame, *, label: str = "") -> None:
        """Pivot ``frame``. The one call a host needs."""
        self._frame = frame
        self.filters.set_frame(frame)
        self.pivot.set_frame(self._filtered())
        self._source.setText(
            label or f"{len(frame):,} rows × {len(frame.columns)} columns")

    def _filtered(self) -> Optional[pd.DataFrame]:
        if self._frame is None:
            return None
        try:
            return self._link.visible(self._frame)
        except Exception as exc:
            LOG.info("the shared filter does not apply to this table: %s", exc)
            return self._frame

    def choose_table(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a measurement table", "",
            "Measurements (*.db *.sqlite *.csv *.tsv);;All files (*)")
        if path:
            self.load_path(path)

    def load_path(self, path: str, table: Optional[str] = None) -> None:
        """Load a CSV or one table of a SQLite measurement database.

        The read runs on a worker thread. ``SELECT * FROM cell`` into pandas
        measures 1.5 s for a 200 000-row measurement table on a warm local
        SSD, and this method used to run it inline: the whole window stopped
        redrawing for the read. Listing the table names stays inline -- it is
        one ``sqlite_master`` query, measured at 0.4 ms -- because the picker
        has to be populated before the read is dispatched, to know which
        table to read.

        Returns as soon as the read is dispatched;
        :meth:`_on_frame_loaded` finishes on the GUI thread.
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
        # A second load supersedes the first. Without this, switching table
        # twice in quick succession delivers the frames in whatever order the
        # reads happen to finish, and the picker ends up disagreeing with the
        # pivot below it.
        self._jobs.cancel()
        self._source.setText(
            f"loading {os.path.basename(path)}"
            + (f" · {chosen}" if chosen else "") + "…")
        self._jobs.submit(
            lambda p=path, t=chosen: (t, read_table(p, t)),
            self._on_frame_loaded)

    def _on_frame_loaded(self, payload) -> None:
        """Hand a worker-read frame to the pivot. GUI thread only."""
        chosen, frame = payload
        path = self._path or ""
        suffix = f" · {chosen}" if chosen else ""
        self.set_frame(
            frame,
            label=f"{os.path.basename(path)}{suffix} · {len(frame):,} rows "
                  f"× {len(frame.columns)} columns")

    def _on_load_failed(self, message: str) -> None:
        """Report a failed read inline. Never a modal — a dialog nobody can
        dismiss is how a headless run hangs."""
        path = self._path or ""
        LOG.info("could not read %s: %s", path, message)
        self._source.setText(
            f"could not read {os.path.basename(path)}: {message}")

    def active_jobs(self) -> int:
        """How many worker threads are still winding down."""
        return self._jobs.active_jobs()

    def is_busy(self) -> bool:
        """True while a table read is in flight."""
        return self._jobs.is_busy()

    def _on_table_picked(self, name: str) -> None:
        if self._path and name:
            self.load_path(self._path, table=name)

    # -- filter -----------------------------------------------------------
    def _on_filter_changed(self) -> None:
        if self._frame is not None:
            self._refilter.start()

    def _recompute_filtered(self) -> None:
        """Re-aggregate the narrowed population.

        The spec survives: the wells hold column names, and the columns do not
        change when the rows do. Only the numbers move.
        """
        frame = self._filtered()
        if frame is not None:
            self.pivot.set_frame(frame)

    # -- results ----------------------------------------------------------
    def _on_computed(self, result) -> None:
        rows, cols = result.shape
        self._source.setText(
            f"{result.n_source_rows:,} rows → {rows:,} × {cols:,} table")

    def plot_summary(self, frame: Optional[pd.DataFrame] = None) -> None:
        """Hand the summary to the Graph Builder.

        The summary rows are groups rather than objects, so the chart's status
        line will say the table carries no object keys and that brushing
        cannot publish a selection. That is the truth about a mean of forty
        cells, and it is better said than worked around.
        """
        if frame is None:
            frame = self.pivot.long_frame()
        if frame is None or frame.empty:
            self._source.setText(
                "Nothing to plot — build a table with at least one non-empty "
                "cell first.")
            return
        self.graph.set_frame(frame)
        self._source.setText(
            f"plotting the summary · {len(frame):,} cell(s) — drag a key onto "
            f"X and a statistic onto Y")

    def closeEvent(self, event):  # noqa: N802 - Qt name
        # Abandon an in-flight read rather than let it outlive the
        # screen: Qt aborts the process if a running QThread is
        # destroyed, and a worker that delivers into a closed widget
        # is a use-after-free.
        self._jobs.shutdown()
        try:
            self._link.filter_changed.disconnect(self._on_filter_changed)
        except (RuntimeError, TypeError):
            pass
        self._refilter.stop()
        self.pivot.close()
        self.graph.close()
        super().closeEvent(event)


def make_tabulate_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory handed to :func:`spacr.qt.app.register_app`."""
    return TabulateScreen()


APP_NAME = "Tabulate"
APP_DESCRIPTION = ("Pivot the measurement table — rows, columns, "
                   "aggregations, and the n behind each one")
APP_INTRO = (
    "Drag columns onto Rows and Columns to group by them, a measurement onto "
    "Values to summarise it, and tick the statistics you want. plateID / "
    "rowID / columnID down the rows is a plate summary. Every cell prints its "
    "n, because a mean over four objects and a mean over four thousand look "
    "the same otherwise, and a combination with no objects is blank rather "
    "than zero. Export the table, or hand it to the Graph Builder below.")
#: What `spacr.cli.INTERACTIVE_ONLY` wants: why this app has no headless run.
APP_CLI_NOTE = ("Interactive pivot table; "
                "spacr.qt.widgets.pivot_spec.pivot() is the headless "
                "equivalent.")


def register() -> bool:
    """Put Tabulate in the app registry, through the public seam. Idempotent.

    The strings above travel with the registration —
    :func:`spacr.qt.app.register_app` fans ``intro``, ``cli_note``,
    ``api_module`` and ``translations`` out into the four tables that used to
    need a hand-edit each.

    :returns: ``True`` if this call is what registered it. Safe to call twice.

    **Not called at import**, for the reason ``app.py``'s
    ``_SELF_REGISTERING_APPS`` table documents: a registration made anywhere
    else is one that some importer's snapshot of ``APPS`` predates. Turning
    this screen on is **one row** in that table::

        ("spacr.qt.screens.tabulate", "register"),

    left out here because ``spacr/qt/app.py`` belongs to another change in
    flight, and because a new ``APPS`` row currently reddens the per-app
    inventory tests for reasons this screen cannot fix.
    """
    from ..app import APPS, SECTION_EXPLORE, STAGE_ALPHA, register_app
    if any(row[0] == APP_KEY for row in APPS):
        return False
    register_app(
        APP_KEY, APP_NAME, APP_DESCRIPTION, SECTION_EXPLORE,
        factory=make_tabulate_screen, stage=STAGE_ALPHA,
        intro=APP_INTRO, cli_note=APP_CLI_NOTE,
        api_module="qt/screens/tabulate",
        translations=("Tabellera", "Tabellieren", "Tabular", "汇总表",
                      "Tabular", "सारणीबद्ध", "표 만들기", "Taflugerð",
                      "Tabuler"))
    return True
