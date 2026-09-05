"""V4 — Feature Explorer: which of the four hundred features separates them.

spaCR measures hundreds of features per object, so the useful question is never
"plot ``cell_area`` by condition" — it is "which of these actually differs, and
by how much". This screen answers that one: every continuous column scored
against a class column, sorted by separation, with the distributions of the top
few drawn underneath.

The statistic is AUC by default and the reason is written down in
:mod:`spacr.qt.widgets.feature_rank`; so is what it cannot see, which the panel
puts on screen next to the picker rather than in a manual.

Assembles the ranking panel with the Local Data Filter (so a ranking can be
restricted to one plate without leaving the screen) and the B7 formula panel
(so a derived feature is ranked alongside the measured ones). The ranking runs
on a worker thread through :class:`spacr.qt.job_runner.JobRunner`: four hundred
features over two hundred thousand objects is a sort per feature, and doing
that on the GUI thread is a frozen window.

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
from ..linked_selection import linked_selection
from ..theme import SPACING
from ..widgets.data_filter_panel import DataFilterPanel
from ..widgets.feature_explorer import FeatureExplorerPanel
from ..widgets.feature_rank import ExplorerSpec
from ..widgets.formula_editor import FormulaPanel
from .graph_builder import read_table, table_names
from .app_screen import ModuleHeader
from ..app_catalog import declared_app, register_declared

LOG = logging.getLogger("spacr.qt.screens.feature_explorer")

__all__ = ["FeatureExplorerScreen", "make_feature_explorer_screen", "register",
           "APP_KEY", "APP_NAME", "APP_DESCRIPTION", "APP_INTRO",
           "APP_CLI_NOTE", "APP_NAME_TRANSLATIONS"]

APP_KEY = "feature_explorer"


class FeatureExplorerScreen(QWidget):
    """A table, a filter, computed columns, and a ranking of every feature.

    :param parent: parent widget.
    :param link: the :class:`~spacr.qt.linked_selection.LinkedSelection` this
        screen's views join, so a selection made here reaches the others.
        ``None`` joins the shared one; pass a private one in a test.
    :param threaded: whether the work runs off the GUI thread. False runs it
        inline, which is what makes a test deterministic.
    """

    def __init__(self, parent=None, *, link=None, threaded: bool = True):
        """Build the screen: the ranking panel beside the filter and column tabs.

        :param parent: parent widget, or ``None``.
        :param link: shared selection link. Injectable so a test drives a
            private one rather than the process-wide link every other open view
            is also listening to.
        :param threaded: read the database on a worker thread. Set ``False`` in
            tests so a load finishes before it returns.
        """
        super().__init__(parent)
        self.setObjectName("FeatureExplorerScreen")
        self._frame: Optional[pd.DataFrame] = None
        self._path: Optional[str] = None
        # Injectable, so a test drives a private link rather than the
        # process-wide one every other open view is also listening to.
        self._link = link if link is not None else linked_selection()
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
            instruction="Load a table, pick the column that says which class "
                        "each object is in, then rank.",
        )
        self._header = header
        head.addWidget(header)

        self._source = QLabel("no table loaded", self)
        self._source.setObjectName("ExplorerSourceLabel")
        head.addWidget(self._source, 1)

        self._table_picker = QComboBox(self)
        self._table_picker.setVisible(False)
        self._table_picker.currentTextChanged.connect(self._on_table_picked)
        head.addWidget(self._table_picker)

        load = QPushButton("Load table…", self)
        load.setObjectName("PrimaryButton")
        load.clicked.connect(self.choose_table)
        head.addWidget(load)

        export = QPushButton("Export ranking…", self)
        export.setToolTip(
            "Write the whole ranking as CSV — every feature, every statistic, "
            "and the n behind each one.")
        export.clicked.connect(self.choose_export)
        head.addWidget(export)
        outer.addLayout(head)

        body = QSplitter(Qt.Horizontal, self)
        body.setChildrenCollapsible(False)
        self.explorer = FeatureExplorerPanel(self)
        body.addWidget(self.explorer)

        side = QTabWidget(self)
        side.setMaximumWidth(340)
        self.filters = DataFilterPanel(self, link=link)
        self.filters.filter_changed.connect(self._on_filter_changed)
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
        install_for(self, "feature_explorer")
        # Hover help belongs on a setting's NAME, not on the field the user
        # is about to type into (instruction 113). One post-pass rather than
        # a convention every hand-built row has to remember.
        from .settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

    # -- data -------------------------------------------------------------
    def set_frame(self, frame: pd.DataFrame, *, label: str = "") -> None:
        """Point the screen at a table to rank.

        :param frame: the rows, or None to clear.
        """
        self._frame = frame
        self.formulas.set_frame(frame)
        self._push_frame()
        self._source.setText(
            label or f"{len(frame):,} rows × {len(frame.columns)} columns")

    def _push_frame(self) -> None:
        """Hand the table plus its computed columns to the panel and filter.

        The ranking is computed over the **filtered** frame, unlike a computed
        column, which is computed over the whole table. That is the right way
        round: a formula defines a property of an object, so it must not move
        when a slider does; a separation is a statement about a population, so
        restricting the population is exactly what the filter is for — and the
        summary line says how many objects it was computed over.
        """
        frame = self.formulas.computed_frame()
        if frame is None:
            return
        self.filters.set_frame(frame)
        self.explorer.set_frame(self._visible(frame))

    def _visible(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Narrow a frame to the rows the shared filter allows.

        :param frame: the frame to narrow.
        :returns: the visible rows, or the whole frame when the filter does not
            apply here -- a filter written against another table should not
            empty this screen.
        """
        try:
            return self._link.visible(frame)
        except Exception as exc:
            LOG.info("the shared filter does not apply here: %s", exc)
            return frame

    def _on_filter_changed(self) -> None:
        """Re-push the computed frame through the new filter."""
        frame = self.formulas.computed_frame()
        if frame is not None:
            self.explorer.set_frame(self._visible(frame))

    def _on_formulas_changed(self) -> None:
        """Recompute the derived columns and push the frame back to the panel."""
        self._push_frame()

    # -- loading ----------------------------------------------------------
    def choose_table(self) -> None:
        """Ask which table in the project to rank."""
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
        """Show a freshly loaded frame, labelled with its file, table and shape.

        :param payload: the worker's ``(table_name, frame)`` pair.
        """
        chosen, frame = payload
        path = self._path or ""
        suffix = f" · {chosen}" if chosen else ""
        self.set_frame(
            frame,
            label=f"{os.path.basename(path)}{suffix} · {len(frame):,} rows "
                  f"× {len(frame.columns)} columns")

    def _on_load_failed(self, message: str) -> None:
        """Log and show a failed table load.

        :param message: the failure text from the job runner.
        """
        path = self._path or ""
        LOG.info("could not read %s: %s", path, message)
        self._source.setText(
            f"could not read {os.path.basename(path)}: {message}")

    def _on_table_picked(self, name: str) -> None:
        """Reload the current database at a newly chosen table.

        :param name: the table to read; a blank one, or no loaded path, does
            nothing.
        """
        if self._path and name:
            self.load_path(self._path, table=name)

    def active_jobs(self) -> int:
        """How many background jobs this screen is running.

        :returns: the job count.
        """
        return self._jobs.active_jobs()

    def is_busy(self) -> bool:
        """Whether anything is still running.

        What the window asks before closing: a ranking exported while its
        run is still going would be an export of half of it.

        :returns: True while work is outstanding.
        """
        return self._jobs.is_busy()

    # -- export -----------------------------------------------------------
    def choose_export(self) -> None:
        """Ask where to write the ranking."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export the ranking", "feature_ranking.csv",
            "CSV (*.csv);;All files (*)")
        if path:
            self.export_ranking(path)

    def ranking_frame(self) -> Optional[pd.DataFrame]:
        """The ranking as a tidy frame — one row per feature, every statistic.

        Every statistic, not only the one ranked by: a reader who wants to know
        whether the top feature is a shift or a spread should not have to
        re-run the screen with a different picker.
        """
        result = self.explorer.result
        if result is None:
            return None
        return pd.DataFrame([{
            "feature": score.feature,
            "rank": position + 1,
            "separation": score.score,
            "statistic": score.statistic,
            "auc": score.auc,
            "cohen_d": score.cohen_d,
            "ks": score.ks,
            "mutual_info": score.mutual_info,
            "higher_in": score.higher_in,
            "against": score.against,
            "min_n": score.smallest_class,
            "shape_not_shift": score.is_shape_not_shift,
        } for position, score in enumerate(result.scores)])

    def export_ranking(self, path: str) -> Optional[str]:
        """Write the ranking to a file.

        :param path: where to write it.
        :returns: True when it was written.
        """
        frame = self.ranking_frame()
        if frame is None:
            self._source.setText("Nothing ranked yet.")
            return None
        frame.to_csv(path, index=False)
        self._source.setText(f"Ranking written to {os.path.basename(path)}")
        return path

    @property
    def spec(self) -> ExplorerSpec:
        """What the screen is currently set to rank.

        :returns: the explorer spec.
        """
        return self.explorer.spec

    def closeEvent(self, event):  # noqa: N802 - Qt name
        """Shut background work down before going away.

        :param event: the Qt close event.
        """
        self._jobs.shutdown()
        self.explorer.close()
        super().closeEvent(event)


def make_feature_explorer_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory handed to :func:`spacr.qt.app.register_app`."""
    return FeatureExplorerScreen()


# The row this screen puts in the registry is declared in
# `spacr.qt.app_catalog`, which is what lets the app be registered without
# importing this module -- the launch reads the table, not the screen. These
# read the same row back rather than restating it, so the name, the blurb and
# the nine translations have one spelling and no second copy to drift from.
_ROW = declared_app(APP_KEY)
APP_NAME = _ROW.name
APP_DESCRIPTION = _ROW.desc
APP_INTRO = _ROW.intro
APP_CLI_NOTE = _ROW.cli_note
APP_NAME_TRANSLATIONS = _ROW.translations


def register() -> bool:
    """Put the Feature Explorer in the app registry. Idempotent.

    The row itself -- the key, the name, the blurb, the section, the "no
    headless run" sentence, the API doc link and the nine translations of the
    display name -- is declared in :mod:`spacr.qt.app_catalog`.
    :func:`spacr.qt.app.register_app` distributes those into the four tables
    each used to need a hand-edit in, and this function's whole job is to name
    which row. That is what lets the app be registered without importing this
    module at all: the launch reads the table, and the screen is imported when
    somebody opens it.

    :returns: ``True`` if this call is what registered it.
    """
    return register_declared(__name__) is not None
