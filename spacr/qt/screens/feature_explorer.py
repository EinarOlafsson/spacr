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

LOG = logging.getLogger("spacr.qt.screens.feature_explorer")

__all__ = ["FeatureExplorerScreen", "make_feature_explorer_screen", "register",
           "APP_KEY", "APP_NAME", "APP_DESCRIPTION", "APP_INTRO",
           "APP_CLI_NOTE", "APP_NAME_TRANSLATIONS"]

APP_KEY = "feature_explorer"


class FeatureExplorerScreen(QWidget):
    """A table, a filter, computed columns, and a ranking of every feature."""

    def __init__(self, parent=None, *, link=None, threaded: bool = True):
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

    # -- data -------------------------------------------------------------
    def set_frame(self, frame: pd.DataFrame, *, label: str = "") -> None:
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
        try:
            return self._link.visible(frame)
        except Exception as exc:
            LOG.info("the shared filter does not apply here: %s", exc)
            return frame

    def _on_filter_changed(self) -> None:
        frame = self.formulas.computed_frame()
        if frame is not None:
            self.explorer.set_frame(self._visible(frame))

    def _on_formulas_changed(self) -> None:
        self._push_frame()

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

    # -- export -----------------------------------------------------------
    def choose_export(self) -> None:
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
        frame = self.ranking_frame()
        if frame is None:
            self._source.setText("Nothing ranked yet.")
            return None
        frame.to_csv(path, index=False)
        self._source.setText(f"ranking written to {os.path.basename(path)}")
        return path

    @property
    def spec(self) -> ExplorerSpec:
        return self.explorer.spec

    def closeEvent(self, event):  # noqa: N802 - Qt name
        self._jobs.shutdown()
        self.explorer.close()
        super().closeEvent(event)


def make_feature_explorer_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory handed to :func:`spacr.qt.app.register_app`."""
    return FeatureExplorerScreen()


APP_NAME = "Feature Explorer"
APP_DESCRIPTION = "Every feature ranked by how well it separates the classes"
APP_INTRO = (
    "spaCR measures hundreds of features per object, so the ranking is the "
    "feature and the plotting is the easy half. Pick the column that says "
    "which class each object is in and every continuous column is scored and "
    "sorted by separation — AUC by default, because it is rank-based, "
    "unit-free and assumes nothing about the distributions. What the chosen "
    "statistic cannot see is printed next to it, a feature whose classes "
    "differ in spread rather than level is flagged, and the shuffle test says "
    "what the best of your features reaches by chance.")
APP_CLI_NOTE = (
    "The Feature Explorer is a ranked table you scroll; run it in the GUI "
    "(spacr-qt). Headless, "
    "spacr.qt.widgets.feature_rank.rank_features(frame, spec) returns the same "
    "ranking with every statistic per feature and no Qt involved.")
#: The display name in the nine non-English UI languages, in
#: `spacr.qt.i18n.LANGUAGES` order (sv, de, es, zh_CN, pt, hi, ko, is, fr).
APP_NAME_TRANSLATIONS = (
    "Egenskapsutforskaren", "Merkmals-Explorer", "Explorador de características",
    "特征浏览器", "Explorador de características", "फ़ीचर एक्सप्लोरर",
    "특징 탐색기", "Eiginleikakönnuður", "Explorateur de caractéristiques")


def register() -> bool:
    """Put the Feature Explorer in the app registry. Idempotent.

    Called from :data:`spacr.qt.SELF_REGISTERING_MODULES`. Everything after
    ``SECTION_EXPLORE`` is a table this key would otherwise need a hand-edit
    in; :func:`spacr.qt.app.register_app` distributes them from this one call.

    :returns: ``True`` if this call is what registered it.
    """
    from ..app import APPS, SECTION_EXPLORE, STAGE_ALPHA, register_app
    if any(row[0] == APP_KEY for row in APPS):
        return False
    register_app(APP_KEY, APP_NAME, APP_DESCRIPTION, SECTION_EXPLORE,
                 factory=make_feature_explorer_screen, stage=STAGE_ALPHA,
                 intro=APP_INTRO, cli_note=APP_CLI_NOTE,
                 api_module="qt/screens/feature_explorer",
                 translations=APP_NAME_TRANSLATIONS)
    return True
