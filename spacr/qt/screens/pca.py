"""The PCA screen — hundreds of features, two axes, and which ones did it.

Assembles four things that already exist:

* :class:`spacr.qt.widgets.pca_view.PCAPanel` — the feature picker, the scree
  plot, the scores plot and the loadings biplot;
* :class:`spacr.qt.widgets.data_filter_panel.DataFilterPanel` — the Local Data
  Filter, unchanged, so narrowing the population narrows this screen and every
  other open view together;
* :mod:`spacr.qt.linked_selection` — so brushing a cluster in PC space
  highlights those cells on the plate map and opens them as crops;
* :func:`spacr.qt.screens.graph_builder.read_table` — the same CSV/SQLite
  reader the Graph Builder loads through, rather than a second one that reads
  ``measurements.db`` slightly differently.

The one thing worth knowing before reading a chart from this screen is in
:mod:`spacr.qt.widgets.pca_model`: features are standardised by default,
because ``cell_area`` in px² would otherwise be PC1 of every table in the
project; NaN is never quietly imputed, because a ``pathogen_*`` NaN means "no
pathogen" and not "value unknown"; and the report under the plot always says
how much of the table the picture is actually about.

Filter, then decompose
----------------------
A filter change **recomputes** the decomposition rather than just re-drawing
it, and that is the interesting decision on this screen. PCA is a property of
a population: the centre, the scale and the component directions are all
computed from the rows in it. Keeping the old components and dropping the
filtered points would draw a plot whose axes belong to a population the user is
no longer looking at — the scores would still be in the old basis, and a
cluster that separates in the filtered subset would not appear. So the filter
is upstream of the maths, the recomputation is debounced, and the report under
the plot re-states the row count every time.

:func:`register` is **not** called at import; see its docstring, and the same
note on :func:`spacr.qt.screens.graph_builder.register`, for the registration
collateral that is still owned by ``app.py``.
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
from ..widgets.pca_view import PCAPanel
from .graph_builder import read_table, table_names

LOG = logging.getLogger("spacr.qt.screens.pca")

__all__ = ["PCAScreen", "make_pca_screen", "register", "APP_KEY", "APP_NAME",
           "APP_DESCRIPTION", "APP_INTRO", "APP_CLI_NOTE"]

#: The registry key. Chosen once and never renamed.
APP_KEY = "pca"

#: A filter change recomputes the whole decomposition, so it is coalesced this
#: long — a dragged range slider must cost one PCA, not one per keystroke.
REFILTER_MS = 250


class PCAScreen(QWidget):
    """Load a measurement table, decompose it, and brush the result.

    :param link: a private
        :class:`~spacr.qt.linked_selection.LinkedSelection` for tests. ``None``
        joins the process-wide one, which is the point of the screen in normal
        use.
    """

    def __init__(self, parent=None, *, link=None, threaded: bool = True):
        super().__init__(parent)
        self.setObjectName("PCAScreen")
        self._frame: Optional[pd.DataFrame] = None
        self._path: Optional[str] = None
        # Every table read goes through here, so it never runs on the GUI
        # thread and always shows up in the run registry (and so in the
        # background-activity spinner).
        self._jobs = JobRunner(self, threaded=threaded, app_key="pca")
        self._jobs.job_failed.connect(self._on_load_failed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["md"], SPACING["md"],
                                 SPACING["md"], SPACING["md"])
        outer.setSpacing(SPACING["sm"])

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(SPACING["sm"])
        title = QLabel("PCA", self)
        title.setObjectName("ScreenTitle")
        head.addWidget(title)

        self._source = QLabel("no table loaded", self)
        self._source.setObjectName("PCASourceLabel")
        head.addWidget(self._source, 1)

        self._table_picker = QComboBox(self)
        self._table_picker.setObjectName("PCATablePicker")
        self._table_picker.setToolTip("Which table of the database to "
                                      "decompose")
        self._table_picker.setVisible(False)
        self._table_picker.currentTextChanged.connect(self._on_table_picked)
        head.addWidget(self._table_picker)

        load = QPushButton("Load table…", self)
        load.setObjectName("PrimaryButton")
        load.setToolTip("A measurements.db, or a CSV of measurements")
        load.clicked.connect(self.choose_table)
        head.addWidget(load)

        self._export = QPushButton("Export…", self)
        self._export.setToolTip("Write the scores, the loadings and the "
                                "explained variance as three CSVs")
        self._export.clicked.connect(self.export_csv)
        self._export.setEnabled(False)
        head.addWidget(self._export)

        self._to_annotate = QPushButton("Open selection in Annotate", self)
        self._to_annotate.setToolTip("Show the brushed objects as image crops")
        self._to_annotate.clicked.connect(self._open_selection)
        self._to_annotate.setEnabled(False)
        head.addWidget(self._to_annotate)
        outer.addLayout(head)

        body = QSplitter(Qt.Horizontal, self)
        body.setChildrenCollapsible(False)
        self.pca = PCAPanel(self, link=link)
        body.addWidget(self.pca)

        self.filters = DataFilterPanel(self, link=link)
        self.filters.setMaximumWidth(320)
        body.addWidget(self.filters)
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 0)
        outer.addWidget(body, 1)

        self.pca.computed.connect(self._on_computed)
        self.pca.failed.connect(self._on_failed)
        self.pca.canvas.rendered.connect(self._on_rendered)

        # The filter is upstream of the maths, so the screen listens for it
        # itself rather than leaving the canvas to redraw stale components.
        self._refilter = QTimer(self)
        self._refilter.setSingleShot(True)
        self._refilter.setInterval(REFILTER_MS)
        self._refilter.timeout.connect(self._recompute_filtered)
        self._link = self.pca.canvas.link
        self._link.filter_changed.connect(self._on_filter_changed)

    # -- data -------------------------------------------------------------
    def set_frame(self, frame: pd.DataFrame, *, label: str = "") -> None:
        """Decompose ``frame``. The one call a host needs."""
        self._frame = frame
        self.filters.set_frame(frame)
        self.pca.set_frame(self._filtered())
        self._source.setText(
            label or f"{len(frame):,} rows × {len(frame.columns)} columns")

    def _filtered(self) -> Optional[pd.DataFrame]:
        """The shared filter applied, or the whole frame if it cannot be.

        A filter naming a column this table does not have is reported rather
        than swallowed — the alternative is a decomposition of more rows than
        the filter panel says are in the population.
        """
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
        # panel below it.
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
        """Re-decompose the narrowed population.

        The components, the centre and the scale all belong to the population,
        so a filter is a new PCA rather than the old one with points removed.
        """
        frame = self._filtered()
        if frame is None:
            return
        selected = self.pca.features.selected()
        self.pca.set_frame(frame, compute=False)
        if selected:
            self.pca.features.set_selected(selected)
        self.pca.recompute()

    # -- results ----------------------------------------------------------
    def _on_computed(self, result) -> None:
        self._export.setEnabled(True)
        self._source.setText(
            f"{len(result):,} objects × {result.n_features} features · "
            f"PC1 {result.explained_variance_ratio[0]:.1%}")

    def _on_failed(self, message: str) -> None:
        self._export.setEnabled(False)
        self._source.setText(message)

    def _on_rendered(self, _data) -> None:
        self._to_annotate.setEnabled(self.pca.canvas.selected_count() > 0)

    def export_csv(self) -> None:
        """Write scores, loadings and explained variance beside each other.

        Three files rather than one, because they have three different row
        meanings — an object, a feature and a component — and a single sheet
        that mixed them would have to be unpicked before anyone could use it.
        """
        result = self.pca.result
        if result is None or self.pca.scores_frame is None:
            self._source.setText("Nothing to export — run a PCA first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export the PCA", "pca_scores.csv", "CSV (*.csv)")
        if not path:
            return
        stem = path[:-4] if path.lower().endswith(".csv") else path
        try:
            self.pca.scores_frame.to_csv(f"{stem}_scores.csv", index=False)
            result.loadings_frame().to_csv(f"{stem}_loadings.csv", index=False)
            result.variance_frame().to_csv(f"{stem}_variance.csv", index=False)
        except OSError as exc:
            LOG.info("could not export the PCA", exc_info=True)
            self._source.setText(f"could not write those files: {exc}")
            return
        self._source.setText(
            f"wrote {os.path.basename(stem)}_scores / _loadings / "
            f"_variance .csv")

    def _open_selection(self) -> None:
        """Send the brushed cluster to whatever shows crops.

        Routed through :func:`spacr.qt.linked_selection.open_objects`, so this
        screen never imports Annotate and Annotate grows no method for it.
        """
        from ..linked_selection import has_object_opener
        canvas = self.pca.canvas
        selection = canvas.link.selection
        if not selection.is_active or not len(selection):
            self._source.setText("Brush a cluster first — nothing is selected.")
            return
        if not has_object_opener("annotate"):
            self._source.setText(
                "Nothing can show crops yet — open the Annotate screen once.")
            return
        plane = canvas.plane()
        where = (f"{canvas.spec.x} / {canvas.spec.y}" if plane
                 else canvas.spec.describe(canvas.kinds))
        try:
            canvas.open_objects(selection.keys,
                                reason=f"brushed in PCA · {where}")
        except Exception as exc:
            LOG.info("could not open the brushed objects", exc_info=True)
            self._source.setText(f"could not open those objects: {exc}")

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
        self.pca.close()
        super().closeEvent(event)


def make_pca_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory handed to :func:`spacr.qt.app.register_app`."""
    return PCAScreen()


APP_NAME = "PCA"
APP_DESCRIPTION = ("Principal components of the measurement table, with a "
                   "loadings biplot")
APP_INTRO = (
    "spaCR measures hundreds of features per object; PCA turns them into two "
    "axes and tells you which measurements built them. Tick the features, "
    "read the scree plot for how many components are real, colour the scores "
    "by any column, and brush a cluster to highlight those cells in every "
    "other open view. Features are standardised by default — without it PC1 "
    "is whichever column is measured in the largest numbers.")
#: What `spacr.cli.INTERACTIVE_ONLY` wants: why this app has no headless run.
APP_CLI_NOTE = ("Interactive multivariate exploration; "
                "spacr.qt.widgets.pca_model.pca() is the headless equivalent.")


def register() -> bool:
    """Put PCA in the app registry, through the public seam. Idempotent.

    Everything after ``section`` is a table this key used to need a hand-edit
    in — the screen header and blurb, the "no headless run" sentence, the API
    doc link and the display name in nine languages.
    :func:`spacr.qt.app.register_app` distributes them; this function only has
    to know them.

    :returns: ``True`` if this call is what registered it. Safe to call
        twice — a module imported from two paths must not raise on the
        duplicate key.

    **Not called at import.** ``app.py`` imports ``spacr.qt.widgets`` before
    ``register_app`` exists, so no module reachable from the top of it can
    register during its import, and a registration that happens later is one
    that some importer's snapshot of ``APPS`` predates. The one place a
    registration is visible to everybody is ``app.py``'s own
    ``_SELF_REGISTERING_APPS`` table, at the bottom of that file. Turning this
    screen on is therefore **one row**::

        ("spacr.qt.screens.pca", "register"),

    and nothing else: the strings above travel with the registration. That row
    is not added here because ``spacr/qt/app.py`` belongs to another change in
    flight, and because a new ``APPS`` row currently reddens the per-app
    inventory tests for reasons this screen cannot fix.
    """
    from ..app import APPS, SECTION_EXPLORE, STAGE_ALPHA, register_app
    if any(row[0] == APP_KEY for row in APPS):
        return False
    register_app(
        APP_KEY, APP_NAME, APP_DESCRIPTION,
        # Explore, not Results & QC: this is asking the measurements a
        # question, not reporting what a finished run produced.
        SECTION_EXPLORE,
        factory=make_pca_screen, stage=STAGE_ALPHA,
        intro=APP_INTRO, cli_note=APP_CLI_NOTE,
        api_module="qt/screens/pca",
        # The acronym where it is the scientific convention, the term where
        # it is not. A tile has room for one word either way.
        translations=("PCA", "PCA", "PCA", "主成分分析", "PCA", "पीसीए",
                      "주성분 분석", "PCA", "ACP"))
    return True
