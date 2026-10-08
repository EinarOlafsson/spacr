"""The Graph Builder screen — a table, a filter, and a chart you drag together.

Assembles three things that already exist into one surface:

* :class:`spacr.qt.widgets.graph_builder.GraphBuilderPanel` — the drop zones
  and the canvas;
* :class:`spacr.qt.widgets.data_filter_panel.DataFilterPanel` — the Local Data
  Filter, unchanged, because a filter that narrows *every* view is worth more
  than a private one that narrows this chart;
* :mod:`spacr.qt.linked_selection` — so a brush here highlights the same cells
  in the UMAP and on the plate map, and a lasso there highlights them here.

**What it is for.** Exploring a measurement table without plotting code,
usually after Measure or Classify: drag columns onto the chart and it
redraws as each one lands.

**What it needs.** One table of a ``measurements.db`` or a CSV or TSV file,
chosen with Load table. The object tables and ``png_list`` are offered first;
every other table in the database stays available. Use **Merge tables** beside
that picker to combine tables at a cell/cytoplasm observation level. The popup
shows the shared spaCR aggregation rules, per-column overrides and a validated
preview. **Customize merging** supplies explicit composite keys, relationships
and join types for external schemas, with acknowledgment and reset controls.
Named results persist beside the database in ``.spacr-merges.json`` and are
revalidated on reuse. **Save chart** includes the chart channels and merge
definition; **Load chart** reconstructs the data before plotting. External
results without verified image provenance support plotting and tabular
filtering, while the image navigation action explains why it is unavailable.

**What it produces.** A chart with six drop zones: x, y, colour, size, facet
row and facet column. Only x and y decide the chart type -- one continuous
column gives a histogram, one categorical column a bar chart of counts, two
continuous columns a scatter plot, one of each a box plot and two categorical
columns a heatmap of counts -- and violin and line plots are explicit choices.
A brushed rectangle becomes the shared selection, highlighted in the UMAP and
on the plate map.

**What to do next.** Press Open selection in Annotate to see the brushed
objects as image crops, narrow every view with the Local Data Filter beside
the chart, or move to Gate Editor when a population should become a named
gate that can be saved and re-applied.

The screen goes into the app registry through
:func:`spacr.qt.app.register_app` rather than through a row in the table
inside ``app.py``, and its styling goes through
:func:`spacr.qt.theme.register_widget_qss`. Both seams exist so that a screen
built in parallel with five others is a new file rather than a merge conflict
in two thousand-line ones.

:func:`register` is **not** called at import — read its docstring for why, and
for the one line plus four side-table entries that finish the wiring. The
screen itself is complete: build it with :func:`make_graph_builder_screen`,
hand it a frame, and everything below works.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import sqlite3
import tempfile
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, NamedTuple, Optional, Tuple

import pandas as pd

if TYPE_CHECKING:
    from ..widgets.fold_strip import FoldStrip
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...condition_annotations import (
    PROVENANCE_TABLE,
    annotation_columns,
    apply_conditions,
    source_context,
)
from ..app_catalog import declared_app, register_declared
from ..i18n import tr
from ..job_runner import JobRunner
from ..theme import SPACING
from ..widgets.collapsible_splitter import CollapsibleSplitter
from ..widgets.data_filter_panel import DataFilterPanel
from ..widgets.derived_table_source import DerivedTableSource
from ..widgets.graph_builder import GraphBuilderPanel
from ..widgets.measurements_example import (
    EXAMPLE_TABLE,
    install_test_data_button,
)
from .app_screen import ModuleHeader

LOG = logging.getLogger("spacr.qt.screens.graph_builder")

__all__ = ["GraphBuilderScreen", "make_graph_builder_screen", "register",
           "APP_KEY", "APP_NAME", "APP_DESCRIPTION", "APP_INTRO",
           "APP_CLI_NOTE", "read_table", "table_names"]

#: The registry key. Chosen once and never renamed — saved user state, the
#: bridge, the CLI and the drag-and-drop handlers all key off it.
APP_KEY = "graph_builder"

#: Tables a measurement database is most likely to be explored through, best
#: first. Only a default for the picker; every table is still offered.
_PREFERRED_TABLES = ("object", "cell", "nucleus", "pathogen", "cytoplasm",
                     "png_list")


def table_names(path: str) -> List[str]:
    """Every user table in the SQLite file at ``path``, in a useful order.

    :param path: path to a SQLite measurement database, opened read-only;
        the preferred tables (``object``, ``cell``, ``nucleus``, …) come
        first, then the rest alphabetically. SQLite internals and the private
        condition-annotation and database-write receipt tables are left out; ordinary user tables
        and named derived results remain available.
    """
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as db:
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    from ...database_concurrency import _WRITE_TICKETS_TABLE
    hidden = {PROVENANCE_TABLE.casefold(), _WRITE_TICKETS_TABLE.casefold()}
    found = [row[0] for row in rows if row[0].casefold() not in hidden]
    ranked = [name for name in _PREFERRED_TABLES if name in found]
    from ...derived_tables import load_definitions
    return ranked + [name for name in found if name not in ranked] + list(load_definitions(path))


def read_table(path: str, table: Optional[str] = None,
               limit: Optional[int] = None) -> pd.DataFrame:
    """Read a CSV or one table of a SQLite measurement database.

    :param path: CSV, TSV, text, or SQLite database path. Delimited files are
        read directly; every other suffix is opened as SQLite in read-only
        mode.
    :param limit: optional row cap, applied in SQL. The chart's own large-data
        policy handles size once the frame is in memory; this is only for the
        case where the *file* is too big to read at all.
    """
    if str(path).lower().endswith((".csv", ".tsv", ".txt")):
        sep = "\t" if str(path).lower().endswith(".tsv") else ","
        return pd.read_csv(path, sep=sep, nrows=limit)
    name = table or (table_names(path) or ["object"])[0]
    from ...derived_tables import execute, load_definitions
    definitions = load_definitions(path)
    if name in definitions:
        frame, _report = execute(path, definitions[name])
        return frame.head(limit) if limit else frame
    query = 'SELECT * FROM "' + name.replace('"', '""') + '"'
    if limit:
        query += f" LIMIT {int(limit)}"
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as db:
        frame = pd.read_sql_query(query, db)
    if not limit:
        from ...condition_annotations import saved_table_annotation
        try:
            annotation = saved_table_annotation(path, name, frame)
            if annotation:
                frame.attrs["saved_condition_definition"] = annotation
        except ValueError as exc:
            frame.attrs["condition_annotation_problem"] = str(exc)
    return frame


def _one_line(exc: BaseException) -> str:
    """One line naming ``exc``, spelled the way the runner would have.

    A read that fails inside the job is reported by this screen rather than
    raised out of it (see :class:`_Loaded`), so this has to produce the text
    ``JobRunner`` used to produce -- otherwise moving the failure onto the
    generation-guarded path would quietly reword every error the user sees.
    ``JobRunner._on_worker_error_text`` takes the last non-empty line of the
    worker's traceback, and the last line of a traceback is exactly what
    :func:`traceback.format_exception_only` returns.
    """
    lines = traceback.format_exception_only(type(exc), exc)
    for candidate in reversed("".join(lines).strip().splitlines()):
        if candidate.strip():
            return candidate.strip()
    return str(exc) or exc.__class__.__name__


class _Loaded(NamedTuple):
    """Everything one load job brings back. Plain data: no widget, no raise.

    Built on the worker thread and read on the GUI thread, which is why a
    read that failed travels in :attr:`problem` instead of being raised.
    An exception out of the job leaves through ``JobRunner.job_failed``, and
    THAT signal carries no generation: ``JobRunner.cancel`` drops a stale
    job's *result*, but a stale job's *failure* is still delivered. A
    database on a sleeping share that gives up twenty seconds after the user
    gave up on it would otherwise report "could not read <whatever is on
    screen now>" over a table that loaded perfectly well.

    Carrying the failure here puts it on the same generation-guarded path as
    the frame, and lets the picker be filled from a job that failed -- which
    is the difference between "this table would not read, try another" and a
    screen with no tables on it.
    """

    #: Every table in the file, in picker order. Empty for a delimited file,
    #: and empty when listing the tables is itself what failed.
    names: List[str]
    #: The table this job read, or ``None`` for a delimited file.
    chosen: Optional[str]
    #: The frame, or ``None`` when the read failed.
    frame: Optional[pd.DataFrame]
    #: One line fit for the source label, or ``None`` when the read worked.
    problem: Optional[str]


class GraphBuilderScreen(DerivedTableSource, QWidget):
    """Drag columns onto channels; the chart follows.

    :param link: a private :class:`~spacr.qt.linked_selection.LinkedSelection`
        for tests. ``None`` joins the process-wide one, which is the point of
        the screen in normal use.
    :param parent: parent widget; ownership only.
    :param threaded: ``False`` runs every table read inline instead of on the
        job runner's thread. A TEST NEEDS THE RESULT ON THE LINE AFTER THE
        CALL; a user needs the window to keep painting while a large table
        loads. The jobs are the same either way -- they still register, and a
        file that cannot be read still comes back through
        :meth:`_on_frame_loaded` as a :class:`_Loaded` carrying a ``problem``
        -- so only the waiting differs.
    """

    def __init__(self, parent=None, *, link=None, threaded: bool = True):
        """Build the screen: the graph builder beside the shared filter.

        The registry key is named here rather than inherited: a screen that
        builds itself rather than being the generic ``AppScreen`` has none, and
        fold installation dispatches on exactly that -- so this screen could
        declare folds and never be handed them.

        :param parent: parent widget, or ``None``.
        :param link: shared selection link, passed to the builder and the
            filter.
        :param threaded: read the database on a worker thread. Set ``False`` in
            tests so a load finishes before it returns.
        """
        super().__init__(parent)
        self.setObjectName("GraphBuilderScreen")
        self.app_key = "graph_builder"
        self._frame: Optional[pd.DataFrame] = None
        self._path: Optional[str] = None
        self._annotation_base_frame = None
        self._condition_source = None
        self._condition_definition = None
        self._condition_definitions = {}
        self._threaded = threaded
        self._jobs = JobRunner(self, threaded=threaded, app_key="graph_builder")
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
            instruction="Load a table, then drop columns on X, Y, colour, "
                        "size and facet.",
        )
        self._header = header
        head.addWidget(header)

        self._source = QLabel("no table loaded", self)
        self._source.setObjectName("GraphSourceLabel")
        head.addWidget(self._source, 1)

        self._table_picker = QComboBox(self)
        self._table_picker.setObjectName("GraphTablePicker")
        self._table_picker.setToolTip("Which table of the database to plot")
        self._table_picker.setVisible(False)
        self._table_picker.currentTextChanged.connect(self._on_table_picked)
        head.addWidget(self._table_picker)
        self._install_merge_button(head)

        load = QPushButton("Load table…", self)
        load.setObjectName("PrimaryButton")
        load.setToolTip("A measurements.db, or a CSV of measurements")
        load.clicked.connect(self.choose_table)
        head.addWidget(load)
        save_graph = QPushButton("Save chart…", self)
        save_graph.clicked.connect(self.choose_save_chart)
        head.addWidget(save_graph)
        load_graph = QPushButton("Load chart…", self)
        load_graph.clicked.connect(self.choose_load_chart)
        head.addWidget(load_graph)
        self._conditions_button = QPushButton(tr("Annotate conditions"), self)
        self._conditions_button.setEnabled(False)
        self._conditions_button.clicked.connect(self.open_condition_dialog)
        head.addWidget(self._conditions_button)
        self._export_table_button = QPushButton(tr("Export table…"), self)
        self._export_table_button.setEnabled(False)
        self._export_table_button.clicked.connect(self.choose_export_table)
        head.addWidget(self._export_table_button)
        self._save_annotated_button = QPushButton(tr("Save annotated table…"), self)
        self._save_annotated_button.setEnabled(False)
        self._save_annotated_button.clicked.connect(self.choose_save_annotated_table)
        head.addWidget(self._save_annotated_button)
        install_test_data_button(
            self, head, lambda _folder, db: self.load_path(
                str(db), table=EXAMPLE_TABLE),
            say=self._source.setText)

        self._to_annotate = QPushButton("Open selection in Annotate", self)
        self._to_annotate.setToolTip(
            "Show the brushed objects as image crops")
        self._to_annotate.clicked.connect(self._open_selection)
        self._to_annotate.setEnabled(False)
        head.addWidget(self._to_annotate)
        outer.addLayout(head)

        body = CollapsibleSplitter(Qt.Horizontal, self,
                                   persist_key=f"{APP_KEY}::body")
        self.builder = GraphBuilderPanel(self, link=link, fold_key=APP_KEY)
        body.add_pane(self.builder, "Graph builder", stretch=1)

        self.filters = DataFilterPanel(self, link=link)
        from ..preferences import scaled_px
        self.filters.setMaximumWidth(scaled_px(320))
        self.filters_section = body.add_section(
            self.filters, "Filter", persist_key=f"{APP_KEY}/Filter",
            stretch=0)
        self._body = body
        outer.addWidget(body, 1)

        self.builder.canvas.rendered.connect(self._on_rendered)
        from ..dnd import install_for
        install_for(self, "graph_builder")
        from .settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

    def set_frame(self, frame: pd.DataFrame, *, label: str = "") -> None:
        """Plot ``frame``. The one call a host needs.

        :param frame: the table to chart; handed to the graph builder and
            the filter panel, and its row and column counts label the source
            unless ``label`` is given.
        """
        saved_annotation = frame.attrs.get("saved_condition_definition")
        annotation_problem = frame.attrs.get("condition_annotation_problem")
        if saved_annotation:
            frame = frame.drop(columns=annotation_columns(saved_annotation))
            saved_key = json.dumps(saved_annotation["source"], sort_keys=True)
            self._condition_definitions.setdefault(saved_key, copy.deepcopy(saved_annotation))
        self._annotation_base_frame = frame
        self._condition_source = source_context(
            self._path, self._table_picker.currentText(), frame.attrs.get("merge_definition"))
        key = json.dumps(self._condition_source, sort_keys=True)
        self._condition_definition = None
        definition = self._condition_definitions.get(key)
        if definition:
            try:
                frame = apply_conditions(frame, definition, self._condition_source)
                self._condition_definition = copy.deepcopy(definition)
            except ValueError as exc:
                annotation_problem = tr("Saved conditions were not applied: {error}", error=str(exc))
        self._frame = frame
        self._conditions_button.setEnabled(True)
        self._export_table_button.setEnabled(True)
        self._update_save_annotated_button()
        self._derived_frame_loaded(frame)
        self.builder.set_frame(frame)
        self.filters.set_frame(frame)
        self._source.setText(
            annotation_problem or label or f"{len(frame):,} rows × {len(frame.columns)} columns")

    def choose_table(self) -> None:
        """Ask which table in the project to use."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a measurement table", "",
            "Measurements (*.db *.sqlite *.csv *.tsv);;All files (*)")
        if path:
            self.load_path(path)

    def load_path(self, path: str, table: Optional[str] = None) -> None:
        """Load a CSV or one table of a SQLite measurement database.

        NOTHING HERE TOUCHES THE FILE. The read has always run on a worker
        thread -- ``SELECT * FROM cell`` into pandas measures 1.5 s for a
        200 000-row measurement table on a warm local SSD -- but listing the
        tables was kept inline on the argument that one ``sqlite_master``
        query costs 0.4 ms. That argument holds only for a disk that answers.
        Measured on one workstation, a single ``stat``
        under ``/nas_mnt`` -- an ``autofs`` mount whose share was asleep --
        had not returned after TWENTY SECONDS, and ``sqlite3.connect`` opens
        the file before it can read a byte of ``sqlite_master``. A user
        picking a measurements.db off a sleeping share, or dropping a project
        folder that resolves onto one, froze the whole window with no
        traceback: a stalled event loop is not a crash.

        So the listing goes to the worker with the read, as one job, and the
        picker is populated by :meth:`_on_frame_loaded` when the answer
        arrives. A ``path_probe`` pre-flight would not have helped: it answers
        optimistically from cache, and it is the ``connect`` itself that
        parks.

        Returns as soon as the job is dispatched; :meth:`_on_frame_loaded`
        finishes on the GUI thread, whether the file read or not -- a failure
        comes back as data in the :class:`_Loaded` rather than as an
        exception, so that it is dropped along with everything else when the
        load it belongs to has been superseded.

        :param path: a ``.csv``, ``.tsv`` or ``.txt`` file, read as delimited
            text, or any other file, opened read-only as a SQLite database.
        """
        self._path = path
        self._jobs.cancel()
        self._source.setText(
            f"loading {os.path.basename(path)}"
            + (f" · {table}" if table else "") + "…")
        delimited = str(path).lower().endswith((".csv", ".tsv", ".txt"))

        def work(source=path, wanted=table, is_text=delimited) -> _Loaded:
            """List and read in one job. Worker thread; no widget here.

            The delimited check stays with the listing rather than in front
            of it: `sqlite_master` has nothing to say about a text file, and
            asking would report "could not read" for a CSV pandas reads
            perfectly well.

            The two halves fail separately on purpose. A listing that fails
            means the file is not a database and there is nothing to offer;
            a READ that fails, on one table of a database whose other tables
            listed fine, must still leave the picker populated -- that is how
            the user reaches the table that does read. Inline, that fell out
            of the order the old code ran in; here it has to be said.
            """
            try:
                names = [] if is_text else table_names(source)
            except Exception as exc:                             # noqa: BLE001
                return _Loaded([], wanted, None, _one_line(exc))
            chosen = wanted or (names[0] if names else None)
            try:
                return _Loaded(names, chosen, read_table(source, chosen), None)
            except Exception as exc:                             # noqa: BLE001
                return _Loaded(names, chosen, None, _one_line(exc))

        self._jobs.submit(work, self._on_frame_loaded)

    def _on_frame_loaded(self, loaded: _Loaded) -> None:
        """Fill the picker and hand the frame to the panel. GUI thread only.

        Reached only for the load that is still current -- ``JobRunner``
        checks the generation ``cancel`` bumped before it calls this -- which
        is why the failure branch may safely name ``self._path``.
        """
        self._table_picker.blockSignals(True)
        self._table_picker.clear()
        self._table_picker.addItems(loaded.names)
        self._table_picker.setVisible(bool(loaded.names))
        if loaded.chosen and loaded.chosen in loaded.names:
            self._table_picker.setCurrentText(loaded.chosen)
        self._table_picker.blockSignals(False)
        if loaded.frame is None:
            self._on_load_failed(loaded.problem or "unknown error")
            return
        path = self._path or ""
        suffix = f" · {loaded.chosen}" if loaded.chosen else ""
        self.set_frame(
            loaded.frame,
            label=f"{os.path.basename(path)}{suffix} · "
                  f"{len(loaded.frame):,} rows "
                  f"× {len(loaded.frame.columns)} columns")

    def _on_load_failed(self, message: str) -> None:
        """Report a failed read inline. Never a modal — a dialog nobody can
        dismiss is how a headless run hangs.

        Two callers. :meth:`_on_frame_loaded` routes the ordinary case here,
        having already filled the picker from the same answer. ``job_failed``
        is the net under everything else: a bug in the delivery above, or an
        error that escaped the job entirely. Both are about the load that is
        current, which is what lets this name ``self._path``.
        """
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
        """Reload the current database at a newly chosen table.

        :param name: the table to read; a blank one, or no loaded path, does
            nothing.
        """
        if self._path and name:
            self.load_path(self._path, table=name)

    def _on_rendered(self, _data) -> None:
        """Enable the Annotate hand-off once something is brushed.

        :param _data: the render payload; the selection is re-read from the
            canvas, so it is not used.
        """
        self._to_annotate.setEnabled(self._has_merge_image_provenance() and
                                     self.builder.canvas.selected_count() > 0)

    def _open_selection(self) -> None:
        """Send the brushed objects to whatever shows crops.

        Routed through :func:`spacr.qt.linked_selection.open_objects`, so this
        screen never imports Annotate and Annotate grows no method for it.
        """
        if not self._has_merge_image_provenance():
            self._source.setText("This merge has no verified image/object provenance.")
            return
        from ..linked_selection import has_object_opener
        canvas = self.builder.canvas
        selection = canvas.link.selection
        if not selection.is_active or not len(selection):
            self._source.setText("Brush a region first — nothing is selected.")
            return
        if not has_object_opener("annotate"):
            self._source.setText(
                "Nothing can show crops yet — open the Annotate screen once.")
            return
        try:
            canvas.open_objects(
                selection.keys,
                reason=f"brushed in the Graph Builder · "
                       f"{canvas.spec.describe(canvas.kinds)}")
        except Exception as exc:
            LOG.info("could not open the brushed objects", exc_info=True)
            self._source.setText(f"could not open those objects: {exc}")

    def open_condition_dialog(self):
        """Reopen conditions for the current physical, merged or imported table."""
        if self._annotation_base_frame is None:
            return
        from ..widgets.condition_annotation_dialog import ConditionAnnotationDialog
        base = self._annotation_base_frame
        source = copy.deepcopy(self._condition_source)
        dialog = ConditionAnnotationDialog(
            base, source, self, definition=self._condition_definition, threaded=self._threaded)
        try:
            if dialog.exec() == QDialog.Accepted:
                if base is not self._annotation_base_frame or source != self._condition_source:
                    self._source.setText(tr("The current table changed while conditions were open; reopen the editor."))
                    return
                self._install_condition_frame(dialog.definition, dialog.result_frame)
                dialog.result_frame = None
        finally:
            dialog.deleteLater()

    def apply_condition_definition(self, definition):
        """Apply validated labels to the working table while preserving the source.

        :param definition: Source-bound condition rules from the annotation editor.
        :returns: Working frame including the requested output columns.
        """
        if self._annotation_base_frame is None:
            raise ValueError("Load a source table before annotating conditions.")
        frame = apply_conditions(self._annotation_base_frame, definition, self._condition_source)
        return self._install_condition_frame(definition, frame)

    def _install_condition_frame(self, definition, frame):
        """Install a validated worker result without copying the full table again."""
        self._condition_definition = copy.deepcopy(definition)
        key = json.dumps(self._condition_source, sort_keys=True)
        self._condition_definitions[key] = copy.deepcopy(definition)
        self._frame = frame
        self._update_save_annotated_button()
        self.builder.set_frame(frame)
        self.filters.set_frame(frame)
        self._source.setText(tr("Conditions applied to {rows} rows in {column}.",
                                rows=f"{len(frame):,}", column=", ".join(annotation_columns(definition))))
        return frame

    def _update_save_annotated_button(self):
        """Allow physical annotation saves only for annotated SQLite sources."""
        path = (self._condition_source or {}).get("path")
        self._save_annotated_button.setEnabled(bool(
            path and self._condition_definition and
            not path.lower().endswith((".csv", ".tsv", ".txt"))))

    def choose_save_annotated_table(self):
        """Ask for a new physical table name in the current SQLite database."""
        current = (self._condition_source or {}).get("table") or "table"
        name, accepted = QInputDialog.getText(
            self, tr("Save annotated table"),
            tr("New table name (existing tables are preserved)"), text=current + "_annotated")
        if accepted and name.strip():
            self.save_annotated_table(name)

    def save_annotated_table(self, name):
        """Create a new physical table and provenance atomically on a worker.

        :param name: New table name in the current SQLite source database.
        """
        from ...condition_annotations import save_annotated_table
        if not self._save_annotated_button.isEnabled():
            raise ValueError("Apply conditions to a SQLite table before saving it.")
        path = self._condition_source["path"]
        frame = self._frame
        definition = copy.deepcopy(self._condition_definition)
        source = copy.deepcopy(self._condition_source)
        merge = copy.deepcopy(self._merge_definition)
        self._jobs.cancel()
        self._source.setText(tr("Saving annotated table…"))
        self._jobs.submit(
            lambda: save_annotated_table(path, name, frame, definition, source, merge_definition=merge),
            lambda saved: self.load_path(path, table=saved))

    def choose_export_table(self):
        """Choose a CSV destination for the current working table and its rules."""
        path, _ = QFileDialog.getSaveFileName(
            self, tr("Export table"), "annotated-table.csv", tr("Tables (*.csv)"))
        if path:
            try:
                self.export_table(path)
                self._source.setText(tr("Table exported to {path}", path=path))
            except (OSError, ValueError) as exc:
                self._source.setText(tr("Could not export table: {error}", error=str(exc)))

    def export_table(self, path):
        """Export working values and reproducible conditions without replacing input.

        :param path: Destination CSV file. A .conditions.json sidecar stores rules.
        :returns: Destination path.

        Both files are prepared before either destination is replaced, so a
        failed CSV conversion never leaves a receipt describing an export that
        did not run.
        """
        if self._frame is None:
            raise ValueError("Load a table before exporting.")
        destination = Path(path).resolve()
        source_path = (self._condition_source or {}).get("path")
        if source_path and destination == Path(source_path).resolve():
            raise ValueError("Choose a new export path to preserve the source table.")
        sidecar = destination.with_suffix(destination.suffix + ".conditions.json")
        payload = {"source": self._condition_source,
                   "merge_definition": self._merge_definition,
                   "condition_annotation": self._condition_definition}
        temporary = []
        try:
            for target in (destination, sidecar):
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                    temporary.append(Path(handle.name))
            self._frame.to_csv(temporary[0], index=False)
            temporary[1].write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temporary[0], destination)
            os.replace(temporary[1], sidecar)
        finally:
            for candidate in temporary:
                candidate.unlink(missing_ok=True)
        return str(destination)

    def choose_save_chart(self):
        """Choose a file for the chart and its reproducible data-source definition."""
        path, _ = QFileDialog.getSaveFileName(self, "Save chart", "chart.json", "Charts (*.json)")
        if path:
            try:
                self.save_chart(path)
            except (OSError, ValueError) as exc:
                self._source.setText(f"Could not save chart: {exc}")

    def choose_load_chart(self):
        """Choose a saved chart and revalidate its source before plotting."""
        path, _ = QFileDialog.getOpenFileName(self, "Load chart", "", "Charts (*.json)")
        if path:
            try:
                self.load_chart(path)
            except (OSError, ValueError) as exc:
                self._source.setText(f"Could not load chart: {exc}")

    def save_chart(self, path):
        """Save chart channels and a source-bound merge definition when applicable.

        :param path: Destination JSON file.
        :returns: Saved path.
        """
        source_path = (self._condition_source or {}).get("path")
        if not source_path:
            raise ValueError("Load a source table before saving a chart.")
        if Path(path).resolve() == Path(source_path).resolve():
            raise ValueError("Choose a new chart path to preserve the source table.")
        payload = {"source": source_path,
                   "table": self._condition_source["table"],
                   "chart": self.builder.spec.to_dict(),
                   "merge_definition": self._merge_definition,
                   "condition_annotation": self._condition_definition}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def load_chart(self, path):
        """Reconstruct a saved chart, validating any embedded merge configuration.

        :param path: Saved chart JSON file.
        """
        from ...derived_tables import execute, save_definition
        from ..widgets.graph_spec import GraphSpec
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        source, table = payload["source"], payload.get("table")
        definition = payload.get("merge_definition")
        annotation = payload.get("condition_annotation")
        spec = GraphSpec.from_dict(payload["chart"])
        self._jobs.cancel()
        self._source.setText("Loading chart…")

        def work():
            """Reconstruct saved data on the worker before the chart is restored."""
            if definition:
                frame, _report = execute(source, definition)
            else:
                frame = read_table(source, table)
            resolved_definition = frame.attrs.get("merge_definition", definition)
            if annotation:
                annotation_base = frame
                saved = frame.attrs.get("saved_condition_definition")
                if saved:
                    annotation_base = frame.drop(columns=annotation_columns(saved))
                apply_conditions(annotation_base, annotation, source_context(source, table, resolved_definition))
            if resolved_definition:
                save_definition(source, resolved_definition)
            names = table_names(source) if not source.lower().endswith((".csv", ".tsv", ".txt")) else []
            return _Loaded(names, table, frame, None)

        def done(loaded):
            """Apply source and chart only after successful reconstruction.

            :param loaded: Revalidated source frame and available table names.
            """
            self._path = source
            resolved_definition = loaded.frame.attrs.get("merge_definition", definition)
            key = json.dumps(source_context(source, table, resolved_definition), sort_keys=True)
            if annotation:
                self._condition_definitions[key] = copy.deepcopy(annotation)
            else:
                self._condition_definitions.pop(key, None)
            self._on_frame_loaded(loaded)
            self.builder.set_spec(spec)

        self._jobs.submit(work, done)

    def closeEvent(self, event):  # noqa: N802 - Qt name
        """Stop background work and unlink before going away.

        :param event: the Qt close event.
        """
        self._jobs.shutdown()
        self.builder.close()
        super().closeEvent(event)


def make_graph_builder_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory handed to :func:`spacr.qt.app.register_app`."""
    return GraphBuilderScreen()


_ROW = declared_app(APP_KEY)
APP_NAME = _ROW.name
APP_DESCRIPTION = _ROW.desc
APP_INTRO = _ROW.intro
APP_CLI_NOTE = _ROW.cli_note
APP_NAME_TRANSLATIONS = _ROW.translations


def register() -> bool:
    """Put the Graph Builder in the app registry. Idempotent.

    Called at import from the bottom of :mod:`spacr.qt.app` — see
    ``_SELF_REGISTERING_APPS`` there. It is called from *there* rather than
    at the top of this module because ``app.py`` imports
    ``spacr.qt.widgets`` at its line 41, before ``register_app`` exists, so
    nothing reachable from the top of that file can register during its
    import; and a registration that happens later is one that some
    importer's snapshot of the registry predates.

    That used to be fatal as well as untidy, because ``SECTIONS`` was
    *rebound* rather than mutated, so a late registration into the
    previously empty Explore section was invisible to every module that had
    already imported the name. It is a list mutated in place now, so a late
    registration is seen everywhere — but registering from one deterministic
    point is still what keeps the app inventory the same on every import
    path, and the ledgers that check it honest.

    The row itself -- the key, the name, the blurb, the section, the "no
    headless run" sentence, the API doc link and the nine translations of the
    display name -- is declared in :mod:`spacr.qt.app_catalog`.
    :func:`spacr.qt.app.register_app` distributes those into the four tables
    each used to need a hand-edit in, and this function's whole job is to name
    which row. That is what lets the app be registered without importing this
    module at all: the launch reads the table, and the screen is imported when
    somebody opens it.

    :returns: ``True`` if this call is what registered it. Safe to call
        again: a module imported twice, or a test that re-imports it, must
        not raise on the duplicate key.
    """
    return register_declared(__name__) is not None



HOST_KEY = "graph_builder"

#: Registry keys of the modules folded into Graph Builder, in strip
#: order. Both ANSWER A PLOTTING QUESTION with a fixed layout, which is
#: exactly what Graph Builder does freehand -- a plate heatmap is a plot
#: whose axes are already decided, and small multiples is one plot
#: repeated over a grouping. Neither is a place to start a session, which
#: is what a Home tile says.
#:
#: `plate_view` still holds a registry row; `trellis` is declared in
#: `app_catalog` and never had one. `fold_description` reads the registry
#: first and the catalogue second, so both buttons state their own name,
#: sentence and maturity without a table here repeating them.
FOLDED_APPS: Tuple[str, ...] = ('plate_view', 'trellis')


def _build_plate_view(host_window: Optional[QWidget] = None) -> QWidget:
    """Plate View, as the window builds it."""
    from .map_barcodes import build_registered_screen

    return build_registered_screen("plate_view", host_window)


def _build_trellis(host_window: Optional[QWidget] = None) -> QWidget:
    """Trellis, as the window builds it."""
    from .map_barcodes import build_registered_screen

    return build_registered_screen("trellis", host_window)


#: One builder per folded module. :func:`install_folds` walks
#: :data:`FOLDED_APPS` and looks each key up here, so the strip's order
#: and the strip's contents cannot disagree.
BUILDERS: Dict[str, Callable[[Optional[QWidget]], QWidget]] = {
    "plate_view": _build_plate_view,
    "trellis": _build_trellis,
}


def install_folds(screen: QWidget) -> Optional["FoldStrip"]:
    """Put graph_builder's fold strip on ``screen``'s masthead.

    Reached by the one pass over the stack that serves every host --
    see :data:`spacr.qt.screens.map_barcodes.FOLD_HOST_MODULES`.
    """
    from .map_barcodes import install_fold_strip

    return install_fold_strip(screen, HOST_KEY, FOLDED_APPS, BUILDERS)
