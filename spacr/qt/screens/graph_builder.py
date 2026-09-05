"""The Graph Builder screen — a table, a filter, and a chart you drag together.

Assembles three things that already exist into one surface:

* :class:`spacr.qt.widgets.graph_builder.GraphBuilderPanel` — the drop zones
  and the canvas;
* :class:`spacr.qt.widgets.data_filter_panel.DataFilterPanel` — the Local Data
  Filter, unchanged, because a filter that narrows *every* view is worth more
  than a private one that narrows this chart;
* :mod:`spacr.qt.linked_selection` — so a brush here highlights the same cells
  in the UMAP and on the plate map, and a lasso there highlights them here.

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

import logging
import os
import sqlite3
import traceback
from typing import (TYPE_CHECKING, Callable, Dict, List, NamedTuple,
                    Optional, Tuple)

import pandas as pd

if TYPE_CHECKING:
    from ..widgets.fold_strip import FoldStrip
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSplitter,
    QVBoxLayout, QWidget,
)

from ..job_runner import JobRunner
from ..theme import SPACING
from ..widgets.data_filter_panel import DataFilterPanel
from ..widgets.graph_builder import GraphBuilderPanel
from .app_screen import ModuleHeader
from ..app_catalog import declared_app, register_declared

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
    """Every user table in the SQLite file at ``path``, in a useful order."""
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as db:
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    found = [row[0] for row in rows]
    ranked = [name for name in _PREFERRED_TABLES if name in found]
    return ranked + [name for name in found if name not in ranked]


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
    query = f'SELECT * FROM "{name}"'
    if limit:
        query += f" LIMIT {int(limit)}"
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30) as db:
        return pd.read_sql_query(query, db)


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


class GraphBuilderScreen(QWidget):
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
        super().__init__(parent)
        self.setObjectName("GraphBuilderScreen")
        # ITS OWN REGISTRY KEY. Screens that build themselves rather
        # than being the generic `AppScreen` had no `app_key`, and
        # `install_folds_on` dispatches on exactly that -- so this screen
        # could declare folds (it does, below) and never be handed them.
        # Every other consumer of `app_key` reads it the same way the
        # generic screen sets it, so naming it here is the screen
        # answering a question it always could.
        self.app_key = "graph_builder"
        self._frame: Optional[pd.DataFrame] = None
        self._path: Optional[str] = None
        # Every table read goes through here, so it never runs on the GUI
        # thread and always shows up in the run registry (and so in the
        # background-activity spinner).
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

        load = QPushButton("Load table…", self)
        load.setObjectName("PrimaryButton")
        load.setToolTip("A measurements.db, or a CSV of measurements")
        load.clicked.connect(self.choose_table)
        head.addWidget(load)

        self._to_annotate = QPushButton("Open selection in Annotate", self)
        self._to_annotate.setToolTip(
            "Show the brushed objects as image crops")
        self._to_annotate.clicked.connect(self._open_selection)
        self._to_annotate.setEnabled(False)
        head.addWidget(self._to_annotate)
        outer.addLayout(head)

        body = QSplitter(Qt.Horizontal, self)
        body.setChildrenCollapsible(False)
        self.builder = GraphBuilderPanel(self, link=link)
        body.addWidget(self.builder)

        self.filters = DataFilterPanel(self, link=link)
        self.filters.setMaximumWidth(320)
        body.addWidget(self.filters)
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 0)
        outer.addWidget(body, 1)

        self.builder.canvas.rendered.connect(self._on_rendered)
        # Drop anywhere on this screen: the path is resolved through spaCR's
        # project layout, so the plate folder finds what this screen reads.
        from ..dnd import install_for
        install_for(self, "graph_builder")
        # Hover help belongs on a setting's NAME, not on the field the user
        # is about to type into (instruction 113). One post-pass rather than
        # a convention every hand-built row has to remember.
        from .settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

    # -- data -----------------------------------------------------------
    def set_frame(self, frame: pd.DataFrame, *, label: str = "") -> None:
        """Plot ``frame``. The one call a host needs."""
        self._frame = frame
        self.builder.set_frame(frame)
        self.filters.set_frame(frame)
        self._source.setText(
            label or f"{len(frame):,} rows × {len(frame.columns)} columns")

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
        Measured on the maintainer's machine 2026-09-04, a single ``stat``
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
        """
        self._path = path
        # A second load supersedes the first. Without this, switching table
        # twice in quick succession delivers the frames in whatever order the
        # reads happen to finish, and the picker ends up disagreeing with the
        # panel below it.
        self._jobs.cancel()
        # The table can only be named here when the caller already knew it --
        # the picker, or a drop that asked. Otherwise the worker is the first
        # thing that can find out, so the label says it a moment later.
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
        # Blocked, because `addItems` moves the current index and
        # `currentTextChanged` is wired to `_on_table_picked` -- unblocked
        # this populates the picker by starting another load of the table it
        # has just loaded.
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
        if self._path and name:
            self.load_path(self._path, table=name)

    # -- selection routing ------------------------------------------------
    def _on_rendered(self, _data) -> None:
        self._to_annotate.setEnabled(self.builder.canvas.selected_count() > 0)

    def _open_selection(self) -> None:
        """Send the brushed objects to whatever shows crops.

        Routed through :func:`spacr.qt.linked_selection.open_objects`, so this
        screen never imports Annotate and Annotate grows no method for it.
        """
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

    def closeEvent(self, event):  # noqa: N802 - Qt name
        # Abandon an in-flight read rather than let it outlive the
        # screen: Qt aborts the process if a running QThread is
        # destroyed, and a worker that delivers into a closed widget
        # is a use-after-free.
        """Stop background work and unlink before going away.

        :param event: the Qt close event.
        """
        self._jobs.shutdown()
        self.builder.close()
        super().closeEvent(event)


def make_graph_builder_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory handed to :func:`spacr.qt.app.register_app`."""
    return GraphBuilderScreen()


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


# ---------------------------------------------------------------------------
# Folded modules
# ---------------------------------------------------------------------------

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
    # IMPORTED HERE, like `install_fold_strip` below. This module was
    # calling `build_registered_screen` without importing it at all, so
    # both folded modules raised NameError the moment their button was
    # pressed -- reported from the Measure console.
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
