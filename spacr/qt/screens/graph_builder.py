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
from typing import List, Optional

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSplitter,
    QVBoxLayout, QWidget,
)

from ..theme import SPACING
from ..widgets.data_filter_panel import DataFilterPanel
from ..widgets.graph_builder import GraphBuilderPanel

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
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    found = [row[0] for row in rows]
    ranked = [name for name in _PREFERRED_TABLES if name in found]
    return ranked + [name for name in found if name not in ranked]


def read_table(path: str, table: Optional[str] = None,
               limit: Optional[int] = None) -> pd.DataFrame:
    """Read a CSV or one table of a SQLite measurement database.

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
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        return pd.read_sql_query(query, db)


class GraphBuilderScreen(QWidget):
    """Drag columns onto channels; the chart follows.

    :param link: a private :class:`~spacr.qt.linked_selection.LinkedSelection`
        for tests. ``None`` joins the process-wide one, which is the point of
        the screen in normal use.
    """

    def __init__(self, parent=None, *, link=None):
        super().__init__(parent)
        self.setObjectName("GraphBuilderScreen")
        self._frame: Optional[pd.DataFrame] = None
        self._path: Optional[str] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["md"], SPACING["md"],
                                 SPACING["md"], SPACING["md"])
        outer.setSpacing(SPACING["sm"])

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(SPACING["sm"])
        title = QLabel("Graph Builder", self)
        title.setObjectName("ScreenTitle")
        head.addWidget(title)

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

    # -- data -----------------------------------------------------------
    def set_frame(self, frame: pd.DataFrame, *, label: str = "") -> None:
        """Plot ``frame``. The one call a host needs."""
        self._frame = frame
        self.builder.set_frame(frame)
        self.filters.set_frame(frame)
        self._source.setText(
            label or f"{len(frame):,} rows × {len(frame.columns)} columns")

    def choose_table(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a measurement table", "",
            "Measurements (*.db *.sqlite *.csv *.tsv);;All files (*)")
        if path:
            self.load_path(path)

    def load_path(self, path: str, table: Optional[str] = None) -> None:
        """Load a CSV or database file and plot it."""
        self._path = path
        names: List[str] = []
        if not str(path).lower().endswith((".csv", ".tsv", ".txt")):
            try:
                names = table_names(path)
            except Exception as exc:
                LOG.info("could not list tables in %s", path, exc_info=True)
                self._source.setText(f"could not read {os.path.basename(path)}: {exc}")
                return
        self._table_picker.blockSignals(True)
        self._table_picker.clear()
        self._table_picker.addItems(names)
        self._table_picker.setVisible(bool(names))
        if table and table in names:
            self._table_picker.setCurrentText(table)
        self._table_picker.blockSignals(False)
        chosen = table or (self._table_picker.currentText() or None)
        try:
            frame = read_table(path, chosen)
        except Exception as exc:
            LOG.info("could not read %s", path, exc_info=True)
            self._source.setText(f"could not read {os.path.basename(path)}: {exc}")
            return
        suffix = f" · {chosen}" if chosen else ""
        self.set_frame(
            frame,
            label=f"{os.path.basename(path)}{suffix} · {len(frame):,} rows "
                  f"× {len(frame.columns)} columns")

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
        self.builder.close()
        super().closeEvent(event)


def make_graph_builder_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory handed to :func:`spacr.qt.app.register_app`."""
    return GraphBuilderScreen()


#: Display name, one-line description, and the header copy the shipped
#: `AppScreen` tables want. Written here, next to the screen, so that wiring
#: the app in is copying four strings from one file rather than inventing
#: them in four.
APP_NAME = "Graph Builder"
APP_DESCRIPTION = "Drag columns onto x / y / colour / size / facet and get a chart"
APP_INTRO = (
    "Drop a column on X or Y and the chart appears; the plot type follows "
    "the column types. Facet down and across for small multiples on shared "
    "axes, and brush a region to highlight the same objects in every other "
    "open view.")
#: What `spacr.cli.INTERACTIVE_ONLY` wants: why this app has no headless run,
#: and what to reach for instead. Printed by `spacr-run graph_builder`, so it
#: is the only thing between a user and "I must have typed the name wrong".
APP_CLI_NOTE = (
    "Graph Builder is interactive chart building — the drop zones and the "
    "brush are the whole feature; run it in the GUI (spacr-qt). Headless, "
    "call spacr.plot from Python and pick the columns yourself.")
#: The display name in the nine non-English UI languages, in
#: `spacr.qt.i18n.LANGUAGES` order (sv, de, es, zh_CN, pt, hi, ko, is, fr).
APP_NAME_TRANSLATIONS = (
    "Diagrambyggare", "Diagramm-Baukasten", "Constructor de gráficos",
    "图表构建器", "Construtor de gráficos", "ग्राफ़ बिल्डर", "그래프 빌더",
    "Grafasmiður", "Générateur de graphiques")


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

    Everything after ``SECTION_EXPLORE`` below is a table this key used to
    need a hand-edit in: the screen header and blurb
    (``app_screen.APP_TITLES`` / ``APP_INTROS``), the "no headless run"
    sentence (``cli.INTERACTIVE_ONLY``), the API doc link
    (``settings_model._APP_API_MODULE``) and the nine translations of the
    display name (``i18n._ROWS``). :func:`spacr.qt.app.register_app`
    distributes them.

    :returns: ``True`` if this call is what registered it. Safe to call
        again: a module imported twice, or a test that re-imports it, must
        not raise on the duplicate key.
    """
    from ..app import APPS, SECTION_EXPLORE, STAGE_ALPHA, register_app
    if any(row[0] == APP_KEY for row in APPS):
        return False
    register_app(APP_KEY, APP_NAME, APP_DESCRIPTION, SECTION_EXPLORE,
                 factory=make_graph_builder_screen, stage=STAGE_ALPHA,
                 intro=APP_INTRO, cli_note=APP_CLI_NOTE,
                 api_module="qt/screens/graph_builder",
                 translations=APP_NAME_TRANSLATIONS)
    return True
