"""``V3`` — a scatter plot where every point is the cell it stands for.

A measurement scatter is thousands of dots, and the question anybody actually
has about it is *what does that one look like*. Without an answer, an outlier
is a coordinate: you can see that something at (0.9, 12.4) is unusual and you
cannot tell whether it is a real phenotype, a debris fragment or two cells
segmented as one. That judgement is the whole reason to look at a plot of
images rather than a plot of numbers, and it needs the image.

So: **hover shows the crop, click opens it.** The click goes through
:func:`spacr.qt.linked_selection.open_objects`, so this screen does not import
the annotation grid and the grid grows no method for it.

Not stuttering is a design constraint, not an optimisation
----------------------------------------------------------

A hover handler runs sixty-plus times a second and a PNG decode costs
milliseconds, so the obvious implementation — decode the crop under the cursor
in ``mouseMoveEvent`` — produces a plot that lags behind the mouse and reads
as broken. Three things keep it honest, and all three are load-bearing:

* The point cloud is painted once into a pixmap and blitted. Hover repaints a
  ring, never ninety thousand dots.
* Crop paths are resolved for the whole plot in one database pass at load
  time (:func:`spacr.qt.crop_thumbs.crop_paths_for_keys`), so hover is a dict
  lookup rather than a query.
* Decoding is behind a short debounce and an LRU
  (:class:`spacr.qt.crop_thumbs.CropThumbnails`). Sweeping the cursor across
  a cluster decodes what the cursor *stopped* on, not everything it crossed;
  a crop already in the cache appears with no delay at all.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFrame, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QSizePolicy,
                               QSplitter, QVBoxLayout, QWidget)

from ...selection import (OBJECT_KEY_COLUMNS, match_keys, object_keys,
                          with_object_type)
from ..crop_thumbs import CropThumbnails, crop_paths_for_keys
from ..job_runner import JobRunner
from ..linked_selection import DEFAULT_OPEN_KIND, LinkedView, has_object_opener
from ..theme import SPACING, active_palette

LOG = logging.getLogger(__name__)

__all__ = [
    "ScatterCanvas",
    "ImageScatterScreen",
    "load_scatter_frame",
    "register",
    "APP_KEY",
    "LINK_SOURCE",
]

#: The app key this screen registers under. Stable; keyed off by saved state.
APP_KEY = "image_scatter"

#: What this view calls itself on the shared selection.
LINK_SOURCE = "image_scatter"

#: How near the cursor has to be, in widget pixels, for a point to count as
#: hovered. Generous: a 3 px dot needs a bigger target than 3 px or the
#: preview flickers on and off as the hand shakes.
HIT_RADIUS = 12.0

#: How long the cursor must rest before a crop is decoded, in ms. Short
#: enough to feel immediate, long enough that crossing a dense cluster does
#: not queue two hundred decodes.
HOVER_DELAY_MS = 70

#: Radius of a plotted point, in pixels.
POINT_RADIUS = 2.5


# ---------------------------------------------------------------------------
# Loading — off the GUI thread, and with no widget in sight
# ---------------------------------------------------------------------------

def load_scatter_frame(db_path: str, table: str,
                       *, limit: int = 200_000) -> pd.DataFrame:
    """Read one measurement table, ready to plot. Runs on a worker thread.

    Deliberately a module-level function taking strings: it must never touch a
    widget, and the surest way to guarantee that is for it not to have one.

    :param limit: a hard row cap. A million-row table plotted at three pixels
        a point is a solid rectangle, so the cap costs no information and
        keeps a mis-click on the wrong table from being a two-minute freeze.
    :raises FileNotFoundError: no such database — said plainly, because an
        empty plot from a wrong path looks exactly like an empty table.
    """
    if not db_path or not os.path.isfile(db_path):
        raise FileNotFoundError(f"no measurements database at {db_path!r}")
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        frame = pd.read_sql_query(
            f'SELECT * FROM "{str(table)}" LIMIT {int(limit)}', connection)
    finally:
        connection.close()
    # The frame does not know what it is; this function does. Without the
    # stamp a point in the nucleus table and a point in the pathogen table
    # publish the same key when they share a label, and clicking one opens
    # whichever crop the table happened to list first.
    return with_object_type(frame, table)


def list_tables(db_path: str) -> List[str]:
    """Every table in ``db_path``, sorted. Worker-thread safe."""
    if not db_path or not os.path.isfile(db_path):
        return []
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    finally:
        connection.close()
    return [str(row[0]) for row in rows]


def numeric_columns(frame: pd.DataFrame) -> List[str]:
    """Columns worth putting on an axis, in the frame's own order."""
    return [str(c) for c in frame.columns
            if pd.api.types.is_numeric_dtype(frame[c])]


# ---------------------------------------------------------------------------
# The canvas
# ---------------------------------------------------------------------------

class ScatterCanvas(QFrame):
    """Points in data space, painted once and blitted.

    Kept separate from the screen so the hit-testing and the caching can be
    tested without a database, and so a second consumer (a facetted view, a
    comparison grid) can reuse it.
    """

    #: The point under the cursor changed. Carries its index, or ``-1``.
    hover_changed = Signal(int)
    #: A point was clicked. Carries its index.
    point_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ImageScatterCanvas")
        self.setMouseTracking(True)
        self.setMinimumSize(240, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._x = np.zeros(0, dtype=float)
        self._y = np.zeros(0, dtype=float)
        self._px = np.zeros(0, dtype=float)
        self._py = np.zeros(0, dtype=float)
        self._cloud: Optional[QPixmap] = None
        self._hover = -1
        self._selected = np.zeros(0, dtype=bool)
        self._x_label = ""
        self._y_label = ""

    # -- data ---------------------------------------------------------------
    def set_points(self, x: Sequence[float], y: Sequence[float], *,
                   x_label: str = "", y_label: str = "") -> int:
        """Plot ``x`` against ``y``. Returns how many points are plottable.

        Rows with a non-finite coordinate are kept in the arrays (so indices
        still line up with the caller's frame) but never drawn and never hit —
        dropping them would silently renumber every point and make a click
        open the wrong object.
        """
        self._x = np.asarray(x, dtype=float).ravel()
        self._y = np.asarray(y, dtype=float).ravel()
        if len(self._x) != len(self._y):
            raise ValueError(
                f"a scatter needs one y per x, got {len(self._x)} and "
                f"{len(self._y)}")
        self._x_label = str(x_label)
        self._y_label = str(y_label)
        self._selected = np.zeros(len(self._x), dtype=bool)
        self._hover = -1
        self._invalidate()
        return int(np.count_nonzero(self.plottable))

    @property
    def plottable(self) -> np.ndarray:
        """Mask of the points with a finite coordinate pair."""
        if not len(self._x):
            return np.zeros(0, dtype=bool)
        return np.isfinite(self._x) & np.isfinite(self._y)

    def set_selected(self, mask: Sequence[bool]) -> None:
        """Ring these points. A selection highlights; it never hides."""
        selected = np.asarray(mask, dtype=bool).ravel()
        if len(selected) != len(self._x):
            selected = np.zeros(len(self._x), dtype=bool)
        self._selected = selected
        self.update()

    @property
    def hovered(self) -> int:
        """Index of the point under the cursor, or ``-1``."""
        return self._hover

    def __len__(self) -> int:
        return int(len(self._x))

    # -- geometry -----------------------------------------------------------
    def _invalidate(self) -> None:
        self._cloud = None
        self._px = np.zeros(0, dtype=float)
        self._py = np.zeros(0, dtype=float)
        self.update()

    def _project(self) -> None:
        """Data → widget pixels, once per resize or data change."""
        good = self.plottable
        n = len(self._x)
        self._px = np.full(n, np.nan)
        self._py = np.full(n, np.nan)
        if not n or not good.any():
            return
        pad = 18.0
        width = max(1.0, self.width() - 2 * pad)
        height = max(1.0, self.height() - 2 * pad)
        xs, ys = self._x[good], self._y[good]
        x0, x1 = float(np.min(xs)), float(np.max(xs))
        y0, y1 = float(np.min(ys)), float(np.max(ys))
        # A constant column would divide by zero; centre it instead of
        # collapsing every point onto the left edge.
        sx = width / (x1 - x0) if x1 > x0 else 0.0
        sy = height / (y1 - y0) if y1 > y0 else 0.0
        self._px[good] = (pad + (xs - x0) * sx if sx else pad + width / 2.0)
        # Screen y grows downward; data y grows upward.
        self._py[good] = (pad + height - (ys - y0) * sy if sy
                          else pad + height / 2.0)

    def _build_cloud(self) -> QPixmap:
        """Paint every point once. Hover must never repaint this."""
        self._project()
        pixmap = QPixmap(max(1, self.width()), max(1, self.height()))
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            palette = active_palette()
            colour = QColor(palette.get("accent", "#4c9aff"))
            colour.setAlpha(170)
            painter.setPen(Qt.NoPen)
            painter.setBrush(colour)
            good = self.plottable
            for index in np.flatnonzero(good):
                painter.drawEllipse(
                    QPointF(float(self._px[index]), float(self._py[index])),
                    POINT_RADIUS, POINT_RADIUS)
        finally:
            painter.end()
        return pixmap

    def resizeEvent(self, event) -> None:
        self._invalidate()
        super().resizeEvent(event)

    # -- painting -----------------------------------------------------------
    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        try:
            if not len(self._x):
                painter.setPen(Qt.gray)
                painter.drawText(self.rect(), Qt.AlignCenter,
                                 "No points yet")
                return
            if self._cloud is None:
                self._cloud = self._build_cloud()
            painter.drawPixmap(0, 0, self._cloud)
            painter.setRenderHint(QPainter.Antialiasing, True)
            palette = active_palette()
            if self._selected.any():
                painter.setPen(QPen(QColor(palette.get("warning", "#ffb020")),
                                    1.5))
                painter.setBrush(Qt.NoBrush)
                for index in np.flatnonzero(self._selected & self.plottable):
                    painter.drawEllipse(
                        QPointF(float(self._px[index]), float(self._py[index])),
                        POINT_RADIUS + 3.0, POINT_RADIUS + 3.0)
            if 0 <= self._hover < len(self._px) and np.isfinite(
                    self._px[self._hover]):
                painter.setPen(QPen(QColor(palette.get("fg", "#ffffff")), 2.0))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(
                    QPointF(float(self._px[self._hover]),
                            float(self._py[self._hover])),
                    POINT_RADIUS + 5.0, POINT_RADIUS + 5.0)
            if self._x_label or self._y_label:
                painter.setPen(QColor(palette.get("fg_muted", "#9aa0a6")))
                painter.drawText(
                    QRectF(4, self.height() - 18, self.width() - 8, 16),
                    int(Qt.AlignRight | Qt.AlignVCenter),
                    f"{self._x_label} ×  {self._y_label} ↑")
        except Exception:
            # A paintEvent that raises takes the window with it.
            LOG.exception("Could not paint the image scatter")
        finally:
            painter.end()

    # -- hit testing --------------------------------------------------------
    def _ensure_projection(self) -> None:
        """Make sure widget coordinates exist before anything reads them.

        Hit-testing can be asked for before the first paint (a synthetic
        click in a test, a screen that opens with the cursor already inside
        it), and an un-projected canvas would answer "nothing here" for every
        point it is in fact showing.
        """
        if len(self._px) != len(self._x):
            self._project()

    def point_position(self, index: int) -> Optional[Tuple[float, float]]:
        """Widget coordinates of point ``index``, or ``None`` if not drawn."""
        self._ensure_projection()
        if not (0 <= index < len(self._px)) or not np.isfinite(
                self._px[index]):
            return None
        return float(self._px[index]), float(self._py[index])

    def index_at(self, x: float, y: float) -> int:
        """Nearest plotted point within :data:`HIT_RADIUS`, or ``-1``.

        One vectorised pass. A spatial index would be faster asymptotically
        and is not worth the invalidation rules: at 200 000 points this is
        well under a millisecond, and the row cap in
        :func:`load_scatter_frame` keeps it there.
        """
        self._ensure_projection()
        if not len(self._px):
            return -1
        dx = self._px - float(x)
        dy = self._py - float(y)
        distance = np.hypot(dx, dy)
        distance[~np.isfinite(distance)] = np.inf
        best = int(np.argmin(distance))
        return best if distance[best] <= HIT_RADIUS else -1

    def _set_hover(self, index: int) -> None:
        if index == self._hover:
            return
        self._hover = int(index)
        self.update()
        self.hover_changed.emit(self._hover)

    def mouseMoveEvent(self, event) -> None:
        position = event.position()
        self._set_hover(self.index_at(position.x(), position.y()))

    def leaveEvent(self, event) -> None:
        self._set_hover(-1)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        position = event.position()
        index = self.index_at(position.x(), position.y())
        if index >= 0:
            self._set_hover(index)
            self.point_clicked.emit(index)


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------

class ImageScatterScreen(LinkedView, QWidget):
    """A measurement scatter with the crop under the cursor beside it.

    :param threaded: ``False`` loads inline, so a test drives the screen
        without a worker thread and gets the same signals in the same order.
    """

    #: A point's crop was shown. Carries the object key.
    crop_shown = Signal(str)

    def __init__(self, parent=None, *, threaded: bool = True):
        super().__init__(parent)
        self.setObjectName("ImageScatterScreen")
        self._jobs = JobRunner(self, threaded=bool(threaded),
                               app_key="image scatter")
        self._jobs.job_failed.connect(self._on_job_failed)
        self._frame = pd.DataFrame()
        self._keys: List[str] = []
        self._paths: Dict[str, str] = {}
        self._thumbs = CropThumbnails()
        # Debounce: the cursor crossing a cluster must not queue a decode per
        # point it passed over. Only what it rests on is worth an image.
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(HOVER_DELAY_MS)
        self._hover_timer.timeout.connect(self._show_hovered_crop)
        self._pending_hover = -1
        self._build()
        self.link_selection(LINK_SOURCE)

    # -- construction -------------------------------------------------------
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["sm"])

        title = QLabel("Image Scatter", self)
        title.setObjectName("DisplayHeading")
        outer.addWidget(title)
        intro = QLabel(
            "Hover a point to see the object it stands for; click it to open "
            "the crop. An outlier you cannot look at is only a coordinate.",
            self)
        intro.setObjectName("Muted")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        source = QHBoxLayout()
        source.addWidget(QLabel("measurements.db", self))
        self._db = QLineEdit(self)
        self._db.setPlaceholderText("…/measurements/measurements.db")
        self._db.returnPressed.connect(self.open_source)
        source.addWidget(self._db, 1)
        self._browse = QPushButton("Browse…", self)
        self._browse.clicked.connect(self._choose_db)
        source.addWidget(self._browse)
        self._table = QComboBox(self)
        self._table.setMinimumWidth(140)
        source.addWidget(self._table)
        self._load = QPushButton("Plot", self)
        self._load.setObjectName("PrimaryButton")
        self._load.clicked.connect(self.load_table)
        source.addWidget(self._load)
        outer.addLayout(source)

        axes = QHBoxLayout()
        axes.addWidget(QLabel("x", self))
        self._x_choice = QComboBox(self)
        self._x_choice.setMinimumWidth(180)
        self._x_choice.currentTextChanged.connect(self._replot)
        axes.addWidget(self._x_choice, 1)
        axes.addWidget(QLabel("y", self))
        self._y_choice = QComboBox(self)
        self._y_choice.setMinimumWidth(180)
        self._y_choice.currentTextChanged.connect(self._replot)
        axes.addWidget(self._y_choice, 1)
        outer.addLayout(axes)

        split = QSplitter(Qt.Horizontal, self)
        self.canvas = ScatterCanvas(self)
        self.canvas.hover_changed.connect(self._on_hover)
        self.canvas.point_clicked.connect(self._on_click)
        split.addWidget(self.canvas)

        side = QWidget(self)
        column = QVBoxLayout(side)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        self.preview = QLabel(side)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(200, 200)
        self.preview.setText("Hover a point")
        column.addWidget(self.preview)
        self.caption = QLabel("", side)
        self.caption.setObjectName("Muted")
        self.caption.setWordWrap(True)
        column.addWidget(self.caption)
        self._open_button = QPushButton("Open this crop", side)
        self._open_button.setEnabled(False)
        self._open_button.clicked.connect(self._open_hovered)
        column.addWidget(self._open_button)
        column.addStretch(1)
        split.addWidget(side)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([700, 240])
        outer.addWidget(split, 1)

        self.status = QLabel("", self)
        self.status.setObjectName("Muted")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

    # -- source -------------------------------------------------------------
    def _choose_db(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a measurements database", self._db.text().strip(),
            "SQLite (*.db *.sqlite);;All files (*)")
        if path:
            self._db.setText(path)
            self.open_source()

    def open_source(self) -> None:
        """List the tables in the chosen database, off the GUI thread."""
        db_path = self._db.text().strip()
        self._thumbs = CropThumbnails(db_path)
        self._jobs.submit(lambda: list_tables(db_path), self._on_tables)

    def _on_tables(self, tables: List[str]) -> None:
        """Fill the table picker. Runs on the GUI thread."""
        self._table.clear()
        self._table.addItems(list(tables))
        self.status.setText(
            f"{len(tables)} table(s). Choose one and select Plot."
            if tables else "No tables — is that a measurements database?")

    def load_table(self) -> None:
        """Read the chosen table and resolve every point's crop, once."""
        db_path = self._db.text().strip()
        table = self._table.currentText().strip()
        if not table:
            self.status.setText("Choose a table first.")
            return
        self.status.setText(f"Reading {table}…")
        self._jobs.submit(lambda: self._read(db_path, table), self._on_loaded)

    @staticmethod
    def _read(db_path: str, table: str) -> Dict[str, Any]:
        """Worker body: the frame, its keys and its crop paths.

        All three in one job, deliberately. The crop resolution is the
        expensive part and it belongs where the frame is, not on the GUI
        thread once the plot is already up.
        """
        frame = load_scatter_frame(db_path, table)
        keys: List[str] = []
        paths: Dict[str, str] = {}
        if all(column in frame.columns for column in OBJECT_KEY_COLUMNS):
            keys = [str(k) for k in object_keys(frame)]
            paths = crop_paths_for_keys(db_path, keys)
        return {"frame": frame, "keys": keys, "paths": paths, "table": table}

    def _on_loaded(self, payload: Dict[str, Any]) -> None:
        """Put a freshly read table on the axes. Runs on the GUI thread."""
        if not payload:
            return
        self.set_frame(payload["frame"], keys=payload["keys"],
                       paths=payload["paths"], note=payload.get("table", ""))

    def _on_job_failed(self, message: str) -> None:
        self.status.setText(message)
        self.status.setStyleSheet(f"color: {active_palette()['error']};")

    # -- the frame ----------------------------------------------------------
    def set_frame(self, frame: pd.DataFrame, *,
                  keys: Optional[Sequence[str]] = None,
                  paths: Optional[Dict[str, str]] = None,
                  x: str = "", y: str = "", note: str = "") -> None:
        """Plot ``frame``. The seam a test — or another screen — goes through.

        :param keys: one object key per row. Derived from the frame when it
            carries :data:`spacr.selection.OBJECT_KEY_COLUMNS` and omitted.
            Without keys the plot still draws, but a click cannot open
            anything and says so rather than doing nothing.
        :param paths: ``{key: crop path}``, resolved once by the caller.
        """
        self._frame = frame if isinstance(frame, pd.DataFrame) \
            else pd.DataFrame()
        if keys is not None:
            self._keys = [str(k) for k in keys]
        elif all(c in self._frame.columns for c in OBJECT_KEY_COLUMNS):
            self._keys = [str(k) for k in object_keys(self._frame)]
        else:
            self._keys = []
        self._paths = dict(paths or {})
        if not self._paths:
            # A frame read straight from a crop table already carries the
            # path; use it rather than going back to the database.
            for column in ("png_path", "sample", "path"):
                if column in self._frame.columns and self._keys:
                    self._paths = {
                        key: str(value) for key, value
                        in zip(self._keys, self._frame[column])}
                    break
        options = numeric_columns(self._frame)
        for combo, wanted, fallback in ((self._x_choice, x, 0),
                                        (self._y_choice, y, 1)):
            blocked = combo.blockSignals(True)
            try:
                combo.clear()
                combo.addItems(options)
                if wanted and wanted in options:
                    combo.setCurrentText(wanted)
                elif len(options) > fallback:
                    combo.setCurrentIndex(fallback)
            finally:
                combo.blockSignals(blocked)
        self._replot()
        self.status.setText(
            f"{len(self._frame)} row(s)"
            + (f" from {note}" if note else "")
            + (f" · {len(self._paths)} with a crop" if self._keys
               else " · no object keys, so points cannot be opened"))

    def _replot(self, *_args) -> None:
        """Redraw from the current axis choices, honouring the shared filter."""
        x_name = self._x_choice.currentText()
        y_name = self._y_choice.currentText()
        if self._frame.empty or not x_name or not y_name:
            self.canvas.set_points([], [])
            return
        self.canvas.set_points(
            pd.to_numeric(self._frame[x_name], errors="coerce").to_numpy(),
            pd.to_numeric(self._frame[y_name], errors="coerce").to_numpy(),
            x_label=x_name, y_label=y_name)
        self._apply_linked_selection()

    def key_at(self, index: int) -> str:
        """The object key of point ``index``, or ``""``."""
        if 0 <= index < len(self._keys):
            return self._keys[index]
        return ""

    def path_at(self, index: int) -> str:
        """The crop path of point ``index``, or ``""``."""
        return self._paths.get(self.key_at(index), "")

    # -- hover --------------------------------------------------------------
    def _on_hover(self, index: int) -> None:
        """The cursor moved onto (or off) a point.

        A crop that is already decoded is shown immediately — waiting 70 ms to
        blit a pixmap that is sitting in memory is a lag nobody asked for. One
        that is not is left to the timer.
        """
        self._pending_hover = int(index)
        if index < 0:
            self._hover_timer.stop()
            self._clear_preview()
            return
        path = self.path_at(index)
        cached = self._thumbs.peek(path) if path else None
        if cached is not None:
            self._hover_timer.stop()
            self._show_crop(index, cached)
            return
        self._hover_timer.start()

    def _show_hovered_crop(self) -> None:
        """The cursor rested: decode what it is on, once."""
        index = self._pending_hover
        if index < 0:
            return
        path = self.path_at(index)
        if not path:
            self._show_crop(index, None)
            return
        self._show_crop(index, self._thumbs.prime(path))

    def _show_crop(self, index: int, pixmap: Optional[QPixmap]) -> None:
        key = self.key_at(index)
        if pixmap is not None and not pixmap.isNull():
            self.preview.setPixmap(pixmap)
        else:
            self.preview.setText(
                "no crop for this object" if key else "no object key")
        self.caption.setText(key or f"point {index}")
        self._open_button.setEnabled(
            bool(key) and has_object_opener(DEFAULT_OPEN_KIND))
        if key:
            self.crop_shown.emit(key)

    def _clear_preview(self) -> None:
        self.preview.clear()
        self.preview.setText("Hover a point")
        self.caption.setText("")
        self._open_button.setEnabled(False)

    # -- click --------------------------------------------------------------
    def _on_click(self, index: int) -> None:
        """A point was clicked: highlight it everywhere, and open its crop.

        Two acts, and both happen, because they answer different questions.
        Publishing puts the same object under a ring on the plate view and the
        UMAP; opening puts the crop on screen. The router deliberately does
        not do the first for you (see
        :meth:`spacr.qt.linked_selection.LinkedSelection.open_request`), so a
        view that wants both says so.
        """
        key = self.key_at(index)
        if not key:
            self.status.setText(
                "This table has no object keys, so a point cannot be opened. "
                "Plot a table that carries plateID/rowID/columnID/fieldID/"
                "object_label.")
            return
        self.publish_selection([key])
        self.open_point(index)

    def open_point(self, index: int) -> Any:
        """Route point ``index``'s object to whatever shows crops.

        :returns: what the opener returned, or ``None`` when there is no key
            or nowhere to open it.
        """
        key = self.key_at(index)
        if not key or not has_object_opener(DEFAULT_OPEN_KIND):
            return None
        x_name = self._x_choice.currentText()
        y_name = self._y_choice.currentText()
        try:
            return self.open_objects(
                [key],
                reason=f"clicked in the scatter of {y_name} against {x_name}",
                context={"x": x_name, "y": y_name, "index": int(index)})
        except Exception as exc:
            LOG.exception("Could not open %s", key)
            self.status.setText(f"Could not open {key}: {exc}")
            return None

    def _open_hovered(self) -> Any:
        return self.open_point(self.canvas.hovered)

    # -- the shared selection ------------------------------------------------
    def on_linked_selection_changed(self, selection) -> None:
        """Ring the points another view selected."""
        self._apply_linked_selection(selection)

    def _apply_linked_selection(self, selection=None) -> None:
        selection = selection if selection is not None else self.link.selection
        if selection.keys is None or not self._keys:
            self.canvas.set_selected([])
            return
        # Matched by specificity rather than by equality: a view that states
        # no object type still has to light up for one that does, and the
        # other way round. See `spacr.selection.match_keys`.
        self.canvas.set_selected(
            list(match_keys(self._keys, selection.keys)))

    def on_linked_filter_changed(self, data_filter) -> None:
        """The population narrowed: say so. A filter hides, so re-plot.

        Applied to the *plotted* frame rather than to the source, so clearing
        the filter widens it back without another database read.
        """
        if self._frame.empty:
            return
        self.status.setText(f"Filter: {data_filter.describe()}")
        self._replot()

    def closeEvent(self, event) -> None:
        self._hover_timer.stop()
        self.unlink_selection()
        self._jobs.cancel()
        self._thumbs.clear()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

APP_NAME = "Image Scatter"
APP_DESCRIPTION = "Hover a point to see the cell; click it to open the crop"
APP_INTRO = (
    "Any two measurements against each other, with the object under the "
    "cursor shown beside the plot. An outlier you cannot look at is only a "
    "coordinate — this is how you find out whether it is a phenotype, a "
    "debris fragment or two cells segmented as one. Clicking a point opens "
    "its crop in the annotation grid and highlights it in every other open "
    "view.")
APP_CLI_NOTE = (
    "Image Scatter is an interactive plot — the hover preview is the whole "
    "feature; run it in the GUI (spacr-qt). Headless, read the same table "
    "with pandas.")


def make_image_scatter_screen(**_kwargs) -> ImageScatterScreen:
    """Build the screen. The ``factory=`` for :func:`spacr.qt.app.register_app`."""
    return ImageScatterScreen()


def register(*, section: Optional[str] = None, stage: Optional[str] = None,
             key: str = APP_KEY):
    """Put Image Scatter in the app registry. Idempotent.

    :returns: the registry row, or ``None`` when the key was already there.
    """
    from ..app import APPS, SECTION_EXPLORE, STAGE_ALPHA, register_app
    if any(row[0] == key for row in APPS):
        return None
    return register_app(
        key, APP_NAME, APP_DESCRIPTION, section or SECTION_EXPLORE,
        factory=make_image_scatter_screen,
        stage=STAGE_ALPHA if stage is None else stage,
        intro=APP_INTRO, cli_note=APP_CLI_NOTE,
        api_module="qt/screens/image_scatter",
        translations=("Bildspridning", "Bild-Streudiagramm",
                      "Dispersión de imágenes", "图像散点图",
                      "Dispersão de imagens", "छवि स्कैटर", "이미지 산점도",
                      "Myndadreifing", "Nuage de points d'images"))
