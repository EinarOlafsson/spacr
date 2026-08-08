"""Interactive Image UMAP viewer with click, lasso, and DB annotation.

The lasso is also the cheapest way spaCR has of asking "and where are those
cells on the plate?", so the explorer joins the shared selection through
:class:`spacr.qt.linked_selection.LinkedView`: a lasso publishes the objects
it caught, and a selection made anywhere else lights up the same points here.

The two directions are deliberately asymmetric, because a filter and a
selection are not the same thing:

* an incoming **selection** draws a ring around the matching points and
  changes nothing else. It never removes a point, and it never becomes the
  local lasso — "Label lasso selection" keeps meaning *the lasso drawn here*,
  or a highlight arriving from the database browser could write annotations
  the user never drew.
* an incoming **filter** DIMS the points it excludes. Removing them would
  redraw the embedding around the survivors, and a UMAP whose axes move when
  you tick a checkbox is unreadable — the whole value of the projection is
  that a point stays where it was.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from PIL.ImageQt import ImageQt

from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QSplitter, QVBoxLayout,
    QWidget,
)

from ... import schema
from ...selection import (OBJECT_KEY_COLUMNS, DataFilter, Selection,
                          match_keys, object_keys)
from ...umap_annotations import write_umap_annotations
from ..linked_selection import LinkedView

LOG = logging.getLogger("spacr.qt.umap_explorer")

#: How much of its opacity a point keeps when the shared filter excludes it.
#: Low enough to read as "not in the population", high enough that the shape
#: of what was filtered out is still visible — which is most of the point of
#: dimming rather than hiding.
DIMMED_ALPHA = 0.12


def _usable(value) -> bool:
    """Whether ``value`` is a real identity token rather than a gap.

    ``None``, ``NaN`` (which is what a missing sqlite value becomes once it
    has been through pandas) and blank strings are all "this record does not
    say", and each of them turns into the literal key ``'nan'`` if it reaches
    :func:`~spacr.selection.object_keys`.
    """
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False
    return str(value).strip() != ""


def _record_identity(record: Dict) -> Optional[Dict[str, str]]:
    """The object key columns for one UMAP record, or ``None``.

    Two sources, in order of trust:

    1. the key columns spelled out on the record;
    2. its ``prcfo``, which :func:`spacr.core.generate_image_umap` copies
       from the measurement row.

    ``prcfo`` is *not* an object key: it spells the object as ``'o7'`` where
    every object table stores ``object_label`` bare. Parsing it and rebuilding
    the columns is what keeps a lasso here naming the same rows as a
    selection made in the database browser.
    """
    known = {c: record.get(c) for c in OBJECT_KEY_COLUMNS}
    if all(_usable(v) for v in known.values()):
        return {c: str(v) for c, v in known.items()}
    text = record.get(schema.PRCFO_KEY)
    if not _usable(text):
        return None
    try:
        obj = schema.parse_prcfo(text)
    except Exception:
        return None
    return {
        schema.PLATE_KEY: obj.plateID,
        schema.ROW_KEY: obj.rowID,
        schema.COLUMN_KEY: obj.columnID,
        schema.FIELD_KEY: obj.fieldID,
        schema.OBJECT_LABEL_KEY: schema.strip_prefix(
            obj.objectID, schema.OBJECT_PREFIX),
    }


class _AnnotationWorker(QThread):
    """Commit a selection without blocking the Qt event loop on SQLite."""

    finished_result = Signal(int, int, str)

    def __init__(self, records, values, column, parent=None):
        super().__init__(parent)
        self._records = list(records)
        self._values = list(values)
        self._column = column

    def run(self):
        try:
            updated, skipped = write_umap_annotations(
                self._records, self._values, self._column)
            self.finished_result.emit(updated, skipped, "")
        except Exception as exc:
            LOG.info("UMAP annotation write failed", exc_info=True)
            self.finished_result.emit(0, len(self._records), str(exc))


class UmapDisplaySettings(QDialog):
    """One window holding every Image UMAP display setting.

    Some apply to the figure on screen and some cannot, and the window says
    which rather than leaving the user to discover it. Asked for exactly
    that way: "the other settings can also be in the same settings window
    even though they cannot be live applied."

    The ones that cannot are not disabled -- they are editable, saved, and
    take effect on the next run. A greyed control that holds a value the
    user wants to change is worse than a live one with a note beside it.
    """

    #: ``key -> (label, kind, low, high, live)``. ``live`` decides which
    #: half of the form the row lands in, and nothing else.
    FIELDS = (
        ("point_size",     "Dot size",        "int",   1, 400,  True),
        ("point_alpha",    "Dot opacity",     "float", 0.0, 1.0, True),
        ("outline_width",  "Outline width",   "float", 0.0, 10.0, True),
        ("point_color",    "Dot colour",      "text",  0, 0,    True),
        ("canvas_width",   "Canvas width",    "int",   200, 4000, True),
        ("sidebar_width",  "Sidebar width",   "int",   120, 2000, True),
        ("figuresize",     "Figure size",     "float", 1.0, 60.0, False),
        ("image_nr",       "Images shown",    "int",   0, 100000, False),
        ("img_zoom",       "Image zoom",      "float", 0.001, 5.0, False),
    )

    def __init__(self, values: Dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image UMAP display settings")
        self._editors: Dict[str, QWidget] = {}

        outer = QVBoxLayout(self)
        live_form = QFormLayout()
        later_form = QFormLayout()

        for key, label, kind, low, high, live in self.FIELDS:
            editor = self._editor(kind, low, high, values.get(key))
            self._editors[key] = editor
            (live_form if live else later_form).addRow(label, editor)

        outer.addWidget(QLabel("<b>Applies now</b>"))
        outer.addLayout(live_form)
        note = QLabel("<b>Applies on the next run</b><br>"
                      "<span style='color:gray;'>These decide what gets "
                      "drawn, so they need the run that draws it.</span>")
        note.setWordWrap(True)
        outer.addWidget(note)
        outer.addLayout(later_form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok
                                   | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    @staticmethod
    def _editor(kind: str, low, high, value):
        if kind == "int":
            box = QSpinBox()
            box.setRange(int(low), int(high))
            if value is not None:
                box.setValue(int(float(value)))
            return box
        if kind == "float":
            box = QDoubleSpinBox()
            box.setDecimals(3)
            box.setRange(float(low), float(high))
            if value is not None:
                box.setValue(float(value))
            return box
        edit = QLineEdit()
        if value is not None:
            edit.setText(str(value))
        return edit

    def values(self) -> Dict:
        """What the user set, keyed as the settings dict keys."""
        out: Dict = {}
        for key, editor in self._editors.items():
            if isinstance(editor, QLineEdit):
                out[key] = editor.text().strip()
            else:
                out[key] = editor.value()
        return out

    def live_values(self) -> Dict:
        """Only the half that can reach the figure already on screen."""
        live = {key for key, _l, _k, _lo, _hi, is_live in self.FIELDS
                if is_live}
        return {k: v for k, v in self.values().items() if k in live}


class ImageUmapExplorer(LinkedView, QWidget):
    """Zoomable embedding: click a point, lasso a group, write labels.

    Linked to the shared selection as ``"umap"``. See the module docstring
    for why an incoming selection highlights and an incoming filter dims.
    """

    annotation_finished = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._embedding = np.empty((0, 2), dtype=float)
        self._labels = np.empty(0, dtype=int)
        self._records: List[Dict] = []
        self._selected = np.empty(0, dtype=int)
        self._picked: Optional[int] = None
        self._worker: Optional[_AnnotationWorker] = None
        #: One frame row per point, carrying whatever identity the payload
        #: could give it — the object key columns, plus any extra columns a
        #: caller attached so the shared filter has something to filter on.
        #: ``None`` when the payload named no objects at all.
        self._point_frame: Optional[pd.DataFrame] = None
        #: The object key of each point, aligned to ``_embedding``.
        self._point_keys: Optional[pd.Index] = None
        #: False for points the shared filter excludes. All-True when there
        #: is no filter, or when this payload cannot honour the one there is.
        self._point_visible = np.ones(0, dtype=bool)
        #: Points named by a selection published elsewhere.
        self._linked_points = np.empty(0, dtype=int)
        #: Set when this payload cannot answer the active filter, so the
        #: status line can say so rather than silently drawing everything.
        self._filter_note = ""
        self._display = {
            "point_size": 26,
            "point_color": "cluster",
            "point_alpha": 0.65,
            "outline_width": 1.0,
            "canvas_width": 900,
            "sidebar_width": 280,
        }
        self._build_ui()
        # After the UI: both hooks repaint, and a filter can already be set
        # by the time this screen opens.
        self.link_selection("umap")

    def _build_ui(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT)

        class _OwnedTimerFigureCanvas(FigureCanvasQTAgg):
            """Figure canvas whose deferred draw cannot outlive the widget.

            Matplotlib's Qt canvas uses static ``QTimer.singleShot`` calls.
            Those callbacks are not owned by the canvas and can consequently
            run after Qt has deleted it. An owned timer is destroyed together
            with the canvas, so lasso/display updates cannot draw a dangling
            C++ object.
            """

            def __init__(self, figure):
                super().__init__(figure)
                self._spacr_draw_timer = QTimer(self)
                self._spacr_draw_timer.setSingleShot(True)
                self._spacr_draw_timer.timeout.connect(self._spacr_draw)

            def draw_idle(self):
                self._draw_pending = True
                if not self._spacr_draw_timer.isActive():
                    self._spacr_draw_timer.start(0)

            def _spacr_draw(self):
                if not self._draw_pending:
                    return
                self._draw_pending = False
                try:
                    self.draw()
                except RuntimeError:
                    # Qt may be closing the parent hierarchy in this same
                    # event-loop turn. There is nothing left to repaint.
                    return

            def cancel_pending_draw(self):
                self._spacr_draw_timer.stop()
                self._draw_pending = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self._body_splitter = QSplitter(Qt.Horizontal, self)
        self._body_splitter.setChildrenCollapsible(False)
        from ..theme import active_palette
        surface = active_palette()["surface"]
        self._figure = Figure(figsize=(8, 6), facecolor=surface)
        self._canvas = _OwnedTimerFigureCanvas(self._figure)
        self._canvas.setStyleSheet(f"background: {surface};")
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        chart = QVBoxLayout()
        chart.addWidget(self._toolbar)
        chart.addWidget(self._canvas, 1)
        chart_wrap = QWidget(self)
        chart_wrap.setLayout(chart)
        chart_wrap.setStyleSheet(f"background: {surface};")
        self._body_splitter.addWidget(chart_wrap)

        side = QVBoxLayout()
        self._preview = QLabel("Click a point to preview its image.", self)
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumSize(220, 220)
        self._preview.setStyleSheet("border: 1px solid palette(mid);")
        side.addWidget(self._preview)
        self._point_label = QLabel("", self)
        self._point_label.setWordWrap(True)
        side.addWidget(self._point_label)

        form = QFormLayout()
        self._cluster_box = QComboBox(self)
        self._cluster_box.currentIndexChanged.connect(self._select_cluster)
        form.addRow("Select cluster", self._cluster_box)
        self._column = QLineEdit("umap_annotation", self)
        self._column.setToolTip(
            "Column created/updated on png_list when labels are applied.")
        form.addRow("DB column", self._column)
        self._value = QSpinBox(self)
        self._value.setRange(-1_000_000, 1_000_000)
        self._value.setValue(1)
        form.addRow("Manual label", self._value)
        side.addLayout(form)

        self._apply_selected = QPushButton("Label lasso selection", self)
        self._apply_selected.setObjectName("PrimaryButton")
        self._apply_selected.clicked.connect(self._write_selected)
        side.addWidget(self._apply_selected)
        self._apply_clusters = QPushButton(
            "Propagate automatic clusters", self)
        self._apply_clusters.setToolTip(
            "Write the current DBSCAN/KMeans cluster number for every point.")
        self._apply_clusters.clicked.connect(self._write_clusters)
        side.addWidget(self._apply_clusters)
        # Every display setting in one window, live and not-live together,
        # as asked. The propagate callback is the same seam the Mask live
        # preview uses, so a value tuned here lands in the settings panel
        # and is saved with the run rather than living only in this widget.
        self._display_btn = QPushButton("Display settings…", self)
        self._display_btn.setToolTip(
            "Dot size, colour and opacity apply to this figure straight "
            "away. Figure size, image count and image zoom are saved and "
            "take effect on the next run.")
        self._display_btn.clicked.connect(self.open_display_settings)
        side.addWidget(self._display_btn)

        self._status = QLabel("Waiting for an embedding.", self)
        self._status.setWordWrap(True)
        side.addWidget(self._status)
        side.addStretch(1)
        side_wrap = QWidget(self)
        side_wrap.setLayout(side)
        side_wrap.setStyleSheet(f"background: {surface};")
        self._body_splitter.addWidget(side_wrap)
        root.addWidget(self._body_splitter, 1)

        self._axes = self._figure.add_subplot(111)
        self._axes.set_facecolor(surface)
        self._scatter = None
        self._selection_artist = None
        self._picked_artist = None
        self._linked_artist = None
        self._lasso = None
        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._body_splitter.setSizes([
            int(self._display["canvas_width"]),
            int(self._display["sidebar_width"]),
        ])

    #: Which display settings can be applied to the CURRENT figure, and
    #: which only take effect on the next run.
    #:
    #: The split is not a policy, it is a fact about the artists: point
    #: size, colour and alpha are settable on a `PathCollection` that
    #: already exists, and the splitter widths are Qt. Everything else --
    #: `figuresize`, `image_nr`, `img_zoom` -- decides what gets DRAWN, and
    #: redrawing it from the same embedding is fine, but a setting that
    #: changes the embedding itself must not be in here at all: a "live
    #: apply" that re-embeds moves every point and the user loses the
    #: arrangement they were reading.
    LIVE_DISPLAY_KEYS = ("point_size", "point_color", "point_alpha",
                         "outline_width", "canvas_width", "sidebar_width")

    def set_propagate_callback(self, callback) -> None:
        """Register ``callback(dict)`` to push values into the settings panel.

        Optional: the explorer is usable without one, and a widget built in
        a test has none.
        """
        self._propagate_cb = callback

    def open_display_settings(self) -> None:
        """Open the one window, apply what can apply, propagate all of it."""
        values = dict(self._display)
        # The not-live half is not held by this widget -- it belongs to the
        # run -- so seed it from the settings panel when there is one, or
        # the dialog opens showing zeros for settings that have values.
        getter = getattr(self, "_settings_getter", None)
        if callable(getter):
            try:
                values.update(getter() or {})
            except Exception:
                LOG.debug("could not read the current run settings",
                          exc_info=True)

        dialog = UmapDisplaySettings(values, self)
        if not dialog.exec():
            return
        applied = self.apply_display(dialog.live_values())
        callback = getattr(self, "_propagate_cb", None)
        if callable(callback):
            try:
                callback(dialog.values())
            except Exception:
                LOG.debug("could not propagate the display settings",
                          exc_info=True)
        self._status.setText(
            "Display updated." if applied
            else "Saved. The changed settings take effect on the next run.")

    def display_settings(self) -> Dict:
        """The current display values, as plain data."""
        return dict(self._display)

    def apply_display(self, values: Dict) -> bool:
        """Apply display settings to the figure that is already on screen.

        :returns: True when something changed and the canvas was redrawn.

        The EMBEDDING is never touched. Only the keys in
        :data:`LIVE_DISPLAY_KEYS` are honoured; anything else is stored for
        the next run and reported by the caller, because silently ignoring
        a setting the user just changed is worse than saying it needs a
        re-run.
        """
        changed = False
        for key, value in (values or {}).items():
            if key not in self._display or value is None:
                continue
            if self._display[key] == value:
                continue
            self._display[key] = value
            changed = changed or key in self.LIVE_DISPLAY_KEYS
        if not changed:
            return False
        self._body_splitter.setSizes([
            int(self._display["canvas_width"]),
            int(self._display["sidebar_width"]),
        ])
        # Redrawn from `self._embedding`, which nothing above touched, so
        # every point keeps its coordinates and its neighbours.
        self._draw_embedding()
        return True

    def set_payload(self, payload: Dict) -> None:
        """Load the arrays/records attached by ``generate_image_umap``.

        ``payload['frame']`` is optional: a DataFrame with one row per point,
        carrying whatever the caller measured. Without it the explorer can
        still identify its points (from the records' ``prcfo``) and so still
        publishes and receives selections — but a filter on a measurement
        column has nothing here to test, and is reported as ignored rather
        than silently drawing everything as if it had applied.
        """
        embedding = np.asarray(payload.get("embedding", []), dtype=float)
        if embedding.ndim != 2 or embedding.shape[1:] != (2,):
            raise ValueError("UMAP payload embedding must have shape (N, 2)")
        labels = np.asarray(payload.get("labels", []))
        records = list(payload.get("records", []))
        if len(labels) != len(embedding) or len(records) != len(embedding):
            raise ValueError("UMAP payload arrays must have equal lengths")
        frame = payload.get("frame")
        if isinstance(frame, pd.DataFrame) and len(frame) != len(embedding):
            raise ValueError("UMAP payload arrays must have equal lengths")
        self._embedding = embedding
        self._labels = labels
        self._records = records
        display = payload.get("display")
        if isinstance(display, dict):
            for key in self._display:
                if key in display and display[key] is not None:
                    self._display[key] = display[key]
            self._body_splitter.setSizes([
                int(self._display["canvas_width"]),
                int(self._display["sidebar_width"]),
            ])
        self._selected = np.empty(0, dtype=int)
        self._picked = None
        self._build_point_identity(frame)
        self._recompute_visible_points()
        self._recompute_linked_points()
        self._draw_embedding()

    # -- identity ----------------------------------------------------------

    def _build_point_identity(self, frame: Optional[pd.DataFrame]) -> None:
        """Work out which measured object each point is, once per payload.

        Derived here rather than per lasso: parsing ninety thousand ``prcfo``
        strings on every drag would make the lasso the slow part of a screen
        whose whole job is to feel immediate.
        """
        self._point_frame = None
        self._point_keys = None
        if not len(self._embedding):
            return
        columns = list(OBJECT_KEY_COLUMNS)
        identity = pd.DataFrame(
            [_record_identity(r) or {} for r in self._records],
            columns=columns)
        if identity.isna().any(axis=None):
            # A payload that names only *some* of its points is worse than
            # one that names none, in both directions: half a lasso gets
            # published as the whole of it, and a filter tested against a
            # column of blanks dims every point as though it had matched
            # nothing. Refuse the lot.
            identity = identity.iloc[:, :0]
        if isinstance(frame, pd.DataFrame):
            table = frame.reset_index(drop=True)
            missing = {c: identity[c] for c in identity.columns
                       if c not in table.columns}
            if missing:
                table = table.assign(**missing)
        else:
            table = identity
        if not len(table.columns):
            return          # nothing to key on, and nothing to filter with
        self._point_frame = table
        if any(c not in table.columns for c in columns):
            return
        if table[columns].isna().any(axis=None):
            return
        try:
            self._point_keys = object_keys(table)
        except Exception:
            self._point_keys = None

    def point_keys(self) -> Optional[pd.Index]:
        """The object key of each point, or ``None`` when unidentifiable."""
        return self._point_keys

    def _draw_embedding(self) -> None:
        from matplotlib.widgets import LassoSelector
        from ..theme import active_palette

        palette = active_palette()
        background = palette["surface_alt"]
        foreground = palette["fg"]
        self._axes.clear()
        self._figure.patch.set_facecolor(background)
        self._axes.set_facecolor(background)
        requested_color = str(self._display["point_color"]).strip()
        color_key = requested_color.lower()
        scatter_kwargs = {}
        if color_key in {"", "cluster", "viridis"}:
            scatter_kwargs.update(c=self._labels, cmap="viridis")
        else:
            from matplotlib.colors import is_color_like
            if is_color_like(requested_color):
                scatter_kwargs["color"] = requested_color
            else:
                scatter_kwargs.update(c=self._labels, cmap="viridis")
        self._scatter = self._axes.scatter(
            self._embedding[:, 0], self._embedding[:, 1],
            s=float(self._display["point_size"]),
            alpha=float(self._display["point_alpha"]),
            **scatter_kwargs,
        )
        self._axes.set_xlabel("UMAP Dimension 1")
        self._axes.set_ylabel("UMAP Dimension 2")
        self._axes.set_title("Click a point to preview · drag a lasso to select")
        self._axes.tick_params(axis="both", colors=foreground)
        self._axes.xaxis.label.set_color(foreground)
        self._axes.yaxis.label.set_color(foreground)
        self._axes.title.set_color(foreground)
        for spine in self._axes.spines.values():
            spine.set_color(foreground)
        self._selection_artist = self._axes.scatter(
            [], [], s=70, facecolors="none", edgecolors=foreground,
            linewidths=float(self._display["outline_width"]))
        self._picked_artist = self._axes.scatter(
            [], [], s=110, facecolors="none", edgecolors="#ffcc33",
            linewidths=float(self._display["outline_width"]))
        # Selections made elsewhere get their own ring, in the accent colour
        # rather than the foreground one, so "what I lassoed" and "what the
        # table is showing me" stay tellable apart at a glance.
        self._linked_artist = self._axes.scatter(
            [], [], s=90, facecolors="none",
            edgecolors=palette.get("accent", "#4A9EFF"),
            linewidths=float(self._display["outline_width"]) * 1.6)
        if self._lasso is not None:
            self._lasso.disconnect_events()
        self._lasso = LassoSelector(
            self._axes, onselect=self._on_lasso,
            props={
                "color": foreground,
                "linewidth": float(self._display["outline_width"]),
            },
        )
        self._cluster_box.blockSignals(True)
        self._cluster_box.clear()
        self._cluster_box.addItem("—", None)
        for label in sorted(np.unique(self._labels), key=lambda value: str(value)):
            self._cluster_box.addItem(str(label), label)
        self._cluster_box.blockSignals(False)
        self._status.setText(self._payload_status())
        if len(self._selected):
            self._selection_artist.set_offsets(
                self._embedding[self._selected])
        if self._picked is not None:
            self._picked_artist.set_offsets(
                self._embedding[self._picked].reshape(1, 2))
        self._apply_point_alpha()
        self._draw_linked_points()
        self._canvas.draw_idle()

    def _payload_status(self) -> str:
        """The resting status line, including how the shared filter landed."""
        writable = sum(
            bool(row.get("db_path") and row.get("db_png_path"))
            for row in self._records)
        return (f"{len(self._records)} points · {writable} database-backed · "
                "drag around points to select them." + self._filter_note)

    # -- the shared filter: dim, never remove -------------------------------

    def _recompute_visible_points(self) -> None:
        """Work out which points the shared filter keeps.

        Degrades to "all of them" when the filter names something this
        payload does not carry — an embedding drawn with every point missing
        is a worse answer than a complete one — but records that it did, so
        the status line can say the filter was ignored rather than let a
        complete picture read as a filtered one.
        """
        count = len(self._embedding)
        self._point_visible = np.ones(count, dtype=bool)
        self._filter_note = ""
        if not count:
            return
        try:
            data_filter = self.link.filter
        except Exception:
            return
        if data_filter.is_empty:
            return
        frame = self._point_frame
        if frame is None:
            self._filter_note = (
                f" · filter ignored ({data_filter.describe()}): this "
                "embedding carries no identities")
            return
        try:
            kept = self.linked_visible(frame).index
        except Exception as exc:
            self._filter_note = (
                f" · filter ignored ({exc.__class__.__name__})")
            return
        mask = np.zeros(count, dtype=bool)
        positions = np.asarray(kept, dtype=np.int64)
        mask[positions[(positions >= 0) & (positions < count)]] = True
        self._point_visible = mask
        self._filter_note = (
            f" · filtered: {data_filter.describe()} "
            f"({int(mask.sum())} of {count} points)")

    def _apply_point_alpha(self) -> None:
        """Repaint opacity so filtered-out points recede without moving.

        A scalar alpha is restored when nothing is filtered, rather than an
        array of identical values: the display settings are read back from
        the artist elsewhere, and ``get_alpha()`` returning an array where a
        float was set is a difference nobody asked for.
        """
        if self._scatter is None:
            return
        base = float(self._display["point_alpha"])
        target = (base if self._point_visible.all()
                  else np.where(self._point_visible, base,
                                base * DIMMED_ALPHA))
        if np.iterable(self._scatter.get_alpha()) and not np.iterable(target):
            # `Artist.set_alpha` short-circuits on `alpha != self._alpha`,
            # which raises "the truth value of an array is ambiguous" when
            # the artist is currently holding a per-point array and a scalar
            # is being set (matplotlib 3.10). Array→array and scalar→scalar
            # are fine; only this direction needs the array dropped first,
            # and there is no public call that does it.
            self._scatter._alpha = None
        self._scatter.set_alpha(target)

    def visible_points(self) -> np.ndarray:
        """Boolean mask of the points the shared filter keeps."""
        return self._point_visible.copy()

    def on_linked_filter_changed(self, data_filter: DataFilter) -> None:
        self._recompute_visible_points()
        self._apply_point_alpha()
        self._status.setText(self._payload_status())
        self._canvas.draw_idle()

    # -- the shared selection: highlight, never hide ------------------------

    def _recompute_linked_points(self,
                                 selection: Optional[Selection] = None) -> None:
        """Which points a selection published elsewhere names."""
        if selection is None:
            try:
                selection = self.link.selection
            except Exception:
                selection = Selection.none()
        keys = self._point_keys
        if keys is None or not selection.is_active or not len(self._embedding):
            self._linked_points = np.empty(0, dtype=int)
            return
        # `match_keys`, not `Index.isin`: the table publishes `..._f1_cell1`
        # now that a reader states which table it read, while these points
        # are keyed off a `prcfo` that states nothing. Exact equality
        # highlighted nothing at all, which on a UMAP is indistinguishable
        # from the user having lassoed empty space.
        self._linked_points = np.flatnonzero(
            match_keys(keys, selection.keys))

    def _draw_linked_points(self) -> None:
        if self._linked_artist is None:
            return
        points = (self._embedding[self._linked_points]
                  if len(self._linked_points) else np.empty((0, 2)))
        self._linked_artist.set_offsets(points)

    def linked_points(self) -> np.ndarray:
        """Indices of the points a selection made elsewhere is highlighting."""
        return self._linked_points.copy()

    def on_linked_selection_changed(self, selection: Selection) -> None:
        """Ring the points somebody else selected. Nothing is hidden, and the
        local lasso — which is what the annotation buttons write — is left
        exactly as the user drew it."""
        self._recompute_linked_points(selection)
        self._draw_linked_points()
        self._canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        """Zoom around the pointer with the mouse wheel."""
        if (event.inaxes is not self._axes or event.xdata is None
                or event.ydata is None):
            return
        factor = 0.8 if event.button == "up" else 1.25
        x0, x1 = self._axes.get_xlim()
        y0, y1 = self._axes.get_ylim()
        self._axes.set_xlim(
            event.xdata - (event.xdata - x0) * factor,
            event.xdata + (x1 - event.xdata) * factor,
        )
        self._axes.set_ylim(
            event.ydata - (event.ydata - y0) * factor,
            event.ydata + (y1 - event.ydata) * factor,
        )
        self._canvas.draw_idle()

    def _on_click(self, event) -> None:
        if (event.inaxes is not self._axes or event.xdata is None
                or not len(self._embedding)):
            return
        click = np.array([event.xdata, event.ydata], dtype=float)
        spans = np.ptp(self._embedding, axis=0)
        spans[spans == 0] = 1.0
        distance = np.linalg.norm((self._embedding - click) / spans, axis=1)
        self.show_point(int(np.argmin(distance)))

    def show_point(self, index: int) -> None:
        """Preview one point's image and database identity."""
        if not (0 <= int(index) < len(self._records)):
            return
        index = int(index)
        self._picked = index
        point = self._embedding[index]
        self._picked_artist.set_offsets(point.reshape(1, 2))
        record = self._records[index]
        source = record.get("image")
        try:
            if source is None:
                raise ValueError("No image source for this point")
            if hasattr(source, "array"):
                image = Image.fromarray(np.asarray(source.array())).convert("RGB")
            else:
                with Image.open(source) as opened:
                    image = opened.convert("RGB")
            image.thumbnail((360, 360), Image.Resampling.LANCZOS)
            qimage = QImage(ImageQt(image)).copy()
            self._preview.setPixmap(QPixmap.fromImage(qimage))
        except Exception as exc:
            self._preview.setPixmap(QPixmap())
            self._preview.setText(f"Preview unavailable\n{exc}")
        self._point_label.setText(
            f"Point {index + 1}/{len(self._records)} · "
            f"cluster {self._labels[index]}\n"
            f"{record.get('db_png_path') or record.get('display_name') or ''}")
        self._canvas.draw_idle()

    def _on_lasso(self, vertices: Sequence) -> None:
        from matplotlib.path import Path

        inside = Path(vertices).contains_points(self._embedding)
        self._selected = np.flatnonzero(inside)
        self._refresh_selection()

    def _select_cluster(self, _index: int) -> None:
        label = self._cluster_box.currentData()
        if label is None:
            return
        self._selected = np.flatnonzero(self._labels == label)
        self._refresh_selection()

    def _refresh_selection(self) -> None:
        points = (self._embedding[self._selected]
                  if len(self._selected) else np.empty((0, 2)))
        self._selection_artist.set_offsets(points)
        self._status.setText(f"{len(self._selected)} point(s) selected."
                             + self._filter_note)
        if len(self._selected):
            self.show_point(int(self._selected[0]))
        self._publish_local_selection()
        self._canvas.draw_idle()

    def _publish_local_selection(self) -> None:
        """Tell every other view what was just lassoed here.

        Silent when the payload carries no identities: publishing a lasso as
        an empty selection would tell the plate view "the user selected
        nothing", wiping a highlight that was never this screen's to clear.

        A lasso that legitimately caught nothing IS published, as an empty
        selection — that is a result, and the resting state is a different
        thing (:meth:`clear_linked_selection`).
        """
        if self._point_keys is None:
            return
        try:
            self.publish_selection(self._point_keys[self._selected])
        except Exception:
            LOG.info("publishing the UMAP selection failed", exc_info=True)

    def _write_selected(self) -> None:
        if not len(self._selected):
            self._status.setText("Draw a lasso or select a cluster first.")
            return
        records = [self._records[i] for i in self._selected]
        values = [self._value.value()] * len(records)
        self._start_write(records, values, "manual selection")

    def _write_clusters(self) -> None:
        self._start_write(
            self._records, self._labels.tolist(), "automatic clusters")

    def _start_write(self, records, values, label: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._status.setText("An annotation write is already running.")
            return
        column = self._column.text().strip()
        self._set_write_enabled(False)
        self._status.setText(f"Writing {label} to {column or '(no column)'}…")
        worker = _AnnotationWorker(records, values, column, self)
        worker.finished_result.connect(self._on_write_done, Qt.QueuedConnection)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _set_write_enabled(self, enabled: bool) -> None:
        self._apply_selected.setEnabled(enabled)
        self._apply_clusters.setEnabled(enabled)

    @Slot(int, int, str)
    def _on_write_done(self, updated: int, skipped: int, error: str) -> None:
        self._worker = None
        self._set_write_enabled(True)
        if error:
            self._status.setText(f"Database write failed: {error}")
        else:
            self._status.setText(
                f"Updated {updated} png_list row(s); skipped {skipped}.")
        self.annotation_finished.emit(updated, skipped)

    def closeEvent(self, event):
        try:
            self.unlink_selection()
        except (RuntimeError, TypeError):
            # The process-wide link's C++ side is gone (interpreter teardown).
            pass
        worker = self._worker
        if worker is not None:
            worker.requestInterruption()
            worker.wait()
            self._worker = None
        # FigureCanvasQTAgg implements draw_idle with a zero-delay Qt timer.
        # Cancel that pending draw before Qt deletes the C++ canvas.
        if self._lasso is not None:
            self._lasso.disconnect_events()
            self._lasso = None
        if getattr(self, "_canvas", None) is not None:
            self._canvas.cancel_pending_draw()
        super().closeEvent(event)
