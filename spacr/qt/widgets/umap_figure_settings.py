"""Every Image UMAP setting, live-editable against the STATIC figure.

Instruction 26 gave the LIVE explorer a settings window whose display half
applies to the figure on screen. This is the same idea for the other half of
the screen -- the ordinary, non-live figure the run leaves in
:class:`spacr.qt.widgets.figure_queue.FigureQueue`::

    "the non live image UMAP figure settings should have all the image UMAP
     settings live editable (you should see changes in the graph directly)"

THE COST OF A SETTING IS NOT UNIFORM, and pretending it is would make the
panel unusable. Three tiers, and every field declares which one it is in:

``TIER_STYLE``
    settable on artists that already exist -- dot size, dot colour, opacity,
    outline width. Microseconds; nothing is re-read from disk.
``TIER_REDRAW``
    needs the figure replotted, but **from the same embedding** -- image
    count, image zoom, figure size, which layers are drawn. Tenths of a
    second, because the thumbnails are re-read, so these are debounced.
``TIER_RERUN``
    changes the embedding itself -- ``n_neighbors``, ``min_dist``, the metric,
    the feature filter. There is no honest way to apply one of these to a
    finished figure: recomputing moves every point, and the arrangement the
    user was reading is the whole value of the projection. They are editable
    and propagated, and the panel says they land on the next run.

The tier is a fact about the artists and the data, not a policy, which is why
it lives beside the field rather than in the caller.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QLineEdit,
    QSpinBox, QVBoxLayout, QWidget,
)

LOG = logging.getLogger("spacr.qt.umap_figure_settings")

#: Applied to the artists already on the figure.
TIER_STYLE = "style"
#: Replotted from the SAME embedding.
TIER_REDRAW = "redraw"
#: Changes the embedding; saved for the next run.
TIER_RERUN = "rerun"

#: How long a value has to stop changing before the figure is redrawn. A
#: spin box drag emits a value per step and replotting a montage is not a
#: per-step cost, so every edit coalesces into one render.
APPLY_DEBOUNCE_MS = 250


class Field(NamedTuple):
    """One editable Image UMAP setting."""

    key: str
    label: str
    kind: str            # int | float | text | bool | choice | int_or_none
    low: float
    high: float
    tier: str
    choices: Tuple[str, ...] = ()


#: Every Image UMAP setting that decides what the figure looks like.
#:
#: The display half is exactly the "UMAP Display" group the settings panel
#: builds (``settings_model._regroup`` for ``umap``), so the two surfaces
#: cannot drift into offering different sets of knobs. The rest are the
#: reduction and clustering settings, which are here because the user asked
#: for "all the image UMAP settings" and a window that silently omitted the
#: ones that matter most would be answering a different question.
IMAGE_UMAP_FIELDS: Tuple[Field, ...] = (
    # -- applies to the artists already drawn ------------------------------
    Field("dot_size",        "Dot size",        "int",   1, 4000, TIER_STYLE),
    Field("point_color",     "Dot colour",      "text",  0, 0,    TIER_STYLE),
    Field("point_alpha",     "Dot opacity",     "float", 0.0, 1.0, TIER_STYLE),
    Field("outline_width",   "Outline width",   "float", 0.0, 10.0, TIER_STYLE),
    # -- needs the same embedding replotted --------------------------------
    Field("figuresize",      "Figure size",     "float", 1.0, 60.0, TIER_REDRAW),
    Field("image_nr",        "Images shown",    "int",   0, 100000, TIER_REDRAW),
    Field("img_zoom",        "Image zoom",      "float", 0.001, 5.0, TIER_REDRAW),
    Field("plot_images",     "Draw images",     "bool",  0, 0,    TIER_REDRAW),
    Field("plot_points",     "Draw points",     "bool",  0, 0,    TIER_REDRAW),
    Field("plot_outlines",   "Draw outlines",   "bool",  0, 0,    TIER_REDRAW),
    Field("smooth_lines",    "Smooth outlines", "bool",  0, 0,    TIER_REDRAW),
    Field("plot_by_cluster", "Sample per cluster", "bool", 0, 0,  TIER_REDRAW),
    Field("remove_image_canvas", "Cut image canvas", "bool", 0, 0, TIER_REDRAW),
    Field("black_background", "Black background", "bool", 0, 0,   TIER_REDRAW),
    # -- changes the embedding: next run -----------------------------------
    Field("reduction_method", "Reduction", "choice", 0, 0, TIER_RERUN,
          ("umap", "tsne")),
    Field("n_neighbors",     "Neighbours",      "int",   2, 1000000, TIER_RERUN),
    Field("min_dist",        "Minimum distance", "float", 0.0, 1.0, TIER_RERUN),
    Field("metric",          "Metric",          "text",  0, 0,    TIER_RERUN),
    Field("clustering",      "Clustering",      "choice", 0, 0,   TIER_RERUN,
          ("dbscan", "kmeans")),
    Field("eps",             "DBSCAN eps",      "float", 0.0, 1000.0, TIER_RERUN),
    Field("min_samples",     "Minimum samples", "int",   1, 1000000, TIER_RERUN),
    Field("remove_cluster_noise", "Drop noise points", "bool", 0, 0,
          TIER_RERUN),
    Field("remove_highly_correlated", "Drop correlated features", "bool",
          0, 0, TIER_RERUN),
    Field("log_data",        "Log-transform features", "bool", 0, 0,
          TIER_RERUN),
    Field("filter_by",       "Feature filter",  "text",  0, 0,    TIER_RERUN),
    Field("row_limit",       "Row limit",       "int_or_none", 0, 100000000,
          TIER_RERUN),
    Field("color_by",        "Colour by column", "text", 0, 0,    TIER_RERUN),
    Field("plot_cluster_grids", "Cluster grid figure", "bool", 0, 0,
          TIER_RERUN),
    Field("save_figure",     "Save figure as PDF", "bool", 0, 0,  TIER_RERUN),
    Field("umap_canvas_width", "Live canvas width", "int", 200, 4000,
          TIER_RERUN),
    Field("umap_sidebar_width", "Live sidebar width", "int", 120, 2000,
          TIER_RERUN),
)

#: ``key -> tier``, for callers that only need the classification.
FIELD_TIERS: Dict[str, str] = {f.key: f.tier for f in IMAGE_UMAP_FIELDS}


def keys_for_tier(tier: str) -> Tuple[str, ...]:
    """Every setting key in ``tier``."""
    return tuple(f.key for f in IMAGE_UMAP_FIELDS if f.tier == tier)


def live_keys() -> Tuple[str, ...]:
    """The settings that reach the figure already on screen."""
    return tuple(f.key for f in IMAGE_UMAP_FIELDS
                 if f.tier in (TIER_STYLE, TIER_REDRAW))


# ---------------------------------------------------------------------------
# Applying values to a finished figure
# ---------------------------------------------------------------------------

def _is_fixed_colour(point_color) -> bool:
    """Whether ``point_color`` names one colour rather than "per cluster"."""
    text = str(point_color or "").strip().lower()
    if text in {"", "cluster", "viridis"}:
        return False
    from matplotlib.colors import is_color_like
    return bool(is_color_like(str(point_color).strip()))


def restyle_umap_figure(fig, values: Dict[str, Any]) -> bool:
    """Push the cheap settings onto the artists ``fig`` already carries.

    Nothing is re-read and nothing is recomputed, so the points cannot move.

    The original per-cluster face colours are stashed on the collection the
    first time a fixed colour is applied. Without that, switching the dot
    colour to ``red`` and back to ``cluster`` would leave every point red:
    the per-cluster colours were overwritten and there is nowhere left to
    read them from short of replotting.
    """
    if fig is None:
        return False
    touched = False
    size = values.get("dot_size")
    alpha = values.get("point_alpha")
    width = values.get("outline_width")
    colour = values.get("point_color")
    for axes in fig.get_axes():
        for collection in axes.collections:
            if size is not None:
                try:
                    collection.set_sizes([float(size)])
                    touched = True
                except Exception:
                    LOG.debug("could not set the dot size", exc_info=True)
            if alpha is not None:
                try:
                    collection.set_alpha(max(0.0, min(1.0, float(alpha))))
                    touched = True
                except Exception:
                    LOG.debug("could not set the dot opacity", exc_info=True)
            if colour is not None:
                try:
                    if getattr(collection, "_spacr_base_facecolor", None) is None:
                        collection._spacr_base_facecolor = (
                            collection.get_facecolor().copy())
                    if _is_fixed_colour(colour):
                        collection.set_facecolor(str(colour).strip())
                    else:
                        collection.set_facecolor(
                            collection._spacr_base_facecolor)
                    touched = True
                except Exception:
                    LOG.debug("could not set the dot colour", exc_info=True)
        if width is not None:
            for line in axes.lines:
                try:
                    line.set_linewidth(max(0.1, float(width)))
                    touched = True
                except Exception:
                    LOG.debug("could not set the outline width", exc_info=True)
    return touched


def redraw_umap_figure(fig, payload: Dict[str, Any],
                       values: Dict[str, Any]) -> bool:
    """Replot ``fig`` FROM THE SAME EMBEDDING with ``values``.

    The embedding in ``payload`` is read, never recomputed: this is what
    makes "live apply" honest on a projection. Every point keeps its
    coordinates and its neighbours; only what is drawn on top of them
    changes.

    :returns: True when the figure was replotted.
    """
    import numpy as np

    if fig is None or not isinstance(payload, dict):
        return False
    embedding = np.asarray(payload.get("embedding"), dtype=float)
    if embedding.ndim != 2 or embedding.shape[0] == 0:
        return False
    labels = payload.get("plot_labels")
    if labels is None:
        labels = payload.get("labels")
    labels = np.asarray(labels)
    if len(labels) != len(embedding):
        return False

    from ...utils import (assign_colors, generate_colors, plot_clusters,
                          plot_umap_images, _plot_theme_colors,
                          _style_plot_axes)

    def _get(key, fallback):
        value = values.get(key)
        return fallback if value is None else value

    black_background = bool(_get("black_background", True))
    figuresize = float(_get("figuresize", 10))
    unique_labels = np.unique(labels)
    colors = generate_colors(len(unique_labels), black_background)
    colors, _index = assign_colors(unique_labels, colors)
    centers = [np.mean(embedding[labels == label], axis=0)
               for label in unique_labels]

    theme = _plot_theme_colors(black_background, payload.get("theme_colors"))
    fig.clear()
    axes = fig.add_subplot(111)
    _style_plot_axes(fig, axes, theme)
    try:
        fig.set_size_inches(figuresize, figuresize)
    except Exception:
        LOG.debug("could not resize the figure", exc_info=True)
    plot_clusters(
        axes, embedding, labels, colors, centers,
        bool(_get("plot_outlines", True)), bool(_get("plot_points", True)),
        bool(_get("smooth_lines", True)), figuresize,
        float(_get("dot_size", 50)), False,
        point_color=_get("point_color", "cluster"),
        point_alpha=float(_get("point_alpha", 0.65)),
        outline_width=float(_get("outline_width", 1.0)),
    )
    records = payload.get("records") or []
    image_paths = [record.get("image") for record in records]
    if (bool(_get("plot_images", False)) and len(image_paths) == len(embedding)
            and any(path is not None for path in image_paths)):
        try:
            plot_umap_images(
                axes, image_paths, embedding, labels,
                int(_get("image_nr", 16)), float(_get("img_zoom", 0.5)),
                colors, bool(_get("plot_by_cluster", True)),
                bool(_get("remove_image_canvas", False)), False)
        except Exception:
            # A montage is decoration; the embedding is the result. Losing
            # the thumbnails must never lose the figure (INVARIANTS 10).
            LOG.debug("could not redraw the image overlay", exc_info=True)
    # `Figure.clear` drops artists, not attributes -- restated rather than
    # relied on, because a figure that loses its payload can never be
    # edited a second time.
    fig._spacr_umap_payload = payload
    return True


def apply_to_figure(fig, payload: Dict[str, Any], values: Dict[str, Any],
                    previous: Optional[Dict[str, Any]] = None) -> str:
    """Apply ``values`` to a finished Image UMAP figure.

    :returns: ``"redraw"``, ``"style"`` or ``""`` -- what it actually had to
        do, so the caller can say whether the graph followed and can skip
        re-rasterising when nothing changed.
    """
    previous = previous or {}
    changed = {key for key, value in (values or {}).items()
               if key not in previous or previous[key] != value}
    if not changed:
        return ""
    if any(FIELD_TIERS.get(key) == TIER_REDRAW for key in changed):
        return "redraw" if redraw_umap_figure(fig, payload, values) else ""
    if any(FIELD_TIERS.get(key) == TIER_STYLE for key in changed):
        return "style" if restyle_umap_figure(fig, values) else ""
    return ""


# ---------------------------------------------------------------------------
# The form
# ---------------------------------------------------------------------------

class UmapFigureSettings(QWidget):
    """The Image UMAP half of the non-live figure-settings window.

    Emits :attr:`settings_changed` with every value, debounced, as soon as
    the user changes one -- there is no Apply button, because "you should see
    changes in the graph directly" is the requirement.
    """

    #: Debounced, and carries the WHOLE value dict rather than the delta: the
    #: applier has to decide which tier changed, and it can only do that
    #: against a complete picture.
    settings_changed = Signal(dict)

    def __init__(self, values: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self._editors: Dict[str, QWidget] = {}
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(APPLY_DEBOUNCE_MS)
        self._timer.timeout.connect(self._emit_changed)

        seeded = dict(self._defaults())
        seeded.update({k: v for k, v in (values or {}).items()
                       if v is not None})

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        for tier, heading, note in (
            (TIER_STYLE, "Applies now",
             "Set straight onto the points already drawn."),
            (TIER_REDRAW, "Applies now (redraws the graph)",
             "The graph is drawn again from the SAME embedding, so no point "
             "moves. Debounced, because the thumbnails are re-read."),
            (TIER_RERUN, "Applies on the next run",
             "These change the embedding itself. Applying one here would "
             "move every point and lose the arrangement you are reading, so "
             "they are saved and propagated instead."),
        ):
            title = QLabel(f"<b>{heading}</b>")
            root.addWidget(title)
            caption = QLabel(f"<span style='color:gray;'>{note}</span>")
            caption.setWordWrap(True)
            root.addWidget(caption)
            form = QFormLayout()
            for field in IMAGE_UMAP_FIELDS:
                if field.tier != tier:
                    continue
                editor = self._editor(field, seeded.get(field.key))
                editor.setProperty("settingKey", field.key)
                editor.setProperty("umapSettingTier", field.tier)
                self._editors[field.key] = editor
                form.addRow(field.label, editor)
            root.addLayout(form)
        self._applied: Dict[str, Any] = dict(self.values())
        self._initial: Dict[str, Any] = dict(self._applied)

    # -- construction ------------------------------------------------------

    @staticmethod
    def _defaults() -> Dict[str, Any]:
        from ...settings import set_default_umap_image_settings
        try:
            return set_default_umap_image_settings({})
        except Exception:
            LOG.debug("could not read the Image UMAP defaults", exc_info=True)
            return {}

    def _editor(self, field: Field, value) -> QWidget:
        if field.kind == "bool":
            box = QCheckBox()
            box.setChecked(bool(value))
            box.toggled.connect(self._schedule)
            return box
        if field.kind == "choice":
            combo = QComboBox()
            combo.addItems(list(field.choices))
            text = str(value or "").strip().lower()
            if text in field.choices:
                combo.setCurrentIndex(field.choices.index(text))
            combo.currentIndexChanged.connect(self._schedule)
            return combo
        if field.kind == "int":
            spin = QSpinBox()
            spin.setRange(int(field.low), int(field.high))
            if value is not None:
                try:
                    spin.setValue(int(float(value)))
                except (TypeError, ValueError):
                    pass
            spin.valueChanged.connect(self._schedule)
            return spin
        if field.kind == "float":
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(float(field.low), float(field.high))
            spin.setSingleStep(0.05)
            if value is not None:
                try:
                    spin.setValue(float(value))
                except (TypeError, ValueError):
                    pass
            spin.valueChanged.connect(self._schedule)
            return spin
        # text and int_or_none. `row_limit` is genuinely nullable -- None
        # means "every row" -- and a spin box has no way to say that, so it
        # is typed rather than clamped.
        edit = QLineEdit()
        if value is not None:
            edit.setText(str(value))
        edit.textChanged.connect(self._schedule)
        return edit

    # -- values ------------------------------------------------------------

    def values(self) -> Dict[str, Any]:
        """Every setting the window holds, keyed as the settings dict keys."""
        out: Dict[str, Any] = {}
        for field in IMAGE_UMAP_FIELDS:
            editor = self._editors.get(field.key)
            if editor is None:
                continue
            if isinstance(editor, QCheckBox):
                out[field.key] = bool(editor.isChecked())
            elif isinstance(editor, QComboBox):
                out[field.key] = editor.currentText()
            elif isinstance(editor, QLineEdit):
                text = editor.text().strip()
                if field.kind == "int_or_none":
                    out[field.key] = _int_or_none(text)
                else:
                    out[field.key] = text or None
            else:
                out[field.key] = editor.value()
        return out

    def live_values(self) -> Dict[str, Any]:
        """Only the half that reaches the figure already on screen."""
        live = set(live_keys())
        return {k: v for k, v in self.values().items() if k in live}

    def initial_values(self) -> Dict[str, Any]:
        """What the window opened on -- what Cancel puts back."""
        return dict(self._initial)

    # -- change plumbing ---------------------------------------------------

    def _schedule(self, *_args) -> None:
        self._timer.start()

    def flush(self) -> None:
        """Emit any pending change now instead of on the timer."""
        if self._timer.isActive():
            self._timer.stop()
            self._emit_changed()

    def _emit_changed(self) -> None:
        values = self.values()
        if values == self._applied:
            return
        self._applied = dict(values)
        self.settings_changed.emit(values)


def _int_or_none(text: str) -> Optional[int]:
    """``"1000"`` -> 1000; blank, ``none`` and junk -> ``None`` (every row)."""
    cleaned = str(text or "").strip()
    if not cleaned or cleaned.lower() in {"none", "null", "all"}:
        return None
    try:
        return int(float(cleaned))
    except (TypeError, ValueError):
        return None


__all__ = [
    "APPLY_DEBOUNCE_MS", "Field", "FIELD_TIERS", "IMAGE_UMAP_FIELDS",
    "TIER_REDRAW", "TIER_RERUN", "TIER_STYLE", "UmapFigureSettings",
    "apply_to_figure", "keys_for_tier", "live_keys", "redraw_umap_figure",
    "restyle_umap_figure",
]
