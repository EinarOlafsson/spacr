"""Plaque Assay's live preview, in its two modes, and the switch between them.

Plaque Assay reads two kinds of folder, and the Mask preview it
used to borrow fitted neither:

* **Plaque mode** -- a folder of cropped plaque images. The run segments
  every image and writes one table of per-image values and one of
  per-plaque values. All there is to check before running is the mask, so
  the preview segments one image with the model and thresholds the run will
  use and draws the plaque outlines, with the count and mean area.
* **Figure mode** -- a folder of published figures. The run finds the plaque
  images inside each figure with a detector, reads the text printed around
  them, decides which condition each one shows, and segments each crop. The
  preview does every one of those steps on one figure and shows them: the
  boxes, the text that was read, the proposed condition per image, and the
  plaque outlines inside each box. The table under it is where a person
  corrects a condition and ticks it OK; saving writes
  ``figure_annotations.csv`` beside the figures, which the run reads.

A missing legend is asked for here rather than at the terminal: when a panel
letter was read and no legend is known for the figure, a paste box appears,
and a pasted legend is stored in ``legends.csv`` beside the figures so the
run keys the same passages.

The detector and the text reader are the ``spacr[papers]`` extra. Without
them Figure mode says so, with the command that installs them, instead of
failing.

Everything that takes time runs on a :class:`~spacr.qt.job_runner.JobRunner`
worker, and the pure pieces below the widgets take their heavy callables as
parameters so they can be tested without a model.
"""
from __future__ import annotations

import csv
import logging
import os
import threading
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import (QActionGroup, QColor, QFont, QImage, QPainter,
                           QPen, QPixmap)
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMenu, QPlainTextEdit, QPushButton, QSizePolicy, QTableWidget,
    QSpinBox, QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from ..i18n import tr
from ..job_runner import JobRunner
from ..hidpi import scaled_for
from .sortable_table import install_sorting, table_item
from .preview_contract import (
    PREVIEW_CANCEL_TEXT, PREVIEW_RUN_TEXT, PREVIEW_RUNNING_MESSAGE,
    LivePreviewContract, preview_failure_message,
)

from ...plaque import segment_plaque_image

LOG = logging.getLogger("spacr.qt.plaque_preview")

__all__ = [
    "PLAQUE_MODE",
    "FIGURE_MODE",
    "MODE_KEY",
    "FIGURE_ONLY_KEYS",
    "PLAQUE_ONLY_KEYS",
    "PAPERS_REQUIREMENT",
    "PlaqueModeSwitch",
    "PlaquePreviewPanel",
    "build_plaque_preview_card",
    "install_plaque_mode",
    "normalise_mode",
    "keys_hidden_in",
    "missing_papers_packages",
    "papers_install_message",
    "parse_sizes",
    "images_in",
    "load_display_image",
    "outline_labels",
    "OverlayStyle",
    "OVERLAY_OUTLINES",
    "OVERLAY_FILL",
    "RANDOM_COLOUR",
    "IMAGE_TABS",
    "overlay_colour",
    "object_palette",
    "render_overlay",
    "render_objects",
    "render_cellprob",
    "PlaqueOverlayDialog",
    "plaque_model_choices",
    "detector_choices",
    "resolve_plaque_model",
    "resolve_detector",
    "segment_plaque_image",
    "write_legend",
    "plaque_pass",
    "figure_pass",
    "detect_figure",
    "region_at",
    "plaque_rows",
    "segment_well",
    "paper_folder_name",
    "PaperDialog",
    "PlaqueSettingsDialog",
    "annotate_figure",
    "TEXT_KEYS",
]

PLAQUE_MODE = "plaque"
FIGURE_MODE = "figure"
MODES = (PLAQUE_MODE, FIGURE_MODE)

MODE_KEY = "plaque_mode"

TEXT_KEYS = ("text_reach_above", "text_reach_left", "text_reach_below",
             "text_use_above", "text_use_left", "text_use_below",
             "text_panel_reach", "text_min_confidence", "text_ignore",
             "text_order", "text_separator", "text_reread",
             "text_reread_scale")

TEXT_READ_KEYS = ("text_reread", "text_reread_scale")

TEXT_ORDERS = ("above,left,below", "left,above,below", "above,below,left",
               "left,below,above", "below,above,left", "below,left,above",
               "above,left", "left,above", "above", "left")

FIGURE_ONLY_KEYS = ("figure_detector", "figure_imgsz", "figure_confidence",
                    "figure_read_text", "confirm_annotations") + TEXT_KEYS

PLAQUE_ONLY_KEYS = ("well_detection", "well_confidence", "well_pad")

PAPERS_REQUIREMENT = "spacr[papers]"

PAPERS_PACKAGES = (("ultralytics", "ultralytics"),
                   ("rapidocr_onnxruntime", "RapidOCR"))

DEFAULT_PLAQUE_MODEL = "toxoplasma_plaque_v2"
DEFAULT_DETECTOR = "toxoplasma_well_detector_v2"
DEFAULT_SIZES = (640, 1280)

OUTLINE_COLOUR = (255, 214, 0)

PICK_A_WELL = "Click a well on the figure or a row in the table."
BOX_OK = QColor(64, 200, 120)
BOX_WAITING = QColor(255, 150, 40)
BOX_SELECTED = QColor(0, 200, 255)

TABLE_COLUMNS = ("#", "Panel", "Label text", "Legend passage", "Condition",
                 "Source", "Plaques", "Mean area", "OK")
CONDITION_COLUMN = 4
SOURCE_COLUMN = 5
PLAQUES_COLUMN = 6
MEAN_AREA_COLUMN = 7
OK_COLUMN = 8

PLAQUE_COLUMNS = ("Well", "Panel", "Condition", "Plaque", "Area (px)",
                  "Vs panel median", "Vs well median", "Perimeter (px)",
                  "Equivalent diameter (px)", "Eccentricity", "Solidity",
                  "Centroid y", "Centroid x", "Area (mm²)", "Scale")

PLAQUE_KEYS = ("well", "panel", "condition", "plaque_id", "area_px",
               "area_vs_panel_median", "area_vs_well_median",
               "perimeter_px", "equivalent_diameter_px", "eccentricity",
               "solidity", "centroid_y", "centroid_x", "area_mm2", "scale")

#: The colour of a Source cell whose label and legend disagree.
CONFLICT_COLOUR = QColor(230, 90, 60)


def _conflict_note(annotation: Any) -> str:
    """Why an image's two readings disagree, for its Source cell's tooltip.

    :param annotation: a :class:`spacr.plaque_papers.Annotation`.
    :returns: ``''`` when they do not disagree.
    """
    if not getattr(annotation, "conflict", False):
        return ""
    return tr("The legend for panel {panel} names {terms}, which the label "
              "beside this image does not: one of the two readings is wrong. "
              "Check the condition before you OK it.",
              panel=annotation.panel or "?",
              terms=", ".join(annotation.conflict_terms))


def _scale_note(scale: Any) -> str:
    """What a plaque image's sizes are measured in, for the Scale column.

    :param scale: a :class:`spacr.plaque_papers._Scale`, or None.
    :returns: the note.
    """
    if scale is None or not getattr(scale, "px_per_mm", None):
        return tr("pixels (no scale bar or whole well)")
    return tr("{source}: {ppm} px/mm", source=scale.source,
              ppm=f"{scale.px_per_mm:.1f}")


def normalise_mode(value: Any) -> str:
    """The mode a setting value names, ``'plaque'`` for anything unknown.

    :param value: ``'plaque'``, ``'figure'``, any case, or anything else.
    :returns: ``'plaque'`` or ``'figure'``.
    """
    text = str(value or "").strip().lower()
    return FIGURE_MODE if text == FIGURE_MODE else PLAQUE_MODE


def keys_hidden_in(mode: Any) -> Tuple[str, ...]:
    """The settings that do not apply in ``mode`` and come off the form.

    :param mode: the plaque mode.
    :returns: setting keys.
    """
    if normalise_mode(mode) == FIGURE_MODE:
        return PLAQUE_ONLY_KEYS
    return FIGURE_ONLY_KEYS


def missing_papers_packages(importable: Optional[Callable[[str], bool]] = None
                            ) -> List[str]:
    """The ``spacr[papers]`` packages this environment cannot import.

    Asked with :func:`importlib.util.find_spec`, which does not import
    anything -- importing ultralytics rewrites its settings file.

    :param importable: ``fn(module) -> bool``, for tests.
    :returns: display names of the missing packages, empty when all are here.
    """
    def found(module: str) -> bool:
        """Whether ``module`` can be found without importing it."""
        try:
            return find_spec(module) is not None
        except (ImportError, ValueError):
            return False

    check = importable or found
    missing = [label for module, label in PAPERS_PACKAGES if not check(module)]
    if missing and importable is None:
        from ...plaque_papers import reader_environment

        if reader_environment() is not None:
            return []
    return missing


def papers_install_message(missing: Sequence[str]) -> str:
    """What Figure mode says when its optional packages are missing.

    :param missing: names from :func:`missing_papers_packages`.
    :returns: one paragraph with the install command.
    """
    return tr("Figure mode needs {names}. Press Install to put them in an "
              "environment of their own, so spaCR's own packages are not "
              "changed. Plaque mode works without them.",
              names=" and ".join(missing))


def parse_sizes(value: Any) -> Tuple[int, ...]:
    """Detector inference sizes from a setting value.

    :param value: ``'640,1280'``, a list of numbers, or one number.
    :returns: the positive sizes, in order; the defaults when none parse.
    """
    if isinstance(value, (list, tuple)):
        parts = [str(v) for v in value]
    else:
        parts = str(value or "").replace(";", ",").replace(" ", ",").split(",")
    out: List[int] = []
    for part in parts:
        try:
            size = int(float(part))
        except (TypeError, ValueError):
            continue
        if size > 0 and size not in out:
            out.append(size)
    return tuple(out) or DEFAULT_SIZES


def images_in(src: Any) -> List[Path]:
    """The images directly inside ``src``, as the run lists them.

    :param src: a folder, or one image file.
    :returns: image paths sorted by name; empty when there are none.
    """
    from ...plaque_papers import IMAGE_SUFFIXES

    path = Path(str(src or "")).expanduser()
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_SUFFIXES else []
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _to_uint8(array: np.ndarray) -> np.ndarray:
    """Stretch any image to 0..255 for display.

    :param array: the pixels.
    :returns: ``uint8`` pixels.
    """
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    data = array.astype(np.float32)
    low, high = np.percentile(data, (1.0, 99.8)) if data.size else (0.0, 1.0)
    if high <= low:
        high = low + 1.0
    return (np.clip((data - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def load_display_image(path: Any) -> np.ndarray:
    """One image as ``H x W x 3`` ``uint8`` RGB, whatever it was stored as.

    :param path: the image file.
    :returns: the pixels.
    """
    path = Path(path)
    array: Optional[np.ndarray] = None
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile

            array = np.asarray(tifffile.imread(str(path)))
        except Exception:
            array = None
    if array is None:
        from PIL import Image

        with Image.open(path) as handle:
            if handle.mode in ("I;16", "I;16B", "I", "F"):
                array = np.asarray(handle)
            else:
                array = np.asarray(handle.convert("RGB"))
    array = np.squeeze(array)
    if array.ndim == 3 and array.shape[0] in (1, 2, 3, 4) \
            and array.shape[-1] not in (1, 2, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 3 and array.shape[-1] > 3:
        array = array[..., :3]
    if array.ndim == 3 and array.shape[-1] == 2:
        array = array[..., 0]
    array = _to_uint8(array)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    return np.ascontiguousarray(array)


def outline_labels(rgb: np.ndarray, labels: np.ndarray,
                   colour: Tuple[int, int, int] = OUTLINE_COLOUR,
                   offset: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """Draw the boundary of every labelled object onto an image.

    :param rgb: ``H x W x 3`` image; changed in place.
    :param labels: a label image, 0 = background.
    :param colour: the outline colour.
    :param offset: ``(y, x)`` where ``labels`` sits inside ``rgb``.
    :returns: ``rgb``.
    """
    from skimage.segmentation import find_boundaries

    labels = np.asarray(labels)
    if labels.ndim != 2 or not labels.any():
        return rgb
    edges = find_boundaries(labels, mode="inner")
    y0, x0 = offset
    height = min(edges.shape[0], rgb.shape[0] - y0)
    width = min(edges.shape[1], rgb.shape[1] - x0)
    if height <= 0 or width <= 0:
        return rgb
    window = rgb[y0:y0 + height, x0:x0 + width]
    window[edges[:height, :width]] = colour
    return rgb


OVERLAY_OUTLINES = "outlines"
OVERLAY_FILL = "fill"
OVERLAY_DISPLAYS = (OVERLAY_OUTLINES, OVERLAY_FILL)
RANDOM_COLOUR = "random"
MAX_OUTLINE_THICKNESS = 20

IMAGE_TABS = ("Overlay", "Objects", "Cell probability", "Flows")


@dataclass(frozen=True)
class OverlayStyle:
    """How the segmented plaques are drawn over the image.

    Changing it redraws what is already segmented; nothing is run again.

    :param display: ``'outlines'`` or ``'fill'``.
    :param outline_colour: an ``(r, g, b)`` triple, or ``'random'`` for one
        colour per object.
    :param outline_thickness: outline width in pixels.
    :param fill_colour: an ``(r, g, b)`` triple, or ``'random'``.
    :param fill_opacity: fill opacity in percent, 0 to 100.
    """

    display: str = OVERLAY_OUTLINES
    outline_colour: Any = OUTLINE_COLOUR
    outline_thickness: int = 1
    fill_colour: Any = RANDOM_COLOUR
    fill_opacity: int = 40

    def normalised(self) -> "OverlayStyle":
        """The same style with every value clamped to what can be drawn.

        :returns: a new :class:`OverlayStyle`.
        """
        display = self.display if self.display in OVERLAY_DISPLAYS \
            else OVERLAY_OUTLINES
        return OverlayStyle(
            display=display,
            outline_colour=overlay_colour(self.outline_colour, OUTLINE_COLOUR),
            outline_thickness=max(1, min(MAX_OUTLINE_THICKNESS,
                                         int(self.outline_thickness or 1))),
            fill_colour=overlay_colour(self.fill_colour, RANDOM_COLOUR),
            fill_opacity=max(0, min(100, int(round(float(
                self.fill_opacity or 0))))))


_SESSION: Dict[str, OverlayStyle] = {"style": OverlayStyle()}


def overlay_colour(value: Any, default: Any = OUTLINE_COLOUR) -> Any:
    """A colour setting as ``'random'`` or an ``(r, g, b)`` triple.

    :param value: ``'random'``, a ``#rrggbb`` string, a :class:`QColor` or a
        triple.
    :param default: what an unreadable value means.
    :returns: ``'random'`` or a tuple of three ints in 0..255.
    """
    if isinstance(value, str):
        if value.strip().lower() == RANDOM_COLOUR:
            return RANDOM_COLOUR
        value = QColor(value.strip())
    if isinstance(value, QColor):
        if not value.isValid():
            return default
        return (value.red(), value.green(), value.blue())
    try:
        red, green, blue = (int(v) for v in list(value)[:3])
    except (TypeError, ValueError):
        return default
    return tuple(max(0, min(255, v)) for v in (red, green, blue))


def object_palette(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The object ids in a label image and one distinct colour for each.

    The palette is the Mask preview's ``color (random)`` one
    (:func:`spacr.qt.widgets.live_preview._random_outline_palette`), so an
    object keeps its colour across the Overlay and Objects tabs and across
    a change of thickness or opacity.

    :param labels: a label image, 0 = background.
    :returns: ``(ids, colours)``: sorted positive ids and an ``N x 3``
        ``uint8`` array.
    """
    from .live_preview import _random_outline_palette

    labels = np.asarray(labels)
    ids = np.unique(labels[labels > 0]).astype(np.int64)
    return ids, _random_outline_palette(ids)


def _per_pixel(labels: np.ndarray, where: np.ndarray, colour: Any
               ) -> np.ndarray:
    """The colour of every pixel in ``where``, by object or all the same.

    :param labels: the label image.
    :param where: the pixels to colour.
    :param colour: ``'random'`` or a triple.
    :returns: ``N x 3`` ``uint8``, one row per True pixel of ``where``.
    """
    picked = labels[where]
    if colour == RANDOM_COLOUR:
        ids, palette = object_palette(labels)
        return palette[np.searchsorted(ids, picked)]
    return np.tile(np.asarray(colour, dtype=np.uint8), (picked.size, 1))


def render_overlay(rgb: np.ndarray, labels: np.ndarray,
                   style: Optional[OverlayStyle] = None,
                   offset: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """Draw the labelled objects onto an image as outlines or a filled overlay.

    :param rgb: ``H x W x 3`` ``uint8`` image; changed in place.
    :param labels: a label image, 0 = background.
    :param style: the :class:`OverlayStyle`; yellow one-pixel outlines when
        None, which is what the preview always drew.
    :param offset: ``(y, x)`` where ``labels`` sits inside ``rgb``.
    :returns: ``rgb``.
    """
    style = (style or OverlayStyle()).normalised()
    labels = np.asarray(labels)
    if labels.ndim != 2 or not labels.any():
        return rgb
    y0, x0 = offset
    height = min(labels.shape[0], rgb.shape[0] - y0)
    width = min(labels.shape[1], rgb.shape[1] - x0)
    if height <= 0 or width <= 0:
        return rgb
    labels = labels[:height, :width].astype(np.int64)
    window = rgb[y0:y0 + height, x0:x0 + width]
    if style.display == OVERLAY_FILL:
        where = labels > 0
        alpha = style.fill_opacity / 100.0
        colours = _per_pixel(labels, where, style.fill_colour)
        blended = (window[where].astype(np.float32) * (1.0 - alpha)
                   + colours.astype(np.float32) * alpha)
        window[where] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
        return rgb
    from skimage.segmentation import find_boundaries

    owners = np.where(find_boundaries(labels, mode="inner"), labels, 0)
    thickness = style.outline_thickness
    if thickness > 1:
        from scipy.ndimage import grey_dilation

        grown = grey_dilation(owners, size=(thickness, thickness))
        owners = np.where(owners > 0, owners, grown)
    where = owners > 0
    if where.any():
        window[where] = _per_pixel(owners, where, style.outline_colour)
    return rgb


def render_objects(labels: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """The label image alone, one colour per object on black.

    :param labels: a label image, 0 = background, or None.
    :returns: ``H x W x 3`` ``uint8``, or None when there is nothing to draw.
    """
    if labels is None:
        return None
    labels = np.asarray(labels).astype(np.int64)
    if labels.ndim != 2:
        return None
    out = np.zeros(labels.shape + (3,), dtype=np.uint8)
    where = labels > 0
    if where.any():
        out[where] = _per_pixel(labels, where, RANDOM_COLOUR)
    return out


def render_cellprob(cellprob: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Cellpose's cell probability as a colour image.

    Cellpose hands the probability back in logits; it is drawn as the
    probability itself, 0 to 1 on a fixed scale, so two images can be
    compared by eye and the ``CP_prob`` threshold (a logit) sits at the
    colour of ``1 / (1 + e^-CP_prob)``.

    :param cellprob: ``H x W`` logits, or None.
    :returns: ``H x W x 3`` ``uint8`` on the ``magma`` scale, or None.
    """
    if cellprob is None:
        return None
    logits = np.asarray(cellprob, dtype=np.float32)
    if logits.ndim != 2:
        return None
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    try:
        from matplotlib import colormaps

        rgba = colormaps["magma"](prob)
        return np.ascontiguousarray(
            (rgba[..., :3] * 255.0).round().astype(np.uint8))
    except Exception:
        grey = (prob * 255.0).round().astype(np.uint8)
        return np.ascontiguousarray(np.stack([grey] * 3, axis=-1))


def _match_image(array: Optional[np.ndarray], shape: Tuple[int, int]
                 ) -> Optional[np.ndarray]:
    """Resize a flow picture or probability map onto ``shape`` when needed.

    :param array: ``H x W`` or ``H x W x 3``, or None.
    :param shape: ``(H, W)``.
    :returns: the array at ``shape``, in its own dtype, or None.
    """
    if array is None:
        return None
    array = np.asarray(array)
    if array.shape[:2] == tuple(shape):
        return array
    from skimage.transform import resize

    target = tuple(shape) + array.shape[2:]
    return resize(array, target, order=1, preserve_range=True,
                  anti_aliasing=False).astype(array.dtype)


def _catalogue() -> List[Any]:
    """The model zoo's listing, or nothing when it cannot be read.

    :returns: zoo entries.
    """
    try:
        from ... import model_zoo

        return list(model_zoo.catalogue())
    except Exception:
        LOG.debug("could not read the model zoo", exc_info=True)
        return []


def plaque_model_choices(entries: Optional[Iterable[Any]] = None) -> List[str]:
    """What the plaque model box offers: zoo plaque models, then ``bundled``.

    :param entries: zoo entries; the zoo's own listing when None.
    :returns: setting values, newest zoo model first.
    """
    entries = _catalogue() if entries is None else list(entries)
    keys = sorted({str(getattr(e, "key", "") or "") for e in entries
                   if "plaque" in str(getattr(e, "key", "") or "")
                   and str(getattr(e, "kind", "")) == "cellpose"},
                  reverse=True)
    if DEFAULT_PLAQUE_MODEL not in keys:
        keys.insert(0, DEFAULT_PLAQUE_MODEL)
    return keys + ["bundled"]


def detector_choices(entries: Optional[Iterable[Any]] = None) -> List[str]:
    """What the detector box offers: the zoo's plaque-image detectors.

    :param entries: zoo entries; the zoo's own listing when None.
    :returns: zoo keys, newest first.
    """
    entries = _catalogue() if entries is None else list(entries)
    keys = sorted({str(getattr(e, "key", "") or "") for e in entries
                   if str(getattr(e, "kind", "")) == "detector"
                   and "well_detector" in str(getattr(e, "key", "") or "")},
                  reverse=True)
    if DEFAULT_DETECTOR not in keys:
        keys.insert(0, DEFAULT_DETECTOR)
    return keys


BUNDLED_NOTE = ("'bundled' is the historical packaged plaque model. It is "
                "kept so an old run can be reproduced; it misses about a "
                "third of the plaques on literature images. "
                "toxoplasma_plaque_v2 is the current model.")


def resolve_plaque_model(settings: Dict[str, Any]) -> Tuple[str, str, Any]:
    """The checkpoint the plaque run would segment with, found without a download.

    The run's own resolver answers (:func:`spacr.submodules._resolve_plaque_model`
    with ``fetch=False``), so the preview cannot drift from it.

    :param settings: the module's settings.
    :returns: ``(path, note, entry)``: the path, or ``''`` when it is not on
        this machine; a sentence to show; and the zoo entry that a download
        would fetch, when there is one.
    """
    from ... import submodules as sm

    requested = str(settings.get("plaque_model") or "bundled")
    note = tr(BUNDLED_NOTE) if requested == "bundled" else ""
    try:
        path = sm._resolve_plaque_model(dict(settings), fetch=False)
    except sm.ModelZooMissing as exc:
        entry = None
        if requested != "bundled":
            entry = next((e for e in _catalogue()
                          if getattr(e, "key", None) == requested), None)
        text = tr("{model} is not on this machine: {reason}",
                  model=requested, reason=exc)
        return "", (note + " " + text).strip(), entry
    except Exception as exc:
        return "", (note + " " + str(exc)).strip(), None
    return str(path), note, None


def _explain_model_failure(path: str, exc: BaseException) -> str:
    """A readable sentence for a checkpoint Cellpose could not load.

    A Cellpose 3 checkpoint loaded under Cellpose 4 fails deep inside torch;
    :func:`spacr.submodules.explain_cellpose3` turns that into a sentence
    when it recognises it.

    :param path: the checkpoint.
    :param exc: what loading raised.
    :returns: the sentence.
    """
    try:
        from ... import submodules as sm

        explained = sm.explain_cellpose3(exc, path)
        if explained is not exc:
            return str(explained)
    except Exception:
        LOG.debug("could not explain the model failure", exc_info=True)
    return preview_failure_message(f"{type(exc).__name__}: {exc}")


def resolve_detector(key: Any, src: Any = None) -> Tuple[str, str, Any]:
    """The detector checkpoint a key names, found without a download.

    Looks where the run looks: the path itself, the zoo entry's own path,
    ``~/.spacr/models`` and the run's model cache under
    ``<src>/plaque_figures/models``.

    :param key: a zoo key or a checkpoint path.
    :param src: the figure folder.
    :returns: ``(path, note, entry)``; ``path`` is ``''`` when it is not
        here, and ``entry`` is what a download would fetch.
    """
    text = str(key or DEFAULT_DETECTOR)
    if Path(text).expanduser().is_file():
        return str(Path(text).expanduser()), "", None
    entry = next((e for e in _catalogue() if getattr(e, "key", None) == text),
                 None)
    if entry is None:
        return "", tr("{key} is neither a checkpoint file nor a model zoo "
                      "detector.", key=text), None
    name = str(getattr(entry, "name", "") or "")
    candidates = [str(getattr(entry, "path", "") or ""),
                  os.path.join(os.path.expanduser("~"), ".spacr", "models", name)]
    if src:
        candidates.append(os.path.join(str(src), "plaque_figures", "models", name))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate, "", entry
    return "", tr("The detector {key} is not on this machine yet. Download "
                  "it here, or the run downloads it.", key=text), entry


def write_legend(path: Any, stem: str, legend: str) -> Path:
    """Store one figure's legend in ``legends.csv``, keeping the others.

    :param path: the CSV (``file`` and ``legend`` columns).
    :param stem: the figure's file stem.
    :param legend: the legend text.
    :returns: the path written.
    """
    from ...plaque_papers import read_legends

    path = Path(path)
    kept = read_legends(path)
    kept[Path(stem).stem] = " ".join(str(legend or "").split())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "legend"])
        for name, text in sorted(kept.items()):
            writer.writerow([name, text])
    return path


_MODELS: Dict[str, Any] = {}
_MODELS_LOCK = threading.Lock()


def _cellpose_model(path: str):
    """One Cellpose model per checkpoint, kept for the next pass.

    Only the last one is kept: a plaque checkpoint is 1.2 GB in memory.

    :param path: the checkpoint.
    :returns: the model.
    """
    from .preview_contract import preview_cellpose_model

    with _MODELS_LOCK:
        if path not in _MODELS:
            _MODELS.clear()
            _MODELS[path] = preview_cellpose_model(path)
        return _MODELS[path]


def _match_shape(labels: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour a label image onto ``shape`` when they differ.

    :param labels: the label image.
    :param shape: ``(H, W)``.
    :returns: the label image at ``shape``.
    """
    labels = np.asarray(labels)
    if labels.shape[:2] == tuple(shape):
        return labels
    from skimage.transform import resize

    return resize(labels, shape, order=0, preserve_range=True,
                  anti_aliasing=False).astype(labels.dtype)


def plaque_pass(path: Any, settings: Dict[str, Any], *,
                segment: Optional[Callable[[Path], np.ndarray]] = None
                ) -> Dict[str, Any]:
    """Plaque mode on one image: segment it and count what was found.

    Runs on a worker thread; touches no widget.

    :param path: the image.
    :param settings: the module's settings.
    :param segment: ``fn(path) -> labels`` or ``fn(path) -> (labels,
        flows)`` with ``flows`` as
        :func:`spacr.plaque.plaque_flow_outputs` gives it; the run's model,
        with its flows, when None.
    :returns: ``{'path', 'image', 'labels', 'overlay', 'flow_rgb',
        'cellprob', 'count', 'mean_area', 'areas', 'note'}``, or
        ``{'error', 'entry'}`` when the model cannot be used. ``flow_rgb``
        and ``cellprob`` are None when the segmenter gave no flows.
    """
    from ...plaque_papers import measure_region

    path = Path(path)
    note = ""
    if segment is None:
        model_path, note, entry = resolve_plaque_model(settings)
        if not model_path:
            return {"error": note, "entry": entry}
        try:
            model = _cellpose_model(model_path)
        except Exception as exc:
            return {"error": _explain_model_failure(model_path, exc)}

        def segment(p: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
            """The plaque label mask of the image at ``p``, with its flows."""
            return segment_plaque_image(model, load_display_image(p),
                                        settings, return_flows=True)

    rgb = load_display_image(path)
    segmented = segment(path)
    flows: Dict[str, Any] = {}
    if isinstance(segmented, tuple):
        segmented, flows = segmented[0], dict(segmented[1] or {})
    labels = _match_shape(segmented, rgb.shape[:2])
    rows = measure_region(labels)
    areas = [row["area_px"] for row in rows]
    overlay = outline_labels(rgb.copy(), labels)
    return {"path": str(path), "image": rgb, "labels": labels,
            "overlay": overlay,
            "flow_rgb": _match_image(flows.get("flow_rgb"), rgb.shape[:2]),
            "cellprob": _match_image(flows.get("cellprob"), rgb.shape[:2]),
            "count": len(rows),
            "mean_area": float(np.mean(areas)) if areas else 0.0,
            "areas": areas, "note": note}


def figure_pass(path: Any, settings: Dict[str, Any], *,
                detect: Optional[Callable] = None,
                read_text: Optional[Callable] = None,
                segment: Optional[Callable[[np.ndarray], np.ndarray]] = None
                ) -> Dict[str, Any]:
    """Figure mode on one figure: find, read and segment, as the run does.

    The annotation itself is left to :func:`annotate_figure`, which is cheap
    and is repeated whenever a legend is pasted.

    :param path: the figure.
    :param settings: the module's settings.
    :param detect: passed to :func:`spacr.plaque_papers.find_plaque_regions`;
        the zoo detector when None.
    :param read_text: ``fn(path) -> [Word]``; RapidOCR, followed by the
        enlarged second reading, when None.
    :param segment: ``fn(crop) -> labels``; the run's plaque model when None.
    :returns: ``{'path', 'overlay', 'regions', 'words', 'counts',
        'mean_areas', 'note'}``, or ``{'error', 'entry'}``.
    """
    from ...plaque_papers import (_load_image, find_plaque_regions,
                                  measure_region, read_words, reread_around)

    path = Path(path)
    weights = "fake"
    if detect is None:
        weights, why, entry = resolve_detector(
            settings.get("figure_detector"), settings.get("src"))
        if not weights:
            return {"error": why, "entry": entry}
    note = ""
    if segment is None:
        model_path, note, entry = resolve_plaque_model(settings)
        if not model_path:
            return {"error": note, "entry": entry}
        try:
            model = _cellpose_model(model_path)
        except Exception as exc:
            return {"error": _explain_model_failure(model_path, exc)}

        def segment(crop: np.ndarray) -> np.ndarray:
            """The Cellpose label mask of one plaque-well crop."""
            return segment_plaque_image(model, crop, settings)

    image = _load_image(path)
    regions = find_plaque_regions(
        image, weights, imgsz=parse_sizes(settings.get("figure_imgsz")),
        confidence=float(settings.get("figure_confidence") or 0.25),
        detect=detect)
    words: List[Any] = []
    if regions and settings.get("figure_read_text", True) not in (False, "False"):
        words = list((read_text or read_words)(path))
        if read_text is None:
            words = reread_around(image, regions, words)
    overlay = image.copy()
    counts: List[int] = []
    mean_areas: List[float] = []
    for region in regions:
        crop = image[region.y0:region.y1, region.x0:region.x1]
        labels = _match_shape(segment(crop), crop.shape[:2])
        rows = measure_region(labels)
        counts.append(len(rows))
        mean_areas.append(float(np.mean([r["area_px"] for r in rows]))
                          if rows else 0.0)
        outline_labels(overlay, labels, offset=(region.y0, region.x0))
    return {"path": str(path), "overlay": overlay, "regions": regions,
            "words": words, "counts": counts, "mean_areas": mean_areas,
            "note": note}


def annotate_figure(result: Dict[str, Any], caption: str, src: Any, *,
                    confirm: bool, options: Any = None) -> List[Any]:
    """Propose a condition per image, with any saved review applied.

    :param result: a :func:`figure_pass` result.
    :param caption: the figure's legend, ``''`` when unknown.
    :param src: the figure folder, where ``figure_annotations.csv`` lives.
    :param confirm: ``confirm_annotations``. When on, an image nobody has
        approved starts unticked; when off it starts ticked, because the run
        measures it.
    :param options: a :class:`spacr.plaque_papers.TextOptions` -- the Text
        detection settings; the defaults when None.
    :returns: :class:`spacr.plaque_papers.Annotation` per region.
    """
    from ...plaque_papers import (ANNOTATIONS_FILE, annotate_regions,
                                  apply_overrides, read_annotation_overrides)

    stem = Path(result["path"]).stem
    found = annotate_regions(result["regions"], result["words"],
                             caption=caption, figure_label=stem,
                             options=options)
    overrides = read_annotation_overrides(Path(str(src)) / ANNOTATIONS_FILE) \
        if src else {}
    apply_overrides(stem, found, overrides, confirm_each=confirm)
    for a in found:
        if a.approved is None:
            a.approved = not confirm
    return found


def _figure_scales(result: Dict[str, Any], annotations: Sequence[Any],
                  caption: str, plate_format: Any = None) -> List[Any]:
    """The ruler of every well in a figure, as the run finds it.

    :param result: a :func:`detect_figure` result.
    :param annotations: its annotations, for each panel's legend passage.
    :param caption: the figure's legend.
    :param plate_format: the ``plate_format`` setting, or None.
    :returns: :class:`spacr.plaque_papers._Scale` per region; empty when the
        result has no image.
    """
    from ...plaque_papers import _scales_for_regions

    image = result.get("image")
    if image is None or not result.get("regions"):
        return []
    fmt = str(plate_format).strip() if plate_format not in (None, "", "None") else None
    return _scales_for_regions(image, result["regions"], result.get("words", []),
                              caption=caption, annotations=annotations,
                              plate_format=fmt)


def detect_figure(path: Any, settings: Dict[str, Any], *,
                  detect: Optional[Callable] = None,
                  read_text: Optional[Callable] = None) -> Dict[str, Any]:
    """Figure mode's Run preview: find the plaque wells and read the text.

    Nothing is segmented here: Run preview detects the plaque wells, and a
    separate Plaque preview button segments the plaques of the highlighted
    row and well box (:func:`segment_well`).

    :param path: the figure.
    :param settings: the module's settings.
    :param detect: passed to :func:`spacr.plaque_papers.find_plaque_regions`;
        the zoo detector when None.
    :param read_text: ``fn(path) -> [Word]``; RapidOCR, followed by the
        enlarged second reading, when None.
    :returns: ``{'path', 'image', 'regions', 'words', 'words_source'}``,
        or ``{'error', 'entry'}``. A PDF's text layer saved beside the
        figure (``text_layer.json``) is read before any OCR.
    """
    from ...plaque_papers import (TEXT_LAYER_FILE, _load_image, _figure_words,
                                  find_plaque_regions, _read_text_layer,
                                  read_words, text_options_from_settings)

    path = Path(path)
    options = text_options_from_settings(settings)
    weights = "fake"
    if detect is None:
        weights, why, entry = resolve_detector(
            settings.get("figure_detector"), settings.get("src"))
        if not weights:
            return {"error": why, "entry": entry}
    image = _load_image(path)
    regions = find_plaque_regions(
        image, weights, imgsz=parse_sizes(settings.get("figure_imgsz")),
        confidence=float(settings.get("figure_confidence") or 0.25),
        detect=detect)
    layer = _read_text_layer(path.parent / TEXT_LAYER_FILE).get(path.stem, [])
    reading = settings.get("figure_read_text", True) not in (False, "False")
    words, words_source = _figure_words(
        image, regions, layer, path=path,
        read_text=(read_text or read_words) if reading else None,
        options=options, default_reader=read_text is None)
    return {"path": str(path), "image": image, "regions": list(regions),
            "words": words, "words_source": words_source,
            "read_with": {k: settings.get(k) for k in TEXT_READ_KEYS}}


def region_at(regions: Sequence[Any], x: float, y: float) -> Optional[int]:
    """Which box a point on the figure falls in.

    :param regions: the detected wells.
    :param x: figure column.
    :param y: figure row.
    :returns: the index of the smallest box holding the point, or None.
    """
    best: Optional[int] = None
    best_area = None
    for index, r in enumerate(regions):
        if r.x0 <= x <= r.x1 and r.y0 <= y <= r.y1:
            area = r.width * r.height
            if best_area is None or area < best_area:
                best, best_area = index, area
    return best


def plaque_rows(labels: np.ndarray) -> List[Dict[str, Any]]:
    """One row per plaque, with the per-plaque values the run writes.

    The columns of the run's ``per_plaque`` table
    (:func:`spacr.submodules.analyze_plaques`), with the ratio taken to the
    median plaque in this well. The ratio to the panel median needs every
    well of the panel and is added by the caller.

    :param labels: a label image, 0 = background.
    :returns: dicts with ``plaque_id``, ``area_px``, ``area_vs_well_median``,
        ``perimeter_px``, ``equivalent_diameter_px``, ``eccentricity``,
        ``solidity``, ``centroid_y`` and ``centroid_x``.
    """
    from skimage.measure import regionprops

    labels = np.asarray(labels)
    if labels.ndim != 2 or not labels.any():
        return []
    props = regionprops(labels.astype(np.int32))
    areas = [float(p.area) for p in props]
    median = float(np.median(areas)) if areas else 0.0
    rows = []
    for p in props:
        diameter = getattr(p, "equivalent_diameter_area", None)
        if diameter is None:
            diameter = p.equivalent_diameter
        rows.append({
            "plaque_id": int(p.label), "area_px": int(p.area),
            "area_vs_well_median": float(p.area) / median if median else None,
            "perimeter_px": float(p.perimeter),
            "equivalent_diameter_px": float(diameter),
            "eccentricity": float(p.eccentricity),
            "solidity": float(p.solidity),
            "centroid_y": float(p.centroid[0]),
            "centroid_x": float(p.centroid[1])})
    return rows


def segment_well(image: np.ndarray, region: Any, settings: Dict[str, Any], *,
                 segment: Optional[Callable[[np.ndarray], np.ndarray]] = None
                 ) -> Dict[str, Any]:
    """Find the plaques in one detected well of a figure.

    Runs on a worker thread; touches no widget. The model and its thresholds
    are the Plaque settings, through the same
    :func:`spacr.plaque.segment_plaque_image` Plaque mode uses.

    :param image: the figure, ``H x W x 3``.
    :param region: the well's box.
    :param settings: the module's settings.
    :param segment: ``fn(crop) -> labels``; the plaque model when None.
    :returns: ``{'labels', 'rows', 'count', 'mean_area', 'note'}``, or
        ``{'error', 'entry'}``.
    """
    crop = np.ascontiguousarray(image[region.y0:region.y1,
                                      region.x0:region.x1])
    note = ""
    if segment is None:
        model_path, note, entry = resolve_plaque_model(settings)
        if not model_path:
            return {"error": note, "entry": entry}
        try:
            model = _cellpose_model(model_path)
        except Exception as exc:
            return {"error": _explain_model_failure(model_path, exc)}

        def segment(c: np.ndarray) -> np.ndarray:
            """The plaque label mask of one well crop."""
            return segment_plaque_image(model, c, settings)

    labels = _match_shape(segment(crop), crop.shape[:2])
    rows = plaque_rows(labels)
    areas = [row["area_px"] for row in rows]
    return {"labels": labels, "rows": rows, "count": len(rows),
            "mean_area": float(np.mean(areas)) if areas else 0.0,
            "note": note}


def paper_folder_name(reference: Any) -> str:
    """The sub-folder a paper's figures are fetched into.

    :param reference: a DOI, PMID, PMC id or PDF path.
    :returns: a name safe on every file system: a PDF's stem, or the
        reference with every character that is not a letter, digit, dot or
        dash turned into ``_``.
    """
    import re

    text = str(reference or "").strip()
    if text.lower().endswith(".pdf"):
        text = Path(text).stem
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
    name = re.sub(r"[^A-Za-z0-9.\-]+", "_", text).strip("._")
    return name or "paper"


class PaperDialog(QDialog):
    """Ask which paper to fetch, and where its folder goes.

    :param parent: the owning widget.
    :param folder: the folder the paper's own folder is made in.
    """

    def __init__(self, parent: Optional[QWidget] = None, folder: str = ""):
        """Build the two fields.

        :param parent: the owning widget.
        :param folder: the starting parent folder.
        """
        super().__init__(parent)
        self.setObjectName("PlaquePaperDialog")
        self.setWindowTitle(tr("Figures from a paper"))
        layout = QVBoxLayout(self)
        intro = QLabel(tr(
            "A DOI, PMID or PMC id is fetched from Europe PMC with its figure "
            "legends; a PDF is read page by page. The figures go into a new "
            "folder, with their legends in legends.csv, and the preview "
            "switches to it."))
        intro.setWordWrap(True)
        layout.addWidget(intro)
        grid = QGridLayout()
        grid.addWidget(QLabel(tr("Paper")), 0, 0)
        self.reference = QLineEdit(self)
        self.reference.setPlaceholderText(tr("DOI, PMID or PMC id"))
        grid.addWidget(self.reference, 0, 1)
        pdf = QToolButton(self)
        pdf.setText(tr("PDF…"))
        pdf.clicked.connect(self._pick_pdf)
        grid.addWidget(pdf, 0, 2)
        grid.addWidget(QLabel(tr("Into")), 1, 0)
        self.folder = QLineEdit(folder, self)
        grid.addWidget(self.folder, 1, 1)
        where = QToolButton(self)
        where.setText(tr("Browse…"))
        where.clicked.connect(self._pick_folder)
        grid.addWidget(where, 1, 2)
        layout.addLayout(grid)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(tr("Fetch"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setMinimumWidth(520)

    def _pick_pdf(self) -> None:
        """Choose a PDF instead of typing an identifier."""
        chosen, _ = QFileDialog.getOpenFileName(
            self, tr("Choose a paper"), "", tr("PDF files (*.pdf)"))
        if chosen:
            self.reference.setText(chosen)

    def _pick_folder(self) -> None:
        """Choose where the paper's folder is made."""
        chosen = QFileDialog.getExistingDirectory(
            self, tr("Put the paper's folder in"), self.folder.text())
        if chosen:
            self.folder.setText(chosen)

    def values(self) -> Tuple[str, str]:
        """The reference and the parent folder, stripped.

        :returns: ``(reference, folder)``.
        """
        return self.reference.text().strip(), self.folder.text().strip()


TEXT_LABELS = (
    ("text_use_above", "Use the column header"),
    ("text_reach_above", "Reach above (image heights)"),
    ("text_use_left", "Use the row label"),
    ("text_reach_left", "Reach left (image widths)"),
    ("text_use_below", "Use the text below"),
    ("text_reach_below", "Reach below (image heights)"),
    ("text_panel_reach", "Panel letter reach (image sizes)"),
    ("text_min_confidence", "Minimum OCR confidence"),
    ("text_ignore", "Ignore words matching"),
    ("text_order", "Order in the condition"),
    ("text_separator", "Separator"),
    ("text_reread", "Read again, enlarged"),
    ("text_reread_scale", "Enlargement for the second reading"),
)


def _setting_tooltip(key: str) -> str:
    """The setting's own tooltip, so the control and the form say the same.

    :param key: a setting name.
    :returns: the tooltip, or ``''``.
    """
    try:
        from ...settings import tooltips

        return str(tooltips.get(key, "") or "")
    except Exception:
        return ""


class PlaqueSettingsDialog(QDialog):
    """The Plaque Assay preview's settings: a Figure tab and a Plaque detection tab.

    The controls belong to the panel and are lent to this window while it is
    open, so a value set here is the value the next pass uses, and it
    survives the window being closed -- :meth:`give_back` returns them.

    In Plaque mode the Figure tab is hidden: nothing there applies to a
    folder of cropped plaque images.

    :param panel: the :class:`PlaquePreviewPanel` whose controls it shows.
    """

    def __init__(self, panel: "PlaquePreviewPanel"):
        """Lay the panel's controls out in two tabs.

        :param panel: the preview panel; also the window's parent.
        """
        from PySide6.QtWidgets import QFormLayout

        super().__init__(panel)
        self._panel = panel
        self.setObjectName("PlaqueSettingsDialog")
        self.setWindowTitle(tr("Plaque Assay preview settings"))
        outer = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("PlaqueSettingsTabs")

        figure = QWidget()
        form = QFormLayout(figure)
        form.addRow(tr("Detector"), panel._detector_row)
        form.addRow(tr("Inference sizes"), panel._sizes)
        form.addRow(tr("Confidence"), panel._confidence)
        form.addRow("", panel._read_text)
        form.addRow("", panel._confirm)
        self.figure_tab = figure

        text = QWidget()
        form = QFormLayout(text)
        for key, label in TEXT_LABELS:
            widget = panel._text[key]
            if isinstance(widget, QCheckBox):
                form.addRow("", widget)
            else:
                form.addRow(tr(label), widget)
        note = QLabel(tr("Changes re-propose the conditions at once from the "
                         "text already read. The second reading needs a new "
                         "Run preview."))
        note.setWordWrap(True)
        form.addRow(note)
        self.text_tab = text

        plaque = QWidget()
        form = QFormLayout(plaque)
        form.addRow(tr("Plaque model"), panel._model_row)
        form.addRow(tr("Diameter"), panel._diameter)
        form.addRow(tr("Flow threshold"), panel._flow)
        form.addRow(tr("Cell probability"), panel._cellprob)
        self.plaque_tab = plaque

        self.tabs.addTab(figure, tr("Figure"))
        self.tabs.addTab(text, tr("Text detection"))
        self.tabs.addTab(plaque, tr("Plaque detection"))
        outer.addWidget(self.tabs)
        for widget in self._lent():
            widget.show()
        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        outer.addWidget(buttons)
        self.setMinimumWidth(460)

    def _lent(self) -> Tuple[QWidget, ...]:
        """The panel controls this window holds while it is open."""
        p = self._panel
        return (p._detector_row, p._sizes, p._confidence, p._read_text,
                p._confirm, p._model_row, p._diameter, p._flow, p._cellprob,
                *p._text.values())

    def show_mode(self, mode: Any, tab: Optional[str] = None) -> None:
        """Show the tabs ``mode`` uses and open the right one.

        :param mode: ``'plaque'`` or ``'figure'``.
        :param tab: ``'figure'``, ``'text'`` or ``'plaque'`` to open on.
        """
        figure = normalise_mode(mode) == FIGURE_MODE
        self.tabs.setTabVisible(0, figure)
        self.tabs.setTabVisible(1, figure)
        wanted = tab or ("figure" if figure else "plaque")
        if not figure:
            wanted = "plaque"
        self.tabs.setCurrentIndex({"figure": 0, "text": 1}.get(wanted, 2))

    def give_back(self) -> None:
        """Return every lent control to the panel, hidden, values intact."""
        holder = self._panel._controls
        for widget in self._lent():
            widget.setParent(holder)
            widget.hide()


class _ColourChoice(QWidget):
    """A colour setting: one fixed colour from a swatch, or random per object.

    :param value: ``'random'`` or an ``(r, g, b)`` triple to start on.
    :param parent: the owning widget.
    """

    changed = Signal(object)

    def __init__(self, value: Any, parent: Optional[QWidget] = None):
        """Build the choice and the swatch.

        :param value: the starting colour.
        :param parent: the owning widget.
        """
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.kind = QComboBox(self)
        self.kind.addItem(tr("Fixed colour"), "fixed")
        self.kind.addItem(tr("Random (per object)"), RANDOM_COLOUR)
        self.swatch = QToolButton(self)
        self.swatch.setToolTip(tr("Choose the colour"))
        self.swatch.setMinimumWidth(64)
        row.addWidget(self.kind, 1)
        row.addWidget(self.swatch)
        self._fixed: Tuple[int, int, int] = OUTLINE_COLOUR
        self.set_value(value)
        self.kind.currentIndexChanged.connect(self._on_kind)
        self.swatch.clicked.connect(self._choose)

    def value(self) -> Any:
        """``'random'`` or the fixed ``(r, g, b)``."""
        if self.kind.currentData() == RANDOM_COLOUR:
            return RANDOM_COLOUR
        return self._fixed

    def set_value(self, value: Any) -> None:
        """Show ``value`` without announcing it.

        :param value: ``'random'`` or a colour.
        """
        value = overlay_colour(value, self._fixed)
        blocked = self.kind.blockSignals(True)
        if value == RANDOM_COLOUR:
            self.kind.setCurrentIndex(1)
        else:
            self._fixed = value
            self.kind.setCurrentIndex(0)
        self.kind.blockSignals(blocked)
        self._paint()

    def _paint(self) -> None:
        """Show the fixed colour on the swatch, or a hue ramp when random.

        A disabled choice -- the settings of the display not shown -- drops
        the colour so it reads as off, like every other disabled control.
        """
        colour = QColor(*self._fixed)
        random = self.kind.currentData() == RANDOM_COLOUR
        self.swatch.setEnabled(not random)
        self.swatch.setText("" if random else colour.name())
        if not self.isEnabled():
            self.swatch.setStyleSheet("")
        elif random:
            self.swatch.setStyleSheet(
                "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
                "stop:0 #ff3b3b, stop:0.25 #ffd600, stop:0.5 #2bd96b, "
                "stop:0.75 #2b8cff, stop:1 #d63bff);")
        else:
            self.swatch.setStyleSheet(
                f"background-color: {colour.name()}; "
                f"color: {'#000' if colour.lightness() > 127 else '#fff'};")

    def changeEvent(self, event):                            # noqa: N802
        """Repaint the swatch when the choice is enabled or disabled.

        :param event: the change event.
        """
        super().changeEvent(event)
        from PySide6.QtCore import QEvent

        if event.type() == QEvent.Type.EnabledChange:
            self._paint()

    def _on_kind(self, _index: int) -> None:
        """Fixed or random was picked."""
        self._paint()
        self.changed.emit(self.value())

    def _choose(self) -> None:
        """Ask for a colour with the GUI's one picker, and keep a real one."""
        from .colour_picker import pick_colour

        colour = pick_colour(self, QColor(*self._fixed), tr("Overlay colour"))
        if colour.isValid():
            self.set_fixed(colour)

    def set_fixed(self, colour: Any) -> None:
        """Make ``colour`` the fixed colour, select it and announce it.

        :param colour: anything :func:`overlay_colour` reads.
        """
        value = overlay_colour(colour, self._fixed)
        if value == RANDOM_COLOUR:
            value = self._fixed
        self._fixed = value
        blocked = self.kind.blockSignals(True)
        self.kind.setCurrentIndex(0)
        self.kind.blockSignals(blocked)
        self._paint()
        self.changed.emit(self.value())


class PlaqueOverlayDialog(QDialog):
    """How the plaques are drawn: outlines or a filled overlay, and in what.

    Opened from a right-click on the preview image. Every change is applied
    at once to what is already segmented, so the window stays open beside
    the picture it changes; nothing is segmented again. A ``QDialog``, so
    :mod:`spacr.qt.widgets.glass` gives it the rounded, translucent card of
    the other settings windows.

    :param style: the :class:`OverlayStyle` to start from.
    :param parent: the preview panel.
    """

    style_changed = Signal(object)

    def __init__(self, style: OverlayStyle,
                 parent: Optional[QWidget] = None):
        """Lay the controls out.

        :param style: the starting style.
        :param parent: the owning widget.
        """
        from PySide6.QtWidgets import QFormLayout, QGroupBox, QSlider

        super().__init__(parent)
        style = style.normalised()
        self.setObjectName("PlaqueOverlayDialog")
        self.setWindowTitle(tr("Plaque overlay"))
        outer = QVBoxLayout(self)

        top = QFormLayout()
        self.display = QComboBox(self)
        self.display.addItem(tr("Outlines"), OVERLAY_OUTLINES)
        self.display.addItem(tr("Filled overlay"), OVERLAY_FILL)
        self.display.setCurrentIndex(OVERLAY_DISPLAYS.index(style.display))
        top.addRow(tr("Display"), self.display)
        outer.addLayout(top)

        self.outline_group = QGroupBox(tr("Outlines"), self)
        form = QFormLayout(self.outline_group)
        self.outline_colour = _ColourChoice(style.outline_colour,
                                            self.outline_group)
        form.addRow(tr("Colour"), self.outline_colour)
        self.thickness = QSpinBox(self.outline_group)
        self.thickness.setRange(1, MAX_OUTLINE_THICKNESS)
        self.thickness.setSuffix(" px")
        self.thickness.setValue(style.outline_thickness)
        form.addRow(tr("Thickness"), self.thickness)
        outer.addWidget(self.outline_group)

        self.fill_group = QGroupBox(tr("Fill"), self)
        form = QFormLayout(self.fill_group)
        self.fill_colour = _ColourChoice(style.fill_colour, self.fill_group)
        form.addRow(tr("Colour"), self.fill_colour)
        opacity_row = QWidget(self.fill_group)
        row = QHBoxLayout(opacity_row)
        row.setContentsMargins(0, 0, 0, 0)
        self.opacity_slider = QSlider(Qt.Horizontal, opacity_row)
        self.opacity_slider.setRange(0, 100)
        self.opacity = QSpinBox(opacity_row)
        self.opacity.setRange(0, 100)
        self.opacity.setSuffix(" %")
        self.opacity.setValue(style.fill_opacity)
        self.opacity_slider.setValue(style.fill_opacity)
        self.opacity_slider.valueChanged.connect(self.opacity.setValue)
        self.opacity.valueChanged.connect(self.opacity_slider.setValue)
        row.addWidget(self.opacity_slider, 1)
        row.addWidget(self.opacity)
        form.addRow(tr("Opacity"), opacity_row)
        outer.addWidget(self.fill_group)

        note = QLabel(tr("Applies at once to the plaques already found; "
                         "nothing is segmented again."))
        note.setWordWrap(True)
        outer.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self.display.currentIndexChanged.connect(self._changed)
        self.outline_colour.changed.connect(self._changed)
        self.thickness.valueChanged.connect(self._changed)
        self.fill_colour.changed.connect(self._changed)
        self.opacity.valueChanged.connect(self._changed)
        self._enable_groups()
        self.setMinimumWidth(360)

    def overlay_style(self) -> OverlayStyle:
        """The style the controls describe."""
        return OverlayStyle(
            display=self.display.currentData(),
            outline_colour=self.outline_colour.value(),
            outline_thickness=self.thickness.value(),
            fill_colour=self.fill_colour.value(),
            fill_opacity=self.opacity.value()).normalised()

    def set_overlay_style(self, style: OverlayStyle) -> None:
        """Show ``style`` without announcing it.

        :param style: the style to show.
        """
        style = style.normalised()
        widgets = (self.display, self.thickness, self.opacity,
                   self.opacity_slider)
        states = [w.blockSignals(True) for w in widgets]
        self.display.setCurrentIndex(OVERLAY_DISPLAYS.index(style.display))
        self.thickness.setValue(style.outline_thickness)
        self.opacity.setValue(style.fill_opacity)
        self.opacity_slider.setValue(style.fill_opacity)
        for widget, state in zip(widgets, states):
            widget.blockSignals(state)
        self.outline_colour.set_value(style.outline_colour)
        self.fill_colour.set_value(style.fill_colour)
        self._enable_groups()

    def _enable_groups(self) -> None:
        """Only the settings of the chosen display can be edited."""
        fill = self.display.currentData() == OVERLAY_FILL
        self.outline_group.setEnabled(not fill)
        self.fill_group.setEnabled(fill)

    def _changed(self, *_args: Any) -> None:
        """Any control moved: announce the style it now describes."""
        self._enable_groups()
        self.style_changed.emit(self.overlay_style())


class PlaqueModeSwitch(QWidget):
    """A two-button Plaque | Figure switch.

    :param parent: the owning widget.
    """

    mode_changed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        """Build the two buttons, Plaque selected.

        :param parent: the owning widget.
        """
        super().__init__(parent)
        self.setObjectName("PlaqueModeSwitch")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        caption = QLabel(tr("Mode"))
        caption.setObjectName("PlaqueModeCaption")
        row.addWidget(caption)
        row.addSpacing(8)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: Dict[str, QToolButton] = {}
        tips = {
            PLAQUE_MODE: tr("Plaque mode: a folder of cropped plaque images. "
                            "Every image is segmented, and the run writes a "
                            "table of per-image values and a table of "
                            "per-plaque values."),
            FIGURE_MODE: tr("Figure mode: a folder of published figures. The "
                            "run finds the plaque images in each figure, "
                            "reads the text around them to name each "
                            "condition, and segments the plaques."),
        }
        for mode, text in ((PLAQUE_MODE, tr("Plaque")),
                           (FIGURE_MODE, tr("Figure"))):
            button = QToolButton(self)
            button.setText(text)
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setToolTip(tips[mode])
            button.setObjectName(f"PlaqueModeButton_{mode}")
            button.setMinimumWidth(84)
            self._group.addButton(button)
            self._buttons[mode] = button
            row.addWidget(button)
            button.toggled.connect(
                lambda on, m=mode: on and self.mode_changed.emit(m))
        row.addStretch(1)
        self._buttons[PLAQUE_MODE].setChecked(True)
        self._restyle()

    def _restyle(self) -> None:
        """Colour the selected half with the theme's accent."""
        try:
            from ..theme import active_palette

            palette = active_palette()
        except Exception:
            palette = {}
        accent = palette.get("accent", "#3b82f6")
        border = palette.get("border", "#666666")
        fg = palette.get("fg", "#dddddd")
        ink = palette.get("bg", "#000000")
        self.setStyleSheet(
            f"QToolButton {{ padding: 5px 14px; border: 1px solid {border};"
            f" color: {fg}; background: transparent; font-weight: 600; }}"
            f"QToolButton#PlaqueModeButton_plaque {{ border-top-left-radius: 6px;"
            f" border-bottom-left-radius: 6px; }}"
            f"QToolButton#PlaqueModeButton_figure {{ border-top-right-radius: 6px;"
            f" border-bottom-right-radius: 6px; border-left: none; }}"
            f"QToolButton:checked {{ background: {accent}; color: {ink};"
            f" border-color: {accent}; }}")

    def mode(self) -> str:
        """The mode selected now."""
        return FIGURE_MODE if self._buttons[FIGURE_MODE].isChecked() \
            else PLAQUE_MODE

    def set_mode(self, mode: Any) -> None:
        """Select ``mode`` without announcing it.

        :param mode: ``'plaque'`` or ``'figure'``.
        """
        mode = normalise_mode(mode)
        if mode == self.mode():
            return
        for button in self._buttons.values():
            button.blockSignals(True)
        self._buttons[mode].setChecked(True)
        for button in self._buttons.values():
            button.blockSignals(False)

    def button(self, mode: str) -> QToolButton:
        """The button for ``mode``.

        :param mode: ``'plaque'`` or ``'figure'``.
        :returns: the button.
        """
        return self._buttons[normalise_mode(mode)]


class _ImageView(QLabel):
    """An image scaled to the width it is given, boxes painted on top.

    A click is reported in IMAGE pixels, through :attr:`clicked`.
    """

    clicked = Signal(float, float)
    context_requested = Signal(QPoint)

    def __init__(self, parent: Optional[QWidget] = None):
        """Start empty.

        :param parent: the owning widget.
        """
        super().__init__(parent)
        self.setObjectName("PlaquePreviewImage")
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumHeight(320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap: Optional[QPixmap] = None
        self._array: Optional[np.ndarray] = None

    def array(self) -> Optional[np.ndarray]:
        """The pixels last shown, before any box was painted, or None."""
        return self._array

    def show_message(self, text: str) -> None:
        """Show a line of text in place of a picture.

        :param text: what to say, already translated.
        """
        self.set_image(None)
        self.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.setMargin(16)
        self.setText(text)

    def set_image(self, rgb: Optional[np.ndarray],
                  boxes: Sequence[Tuple[Any, bool]] = (),
                  selected: Optional[int] = None) -> None:
        """Show ``rgb`` with numbered boxes.

        :param rgb: ``H x W x 3`` ``uint8``, or None to clear.
        :param boxes: ``(region, approved)`` pairs, numbered from 1.
        :param selected: the index of the box to highlight.
        """
        if rgb is None:
            self._pixmap = None
            self._array = None
            self.clear()
            return
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        self._array = rgb
        self.setAlignment(Qt.AlignCenter)
        self.setMargin(0)
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, 3 * width,
                       QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        if boxes:
            painter = QPainter(pixmap)
            thickness = max(2, int(round(max(width, height) / 400)))
            font = QFont()
            font.setBold(True)
            font.setPixelSize(max(12, int(max(width, height) / 45)))
            painter.setFont(font)
            for number, (region, approved) in enumerate(boxes, start=1):
                chosen = selected == number - 1
                colour = BOX_SELECTED if chosen else (
                    BOX_OK if approved else BOX_WAITING)
                rect = QRectF(region.x0, region.y0, region.width,
                              region.height)
                if chosen:
                    painter.fillRect(rect, QColor(0, 200, 255, 50))
                painter.setPen(QPen(colour, thickness * (2 if chosen else 1)))
                painter.drawRect(rect)
                painter.drawText(region.x0 + thickness + 2,
                                 region.y0 + font.pixelSize() + thickness,
                                 str(number))
            painter.end()
        self._pixmap = pixmap
        self._rescale()

    def has_image(self) -> bool:
        """Whether an image is shown."""
        return self._pixmap is not None

    def image_point(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """Where a point on the label falls in the image.

        :param x: label column.
        :param y: label row.
        :returns: ``(x, y)`` in image pixels, or None off the image.
        """
        shown = self.pixmap()
        if self._pixmap is None or shown is None or shown.isNull():
            return None
        ratio = shown.devicePixelRatio() or 1.0
        width, height = shown.width() / ratio, shown.height() / ratio
        if not width or not height:
            return None
        left = (self.width() - width) / 2.0
        top = (self.height() - height) / 2.0
        if not (left <= x <= left + width and top <= y <= top + height):
            return None
        return ((x - left) * self._pixmap.width() / width,
                (y - top) * self._pixmap.height() / height)

    def mousePressEvent(self, event):                        # noqa: N802
        """Report a click in image pixels.

        :param event: the mouse event.
        """
        position = event.position()
        point = self.image_point(position.x(), position.y())
        if point is not None and event.button() == Qt.LeftButton:
            self.clicked.emit(point[0], point[1])
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):                       # noqa: N802
        """A right-click asks the panel for the overlay options.

        :param event: the context-menu event.
        """
        self.context_requested.emit(event.globalPos())
        event.accept()

    def _rescale(self) -> None:
        """Fit the pixmap to the label, keeping its shape."""
        if self._pixmap is None:
            return
        self.setPixmap(scaled_for(self._pixmap, self, max(1, self.width()),
                                  max(1, self.height())))

    def resizeEvent(self, event):                            # noqa: N802
        """Refit on resize."""
        super().resizeEvent(event)
        self._rescale()


class PlaquePreviewPanel(QWidget, LivePreviewContract):
    """Plaque Assay's live preview, in Plaque mode or Figure mode.

    The screen drives it through the same four calls every live preview
    answers: :meth:`load_source_async`, :meth:`apply_settings`,
    :meth:`set_propagate_callback` and :meth:`settings_for_propagation`.

    :param parent: the owning widget.
    :param threaded: run passes on worker threads; False runs them inline,
        which is what tests use.
    """

    PREVIEW_SOURCE_HINT = "Choose a source folder with images first."

    #: Where this preview's section folds and sizes are remembered (item
    #: 471): the pictures and the wells/plaques tables fold by their
    #: headings and trade height by their edge; the well picture beside the
    #: image collapses to the right by its handle.
    SECTION_KEY = "plaque_preview"

    mode_changed = Signal(str)
    preview_ready = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None, *,
                 threaded: bool = True):
        """Build the controls; nothing is loaded or resolved here.

        :param parent: the owning widget.
        :param threaded: see the class docstring.
        """
        super().__init__(parent)
        self.setObjectName("PlaquePreviewPanel")
        self._settings: Dict[str, Any] = {}
        self._src = ""
        self._auto_loaded_src = ""
        self._paths: List[Path] = []
        self._propagate_cb: Optional[Callable[[Dict[str, Any]], None]] = None
        self._run_token = 0
        self._load_token = 0
        self._seeded_model = ""
        self._seeded_detector = ""
        self._figure: Optional[Dict[str, Any]] = None
        self._annotations: List[Any] = []
        self._scales: List[Any] = []
        self._caption = ""
        self._missing_entry: Any = None
        self._download = None
        self._install = None
        self._filling_table = False
        self._wells: Dict[int, Dict[str, Any]] = {}
        self._selected: Optional[int] = None
        self._batch: List[int] = []
        self._batch_total = 0
        self._batch_segment: Optional[Callable] = None
        self._overlay_style: OverlayStyle = _SESSION["style"]
        self._fixed_colours: Dict[str, Tuple[int, int, int]] = {
            "outline": OUTLINE_COLOUR, "fill": OUTLINE_COLOUR}
        for key, colour in (("outline", self._overlay_style.outline_colour),
                            ("fill", self._overlay_style.fill_colour)):
            if colour != RANDOM_COLOUR:
                self._fixed_colours[key] = colour
        self._overlay_dialog: Optional[PlaqueOverlayDialog] = None
        self._plaque_result: Optional[Dict[str, Any]] = None
        self._jobs = JobRunner(self, threaded=threaded, app_key="plaque preview")
        self._load_jobs = JobRunner(self, threaded=threaded,
                                    app_key="plaque preview image",
                                    user_visible=False)
        self._jobs.job_failed.connect(self._on_job_failed)
        self._load_jobs.job_failed.connect(self._on_job_failed)
        self._paper_jobs = JobRunner(self, threaded=threaded,
                                     app_key="plaque paper")
        self._paper_jobs.job_failed.connect(self._on_paper_failed)
        self._build()
        self._stow_free_widgets()
        self.set_mode(PLAQUE_MODE)

    def _build(self) -> None:
        """Lay the controls out."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        top = QHBoxLayout()
        self._mode_switch = PlaqueModeSwitch(self)
        self._mode_switch.mode_changed.connect(self._on_switch)
        top.addWidget(self._mode_switch)
        top.addStretch(1)
        self._prev_btn = QToolButton(self)
        self._prev_btn.setText("◀")
        self._prev_btn.setToolTip(tr("Previous image"))
        self._prev_btn.clicked.connect(lambda: self._step(-1))
        self._picker = QComboBox(self)
        self._picker.setObjectName("PlaquePreviewPicker")
        self._picker.setMinimumWidth(220)
        self._picker.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._picker.currentIndexChanged.connect(self._on_picked)
        self._next_btn = QToolButton(self)
        self._next_btn.setText("▶")
        self._next_btn.setToolTip(tr("Next image"))
        self._next_btn.clicked.connect(lambda: self._step(1))
        self._position = QLabel("")
        for widget in (self._prev_btn, self._picker, self._next_btn,
                       self._position):
            top.addWidget(widget)
        outer.addLayout(top)

        self._deps_banner = QFrame(self)
        self._deps_banner.setObjectName("PlaquePapersMissing")
        self._deps_banner.setFrameShape(QFrame.StyledPanel)
        banner = QHBoxLayout(self._deps_banner)
        self._deps_text = QLabel("")
        self._deps_text.setWordWrap(True)
        self._deps_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        banner.addWidget(self._deps_text, 1)
        self._install_btn = QPushButton(tr("Install"))
        self._install_btn.setToolTip(tr(
            "Install the plaque figure reader (YOLO and RapidOCR) into an "
            "environment of its own under ~/.spacr/backends; spaCR's own "
            "packages are not changed."))
        self._install_btn.clicked.connect(lambda _checked=False: self._offer_install())
        banner.addWidget(self._install_btn)
        outer.addWidget(self._deps_banner)

        self._controls = QWidget(self)
        self._controls.setObjectName("PlaquePreviewControls")
        self._controls.hide()
        self._model_row = QWidget(self._controls)
        model_row = QHBoxLayout(self._model_row)
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.setSpacing(4)
        self._model_box = QComboBox(self._model_row)
        self._model_box.setObjectName("PlaquePreviewModel")
        self._model_box.setEditable(True)
        self._model_box.setMinimumWidth(260)
        self._model_box.setToolTip(tr(
            "The Cellpose checkpoint that segments the plaques: a model zoo "
            "key, 'bundled' (the historical packaged model), or a path."))
        self._model_box.currentTextChanged.connect(self._on_model_changed)
        model_row.addWidget(self._model_box, 1)
        browse = QToolButton(self._model_row)
        browse.setText(tr("Browse…"))
        browse.clicked.connect(self._browse_model)
        model_row.addWidget(browse)
        self._model_zoo_btn = self._zoo_button(
            self._model_row, "cellpose", lambda: self._model_box)
        model_row.addWidget(self._model_zoo_btn)
        self._diameter = self._spin(0, 2000, 1, 30, 1)
        self._flow = self._spin(0, 3, 2, 0.4, 0.05)
        self._cellprob = self._spin(-8, 8, 2, 0, 0.25)
        self._detector_row = QWidget(self._controls)
        detector_row = QHBoxLayout(self._detector_row)
        detector_row.setContentsMargins(0, 0, 0, 0)
        detector_row.setSpacing(4)
        self._detector_box = QComboBox(self._detector_row)
        self._detector_box.setObjectName("PlaquePreviewDetector")
        self._detector_box.setEditable(True)
        self._detector_box.setMinimumWidth(260)
        detector_row.addWidget(self._detector_box, 1)
        self._detector_zoo_btn = self._zoo_button(
            self._detector_row, "detector", lambda: self._detector_box)
        detector_row.addWidget(self._detector_zoo_btn)
        self._sizes = QLineEdit(",".join(str(s) for s in DEFAULT_SIZES),
                                self._controls)
        self._sizes.setToolTip(tr("Detector inference sizes, comma "
                                  "separated. Each size is asked and the "
                                  "boxes are merged."))
        self._confidence = self._spin(0, 1, 2, 0.25, 0.05)
        self._read_text = QCheckBox(tr("Read the text around each well"),
                                    self._controls)
        self._read_text.setChecked(True)
        self._confirm = QCheckBox(tr("Confirm annotations"), self._controls)
        self._confirm.setToolTip(tr("When on, the run measures only the "
                                    "images ticked OK and saved here."))
        self._confirm.toggled.connect(self._on_confirm_toggled)
        self._build_text_controls()
        self._settings_dialog: Optional[PlaqueSettingsDialog] = None
        outer.addWidget(self._controls)
        self._outer = outer

        note_row = QHBoxLayout()
        self._model_note = QLabel("")
        self._model_note.setObjectName("PlaquePreviewModelNote")
        self._model_note.setWordWrap(True)
        note_row.addWidget(self._model_note, 1)
        self._download_btn = QPushButton(tr("Download"))
        self._download_btn.clicked.connect(self._start_download)
        self._download_btn.hide()
        note_row.addWidget(self._download_btn)
        outer.addLayout(note_row)

        buttons = QHBoxLayout()
        self._run_btn = QPushButton(tr(PREVIEW_RUN_TEXT))
        self._run_btn.setObjectName("PlaquePreviewRun")
        self._run_btn.clicked.connect(self.run_preview)
        self._cancel_btn = QPushButton(tr(PREVIEW_CANCEL_TEXT))
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self.cancel_preview)
        self._use_btn = QPushButton(tr("Use these settings"))
        self._use_btn.setToolTip(tr("Write the values tuned here into the "
                                    "settings the run reads."))
        self._use_btn.clicked.connect(self.propagate)
        self._settings_btn = QPushButton(tr("Settings…"))
        self._settings_btn.setObjectName("PlaquePreviewSettings")
        self._settings_btn.setToolTip(tr(
            "The preview's settings, in two tabs: Figure (finding the plaque "
            "wells and reading the figure) and Plaque detection (the plaque "
            "model and its thresholds)."))
        self._settings_btn.clicked.connect(lambda: self.open_settings())
        self._well_btn = QPushButton(tr("Plaque preview"))
        self._well_btn.setObjectName("PlaqueWellPreview")
        self._well_btn.setToolTip(tr(
            "Find the plaques in the highlighted well with the plaque "
            "settings."))
        self._well_btn.clicked.connect(lambda: self.preview_selected_well())
        self._all_btn = QPushButton(tr("Find plaques in all wells"))
        self._all_btn.setObjectName("PlaqueAllWells")
        self._all_btn.setToolTip(tr(
            "Find the plaques in every well found on this figure, one after "
            "another. Cancel stops after the well in progress."))
        self._all_btn.clicked.connect(lambda: self.find_plaques_in_all_wells())
        self._paper_btn = QPushButton(tr("From a paper…"))
        self._paper_btn.setObjectName("PlaqueFromPaper")
        self._paper_btn.setToolTip(tr(
            "Fetch a paper's figures and legends by DOI, PMID, PMC id or PDF "
            "into a new folder, and preview them."))
        self._paper_btn.clicked.connect(self._ask_for_paper)
        for widget in (self._paper_btn, self._settings_btn,
                       self._run_btn, self._well_btn, self._all_btn,
                       self._cancel_btn, self._use_btn):
            buttons.addWidget(widget)
        buttons.addStretch(1)
        from .preview_scale import install_preview_scale
        self._scale_control = install_preview_scale(self, "plaque", buttons)
        outer.addLayout(buttons)

        self._paper_note = QLabel("")
        self._paper_note.setObjectName("PlaquePaperNote")
        self._paper_note.setWordWrap(True)
        self._paper_note.hide()
        outer.addWidget(self._paper_note)

        self._status = QLabel(tr(self.PREVIEW_SOURCE_HINT))
        self._status.setObjectName("PlaquePreviewStatus")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        from .collapsible_splitter import EDGE, CollapsibleSplitter
        key = self.SECTION_KEY
        pictures = CollapsibleSplitter(Qt.Horizontal, self,
                                       persist_key=f"{key}::pictures")
        self._pictures_split = pictures
        self._view = _ImageView(self)
        self._view.setCursor(Qt.PointingHandCursor)
        self._view.clicked.connect(self._on_figure_clicked)
        self._objects_view = _ImageView(self)
        self._objects_view.setObjectName("PlaqueObjectsImage")
        self._prob_view = _ImageView(self)
        self._prob_view.setObjectName("PlaqueCellprobImage")
        self._flow_view = _ImageView(self)
        self._flow_view.setObjectName("PlaqueFlowsImage")
        self._image_tabs = QTabWidget(self)
        self._image_tabs.setObjectName("PlaqueImageTabs")
        tips = (tr("The image with the plaques drawn over it. Right-click "
                   "for outlines or a filled overlay, colour, thickness and "
                   "opacity."),
                tr("The plaque mask alone, one colour per plaque."),
                tr("Cellpose's cell probability, 0 to 1."),
                tr("Cellpose's flow field: direction as hue, strength as "
                   "brightness."))
        for index, (view, title) in enumerate(zip(
                (self._view, self._objects_view, self._prob_view,
                 self._flow_view), IMAGE_TABS)):
            self._image_tabs.addTab(view, tr(title))
            self._image_tabs.setTabToolTip(index, tips[index])
            view.context_requested.connect(self._on_view_context)
        self._sections = {}
        pictures.add_pane(self._image_tabs, "Image", stretch=3)
        self._well_side = QWidget(self)
        side = QVBoxLayout(self._well_side)
        side.setContentsMargins(0, 0, 0, 0)
        self._well_title = QLabel(tr(PICK_A_WELL))
        self._well_title.setObjectName("PlaqueWellTitle")
        self._well_title.setWordWrap(True)
        side.addWidget(self._well_title)
        self._well_view = _ImageView(self._well_side)
        self._well_view.setObjectName("PlaqueWellImage")
        self._well_view.context_requested.connect(self._on_view_context)
        side.addWidget(self._well_view, 1)
        pictures.add_pane(self._well_side, "Well", mode=EDGE, stretch=2,
                          fold_key=f"{key}/Well")
        picture_host = QWidget(self)
        picture_col = QVBoxLayout(picture_host)
        picture_col.setContentsMargins(0, 0, 0, 0)
        picture_col.addWidget(pictures, 1)

        self._legend_box = QFrame(self)
        self._legend_box.setObjectName("PlaqueLegendPrompt")
        self._legend_box.setFrameShape(QFrame.StyledPanel)
        legend = QVBoxLayout(self._legend_box)
        self._legend_text = QLabel("")
        self._legend_text.setWordWrap(True)
        legend.addWidget(self._legend_text)
        self._legend_edit = QPlainTextEdit(self._legend_box)
        self._legend_edit.setPlaceholderText(tr(
            "Paste the figure legend, or annotate by hand"))
        self._legend_edit.setMaximumHeight(90)
        legend.addWidget(self._legend_edit)
        legend_buttons = QHBoxLayout()
        self._legend_use = QPushButton(tr("Use this legend"))
        self._legend_use.clicked.connect(self._use_pasted_legend)
        self._legend_skip = QPushButton(tr("Annotate by hand"))
        self._legend_skip.clicked.connect(self._annotate_by_hand)
        legend_buttons.addWidget(self._legend_use)
        legend_buttons.addWidget(self._legend_skip)
        legend_buttons.addStretch(1)
        legend.addLayout(legend_buttons)
        picture_col.addWidget(self._legend_box)

        self._confirm_note = QLabel(tr(
            "Confirm annotations is on: the run measures ONLY the images "
            "ticked OK below and saved with Save annotations. Every other "
            "image is left out."))
        self._confirm_note.setObjectName("PlaqueConfirmNotice")
        self._confirm_note.setWordWrap(True)
        self._confirm_note.setStyleSheet("font-weight: 600;")
        picture_col.addWidget(self._confirm_note)

        self._table = QTableWidget(0, len(TABLE_COLUMNS), self)
        self._table.setObjectName("PlaqueAnnotationTable")
        self._table.setHorizontalHeaderLabels([tr(c) for c in TABLE_COLUMNS])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(CONDITION_COLUMN, QHeaderView.Stretch)
        self._table.itemChanged.connect(self._on_table_edit)
        self._table.itemSelectionChanged.connect(self._on_table_selection)
        self._table.setMinimumHeight(180)
        install_sorting(self._table)
        self._plaque_table = QTableWidget(0, len(PLAQUE_COLUMNS), self)
        self._plaque_table.setObjectName("PlaquePerPlaqueTable")
        self._plaque_table.setHorizontalHeaderLabels(
            [tr(c) for c in PLAQUE_COLUMNS])
        self._plaque_table.verticalHeader().setVisible(False)
        self._plaque_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._plaque_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._plaque_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self._plaque_table.itemSelectionChanged.connect(
            self._on_plaque_selection)
        install_sorting(self._plaque_table)
        self._tabs = QTabWidget(self)
        self._tabs.setObjectName("PlaqueFigureTabs")
        self._tabs.addTab(self._table, tr("Wells"))
        self._tabs.addTab(self._plaque_table, tr("Plaques"))
        self._tabs.setMinimumHeight(200)
        split = CollapsibleSplitter(Qt.Vertical, self,
                                    persist_key=f"{key}::sections")
        self._section_split = split
        self._sections["Pictures"] = split.add_section(
            picture_host, "Pictures", stretch=3,
            persist_key=f"{key}/Pictures")
        self._sections["Wells and plaques"] = split.add_section(
            self._tabs, "Wells and plaques", stretch=2,
            persist_key=f"{key}/Wells and plaques")
        outer.addWidget(split, 5)

        save_row = QHBoxLayout()
        self._save_btn = QPushButton(tr("Save annotations"))
        self._save_btn.setObjectName("PlaqueSaveAnnotations")
        self._save_btn.setToolTip(tr(
            "Write the conditions and OK ticks to figure_annotations.csv in "
            "the source folder. The Figure-mode run reads it."))
        self._save_btn.clicked.connect(self.save_annotations)
        save_row.addWidget(self._save_btn)
        save_row.addStretch(1)
        self._save_row = QWidget(self)
        self._save_row.setLayout(save_row)
        outer.addWidget(self._save_row)

    def _stow_free_widgets(self) -> int:
        """Put every child that is in no layout into the holder that never shows.

        Without this, starting the live preview shows a field covering the
        Mode label beside the Plaque and Figure buttons until Settings is
        opened and closed once. The settings
        controls are homeless on purpose -- the panel owns them so their
        values outlive the Settings window, which lays them out only while it
        is open -- and a QWidget parented to the panel but in no layout
        paints at the panel's top left, over the Mode label, until something
        hides it. The Settings window hid them on its way out, which is why
        the field went away once it had been opened.

        The same class of defect, and the same fix, as
        :meth:`spacr.qt.widgets.live_preview.LivePreviewPanel._stow_free_widgets`:
        blunt on purpose, it moves whatever it finds, so a control added
        later is covered without anyone remembering this.

        :returns: how many widgets were moved; zero on a second call.
        """
        holder = self._controls
        laid_out = set()
        stack = [self.layout()]
        while stack:
            layout = stack.pop()
            if layout is None:
                continue
            for index in range(layout.count()):
                item = layout.itemAt(index)
                child = item.widget()
                if child is not None:
                    laid_out.add(id(child))
                stack.append(item.layout())
        moved = 0
        for child in self.findChildren(QWidget,
                                       options=Qt.FindDirectChildrenOnly):
            if child is holder or id(child) in laid_out or child.isWindow():
                continue
            child.setParent(holder)
            moved += 1
        return moved

    def _build_text_controls(self) -> None:
        """One control per Text detection setting, seeded with the defaults.

        Every change re-proposes the conditions from the text already read
        (:meth:`_on_text_changed`), so a label can be tuned until it is right
        without running the detector or OCR again.
        """
        from ...plaque_papers import DEFAULT_TEXT_OPTIONS as d

        self._text: Dict[str, QWidget] = {}
        self._seeding_text = False
        for key, value in (("text_reach_above", d.reach_above),
                           ("text_reach_left", d.reach_left),
                           ("text_reach_below", d.reach_below),
                           ("text_panel_reach", d.panel_reach)):
            spin = self._spin(0, 10, 2, float(value), 0.1)
            spin.valueChanged.connect(lambda _v, k=key: self._on_text_changed(k))
            self._text[key] = spin
        spin = self._spin(0, 1, 2, float(d.min_confidence), 0.05)
        spin.valueChanged.connect(
            lambda _v: self._on_text_changed("text_min_confidence"))
        self._text["text_min_confidence"] = spin
        for key, value, label in (
                ("text_use_above", d.use_above, "text_use_above"),
                ("text_use_left", d.use_left, "text_use_left"),
                ("text_use_below", d.use_below, "text_use_below"),
                ("text_reread", d.reread, "text_reread")):
            text = dict(TEXT_LABELS)[label]
            box = QCheckBox(tr(text), self._controls)
            box.setChecked(bool(value))
            box.toggled.connect(lambda _on, k=key: self._on_text_changed(k))
            self._text[key] = box
        scale = QSpinBox(self._controls)
        scale.setRange(1, 8)
        scale.setValue(int(d.reread_scale))
        scale.valueChanged.connect(
            lambda _v: self._on_text_changed("text_reread_scale"))
        self._text["text_reread_scale"] = scale
        ignore = QLineEdit(",".join(d.ignore), self._controls)
        ignore.setPlaceholderText(tr("comma-separated regular expressions, "
                                     "e.g. ^\\d+$, μm"))
        ignore.textChanged.connect(lambda _t: self._on_text_changed("text_ignore"))
        self._text["text_ignore"] = ignore
        order = QComboBox(self._controls)
        order.setEditable(True)
        order.addItems(list(TEXT_ORDERS))
        order.setCurrentText(",".join(d.order))
        order.currentTextChanged.connect(
            lambda _t: self._on_text_changed("text_order"))
        self._text["text_order"] = order
        separator = QLineEdit(d.separator, self._controls)
        separator.textChanged.connect(
            lambda _t: self._on_text_changed("text_separator"))
        self._text["text_separator"] = separator
        for key, widget in self._text.items():
            widget.setObjectName(f"PlaqueText_{key}")
            tip = _setting_tooltip(key)
            if tip:
                widget.setToolTip(tip)
            widget.setParent(self._controls)
            widget.hide()

    def text_values(self) -> Dict[str, Any]:
        """The Text detection controls as ``text_*`` settings.

        :returns: setting name -> value, in the form's own types.
        """
        out: Dict[str, Any] = {}
        for key, widget in self._text.items():
            if isinstance(widget, QCheckBox):
                out[key] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                out[key] = int(widget.value())
            elif isinstance(widget, QDoubleSpinBox):
                out[key] = float(widget.value())
            elif isinstance(widget, QComboBox):
                out[key] = widget.currentText().strip()
            else:
                out[key] = widget.text()
        return out

    def text_options(self):
        """The :class:`spacr.plaque_papers.TextOptions` the controls describe."""
        from ...plaque_papers import text_options_from_settings

        return text_options_from_settings(self.text_values())

    def _seed_text(self, settings: Dict[str, Any]) -> None:
        """Put the form's ``text_*`` values into the controls, quietly.

        :param settings: the form's values.
        """
        self._seeding_text = True
        try:
            for key, widget in self._text.items():
                if key not in settings or settings[key] is None:
                    continue
                value = settings[key]
                try:
                    if isinstance(widget, QCheckBox):
                        widget.setChecked(bool(value))
                    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                        widget.setValue(type(widget.value())(float(value)))
                    elif isinstance(widget, QComboBox):
                        if isinstance(value, (list, tuple)):
                            value = ",".join(str(v) for v in value)
                        widget.setCurrentText(str(value))
                    else:
                        if isinstance(value, (list, tuple)):
                            value = ",".join(str(v) for v in value)
                        widget.setText(str(value))
                except (TypeError, ValueError):
                    continue
        finally:
            self._seeding_text = False

    def _on_text_changed(self, key: str) -> None:
        """Re-propose the conditions from the words already read.

        A change to the second reading cannot be applied to words already
        read, so it says a new Run preview is needed instead.

        :param key: the setting that changed.
        """
        if self._seeding_text or self._figure is None:
            return
        if key in TEXT_READ_KEYS:
            self.set_preview_status(tr(
                "The second reading changed: press Run preview to read the "
                "figure again."))
            return
        self._reannotate()
        if self._selected is not None:
            self._show_well(self._selected)
        self.set_preview_status(tr("Conditions proposed again with the new "
                                   "text settings."))

    def _spin(self, low: float, high: float, decimals: int, value: float,
              step: float) -> QDoubleSpinBox:
        """A number box.

        :returns: the box.
        """
        spin = QDoubleSpinBox(self._controls)
        spin.setRange(low, high)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def mode(self) -> str:
        """The mode the panel is in."""
        return self._mode_switch.mode()

    def set_mode(self, mode: Any) -> None:
        """Show the controls ``mode`` needs, without announcing the change.

        :param mode: ``'plaque'`` or ``'figure'``.
        """
        mode = normalise_mode(mode)
        self._mode_switch.set_mode(mode)
        figure = mode == FIGURE_MODE
        dialog = self._settings_dialog
        if dialog is not None:
            dialog.show_mode(mode)
        self._tabs.setVisible(figure)
        self._image_tabs.tabBar().setVisible(not figure)
        if figure:
            self._image_tabs.setCurrentIndex(0)
        self._plaque_result = None
        self._well_side.setVisible(figure)
        self._well_btn.setVisible(figure)
        self._paper_btn.setVisible(figure)
        self._paper_note.setVisible(figure and bool(self._paper_note.text()))
        self._all_btn.setVisible(figure)
        self._save_row.setVisible(figure)
        self._confirm_note.setVisible(figure and self._confirm.isChecked())
        if not figure:
            self._legend_box.hide()
        missing = missing_papers_packages() if figure else []
        self._deps_banner.setVisible(bool(missing))
        if missing:
            self._deps_text.setText(papers_install_message(missing))
            self._install_btn.setVisible(True)
        if self._figure is not None and not figure:
            self._clear_figure()
        self._view.set_image(None)
        self._show_selected_image()

    def open_settings(self, tab: Optional[str] = None) -> "PlaqueSettingsDialog":
        """Open (or raise) the one settings window, in the tab for this mode.

        One settings button serves the whole live preview, with a tab for
        figure detection and one for plaque detection. A ``QDialog``, so :mod:`spacr.qt.widgets.glass` gives it
        the same rounded, translucent card every other settings window has.

        :param tab: ``'figure'`` or ``'plaque'`` to open on; the mode's own
            first tab when None.
        :returns: the dialog.
        """
        dialog = self._settings_dialog
        if dialog is None or not dialog.isVisible():
            dialog = PlaqueSettingsDialog(self)
            self._settings_dialog = dialog
            dialog.finished.connect(self._settings_closed)
        dialog.show_mode(self.mode(), tab)
        dialog.show()
        dialog.raise_()
        return dialog

    def _settings_closed(self, *_args: Any) -> None:
        """Take the controls back when the settings window closes."""
        dialog = self._settings_dialog
        self._settings_dialog = None
        if dialog is not None:
            dialog.give_back()

    def _on_switch(self, mode: str) -> None:
        """The panel's own switch was clicked."""
        self.set_mode(mode)
        self.mode_changed.emit(mode)

    def load_source_async(self, source: Any, **_ignored: Any) -> bool:
        """List the images in ``source`` off the GUI thread and show the first.

        :param source: the source folder, or one image.
        :returns: True when a listing was started.
        """
        text = str(source or "").strip()
        if not text:
            return False
        self.cancel_preview()
        self._src = text
        self._load_token += 1
        token = self._load_token
        self.set_preview_status(tr("Loading preview from {path}…", path=text))
        self._load_jobs.submit(lambda: images_in(text),
                               lambda paths, t=token: self._on_listing(t, paths))
        return True

    def _on_listing(self, token: int, paths: List[Path]) -> None:
        """Fill the picker from a finished listing."""
        if token != self._load_token:
            return
        paths = list(paths or [])
        if paths and paths == self._paths and self.current_path() in paths:
            self.set_preview_status(tr("{n} images in {path}, unchanged.",
                                       n=len(paths), path=self._src))
            return
        self._paths = paths
        self._picker.blockSignals(True)
        self._picker.clear()
        for path in self._paths:
            self._picker.addItem(path.name, str(path))
        self._picker.blockSignals(False)
        if not self._paths:
            self._clear_figure()
            self._plaque_result = None
            self._show_plaque_tabs()
            self._legend_box.hide()
            self.set_preview_status(tr("No images found in {path}.",
                                       path=self._src))
            self._view.set_image(None)
            self._position.setText("")
            return
        self._picker.setCurrentIndex(0)
        self._on_picked(0)

    def _step(self, delta: int) -> None:
        """Move the picker ``delta`` images along, wrapping."""
        count = self._picker.count()
        if count:
            self._picker.setCurrentIndex(
                (self._picker.currentIndex() + delta) % count)

    def current_path(self) -> Optional[Path]:
        """The image selected in the picker."""
        data = self._picker.currentData()
        return Path(data) if data else None

    def _on_picked(self, index: int) -> None:
        """A new image was picked: show it unsegmented."""
        count = self._picker.count()
        self._position.setText(tr("{n} of {total}", n=index + 1, total=count)
                               if count else "")
        self._clear_figure()
        self._legend_box.hide()
        self._show_selected_image()

    def _show_selected_image(self) -> None:
        """Decode the selected image off the GUI thread and show it."""
        self.cancel_preview()
        path = self.current_path()
        if path is None:
            return
        self._load_token += 1
        token = self._load_token
        self._plaque_result = None
        self._show_plaque_tabs()
        self.set_preview_status(tr("{name}: press Run preview.",
                                   name=path.name))
        self._load_jobs.submit(
            lambda: load_display_image(path),
            lambda rgb, t=token: t == self._load_token
            and self._view.set_image(rgb))

    def apply_settings(self, settings: Dict[str, Any]) -> None:
        """Seed every control from the module's settings.

        :param settings: the form's values.
        """
        self._settings = dict(settings or {})
        s = self._settings
        if MODE_KEY in s:
            self.set_mode(s.get(MODE_KEY))
        if s.get("src") and not self._src:
            self._src = str(s["src"])
        self._fill_model_box(str(s.get("plaque_model") or "bundled"))
        self._seeded_model = self._model_box.currentText()
        self._fill_detector_box(str(s.get("figure_detector") or DEFAULT_DETECTOR))
        self._seeded_detector = self._detector_box.currentText()
        for key, spin in (("diameter", self._diameter),
                          ("flow_threshold", self._flow),
                          ("CP_prob", self._cellprob),
                          ("figure_confidence", self._confidence)):
            try:
                if s.get(key) is not None:
                    spin.setValue(float(s[key]))
            except (TypeError, ValueError):
                continue
        if s.get("figure_imgsz") is not None:
            self._sizes.setText(",".join(
                str(v) for v in parse_sizes(s.get("figure_imgsz"))))
        if "figure_read_text" in s:
            self._read_text.setChecked(bool(s.get("figure_read_text")))
        if "confirm_annotations" in s:
            self._confirm.setChecked(bool(s.get("confirm_annotations")))
        self._seed_text(s)
        self._describe_model()

    def _fill_model_box(self, wanted: str) -> None:
        """Offer the plaque models and select ``wanted``."""
        self._model_box.blockSignals(True)
        self._model_box.clear()
        for key in plaque_model_choices():
            self._model_box.addItem(key)
        if self._model_box.findText(wanted) < 0:
            self._model_box.addItem(wanted)
        self._model_box.setCurrentText(wanted)
        self._model_box.blockSignals(False)

    def _fill_detector_box(self, wanted: str) -> None:
        """Offer the detectors and select ``wanted``."""
        self._detector_box.clear()
        for key in detector_choices():
            self._detector_box.addItem(key)
        if self._detector_box.findText(wanted) < 0:
            self._detector_box.addItem(wanted)
        self._detector_box.setCurrentText(wanted)

    def _zoo_button(self, parent: QWidget, kind: str,
                    box: Callable[[], QComboBox]) -> QToolButton:
        """A "Model zoo…" button that fills ``box`` from the zoo.

        The same second way in the settings panel gives every Cellpose
        field: browse what spaCR knows about, download what is not on this
        machine, and have it chosen here. The zoo is shown filtered to
        ``kind`` -- the plaque model is a Cellpose checkpoint, the detector a
        YOLO detector -- so nothing offered fails when the preview runs.

        :param parent: the row the button sits in.
        :param kind: the :data:`spacr.model_zoo.KINDS` entry to list.
        :param box: returns the combo box the choice is written into.
        :returns: the button.
        """
        button = QToolButton(parent)
        button.setText(tr("Model zoo…"))
        button.setToolTip(tr(
            "Browse the models spaCR knows about, see what each was trained "
            "on, download one and fill in this field. You can still type a "
            "path yourself. "
            "API: spacr.qt.widgets.model_zoo_picker.choose_model."))
        button.clicked.connect(
            lambda _checked=False: self._choose_from_zoo(kind, box()))
        return button

    def _choose_from_zoo(self, kind: str, box: QComboBox) -> Optional[str]:
        """Open the zoo for ``kind`` and put the chosen model in ``box``.

        The zoo KEY is written when the chosen file is a zoo entry's, so the
        setting stays portable between machines; a file the zoo does not
        name is written as its path.

        :param kind: the model kind to list.
        :param box: the combo box to fill.
        :returns: what was written, or None when cancelled.
        """
        from PySide6.QtWidgets import QDialog

        from .model_zoo_picker import ModelZooPicker

        dialog = ModelZooPicker(kinds=(kind,), parent=self)
        if dialog.exec() != QDialog.Accepted or not dialog.chosen_path():
            return None
        entry = dialog.selected_entry()
        key = str(getattr(entry, "key", "") or "")
        value = key if key else str(dialog.chosen_path())
        if box.findText(value) < 0:
            box.addItem(value)
        box.setCurrentText(value)
        return value

    def _browse_model(self) -> None:
        """Pick a checkpoint file for the plaque model."""
        chosen, _ = QFileDialog.getOpenFileName(
            self, tr("Choose a plaque checkpoint"), "")
        if chosen:
            if self._model_box.findText(chosen) < 0:
                self._model_box.addItem(chosen)
            self._model_box.setCurrentText(chosen)

    def _on_model_changed(self, _text: str) -> None:
        """Say what the chosen model is."""
        self._describe_model()

    def _describe_model(self) -> None:
        """The note under the model box, which 'bundled' always gets."""
        chosen = self._model_box.currentText()
        self._model_note.setText(tr(BUNDLED_NOTE) if chosen == "bundled"
                                 else "")
        self._download_btn.hide()
        self._missing_entry = None

    def current_settings(self) -> Dict[str, Any]:
        """The module's settings with this panel's values over them."""
        out = dict(self._settings)
        out.update({
            MODE_KEY: self.mode(),
            "src": self._src or out.get("src"),
            "plaque_model": self._model_box.currentText() or "bundled",
            "diameter": float(self._diameter.value()),
            "flow_threshold": float(self._flow.value()),
            "CP_prob": float(self._cellprob.value()),
            "figure_detector": self._detector_box.currentText()
            or DEFAULT_DETECTOR,
            "figure_imgsz": ",".join(str(v) for v in
                                     parse_sizes(self._sizes.text())),
            "figure_confidence": float(self._confidence.value()),
            "figure_read_text": self._read_text.isChecked(),
            "confirm_annotations": self._confirm.isChecked(),
        })
        out.update(self.text_values())
        return out

    def settings_for_propagation(self) -> Dict[str, Any]:
        """The values to write back into the form, in its own names.

        A model the panel only seeded is not written back: the form already
        holds it, and writing a resolved path over a zoo key would change
        what a recorded run says it asked for.

        :returns: setting name -> value.
        """
        s = self.current_settings()
        out = {MODE_KEY: s[MODE_KEY], "diameter": s["diameter"],
               "flow_threshold": s["flow_threshold"], "CP_prob": s["CP_prob"]}
        if s["plaque_model"] != self._seeded_model:
            out["plaque_model"] = s["plaque_model"]
        if self.mode() == FIGURE_MODE:
            for key in ("figure_imgsz", "figure_confidence",
                        "figure_read_text", "confirm_annotations") + TEXT_KEYS:
                out[key] = s[key]
            if s["figure_detector"] != self._seeded_detector:
                out["figure_detector"] = s["figure_detector"]
        return out

    def set_propagate_callback(self, callback) -> None:
        """Where :meth:`propagate` sends the tuned values.

        :param callback: ``fn(dict)``.
        """
        self._propagate_cb = callback

    def propagate(self) -> None:
        """Write the tuned values into the module's settings."""
        if self._propagate_cb is None:
            return
        values = self.settings_for_propagation()
        self._propagate_cb(values)
        self._seeded_model = self._model_box.currentText()
        self._seeded_detector = self._detector_box.currentText()
        self.set_preview_status(tr("Wrote {n} settings into the form.",
                                   n=len(values)))

    def _preview_blocked_reason(self) -> str:
        """Why a pass cannot start, or ``''``."""
        if self.current_path() is None:
            return tr(self.PREVIEW_SOURCE_HINT)
        if self.mode() == FIGURE_MODE:
            missing = missing_papers_packages()
            if missing:
                return papers_install_message(missing)
        return ""

    def preview_running(self) -> bool:
        """Whether a pass is in flight."""
        return self._jobs.is_busy()

    def _extra_work_in_flight(self) -> bool:
        """Whether a pass is in flight, for :meth:`cancel_preview`."""
        return self._jobs.is_busy()

    def _cancel_extra_work(self) -> None:
        """Drop the pass in flight, and any wells still queued."""
        self._batch = []
        self._jobs.cancel()

    def run_preview(self, *_args: Any, detect: Optional[Callable] = None,
                    read_text: Optional[Callable] = None,
                    segment: Optional[Callable] = None) -> bool:
        """Run the selected mode on the selected image, off the GUI thread.

        :param detect: replaces the detector (tests).
        :param read_text: replaces the text reader (tests).
        :param segment: replaces the plaque model (tests); Plaque mode only,
            Figure mode segments per well.
        :returns: True when a pass was started.
        """
        if not self.begin_preview():
            return False
        self._run_token += 1
        token = self._run_token
        path = self.current_path()
        settings = self.current_settings()
        self.set_preview_status(tr(PREVIEW_RUNNING_MESSAGE))
        if self.mode() == FIGURE_MODE:
            work = (lambda: detect_figure(path, settings, detect=detect,
                                          read_text=read_text))
        else:
            work = lambda: plaque_pass(path, settings, segment=segment)
        self._jobs.submit(work, lambda result, t=token: self._on_result(t, result))
        return True

    def _on_job_failed(self, message: str) -> None:
        """A worker raised: say so and go idle."""
        self.set_preview_busy(False)
        self.set_preview_status(preview_failure_message(message))

    def _on_result(self, token: int, result: Dict[str, Any]) -> None:
        """Show a finished pass."""
        if self.preview_stale(token):
            return
        self.set_preview_busy(False)
        if not isinstance(result, dict):
            return
        if result.get("error"):
            self.set_preview_status(result["error"])
            self._offer_download(result.get("entry"))
            return
        if result.get("note"):
            self._model_note.setText(result["note"])
        if "regions" in result:
            self._show_figure(result)
        else:
            self._plaque_result = result
            self._show_plaque_tabs()
            self.set_preview_status(tr(
                "{name}: {count} plaques, mean area {area:.0f} px.",
                name=Path(result["path"]).name, count=result["count"],
                area=result["mean_area"]))
        self.preview_ready.emit(result)

    def overlay_style(self) -> OverlayStyle:
        """How the plaques are drawn now."""
        return self._overlay_style

    def set_overlay_style(self, style: OverlayStyle) -> None:
        """Draw the plaques another way, from what is already segmented.

        The style is kept for the rest of the session: a panel built later
        starts from it.

        :param style: the new :class:`OverlayStyle`.
        """
        style = style.normalised()
        for key, colour in (("outline", style.outline_colour),
                            ("fill", style.fill_colour)):
            if colour != RANDOM_COLOUR:
                self._fixed_colours[key] = colour
        self._overlay_style = style
        _SESSION["style"] = style
        dialog = self._overlay_dialog
        if dialog is not None:
            dialog.set_overlay_style(style)
        self._redraw_plaques()

    def _redraw_plaques(self) -> None:
        """Redraw every picture that shows plaques, in the current style."""
        if self._plaque_result is not None:
            self._show_plaque_tabs()
        if self._figure is not None:
            self._repaint_overlay()
            self._redraw_boxes()
            if self._selected is not None:
                self._show_well(self._selected)

    def _show_plaque_tabs(self) -> None:
        """Fill the Overlay, Objects, Cell probability and Flows tabs.

        Before a run, and for a run that gave no flows, a tab says what is
        missing instead of staying blank. The Overlay tab keeps the plain
        image until there is something to draw on it.
        """
        result = self._plaque_result
        if result is None:
            waiting = tr("Press Run preview to see this.")
            for view in (self._objects_view, self._prob_view,
                         self._flow_view):
                view.show_message(waiting)
            return
        image = result.get("image")
        labels = result.get("labels")
        if image is not None and labels is not None:
            self._view.set_image(render_overlay(
                np.array(image, dtype=np.uint8, copy=True), labels,
                self._overlay_style))
        else:
            self._view.set_image(result.get("overlay"))
        objects = render_objects(labels)
        if objects is None:
            self._objects_view.show_message(tr("This run gave no mask."))
        elif not np.any(np.asarray(labels) > 0):
            self._objects_view.show_message(tr(
                "No plaques were found in this image."))
        else:
            self._objects_view.set_image(objects)
        prob = render_cellprob(result.get("cellprob"))
        if prob is None:
            self._prob_view.show_message(tr(
                "This run gave no cell probability map."))
        else:
            self._prob_view.set_image(prob)
        flows = result.get("flow_rgb")
        if flows is None:
            self._flow_view.show_message(tr("This run gave no flows."))
        else:
            self._flow_view.set_image(flows)

    def overlay_menu(self) -> QMenu:
        """The right-click menu of the preview images.

        :returns: a menu with Outlines / Filled overlay, a random-colour
            toggle for whichever is shown, and the full overlay settings.
        """
        style = self._overlay_style
        menu = QMenu(self)
        menu.setObjectName("PlaqueOverlayMenu")
        group = QActionGroup(menu)
        group.setExclusive(True)
        for display, label in ((OVERLAY_OUTLINES, tr("Outlines")),
                               (OVERLAY_FILL, tr("Filled overlay"))):
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(style.display == display)
            action.setData(display)
            group.addAction(action)
            action.triggered.connect(
                lambda _checked=False, d=display: self._set_display(d))
        menu.addSeparator()
        fill = style.display == OVERLAY_FILL
        current = style.fill_colour if fill else style.outline_colour
        randomise = menu.addAction(tr("Random colour per object"))
        randomise.setObjectName("PlaqueOverlayRandom")
        randomise.setCheckable(True)
        randomise.setChecked(current == RANDOM_COLOUR)
        randomise.toggled.connect(self._set_random)
        menu.addSeparator()
        settings = menu.addAction(tr("Overlay settings…"))
        settings.setObjectName("PlaqueOverlaySettings")
        settings.triggered.connect(lambda _checked=False:
                                   self.open_overlay_settings())
        return menu

    def _set_display(self, display: str) -> None:
        """Switch between outlines and a filled overlay."""
        from dataclasses import replace

        self.set_overlay_style(replace(self._overlay_style, display=display))

    def _set_random(self, on: bool) -> None:
        """Random colours per object for the shown display, or the last fixed one."""
        from dataclasses import replace

        style = self._overlay_style
        fill = style.display == OVERLAY_FILL
        if on:
            colour: Any = RANDOM_COLOUR
        else:
            colour = self._fixed_colours["fill" if fill else "outline"]
        key = "fill_colour" if fill else "outline_colour"
        self.set_overlay_style(replace(style, **{key: colour}))

    @staticmethod
    def _exec_menu(menu: QMenu, position: QPoint) -> Any:
        """Show a menu at ``position`` and wait for it; tests replace this."""
        return menu.exec(position)

    def _on_view_context(self, position: QPoint) -> None:
        """A preview image was right-clicked: offer the overlay options.

        :param position: where, in global coordinates.
        """
        self._exec_menu(self.overlay_menu(), position)

    def open_overlay_settings(self) -> PlaqueOverlayDialog:
        """Open (or raise) the overlay settings, applied live.

        Non-modal, because what it answers is how the picture beside it
        looks.

        :returns: the dialog, shown.
        """
        dialog = self._overlay_dialog
        if dialog is None:
            dialog = PlaqueOverlayDialog(self._overlay_style, self)
            dialog.style_changed.connect(self.set_overlay_style)
            dialog.finished.connect(self._overlay_closed)
            self._overlay_dialog = dialog
        dialog.show()
        dialog.raise_()
        return dialog

    def _overlay_closed(self, *_args: Any) -> None:
        """Forget the overlay window once it is closed."""
        dialog = self._overlay_dialog
        self._overlay_dialog = None
        if dialog is not None:
            dialog.deleteLater()

    def _offer_download(self, entry: Any) -> None:
        """Show the Download button for a zoo model that is not here."""
        self._missing_entry = entry
        if entry is None:
            self._download_btn.hide()
            return
        self._download_btn.setText(tr("Download {name}", name=getattr(
            entry, "key", "") or getattr(entry, "name", "")))
        self._download_btn.show()

    def _start_download(self) -> None:
        """Download the missing zoo model into ``~/.spacr/models``."""
        entry = self._missing_entry
        if entry is None:
            return
        from ..model_install import CheckpointDownload

        folder = os.path.join(os.path.expanduser("~"), ".spacr", "models")
        os.makedirs(folder, exist_ok=True)
        self._download = CheckpointDownload(entry, folder)
        self._download.progressed.connect(self._on_download_progress)
        self._download.finished.connect(self._on_download_finished)
        self._download_btn.setEnabled(False)
        self._download.start()

    def _on_download_progress(self, done: int, total: int) -> None:
        """Say how far the download has got."""
        from ..model_install import human_bytes

        self.set_preview_status(tr("Downloading {done} of {total}…",
                                   done=human_bytes(done),
                                   total=human_bytes(total) if total else "?"))

    def _on_download_finished(self, worked: bool, message: str) -> None:
        """Report the download."""
        self._download_btn.setEnabled(True)
        if worked:
            self._download_btn.hide()
            self.set_preview_status(tr("Downloaded to {path}. Press Run "
                                       "preview.", path=message))
        else:
            self.set_preview_status(tr("Download failed: {why}", why=message))

    def _offer_install(self, *, dialog: Any = None) -> None:
        """Install the figure reader into an environment of its own.

        Its download brings dependencies of its own, so they are contained
        in a separate environment. The same dialog the Model Zoo uses for
        Cellpose 3, DINOCell and SAMCell: it says where it installs and what
        it downloads, shows progress, and Cancel removes what it built.

        :param dialog: replaces the install dialog, for tests.
        """
        from ...plaque_papers import READER_BACKEND

        if dialog is None:
            from .model_zoo_picker import BackendInstallDialog

            dialog = BackendInstallDialog(READER_BACKEND, self)
        dialog.exec()
        if getattr(dialog, "installed", False):
            self.set_mode(self.mode())

    def _legend_for(self, stem: str) -> str:
        """The legend ``legends.csv`` holds for a figure, or ``''``."""
        from ...plaque_papers import LEGENDS_FILE, read_legends

        if not self._src:
            return ""
        folder = Path(self._src)
        folder = folder if folder.is_dir() else folder.parent
        return read_legends(folder / LEGENDS_FILE).get(stem, "")

    def _folder(self) -> Optional[Path]:
        """The figure folder."""
        if not self._src:
            return None
        folder = Path(self._src)
        return folder if folder.is_dir() else folder.parent

    def _show_figure(self, result: Dict[str, Any]) -> None:
        """Annotate a finished figure pass and fill the table."""
        self._clear_figure()
        result.setdefault("overlay", np.array(result["image"], copy=True))
        self._figure = result
        stem = Path(result["path"]).stem
        self._caption = self._legend_for(stem)
        self._reannotate()
        panels = sorted({a.panel for a in self._annotations if a.panel})
        if panels and not self._caption:
            self._legend_text.setText(tr(
                "Panel {letters} was read in {name}, but no legend is known "
                "for this figure. Paste the figure legend, or annotate by "
                "hand.", letters=", ".join(panels), name=Path(
                    result["path"]).name))
            self._legend_edit.clear()
            self._legend_box.show()
        else:
            self._legend_box.hide()
        status = tr(
            "{name}: {regions} plaque wells found, {words} words read. Click "
            "a well, then Plaque preview, or Find plaques in all wells.",
            name=Path(result["path"]).name,
            regions=len(result["regions"]), words=len(result["words"]))
        if str(result.get("words_source", "")).startswith("pdf text layer"):
            status += " " + tr("The words were read from the PDF's own text.")
        conflicts = sum(1 for a in self._annotations if a.conflict)
        if conflicts:
            status += " " + tr(
                "{n} well(s): the label and the legend disagree (marked "
                "conflict in Source).", n=conflicts)
        if result["regions"] and not any(
                getattr(s, "px_per_mm", None) for s in self._scales):
            status += " " + tr("No scale bar or whole well was found: plaque "
                               "sizes are in pixels.")
        self.set_preview_status(status)
        if result["regions"]:
            self.select_well(0)

    def _reannotate(self) -> None:
        """Propose conditions again, with the legend known now."""
        result = self._figure
        if result is None:
            return
        self._annotations = annotate_figure(
            result, self._caption, self._folder(),
            confirm=self._confirm.isChecked(), options=self.text_options())
        self._scales = _figure_scales(result, self._annotations, self._caption,
                                     self.current_settings().get("plate_format"))
        self._fill_table()
        self._fill_plaque_table()
        self._redraw_boxes()

    def _redraw_boxes(self) -> None:
        """Draw the figure with each box coloured by its OK tick."""
        result = self._figure
        if result is None:
            return
        ticks = [self._row_ok(i) for i in range(len(self._annotations))]
        self._view.set_image(result["overlay"],
                             list(zip(result["regions"], ticks)),
                             selected=self._selected)

    def _fill_table(self) -> None:
        """One row per plaque image.

        Every cell carries its annotation's index under ``Qt.UserRole``, so a
        row still finds its annotation after the user sorts the table. Sorting
        is off while the rows are written: a sorted table moves a row the
        moment a cell in the sorted column is set, and the rest of that row
        would land in whichever row took its place.
        """
        self._filling_table = True
        self._table.blockSignals(True)
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(self._annotations))
        for row, a in enumerate(self._annotations):
            well = self._wells.get(row)
            source = f"{a.source} / {a.strength}"
            if a.conflict:
                source = tr("{source} / conflict", source=source)
            cells = (str(row + 1), a.panel or "", a.label_text,
                     a.legend_text, a.condition, source,
                     str(well["count"]) if well else "",
                     f"{well['mean_area']:.0f}" if well else "")
            for column, text in enumerate(cells):
                item = table_item(text)
                item.setData(Qt.UserRole, row)
                item.setToolTip(text)
                if column != CONDITION_COLUMN:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if column == SOURCE_COLUMN and a.conflict:
                    item.setForeground(CONFLICT_COLOUR)
                    item.setToolTip(_conflict_note(a))
                self._table.setItem(row, column, item)
            ok = table_item("")
            ok.setData(Qt.UserRole, row)
            ok.setFlags((ok.flags() | Qt.ItemIsUserCheckable)
                        & ~Qt.ItemIsEditable)
            ok.setCheckState(Qt.Checked if a.approved else Qt.Unchecked)
            self._table.setItem(row, OK_COLUMN, ok)
        self._table.setSortingEnabled(True)
        if self._selected is not None:
            self._table.selectRow(self._view_row(self._selected))
        self._table.blockSignals(False)
        self._filling_table = False

    def _annotation_index(self, view_row: int) -> int:
        """The annotation a row of the Wells table shows, however it is sorted.

        :param view_row: a row as the table currently displays it.
        :returns: the 0-based annotation index; the row itself when the row
            carries no index.
        """
        item = self._table.item(view_row, 0)
        index = None if item is None else item.data(Qt.UserRole)
        return view_row if index is None else int(index)

    def _view_row(self, index: int) -> int:
        """The row of the Wells table showing annotation ``index`` now.

        :param index: 0-based annotation index.
        :returns: the displayed row; ``index`` itself when no row claims it.
        """
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is not None and item.data(Qt.UserRole) == index:
                return row
        return index

    def _clear_figure(self) -> None:
        """Forget the figure, its wells and both tables."""
        self._figure = None
        self._annotations = []
        self._scales = []
        self._wells = {}
        self._selected = None
        self._batch = []
        self._filling_table = True
        self._table.setRowCount(0)
        self._filling_table = False
        self._plaque_table.setRowCount(0)
        self._tabs.setTabText(1, tr("Plaques"))
        self._well_view.set_image(None)
        self._well_title.setText(tr(PICK_A_WELL))

    def _on_figure_clicked(self, x: float, y: float) -> None:
        """Select the well under a click on the figure."""
        if self._figure is None:
            return
        index = region_at(self._figure["regions"], x, y)
        if index is not None:
            self.select_well(index)

    def _on_table_selection(self) -> None:
        """Select the well of the row picked in the Wells table."""
        rows = {i.row() for i in self._table.selectedItems()}
        if len(rows) == 1:
            self.select_well(self._annotation_index(rows.pop()),
                             from_table=True)

    def _on_plaque_selection(self) -> None:
        """Select the well a plaque row belongs to."""
        rows = {i.row() for i in self._plaque_table.selectedItems()}
        if len(rows) != 1:
            return
        item = self._plaque_table.item(rows.pop(), 0)
        try:
            self.select_well(int(item.text()) - 1)
        except (AttributeError, ValueError):
            return

    def selected_well(self) -> Optional[int]:
        """The index of the highlighted well, or None."""
        return self._selected

    def select_well(self, index: int, *, from_table: bool = False) -> None:
        """Highlight one well: its box, its row, and its crop on the right.

        :param index: the well, 0-based in reading order.
        :param from_table: True when the Wells table asked, so its selection
            is left as the user made it.
        """
        result = self._figure
        if result is None or not 0 <= index < len(result["regions"]):
            return
        self._selected = index
        if not from_table:
            self._table.blockSignals(True)
            self._table.selectRow(self._view_row(index))
            self._table.blockSignals(False)
        self._redraw_boxes()
        self._show_well(index)

    def _show_well(self, index: int) -> None:
        """Draw the well's crop, with its plaques once they are found."""
        result = self._figure
        region = result["regions"][index]
        crop = np.array(result["image"][region.y0:region.y1,
                                        region.x0:region.x1], copy=True)
        well = self._wells.get(index)
        if well is not None:
            render_overlay(crop, well["labels"], self._overlay_style)
        self._well_view.set_image(crop)
        a = self._annotations[index] if index < len(self._annotations) else None
        parts = [tr("Well {n}", n=index + 1)]
        if a is not None and a.panel:
            parts.append(tr("panel {p}", p=a.panel))
        if a is not None and a.condition:
            parts.append(a.condition)
        if well is not None:
            parts.append(tr("{count} plaques, mean area {area:.0f} px",
                            count=well["count"], area=well["mean_area"]))
        else:
            parts.append(tr("press Plaque preview to find its plaques"))
        self._well_title.setText(" · ".join(parts))

    def preview_selected_well(self, *, segment: Optional[Callable] = None
                              ) -> bool:
        """Find the plaques in the highlighted well, off the GUI thread.

        :param segment: replaces the plaque model (tests).
        :returns: True when a pass was started.
        """
        if self._figure is None:
            self.set_preview_status(tr("Run preview on a figure first."))
            return False
        if self._selected is None:
            self.set_preview_status(tr(PICK_A_WELL))
            return False
        return self._start_wells([self._selected], segment)

    def find_plaques_in_all_wells(self, *, segment: Optional[Callable] = None
                                  ) -> bool:
        """Find the plaques in every detected well, one after another.

        :param segment: replaces the plaque model (tests).
        :returns: True when a pass was started.
        """
        if self._figure is None or not self._figure["regions"]:
            self.set_preview_status(tr("Run preview on a figure first."))
            return False
        return self._start_wells(list(range(len(self._figure["regions"]))),
                                 segment)

    def _start_wells(self, indices: List[int],
                     segment: Optional[Callable]) -> bool:
        """Queue wells for segmentation and start the first.

        :param indices: the wells, in the order to do them.
        :param segment: replaces the plaque model (tests).
        :returns: True when the first was started.
        """
        if self._jobs.is_busy():
            self.set_preview_status(tr("Preview already running."))
            return False
        self._run_token += 1
        self._batch = list(indices)
        self._batch_total = len(indices)
        self._batch_segment = segment
        self.set_preview_busy(True)
        self._next_well(self._run_token)
        return True

    def _next_well(self, token: int) -> None:
        """Segment the next queued well, or say the queue is done."""
        if self.preview_stale(token) or self._figure is None:
            return
        if not self._batch:
            self.set_preview_busy(False)
            total = sum(w["count"] for w in self._wells.values())
            self.set_preview_status(tr(
                "Done: {total} plaques in the {n} well(s) segmented so far.",
                total=total, n=len(self._wells)))
            return
        index = self._batch.pop(0)
        position = self._batch_total - len(self._batch)
        self.set_preview_status(tr(
            "Finding plaques in well {n} ({k} of {total})…", n=index + 1,
            k=position, total=self._batch_total))
        image = self._figure["image"]
        region = self._figure["regions"][index]
        settings = self.current_settings()
        segment = self._batch_segment
        self._jobs.submit(
            lambda: segment_well(image, region, settings, segment=segment),
            lambda result, t=token, i=index: self._on_well(t, i, result))

    def _on_well(self, token: int, index: int, result: Dict[str, Any]) -> None:
        """Adopt one segmented well and move on to the next."""
        if self.preview_stale(token) or self._figure is None:
            return
        if not isinstance(result, dict) or result.get("error"):
            self._batch = []
            self.set_preview_busy(False)
            if isinstance(result, dict):
                self.set_preview_status(result["error"])
                self._offer_download(result.get("entry"))
            return
        if result.get("note"):
            self._model_note.setText(result["note"])
        self._wells[index] = result
        self._repaint_overlay()
        self._fill_table()
        self._fill_plaque_table()
        if self._selected is None or self._selected == index:
            self.select_well(index)
        else:
            self._redraw_boxes()
        self.preview_ready.emit({"well": index, **result})
        self._next_well(token)

    def _repaint_overlay(self) -> None:
        """The figure with the outlines of every segmented well."""
        result = self._figure
        overlay = np.array(result["image"], copy=True)
        for index, well in self._wells.items():
            region = result["regions"][index]
            render_overlay(overlay, well["labels"], self._overlay_style,
                           offset=(region.y0, region.x0))
        result["overlay"] = overlay

    def plaque_table_rows(self) -> List[Dict[str, Any]]:
        """Every plaque found so far, with the well it is in.

        The ratio to the panel median is taken over every segmented well of
        the same panel, as the run's ``area_vs_panel_median`` is.

        :returns: dicts, one per plaque, well by well.
        """
        def annotation(index: int) -> Any:
            """The annotation of region ``index``, or None if it has none."""
            return self._annotations[index] \
                if index < len(self._annotations) else None

        by_panel: Dict[Optional[str], List[float]] = {}
        for index, well in self._wells.items():
            a = annotation(index)
            by_panel.setdefault(a.panel if a is not None else None, []).extend(
                r["area_px"] for r in well["rows"])
        medians = {k: float(np.median(v)) if v else 0.0
                   for k, v in by_panel.items()}
        scales = self._scales
        out = []
        for index in sorted(self._wells):
            a = annotation(index)
            panel = a.panel if a is not None else None
            median = medians.get(panel) or 0.0
            scale = scales[index] if index < len(scales) else None
            ppm = getattr(scale, "px_per_mm", None)
            for row in self._wells[index]["rows"]:
                out.append({"well": index + 1, "panel": panel or "",
                            "condition": a.condition if a is not None else "",
                            "area_vs_panel_median":
                            row["area_px"] / median if median else None,
                            **row,
                            "area_mm2": row["area_px"] / ppm ** 2 if ppm else None,
                            "scale": _scale_note(scale)})
        return out

    def _fill_plaque_table(self) -> None:
        """Rewrite the Plaques tab from the wells segmented so far."""
        rows = self.plaque_table_rows()
        self._plaque_table.blockSignals(True)
        self._plaque_table.setSortingEnabled(False)
        self._plaque_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, key in enumerate(PLAQUE_KEYS):
                value = row.get(key)
                if value is None:
                    text = ""
                elif isinstance(value, float):
                    text = f"{value:.2f}" if abs(value) < 100 else f"{value:.0f}"
                else:
                    text = str(value)
                self._plaque_table.setItem(r, c, table_item(text))
        self._plaque_table.setSortingEnabled(True)
        self._plaque_table.blockSignals(False)
        self._tabs.setTabText(1, tr("Plaques ({n})", n=len(rows)))

    def _row_ok(self, index: int) -> bool:
        """Whether annotation ``index`` is ticked OK, wherever its row sits."""
        item = self._table.item(self._view_row(index), OK_COLUMN)
        return item is not None and item.checkState() == Qt.Checked

    def _on_table_edit(self, item: QTableWidgetItem) -> None:
        """Keep the annotation and the boxes in step with an edit."""
        if self._filling_table:
            return
        row = self._annotation_index(item.row())
        if row >= len(self._annotations):
            return
        a = self._annotations[row]
        if item.column() == CONDITION_COLUMN:
            text = item.text().strip()
            if text and text != a.condition:
                a.condition, a.source, a.strength = text, "manual", "manual"
            self._fill_plaque_table()
        elif item.column() == OK_COLUMN:
            a.approved = self._row_ok(row)
            self._redraw_boxes()

    def table_rows(self) -> List[Dict[str, Any]]:
        """The review as :func:`spacr.plaque_papers.write_annotation_overrides`
        rows."""
        result = self._figure
        if result is None:
            return []
        from ...plaque_papers import _annotation_file_row

        name = Path(result["path"]).name
        rows = []
        for row in range(self._table.rowCount()):
            condition = self._table.item(self._view_row(row), CONDITION_COLUMN)
            text = condition.text().strip() if condition is not None else ""
            if row < len(self._annotations):
                rows.append(_annotation_file_row(
                    name, row + 1, self._annotations[row], condition=text,
                    approved=self._row_ok(row)))
            else:
                rows.append({"file": name, "region": row + 1,
                             "condition": text, "approved": self._row_ok(row)})
        return rows

    def save_annotations(self) -> Optional[Path]:
        """Write the review to ``figure_annotations.csv``, keeping other figures'.

        :returns: the file written, or None when there is nothing to save.
        """
        from ...plaque_papers import ANNOTATIONS_FILE, write_annotation_overrides

        rows = self.table_rows()
        folder = self._folder()
        if not rows or folder is None:
            self.set_preview_status(tr("Nothing to save: run the preview on "
                                       "a figure first."))
            return None
        path = write_annotation_overrides(folder / ANNOTATIONS_FILE, rows)
        ok = sum(1 for r in rows if r["approved"])
        self.set_preview_status(tr(
            "Saved {n} annotations ({ok} OK) to {path}.", n=len(rows), ok=ok,
            path=path))
        return path

    def _use_pasted_legend(self) -> None:
        """Key the pasted legend, and keep it for the run in ``legends.csv``."""
        from ...plaque_papers import LEGENDS_FILE

        text = self._legend_edit.toPlainText().strip()
        folder = self._folder()
        if not text or self._figure is None or folder is None:
            return
        stem = Path(self._figure["path"]).stem
        write_legend(folder / LEGENDS_FILE, stem, text)
        self._caption = " ".join(text.split())
        self._reannotate()
        self._legend_box.hide()
        self.set_preview_status(tr("Legend saved to {path}; conditions "
                                   "proposed again.",
                                   path=folder / LEGENDS_FILE))

    def _annotate_by_hand(self) -> None:
        """Put the cursor in the first condition cell."""
        self._legend_box.hide()
        if self._table.rowCount():
            self._table.setCurrentCell(0, CONDITION_COLUMN)
            self._table.editItem(self._table.item(0, CONDITION_COLUMN))

    def _on_confirm_toggled(self, on: bool) -> None:
        """Say what Confirm annotations does to the run."""
        self._confirm_note.setVisible(bool(on) and self.mode() == FIGURE_MODE)

    def _ask_for_paper(self) -> None:
        """Ask for a paper, then fetch it."""
        folder = self._folder()
        dialog = PaperDialog(self, str(folder) if folder else "")
        if dialog.exec() != QDialog.Accepted:
            return
        reference, parent = dialog.values()
        if not parent:
            parent = QFileDialog.getExistingDirectory(
                self, tr("Put the paper's folder in"), "")
        if reference and parent:
            self.fetch_paper(reference, parent)

    def fetch_paper(self, reference: str, parent: Any, *,
                    fetch: Optional[Callable] = None) -> bool:
        """Fetch a paper's figures into ``parent/<paper>``, off the GUI thread.

        The figure legends are gathered automatically:
        :func:`spacr.plaque_papers.fetch_paper_to_folder` writes the figures
        and ``legends.csv``, which this panel already reads. When it is done
        the form's ``src`` is pointed at the new folder, so the figures load
        the way any folder does.

        :param reference: a DOI, PMID, PMC id or PDF path.
        :param parent: the folder the paper's folder is made in.
        :param fetch: replaces ``fetch_paper_to_folder`` (tests).
        :returns: True when the fetch was started.
        """
        reference = str(reference or "").strip()
        if not reference or not parent:
            return False
        if self._paper_jobs.is_busy():
            self.set_preview_status(tr("A paper is already being fetched."))
            return False
        dest = Path(str(parent)).expanduser() / paper_folder_name(reference)
        if fetch is None:
            from ...plaque_papers import fetch_paper_to_folder as fetch
        self._paper_btn.setEnabled(False)
        self._paper_btn.setText(tr("Fetching…"))
        self.set_preview_status(tr("Fetching {ref} into {path}…",
                                   ref=reference, path=dest))
        self._paper_jobs.submit(lambda: fetch(reference, dest),
                                self._on_paper_fetched)
        return True

    def _paper_idle(self) -> None:
        """Put the paper button back."""
        self._paper_btn.setEnabled(True)
        self._paper_btn.setText(tr("From a paper…"))

    def _on_paper_failed(self, message: str) -> None:
        """Say why a fetch failed."""
        self._paper_idle()
        self.set_preview_status(tr("Could not fetch the paper: {why}",
                                   why=message))

    def _on_paper_fetched(self, result: Dict[str, Any]) -> None:
        """Report the fetch and switch the preview to the new folder."""
        self._paper_idle()
        folder = str(result.get("folder") or "")
        licence = result.get("licence") or tr("not stated")
        self._paper_note.setText(tr(
            "{paper}: {n} figures, {m} with legends (licence {licence}), in "
            "{path}.", paper=result.get("paper") or "", n=result.get("figures", 0),
            m=result.get("with_legend", 0), licence=licence, path=folder))
        self._paper_note.show()
        if not folder:
            return
        if self._propagate_cb is not None:
            try:
                self._propagate_cb({"src": folder})
            except Exception:
                LOG.debug("could not write src", exc_info=True)
        self.load_source_async(folder)

    def shutdown(self) -> None:
        """Leave no worker thread behind."""
        for runner in (self._jobs, self._load_jobs, self._paper_jobs):
            runner.shutdown()

    def closeEvent(self, event):                             # noqa: N802
        """Stop the workers with the panel.

        :param event: the close event.
        """
        self.shutdown()
        super().closeEvent(event)


def build_plaque_preview_card(host: Any):
    """Build the Plaque Assay ``Live preview`` card and panel, unplaced.

    :param host: the screen.
    :returns: ``(panel, card)``.
    """
    from PySide6.QtWidgets import QScrollArea

    from .card import Card

    card = Card(title="Live preview")
    scroll = QScrollArea(card)
    scroll.setObjectName("PlaquePreviewScroll")
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.viewport().setAutoFillBackground(False)
    panel = PlaquePreviewPanel()
    scroll.setWidget(panel)
    card.body_layout.addWidget(scroll)
    card.setMinimumHeight(360)
    return panel, card


def _form_value(widget: Any) -> str:
    """The text a form control holds.

    :param widget: a combo box, line edit or spaCR value widget.
    :returns: its value as text.
    """
    for name in ("get_value", "currentText", "text"):
        getter = getattr(widget, name, None)
        if callable(getter):
            try:
                return str(getter() or "")
            except Exception:
                continue
    return ""


def install_plaque_mode(screen: Any) -> Optional[PlaqueModeSwitch]:
    """Put the Plaque | Figure switch at the top of the settings and bind it.

    The form's ``plaque_mode`` control is the one source of truth. Both
    switches -- this one and the preview's -- write it, and follow it when
    something else changes it, such as a loaded settings file. Its row is
    hidden, because the switch speaks for it, and so are the settings the
    other mode does not read.

    :param screen: the Plaque Assay screen.
    :returns: the switch, or None when the screen has no settings column.
    """
    if getattr(screen, "_plaque_mode_switch", None) is not None:
        return screen._plaque_mode_switch
    layout = getattr(screen, "_settings_layout", None)
    model = getattr(screen, "_settings_model", None)
    if layout is None or model is None:
        return None
    switch = PlaqueModeSwitch()
    switch.setToolTip(tr("Plaque mode reads cropped plaque images; Figure "
                         "mode reads published figures. The settings below "
                         "and the live preview follow this switch."))
    layout.insertWidget(0, switch)
    screen._plaque_mode_switch = switch
    panel = getattr(screen, "_live_preview", None)
    widgets = getattr(model, "_widgets", {}) or {}
    field = widgets.get(MODE_KEY)
    state = {"mode": normalise_mode(
        _form_value(field) if field is not None else
        getattr(model, "_defaults", {}).get(MODE_KEY))}

    def show(mode: str) -> None:
        """Show ``mode`` on the switch and the preview, and hide the other mode's rows."""
        mode = normalise_mode(mode)
        state["mode"] = mode
        switch.set_mode(mode)
        if isinstance(panel, PlaquePreviewPanel) and panel.mode() != mode:
            panel.set_mode(mode)
        hide = set(keys_hidden_in(mode)) | {MODE_KEY}
        hider = getattr(model, "hide_the_rows_the_mode_leaves_out", None)
        if callable(hider):
            try:
                hider(hide)
            except Exception:
                LOG.debug("could not hide the other mode's rows",
                          exc_info=True)

    def choose(mode: str) -> None:
        """Store ``mode`` in the form, then show it."""
        mode = normalise_mode(mode)
        if field is not None:
            try:
                model.set_value_for_key(MODE_KEY, mode)
            except Exception:
                LOG.debug("could not write plaque_mode", exc_info=True)
        else:
            setter = getattr(model, "set_hidden_value", None)
            if callable(setter):
                try:
                    setter(MODE_KEY, mode)
                except Exception:
                    LOG.debug("could not keep plaque_mode", exc_info=True)
        show(mode)

    switch.mode_changed.connect(choose)
    if isinstance(panel, PlaquePreviewPanel):
        panel.mode_changed.connect(choose)
    if field is not None:
        for name in ("currentTextChanged", "textChanged", "value_changed"):
            signal = getattr(field, name, None)
            if signal is None:
                continue
            try:
                signal.connect(lambda *_a: show(_form_value(field)))
            except Exception:
                continue
            break
    show(state["mode"])
    return switch
