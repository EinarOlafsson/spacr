"""Four views of one segmentation, on one canvas, for any preview.

ONE WIDGET, TWO OWNERS. A segmentation preview has four things worth
looking at: the image with the outlines on it, the label masks alone, the
flow field the masks were pooled from, and the cell probability the flow
was thresholded at. Mask generation's live preview showed three of them
through a dropdown; the plaque preview showed its own four through tabs,
drawn by its own code. This module is the one place both can draw them
from, so that a change to how a cell probability is coloured reaches both
previews at once.

THE WIDGET KNOWS ARRAYS, NOT MODULES. It is handed an image, label masks,
flow pictures and cell probability maps -- each either one array or one per
object -- and draws whichever of :data:`VIEWS` is chosen. Nothing in here
names the Mask module, Cellpose settings or a plaque.

WHAT AN OWNER CAN CHANGE. The default renderers draw a plain overlay,
categorical masks, the flows as Cellpose hands them and the probability on
the magma scale. An owner with its own idea of an outline -- the live
preview's colour choice and thickness -- registers a renderer for that view
with :meth:`SegmentationViews.set_renderer` and keeps the other three. A
renderer answers with a picture, with a sentence to show in the picture's
place, or with ``None`` for the widget's own "nothing yet" sentence.

THE CANVAS IS PLUGGABLE. An owner that already has a canvas with a ruler
and a hover line on it (the live preview's ``_ZoomView``) hands it in;
otherwise a :class:`PictureCanvas` is built, which fits, zooms, pans and
saves on a right click like every other picture in the program.
"""
from __future__ import annotations

import colorsys
import logging
from typing import Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QComboBox, QLabel, QStackedLayout, QWidget

from ..i18n import set_translatable_items, tr
from .picture_export import install_picture_save
from .zoom_view import ZoomableImageView

LOG = logging.getLogger(__name__)

OVERLAY = "Overlay"
MASKS = "Masks"
FLOWS = "Flows"
CELLPROB = "Cell probability"

VIEWS: Tuple[str, ...] = (OVERLAY, MASKS, FLOWS, CELLPROB)
"""The four views, in the order every selector offers them."""

VIEW_HINTS: Dict[str, str] = {
    OVERLAY: "The image with every object's outline drawn over it.",
    MASKS: "The label masks alone, one colour per object.",
    FLOWS: "Cellpose's flow field: direction as hue, strength as brightness.",
    CELLPROB: "Cellpose's cell probability, 0 to 1, on the magma scale.",
}
"""One sentence per view, for a tooltip or a tab."""

SELECTOR_HINT = ("Right canvas: outline overlay · label masks · Cellpose "
                 "flows · cell probability")

WAITING = "Press Run preview to see this."
NO_CELLPROB = "This run gave no cell probability map."
NO_FLOWS = "This run gave no flows."
NO_MASKS = "This run gave no mask."
COULD_NOT_DRAW = "This view could not be drawn."

CELLPROB_COLORMAP = "magma"
"""The perceptual scale the probability is drawn on.

Magma runs dark to bright with no hue reversal, so "more probable" reads
as "brighter" and a threshold is one colour on it; the plaque preview
draws its probability on the same scale.
"""

DEFAULT_COLOURS: Tuple[Tuple[int, int, int], ...] = (
    (32, 220, 32), (222, 82, 200), (32, 200, 220), (255, 220, 32),
    (240, 60, 60), (120, 120, 255), (255, 140, 0), (240, 240, 240),
)
"""Colours dealt to object sets by position when an owner names none:
green, magenta, cyan, yellow, then round again."""

Arrays = Dict[str, np.ndarray]
Renderer = Callable[[Mapping[str, object]], Union[np.ndarray, str, None]]


def per_object(value) -> Arrays:
    """One array per object, whatever shape the owner handed over.

    :param value: ``None``, one array, or ``{object: array}``.
    :returns: ``{object: array}``; a bare array is keyed ``""`` and ``None``
        entries are dropped.
    """
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): np.asarray(array)
                for key, array in value.items() if array is not None}
    return {"": np.asarray(value)}


def to_rgb8(image) -> Optional[np.ndarray]:
    """``image`` as ``H x W x 3`` ``uint8``, or ``None`` for nothing drawable.

    A ``uint8`` picture passes through untouched. Anything else is stretched
    from its own minimum to its own maximum: the default for an owner that
    did not normalise, and no more than that, because normalisation is the
    owner's business (the live preview has percentiles for it).

    :param image: ``H x W``, ``H x W x 1`` or ``H x W x C`` in any dtype.
    """
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim not in (2, 3) or arr.size == 0:
        return None
    if arr.dtype != np.uint8:
        floats = arr.astype(np.float32)
        low, high = float(np.nanmin(floats)), float(np.nanmax(floats))
        if high <= low:
            arr = np.zeros(floats.shape, dtype=np.uint8)
        else:
            arr = np.clip(255.0 * (floats - low) / (high - low), 0, 255
                          ).astype(np.uint8)
    if arr.ndim == 2:
        return np.ascontiguousarray(np.stack([arr, arr, arr], axis=-1))
    if arr.shape[-1] >= 3:
        return np.ascontiguousarray(arr[..., :3])
    pad = np.zeros(arr.shape[:2] + (3 - arr.shape[-1],), dtype=np.uint8)
    return np.ascontiguousarray(np.concatenate([arr, pad], axis=-1))


def to_qpixmap(rgb) -> QPixmap:
    """A ``QPixmap`` of ``rgb``; a null one when there is nothing to draw."""
    picture = to_rgb8(rgb)
    if picture is None:
        return QPixmap()
    height, width = picture.shape[:2]
    image = QImage(picture.tobytes(), width, height, width * 3,
                   QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def boundary_of(labels: np.ndarray) -> np.ndarray:
    """The 4-connected boundary of a label image, as a bool array."""
    edge = np.zeros(labels.shape, dtype=bool)
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return edge


def _dilate(pixels: np.ndarray, times: int) -> np.ndarray:
    """``pixels`` grown by one 4-neighbour ring, ``times`` over."""
    out = pixels
    for _ in range(max(0, int(times))):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def label_palette(count: int, seed: int = 0) -> np.ndarray:
    """``count`` vivid colours, spread round the hue circle by the golden angle.

    Deterministic for a given ``seed``, so the same object keeps its colour
    from one redraw to the next.

    :returns: ``count x 3`` ``uint8``.
    """
    out = np.zeros((max(0, int(count)), 3), dtype=np.uint8)
    for index in range(out.shape[0]):
        hue = ((index + int(seed) * 7) * 0.618033988749895) % 1.0
        value = 0.95 if index % 2 == 0 else 0.75
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, value)
        out[index] = (int(red * 255), int(green * 255), int(blue * 255))
    return out


def render_overlay(image, labels, *,
                   colours: Optional[Mapping[str, Tuple[int, int, int]]] = None,
                   thickness: int = 1) -> Optional[np.ndarray]:
    """The image with each object set's outline drawn over it.

    :param image: the field, any dtype.
    :param labels: one label image or ``{object: labels}``.
    :param colours: outline colour per object key; unnamed keys are dealt
        :data:`DEFAULT_COLOURS` by position.
    :param thickness: outline width in pixels, 1 to 5.
    :returns: ``H x W x 3`` ``uint8``, the plain image when there are no
        labels, or ``None`` when there is no image either.
    """
    rgb = to_rgb8(image)
    if rgb is None:
        return None
    rgb = rgb.copy()
    thickness = max(1, min(5, int(thickness)))
    for index, (key, mask) in enumerate(per_object(labels).items()):
        if mask.ndim != 2 or mask.shape != rgb.shape[:2] or not mask.any():
            continue
        edge = boundary_of(mask.astype(np.int64)) & (mask > 0)
        edge = _dilate(edge, thickness - 1)
        colour = (colours or {}).get(key)
        if colour is None:
            colour = DEFAULT_COLOURS[index % len(DEFAULT_COLOURS)]
        rgb[edge] = np.array(colour, dtype=np.uint8)
    return rgb


def render_labels(labels) -> Optional[np.ndarray]:
    """Every object in its own colour on black.

    :param labels: one label image or ``{object: labels}``.
    :returns: ``H x W x 3`` ``uint8``, or ``None`` without any labels.
    """
    sets = [(key, mask) for key, mask in per_object(labels).items()
            if mask.ndim == 2]
    if not sets:
        return None
    shape = sets[0][1].shape
    out = np.zeros(shape + (3,), dtype=np.uint8)
    for index, (_key, mask) in enumerate(sets):
        if mask.shape != shape:
            continue
        present = mask > 0
        if not present.any():
            continue
        ids = np.unique(mask[present])
        palette = label_palette(len(ids), seed=index)
        out[present] = palette[np.searchsorted(ids, mask[present])]
    return out


def render_flows(flows) -> Optional[np.ndarray]:
    """The flow pictures, one per object, blended by maximum.

    :param flows: one ``H x W x 3`` picture or ``{object: picture}``.
    :returns: ``H x W x 3`` ``uint8``, or ``None`` without any flows.
    """
    pictures = [to_rgb8(flow) for flow in per_object(flows).values()
                if flow.ndim == 3]
    pictures = [picture for picture in pictures if picture is not None]
    if not pictures:
        return None
    out = pictures[0].copy()
    for picture in pictures[1:]:
        if picture.shape == out.shape:
            out = np.maximum(out, picture)
    return out


def cell_probability(cellprob) -> Optional[np.ndarray]:
    """Cellpose's logits as a probability, 0 to 1, over every object.

    Several objects give one map: the highest probability at each pixel,
    so a nucleus pass and a cell pass do not hide one another.

    :param cellprob: one ``H x W`` logit map or ``{object: map}``.
    :returns: ``H x W`` ``float32``, or ``None`` without any map.
    """
    prob = None
    for logits in per_object(cellprob).values():
        if logits.ndim != 2:
            continue
        clipped = np.clip(logits.astype(np.float32), -30.0, 30.0)
        here = 1.0 / (1.0 + np.exp(-clipped))
        if prob is None:
            prob = here
        elif here.shape == prob.shape:
            prob = np.maximum(prob, here)
    return prob


def render_cellprob(cellprob) -> Optional[np.ndarray]:
    """The cell probability on the magma scale.

    Cellpose hands the probability back as logits; it is drawn as the
    probability itself on a FIXED scale, so two runs can be compared by eye
    and a ``cellprob_threshold`` of ``t`` sits at the colour of
    ``1 / (1 + e^-t)`` in every picture.

    :param cellprob: one ``H x W`` logit map or ``{object: map}``.
    :returns: ``H x W x 3`` ``uint8``, or ``None`` without any map.
    """
    prob = cell_probability(cellprob)
    if prob is None:
        return None
    try:
        from matplotlib import colormaps

        rgba = colormaps[CELLPROB_COLORMAP](prob)
        return np.ascontiguousarray(
            (rgba[..., :3] * 255.0).round().astype(np.uint8))
    except Exception:
        LOG.debug("matplotlib could not colour the probability", exc_info=True)
        grey = (prob * 255.0).round().astype(np.uint8)
        return np.ascontiguousarray(np.stack([grey] * 3, axis=-1))


def picture_name_of(view: str) -> str:
    """The file name a saved picture of ``view`` is offered under."""
    return str(view or "picture").lower().replace(" ", "_")


class PictureCanvas(ZoomableImageView):
    """The canvas built when an owner brings none.

    :class:`~spacr.qt.widgets.zoom_view.ZoomableImageView` fits, zooms and
    pans; this adds what :class:`SegmentationViews` asks of a canvas -- the
    picture at its own resolution and a name for it -- and the right-click
    save every other picture in the program has.

    :param parent: parent widget; ownership only.
    """

    def __init__(self, parent=None) -> None:
        """Build the canvas with the save menu on it."""
        super().__init__(parent)
        self._picture_name = "picture"
        install_picture_save(self, self.picture, self.picture_name)

    def picture(self) -> Optional[QPixmap]:
        """What is shown, at the resolution it was rendered at, or ``None``."""
        if self._item is None:
            return None
        pixmap = self._item.pixmap()
        return None if pixmap.isNull() else pixmap

    def picture_name(self) -> str:
        """The name offered when the picture is saved."""
        return self._picture_name

    def set_picture_name(self, name: str) -> None:
        """Name what the canvas shows, for the save dialog."""
        self._picture_name = str(name or "picture")


class SegmentationViews(QWidget):
    """Overlay, masks, flows and cell probability, one at a time, on a canvas.

    :param parent: parent widget.
    :param canvas: the widget the picture is drawn on. It needs
        ``set_pixmap(QPixmap)`` and ``picture()``; ``set_picture_name`` is
        used when it is there. A :class:`PictureCanvas` is built when none
        is given.

    The widget is a stack of two pages: the canvas, and a sentence for a
    view that has nothing to show yet. :attr:`canvas` and :attr:`message`
    are both public so an owner can style or wire them.
    """

    view_changed = Signal(str)
    """Emitted with the view's name when the chosen view changes."""

    def __init__(self, parent=None, *, canvas=None) -> None:
        """Build the two pages and start on the waiting sentence."""
        super().__init__(parent)
        self.setObjectName("SegmentationViews")
        self.canvas = canvas if canvas is not None else PictureCanvas(self)
        self.message = QLabel(self)
        self.message.setObjectName("SegmentationViewMessage")
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignCenter)
        self.message.setMargin(16)
        self._pages = QStackedLayout(self)
        self._pages.setContentsMargins(0, 0, 0, 0)
        self._pages.addWidget(self.canvas)
        self._pages.addWidget(self.message)
        self._view: str = OVERLAY
        self._arrays: Dict[str, object] = {
            "image": None, "labels": {}, "flows": {}, "cellprob": {}}
        self._renderers: Dict[str, Renderer] = {}
        self._selectors: List[QComboBox] = []
        self._drawn: Optional[np.ndarray] = None
        self.show_message(tr(WAITING))

    def views(self) -> Tuple[str, ...]:
        """The view names, in the order they are offered."""
        return VIEWS

    def view(self) -> str:
        """The view being shown, one of :data:`VIEWS`."""
        return self._view

    def set_view(self, name: str, *, refresh: bool = True) -> None:
        """Show ``name``, and put every selector on it.

        :param name: one of :data:`VIEWS`.
        :param refresh: redraw now; ``False`` when arrays are about to
            change too and one redraw at the end is wanted.
        :raises ValueError: for a name that is not a view.
        """
        name = str(name or "")
        if name not in VIEWS:
            raise ValueError(f"{name!r} is not one of {VIEWS}")
        changed = name != self._view
        self._view = name
        for combo in self._selectors:
            if combo.currentData() != name:
                blocked = combo.blockSignals(True)
                try:
                    combo.setCurrentIndex(combo.findData(name))
                finally:
                    combo.blockSignals(blocked)
        if changed:
            self.view_changed.emit(name)
        if refresh:
            self.refresh()

    def make_selector(self, parent=None) -> QComboBox:
        """A dropdown of the four views, kept in step with this widget.

        The captions follow the language and the entries carry the English
        names as their data, as every value-carrying dropdown does. An owner
        may place it wherever its layout wants; there can be several.

        :param parent: the dropdown's parent; this widget by default.
        """
        combo = QComboBox(parent if parent is not None else self)
        set_translatable_items(combo, VIEWS)
        combo.setToolTip(tr(SELECTOR_HINT))
        combo.setCurrentIndex(combo.findData(self._view))
        combo.currentIndexChanged.connect(
            lambda _index, box=combo: self.set_view(box.currentData()))
        self._selectors.append(combo)
        return combo

    def set_renderer(self, view: str, renderer: Optional[Renderer]) -> None:
        """Draw ``view`` with ``renderer`` instead of the default.

        :param view: one of :data:`VIEWS`.
        :param renderer: called with the arrays (``image``, ``labels``,
            ``flows``, ``cellprob``); answers a picture, a sentence to show
            instead, or ``None`` for the waiting sentence. ``None`` here
            restores the default renderer.
        """
        if view not in VIEWS:
            raise ValueError(f"{view!r} is not one of {VIEWS}")
        if renderer is None:
            self._renderers.pop(view, None)
        else:
            self._renderers[view] = renderer

    def set_arrays(self, *, image=None, labels=None, flows=None,
                   cellprob=None, refresh: bool = True) -> None:
        """Hand over what there is to draw.

        :param image: the field, any dtype, or ``None``.
        :param labels: one label image or ``{object: labels}``.
        :param flows: one flow picture or ``{object: picture}``.
        :param cellprob: one logit map or ``{object: map}``.
        :param refresh: redraw the chosen view now.
        """
        self._arrays = {
            "image": None if image is None else np.asarray(image),
            "labels": per_object(labels),
            "flows": per_object(flows),
            "cellprob": per_object(cellprob),
        }
        if refresh:
            self.refresh()

    def arrays(self) -> Dict[str, object]:
        """What was last handed over, as the renderers see it."""
        return dict(self._arrays)

    def refresh(self) -> None:
        """Redraw the chosen view from the arrays."""
        rendered = self._render(self._view)
        if isinstance(rendered, str):
            self.show_message(rendered)
        elif rendered is None:
            self.show_message(tr(WAITING))
        else:
            self.show_picture(rendered)

    def _render(self, view: str):
        """The picture or sentence for ``view``; a sentence on a failure."""
        renderer = self._renderers.get(view)
        try:
            if renderer is not None:
                return renderer(self._arrays)
            return self._default(view)
        except Exception:
            LOG.debug("the %s view could not be drawn", view, exc_info=True)
            return tr(COULD_NOT_DRAW)

    def _default(self, view: str):
        """The built-in rendering of ``view``.

        Once there are masks, a view with nothing to draw says which array
        the run did not give -- a classical method makes no cell
        probability -- rather than asking for a run that has happened.
        """
        arrays = self._arrays
        if view == OVERLAY:
            return render_overlay(arrays["image"], arrays["labels"])
        if view == MASKS:
            picture, missing = render_labels(arrays["labels"]), NO_MASKS
        elif view == FLOWS:
            picture, missing = render_flows(arrays["flows"]), NO_FLOWS
        else:
            picture, missing = (render_cellprob(arrays["cellprob"]),
                                NO_CELLPROB)
        if picture is None and arrays["labels"]:
            return tr(missing)
        return picture

    def show_picture(self, picture) -> None:
        """Put ``picture`` on the canvas and bring the canvas to the front.

        :param picture: ``H x W x 3`` ``uint8`` (or anything
            :func:`to_rgb8` takes), or a ``QPixmap``.
        """
        if isinstance(picture, QPixmap):
            pixmap = picture
            self._drawn = None
        else:
            self._drawn = to_rgb8(picture)
            pixmap = to_qpixmap(self._drawn)
        namer = getattr(self.canvas, "set_picture_name", None)
        if callable(namer):
            namer(picture_name_of(self._view))
        self.canvas.set_pixmap(pixmap)
        self._pages.setCurrentWidget(self.canvas)

    def show_message(self, text: str) -> None:
        """Show ``text`` where the picture would be.

        :param text: what to say, already translated.
        """
        self._drawn = None
        self.message.setText(str(text))
        self._pages.setCurrentWidget(self.message)

    def is_showing_message(self) -> bool:
        """Whether a sentence, rather than a picture, is on show."""
        return self._pages.currentWidget() is self.message

    def message_text(self) -> str:
        """The sentence on show, or ``""`` while a picture is."""
        return self.message.text() if self.is_showing_message() else ""

    def rendered(self) -> Optional[np.ndarray]:
        """The picture drawn last, as ``uint8`` RGB, or ``None``."""
        return self._drawn

    def picture(self) -> Optional[QPixmap]:
        """The picture on the canvas, or ``None`` while a sentence shows."""
        if self.is_showing_message():
            return None
        return self.canvas.picture()

    def picture_name(self) -> str:
        """The file name a saved picture of the chosen view is offered under."""
        return picture_name_of(self._view)
