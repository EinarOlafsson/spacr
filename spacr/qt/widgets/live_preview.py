"""
Live-preview segmentation widget — v2.

Interactive Cellpose tuning surface for the Mask app screen. It provides:

* **Zoomable canvases (Ctrl+scroll, in sync).** Both the original and
  the mask overlay live in a shared :class:`QGraphicsView` pair — pan
  and zoom on one and the other tracks pixel-for-pixel.
* **Hover tooltip.** Move the cursor over the original and a pinned
  status line shows the pixel intensity for every channel plus, when
  present, the object label at that position from the last segmenta-
  tion. Same tooltip regardless of which view holds the cursor.
* **Normalise toggle.** Optional 2–98 % percentile stretch (per channel
  for RGB) so raw low-contrast tiles are legible.
* **Model-aware options.** Every model shows the full segmentation set.
  Cellpose-SAM does *not* ignore ``flow_threshold``, ``cellprob`` or
  ``diameter`` — see :data:`DIAMETER_TOOLTIP` for the measurement that
  killed that belief.
* **Outline colour + thickness.** Chosen from the toolbar; effect is
  live once a mask exists. ``color (random)`` assigns a stable categorical
  colour to every object label so touching masks remain distinguishable.
* **Multi-object segmentation.** An "object type" combo picks between
  ``cell``, ``nucleus``, and ``cell + nucleus``. In cell+nucleus mode
  the panel runs two Cellpose passes and overlays both masks in
  distinct colours.
* **The model the RUN will use, and it says which.** The panel reads the
  same setting the pipeline reads -- ``pathogen_model`` over
  ``pathogen_model_name`` for pathogens, ``<object>_model_name``
  otherwise (:func:`_model_keys_for`) -- offers the model zoo beside the
  combo, and names the model that produced the masks on the status line.
  A checkpoint that is not on this machine previews with cpsam AND SAYS
  SO rather than stalling or substituting in silence.
* **Pre / Post filters.** When the object type is ``cell`` (or the
  combined mode) the panel routes pre / post-processing settings from
  the Mask app (``cell_min_size``, ``cell_max_size``,
  ``remove_background_cell``, background intensity, ...) through the
  segmentation. Users toggle these on/off with dedicated "Pre" / "Post"
  clickable labels sitting next to "Run preview" in the same visual
  style as the LP / AI toggles.

The whole file stays safe to import without cellpose — every cellpose
call is lazy-imported inside the worker thread.
"""
from __future__ import annotations

import colorsys
from copy import deepcopy
import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QGraphicsPixmapItem,
    QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QPushButton,
    QHeaderView, QSizePolicy, QSpinBox, QTableWidget,
    QTableWidgetItem,
    QVBoxLayout, QWidget,
)
from .preview_controls import (
    DEFAULT_MAX_SETS, DEFAULT_METADATA_TYPE, MAX_SETS_TOOLTIP, FlatButton,
    FlatComboBox, FlatSpinBox, ImageSetSampler, apply_sample_to_combo,
    channel_view, enumerate_image_sets, populate_channel_combo,
    sample_image_sets, sample_seed, selected_channel,
)
from .preview_contract import (
    PREVIEW_CANCEL_TEXT, PREVIEW_RUN_TEXT, PREVIEW_RUNNING_MESSAGE,
    LivePreviewContract, preview_cellpose_model, preview_failure_message,
)
from .percentile_pair import DECIMALS as PERCENTILE_DECIMALS
from .toggle import Toggle
from ..i18n import set_translatable_items, tr
from ..job_runner import JobRunner
from ...organelle_types import (organelle_count, organelle_role,
                                organelle_roles)

LOG = logging.getLogger("spacr.qt.live_preview")

SUPPORTED_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

_PLANE_ROLE = int(Qt.UserRole) + 1

#: Images drawn at once before the selection is truncated. One keeps the
#: panel behaving as it always did until the user asks for more.
DEFAULT_MAX_IMAGES = 1

#: Tooltip for the diameter spinner, in every model.
#:
#: The panel used to disable this control for ``cpsam`` and label it
#: "Ignored by Cellpose-SAM". That is false. ``CellposeModel._run_cp``
#: in cellpose 4.0.7 still does ``image_scaling = 30. / diameter``
#: whenever ``diameter is not None``, so the value decides the scale the
#: image is segmented at. Measured on an RTX 3090 against a real
#: micrograph (``plate1_E01_10.tif``, 1994x1994, cpsam, flow 0.4,
#: cellprob 0.0), counting objects per pass:
#:
#: ===============  ========  ===========
#: diameter         cells     nuclei
#: ===============  ========  ===========
#: unset (``None``)       66           65
#: 30                     66           65
#: 60                     71           63
#: ===============  ========  ===========
#:
#: 30 matches "unset" because 30/30 is a no-op rescale. That is the only
#: reason the control ever looked inert, and 30 is the spinner's default
#: — anyone who checked the claim without moving the value saw no change
#: and believed it. Greying it out took away a control that measurably
#: changes what cpsam finds.
DIAMETER_TOOLTIP = (
    "(float, px) Expected object diameter. Cellpose-SAM uses it: the "
    "image is rescaled by 30/diameter before segmentation, so raising it "
    "finds bigger objects and lowering it finds smaller ones. 30 is the "
    "no-op (30/30 = 1) and 0 means 'unset', which is the same thing."
)

OBJECT_TYPES = ("cell", "nucleus", "cell + nucleus", "pathogen", "organelle")

#: The object choices that are never per-slot, in the order they are offered.
FIXED_OBJECT_TYPES = ("cell", "nucleus", "cell + nucleus", "pathogen")


def organelle_label(number: int) -> str:
    """The dropdown caption for organelle slot ``number``.

    Slot 1 stays plain ``organelle``: one organelle is the ordinary case, and
    numbering it "organelle 1" would relabel every existing screen to say
    something new about a run that has not changed.
    """
    return "organelle" if int(number) <= 1 else f"organelle {int(number)}"


def object_role(label: str) -> str:
    """The settings ROLE an object-dropdown caption stands for.

    ``organelle`` is slot 1, whose role has the same name; ``organelle 2`` is
    ``organelleb``, which is the prefix its settings keys actually carry. The
    dropdown counts because that is what the main panel counts, and the roles
    use letters because a digit cannot start a Python identifier.
    """
    if not isinstance(label, str) or not label.startswith("organelle"):
        return label
    tail = label[len("organelle"):].strip()
    if not tail:
        return "organelle"
    try:
        return organelle_role(int(tail))
    except (TypeError, ValueError):
        return "organelle"

COMPARTMENTS = ("cell", "nucleus", "pathogen", "organelle")

OBJECT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "cell":      (32, 220, 32),
    "nucleus":   (222, 82, 200),
    "pathogen":  (32, 200, 220),
    "organelle": (255, 220, 32),
}

RANDOM_OUTLINE_SEEDS: Dict[str, int] = {
    "cell": 11,
    "nucleus": 37,
    "pathogen": 61,
    "organelle": 89,
}

#: The organelle's own segmentation settings, grouped the way the pipeline
#: dispatches them.
#:
#: WHY THIS TABLE EXISTS. Every other compartment is segmented by Cellpose and
#: needs the generic filters in :data:`COMPARTMENT_FIELDS`. An organelle is
#: not: `spacr.object._segment_single_image` dispatches on
#: ``organelle_morphology`` first and ``organelle_method`` second, and each
#: morphology reads a different set of about half a dozen knobs. Fifty-odd
#: settings existed for that and NONE of them were reachable from the live
#: preview, which is what "there is no way to live preview the organelle
#: settings except for the cellpose model" meant.
#:
#: Keyed by morphology, so only the knobs that morphology actually reads are
#: shown. ``None`` holds the four that always apply. The groups mirror
#: `spacr.object._extract_classical_settings` -- if they drift, the panel
#: offers a setting the segmentation never reads.
ORGANELLE_METHOD_FIELDS: Dict[Optional[str], tuple] = {
    None: (
        ("morphology", "Morphology", "morphology", None),
        ("method", "Method", "method_choice", None),
        ("min_size", "Min size (px²)", "int", (0, 100_000_000, 0)),
        ("max_size", "Max size (px²)", "int", (0, 100_000_000, 0)),
    ),
    "spots": (
        ("tophat_radius", "Top-hat radius", "int", (0, 1_000, 0)),
        ("watershed_spots", "Watershed spots", "bool", None),
        ("log_min_sigma", "LoG min sigma", "int", (0, 1_000, 1)),
        ("log_max_sigma", "LoG max sigma", "int", (0, 1_000, 5)),
        ("log_num_sigma", "LoG sigma steps", "int", (1, 100, 5)),
        ("log_threshold", "LoG threshold", "float", (0.0, 1_000.0, 0.1)),
        ("dog_sigma_low", "DoG sigma low", "float", (0.0, 1_000.0, 1.0)),
        ("dog_sigma_high", "DoG sigma high", "float", (0.0, 1_000.0, 5.0)),
    ),
    "network": (
        ("ridge_filter", "Ridge filter", "ridge", None),
        ("network_threshold", "Network threshold", "network", None),
        ("skeletonize", "Skeletonize", "bool", None),
        ("hysteresis_low", "Hysteresis low", "float", (0.0, 1.0, 0.1)),
        ("hysteresis_high", "Hysteresis high", "float", (0.0, 1.0, 0.3)),
    ),
    "irregular": (
        ("adaptive_block_size", "Adaptive block size", "int", (1, 9_999, 51)),
        ("adaptive_offset", "Adaptive offset", "int", (-1_000, 1_000, 0)),
        ("morph_radius", "Morph radius", "int", (0, 1_000, 1)),
        ("fill_holes", "Fill holes", "int", (0, 100_000_000, 0)),
    ),
    "ring": (
        ("ring_sigma_inner", "Ring sigma inner", "float", (0.0, 1_000.0, 1.0)),
        ("ring_sigma_outer", "Ring sigma outer", "float", (0.0, 1_000.0, 3.0)),
        ("ring_min_prominence", "Ring min prominence",
         "float", (0.0, 1_000.0, 0.0)),
        ("ring_fill_method", "Ring fill", "ring_fill", None),
    ),
}

#: The morphologies, in the order the settings panel offers them.
ORGANELLE_MORPHOLOGIES = ("spots", "network", "irregular", "ring")

COMPARTMENT_FIELDS = (
    ("min_area",                   "Min area (px²)",        "int",   (0, 100_000_000, 0)),
    ("max_area",                   "Max area (px²)",        "int",   (0, 100_000_000, 0)),
    ("min_intensity",              "Min intensity",         "float", (0.0, 1_000_000_000_000.0, 0.0)),
    ("max_intensity",              "Max intensity",         "float", (0.0, 1_000_000_000_000.0, 0.0)),
    ("perimeter_fraction",         "Perimeter fraction",    "float", (0.0, 1.0, 0.0)),
    ("remove_border_objects",      "Remove border objects", "bool",  None),
)

OUTLINE_CHOICES = ("auto", "color (random)", "green", "magenta",
                   "yellow", "cyan", "white", "red")

VIEW_MODES = ("Overlay", "Masks", "Flows")



def load_preview_image(path: Path) -> np.ndarray:
    """Read *path* into an (H, W) or (H, W, C) uint8/uint16 array.

    Tifffile is used for TIFFs to preserve bit-depth; other formats fall
    back to PIL. Raises :class:`FileNotFoundError` if the path is bad.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suf = path.suffix.lower()
    if suf in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(str(path))
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im)


def load_preview_mip(paths) -> np.ndarray:
    """Max-project a field's planes, the way the ingest already does.

    ``io._rename_and_organize_image_files`` reduces every z-stack to
    ``np.max`` over its planes, per field and per channel, before anything
    reaches ``stack/``. This is the preview's copy of that, so what the user
    is looking at is what masking will actually run on.

    Planes are folded one at a time rather than stacked: a 60-plane field at
    2048x2048 uint16 is 500 MB as one array and 8 MB folded.

    :param paths: plane paths in acquisition order; one path is returned
        unchanged, so a flat 2-D field costs nothing.
    :raises FileNotFoundError: if no path can be read.
    """
    projected = None
    for path in paths:
        plane = load_preview_image(path)
        if projected is None:
            projected = plane
            continue
        if plane.shape != projected.shape:
            raise ValueError(
                f"plane {Path(path).name} is {plane.shape}, expected "
                f"{projected.shape} — these files are not one z-stack")
        projected = np.maximum(projected, plane)
    if projected is None:
        raise FileNotFoundError("no readable planes")
    return projected


def _widget_text(widget) -> str:
    """Best-effort current value of a settings widget, or ``""``."""
    if widget is None:
        return ""
    for attr in ("currentText", "text"):
        getter = getattr(widget, attr, None)
        if callable(getter):
            try:
                return (getter() or "").strip()
            except Exception:
                return ""
    return ""


def _full_range_max(img: np.ndarray) -> float:
    """Return the value that maps to white for a *raw* (un-normalised) view.

    For integer images this is the dtype maximum (e.g. 65535 for uint16), so
    a 16-bit image whose real values are small reads dark — the true raw
    view. For float images we assume a [0, 1] range unless the data clearly
    exceeds it, in which case we use the data max.
    """
    if np.issubdtype(img.dtype, np.integer):
        return float(np.iinfo(img.dtype).max)
    m = float(np.nanmax(img)) if img.size else 1.0
    return 1.0 if m <= 1.0 else m


def _to_uint8(img: np.ndarray, normalise: bool = True,
                lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    """Return a viewable uint8 version of *img*.

    :param normalise: when True apply a per-channel percentile stretch. When
        False, map the *full bit-depth range* (0 → dtype max) to 0–255, i.e.
        the raw view — a 16-bit image with small values reads dark/black,
        not blown out. (Previously this clipped to [0, 255], which turned a
        16-bit image mostly white.)
    :param lo_pct: lower percentile for the stretch (default 2 %).
    :param hi_pct: upper percentile for the stretch (default 98 %).
    """
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    full_max = _full_range_max(img) or 1.0
    if img.ndim == 3:
        out = np.zeros(img.shape[:2] + (3,), dtype=np.uint8)
        for c in range(min(3, img.shape[-1])):
            slice_ = img[..., c].astype(np.float32)
            if normalise:
                lo, hi = np.percentile(slice_, (lo_pct, hi_pct))
                if hi <= lo:
                    continue
                out[..., c] = np.clip(
                    255 * (slice_ - lo) / (hi - lo), 0, 255,
                ).astype(np.uint8)
            else:
                out[..., c] = np.clip(
                    255 * slice_ / full_max, 0, 255).astype(np.uint8)
        return out
    arr = img.astype(np.float32)
    if normalise:
        lo, hi = np.percentile(arr, (lo_pct, hi_pct))
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)
        return np.clip(
            255 * (arr - lo) / (hi - lo), 0, 255,
        ).astype(np.uint8)
    return np.clip(255 * arr / full_max, 0, 255).astype(np.uint8)


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Return a bool array marking the 4-connected boundary of ``mask``."""
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :]  |= mask[1:, :]  != mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, 1:]  |= mask[:, 1:]  != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundary


def _labelled_boundary(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Return each outline pixel's positive object label.

    Unlike :func:`_boundary_mask`, the result retains object identity so a
    categorical colour map can draw every segmented object differently. The
    label is also propagated onto the exterior half of an outline and through
    any requested dilation. Where two dilated outlines meet, the larger label
    wins deterministically.

    :param mask: two-dimensional integer label image; zero is background.
    :param thickness: outline thickness in pixels, clamped to ``1..5``.
    :returns: int64 array containing object labels only on outline pixels.
    """
    labels = np.asarray(mask, dtype=np.int64)
    if labels.ndim != 2 or not np.any(labels > 0):
        return np.zeros(labels.shape, dtype=np.int64)

    thickness = max(1, min(5, int(thickness)))
    boundary = _boundary_mask(labels)
    owners = np.where(boundary & (labels > 0), labels, 0)

    neighbours = np.zeros_like(labels)
    neighbours[1:, :] = np.maximum(neighbours[1:, :], labels[:-1, :])
    neighbours[:-1, :] = np.maximum(neighbours[:-1, :], labels[1:, :])
    neighbours[:, 1:] = np.maximum(neighbours[:, 1:], labels[:, :-1])
    neighbours[:, :-1] = np.maximum(neighbours[:, :-1], labels[:, 1:])
    exterior = boundary & (owners == 0)
    owners[exterior] = neighbours[exterior]

    for _ in range(thickness - 1):
        expanded = np.zeros_like(owners)
        expanded[1:, :] = np.maximum(expanded[1:, :], owners[:-1, :])
        expanded[:-1, :] = np.maximum(expanded[:-1, :], owners[1:, :])
        expanded[:, 1:] = np.maximum(expanded[:, 1:], owners[:, :-1])
        expanded[:, :-1] = np.maximum(expanded[:, :-1], owners[:, 1:])
        owners = np.where(owners > 0, owners, expanded)
    return owners


def _random_outline_palette(
    labels: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Return vivid, deterministic random-looking RGB colours for labels.

    Golden-ratio hue spacing keeps adjacent integer labels separated while
    deterministic saturation/value jitter makes the result look like a
    random categorical colormap. Stability is intentional: changing zoom,
    normalisation, or thickness must not recolour every object.

    :param labels: one-dimensional array of positive object labels.
    :param seed: stable compartment-specific colour offset.
    :returns: ``(N, 3)`` uint8 RGB array in the same order as ``labels``.
    """
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if values.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    phase = (int(seed) % 997) / 997.0
    hues = np.mod(values * 0.618033988749895 + phase, 1.0)
    saturations = 0.72 + 0.25 * np.mod(values * 37 + seed, 101) / 100.0
    brightness = 0.86 + 0.13 * np.mod(values * 53 + seed, 97) / 96.0
    colours = [
        colorsys.hsv_to_rgb(float(hue), float(saturation), float(value))
        for hue, saturation, value in zip(hues, saturations, brightness)
    ]
    return np.rint(np.asarray(colours) * 255.0).astype(np.uint8)


#: Random-colour generator for the ``auto`` outline mode. Module level so a
#: test can seed it; unseeded it draws from the OS entropy pool, which is what
#: makes two preview runs come out in two different colours.
_AUTO_COLOUR_RNG = random.Random()


def safe_outline_palette() -> Optional[List[Tuple[int, int, int]]]:
    """Colours ``auto`` may draw from, or ``None`` when any colour will do.

    A random hue is right for a sighted user and exactly wrong for a
    colour-blind one: uniform over the circle, it will sooner or later hand
    two adjacent compartments a pair that user cannot tell apart, and the
    outlines are the one thing on the screen whose whole job is to be told
    apart. When a colour-vision mode is set, ``auto`` draws from the
    Okabe-Ito set instead -- eight colours chosen to stay distinct under all
    three deficiencies.

    :returns: RGB triples, or ``None`` when the preference is ``off``.
    """
    try:
        from ..preferences import (color_blind_categorical_palette,
                                   get_color_blind_mode)
        if get_color_blind_mode() == "off":
            return None
        hexes = color_blind_categorical_palette()
    except Exception:
        return None
    out: List[Tuple[int, int, int]] = []
    for value in hexes:
        text = str(value).lstrip("#")
        if len(text) != 6:
            continue
        try:
            out.append((int(text[0:2], 16), int(text[2:4], 16),
                        int(text[4:6], 16)))
        except ValueError:
            continue
    return out or None


def random_outline_colour(rng: Optional[random.Random] = None,
                          palette: Optional[Sequence[Tuple[int, int, int]]] = None
                          ) -> Tuple[int, int, int]:
    """Return one vivid random RGB triple for the ``auto`` outline mode.

    Hue is uniform over the full circle while saturation and value stay high,
    so the colour is always legible on top of a micrograph — a uniform draw in
    RGB would regularly produce muddy near-grey outlines nobody can see.

    :param rng: optional generator, for reproducible tests.
    :param palette: draw from these instead of the hue circle. This is how
        :func:`safe_outline_palette` reaches the ``auto`` mode.
    :returns: ``(r, g, b)`` in 0..255.
    """
    source = rng if rng is not None else _AUTO_COLOUR_RNG
    if palette:
        return tuple(source.choice(list(palette)))
    hue = source.random()
    saturation = 0.70 + 0.30 * source.random()
    value = 0.85 + 0.15 * source.random()
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(round(red * 255)), int(round(green * 255)),
            int(round(blue * 255)))


def overlay_masks(image: np.ndarray,
                    masks: Dict[str, np.ndarray],
                    outline_rgb: Optional[Tuple[int, int, int]] = None,
                    outline_thickness: int = 1,
                    normalise: bool = True,
                    lo_pct: float = 2.0,
                    hi_pct: float = 98.0,
                    random_outline: bool = False,
                    outline_colors: Optional[
                        Dict[str, Tuple[int, int, int]]] = None,
                    primaries: str = "rgb") -> np.ndarray:
    """Return an RGB uint8 view of ``image`` with every mask's boundary
    drawn in the object's colour (or ``outline_rgb`` when supplied).

    :param image: (H, W) or (H, W, C) source image.
    :param masks: ``{object_type: label_array}`` — one entry per object
        type currently visible on the panel.
    :param outline_rgb: overrides the per-object colour when the user
        picks a global outline colour from the toolbar.
    :param outline_thickness: number of pixels the boundary is dilated
        by (1 = crisp, 3 = highlighter). Tops out at 5.
    :param normalise: forwarded to :func:`_to_uint8`.
    :param random_outline: assign every positive object label a vivid,
        stable categorical colour. This takes precedence over
        ``outline_rgb`` and corresponds to ``color (random)`` in Mask Live.
    :param outline_colors: per-compartment colour overrides used when no
        global ``outline_rgb`` is given. This is how the panel's ``auto``
        mode reaches the renderer: it holds one random colour per
        compartment for the current run. Falls back to
        :data:`OBJECT_COLORS` for anything it does not name.
    :param primaries: one of :data:`spacr.crops.DISPLAY_PRIMARIES`. Applied
        to the IMAGE ONLY, before a single outline is drawn.

    WHY THE ORDER MATTERS, and it is the whole reason this parameter is here
    rather than in :func:`numpy_to_qpixmap`. The primaries are a
    channel-to-colour mapping and only channels belong in it. An outline is
    not a channel -- it is a colour the user chose, or a categorical label
    -- so putting it through the same matrix would answer a request for a
    red outline with a yellow one. Recolour the image, then draw on top.
    """
    base = _to_uint8(image, normalise=normalise,
                        lo_pct=lo_pct, hi_pct=hi_pct)
    if base.ndim == 2:
        rgb = np.stack([base, base, base], axis=-1)
    else:
        rgb = base[..., :3].copy()
    if str(primaries or "rgb").lower() != "rgb":
        from ...crops import apply_display_primaries
        rgb = np.ascontiguousarray(apply_display_primaries(rgb, primaries))
    outline_thickness = max(1, min(5, int(outline_thickness)))
    for obj_type, mask in masks.items():
        if mask is None:
            continue
        mask = np.asarray(mask)
        if mask.ndim != 2 or mask.shape != rgb.shape[:2]:
            LOG.debug("overlay_masks: skipping %s mask %s — image is %s",
                      obj_type, mask.shape, rgb.shape[:2])
            continue
        if not mask.any():
            continue
        if random_outline:
            labelled_boundary = _labelled_boundary(mask, outline_thickness)
            pixels = labelled_boundary > 0
            if not pixels.any():
                continue
            object_labels = np.unique(labelled_boundary[pixels])
            palette = _random_outline_palette(
                object_labels,
                RANDOM_OUTLINE_SEEDS.get(obj_type, 0),
            )
            palette_indices = np.searchsorted(
                object_labels, labelled_boundary[pixels],
            )
            rgb[pixels] = palette[palette_indices]
            continue
        boundary = _boundary_mask(mask.astype(np.int32))
        for _ in range(outline_thickness - 1):
            b2 = boundary.copy()
            b2[1:, :]  |= boundary[:-1, :]
            b2[:-1, :] |= boundary[1:, :]
            b2[:, 1:]  |= boundary[:, :-1]
            b2[:, :-1] |= boundary[:, 1:]
            boundary = b2
        colour = outline_rgb
        if colour is None and outline_colors:
            colour = outline_colors.get(obj_type)
        if colour is None:
            colour = OBJECT_COLORS.get(obj_type, (32, 220, 32))
        rgb[boundary] = np.array(colour, dtype=np.uint8)
    return rgb


def numpy_to_qpixmap(arr: np.ndarray, normalise: bool = True,
                        lo_pct: float = 2.0,
                        hi_pct: float = 98.0) -> QPixmap:
    """Convert an (H, W) or (H, W, C) array to a :class:`QPixmap`.

    The result is always RGB888, so the caller cannot hand Qt a buffer whose
    real row length disagrees with the ``w * 3`` stride below. Channel counts
    other than three are reconciled here — extra channels are dropped, missing
    ones are filled with black — because a mismatch made ``QImage`` read
    ``h * w * 3`` bytes out of a buffer that only held ``h * w``.
    """
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = _to_uint8(arr, normalise=normalise,
                          lo_pct=lo_pct, hi_pct=hi_pct)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    elif arr.shape[-1] < 3:
        pad = np.zeros(arr.shape[:2] + (3 - arr.shape[-1],), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=-1)
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w, _ = arr.shape
    img = QImage(arr.tobytes(), w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())



def first_supported_image(source: Path) -> Optional[Path]:
    """Return the first supported image at or below ``source``.

    Direct image files are returned unchanged. Directory traversal stops as
    soon as the first sorted match is found instead of materialising and
    sorting every image in a potentially enormous plate. Files whose names
    start with a dot are skipped: ``._<name>.tif``, the sidecar macOS writes
    on exFAT and network volumes, sorts before every image and holds none.

    :param source: image path or directory to inspect.
    :returns: the first supported image, or ``None``.
    """
    source = Path(source)
    if source.is_file():
        return source if source.suffix.lower() in SUPPORTED_SUFFIXES else None
    if not source.is_dir():
        return None

    walk_errors: List[OSError] = []
    for folder, dirs, files in os.walk(
            source, topdown=True, onerror=walk_errors.append,
            followlinks=False):
        dirs.sort(key=str.casefold)
        for name in sorted(files, key=str.casefold):
            if name.startswith("."):
                continue
            if Path(name).suffix.lower() in SUPPORTED_SUFFIXES:
                return Path(folder) / name
    if walk_errors:
        raise OSError(
            f"Could not inspect {source}: {walk_errors[0]}")
    return None


def load_source_payload(source, max_sets: int = DEFAULT_MAX_SETS,
                        enumerate_sets: bool = True, *,
                        project: bool = False, known_sets=()) -> Dict[str, Any]:
    """Discover, enumerate and decode one preview source. Data in, data out.

    This is the whole of a preview load, written so it touches **no widget and
    no Qt object** and can therefore be handed straight to
    :class:`spacr.qt.job_runner.JobRunner`. It used to be the ``run`` method of
    a hand-rolled ``QThread`` that emitted two signals, which kept the panel's
    sampler warm by ordering ``enumerated`` before ``loaded``; returning both
    halves in one dict gets the same ordering for free, because the caller
    adopts the enumeration and installs the image in a single GUI-thread call.

    The enumeration reads **file names only**. Decoding reads the selected
    image, plus its channel's z-planes when projection is requested.

    :param source: image file or directory to load a preview from.
    :param max_sets: cap for the sample drawn when ``source`` is a directory.
    :param enumerate_sets: ``False`` skips the folder scan entirely. The FOV
        dropdown hands out a path from a set the sampler already produced, so
        re-scanning for it would burn a full pass over a 98 000-file plate to
        rediscover what is already cached.
    :param project: project the selected channel's z-stack on this worker.
    :param known_sets: cached image sets used when enumeration is skipped.
    :returns: ``{path, array, directory, sets, channels, error}``. ``sets`` is
        ``None`` when no enumeration was done or it failed, which the caller
        reads as "leave the sampler alone".
    """
    out: Dict[str, Any] = {
        "path": None, "array": None, "directory": None,
        "sets": None, "channels": None, "error": "",
    }
    try:
        source = Path(source)
        path = first_supported_image(source)
        if path is not None and enumerate_sets:
            try:
                sets, channels = enumerate_image_sets(
                    path.parent, SUPPORTED_SUFFIXES)
                out["directory"] = str(path.parent)
                out["sets"] = sets
                out["channels"] = channels
                if sets and source.is_dir():
                    picked = sample_image_sets(
                        sets, max_sets,
                        sample_seed(path.parent, len(sets), max_sets))
                    if picked:
                        path = picked[0].path()
            except Exception:
                LOG.exception("Could not enumerate image sets under %s",
                              path.parent)
        out["path"] = path
        out["array"] = load_preview_image(path) if path is not None else None
        if project and path is not None:
            for picked in out["sets"] if out["sets"] is not None else known_sets:
                if Path(picked.directory) != path.parent or picked.z_count <= 1:
                    continue
                channel = next((ch for ch, names in picked.planes.items()
                                if path.name in names), None)
                if channel is not None:
                    try:
                        out["array"] = load_preview_mip(picked.plane_paths(channel))
                    except Exception:
                        LOG.exception("Could not project preview source %s", path)
                    break
    except Exception as exc:
        LOG.exception("Could not load live-preview source %s", source)
        out["error"] = str(exc) or exc.__class__.__name__
    return out


@dataclass
class PreviewRequest:
    """Everything the worker needs to run one segmentation pass.

    Kept as a plain dataclass so tests can construct it directly; the
    panel builds one from its widget state on each Run.
    """
    image:               np.ndarray
    model:               str = "cpsam"
    diameter:            float = 30.0
    flow_threshold:      float = 0.4
    cellprob:            float = 0.0
    channels:            Dict[str, int] = field(default_factory=dict)
    object_types:        Tuple[str, ...] = ("cell",)
    preprocess_settings: Dict[str, Any] = field(default_factory=dict)
    postprocess_settings: Dict[str, Any] = field(default_factory=dict)
    model_note:          str = ""
    cancel:             Event = field(default_factory=Event, repr=False)
    provenance:         Dict[str, Any] = field(default_factory=dict)


class _PreviewWorker(QThread):
    """Runs one (or two) Cellpose passes in the background."""

    finished_masks = Signal(object, str, int)
    flows_ready = Signal(object, int)
    provenance_ready = Signal(object, int)

    def __init__(self, request: PreviewRequest, parent=None, token: int = 0):
        """Prepare the worker.

        :param request: the pass to run, READ ON THE WORKER THREAD rather
            than here -- so it must not be mutated after the worker is
            started; build a new request instead.
        :param parent: parent object; ownership only.
        :param token: the panel's run token at the moment this worker was
            started. It rides back out on both result signals so the panel
            can recognise -- and drop -- a result produced for an image it
            has since replaced. PSF processing stops cooperatively between
            convolutions. Native Cellpose inference runs itself out after
            cancellation and its answer lands as a no-op.
        """
        super().__init__(parent)
        self._request = request
        self.token = int(token)

    def run(self):
        """Segment the request and emit the masks, then the flows.

        The segmenter returns masks alone on the stubbed test path and
        ``(masks, flows)`` in the real one, so both shapes are accepted rather
        than the test path being made to fake a second value.

        A failure is emitted rather than raised: this runs on a worker thread,
        where an exception has nobody to catch it, and the panel needs the
        message to show.
        """
        try:
            res = _segment_multi(self._request)
            if isinstance(res, tuple):
                masks, flows = res
            else:
                masks, flows = res, {}
            _check_preview_cancel(self._request)
            if masks:
                record = deepcopy(self._request.provenance)
                record.update(model=self._request.model,
                              model_note=self._request.model_note)
                self.provenance_ready.emit(record, self.token)
            self.finished_masks.emit(masks, "", self.token)
            self.flows_ready.emit(flows or {}, self.token)
        except Exception as e:
            LOG.info("live-preview segmentation failed: %s", e,
                       exc_info=True)
            self.finished_masks.emit(None, str(e), self.token)


def _classical_organelle_mask(image_2d: np.ndarray, role: str,
                              settings: Dict[str, Any]) -> np.ndarray:
    """Segment one organelle plane the way the RUN would.

    ``organelle_method`` has eight values and only one of them is ``cellpose``.
    The preview ran Cellpose unconditionally, so seven of the eight could not
    be previewed at all -- which is most of the fifty-odd organelle settings
    having no effect on anything the user could see. Reported as "there is no
    way to live preview the organelle settings except for the cellpose model".

    THE PIPELINE'S OWN FUNCTION IS CALLED, not a reimplementation of it.
    `spacr.object._segment_single_image` is what a run dispatches each 2-D
    image to, so a preview that disagrees with the run is a bug in one place
    rather than a difference between two.

    :param role: the organelle slot, e.g. ``organelleb``. Its keys are
        remapped onto the plain ``organelle_`` prefix the pipeline function
        reads, so slot 2 previews with slot 2's settings.
    """
    from ...object import _extract_classical_settings, _segment_single_image
    from ...settings import _set_organelle_defaults

    remapped = dict(settings)
    if role != "organelle":
        for key, value in settings.items():
            if key.startswith(role + "_"):
                remapped["organelle_" + key[len(role) + 1:]] = value
    try:
        _set_organelle_defaults(remapped)
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not fill organelle defaults", exc_info=True)
    classical = _extract_classical_settings(remapped)
    mask = _segment_single_image(image_2d, classical)
    return np.asarray(mask).astype(np.int32)


def _check_preview_cancel(req: PreviewRequest) -> None:
    """Stop between processing stages without interrupting native inference."""
    if req.cancel.is_set():
        from ...cancellation import PipelineCancelled
        raise PipelineCancelled('Preview cancelled')


def _segment_multi(req: PreviewRequest) -> Dict[str, np.ndarray]:
    """Run one Cellpose pass per requested object type.

    Cellpose is lazy-imported here so importing this file cold — as
    unit tests do — does not require a CUDA-capable stack.

    Post-processing (min/max size filter, background removal) is
    applied per-object-type after the model returns, using the
    ``postprocess_settings`` dict on the request.
    """
    from ...psf_pipeline import prepare_psf

    _check_preview_cancel(req)
    plan = prepare_psf(req.preprocess_settings)
    _check_preview_cancel(req)
    model = None
    processed = {}
    req.provenance = {
        'processing': plan.provenance() if plan else {'operation': 'none'},
        'stage': 'loaded preview field, before background and model normalization',
        'normalization': 'field-local Cellpose defaults; classical method specific',
        'illumination': 'no preview illumination correction',
        'input_modified': False,
        'filter_intensity_source': 'original loaded preview field',
        'input_shape': list(req.image.shape),
        'input_dtype': str(req.image.dtype),
        'channels': {},
        'methods': {},
        'preprocess_settings': deepcopy(req.preprocess_settings),
        'diameter': float(req.diameter),
        'flow_threshold': float(req.flow_threshold),
        'cellprob_threshold': float(req.cellprob),
    }

    out: Dict[str, np.ndarray] = {}
    flows_out: Dict[str, np.ndarray] = {}
    for obj in req.object_types:
        _check_preview_cancel(req)
        ch_idx = int(req.channels.get(obj, 0))
        ch_idx = ch_idx % req.image.shape[-1] if req.image.ndim == 3 else 0
        req.provenance['channels'][obj] = ch_idx
        if ch_idx not in processed:
            plane = _select_channel(req.image, ch_idx)
            processed[ch_idx] = (plan.apply(plane[..., None], cancel=req.cancel)[..., 0]
                                 if plan else plane)
        image_2d = processed[ch_idx].copy()

        if req.preprocess_settings.get(f"remove_background_{obj}"):
            bg = float(req.preprocess_settings.get(
                f"{obj}_background",
                req.preprocess_settings.get("background", 100.0)))
            image_2d = image_2d.copy()
            image_2d[image_2d < bg] = 0

        method = str(req.preprocess_settings.get(
            f"{obj}_method",
            req.preprocess_settings.get("organelle_method", "cellpose"))
            or "cellpose").strip().lower()
        req.provenance['methods'][obj] = (
            method if obj.startswith('organelle') else 'cellpose')
        if obj.startswith("organelle") and method != "cellpose":
            out[obj] = _classical_organelle_mask(
                image_2d, obj, req.preprocess_settings)
            continue

        _check_preview_cancel(req)
        if model is None:
            model = preview_cellpose_model(req.model)
        _check_preview_cancel(req)
        result = model.eval(
            image_2d,
            channel_axis=None,
            diameter=float(req.diameter) or None,
            flow_threshold=float(req.flow_threshold),
            cellprob_threshold=float(req.cellprob),
        )
        mask = result[0]
        if isinstance(mask, list):
            mask = mask[0]
        mask = np.asarray(mask).astype(np.int32)

        try:
            flows = result[1]
            flow_rgb = flows[0] if isinstance(flows, (list, tuple)) else flows
            if isinstance(flow_rgb, list):
                flow_rgb = flow_rgb[0]
            flows_out[obj] = np.asarray(flow_rgb)
        except Exception:
            pass

        out[obj] = mask
    _check_preview_cancel(req)
    return out, flows_out


def _select_channel(image: np.ndarray, ch: int) -> np.ndarray:
    """Return a 2-D slice from ``image`` at channel index ``ch``."""
    if image.ndim == 3 and image.shape[-1] > 1:
        return image[..., int(ch) % image.shape[-1]]
    return image.squeeze()


def _apply_size_filter(mask: np.ndarray,
                          settings: Dict[str, Any],
                          obj: str,
                          intensity_img: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply the *same* post-segmentation filters the pipeline uses, so the
    live preview matches a real run.

    Reads the per-compartment area, mean-intensity and border limits — the
    exact keys the compartment panels write — and runs them through
    :func:`spacr.utils._filter_objects`, after the pipeline's perimeter merge
    when enabled. The intensity plane contains the
    original values in the object's own channel. Legacy
    ``{obj}_min_size``/``{obj}_max_size`` are honoured as a fallback. No-ops
    when nothing is set."""
    if not settings or mask is None:
        return mask

    def _num(key, default):
        """One size-filter setting as a number, or the default."""
        v = settings.get(key, default)
        try:
            return type(default)(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    min_area = _num(f"{obj}_min_area", _num(f"{obj}_min_size", 0))
    max_area = _num(f"{obj}_max_area", _num(f"{obj}_max_size", 0))
    perimeter_fraction = _num(f"{obj}_perimeter_fraction", 0.0)
    from spacr.utils import _validated_intensity_bounds
    min_intensity, max_intensity = _validated_intensity_bounds(
        settings.get(f"{obj}_min_intensity"),
        settings.get(f"{obj}_max_intensity"))
    remove_border = bool(settings.get(f"{obj}_remove_border_objects", False))
    if obj.startswith("organelle"):
        remove_border = remove_border or bool(
            settings.get(f"{obj}_remove_border", False))

    if not (min_area > 0 or max_area > 0 or remove_border or perimeter_fraction > 0
            or min_intensity != 0 or max_intensity != 0):
        return mask

    if perimeter_fraction > 0:
        from spacr.utils import _process_single_fov_in_memory
        return _process_single_fov_in_memory(
            mask, intensity_img=intensity_img,
            do_perimeter_merge=True, perimeter_fraction=perimeter_fraction,
            min_area=int(min_area), max_area=int(max_area),
            remove_border_objects=remove_border,
            min_intensity=min_intensity, max_intensity=max_intensity,
        ).astype(mask.dtype)

    from spacr.utils import _filter_objects
    return _filter_objects(
        mask.astype(np.uint16).copy(),
        intensity_img=intensity_img,
        min_area=int(min_area), max_area=int(max_area),
        remove_border=remove_border,
        min_intensity=min_intensity, max_intensity=max_intensity,
    ).astype(mask.dtype)



#: How far the pointer may travel and still count as a click rather than a
#: drag. This view pans with the left button, so without a slop threshold
#: every pan would end in a click.
CLICK_SLOP_PX = 4


class _ZoomView(QGraphicsView):
    """A :class:`QGraphicsView` that emits pixel-hover events + supports
    plain wheel-scroll zoom.

    Two big usability upgrades over the first-cut version:

    * **Wheel = zoom (no Ctrl needed).** Turning the wheel zooms
      centred on the cursor. Shift+wheel scrolls the viewport
      vertically if the user wants scroll behaviour.
    * **Fit-to-height on load + resize.** The image always fills the
      canvas at 100 % zoom until the user starts scrolling, so a small
      preview panel doesn't leave the tile 1-cm tall in the corner.
      Every ``resizeEvent`` re-fits — as the splitter is dragged, the
      image grows to match.

    Zoom is broadcast to a peer view via :meth:`set_peer` so the mask
    canvas mirrors what the original canvas is doing (and vice versa).

    :param parent: parent widget; ownership only.
    """

    hover_pixel = Signal(int, int)
    zoom_changed = Signal(float)
    #: A press-and-release with no drag in between. Distinct from a pan,
    #: which this view already uses the left button for -- so a listener
    #: gets "the user clicked the picture" without stealing dragging.
    clicked = Signal()

    def __init__(self, parent=None):
        """Build the view with its own scene and no peer yet."""
        super().__init__(parent)
        from .image_ruler import ImageRuler

        self.ruler = ImageRuler(self)
        self.ruler.changed.connect(self.viewport().update)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._peer: Optional["_ZoomView"] = None
        self._syncing = False
        self._scale = 1.0
        self._user_zoomed = False
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.horizontalScrollBar().valueChanged.connect(self._mirror_pan)
        self.verticalScrollBar().valueChanged.connect(self._mirror_pan)
        self._picture_name = "picture"
        from .picture_export import install_picture_save

        install_picture_save(self, self.picture, self.picture_name,
                             unless=self._ruler_wants_the_right_button)

    def _ruler_wants_the_right_button(self) -> bool:
        """Whether the ruler is out, in which case right-click clears it.

        Read through the attribute rather than captured, because the mask
        canvas is handed the source canvas's ruler after both are built.
        """
        return bool(getattr(getattr(self, "ruler", None), "active", False))

    def picture(self) -> Optional[QPixmap]:
        """What this view is showing, at the resolution it was rendered at.

        NOT a grab of the widget. The user may be zoomed into a corner of a
        2048-pixel field inside a 300-pixel panel, and the thing they want
        in a figure is the field, not the corner at the size of the panel.

        :returns: the pixmap, or ``None`` while the view is empty.
        """
        if self._pixmap_item is None:
            return None
        pixmap = self._pixmap_item.pixmap()
        return None if pixmap.isNull() else pixmap

    def picture_name(self) -> str:
        """The file name offered when this view's picture is saved."""
        return self._picture_name

    def set_picture_name(self, name: str) -> None:
        """Name what this view is showing, for the save dialog.

        Asked at save time rather than stored in the menu, so a view that
        is Overlay one moment and Flows the next offers the right name.
        """
        self._picture_name = str(name or "picture")

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Show a new image, fitted, and forget any zoom the user had applied.

        :param pixmap: the image to show. The zoom is reset so a new field
            starts at the whole canvas rather than inside the last one's crop.
        """
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.setSceneRect(self._scene.sceneRect())
        self._user_zoomed = False
        self._scale = 1.0
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def set_peer(self, peer: "_ZoomView") -> None:
        """Link this view to another, so the two pan and zoom together.

        :param peer: the view to stay in step with.
        """
        self._peer = peer

    def scale_factor(self) -> float:
        """The view's current zoom.

        :returns: the scale, 1.0 at fit.
        """
        return self._scale

    def reset_zoom(self) -> None:
        """Snap back to fit-in-view (100 % of the container)."""
        self._user_zoomed = False
        self._scale = 1.0
        self.setSceneRect(self._scene.sceneRect())
        self.resetTransform()
        if self._pixmap_item is not None:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)


    def wheelEvent(self, event):
        """Plain wheel = zoom around cursor. Shift+wheel = scroll."""
        if event.modifiers() & Qt.ShiftModifier:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if not delta:
            event.ignore()
            return
        factor = 1.20 if delta > 0 else 1.0 / 1.20
        self._apply_zoom(factor, broadcast=True, position=event.position())
        event.accept()

    def resizeEvent(self, event):
        """Refit the tile whenever the container size changes, unless
        the user has manually zoomed in / out."""
        super().resizeEvent(event)
        if not self._user_zoomed and self._pixmap_item is not None:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def _apply_zoom(self, factor: float, broadcast: bool = False, position=None) -> None:
        """Zoom by ``factor``, optionally taking the twin view with it.

        :param factor: magnification multiplier.
        :param broadcast: copy the finished transform and pan to the peer.
        :param position: viewport cursor position; None uses the view center.
            Guard the complete operation so scrollbar changes cannot feed an
            intermediate transform back from the other canvas.
        """
        if self._syncing:
            return
        from .cursor_zoom import zoom_at_pointer

        self._syncing = True
        try:
            self._user_zoomed = True
            if not zoom_at_pointer(self, factor, position):
                return
            self._scale *= factor
            if broadcast and self._peer is not None:
                peer = self._peer
                peer._syncing = True
                try:
                    peer._user_zoomed = True
                    peer._scale = self._scale
                    peer.setSceneRect(self.sceneRect())
                    peer.setTransform(self.transform())
                    peer.horizontalScrollBar().setValue(self.horizontalScrollBar().value())
                    peer.verticalScrollBar().setValue(self.verticalScrollBar().value())
                finally:
                    peer._syncing = False
                peer.zoom_changed.emit(peer._scale)
        finally:
            self._syncing = False
        self.zoom_changed.emit(self._scale)

    def _mirror_pan(self, _value: int = 0) -> None:
        """Put the peer at the same scroll offset as this view.

        Guarded on THIS view for the same reason ``_apply_zoom`` is: setting
        the flag on the peer would make the peer's own handler a no-op, and
        since assigning to its scroll bars fires that handler, the guard has
        to be on the sender or the two views ping-pong.

        Raw scroll-bar values rather than a mapped scene point: the two
        canvases show the same image at the same scale and the same viewport
        size, so their scroll ranges are identical, and copying the value
        keeps them aligned to the pixel without a round trip through scene
        coordinates.
        """
        peer = self._peer
        if peer is None or self._syncing:
            return
        self._syncing = True
        try:
            peer.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value())
            peer.verticalScrollBar().setValue(
                self.verticalScrollBar().value())
        finally:
            self._syncing = False

    def mousePressEvent(self, event):        # noqa: N802 (Qt naming)
        """Remember where a press started, to tell a click from a drag."""
        if self.ruler.handle(event, self._ruler_point):
            self._press_pos = None
            return
        self._press_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):      # noqa: N802 (Qt naming)
        """Emit :attr:`clicked` when the pointer did not really move.

        The left button already pans this view, so a click cannot be defined as
        "left release" -- that would fire at the end of every drag. A few
        pixels of slop, because a click with a real mouse is rarely exactly
        zero movement.
        """
        start = getattr(self, "_press_pos", None)
        if self.ruler.handle(event, self._ruler_point):
            self._press_pos = None
            return
        super().mouseReleaseEvent(event)
        if start is None:
            return
        moved = (event.position().toPoint() - start).manhattanLength()
        self._press_pos = None
        if moved <= CLICK_SLOP_PX:
            self.clicked.emit()

    def mouseMoveEvent(self, event):
        """Announce which image pixel the pointer is over.

        The point is mapped into SCENE coordinates, so the reported pixel is the
        image's own regardless of zoom or pan.

        :param event: the mouse event.
        """
        if self.ruler.handle(event, self._ruler_point):
            return
        if self._pixmap_item is not None:
            scene_pt = self.mapToScene(event.position().toPoint())
            x = int(scene_pt.x())
            y = int(scene_pt.y())
            self.hover_pixel.emit(x, y)
        super().mouseMoveEvent(event)

    def _ruler_point(self, point):
        """Map a viewport point into image pixels, excluding letterboxing.

        :param point: mouse position in viewport coordinates.
        :returns: image (x, y), or None outside the current image.
        """
        if self._pixmap_item is None or self._pixmap_item.pixmap().isNull():
            return None
        scene_point = self.mapToScene(point.toPoint())
        if not self._pixmap_item.boundingRect().contains(scene_point):
            return None
        return scene_point.x(), scene_point.y()

    def paintEvent(self, event):
        """Draw the pixel ruler above the image using the current view transform.

        :param event: Qt viewport paint event.
        """
        super().paintEvent(event)
        if self.ruler.start is not None:
            painter = QPainter(self.viewport())
            self.ruler.paint(painter, lambda x, y: self.mapFromScene(QPointF(x, y)))
            painter.end()



#: Last resort if `spacr.settings` cannot be reached at all — a stub in
#: sys.modules, a partially-installed tree. A dropdown with nothing in it
#: is a dead end, so there is always something here.
_FALLBACK_MODELS = ("cpsam", "cyto3", "cyto2", "nuclei")


def _is_a_real_model_name(value: str) -> bool:
    """Whether ``value`` names a model spaCR can actually load.

    Two things qualify and nothing else: a retired pre-SAM spelling, which
    Cellpose still resolves to cpsam and which a settings file written years
    ago may hold; and a checkpoint that exists on disk.

    A name that is neither is a typo, and putting it in the combo would let
    the preview run against a model that does not exist.
    """
    name = str(value or "").strip()
    if not name:
        return False
    try:
        import os

        if os.path.isfile(name):
            return True
    except Exception:                                        # noqa: BLE001
        pass
    try:
        from ...settings import _CELLPOSE_ALIASES

        return name in set(_CELLPOSE_ALIASES)
    except Exception:                                        # noqa: BLE001
        return False


#: What a preview segments with when the model it was asked for cannot be
#: loaded. Cellpose 4 ships exactly one stock model and this is it.
_STOCK_MODEL = "cpsam"


#: The app key of the one module whose model is RESOLVED rather than read.
_PLAQUE_MODULE = "analyze_plaques"

#: Modules whose RUN names its model with other keys than Mask's, in the
#: run's own precedence; the first one set wins.
#:
#: ``cellpose_masks`` -- :func:`spacr.spacr_cellpose.identify_masks_finetune`
#: loads ``custom_model`` whenever it is not ``None`` and reads ``model_name``
#: only then. Measured on a built Cellpose Masks screen the form carries both,
#: ``{'model_name': 'cpsam', 'custom_model': None}``, and this panel read only
#: the second, so a user with a custom checkpoint previewed stock cpsam.
_RUN_MODEL_KEYS: Dict[str, Tuple[str, ...]] = {
    "cellpose_masks": ("custom_model", "model_name"),
}


def _model_keys_for(primary: str, module: str = "") -> Tuple[str, ...]:
    """The settings keys that decide ``primary``'s model, in the RUN's order.

    NOT a second opinion about which model to use -- it is the pipeline's own
    precedence, written down where the preview can reach it. The preview used
    to read ``model_name`` and nothing else, which the Mask module does not
    declare at all: a user who chose a checkpoint for the pathogens was shown
    a preview made with stock cpsam and tuned diameter and thresholds against
    it.

    The order, read off the run:

      * ``<object_type>_model_name`` is what
        :func:`spacr.settings._get_object_settings` puts in
        ``object_settings['model_name']``;
      * ``pathogen_model`` OVERRIDES it for pathogens --
        ``spacr/object.py`` lines 696-697 in
        :func:`~spacr.object.generate_cellpose_masks_sam`, and again at
        1069-1070 on the older path. (The ledger cited 769-771; the lines
        have moved, the rule has not.)
      * the bare ``model_name`` comes last, for a panel that serves no
        named module. Mask never sets it.

    The two modules that reach this panel through
    :mod:`spacr.qt.preview_registry` DO NOT read the bare ``model_name`` the
    way this list once assumed they did. ``cellpose_masks`` loads
    ``custom_model`` first (:data:`_RUN_MODEL_KEYS`), and ``analyze_plaques``
    resolves its model instead of reading one
    (:func:`_plaque_model_the_run_would_use`).

    :param primary: the compartment the panel's common controls target.
    :param module: the app key whose run the panel stands in for. ``""``
        and ``"mask"`` both mean Mask's keys.
    :returns: the keys to try, first one SET wins.
    """
    if module in _RUN_MODEL_KEYS:
        return _RUN_MODEL_KEYS[module]
    role = str(primary or "cell")
    if role == "pathogen":
        return ("pathogen_model", "pathogen_model_name", "model_name")
    return (f"{role}_model_name", "model_name")


def _plaque_model_the_run_would_use(
        settings: Optional[Dict[str, Any]]) -> Tuple[str, str, bool]:
    """The checkpoint the plaque RUN would segment with, by its own resolver.

    ``analyze_plaques`` never hands ``model_name`` to Cellpose. It resolves
    ``plaque_model`` -- ``'bundled'`` by default, a :mod:`spacr.model_zoo`
    key, or a path -- through :func:`spacr.submodules._resolve_plaque_model`
    and loads the answer as ``custom_model``. Measured on a built Plaque Assay
    screen the form carries ``plaque_model='bundled'`` AND
    ``model_name='cpsam'``, and this panel seeded the second: a preview on
    stock cpsam, against a run on the plaque model.

    THE RESOLVER IS CALLED, NOT COPIED, with ``fetch=False``. The run
    downloads what is missing and a preview must not. What is not on this
    machine -- a zoo key never downloaded, a bundled pack that is absent, a
    value that is neither a file nor a key, on which the run itself stops --
    comes back as the REQUESTED value with ``here=False``; the pass then
    falls back and says so.

    Importing :mod:`spacr.submodules` costs about 3.5 s cold (torch, cellpose,
    scikit-learn), so the panel calls this off the GUI thread; see
    :meth:`LivePreviewPanel._seed_the_model`.

    :param settings: the plaque settings dict.
    :returns: ``(model, "plaque_model", here)``.
    """
    settings = dict(settings or {})
    try:
        from spacr.submodules import (_requested_plaque_model,
                                      _resolve_plaque_model)
    except Exception:                                        # noqa: BLE001
        LOG.debug("the plaque model resolver could not be imported",
                  exc_info=True)
        return "", "plaque_model", False
    requested = _requested_plaque_model(settings)
    try:
        return (str(_resolve_plaque_model(settings, fetch=False)),
                "plaque_model", True)
    except (FileNotFoundError, ValueError):
        # ModelZooMissing is a FileNotFoundError. ValueError is a value the
        # run cannot resolve either.
        return requested, "plaque_model", False
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not resolve plaque_model=%r", requested,
                  exc_info=True)
        return requested, "plaque_model", False


def _model_the_run_would_use(settings: Optional[Dict[str, Any]],
                            primary: str,
                            module: str = "") -> Tuple[str, str, bool]:
    """Which model the RUN would segment ``primary`` with, which key said so,
    and whether the run could find it here.

    A key that is present but ``None`` does not count. That is not a corner
    case: Mask always carries ``pathogen_model`` and spells "not set" as
    ``None``, and the run tests it with ``is not None`` for exactly that
    reason. Treating the key's presence as an answer would have the preview
    read a model of ``"None"``.

    ``analyze_plaques`` is answered by the plaque run's own resolver rather
    than by a key (:func:`_plaque_model_the_run_would_use`), and that is the
    only answer that can come back ``here=False``. A key read leaves a
    missing checkpoint to :func:`_checkpoint_is_missing`, which can see a
    path; nothing path-shaped can be seen in a zoo key, which is why the
    resolver's verdict is carried out rather than re-derived.

    :param settings: the module's settings dict.
    :param primary: the compartment the preview is tuned for.
    :param module: the app key whose run the panel stands in for.
    :returns: ``(model, key, here)``, or ``("", "", True)`` when no key names
        a model.
    """
    if module == _PLAQUE_MODULE:
        return _plaque_model_the_run_would_use(settings)
    for key in _model_keys_for(primary, module):
        value = (settings or {}).get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text, key, True
    return "", "", True


def _checkpoint_is_missing(model_name: Any) -> bool:
    """Whether ``model_name`` is a checkpoint PATH with no file behind it.

    THE RUN STOPS ON THIS and should:
    :func:`spacr.utils._resolve_cellpose_pretrained` raises
    ``FileNotFoundError`` rather than let Cellpose quietly fall back to the
    stock weights. A PREVIEW must not stop. A zoo model the user has picked
    but not downloaded would turn Run preview into a button that only ever
    shows an error, and the preview is the thing they are looking at while
    deciding whether the settings are right.

    So the preview falls back to :data:`_STOCK_MODEL` and SAYS SO --
    :meth:`LivePreviewPanel._model_for_this_pass` and the provenance clause on
    the status line. A preview whose provenance is unstated is the defect
    this panel was fixed for wearing a different hat.

    The test is the run's own, so the two cannot come to disagree about what
    counts as a path: a separator in it, or a checkpoint suffix.

    :param model_name: the model name or path the user picked.
    :returns: True when it names a file that is not there.
    """
    text = str(model_name or "").strip()
    if not text or os.path.isfile(text):
        return False
    return os.sep in text or text.endswith((".pth", ".pt"))


def _offer_the_run_model(combo: QComboBox, wanted: str,
                         here: bool = True) -> bool:
    """Select the model a run would use, adding it to ``combo`` if needed.

    Added when it can load -- a checkpoint on disk, or a retired pre-SAM
    spelling (:func:`_is_a_real_model_name`) -- AND when it cannot: a
    checkpoint path with no file behind it, or anything the run's resolver
    reported absent (``here=False``). The second is on purpose. It is what
    the run is configured with, and hiding it would put the preview back to
    showing cpsam while saying nothing. The pass falls back and states it; see
    :meth:`LivePreviewPanel._model_for_this_pass`.

    Anything else is a typo, and is not offered.

    Shared by every panel that segments with a run's model, so the rule for
    what is offered cannot differ between them.

    :param combo: the model dropdown.
    :param wanted: the model the run would use.
    :param here: False when the run's resolver could not find it here.
    :returns: True when ``wanted`` is now selected.
    """
    if not wanted:
        return False
    index = combo.findText(wanted)
    if index < 0 and (not here or _is_a_real_model_name(wanted)
                      or _checkpoint_is_missing(wanted)):
        combo.addItem(wanted)
        index = combo.count() - 1
    if index < 0:
        return False
    combo.setCurrentIndex(index)
    return True


def _model_menu():
    """What the Cellpose model combo offers, read from the Cellpose API.

    Delegates to :func:`spacr.settings.cellpose_model_menu`, which asks
    ``cellpose.models`` for its stock list plus any checkpoint the user
    registered, then appends the accepted-but-mapped legacy spellings so a
    saved preview setting still loads.

    Wrapped because this is a *widget*: it must build even when
    ``spacr.settings`` is a stand-in (a test that stubs the descriptions
    table does exactly that). It degrades to the shipped list rather than
    to an empty combo.
    """
    try:
        from ...settings import cellpose_live_model_menu
        menu = tuple(cellpose_live_model_menu())
    except Exception:
        return _FALLBACK_MODELS
    return menu or _FALLBACK_MODELS


def _combo_value(combo: QComboBox) -> str:
    """What a dropdown entry MEANS, rather than what its caption reads.

    Every value-carrying dropdown in this panel is filled through
    :func:`spacr.qt.i18n.set_translatable_items`, which keeps the English
    value in the entry's data so the caption can follow the language. A
    dropdown filled any other way still reads back as its caption, which is
    what a list of file names or model names wants.
    """
    value = combo.currentData()
    if isinstance(value, str) and value:
        return value
    return combo.currentText()


class _AsWritten:
    """A stand-in whose ``currentText()`` is one entry, untranslated.

    :func:`spacr.qt.widgets.preview_controls.selected_channel` interprets a
    channel dropdown by reading its current text, and the text on screen is
    now translated. Handing it the entry as written keeps one definition of
    what ``All channels`` and ``Ch 3`` mean instead of a second copy here.
    """

    __slots__ = ("_text",)

    def __init__(self, text: str) -> None:
        """Stand in for a combo box that reports exactly this text.

        :param text: the entry AS WRITTEN in the catalogue, not as shown on
            screen. That is the whole point: the reader of this interprets
            the text, and the text on screen is translated.
        """
        self._text = str(text)

    def currentText(self) -> str:
        """The text this stand-in reports.

        Named for ``QComboBox``'s API so it can be read by the same code that
        reads a real picker, without that code having to know which it has.
        """
        return self._text


class LivePreviewPanel(LivePreviewContract, QWidget):
    """Interactive segmentation preview — Mask app only.

    The reference implementation of
    :class:`~spacr.qt.widgets.preview_contract.LivePreviewContract`: the
    other three live views wear the same run button, the same cancel
    button and the same words as this one.

    :param parent: parent widget.
    :param threaded: whether the panel's jobs run off the GUI thread. False
        runs each one inline, emitting the same signals in the same order, so
        a test can drive the panel synchronously without the behaviour
        diverging.
    """

    #: Where this preview's section folds and sizes are remembered
    #: (item 471): folds under ``"<key>/<section>"``, sizes under
    #: ``"<key>::sections"``.
    SECTION_KEY = "live_preview"

    preview_ready = Signal(object)

    PREVIEW_SOURCE_HINT = "Load an image first."

    def __init__(self, parent=None, *, threaded: bool = True,
                 module: str = ""):
        """Build the preview panel and arm it to accept dropped images.

        :param parent: parent widget, or ``None``.
        :param threaded: load images on a worker thread. Loads go through
            ``JobRunner`` rather than a hand-rolled thread because that is what
            registers the job with the process-wide run registry, which is the
            only thing the activity spinner watches.
        """
        super().__init__(parent)
        self._image: Optional[np.ndarray] = None
        self._image_path: Optional[Path] = None
        self._masks: Dict[str, np.ndarray] = {}
        self._raw_masks: Dict[str, np.ndarray] = {}
        self._flows: Dict[str, np.ndarray] = {}
        self._processing_provenance: Dict[str, Any] = {}
        self._pending_provenance = None
        self._settings: Dict[str, Any] = {}
        #: The model the masks on screen were actually made with, and the
        #: clause explaining it when that is not the model that was asked
        #: for. Read from the run rather than from the combo: the combo can
        #: be changed after a pass, and the picture would then be captioned
        #: with a model that never touched it.
        self._model_that_ran: str = ""
        self._model_note: str = ""
        #: The model this panel last SEEDED into the combo, or ``None`` when
        #: it has seeded none. It is how the panel tells its own value from
        #: one the user picked: see :meth:`_reseed_the_model_for_the_object`.
        self._model_seeded_to: Optional[str] = None
        #: ``module``: the app key whose RUN this preview stands in for. Mask,
        #: Cellpose Masks and Plaque Assay name the model with different
        #: settings (:func:`_model_the_run_would_use`); ``""`` means Mask's.
        #: Not in the docstring above because AutoAPI merges ``__init__``
        #: prose into the class's pinned, nine-times-translated entry.
        self._module: str = str(module or "")
        #: Models the RUN's resolver could not find on this machine. A zoo
        #: key is not path-shaped, so :func:`_checkpoint_is_missing` cannot
        #: see that one is missing; the resolver's verdict is kept instead.
        self._models_not_here: set = set()
        #: A seed resolving off the GUI thread, as ``(token, settings,
        #: primary, combo text when it started)``; see :meth:`_seed_the_model`.
        self._run_model_pending: Optional[Tuple[int, Dict[str, Any], str,
                                                str]] = None
        self._run_model_token: int = 0
        self._worker: Optional[_PreviewWorker] = None
        self._load_jobs = JobRunner(self, threaded=threaded,
                                    app_key="preview image")
        self._model_jobs = JobRunner(self, threaded=threaded,
                                     app_key="preview model",
                                     user_visible=False)
        self._image_load_token: int = 0
        self._run_token: int = 0
        self._propagate_cb = None
        self._auto_outline_colours: Dict[str, Tuple[int, int, int]] = {}
        self._loading_fov = False
        self._sampler = ImageSetSampler(DEFAULT_MAX_SETS)
        self._mip_enabled = False
        self._table_row = 0
        self._table_col = 0
        #: (row, col) cells the user has shift-selected, in click order.
        #: The last entry is the ACTIVE one — the image live settings apply
        #: to — so a plain click leaves a one-entry selection and the
        #: single-image behaviour is the same code path as the many-image
        #: one rather than a special case beside it.
        self._selected_cells = [(0, 0)]
        self._build_ui()
        self._build_compartment_widgets()
        self.setAcceptDrops(True)
        for _v in (getattr(self, "_src_view", None),
                   getattr(self, "_mask_view", None)):
            if _v is not None:
                _v.setAcceptDrops(False)
        from ..screens.settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)
        self._stow_free_widgets()

    def _stow_free_widgets(self) -> int:
        """Put every free-floating child in the container that never shows.

        A ``QWidget`` parented to this panel but in NO layout occupies
        ``(0, 0, 100, 30)`` -- the top left, exactly where the loaded-path
        label sits. Only ``setVisible(False)`` keeps it off screen, and a
        single stray ``show()`` puts a combo box or a spin box over the path.

        THAT HAS NOW BEEN REPORTED THREE TIMES, and the first two fixes each
        moved ONE widget: the path label's eliding, then ``_fov_box`` and
        ``_channel_box`` into :attr:`_offscreen_controls`. Neither addressed
        the class. Measured on a headless build before this method existed:
        113 direct children, **95 of them in no layout at all** -- 45 spin
        boxes, 21 double spin boxes, 17 toggles and 12 combo boxes, every one
        of them one ``show()`` from the same defect. The third report was a
        black field reading "3", which is a spin box, and there were 66 spin
        boxes it could have been.

        The panel's settings controls are *supposed* to be homeless: they
        belong to the panel so their values outlive the dialog, and
        :class:`LiveSettingsDialog` lays them out only while it is open. So
        the fix is not to lay them out here -- it is to give them somewhere
        to wait that cannot be drawn, which is what
        :attr:`_offscreen_controls` already was for two of them.

        Blunt on purpose: it moves whatever it finds rather than naming the
        widgets, so a control added later is covered without anyone
        remembering this. A child that genuinely wants free geometry must be
        created after this runs, or parented somewhere other than the panel.

        :returns: how many widgets were moved, which is what the test asserts
            on -- it must reach zero on a second call.
        """
        container = getattr(self, "_offscreen_controls", None)
        if container is None:
            return 0
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
        for child in self.findChildren(
                QWidget, options=Qt.FindDirectChildrenOnly):
            if child is container or id(child) in laid_out:
                continue
            if child.isWindow():
                continue
            child.setParent(container)
            moved += 1
        return moved


    _DND_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp",
                 ".gif", ".webp")

    def _dropped_image_path(self, event) -> Optional[str]:
        """Return the first dropped local image path, or None."""
        mime = event.mimeData()
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            p = url.toLocalFile()
            if Path(p).suffix.lower() in self._DND_EXTS:
                return p
        return None

    def dragEnterEvent(self, event):    # noqa: N802 (Qt naming)
        """Accept the drag only if it carries a supported image file."""
        if self._dropped_image_path(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):     # noqa: N802
        """Keep accepting while droppable input stays over the panel.

        :param event: the Qt drag event.
        """
        if self._dropped_image_path(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):         # noqa: N802
        """Load the dropped image into the preview."""
        path = self._dropped_image_path(event)
        if path is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self.load_source_async(path)


    def _build_ui(self):
        """Build every parameter widget and lay out the collapsed panel.

        Every control lives here even though only a subset is shown: the Live
        Settings dialog re-parents them into its own form when it opens and
        hands them back on close, so their values persist across opens. They are
        children of the panel throughout, so nothing is collected while
        re-parented.
        """
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        self._model_box = QComboBox(self)
        self._model_box.addItems(list(_model_menu()))
        self._model_box.currentIndexChanged.connect(
            self._on_model_or_object_changed)

        self._object_box = QComboBox(self)
        set_translatable_items(self._object_box, OBJECT_TYPES)
        self._object_box.currentIndexChanged.connect(
            self._on_model_or_object_changed)

        self._cell_channel = QSpinBox(self); self._cell_channel.setRange(0, 8)
        self._nucleus_channel = QSpinBox(self); self._nucleus_channel.setRange(0, 8)
        self._nucleus_channel.setValue(1)
        self._pathogen_channel = QSpinBox(self)
        self._pathogen_channel.setRange(0, 8); self._pathogen_channel.setValue(2)
        #: One channel per organelle slot, behind the single spinner below.
        self._organelle_channel_values: Dict[str, int] = {}
        self._active_organelle_role = "organelle"
        self._organelle_channel = QSpinBox(self)
        self._organelle_channel.setRange(0, 8); self._organelle_channel.setValue(3)

        self._diameter = QDoubleSpinBox(self)
        self._diameter.setRange(0, 400); self._diameter.setValue(30.0)
        self._diameter.setSuffix(" px")
        self._flow = QDoubleSpinBox(self)
        self._flow.setRange(-1, 100); self._flow.setSingleStep(0.05)
        self._flow.setValue(0.4)
        self._prob = QDoubleSpinBox(self)
        self._prob.setRange(-6, 6); self._prob.setSingleStep(0.1)
        self._prob.setValue(0.0)

        self._normalise_check = Toggle("Normalise", self)
        self._normalise_check.setChecked(True)
        self._normalise_check.toggled.connect(self._refresh_canvases)
        self._lo_pct = QDoubleSpinBox(self)
        self._lo_pct.setDecimals(PERCENTILE_DECIMALS)
        self._lo_pct.setRange(0, 50); self._lo_pct.setValue(2.0)
        self._lo_pct.setSuffix(" %")
        self._lo_pct.setSingleStep(0.01)
        self._lo_pct.valueChanged.connect(self._refresh_canvases)
        self._hi_pct = QDoubleSpinBox(self)
        self._hi_pct.setDecimals(PERCENTILE_DECIMALS)
        self._hi_pct.setRange(50, 100); self._hi_pct.setValue(98.0)
        self._hi_pct.setSuffix(" %")
        self._hi_pct.setSingleStep(0.01)
        self._hi_pct.valueChanged.connect(self._refresh_canvases)

        self._outline_colour = QComboBox(self)
        set_translatable_items(self._outline_colour, OUTLINE_CHOICES)
        self._outline_colour.setCurrentIndex(
            self._outline_colour.findData("color (random)"))
        self._outline_colour.currentIndexChanged.connect(
            self._on_outline_colour_changed)
        self._outline_thickness = QSpinBox(self)
        self._outline_thickness.setRange(1, 5)
        self._outline_thickness.setValue(1)
        self._outline_thickness.valueChanged.connect(
            self._refresh_canvases)

        self._model_box.setToolTip(
            "(str) Cellpose model. Cellpose 4 ships exactly one, 'cpsam'. "
            "cyto3/cyto2/nuclei are kept only so older saved settings still "
            "load — Cellpose removed those weights and resolves all of them "
            "to cpsam, so picking one does not change the segmentation. Of "
            "the parameters that used to differ per model, only diameter "
            "still does anything (the image is rescaled by 30/diameter); "
            "model_type and diam_mean are logged as 'not used in v4.0.1+' "
            "and dropped.")
        self._object_box.setToolTip(
            "(str) Object(s) to segment. 'cell + nucleus' runs both passes.")
        self._cell_channel.setToolTip(
            "(int) Image channel index used for cell segmentation.")
        self._nucleus_channel.setToolTip(
            "(int) Image channel index used for nucleus segmentation.")
        self._pathogen_channel.setToolTip(
            "(int) Image channel index used for pathogen segmentation.")
        self._organelle_channel.setToolTip(
            "(int) Image channel index used for organelle segmentation.")
        self._diameter.setToolTip(DIAMETER_TOOLTIP)
        self._flow.setToolTip(
            "(float) Cellpose flow threshold — higher keeps more masks.")
        self._prob.setToolTip(
            "(float) Cellpose cell-probability threshold — lower keeps more "
            "(dimmer) objects.")
        self._normalise_check.setToolTip(
            "(bool) Percentile-normalise the image for display + segmentation.")
        self._lo_pct.setToolTip(
            "(float, %) Lower percentile for normalisation. Six decimals, so "
            "0.0001 clips only the darkest few pixels of a megapixel field.")
        self._hi_pct.setToolTip(
            "(float, %) Upper percentile for normalisation. Six decimals, so "
            "99.9999 clips a handful of hot pixels where 99.99 clips 400.")
        self._outline_colour.setToolTip(
            "(str) Overlay outline colour. 'auto' uses one colour per "
            "compartment; 'color (random)' gives every segmented object a "
            "different stable categorical colour.")
        self._outline_thickness.setToolTip(
            "(int, px) Overlay outline thickness.")

        for w in (self._model_box, self._object_box,
                    self._cell_channel, self._nucleus_channel,
                    self._diameter, self._flow, self._prob,
                    self._normalise_check, self._lo_pct, self._hi_pct,
                    self._outline_colour, self._outline_thickness):
            w.hide()


        pick_row = QHBoxLayout()
        self._pick_row = pick_row
        self._path_label = QLabel(
            "No preview image loaded — drag & drop an image here to load it",
            self)
        self._path_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._path_label.setMinimumWidth(0)
        #: The path in full. The label shows an elided version sized to
        #: whatever width it actually gets, so the text can never be the thing
        #: that decides the layout; this is what the tooltip and any reader
        #: needs.
        self._path_full = ""
        from .ai_toggle_label import AiToggleLabel
        self._mip_toggle = AiToggleLabel(
            self, text="MIP",
            tooltip="Load an image folder to determine whether z-stacks are available.")
        self._mip_toggle.setEnabled(False)
        self._mip_toggle.toggled.connect(self._on_mip_toggled)
        self._max_images_box = FlatSpinBox(
            self, value=DEFAULT_MAX_IMAGES,
            tooltip=("Maximum images shown at once.\n\n"
                     "Shift-click cells to show several together; "
                     "shift-clicking a row header takes every channel of "
                     "that field, a column header every field of that "
                     "channel. Whatever the selection, this many are "
                     "drawn."))
        self._max_images_box.setMinimum(1)
        self._max_images_box.valueChanged.connect(self._on_max_images_changed)
        self._max_sets_box = FlatSpinBox(self, value=DEFAULT_MAX_SETS,
                                         tooltip=MAX_SETS_TOOLTIP)
        self._max_sets_box.valueChanged.connect(self._on_max_sets_changed)
        self._fov_box = FlatComboBox(
            self,
            tooltip=("Field of view. Lists a random sample of the image sets "
                     "in this folder — set the sample size on its left."))
        self._fov_box.currentIndexChanged.connect(self._on_fov_changed)
        self._channel_box = FlatComboBox(
            self,
            tooltip=("Displayed channel. 'All channels' shows the image as "
                     "stored; picking one shows that plane alone. This is a "
                     "view control — the segmentation channels live in Live "
                     "settings."))
        self._channel_box.currentIndexChanged.connect(
            self._on_display_channel_changed)
        populate_channel_combo(self._channel_box, 0)
        self._localise_channel_combo()
        self._pick_btn = FlatButton("Choose image…", self)
        self._pick_btn.clicked.connect(self._pick_file)
        self._set_table = QTableWidget(0, 0, self)
        install_sorting(self._set_table)
        self._set_table.setObjectName("PreviewSetTable")
        self._set_table.setSelectionBehavior(QTableWidget.SelectItems)
        self._set_table.setSelectionMode(QTableWidget.SingleSelection)
        self._set_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._set_table.verticalHeader().setVisible(True)
        self._set_table.setAlternatingRowColors(True)
        self._set_table.horizontalHeader().setStretchLastSection(True)
        self._set_table.setSizePolicy(QSizePolicy.Expanding,
                                      QSizePolicy.Expanding)
        self._set_table.setToolTip(
            "One row per image set, one column per channel. Click a cell to "
            "show that channel of that field. The number on the left is how "
            "many sets are sampled.")
        self._set_table.cellClicked.connect(self._on_set_cell_clicked)
        self._set_table.horizontalHeader().sectionClicked.connect(
            self._on_channel_header_clicked)
        self._set_table.verticalHeader().sectionClicked.connect(
            self._on_set_header_clicked)

        self._cycle_prev_btn = FlatButton("◀", self)
        self._cycle_prev_btn.setToolTip(
            "Show the previous object's channel. Cycles cell, nucleus, both.")
        self._cycle_prev_btn.clicked.connect(lambda: self._cycle_view(-1))
        self._cycle_next_btn = FlatButton("▶", self)
        self._cycle_next_btn.setToolTip(
            "Show the next object's channel. Cycles cell, nucleus, both.")
        self._cycle_next_btn.clicked.connect(lambda: self._cycle_view(1))
        self._cycle_label = QLabel("", self)
        self._cycle_label.setProperty("i18nSkipText", True)
        self._cycle_label.setToolTip(
            "Which of the segmented objects the source view is showing.")
        #: Roles composited into one view. Empty means the ordinary
        #: single-channel view driven by the channel dropdown.
        self._composite_roles: Tuple[str, ...] = ()
        self._cycle_index = 0

        pick_row.addWidget(self._path_label, 1)
        pick_row.addWidget(self._cycle_prev_btn)
        pick_row.addWidget(self._cycle_label)
        pick_row.addWidget(self._cycle_next_btn)
        pick_row.addWidget(self._mip_toggle)
        for caption, field in ((tr('Images'), self._max_images_box),
                               (tr('Fields'), self._max_sets_box)):
            group = QWidget(self)
            group_layout = QHBoxLayout(group)
            group_layout.setContentsMargins(0, 0, 0, 0)
            group_layout.setSpacing(4)
            group_layout.addWidget(QLabel(caption, group))
            group_layout.addWidget(field)
            pick_row.addWidget(group)
        pick_row.addWidget(self._pick_btn)
        self._offscreen_controls = QWidget(self)
        self._offscreen_controls.setVisible(False)
        _offscreen = QVBoxLayout(self._offscreen_controls)
        _offscreen.setContentsMargins(0, 0, 0, 0)
        self._fov_box.setParent(self._offscreen_controls)
        self._channel_box.setParent(self._offscreen_controls)
        _offscreen.addWidget(self._fov_box)
        _offscreen.addWidget(self._channel_box)
        self._fov_box.setVisible(False)
        self._channel_box.setVisible(False)
        root.addLayout(pick_row)

        from .flow import FlowHost, FlowLayout
        act_host = FlowHost(self)
        act_host.setObjectName("LivePreviewActions")
        act = FlowLayout(act_host, spacing=6)
        self._run_btn = QPushButton(PREVIEW_RUN_TEXT, self)
        self._run_btn.clicked.connect(self.run_preview)
        self._cancel_btn = QPushButton(PREVIEW_CANCEL_TEXT, self)
        self._cancel_btn.setToolTip(
            "Abandon the preview in flight. Cellpose cannot be interrupted, "
            "so the pass finishes in the background and its result is "
            "dropped.")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self.cancel_preview)
        self._live_settings_btn = QPushButton("Live settings…", self)
        self._live_settings_btn.clicked.connect(self.open_live_settings)
        self._view_mode = QComboBox(self)
        set_translatable_items(self._view_mode, VIEW_MODES)
        self._view_mode.setToolTip(
            "Right canvas: outline overlay · label masks · Cellpose flows")
        self._view_mode.currentTextChanged.connect(
            lambda *_: self._refresh_canvases())
        self._status = QLabel("", self)
        self._status.setWordWrap(True)
        act.addWidget(self._run_btn)
        act.addWidget(self._cancel_btn)
        act.addWidget(self._live_settings_btn)
        view_group = QWidget(act_host)
        view_row = QHBoxLayout(view_group)
        view_row.setContentsMargins(0, 0, 0, 0)
        view_row.setSpacing(6)
        view_row.addWidget(QLabel("View:", view_group))
        view_row.addWidget(self._view_mode)
        act.addWidget(view_group)
        from .preview_scale import install_preview_scale
        self._scale_control = install_preview_scale(self, "mask", act)
        root.addWidget(act_host)
        root.addWidget(self._status)

        canvas = QHBoxLayout()
        self._src_view = _ZoomView(self)
        self._src_view.setMinimumHeight(160)
        self._src_view.set_picture_name("field")
        self._mask_view = _ZoomView(self)
        self._mask_view.setMinimumHeight(160)
        self._mask_view.set_picture_name("overlay")
        self._src_view.set_peer(self._mask_view)
        self._mask_view.set_peer(self._src_view)
        self._mask_view.ruler = self._src_view.ruler
        self._src_view.ruler.changed.connect(self._mask_view.viewport().update)
        self._ruler_btn = QPushButton(tr("Ruler"), self)
        self._ruler_btn.setCheckable(True)
        self._ruler_btn.setToolTip(tr(
            "Drag a line on either image to measure its length in image pixels. "
            "Right-click with Ruler selected to clear it. Turn Ruler off to pan."))
        self._ruler_btn.toggled.connect(self._src_view.ruler.set_active)
        act.addWidget(self._ruler_btn)
        self._src_view.hover_pixel.connect(self._on_hover)
        self._mask_view.hover_pixel.connect(self._on_hover)
        canvas.addWidget(self._src_view, 1)
        canvas.addWidget(self._mask_view, 1)
        canvas_host = QWidget(self)
        canvas_host.setLayout(canvas)
        from .collapsible_splitter import CollapsibleSplitter
        self._table_split = CollapsibleSplitter(
            Qt.Vertical, self, persist_key=f"{self.SECTION_KEY}::sections")
        self._table_split.setObjectName('PreviewTableSplit')
        self._sections = {}
        self._sections["Image sets"] = self._table_split.add_section(
            self._set_table, "Image sets", stretch=0, extent=170,
            persist_key=f"{self.SECTION_KEY}/Image sets")
        self._sections["Images"] = self._table_split.add_section(
            canvas_host, "Images", stretch=1, extent=600,
            persist_key=f"{self.SECTION_KEY}/Images")

        info = QWidget(self)
        info_col = QVBoxLayout(info)
        info_col.setContentsMargins(0, 0, 0, 0)
        info_col.setSpacing(2)
        self._hover_label = QLabel("Hover over the image to inspect pixels.",
                                     self)
        self._hover_label.setStyleSheet("color: #ffffff; "
                                            "font-family: monospace;")
        info_col.addWidget(self._hover_label)

        from PySide6.QtWidgets import QSlider
        self._history: list = []
        self._compare_row = QWidget(self)
        comp = QHBoxLayout(self._compare_row)
        comp.setContentsMargins(0, 0, 0, 0)
        comp.addWidget(QLabel("Compare runs", self))
        self._compare_slider = QSlider(Qt.Horizontal, self)
        self._compare_slider.setMinimum(0)
        self._compare_slider.setMaximum(0)
        self._compare_slider.valueChanged.connect(self._on_compare_scrub)
        comp.addWidget(self._compare_slider, 1)
        self._compare_label = QLabel("", self)
        self._compare_label.setStyleSheet("color: #ffffff; font-family: monospace;")
        comp.addWidget(self._compare_label)
        self._compare_row.setVisible(False)
        info_col.addWidget(self._compare_row)
        self._sections["Pixel info"] = self._table_split.add_section(
            info, "Pixel info", stretch=0,
            persist_key=f"{self.SECTION_KEY}/Pixel info")
        root.addWidget(self._table_split, 1)

        self._live_settings_dialog: Optional["LiveSettingsDialog"] = None
        self._on_model_or_object_changed()


    def load_image(self, path):
        """Synchronously load one image.

        Intended for explicit programmatic calls and tests, and for those only.
        **Every** GUI path — the drop handler, the FOV dropdown and the
        Choose-image dialog — goes through :meth:`load_source_async`, so that
        neither the decode nor the folder enumeration behind
        ``_refresh_source_selectors`` can block the application thread. Three of
        them used to call this instead, which is what the docstring already
        claimed was not happening.
        """
        try:
            arr = self._load_for_display(Path(path))
        except Exception as e:
            self._status.setText(f"Load failed: {e}")
            return False
        self._install_loaded_image(Path(path), arr)
        return True

    def _load_for_display(self, path: Path) -> "np.ndarray":
        """Read one plane, or project the field's stack when MIP is on.

        Falls back to the single file whenever the MIP switch is off, the
        folder has no stacks, or the path is not part of a known set — so
        this is the plain reader in every case the projection does not apply
        to.
        """
        if not getattr(self, "_mip_enabled", False):
            return load_preview_image(path)
        picked = None
        try:
            picked = self._sampler.set_for_path(path)
        except Exception:
            picked = None
        if picked is None or picked.z_count <= 1:
            return load_preview_image(path)
        channel = None
        for chan, name in picked.channels.items():
            if name == path.name:
                channel = chan
                break
        for chan, names in picked.planes.items():
            if path.name in names:
                channel = chan
                break
        return load_preview_mip(picked.plane_paths(channel))

    @property
    def _image_loaders(self) -> List[int]:
        """The loads still in flight, as a list so ``not ...`` reads naturally.

        Kept under its historical name because callers and tests wait on it
        going empty. It is now derived from the runner rather than stored, so
        a cancelled load empties it too — a stored list would have to be
        pruned by hand on every exit path and would strand the panel as
        permanently "loading" the one time that was missed.
        """
        runner = getattr(self, "_load_jobs", None)
        return [] if runner is None else [0] * runner.pending_jobs()

    def load_source_async(self, source, *, enumerate_sets: bool = True,
                          display_plane: Optional[int] = None) -> bool:
        """Discover and decode a file/folder source on a worker thread.

        New requests supersede older ones by token. An old decoder is allowed
        to finish safely, but its result is ignored.

        :param source: direct supported image or directory containing images.
        :param enumerate_sets: ``False`` reuses the sampler's cached listing
            instead of re-scanning. See :func:`load_source_payload`.
        :param display_plane: channel plane selected by the table, if any.
        :returns: ``True`` when a worker was started.
        """
        text = os.fspath(source).strip() if source is not None else ""
        if not text:
            return False
        self._image_load_token += 1
        token = self._image_load_token
        max_sets = int(self._sampler.max_sets)
        project = self._mip_enabled
        known_sets = tuple(self._sampler.sets)
        self._load_request = (text, enumerate_sets, display_plane)
        self._status.setText(f"Loading preview from {text}…")
        self._load_jobs.submit(
            lambda: load_source_payload(text, max_sets, enumerate_sets,
                                        project=project, known_sets=known_sets),
            lambda payload, _t=token: self._on_source_payload(
                _t, payload, display_plane=display_plane))
        return True

    def _on_source_payload(self, token: int, payload, *,
                           display_plane: Optional[int] = None) -> None:
        """Apply the newest asynchronous load result. Always on the GUI thread.

        Adopting the enumeration *before* installing the image is what keeps
        the folder scan off this thread: ``_refresh_source_selectors`` asks the
        sampler to enumerate on every single load, and that call is a cache hit
        only because the worker's listing has already landed here.
        """
        if token != self._image_load_token or not isinstance(payload, dict):
            return
        self._load_request = None
        error = payload.get("error") or ""
        if error:
            self._status.setText(f"Load failed: {error}")
            return
        sets = payload.get("sets")
        if sets is not None:
            self._sampler.adopt(payload.get("directory"), sets,
                                payload.get("channels") or [])
        path, arr = payload.get("path"), payload.get("array")
        if path is None or arr is None:
            self._status.setText("No supported preview image found.")
            return
        self._install_loaded_image(Path(path), arr, project=False,
                                   display_plane=display_plane)

    def shutdown(self) -> None:
        """Abandon any load in flight and leave no QThread behind.

        Called from :meth:`closeEvent`, and safe to call directly when a
        screen is torn down without one.
        """
        self.cancel_preview()
        worker = self._worker
        if worker is not None:
            from ..bridge import drain_thread
            worker.setParent(None)
            for signal in (worker.finished_masks, worker.flows_ready,
                           worker.provenance_ready, worker.finished):
                signal.disconnect()
            drain_thread(worker, timeout_ms=0)
            self._worker = None
        for name in ("_load_jobs", "_model_jobs"):
            runner = getattr(self, name, None)
            if runner is not None:
                runner.shutdown()

    def closeEvent(self, event):    # noqa: N802 (Qt naming)
        """Cancel a load in progress rather than let it outlive the panel."""
        self.shutdown()
        super().closeEvent(event)

    def _install_loaded_image(self, path: Path, arr: np.ndarray, *,
                              project: bool = True,
                              display_plane: Optional[int] = None) -> None:
        """Replace preview state with an already-decoded image.

        Synchronous callers can request projection here. Worker results have
        already applied the projection setting and pass ``project=False`` to
        keep decoding off the GUI thread. The selected plane is installed
        before the first repaint.
        """
        if project and getattr(self, "_mip_enabled", False):
            try:
                projected = self._load_for_display(Path(path))
            except Exception:
                projected = None
            if projected is not None:
                arr = projected
        self.cancel_preview()
        self._src_view.ruler.clear()
        self._src_view.ruler.set_spacing()
        self._image = arr
        self._image_path = Path(path)
        self._masks = {}
        self._raw_masks = {}
        self._flows = {}
        self._processing_provenance = {}
        self._pending_provenance = None
        self._model_that_ran = ""
        self._model_note = ""
        self._status.setToolTip("")
        self._path_full = str(path)
        self._show_elided_path()
        self._refresh_source_selectors()
        self._loaded_projection = self._mip_enabled
        if display_plane is not None:
            self._select_display_channel(display_plane)
            self._composite_roles = ()
            self._refresh_cycle_controls()
        note = self.sample_note()
        self._status.setText(f"Loaded {arr.shape} {arr.dtype}"
                             + (f" — {note}" if note else ""))
        self._refresh_canvases()

    def _show_elided_path(self) -> None:
        """Draw the loaded path elided to the width the label actually has.

        ELIDED IN THE MIDDLE, not at the end: the two ends of an image path
        are the parts that identify it -- the plate folder and the file name --
        and a tail-elided path is a column of identical prefixes.

        The full path stays in the tooltip, so nothing is lost, and it is set
        here rather than at load time so the two can never disagree.
        """
        from PySide6.QtCore import Qt as _Qt
        from PySide6.QtGui import QFontMetrics

        full = getattr(self, "_path_full", "") or ""
        if not full:
            return
        self._path_label.setToolTip(full)
        width = max(self._path_label.width(), 80)
        metrics = QFontMetrics(self._path_label.font())
        self._path_label.setText(
            metrics.elidedText(full, _Qt.ElideMiddle, width))

    def resizeEvent(self, event):                            # noqa: N802
        """Re-elide the path when the panel changes width."""
        super().resizeEvent(event)
        try:
            self._show_elided_path()
        except Exception:                                    # noqa: BLE001
            pass


    def _refresh_source_selectors(self) -> None:
        """Re-fill the sets and channel dropdowns for the loaded image.

        The sets dropdown lists a **sample**, not the folder: see
        :class:`~spacr.qt.widgets.preview_controls.ImageSetSampler`. The
        enumeration behind it is cached per folder, so this — which runs on
        every single image load — re-scans nothing once the folder is known.
        """
        if self._image_path is not None:
            meta, custom = self._regex_config()
            self._sampler.enumerate(
                Path(self._image_path).parent, SUPPORTED_SUFFIXES,
                metadata_type=meta, custom_regex=custom)
        self._sample_note = apply_sample_to_combo(
            self._fov_box, self._max_sets_box, self._sampler,
            self._image_path, tooltip="Field of view")
        channels = (int(self._image.shape[2])
                    if self._image is not None and self._image.ndim == 3
                    else 0)
        if channels <= 1:
            try:
                channels = max(channels, len(self._sampler.channels or ()))
            except Exception:
                pass
        canonical = self._channel_box.currentData()
        populate_channel_combo(
            self._channel_box, channels,
            keep=canonical if isinstance(canonical, str) else None)
        self._localise_channel_combo()
        self._populate_set_table()
        self._refresh_mip_toggle()
        QTimer.singleShot(0, self, self._follow_object_channel)

    def sample_note(self) -> str:
        """The sentence stating this preview is a sample of N of M sets."""
        return getattr(self, "_sample_note", "")

    def regroup_the_folder(self) -> bool:
        """Group the loaded folder again, by the naming the form names now.

        The table is grouped when a folder is loaded, with the
        ``metadata_type`` and ``custom_regex`` the Mask form held at that
        moment. Loading first and choosing the naming second left every file
        under one column until something else reloaded the folder. The
        screen calls this when either setting changes. The folder's file
        names are read off the GUI thread, and nothing is decoded.

        :returns: ``True`` when a regrouping was started, ``False`` when no
            image is loaded.
        """
        path = self._image_path
        if path is None:
            return False
        folder = Path(path).parent
        meta, custom = self._regex_config()
        self._regroup_token = getattr(self, "_regroup_token", 0) + 1
        token = self._regroup_token
        self._load_jobs.submit(
            lambda: enumerate_image_sets(folder, SUPPORTED_SUFFIXES,
                                         meta, custom),
            lambda found, _t=token: self._adopt_the_regrouping(
                _t, folder, meta, custom, found))
        return True

    def _adopt_the_regrouping(self, token: int, folder: Path, meta: str,
                              custom, found) -> None:
        """Show a regrouping, unless a newer one or another folder won.

        A grouping read under a naming the form no longer holds is dropped
        too. The screen asks for the next one 400 ms after the naming
        changes, and a job that finishes inside that wait would otherwise be
        adopted, and the selectors refreshed under the new naming would then
        read the folder again on the GUI thread.

        :param token: which :meth:`regroup_the_folder` call produced it.
        :param folder: the folder that was grouped.
        :param meta: the naming dialect it was grouped by.
        :param custom: the custom pattern, or ``None``.
        :param found: ``(sets, channels)`` from
            :func:`~spacr.qt.widgets.preview_controls.enumerate_image_sets`.
        """
        if token != getattr(self, "_regroup_token", 0):
            return
        if self._image_path is None or Path(self._image_path).parent != folder:
            return
        if tuple(self._regex_config()) != (meta, custom):
            return
        sets, channels = found
        self._sampler.adopt(folder, sets, channels,
                            metadata_type=meta, custom_regex=custom)
        self._refresh_source_selectors()
        self._announce_sample()

    def _regex_config(self) -> tuple:
        """The naming dialect the user configured, for grouping their files.

        The preview used to enumerate with the default dialect no matter what
        the module was set to, so a folder whose names only the user's own
        regex understands produced one set per file with no channel at all --
        which is an empty channel list, no field grouping, no z detection, and
        a MIP switch that could never enable. Confirming a regex on import
        then had no effect on the thing standing next to it.

        Read by walking up to the screen that owns the settings widgets, so a
        panel used on its own (or in a test) still works and simply gets the
        defaults.

        :returns: ``(metadata_type, custom_regex or None)``.
        """
        meta, custom = DEFAULT_METADATA_TYPE, None
        widget = self
        for _ in range(12):
            widget = widget.parent() if hasattr(widget, "parent") else None
            if widget is None:
                break
            model = getattr(widget, "_settings_model", None)
            widgets = getattr(model, "_widgets", None) if model else None
            if not widgets:
                continue
            meta = _widget_text(widgets.get("metadata_type")) or meta
            custom = _widget_text(widgets.get("custom_regex")) or None
            break
        return meta, custom

    def _populate_set_table(self) -> None:
        """Fill the table with the sampled sets: a row each, a column per channel.

        Built from the same sample the count field sizes, so the table is a
        readable form of what the dropdown listed rather than a second,
        differently-populated view of the folder. See
        :meth:`_set_table_columns` for which columns there are.
        """
        table = getattr(self, "_set_table", None)
        if table is None:
            return
        try:
            sets = list(self._sampler.sample())
        except Exception:
            sets = []
        pinned = getattr(self, "_pin_path", None)
        if pinned is not None:
            try:
                chosen = self._sampler.set_for_path(pinned)
            except Exception:
                chosen = None
            if chosen is not None and chosen not in sets:
                sets = sorted(sets + [chosen], key=lambda s: s.key)
        columns = self._set_table_columns(sets)
        table.blockSignals(True)
        try:
            table.clear()
            table.setRowCount(len(sets))
            table.setColumnCount(len(columns))
            table.setHorizontalHeaderLabels(
                [caption for caption, _chan, _plane in columns])
            self._column_channels = [
                index if (chan or plane is not None) and caption.startswith("ch ")
                else None
                for index, (caption, chan, plane) in enumerate(columns)]
            for index, (caption, chan, _plane) in enumerate(columns):
                header_item = table.horizontalHeaderItem(index)
                if header_item is not None and chan:
                    header_item.setToolTip(tr(
                        "Channel {index}; the file names call it {name}.",
                        index=index, name=chan))
            table.setVerticalHeaderLabels([s.label for s in sets])
            for row, image_set in enumerate(sets):
                for col, (_caption, chan, plane) in enumerate(columns):
                    name = (image_set.channels.get(chan) if chan is not None
                            else next(iter(image_set.channels.values()), ""))
                    if not name:
                        continue
                    key = chan if chan is not None else next(
                        iter(image_set.channels), None)
                    planes = len(image_set.planes.get(key) or ()) or 1
                    text = name if planes <= 1 else f"{name}  ({planes}z)"
                    item = table_item(text)
                    item.setToolTip(str(image_set.path(key)))
                    item.setData(Qt.UserRole, str(image_set.path(key)))
                    if plane is not None:
                        item.setData(_PLANE_ROLE, int(plane))
                    table.setItem(row, col, item)
            table.resizeColumnsToContents()
            header = table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.Stretch)
        finally:
            table.blockSignals(False)

    def _set_table_columns(self, sets) -> List[Tuple[str, Optional[str],
                                                        Optional[int]]]:
        """The table's columns, as ``(caption, channel ID, plane)``.

        Three sources of channels, in this order:

        * the channel IDs the naming dialect read out of the file names,
          taken from the WHOLE folder rather than from the sample, so a
          channel that only some fields have keeps its column whichever
          fields the sample drew;
        * files the dialect could not read share one column captioned
          "image", not "ch" -- a channel with no number was how every file
          of a folder in another naming came to sit under one column;
        * when no file name carries a channel at all and the loaded image
          holds several planes on its last axis, one column per plane, so a
          folder of multi-channel files is laid out by channel too. A cell
          there opens its file and shows that plane, the same plane the
          channel spin boxes in Live settings number. More planes than those
          spin boxes can name is taken for something other than channels.

        The caption is the channel's index, from 0: the number the
        Cell / Nucleus / Pathogen channel settings
        take, not the ID the file name carries. The pipeline stacks the
        channels in the sorted order of their IDs (``spacr.io``), which is the
        order the columns come in, so column N is channel N; a Yokogawa
        ``C01`` is ``ch 0``. The file's own ID stays in the header's tooltip.

        :param sets: the sampled image sets the rows show.
        :returns: the columns, never empty.
        """
        found = {chan for image_set in sets for chan in image_set.channels}
        try:
            named = set(self._sampler.channels or ())
        except Exception:                                    # noqa: BLE001
            named = set()
        named = sorted((named | found) - {""})
        unread = "" in found
        columns: List[Tuple[str, Optional[str], Optional[int]]] = [
            (f"ch {index}", chan, None) for index, chan in enumerate(named)]
        if named:
            if unread:
                columns.append(("image", "", None))
            return columns
        image = self._image
        planes = (int(image.shape[2])
                  if image is not None and getattr(image, "ndim", 0) == 3
                  else 0)
        if unread and 1 < planes <= int(self._cell_channel.maximum()) + 1:
            return [(f"ch {plane}", "", plane) for plane in range(planes)]
        return [("image", "" if unread else None, None)]

    def max_images(self) -> int:
        """How many images may be drawn at once."""
        try:
            return max(1, int(self._max_images_box.value()))
        except Exception:
            return DEFAULT_MAX_IMAGES

    def _shift_held(self) -> bool:
        """Whether shift is down right now.

        ``cellClicked`` and ``sectionClicked`` carry no modifier, so the
        keyboard is asked directly rather than the table being subclassed to
        intercept the mouse event.
        """
        try:
            from PySide6.QtWidgets import QApplication
            return bool(QApplication.keyboardModifiers() & Qt.ShiftModifier)
        except Exception:
            return False

    def _cells_with_images(self, cells) -> list:
        """Drop cells with no file behind them, keep order, drop duplicates."""
        table, seen, kept = self._set_table, set(), []
        for row, col in cells:
            if (row, col) in seen:
                continue
            item = table.item(row, col)
            if item is None or not item.data(Qt.UserRole):
                continue
            seen.add((row, col))
            kept.append((row, col))
        return kept

    def _set_selection(self, cells, extend: bool) -> None:
        """Replace or extend the shown selection, honouring the image cap.

        Truncation keeps the MOST RECENT cells: a user shift-clicking a fifth
        image with a cap of four means the fifth, not "nothing happened".
        """
        cells = self._cells_with_images(cells)
        if not cells:
            return
        if extend:
            combined = [c for c in self._selected_cells if c not in cells]
            combined.extend(cells)
        else:
            combined = cells
        cap = self.max_images()
        if len(combined) > cap:
            combined = combined[-cap:]
        self._selected_cells = combined
        active_row, active_col = combined[-1]
        self._table_row, self._table_col = active_row, active_col
        self._sync_table_selection()
        item = self._set_table.item(active_row, active_col)
        self._open_cell(item)

    def _sync_table_selection(self) -> None:
        """Show the selection in the table, active cell current."""
        table = self._set_table
        table.blockSignals(True)
        try:
            table.clearSelection()
            for row, col in self._selected_cells:
                item = table.item(row, col)
                if item is not None:
                    item.setSelected(True)
            table.setCurrentCell(self._table_row, self._table_col)
        finally:
            table.blockSignals(False)

    def _on_max_images_changed(self, _value: int) -> None:
        """Re-apply the cap to what is already selected."""
        self._set_selection(list(self._selected_cells), extend=False)

    def _on_channel_header_clicked(self, column: int) -> None:
        """Same field, different channel — the column is the channel.

        With shift, take this channel across every field instead: the column
        IS that channel, so shift-clicking it means "all of these".
        """
        if self._shift_held():
            rows = range(self._set_table.rowCount())
            self._set_selection([(r, column) for r in rows], extend=False)
            return
        self._on_set_cell_clicked(self._table_row, column)

    def _on_set_header_clicked(self, row: int) -> None:
        """Same channel, different field — the row is the field.

        Keeping the column is the point: a user comparing channel 2 across
        fields should not be dropped back to channel 1 by moving down a row.

        With shift, take every channel of this field instead — the row IS
        that field, so shift-clicking it means "all of these".
        """
        if self._shift_held():
            cols = range(self._set_table.columnCount())
            self._set_selection([(row, c) for c in cols], extend=False)
            return
        self._on_set_cell_clicked(row, self._table_col)

    def _on_set_cell_clicked(self, row: int, column: int) -> None:
        """Show the field and channel the user clicked."""
        table = self._set_table
        item = table.item(row, column)
        if item is None:
            return
        path = item.data(Qt.UserRole)
        if not path:
            return
        if self._shift_held():
            self._set_selection([(row, column)], extend=True)
            return
        self._selected_cells = [(row, column)]
        self._table_row, self._table_col = row, column
        table.setCurrentCell(row, column)
        self._fov_box.blockSignals(True)
        try:
            index = self._fov_box.findData(path)
            if index >= 0:
                self._fov_box.setCurrentIndex(index)
        finally:
            self._fov_box.blockSignals(False)
        self._adopt_clicked_channel(column)
        self._open_cell(item)

    def _adopt_clicked_channel(self, column: int) -> bool:
        """Give the chosen object the channel of the column the user clicked.

        Clicking a channel column updates the selected object's channel
        setting so the view stays on that channel. Changing the setting
        follows the same mapping through :meth:`_follow_in_table`.

        Only with ONE object chosen: with "cell + nucleus" there is no single
        setting the click could mean, and the click just shows the channel.
        A column that is not a channel (a file the naming could not read) sets
        nothing.

        :param column: the table column clicked.
        :returns: whether a channel setting was changed.
        """
        channels = getattr(self, "_column_channels", None) or []
        if not (0 <= column < len(channels)) or channels[column] is None:
            return False
        ordered = self._selected_object_types()
        if len(ordered) != 1:
            return False
        role = ordered[0]
        spinner = {"cell": self._cell_channel, "nucleus": self._nucleus_channel,
                   "pathogen": self._pathogen_channel}.get(role)
        if spinner is None and role == getattr(self, "_active_organelle_role", None):
            spinner = self._organelle_channel
        if spinner is None:
            return False
        wanted = int(channels[column])
        if int(spinner.value()) == wanted or wanted > spinner.maximum():
            return False
        spinner.setValue(wanted)
        return True

    def _open_cell(self, item) -> None:
        """Show the file a table cell names, at the plane it names if any.

        A plane column (see :meth:`_set_table_columns`) names one plane of a
        multi-channel file. The file is read only when it is not the one on
        screen already, so moving along a row changes the plane shown and
        reads nothing.

        :param item: the table cell, or ``None``.
        """
        path = item.data(Qt.UserRole) if item is not None else None
        if not path:
            return
        plane = item.data(_PLANE_ROLE)
        if (self._image is None or str(self._image_path) != str(path)
                or getattr(self, "_loaded_projection", False) != self._mip_enabled):
            self.load_source_async(path, enumerate_sets=False,
                                   display_plane=plane)
            return
        self._image_load_token += 1
        self._load_request = None
        note = self.sample_note()
        self._status.setText(f"Loaded {self._image.shape} {self._image.dtype}"
                             + (f" — {note}" if note else ""))
        if plane is None:
            self._refresh_canvases()
            return
        self._select_display_channel(int(plane))
        self._on_display_channel_changed()

    def _refresh_mip_toggle(self) -> None:
        """Enable the MIP switch only where there is a stack to project.

        Says how many planes a field has. This discovery surface has no time
        metadata, so it deliberately makes no claim about a time axis; 4-D
        axis order belongs to the pipeline's explicit ``t_axis_order``.
        """
        toggle = getattr(self, "_mip_toggle", None)
        if toggle is None:
            return
        sets = list(getattr(self._sampler, "sets", None) or ())
        planes = max((s.z_count for s in sets), default=1)
        if planes > 1:
            toggle.setEnabled(True)
            note = (f"Max-intensity projection over {planes} z-planes per "
                    f"field and channel — the same projection the ingest "
                    f"applies before masking.")
            toggle.setToolTip(note)
        else:
            if toggle.isChecked():
                toggle.setChecked(False)
            toggle.setEnabled(False)
            toggle.setToolTip(
                "No z-stacks here — every field has one plane per channel, "
                "so there is nothing to project.")

    def _on_mip_toggled(self, on: bool) -> None:
        """Redraw the current field projected, or as a single plane."""
        self._mip_enabled = bool(on)
        try:
            self._reload_for_mip()
        except Exception:
            pass
        self._announce_sample()

    def _reload_for_mip(self) -> None:
        """Re-read the file on screen under the new projection setting."""
        requested = getattr(self, "_load_request", None)
        if requested is not None:
            path, enumerate_sets, plane = requested
            self.load_source_async(path, enumerate_sets=enumerate_sets,
                                   display_plane=plane)
            return
        path = getattr(self, "_image_path", None)
        if not path:
            return
        self.load_source_async(path, enumerate_sets=False,
                               display_plane=self.display_channel())

    def _on_max_sets_changed(self, value: int) -> None:
        """Draw a new sample at the user's new cap — without re-enumerating."""
        if not self._sampler.set_max(int(value)):
            return
        self._refresh_source_selectors()
        self._announce_sample()

    def _announce_sample(self) -> None:
        """Restate the sample on the status line, where the user is looking."""
        note = self.sample_note()
        if note:
            self._status.setText(note[:1].upper() + note[1:])

    def _on_fov_changed(self, *_args) -> None:
        """Load the field of view the user picked from the dropdown."""
        if self._loading_fov:
            return
        path = self._fov_box.currentData()
        if not path:
            return
        picked = self._sampler.set_for_path(path)
        if picked is not None and picked == self._sampler.set_for_path(
                self._image_path):
            return
        if picked is None and self._image_path is not None \
                and str(self._image_path) == str(path):
            return
        self._loading_fov = True
        try:
            self.load_source_async(path, enumerate_sets=False)
        finally:
            self._loading_fov = False

    def display_channel(self) -> Optional[int]:
        """Channel index the canvases show, or ``None`` for all channels.

        The captions are translated — ``All channels`` reads ``Alla
        kanaler`` on a Swedish screen — so what the shared reader is given
        is the entry as written, kept in the item's data.
        """
        canonical = self._channel_box.currentData()
        if isinstance(canonical, str) and canonical:
            return selected_channel(_AsWritten(canonical))
        return selected_channel(self._channel_box)

    def _localise_channel_combo(self) -> None:
        """Translate the channel dropdown's captions, keeping its entries.

        ``All channels`` is prose a user reads; ``Ch 3`` names a plane and
        stays as written in every language. Both keep the English entry in
        the item's data, which is what :meth:`display_channel` reads.
        """
        box = self._channel_box
        sources = []
        for index in range(box.count()):
            written = box.itemData(index)
            sources.append(written
                           if isinstance(written, str) and written
                           else box.itemText(index))
        chosen = box.currentIndex()
        blocked = box.blockSignals(True)
        try:
            set_translatable_items(
                box, sources, language=getattr(self, "_i18n_language", None))
            if 0 <= chosen < box.count():
                box.setCurrentIndex(chosen)
        finally:
            box.blockSignals(blocked)

    def retranslate_dynamic_content(self, language: str) -> None:
        """Record the language used for subsequently generated panel content.

        Channel choices are rebuilt when a source folder is enumerated, which
        may occur after the standard translation pass. Storing the language
        applied to the widget tree ensures that regenerated choices use the
        current display language even before the preference is persisted.

        :param language: Language code currently applied to the panel.
        """
        self._i18n_language = str(language)

    def _background_for_channel(self, channel: Optional[int]) -> Optional[float]:
        """The background threshold that applies to one displayed channel.

        ``None`` when nothing should be removed from it. The channel is
        matched to an object type the same way the pipeline does it in
        :func:`spacr.io._normalize_img_batch`: a channel is the cell channel
        or the nucleus channel, and it takes that object's background. A
        channel belonging to neither -- a stain the user is only looking at
        -- is left alone, because no background was ever chosen for it.
        """
        if channel is None or not hasattr(self, "_common_widgets"):
            return None
        if not self._widget_value(self._common_widgets["remove_background"]):
            return None
        channels = {"cell": int(self._cell_channel.value()),
                    "nucleus": int(self._nucleus_channel.value()),
                    "pathogen": int(self._pathogen_channel.value()),
                    "organelle": int(self._organelle_channel.value())}
        for obj in self._selected_object_types():
            if channels.get(obj) == int(channel):
                return float(self._widget_value(
                    self._common_widgets["background"]))
        return None

    def _apply_display_background(self, shown):
        """Show the intensity image the segmentation actually ran on.

        Background removal used to happen only inside the worker, so the
        masks moved when it was switched on and the image they were drawn
        over did not -- the one pane that could show you *why* the objects
        changed was the pane still displaying the original pixels.

        The threshold is the same one the worker applies, so this is not a
        second implementation of the rule: both zero everything below
        ``{obj}_background``, and both leave what is above it untouched.
        """
        if shown is None:
            return shown
        channel = self.display_channel()
        if channel is not None:
            background = self._background_for_channel(channel)
            if background is None:
                return shown
            out = shown.copy()
            out[out < background] = 0
            return out

        if getattr(shown, "ndim", 0) != 3:
            return shown
        out = None
        for index in range(shown.shape[2]):
            background = self._background_for_channel(index)
            if background is None:
                continue
            if out is None:
                out = shown.copy()
            plane = out[..., index]
            plane[plane < background] = 0
        return shown if out is None else out

    def _display_image(self) -> Optional[np.ndarray]:
        """The loaded image reduced to the selected display channel."""
        composite = self._composite_view()
        if composite is not None:
            return self._apply_display_background(composite)
        return self._apply_display_background(
            channel_view(self._image, self.display_channel()))

    def _on_display_channel_changed(self, *_args) -> None:
        """Re-render both canvases for the newly selected channel.

        Choosing a channel by hand ends a composite view: the dropdown names
        ONE plane, and leaving the composite up would show something the
        control does not describe.
        """
        if self._composite_roles:
            self._composite_roles = ()
            self._refresh_cycle_controls()
        self._refresh_canvases()

    def set_propagate_callback(self, cb) -> None:
        """Register a callback(dict) used to push tuned live settings back to
        the main settings panel (wired by the AppScreen)."""
        self._propagate_cb = cb

    #: The three segmentation settings, as ``(panel name, Mask suffix)``.
    #:
    #: The panel has ONE diameter / flow / probability triple and an object
    #: selector, while Mask declares all three per compartment
    #: (``cell_diameter``, ``nucleus_flow_threshold``, ...). Which compartment the
    #: triple means is therefore decided by the selector, and this table is
    #: the whole of the translation — used in BOTH directions so the two
    #: cannot drift apart again.
    #:
    #: The bare panel names are real settings for the modules that reach
    #: this panel through :mod:`spacr.qt.preview_registry`
    #: (``cellpose_masks``, ``analyze_plaques``), which have one object type
    #: and call it ``diameter``. Those keep working: a native name present
    #: in the dict wins over the compartment alias.
    #: ``(this panel's own name, the compartment key's suffix)``.
    #:
    #: THE SUFFIXES WENT STALE ON 2026-09-02 and the panel stopped seeding
    #: two of its three spinners. Commit `b7ae412af` renamed the Mask
    #: settings -- `cell_FT` became `cell_flow_threshold` and `cell_CP_prob`
    #: became `cell_cellprob_threshold` -- as a suffix substitution on names
    #: beginning with an underscore, and these two are written WITHOUT one,
    #: so it walked straight past them. `settings_for_propagation` a few
    #: lines up already wrote the new names, so the panel was propagating
    #: `cell_flow_threshold` OUT and reading `cell_FT` back IN: the round
    #: trip the seeding exists for was broken in the middle, and the flow
    #: and cell-probability spinners silently showed their defaults instead
    #: of the values Mask holds.
    _SEGMENTATION_ALIASES: Tuple[Tuple[str, str], ...] = (
        ("diameter", "diameter"),
        ("flow_threshold", "flow_threshold"),
        ("CP_prob", "cellprob_threshold"),
    )

    def settings_for_propagation(self) -> dict:
        """Map the live-preview widget values to main-panel settings keys.

        THE MODEL IS WRITTEN BACK TO THE KEY IT WAS READ FROM. Propagation
        used to write ``model_name`` and ``<primary>_model_name`` only, and
        for pathogens the run reads neither first: ``pathogen_model``
        overrides both when it is set. A user seeded from a
        ``pathogen_model`` checkpoint, switched the live model, and
        propagated, and the run went on using the checkpoint -- the same
        preview/run disagreement as before, pointing the other way.

        Only when the settings the panel holds ALREADY set that key. Writing
        it otherwise would newly switch the override on for a user who never
        asked for it, and ``pathogen_model`` is validated harder than the
        name key (:mod:`spacr.validate` stops a run on a path that is not
        there).
        """
        model = self._model_box.currentText()
        primary = self._primary_object()
        out = {
            "model_name": model,
            f"{primary}_model_name": model,
            "cell_channel": int(self._cell_channel.value()),
            "nucleus_channel": int(self._nucleus_channel.value()),
            "pathogen_channel": int(self._pathogen_channel.value()),
            f"{self._active_organelle_role}_channel": int(
                self._organelle_channel.value()),
            f"{primary}_diameter": self._unclamped(
                self._diameter, float(self._diameter.value())),
            f"{primary}_flow_threshold": self._unclamped(
                self._flow, float(self._flow.value())),
            f"{primary}_cellprob_threshold": self._unclamped(
                self._prob, float(self._prob.value())),
            "normalize": bool(self._normalise_check.isChecked()),
            "lower_percentile": float(self._lo_pct.value()),
        }
        if primary == "pathogen" and self._settings.get(
                "pathogen_model") is not None:
            out["pathogen_model"] = model
        out.update(self._model_write_back(model))
        if hasattr(self, "_compartment_widgets"):
            out.update(self._compartment_settings())
        return out

    def _model_write_back(self, model: str) -> Dict[str, Any]:
        """The model keys a registry module's run reads, written back safely.

        :meth:`settings_for_propagation` writes ``model_name``, and neither
        registry module's run segments with that alone:

        * ``cellpose_masks`` loads ``custom_model`` whenever it is set.
          Leaving it untouched kept the old checkpoint in charge, and writing
          a stock name INTO it stops the run outright -- it prints "Custom
          model not found" and returns. So a checkpoint path is written as
          itself and a stock name CLEARS the override. Only when the settings
          already set it: the rule the pathogen override follows, for the
          same reason.
        * ``analyze_plaques`` resolves ``plaque_model``, which is often
          ``'bundled'`` or a zoo key that this panel shows as the path it
          resolved to. It is written only when the user CHANGED the model to
          a checkpoint file. An untouched preview must not rewrite what a
          recorded run asked for, and the plaque run cannot load a stock name.

        :param model: the combo's current value.
        :returns: the extra keys to propagate, possibly none.
        """
        if self._module == "cellpose_masks":
            if self._settings.get("custom_model") is None:
                return {}
            is_a_checkpoint = (os.path.isfile(model)
                               or _checkpoint_is_missing(model))
            return {"custom_model": model if is_a_checkpoint else None}
        if self._module == _PLAQUE_MODULE:
            if model != self._model_seeded_to and os.path.isfile(model):
                return {"plaque_model": model}
        return {}

    def propagate_settings(self) -> None:
        """Send the current live settings to the main panel (if a callback is
        registered). Called on any live-settings change while the dialog's
        Propagate toggle is on."""
        if self._propagate_cb is not None:
            try:
                self._propagate_cb(self.settings_for_propagation())
            except Exception:
                LOG.debug("propagate_settings failed", exc_info=True)

    def _choose_a_preview_model(self) -> None:
        """Open the model zoo and preview with what the user picks.

        The chosen value is ADDED to the combo when it is not already there:
        a downloaded checkpoint is a path, and the menu only lists what was on
        disk when the panel was built. Selecting an item the combo does not
        hold would otherwise silently do nothing.

        ``kinds`` is a rule rather than a parameter -- the zoo also carries
        the YOLO well detector, and CellposeModel cannot load it, so offering
        it here would produce a preview that fails on selection.
        """
        from .model_zoo_picker import choose_model

        path = choose_model(self, kinds=("cellpose",))
        if not path:
            return
        index = self._model_box.findText(str(path))
        if index < 0:
            self._model_box.addItem(str(path))
            index = self._model_box.count() - 1
        self._model_box.setCurrentIndex(index)

    def apply_settings(self, settings: dict):
        """Seed the panel from a module's settings, and cache the whole dict
        for the Pre / Post routes to read from.

        This is the inverse of :meth:`settings_for_propagation` and is
        tested as one — the defect it was written for is that the two spoke
        different vocabularies. The panel emitted ``cell_diameter`` and read
        back ``diameter``, which Mask does not declare, so a Mask screen
        seeded here kept the panel's own hardcoded 30 px, 0.4 flow and 0.0
        probability while ``cell_channel`` and ``nucleus_channel`` DID land
        — the preview visibly changed and looked seeded, having silently
        dropped exactly the three settings it is opened to check.

        Every field is copied independently. A single unusable value used to
        abort the whole copy through the shared ``except``, so one junk
        diameter also cost the flow threshold, the channels and the model.
        """
        settings = dict(settings or {})
        try:
            self._rebuild_object_choices(organelle_count(settings))
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not rebuild the object choices", exc_info=True)
        self._settings = settings
        for role in organelle_roles(max(1, organelle_count(settings))):
            raw = settings.get(f"{role}_channel")
            if raw is None:
                continue
            try:
                self._organelle_channel_values[role] = int(raw)
            except (TypeError, ValueError):
                LOG.debug("apply_settings: %r is not a channel for %r",
                          raw, role, exc_info=True)
        primary = self._primary_object()

        def _seed(widget, keys, cast):
            """Write the first present, usable value of ``keys``.

            A SPIN BOX CLAMPS WHAT IT CANNOT HOLD, silently. The flow
            threshold runs -1 to 3 here while Mask ships 100 -- "accept
            everything Cellpose proposes" -- so seeding wrote 3, and
            propagating then handed 3 back as if the user had chosen it. The
            value that did not fit is remembered so propagation can return it
            untouched; see :meth:`_unclamped`.
            """
            for key in keys:
                if key not in settings or settings[key] is None:
                    continue
                try:
                    wanted = cast(settings[key])
                    widget.setValue(wanted)
                    held = widget.value()
                    if held != wanted:
                        self._clamped_on_seeding[id(widget)] = (held, wanted)
                    else:
                        self._clamped_on_seeding.pop(id(widget), None)
                except Exception:
                    LOG.debug("apply_settings: %r is not usable for %r",
                              settings[key], key, exc_info=True)
                return

        for native, suffix in self._SEGMENTATION_ALIASES:
            widget = {"diameter": self._diameter,
                      "flow_threshold": self._flow,
                      "CP_prob": self._prob}[native]
            _seed(widget, (native, f"{primary}_{suffix}"), float)

        for comp in COMPARTMENTS:
            key = (f"{self._active_organelle_role}_channel"
                   if comp == "organelle" else f"{comp}_channel")
            _seed(getattr(self, f"_{comp}_channel"), (key,), int)
            self._seed_compartment_widgets(comp)

        self._seed_organelle_column(settings)

        _seed(self._lo_pct, ("lower_percentile",), float)
        if settings.get("adjust_cells") is not None:
            try:
                self._adjust_cells.setChecked(bool(settings["adjust_cells"]))
            except Exception:                                # noqa: BLE001
                LOG.debug("apply_settings: bad adjust_cells", exc_info=True)
        if settings.get("normalize") is not None:
            try:
                self._normalise_check.setChecked(bool(settings["normalize"]))
            except Exception:
                LOG.debug("apply_settings: bad normalize", exc_info=True)
        self._seed_the_model(settings, primary)
        self._recompute_masks()

    def _seed_the_model(self, settings: dict, primary: str) -> None:
        """Select the model the RUN would use for ``primary``.

        THIS READ WAS ``settings.get("model_name")`` AND NOTHING ELSE, and
        Mask does not declare ``model_name``. Measured on a built Mask
        screen, ``_settings_model.collect()`` carries ``cell_model_name``,
        ``nucleus_model_name``, ``organelle_model_name``,
        ``pathogen_model_name`` and ``pathogen_model`` -- and no bare
        ``model_name`` at all. So the combo was never seeded from Mask: a
        user who picked a zoo checkpoint for the pathogens opened the
        preview, saw cpsam's masks, and tuned diameter and thresholds
        against a model the run was not going to use. The preview did not
        fail; it answered a different question and looked authoritative
        doing it.

        :func:`_model_the_run_would_use` holds the key order, taken from the
        run. A checkpoint the combo has never heard of is ADDED rather than
        ignored, the same way :meth:`_choose_a_preview_model` adds one the
        zoo just downloaded.

        PLAQUE ASSAY IS RESOLVED OFF THE GUI THREAD. Its run's resolver lives
        in :mod:`spacr.submodules`, which imports torch, cellpose and
        scikit-learn -- 3.5 s cold, measured, in an app that has not imported
        it by the time this card is first shown. So the answer is computed on
        :attr:`_model_jobs` and adopted when it lands, unless the user has
        picked a model meanwhile. A pass that starts first settles it on the
        spot (:meth:`_settle_the_run_model`) rather than segment with
        whatever the combo held.

        :param settings: the module's settings, as collected from its form.
        :param primary: the compartment the common controls target.
        """
        if self._module != _PLAQUE_MODULE:
            self._select_the_run_model(
                *_model_the_run_would_use(settings, primary, self._module))
            return
        self._run_model_token += 1
        token = self._run_model_token
        snapshot = dict(settings or {})
        self._run_model_pending = (token, snapshot, primary,
                                   self._model_box.currentText())
        self._model_jobs.submit(
            lambda: _model_the_run_would_use(snapshot, primary,
                                             _PLAQUE_MODULE),
            lambda answer, _t=token: self._on_run_model_resolved(_t, answer))

    def _on_run_model_resolved(self, token: int, answer) -> None:
        """Adopt an off-thread answer unless it is stale or overruled.

        :param token: which request this answers; a newer one supersedes it,
            and a pass that settled it first leaves nothing to adopt.
        :param answer: ``(model, key, here)``.
        """
        pending = self._run_model_pending
        if pending is None or pending[0] != token:
            return
        self._run_model_pending = None
        if self._model_box.currentText() != pending[3]:
            return                          # the user picked one meanwhile
        self._select_the_run_model(*answer)

    def _settle_the_run_model(self) -> None:
        """Resolve a pending seed now, on this thread, before a pass needs it.

        The job's own answer then finds nothing pending when it lands.
        """
        pending = self._run_model_pending
        if pending is None:
            return
        self._run_model_pending = None
        _token, snapshot, primary, before = pending
        if self._model_box.currentText() != before:
            return
        self._select_the_run_model(
            *_model_the_run_would_use(snapshot, primary, self._module))

    def _select_the_run_model(self, wanted: str, _key: str = "",
                              here: bool = True) -> None:
        """Put the run's model in the combo and remember that the panel did.

        :param wanted: the model the run would use.
        :param _key: the setting that named it. Unused; it keeps the shape of
            :func:`_model_the_run_would_use`'s answer.
        :param here: False when the run's resolver could not find it here.
        """
        if not wanted:
            return
        if here:
            self._models_not_here.discard(wanted)
        else:
            self._models_not_here.add(wanted)
        if _offer_the_run_model(self._model_box, wanted, here):
            self._model_seeded_to = self._model_box.currentText()

    def _reseed_the_model_for_the_object(self) -> None:
        """Follow the object selector onto that object's model.

        THE MODEL IS A PER-OBJECT SETTING AND THIS PANEL HAS ONE COMBO, so
        without this the pathogen case -- the case this was reported for
        -- never fires: the panel opens on ``cell``, seeds ``cell_model_name``,
        and a user who switches the selector to ``pathogen`` to look at the
        parasites is shown cpsam while the run would use their checkpoint.
        The channel selector already follows the object for the same reason
        (:meth:`_follow_object_channel`).

        A MODEL THE USER PICKED IS NEVER OVERWRITTEN. The panel re-seeds only
        while the combo still holds what the panel itself put there, so
        choosing a checkpoint from the zoo and then flipping the object
        selector does not silently undo the choice -- which would be the very
        defect this follows the object to avoid, committed by the fix for it.
        """
        if not self._settings:
            return
        if (self._model_seeded_to is not None
                and self._model_box.currentText() != self._model_seeded_to):
            return
        self._seed_the_model(self._settings, self._primary_object())

    def current_params(self) -> dict:
        """Snapshot for tests + external callers."""
        return {
            "model": self._model_box.currentText(),
            "diameter": self._diameter.value(),
            "flow_threshold": self._flow.value(),
            "cellprob": self._prob.value(),
            "object_types": self._selected_object_types(),
            "cell_channel": self._cell_channel.value(),
            "nucleus_channel": self._nucleus_channel.value(),
            "normalise": self._normalise_check.isChecked(),
            "lo_pct": float(self._lo_pct.value()),
            "hi_pct": float(self._hi_pct.value()),
            "outline_thickness": self._outline_thickness.value(),
            "outline_colour": self._outline_choice(),
            "display_channel": self.display_channel(),
            "fov": self._fov_box.currentText(),
        }

    def _preview_blocked_reason(self) -> str:
        """Why this panel cannot segment right now, or ``""``."""
        if self._image is None:
            return self.PREVIEW_SOURCE_HINT
        return ""

    def run_preview(self):
        """Segment the loaded image off the GUI thread.

        The guard, the refusals and the busy state are the shared ones —
        see :class:`~spacr.qt.widgets.preview_contract.LivePreviewContract`.
        """
        if not self.begin_preview():
            return
        self._release_worker()
        self._run_token += 1
        req = self._build_request()
        self._status.setText(PREVIEW_RUNNING_MESSAGE)
        worker = _PreviewWorker(req, self, token=self._run_token)
        worker.provenance_ready.connect(self._on_processing_provenance)
        worker.finished_masks.connect(self._on_worker_done)
        worker.flows_ready.connect(self._on_flows_ready)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def cancel_preview(self) -> bool:
        """Cancel PSF work cooperatively and discard any native inference result."""
        worker = getattr(self, '_worker', None)
        if worker is not None:
            worker._request.cancel.set()
        return super().cancel_preview()

    def _on_processing_provenance(self, record, token: int = -1) -> None:
        """Stage captured settings until the matching masks are accepted."""
        if self._stale(token):
            return
        self._pending_provenance = (token, deepcopy(record))

    def _processing_tooltip(self, record) -> str:
        """Explain preview scope and expose the captured scientific settings."""
        note = tr("Preview uses the loaded field and field-local normalization. "
                  "Full Mask runs can use batch normalization and illumination "
                  "correction, so their masks may differ. Intensity filters "
                  "use the original preview pixels.")
        return note + '\n\n' + json.dumps(record, indent=2, default=str)

    def _release_worker(self) -> None:
        """Free the previous worker, whose thread has already finished.

        The worker is parented to the panel, so C++ owns it and it would
        otherwise live — holding a reference to a full-size preview image —
        until the panel itself is destroyed. Unparenting hands ownership back
        to Python, which frees it here, on the thread that holds it. Only ever
        called for a worker that is no longer running, so ``wait`` returns at
        once.
        """
        old = self._worker
        self._worker = None
        if old is None:
            return
        old.wait()
        old.setParent(None)

    def _on_worker_finished(self) -> None:
        """Relay for the worker thread's own ``finished`` signal.

        A bound method on purpose (see :meth:`run_preview`). Returning the
        buttons to the idle state here as well as in :meth:`_on_worker_done`
        is what keeps them usable after a run whose result was discarded as
        stale, or a worker that died without emitting a result at all.
        """
        if not self.preview_running():
            self.set_preview_busy(False)



    def _build_compartment_widgets(self) -> None:
        """Create the common + per-compartment tuning widgets.

        They live on the panel (hidden) so their values persist across opens
        of the Live settings dialog, which re-parents them into its panels and
        hands them back on close — the same pattern the segmentation widgets
        use. Nothing is added to the compact panel layout.

        Populates:
          * ``self._common_widgets`` — signal-to-noise / remove-background /
            background controls that apply to whichever object is chosen.
          * ``self._compartment_widgets[compartment][suffix]`` — the per-
            compartment tuning spinners/checks/combos.
          * ``self._adjust_cells`` — the cell-only "adjust cells" toggle.
        """
        def _spin(kind, spin_args):
            """One spin box of the right kind for this setting."""
            if kind == "float":
                w = QDoubleSpinBox(self)
                lo, hi, dv = spin_args
                w.setRange(float(lo), float(hi)); w.setValue(float(dv))
                w.setDecimals(3)
            elif kind == "int":
                w = QSpinBox(self)
                lo, hi, dv = spin_args
                w.setRange(int(lo), int(hi)); w.setValue(int(dv))
            elif kind == "bool":
                w = Toggle(parent=self)
            else:
                raise ValueError(kind)
            w.hide()
            return w

        try:
            from spacr.settings import descriptions as _spacr_desc
        except Exception:
            _spacr_desc = {}

        self._common_widgets: Dict[str, QWidget] = {
            "signal_to_noise": _spin("int", (0, 100_000, 10)),
            "remove_background": _spin("bool", None),
            "background": _spin("int", (0, 100_000, 100)),
        }
        self._common_widgets["remove_background"].toggled.connect(
            self._refresh_canvases)
        self._common_widgets["background"].valueChanged.connect(
            self._refresh_canvases)
        self._cell_channel.valueChanged.connect(self._on_object_channel_changed)
        self._nucleus_channel.valueChanged.connect(
            self._on_object_channel_changed)
        self._pathogen_channel.valueChanged.connect(
            self._on_object_channel_changed)
        self._organelle_channel.valueChanged.connect(
            self._on_object_channel_changed)
        self._object_box.currentIndexChanged.connect(self._refresh_canvases)
        self._object_box.currentIndexChanged.connect(
            self._on_primary_object_changed)
        self._refresh_cycle_controls()
        for _channel_spinner in (self._cell_channel, self._nucleus_channel,
                                 self._pathogen_channel,
                                 self._organelle_channel):
            _channel_spinner.valueChanged.connect(
                lambda *_: self._recompute_masks())
        self._common_widgets["signal_to_noise"].setToolTip(
            "(int) Signal-to-noise ratio used to set the normalisation "
            "intensity range for the chosen object's channel.")
        self._common_widgets["remove_background"].setToolTip(
            "(bool) Zero every pixel below the background intensity in the "
            "chosen object's channel before segmentation. Applies to the "
            "object selected above — with 'cell + nucleus' chosen, each "
            "channel uses its own background.")
        self._common_widgets["background"].setToolTip(
            "(int) Pixels below this intensity are set to 0 in the chosen "
            "object's channel when 'Remove background' is on. Everything "
            "above it is left where it is.")
        def _organelle_widget(kind, spin_args):
            """The control an organelle setting needs, by its kind."""
            if kind == "morphology":
                widget = QComboBox(self)
                set_translatable_items(widget, list(ORGANELLE_MORPHOLOGIES))
            elif kind == "method_choice":
                widget = QComboBox(self)
            elif kind == "ridge":
                widget = QComboBox(self)
                set_translatable_items(widget,
                                       ["frangi", "sato", "meijering"])
            elif kind == "network":
                widget = QComboBox(self)
                set_translatable_items(widget, ["otsu", "adaptive"])
            elif kind == "ring_fill":
                widget = QComboBox(self)
                set_translatable_items(widget, ["flood", "convex"])
            else:
                return _spin(kind, spin_args)
            widget.hide()
            return widget

        self._organelle_widgets: Dict[str, QWidget] = {}
        for group in ORGANELLE_METHOD_FIELDS.values():
            for suffix, _label, kind, spin_args in group:
                self._organelle_widgets[suffix] = _organelle_widget(
                    kind, spin_args)
                key = f"organelle_{suffix}"
                if key in _spacr_desc:
                    self._organelle_widgets[suffix].setToolTip(
                        str(_spacr_desc[key]))
        self._organelle_widgets["morphology"].currentTextChanged.connect(
            self._on_organelle_morphology_changed)
        self._refresh_organelle_methods()

        self._adjust_cells = _spin("bool", None)
        self._adjust_cells.setToolTip(
            "(bool) Adjust cell masks using the nucleus/pathogen masks.")

        self._compartment_widgets: Dict[str, Dict[str, QWidget]] = {}
        self._compartment_defaults: Dict[str, Dict[str, Any]] = {}
        #: ``id(widget) -> (what it holds, what it was given)`` for a
        #: seeded value the widget could not represent.
        self._clamped_on_seeding: Dict[int, tuple] = {}
        try:
            from spacr.settings import (
                set_default_settings_preprocess_generate_masks as _mask_defaults)
            pipeline_defaults = _mask_defaults({})
        except Exception:                                    # noqa: BLE001
            LOG.debug("live preview: no pipeline defaults", exc_info=True)
            pipeline_defaults = {}

        for comp in COMPARTMENTS:
            group: Dict[str, QWidget] = {}
            for suffix, label, kind, spin_args in COMPARTMENT_FIELDS:
                shipped = pipeline_defaults.get(f"{comp}_{suffix}")
                if shipped is not None and kind in ("int", "float"):
                    spin_args = (spin_args[0], spin_args[1], shipped)
                w = _spin(kind, spin_args)
                if suffix in ("min_intensity", "max_intensity"):
                    w.setDecimals(6)
                    # Return may emit valueChanged even without an edit.
                    # Conversely, typing 0 over a rounded-to-0 seed need
                    # not change the number. Observe actual text edits too.
                    w.valueChanged.connect(
                        lambda *_args, widget=w:
                        self._forget_edited_intensity_seed(widget))
                    w.lineEdit().textEdited.connect(
                        lambda *_args, widget=w:
                        self._forget_edited_intensity_seed(
                            widget, text_edited=True))
                key = f"{comp}_{suffix}"
                desc = _spacr_desc.get(key) or _spacr_desc.get(suffix)
                w.setToolTip(desc if desc else f"{label} for {comp} objects.")
                group[suffix] = w
            self._compartment_widgets[comp] = group
            self._compartment_defaults[comp] = {
                suffix: self._widget_value(widget)
                for suffix, widget in group.items()
            }

        for w in self._all_compartment_widgets():
            for sig_name in ("valueChanged", "currentTextChanged", "toggled"):
                sig = getattr(w, sig_name, None)
                if sig is not None:
                    try:
                        sig.connect(lambda *_: self._recompute_masks())
                    except (TypeError, RuntimeError):
                        pass

    def _forget_edited_intensity_seed(self, widget, *, text_edited=False):
        """Preserve untouched seeds but let explicit edits replace hidden values."""
        remembered = self._clamped_on_seeding.get(id(widget))
        if remembered is None:
            return
        if text_edited or self._widget_value(widget) != remembered[0]:
            self._clamped_on_seeding.pop(id(widget))
            if text_edited:
                # A same-number edit emits no numeric change to trigger the
                # ordinary cached-mask refresh below.
                self._recompute_masks()

    def _all_compartment_widgets(self) -> List[QWidget]:
        """Collect every control the Live Settings dialog manages.

        :returns: the common controls, the cell-adjustment toggle and every
            per-compartment control, in that order.
        """
        ws: List[QWidget] = list(self._common_widgets.values())
        ws.append(self._adjust_cells)
        for group in self._compartment_widgets.values():
            ws.extend(group.values())
        return ws

    def _seed_compartment_widgets(self, comp: str) -> None:
        """Restore the selected role's filters without transient re-filtering.

        Each organelle slot shares one set of controls. Unrepresentable input
        is retained for the shared filter to validate, rather than silently
        replacing an invalid intensity limit with a disabled one.
        """
        role = self._active_organelle_role if comp == "organelle" else comp
        for suffix, widget in self._compartment_widgets[comp].items():
            default = self._compartment_defaults[comp][suffix]
            wanted = self._settings.get(f"{role}_{suffix}", default)
            if wanted is None:
                wanted = default
            if comp == "organelle" and suffix == "remove_border_objects":
                wanted = bool(wanted) or bool(
                    self._settings.get(f"{role}_remove_border", False))
            blocked = widget.blockSignals(True)
            try:
                if isinstance(widget, Toggle):
                    widget.setChecked(bool(wanted))
                else:
                    value = type(widget.value())(wanted)
                    if not np.isfinite(value):
                        raise ValueError("Nonfinite filter setting")
                    widget.setValue(value)
            except (TypeError, ValueError, OverflowError):
                LOG.debug("unrepresentable filter setting %s_%s=%r",
                          role, suffix, wanted)
            finally:
                widget.blockSignals(blocked)
            held = self._widget_value(widget)
            if held != wanted:
                self._clamped_on_seeding[id(widget)] = (held, wanted)
            else:
                self._clamped_on_seeding.pop(id(widget), None)

    def _primary_object(self) -> str:
        """The compartment the common controls target — the first selected."""
        return self._selected_object_types()[0]

    @staticmethod
    def _widget_value(w):
        """Read one control's value in the form the settings dict wants.

        :param w: the control.
        :returns: a ``bool`` for a toggle, a combo box's *value* rather than its
            caption -- a translated caption would land in the settings dict as
            the setting itself -- and otherwise the spin box's number.
        """
        if isinstance(w, Toggle):
            return bool(w.isChecked())
        if isinstance(w, QComboBox):
            return _combo_value(w)
        return w.value()

    #: Keys whose "no limit" the pipeline spells ``None`` rather than 0.
    #: Filled on first use from the module's own defaults.
    _OFF_IS_NONE: Optional[frozenset] = None

    @classmethod
    def _keys_whose_off_is_none(cls) -> frozenset:
        """Setting keys whose shipped disabled value is ``None``.

        Mask's filtering path accepts both 0 and None as no upper limit,
        including for organelles. Preserve the module's shipped spelling
        when propagating a spin box's zero, for compatibility with saved
        settings and other consumers of legacy size limits.
        """
        if cls._OFF_IS_NONE is None:
            try:
                from spacr.settings import (
                    set_default_settings_preprocess_generate_masks as _d)
                shipped = _d({})
            except Exception:                                # noqa: BLE001
                shipped = {}
            cls._OFF_IS_NONE = frozenset(
                key for key, value in shipped.items()
                if value is None and key.endswith(
                    ("_max_area", "_max_size")))
        return cls._OFF_IS_NONE

    def _unclamped(self, widget, value):
        """The value the panel gave, when this widget could not hold it.

        Only while the widget still shows what the clamp left: the moment a
        user moves it, the number on screen is their answer and is what
        propagates.
        """
        remembered = self._clamped_on_seeding.get(id(widget))
        if remembered is None:
            return value
        held, wanted = remembered
        return wanted if value == held else value

    def _off_as_the_run_spells_it(self, key: str, value):
        """Preserve the shipped spelling of a disabled upper limit.

        A spin box represents ``None`` with zero. Writing None back for a
        key that ships it keeps existing settings round trips compatible.
        """
        if value == 0 and key in self._keys_whose_off_is_none():
            return None
        return value

    def _compartment_settings(self) -> dict:
        """Map every compartment + common tuning widget to its setting key."""
        out: dict = {}
        for comp, group in self._compartment_widgets.items():
            prefix = (self._active_organelle_role
                      if comp == "organelle" else comp)
            for suffix, w in group.items():
                key = f"{prefix}_{suffix}"
                out[key] = self._off_as_the_run_spells_it(
                    key, self._unclamped(w, self._widget_value(w)))
            if comp == "organelle":
                out[f"{prefix}_remove_border"] = out[
                    f"{prefix}_remove_border_objects"]
        for obj in self._selected_object_types():
            out[f"{obj}_signal_to_noise"] = self._widget_value(
                self._common_widgets["signal_to_noise"])
            out[f"remove_background_{obj}"] = self._widget_value(
                self._common_widgets["remove_background"])
            out[f"{obj}_background"] = self._widget_value(
                self._common_widgets["background"])
        out["adjust_cells"] = self._widget_value(self._adjust_cells)
        if self._primary_object().startswith("organelle"):
            out.update(self._organelle_settings())
        return out

    def _seed_organelle_column(self, settings: dict) -> None:
        """Fill the organelle column from a module's settings.

        Each widget independently: one unusable value must not cost the rest
        of the column, which is the failure `apply_settings` was rewritten for
        in the first place.
        """
        widgets = getattr(self, "_organelle_widgets", None)
        if not widgets:
            return
        role = self._active_organelle_role
        ordered = ["morphology"] + [k for k in widgets if k != "morphology"]
        for suffix in ordered:
            widget = widgets.get(suffix)
            if widget is None:
                continue
            value = None
            for key in (f"{role}_{suffix}", f"organelle_{suffix}"):
                if settings.get(key) is not None:
                    value = settings[key]
                    break
            if value is None:
                continue
            try:
                if isinstance(widget, QComboBox):
                    index = -1
                    for position in range(widget.count()):
                        written = (widget.itemData(position)
                                   or widget.itemText(position))
                        if str(written) == str(value):
                            index = position
                            break
                    if index >= 0:
                        widget.setCurrentIndex(index)
                elif isinstance(widget, Toggle):
                    widget.setChecked(bool(value))
                else:
                    widget.setValue(type(widget.value())(value))
            except Exception:                                # noqa: BLE001
                LOG.debug("apply_settings: %r is not usable for %s_%s",
                          value, role, suffix, exc_info=True)
            if suffix == "morphology":
                self._refresh_organelle_methods()

    def _organelle_morphology(self) -> str:
        """The morphology selected in the organelle column."""
        widget = getattr(self, "_organelle_widgets", {}).get("morphology")
        value = _combo_value(widget) if widget is not None else ""
        return value if value in ORGANELLE_MORPHOLOGIES else "spots"

    def _refresh_organelle_methods(self) -> None:
        """Offer only the methods this morphology can actually run.

        `spacr.organelle_types.LEGAL_METHODS` is the pipeline's own table, and
        `_segment_single_image` raises for a pairing outside it -- so a method
        the morphology cannot use is not a choice, it is a preview that fails.
        The current selection is kept when it survives the change.
        """
        from ...organelle_types import LEGAL_METHODS

        widget = getattr(self, "_organelle_widgets", {}).get("method")
        if widget is None:
            return
        legal = list(LEGAL_METHODS.get(self._organelle_morphology(), ()))
        if not legal:
            return
        wanted = _combo_value(widget)
        blocked = widget.blockSignals(True)
        try:
            set_translatable_items(
                widget, legal,
                language=getattr(self, "_i18n_language", None))
            index = -1
            for position in range(widget.count()):
                written = (widget.itemData(position)
                           or widget.itemText(position))
                if written == wanted:
                    index = position
                    break
            widget.setCurrentIndex(index if index >= 0 else 0)
        finally:
            widget.blockSignals(blocked)

    def _on_organelle_morphology_changed(self, *_args) -> None:
        """Re-offer the methods, and re-gate which knobs are shown."""
        self._refresh_organelle_methods()
        dialog = getattr(self, "_live_settings_dialog", None)
        if dialog is not None:
            try:
                dialog.refresh_visibility()
            except Exception:                                # noqa: BLE001
                LOG.debug("could not re-gate the organelle column",
                          exc_info=True)

    def _organelle_settings(self) -> dict:
        """The organelle column's values, under the SELECTED slot's prefix.

        Only the knobs the current morphology reads are written. Writing all
        of them would put a ring's settings into a spots run -- harmless to
        the segmentation, which ignores them, but they are then propagated
        into the main panel and saved, where they read as deliberate.
        """
        widgets = getattr(self, "_organelle_widgets", None)
        if not widgets:
            return {}
        role = self._active_organelle_role
        groups = [ORGANELLE_METHOD_FIELDS[None],
                  ORGANELLE_METHOD_FIELDS.get(
                      self._organelle_morphology(), ())]
        out: dict = {}
        for group in groups:
            for suffix, _label, _kind, _args in group:
                widget = widgets.get(suffix)
                if widget is not None:
                    out[f"{role}_{suffix}"] = self._widget_value(widget)
        return out

    def _cycle_stops(self) -> List[Tuple[str, ...]]:
        """The views the arrows step through, in order.

        One stop per object being segmented, then a final stop showing them
        together. With a single object there is nothing to cycle and the
        arrows are hidden rather than left to do nothing.
        """
        roles = self._selected_object_types()
        if len(roles) < 2:
            return []
        return [(role,) for role in roles] + [tuple(roles)]

    def _cycle_view(self, step: int) -> None:
        """Move the source view one stop along, wrapping at both ends."""
        stops = self._cycle_stops()
        if not stops:
            return
        self._cycle_index = (self._cycle_index + int(step)) % len(stops)
        self._apply_cycle_stop()

    def _apply_cycle_stop(self) -> None:
        """Show whatever the current stop names."""
        stops = self._cycle_stops()
        if not stops:
            self._composite_roles = ()
            self._refresh_cycle_controls()
            return
        self._cycle_index %= len(stops)
        roles = stops[self._cycle_index]
        if len(roles) == 1:
            self._composite_roles = ()
            self._select_display_channel(self._channel_for_object(roles[0]))
        else:
            self._composite_roles = tuple(roles)
        self._refresh_cycle_controls()
        self._refresh_canvases()

    def _refresh_cycle_controls(self) -> None:
        """Show the arrows only when there is more than one object, and say
        which object is on screen."""
        stops = self._cycle_stops()
        shown = bool(stops)
        for widget in (self._cycle_prev_btn, self._cycle_next_btn,
                       self._cycle_label):
            widget.setVisible(shown)
        if not shown:
            self._cycle_label.setText("")
            return
        roles = stops[self._cycle_index % len(stops)]
        self._cycle_label.setText(
            tr("both") if len(roles) > 1 else roles[0])

    def _select_display_channel(self, channel: Optional[int]) -> None:
        """Point the channel dropdown at ``channel`` if the image has it."""
        if channel is None:
            return
        box = self._channel_box
        target = f"Ch {channel}"
        for index in range(box.count()):
            written = box.itemData(index)
            if not isinstance(written, str) or not written:
                written = box.itemText(index)
            if written == target and box.currentIndex() != index:
                blocked = box.blockSignals(True)
                try:
                    box.setCurrentIndex(index)
                finally:
                    box.blockSignals(blocked)
                return

    def _composite_view(self) -> Optional[np.ndarray]:
        """The current stop's objects in one image, or ``None``.

        Stacked and handed to :func:`_to_uint8`, which stretches EACH plane on
        its own percentiles and maps the first three onto R/G/B. Two objects
        are ordered so the second lands in red and blue and the first in
        green, which reproduces the outline colours the panel already uses --
        green cells, magenta nuclei -- without a second colour table to keep
        in step with :data:`OBJECT_COLORS`.
        """
        roles = self._composite_roles
        if not roles or self._image is None:
            return None
        planes = []
        for role in roles:
            channel = self._channel_for_object(role)
            planes.append(_select_channel(self._image,
                                          0 if channel is None else channel))
        if len(planes) == 1:
            return planes[0]
        if len(planes) == 2:
            return np.stack([planes[1], planes[0], planes[1]], axis=-1)
        return np.stack(planes[:3], axis=-1)

    def _channel_for_object(self, obj: str) -> Optional[int]:
        """The channel index an object is segmented from, or ``None``.

        Read from the same spinner the run uses, so the view cannot disagree
        with what the segmentation will actually be given.
        """
        if obj.startswith("organelle"):
            if obj == self._active_organelle_role:
                return int(self._organelle_channel.value())
            stored = self._organelle_channel_values.get(obj)
            return None if stored is None else int(stored)
        spinner = {
            "cell": self._cell_channel,
            "nucleus": self._nucleus_channel,
            "pathogen": self._pathogen_channel,
        }.get(obj)
        return None if spinner is None else int(spinner.value())

    def _follow_object_channel(self) -> bool:
        """Show the primary object's own channel.

        :returns: True when the displayed plane was moved (and repainted),
            False when it was already right or there is no one answer.

        Switching the primary object used to leave the displayed plane where
        it was, so picking "cell" while a nucleus plane was up meant tuning
        cell diameter, flow and background against nucleus pixels -- with
        nothing on screen saying so.

        Only the PRIMARY object's channel is followed. With "cell + nucleus"
        both are being segmented and neither is the answer, so the selection
        is left alone rather than made to flicker between the two.

        Signals are blocked around the change and the repaint is issued once,
        explicitly: the channel box is wired to `_refresh_canvases` too, and
        letting both fire repaints the full-size image twice per keystroke
        while a spinner is being typed into.
        """
        ordered = self._selected_object_types()
        if len(ordered) != 1:
            return False
        wanted = self._channel_for_object(ordered[0])
        if wanted is None:
            return False
        if int(wanted) in (getattr(self, "_column_channels", None) or []):
            return self._follow_in_table(int(wanted))
        box = self._channel_box
        target = f"Ch {wanted}"
        for index in range(box.count()):
            written = box.itemData(index)
            if not isinstance(written, str) or not written:
                written = box.itemText(index)
            if written != target:
                continue
            if box.currentIndex() == index:
                return False
            blocked = box.blockSignals(True)
            try:
                box.setCurrentIndex(index)
            finally:
                box.blockSignals(blocked)
            self._refresh_canvases()
            return True
        return False

    def _follow_in_table(self, wanted: int) -> bool:
        """Move the set table to channel ``wanted``'s column, in the same row.

        With cell chosen and cell channel 1, a table showing another
        channel's column switches to channel 1 while staying on the same
        field, so what is on screen is what the
        object will be segmented on.

        :param wanted: the channel index.
        :returns: whether the table moved (and opened that cell).
        """
        table = getattr(self, "_set_table", None)
        channels = getattr(self, "_column_channels", None) or []
        if table is None or wanted not in channels:
            return False
        column = channels.index(wanted)
        row = getattr(self, "_table_row", 0) or 0
        item = table.item(row, column)
        if item is None or not item.data(Qt.UserRole):
            return False
        if column == getattr(self, "_table_col", None):
            plane = item.data(_PLANE_ROLE)
            if (plane is None or self.display_channel() == int(plane)
                    or str(self._image_path) != str(item.data(Qt.UserRole))):
                return False
        self._selected_cells = [(row, column)]
        self._table_row, self._table_col = row, column
        blocked = table.blockSignals(True)
        try:
            table.setCurrentCell(row, column)
        finally:
            table.blockSignals(blocked)
        self._open_cell(item)
        return True

    def _on_object_channel_changed(self, *_args) -> None:
        """Move the view onto the channel the user just typed, then repaint.

        The follow was wired only to a change of WHICH object is primary, so
        setting cell channel to 2 with cell already primary repainted the
        plane that was already on screen: the settings said channel 2 and the
        picture stayed channel 1, and every diameter, flow and background
        judgement from then on was made against the wrong pixels.

        The follow repaints when it moves, so this repaints only when it did
        not -- otherwise the full-size image is redrawn twice for every
        keystroke while a number is being typed into a spinner.
        """
        if not self._follow_object_channel():
            self._refresh_canvases()

    def _selected_object_types(self) -> Tuple[str, ...]:
        """The compartment ROLES selected, not the captions.

        ``organelle 2`` is the caption; ``organelleb`` is the prefix its
        settings keys carry, and every consumer here -- the channel the view
        follows, the keys propagation writes, the compartment the common
        controls retarget -- wants the role.
        """
        current = _combo_value(self._object_box)
        if current == "cell + nucleus":
            return ("cell", "nucleus")
        return (object_role(current),)

    def _rebuild_object_choices(self, count: int) -> None:
        """Offer one organelle entry per slot the main settings declare.

        With `number_of_organelles` at 2 the panel offered a single
        "organelle" entry, so the second slot could not be previewed at all
        and anything tuned for it propagated into the FIRST slot's keys --
        silently re-tuning an organelle the user was not looking at.

        The current selection is kept across the rebuild by role, so raising
        the count does not throw the user back to "cell".
        """
        labels = list(FIXED_OBJECT_TYPES) + [
            organelle_label(n) for n in range(1, max(1, int(count)) + 1)]
        box = self._object_box
        wanted = _combo_value(box)
        blocked = box.blockSignals(True)
        try:
            set_translatable_items(
                box, labels,
                language=getattr(self, "_i18n_language", None))
            index = -1
            for position in range(box.count()):
                written = box.itemData(position) or box.itemText(position)
                if written == wanted:
                    index = position
                    break
            box.setCurrentIndex(index if index >= 0 else 0)
        finally:
            box.blockSignals(blocked)

    def _swap_organelle_channel(self) -> None:
        """Give each organelle slot its own channel behind the one spinner.

        There is a single "Organelle channel" spinner and up to twenty-six
        slots. Without this, moving from `organelle` to `organelleb` carried
        slot 1's channel across and then wrote it into slot 2's settings.
        """
        role = self._selected_object_types()[0]
        previous = getattr(self, "_active_organelle_role", "organelle")
        if previous.startswith("organelle"):
            self._organelle_channel_values[previous] = int(
                self._organelle_channel.value())
        if not role.startswith("organelle"):
            return
        if role != previous:
            for suffix, widget in self._compartment_widgets["organelle"].items():
                key = f"{previous}_{suffix}"
                self._settings[key] = self._off_as_the_run_spells_it(
                    key, self._unclamped(widget, self._widget_value(widget)))
            self._settings[f"{previous}_remove_border"] = self._settings[
                f"{previous}_remove_border_objects"]
        self._active_organelle_role = role
        if role != previous:
            self._seed_compartment_widgets("organelle")
        stored = self._organelle_channel_values.get(role)
        if stored is None or int(stored) == self._organelle_channel.value():
            return
        blocked = self._organelle_channel.blockSignals(True)
        try:
            self._organelle_channel.setValue(int(stored))
        finally:
            self._organelle_channel.blockSignals(blocked)

    def _on_primary_object_changed(self) -> None:
        """Swap the slot's channel in, then move the view onto it. In that
        order: following first would follow the outgoing slot's channel."""
        self._swap_organelle_channel()
        self._cycle_index = 0
        self._composite_roles = ()
        self._follow_object_channel()
        self._reseed_the_model_for_the_object()
        self._refresh_cycle_controls()
        self._recompute_masks()

    def _model_for_this_pass(self) -> Tuple[str, str]:
        """The model this preview will really load, and what to say about it.

        A MODEL THAT IS NOT ON DISK MUST NOT COST THE PREVIEW. The run stops
        on one -- see :func:`_checkpoint_is_missing` -- and a preview that did
        the same would leave the user with an error where the picture goes
        while they are trying to decide whether the settings are right. So
        the pass runs with what is available and the fallback is stated; it
        is never substituted in silence, which is the defect this whole
        mechanism exists to end.

        Missing means a checkpoint path with no file behind it, or a value
        the run's own resolver reported absent -- a zoo key never downloaded,
        a bundled plaque pack that is not installed -- which no path test
        can see.

        The note is built from the existing ``missing`` catalogue row, so it
        reads in the user's language without adding a caption; see
        :meth:`_model_provenance`.

        :returns: ``(model, note)``. ``note`` is empty when the model that
            loads is the model that was asked for.
        """
        self._settle_the_run_model()
        chosen = str(self._model_box.currentText() or "").strip()
        if (chosen not in self._models_not_here
                and not _checkpoint_is_missing(chosen)):
            return chosen, ""
        return _STOCK_MODEL, f"{chosen}: {tr('missing')}"

    def _model_provenance(self) -> str:
        """One clause naming the model that made the picture on screen.

        Read from the last pass rather than from the combo: changing the
        combo does not re-segment, so captioning the masks with the current
        selection would name a model that never touched them.

        COMPOSED FROM CATALOGUE SOURCES THAT ALREADY EXIST, and that is a
        constraint rather than a preference. A new literal caption anywhere
        under ``spacr/qt`` enters the generated i18n layer, whose inventory
        is pinned by count AND digest in
        ``tests/qt/test_i18n_caption_ratchet.py``; adding one means
        regenerating nine locale catalogues. ``Model`` (a term row) and
        ``missing`` (a reviewed compact row the AI panel's status already
        uses) cover the whole clause, so it reads in every locale --
        ``Modell: cpsam — toxoplasma_plaque_v1: saknas.`` -- where the note
        used to say "is not on this machine" in English whatever the
        language.

        :returns: the clause, e.g. ``Model: cpsam.``, or
            ``Model: cpsam — <requested>: missing.`` after a fallback.
        """
        methods = self._processing_provenance.get('methods', {})
        if methods and 'cellpose' not in methods.values():
            return tr('Segmentation: {methods}.',
                      methods=', '.join(sorted(set(methods.values()))))
        model = self._model_that_ran or self._model_box.currentText()
        label = tr("Model")
        if self._model_note:
            return f"{label}: {model} \u2014 {self._model_note}."
        return f"{label}: {model}."

    def _build_request(self) -> PreviewRequest:
        """Assemble a preview request from the current controls.

        One merged settings dict drives both background subtraction and
        filtering: the settings apply wherever they are set, rather than behind
        separate pre/post switches.

        :returns: the request to hand the preview worker.
        """
        obj_types = self._selected_object_types()
        roles = (*COMPARTMENTS, *self._organelle_channel_values, *obj_types)
        channels = {
            role: self._obj_channel(role) for role in roles
        }
        merged = dict(self._settings)
        if hasattr(self, "_compartment_widgets"):
            merged.update(self._compartment_settings())
        pre = deepcopy(merged)
        post = pre
        model, note = self._model_for_this_pass()
        return PreviewRequest(
            image=self._image,
            model=model,
            model_note=note,
            diameter=self._diameter.value(),
            flow_threshold=self._flow.value(),
            cellprob=self._prob.value(),
            channels=channels,
            object_types=obj_types,
            preprocess_settings=pre,
            postprocess_settings=post,
        )

    #: Fixed colours the outline-colour combo offers by name.
    OUTLINE_COLOURS: Dict[str, Tuple[int, int, int]] = {
        "green":   (32, 220, 32),
        "magenta": (222, 82, 200),
        "yellow":  (255, 220, 32),
        "cyan":    (32, 200, 220),
        "white":   (240, 240, 240),
        "red":     (240, 60, 60),
    }

    def _outline_rgb(self) -> Optional[Tuple[int, int, int]]:
        """Translate the outline-colour combo choice into an RGB tuple,
        or ``None`` for ``auto`` and ``color (random)``. ``auto`` is drawn
        from :meth:`_auto_outline_colour` and ``color (random)`` is handled
        per object label by :func:`overlay_masks`."""
        return self.OUTLINE_COLOURS.get(self._outline_choice())

    def _roll_auto_outline_colours(self) -> None:
        """Draw a fresh random colour per compartment for ``auto`` mode.

        ``auto`` used to mean "the compartment's fixed colour", which made
        every cell preview green no matter what — the setting looked stuck.
        It now means a random colour, re-rolled once per preview run so the
        outline stays put while the user tunes thickness or normalisation.

        UNDER A COLOUR-VISION MODE the colours are DEALT, not drawn: one
        compartment per palette entry, without replacement. Drawing
        independently from a safe palette is not enough -- eight safe
        colours still collide by chance, and two compartments sharing one is
        the exact failure the safe palette exists to prevent.
        """
        palette = safe_outline_palette()
        if palette:
            order = list(palette)
            _AUTO_COLOUR_RNG.shuffle(order)
            self._auto_outline_colours = {
                comp: order[i % len(order)]
                for i, comp in enumerate(COMPARTMENTS)}
            return
        self._auto_outline_colours = {
            comp: random_outline_colour() for comp in COMPARTMENTS}

    def _outline_choice(self) -> str:
        """The outline colour the user picked, in the words the code uses."""
        return _combo_value(self._outline_colour)

    def _view_mode_choice(self) -> str:
        """Which canvas the right-hand view shows.

        ``Overlay`` before the control exists: both callers can run from a
        signal a partly-built panel already emits.
        """
        combo = getattr(self, "_view_mode", None)
        return _combo_value(combo) if combo is not None else "Overlay"

    def _auto_outline_colour(self, obj_type: str) -> Tuple[int, int, int]:
        """The current random ``auto`` colour for one compartment."""
        colour = self._auto_outline_colours.get(obj_type)
        if colour is None:
            colour = random_outline_colour(palette=safe_outline_palette())
            self._auto_outline_colours[obj_type] = colour
        return colour

    def _auto_outline_map(self) -> Dict[str, Tuple[int, int, int]]:
        """Per-compartment ``auto`` colours covering everything on screen."""
        if not self._auto_outline_colours:
            self._roll_auto_outline_colours()
        for obj_type in self._masks:
            self._auto_outline_colour(obj_type)
        return dict(self._auto_outline_colours)

    def _on_outline_colour_changed(self, *_args) -> None:
        """Re-render, re-rolling the random colours when ``auto`` is chosen."""
        if self._outline_choice() == "auto":
            self._roll_auto_outline_colours()
        self._refresh_canvases()

    def _refresh_canvases(self):
        """Re-render both views from the current image + masks."""
        if self._image is None:
            return
        norm = self._normalise_check.isChecked()
        lo = float(self._lo_pct.value())
        hi = float(self._hi_pct.value())
        shown = self._display_image()
        src_pix = numpy_to_qpixmap(
            _to_uint8(shown, normalise=norm, lo_pct=lo, hi_pct=hi))
        self._src_view.set_pixmap(src_pix)

        mode = self._view_mode_choice()
        self._mask_view.set_picture_name(str(mode or "overlay").lower())
        if mode == "Flows" and self._flows:
            self._mask_view.set_pixmap(numpy_to_qpixmap(
                self._flows_rgb()))
        elif mode == "Masks" and self._masks:
            self._mask_view.set_pixmap(numpy_to_qpixmap(
                self._label_rgb()))
        elif self._masks:
            overlay = overlay_masks(
                shown, self._masks,
                outline_rgb=self._outline_rgb(),
                outline_thickness=self._outline_thickness.value(),
                normalise=norm, lo_pct=lo, hi_pct=hi,
                random_outline=(
                    self._outline_choice() == "color (random)"
                ),
                outline_colors=self._auto_outline_map(),
                primaries=self.display_primaries())
            self._mask_view.set_pixmap(numpy_to_qpixmap(overlay))
        else:
            self._mask_view.set_pixmap(src_pix)

    def _on_flows_ready(self, flows, token: int = -1) -> None:
        """Store the per-object Cellpose flow RGB images from a preview run."""
        if self._stale(token):
            return
        self._flows = flows or {}
        if self._view_mode_choice() == "Flows":
            self._refresh_canvases()

    def _label_rgb(self) -> np.ndarray:
        """Render the current label masks as a distinct-colour image (0 = black).

        The chosen outline colour tints this view too. It used to be painted
        straight from :data:`OBJECT_COLORS`, so the ``Masks`` view stayed
        green for cells no matter which colour the user picked — the colour
        control simply did not reach this renderer.
        """
        h, w = self._image.shape[:2]
        out = np.zeros((h, w, 3), dtype=np.uint8)
        chosen = self._outline_rgb()
        random_mode = self._outline_choice() == "color (random)"
        auto_colours = self._auto_outline_map()
        for obj, mask in self._masks.items():
            if mask is None or mask.shape[:2] != (h, w):
                continue
            labels = mask.astype(np.int64)
            present = labels > 0
            if not present.any():
                continue
            if random_mode:
                ids = np.unique(labels[present])
                palette = _random_outline_palette(
                    ids, RANDOM_OUTLINE_SEEDS.get(obj, 0))
                out[present] = palette[np.searchsorted(ids, labels[present])]
                continue
            if chosen is not None:
                base_rgb = chosen
            else:
                base_rgb = auto_colours.get(
                    obj, OBJECT_COLORS.get(obj, (200, 200, 200)))
            base = np.array(base_rgb, dtype=np.uint8)
            shade = (0.5 + 0.5 * ((labels % 7) / 6.0)).astype(np.float32)
            for c in range(3):
                out[..., c] = np.where(
                    present,
                    np.clip(base[c] * shade, 0, 255).astype(np.uint8),
                    out[..., c])
        return out

    def _flows_rgb(self) -> np.ndarray:
        """Combine per-object flow RGB images (first available / max-blend)."""
        imgs = [np.asarray(f) for f in self._flows.values()
                if f is not None and np.asarray(f).ndim == 3]
        if not imgs:
            h, w = self._image.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)
        out = imgs[0].astype(np.uint8)
        for f in imgs[1:]:
            if f.shape == out.shape:
                out = np.maximum(out, f.astype(np.uint8))
        return out[..., :3]

    def _on_model_or_object_changed(self, *_):
        """Refresh visibility state — no visible-widget mutation on
        the compact layout anymore (options are hidden by default and
        only shown inside the Live Settings dialog when it's open).
        The dialog re-reads visibility rules on open, so nothing to
        do here at rest."""
        dlg = self._live_settings_dialog
        if dlg is not None:
            try:
                dlg.refresh_visibility()
            except Exception:
                pass

    def open_live_settings(self):
        """Open (or focus) the Live Settings modal.

        The dialog rehomes every hidden state widget into its form so
        the user's edits go straight into `self._*` — nothing to sync.
        On close, widgets are re-parented back to `self` (hidden again)
        so state persists across opens.
        """
        if (self._live_settings_dialog is not None
                and self._live_settings_dialog.isVisible()):
            self._live_settings_dialog.raise_()
            self._live_settings_dialog.activateWindow()
            return
        self._live_settings_dialog = LiveSettingsDialog(self)
        self._live_settings_dialog.finished.connect(self._on_settings_closed)
        self._live_settings_dialog.show()

    def _on_settings_closed(self, *_):
        """Redraw the canvases after the Live Settings dialog closes.

        A visual-only change -- an outline colour, say -- alters nothing the
        worker computed, so it reaches the picture only through this redraw.

        :param _: whatever the dialog's finished signal passes; unused.
        """
        self._refresh_canvases()
        self._live_settings_dialog = None

    def _pick_file(self):
        """Ask for a preview image and load it.

        The chosen file may be in a folder the sampler has never enumerated, so
        this load does enumerate -- off the GUI thread.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose preview image", "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg)",
        )
        if path:
            self._pin_path = Path(path)
            self.load_source_async(path)

    def _on_hover(self, x: int, y: int):
        """Render the pinned hover-info line for the pixel under the cursor."""
        if self._image is None:
            return
        h = self._image.shape[0]
        w = self._image.shape[1] if self._image.ndim >= 2 else 0
        if not (0 <= y < h and 0 <= x < w):
            self._hover_label.setText(
                "Hover over the image to inspect pixels.")
            return
        if self._image.ndim == 3:
            vals = tuple(int(v) for v in self._image[y, x])
            i_str = f"channels={vals}"
        else:
            i_str = f"intensity={int(self._image[y, x])}"
        hits = []
        for obj, mask in self._masks.items():
            if mask is None or mask.size == 0:
                continue
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                lbl = int(mask[y, x])
                if lbl > 0:
                    area = int((mask == lbl).sum())
                    hits.append(f"{obj}=#{lbl} area={area}px")
        obj_str = f"  {'  '.join(hits)}" if hits else ""
        self._hover_label.setText(f"(x={x:>4d}, y={y:>4d})  {i_str}{obj_str}")

    def _stale(self, token: int) -> bool:
        """True when ``token`` belongs to a superseded run.

        ``-1`` is the direct-call escape hatch used by tests and by callers
        that push a result in by hand; those are never stale.
        """
        return token >= 0 and token != self._run_token

    def _on_worker_done(self, masks, err, token: int = -1):
        """Install a finished preview, unless a newer run has superseded it.

        The raw masks are cached before filtering, so a filter change can be
        re-applied without segmenting again.

        :param masks: the worker's masks by compartment.
        :param err: the failure, or a falsy value on success.
        :param token: the run token this result carries; a stale one is dropped.
        """
        if self._stale(token):
            LOG.debug("dropping stale preview result (token %s, now %s)",
                      token, self._run_token)
            return
        self.set_preview_busy(False)
        if err:
            self._status.setText(preview_failure_message(err))
            self.preview_ready.emit(None)
            return
        if masks is None or not masks:
            self._status.setText("Preview returned no masks.")
            return
        pending = self._pending_provenance
        if pending is not None and pending[0] == token:
            self._processing_provenance = pending[1]
            self._model_that_ran = pending[1].get('model', '')
            self._model_note = pending[1].get('model_note', '')
            self._status.setToolTip(self._processing_tooltip(pending[1]))
        self._pending_provenance = None
        self._raw_masks = masks
        self._recompute_masks(snapshot=True)

    def _obj_channel(self, obj: str) -> int:
        """Use the same own-channel selection for segmentation and filtering.

        Numbered organelles retain their stored channel while another slot
        owns the shared spinner. Unknown roles keep the legacy zero fallback.
        """
        channel = self._channel_for_object(obj)
        return 0 if channel is None else channel

    def _recompute_masks(self, snapshot: bool = False) -> None:
        """Re-apply the current per-compartment filters to the cached raw
        masks and refresh the views — no Cellpose re-run. Called both after a
        preview and whenever a filter widget changes."""
        raw = getattr(self, "_raw_masks", None)
        if not raw:
            return
        if snapshot:
            self._roll_auto_outline_colours()
        try:
            if self._image is None:
                raise ValueError(self.PREVIEW_SOURCE_HINT)
            post = dict(self._settings)
            if hasattr(self, "_compartment_widgets"):
                post.update(self._compartment_settings())
            out = {}
            for obj, raw_mask in raw.items():
                intensity = _select_channel(self._image, self._obj_channel(obj))
                out[obj] = _apply_size_filter(raw_mask, post, obj,
                                              intensity_img=intensity)
        except Exception as exc:
            LOG.debug("preview filtering failed", exc_info=True)
            self._masks = {}
            self._status.setText(preview_failure_message(exc))
            if self._image is None:
                self._mask_view.set_pixmap(QPixmap())
            else:
                self._refresh_canvases()
            self.preview_ready.emit(None)
            return
        self._masks = out
        counts = [f"{k}={int(v.max() if v.size else 0)}"
                    for k, v in out.items()]
        self._status.setText(
            f"Found {', '.join(counts)}.  {self._model_provenance()}")
        operation = self._processing_provenance.get('processing', {}).get('operation')
        if operation and operation != 'none':
            self._status.setText(self._status.text() + '  ' + tr(
                'PSF: {operation} (preview field).', operation=operation))
        self._refresh_canvases()
        if snapshot:
            self._snapshot_run(out, counts)
        self.preview_ready.emit(out)


    def _snapshot_run(self, masks, counts) -> None:
        """Record a preview run (image + masks + display params) so the user
        can scrub back to compare it against later runs."""
        if self._image is None:
            return
        snap = {
            "image": self._image,
            "masks": {k: v for k, v in masks.items()},
            "norm": self._normalise_check.isChecked(),
            "lo": float(self._lo_pct.value()),
            "hi": float(self._hi_pct.value()),
            # The model that RAN, not the one now selected: the history is
            # scrubbed back to compare passes, and a pass labelled with a
            # model chosen after it is a comparison of the wrong two things.
            "model": self._model_that_ran or self._model_box.currentText(),
            "object": _combo_value(self._object_box),
            "summary": ", ".join(counts),
            "processing_provenance": deepcopy(self._processing_provenance),
        }
        methods = self._processing_provenance.get('methods', {})
        if methods and 'cellpose' not in methods.values():
            snap['model'] = ', '.join(sorted(set(methods.values())))
        self._history.append(snap)
        if len(self._history) > 50:
            self._history = self._history[-50:]
        n = len(self._history)
        self._compare_row.setVisible(n >= 2)
        self._compare_slider.blockSignals(True)
        self._compare_slider.setMaximum(n - 1)
        self._compare_slider.setValue(n - 1)
        self._compare_slider.blockSignals(False)
        self._compare_label.setText(f"{n}/{n}")

    def _on_compare_scrub(self, idx: int) -> None:
        """Render the historical run at ``idx`` into the two canvases."""
        if not (0 <= idx < len(self._history)):
            return
        snap = self._history[idx]
        img = channel_view(snap["image"], self.display_channel())
        norm, lo, hi = snap["norm"], snap["lo"], snap["hi"]
        src_pix = numpy_to_qpixmap(
            _to_uint8(img, normalise=norm, lo_pct=lo, hi_pct=hi))
        self._src_view.set_pixmap(src_pix)
        if snap["masks"]:
            overlay = overlay_masks(
                img, snap["masks"], outline_rgb=self._outline_rgb(),
                outline_thickness=self._outline_thickness.value(),
                normalise=norm, lo_pct=lo, hi_pct=hi,
                random_outline=(
                    self._outline_choice() == "color (random)"
                ),
                outline_colors=self._auto_outline_map(),
                primaries=self.display_primaries())
            self._mask_view.set_pixmap(numpy_to_qpixmap(overlay))
        else:
            self._mask_view.set_pixmap(src_pix)
        self._compare_label.setText(
            f"{idx + 1}/{len(self._history)}  "
            f"{snap['model']}/{snap['object']}  {snap['summary']}")
        self._compare_label.setToolTip(self._processing_tooltip(
            snap.get('processing_provenance', {})))

    def refresh_model_choices(self) -> None:
        """Re-read the Cellpose model list and add anything new.

        `spacr.settings.cellpose_model_choices` only reads the API when
        Cellpose is already imported, because importing it costs ~2.5 s and
        this panel is built while a page is being laid out. That means the
        first build usually gets the shipped fallback — so ask again every
        time the panel is shown. After the first segmentation Cellpose is
        loaded and a checkpoint the user registered appears here.

        Additive on purpose: the current selection is never disturbed, and
        an entry is never removed, so a value the user picked cannot vanish
        under them because a probe came back thinner.
        """
        wanted = _model_menu()
        have = {self._model_box.itemText(i)
                for i in range(self._model_box.count())}
        for index, name in enumerate(wanted):
            if name not in have:
                self._model_box.insertItem(index, name)

    def showEvent(self, event):  # noqa: N802 (Qt naming)
        """Refresh the model list whenever the panel comes back on screen."""
        super().showEvent(event)
        self.refresh_model_choices()



from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QScrollArea,
)
from .sortable_table import install_sorting, table_item


class LiveSettingsDialog(QDialog):
    """Modal dialog that surfaces every live-preview setting.

    Re-parents the panel's hidden state widgets into a QFormLayout so
    edits go straight into the panel's canonical fields — nothing to
    sync manually. On close, widgets are returned to the panel hidden
    so their values persist across opens.

    Rows shown (per the user's spec):
      * Normalisation upper + lower percentile
      * Outline colour
      * Outline thickness
      * Model
      * Flow threshold
      * Cell probability
      * Object type
      * Object channel (cell / nucleus depending on selection)
      * Pre  (bool)
      * Post (bool)

    :param panel: the preview panel this dialog edits. It is also the
        dialog's PARENT, and the widgets the dialog lays out belong to the
        panel rather than to it -- the dialog only knows which rows they sit
        on, which is what lets a morphology change re-gate them.
    """

    def __init__(self, panel: "LivePreviewPanel"):
        """Build the dialog around the panel's own controls.

        The controls are the panel's and are re-parented in here for the
        lifetime of the dialog, so their values survive it being closed and
        reopened. The panel is told which dialog is open, so a morphology change
        can re-gate the rows -- the widgets live on the panel, but it is the
        dialog that knows which row each sits on.

        :param panel: the live-preview panel whose controls this edits.
        """
        super().__init__(panel)
        self._panel = panel
        panel._live_settings_dialog = self
        self.setWindowTitle(tr("Live settings"))
        outer = QVBoxLayout(self)

        for w in self._managed_widgets():
            w.show()

        panels_row = QHBoxLayout()
        panels_row.setSpacing(12)

        seg_group = QGroupBox("Segmentation")
        form = QFormLayout(seg_group)
        model_row = QWidget(seg_group)
        model_row_layout = QHBoxLayout(model_row)
        model_row_layout.setContentsMargins(0, 0, 0, 0)
        model_row_layout.setSpacing(4)
        model_row_layout.addWidget(panel._model_box, 1)
        panel._model_zoo_btn = QPushButton("Model zoo…", model_row)
        panel._model_zoo_btn.setToolTip(
            "Browse the models spaCR knows about, download one and preview "
            "with it. The same list the object model settings offer.")
        panel._model_zoo_btn.clicked.connect(panel._choose_a_preview_model)
        model_row_layout.addWidget(panel._model_zoo_btn)
        form.addRow("Model", model_row)
        form.addRow("Primary object", panel._object_box)
        form.addRow("Cell channel", panel._cell_channel)
        form.addRow("Nucleus channel", panel._nucleus_channel)
        form.addRow("Pathogen channel", panel._pathogen_channel)
        form.addRow("Organelle channel", panel._organelle_channel)
        form.addRow("Diameter", panel._diameter)
        form.addRow("Flow threshold", panel._flow)
        form.addRow("Cell probability", panel._prob)
        form.addRow(panel._normalise_check)
        form.addRow("Lower percentile", panel._lo_pct)
        form.addRow("Upper percentile", panel._hi_pct)
        form.addRow("Outline colour", panel._outline_colour)
        form.addRow("Outline thickness", panel._outline_thickness)
        panel._common_widgets["signal_to_noise"].show()
        panel._common_widgets["remove_background"].show()
        panel._common_widgets["background"].show()
        form.addRow("Signal to noise", panel._common_widgets["signal_to_noise"])
        form.addRow("Remove background", panel._common_widgets["remove_background"])
        form.addRow("Background", panel._common_widgets["background"])
        panels_row.addWidget(seg_group)

        self._compartment_groupboxes: Dict[str, QGroupBox] = {}
        for comp in COMPARTMENTS:
            box = QGroupBox(comp.capitalize())
            cform = QFormLayout(box)
            for suffix, label, _kind, _args in COMPARTMENT_FIELDS:
                w = panel._compartment_widgets[comp][suffix]
                w.show()
                cform.addRow(label, w)
            if comp == "cell":
                panel._adjust_cells.show()
                cform.addRow("Adjust cells", panel._adjust_cells)
            self._compartment_groupboxes[comp] = box
            panels_row.addWidget(box)

        self._organelle_group = QGroupBox("Organelle segmentation")
        organelle_form = QFormLayout(self._organelle_group)
        self._organelle_rows: Dict[str, tuple] = {}
        for morphology, group in ORGANELLE_METHOD_FIELDS.items():
            for suffix, label, _kind, _args in group:
                widget = panel._organelle_widgets[suffix]
                widget.show()
                organelle_form.addRow(label, widget)
                self._organelle_rows[suffix] = (morphology, widget)
        panels_row.addWidget(self._organelle_group)

        row_host = QWidget()
        row_host.setLayout(panels_row)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(row_host)
        outer.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self._run_btn = QPushButton("Run preview")
        self._run_btn.setDefault(True)
        self._run_btn.clicked.connect(self._panel.run_preview)
        buttons.addButton(self._run_btn, QDialogButtonBox.ActionRole)
        self._propagate_btn = QPushButton("Propagate settings")
        self._propagate_btn.setObjectName("ToggleButton")
        self._propagate_btn.setCheckable(True)
        self._propagate_btn.setToolTip(
            "When on, changes made here are copied into the main settings "
            "panel.")
        self._propagate_btn.toggled.connect(self._on_propagate_toggled)
        buttons.addButton(self._propagate_btn, QDialogButtonBox.ActionRole)
        buttons.rejected.connect(self.close)
        buttons.accepted.connect(self.close)
        outer.addWidget(buttons)

        panel._object_box.currentTextChanged.connect(self.refresh_visibility)
        panel._model_box.currentTextChanged.connect(self.refresh_visibility)
        panel._normalise_check.toggled.connect(self.refresh_visibility)

        self._propagate_sources = [
            panel._model_box, panel._object_box, panel._cell_channel,
            panel._nucleus_channel, panel._pathogen_channel,
            panel._organelle_channel, panel._diameter, panel._flow,
            panel._prob, panel._normalise_check, panel._lo_pct, panel._hi_pct,
        ] + panel._all_compartment_widgets()

        self._show_every_control_on_a_row()
        self.refresh_visibility()

        try:
            from ..hidpi import screen_for_widget
            avail = screen_for_widget(self).availableGeometry()
            want = row_host.sizeHint().width() + 48
            self.resize(min(want, avail.width() - 80), min(760, avail.height() - 80))
        except Exception:
            self.resize(1400, 720)

    def _on_propagate_toggled(self, on: bool) -> None:
        """Connect/disconnect live→main propagation and do an initial push."""
        for w in self._propagate_sources:
            for sig_name in ("valueChanged", "currentTextChanged", "toggled"):
                sig = getattr(w, sig_name, None)
                if sig is None:
                    continue
                try:
                    if on:
                        sig.connect(self._panel.propagate_settings)
                    else:
                        sig.disconnect(self._panel.propagate_settings)
                except (TypeError, RuntimeError):
                    pass
        if on:
            self._panel.propagate_settings()

    def _managed_widgets(self):
        """List the panel controls this dialog re-parents.

        :returns: the segmentation and normalisation controls followed by every
            per-compartment one.
        """
        p = self._panel
        return [p._model_box, p._object_box, p._cell_channel,
                p._nucleus_channel, p._pathogen_channel,
                p._organelle_channel, p._diameter, p._flow, p._prob,
                p._normalise_check, p._lo_pct, p._hi_pct,
                p._outline_colour, p._outline_thickness,
                ] + p._all_compartment_widgets()

    def _show_every_control_on_a_row(self) -> int:
        """Show every widget this dialog has put on a form row.

        :meth:`closeEvent` hides each borrowed control as it hands it back,
        and a widget hidden that way stays hidden when a layout takes it
        again. A control that is on a row but not named by
        :meth:`_managed_widgets` therefore came back as a caption over an
        empty field on every open after the first. The sweep is by form
        row, so a control added to the dialog later is shown without a
        second list to keep in step. Rows the dialog gates on purpose are
        hidden with ``QFormLayout.setRowVisible`` in
        :meth:`refresh_visibility`, which this does not touch.

        :returns: how many widgets were shown.
        """
        shown = 0
        for form in self.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                for role in (QFormLayout.LabelRole, QFormLayout.FieldRole,
                             QFormLayout.SpanningRole):
                    item = form.itemAt(row, role)
                    widget = item.widget() if item is not None else None
                    if widget is None or not widget.isHidden():
                        continue
                    widget.show()
                    shown += 1
        return shown

    def _install_api_tooltips(self) -> None:
        """Attach linked Mask API help to every setting in this popup."""
        from ..screens.settings_model import install_api_tooltips

        p = self._panel
        widget_keys = {
            p._model_box: "model_name",
            p._object_box: "object_type",
            p._cell_channel: "cell_channel",
            p._nucleus_channel: "nucleus_channel",
            p._pathogen_channel: "pathogen_channel",
            p._organelle_channel: "organelle_channel",
            p._diameter: "cell_diameter",
            p._flow: "cell_flow_threshold",
            p._prob: "cell_cellprob_threshold",
            p._normalise_check: "normalize",
            p._lo_pct: "lower_percentile",
            p._hi_pct: "upper_percentile",
            p._outline_colour: "outline_color",
            p._outline_thickness: "outline_thickness",
            p._common_widgets["signal_to_noise"]: "cell_signal_to_noise",
            p._common_widgets["remove_background"]: "remove_background_cell",
            p._common_widgets["background"]: "cell_background",
            p._adjust_cells: "adjust_cells",
        }
        for compartment, fields in p._compartment_widgets.items():
            for suffix, widget in fields.items():
                widget_keys[widget] = f"{compartment}_{suffix}"
        install_api_tooltips(self, "mask", widget_keys)

    def refresh_visibility(self):
        """Grey out settings that don't apply to the current selection.

        Rules (mirroring the pipeline's own relevance):
          * Nothing in the Segmentation group greys out for the model.
            Cellpose 4 ships one set of weights and all three knobs
            (diameter / flow / cell-prob) still reach it — see
            :data:`DIAMETER_TOOLTIP` for the measurement.
          * The object type decides which channel spinners are live: the cell
            channel greys out for a nucleus-only object and vice-versa.
          * Pre-processing knobs (normalise + its two percentiles) are only
            relevant when the *Pre* step is enabled.
          * Overlay / post knobs (outline colour + thickness) are only
            relevant when the *Post* step is enabled.
        """
        p = self._panel

        p._diameter.setEnabled(True)
        p._diameter.setToolTip(DIAMETER_TOOLTIP)
        p._flow.setEnabled(True)
        p._prob.setEnabled(True)
        p._flow.setToolTip("")
        p._prob.setToolTip("")

        selected = set(p._selected_object_types())
        p._cell_channel.setEnabled("cell" in selected)
        p._nucleus_channel.setEnabled("nucleus" in selected)

        ordered = list(p._selected_object_types())
        primary = ordered[0] if ordered else "cell"
        for comp, box in self._compartment_groupboxes.items():
            is_primary = (comp == primary)
            is_secondary = (comp == "nucleus" and "nucleus" in selected
                            and not is_primary)
            box.setVisible(is_primary or is_secondary)
            box.setEnabled(True)
            if is_primary:
                box.setTitle(f"{comp.capitalize()} (primary object)")
            elif is_secondary:
                box.setTitle("Nucleus (secondary object)")

        organelle_primary = primary.startswith("organelle")
        self._organelle_group.setVisible(organelle_primary)
        if organelle_primary:
            self._organelle_group.setTitle(
                f"{primary.capitalize()} segmentation")
            morphology = p._organelle_morphology()
            form = self._organelle_group.layout()
            for suffix, (owner, widget) in self._organelle_rows.items():
                wanted = owner is None or owner == morphology
                position = form.getWidgetPosition(widget)[0]
                if position >= 0:
                    form.setRowVisible(position, wanted)

        p._normalise_check.setEnabled(True)
        p._normalise_check.setToolTip("")
        norm_on = p._normalise_check.isChecked()
        for w in (p._lo_pct, p._hi_pct):
            w.setEnabled(norm_on)
            w.setToolTip("" if norm_on
                         else "Enable 'Normalise' to set percentile bounds")

        for w in (p._outline_colour, p._outline_thickness):
            w.setEnabled(True)
        self._install_api_tooltips()

    def closeEvent(self, event):
        """Give back every panel control this dialog borrowed, not just the
        declared ones.

        Qt destroys a dialog's children with it, so anything of the panel's
        still parented under this dialog when it goes would go with it. The
        controls are the PANEL's and outlive the dialog by design -- their
        values are what the user tuned.

        `_managed_widgets()` is the declared list and it was INCOMPLETE:
        `_pathogen_channel`, `_organelle_channel` and `_model_zoo_btn` are
        laid out here too and were not in it, so after one open and close
        they sat parented to a group box belonging to a closed dialog.
        Nothing failed -- they survived, because the panel still held Python
        references -- which is exactly why it went unnoticed.

        So the sweep is by IDENTITY rather than by list: any widget still
        under this dialog that the panel holds an attribute for goes back.
        Blunt on purpose, like `LivePreviewPanel._stow_free_widgets`, so a
        control added to a row later is covered without anyone remembering
        to add it here as well.

        They go to `_offscreen_controls` and not to the panel: parented to
        the panel with no layout, each sits at (0, 0) over the loaded-path
        label, held off screen by nothing but the `hide()`.
        """
        panel = self._panel
        stow = getattr(panel, "_offscreen_controls", None) or panel
        owned = {id(value) for value in vars(panel).values()
                 if isinstance(value, QWidget)}
        for group in getattr(panel, "_compartment_widgets", {}).values():
            owned.update(id(w) for w in group.values())
        owned.update(id(w) for w in
                     getattr(panel, "_common_widgets", {}).values())

        borrowed = list(self._managed_widgets())
        borrowed += [w for w in self.findChildren(QWidget)
                     if id(w) in owned]
        seen = set()
        for w in borrowed:
            if id(w) in seen:
                continue
            seen.add(id(w))
            w.hide()
            w.setParent(stow)
        super().closeEvent(event)



def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Legacy single-mask overlay retained for older imports."""
    return overlay_masks(image, {"cell": mask})
