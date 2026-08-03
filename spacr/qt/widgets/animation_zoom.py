"""Content-aware zoom for the packaged setting animations.

Every generated GIF draws its scene inside the same rounded "well" — the
336-pixel frame that :func:`tools.generate_setting_animations._well` paints
at (12, 12)-(348, 348) on a 360-pixel black square. The *scene* inside it is
usually far smaller than the well. Measured across all 94 packaged
animations, the median content covers only 63.9 % of the square, 72 of them
are below 70 %, and the smallest (``nucleus_diameter``) covers 22.8 %. At
tooltip size that leaves a few dozen pixels of actual illustration floating
in black.

So the frames are not shown as generated. Each animation's real content
bounds are measured once — the union, across every frame, of everything that
is neither background nor well chrome — and the frames are then cropped and
rescaled so the content covers a fixed :data:`TARGET_FILL` share of the
square. Animations whose content already overflows that share (the
``remove_border_objects`` scenes reach 91.9 %) are scaled *down* to the same
target, so every animation in the app is presented at one consistent size.

The measurement decodes and scans every frame, which is far too expensive to
repeat on an animation tick — :func:`zoomed_animation` is therefore cached on
``(file, size, target)`` and hands back finished, display-sized frames.

This module is deliberately Qt-free except for :func:`to_qimage`, so the
measurement can be used from tests and tools without a QApplication.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageSequence


#: Side of the generated square in source pixels. The well geometry below is
#: expressed against this and scaled for any other frame size, so an animation
#: regenerated at a different resolution still measures correctly.
SOURCE_SIZE = 360

#: The rounded field boundary drawn by ``_well()`` in
#: ``tools/generate_setting_animations.py``: an inset rounded rectangle with a
#: 20-pixel corner radius. It is chrome, not content — every scene has it, so
#: including it would make every animation measure ~93 % full and defeat the
#: whole point of the zoom.
FIELD_BOX = (12, 12, SOURCE_SIZE - 12, SOURCE_SIZE - 12)
FIELD_RADIUS = 20

#: Half-width of the band masked out around the field path. The stroke is
#: sub-pixel at output scale and LANCZOS smears it across two to three pixels,
#: so the band has to be wider than the nominal line or stray remnants of it
#: re-enter the measurement and read as content at the frame's full extent.
FIELD_PAD = 3

#: Channel value at or below which a pixel counts as background. The frames
#: are drawn on pure black and anti-aliased, so a small floor keeps the
#: faintest edge pixels of a shape without picking up encoder noise.
BACKGROUND_LEVEL = 8

#: Lit 8-neighbours a lit pixel needs before it counts as content.
#:
#: GIF quantisation leaves single specks one level above the background —
#: ``nucleus_intensity_merge`` has exactly one, value 9, in the far corner of
#: one frame out of eighteen. Left in, that speck alone stretched the measured
#: content from 132x86 to 231x203 and the zoom then framed an empty corner.
#: Real content here is anti-aliased line art and filled shapes, where every
#: pixel of a stroke has neighbours; an isolated pixel is never a drawing.
MIN_NEIGHBOURS = 1

#: Share of the square the content is scaled to occupy. The requirement is a
#: 70-80 % band; aiming at the middle leaves room for the rounding and
#: resampling slack that a hard edge of the band would not survive.
TARGET_FILL = 0.75
MIN_FILL = 0.70
MAX_FILL = 0.80

#: Frame duration used when a GIF frame declares none.
DEFAULT_DELAY_MS = 80


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def read_frames(path) -> Tuple[Tuple[np.ndarray, ...], Tuple[int, ...]]:
    """Decode ``path`` into composed RGB frames and their delays.

    :param path: any path-like pointing at a packaged animation.
    :returns: ``(frames, delays_ms)``; frames are ``(h, w, 3)`` uint8 arrays
        already composed against the preceding frame by Pillow, so a GIF that
        encodes only the changed rectangle still yields whole pictures.
    """
    frames: list[np.ndarray] = []
    delays: list[int] = []
    with Image.open(str(path)) as image:
        for frame in ImageSequence.Iterator(image):
            frames.append(np.array(frame.convert("RGB"), dtype=np.uint8))
            duration = frame.info.get("duration") or 0
            delays.append(int(duration) or DEFAULT_DELAY_MS)
    return tuple(frames), tuple(delays)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def field_geometry(size: int) -> Tuple[Tuple[float, float, float, float], float]:
    """Return the well rectangle and corner radius for a frame of ``size``."""
    scale = float(size) / float(SOURCE_SIZE)
    box = tuple(value * scale for value in FIELD_BOX)
    return box, FIELD_RADIUS * scale


def _field_masks(
    size: int,
    box: Optional[Sequence[float]] = None,
    radius: Optional[float] = None,
    pad: float = FIELD_PAD,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(ring, inside_outer)`` for the well of a ``size`` frame."""
    if box is None or radius is None:
        default_box, default_radius = field_geometry(size)
        box = default_box if box is None else box
        radius = default_radius if radius is None else radius
    left, top, right, bottom = (float(value) for value in box)

    outer = Image.new("L", (size, size), 0)
    ImageDraw.Draw(outer).rounded_rectangle(
        (left - pad, top - pad, right + pad, bottom + pad),
        radius=max(0.0, radius + pad),
        fill=255,
    )
    inner = Image.new("L", (size, size), 0)
    inner_box = (left + pad, top + pad, right - pad, bottom - pad)
    if inner_box[2] > inner_box[0] and inner_box[3] > inner_box[1]:
        ImageDraw.Draw(inner).rounded_rectangle(
            inner_box, radius=max(0.0, radius - pad), fill=255,
        )
    inside_outer = np.asarray(outer) > 0
    inside_inner = np.asarray(inner) > 0
    return inside_outer & ~inside_inner, inside_outer


def field_ring_mask(
    size: int,
    box: Optional[Sequence[float]] = None,
    radius: Optional[float] = None,
    pad: float = FIELD_PAD,
) -> np.ndarray:
    """Mask covering only the drawn field stroke, not the space outside it."""
    ring, _inside = _field_masks(size, box, radius, pad)
    return ring


def chrome_mask(
    size: int,
    box: Optional[Sequence[float]] = None,
    radius: Optional[float] = None,
    pad: float = FIELD_PAD,
) -> np.ndarray:
    """Mask of everything that is decoration rather than scene content.

    That is the band of width ``2 * pad`` straddling the rounded field path,
    plus everything outside the field. Used both on the source frames (to
    measure real content) and on the zoomed output (to measure it again after
    the transform, when the field survived the crop).

    :returns: boolean ``(size, size)`` array, ``True`` where a pixel must be
        ignored by the content measurement.
    """
    ring, inside_outer = _field_masks(size, box, radius, pad)
    return ring | ~inside_outer


def drop_specks(mask: np.ndarray, minimum: int = MIN_NEIGHBOURS) -> np.ndarray:
    """Return ``mask`` without lit pixels that have too few lit neighbours.

    See :data:`MIN_NEIGHBOURS` for why an isolated pixel has to go. Done with
    shifted slices rather than a convolution so the module keeps its only
    numeric dependency.
    """
    if minimum <= 0 or mask.size == 0:
        return mask
    height, width = mask.shape
    padded = np.zeros((height + 2, width + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask
    neighbours = np.zeros((height, width), dtype=np.uint8)
    for row in (0, 1, 2):
        for column in (0, 1, 2):
            if row == 1 and column == 1:
                continue
            neighbours += padded[row:row + height, column:column + width]
    return mask & (neighbours >= minimum)


def content_mask(
    frames: Sequence[np.ndarray],
    chrome: Optional[np.ndarray] = None,
    minimum_neighbours: int = MIN_NEIGHBOURS,
) -> np.ndarray:
    """Union across ``frames`` of every non-background, non-chrome pixel.

    The union — not one representative frame — is what has to be framed: an
    animation whose object drifts across the well would otherwise be cropped
    to wherever it happened to be when the measurement ran.

    Specks are dropped per frame, before the union: a stray pixel is noise in
    the frame that produced it, and unioning first would let one frame's speck
    borrow a neighbour from another frame's real content.
    """
    accumulated: Optional[np.ndarray] = None
    for frame in frames:
        # Three plane comparisons, not ``frame.max(axis=2) > level``: the
        # reduction runs along the length-3 interleaved axis and measured ten
        # times slower, which across every frame of every animation is the
        # difference between a hover that stutters and one that does not.
        lit = (
            (frame[..., 0] > BACKGROUND_LEVEL)
            | (frame[..., 1] > BACKGROUND_LEVEL)
            | (frame[..., 2] > BACKGROUND_LEVEL)
        )
        if chrome is not None:
            lit = lit & ~chrome
        lit = drop_specks(lit, minimum_neighbours)
        accumulated = lit if accumulated is None else (accumulated | lit)
    if accumulated is None:
        return np.zeros((0, 0), dtype=bool)
    return accumulated


def content_bounds(
    frames: Sequence[np.ndarray],
    chrome: Optional[np.ndarray] = None,
    minimum_neighbours: int = MIN_NEIGHBOURS,
) -> Optional[Tuple[int, int, int, int]]:
    """Inclusive ``(left, top, right, bottom)`` of the content, or ``None``.

    ``None`` means the animation is blank once chrome is discounted, which is
    not a failure — it simply cannot be zoomed, and callers show it as-is.
    """
    mask = content_mask(frames, chrome, minimum_neighbours)
    if not mask.size:
        return None
    rows = np.nonzero(mask.any(axis=1))[0]
    columns = np.nonzero(mask.any(axis=0))[0]
    if not rows.size or not columns.size:
        return None
    return (
        int(columns[0]), int(rows[0]), int(columns[-1]), int(rows[-1]),
    )


def content_extent(
    frames: Sequence[np.ndarray],
    chrome: Optional[np.ndarray] = None,
    minimum_neighbours: int = MIN_NEIGHBOURS,
) -> float:
    """Share of the square the content spans, on its longer axis.

    This is *the* number the 70-80 % requirement is stated in, and the same
    function measures the source and the zoomed result — a target the
    transform is checked against rather than one it defines for itself.
    """
    if not len(frames):
        return 0.0
    bounds = content_bounds(frames, chrome, minimum_neighbours)
    if bounds is None:
        return 0.0
    left, top, right, bottom = bounds
    side = max(right - left + 1, bottom - top + 1)
    return float(side) / float(frames[0].shape[0])


def source_content_extent(path) -> float:
    """Measure the packaged animation at ``path`` as generated."""
    frames, _delays = read_frames(path)
    if not frames:
        return 0.0
    return content_extent(frames, chrome_mask(frames[0].shape[0]))


# ---------------------------------------------------------------------------
# Zoom
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ZoomedAnimation:
    """One packaged animation, cropped and rescaled for display.

    :param frames: display-sized ``(size, size, 3)`` uint8 RGB frames.
    :param delays: per-frame duration in milliseconds.
    :param source_extent: content share of the square before the zoom.
    :param fill: content share after it — nominally :data:`TARGET_FILL`.
    :param crop: ``(left, top, side)`` in source pixels; ``left``/``top`` may
        be negative and ``side`` may exceed the source when content had to be
        scaled *down*, in which case the frame is padded with background.
    :param shows_field: whether the rounded well survived the crop whole. When
        it did not it is erased, because a well sliced by the crop reads as
        two stray lines rather than as a boundary.
    """

    path: str
    size: int
    frames: Tuple[np.ndarray, ...]
    delays: Tuple[int, ...]
    source_extent: float
    fill: float
    crop: Tuple[int, int, int]
    shows_field: bool

    def chrome_mask(self) -> Optional[np.ndarray]:
        """Chrome mask in *output* coordinates, or ``None`` if there is none.

        Only meaningful when the well survived: the same rounded rectangle,
        mapped through the crop and the scale, so the zoomed frames can be
        measured by exactly the rule the source frames were measured by.
        """
        if not self.shows_field:
            return None
        left, top, side = self.crop
        scale = float(self.size) / float(side)
        box = tuple(
            (value - offset) * scale
            for value, offset in zip(
                FIELD_BOX, (left, top, left, top)
            )
        )
        # The mask has to stay wider than the line after the downscale, or
        # remnants of the well re-enter the measurement.
        pad = max(3.0, FIELD_PAD * scale + 1.0)
        return chrome_mask(self.size, box, FIELD_RADIUS * scale, pad)

    def measured_fill(self) -> float:
        """Re-measure the produced frames rather than trusting the maths."""
        return content_extent(self.frames, self.chrome_mask())


def zoom_frames(
    frames: Sequence[np.ndarray],
    size: int,
    target: float = TARGET_FILL,
) -> Tuple[Tuple[np.ndarray, ...], Tuple[int, int, int], float, float, bool]:
    """Crop and rescale ``frames`` so their content covers ``target``.

    :returns: ``(frames, crop, source_extent, fill, shows_field)``.
    """
    source_size = frames[0].shape[0]
    chrome = chrome_mask(source_size)
    bounds = content_bounds(frames, chrome)
    if bounds is None:
        scaled = tuple(
            _resize(frame, size) for frame in frames
        )
        return scaled, (0, 0, source_size), 0.0, 0.0, True

    left, top, right, bottom = bounds
    extent = max(right - left + 1, bottom - top + 1)
    source_extent = float(extent) / float(source_size)

    side = max(1, int(round(extent / float(target))))
    centre_x = (left + right + 1) / 2.0
    centre_y = (top + bottom + 1) / 2.0
    crop_left = int(round(centre_x - side / 2.0))
    crop_top = int(round(centre_y - side / 2.0))
    fill = float(extent) / float(side)

    field_left, field_top, field_right, field_bottom = FIELD_BOX
    shows_field = (
        crop_left <= field_left - FIELD_PAD
        and crop_top <= field_top - FIELD_PAD
        and crop_left + side >= field_right + FIELD_PAD
        and crop_top + side >= field_bottom + FIELD_PAD
    )

    # Built once, not per frame: the ring is the same in every frame and
    # rasterising a rounded rectangle twice per frame would dominate the load.
    ring = None if shows_field else field_ring_mask(source_size)

    scaled = []
    for frame in frames:
        source = frame
        if ring is not None:
            source = frame.copy()
            source[ring] = 0
        # Pillow's crop pads out-of-bounds regions with black, which is the
        # animations' own background — so scaling content *down* needs no
        # special case.
        cropped = Image.fromarray(source).crop(
            (crop_left, crop_top, crop_left + side, crop_top + side)
        )
        scaled.append(
            np.array(
                cropped.resize((size, size), Image.Resampling.LANCZOS),
                dtype=np.uint8,
            )
        )
    return (
        tuple(scaled), (crop_left, crop_top, side), source_extent, fill,
        shows_field,
    )


def _resize(frame: np.ndarray, size: int) -> np.ndarray:
    return np.array(
        Image.fromarray(frame).resize(
            (size, size), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


#: How many finished animations stay resident.
#:
#: Deliberately small. A zoomed animation is ~18 frames of 220x220 RGB, about
#: 2.6 MB, so caching all 94 would cost a quarter of a gigabyte to save 40 ms
#: on a hover. Eight covers re-hovering a setting, moving to its neighbour and
#: coming back, which is what people actually do while reading a form.
CACHE_SIZE = 8


@lru_cache(maxsize=CACHE_SIZE)
def zoomed_animation(
    path: str,
    size: int,
    target: float = TARGET_FILL,
) -> Optional[ZoomedAnimation]:
    """Load, measure and zoom one animation — cached per ``(path, size)``.

    Decoding a GIF and scanning every frame costs tens of milliseconds; doing
    it per animation tick would be absurd, and doing it per hover would make
    every tooltip stutter. Callers may treat the result as immutable.

    :returns: ``None`` when the file cannot be decoded, so a missing or
        corrupt asset degrades to a text-only tooltip instead of raising into
        the event loop.
    """
    try:
        frames, delays = read_frames(path)
    except (OSError, ValueError):
        return None
    if not frames:
        return None

    scaled, crop, source_extent, fill, shows_field = zoom_frames(
        frames, int(size), float(target))
    return ZoomedAnimation(
        path=str(path),
        size=int(size),
        frames=scaled,
        delays=delays,
        source_extent=source_extent,
        fill=fill,
        crop=crop,
        shows_field=shows_field,
    )


def clear_cache() -> None:
    """Drop every cached zoom — used by tests and by asset regeneration."""
    zoomed_animation.cache_clear()


# ---------------------------------------------------------------------------
# Qt bridge
# ---------------------------------------------------------------------------

def to_qimage(frame: np.ndarray):
    """Convert one RGB frame to a self-owned :class:`QImage`.

    ``QImage`` does not take ownership of a Python buffer, so the copy is not
    optional: without it the image points at freed memory the moment the
    array goes out of scope.
    """
    from PySide6.QtGui import QImage

    height, width = frame.shape[:2]
    contiguous = np.ascontiguousarray(frame, dtype=np.uint8)
    image = QImage(
        contiguous.tobytes(), width, height, width * 3,
        QImage.Format_RGB888,
    )
    return image.copy()


def from_qimage(image) -> np.ndarray:
    """Convert a :class:`QImage` back to an RGB array.

    The inverse of :func:`to_qimage`, so what is actually on screen can be
    measured by the same rule the source frames were measured by — the scan
    lines are padded to a 4-byte boundary, hence the stride arithmetic.
    """
    from PySide6.QtGui import QImage

    converted = image.convertToFormat(QImage.Format_RGB888)
    width = converted.width()
    height = converted.height()
    stride = converted.bytesPerLine()
    raw = bytes(converted.constBits())[:stride * height]
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, stride)
    return rows[:, :width * 3].reshape(height, width, 3).copy()


__all__ = [
    "BACKGROUND_LEVEL",
    "from_qimage",
    "CACHE_SIZE",
    "MIN_NEIGHBOURS",
    "drop_specks",
    "FIELD_BOX",
    "FIELD_PAD",
    "FIELD_RADIUS",
    "MAX_FILL",
    "MIN_FILL",
    "SOURCE_SIZE",
    "TARGET_FILL",
    "ZoomedAnimation",
    "chrome_mask",
    "clear_cache",
    "content_bounds",
    "content_extent",
    "content_mask",
    "field_geometry",
    "field_ring_mask",
    "read_frames",
    "source_content_extent",
    "to_qimage",
    "zoom_frames",
    "zoomed_animation",
]
