"""Pure-Python mask editing and persistence for the Qt Make Masks screen.

This module provides image and mask I/O plus non-brush label operations,
including fill, relabel, inversion, size and intensity filtering, Otsu
detection, and magic-wand selection. It has no Qt dependency, so the editing
operations can be tested without a display.

:func:`save_mask` passes labels through :func:`canonical_labels`, which
preserves existing nonzero object identifiers rather than renumbering
connected components. This maintains correspondence with measurements,
tracks, and crops keyed by those identifiers.

Saving also writes the artifact's :class:`spacr.curation.CurationLog`
sidecar, consistent with :mod:`spacr.napari_bridge` and
:mod:`spacr.qt.curation_tool`. The sidecar allows
:func:`spacr.curation.is_curated` to distinguish manually edited masks from
pipeline-generated masks.
"""
from __future__ import annotations

import json
import os
from collections import deque
from typing import List, NamedTuple, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import binary_fill_holes, label

from ..curation import LOG_SUFFIX, CurationLog
from ..tiff_io import write_tiff


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

#: What a ledger created by this screen records as having made the edits.
CURATION_SOURCE = "spacr-qt make_masks"

#: Eight-connectivity: two objects touching only at a corner are one blob to
#: the eye and must be one object to the label image too, or a hand-drawn
#: diagonal stroke arrives on disk as a string of separate cells.
_EIGHT = np.ones((3, 3), dtype=np.uint8)


def list_images(folder: str) -> List[str]:
    """Return filenames of image files in `folder`, sorted, or []."""
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS)
    )


def load_image_and_mask(folder: str, filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image and its accompanying mask (from `folder/masks/`).

    - Multi-channel images are collapsed to grayscale via BT.601 weights.
    - Missing masks are created as zeros of the image shape.
    - Images are returned as uint16; masks preserve uint8/uint16 label IDs.
    - A mask saved by :func:`save_mask` is found even when the source image
      had a non-TIFF extension.

    :raises ValueError: for unsupported dimensions or an image/mask shape
        mismatch.
    """
    image_path = os.path.join(folder, filename)
    image = imageio.imread(image_path)
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = np.squeeze(image, axis=-1)
        elif image.shape[2] == 4:
            image = image[..., :3]
            image = np.dot(
                image, [0.2989, 0.5870, 0.1140]
            )
        elif image.shape[2] == 3:
            image = np.dot(
                image, [0.2989, 0.5870, 0.1140]
            )
        else:
            raise ValueError(
                f"Unsupported channel count {image.shape[2]} in {image_path}; "
                "expected grayscale, RGB, or RGBA."
            )
    if image.ndim != 2:
        raise ValueError(
            f"Unsupported image shape {image.shape} in {image_path}; "
            "Make Masks expects one 2-D field."
        )
    if not np.all(np.isfinite(image)):
        raise ValueError(f"Image contains non-finite values: {image_path}")
    if image.size and float(image.min()) < 0:
        raise ValueError(f"Image contains negative intensities: {image_path}")
    if image.dtype != np.uint16:
        max_val = float(image.max()) if image.size else 1.0
        if max_val <= 0:
            max_val = 1.0
        image = (image / max_val * 65535.0).astype(np.uint16)

    mask_dir = os.path.join(folder, "masks")
    stem = os.path.splitext(filename)[0]
    candidates = [
        os.path.join(mask_dir, filename),
        os.path.join(mask_dir, stem + ".tif"),
        os.path.join(mask_dir, stem + ".tiff"),
    ]
    mask_path = next((path for path in candidates if os.path.isfile(path)), "")
    if mask_path:
        mask = imageio.imread(mask_path)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)
        if mask.ndim != 2:
            raise ValueError(
                f"Unsupported mask shape {mask.shape} in {mask_path}; "
                "expected a 2-D label image."
            )
        if mask.shape != image.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape "
                f"{image.shape} for {filename}."
            )
        if not np.issubdtype(mask.dtype, np.integer):
            if not np.all(np.isfinite(mask)):
                raise ValueError(f"Mask contains non-finite values: {mask_path}")
            if np.any(mask < 0) or np.any(mask != np.floor(mask)):
                raise ValueError(
                    f"Mask must contain non-negative integer labels: {mask_path}"
                )
        maximum = int(mask.max()) if mask.size else 0
        if maximum > np.iinfo(np.uint16).max:
            raise ValueError(
                f"Mask label {maximum} exceeds uint16 capacity: {mask_path}"
            )
        mask = mask.astype(np.uint8 if maximum <= 255 else np.uint16)
    else:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
    return image, mask


def mask_save_path(folder: str, filename: str) -> str:
    """Where this field's mask is written -- and where its ledger sits.

    :func:`load_image_and_mask` will accept a mask under the image's own
    extension, but everything :func:`save_mask` writes lands on
    ``<folder>/masks/<stem>.tif``. The ledger is keyed on the file that was
    actually written, so both have to agree on one name; ask here rather
    than rebuilding it at each call site.
    """
    stem = os.path.splitext(filename)[0]
    return os.path.join(folder, "masks", stem + ".tif")


def canonical_labels(mask: np.ndarray) -> np.ndarray:
    """Return ``mask`` as uint16 labels, keeping every id it already had.

    The old behaviour here was ``label(mask > 0)``, which renumbers the
    connected components 1..N on every save. That throws away the identity
    of every object: erase object 7 of 20 and objects 8..20 each slide down
    by one, so the saved mask no longer keys against the measurements, the
    crops or the tracks derived from the segmentation it was edited from.

    So ids are kept, with two things settled:

    * **A mask with one foreground value carries no ids to keep.** That is
      what a purely brush-painted mask looks like -- every stroke writes the
      same value -- and it is a binary image, not a label image. Its
      components are numbered 1..N, which loses nothing.
    * **A label that names two separated blobs is split.** One id must mean
      one object. The largest piece keeps the id and the rest are given the
      smallest ids not already in use, so painting a second blob with the
      brush over a real segmentation adds an object instead of extending a
      distant one.
    """
    m = np.asarray(mask)
    values = np.unique(m[m > 0])
    if values.size <= 1:
        labeled, _ = label(m > 0, structure=_EIGHT)
        return labeled.astype(np.uint16)

    out = m.astype(np.int64, copy=True)
    used = {int(v) for v in values}
    candidate = 1
    for value in values:
        pieces, count = label(m == value, structure=_EIGHT)
        if count <= 1:
            continue
        areas = np.bincount(pieces.ravel())
        keep = int(np.argmax(areas[1:])) + 1
        for piece in range(1, count + 1):
            if piece == keep:
                continue
            while candidate in used:
                candidate += 1
            out[pieces == piece] = candidate
            used.add(candidate)
    top = int(out.max()) if out.size else 0
    if top > np.iinfo(np.uint16).max:
        # Wrapping would fuse object 65536 with object 0 — background —
        # and lose it silently. A mask is uint16 everywhere in spaCR, so
        # this is a mask that cannot be written, not one to truncate.
        raise ValueError(
            f"mask carries label {top}, past what a uint16 mask can hold.")
    return out.astype(np.uint16)


def save_mask(folder: str, filename: str, mask: np.ndarray,
              log: Optional[CurationLog] = None) -> str:
    """Write the mask to ``<folder>/masks/<stem>.tif`` and return that path.

    Object ids are preserved -- see :func:`canonical_labels` for what that
    costs and why the alternative is worse.

    :param folder: field directory under which the ``masks`` directory is
        created.
    :param filename: source image name; its extension is discarded and its
        stem becomes the TIFF mask name.
    :param mask: label image to canonicalise and write. Existing multi-label
        object identifiers are retained where possible.
    :param log: the session's :class:`spacr.curation.CurationLog` for this
        field. Given one holding at least one edit, it is written to
        ``<mask>.curation.json`` beside the mask, so
        :func:`spacr.curation.is_curated` reports the saved mask as
        hand-edited. The log is written whole, so it must have been seeded
        from :meth:`CurationLog.read_beside` to keep earlier sessions'
        entries -- which is what the screen does on load. A log with no
        edits writes no sidecar: a session that opened the editor and
        painted nothing has not curated anything, and a ledger that exists
        for every mask ever opened answers no question.
    """
    save_path = mask_save_path(folder, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    write_tiff(save_path, canonical_labels(mask))
    if log is not None and len(log):
        if not log.artifact:
            log.artifact = save_path
        log.write_beside(save_path)
    return save_path


def normalize_uint16(image: np.ndarray,
                     lower_pct: float = 1.0,
                     upper_pct: float = 99.9) -> np.ndarray:
    """Return image clipped + rescaled to its dtype's full range."""
    if not image.size:
        return image
    lo = np.percentile(image, lower_pct)
    hi = np.percentile(image, upper_pct)
    if hi <= lo:
        hi = lo + 1
    out = np.clip(image, lo, hi)
    out = (out - lo) / (hi - lo)
    max_val = float(np.iinfo(image.dtype).max)
    return (out * max_val).astype(image.dtype)


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a colorized label mask onto a grayscale image, uint8 RGB."""
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    m = mask.astype(np.int32)
    max_label = int(np.max(m)) if m.size else 0
    rng = np.random.default_rng(0)
    colors = rng.integers(30, 255, size=(max_label + 1, 3), dtype=np.uint8)
    if max_label >= 0:
        colors[0] = [0, 0, 0]
    colored = colors[m]
    image_8bit = (image.astype(np.float32) / 256.0).clip(0, 255).astype(np.uint8)
    combined = np.where(
        m[..., None] > 0,
        np.clip(image_8bit * (1 - alpha) + colored * alpha, 0, 255),
        image_8bit,
    ).astype(np.uint8)
    return combined


# ---------------------------------------------------------------------------
# Mask edits — brush / erase / object-level ops
# ---------------------------------------------------------------------------

def paint_disk(mask: np.ndarray, cx: int, cy: int, radius: int,
               value: int = 255) -> None:
    """In-place stamp a filled square (radius half-width) at (cx, cy)."""
    if radius < 1:
        radius = 1
    h, w = mask.shape[:2]
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius)
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = value


def paint_line(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int,
               radius: int, value: int = 255) -> None:
    """In-place stamp a line of disks between two points (Bresenham)."""
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        paint_disk(mask, x, y, radius, value)
        if x == x1 and y == y1:
            return
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


# ---------------------------------------------------------------------------
# Region tools — the free-form outline and the dividing line
# ---------------------------------------------------------------------------

#: Width, in image pixels, of the cut a divide draws through an object.
#:
#: Not cosmetic. Everything here calls two pixels that touch only at a
#: corner one object (:data:`_EIGHT`), so a one-pixel-wide cut across a
#: diagonal leaves the two halves corner-to-corner and they are still one
#: blob — the split does not take. Measured on a disk over 60 orientations:
#: a 1.0 px cut failed to separate in 58 of them, 1.5 px separated in all
#: 1800 orientation/offset combinations tried, and anything wider only eats
#: more of the object (1.5 px costs ~90 px of a 2800 px object, 2.0 px ~120).
DIVIDE_CUT_WIDTH = 1.5


def next_label(mask: np.ndarray) -> int:
    """The id to give the next object drawn on ``mask``: one past its top.

    Above the maximum rather than the lowest free id, because ids are what
    the ledger, the measurements and the crops name objects by. Handing a
    new object the id of one that was deleted makes two different cells
    share a name across a session, and nothing downstream can tell them
    apart afterwards.
    """
    return (int(mask.max()) if mask.size else 0) + 1


def _fit_label_width(out: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Cast a working int64 mask back down, widening only when it must.

    Same rule as :func:`combine_masks`: the width follows the values. Keeping
    uint8 for a mask that has just been given label 256 would wrap it round
    to 0 and the new object would vanish into the background.
    """
    top = int(out.max()) if out.size else 0
    if top > np.iinfo(np.uint16).max:
        raise ValueError(
            f"mask needs label {top}, past what a uint16 mask can hold.")
    if np.issubdtype(like.dtype, np.integer) and top <= np.iinfo(like.dtype).max:
        return out.astype(like.dtype)
    return out.astype(np.uint8 if top <= 255 else np.uint16)


def fill_polygon(mask: np.ndarray, points, label_value: Optional[int] = None):
    """Fill a traced outline as ONE object; return ``(mask, label)``.

    This is the tool a brush is not. A brush stamps disks along the path, so
    tracing a cell's rim with it labels the rim and leaves the middle
    background; ``draw`` closes the path (last point back to the first) and
    fills what it encloses, so one gesture produces one solid object with
    one id. Anything already labelled inside the outline is overwritten,
    which is the point: the outline asserts "all of this is one object".

    :param mask: existing label image to copy and edit; labels outside the
        enclosed pixels are preserved and the dtype widens when the new id
        requires it.
    :param points: the traced path as image-pixel ``(x, y)`` pairs. A path
        that encloses less than one pixel -- two points, or a straight line
        traced back over itself -- is returned unchanged with label 0. It is
        a gesture that enclosed nothing, and the alternative is an object a
        pixel wide that the user then has to find and delete.
    :param label_value: the id to give it; by default :func:`next_label`.
    """
    from skimage.draw import polygon as _polygon

    pts = np.asarray(list(points), dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return mask.copy(), 0
    # Shoelace area of the closed path. skimage's polygon() hands back the
    # traced pixels themselves for a degenerate outline, which would make a
    # straight drag into a hairline "object".
    x, y = pts[:, 0], pts[:, 1]
    if abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) < 1.0:
        return mask.copy(), 0
    rows, cols = _polygon(pts[:, 1], pts[:, 0], shape=mask.shape[:2])
    if not len(rows):
        return mask.copy(), 0
    value = int(next_label(mask) if label_value is None else label_value)
    out = mask.astype(np.int64, copy=True)
    out[rows, cols] = value
    return _fit_label_width(out, mask), value


def _segment_band(shape, p0, p1, width: float) -> np.ndarray:
    """Boolean mask of the pixels within ``width``/2 of the segment p0-p1.

    A distance band rather than a rasterised line: the sampled-line cut the
    standalone curation tool uses (400 points between the ends) both leaves
    gaps on a segment longer than 400 px and is one pixel wide wherever it
    lands, and a one-pixel cut does not separate — see
    :data:`DIVIDE_CUT_WIDTH`.
    """
    height, width_px = shape[:2]
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    half = max(0.5, float(width) / 2.0)
    # Only the segment's bounding box can be within half a width of it, so
    # the distance is computed there instead of over the whole field.
    lo_x = max(0, int(np.floor(min(x0, x1) - half)))
    hi_x = min(width_px, int(np.ceil(max(x0, x1) + half)) + 1)
    lo_y = max(0, int(np.floor(min(y0, y1) - half)))
    hi_y = min(height, int(np.ceil(max(y0, y1) + half)) + 1)
    band = np.zeros((height, width_px), dtype=bool)
    if hi_x <= lo_x or hi_y <= lo_y:
        return band
    yy, xx = np.mgrid[lo_y:hi_y, lo_x:hi_x].astype(np.float64)
    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq <= 0:
        t = np.zeros_like(xx)
    else:
        t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / length_sq, 0.0, 1.0)
    distance = np.hypot(xx - (x0 + t * dx), yy - (y0 + t * dy))
    band[lo_y:hi_y, lo_x:hi_x] = distance <= half
    return band


def divide_object(mask: np.ndarray, p0, p1,
                  width: float = DIVIDE_CUT_WIDTH):
    """Cut every object the segment crosses in two; return ``(mask, splits)``.

    ``splits`` is a list of ``(id_split, id_created)`` pairs, empty when the
    line separated nothing.

    Three decisions make the result usable:

    * **Only the objects the line actually crosses are touched.** The cut is
      clipped to them, so a line drawn past a neighbour leaves that
      neighbour's every pixel where it was. (The standalone tool relabels
      the whole field after cutting, which renumbers every other object in
      it — the same re-keying :func:`canonical_labels` exists to avoid.)
    * **The larger piece keeps the original id**, and the smaller pieces get
      fresh ones above the mask's top label. That is the rule
      :func:`canonical_labels` already applies when one id names two blobs,
      so dividing and then saving does not renumber anything, and the id
      stays on the piece that carries most of what it used to name.
    * **A line that does not separate an object leaves it alone.** Stopping
      halfway across would otherwise carve a groove into the object and
      call it a division; treating it as a miss means the gesture can just
      be redrawn.
    """
    band = _segment_band(mask.shape, p0, p1, width)
    if not band.any():
        return mask.copy(), []
    crossed = [int(v) for v in np.unique(mask[band]) if int(v) > 0]
    if not crossed:
        return mask.copy(), []
    out = mask.astype(np.int64, copy=True)
    free_id = next_label(mask)
    splits = []
    for source in crossed:
        body = out == source
        remainder = body & ~band
        pieces, count = label(remainder, structure=_EIGHT)
        if count < 2:
            continue                      # the line stopped short: not a cut
        areas = np.bincount(pieces.ravel())
        keeps = int(np.argmax(areas[1:])) + 1
        out[body] = 0                     # drop the cut pixels with the rest
        out[pieces == keeps] = source
        for piece in range(1, count + 1):
            if piece == keeps:
                continue
            out[pieces == piece] = free_id
            splits.append((source, free_id))
            free_id += 1
    if not splits:
        return mask.copy(), []
    return _fit_label_width(out, mask), splits

def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes inside True regions; returns a relabeled mask."""
    binary = mask > 0
    filled = binary_fill_holes(binary)
    labeled, _ = label(filled)
    return labeled.astype(mask.dtype)


def relabel_objects(mask: np.ndarray) -> np.ndarray:
    """Return a mask whose connected components are labeled 1..N."""
    labeled, _ = label(mask > 0)
    return labeled.astype(mask.dtype)


def clear_mask(mask: np.ndarray) -> np.ndarray:
    """Return an all-zero array shaped like ``mask``."""
    return np.zeros_like(mask)


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Return the mask with foreground/background flipped and relabeled."""
    out = np.where(mask > 0, 0, 1).astype(mask.dtype)
    labeled, _ = label(out)
    return labeled.astype(mask.dtype)


def remove_small_objects(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected components with area < min_area (in pixels)."""
    if min_area <= 0:
        return mask.copy()
    labeled, n = label(mask > 0)
    if n == 0:
        return mask.copy()
    counts = np.bincount(labeled.ravel())
    keep = np.zeros_like(counts, dtype=bool)
    for i in range(1, len(counts)):
        if counts[i] >= min_area:
            keep[i] = True
    filtered = keep[labeled]
    out = np.where(filtered, mask, 0)
    labeled, _ = label(out > 0)
    return labeled.astype(mask.dtype)


def erase_object_at(mask: np.ndarray, x: int, y: int) -> np.ndarray:
    """Zero out the object under (x, y). No-op if no object there."""
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
        return mask
    label_to_remove = int(mask[y, x])
    if label_to_remove <= 0:
        return mask
    out = mask.copy()
    out[out == label_to_remove] = 0
    return out


def erase_object_in_place(mask: np.ndarray, x: int, y: int) -> int:
    """Zero the object under (x, y) *in place*; return the id removed, or 0.

    The copy-and-return :func:`erase_object_at` is right for a single click,
    where one edit is one undo step. A right-button sweep is dozens of move
    events and one thing the user did, so it deletes in place against a
    single pre-sweep snapshot: copying a 16-bit field per mouse event would
    both stutter and put every object of the sweep on its own undo step.

    The returned id is what the ledger records as the sweep's targets.
    """
    height, width = mask.shape[:2]
    if not (0 <= y < height and 0 <= x < width):
        return 0
    label_to_remove = int(mask[y, x])
    if label_to_remove <= 0:
        return 0
    mask[mask == label_to_remove] = 0
    return label_to_remove


def relative_tolerance(image: np.ndarray, percent: float) -> float:
    """Magic-wand tolerance as ``percent`` of ``image``'s intensity range.

    An absolute tolerance is not a portable setting. The value that grabs
    one nucleus in an 8-bit field (range 0..255) selects nothing at all in
    a 16-bit one (range 0..65535), and the one tuned for 16-bit floods the
    entire 8-bit frame. A percentage of *this* image's own range means one
    number behaves the same on both.

    The floor of 1.0 keeps the wand usable on a flat field: a range of zero
    would otherwise give a tolerance of zero, and a tolerance of zero fills
    only pixels exactly equal to the seed.
    """
    values = np.asarray(image, dtype=np.float32)
    if not values.size:
        return 1.0
    span = float(values.max() - values.min())
    return max(1.0, (float(percent) / 100.0) * span)


def filter_objects(mask: np.ndarray, image: np.ndarray, *,
                   min_area: int = 0, max_area: int = 0,
                   min_intensity: float = 0.0,
                   max_intensity: float = 0.0) -> Tuple[np.ndarray, List[int]]:
    """Drop objects outside the size/intensity bounds. Each bound is off at 0.

    Area is the object's pixel count; intensity is its MEAN value on the
    *raw* image, not on the contrast-stretched display -- the display
    percentiles are a viewing choice and a filter that moved when you
    changed them would not be reproducible.

    Zero means "no bound on this side" rather than "reject everything",
    which is what makes all four bounds independently optional: a minimum
    area of 0 would exclude nothing anyway, so the value is free to carry
    the off switch.

    :returns: ``(mask, dropped)`` -- a new mask with the failing objects
        zeroed and the sorted ids that were dropped. The ids are what the
        curation ledger records, so an automatic filter is as traceable as a
        click. Nothing to do returns the original array untouched and an
        empty list.
    """
    bounds = (int(min_area or 0), int(max_area or 0),
              float(min_intensity or 0.0), float(max_intensity or 0.0))
    lo_area, hi_area, lo_int, hi_int = bounds
    if not any(bounds) or mask is None or not mask.size or not mask.max():
        return mask, []

    from skimage.measure import regionprops

    grey = np.asarray(image, dtype=np.float32)
    if grey.ndim == 3:
        grey = grey.mean(axis=2)
    # Measured on the canonical labelling, not on the raw array: two
    # separate blobs a brush painted with the same value are one region to
    # regionprops, and their combined area and mean intensity describe
    # neither of them.
    labels = canonical_labels(mask)
    dropped: List[int] = []
    for region in regionprops(labels.astype(np.int32), intensity_image=grey):
        area = int(region.area)
        # scikit-image renamed mean_intensity to intensity_mean and warns on
        # the old spelling; both names are live across the versions spaCR
        # supports, so ask for the new one and fall back.
        mean = float(region.intensity_mean
                     if hasattr(region, "intensity_mean")
                     else region.mean_intensity)
        if ((lo_area and area < lo_area) or (hi_area and area > hi_area)
                or (lo_int and mean < lo_int) or (hi_int and mean > hi_int)):
            dropped.append(int(region.label))
    if not dropped:
        return mask, []
    out = mask.copy()
    out[np.isin(labels, dropped)] = 0
    return out, sorted(dropped)


def connected_instances(binary: np.ndarray, min_area: int = 0) -> np.ndarray:
    """Label every separated foreground region as its own object.

    :param binary: array whose truthy pixels are foreground. Regions touching
        diagonally are connected under the editor's eight-neighbour rule.
    :param min_area: regions smaller than this are dropped rather than
        labelled, so a detection does not hand back a field of single-pixel
        speckles for the user to delete by hand.
    """
    components, count = label(np.asarray(binary, dtype=bool), structure=_EIGHT)
    out = np.zeros(components.shape, dtype=np.int32)
    next_id = 1
    for old in range(1, count + 1):
        region = components == old
        if int(region.sum()) >= int(min_area):
            out[region] = next_id
            next_id += 1
    return out


def otsu_instances(image: np.ndarray, *, bright: bool = True,
                   min_area: int = 0) -> np.ndarray:
    """Threshold ``image`` at Otsu's level and label what is left.

    :param image: numeric intensity image. It is converted to float32 before
        the threshold is estimated and must contain at least one pixel.
    :param bright: objects are brighter than background (fluorescence).
        False takes the dark side instead, for a brightfield or a
        stained-plaque image where the objects absorb.
    :param min_area: passed to :func:`connected_instances`.
    :raises ValueError: on an empty image, which has no threshold to find.
    """
    from skimage.filters import threshold_otsu

    values = np.asarray(image, dtype=np.float32)
    if not values.size:
        raise ValueError("Otsu needs an image; this one is empty.")
    threshold = float(threshold_otsu(values))
    binary = (values > threshold) if bright else (values < threshold)
    return connected_instances(binary, min_area=min_area)


def combine_masks(old: np.ndarray, new: np.ndarray,
                  mode: str = "replace") -> np.ndarray:
    """Fold a fresh detection into an existing mask.

    :param old: existing label image used as the merge base; ignored when
        ``mode`` is ``"replace"``.
    :param new: newly detected label image. In merge mode its positive labels
        are offset above ``old`` and copied only into background pixels.
    :param mode: ``"replace"`` -- the detection is the mask, and whatever
        was there is gone. ``"merge"`` -- keep every existing object and add
        the detected ones only where nothing is labelled yet, with fresh ids
        above the existing maximum. Merge never overwrites or splits an
        object that was curated by hand, which is the point of offering the
        choice: a detection run halfway through an editing session should
        not be able to silently undo the first half of it.
    :raises ValueError: for an unknown mode, rather than quietly picking
        one and discarding the user's edits.
    """
    if mode not in ("replace", "merge"):
        raise ValueError(
            f"combine mode must be 'replace' or 'merge', not {mode!r}")
    incoming = np.asarray(new).astype(np.int64)
    if mode == "replace":
        out = incoming
    else:
        base = int(old.max()) if old.size else 0
        out = old.astype(np.int64, copy=True)
        added = np.where(incoming > 0, incoming + base, 0)
        free = out == 0
        out[free] = added[free]
    # Width follows the values, as it does everywhere else a mask is made
    # here: merging 300 detected objects into a uint8 mask and keeping uint8
    # would wrap object 300 round to 44 and silently fuse it with another.
    top = int(out.max()) if out.size else 0
    if top > np.iinfo(np.uint16).max:
        raise ValueError(
            f"combined mask needs label {top}, past uint16; "
            "raise the minimum area so the detection makes fewer objects.")
    return out.astype(np.uint8 if top <= 255 else np.uint16)


# ---------------------------------------------------------------------------
# Magic wand — flood-fill by intensity tolerance (mirrors ModifyMaskApp)
# ---------------------------------------------------------------------------

#: How many pixels the wand may EXAMINE per pixel it is allowed to change.
#:
#: The budget is on WORK, and it has to accommodate a case the change
#: budget deliberately does not bound: re-wanding an object the mask
#: already owns, with a wider tolerance, to grow it. Crossing owned
#: pixels costs nothing against ``max_pixels`` ON PURPOSE -- otherwise
#: the second click would stop at the first owned pixel and do nothing --
#: so the walk is bounded here instead.
VISIT_BUDGET_FACTOR = 4

#: The smallest visit budget, whatever ``max_pixels`` is, and the number
#: that actually matters.
#:
#: It has to be BIGGER THAN ANY OBJECT SOMEBODY RE-WANDS and smaller than
#: a frame. A hundred thousand is a 316x316 region, larger than any single
#: object in a field of cells, and it caps the pathological case -- a
#: mis-click on uniform background -- at about 0.7 s instead of the 4.8 s
#: measured on an 800x800 field or the half-minute at 2048x2048.
#:
#: It cannot be tight. A pixel count cannot tell "usefully growing a large
#: object" from "clicked on the background", because both walk until
#: tolerance stops them and on a uniform field neither does. So this is
#: set to keep every real fill working and to make the wrong click a
#: hitch rather than a hang, which is the honest trade rather than a
#: pretence that the two can be separated.
VISIT_BUDGET_FLOOR = 100_000


def magic_wand(
    image: np.ndarray,
    mask: np.ndarray,
    seed_x: int,
    seed_y: int,
    tolerance: float,
    max_pixels: int = 100_000,
    action: str = "add",
) -> np.ndarray:
    """BFS flood-fill from (seed_x, seed_y) filling pixels whose intensity
    is within `tolerance` (L2 distance) of the seed. Writes 255 (add) or
    0 (erase) into the returned mask copy.
    """
    if not (0 <= seed_y < image.shape[0] and 0 <= seed_x < image.shape[1]):
        return mask
    out = mask.copy()
    initial = image[seed_y, seed_x].astype(np.float32)
    visited = np.zeros(image.shape[:2], dtype=bool)
    q = deque([(seed_x, seed_y)])
    added = 0
    # A SECOND BUDGET, ON WORK RATHER THAN ON CHANGES.
    #
    # `added` counts only pixels that CHANGE state, which is the budget a
    # user thinks in -- "fill at most this many". But a flood that changes
    # nothing never increments it, so `added < max_pixels` stayed true
    # forever and the search walked the entire frame. Erasing where the
    # mask is already empty, or adding over ground the mask already owns,
    # is the most ordinary wrong click there is: measured at 4.8 s on an
    # 800x800 field with max_pixels=100, and roughly half a minute at
    # 2048x2048, with the GUI unresponsive and no way to cancel.
    #
    # So visits are bounded too. The multiplier is generous on purpose --
    # a legitimate fill examines its region AND the out-of-tolerance
    # perimeter around it, and a thin structure can have as much perimeter
    # as area -- so this stops the pathological case without shortening
    # any fill a user would recognise. The floor keeps small budgets
    # workable, since a max_pixels of 10 still needs room to look around.
    examined = 0
    visit_budget = max(VISIT_BUDGET_FACTOR * max_pixels,
                       max_pixels + VISIT_BUDGET_FLOOR)
    fill_val = 255 if action == "add" else 0
    while q and added < max_pixels and examined < visit_budget:
        cx, cy = q.popleft()
        if not (0 <= cx < image.shape[1] and 0 <= cy < image.shape[0]):
            continue
        if visited[cy, cx]:
            continue
        visited[cy, cx] = True
        examined += 1
        cur = image[cy, cx].astype(np.float32)
        if float(np.linalg.norm(cur - initial)) > tolerance:
            continue
        if out[cy, cx] == 0 and action == "add":
            added += 1
        elif out[cy, cx] > 0 and action == "erase":
            added += 1
        out[cy, cx] = fill_val
        if added >= max_pixels:
            break
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and not visited[ny, nx]:
                q.append((nx, ny))
    return out


# ---------------------------------------------------------------------------
# Undo history — small bounded ring of mask snapshots
# ---------------------------------------------------------------------------

class MaskHistory:
    """Bounded undo/redo stack of mask arrays. Deep-copies on push so
    callers can mutate in place without corrupting older snapshots."""

    def __init__(self, capacity: int = 20):
        """Prepare an empty history with a bounded snapshot capacity.

        :param capacity: max snapshots kept in the undo (and redo) stack.
        """
        self.capacity = max(1, int(capacity))
        self._undo: deque = deque(maxlen=self.capacity)
        self._redo: deque = deque(maxlen=self.capacity)

    def clear(self) -> None:
        """Discard every snapshot from both the undo and redo stacks."""
        self._undo.clear()
        self._redo.clear()

    def push(self, mask: np.ndarray) -> None:
        """Store a deep-copy of ``mask`` and drop any redo history."""
        self._undo.append(np.array(mask, copy=True))
        self._redo.clear()

    def head(self) -> Optional[np.ndarray]:
        """The newest snapshot — what an edit in progress started from.

        Returned as held, not copied, so a caller that only wants to diff
        against it does not pay for a copy of a 16-bit field on every mouse
        release. It is already a private copy of whatever was pushed, so
        reading it cannot disturb the history; writing to it would.
        """
        return self._undo[-1] if self._undo else None

    def can_undo(self) -> bool:
        """Return True when at least one prior snapshot is available to undo to."""
        return len(self._undo) >= 2

    def can_redo(self) -> bool:
        """Return True when the redo stack has a snapshot to restore."""
        return bool(self._redo)

    def undo(self) -> Optional[np.ndarray]:
        """Pop the top snapshot, save it to the redo stack, and return the
        previous snapshot (i.e. one step back). None if not possible."""
        if not self.can_undo():
            return None
        current = self._undo.pop()
        self._redo.append(current)
        return np.array(self._undo[-1], copy=True)

    def redo(self) -> Optional[np.ndarray]:
        """Restore the most-recently-undone snapshot, or ``None`` if empty."""
        if not self._redo:
            return None
        snap = self._redo.pop()
        self._undo.append(np.array(snap, copy=True))
        return np.array(snap, copy=True)


# ---------------------------------------------------------------------------
# Recrop — cutting one field into the several fields it should have been
# ---------------------------------------------------------------------------
#
# Every other tool in this module edits the mask on the field in view.
# Recrop is the one that changes WHICH field is in view: a staged crop that
# holds several cells, wells or plaques is not one training example, and
# curating it as though it were teaches the network that two objects are one
# picture. So the user boxes each one, every box becomes a field of its own
# carrying that region of BOTH the image and the draft mask, and the
# multi-object original is retired rather than curated.
#
# WHAT spaCR CAN AND CANNOT RETIRE. The Make Masks queue is a FOLDER:
# :func:`list_images` sorts the image files in it and the screen walks that
# list, with each mask at ``<folder>/masks/<stem>.tif``. spaCR does have a
# crop DATABASE -- ``png_list`` in ``measurements.db``, which is what the
# Annotate app and the classifiers read -- but it is keyed on each crop's
# absolute ``png_path`` and carries no lifecycle column: there is no field in
# it that can be set to "recropped", and no row that a screen reading a
# folder has any claim to rewrite. So the original CANNOT be marked retired
# in spaCR's database the way the standalone marks it in its status CSV.
#
# The nearest thing that is recoverable, and what these functions do, is to
# move the original out of the enumeration and leave every byte of it on
# disk: image, mask and curation ledger go into ``<folder>/recropped_originals/``
# (the mask keeping its ``masks/`` sub-layout), which :func:`list_images`
# does not descend into, and :data:`RECROP_MANIFEST` inside that folder
# records what was moved, which boxes were cut out of it and what the
# children were called. A recrop drawn wrong is undone by moving two files
# back; a dataset registered in ``png_list`` can be repointed from the
# manifest rather than from a guess.

#: Smallest side, in image pixels, a recrop box may have. A box smaller than
#: this is a mis-click or the tail of a drag that never really started, and
#: cutting it writes a field too small to hold the object it was aimed at.
RECROP_MIN_SIDE = 32

#: How much a new box may overlap one already cut out of this field, as
#: intersection over union. Above it the box is the SAME region drawn again,
#: not a second object, and it is refused rather than written: without that
#: refusal one object reached disk three times as three near-identical
#: fields, because nothing on screen said the first box had worked.
RECROP_MAX_OVERLAP = 0.5

#: Folder the retired originals are moved into, beside the images they were
#: enumerated with. Not a delete: see the note above.
RECROP_ARCHIVE_DIRNAME = "recropped_originals"

#: The record of every retirement, written inside the archive folder.
RECROP_MANIFEST = "recropped.json"

#: What separates a child crop's name from the field it was cut from:
#: ``<field>__r00``, ``<field>__r01``. Recropping a child re-uses the
#: original field's name rather than nesting, so a name says which field a
#: crop came from however many passes it took.
RECROP_INFIX = "__r"

#: The verb a recrop goes into the curation ledger under.
RECROP_KIND = "recrop"


class RecropRefused(ValueError):
    """Raised when a proposed recrop does not satisfy recropping rules.

    :ivar reason: Stable reason code: ``"no_field"``, ``"too_small"``, or
        ``"redraw"``.

    :param reason: the stable code above. Callers branch on it, so it must
        be one of the three rather than prose.
    :param message: what to say to the user. This is the exception's own
        message, so it is what an unhandled raise would print.
    """

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = str(reason)


def _ordered_box(shape, p0, p1) -> Tuple[int, int, int, int]:
    """``(x0, y0, x1, y1)`` for two dragged corners, clipped to ``shape``."""
    height, width = int(shape[0]), int(shape[1])
    x0, x1 = sorted((int(p0[0]), int(p1[0])))
    y0, y1 = sorted((int(p0[1]), int(p1[1])))
    return (max(0, min(width, x0)), max(0, min(height, y0)),
            max(0, min(width, x1)), max(0, min(height, y1)))


def box_overlap(a, b) -> float:
    """Intersection over union of two ``(x0, y0, x1, y1)`` boxes.

    :returns: A value from 0 for disjoint boxes to 1 for identical boxes.
    """
    ax0, ay0, ax1, ay1 = (int(v) for v in a[:4])
    bx0, by0, bx1, by1 = (int(v) for v in b[:4])
    inter = (max(0, min(ax1, bx1) - max(ax0, bx0))
             * max(0, min(ay1, by1) - max(ay0, by0)))
    union = ((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)
    return (inter / union) if union > 0 else 0.0


def recrop_box(shape, p0, p1, existing=()) -> Tuple[int, int, int, int]:
    """Validate and clip a rectangular recrop selection.

    :param shape: Image shape as ``(height, width)``. Coordinates outside the
        image are clipped to this extent.
    :param p0: First selection corner as ``(x, y)``.
    :param p1: Opposite selection corner as ``(x, y)``.
    :param existing: Previously accepted boxes. A selection whose intersection
        over union exceeds :data:`RECROP_MAX_OVERLAP` is rejected.
    :returns: Validated ``(x0, y0, x1, y1)`` coordinates.
    :raises RecropRefused: If a side is shorter than
        :data:`RECROP_MIN_SIDE` or the selection duplicates an existing box.
    """
    x0, y0, x1, y1 = _ordered_box(shape, p0, p1)
    if (x1 - x0) < RECROP_MIN_SIDE or (y1 - y0) < RECROP_MIN_SIDE:
        raise RecropRefused(
            "too_small",
            f"Recrop box is {x1 - x0}x{y1 - y0} px — at least "
            f"{RECROP_MIN_SIDE} px on each side is needed.")
    for other in existing:
        if box_overlap((x0, y0, x1, y1), other) > RECROP_MAX_OVERLAP:
            name = other[4] if len(other) > 4 else "an earlier box"
            raise RecropRefused(
                "redraw",
                f"That region is already cut out as {name} — it is saved, "
                "draw the next one.")
    return (x0, y0, x1, y1)


def cut_recrop(image: np.ndarray, mask: np.ndarray,
               box) -> Tuple[np.ndarray, np.ndarray]:
    """Extract an image region and its complete labelled objects.

    Objects touching the crop boundary are removed because their masks are
    incomplete. Remaining labels are renumbered consecutively from one.

    :param image: Source microscopy image.
    :param mask: Label image aligned with ``image``.
    :param box: Crop coordinates as ``(x0, y0, x1, y1)``.
    :returns: Cropped image and relabelled mask.
    """
    x0, y0, x1, y1 = (int(v) for v in box[:4])
    sub_image = np.ascontiguousarray(np.asarray(image)[y0:y1, x0:x1])
    sub_mask = np.ascontiguousarray(np.asarray(mask)[y0:y1, x0:x1])
    if sub_mask.size and sub_mask.max():
        edge = np.unique(np.concatenate([
            sub_mask[0, :], sub_mask[-1, :], sub_mask[:, 0], sub_mask[:, -1],
        ]))
        edge = edge[edge > 0]
        if edge.size:
            sub_mask = np.where(np.isin(sub_mask, edge), 0, sub_mask)
    relabelled, _ = label(sub_mask > 0, structure=_EIGHT)
    return sub_image, relabelled.astype(np.uint16)


def _recrop_base(filename: str) -> str:
    """The field name a child crop is named after.

    A recrop of a recrop is named after the ORIGINAL field, not after its
    parent: ``well_A1__r00`` recropped again yields ``well_A1__r03``, never
    ``well_A1__r00__r00``. Nesting would make the name grow with every pass
    while saying nothing more than the manifest already records.
    """
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    return stem.split(RECROP_INFIX)[0]


def recrop_child_name(folder: str, filename: str, ext: str = ".tif") -> str:
    """Return the next unused ``<field>__rNN`` filename.

    Names are checked against the image queue, mask directory, and recrop
    archive to prevent overwriting output from an earlier editing session.
    """
    base = _recrop_base(filename)
    archive = os.path.join(folder, RECROP_ARCHIVE_DIRNAME)
    index = 0
    while True:
        stem = f"{base}{RECROP_INFIX}{index:02d}"
        taken = [os.path.join(folder, "masks", stem + ".tif"),
                 os.path.join(archive, "masks", stem + ".tif")]
        taken += [os.path.join(d, stem + e)
                  for d in (folder, archive) for e in IMAGE_EXTS]
        if not any(os.path.exists(path) for path in taken):
            return stem + ext
        index += 1


class Recrop(NamedTuple):
    """Result of writing a recropped image and mask.

    :ivar name: Filename assigned to the recropped field.
    :ivar image_path: Path of the written image.
    :ivar mask_path: Path of the written label mask.
    :ivar n_objects: Number of complete labelled objects retained.
    :ivar box: Source coordinates as ``(x0, y0, x1, y1)``.
    """

    name: str
    image_path: str
    mask_path: str
    n_objects: int
    box: Tuple[int, int, int, int]


def write_recrop(folder: str, filename: str, image: np.ndarray,
                 mask: np.ndarray, box) -> "Recrop":
    """Write a recropped field, mask, and curation record.

    The image is stored as an unscaled uint16 TIFF beside the source images,
    and the relabelled mask is stored in ``<folder>/masks``. The curation
    record distinguishes deliberately removed boundary objects from missed
    segmentation objects.

    :param folder: Image-queue directory.
    :param filename: Source image filename.
    :param image: Source microscopy image.
    :param mask: Label image aligned with ``image``.
    :param box: Coordinates returned by :func:`recrop_box`.
    :returns: Filename and retained-object count for the new field.
    """
    x0, y0, x1, y1 = (int(v) for v in box[:4])
    sub_image, sub_mask = cut_recrop(image, mask, (x0, y0, x1, y1))
    child = recrop_child_name(folder, filename)
    image_path = os.path.join(folder, child)
    mask_path = mask_save_path(folder, child)
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    write_tiff(image_path, np.asarray(sub_image).astype(np.uint16))
    write_tiff(mask_path, sub_mask)
    log = CurationLog(mask_path, source=CURATION_SOURCE)
    log.append(RECROP_KIND, child,
               n_changed=int(np.count_nonzero(sub_mask)),
               parent=os.path.basename(str(filename)),
               box=[x0, y0, x1, y1],
               n_objects=int(sub_mask.max()))
    log.write_beside(mask_path)
    return Recrop(child, image_path, mask_path,
                  int(sub_mask.max()), (x0, y0, x1, y1))


def recrop_archive_dir(folder: str) -> str:
    """Return the archive directory for recropped source fields."""
    return os.path.join(folder, RECROP_ARCHIVE_DIRNAME)


def retire_recropped_original(folder: str, filename: str, *,
                              children=(), boxes=()) -> dict:
    """Archive a source field after recropped children have been created.

    The source image, mask, and curation ledger are moved to
    ``<folder>/recropped_originals`` and recorded in :data:`RECROP_MANIFEST`.
    This removes the multi-object source from the training queue without
    deleting it.

    :param folder: Image-queue directory.
    :param filename: Source image filename.
    :param children: Filenames created from the source field.
    :param boxes: Crop boxes corresponding to ``children``.
    :returns: Manifest record, including the original and archived paths.
    """
    archive = recrop_archive_dir(folder)
    mask_path = mask_save_path(folder, filename)
    moves = [
        (os.path.join(folder, filename),
         os.path.join(archive, os.path.basename(filename))),
        (mask_path, os.path.join(archive, "masks",
                                 os.path.basename(mask_path))),
        (mask_path + LOG_SUFFIX,
         os.path.join(archive, "masks",
                      os.path.basename(mask_path) + LOG_SUFFIX)),
    ]
    moved = []
    for source, target in moves:
        if not os.path.exists(source):
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.replace(source, target)
        moved.append([source, target])
    record = {
        "original": os.path.basename(str(filename)),
        "children": [str(c) for c in children],
        "boxes": [[int(v) for v in box[:4]] for box in boxes],
        "moved": moved,
    }
    _append_recrop_manifest(archive, record)
    return record


def _append_recrop_manifest(archive: str, record: dict) -> str:
    """Append one retirement to the archive's manifest; return its path.

    Read-modify-write of a list rather than a line per record, so the file
    is ordinary JSON that anything can open. A manifest that cannot be read
    is replaced rather than allowed to stop the retirement: the files
    themselves are the recovery, and the manifest is the map to them.
    """
    os.makedirs(archive, exist_ok=True)
    path = os.path.join(archive, RECROP_MANIFEST)
    records = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, list):
                records = loaded
        except (OSError, ValueError):
            records = []
    records.append(record)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=1)
    return path


def read_recrop_manifest(folder: str) -> List[dict]:
    """Return recrop-archive records in chronological order."""
    path = os.path.join(recrop_archive_dir(folder), RECROP_MANIFEST)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return []
    return loaded if isinstance(loaded, list) else []


def restore_recropped_original(folder: str, original: str) -> List[str]:
    """Restore an archived source field to the image queue.

    Existing archived files listed for ``original`` are moved back to their
    source locations. Recropped child fields are not modified.

    :param folder: Image-queue directory.
    :param original: Original source filename recorded in the manifest.
    :returns: Paths restored to the queue.
    """
    restored: List[str] = []
    name = os.path.basename(str(original))
    for record in reversed(read_recrop_manifest(folder)):
        if record.get("original") != name:
            continue
        for pair in record.get("moved") or ():
            if len(pair) != 2:
                continue
            source, target = pair
            if not os.path.exists(target) or os.path.exists(source):
                continue
            os.makedirs(os.path.dirname(source), exist_ok=True)
            os.replace(target, source)
            restored.append(source)
        break
    return restored
