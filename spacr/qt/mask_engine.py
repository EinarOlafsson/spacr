"""
Pure-Python backend for the Qt make-masks screen.

Image + mask I/O and the label-mutation helpers the screen above it calls
for everything that is not a brush stroke: fill, relabel, invert, remove
small, the size/intensity object filter, Otsu detection, and the magic
wand. Nothing here touches Qt, so every edit the screen offers can be
driven and asserted without a display.

Two things about how a mask leaves here are load-bearing:

**Object ids survive a save.** :func:`save_mask` writes what
:func:`canonical_labels` produces, and that preserves the id every object
already had. Renumbering the components of an edited mask would silently
re-key it against the measurements, the tracks and the crops made from the
segmentation it came from -- erasing object 7 of 20 would slide 8..20 down
by one and every downstream table would then name different cells.

**A hand-edited mask says so.** :func:`save_mask` writes the artefact's
:class:`spacr.curation.CurationLog` beside it, the same append-only ledger
:mod:`spacr.napari_bridge` and :mod:`spacr.qt.curation_tool` write, so
:func:`spacr.curation.is_curated` can tell a curated mask from one the
pipeline produced. A curated mask that is byte-indistinguishable from a
segmented one cannot be reproduced from the settings that made it.
"""
from __future__ import annotations

import os
from collections import deque
from typing import List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import binary_fill_holes, label

from ..curation import CurationLog
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
    fill_val = 255 if action == "add" else 0
    while q and added < max_pixels:
        cx, cy = q.popleft()
        if not (0 <= cx < image.shape[1] and 0 <= cy < image.shape[0]):
            continue
        if visited[cy, cx]:
            continue
        visited[cy, cx] = True
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
