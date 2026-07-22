"""
Pure-Python backend for the Qt make-masks screen.

Mirrors the image + mask I/O and label-mutation helpers from
`spacr.gui_elements.ModifyMaskApp`, without touching Tk. The Qt screen
above this reads/writes `self.image` and `self.mask` directly and calls
these helpers for the non-brush operations (fill / relabel / invert /
remove small).
"""
from __future__ import annotations

import os
from collections import deque
from typing import List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import binary_fill_holes, label


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


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
    - Both are returned as uint16 / uint8 arrays (image / mask).
    """
    image_path = os.path.join(folder, filename)
    image = imageio.imread(image_path)
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = image[..., :3]
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 1:
        image = np.squeeze(image, axis=-1)
    if image.dtype != np.uint16:
        max_val = float(image.max()) if image.size else 1.0
        if max_val <= 0:
            max_val = 1.0
        image = (image / max_val * 65535.0).astype(np.uint16)

    mask_path = os.path.join(folder, "masks", filename)
    if os.path.isfile(mask_path):
        mask = imageio.imread(mask_path)
        if mask.dtype != np.uint8:
            m = float(mask.max()) if mask.size else 1.0
            if m <= 0:
                m = 1.0
            mask = (mask / m * 255.0).astype(np.uint8)
    else:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
    return image, mask


def save_mask(folder: str, filename: str, mask: np.ndarray) -> str:
    """Relabel connected components and write to <folder>/masks/<name>.tif.

    Returns the absolute save path.
    """
    save_dir = os.path.join(folder, "masks")
    os.makedirs(save_dir, exist_ok=True)
    labeled, _ = label(mask > 0)
    stem = os.path.splitext(filename)[0]
    save_path = os.path.join(save_dir, stem + ".tif")
    imageio.imwrite(save_path, labeled.astype(np.uint16))
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
