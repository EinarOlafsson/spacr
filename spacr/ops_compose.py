"""The canonical nuclear map, composed one window at a time.

`ops_stitch` solves where every tile sits and deliberately does not hold the
pixels -- well A1 is 26,855 x 26,865, which is 1.4 GB as uint16 for one
channel, and every later phase would read through it. This module composes the
same well window by window instead, which is what segmentation wants anyway:
segment once, on the composite, a window at a time.

    A window that a segmenter can hold is also a window a composer can build,
    so nothing ever materialises the whole canvas. The join key the object
    numbering needs -- a plate-level object number in the well frame --
    survives because every window carries its own offset into that frame.

Averaging the overlaps gives a measured noise gain of about 1.15x on the
channel segmentation runs on. There is only one DAPI acquisition, so averaging
across cycles does not apply; the gain comes from the raster's 14% overlap,
where two tiles imaged the same nuclei.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, Mapping, Optional, Tuple

import numpy as np

__all__ = [
    "ComposeError", "Window", "windows_over", "compose_window",
    "overlap_gain",
]

LOG = logging.getLogger("spacr.ops_compose")


class ComposeError(ValueError):
    """Raised when a window cannot be composed, with the reason."""


@dataclass(frozen=True)
class Window:
    """A rectangle of the well frame, and where it sits in it.

    :param top: first row in well pixels.
    :param left: first column in well pixels.
    :param height: rows.
    :param width: columns.
    """

    top: int
    left: int
    height: int
    width: int

    @property
    def bottom(self) -> int:
        """One past the last row."""
        return self.top + self.height

    @property
    def right(self) -> int:
        """One past the last column."""
        return self.left + self.width

    def offset(self) -> Tuple[int, int]:
        """``(top, left)`` -- what a local coordinate adds to reach the well."""
        return (self.top, self.left)


def windows_over(canvas: Tuple[int, int], size: int = 4096,
                 overlap: int = 256) -> Iterator[Window]:
    """Tile a canvas into overlapping windows, in raster order.

    The windows overlap so objects can be sewn across them: an object
    straddling a window boundary is one object, and it can only be matched
    across the seam if both windows saw all of it. The overlap has to exceed
    the largest object, not merely touch.

    :param canvas: ``(height, width)`` of the well frame.
    :param size: window edge in pixels.
    :param overlap: how much neighbouring windows share.
    :raises ComposeError: when the overlap is not smaller than the window,
        which would make the stride zero and the iteration infinite.
    """
    height, width = int(canvas[0]), int(canvas[1])
    size, overlap = int(size), int(overlap)
    if size <= 0:
        raise ComposeError("window size must be positive")
    if overlap < 0 or overlap >= size:
        raise ComposeError(
            f"overlap {overlap} must be at least 0 and less than the window "
            f"{size}; an overlap of the whole window never advances")
    stride = size - overlap
    for top in range(0, max(height, 1), stride):
        for left in range(0, max(width, 1), stride):
            yield Window(top=top, left=left,
                         height=min(size, height - top),
                         width=min(size, width - left))
            if left + size >= width:
                break
        if top + size >= height:
            break


def compose_window(window: Window,
                   placements: Mapping[int, Tuple[float, float]],
                   read_tile: Callable[[int], np.ndarray], *,
                   tile_shape: Optional[Tuple[int, int]] = None,
                   dtype: type = np.float32) -> Tuple[np.ndarray, np.ndarray]:
    """Average every tile that overlaps ``window`` into one image.

    Averaged, not last-writer-wins. A tile pasted over its neighbour keeps a
    seam where two tiles disagree, and a segmenter can detect that seam as an
    edge. Averaging removes the seam and also gives the 1.15x noise gain.

    :param window: the rectangle of the well frame to build.
    :param placements: ``site -> (y, x)`` from the solve, in well pixels.
    :param read_tile: called with a site number, returns that tile. Called only
        for tiles that touch the window, so a well of 333 tiles costs a handful
        of reads per window rather than 333.
    :param tile_shape: ``(height, width)``; read from the first tile when
        omitted.
    :returns: ``(image, coverage)`` -- the averaged window and how many tiles
        contributed to each pixel. Coverage is returned because a pixel no tile
        reached is not a black pixel, and only the count tells them apart.
    """
    if window.height <= 0 or window.width <= 0:
        raise ComposeError("a window needs a positive height and width")

    total = np.zeros((window.height, window.width), dtype=dtype)
    count = np.zeros((window.height, window.width), dtype=np.uint16)
    shape = tile_shape

    for site, (top, left) in sorted(placements.items()):
        if shape is None:
            probe = np.asarray(read_tile(site))
            shape = (int(probe.shape[0]), int(probe.shape[1]))
        t_top, t_left = int(round(float(top))), int(round(float(left)))
        t_bottom, t_right = t_top + shape[0], t_left + shape[1]
        if t_bottom <= window.top or t_top >= window.bottom:
            continue
        if t_right <= window.left or t_left >= window.right:
            continue
        tile = np.asarray(read_tile(site))
        if tile.ndim != 2:
            raise ComposeError(
                f"site {site} is not a single plane; compose one channel at "
                f"a time (got shape {tile.shape})")
        wy0, wx0 = max(t_top, window.top), max(t_left, window.left)
        wy1, wx1 = min(t_bottom, window.bottom), min(t_right, window.right)
        ty0, tx0 = wy0 - t_top, wx0 - t_left
        ty1, tx1 = ty0 + (wy1 - wy0), tx0 + (wx1 - wx0)
        patch = tile[ty0:ty1, tx0:tx1]
        total[wy0 - window.top:wy1 - window.top,
              wx0 - window.left:wx1 - window.left] += patch.astype(dtype)
        count[wy0 - window.top:wy1 - window.top,
              wx0 - window.left:wx1 - window.left] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        image = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    return image.astype(dtype), count


def overlap_gain(count: np.ndarray) -> float:
    """The noise gain from averaging, over pixels a tile reached.

    sqrt of the mean coverage: averaging n independent samples of one signal
    divides the noise by sqrt(n). Reported over covered pixels only, because a
    well's corners are reached by one tile and including the uncovered
    background would report a gain for pixels nothing measured.

    A gain of 1.15x was measured for well A1 at a 14% raster overlap; a much
    larger value on the same geometry indicates that something is averaged
    twice.

    :param count: the coverage array from :func:`compose_window`.
    """
    covered = np.asarray(count)[np.asarray(count) > 0]
    if covered.size == 0:
        return 0.0
    return float(np.sqrt(covered.mean()))
