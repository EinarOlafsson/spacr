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

The overlaps are feathered by default: each tile is weighted by how far a
pixel lies from that tile's nearest edge, so a tile fades out where its
neighbour fades in and no seam is left where two tiles disagree in
brightness. Plain averaging (``blend='average'``) gives a noise gain of about
1.15x in the raster's 14% overlap but keeps a brightness step at each tile
edge; on Funk et al. 2022 the feathered seams scored like a single unstitched
tile cut at the same place, and the averaged ones did not
(:func:`compose_window`).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterator, Mapping, Optional, Tuple

import numpy as np

__all__ = [
    "ComposeError", "Window", "windows_over", "compose_window",
    "overlap_gain",
]

LOG = logging.getLogger("spacr.ops_compose")

#: The ways :func:`compose_window` may blend overlapping tiles, the first
#: being the default (``ops_compose_blend``).
_BLENDS = ("feather", "average")


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
    :raises ComposeError: when the window size is not positive, or the overlap is
        negative or not smaller than the window.
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
                   dtype: type = np.float32,
                   blend: str = "feather") -> Tuple[np.ndarray, np.ndarray]:
    """Blend every tile that overlaps ``window`` into one image.

    Blended, not last-writer-wins: a tile pasted over its neighbour keeps a
    seam where two tiles disagree, and a segmenter can detect that seam as an
    edge.

    FEATHERED BY DEFAULT. Each tile's pixel is weighted by its distance, in
    pixels, to the nearest edge of that tile, plus one, and the composite is
    ``sum(weight x tile) / sum(weight)``. A tile's edge pixels weigh 1 and
    its pixels deep in the overlap weigh more, so each tile fades out where
    its neighbour fades in. A plain average weighs both tiles equally across
    the whole overlap and so keeps a step where one tile ends: the edge of
    the overlap is where the composite switches from the mean of two tiles
    to one tile. On Funk et al. 2022 (DAPI, cycle 1; well A1 of plates A
    and B, then well B2 of held-out plates C and D) the share of feathered
    seams that step more than the tile centres matched that of a single
    unstitched tile cut at exactly the same place, within 1 point (95 %
    CI); the averaged seams' share was 11 to 12 points higher on C and D.
    No flat-field or gain correction is applied: feathering alone passed
    and was the fastest arm that did.
    Where one tile covers a pixel the weights cancel and the pixel is that
    tile's value. The feathered sums are kept in float64, where products of
    16-bit values and weights of several hundred are exact.

    ``blend='average'`` gives the earlier plain mean, unchanged. It has the
    larger noise gain in the overlaps (about 1.15x, :func:`overlap_gain`);
    feathering gives up some of it near the tile edges for the seam.

    :param window: the rectangle of the well frame to build.
    :param placements: ``site -> (y, x)`` from the solve, in well pixels.
    :param read_tile: called with a site number, returns that tile. Called only
        for tiles that touch the window, so a well of 333 tiles costs a handful
        of reads per window rather than 333.
    :param tile_shape: ``(height, width)``; read from the first tile when
        omitted.
    :param dtype: the dtype of the returned image.
    :param blend: ``'feather'`` (default) or ``'average'``.
    :returns: ``(image, coverage)`` -- the blended window and how many tiles
        contributed to each pixel. Coverage is returned because a pixel no tile
        reached is not a black pixel, and only the count tells them apart.
        It is the same count for either blend.
    :raises ComposeError: for an empty window, a tile that is not one plane,
        or an unknown ``blend``.
    """
    if window.height <= 0 or window.width <= 0:
        raise ComposeError("a window needs a positive height and width")
    blend = str(blend).strip().lower()
    if blend not in _BLENDS:
        raise ComposeError(
            f"blend must be one of {list(_BLENDS)}; got {blend!r}")
    feather = blend == "feather"
    if feather:
        from .stitch_core import feather_weights

    total = np.zeros((window.height, window.width),
                     dtype=np.float64 if feather else dtype)
    weights = (np.zeros((window.height, window.width), dtype=np.float64)
               if feather else None)
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
        rows = slice(wy0 - window.top, wy1 - window.top)
        cols = slice(wx0 - window.left, wx1 - window.left)
        if feather:
            weight = feather_weights(shape, (ty0, ty1), (tx0, tx1),
                                     mode="distance")
            total[rows, cols] += patch.astype(np.float64) * weight
            weights[rows, cols] += weight
        else:
            total[rows, cols] += patch.astype(dtype)
        count[rows, cols] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        if feather:
            image = np.where(weights > 0, total / np.maximum(weights, 1e-9),
                             0.0)
        else:
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
