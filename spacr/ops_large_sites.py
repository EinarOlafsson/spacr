"""Segmentation of large cropped OPS sites (item 921).

PERISCOPE publishes its painting images as 100 crops per well, each
5,500 x 5,500 px x 5 channels of uint16. One site is about 300 MB raw and
more than 1 GB once Cellpose holds float32 copies, flows and probabilities,
so the OPS phenotype path, sized for 2,960 px fields, cannot segment it in
one piece on a worker with a fixed RAM budget. This module adds:

* tiled segmentation (:func:`segment_tiled`): tiles of ``seg_tile_px`` with
  ``seg_tile_overlap_px`` of overlap, segmented in row-major order; copies of
  one object in two tiles are merged by IoU, keeping the copy whose centroid
  is farther from an inner tile edge (:func:`stitch_tile_labels`); a site no
  larger than one tile goes through the segmenter unchanged; each channel is
  normalised once on the whole site (:func:`normalize_site`) and the tiles
  are segmented with ``normalize=False``, so a tile sees the same values as
  the untiled site, and tiles are placed on Cellpose's own network-tile grid
  (:func:`tiling_layout`), so the network sees the same pixels tiled and
  untiled;
* chunked measurement (:func:`measure_chunked`): objects measured in row
  bands, each band read with a halo that holds every object whose centroid
  lies in it, so each object is measured once and whole;
* a RAM planner (:func:`plan_site_memory`) that picks the tile size and the
  worker count from ``seg_max_ram_gb`` and refuses a cap it cannot meet;
* the crop-edge policy (:func:`apply_edge_policy`): ``'exclude'``,
  ``'keep'`` or ``'own'`` (an object cut by a seam between two placed crops
  is kept by the crop holding its larger part);
* the ``cell_painting_phalloidin`` preset (:func:`segment_cell_painting_site`):
  nuclei on DNA, cells on ``[Phalloidin, DNA]``, one nucleus per cell, with
  the plate's nucleus diameter and the tiling-control site taken from
  populated sites spread over the well (:func:`survey_sites`);
* the agreement report against reference label masks (:func:`mask_agreement`).

:func:`run_large_site_segmentation` is the settings-file entry. The memory
model of the planner is an estimate; the peak memory actually used per site
is measured with :class:`spacr.resource_log.ResourceSampler` and written to
the site table, so the estimate can be checked against every run.
"""
from __future__ import annotations

import glob
import json
import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: The item 921 settings and their defaults.
LARGE_SITE_DEFAULTS: Dict[str, Any] = {
    "seg_tile_px": 2048,
    "seg_tile_overlap_px": 256,
    "seg_stitch_iou": 0.5,
    "seg_max_ram_gb": 24,
    "seg_edge_policy": "exclude",
    "seg_preset": None,
    "nucleus_diameter": None,
    "cell_model": "cpsam",
    "cell_channels": ["CorrPhalloidin", "CorrDNA"],
    "seg_report_reference": None,
    "seg_sites_src": None,
    "seg_site_channels": ["CorrDNA", "CorrPhalloidin", "CorrWGA", "CorrMito", "CorrER"],
    "seg_sites": None,
    "seg_dst": None,
    "seg_workers": 1,
    "seg_measure_band_px": 1024,
    "seg_gpu": False,
    "seg_tiling_control_px": 0,
    "seg_nucleus_channel": "CorrDNA",
    "seg_nucleus_model": "cpsam",
    "seg_nucleus_diameter_range": [10.0, 80.0],
    "seg_diameter_sites": 9,
    "seg_diameter_min_objects": 50,
}

#: What ``seg_preset='cell_painting_phalloidin'`` sets, after PERISCOPE's
#: ``9_Analysis.cppipe`` (nuclei on DNA, 10-80 px; cells on Phalloidin).
PHALLOIDIN_PRESET: Dict[str, Any] = {
    "seg_nucleus_channel": "CorrDNA",
    "seg_nucleus_model": "cpsam",
    "seg_nucleus_diameter_range": [10.0, 80.0],
    "cell_model": "cpsam",
    "cell_channels": ["CorrPhalloidin", "CorrDNA"],
    "seg_edge_policy": "exclude",
    "seg_max_ram_gb": 24,
}

#: The edge policies ``seg_edge_policy`` may name.
EDGE_POLICIES = ("exclude", "keep", "own")

#: The presets ``seg_preset`` may name.
SEG_PRESETS = ("cell_painting_phalloidin",)

#: Bytes per tile pixel while Cellpose runs: float32 input of two channels,
#: three flow planes, the probability map, the label image and the network's
#: activations, measured as roughly 4x the arrays it returns.
TILE_BYTES_PER_PIXEL = 4 * (2 * 4 + 3 * 4 + 4 + 4)

#: Resident bytes of the Cellpose-SAM network on the CPU.
MODEL_BYTES = int(1.5 * 2 ** 30)

#: The smallest tile the planner will fall back to.
MIN_TILE_PX = 512

#: Cellpose-SAM's network tile side, and the overlap fraction Cellpose puts
#: between network tiles when ``eval`` runs with its defaults.
NET_TILE_PX = 256
NET_TILE_OVERLAP = 0.1


class MemoryPlanError(ValueError):
    """The per-worker RAM cap cannot be met for this site."""


def large_site_settings(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Item 921's settings with defaults, the preset expanded and checked.

    Values the caller sets win over the preset; the preset wins over the
    defaults.

    :param settings: the caller's settings.
    :returns: a new dict.
    :raises ValueError: for an unknown preset or edge policy.
    """
    given = dict(settings or {})
    out = dict(LARGE_SITE_DEFAULTS)
    preset = given.get("seg_preset")
    if preset not in (None, "", "none"):
        if preset not in SEG_PRESETS:
            raise ValueError(f"seg_preset must be one of {list(SEG_PRESETS)}; got {preset!r}")
        out.update(PHALLOIDIN_PRESET)
    out.update({k: v for k, v in given.items() if v is not None or k not in out})
    if out["seg_edge_policy"] not in EDGE_POLICIES:
        raise ValueError(f"seg_edge_policy must be one of {list(EDGE_POLICIES)}; "
                         f"got {out['seg_edge_policy']!r}")
    return out


def tile_grid(shape: Tuple[int, int], tile_px: int, overlap_px: int
              ) -> List[Tuple[int, int, int, int]]:
    """The tiles covering an image, in row-major order.

    :param shape: ``(height, width)``.
    :param tile_px: the tile side; 0 or a side covering the image gives one
        tile.
    :param overlap_px: pixels shared by neighbouring tiles.
    :returns: ``(y0, y1, x0, x1)`` boxes; the last tile of a row or column
        is moved back so it ends at the image edge and stays full size.
    :raises ValueError: when the overlap is not smaller than the tile.
    """
    height, width = int(shape[0]), int(shape[1])
    if not tile_px or (height <= tile_px and width <= tile_px):
        return [(0, height, 0, width)]
    if overlap_px >= tile_px:
        raise ValueError(f"seg_tile_overlap_px ({overlap_px}) must be smaller than "
                         f"seg_tile_px ({tile_px})")

    def starts(size: int) -> List[int]:
        """Tile starts along one axis."""
        if size <= tile_px:
            return [0]
        step = tile_px - overlap_px
        out = list(range(0, size - tile_px, step))
        out.append(size - tile_px)
        return sorted(set(out))

    return [(y, min(y + tile_px, height), x, min(x + tile_px, width))
            for y in starts(height) for x in starts(width)]


def network_tile_starts(length: int, bsize: int = NET_TILE_PX,
                        overlap: float = NET_TILE_OVERLAP) -> np.ndarray:
    """Where Cellpose runs its network along one image axis, in image coordinates.

    Cellpose pads an image to a multiple of 16 plus 8 px on each side and
    spreads ``ceil((1 + 2 * overlap) * padded / bsize)`` network tiles evenly
    over the padded side (``cellpose.core.run_net`` without resizing). A
    pixel's flows are the tapered mean of the network tiles that cover it, so
    two runs give a pixel the same flows only when the same network tiles,
    with the same pixels, cover it.

    :param length: the image side in pixels.
    :param bsize: the network tile side.
    :param overlap: Cellpose's ``tile_overlap``.
    :returns: the start of each network tile; negative starts lie in the
        padding.
    """
    length = int(length)
    if length >= bsize:
        extra = int(16 * np.ceil(length / 16) - length)
    else:
        extra = max(0, bsize - length - 16)
    padded = length + 16 + extra
    overlap = min(0.5, max(0.05, float(overlap)))
    count = 1 if padded <= bsize else int(np.ceil((1.0 + 2 * overlap) * padded / bsize))
    return np.linspace(0, padded - min(bsize, padded), count).astype(int) - (8 + extra // 2)


def _lattice_stride(length: int) -> Optional[int]:
    """The network-tile stride of a side whose network tiles start at -8 and
    are evenly spaced by a whole number of pixels, else None."""
    starts = network_tile_starts(length)
    if starts.size < 2 or starts[0] != -8:
        return None
    steps = np.diff(starts)
    return int(steps[0]) if bool((steps == steps[0]).all()) else None


def aligned_tiling(tile_px: int, overlap_px: int) -> Optional[Tuple[int, int, int]]:
    """A tile side and step that put every tile's network tiles on one grid.

    Tiles whose sides have evenly spaced network tiles (stride ``s``) and
    whose starts are multiples of ``s`` all run Cellpose's network on the
    same network tiles as the untiled image would: a pixel at least
    ``NET_TILE_PX`` from a cut tile edge then gets bit-identical flows tiled
    and untiled. Holds for Cellpose's defaults (``bsize=256``,
    ``tile_overlap=0.1``) and no resizing, that is no ``diameter``.

    :param tile_px: the largest tile side wanted.
    :param overlap_px: the smallest overlap wanted.
    :returns: ``(tile, step, stride)``: the largest tile side no larger than
        ``tile_px``, a step between tile starts that is a multiple of the
        stride and leaves at least ``overlap_px`` of overlap, and the
        stride; None when no such tile exists.
    """
    for length in range(int(tile_px) // 16 * 16, NET_TILE_PX, -16):
        stride = _lattice_stride(length)
        if stride is None:
            continue
        step = stride * ((length - int(overlap_px)) // stride)
        if step >= stride:
            return length, step, stride
    return None


def tiling_layout(shape: Tuple[int, int], tile_px: int, overlap_px: int,
                  align: bool = False) -> Dict[str, Any]:
    """The tiles :func:`segment_tiled` uses for one image.

    Without ``align`` these are :func:`tile_grid`'s tiles. With ``align``,
    and when the image needs more than one tile, the tile side and step come
    from :func:`aligned_tiling`, and each tiled side is padded at its end so
    that the last tile also starts on the network grid; the stitched labels
    are then cut back to the image.

    :param shape: ``(height, width)``.
    :param tile_px: the tile side asked for.
    :param overlap_px: the overlap asked for.
    :param align: put the tiles on Cellpose's network grid.
    :returns: ``boxes`` (in padded coordinates), ``padded`` (the padded
        shape), ``tile_px`` and ``overlap_px`` (those used) and ``aligned``.
    """
    shape = (int(shape[0]), int(shape[1]))
    boxes = tile_grid(shape, tile_px, overlap_px)
    plan = aligned_tiling(tile_px, overlap_px) if align and len(boxes) > 1 else None
    if plan is None:
        return {"boxes": boxes, "padded": shape, "tile_px": int(tile_px),
                "overlap_px": int(overlap_px), "aligned": False}
    tile, step, stride = plan
    padded = tuple(size if size <= tile else tile + stride * int(np.ceil((size - tile) / stride))
                   for size in shape)
    return {"boxes": tile_grid(padded, tile, tile - step), "padded": padded, "tile_px": tile,
            "overlap_px": tile - step, "aligned": True}


def control_layout(side: int, overlap_px: int) -> Tuple[int, int, bool]:
    """The region and tile sides of the tiling control.

    The region is the largest side no larger than ``side`` whose network
    tiles fall on one grid, and the tile the smallest side on the same grid
    that covers the region in two tiles per axis with at least
    ``overlap_px`` of overlap, so tiled and untiled runs share every network
    tile away from the seams (:func:`aligned_tiling`). Without such a pair
    the region is ``side`` and the tile ``(side + overlap_px) // 2 + 1``,
    unaligned.

    :param side: the region side asked for (``seg_tiling_control_px``).
    :param overlap_px: the overlap asked for.
    :returns: ``(region_px, tile_px, aligned)``.
    """
    for region in range(int(side) // 16 * 16, NET_TILE_PX, -16):
        stride = _lattice_stride(region)
        if stride is None:
            continue
        smallest = (region + int(overlap_px) + 1) // 2
        for tile in range(-(-smallest // 16) * 16, region, 16):
            plan = aligned_tiling(tile, overlap_px)
            if plan is not None and plan[0] == tile and plan[2] == stride:
                return region, tile, True
    return int(side), (int(side) + int(overlap_px)) // 2 + 1, False


def plan_site_memory(shape: Tuple[int, int], n_channels: int, settings: Mapping[str, Any], *,
                     available_bytes: Optional[int] = None) -> Dict[str, Any]:
    """Tile size and worker count that keep each worker under the cap.

    One worker holds the raw site (uint16, every channel), two label images
    (int32), the network and one tile in flight
    (:data:`TILE_BYTES_PER_PIXEL`). The tile is halved from ``seg_tile_px``
    down to :data:`MIN_TILE_PX` until it fits; workers are capped by
    ``seg_workers`` and by the machine's available memory divided by the cap.

    :param shape: the site's ``(height, width)``.
    :param n_channels: channels read per site.
    :param settings: read for ``seg_max_ram_gb``, ``seg_tile_px``,
        ``seg_tile_overlap_px`` and ``seg_workers``.
    :param available_bytes: memory the machine can give; ``None`` reads it
        with psutil.
    :returns: ``tile_px``, ``workers``, ``estimate_gb`` and ``cap_gb``.
    :raises MemoryPlanError: when even the smallest tile exceeds the cap, or
        the machine cannot hold one worker.
    """
    cap = float(settings.get("seg_max_ram_gb") or LARGE_SITE_DEFAULTS["seg_max_ram_gb"]) * 2 ** 30
    pixels = int(shape[0]) * int(shape[1])
    fixed = pixels * (2 * int(n_channels) + 2 * 4) + MODEL_BYTES
    requested = int(settings.get("seg_tile_px") or 0)
    overlap = int(settings.get("seg_tile_overlap_px") or 0)
    tile = requested if requested else max(int(shape[0]), int(shape[1]))
    while True:
        side = min(tile, max(int(shape[0]), int(shape[1])))
        estimate = fixed + side * side * TILE_BYTES_PER_PIXEL
        if estimate <= cap:
            break
        if tile // 2 < max(MIN_TILE_PX, 2 * overlap + 1):
            raise MemoryPlanError(
                f"seg_max_ram_gb={cap / 2 ** 30:g} cannot hold a {shape[0]}x{shape[1]} site with "
                f"{n_channels} channels: the smallest tile ({tile} px) needs "
                f"{estimate / 2 ** 30:.1f} GB")
        tile //= 2
    if available_bytes is None:
        import psutil

        available_bytes = int(psutil.virtual_memory().available)
    workers = min(max(1, int(settings.get("seg_workers") or 1)), int(available_bytes // cap))
    if workers < 1:
        raise MemoryPlanError(
            f"the machine has {available_bytes / 2 ** 30:.1f} GB available, less than one "
            f"worker's seg_max_ram_gb={cap / 2 ** 30:g}")
    tiled = bool(requested) or tile < max(int(shape[0]), int(shape[1]))
    return {"tile_px": int(tile) if tiled else 0, "workers": int(workers),
            "estimate_gb": round(estimate / 2 ** 30, 3), "cap_gb": cap / 2 ** 30}


def _edge_distance(box: Tuple[int, int, int, int], shape: Tuple[int, int],
                   cy: float, cx: float) -> float:
    """Distance from a centroid to the nearest inner edge of its tile.

    :param box: the tile ``(y0, y1, x0, x1)``.
    :param shape: the image shape; tile edges on the image border do not
        count, since no other tile sees past them.
    :param cy: centroid row in image coordinates.
    :param cx: centroid column in image coordinates.
    :returns: the distance in pixels, ``inf`` for a tile with no inner edge.
    """
    y0, y1, x0, x1 = box
    gaps = [cy - y0 if y0 > 0 else np.inf, (y1 - 1) - cy if y1 < shape[0] else np.inf,
            cx - x0 if x0 > 0 else np.inf, (x1 - 1) - cx if x1 < shape[1] else np.inf]
    return float(min(gaps))


def stitch_tile_labels(tiles: Sequence[Tuple[Tuple[int, int, int, int], np.ndarray]],
                       shape: Tuple[int, int], iou: float = 0.5) -> np.ndarray:
    """One label image from per-tile label images.

    Every object of every tile is ranked by the distance of its centroid to
    the nearest inner tile edge, farthest first (ties by tile order, then
    label). Objects are painted in that order onto unclaimed pixels; an
    object is dropped when it overlaps an object already painted with IoU of
    at least ``iou``, or when half or more of its pixels are already
    claimed (a cut copy of an object seen whole in another tile). Labels are
    then renumbered 1..N in raster order of each object's first pixel.

    :param tiles: ``((y0, y1, x0, x1), labels)`` per tile.
    :param shape: the full image ``(height, width)``.
    :param iou: the overlap IoU at which two copies are one object.
    :returns: an int32 label image.
    """
    from scipy import ndimage

    candidates = []
    for order, (box, labels) in enumerate(tiles):
        labels = np.asarray(labels)
        slices = ndimage.find_objects(labels)
        for value, where in enumerate(slices, start=1):
            if where is None:
                continue
            ys, xs = np.nonzero(labels[where] == value)
            cy = float(ys.mean() + where[0].start + box[0])
            cx = float(xs.mean() + where[1].start + box[2])
            candidates.append((-_edge_distance(box, shape, cy, cx), order, value, where))
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    out = np.zeros(shape, np.int32)
    areas: Dict[int, int] = {}
    next_id = 1
    for _, order, value, where in candidates:
        box, labels = tiles[order]
        mask = np.asarray(labels)[where] == value
        target = out[where[0].start + box[0]:where[0].stop + box[0],
                     where[1].start + box[2]:where[1].stop + box[2]]
        claimed = target[mask]
        area = int(mask.sum())
        hits = claimed[claimed > 0]
        if hits.size:
            ids, counts = np.unique(hits, return_counts=True)
            overlap = counts / (area + np.array([areas[int(i)] for i in ids]) - counts)
            if overlap.max() >= iou or hits.size * 2 >= area:
                continue
        free = mask & (target == 0)
        target[free] = next_id
        areas[next_id] = int(free.sum())
        next_id += 1
    return relabel_raster(out)


def relabel_raster(labels: np.ndarray) -> np.ndarray:
    """Renumber labels 1..N in raster order of each object's first pixel.

    :param labels: a label image.
    :returns: an int32 label image with the same objects.
    """
    flat = np.asarray(labels).ravel()
    values, first = np.unique(flat, return_index=True)
    keep = values > 0
    order = values[keep][np.argsort(first[keep], kind="stable")]
    lookup = np.zeros(int(values.max()) + 1 if values.size else 1, np.int32)
    lookup[order] = np.arange(1, order.size + 1, dtype=np.int32)
    return lookup[flat].reshape(np.shape(labels))


def segment_tiled(image: np.ndarray, segment: Callable[[np.ndarray], np.ndarray], *,
                  tile_px: int = 2048, overlap_px: int = 256, iou: float = 0.5,
                  align: bool = False) -> np.ndarray:
    """Segment an image tile by tile and stitch the labels.

    :param image: ``(H, W)`` or ``(H, W, C)``.
    :param segment: maps an image of the same layout to a label image.
    :param tile_px: tile side; 0, or an image no larger than one tile, calls
        ``segment`` once on the whole image and returns its output unchanged.
    :param overlap_px: overlap between tiles, at least twice the largest
        expected object diameter.
    :param iou: passed to :func:`stitch_tile_labels`.
    :param align: place the tiles on Cellpose's network grid
        (:func:`tiling_layout`), so that objects away from the seams are the
        untiled run's objects pixel for pixel; only for a segmenter that
        does not resize the image.
    :returns: an int32 label image of the image's height and width.
    """
    shape = tuple(int(v) for v in np.shape(image)[:2])
    layout = tiling_layout(shape, tile_px, overlap_px, align)
    if len(layout["boxes"]) == 1:
        return np.asarray(segment(image))
    padded = layout["padded"]
    if padded != shape:
        image = np.pad(image, [(0, padded[0] - shape[0]), (0, padded[1] - shape[1])]
                       + [(0, 0)] * (np.ndim(image) - 2))
    tiles = [((y0, y1, x0, x1), np.asarray(segment(image[y0:y1, x0:x1])))
             for y0, y1, x0, x1 in layout["boxes"]]
    labels = stitch_tile_labels(tiles, padded, iou)
    return labels if padded == shape else relabel_raster(labels[:shape[0], :shape[1]])


def tiling_control(image: np.ndarray, segment: Callable[[np.ndarray], np.ndarray], *,
                   tile_px: int, overlap_px: int, iou: float = 0.5,
                   align: bool = False) -> Dict[str, Any]:
    """Tiled against untiled segmentation of one region that fits untiled.

    Objects of the untiled run that stay clear of every seam zone (the
    strips that two or more tiles share) must come back with identical
    pixels; objects in a seam zone are scored by their best IoU with a tiled
    object. ``segment`` must not rescale intensities per call (Cellpose's
    default), or every tile is scaled differently from the region; pass a
    region normalised once (:func:`normalize_site`) and a segmenter run with
    ``normalize=False``.

    :param image: the region, ``(H, W)`` or ``(H, W, C)``.
    :param segment: maps an image to a label image.
    :param tile_px: the tile side of the tiled run; smaller than the region.
    :param overlap_px: the overlap of the tiled run.
    :param iou: passed to :func:`stitch_tile_labels`.
    :param align: passed to :func:`segment_tiled`.
    :returns: ``objects``, ``away_from_seams``, ``away_identical``,
        ``at_seams``, ``at_seams_iou_ge_0_9``, ``at_seams_share`` (``nan``
        without seam objects), and ``tile_px``, ``overlap_px`` and
        ``aligned`` of the tiles used.
    """
    shape = tuple(int(v) for v in np.shape(image)[:2])
    untiled = relabel_raster(np.asarray(segment(image)))
    tiled = segment_tiled(image, segment, tile_px=tile_px, overlap_px=overlap_px, iou=iou,
                          align=align)
    layout = tiling_layout(shape, tile_px, overlap_px, align)
    cover = np.zeros(layout["padded"], np.int32)
    for y0, y1, x0, x1 in layout["boxes"]:
        cover[y0:y1, x0:x1] += 1
    seam = cover[:shape[0], :shape[1]] > 1
    a, b = untiled.ravel().astype(np.int64), tiled.ravel().astype(np.int64)
    area_a, area_b = np.bincount(a), np.bincount(b)
    on_seam = np.zeros(area_a.size, bool)
    on_seam[np.unique(a[seam.ravel()])] = True
    both = (a > 0) & (b > 0)
    pairs, counts = np.unique(np.stack([a[both], b[both]]), axis=1, return_counts=True)
    best = np.zeros(area_a.size)
    identical = np.zeros(area_a.size, bool)
    for (i, j), n in zip(pairs.T, counts):
        best[i] = max(best[i], n / (area_a[i] + area_b[j] - n))
        identical[i] |= bool(n == area_a[i] == area_b[j])
    ids = np.unique(a[a > 0])
    away, near = ids[~on_seam[ids]], ids[on_seam[ids]]
    good = int((best[near] >= 0.9).sum())
    return {"objects": int(ids.size), "away_from_seams": int(away.size),
            "away_identical": int(identical[away].sum()), "at_seams": int(near.size),
            "at_seams_iou_ge_0_9": good,
            "at_seams_share": good / near.size if near.size else float("nan"),
            "tile_px": layout["tile_px"], "overlap_px": layout["overlap_px"],
            "aligned": layout["aligned"]}


def _border_labels(labels: np.ndarray) -> Dict[str, np.ndarray]:
    """The labels touching each side of a label image.

    :param labels: a label image.
    :returns: ``{'top', 'bottom', 'left', 'right'}`` to sorted label arrays.
    """
    sides = {"top": labels[0], "bottom": labels[-1], "left": labels[:, 0], "right": labels[:, -1]}
    return {side: np.unique(line[line > 0]) for side, line in sides.items()}


#: The side of a neighbour that meets each side of a crop.
_OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


def _seam_line(labels: np.ndarray, side: str) -> np.ndarray:
    """The pixel line of one side of a label image."""
    return {"top": labels[0], "bottom": labels[-1], "left": labels[:, 0],
            "right": labels[:, -1]}[side]


def apply_edge_policy(labels: np.ndarray, policy: str = "exclude", *,
                      neighbours: Optional[Mapping[str, np.ndarray]] = None
                      ) -> Tuple[np.ndarray, Dict[str, int]]:
    """Apply the crop-edge cell policy to one site's labels.

    ``'keep'`` keeps every object. ``'exclude'`` drops every object touching
    the crop border. ``'own'`` resolves objects cut by a seam with a placed
    neighbour crop (``neighbours`` maps a side to the neighbour's labels, the
    neighbour abutting without overlap): the piece in this crop and the
    pieces in the neighbour that meet it across the seam are one cell, and
    the crop holding the larger part keeps it; on a tie the top or left crop
    keeps it, so two crops never both keep or both drop a cell. Objects on a
    side with no known neighbour are dropped, as with ``'exclude'``.

    :param labels: the site's label image.
    :param policy: one of :data:`EDGE_POLICIES`.
    :param neighbours: for ``'own'``, side to neighbour label image.
    :returns: ``(labels, counts)``; ``counts`` has ``edge_objects`` (objects
        touching the border), ``excluded`` and ``owned`` (border objects
        kept because this crop owns them).
    :raises ValueError: for an unknown policy.
    """
    if policy not in EDGE_POLICIES:
        raise ValueError(f"seg_edge_policy must be one of {list(EDGE_POLICIES)}; got {policy!r}")
    labels = np.asarray(labels)
    border = _border_labels(labels)
    edge = np.unique(np.concatenate(list(border.values()))) if labels.size else np.array([], int)
    if policy == "keep":
        return labels.copy(), {"edge_objects": int(edge.size), "excluded": 0, "owned": 0}
    drop = set(int(v) for v in edge)
    owned = 0
    if policy == "own":
        areas = np.bincount(labels.ravel())
        keep = set()
        lost = set()
        for side, other in (neighbours or {}).items():
            other = np.asarray(other)
            other_areas = np.bincount(other.ravel())
            mine_line, their_line = _seam_line(labels, side), _seam_line(other, _OPPOSITE[side])
            if mine_line.shape != their_line.shape:
                raise ValueError(f"the {side} neighbour's seam has {their_line.size} px, "
                                 f"this crop's has {mine_line.size}")
            for value in border[side]:
                touching = np.unique(their_line[(mine_line == value) & (their_line > 0)])
                theirs = int(other_areas[touching].sum()) if touching.size else 0
                mine = int(areas[value])
                if mine > theirs or (mine == theirs and side in ("bottom", "right")):
                    keep.add(int(value))
                else:
                    lost.add(int(value))
        unresolved = {int(v) for side, values in border.items()
                      if side not in (neighbours or {}) for v in values}
        drop = lost | unresolved
        owned = len(keep - drop)
    out = labels.copy()
    if drop:
        out[np.isin(out, sorted(drop))] = 0
    return out, {"edge_objects": int(edge.size), "excluded": len(drop), "owned": owned}


def basic_features(labels: np.ndarray, image: np.ndarray,
                   channel_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Shape and intensity features of every object in a label image.

    :param labels: a label image.
    :param image: ``(H, W)`` or ``(H, W, C)`` of the same height and width.
    :param channel_names: one name per channel; ``ch0``... by default.
    :returns: one row per object: ``label``, ``area``, ``centroid_y``,
        ``centroid_x``, ``bbox_*``, ``eccentricity``, ``solidity``,
        ``perimeter``, and per channel ``<name>_mean``, ``<name>_sum``,
        ``<name>_std``; sorted by label.
    """
    from skimage.measure import regionprops_table

    image = np.asarray(image)
    if image.ndim == 2:
        image = image[..., None]
    names = list(channel_names or [f"ch{i}" for i in range(image.shape[-1])])
    table = regionprops_table(np.asarray(labels), properties=(
        "label", "area", "centroid", "bbox", "eccentricity", "solidity", "perimeter"))
    frame = pd.DataFrame(table).rename(columns={
        "centroid-0": "centroid_y", "centroid-1": "centroid_x", "bbox-0": "bbox_y0",
        "bbox-1": "bbox_x0", "bbox-2": "bbox_y1", "bbox-3": "bbox_x1"})
    from scipy import ndimage

    index = frame["label"].to_numpy()
    for c, name in enumerate(names):
        plane = image[..., c].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            frame[f"{name}_mean"] = ndimage.mean(plane, labels, index) if index.size else []
            frame[f"{name}_sum"] = ndimage.sum(plane, labels, index) if index.size else []
            frame[f"{name}_std"] = (ndimage.standard_deviation(plane, labels, index)
                                    if index.size else [])
    return frame.sort_values("label").reset_index(drop=True)


def measure_chunked(labels: np.ndarray, image: np.ndarray, *, band_px: int = 1024,
                    channel_names: Optional[Sequence[str]] = None,
                    measure: Optional[Callable[..., pd.DataFrame]] = None) -> pd.DataFrame:
    """Measure every object once, whole, one row band at a time.

    Each band holds the objects whose centroid row falls in it; it is read
    with a halo reaching the top of the highest and the bottom of the lowest
    of those objects, and every other object in the halo is masked out, so
    an object is measured exactly once and never cut. Positions are given in
    full-image coordinates.

    :param labels: the site's label image.
    :param image: the site's ``(H, W)`` or ``(H, W, C)`` pixels.
    :param band_px: band height; 0 measures the whole image at once.
    :param channel_names: passed to ``measure``.
    :param measure: ``measure(labels, image, channel_names)`` returning one
        row per object with ``label``; :func:`basic_features` by default.
    :returns: one row per object, sorted by label.
    """
    from scipy import ndimage

    measure = measure or basic_features
    labels = np.asarray(labels)
    if not band_px or band_px >= labels.shape[0]:
        return measure(labels, image, channel_names).sort_values("label").reset_index(drop=True)
    slices = ndimage.find_objects(labels)
    rows = []
    found = [(value, where) for value, where in enumerate(slices, start=1) if where is not None]
    centre = {value: float(np.nonzero(labels[where] == value)[0].mean() + where[0].start)
              for value, where in found}
    for top in range(0, labels.shape[0], band_px):
        members = [(v, w) for v, w in found if top <= centre[v] < top + band_px]
        if not members:
            continue
        y0 = min(w[0].start for _, w in members)
        y1 = max(w[0].stop for _, w in members)
        part = labels[y0:y1]
        part = np.where(np.isin(part, [v for v, _ in members]), part, 0)
        table = measure(part, np.asarray(image)[y0:y1], channel_names)
        for column in ("centroid_y", "bbox_y0", "bbox_y1"):
            if column in table:
                table[column] = table[column] + y0
        rows.append(table)
    if not rows:
        return measure(np.zeros_like(labels), image, channel_names)
    return pd.concat(rows, ignore_index=True).sort_values("label").reset_index(drop=True)


def estimate_nucleus_diameter(nuclei: Sequence[np.ndarray],
                              bounds: Tuple[float, float] = (10.0, 80.0)) -> float:
    """The plate's nucleus diameter from a first segmentation pass.

    Cellpose-SAM has no size model, so the diameter is read from nuclei
    segmented without one: the median equivalent diameter over every object
    of the sample, clipped to ``bounds`` (CellProfiler's
    ``IdentifyPrimaryObjects`` range in PERISCOPE's pipeline).

    :param nuclei: label images of the sample sites.
    :param bounds: ``(low, high)`` in pixels.
    :returns: the diameter in pixels; ``low`` when the sample has no nuclei.
    """
    areas = [np.bincount(np.asarray(n).ravel())[1:] for n in nuclei]
    areas = np.concatenate(areas) if areas else np.array([])
    areas = areas[areas > 0]
    if not areas.size:
        return float(bounds[0])
    return float(np.clip(np.median(2 * np.sqrt(areas / np.pi)), bounds[0], bounds[1]))


def survey_sites(sites: Sequence[str], nuclei: Mapping[str, np.ndarray],
                 bounds: Tuple[float, float] = (10.0, 80.0), *, min_objects: int = 50
                 ) -> Dict[str, Any]:
    """The plate's nucleus diameter and the tiling-control site from a sample of sites.

    A well's first site is often an empty corner of the well, so neither the
    diameter nor the tiling control may come from it. A sampled site counts
    only when it holds at least ``min_objects`` nuclei; the diameter is the
    median equivalent diameter over every nucleus of those sites, clipped to
    ``bounds`` (:func:`estimate_nucleus_diameter`), and the control site is
    the one with the median nucleus count among them (the lower of the two
    middle sites when their number is even).

    :param sites: the sampled sites, in run order.
    :param nuclei: per site, its nuclei as a label image.
    :param bounds: ``(low, high)`` in pixels.
    :param min_objects: the fewest nuclei a site needs to be used.
    :returns: ``diameter_px``, ``sites_sampled``, ``objects_per_site``,
        ``diameter_per_site`` (``None`` for a site without nuclei),
        ``sites_used``, ``min_objects`` and ``control_site``.
    :raises ValueError: when no sampled site holds ``min_objects`` nuclei.
    """
    objects, per_site = {}, {}
    for site in sites:
        areas = np.bincount(np.asarray(nuclei[site]).ravel())[1:]
        areas = areas[areas > 0]
        objects[site] = int(areas.size)
        per_site[site] = (float(np.median(2 * np.sqrt(areas / np.pi))) if areas.size else None)
    used = [site for site in sites if objects[site] >= int(min_objects)]
    if not used:
        raise ValueError(
            f"no sampled site has at least {int(min_objects)} nuclei "
            f"({', '.join(f'site {s}: {objects[s]}' for s in sites)}), so the nucleus diameter "
            "and the tiling-control site cannot be chosen; set nucleus_diameter, sample more "
            "sites with seg_diameter_sites or lower seg_diameter_min_objects")
    ranked = sorted(used, key=lambda site: (objects[site], sites.index(site)))
    return {"diameter_px": estimate_nucleus_diameter([nuclei[s] for s in used], bounds),
            "sites_sampled": list(sites), "objects_per_site": objects,
            "diameter_per_site": per_site, "sites_used": used,
            "min_objects": int(min_objects), "control_site": ranked[(len(ranked) - 1) // 2]}


def sample_sites(sites: Sequence[str], n: int) -> List[str]:
    """``n`` sites spread evenly over a well's site list, first and last included.

    :param sites: the well's sites in run order.
    :param n: how many to take; at least one.
    :returns: the sampled sites, in run order.
    """
    sites = list(sites)
    if not sites:
        return []
    picks = np.unique(np.linspace(0, len(sites) - 1, max(1, min(int(n), len(sites))))
                      .round().astype(int))
    return [sites[i] for i in picks]


def preset_models(settings: Mapping[str, Any], gpu: bool = False):
    """The nucleus and cell Cellpose models the preset segments with.

    Names go through spaCR's resolver, so the legacy names (``'nuclei'``,
    ``'cyto3'``) load what Cellpose 4 really has, and say so.

    :param settings: read for ``seg_nucleus_model`` and ``cell_model``.
    :param gpu: put the models on this machine's card.
    :returns: ``(nucleus_model, cell_model)``.
    """
    from .ops_engine import _cellpose_model
    from .utils import _resolve_cellpose_pretrained

    out = []
    for key, kind in (("seg_nucleus_model", "nucleus"), ("cell_model", "cell")):
        name = _resolve_cellpose_pretrained(settings.get(key) or "cpsam", kind, setting=key)
        out.append(_cellpose_model({key: name}, gpu, key=key))
    return tuple(out)


def normalize_site(image: np.ndarray, percentiles: Tuple[float, float] = (1.0, 99.0)
                   ) -> np.ndarray:
    """Cellpose's percentile normalisation, computed once on a whole site.

    Cellpose normalises every image it is given, so a tile is scaled by its
    own 1st and 99th percentiles and the same pixel gets different values in
    two tiles and in the untiled site. Here the percentiles of each channel
    come from the whole site, with the arithmetic of Cellpose's
    ``normalize_img`` (``normalize99`` with its subsampling; a flat channel
    becomes zeros), and every tile is then segmented with
    ``normalize=False``. A site that fits in one tile gets exactly the values
    Cellpose would have computed itself.

    :param image: ``(H, W)`` or ``(H, W, C)``.
    :param percentiles: ``(lower, upper)``.
    :returns: a float32 copy of ``image``, normalised per channel.
    """
    from cellpose.transforms import normalize99

    out = np.array(image, dtype=np.float32, copy=True)
    planes = out[..., np.newaxis] if out.ndim == 2 else out
    for c in range(planes.shape[-1]):
        if np.ptp(planes[..., c]) > 0:
            planes[..., c] = normalize99(planes[..., c], lower=percentiles[0],
                                         upper=percentiles[1], copy=False, downsample=True)
        else:
            planes[..., c] = 0
    return out


def _eval_labels(model: Any, image: np.ndarray, settings: Mapping[str, Any],
                 diameter: Optional[float], **kwargs: Any) -> np.ndarray:
    """One Cellpose call on a site-normalised image, returning the labels."""
    from ._gpu_governor import _governed_eval

    extra = {"diameter": float(diameter)} if diameter else {}
    return np.asarray(_governed_eval(model, image, site="ops_large_sites", settings=settings,
                                     batch_size=8, normalize=False, **extra, **kwargs)[0],
                      np.int32)


def segment_cell_painting_site(stack: np.ndarray, channel_names: Sequence[str],
                               settings: Mapping[str, Any], models: Tuple[Any, Any], *,
                               nucleus_diameter: Optional[float] = None
                               ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Nuclei and cells of one site with the ``cell_painting_phalloidin`` preset.

    Nuclei are segmented on the DNA channel, cells on ``cell_channels``
    (Phalloidin, then DNA), each normalised once on the whole site
    (:func:`normalize_site`) and tiled by :func:`segment_tiled` on Cellpose's
    network grid (nuclei only without a diameter, since a diameter resizes
    each tile); then each
    nucleus is paired with one cell by
    :func:`spacr.ops_sample.pair_nuclei_with_cells` (cells without a nucleus
    are dropped, as in the OPS phenotype path).

    :param stack: the ``(H, W, C)`` site in the reader's channel order.
    :param channel_names: one name per channel of ``stack``.
    :param settings: item 921 settings (see :func:`large_site_settings`).
    :param models: ``(nucleus_model, cell_model)``, anything with Cellpose's
        ``eval``.
    :param nucleus_diameter: the plate's nucleus diameter; ``None`` uses
        ``nucleus_diameter`` from the settings, or no diameter.
    :returns: ``(nuclei, cells, counts)`` with nucleus ``k`` inside cell ``k``.
    :raises ValueError: when a preset channel is not in ``channel_names``.
    """
    from .ops_sample import pair_nuclei_with_cells

    names = list(channel_names)
    wanted = [settings.get("seg_nucleus_channel") or "CorrDNA"] + list(settings["cell_channels"])
    missing = [c for c in wanted if c not in names]
    if missing:
        raise ValueError(f"the site has channels {names}; the preset needs {missing}")
    tile = int(settings.get("seg_tile_px") or 0)
    overlap = int(settings.get("seg_tile_overlap_px") or 0)
    iou = float(settings.get("seg_stitch_iou") or 0.5)
    diameter = nucleus_diameter or settings.get("nucleus_diameter")
    nucleus_model, cell_model = models
    dna = normalize_site(stack[..., names.index(wanted[0])])
    nuclei = segment_tiled(dna, lambda x: _eval_labels(nucleus_model, x, settings, diameter),
                           tile_px=tile, overlap_px=overlap, iou=iou, align=not diameter)
    body = normalize_site(np.stack([stack[..., names.index(c)] for c in settings["cell_channels"]],
                                   axis=-1))
    cells = segment_tiled(body, lambda x: _eval_labels(cell_model, x, settings, None,
                                                       channel_axis=-1),
                          tile_px=tile, overlap_px=overlap, iou=iou, align=True)
    return pair_nuclei_with_cells(nuclei, cells)


def mask_agreement(labels: np.ndarray, reference: np.ndarray, *, iou: float = 0.5,
                   drop_edge: bool = True) -> Dict[str, Any]:
    """Object agreement of a label image with a reference label image.

    Objects are paired when their IoU is at least ``iou`` (above 0.5 a pair
    is unique). Agreement, not ground truth: the reference is another
    pipeline's segmentation of the same pixels.

    :param labels: spaCR's labels.
    :param reference: the reference labels, same shape.
    :param iou: the pairing threshold.
    :param drop_edge: count only objects not touching the image border, in
        both images.
    :returns: ``n_spacr``, ``n_reference``, ``matched``, ``precision``,
        ``recall``, ``f1``, ``median_iou`` (of matched pairs) and
        ``count_ratio`` (spaCR over reference).
    :raises ValueError: when the shapes differ.
    """
    labels, reference = np.asarray(labels), np.asarray(reference)
    if labels.shape != reference.shape:
        raise ValueError(f"label shapes differ: {labels.shape} and {reference.shape}")
    if drop_edge:
        labels, _ = apply_edge_policy(labels, "exclude")
        reference, _ = apply_edge_policy(reference, "exclude")
    a, b = labels.ravel().astype(np.int64), reference.ravel().astype(np.int64)
    ids_a, ids_b = np.unique(a[a > 0]), np.unique(b[b > 0])
    area_a, area_b = np.bincount(a), np.bincount(b)
    both = (a > 0) & (b > 0)
    pairs, counts = np.unique(np.stack([a[both], b[both]]), axis=1, return_counts=True)
    scores = (counts / (area_a[pairs[0]] + area_b[pairs[1]] - counts)) if counts.size else np.array([])
    hit = scores >= iou
    matched = int(hit.sum())
    precision = matched / ids_a.size if ids_a.size else float("nan")
    recall = matched / ids_b.size if ids_b.size else float("nan")
    f1 = (2 * matched / (ids_a.size + ids_b.size)) if (ids_a.size + ids_b.size) else float("nan")
    return {"n_spacr": int(ids_a.size), "n_reference": int(ids_b.size), "matched": matched,
            "precision": precision, "recall": recall, "f1": f1,
            "median_iou": float(np.median(scores[hit])) if matched else float("nan"),
            "count_ratio": ids_a.size / ids_b.size if ids_b.size else float("nan")}


def read_cropped_site(well_dir: str, site: Any, channels: Sequence[str]) -> np.ndarray:
    """One cropped site of a PERISCOPE-style well folder as ``(H, W, C)``.

    The layout is ``<well_dir>/<channel>/<channel>_Site_<site>.tif(f)`` (the
    Cell Painting Gallery's ``images_corrected_cropped``), or one
    ``<well_dir>/<site>.npy`` holding ``(H, W, C)`` in ``channels`` order.

    :param well_dir: the well folder.
    :param site: the site number or name.
    :param channels: the channels to read, in order.
    :returns: the stacked site, the files' dtype kept.
    :raises FileNotFoundError: when a channel file is missing.
    """
    import tifffile

    stacked = os.path.join(well_dir, f"{site}.npy")
    if os.path.isfile(stacked):
        return np.load(stacked)
    planes = []
    for channel in channels:
        found = sorted(glob.glob(os.path.join(well_dir, channel, f"{channel}_Site_{site}.tif*")))
        if not found:
            raise FileNotFoundError(f"no {channel} image for site {site} in {well_dir}")
        planes.append(tifffile.imread(found[0]))
    return np.stack(planes, axis=-1)


def list_sites(well_dir: str, channels: Sequence[str]) -> List[str]:
    """The sites of a well folder, in numeric order where they are numbers.

    :param well_dir: the well folder (see :func:`read_cropped_site`).
    :param channels: the reader's channels; the first one's files are listed.
    :returns: site names.
    """
    names = [os.path.basename(p)[:-4] for p in glob.glob(os.path.join(well_dir, "*.npy"))]
    if not names and channels:
        prefix = f"{channels[0]}_Site_"
        names = [os.path.basename(p).split(prefix, 1)[1].split(".")[0]
                 for p in glob.glob(os.path.join(well_dir, channels[0], prefix + "*.tif*"))]
    return sorted(set(names), key=lambda s: (0, int(s), "") if s.isdigit() else (1, 0, s))


def _reference_mask(folder: Optional[str], site: str, kind: str) -> Optional[np.ndarray]:
    """A reference ``*_Site_<site>_<kind>_Objects.tif(f)`` under ``folder``, or None."""
    import tifffile

    if not folder:
        return None
    found = sorted(glob.glob(os.path.join(folder, "**", f"*_Site_{site}_{kind}_Objects.tif*"),
                             recursive=True))
    return tifffile.imread(found[0]) if found else None


def run_large_site_segmentation(settings: Mapping[str, Any], *, models: Optional[Tuple[Any, Any]] = None,
                                available_bytes: Optional[int] = None) -> Dict[str, Any]:
    """Segment, measure and report every site of one cropped well.

    Writes, under ``seg_dst``: ``labels/<site>_nuclei.npy`` and
    ``labels/<site>_cells.npy``, ``tables/<site>_cells.parquet`` (one row per
    cell, measured by :func:`measure_chunked`), ``sites.csv`` (per site: objects,
    edge counts, the agreement with ``seg_report_reference`` when given, peak
    memory and seconds), ``agreement_well.csv``, ``settings.json``,
    ``nucleus_diameter.json`` (the diameter, whether it was fixed or
    estimated, and the survey it came from) and, with
    ``seg_tiling_control_px``, ``tiling_control.json`` (:func:`tiling_control`
    on a centred region of the control site's cell channels, sized by
    :func:`control_layout`). The nucleus
    diameter, when not fixed, and the tiling-control site come from
    ``seg_diameter_sites`` sites spread over the well, of which only those
    with at least ``seg_diameter_min_objects`` nuclei count
    (:func:`survey_sites`), never from one site that may be empty.

    :param settings: item 921 settings; ``seg_sites_src`` (the well folder)
        and ``seg_dst`` are required.
    :param models: ``(nucleus_model, cell_model)``; built by
        :func:`preset_models` when ``None``, on the card when ``seg_gpu``.
    :param available_bytes: passed to :func:`plan_site_memory`.
    :returns: ``plan``, ``sites`` (the site table as records), ``dst`` and
        ``nucleus_diameter`` (what ``nucleus_diameter.json`` holds).
    :raises ValueError: when ``seg_sites_src`` or ``seg_dst`` is missing, or
        when no sampled site holds enough nuclei to estimate the diameter.
    :raises MemoryPlanError: when the RAM cap cannot be met.
    """
    from .resource_log import ResourceSampler
    from .tabular import write_table

    settings = large_site_settings(settings)
    src, dst = settings.get("seg_sites_src"), settings.get("seg_dst")
    if not src or not dst:
        raise ValueError("seg_sites_src and seg_dst are required")
    channels = list(settings["seg_site_channels"])
    sites = [str(s) for s in (settings.get("seg_sites") or list_sites(src, channels))]
    os.makedirs(os.path.join(dst, "labels"), exist_ok=True)
    os.makedirs(os.path.join(dst, "tables"), exist_ok=True)
    os.makedirs(os.path.join(dst, "resource_log"), exist_ok=True)
    with open(os.path.join(dst, "settings.json"), "w", encoding="utf-8") as handle:
        json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in settings.items()},
                  handle, indent=2, sort_keys=True, default=str)
    if models is None:
        models = preset_models(settings, gpu=bool(settings.get("seg_gpu")))
    plan = None
    diameter = settings.get("nucleus_diameter")
    control_px = int(settings.get("seg_tiling_control_px") or 0)
    estimate = not diameter and bool(settings.get("seg_preset"))
    survey: Dict[str, Any] = {}
    if estimate or control_px:
        sampled = sample_sites(sites, int(settings.get("seg_diameter_sites") or 1))
        found = {}
        for site in sampled:
            stack = read_cropped_site(src, site, channels)
            if plan is None:
                plan = plan_site_memory(stack.shape[:2], len(channels), settings,
                                        available_bytes=available_bytes)
            local = dict(settings, seg_tile_px=plan["tile_px"])
            dna = normalize_site(stack[..., channels.index(local.get("seg_nucleus_channel")
                                                           or "CorrDNA")])
            found[site] = segment_tiled(
                dna, lambda x: _eval_labels(models[0], x, local, None),
                tile_px=plan["tile_px"], overlap_px=int(local["seg_tile_overlap_px"]),
                iou=float(local["seg_stitch_iou"]), align=True)
            del stack, dna
        survey = survey_sites(sampled, found,
                              tuple(settings.get("seg_nucleus_diameter_range") or (10.0, 80.0)),
                              min_objects=int(settings.get("seg_diameter_min_objects") or 1))
        del found
        if estimate:
            diameter = survey["diameter_px"]
    survey.update({"nucleus_diameter_px": diameter,
                   "nucleus_diameter_source": ("estimated" if estimate
                                               else "fixed" if diameter else "none")})
    with open(os.path.join(dst, "nucleus_diameter.json"), "w", encoding="utf-8") as handle:
        json.dump(survey, handle, indent=2, sort_keys=True)
    rows = []
    for site in sites:
        sampler = ResourceSampler(path=os.path.join(dst, "resource_log", f"site_{site}.jsonl"),
                                  level="summary", interval=1.0, label=f"site {site}")
        started = time.perf_counter()
        with sampler:
            sampler.sample_once()
            stack = read_cropped_site(src, site, channels)
            if plan is None:
                plan = plan_site_memory(stack.shape[:2], len(channels), settings,
                                        available_bytes=available_bytes)
            local = dict(settings, seg_tile_px=plan["tile_px"])
            if control_px and site == survey.get("control_site"):
                overlap = int(local["seg_tile_overlap_px"])
                side, control_tile, _ = control_layout(min(control_px, *stack.shape[:2]), overlap)
                y0, x0 = (stack.shape[0] - side) // 2, (stack.shape[1] - side) // 2
                body = normalize_site(np.stack([stack[..., channels.index(c)]
                                                for c in local["cell_channels"]], axis=-1))
                region = np.ascontiguousarray(body[y0:y0 + side, x0:x0 + side])
                del body
                control = tiling_control(
                    region, lambda x: _eval_labels(models[1], x, local, None, channel_axis=-1),
                    tile_px=control_tile, overlap_px=overlap,
                    iou=float(local["seg_stitch_iou"]), align=True)
                control.update({"site": site, "region_px": side, "origin": [y0, x0],
                                "site_nuclei": survey["objects_per_site"][site]})
                with open(os.path.join(dst, "tiling_control.json"), "w", encoding="utf-8") as handle:
                    json.dump(control, handle, indent=2, sort_keys=True)
            nuclei, cells, paired = segment_cell_painting_site(stack, channels, local, models,
                                                               nucleus_diameter=diameter)
            cells_kept, edge = apply_edge_policy(cells, local["seg_edge_policy"])
            nuclei_kept = np.where(np.isin(nuclei, np.unique(cells_kept)), nuclei, 0)
            table = measure_chunked(cells_kept, stack, band_px=int(local["seg_measure_band_px"]),
                                    channel_names=channels)
            sampler.sample_once()
        np.save(os.path.join(dst, "labels", f"{site}_nuclei.npy"), nuclei_kept.astype(np.int32))
        np.save(os.path.join(dst, "labels", f"{site}_cells.npy"), cells_kept.astype(np.int32))
        table.insert(0, "site", site)
        write_table(table, os.path.join(dst, "tables", f"{site}_cells.parquet"), canonicalise=False)
        row: Dict[str, Any] = {"site": site, "cells": int(table.shape[0]),
                               "edge_objects": edge["edge_objects"], "excluded_edge": edge["excluded"],
                               "nucleus_diameter_px": diameter,
                               "nucleus_diameter_sites": ";".join(survey.get("sites_used", [])
                                                                  if estimate else []),
                               "seconds": round(time.perf_counter() - started, 2)}
        row.update({f"paired_{k}": int(v) for k, v in paired.items()})
        peak = sampler.summary().get("peak_total")
        row["peak_memory_gb"] = round(peak / 2 ** 30, 3) if isinstance(peak, int) else float("nan")
        row["within_cap"] = bool(peak is not None and peak <= plan["cap_gb"] * 2 ** 30)
        for kind, mine in (("Cells", cells), ("Nuclei", nuclei)):
            ref = _reference_mask(settings.get("seg_report_reference"), site, kind)
            if ref is not None:
                score = mask_agreement(mine, ref)
                row.update({f"{kind.lower()}_{k}": v for k, v in score.items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    write_table(frame, os.path.join(dst, "sites.csv"), canonicalise=False)
    if "cells_f1" in frame:
        well = {"sites_with_reference": int(frame["cells_f1"].notna().sum())}
        for kind in ("cells", "nuclei"):
            for metric in ("f1", "median_iou", "count_ratio"):
                column = f"{kind}_{metric}"
                if column in frame:
                    well[f"{column}_median"] = float(frame[column].median())
            if f"{kind}_count_ratio" in frame:
                ratio = frame[f"{kind}_count_ratio"]
                well[f"{kind}_sites_outside_10pct"] = ";".join(
                    frame.loc[(ratio - 1).abs() > 0.10, "site"].astype(str))
        write_table(pd.DataFrame([well]), os.path.join(dst, "agreement_well.csv"),
                    canonicalise=False)
    return {"plan": plan, "sites": rows, "dst": dst, "nucleus_diameter": survey}
