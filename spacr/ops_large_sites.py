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
  nuclei on DNA, small nucleus pieces merged into the nucleus they touch
  (:func:`merge_small_nuclei`), one cell grown from every nucleus
  (:func:`cells_from_nucleus_seeds`) inside the cell network's foreground on
  ``[Phalloidin, DNA]``, with the plate's nucleus diameter and the
  tiling-control site taken from populated sites spread over the well
  (:func:`survey_sites`);
* the agreement report against reference label masks (:func:`mask_agreement`).

:func:`run_large_site_segmentation` is the settings-file entry. The memory
model of the planner is an estimate; the peak memory actually used per site
is measured with :class:`spacr.resource_log.ResourceSampler` and written to
the site table, so the estimate can be checked against every run.
"""
from __future__ import annotations

import glob
import json
import logging
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
    "seg_cells_from_nucleus_seeds": True,
    "seg_merge_small_nuclei_frac": 0.25,
    "seg_mask_vram_gb": 6.0,
    "seg_mask_device": "auto",
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

#: How far in from a cut tile edge Cellpose's flows still depend on the tile:
#: the first and last network tiles of a tile start 8 px into its padding, so
#: they reach ``NET_TILE_PX - 8`` px in. Tiles whose flows are stitched
#: overlap by at least twice this, so that every pixel lies at least this far
#: from the cut edges of the tile it is taken from.
FLOW_MARGIN_PX = NET_TILE_PX - 8

#: Bytes per site pixel for the stitched flows (three float32 planes) and for
#: Cellpose's mask step on the whole site (flow following, the seed histogram
#: and the flow check of every mask). Measured on CP186A Well1 site 76 at
#: 5,500 px: 52 bytes per pixel with 10 % and 205 with 60 % of pixels inside
#: cells; 330 covers a site whose every pixel is inside a cell.
SITE_FLOW_BYTES_PER_PIXEL = 3 * 4 + 330


#: Where ``seg_mask_device`` may put Cellpose's whole-site mask step (item 945).
MASK_DEVICES = ("auto", "gpu", "cpu")

#: The GPU governor's call-site name for the whole-site mask step.
MASK_STEP_SITE = "ops_large_sites_mask_step"

#: Card bytes of the mask step's flow following and seed finding
#: (``cellpose.dynamics.compute_masks`` without the flow check): per site
#: pixel the flow field, the seed histogram and its maximum, the label grid;
#: per cell pixel (cell probability above 0) the positions, the sampled flows
#: and the final positions. Estimates from Cellpose 4's arrays, rounded up.
MASK_DYNAMICS_BYTES_PER_PIXEL = 32
MASK_DYNAMICS_BYTES_PER_CELL_PIXEL = 64

#: Card bytes of Cellpose's flow check (``flow_error`` through
#: ``masks_to_flows_gpu``): per pixel of the checked region the padded label
#: image and the diffusion map; per mask pixel the 9-neighbour index tables,
#: the neighbour flags and the gathered values. The whole-site step measured
#: 342 bytes per cell pixel in all on CP186A Well1 site 76; the flow check is
#: most of it.
FLOW_CHECK_BYTES_PER_PIXEL = 24
FLOW_CHECK_BYTES_PER_MASK_PIXEL = 320

#: Cellpose's ``_extend_centers_gpu`` diffuses in float32 above this many
#: padded pixels and in float64 below; a checked band of a site above it is
#: padded with empty rows past it, so its masks diffuse in the same precision.
CELLPOSE_FLOAT32_PIXELS = 4e7

_LOG = logging.getLogger("spacr.ops_large_sites")

#: The item 956 agreement-report settings and their defaults. They are read
#: on their own (:func:`_agreement_settings`), not merged into
#: ``LARGE_SITE_DEFAULTS``, so a run that leaves them unset writes the same
#: ``settings.json`` as before.
MASK_AGREEMENT_DEFAULTS: Dict[str, Any] = {
    "ops_mask_agreement_split_merge": False,
    "ops_mask_agreement_identity_control": False,
    "ops_mask_agreement_displace_px": 30,
    "ops_mask_agreement_labels": None,
    "ops_mask_agreement_frame": "same",
}

#: The frames ``ops_mask_agreement_frame`` may name.
MASK_AGREEMENT_FRAMES = ("same", "periscope_align")

#: Site-bootstrap resamples and seed of the pooled agreement row.
MASK_AGREEMENT_BOOTSTRAP = 2000
MASK_AGREEMENT_SEED = 0

#: Tooltips of the item 945, 956 and 965 settings (the other item 921 keys
#: have none yet).
LARGE_SITE_TOOLTIPS: Dict[str, str] = {
    "seg_mask_vram_gb":
        "(float) - The most GPU memory, in GB, the job may hold while "
        "Cellpose's whole-site mask step runs; what it already holds, such as "
        "the models, counts. A site that needs more has its flow check run in "
        "row bands that fit; a step that cannot fit, or that runs out of GPU "
        "memory, runs on the CPU, and sites.csv says so. 0 sets no cap of its "
        "own (the GPU governor's free-memory budget and gpu_memory_fraction "
        "still apply). Default 6. Alpha.",
    "seg_mask_device":
        "(str) - Where the whole-site mask step runs when the models are on a "
        "GPU: 'auto' sizes it to seg_mask_vram_gb, 'gpu' runs it whole on the "
        "card as before (falling back to the CPU only on an out-of-memory "
        "error), 'cpu' always on the CPU. Default 'auto'. Alpha.",
    "seg_merge_small_nuclei_frac":
        "(float) - Before cells are grown from the nuclei, every nucleus "
        "piece smaller than this fraction of the site's median nucleus area "
        "is merged into the touching nucleus it shares the longest border "
        "with. A chain of small pieces ends on a kept nucleus; a small piece "
        "that touches no nucleus is kept, never deleted. 0 is off. Default "
        "0.25. Alpha.",
    "ops_mask_agreement_split_merge":
        "(bool) - Add split and merge counts to the agreement with "
        "seg_report_reference, written to mask_agreement.csv: each object goes "
        "to the object of the other image holding more than half of its "
        "pixels, border objects dropped. A reference object that receives two "
        "or more objects is a split, an object that receives two or more "
        "reference objects a merge; rates are per reference object. Also F1 at "
        "IoU 0.3 and 0.7, and a pooled row with site-bootstrap 95 % CIs. "
        "Default False. Alpha.",
    "ops_mask_agreement_identity_control":
        "(bool) - In mask_agreement.csv, score the reference against the same "
        "file read again: a working matcher gives F1 1, 0 splits and 0 merges. "
        "Default False. Alpha.",
    "ops_mask_agreement_displace_px":
        "(int) - With the agreement report on, a negative control: the labels "
        "are moved this many px further down and right and scored again, and "
        "their F1 must collapse. 0 is off. Default 30. Alpha.",
    "ops_mask_agreement_labels":
        "(str) - Score the saved labels in this folder (<site>_nuclei.npy and "
        "<site>_cells.npy, or a run's seg_dst holding labels/) instead of "
        "segmenting: only mask_agreement.csv and settings.json are written to "
        "seg_dst. Needs the split-merge or identity report on. Default None. "
        "Alpha.",
    "ops_mask_agreement_frame":
        "(str) - The frame the labels are scored in. 'same': as they are. "
        "'periscope_align': moved by minus PERISCOPE's "
        "Align_Yshift/Xshift_DAPI_Painting from the Image.csv of the reference "
        "mask's site folder, and both images cut to their overlap. Default "
        "'same'. Alpha.",
}

#: Settings-panel headings of the item 945 settings.
LARGE_SITE_CATEGORIES: Dict[str, List[str]] = {
    "Mask step GPU memory": ["seg_mask_device", "seg_mask_vram_gb"],
    "Mask agreement report": ["ops_mask_agreement_split_merge",
                              "ops_mask_agreement_identity_control",
                              "ops_mask_agreement_displace_px",
                              "ops_mask_agreement_labels",
                              "ops_mask_agreement_frame"],
}


class MemoryPlanError(ValueError):
    """The per-worker RAM cap cannot be met for this site."""


def large_site_settings(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Item 921's settings with defaults, the preset expanded and checked.

    Values the caller sets win over the preset; the preset wins over the
    defaults.

    :param settings: the caller's settings.
    :returns: a new dict.
    :raises ValueError: for an unknown preset, edge policy or mask device, or
        a negative ``seg_mask_vram_gb``.
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
    if out["seg_mask_device"] not in MASK_DEVICES:
        raise ValueError(f"seg_mask_device must be one of {list(MASK_DEVICES)}; "
                         f"got {out['seg_mask_device']!r}")
    if float(out["seg_mask_vram_gb"] or 0) < 0:
        raise ValueError(f"seg_mask_vram_gb must be 0 or more; got {out['seg_mask_vram_gb']!r}")
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
    boxes = tile_grid(shape, tile_px, min(int(overlap_px), max(int(tile_px) - 1, 0)))
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
    (int32), the site's stitched flows and their mask post-processing
    (:data:`SITE_FLOW_BYTES_PER_PIXEL`), the network and one tile in flight
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
    fixed = pixels * (2 * int(n_channels) + 2 * 4 + SITE_FLOW_BYTES_PER_PIXEL) + MODEL_BYTES
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


def _flow_bounds(starts: Sequence[int], ends: Sequence[int], size: int) -> List[Tuple[int, int]]:
    """The span each tile gives along one axis: neighbours split their
    shared strip at its middle."""
    bounds, lo = [], 0
    for i, (start, end) in enumerate(zip(starts, ends)):
        hi = size if i == len(starts) - 1 else (starts[i + 1] + end) // 2
        bounds.append((max(lo, start), hi))
        lo = hi
    return bounds


def stitch_tile_flows(tiles: Sequence[Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray]],
                      shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """One flow field and cell probability map from per-tile network outputs.

    Each pixel is taken from one tile: along each axis, two neighbouring
    tiles split the strip they share at its middle, so a pixel comes from
    the tile in which it lies deepest. With tiles from
    :func:`tiling_layout` (``align=True``) that overlap by at least
    ``2 * FLOW_MARGIN_PX``, every pixel's value is the one Cellpose computes
    for the untiled image, bit for bit.

    :param tiles: ``((y0, y1, x0, x1), flows, cellprob)`` per tile, in the
        row-major order of :func:`tile_grid`; ``flows`` is ``(2, h, w)``.
    :param shape: the full image ``(height, width)``.
    :returns: ``(flows, cellprob)`` of shapes ``(2, H, W)`` and ``(H, W)``,
        float32.
    """
    boxes = [box for box, _, _ in tiles]
    rows = sorted({(b[0], b[1]) for b in boxes})
    cols = sorted({(b[2], b[3]) for b in boxes})
    row_span = dict(zip(rows, _flow_bounds([r[0] for r in rows], [r[1] for r in rows],
                                           int(shape[0]))))
    col_span = dict(zip(cols, _flow_bounds([c[0] for c in cols], [c[1] for c in cols],
                                           int(shape[1]))))
    flows = np.zeros((2, int(shape[0]), int(shape[1])), np.float32)
    cellprob = np.zeros((int(shape[0]), int(shape[1])), np.float32)
    for (y0, y1, x0, x1), tile_flows, tile_prob in tiles:
        a0, a1 = row_span[(y0, y1)]
        b0, b1 = col_span[(x0, x1)]
        flows[:, a0:a1, b0:b1] = np.asarray(tile_flows)[:, a0 - y0:a1 - y0, b0 - x0:b1 - x0]
        cellprob[a0:a1, b0:b1] = np.asarray(tile_prob)[a0 - y0:a1 - y0, b0 - x0:b1 - x0]
    return flows, cellprob


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


def _flow_overlap(tile_px: int, overlap_px: int, align: bool, flows: Any, masks: Any) -> int:
    """The overlap flow stitching needs, or 0 when the tiles' flows are not
    stitched (no ``align``, ``flows`` or ``masks``, or no aligned tile of
    ``tile_px`` leaves that much overlap)."""
    if not (align and flows is not None and masks is not None):
        return 0
    overlap = max(int(overlap_px), 2 * FLOW_MARGIN_PX)
    return overlap if aligned_tiling(tile_px, overlap) is not None else 0


def segment_tiled(image: np.ndarray, segment: Callable[[np.ndarray], np.ndarray], *,
                  tile_px: int = 2048, overlap_px: int = 256, iou: float = 0.5,
                  align: bool = False,
                  flows: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
                  masks: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
                  ) -> np.ndarray:
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
    :param flows: with ``masks`` and ``align``, the network part of
        ``segment`` (an image to its ``(flows, cellprob)``): the tiles'
        network outputs are stitched (:func:`stitch_tile_flows`) and
        ``masks`` turns the whole image's flows into labels once, so that
        Cellpose's mask steps, which look at the whole image, see what they
        see untiled. The overlap is raised to ``2 * FLOW_MARGIN_PX``.
    :param masks: ``(flows, cellprob)`` to a label image.
    :returns: an int32 label image of the image's height and width.
    """
    shape = tuple(int(v) for v in np.shape(image)[:2])
    by_flows = _flow_overlap(tile_px, overlap_px, align, flows, masks)
    if by_flows:
        overlap_px = by_flows
    layout = tiling_layout(shape, tile_px, overlap_px, align)
    if len(layout["boxes"]) == 1:
        return np.asarray(segment(image))
    padded = layout["padded"]
    if padded != shape:
        image = np.pad(image, [(0, padded[0] - shape[0]), (0, padded[1] - shape[1])]
                       + [(0, 0)] * (np.ndim(image) - 2))
    if by_flows:
        stitched = stitch_tile_flows(
            [((y0, y1, x0, x1),) + tuple(flows(image[y0:y1, x0:x1]))
             for y0, y1, x0, x1 in layout["boxes"]], padded)
        del image
        labels = np.asarray(masks(*stitched), np.int32)
        return relabel_raster(labels[:shape[0], :shape[1]])
    tiles = [((y0, y1, x0, x1), np.asarray(segment(image[y0:y1, x0:x1])))
             for y0, y1, x0, x1 in layout["boxes"]]
    labels = stitch_tile_labels(tiles, padded, iou)
    return labels if padded == shape else relabel_raster(labels[:shape[0], :shape[1]])


def tiling_control(image: np.ndarray, segment: Callable[[np.ndarray], np.ndarray], *,
                   tile_px: int, overlap_px: int, iou: float = 0.5,
                   align: bool = False,
                   flows: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
                   masks: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
                   ) -> Dict[str, Any]:
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
    :param flows: passed to :func:`segment_tiled`.
    :param masks: passed to :func:`segment_tiled`.
    :returns: ``objects``, ``away_from_seams``, ``away_identical``,
        ``at_seams``, ``at_seams_iou_ge_0_9``, ``at_seams_share`` (``nan``
        without seam objects), and ``tile_px``, ``overlap_px`` and
        ``aligned`` of the tiles used.
    """
    shape = tuple(int(v) for v in np.shape(image)[:2])
    untiled = relabel_raster(np.asarray(segment(image)))
    tiled = segment_tiled(image, segment, tile_px=tile_px, overlap_px=overlap_px, iou=iou,
                          align=align, flows=flows, masks=masks)
    overlap_px = _flow_overlap(tile_px, overlap_px, align, flows, masks) or overlap_px
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


def _eval_flows(model: Any, image: np.ndarray, settings: Mapping[str, Any],
                **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Cellpose's network outputs for a site-normalised image, without masks."""
    from ._gpu_governor import _governed_eval

    out = _governed_eval(model, image, site="ops_large_sites", settings=settings, batch_size=8,
                         normalize=False, compute_masks=False, **kwargs)
    return np.asarray(out[1][1], np.float32), np.asarray(out[1][2], np.float32)


def _mask_setting(settings: Optional[Mapping[str, Any]], key: str) -> Any:
    """One item 945 setting, its default when unset or ``None``."""
    value = (settings or {}).get(key)
    return LARGE_SITE_DEFAULTS[key] if value is None else value


def mask_step_bytes(shape: Tuple[int, int], cell_pixels: int) -> Dict[str, int]:
    """Estimated card bytes of Cellpose's mask step on one whole image.

    :param shape: the image's ``(height, width)``.
    :param cell_pixels: pixels with cell probability above 0.
    :returns: ``dynamics`` (flow following and seed finding), ``flow_check``
        (the flow check of every mask at once) and ``whole`` (the larger of
        the two, since the first is freed before the second runs).
    """
    height, width = int(shape[0]), int(shape[1])
    dynamics = (height * width * MASK_DYNAMICS_BYTES_PER_PIXEL
                + int(cell_pixels) * MASK_DYNAMICS_BYTES_PER_CELL_PIXEL)
    check = _flow_check_bytes(height, width, int(cell_pixels))
    return {"dynamics": int(dynamics), "flow_check": int(check),
            "whole": int(max(dynamics, check))}


def _flow_check_bytes(rows: int, width: int, mask_pixels: int) -> int:
    """Estimated card bytes of Cellpose's flow check on a band of ``rows`` x
    ``width`` pixels holding ``mask_pixels`` mask pixels."""
    return int((int(rows) + 2) * (int(width) + 2) * FLOW_CHECK_BYTES_PER_PIXEL
               + int(mask_pixels) * FLOW_CHECK_BYTES_PER_MASK_PIXEL)


def _float32_rows(shape: Tuple[int, int]) -> int:
    """The fewest rows a checked band needs to diffuse in the precision of the
    whole image: 0 when the whole image diffuses in float64, else enough rows
    to put the padded band above :data:`CELLPOSE_FLOAT32_PIXELS` too."""
    height, width = int(shape[0]), int(shape[1])
    if (height + 2) * (width + 2) <= CELLPOSE_FLOAT32_PIXELS:
        return 0
    return int(CELLPOSE_FLOAT32_PIXELS // (width + 2)) - 1


def _touching(masks: np.ndarray) -> Dict[int, set]:
    """The labels each mask touches side by side (4-connected)."""
    pairs = []
    for one, two in ((masks[:-1, :], masks[1:, :]), (masks[:, :-1], masks[:, 1:])):
        where = (one != two) & (one > 0) & (two > 0)
        pairs.append(np.stack([one[where], two[where]], axis=1))
    found: Dict[int, set] = {}
    for one, two in np.unique(np.concatenate(pairs).astype(np.int64), axis=0):
        found.setdefault(int(one), set()).add(int(two))
        found.setdefault(int(two), set()).add(int(one))
    return found


def flow_check_bands(masks: np.ndarray, budget_bytes: float
                     ) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    """Full-width row bands of masks whose flow check fits ``budget_bytes``.

    Cellpose's flow of a mask's edge pixel reads the diffusion of the masks
    it touches, so a band holds its own masks and, as context, every mask
    they touch, each wholly inside the band's rows. Masks are taken in order
    of their top row and added to a band while its estimate
    (:data:`FLOW_CHECK_BYTES_PER_PIXEL` per band pixel and
    :data:`FLOW_CHECK_BYTES_PER_MASK_PIXEL` per pixel of its masks and
    context; a band of a site above :data:`CELLPOSE_FLOAT32_PIXELS` counts
    the padding rows it gets) fits; a mask that does not
    fit an empty band gets a band of its own. Every mask is checked in
    exactly one band.

    :param masks: a label image with consecutive labels ``1..n``.
    :param budget_bytes: card bytes one band may take.
    :returns: ``(y0, y1, labels, context)`` per band, both ascending.
    """
    from scipy.ndimage import find_objects

    masks = np.asarray(masks)
    slices = find_objects(masks)
    areas = np.bincount(masks.ravel())
    touching = _touching(masks)
    width = int(masks.shape[1])
    least = _float32_rows(masks.shape)
    order = sorted((s[0].start, label + 1) for label, s in enumerate(slices) if s is not None)

    def extent(members: set) -> Tuple[int, int, int]:
        """Top row, bottom row and pixels of a set of masks."""
        return (min(slices[m - 1][0].start for m in members),
                max(slices[m - 1][0].stop for m in members),
                int(sum(int(areas[m]) for m in members)))

    def close(core: List[int], held: set) -> Tuple[int, int, np.ndarray, np.ndarray]:
        """One finished band."""
        top, bottom, _ = extent(held)
        return (top, bottom, np.array(sorted(core), np.int64),
                np.array(sorted(held - set(core)), np.int64))

    bands: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
    core: List[int] = []
    held: set = set()
    for _, label in order:
        grown = held | {label} | touching.get(label, set())
        top, bottom, pixels = extent(grown)
        cost = _flow_check_bytes(max(bottom - top, least), width, pixels)
        if core and cost > budget_bytes:
            bands.append(close(core, held))
            core, grown = [], {label} | touching.get(label, set())
        core.append(label)
        held = grown
    if core:
        bands.append(close(core, held))
    return bands


def _band_bytes(masks: np.ndarray, band: Tuple[int, int, np.ndarray, np.ndarray]) -> int:
    """Estimated card bytes of one band of :func:`flow_check_bands`."""
    held = np.concatenate([band[2], band[3]])
    pixels = int(np.bincount(masks.ravel(), minlength=int(held.max()) + 1)[held].sum())
    return _flow_check_bytes(max(band[1] - band[0], _float32_rows(masks.shape)),
                            masks.shape[1], pixels)


def _whole_mask_step(flows: np.ndarray, cellprob: np.ndarray, device: Any) -> np.ndarray:
    """Cellpose's mask step on a whole image's flows on ``device``, with
    ``eval``'s defaults for an image that is not resized: item 921's call."""
    from cellpose.dynamics import resize_and_compute_masks

    return resize_and_compute_masks(
        flows, cellprob, niter=200, cellprob_threshold=0.0, flow_threshold=0.4, do_3D=False,
        min_size=15, max_size_fraction=0.4, resize=None, device=device)


def _mask_dynamics(flows: np.ndarray, cellprob: np.ndarray, device: Any) -> np.ndarray:
    """The mask step without its flow check: Cellpose's ``compute_masks``
    with ``flow_threshold=0``, everything else as :func:`_whole_mask_step`."""
    from cellpose.dynamics import compute_masks

    return compute_masks(flows, cellprob, niter=200, cellprob_threshold=0.0,
                         flow_threshold=0.0, do_3D=False, max_size_fraction=0.4,
                         device=device)


def _band_flow_errors(masks: np.ndarray, flows: np.ndarray,
                      band: Tuple[int, int, np.ndarray, np.ndarray],
                      niter: int, rows: int, device: Any) -> np.ndarray:
    """Cellpose's flow error (``cellpose.dynamics.flow_error``) of the masks
    of one band, in the order of the band's labels.

    The band holds its masks and their context (the masks they touch),
    relabelled ``1..k`` in label order, and is padded with empty rows to at
    least ``rows``. Each mask's diffusion reads only its own pixels, an edge
    pixel's flow reads only the masks it touches, a centre comes from the
    mask's own bounding box and ``niter`` is the whole image's, so every
    error equals the whole-image flow check's up to PyTorch's summation
    order (measured: at most one unit in the last place, 1.2e-16 relative),
    which changes a label only for a mask whose error lies that close to the
    0.4 threshold.
    """
    from cellpose.dynamics import masks_to_flows_gpu

    y0, y1, labels, context = band
    held = np.concatenate([labels, context])
    held.sort()
    lut = np.zeros(int(masks.max()) + 1, np.int32)
    lut[held] = np.arange(1, len(held) + 1, dtype=np.int32)
    part = np.zeros((max(y1 - y0, int(rows)), masks.shape[1]), np.int32)
    part[:y1 - y0] = lut[masks[y0:y1]]
    flow, slices = masks_to_flows_gpu(part, device=device, niter=int(niter))
    error = ((flow[:, :y1 - y0] - flows[:, y0:y1] / 5.) ** 2).sum(axis=0)
    return np.array([error[slices[j - 1][0], slices[j - 1][1]][
        part[slices[j - 1][0], slices[j - 1][1]] == j].mean() for j in lut[labels]])


def _mask_step_budget(device: Any, settings: Mapping[str, Any]) -> Optional[float]:
    """Card bytes the mask step may allocate: the GPU governor's budget (free
    memory less its margin and other spaCR jobs' claims, under
    ``gpu_memory_fraction``), and no more than ``seg_mask_vram_gb`` less what
    this job already holds. ``None`` when the card cannot be read."""
    from . import _gpu_governor as governor

    index = governor._device_index(device)
    if index is None:
        return None
    config = governor._config(settings)
    governor._release()
    budget = governor._controller(MASK_STEP_SITE, device)._budget(index, config)
    cap = float(_mask_setting(settings, "seg_mask_vram_gb") or 0)
    if cap > 0:
        capped = cap * 2 ** 30 - governor._held(index)
        budget = capped if budget is None else min(budget, capped)
    return None if budget is None else max(0.0, float(budget))


def _banded_flow_check(masks: np.ndarray, flows: np.ndarray, device: Any, budget: float,
                       settings: Mapping[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Cellpose's flow check (``remove_bad_flow_masks``, threshold 0.4) run
    band by band (:func:`flow_check_bands`) under the GPU governor.

    A band whose estimate exceeds ``budget`` runs on the CPU; a band that
    runs out of card memory is retried by the governor and then run on the
    CPU, and so are the bands after it.

    :returns: the checked masks and ``bands``, ``devices`` and ``reasons``.
    """
    import fastremap
    import torch
    from scipy.ndimage import find_objects

    from . import _gpu_governor as governor

    info: Dict[str, Any] = {"bands": 0, "devices": set(), "reasons": set()}
    if masks.max() == 0:
        return masks, info
    cpu = torch.device("cpu")
    slices = find_objects(masks)
    niter = 2 * max((s[0].stop - s[0].start) + (s[1].stop - s[1].start) + 2
                    for s in slices if s is not None)
    rows = _float32_rows(masks.shape)
    errors = np.zeros(len(slices))
    on_card = True
    bands = flow_check_bands(masks, budget)
    info["bands"] = len(bands)
    for band in bands:
        need = _band_bytes(masks, band)
        if on_card and need <= budget:
            fell: List[bool] = []

            def on_cpu(band=band, fell=fell):
                """The band on the CPU, noted as a fallback."""
                fell.append(True)
                return _band_flow_errors(masks, flows, band, niter, rows, cpu)

            found = governor._guarded(
                lambda band=band: _band_flow_errors(masks, flows, band, niter, rows, device),
                on_cpu, site=MASK_STEP_SITE, device=device, settings=settings)
            if fell:
                on_card = False
                info["reasons"].add("oom")
            info["devices"].add("cpu" if fell else "gpu")
        else:
            found = _band_flow_errors(masks, flows, band, niter, rows, cpu)
            info["devices"].add("cpu")
            info["reasons"].add("cap" if on_card else "oom")
        errors[band[2] - 1] = found
    bad = 1 + (errors > 0.4).nonzero()[0]
    fastremap.mask(masks, bad, in_place=True)
    fastremap.renumber(masks, in_place=True)
    return masks, info


def _governed_mask_step(flows: np.ndarray, cellprob: np.ndarray, device: Any,
                        settings: Mapping[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """The whole-site mask step for models on a CUDA card (item 945).

    ``seg_mask_device='cpu'`` runs it on the CPU. ``'gpu'``, or ``'auto'``
    when the estimate (:func:`mask_step_bytes`) fits the budget
    (:func:`_mask_step_budget`), runs item 921's call whole on the card
    through the GPU governor, so an out-of-memory error is retried and then
    run on the CPU instead of stopping the run. Otherwise the flow following
    runs on the card when it fits (else on the CPU) and the flow check in
    bands (:func:`_banded_flow_check`); the labels equal the whole step's on
    the same device (see :func:`_band_flow_errors` for the one exception),
    and differ between card and CPU only by their float order (item 921's
    tolerance: every object at IoU >= 0.9).

    :returns: the labels and a note: ``device`` (``'gpu'``, ``'gpu_banded'``,
        ``'cpu'`` or ``'mixed'``), ``reason`` (``'fits'``, ``'setting'``,
        ``'cap'``, ``'oom'``, joined by ``;`` when several), ``estimate_gb``,
        ``budget_gb`` and ``bands``.
    """
    import torch
    from cellpose.utils import fill_holes_and_remove_small_masks

    from . import _gpu_governor as governor

    cpu = torch.device("cpu")
    mode = _mask_setting(settings, "seg_mask_device")
    need = mask_step_bytes(np.shape(cellprob), int(np.count_nonzero(np.asarray(cellprob) > 0)))
    note: Dict[str, Any] = {"device": "cpu", "reason": "setting",
                            "estimate_gb": round(need["whole"] / 2 ** 30, 3),
                            "budget_gb": None, "bands": 0}
    if mode == "cpu":
        return _whole_mask_step(flows, cellprob, cpu), note
    budget = None if mode == "gpu" else _mask_step_budget(device, settings)
    note["budget_gb"] = None if budget is None else round(budget / 2 ** 30, 3)
    fell: List[bool] = []

    def fallback(step):
        """``step`` on the CPU, noted as a fallback."""
        def run():
            """Run the step on the CPU."""
            fell.append(True)
            return step(flows, cellprob, cpu)
        return run

    if budget is None or need["whole"] <= budget:
        labels = governor._guarded(lambda: _whole_mask_step(flows, cellprob, device),
                                   fallback(_whole_mask_step), site=MASK_STEP_SITE,
                                   device=device, settings=settings)
        note.update(device="cpu" if fell else "gpu",
                    reason="oom" if fell else ("setting" if mode == "gpu" else "fits"))
        return labels, note
    reasons = set()
    if need["dynamics"] <= budget:
        masks = governor._guarded(lambda: _mask_dynamics(flows, cellprob, device),
                                  fallback(_mask_dynamics), site=MASK_STEP_SITE,
                                  device=device, settings=settings)
        devices = {"cpu" if fell else "gpu"}
        if fell:
            reasons.add("oom")
    else:
        masks = _mask_dynamics(flows, cellprob, cpu)
        devices, reasons = {"cpu"}, {"cap"}
    masks, checked = _banded_flow_check(masks, flows, device, budget, settings)
    devices |= checked["devices"]
    reasons |= checked["reasons"]
    if masks.dtype == np.uint32 and masks.max() < 2 ** 16:
        masks = masks.astype(np.uint16)
    labels = fill_holes_and_remove_small_masks(masks, min_size=15)
    note.update(device=("gpu_banded" if devices == {"gpu"} else "cpu" if devices == {"cpu"}
                        else "mixed"),
                reason=";".join(sorted(reasons)) or "fits", bands=checked["bands"])
    return labels, note


def _cellpose_masks(model: Any, flows: np.ndarray, cellprob: np.ndarray, *,
                    settings: Optional[Mapping[str, Any]] = None,
                    record: Optional[List[Dict[str, Any]]] = None,
                    label: str = "") -> np.ndarray:
    """Cellpose's mask step on a whole image's flows, with ``eval``'s defaults
    for an image that is not resized.

    On a model off CUDA this is item 921's call on the model's device,
    unchanged. On a CUDA card it is sized to ``seg_mask_vram_gb`` and run
    through the GPU governor (:func:`_governed_mask_step`); a step that does
    not fit, or runs out of card memory, runs on the CPU and is logged.

    :param model: the Cellpose model whose device the step runs on.
    :param flows: ``(2, H, W)`` network flows.
    :param cellprob: ``(H, W)`` cell probability.
    :param settings: read for ``seg_mask_device``, ``seg_mask_vram_gb`` and
        the GPU governor keys.
    :param record: a list the step's note is appended to (``objects``,
        ``device``, ``reason``, ``estimate_gb``, ``budget_gb``, ``bands``).
    :param label: what the masks are, for the note and the log.
    :returns: int32 labels.
    """
    import torch

    device = getattr(model, "device", None) or torch.device("cpu")
    if getattr(device, "type", str(device).partition(":")[0]) != "cuda":
        labels = _whole_mask_step(flows, cellprob, device)
        note: Dict[str, Any] = {"device": "cpu", "reason": "model_on_cpu", "estimate_gb": None,
                                "budget_gb": None, "bands": 0}
    else:
        labels, note = _governed_mask_step(flows, cellprob, device, settings or {})
    note = dict(note, objects=label)
    if note["reason"] not in ("fits", "setting", "model_on_cpu"):
        _LOG.warning("[gpu] %s: the %s mask step ran on %s (%s; estimate %s GB, budget %s GB, "
                     "seg_mask_vram_gb=%s)", MASK_STEP_SITE, label or "whole-site",
                     note["device"], note["reason"], note["estimate_gb"], note["budget_gb"],
                     _mask_setting(settings, "seg_mask_vram_gb"))
    if record is not None:
        record.append(note)
    return np.asarray(labels, np.int32)


def summarise_mask_steps(notes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """One site's mask-step notes as the ``sites.csv`` columns.

    :param notes: what :func:`_cellpose_masks` recorded for the site.
    :returns: ``mask_step_device`` (the one device, ``'mixed'``, or
        ``'in_eval'`` when no stitched mask step ran because the site fitted
        one tile and Cellpose's ``eval`` made the masks under the governor),
        ``mask_step_reason`` and ``mask_step_bands``.
    """
    if not notes:
        return {"mask_step_device": "in_eval", "mask_step_reason": "",
                "mask_step_bands": 0}
    devices = {str(n["device"]) for n in notes}
    reasons = sorted({part for n in notes for part in str(n["reason"]).split(";") if part})
    return {"mask_step_device": devices.pop() if len(devices) == 1 else "mixed",
            "mask_step_reason": ";".join(reasons),
            "mask_step_bands": int(sum(int(n.get("bands") or 0) for n in notes))}


def segment_cell_painting_site(stack: np.ndarray, channel_names: Sequence[str],
                               settings: Mapping[str, Any], models: Tuple[Any, Any], *,
                               nucleus_diameter: Optional[float] = None,
                               mask_log: Optional[List[Dict[str, Any]]] = None
                               ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Nuclei and cells of one site with the ``cell_painting_phalloidin`` preset.

    Nuclei are segmented on the DNA channel, cells on ``cell_channels``
    (Phalloidin, then DNA), each normalised once on the whole site
    (:func:`normalize_site`) and tiled by :func:`segment_tiled` on Cellpose's
    network grid, with the tiles' flows stitched and Cellpose's mask step run
    once on the whole site (nuclei only without a diameter, since a diameter
    resizes each tile). With ``seg_merge_small_nuclei_frac`` above 0
    (default 0.25), nucleus pieces smaller than that fraction of the site's
    median nucleus area are then merged into the nucleus they touch
    (:func:`merge_small_nuclei`). With ``seg_cells_from_nucleus_seeds``
    (default on) one cell is grown from every nucleus inside the cell
    network's foreground (:func:`cells_from_nucleus_seeds`), so a cell
    outline that holds two nuclei no longer loses one; with it off, the
    cells are the cell model's own masks. Then each
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
    :param mask_log: a list each whole-site mask step's note is appended to
        (see :func:`_cellpose_masks`).
    :returns: ``(nuclei, cells, counts)`` with nucleus ``k`` inside cell ``k``;
        when small pieces were merged, ``counts`` also holds
        ``merged_small_nuclei`` and ``median_nucleus_area_px``.
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
                           tile_px=tile, overlap_px=overlap, iou=iou, align=not diameter,
                           flows=lambda x: _eval_flows(nucleus_model, x, settings),
                           masks=lambda f, p: _cellpose_masks(nucleus_model, f, p,
                                                              settings=settings, record=mask_log,
                                                              label="nuclei"))
    frac = float(settings.get("seg_merge_small_nuclei_frac") or 0)
    merged = None
    if frac > 0:
        nuclei, merged = merge_small_nuclei(nuclei, frac)
    body = normalize_site(np.stack([stack[..., names.index(c)] for c in settings["cell_channels"]],
                                   axis=-1))
    if settings.get("seg_cells_from_nucleus_seeds"):
        cellprob = _site_cellprob(body, lambda x: _eval_flows(cell_model, x, settings,
                                                              channel_axis=-1),
                                  tile_px=tile, overlap_px=overlap)
        del body
        out = pair_nuclei_with_cells(nuclei, cells_from_nucleus_seeds(nuclei, cellprob))
        return out if merged is None else _with_merge_counts(out, merged)
    cells = segment_tiled(body, lambda x: _eval_labels(cell_model, x, settings, None,
                                                       channel_axis=-1),
                          tile_px=tile, overlap_px=overlap, iou=iou, align=True,
                          flows=lambda x: _eval_flows(cell_model, x, settings, channel_axis=-1),
                          masks=lambda f, p: _cellpose_masks(cell_model, f, p, settings=settings,
                                                             record=mask_log, label="cells"))
    out = pair_nuclei_with_cells(nuclei, cells)
    return out if merged is None else _with_merge_counts(out, merged)


def _with_merge_counts(paired: Tuple[np.ndarray, np.ndarray, Dict[str, int]],
                       merged: Mapping[str, Any]
                       ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """The paired labels with :func:`merge_small_nuclei`'s merged-piece
    count and the site's median nucleus area added to their counts."""
    nuclei, cells, counts = paired
    counts = dict(counts, merged_small_nuclei=int(merged["n_merged"]),
                  median_nucleus_area_px=float(merged["median_area_px"]))
    return nuclei, cells, counts


def merge_small_nuclei(nuclei: np.ndarray, frac: float = 0.25
                       ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Merge small nucleus pieces into the nucleus they touch (item 965).

    Seeding one cell from every nucleus (:func:`cells_from_nucleus_seeds`)
    keeps the second nucleus of a cell that the cell model merged, but it
    also gives every small fragment of a nucleus its own cell. This step
    runs on the nucleus labels of a whole site, before the cells are
    seeded, so that every later step sees the merged nuclei.

    The threshold is ``frac`` times the median area of the site's nuclei,
    measured before any merge. A nucleus below it that shares a border
    with another nucleus (pixel pairs in 4-connectivity) is merged into the
    neighbour with which it shares the longest border; ties go to the
    larger neighbour, then to the lower label. When that neighbour is
    itself small and merged, the chain is followed to the kept nucleus it
    ends on; two small pieces that choose each other end on the larger
    one. A small piece that touches no nucleus is kept: the step merges
    and never deletes. The labels are then renumbered 1..n in the order of
    their old labels, so the result is deterministic.

    :param nuclei: a 2-D nucleus label image of one site.
    :param frac: the fraction of the median nucleus area below which a
        nucleus counts as a small piece; 0 or less returns the labels
        unchanged.
    :returns: ``(labels, info)``: an int32 label image and a dict with
        ``threshold_px``, ``median_area_px``, ``n_small`` (pieces below the
        threshold) and ``n_merged`` (pieces merged into another nucleus).
    """
    lab = np.asarray(nuclei)
    if frac <= 0 or lab.size == 0 or lab.max() == 0:
        return lab.astype(np.int32), {"threshold_px": 0.0, "median_area_px": 0.0,
                                      "n_small": 0, "n_merged": 0}
    area = np.bincount(lab.ravel().astype(np.int64))
    present = np.nonzero(area[1:])[0] + 1
    median = float(np.median(area[present]))
    thr = float(frac * median)
    small = np.zeros(area.size, bool)
    small[present[area[present] < thr]] = True
    pairs = []
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        m = (a != b) & (a > 0) & (b > 0)
        pairs.append(np.stack([a[m], b[m]], 1))
    p = np.concatenate(pairs).astype(np.int64)
    p = np.concatenate([p, p[:, ::-1]])
    p = p[small[p[:, 0]]]
    target = np.arange(area.size)
    if p.size:
        key, border = np.unique(p, axis=0, return_counts=True)
        order = np.lexsort((key[:, 1], -area[key[:, 1]], -border, key[:, 0]))
        key = key[order]
        first = np.ones(len(key), bool)
        first[1:] = key[1:, 0] != key[:-1, 0]
        target[key[first, 0]] = key[first, 1]
    for _ in range(64):
        nxt = target[target]
        if np.array_equal(nxt, target):
            break
        cyc = (nxt == np.arange(area.size)) & (target != np.arange(area.size))
        if cyc.any():
            i = np.nonzero(cyc)[0]
            j = target[i]
            keep = np.where((area[i] > area[j]) | ((area[i] == area[j]) & (i < j)), i, j)
            target[keep] = keep
            nxt = target[target]
        target = nxt
    n_merged = int((target[present] != present).sum())
    out = target[lab]
    ids = np.unique(out)
    lut = np.zeros(int(ids.max()) + 1, np.int64)
    lut[ids[ids > 0]] = np.arange(1, int((ids > 0).sum()) + 1)
    return lut[out].astype(np.int32), {"threshold_px": thr, "median_area_px": median,
                                       "n_small": int(small.sum()), "n_merged": n_merged}


def cells_from_nucleus_seeds(nuclei: np.ndarray, cellprob: np.ndarray, *,
                             cellprob_threshold: float = 0.0) -> np.ndarray:
    """Grow one cell from every nucleus inside the cell network's foreground.

    The cell model can merge two touching cells into one mask. Pairing then
    keeps one nucleus of the merged cell and drops the other, so both cells
    and nuclei are undercounted. Here every nucleus seeds its own cell
    instead. The foreground is every pixel whose cell probability is above
    ``cellprob_threshold``, together with the nuclei themselves. A seeded
    watershed divides it. The watershed floods the foreground outward from
    the nuclei, in order of the distance to the nearest nucleus. A pixel
    therefore goes to the nearest nucleus that it is connected to through
    the foreground. The boundary between two touching cells runs halfway
    between their nuclei, as with CellProfiler's distance method. It does
    not follow the cell model's own boundaries.

    The result does not depend on how the site was tiled, provided the
    nuclei and the cell probability cover the whole site. Foreground that no
    nucleus can reach stays 0.

    :param nuclei: a 2-D nucleus label image of the whole site.
    :param cellprob: the cell model's probability map (logits) of the same
        shape, as Cellpose returns it.
    :param cellprob_threshold: pixels above this value are foreground; 0
        matches the threshold of Cellpose's own mask step.
    :returns: an int32 cell label image in which cell ``k`` holds nucleus
        ``k`` whole.
    :raises ValueError: when the two images differ in shape or are not 2-D.
    """
    from scipy import ndimage
    from skimage.segmentation import watershed

    nuclei, cellprob = np.asarray(nuclei), np.asarray(cellprob)
    if nuclei.ndim != 2 or nuclei.shape != cellprob.shape:
        raise ValueError(f"nuclei {nuclei.shape} and cellprob {cellprob.shape} must be "
                         "2-D images of one shape")
    foreground = (cellprob > cellprob_threshold) | (nuclei > 0)
    distance = ndimage.distance_transform_edt(nuclei == 0)
    return watershed(distance, nuclei, mask=foreground).astype(np.int32)


def _site_cellprob(image: np.ndarray,
                   flows: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]], *,
                   tile_px: int, overlap_px: int) -> np.ndarray:
    """The network's cell probability for a whole site, from tiles on
    Cellpose's network grid with stitched outputs, as :func:`segment_tiled`
    places them; a site within one tile runs untiled. When no tile of
    ``tile_px`` fits the network grid with that overlap, the tiles of
    :func:`tile_grid` are stitched the same way."""
    shape = tuple(int(v) for v in np.shape(image)[:2])
    overlap = _flow_overlap(tile_px, overlap_px, True, flows, True) or int(overlap_px)
    layout = tiling_layout(shape, tile_px, overlap, True)
    if len(layout["boxes"]) == 1:
        return np.asarray(flows(image)[1], np.float32)
    padded = layout["padded"]
    if padded != shape:
        image = np.pad(image, [(0, padded[0] - shape[0]), (0, padded[1] - shape[1])]
                       + [(0, 0)] * (np.ndim(image) - 2))
    _, cellprob = stitch_tile_flows(
        [((y0, y1, x0, x1),) + tuple(flows(image[y0:y1, x0:x1]))
         for y0, y1, x0, x1 in layout["boxes"]], padded)
    return np.ascontiguousarray(cellprob[:shape[0], :shape[1]])


def _edge_step(nuclei: np.ndarray, cells: np.ndarray, settings: Mapping[str, Any]
               ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """The crop-edge policy on one site's paired labels.

    Cells follow ``seg_edge_policy``. A nucleus is kept with its cell, or,
    with ``seg_cells_from_nucleus_seeds``, by the same policy applied to the
    nucleus itself, so that it is dropped only when it touches the border.
    """
    cells_kept, edge = apply_edge_policy(cells, settings["seg_edge_policy"])
    if settings.get("seg_cells_from_nucleus_seeds"):
        nuclei_kept, _ = apply_edge_policy(nuclei, settings["seg_edge_policy"])
    else:
        nuclei_kept = np.where(np.isin(nuclei, np.unique(cells_kept)), nuclei, 0)
    return nuclei_kept, cells_kept, edge


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


def _agreement_settings(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """The item 956 report settings of a run, defaults filled in and checked."""
    out = {key: (default if settings.get(key) is None else settings.get(key))
           for key, default in MASK_AGREEMENT_DEFAULTS.items()}
    out["ops_mask_agreement_displace_px"] = int(out["ops_mask_agreement_displace_px"] or 0)
    if out["ops_mask_agreement_frame"] not in MASK_AGREEMENT_FRAMES:
        raise ValueError(f"ops_mask_agreement_frame must be one of {list(MASK_AGREEMENT_FRAMES)}; "
                         f"got {out['ops_mask_agreement_frame']!r}")
    if out["ops_mask_agreement_displace_px"] < 0:
        raise ValueError("ops_mask_agreement_displace_px must be 0 or more; "
                         f"got {out['ops_mask_agreement_displace_px']!r}")
    out["report"] = bool(out["ops_mask_agreement_split_merge"]
                         or out["ops_mask_agreement_identity_control"])
    if out["ops_mask_agreement_labels"] and not out["report"]:
        raise ValueError("ops_mask_agreement_labels scores saved labels for the agreement report; "
                         "turn on ops_mask_agreement_split_merge or "
                         "ops_mask_agreement_identity_control")
    if out["report"] and not settings.get("seg_report_reference"):
        raise ValueError("the mask agreement report needs seg_report_reference")
    return out


def _shift_labels(labels: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """``labels`` moved by ``(dy, dx)`` px, the uncovered pixels 0."""
    out = np.zeros_like(labels)
    h, w = labels.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = labels[ys, xs]
    return out


def _frame_overlap(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """The part of a frame that a frame moved by ``(dy, dx)`` also covers."""
    h, w = image.shape
    return image[abs(dy):h - abs(dy), abs(dx):w - abs(dx)]


def _split_merge(labels: np.ndarray, reference: np.ndarray) -> Dict[str, int]:
    """Split and merge counts by majority overlap, border objects dropped.

    Each object is assigned to the object of the other image that holds more
    than half of its pixels. A reference object receiving two or more
    objects is a split; an object receiving two or more reference objects is
    a merge.
    """
    labels, _ = apply_edge_policy(labels, "exclude")
    reference, _ = apply_edge_policy(reference, "exclude")
    a, b = labels.ravel().astype(np.int64), reference.ravel().astype(np.int64)
    area_a, area_b = np.bincount(a), np.bincount(b)
    both = (a > 0) & (b > 0)
    pairs, counts = np.unique(np.stack([a[both], b[both]]), axis=1, return_counts=True)
    a_major = counts > 0.5 * area_a[pairs[0]]
    b_major = counts > 0.5 * area_b[pairs[1]]
    per_ref = np.bincount(pairs[1][a_major], minlength=area_b.size)
    per_mine = np.bincount(pairs[0][b_major], minlength=area_a.size)
    return {"splits": int((per_ref >= 2).sum()), "merges": int((per_mine >= 2).sum())}


def _periscope_shift(mask_path: str) -> Tuple[int, int]:
    """Minus PERISCOPE's 04Align DAPI-to-painting shift ``(dy, dx)``, from
    the ``Image.csv`` of the site folder holding ``segmentation_masks``."""
    table = os.path.join(os.path.dirname(os.path.dirname(mask_path)), "Image.csv")
    if not os.path.isfile(table):
        raise FileNotFoundError(f"ops_mask_agreement_frame 'periscope_align' needs {table}")
    from .tabular import read_table

    row = read_table(table, canonicalise=False, report=None,
                     usecols=["Align_Xshift_DAPI_Painting", "Align_Yshift_DAPI_Painting"])
    return (-int(row["Align_Yshift_DAPI_Painting"].iloc[0]),
            -int(row["Align_Xshift_DAPI_Painting"].iloc[0]))


def _agreement_rows(labels_dir: str, reference_dir: str, sites: Sequence[str],
                    agree: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One agreement row per site and kind with saved labels and a reference."""
    import tifffile

    displace = agree["ops_mask_agreement_displace_px"]
    rows = []
    for site in sites:
        for kind in ("Nuclei", "Cells"):
            path = os.path.join(labels_dir, f"{site}_{kind.lower()}.npy")
            ref_path = _reference_mask_path(reference_dir, site, kind)
            if not os.path.isfile(path) or ref_path is None:
                continue
            ref = tifffile.imread(ref_path)
            dy, dx = (_periscope_shift(ref_path)
                      if agree["ops_mask_agreement_frame"] == "periscope_align" else (0, 0))
            mine = _shift_labels(np.load(path), dy, dx)
            a, r = _frame_overlap(mine, dy, dx), _frame_overlap(ref, dy, dx)
            row: Dict[str, Any] = {"site": str(site), "kind": kind.lower(), "dy": dy, "dx": dx}
            row.update(mask_agreement(a, r))
            if agree["ops_mask_agreement_split_merge"]:
                for iou in (0.3, 0.7):
                    score = mask_agreement(a, r, iou=iou)
                    row[f"f1_iou{iou}"], row[f"matched_iou{iou}"] = score["f1"], score["matched"]
                row.update(_split_merge(a, r))
                n_ref = row["n_reference"]
                row["split_rate"] = row["splits"] / n_ref if n_ref else float("nan")
                row["merge_rate"] = row["merges"] / n_ref if n_ref else float("nan")
            if agree["ops_mask_agreement_identity_control"]:
                again = tifffile.imread(ref_path)
                same = mask_agreement(ref, again)
                row["identity_f1"], row["identity_count_ratio"] = same["f1"], same["count_ratio"]
                same_sm = _split_merge(ref, again)
                row["identity_splits"], row["identity_merges"] = same_sm["splits"], same_sm["merges"]
            if displace:
                wrong = _frame_overlap(_shift_labels(mine, displace, displace), dy, dx)
                moved = mask_agreement(wrong, r)
                row["displaced_f1"], row["displaced_matched"] = moved["f1"], moved["matched"]
            rows.append(row)
            del ref, mine, a, r
    return rows


def _ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator``, NaN when the denominator is 0."""
    return numerator / denominator if denominator else float("nan")


def _pooled_stats(frame: pd.DataFrame, agree: Mapping[str, Any]) -> Dict[str, float]:
    """Agreement pooled over the sites of ``frame`` from summed counts."""
    ns, nr, m = (int(frame[c].sum()) for c in ("n_spacr", "n_reference", "matched"))
    out = {"f1": _ratio(2 * m, ns + nr), "count_ratio": _ratio(ns, nr),
           "precision": _ratio(m, ns), "recall": _ratio(m, nr)}
    if agree["ops_mask_agreement_split_merge"]:
        for iou in (0.3, 0.7):
            out[f"f1_iou{iou}"] = _ratio(2 * int(frame[f"matched_iou{iou}"].sum()), ns + nr)
        out["split_rate"] = _ratio(int(frame["splits"].sum()), nr)
        out["merge_rate"] = _ratio(int(frame["merges"].sum()), nr)
    if agree["ops_mask_agreement_displace_px"]:
        out["displaced_f1"] = _ratio(2 * int(frame["displaced_matched"].sum()), ns + nr)
    return out


def _pooled_row(frame: pd.DataFrame, agree: Mapping[str, Any]) -> Dict[str, Any]:
    """The pooled row of one kind: summed counts, pooled rates and their
    site-bootstrap 95 % CIs; the identity control's lowest F1 and count
    ratio and summed splits and merges."""
    rng = np.random.default_rng(MASK_AGREEMENT_SEED)
    point = _pooled_stats(frame, agree)
    draws = [_pooled_stats(frame.iloc[rng.integers(0, len(frame), len(frame))], agree)
             for _ in range(MASK_AGREEMENT_BOOTSTRAP)]
    row: Dict[str, Any] = {"site": "pooled", "kind": frame["kind"].iloc[0]}
    sums = ["n_spacr", "n_reference", "matched", "matched_iou0.3", "matched_iou0.7", "splits",
            "merges", "identity_splits", "identity_merges", "displaced_matched"]
    row.update({c: int(frame[c].sum()) for c in sums if c in frame})
    row.update(point)
    if "identity_f1" in frame:
        row["identity_f1"] = float(frame["identity_f1"].min())
        row["identity_count_ratio"] = float(frame["identity_count_ratio"].min())
    for key in point:
        values = [d[key] for d in draws]
        row[f"{key}_ci_low"] = float(np.quantile(values, 0.025))
        row[f"{key}_ci_high"] = float(np.quantile(values, 0.975))
    return row


def _write_agreement_report(rows: List[Dict[str, Any]], dst: str,
                            agree: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Site rows plus one pooled row per kind, written to ``mask_agreement.csv``."""
    from .tabular import write_table

    table = pd.DataFrame(rows)
    out = list(rows)
    if len(table):
        for kind in ("nuclei", "cells"):
            part = table[table["kind"] == kind]
            if len(part):
                out.append(_pooled_row(part.reset_index(drop=True), agree))
    write_table(pd.DataFrame(out), os.path.join(dst, "mask_agreement.csv"), canonicalise=False)
    return out


def _saved_label_sites(labels_dir: str) -> List[str]:
    """The sites with saved ``<site>_nuclei.npy`` or ``<site>_cells.npy``."""
    names = {os.path.basename(p).rsplit("_", 1)[0]
             for kind in ("nuclei", "cells")
             for p in glob.glob(os.path.join(labels_dir, f"*_{kind}.npy"))}
    return sorted(names, key=lambda s: (0, int(s), "") if s.isdigit() else (1, 0, s))


def _score_saved_labels(settings: Mapping[str, Any], agree: Mapping[str, Any]) -> Dict[str, Any]:
    """The agreement report on saved labels, with no segmentation."""
    dst = settings.get("seg_dst")
    if not dst:
        raise ValueError("seg_dst is required")
    labels_dir = agree["ops_mask_agreement_labels"]
    if os.path.isdir(os.path.join(labels_dir, "labels")):
        labels_dir = os.path.join(labels_dir, "labels")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "settings.json"), "w", encoding="utf-8") as handle:
        json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in settings.items()},
                  handle, indent=2, sort_keys=True, default=str)
    sites = [str(s) for s in (settings.get("seg_sites") or _saved_label_sites(labels_dir))]
    rows = _agreement_rows(labels_dir, settings["seg_report_reference"], sites, agree)
    return {"dst": dst, "mask_agreement": _write_agreement_report(rows, dst, agree)}


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


def _reference_mask_path(folder: Optional[str], site: str, kind: str) -> Optional[str]:
    """The path of a reference ``*_Site_<site>_<kind>_Objects.tif(f)`` under
    ``folder``, or None."""
    if not folder:
        return None
    found = sorted(glob.glob(os.path.join(folder, "**", f"*_Site_{site}_{kind}_Objects.tif*"),
                             recursive=True))
    return found[0] if found else None


def _reference_mask(folder: Optional[str], site: str, kind: str) -> Optional[np.ndarray]:
    """A reference ``*_Site_<site>_<kind>_Objects.tif(f)`` under ``folder``, or None."""
    import tifffile

    path = _reference_mask_path(folder, site, kind)
    return tifffile.imread(path) if path else None


def run_large_site_segmentation(settings: Mapping[str, Any], *, models: Optional[Tuple[Any, Any]] = None,
                                available_bytes: Optional[int] = None) -> Dict[str, Any]:
    """Segment, measure and report every site of one cropped well.

    Writes, under ``seg_dst``: ``labels/<site>_nuclei.npy`` and
    ``labels/<site>_cells.npy``, ``tables/<site>_cells.parquet`` (one row per
    cell, measured by :func:`measure_chunked`), ``sites.csv`` (per site: objects,
    edge counts, the agreement with ``seg_report_reference`` when given, peak
    memory, seconds, where the whole-site mask step ran:
    :func:`summarise_mask_steps`, and, when ``seg_merge_small_nuclei_frac``
    is above 0, ``merged_small_nuclei`` and ``median_nucleus_area_px``), ``mask_step.json`` (every mask step's
    note, per site and for the survey and the control), ``agreement_well.csv``,
    ``settings.json``,
    ``nucleus_diameter.json`` (the diameter, whether it was fixed or
    estimated, and the survey it came from) and, with
    ``seg_tiling_control_px``, ``tiling_control.json`` (:func:`tiling_control`
    on a centred region of the control site's cell channels, sized by
    :func:`control_layout`). The nucleus
    diameter, when not fixed, and the tiling-control site come from
    ``seg_diameter_sites`` sites spread over the well, of which only those
    with at least ``seg_diameter_min_objects`` nuclei count
    (:func:`survey_sites`), never from one site that may be empty. With
    ``ops_mask_agreement_split_merge`` or
    ``ops_mask_agreement_identity_control`` (item 956), the saved labels are
    also scored against ``seg_report_reference`` into ``mask_agreement.csv``:
    one row per site and kind, with split and merge counts, F1 at IoU 0.3,
    0.5 and 0.7, the identity and displaced controls, and a pooled row per
    kind with site-bootstrap 95 % CIs. With ``ops_mask_agreement_labels``
    nothing is segmented: the labels saved there are scored and only
    ``mask_agreement.csv`` and ``settings.json`` are written.

    :param settings: item 921 settings; ``seg_sites_src`` (the well folder)
        and ``seg_dst`` are required.
    :param models: ``(nucleus_model, cell_model)``; built by
        :func:`preset_models` when ``None``, on the card when ``seg_gpu``.
    :param available_bytes: passed to :func:`plan_site_memory`.
    :returns: ``plan``, ``sites`` (the site table as records), ``dst``,
        ``nucleus_diameter`` (what ``nucleus_diameter.json`` holds) and
        ``mask_steps`` (what ``mask_step.json`` holds), plus ``merge_line``
        (the logged line "R2: merged n small nuclear pieces (x % of
        nuclei)") when ``seg_merge_small_nuclei_frac`` is above 0, plus
        ``mask_agreement`` (the rows of ``mask_agreement.csv``) when that
        report runs; with ``ops_mask_agreement_labels`` only ``dst`` and
        ``mask_agreement``.
    :raises ValueError: when ``seg_sites_src`` or ``seg_dst`` is missing, or
        when no sampled site holds enough nuclei to estimate the diameter, or
        when the agreement report has no ``seg_report_reference``.
    :raises MemoryPlanError: when the RAM cap cannot be met.
    """
    from .resource_log import ResourceSampler
    from .tabular import write_table

    settings = large_site_settings(settings)
    agree = _agreement_settings(settings)
    if agree["ops_mask_agreement_labels"]:
        return _score_saved_labels(settings, agree)
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
    survey_steps: List[Dict[str, Any]] = []
    control_steps: List[Dict[str, Any]] = []
    site_steps: Dict[str, List[Dict[str, Any]]] = {}
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
                iou=float(local["seg_stitch_iou"]), align=True,
                flows=lambda x: _eval_flows(models[0], x, local),
                masks=lambda f, p: _cellpose_masks(models[0], f, p, settings=local,
                                                   record=survey_steps, label="nuclei"))
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
                side, control_tile, _ = control_layout(min(control_px, *stack.shape[:2]),
                                                       max(overlap, 2 * FLOW_MARGIN_PX))
                y0, x0 = (stack.shape[0] - side) // 2, (stack.shape[1] - side) // 2
                body = normalize_site(np.stack([stack[..., channels.index(c)]
                                                for c in local["cell_channels"]], axis=-1))
                region = np.ascontiguousarray(body[y0:y0 + side, x0:x0 + side])
                del body
                control = tiling_control(
                    region, lambda x: _eval_labels(models[1], x, local, None, channel_axis=-1),
                    tile_px=control_tile, overlap_px=overlap,
                    iou=float(local["seg_stitch_iou"]), align=True,
                    flows=lambda x: _eval_flows(models[1], x, local, channel_axis=-1),
                    masks=lambda f, p: _cellpose_masks(models[1], f, p, settings=local,
                                                       record=control_steps, label="cells"))
                control.update({"site": site, "region_px": side, "origin": [y0, x0],
                                "site_nuclei": survey["objects_per_site"][site]})
                with open(os.path.join(dst, "tiling_control.json"), "w", encoding="utf-8") as handle:
                    json.dump(control, handle, indent=2, sort_keys=True)
            steps = site_steps.setdefault(site, [])
            nuclei, cells, paired = segment_cell_painting_site(stack, channels, local, models,
                                                               nucleus_diameter=diameter,
                                                               mask_log=steps)
            nuclei_kept, cells_kept, edge = _edge_step(nuclei, cells, local)
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
        merge = {k: paired.pop(k) for k in ("merged_small_nuclei", "median_nucleus_area_px")
                 if k in paired}
        row.update({f"paired_{k}": int(v) for k, v in paired.items()})
        row.update(merge)
        row.update(summarise_mask_steps(steps))
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
    mask_steps = {"seg_mask_device": settings["seg_mask_device"],
                  "seg_mask_vram_gb": settings["seg_mask_vram_gb"],
                  "sites": site_steps, "survey": survey_steps, "control": control_steps}
    with open(os.path.join(dst, "mask_step.json"), "w", encoding="utf-8") as handle:
        json.dump(mask_steps, handle, indent=2, sort_keys=True)
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
    out = {"plan": plan, "sites": rows, "dst": dst, "nucleus_diameter": survey,
           "mask_steps": mask_steps}
    if "merged_small_nuclei" in frame:
        pieces = int(frame["merged_small_nuclei"].sum())
        total = pieces + int(frame["paired_nuclei"].sum()) if "paired_nuclei" in frame else 0
        out["merge_line"] = (f"R2: merged {pieces} small nuclear pieces "
                             f"({100.0 * pieces / total if total else 0.0:.1f} % of nuclei)")
        _LOG.info(out["merge_line"])
    if agree["report"]:
        out["mask_agreement"] = _write_agreement_report(
            _agreement_rows(os.path.join(dst, "labels"), settings["seg_report_reference"],
                            sites, agree), dst, agree)
    return out
