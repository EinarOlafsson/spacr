"""Synthetic tile builders for the spacr.spacrops stitching tests.

Everything here is deterministic (explicit seeds).  The point of these
helpers is that every tile is a *crop of one big canvas at a known
offset*, so the stitcher's recovered translation can be compared against
a ground truth rather than merely "an array came back".
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import tifffile


def blob_canvas(H: int = 900, W: int = 900, n: int = 420, seed: int = 0,
                rad: int = 14) -> np.ndarray:
    """A uint16 'microscopy-like' canvas: Gaussian blobs on a noisy background.

    Blobs are large enough (sigma 4-7 px) to survive a 2x downsample, which
    is what the stitcher's feature pass runs on.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[0:2 * rad + 1, 0:2 * rad + 1]
    for _ in range(n):
        y = int(rng.integers(rad + 1, H - rad - 1))
        x = int(rng.integers(rad + 1, W - rad - 1))
        a = float(rng.uniform(1500, 6000))
        sg = float(rng.uniform(4.0, 7.0))
        img[y - rad:y + rad + 1, x - rad:x + rad + 1] += a * np.exp(
            -(((yy - rad) ** 2 + (xx - rad) ** 2) / (2 * sg * sg)))
    img += rng.normal(120, 10, (H, W)).astype(np.float32)
    return np.clip(img, 0, 65535).astype(np.uint16)


def crop(canvas: np.ndarray, y0: int, x0: int, tile: int) -> np.ndarray:
    """Tile-sized crop of ``canvas`` whose top-left is (y0, x0)."""
    return canvas[y0:y0 + tile, x0:x0 + tile].copy()


def tile_name(well: str = "A1", site: int = 1, chan: int = 1,
              mag: str = "10X") -> str:
    """Yokogawa-ish file name the stitcher's default meta regex understands."""
    return f"{mag}_c{chan}_{well}_r01f{site:02d}_Site-{site}.tif"


def write_plane(path: str, arr: np.ndarray) -> str:
    tifffile.imwrite(path, arr)
    return path


def write_cyx(path: str, planes: List[np.ndarray]) -> str:
    """Write a (C, Y, X) TIFF with explicit axes metadata."""
    tifffile.imwrite(path, np.stack(planes, axis=0), metadata={"axes": "CYX"})
    return path


def channel_variant(base: np.ndarray, c: int) -> np.ndarray:
    """Deterministic per-channel variant so tests can tell channels apart.

    Channel 0 is the raw canvas; channel c>0 is the canvas mirrored and
    scaled, which changes both the values and the spatial layout.
    """
    if c == 0:
        return base.copy()
    out = (base[::-1, ::-1].astype(np.float32) * (0.25 * (c + 1))) + 100 * c
    return np.clip(out, 0, 65535).astype(base.dtype)


def row_of_tiles(dirpath: str, *, canvas: np.ndarray, n: int = 3,
                 tile: int = 384, step: int = 150, y0: int = 100, x0: int = 100,
                 well: str = "A1", channels: int = 1,
                 first_site: int = 1) -> Tuple[List[str], Dict[str, int]]:
    """Write ``n`` horizontally overlapping tiles cropped from ``canvas``.

    Tile *i* is ``canvas[y0:y0+tile, x0+i*step : x0+i*step+tile]``, so the
    ground-truth B->A translation for the (i, i+1) pair is ``dx=+step, dy=0``.

    :returns: (paths, ground-truth dict with keys ``step``, ``tile``, ``y0``, ``x0``)
    """
    os.makedirs(dirpath, exist_ok=True)
    paths = []
    for i in range(n):
        sub = crop(canvas, y0, x0 + i * step, tile)
        p = os.path.join(dirpath, tile_name(well=well, site=first_site + i))
        if channels <= 1:
            write_plane(p, sub)
        else:
            write_cyx(p, [channel_variant(sub, c) for c in range(channels)])
        paths.append(p)
    return paths, {"step": step, "tile": tile, "y0": y0, "x0": x0}
