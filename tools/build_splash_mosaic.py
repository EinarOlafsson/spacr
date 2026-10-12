#!/usr/bin/env python3
"""Build the startup splash's photomosaic resource.

The spaCR logo photomosaic in full colour (``color_final_v1.png``) was made by
``mosaic.py color`` in the genAI talk material: an 80 x 80 grid of 41 px
single-cell microscopy tiles, a 3280 px square -- a yellow-green cell outline,
a blue nucleus and red parasites on black. This tool reads that picture and
writes the two files the splash needs:

``spacr/resources/splash/photomosaic.jpg``
    The mosaic with each tile resampled on its own to ``--tile-px`` pixels, so
    tile boundaries land on exact pixel rows and columns and the splash can
    cut the picture back into its real tiles. 20 px tiles give a 1600 px
    square: an 800 px splash on a 2x screen at full resolution.

``spacr/resources/splash/photomosaic.json``
    The grid, the tile size and which tiles hold the picture. The background
    tiles are not empty -- they carry faint tile texture -- so a tile counts
    as occupied when its mean brightness is above ``--background-factor``
    times the background level, the median of the darkest 30 % of tiles.
    Measured on color_final_v1: background 2.0, cell interior 6-8, logo
    15-95; the occupied count is flat (4,003-4,020) for any cut from 4 to 5.

Run::

    python tools/build_splash_mosaic.py \
        --mosaic /path/to/spacr_logo_photomosaic.png

The source picture is only read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GRID = 80
SOURCE_TILE_PX = 41
DEFAULT_MOSAIC = Path(
    "/nas_mnt/data/sync/presentations/genAI/logo/color_final/color_final_v1.png")
OUT_DIR = ROOT / "spacr" / "resources" / "splash"


def occupied_cells(mosaic_path: Path, factor: float = 2.5,
                   grid: int = GRID, tile: int = SOURCE_TILE_PX) -> np.ndarray:
    """Which cells of the mosaic hold the picture rather than background.

    :param mosaic_path: the full-size photomosaic.
    :param factor: how many times the background level a tile's mean
        brightness must reach.
    :param grid: cells per side.
    :param tile: source pixels per cell.
    :returns: a ``grid x grid`` boolean array, row-major.
    """
    pixels = np.asarray(Image.open(mosaic_path).convert("RGB"), np.float32)
    mean = pixels.reshape(grid, tile, grid, tile, 3).mean(axis=(1, 3, 4))
    darkest = np.sort(mean.ravel())[: int(0.3 * mean.size)]
    return mean > factor * float(np.median(darkest))


def retile(mosaic_path: Path, out_tile: int, grid: int = GRID,
           tile: int = SOURCE_TILE_PX) -> Image.Image:
    """The mosaic with every tile resampled on its own to ``out_tile`` px.

    :param mosaic_path: the full-size photomosaic.
    :param out_tile: pixels per tile in the result.
    :returns: an RGB picture ``grid * out_tile`` pixels square.
    """
    source = Image.open(mosaic_path).convert("RGB")
    if source.size != (grid * tile, grid * tile):
        raise ValueError(f"{mosaic_path} is {source.size}, expected a "
                         f"{grid * tile} px square of {grid} x {grid} tiles")
    out = Image.new("RGB", (grid * out_tile, grid * out_tile))
    for row in range(grid):
        for col in range(grid):
            box = (col * tile, row * tile, (col + 1) * tile, (row + 1) * tile)
            cell = source.crop(box).resize((out_tile, out_tile), Image.LANCZOS)
            out.paste(cell, (col * out_tile, row * out_tile))
    return out


def bitmask_hex(cells: np.ndarray) -> str:
    """Row-major occupancy as a hex string, most significant bit first."""
    bits = np.packbits(cells.astype(np.uint8).ravel())
    return bits.tobytes().hex()


def main(argv=None) -> int:
    """Write the splash mosaic picture and its tile map."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mosaic", type=Path, default=DEFAULT_MOSAIC)
    parser.add_argument("--background-factor", type=float, default=2.5)
    parser.add_argument("--tile-px", type=int, default=20)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    cells = occupied_cells(args.mosaic, args.background_factor)
    picture = retile(args.mosaic, args.tile_px)
    args.out.mkdir(parents=True, exist_ok=True)
    image_path = args.out / "photomosaic.jpg"
    picture.save(image_path, "JPEG", quality=args.quality, optimize=True,
                 progressive=False, subsampling=2)
    meta = {
        "generator": "tools/build_splash_mosaic.py",
        "source": f"{args.mosaic.name}, 80 x 80 tiles of 41 px",
        "image": image_path.name,
        "grid": GRID,
        "tile_px": args.tile_px,
        "occupied": bitmask_hex(cells),
    }
    (args.out / "photomosaic.json").write_text(
        json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"{image_path}: {image_path.stat().st_size} bytes, "
          f"{int(cells.sum())} of {GRID * GRID} tiles occupied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
