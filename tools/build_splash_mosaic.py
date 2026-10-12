#!/usr/bin/env python3
"""Build the startup splash's photomosaic resource.

The spaCR logo photomosaic in full colour (``color_final_v1.png``) was made by
``mosaic.py color`` in the genAI talk material: an 80 x 80 grid of
single-cell microscopy tiles -- a yellow-green cell outline, a blue nucleus
and red parasites on black. It exists as a 3280 px square (41 px tiles) and
as a 17920 px square (224 px tiles, ``color_final_v1_highres_17920px.png``)
of the same grid; the tile size is read from the picture. This tool reads
the picture and writes the two files the splash needs:

``spacr/resources/splash/photomosaic.jpg``
    The mosaic with each tile resampled on its own to ``--tile-px`` pixels, so
    tile boundaries land on exact pixel rows and columns and the splash can
    cut the picture back into its real tiles. 27 px tiles (the default since
    item 695) give a 2160 px square, built from the 17920 px picture: the
    logo is drawn at half the screen height, 1080 device pixels on a 4K
    screen, so 2160 px stays sharp up to a 4320 px tall screen. Item 694
    shipped 20 px tiles (1600 px) from the 3280 px picture.

``spacr/resources/splash/photomosaic.json``
    The grid, the tile size and which tiles hold the picture. The background
    tiles are not empty -- they carry faint tile texture -- so a tile counts
    as occupied when its mean brightness is above ``--background-factor``
    times the background level, the median of the darkest 30 % of tiles.
    Measured on color_final_v1: background 2.0, cell interior 6-8, logo
    15-95; the occupied count is flat (4,003-4,020) for any cut from 4 to 5.
    The 17920 px picture gives 3,999 tiles at the default cut (9 tiles
    differ from the 3280 px picture's 4,004, all at the cut).

Run::

    python tools/build_splash_mosaic.py \
        --mosaic /path/to/color_final_v1_highres_17920px.png

The source picture is only read. Copy it off the NAS first; the 17920 px
picture needs about 1 GB of memory to decode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GRID = 80
DEFAULT_MOSAIC = Path(
    "/nas_mnt/data/sync/presentations/genAI/logo/color_final/"
    "color_final_v1_highres_17920px.png")
OUT_DIR = ROOT / "spacr" / "resources" / "splash"


Image.MAX_IMAGE_PIXELS = None


def open_mosaic(mosaic_path: Path, grid: int = GRID):
    """The photomosaic in RGB and its source tile size.

    :param mosaic_path: the full-size photomosaic, a square of ``grid`` tiles.
    :param grid: cells per side.
    :returns: ``(picture, tile)``.
    """
    source = Image.open(mosaic_path).convert("RGB")
    tile = source.size[0] // grid
    if tile < 1 or source.size != (grid * tile, grid * tile):
        raise ValueError(f"{mosaic_path} is {source.size}, expected a "
                         f"square of {grid} x {grid} whole tiles")
    return source, tile


def occupied_cells(source: Image.Image, tile: int, factor: float = 2.5,
                   grid: int = GRID) -> np.ndarray:
    """Which cells of the mosaic hold the picture rather than background.

    :param source: the full-size photomosaic, RGB.
    :param tile: source pixels per cell.
    :param factor: how many times the background level a tile's mean
        brightness must reach.
    :param grid: cells per side.
    :returns: a ``grid x grid`` boolean array, row-major.
    """
    means = np.asarray(source.reduce(tile), np.float32)[:grid, :grid]
    mean = means.mean(axis=2)
    darkest = np.sort(mean.ravel())[: int(0.3 * mean.size)]
    return mean > factor * float(np.median(darkest))


def retile(source: Image.Image, tile: int, out_tile: int,
           grid: int = GRID) -> Image.Image:
    """The mosaic with every tile resampled on its own to ``out_tile`` px.

    :param source: the full-size photomosaic, RGB.
    :param tile: source pixels per cell.
    :param out_tile: pixels per tile in the result.
    :returns: an RGB picture ``grid * out_tile`` pixels square.
    """
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
    parser.add_argument("--tile-px", type=int, default=27)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    source, tile = open_mosaic(args.mosaic)
    cells = occupied_cells(source, tile, args.background_factor)
    picture = retile(source, tile, args.tile_px)
    args.out.mkdir(parents=True, exist_ok=True)
    image_path = args.out / "photomosaic.jpg"
    picture.save(image_path, "JPEG", quality=args.quality, optimize=True,
                 progressive=False, subsampling=2)
    meta = {
        "generator": "tools/build_splash_mosaic.py",
        "source": f"{args.mosaic.name}, {GRID} x {GRID} tiles of {tile} px",
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
