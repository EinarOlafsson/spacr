#!/usr/bin/env python3
"""Prepare a reproducible 3x3 tile acquisition from one real spaCR field.

The source pixels are never modified.  Nine overlapping, four-channel NPY
tiles are exported solely to demonstrate the Align & Stitch workflow with a
known ground-truth mosaic.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    "/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/"
    "spacr/tutorials/orig/test/stack/test_E11_2_1.npy"
)
DERIVED = ROOT / "derived" / "align_stitch"
TILES = DERIVED / "tiles"
GENERATED = ROOT / "generated" / "align_stitch"
TILE_SIZE = 800
STEP = 600


def main() -> int:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if TILES.exists() or GENERATED.exists():
        raise FileExistsError(
            "align tutorial staging already exists; remove only "
            f"{DERIVED} and {GENERATED} before intentionally regenerating"
        )

    source = np.load(SOURCE, mmap_mode="r")
    if source.shape != (2000, 2000, 4) or source.dtype != np.uint16:
        raise RuntimeError(
            f"unexpected source contract: {source.shape} {source.dtype}"
        )

    TILES.mkdir(parents=True)
    records = []
    field = 1
    for row, y0 in enumerate(range(0, STEP * 3, STEP)):
        for column, x0 in enumerate(range(0, STEP * 3, STEP)):
            path = TILES / f"tutorial_A01_{field:03d}.npy"
            tile = np.asarray(
                source[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE, :]
            )
            if tile.shape != (TILE_SIZE, TILE_SIZE, 4):
                raise RuntimeError(f"incomplete tile {field}: {tile.shape}")
            np.save(path, tile)
            records.append({
                "field": field,
                "row": row,
                "column": column,
                "source_yx": [y0, x0],
                "path": str(path),
                "shape": list(tile.shape),
            })
            field += 1

    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE),
        "source_shape": list(source.shape),
        "source_dtype": str(source.dtype),
        "staging": str(TILES),
        "destination": str(GENERATED),
        "grid": [3, 3],
        "tile_size": TILE_SIZE,
        "step": STEP,
        "nominal_overlap": 1.0 - STEP / TILE_SIZE,
        "reference_channel": 1,
        "disclosure": (
            "Nine overlapping tiles exported from one real four-channel "
            "microscopy field solely to emulate a 3 by 3 tiled acquisition."
        ),
        "tiles": records,
    }
    (DERIVED / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(DERIVED / "source_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
