#!/usr/bin/env python3
"""Build a small, reproducible Train Cellpose tutorial set from real fields.

The source merged arrays contain four acquisition channels followed by nucleus,
pathogen, and cell labels.  We pair the ER/cell channel (1) with the cell-label
plane (6), retaining only crops with several complete cells.  These pairs are
appropriate for demonstrating the training workflow; a production model should
be trained from independently reviewed masks and evaluated on held-out fields.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile


SOURCE = Path(
    "/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/"
    "spacr/tutorials/merged"
)
ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "derived" / "train_cellpose"

# Fixed fields and crop origins keep every regeneration visually identical.
PAIRS = (
    ("plate1_B01_1_1.npy", 120, 180),
    ("plate1_B01_2_1.npy", 520, 240),
    ("plate1_B01_3_1.npy", 940, 520),
    ("plate1_B02_1_1.npy", 180, 980),
    ("plate1_B02_2_1.npy", 700, 1080),
    ("plate1_B02_3_1.npy", 1120, 900),
)
CROP_SIZE = 512


def main() -> int:
    image_dir = DESTINATION / "train" / "images"
    mask_dir = DESTINATION / "train" / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for index, (name, row, column) in enumerate(PAIRS, start=1):
        source = SOURCE / name
        if not source.exists():
            raise FileNotFoundError(source)
        merged = np.load(source, mmap_mode="r")
        image = np.asarray(
            merged[row : row + CROP_SIZE, column : column + CROP_SIZE, 1]
        )
        mask = np.asarray(
            merged[row : row + CROP_SIZE, column : column + CROP_SIZE, 6]
        )
        if image.shape != (CROP_SIZE, CROP_SIZE):
            raise RuntimeError(f"incomplete crop from {source}: {image.shape}")
        filename = f"cell_pair_{index:02d}.tif"
        tifffile.imwrite(image_dir / filename, image.astype(np.uint16))
        tifffile.imwrite(mask_dir / filename, mask.astype(np.uint16))
        labels = np.unique(mask)
        records.append(
            {
                "file": filename,
                "source": name,
                "origin_yx": [row, column],
                "objects": int(np.count_nonzero(labels)),
                "image_range": [int(image.min()), int(image.max())],
            }
        )

    settings = {
        "src": str(DESTINATION),
        "model_name": "spacr_tutorial_cells",
        "model_type": "cpsam",
        "Signal_to_noise": 10,
        "background": 200,
        "remove_background": False,
        "learning_rate": 0.01,
        "weight_decay": 0.00001,
        "batch_size": 2,
        "n_epochs": 1,
        "from_scratch": False,
        "diameter": 30,
        "resize": True,
        "width_height": [512, 512],
        "target_size": 256,
        "augment": False,
        "verbose": True,
    }
    (DESTINATION / "tutorial_settings.json").write_text(
        json.dumps(settings, indent=2) + "\n"
    )
    (DESTINATION / "source_manifest.json").write_text(
        json.dumps(
            {
                "source_root": str(SOURCE),
                "image_channel": 1,
                "mask_plane": 6,
                "crop_size": CROP_SIZE,
                "pairs": records,
                "purpose": "workflow demonstration; review labels before production training",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Prepared {len(records)} paired fields in {DESTINATION}")
    print(f"Settings: {DESTINATION / 'tutorial_settings.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
