#!/usr/bin/env python3
"""Prepare real fields and the previous lesson's checkpoint for Cellpose Masks."""
from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "derived" / "train_cellpose"
DESTINATION = ROOT / "derived" / "cellpose_masks"
CHECKPOINT = (
    TRAIN_ROOT
    / "models"
    / "cellpose_model"
    / "models"
    / "new_model_cpsam_e1_X256_Y256.CP_model"
)


def main() -> int:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Run the Train Cellpose tutorial first; checkpoint missing: {CHECKPOINT}"
        )
    DESTINATION.mkdir(parents=True, exist_ok=True)
    source_images = sorted((TRAIN_ROOT / "train" / "images").glob("*.tif"))
    if not source_images:
        raise RuntimeError("Train Cellpose tutorial images are missing")
    for source in source_images:
        shutil.copy2(source, DESTINATION / source.name)

    # Clear only masks created by this reproducible tutorial, leaving the
    # source fields and checkpoint untouched.
    mask_dir = DESTINATION / "masks"
    if mask_dir.exists():
        for mask in mask_dir.glob("cell_pair_*.tif"):
            mask.unlink()

    settings = {
        "src": str(DESTINATION),
        "model_name": "cpsam",
        "custom_model": str(CHECKPOINT),
        "channels": [0, 0],
        "background": 100,
        "remove_background": False,
        "Signal_to_noise": 10,
        "CP_prob": 0.0,
        "diameter": 30,
        "batch_size": 2,
        "flow_threshold": 0.4,
        "save": True,
        "verbose": True,
        "normalize": True,
        "percentiles": [2.0, 99.5],
        "invert": False,
        "resize": False,
        "target_height": None,
        "target_width": None,
        "rescale": False,
        "resample": False,
        "grayscale": True,
        "fill_in": True,
    }
    (DESTINATION / "tutorial_settings.json").write_text(
        json.dumps(settings, indent=2) + "\n"
    )
    (DESTINATION / "source_manifest.json").write_text(
        json.dumps(
            {
                "source_images": [str(path) for path in source_images],
                "checkpoint": str(CHECKPOINT),
                "input_count": len(source_images),
                "image_origin": "real ER/cell-channel crops from the spaCR tutorial experiment",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Prepared {len(source_images)} Cellpose fields in {DESTINATION}")
    print(f"Custom checkpoint: {CHECKPOINT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
