#!/usr/bin/env python3
"""Derive a compact external-mask import from real tutorial fields."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    "/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/"
    "spacr/tutorials/orig/test"
)
STAGING = ROOT / "derived" / "external_masks" / "inputs"
OUTPUT = ROOT / "generated" / "external_masks" / "project"
FIELDS = ("test_E11_2_1", "test_M07_3_1", "test_G04_2_1")
OBJECTS = ("cell", "nucleus", "pathogen", "organelle")
CROP = (slice(616, 1384), slice(616, 1384))


def _write(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        np.asarray(array),
        photometric="minisblack",
    )


def main() -> int:
    for target in (STAGING, OUTPUT):
        if target.exists():
            shutil.rmtree(target)
    records = []
    for index, stem in enumerate(FIELDS, start=1):
        stack_path = SOURCE / "stack" / f"{stem}.npy"
        if not stack_path.is_file():
            raise FileNotFoundError(stack_path)
        stack = np.load(stack_path, mmap_mode="r")
        crop = np.asarray(stack[CROP[0], CROP[1], :])
        if crop.ndim != 3 or crop.shape[2] != 4:
            raise RuntimeError(f"unexpected stack shape {stack.shape}")
        field = f"fov{index:03d}"
        for channel in range(crop.shape[2]):
            _write(
                STAGING / "images" / f"{field}_C{channel + 1}.tif",
                crop[..., channel],
            )
        object_counts = {}
        for object_type in OBJECTS:
            mask_path = (
                SOURCE / "masks" / f"{object_type}_mask_stack" /
                f"{stem}.npy"
            )
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            labels = np.asarray(np.load(mask_path, mmap_mode="r")[CROP])
            _write(
                STAGING / f"{object_type}_masks" /
                f"{field}_{object_type}_mask.tif",
                labels,
            )
            object_counts[object_type] = int(
                np.unique(labels[labels > 0]).size)
        records.append({
            "derived_field": field,
            "source_stack": str(stack_path),
            "source_crop_yx": [616, 1384, 616, 1384],
            "shape_yxc": list(crop.shape),
            "object_counts": object_counts,
        })

    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE),
        "staging": str(STAGING),
        "destination": str(OUTPUT),
        "disclosure": (
            "Real 768 by 768 center crops from three transferred spaCR test "
            "fields. Intensity planes and existing label planes were exported "
            "to TIFF solely to emulate a third-party external-mask handoff."
        ),
        "fields": records,
    }
    (STAGING.parent / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
