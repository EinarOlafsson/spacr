#!/usr/bin/env python3
"""Create a small, deterministic synthetic plaque-assay dataset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "synthetic" / "plaque"


def field(seed: int, plaques: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    size = 768
    base = Image.new("L", (size, size), 213)
    texture = rng.normal(0, 4.5, (size, size))
    base_array = np.clip(np.asarray(base, dtype=float) + texture, 0, 255)
    image = Image.fromarray(base_array.astype(np.uint8), "L")
    shade = Image.new("L", image.size, 0)
    shade_draw = ImageDraw.Draw(shade)
    mask = Image.new("I", image.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    for label, (cx, cy, rx, ry) in enumerate(plaques, start=1):
        box = (cx - rx, cy - ry, cx + rx, cy + ry)
        shade_draw.ellipse(box, fill=88 + (label % 3) * 8)
        mask_draw.ellipse(box, fill=label)
    shade = shade.filter(ImageFilter.GaussianBlur(11))
    image = Image.fromarray(
        np.clip(np.asarray(image, dtype=np.int16) - np.asarray(shade, dtype=np.int16), 0, 255).astype(np.uint8),
        "L",
    )
    return np.asarray(image), np.asarray(mask, dtype=np.uint16)


def main() -> int:
    masks = OUTPUT / "masks"
    masks.mkdir(parents=True, exist_ok=True)
    wells = {
        "plate1_A01_control_1.tif": [(180, 180, 83, 70), (540, 220, 105, 92), (310, 545, 116, 100), (625, 575, 72, 62)],
        "plate1_A02_control_2.tif": [(210, 220, 110, 90), (575, 205, 92, 80), (390, 550, 136, 104)],
        "plate1_B01_treatment_1.tif": [(245, 265, 74, 62), (575, 520, 66, 55)],
        "plate1_B02_treatment_2.tif": [(390, 390, 86, 72)],
    }
    records = []
    for seed, (name, plaques) in enumerate(wells.items(), start=2401):
        image, labels = field(seed, plaques)
        Image.fromarray(image).save(OUTPUT / name)
        Image.fromarray(labels).save(masks / name)
        records.append({
            "file": name,
            "condition": "control" if "control" in name else "treatment",
            "expected_objects": len(plaques),
            "label_areas": [int((labels == label).sum()) for label in range(1, len(plaques) + 1)],
        })
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "schema": 1,
        "kind": "synthetic plaque-assay fields with precomputed labeled masks",
        "size": [768, 768],
        "records": records,
    }, indent=2) + "\n")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
