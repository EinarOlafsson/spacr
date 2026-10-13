#!/usr/bin/env python3
"""Extract real ER fields and cell labels into an isolated Make Masks project."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


DEFAULT_DATASET = Path(
    "/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/"
    "spacr/tutorials"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "synthetic" / "make_masks_real"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fields", type=int, default=3)
    args = parser.parse_args()

    sources = sorted((args.dataset / "merged").glob("*.npy"))[: args.fields]
    if len(sources) < args.fields:
        raise FileNotFoundError("not enough merged microscopy fields")

    output = args.output.resolve()
    masks = output / "masks"
    masks.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, source in enumerate(sources, start=1):
        merged = np.load(source)
        if merged.ndim != 3 or merged.shape[-1] < 5:
            raise ValueError(f"expected image and mask channels in {source}")
        name = f"field_{index:02d}_ER.tif"
        image = np.asarray(merged[..., 1], dtype=np.uint16)
        mask = np.asarray(merged[..., 4], dtype=np.uint16)
        tifffile.imwrite(output / name, image)
        # Leave the third field deliberately blank so the tutorial can show
        # how a new label mask is started from an image alone.
        if index < args.fields:
            tifffile.imwrite(masks / name, mask)
        manifest.append({
            "tutorial_file": name,
            "source": str(source.resolve()),
            "image_channel": 1,
            "image_label": "ER",
            "mask_channel": 4,
            "mask_label": "cell",
            "existing_mask": index < args.fields,
            "shape": list(image.shape),
            "object_count": int(mask.max()),
        })

    (output / "manifest.json").write_text(json.dumps({
        "schema": 1,
        "source_dataset": str(args.dataset.resolve()),
        "purpose": "isolated real-data copy for the Make Masks tutorial",
        "fields": manifest,
    }, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
