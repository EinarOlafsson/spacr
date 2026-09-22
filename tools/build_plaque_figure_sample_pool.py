#!/usr/bin/env python
"""Write the pool Make Masks and Plaque Assay draw their plaque FIGURES sample from.

The maintainer, 2026-09-21: the figures sample must come from the well
detector's TRAINING set, every figure must have plaque wells on it, and only
high-resolution figures are to be picked. The dataset's split.csv says which
figures are in train and how many wells each has, but not how big they are,
so this downloads the training images once, measures them, and writes

    spacr/resources/data/plaque_figures_sample_pool.csv

with every training figure that has at least one labelled well and whose
LONGER side is at least ``--min-long-side`` pixels (1,500 by default: of the
506 training figures with wells, the median long side is 828 px and the top
quarter starts near 2,000; 202 pass). The sample is then a seeded random draw
from that pool, as every other sample is.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = "einarolafsson/toxoplasma-plaque-well-detector-dataset"
OUT = (Path(__file__).resolve().parent.parent / "spacr" / "resources" / "data"
       / "plaque_figures_sample_pool.csv")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--min-long-side", type=int, default=1500)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args(argv)
    from huggingface_hub import snapshot_download
    from PIL import Image

    root = Path(snapshot_download(
        REPO, repo_type="dataset", cache_dir=args.cache_dir, max_workers=8,
        allow_patterns=["images/train/*", "labels/train/*"]))
    rows = []
    for image in sorted((root / "images" / "train").iterdir()):
        label = root / "labels" / "train" / f"{image.stem}.txt"
        wells = (len([line for line in label.read_text().splitlines()
                      if line.strip()]) if label.is_file() else 0)
        with Image.open(image) as opened:
            width, height = opened.size
        if wells and max(width, height) >= args.min_long_side:
            rows.append((image.name, width, height, wells))
    with open(OUT, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "width", "height", "n_wells"])
        writer.writerows(rows)
    print(f"{len(rows)} figures -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
