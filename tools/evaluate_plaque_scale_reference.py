"""Run the real figure reader on a pinned image with independently supplied crop boxes.

The reference JSON records the source, licence, image hashes, caption scale and
crop boxes. Download its image first; this tool performs no downloads itself.
It calls the installed OCR reader and production scale resolver. It measures
scale resolution conditional on the supplied boxes, not detector accuracy or
plaque segmentation. No input, checkpoint or user configuration is modified.

Run under tools/run_capped.sh with CUDA_VISIBLE_DEVICES='' and private HOME,
XDG_CONFIG_HOME and YOLO_CONFIG_DIR. SPACR_BACKENDS_DIR may point to an existing
isolated papers reader. A reported crop-width discrepancy is evidence, not a
reason to adjust the detected ruler to match the reference.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from spacr import plaque_papers as papers


def main():
    """Validate source identity, read the real figure, and write calibration evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve() in {args.image.resolve(), args.reference.resolve()}:
        parser.error("output must differ from image and reference")
    reference = json.loads(args.reference.read_text())
    image_bytes = args.image.read_bytes()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    if image_hash not in reference["accepted_file_sha256"]:
        parser.error("image hash does not match the reference")
    with Image.open(args.image) as loaded:
        image = np.asarray(loaded.convert("RGB"))
    if list(image.shape) != reference["shape"]:
        parser.error("image shape does not match the reference")
    regions = [papers.Region(*box) for box in reference["crop_boxes_xyxy"]]
    started = time.monotonic()
    words = papers.read_words(image)
    scales = papers._scales_for_regions(image, regions, words,
                                        caption=reference["scale_caption"])
    rows = []
    for region, scale in zip(regions, scales):
        width_mm = region.width / scale.px_per_mm if scale.px_per_mm else None
        rows.append({"region": asdict(region), "scale": asdict(scale),
                     "detected_unlabelled_bar": papers._unlabelled_bar(image, region),
                     "inferred_crop_width_mm": width_mm,
                     "relative_crop_width_difference": (
                         width_mm / reference["published_crop_width_mm"] - 1
                         if width_mm is not None else None)})
    result = {"reference": reference, "image_sha256": image_hash,
              "reader_environment": papers.reader_environment(),
              "source_sha256": hashlib.sha256(Path(papers.__file__).read_bytes()).hexdigest(),
              "seconds": time.monotonic() - started,
              "words": [asdict(word) for word in words], "regions": rows,
              "calibrated_regions": sum(scale.px_per_mm is not None for scale in scales),
              "limitations": ["Crop boxes are supplied, not predicted by a detector",
                              "One published figure; no corpus-wide accuracy claim",
                              "No plaque segmentation or biological comparison"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"calibrated_regions": result["calibrated_regions"],
                      "total_regions": len(regions), "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
