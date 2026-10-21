"""Build the bundled, synthetic Embeddings example without downloading data."""
from __future__ import annotations

import argparse
import gzip
import io
from pathlib import Path
import sys
import tarfile
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spacr.tabular import write_table


README = """SYNTHETIC Embeddings example
============================

These 24 RGB PNGs are computer-drawn geometric shapes: eight circles, eight
squares and eight triangles. Each image is 64 by 64 pixels. There are no
biological specimens, experimental conditions or human annotations here.
The labels describe only the generated shapes. The example demonstrates
loading crops, making embeddings and inspecting a label scorecard; its
scores are not a scientific benchmark or evidence of biological accuracy.

Try it in spaCR
--------------
The embedding dependencies must be installed. For a Python installation,
run `python -m pip install "spacr[embeddings]"` in the spaCR environment if
the app asks for torch and timm. This example does not bundle Python packages.

1. Enable Show alpha features, then open Embeddings.
2. Click Load synthetic demo. This opens these crops and their aligned
   labels, selects the Foundation model "DINOv2 (Cell-DINO when
   installed)" with Project to three (one pass) and a batch size of 8, and
   selects the CPU for embedding. The line under the controls says whether
   official Cell-DINO or the open DINOv2 ViT-S/14 will run.
3. Press Embed. The preview should contain 24 rows with finite values.
   The label scorecard reports retrieval performance for the drawn shapes.
4. To try the classifier workflow, choose Use for, then Classifier, then
   Run. Treat the result as a functional demonstration on this tiny set.

The images and labels are bundled with spaCR and require no data download.
Pretrained DINOv2 weights (about 90 MB) are separate: the normal model
loader may download them on first use, then reuse its cache. No Cell-DINO
weights are included or required; Meta grants them by request. The example
does not train a backbone. For real cells, Load cell example opens the
example plate's annotated single-cell crops the same way.

Open the files manually
----------------------
In Embeddings, select the crop-folder source, choose the crops/ directory,
clear any selection filters, set the crop limit to at least 24 and press
Load crops. Select the Foundation model "DINOv2 (Cell-DINO when installed)"
and Project to three (one pass). Open Labels… and select
labels.csv, then press Embed. The built-in demo button explicitly uses CPU;
manual loading follows the application's normal device selection.

labels.csv has crop_filename and label columns. Rows match the alphabetical
order of the zero-padded PNG filenames exactly, with one row per crop.
Keep that row order: the current Labels action reads labels by crop order,
not by joining the crop_filename column. Do not filter or reorder one input
without making the corresponding change to the other.

Rebuild
-------
From the source checkout, run:
    python tools/build_embeddings_example_dataset.py
The generator uses fixed geometric formulas and no network or random state.
It writes spacr/resources/data/embeddings_synthetic_example.tar.gz.
"""


def _crops():
    """Yield ordered filenames, geometric labels and deterministic RGB crops."""
    y, x = np.mgrid[:64, :64]
    for shape_index, label in enumerate(("circle", "square", "triangle")):
        for variant in range(8):
            cx = 30 + variant % 3
            cy = 30 + (variant // 3) % 3
            radius = 14 + variant % 4
            dx, dy = x - cx, y - cy
            if label == "circle":
                mask = dx * dx + dy * dy <= radius * radius
            elif label == "square":
                mask = np.maximum(np.abs(dx), np.abs(dy)) <= radius
            else:
                mask = (dy >= -radius) & (dy <= radius)
                mask &= np.abs(dx) * 2 <= dy + radius
            base = 18 + (x * 3 + y * 5 + variant) % 9
            gray = np.where(mask, 200 + variant * 3, base).astype(np.uint8)
            rgb = np.repeat(gray[..., None], 3, axis=2)
            filename = f"crop_{shape_index * 8 + variant:03d}.png"
            yield filename, label, rgb


def build(destination: Path) -> Path:
    """Write a deterministic archive containing crops, aligned labels and help."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    members = {}
    rows = []
    for filename, label, pixels in _crops():
        image = io.BytesIO()
        Image.fromarray(pixels).save(image, format="PNG", compress_level=9)
        members[f"crops/{filename}"] = image.getvalue()
        rows.append({"crop_filename": filename, "label": label})
    with tempfile.TemporaryDirectory(prefix="spacr-embeddings-example-") as temp:
        labels = Path(temp) / "labels.csv"
        write_table(pd.DataFrame(rows), labels, index=False, canonicalise=False,
                    lineterminator="\n")
        members["labels.csv"] = labels.read_bytes()
    members["README.txt"] = README.encode("utf-8")
    with destination.open("wb") as output:
        with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT) as tar:
                for name, payload in sorted(members.items()):
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    info.mode = 0o644
                    info.mtime = 0
                    tar.addfile(info, io.BytesIO(payload))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", type=Path, default=(
        REPO / "spacr/resources/data/embeddings_synthetic_example.tar.gz"))
    print(build(parser.parse_args().destination))
