"""Stage Make Masks' FIB-SEM test data: one volume with mitochondria labels.

Source: EMPIAR-10982, "Seven benchmark datasets of instance segmentation of
mitochondria: 6 diverse volume EM + 1 TEM (100 images) datasets" (Narayan K,
Conrad RW; NCI/FNLCR), folder ``data/mito_benchmarks/hela_cell``: a
256 x 256 x 256 8-bit FIB-SEM volume of a HeLa cell and its 3-D instance
label map of every mitochondrion. EMPIAR's experiment type for the entry is
FIB-SEM and EMPIAR releases its entries under CC0 1.0.

What is cut, by a fixed rule: the central 128 sections (Z 64 to 191) at the
full 256 x 256 field. The image is written unchanged as 8-bit; the labels
are renumbered 1..n in order of first appearance (``np.unique`` order) and
written as uint16. A mitochondrion the crop cuts keeps one label.

Voxel size: EMPIAR records a 150 angstrom (15 nm) pixel for this image set
and no separate section thickness, so 15 nm is written on all three axes.
That Z value is an assumption; the card says so, and a measurement that
depends on Z should check it.

Both files are ImageJ hyperstacks with axes ZYX, unit nm and 15 nm spacing,
which Fiji and spaCR's 3-D editor read alike. The image opens UNSEGMENTED from
the volume-only menu entry; the labels sit in ``ground_truth_masks/`` and the
"with masks" entry copies them into ``corrected_masks/`` for correction.

Output (``dest``): ``make_masks_fibsem/`` (the unpacked set), ``README.md``
(the dataset card) and ``spacr-example-fibsem.tar`` (manifest last, so a
transfer that died part-way reads as absent).

Usage: python tools/build_fibsem_example_dataset.py DEST [--source DIR]
           [--upload]

``--source`` is a folder holding ``hela_cell_em.tif`` and
``hela_cell_mito.tif``; without it they are fetched from EMPIAR. ``--upload``
publishes the archive and card to ``einarolafsson/spacr-example-fibsem`` (a
Hugging Face dataset repository).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import tifffile

REPO = "einarolafsson/spacr-example-fibsem"
ARCHIVE = "spacr-example-fibsem.tar"
FOLDER = "make_masks_fibsem"
STEM = "hela_fibsem"
EMPIAR = ("https://ftp.ebi.ac.uk/empiar/world_availability/10982/data/"
          "mito_benchmarks/hela_cell/")
SOURCE_IMAGE = "hela_cell_em.tif"
SOURCE_LABELS = "hela_cell_mito.tif"
Z_RANGE = (64, 192)
VOXEL_NM = 15.0

CARD = """---
license: cc0-1.0
pretty_name: spaCR example data, Make Masks on FIB-SEM
task_categories:
- image-segmentation
tags:
- electron-microscopy
- fib-sem
- volume-em
- mitochondria
- 3d
- spacr
size_categories:
- n<1K
---

# spaCR example data: Make Masks on a FIB-SEM volume

One focused-ion-beam scanning electron microscopy (FIB-SEM) volume of a HeLa
cell, 128 x 256 x 256 voxels (Z, Y, X; 8-bit), with a 3-D instance label map
of every mitochondrion (uint16), for the 3-D mask editor on spaCR's Make Masks
"Load test data" menu. One menu entry opens the volume unsegmented; the other
opens it with these labels loaded for correction.

## Files

- `hela_fibsem.tif`: the volume, an ImageJ hyperstack (axes ZYX, unit nm,
  15 nm spacing).
- `ground_truth_masks/hela_fibsem.tif`: the labels, the same shape and
  calibration.
- `manifest.csv`: what a complete copy holds, with source and crop.

## Source

EMPIAR-10982, *Seven benchmark datasets of instance segmentation of
mitochondria: 6 diverse volume EM + 1 TEM (100 images) datasets*, folder
`data/mito_benchmarks/hela_cell` (Narayan K, Conrad RW; NCI/FNLCR).
https://www.ebi.ac.uk/empiar/EMPIAR-10982/ - DOI 10.6019/EMPIAR-10982

The crop is the central 128 sections (Z 64-191) of the 256^3 benchmark
volume at its full field. The image is unchanged; the labels are renumbered
1..n and stored as uint16. EMPIAR lists a 150 angstrom (15 nm) pixel and no
separate section thickness, so 15 nm is written on all three axes. The Z
value is an assumption: check it before measuring physical volumes.

Built by `tools/build_fibsem_example_dataset.py` in the spaCR repository.

## Licence

CC0 1.0, as EMPIAR releases its entries.

## Citation

Conrad R, Narayan K. Instance segmentation of mitochondria in electron
microscopy images with a generalist deep learning model trained on a diverse
dataset. Cell Systems 14(1):58-71.e5 (2023). doi:10.1016/j.cels.2022.12.006

Narayan K, Conrad RW. EMPIAR-10982 (2022). doi:10.6019/EMPIAR-10982
"""


def _read(source, name):
    """One source volume from ``source`` or fetched from EMPIAR."""
    if source is not None:
        return tifffile.imread(Path(source) / name)
    with urllib.request.urlopen(EMPIAR + name, timeout=600) as handle:
        return tifffile.imread(io.BytesIO(handle.read()))


def crop(image, labels):
    """Cut the fixed Z range and renumber the labels 1..n.

    :param image: the 256^3 source volume.
    :param labels: its instance label map.
    :returns: ``(image, labels)`` as 8-bit and uint16 ZYX arrays.
    """
    if image.shape != labels.shape or image.ndim != 3:
        raise ValueError(f"volume {image.shape} and labels {labels.shape} differ")
    low, high = Z_RANGE
    image = np.ascontiguousarray(image[low:high]).astype(np.uint8)
    part = labels[low:high]
    ids = np.unique(part)
    ids = ids[ids > 0]
    lookup = np.zeros(int(part.max()) + 1, dtype=np.uint16)
    lookup[ids] = np.arange(1, len(ids) + 1, dtype=np.uint16)
    return image, lookup[part]


def _write(path, array):
    """Write one ZYX ImageJ hyperstack with the 15 nm calibration."""
    tifffile.imwrite(path, array, imagej=True,
                     resolution=(1.0 / VOXEL_NM, 1.0 / VOXEL_NM),
                     metadata={"axes": "ZYX", "spacing": VOXEL_NM, "unit": "nm"})


def build(dest: Path, source=None) -> Path:
    """Stage the set and its archive under ``dest``.

    :param dest: output folder.
    :param source: folder with the two benchmark TIFFs, or None to fetch them.
    :returns: the archive path.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    image, labels = crop(_read(source, SOURCE_IMAGE), _read(source, SOURCE_LABELS))
    folder = dest / FOLDER
    shutil.rmtree(folder, ignore_errors=True)
    (folder / "ground_truth_masks").mkdir(parents=True)
    _write(folder / f"{STEM}.tif", image)
    _write(folder / "ground_truth_masks" / f"{STEM}.tif", labels)
    row = {"name": STEM, "image": f"{STEM}.tif",
           "ground_truth_mask": f"ground_truth_masks/{STEM}.tif",
           "empiar": "EMPIAR-10982", "source": "data/mito_benchmarks/hela_cell",
           "modality": "FIB-SEM", "z_range": f"{Z_RANGE[0]}-{Z_RANGE[1] - 1}",
           "shape_zyx": "x".join(map(str, image.shape)),
           "voxel_nm": VOXEL_NM, "mitochondria": int(labels.max()),
           "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
           "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest()}
    with (folder / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (dest / "README.md").write_text(CARD, encoding="utf-8")
    archive = dest / ARCHIVE
    with tarfile.open(archive, "w") as tar:
        for path in sorted(folder.rglob("*")):
            if path.name != "manifest.csv":
                tar.add(path, arcname=path.relative_to(folder).as_posix(),
                        recursive=False)
        tar.add(folder / "manifest.csv", arcname="manifest.csv")
    return archive


def upload(dest: Path) -> None:
    """Publish the archive and card to :data:`REPO`."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=str(Path(dest) / "README.md"),
                    path_in_repo="README.md", repo_id=REPO, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(Path(dest) / ARCHIVE),
                    path_in_repo=ARCHIVE, repo_id=REPO, repo_type="dataset")


def main(argv=None) -> int:
    """Build, and with ``--upload`` publish, the FIB-SEM example set."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dest", type=Path)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args(argv)
    archive = build(args.dest, args.source)
    print(f"{archive} {archive.stat().st_size} bytes")
    if args.upload:
        upload(args.dest)
        print(f"published to {REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
