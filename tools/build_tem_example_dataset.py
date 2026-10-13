"""Stage Make Masks' TEM test data: ten micrographs with mitochondria masks.

Source: EMPIAR-10982, "TEM 100 image benchmark dataset of instance
segmentation of mitochondria" (Conrad and Narayan, NCI/FNLCR), folder
``data/tem_benchmark``: grayscale TEM images and 2-D instance label maps of
every mitochondrion. EMPIAR releases its entries under CC0 1.0.

What is cut, by a fixed rule and without hand-picking: the benchmark's own
NCI-TEM acquisitions (the ``Datset info`` column of
``tem_benchmark_metadata.csv``) imaged by TEM at 1024 x 1024 with at least
one mitochondrion, ordered by ``sha256("spacr-example-tem:" + uid)``, the
first ten. The images are written unchanged as 8-bit TIFFs; the masks as
uint16 TIFFs (the source stores int32), label for label.

The images open UNSEGMENTED, as the Toxoplasma test data does: the masks go
in ``ground_truth_masks/``, which Make Masks does not read. The benchmark
carries no pixel size, so neither do these files.

Output (``dest``): ``make_masks_tem/`` (the unpacked set), ``README.md``
(the dataset card) and ``spacr-example-tem.tar`` (manifest last, so a
transfer that died part-way reads as absent).

Usage: python tools/build_tem_example_dataset.py DEST [--source DIR]
           [--upload]

``--source`` is a local copy holding ``tem_benchmark_metadata.csv`` and
``tem/images`` and ``tem/masks``; without it the files are fetched from
EMPIAR. ``--upload`` publishes the archive and card to
``einarolafsson/spacr-example-tem`` (a Hugging Face dataset repository).
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

REPO = "einarolafsson/spacr-example-tem"
ARCHIVE = "spacr-example-tem.tar"
FOLDER = "make_masks_tem"
EMPIAR = ("https://ftp.ebi.ac.uk/empiar/world_availability/10982/data/")
METADATA = "tem_benchmark_metadata.csv"
SALT = "spacr-example-tem:"
COUNT = 10

CARD = """---
license: cc0-1.0
pretty_name: spaCR example data, Make Masks on TEM
task_categories:
- image-segmentation
tags:
- electron-microscopy
- tem
- mitochondria
- spacr
size_categories:
- n<1K
---

# spaCR example data: Make Masks on transmission electron micrographs

Ten transmission electron micrographs (1024 x 1024, 8-bit) with instance
masks of every mitochondrion, for spaCR's Make Masks "Load test data" menu.
The images open unsegmented; the published masks sit in
`ground_truth_masks/` to compare a draft against.

## Source

EMPIAR-10982, *Seven benchmark datasets of instance segmentation of
mitochondria: 6 diverse volume EM + 1 TEM (100 images) datasets*, folder
`data/tem_benchmark` (Narayan K, Conrad RW; NCI/FNLCR).
https://www.ebi.ac.uk/empiar/EMPIAR-10982/ - DOI 10.6019/EMPIAR-10982

The ten fields are the benchmark's own NCI-TEM acquisitions imaged by TEM at
1024 x 1024 with at least one mitochondrion, ordered by
`sha256("spacr-example-tem:" + uid)`, the first ten. Images are unchanged;
masks are the published label maps written as uint16. The benchmark records
no pixel size. `manifest.csv` lists each field's EMPIAR sample UID and the
benchmark's metadata for it.

Built by `tools/build_tem_example_dataset.py` in the spaCR repository.

## Licence

CC0 1.0, as EMPIAR releases its entries.

## Citation

Conrad R, Narayan K. Instance segmentation of mitochondria in electron
microscopy images with a generalist deep learning model trained on a diverse
dataset. Cell Systems 14(1):58-71.e5 (2023). doi:10.1016/j.cels.2022.12.006
"""


def chosen(rows, masks_of):
    """The ten sample UIDs the rule picks.

    :param rows: the benchmark metadata rows.
    :param masks_of: ``uid -> (image, mask)`` arrays.
    :returns: the UIDs, in the rule's order.
    """
    pool = sorted((r["Sample UID"] for r in rows
                   if r["Datset info"] == "NCI-TEM"
                   and r["Imaging Method"] == "TEM"),
                  key=lambda uid: hashlib.sha256((SALT + uid).encode()).hexdigest())
    picked = []
    for uid in pool:
        image, mask = masks_of(uid)
        if image.shape == (1024, 1024) and mask.max() > 0:
            picked.append(uid)
        if len(picked) == COUNT:
            break
    return picked


def _reader(source):
    """``uid -> (image, mask)`` from a local copy, or fetched from EMPIAR."""
    def read(uid):
        """Read one field and its mask."""
        out = []
        for kind in ("images", "masks"):
            if source is not None:
                out.append(tifffile.imread(source / "tem" / kind / f"{uid}.tiff"))
            else:
                url = f"{EMPIAR}tem_benchmark/{kind}/{uid}.tiff"
                with urllib.request.urlopen(url, timeout=120) as handle:
                    out.append(tifffile.imread(io.BytesIO(handle.read())))
        return out[0], out[1]
    return read


def build(dest: Path, source=None) -> Path:
    """Stage the set and its archive under ``dest``.

    :param dest: output folder.
    :param source: local copy of the benchmark, or None to fetch it.
    :returns: the archive path.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if source is not None:
        metadata = (source / METADATA).read_text(encoding="utf-8")
    else:
        with urllib.request.urlopen(EMPIAR + METADATA, timeout=120) as handle:
            metadata = handle.read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(metadata)))
    by_uid = {r["Sample UID"]: r for r in rows}
    read = _reader(source)
    folder = dest / FOLDER
    shutil.rmtree(folder, ignore_errors=True)
    (folder / "ground_truth_masks").mkdir(parents=True)
    manifest = []
    for index, uid in enumerate(chosen(rows, read), 1):
        image, mask = read(uid)
        stem = f"tem_{index:02d}_{uid}"
        tifffile.imwrite(folder / f"{stem}.tif", image.astype(np.uint8),
                         photometric="minisblack")
        tifffile.imwrite(folder / "ground_truth_masks" / f"{stem}.tif",
                         mask.astype(np.uint16), photometric="minisblack")
        row = by_uid[uid]
        manifest.append({
            "name": stem, "image": f"{stem}.tif",
            "ground_truth_mask": f"ground_truth_masks/{stem}.tif",
            "empiar": "EMPIAR-10982", "sample_uid": uid,
            "organism": row["Organism"], "cell_type": row["Cell Type"],
            "biological_entity": row["Biological entity"],
            "mitochondria": int(len(np.unique(mask)) - 1)})
    with (folder / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
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
    api.upload_file(path_or_fileobj=str(dest / "README.md"),
                    path_in_repo="README.md", repo_id=REPO, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(dest / ARCHIVE), path_in_repo=ARCHIVE,
                    repo_id=REPO, repo_type="dataset")


def main(argv=None) -> int:
    """Build, and with ``--upload`` publish, the TEM example set."""
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
