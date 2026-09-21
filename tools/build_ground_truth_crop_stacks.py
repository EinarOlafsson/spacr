#!/usr/bin/env python
"""Build the stacks item 470's object crops are cut from.

ITEM 470 (2026-09-21): one crop per ground-truth object of the four
cross-channel datasets -- Toxoplasma PVs, cells and nuclei -- each with the
same three channels, Hoechst, the parasite stain and CellMask, so the
maintainer can call every label real or not.

This writes, per field, one ``(H, W, 6)`` uint16 stack in a ``merged/``
folder per experiment-and-plate, named the way Measure reads a field
(:func:`crop_names`):

    0 hoechst    1 toxoplasma    2 cellmask
    3 cell mask  4 nucleus mask  5 PV mask

WHERE EACH PLANE COMES FROM, local first (the project tree on this disk):

    hoechst      toxoplasma_from_hoechst/images, else cell_from_hoechst/images
    toxoplasma   data/_dsred_source (the parasite channel the project staged),
                 else the field's own merged array on the NAS, at the plate's
                 pathogen channel (default 2), copied through nas_guard.sh
    cellmask     toxoplasma_from_cellmask/images, else nuclei_from_cellmask/images
    cell mask    cell_from_hoechst/masks
    nucleus mask nuclei_from_cellmask/masks
    PV mask      toxoplasma_from_cellmask/masks_pv, else toxoplasma_from_hoechst/masks_pv

A MASK A FIELD DOES NOT HAVE IS WRITTEN EMPTY: that field simply yields no
crop of that object type, which is right -- only the dataset that labelled
an object type should contribute crops of it. A missing CHANNEL is not
filled: the field is skipped and said, because a crop with a blank channel
would be judged on a picture the stain never made.

Resumable, and one field in memory at a time.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_pv_count_dataset import field_name  # noqa: E402

PROJECT = Path("/mnt/wd4tb/af3/projects/cross_channel_models")
DATA = PROJECT / "data"
DATASETS = ("toxoplasma_from_hoechst", "toxoplasma_from_cellmask",
            "cell_from_hoechst", "nuclei_from_cellmask")
PLANES = ("hoechst", "toxoplasma", "cellmask", "cell_mask", "nucleus_mask",
          "pv_mask")
SOURCES = {
    "hoechst": [("toxoplasma_from_hoechst", "images"),
                ("cell_from_hoechst", "images")],
    "cellmask": [("toxoplasma_from_cellmask", "images"),
                 ("nuclei_from_cellmask", "images")],
    "cell_mask": [("cell_from_hoechst", "masks")],
    "nucleus_mask": [("nuclei_from_cellmask", "masks")],
    "pv_mask": [("toxoplasma_from_cellmask", "masks_pv"),
                ("toxoplasma_from_hoechst", "masks_pv")],
}
MASKS = ("cell_mask", "nucleus_mask", "pv_mask")
NAS_GUARD = HERE / "nas_guard.sh"


def crop_names(stems: Sequence[str]) -> Dict[str, str]:
    """``stem -> <plate>_<well>_<field>`` names that are unique across stems.

    :func:`build_pv_count_dataset.field_name` folds the screen into the plate,
    which is enough for the CSA and MTOC screens. It is NOT enough for the
    THP1 screen: three acquisitions, 2025-02-18, -19 and -26, each have a
    ``PLATE1``, so ``THP1screen-PLATE1_A01_13`` would name three different
    fields. Where one screen-and-plate comes from more than one acquisition,
    the acquisition's date and time go into the plate id as well
    (``THP1screen-PLATE1-20250218153752``), still without an underscore, so
    Measure keeps it whole.

    :param stems: every field.
    :returns: the mapping.
    """
    acquisitions: Dict[str, set] = {}
    for stem in stems:
        named = field_name(stem)
        if named != stem:
            acquisitions.setdefault(named.rsplit("_", 2)[0], set()).add(
                stem.split("__")[1])
    out: Dict[str, str] = {}
    for stem in stems:
        named = field_name(stem)
        plate = named.rsplit("_", 2)[0]
        if named != stem and len(acquisitions[plate]) > 1:
            stamp = "".join(re.findall(r"[0-9]+", stem.split("__")[1]))
            named = f"{plate}-{stamp}_" + "_".join(named.rsplit("_", 2)[1:])
        out[stem] = named
    return out


def every_field() -> Dict[str, List[str]]:
    """``stem -> datasets that hold it`` across the four datasets.

    :returns: the mapping.
    """
    out: Dict[str, List[str]] = {}
    for dataset in DATASETS:
        for name in os.listdir(DATA / dataset / "images"):
            if name.endswith(".tif"):
                out.setdefault(name[:-4], []).append(dataset)
    return out


def _read(path: Path) -> Optional[np.ndarray]:
    """A TIFF plane, or None when it is not there.

    :param path: the file.
    :returns: the array.
    """
    import tifffile

    return np.asarray(tifffile.imread(path)) if path.is_file() else None


def local_plane(plane: str, stem: str) -> Optional[np.ndarray]:
    """The first local copy of ``plane`` for ``stem``.

    :param plane: a :data:`PLANES` name.
    :param stem: the field.
    :returns: the array, or None.
    """
    if plane == "toxoplasma":
        return _read(DATA / "_dsred_source" / f"{stem}.tif")
    for dataset, folder in SOURCES[plane]:
        found = _read(DATA / dataset / folder / f"{stem}.tif")
        if found is not None:
            return found
    return None


def _merged_folders() -> Dict[str, List[str]]:
    """``experiment -> merged folders on the NAS``, from the project's index.

    :returns: the mapping, merged over every host's copy of the index.
    """
    out: Dict[str, List[str]] = {}
    for path in DATA.glob("_merged_index*.json"):
        try:
            for key, folders in json.loads(path.read_text()).items():
                out.setdefault(key, [])
                out[key] += [f for f in folders if f not in out[key]]
        except (ValueError, OSError):
            continue
    return out


def _pathogen_channel(plate_dir: str) -> int:
    """The plate's parasite channel, from its own mask settings (default 2).

    :param plate_dir: the plate folder on the NAS.
    :returns: the channel index.
    """
    settings = os.path.join(plate_dir, "settings", "gen_mask_settings.csv")
    try:
        text = Path(settings).read_text()
    except OSError:
        return 2
    found = re.search(r'pathogen_channel,"?([0-9]+)', text)
    return int(found.group(1)) if found else 2


def nas_toxoplasma(stem: str, index: Dict[str, List[str]]) -> Optional[np.ndarray]:
    """The parasite channel of one field, read from its merged array.

    The NAS can hang a process for good, so the array is copied to local disk
    by ``tools/nas_guard.sh run`` with a time limit and read from there.

    :param stem: ``<screen>__<experiment>__<plate>_<well>_<field>_<n>``.
    :param index: from :func:`_merged_folders`.
    :returns: the plane, or None.
    """
    parts = stem.split("__")
    if len(parts) < 3:
        return None
    experiment, tail = parts[1], parts[2]
    for folder in index.get(experiment, []):
        source = os.path.join(folder, f"{tail}.npy")
        with tempfile.TemporaryDirectory(prefix="gt_crops_") as scratch:
            target = os.path.join(scratch, "field.npy")
            done = subprocess.run(
                ["bash", str(NAS_GUARD), "run", "180", "cp", source, target],
                capture_output=True, text=True)
            if done.returncode != 0 or not os.path.isfile(target):
                continue
            array = np.load(target, mmap_mode="r")
            channel = _pathogen_channel(os.path.dirname(folder.rstrip("/")))
            if array.ndim == 3 and channel < array.shape[2]:
                return np.array(array[:, :, channel])
    return None


def build_stack(stem: str, index: Dict[str, List[str]]) -> Optional[np.ndarray]:
    """Every plane of one field, in :data:`PLANES` order.

    :param stem: the field.
    :param index: NAS merged folders, for a parasite plane not staged locally.
    :returns: ``(H, W, 6)`` uint16, or None when a channel is missing.
    """
    planes: Dict[str, Optional[np.ndarray]] = {p: local_plane(p, stem) for p in PLANES}
    if planes["toxoplasma"] is None:
        planes["toxoplasma"] = nas_toxoplasma(stem, index)
    for channel in ("hoechst", "toxoplasma", "cellmask"):
        if planes[channel] is None:
            print(f"    {stem}: no {channel}; skipped")
            return None
    shape = planes["hoechst"].shape
    for mask in MASKS:
        if planes[mask] is None:
            planes[mask] = np.zeros(shape, np.uint16)
    if {p.shape for p in planes.values()} != {shape}:
        print(f"    {stem}: planes disagree on shape; skipped")
        return None
    return np.stack([planes[p].astype(np.uint16, copy=False) for p in PLANES], -1)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out", required=True)
    parser.add_argument("--plate", action="append", default=None)
    parser.add_argument("--fields", type=int, default=0,
                        help="stop after this many fields per plate (0 = all)")
    parser.add_argument("--list-plates", action="store_true")
    args = parser.parse_args(argv)

    fields = every_field()
    names = crop_names(sorted(fields))
    if len(set(names.values())) != len(names):
        raise SystemExit("two fields share a name; nothing written")
    by_plate: Dict[str, List[str]] = {}
    for stem in sorted(fields):
        by_plate.setdefault(names[stem].rsplit("_", 2)[0], []).append(stem)
    if args.list_plates:
        for plate, stems in sorted(by_plate.items()):
            print(f"  {plate:24s} {len(stems):5d} fields  "
                  f"{len(stems) * 48 / 1000:.1f} GB as stacks")
        return 0
    index = _merged_folders()
    written = existing = skipped = 0
    for plate in args.plate or sorted(by_plate):
        stems = by_plate.get(plate, [])
        if args.fields:
            stems = stems[:args.fields]
        folder = Path(args.out) / plate / "merged"
        folder.mkdir(parents=True, exist_ok=True)
        print(f"\n{plate}: {len(stems)} fields -> {folder}")
        with open(Path(args.out) / plate / "fields.csv", "w", newline="",
                  encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["field", "stem", "datasets"])
            for number, stem in enumerate(stems, 1):
                target = folder / f"{names[stem]}.npy"
                if not target.exists():
                    stack = build_stack(stem, index)
                    if stack is None:
                        skipped += 1
                        continue
                    temporary = str(target) + ".tmp.npy"
                    np.save(temporary, stack)
                    os.replace(temporary, target)
                    written += 1
                else:
                    existing += 1
                writer.writerow([names[stem], stem, ";".join(fields[stem])])
                if number % 50 == 0 or number == len(stems):
                    print(f"    {number}/{len(stems)}")
    print(f"\nwritten {written}, already there {existing}, skipped {skipped}")
    print("planes: " + ", ".join(f"{i} {p}" for i, p in enumerate(PLANES)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
