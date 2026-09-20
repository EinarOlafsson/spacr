#!/usr/bin/env python
"""Build the merged stacks the PV-count classifier's training crops come from.

ITEM 449. The maintainer's plan of 2026-09-20: put the PV mask and the
channels back together as one array per field, in a `merged/` folder per
plate, and then run the Measure module over them so it cuts one PNG per
vacuole. Those PNGs are what he annotates, and the annotated set is what
the counting heads train on.

WHERE THE CHANNELS COME FROM. spaCR's three cross-channel datasets cover
the same fields -- the stem is the same in all three -- and between them
they carry four of the five planes:

    Hoechst      cross-channel-cell-from-hoechst/images
    CellMask     cross-channel-nuclei-from-cellmask/images
    cell mask    cross-channel-cell-from-hoechst/masks
    nucleus mask cross-channel-nuclei-from-cellmask/masks
    PV mask      cross-channel-toxoplasma-from-cellmask/masks_pv

THE PARASITE STAIN IS NOT IN ANY OF THEM, and that was measured rather than
assumed: `cross-channel-toxoplasma-from-cellmask/images` is BYTE-IDENTICAL
to `cross-channel-nuclei-from-cellmask/images` (same sha256 for the same
stem), so both are the CellMask input. These are cross-channel datasets
precisely because they predict toxoplasma WITHOUT its own stain. So two of
the three counting heads the maintainer asked for -- count from Hoechst and
count from CellMask -- can be trained from this; the parasite-stain head
needs the original acquisitions.

SIZE, BEFORE YOU RUN IT. A field is 1998x1998 uint16, so one plane is 8 MB
and a five-plane stack is 40 MB. All 3,030 fields would be 121 GB written
and about the same again downloaded. One plate is enough to start: plate1
is 515 fields and 6,642 vacuoles, which is already far more than anyone
will annotate by hand.

Resumable: a stack that is already written is skipped, and the downloads go
through the Hugging Face cache, so an interrupted run costs only what it
had not finished.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Each plane of the stack: (name, dataset repo, folder inside it).
#: The ORDER IS THE CONTRACT -- `measure_crop` is told which plane is which
#: by index, and the settings printed at the end of a run use these.
PLANES: Tuple[Tuple[str, str, str], ...] = (
    ("hoechst", "einarolafsson/cross-channel-cell-from-hoechst", "images"),
    ("cellmask", "einarolafsson/cross-channel-nuclei-from-cellmask", "images"),
    ("cell_mask", "einarolafsson/cross-channel-cell-from-hoechst", "masks"),
    ("nucleus_mask", "einarolafsson/cross-channel-nuclei-from-cellmask", "masks"),
    ("pv_mask", "einarolafsson/cross-channel-toxoplasma-from-cellmask", "masks_pv"),
)

#: Where the field list and the train/test split are read from. Any of the
#: three would do -- they cover the same fields -- and this is the one whose
#: object counts are the vacuoles.
INDEX_REPO = "einarolafsson/cross-channel-toxoplasma-from-cellmask"

#: Plate names are inside the stem, between the acquisition and the well.
_PLATE = re.compile(r"__([A-Za-z0-9]*plate[0-9]+)_")


def plate_of(stem: str) -> str:
    """Which plate a field belongs to, from its stem.

    :param stem: a field stem, without the extension.
    :returns: the plate name, or ``"unplated"`` when the stem does not
        carry one -- 765 of the 3,030 do not, and they are kept together
        rather than dropped.
    """
    found = _PLATE.search(stem)
    return found.group(1) if found else "unplated"


def read_index() -> List[Dict[str, str]]:
    """Every field, with its split and its vacuole count."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(INDEX_REPO, "fields.csv", repo_type="dataset")
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fetch_plane(repo: str, folder: str, stem: str) -> Optional[np.ndarray]:
    """One plane of one field, or None when that dataset lacks it."""
    import imageio.v2 as imageio
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        path = hf_hub_download(repo, f"{folder}/{stem}.tif", repo_type="dataset")
    except EntryNotFoundError:
        return None
    except Exception as exc:                                  # noqa: BLE001
        print(f"    {stem}: {folder} could not be fetched ({exc})")
        return None
    return np.asarray(imageio.imread(path))


def build_stack(stem: str) -> Optional[np.ndarray]:
    """Every plane of one field, stacked in :data:`PLANES` order.

    A field missing any plane is refused outright rather than written with
    a zeroed one: a stack whose PV mask is silently empty would be measured
    as a field with no vacuoles, which is a wrong number rather than a
    missing one.

    :param stem: the field.
    :returns: ``(H, W, len(PLANES))`` uint16, or None.
    """
    planes = []
    for name, repo, folder in PLANES:
        plane = fetch_plane(repo, folder, stem)
        if plane is None:
            print(f"    {stem}: no {name}; field skipped")
            return None
        planes.append(plane)
    shapes = {plane.shape for plane in planes}
    if len(shapes) != 1:
        print(f"    {stem}: planes disagree on shape {sorted(shapes)}; skipped")
        return None
    return np.stack([plane.astype(np.uint16, copy=False) for plane in planes],
                    axis=-1)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge the cross-channel datasets into per-plate "
                    "merged/ stacks for the Measure module.")
    parser.add_argument("--out", required=True,
                        help="root folder; one subfolder per plate is made "
                             "under it")
    parser.add_argument("--plate", action="append", default=None,
                        help="only this plate; repeatable. Default: all.")
    parser.add_argument("--split", default="all",
                        choices=["all", "train", "test"],
                        help="only fields on this side of the split")
    parser.add_argument("--fields", type=int, default=0,
                        help="stop after this many fields per plate "
                             "(0 = every one)")
    parser.add_argument("--list-plates", action="store_true",
                        help="print the plates and what they hold, write "
                             "nothing")
    args = parser.parse_args(argv)

    rows = read_index()
    by_plate: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        stem = (row.get("name") or "").strip()
        if not stem:
            continue
        if args.split != "all" and (row.get("split") or "") != args.split:
            continue
        by_plate.setdefault(plate_of(stem), []).append(row)

    if args.list_plates:
        print(f"{len(rows)} fields, split={args.split}")
        for plate in sorted(by_plate):
            fields = by_plate[plate]
            objects = sum(int(row.get("n_objects") or 0) for row in fields)
            print(f"  {plate:12s} {len(fields):5d} fields  {objects:7d} "
                  f"vacuoles  {len(fields) * 40 / 1000:.1f} GB as stacks")
        return 0

    wanted = args.plate or sorted(by_plate)
    written = skipped = existing = 0
    for plate in wanted:
        fields = by_plate.get(plate)
        if not fields:
            print(f"{plate}: no fields; skipped")
            continue
        if args.fields:
            fields = fields[:args.fields]
        folder = os.path.join(args.out, plate, "merged")
        os.makedirs(folder, exist_ok=True)
        print(f"\n{plate}: {len(fields)} fields -> {folder}")
        manifest = os.path.join(args.out, plate, "fields.csv")
        with open(manifest, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["stem", "split", "n_objects", "stack"])
            for index, row in enumerate(fields, 1):
                stem = row["name"].strip()
                target = os.path.join(folder, f"{stem}.npy")
                if os.path.exists(target):
                    existing += 1
                    writer.writerow([stem, row.get("split", ""),
                                     row.get("n_objects", ""), target])
                    continue
                stack = build_stack(stem)
                if stack is None:
                    skipped += 1
                    continue
                temporary = target + ".tmp.npy"
                np.save(temporary, stack)
                os.replace(temporary, target)
                written += 1
                writer.writerow([stem, row.get("split", ""),
                                 row.get("n_objects", ""), target])
                if index % 25 == 0 or index == len(fields):
                    print(f"    {index}/{len(fields)} fields")

    print(f"\nwritten {written}, already there {existing}, skipped {skipped}")
    if written or existing:
        print("\nPLANE ORDER, which is what Measure has to be told:")
        for index, (name, _repo, _folder) in enumerate(PLANES):
            print(f"    {index}  {name}")
        print("\nMeasure settings for one plate:")
        print(f"    src                = '<out>/<plate>/merged'")
        print( "    channels           = [0, 1]")
        print( "    cell_mask_dim      = 2")
        print( "    nucleus_mask_dim   = 3")
        print( "    pathogen_mask_dim  = 4")
        print( "    save_png           = True")
        print( "    png_objects        = ['pathogen']   # one PNG per vacuole")
    return 0


if __name__ == "__main__":
    sys.exit(main())
