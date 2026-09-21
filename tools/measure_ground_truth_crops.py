#!/usr/bin/env python
"""Cut one crop per ground-truth object from the stacks of
:mod:`build_ground_truth_crop_stacks`, one plate at a time.

Every labelled object must come out, so everything Measure normally uses to
drop objects is off: no minimum or maximum sizes, no merging of cells that
share an edge parasite, uninfected cells kept. The heavy measurements
(radial, spatial, distances, correlation, texture) are off too; the point is
the pictures and the ``png_list`` table that ties each one to its field and
label.

ONE OBJECT TYPE PER RUN. The three ground-truth masks were drawn for three
different datasets and do not nest -- a PV can straddle two ground-truth
cells -- and Measure, asked for all three at once, relates them and refuses
such a field. So each type is measured on its own, with the other two masks
switched off, under its own folder:

    <root>/<type>/<plate>/merged -> <root>/<plate>/merged   (a link)
    <root>/<type>/<plate>/<type>_png/...                    (the crops)
    <root>/<type>/<plate>/measurements/measurements.db      (png_list)

Each crop is RGB with CellMask red, Toxoplasma green and Hoechst blue; the
full-bit-depth arrays go under ``region_array`` when ``--arrays`` is given.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

SETTINGS = {
    "channels": [0, 1, 2],
    "cell_min_size": 0,
    "nucleus_min_size": 0,
    "pathogen_min_size": 0,
    "cytoplasm_min_size": 0,
    "cytoplasm": False,
    "uninfected": True,
    "merge_edge_pathogen_cells": False,
    "save_measurements": True,
    "radial_dist": False,
    "spatial_measurements": False,
    "object_distances": False,
    "object_distance_maxima": False,
    "object_distance_intensity": False,
    "calculate_correlation": False,
    "homogeneity": False,
    "save_png": True,
    "png_channel_mapping": {"r": 2, "g": 1, "b": 0},
    "normalize": [1, 99],
    "normalize_by": "png",
    "use_bounding_box": False,
    "plot": False,
    "resume": True,
}
TYPES = {"pathogen": ("pathogen_mask_dim", 5), "cell": ("cell_mask_dim", 3),
         "nucleus": ("nucleus_mask_dim", 4)}
#: ``type -> (window px, rim)``. Crops are cut at full resolution, never
#: resized, so the window is set from the objects' measured extent (a sample
#: of 30 fields, 2026-09-21: 99th percentile cell 191 px, nucleus 75 px,
#: PV 52 px). The rim grows the label by ``rim x sqrt(area)`` before the
#: crop, so a label is judged with what is around it: wide for a PV, whose
#: host cell is the evidence, narrow for a cell, which fills its window.
FRAMING = {"pathogen": (128, 1.5), "nucleus": (128, 0.5), "cell": (320, 0.3)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True,
                        help="the --out of build_ground_truth_crop_stacks")
    parser.add_argument("--plate", action="append", default=None)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--arrays", action="store_true")
    parser.add_argument("--type", action="append", choices=sorted(TYPES),
                        default=None)
    args = parser.parse_args(argv)
    from spacr.measure import measure_crop
    from spacr.settings import get_measure_crop_settings

    plates = args.plate or sorted(
        p.name for p in Path(args.root).iterdir()
        if (p / "merged").is_dir() and not (p / "merged").is_symlink()
        and p.name not in TYPES)
    for kind in args.type or ["pathogen", "cell", "nucleus"]:
        for plate in plates:
            stacks = (Path(args.root) / plate / "merged").resolve()
            folder = Path(args.root) / kind / plate
            folder.mkdir(parents=True, exist_ok=True)
            src = folder / "merged"
            if not src.exists():
                src.symlink_to(stacks, target_is_directory=True)
            print(f"\n== {kind} / {plate}: {src}", flush=True)
            settings = dict(SETTINGS, src=str(src), n_jobs=args.jobs,
                            experiment=plate, save_arrays=args.arrays,
                            crop_mode=[kind], cell_mask_dim=None,
                            png_size=[FRAMING[kind][0]] * 2,
                            dialate_pngs=True,
                            dialate_png_ratios=[FRAMING[kind][1]],
                            nucleus_mask_dim=None, pathogen_mask_dim=None)
            settings[TYPES[kind][0]] = TYPES[kind][1]
            measure_crop(get_measure_crop_settings(settings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
