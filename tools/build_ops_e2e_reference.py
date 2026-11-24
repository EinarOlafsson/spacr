"""Rebuild the phenotype and merge references of the OPS end-to-end test data.

The route of ``reference/PROVENANCE_rev2_addendum.json`` (how revision 2 was
made), which is also the route of ``tests/test_ops_e2e_testdata.py``:

1. ``stitch`` with the tier's shipped settings (paths made absolute,
   ``n_workers=1``, ``ops_gpu=False``, the shipped placement seed);
2. the published ``reference/objects/ops_objects.parquet`` and
   ``reference/genotype/ops_barcodes.parquet`` written into the run's
   database (fixed inputs; never re-segmented or re-decoded);
3. ``phenotype`` (seeded), ``ph_flatfield``, ``ph_segment``, ``ph_measure``;
4. ``merge`` with ``tile_affine`` (the settings file), and on a copy of the
   run folder after ``ph_measure``, ``merge`` with the similarity settings.

The outputs are written under ``<out>/reference/{phenotype,merge}`` in the
layout revision 2 ships, plus a provenance addendum naming the spaCR commit,
the environment and the reason. ``--tar`` packs them for publishing.

Revision 3 (2026-11-23) exists because the maintainer's ownership rule
(Measure's nucleus-cell merge, ``spacr.utils._merge_overlapping_objects``,
adopted by OPS through ``spacr.ops_sample.pair_nuclei_with_cells``) moves
the phenotype cells from 490 to 484.

usage: python tools/build_ops_e2e_reference.py <tier root> <out> --reason TEXT
       [--tar FILE] [--override KEY=VALUE ...]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time
from importlib import metadata
from pathlib import Path

WELL = "A1"
PLATE = "20200202_6W-LaC024A"
PATH_KEYS = ("genotype_source", "phenotype_source", "ops_library",
             "ops_flatfield_maps")
MODELS = {"tile_affine": "ops_testdata_settings.csv",
          "similarity": "ops_testdata_settings_similarity.csv"}


#: ``--override key=value`` pairs, applied to every settings file read.
OVERRIDES: dict = {}


def _settings(root: Path, dst: Path, name: str):
    from spacr.cli import load_settings_file

    settings = dict(load_settings_file(str(root / "settings" / name)))
    for key in PATH_KEYS:
        settings[key] = str(root / settings[key])
    settings["dst_root"] = str(dst)
    settings["n_workers"] = 1
    settings["ops_gpu"] = False
    settings["ops_ph_placement_seed"] = str(
        root / "reference" / "inputs" / "ops_ph_placement_seed.json")
    settings.update(OVERRIDES)
    return settings


def _run(settings, phases):
    from spacr.ops_engine import run_ops

    return run_ops(settings, wells=[WELL], phases=tuple(phases))


def _table(db: Path, name: str):
    import pandas as pd

    with sqlite3.connect(str(db)) as con:
        return pd.read_sql(f"select * from {name} where well = ?", con,
                           params=(WELL,))


def _environment():
    names = ("numpy", "scipy", "torch", "cellpose", "tifffile", "pandas",
             "scikit-image", "imagecodecs")
    out = {"python": platform.python_version()}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    out["torch_device"] = f"cpu (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})"
    out["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS")
    return out


def main(argv=None):
    import pandas as pd
    from spacr.ops_engine import _replace_well_rows

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="the unpacked tier root")
    parser.add_argument("out", type=Path, help="an empty output folder")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--tar", type=Path)
    parser.add_argument("--override", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="a settings value for every run (recorded in "
                             "the provenance), e.g. ops_ph_image_format=npz, "
                             "revision 2's mask format")
    args = parser.parse_args(argv)
    for item in args.override:
        key, sep, value = item.partition("=")
        if not sep or not key:
            parser.error(f"--override wants KEY=VALUE, got {item!r}")
        OVERRIDES[key] = value
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"{args.out} is not empty")
    work = args.out / "work"
    run = work / "tile_affine"
    run.mkdir(parents=True)
    timing = {}

    tick = time.perf_counter()
    settings = _settings(args.root, run, MODELS["tile_affine"])
    db = Path(_run(settings, ["stitch"])["db"])
    for table, path in (("ops_objects", "objects/ops_objects.parquet"),
                        ("ops_barcodes", "genotype/ops_barcodes.parquet")):
        _replace_well_rows(str(db), table, pd.read_parquet(
            args.root / "reference" / path), PLATE, WELL)
    report = _run(settings, ["phenotype", "ph_flatfield", "ph_segment",
                             "ph_measure"])["wells"][WELL]
    assert report["phenotype"]["placement"] == "seeded"
    timing["stitch_to_measure_s"] = round(time.perf_counter() - tick, 1)
    copy = work / "similarity"
    shutil.copytree(run, copy)

    ref = args.out / "reference"
    pheno = ref / "phenotype"
    pheno.mkdir(parents=True)
    shutil.copy(next(run.rglob(f"ops_phenotype_cells_{WELL}.parquet")), pheno)
    for folder in ("phenotype_cells", "phenotype_masks"):
        shutil.copytree(next(run.rglob(f"{WELL}/{folder}")), pheno / folder)
    (pheno / "phenotype_report.json").write_text(
        json.dumps({"phenotype": report["phenotype"]}, indent=1, default=str))

    for model, folder in (("tile_affine", run), ("similarity", copy)):
        model_settings = _settings(args.root, folder, MODELS[model])
        result = _run(model_settings, ["merge"])
        merged = result["wells"][WELL]["merge"]
        model_db = Path(result["db"])
        target = ref / "merge" / model
        target.mkdir(parents=True)
        (target / "merge_report.json").write_text(
            json.dumps(merged, indent=1, default=str))
        _table(model_db, "ops_merge").to_parquet(target / "ops_merge.parquet", index=False)
        _table(model_db, "ops_phenotype").to_parquet(target / "ops_phenotype.parquet",
                                                     index=False)

    commit = subprocess.run(["git", "rev-parse", "--short=10", "HEAD"],
                            capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    cells = len(pd.read_parquet(pheno / f"ops_phenotype_cells_{WELL}.parquet"))
    (ref / "PROVENANCE_rev3_addendum.json").write_text(json.dumps({
        "what": "revision-3 phenotype and merge references (core tier); every "
                "other reference file is revision 2's, unchanged",
        "reason": args.reason,
        "spacr_commit": commit,
        "route": "tools/build_ops_e2e_reference.py (the route of "
                 "PROVENANCE_rev2_addendum.json)",
        "settings_overrides": OVERRIDES,
        "phenotype_cells": cells,
        "timing": timing,
        "environment": _environment(),
        "machine": platform.node(),
    }, indent=1))
    shutil.rmtree(work)
    import hashlib

    sums = []
    for path in sorted(p for p in ref.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sums.append(f"{digest}  {path.relative_to(args.out).as_posix()}")
    (args.out / "SHA256SUMS_rev3").write_text("\n".join(sums) + "\n")
    if args.tar:
        with tarfile.open(args.tar, "w") as tar:
            tar.add(args.out / "SHA256SUMS_rev3",
                    arcname="ops_testdata_funk2022_A1/SHA256SUMS_rev3")
            tar.add(ref, arcname="ops_testdata_funk2022_A1/reference")
    print(json.dumps({"cells": cells, "out": str(args.out),
                      "tar": str(args.tar) if args.tar else None}))


if __name__ == "__main__":
    sys.exit(main())
