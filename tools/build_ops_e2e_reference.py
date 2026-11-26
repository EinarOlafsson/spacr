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

With ``--regions`` the tier is the "regions" tier (revision 4, item 0at):
three 2 x 2 blocks of well A1, listed in its ``regions.json``. Each block is
built on its own (the stitch keeps one connected group of tiles), with
``ops_sites`` and ``ops_ph_sites`` set to the block's tiles and fields, by
the route of revision 1 for the sequencing side (``stitch``, ``objects``,
``decode``) and the route above for the phenotype side. Every block's
references go to ``<out>/reference/regions/block_<tiles>/`` in the layout
of the full tier's ``reference/``.

With ``--l6 <full tier root>`` the tool builds the L6 references
(``tests/data/ops_e2e_l6/``) instead: the SBS nuclei (``stitch``, ``objects``),
the calls (the reference nuclei decoded on the full tier with the test's pins)
and the phenotype cells, all under today's defaults, on the CPU. They are the
references of the default-settings run of ``tests/test_ops_e2e_testdata.py``;
the shipped (pre-L6) references stay the ones of the old-settings run.

usage: python tools/build_ops_e2e_reference.py <tier root> <out> --reason TEXT
       [--tar FILE] [--override KEY=VALUE ...] [--regions] [--l6 FULL_ROOT
       [--l6-parts objects,barcodes,phenotype]]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
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


def _settings(root: Path, dst: Path, name: str, extra=None):
    from spacr.cli import load_settings_file

    path = root / "settings" / name
    if not path.is_file() and path.with_suffix(".json").is_file():
        path = path.with_suffix(".json")
    settings = dict(load_settings_file(str(path)))
    for key in PATH_KEYS:
        settings[key] = str(root / settings[key])
    settings["dst_root"] = str(dst)
    settings["n_workers"] = 1
    settings["ops_gpu"] = False
    settings["ops_ph_placement_seed"] = str(
        root / "reference" / "inputs" / "ops_ph_placement_seed.json")
    settings.update(extra or {})
    settings.update(OVERRIDES)
    return settings


def _run(settings, phases):
    from spacr.ops_engine import run_ops

    return run_ops(settings, wells=[WELL], phases=tuple(phases))


def _table(db: Path, name: str):
    import pandas as pd

    from spacr.ops_store import _read_rows

    frame = _read_rows(str(db), name, PLATE, [WELL])
    # The store's Measure-style object_key is not part of the references.
    return frame.drop(columns=[c for c in ("object_key",) if c in frame.columns])


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


def _pack(out: Path, tar_path: Path) -> None:
    """Write ``SHA256SUMS_rev3`` for ``out/reference`` and pack both.

    The archive unpacks as ``ops_testdata_funk2022_A1/...``, over the tier
    folder, as revision 2's reference archives do.
    """
    import hashlib

    ref = out / "reference"
    sums = []
    for path in sorted(p for p in ref.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sums.append(f"{digest}  {path.relative_to(out).as_posix()}")
    (out / "SHA256SUMS_rev3").write_text("\n".join(sums) + "\n")
    with tarfile.open(tar_path, "w") as tar:
        tar.add(out / "SHA256SUMS_rev3",
                arcname="ops_testdata_funk2022_A1/SHA256SUMS_rev3")
        tar.add(ref, arcname="ops_testdata_funk2022_A1/reference")


def _phenotype_and_merge(root: Path, run: Path, work: Path, ref: Path, extra=None,
                         tick=None) -> tuple:
    """Place, segment and measure the phenotype fields, then merge with both models.

    The run folder already holds the stitch and the SBS nuclei and calls.

    :returns: the phenotype report, and the seconds from ``tick`` to the end
        of ``ph_measure``.
    """
    settings = _settings(root, run, MODELS["tile_affine"], extra)
    report = _run(settings, ["phenotype", "ph_flatfield", "ph_segment",
                             "ph_measure"])["wells"][WELL]
    assert report["phenotype"]["placement"] == "seeded"
    seconds = round(time.perf_counter() - tick, 1) if tick is not None else None
    copy = work / ("similarity_" + run.name)
    shutil.copytree(run, copy)

    pheno = ref / "phenotype"
    pheno.mkdir(parents=True)
    shutil.copy(next(run.rglob(f"ops_phenotype_cells_{WELL}.parquet")), pheno)
    for folder in ("phenotype_cells", "phenotype_masks"):
        shutil.copytree(next(run.rglob(f"{WELL}/{folder}")), pheno / folder)
    (pheno / "phenotype_report.json").write_text(
        json.dumps({"phenotype": report["phenotype"]}, indent=1, default=str))

    for model, folder in (("tile_affine", run), ("similarity", copy)):
        model_settings = _settings(root, folder, MODELS[model], extra)
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

    return report, seconds


DECODE_FILES = (f"calls_{WELL}.parquet", "decode_report.json", f"field_stats_{WELL}.csv",
                f"decode_fdr_{WELL}.json", f"decode_fdr_{PLATE}.json")


def _build_regions(args) -> int:
    """Build every block of the regions tier: sequencing, phenotype and merge references.

    :param args: the parsed command line; ``args.root`` holds ``regions.json``.
    """
    import pandas as pd

    regions = json.loads((args.root / "regions.json").read_text())["regions"]
    work = args.out / "work"
    built = []
    for region in regions:
        tiles = sorted(int(t) for t in region["tiles"])
        name = "block_" + "-".join(str(t) for t in tiles)
        extra = {"ops_sites": ",".join(str(t) for t in tiles),
                 "ops_ph_sites": ",".join(str(int(f)) for f in region["fields"])}
        run = work / name
        run.mkdir(parents=True)
        ref = args.out / "reference" / "regions" / name
        tick = time.perf_counter()
        settings = _settings(args.root, run, MODELS["tile_affine"], extra)
        db = Path(_run(settings, ["stitch", "objects", "decode"])["db"])
        sequencing = round(time.perf_counter() - tick, 1)
        for table, path in (("ops_geometry", "stitch/ops_geometry.parquet"),
                            ("ops_objects", "objects/ops_objects.parquet"),
                            ("ops_barcodes", "genotype/ops_barcodes.parquet")):
            (ref / path).parent.mkdir(parents=True, exist_ok=True)
            _table(db, table).to_parquet(ref / path, index=False)
        for filename in DECODE_FILES:
            found = next(run.rglob(filename), None)
            if found is not None:
                shutil.copy(found, ref / "genotype" / filename)
        report, measured = _phenotype_and_merge(args.root, run, work, ref, extra, tick)
        cells = len(pd.read_parquet(ref / "phenotype" / f"ops_phenotype_cells_{WELL}.parquet"))
        built.append({"block": name, "tiles": tiles, "fields": list(region["fields"]),
                      "nuclei": int(len(pd.read_parquet(ref / "objects/ops_objects.parquet"))),
                      "calls": int(len(pd.read_parquet(ref / "genotype/ops_barcodes.parquet"))),
                      "phenotype_cells": cells, "stitch_to_decode_s": sequencing,
                      "stitch_to_measure_s": measured})
        print(json.dumps(built[-1]), flush=True)
    commit = subprocess.run(["git", "rev-parse", "--short=10", "HEAD"],
                            capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    (args.out / "reference" / "PROVENANCE_regions.json").write_text(json.dumps({
        "what": "revision-4 regions tier references: every block built from its raw "
                "images (stitch, objects, decode, phenotype, ph_flatfield, ph_segment, "
                "ph_measure, merge with tile_affine and similarity)",
        "reason": args.reason, "spacr_commit": commit,
        "route": "tools/build_ops_e2e_reference.py --regions",
        "settings_overrides": OVERRIDES, "blocks": built,
        "environment": _environment(), "machine": platform.node(),
    }, indent=1))
    shutil.rmtree(work)
    print(json.dumps({"blocks": len(built), "out": str(args.out)}))
    return 0


def _build_l6(args) -> int:
    """Build the L6 references: nuclei, calls and phenotype cells under today's defaults.

    The routes are those of ``tests/test_ops_e2e_testdata.py``: the nuclei by
    ``stitch`` and ``objects`` on the core tier; the calls by ``decode`` of the
    shipped reference nuclei on the full tier, with the test's pins of
    ``ops_read_neighbour_discount`` and ``ops_pixel_read_gate``; the cells by the
    seeded phenotype route of the shipped references.

    :param args: the parsed command line; ``args.root`` is the core tier and
        ``args.l6`` the full tier.
    """
    import pandas as pd
    from spacr.ops_engine import _replace_well_rows

    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "work"
    timing = {}
    parts = set(args.l6_parts.split(","))
    tick = time.perf_counter()
    if "objects" in parts:
        objects_run = work / "objects"
        objects_run.mkdir(parents=True)
        db = Path(_run(_settings(args.root, objects_run, MODELS["tile_affine"]),
                       ["stitch", "objects"])["db"])
        _table(db, "ops_objects").sort_values("object_id").reset_index(
            drop=True).to_parquet(args.out / "ops_objects.parquet", index=False)
        timing["objects_s"] = round(time.perf_counter() - tick, 1)

    tick = time.perf_counter()
    decode_run = work / "decode"
    if "barcodes" in parts:
        decode_run.mkdir(parents=True)
    settings = _settings(args.l6, decode_run, MODELS["tile_affine"])
    settings["ops_read_neighbour_discount"] = 1.0
    settings["ops_pixel_read_gate"] = False
    if "barcodes" in parts:
        db = Path(_run(settings, ["stitch"])["db"])
        _replace_well_rows(str(db), "ops_objects", pd.read_parquet(
            args.l6 / "reference" / "objects" / "ops_objects.parquet"), PLATE, WELL)
        _run(settings, ["decode"])
        _table(db, "ops_barcodes").sort_values("object_id").reset_index(
            drop=True).to_parquet(args.out / "ops_barcodes.parquet", index=False)
        timing["decode_s"] = round(time.perf_counter() - tick, 1)

    tick = time.perf_counter()
    if "phenotype" in parts:
        cells_run = work / "phenotype"
        cells_run.mkdir(parents=True)
        settings = _settings(args.root, cells_run, MODELS["tile_affine"])
        db = Path(_run(settings, ["stitch"])["db"])
        for table, path in (("ops_objects", "objects/ops_objects.parquet"),
                            ("ops_barcodes", "genotype/ops_barcodes.parquet")):
            _replace_well_rows(str(db), table, pd.read_parquet(
                args.root / "reference" / path), PLATE, WELL)
        report = _run(settings, ["phenotype", "ph_flatfield", "ph_segment",
                                 "ph_measure"])["wells"][WELL]
        assert report["phenotype"]["placement"] == "seeded"
        shutil.copy(next(cells_run.rglob(f"ops_phenotype_cells_{WELL}.parquet")),
                    args.out)
        timing["phenotype_s"] = round(time.perf_counter() - tick, 1)

    commit = subprocess.run(["git", "rev-parse", "--short=10", "HEAD"],
                            capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    counts = {"objects": int(len(pd.read_parquet(args.out / "ops_objects.parquet"))),
              "barcodes": int(len(pd.read_parquet(args.out / "ops_barcodes.parquet"))),
              "phenotype_cells": int(len(pd.read_parquet(
                  args.out / f"ops_phenotype_cells_{WELL}.parquet")))}
    record = args.out / "PROVENANCE_l6.json"
    built = {}
    if record.is_file():
        previous = json.loads(record.read_text())
        timing = {**previous.get("timing", {}), **timing}
        built = previous.get("built") or {
            part: {key: previous.get(key) for key in (
                "spacr_commit", "reason", "settings_overrides", "environment",
                "machine")} for part in ("objects", "barcodes", "phenotype")}
    for part in parts:
        built[part] = {"spacr_commit": commit, "reason": args.reason,
                       "settings_overrides": dict(OVERRIDES),
                       "environment": _environment(), "machine": platform.node()}
    record.write_text(json.dumps({
        "what": "L6 references of tests/test_ops_e2e_testdata.py under today's defaults",
        "route": "tools/build_ops_e2e_reference.py --l6", "counts": counts,
        "timing": timing, "built": built,
    }, indent=1))
    shutil.rmtree(work)
    print(json.dumps({"out": str(args.out), **counts}))
    return 0


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
    parser.add_argument("--l6", type=Path, metavar="FULL_ROOT",
                        help="build the L6 references (nuclei, calls, phenotype "
                             "cells under today's defaults); the value is the "
                             "full tier root")
    parser.add_argument("--l6-parts", default="objects,barcodes,phenotype",
                        help="with --l6, the references to (re)build, a "
                             "comma-separated subset of objects, barcodes, "
                             "phenotype; the others in the folder are kept")
    parser.add_argument("--regions", action="store_true",
                        help="build every block of the regions tier from its "
                             "raw images (revision 4)")
    args = parser.parse_args(argv)
    for item in args.override:
        key, sep, value = item.partition("=")
        if not sep or not key:
            parser.error(f"--override wants KEY=VALUE, got {item!r}")
        OVERRIDES[key] = value
    if args.out.exists() and any(args.out.iterdir()) and not args.l6:
        parser.error(f"{args.out} is not empty")
    if args.l6:
        return _build_l6(args)
    if args.regions:
        return _build_regions(args)
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
    ref = args.out / "reference"
    pheno = ref / "phenotype"
    _report, timing["stitch_to_measure_s"] = _phenotype_and_merge(
        args.root, run, work, ref, tick=tick)

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
    # Revision 2's addendum stays beside revision 3's: the loaders look for
    # it, and it still describes how every unchanged reference file was made.
    previous = args.root / "reference" / "PROVENANCE_rev2_addendum.json"
    if previous.is_file():
        shutil.copy(previous, ref / previous.name)
    if args.tar:
        _pack(args.out, args.tar)
    print(json.dumps({"cells": cells, "out": str(args.out),
                      "tar": str(args.tar) if args.tar else None}))


if __name__ == "__main__":
    sys.exit(main())
