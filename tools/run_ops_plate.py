#!/usr/bin/env python3
"""Run every well of one OPS plate through ``spacr.ops_engine.run_ops``, resumably.

Instruction 372, the full-plate run. Wells A1 and A2 of
``screenA/20200202_6W-LaC024A`` were validated on 2026-09-15 and B1-B3 ran
on 2026-09-20; the first full-plate run was started that evening, reached
well A1's decode, and died with the session that launched it. Its driver
lived in a scratchpad that was wiped, so nobody could restart it without
writing it again. This is that driver, committed.

    tools/run_capped.sh 64G python tools/run_ops_plate.py \\
        --plate /nas_mnt/data/ops/OpticalPooledScreens_data/screenA/20200202_6W-LaC024A \\
        --design /local/copy/of/pool10_design.csv \\
        --out /mnt/wd4tb/spacr_testdata/ops_plate_run

THE PATH. 372 once wrote the plate as ``/nas_mnt/data/ops/screenA/...``,
which is one level short: the plate is under
``/nas_mnt/data/ops/OpticalPooledScreens_data/screenA/``. The sequencing
tiles are ``<plate>/sequencing/images/input`` (``c1`` .. ``c11``) and the
phenotype tiles ``<plate>/phenotype/images/input``; both are passed
explicitly rather than as the plate root, although run_ops accepts either.

THE SETTINGS ARE THE B1-B3 RUN'S: ``ops_gpu=True``, ``n_workers=6``, every
other key at the engine's default (the measured constants of this plate and
the library's default Cellpose model), and the library rebuilt from the
reference's ``pool10_design.csv`` the way its ``*_0_sbs.smk`` does it:
``dialout`` 0 or 1, one row per unique ``sgRNA``, the first
``prefix_length`` bases. That gives 20,445 prefixes on this design table; a
different count is reported, because it means the table is not the one the
validated runs used.

RESUMABLE PER WELL. Each well writes ``<out>/results/<well>.json`` when all
four phases (stitch, phenotype, objects, decode) finish. A well whose file
says ``complete`` and whose barcodes are still in ``measurements.db`` is
skipped; any other well is run again from its stitch, which run_ops allows
because every phase replaces only that well's rows. A well that raises is
recorded with its error and the plate goes on to the next one.

``--store-reads WELL`` turns on ``ops_store_reads`` for the named wells
only. It changes nothing that is decoded; it writes ``ops_reads`` (about
thirty million rows for a well), which is what a per-field or per-cycle
look at one well needs. A completed well is rerun through all four phases
if requested reads were not saved, or their persisted count differs from
its saved decode report. Unrequested wells keep ordinary resume behavior.
The plate summary names the wells it was on.

THE NAS IS READ, NEVER WRITTEN. Everything goes under ``--out``. Launch the
whole driver through ``tools/nas_guard.sh run`` so a dead mount leaves an
abandoned process rather than a hung session.

At the end ``<out>/plate_summary.json`` carries one row per well -- tiles
placed, edges accepted, stitch residual median and max, phenotype fields
placed, objects, spots, library-exact rate, objects assigned, of those
library-exact, cycles and channels refused -- and the plate totals. The
plate's library-exact rate is weighted by spots and rebuilt from each
well's rate, which run_ops rounds to four places, so it carries that
rounding.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sqlite3
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASES = ("stitch", "phenotype", "objects", "decode")
EXPECTED_PREFIXES = 20445


def build_library(design: Path, destination: Path) -> int:
    """Write the pool10 prefix library the reference pipeline decodes against.

    :param design: the reference's ``pool10_design.csv``.
    :param destination: the ``prefix`` CSV to write.
    :returns: how many prefixes it holds.
    """
    seen = set()
    prefixes: List[str] = []
    with open(design, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("dialout", "")).strip() not in ("0", "1"):
                continue
            guide = row["sgRNA"]
            if guide in seen:
                continue
            seen.add(guide)
            prefixes.append(guide[:int(float(row["prefix_length"]))])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prefix"])
        writer.writerows([p] for p in prefixes)
    return len(prefixes)


def barcode_rows(db: Path, plate: str, well: str) -> int:
    """How many ``ops_barcodes`` rows the database holds for one well.

    :param db: the measurements database.
    :param plate: the plate name.
    :param well: the well name.
    :returns: the count, 0 when the table or file is missing.
    """
    if not db.exists():
        return 0
    try:
        with sqlite3.connect(str(db)) as connection:
            return int(connection.execute(
                "SELECT COUNT(*) FROM ops_barcodes WHERE plate = ? AND well = ?",
                (plate, well)).fetchone()[0])
    except sqlite3.Error:
        return 0


def well_row(report: Dict[str, Any]) -> Dict[str, Any]:
    """The per-well table row, taken from run_ops' report for that well.

    :param report: ``run_ops(...)["wells"][well]``.
    :returns: the row.
    """
    stitch = report.get("stitch") or {}
    phenotype = report.get("phenotype") or {}
    objects = report.get("objects") or {}
    decode = report.get("decode") or {}
    return {
        "tiles_placed": stitch.get("placed"),
        "tiles": stitch.get("sites"),
        "edges_accepted": stitch.get("edges_accepted"),
        "edges_proposed": stitch.get("edges_proposed"),
        "residual_median_px": stitch.get("residual_median_px"),
        "residual_max_px": stitch.get("residual_max_px"),
        "canvas_agrees": stitch.get("canvas_agrees"),
        "phenotype_fields_placed": phenotype.get("fields_mapped"),
        "phenotype_fields": phenotype.get("fields"),
        "phenotype_anchors": phenotype.get("anchors_used"),
        "objects": objects.get("objects"),
        "strict_refusal_groups": (objects.get("refusals") or {}).get("groups", 0),
        "strict_refusal_largest_px": (objects.get("refusals") or {}).get("largest_px"),
        "spots": decode.get("spots"),
        "library_exact_rate": decode.get("library_exact_rate"),
        "containment": decode.get("containment"),
        "objects_assigned": decode.get("objects_assigned"),
        "objects_assigned_library_exact": decode.get("objects_assigned_library_exact"),
        "objects_mapped": decode.get("objects_mapped"),
        "cycles_refused": decode.get("cycles_refused"),
        "channels_refused": decode.get("channels_refused"),
        "fields_decoded": decode.get("fields_decoded"),
        "fields": decode.get("fields"),
        "seconds": report.get("seconds"),
        "peak_rss_gb": report.get("peak_rss_gb"),
    }


def _stored_reads_match(db: Path, plate: str, well: str, record: Dict[str, Any]) -> bool:
    """Require evidence that the requested detailed reads survived on disk."""
    if not (record.get("settings") or {}).get("ops_store_reads"):
        return False
    expected = ((record.get("report") or {}).get("decode") or {}).get("ops_reads_rows")
    if type(expected) is not int or expected < 0:
        return False
    try:
        with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            actual = connection.execute(
                "SELECT COUNT(*) FROM ops_reads WHERE plate = ? AND well = ?",
                (plate, well)).fetchone()[0]
        return actual == expected
    except sqlite3.Error:
        return False


def plate_totals(rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Sum the per-well rows into the plate's.

    :param rows: ``well -> row`` for the completed wells.
    :returns: the totals.
    """
    def total(key: str) -> int:
        return int(sum(row.get(key) or 0 for row in rows.values()))

    spots = total("spots")
    exact_spots = sum((row.get("library_exact_rate") or 0) * (row.get("spots") or 0)
                      for row in rows.values())
    return {
        "wells": len(rows),
        "tiles_placed": total("tiles_placed"), "tiles": total("tiles"),
        "edges_accepted": total("edges_accepted"),
        "edges_proposed": total("edges_proposed"),
        "phenotype_fields_placed": total("phenotype_fields_placed"),
        "phenotype_fields": total("phenotype_fields"),
        "objects": total("objects"), "spots": spots,
        "library_exact_rate": round(exact_spots / spots, 4) if spots else None,
        "objects_assigned": total("objects_assigned"),
        "objects_assigned_library_exact": total("objects_assigned_library_exact"),
        "channels_refused": total("channels_refused"),
        "wells_with_cycles_refused": sorted(w for w, row in rows.items()
                                            if row.get("cycles_refused")),
        "seconds": round(sum(row.get("seconds") or 0 for row in rows.values()), 1),
    }


def format_table(rows: Dict[str, Dict[str, Any]], totals: Dict[str, Any]) -> str:
    """The plate as the text table 372's B1-B3 section uses.

    :param rows: ``well -> row``.
    :param totals: :func:`plate_totals` of them.
    :returns: the table.
    """
    def ratio(a, b):
        return "--" if a is None else f"{a}/{b}"

    def px(value):
        return "--" if value is None else f"{value:.2f} px"

    def count(value):
        return "--" if value is None else f"{value:,}"

    wells = sorted(rows)
    lines = [
        ("tiles placed", lambda r: ratio(r["tiles_placed"], r["tiles"]),
         ratio(totals["tiles_placed"], totals["tiles"])),
        ("edges accepted", lambda r: ratio(r["edges_accepted"], r["edges_proposed"]),
         ratio(totals["edges_accepted"], totals["edges_proposed"])),
        ("stitch residual median", lambda r: px(r["residual_median_px"]), ""),
        ("stitch residual max", lambda r: px(r["residual_max_px"]), ""),
        ("phenotype fields",
         lambda r: ratio(r["phenotype_fields_placed"], r["phenotype_fields"]),
         ratio(totals["phenotype_fields_placed"], totals["phenotype_fields"])),
        ("objects", lambda r: count(r["objects"]), count(totals["objects"])),
        ("spots", lambda r: count(r["spots"]), count(totals["spots"])),
        ("LIBRARY-EXACT RATE",
         lambda r: "--" if r["library_exact_rate"] is None else f"{r['library_exact_rate']:.4f}",
         "--" if totals["library_exact_rate"] is None else f"{totals['library_exact_rate']:.4f}"),
        ("objects assigned", lambda r: count(r["objects_assigned"]),
         count(totals["objects_assigned"])),
        ("of those, exact", lambda r: count(r["objects_assigned_library_exact"]),
         count(totals["objects_assigned_library_exact"])),
        ("cycles refused", lambda r: "none" if not r["cycles_refused"]
         else str(len(r["cycles_refused"])) + " fields", ""),
        ("channels refused", lambda r: count(r["channels_refused"]),
         count(totals["channels_refused"])),
    ]
    width = 12
    out = ["".ljust(24) + "".join(w.rjust(width) for w in wells) + "plate".rjust(width + 2)]
    for label, cell, plate in lines:
        out.append(label.ljust(24) + "".join(cell(rows[w]).rjust(width) for w in wells)
                   + str(plate).rjust(width + 2))
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the plate, one well at a time, and write the summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plate", type=Path, required=True,
                        help="the plate folder holding sequencing/ and phenotype/")
    parser.add_argument("--genotype", type=Path,
                        help="the sequencing tiles; <plate>/sequencing/images/input")
    parser.add_argument("--phenotype", type=Path,
                        help="the phenotype tiles; <plate>/phenotype/images/input")
    parser.add_argument("--design", type=Path, required=True,
                        help="a local copy of the reference's pool10_design.csv")
    parser.add_argument("--out", type=Path, required=True,
                        help="local output folder; never the NAS")
    parser.add_argument("--wells", nargs="*",
                        help="which wells; every well of the acquisition when omitted")
    parser.add_argument("--n-workers", type=int, default=6)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--store-reads", nargs="*", default=[],
                        help="wells to write ops_reads for")
    parser.add_argument("--force", action="store_true",
                        help="run wells that are already complete again")
    args = parser.parse_args(argv)

    from spacr.ops_engine import _index_tiles, run_ops

    genotype = args.genotype or args.plate / "sequencing" / "images" / "input"
    phenotype = args.phenotype or args.plate / "phenotype" / "images" / "input"
    plate = args.plate.name
    out = args.out
    results = out / "results"
    results.mkdir(parents=True, exist_ok=True)
    db = out / "measurements.db"

    library = out / "library" / "pool10_prefixes.csv"
    prefixes = build_library(args.design, library)
    print(f"library: {prefixes} prefixes -> {library}", flush=True)
    if prefixes != EXPECTED_PREFIXES:
        print(f"WARNING: {EXPECTED_PREFIXES} prefixes expected; this design "
              "table is not the one the validated runs used", flush=True)

    wells = ([w.upper() for w in args.wells] if args.wells
             else sorted(_index_tiles(str(genotype), "cycled")))
    store_reads = {w.upper() for w in args.store_reads}
    print(f"plate {plate}: wells {wells}", flush=True)

    base_settings = {
        "genotype_source": str(genotype), "phenotype_source": str(phenotype),
        "dst_root": str(out), "plate": plate,
        "ops_gpu": not args.no_gpu, "n_workers": args.n_workers,
    }
    plate_started = time.perf_counter()
    for well in wells:
        target = results / f"{well}.json"
        if target.exists() and not args.force:
            try:
                done = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                done = {}
            if done.get("complete") and barcode_rows(db, plate, well) > 0:
                if well not in store_reads or _stored_reads_match(db, plate, well, done):
                    print(f"{well}: complete, skipped", flush=True)
                    continue
                print(f"{well}: requested stored reads are missing or incomplete; rerunning", flush=True)
        settings = dict(base_settings, ops_store_reads=well in store_reads)
        started_at = datetime.datetime.now().isoformat(timespec="seconds")
        started = time.perf_counter()
        print(f"{well}: started {started_at}", flush=True)
        record: Dict[str, Any] = {"plate": plate, "well": well,
                                  "started": started_at, "settings": settings,
                                  "library": str(library),
                                  "library_prefixes": prefixes,
                                  "phases": list(PHASES)}
        try:
            report = run_ops(settings, wells=[well], phases=PHASES,
                             library=str(library))["wells"][well]
            record.update(complete=all(p in report for p in PHASES),
                          row=well_row(report), report=report)
        except Exception as error:
            record.update(complete=False, error=f"{type(error).__name__}: {error}",
                          traceback=traceback.format_exc())
            print(f"{well}: FAILED {record['error']}", flush=True)
        record["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
        record["wall_seconds"] = round(time.perf_counter() - started, 1)
        target.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
        print(f"{well}: {'complete' if record['complete'] else 'INCOMPLETE'} "
              f"in {record['wall_seconds']} s", flush=True)

    rows: Dict[str, Dict[str, Any]] = {}
    failed: Dict[str, str] = {}
    for well in wells:
        target = results / f"{well}.json"
        record = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
        if record.get("complete"):
            rows[well] = record["row"]
        else:
            failed[well] = record.get("error", "not run")
    totals = plate_totals(rows)
    summary = {
        "plate": plate, "genotype_source": str(genotype),
        "phenotype_source": str(phenotype), "db": str(db),
        "db_bytes": db.stat().st_size if db.exists() else None,
        "library": str(library), "library_prefixes": prefixes,
        "settings": base_settings, "store_reads_wells": sorted(store_reads),
        "wells": rows, "failed": failed, "totals": totals,
        "this_invocation_seconds": round(time.perf_counter() - plate_started, 1),
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (out / "plate_summary.json").write_text(json.dumps(summary, indent=1, default=str),
                                            encoding="utf-8")
    table = format_table(rows, totals) if rows else "no well completed"
    (out / "plate_summary.txt").write_text(table + "\n", encoding="utf-8")
    print(table, flush=True)
    if failed:
        print(f"not complete: {failed}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
