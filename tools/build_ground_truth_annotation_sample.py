#!/usr/bin/env python
"""Draw the sample of ground-truth crops the maintainer judges real or not real.

Item 470 cut one crop per ground-truth object -- 585,311 of them over three
object types and twelve experiment-plates (tools/measure_ground_truth_crops.py).
Nobody labels that many by hand, so the first round is a sample: for each
object type, ``--per-type`` crops drawn at random from every plate at once,
every object with the same chance, seeded so the draw is the same each time.

Each type's sample is one folder Annotate can open as its source:

    <root>/annotation_sample/<type>/measurements/measurements.db

holding a ``png_list`` table whose ``png_path`` points at the crops where they
already are -- nothing is copied -- plus ``dataset_plate`` (which plate the
crop came from) and an empty ``real`` column for the annotation. A crop drawn
twice in a later, larger round keeps its first judgement: rows already in the
sample are left alone.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path("/mnt/wd4tb/af3/projects/cross_channel_models/ground_truth_crops")
TYPES = ("pathogen", "cell", "nucleus")


def plate_databases(root: Path, kind: str):
    """``(plate, measurements.db)`` for every plate of one object type."""
    for plate in sorted(os.listdir(root / kind)):
        database = root / kind / plate / "measurements" / "measurements.db"
        if database.is_file():
            yield plate, database


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--per-type", type=int, default=1000)
    parser.add_argument("--seed", default="470")
    args = parser.parse_args(argv)
    import random

    for kind in TYPES:
        rows = []
        for plate, database in plate_databases(args.root, kind):
            with sqlite3.connect(database) as source:
                columns = [c[1] for c in source.execute("PRAGMA table_info(png_list)")]
                for record in source.execute("SELECT * FROM png_list"):
                    rows.append((plate, dict(zip(columns, record))))
        rng = random.Random(int(hashlib.sha256(
            f"{args.seed}/{kind}".encode()).hexdigest()[:16], 16))
        picked = sorted(rng.sample(rows, min(args.per_type, len(rows))),
                        key=lambda r: (r[0], r[1]["png_path"]))
        folder = args.root / "annotation_sample" / kind / "measurements"
        folder.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(folder / "measurements.db", timeout=30) as target:
            target.execute(
                "CREATE TABLE IF NOT EXISTS png_list (png_path TEXT PRIMARY KEY, "
                "file_name TEXT, plateID TEXT, rowID TEXT, columnID TEXT, "
                "fieldID TEXT, prcfo TEXT, object_id TEXT, dataset_plate TEXT, "
                "real INTEGER)")
            before = target.execute("SELECT COUNT(*) FROM png_list").fetchone()[0]
            id_column = f"{kind}_id"
            target.executemany(
                "INSERT OR IGNORE INTO png_list (png_path, file_name, plateID, "
                "rowID, columnID, fieldID, prcfo, object_id, dataset_plate) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(r["png_path"], r.get("file_name"), r.get("plateID"),
                  r.get("rowID"), r.get("columnID"), r.get("fieldID"),
                  r.get("prcfo"), r.get(id_column), plate) for plate, r in picked])
            after = target.execute("SELECT COUNT(*) FROM png_list").fetchone()[0]
        print(f"{kind}: {len(rows)} crops, {after - before} added, {after} in the "
              f"sample -> {folder.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
