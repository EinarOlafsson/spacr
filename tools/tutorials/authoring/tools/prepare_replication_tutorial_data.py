#!/usr/bin/env python3
"""Create deterministic explicit-vacuole measurements for Replication."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "synthetic" / "replication"
DB = OUTPUT / "measurements" / "measurements.db"


DISTRIBUTIONS = {
    "vehicle": [1] * 2 + [2] * 6 + [4] * 12 + [8] * 16 + [16] * 2 + [3] * 2,
    "inhibitor": [1] * 10 + [2] * 16 + [4] * 10 + [8] * 2 + [3] * 2,
}


def main() -> int:
    rng = np.random.default_rng(2708)
    DB.parent.mkdir(parents=True, exist_ok=True)
    if DB.exists():
        DB.unlink()
    pathogen_rows = []
    cell_rows = []
    expected = {}
    object_label = 0
    for row_number in range(1, 7):
        row_id = f"r{row_number}"
        for column_id, condition in (("c1", "vehicle"), ("c2", "inhibitor")):
            counts = list(DISTRIBUTIONS[condition])
            rng.shuffle(counts)
            key = f"plate1_{row_id}_{column_id}"
            expected[key] = {
                "condition": condition,
                "n_vacuoles": len(counts),
                "n_parasites": int(sum(counts)),
                "counts": counts,
            }
            for vacuole_index, count in enumerate(counts, start=1):
                field_id = f"f{1 + (vacuole_index - 1) // 20}"
                prcf = f"plate1_{row_id}_{column_id}_{field_id}"
                # Pair consecutive vacuoles in the same host cell to ensure
                # that explicit vacuole identity—not cell_id—is the unit.
                cell_id = 1 + (vacuole_index - 1) // 2
                vacuole_id = f"{row_id}_{column_id}_v{vacuole_index}"
                for _ in range(count):
                    object_label += 1
                    area = float(np.clip(rng.normal(420, 65), 220, 720))
                    pathogen_rows.append((
                        object_label, "plate1", row_id, column_id, field_id,
                        prcf, prcf, cell_id, area, vacuole_id,
                    ))
            cell_rows.append(("plate1", row_id, column_id, "f1"))
    with sqlite3.connect(DB) as conn:
        conn.execute(
            """CREATE TABLE pathogen (
                object_label INTEGER,
                plateID TEXT,
                rowID TEXT,
                columnID TEXT,
                fieldID TEXT,
                prcf TEXT,
                file_name TEXT,
                cell_id INTEGER,
                pathogen_area REAL,
                vacuole_id TEXT
            )"""
        )
        conn.executemany(
            "INSERT INTO pathogen VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            pathogen_rows,
        )
        conn.execute(
            "CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, fieldID TEXT)"
        )
        conn.executemany("INSERT INTO cell VALUES (?, ?, ?, ?)", cell_rows)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "schema": 1,
        "kind": "synthetic parasite rows with explicit vacuole identity",
        "conditions": {"c1": "vehicle", "c2": "inhibitor"},
        "pathogen_rows": len(pathogen_rows),
        "expected": expected,
    }, indent=2) + "\n")
    print(DB)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
