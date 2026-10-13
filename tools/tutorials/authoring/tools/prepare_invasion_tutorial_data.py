#!/usr/bin/env python3
"""Create deterministic two-colour measurements for the Invasion tutorial."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "synthetic" / "invasion"
DB = OUTPUT / "measurements" / "measurements.db"


def main() -> int:
    rng = np.random.default_rng(2608)
    DB.parent.mkdir(parents=True, exist_ok=True)
    if DB.exists():
        DB.unlink()
    pathogen_rows = []
    cell_rows = []
    object_id = 0
    expected = {}
    for row_number in range(1, 7):
        row_id = f"r{row_number}"
        # c1 is a no-primary staining control: it defines the honest-negative
        # outside-channel distribution and is excluded from assay results.
        wells = (("c1", "staining control", 0.0, 60),
                 ("c2", "vehicle", 0.40, 120),
                 ("c3", "inhibitor", 0.70, 120))
        for column_id, condition, invaded_fraction, total in wells:
            invaded_count = int(round(total * invaded_fraction))
            labels = np.array([1] * invaded_count + [0] * (total - invaded_count))
            rng.shuffle(labels)
            expected[f"plate1_{row_id}_{column_id}"] = {
                "condition": condition,
                "n_total": total,
                "expected_invaded": invaded_count,
            }
            for local_index, invaded in enumerate(labels):
                field_id = f"f{1 + local_index // max(1, total // 2)}"
                object_id += 1
                # Outside stain is absent for invaded parasites and for the
                # c1 staining control. Total stain remains high in every row.
                if column_id == "c1" or invaded:
                    outside = float(np.clip(rng.normal(16, 3.2), 3, 29))
                else:
                    outside = float(np.clip(rng.normal(145, 24), 70, 230))
                total_signal = float(np.clip(rng.normal(220, 28), 120, 330))
                area = float(np.clip(rng.normal(430, 75), 220, 720))
                cell_id = 1 + local_index
                pathogen_rows.append((
                    object_id, "plate1", row_id, column_id, field_id,
                    f"plate1_{row_id}_{column_id}_{field_id}",
                    f"plate1_{row_id}_{column_id}_{field_id}",
                    cell_id, area, total_signal, outside,
                ))
            # One cell row is enough to seed the well if a later object filter
            # leaves no scored parasite in it.
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
                pathogen_channel_0_mean_intensity REAL,
                pathogen_channel_1_mean_intensity REAL
            )"""
        )
        conn.executemany(
            "INSERT INTO pathogen VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            pathogen_rows,
        )
        conn.execute(
            "CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, fieldID TEXT)"
        )
        conn.executemany("INSERT INTO cell VALUES (?, ?, ?, ?)", cell_rows)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "schema": 1,
        "kind": "synthetic two-colour invasion measurements",
        "channels": {"0": "total post-permeabilization stain", "1": "outside pre-permeabilization stain"},
        "control_wells": ["c1"],
        "conditions": {"c2": "vehicle", "c3": "inhibitor"},
        "pathogen_rows": len(pathogen_rows),
        "expected": expected,
    }, indent=2) + "\n")
    print(DB)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
