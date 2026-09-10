#!/usr/bin/env python3
"""Generate a small pooled-screen dataset with known positive/negative genes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "synthetic" / "regression"
GENES = ("000000", "233460", "239740", "111111")
ROWS = ("r1", "r2", "r3", "r4")
COLS = tuple(f"c{i}" for i in range(1, 9))


def main() -> int:
    scores_dir = OUTPUT / "scores"
    counts_dir = OUTPUT / "counts"
    scores_dir.mkdir(parents=True, exist_ok=True)
    counts_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    guides = [
        f"TGGT1_{gene}_{index}"
        for gene in GENES for index in range(1, 4)
    ]
    score_rows = []
    count_rows = []
    well_effects = {}
    for row in ROWS:
        for column in COLS:
            fractions = rng.dirichlet(np.full(len(guides), 2.5))
            total = int(rng.integers(2200, 4200))
            counts = np.maximum(1, np.rint(fractions * total).astype(int))
            positive_fraction = sum(
                value for guide, value in zip(guides, fractions)
                if "239740" in guide
            )
            negative_fraction = sum(
                value for guide, value in zip(guides, fractions)
                if "233460" in guide
            )
            phenotype = float(np.clip(
                0.38 + 1.45 * positive_fraction
                - 0.85 * negative_fraction + rng.normal(0, 0.018),
                0.03, 0.97,
            ))
            well_effects[f"plate1_{row}_{column}"] = phenotype
            for guide, count in zip(guides, counts):
                count_rows.append({
                    "plateID": "plate1",
                    "rowID": row,
                    "columnID": column,
                    "grna": guide,
                    "count": int(count),
                })
            for cell in range(12):
                score_rows.append({
                    "plateID": "plate1",
                    "rowID": row,
                    "columnID": column,
                    "fieldID": f"f{cell // 4 + 1}",
                    "objectID": cell + 1,
                    "pred": float(np.clip(
                        phenotype + rng.normal(0, 0.035), 0.01, 0.99
                    )),
                })

    score_path = scores_dir / "synthetic_scores.csv"
    count_path = counts_dir / "synthetic_counts.csv"
    pd.DataFrame(score_rows).to_csv(score_path, index=False)
    pd.DataFrame(count_rows).to_csv(count_path, index=False)
    manifest = {
        "schema": 1,
        "seed": 42,
        "plates": 1,
        "wells": len(ROWS) * len(COLS),
        "cells": len(score_rows),
        "guides": len(guides),
        "genes": list(GENES),
        "negative_gene": "233460",
        "positive_gene": "239740",
        "neutral_gene": "000000",
        "score_data": str(score_path),
        "count_data": str(count_path),
        "well_effects": well_effects,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(score_path)
    print(count_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
