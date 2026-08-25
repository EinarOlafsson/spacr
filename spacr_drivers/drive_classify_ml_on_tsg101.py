"""Drive classify's tabular half on the real tsg101 measurements.

One plate of the screen is sixty thousand cells with two control columns, so
this is the machine-learning half of classify on a real screen: fit on the
per-object features, score every well, compute permutation importance and
SHAP, write the figures, and merge the scores back into ``png_list``.

THE MERGE IS THE PART THAT SILENTLY FAILS. A screen stamped with a doubled
plate prefix on disk (``pplate1``) and a plain one everywhere computed since
(``plate1``) produces keys that never meet, and the run then reports that its
results "probably come from a different experiment" about the database it has
just read. ``schema.canonical_plate_id`` is the rule that collapses the
doubling. This driver reports how many rows merged, because zero merged rows
is what that failure looks like from outside.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _support import (check, dataset_root, preflight, require, run, scratch,
                      stage)

DEFAULT_ROOT = "/mnt/firecuda2/Claude/toxoplasma_projects/tsg101_screen"

#: One plate is enough, and the plate folder is the project the module reads.
PLATE = "plate1"

REQUIRED = (f"{PLATE}/measurements/measurements.db",)

SETTINGS = dict(
    classifier_family="ml", model_type_ml="xgboost",
    n_estimators=100, n_jobs=4,
    negative_control="c1", positive_control="c2",
    channel_of_interest=3, min_cell_count=25,
    cross_validation=False, n_repeats=2, verbose=False)


def merged_rows(database):
    """How many ``png_list`` rows carry a score, and how many do not.

    A score column that exists and is empty is the failure this driver is
    watching for: the run reports success and every row is NaN.
    """
    import sqlite3

    import pandas as pd

    with sqlite3.connect(database) as connection:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'", connection)
        if "png_list" not in set(tables["name"]):
            return 0, 0
        rows = pd.read_sql("SELECT * FROM png_list", connection)
    scored = [c for c in rows.columns if "pred" in c or "score" in c]
    if not scored:
        return 0, len(rows)
    filled = int(rows[scored].notna().any(axis=1).sum())
    return filled, len(rows)


def main(argv):
    """Stage one plate of the screen and fit the tabular classifier on it."""
    root = require(dataset_root(argv, DEFAULT_ROOT), REQUIRED,
                   what="the tsg101 screen")
    print(f"dataset root: {root}")

    work = scratch("classify_ml_on_tsg101")
    print(f"staging {PLATE} (this copies a large measurements database)")
    stage(root / PLATE, ["measurements/measurements.db"], work)

    settings = dict(SETTINGS, src=str(work))
    preflight(settings, "classify")

    import matplotlib

    matplotlib.use("Agg")
    from spacr.classify import classify

    classify(settings)

    filled, total = merged_rows(work / "measurements" / "measurements.db")
    print(f"\nscores merged back into png_list: {filled} of {total} rows")
    check(total > 0, "png_list is empty, so there was nothing to merge scores "
                     "into and nothing this run can be judged against")
    check(filled > 0,
          f"none of the {total} png_list rows carries a score. The fit ran, "
          f"the figures were written and the run reported success -- the "
          f"plate identity on disk and the one the run computed did not meet, "
          f"which is a join that silently matched nothing rather than an "
          f"error anything raised.")


if __name__ == "__main__":
    run(main)
