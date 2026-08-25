"""Drive measure on real plate1 fields and compare with the reference database.

The plate1 dataset is a complete pipeline kept stage by stage, so the
``measurements.db`` shipped beside it is the reference an earlier run of this
same pipeline produced. This driver re-measures three of its merged fields
into a scratch copy and then compares, object for object and column for
column, against that reference restricted to the same fields.

WHAT A DISAGREEMENT MEANS. A different object count is a segmentation or
filtering change and has to be explained. A different value in a shared
numeric column is either a fix or a regression, so every column that is
expected to differ is listed below with the reason; anything else is reported
as unexplained.

The comparison does not stop at "these two numbers differ". ``cell_id`` -- the
parent cell each nucleus and pathogen belongs to -- is re-derived from the
mask planes themselves, by the label that covers most of the object, so the
driver can say which database is right rather than only that they disagree.

Zernike columns are absent unless mahotas is installed, and the run says so.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _support import (dataset_root, preflight, read_settings, require, run,
                      scratch, settings_file, stage, undeclared)

DEFAULT_ROOT = "/home/olafsson/datasets/plate1"

#: What the run reads. `merged/` is the input; the database is the reference
#: the result is judged against, so its absence is a refusal too -- a run with
#: nothing to compare against is not what this driver is for.
REQUIRED = ("merged/*.npy", "measurements/measurements.db")

#: Where the recorded run kept its settings, nearest first.
SETTINGS_CANDIDATES = ("../settings/crop_measure_settings.csv",
                       "settings/measure_crop_settings.csv")

#: Three fields is enough for the comparison to mean something (143 cells) and
#: small enough to rerun in a couple of minutes.
FIELDS = ("1_1", "9_1", "10_1")

#: Object tables the comparison covers, and the field column they carry.
TABLES = ("cell", "nucleus", "pathogen", "cytoplasm")

#: Column fragments whose disagreement with the reference is a documented fix,
#: each with the reason, so a difference is explained where it is reported.
EXPECTED_TO_DIFFER = {
    "frac_high90": "thresholded on each object's own percentile, which pinned "
                   "the fraction near 0.10 for every object",
    "frac_low10": "thresholded on each object's own percentile, which pinned "
                  "the fraction near 0.10 for every object",
    "rad_dist_channel_0_bin_0": "bin 0 held the field background, because the "
                                "distance map was multiplied by the cell mask "
                                "instead of read inside it",
    "rad_dist_channel_1_bin_0": "bin 0 held the field background",
    "rad_dist_channel_2_bin_0": "bin 0 held the field background",
    "rad_dist_channel_3_bin_0": "bin 0 held the field background",
    "rad_dist_channel_0_bin_5": "the outermost bin is now closed, so the "
                                "farthest pixel is no longer dropped",
    "rad_dist_channel_1_bin_5": "the outermost bin is now closed",
    "rad_dist_channel_2_bin_5": "the outermost bin is now closed",
    "rad_dist_channel_3_bin_5": "the outermost bin is now closed",
    "blur": "measured on a 1-D vector of the object's pixels, which OpenCV "
            "read as an Nx1 image, so it was a second difference along raster "
            "order rather than a focus measure",
    "cell_id": "re-derived from the mask planes below rather than compared "
               "against a reference that was itself wrong",
}


def field_ids(fields):
    """The ``fieldID`` spelling the database uses for each merged field name.

    ``plate1_E01_9_1.npy`` is field ``f9``: the field number is the third
    part of the stem and the database prefixes it with ``f``.
    """
    return {f"f{name.split('_')[0]}" for name in fields}


def collapse_doubled_prefix(name):
    """Fold a column name that carries its own prefix twice.

    The reference database spells the blur columns
    ``cell_channel_0_cell_channel_0_blur``: the object-and-channel prefix was
    applied once by the measurement and again by the writer. The name is
    ``cell_channel_0_blur`` now, and folding the old spelling is what lets the
    VALUES be compared instead of the column being reported as missing.
    """
    for split in range(1, len(name)):
        head = name[:split]
        if name.startswith(f"{head}_{head}_"):
            return name.replace(f"{head}_{head}_", f"{head}_", 1)
    return name


def why_expected(column):
    """The recorded reason a column is expected to differ, or None."""
    for fragment, reason in EXPECTED_TO_DIFFER.items():
        if fragment in column:
            return reason
    return None


def dominant_cell(cell_plane, object_plane, label):
    """The cell label that covers most of one object, straight from the masks.

    This is the ground truth for ``cell_id``: whichever cell owns the majority
    of the object's pixels owns the object. 0 means the object lies outside
    every cell.
    """
    import numpy as np

    covered = cell_plane[object_plane == label]
    covered = covered[covered > 0]
    if covered.size == 0:
        return 0
    values, counts = np.unique(covered, return_counts=True)
    return int(values[counts.argmax()])


def check_parent_assignment(database, merged_dir, fields, cell_dim=4,
                            child_dims=(("nucleus", 5), ("pathogen", 6))):
    """Re-derive every child object's parent cell from the mask planes.

    The rule is the one ``_map_child_to_parent`` uses: the cell label covering
    most of the child's pixels. It is computed here from the UNFILTERED planes
    in the merged file, so a handful of objects legitimately disagree -- the
    run drops cells below ``cell_min_size`` before it links anything, and a
    child whose majority parent was dropped is then linked to the next one.
    Wholesale disagreement is the finding; a few are the filter.

    :param database: a measurements database to check.
    :param merged_dir: the merged ``.npy`` files the run measured.
    :param fields: the merged field names, e.g. ``('1_1', '9_1')``.
    :returns: ``{table: (agreeing, total)}``.
    """
    import sqlite3

    import numpy as np
    import pandas as pd

    result = {}
    for table, dim in child_dims:
        agreeing = total = 0
        for field in fields:
            planes = np.load(Path(merged_dir) / f"plate1_E01_{field}.npy")
            field_id = f"f{field.split('_')[0]}"
            with sqlite3.connect(database) as connection:
                rows = pd.read_sql(
                    f"SELECT object_label, cell_id FROM {table} "
                    f"WHERE fieldID = ?", connection, params=(field_id,))
            for label, recorded in zip(rows["object_label"], rows["cell_id"]):
                total += 1
                truth = dominant_cell(planes[..., cell_dim], planes[..., dim],
                                      int(label))
                # A child outside every cell is NaN in the database and 0
                # here; they mean the same thing and must not count as a
                # disagreement.
                claimed = 0 if pd.isna(recorded) else int(float(recorded))
                agreeing += int(claimed == truth)
        result[table] = (agreeing, total)
    return result


def compare_with_reference(measured_db, reference_db, fields):
    """Report where a fresh measurement disagrees with the reference.

    :param measured_db: the database this run wrote.
    :param reference_db: the database shipped with the dataset.
    :param fields: the merged field names that were re-measured.
    :returns: True when every object count matches and every column that
        differs is one of the differences recorded above.
    """
    import sqlite3

    import numpy as np
    import pandas as pd

    wanted = sorted(field_ids(fields))
    clause = ",".join("?" * len(wanted))
    agreed = True
    for table in TABLES:
        with sqlite3.connect(measured_db) as fresh, \
                sqlite3.connect(reference_db) as reference:
            query = f"SELECT * FROM {table} WHERE fieldID IN ({clause})"
            new = pd.read_sql(query, fresh, params=wanted)
            old = pd.read_sql(query, reference, params=wanted)
        if len(new) != len(old):
            agreed = False
            print(f"  {table}: {len(new)} objects, reference has {len(old)}")
            continue
        print(f"  {table}: {len(new)} objects, same as the reference")

        old = old.rename(columns={c: collapse_doubled_prefix(c)
                                  for c in old.columns})
        key = ["fieldID", "object_label"]
        new = new.sort_values(key).reset_index(drop=True)
        old = old.sort_values(key).reset_index(drop=True)
        shared = [column for column in new.columns
                  if column in old.columns
                  and pd.api.types.is_numeric_dtype(new[column])
                  and pd.api.types.is_numeric_dtype(old[column])]
        differing = [column for column in shared
                     if not np.allclose(new[column].to_numpy(dtype=float),
                                        old[column].to_numpy(dtype=float),
                                        rtol=1e-6, atol=1e-9, equal_nan=True)]
        explained = [c for c in differing if why_expected(c)]
        unexplained = [c for c in differing if not why_expected(c)]
        print(f"    {len(shared) - len(differing)}/{len(shared)} shared numeric "
              f"columns agree to 1e-6")
        for column in explained:
            print(f"    differs, and why: {column} -- {why_expected(column)}")
        if unexplained:
            agreed = False
            print(f"    UNEXPLAINED differences: {unexplained}")
        missing = sorted(set(old.columns) - set(new.columns))
        zernike = [c for c in missing if "zernike" in c]
        other = [c for c in missing if "zernike" not in c]
        if zernike:
            print(f"    {len(zernike)} zernike columns are absent: mahotas is "
                  f"not installed, so they were not measured")
        if other:
            agreed = False
            print(f"    columns the reference has and this run does not: {other}")
    return agreed


def main(argv):
    """Stage three merged fields, measure them, compare with the reference."""
    root = require(dataset_root(argv, DEFAULT_ROOT), REQUIRED,
                   what="the plate1 pipeline dataset")
    print(f"dataset root: {root}")

    recorded = (Path(argv[2]).expanduser() if len(argv) > 2 and argv[2]
                else settings_file(root, SETTINGS_CANDIDATES,
                                   what="the measure run"))
    print(f"settings:     {recorded}")

    work = scratch("measure_on_plate1")
    stage(root, [f"merged/plate1_E01_{name}.npy" for name in FIELDS], work)

    settings = read_settings(recorded)
    settings["src"] = str(work / "merged")
    settings["n_jobs"] = 4
    settings["plot"] = False
    stale = undeclared(settings, "measure")
    if stale:
        print(f"settings this spaCR no longer declares, so nothing reads them: "
              f"{stale}")
    preflight(settings, "measure")

    from spacr.measure import measure_crop

    measure_crop(settings)

    print("\ncomparison with the reference database:")
    measured_db = work / "measurements" / "measurements.db"
    reference_db = root / "measurements" / "measurements.db"
    agreed = compare_with_reference(measured_db, reference_db, FIELDS)

    print("\nparent cell of every child object, re-derived from the masks:")
    for database, name in ((measured_db, "this run"),
                           (reference_db, "the reference")):
        for table, (right, total) in check_parent_assignment(
                database, work / "merged", FIELDS).items():
            print(f"  {name}: {table} cell_id matches the majority-overlap "
                  f"parent for {right}/{total}")

    print("\nreproduces the reference" if agreed
          else "\nDIFFERS from the reference; every difference is named above")


if __name__ == "__main__":
    run(main)
