"""Drive classify's computer-vision half on a real annotated project.

plate1's exported crops are already sorted into phenotype folders, so writing
the folder each crop came from into ``png_list.annotate`` turns the project
into a genuine two-class problem -- and one with enough independent fields
that a leakage-safe split is possible. That is the path this driver takes:
dataset generation FROM ANNOTATIONS rather than from metadata, training, the
model card, checkpoint selection, testing, and inference back into the
project.

INFERENCE HAS TO SAY WHERE A SCORE CAME FROM. A crop file name is
``plate_well_field_time_object`` -- five parts -- and a field name is three or
four. Parsing a crop with the field parser gives every row plate/row/column/
field of ``error``, so the scores cannot be joined back to a well and nothing
downstream of a vision model is possible. This driver counts the rows whose
well could not be resolved, because that failure is otherwise invisible: the
run completes and writes a full table.

A ONE-WELL PROJECT CANNOT TRAIN, and should not. The leakage-safe split
refuses when a class lives in a single group, which is the right answer; the
grouping here is by field.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _support import cap_gpu, dataset_root, preflight, require, run, scratch, stage

DEFAULT_ROOT = "/home/olafsson/datasets/plate1"

REQUIRED = ("measurements/measurements.db", "data/single_nucleus",
            "data/multiple_nucleus")

#: The phenotype each exported folder holds, in class order. This is
#: `class_folder_names` -- where the crops ARE -- and not `classes`, which
#: says what they mean. Passing the list as `classes` still works (classify
#: translates a pre-split list on read) but spaCR's own pre-flight declares
#: that key a dict and refuses the list at error level.
CLASSES = ("single_nucleus", "multiple_nucleus")


def annotate_from_folders(database, classes):
    """Write the folder each crop was exported into as its class label.

    :returns: ``{class_name: rows}``, so a run against a project whose crops
        are not sorted this way says so instead of training on one class.
    """
    import sqlite3

    counts = {}
    with sqlite3.connect(database) as connection:
        for label, name in enumerate(classes, start=1):
            cursor = connection.execute(
                "UPDATE png_list SET annotate = ? WHERE png_path LIKE ?",
                (label, f"%/{name}/%"))
            counts[name] = cursor.rowcount
    return counts


def unresolved_wells(database):
    """Rows whose plate/row/column/field could not be parsed from the crop name.

    :returns: ``(unresolved, total)`` over whichever results table exists.
    """
    import sqlite3

    import pandas as pd

    with sqlite3.connect(database) as connection:
        tables = set(pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'",
            connection)["name"])
        for name in ("inference", "png_list"):
            if name not in tables:
                continue
            rows = pd.read_sql(f"SELECT * FROM {name}", connection)
            columns = [c for c in ("plateID", "rowID", "columnID", "fieldID",
                                   "prc", "prcfo") if c in rows.columns]
            if not columns:
                continue
            broken = rows[columns].astype(str).apply(
                lambda column: column.str.contains("error")).any(axis=1)
            return int(broken.sum()), len(rows)
    return 0, 0


def main(argv):
    """Stage the project, label its crops from their folders, and train."""
    root = require(dataset_root(argv, DEFAULT_ROOT), REQUIRED,
                   what="the plate1 pipeline dataset")
    print(f"dataset root: {root}")

    work = scratch("classify_cv_on_plate1")
    stage(root, ["measurements/measurements.db", "data"], work)
    database = work / "measurements" / "measurements.db"

    counts = annotate_from_folders(database, CLASSES)
    print(f"crops labelled from their export folder: {counts}")
    if min(counts.values()) == 0:
        raise SystemExit(
            f"one of {CLASSES} has no crops in this project, so there is no "
            f"two-class problem to train on.")

    on_gpu = cap_gpu()
    settings = dict(
        src=str(work), classifier_family="cv",
        image_source="load_images", cv_group_by="field",
        train=True, test=True, apply_model_to_dataset=True,
        generate_training_dataset=True, generate_full_dataset=True,
        model_type="resnet18", epochs=1, batch_size=8, n_jobs=2,
        image_size=224, train_channels=["r", "g", "b"],
        class_folder_names=list(CLASSES), annotation_column="annotate",
        val_split=0.1, test_split=0.2, balance_to_smallest=True,
        verbose=False, gpu=on_gpu)
    preflight(settings, "classify")

    from spacr.classify import classify

    classify(settings)

    broken, total = unresolved_wells(database)
    print(f"\nscored rows whose well could not be resolved: {broken} of {total}")
    if broken:
        print("  a score that cannot name its well cannot be joined to "
              "anything downstream")


if __name__ == "__main__":
    run(main)
