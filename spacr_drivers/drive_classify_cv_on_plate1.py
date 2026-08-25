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
downstream of a vision model is possible. This driver REFUSES on any such
row, because that failure is otherwise invisible: the run completes and
writes a full table of scores that cannot be joined to anything.

AND A CLASSIFIER HAS TO BEAT COUNTING. plate1's crops are 518 of one
phenotype and 35 of the other, so always answering "single_nucleus" scores
94% -- a number that reads like a working model. The driver prints that
majority-class rate beside the run so it is on the page, and refuses a model
whose predictions collapse to a single class, which is what a model that
learnt nothing actually produces.

A ONE-WELL PROJECT CANNOT TRAIN, and should not. The leakage-safe split
refuses when a class lives in a single group, which is the right answer; the
grouping here is by field.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _support import (cap_gpu, check, dataset_root, preflight, require, run,
                      scratch, stage)

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


def scored_rows(database):
    """The table an inference run wrote its scores into, or None.

    :returns: ``(table_name, frame)`` for the first of ``inference`` and
        ``png_list`` that carries well identifiers, else None.
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
            if any(c in rows.columns for c in
                   ("plateID", "rowID", "columnID", "fieldID", "prc",
                    "prcfo")):
                return name, rows
    return None


def unresolved_wells(rows):
    """Rows whose plate/row/column/field could not be parsed from the crop name.

    :param rows: the scored table, as returned by :func:`scored_rows`.
    :returns: ``(unresolved, total)``.
    """
    columns = [c for c in ("plateID", "rowID", "columnID", "fieldID", "prc",
                           "prcfo") if c in rows.columns]
    if not columns:
        return 0, len(rows)
    broken = rows[columns].astype(str).apply(
        lambda column: column.str.contains("error")).any(axis=1)
    return int(broken.sum()), len(rows)


def predicted_classes(rows):
    """How many distinct classes the model actually predicted, and their split.

    A model that learnt nothing answers with one class for every crop, and
    on an imbalanced project that scores as well as the majority-class rate.
    Counting the classes it emitted is what tells the two apart without a
    held-out label.

    :returns: ``{class: rows}``, empty when the table carries no prediction.
    """
    for column in ("cv_predictions", "pred_class", "predicted_class"):
        if column in rows.columns:
            return rows[column].value_counts().to_dict()
    return {}


def majority_rate(counts):
    """The accuracy of always answering with the commonest class."""
    total = sum(counts.values())
    return max(counts.values()) / total if total else 0.0


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

    scored = scored_rows(database)
    check(scored is not None,
          "the run wrote no table carrying well identifiers, so nothing was "
          "scored back into the project")
    table, rows = scored
    broken, total = unresolved_wells(rows)
    print(f"\nscored rows in {table}: {total}; wells that could not be "
          f"resolved: {broken}")
    check(broken == 0,
          f"{broken} of {total} scored rows carry 'error' in place of their "
          f"plate/row/column/field. A score that cannot name its well cannot "
          f"be joined to anything downstream, and the run still reports as "
          f"complete.")

    baseline = majority_rate(counts)
    predicted = predicted_classes(rows)
    print(f"majority-class rate on this project: {baseline:.1%} "
          f"({counts}) -- a model has to beat that to be worth anything")
    print(f"classes the model actually predicted: {predicted or 'none recorded'}")
    if predicted:
        check(len(predicted) > 1,
              f"every one of the {total} scored crops was given the same "
              f"class. That scores {baseline:.1%} on this project by "
              f"answering without looking, which is what a model that learnt "
              f"nothing produces.")


if __name__ == "__main__":
    run(main)
