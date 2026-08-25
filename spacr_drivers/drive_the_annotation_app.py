"""Drive the annotation app's own path, not the screen around it.

``spacr.qt.annotate_engine`` is what the Annotate screen calls: ensure the
annotation column exists, count the rows, fetch a page, load each crop, write
annotations through the save worker, count the classes, resume where the last
session stopped, and clear the column again.

THE CROP SOURCE IS THE PART WORTH PROVING. The annotator can read the crops a
measure run exported or stream them out of the merged planes on demand, and
offering that choice is only safe if the two produce the same pixels. This
driver compares them byte for byte.

AND EVERY LABEL IS READ BACK AGAINST ITS OWN CROP. Counting how many rows
were annotated does not distinguish a session that saved correctly from one
that wrote each label onto its neighbour: the class totals are identical
either way, and so is the number of rows touched. The only thing that tells
them apart is the pair, so the driver submits a known label per crop and
compares the table row by row.

Streaming for the annotator has its own trap: ``png_list`` spells both key
fields unlike the measurement tables -- the object id is ``o2`` rather than a
number, and the file name is the crop rather than the field -- so a streamer
written against the measurement spelling asks for a merged array named after
a crop and fails on an integer conversion first.

Nothing is written to the dataset: the project is copied to scratch, and the
column the driver writes is removed again at the end of the run.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _support import (WrongAnswer, check, dataset_root, require, run, scratch,
                      stage)

DEFAULT_ROOT = "/home/olafsson/datasets/plate1"

REQUIRED = ("measurements/measurements.db", "data", "merged/*.npy")

#: A column of this driver's own, so a real annotation column is never touched.
COLUMN = "driver_annotation"

PAGE = 6


def streamed_matches_exported(project, database, limit=3):
    """Whether streamed crops are the same pixels as the exported PNGs.

    :returns: ``(matching, compared)``.
    """
    import sqlite3

    import numpy as np
    import pandas as pd

    from spacr.crops import read_crop_png, resolve_crop_source

    with sqlite3.connect(database) as connection:
        rows = pd.read_sql(f"SELECT * FROM png_list LIMIT {limit}",
                           connection).to_dict("records")
    streamed = resolve_crop_source(str(project), prefer="merged").get_many(rows)
    matching = 0
    for crop, row in zip(streamed, rows):
        tail = str(row["png_path"]).split("/data/")[-1].split("/")
        exported = read_crop_png(str(Path(project, "data", *tail)))
        matching += int(np.array_equal(crop, exported))
    return matching, len(rows)


def annotations_on_disk(database, column, paths):
    """The label the database holds for each of ``paths``.

    Read back through ``png_path`` rather than through row order, because a
    label written against the wrong crop is still a label on some row: it is
    the pairing that has to be checked, and only the key carries it.

    :returns: ``{png_path: label}``, missing keys left out.
    """
    import sqlite3

    quoted = column.replace('"', '""')
    with sqlite3.connect(database) as connection:
        placeholders = ",".join("?" * len(paths))
        rows = connection.execute(
            f'SELECT png_path, "{quoted}" FROM png_list '
            f"WHERE png_path IN ({placeholders})", list(paths)).fetchall()
    return {path: label for path, label in rows if label is not None}


def main(argv):
    """Drive every step the Annotate screen calls, on a scratch copy."""
    root = require(dataset_root(argv, DEFAULT_ROOT), REQUIRED,
                   what="the plate1 pipeline dataset")
    print(f"dataset root: {root}")

    work = scratch("annotation_app")
    stage(root, ["measurements/measurements.db", "data", "merged"], work)
    database = str(work / "measurements" / "measurements.db")

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    from spacr.qt import annotate_engine as engine

    engine.ensure_annotation_column(database, COLUMN)
    total = engine.count_rows(database)
    print(f"rows to annotate: {total}")

    page = engine.fetch_page(database, COLUMN, offset=0, page_size=PAGE)
    print(f"page fetched: {len(page)} crops")
    image = engine.load_crop_image(page[0][0], db_path=database)
    print(f"first crop: {None if image is None else (image.size, image.mode)}")

    worker = engine.SaveWorker(database, COLUMN)
    worker.start()
    written = {path: (1 if index % 2 else 2)
               for index, (path, _annotation) in enumerate(page)}
    worker.submit(written)
    deadline = time.monotonic() + 20
    counts = []
    while time.monotonic() < deadline:
        counts = engine.class_counts(database, COLUMN)
        if sum(rows for _label, rows in counts) >= len(written):
            break
        time.sleep(0.2)
    print(f"annotations read back: {counts}")

    stored = annotations_on_disk(database, COLUMN, list(written))
    misplaced = {path: (written[path], stored.get(path))
                 for path in written if stored.get(path) != written[path]}
    print(f"labels that came back on the crop they were written for: "
          f"{len(written) - len(misplaced)} of {len(written)}")

    resume = engine.find_last_annotated_offset(database, COLUMN, PAGE)
    print(f"resume offset: {resume}")

    matching, compared = streamed_matches_exported(work, database)
    print(f"streamed crops identical to the exported PNGs: "
          f"{matching} of {compared}")

    engine.clear_column(database, COLUMN)
    print(f"after clearing the column: {engine.class_counts(database, COLUMN)}")

    check(sum(rows for _label, rows in counts) >= len(written),
          f"the save worker wrote {sum(r for _l, r in counts)} annotations of "
          f"the {len(written)} it was given")
    if misplaced:
        crop, (submitted, stored_label) = next(iter(misplaced.items()))
        raise WrongAnswer(
            f"{len(misplaced)} of {len(written)} labels came back on a crop "
            f"they were not written for -- {Path(crop).name} was given "
            f"{submitted} and holds {stored_label}. The class totals are the "
            f"same either way, so only the pairing shows it.")
    check(matching == compared,
          f"{compared - matching} of {compared} streamed crops differ from "
          f"the exported PNGs, so the two crop sources are not "
          f"interchangeable and offering the choice is not safe")


if __name__ == "__main__":
    run(main)
