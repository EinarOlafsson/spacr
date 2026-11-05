"""A small synthetic measurements.db with the column schema of a real one.

Several tests used to need a real plate's ``measurements.db`` because what they
check depends on the shape of a database spaCR actually writes: hundreds of
feature columns per object table, ``png_list`` spelling objects ``cell_id =
'o2'``, pathogens labelled in their own mask so their ``object_label`` differs
from their cell's, and uninfected cells that have no pathogen row at all.

This builder writes exactly that, at a size that takes a fraction of a second.
The column names are not invented here: they are read from database B of
``tests/data/real_measurement_columns.tsv``, the column list of a current
release run. Only the values are synthetic.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS_FILE = Path(__file__).parent / "data" / "real_measurement_columns.tsv"
TABLES = ("cell", "nucleus", "pathogen", "cytoplasm", "png_list")


def real_columns(database="B"):
    """``{table: [column, ...]}`` in the order the real database holds them."""
    columns = {}
    for line in COLUMNS_FILE.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        source, table, column = line.split("\t")
        if source != database or table not in TABLES:
            continue
        kept = columns.setdefault(table, [])
        if column not in kept:
            kept.append(column)
    return columns


def _frame(columns, identity, labels, rng, *, cell_ids=None, prefix=""):
    n = len(labels)
    data = {}
    for column in columns:
        if column in identity:
            data[column] = [identity[column]] * n
        elif column == "object_label":
            data[column] = list(labels)
        elif column == "cell_id":
            data[column] = list(cell_ids)
        elif column == "prcf":
            data[column] = [identity["prcf"]] * n
        elif column in ("label_list_morphology", "label_list_intensity"):
            data[column] = [str([int(v)]) for v in labels]
        elif column.endswith("_area"):
            data[column] = rng.gamma(4.0, 120.0, n).round(1)
        else:
            data[column] = rng.normal(10.0, 3.0, n).round(4)
    return pd.DataFrame(data)


def build_measurements_db(directory, *, plate="plate1",
                          fields=("f1", "f9", "f10", "f11"),
                          cells_per_field=(48, 41, 52, 45), seed=0):
    """Write ``<directory>/measurements/measurements.db`` and return its path.

    One well (``r5``/``c1``) with the given fields. Every third cell is
    uninfected, the rest carry one or two pathogens whose labels count up
    through the field's pathogen mask independently of the cell labels.
    """
    rng = np.random.default_rng(seed)
    schema = real_columns()
    tables = {table: [] for table in TABLES}
    for field, n_cells in zip(fields, cells_per_field):
        file_name = f"{plate}_E01_{field[1:]}"
        prcf = f"{plate}_r5_c1_{field}"
        identity = {"plateID": plate, "rowID": "r5", "columnID": "c1",
                    "fieldID": field, "prcf": prcf, "file_name": file_name,
                    "path_name": os.path.join(str(directory), "merged",
                                              f"{file_name}.npy")}
        cells = np.arange(1, n_cells + 1)
        tables["cell"].append(_frame(schema["cell"], identity, cells, rng))
        tables["cytoplasm"].append(
            _frame(schema["cytoplasm"], identity, cells, rng))
        tables["nucleus"].append(_frame(schema["nucleus"], identity, cells,
                                        rng, cell_ids=cells))
        hosts = [int(c) for c in cells if c % 3
                 for _ in range(1 + int(c % 4 == 0))]
        pathogen_labels = np.arange(1, len(hosts) + 1) + 100
        tables["pathogen"].append(_frame(schema["pathogen"], identity,
                                         pathogen_labels, rng,
                                         cell_ids=hosts))
        tables["png_list"].append(pd.DataFrame({
            "png_path": [os.path.join(str(directory), "data", file_name,
                                      "cell_png", f"{file_name}_{c}.png")
                         for c in cells],
            "file_name": [f"{file_name}_{c}.png" for c in cells],
            "plateID": plate, "rowID": "r5", "columnID": "c1",
            "fieldID": field, "prcfo": [f"{prcf}_o{c}" for c in cells],
            "cell_id": [f"o{c}" for c in cells],
        })[schema["png_list"]])
    folder = Path(directory) / "measurements"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "measurements.db"
    with sqlite3.connect(path) as db:
        for table, parts in tables.items():
            pd.concat(parts, ignore_index=True).to_sql(table, db, index=False)
    return str(path)
