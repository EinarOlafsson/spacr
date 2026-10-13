"""Item 288: a FEATURES run with the optional measurement blocks switched on.

These go through :func:`spacr.measure.measure_from_field_table`, which is an
ordinary Measure run, on two small hand-drawn fields. Each pins what the
database holds afterwards:

* a bystander classification that fails, or that answers without a
  neighbourhood column, costs the bystander columns and nothing else -- the
  cell table is still written and the reason is printed;
* a table with no cell mask, measured with nucleus tracking, a cytoplasm
  request and FOV normalisation switched off, still measures its nuclei and
  its organelles, and writes no cell rows.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from spacr.measure import FieldRow, FieldTable, assign_paths_by_regex, \
    measure_from_field_table
from tests.test_features_button_measures_hand_drawn_masks import (  # noqa: F401
    LEAN, REGEX, _paths, _write, drawn)


def _columns(db, table):
    with sqlite3.connect(db) as connection:
        return {row[1] for row in connection.execute(
            f'pragma table_info("{table}")')}


def _count(db, table):
    with sqlite3.connect(db) as connection:
        return connection.execute(f'select count(*) from "{table}"').fetchone()[0]


@pytest.mark.parametrize("answer", ["raises", "no_neighbourhood"])
def test_a_bystander_block_that_cannot_answer_costs_only_its_columns(
        tmp_path, drawn, monkeypatch, capfd, answer):
    import pandas as pd

    import spacr.bystanders as bystanders

    def classify(mask, infected, reach, spacing):
        if answer == "raises":
            raise RuntimeError("no reach for this field")
        return pd.DataFrame({"label": np.unique(mask[mask > 0])})

    monkeypatch.setattr(bystanders, "classify", classify)
    found = assign_paths_by_regex(_paths(drawn), REGEX)
    settings = dict(LEAN, cytoplasm=True, bystander_measurements=True)
    result = measure_from_field_table(found.table, settings,
                                      dst=str(tmp_path / "run"))
    db = result["db_path"]
    out = capfd.readouterr().out
    said = ("bystanders were not measured: RuntimeError: no reach for this "
            "field")
    assert (said in out) == (answer == "raises")
    cell = _columns(db, "cell")
    assert not {"is_bystander", "is_distal"} & cell
    assert _count(db, "cell") == 2


def test_a_table_without_cells_measures_its_nuclei_and_organelles(
        tmp_path, drawn):
    organelle = np.zeros((48, 48), np.uint16)
    organelle[20:22, 20:22] = 1
    organelle[30:40, 30:40] = 2
    path = _write(str(tmp_path / "organelle.tif"), organelle)
    rows = []
    for index in (1, 2):
        rows.append(FieldRow(
            label=f"fov00{index}", field=index,
            channels={0: str(drawn / f"fov00{index}_C1.tif"),
                      1: str(drawn / f"fov00{index}_C2.tif")},
            masks={"nucleus": str(drawn / f"fov00{index}_nucleus_mask.tif"),
                   "organelle": path}))
    table = FieldTable(rows=rows, n_channels=2, roles=("nucleus", "organelle"))
    settings = dict(LEAN, timelapse_objects="nucleus", cytoplasm=True,
                    normalize_by="fov", normalize=False,
                    crop_mode=["nucleus"])
    result = measure_from_field_table(table, settings,
                                      dst=str(tmp_path / "run"))
    db = result["db_path"]
    assert _count(db, "nucleus") == 2
    assert _count(db, "organelle") >= 2
    with sqlite3.connect(db) as connection:
        tables = {row[0] for row in connection.execute(
            "select name from sqlite_master where type='table'")}
    assert "cell" not in tables or _count(db, "cell") == 0
