"""Investigate Hit joins predictions to the database the Measure module wrote.

A Measure database keeps ``prcfo`` on ``png_list`` only: its ``cell`` table
carries ``prcf`` and the integer ``object_label``. ``_read_cells`` used to ask
the measured cells for a ``prcfo`` column they never had, so crop-path
predictions from Classify or Annotate failed with ``KeyError: 'prcfo'`` on a
normal database. The ``png_list`` rows here are written by the real writer,
:func:`spacr.utils.filepaths_to_database`, and the ``cell`` table has the
Measure columns, so the fixture cannot hand the reader a key the real
database lacks.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.hit_attribution import HitAttributionError
from spacr.hit_investigation import _read_cells, investigate_hit
from spacr.utils import filepaths_to_database

WELLS = [("plate1", 1, 0.0), ("plate1", 2, 0.5)]


def _write_measure_db(root, wells=WELLS, per_well=3, *, cell_prcfo=False,
                      png_list=True, prcf=True, seed=0):
    """Write ``cell`` and ``png_list`` as Measure does; return db and crops."""
    rng = np.random.default_rng(seed)
    cells, crops = [], []
    for plate, column, fraction in wells:
        well_key = f"{plate}_r1_c{column}_f1"
        for label in range(1, per_well + 1):
            identity = int(rng.random() < fraction)
            row = {"object_label": label, "plateID": plate, "rowID": "r1",
                   "columnID": f"c{column}", "fieldID": "f1",
                   "cell_area": rng.normal(2.2 * identity, 0.55),
                   "cell_texture": rng.normal(1.4 * identity, 0.65)}
            if prcf:
                row["prcf"] = well_key
            if cell_prcfo:
                row["prcfo"] = f"{well_key}_o{label}"
            cells.append(row)
            crops.append({
                "png_path": str(root / "data" / "cell"
                                / f"{plate}_A{column:02d}_1_{label}.png"),
                "prcfo": f"{well_key}_o{label}",
                "plateID": plate, "rowID": "r1", "columnID": f"c{column}",
            })
    folder = root / "measurements"
    folder.mkdir(parents=True, exist_ok=True)
    database = folder / "measurements.db"
    with sqlite3.connect(database) as connection:
        pd.DataFrame(cells).to_sql("cell", connection, index=False)
    crops = pd.DataFrame(crops)
    if png_list:
        filepaths_to_database(list(crops["png_path"]), {"timelapse": False},
                              str(root), "cell")
    return database, crops


def _scores(count):
    return [round(0.05 + 0.9 * index / max(count - 1, 1), 6)
            for index in range(count)]


def test_the_fixture_has_the_measure_layout(tmp_path):
    """The cell table lacks prcfo and png_list carries it, as on a real run."""
    database, crops = _write_measure_db(tmp_path)
    with sqlite3.connect(database) as connection:
        cell = {row[1] for row in connection.execute('PRAGMA table_info("cell")')}
        png = pd.read_sql('SELECT prcfo, png_path FROM png_list', connection)
    assert "prcfo" not in cell and {"prcf", "object_label"} <= cell
    assert dict(zip(png["png_path"], png["prcfo"])) == dict(
        zip(crops["png_path"], crops["prcfo"]))


def test_crop_path_predictions_join_cells_that_lack_prcfo(tmp_path):
    """The reported failure: crop paths against a normal Measure database."""
    database, crops = _write_measure_db(tmp_path)
    predictions = tmp_path / "predictions.csv"
    moved = [os.path.join("/elsewhere/crops", os.path.basename(path))
             for path in crops["png_path"]]
    pd.DataFrame({"path": moved, "pred": _scores(len(crops))}).to_csv(
        predictions, index=False)

    cells = _read_cells(str(database), str(predictions), "pred", "path")

    assert len(cells) == len(crops)
    assert dict(zip(cells["prcfo"], cells["pred"])) == dict(
        zip(crops["prcfo"], _scores(len(crops))))
    assert [column for column in cells if column.startswith("png_path")] == [
        "png_path"]
    assert dict(zip(cells["prcfo"], cells["png_path"])) == dict(
        zip(crops["prcfo"], crops["png_path"]))


def test_crop_basenames_in_a_named_column_score_only_their_cells(tmp_path):
    """Explain CV's export: basenames under ``filename``, a subset of crops."""
    database, crops = _write_measure_db(tmp_path, per_well=4)
    chosen = crops.iloc[[0, 3, 5, 6]]
    predictions = tmp_path / "recorded_predictions.csv"
    pd.DataFrame({
        "filename": [os.path.basename(path) for path in chosen["png_path"]],
        "predicted_label": [1, 0, 1, 0],
    }).to_csv(predictions, index=False)

    cells = _read_cells(str(database), str(predictions), "predicted_label",
                        "filename")

    assert dict(zip(cells["prcfo"], cells["predicted_label"])) == dict(
        zip(chosen["prcfo"], [1, 0, 1, 0]))


def test_a_cell_table_that_stores_prcfo_keeps_one_crop_path(tmp_path):
    """prcfo already on the cells is used as is, without a second png_path."""
    database, crops = _write_measure_db(tmp_path, cell_prcfo=True)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"png_path": list(crops["png_path"]),
                  "pred": _scores(len(crops))}).to_csv(predictions, index=False)

    cells = _read_cells(str(database), str(predictions), "pred", "png_path")

    assert dict(zip(cells["prcfo"], cells["pred"])) == dict(
        zip(crops["prcfo"], _scores(len(crops))))
    assert [column for column in cells if column.startswith("png_path")] == [
        "png_path"]


@pytest.mark.parametrize("png_list", [True, False])
def test_prcfo_predictions_join_cells_that_lack_prcfo(tmp_path, png_list):
    """prcfo predictions need no png_list; the cell key is composed."""
    database, crops = _write_measure_db(tmp_path, png_list=png_list)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"prcfo": list(crops["prcfo"]),
                  "pred": _scores(len(crops))}).to_csv(predictions, index=False)

    cells = _read_cells(str(database), str(predictions), "pred", "path")

    assert dict(zip(cells["prcfo"], cells["pred"])) == dict(
        zip(crops["prcfo"], _scores(len(crops))))
    assert ("png_path" in cells) is png_list


def test_a_doubled_plate_prefix_in_predictions_still_joins(tmp_path):
    """``pplate1`` and ``plate1`` are one plate on every key compared."""
    database, crops = _write_measure_db(tmp_path)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"prcfo": ["p" + key for key in crops["prcfo"]],
                  "pred": _scores(len(crops))}).to_csv(predictions, index=False)

    cells = _read_cells(str(database), str(predictions), "pred", "path")

    assert set(cells["prcfo"]) == set(crops["prcfo"])


def test_crop_paths_without_png_list_are_refused_by_name(tmp_path):
    """Crop names cannot reach an object when no png_list records them."""
    database, crops = _write_measure_db(tmp_path, png_list=False)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"path": list(crops["png_path"]),
                  "pred": _scores(len(crops))}).to_csv(predictions, index=False)

    with pytest.raises(HitAttributionError, match="png_list"):
        _read_cells(str(database), str(predictions), "pred", "path")


@pytest.mark.parametrize("key", ["path", "prcfo"])
def test_predictions_from_another_experiment_are_refused(tmp_path, key):
    """Nothing matching is an error, not an empty investigation."""
    database, _crops = _write_measure_db(tmp_path)
    predictions = tmp_path / "predictions.csv"
    foreign = ("/x/plate9_A01_1_1.png" if key == "path"
               else "plate9_r1_c1_f1_o1")
    pd.DataFrame({key: [foreign], "pred": [0.5]}).to_csv(predictions,
                                                         index=False)

    with pytest.raises(HitAttributionError, match="none of the 1"):
        _read_cells(str(database), str(predictions), "pred", "path")


def test_cells_without_any_object_key_are_refused(tmp_path):
    """No prcfo and no prcf to compose it from is named, not a KeyError."""
    database, crops = _write_measure_db(tmp_path, png_list=False, prcf=False)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"prcfo": list(crops["prcfo"]),
                  "pred": _scores(len(crops))}).to_csv(predictions, index=False)

    with pytest.raises(HitAttributionError, match="prcf and object_label"):
        _read_cells(str(database), str(predictions), "pred", "path")


def test_a_measure_database_with_crop_predictions_reaches_the_gallery(tmp_path):
    """End to end: cells keyed by composed prcfo keep png_path for review."""
    wells = [(f"plate{plate}", column, fraction)
             for plate in (1, 2, 3)
             for column, fraction in enumerate((0.0, 0.25, 0.0, 0.65), 1)]
    database, crops = _write_measure_db(tmp_path, wells, per_well=12, seed=17)
    cells = pd.read_sql("SELECT * FROM cell", sqlite3.connect(database))
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({
        "path": [os.path.basename(path) for path in crops["png_path"]],
        "phenotype_score": 1 / (1 + np.exp(-cells["cell_area"].to_numpy())),
    }).to_csv(predictions, index=False)
    fractions = tmp_path / "guide_fractions.csv"
    pd.DataFrame([
        {"plateID": plate, "rowID": "r1", "columnID": f"c{column}",
         "grna": guide, "fraction": value}
        for plate, column, fraction in wells
        for guide, value in (("EAF1_1", fraction), ("NTC", 1.0 - fraction))
    ]).to_csv(fractions, index=False)
    results = tmp_path / "regression_run"
    results.mkdir()
    pd.DataFrame({"gene": ["EAF1"], "effect": [0.7]}).to_csv(
        results / "results_gene.csv", index=False)

    payload = investigate_hit({
        "db_path": str(database), "predictions_file": str(predictions),
        "guide_fractions_file": str(fractions),
        "results_folder": str(results), "target_gene": "EAF1",
        "target_guides": ["EAF1_1"], "score_column": "phenotype_score",
        "hit_feature_columns": ["cell_area", "cell_texture"],
        "hit_bootstrap": 20, "hit_permutations": 20,
        "hit_pipeline_permutations": 2, "hit_gallery_per_stratum": 3,
        "hit_store_database": False, "verbose": False,
    })

    assert len(payload["result"].cells) == len(crops)
    assert set(payload["result"].cells["prcfo"]) == set(crops["prcfo"])
    assert not payload["gallery"].empty
    assert set(payload["gallery"]["png_path"]) <= set(crops["png_path"])
