"""Measure side steps: original filenames, backends and CellProfiler edges."""
from __future__ import annotations

import sqlite3
import sys
import types

import numpy as np
import pytest

from spacr import measure as m


def test_original_filenames_need_a_database(tmp_path):
    assert m._add_original_filename_columns(str(tmp_path / "none.db"), "") == {}
    assert m._add_original_filename_columns("", "") == {}


def test_a_table_the_manifest_cannot_join_is_skipped(tmp_path, monkeypatch):
    import spacr.original_filenames as of

    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, fieldID TEXT)")
        con.execute("INSERT INTO cell VALUES ('p', 'r1', 'c1', 'f1')")
    monkeypatch.setattr(m, "_original_filename_map", lambda db, src: "map.csv")

    def unjoinable(frame, path):
        raise ValueError("no matching identity")

    monkeypatch.setattr(of, "_original_columns", unjoinable)
    assert m._add_original_filename_columns(str(db), str(tmp_path)) == {}


def test_a_postgres_target_gains_its_scheme(tmp_path):
    target = m._measurement_backend_target(
        str(tmp_path / "measurements.db"),
        {"measurement_backend": "postgres",
         "measurement_backend_target": "user@host/db"})
    assert target == "postgresql://user@host/db"


def test_a_backend_that_already_has_every_table_copies_nothing(tmp_path,
                                                              monkeypatch):
    import spacr.tabular as tabular

    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE cell (a INTEGER)")
    copied = []
    monkeypatch.setattr(tabular, "database_tables",
                        lambda path: ["cell"])
    monkeypatch.setattr(tabular, "_migrate_database",
                        lambda *a, **k: copied.append(a))
    m._copy_to_measurement_backend(str(db), {"measurement_backend": "duckdb"})
    assert copied == []


def test_cellprofiler_roles_ignore_planes_outside_the_array():
    roles = m._cellprofiler_roles({"cell_mask_dim": 9, "nucleus_mask_dim": 1}, 3)
    assert roles == {"nucleus": 1}


def test_cellprofiler_overlaps_refuse_bad_masks_and_non_arrays(tmp_path,
                                                              monkeypatch):
    plane = tmp_path / "p.npy"
    np.save(plane, np.ones((4, 4), np.int32))
    assert m._cellprofiler_overlap_labels([str(plane)],
                                          np.full((4, 4), -1)) == {}

    class _Closable:
        closed = False

        def close(self):
            _Closable.closed = True

    monkeypatch.setattr(m.np, "load", lambda *a, **k: _Closable())
    assert m._cellprofiler_overlap_labels([str(plane)],
                                          np.ones((4, 4), np.int32)) == {}
    assert _Closable.closed


def test_a_cellprofiler_run_that_imports_nothing_says_so(tmp_path, monkeypatch,
                                                         capsys):
    pipeline = tmp_path / "p.cppipe"
    pipeline.write_text("pipeline")
    db = tmp_path / "plate" / "measurements" / "measurements.db"
    db.parent.mkdir(parents=True)
    monkeypatch.setattr(m, "_cellprofiler_export", lambda *a: [])
    monkeypatch.setattr(m, "_cellprofiler_tables", lambda *a: {})
    backends = types.ModuleType("spacr._segmentation_backends")
    backends._run_cellprofiler = lambda pipeline, files, out: {}
    import spacr._segmentation_backends as real

    monkeypatch.setattr(real, "_run_cellprofiler", backends._run_cellprofiler,
                        raising=False)
    counts = m._run_cellprofiler_step(
        str(db), {"cellprofiler_pipeline": str(pipeline), "src": str(tmp_path)})
    assert counts == {}
    assert "measured no objects" in capsys.readouterr().out
    assert sys


def test_a_measure_run_reaches_barcodes_cellprofiler_and_a_backend(tmp_path,
                                                                   monkeypatch):
    from tests.test_live_dead_viability import measure_settings, write_plate

    merged, _truths = write_plate(tmp_path, {"A01": (0.0, 1.0, None)},
                                  size=64, base_cells=4)
    steps = []
    monkeypatch.setattr(m, "_run_plate_barcode_step",
                        lambda settings: steps.append("barcodes"))
    monkeypatch.setattr(m, "_run_cellprofiler_step",
                        lambda db, settings: steps.append("cellprofiler"))
    monkeypatch.setattr(m, "_copy_to_measurement_backend",
                        lambda db, settings: steps.append("backend"))
    pipeline = tmp_path / "p.cppipe"
    pipeline.write_text("x")
    m.measure_crop(measure_settings(
        merged, viability=False, n_jobs=1, plate_barcode_source="records.csv",
        cellprofiler_pipeline=str(pipeline), measurement_backend="parquet"))
    assert steps == ["barcodes", "cellprofiler", "backend"]


def test_a_gpu_morphology_table_is_used_when_it_agrees(monkeypatch):
    from tests.test_cov_11_measure import _masks, _settings

    used = []

    def gpu_table(mask, props):
        used.append(True)
        return m._safe_morphology_table(mask, properties=props)

    monkeypatch.setattr(m, "_measurement_device", lambda settings: "cuda")
    monkeypatch.setattr(m, "_gpu_measurable", lambda mask, spacing=None: True)
    monkeypatch.setattr(m, "_cucim_morphology_table", gpu_table)
    cell, nucleus, pathogen = _masks()
    frames = m._morphological_measurements(cell, nucleus, pathogen, None, None,
                                           _settings(), zernike=False)
    assert used and len(frames[0])


def test_a_log_scale_single_population_is_cut_by_its_spread():
    rng = np.random.default_rng(0)
    values = np.exp(rng.normal(0.0, 0.1, 400))
    cut = m._stain_cut(values, single_is_positive=True, log_scale=True)
    assert cut.source == "single" and cut.threshold < 1.0


def test_infection_needs_a_pathogen_count(tmp_path, monkeypatch):
    import pandas as pd
    import spacr.infection as infection

    monkeypatch.setattr(infection, "parasites_per_cell",
                        lambda db: pd.DataFrame({"cell_id": [1]}))
    nuclei = pd.DataFrame({"cell_id": [1]})
    assert m._nucleus_infection(str(tmp_path / "x.db"), nuclei).isna().all()


def test_a_postgres_url_is_kept_as_given(tmp_path):
    url = "postgresql://user@host/db"
    assert m._measurement_backend_target(
        str(tmp_path / "measurements.db"),
        {"measurement_backend": "postgres",
         "measurement_backend_target": url}) == url


def _cp_reply(tmp_path, objects, *, images=None, labels=None):
    merged = tmp_path / "merged"
    merged.mkdir(exist_ok=True)
    mask = np.zeros((4, 5), np.uint16)
    mask[1:3, 1:3] = 1
    mask2 = np.zeros((4, 5), np.uint16)
    mask2[0, 0] = 2
    np.save(merged / "plate1_A01_1.npy", np.stack([mask, mask2], axis=-1))
    blocks = {}
    for name, (columns, rows) in objects.items():
        path = tmp_path / f"{name}.npy"
        np.save(path, np.asarray(rows, dtype=float))
        blocks[name] = {"columns": columns, "path": str(path)}
    reply = {"images": images if images is not None else
             {"1": ["plate1_A01_1_ch0.tif", "notes.txt"], "2": ["other.txt"]},
             "objects": blocks}
    if labels is not None:
        reply["labels"] = labels
    return reply, merged


def test_cellprofiler_objects_without_centres_or_fields_are_handled(
        tmp_path, capsys):
    columns = ["ImageNumber", "ObjectNumber", "Location_Center_X",
               "Location_Center_Y"]
    reply, merged = _cp_reply(tmp_path, {
        "Blobs": (["ImageNumber", "ObjectNumber", "AreaShape_Area"],
                  [[1, 1, 5.0]]),
        "Things": (columns, [[1, 1, 1.5, 1.5], [2, 2, 1.0, 1.0],
                             [1, 3, 0.0, 0.0]]),
    })
    settings = {"cell_mask_dim": 0, "nucleus_mask_dim": 1,
                "pathogen_mask_dim": None, "timelapse": False}
    tables = m._cellprofiler_tables(reply, str(merged), settings)
    assert "has no Location_Center_X/Y" in capsys.readouterr().out
    assert list(tables) == ["cellprofiler_things"]


def test_cellprofiler_tied_roles_with_supplied_labels_stay_unmatched(tmp_path):
    columns = ["ImageNumber", "ObjectNumber"]
    plane = tmp_path / "cp_label.npy"
    np.save(plane, np.zeros((4, 5), np.int32))
    reply, merged = _cp_reply(
        tmp_path, {"Things": (columns, [[1, 1]])},
        labels={"1": {"Things": [str(plane)]}})
    settings = {"cell_mask_dim": 0, "nucleus_mask_dim": 1,
                "pathogen_mask_dim": None, "timelapse": False}
    tables = m._cellprofiler_tables(reply, str(merged), settings)
    frame = tables["cellprofiler_things"]
    assert frame["object_type"].isna().all() or frame["object_label"].isna().all()


def test_viability_without_qc_rows_writes_no_qc_table(tmp_path, monkeypatch):
    import pandas as pd
    from tests.test_live_dead_viability import _signal_table
    from tests.test_viability_at_its_edges import _db

    db = _db(tmp_path, _signal_table(n_live=120, n_dead=30), "nucleus")
    monkeypatch.setattr(m, "_viability_qc", lambda wells, cuts: pd.DataFrame())
    m._classify_viability(db, {"channels": [0, 1, 2], "viability_dead_channel": 1},
                          plot=False)
    with sqlite3.connect(db) as conn:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert m._VIABILITY_QC_TABLE not in names


def test_cell_cycle_without_wells_writes_no_well_table(tmp_path, monkeypatch):
    import pandas as pd
    from spacr.tabular import write_database
    from tests.test_cell_cycle_at_its_edges import _nucleus_table

    table, _ = _nucleus_table(n=300)
    db = tmp_path / "measurements" / "measurements.db"
    db.parent.mkdir()
    write_database(table, str(db), "nucleus", if_exists="replace")
    monkeypatch.setattr(m, "_cell_cycle_by_well", lambda table, counted: pd.DataFrame())
    m._classify_cell_cycle(str(db), {"channels": [0, 1], "cell_cycle_channel": 0,
                                     "cell_cycle_method": "measurements"}, plot=False)
    with sqlite3.connect(db) as conn:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert m._CELL_CYCLE_WELL_TABLE not in names
