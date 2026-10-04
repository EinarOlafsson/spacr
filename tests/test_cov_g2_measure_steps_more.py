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
