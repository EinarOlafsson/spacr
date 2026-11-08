"""Measure helpers added for shared nuclei, measured-mask records and exports.

These tests hold the small Measure helpers directly: the flat-field marker
digest, the image table of a database without one, the measured-mask record
and its failure message, the shared-nucleus assignment and its report, the
per-cell export summary lines, the calibrated cell export on datasets with
many rows, none, or no fields, and the worker thread limits without OpenCV.
"""
from __future__ import annotations

import hashlib
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

from spacr import measure


def _db(path, create=None):
    with sqlite3.connect(path) as conn:
        if create:
            conn.executescript(create)
    return str(path)


def test_a_flat_field_marker_carries_the_models_digest(tmp_path):
    model = tmp_path / "model.npz"
    model.write_bytes(b"flat-field model bytes")
    extra = measure._crop_marker_extra({"illumination_model": str(model)}, True)
    assert extra["flat_field_sha256"] == hashlib.sha256(
        b"flat-field model bytes").hexdigest()
    assert extra["mask_mode"] == "masked"
    assert "flat_field_sha256" not in measure._crop_marker_extra(
        {"illumination_model": str(model)}, False)
    assert "flat_field_sha256" not in measure._crop_marker_extra(
        {"illumination_model": str(tmp_path / "missing.npz")}, True)


def test_a_database_without_an_image_table_gives_an_empty_frame(tmp_path):
    other = _db(tmp_path / "other.db", "CREATE TABLE cell (label INTEGER);")
    assert measure._read_image_table(other).empty
    assert measure._read_image_table(str(tmp_path / "absent.db")).empty


def _merged(tmp_path):
    cell = np.zeros((16, 16), np.uint16)
    cell[2:8, 2:8] = 1
    cell[9:15, 9:15] = 2
    nucleus = np.zeros_like(cell)
    nucleus[3:6, 3:6] = 1
    intensity = np.full((16, 16), 100, np.uint16)
    path = tmp_path / "merged" / "plate_A01_1.npy"
    path.parent.mkdir()
    np.save(path, np.stack([intensity, cell, nucleus], axis=-1))
    return str(path), cell, nucleus


def test_a_measured_cytoplasm_that_differs_from_the_derived_one_is_recorded(tmp_path):
    from spacr import crops

    path, cell, nucleus = _merged(tmp_path)
    settings = {"cell_mask_dim": 1, "nucleus_mask_dim": 2}
    measured_cell = cell.copy()
    measured_cell[9:15, 9:15] = 0
    cytoplasm = np.zeros_like(cell)
    cytoplasm[2:4, 2:4] = 1
    record = measure._record_measured_masks(
        path, settings, {"cell": measured_cell, "nucleus": nucleus}, cytoplasm)
    assert record == crops._measured_masks_path(path)
    with np.load(record) as saved:
        np.testing.assert_array_equal(saved["cytoplasm"], cytoplasm)
        np.testing.assert_array_equal(saved["plane_1"], measured_cell)


def test_a_measured_mask_record_that_cannot_be_written_is_said(
        tmp_path, monkeypatch, capsys):
    from spacr import crops

    path, cell, _nucleus = _merged(tmp_path)

    def refuse(*args, **kwargs):
        raise OSError("read-only folder")

    monkeypatch.setattr(crops, "_write_measured_masks", refuse)
    changed = cell.copy()
    changed[2:8, 2:8] = 0
    assert measure._record_measured_masks(
        path, {"cell_mask_dim": 1}, {"cell": changed}) is None
    assert "could not record the measured masks of plate_A01_1.npy" in (
        capsys.readouterr().out)
    assert measure._record_measured_masks(path, {}, {"cell": cell}) is None


def test_a_nucleus_spanning_two_cells_goes_to_the_larger_overlap():
    cell = np.zeros((12, 12), np.uint16)
    cell[0:12, 0:6] = 1
    cell[0:12, 6:12] = 2
    nucleus = np.zeros_like(cell)
    nucleus[4:8, 2:7] = 1
    nucleus[10:12, 0:2] = 2
    nucleus[0:1, 0:1] = 0
    fixed, nuclei, cells = measure._assign_shared_nuclei(nucleus.copy(), cell)
    assert nuclei == {1}
    assert cells == {1, 2}
    assert (fixed[4:8, 2:6] == 1).all()
    assert (fixed[4:8, 6] == 0).all()
    assert (fixed[10:12, 0:2] == 2).all()


def test_the_shared_nucleus_report_counts_flagged_nuclei(tmp_path, capsys):
    assert measure._emit_shared_nucleus_report(str(tmp_path / "absent.db")) == 0
    no_table = _db(tmp_path / "a.db", "CREATE TABLE cell (label INTEGER);")
    assert measure._emit_shared_nucleus_report(no_table) == 0
    plain = _db(tmp_path / "b.db",
                "CREATE TABLE nucleus (label INTEGER);"
                "INSERT INTO nucleus VALUES (1);")
    assert measure._emit_shared_nucleus_report(plain) == 0
    flagged = _db(tmp_path / "c.db",
                  "CREATE TABLE nucleus (label INTEGER, nucleus_shared INTEGER);"
                  "INSERT INTO nucleus VALUES (1, 1), (2, 0), (3, 1);")

    class Log:
        def __init__(self):
            self.lines = []

        def info(self, message):
            self.lines.append(message)

    class Run:
        log = Log()

    assert measure._emit_shared_nucleus_report(flagged, Run()) == 2
    assert Run.log.lines == [
        "2 nucleus label(s) shared by two cells were given to the cell they "
        "overlap most (nucleus_shared=True)."]
    assert "2 nucleus label(s) shared" in capsys.readouterr().out
    assert measure._emit_shared_nucleus_report(flagged) == 2


def test_the_per_cell_export_says_what_it_wrote_and_skipped(
        tmp_path, monkeypatch, capsys):
    from spacr import sp_stats
    import json

    db = _db(tmp_path / "m.db")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "n_rows": 5, "n_features": 7, "cells_without_nucleus": 2,
        "cells_multi_nucleus": 1, "skipped": ["table X (no label column)"]}))
    monkeypatch.setattr(sp_stats, "export_per_cell", lambda *a, **k: {
        "summary": str(summary), "parquet": "cells.parquet", "csv": "cells.csv"})
    measure._emit_per_cell_export({}, db)
    out = capsys.readouterr().out
    assert "Per-cell export: 5 cells, 7 features, 2 without a nucleus, 1 with more than one" in out
    assert "Per-cell table: cells.parquet" in out
    assert "Per-cell table: cells.csv" in out
    assert "Per-cell export skipped table X (no label column)" in out
    summary.write_text(json.dumps({"n_rows": 3, "n_features": 4}))
    monkeypatch.setattr(sp_stats, "export_per_cell", lambda *a, **k: {
        "summary": str(summary), "parquet": "only.parquet"})
    measure._emit_per_cell_export({}, db)
    out = capsys.readouterr().out
    assert "Per-cell export: 3 cells, 4 features" in out
    assert "without a nucleus" not in out
    assert "Per-cell table: only.parquet" in out
    assert "cells.csv" not in out


def test_session_metadata_joins_the_calibration_reference_hashes(
        tmp_path, monkeypatch):
    from spacr import intensity_rescale

    reference = tmp_path / "ref.npy"
    reference.write_bytes(b"reference")
    sessions = tmp_path / "sessions.csv"
    sessions.write_bytes(b"sessions")
    monkeypatch.setattr(intensity_rescale, "_calibration_inputs",
                        lambda *a, **k: {
                            "external_beads": False,
                            "references": [{"filename": "ref.npy",
                                            "path": str(reference)}],
                            "sessions_csv": str(sessions)})
    hashes = measure._calibration_reference_hashes(
        {"src": str(tmp_path), "intensity_calibration_sessions": "x"}, [])
    assert hashes["ref.npy"] == hashlib.sha256(b"reference").hexdigest()
    assert hashes["sessions_csv"] == hashlib.sha256(b"sessions").hexdigest()
    monkeypatch.setattr(intensity_rescale, "_calibration_inputs",
                        lambda *a, **k: {
                            "external_beads": True,
                            "references": [{"filename": "ref.npy",
                                            "path": str(reference)}],
                            "sessions_csv": None})
    hashes = measure._calibration_reference_hashes(
        {"src": str(tmp_path), "intensity_calibration_beads": "x"}, [])
    assert list(hashes) == ["beads/ref.npy"]


def test_a_raw_crop_is_shown_in_the_grid_as_its_stretched_view(tmp_path):
    from spacr.crops import _CROP_FORMAT_DECLARED_RAW

    crop = np.zeros((6, 6, 3), np.uint16)
    crop[1:5, 1:5] = 3000
    crop[2:4, 2:4] = 9000
    grid = measure.save_and_add_image_to_grid(
        crop, str(tmp_path / "crop.png"), [], plot=True,
        crop_format=_CROP_FORMAT_DECLARED_RAW)
    assert len(grid) == 1
    assert grid[0].dtype == np.uint8 and grid[0].shape == (6, 6, 3)


def test_worker_thread_limits_skip_opencv_when_it_is_not_loaded(monkeypatch):
    monkeypatch.delitem(sys.modules, "cv2", raising=False)
    applied = measure._limit_worker_threads()
    assert "cv2" not in applied


class _FakeDataset:
    channels = (0,)
    pixel_size_um = (0.5, 0.5)
    record_sha256 = "abc"

    def __init__(self, fields):
        self._fields = fields

    def provenance(self):
        return {"kind": "fake"}

    def fields(self):
        return iter(self._fields)


def _patch_dataset(monkeypatch, dataset, rows_per_field):
    import contextlib
    from spacr import psf_pipeline

    @contextlib.contextmanager
    def opened(source, channels=None, cancelled=None):
        yield dataset

    def statistics(field, channels, cancelled):
        for index in range(rows_per_field[field]):
            yield {name: float(index) for name in _COLUMNS}

    monkeypatch.setattr(psf_pipeline, "_open_calibrated_cell_dataset", opened)
    monkeypatch.setattr(measure, "_calibrated_cell_statistics", statistics)


_COLUMNS = (
    'source_dataset', 'plateID', 'wellID', 'fieldID', 'timeID', 'sliceID',
    'measurement_field_id', 'label_id', 'physical_channel_id', 'area_px',
    'area_um2', 'centroid_y_px', 'centroid_x_px', 'centroid_y_um',
    'centroid_x_um', 'bbox_min_y_px', 'bbox_min_x_px', 'bbox_max_y_px',
    'bbox_max_x_px', 'bbox_min_y_um', 'bbox_min_x_um', 'bbox_max_y_um',
    'bbox_max_x_um', 'intensity_mean', 'intensity_min', 'intensity_max',
    'intensity_sum', 'intensity_std',
)


def test_a_calibrated_export_flushes_every_4096_rows(tmp_path, monkeypatch):
    source = tmp_path / "src"
    source.mkdir()
    _patch_dataset(monkeypatch, _FakeDataset(["f1", "f2"]),
                   {"f1": 4096, "f2": 0})
    result = measure.measure_calibrated_cells(
        str(source), dst=str(tmp_path / "out"))
    assert result["fields"] == 2 and result["rows"] == 4096
    table = pd.read_csv(result["csv_path"])
    assert len(table) == 4096
    assert list(table.columns) == list(_COLUMNS)


def test_a_calibrated_export_with_no_cells_writes_a_header_only_table(
        tmp_path, monkeypatch):
    source = tmp_path / "src"
    source.mkdir()
    _patch_dataset(monkeypatch, _FakeDataset(["f1"]), {"f1": 0})
    result = measure.measure_calibrated_cells(
        str(source), dst=str(tmp_path / "out"))
    assert result["rows"] == 0
    assert list(pd.read_csv(result["csv_path"]).columns) == list(_COLUMNS)


def test_a_calibrated_dataset_without_fields_is_refused(tmp_path, monkeypatch):
    source = tmp_path / "src"
    source.mkdir()
    _patch_dataset(monkeypatch, _FakeDataset([]), {})
    with pytest.raises(ValueError, match="no completed fields"):
        measure.measure_calibrated_cells(str(source), dst=str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def _field():
    labels = np.zeros((16, 16), np.int32)
    labels[1:6, 1:6] = 1
    labels[8:14, 8:14] = 2
    image = np.random.default_rng(3).gamma(2.0, 100.0, labels.shape).astype(
        np.float32)
    return labels, image


def test_a_repeated_distance_is_computed_once_and_reported_in_each_column():
    torch = pytest.importorskip("torch")
    labels, image = _field()
    table = measure._torch_homogeneity(
        labels, image, [2, 2, 4], torch.device("cpu"))
    assert list(table.columns) == [
        "homogeneity_distance_2", "homogeneity_distance_2",
        "homogeneity_distance_4"]
    np.testing.assert_array_equal(table.iloc[:, 0].to_numpy(),
                                  table.iloc[:, 1].to_numpy())


def test_a_region_with_no_co_occurring_pairs_has_no_homogeneity(monkeypatch):
    labels, image = _field()
    monkeypatch.setattr(
        measure, "graycomatrix",
        lambda image, *args, **kwargs: np.zeros((256, 256, 1, 1)))
    table = measure._calculate_homogeneity(
        labels, (image % 255).astype(np.uint8), distances=[2])
    assert table["homogeneity_distance_2"].isna().all()
    assert len(table) == 2


def test_a_cucim_table_that_falls_back_after_a_failure_is_none(monkeypatch):
    mask = np.zeros((10, 10), np.int32)
    mask[1:4, 1:4] = 2
    cupy = type("Cupy", (), {"asarray": staticmethod(np.asarray),
                             "asnumpy": staticmethod(np.asarray)})
    monkeypatch.setattr(measure, "_CUCIM_STATE",
                        {"table": (cupy, lambda *a, **k: {"label": [1]})})
    monkeypatch.setattr(measure, "_guarded",
                        lambda gpu_fn, cpu_fn=None, **kwargs: cpu_fn())
    assert measure._cucim_morphology_table(mask, ["label", "area"]) is None
