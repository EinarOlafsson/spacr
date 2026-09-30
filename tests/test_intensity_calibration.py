"""Cross-plate intensity calibration from beads or reference wells.

Two plates image the same bead well and the same cell sample, the second at
2.5x the exposure, both over a camera offset of 100. Calibrated against the
beads, the second plate's cell intensities agree with the first's within 2%;
uncalibrated they differ 2.5-fold.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import measure
from spacr.intensity_rescale import (
    CALIBRATION_SETTINGS_KEY, _apply_calibration, _build_calibration_plan,
    _calibration_wells, _reference_statistic)

OFFSET = 100
EXPOSURES = {"plate1": 1.0, "plate2": 2.5}
TOLERANCE = 0.02
SIZE = 128


def _beads(seed):
    """Sparse bright Gaussian beads on a dark field, noiseless."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    image = np.zeros((SIZE, SIZE))
    for y, x in rng.uniform(8, SIZE - 8, size=(25, 2)):
        image += 3000 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 4.0)
    return image + 20


def _cells():
    """Four square cells of known brightness and their label plane."""
    labels = np.zeros((SIZE, SIZE), dtype=np.uint16)
    signal = np.full((SIZE, SIZE), 30.0)
    for label, (y, x, level) in enumerate(
            [(10, 10, 400), (10, 70, 800), (70, 10, 1200), (70, 70, 1600)],
            start=1):
        labels[y:y + 40, x:x + 40] = label
        signal[y:y + 40, x:x + 40] = level
    return signal, labels


def _session(signal, exposure, seed):
    """One exposure of ``signal``: offset plus Poisson photon counts."""
    rng = np.random.default_rng(seed)
    return OFFSET + rng.poisson(signal * exposure)


def _write_sessions(root):
    merged = root / "merged"
    merged.mkdir(parents=True)
    signal, labels = _cells()
    for index, (plate, exposure) in enumerate(EXPOSURES.items()):
        for field in (1, 2):
            beads = _beads(field)
            stack = np.stack([_session(beads, exposure, 10 * index + field),
                              _session(beads * 0.5, exposure, 20 + field),
                              np.zeros_like(labels)], axis=-1)
            np.save(merged / f"{plate}_A01_{field}.npy", stack.astype(np.uint16))
            stack = np.stack([_session(signal, exposure, 30 + field),
                              _session(signal * 0.5, exposure, 40 + field),
                              labels], axis=-1)
            np.save(merged / f"{plate}_B02_{field}.npy", stack.astype(np.uint16))
    return merged


def _settings(merged, **over):
    from spacr.settings import get_measure_crop_settings

    settings = get_measure_crop_settings({})
    settings.update({
        "src": str(merged), "channels": [0, 1],
        "cell_mask_dim": 2, "nucleus_mask_dim": None,
        "pathogen_mask_dim": None, "cell_min_size": 0,
        "nucleus_min_size": 0, "pathogen_min_size": 0,
        "cytoplasm_min_size": 0, "save_png": False, "save_arrays": False,
        "plot": False, "verbose": False, "n_jobs": 1,
        "intensity_calibration_wells": ["A01"],
        "intensity_calibration_offset": OFFSET,
    })
    settings.update(over)
    return settings


def _cell_means(root):
    db = root / "measurements" / "measurements.db"
    with sqlite3.connect(db) as conn:
        cells = pd.read_sql_query("SELECT * FROM cell", conn)
        provenance = pd.read_sql_query(
            "SELECT * FROM intensity_rescale", conn)
    columns = [c for c in cells.columns
               if c.startswith("cell_channel_") and c.endswith("_mean_intensity")
               and c[len("cell_channel_"):-len("_mean_intensity")] in ("0", "1")]
    assert len(columns) == 2, cells.columns.tolist()
    cells = cells[cells["file_name"].str.contains("B02")]
    return cells.groupby(["plateID", "object_label"])[columns].mean(), provenance


def _relative_gap(means):
    one = means.xs("plate1", level="plateID") - OFFSET
    two = means.xs("plate2", level="plateID") - OFFSET
    return (two / one - 1).abs().to_numpy()


def test_two_exposures_agree_after_bead_calibration(tmp_path):
    merged = _write_sessions(tmp_path / "raw")
    measure.measure_crop(_settings(merged))
    raw_means, _ = _cell_means(tmp_path / "raw")
    assert np.allclose(_relative_gap(raw_means), 1.5, atol=0.05)

    merged = _write_sessions(tmp_path / "calibrated")
    measure.measure_crop(_settings(merged, intensity_calibration=True))
    means, provenance = _cell_means(tmp_path / "calibrated")
    assert _relative_gap(means).max() <= TOLERANCE

    records = [json.loads(text) for text in provenance["intensity_calibration"]]
    assert len(records) == 8
    assert {r["reference_plate"] for r in records} == {"plate1"}
    for record in records:
        expected = 1.0 if record["plateID"] == "plate1" else 1 / 2.5
        assert record["gain"].keys() == {"0", "1"}
        for gain in record["gain"].values():
            assert gain == pytest.approx(expected, rel=TOLERANCE)
        assert record["offset"] == OFFSET
        assert record["wells"] == ["A01"]


def test_the_median_statistic_calibrates_a_uniform_reference_well(tmp_path):
    merged = _write_sessions(tmp_path)
    uniform = np.full((SIZE, SIZE), 1000.0)
    for index, (plate, exposure) in enumerate(EXPOSURES.items()):
        stack = np.stack([_session(uniform, exposure, 50 + index),
                          _session(uniform * 0.5, exposure, 60 + index),
                          np.zeros((SIZE, SIZE))], axis=-1)
        np.save(merged / f"{plate}_C03_1.npy", stack.astype(np.uint16))
    settings = _settings(merged, intensity_calibration_wells="C03",
                         intensity_calibration_statistic="median")
    plan = _build_calibration_plan(merged, sorted(p.name for p in merged.iterdir()),
                                   settings)
    assert plan["reference_plate"] == "plate1"
    assert plan["plates"]["plate1"]["n_reference_fields"] == 1
    for gain in plan["plates"]["plate2"]["gain"].values():
        assert gain == pytest.approx(1 / 2.5, rel=TOLERANCE)


def test_calibration_keeps_labels_and_the_offset_and_widens_uint8():
    data = np.zeros((4, 4, 2), dtype=np.uint8)
    data[..., 0] = [[10, 20, 30, 250]] * 4
    data[..., 1] = 7
    settings = {"cell_mask_dim": 1, CALIBRATION_SETTINGS_KEY: {
        "reference_plate": "plate1", "statistic": "median", "offset": 10.0,
        "wells": ["A01"], "plates": {"plate2": {
            "gain": {"0": 2.0}, "reference_statistic": {"0": 5.0},
            "n_reference_fields": 1},
            "plate1": {"gain": {"0": 1.0}, "reference_statistic": {"0": 10.0},
                       "n_reference_fields": 1}}}}
    out, record = _apply_calibration(data, "plate2_A01_1.npy", settings)
    assert out.dtype == np.uint16
    assert out[0, :, 0].tolist() == [10, 30, 50, 490]
    assert (out[..., 1] == 7).all()
    assert record["gain"] == {"0": 2.0}
    unchanged, none = _apply_calibration(data, "plate2_A01_1.npy", {})
    assert none is None and unchanged is data


def test_a_plate_without_reference_wells_stops_the_run(tmp_path):
    merged = _write_sessions(tmp_path)
    (merged / "plate2_A01_1.npy").unlink()
    (merged / "plate2_A01_2.npy").unlink()
    with pytest.raises(ValueError, match="plate2"):
        _build_calibration_plan(merged, sorted(p.name for p in merged.iterdir()),
                                _settings(merged))


def test_reference_wells_and_statistic_are_validated():
    assert sorted(_calibration_wells(
        {"intensity_calibration_wells": "a1, P24"})) == [("r1", "c1"),
                                                         ("r16", "c24")]
    with pytest.raises(ValueError, match="empty"):
        _calibration_wells({"intensity_calibration_wells": None})
    with pytest.raises(ValueError, match="not a well"):
        _calibration_wells({"intensity_calibration_wells": ["wt"]})
    beads = _beads(1)
    assert _reference_statistic(beads * 2.5, 0, "foreground") == pytest.approx(
        2.5 * _reference_statistic(beads, 0, "foreground"), rel=1e-9)


@pytest.fixture(scope="module")
def measured_calibration_projects(tmp_path_factory):
    """Keep genuine measured rows available for independent resume regressions."""
    projects = {}
    for enabled in (False, True):
        root = tmp_path_factory.mktemp(f"calibration-history-{enabled}")
        merged = _write_sessions(root)
        measure.measure_crop(_settings(merged, intensity_calibration=enabled))
        with sqlite3.connect(root / "measurements" / "measurements.db") as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        projects[enabled] = root
    return projects


def _copy_measured_project(projects, destination, enabled=True):
    """Copy a closed measurement project without sharing mutable SQLite state."""
    shutil.copytree(projects[enabled], destination)
    return destination / "merged", destination / "measurements" / "measurements.db"


def _assert_refused_before_database_writes(monkeypatch, settings, database):
    """Reject incompatible history before WAL, settings replacement or cleanup."""
    from spacr import database_concurrency, io

    def forbidden(*args, **kwargs):
        raise AssertionError("Database mutation was reached before calibration refusal")

    before = database.read_bytes()
    monkeypatch.setattr(database_concurrency, "enable_wal_where_safe", forbidden)
    monkeypatch.setattr(measure, "plan_measure_resume", forbidden)
    monkeypatch.setattr(io, "_save_settings_to_db", forbidden)
    with pytest.raises(ValueError, match="different or unverified intensity calibration"):
        measure.measure_crop(settings)
    assert database.read_bytes() == before


@pytest.mark.parametrize("enabled", [False, True])
def test_unchanged_calibration_and_legacy_disabled_projects_resume(
        tmp_path, monkeypatch, measured_calibration_projects, enabled):
    merged, database = _copy_measured_project(
        measured_calibration_projects, tmp_path / "resume", enabled)
    if not enabled:
        with sqlite3.connect(database) as conn:
            conn.execute("DELETE FROM settings WHERE setting_key = ?",
                         (measure._CALIBRATION_IDENTITY_KEY,))
    else:
        reference = merged / "plate2_A01_1.npy"
        previous = reference.stat()
        os.utime(reference, ns=(previous.st_atime_ns,
                               previous.st_mtime_ns + 10_000_000_000))
    before, provenance = _cell_means(merged.parent)
    plans = []
    original = measure.plan_measure_resume

    def observe(settings):
        plan = original(settings)
        plans.append(plan)
        return plan

    monkeypatch.setattr(measure, "plan_measure_resume", observe)
    measure.measure_crop(_settings(
        merged, intensity_calibration=enabled, resume=True))
    after, resumed_provenance = _cell_means(merged.parent)
    pd.testing.assert_frame_equal(before, after)
    assert len(plans) == 1
    assert {f"{plate}_B02_{field}" for plate in EXPOSURES for field in (1, 2)} <= set(plans[0].skipped)
    if enabled:
        identities = {json.loads(text)["identity"]
                      for text in provenance["intensity_calibration"]}
        assert len(identities) == 1
        assert {json.loads(text)["identity"] for text in
                resumed_provenance["intensity_calibration"]} == identities
        assert all(len(json.loads(text)["reference_files"]) == 4
                   for text in resumed_provenance["intensity_calibration"])


@pytest.mark.parametrize("change", ["gain", "same_statistics"])
@pytest.mark.parametrize("resume", [False, True])
def test_changed_reference_bytes_refuse_even_with_original_size_and_timestamp(
        tmp_path, monkeypatch, measured_calibration_projects, change, resume):
    merged, database = _copy_measured_project(
        measured_calibration_projects, tmp_path / "changed")
    reference = merged / "plate2_A01_1.npy"
    previous = reference.stat()
    data = np.load(reference)
    if change == "gain":
        data[..., 0] += 500
    else:
        data = data[::-1].copy()
    np.save(reference, data)
    os.utime(reference, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert reference.stat().st_size == previous.st_size
    assert reference.stat().st_mtime_ns == previous.st_mtime_ns
    _assert_refused_before_database_writes(monkeypatch, _settings(
        merged, intensity_calibration=True, resume=resume), database)


@pytest.mark.parametrize("resume", [False, True])
def test_disabling_calibration_cannot_mix_with_retained_calibrated_rows(
        tmp_path, monkeypatch, measured_calibration_projects, resume):
    merged, database = _copy_measured_project(
        measured_calibration_projects, tmp_path / "disabled")
    _assert_refused_before_database_writes(monkeypatch, _settings(
        merged, intensity_calibration=False, resume=resume), database)


@pytest.mark.parametrize("damage", ["legacy_identity", "missing_field_provenance"])
def test_unverified_calibrated_history_refuses_before_resume_clears_rows(
        tmp_path, monkeypatch, measured_calibration_projects, damage):
    merged, database = _copy_measured_project(
        measured_calibration_projects, tmp_path / "unverified")
    with sqlite3.connect(database) as conn:
        if damage == "legacy_identity":
            conn.execute("DELETE FROM settings WHERE setting_key = ?",
                         (measure._CALIBRATION_IDENTITY_KEY,))
            for rowid, text in conn.execute(
                    "SELECT rowid, intensity_calibration FROM intensity_rescale").fetchall():
                record = json.loads(text)
                record.pop("identity")
                record.pop("reference_files")
                conn.execute("UPDATE intensity_rescale SET intensity_calibration = ? WHERE rowid = ?",
                             (json.dumps(record), rowid))
        else:
            conn.execute("DELETE FROM intensity_rescale WHERE file_name = 'plate1_B02_1'")
            assert conn.execute("SELECT changes()").fetchone()[0] == 1
    _assert_refused_before_database_writes(monkeypatch, _settings(
        merged, intensity_calibration=True, resume=True), database)


def test_blank_well_names_are_skipped_and_an_empty_plane_has_no_statistic():
    assert sorted(_calibration_wells(
        {"intensity_calibration_wells": ["A01", "  ", ""]})) == [("r1", "c1")]
    assert np.isnan(_reference_statistic(np.full((3, 3), np.nan), 0,
                                         "foreground"))


def test_an_unknown_calibration_statistic_is_refused(tmp_path):
    merged = _write_sessions(tmp_path)
    with pytest.raises(ValueError, match="intensity_calibration_statistic"):
        _build_calibration_plan(
            merged, sorted(p.name for p in merged.iterdir()),
            _settings(merged, intensity_calibration_statistic="mode"))


def test_a_reference_well_with_no_signal_above_the_offset_stops_the_run(
        tmp_path):
    """A plate whose reference wells sit at the camera offset has no
    positive reference intensity, so no gain can be computed for it."""
    merged = _write_sessions(tmp_path)
    for field in (1, 2):
        dark = np.stack([np.full((SIZE, SIZE), OFFSET),
                         np.full((SIZE, SIZE), OFFSET),
                         np.zeros((SIZE, SIZE))], axis=-1)
        np.save(merged / f"plate2_A01_{field}.npy", dark.astype(np.uint16))
    with pytest.raises(ValueError, match="no positive reference intensity"):
        _build_calibration_plan(merged, sorted(p.name for p in merged.iterdir()),
                                _settings(merged))


def test_a_field_on_an_uncalibrated_plate_is_refused_and_float_data_is_not_clipped():
    plan = {"reference_plate": "plate1", "statistic": "median",
            "offset": 0.0, "wells": ["A01"], "plates": {
                "plate1": {"gain": {"0": 3.0, "5": 2.0},
                           "reference_statistic": {"0": 1.0, "5": 1.0},
                           "n_reference_fields": 1}}}
    settings = {"cell_mask_dim": None, CALIBRATION_SETTINGS_KEY: plan}
    with pytest.raises(ValueError, match="has no calibration gain"):
        _apply_calibration(np.zeros((2, 2, 1), np.uint16), "plate9_A01_1.npy",
                           settings)
    data = np.full((2, 2, 1), 0.5, dtype=np.float32)
    out, record = _apply_calibration(data, "plate1_A01_1.npy", settings)
    assert out.dtype == np.float32
    assert np.allclose(out[..., 0], 1.5)
    assert set(record["gain"]) == {"0", "5"}
