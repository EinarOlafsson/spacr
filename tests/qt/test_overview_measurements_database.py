"""Raw overview measurements publish atomically beside other spaCR tables."""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.qc_field_browser import (
    _OverviewField,
    _overview_measurement,
    _overview_measurements_export,
)

pytestmark = pytest.mark.qt


def _field(tmp_path, name="one", pixels=None):
    path = tmp_path / (name + ".npy")
    if pixels is None:
        pixels = np.arange(24, dtype=np.uint16).reshape(4, 6)
    np.save(path, pixels)
    return _OverviewField("plate1", "A01", "A", 1, name, (("", str(path)),))


def _rows(path):
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute("SELECT * FROM fov")]


def test_raw_whole_image_geometry_and_finite_intensity_statistics(tmp_path):
    pixels = np.arange(24, dtype=float).reshape(4, 6)
    pixels[0, 0] = np.nan
    field = _field(tmp_path, pixels=pixels)
    row = _overview_measurement(field, (0,))
    assert row["status"] == "ok"
    assert row["roi_area"] == 24
    assert row["roi_bbox_2"] == 4
    assert row["roi_bbox_3"] == 6
    assert row["channel_0_finite_pixel_count"] == 23
    assert row["channel_0_nonfinite_pixel_count"] == 1
    assert row["channel_0_intensity_mean"] == 12
    assert row["channel_0_intensity_sum"] == 276
    assert row["channel_0_intensity_median"] == 12
    assert row["channel_0_intensity_p95"] == pytest.approx(21.9)
    assert row["intensity_units"] == "raw_stored_pixel_values"


def test_export_preserves_other_tables_and_replaces_field_without_duplicates(tmp_path):
    field = _field(tmp_path)
    path = tmp_path / "measurements" / "measurements.db"
    path.parent.mkdir()
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE cell (area INTEGER)")
        connection.execute("INSERT INTO cell VALUES (42)")
    calls = []
    report = _overview_measurements_export((field,), tmp_path, (0,),
        annotations={("plate1", "A01"): {"condition": "control", "drug": "none"}},
        progress=lambda done, total: calls.append((done, total)))
    assert report["count"] == 1
    assert report["path"] == str(path)
    assert not report["errors"]
    assert calls == [(1, 1)]
    first = _rows(path)[0]
    assert json.loads(first["annotations"])["condition"] == "control"
    assert first["condition"] == "control"
    assert first["plateID"] == "plate1"
    assert first["intensity_policy"] == "finite_pixels_only"
    np.save(field.files[0][1], np.full((4, 6), 100, dtype=np.uint16))
    _overview_measurements_export((field,), tmp_path, (0,))
    rows = _rows(path)
    assert len(rows) == 1
    assert rows[0]["field_uid"] == first["field_uid"]
    assert rows[0]["channel_0_intensity_mean"] == 100
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT * FROM cell").fetchall() == [(42,)]


def test_failed_run_does_not_publish_successful_siblings_and_retains_diagnostics(tmp_path):
    first = _field(tmp_path)
    previous = _overview_measurements_export((first,), tmp_path, (0,))
    before = _rows(previous["path"])
    good = _field(tmp_path, "two")
    missing = replace(first, field="missing", files=(("", str(tmp_path / "missing.npy")),))
    report = _overview_measurements_export((good, missing), tmp_path, (0,))
    assert report["count"] == 0
    assert report["attempted"] == 2
    assert report["errors"][0]["field"] == "missing"
    assert report["diagnostics"][0]["source_files"] == missing.files
    assert _rows(previous["path"]) == before


def test_cancelled_run_preserves_prior_measurements(tmp_path):
    first = _field(tmp_path)
    previous = _overview_measurements_export((first,), tmp_path, (0,))
    before = _rows(previous["path"])
    event = threading.Event()
    report = _overview_measurements_export((_field(tmp_path, "two"),), tmp_path, (0,),
        cancelled=event, progress=lambda done, total: event.set())
    assert report["cancelled"]
    assert report["count"] == 0
    assert _rows(previous["path"]) == before


def test_parallel_images_and_independent_time_z_identities(tmp_path):
    field = _field(tmp_path)
    fields = (replace(field, time_id="0", z_id="0"), replace(field, time_id="1", z_id="0"),
              replace(field, time_id="0", z_id="1"))
    report = _overview_measurements_export(fields, tmp_path, (0,), workers=2)
    assert report["count"] == 3
    rows = _rows(report["path"])
    assert len({row["field_uid"] for row in rows}) == 3
    assert {(row["timeID"], row["sliceID"]) for row in rows} == {
        ("t0", "0"), ("t1", "0"), ("t0", "1")}


def test_invalid_channel_selection_and_missing_channel_do_not_create_database(tmp_path):
    field = _field(tmp_path)
    with pytest.raises(ValueError, match="channel"):
        _overview_measurements_export((field,), tmp_path, ())
    report = _overview_measurements_export((field,), tmp_path, (1,))
    assert report["count"] == 0
    assert report["errors"]
    assert not (tmp_path / "measurements" / "measurements.db").exists()


def test_cancellation_during_database_publication_rolls_back_all_rows(tmp_path, monkeypatch):
    from spacr import database_concurrency

    first = _field(tmp_path)
    previous = _overview_measurements_export((first,), tmp_path, (0,))
    before = _rows(previous["path"])
    event = threading.Event()
    original_connect = database_concurrency.connect

    class CancelAfterInsert:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def execute(self, sql, *args):
            result = self.connection.execute(sql, *args)
            if sql.startswith("INSERT INTO fov"):
                event.set()
            return result

    monkeypatch.setattr(database_concurrency, "connect",
        lambda *args, **kwargs: CancelAfterInsert(original_connect(*args, **kwargs)))
    report = _overview_measurements_export(
        (_field(tmp_path, "two"), _field(tmp_path, "three")), tmp_path, (0,), cancelled=event)
    assert report["cancelled"]
    assert report["count"] == 0
    assert _rows(previous["path"]) == before


@pytest.mark.parametrize("workers", (1, 2))
def test_every_measurement_job_uses_selected_axes_and_projection(tmp_path, workers):
    import tifffile
    from spacr.qt.widgets.qc_field_browser import _overview_index
    source = tmp_path / str(workers)
    source.mkdir()
    for field in range(5):
        values = np.empty((2, 2, 4, 6), np.uint16)
        for time in range(2):
            for z in range(2):
                values[time, z] = 1000 * field + 100 * time + 10 * z
        tifffile.imwrite(source / f"plate1_A01_F{field}.tif", values,
                         metadata={"axes": "TZYX"}, photometric="minisblack")
    fields = _overview_index(source)
    report = _overview_measurements_export(fields, source, (0,), workers=workers,
        time_index=1, z_mode="plane", z_index=1)
    assert report["count"] == 5 and not report["errors"]
    rows = _rows(report["path"])
    assert {row["channel_0_intensity_mean"] for row in rows} == {1000 * field + 110 for field in range(5)}
    assert {(row["time_index"], row["z_mode"], row["z_index"]) for row in rows} == {(1, "plane", 1)}
    first_ids = {row["field_uid"] for row in rows}
    projected = _overview_measurements_export(fields, source, (0,), workers=workers,
        time_index=1, z_mode="mean", z_index=None)
    assert projected["count"] == 5 and not projected["errors"]
    rows = _rows(report["path"])
    projected_rows = [row for row in rows if row["z_mode"] == "mean"]
    assert len(rows) == 10 and len(projected_rows) == 5
    assert not first_ids.intersection({row["field_uid"] for row in projected_rows})
    assert {row["channel_0_intensity_mean"] for row in projected_rows} == {1000 * field + 105 for field in range(5)}


def test_fov_keys_join_measure_schema_and_first_export_is_recognized_by_doctor(tmp_path):
    from spacr import schema
    from spacr.doctor import _database_table_rows, PASS
    field = replace(_field(tmp_path), plate="screen_one/plate_alpha", field="site_beta / T12 / Z2",
                    field_id="site_beta", time_id="0012", z_id="2")
    report = _overview_measurements_export((field,), tmp_path, (0,))
    assert not report["errors"]
    row, = _rows(report["path"])
    assert row["rowID"] == "r1" and row["columnID"] == "c1"
    assert row["fieldID"] == schema.field_id("site_beta") and row["timeID"] == "t12"
    assert row["prc"] == schema.compose_prc(field.plate, 1, 1)
    assert row["prcf"] == schema.compose_prcf(field.plate, 1, 1, "site_beta", "0012")
    identity = schema.parse_prcf(row["prcf"])
    assert (identity.plateID, identity.rowID, identity.columnID, identity.fieldID, identity.timeID) == (
        field.plate, "r1", "c1", row["fieldID"], "t12")
    assert "fov" in schema.MEASUREMENT_TABLES and "fov" not in schema.OBJECT_TABLES
    assert schema.table_key_columns("fov") == ("field_uid",)
    assert schema.table_key_columns("fov", timelapse=True) == ("field_uid",)
    from pathlib import Path
    assert _database_table_rows(Path(report["path"]))[0].status == PASS


def test_projection_identity_ignores_unused_plane_and_static_identity_ignores_absent_axes(tmp_path):
    import tifffile
    from spacr.qt.widgets.qc_field_browser import _overview_index
    source = tmp_path / "projection"
    source.mkdir()
    tifffile.imwrite(source / "plate1_A01_F1.tif", np.arange(2 * 4 * 6, dtype=np.uint16).reshape(2, 4, 6),
                     metadata={"axes": "ZYX"}, photometric="minisblack")
    field, = _overview_index(source)
    first = _overview_measurements_export((field,), source, (0,), z_mode="max", z_index=0)
    uid = _rows(first["path"])[0]["field_uid"]
    second = _overview_measurements_export((field,), source, (0,), z_mode="max", z_index=1)
    row, = _rows(second["path"])
    assert row["field_uid"] == uid and row["z_index"] is None
    static = _field(tmp_path)
    first = _overview_measurements_export((static,), tmp_path, (0,))
    uid = _rows(first["path"])[0]["field_uid"]
    second = _overview_measurements_export((static,), tmp_path, (0,), time_index=0, z_index=0)
    row, = _rows(second["path"])
    assert row["field_uid"] == uid and row["time_index"] is None and row["z_index"] is None


@pytest.mark.parametrize("workers", (1, 2))
def test_each_selected_field_keeps_its_own_internal_axes_in_every_worker(tmp_path, workers):
    fields, selections = [], {}
    for index in range(5):
        raw = np.zeros((2, 2, 4, 6, 1), np.float32)
        for time in range(2):
            for z in range(2):
                raw[time, z, ..., 0] = index * 1000 + time * 100 + z * 10
        field = _field(tmp_path, f"field_{index}", raw)
        fields.append(field)
        selections[repr(field.identity)] = dict(time_index=index % 2, z_mode="plane", z_index=(index + 1) % 2)
    report = _overview_measurements_export(fields, tmp_path, (0,), workers=workers,
        time_index=0, z_index=0, field_options=selections)
    assert not report["errors"] and report["count"] == 5
    rows = sorted(_rows(report["path"]), key=lambda row: row["field"])
    for index, row in enumerate(rows):
        assert row["time_index"] == index % 2
        assert row["z_index"] == (index + 1) % 2
        assert row["channel_0_intensity_mean"] == index * 1000 + (index % 2) * 100 + ((index + 1) % 2) * 10


def test_existing_legacy_fov_table_keeps_rows_while_new_export_adds_stable_keys(tmp_path):
    path = tmp_path / "measurements" / "measurements.db"
    path.parent.mkdir()
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE fov (field TEXT, legacy_value INTEGER)")
        connection.execute("INSERT INTO fov VALUES ('legacy', 42)")
    report = _overview_measurements_export((_field(tmp_path),), tmp_path, (0,))
    assert not report["errors"] and report["count"] == 1
    rows = _rows(path)
    assert len(rows) == 2
    legacy, = (row for row in rows if row["field"] == "legacy")
    assert legacy["legacy_value"] == 42 and legacy["field_uid"] == "legacy-rowid-1"
    fresh, = (row for row in rows if row["field"] == "one")
    assert len(fresh["field_uid"]) == 64 and fresh["channel_0_intensity_mean"] == 11.5


def test_empty_measurement_run_does_not_create_an_empty_database(tmp_path):
    report = _overview_measurements_export((), tmp_path, (0,))
    assert report["attempted"] == report["count"] == 0 and not report["errors"]
    assert not (tmp_path / "measurements").exists()


def test_parallel_cancel_discards_partial_rows_and_retires_pending_image_jobs(tmp_path):
    field = _field(tmp_path)
    previous = _overview_measurements_export((field,), tmp_path, (0,))
    before = _rows(previous["path"])
    fields = tuple(_field(tmp_path, f"pending{index}") for index in range(5))
    stopped = threading.Event()
    report = _overview_measurements_export(fields, tmp_path, (0,), workers=2,
        cancelled=stopped, progress=lambda done, total: stopped.set())
    assert report["cancelled"] and report["count"] == 0 and report["attempted"] >= 1
    assert _rows(previous["path"]) == before


def test_parallel_export_with_fewer_images_than_workers_keeps_actual_row_count(tmp_path):
    field = _field(tmp_path)
    report = _overview_measurements_export((field,), tmp_path, (0,), workers=2)
    assert report["count"] == report["attempted"] == 1
    assert not report["errors"]
    assert len(_rows(report["path"])) == 1


def test_worker_exception_preserves_database_and_records_source_diagnostic(tmp_path, monkeypatch):
    from concurrent import futures
    from spacr.qt.widgets import qc_field_browser as browser
    field = _field(tmp_path)
    before = _overview_measurements_export((field,), tmp_path, (0,))
    rows = _rows(before["path"])
    def executor(*, max_workers, mp_context):
        return futures.ThreadPoolExecutor(max_workers=max_workers)
    def failed(*args, **kwargs):
        raise MemoryError("controlled worker allocation refusal")
    monkeypatch.setattr(futures, "ProcessPoolExecutor", executor)
    monkeypatch.setattr(browser, "_overview_measurement", failed)
    report = _overview_measurements_export((field,), tmp_path, (0,), workers=2)
    assert report["count"] == 0 and report["attempted"] == 1
    assert "allocation refusal" in report["errors"][0]["error"]
    assert report["diagnostics"][0]["source_files"] == field.files
    assert _rows(before["path"]) == rows
