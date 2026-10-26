"""Overview measurements honour the selection and run on Measure's pool."""
from __future__ import annotations

import json
import multiprocessing
import sqlite3
import threading

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import qc_field_browser as browser
from spacr.qt.widgets.qc_field_browser import (
    _OverviewField,
    _overview_measurement,
    _overview_measurements_export,
)

pytestmark = pytest.mark.qt

_VOLATILE = {"measured_at"}


@pytest.fixture(autouse=True)
def _unpaced_workers(monkeypatch):
    monkeypatch.setattr(browser, "_OVERVIEW_WORKER_START_DELAY", 0.0)


def _fields(folder, count=4):
    rng = np.random.default_rng(3)
    fields = []
    for index in range(count):
        pixels = rng.poisson(20, (48, 64)).astype(np.uint16)
        pixels[10:30, 10 + index:40 + index] += 300
        path = folder / f"field{index}.npy"
        np.save(path, pixels)
        fields.append(_OverviewField("plate1", f"A0{index + 1}", "A", index + 1,
                                     str(index), (("", str(path)),)))
    return tuple(fields)


def _rows(path):
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute("SELECT * FROM fov")]
    return {row["field_uid"]: {key: value for key, value in row.items() if key not in _VOLATILE}
            for row in rows}


class _RecordingPool:
    """Wrap the real overview pool and record what each task carries."""

    def __init__(self, pool, log):
        self._pool, self._log = pool, log
        self._outstanding = 0

    def __enter__(self):
        self._pool.__enter__()
        return self

    def __exit__(self, *args):
        return self._pool.__exit__(*args)

    def close(self):
        self._log["closed"] = True
        return self._pool.close()

    def join(self):
        self._log["joined"] = self._log.get("closed", False)
        return self._pool.join()

    def apply_async(self, function, args=(), kwds=None):
        self._outstanding += 1
        self._log["tasks"].append((function.__name__, args[0]))
        self._log["peak"] = max(self._log["peak"], self._outstanding)
        result = self._pool.apply_async(function, args, kwds)
        owner = self

        class _Result:
            def ready(self):
                return result.ready()

            def get(self, timeout=None):
                owner._outstanding -= 1
                return result.get(timeout)

        return _Result()


def test_measurement_selection_is_honoured_in_the_image_row(tmp_path):
    field = _fields(tmp_path, 1)[0]
    selection = dict(overview_measure_intensity=False, overview_measure_texture=False,
                     overview_measure_objects=False)
    row = _overview_measurement(field, (0,), measurements=selection)
    assert row["status"] == "ok"
    assert "channel_0_intensity_mean" not in row and "channel_0_intensity_p95" not in row
    assert not any("texture_glcm" in key or "objects_count" in key for key in row)
    assert "object_source" not in row
    assert row["channel_0_focus_brenner"] > 0
    assert row["channel_0_finite_pixel_count"] == 48 * 64
    assert row["image_features"] == "focus,background,cross_channel,distribution,radial,granularity,radial_texture"
    assert not any(key.endswith("_saturation_fraction") for key in row)
    full = _overview_measurement(field, (0,))
    assert "channel_0_intensity_mean" in full and "channel_0_texture_glcm_energy" in full


def test_pool_runs_one_image_per_task_and_matches_the_serial_path(tmp_path, monkeypatch):
    fields = _fields(tmp_path)
    selection = dict(overview_measure_texture=False, overview_measure_filter_bank=True)
    serial = _overview_measurements_export(fields, tmp_path / "serial", (0,), workers=1,
                                           measurements=selection)
    log = dict(tasks=[], peak=0)
    real = browser._overview_measurement_pool
    monkeypatch.setattr(browser, "_overview_measurement_pool",
                        lambda workers, cancelled=None: _RecordingPool(real(workers, cancelled), log))
    progress = []
    parallel = _overview_measurements_export(fields, tmp_path / "parallel", (0,), workers=2,
                                             measurements=selection,
                                             progress=lambda done, total: progress.append(done))
    assert serial["count"] == parallel["count"] == 4 and not parallel["errors"]
    assert parallel["workers"] == 2
    assert [name for name, _ in log["tasks"]] == ["_overview_measurement"] * 4
    assert sorted(field.field for _, field in log["tasks"]) == ["0", "1", "2", "3"]
    assert log["peak"] <= 2
    assert progress == [1, 2, 3, 4]
    assert log.get("closed") and log.get("joined")
    serial_rows, parallel_rows = _rows(serial["path"]), _rows(parallel["path"])
    assert len(parallel_rows) == 4
    for uid, row in serial_rows.items():
        twin = parallel_rows[[key for key, value in parallel_rows.items()
                              if value["well"] == row["well"]][0]]
        assert set(row) - {"field_uid", "source"} == set(twin) - {"field_uid", "source"}
        for key, value in row.items():
            if key in ("field_uid", "source"):
                continue
            assert twin[key] == value or (isinstance(value, float) and np.isnan(value)
                                          and np.isnan(twin[key])), key
        assert row["filter_bank_status"] == "ok"
        assert "channel_0_texture_glcm_contrast" not in row


def test_cancelling_a_parallel_run_stops_its_workers_and_publishes_nothing(tmp_path):
    fields = _fields(tmp_path, 6)
    stopped = threading.Event()
    report = _overview_measurements_export(fields, tmp_path, (0,), workers=2, cancelled=stopped,
                                           progress=lambda done, total: stopped.set())
    assert report["cancelled"] and report["count"] == 0
    assert 1 <= report["attempted"] < 6
    assert not (tmp_path / "measurements" / "measurements.db").exists()
    assert not [child for child in multiprocessing.active_children()
                if child.is_alive() and "SpawnPoolWorker" in child.name]


def test_cancelling_before_workers_start_publishes_nothing(tmp_path):
    stopped = threading.Event()
    stopped.set()
    report = _overview_measurements_export(_fields(tmp_path, 3), tmp_path, (0,), workers=2,
                                           cancelled=stopped)
    assert report["cancelled"] and report["attempted"] == 0
    assert not (tmp_path / "measurements").exists()


def test_worker_count_follows_the_settings_and_the_image_count(tmp_path):
    fields = _fields(tmp_path, 2)
    report = _overview_measurements_export(fields, tmp_path, (0,), workers=None,
                                           measurements=dict(overview_measure_n_jobs=1))
    assert report["workers"] == 1 and report["count"] == 2
    report = _overview_measurements_export(fields, tmp_path / "many", (0,), workers=8)
    assert report["workers"] <= 2


def test_advanced_embedding_unavailable_is_recorded_per_image(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACR_MODEL_DOWNLOAD", "0")
    fields = _fields(tmp_path, 2)
    report = _overview_measurements_export(fields, tmp_path, (0,), measurements=dict(
        overview_measure_embeddings=True, overview_embedding_backbone="no_such_backbone_xyz"))
    assert report["count"] == 2 and "embedding" in report["advanced_unavailable"]
    rows = list(_rows(report["path"]).values())
    assert all(row["embedding_status"].startswith("unavailable") for row in rows)
    assert all(row["advanced_features"] == "embedding" for row in rows)
    assert json.loads(rows[0]["channels"]) == [0]


def test_a_lost_worker_is_recorded_against_its_image(tmp_path, monkeypatch):
    class _LostResult:
        def ready(self):
            raise RuntimeError("Processing worker 7 exited with code -9")

    class _LostPool:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def close(self):
            pass

        def join(self):
            pass

        def apply_async(self, function, args=(), kwds=None):
            return _LostResult()

    monkeypatch.setattr(browser, "_overview_measurement_pool", lambda workers, cancelled=None: _LostPool())
    missing = _OverviewField("plate1", "B01", "B", 1, "9", (("", str(tmp_path / "gone.npy")),))
    report = _overview_measurements_export((missing,) + _fields(tmp_path, 1), tmp_path, (0,), workers=2)
    assert report["count"] == 0 and report["attempted"] == 2
    assert all("exited with code -9" in error["error"] for error in report["errors"])
    assert not (tmp_path / "measurements").exists()


def test_serial_measurement_runs_advanced_switches_offline_and_stops_before_them(tmp_path):
    field = _fields(tmp_path, 1)[0]
    row = _overview_measurement(field, (0,), measurements=dict(overview_measure_filter_bank=True))
    assert row["status"] == "ok" and row["filter_bank_status"] == "ok"
    assert row["advanced_features"] == "filter_bank"

    class _StopsAfterReading:
        calls = 0

        def is_set(self):
            self.calls += 1
            return self.calls > 1

    from concurrent.futures import CancelledError
    with pytest.raises(CancelledError):
        _overview_measurement(field, (0,), _StopsAfterReading())
