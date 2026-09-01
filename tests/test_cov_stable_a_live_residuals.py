"""Driven residual paths for instruction 288 stable group A."""

from __future__ import annotations

import json
import multiprocessing
import subprocess
import sys
import types


def test_checkout_version_fallback_succeeds_and_can_be_unavailable(monkeypatch):
    """Version provenance is useful when present and harmless when absent."""
    import spacr._version as checkout
    import spacr.version as installed
    from spacr import ome_zarr

    monkeypatch.setattr(installed, "__version__", "unknown")
    monkeypatch.setattr(checkout, "__version__", "7.6.5-checkout")
    assert ome_zarr._spacr_version() == "7.6.5-checkout"

    monkeypatch.setitem(sys.modules, "spacr._version", None)
    assert ome_zarr._spacr_version() == "unknown"


def test_contained_trial_result_is_registered_only_by_the_main_process(
    tmp_path, monkeypatch,
):
    """A pool worker preserves transport stamps for its parent to consume."""
    from spacr import parameter_sweep as sweep

    result_path = tmp_path / "_trial_result.json"
    result = {"status": "ok", "_resource_worker": {"pid": 123}}
    result_path.write_text(json.dumps(result), encoding="utf-8")
    monkeypatch.setattr(sweep, "containment_available", lambda: False)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(returncode=0, stderr=""),
    )

    registered = []

    def register(row):
        registered.append(row)
        return {"status": "registered"}

    monkeypatch.setattr(sweep, "_register_resource_workers", register)
    monkeypatch.setattr(
        multiprocessing,
        "current_process",
        lambda: types.SimpleNamespace(name="SpawnPoolWorker-1"),
    )
    worker_row = sweep.run_trial_contained(
        {"src": str(tmp_path)}, trial_id=1,
    )
    assert worker_row == result
    assert registered == []

    monkeypatch.setattr(
        multiprocessing,
        "current_process",
        lambda: types.SimpleNamespace(name="MainProcess"),
    )
    main_row = sweep.run_trial_contained(
        {"src": str(tmp_path)}, trial_id=2,
    )
    assert main_row == {"status": "registered"}
    assert registered == [result]


def test_parallel_sweep_refills_after_each_completed_future(tmp_path,
                                                            monkeypatch):
    """The one-completion loop processes trials added by later refills."""
    from concurrent import futures as futures_module

    from spacr import parameter_sweep as sweep

    trials = [{"trial_id": trial_id} for trial_id in (1, 2, 3)]
    monkeypatch.setattr(sweep, "build_trials", lambda *args, **kwargs: trials)
    monkeypatch.setattr(
        sweep, "recommended_workers", lambda **kwargs: (2, "test budget"))
    monkeypatch.setattr(sweep, "memory_is_low", lambda: False)
    monkeypatch.setattr(sweep, "_pin_threads", lambda: None)
    monkeypatch.setattr(
        multiprocessing, "get_context", lambda method: f"{method}-context")

    submitted = []

    class FakeFuture:
        def __init__(self, trial_id):
            self.trial_id = trial_id

        def result(self):
            return {"trial_id": self.trial_id, "status": "ok"}

    class FakeExecutor:
        def __init__(self, max_workers, mp_context):
            assert max_workers == 2
            assert mp_context == "spawn-context"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def submit(self, function, payload):
            assert function is sweep._execute_trial
            trial_id = payload[1]["trial_id"]
            submitted.append(trial_id)
            return FakeFuture(trial_id)

    monkeypatch.setattr(futures_module, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(futures_module, "as_completed", lambda snapshot: iter(snapshot))

    rows = sweep.run_sweep_parallel(
        {}, tmp_path, n_jobs=2, progress_every=0,
    )

    assert submitted == [1, 2, 3]
    assert rows["trial_id"].tolist() == [1, 2, 3]
    assert rows["status"].tolist() == ["ok", "ok", "ok"]
