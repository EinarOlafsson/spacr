"""Cancellation and overload-retry paths of ``run_sweep_parallel``."""

from __future__ import annotations

import os

import pandas as pd
import pytest


def _install_fake_pool(monkeypatch, sweep, trials, outcome):
    """Run the sweep loop in-process; ``outcome(trial_id, attempt)`` decides."""
    from concurrent import futures as futures_module

    monkeypatch.setattr(sweep, "build_trials", lambda *args, **kwargs: trials)
    monkeypatch.setattr(
        sweep, "recommended_workers", lambda **kwargs: (2, "test budget"))
    monkeypatch.setattr(sweep, "memory_is_low", lambda: False)
    monkeypatch.setattr(sweep, "_pin_threads", lambda: None)
    submitted = []

    class FakeFuture:
        def __init__(self, trial_id, attempt):
            self.trial_id = trial_id
            self.attempt = attempt

        def result(self):
            return outcome(self.trial_id, self.attempt)

    class FakeExecutor:
        def __init__(self, max_workers, mp_context):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def submit(self, function, payload):
            assert function is sweep._execute_trial
            trial_id = payload[1]["trial_id"]
            attempt = sum(1 for seen in submitted if seen == trial_id) + 1
            submitted.append(trial_id)
            return FakeFuture(trial_id, attempt)

    monkeypatch.setattr(futures_module, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(futures_module, "as_completed",
                        lambda snapshot: iter(snapshot))
    return submitted


def test_cancelling_a_primary_trial_stops_the_sweep(tmp_path, monkeypatch):
    """A cancelled trial is not recorded as an ordinary failed row."""
    from spacr import parameter_sweep as sweep
    from spacr.cancellation import PipelineCancelled

    trials = [{"trial_id": trial_id} for trial_id in (1, 2, 3)]

    def outcome(trial_id, attempt):
        if trial_id == 1:
            raise PipelineCancelled("user pressed stop")
        return {"trial_id": trial_id, "status": "ok"}

    submitted = _install_fake_pool(monkeypatch, sweep, trials, outcome)
    with pytest.raises(PipelineCancelled, match="user pressed stop"):
        sweep.run_sweep_parallel({"ram_guard": False}, tmp_path, n_jobs=2,
                                 progress_every=0)
    assert submitted == [1, 2]
    assert not (tmp_path / "sweep_results.csv").exists()


def test_overloaded_trials_are_retried_and_keep_their_primary_files(
        tmp_path, monkeypatch):
    """Only files that exist are preserved as ``.primary`` before the retry."""
    from spacr import parameter_sweep as sweep

    trials = [{"trial_id": trial_id} for trial_id in (1, 2)]
    folder = tmp_path / "trial_0001"
    folder.mkdir()
    (folder / "error.txt").write_text("MemoryError: first try",
                                      encoding="utf-8")

    def outcome(trial_id, attempt):
        if trial_id == 1 and attempt == 1:
            return {"trial_id": 1, "status": "failed", "_overload": True,
                    "error_type": "MemoryError", "error": "first try",
                    "seconds": 4.5}
        if trial_id == 1:
            (folder / "error.txt").write_text("retry", encoding="utf-8")
            return {"trial_id": 1, "status": "ok", "_overload": False,
                    "seconds": 2.0}
        return {"trial_id": trial_id, "status": "ok"}

    submitted = _install_fake_pool(monkeypatch, sweep, trials, outcome)
    rows = sweep.run_sweep_parallel({"ram_guard": False}, tmp_path, n_jobs=2,
                                    progress_every=0)

    assert submitted == [1, 2, 1]
    assert rows["trial_id"].tolist() == [1, 2]
    assert rows["status"].tolist() == ["ok", "ok"]
    first = rows.set_index("trial_id").loc[1]
    assert bool(first["overload_retry"]) is True
    assert first["primary_error_type"] == "MemoryError"
    assert first["primary_error"] == "first try"
    assert first["primary_seconds"] == pytest.approx(4.5)
    assert "_overload" not in rows.columns
    assert (folder / "error.txt.primary").read_text(
        encoding="utf-8") == "MemoryError: first try"
    assert not os.path.exists(folder / "_trial_result.json.primary")
    saved = pd.read_csv(tmp_path / "sweep_results.csv")
    assert saved["status"].tolist() == ["ok", "ok"]


def test_cancelling_an_overload_retry_stops_the_sweep(tmp_path, monkeypatch):
    """The serial retry pass propagates cancellation instead of failing a row."""
    from spacr import parameter_sweep as sweep
    from spacr.cancellation import PipelineCancelled

    trials = [{"trial_id": 1}]

    def outcome(trial_id, attempt):
        if attempt == 1:
            return {"trial_id": 1, "status": "failed", "_overload": True,
                    "error_type": "MemoryError", "error": "busy",
                    "seconds": 1.0}
        raise PipelineCancelled("stopped during retry")

    submitted = _install_fake_pool(monkeypatch, sweep, trials, outcome)
    with pytest.raises(PipelineCancelled, match="stopped during retry"):
        sweep.run_sweep_parallel({"ram_guard": False}, tmp_path, n_jobs=2,
                                 progress_every=0)
    assert submitted == [1, 1]
    saved = pd.read_csv(tmp_path / "sweep_results.csv")
    assert saved["status"].tolist() == ["failed"]
    assert "overload_retry" not in saved.columns
