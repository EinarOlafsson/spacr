"""Named worker processes reach the run resource record without table leaks."""
from __future__ import annotations

import json
import types

import pandas as pd

from spacr import parameter_sweep, runctx, sequencing, sweep_child


def test_parallel_trial_returns_both_pool_and_contained_child_stamps(
        tmp_path, monkeypatch):
    monkeypatch.setattr(parameter_sweep, "be_polite", lambda: None)
    monkeypatch.setattr(parameter_sweep, "_pin_threads", lambda: None)
    monkeypatch.setattr(
        parameter_sweep, "_trial_settings",
        lambda *_args, **_kwargs: ({}, str(tmp_path)))
    child_stamp = {
        "pid": 1234,
        "create_time": 2.0,
        "worker_kind": "parameter_sweep_trial",
        "worker_id": "7",
    }
    monkeypatch.setattr(
        parameter_sweep, "run_trial_contained",
        lambda *_args, **_kwargs: {
            "status": "ok", "seconds": 0.1,
            "_resource_worker": dict(child_stamp),
        })

    row = parameter_sweep._execute_trial(
        ({}, {"trial_id": 7}, str(tmp_path), {}, True))

    assert [stamp["worker_kind"] for stamp in row["_resource_workers"]] == [
        "parameter_sweep_pool", "parameter_sweep_trial"]
    assert row["_resource_workers"][1] == child_stamp


def test_private_worker_stamps_are_registered_and_never_reach_the_sweep_table(
        monkeypatch):
    seen = []
    context = types.SimpleNamespace(register_worker=seen.append)
    monkeypatch.setattr(runctx, "current_run_context", lambda: context)
    row = {
        "trial_id": 3,
        "status": "ok",
        "_resource_workers": [
            {"pid": 10, "worker_kind": "pool", "worker_id": "3"},
        ],
        "_resource_worker": {
            "pid": 11, "worker_kind": "trial", "worker_id": "3",
        },
    }

    cleaned = parameter_sweep._register_resource_workers(row)

    assert cleaned == {"trial_id": 3, "status": "ok"}
    assert [stamp["pid"] for stamp in seen] == [10, 11]
    assert not any(key.startswith("_resource") for key in cleaned)


def test_the_memory_guard_can_attribute_pressure_to_spacr(monkeypatch):
    gib = 1024 ** 3
    sampler = types.SimpleNamespace(
        _summary={"last_tree_memory_bytes": 5 * gib})
    context = types.SimpleNamespace(_resource_sampler=sampler)
    monkeypatch.setattr(runctx, "current_run_context", lambda: context)

    import psutil
    monkeypatch.setattr(
        psutil, "virtual_memory",
        lambda: types.SimpleNamespace(available=100 * gib))

    assert parameter_sweep.memory_is_low(
        floor_gib=8, spacr_ceiling_gib=4) is True
    assert parameter_sweep._LAST_MEMORY_STATE == {
        "available_gib": 100.0,
        "spacr_tree_gib": 5.0,
        "machine_low": False,
        "spacr_low": True,
    }


def test_sequencing_names_its_saver_and_chunk_processes(monkeypatch):
    seen = []
    context = types.SimpleNamespace(
        register_worker=lambda kind, worker, **kwargs:
        seen.append((kind, worker, kwargs["pid"])))
    monkeypatch.setattr(runctx, "current_run_context", lambda: context)

    sequencing._label_resource_process(
        types.SimpleNamespace(pid=21), "sequencing_saver", "paired")
    sequencing._label_chunk_pool(types.SimpleNamespace(
        _pool=[types.SimpleNamespace(pid=22), types.SimpleNamespace(pid=23)]))

    assert seen == [
        ("sequencing_saver", "paired", 21),
        ("sequencing_chunk", 1, 22),
        ("sequencing_chunk", 2, 23),
    ]


def test_the_contained_child_writes_its_creation_time_stamp(
        tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    output_path = tmp_path / "result.json"
    settings_path.write_text(json.dumps({
        "settings": {}, "trial_id": 19,
    }), encoding="utf-8")

    from spacr import ml, trial_metrics
    monkeypatch.setattr(parameter_sweep, "_pin_threads", lambda: None)
    monkeypatch.setattr(
        ml, "perform_regression",
        lambda _settings: {"results": pd.DataFrame()})
    monkeypatch.setattr(trial_metrics, "summarise_trial",
                        lambda _output, _settings: {})

    assert sweep_child.main([str(settings_path), str(output_path)]) == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    stamp = result["_resource_worker"]
    assert stamp["pid"] > 0
    assert stamp["create_time"] is None or stamp["create_time"] > 0
    assert stamp["worker_kind"] == "parameter_sweep_trial"
    assert stamp["worker_id"] == "19"
