"""Headless contracts for instruction 312's process-tree accounting core."""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import threading
import time
import types

import psutil
import pytest

import spacr.fit_resources as resources
from spacr.fit_resources import (
    _PERFORMANCE_LOG_ENV,
    _atomic_json,
    _cpu_reading,
    _memory_reading,
    _performance_mode,
    _process_key,
    _process_tree_snapshot,
    _python_thread_names,
    _read_process,
    _ResourceSampler,
    _select_performance_mode,
    _thread_reading,
    _tree_measure,
    _tree_stage_reading,
    _worker_stamp,
    host_rss,
    record_stage,
)


def _wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    return None


def _allocated_worker(connection, allocated_bytes):
    payload = bytearray(allocated_bytes)
    for offset in range(0, allocated_bytes, 4096):
        payload[offset] = 1
    connection.send(_worker_stamp("sweep_trial", "trial-17"))
    connection.close()
    time.sleep(30.0)
    if payload[-1] == 257:  # pragma: no cover - keeps payload alive
        raise AssertionError


@pytest.mark.parametrize(
    ("environment", "preference", "expected"),
    [
        ({}, None, "summary"),
        ({}, "off", "off"),
        ({}, False, "off"),
        ({}, True, "summary"),
        ({}, "detailed", "detailed"),
        ({_PERFORMANCE_LOG_ENV: "0"}, "detailed", "off"),
        ({_PERFORMANCE_LOG_ENV: "true"}, "off", "summary"),
        ({_PERFORMANCE_LOG_ENV: "full"}, None, "detailed"),
        ({_PERFORMANCE_LOG_ENV: ""}, None, "summary"),
    ],
)
def test_performance_mode_is_independent_and_headless(
        environment, preference, expected):
    assert _performance_mode(preference, environment) == expected


def test_an_invalid_environment_mode_is_recorded_and_safely_uses_summary():
    selected = _select_performance_mode(
        "off", {_PERFORMANCE_LOG_ENV: "detialed"}
    )
    assert selected.mode == "summary"
    assert selected.source == "environment"
    assert selected.requested == "detialed"
    assert _PERFORMANCE_LOG_ENV in selected.warning
    assert "performance_logging" in _select_performance_mode(
        "broken", {}
    ).warning
    assert _process_key(123, None) == "123:unknown"
    assert _worker_stamp("saver", 4)["worker_id"] == "4"


def test_worker_stamp_and_thread_names_degrade_without_platform_details(monkeypatch):
    monkeypatch.setattr(psutil, "Process", lambda: (_ for _ in ()).throw(
        psutil.AccessDenied(os.getpid())
    ))
    assert _worker_stamp("trial", 9) == {
        "pid": os.getpid(),
        "create_time": None,
        "worker_kind": "trial",
        "worker_id": "9",
    }
    monkeypatch.setattr(
        resources.threading,
        "enumerate",
        lambda: [types.SimpleNamespace(native_id=None, name="unnamed")],
    )
    assert _python_thread_names() == {}


def test_memory_measure_prefers_pss_and_records_the_uss_fallback():
    fake_psutil = types.SimpleNamespace(
        NoSuchProcess=psutil.NoSuchProcess,
        ZombieProcess=psutil.ZombieProcess,
    )

    class WithUssOnly:
        def memory_full_info(self):
            return types.SimpleNamespace(uss=1234)

    reading = _memory_reading(WithUssOnly(), fake_psutil)
    assert reading["memory_bytes"] == 1234
    assert reading["memory_measure"] == "uss"
    assert reading["memory_fallbacks"] == [
        {"measure": "pss", "reason": "not-exposed"}
    ]


def test_memory_and_cpu_fallbacks_distinguish_unavailable_from_zero():
    class RssOnly:
        def memory_full_info(self):
            raise psutil.AccessDenied(os.getpid())

        def memory_info(self):
            return types.SimpleNamespace(rss=0)

    reading = _memory_reading(RssOnly(), psutil)
    assert reading == {
        "memory_bytes": 0,
        "memory_measure": "rss",
        "memory_fallbacks": [
            {"measure": "pss", "reason": "AccessDenied"},
            {"measure": "uss", "reason": "AccessDenied"},
        ],
    }

    class NothingReadable(RssOnly):
        def memory_info(self):
            raise psutil.AccessDenied(os.getpid())

        def cpu_times(self):
            raise psutil.AccessDenied(os.getpid())

    reading = _memory_reading(NothingReadable(), psutil)
    assert reading["memory_bytes"] is None
    assert reading["memory_measure"] is None
    assert reading["memory_fallbacks"][-1] == {
        "measure": "rss", "reason": "AccessDenied"
    }
    assert _cpu_reading(NothingReadable(), psutil) == (None, "AccessDenied")


def test_process_disappearance_exceptions_are_propagated_to_the_tree_reader():
    class GoneAtMemory:
        def memory_full_info(self):
            raise psutil.NoSuchProcess(888)

    with pytest.raises(psutil.NoSuchProcess):
        _memory_reading(GoneAtMemory(), psutil)

    class GoneAtRss:
        def memory_full_info(self):
            return types.SimpleNamespace()

        def memory_info(self):
            raise psutil.ZombieProcess(889)

    with pytest.raises(psutil.ZombieProcess):
        _memory_reading(GoneAtRss(), psutil)

    class GoneAtCpu:
        def cpu_times(self):
            raise psutil.NoSuchProcess(890)

    with pytest.raises(psutil.NoSuchProcess):
        _cpu_reading(GoneAtCpu(), psutil)

    class GoneAtThreads:
        def threads(self):
            raise psutil.ZombieProcess(891)

    with pytest.raises(psutil.ZombieProcess):
        _thread_reading(GoneAtThreads(), psutil)


def test_a_child_that_dies_between_enumeration_and_read_is_expected():
    class Root:
        pid = os.getpid()

        def __init__(self):
            self.child = DyingChild()

        def children(self, recursive=False):
            assert recursive is True
            return [self.child]

        def create_time(self):
            return 10.0

        def memory_full_info(self):
            return types.SimpleNamespace(pss=100)

        def cpu_times(self):
            return types.SimpleNamespace(user=1.0, system=2.0)

        def name(self):
            return "root"

    class DyingChild:
        pid = 765432

        def create_time(self):
            raise psutil.NoSuchProcess(self.pid)

    root = Root()
    sample = _process_tree_snapshot(process_factory=lambda _pid: root)
    assert sample["tree_memory_bytes"] == 100
    assert sample["tree_memory_measure"] == "pss"
    assert sample["process_count"] == 1
    assert sample["unavailable_processes"] == [
        {"pid": 765432, "reason": "NoSuchProcess"}
    ]


def test_a_partly_readable_process_keeps_explicit_unavailable_fields():
    class PartlyReadable:
        pid = 444

        def create_time(self):
            raise psutil.AccessDenied(self.pid)

        def memory_full_info(self):
            return types.SimpleNamespace(uss=7)

        def cpu_times(self):
            raise psutil.AccessDenied(self.pid)

        def name(self):
            raise psutil.AccessDenied(self.pid)

        def threads(self):
            raise psutil.NoSuchProcess(self.pid)

    row, missing = _read_process(
        PartlyReadable(),
        root_pid=os.getpid(),
        detailed=True,
        labels={(444, None): {"worker_id": "four"}},
        psutil_module=psutil,
    )
    assert missing is None
    assert row["identity"] == "444:unknown"
    assert row["memory_measure"] == "uss"
    assert row["cpu_seconds"] is None
    assert row["cpu_unavailable_reason"] == "AccessDenied"
    assert row["name"] == ""
    assert row["worker"] == {"kind": "worker", "id": "four"}
    assert row["thread_cpu_available"] is False
    assert row["threads"] is None


def test_a_process_gone_after_identity_is_an_unavailable_row():
    class Gone:
        pid = 555

        def create_time(self):
            return 20.0

        def memory_full_info(self):
            raise psutil.NoSuchProcess(self.pid)

    row, missing = _read_process(
        Gone(), root_pid=1, detailed=False, labels={}, psutil_module=psutil
    )
    assert row is None
    assert missing == {
        "pid": 555,
        "create_time": 20.0,
        "identity": "555:20.000000",
        "reason": "NoSuchProcess",
    }


def test_tree_measure_names_empty_single_and_mixed_definitions():
    assert _tree_measure({}) is None
    assert _tree_measure({"pss": 2, "rss": 0}) == "pss"
    assert _tree_measure({"pss": 1, "uss": 1}) == "mixed"


@pytest.mark.parametrize("failure", [psutil.NoSuchProcess(1), psutil.AccessDenied(1)])
def test_child_enumeration_failure_is_in_the_sample(failure):
    class Root:
        pid = os.getpid()

        def children(self, recursive=False):
            raise failure

        def create_time(self):
            return 1.0

        def memory_full_info(self):
            return types.SimpleNamespace(pss=9)

        def cpu_times(self):
            return types.SimpleNamespace(user=0, system=0)

        def name(self):
            return "root"

    sample = _process_tree_snapshot(process_factory=lambda _pid: Root())
    assert sample["process_count"] == 1
    assert sample["unavailable_processes"][0]["operation"] == "children"
    assert sample["unavailable_processes"][0]["reason"] == failure.__class__.__name__


def test_tree_enumeration_skips_duplicate_and_unidentifiable_processes():
    class Root:
        pid = os.getpid()

        def children(self, recursive=False):
            return [self, BadPid()]

        def create_time(self):
            return 1.0

        def memory_full_info(self):
            return types.SimpleNamespace(pss=4)

        def cpu_times(self):
            return types.SimpleNamespace(user=0, system=0)

        def name(self):
            return "root"

    class BadPid:
        @property
        def pid(self):
            raise psutil.AccessDenied(123)

    sample = _process_tree_snapshot(process_factory=lambda _pid: Root())
    assert sample["process_count"] == 1
    assert sample["unavailable_processes"] == [
        {"pid": None, "reason": "AccessDenied"}
    ]


def test_unavailable_thread_cpu_is_none_and_never_counterfeit_zero():
    class UnreadableThreads:
        def threads(self):
            raise psutil.AccessDenied(os.getpid())

    result = _thread_reading(UnreadableThreads(), psutil)
    assert result["thread_cpu_available"] is False
    assert result["thread_cpu_unavailable_reason"] == "AccessDenied"
    assert result["threads"] is None


def test_off_mode_starts_no_thread_writes_no_file_and_installs_no_profile_hook(
        tmp_path):
    output = tmp_path / "resources.json"
    census_before = {
        thread.ident for thread in threading.enumerate()
        if thread.name.startswith("spacr-resource-sampler-")
    }
    sys_profile = sys.getprofile()
    thread_profile = threading.getprofile()

    sampler = _ResourceSampler(output, mode="off", environ={})._start()
    assert sampler._thread is None
    assert sampler._stop() is None

    census_after = {
        thread.ident for thread in threading.enumerate()
        if thread.name.startswith("spacr-resource-sampler-")
    }
    assert census_after == census_before
    assert not output.exists()
    assert sys.getprofile() is sys_profile
    assert threading.getprofile() is thread_profile


def test_summary_is_daemon_sampled_without_a_time_series_and_is_atomic(tmp_path):
    output = tmp_path / "resources.json"
    sys_profile = sys.getprofile()
    thread_profile = threading.getprofile()
    sampler = _ResourceSampler(
        output,
        mode="summary",
        environ={},
        interval_seconds=0.02,
        checkpoint_samples=1,
    )._start()
    try:
        assert sampler._thread is not None
        assert sampler._thread.daemon is True
        assert _wait_until(
            lambda: sampler._summary["samples_recorded"] >= 3
        )
    finally:
        sampler._stop("completed")

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["mode"] == "summary"
    assert "samples" not in document
    assert document["summary"]["samples_recorded"] >= 3
    assert document["summary"]["peak_tree_memory_bytes"] > 0
    assert document["configuration"]["sample_interval_seconds"] == 0.02
    assert document["configuration"]["profile_hook_installed"] is False
    assert sys.getprofile() is sys_profile
    assert threading.getprofile() is thread_profile
    assert not list(tmp_path.glob(".resources.json.*.tmp"))


def test_detailed_series_is_bounded_and_reports_thread_availability(tmp_path):
    output = tmp_path / "resources.json"
    sampler = _ResourceSampler(
        output,
        mode="detailed",
        environ={},
        interval_seconds=0.01,
        sample_limit=2,
        checkpoint_samples=1,
    )._start()
    try:
        assert _wait_until(
            lambda: sampler._summary["samples_recorded"] >= 5
        )
    finally:
        sampler._stop("completed")

    document = json.loads(output.read_text(encoding="utf-8"))
    assert len(document["samples"]) == 2
    assert document["samples_dropped"] >= 3
    root = document["samples"][-1]["processes"][0]
    assert isinstance(root["thread_cpu_available"], bool)
    if root["thread_cpu_available"]:
        assert root["threads"]
        assert all(row["cpu_total_seconds"] >= 0 for row in root["threads"])
    else:
        assert root["threads"] is None
        assert root["thread_cpu_unavailable_reason"]


def test_sampler_failure_paths_remain_diagnostics_not_run_failures(
        tmp_path, monkeypatch):
    output = tmp_path / "resources.json"
    sampler = _ResourceSampler(output, mode="summary", environ={})
    with pytest.raises(ValueError, match="output JSON path"):
        _ResourceSampler(None, mode="summary", environ={})._start()

    monkeypatch.setattr(resources, "_atomic_json", lambda *_args: (_ for _ in ()).throw(
        OSError("read only")
    ))
    assert sampler._persist() is False
    assert sampler._write_error == "OSError"
    assert _ResourceSampler(output, mode="off", environ={})._persist() is False

    monkeypatch.setattr(resources, "_atomic_json", _atomic_json)
    sampler._record_sampler_error(RuntimeError("sample failed"))
    events = json.loads(output.read_text(encoding="utf-8"))["events"]
    assert len(events) == 1
    assert events[0]["error"] == "RuntimeError"
    assert events[0]["kind"] == "sampler_error"
    assert events[0]["utc"].endswith("Z")


def test_atomic_writer_tolerates_directory_sync_unavailability(tmp_path, monkeypatch):
    output = tmp_path / "resources.json"
    real_open = os.open

    def no_directory_open(path, flags, *args, **kwargs):
        if str(path) == str(tmp_path):
            raise OSError("directory handles unavailable")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(resources.os, "open", no_directory_open)
    _atomic_json(output, {"complete": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"complete": True}


def test_atomic_writer_tolerates_directory_fsync_unavailability(tmp_path, monkeypatch):
    output = tmp_path / "resources.json"
    real_fsync = os.fsync
    calls = 0

    def fail_second_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync unavailable")
        return real_fsync(descriptor)

    monkeypatch.setattr(resources.os, "fsync", fail_second_fsync)
    _atomic_json(output, {"complete": True})
    assert calls == 2
    assert json.loads(output.read_text(encoding="utf-8")) == {"complete": True}


def test_sampler_loop_and_stop_capture_measurement_errors(tmp_path, monkeypatch):
    loop_output = tmp_path / "loop.json"
    looping = _ResourceSampler(loop_output, mode="summary", environ={})

    def fail_loop_sample():
        looping._stop_event.set()
        raise RuntimeError("sample")

    monkeypatch.setattr(looping, "_sample_once", fail_loop_sample)
    looping._run()
    assert json.loads(loop_output.read_text(encoding="utf-8"))["events"][0][
        "kind"
    ] == "sampler_error"
    looping._run()  # an already-stopped loop performs no work

    stop_output = tmp_path / "stop.json"
    stopping = _ResourceSampler(stop_output, mode="summary", environ={})
    monkeypatch.setattr(
        stopping,
        "_sample_once",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("final sample")),
    )
    stopping._stop("failed measurement")
    document = json.loads(stop_output.read_text(encoding="utf-8"))
    assert document["stop_reason"] == "failed measurement"
    assert document["events"][0]["kind"] == "sampler_error"


def test_sampler_context_records_success_and_failure(tmp_path):
    completed = tmp_path / "completed.json"
    with _ResourceSampler(
        completed, mode="summary", environ={}, interval_seconds=0.01
    ):
        pass
    assert json.loads(completed.read_text(encoding="utf-8"))["stop_reason"] == "completed"

    failed = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="boom"):
        with _ResourceSampler(
            failed, mode="summary", environ={}, interval_seconds=0.01
        ):
            raise RuntimeError("boom")
    assert json.loads(failed.read_text(encoding="utf-8"))["stop_reason"] == "failed"


def test_sampler_restart_and_stop_timeout_are_idempotent(tmp_path, monkeypatch):
    sampler = _ResourceSampler(
        tmp_path / "restart.json",
        mode="summary",
        environ={},
        interval_seconds=0.1,
    )._start()
    try:
        original = sampler._thread
        assert sampler._start()._thread is original
    finally:
        sampler._stop()

    timed_out = _ResourceSampler(tmp_path / "timeout.json", mode="summary", environ={})
    timed_out._thread = threading.current_thread()
    monkeypatch.setattr(timed_out, "_persist", lambda: True)
    timed_out._stop("timeout")
    assert timed_out._events[-1]["kind"] == "sampler_stop_timeout"


def test_unlabelled_child_disappearance_and_unknown_measure_are_retained(tmp_path):
    sampler = _ResourceSampler(tmp_path / "events.json", mode="summary", environ={})
    sampler._seen_children = {
        "2:1.000000": {"identity": "2:1.000000", "pid": 2}
    }
    events = sampler._disappearance_events({"utc": "now", "processes": []})
    assert events == [{
        "kind": "process_disappeared",
        "utc": "now",
        "identity": "2:1.000000",
        "pid": 2,
        "exit_status": None,
        "exit_status_available": False,
    }]
    sampler._update_summary({
        "tree_memory_bytes": None,
        "tree_memory_measure": "unknown",
        "process_count": 0,
        "unavailable_processes": [],
    })
    assert sampler._summary["peak_tree_memory_bytes"] is None


def test_a_stamped_short_lived_worker_is_named_even_before_its_first_sample(tmp_path):
    sampler = _ResourceSampler(tmp_path / "short.json", mode="summary", environ={})
    sampler._register_worker({
        "pid": 999999,
        "create_time": 42.0,
        "worker_kind": "sequencing_saver",
        "worker_id": "plate-7",
    })
    events = sampler._disappearance_events({"utc": "later", "processes": []})
    assert events[0]["identity"] == "999999:42.000000"
    assert events[0]["worker"] == {
        "kind": "sequencing_saver", "id": "plate-7"
    }
    assert sampler._labels == {}

    already_seen = _ResourceSampler(
        tmp_path / "already.json", mode="summary", environ={}
    )
    already_seen._seen_children["888888:8.000000"] = {
        "identity": "888888:8.000000", "pid": 888888, "create_time": 8.0
    }
    already_seen._register_worker({
        "pid": 888888,
        "create_time": 8.0,
        "worker_kind": "trial",
        "worker_id": "seen-first",
    })
    assert already_seen._seen_children["888888:8.000000"]["worker"] == {
        "kind": "trial", "id": "seen-first"
    }

    unknown_pid = _ResourceSampler(
        tmp_path / "unknown.json", mode="summary", environ={}
    )
    unknown_pid._seen_children["unknown"] = {
        "identity": "unknown", "pid": None
    }
    assert unknown_pid._disappearance_events({
        "utc": "later", "processes": []
    })[0]["pid"] is None

    unknown_creation = _ResourceSampler(
        tmp_path / "unknown-creation.json", mode="summary", environ={}
    )
    unknown_creation._register_worker({
        "pid": 777777,
        "create_time": None,
        "worker_kind": "trial",
        "worker_id": "still-running",
    })
    assert unknown_creation._disappearance_events({
        "utc": "later",
        "processes": [{
            "identity": "777777:12.000000",
            "pid": 777777,
            "create_time": 12.0,
            "relation": "child",
        }],
    }) == []
    assert unknown_creation._seen_children["777777:12.000000"]["worker"] == {
        "kind": "trial", "id": "still-running"
    }

    unknown_unlabelled = _ResourceSampler(
        tmp_path / "unknown-unlabelled.json", mode="summary", environ={}
    )
    unknown_unlabelled._seen_children["666666:unknown"] = {
        "identity": "666666:unknown", "pid": 666666, "create_time": None
    }
    assert unknown_unlabelled._disappearance_events({
        "utc": "later",
        "processes": [{
            "identity": "666666:6.000000",
            "pid": 666666,
            "create_time": 6.0,
            "relation": "child",
        }],
    }) == []


def test_stage_tree_failure_is_explicitly_unavailable(monkeypatch):
    monkeypatch.setattr(
        resources,
        "_process_tree_snapshot",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("no process table")),
    )
    assert _tree_stage_reading() == {
        "tree_memory_bytes": None,
        "tree_memory_measure": None,
        "tree_process_count": None,
    }


def test_allocated_child_is_separate_named_and_durably_ends_at_disappearance(
        tmp_path):
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    allocated_bytes = 32 * 1024 * 1024
    output = tmp_path / "resources.json"
    sampler = _ResourceSampler(
        output,
        mode="detailed",
        environ={},
        interval_seconds=0.05,
        sample_limit=20,
        checkpoint_samples=100,
    )
    parent_rss_before = host_rss()
    tree_before = _process_tree_snapshot()["tree_memory_bytes"]
    process = context.Process(
        target=_allocated_worker,
        args=(child_connection, allocated_bytes),
    )
    process.start()
    child_connection.close()
    try:
        assert parent_connection.poll(15.0), "allocated child did not stamp itself"
        stamp = parent_connection.recv()
        sampler._register_worker(stamp)
        sampler._start()

        def named_sample():
            with sampler._state_lock:
                samples = list(sampler._samples)
            for sample in reversed(samples):
                for row in sample["processes"]:
                    if row.get("worker") == {
                        "kind": "sweep_trial", "id": "trial-17"
                    }:
                        return sample, row
            return None

        found = _wait_until(named_sample, timeout=15.0)
        assert found is not None, "process tree never named the allocated trial"
        sample, worker = found
        root = next(
            row for row in sample["processes"] if row["relation"] == "root"
        )
        assert worker["pid"] == process.pid
        assert worker["memory_measure"] in {"pss", "uss", "rss"}
        assert worker["memory_bytes"] >= allocated_bytes // 2
        assert sample["tree_memory_bytes"] >= (
            root["memory_bytes"] + worker["memory_bytes"]
        )

        parent_rss_after = host_rss()
        assert parent_rss_before is not None and parent_rss_after is not None
        assert parent_rss_after - parent_rss_before < allocated_bytes // 2
        assert tree_before is not None
        assert sample["tree_memory_bytes"] - tree_before > allocated_bytes // 2

        process.terminate()
        process.join(timeout=10.0)
        assert not process.is_alive()

        def durable_disappearance():
            if not output.exists():
                return None
            document = json.loads(output.read_text(encoding="utf-8"))
            for event in document["events"]:
                if (
                    event.get("kind") == "process_disappeared"
                    and event.get("worker") == {
                        "kind": "sweep_trial", "id": "trial-17"
                    }
                ):
                    return document
            return None

        document = _wait_until(durable_disappearance, timeout=15.0)
        assert document is not None
        event = next(
            event for event in document["events"]
            if event.get("worker") == {
                "kind": "sweep_trial", "id": "trial-17"
            }
        )
        assert event["exit_status"] is None
        assert event["exit_status_available"] is False
        assert not list(tmp_path.glob(".resources.json.*.tmp"))
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10.0)
        sampler._stop("test cleanup")
        parent_connection.close()


def test_existing_fit_stage_also_names_its_process_tree_measure():
    reading = record_stage({}, "fit")
    assert reading["tree_memory_bytes"] > 0
    assert reading["tree_memory_measure"] in {"pss", "uss", "rss", "mixed"}
    assert reading["tree_process_count"] >= 1
