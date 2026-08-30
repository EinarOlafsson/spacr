"""The startup benchmark's fresh-process and exact-registry contract."""
from __future__ import annotations

import json
import subprocess
import time
from copy import deepcopy
from pathlib import Path

import pytest

from tools import spacr_startup_benchmark as driver


def _artifact(label: str, keys: list[str], violations=()):
    home = {
        "at": 1.0,
        "started_at": 0.0,
        "event_loop_started_at": 0.5,
        "duration_s": 1.0,
        "name": "interactive Home",
        "detail": "__home__",
        "budget_s": driver.HOME_BUDGET_S,
        "within_budget": True,
        "root_painted": False,
        "screen_tree_painted": True,
        "painted_usable_controls": 1,
        "usable_controls": 2,
        "controls": ["HomeButton"],
        "thread": "MainThread",
        "stall_window_started_at": 0.0,
        "stall_window_ended_at": 1.1,
        "worst_event_loop_stall_ms": 0.0,
        "worst_overlapping_frame_interval_ms": 0.0,
        "event_loop_stall_budget_met": True,
        "stall_samples": 0,
    }
    preferences = {
        "at": 1.6,
        "started_at": 1.5,
        "event_loop_started_at": 0.5,
        "duration_s": 0.1,
        "name": "interactive preferences",
        "detail": "__preferences__",
        "budget_s": driver.PREFERENCES_BUDGET_S,
        "within_budget": True,
        "stall_window_started_at": 1.5,
        "stall_window_ended_at": 1.7,
        "worst_event_loop_stall_ms": 0.0,
        "worst_overlapping_frame_interval_ms": 0.0,
        "event_loop_stall_budget_met": True,
        "stall_samples": 0,
    }
    modules = []
    for index, key in enumerate(keys):
        row = dict(home)
        row.update({
            "at": 3.0 + index,
            "started_at": 2.0 + index,
            "event_loop_started_at": 0.5,
            "duration_s": 1.0,
            "name": "interactive module",
            "detail": key,
            "budget_s": driver.MODULE_BUDGET_S,
            "controls": [f"{key}Button"],
            "stall_window_started_at": 2.0 + index,
            "stall_window_ended_at": 3.1 + index,
        })
        modules.append(row)
    return {
        "schema_version": driver.WORKER_SCHEMA_VERSION,
        "elapsed_s": 4.0 + len(keys),
        "budgets": {
            "home_ready_s": driver.HOME_BUDGET_S,
            "module_ready_s": driver.MODULE_BUDGET_S,
            "max_event_loop_stall_ms": driver.STALL_BUDGET_MS,
            "watchdog_record_floor_ms": 50.0,
        },
        "import_timing_enabled": False,
        "environment": {
            "python": "3.12.4",
            "implementation": "CPython",
            "platform": "Linux-test",
            "machine": "x86_64",
            "processor": "",
            "qt": "6.11.2",
            "executable": "/test/bin/python",
            "pid": 1234,
            "spacr_file": str(driver.PACKAGE_ROOT / "spacr" / "__init__.py"),
            "qt_package_file": str(
                driver.PACKAGE_ROOT / "spacr" / "qt" / "__init__.py"),
            "spacr_version": "1.5.0.4",
            "hardware": {
                "logical_cpu_count": 4,
                "total_memory_mb": 8192.0,
                "performance_level": "laptop",
                "qt_platform": "xcb",
                "displays": [{
                    "name": "test-display",
                    "logical_width": 1920,
                    "logical_height": 1080,
                    "device_pixel_ratio": 1.0,
                    "refresh_hz": 60.0,
                }],
            },
        },
        "resources": {
            "peak_rss_mb": 320.0,
            "gpu": {"allocated_mb": None, "peak_allocated_mb": None},
        },
        "event_loop_started_at": 0.5,
        "worst_event_loop_stall_ms": 0.0,
        "stall_budget_met": True,
        "spans": [],
        "imports": [],
        "stalls": [],
        "marks": [],
        "readiness": [home, *modules],
        "benchmark": {
            "run": label,
            "exit_reason": "registry sweep complete",
            "registry_keys": keys,
            "registry_count": len(keys),
            "final_registry_keys": keys,
            "registry_stable": True,
            "measured_keys": keys,
            "measured_count": len(keys),
            "registry_matches_measurements": True,
            "preferences_measured": True,
            "preferences_budget_s": driver.PREFERENCES_BUDGET_S,
            "results": [home, preferences, *modules],
            "violations": list(violations),
        },
        "worker": {
            "returncode": 0,
            "elapsed_s": 4.0,
            "stdout_tail": "",
            "stderr_tail": "",
        },
    }


def test_worker_uses_a_real_timed_entry_point_without_import_profiler_skew(
        tmp_path, monkeypatch):
    monkeypatch.delenv("SPACR_TIMING", raising=False)
    env = driver._worker_environment(
        tmp_path / "home", tmp_path / "run.json", "cold-process", 17.0,
        True)

    assert env["SPACR_TIMING"] == "1"
    assert env["SPACR_TIMING_IMPORTS"] == "0"
    assert time.time() - 5.0 <= float(
        env["SPACR_TIMING_PROCESS_START"]) <= time.time()
    assert env["SPACR_BENCHMARK_TIMEOUT_S"] == "17.0"
    assert env["SPACR_BENCHMARK_HARD_TIMEOUT"] == "1"
    assert env["QT_QPA_PLATFORM"] == "offscreen"
    assert env["HOME"] == str(tmp_path / "home")
    assert env["PYTHONPATH"] == str(driver.PACKAGE_ROOT)
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["SPACR_BENCHMARK_PACKAGE_ROOT"] == str(driver.PACKAGE_ROOT)
    assert "PYTHONHOME" not in env
    assert "outside expected package root" in driver.WORKER
    assert "spacr.qt.run" in driver.WORKER
    assert "--no-setup" in driver.WORKER


def test_cold_and_warm_runs_share_only_the_isolated_disk_home(
        tmp_path, monkeypatch):
    calls = []
    keys = [f"app_{index}" for index in range(44)]

    def fake_worker(home, output, label, timeout_s, offscreen, *, package_root):
        calls.append((
            Path(home), Path(output), label, timeout_s, offscreen,
            Path(package_root),
        ))
        return _artifact(label, keys)

    monkeypatch.setattr(driver, "_run_worker", fake_worker)
    output = tmp_path / "combined.json"
    result = driver.run_benchmark(
        output, runs=2, timeout_s=21.0, offscreen=True)

    assert result["passed"] is True
    assert result["registry_count"] == 44
    assert [call[2] for call in calls] == ["cold-process", "warm-process"]
    assert calls[0][0] == calls[1][0]
    assert calls[0][1] != calls[1][1]
    assert all(call[3:] == (21.0, True, driver.PACKAGE_ROOT) for call in calls)


def test_registry_drift_or_one_worker_without_a_registry_fails_the_ratchet():
    mismatch = driver._combined_violations([
        _artifact("cold-process", ["a", "b"]),
        _artifact("warm-process", ["a", "c"]),
    ])
    assert "the live registry changed between cold and warm runs" in mismatch

    missing = driver._combined_violations([
        _artifact("cold-process", ["a"]),
        {"benchmark": {"run": "warm-process", "violations": []}},
    ])
    assert "one or more runs did not report the live registry" in missing

    inexact = _artifact("cold-process", ["a"])
    inexact["benchmark"]["registry_matches_measurements"] = False
    violations = driver._combined_violations([inexact])
    assert "cold-process: measured app sequence did not equal its registry" in (
        violations)

    incomplete = _artifact("cold-process", ["a"])
    incomplete["benchmark"]["exit_reason"] = "registry sweep in progress"
    violations = driver._combined_violations([incomplete])
    assert "cold-process: worker did not complete the registry sweep" in (
        violations)


def test_a_complete_worker_artifact_passes_the_independent_schema_ratchet():
    assert driver._combined_violations([
        _artifact("cold-process", ["first", "second"]),
        _artifact("warm-process", ["first", "second"]),
    ]) == []


def test_trace_integrity_is_a_new_worker_and_combined_artifact_schema():
    assert driver.WORKER_SCHEMA_VERSION == 2
    assert driver.SCHEMA_VERSION == 2
    old = _artifact("cold-process", ["probe"])
    old["schema_version"] = 1

    assert any(
        "worker schema_version must be 2" in row
        for row in driver._combined_violations([old])
    )


@pytest.mark.parametrize(("path", "replacement", "message"), [
    (("schema_version",), None, "worker schema_version must be 2"),
    (("environment", "hardware", "displays"), [],
     "displays must contain at least one display"),
    (("resources", "peak_rss_mb"), None,
     "resources.peak_rss_mb must be reported"),
    (("benchmark", "results"), [],
     "results count does not equal Home + Preferences + the live registry"),
    (("benchmark", "results", 0, "painted_usable_controls"), 0,
     "painted_usable_controls is not positive"),
    (("benchmark", "results", 0, "controls"), [],
     "controls lacks painted control names"),
    (("benchmark", "results", 0, "duration_s"), "fast",
     "duration_s was not recorded"),
    (("benchmark", "results", 0, "at"), 0.25,
     "event_loop_started_at is after readiness"),
    (("benchmark", "results", 1, "worst_event_loop_stall_ms"), None,
     "worst_event_loop_stall_ms was not recorded"),
    (("benchmark", "results", 2, "budget_s"), 99.0,
     "budget_s must equal 10.0"),
    (("readiness",), [],
     "readiness sequence does not equal Home + the live registry"),
    (("worker",), None, "worker process evidence is missing"),
])
def test_skeletal_or_malformed_worker_evidence_cannot_pass(
        path, replacement, message):
    artifact = deepcopy(_artifact("cold-process", ["probe"]))
    target = artifact
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = replacement

    violations = driver._combined_violations([artifact])

    assert any(message in violation for violation in violations), violations


def test_the_old_skeletal_registry_only_shape_is_rejected():
    skeletal = {
        "environment": {
            "spacr_file": str(driver.PACKAGE_ROOT / "spacr" / "__init__.py"),
            "qt_package_file": str(
                driver.PACKAGE_ROOT / "spacr" / "qt" / "__init__.py"),
        },
        "benchmark": {
            "run": "cold-process",
            "exit_reason": "registry sweep complete",
            "registry_keys": ["probe"],
            "registry_matches_measurements": True,
            "violations": [],
        },
    }

    violations = driver._combined_violations([skeletal])

    assert "cold-process: worker schema_version must be 2" in violations
    assert any("worker process evidence is missing" in row for row in violations)
    assert any("benchmark.results" in row for row in violations)


def test_readiness_evidence_must_match_the_budgeted_result_copy():
    artifact = deepcopy(_artifact("cold-process", ["probe"]))
    artifact["readiness"][0] = dict(artifact["readiness"][0])
    artifact["readiness"][0]["controls"] = ["different-control"]

    violations = driver._combined_violations([artifact])

    assert any(
        "readiness[0].controls does not match its benchmark result" in row
        for row in violations
    )


def test_raw_frame_gaps_are_recomputed_instead_of_trusting_worker_summaries():
    artifact = deepcopy(_artifact("cold-process", ["probe"]))
    artifact["stalls"] = [{
        "started_at": 0.6,
        "at": 0.8,
        "late_ms": 200.0,
        "source": "event-loop watchdog",
        "thread": "MainThread",
    }]

    violations = driver._combined_violations([artifact])

    assert any(
        "worst_event_loop_stall_ms does not match raw stalls" in row
        for row in violations
    )
    assert any(
        "benchmark.results[0].stall_samples does not match raw stalls" in row
        for row in violations
    )
    assert any(
        "benchmark.results[0].worst_overlapping_frame_interval_ms does not "
        "match raw stalls" in row
        for row in violations
    )


def test_a_raw_frame_gap_and_its_independently_derived_fields_can_agree():
    artifact = deepcopy(_artifact("cold-process", ["probe"]))
    artifact["stalls"] = [{
        "started_at": 0.6,
        "at": 0.8,
        "late_ms": 200.0,
        "source": "event-loop watchdog",
        "thread": "MainThread",
    }]
    artifact["worst_event_loop_stall_ms"] = 200.0
    home = artifact["benchmark"]["results"][0]
    home.update({
        "worst_event_loop_stall_ms": 200.0,
        "worst_overlapping_frame_interval_ms": 200.0,
        "stall_samples": 1,
    })

    assert driver._combined_violations([artifact]) == []


def test_a_later_beat_beginning_at_a_sealed_window_is_not_charged_backwards():
    artifact = deepcopy(_artifact("cold-process", ["probe"]))
    artifact["stalls"] = [{
        "started_at": 1.1,
        "at": 1.3,
        "late_ms": 200.0,
        "source": "event-loop watchdog",
        "thread": "MainThread",
    }]
    artifact["worst_event_loop_stall_ms"] = 200.0

    assert driver._combined_violations([artifact]) == []


def test_a_shifted_stall_window_cannot_hide_the_actual_readiness_interval():
    artifact = deepcopy(_artifact("cold-process", ["probe"]))
    artifact["stalls"] = [{
        "started_at": 0.6,
        "at": 0.8,
        "late_ms": 200.0,
        "source": "event-loop watchdog",
        "thread": "MainThread",
    }]
    artifact["worst_event_loop_stall_ms"] = 200.0
    home = artifact["benchmark"]["results"][0]
    home["stall_window_started_at"] = 1.1
    home["stall_window_ended_at"] = 1.4

    violations = driver._combined_violations([artifact])

    assert any(
        "benchmark.results[0].stall_window_started_at does not match "
        "started_at" in row
        for row in violations
    )


def test_a_raw_gap_cannot_extend_beyond_the_artifact_clock():
    artifact = deepcopy(_artifact("cold-process", ["probe"]))
    elapsed = artifact["elapsed_s"]
    artifact["stalls"] = [{
        "started_at": elapsed + 0.1,
        "at": elapsed + 0.3,
        "late_ms": 200.0,
        "source": "event-loop watchdog",
        "thread": "MainThread",
    }]
    artifact["worst_event_loop_stall_ms"] = 200.0

    violations = driver._combined_violations([artifact])

    assert any(
        "stalls[0].at is after artifact elapsed_s" in row
        for row in violations
    )


def test_timestamp_duration_decides_the_stall_budget_at_the_500ms_edge():
    artifact = deepcopy(_artifact("cold-process", ["probe"]))
    artifact["stalls"] = [{
        "started_at": 0.0,
        "at": 0.5000004,
        "late_ms": 499.9995,
        "source": "event-loop watchdog",
        "thread": "MainThread",
    }]
    artifact["worst_event_loop_stall_ms"] = 499.9995
    home = artifact["benchmark"]["results"][0]
    home.update({
        "worst_event_loop_stall_ms": 499.9995,
        "worst_overlapping_frame_interval_ms": 499.9995,
        "stall_samples": 1,
    })

    violations = driver._combined_violations([artifact])

    assert any(
        "worst_event_loop_stall_ms reached the 500 ms ceiling" in row
        for row in violations
    )
    assert any(
        "stall_budget_met does not match raw stalls" in row
        for row in violations
    )


@pytest.mark.parametrize(("field", "replacement", "message"), [
    ("started_at", None, "stalls[0].started_at must be finite"),
    ("at", 0.5, "stalls[0].at is before started_at"),
    ("late_ms", 1.0, "stalls[0].late_ms does not match its timestamps"),
])
def test_malformed_raw_frame_gaps_cannot_be_release_evidence(
        field, replacement, message):
    artifact = deepcopy(_artifact("cold-process", ["probe"]))
    stall = {
        "started_at": 0.6,
        "at": 0.8,
        "late_ms": 200.0,
        "source": "event-loop watchdog",
        "thread": "MainThread",
    }
    stall[field] = replacement
    artifact["stalls"] = [stall]

    violations = driver._combined_violations([artifact])

    assert any(message in row for row in violations), violations


def test_combined_artifacts_have_exactly_one_or_two_processes(tmp_path):
    assert driver._combined_violations([]) == [
        "combined artifact must contain one cold run or cold + warm runs"]
    with pytest.raises(ValueError, match="runs must be one cold process"):
        driver.run_benchmark(tmp_path / "bad.json", runs=True)


def test_run_benchmark_writes_a_failed_combined_artifact_for_bad_worker_shape(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        driver, "_run_worker", lambda *_args, **_kwargs: {"benchmark": []})
    output = tmp_path / "malformed-combined.json"

    artifact = driver.run_benchmark(output, runs=1)

    assert artifact["passed"] is False
    assert artifact["registry_count"] == 0
    assert any(
        "benchmark is not a JSON object" in row
        for row in artifact["violations"]
    )
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is False


def test_an_import_outside_the_requested_installation_fails_the_ratchet(
        tmp_path):
    root = tmp_path / "installed"
    artifact = _artifact("cold-process", ["a"])
    artifact["environment"]["spacr_file"] = str(
        tmp_path / "checkout" / "spacr" / "__init__.py")

    violations = driver._combined_violations([artifact], root)

    assert f"cold-process: spacr_file did not resolve inside {root}" in violations
    assert f"cold-process: qt_package_file did not resolve inside {root}" in (
        violations)


def test_outer_timeout_preserves_the_last_checkpoint_and_decodes_output(
        tmp_path, monkeypatch):
    output = tmp_path / "worker.json"
    output.write_text(
        json.dumps(_artifact("cold-process", ["a"])),
        encoding="utf-8")

    def expire(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd="python", timeout=1, output=b"partial stdout",
            stderr=b"partial stderr")

    monkeypatch.setattr(subprocess, "run", expire)
    result = driver._run_worker(
        tmp_path / "home", output, "cold-process", 1.0, True)

    assert result["benchmark"]["registry_keys"] == ["a"]
    assert "worker exceeded its controlled" in result["benchmark"][
        "violations"][-1]
    assert result["worker"]["stdout_tail"] == "partial stdout"
    assert result["worker"]["stderr_tail"] == "partial stderr"


def test_a_non_object_worker_json_is_a_failed_artifact_not_a_driver_crash(
        tmp_path, monkeypatch):
    output = tmp_path / "worker.json"
    output.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["python"], returncode=0, stdout="", stderr=""),
    )

    result = driver._run_worker(
        tmp_path / "home", output, "cold-process", 1.0, True)

    assert result["benchmark"]["violations"] == [
        "invalid worker artifact: root is not a JSON object"]
    assert result["worker"]["returncode"] == 0
    assert any(
        "worker schema_version must be 2" in row
        for row in driver._combined_violations([result])
    )


def test_combined_artifact_is_atomically_replaced(
        tmp_path, monkeypatch):
    keys = [f"app_{index}" for index in range(44)]
    monkeypatch.setattr(
        driver, "_run_worker",
        lambda *_args, **_kwargs: _artifact("cold-process", keys),
    )
    replaced = []
    real_replace = driver.os.replace

    def observe_replace(source, destination):
        replaced.append((Path(source), Path(destination)))
        assert Path(source).read_text(encoding="utf-8").endswith("\n")
        real_replace(source, destination)

    monkeypatch.setattr(driver.os, "replace", observe_replace)
    output = tmp_path / "combined.json"

    artifact = driver.run_benchmark(output, runs=1)

    assert artifact["passed"] is True
    assert artifact["schema_version"] == driver.SCHEMA_VERSION
    assert replaced and replaced[-1][1] == output
    assert replaced[-1][0] != output
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
