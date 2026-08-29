"""The startup benchmark's fresh-process and exact-registry contract."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from tools import spacr_startup_benchmark as driver


def _artifact(label: str, keys: list[str], violations=()):
    return {
        "environment": {
            "spacr_file": str(driver.PACKAGE_ROOT / "spacr" / "__init__.py"),
            "qt_package_file": str(
                driver.PACKAGE_ROOT / "spacr" / "qt" / "__init__.py"),
        },
        "benchmark": {
            "run": label,
            "exit_reason": "registry sweep complete",
            "registry_keys": keys,
            "registry_count": len(keys),
            "measured_keys": keys,
            "measured_count": len(keys),
            "registry_matches_measurements": True,
            "violations": list(violations),
        }
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
    assert replaced and replaced[-1][1] == output
    assert replaced[-1][0] != output
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
