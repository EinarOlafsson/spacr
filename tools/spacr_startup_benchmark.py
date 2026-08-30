#!/usr/bin/env python3
"""Benchmark the real spaCR entry point through every live Qt app.

Each run is a fresh interpreter executing ``spacr.qt.run(["--no-setup"])``.
Production instrumentation waits for a post-event-loop paint of Home and an
enabled control, presses every live sidebar app button, and exits Qt after the
last post-paint ready state.  The first process is labelled cold and the
second warm; both use the same isolated home so the latter sees filesystem and
spaCR caches without inheriting Python modules or Qt objects.

The default is a strict release ratchet: Home must be ready in 5 seconds,
every registry module in 10 seconds, every key must be measured exactly once,
and no measured interval may contain a 500 ms event-loop stall.  ``--record-
only`` writes evidence without turning a budget miss into a non-zero exit.

Hosted CI deliberately ratchets this artifact's schema through unit fixtures,
not by claiming its offscreen Qt plugin is release performance evidence.  A
hosted runner has neither the documented lower-end hardware profile nor a real
display server/compositor/refresh path, and its shared-host scheduling noise is
not hardware-normalised.  ``--offscreen`` therefore remains diagnostic; final
acceptance is a real-display run whose complete artifact passes this driver.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
WORKER_SCHEMA_VERSION = 1
HOME_BUDGET_S = 5.0
MODULE_BUDGET_S = 10.0
PREFERENCES_BUDGET_S = 3.0
STALL_BUDGET_MS = 500.0

WORKER = """
import os
from pathlib import Path
import spacr.qt

expected = Path(os.environ["SPACR_BENCHMARK_PACKAGE_ROOT"]).resolve()
actual = Path(spacr.qt.__file__).resolve()
try:
    actual.relative_to(expected)
except ValueError:
    raise SystemExit(
        f"benchmark imported {actual}, outside expected package root {expected}"
    )
raise SystemExit(spacr.qt.run(["--no-setup"]))
"""


def _default_output() -> Path:
    root = Path.home() / ".spacr" / "reports"
    return root / f"startup-{time.strftime('%Y%m%d-%H%M%S')}.json"


def _worker_environment(home: Path, output: Path, label: str,
                        timeout_s: float, offscreen: bool, *,
                        package_root: Optional[Path] = None) -> dict[str, str]:
    root = (package_root or PACKAGE_ROOT).expanduser().resolve()
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env.update({
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "MPLBACKEND": "Agg",
        "PYTHONDONTWRITEBYTECODE": "1",
        # The child starts in its isolated HOME, so an inherited editable
        # install or another agent's checkout must not decide what is timed.
        # A wheel audit passes its site-packages directory here; a checkout
        # audit defaults to the repository containing this driver.
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(root),
        "SPACR_NO_SETUP": "1",
        "SPACR_TIMING": "1",
        "SPACR_TIMING_IMPORTS": "0",
        # Wall time is portable between processes on Python 3.9 Windows;
        # timing.begin() translates this short age onto the child monotonic
        # clock so interval measurements remain monotonic afterward.
        "SPACR_TIMING_PROCESS_START": repr(time.time()),
        "SPACR_TIMING_LOG": str(output.with_suffix(".timing.txt")),
        "SPACR_BENCHMARK_JSON": str(output),
        "SPACR_BENCHMARK_RUN": label,
        "SPACR_BENCHMARK_TIMEOUT_S": str(timeout_s),
        # The in-process QTimer cannot fire while the GUI thread itself is
        # wedged. A benchmark-only wall timer lives on a Python thread and
        # terminates that worker after the same per-state deadline plus a
        # short grace period, so a real hang fails in minutes rather than
        # waiting for this driver's whole-sweep timeout.
        "SPACR_BENCHMARK_HARD_TIMEOUT": "1",
        "SPACR_BENCHMARK_PACKAGE_ROOT": str(root),
    })
    if offscreen:
        env["QT_QPA_PLATFORM"] = "offscreen"
    return env


def _run_worker(home: Path, output: Path, label: str, timeout_s: float,
                offscreen: bool, *,
                package_root: Optional[Path] = None) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    env = _worker_environment(
        home, output, label, timeout_s, offscreen,
        package_root=package_root)
    # A registry currently has 44 apps.  The multiplier is intentionally
    # dynamic headroom rather than a second hard-coded app inventory; the
    # in-process controller owns the exact live key set and its JSON proves it.
    process_timeout = max(3600.0, timeout_s * 64.0)
    started = time.perf_counter()

    def _tail(value) -> str:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return str(value or "")[-4000:]

    def _append_failure(artifact: dict, message: str) -> None:
        benchmark = artifact.get("benchmark")
        if not isinstance(benchmark, dict):
            benchmark = {
                "run": label,
                "violations": [
                    "invalid worker artifact: benchmark is not a JSON object"],
            }
            artifact["benchmark"] = benchmark
        failures = benchmark.get("violations")
        if not isinstance(failures, list):
            failures = [
                "invalid worker artifact: benchmark.violations is not an array"]
            benchmark["violations"] = failures
        failures.append(message)

    try:
        completed = subprocess.run(
            [sys.executable, "-c", WORKER],
            cwd=str(home), env=env, capture_output=True, text=True,
            timeout=process_timeout,
        )
    except subprocess.TimeoutExpired as error:
        try:
            artifact = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            artifact = {"benchmark": {"run": label, "violations": []}}
        if not isinstance(artifact, dict):
            artifact = {
                "benchmark": {
                    "run": label,
                    "violations": [
                        "invalid worker artifact: root is not a JSON object"],
                }
            }
        _append_failure(
            artifact,
            f"worker exceeded its controlled {process_timeout:.0f} s timeout",
        )
        artifact["worker"] = {
            "returncode": None,
            "elapsed_s": time.perf_counter() - started,
            "stdout_tail": _tail(error.stdout),
            "stderr_tail": _tail(error.stderr),
        }
        return artifact

    elapsed = time.perf_counter() - started
    if output.is_file():
        try:
            artifact = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            artifact = {
                "benchmark": {
                    "run": label,
                    "violations": [f"invalid worker artifact: {error}"],
                }
            }
        if not isinstance(artifact, dict):
            artifact = {
                "benchmark": {
                    "run": label,
                    "violations": [
                        "invalid worker artifact: root is not a JSON object"],
                }
            }
    else:
        artifact = {
            "benchmark": {
                "run": label,
                "violations": ["worker produced no benchmark artifact"],
            }
        }
    artifact["worker"] = {
        "returncode": completed.returncode,
        "elapsed_s": elapsed,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }
    if completed.returncode != 0:
        _append_failure(
            artifact, f"worker exited with status {completed.returncode}")
    return artifact


def _path_is_within(value: object, root: Path) -> bool:
    if not value:
        return False
    try:
        path = Path(str(value)).expanduser().resolve()
        path.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _finite_number(value: object, *, minimum: Optional[float] = None,
                   maximum: Optional[float] = None) -> bool:
    """Return whether *value* is a finite non-boolean number in the range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    if not math.isfinite(number):
        return False
    if minimum is not None and number < minimum:
        return False
    if maximum is not None and number > maximum:
        return False
    return True


def _raw_stall_trace(stalls: object, reject) -> list[dict[str, float]]:
    """Validate and normalize the worker's raw event-loop watchdog trace."""
    if not isinstance(stalls, list):
        return []
    normalized: list[dict[str, float]] = []
    for index, row in enumerate(stalls):
        path = f"stalls[{index}]"
        if not isinstance(row, dict):
            reject(f"{path} is not a JSON object")
            continue
        valid = True
        for field in ("started_at", "at", "late_ms"):
            if not _finite_number(row.get(field), minimum=0.0):
                reject(f"{path}.{field} must be finite and non-negative")
                valid = False
        for field in ("source", "thread"):
            if not isinstance(row.get(field), str) or not row[field]:
                reject(f"{path}.{field} is missing or empty")
                valid = False
        if not valid:
            continue
        started_at = float(row["started_at"])
        ended_at = float(row["at"])
        late_ms = float(row["late_ms"])
        if ended_at < started_at:
            reject(f"{path}.at is before started_at")
            continue
        measured_ms = (ended_at - started_at) * 1000.0
        if not math.isclose(
                late_ms, measured_ms, rel_tol=1e-9, abs_tol=0.001):
            reject(f"{path}.late_ms does not match its timestamps")
            continue
        normalized.append({
            "started_at": started_at,
            "at": ended_at,
            "late_ms": late_ms,
        })
    return normalized


def _overlapping_stalls(stalls: Sequence[dict[str, float]],
                        started_at: float,
                        ended_at: float) -> list[tuple[float, float]]:
    """Return ``(clipped_ms, raw_ms)`` for gaps touching one result window."""
    if ended_at <= started_at:
        return []
    overlapping: list[tuple[float, float]] = []
    for row in stalls:
        overlap_ms = max(
            0.0,
            min(ended_at, row["at"])
            - max(started_at, row["started_at"]),
        ) * 1000.0
        if overlap_ms <= 0.0:
            continue
        overlapping.append((min(row["late_ms"], overlap_ms), row["late_ms"]))
    return overlapping


def _worker_schema_violations(
        artifact: object, expected_label: str,
        package_root: Optional[Path] = None) -> list[str]:
    """Validate one worker artifact without trusting its own verdict flags.

    The worker JSON is release evidence, not a best-effort diagnostic blob.
    Every field used to claim a painted, budgeted result is therefore checked
    here by the parent process.  This also makes an old, truncated or manually
    assembled artifact fail closed instead of inheriting ``passed=True`` from
    a handful of self-reported booleans.
    """
    prefix = f"{expected_label}:"
    violations: list[str] = []

    def reject(message: str) -> None:
        violations.append(f"{prefix} {message}")

    if not isinstance(artifact, dict):
        reject("worker artifact root is not a JSON object")
        return violations
    if (type(artifact.get("schema_version")) is not int
            or artifact.get("schema_version") != WORKER_SCHEMA_VERSION):
        reject(
            f"worker schema_version must be {WORKER_SCHEMA_VERSION}")
    if not _finite_number(artifact.get("elapsed_s"), minimum=0.0):
        reject("elapsed_s is missing or is not a finite non-negative number")

    budgets = artifact.get("budgets")
    expected_budgets = {
        "home_ready_s": HOME_BUDGET_S,
        "module_ready_s": MODULE_BUDGET_S,
        "max_event_loop_stall_ms": STALL_BUDGET_MS,
    }
    if not isinstance(budgets, dict):
        reject("budgets is not a JSON object")
        budgets = {}
    for field, expected in expected_budgets.items():
        if (not _finite_number(budgets.get(field), minimum=0.0)
                or float(budgets[field]) != expected):
            reject(f"budgets.{field} must equal {expected}")
    floor = budgets.get("watchdog_record_floor_ms")
    if (not _finite_number(floor, minimum=0.0)
            or (_finite_number(budgets.get("max_event_loop_stall_ms"))
                and float(floor) >= float(
                    budgets["max_event_loop_stall_ms"]))):
        reject(
            "budgets.watchdog_record_floor_ms must be finite, non-negative, "
            "and below the stall budget")

    environment = artifact.get("environment")
    if not isinstance(environment, dict):
        reject("environment is not a JSON object")
        environment = {}
    for field in (
            "python", "implementation", "platform", "machine", "qt",
            "executable", "spacr_version"):
        if not isinstance(environment.get(field), str) or not environment[field]:
            reject(f"environment.{field} is missing or empty")
    if not isinstance(environment.get("processor"), str):
        # Some supported platforms honestly return an empty processor string;
        # the field must still be present and typed rather than silently lost.
        reject("environment.processor is missing or is not a string")
    if (type(environment.get("pid")) is not int
            or environment.get("pid", 0) <= 0):
        reject("environment.pid is missing or is not a positive integer")
    root = package_root.expanduser().resolve() if package_root else None
    if root is not None:
        for field in ("spacr_file", "qt_package_file"):
            if not _path_is_within(environment.get(field), root):
                reject(f"{field} did not resolve inside {root}")
    else:
        for field in ("spacr_file", "qt_package_file"):
            if not isinstance(environment.get(field), str) or not environment[field]:
                reject(f"environment.{field} is missing or empty")

    hardware = environment.get("hardware")
    if not isinstance(hardware, dict):
        reject("environment.hardware is not a JSON object")
        hardware = {}
    if (type(hardware.get("logical_cpu_count")) is not int
            or hardware.get("logical_cpu_count", 0) <= 0):
        reject("environment.hardware.logical_cpu_count must be positive")
    if not _finite_number(hardware.get("total_memory_mb"), minimum=0.0):
        reject("environment.hardware.total_memory_mb must be reported")
    if (not isinstance(hardware.get("performance_level"), str)
            or not hardware.get("performance_level")):
        reject("environment.hardware.performance_level is missing or empty")
    if (not isinstance(hardware.get("qt_platform"), str)
            or not hardware.get("qt_platform")):
        reject("environment.hardware.qt_platform is missing or empty")
    displays = hardware.get("displays")
    if not isinstance(displays, list) or not displays:
        reject("environment.hardware.displays must contain at least one display")
        displays = []
    for index, display in enumerate(displays):
        path = f"environment.hardware.displays[{index}]"
        if not isinstance(display, dict):
            reject(f"{path} is not a JSON object")
            continue
        if not isinstance(display.get("name"), str):
            reject(f"{path}.name is missing or is not a string")
        for field in ("logical_width", "logical_height"):
            if (type(display.get(field)) is not int
                    or display.get(field, 0) <= 0):
                reject(f"{path}.{field} must be a positive integer")
        if not _finite_number(
                display.get("device_pixel_ratio"), minimum=0.000001):
            reject(f"{path}.device_pixel_ratio must be positive")
        if not _finite_number(display.get("refresh_hz"), minimum=0.0):
            reject(f"{path}.refresh_hz must be finite and non-negative")

    resources = artifact.get("resources")
    if not isinstance(resources, dict):
        reject("resources is not a JSON object")
        resources = {}
    if not _finite_number(resources.get("peak_rss_mb"), minimum=0.000001):
        reject("resources.peak_rss_mb must be reported and positive")
    gpu = resources.get("gpu")
    if not isinstance(gpu, dict):
        reject("resources.gpu is not a JSON object")
        gpu = {}
    for field in ("allocated_mb", "peak_allocated_mb"):
        if field not in gpu:
            reject(f"resources.gpu.{field} is missing")
            continue
        value = gpu.get(field)
        if value is not None and not _finite_number(value, minimum=0.0):
            reject(f"resources.gpu.{field} must be null or non-negative")
    allocated = gpu.get("allocated_mb")
    peak_allocated = gpu.get("peak_allocated_mb")
    if (_finite_number(allocated, minimum=0.0)
            and _finite_number(peak_allocated, minimum=0.0)
            and float(peak_allocated) < float(allocated)):
        reject("resources.gpu.peak_allocated_mb is below allocated_mb")

    for field in ("spans", "imports", "stalls", "marks", "readiness"):
        if not isinstance(artifact.get(field), list):
            reject(f"{field} is not a JSON array")
    raw_stalls = _raw_stall_trace(artifact.get("stalls"), reject)
    if not _finite_number(
            artifact.get("event_loop_started_at"), minimum=0.0):
        reject("event_loop_started_at was not recorded")
    worst_global = artifact.get("worst_event_loop_stall_ms")
    measured_worst_global = max(
        (row["late_ms"] for row in raw_stalls), default=0.0)
    if not _finite_number(worst_global, minimum=0.0):
        reject("worst_event_loop_stall_ms was not recorded")
    else:
        if not math.isclose(
                float(worst_global), measured_worst_global,
                rel_tol=1e-9, abs_tol=0.001):
            reject("worst_event_loop_stall_ms does not match raw stalls")
        if measured_worst_global >= STALL_BUDGET_MS:
            reject("worst_event_loop_stall_ms reached the 500 ms ceiling")
    measured_global_budget_met = measured_worst_global < STALL_BUDGET_MS
    if artifact.get("stall_budget_met") is not measured_global_budget_met:
        reject("stall_budget_met does not match raw stalls")
    if artifact.get("stall_budget_met") is not True:
        reject("stall_budget_met is not true")
    if artifact.get("import_timing_enabled") is not False:
        reject("import_timing_enabled must be false for an unbiased sweep")

    worker = artifact.get("worker")
    if not isinstance(worker, dict):
        reject("worker process evidence is missing")
        worker = {}
    if type(worker.get("returncode")) is not int or worker.get("returncode") != 0:
        reject("worker.returncode is not zero")
    if not _finite_number(worker.get("elapsed_s"), minimum=0.0):
        reject("worker.elapsed_s was not recorded")
    for field in ("stdout_tail", "stderr_tail"):
        if not isinstance(worker.get(field), str):
            reject(f"worker.{field} is missing or is not a string")

    benchmark = artifact.get("benchmark")
    if not isinstance(benchmark, dict):
        reject("benchmark is not a JSON object")
        return violations
    if benchmark.get("run") != expected_label:
        reject(f"benchmark.run must equal {expected_label!r}")
    if benchmark.get("exit_reason") != "registry sweep complete":
        reject("worker did not complete the registry sweep")
    reported_violations = benchmark.get("violations")
    if not isinstance(reported_violations, list):
        reject("benchmark.violations is not a JSON array")
    elif reported_violations:
        for message in reported_violations:
            reject(f"worker reported violation: {message}")

    keys = benchmark.get("registry_keys")
    keys_valid = (
        isinstance(keys, list)
        and bool(keys)
        and all(isinstance(key, str) and key for key in keys)
        and len(keys) == len(set(keys))
    )
    if not keys_valid:
        reject("benchmark.registry_keys must be non-empty, unique strings")
        keys = []
    if (type(benchmark.get("registry_count")) is not int
            or benchmark.get("registry_count") != len(keys)):
        reject("benchmark.registry_count does not equal registry_keys")
    if benchmark.get("final_registry_keys") != keys:
        reject("benchmark.final_registry_keys does not equal registry_keys")
    if benchmark.get("registry_stable") is not True:
        reject("benchmark.registry_stable is not true")
    if benchmark.get("measured_keys") != keys:
        reject("benchmark.measured_keys does not equal registry_keys")
    if (type(benchmark.get("measured_count")) is not int
            or benchmark.get("measured_count") != len(keys)):
        reject("benchmark.measured_count does not equal registry_keys")
    if benchmark.get("registry_matches_measurements") is not True:
        reject("measured app sequence did not equal its registry")
    if benchmark.get("preferences_measured") is not True:
        reject("benchmark.preferences_measured is not true")
    if (not _finite_number(benchmark.get("preferences_budget_s"), minimum=0.0)
            or float(benchmark["preferences_budget_s"])
            != PREFERENCES_BUDGET_S):
        reject(
            f"benchmark.preferences_budget_s must equal "
            f"{PREFERENCES_BUDGET_S}")

    expected_details = ["__home__", "__preferences__", *keys]
    results = benchmark.get("results")
    if not isinstance(results, list):
        reject("benchmark.results is not a JSON array")
        results = []
    if len(results) != len(expected_details):
        reject(
            "benchmark.results count does not equal Home + Preferences + "
            "the live registry")

    for index, expected_detail in enumerate(expected_details):
        if index >= len(results):
            break
        row = results[index]
        row_path = f"benchmark.results[{index}]"
        if not isinstance(row, dict):
            reject(f"{row_path} is not a JSON object")
            continue
        if row.get("detail") != expected_detail:
            reject(f"{row_path}.detail must equal {expected_detail!r}")
        expected_name = (
            "interactive Home" if expected_detail == "__home__"
            else "interactive preferences"
            if expected_detail == "__preferences__"
            else "interactive module"
        )
        if row.get("name") != expected_name:
            reject(f"{row_path}.name must equal {expected_name!r}")
        duration = row.get("duration_s")
        budget = row.get("budget_s")
        expected_budget = (
            HOME_BUDGET_S if expected_detail == "__home__"
            else PREFERENCES_BUDGET_S
            if expected_detail == "__preferences__"
            else MODULE_BUDGET_S
        )
        if not _finite_number(duration, minimum=0.0):
            reject(f"{row_path}.duration_s was not recorded")
        if (not _finite_number(budget, minimum=0.0)
                or float(budget) != expected_budget):
            reject(f"{row_path}.budget_s must equal {expected_budget}")
        if row.get("within_budget") is not True:
            reject(f"{row_path}.within_budget is not true")
        if (_finite_number(duration, minimum=0.0)
                and _finite_number(budget, minimum=0.0)
                and float(duration) > float(budget)):
            reject(f"{row_path}.duration_s exceeds its declared budget")
        worst = row.get("worst_event_loop_stall_ms")
        raw_worst = row.get("worst_overlapping_frame_interval_ms")
        window_start = row.get("stall_window_started_at")
        window_end = row.get("stall_window_ended_at")
        window_valid = True
        if not _finite_number(window_start, minimum=0.0):
            reject(f"{row_path}.stall_window_started_at was not recorded")
            window_valid = False
        if not _finite_number(window_end, minimum=0.0):
            reject(f"{row_path}.stall_window_ended_at was not recorded")
            window_valid = False
        if (window_valid and float(window_end) < float(window_start)):
            reject(f"{row_path}.stall window ends before it starts")
            window_valid = False
        if (window_valid
                and _finite_number(artifact.get("elapsed_s"), minimum=0.0)
                and float(window_end) > float(artifact["elapsed_s"])):
            reject(f"{row_path}.stall window ends after the artifact")
            window_valid = False
        overlaps = (
            _overlapping_stalls(
                raw_stalls, float(window_start), float(window_end))
            if window_valid else []
        )
        measured_worst = max(
            (clipped for clipped, _raw in overlaps), default=0.0)
        measured_raw_worst = max(
            (raw for _clipped, raw in overlaps), default=0.0)
        if not _finite_number(worst, minimum=0.0):
            reject(f"{row_path}.worst_event_loop_stall_ms was not recorded")
        else:
            if (window_valid and not math.isclose(
                    float(worst), measured_worst,
                    rel_tol=1e-9, abs_tol=0.001)):
                reject(
                    f"{row_path}.worst_event_loop_stall_ms does not match "
                    "raw stalls")
            if measured_worst >= STALL_BUDGET_MS:
                reject(
                    f"{row_path} reached the 500 ms event-loop stall ceiling")
        if not _finite_number(raw_worst, minimum=0.0):
            reject(
                f"{row_path}.worst_overlapping_frame_interval_ms was not "
                "recorded")
        else:
            if (window_valid and not math.isclose(
                    float(raw_worst), measured_raw_worst,
                    rel_tol=1e-9, abs_tol=0.001)):
                reject(
                    f"{row_path}.worst_overlapping_frame_interval_ms does "
                    "not match raw stalls")
            if (_finite_number(worst, minimum=0.0)
                    and float(raw_worst) < float(worst)):
                reject(
                    f"{row_path}.worst_overlapping_frame_interval_ms is below "
                    "the clipped stall")
        measured_interval_budget_met = measured_worst < STALL_BUDGET_MS
        if (row.get("event_loop_stall_budget_met")
                is not measured_interval_budget_met):
            reject(
                f"{row_path}.event_loop_stall_budget_met does not match "
                "raw stalls")
        if row.get("event_loop_stall_budget_met") is not True:
            reject(f"{row_path}.event_loop_stall_budget_met is not true")
        if (type(row.get("stall_samples")) is not int
                or row.get("stall_samples", -1) < 0):
            reject(f"{row_path}.stall_samples was not recorded")
        elif window_valid and row["stall_samples"] != len(overlaps):
            reject(f"{row_path}.stall_samples does not match raw stalls")
        if row.get("error"):
            reject(f"{row_path} contains an error")

        # Preferences has its own shown-dialog observation. Home and every
        # registry app, however, may pass only with the stronger production
        # probe: an event-loop callback followed by a painted usable control.
        if expected_detail != "__preferences__":
            for field in ("at", "started_at", "event_loop_started_at"):
                if not _finite_number(row.get(field), minimum=0.0):
                    reject(f"{row_path}.{field} was not recorded")
            at = row.get("at")
            started_at = row.get("started_at")
            event_loop_at = row.get("event_loop_started_at")
            if (_finite_number(at, minimum=0.0)
                    and _finite_number(started_at, minimum=0.0)
                    and float(started_at) > float(at)):
                reject(f"{row_path}.started_at is after readiness")
            if (_finite_number(at, minimum=0.0)
                    and _finite_number(event_loop_at, minimum=0.0)
                    and float(event_loop_at) > float(at)):
                reject(f"{row_path}.event_loop_started_at is after readiness")
            if (_finite_number(at, minimum=0.0)
                    and _finite_number(started_at, minimum=0.0)
                    and _finite_number(duration, minimum=0.0)
                    and not math.isclose(
                        float(duration), float(at) - float(started_at),
                        rel_tol=1e-9, abs_tol=1e-6)):
                reject(f"{row_path}.duration_s does not match its timestamps")
            if not isinstance(row.get("root_painted"), bool):
                reject(f"{row_path}.root_painted is missing")
            if row.get("screen_tree_painted") is not True:
                reject(f"{row_path}.screen_tree_painted is not true")
            painted = row.get("painted_usable_controls")
            usable = row.get("usable_controls")
            if type(painted) is not int or painted <= 0:
                reject(f"{row_path}.painted_usable_controls is not positive")
            if type(usable) is not int or usable <= 0:
                reject(f"{row_path}.usable_controls is not positive")
            elif type(painted) is int and usable < painted:
                reject(f"{row_path}.usable_controls is below painted controls")
            controls = row.get("controls")
            if (not isinstance(controls, list) or not controls
                    or not all(isinstance(value, str) and value
                               for value in controls)):
                reject(f"{row_path}.controls lacks painted control names")
            if row.get("thread") != "MainThread":
                reject(f"{row_path}.thread is not MainThread")

    readiness = artifact.get("readiness")
    expected_readiness = ["__home__", *keys]
    if isinstance(readiness, list):
        readiness_details = [
            row.get("detail") if isinstance(row, dict) else None
            for row in readiness
        ]
        if readiness_details != expected_readiness:
            reject(
                "readiness sequence does not equal Home + the live registry")
        result_by_detail = {
            row.get("detail"): row for row in results
            if isinstance(row, dict)
        }
        evidence_fields = (
            "at", "started_at", "duration_s", "name", "detail", "budget_s",
            "within_budget", "event_loop_started_at", "root_painted",
            "screen_tree_painted", "painted_usable_controls",
            "usable_controls", "controls", "thread",
        )
        for index, row in enumerate(readiness):
            if not isinstance(row, dict):
                continue
            detail = row.get("detail")
            result = result_by_detail.get(detail)
            if not isinstance(result, dict):
                continue
            for field in evidence_fields:
                if row.get(field) != result.get(field):
                    reject(
                        f"readiness[{index}].{field} does not match its "
                        "benchmark result")

    return violations


def _combined_violations(
    runs: Sequence[dict], package_root: Optional[Path] = None) -> list[str]:
    violations: list[str] = []
    registries: list[list[str]] = []
    if len(runs) not in (1, 2):
        violations.append(
            "combined artifact must contain one cold run or cold + warm runs")
    expected_labels = ("cold-process", "warm-process")
    for index, artifact in enumerate(runs):
        label = (
            expected_labels[index] if index < len(expected_labels)
            else f"unexpected-process-{index + 1}"
        )
        violations.extend(
            _worker_schema_violations(artifact, label, package_root))
        if not isinstance(artifact, dict):
            continue
        benchmark = artifact.get("benchmark", {})
        if not isinstance(benchmark, dict):
            continue
        keys = benchmark.get("registry_keys")
        if (isinstance(keys, list)
                and all(isinstance(key, str) for key in keys)):
            registries.append([str(key) for key in keys])
    if len(registries) != len(runs):
        violations.append("one or more runs did not report the live registry")
    elif any(keys != registries[0] for keys in registries[1:]):
        violations.append("the live registry changed between cold and warm runs")
    return violations


def run_benchmark(output: Path, *, runs: int = 2, timeout_s: float = 30.0,
                  offscreen: bool = False,
                  package_root: Optional[Path] = None) -> dict:
    """Run fresh cold/warm processes and return their combined artifact."""
    if type(runs) is not int or runs not in (1, 2):
        raise ValueError("runs must be one cold process or cold + warm processes")
    output = output.expanduser().resolve()
    package_root = (package_root or PACKAGE_ROOT).expanduser().resolve()
    if not (package_root / "spacr" / "__init__.py").is_file():
        raise ValueError(
            f"package root {package_root} does not contain spacr/__init__.py")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spacr-startup-benchmark-") as raw:
        home = Path(raw) / "home"
        home.mkdir(parents=True)
        labels = ("cold-process", "warm-process")[:runs]
        artifacts = []
        for index, label in enumerate(labels):
            worker_output = output.with_name(
                f"{output.stem}.{index + 1}-{label}.json")
            artifacts.append(_run_worker(
                home, worker_output, label, timeout_s, offscreen,
                package_root=package_root))

    violations = _combined_violations(artifacts, package_root)
    registries = []
    for run in artifacts:
        if not isinstance(run, dict):
            continue
        benchmark = run.get("benchmark")
        if not isinstance(benchmark, dict):
            continue
        keys = benchmark.get("registry_keys")
        if isinstance(keys, list):
            registries.append(keys)
    combined = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "driver": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "offscreen": bool(offscreen),
            "runs": runs,
            "per_state_timeout_s": timeout_s,
            "package_root": str(package_root),
        },
        "registry_keys": registries[0] if registries else [],
        "registry_count": len(registries[0]) if registries else 0,
        "runs": artifacts,
        "violations": violations,
        "passed": not violations,
    }
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(combined, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(str(temporary), str(output))
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
    return combined


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None,
                        help="combined JSON path (default: ~/.spacr/reports)")
    parser.add_argument("--runs", type=int, choices=(1, 2), default=2,
                        help="one cold process or cold + warm processes")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="controlled timeout for each Home/module state")
    parser.add_argument("--offscreen", action="store_true",
                        help=("use Qt's offscreen platform for diagnostics; "
                              "it is not real-display release evidence"))
    parser.add_argument(
        "--package-root", type=Path, default=PACKAGE_ROOT,
        help=("directory containing the exact spacr package to measure; pass "
              "the clean environment's site-packages for a wheel audit"),
    )
    parser.add_argument("--record-only", action="store_true",
                        help="write budget misses without a failing exit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if not 1.0 <= args.timeout <= 300.0:
        raise SystemExit("--timeout must be between 1 and 300 seconds")
    output = (args.out or _default_output()).expanduser().resolve()
    artifact = run_benchmark(
        output, runs=args.runs, timeout_s=args.timeout,
        offscreen=args.offscreen, package_root=args.package_root)
    print(f"spaCR startup benchmark written to {output}")
    print(
        f"registry: {artifact['registry_count']} apps; "
        f"violations: {len(artifact['violations'])}")
    for violation in artifact["violations"]:
        print(f"  - {violation}")
    return 0 if args.record_only or artifact["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
