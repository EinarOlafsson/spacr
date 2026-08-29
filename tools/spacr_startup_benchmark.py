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
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

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
        artifact.setdefault("benchmark", {}).setdefault(
            "violations", []).append(
                f"worker exceeded its controlled {process_timeout:.0f} s timeout")
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
        artifact.setdefault("benchmark", {}).setdefault(
            "violations", []).append(
                f"worker exited with status {completed.returncode}")
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


def _combined_violations(
        runs: Sequence[dict], package_root: Optional[Path] = None) -> list[str]:
    violations: list[str] = []
    registries: list[list[str]] = []
    root = package_root.expanduser().resolve() if package_root else None
    for artifact in runs:
        benchmark = artifact.get("benchmark", {})
        label = str(benchmark.get("run", "run"))
        if benchmark.get("exit_reason") != "registry sweep complete":
            violations.append(
                f"{label}: worker did not complete the registry sweep")
        violations.extend(
            f"{label}: {message}"
            for message in benchmark.get("violations", ())
        )
        keys = benchmark.get("registry_keys")
        if isinstance(keys, list):
            registries.append([str(key) for key in keys])
        if benchmark.get("registry_matches_measurements") is not True:
            violations.append(
                f"{label}: measured app sequence did not equal its registry")
        if root is not None:
            environment = artifact.get("environment", {})
            for field in ("spacr_file", "qt_package_file"):
                if not _path_is_within(environment.get(field), root):
                    violations.append(
                        f"{label}: {field} did not resolve inside {root}")
    if len(registries) != len(runs):
        violations.append("one or more runs did not report the live registry")
    elif any(keys != registries[0] for keys in registries[1:]):
        violations.append("the live registry changed between cold and warm runs")
    return violations


def run_benchmark(output: Path, *, runs: int = 2, timeout_s: float = 30.0,
                  offscreen: bool = False,
                  package_root: Optional[Path] = None) -> dict:
    """Run fresh cold/warm processes and return their combined artifact."""
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
    registries = [
        run.get("benchmark", {}).get("registry_keys") for run in artifacts
        if isinstance(run.get("benchmark", {}).get("registry_keys"), list)
    ]
    combined = {
        "schema_version": 1,
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
                        help="use Qt's offscreen platform (CI/headless only)")
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
