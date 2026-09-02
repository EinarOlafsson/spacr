#!/usr/bin/env python3
"""Run one command and stop its process tree before RAM exhaustion.

The workstation has 125 GiB of RAM and also hosts the interactive desktop.
Long test and coverage runs must use this wrapper so VS Code is not sacrificed
to an accumulating worker leak.  The default policy stops the guarded command
when total system RAM in use reaches 115 GiB, leaving roughly 10 GiB for the
desktop and the operating system.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

DEFAULT_LIMIT_GIB = 115.0
DEFAULT_POLL_SECONDS = 1.0
DEFAULT_GRACE_SECONDS = 10.0
MEMORY_ABORT_EXIT = 137


def memory_usage_gib() -> tuple[float, float]:
    """Return ``(total, used)`` physical memory in GiB.

    ``used`` is total minus the kernel's available-memory estimate, rather
    than total minus ``free``.  File-system cache is reclaimable and must not
    make a safe process look as if it exhausted RAM.
    """
    try:
        import psutil

        reading = psutil.virtual_memory()
        total = float(reading.total)
        available = float(reading.available)
    except (ImportError, AttributeError, OSError):
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, raw = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                values[key] = int(raw.split()[0]) * 1024
        total = float(values["MemTotal"])
        available = float(values["MemAvailable"])
    gib = float(1024 ** 3)
    return total / gib, max(0.0, total - available) / gib


def _signal_tree(process: subprocess.Popen, sig: signal.Signals) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(process.pid), sig)
        except ProcessLookupError:
            return
        return

    # Windows has no POSIX process group to signal. psutil is a direct spaCR
    # dependency and lets the guard stop descendants before their parent.
    try:
        import psutil

        parent = psutil.Process(process.pid)
        members = parent.children(recursive=True) + [parent]
        action = "terminate" if sig == signal.SIGTERM else "kill"
        for member in reversed(members):
            try:
                getattr(member, action)()
            except psutil.NoSuchProcess:
                pass
    except (ImportError, OSError):
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()


def terminate_process_tree(
    process: subprocess.Popen,
    *,
    grace_seconds: float = DEFAULT_GRACE_SECONDS,
) -> None:
    """Send TERM to the guarded tree, then KILL after a bounded grace."""
    _signal_tree(process, signal.SIGTERM)
    try:
        process.wait(timeout=max(0.0, float(grace_seconds)))
    except subprocess.TimeoutExpired:
        _signal_tree(process, signal.SIGKILL)
        process.wait()


def run_guarded(
    command: Sequence[str],
    *,
    limit_gib: float = DEFAULT_LIMIT_GIB,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    grace_seconds: float = DEFAULT_GRACE_SECONDS,
    memory_reader: Callable[[], tuple[float, float]] = memory_usage_gib,
) -> int:
    """Run ``command`` and return its exit code, or 137 after a RAM abort."""
    if not command:
        raise ValueError("a command is required")
    if limit_gib <= 0:
        raise ValueError("limit_gib must be positive")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")

    total_gib, used_gib = memory_reader()
    if used_gib >= limit_gib:
        print(
            f"memory guard: refusing to start at {used_gib:.1f} GiB used "
            f"(limit {limit_gib:.1f} GiB; total {total_gib:.1f} GiB)",
            file=sys.stderr,
            flush=True,
        )
        return MEMORY_ABORT_EXIT

    popen_options = {"start_new_session": True} if os.name == "posix" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
    }
    try:
        process = subprocess.Popen(list(command), **popen_options)
    except FileNotFoundError as error:
        print(f"memory guard: {error}", file=sys.stderr)
        return 127

    try:
        while process.poll() is None:
            total_gib, used_gib = memory_reader()
            if used_gib >= limit_gib:
                print(
                    f"memory guard: stopping pid {process.pid}: system RAM "
                    f"reached {used_gib:.1f} GiB used (limit "
                    f"{limit_gib:.1f} GiB; total {total_gib:.1f} GiB)",
                    file=sys.stderr,
                    flush=True,
                )
                terminate_process_tree(
                    process, grace_seconds=grace_seconds,
                )
                return MEMORY_ABORT_EXIT
            time.sleep(float(poll_seconds))
    except KeyboardInterrupt:
        terminate_process_tree(process, grace_seconds=grace_seconds)
        return 130
    return int(process.returncode or 0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-gib", type=float, default=DEFAULT_LIMIT_GIB)
    parser.add_argument(
        "--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS,
    )
    parser.add_argument(
        "--grace-seconds", type=float, default=DEFAULT_GRACE_SECONDS,
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    command = list(arguments.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        _parser().error("put the command after --")
    return run_guarded(
        command,
        limit_gib=arguments.limit_gib,
        poll_seconds=arguments.poll_seconds,
        grace_seconds=arguments.grace_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
