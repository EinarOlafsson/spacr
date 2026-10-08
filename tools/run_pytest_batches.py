#!/usr/bin/env python3
"""Run a pytest suite in bounded file batches.

Long-lived pytest workers retain imported scientific libraries, fitted models,
and plotting state.  Splitting the file list across fresh pytest processes
keeps that accumulated resident memory below the hosted-runner limit while
preserving the same marker selection and test coverage.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

NO_TESTS_COLLECTED = 5


#: Exit status reported for a batch killed by ``--batch-timeout``
#: (the same number GNU ``timeout`` uses).
BATCH_TIMED_OUT = 124


def _run_bounded(command, *, env=None, timeout: float = 0):
    """Run ``command``; past ``timeout`` seconds kill its whole process group.

    A per-test timeout only sees a test's own phases. Run 37245630937 lost
    hours to time spent OUTSIDE any test -- a pytest-xdist controller moving
    a one-megabyte test id, a pool join in teardown -- so the batch as a
    whole also needs a ceiling. The process group matters: killing only the
    pytest controller would orphan its xdist workers.
    """
    if not timeout or timeout <= 0:
        kwargs = {"check": False}
        if env is not None:
            kwargs["env"] = env
        return subprocess.run(command, **kwargs)
    import signal as _signal

    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(
            f"::error title=Batch timed out::killed after {timeout:g} s: "
            + " ".join(str(part) for part in command),
            flush=True,
        )
        for sig, grace in ((_signal.SIGTERM, 15), (_signal.SIGKILL, 30)):
            try:
                os.killpg(process.pid, sig)
            except (ProcessLookupError, PermissionError):
                break
            try:
                process.wait(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue
        return subprocess.CompletedProcess(command, BATCH_TIMED_OUT)
    return subprocess.CompletedProcess(command, process.returncode)


def _test_files(paths: Sequence[str]) -> list[str]:
    """Return the sorted, de-duplicated test files below ``paths``."""
    found: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            found.add(path)
        elif path.is_dir():
            found.update(path.rglob("test_*.py"))
        else:
            raise FileNotFoundError(f"test path does not exist: {path}")
    return [str(path) for path in sorted(found, key=lambda item: str(item))]


def _batches(items: Sequence[str], size: int) -> list[list[str]]:
    """Split ``items`` into non-empty lists of at most ``size`` entries."""
    if size < 1:
        raise ValueError("batch size must be at least 1")
    return [list(items[start:start + size])
            for start in range(0, len(items), size)]


def _timeout_plugin_available() -> bool:
    """Whether ``pytest-timeout`` is importable in this interpreter.

    Checked rather than assumed: ``--timeout`` is that plugin's option and
    pytest rejects an unknown one before collecting anything, so passing it
    blind turns a missing diagnostic into a job that runs no tests at all.
    """
    try:
        return importlib.util.find_spec("pytest_timeout") is not None
    except (ImportError, ValueError):
        return False


def _qt_batches_are_excluded(marker: str) -> bool:
    """Accept only a conjunction that explicitly excludes the Qt marker."""
    terms = [term.strip() for term in marker.split(" and ")]
    return "not qt" in terms and all(
        re.fullmatch(r"not [A-Za-z_][A-Za-z_0-9]*", term)
        for term in terms
    )


def _only_automatically_qt_files(batch: Sequence[str]) -> bool:
    """Recognize a batch whose files all get the repository's Qt marker."""
    root = Path.cwd().resolve()
    for raw_path in batch:
        try:
            relative = Path(raw_path).resolve().relative_to(root)
        except ValueError:
            return False
        if relative.parts[:2] != ("tests", "qt"):
            return False
    return bool(batch)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", default=["tests"],
        help="Test files or directories (default: tests).",
    )
    parser.add_argument(
        "--marker", required=True,
        help="Pytest marker expression applied to every batch.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Maximum files per fresh pytest process (default: 32).",
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="xdist workers within each batch (default: 2).",
    )
    parser.add_argument(
        "--ignore", action="append", default=[], metavar="PATH",
        help="Exclude a file or directory before batching; repeat as needed.",
    )
    parser.add_argument(
        "--faulthandler-timeout", type=int, default=0, metavar="SECONDS",
        help="Pytest faulthandler timeout; 0 retains pytest's configured value.",
    )
    parser.add_argument(
        "--per-test-timeout", type=int, default=0, metavar="SECONDS",
        help=(
            "Kill any single test that runs longer than SECONDS and report "
            "it by name (default: 0, no ceiling). Needs pytest-timeout."
        ),
    )
    parser.add_argument(
        "--batch-timeout", type=float, default=0, metavar="SECONDS",
        help=(
            "Kill a whole batch (controller and workers) that runs longer "
            f"than SECONDS and report exit {BATCH_TIMED_OUT} (default: 0, "
            "no ceiling)."
        ),
    )
    parser.add_argument(
        "--skip-qt-only-batches", action="store_true",
        help="skip global batches containing only automatically Qt-marked "
        "files when --marker is a conjunction excluding qt",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run EVERY selected batch and return the first failing exit status.

    EVERY, and that word is the whole of this function's contract. It used
    to return the moment a batch failed, which reads as a reasonable
    economy and is not: the batches partition the suite, so stopping at
    the first failure means the batches after it never run at all.

    Measured on one commit: this stopped at batch 19 of 54, so thirty-five
    batches -- about two thirds of the partition -- were never executed,
    and the job reported "one failure". A file in batch 39 had three real
    failures that had gone unreported for days, because no run ever
    reached it. A CI job that describes a PREFIX of the suite while
    looking like a verdict on all of it is worse than one that is simply
    slow.
    """
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    if args.skip_qt_only_batches and not _qt_batches_are_excluded(args.marker):
        raise ValueError(
            "--skip-qt-only-batches requires a conjunction of negative "
            "markers including 'not qt'"
        )

    ignored = [Path(path).resolve() for path in args.ignore]
    files = [path for path in _test_files(args.paths)
             if not any(Path(path).resolve() == excluded
                        or excluded in Path(path).resolve().parents
                        for excluded in ignored)]
    batches = _batches(files, args.batch_size)
    if not batches:
        raise FileNotFoundError("no test_*.py files found")

    failed: list = []
    for number, batch in enumerate(batches, start=1):
        if args.skip_qt_only_batches and _only_automatically_qt_files(batch):
            print(
                f"pytest batch {number}/{len(batches)}: {len(batch)} files; "
                "all automatically qt-marked and excluded by --marker",
                flush=True,
            )
            continue
        print(
            f"pytest batch {number}/{len(batches)}: {len(batch)} files",
            flush=True,
        )
        command = [
            sys.executable, "-m", "pytest", *batch,
            "-m", args.marker,
        ]
        if args.workers > 1:
            command.extend([
                "-n", str(args.workers), "--dist", "loadfile",
            ])
        if args.per_test_timeout > 0 and not _timeout_plugin_available():
            # SAY IT ONCE, LOUDLY, AND STILL RUN. `--timeout` is pytest-timeout's
            # option, not pytest's, so passing it without the plugin makes
            # pytest exit with "unrecognized arguments" -- EVERY batch, before
            # a single test runs. A ceiling is a diagnostic; the suite is the
            # job. Failing the whole run because the diagnostic is unavailable
            # would be a worse trade than losing the diagnostic.
            #
            # It is not declared in setup.py, pyproject.toml or any
            # requirements file: `_pytest-suite.yml` pip-installs it for the Qt
            # suite and nothing installs it for this one.
            if not getattr(main, "_said_no_timeout_plugin", False):
                print("run_pytest_batches: pytest-timeout is not installed, so "
                      "--per-test-timeout has no effect. A test that wedges "
                      "will use the whole job budget and name nothing. "
                      "`pip install \"pytest-timeout>=2.3,<3\"` to arm it.",
                      flush=True)
                main._said_no_timeout_plugin = True
        elif args.per_test_timeout > 0:
            # A JOB THAT IS KILLED BY ITS OWN BUDGET NAMES NOTHING. The Qt
            # suite learned this the expensive way and its comment records
            # it: a shard "reached 99%, then sat in silence until the
            # runner killed it", so the log named neither the test nor a
            # stack. A per-test ceiling turns a lost job into one named
            # failure.
            #
            # MEASURED HERE, and it is why this flag exists: on two
            # independent DISPATCHED runs -- which are exempt from the
            # cancellation storm, so they were not killed by a push --
            # `Fast / Full suite control` was cancelled at 1:30:17 and
            # again at 1:30:32 against a 90-minute budget, and `Minimum
            # dependencies` at 2:01:09 against 120 minutes. Both are
            # blocking jobs, so the branch could not go green whatever the
            # tests said, and neither run left a record of where the time
            # went.
            command.extend(["--timeout", str(args.per_test_timeout),
                            "--timeout-method", "thread"])
            if args.workers > 1:
                # --max-worker-restart=0 is what keeps the ceiling from
                # costing MORE than the hang it replaces: xdist otherwise
                # hands the very same test to a replacement worker, which
                # wedges again, so a deterministic hang pays the ceiling
                # once per restart. The same reasoning, and the same flag,
                # as `_pytest-suite.yml`.
                command.append("--max-worker-restart=0")
        if args.faulthandler_timeout > 0:
            command.extend([
                "-o", f"faulthandler_timeout={args.faulthandler_timeout}",
            ])
        command.extend(["-v", "--tb=short"])
        result = _run_bounded(command, timeout=args.batch_timeout)
        if result.returncode not in (0, NO_TESTS_COLLECTED):
            # REMEMBERED, NOT RETURNED. The first failing status is what
            # the job exits with, so the signal is unchanged; what changes
            # is that the remaining batches still run and their failures
            # are still reported.
            failed.append((number, int(result.returncode), batch))
    if failed:
        # A BATCH NUMBER IS NOT A FILE NAME. Reading one back means
        # re-deriving the sorted file list and slicing it by the batch
        # size, which nobody does. Name the files instead: a batch that
        # ends in a segfault or a runner timeout prints no pytest summary
        # at all, so this list is the only record of what it was running.
        for number, code, batch in failed:
            print(
                f"batch {number} (exit {code}) ran: " + " ".join(batch),
                flush=True,
            )
        print(
            f"{len(failed)} of {len(batches)} batches failed: "
            + ", ".join(
                f"batch {number} (exit {code})"
                for number, code, _files in failed
            ),
            flush=True,
        )
        return failed[0][1]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
