#!/usr/bin/env python3
"""Run a pytest suite in bounded file batches.

Long-lived pytest workers retain imported scientific libraries, fitted models,
and plotting state.  Splitting the file list across fresh pytest processes
keeps that accumulated resident memory below the hosted-runner limit while
preserving the same marker selection and test coverage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Sequence


NO_TESTS_COLLECTED = 5


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

    files = _test_files(args.paths)
    batches = _batches(files, args.batch_size)
    if not batches:
        raise FileNotFoundError("no test_*.py files found")

    failed: list = []
    for number, batch in enumerate(batches, start=1):
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
        command.extend(["-v", "--tb=short"])
        result = subprocess.run(command, check=False)
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
