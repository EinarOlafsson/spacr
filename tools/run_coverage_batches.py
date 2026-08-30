#!/usr/bin/env python3
"""Run one deterministic CI coverage shard in fresh pytest batches.

Scientific and Qt tests retain substantial process state.  Each batch gets a
fresh pair of workers and a unique coverage data basename, so the hosted
runner stays bounded and the combine job can merge every process safely.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from coverage import CoverageData
from coverage.exceptions import CoverageException

NO_TESTS_COLLECTED = 5


def _test_files(root: Path, paths: Sequence[str]) -> list[Path]:
    found: set[Path] = set()
    for raw_path in paths:
        path = (root / raw_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"test path escapes repository: {raw_path}") from exc
        if path.is_file():
            found.add(path)
        elif path.is_dir():
            found.update(path.rglob("test_*.py"))
        else:
            raise FileNotFoundError(f"test path does not exist: {raw_path}")
    return sorted(found, key=lambda item: item.relative_to(root).as_posix())


def _shard(path: Path, root: Path, count: int) -> int:
    label = path.relative_to(root).as_posix()
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


def _batches(items: Sequence[Path], size: int) -> list[list[Path]]:
    return [list(items[start:start + size]) for start in range(0, len(items), size)]


def _discard_unreadable_process_data(data_dir: Path) -> list[Path]:
    """Remove coverage files left incomplete by terminated child processes.

    coverage.py normally combines every ``.coverage.*`` file it finds.  A
    process killed while coverage is opening its database can leave behind a
    valid SQLite shell with no coverage metadata, however.  ``coverage
    combine`` warns about that shell but leaves it in the downloaded artifact.
    Validate files after every batch run so only readable process data reaches
    the combine job; the module ratchet still detects any measurements the
    terminated process failed to record.
    """
    discarded: list[Path] = []
    for path in sorted(data_dir.glob(".coverage.*")):
        if not path.is_file():
            continue
        try:
            CoverageData(basename=str(path)).read()
        except CoverageException as exc:
            print(
                f"discarding unreadable coverage process data {path.name}: "
                f"{exc}",
                flush=True,
            )
            path.unlink()
            discarded.append(path)
    return discarded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", default=["tests"],
        help="test files or directories (default: tests)",
    )
    parser.add_argument("--marker", required=True, help="pytest marker expression")
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--exclude-file", action="append", default=[],
        help="repository-relative measurement file excluded from tracing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd().resolve()
    if args.shard_count < 1:
        raise ValueError("shard count must be at least 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError(
            f"shard index must be within [0, {args.shard_count}); "
            f"got {args.shard_index}"
        )
    if args.batch_size < 1 or args.workers < 1:
        raise ValueError("batch size and worker count must be at least 1")

    excluded = {Path(value).as_posix() for value in args.exclude_file}
    selected = [
        path for path in _test_files(root, args.paths)
        if path.relative_to(root).as_posix() not in excluded
        and _shard(path, root, args.shard_count) == args.shard_index
    ]
    batches = _batches(selected, args.batch_size)
    if not batches:
        raise FileNotFoundError(
            f"coverage shard {args.shard_index} selected no test files"
        )

    args.data_dir.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[int, int, list[Path]]] = []
    for number, batch in enumerate(batches, start=1):
        relative = [path.relative_to(root).as_posix() for path in batch]
        print(
            f"coverage shard {args.shard_index}, batch {number}/"
            f"{len(batches)}: {len(batch)} files",
            flush=True,
        )
        data_name = (
            f".coverage.shard-{args.shard_index:02d}.batch-{number:03d}"
        )
        environment = os.environ.copy()
        environment["COVERAGE_FILE"] = str(args.data_dir / data_name)
        command = [
            sys.executable,
            "-m",
            "pytest",
            *relative,
            "-m",
            args.marker,
            "--cov=spacr",
            "--cov-branch",
            "--cov-report=",
            "--cov-config=.coveragerc",
            "-o",
            "faulthandler_timeout=900",
        ]
        if args.workers > 1:
            command.extend(["-n", str(args.workers), "--dist", "loadfile"])
        command.extend(["-v", "--tb=short"])
        result = subprocess.run(command, env=environment, check=False)
        if result.returncode not in (0, NO_TESTS_COLLECTED):
            failures.append((number, int(result.returncode), batch))

    _discard_unreadable_process_data(args.data_dir)

    if failures:
        for number, code, batch in failures:
            names = " ".join(path.relative_to(root).as_posix() for path in batch)
            print(f"coverage batch {number} (exit {code}) ran: {names}")
        return failures[0][1]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
