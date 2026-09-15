#!/usr/bin/env python3
"""Run one deterministic CI coverage shard in fresh pytest batches.

Scientific and Qt tests retain substantial process state.  Each batch gets a
fresh pair of workers and a unique coverage data basename, so the hosted
runner stays bounded and the combine job can merge every process safely.

A CRASHED WORKER'S COVERAGE IS RECOVERED OR NAMED, NEVER SILENTLY DROPPED
(item 288).  pytest-cov writes a worker's data when that worker's session
finishes, so a worker that dies by a signal takes EVERY line it measured with
it -- including the lines of tests it had already reported as passed.
Measured on a toy package: a test passed on gw0, gw0 segfaulted in a later
test, and the combined data held none of that passed test's lines.  On
2026-09-15 two coverage shards lost a worker that way and the ratchet then
reported thirteen modules as regressions (pivot_spec.py "from 100% to 24
uncovered statements") that no code change had caused.

So every batch loads ``tools/pytest_plugins/xdist_coverage_ledger.py``, which
records which test files each worker reported on and whether pytest-cov got
that worker's data back.  After the batch, every file whose data is missing
is re-run ON ITS OWN, serially, in a fresh process with ``--cov-append`` and
the same coverage options, writing to its own data file, so a file that
crashes again cannot take anything else with it.  The outcome is written to
``spacr-coverage-integrity.shard-NN.json`` in the data directory after EVERY
batch, so a shard that is killed part-way still says how far it got.  The
gate (``tools/verify_module_coverage.py --shard-integrity``) reads those
records and fails a run whose losses were not all recovered as an
INCOMPLETE MEASUREMENT, which is a different verdict from a regression.

Recovering coverage does not hide the crash: the batch's own exit status is
still the job's, and a file that cannot be recovered fails the job too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from coverage import CoverageData
from coverage.exceptions import CoverageException

NO_TESTS_COLLECTED = 5
#: pytest exit statuses after which pytest-cov has saved its data: passed,
#: tests failed, no tests collected.  A signal (negative), an interrupt (2),
#: an internal error (3) or a usage error (4) proves nothing was saved.
DATA_SAVED_STATUSES = (0, 1, NO_TESTS_COLLECTED)
UNRECOVERED_STATUS = 1
#: tests/conftest.py ends a pytest process whose RSS passes
#: SPACR_TEST_MEMORY_GB (default 6) with ``os._exit(3)``.
MEMORY_GUARD_STATUS = 3

PLUGIN_DIR = Path(__file__).resolve().parent / "pytest_plugins"
PLUGIN = "xdist_coverage_ledger"
LEDGER_ENV = "SPACR_COVERAGE_LEDGER"
LEDGER_SCHEMA = "spacr.xdist-coverage-ledger/v1"
INTEGRITY_SCHEMA = "spacr.coverage-shard-integrity/v1"


def integrity_record_name(shard_index: int) -> str:
    """The shard's integrity record; deliberately not a ``.coverage.*`` name."""
    return f"spacr-coverage-integrity.shard-{shard_index:02d}.json"


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


def _unreadable(path: Path) -> str | None:
    """Why a coverage data file cannot be read, or None when it can."""
    try:
        CoverageData(basename=str(path)).read()
    except CoverageException as exc:
        return str(exc)
    return None


def _discard_unreadable_process_data(
    data_dir: Path, prefix: str = ".coverage.",
) -> list[Path]:
    """Remove coverage files left incomplete by terminated child processes.

    coverage.py normally combines every ``.coverage.*`` file it finds.  A
    process killed while coverage is opening its database can leave behind a
    valid SQLite shell with no coverage metadata, however.  ``coverage
    combine`` warns about that shell but leaves it in the downloaded artifact.
    Validate files after every batch run so only readable process data reaches
    the combine job.  Every name discarded after a batch is written into the
    shard's integrity record, so the gate can tell a discard that a lost
    worker explains from one that nothing does.
    """
    discarded: list[Path] = []
    for path in sorted(data_dir.glob(f"{prefix}*")):
        if not path.is_file():
            continue
        problem = _unreadable(path)
        if problem is not None:
            print(
                f"discarding unreadable coverage process data {path.name}: "
                f"{problem}",
                flush=True,
            )
            path.unlink()
            discarded.append(path)
    return discarded


def _file_of(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def describe_exit(status: int | None, *, worker: bool) -> tuple[str, str]:
    """Return ``(kind, words)`` for how a pytest process ended.

    xdist reports every dead worker as "Not properly terminated", and both
    of the ways this suite loses one lose its coverage identically, so the
    exit status is what names the cause.  They point at different fixes:

      segfault      killed by SIGSEGV -- what "Fatal Python error:
                    Segmentation fault" ends in; the crash family of item 43
      memory-guard  exit 3 from tests/conftest.py: a process passed
                    SPACR_TEST_MEMORY_GB (default 6 GB), so a test, or a
                    worker's accumulated state, needs more than that
      signal        any other signal (SIGKILL is also the kernel OOM killer)
      exit          any other status
      unknown       the status could not be read
    """
    if status is None:
        return "unknown", "ended, and its exit status could not be read"
    if status < 0:
        try:
            name = signal.Signals(-status).name
        except ValueError:
            name = f"signal {-status}"
        if name == "SIGSEGV":
            return "segfault", (
                "was killed by SIGSEGV (a segfault: the crash family of "
                "item 43)"
            )
        if name == "SIGKILL":
            return "signal", (
                "was killed by SIGKILL (also what the kernel OOM killer sends)"
            )
        return "signal", f"was killed by {name}"
    if status == MEMORY_GUARD_STATUS:
        words = (
            "was ended by the test memory guard (exit 3: its RSS passed "
            "SPACR_TEST_MEMORY_GB, default 6 GB)"
        )
        if not worker:
            words += "; pytest's internal-error status is also 3"
        return "memory-guard", words
    return "exit", f"exited with status {status}"


def read_ledger(path: Path) -> dict[str, Any] | None:
    """The batch's ledger, or None when it is missing or not one we wrote."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict) or document.get("schema") != LEDGER_SCHEMA:
        return None
    return document


def lost_coverage(
    ledger: Mapping[str, Any] | None,
    batch_files: Sequence[str],
    batch_status: int | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return the test files whose coverage data is missing, and why.

    A worker's data is present only if pytest-cov's ``cov_worker_node_id``
    came back when it went down.  Anything else -- a crash, a worker that
    never went down, no ledger at all -- means its files have to run again.
    Every reason names how the process ended (see :func:`describe_exit`).
    """
    if ledger is None:
        files = sorted(batch_files)
        kind, words = describe_exit(batch_status, worker=False)
        return files, [{
            "worker": None,
            "error": "the batch left no readable ledger, so its pytest "
                     "controller did not finish and nothing proves any "
                     "worker's data was saved",
            "exit_status": batch_status,
            "exit_kind": kind,
            "exit": words,
            "running": [],
            "files": files,
        }]
    crashes: dict[str, list[str]] = {}
    for crash in ledger.get("crashes", []):
        crashes.setdefault(str(crash.get("worker")), []).append(
            str(crash.get("nodeid")),
        )
    lost = set(ledger.get("unreported_files", []))
    reasons: list[dict[str, Any]] = []
    if lost:
        reasons.append({
            "worker": None,
            "error": "collected tests that no worker ever reported",
            "exit_status": None,
            "exit_kind": None,
            "exit": None,
            "running": [],
            "files": sorted(lost),
        })
    for worker, entry in sorted(ledger.get("workers", {}).items()):
        # A serial batch has no workers; its controller ran the tests and
        # wrote the ledger after pytest-cov saved, so nothing is missing.
        if worker == "controller" or entry.get("coverage_returned") is True:
            continue
        files = set(entry.get("files", []))
        files.update(_file_of(nodeid) for nodeid in crashes.get(worker, []))
        status = entry.get("exit_status")
        kind, words = describe_exit(status, worker=True)
        reasons.append({
            "worker": worker,
            "error": entry.get("error") or (
                "went down without returning coverage data"
                if entry.get("down") else "never went down"
            ),
            "exit_status": status,
            "exit_kind": kind,
            "exit": words,
            "running": crashes.get(worker, []),
            "files": sorted(files),
        })
        lost.update(files)
    return sorted(lost), reasons


def _coverage_options(source: str) -> list[str]:
    return [
        f"--cov={source}",
        "--cov-branch",
        "--cov-report=",
        "--cov-config=.coveragerc",
        "-o",
        "faulthandler_timeout=900",
    ]


def _data_files(data_file: Path) -> list[Path]:
    return sorted(
        path for path in data_file.parent.glob(f"{data_file.name}*")
        if path.is_file()
    )


def recover_file(
    test_file: str,
    data_file: Path,
    *,
    marker: str,
    source: str,
    attempts: int,
    timeout: float,
) -> dict[str, Any]:
    """Re-run one test file serially until its coverage data is saved."""
    tries: list[dict[str, Any]] = []
    environment = os.environ.copy()
    environment["COVERAGE_FILE"] = str(data_file)
    environment.pop(LEDGER_ENV, None)
    command = [
        sys.executable, "-m", "pytest", test_file, "-m", marker,
        *_coverage_options(source), "--cov-append", "-v", "--tb=short",
    ]
    for attempt in range(1, attempts + 1):
        # A failed attempt may leave a shell behind; the next one starts
        # from nothing so --cov-append can never fold that shell in.
        for stale in _data_files(data_file):
            stale.unlink()
        print(
            f"coverage recovery: {test_file} (attempt {attempt}/{attempts}) "
            f"-> {data_file.name}",
            flush=True,
        )
        try:
            status: int | None = int(subprocess.run(
                command, env=environment, check=False, timeout=timeout,
            ).returncode)
        except subprocess.TimeoutExpired:
            status = None
        unreadable = [
            path.name for path in _data_files(data_file)
            if _unreadable(path) is not None
        ]
        saved = status in DATA_SAVED_STATUSES and not unreadable
        tries.append({
            "attempt": attempt,
            "exit_code": status,
            "exit": (
                "saved its data" if saved
                else f"timed out after {timeout:g} s" if status is None
                else describe_exit(status, worker=False)[1]
            ),
            "timed_out": status is None,
            "unreadable_data": unreadable,
            "data_saved": saved,
        })
        if saved:
            return {"file": test_file, "recovered": True, "attempts": tries}
    for stale in _data_files(data_file):
        stale.unlink()
    return {"file": test_file, "recovered": False, "attempts": tries}


def _write_record(path: Path, record: Mapping[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def _describe(reasons: Sequence[Mapping[str, Any]]) -> str:
    """One line per lost process: who, how it ended, what xdist said, where.

    Kept in step with ``_describe_loss`` in tools/verify_module_coverage.py,
    which renders the same records in the gate's report.
    """
    parts = []
    for reason in reasons:
        who = f"worker {reason['worker']}" if reason["worker"] else "the batch"
        running = (
            " while running " + ", ".join(reason["running"])
            if reason["running"] else ""
        )
        if reason.get("exit"):
            parts.append(f"{who} {reason['exit']} [{reason['error']}]{running}")
        else:
            parts.append(f"{who}: {reason['error']}{running}")
    return "; ".join(parts)


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
    parser.add_argument(
        "--recovery-attempts", type=int, default=2,
        help="serial re-runs of a file whose coverage data was lost before it "
        "is reported unrecovered (default: 2; 0 disables recovery)",
    )
    parser.add_argument(
        "--recovery-timeout", type=float, default=1800.0,
        help="seconds one recovery attempt may take (default: 1800)",
    )
    parser.add_argument(
        "--cov-source", default="spacr",
        help="package measured with --cov (default: spacr)",
    )
    return parser


def _run_batch(
    args: argparse.Namespace, number: int, relative: list[str],
) -> dict[str, Any]:
    """Run one batch, recover what its lost workers took, and describe it."""
    data_name = f".coverage.shard-{args.shard_index:02d}.batch-{number:03d}"
    ledger_path = args.data_dir / (
        f"spacr-xdist-ledger.shard-{args.shard_index:02d}.batch-{number:03d}.json"
    )
    ledger_path.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment["COVERAGE_FILE"] = str(args.data_dir / data_name)
    environment[LEDGER_ENV] = str(ledger_path)
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(PLUGIN_DIR), environment.get("PYTHONPATH", ""))
        if part
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        *relative,
        "-m",
        args.marker,
        "-p",
        PLUGIN,
        *_coverage_options(args.cov_source),
    ]
    if args.workers > 1:
        command.extend(["-n", str(args.workers), "--dist", "loadfile"])
    command.extend(["-v", "--tb=short"])
    result = subprocess.run(command, env=environment, check=False)

    lost, reasons = lost_coverage(
        read_ledger(ledger_path), relative, int(result.returncode),
    )
    entry: dict[str, Any] = {
        "batch": number,
        "exit_code": int(result.returncode),
        "ledger": ledger_path.name if ledger_path.exists() else None,
        "lost": reasons,
        "lost_files": lost,
        "recovered_files": [],
        "unrecovered_files": [],
        "discarded_data_files": [
            path.name for path in
            _discard_unreadable_process_data(args.data_dir, data_name + ".")
        ],
    }
    if not lost:
        return entry
    print(
        f"::warning title=Coverage data lost in shard {args.shard_index} batch "
        f"{number}::{_describe(reasons)}. Re-running {len(lost)} test file(s) "
        "serially to recover their coverage.",
        flush=True,
    )
    for index, test_file in enumerate(lost, start=1):
        outcome = recover_file(
            test_file,
            args.data_dir / f"{data_name}-recovery-{index:03d}",
            marker=args.marker,
            source=args.cov_source,
            attempts=args.recovery_attempts,
            timeout=args.recovery_timeout,
        )
        key = "recovered_files" if outcome["recovered"] else "unrecovered_files"
        entry[key].append(outcome)
    return entry


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
    if args.recovery_attempts < 0:
        raise ValueError("recovery attempts cannot be negative")

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
    record_path = args.data_dir / integrity_record_name(args.shard_index)
    record: dict[str, Any] = {
        "schema": INTEGRITY_SCHEMA,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "batches_total": len(batches),
        "batches_finished": 0,
        "batches": [],
    }
    # Written before the first batch: a shard killed during batch 1 leaves
    # "0 of N finished" behind rather than no record at all.
    _write_record(record_path, record)
    failures: list[tuple[int, int, list[Path]]] = []
    for number, batch in enumerate(batches, start=1):
        relative = [path.relative_to(root).as_posix() for path in batch]
        print(
            f"coverage shard {args.shard_index}, batch {number}/"
            f"{len(batches)}: {len(batch)} files",
            flush=True,
        )
        entry = _run_batch(args, number, relative)
        if entry["exit_code"] not in (0, NO_TESTS_COLLECTED):
            failures.append((number, entry["exit_code"], batch))
        record["batches"].append(entry)
        record["batches_finished"] = number
        _write_record(record_path, record)

    _discard_unreadable_process_data(args.data_dir)

    unrecovered = [
        (entry["batch"], outcome["file"])
        for entry in record["batches"] for outcome in entry["unrecovered_files"]
    ]
    for entry in record["batches"]:
        for outcome in entry["recovered_files"]:
            print(
                f"coverage batch {entry['batch']}: recovered the coverage of "
                f"{outcome['file']}"
            )
    for number, test_file in unrecovered:
        print(
            f"::error title=Coverage data lost in shard {args.shard_index}::"
            f"batch {number}: the coverage of {test_file} could not be "
            "recovered; the coverage gate will report an INCOMPLETE "
            "MEASUREMENT"
        )
    if failures:
        for number, code, batch in failures:
            names = " ".join(path.relative_to(root).as_posix() for path in batch)
            print(f"coverage batch {number} (exit {code}) ran: {names}")
        return failures[0][1]
    if unrecovered:
        return UNRECOVERED_STATUS
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
