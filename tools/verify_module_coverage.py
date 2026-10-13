#!/usr/bin/env python3
"""Gate shipped-module coverage on a committed per-module ratchet baseline.

THE DENOMINATOR is packaging, not an editable manifest.  This tool reads the
``packages=find_packages(...)`` expression passed to ``setup()`` in
``setup.py``, calls :func:`setuptools.find_packages` with those literal
arguments, and includes every direct ``*.py`` child of every discovered
package.  A new package module therefore enters the gate automatically.

THE GATE (maintainer decision, 2026-09-19, item 288): the goal is 90%
statement-and-branch coverage per module, not 100% -- "if the module would
benefit from more than 90% coverage, implement that, but only in cases where
coverage is useful".  So a module's bar is ``max(90%, what that module
already has recorded)``: a 90% FLOOR plus no-regression.  For every shipped
module four ABSOLUTE counts are measured -- uncovered statements, uncovered
branches, ``pragma: no cover`` comments and coverage-excluded lines -- and
compared with the same four counts recorded for that module in the baseline
(``tools/coverage_baseline.json``).  The run FAILS when:

  (a) any count of a module rises above its baseline count;
  (b) a module recorded at 100% in the baseline is no longer at 100%;
  (c) a shipped module is below the 90% FLOOR and carries no exemption.
      The floor counts statements and branches together:
      ``(covered_lines + covered_branches) / (statements + branches)``.
      A module already in the baseline must clear the floor AND rule (a).
      A module NOT in the baseline must clear the floor and additionally
      carry no ``pragma: no cover`` and no coverage-excluded line: the
      floor forgives code a test has not reached yet, never code hidden
      from the measurement, and zero is the only baseline a new module
      gets for those two counts;
  (d) a module is in the baseline but is no longer shipped.  A deleted or
      renamed file must not carry its allowance away silently, so the
      baseline has to be tightened deliberately (``--update-baseline``
      trims the entry; a module that returns under any name is new and must
      arrive at or above the floor);
  (e) anything cannot be measured: an unreadable coverage file, coverage
      without branch data, a shipped module with no valid coverage row, or
      a baseline this tool did not write.  "Checked N of M" must be M of M.
  (f) the measurement is INCOMPLETE (only with ``--shard-integrity``): a
      coverage shard left no integrity record, did not finish its batches,
      or lost a worker's coverage data that ``tools/run_coverage_batches.py``
      could not recover by re-running that worker's files.  A worker killed
      by a signal never writes its data, so every module it exercised reads
      as less covered than it is.  That is its OWN verdict, not a
      regression: counts that such a loss can raise (uncovered statements
      and branches) are reported UNCONFIRMED instead of as failures, and the
      run exits 3.  A new pragma or excluded line is read from the source,
      so it is still an ERROR and still exits 1.  Losses the runner DID
      recover are listed as RECOVERED and judged normally.

Exit status: 0 pass, 1 a confirmed failure, 2 the gate could not run, 3 an
incomplete measurement with nothing confirmed.  3 is never a pass.

THE EXEMPTIONS (``--floor-exemptions``) are for a module that genuinely
cannot reach the floor -- a GPU-only path, a branch only one platform takes.
Each line of the file names one module and says WHY, and an exemption lifts
only the floor: rule (a) still forbids that module losing anything it has.
An exemption for a module that no longer ships fails, exactly as a stale
baseline entry does, and one whose module now clears the floor is printed so
it can be removed.  A ``# pragma: no cover`` is NOT the way to do this: the
repository spent an item removing them and rule (a) counts every one.

It PASSES, and prints an improvement notice, when a module's counts fall.
Every module not yet at 100% is listed on every run, with its baseline
allowance and the date that allowance was written; the ones below the floor
are named first, because those are the work.

COUNTS, NOT LINE NUMBERS.  The baseline stores how many statements and
branches are uncovered, not which ones.  An unrelated edit that moves code
up or down therefore cannot trip the gate.  The price is that a change which
covers one line and uncovers another in the same module passes unnoticed;
the per-module listing still shows the lines, and the count can never grow.

THE BASELINE IS AN OUTPUT.  Only this tool writes it, and it stores a
SHA-256 checksum of its own body.  The gate refuses a baseline whose
checksum does not match, which is how a hand edit is caught.  This is a
tripwire, not a security boundary: anyone can recompute a checksum, but
nobody does it by accident.  Two explicit writes exist, both requiring
``--reason`` and both appending a dated history entry (time, commit, mode,
reason).  CI runs neither; a green run never changes the baseline, because
a baseline that follows the measurement cannot see a slow slide.

  --update-baseline   tighten only: lowers counts that fell, adds new
                      modules that are at or above the floor, trims modules
                      no longer shipped.  It never raises a count and never
                      admits a module below the floor.
  --reset-baseline    deliberate regeneration from the measurement, which
                      CAN loosen; every loosened module is printed.  Use it
                      to seed the file or to admit a new module below the
                      floor after review.
  --retire-module P   remove exactly one entry, for a module deleted on
                      purpose.  A deleted file has no coverage to measure,
                      so this reads no coverage data; it refuses a module
                      that still ships or is not in the baseline, and it
                      never touches another entry.

The first two refuse to write from an incomplete measurement.  To regenerate from a
CI run, download the ``spacr-module-coverage-report-<run>-<attempt>``
artifact (it holds ``coverage.json``), check out THAT run's commit so the
inventory and pragma comments match the data, and run::

    python tools/verify_module_coverage.py \\
        --coverage-json coverage.json --root . \\
        --baseline tools/coverage_baseline.json \\
        --update-baseline --reason "why" \\
        --json-out report.json --text-out report.txt
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import json
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from setuptools import find_packages

REPORT_SCHEMA = "spacr.module-coverage-ratchet/v4"
BASELINE_SCHEMA = "spacr.module-coverage-baseline/v1"
PRAGMA_NO_COVER = re.compile(r"#\s*pragma\s*:\s*no\s*cover\b", re.IGNORECASE)

#: The maintainer's floor, item 288, 2026-09-19: every shipped module is at
#: this percentage of statements-and-branches or better.  It is an integer so
#: the comparison can be done in exact integer arithmetic; no module is ever
#: failed or passed by a floating-point rounding error.
COVERAGE_FLOOR_PERCENT = 90

COUNT_FIELDS = (
    "uncovered_statements",
    "uncovered_branches",
    "pragma_no_cover",
    "excluded_lines",
)
COUNT_LABELS = {
    "uncovered_statements": "uncovered statements",
    "uncovered_branches": "uncovered branches",
    "pragma_no_cover": "pragma: no cover comments",
    "excluded_lines": "coverage-excluded lines",
}
#: The counts coverage data lost with a crashed worker can raise.  Pragma
#: comments are read from the source and excluded lines follow from it.
EXECUTION_FIELDS = ("uncovered_statements", "uncovered_branches")
INTEGRITY_SCHEMA = "spacr.coverage-shard-integrity/v1"
INTEGRITY_GLOB = "spacr-coverage-integrity.shard-*.json"
INCOMPLETE_STATUS = 3
BASELINE_ABOUT = (
    "Written only by tools/verify_module_coverage.py. Do not edit: the "
    "checksum covers every other key and the gate refuses a file it did not "
    "write. Tighten with --update-baseline --reason; regenerate deliberately "
    "with --reset-baseline --reason (see the tool's docstring)."
)


class InventoryError(ValueError):
    """The packaging declaration cannot be interpreted safely."""


class ExemptionError(ValueError):
    """The floor-exemption file cannot be read as named, reasoned entries."""


class BaselineError(ValueError):
    """The baseline cannot be trusted or cannot be written."""


@dataclass(frozen=True)
class FindPackagesCall:
    """Literal arguments from setup.py's one packaging ``find_packages``."""

    where: str = "."
    exclude: tuple[str, ...] = ()
    include: tuple[str, ...] = ("*",)


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _find_packages_call(setup_path: Path) -> FindPackagesCall:
    """Return the literal ``find_packages`` call used by ``setup()``.

    Executing a setup script just to learn its package list would also
    execute any future module-level side effect.  AST evaluation keeps this
    inventory read-only and deliberately rejects computed arguments: an
    unfamiliar packaging shape must be reviewed instead of silently changing
    the coverage denominator.
    """
    tree = ast.parse(setup_path.read_text(encoding="utf-8"), setup_path.name)
    package_nodes: list[ast.AST] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node.func) != "setup":
            continue
        package_nodes.extend(
            keyword.value for keyword in node.keywords
            if keyword.arg == "packages"
        )
    if len(package_nodes) != 1:
        raise InventoryError(
            "setup.py must pass exactly one packages= value to setup(); "
            f"found {len(package_nodes)}"
        )

    call = package_nodes[0]
    if not isinstance(call, ast.Call) or _call_name(call.func) != "find_packages":
        raise InventoryError(
            "setup(packages=...) must call setuptools.find_packages directly"
        )
    if len(call.args) > 3:
        raise InventoryError("find_packages accepts at most three positional arguments")

    names = ("where", "exclude", "include")
    values: dict[str, Any] = {}
    for name, value in zip(names, call.args):
        try:
            values[name] = ast.literal_eval(value)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise InventoryError(
                f"find_packages {name} must be a literal"
            ) from exc
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg not in names:
            raise InventoryError(
                f"unsupported find_packages keyword: {keyword.arg!r}"
            )
        if keyword.arg in values:
            raise InventoryError(
                f"find_packages supplies {keyword.arg!r} more than once"
            )
        try:
            values[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise InventoryError(
                f"find_packages {keyword.arg} must be a literal"
            ) from exc

    where = values.get("where", ".")
    exclude = values.get("exclude", ())
    include = values.get("include", ("*",))
    if not isinstance(where, str):
        raise InventoryError("find_packages where must be a string")
    if not isinstance(exclude, (list, tuple)) or not all(
            isinstance(value, str) for value in exclude):
        raise InventoryError("find_packages exclude must contain only strings")
    if not isinstance(include, (list, tuple)) or not all(
            isinstance(value, str) for value in include):
        raise InventoryError("find_packages include must contain only strings")
    return FindPackagesCall(where, tuple(exclude), tuple(include))


def discover_shipped_python_files(root: Path) -> list[str]:
    """Return repository-relative Python files installed by setup.py.

    ``find_packages`` decides which directories ship.  Enumerating only each
    package's direct Python children mirrors what setuptools installs while
    excluding Python-looking asset generators below non-package resource
    directories.
    """
    root = root.resolve()
    declaration = _find_packages_call(root / "setup.py")
    package_root = (root / declaration.where).resolve()
    try:
        package_root.relative_to(root)
    except ValueError as exc:
        raise InventoryError("find_packages where escapes the repository") from exc

    packages = find_packages(
        where=str(package_root),
        exclude=declaration.exclude,
        include=declaration.include,
    )
    files: set[str] = set()
    for package in packages:
        directory = package_root.joinpath(*package.split("."))
        for source in directory.glob("*.py"):
            if source.is_file():
                files.add(source.resolve().relative_to(root).as_posix())
    return sorted(files)


def _pragma_lines(source: Path) -> list[int]:
    """Return real comment lines carrying a coverage exclusion pragma."""
    found: list[int] = []
    with tokenize.open(source) as stream:
        tokens = tokenize.generate_tokens(stream.readline)
        for token in tokens:
            if token.type == tokenize.COMMENT and PRAGMA_NO_COVER.search(token.string):
                found.append(token.start[0])
    return found


def _normalise_coverage_rows(
    files: Mapping[str, Any], root: Path,
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    """Map coverage.py path keys to safe repository-relative paths."""
    root = root.resolve()
    normalised: dict[str, Mapping[str, Any]] = {}
    issues: list[str] = []
    for raw_path, row in files.items():
        if not isinstance(raw_path, str) or not isinstance(row, Mapping):
            issues.append(f"invalid coverage file row: {raw_path!r}")
            continue
        # Coverage data generated on Windows can be inspected on another OS;
        # its ordinary relative paths still have backslashes in that case.
        portable = raw_path.replace("\\", "/")
        path = Path(portable)
        resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError:
            # Rows outside the distribution are harmless test/helper coverage.
            continue
        if relative in normalised:
            issues.append(f"duplicate coverage rows resolve to {relative}")
            continue
        normalised[relative] = row
    return normalised, issues


def _line_ranges(lines: Sequence[int]) -> str:
    values = sorted({int(line) for line in lines})
    if not values:
        return "none"
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


def _branch_list(branches: Sequence[Sequence[int]]) -> str:
    return ", ".join(
        f"{int(branch[0])}->{int(branch[1])}"
        for branch in branches
        if len(branch) == 2
    ) or "none"


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _describe_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(
        f"{counts[field]} {COUNT_LABELS[field]}" for field in COUNT_FIELDS
    )


def _slash(counts: Mapping[str, int]) -> str:
    return "/".join(str(counts[field]) for field in COUNT_FIELDS)


def coverage_fraction(entry: Mapping[str, Any]) -> tuple[int, int] | None:
    """``(covered, total)`` over statements AND branches, or None.

    The uncovered halves are the ratchet's own counts rather than coverage's
    ``covered_lines``, so the percentage on the report and the number the
    gate ratchets can never disagree about the same module.
    """
    counts = entry.get("counts")
    statements = entry.get("num_statements")
    branches = entry.get("num_branches")
    if counts is None or not _is_count(statements) or not _is_count(branches):
        return None
    total = statements + branches
    uncovered = counts["uncovered_statements"] + counts["uncovered_branches"]
    return max(total - uncovered, 0), total


def coverage_percent(entry: Mapping[str, Any]) -> float | None:
    """The module's statement-and-branch percentage, or None.

    A module with nothing to measure -- an empty ``__init__.py`` -- is 100%;
    there is no gap in it to find.
    """
    fraction = coverage_fraction(entry)
    if fraction is None:
        return None
    covered, total = fraction
    return 100.0 if total == 0 else 100.0 * covered / total


def is_below_floor(entry: Mapping[str, Any], floor: int) -> bool:
    """Whether the module is under ``floor`` percent, in integer arithmetic."""
    fraction = coverage_fraction(entry)
    if fraction is None:
        return False
    covered, total = fraction
    return covered * 100 < floor * total


def parse_floor_exemptions(text: str, *, source: str) -> dict[str, str]:
    """Read ``path: reason`` lines; every exemption must say why.

    Blank lines and ``#`` comments are ignored.  A line without a reason, a
    duplicate path or an absolute path raises, because an exemption nobody
    can read is the thing this file exists to prevent.
    """
    exemptions: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        path, separator, reason = line.partition(":")
        path = path.strip()
        reason = reason.strip()
        where = f"{source} line {number}"
        if not separator or not reason:
            raise ExemptionError(
                f"{where}: an exemption is '<module path>: <why it cannot "
                f"reach the floor>', and this one gives no reason: {line!r}"
            )
        if not path or Path(path).is_absolute():
            raise ExemptionError(
                f"{where}: {path!r} is not a repository-relative module path"
            )
        path = Path(path).as_posix()
        if path in exemptions:
            raise ExemptionError(f"{where}: {path} is exempted twice")
        exemptions[path] = reason
    return exemptions


def load_floor_exemptions(path: Path) -> dict[str, str]:
    """Read the committed exemption file; a missing file is not an empty one."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExemptionError(f"cannot read floor exemptions {path}: {exc}") from exc
    return parse_floor_exemptions(text, source=str(path))


def _measure_module(
    root: Path, relative: str, row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Measure one shipped module; ``counts`` is None when it cannot be."""
    pragmas = _pragma_lines(root / relative)
    errors: list[str] = []
    gaps: list[str] = []
    entry: dict[str, Any] = {
        "path": relative,
        "num_statements": None,
        "covered_lines": None,
        "missing_lines": [],
        "num_branches": None,
        "covered_branches": None,
        "missing_branches": [],
        "excluded_lines": [],
        "pragma_no_cover_lines": pragmas,
        "counts": None,
    }
    if row is None:
        errors.append("missing coverage row")
    else:
        summary = row.get("summary", {})
        if not isinstance(summary, Mapping):
            summary = {}
            errors.append("coverage row has no summary object")
        lists: dict[str, list[Any]] = {}
        for name in ("missing_lines", "missing_branches", "excluded_lines"):
            value = row.get(name, [])
            if not isinstance(value, list):
                value = []
                errors.append(f"coverage row has invalid {name}")
            lists[name] = value
        required = {
            name: summary.get(name)
            for name in (
                "num_statements", "covered_lines",
                "num_branches", "covered_branches",
            )
        }
        entry.update(required)
        entry.update(lists)
        invalid = [name for name, value in required.items() if not _is_count(value)]
        summary_excluded = summary.get("excluded_lines", 0)
        if not _is_count(summary_excluded):
            invalid.append("excluded_lines")
        if invalid:
            errors.append(
                "coverage summary has invalid counts: " + ", ".join(invalid)
            )
        else:
            counts = {
                "uncovered_statements": max(
                    len({int(line) for line in lists["missing_lines"]}),
                    required["num_statements"] - required["covered_lines"],
                ),
                "uncovered_branches": max(
                    len(lists["missing_branches"]),
                    required["num_branches"] - required["covered_branches"],
                ),
                "pragma_no_cover": len(pragmas),
                "excluded_lines": max(
                    len(lists["excluded_lines"]), summary_excluded,
                ),
            }
            entry["counts"] = counts
            if counts["uncovered_statements"]:
                gaps.append(
                    "uncovered statements: "
                    + _line_ranges(lists["missing_lines"])
                )
            if counts["uncovered_branches"]:
                gaps.append(
                    "uncovered branches: "
                    + _branch_list(lists["missing_branches"])
                )
            if counts["excluded_lines"]:
                gaps.append(
                    "coverage-excluded lines: "
                    + _line_ranges(lists["excluded_lines"])
                )
            if pragmas:
                gaps.append("pragma: no cover comments: " + _line_ranges(pragmas))
    entry["measurement_errors"] = errors
    entry["gaps"] = gaps
    return entry


def measure(
    *,
    root: Path,
    coverage_data: Mapping[str, Any],
    expected_file_count: int | None = None,
) -> dict[str, Any]:
    """Measure every shipped module; no baseline is involved yet."""
    root = root.resolve()
    shipped = discover_shipped_python_files(root)
    raw_files = coverage_data.get("files", {})
    if not isinstance(raw_files, Mapping):
        raise ValueError("coverage JSON 'files' must be an object")
    rows, global_issues = _normalise_coverage_rows(raw_files, root)

    meta = coverage_data.get("meta", {})
    if not isinstance(meta, Mapping):
        meta = {}
    if meta.get("branch_coverage") is not True:
        global_issues.append(
            "coverage JSON was not produced with branch coverage enabled"
        )
    if expected_file_count is not None and len(shipped) != expected_file_count:
        global_issues.append(
            f"shipped-file inventory changed: expected {expected_file_count}, "
            f"found {len(shipped)}"
        )
    modules = [
        _measure_module(root, relative, rows.get(relative))
        for relative in shipped
    ]
    checked = sum(
        module["counts"] is not None and not module["measurement_errors"]
        for module in modules
    )
    if checked != len(shipped):
        global_issues.append(
            f"checked {checked} of {len(shipped)} shipped modules; the gate "
            "never passes on a partial measurement"
        )
    return {
        "root": str(root),
        "coverage": {
            "version": meta.get("version"),
            "timestamp": meta.get("timestamp"),
            "branch_coverage": meta.get("branch_coverage"),
            "input_rows": len(raw_files),
            "repository_rows": len(rows),
        },
        "inventory": {
            "expected_file_count": expected_file_count,
            "shipped_file_count": len(shipped),
            "files": shipped,
        },
        "modules_checked": checked,
        "global_issues": global_issues,
        "modules": modules,
    }


def _integrity_record_problem(document: Any, shard_count: int) -> str | None:
    """Why a shard integrity record cannot be trusted, or None."""
    if not isinstance(document, dict) or document.get("schema") != INTEGRITY_SCHEMA:
        return f"not a {INTEGRITY_SCHEMA} record"
    index = document.get("shard_index")
    if not _is_count(index) or index >= shard_count:
        return f"shard_index {index!r} is not one of {shard_count} shards"
    if document.get("shard_count") != shard_count:
        return (
            f"coverage shard {index} ran as one of "
            f"{document.get('shard_count')!r} shards, not {shard_count}"
        )
    total = document.get("batches_total")
    finished = document.get("batches_finished")
    batches = document.get("batches")
    if not (
        _is_count(total) and _is_count(finished) and finished <= total
        and isinstance(batches, list) and len(batches) == finished
    ):
        return f"coverage shard {index} has inconsistent batch counts"
    for batch in batches:
        if not isinstance(batch, dict) or not _is_count(batch.get("batch")):
            return f"coverage shard {index} has a malformed batch entry"
        lists = (
            "lost", "lost_files", "recovered_files", "unrecovered_files",
            "discarded_data_files",
        )
        if not all(isinstance(batch.get(key, []), list) for key in lists):
            return f"coverage shard {index} batch {batch['batch']} is malformed"
        outcomes = [
            *batch.get("recovered_files", []), *batch.get("unrecovered_files", []),
        ]
        if not all(
                isinstance(outcome, dict) and isinstance(outcome.get("file"), str)
                for outcome in outcomes):
            return (
                f"coverage shard {index} batch {batch['batch']} has a "
                "malformed recovery outcome"
            )
    return None


def _describe_loss(reasons: Any) -> str:
    """Name each lost process, HOW it ended, and what it was running.

    xdist says "Not properly terminated" for a segfault and for the test
    memory guard alike; the runner records the exit status that tells them
    apart, and the words are rendered here (see ``describe_exit`` in
    tools/run_coverage_batches.py).
    """
    parts: list[str] = []
    for reason in reasons if isinstance(reasons, list) else []:
        if not isinstance(reason, dict):
            continue
        who = f"worker {reason['worker']}" if reason.get("worker") else "the batch"
        running = reason.get("running") or []
        during = f" while running {', '.join(map(str, running))}" if running else ""
        error = reason.get("error") or "lost its coverage data"
        if reason.get("exit"):
            parts.append(f"{who} {reason['exit']} [{error}]{during}")
        else:
            parts.append(f"{who}: {error}{during}")
    return "; ".join(parts) or "coverage data was lost"


def _attempts(outcomes: Sequence[Mapping[str, Any]]) -> str:
    tried = [
        f"{outcome['file']}: " + ", ".join(
            str(attempt.get("exit") or attempt.get("exit_code"))
            for attempt in outcome.get("attempts", [])
            if isinstance(attempt, dict)
        )
        for outcome in outcomes if outcome.get("attempts")
    ]
    return f" (recovery attempts -- {'; '.join(tried)})" if tried else ""


def _judge_batch(shard: int, batch: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Return (issues, recovered notes) for one batch of one shard."""
    where = f"coverage shard {shard} batch {batch['batch']}"
    cause = _describe_loss(batch.get("lost", []))
    lost = {str(name) for name in batch.get("lost_files", [])}
    recovered = [outcome["file"] for outcome in batch.get("recovered_files", [])]
    unrecovered = [outcome["file"] for outcome in batch.get("unrecovered_files", [])]
    issues: list[str] = []
    notes: list[str] = []
    if unrecovered:
        issues.append(
            f"{where}: {cause}; the coverage of {len(unrecovered)} test "
            f"file(s) could not be recovered: {', '.join(unrecovered)}"
            + _attempts(batch.get("unrecovered_files", []))
        )
    unaccounted = sorted(lost - set(recovered) - set(unrecovered))
    if unaccounted:
        issues.append(
            f"{where}: {cause}; {len(unaccounted)} test file(s) lost their "
            f"coverage and were never re-run: {', '.join(unaccounted)}"
        )
    if batch.get("discarded_data_files") and not lost:
        issues.append(
            f"{where}: unreadable coverage data was discarded "
            f"({', '.join(map(str, batch['discarded_data_files']))}) and no "
            "lost worker explains it"
        )
    if recovered:
        notes.append(
            f"{where}: {cause}; {len(recovered)} test file(s) re-run serially "
            f"and their coverage recovered: {', '.join(recovered)}"
        )
    return issues, notes


def check_shard_integrity(directory: Path, shard_count: int) -> dict[str, Any]:
    """Read every coverage shard's integrity record and name what is missing.

    ``tools/run_coverage_batches.py`` writes one record per shard and
    rewrites it after every batch.  The measurement is complete only when
    every one of ``shard_count`` shards left a record, finished all its
    batches, and recovered every worker's lost coverage.
    """
    issues: list[str] = []
    recovered: list[str] = []
    kinds: dict[str, int] = {}
    records: dict[int, Mapping[str, Any]] = {}
    for path in sorted(Path(directory).rglob(INTEGRITY_GLOB)):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            issues.append(f"{path.name}: unreadable shard integrity record ({exc})")
            continue
        problem = _integrity_record_problem(document, shard_count)
        if problem is not None:
            issues.append(f"{path.name}: {problem}")
        elif document["shard_index"] in records:
            issues.append(
                f"coverage shard {document['shard_index']} has more than one "
                "integrity record"
            )
        else:
            records[document["shard_index"]] = document
    for shard in range(shard_count):
        document = records.get(shard)
        if document is None:
            issues.append(
                f"coverage shard {shard} left no integrity record, so nothing "
                "shows that its batches all ran and every worker's coverage "
                "data was saved"
            )
            continue
        if document["batches_finished"] != document["batches_total"]:
            issues.append(
                f"coverage shard {shard} finished {document['batches_finished']} "
                f"of {document['batches_total']} batches"
            )
        for batch in document["batches"]:
            batch_issues, notes = _judge_batch(shard, batch)
            issues.extend(batch_issues)
            recovered.extend(notes)
            for reason in batch.get("lost", []):
                kind = reason.get("exit_kind") if isinstance(reason, dict) else None
                if kind:
                    kinds[kind] = kinds.get(kind, 0) + 1
    return {
        "checked": True,
        "shard_count": shard_count,
        "shards_with_records": len(records),
        "status": "incomplete" if issues else "complete",
        "issues": issues,
        "recovered": recovered,
        # How the lost processes ended, counted: "segfault" points at the
        # crash family, "memory-guard" at a test needing more than
        # SPACR_TEST_MEMORY_GB.
        "loss_kinds": dict(sorted(kinds.items())),
    }


def _integrity_issues(measurement: Mapping[str, Any]) -> list[str]:
    integrity = measurement.get("integrity") or {}
    if integrity.get("status") != "incomplete":
        return []
    return list(integrity.get("issues", [])) or ["shard integrity is incomplete"]


def measurement_is_complete(measurement: Mapping[str, Any]) -> bool:
    """True when every shipped module was measured, nothing global failed,
    and (when it was checked) no coverage shard lost data it did not recover."""
    return not measurement["global_issues"] and not _integrity_issues(
        measurement
    ) and (
        measurement["modules_checked"]
        == measurement["inventory"]["shipped_file_count"]
    )


# -- baseline file ---------------------------------------------------------


def baseline_checksum(document: Mapping[str, Any]) -> str:
    """SHA-256 of every key except ``checksum``, in canonical JSON."""
    body = {key: value for key, value in document.items() if key != "checksum"}
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _baseline_totals(modules: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    totals = {
        "modules": len(modules),
        "modules_at_100_percent": sum(
            not any(entry[field] for field in COUNT_FIELDS)
            for entry in modules.values()
        ),
    }
    totals["modules_with_gaps"] = totals["modules"] - totals["modules_at_100_percent"]
    for field in COUNT_FIELDS:
        totals[field] = sum(entry[field] for entry in modules.values())
    return totals


def build_baseline_document(
    modules: Mapping[str, Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble a baseline whose totals and checksum agree with its body."""
    ordered = {path: dict(modules[path]) for path in sorted(modules)}
    document: dict[str, Any] = {
        "schema": BASELINE_SCHEMA,
        "about": BASELINE_ABOUT,
        "history": [dict(entry) for entry in history],
        "totals": _baseline_totals(ordered),
        "modules": ordered,
    }
    document["checksum"] = baseline_checksum(document)
    return document


def validate_baseline(document: Any) -> dict[str, Any]:
    """Return ``document`` if this tool wrote it unchanged; raise otherwise."""
    if not isinstance(document, dict):
        raise BaselineError("baseline root must be a JSON object")
    if document.get("schema") != BASELINE_SCHEMA:
        raise BaselineError(
            f"baseline schema is {document.get('schema')!r}, "
            f"expected {BASELINE_SCHEMA!r}"
        )
    stored = document.get("checksum")
    if stored != baseline_checksum(document):
        raise BaselineError(
            "baseline checksum does not match its body: it was edited by hand "
            "or damaged. Only tools/verify_module_coverage.py writes it; "
            "regenerate deliberately with --reset-baseline --reason"
        )
    history = document.get("history")
    if not isinstance(history, list) or not history or not all(
            isinstance(entry, dict)
            and all(isinstance(entry.get(key), str)
                    for key in ("written_at", "mode", "commit", "reason"))
            for entry in history):
        raise BaselineError("baseline history must list dated, reasoned writes")
    written = {entry["written_at"] for entry in history}
    modules = document.get("modules")
    if not isinstance(modules, dict):
        raise BaselineError("baseline modules must be an object")
    for path, entry in modules.items():
        if not isinstance(entry, dict) or not all(
                _is_count(entry.get(field)) for field in COUNT_FIELDS):
            raise BaselineError(f"baseline entry for {path} has invalid counts")
        if entry.get("since") not in written:
            raise BaselineError(
                f"baseline entry for {path} names a write not in the history"
            )
    if document.get("totals") != _baseline_totals(modules):
        raise BaselineError("baseline totals disagree with its modules")
    return document


def load_baseline(path: Path) -> dict[str, Any]:
    """Read and validate a baseline; a missing or foreign file raises."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineError(f"baseline {path} is not valid JSON: {exc}") from exc
    return validate_baseline(document)


def write_baseline(path: Path, document: Mapping[str, Any]) -> None:
    """Write a baseline this tool built; refuses one that does not validate."""
    validate_baseline(dict(document))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# -- the ratchet -----------------------------------------------------------


def _judge_module(
    module: Mapping[str, Any], base: Mapping[str, Any] | None,
    *, floor: int = COVERAGE_FLOOR_PERCENT, exemption: str | None = None,
) -> tuple[list[tuple[str, bool]], list[str]]:
    """Return (failures, improvements) for one measured module.

    Two things are judged: the FLOOR, which every shipped module clears
    unless ``exemption`` says why it cannot, and NO-REGRESSION against
    ``base``.  The module's bar is whichever of the two is higher.

    Every failure carries whether coverage data lost with a crashed worker
    could have produced it.  Only uncovered statements and branches can
    rise that way -- and the floor is measured from exactly those -- while a
    measurement error, a pragma or an excluded line is never explained away.
    """
    failures = [(error, False) for error in module["measurement_errors"]]
    improvements: list[str] = []
    counts = module["counts"]
    if counts is None or failures:
        return failures, improvements
    measured_full = not any(counts[field] for field in COUNT_FIELDS)
    execution_only = not any(
        counts[field] for field in COUNT_FIELDS if field not in EXECUTION_FIELDS
    )
    if exemption is None and is_below_floor(module, floor):
        percent = coverage_percent(module)
        where = (
            "and is not in the baseline" if base is None
            else "and its baseline does not lift the floor"
        )
        failures.append((
            f"is below the {floor}% floor at {percent:.2f}% {where}: "
            + _describe_counts(counts),
            True,
        ))
    if base is None:
        for field in COUNT_FIELDS:
            if field not in EXECUTION_FIELDS and counts[field]:
                failures.append((
                    f"is not in the baseline and has {counts[field]} "
                    f"{COUNT_LABELS[field]}; the floor never excuses hiding "
                    "code from coverage, only failing to reach it",
                    False,
                ))
        return failures, improvements
    if not any(base[field] for field in COUNT_FIELDS) and not measured_full:
        failures.append((
            "was at 100% in the baseline and now has "
            + _describe_counts(counts),
            execution_only,
        ))
        return failures, improvements
    for field in COUNT_FIELDS:
        label = COUNT_LABELS[field]
        if counts[field] > base[field]:
            failures.append((
                f"{label} rose from {base[field]} to {counts[field]}",
                field in EXECUTION_FIELDS,
            ))
        elif counts[field] < base[field]:
            improvements.append(
                f"{label} fell from {base[field]} to {counts[field]}"
            )
    return failures, improvements


def evaluate(
    measurement: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    *,
    baseline_path: str | None = None,
    exemptions: Mapping[str, str] | None = None,
    exemptions_path: str | None = None,
    floor: int = COVERAGE_FLOOR_PERCENT,
) -> dict[str, Any]:
    """Judge a measurement against the floor and a baseline.

    ``baseline`` None means no module has a recorded allowance, so the floor
    is the whole gate.  ``exemptions`` maps a module path to the reason it
    cannot reach the floor; it lifts the floor for that module and nothing
    else.

    When the measurement's shard integrity check is INCOMPLETE, a failure
    that lost coverage data could explain is reported as ``unconfirmed``
    rather than as a failure, and the status is ``incomplete`` unless
    something data loss cannot explain failed too.
    """
    base_modules: Mapping[str, Mapping[str, Any]] = (
        baseline["modules"] if baseline is not None else {}
    )
    exempt: Mapping[str, str] = dict(exemptions or {})
    integrity = dict(measurement.get("integrity") or {"checked": False})
    incomplete = integrity.get("status") == "incomplete"
    shipped = set(measurement["inventory"]["files"])
    modules: list[dict[str, Any]] = []
    for measured in measurement["modules"]:
        entry = dict(measured)
        base = base_modules.get(entry["path"])
        judged, improvements = _judge_module(
            entry, base, floor=floor, exemption=exempt.get(entry["path"]),
        )
        failures = [
            message for message, explainable in judged
            if not (incomplete and explainable)
        ]
        unconfirmed = [
            message for message, explainable in judged
            if incomplete and explainable
        ]
        counts = entry["counts"]
        entry["baseline"] = dict(base) if base is not None else None
        entry["at_100_percent"] = (
            counts is not None
            and not entry["measurement_errors"]
            and not any(counts[field] for field in COUNT_FIELDS)
        )
        entry["percent"] = coverage_percent(entry)
        entry["floor_exemption"] = exempt.get(entry["path"])
        entry["below_floor"] = (
            counts is not None
            and not entry["measurement_errors"]
            and is_below_floor(entry, floor)
        )
        entry["failures"] = failures
        entry["unconfirmed"] = unconfirmed
        entry["improvements"] = improvements
        entry["status"] = (
            "fail" if failures else "unconfirmed" if unconfirmed else "pass"
        )
        modules.append(entry)
    stale = sorted(path for path in base_modules if path not in shipped)
    stale_exemptions = sorted(path for path in exempt if path not in shipped)
    spent_exemptions = sorted(
        module["path"] for module in modules
        if module["floor_exemption"] is not None
        and module["counts"] is not None
        and not module["measurement_errors"]
        and not module["below_floor"]
    )
    failed = sum(module["status"] == "fail" for module in modules)
    measured_below = sum(
        module["counts"] is not None
        and not module["measurement_errors"]
        and not module["at_100_percent"]
        for module in modules
    )
    passed = (
        not measurement["global_issues"]
        and failed == 0
        and not stale
        and not stale_exemptions
    )
    status = "fail" if not passed else "incomplete" if incomplete else "pass"
    return {
        "schema": REPORT_SCHEMA,
        "status": status,
        "floor_percent": floor,
        "measurement_integrity": integrity,
        "root": measurement["root"],
        "coverage": dict(measurement["coverage"]),
        "inventory": dict(measurement["inventory"]),
        "exemptions": {
            "path": exemptions_path,
            "modules": dict(sorted(exempt.items())),
            "stale": stale_exemptions,
            "no_longer_needed": spent_exemptions,
        },
        "baseline": {
            "path": baseline_path,
            "modules": len(base_modules),
            "totals": dict(baseline["totals"]) if baseline is not None else None,
            "checksum": baseline["checksum"] if baseline is not None else None,
            "history": list(baseline["history"]) if baseline is not None else [],
        },
        "summary": {
            "shipped_modules": measurement["inventory"]["shipped_file_count"],
            "modules_checked": measurement["modules_checked"],
            "modules_at_100_percent": sum(m["at_100_percent"] for m in modules),
            "modules_below_100_percent": measured_below,
            "modules_below_floor": sum(m["below_floor"] for m in modules),
            "modules_exempt_from_floor": sum(
                m["floor_exemption"] is not None for m in modules
            ),
            "stale_exemptions": len(stale_exemptions),
            "failed_modules": failed,
            "unconfirmed_modules": sum(
                m["status"] == "unconfirmed" for m in modules
            ),
            "integrity_issue_count": len(integrity.get("issues", [])),
            "improved_modules": sum(bool(m["improvements"]) for m in modules),
            "stale_baseline_entries": len(stale),
            "global_issue_count": len(measurement["global_issues"]),
        },
        "global_issues": list(measurement["global_issues"]),
        "stale_baseline_entries": stale,
        "modules": modules,
    }


def build_report(
    *,
    root: Path,
    coverage_data: Mapping[str, Any],
    expected_file_count: int | None = None,
    baseline: Mapping[str, Any] | None = None,
    integrity: Mapping[str, Any] | None = None,
    exemptions: Mapping[str, str] | None = None,
    floor: int = COVERAGE_FLOOR_PERCENT,
) -> dict[str, Any]:
    """Measure and judge in one call, without writing or exiting."""
    measurement = measure(
        root=root,
        coverage_data=coverage_data,
        expected_file_count=expected_file_count,
    )
    if integrity is not None:
        measurement["integrity"] = dict(integrity)
    return evaluate(measurement, baseline, exemptions=exemptions, floor=floor)


def history_entry(
    *, mode: str, commit: str, reason: str, measurement: Mapping[str, Any],
    now: str | None = None,
) -> dict[str, str]:
    """The dated header every baseline write appends; refuses a blank reason."""
    if not reason.strip():
        raise BaselineError("a baseline write needs a non-empty --reason")
    if not commit.strip():
        raise BaselineError(
            "a baseline write needs the commit the data came from (--commit)"
        )
    written_at = now or _dt.datetime.now(_dt.timezone.utc).replace(
        microsecond=0).isoformat().replace("+00:00", "Z")
    coverage = measurement["coverage"]
    return {
        "written_at": written_at,
        "mode": mode,
        "commit": commit.strip(),
        "reason": reason.strip(),
        "coverage_timestamp": str(coverage.get("timestamp")),
        "coverage_version": str(coverage.get("version")),
    }


def _require_complete(measurement: Mapping[str, Any], action: str) -> None:
    if not measurement_is_complete(measurement):
        problems = "; ".join(
            [*measurement["global_issues"], *_integrity_issues(measurement)]
        ) or "incomplete"
        raise BaselineError(
            f"{action} refused: the measurement is incomplete ({problems})"
        )


def tighten_baseline(
    measurement: Mapping[str, Any],
    baseline: Mapping[str, Any],
    entry: Mapping[str, str],
    *,
    floor: int = COVERAGE_FLOOR_PERCENT,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Lower counts that fell; never raise one.  None when nothing changed."""
    _require_complete(measurement, "--update-baseline")
    stamp = entry["written_at"]
    modules = {path: dict(value) for path, value in baseline["modules"].items()}
    notes: list[str] = []
    changed = False
    shipped = {module["path"]: module for module in measurement["modules"]}
    for path in sorted(modules):
        if path not in shipped:
            del modules[path]
            changed = True
            notes.append(
                f"TRIMMED: {path}: no longer shipped; its allowance is gone, "
                "and a module that returns under any name must arrive at or "
                "above the floor"
            )
    for path in sorted(shipped):
        counts = shipped[path]["counts"]
        base = modules.get(path)
        if base is None:
            if is_below_floor(shipped[path], floor):
                notes.append(
                    f"NOT ADMITTED: {path}: a new module below the {floor}% "
                    f"floor ({_describe_counts(counts)}) is never added by "
                    "--update-baseline; cover it, or admit it after review "
                    "with --reset-baseline"
                )
            else:
                modules[path] = {**counts, "since": stamp}
                changed = True
                notes.append(
                    f"ADDED: {path}: new module at "
                    f"{coverage_percent(shipped[path]):.2f}%"
                )
            continue
        lowered = {
            field: counts[field] for field in COUNT_FIELDS
            if counts[field] < base[field]
        }
        for field in COUNT_FIELDS:
            if counts[field] > base[field]:
                notes.append(
                    f"KEPT: {path}: {COUNT_LABELS[field]} rose from "
                    f"{base[field]} to {counts[field]}; --update-baseline "
                    "never loosens"
                )
        if lowered:
            changes = ", ".join(
                f"{COUNT_LABELS[field]} {base[field]} -> {value}"
                for field, value in lowered.items()
            )
            modules[path] = {**base, **lowered, "since": stamp}
            changed = True
            notes.append(f"TIGHTENED: {path}: {changes}")
    if not changed:
        notes.append(
            "the baseline already matches; nothing to tighten, nothing written"
        )
        return None, notes
    history = [*baseline["history"], dict(entry)]
    return build_baseline_document(modules, history), notes


def reset_baseline(
    measurement: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    entry: Mapping[str, str],
    *,
    floor: int = COVERAGE_FLOOR_PERCENT,
) -> tuple[dict[str, Any], list[str]]:
    """Record the measurement as the baseline, naming every loosening."""
    _require_complete(measurement, "--reset-baseline")
    stamp = entry["written_at"]
    old = previous["modules"] if previous is not None else {}
    modules = {
        module["path"]: {**module["counts"], "since": stamp}
        for module in measurement["modules"]
    }
    measured = {module["path"]: module for module in measurement["modules"]}
    notes: list[str] = []
    for path in sorted(set(old) - set(modules)):
        notes.append(f"DROPPED: {path}: no longer shipped")
    for path in sorted(modules):
        counts = modules[path]
        if path not in old:
            if previous is not None and is_below_floor(measured[path], floor):
                notes.append(
                    f"LOOSENED: {path}: admitted below the {floor}% floor at "
                    f"{coverage_percent(measured[path]):.2f}% with "
                    + _describe_counts(counts)
                )
            elif previous is not None and any(
                    counts[field] for field in COUNT_FIELDS):
                notes.append(
                    f"ADMITTED: {path}: at "
                    f"{coverage_percent(measured[path]):.2f}% with "
                    + _describe_counts(counts)
                )
            continue
        for field in COUNT_FIELDS:
            if counts[field] > old[path][field]:
                notes.append(
                    f"LOOSENED: {path}: {COUNT_LABELS[field]} "
                    f"{old[path][field]} -> {counts[field]}"
                )
    history = [*previous["history"], dict(entry)] if previous is not None else [dict(entry)]
    return build_baseline_document(modules, history), notes


def retire_module(
    baseline: Mapping[str, Any],
    path: str,
    *,
    root: Path,
    commit: str,
    reason: str,
    now: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Remove one deleted module's entry; nothing else in the baseline moves.

    WHY A THIRD WRITE. Rule (d) fails while a module that no longer ships is
    still in the baseline, so a deleted file cannot carry its gaps away. The
    trim ``--update-baseline`` does is correct but refuses without a
    complete measurement of every shipped module, which only a full CI run
    produces -- so a deliberate deletion could not land green. A deleted
    file has no coverage to measure, and removing its entry loosens nothing
    that still ships; this is that one operation and no other.

    :param baseline: a validated baseline document.
    :param path: the module's repository-relative path, as the baseline
        lists it.
    :param root: repository root whose packaging decides what ships.
    :param commit: the commit that deleted the module.
    :param reason: why it was deleted; required.
    :param now: the write's timestamp, for tests.
    :returns: the new document and the notes to print.
    :raises BaselineError: without a reason or commit, for a path not in the
        baseline, or for a module that still ships.
    """
    if not reason.strip():
        raise BaselineError("a baseline write needs a non-empty --reason")
    if not commit.strip():
        raise BaselineError(
            "a baseline write needs the commit that deleted the module (--commit)"
        )
    path = Path(path).as_posix()
    modules = {key: dict(value) for key, value in baseline["modules"].items()}
    if path not in modules:
        raise BaselineError(
            f"--retire-module refused: {path} is not in the baseline"
        )
    if path in discover_shipped_python_files(root.resolve()):
        raise BaselineError(
            f"--retire-module refused: {path} still ships; only a module "
            "deleted from the package can be retired"
        )
    written_at = now or _dt.datetime.now(_dt.timezone.utc).replace(
        microsecond=0).isoformat().replace("+00:00", "Z")
    removed = modules.pop(path)
    entry = {
        "written_at": written_at,
        "mode": "retire",
        "commit": commit.strip(),
        "reason": reason.strip(),
        "retired": path,
        "coverage_timestamp": "none: a deleted module has no coverage",
        "coverage_version": "none",
    }
    gaps = ", ".join(
        f"{removed[field]} {COUNT_LABELS[field]}"
        for field in COUNT_FIELDS if removed[field]
    ) or "none"
    notes = [
        f"RETIRED: {path}: deleted on purpose; its allowance ({gaps}) is "
        "gone, and a module that returns under any name must arrive at or "
        "above the floor"
    ]
    history = [*baseline["history"], entry]
    return build_baseline_document(modules, history), notes


# -- rendering and CLI -----------------------------------------------------


def _integrity_line(integrity: Mapping[str, Any]) -> str:
    if not integrity.get("checked"):
        return "Shard integrity: not checked (no --shard-integrity given)"
    recovered = len(integrity["recovered"])
    count = integrity["shard_count"]
    kinds = integrity.get("loss_kinds") or {}
    lost = (
        "; lost processes by how they ended: "
        + ", ".join(f"{kind} {number}" for kind, number in kinds.items())
        if kinds else ""
    )
    if integrity["status"] == "complete":
        return (
            f"Shard integrity: complete ({count} of {count} shards finished; "
            f"{recovered} batch(es) recovered a lost worker's coverage{lost})"
        )
    return (
        f"Shard integrity: INCOMPLETE ({len(integrity['issues'])} problem(s); "
        f"{recovered} batch(es) recovered a lost worker's coverage{lost})"
    )


def render_text(report: Mapping[str, Any], notes: Sequence[str] = ()) -> str:
    """Render a concise human-readable twin of the JSON artifact."""
    summary = report["summary"]
    coverage = report["coverage"]
    baseline = report["baseline"]
    integrity = report["measurement_integrity"]
    headline = (
        "INCOMPLETE MEASUREMENT" if report["status"] == "incomplete"
        else str(report["status"]).upper()
    )
    floor = report.get("floor_percent", COVERAGE_FLOOR_PERCENT)
    exemptions = report.get("exemptions") or {
        "path": None, "modules": {}, "stale": [], "no_longer_needed": [],
    }
    lines = [
        f"spaCR shipped-module coverage ratchet: {headline}",
        f"Rule: every shipped module is at {floor}% or better and none of "
        f"them loses coverage; a module's bar is max({floor}%, what it "
        "already has).",
        f"Shipped modules: {summary['shipped_modules']}",
        f"Modules checked: {summary['modules_checked']} of "
        f"{summary['shipped_modules']}",
        f"Coverage rows: {coverage['input_rows']} "
        f"({coverage['repository_rows']} inside repository)",
        _integrity_line(integrity),
    ]
    if baseline["totals"] is None:
        lines.append(
            "Baseline: none (no module has a recorded allowance, so the "
            f"{floor}% floor is the whole gate)"
        )
    else:
        totals = baseline["totals"]
        last = baseline["history"][-1]
        lines.append(
            f"Baseline: {baseline['path'] or 'given'}, {totals['modules']} "
            f"modules ({totals['modules_at_100_percent']} at 100%, "
            f"{totals['modules_with_gaps']} with gaps), last written "
            f"{last['written_at']} by {last['mode']} at {last['commit'][:12]}"
        )
    lines += [
        f"Modules at 100%: {summary['modules_at_100_percent']}",
        f"Modules below the {floor}% floor: "
        f"{summary.get('modules_below_floor', 0)}",
        f"Modules exempt from the floor, with a stated reason: "
        f"{summary.get('modules_exempt_from_floor', 0)}"
        + (f" ({exemptions['path']})" if exemptions.get("path") else ""),
        f"Modules between the floor and 100% (more where coverage is "
        f"useful): {summary['modules_below_100_percent'] - summary.get('modules_below_floor', 0)}",
        f"Modules failing the ratchet: {summary['failed_modules']}",
        "Modules with unconfirmed rises (measurement incomplete): "
        f"{summary['unconfirmed_modules']}",
        f"Modules improved: {summary['improved_modules']}",
        f"Stale baseline entries: {summary['stale_baseline_entries']}",
    ]
    for issue in integrity.get("issues", []):
        lines.append(f"INCOMPLETE MEASUREMENT: {issue}")
    for note in integrity.get("recovered", []):
        lines.append(f"RECOVERED: {note}")
    for issue in report["global_issues"]:
        lines.append(f"ERROR: {issue}")
    exemption_file = exemptions.get("path") or "the exemption file"
    for path in exemptions.get("stale", []):
        lines.append(
            f"ERROR: {path}: exempted from the floor but no longer shipped; "
            f"an exemption cannot outlive its module. Remove the line from "
            f"{exemption_file}"
        )
    for path in exemptions.get("no_longer_needed", []):
        lines.append(
            f"EXEMPTION SPENT: {path}: it now clears the floor on its own, "
            f"so its line in {exemption_file} has nothing left to excuse"
        )
    for path in report["stale_baseline_entries"]:
        lines.append(
            f"ERROR: {path}: in the baseline but no longer shipped; a deleted "
            "or renamed module cannot take its allowance away silently. "
            "Trim it deliberately with --update-baseline --reason, or, for "
            "a module deleted on purpose, --retire-module <path> --reason"
        )
    for module in report["modules"]:
        for failure in module["failures"]:
            lines.append(f"ERROR: {module['path']}: {failure}")
    for module in report["modules"]:
        for failure in module["unconfirmed"]:
            lines.append(
                f"UNCONFIRMED: {module['path']}: {failure} (the measurement "
                "is incomplete, so this may be coverage data lost with a "
                "crashed worker rather than coverage the code lost)"
            )
    for module in report["modules"]:
        for improvement in module["improvements"]:
            lines.append(
                f"IMPROVED: {module['path']}: {improvement}; tighten the "
                "baseline with --update-baseline --reason"
            )
    for note in notes:
        lines.append(f"BASELINE: {note}")
    below = [
        module for module in report["modules"]
        if module["counts"] is not None and not module["at_100_percent"]
    ]
    below.sort(key=lambda module: (
        module["percent"] if module["percent"] is not None else 0.0,
        module["path"],
    ))
    if below:
        lines.append(
            f"Every module not at 100%, lowest first; the ones under {floor}% "
            "are the work, the rest get more coverage only where it is useful:"
        )
    for module in below:
        base = module["baseline"]
        allowance = (
            f"baseline {_slash(base)} since {base['since']}"
            if base is not None else "no baseline entry"
        )
        percent = (
            f"{module['percent']:.2f}%" if module["percent"] is not None
            else "unmeasured"
        )
        label = "BELOW FLOOR" if module.get("below_floor") else "GAP"
        excused = (
            f" [EXEMPT: {module['floor_exemption']}]"
            if module.get("floor_exemption") else ""
        )
        lines.append(
            f"{label}: {module['path']}: {percent}, "
            f"{_describe_counts(module['counts'])} ({allowance}){excused}"
        )
        lines.extend(f"    {gap}" for gap in module["gaps"])
    for entry in baseline["history"]:
        lines.append(
            f"Baseline history: {entry['written_at']} {entry['mode']} "
            f"{entry['commit'][:12]}: {entry['reason']}"
        )
    if report["status"] == "pass":
        lines.append(
            f"Every shipped module is at {floor}% or better, or says why it "
            "cannot be, and none of them lost coverage."
        )
    elif report["status"] == "incomplete":
        lines.append(
            "INCOMPLETE MEASUREMENT: coverage data was lost and not recovered, "
            "so this run cannot say whether any module lost coverage. It "
            "fails so the shards are re-run, not so a regression is chased; "
            "every UNCONFIRMED line above is unproven."
        )
    return "\n".join(lines) + "\n"


def _write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _head_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--coverage-json", type=Path,
        help="coverage.py JSON report to verify; required except with --retire-module",
    )
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(),
        help="repository root containing setup.py (default: current directory)",
    )
    parser.add_argument(
        "--expected-file-count", type=int,
        help="review lock for the current shipped-file inventory",
    )
    parser.add_argument(
        "--baseline", type=Path,
        help="per-module ratchet baseline written by this tool; without it "
        "no module has a recorded allowance and the floor is the whole gate",
    )
    parser.add_argument(
        "--floor-exemptions", type=Path,
        help="file of '<module path>: <why it cannot reach the floor>' lines "
        "for modules the floor cannot apply to (GPU-only, platform-only). It "
        "lifts the floor and nothing else: such a module still may not lose "
        "coverage. A missing file, a line with no reason, or an exemption for "
        "a module that no longer ships is an error",
    )
    writes = parser.add_mutually_exclusive_group()
    writes.add_argument(
        "--update-baseline", action="store_true",
        help="tighten the baseline to measured counts that fell (never loosens)",
    )
    writes.add_argument(
        "--reset-baseline", action="store_true",
        help="deliberately rewrite the baseline from the measurement (may loosen)",
    )
    writes.add_argument(
        "--retire-module", metavar="PATH",
        help="remove one entry for a module deleted on purpose; reads no "
        "coverage data and refuses a module that still ships",
    )
    parser.add_argument(
        "--reason", default="",
        help="why the baseline is being written; required with either write",
    )
    parser.add_argument(
        "--commit", default="",
        help="commit the coverage data came from (default: git HEAD of --root)",
    )
    parser.add_argument(
        "--shard-integrity", type=Path,
        help="directory holding every coverage shard's "
        "spacr-coverage-integrity.shard-NN.json; a shard that lost coverage "
        "data it did not recover makes the run an INCOMPLETE MEASUREMENT "
        "(exit 3). Needs --shard-count; ignored by --retire-module, which "
        "reads no coverage data",
    )
    parser.add_argument(
        "--shard-count", type=int,
        help="how many coverage shards must have left an integrity record",
    )
    parser.add_argument(
        "--json-out", type=Path,
        help="machine-readable ratchet report destination; required except "
        "with --retire-module",
    )
    parser.add_argument(
        "--text-out", type=Path,
        help="human-readable ratchet report destination; required except "
        "with --retire-module",
    )
    return parser


def _apply_write(
    args: argparse.Namespace,
    measurement: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any] | None, list[str]]:
    """Run --update-baseline or --reset-baseline; return the baseline to judge."""
    entry = history_entry(
        mode="update" if args.update_baseline else "reset",
        commit=args.commit or _head_commit(args.root),
        reason=args.reason,
        measurement=measurement,
    )
    document: dict[str, Any] | None
    if args.update_baseline:
        if baseline is None:
            raise BaselineError("--update-baseline needs an existing baseline")
        document, notes = tighten_baseline(
            measurement, baseline, entry, floor=COVERAGE_FLOOR_PERCENT,
        )
    else:
        document, notes = reset_baseline(
            measurement, baseline, entry, floor=COVERAGE_FLOOR_PERCENT,
        )
    if document is None:
        return baseline, notes
    write_baseline(args.baseline, document)
    notes.append(f"wrote {args.baseline} ({document['checksum']})")
    return document, notes


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.retire_module is not None:
        if args.baseline is None:
            parser.error("--retire-module needs --baseline")
        try:
            document, notes = retire_module(
                load_baseline(args.baseline), args.retire_module,
                root=args.root, commit=args.commit or _head_commit(args.root),
                reason=args.reason,
            )
            write_baseline(args.baseline, document)
        except (OSError, ValueError, SyntaxError) as exc:
            print(f"coverage ratchet could not retire: {exc}", file=sys.stderr)
            return 2
        for note in notes:
            print(f"BASELINE: {note}")
        print(f"BASELINE: wrote {args.baseline} ({document['checksum']})")
        return 0
    missing = [
        flag for flag, value in (
            ("--coverage-json", args.coverage_json),
            ("--json-out", args.json_out),
            ("--text-out", args.text_out),
        ) if value is None
    ]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))
    writing = args.update_baseline or args.reset_baseline
    if writing and args.baseline is None:
        parser.error("--update-baseline and --reset-baseline need --baseline")
    if (args.shard_integrity is None) != (args.shard_count is None):
        parser.error("--shard-integrity and --shard-count are given together")
    if args.shard_count is not None and args.shard_count < 1:
        parser.error("--shard-count must be at least 1")
    notes: list[str] = []
    try:
        coverage_data = json.loads(args.coverage_json.read_text(encoding="utf-8"))
        if not isinstance(coverage_data, Mapping):
            raise ValueError("coverage JSON root must be an object")
        exemptions: Mapping[str, str] = {}
        if args.floor_exemptions is not None:
            exemptions = load_floor_exemptions(args.floor_exemptions)
        baseline: Mapping[str, Any] | None = None
        if args.baseline is not None:
            try:
                baseline = load_baseline(args.baseline)
            except (OSError, BaselineError) as exc:
                if not args.reset_baseline:
                    raise
                notes.append(
                    f"previous baseline not carried forward ({exc}); "
                    "the history starts again"
                )
        measurement = measure(
            root=args.root,
            coverage_data=coverage_data,
            expected_file_count=args.expected_file_count,
        )
        if args.shard_integrity is not None:
            measurement["integrity"] = check_shard_integrity(
                args.shard_integrity, args.shard_count,
            )
        if writing:
            baseline, write_notes = _apply_write(args, measurement, baseline)
            notes.extend(write_notes)
        report = evaluate(
            measurement, baseline,
            baseline_path=str(args.baseline) if args.baseline else None,
            exemptions=exemptions,
            exemptions_path=(
                str(args.floor_exemptions) if args.floor_exemptions else None
            ),
        )
    except (OSError, ValueError, SyntaxError, tokenize.TokenError) as exc:
        for note in notes:
            print(f"BASELINE: {note}", file=sys.stderr)
        print(f"coverage ratchet could not run: {exc}", file=sys.stderr)
        return 2

    report["baseline_notes"] = notes
    json_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    text_report = render_text(report, notes)
    _write_report(args.json_out, json_text)
    _write_report(args.text_out, text_report)
    print(text_report, end="")
    return {"pass": 0, "fail": 1}.get(report["status"], INCOMPLETE_STATUS)


if __name__ == "__main__":
    raise SystemExit(main())
