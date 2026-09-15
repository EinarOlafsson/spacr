#!/usr/bin/env python3
"""Gate shipped-module coverage on a committed per-module ratchet baseline.

THE DENOMINATOR is packaging, not an editable manifest.  This tool reads the
``packages=find_packages(...)`` expression passed to ``setup()`` in
``setup.py``, calls :func:`setuptools.find_packages` with those literal
arguments, and includes every direct ``*.py`` child of every discovered
package.  A new package module therefore enters the gate automatically.

THE GATE (maintainer decision, 2026-09-15, item 288): no shipped module may
lose coverage, and no new module may arrive below 100%.  100% statement and
branch coverage per module stays the goal; the ratchet is what CI enforces.
For every shipped module four ABSOLUTE counts are measured -- uncovered
statements, uncovered branches, ``pragma: no cover`` comments and
coverage-excluded lines -- and compared with the same four counts recorded
for that module in the baseline (``tools/coverage_baseline.json``).
The run FAILS when:

  (a) any count of a module rises above its baseline count;
  (b) a module recorded at 100% in the baseline is no longer at 100%;
  (c) a shipped module that is not in the baseline is not at 100%;
  (d) a module is in the baseline but is no longer shipped.  A deleted or
      renamed file must not carry its allowance away silently, so the
      baseline has to be tightened deliberately (``--update-baseline``
      trims the entry; a module that returns under any name is new and must
      arrive at 100%);
  (e) anything cannot be measured: an unreadable coverage file, coverage
      without branch data, a shipped module with no valid coverage row, or
      a baseline this tool did not write.  "Checked N of M" must be M of M.

It PASSES, and prints an improvement notice, when a module's counts fall.
Every module not yet at 100% is listed on every run, with its baseline
allowance and the date that allowance was written.

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
                      modules that are at 100%, trims modules no longer
                      shipped.  It never raises a count and never admits a
                      module below 100%.
  --reset-baseline    deliberate regeneration from the measurement, which
                      CAN loosen; every loosened module is printed.  Use it
                      to seed the file or to admit a new module below 100%
                      after review.

Both refuse to write from an incomplete measurement.  To regenerate from a
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

REPORT_SCHEMA = "spacr.module-coverage-ratchet/v2"
BASELINE_SCHEMA = "spacr.module-coverage-baseline/v1"
PRAGMA_NO_COVER = re.compile(r"#\s*pragma\s*:\s*no\s*cover\b", re.IGNORECASE)

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
BASELINE_ABOUT = (
    "Written only by tools/verify_module_coverage.py. Do not edit: the "
    "checksum covers every other key and the gate refuses a file it did not "
    "write. Tighten with --update-baseline --reason; regenerate deliberately "
    "with --reset-baseline --reason (see the tool's docstring)."
)


class InventoryError(ValueError):
    """The packaging declaration cannot be interpreted safely."""


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


def measurement_is_complete(measurement: Mapping[str, Any]) -> bool:
    """True when every shipped module was measured and nothing global failed."""
    return not measurement["global_issues"] and (
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
) -> tuple[list[str], list[str]]:
    """Return (failures, improvements) for one measured module."""
    failures = list(module["measurement_errors"])
    improvements: list[str] = []
    counts = module["counts"]
    if counts is None or failures:
        return failures, improvements
    measured_full = not any(counts[field] for field in COUNT_FIELDS)
    if base is None:
        if not measured_full:
            failures.append(
                "new module is not at 100% and is not in the baseline: "
                + _describe_counts(counts)
            )
        return failures, improvements
    if not any(base[field] for field in COUNT_FIELDS) and not measured_full:
        failures.append(
            "was at 100% in the baseline and now has "
            + _describe_counts(counts)
        )
        return failures, improvements
    for field in COUNT_FIELDS:
        label = COUNT_LABELS[field]
        if counts[field] > base[field]:
            failures.append(
                f"{label} rose from {base[field]} to {counts[field]}"
            )
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
) -> dict[str, Any]:
    """Judge a measurement against a baseline (None: every module is new)."""
    base_modules: Mapping[str, Mapping[str, Any]] = (
        baseline["modules"] if baseline is not None else {}
    )
    shipped = set(measurement["inventory"]["files"])
    modules: list[dict[str, Any]] = []
    for measured in measurement["modules"]:
        entry = dict(measured)
        base = base_modules.get(entry["path"])
        failures, improvements = _judge_module(entry, base)
        counts = entry["counts"]
        entry["baseline"] = dict(base) if base is not None else None
        entry["at_100_percent"] = (
            counts is not None
            and not entry["measurement_errors"]
            and not any(counts[field] for field in COUNT_FIELDS)
        )
        entry["failures"] = failures
        entry["improvements"] = improvements
        entry["status"] = "fail" if failures else "pass"
        modules.append(entry)
    stale = sorted(path for path in base_modules if path not in shipped)
    failed = sum(module["status"] == "fail" for module in modules)
    measured_below = sum(
        module["counts"] is not None
        and not module["measurement_errors"]
        and not module["at_100_percent"]
        for module in modules
    )
    passed = not measurement["global_issues"] and failed == 0 and not stale
    return {
        "schema": REPORT_SCHEMA,
        "status": "pass" if passed else "fail",
        "root": measurement["root"],
        "coverage": dict(measurement["coverage"]),
        "inventory": dict(measurement["inventory"]),
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
            "failed_modules": failed,
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
) -> dict[str, Any]:
    """Measure and judge in one call, without writing or exiting."""
    return evaluate(
        measure(
            root=root,
            coverage_data=coverage_data,
            expected_file_count=expected_file_count,
        ),
        baseline,
    )


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
        problems = "; ".join(measurement["global_issues"]) or "incomplete"
        raise BaselineError(
            f"{action} refused: the measurement is incomplete ({problems})"
        )


def tighten_baseline(
    measurement: Mapping[str, Any],
    baseline: Mapping[str, Any],
    entry: Mapping[str, str],
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
                "and a module that returns under any name must arrive at 100%"
            )
    for path in sorted(shipped):
        counts = shipped[path]["counts"]
        base = modules.get(path)
        if base is None:
            if any(counts[field] for field in COUNT_FIELDS):
                notes.append(
                    f"NOT ADMITTED: {path}: a new module below 100% "
                    f"({_describe_counts(counts)}) is never added by "
                    "--update-baseline; cover it, or admit it after review "
                    "with --reset-baseline"
                )
            else:
                modules[path] = {**counts, "since": stamp}
                changed = True
                notes.append(f"ADDED: {path}: new module at 100%")
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
) -> tuple[dict[str, Any], list[str]]:
    """Record the measurement as the baseline, naming every loosening."""
    _require_complete(measurement, "--reset-baseline")
    stamp = entry["written_at"]
    old = previous["modules"] if previous is not None else {}
    modules = {
        module["path"]: {**module["counts"], "since": stamp}
        for module in measurement["modules"]
    }
    notes: list[str] = []
    for path in sorted(set(old) - set(modules)):
        notes.append(f"DROPPED: {path}: no longer shipped")
    for path in sorted(modules):
        counts = modules[path]
        if path not in old:
            if previous is not None and any(counts[field] for field in COUNT_FIELDS):
                notes.append(
                    f"LOOSENED: {path}: admitted below 100% with "
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


# -- rendering and CLI -----------------------------------------------------


def render_text(report: Mapping[str, Any], notes: Sequence[str] = ()) -> str:
    """Render a concise human-readable twin of the JSON artifact."""
    summary = report["summary"]
    coverage = report["coverage"]
    baseline = report["baseline"]
    lines = [
        f"spaCR shipped-module coverage ratchet: {str(report['status']).upper()}",
        "Rule: no shipped module loses coverage, and no new module arrives "
        "below 100%.",
        f"Shipped modules: {summary['shipped_modules']}",
        f"Modules checked: {summary['modules_checked']} of "
        f"{summary['shipped_modules']}",
        f"Coverage rows: {coverage['input_rows']} "
        f"({coverage['repository_rows']} inside repository)",
    ]
    if baseline["totals"] is None:
        lines.append("Baseline: none (every module must be at 100%)")
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
        "Modules not yet at 100% (100% per module is still the goal): "
        f"{summary['modules_below_100_percent']}",
        f"Modules failing the ratchet: {summary['failed_modules']}",
        f"Modules improved: {summary['improved_modules']}",
        f"Stale baseline entries: {summary['stale_baseline_entries']}",
    ]
    for issue in report["global_issues"]:
        lines.append(f"ERROR: {issue}")
    for path in report["stale_baseline_entries"]:
        lines.append(
            f"ERROR: {path}: in the baseline but no longer shipped; a deleted "
            "or renamed module cannot take its allowance away silently. "
            "Trim it deliberately with --update-baseline --reason"
        )
    for module in report["modules"]:
        for failure in module["failures"]:
            lines.append(f"ERROR: {module['path']}: {failure}")
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
    if below:
        lines.append(
            "Not yet at 100%, every module and its allowance "
            "(information; the goal is 100%):"
        )
    for module in below:
        base = module["baseline"]
        allowance = (
            f"baseline {_slash(base)} since {base['since']}"
            if base is not None else "no baseline entry"
        )
        lines.append(
            f"GAP: {module['path']}: {_describe_counts(module['counts'])} "
            f"({allowance})"
        )
        lines.extend(f"    {gap}" for gap in module["gaps"])
    for entry in baseline["history"]:
        lines.append(
            f"Baseline history: {entry['written_at']} {entry['mode']} "
            f"{entry['commit'][:12]}: {entry['reason']}"
        )
    if report["status"] == "pass":
        lines.append(
            "No shipped module lost coverage and no new module arrived "
            "below 100%."
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
        "--coverage-json", type=Path, required=True,
        help="coverage.py JSON report to verify",
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
        "every module must be at 100%%",
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
    parser.add_argument(
        "--reason", default="",
        help="why the baseline is being written; required with either write",
    )
    parser.add_argument(
        "--commit", default="",
        help="commit the coverage data came from (default: git HEAD of --root)",
    )
    parser.add_argument(
        "--json-out", type=Path, required=True,
        help="machine-readable ratchet report destination",
    )
    parser.add_argument(
        "--text-out", type=Path, required=True,
        help="human-readable ratchet report destination",
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
        document, notes = tighten_baseline(measurement, baseline, entry)
    else:
        document, notes = reset_baseline(measurement, baseline, entry)
    if document is None:
        return baseline, notes
    write_baseline(args.baseline, document)
    notes.append(f"wrote {args.baseline} ({document['checksum']})")
    return document, notes


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    writing = args.update_baseline or args.reset_baseline
    if writing and args.baseline is None:
        parser.error("--update-baseline and --reset-baseline need --baseline")
    notes: list[str] = []
    try:
        coverage_data = json.loads(args.coverage_json.read_text(encoding="utf-8"))
        if not isinstance(coverage_data, Mapping):
            raise ValueError("coverage JSON root must be an object")
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
        if writing:
            baseline, write_notes = _apply_write(args, measurement, baseline)
            notes.extend(write_notes)
        report = evaluate(
            measurement, baseline,
            baseline_path=str(args.baseline) if args.baseline else None,
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
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
