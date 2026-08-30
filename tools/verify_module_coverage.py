#!/usr/bin/env python3
"""Require exact statement and branch coverage for every shipped module.

The coverage denominator is packaging, not an editable manifest.  This tool
reads the ``packages=find_packages(...)`` expression passed to ``setup()`` in
``setup.py``, calls :func:`setuptools.find_packages` with those literal
arguments, and includes every direct ``*.py`` child of every discovered
package.  A new package module therefore enters the gate automatically.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from setuptools import find_packages

REPORT_SCHEMA = "spacr.module-coverage-ratchet/v1"
PRAGMA_NO_COVER = re.compile(r"#\s*pragma\s*:\s*no\s*cover\b", re.IGNORECASE)


class InventoryError(ValueError):
    """The packaging declaration cannot be interpreted safely."""


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


def build_report(
    *,
    root: Path,
    coverage_data: Mapping[str, Any],
    expected_file_count: int | None = None,
) -> dict[str, Any]:
    """Build the deterministic ratchet report without writing or exiting."""
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

    modules: list[dict[str, Any]] = []
    for relative in shipped:
        row = rows.get(relative)
        pragmas = _pragma_lines(root / relative)
        issues: list[str] = []
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
        }
        if row is None:
            issues.append("missing coverage row")
        else:
            summary = row.get("summary", {})
            if not isinstance(summary, Mapping):
                summary = {}
                issues.append("coverage row has no summary object")
            missing_lines = row.get("missing_lines", [])
            missing_branches = row.get("missing_branches", [])
            excluded_lines = row.get("excluded_lines", [])
            if not isinstance(missing_lines, list):
                missing_lines = []
                issues.append("coverage row has invalid missing_lines")
            if not isinstance(missing_branches, list):
                missing_branches = []
                issues.append("coverage row has invalid missing_branches")
            if not isinstance(excluded_lines, list):
                excluded_lines = []
                issues.append("coverage row has invalid excluded_lines")

            num_statements = summary.get("num_statements")
            covered_lines = summary.get("covered_lines")
            num_branches = summary.get("num_branches")
            covered_branches = summary.get("covered_branches")
            entry.update({
                "num_statements": num_statements,
                "covered_lines": covered_lines,
                "missing_lines": missing_lines,
                "num_branches": num_branches,
                "covered_branches": covered_branches,
                "missing_branches": missing_branches,
                "excluded_lines": excluded_lines,
            })
            required_counts = {
                "num_statements": num_statements,
                "covered_lines": covered_lines,
                "num_branches": num_branches,
                "covered_branches": covered_branches,
            }
            invalid_counts = [
                name for name, value in required_counts.items()
                if not isinstance(value, int) or isinstance(value, bool) or value < 0
            ]
            if invalid_counts:
                issues.append(
                    "coverage summary has invalid counts: " + ", ".join(invalid_counts)
                )
            else:
                if missing_lines or covered_lines != num_statements:
                    issues.append(
                        "uncovered statements: " + _line_ranges(missing_lines)
                    )
                if missing_branches or covered_branches != num_branches:
                    issues.append(
                        "uncovered branches: " + _branch_list(missing_branches)
                    )
            if excluded_lines or summary.get("excluded_lines", 0) != 0:
                issues.append(
                    "coverage-excluded lines: " + _line_ranges(excluded_lines)
                )
        if pragmas:
            issues.append("pragma: no cover comments: " + _line_ranges(pragmas))
        entry["issues"] = issues
        entry["status"] = "pass" if not issues else "fail"
        modules.append(entry)

    failed_modules = sum(module["status"] == "fail" for module in modules)
    passed = not global_issues and failed_modules == 0
    return {
        "schema": REPORT_SCHEMA,
        "status": "pass" if passed else "fail",
        "root": str(root),
        "coverage": {
            "version": meta.get("version"),
            "branch_coverage": meta.get("branch_coverage"),
            "input_rows": len(raw_files),
            "repository_rows": len(rows),
        },
        "inventory": {
            "expected_file_count": expected_file_count,
            "shipped_file_count": len(shipped),
            "files": shipped,
        },
        "summary": {
            "passed_modules": len(modules) - failed_modules,
            "failed_modules": failed_modules,
            "global_issue_count": len(global_issues),
        },
        "global_issues": global_issues,
        "modules": modules,
    }


def render_text(report: Mapping[str, Any]) -> str:
    """Render a concise human-readable twin of the JSON artifact."""
    summary = report["summary"]
    inventory = report["inventory"]
    coverage = report["coverage"]
    lines = [
        f"spaCR shipped-module coverage ratchet: {str(report['status']).upper()}",
        f"Shipped modules: {inventory['shipped_file_count']}",
        f"Coverage rows: {coverage['input_rows']} "
        f"({coverage['repository_rows']} inside repository)",
        f"Modules passing: {summary['passed_modules']}",
        f"Modules failing: {summary['failed_modules']}",
    ]
    for issue in report["global_issues"]:
        lines.append(f"ERROR: {issue}")
    for module in report["modules"]:
        for issue in module["issues"]:
            lines.append(f"ERROR: {module['path']}: {issue}")
    if report["status"] == "pass":
        lines.append(
            "Every shipped module has 100% statement and branch coverage; "
            "no line is excluded and no no-cover pragma exists."
        )
    return "\n".join(lines) + "\n"


def _write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--json-out", type=Path, required=True,
        help="machine-readable ratchet report destination",
    )
    parser.add_argument(
        "--text-out", type=Path, required=True,
        help="human-readable ratchet report destination",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        coverage_data = json.loads(args.coverage_json.read_text(encoding="utf-8"))
        if not isinstance(coverage_data, Mapping):
            raise ValueError("coverage JSON root must be an object")
        report = build_report(
            root=args.root,
            coverage_data=coverage_data,
            expected_file_count=args.expected_file_count,
        )
    except (OSError, ValueError, SyntaxError, tokenize.TokenError) as exc:
        print(f"coverage ratchet could not run: {exc}", file=sys.stderr)
        return 2

    json_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    text_report = render_text(report)
    _write_report(args.json_out, json_text)
    _write_report(args.text_out, text_report)
    print(text_report, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
