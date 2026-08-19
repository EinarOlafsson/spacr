"""Prevent internal development history from leaking into public API docs."""

from __future__ import annotations

import ast
import re
from pathlib import Path


PACKAGE = Path(__file__).parents[1] / "spacr"
INTERNAL_PROVENANCE = re.compile(
    r"(?i)(\binstruction\s+\d+|\bmaintainer(?:'s)?\b|\.claude/skills|"
    r"\b(?:asked for|requested|reported)(?:\s+on)?\s+20\d\d-)"
)


def _public_docstrings(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        if not isinstance(node, ast.Module) and node.name.startswith("_"):
            continue
        docstring = ast.get_docstring(node, clean=False)
        if docstring:
            yield getattr(node, "lineno", 1), docstring


def test_public_docstrings_do_not_expose_internal_task_history():
    failures = []
    for path in PACKAGE.rglob("*.py"):
        for line, docstring in _public_docstrings(path):
            match = INTERNAL_PROVENANCE.search(docstring)
            if match:
                failures.append(
                    f"{path.relative_to(PACKAGE.parent)}:{line}: {match.group(0)!r}"
                )

    assert not failures, "Internal task history found in public docstrings:\n" + "\n".join(
        failures
    )
