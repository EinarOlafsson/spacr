"""Prevent internal development history from leaking into public API docs."""

from __future__ import annotations

import ast
import re
from pathlib import Path

PACKAGE = Path(__file__).parents[1] / "spacr"
INTERNAL_PROVENANCE = re.compile(
    r"(?i)(\binstruction\s+\d+|\bmaintainer(?:'s)?\b|\.claude/skills|"
    r"\b(?:asked for|requested|reported|measured|changed)(?:\s+on)?\s+20\d\d-|"
    r"\b(?:the|this)\s+instruction(?:'s)?\s+(?:asked|words|is about)|"
    r"\banother agent(?:'s)?\s+work-in-progress|"
    r"\b(?:because that is )?where\s+(?:the\s+)?user\s+"
    r"(?:asked|requested|wanted)|"
    r"\b(?:feature|layout|panel|widget|style|theme|position|location|"
    r"drop targets?|categories)\b[^.\n]{0,100}\b(?:the\s+)?user\s+"
    r"(?:asked|requested|wanted|picked)\b)"
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


def _all_docstrings(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
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


def test_internal_docstrings_describe_behavior_not_task_history():
    failures = []
    for path in PACKAGE.rglob("*.py"):
        for line, docstring in _all_docstrings(path):
            match = INTERNAL_PROVENANCE.search(docstring)
            if match:
                failures.append(
                    f"{path.relative_to(PACKAGE.parent)}:{line}: {match.group(0)!r}"
                )

    assert not failures, "Internal task history found in docstrings:\n" + "\n".join(
        failures
    )


def test_provenance_pattern_allows_runtime_request_vocabulary():
    examples = (
        "Return the format the user requested.",
        "Rendering instructions for the selected theme.",
        "Show an installation instruction when the dependency is absent.",
        "Report values measured in micrometres.",
    )

    unexpected = [text for text in examples if INTERNAL_PROVENANCE.search(text)]
    assert not unexpected, "Legitimate API prose was rejected:\n" + "\n".join(
        unexpected
    )


def test_provenance_pattern_catches_creator_directed_history():
    examples = (
        "Measured 2026-08-18 while repairing the dialog.",
        "The instruction's words require this layout.",
        "Another agent's work-in-progress may add a theme.",
        "The panel sits here because the user asked for it.",
    )

    missed = [text for text in examples if not INTERNAL_PROVENANCE.search(text)]
    assert not missed, "Internal provenance was not detected:\n" + "\n".join(
        missed
    )
