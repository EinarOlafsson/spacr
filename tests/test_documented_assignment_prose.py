"""Keep generated assignment documentation free of development history."""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from tokenize import COMMENT, generate_tokens

PACKAGE = Path(__file__).parents[1] / "spacr"

INTERNAL_PROVENANCE = re.compile(
    r"(?ix)(?:"
    r"\binstruction\s+\d+\b|"
    r"\bmaintainer(?:'s)?\s+"
    r"(?:decision|design|example|listed|own|reported|restatement|"
    r"run|screen|tsg101)\b|"
    r"\.claude/skills|"
    r"\b(?:standing\s+handoff|another\s+territor(?:y|ies))\b|"
    r"\bhanded\s+over\s+by\s+the\s+agent\b|"
    r"\bthe\s+skill(?:'s)?\s+(?:is|measures?|rules?|states?)\b|"
    r"\bin\s+the\s+words\s+(?:the\s+)?user\b|"
    r"\b(?:asked\s+for|requested|reported|measured|taken)"
    r"(?:\s+on)?\s+20\d\d-\d\d-\d\d\b|"
    r"\b(?:as\s+asked\s+for|asked\s+for\s+by\s+name)\b|"
    r"\b(?:changed|chosen|raised|reported|requested|set|sharper)\b"
    r"[^.\n]{0,80}\bon\s+request\b|"
    r"\bthis\s+instruction\s+(?:exists|is\s+correcting)\b"
    r")"
)


def _documented_blocks(path: Path):
    """Yield ``(line, text)`` for contiguous ``#:`` comment blocks."""
    source = path.read_text(encoding="utf-8")
    block = []
    start = 0
    previous_line = 0
    previous_column = -1

    for token in generate_tokens(StringIO(source).readline):
        if token.type != COMMENT or not token.string.startswith("#:"):
            continue

        line_number, column = token.start
        text = token.string[2:].strip()

        # An inline ``value = 1  #: explanation`` documents that one
        # assignment and cannot be contiguous with a preceding standalone
        # block.
        if token.line[:column].strip():
            if block:
                yield start, "\n".join(block)
            block = []
            start = 0
            previous_line = 0
            previous_column = -1
            yield line_number, text
            continue

        contiguous = (
            block
            and line_number == previous_line + 1
            and column == previous_column
        )
        if block and not contiguous:
            yield start, "\n".join(block)
            block = []
        if not block:
            start = line_number
        block.append(text)
        previous_line = line_number
        previous_column = column

    if block:
        yield start, "\n".join(block)


def test_documented_assignments_do_not_expose_internal_provenance():
    failures = []
    for path in PACKAGE.rglob("*.py"):
        for line, text in _documented_blocks(path):
            match = INTERNAL_PROVENANCE.search(text)
            if match:
                failures.append(
                    f"{path.relative_to(PACKAGE.parent)}:{line}: "
                    f"{match.group(0)!r}"
                )

    assert not failures, (
        "Internal development history found in documented assignments:\n"
        + "\n".join(failures)
    )


def test_provenance_pattern_allows_public_api_vocabulary():
    examples = (
        "Download model weights on request.",
        "An antimicrobial agent that inhibits parasite growth.",
        "Measured territory, in pixels, assigned to each object.",
        "Worker handoff timeout, in seconds.",
        "Package maintainer contact address.",
        "The skill score reported by the classifier.",
    )

    unexpected = [text for text in examples if INTERNAL_PROVENANCE.search(text)]
    assert not unexpected, (
        "Legitimate API prose was rejected:\n" + "\n".join(unexpected)
    )


def test_provenance_pattern_catches_internal_history():
    examples = (
        "Added for instruction 172.",
        "The maintainer's own run used this value.",
        "Copied from .claude/skills/api-writer.",
        "A standing handoff because this module was another territory.",
        "Handed over by the agent that fixed the panel.",
        "The skill's rule requires this order.",
        "In the words the user asked for.",
        "Reported on 2026-08-19.",
        "Set on request once the registry grew.",
        "This instruction is correcting the layout.",
    )

    missed = [text for text in examples if not INTERNAL_PROVENANCE.search(text)]
    assert not missed, (
        "Internal provenance was not detected:\n" + "\n".join(missed)
    )
