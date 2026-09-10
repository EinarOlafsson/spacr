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
    r"|\bthe\s+instruction\s+(?:asks?|names?|requires?)\b"
    r"|\banother\s+agent(?:'s)?\s+work-in-progress\b"
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


def _api_builder():
    """The API extractor, imported the way `tools/` expects.

    `build_documentation_i18n` imports `build_i18n_catalogs` as a TOP-LEVEL
    name, so `tools/` has to be on the path -- importing it as
    `tools.build_documentation_i18n` finds the parent package and then fails
    on the sibling.
    """
    import importlib
    import sys
    from pathlib import Path

    tools = str(Path(__file__).resolve().parents[1] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("build_documentation_i18n")


#: Published DOCSTRINGS that still name an instruction, with the session
#: that owns each. A SHRINKING LIST: an entry may be removed and never
#: added, and the test below fails on anything not named here.
#:
#: WHY A LIST RATHER THAN SIX FIXES IN THIS COMMIT. Editing a docstring
#: stales all nine API catalogs, and 288's rule -- written after five
#: rebuild passes in one night -- is to finish the code and rebuild ONCE.
#: Six docstrings owned by two sessions is one batch. Four of these are the
#: other session's and it is actively in those files; racing it there to
#: save a rebuild that has to happen anyway would trade a merge conflict
#: for nothing.
#:
#: So the class is closed for NEW code today, the existing six are named,
#: and whoever writes the last of the six runs the one pass.
DOCSTRINGS_STILL_NAMING_AN_INSTRUCTION = {
    # home session
    "spacr.ops_cycles": "instruction 372",
    "spacr.ops_phenotype": "instruction 372",
    # work session -- 359, 01 and 327, all landed 2026-09-09/10
    "spacr.qt.app.open_at_the_measured_width": "INSTRUCTION 359",
    "spacr.qt.app.the_missing_pip_escape": "instruction 01",
    "spacr.qt.layout_policy": "INSTRUCTION 359",
    "spacr.qt.widgets.fractal_travel.TourPilot": "instruction 327",
}


def test_published_docstrings_do_not_expose_internal_provenance():
    """The same rule as below, applied where AutoAPI actually publishes.

    THE `#:` SWEEP BELOW SCANS COMMENT BLOCKS AND NOTHING ELSE, which is
    the gap the other session found: a module docstring opening
    "INSTRUCTION 359" is published exactly as visibly as a documented
    assignment and no test looked at it. Measured when the gap was found:
    six of 10,306 published docstrings, across both sessions.

    WHY IT MATTERS AND IS NOT PEDANTRY. 368 retired a `#:` comment reading
    "asked for on 2026-09-08" for this reason: an instruction number is a
    fact about how spaCR is DEVELOPED, and the reader of an API page is
    trying to use it. "INSTRUCTION 359" tells them nothing they can act on
    and implies a document they cannot open.
    """
    builder = _api_builder()

    unexpected = []
    for symbol, text in builder.public_docstrings().items():
        match = INTERNAL_PROVENANCE.search(text)
        if match and symbol not in DOCSTRINGS_STILL_NAMING_AN_INSTRUCTION:
            unexpected.append(f"{symbol}: {match.group(0)!r}")

    assert not unexpected, (
        "a published docstring names an instruction, or other development "
        "history a reader cannot act on:\n  " + "\n  ".join(unexpected) +
        "\n\nSay what the code does, not which item asked for it. If this "
        "is one of the six being cleared in a batch, add it to "
        "DOCSTRINGS_STILL_NAMING_AN_INSTRUCTION with its owner.")


def test_the_docstring_allowlist_only_shrinks():
    """An entry that has been cleared must be REMOVED, not left standing.

    A stale allowlist is worse than none: it reads as debt somebody is
    carrying deliberately when it is really debt somebody already paid.
    """
    builder = _api_builder()

    docs = builder.public_docstrings()
    cleared = []
    for symbol in DOCSTRINGS_STILL_NAMING_AN_INSTRUCTION:
        text = docs.get(symbol)
        if text is None or not INTERNAL_PROVENANCE.search(text):
            cleared.append(symbol)
    assert not cleared, (
        "these are named in DOCSTRINGS_STILL_NAMING_AN_INSTRUCTION and no "
        f"longer need to be: {cleared}. Delete the entries.")


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
        "The instruction names this empty state.",
        "Another agent's work-in-progress may add a theme.",
    )

    missed = [text for text in examples if not INTERNAL_PROVENANCE.search(text)]
    assert not missed, (
        "Internal provenance was not detected:\n" + "\n".join(missed)
    )
