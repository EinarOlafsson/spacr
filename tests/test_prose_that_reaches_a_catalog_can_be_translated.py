"""Prose bound for a catalog is checked at edit time, not after 65 minutes.

INSTRUCTION 394. Both i18n catalog lanes gate on translation QUALITY, and
that verdict exists only after a ~40 minute nine-language model build plus a
~25 minute audit. A docstring paragraph added between a build and its audit
therefore costs the whole cycle again before anyone learns it was
untranslatable -- four cycles in one evening, three of them that shape.

This runs in about two seconds, beside
``tests/test_every_docstring_is_valid_rst.py``, which is the same shape of
check one stage earlier: prose that will break a later build fails a test now.

WHAT IT DECIDES, AND WHAT IT CANNOT. One cause of three. The other two are
not visible in the English at all, and the evidence for that is not an
argument: ``--repair-api-blocks`` once cleared about thirty-seven blocks per
language in ``ko`` and ``is`` WITH NO SOURCE EDIT. Every assertion below
carries that sentence, because it is the one a reader needs on the day this
test passes and the build still fails.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "check_translatable_prose.py"

#: Said in every failure message rather than in this docstring, because a
#: docstring is not read by the person whose test just went red.
NOT_A_GREEN_BUILD = (
    "NOTE: this test decides ONE of the three causes of a catalog rejection "
    "-- a project label in a slot where a translator expects a name. It "
    "cannot see the other two (model brittleness on particular token pairs, "
    "and the expansion layer injecting a word), because neither is present "
    "in the English. Passing it does NOT predict a green build."
)


def _check():
    """Import the real checker, so this test cannot drift from the tool."""
    spec = importlib.util.spec_from_file_location(
        "_spacr_translatable_prose", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def check():
    """The checker module, loaded once."""
    if not TOOL.is_file():
        pytest.fail(f"the checker is missing: {TOOL}")
    return _check()


def test_no_catalog_bound_block_puts_a_bare_project_label_in_subject_position(
        check):
    """The check that closes the 65-minute loop for the one cause it can.

    A label standing as the grammatical SUBJECT of a sentence -- "372 states
    the contract", "B2: run a segmenter", "C4 samples phenotype channels" --
    is an opaque token where a translator expects a name, so it is
    translated, renumbered or dropped and the block fails the audit.
    """
    builder = check._builder()
    flagged = []
    examined = 0
    for symbol, text in builder.public_docstrings().items():
        blocks, _layout = builder.translatable_blocks(text)
        for index, block in enumerate(blocks):
            examined += 1
            for signal, why, fix in check.check_block(block):
                flagged.append(f"{symbol}[{index}]: {why}\n      fix: {fix}")

    assert examined > 1000, (
        f"only {examined} blocks were examined, so this test is not looking "
        "at the corpus it claims to -- check that public_docstrings() and "
        "translatable_blocks() still resolve")
    assert not flagged, (
        f"{len(flagged)} block(s) of catalog-bound prose put a bare project "
        "label where a translator expects a name:\n\n  "
        + "\n\n  ".join(flagged[:12])
        + f"\n\n{NOT_A_GREEN_BUILD}")


def test_the_check_still_names_every_failure_it_was_built_from(check):
    """CALIBRATION, RUN EVERY TIME, because a check nobody re-calibrates is a
    check that quietly stops matching the gate.

    These four blocks really failed the catalog gate on 2026-09-11. Their
    text is carried verbatim rather than fetched from git: the revision a
    block failed AT is not the revision it is easiest to reach, and
    recovering "the failing input" from whatever commit comes to hand
    produces the wrong input. That mistake was made once already -- two of
    these four had been rewritten by the time they were listed, so the
    recovered text was the FIXED text and the pattern looked like three of
    four rather than four of four.
    """
    missed = [name for name, block in check.CALIBRATION
              if not check.check_block(block)]
    assert not missed, (
        "the check no longer names blocks that really failed the gate: "
        + ", ".join(missed)
        + "\n\nEither the signal was narrowed past the evidence, or the "
        "shape of the failure moved. Re-derive it from the diffs before "
        "loosening anything.\n\n" + NOT_A_GREEN_BUILD)


@pytest.mark.parametrize("prose", [
    "A4 DOES NOT NEED MASKS.",
    "B9 - Control Charts:",
    "Phase B segments, once, on the composite.",
    "372's PART 6-A names the two halves.",
    "333 tiles of 1480 px are written per well.",
    "389 genes survived the filter.",
    "The scorecard in graph form, for Hugging Face and for the API page.",
])
def test_prose_that_ships_today_is_not_flagged(check, prose):
    """THE NEGATIVE HALF, AND IT IS THE HALF THAT DECIDES WHETHER THIS LIVES.

    A check that flags five per cent of a 38,000-block corpus flags 1,900
    blocks and is switched off within a day. Every string here is real prose
    from the current tree that passes the gate, and each one was a measured
    false positive of some earlier draft of the signal: the heading forms
    ("A4 DOES NOT NEED MASKS.") are separated only by requiring a lowercase
    continuation, and the quantities ("333 tiles", "389 genes") only by
    requiring an attribution verb after a number that names a real
    instruction file.
    """
    assert not check.check_block(prose), (
        f"this ships today and passes the real gate, but the check flags it: "
        f"{prose!r}\n\nA false positive here is worse than a missed "
        "detection, because it is what gets the whole check switched off."
    )
