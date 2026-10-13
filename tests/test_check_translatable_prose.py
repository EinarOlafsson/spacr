"""The checker must read the string the MODEL is handed, not the docstring.

INSTRUCTION 394. ``tests/test_prose_that_reaches_a_catalog_can_be_translated``
already guards the one signal the checker decides. This file guards the two
things added after it: that the signal and the registry are evaluated on
``_api_translation_source(block)`` -- the expansion, which is what
``_translate_api_documents`` actually hands the model -- and that the registry
of known-bad tokens reports rather than decides.

WHY THE EXPANSION IS NOT A DETAIL. On this checkout the expansion rewrites
7,824 of 38,368 blocks. It injects words nobody wrote (1,382 ``microplate``,
1,185 ``view``) and erases others (832 ``screen``, 161 ``worker``). For
``worker``, a word the Icelandic checkpoint declines, it erases the word from
161 blocks and INVENTS it in 8 -- so a registry checked against the docstring
is wrong in both directions on 169 blocks.

AND EXPANDING NAIVELY IS ALSO WRONG, WHICH IS THE REGRESSION THIS FILE
EXISTS FOR. ``_lower_shouted_emphasis`` runs inside the expansion, so
"A4 DOES NOT NEED MASKS." reaches the model as "A4 does not need masks." --
the defect shape exactly. Both corpus blocks of that shape were read against
the cached model output for all nine languages: both translate and the label
survives verbatim in every target. The heading exemption therefore belongs to
the author's text while the signal reads the model's, and a future
simplification that drops that distinction buys two false positives and no
true one.

EVERY ASSERTION CARRIES ``NOT_A_GREEN_BUILD``, imported from the tool rather
than restated, because a green run here still means only that one cause of
several is absent.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "check_translatable_prose.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "_spacr_translatable_prose_expanded", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def check():
    """The checker module, loaded once."""
    if not TOOL.is_file():
        pytest.fail(f"the checker is missing: {TOOL}")
    return _load()


@pytest.fixture(scope="module")
def caveat(check):
    """The sentence every failure here has to carry."""
    return "\n\n" + check.NOT_A_GREEN_BUILD


# ---------------------------------------------------------------------------
# cause 3 -- the expansion layer decides which English is checked
# ---------------------------------------------------------------------------
def test_the_expansion_is_taken_from_the_builder_and_not_reimplemented(
        check, caveat):
    """``expanded_source`` must BE the builder's function, not resemble it.

    A checker that rewrites its input differently from the tool it checks
    measures its own rewriter. That mistake has already been made three times
    on this problem in one night: 1,569 stale reviewed records where the true
    number was one; a docstring's first block read as its first line; and a
    density ranking inverted by a backtick regex that could not see the bare
    snake_case ``_PROTECT_RE`` protects.
    """
    builder = check._builder()
    for block in (
        "The crops are written before the screen reads them.",
        "Retire spaCR's own idle workers and lower its thread counts.",
        "B2: run a segmenter over each window.",
    ):
        assert check.expanded_source(block) == (
            builder._api_translation_source(block)), (
            "expanded_source diverged from the builder's own expansion, so "
            "this check is measuring its own rewriter" + caveat)


def test_the_expansion_really_changes_the_english_the_model_sees(
        check, caveat):
    """The premise of the whole file, asserted rather than assumed."""
    source = "False is a test seam; production image loads are threaded."
    expanded = check.expanded_source(source)
    assert expanded != source, (
        "this block is documented as one the expansion rewrites; if the "
        "expansion no longer touches it, re-measure before trusting any "
        "claim in this file about what the model sees" + caveat)
    assert "worker" in expanded, (
        f"expected the expansion to inject 'worker' here, got {expanded!r}"
        + caveat)


def test_a_registry_token_the_author_never_wrote_is_still_found(
        check, caveat):
    """CAUSE 3 AND CAUSE 2 ARE THE SAME EVENT WHEN THE EXPANSION INJECTS.

    This is the shape of the German ``builder`` leak -- a word the tool put
    into the model's input, which the model then kept -- reproduced on this
    checkout with the word ``worker`` and the Icelandic checkpoint. Checking
    the docstring finds nothing, because the docstring does not contain the
    word.
    """
    source = "False is a test seam; production image loads are threaded."
    assert not ({"worker", "workers"} & check._lexical_words(source)), (
        "the control is broken: this sentence is supposed to contain no "
        "spelling of the registry token at all" + caveat)

    findings = check.registry_findings(source)
    signals = [signal for signal, _why, _fix in findings]
    assert "KNOWN_BAD_TOKENS/is/worker" in signals, (
        "the registry missed a token the expansion injected, which is the "
        f"one case only expansion-aware checking can reach: {findings!r}"
        + caveat)
    why = next(why for signal, why, _fix in findings
               if signal == "KNOWN_BAD_TOKENS/is/worker")
    assert "NOBODY WROTE" in why, (
        "an injected token must say so; rewording the docstring will not "
        f"remove it and the author needs to be told that: {why!r}" + caveat)


def test_a_token_the_author_did_write_is_not_reported_as_injected(
        check, caveat):
    """THE OTHER HALF, AND THE ONE THAT KEEPS THE MESSAGE HONEST.

    "workers" in the docstring arriving as "worker" in the expansion is the
    expansion editing the author's word, not inventing one. Telling somebody
    nobody wrote their own word is the kind of wrong message that gets a
    check switched off.
    """
    source = "Retire spaCR's own idle workers and lower its thread counts."
    findings = [(signal, why) for signal, why, _fix
                in check.registry_findings(source)
                if signal == "KNOWN_BAD_TOKENS/is/worker"]
    assert findings, (
        "the author wrote 'workers' and the registry should still report it"
        + caveat)
    assert all("NOBODY WROTE" not in why for _signal, why in findings), (
        f"the author wrote this word themselves: {findings!r}" + caveat)


def test_the_registry_is_matched_after_the_expansion_erased_a_token(
        check, caveat):
    """161 blocks lose ``worker`` to the expansion; none of them is a hit.

    Checking the docstring would report every one of them, and every one
    would be a block in which the model never sees the word.
    """
    # Real corpus prose: spacr.lineage.read_object_tables[0]. "worker
    # thread" becomes "background execution unit", so the Icelandic model is
    # never handed the word the registry knows it declines.
    source = "Read the object tables a lineage needs. Safe on a worker thread."
    expanded = check.expanded_source(source)
    assert not ({"worker", "workers"} & check._lexical_words(expanded)), (
        "the control is stale: this checkout's expansion no longer removes "
        f"'worker' here, so re-measure before trusting the 161-block figure "
        f"in this file: {expanded!r}" + caveat)
    signals = [signal for signal, _why, _fix
               in check.registry_findings(source)]
    assert "KNOWN_BAD_TOKENS/is/worker" not in signals, (
        "the registry fired on a word the expansion removed before the "
        f"model saw it: {expanded!r}" + caveat)


# ---------------------------------------------------------------------------
# the label signal, evaluated on the expansion
# ---------------------------------------------------------------------------
def test_a_shouted_heading_stays_exempt_after_the_expansion_lowers_it(
        check, caveat):
    """THE MEASURED FALSE POSITIVES OF EXPANDING NAIVELY, pinned here.

    ``_lower_shouted_emphasis`` turns these into the defect shape. Read
    against the cached model output for sv, es, de, fr, pt, zh_CN, hi, ko and
    is, every one of them translates and carries ``A4``/``C1`` through
    verbatim. Flagging them costs two false positives and buys nothing.
    """
    for source in (
        "A4 DOES NOT NEED MASKS.",
        "A4 DOES NOT NEED MASKS. It needs points that mean the same thing "
        "in two frames, and a smoothed local maximum on a nuclear stain is "
        "that at a fraction of a segmentation's cost.",
        "C1  SEQUENCING CHANNELS, PER CYCLE ... NEVER averaged across "
        "cycles -- that is the one operation that destroys a barcode while "
        "leaving it decodable.",
    ):
        expanded = check.expanded_source(source)
        assert not check.check_block_as_translated(source), (
            "a shouted heading was flagged once the expansion lowered it. "
            "The exemption belongs to the AUTHOR'S text, not the model's: "
            f"{source[:60]!r} -> {expanded[:60]!r}" + caveat)


def test_the_naive_expansion_really_would_have_flagged_those_headings(
        check, caveat):
    """The test above is only worth running if the trap is real.

    If this ever stops flagging, ``_lower_shouted_emphasis`` has changed and
    the exemption above should be re-measured rather than left standing.
    """
    source = "A4 DOES NOT NEED MASKS."
    naive = check.check_block(check.expanded_source(source))
    assert naive, (
        "the expansion no longer lowers shouted emphasis, so the heading "
        "exemption in check_block_as_translated is now guarding nothing -- "
        "re-measure it instead of trusting it" + caveat)


def test_the_four_real_failures_are_still_named_through_the_expansion(
        check, caveat):
    """CALIBRATION AGAINST THE EXPANSION, not only against the source.

    The four blocks really failed the gate on 2026-09-11. The gate's input
    was their expansion, so the expansion is where recall has to hold.
    """
    missed = [name for name, block in check.CALIBRATION
              if not check.check_block_as_translated(block)]
    assert not missed, (
        "the expansion-aware check no longer names blocks that really "
        "failed the gate: " + ", ".join(missed) + caveat)


def test_the_corpus_is_clean_in_the_text_the_model_is_handed(check, caveat):
    """The whole corpus, judged on the expansion rather than the docstring."""
    flagged = []
    examined = rewritten = 0
    for name, block in check.corpus_blocks():
        examined += 1
        if check.expanded_source(block) != block:
            rewritten += 1
        for _signal, why, fix in check.check_block_as_translated(block):
            flagged.append(f"{name}: {why}\n      fix: {fix}")

    assert examined > 1000, (
        f"only {examined} blocks were examined, so this test is not looking "
        "at the corpus it claims to" + caveat)
    assert rewritten > 100, (
        f"only {rewritten} of {examined} blocks are rewritten before the "
        "model sees them, which contradicts the measurement this check is "
        "built on -- re-measure the expansion before trusting anything here"
        + caveat)
    assert not flagged, (
        f"{len(flagged)} block(s) put a bare project label where a "
        "translator expects a name, in the English the model receives:\n\n  "
        + "\n\n  ".join(flagged[:12]) + caveat)


def test_check_block_still_judges_exactly_the_text_it_is_handed(
        check, caveat):
    """The lower-level contract, kept so callers can point it anywhere.

    ``check_block`` must NOT expand. A caller that wants the model's input
    asks for ``check_block_as_translated``; a caller checking a runtime
    catalog row, which never goes through ``_api_translation_source``, wants
    the raw form.
    """
    source = "The crops are written before the screen reads them."
    assert check.expanded_source(source) != source, (
        "control broken: pick a block the expansion actually rewrites"
        + caveat)
    assert check.check_block(source) == [], (
        "no label here, expanded or not" + caveat)
    assert check.check_block("B2: run a segmenter over each window."), (
        "check_block must still decide the text it is handed" + caveat)


# ---------------------------------------------------------------------------
# cause 2 -- the registry reports, and says what believing it costs
# ---------------------------------------------------------------------------
def test_the_registry_carries_the_four_known_failures(check, caveat):
    """The observations that cost somebody a rebuild, written down once."""
    names = {(entry.name, entry.language) for entry in check.KNOWN_BAD_TOKENS}
    for expected in (("channel+stain", "zh_CN"), ("patch", "is"),
                     ("worker", "is"), ("builder", "de")):
        assert expected in names, (
            f"the registry lost {expected}, which is an observation nobody "
            "can re-derive from the English -- it only exists because "
            "somebody ran the model" + caveat)


def test_every_registry_entry_states_its_evidence_and_its_price(
        check, caveat):
    """A REGISTRY WITHOUT A PRICE IS A MOOD.

    394 prices one candidate signal out on exactly this ground: a density cut
    that caught its single true positive flagged 2,917 other blocks. Every
    entry here has to say how many blocks it flags and how many of those were
    confirmed against real cached output, so the next reader can decide
    whether to believe it.
    """
    for entry in check.KNOWN_BAD_TOKENS:
        assert entry.evidence.strip(), f"{entry.name} has no evidence{caveat}"
        assert entry.price.strip(), f"{entry.name} has no price{caveat}"
        assert any(character.isdigit() for character in entry.price), (
            f"{entry.name}'s price has no numbers in it, so it is an "
            "opinion" + caveat)
        assert entry.token_groups and all(
            group and all(form == form.lower() for form in group)
            for group in entry.token_groups), (
            f"{entry.name} has empty or non-lowercase token forms; matching "
            "is done on lower-cased words" + caveat)


def test_a_registry_advisory_never_gates_the_run(check, caveat):
    """Advisory means advisory, and the exit code is where that is decided.

    Measured precision across the four entries runs from 0/16 to 22/24. A
    signal in that range is worth printing and is not worth failing a build
    on, and the distinction has to live somewhere a reader can check.
    """
    source = "Retire spaCR's own idle workers and lower its thread counts."
    assert check.registry_findings(source), (
        "control broken: this block is supposed to hit the registry"
        + caveat)
    assert not check.check_block_as_translated(source), (
        "a registry hit must not become a gating finding" + caveat)


def test_the_registry_only_reads_prose_the_model_will_see(check, caveat):
    """Protected literals are supposed to survive untranslated.

    The other session's first residual-English pass counted words inside
    ``...`` and :role:`...` spans and reported 7 and 10 residual words on
    paragraphs that were clean.
    """
    assert not check.registry_findings(
        "The ``worker`` argument and :func:`patch` are documented above."), (
        "the registry fired on words inside protected spans, which are "
        "never translated and are never the defect" + caveat)


def test_the_tools_own_caveat_names_what_it_cannot_see(check, caveat):
    """The sentence a reader needs on the day this passes and the build fails.

    It is in the assertion text and in the tool's printed output rather than
    only in a docstring, because it is the first thing deleted by whoever
    finds the check annoying.
    """
    caveat_text = check.NOT_A_GREEN_BUILD
    assert "NOT A GREEN BUILD" in caveat_text.upper(), caveat
    for phrase in ("identity", "declines", "per-language",
                   "repair-api-blocks"):
        assert phrase in caveat_text, (
            f"the caveat stopped naming {phrase!r}, which is one of the "
            "causes no source-only check can reach" + caveat)


def test_verify_registry_prices_itself_against_real_cached_output(
        check, caveat):
    """Skipped without checkpoints, which is most machines.

    Where a cache exists this is the only part of the file that is evidence
    rather than bookkeeping: it reads what the model really returned.
    """
    corpus = [(name, block) for name, block in check.CALIBRATION]
    corpus.append(("probe",
                   "Retire spaCR's own idle workers and lower thread counts."))
    rows = check.verify_registry(corpus)
    assert len(rows) == len(check.KNOWN_BAD_TOKENS), caveat
    if not any(row["cache"] for row in rows):
        pytest.skip("no checkpoint cache on this machine")
    for row in rows:
        if not row["cache"]:
            continue
        assert row["cached"] <= row["flagged"], (
            f"{row['entry'].name}: more cached than flagged" + caveat)
        assert row["confirmed"] <= row["cached"], (
            f"{row['entry'].name}: more confirmed than cached" + caveat)
