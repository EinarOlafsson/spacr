#!/usr/bin/env python
"""Predict, from the ENGLISH THE MODEL SEES, prose a catalog build rejects.

INSTRUCTION 394. Both i18n catalog lanes gate on translation QUALITY, and
that verdict exists only after a ~40 minute nine-language model build plus a
~25 minute audit. Adding one docstring paragraph therefore costs 65 minutes
before anyone learns whether the paragraph was translatable; four cycles were
burned that way in one evening. This runs in about fifteen seconds.

    python tools/check_translatable_prose.py
    python tools/check_translatable_prose.py --verify-registry

IT DECIDES ONE CAUSE OF THREE, AND THAT IS THE FIRST THING TO KNOW. The
failures have at least three independent causes and only the first is decided
here:

    1. a project label in a slot where a translator expects a name
       -- static, cheap, and what this file DECIDES;
    2. model brittleness on particular tokens -- "channel" and "stain" in one
       row return identity in zh_CN while either alone is fine, and "dye" for
       "stain" is fine. Only running the model finds it. A REGISTRY of pairs
       learned from past failures is checkable and is REPORTED here, never
       decided -- see KNOWN_BAD_TOKENS for the measured price of each entry;
    3. the expansion layer injecting a word -- ``_api_translation_source``
       rewrites the block before the model sees it, and the model then keeps
       what it was handed. Nobody wrote it.

The bound is not a gap in this implementation, it is a property of the
problem: ``--repair-api-blocks`` once cleared about thirty-seven blocks per
language in ``ko`` and ``is`` WITH NO SOURCE EDIT AT ALL. Whatever failed
there was not in the English, so no amount of reading the English could have
predicted it.

  A GREEN RUN HERE IS NOT A GREEN BUILD. It means one class of defect is
  absent. Believing otherwise is worse than not running it, because it is
  the belief that gets acted on the day it is wrong.

--------------------------------------------------------------------------
CAUSE 3 IS NOT A SEPARATE SIGNAL, IT IS WHICH STRING EVERY SIGNAL READS
--------------------------------------------------------------------------

The model never sees the docstring. ``_translate_api_documents`` builds its
input as ``translation_inputs[block] = _api_translation_source(block)``, so
the string that is translated, hashed and gated is the EXPANSION, not the
source. Measured on this checkout:

    38,368 blocks, 7,824 of them (20.4%) rewritten before the model sees them

and the rewrite is not cosmetic. The words it INJECTS most often, none of
which any author wrote:

    1,382 microplate   1,185 view   1,136 data   1,047 sample
    1,043 image          760 position   546 extracted   526 laboratory

and the words it ERASES: 832 ``screen``, 384 ``crop``, 161 ``worker``.

SO A CHECK THAT READS THE DOCSTRING IS CHECKING THE WRONG TEXT, and the cost
is measurable in both directions. For ``worker`` -- a word the Icelandic
checkpoint declines -- the expansion ERASES it from 161 blocks and INJECTS it
into 8 that never contained it:

    Retire spaCR's own idle workers and lower its thread counts.
      -> ... and lower its worker-count limits.
    production image loads are threaded.
      -> production image loads are executed in a worker.

That is the same event as the German ``builder`` leak this instruction
records, alive in this checkout with a different word. ``check_block`` still
takes whatever text it is handed, but ``check_block_as_translated`` is what
the corpus scan and the registry use, and it expands first.

    AND EXPANDING CHANGED THE ANSWER, WHICH IS WHY IT NEEDED MEASURING
    RATHER THAN ASSUMING. Run naively on the expanded text, the label signal
    flags two blocks it does not flag on the source -- because
    ``_lower_shouted_emphasis`` runs INSIDE the expansion and turns
    "A4 DOES NOT NEED MASKS." into "A4 does not need masks.", which is the
    defect shape exactly. Both were then read against the cached model
    output for all nine languages: both translate, and ``A4`` and ``C1``
    survive verbatim in every one. They are false positives, so the heading
    exemption is evaluated on the AUTHOR'S text while the signal reads the
    model's. A shouted heading is a heading whatever case the expansion
    leaves it in.

--------------------------------------------------------------------------
THE ONE SIGNAL IT DECIDES: BARE_LABEL_SUBJECT
--------------------------------------------------------------------------

A project-internal label -- an instruction number (``372``) or a phase code
(``B2``, ``C4``) -- standing as the GRAMMATICAL SUBJECT of a prose sentence,
with no noun to carry it:

    "372 states the contract:"          "370 asks for the scorecard ..."
    "C4 samples phenotype channels"     "B2: run a segmenter over ..."

Those are the four blocks that really failed the gate on 2026-09-11, and the
rewrite that made each pass removed exactly this construction and nothing
else. The label is an opaque token where a translator expects a name, so it
is translated, renumbered or dropped, and the audit rejects the block.

    THE SIGNAL WAS RECOVERED BY DIFFING, NOT THEORISED. Two theories that
    sounded better were measured and thrown away. ALL-CAPS emphasis: the
    caps block in the same failing docstring is BYTE-IDENTICAL to its
    passing rewrite. Bare identifier runs: ``_PROTECT_RE`` already protects
    snake_case, so the pseudo-table everyone points at is invisible to the
    gate, and the rewrite kept the identical identifier list anyway.

AND THE LENGTH THEORY IS FALSE, MEASURED RATHER THAN ASSERTED. 394 suspects
sentence length and clause density. Over all passing blocks and the four
positives, by percentile:

    metric            corpus p50/p90/p95   P1     P2     P3     P4
    words             10 / 43 / 57         70.7   55.9   87.2   93.5
    mean sentence      9 / 24 / 30         44.9   60.3   98.2   91.9
    longest sentence   9 / 31 / 38         66.8   57.9   95.0   88.8
    clause marks       1 /  4 /  6         66.7   42.9   79.3   79.3

P2 -- "B2: run a segmenter over each window and collect the observations." --
sits between the 43rd and 60th percentile on EVERY one of them. Any threshold
low enough to catch it catches more than half the corpus.

THIS IS NOT A STYLE RULE. Dense, clause-heavy, ALL-CAPS prose is the house
style and is fine. Referring to an instruction is fine -- "372's PART 6-A",
"Phase B segments, once, on the composite" and "A4 DOES NOT NEED MASKS." all
ship today and all pass. Only the bare label in SUBJECT position is flagged,
and only in text that reaches a catalog: public docstrings and UI strings,
never comments.

MEASURED against this checkout:

    38,368 blocks of public-docstring prose examined, on the expanded text
         0 flagged
       4/4 recovered real failures named, and no sibling block of those same
           four docstrings flagged

It calls ``translatable_blocks()`` and ``_api_translation_source()`` from the
real builder and never reimplements either. A checker that splits or rewrites
its input differently from the tool it is checking measures its own splitter:
that mistake produced a report of 1,569 stale reviewed records where the true
number was one, and, the same evening, a claim that a docstring's first block
carried no label when the block that failed was the paragraph below it.

--------------------------------------------------------------------------
WHAT IT STILL CANNOT CATCH, STATED WHERE IT CANNOT BE MISSED
--------------------------------------------------------------------------

* IDENTITY RETURNS ON UNSEEN TOKEN COMBINATIONS. The registry knows the
  combinations that have already burned somebody. It cannot know the next
  one, and "spaCR reads the table" translating while "The screen makes spaCR
  a route to that embedding" does not is the proof that no readable property
  of the English predicts them.
* VOCABULARY A GIVEN CHECKPOINT DECLINES. Per-language, invisible in the
  English, and only produced by running the model. The registry is a cache of
  four such observations, not a derivation.
* WHETHER THE EXPANSION'S OWN OUTPUT TRANSLATES. This file can prove the
  model sees a different string; it cannot predict what the model does with
  it. ``--verify-registry`` reads the real cached output where a checkpoint
  cache exists, which is the only honest way any of this gets priced.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import re
import sys
from functools import lru_cache
from typing import NamedTuple


# --------------------------------------------------------------------------
# locating the checkout and the real block splitter
# --------------------------------------------------------------------------
def _find_root() -> pathlib.Path:
    starts = []
    if os.environ.get("SPACR_ROOT"):
        starts.append(pathlib.Path(os.environ["SPACR_ROOT"]))
    starts.append(pathlib.Path(__file__).resolve().parent)
    starts.append(pathlib.Path.cwd())
    for start in starts:
        start = start.resolve()
        for candidate in [start, *start.parents]:
            if (candidate / "tools" / "build_documentation_i18n.py").is_file():
                return candidate
    raise SystemExit("cannot locate the spaCR checkout")


ROOT = _find_root()


@lru_cache(maxsize=1)
def _builder():
    """The production splitter and expander -- imported, never rewritten."""
    tools = ROOT / "tools"
    for entry in (str(tools), str(ROOT)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location(
        "_i394_documentation_builder", tools / "build_documentation_i18n.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=None)
def expanded_source(block: str) -> str:
    """The exact string the translation model is handed for ``block``.

    ``_translate_api_documents`` builds its model input as
    ``translation_inputs[block] = _api_translation_source(block)`` and hashes
    that same string into ``translation_source_blocks_sha256``. Anything that
    judges what the model will do must judge THIS text: on this checkout the
    expansion rewrites 20.4% of blocks, injecting words no author wrote
    (1,382 ``microplate``, 1,185 ``view``) and erasing others (832
    ``screen``, 161 ``worker``).
    """
    return _builder()._api_translation_source(str(block))


# --------------------------------------------------------------------------
# the check
# --------------------------------------------------------------------------

#: ``literal``, :role:`target` and `single` spans are already protected from
#: the model, so a label inside one is not the defect.  Masked with spaces so
#: every offset in the block is preserved.
_PROTECTED = re.compile(r'``[^`]*``|:[a-zA-Z:+._-]+:`[^`]*`|`[^`]*`')

#: A colon ends a sentence here because the label-colon form ("B2: run ...")
#: is one of the two shapes the defect takes.
_SENTENCE = re.compile(r'(?<=[.!?:])\s+')

_PHASE = re.compile(r"^([A-E][0-9]{1,2})(?![\w.])")
_NUMBER = re.compile(r"^([0-9]{3})(?![\w.,%/-])")
_FOLLOWING_WORD = re.compile(r"[\s:]*([A-Za-z'-]+)")

#: A three-digit number at the head of a sentence is USUALLY a quantity.  It
#: reads as a REFERENCE only when the sentence attributes an act of
#: specification to it.  Measured on this checkout: without this list the
#: instruction-number half flags 6 blocks of the passing corpus -- "333 tiles
#: of 1480 px", "389 genes", "141 settings have one" -- every one a quantity
#: followed by the noun it counts.  With it, none.  Extending the list is a
#: one-line change and costs recall, not precision.
ATTRIBUTION_VERBS = frozenset("""
    say says said state states stated ask asks asked require requires
    required specify specifies specified name names named call calls called
    want wants wanted demand demands demanded note notes noted record records
    recorded describe describes described define defines defined list lists
    listed warn warns warned allow allows allowed forbid forbids forbade add
    adds added give gives gave put puts make makes made mean means meant is
    was are were has had covers covered set sets uses used does did
    introduces introduced promises promised expects expected treats treated
    orders ordered gates gated
""".split())


def _known_instruction_numbers() -> frozenset:
    """Instruction numbers that actually exist, so 333 px is not a reference.

    Empty when ``instructions/`` is absent, in which case the attribution
    verb alone decides -- the check degrades in recall, never in precision.
    """
    found = set()
    base = ROOT / "instructions"
    for directory in (base, base / "open", base / "done"):
        if not directory.is_dir():
            continue
        for entry in directory.iterdir():
            match = re.match(r"(\d{1,3})[_.]", entry.name)
            if match:
                found.add(int(match.group(1)))
    return frozenset(found)


INSTRUCTION_NUMBERS = _known_instruction_numbers()


def _mask(text: str) -> str:
    return _PROTECTED.sub(lambda m: " " * len(m.group(0)), text)


def _lexical_words(text: str) -> frozenset:
    """Lower-cased prose words, with protected literals removed first.

    The other session's first residual-English pass counted words inside
    ``...`` and :role:`...` spans and reported 7 and 10 residual words on
    paragraphs that were clean. Those spans are SUPPOSED to survive
    untranslated, so nothing that reasons about model-visible vocabulary may
    count them.
    """
    return frozenset(word.lower() for word in re.findall(r"[A-Za-z]+",
                                                         _mask(text)))


def _label_at_head(sentence: str):
    """``(label, kind, tail)`` when the sentence opens with a bare label."""
    match = _PHASE.match(sentence)
    if match:
        return match.group(1), "phase code", sentence[match.end():]
    match = _NUMBER.match(sentence)
    if match:
        number = int(match.group(1))
        if INSTRUCTION_NUMBERS and number not in INSTRUCTION_NUMBERS:
            return None, None, None
        following = _FOLLOWING_WORD.match(sentence[match.end():])
        if following and following.group(1).lower() in ATTRIBUTION_VERBS:
            return match.group(1), "instruction number", sentence[match.end():]
    return None, None, None


_FIXES = {
    "label-colon": (
        "Delete the label. '{label}: do X.' -> 'Do X.'  The label belongs in "
        "the commit message and the instruction file, neither of which "
        "reaches a catalog."),
    "subject-verb": (
        "Give the sentence a real subject. '{label} <verb> X' -> name the "
        "thing {label} is about: 'The storage contract is X', 'Phenotype "
        "channels are sampled at ...'.  Keep the reference in the commit "
        "message instead."),
}


class _Label(NamedTuple):
    """One bare-label finding, before it is rendered for a human."""

    label: str
    kind: str
    form: str
    sentence: str


def _label_findings(text: str):
    """Every bare-label-subject construction in ``text``, structured."""
    for sentence in _SENTENCE.split(_mask(text)):
        sentence = sentence.strip()
        # A sentence that does not close is a label or a table row, not
        # prose: "170 px", "200 console lines a second, nothing else".
        if not sentence.endswith((".", "!", "?", ":")):
            continue
        label, kind, tail = _label_at_head(sentence)
        if label is None:
            continue
        if re.match(r"\s*:", tail):
            form = "label-colon"
        elif re.match(r"\s+[a-z]", tail):
            # A lowercase continuation means the label is the SUBJECT of a
            # sentence.  An uppercase or dashed continuation is a heading --
            # "A4 DOES NOT NEED MASKS.", "B9 - Control Charts:" -- which
            # ships today and passes.
            form = "subject-verb"
        else:
            continue
        yield _Label(label, kind, form, sentence)


def _render(finding: _Label, note: str = "") -> tuple:
    return (
        f"BARE_LABEL_SUBJECT/{finding.form}",
        f"bare {finding.kind} {finding.label!r} is the subject of "
        f"\"{finding.sentence[:72]}\"{note}",
        _FIXES[finding.form].format(label=finding.label),
    )


def check_block(block: str):
    """Findings for the text HANDED IN: ``(signal, why, fix)`` tuples.

    Pure text in, tuples out -- point it at any catalog text, not only
    docstrings.  It does NOT expand: a caller that wants the string the model
    actually receives wants :func:`check_block_as_translated`.
    """
    return [_render(finding) for finding in _label_findings(block)]


def _shouted_in_source(source: str, label: str) -> bool:
    """Was ``label`` followed by SHOUTED prose in the author's own text?

    THE EXEMPTION HAS TO BE EVALUATED ON THE SOURCE EVEN THOUGH THE SIGNAL
    READS THE EXPANSION, and that is not a convenience. A shouted run is a
    heading, and headings pass; but ``_lower_shouted_emphasis`` runs inside
    ``_api_translation_source``, so by the time the model sees it,
    "A4 DOES NOT NEED MASKS." is "A4 does not need masks." -- the defect
    shape exactly. Both corpus blocks of that shape were read against the
    cached model output for all nine languages before this exemption was
    written: both translate, and ``A4`` and ``C1`` survive verbatim in every
    target. Flagging them would have been two false positives bought with no
    true one.
    """
    masked = _mask(source)
    for match in re.finditer(rf"(?<![\w.]){re.escape(label)}(?![\w.])",
                             masked):
        following = _FOLLOWING_WORD.match(masked[match.end():])
        if not following:
            continue
        word = following.group(1)
        if word.isupper() and sum(char.isalpha() for char in word) >= 2:
            return True
    return False


def check_block_as_translated(block: str):
    """Findings for the string the MODEL receives: ``(signal, why, fix)``.

    This is the entry point the corpus scan uses.  It expands ``block``
    exactly as ``_translate_api_documents`` does and judges that, because the
    gate's verdict is about the expansion and not about the docstring.
    """
    source = str(block)
    expanded = expanded_source(source)
    findings = []
    for finding in _label_findings(expanded):
        if _shouted_in_source(source, finding.label):
            continue
        note = ""
        if expanded != source:
            note = ("  [seen after the expansion layer rewrote the block; "
                    "the docstring reads differently]")
        findings.append(_render(finding, note))
    return findings


# --------------------------------------------------------------------------
# cause 2: a registry of token combinations already known to break a model
# --------------------------------------------------------------------------
class KnownBadTokens(NamedTuple):
    """One observation that a checkpoint mishandles a token combination.

    ``token_groups`` is an AND of ORs: every group must contribute one form
    to the block, so ``(("channel", "channels"), ("stain", "stains"))`` means
    "some spelling of channel AND some spelling of stain".  Forms are written
    out rather than stemmed; a stemmer here would be this file guessing about
    a model it has not run.
    """

    name: str
    language: str
    token_groups: tuple
    failure: str
    evidence: str
    price: str


#: EVERY ENTRY CARRIES THE PRICE OF BELIEVING IT, measured by
#: ``--verify-registry`` against the checkpoint cache on this machine:
#: how many blocks the entry flags, how many of those have real cached model
#: output, and in how many of those the failure it predicts actually happened.
#:
#: THIS IS WHY THE REGISTRY IS REPORTED AND NOT DECIDED. 394 already prices
#: one signal out -- a density cut that caught its one true positive flagged
#: 2,917 others, "not a check, a mood" -- and two of these four entries are
#: in that territory on this corpus. An entry is worth keeping even at 0%
#: here: the observation was made on RUNTIME CATALOG ROWS, which are short
#: and are not in this corpus, and a registry that deletes what it cannot
#: currently reproduce forgets exactly the thing it exists to remember.
KNOWN_BAD_TOKENS = (
    KnownBadTokens(
        name="channel+stain",
        language="zh_CN",
        token_groups=(("channel", "channels"), ("stain", "stains")),
        failure="identity output -- the row comes back as its own English",
        evidence=(
            "2026-09-11, by calling build_i18n_catalogs._translate_batches "
            "directly rather than by reasoning: 'Per channel encodes each "
            "stain separately.' returns identity; 'Encodes each stain "
            "separately.' and 'Per channel encodes each dye separately.' "
            "both translate. Either word alone is fine and a synonym for "
            "either is fine. Specific to the four M2M targets; Swedish, a "
            "Marian model, translates all of them."),
        price=(
            "20 API blocks flagged, 16 of them in the zh_CN checkpoint "
            "cache, 0 identity. Zero confirmed positives HERE -- the "
            "observation is from short runtime rows, not API prose."),
    ),
    KnownBadTokens(
        name="patch",
        language="is",
        token_groups=(("patch", "patches"),),
        failure="the word is declined and left in English inside real prose",
        evidence=(
            "2026-09-12, read off the cached Icelandic for "
            "spacr.qt.stall_watch: 'The ``sitecustomize`` moddur er inn af "
            "vinnumennum, og hlutinn sem hann patch er aldrei "
            "framkvaemdastjorinn.' The model DID translate and declined two "
            "words. Two rewrites were spent shortening the block before "
            "anybody read the output; removing 'patch' and 'worker' fixed "
            "it at the same length."),
        price=(
            "10 API blocks flagged, 5 cached, 4 of those kept the word. The "
            "strongest entry on this corpus, and n=5."),
    ),
    KnownBadTokens(
        name="worker",
        language="is",
        token_groups=(("worker", "workers"),),
        failure="the word is declined and left in English inside real prose",
        evidence=(
            "2026-09-12, the same stall_watch block and the same fix. It is "
            "also the entry that proves why this runs on the EXPANSION: "
            "_api_translation_source erases 'worker' from 161 blocks and "
            "injects it into 8 that never contained it ('lower its thread "
            "counts' -> 'lower its worker-count limits')."),
        price=(
            "316 API blocks flagged, 233 cached, 9 kept the word -- 3.9%, "
            "and two of the nine are inside proper names. Read it as a "
            "prompt to look, never as a verdict."),
    ),
    KnownBadTokens(
        name="builder",
        language="de",
        token_groups=(("builder", "builders"),),
        failure="the word is declined and left in English inside real prose",
        evidence=(
            "EmbeddingsScreen.set_crops#1 leaked 'builder' into German from "
            "the TOOL'S OWN expansion of :mod:`spacr.crops` to 'the crop "
            "builder', not from anything the author wrote. That particular "
            "rewrite is gone from this checkout -- :mod:`spacr.crops` now "
            "expands to 'extracted image regions' and 0 blocks gain "
            "'builder' from the expansion -- so this entry is kept for the "
            "word, which authors still write, and the mechanism it records "
            "is now carried by 'worker' above."),
        price=(
            "48 API blocks flagged, 24 cached, 22 kept the word. High, and "
            "read it carefully: German takes 'Builder', 'Worker' and "
            "'Benchmark' as ordinary loanwords, so a kept word here is not "
            "automatically a failure. The build's own gate is the "
            "authority; this only says where to look."),
    ),
)


def registry_findings(block: str):
    """ADVISORY findings for ``block``: ``(signal, why, fix)`` tuples.

    Matched against the EXPANDED text, because that is the vocabulary the
    model is handed.  Never gates: see ``KNOWN_BAD_TOKENS`` for the measured
    precision of each entry on this corpus.
    """
    source = str(block)
    expanded = expanded_source(source)
    source_words = _lexical_words(source)
    expanded_words = _lexical_words(expanded)
    findings = []
    for entry in KNOWN_BAD_TOKENS:
        matched = []
        injected = []
        for group in entry.token_groups:
            hit = next((form for form in group if form in expanded_words),
                       None)
            if hit is None:
                break
            matched.append(hit)
            # INJECTED MEANS NO SPELLING OF THE WORD WAS IN THE SOURCE.
            # "workers" in the docstring becoming "worker" in the expansion
            # is the expansion editing the author's word, not inventing one,
            # and telling somebody nobody wrote it would be false.
            if not any(form in source_words for form in group):
                injected.append(hit)
        else:
            note = ""
            if injected:
                note = (
                    f"  NOBODY WROTE {', '.join(repr(w) for w in injected)}: "
                    "the expansion layer put it there, so rewording the "
                    "docstring may not remove it.")
            reach = "reach" if len(matched) > 1 else "reaches"
            together = " together" if len(matched) > 1 else ""
            findings.append((
                f"KNOWN_BAD_TOKENS/{entry.language}/{entry.name}",
                f"{' + '.join(repr(word) for word in matched)} {reach} the "
                f"{entry.language} model{together}; recorded failure: "
                f"{entry.failure}.{note}",
                f"ADVISORY, NOT A VERDICT -- {entry.price} Evidence: "
                f"{entry.evidence}",
            ))
    return findings


# --------------------------------------------------------------------------
# pricing the registry against real cached model output
# --------------------------------------------------------------------------
def _checkpoint_cache(language: str):
    """The build's own translation cache for ``language``, or ``None``.

    Read-only, and absent on any machine without checkpoints -- which is most
    of them, and is why nothing here depends on it.
    """
    builder = _builder()
    try:
        root = pathlib.Path(builder.default_model_root())
    except Exception:
        return None
    path = root / ".spacr_translation_cache" / f"{language}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def verify_registry(corpus):
    """Price every registry entry against the cached output of the real model.

    THE ONLY HONEST WAY TO KEEP A REGISTRY IS TO MAKE IT PROVE ITSELF. 394
    records three theories that survived because nobody read the output --
    ALL-CAPS, the pseudo-table, and the length threshold -- and each cost a
    full rebuild. This reads the output.

    :param corpus: ``(name, block)`` pairs of source blocks.
    :returns: one ``dict`` per entry, with ``flagged``, ``cached`` and
        ``confirmed`` counts, or an empty list when no cache exists.
    """
    builder = _builder()
    namespace = builder.API_BLOCK_CACHE_NAMESPACE
    rows = []
    for entry in KNOWN_BAD_TOKENS:
        cache = _checkpoint_cache(entry.language)
        if cache is None:
            rows.append({"entry": entry, "cache": False,
                         "flagged": 0, "cached": 0, "confirmed": 0,
                         "examples": []})
            continue
        flagged = cached = confirmed = 0
        examples = []
        for name, block in corpus:
            expanded = expanded_source(block)
            words = _lexical_words(expanded)
            if not all(any(form in words for form in group)
                       for group in entry.token_groups):
                continue
            flagged += 1
            output = cache.get(f"{namespace}\0{expanded}")
            if output is None:
                continue
            cached += 1
            if entry.failure.startswith("identity"):
                bad = output.strip() == expanded.strip()
            else:
                bad = bool(_lexical_words(output) & {
                    form for group in entry.token_groups for form in group})
            if bad:
                confirmed += 1
                if len(examples) < 3:
                    examples.append((name, output[:110]))
        rows.append({"entry": entry, "cache": True, "flagged": flagged,
                     "cached": cached, "confirmed": confirmed,
                     "examples": examples})
    return rows


# --------------------------------------------------------------------------
# the four real failures of 2026-09-11, carried inline so every run proves
# recall.  Verbatim block text, as translatable_blocks() produced it from
# 798272cde:spacr/ops_store.py, 798272cde:spacr/ops_objects.py,
# 224748854:spacr/ops_sample.py and db50c1420:spacr/scorecard.py.
# --------------------------------------------------------------------------
CALIBRATION = [
    ("spacr.ops_store[1] @798272cde",
     "PHASE D OF INSTRUCTION 372, and the B5 gate that depends on it. 372 "
     "states the contract:"),
    ("spacr.ops_objects.segment_windows[0] @798272cde",
     "B2: run a segmenter over each window and collect the observations."),
    ("spacr.ops_sample.average_over_cycles[1] @224748854",
     'C4 samples phenotype channels "at the same object ids", and a '
     "phenotype measured in several cycles is several samples of one "
     "quantity -- so averaging raises its precision, exactly as averaging "
     "the Hoechst across tiles did in B1."),
    ("spacr.scorecard.scorecard_figure[1] @db50c1420",
     '370 asks for the scorecard "in graph form" on Hugging Face and on the '
     "API page, beside the CSV and the table. This is that rendering, and it "
     "reads the SAME parsed metrics the tooltip and the zoo screen do, so "
     "the picture cannot disagree with the numbers printed next to it."),
]

#: Printed by every run, and it is the only sentence here that has to
#: survive: a green run means ONE class of defect is absent.
NOT_A_GREEN_BUILD = (
    "A GREEN RUN HERE IS NOT A GREEN BUILD. It decides one cause -- a bare "
    "project label where a translator expects a name -- and reports a second "
    "from a registry of four past failures. It cannot see an identity return "
    "on a token combination nobody has hit yet, and it cannot see vocabulary "
    "a checkpoint declines that is not already in the registry. Both sets are "
    "per-language and only running the model produces them: "
    "--repair-api-blocks once cleared ~37 blocks per language in ko and is "
    "WITH NO SOURCE EDIT AT ALL."
)


def corpus_blocks():
    """``(name, block)`` for every block of public-docstring prose."""
    builder = _builder()
    for symbol, text in builder.public_docstrings().items():
        blocks, _layout = builder.translatable_blocks(text)
        for index, block in enumerate(blocks):
            yield f"{symbol}[{index}]", block


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verify-registry", action="store_true",
        help="price every KNOWN_BAD_TOKENS entry against the real cached "
             "model output, where a checkpoint cache exists on this machine")
    parser.add_argument(
        "--show-advisory", type=int, default=8, metavar="N",
        help="how many registry-advisory blocks to print (default 8)")
    args = parser.parse_args(argv)

    corpus = list(corpus_blocks())
    examined = expanded = 0
    flagged = []
    advisory = []
    for name, block in corpus:
        examined += 1
        if expanded_source(block) != block:
            expanded += 1
        findings = check_block_as_translated(block)
        if findings:
            flagged.append((name, findings))
        advice = registry_findings(block)
        if advice:
            advisory.append((name, advice))

    percent = 100.0 * expanded / examined if examined else 0.0
    print(f"blocks examined       : {examined}")
    print(f"rewritten by expansion: {expanded} ({percent:.1f}%) -- the model "
          "never sees the docstring, so this check reads the expansion")
    print(f"blocks flagged        : {len(flagged)}")
    print()
    if flagged:
        print(f"first {min(20, len(flagged))} flagged:")
        for where, findings in flagged[:20]:
            print(f"\n  {where}")
            for signal, why, fix in findings:
                print(f"    signal : {signal}")
                print(f"    why    : {why}")
                print(f"    fix    : {fix}")
    else:
        print("no block of the current corpus carries a bare project label "
              "in subject position, in the text the model is handed.")

    print(f"\nregistry advisories   : {len(advisory)} block(s) carry a token "
          "combination a checkpoint has failed on before")
    counts = {}
    for _where, findings in advisory:
        for signal, _why, _fix in findings:
            counts[signal] = counts.get(signal, 0) + 1
    for signal in sorted(counts):
        print(f"  {counts[signal]:5d}  {signal}")
    print("  ADVISORY ONLY. Measured precision on this corpus runs from 0% "
          "to 92% -- see KNOWN_BAD_TOKENS, or run --verify-registry.")
    for where, findings in advisory[:max(0, args.show_advisory)]:
        print(f"\n  {where}")
        for signal, why, _fix in findings:
            print(f"    {signal}")
            print(f"      {why}")

    named = sum(1 for _name, block in CALIBRATION if check_block(block))
    print(f"\ncalibration (the four real gate failures of 2026-09-11): "
          f"{named}/{len(CALIBRATION)} named")
    for name, block in CALIBRATION:
        findings = check_block(block)
        mark = findings[0][0] if findings else "NOT NAMED"
        print(f"  {mark:35s} {name}")

    if args.verify_registry:
        print("\nregistry priced against the real cached model output:")
        rows = verify_registry(corpus)
        header = (f"  {'entry':16s}{'lang':7s}{'flagged':>9s}{'cached':>8s}"
                  f"{'confirmed':>11s}")
        print(header)
        for row in rows:
            entry = row["entry"]
            if not row["cache"]:
                print(f"  {entry.name:16s}{entry.language:7s}"
                      f"{'no checkpoint cache on this machine':>37s}")
                continue
            print(f"  {entry.name:16s}{entry.language:7s}"
                  f"{row['flagged']:9d}{row['cached']:8d}"
                  f"{row['confirmed']:11d}")
            for name, output in row["examples"]:
                print(f"      {name}: {output}")
        print("  'confirmed' is the recorded failure actually happening in "
              "cached output. A 0 is information, not a reason to delete the "
              "entry: three of these were measured on runtime catalog rows, "
              "which are not in this corpus.")

    print(f"\n{NOT_A_GREEN_BUILD}")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
