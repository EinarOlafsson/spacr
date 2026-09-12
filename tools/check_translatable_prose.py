#!/usr/bin/env python
"""Predict, from the ENGLISH SOURCE ALONE, prose a catalog build will reject.

INSTRUCTION 394. Both i18n catalog lanes gate on translation QUALITY, and
that verdict exists only after a ~40 minute nine-language model build plus a
~25 minute audit. Adding one docstring paragraph therefore costs 65 minutes
before anyone learns whether the paragraph was translatable; four cycles were
burned that way in one evening. This runs in about two seconds.

    python tools/check_translatable_prose.py

IT COVERS ONE CAUSE OF THREE, AND THAT IS THE FIRST THING TO KNOW. The
failures have at least three independent causes and only the first is visible
in the English:

    1. a project label in a slot where a translator expects a name
       -- static, cheap, and what this file decides;
    2. model brittleness on particular token pairs -- "channel" and "stain"
       in one row return identity in zh_CN while either alone is fine, and
       "dye" for "stain" is fine. Only running the model finds it;
    3. the expansion layer injecting a word -- `:mod:`spacr.crops`` becomes
       "the crop builder", which then leaked into German. Nobody wrote it.

The bound is not a gap in this implementation, it is a property of the
problem: `--repair-api-blocks` once cleared about thirty-seven blocks per
language in `ko` and `is` WITH NO SOURCE EDIT AT ALL. Whatever failed there
was not in the English, so no amount of reading the English could have
predicted it.

  A GREEN RUN HERE IS NOT A GREEN BUILD. It means one class of defect is
  absent. Believing otherwise is worse than not running it, because it is
  the belief that gets acted on the day it is wrong.

THE ONE SIGNAL: BARE_LABEL_SUBJECT. A project-internal label -- an
instruction number (``372``) or a phase code (``B2``, ``C4``) -- standing as
the GRAMMATICAL SUBJECT of a prose sentence, with no noun to carry it:

    "372 states the contract:"          "370 asks for the scorecard ..."
    "C4 samples phenotype channels"     "B2: run a segmenter over ..."

Those are the four blocks that really failed the gate on 2026-09-11, and the
rewrite that made each pass removed exactly this construction and nothing
else -- ``B2:`` was deleted, ``C4 samples`` became "Phenotype channels are
sampled", ``370 asks for`` became "The scorecard in graph form". The label is
an opaque token where a translator expects a name, so it is translated,
renumbered or dropped, and the audit rejects the block.

    THE SIGNAL WAS RECOVERED BY DIFFING, NOT THEORISED. That matters because
    two theories that sounded better were measured and thrown away. ALL-CAPS
    emphasis: the caps block in the same failing docstring is BYTE-IDENTICAL
    to its passing rewrite, so caps cannot be it -- and real Swedish output
    shows the caps translated (badly) rather than passed through. Bare
    identifier runs: `_PROTECT_RE` already protects snake_case, so the
    pseudo-table everyone points at is invisible to the gate, and the
    rewrite kept the identical identifier list anyway.

AND THE LENGTH THEORY IS FALSE, MEASURED RATHER THAN ASSERTED. 394 suspects
sentence length and clause density. Over all 38,352 passing blocks and the
four positives, by percentile:

    metric            corpus p50/p90/p95   P1     P2     P3     P4
    words             10 / 43 / 57         70.7   55.9   87.2   93.5
    mean sentence      9 / 24 / 30         44.9   60.3   98.2   91.9
    longest sentence   9 / 31 / 38         66.8   57.9   95.0   88.8
    clause marks       1 /  4 /  6         66.7   42.9   79.3   79.3

P2 -- "B2: run a segmenter over each window and collect the observations." --
sits between the 43rd and 60th percentile on EVERY one of them. Any threshold
low enough to catch it catches more than half the corpus, about 20,000
blocks. P3 and P4 are genuinely long, but that is spaCR house style and not a
property of failure. The answer to 394's suspicion is no, and this file says
so rather than shipping a signal that sounds right and measures nothing.

THIS IS NOT A STYLE RULE. Dense, clause-heavy, ALL-CAPS prose is the house
style and is fine. Referring to an instruction is fine -- "372's PART 6-A",
"Phase B segments, once, on the composite" and "A4 DOES NOT NEED MASKS." all
ship today and all pass. Only the bare label in SUBJECT position is flagged,
and only in text that reaches a catalog: public docstrings and UI strings,
never comments.

MEASURED against this checkout:

    38,352 blocks of public-docstring prose examined
         0 flagged
       4/4 recovered real failures named, and no sibling block of those same
           four docstrings flagged

It calls ``translatable_blocks()`` from the real builder and never
reimplements it. A checker that splits its input differently from the tool it
is checking measures its own splitter: that mistake produced a report of
1,569 stale reviewed records where the true number was one, and, the same
evening, a claim that a docstring's first block carried no label when the
block that failed was the paragraph below it.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import re
import sys


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


def _builder():
    """The production splitter, imported -- never reimplemented."""
    tools = ROOT / "tools"
    for entry in (str(tools), str(ROOT)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location(
        "_i394_documentation_builder", tools / "build_documentation_i18n.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def check_block(block: str):
    """Findings for one translatable block: ``(signal, why, fix)`` tuples.

    Pure text in, tuples out -- point it at any catalog text, not only
    docstrings.
    """
    findings = []
    for sentence in _SENTENCE.split(_mask(block)):
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
        findings.append((
            f"BARE_LABEL_SUBJECT/{form}",
            f"bare {kind} {label!r} is the subject of "
            f"\"{sentence[:72]}\"",
            _FIXES[form].format(label=label),
        ))
    return findings


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


def main() -> int:
    builder = _builder()
    examined = 0
    flagged = []
    for symbol, text in builder.public_docstrings().items():
        blocks, _layout = builder.translatable_blocks(text)
        for index, block in enumerate(blocks):
            examined += 1
            findings = check_block(block)
            if findings:
                flagged.append((f"{symbol}[{index}]", findings))

    print(f"blocks examined : {examined}")
    print(f"blocks flagged  : {len(flagged)}")
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
              "in subject position.")

    named = sum(1 for _name, block in CALIBRATION if check_block(block))
    print(f"\ncalibration (the four real gate failures of 2026-09-11): "
          f"{named}/{len(CALIBRATION)} named")
    for name, block in CALIBRATION:
        findings = check_block(block)
        mark = findings[0][0] if findings else "NOT NAMED"
        print(f"  {mark:35s} {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())