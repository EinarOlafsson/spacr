#!/usr/bin/env python
"""Regenerate ``features/00_INDEX.txt`` from the feature files.

TWO LISTS, NEITHER OF THEM A GATE. ``features/new/`` is what spaCR has
gained; ``features/future/`` is what it might gain next. Asked for on
2026-09-12, replacing a single instructions list whose items carried
release-blocking status and so turned a plan into a gate.

WHY THIS IS A TOOL AND NOT A DOCUMENT. The index was hand-written on
2026-08-05 and was nine days stale by 2026-08-14: it listed work that had
shipped and omitted work that had been filed. An index that disagrees with the
folder is worse than no index, because it is believed -- and this repository
has already lost time to eight instruction files that were wrong about their
own state.

Everything here is read off the filesystem at run time. The only hand-written
part is the small table of stages and blockers below, which is the one thing
the files themselves do not say in a machine-readable way.

TWO FOLDERS, NOT THREE, AND THE HEADER NOW SAYS WHICH. The scan is
``features/future/*.txt`` and ``features/new/*.txt`` -- the folders named in
``SCANNED`` -- and nothing else. ``features/`` itself still holds ledger files
the rename from ``instructions/`` left behind, and an index that told a reader
it was generated "from the files themselves" while skipping them is what item
398 was filed about. They are globbed per run by ``_unscanned_top_level`` and
printed in the index header BY NAME, never as a count: a number written into
prose does not move when the folder does, and this ledger has been caught by
exactly that several times. Listing them also puts them under ``--check``, so
a file arriving at the top level makes the committed index stale instead of
being invisible to every check that exists.

Usage::

    python tools/build_instruction_index.py            # rewrite
    python tools/build_instruction_index.py --check     # exit 1 if stale

``--check`` is what a test calls, so the index cannot go stale again without
CI saying so.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
INSTRUCTIONS = REPO / "features"
INDEX = INSTRUCTIONS / "00_INDEX.txt"

#: The folders this index is generated from, named once. Every scan below
#: reads this tuple and the header prints it, so the folders the tool claims
#: to read cannot drift from the folders it does read.
SCANNED: Tuple[str, ...] = ("future", "new")

#: Instructions owned by the concurrent codex session. Named here rather than
#: inferred, because "do not touch this" is not something a file says about
#: itself and getting it wrong means two sessions editing one file.
#: Empty because both codex-owned instructions -- 48 (tutorials) and 83
#: (catalogs) -- are DONE. It fills again the moment two sessions share
#: this folder. Named here rather than inferred, because "do not touch
#: this" is not something a file says about itself, and getting it
#: wrong means two sessions editing one file.
OWNERS: Dict[str, str] = {}

#: Item numbers that are already used twice, measured on 2026-09-13 and
#: named rather than counted. A number is the only handle anyone has on an
#: item -- HANDOFF, the trailing notes and the commit messages all cite items
#: by number -- so two files sharing one makes every citation ambiguous.
#:
#: THIS IS A RATCHET, NOT A PARDON. `--check` fails on any duplicate NOT in
#: this set, so the eight below can be resolved one at a time without a new
#: one slipping in behind them. Removing a member is the only edit that
#: should ever be made here.
#:
#: Three are the same topic recorded twice at different stages (58, 82, 83)
#: and want merging; five are genuinely different topics that collided (84,
#: 176, 177, 234, 377) and want renumbering.
KNOWN_DUPLICATE_NUMBERS: frozenset[str] = frozenset({
    "58", "82", "83", "84", "176", "177", "234", "377",
})


#: Why an item cannot be worked, when the reason is outside the repository.
BLOCKED: Dict[str, str] = {
    "44": "needs a macOS/Windows host",
    "45": "needs a macOS/Windows host",
    "53": "needs makensis + both other OSes",
    "59": "needs the maintainer's accounts",
    "81": "needs the reporter's `df -T` and a stack trace",
}

#: The two the maintainer has scheduled at the end, in this order.
LAST: Dict[str, str] = {
    "82": "SECOND TO LAST",
}

#: How far along something is, where the file's own header does not say.
STAGE: Dict[str, str] = {
    # Audited against the code on 2026-09-01, not read off the headers.
    # Three items were found already complete and moved to new/ that day
    # (319, 330, 336); two more were badly wrong about themselves (327 read
    # as not-started with all five parts shipped, 306 read as finished with
    # its ratchet red). Re-audit before trusting any figure here.
    "01": "100% of the code; blocked on publishing 1.5.0.5",
    "05": "~40% -- mechanism verified at 1.5.0.4; needs one green SHA, then approval",
    "253": "0% by construction -- closes last",
    "288": "coverage 99.87%, 355 items in 108 modules (measured 2026-08-31, now stale); CI red; zero open issues",
    "304": "~60% -- metadata in place; needs the Zenodo toggle and the bump",
    "305": "~60% -- startup accepted from an installed wheel; sdist, GPU, matrix, profiles left",
    "315": "~75% -- 3a/3b/3c fixed; 3d now itemised into three named optimisations",
    "316": "READMEs delivered in all nine; lane triaged 2026-09-06, 26 red -> 19: 1,089 catalog rows blocked on OPUS models absent from this machine, 5 are 372's OPS tooltips, the rest are pins and two stale strings",
    "325": "the channel between the two sessions -- open while both are running",
    "326": "~55% -- settings follow the count (2 means 2); the ceiling of 26 is what remains",
    "327": "~95% -- all five parts shipped; only the frame-rate evidence is missing",
    "331": "a checklist over the other items; regenerated 2026-09-02 after three closures and four new filings",
    "337": "~75% -- Manders, spatial defaults and both labels done; part 3 needs the maintainer's measure settings",
    "339": "0% -- illumination is called from measure and nowhere else",
    "341": "0% -- three tests confirmed still failing on 2026-09-01",
    "372": "unblocked 2026-09-04 -- the maintainer answered the stitch question; PART 0's survivability audit of 3,891 unreached lines still comes first",
    "350": "~50% -- no proven clipping in 4 screens x 3 locales; three false-positive classes recorded",
    "353": "~60% -- the buttons are at the top; aligning them to their columns is not done",
    "345": "~35% -- the stale stub is fixed, 3 down to 2; the rest are order-dependent",
    "346": "~90% -- 21 down to 3; the last two are two live copies of one function, diagnosed",
    "348": "~35% -- Help is a dock heading and is last; the magnifier and the text move are open",
}


def _instruction_title(lines: List[str], number: str, fallback: str) -> str:
    """Return a title from either instruction format used in the ledger.

    Older records put an uppercase title between ``====`` rules. Recent
    records begin directly with ``NNN — Title``. The index used to assume
    only the first form, which silently produced blank rows as soon as the
    second form reached ``future/`` or ``new/``.
    """
    if (len(lines) >= 3 and lines[0].strip()
            and set(lines[0].strip()) == {"="} and lines[1].strip()):
        return lines[1].strip()

    prefix = f"{number} —"
    for line in lines:
        candidate = line.strip()
        if not candidate:
            continue
        if candidate.startswith(prefix):
            title = candidate[len(prefix):].strip()
            if title:
                return title
        if set(candidate) <= {"=", "-"}:
            continue
        return candidate
    return fallback


def _entries(folder: str) -> List[Tuple[str, str, str]]:
    """``(number, title, filename)`` for one folder, in numeric order."""
    out = []
    for path in (INSTRUCTIONS / folder).glob("*.txt"):
        number = path.name.split("_", 1)[0]
        if not number.isdigit():
            continue
        lines = path.read_text(errors="replace").splitlines()
        title = _instruction_title(lines, number, path.stem)
        out.append((number, title, path.name))
    # Instruction 84 exists twice, so the numeric id is not a unique sort
    # key.  ``Path.glob`` preserves the filesystem's directory-entry order,
    # which is not stable between a developer checkout and a GitHub runner.
    # Include the filename as the tie-breaker so a checkout cannot make the
    # committed index appear stale without any content changing.
    return sorted(out, key=lambda row: (int(row[0]), row[2]))


def _note_for(number: str) -> str:
    if number in OWNERS:
        return f"[{OWNERS[number]} -- DO NOT TOUCH]"
    if number in LAST:
        return f"[{LAST[number]}]"
    if number in BLOCKED:
        return f"[BLOCKED: {BLOCKED[number]}]"
    if number in STAGE:
        return f"[{STAGE[number]}]"
    return ""


def _sort_key(number: str) -> tuple[int, str]:
    """Order item numbers numerically, tolerating a non-numeric name."""
    return (int(number), "") if number.isdigit() else (1 << 30, number)


def _files_for_number(number: str) -> List[str]:
    """Every feature file whose name starts with this item number."""
    found: List[str] = []
    for folder in SCANNED:
        base = INSTRUCTIONS / folder
        if not base.is_dir():
            continue
        found += sorted(path.name for path in base.glob(f"{number}_*"))
    return found


def _unscanned_top_level() -> List[str]:
    """``features/*.txt`` files that no list in this index is built from.

    GLOBBED PER RUN, NEVER PINNED AT A NUMBER. Item 398 is the record of what
    these are: `instructions/` held files in three places -- `open/`, `done/`
    and its own top level -- and the rename carried all three across, so the
    top level kept ledger files that neither ``SCANNED`` folder contains. The
    count of them has been written into prose three times and has been wrong
    since the day it was written each time; the index prints the names this
    call returns instead, so it is right on the run and shrinks when somebody
    resolves one.

    The index itself is excluded: a file cannot be an input to its own
    generation.
    """
    return sorted((path.name for path in INSTRUCTIONS.glob("*.txt")
                   if path.name != INDEX.name), key=str.lower)


def _duplicate_numbers() -> set:
    """Item numbers carried by more than one file, across both lists."""
    seen: Dict[str, int] = {}
    for folder in SCANNED:
        base = INSTRUCTIONS / folder
        if not base.is_dir():
            continue
        for path in base.iterdir():
            number = path.name.split("_", 1)[0]
            if not number.isdigit():
                continue
            seen[number] = seen.get(number, 0) + 1
    return {number for number, count in seen.items() if count > 1}


def render(today: str = "") -> str:
    """The whole index as text."""
    # Keyed off SCANNED, which the header also prints, so the folders named
    # and the folders read are one list. DROPPING a folder from SCANNED is a
    # KeyError on the next two lines rather than a header that quietly
    # describes the wrong scan. ADDING one is not: it would be globbed, and
    # printed in the header, while its rows went into no list -- so a third
    # folder needs a list of its own below, which is why the two lookups are
    # spelled out here instead of iterated.
    rows = {folder: _entries(folder) for folder in SCANNED}
    future_rows = rows["future"]
    new_rows = rows["new"]
    # Kept under the old names below so the rest of this renderer, which
    # predates the split into two lists, does not have to be rewritten to
    # say the same thing.
    open_rows, done_rows = future_rows, new_rows
    total = len(open_rows) + len(done_rows)
    percent = (len(done_rows) * 100 // total) if total else 0
    stamp = today or datetime.date.today().isoformat()

    unscanned = _unscanned_top_level()

    lines = [
        "=" * 80,
        "spaCR FEATURES -- NEW, AND FUTURE",
        "=" * 80,
        "",
        f"Regenerated {stamp} by `tools/build_instruction_index.py`, from "
        "the files in",
        "these globs and nothing else:",
        "",
    ]
    lines += [f"  features/{folder}/*.txt" for folder in SCANNED]
    lines += [
        "",
        "Do not hand-edit: it went nine days stale last time, and an index "
        "that",
        "disagrees with the folder is worse than none, because it is "
        "believed.",
        "",
    ]
    if unscanned:
        lines += [
            "features/ ITSELF IS NOT SCANNED, and the .txt files below sit "
            "there, in",
            "neither list. They are globbed on every run rather than counted "
            "once in",
            "prose, so this list is right on the day it is read and shrinks "
            "when one is",
            "resolved; item 398 holds what becomes of the ledger ones. Where "
            "the number",
            "a name here opens with also has a row below, 398 measured this "
            "top-level",
            "copy as the staler of the two -- and it is the one a reader "
            "reaches first.",
            "",
        ]
        lines += [f"  {name}" for name in unscanned]
    else:
        lines += [
            "features/ ITSELF IS NOT SCANNED, and today it holds no .txt "
            "file but this",
            "one, so the globs above are the whole ledger.",
        ]
    lines += [
        "",
        "Each file says the same four things: what the state is, why it "
        "matters, what",
        "to do, and how to know it worked. Where something is NOT worth "
        "doing, that is",
        "said too, with the reason -- a decision not to act is also a result, "
        "and",
        "re-deriving it later costs the same as deriving it once.",
        "",
        "READ features/HANDOFF.md FIRST. It carries the traps, what needs "
        "the",
        "maintainer, and the standing rules.",
        "",
        "THE TRAILING NOTES AT THE END OF EACH FILE ARE THE CURRENT STATE. "
        "The header",
        "often says 'not started' when it is done -- eight files have been "
        "wrong about",
        "themselves this week.",
        "",
        f"{len(done_rows)} in features/new, {len(open_rows)} in "
        f"features/future. NEITHER LIST BLOCKS A RELEASE.",
        "",
        "-" * 80,
        "FUTURE FEATURES",
        "-" * 80,
        "",
    ]
    for number, title, name in open_rows:
        lines.append(f"  {number:>3}  {title}".rstrip())
        note = _note_for(number)
        if note:
            lines.append(f"       {note}")
        lines.append(f"       {name}")
        lines.append("")

    lines += ["-" * 80, "NEW FEATURES", "-" * 80, ""]
    for number, title, _name in done_rows:
        lines.append(f"  {number:>3}  {title}".rstrip())
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if the index is stale")
    args = parser.parse_args(argv)

    if not INSTRUCTIONS.is_dir():
        print("no features/ folder here")
        return 0

    current = INDEX.read_text() if INDEX.exists() else ""
    # The stamp is the one line that changes without the content changing, so
    # --check compares everything BELOW it. An index that fails CI because a
    # day passed would be an index nobody keeps.
    def body(text: str) -> str:
        return "\n".join(line for line in text.splitlines()
                         if not line.startswith("Regenerated "))

    duplicates = _duplicate_numbers()
    unexpected = sorted(duplicates - KNOWN_DUPLICATE_NUMBERS, key=_sort_key)
    resolved = sorted(KNOWN_DUPLICATE_NUMBERS - duplicates, key=_sort_key)
    for number in unexpected:
        print(f"DUPLICATE item number {number}: "
              + ", ".join(_files_for_number(number)))
    if resolved:
        print("resolved duplicates, drop from KNOWN_DUPLICATE_NUMBERS: "
              + ", ".join(resolved))

    fresh = render()
    if args.check:
        if unexpected:
            print(f"STALE -- {len(unexpected)} item number(s) used twice and "
                  "not in KNOWN_DUPLICATE_NUMBERS")
            return 1
        if body(current) == body(fresh):
            print("the index matches the instruction files")
            return 0
        print("STALE — run tools/build_instruction_index.py")
        return 1
    INDEX.write_text(fresh)
    print(f"index regenerated: {INDEX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
