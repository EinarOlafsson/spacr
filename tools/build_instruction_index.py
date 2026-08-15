#!/usr/bin/env python
"""Regenerate ``instructions/00_INDEX.txt`` from the instruction files.

WHY THIS IS A TOOL AND NOT A DOCUMENT. The index was hand-written on
2026-08-05 and was nine days stale by 2026-08-14: it listed work that had
shipped and omitted work that had been filed. An index that disagrees with the
folder is worse than no index, because it is believed -- and this repository
has already lost time to eight instruction files that were wrong about their
own state.

Everything here is read off the filesystem at run time. The only hand-written
part is the small table of stages and blockers below, which is the one thing
the files themselves do not say in a machine-readable way.

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
INSTRUCTIONS = REPO / "instructions"
INDEX = INSTRUCTIONS / "00_INDEX.txt"

#: Instructions owned by the concurrent codex session. Named here rather than
#: inferred, because "do not touch this" is not something a file says about
#: itself and getting it wrong means two sessions editing one file.
#: Empty because both codex-owned instructions -- 48 (tutorials) and 83
#: (catalogs) -- are DONE. It fills again the moment two sessions share
#: this folder. Named here rather than inferred, because "do not touch
#: this" is not something a file says about itself, and getting it
#: wrong means two sessions editing one file.
OWNERS: Dict[str, str] = {}

#: Why an item cannot be worked, when the reason is outside the repository.
BLOCKED: Dict[str, str] = {
    "44": "needs a macOS/Windows host",
    "45": "needs a macOS/Windows host",
    "53": "needs makensis + both other OSes",
    "59": "needs the maintainer's accounts",
    "81": "needs the reporter's `df -T` and a stack trace",
}

#: The two the maintainer has scheduled at the end, in this order.
LAST: Dict[str, str] = {"82": "SECOND TO LAST", "58": "LAST"}

#: How far along something is, where the file's own header does not say.
STAGE: Dict[str, str] = {
    "52": "controls rebuilt 2026-08-13; geometry was already right",
    "75": "SUPERSEDED by 95 -- can be closed",
    "93": "filed, not started",
    "94": "~40% -- ladder built, five sites still ungrouped",
    "95": "model + GPU button built; viewer, grid, walk NOT",
}


def _entries(folder: str) -> List[Tuple[str, str, str]]:
    """``(number, title, filename)`` for one folder, in numeric order.

    The title is line 2 of the file, which is the convention every
    instruction follows -- line 1 and line 3 are the ``====`` rules.
    """
    out = []
    for path in (INSTRUCTIONS / folder).glob("*.txt"):
        number = path.name.split("_", 1)[0]
        if not number.isdigit():
            continue
        lines = path.read_text(errors="replace").splitlines()
        title = lines[1].strip() if len(lines) > 1 else path.stem
        out.append((number, title, path.name))
    return sorted(out, key=lambda row: int(row[0]))


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


def render(today: str = "") -> str:
    """The whole index as text."""
    open_rows = _entries("open")
    done_rows = _entries("done")
    total = len(open_rows) + len(done_rows)
    percent = (len(done_rows) * 100 // total) if total else 0
    stamp = today or datetime.date.today().isoformat()

    lines = [
        "=" * 80,
        "WHAT IS LEFT -- INDEX",
        "=" * 80,
        "",
        f"Regenerated {stamp} by `tools/build_instruction_index.py`, from the "
        "files",
        "themselves. Do not hand-edit: it went nine days stale last time, and "
        "an index",
        "that disagrees with the folder is worse than none, because it is "
        "believed.",
        "",
        "Each file says the same four things: what the state is, why it "
        "matters, what",
        "to do, and how to know it worked. Where something is NOT worth "
        "doing, that is",
        "said too, with the reason -- a decision not to act is also a result, "
        "and",
        "re-deriving it later costs the same as deriving it once.",
        "",
        "READ instructions/HANDOFF.md FIRST. It carries the traps, what needs "
        "the",
        "maintainer, and the standing rules.",
        "",
        "THE TRAILING NOTES AT THE END OF EACH FILE ARE THE CURRENT STATE. "
        "The header",
        "often says 'not started' when it is done -- eight files have been "
        "wrong about",
        "themselves this week.",
        "",
        f"{len(done_rows)} done / {len(open_rows)} open ({percent}%).",
        "",
        "-" * 80,
        "OPEN",
        "-" * 80,
        "",
    ]
    for number, title, name in open_rows:
        lines.append(f"  {number:>3}  {title}")
        note = _note_for(number)
        if note:
            lines.append(f"       {note}")
        lines.append(f"       {name}")
        lines.append("")

    lines += ["-" * 80, "DONE", "-" * 80, ""]
    for number, title, _name in done_rows:
        lines.append(f"  {number:>3}  {title}")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if the index is stale")
    args = parser.parse_args(argv)

    if not INSTRUCTIONS.is_dir():
        print("no instructions/ folder here")
        return 0

    current = INDEX.read_text() if INDEX.exists() else ""
    # The stamp is the one line that changes without the content changing, so
    # --check compares everything BELOW it. An index that fails CI because a
    # day passed would be an index nobody keeps.
    def body(text: str) -> str:
        return "\n".join(line for line in text.splitlines()
                         if not line.startswith("Regenerated "))

    fresh = render()
    if args.check:
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
