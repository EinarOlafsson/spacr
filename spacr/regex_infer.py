"""
Work the regex out from the filenames, instead of asking for one.

Instruction 137 B. `spacr.utils._get_regex` is a four-branch lookup --
'cellvoyager', 'cq1', 'auto' and 'custom' -- of which three are FIXED PATTERNS
('auto' is not automatic; it is a fourth hard-coded regex) and the fourth means
the user types a Python regex with the right named groups by hand::

    (?P<plateID>.*)_(?P<wellID>.*)_T(?P<timeID>.*)F(?P<fieldID>.*)
    L(?P<laserID>..)A(?P<AID>..)Z(?P<sliceID>.*)C(?P<chanID>.*).tif

So a microscope spaCR has not met is a wall before anything can be imported.
This is what takes the wall down::

    from spacr.regex_infer import propose

    best, *rest = propose(filenames)
    best.pattern          # a regex with named groups
    best.matched          # how many of the names it matches
    best.unmatched        # WHICH ones it does not -- the useful list
    best.fields           # {group: FieldEvidence} -- distinct values, samples

HOW IT WORKS. Every name is split into runs of digits and runs of non-digits.
Names with the same SHAPE -- the same sequence of literal separators around the
same number of variable slots -- form a family. Within a family the slots that
never vary are folded back into the literal text and the ones that do become
groups. The largest family wins, and the rest are offered behind it.

WHAT IT NEVER DOES IS PICK SILENTLY. The whole failure being fixed is a regex
that looked right and grouped the files wrong, so every proposal carries the
evidence beside it: how many files it matches, how many distinct values each
group takes, and which files it leaves out. An unmatched file is the most
useful thing on that screen.

ROLES ARE GUESSED AND SAID TO BE GUESSES. A slot matching a well pattern is
offered as `wellID`, one taking two to eight distinct values as `chanID`, one
following a literal 'F' as `fieldID`, and so on -- but the guess is a
SUGGESTION carried on the field, and instruction 137 C has the user confirm it
from a closed dropdown. A group name spaCR's importer does not read is a silent
import of nothing, which is why nobody types one.

Standard library only.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: The group names spaCR's importer actually reads. A name outside this set
#: reaches `_rename_and_organize_image_files` and matches nothing, so the
#: proposals only ever suggest from here — see instruction 137 C.
KNOWN_ROLES = ("plateID", "wellID", "fieldID", "timeID", "sliceID", "chanID",
               "laserID", "AID")

#: What a well looks like in the two conventions spaCR meets: A01 / H12, and
#: the r##c## form the Yokogawa exports and spaCR's own `prc` use.
WELL_PATTERNS = (re.compile(r"^[A-Za-z]\d{1,2}$"),
                 re.compile(r"^r\d{1,2}c\d{1,2}$", re.IGNORECASE))

#: A slot taking this few distinct values across the whole drop is a channel
#: rather than a field or a slice. Four channels is the common case and eight
#: is the most spaCR's own `channels` list offers.
CHANNEL_MAX_DISTINCT = 8

#: What a literal immediately before a slot says about it, when it says
#: anything. The single letters are the ones the microscope vendors use and
#: the shipped patterns already encode; the words are what everybody ELSE
#: writes, and a drop from a microscope spaCR has not met is the case this
#: whole module exists for. Matched LONGEST FIRST, so `ch` beats `h` -- the
#: single-letter table alone read `-ch1` as an unnamed slot and let a plate
#: code two tokens earlier take `chanID` on nothing but its value count.
LITERAL_HINTS = {
    "T": "timeID", "F": "fieldID", "Z": "sliceID", "C": "chanID",
    "L": "laserID", "A": "AID", "W": "wellID", "P": "plateID", "S": "fieldID",
    "CH": "chanID", "CHAN": "chanID", "CHANNEL": "chanID",
    "WELL": "wellID", "PLATE": "plateID", "FIELD": "fieldID",
    "FLD": "fieldID", "SITE": "fieldID", "TIME": "timeID",
    "SLICE": "sliceID", "PLANE": "sliceID",
}

#: The hint keys, longest first, so a suffix match takes the most specific.
_HINT_ORDER = tuple(sorted(LITERAL_HINTS, key=len, reverse=True))


def hint_for(before: str) -> str:
    """The role the literal in front of a slot suggests, or "".

    Read off the END of the literal run, because that is what abuts the slot:
    in `exp7-P1-A01` the run before the plate number is `exp7-P`, and only its
    last character is about the slot.
    """
    # THE TRAILING ALPHA RUN ONLY, stopping at the first non-letter. Reading
    # every letter in the run made `plate1_` end in "PLATE" and claim the well
    # letter after it as a plate id -- the separator is exactly what says the
    # word is not about this slot.
    run = str(before or "")
    text = ""
    for ch in reversed(run):
        if not ch.isalpha():
            break
        text = ch + text
    text = text.upper()
    if not text:
        return ""
    for key in _HINT_ORDER:
        if text.endswith(key):
            return LITERAL_HINTS[key]
    return ""


@dataclass
class FieldEvidence:
    """What is known about one variable slot, so a user can judge the guess."""

    index: int
    values: Tuple[str, ...]
    numeric: bool
    before: str = ""
    role: str = ""
    #: A constant tail folded into this slot's group -- see
    #: :func:`_absorb_constant_well_digits`. "" for every other slot.
    fixed_tail: str = ""
    #: Why that role was suggested. Shown beside it: a guess presented without
    #: its reason is indistinguishable from a fact.
    because: str = ""

    @property
    def distinct(self) -> int:
        return len(set(self.values))

    def samples(self, limit: int = 4) -> Tuple[str, ...]:
        seen, out = set(), []
        for value in self.values:
            if value not in seen:
                seen.add(value)
                out.append(value)
            if len(out) >= limit:
                break
        return tuple(out)


@dataclass
class Proposal:
    """One candidate regex, with everything needed to judge it."""

    pattern: str
    fields: Dict[str, FieldEvidence] = field(default_factory=dict)
    matched: int = 0
    total: int = 0
    unmatched: Tuple[str, ...] = ()
    suffix: str = ""

    @property
    def coverage(self) -> float:
        return (self.matched / self.total) if self.total else 0.0

    def evidence(self) -> str:
        """The sentence that goes beside the pattern on screen."""
        lines = [f"{self.matched} of {self.total} files match "
                 f"({self.coverage:.0%})."]
        for name, info in self.fields.items():
            lines.append(
                f"  {name}: {info.distinct} distinct value"
                f"{'' if info.distinct == 1 else 's'}"
                f" — {', '.join(info.samples())}"
                + (f" ({info.because})" if info.because else ""))
        if self.unmatched:
            shown = ", ".join(self.unmatched[:3])
            more = (f" and {len(self.unmatched) - 3} more"
                    if len(self.unmatched) > 3 else "")
            lines.append(f"  NOT matched: {shown}{more}")
        return "\n".join(lines)

    def compiled(self):
        return re.compile(self.pattern)


_TOKEN = re.compile(r"\d+|\D+")


def tokenise(name: str) -> List[str]:
    """Split a filename into runs of digits and runs of everything else.

    The extension is left on: it is a literal like any other and dropping it
    here would mean putting it back in three places.
    """
    return _TOKEN.findall(str(name))


#: How many non-digit positions may vary inside one family before it is
#: split back apart. One or two is a well letter and a channel code; five is
#: two different microscopes that happen to tokenise to the same length, and
#: merging those produces a regex that matches everything and means nothing.
MAX_VARYING_LITERALS = 2


def shape_of(tokens: Sequence[str]) -> Tuple[str, ...]:
    """The name with its digit runs blanked — what makes two names a family.

    Two files from one microscope differ only in their numbers and in the odd
    letter code, so the shape is what groups them, and a name whose shape
    nobody else shares is exactly the "not matched" line the user needs to
    see.
    """
    return tuple("#" if token.isdigit() else token for token in tokens)


def mask_of(tokens: Sequence[str]) -> Tuple[str, ...]:
    """The shape with the LITERAL TEXT blanked too.

    A coarser family than :func:`shape_of`, and it is the one that gets
    Yokogawa right. `plate1_A01_...` and `plate1_B01_...` have different
    SHAPES -- the well letter is a non-digit token and it differs -- so they
    landed in two families, each matching a third of the drop, and the well
    letter was folded into the literal text where it could never become a
    group. Under the mask they are one family and the letter is a slot, which
    is what it is.

    Both are tried. A mask family that turns out to vary in more than
    :data:`MAX_VARYING_LITERALS` non-digit positions is two microscopes rather
    than one, and the shape families are the answer there.
    """
    return tuple("#" if token.isdigit() else "@" for token in tokens)


def _factor(values: Sequence[str]) -> Tuple[str, str, Tuple[str, ...]]:
    """``(common prefix, common suffix, what is left)`` for a set of values."""
    head = os.path.commonprefix(list(values))
    rest = [v[len(head):] for v in values]
    tail = os.path.commonprefix([r[::-1] for r in rest])[::-1]
    if tail:
        rest = [r[:len(r) - len(tail)] for r in rest]
    return head, tail, tuple(rest)


def _class_for(values: Sequence[str], numeric: bool) -> str:
    """The character class for a slot, built from what it actually holds.

    Derived rather than guessed, because a class that cannot match its own
    observed values is a regex that matches nothing and says nothing about
    why. Letters and digits are named as ranges; anything else is escaped
    into the class verbatim.
    """
    if numeric:
        return r"\d+"
    seen = set("".join(values))
    pieces = []
    if any(c.isalpha() for c in seen):
        pieces.append("A-Za-z")
    if any(c.isdigit() for c in seen):
        pieces.append("0-9")
    others = "".join(sorted(c for c in seen if not c.isalnum()))
    if others:
        pieces.append(re.escape(others))
    return "[" + "".join(pieces) + "]+" if pieces else r"[^_.]+"


def _proposal_for(names: Sequence[str], all_names: Sequence[str]) -> Optional[Proposal]:
    """Build one proposal from a family of same-shaped names.

    TWO PASSES, and the split is what makes the roles right. The first walks
    the tokens and decides only WHAT VARIES; the second names the slots. Doing
    both at once meant a slot was named from the evidence in front of it and
    the name was then taken -- so the slot after a literal 'C', which is
    unambiguously the channel, arrived to find `chanID` already used by an
    earlier slot that merely had few distinct values.
    """
    rows = [tokenise(n) for n in names]
    if not rows:
        return None
    width = len(rows[0])
    varies = [len({row[i] for row in rows}) > 1 for i in range(width)]
    if not any(varies):
        return None

    # -- pass one: what varies, and what literal sits in front of it --------
    pieces: List[dict] = []          # {"literal": str} or {"slot": FieldEvidence}
    literal_run = ""
    for i in range(width):
        column = tuple(row[i] for row in rows)
        if not varies[i]:
            literal_run += rows[0][i]
            continue
        numeric = all(v.isdigit() for v in column)
        tail = ""
        if not numeric:
            head, tail, column = _factor(column)
            literal_run += head
            if not any(column):
                literal_run += tail
                continue
        if literal_run:
            pieces.append({"literal": literal_run})
        pieces.append({"slot": FieldEvidence(index=i, values=column,
                                             numeric=numeric,
                                             before=literal_run)})
        literal_run = tail
    if literal_run:
        pieces.append({"literal": literal_run})
    _merge_wells(pieces)

    slots = [p["slot"] for p in pieces if "slot" in p]
    if not slots:
        return None
    _assign_roles(slots)

    parts, fields = [], {}
    for index, piece in enumerate(pieces):
        if "literal" in piece:
            parts.append(re.escape(piece["literal"]))
            continue
        info = piece["slot"]
        name = info.role or f"group{index}"
        fields[name] = info
        body = _class_for(
            tuple(v[:len(v) - len(info.fixed_tail)] if info.fixed_tail else v
                  for v in info.values), info.numeric)
        parts.append(f"(?P<{name}>{body}"
                     + (re.escape(info.fixed_tail) if info.fixed_tail else "")
                     + ")")

    pattern = "".join(parts)
    compiled = re.compile(pattern)
    matched = sum(1 for n in all_names if compiled.fullmatch(str(n)))
    unmatched = tuple(str(n) for n in all_names
                      if compiled.fullmatch(str(n)) is None)
    return Proposal(pattern=pattern, fields=fields, matched=matched,
                    total=len(all_names), unmatched=unmatched,
                    suffix=os.path.splitext(str(names[0]))[1].lstrip("."))


def _merge_wells(pieces: List[dict]) -> None:
    """Fold an adjacent letter slot and digit slot into ONE well slot.

    `A01` tokenises as a letter run and a digit run, so a plate's wells became
    two groups -- and neither of them was a well. Merged when they are
    adjacent with nothing between and every joined value looks like a well,
    which is the check that keeps `L01` and `C02` out of it.
    """
    i = 0
    while i < len(pieces) - 1:
        first, second = pieces[i], pieces[i + 1]
        if "slot" not in first or "slot" not in second:
            i += 1
            continue
        left, right = first["slot"], second["slot"]
        if left.numeric or not right.numeric:
            i += 1
            continue
        joined = tuple(a + b for a, b in zip(left.values, right.values))
        if all(any(p.match(v) for p in WELL_PATTERNS) for v in joined):
            pieces[i] = {"slot": FieldEvidence(
                index=left.index, values=joined, numeric=False,
                before=left.before, role="wellID",
                because="a letter and digits that read as a well")}
            del pieces[i + 1]
        i += 1
    _absorb_constant_well_digits(pieces)


_LEADING_DIGITS = re.compile(r"^\d{1,2}")


def _absorb_constant_well_digits(pieces: List[dict]) -> None:
    """Take a CONSTANT well number into the well slot in front of it.

    A plate whose files are only A01 and B01 has a well NUMBER that never
    varies, so it is folded into the literal text and the letter is left alone
    -- and a bare `A` is not a well by any test, so the slot fell through to
    whatever heuristic happened to be next. The digits are still part of the
    well; they are simply always the same one. Absorbed as a fixed tail of the
    group, so the group captures `A01` and the regex still says `01`.
    """
    for i, piece in enumerate(pieces[:-1]):
        slot = piece.get("slot")
        following = pieces[i + 1].get("literal")
        if slot is None or slot.role or slot.numeric or not following:
            continue
        digits = _LEADING_DIGITS.match(following)
        if digits is None:
            continue
        tail = digits.group(0)
        joined = tuple(v + tail for v in slot.values)
        if not all(any(p.match(v) for p in WELL_PATTERNS) for v in joined):
            continue
        slot.values = joined
        slot.role = "wellID"
        slot.because = "a letter and a well number that never changes"
        slot.fixed_tail = tail
        pieces[i + 1] = {"literal": following[len(tail):]}


def _assign_roles(slots: Sequence[FieldEvidence]) -> None:
    """Name every slot, STRONGEST EVIDENCE FIRST.

    Order matters because a role can only be used once -- two groups with one
    name is a regex Python refuses to compile. A slot that follows a literal
    'C' IS the channel; a slot that merely takes few values only looks like
    one, and it must not take the name before the other slot is considered.
    """
    taken = set()
    for info in slots:
        if info.role:
            taken.add(info.role)

    def claim(info, role, because):
        if role and role not in taken:
            info.role, info.because = role, because
            taken.add(role)
            return True
        return False

    # 1. a well is a well, whatever is in front of it.
    for info in slots:
        if info.role or not info.values:
            continue
        if all(any(p.match(v) for p in WELL_PATTERNS) for v in info.values):
            claim(info, "wellID", "every value looks like a well")

    # 2. the vendor letters, which are the strongest thing a name carries.
    for info in slots:
        if info.role:
            continue
        hint = hint_for(info.before)
        if hint:
            claim(info, hint, f"follows {info.before.strip('_-.')!r}")

    # 3. and only then the shape of the values themselves.
    for info in slots:
        if info.role:
            continue
        if info.numeric and 2 <= info.distinct <= CHANNEL_MAX_DISTINCT:
            if claim(info, "chanID", f"only {info.distinct} distinct values"):
                continue
        if info.distinct == 1:
            if claim(info, "plateID", "the same in every file"):
                continue
        if info.numeric:
            claim(info, "fieldID", f"{info.distinct} distinct numbers")


def propose(names: Iterable[str], limit: int = 4) -> List[Proposal]:
    """Rank candidate regexes for ``names``, best first. ``[]`` for nothing.

    :param names: filenames, with or without directories. Only the basename
        is read -- a user dropping a folder should not get a regex that
        matches their home directory's spelling.
    :param limit: how many proposals to return.

    THE LARGEST FAMILY WINS, and the others are offered behind it rather than
    discarded: a drop that mixes two microscopes has two right answers and
    picking one silently is the failure this exists to prevent.
    """
    basenames = [os.path.basename(str(n)) for n in names if str(n).strip()]
    if not basenames:
        return []
    families: Dict[Tuple[str, ...], List[str]] = {}
    for name in basenames:
        families.setdefault(shape_of(tokenise(name)), []).append(name)
    coarse: Dict[Tuple[str, ...], List[str]] = {}
    for name in basenames:
        coarse.setdefault(mask_of(tokenise(name)), []).append(name)

    candidates = list(coarse.values()) + list(families.values())
    proposals, seen = [], set()
    for family in sorted(candidates, key=len, reverse=True):
        proposal = _proposal_for(family, basenames)
        if proposal is None or proposal.pattern in seen:
            continue
        # TWO MICROSCOPES ARE NOT ONE FAMILY. A mask family varying in many
        # non-digit positions has merged unrelated names, and its regex would
        # match everything while meaning nothing.
        if sum(1 for info in proposal.fields.values() if not info.numeric
               ) > MAX_VARYING_LITERALS:
            continue
        seen.add(proposal.pattern)
        proposals.append(proposal)
    # By coverage, then by how many groups it found: between two patterns
    # that match every file, the one that pulled out more metadata is the
    # more useful answer and the one a user can always simplify.
    proposals.sort(key=lambda p: (p.matched, len(p.fields)), reverse=True)
    return proposals[:limit]


def rename_preview(proposal: Proposal, names: Iterable[str],
                   roles: Optional[Dict[str, str]] = None) -> List[dict]:
    """What each file would become. Instruction 137 D: it SHOWS, never writes.

    :param roles: ``{group name: role}`` from the user's dropdowns, overriding
        the suggested roles. 137 C: nobody types a group name.
    :returns: one dict per file -- ``old``, ``matched``, ``values`` and the
        ``folder`` the spaCR structure would put it in. A file that does not
        match is in the list with ``matched=False``, because 412 files
        appearing without comment is how half a plate goes missing.
    """
    compiled = proposal.compiled()
    mapping = dict(roles or {})
    out = []
    for raw in names:
        name = os.path.basename(str(raw))
        match = compiled.fullmatch(name)
        if match is None:
            out.append({"old": name, "matched": False, "values": {},
                        "folder": ""})
            continue
        values = {mapping.get(k, k): v
                  for k, v in (match.groupdict() or {}).items()}
        out.append({"old": name, "matched": True, "values": values,
                    "folder": _folder_for(values)})
    return out


def _folder_for(values: Dict[str, str]) -> str:
    """Where the spaCR structure puts a file with these metadata values."""
    parts = [values.get(role, "") for role in
             ("plateID", "wellID", "fieldID", "chanID")]
    return "/".join(part for part in parts if part)


def structure(preview: Sequence[dict]) -> Dict[str, int]:
    """``{folder: how many files}`` for a preview. The tree, with counts."""
    counts: Counter = Counter()
    for row in preview:
        if row.get("matched") and row.get("folder"):
            counts[row["folder"]] += 1
    return dict(sorted(counts.items()))
