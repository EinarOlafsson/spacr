"""Infer filename-parsing regexes for microscope image collections.

:func:`propose` groups filenames with a shared token structure and returns
ranked regular expressions with named metadata groups. Each proposal reports
its coverage, unmatched files, sampled values, and the evidence used to
suggest roles such as ``wellID``, ``fieldID``, and ``chanID``. Review these
suggestions before importing files from a new naming convention.

Use :func:`rename_preview` and :func:`structure` to inspect the proposed
renaming and folder layout. These functions do not move, rename, or write
files.

Examples
--------
>>> names = ["WA01F001C1.tif", "WA01F002C1.tif"]
>>> candidate = propose(names)[0]
>>> candidate.matched, candidate.unmatched
(2, ())
>>> rename_preview(candidate, names)[0]["matched"]
True
"""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: The group names spaCR's importer actually reads. A name outside this set
#: reaches `_rename_and_organize_image_files` and matches nothing, so the
#: inference engine only proposes names from this set.
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
    """Infer a metadata role from the literal preceding a value slot.

    Parameters
    ----------
    before
        Literal text immediately before the variable filename component. Only
        its trailing alphabetic run is considered.

    Returns
    -------
    str
        A role from :data:`KNOWN_ROLES`, or an empty string when the literal
        provides no recognized hint.
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
    """Describe the observed values and suggested role of one filename slot.

    :param index: token position in the filename family.
    :param values: observed values for this slot in input order.
    :param numeric: whether every observed value contains only digits.
    :param before: literal filename text immediately preceding this slot.
    :param role: suggested spaCR metadata field, or an empty string when the
        evidence does not support a role.
    :param fixed_tail: constant suffix absorbed into this slot's capture group.
    :param because: human-readable evidence supporting the suggested role.
    """

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
        """Return the number of distinct observed values."""
        return len(set(self.values))

    def samples(self, limit: int = 4) -> Tuple[str, ...]:
        """Return unique example values in their first-seen order.

        Parameters
        ----------
        limit
            Maximum number of values to return.

        Returns
        -------
        tuple of str
            Up to ``limit`` unique values.
        """
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
    """Store a candidate filename regex and the evidence for reviewing it.

    :param pattern: regular-expression pattern containing the proposed named
        capture groups.
    :param fields: evidence indexed by proposed capture-group name.
    :param matched: number of evaluated basenames matched by ``pattern``.
    :param total: number of non-empty input basenames evaluated.
    :param unmatched: basenames the pattern could not parse, retained for
        review.
    :param suffix: shared filename extension without its leading period.
    """

    pattern: str
    fields: Dict[str, FieldEvidence] = field(default_factory=dict)
    matched: int = 0
    total: int = 0
    unmatched: Tuple[str, ...] = ()
    suffix: str = ""

    @property
    def coverage(self) -> float:
        """Return the fraction of evaluated filenames that match."""
        return (self.matched / self.total) if self.total else 0.0

    def evidence(self) -> str:
        """Format coverage, field samples, role evidence, and unmatched files."""
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
        """Compile and return :attr:`pattern`."""
        return re.compile(self.pattern)


_TOKEN = re.compile(r"\d+|\D+")


def tokenise(name: str) -> List[str]:
    """Split a filename into alternating digit and non-digit runs.

    Parameters
    ----------
    name
        Filename or filename-like string. The extension remains part of the
        final non-digit run.

    Returns
    -------
    list of str
        Token runs in their original order.
    """
    return _TOKEN.findall(str(name))


#: How many non-digit positions may vary inside one family before it is
#: split back apart. One or two is a well letter and a channel code; five is
#: two different microscopes that happen to tokenise to the same length, and
#: merging those produces a regex that matches everything and means nothing.
MAX_VARYING_LITERALS = 2


def shape_of(tokens: Sequence[str]) -> Tuple[str, ...]:
    """Replace digit tokens with placeholders to identify a filename family.

    Parameters
    ----------
    tokens
        Tokens returned by :func:`tokenise`.

    Returns
    -------
    tuple of str
        Tokens with each all-digit run replaced by ``"#"``.
    """
    return tuple("#" if token.isdigit() else token for token in tokens)


def mask_of(tokens: Sequence[str]) -> Tuple[str, ...]:
    """Replace digit and non-digit tokens with coarse family placeholders.

    Parameters
    ----------
    tokens
        Tokens returned by :func:`tokenise`.

    Returns
    -------
    tuple of str
        ``"#"`` for each digit run and ``"@"`` for each non-digit run.

    Notes
    -----
    This coarser grouping lets changing well letters remain variable. Families
    that exceed :data:`MAX_VARYING_LITERALS` are rejected later to avoid
    merging unrelated naming conventions.
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
        """Assign one still-available role to a slot.

        :param info: field evidence to update when the claim succeeds.
        :param role: proposed regex-group role; false-like roles are rejected.
        :param because: explanation recorded with a successful assignment.
        :returns: True after assigning and reserving a previously unused role;
            otherwise False without changing the evidence or captured set.
        """
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
    """Infer and rank candidate regexes for a collection of filenames.

    Parameters
    ----------
    names
        Filenames or paths. Only each basename is inspected; empty names are
        ignored.
    limit
        Maximum number of proposals to return.

    Returns
    -------
    list of Proposal
        Candidates ordered by matched-file count and then by the number of
        inferred metadata fields. Returns an empty list when the input has no
        comparable filename family.

    Notes
    -----
    Mixed naming conventions can produce several proposals. Inspect
    :attr:`Proposal.unmatched` and :meth:`Proposal.evidence` before choosing
    one.
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
    """Preview parsed metadata and destination folders without writing files.

    Parameters
    ----------
    proposal
        Candidate returned by :func:`propose`.
    names
        Filenames or paths to preview.
    roles
        Optional mapping from capture-group names to spaCR metadata roles.
        Values override the roles suggested by ``proposal``.

    Returns
    -------
    list of dict
        One record per input with ``old``, ``matched``, ``values``, and
        ``folder`` keys. Unmatched files remain in the result with
        ``matched=False``.

    Notes
    -----
    This function performs no file-system writes or renames.
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
    """Count matched preview files by proposed destination folder.

    Parameters
    ----------
    preview
        Records returned by :func:`rename_preview`.

    Returns
    -------
    dict of str to int
        Sorted mapping from folder path to matched-file count. Unmatched files
        and records without a folder are omitted.
    """
    counts: Counter = Counter()
    for row in preview:
        if row.get("matched") and row.get("folder"):
            counts[row["folder"]] += 1
    return dict(sorted(counts.items()))
