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

    pieces: List[dict] = []
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

    for info in slots:
        if info.role or not info.values:
            continue
        if all(any(p.match(v) for p in WELL_PATTERNS) for v in info.values):
            claim(info, "wellID", "every value looks like a well")

    for info in slots:
        if info.role:
            continue
        hint = hint_for(info.before)
        if hint:
            claim(info, hint, f"follows {info.before.strip('_-.')!r}")

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
        if sum(1 for info in proposal.fields.values() if not info.numeric
               ) > MAX_VARYING_LITERALS:
            continue
        seen.add(proposal.pattern)
        proposals.append(proposal)
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


#: The convention table. One record per naming scheme spaCR can parse without
#: the user writing a regular expression, keyed by the value stored in the
#: ``metadata_type`` setting.
#:
#: Every record carries:
#:
#: ``key``
#:     the stored ``metadata_type`` value. NEVER changed once shipped: it is
#:     written into settings CSVs and read back by name.
#: ``label``
#:     the line the dropdown shows.
#: ``vendor`` / ``instrument``
#:     what the dropdown groups by, and what a user recognises.
#: ``pattern``
#:     the regular expression, with ``{ext}`` standing in for the image
#:     extension :func:`spacr.utils._get_regex` interpolates.
#: ``ext``
#:     ``'verbatim'`` substitutes ``img_format`` exactly as given and is what
#:     the four original arms did -- including the bare ``.`` before the
#:     extension, which matches ANY character and is kept because those four
#:     patterns are pinned byte for byte. ``'suffix'`` is for everything added
#:     since: the leading dot is stripped from ``img_format``, the dot is
#:     escaped, the extension is matched case-insensitively (a plate from an
#:     ImageXpress arrives as ``.TIF``) and the pattern is anchored.
#: ``examples``
#:     REAL filenames, at least two, taken from the source named below. A
#:     pattern whose examples were invented is a guess, and
#:     ``tests/test_a_metadata_type_for_every_common_microscope.py`` feeds
#:     every one of these through its own pattern and through every other.
#: ``groups``
#:     what each named group means, in this convention's own vocabulary --
#:     'site' and 'field' are the same axis under two vendors' names, and a
#:     user reading the row should see the word their software uses.
#: ``source``
#:     where the convention and the examples came from.
#: ``status``
#:     ``'confirmed'`` when the pattern was read off vendor documentation,
#:     Bio-Formats' own reader source, or a published dataset's file listing;
#:     ``'provisional'`` when it was inferred from prose or from a single
#:     report. The dropdown SAYS WHICH, because a user whose instrument is
#:     guessed at should know before a run rather than after.
#:
#: NO OPTIONAL CAPTURING GROUPS, and this is a correctness rule rather than a
#: style one. :func:`spacr.utils._extract_filename_metadata` reads
#: ``match.group('fieldID')[0]`` to decide whether to unpad it; a group that
#: did not participate returns ``None`` and that subscript raises TypeError,
#: which the caller does not catch. A group that cannot always match belongs
#: in a second convention, not behind a ``?``.
_METADATA_CONVENTIONS = (
    {
        "key": "cellvoyager",
        "label": "Yokogawa CV7000 / CV8000 (CellVoyager)",
        "vendor": "Yokogawa",
        "instrument": "CV7000 / CV8000 (CellVoyager)",
        "pattern": (
            "(?P<plateID>.*)_(?P<wellID>.*)_T(?P<timeID>.*)F(?P<fieldID>.*)"
            "L(?P<laserID>..)A(?P<AID>..)Z(?P<sliceID>.*)C(?P<chanID>.*)"
            ".{ext}"),
        "ext": "verbatim",
        "examples": (
            "AssayPlate_Greiner_#655090_B02_T0001F004L01A01Z01C01.tif",
            "20200812-CardiomyocyteDifferentiation14-Cycle1_B03_"
            "T0001F036L01A01Z18C01.png",
            "210305NAR005AAN_210416_164828_B11_T0001F006L01A04Z14C01.tif",
            "CM00619158_A01_T0001F001L01A01Z01C01.tif",
        ),
        "groups": {
            "plateID": "plate / measurement name",
            "wellID": "well, as A01 .. P24",
            "timeID": "time point",
            "fieldID": "field within the well",
            "laserID": "laser / light source index",
            "AID": "action index within the timeline",
            "sliceID": "Z plane",
            "chanID": "channel",
        },
        "source": (
            "spaCR's original built-in. The first three examples are "
            "regression fixtures in fractal-analytics-platform/"
            "fractal-tasks-core 1.6.0, tests/"
            "test_unit_parse_metadata_from_filename.py, taken from real "
            "CV7000/CV8000 acquisitions; the fourth is quoted as a typical "
            "Yokogawa filename in Novartis/Jenkins-LSCI "
            "userContent/Yokogawa/readme.txt. Note the second is a .png -- "
            "the convention is the NAME, not the container."),
        "status": "confirmed",
    },
    {
        "key": "cq1",
        "label": "Yokogawa CQ1 (benchtop confocal)",
        "vendor": "Yokogawa",
        "instrument": "CQ1",
        "pattern": (
            "W(?P<wellID>.*)F(?P<fieldID>.*)T(?P<timeID>.*)"
            "Z(?P<sliceID>.*)C(?P<chanID>.*).{ext}"),
        "ext": "verbatim",
        "examples": (
            "W0262F0001T0001Z000C2.tif",
            "W1F001T0001Z01C1.tif",
        ),
        "groups": {
            "wellID": "well INDEX, not name -- 1 is A01; "
                      "spacr.utils._convert_cq1_well_id turns it into one",
            "fieldID": "field within the well",
            "timeID": "time point",
            "sliceID": "Z plane",
            "chanID": "channel",
        },
        "source": (
            "First example is a real CQ1 file, Projection/"
            "W0262F0001T0001Z000C2.tif, quoted from the dataset a user "
            "uploaded for forum.image.sc thread 52615 (\"Opening images "
            "from a Yokogawa CQ1 with Bio-Formats\", CQ1 Software 1.05). "
            "Second is what Bio-Formats itself builds: "
            "SINGLE_TIFF_PATH_BUILDER = \"W%dF%03dT%04dZ%02dC%d.tif\" in "
            "components/formats-gpl/src/loci/formats/in/"
            "CellVoyagerReader.java:112. THE TWO DISAGREE ABOUT PADDING -- "
            "F0001 against F001, Z000 against Z01 -- which is why this "
            "pattern is left as the loose `.*` it has always been rather "
            "than tightened to a digit count that would drop one of them."),
        "status": "confirmed",
    },
    {
        "key": "auto",
        "label": "Auto -- rename to Yokogawa naming first, then parse",
        "vendor": "spaCR",
        "instrument": "any -- detected from the folder",
        "pattern": (
            "(?P<plateID>.*)_(?P<wellID>.*)_T(?P<timeID>.*)F(?P<fieldID>.*)"
            "L(?P<laserID>.*)C(?P<chanID>.*).tif"),
        "ext": "verbatim",
        "examples": (
            "plate1_A01_T0001F001L01C01.tif",
            "plate1_D05_T0001F001L01C01.tif",
        ),
        "groups": {
            "plateID": "plate name",
            "wellID": "well",
            "timeID": "time point",
            "fieldID": "field",
            "laserID": "laser / light source index",
            "chanID": "channel",
        },
        "source": (
            "Both examples are names spaCR itself writes and reads: "
            "spacr/io.py:8176 builds "
            "f\"{well}_T{t:04d}F{field:03d}L01C{chan:02d}.tif\" and the "
            "plate-prefixed forms are pinned in "
            "tests/test_a_failed_regex_conversion_never_renumbers_wells.py"
            " and tests/test_a_stack_file_spacr_wrote_is_one_it_can_read.py."
            " spaCR's own intermediate naming. The extension is LITERAL "
            "'.tif' here and not the requested img_format, which is how this "
            "arm has always behaved: the rename writes .tif whatever came "
            "in."),
        "status": "confirmed",
    },
    {
        "key": "custom",
        "label": "Custom -- my own regular expression",
        "vendor": "spaCR",
        "instrument": "whatever custom_regex describes",
        "pattern": "({custom_regex}).{ext}",
        "ext": "verbatim",
        "examples": (),
        "groups": {
            "wellID": "required",
            "fieldID": "required",
            "chanID": "required",
            "plateID": "optional -- the source folder name is used instead",
        },
        "source": "The user's own pattern, from the custom_regex setting.",
        "status": "confirmed",
    },
    {
        "key": "opera_phenix",
        "label": "Opera Phenix / Operetta CLS (Harmony export)",
        "vendor": "PerkinElmer / Revvity",
        "instrument": "Opera Phenix, Operetta, Operetta CLS (Harmony)",
        "pattern": (
            r"(?P<wellID>r\d{2}c\d{2})f(?P<fieldID>\d{2})p(?P<sliceID>\d{2})"
            r"(?P<AID>(?:rc\d+)?)-ch(?P<chanID>\d+)sk(?P<timeID>\d+)"
            r"fk\d+fl\d+\.(?i:{ext})$"),
        "ext": "suffix",
        "examples": (
            "r01c01f01p01-ch1sk1fk1fl1.tiff",
            "r01c01f01p01-ch2sk1fk1fl1.tiff",
            "r04c17f01p07-ch1sk1fk1fl1.tiff",
            "r03c03f01p01-ch1sk1fk1fl1.tiff",
        ),
        "groups": {
            "wellID": "row AND column together, as the literal 'r01c01'. "
                      "Harmony never writes 'A01', so there is no well NAME "
                      "in the filename to recover -- spacr.regex_infer's "
                      "WELL_PATTERNS already treats r##c## as a well",
            "fieldID": "field / site within the well",
            "sliceID": "focal plane",
            "AID": "the 'rc' record token, present in some Harmony versions "
                   "and empty in most",
            "chanID": "channel",
            "timeID": "the sk time-sequence index",
        },
        "source": (
            "The first two are real files listed live in the public AWS "
            "Open Data bucket cellpainting-gallery, "
            "cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1/images/"
            "BR00116991__2020-11-05T19_51_35-Measurement1/Images/. The "
            "third is one of six files an Operetta CLS user attached to "
            "forum.image.sc thread 41647; the fourth is from an Opera "
            "Phenix user on forum.image.sc thread 16389. PerkinElmer's own "
            "wording, quoted on the OME forum (viewtopic.php?p=3246), gives "
            "r=row, c=column, f=field, p=plane, ch=channel and warns that "
            "'the remaining parts of the file name are currently not in "
            "use' -- which is why fk and fl are matched and discarded."),
        "status": "confirmed",
    },
    {
        "key": "imagexpress",
        "label": "ImageXpress / MetaXpress (Plate_A01_s1_w1)",
        "vendor": "Molecular Devices",
        "instrument": "ImageXpress Micro / Confocal, MetaXpress export",
        "pattern": (
            r"(?P<plateID>.+)_(?P<wellID>[A-Z]\d{2})_s(?P<fieldID>\d+)"
            r"_w(?P<chanID>\d)(?P<AID>[^._]*)\.(?i:{ext})$"),
        "ext": "suffix",
        "examples": (
            "ACHN-20X-P009041_B02_s1_w1836B5662-3526-47FE-A453-"
            "0D1775E08D17.tif",
            "20220209-exp50-test2_B01_s1_w16DE341F2-7E20-4CE1-859F-"
            "E368584F24DB.tif",
            "Plate1_A01_s1_w1.TIF",
            "TE12345_A05_s1_w1.tif",
        ),
        "groups": {
            "plateID": "plate / experiment name",
            "wellID": "well, A01 .. P24",
            "fieldID": "SITE in MetaXpress's own vocabulary -- the field "
                       "within the well",
            "chanID": "WAVELENGTH in MetaXpress's vocabulary -- the channel. "
                      "ONE DIGIT, deliberately: MetaXpress appends a GUID "
                      "straight after it with no separator, and a greedy "
                      "\\d+ reads w1836B56.. as wavelength 1836",
            "AID": "the per-acquisition GUID MetaXpress stamps on, or empty",
        },
        "source": (
            "First example from pharmbio/omero pull request 2, an OMERO "
            "import pattern file written against a real ImageXpress plate; "
            "second from a user's own Cell Painting export on "
            "forum.image.sc thread 63315; third is the tree "
            "tests/import_corpus.py::build_imagexpress writes; fourth is "
            "the literal example in CellProfiler's own Metadata module "
            "manual. Bio-Formats builds the same grammar in "
            "components/formats-gpl/src/loci/formats/in/"
            "MetaxpressTiffReader.java:168-190 -- plateName + well, then "
            "'_s'+site, '_w'+channel, and .tif or .TIF. THE _thumb SIDECAR "
            "IS DELIBERATELY NOT MATCHED: MetaXpress writes "
            "..._w1_thumb_<GUID>.tif beside every full-resolution file, and "
            "matching it would double every field."),
        "status": "confirmed",
    },
    {
        "key": "incell",
        "label": "IN Cell Analyzer (A - 01(fld 1 wv ...))",
        "vendor": "GE / Cytiva",
        "instrument": "IN Cell Analyzer 1000 / 2000 / 2200 / 6000",
        "pattern": (
            r"(?P<wellID>[A-Z] - \d{1,2})\(fld (?P<fieldID>\d+) "
            r"wv (?P<chanID>[^)]+?)(?: z \d+)?(?: time \d+ - \d+ms)?\)"
            r"\.(?i:{ext})$"),
        "ext": "suffix",
        "examples": (
            "A - 01(fld 2 wv Blue - FITC).tif",
            "A - 01(fld 1 wv UV - DAPI z 3 time 2 - 2636ms).tif",
            "C - 6(fld 1 wv TL - Bright field - open z 01).tif",
        ),
        "groups": {
            "wellID": "row letter, ' - ', column -- kept whole, because "
                      "the row and the column are separated by a SPACE "
                      "HYPHEN SPACE and there is no 'A01' anywhere in the "
                      "name to recover",
            "fieldID": "field",
            "chanID": "the excitation and emission filter names, e.g. "
                      "'UV - DAPI'. The z and exposure suffixes are matched "
                      "and discarded rather than captured: they are absent "
                      "from most of these names, and a group that does not "
                      "always participate crashes the importer",
        },
        "source": (
            "Bio-Formats constructs exactly this string in "
            "components/formats-gpl/src/loci/formats/in/InCellReader.java"
            ":436-438 -- getWellRowName(row) + \" - \" + (col+1) + \"(fld \" "
            "+ (field+1) + \" wv \" + exFilter + \" - \" + emFilter + "
            "\").tif\". The first two examples are real IN Cell filenames "
            "quoted in CellProfiler issue 3725; the third is a user's own "
            "2 TB IN Cell dataset on forum.image.sc thread 16010. "
            "SINGLE-CHANNEL IN CELL NAMES ARE NOT COVERED: 'A - 13(fld 2)"
            ".tif' carries no channel at all, so there is nothing for "
            "chanID to read."),
        "status": "confirmed",
    },
    {
        "key": "arrayscan",
        "label": "ArrayScan / CellInsight, fixed endpoint (A01f01d2)",
        "vendor": "Thermo Fisher",
        "instrument": "Cellomics ArrayScan VTI / XTI, CellInsight CX5 / CX7",
        "pattern": (
            r"(?!.*_[RDM]_p\d+_z\d+_\d+_)(?:.*_)?"
            r"(?P<wellID>[A-Z]\d{2})f(?P<fieldID>\d{2,3})"
            r"d(?P<chanID>\d)\.(?i:{ext})$"),
        "ext": "suffix",
        "examples": (
            "A01f01d2.TIF",
            "020506platerun_A01f01d2.TIF",
        ),
        "groups": {
            "wellID": "well, A01 .. Z99",
            "fieldID": "field, TWO DIGITS AND ZERO-BASED -- f00 is field 1",
            "chanID": "the DYE index, one digit and zero-based -- d0 is "
                      "channel 1, six channels maximum",
        },
        "source": (
            "Cellomics, Inc., 'ArrayScan VTI HCS Reader: User's Guide "
            "Addendum', chapter 5, 'Image File Naming Convention for Fixed "
            "Endpoint Plates': <UniquePlateID>_<WellName>f<iField>d<iDye>."
            "<extension>, with both examples above given verbatim and the "
            "plate id documented as OPTIONAL -- which is why the prefix "
            "here is matched and discarded rather than captured, and the "
            "plate name comes from the folder. Bio-Formats agrees: "
            "CellomicsReader.java:86 compiles "
            "\"(.*)_(\\p{Alpha}\\d{2})(f\\d{2,3})?(d\\d+)?[^_]+$\". "
            "THE LEADING NEGATIVE LOOKAHEAD IS NOT DECORATION. EVOS reuses "
            "this exact <well>f<field>d<dye> tail, so "
            "scan_R_p2_z1_0_B03f01d0.tif matched BOTH conventions until it "
            "was added -- an ArrayScan plate and an EVOS plate scan silently "
            "reading as each other is precisely the defect this table exists "
            "to close."),
        "status": "confirmed",
    },
    {
        "key": "arrayscan_kinetic",
        "label": "ArrayScan / CellInsight, kinetic (i3t001A01f01d2)",
        "vendor": "Thermo Fisher",
        "instrument": "Cellomics ArrayScan VTI / XTI, CellInsight CX5 / CX7",
        "pattern": (
            r".*i3t(?P<timeID>\d{3})(?P<wellID>[A-Z]\d{2})"
            r"f(?P<fieldID>\d{2,3})d(?P<chanID>\d)\.(?i:{ext})$"),
        "ext": "suffix",
        "examples": (
            "i3t001A01f01d2.TIF",
            "020506plateruni3t001A01f01d2.TIF",
        ),
        "groups": {
            "timeID": "time point, THREE DIGITS AND ONE-BASED -- i3t001 is "
                      "the first, unlike the field and dye indices beside "
                      "it, which are zero-based",
            "wellID": "well",
            "fieldID": "field, zero-based",
            "chanID": "dye index, zero-based",
        },
        "source": (
            "The same Cellomics addendum, 'Image File Naming Convention for "
            "Kinetic Plates': <UniquePlateID>i3t<iTimePoint><WellName>"
            "f<iField>d<iDye>. Both examples are the manual's own. NOTE "
            "THERE IS NO SEPARATOR before i3t, which is why the fixed-"
            "endpoint pattern above requires an underscore before the well "
            "-- without that requirement it would also match these names "
            "and read the time point as part of the plate id."),
        "status": "confirmed",
    },
    {
        "key": "evos",
        "label": "EVOS plate scan (scan_R_p2_z1_0_B03f01d0)",
        "vendor": "Thermo Fisher",
        "instrument": "EVOS FL Auto 2 / M7000 (and, untested, M5000)",
        "pattern": (
            r"(?P<plateID>.+)_(?P<AID>[RDM])_p(?P<timeID>\d+)"
            r"_z(?P<sliceID>\d+)_(?P<laserID>\d+)_(?P<wellID>[A-Z]\d{2})"
            r"f(?P<fieldID>\d{2})d(?P<chanID>\d)\.(?i:{ext})$"),
        "ext": "suffix",
        "examples": (
            "scan_R_p2_z1_0_B03f01d0.tif",
        ),
        "groups": {
            "plateID": "the capture prefix -- 'scan' for an Automate-tab "
                       "plate scan, 'image' for a manual Capture-tab shot",
            "AID": "image format: R raw, D displayed, M merged",
            "timeID": "time point",
            "sliceID": "Z plane",
            "laserID": "the GRID index, for multi-grid vessels. Named "
                       "laserID only because that is the group spaCR's "
                       "importer already carries for an axis it does not "
                       "read",
            "wellID": "row letter and column, run together as B03",
            "fieldID": "field, in capture order",
            "chanID": "channel -- d0 for a single-channel assay",
        },
        "source": (
            "Thermo Fisher, 'EVOS FL Auto 2 Imaging System User Guide' "
            "(MAN0014072) p.148 and 'EVOS M7000 Imaging System User Guide' "
            "(MAN0018326) p.198, both carrying the same 'File naming "
            "convention' section and the same labelled example. PROVISIONAL "
            "FOR ONE REASON ONLY: the vendor documents the template in full "
            "but publishes exactly ONE literal filename, and no second "
            "EVOS name could be sourced anywhere. The grammar is not in "
            "doubt; the breadth of it is."),
        "status": "provisional",
    },
    {
        "key": "scanr",
        "label": "ScanR (A1--W00001--P00001--Z00000--T00000--Channel)",
        "vendor": "Olympus / Evident",
        "instrument": "ScanR high-content screening station",
        "pattern": (
            r"(?P<wellID>[A-Z]\d{1,2})--W\d{5}--P(?P<fieldID>\d{5})"
            r"--Z(?P<sliceID>\d{5})--T(?P<timeID>\d{5})"
            r"--(?P<chanID>[^.]+)\.(?i:{ext})$"),
        "ext": "suffix",
        "examples": (
            "A1--W00001--P00001--Z00000--T00000--Alexa 488.tif",
        ),
        "groups": {
            "wellID": "the well NAME, which ScanR writes before the "
                      "well INDEX -- the W block is the index and is "
                      "matched and discarded",
            "fieldID": "the P block: the position within the well",
            "sliceID": "Z plane",
            "timeID": "time point",
            "chanID": "the channel NAME, free text -- 'Alexa 488', 'GFP'",
        },
        "source": (
            "forum.image.sc thread 14872, 'Olympus Scan^R metadata', where "
            "a ScanR user gives 'A1--W00001--P00001--Z00000--T00000--Alexa "
            "488' as an example of their own 96/384-well filenames and a "
            "CellProfiler developer answers with a regex naming the first "
            "block Well and the rest W/Position/Z/T/Channel. Bio-Formats "
            "confirms the block grammar independently: ScanrReader.java's "
            "getBlock() zero-pads to five digits and prepends the axis "
            "letter, so getBlock(1, \"W\") is 'W00001'. PROVISIONAL "
            "because ONE literal filename could be sourced, and because "
            "Bio-Formats matches these blocks by substring search rather "
            "than by a pattern, so it does not vouch for the '--' "
            "separators or for the leading well name."),
        "status": "provisional",
    },
    {
        "key": "cytation",
        "label": "Cytation / Gen5 (A1ROI1_02_1_1_Bright Field_..._003)",
        "vendor": "Agilent / BioTek",
        "instrument": "Cytation 1 / 5 / C10, Gen5 export",
        "pattern": (
            r"(?P<wellID>[A-H]\d{1,2})(?P<AID>(?:ROI\d+)?)_\d{1,2}"
            r"_(?P<laserID>\d)_(?P<fieldID>\d{1,2})_(?P<chanID>.+)"
            r"_(?P<timeID>\d{3})\.(?i:{ext})$"),
        "ext": "suffix",
        "examples": (
            "A1ROI1_02_1_1_Bright Field_High Contrast_003.tif",
        ),
        "groups": {
            "wellID": "well",
            "AID": "the region of interest, on a Cytation C10; empty on the "
                   "widefield Cytations",
            "laserID": "the WAVELENGTH index, named as such by the "
                       "CellProfiler pipeline below",
            "fieldID": "the SITE within the well",
            "chanID": "the channel name, and on a C10 the capture mode "
                      "after it -- 'Bright Field_High Contrast'",
            "timeID": "the trailing sequence number",
        },
        "source": (
            "The example is a real Cytation C10 .tif attached to "
            "forum.image.sc thread 69372 by Christian Tischer (EMBL) while "
            "reporting a Bio-Formats metadata bug. The field names come "
            "from a CellProfiler pipeline built and run against real "
            "Cytation 5 files -- Zenodo record 5933221, "
            "TricornutumSegmentation.cppipe, whose regex reads "
            "^(?P<Well>[A-H][0-9]{1,2})_(?P<Plate>[0-9]{1,2})_"
            "(?P<Wavelength>[0-9]{1})_(?P<site>[0-9]{1,2})_"
            "(?P<channel>.*)_001. PROVISIONAL: Agilent's own Gen5 export "
            "documentation is behind a WAF that refused every fetch, so "
            "the vendor's wording is unverified, and only one literal "
            "filename could be sourced."),
        "status": "provisional",
    },
    {
        "key": "leica_matrix_screener",
        "label": "LAS AF / LAS X Matrix Screener (image--L..--S..--U..)",
        "vendor": "Leica",
        "instrument": "Matrix Screener on SP5 / SP8",
        "pattern": (
            r"image--L\d{2,4}--S\d{2,4}--(?P<wellID>U\d{2,4}--V\d{2,4})"
            r"--J\d{2,4}--E\d{2,4}--O\d{2,4}"
            r"--(?P<fieldID>X\d{2,4}--Y\d{2,4})--T(?P<timeID>\d{2,4})"
            r"--Z(?P<sliceID>\d{2,4})--C(?P<chanID>\d{2,4})"
            r"(?:\.ome)?\.(?i:{ext})$"),
        "ext": "suffix",
        "examples": (
            "image--L00--S00--U00--V00--J20--E00--O00--X00--Y00--T00--Z00--"
            "C00.ome.tif",
            "image--L00--S00--U00--V00--J20--E00--O00--X00--Y00--T00--Z00--"
            "C01.ome.tif",
            "image--L00--S00--U00--V00--J20--E00--O00--X00--Y01--T00--Z00--"
            "C00.ome.tif",
        ),
        "groups": {
            "wellID": "the chamber, as 'U00--V00' -- U is the well row and "
                      "V the well column, kept together because neither "
                      "alone identifies a well",
            "fieldID": "the field, as 'X00--Y00' -- field column and row",
            "timeID": "time point",
            "sliceID": "Z plane",
            "chanID": "channel",
        },
        "source": (
            "All three are literal filenames in the test fixture tree of "
            "arve0/leicaexperiment, the reference parser for this format "
            "(test/experiment--test/slide--S00/chamber--U00--V00/"
            "field--X00--Y00/). U and V are named as the well coordinates "
            "and X and Y as the field position in the READMEs of "
            "arve0/leicaexperiment and VolkerH/"
            "LeicaMatrixScreener2BigStitcher, which also identifies J as "
            "the scan JOB number. L, E and O ARE NOT DOCUMENTED ANYWHERE "
            "THAT COULD BE FOUND -- they are matched and discarded rather "
            "than guessed at."),
        "status": "confirmed",
    },
    {
        "key": "leica_lasx_series",
        "label": "LAS X series export (Series002_z00_ch00)",
        "vendor": "Leica",
        "instrument": "LAS AF / LAS X, TIFF series export from a .lif",
        "pattern": (
            r"(?P<wellID>[A-Za-z]+)(?P<fieldID>\d+)_z(?P<sliceID>\d+)"
            r"_ch(?P<chanID>\d+)\.(?i:{ext})$"),
        "ext": "suffix",
        "examples": (
            "Series002_z00_ch00.tif",
            "Series002_z01_ch00.tif",
            "Series002_z00_ch01.tif",
        ),
        "groups": {
            "wellID": "the WORD of the series label -- 'Series', 'Pos'. A "
                      "series export has no plate, so every series lands in "
                      "one well and the series NUMBER becomes the field",
            "fieldID": "the series / position number",
            "sliceID": "Z plane",
            "chanID": "channel",
        },
        "source": (
            "The three examples are the literal 'Example of image "
            "filenames' in the doc comment of "
            "standardgalactic/wingj matlab-toolbox/lib/"
            "open_image_sequence.m. PROVISIONAL BECAUSE OF THE "
            "ATTRIBUTION, NOT THE FILENAMES: this is the long-recognised "
            "LAS AF/LAS X series-export shape and several independent "
            "repositories parse exactly it, but no source found says "
            "'Leica' in the same breath as one of these names, and LAS X "
            "lets the user change the template."),
        "status": "provisional",
    },
    {
        "key": "leica_lasx_series_time",
        "label": "LAS X series export, timelapse (Pos0_t000_z00_ch00)",
        "vendor": "Leica",
        "instrument": "LAS AF / LAS X, TIFF series export from a .lif",
        "pattern": (
            r"(?P<wellID>[A-Za-z]+)(?P<fieldID>\d+)(?P<AID>(?:_S\d+)?)"
            r"_t(?P<timeID>\d+)_z(?P<sliceID>\d+)_ch(?P<chanID>\d+)"
            r"\.(?i:{ext})$"),
        "ext": "suffix",
        "examples": (
            "Pos007_S001_t50_z00_ch00.tif",
            "Pos007_S001_t6_z00_ch00.tif",
            "Pos0_t000_z00_ch00.tif",
        ),
        "groups": {
            "wellID": "the WORD of the position label -- see leica_lasx_"
                      "series",
            "fieldID": "the position number",
            "AID": "the sub-series '_S001', when the export wrote one",
            "timeID": "time point",
            "sliceID": "Z plane",
            "chanID": "channel",
        },
        "source": (
            "First two from the example dataset shipped with saenopy "
            "(rgerum/saenopy docs/source/3d_tfm/examples/4_OrganoidTFM.py, "
            "a confocal reflection acquisition of 52 z slices and 2 time "
            "points); third is the shape "
            "Alexander-Zangl/retina-track parses in "
            "src/retina_track/pipeline/io/_metadata_utils.py. PROVISIONAL "
            "for the same reason as leica_lasx_series: the filenames are "
            "real and the vendor attribution is not textually proven."),
        "status": "provisional",
    },
    {
        "key": "nikon_nis_xy",
        "label": "NIS-Elements TIFF sequence (seq0000xy01c1)",
        "vendor": "Nikon",
        "instrument": "NIS-Elements, Save/Export to TIFF Files",
        "pattern": (
            r"(?P<wellID>[A-Za-z][A-Za-z0-9_]*)xy(?P<fieldID>\d+)"
            r"c(?P<chanID>\d+)\.(?i:{ext})$"),
        "ext": "suffix",
        "examples": (
            "seq0000xy01c1.tif",
            "ldim_xy01c1.tif",
        ),
        "groups": {
            "wellID": "the FILE PREFIX the export dialog was given. There "
                      "is no well in this convention; the prefix names the "
                      "acquisition and the xy index is the position",
            "fieldID": "the xy stage position",
            "chanID": "channel",
        },
        "source": (
            "NIS-Elements Viewer User's Guide v5.21.00, sections 4.1.3-"
            "4.1.4: 'specify the prefix which will be used to name all "
            "files of the sequence. Dimension names and numbers will be "
            "appended to this prefix', with channels indexed c1, c2. First "
            "example is an OME test fixture, "
            "imaging-formats/ome-types tests/data/seq0000xy01c1.ome.xml, "
            "whose own comment names NIS-Elements 4.30; second is from "
            "DeepBioVision/DeepBIT. A NAME THAT ALSO CARRIES A TIME INDEX "
            "-- t000001xy01c1.tif -- MATCHES THIS PATTERN WITH THE "
            "TIMESTAMP READ AS THE PREFIX, which is a property of the "
            "convention rather than of the regex: NIS-Elements writes the "
            "dimensions the user ticked, in the order they chose, with no "
            "separator to tell a prefix from an index."),
        "status": "confirmed",
    },
    {
        "key": "nikon_jobs",
        "label": "NIS-Elements JOBS / HCA (WellA01_ChannelDAPI_Seq0001)",
        "vendor": "Nikon",
        "instrument": "NIS-Elements JOBS / HCA well-plate acquisition",
        "pattern": (
            r".*Well(?P<wellID>[A-Z]\d{1,2})_Channel(?P<chanID>[^.]+?)"
            r"_Seq(?P<fieldID>\d+)\.(?i:{ext})$"),
        "ext": "suffix",
        "examples": (
            "WellA01_ChannelDAPI_Seq0001.nd2",
            "20201115_184113_906__WellA2_ChannelDAPI_Seq0003.nd2",
            "WellA1_ChannelDAPI_1x1-GFP_1x1_Seq0012.nd2",
        ),
        "groups": {
            "wellID": "well -- JOBS writes A1 as often as A01, so the "
                      "column is 1 or 2 digits",
            "chanID": "the channel NAME as configured, hyphen-joined when "
                      "several were acquired at once",
            "fieldID": "the Seq acquisition index within the well",
        },
        "source": (
            "Three independent published datasets: "
            "ZMB-UZH/omero-docker-extended's own filename-parsing "
            "reference table, arjunrajlaboratory/FateMap_Goyal2023, and "
            "cheeseman-lab/aconcagua-analysis. No official Nikon manual "
            "page states this form, but it recurs identically across "
            "unrelated labs and years. The leading timestamp is matched "
            "and discarded rather than read as a plate, so the plate name "
            "comes from the folder."),
        "status": "confirmed",
    },
    {
        "key": "micromanager_mda",
        "label": "Micro-Manager 2.0 MDA, separate image files",
        "vendor": "Open source",
        "instrument": "Micro-Manager 2.0 multi-dimensional acquisition",
        "pattern": (
            r"(?P<wellID>img)_channel(?P<chanID>\d{3})"
            r"_position(?P<fieldID>\d{3})_time(?P<timeID>\d{9})"
            r"_z(?P<sliceID>\d{3})\.(?i:{ext})$"),
        "ext": "suffix",
        "examples": (
            "img_channel000_position000_time000000000_z000.tif",
            "img_channel001_position002_time000000005_z003.tif",
        ),
        "groups": {
            "wellID": "the literal 'img'. Micro-Manager has no plate and no "
                      "well; every image lands in one well and the STAGE "
                      "POSITION becomes the field",
            "chanID": "channel index",
            "fieldID": "stage position index",
            "timeID": "time point -- NINE digits, not six",
            "sliceID": "Z slice index",
        },
        "source": (
            "Micro-Manager's own writer, "
            "mmstudio/src/main/java/org/micromanager/data/internal/"
            "StorageSinglePlaneTiffSeries.java::createFileName, which "
            "appends '_<axis><index>' for each axis in ALPHABETICAL order "
            "-- channel, position, time, z -- with %03d for every axis "
            "except time, which gets %09d. The axis names come from "
            "Coords.java. The examples are what that code writes; they "
            "were not copied from a dataset listing."),
        "status": "confirmed",
    },
    {
        "key": "zeiss_zen_split_tiles",
        "label": "ZEN split tiles / scenes (name_S00001_T00001_C00001)",
        "vendor": "Zeiss",
        "instrument": "ZEN blue, 'Save Tiles as Single Images' OAD script",
        "pattern": (
            r"(?P<plateID>.+)_S(?P<wellID>\d{5})_T(?P<fieldID>\d{5})"
            r"_C(?P<chanID>\d{5})\.(?i:{ext})$"),
        "ext": "suffix",
        "examples": (
            "mydata_S00001_T00001_C00001.tiff",
            "mydata_S00001_T00002_C00001.tiff",
        ),
        "groups": {
            "plateID": "the CZI's own file name, without its extension",
            "wellID": "SCENE index. A scene is a position on the slide or "
                      "plate, which is the closest thing ZEN writes to a "
                      "well",
            "fieldID": "TILE index within the scene",
            "chanID": "channel",
        },
        "source": (
            "Carl Zeiss Microscopy's own published OAD script, "
            "zeiss-microscopy/OAD Scripts/Data_Tools/"
            "SaveSingeTiles_to_Folder.py, line 143: imgTile.Name = "
            "nameParent[:-4] + '_S' + s_str + '_T' + m_str + '_C' + c_str + "
            "ft[1:], where addzeros() pads to five characters. PROVISIONAL, "
            "AND THE REASON MATTERS: the examples are CONSTRUCTED from that "
            "code rather than observed in a dataset, and this is a macro a "
            "user has to run -- ZEN's built-in Image Export dialog takes a "
            "user-typed prefix and appends dimension letters, so there is "
            "no single Zeiss default to confirm. No Zeiss well-plate "
            "filename could be sourced at all; the '_A01_T0001F001L01A01"
            "Z01C01' names that circulate online as 'well-plate naming' "
            "are Yokogawa's, not Zeiss's."),
        "status": "provisional",
    },
)


#: Vendors in dropdown order. Yokogawa leads because `cellvoyager` is the
#: default and a user who does not change the setting should find it at the
#: top; then the rest, roughly by how often a high-content plate arrives
#: from one; then spaCR's own `auto` and `custom`, which are not microscopes
#: and are the last thing to reach for rather than the first.
_METADATA_VENDOR_ORDER = (
    "Yokogawa", "PerkinElmer / Revvity", "Molecular Devices",
    "Thermo Fisher", "GE / Cytiva", "Olympus / Evident", "Zeiss", "Nikon",
    "Leica", "Agilent / BioTek", "Sartorius", "Open source", "spaCR",
)


def _metadata_convention(key):
    """One record from the convention table, or ``None``.

    :param key: the stored ``metadata_type`` value. Matched EXACTLY -- a
        settings file saying ``'CellVoyager'`` is a typo the caller should
        hear about, not a spelling to normalise silently.
    :returns: the record dict, or ``None`` when nothing has that key.
    """
    for record in _METADATA_CONVENTIONS:
        if record["key"] == key:
            return record
    return None


def _metadata_convention_keys():
    """Every key in the table, in dropdown order.

    :returns: a tuple of the stored ``metadata_type`` values.
    """
    return tuple(record["key"] for record in _METADATA_CONVENTIONS)


def _metadata_pattern(key, img_format="tif", custom_regex=None):
    """The regular expression one convention parses filenames with.

    :param key: the stored ``metadata_type`` value.
    :param img_format: the image extension, with or without its dot. ``None``
        means ``'tif'``.
    :param custom_regex: the user's own pattern, for ``'custom'``.
    :returns: the pattern string.
    :raises KeyError: when ``key`` names no convention. The caller decides
        what to say about it; :func:`spacr.utils._get_regex` turns this into
        a ValueError naming the whole vocabulary.
    """
    record = _metadata_convention(key)
    if record is None:
        raise KeyError(key)
    if img_format is None:
        img_format = "tif"
    pattern = record["pattern"]
    if "{custom_regex}" in pattern:
        pattern = pattern.replace("{custom_regex}", str(custom_regex))
    if record.get("ext") == "verbatim":
        return pattern.replace("{ext}", str(img_format))
    return pattern.replace("{ext}",
                           re.escape(str(img_format).lstrip(".")))


def _metadata_pattern_any_extension(key, extensions):
    """The pattern for ``key`` with a whole alternation where the extension goes.

    For a caller that is testing names it has not chosen an extension for --
    :func:`spacr.validate._candidate_patterns` sweeps a raw folder that may
    hold ``.tif`` beside ``.png``.

    :param key: the stored ``metadata_type`` value.
    :param extensions: extensions WITHOUT their dots, e.g. ``('tif', 'png')``.
    :returns: the pattern.
    :raises KeyError: when ``key`` names no convention.
    """
    record = _metadata_convention(key)
    if record is None:
        raise KeyError(key)
    alternation = "(?:" + "|".join(str(e).lstrip(".")
                                   for e in extensions) + ")"
    return record["pattern"].replace("{ext}", alternation)


def _metadata_convention_menu():
    """The table as a grouped menu: ``[(vendor, [(key, label, status)])]``.

    Grouped because an unlabelled alphabetical list of twenty conventions is
    a menu that hides its own contents -- a Leica user scanning it for
    'Leica' finds `las_x_tiff_series` only if they already knew the name.
    Vendors appear in :data:`_METADATA_VENDOR_ORDER`; anything whose vendor
    is not listed there follows, in table order, so a convention added
    without touching the order list still appears.

    :returns: a list of ``(vendor, rows)`` pairs; each row is
        ``(key, label, status)``.
    """
    groups = {}
    for record in _METADATA_CONVENTIONS:
        groups.setdefault(record["vendor"], []).append(
            (record["key"], record["label"], record["status"]))
    ordered = [(vendor, groups.pop(vendor))
               for vendor in _METADATA_VENDOR_ORDER if vendor in groups]
    ordered.extend(sorted(groups.items()))
    return ordered


def _metadata_convention_example(key):
    """The first real example filename for ``key``, or ``''``.

    :param key: the stored ``metadata_type`` value.
    :returns: one filename, shown under the dropdown so the user can compare
        it against what is actually in their folder.
    """
    record = _metadata_convention(key)
    if record is None:
        return ""
    examples = record.get("examples") or ()
    return examples[0] if examples else ""


def _metadata_parse_report(names, key, img_format="tif", custom_regex=None):
    """How many of ``names`` one convention parses, and the first that fails.

    Opens nothing and touches no disk: it is handed the names. The caller
    reads the directory, which is the part that must not run on the GUI
    thread.

    :param names: bare filenames.
    :param key: the stored ``metadata_type`` value.
    :param img_format: the extension to build the pattern for.
    :param custom_regex: the user's own pattern, for ``'custom'``.
    :returns: ``(matched, total, first_unparsed)``. ``first_unparsed`` is
        ``''`` when everything matched. A pattern that does not compile
        reports zero matches rather than raising -- a user's own regex is a
        thing they are still typing.
    """
    names = list(names)
    try:
        compiled = re.compile(_metadata_pattern(key, img_format,
                                                custom_regex))
    except (KeyError, re.error):
        return 0, len(names), names[0] if names else ""
    matched = 0
    first_unparsed = ""
    for name in names:
        if compiled.match(name):
            matched += 1
        elif not first_unparsed:
            first_unparsed = name
    return matched, len(names), first_unparsed


def _metadata_autodetect(names, img_format="tif"):
    """Rank every convention by how many of ``names`` it parses.

    OFFERED, NEVER APPLIED. The caller shows the ranking and lets the user
    choose: a convention that parses 100% of a folder can still be the wrong
    one -- two of the patterns here are deliberately loose because they are
    pinned byte for byte -- and a setting changed without being asked for is
    the failure this whole item is about.

    ``custom`` is excluded: it parses whatever the user last typed, which is
    not evidence about the folder. So is ``auto``, whose whole behaviour is
    to rename first and which therefore cannot be judged by matching alone.

    :param names: bare filenames.
    :param img_format: the extension to build the patterns for.
    :returns: ``[(key, matched, total)]``, best first, then by table order.
        Conventions that matched nothing are omitted -- a list of twenty
        zeroes is not a report.
    """
    names = list(names)
    ranked = []
    for index, record in enumerate(_METADATA_CONVENTIONS):
        key = record["key"]
        if key in ("custom", "auto"):
            continue
        matched, total, _first = _metadata_parse_report(names, key,
                                                        img_format)
        if matched:
            ranked.append((-matched, index, key, matched, total))
    ranked.sort()
    return [(key, matched, total) for _n, _i, key, matched, total in ranked]
