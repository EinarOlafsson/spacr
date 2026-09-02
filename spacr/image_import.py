"""Work out how a folder of images is named, instead of being told.

Instruction 363: spaCR's import path asks the user to pick a filename
convention from a closed list -- ``cellvoyager``, ``cq1``, ``custom`` (write
your own regular expression), ``auto`` -- and the corpus in
``tests/import_corpus.py`` measures what that costs: of ten real acquisition
layouts, two parse and eight recover nothing. Opera Phenix and ImageXpress
are among the eight.

THE FIXED LIST IS THE PROBLEM, NOT THE REGULAR EXPRESSIONS IN IT. Every one
of the ten encodes the SAME six facts -- plate, well, field, channel, z, t --
and differs only in how. So rather than matching a whole filename against one
of N templates, this module reads the parts of the name that VARY across the
folder and works out what each varying part means.

WHY VARIANCE IS THE RIGHT SIGNAL. A token that is the same in every file
carries no information: ``L01`` in ``plate1_A01_T0001F001L01A01Z01C01.tif`` is
a constant for the whole plate and identifies nothing. A token that takes two
values across eight files is an axis with two positions. Reading the folder
tells you which tokens are axes; nothing about the filename alone can.

WHY MARKERS ARE STILL NEEDED. Variance says a token IS an axis; it cannot say
WHICH axis. ``F001`` and ``C01`` both vary and are not interchangeable. The
letter in front is what the convention uses to say so, and every convention
in the corpus uses one -- so the marker table below is the vocabulary, and it
is short because the conventions agree more than they disagree.

WHAT THIS DELIBERATELY DOES NOT DO. It never guesses a role it has no
evidence for. A token that varies and carries no recognisable marker is
reported as UNPLACED, with its values, so the caller can ask rather than
assume. Instruction 363's own criterion is that a partly-unparseable tree
"imports what it can and REPORTS the rest": a wrong answer that looks
plausible is the failure mode this whole module exists to avoid, and it is
the one the ``consolidate`` bugs demonstrated -- images disappearing with no
error rather than an import refusing.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "AXES",
    "MARKERS",
    "ImportPlan",
    "InsideFile",
    "InferredLayout",
    "TokenSlot",
    "infer_layout",
    "plan_import",
    "read_axes_inside",
    "tokenise",
]

#: The facts every layout in the corpus encodes, whatever it calls them.
AXES = ("plate", "well", "row", "column", "field", "channel", "z", "t", "tile")

#: Marker -> axis. Lower-cased, longest first at match time so ``ch`` wins
#: over ``c`` and ``sk`` over ``s``.
#:
#: EVERY ENTRY IS FROM A REAL CONVENTION IN THE CORPUS, not invented:
#: CellVoyager (T/F/L/A/Z/C), CQ1 (W/F/T/Z/C), Opera Phenix
#: (r/c/f/p/ch/sk/fl), ImageXpress (s for site, w for wavelength), and the
#: hand-made trees (field/tile/well spelled out).
MARKERS: Dict[str, str] = {
    "plate": "plate", "p": "z", "well": "well", "w": "channel",
    "r": "row", "c": "column", "ch": "channel", "chan": "channel",
    "channel": "channel", "f": "field", "fld": "field", "field": "field",
    "s": "field", "site": "field", "t": "t", "sk": "t", "time": "t",
    "z": "z", "slice": "z", "tile": "tile", "m": "tile",
}

#: ``c`` means COLUMN in Opera's ``r01c01`` and CHANNEL in ``_C01``. The
#: difference is whether an ``r`` slot was seen in the same name, so the
#: ambiguity is resolved per file rather than by preferring one globally.
_AMBIGUOUS = {"c": ("column", "channel"), "w": ("well", "channel"),
              "p": ("z", "plate")}

#: ``A01``, ``B12`` -- a well name. Anchored, so ``C01`` in a channel token is
#: not mistaken for one: a well letter is followed by two digits and nothing
#: else, and a channel marker is followed by digits and then more name.
_WELL_NAME = re.compile(r"^([A-Z])(\d{1,2})$")

_TOKEN = re.compile(r"[A-Za-z]+|\d+")


def tokenise(name: str) -> List[Tuple[str, str]]:
    """Split a name into ``(kind, text)`` runs, kind ``"alpha"`` or ``"digit"``.

    ``"plate1_A01_F001"`` becomes alpha/digit pairs. Separators are dropped:
    conventions disagree about ``_`` versus ``-`` versus nothing, and the
    disagreement carries no information.
    """
    return [("digit" if t[0].isdigit() else "alpha", t)
            for t in _TOKEN.findall(name)]


@dataclass
class TokenSlot:
    """One position in a filename, and what was seen there across the folder.

    :param index: position among the tokens of a name.
    :param marker: the alphabetic run immediately before it, lower-cased.
    :param values: every value seen at this position, in first-seen order.
    :param axis: the axis it was resolved to, or ``""`` when unplaced.
    """

    index: int
    marker: str
    values: List[str] = field(default_factory=list)
    axis: str = ""

    @property
    def varies(self) -> bool:
        """Whether this slot takes more than one value across the folder."""
        return len(set(self.values)) > 1


@dataclass
class InferredLayout:
    """What was worked out about a folder, and what was not.

    :param root: the folder examined.
    :param per_file: relative path -> ``{axis: value}`` for what was resolved.
    :param slots: the token slots, resolved and unresolved alike.
    :param unplaced: axes-shaped tokens that vary but carry no known marker,
        as ``{position: [values]}``. THE IMPORTANT FIELD: it is what the
        caller shows the user instead of guessing.
    :param skipped: files whose token shape did not match the majority, so
        nothing was claimed about them.
    :param sampled: how many files were read to decide.
    """

    root: Path
    per_file: Dict[str, Dict[str, object]] = field(default_factory=dict)
    slots: List[TokenSlot] = field(default_factory=list)
    unplaced: Dict[int, List[str]] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)
    sampled: int = 0

    @property
    def axes(self) -> set:
        """Which axes were resolved anywhere."""
        return {a for m in self.per_file.values() for a in m}

    def summary(self) -> str:
        """One line per axis, plus what could not be placed."""
        lines = [f"{len(self.per_file)} files, {len(self.axes)} axes resolved"]
        for axis in AXES:
            values = {m[axis] for m in self.per_file.values() if axis in m}
            if values:
                lines.append(f"  {axis:8s} {len(values):3d} distinct")
        for index, values in self.unplaced.items():
            lines.append(f"  UNPLACED at token {index}: "
                         f"{len(set(values))} distinct, e.g. {values[:3]}")
        if self.skipped:
            lines.append(f"  {len(self.skipped)} files of another shape, skipped")
        return "\n".join(lines)


def _shape(tokens: Sequence[Tuple[str, str]]) -> Tuple[str, ...]:
    """A name's shape: the SEQUENCE OF KINDS, not the text.

    KINDS AND NOT TEXT, and the first cut got this wrong in a way worth
    recording. Keeping the alphabetic runs seemed obviously right -- two files
    of one convention differ only in their numbers -- but the WELL LETTER is
    alphabetic and varies: ``plate1_A01_...`` and ``plate1_B02_...`` came out
    as two different shapes, the majority filter kept one, and half of every
    plate was silently dropped before anything was inferred.

    A varying letter is an AXIS, exactly like a varying number, and the shape
    must not encode it. Whether two names really follow one convention is then
    decided by the per-position analysis, which has the values to decide it
    with.
    """
    return tuple(k for k, _text in tokens)


def _resolve_marker(marker: str, seen_markers: set) -> str:
    """Which axis a marker means, given the other markers in the same name."""
    marker = marker.lower()
    if marker in _AMBIGUOUS:
        first, second = _AMBIGUOUS[marker]
        # `c` is a column only when an `r` sits beside it, as in Opera's
        # r01c01; otherwise it is a channel. Same shape of question for `w`
        # (a well in CQ1's W1F001, a wavelength in ImageXpress's _w1) and `p`.
        partner = {"column": "r", "well": "f", "z": "ch"}.get(first, "")
        return first if partner and partner in seen_markers else second
    return MARKERS.get(marker, "")


def infer_layout(root, *, sample: int = 400,
                 extensions: Sequence[str] = (".tif", ".tiff", ".png",
                                              ".jpg", ".jpeg", ".bmp")) -> InferredLayout:
    """Work out what the filenames under ``root`` encode.

    :param root: the folder to inspect.
    :param sample: how many files to read before deciding. A SAMPLE, not the
        whole tree: the answer is a naming convention, and a convention is
        visible in a few hundred names. This is what keeps inspecting a
        400-plate archive as fast as inspecting one plate, which instruction
        363 requires.
    :param extensions: which files count as images.
    :returns: an :class:`InferredLayout`. Never raises for an unrecognised
        tree -- an empty ``per_file`` with populated ``unplaced`` is the
        honest answer and the caller is expected to show it.
    """
    root = Path(root)
    paths = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            paths.append(path)
            if len(paths) >= sample:
                break
    layout = InferredLayout(root=root, sampled=len(paths))
    if not paths:
        return layout

    # The FOLDER segments are part of the name. A tree that puts the well in a
    # directory encodes exactly as much as one that puts it in the filename,
    # and reading only the basename is why per-well and per-channel trees
    # recover nothing today.
    tokenised = {}
    for path in paths:
        rel = path.relative_to(root)
        stem = str(rel.with_suffix(""))
        tokenised[str(rel)] = tokenise(stem)

    shapes = Counter(_shape(t) for t in tokenised.values())
    majority, _count = shapes.most_common(1)[0]
    members = {rel: toks for rel, toks in tokenised.items()
               if _shape(toks) == majority}
    layout.skipped = sorted(set(tokenised) - set(members))

    # Collect what each digit slot takes across the folder, with the
    # alphabetic run in front of it as its marker.
    slots: Dict[int, TokenSlot] = {}
    for toks in members.values():
        for i, (kind, text) in enumerate(toks):
            if kind != "digit":
                continue
            marker = toks[i - 1][1] if i and toks[i - 1][0] == "alpha" else ""
            slots.setdefault(i, TokenSlot(index=i, marker=marker.lower()))
            slots[i].values.append(text)

    seen = {s.marker for s in slots.values() if s.marker}
    for slot in slots.values():
        if not slot.varies:
            continue                    # a constant identifies nothing
        slot.axis = _resolve_marker(slot.marker, seen)
        if not slot.axis:
            layout.unplaced[slot.index] = list(dict.fromkeys(slot.values))
    layout.slots = [slots[i] for i in sorted(slots)]

    # A WELL LETTER IS ONE THAT VARIES. `A01` is a well name; so, by shape
    # alone, are `L01`, `Z01` and `C01` in
    # `plate1_A01_T0001F001L01A01Z01C01` -- and matching on shape marked all
    # four as wells, the last of them overwriting the channel axis. Every
    # plate then had four wells and no channels.
    #
    # The letter is what separates them: `A` takes A and B across the folder,
    # while `L`, `Z` and `C` are the same letter in every file. A constant
    # letter is part of the convention's punctuation; a varying one is an
    # axis. This is the same variance rule the digit slots use, applied to
    # the half of the name the first cut did not apply it to.
    alpha_values: Dict[int, List[str]] = defaultdict(list)
    for toks in members.values():
        for i, (kind, text) in enumerate(toks):
            if kind == "alpha":
                alpha_values[i].append(text)
    for i, values in alpha_values.items():
        if len(set(values)) < 2:
            continue                    # constant: punctuation, not an axis
        if not all(len(v) == 1 and v.isupper() for v in values):
            # A varying word rather than a letter -- a dye name in a folder,
            # say. It IS an axis, and one nothing here can name, so it is
            # reported rather than guessed at.
            layout.unplaced[i] = list(dict.fromkeys(values))
            continue
        if i + 1 in slots and _WELL_NAME.match(values[0] + slots[i + 1].values[0]):
            slots[i + 1].axis = "well"
            layout.unplaced.pop(i + 1, None)

    for rel, toks in members.items():
        found: Dict[str, object] = {}
        for i, (kind, text) in enumerate(toks):
            if kind != "digit":
                continue
            slot = slots.get(i)
            if slot is None or not slot.axis:
                continue
            if slot.axis == "well":
                found["well"] = f"{toks[i - 1][1]}{int(text):02d}"
            else:
                found[slot.axis] = int(text)
        if "row" in found and "column" in found and "well" not in found:
            # Opera keeps them apart; spaCR's vocabulary is a well name.
            found["well"] = f"{chr(ord('A') + int(found['row']) - 1)}" \
                            f"{int(found['column']):02d}"
        if found:
            layout.per_file[rel] = found
    return layout


@dataclass(frozen=True)
class InsideFile:
    """What one file's own metadata says about the axes it holds.

    :param pages: how many pages the file has. ``1`` means a plain 2-D image
        and nothing below matters.
    :param axes: the axis letters the file declares, e.g. ``"CYX"``,
        ``"ZYX"``, ``"TYX"``. Empty when the file declares none.
    :param sizes: ``{axis: length}`` for the non-spatial axes only, so
        ``{"c": 2}`` or ``{"z": 5}``.
    :param declared: whether the axes came from the FILE or were guessed.
        ``False`` with ``pages > 1`` is the honest unknown -- a multi-page
        TIFF with no metadata could be Z, T or C and nothing can say which.
    """

    pages: int
    axes: str = ""
    sizes: Dict[str, int] = field(default_factory=dict)
    declared: bool = False

    @property
    def is_ambiguous(self) -> bool:
        """Several pages and nothing saying what they are."""
        return self.pages > 1 and not self.declared


def read_axes_inside(path) -> InsideFile:
    """What ``path``'s own metadata says its pages are.

    THE AXIS THAT IS NOT IN THE NAME. ``infer_layout`` reads names, and a name
    cannot carry what an acquisition put inside the file: an OME-TIFF holding
    two channels is one filename, and a Z-stack and a timelapse of the same
    field have IDENTICAL names. Only the file says which.

    NEVER GUESSES. A multi-page TIFF with no axis metadata could be Z, T or C,
    and this returns ``declared=False`` with the page count rather than
    picking one. The caller shows that to the user; instruction 363's whole
    complaint is about a page index being treated as meaningful on its own.

    :param path: an image file.
    :returns: an :class:`InsideFile`. A file that cannot be opened at all
        comes back as ``InsideFile(pages=0)`` rather than raising -- a folder
        of thousands should not fail wholesale because one file is truncated.
    """
    try:
        import tifffile
    except ImportError:                    # pragma: no cover - tifffile ships
        return InsideFile(pages=0)

    path = Path(path)
    if path.suffix.lower() not in (".tif", ".tiff"):
        # Only TIFF carries this. A PNG or JPEG is one plane by construction.
        return InsideFile(pages=1 if path.is_file() else 0)
    try:
        with tifffile.TiffFile(str(path)) as handle:
            pages = len(handle.pages)
            series = handle.series[0] if handle.series else None
            axes = (series.axes or "") if series is not None else ""
            shape = tuple(series.shape) if series is not None else ()
            sizes = {a.lower(): int(n) for a, n in zip(axes, shape)
                     if a in "CZT"}
            # DECLARED MEANS A NAMED NON-SPATIAL AXIS WAS FOUND, and nothing
            # weaker. The first version accepted "any axis letter that is not
            # Y, X or S", which let tifffile's own `Q` through -- and `Q` is
            # precisely tifffile's word for "these pages exist and I do not
            # know what they are". An unlabelled three-page stack came back
            # declared, which is the guess this function exists not to make.
            #
            # is_ome and is_imagej are not sufficient either: a file can carry
            # either container and still not say what its pages mean.
            return InsideFile(pages=pages, axes=axes, sizes=sizes,
                              declared=bool(sizes))
    except Exception:
        # Truncated, unreadable, or not really a TIFF. Reported as unknown so
        # the folder-level scan carries on and the caller can list what it
        # could not read.
        return InsideFile(pages=0)


@dataclass
class ImportPlan:
    """What WOULD be imported, for a human to check before anything is written.

    THE PLAN IS THE FEATURE. `metadata_type` is unusable today not because its
    regular expressions are bad but because the user cannot see what they did
    until masks come out wrong, so the whole point of this module is that the
    proposal is visible and correctable BEFORE it is acted on. This mirrors
    :mod:`spacr.foreign`, whose column mapping is the middle of its screen and
    is editable in place -- the same split, applied to images.

    Nothing here touches the destination. :meth:`problems` says what would
    stop an import; :meth:`with_mapping` returns a NEW plan, resolved in
    memory, so editing is instant and a rejected edit costs nothing.

    :param layout: what the names gave.
    :param inside: per relative path, what the file's own metadata gave.
    :param mapping: user answers for axes inference could not name, as
        ``{token position: {value: index}}`` -- e.g. ``{0: {"DAPI": 1,
        "GFP": 2}}`` for a tree with dye folders.
    """

    layout: InferredLayout
    inside: Dict[str, InsideFile] = field(default_factory=dict)
    mapping: Dict[int, Dict[str, int]] = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return self.layout.root

    @property
    def files(self) -> Dict[str, Dict[str, object]]:
        """Relative path -> every axis known, from all three sources.

        Names first, then the file's own metadata, then the user's mapping --
        in that order because each is more specific than the last, and the
        user is the only one who can be right about the last of them.
        """
        merged: Dict[str, Dict[str, object]] = {}
        for rel, found in self.layout.per_file.items():
            entry = dict(found)
            inside = self.inside.get(rel)
            if inside is not None:
                for axis, size in inside.sizes.items():
                    # A COUNT, not a position: the file holds `size` planes of
                    # this axis, which is a different fact from "this file is
                    # plane 3" and must not be written as one.
                    entry[f"{axis}_count"] = size
            for position, answers in self.mapping.items():
                value = self._token_at(rel, position)
                if value is not None and value in answers:
                    entry["channel"] = answers[value]
            merged[rel] = entry
        return merged

    def _token_at(self, rel: str, position: int) -> Optional[str]:
        toks = tokenise(str(Path(rel).with_suffix("")))
        return toks[position][1] if position < len(toks) else None

    @property
    def unmapped(self) -> Dict[int, List[str]]:
        """Axes still waiting on an answer, after the mapping is applied."""
        return {position: values
                for position, values in self.layout.unplaced.items()
                if not set(values) <= set(self.mapping.get(position, {}))}

    def problems(self) -> List[str]:
        """Everything that would make this import wrong, in plain sentences.

        REPORTED, NOT RAISED, and all of them at once. The first version of
        the translation audit raised on its first finding and reported one
        problem where there were three; a plan that stops at the first
        complaint makes the user fix things one round-trip at a time.
        """
        issues = []
        if not self.files:
            issues.append("No images could be read from this folder.")
        for position, values in self.unmapped.items():
            issues.append(
                f"Token {position} varies across the folder "
                f"({', '.join(sorted(set(values))[:4])}) and nothing says "
                f"what it means. Map it, or those images cannot be told apart.")
        ambiguous = [rel for rel, i in self.inside.items() if i.is_ambiguous]
        if ambiguous:
            issues.append(
                f"{len(ambiguous)} file(s) hold several pages that are not "
                f"labelled Z, T or channel, e.g. {ambiguous[0]}. Say which "
                f"they are, or each file is treated as a single plane.")
        unreadable = [rel for rel, i in self.inside.items() if i.pages == 0]
        if unreadable:
            issues.append(
                f"{len(unreadable)} file(s) could not be opened, e.g. "
                f"{unreadable[0]}.")
        if self.layout.skipped:
            issues.append(
                f"{len(self.layout.skipped)} file(s) are named unlike the "
                f"rest and were not interpreted, e.g. {self.layout.skipped[0]}.")
        return issues

    def counts(self) -> Dict[str, int]:
        """Distinct values per axis -- the numbers a user checks first.

        A plate with the wrong number of wells or channels is visible here
        and nowhere else until the run has finished.
        """
        out: Dict[str, int] = {}
        for entry in self.files.values():
            for axis, value in entry.items():
                out.setdefault(axis, set()).add(value)  # type: ignore[arg-type]
        return {axis: len(values) for axis, values in out.items()}  # type: ignore[arg-type]

    def with_mapping(self, mapping: Dict[int, Dict[str, int]]) -> "ImportPlan":
        """A new plan with ``mapping`` merged in. Nothing on disk is touched."""
        merged = {position: dict(answers)
                  for position, answers in self.mapping.items()}
        for position, answers in mapping.items():
            merged.setdefault(position, {}).update(answers)
        return ImportPlan(layout=self.layout, inside=self.inside,
                          mapping=merged)

    def table(self, limit: int = 8) -> str:
        """The proposal, as the table a user reads before pressing anything.

        Example filenames BESIDE the fields they were parsed into, because a
        wrong guess is only visible next to the name it came from -- reading
        a column of numbers cannot tell you the field and the channel were
        swapped.
        """
        entries = self.files
        axes = [a for a in AXES if any(a in e for e in entries.values())]
        extra = sorted({k for e in entries.values() for k in e
                        if k.endswith("_count")})
        header = ["file"] + axes + extra
        widths = [max(len(h), 12) for h in header]
        rows = []
        for rel in sorted(entries)[:limit]:
            entry = entries[rel]
            cells = [rel] + [str(entry.get(a, "")) for a in axes + extra]
            widths = [max(w, len(c)) for w, c in zip(widths, cells)]
            rows.append(cells)
        lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths)),
                 "  ".join("-" * w for w in widths)]
        lines += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
        if len(entries) > limit:
            lines.append(f"... and {len(entries) - limit} more")
        counts = self.counts()
        lines.append("")
        lines.append("  ".join(f"{a}={counts[a]}" for a in axes if a in counts))
        for issue in self.problems():
            lines.append(f"  ! {issue}")
        return "\n".join(lines)


def plan_import(root, *, sample: int = 400,
                mapping: Optional[Dict[int, Dict[str, int]]] = None,
                read_files: bool = True) -> ImportPlan:
    """Propose an import of ``root``. Writes nothing.

    :param root: the folder of images.
    :param sample: how many files to inspect; see :func:`infer_layout`.
    :param mapping: answers for axes the names cannot name.
    :param read_files: also open each file for its axis metadata. On by
        default because a channel that lives inside a file is invisible
        without it; turn it off for a fast first look at a large archive.
    """
    layout = infer_layout(root, sample=sample)
    inside: Dict[str, InsideFile] = {}
    if read_files:
        for rel in layout.per_file:
            inside[rel] = read_axes_inside(Path(root) / rel)
    return ImportPlan(layout=layout, inside=inside, mapping=dict(mapping or {}))
