"""What an import WOULD do, before it does any of it (137 A, C and D).

    "the user can hit test to show how file names would be changed and how
     files would be organized into a spacr structure."

TEST SHOWS, IT DOES NOT WRITE. `io._run_test_mode` already answers this
question by COPYING real files into a `test` folder -- which is the same
question asked after the commitment rather than before it. This module
answers it from the filenames alone: nothing is opened, nothing is copied,
and the answer is a value the GUI renders and a test can assert on.

    plan(filenames, regex, roles) -> ImportPlan

A FILE THAT DOES NOT MATCH IS NAMED. "412 of 480 files matched" with the
other 68 listed is an answer; 412 files appearing without comment is how half
a plate goes missing, and it is the fault this whole instruction exists to
stop.

THE NEW NAME IS `io`'S OWN, not a second opinion about it: the stem comes
from :func:`spacr.io._escaped_field_stem`, so what this previews is what the
import writes, down to the plate escaping that `schema.parse_field_stem`
reads back.
"""
from __future__ import annotations

import os
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: The roles a regex group may be given, and what each one means downstream.
#: A CLOSED SET, because these names are not decoration:
#: `_rename_and_organize_image_files` and `_move_to_chan_folder` read
#: `plateID` / `wellID` / `fieldID` / `chanID` BY NAME, so a typo in a
#: hand-written group name is a silent import of nothing.
ROLES: Tuple[Tuple[str, str], ...] = (
    ("plateID", "the plate this file belongs to"),
    ("wellID", "the well — a letter and one or two digits, or r##c##"),
    ("fieldID", "the field of view within the well"),
    ("chanID", "which channel was imaged"),
    ("timeID", "the timepoint, for a timelapse"),
    ("sliceID", "the z slice"),
    ("", "ignore this group"),
)

#: The roles a file cannot be organised without.
REQUIRED: Tuple[str, ...] = ("wellID", "fieldID", "chanID")

#: What a channel number may be called on the screen. The IMPORT still writes
#: `chanID`; this is the vocabulary the user picks a MEANING from -- "channel
#: 1 channel 2 channel 3 and channel 4 and cell and nuclei".
CHANNEL_MEANINGS: Tuple[str, ...] = (
    "channel 1", "channel 2", "channel 3", "channel 4",
    "cell", "nucleus", "pathogen", "cytoplasm",
)


@dataclass(frozen=True)
class Renamed:
    """One file, and what the import would call it."""

    before: str
    after: str
    plate: str
    well: str
    field: str
    channel: str
    time: str = ""


@dataclass
class ImportPlan:
    """The whole answer: what matches, what it becomes, and what does not.

    :param renamed: one entry per matched file, in the order dropped.
    :param unmatched: the files the regex did not match, BY NAME.
    :param trouble: why the plan is not usable, or ``()``. A plan with
        trouble still carries whatever it did work out, because "these 12
        matched and the role for group 2 is missing" is more useful than an
        empty screen.
    """

    renamed: Tuple[Renamed, ...] = ()
    unmatched: Tuple[str, ...] = ()
    trouble: Tuple[str, ...] = ()

    @property
    def n_matched(self) -> int:
        return len(self.renamed)

    @property
    def n_files(self) -> int:
        return len(self.renamed) + len(self.unmatched)

    def summary(self) -> str:
        """The one line the panel leads with."""
        if not self.n_files:
            return "No files yet. Drop images or a folder here."
        said = f"{self.n_matched} of {self.n_files} file(s) matched"
        if self.unmatched:
            said += f"; {len(self.unmatched)} did not"
        return said + "."

    def tree(self) -> "OrderedDict[str, Any]":
        """The spaCR structure this would produce, with counts at each level.

        ``{plate: {well: {field: Counter(channel -> n)}}}``, insertion
        ordered so the tree reads in the order the files arrived rather than
        alphabetically -- which is what makes a missing well visible.
        """
        out: "OrderedDict[str, Any]" = OrderedDict()
        for row in self.renamed:
            plate = out.setdefault(row.plate, OrderedDict())
            well = plate.setdefault(row.well, OrderedDict())
            field_counts = well.setdefault(row.field, Counter())
            field_counts[row.channel] += 1
        return out

    def tree_lines(self) -> Tuple[str, ...]:
        """The tree as indented text, with the counts written in."""
        lines: List[str] = []
        tree = self.tree()
        for plate, wells in tree.items():
            n_fields = sum(len(f) for f in wells.values())
            lines.append(f"{plate}/    {len(wells)} well(s), "
                         f"{n_fields} field(s)")
            for well, fields in wells.items():
                channels = Counter()
                for counts in fields.values():
                    channels.update(counts)
                lines.append(
                    f"    {well}/    {len(fields)} field(s), "
                    f"channel(s) {', '.join(sorted(channels))}")
                for name, counts in fields.items():
                    lines.append(
                        f"        {name}/    "
                        + ", ".join(f"{c} x{n}" for c, n in sorted(counts.items())))
        return tuple(lines)


def group_names(regex: str) -> Tuple[str, ...]:
    """The named groups in ``regex``, in the order they appear.

    Empty for a pattern that does not compile -- the caller is editing it
    live and a half-typed regex is not an error to shout about.
    """
    try:
        compiled = re.compile(str(regex or ""))
    except re.error:
        return ()
    return tuple(sorted(compiled.groupindex, key=compiled.groupindex.get))


def role_trouble(roles: Mapping[str, str]) -> Tuple[str, ...]:
    """What is wrong with the role assignment, in words.

    TWO COLUMNS CLAIMING THE SAME ROLE IS AN ERROR STATED ON THE SPOT, not at
    run time -- which is where it would otherwise appear, as an import that
    read one group and silently ignored the other.
    """
    said: List[str] = []
    taken: Dict[str, List[str]] = {}
    for group, role in (roles or {}).items():
        if role:
            taken.setdefault(str(role), []).append(str(group))
    for role, groups in taken.items():
        if len(groups) > 1:
            said.append(f"{', '.join(sorted(groups))} are all set to "
                        f"{role!r}; a role belongs to one group")
    missing = [r for r in REQUIRED if r not in taken]
    if missing:
        said.append(f"no group is the {', '.join(missing)}; spaCR cannot "
                    f"organise a file without {'them' if len(missing) > 1 else 'it'}")
    return tuple(said)


def plan(filenames: Sequence[str], regex: str,
         roles: Optional[Mapping[str, str]] = None,
         *, plate: str = "", timelapse: bool = False) -> ImportPlan:
    """What the import would do to ``filenames`` under ``regex``.

    :param filenames: bare names, as dropped. Paths are reduced to their
        basename, because that is what the regex is matched against.
    :param regex: the pattern, with named groups.
    :param roles: ``{group name: role}``, overriding what the group is
        called. A group whose name IS a role needs no entry.
    :param plate: the plate to use when no group supplies one -- the import
        falls back to the source folder's name, so the preview does too.
    :param timelapse: pass the timepoint through to the stem.
    :returns: an :class:`ImportPlan`. NEVER raises for a bad regex or a
        missing role: both are things a user is in the middle of fixing, and
        a traceback is not a preview.
    """
    from .io import _escaped_field_stem

    roles = dict(roles or {})
    out: List[Renamed] = []
    missed: List[str] = []
    trouble: List[str] = []

    try:
        compiled = re.compile(str(regex or ""))
    except re.error as error:
        return ImportPlan(trouble=(f"that is not a regex yet: {error}",),
                          unmatched=tuple(os.path.basename(str(f))
                                          for f in filenames or ()))
    if not str(regex or ""):
        return ImportPlan(unmatched=tuple(os.path.basename(str(f))
                                          for f in filenames or ()))

    trouble.extend(role_trouble(
        {g: roles.get(g, g if g in dict(ROLES) else "")
         for g in group_names(regex)}))

    for raw in filenames or ():
        name = os.path.basename(str(raw))
        match = compiled.search(name)
        if match is None:
            missed.append(name)
            continue
        got = match.groupdict()
        # THE ROLE WINS OVER THE GROUP NAME, because the dropdown is what the
        # user actually said and the group name may be `g1`.
        values: Dict[str, str] = {}
        for group, value in got.items():
            role = str(roles.get(group, group))
            if role:
                values[role] = str(value or "")
        this_plate = values.get("plateID") or str(plate or "") or "plate1"
        out.append(Renamed(
            before=name,
            after=_escaped_field_stem(
                this_plate, values.get("wellID", ""),
                values.get("fieldID", ""), values.get("timeID", "")) + ".tif",
            plate=this_plate,
            well=values.get("wellID", ""),
            field=values.get("fieldID", ""),
            channel=values.get("chanID", ""),
            time=values.get("timeID", "")))
    return ImportPlan(tuple(out), tuple(missed), tuple(trouble))


#: Extensions `spacr.utils._get_regex` appends to a custom pattern itself.
APPENDED_SUFFIXES: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg",
                                      ".jpeg")


def for_get_regex(pattern: str) -> str:
    """A pattern trimmed for `_get_regex`'s ``'custom'`` branch.

    THIS IS NOT TIDINESS AND IT IS NOT NEW BEHAVIOUR; it is a bug fix with a
    number. `_get_regex` builds the custom pattern as::

        f"({custom_regex}).{img_format}"

    -- it appends the extension ITSELF. So a pattern that already ends in
    ``\.tif``, or in the alternation-plus-anchor
    ``\.(?:tif|tiff|png|jpg|jpeg)$`` that `auto_detect_regex` returns,
    becomes ``(...$)..tif``: an anchor with characters after it, which can
    never match anything.

    MEASURED on eight cellvoyager filenames through the real path -- drop,
    auto-detect, save, `_get_regex` -- 0 of 8 matched. The import then runs,
    finds no files, and reports only that it found none: there is no error
    anywhere, and the pattern in the box looks exactly right.

    :param pattern: what the editor holds.
    :returns: the same pattern with any trailing extension match and anchor
        removed, and unchanged when it has none.
    """
    text = str(pattern or "").strip()
    if text.endswith("$"):
        text = text[:-1]
    alternation = "|".join(s.lstrip(".") for s in APPENDED_SUFFIXES)
    for tail in (rf"\.(?:{alternation})", rf"\.({alternation})"):
        if text.endswith(tail):
            return text[:-len(tail)]
    for suffix in APPENDED_SUFFIXES:
        for spelling in ("\\" + suffix, suffix):
            if text.endswith(spelling):
                return text[:-len(spelling)]
    return text
