"""Preview filename parsing and destination paths before importing images.

:func:`plan` evaluates filenames, a regular expression, and group roles
without opening, copying, or moving files. Its :class:`ImportPlan` reports
every matched rename, every unmatched filename, and any role problem. Output
stems use :func:`spacr.io._escaped_field_stem`, so the preview matches the
paths produced by the import pipeline.
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
    """One file, and what the import would call it.

    :ivar before: basename supplied to the import preview.
    :ivar after: TIFF filename that the import would write.
    :ivar plate: parsed or fallback plate identifier.
    :ivar well: captured well identifier.
    :ivar field: captured field-of-view identifier.
    :ivar channel: captured imaging-channel identifier.
    :ivar time: captured timepoint identifier, or an empty string for a
        non-timelapse filename.
    """

    before: str
    after: str
    plate: str
    well: str
    field: str
    channel: str
    time: str = ""


@dataclass
class ImportPlan:
    """Describe matched renames, unmatched files, and role problems.

    :param renamed: one entry per matched file, in input order.
    :param unmatched: filenames that did not match the pattern.
    :param trouble: explanations that make the plan incomplete or unusable.
        Valid partial results remain available when problems are present.
    """

    renamed: Tuple[Renamed, ...] = ()
    unmatched: Tuple[str, ...] = ()
    trouble: Tuple[str, ...] = ()

    @property
    def n_matched(self) -> int:
        """How many files the naming pattern resolved.

        :returns: the matched count.
        """
        return len(self.renamed)

    @property
    def n_files(self) -> int:
        """Every file the plan covers, matched or not.

        THE DENOMINATOR. A pattern matching 900 files means nothing until you
        know whether there were 900 or 9,000, and this is the number that
        makes the first one readable.

        :returns: the total count.
        """
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

    :param regex: filename regular expression to inspect.

    Returns an empty tuple while the pattern is incomplete or invalid.
    """
    try:
        compiled = re.compile(str(regex or ""))
    except re.error:
        return ()
    return tuple(sorted(compiled.groupindex, key=compiled.groupindex.get))


def role_trouble(roles: Mapping[str, str]) -> Tuple[str, ...]:
    """Return user-facing problems in a regex-group role assignment.

    :param roles: named capture groups mapped to their selected import roles.

    Duplicate roles and missing required roles are reported before import so
    no captured group is silently ignored.
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
    :param roles: ``{group name: role}``, overriding the captured group name.
        A group whose name is already a role needs no entry.
    :param plate: the plate to use when no group supplies one -- the import
        falls back to the source folder's name, so the preview does too.
    :param timelapse: pass the timepoint through to the stem.
    :returns: an :class:`ImportPlan`. Invalid patterns and missing roles are
        reported in :attr:`ImportPlan.trouble` rather than raised.
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


#: A trailing `\.ext`, `\.(ext|ext)` or `\.(?:ext|ext)`. Anchored at the
#: end, so a pattern that merely CONTAINS one keeps it.
#:
#: THE BACKSLASH IS OPTIONAL. A pattern ending in a bare `.tif` -- the dot as
#: a wildcard -- is just as broken once `_get_regex` appends its own: the
#: composed `(....tif)..tif` would need a name reading "…tifxtif".
_TRAILING_EXTENSION = re.compile(
    r"\\?\.(?:\((?:\?:)?[A-Za-z0-9]+(?:\|[A-Za-z0-9]+)*\)|[A-Za-z0-9]+)$")


def for_get_regex(pattern: str) -> str:
    r"""Prepare a filename pattern for ``utils._get_regex``.

    ``_get_regex`` appends its own image-extension pattern. This helper
    removes a trailing extension match and end anchor to prevent an inferred
    pattern such as ``\.(?:tif|tiff|png|jpg|jpeg)$`` from receiving a second,
    unreachable suffix.

    Parameters
    ----------
    pattern : str
        Pattern stored by the import editor.

    Returns
    -------
    str
        The pattern without a trailing supported-image extension or end
        anchor. Patterns without either are returned unchanged.
    """
    text = str(pattern or "").strip()
    if text.endswith("$"):
        text = text[:-1]
    # ANY ALTERNATION OF IMAGE EXTENSIONS, not one exact spelling of it.
    # `auto_detect_regex` returns the full five -- `(?:tif|tiff|png|jpg|jpeg)`
    # -- while the bundled YOKOGAWA pattern carries `(?:tif|tiff)`, and a
    # literal comparison against one of them silently left the other on.
    match = _TRAILING_EXTENSION.search(text)
    if match:
        return text[:match.start()]
    return text
