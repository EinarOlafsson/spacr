"""A crop path recorded on one computer, resolved on another.

``png_list.png_path`` is written ABSOLUTE at crop time. Move the screen to
another machine -- or mount the same NAS somewhere else -- and every one of
those paths is dead while the files themselves are all present, because what
the user preserved is the STRUCTURE BELOW THE SCREEN, not the prefix above it:

    <recorded root>/plate1/data/single_nucleus/single_pathogen/plate1_H19/...
    <current  root>/plate1/data/single_nucleus/single_pathogen/plate1_H19/...

Measured on the TSG101 screen: 0 of 60,816 recorded paths existed, and
60,816 of 60,816 existed once the structure below the plate was rebuilt under
the folder the database was opened from.

THE DATABASE IS NEVER WRITTEN TO. Resolution happens on the frame a reader
already holds, every time it is read. That is the maintainer's own design and
the reason it is safe to apply everywhere: a screen can be copied to a third
machine, read there, and copied back, and the recorded paths are still
whatever they were.

THE RULE
--------
Find the DEEPEST SUFFIX of the recorded path that exists under the current
root, and use it. Not a fixed split on ``data/``: the root a caller has may
be the plate folder, the screen folder above it, the ``measurements/`` folder
the database sits in, or the database file itself, and one rule covers all of
them rather than four special cases that each work somewhere else.

AND ONLY WHAT LANDS ON A FILE THAT EXISTS. A path rewritten to somewhere
equally absent is strictly worse than the original, because the error then
names a folder the user never chose. Nothing here changes a path it cannot
first confirm.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: The folder exported crops live under. Only used to PREFER a suffix that
#: starts at it, never to require one.
DATA_FOLDER = "data"

#: Folder names that sit BESIDE ``data/`` rather than above it, so a root
#: pointing at one of them means the screen is one level up.
_SIBLINGS: Tuple[str, ...] = ("measurements", "results", "settings", "merged",
                              "orig", "stack")

#: How far above a given root to look for the screen. Two is enough for
#: ``<plate>/measurements/measurements.db``; more would start matching
#: unrelated folders that happen to share a name.
_MAX_CLIMB = 2


def _parts(path: str) -> List[str]:
    """``path`` split into components, separator-agnostic."""
    return [p for p in str(path).replace("\\", "/").split("/") if p]


def candidate_roots(root: Optional[str]) -> Tuple[str, ...]:
    """Every folder ``root`` could mean, nearest first.

    Accepts the plate folder, the screen folder, the ``measurements/`` folder,
    or the ``measurements.db`` file itself -- callers hold different ones and
    should not each have to normalise.
    """
    if not root:
        return ()
    here = os.path.abspath(os.path.expanduser(os.fspath(root)))
    if os.path.isfile(here):
        here = os.path.dirname(here)
    out: List[str] = []
    for _ in range(_MAX_CLIMB + 1):
        if here and here not in out:
            out.append(here)
        parent = os.path.dirname(here)
        if parent == here:
            break
        # Climb unconditionally for the first step when the folder is a known
        # sibling of `data/`; otherwise still climb, because a caller may hand
        # us a screen root whose plates are one level down.
        here = parent
    return tuple(out)


def _suffixes(path: str) -> List[Tuple[str, ...]]:
    """Suffixes of ``path``, deepest structure first.

    A suffix beginning at a ``data`` component is tried before the others of
    the same length, because that is the layout ``measure`` writes and the one
    a match is most likely to be meaningful in.
    """
    parts = _parts(path)
    if len(parts) < 2:
        return []
    ordered: List[Tuple[str, ...]] = []
    # Longest first: the more of the recorded structure that matches, the
    # less chance the match is a coincidence.
    for start in range(len(parts) - 1):
        ordered.append(tuple(parts[start:]))
    ordered.sort(key=lambda s: (0 if s[0] == DATA_FOLDER else 1, -len(s)))
    return ordered


def reroot_crop_path(path: Optional[str], src_root: Optional[str]
                     ) -> Optional[str]:
    """``path`` as it exists under ``src_root``, or ``path`` unchanged.

    :param path: the recorded absolute path, or anything falsy.
    :param src_root: the plate folder, the screen folder, the ``measurements``
        folder, or the database file. All four resolve.
    :returns: a path that EXISTS when one could be built; otherwise the input
        untouched, so a caller's error still names what was recorded.
    """
    mapped, _prefixes = _reroot_with_prefix(path, src_root)
    return mapped


def _reroot_with_prefix(path: Optional[str], src_root: Optional[str]
                        ) -> Tuple[Optional[str], Optional[Tuple[str, str]]]:
    """:func:`reroot_crop_path`, also returning the (old, new) prefix used.

    The prefix is what makes a 60,000-row frame cheap: it is discovered once
    and then applied as a string replacement, instead of asking the filesystem
    about every row.
    """
    if not path or not isinstance(path, str) or not path.strip():
        return path, None
    if os.path.exists(path):
        return path, None
    roots = candidate_roots(src_root)
    if not roots:
        return path, None
    normalised = str(path).replace("\\", "/")
    for suffix in _suffixes(path):
        tail = "/".join(suffix)
        if not normalised.endswith(tail):
            continue
        head = normalised[: len(normalised) - len(tail)]
        for root in roots:
            candidate = os.path.join(root, *suffix)
            if os.path.exists(candidate):
                new_head = candidate[: len(candidate) - len(os.path.join(*suffix))]
                return candidate, (head, new_head)
    return path, None


@dataclass(frozen=True)
class RerootReport:
    """What one re-rooting pass did, INCLUDING what it could not do.

    The last two fields are the reason this is a record and not a bare count.
    A path with no recognisable structure under the root is returned unchanged
    and fails later as a missing file, somewhere with less context -- which is
    how a silent pass-through stays invisible. Instruction 155 F: "Count them
    and name one".
    """

    column: str = ""
    moved: int = 0
    unresolved: int = 0
    first_unresolved: str = ""
    root: str = ""

    def __bool__(self) -> bool:
        return bool(self.moved)

    def __int__(self) -> int:
        return int(self.moved)

    @property
    def partial(self) -> bool:
        """Some of this column resolved and some did not.

        THE DISTINCTION THAT DECIDES WHETHER TO SHOUT. A column where nothing
        resolved and nothing existed is a ROUTE THAT IS NOT ON THIS MACHINE --
        a screen with PNG crops and no ``merged/`` folder has 60,816 unplaceable
        ``path_name`` values and is completely healthy. A column where most
        resolved and a few did not is the real signal instruction 155 F asks
        to be counted and named.
        """
        return bool(self.moved) and bool(self.unresolved)

    @property
    def absent(self) -> bool:
        """Nothing in this column could be placed, and nothing already was."""
        return not self.moved and bool(self.unresolved)

    def describe(self) -> str:
        """One line for a caller to print, or "" when there is nothing to say."""
        parts: List[str] = []
        if self.moved:
            parts.append(f"re-rooted {self.moved:,} {self.column} value(s) "
                         f"under {self.root}")
        if self.absent:
            return (f"none of the {self.unresolved:,} {self.column} value(s) "
                    f"are under {self.root} — that route's files are not on "
                    f"this machine")
        if self.unresolved:
            parts.append(
                f"{self.unresolved:,} could not be placed under {self.root}; "
                f"the first is {self.first_unresolved}")
        return "; ".join(parts)


def reroot_column(frame, column: str, src_root: Optional[str]):
    """Re-root one path column of ``frame`` IN THE FRAME, never on disk.

    :param frame: any DataFrame; a missing column is not an error, because the
        PNG route and the merged route carry different ones and a caller
        should be able to ask for both.
    :param src_root: anything :func:`candidate_roots` accepts.
    :returns: a :class:`RerootReport`, which counts as its own ``moved`` in a
        boolean or integer context, so a caller that only wants the number
        still gets it.

    Resolves the first dead path against the filesystem, then applies the
    prefix that worked to the rest -- 60,816 rows cost one search plus one
    string replacement each, rather than 60,816 searches. Any row the prefix
    does not fix is still resolved on its own, so a frame holding crops from
    two different screens is not half-abandoned.
    """
    root_label = (candidate_roots(src_root) or ("",))[0]
    if frame is None or column not in getattr(frame, "columns", ()):
        return RerootReport(column=column, root=root_label)
    values = frame[column].tolist()
    unresolved = 0
    first_unresolved = ""
    prefix: Optional[Tuple[str, str]] = None
    # Folders already searched and NOT found. Every crop of a well shares a
    # folder, so without this a root that resolves nothing costs one full
    # search per ROW -- measured at 8.2s over 60,816 rows against 0.6s when a
    # prefix is found. With it, the same case costs one search per folder.
    unresolvable: set = set()
    out: List[object] = []
    moved = 0
    for value in values:
        if not isinstance(value, str) or not value.strip():
            out.append(value)
            continue
        if prefix is not None:
            was, now = prefix
            forward = value.replace("\\", "/")
            if forward.startswith(was):
                candidate = now + forward[len(was):]
                if os.path.exists(candidate):
                    out.append(candidate)
                    moved += 1
                    continue
        folder = os.path.dirname(value.replace("\\", "/"))
        if folder in unresolvable:
            unresolved += 1
            first_unresolved = first_unresolved or value
            out.append(value)
            continue
        mapped, found = _reroot_with_prefix(value, src_root)
        if found is not None:
            prefix = found
        elif mapped == value and not os.path.exists(value):
            unresolvable.add(folder)
        if mapped != value:
            moved += 1
        elif not os.path.exists(value):
            unresolved += 1
            first_unresolved = first_unresolved or value
        out.append(mapped)
    if moved:
        frame[column] = out
    return RerootReport(column=column, moved=moved, unresolved=unresolved,
                        first_unresolved=first_unresolved, root=root_label)


def source_root_for_database(db_path: str) -> str:
    """The plate folder a ``measurements.db`` belongs to.

    ``<plate>/measurements/measurements.db`` -> ``<plate>``, which is the
    folder that holds ``data/``. Derived rather than passed so a reader gains
    portability without new plumbing through every caller;
    :func:`candidate_roots` then covers the cases where the layout differs.
    """
    if not db_path:
        return ""
    return os.path.dirname(os.path.dirname(os.path.abspath(db_path)))
