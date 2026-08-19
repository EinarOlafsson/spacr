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


def reroot_column(frame, column: str, src_root: Optional[str]):
    """Re-root one path column of ``frame`` IN THE FRAME, never on disk.

    :param frame: any DataFrame; a missing column is not an error, because the
        PNG route and the merged route carry different ones and a caller
        should be able to ask for both.
    :param src_root: anything :func:`candidate_roots` accepts.
    :returns: how many values were rewritten.

    Resolves the first dead path against the filesystem, then applies the
    prefix that worked to the rest -- 60,816 rows cost one search plus one
    string replacement each, rather than 60,816 searches. Any row the prefix
    does not fix is still resolved on its own, so a frame holding crops from
    two different screens is not half-abandoned.
    """
    if frame is None or column not in getattr(frame, "columns", ()):
        return 0
    values = frame[column].tolist()
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
            out.append(value)
            continue
        mapped, found = _reroot_with_prefix(value, src_root)
        if found is not None:
            prefix = found
        elif mapped == value:
            unresolvable.add(folder)
        if mapped != value:
            moved += 1
        out.append(mapped)
    if moved:
        frame[column] = out
    return moved


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
