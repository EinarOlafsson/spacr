"""Report missing columns and offer the columns that are available.

Without this check, a misnamed `dependent_variable` passes the early checks
and fails inside the merge, after the whole score table has been read, with a
message that names the missing column but not the available ones.

Three rules apply:

1.  Only the header row is read (`nrows=0`). The check runs before the run,
    sometimes to populate a GUI dropdown on the GUI thread, and a score CSV
    can be hundreds of megabytes.

2.  A missing file is reported separately from a missing column. A missing
    file has no header to compare against; a missing column is reported with
    the columns that do exist.

3.  Suggestions are given in addition to the full column list. Near-misses
    are listed first because `predictions` for `prediction` is a common typo,
    and the full list is always included as well.
"""

from __future__ import annotations

import difflib
import os
from typing import Dict, Iterable, List, Optional, Sequence, Union

#: Near-misses offered before the full list. Three: enough for the plural
#: typo, the singular typo and the case slip, few enough that the sentence
#: still reads as a suggestion rather than as a second list.
MAX_SUGGESTIONS = 3

#: How close a name has to be to be suggested at all. difflib's ratio; 0.6 is
#: its own default for `get_close_matches` and it puts `prediction` next to
#: `predictions` without putting `plate` next to `parasite`.
CUTOFF = 0.6


class ColumnNotFound(KeyError):
    """Raised when a named column is absent; the message lists those present.

    :param message: human-readable explanation returned for the exception.
    :param name: the column that was asked for, kept so a caller can recover
        it without parsing ``message``.
    :param available: the columns the file does have, in file order. Not
        sorted, because file order is what a user reading the header sees,
        and sorting hides that related columns are grouped together.

    A KeyError subclass so existing `except KeyError` paths still catch it,
    and so `settings['x']` failures and this one read the same way to a
    caller that does not care which it was.
    """

    def __init__(self, message: str, *, name: str = "",
                 available: Sequence[str] = ()):
        """Initialize a readable error with structured column context."""
        super().__init__(message)
        self.message = message
        #: What was asked for.
        self.name = name
        #: What the file has, in file order -- the order a user reading the
        #: header sees, not sorted, because sorting hides that the score
        #: columns are grouped together.
        self.available = list(available)

    def __str__(self) -> str:
        """Return the message without :class:`KeyError`'s added quoting."""
        return self.message


def _as_paths(paths) -> List[str]:
    """One path, several, or None, as a list of expanded paths."""
    if paths is None:
        return []
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    return [os.path.expanduser(os.path.expandvars(os.fspath(one)))
            for one in paths if one]


def headers(paths) -> Dict[str, List[str]]:
    """``{path: [column, ...]}`` for every readable CSV in ``paths``.

    :param paths: one path, a sequence of them, or None.
    :returns: a dict in the order given. A path that does not exist or cannot
        be parsed is absent from the result rather than mapped to an empty
        list -- see rule 2. Use :func:`missing` to ask which those were.
    """
    import pandas as pd

    found: Dict[str, List[str]] = {}
    for path in _as_paths(paths):
        if not os.path.isfile(path):
            continue
        try:
            found[path] = list(pd.read_csv(path, nrows=0).columns)
        except Exception:                           # noqa: BLE001
            continue
    return found


def missing(paths) -> List[str]:
    """Return the paths whose header could not be read by :func:`headers`.

    :param paths: one CSV path, a sequence of paths, or None.
    """
    readable = headers(paths)
    return [path for path in _as_paths(paths) if path not in readable]


def available(paths) -> List[str]:
    """Every column across ``paths``, in order, without duplicates.

    :param paths: one CSV path, a sequence of paths, or None.

    Across, not per file: `score_data` is routinely a list of one CSV per
    plate with identical headers, and a user choosing a column does not care
    which plate it came from.
    """
    seen: List[str] = []
    for columns in headers(paths).values():
        for column in columns:
            if column not in seen:
                seen.append(column)
    return seen


def suggest(name, columns: Iterable[str]) -> List[str]:
    """Column names close to ``name``, best first.

    :param name: missing column name for which to find near-matches.
    :param columns: available column names to search.

    Matching is case-insensitive, so `Prediction` is suggested for
    `prediction`.
    """
    columns = list(columns)
    if not name:
        return []
    lowered = {column.lower(): column for column in columns}
    close = difflib.get_close_matches(str(name).lower(), list(lowered),
                                      n=MAX_SUGGESTIONS, cutoff=CUTOFF)
    return [lowered[one] for one in close]


def describe(name, paths, *, what: str = "column",
             setting: str = "") -> str:
    """Return the message to print or raise when column ``name`` is missing.

    :param name: missing column name to explain.
    :param paths: CSV paths whose headers should be reported.

    Names the setting, the files that were read, the near-misses and then
    every column.
    """
    columns = available(paths)
    unreadable = missing(paths)
    where = ", ".join(os.path.basename(p) for p in headers(paths)) or "no file"
    label = f"{setting}={name!r}" if setting else repr(name)

    if not columns:
        if unreadable:
            return (f"{label} could not be checked: none of "
                    f"{', '.join(os.path.basename(p) for p in unreadable)} "
                    f"could be read as a CSV.")
        return f"{label} could not be checked: no input CSV was given."

    lines = [f"no {what} {label} in {where}."]
    close = suggest(name, columns)
    if close:
        lines.append(f"Did you mean {', '.join(repr(c) for c in close)}?")
    lines.append(f"The {len(columns)} column(s) available: "
                 f"{', '.join(repr(c) for c in columns)}.")
    if unreadable:
        lines.append(f"Not read: "
                     f"{', '.join(os.path.basename(p) for p in unreadable)}.")
    return " ".join(lines)


def resolve(name, paths, *, what: str = "column", setting: str = "") -> str:
    """``name`` if the CSVs have it, else raise with what they do have.

    :param name: the column asked for.
    :param paths: the CSVs to look in.
    :param what: what kind of column, for the message ("response column",
        "count column").
    :param setting: the settings key this came from, so the message names the
        control the user has to change.
    :raises ColumnNotFound: carrying `.available`, so a GUI can offer the
        list rather than re-reading the files to build it.

    A case-insensitive match is accepted and returned in the file's spelling:
    `Predictions` resolves to `predictions` when the file uses that name.
    """
    columns = available(paths)
    if name in columns:
        return name
    lowered = {column.lower(): column for column in columns}
    if name is not None and str(name).lower() in lowered:
        return lowered[str(name).lower()]
    raise ColumnNotFound(describe(name, paths, what=what, setting=setting),
                         name=str(name) if name is not None else "",
                         available=columns)


__all__ = ["CUTOFF", "ColumnNotFound", "MAX_SUGGESTIONS", "available",
           "describe", "headers", "missing", "resolve", "suggest"]
