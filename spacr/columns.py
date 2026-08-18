"""A column that is not there offers the columns that are.

Instruction 135, asked for on 2026-08-17: "if dependent variable is not found
in the score table then present the user with the columns in the score csvs so
the user can choose which column ... also if the count columns ar not found id
like similar behaviour".

THE FAILURE THIS REPLACES. A misnamed `dependent_variable` survives every
early check and dies inside the merge, after the whole score table has been
read, with a message naming a column the file does not have and saying nothing
about what it does have. The user then opens the CSV by hand to read its
header. On a large screen that is minutes of reading to answer a question the
header row answers instantly.

THREE RULES, and each is a different failure this must not blur:

1.  THE HEADER ROW ONLY. `nrows=0`. This runs before the run, sometimes to
    populate a GUI dropdown on the GUI thread, and a score CSV is hundreds of
    megabytes.

2.  A MISSING FILE IS NOT A MISSING COLUMN. The first cannot answer the
    question; the second answers it with "not that name, one of these". A
    caller told "column not found" about a path that does not exist goes
    looking in the wrong place.

3.  THE SUGGESTION IS SEPARATE FROM THE LIST. Near-misses are offered first
    because `predictions` for `prediction` is the common typo, but the FULL
    list is always there too -- a suggestion that is wrong and a list that is
    absent is worse than no suggestion at all.
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
    """A named column is absent, and the message says what is present.

    A KeyError subclass so existing `except KeyError` paths still catch it,
    and so `settings['x']` failures and this one read the same way to a
    caller that does not care which it was.
    """

    def __init__(self, message: str, *, name: str = "",
                 available: Sequence[str] = ()):
        super().__init__(message)
        self.message = message
        #: What was asked for.
        self.name = name
        #: What the file has, in file order -- the order a user reading the
        #: header sees, not sorted, because sorting hides that the score
        #: columns are grouped together.
        self.available = list(available)

    def __str__(self) -> str:                       # pragma: no cover - trivial
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
        be parsed is ABSENT from the result rather than mapped to an empty
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
            # A file that is not a CSV, or is empty, or is being written.
            # Not fatal: the other paths may answer the question, and the
            # caller is about to be told which files it could read.
            continue
    return found


def missing(paths) -> List[str]:
    """The paths that could not be read. The other half of :func:`headers`."""
    readable = headers(paths)
    return [path for path in _as_paths(paths) if path not in readable]


def available(paths) -> List[str]:
    """Every column across ``paths``, in order, without duplicates.

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

    Case-insensitive, because `Prediction` for `prediction` is a typo nobody
    should have to see spelled out.
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
    """The sentence to print or raise when ``name`` is not there.

    Names the setting, the files that were read, the near-misses and then
    every column. Long on purpose: this is the message that decides whether
    the user fixes it in two minutes or re-runs the screen to find out.
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

    A CASE-INSENSITIVE MATCH IS ACCEPTED AND RETURNED IN THE FILE'S SPELLING.
    `Predictions` when the file says `predictions` is not a different column,
    and failing on it teaches a user to distrust the message rather than to
    fix the name.
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
