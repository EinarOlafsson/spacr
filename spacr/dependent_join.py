"""Join the dependent-variable table onto the object rows (instruction 213 B).

    "in my cas the dependent variable has plate, row, col, field, object
     which is enough to join with cell table but it also necessarily has the
     path to each image with is necessarily plate_well_field_object which can
     be split into its parts and merged directly or well translated into row
     and column and (with the format r1 and c2) and merged into the cell
     table so the latter two can be backups if any of the plate, row, col,
     filed, object columns cant be found or are missed for wahtever reason."

THREE ROUTES, TRIED IN ORDER, and the order is the instruction.

A FALLBACK IS A FALLBACK, NOT A SECRET. Which route was used is returned
with the number of rows it matched: a join that silently degraded is a join
whose result nobody can check, and the degradation itself is worth knowing
because it means a column is missing upstream.

AND A ROUTE THAT MATCHES NOTHING IS A FAILURE, not an empty answer -- the
same rule instruction 203 states for the object tables.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .schema import (COLUMN_KEY, FIELD_KEY, OBJECT_KEY, PLATE_KEY, ROW_KEY)

LOG = logging.getLogger("spacr.dependent_join")

#: The import's spelling, everywhere -- READ OFF `spacr.schema`, never
#: respelled here. A module reading `plate` where the rest of spaCR writes
#: `plateID` is a module that will silently match nothing the first time it
#: meets a real database, and a second copy of the names in this file would
#: be exactly that module waiting to happen.
ID_COLUMNS: Tuple[str, ...] = (PLATE_KEY, ROW_KEY, COLUMN_KEY, FIELD_KEY,
                               OBJECT_KEY)

#: Columns a dependent-variable table might carry its crop path under.
PATH_COLUMNS: Tuple[str, ...] = ("png_path", "path", "image_path", "file",
                                 "filename", "prcfo")

#: `plate_well_field_object`, which the path necessarily is.
_PARTS = re.compile(
    r"(?P<plate>[^_/\\]+)_(?P<well>[A-Za-z]?\d+|[A-Za-z]\d+|r\d+c\d+)"
    r"_(?P<field>\d+)_(?P<object>\d+)")

#: A well as a letter and a number: 'A01', 'B12'.
_WELL = re.compile(r"^(?P<row>[A-Za-z]+)(?P<column>\d+)$")


def _text(series) -> pd.Series:
    """As string, stripped. The join is on names, and names are text."""
    return pd.Series(series).astype(str).str.strip()


def well_to_row_and_column(well: str) -> Tuple[str, str]:
    """``'A01'`` -> ``('r1', 'c1')``.

    THE SPELLING IS `png_list`'s OWN. `spacr/predictions.py` records that
    its `rowID` returns 'r1', 'r2' rather than a number, and a route
    producing `1` would match nothing -- which is a failure that looks
    exactly like a screen with no overlap.
    """
    text = str(well or "").strip()
    already = re.match(r"^r(\d+)c(\d+)$", text, re.IGNORECASE)
    if already:
        return f"r{int(already.group(1))}", f"c{int(already.group(2))}"
    found = _WELL.match(text)
    if not found:
        return "", ""
    letters = found.group("row").upper()
    number = 0
    for character in letters:
        number = number * 26 + (ord(character) - ord("A") + 1)
    return f"r{number}", f"c{int(found.group('column'))}"


def parts_from_path(path: str) -> Dict[str, str]:
    """Recover plate, well, field and object from a crop path.

    The path is necessarily ``plate_well_field_object``, so the parts are
    there whenever it is.
    """
    stem = os.path.splitext(os.path.basename(str(path or "")))[0]
    found = _PARTS.search(stem)
    if not found:
        return {}
    row, column = well_to_row_and_column(found.group("well"))
    return {PLATE_KEY: found.group("plate"),
            "well": found.group("well"),
            ROW_KEY: row, COLUMN_KEY: column,
            FIELD_KEY: found.group("field"),
            OBJECT_KEY: found.group("object")}


def _key(frame: pd.DataFrame, columns: Sequence[str]) -> Optional[pd.Series]:
    """A join key from ``columns``, or None if any is missing."""
    if any(c not in frame.columns for c in columns):
        return None
    parts = [_text(frame[c]) for c in columns]
    return parts[0].str.cat(parts[1:], sep="_")


def _from_paths(frame: pd.DataFrame) -> Optional[pd.DataFrame]:
    """The five ID columns recovered from whichever path column exists."""
    column = next((c for c in PATH_COLUMNS if c in frame.columns), None)
    if column is None:
        return None
    rows = [parts_from_path(value) for value in frame[column]]
    if not any(rows):
        return None
    return pd.DataFrame(rows, index=frame.index)


#: The routes, in the instruction's order, as ``(name, columns, source)``.
#: `source` is what the columns are read off: the frame itself, or the parts
#: split out of its path.
ROUTES: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("the ID columns", ID_COLUMNS, "frame"),
    ("the image path, split into its parts", ID_COLUMNS, "path"),
    ("the well from the path, translated to row and column",
     (PLATE_KEY, ROW_KEY, COLUMN_KEY, FIELD_KEY, OBJECT_KEY), "path"),
)


def join(objects: pd.DataFrame, dependent: pd.DataFrame, *,
         value: str = "") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Attach ``dependent`` to ``objects``. Returns the frame and a report.

    :param objects: the object rows -- the cell table, or the montage's.
    :param dependent: the dependent-variable table.
    :param value: the column to bring across; every new column when empty.
    :returns: ``(frame, report)``. The report names the ROUTE used and the
        rows matched, always -- see the module docstring.
    :raises ValueError: when no route matched a single row. A route that
        matches nothing is a failure, not an empty answer.
    """
    report: Dict[str, Any] = {"route": "", "matched": 0,
                              "rows": int(len(objects)), "tried": []}
    if dependent is None or not len(dependent):
        raise ValueError("the dependent-variable table is empty, so there is "
                         "nothing to join")

    theirs_source = {"frame": dependent, "path": None}
    mine_source = {"frame": objects, "path": None}
    for name, columns, source in ROUTES:
        if source == "path":
            if theirs_source["path"] is None:
                theirs_source["path"] = _from_paths(dependent)
                mine_source["path"] = _from_paths(objects)
            theirs_frame = theirs_source["path"]
            # THE OBJECT SIDE KEEPS ITS OWN COLUMNS WHEN IT HAS THEM. Only
            # the dependent table is the one with something missing; making
            # both sides go through the path would break a join that the ID
            # columns would have made.
            mine_frame = objects if _key(objects, columns) is not None \
                else mine_source["path"]
        else:
            theirs_frame, mine_frame = dependent, objects
        if theirs_frame is None or mine_frame is None:
            report["tried"].append(f"{name}: no usable columns")
            continue
        theirs = _key(theirs_frame, columns)
        mine = _key(mine_frame, columns)
        if theirs is None or mine is None:
            missing = [c for c in columns
                       if c not in getattr(theirs_frame, "columns", ())]
            report["tried"].append(
                f"{name}: missing {', '.join(missing) or 'columns'}")
            continue
        matched = int(pd.Index(mine).isin(set(theirs)).sum())
        if not matched:
            report["tried"].append(f"{name}: matched no row")
            continue
        report["route"] = name
        report["matched"] = matched
        break
    else:
        raise ValueError(
            "the dependent variable could not be joined by any route: "
            + "; ".join(report["tried"]))

    columns = next(c for n, c, _ in ROUTES if n == report["route"])
    source = next(s for n, _, s in ROUTES if n == report["route"])
    theirs_frame = dependent if source == "frame" else theirs_source["path"]
    mine_frame = objects if _key(objects, columns) is not None \
        else mine_source["path"]
    bring = [c for c in dependent.columns
             if (not value or str(c) == value)
             and str(c) not in set(map(str, objects.columns))]
    lookup = dependent[bring].copy()
    lookup.index = pd.Index(_key(theirs_frame, columns))
    lookup = lookup[~lookup.index.duplicated(keep="first")]
    out = objects.copy()
    added = lookup.reindex(pd.Index(_key(mine_frame, columns)))
    for name in bring:
        out[name] = added[name].to_numpy()
    report["added"] = [str(c) for c in bring]
    return out, report


def describe(report: Dict[str, Any]) -> str:
    """The report line. Names the route and what it matched."""
    if not report.get("route"):
        return "the dependent variable was not joined"
    fallback = "" if report["route"] == ROUTES[0][0] else \
        " (a fallback -- a column from the direct join is missing upstream)"
    return (f"joined the dependent variable by {report['route']}: "
            f"{report['matched']:,} of {report['rows']:,} rows matched"
            f"{fallback}.")
