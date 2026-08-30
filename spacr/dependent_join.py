"""Join dependent-variable values to object-level measurement rows.

The direct identifier columns are preferred. When they are incomplete, crop
paths provide a documented fallback from which plate, well, field, and object
identifiers can be recovered. Each successful join reports the route used and
the number of matched rows; a join with no matches raises an error.
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

#: Canonical object identifiers imported from :mod:`spacr.schema`.
ID_COLUMNS: Tuple[str, ...] = (PLATE_KEY, ROW_KEY, COLUMN_KEY, FIELD_KEY,
                               OBJECT_KEY)

#: Supported column names for image or crop paths.
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
    """Convert a well name to canonical row and column identifiers.

    Parameters
    ----------
    well : str
        Well name such as ``"A01"`` or canonical identifier such as
        ``"r1c1"``.

    Returns
    -------
    tuple of str
        Canonical ``(rowID, columnID)`` values, for example ``("r1", "c1")``.
        Invalid well names return two empty strings.
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
    """Extract object identifiers from a crop-image path.

    Parameters
    ----------
    path : str
        Path whose stem contains ``plate_well_field_object``.

    Returns
    -------
    dict
        Canonical plate, row, column, field, and object identifiers plus the
        parsed well name. An empty mapping is returned when the pattern is not
        present.
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


#: Ordered join routes represented as ``(description, columns, source)``.
#: ``source`` identifies whether columns come from the table or parsed path.
ROUTES: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("the ID columns", ID_COLUMNS, "frame"),
    ("the image path, split into its parts", ID_COLUMNS, "path"),
    ("the well from the path, translated to row and column",
     (PLATE_KEY, ROW_KEY, COLUMN_KEY, FIELD_KEY, OBJECT_KEY), "path"),
)


def join(objects: pd.DataFrame, dependent: pd.DataFrame, *,
         value: str = "") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Attach dependent-variable columns to object-level rows.

    Parameters
    ----------
    objects : pandas.DataFrame
        Object-level measurement rows.
    dependent : pandas.DataFrame
        Table containing dependent variables and join identifiers or paths.
    value : str, optional
        Single dependent-variable column to add. When omitted, all columns
        not already present in ``objects`` are added.

    Returns
    -------
    pandas.DataFrame
        Copy of ``objects`` with matched dependent-variable columns.
    dict
        Join route, matched-row count, total-row count, attempted routes, and
        names of added columns.

    Raises
    ------
    ValueError
        If the dependent table is empty or no route matches any object row.
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
    """Format the join route and matched-row count for display.

    :param report: dependent-variable join report to summarize.
    """
    if not report.get("route"):
        return "the dependent variable was not joined"
    fallback = "" if report["route"] == ROUTES[0][0] else \
        " (a fallback -- a column from the direct join is missing upstream)"
    return (f"joined the dependent variable by {report['route']}: "
            f"{report['matched']:,} of {report['rows']:,} rows matched"
            f"{fallback}.")
