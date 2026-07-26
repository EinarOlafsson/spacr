"""
Plate-level QC: is that hit biology, or is it the edge of the plate?

The outer ring of a microtitre plate evaporates faster, sits at a
different temperature and is handled differently from the interior. A
well in row A or column 24 that reads three sigma from the plate mean is
therefore much more likely to be an artefact of *where it sits* than of
what is in it — and the usual way that gets discovered is a failed
follow-up experiment six weeks later.

This module answers the question up front. It takes the same long-format
per-object frame the rest of spaCR passes around (a ``prc`` identifier
plus feature columns), collapses it to one value per well, and then asks
three separate questions of the resulting grid:

``detect_edge_effect``
    Does the outer ring read differently from the interior — and by how
    much? Ring-by-ring, not just outermost-vs-rest.
``row_column_trends``
    What does each row and each column average, and is there a monotonic
    drift across the plate?
``plate_layout`` / ``layout_matrix``
    The well grid itself, tidy or pivoted, ready to draw.
``format_edge_report``
    All of it as text.

Statistics this module refuses to get wrong
-------------------------------------------
**Rank tests, not t-tests.** Per-well aggregates of object measurements
are routinely skewed and heavy-tailed — a well with three enormous cells
is not a normal deviate. The ring comparison is a two-sided
Mann-Whitney U; the gradient test is Spearman. Neither assumes a shape.

**Effect size leads, p-value follows.** With 384 wells, almost any
systematic drift clears p < 0.05; that fact carries no information. Every
comparison therefore reports the median difference, the difference as a
percentage of the interior median, *and* Cliff's delta — a rank-based
standardised effect in [-1, 1] read straight off the same U statistic
(``δ = 2U/n₁n₂ − 1``). Detection requires both a small p **and** an
effect size above :data:`DEFAULT_MIN_EFFECT`, so "significant but 0.4 %"
is reported as what it is: not an edge effect.

**Evaporation is not a step function.** The profile walks inward ring by
ring (outermost = ring 0), comparing each against the plate core, so a
gradient that reaches two wells in is visible instead of being averaged
into "the interior".

**A gradient is not an edge effect.** An incubator or plate-reader
gradient runs *across* the plate; evaporation runs *around* it. A linear
column gradient leaves the outer ring with the same median as the
interior (it collects both the high and the low end), so the ring test
stays quiet while Spearman on the column index fires. Both are computed,
both are reported, and :attr:`EdgeEffectReport.dominant` names which one
better explains the plate.

**Wells with too few objects are noise.** ``min_count`` drops them — and
the number dropped is reported everywhere, because a heatmap quietly
missing a third of its wells looks exactly like data.

**The plate format is inferred, never assumed.** 6/12/24/48/96/384/1536
are recognised from the observed row and column labels, and the ring
geometry uses the *nominal* grid of the inferred format. A screen that
only used rows A-H of a 384 plate does not get row H promoted to "edge"
just because nothing was pipetted below it.

**Degenerate input explains itself.** An empty frame, a single well, or a
plate where every well reads the same comes back as a report that says
so, with ``None`` where a statistic is genuinely undefined. Nothing here
returns NaN and nothing here raises on empty.

Nothing in this module imports torch, cellpose or any GPU stack — numpy,
pandas, scipy and the standard library only — so the Qt screen can draw a
plate without waking a multi-second import chain.
"""
from __future__ import annotations

import csv
import math
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote as _urlquote

import numpy as np
import pandas as pd
from scipy import stats as _sps

from . import schema

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_CORE_DEPTH",
    "DEFAULT_MIN_EFFECT",
    "DEFAULT_MIN_GRADIENT_RHO",
    "EdgeEffectReport",
    "GradientStats",
    "GROUPINGS",
    "PLATE_FORMATS",
    "RingStats",
    "colour_limits",
    "detect_edge_effect",
    "format_edge_report",
    "infer_plate_format",
    "layout_matrix",
    "load_plate_frame",
    "numeric_columns",
    "parse_column_label",
    "parse_row_label",
    "plate_layout",
    "plates_in",
    "row_column_trends",
    "row_label",
    "table_columns",
    "tables",
    "well_id",
    "write_layout_csv",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Standard SBS plate formats as ``n_wells -> (n_rows, n_columns)``, in
#: ascending size. Inference picks the *smallest* format whose nominal
#: grid contains every observed row and column label.
#:
#: The numbers come from :data:`spacr.schema.PLATE_FORMATS`; only the shape
#: differs (an ordered tuple of pairs, which is what this module's public API
#: has always exposed). Two copies of the plate geometry is how QC and the
#: database come to disagree about whether column 30 exists.
PLATE_FORMATS: Tuple[Tuple[int, Tuple[int, int]], ...] = tuple(
    (n_wells, schema.PLATE_FORMATS[n_wells])
    for n_wells in sorted(schema.PLATE_FORMATS)
)

#: Per-well aggregations ``plate_layout`` understands. ``'count'`` ignores
#: ``value_col`` entirely and plots the number of objects per well.
GROUPINGS: Tuple[str, ...] = (
    "mean", "median", "sum", "count", "min", "max", "std",
)

#: Significance threshold. Necessary but nowhere near sufficient — see
#: :data:`DEFAULT_MIN_EFFECT`.
DEFAULT_ALPHA = 0.05

#: Minimum |Cliff's delta| for the outer ring to count as an edge effect.
#: Cliff's conventions are 0.147 small / 0.33 medium / 0.474 large; 0.2
#: sits just above "small", which is where a plate stops being worth
#: re-running and starts being worth re-plating. With 384 wells a p-value
#: alone would flag drifts of well under one percent.
DEFAULT_MIN_EFFECT = 0.2

#: Minimum |Spearman rho| for a row/column drift to count as a gradient.
DEFAULT_MIN_GRADIENT_RHO = 0.3

#: Ring depth at or beyond which wells are called "the core" for the
#: ring-by-ring profile. Depth 2 means the profile compares ring 0 and
#: ring 1 against wells at least two in from every edge, so ring 1 is not
#: measured against a baseline that includes ring 1.
DEFAULT_CORE_DEPTH = 2

#: How many rings inward the profile walks by default.
DEFAULT_MAX_RINGS = 3

#: Identifier columns ``load_plate_frame`` pulls alongside the value.
WELL_ID_COLUMNS: Tuple[str, ...] = (
    "prc", "plateID", "rowID", "columnID", "plate", "row", "column",
    "plate_name", "row_name", "column_name", "well",
)

#: Tidy layout column order — also the CSV export order.
LAYOUT_COLUMNS: Tuple[str, ...] = (
    "plateID", "well", "rowID", "columnID", "row_index", "column_index",
    "n", "value", "ring", "is_edge",
)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _finite(value: Any) -> Optional[float]:
    """Return ``value`` as a float, or ``None`` if it is not finite.

    Every public statistic goes through here. A NaN in a QC report is
    indistinguishable from "we didn't check", so undefined answers come
    back as ``None`` and get rendered as the *reason* they're undefined.
    """
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) or math.isinf(out) else out


def _median(values: np.ndarray) -> Optional[float]:
    return _finite(np.median(values)) if values.size else None


def _mean(values: np.ndarray) -> Optional[float]:
    return _finite(np.mean(values)) if values.size else None


def _pct_of(delta: Optional[float], baseline: Optional[float]) -> Optional[float]:
    """``delta`` as a percentage of ``|baseline|``.

    ``None`` when the baseline is zero — "300 % of nothing" is not a
    number a user should be shown.
    """
    if delta is None or baseline is None or baseline == 0:
        return None
    return _finite(100.0 * delta / abs(baseline))


def _rank_compare(a: np.ndarray, b: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Two-sided Mann-Whitney U of ``a`` vs ``b`` → ``(p, cliffs_delta)``.

    Cliff's delta is read directly off the U statistic —
    ``δ = 2U₁/(n₁n₂) − 1`` — so the effect size and the p-value describe
    exactly the same comparison rather than two different ones. δ > 0
    means ``a`` tends to read higher than ``b``.

    Returns ``(None, None)`` when either group is empty, and
    ``(1.0, 0.0)`` when every value in both groups is identical (scipy
    cannot rank a constant, but the honest answer there is "no
    difference", not an error).
    """
    if a.size == 0 or b.size == 0:
        return None, None
    both = np.concatenate([a, b])
    if float(np.max(both) - np.min(both)) == 0.0:
        return 1.0, 0.0
    try:
        res = _sps.mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        return None, None
    delta = 2.0 * float(res.statistic) / float(a.size * b.size) - 1.0
    return _finite(res.pvalue), _finite(delta)


# ---------------------------------------------------------------------------
# Well / row / column labels
# ---------------------------------------------------------------------------

_ROW_NUMERIC_RE = re.compile(r"^\s*(?:row)?[\s_-]*r?[\s_-]*(\d+)\s*$", re.IGNORECASE)
_COL_NUMERIC_RE = re.compile(r"^\s*(?:col(?:umn)?)?[\s_-]*c?[\s_-]*(\d+)\s*$", re.IGNORECASE)
_ALPHA_RE = re.compile(r"^\s*([A-Za-z]{1,2})\s*$")
_WELL_RE = re.compile(r"^\s*([A-Za-z]{1,2})[\s_-]*(\d{1,3})\s*$")


def _alpha_to_index(text: str) -> Optional[int]:
    """``A`` → 1, ``Z`` → 26, ``AA`` → 27 … (1536 plates run to ``AF``).

    :func:`spacr.schema.row_index_from_letters` is the definition; this
    wrapper only coerces to text first, because the callers here hand it
    regex groups rather than guaranteed strings.
    """
    return schema.row_index_from_letters(str(text))


def _index_to_alpha(index: int) -> str:
    """Inverse of :func:`_alpha_to_index`. ``27`` → ``'AA'``.

    ``'?'`` for an index that is not a row — a label in a report, never a
    key, so it says "unknown" rather than raising the way
    :func:`spacr.schema.letters_from_row_index` does.
    """
    try:
        return schema.letters_from_row_index(index)
    except schema.KeyParseError:
        return "?"


def parse_row_label(label: Any) -> Optional[int]:
    """Return the 1-based row index of ``label``, or ``None``.

    Understands every row spelling spaCR and plate readers produce:
    ``'r3'``, ``'R3'``, ``'row3'``, ``3``, ``'C'`` (letter rows) and
    ``'AA'`` (1536 plates go past Z).

    :param label: row identifier of any of the above shapes.
    :returns: 1-based row index, or ``None`` when unparseable.
    """
    if label is None or (isinstance(label, float) and math.isnan(label)):
        return None
    if isinstance(label, (int, np.integer)) and not isinstance(label, bool):
        return int(label) if int(label) > 0 else None
    text = str(label).strip()
    if not text:
        return None
    m = _ROW_NUMERIC_RE.match(text)
    if m:
        value = int(m.group(1))
        return value if value > 0 else None
    m = _ALPHA_RE.match(text)
    if m:
        return _alpha_to_index(m.group(1))
    return None


def parse_column_label(label: Any) -> Optional[int]:
    """Return the 1-based column index of ``label``, or ``None``.

    Understands ``'c12'``, ``'C12'``, ``'column12'``, ``'12'`` and ``12``.

    :param label: column identifier.
    :returns: 1-based column index, or ``None`` when unparseable.
    """
    if label is None or (isinstance(label, float) and math.isnan(label)):
        return None
    if isinstance(label, (int, np.integer)) and not isinstance(label, bool):
        return int(label) if int(label) > 0 else None
    text = str(label).strip()
    if not text:
        return None
    m = _COL_NUMERIC_RE.match(text)
    if m:
        value = int(m.group(1))
        return value if value > 0 else None
    return None


def _parse_well_label(label: Any) -> Optional[Tuple[int, int]]:
    """``'A01'`` → ``(1, 1)``; ``'AF48'`` → ``(32, 48)``."""
    if label is None:
        return None
    m = _WELL_RE.match(str(label).strip())
    if not m:
        return None
    row = _alpha_to_index(m.group(1))
    col = int(m.group(2))
    if not row or col <= 0:
        return None
    return row, col


def row_label(row_index: int) -> str:
    """Return the letter label of a 1-based row index: ``3`` → ``'C'``."""
    return _index_to_alpha(int(row_index))


def well_id(row_index: int, column_index: int) -> str:
    """Return the canonical well name, e.g. ``(3, 7)`` → ``'C07'``.

    Agrees with :func:`spacr.schema.well_id` on every real well; it differs
    only in refusing to raise, because a layout table has to render every
    cell it was handed and ``'?00'`` is a more useful report than a
    traceback.
    """
    try:
        return schema.well_id(int(row_index), int(column_index))
    except (schema.SchemaError, TypeError, ValueError):
        return f"{_index_to_alpha(row_index)}{int(column_index):02d}"


def infer_plate_format(n_rows: int, n_cols: int) -> Tuple[Optional[int], int, int]:
    """Infer the plate format containing a ``n_rows`` × ``n_cols`` extent.

    Picks the smallest standard format whose nominal grid contains every
    observed label, so an assay run only in rows A-H of a 384 plate is
    still recognised as a 96 grid — and one run in rows A-H across 24
    columns is recognised as a 384 plate whose lower half is empty,
    rather than having row H mistaken for the bottom edge.

    :param n_rows: largest observed row index.
    :param n_cols: largest observed column index.
    :returns: ``(n_wells_or_None, nominal_rows, nominal_cols)``. The
        format is ``None`` for a grid bigger than 1536, in which case the
        observed extent is returned unchanged.
    """
    n_rows = max(int(n_rows), 0)
    n_cols = max(int(n_cols), 0)
    for wells, (rows, cols) in PLATE_FORMATS:
        if n_rows <= rows and n_cols <= cols:
            return wells, rows, cols
    return None, n_rows, n_cols


# ---------------------------------------------------------------------------
# Read-only database access (mirrors spacr.agreement / the Database Browser)
# ---------------------------------------------------------------------------

def _quote_ident(name: str) -> str:
    """Double-quote a SQL identifier that has already been schema-checked."""
    return '"' + str(name).replace('"', '""') + '"'


def _read_only_uri(path: str) -> str:
    """``file:…?mode=ro`` URI — SQLite itself then refuses every write."""
    return "file:" + _urlquote(str(path).replace("\\", "/"), safe="/:") + "?mode=ro"


def _connect(db_path: str) -> sqlite3.Connection:
    """Open ``db_path`` read-only. Never writes, never journals.

    :raises ValueError: when no path was given.
    :raises FileNotFoundError: when the file is not there — sqlite's own
        "unable to open database file" never says *which* file.
    """
    if not db_path or not str(db_path).strip():
        raise ValueError("No database path given.")
    path = os.path.abspath(os.path.expanduser(str(db_path).strip()))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such database: {path}")
    con = sqlite3.connect(_read_only_uri(path), uri=True)
    con.execute("PRAGMA query_only = ON")
    return con


def tables(db_path: str) -> List[str]:
    """Return the user tables + views of ``db_path``, alphabetically."""
    con = _connect(db_path)
    try:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def table_columns(db_path: str, table: str) -> List[str]:
    """Return the column names of ``table``, in declaration order.

    :raises ValueError: when the database has no such table. The name is
        checked against ``sqlite_master`` before it is ever interpolated
        into SQL — identifiers cannot be bound, so this is the gate.
    """
    known = tables(db_path)
    if table not in known:
        raise ValueError(
            f"{os.path.basename(str(db_path))} has no {table!r} table "
            f"(found: {', '.join(known) or 'nothing'}).")
    con = _connect(db_path)
    try:
        rows = con.execute(f"PRAGMA table_info({_quote_ident(table)})").fetchall()
    finally:
        con.close()
    return [r[1] for r in rows]


def numeric_columns(db_path: str, table: str,
                    sample: int = 500) -> List[str]:
    """Return the columns of ``table`` that hold plottable numbers.

    Declared affinity is the first filter, but spaCR writes plenty of
    feature tables through ``pandas.to_sql`` where everything lands as
    ``REAL``/``INTEGER`` anyway — including ``object_label`` and the row
    index. So a column also has to actually contain numbers in the first
    ``sample`` rows, and the well-identifier columns are excluded by name.

    :param db_path: path to ``measurements.db``.
    :param table: table to inspect.
    :param sample: rows sampled to confirm the values are numeric.
    :returns: candidate column names, in table order.
    """
    columns = table_columns(db_path, table)
    exclude = {c.lower() for c in WELL_ID_COLUMNS} | {"prcf", "prcfo", "index"}
    con = _connect(db_path)
    out: List[str] = []
    try:
        info = con.execute(
            f"PRAGMA table_info({_quote_ident(table)})").fetchall()
        declared = {r[1]: (r[2] or "").upper() for r in info}
        for col in columns:
            if col.lower() in exclude:
                continue
            affinity = declared.get(col, "")
            if affinity and not any(t in affinity for t in
                                    ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC")):
                continue
            rows = con.execute(
                f"SELECT {_quote_ident(col)} FROM {_quote_ident(table)} "
                f"WHERE {_quote_ident(col)} IS NOT NULL LIMIT ?",
                (int(sample),)).fetchall()
            values = pd.to_numeric(pd.Series([r[0] for r in rows]), errors="coerce")
            if len(values) and values.notna().any():
                out.append(col)
    finally:
        con.close()
    return out


def load_plate_frame(db_path: str, table: str, value_col: str,
                     limit: Optional[int] = None) -> pd.DataFrame:
    """Read the well identifiers plus ``value_col`` from ``table``.

    Only the columns needed to place and colour a well are selected — a
    spaCR feature table can be 500 columns wide and half a million rows
    long, and ``SELECT *`` on one is a minute of nothing happening.

    :param db_path: path to ``measurements.db`` (opened read-only).
    :param table: table to read.
    :param value_col: the measurement column to plot.
    :param limit: optional ``LIMIT`` for previews.
    :returns: a long DataFrame, one row per object.
    :raises ValueError: for an unknown table or column, or when the table
        carries no usable well identifier at all.
    """
    real = table_columns(db_path, table)
    if value_col not in real:
        raise ValueError(
            f"{table!r} has no column {value_col!r}. "
            f"Available: {', '.join(real[:20]) or 'nothing'}"
            f"{'…' if len(real) > 20 else ''}")
    ids = [c for c in WELL_ID_COLUMNS if c in real and c != value_col]
    if not ids:
        raise ValueError(
            f"{table!r} carries no well identifier — expected one of "
            f"{', '.join(WELL_ID_COLUMNS[:6])}. Without it there is no way "
            f"to say which well a row belongs to.")
    select = ", ".join(_quote_ident(c) for c in ids + [value_col])
    sql = f"SELECT {select} FROM {_quote_ident(table)}"
    params: Tuple = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (int(limit),)
    con = _connect(db_path)
    try:
        rows = con.execute(sql, params).fetchall()
    finally:
        con.close()
    return pd.DataFrame(rows, columns=ids + [value_col])


# ---------------------------------------------------------------------------
# Long frame → well grid
# ---------------------------------------------------------------------------

def _prc_parts(series: pd.Series) -> Optional[pd.DataFrame]:
    """Split a ``prc`` column into ``plateID``/``rowID``/``columnID``.

    spaCR writes ``prc`` as ``plateID_rowID_columnID`` (``spacr.io``),
    but a four-token ``plateID_rowID_columnID_fieldID`` shows up too, and
    plate names themselves sometimes contain underscores. Rather than
    hard-coding a token offset, the row/column pair is located by asking
    which adjacent tokens actually *parse* as a row and a column label.
    """
    split = series.astype(str).str.split("_", expand=True)
    n_parts = split.shape[1]
    if n_parts < 3:
        return None
    # Candidate (row, column) token offsets, most likely first:
    #   1,2 -> plateID_rowID_columnID[_fieldID]   (what spacr.io writes)
    #   2,3 -> a plate name containing one underscore
    best: Optional[Tuple[int, int]] = None
    best_score = 0.0
    for r_i, c_i in ((1, 2), (2, 3)):
        if c_i >= n_parts:
            continue
        rows = split[r_i].map(parse_row_label)
        cols = split[c_i].map(parse_column_label)
        score = float((rows.notna() & cols.notna()).mean())
        if score > best_score:
            best, best_score = (r_i, c_i), score
    if best is None or best_score == 0.0:
        return None
    r_i, c_i = best
    plate = split[0] if r_i == 1 else split[0] + "_" + split[1]
    return pd.DataFrame({
        "plateID": plate.values,
        "rowID": split[r_i].values,
        "columnID": split[c_i].values,
    }, index=series.index)


def _first_column(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def _identify_wells(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Return ``df`` plus ``plateID``/``rowID``/``columnID`` columns.

    Explicit identifier columns win over ``prc``: they are what the
    measurement code wrote, whereas ``prc`` is a string it derived.

    :returns: ``(frame, notes)``.
    :raises ValueError: when nothing in the frame locates a well.
    """
    notes: List[str] = []
    out = df.copy()

    row_col = _first_column(out, ("rowID", "row_name", "row"))
    col_col = _first_column(out, ("columnID", "column_name", "column"))
    plate_col = _first_column(out, ("plateID", "plate_name", "plate"))

    if row_col and col_col:
        source = f"{row_col}/{col_col}"
        out["rowID"] = out[row_col]
        out["columnID"] = out[col_col]
        if plate_col:
            out["plateID"] = out[plate_col].astype(str)
        elif "prc" in out.columns:
            parts = _prc_parts(out["prc"])
            out["plateID"] = parts["plateID"] if parts is not None else "p1"
        else:
            out["plateID"] = "p1"
            notes.append("No plate column — every well assigned to plate 'p1'.")
    elif "prc" in out.columns and len(out):
        parts = _prc_parts(out["prc"])
        if parts is None:
            raise ValueError(
                "The 'prc' column does not look like "
                "'plateID_rowID_columnID' — no row/column pair in it parses "
                "as a well position.")
        source = "prc"
        out["plateID"] = parts["plateID"]
        out["rowID"] = parts["rowID"]
        out["columnID"] = parts["columnID"]
    elif "well" in out.columns and len(out):
        source = "well"
        parsed = out["well"].map(_parse_well_label)
        out["rowID"] = [p[0] if p else None for p in parsed]
        out["columnID"] = [p[1] if p else None for p in parsed]
        out["plateID"] = (out[plate_col].astype(str) if plate_col else "p1")
        if not plate_col:
            notes.append("No plate column — every well assigned to plate 'p1'.")
    else:
        raise ValueError(
            "No well identifier in this table — expected 'prc', a "
            "rowID/columnID pair, or a 'well' column such as 'A01'.")

    notes.append(f"Wells located from {source}.")
    return out, notes


def plates_in(df: pd.DataFrame) -> List[str]:
    """Return the plate IDs present in ``df``, sorted.

    Accepts either a raw long frame or a layout from
    :func:`plate_layout`. Returns ``[]`` for anything unusable rather
    than raising — this feeds a combo box.
    """
    if df is None or not len(df):
        return []
    try:
        located, _ = _identify_wells(df)
    except ValueError:
        return []
    return sorted(pd.Series(located["plateID"]).dropna().astype(str).unique().tolist())


def _is_layout(df: pd.DataFrame) -> bool:
    """True when ``df`` already came out of :func:`plate_layout`."""
    return isinstance(df, pd.DataFrame) and {
        "row_index", "column_index", "value", "n"}.issubset(df.columns)


def _empty_layout() -> pd.DataFrame:
    frame = pd.DataFrame({
        "plateID": pd.Series(dtype=object),
        "well": pd.Series(dtype=object),
        "rowID": pd.Series(dtype=object),
        "columnID": pd.Series(dtype=object),
        "row_index": pd.Series(dtype="int64"),
        "column_index": pd.Series(dtype="int64"),
        "n": pd.Series(dtype="int64"),
        "value": pd.Series(dtype="float64"),
        "ring": pd.Series(dtype="int64"),
        "is_edge": pd.Series(dtype=bool),
    })
    frame.attrs = dict(_DEFAULT_META)
    return frame


_DEFAULT_META: Dict[str, Any] = {
    "plate": None,
    "value_col": None,
    "grouping": "mean",
    "min_count": 0,
    "plate_format": None,
    "n_rows": 0,
    "n_cols": 0,
    "n_wells": 0,
    "n_dropped_min_count": 0,
    "n_unparsed_rows": 0,
    "n_plates": 0,
    "notes": (),
}


def plate_layout(df: pd.DataFrame,
                 value_col: Optional[str] = None,
                 plate: Optional[str] = None,
                 grouping: str = "mean",
                 min_count: int = 0,
                 plate_format: Optional[int] = None) -> pd.DataFrame:
    """Collapse a long per-object frame into one row per well.

    This is the grid everything else in the module works on: the
    heatmap draws it, :func:`detect_edge_effect` tests it, and
    :func:`write_layout_csv` exports it.

    Aggregation and the well-identifier handling follow
    :func:`spacr.plot.generate_plate_heatmap` so the two agree well for
    well, with two deliberate departures, each of which fixes a way the
    plotter misleads:

    * **Missing wells stay missing.** ``generate_plate_heatmap`` ends in
      ``.fillna(0)``, which paints an unpipetted or filtered-out well as
      a real measurement of zero. Here an absent well is absent.
    * **The grid is nominal, not observed.** The plotter's axes span the
      wells that are present; these span the whole inferred format, so a
      row nobody pipetted still holds its place on the plate and the ring
      geometry :func:`detect_edge_effect` needs stays intact.

    The third departure has since been closed from the other side. The
    plotter used to pin rows to ``r1..r16`` and columns to ``c1..c27``,
    silently dropping every well of a 1536 plate past row P; it now reads
    its axes off the data through :func:`parse_row_label` /
    :func:`parse_column_label` — these functions — so ``'B'`` and ``'AA'``
    are row labels in both places, and neither module owns a private
    letter walk.

    :param df: long frame with a ``prc`` identifier (or ``rowID`` +
        ``columnID``, or a ``well`` column) and ``value_col``. A layout
        produced by a previous call is passed straight back.
    :param value_col: measurement to aggregate. Ignored (and optional)
        when ``grouping='count'``.
    :param plate: plate to keep. ``None`` takes the first plate present
        and records the choice in the layout's ``notes``.
    :param grouping: one of :data:`GROUPINGS`.
    :param min_count: drop wells with fewer than this many objects. The
        number dropped is recorded in ``layout.attrs['n_dropped_min_count']``.
    :param plate_format: force a format (96 / 384 / 1536) instead of
        inferring one.
    :returns: tidy DataFrame with :data:`LAYOUT_COLUMNS`, carrying the
        inferred geometry and drop counts in ``.attrs``.
    :raises ValueError: for an unknown ``grouping``, a missing
        ``value_col``, or a frame with no usable well identifier.

    Example:
        .. code-block:: python

            from spacr.plate_qc import plate_layout, layout_matrix
            wells = plate_layout(df, 'cell_area', grouping='median',
                                 min_count=20)
            grid = layout_matrix(wells)      # 16 x 24, NaN where empty
    """
    if grouping not in GROUPINGS:
        raise ValueError(
            f"grouping must be one of {', '.join(GROUPINGS)} — got {grouping!r}")

    if _is_layout(df):
        out = df.copy()
        meta = dict(_DEFAULT_META)
        meta.update(getattr(df, "attrs", {}) or {})
        out.attrs = meta
        return out

    if df is None or not len(df):
        empty = _empty_layout()
        meta = dict(empty.attrs)
        meta.update({"value_col": value_col, "grouping": grouping,
                     "min_count": int(min_count), "plate": plate,
                     "notes": ("The frame is empty — no wells to lay out.",)})
        empty.attrs = meta
        return empty

    located, notes = _identify_wells(df)

    plate_ids = pd.Series(located["plateID"]).dropna().astype(str)
    all_plates = sorted(plate_ids.unique().tolist())
    if plate is None:
        if len(all_plates) > 1:
            notes.append(
                f"{len(all_plates)} plates present ({', '.join(all_plates[:4])}"
                f"{'…' if len(all_plates) > 4 else ''}); showing "
                f"{all_plates[0]!r}.")
        plate = all_plates[0] if all_plates else None
    located = located[located["plateID"].astype(str) == str(plate)].copy()
    if not len(located):
        empty = _empty_layout()
        meta = dict(empty.attrs)
        meta.update({"plate": plate, "value_col": value_col,
                     "grouping": grouping, "min_count": int(min_count),
                     "n_plates": len(all_plates),
                     "notes": tuple(notes + [
                         f"No rows for plate {plate!r}. Present: "
                         f"{', '.join(all_plates) or 'none'}."])})
        empty.attrs = meta
        return empty

    located["row_index"] = located["rowID"].map(parse_row_label)
    located["column_index"] = located["columnID"].map(parse_column_label)
    n_before = len(located)
    located = located.dropna(subset=["row_index", "column_index"])
    n_unparsed = n_before - len(located)
    if n_unparsed:
        notes.append(
            f"{n_unparsed} of {n_before} rows had a row/column label that "
            f"could not be read as a well position and were skipped.")
    if not len(located):
        empty = _empty_layout()
        meta = dict(empty.attrs)
        meta.update({"plate": plate, "value_col": value_col,
                     "grouping": grouping, "min_count": int(min_count),
                     "n_unparsed_rows": n_unparsed,
                     "n_plates": len(all_plates),
                     "notes": tuple(notes)})
        empty.attrs = meta
        return empty

    located["row_index"] = located["row_index"].astype(int)
    located["column_index"] = located["column_index"].astype(int)

    # -- aggregate ---------------------------------------------------------
    if grouping == "count":
        located["__value__"] = 1.0
    else:
        if not value_col:
            raise ValueError(
                f"grouping={grouping!r} needs a value column to aggregate.")
        if value_col not in located.columns:
            raise ValueError(
                f"No column {value_col!r} in the frame. Available: "
                f"{', '.join(map(str, list(df.columns)[:20]))}")
        located["__value__"] = pd.to_numeric(located[value_col], errors="coerce")

    keys = ["row_index", "column_index"]
    grouped = located.groupby(keys, observed=True)
    counts = grouped.size().rename("n")
    if grouping == "count":
        values = counts.astype(float).rename("value")
    else:
        values = grouped["__value__"].agg(grouping).rename("value")
    wells = pd.concat([counts, values], axis=1).reset_index()

    # -- min_count ---------------------------------------------------------
    n_dropped = 0
    if min_count and min_count > 0:
        keep = wells["n"] >= int(min_count)
        n_dropped = int((~keep).sum())
        wells = wells[keep]
        if n_dropped:
            notes.append(
                f"{n_dropped} well(s) held fewer than {int(min_count)} "
                f"objects and were dropped — they are blank on the heatmap, "
                f"not zero.")

    if not len(wells):
        # Everything was filtered away. There is no grid to infer — a
        # 2x3 "plate" invented from an empty extent would be a lie with
        # a shape.
        empty = _empty_layout()
        meta = dict(empty.attrs)
        meta.update({"plate": str(plate), "value_col": value_col,
                     "grouping": grouping, "min_count": int(min_count or 0),
                     "n_dropped_min_count": int(n_dropped),
                     "n_unparsed_rows": int(n_unparsed),
                     "n_plates": len(all_plates),
                     "notes": tuple(notes + [
                         "No wells left after filtering — lower min_count."])})
        empty.attrs = meta
        return empty

    # -- geometry ----------------------------------------------------------
    obs_rows = int(wells["row_index"].max())
    obs_cols = int(wells["column_index"].max())
    if plate_format:
        forced = dict(PLATE_FORMATS).get(int(plate_format))
        if forced is None:
            raise ValueError(
                f"plate_format must be one of "
                f"{', '.join(str(w) for w, _ in PLATE_FORMATS)} — got "
                f"{plate_format!r}")
        fmt, n_rows, n_cols = int(plate_format), forced[0], forced[1]
        if obs_rows > n_rows or obs_cols > n_cols:
            notes.append(
                f"Forced {fmt}-well geometry ({n_rows}x{n_cols}) is smaller "
                f"than the observed {obs_rows}x{obs_cols} extent.")
            n_rows, n_cols = max(n_rows, obs_rows), max(n_cols, obs_cols)
    else:
        fmt, n_rows, n_cols = infer_plate_format(obs_rows, obs_cols)
        if fmt:
            notes.append(
                f"Inferred a {fmt}-well plate ({n_rows}x{n_cols}) from the "
                f"observed labels (rows to {_index_to_alpha(obs_rows)}, "
                f"columns to {obs_cols}).")
        else:
            notes.append(
                f"Observed {obs_rows}x{obs_cols} matches no standard plate "
                f"format; using the observed extent as the grid.")

    depth = np.minimum.reduce([
        wells["row_index"].to_numpy() - 1,
        n_rows - wells["row_index"].to_numpy(),
        wells["column_index"].to_numpy() - 1,
        n_cols - wells["column_index"].to_numpy(),
    ])
    wells["ring"] = np.maximum(depth, 0).astype(int)
    wells["is_edge"] = wells["ring"] == 0
    wells["plateID"] = str(plate)
    wells[schema.ROW_KEY] = [schema.row_id(int(r)) for r in wells["row_index"]]
    wells[schema.COLUMN_KEY] = [schema.column_id(int(c))
                                for c in wells["column_index"]]
    wells["well"] = [well_id(r, c) for r, c in
                     zip(wells["row_index"], wells["column_index"])]

    wells = wells[list(LAYOUT_COLUMNS)].sort_values(
        ["row_index", "column_index"], ignore_index=True)
    wells.attrs = {
        "plate": str(plate) if plate is not None else None,
        "value_col": None if grouping == "count" else value_col,
        "grouping": grouping,
        "min_count": int(min_count or 0),
        "plate_format": fmt,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "n_wells": int(len(wells)),
        "n_dropped_min_count": int(n_dropped),
        "n_unparsed_rows": int(n_unparsed),
        "n_plates": len(all_plates),
        "notes": tuple(notes),
    }
    return wells


def layout_matrix(layout: pd.DataFrame,
                  column: str = "value") -> pd.DataFrame:
    """Pivot a layout onto the full nominal plate grid.

    The result always spans every row and column of the inferred format,
    with ``NaN`` — never ``0`` — where a well is absent, so a heatmap
    cannot pass an empty well off as a measurement.

    :param layout: frame from :func:`plate_layout`.
    :param column: which layout column to pivot (``'value'`` or ``'n'``).
    :returns: DataFrame indexed by row letter (``A``, ``B``, …) with
        integer column labels ``1..n_cols``.
    """
    meta = dict(_DEFAULT_META)
    meta.update(getattr(layout, "attrs", {}) or {})
    n_rows = int(meta.get("n_rows") or 0)
    n_cols = int(meta.get("n_cols") or 0)
    if len(layout):
        n_rows = max(n_rows, int(layout["row_index"].max()))
        n_cols = max(n_cols, int(layout["column_index"].max()))
    grid = pd.DataFrame(
        np.full((n_rows, n_cols), np.nan, dtype=float),
        index=[_index_to_alpha(i) for i in range(1, n_rows + 1)],
        columns=list(range(1, n_cols + 1)))
    for r, c, v in zip(layout["row_index"], layout["column_index"],
                       layout[column]):
        grid.iat[int(r) - 1, int(c) - 1] = float(v) if pd.notna(v) else np.nan
    grid.attrs = meta
    return grid


def colour_limits(layout: pd.DataFrame,
                  min_max: Any = "allq") -> Tuple[float, float]:
    """Return ``(vmin, vmax)`` for a heatmap of ``layout``.

    Same specification language as
    :func:`spacr.plot.generate_plate_heatmap`: ``'all'`` for the full
    range, ``'allq'`` for the 2nd-98th percentile (which stops one dead
    well from flattening the whole plate to a single colour), or an
    explicit ``[vmin, vmax]`` — floats in ``[0, 1]`` are read as
    quantiles, anything else as absolute limits.

    Unlike the original, quantiles are computed over the *present* wells
    only; the original quantiles a grid where every absent well has
    already been turned into a zero.

    :param layout: frame from :func:`plate_layout`.
    :param min_max: colour-scale specification.
    :returns: ``(vmin, vmax)``, never degenerate.
    """
    values = pd.to_numeric(layout.get("value", pd.Series(dtype=float)),
                           errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return 0.0, 1.0
    if isinstance(min_max, (list, tuple)) and len(min_max) == 2:
        if all(isinstance(x, float) and 0.0 <= x <= 1.0 for x in min_max):
            vmin, vmax = (float(x) for x in np.quantile(values, list(min_max)))
        else:
            vmin, vmax = float(min_max[0]), float(min_max[1])
    elif min_max == "allq":
        vmin, vmax = (float(x) for x in np.quantile(values, [0.02, 0.98]))
    else:
        vmin, vmax = float(np.min(values)), float(np.max(values))
    if vmin > vmax:
        vmin, vmax = vmax, vmin
    if vmin == vmax:
        vmax = vmin + 1e-6
    return vmin, vmax


def write_layout_csv(layout: pd.DataFrame, path: str) -> str:
    """Write the well grid to ``path`` as CSV, one row per well.

    The tidy form is exported rather than the pivoted matrix: it carries
    the object count, the ring index and the edge flag alongside the
    value, which is what anybody re-analysing the plate outside spaCR
    actually needs.

    :param layout: frame from :func:`plate_layout`.
    :param path: destination file. Parent directories are created.
    :returns: the absolute path written.
    :raises ValueError: when ``path`` is empty.
    """
    if not path or not str(path).strip():
        raise ValueError("No output path given for the CSV export.")
    out = os.path.abspath(os.path.expanduser(str(path).strip()))
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    frame = layout[list(LAYOUT_COLUMNS)] if len(layout) else _empty_layout()
    frame.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)
    return out


# ---------------------------------------------------------------------------
# Row / column trends
# ---------------------------------------------------------------------------

def _spearman(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Two-sided Spearman of ``x`` vs ``y``, ``(rho, p)``, never NaN.

    Returns ``(None, None)`` when either side is constant — the
    correlation is genuinely undefined there, and scipy's NaN would be
    read as "no correlation" instead of "no answer".
    """
    if x.size < 3 or y.size < 3:
        return None, None
    if float(np.max(x) - np.min(x)) == 0 or float(np.max(y) - np.min(y)) == 0:
        return None, None
    res = _sps.spearmanr(x, y)
    return _finite(res.statistic), _finite(res.pvalue)


def row_column_trends(df: pd.DataFrame,
                      value_col: Optional[str] = None,
                      plate: Optional[str] = None,
                      grouping: str = "mean",
                      min_count: int = 0) -> pd.DataFrame:
    """Per-row and per-column summaries, plus the drift across each axis.

    One output row per plate row and per plate column, each carrying the
    number of wells behind it — a row average over three surviving wells
    is not the same claim as one over twenty-four, and the ``n_wells``
    column is what stops it being read as one.

    The Spearman statistic attached to each axis is computed over the
    individual *wells* (not over the 16 row means), because that is where
    the degrees of freedom are.

    :param df: long per-object frame, or a layout from
        :func:`plate_layout`.
    :param value_col: measurement to aggregate (see :func:`plate_layout`).
    :param plate: plate to summarise; ``None`` takes the first.
    :param grouping: per-well aggregation, one of :data:`GROUPINGS`.
    :param min_count: drop wells with fewer than this many objects.
    :returns: DataFrame with ``axis``, ``label``, ``index``, ``n_wells``,
        ``n_objects``, ``mean``, ``median``, ``std``,
        ``delta_vs_plate_median``, ``spearman_rho`` and ``spearman_p``.
        Empty (but correctly typed) when there are no wells.
    """
    layout = plate_layout(df, value_col=value_col, plate=plate,
                          grouping=grouping, min_count=min_count)
    columns = ["axis", "label", "index", "n_wells", "n_objects", "mean",
               "median", "std", "delta_vs_plate_median", "spearman_rho",
               "spearman_p"]
    if not len(layout):
        return pd.DataFrame({c: pd.Series(dtype=object) for c in columns})

    usable = layout.dropna(subset=["value"])
    if not len(usable):
        return pd.DataFrame({c: pd.Series(dtype=object) for c in columns})

    plate_median = float(np.median(usable["value"].to_numpy(dtype=float)))
    values = usable["value"].to_numpy(dtype=float)

    records: List[Dict[str, Any]] = []
    for axis, index_col in (("row", "row_index"), ("column", "column_index")):
        rho, p = _spearman(usable[index_col].to_numpy(dtype=float), values)
        for index, chunk in usable.groupby(index_col, observed=True):
            vals = chunk["value"].to_numpy(dtype=float)
            label = (_index_to_alpha(int(index)) if axis == "row"
                     else str(int(index)))
            records.append({
                "axis": axis,
                "label": label,
                "index": int(index),
                "n_wells": int(len(vals)),
                "n_objects": int(chunk["n"].sum()),
                "mean": _mean(vals),
                "median": _median(vals),
                "std": _finite(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "delta_vs_plate_median": _finite(np.median(vals) - plate_median),
                "spearman_rho": rho,
                "spearman_p": p,
            })
    out = pd.DataFrame(records, columns=columns)
    out.attrs = dict(getattr(layout, "attrs", {}) or {})
    return out


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

@dataclass
class RingStats:
    """One concentric ring of the plate, compared against its core.

    :ivar ring: 0 for the outermost ring, 1 for one well in, and so on.
    :ivar n_wells: wells surviving ``min_count`` in this ring.
    :ivar median: median well value in the ring.
    :ivar mean: mean well value in the ring.
    :ivar delta: ``median(ring) - median(core)``.
    :ivar pct: ``delta`` as a percentage of the core median, or ``None``
        when the core median is zero.
    :ivar p_value: two-sided Mann-Whitney U against the core.
    :ivar cliffs_delta: rank effect size in ``[-1, 1]``; positive means
        this ring reads higher than the core.
    """
    ring: int
    n_wells: int
    median: Optional[float]
    mean: Optional[float]
    delta: Optional[float]
    pct: Optional[float]
    p_value: Optional[float]
    cliffs_delta: Optional[float]


@dataclass
class GradientStats:
    """A monotonic drift along one axis of the plate.

    :ivar axis: ``'row'`` or ``'column'``.
    :ivar spearman_rho: rank correlation of well value with the axis
        index; ``None`` when undefined (constant values, <3 wells).
    :ivar p_value: two-sided p for ``spearman_rho``.
    :ivar first_label: label of the lowest-index row/column present.
    :ivar last_label: label of the highest-index row/column present.
    :ivar delta_first_last: median of the last minus median of the first.
    :ivar pct_first_last: that difference as a percentage of the first.
    :ivar detected: both ``p_value`` and ``|rho|`` cleared their
        thresholds.
    """
    axis: str
    spearman_rho: Optional[float]
    p_value: Optional[float]
    first_label: str
    last_label: str
    delta_first_last: Optional[float]
    pct_first_last: Optional[float]
    detected: bool


@dataclass
class EdgeEffectReport:
    """Everything :func:`detect_edge_effect` worked out, in one object.

    The headline numbers are :attr:`pct_difference` and
    :attr:`cliffs_delta` — "the outer ring reads 31 % higher, δ = 0.78" —
    with :attr:`p_value` as supporting evidence rather than the verdict.

    :ivar ok: False when the plate could not be tested at all (empty
        frame, one well, no interior); :attr:`notes` says why.
    :ivar edge_detected: the outer ring differs from the interior by more
        than :attr:`min_effect`, at better than :attr:`alpha`.
    :ivar gradient_detected: at least one axis shows a monotonic drift.
    :ivar dominant: ``'edge'``, ``'gradient'``, or ``'none'`` — which
        pattern better explains the plate.
    :ivar rings: ring-by-ring profile, outermost first.
    :ivar gradients: one :class:`GradientStats` per axis.
    :ivar n_dropped_min_count: wells removed by ``min_count``. A heatmap
        missing a third of its wells looks like data; this is the number
        that says it isn't.
    """
    plate: Optional[str]
    value_col: Optional[str]
    grouping: str
    ok: bool = False
    plate_format: Optional[int] = None
    n_rows: int = 0
    n_cols: int = 0
    n_wells: int = 0
    n_edge_wells: int = 0
    n_interior_wells: int = 0
    n_dropped_min_count: int = 0
    min_count: int = 0
    edge_detected: bool = False
    p_value: Optional[float] = None
    cliffs_delta: Optional[float] = None
    edge_median: Optional[float] = None
    interior_median: Optional[float] = None
    median_difference: Optional[float] = None
    pct_difference: Optional[float] = None
    rings: List[RingStats] = field(default_factory=list)
    gradients: List[GradientStats] = field(default_factory=list)
    gradient_detected: bool = False
    dominant: str = "none"
    alpha: float = DEFAULT_ALPHA
    min_effect: float = DEFAULT_MIN_EFFECT
    min_gradient_rho: float = DEFAULT_MIN_GRADIENT_RHO
    notes: List[str] = field(default_factory=list)

    def gradient(self, axis: str) -> Optional[GradientStats]:
        """Return the :class:`GradientStats` for ``'row'`` or ``'column'``."""
        for g in self.gradients:
            if g.axis == axis:
                return g
        return None

    def ring(self, index: int) -> Optional[RingStats]:
        """Return the :class:`RingStats` for ring ``index``, if computed."""
        for r in self.rings:
            if r.ring == index:
                return r
        return None

    @property
    def magnitude(self) -> str:
        """The edge difference in the most meaningful units available.

        A percentage where the interior median is non-zero, absolute
        units where it is not — "+300 % of nothing" is not a number to
        put in front of a user.
        """
        if self.pct_difference is not None:
            return f"{self.pct_difference:+.1f} %"
        if self.median_difference is not None:
            return (f"{_fmt_num(self.median_difference)} in absolute units "
                    f"(the interior median is zero)")
        return "an undetermined amount"

    @property
    def summary(self) -> str:
        """One-line verdict, effect size first."""
        if not self.ok:
            return self.notes[0] if self.notes else "Nothing to test."
        if self.edge_detected:
            return (f"Edge effect: the outer ring reads {self.magnitude} vs "
                    f"the interior (δ = {_fmt_num(self.cliffs_delta)}, "
                    f"p = {_fmt_p(self.p_value)}).")
        if self.gradient_detected:
            worst = max((g for g in self.gradients
                         if g.detected and g.spearman_rho is not None),
                        key=lambda g: abs(g.spearman_rho), default=None)
            axis = worst.axis if worst else "row/column"
            rho = _fmt_num(worst.spearman_rho) if worst else "?"
            return (f"No edge effect, but a monotonic {axis} gradient "
                    f"(Spearman rho = {rho}).")
        return (f"No edge effect: the outer ring is within "
                f"{self.magnitude} of the interior "
                f"(δ = {_fmt_num(self.cliffs_delta)}).")


def detect_edge_effect(df: pd.DataFrame,
                       value_col: Optional[str] = None,
                       plate: Optional[str] = None,
                       grouping: str = "mean",
                       min_count: int = 0,
                       alpha: float = DEFAULT_ALPHA,
                       min_effect: float = DEFAULT_MIN_EFFECT,
                       min_gradient_rho: float = DEFAULT_MIN_GRADIENT_RHO,
                       core_depth: int = DEFAULT_CORE_DEPTH,
                       max_rings: int = DEFAULT_MAX_RINGS,
                       plate_format: Optional[int] = None) -> EdgeEffectReport:
    """Test a plate for an edge artefact and for a row/column gradient.

    The outer ring is compared against everything inside it with a
    two-sided Mann-Whitney U — rank-based, because per-well aggregates
    of object measurements are skewed often enough that a t-test would be
    testing its own assumptions rather than the plate. The accompanying
    Cliff's delta comes off the same U statistic, so effect size and
    p-value cannot disagree about which comparison was made.

    Detection deliberately needs *both* ``p < alpha`` **and**
    ``|cliffs_delta| >= min_effect``. On 384 wells a p-value on its own
    flags drifts far below the level at which anyone would act.

    The ring-by-ring profile then walks inward, comparing each ring
    against the plate core (depth >= ``core_depth``), because evaporation
    reaches past the outermost well and a single outer-vs-rest test would
    average that away.

    Row and column gradients are tested separately with Spearman on the
    well index. This is what keeps a plate-reader or incubator gradient
    from being reported as an edge effect: a linear gradient across the
    plate leaves the outer ring straddling both extremes, so its median
    matches the interior and the ring test correctly stays quiet.

    :param df: long per-object frame, or a layout from
        :func:`plate_layout`.
    :param value_col: measurement to aggregate per well.
    :param plate: plate to test; ``None`` takes the first present.
    :param grouping: per-well aggregation, one of :data:`GROUPINGS`.
    :param min_count: drop wells with fewer than this many objects.
    :param alpha: significance threshold. Default :data:`DEFAULT_ALPHA`.
    :param min_effect: minimum ``|Cliff's delta|`` to call an edge
        effect. Default :data:`DEFAULT_MIN_EFFECT`.
    :param min_gradient_rho: minimum ``|Spearman rho|`` to call a
        gradient. Default :data:`DEFAULT_MIN_GRADIENT_RHO`.
    :param core_depth: ring depth defining "the core" for the profile.
    :param max_rings: how many rings inward to profile.
    :param plate_format: force 96 / 384 / 1536 instead of inferring.
    :returns: an :class:`EdgeEffectReport`. Degenerate input comes back
        with ``ok=False`` and an explanation in ``notes`` — never a NaN
        and never an exception.

    Example:
        .. code-block:: python

            from spacr.plate_qc import detect_edge_effect, format_edge_report
            report = detect_edge_effect(df, 'cell_area', min_count=20)
            print(format_edge_report(report))

    See Also:
        :func:`row_column_trends` — the per-row/per-column detail behind
        the gradient statistics.
    """
    layout = plate_layout(df, value_col=value_col, plate=plate,
                          grouping=grouping, min_count=min_count,
                          plate_format=plate_format)
    meta = dict(_DEFAULT_META)
    meta.update(getattr(layout, "attrs", {}) or {})

    report = EdgeEffectReport(
        plate=meta.get("plate"),
        value_col=meta.get("value_col") if grouping != "count" else None,
        grouping=grouping,
        plate_format=meta.get("plate_format"),
        n_rows=int(meta.get("n_rows") or 0),
        n_cols=int(meta.get("n_cols") or 0),
        n_dropped_min_count=int(meta.get("n_dropped_min_count") or 0),
        min_count=int(meta.get("min_count") or 0),
        alpha=float(alpha),
        min_effect=float(min_effect),
        min_gradient_rho=float(min_gradient_rho),
        notes=list(meta.get("notes") or ()),
    )

    usable = layout.dropna(subset=["value"]) if len(layout) else layout
    n_nan = len(layout) - len(usable)
    if n_nan:
        report.notes.insert(
            0, f"{n_nan} well(s) had no usable {value_col!r} value and were "
               f"left blank.")
    report.n_wells = int(len(usable))

    if report.n_wells == 0:
        report.notes.insert(
            0, "No wells to test — the frame is empty after filtering.")
        return report
    if report.n_wells == 1:
        only = usable.iloc[0]
        report.notes.insert(
            0, f"Only one well ({only['well']}) survived filtering — an edge "
               f"effect needs an outer ring and an interior to compare.")
        return report

    values = usable["value"].to_numpy(dtype=float)
    ring_index = usable["ring"].to_numpy(dtype=int)
    constant = float(np.max(values) - np.min(values)) == 0.0
    if constant:
        report.notes.insert(
            0, f"Every one of the {report.n_wells} wells reads "
               f"{_fmt_num(_finite(values[0]))} — there is no variation to "
               f"attribute to the edge or to anything else.")

    outer = values[ring_index == 0]
    interior = values[ring_index >= 1]
    report.n_edge_wells = int(outer.size)
    report.n_interior_wells = int(interior.size)
    report.edge_median = _median(outer)
    report.interior_median = _median(interior)

    if outer.size == 0 or interior.size == 0:
        missing = "outer ring" if outer.size == 0 else "interior"
        report.notes.insert(
            0, f"This plate has no {missing} on a "
               f"{report.n_rows}x{report.n_cols} grid "
               f"({report.n_edge_wells} edge / {report.n_interior_wells} "
               f"interior wells), so the two cannot be compared.")
        report.ok = False
    else:
        report.ok = True
        p, delta = _rank_compare(outer, interior)
        report.p_value = p
        report.cliffs_delta = delta
        if report.edge_median is not None and report.interior_median is not None:
            report.median_difference = _finite(
                report.edge_median - report.interior_median)
        report.pct_difference = _pct_of(report.median_difference,
                                        report.interior_median)
        if report.pct_difference is None and report.median_difference is not None:
            report.notes.append(
                "The interior median is zero, so the edge difference is "
                "reported in absolute units only.")
        report.edge_detected = bool(
            p is not None and delta is not None
            and p < alpha and abs(delta) >= min_effect)

    # -- ring-by-ring profile ---------------------------------------------
    report.rings = _ring_profile(values, ring_index, core_depth, max_rings,
                                 report)

    # -- gradients ---------------------------------------------------------
    report.gradients = _gradient_profile(usable, values, alpha,
                                         min_gradient_rho)
    report.gradient_detected = any(g.detected for g in report.gradients)

    # -- which pattern wins ------------------------------------------------
    report.dominant = _dominant(report)
    return report


def _ring_profile(values: np.ndarray, ring_index: np.ndarray,
                  core_depth: int, max_rings: int,
                  report: EdgeEffectReport) -> List[RingStats]:
    """Compare each ring against the plate core, outermost first."""
    depth = int(core_depth)
    core = values[ring_index >= depth]
    while core.size < 3 and depth > 0:
        depth -= 1
        core = values[ring_index >= depth]
    if core.size == 0:
        return []
    if depth != int(core_depth):
        report.notes.append(
            f"Too few wells at least {int(core_depth)} in from the edge; the "
            f"ring profile uses depth >= {depth} as the core.")
    core_median = _median(core)

    out: List[RingStats] = []
    max_ring = int(ring_index.max()) if ring_index.size else 0
    for k in range(min(int(max_rings), max_ring + 1)):
        if k >= depth:
            # This ring is part of the core; comparing it against itself
            # would be circular, so the profile stops here.
            break
        vals = values[ring_index == k]
        if vals.size == 0:
            continue
        p, delta = _rank_compare(vals, core)
        median = _median(vals)
        diff = (_finite(median - core_median)
                if median is not None and core_median is not None else None)
        out.append(RingStats(
            ring=k, n_wells=int(vals.size), median=median, mean=_mean(vals),
            delta=diff, pct=_pct_of(diff, core_median),
            p_value=p, cliffs_delta=delta))
    return out


def _gradient_profile(usable: pd.DataFrame, values: np.ndarray,
                      alpha: float,
                      min_rho: float) -> List[GradientStats]:
    """Spearman drift along each axis, plus the first-vs-last difference."""
    out: List[GradientStats] = []
    for axis, index_col in (("row", "row_index"), ("column", "column_index")):
        idx = usable[index_col].to_numpy(dtype=float)
        rho, p = _spearman(idx, values)
        lo, hi = int(np.min(idx)), int(np.max(idx))
        first = values[idx == lo]
        last = values[idx == hi]
        first_median, last_median = _median(first), _median(last)
        delta = (_finite(last_median - first_median)
                 if first_median is not None and last_median is not None
                 else None)
        label = (_index_to_alpha if axis == "row" else str)
        out.append(GradientStats(
            axis=axis,
            spearman_rho=rho,
            p_value=p,
            first_label=label(lo),
            last_label=label(hi),
            delta_first_last=delta,
            pct_first_last=_pct_of(delta, first_median),
            detected=bool(rho is not None and p is not None
                          and p < alpha and abs(rho) >= min_rho),
        ))
    return out


def _dominant(report: EdgeEffectReport) -> str:
    """Name the pattern that better explains the plate.

    Both candidate statistics are rank-based and live on ``[-1, 1]``, so
    comparing ``|Cliff's delta|`` against ``|Spearman rho|`` compares like
    with like — neither is scaled by the units of the measurement.
    """
    edge_mag = abs(report.cliffs_delta) if (
        report.edge_detected and report.cliffs_delta is not None) else 0.0
    rhos = [abs(g.spearman_rho) for g in report.gradients
            if g.detected and g.spearman_rho is not None]
    grad_mag = max(rhos) if rhos else 0.0
    if edge_mag == 0.0 and grad_mag == 0.0:
        return "none"
    return "edge" if edge_mag >= grad_mag else "gradient"


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

def _fmt_num(value: Optional[float], places: int = 3) -> str:
    """Format a statistic, or say it is undefined — never print ``nan``."""
    if value is None:
        return "undefined"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e5 or magnitude < 1e-3:
        return f"{value:.{places}g}"
    return f"{value:.{places}f}".rstrip("0").rstrip(".")


def _fmt_pct(value: Optional[float]) -> str:
    """Signed percentage, or ``"undefined"`` when the baseline is zero."""
    if value is None:
        return "undefined"
    return f"{value:+.1f} %"


def _fmt_p(value: Optional[float]) -> str:
    if value is None:
        return "undefined"
    if value < 1e-4:
        return "< 1e-4"
    return f"{value:.4g}"


def format_edge_report(report: EdgeEffectReport) -> str:
    """Render an :class:`EdgeEffectReport` as text, effect size first.

    The layout is deliberate: the verdict and the percentage difference
    come before the p-value, because on a 384-well plate the p-value is
    the least informative number in the report.

    :param report: the report to render.
    :returns: a multi-line string, safe to drop into a label or a log.
    """
    lines: List[str] = []
    header = f"Plate {report.plate}" if report.plate else "Plate"
    what = (report.value_col if report.value_col
            else f"objects per well ({report.grouping})")
    if report.value_col:
        what = f"{report.grouping} {report.value_col}"
    fmt = (f"{report.plate_format}-well" if report.plate_format
           else f"{report.n_rows}x{report.n_cols} (non-standard)")
    lines.append(f"{header} — {what}")
    lines.append(f"{fmt} plate, {report.n_wells} wells tested "
                 f"({report.n_edge_wells} edge, "
                 f"{report.n_interior_wells} interior).")
    if report.n_dropped_min_count:
        lines.append(
            f"{report.n_dropped_min_count} well(s) dropped for holding fewer "
            f"than {report.min_count} objects — blank on the map, not zero.")
    lines.append("")
    lines.append(report.summary)

    if report.ok:
        lines.append("")
        lines.append("Outer ring vs interior")
        lines.append(f"  median difference   {_fmt_num(report.median_difference)}"
                     f"  ({_fmt_pct(report.pct_difference)})")
        lines.append(f"  Cliff's delta       {_fmt_num(report.cliffs_delta)}"
                     f"   [-1, 1]; |δ| >= {report.min_effect} to flag")
        lines.append(f"  Mann-Whitney p      {_fmt_p(report.p_value)}"
                     f"   (two-sided, rank-based)")
        lines.append(f"  edge median         {_fmt_num(report.edge_median)}")
        lines.append(f"  interior median     {_fmt_num(report.interior_median)}")

    if report.rings:
        lines.append("")
        lines.append("Ring profile (0 = outermost, vs the plate core)")
        lines.append("  ring  wells    median      vs core          δ        p")
        for r in report.rings:
            lines.append(
                f"  {r.ring:>4}  {r.n_wells:>5}  {_fmt_num(r.median):>9}  "
                f"{_fmt_pct(r.pct):>13}  {_fmt_num(r.cliffs_delta):>7}  "
                f"{_fmt_p(r.p_value):>7}")

    if report.gradients:
        lines.append("")
        lines.append("Monotonic gradients (Spearman on the well index)")
        for g in report.gradients:
            flag = "  <-- gradient" if g.detected else ""
            lines.append(
                f"  {g.axis:<7} rho = {_fmt_num(g.spearman_rho):>7}  "
                f"p = {_fmt_p(g.p_value):>8}  "
                f"{g.first_label}->{g.last_label} "
                f"{_fmt_pct(g.pct_first_last)}{flag}")
        lines.append(f"  (|rho| >= {report.min_gradient_rho} and "
                     f"p < {report.alpha} to flag. A gradient runs across the "
                     f"plate; an edge effect runs around it.)")

    if report.ok:
        lines.append("")
        lines.append(f"Dominant pattern: {report.dominant}")

    if report.notes:
        lines.append("")
        lines.append("Notes")
        for note in report.notes:
            lines.append(f"  - {note}")
    return "\n".join(lines)
