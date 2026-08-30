"""Persist model-based guide attributions in ``png_list`` on explicit request.

The pooled design does not observe which guide each individual cell carries.
These columns therefore store inferred attributions, their probabilities, and
coverage diagnostics rather than genotypes. A database note records that
interpretation beside the values. The module is GUI-independent and requires
the caller to confirm every write.
"""
from __future__ import annotations

import sqlite3
from typing import Dict, Iterable, Mapping, Optional, Sequence

#: The columns written, and the SQLite type each takes.
COLUMNS = {
    "grna_attributed": "TEXT",
    "grna_attributed_p": "REAL",
    "gene_attributed": "TEXT",
    "grna_attribution_entropy": "REAL",
    "grna_attribution_coverage": "REAL",
}

#: Interpretation stored beside attributed values in the database.
ATTRIBUTION_NOTE = (
    "grna_attributed is a model-based attribution, not an observation. "
    "This is a pooled screen: the sequencing reports what fraction of each "
    "well carried each guide, never which cells did. The value is the guide "
    "whose posterior was highest for that cell given the well's read "
    "fractions and the fitted effects, and grna_attributed_p is that "
    "posterior. A cell whose best posterior fell under the threshold is "
    "'ambiguous' and carries the posterior anyway."
)

#: Where the note is kept. The settings table is already a key/value store
#: every spaCR database has.
NOTE_KEY = "grna_attribution_note"


class AttributionWriteError(RuntimeError):
    """Raised when an attribution write cannot be completed safely."""


def _columns_of(cursor, table: str) -> set:
    return {row[1] for row in cursor.execute(f"PRAGMA table_info('{table}')")}


def describe(rows: Sequence[Mapping]) -> str:
    """Summarize a proposed attribution write for user confirmation.

    :param rows: proposed per-cell attribution rows.
    """
    total = len(rows)
    called = sum(1 for row in rows
                 if str(row.get("grna_attributed") or "") not in
                 ("", "ambiguous"))
    return (f"{total} cell(s): {called} attributed to a guide and "
            f"{total - called} left ambiguous. Five columns are added to "
            f"png_list. They are an attribution under a model, not a "
            f"genotype.")


def write(db_path: str, rows: Iterable[Mapping], *,
          key_column: str = "prcfo",
          confirmed: bool = False) -> Dict[str, int]:
    """Write model-based cell attributions to a ``png_list`` table.

    Parameters
    ----------
    db_path : str
        Measurement database containing ``png_list``.
    rows : iterable of mappings
        Records carrying ``key_column`` and any fields in :data:`COLUMNS`.
    key_column : str, default='prcfo'
        Column used to match records to ``png_list``. ``file_name`` supports
        databases created before the per-object ``prcfo`` key was available.
    confirmed : bool, default=False
        Must be ``True``. The function refuses automatic or implicit writes.

    Returns
    -------
    dict
        Counts of matched rows and newly added columns.

    Raises
    ------
    AttributionWriteError
        If confirmation is absent, ``png_list`` is missing, or the match key
        is unavailable.
    """
    if not confirmed:
        raise AttributionWriteError(
            "writing an attribution into png_list is opt-in: pass "
            "confirmed=True once the user has agreed. Writing into a "
            "measurements database is not something a viewer does behind "
            "them.")
    records = [dict(row) for row in rows]
    if not records:
        return {"matched": 0, "added": 0}

    with sqlite3.connect(str(db_path), timeout=30.0) as connection:
        cursor = connection.cursor()
        tables = {row[0] for row in cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "png_list" not in tables:
            raise AttributionWriteError(
                f"{db_path} has no png_list table, so there is nothing to "
                f"attribute to. Tables here: {sorted(tables)}.")
        present = _columns_of(cursor, "png_list")
        if key_column not in present:
            raise AttributionWriteError(
                f"png_list has no {key_column!r} column to match on. It "
                f"carries {sorted(present)}.")

        added = 0
        for name, kind in COLUMNS.items():
            if name not in present:
                # SQLite has no ADD COLUMN IF NOT EXISTS, and re-running a
                # write must not be an error -- an attribution is something
                # a user redoes with a different threshold.
                cursor.execute(
                    f"ALTER TABLE png_list ADD COLUMN {name} {kind}")
                added += 1

        names = [name for name in COLUMNS
                 if any(name in record for record in records)]
        assignments = ", ".join(f"{name} = ?" for name in names)
        matched = 0
        for record in records:
            key = record.get(key_column)
            if key is None:
                continue
            values = [record.get(name) for name in names]
            cursor.execute(
                f"UPDATE png_list SET {assignments} WHERE {key_column} = ?",
                values + [key])
            matched += cursor.rowcount

        if "settings" in tables:
            cursor.execute(
                "INSERT OR REPLACE INTO settings (setting_key, setting_value)"
                " VALUES (?, ?)", (NOTE_KEY, ATTRIBUTION_NOTE))
        connection.commit()
    return {"matched": int(matched), "added": int(added)}


def rows_from(attributions, keys, *, genes=None,
              coverage: Optional[float] = None) -> list:
    """Convert guide-attribution results into database records.

    Parameters
    ----------
    attributions : iterable
        One attribution per cell, in the same order as ``keys``.
    keys : iterable
        Per-object database keys.
    genes : mapping, optional
        Guide-to-gene mapping. Without one, the guide prefix is used.
    coverage : float, optional
        Fraction of well reads covered by attributed guides. The well-level
        value is repeated per cell so it remains available after filtering.

    Returns
    -------
    list of dict
        Records ready for :func:`write`.
    """
    out = []
    for call, key in zip(attributions, keys):
        guide = str(getattr(call, "guide", "") or "")
        gene = ""
        if guide and guide != "ambiguous":
            gene = (genes or {}).get(guide) or guide.rsplit("_", 1)[0]
        out.append({
            "prcfo": key,
            "grna_attributed": guide,
            "grna_attributed_p": float(getattr(call, "probability", 0.0)),
            "gene_attributed": gene,
            "grna_attribution_entropy": float(getattr(call, "entropy", 0.0)),
            "grna_attribution_coverage": (None if coverage is None
                                          else float(coverage)),
        })
    return out
