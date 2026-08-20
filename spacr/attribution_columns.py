"""Write a guide attribution into ``png_list`` -- only when asked.

Instruction 173's last owed piece:

    An OPT-IN write of the columns into `png_list`:
        grna_attributed, grna_attributed_p, gene_attributed,
        grna_attribution_entropy, grna_attribution_coverage
    Never automatic. Writing into measurements.db is not something a viewer
    does behind the user, which is the rule the montage already keeps.

IT IS NOT A GENOTYPE, AND THE COLUMN NAMES SAY SO. The pooled design never
observed which cell carried what; this is an attribution under a model, and a
reader who takes `grna_attributed` for an observation has been misled by the
name alone. `_attributed` is in every one of them for that reason, and
`ATTRIBUTION_NOTE` is written into the database beside them so the claim
travels with the data rather than living in a docstring.

NOTHING HERE IMPORTS QT. The confirmation is the caller's to obtain.
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

#: Recorded in the database so the claim travels with the numbers.
ATTRIBUTION_NOTE = (
    "grna_attributed is an ATTRIBUTION UNDER A MODEL, not an observation. "
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
    """The write could not be done, and the message says what stopped it."""


def _columns_of(cursor, table: str) -> set:
    return {row[1] for row in cursor.execute(f"PRAGMA table_info('{table}')")}


def describe(rows: Sequence[Mapping]) -> str:
    """What a write would do, for the confirmation the caller must obtain."""
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
    """Write the attribution columns for ``rows`` into ``db_path``.

    :param rows: mappings carrying ``key_column`` and any of :data:`COLUMNS`.
    :param key_column: how a row is matched to ``png_list``. ``prcfo`` is the
        per-object key every spaCR database carries; ``file_name`` also works
        for a database written before it.
    :param confirmed: MUST be True. The default refuses, because "never
        automatic" is a property of this function rather than of every caller
        remembering -- a viewer that wrote into a measurements database
        behind the user is the failure this guards.
    :returns: ``{"matched": n, "added": n_columns}``.
    :raises AttributionWriteError: unconfirmed, no such table, or no key.
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

    with sqlite3.connect(str(db_path)) as connection:
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
    """Turn :class:`~spacr.guide_attribution.Attribution` objects into rows.

    :param attributions: one per cell, in the order ``keys`` came.
    :param keys: the per-object key each attribution belongs to.
    :param genes: ``{guide: gene}``. Absent, the gene is the guide's own
        prefix, which is spaCR's rule everywhere else.
    :param coverage: what share of the well's reads the attributed guides
        covered -- the same number for every cell of a well, and recorded
        per cell so a reader filtering the table keeps it.
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
