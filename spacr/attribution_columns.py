"""Persist model-based guide attributions in ``png_list`` on explicit request.

The pooled design does not observe which guide each individual cell carries.
These columns therefore store inferred attributions, their probabilities, and
coverage diagnostics rather than genotypes. A database note records that
interpretation beside the values. The module is GUI-independent and requires
the caller to confirm every write.
"""
from __future__ import annotations

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

    from . import store

    with store.open(str(db_path), verify="none") as reader:
        tables = set(reader.tables())
        if "png_list" not in tables:
            raise AttributionWriteError(
                f"{db_path} has no png_list table, so there is nothing to "
                f"attribute to. Tables here: {sorted(tables)}.")
        present = set(reader.columns("png_list")["name"])
    if key_column not in present:
        raise AttributionWriteError(
            f"png_list has no {key_column!r} column to match on. It "
            f"carries {sorted(present)}.")

    added = len(store.ensure_columns(str(db_path), "png_list", COLUMNS,
                                     quoted=False, timeout=30.0))
    names = [name for name in COLUMNS
             if any(name in record for record in records)]
    rows = [{"key": record.get(key_column),
             **{name: record.get(name) for name in names}}
            for record in records if record.get(key_column) is not None]
    matched = store.edit(str(db_path), rows, kind="attribution",
                         source="attribution", table="png_list",
                         columns=names, key_column=key_column,
                         types=COLUMNS, timeout=30.0) if names else 0
    if "settings" in tables:
        store.edit(str(db_path), [{"key": NOTE_KEY,
                                   "value": ATTRIBUTION_NOTE}],
                   kind="attribution", source="attribution",
                   table="settings", column="setting_value",
                   key_column="setting_key", upsert=True, timeout=30.0)
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
