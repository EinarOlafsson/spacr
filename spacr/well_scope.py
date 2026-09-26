"""Select object populations for cell-table visualizations.

Plots can show objects assigned to selected guides, those objects together
with other objects from the same wells, or the complete table. The well set is
derived from the guide selection and can then be restricted explicitly.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

LOG = logging.getLogger("spacr.well_scope")

#: The three populations, as ``(value, caption)``.
SCOPES: Tuple[Tuple[str, str], ...] = (
    ("guides", "gRNAs"),
    ("wells", "gRNAs + other datapoints in selected wells"),
    ("all", "All datapoints"),
)

#: Column distinguishing selected-guide objects from other objects in their wells.
MATE_COLUMN = "in_selection"

#: What each side is called on the plot.
CHOSEN, MATE = "chosen gRNA", "well-mate"

#: Columns an object table might carry its well under, most canonical first.
WELL_COLUMNS: Tuple[str, ...] = ("prc", "well", "wellID")

#: Columns an object table might carry its guide under.
GUIDE_COLUMNS: Tuple[str, ...] = ("grna", "grna_name", "gRNA", "guide",
                                  "feature")


def _column(frame: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    """Return the first candidate column present in a table.

    :param frame: Table, or table-like object, whose columns are inspected.
    :param names: Candidate names in preference order.
    :returns: First present candidate, or ``None`` when none are available.
    """
    return next((c for c in names if c in getattr(frame, "columns", ())),
                None)


def wells_of(frame: pd.DataFrame, guides: Sequence[str]) -> List[str]:
    """Return wells containing the selected guides in table order.

    Parameters
    ----------
    frame : pandas.DataFrame
        Object table containing guide and well identifiers.
    guides : sequence of str
        Guide identifiers selected for display.

    Returns
    -------
    list of str
        Unique well identifiers in their first-occurrence order. An empty list
        is returned when required columns or guide selections are absent.
    """
    well = _column(frame, WELL_COLUMNS)
    guide = _column(frame, GUIDE_COLUMNS)
    if well is None or guide is None or not len(guides):
        return []
    wanted = {str(g) for g in guides}
    hit = frame[frame[guide].astype(str).isin(wanted)]
    seen, out = set(), []
    for value in hit[well].astype(str):
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


TOP_HITS = 5

TOP_HITS_ALPHA = 0.05

_RESULT_NAMES: Tuple[str, ...] = ("feature", "coefficient_name", "term",
                                  "name", "grna", "guide", "gene")

_RESULT_P: Tuple[str, ...] = ("p_value", "pvalue", "p-value", "P>|t|",
                              "P>|z|", "p")


def _read_results(results: Any) -> Optional[pd.DataFrame]:
    """A regression results table from a frame, a CSV or a run folder.

    A folder is read for ``results.csv``, then the guide and gene tables
    (:data:`spacr.hits.RESULT_FILES`). ``None`` when nothing is readable.
    """
    if results is None:
        return None
    if isinstance(results, pd.DataFrame):
        return results
    path = os.fspath(results)
    if path and os.path.isdir(path):
        from .hits import RESULT_FILES

        inside = [os.path.join(path, RESULT_FILES[role])
                  for role in ("all", "grna", "gene")]
        path = next((f for f in inside if os.path.isfile(f)), "")
    if not path or not os.path.isfile(path):
        return None
    from .tabular import read_table

    try:
        return read_table(path, canonicalise=False, report=None)
    except Exception:
        LOG.debug("could not read results %s", path, exc_info=True)
        return None


def _top_hits(frame: pd.DataFrame, results: Any, *,
             alpha: float = TOP_HITS_ALPHA,
             limit: int = TOP_HITS) -> List[str]:
    """Return the guides of the most significant hits, as ``frame`` spells them.

    Decision 2026-09-25 (item 205): "the regression Compare panel opens with
    the TOP HITS pre-selected (top significant guides/genes and the wells
    that carry them; user can change it)".

    Parameters
    ----------
    frame : pandas.DataFrame
        Object table carrying a guide column.
    results : pandas.DataFrame or path-like
        The regression's coefficient table: a term column (``feature`` in
        spaCR's own output) and a p-value column.
    alpha : float, default=0.05
        Only terms with ``p <= alpha`` count as hits.
    limit : int, default=:data:`TOP_HITS`
        How many hits, most significant first, contribute their guides. A
        hit whose guides are not in ``frame`` is skipped and does not use up
        a place.

    Returns
    -------
    list of str
        Guides in ``frame``'s own spelling, in table order. A guide-level
        term contributes its one guide; a gene-level term contributes every
        guide of that gene. Empty when nothing is significant, the table
        cannot be read, or ``frame`` names no guide.
    """
    from .hits import _gene_id_of, gene_of

    table = _read_results(results)
    guide = _column(frame, GUIDE_COLUMNS)
    if table is None or not len(table) or guide is None or not len(frame):
        return []
    name = _column(table, _RESULT_NAMES)
    p_col = _column(table, _RESULT_P)
    if name is None or p_col is None:
        return []
    p_values = pd.to_numeric(table[p_col], errors="coerce")
    ranked = (table.assign(_p=p_values).dropna(subset=["_p"])
              .loc[lambda t: t["_p"] <= float(alpha)]
              .sort_values("_p", kind="stable"))
    spelled = frame[guide].astype(str)
    unique = list(dict.fromkeys(spelled))
    genes = {g: _gene_id_of(g) for g in unique}
    wanted: set = set()
    used = 0
    for term in ranked[name].astype(str):
        if used >= int(limit):
            break
        match = re.search(r"\[(?:T\.)?(.*?)\]", term)
        token = match.group(1) if match else term
        gene = gene_of(term) or _gene_id_of(token)
        if token in genes:
            hits = {token}
        elif gene and gene != token:
            hits = {g for g in unique if g == token or g.endswith("_" + token)}
        else:
            hits = {g for g in unique if genes[g] == gene}
        hits -= wanted
        if hits:
            wanted |= hits
            used += 1
    return [g for g in unique if g in wanted]


def _guides_at(frame: pd.DataFrame, members: Sequence[Any]) -> List[str]:
    """Return the guides of the rows whose index is in ``members``.

    :param frame: object table carrying a guide column.
    :param members: object-index values, e.g. one montage group's.
    :returns: guides in table order; empty when ``frame`` names no guide.
    """
    guide = _column(frame, GUIDE_COLUMNS)
    if guide is None or frame is None or not len(frame):
        return []
    rows = frame[frame.index.isin(list(members))]
    return list(dict.fromkeys(rows[guide].astype(str)))


def select(frame: pd.DataFrame, *, scope: str = "guides",
           guides: Sequence[str] = (),
           wells: Optional[Sequence[str]] = None
           ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Select the object population requested by a plot.

    Parameters
    ----------
    frame : pandas.DataFrame
        Object table containing guide and well identifiers.
    scope : {"guides", "wells", "all"}, default="guides"
        Population to retain.
    guides : sequence of str, optional
        Selected guide identifiers.
    wells : sequence of str, optional
        Explicit well subset for ``scope="wells"``. When omitted, wells are
        derived from ``guides``.

    Returns
    -------
    pandas.DataFrame
        Selected rows. The well scope adds :data:`MATE_COLUMN` to distinguish
        selected-guide objects from their well-mates.
    dict
        Scope, selections, row counts, and any explanatory note.

    Raises
    ------
    ValueError
        If ``scope`` is not supported.
    """
    scope = str(scope or "guides").strip().lower()
    if scope not in {value for value, _ in SCOPES}:
        raise ValueError(
            f"unknown scope {scope!r}; this offers "
            f"{', '.join(v for v, _ in SCOPES)}")
    report: Dict[str, Any] = {"scope": scope, "wells": [], "guides":
                              [str(g) for g in guides], "rows": 0,
                              "chosen": 0, "mates": 0}
    if frame is None or not len(frame):
        return (frame if frame is not None else pd.DataFrame()), report

    if scope == "all":
        report["rows"] = int(len(frame))
        report["chosen"] = int(len(frame))
        return frame, report

    guide = _column(frame, GUIDE_COLUMNS)
    if guide is None or not len(guides):
        report["note"] = ("no gRNA was chosen, so there is no population to "
                          "draw; pick one or more points on the volcano")
        return frame.iloc[0:0], report

    wanted = {str(g) for g in guides}
    chosen = frame[guide].astype(str).isin(wanted)
    if scope == "guides":
        out = frame[chosen]
        report.update(rows=int(len(out)), chosen=int(len(out)))
        return out, report

    well = _column(frame, WELL_COLUMNS)
    if well is None:
        report["note"] = ("this table names no well, so the guides' "
                          "well-mates cannot be found")
        out = frame[chosen]
        report.update(rows=int(len(out)), chosen=int(len(out)))
        return out, report
    here = list(wells) if wells is not None else wells_of(frame, guides)
    here = [str(w) for w in here]
    report["wells"] = here
    inside = frame[well].astype(str).isin(set(here))
    out = frame[inside].copy()
    out[MATE_COLUMN] = [CHOSEN if flag else MATE
                        for flag in chosen.reindex(out.index).fillna(False)]
    report.update(rows=int(len(out)),
                  chosen=int((out[MATE_COLUMN] == CHOSEN).sum()),
                  mates=int((out[MATE_COLUMN] == MATE).sum()))
    return out, report


def describe(report: Dict[str, Any]) -> str:
    """Format a concise description of the plotted population.

    :param report: well-scope selection report to summarize.
    """
    if report.get("note"):
        return report["note"]
    scope = report.get("scope")
    if scope == "all":
        return f"every datapoint in the table: {report['rows']:,}."
    if scope == "guides":
        return (f"{report['rows']:,} objects annotated with "
                f"{len(report['guides'])} chosen gRNA(s).")
    return (f"{report['chosen']:,} objects from the chosen gRNA(s) and "
            f"{report['mates']:,} of their well-mates, across "
            f"{len(report['wells'])} well(s) — the comparison that shares a "
            f"plate, a day, a stain and an imaging session.")
