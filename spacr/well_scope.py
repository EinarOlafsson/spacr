"""Select object populations for cell-table visualizations.

Plots can show objects assigned to selected guides, those objects together
with other objects from the same wells, or the complete table. The well set is
derived from the guide selection and can then be restricted explicitly.
"""
from __future__ import annotations

import logging
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
        # NOT AN ERROR AND NOT SILENTLY EVERYTHING. With no guide column or
        # no selection there is no "the chosen guides", so the honest answer
        # is an empty population and a report that says why -- drawing the
        # whole table instead would look like a selection nobody made.
        report["note"] = ("no gRNA was chosen, so there is no population to "
                          "draw; pick one or more points on the volcano")
        return frame.iloc[0:0], report

    wanted = {str(g) for g in guides}
    chosen = frame[guide].astype(str).isin(wanted)
    if scope == "guides":
        out = frame[chosen]
        report.update(rows=int(len(out)), chosen=int(len(out)))
        return out, report

    # scope == "wells"
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
    # DISTINGUISHABLE ON THE PLOT, which is the whole point of this scope.
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
