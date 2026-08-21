"""Which objects a cell-table graph draws (instruction 205).

    "for the sow option in the cell table graphs, i whould be able to gRNAs,
     gRNAs + other datapoints in selected wells, and All datapoints.
     presently i cannot pick gRNAs + other datapoints in selected wells.
     selected well are the wells that contain the gRNAs chosen on the
     volcano plot. the user should be able to choose a selection of these
     wells."

THE MIDDLE ONE IS THE COMPARISON THAT MATTERS, and it is the one that was
not offered. A guide's cells against every cell in the screen compares two
different experiments; a guide's cells against their own WELL-MATES compares
two populations that shared a plate, a day, a stain and an imaging session.

THE WELL SET HAS TWO STAGES: derived from the guide selection, then
editable. Which means it has to be VISIBLE -- a set the user is invited to
narrow and cannot see is a set they cannot narrow.
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

#: The column marking which side of the middle scope an object is on. Named
#: rather than inferred, because the whole point of that scope is that the
#: two populations are DISTINGUISHABLE on the plot -- drawing them in one
#: colour would answer the question by hiding it.
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
    """The wells the chosen guides are in, in table order.

    THE VOLCANO IS THE SELECTOR: the same click that picks a point picks
    these. Order is the table's rather than sorted, so a user narrowing the
    set sees them in the order the screen was laid out.
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
    """The objects to draw, and a report of what was chosen.

    :param scope: one of :data:`SCOPES`.
    :param guides: the guides chosen on the volcano.
    :param wells: the well set, already narrowed by the user. ``None`` means
        derive it from ``guides``, which is the first of its two stages.
    :returns: ``(frame, report)``. Under ``scope='wells'`` the frame carries
        :data:`MATE_COLUMN` saying which side each object is on.
    :raises ValueError: for a scope this module does not have. Falling back
        to "all" would draw a different population from the one asked for
        and say nothing.
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
    """The sentence under the plot. Says which population is drawn."""
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
