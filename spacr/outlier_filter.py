"""Remove segmentation artefacts BEFORE annotation (instruction 210).

BEFORE, NOT AFTER, and the order is the substance of the request. Annotation
methods that normalise -- everything that makes fractions sum to 1, and every
method instruction 209 proposes -- have their denominator set by which
objects are present. Removing a segmentation artefact AFTER the fractions are
computed leaves its reads redistributed across the guides; removing it FIRST
means it never contributed.

WHAT AN OUTLIER IS HAS TO BE STATED, NOT ASSUMED. A multiple of the MAD about
the median is the defensible default for these four: areas and intensities
are skewed, and a standard-deviation cut on skewed data removes real cells
from the long tail -- the exact population a screen is usually looking for.

OPTIONAL MEANS OFF BY DEFAULT. This changes which cells exist, and a filter
that silently drops objects is a filter that will be forgotten and then
blamed on the annotation.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger("spacr.outlier_filter")

#: The four the instruction names, as ``(setting, caption)``. The column each
#: one reads is resolved at filter time, because an intensity column carries
#: its channel in its name and the channel is the user's.
CRITERIA: Tuple[Tuple[str, str], ...] = (
    ("cell_area", "cell area"),
    ("nucleus_area", "nucleus area"),
    ("cell_intensity", "cell channel intensity"),
    ("nucleus_intensity", "nucleus channel intensity"),
)

#: Column names each criterion may appear under, most canonical first.
COLUMNS: Dict[str, Tuple[str, ...]] = {
    "cell_area": ("cell_area", "cell_area_px", "area_cell"),
    "nucleus_area": ("nucleus_area", "nucleus_area_px", "area_nucleus"),
    "cell_intensity": ("cell_channel_1_mean_intensity",
                       "cell_mean_intensity", "cell_intensity"),
    "nucleus_intensity": ("nucleus_channel_1_mean_intensity",
                          "nucleus_mean_intensity", "nucleus_intensity"),
}

#: How many MADs from the median before an object is an outlier. STATED, and
#: the user's -- 5 is loose enough not to touch a normal screen and tight
#: enough to catch a merged-object artefact an order of magnitude out.
DEFAULT_MADS = 5.0

#: `1.4826 * MAD` estimates sigma for a normal distribution, which is what
#: makes "5 MADs" comparable to "5 sigma" for the reader who thinks in sigma.
_TO_SIGMA = 1.4826


def column_for(frame: pd.DataFrame, criterion: str) -> Optional[str]:
    """The column ``criterion`` reads, or ``None`` if the table has none."""
    for name in COLUMNS.get(str(criterion), ()):
        if name in getattr(frame, "columns", ()):
            return name
    # An intensity column carries its channel in its name, and the channel is
    # the user's -- so accept the first that matches the shape rather than
    # insisting on channel 1.
    if "intensity" in str(criterion):
        stem = str(criterion).split("_")[0]
        for name in getattr(frame, "columns", ()):
            text = str(name)
            if text.startswith(stem) and "intensity" in text:
                return text
    return None


def outliers(values, *, mads: float = DEFAULT_MADS) -> np.ndarray:
    """A boolean mask of the values further than ``mads`` MADs from median.

    A MAD OF ZERO IS NOT A REASON TO DROP EVERYTHING. It means over half the
    values are identical, which happens on a small or a quantised column, and
    a rule that flagged every non-modal value there would empty the table.
    """
    data = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    good = np.isfinite(data)
    out = np.zeros(data.shape, dtype=bool)
    if good.sum() < 3:
        return out
    median = float(np.median(data[good]))
    mad = float(np.median(np.abs(data[good] - median)))
    if mad <= 0:
        return out
    limit = float(mads) * mad * _TO_SIGMA
    out[good] = np.abs(data[good] - median) > limit
    return out


def apply(frame: pd.DataFrame, settings: Optional[Dict[str, Any]] = None
          ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Drop the outliers the settings ask for. Returns the rows and a report.

    :param settings: ``{'<criterion>_outlier_mads': float}`` per criterion.
        A criterion with no entry, or ``None``, is OFF -- which is the
        default, because this changes which cells exist.
    :returns: ``(frame, report)``. The report has one row per criterion that
        was ON, with the column it read, the threshold and the count removed
        -- PER CRITERION, because "412 objects removed" does not tell a user
        whether their area cut or their intensity cut was the loose one.
    """
    settings = dict(settings or {})
    report: List[Dict[str, Any]] = []
    if frame is None or not len(frame):
        return (frame if frame is not None else pd.DataFrame()), report

    keep = np.ones(len(frame), dtype=bool)
    for criterion, caption in CRITERIA:
        mads = settings.get(f"{criterion}_outlier_mads")
        if mads is None:
            continue
        try:
            mads = float(mads)
        except (TypeError, ValueError):
            report.append({"criterion": criterion, "caption": caption,
                           "column": "", "mads": None, "removed": 0,
                           "note": f"{mads!r} is not a number of MADs"})
            continue
        if mads <= 0:
            continue
        column = column_for(frame, criterion)
        if column is None:
            # NOT SILENT. A filter the user switched on that found no column
            # removed nothing, and a run that says nothing about it looks
            # exactly like one where the filter worked and found nothing.
            report.append({"criterion": criterion, "caption": caption,
                           "column": "", "mads": mads, "removed": 0,
                           "note": f"this table has no {caption} column, so "
                                   f"nothing was filtered on it"})
            continue
        mask = outliers(frame[column], mads=mads)
        keep &= ~mask
        report.append({"criterion": criterion, "caption": caption,
                       "column": column, "mads": mads,
                       "removed": int(mask.sum()), "note": ""})
    return frame[keep], report


def describe(report: Sequence[Dict[str, Any]]) -> str:
    """The lines the run prints. Empty when no filter was on."""
    if not report:
        return ""
    lines = ["Outliers removed before annotation:"]
    for row in report:
        if row.get("note"):
            lines.append(f"  {row['caption']}: {row['note']}")
            continue
        lines.append(
            f"  {row['caption']} ({row['column']}): {row['removed']:,} "
            f"object(s) beyond {row['mads']:g} MADs of the median")
    lines.append("  The fractions below are computed on what survived — "
                 "removing an artefact after they were formed would leave "
                 "its reads redistributed across the guides.")
    return "\n".join(lines)
