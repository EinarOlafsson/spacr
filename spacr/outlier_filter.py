"""Remove robustly defined object outliers before guide annotation.

Optional filters operate on cell or nucleus area and intensity. Each filter
uses distance from the median in scaled median absolute deviations (MADs),
which is less sensitive to skewed measurements than a standard-deviation
threshold. Filtering precedes fraction-based annotation so excluded objects
do not contribute to normalization denominators.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger("spacr.outlier_filter")

#: Supported filter criteria represented as ``(setting, display label)``.
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

#: Default number of scaled MADs separating an outlier from the median.
DEFAULT_MADS = 5.0

#: `1.4826 * MAD` estimates sigma for a normal distribution, which is what
#: makes "5 MADs" comparable to "5 sigma" for the reader who thinks in sigma.
_TO_SIGMA = 1.4826


def column_for(frame: pd.DataFrame, criterion: str) -> Optional[str]:
    """Resolve the measurement column used by an outlier criterion."""
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
    """Identify values beyond a scaled-MAD threshold.

    Parameters
    ----------
    values : array-like
        Values to evaluate. Non-numeric and non-finite values are not flagged.
    mads : float, default=DEFAULT_MADS
        Distance from the median in robust sigma units, computed as
        ``1.4826 * MAD``.

    Returns
    -------
    numpy.ndarray
        Boolean outlier mask. Fewer than three finite values or a zero MAD
        produces an all-false mask.
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
    """Apply enabled outlier criteria to an object table.

    Parameters
    ----------
    frame : pandas.DataFrame
        Object-level measurements.
    settings : dict, optional
        Thresholds keyed as ``"<criterion>_outlier_mads"``. Missing or
        ``None`` values disable that criterion.

    Returns
    -------
    pandas.DataFrame
        Rows retained after all enabled criteria.
    list of dict
        Per-criterion measurement column, threshold, removal count, and any
        validation note.
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
    """Format the per-criterion outlier report for run output."""
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
