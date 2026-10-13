"""The outlier criteria, and the columns each one may be stored under.

SEPARATE FROM :mod:`spacr.outlier_filter` FOR ONE REASON: this is data and
that module needs pandas. `spacr.settings._outlier_criteria` wants nothing but
the tuple below, and reaching it through `outlier_filter` charged a 211 ms
pandas import to read four pairs of strings -- on the main thread, while a
settings panel was being laid out, which is a visible pause in the interface.

The tuple used to be written TWICE: here in spirit, and again as a fallback
inside `_outlier_criteria` for the case where the import failed. The two
copies were identical and nothing compared them, so either could have drifted
silently. There is one now, and `outlier_filter` re-exports it so every
existing importer keeps working.
"""

from __future__ import annotations

from typing import Dict, Tuple

__all__ = ["CRITERIA", "COLUMNS"]

#: The outlier criteria a run can filter on, as (key, human name) pairs.
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
