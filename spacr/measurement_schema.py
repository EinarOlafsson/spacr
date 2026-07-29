"""Shared measurement-table provenance schema.

This module is intentionally dependency-free so measurement writers, database
mergers, and feature documentation can import the same contract without
creating circular imports or pulling analysis dependencies into workers.
"""

from __future__ import annotations

__all__ = ["MEASUREMENT_STAMP_COLUMNS"]


#: Provenance written on every object-measurement row. The order is stable
#: because database writes, merges, and exported feature dictionaries expose
#: it as part of spaCR's public data contract.
MEASUREMENT_STAMP_COLUMNS: tuple[str, ...] = (
    "measurement_ndim",
    "measurement_units",
    "n_z",
    "voxel_size_z_um",
    "voxel_size_xy_um",
)
