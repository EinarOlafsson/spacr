"""Shared builders for ``analyze_endodyogeny`` / ``analyze_replication`` suites.

``_write_endodyogeny_db``, ``AREAS_C1``, ``AREAS_C2`` and ``_settings`` are
imported by ``tests/test_assay_panels.py``, ``tests/test_assay_report.py`` and
``tests/test_submodules_panels_coverage.py``.  The assay's own tests live in
``tests/test_submodules_class_and_volume_bins.py``.
"""
from __future__ import annotations

import os
import sqlite3

import pandas as pd

import matplotlib
matplotlib.use("Agg")

# analyze_endodyogeny imports spacr.io / spacr.plot / spacr.settings lazily
# *inside* the call, which would otherwise charge the whole heavy
# cellpose/torch/seaborn import chain to whichever test happens to run first.
# Pull them in at collection time instead. The symbol under test is still
# imported per-test below.
import spacr.io  # noqa: E402,F401
import spacr.plot  # noqa: E402,F401
import spacr.settings  # noqa: E402,F401
import spacr.sp_stats  # noqa: E402,F401
import spacr.submodules  # noqa: E402,F401
from statsmodels.stats.multitest import multipletests  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


# Areas chosen so that (area ** 1.5) lands in four distinct doubling bins
# relative to min_volume_bin = 500 ** 1.5 = 11180.34:
#   ~600  -> bin 1   [11180.34,  22360.68)
#   ~1000 -> bin 2   [22360.68,  44721.36)
#   ~1600 -> bin 3   [44721.36,  89442.72)
#   ~2600 -> bin 4   [89442.72, 178885.44)
AREAS_C1 = [600.0, 620.0, 640.0, 1000.0, 1050.0, 1100.0,
            1600.0, 1650.0, 1700.0, 2600.0, 2700.0, 2800.0]
AREAS_C2 = [610.0, 1010.0, 1020.0, 1030.0, 1610.0, 1620.0,
            1630.0, 1640.0, 2610.0, 2620.0, 2630.0, 2640.0]


def _write_endodyogeny_db(root, areas_by_column):
    """Write ``<root>/measurements/measurements.db`` with the tables the
    merge layer of :func:`spacr.io._read_and_merge_data` expects.

    One pathogen per cell (so ``pathogen_limit=1`` keeps everything and the
    per-object summed ``pathogen_area`` is the object's own area), plus a
    ``png_list`` table carrying an integer ``predictions`` class column.
    """
    meas = os.path.join(str(root), "measurements")
    os.makedirs(meas, exist_ok=True)
    db = os.path.join(meas, "measurements.db")

    cell_rows, pathogen_rows, png_rows = [], [], []
    obj = 0
    for column, areas in areas_by_column.items():
        for k, area in enumerate(areas):
            obj += 1
            row = f"r{(k % 3) + 1}"
            field = f"f{(k % 2) + 1}"
            base = {
                "plateID": "plate1", "rowID": row, "columnID": column,
                "fieldID": field, "prcf": f"plate1_{row}_{column}_{field}",
            }
            cell_rows.append({**base, "object_label": obj,
                              "cell_area": 20000.0 + float(area),
                              "cell_channel_0_mean_intensity": 100.0 + k})
            # cell_id is the INTEGER parent label here — spacr prefixes 'o'.
            pathogen_rows.append({**base, "object_label": obj, "cell_id": obj,
                                  "pathogen_area": float(area),
                                  "pathogen_channel_0_mean_intensity": 50.0 + k})
            # png_list stores the already-prefixed 'o<N>' string form.
            png_rows.append({**base, "cell_id": f"o{obj}",
                             "png_path": f"/x/plate1_{row}_{column}_{field}_o{obj}.png",
                             "predictions": k % 2})

    with sqlite3.connect(db) as con:
        pd.DataFrame(cell_rows).to_sql("cell", con, index=False, if_exists="replace")
        pd.DataFrame(pathogen_rows).to_sql("pathogen", con, index=False,
                                           if_exists="replace")
        pd.DataFrame(png_rows).to_sql("png_list", con, index=False,
                                      if_exists="replace")
    return db


def _settings(src, **overrides):
    """Baseline settings: pixel units, object level, nothing saved."""
    settings = {
        "src": src,
        "tables": ["cell", "pathogen"],
        "cell_types": ["Hela"],
        "cell_plate_metadata": None,
        "pathogen_types": ["nc", "pc"],
        "pathogen_plate_metadata": [["c1"], ["c2"]],
        "treatments": None,
        "treatment_plate_metadata": None,
        "um_per_px": None,
        "min_area_bin": 500,
        "max_area": 10 ** 9,
        "level": "object",
        "nuclei_limit": 10,
        "pathogen_limit": 1,
        "verbose": False,
        "save": False,
    }
    settings.update(overrides)
    return settings
