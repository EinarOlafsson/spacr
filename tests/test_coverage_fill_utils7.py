"""Coverage-fill batch 7 for spacr.utils settings/df/file helpers."""
from __future__ import annotations


import pandas as pd

import matplotlib
matplotlib.use("Agg")

from spacr import utils as U


# ---------------------------------------------------------------------------
# load_settings — every parse_value branch
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# save_settings  (+ round-trip via explicit column names)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _group_by_well
# ---------------------------------------------------------------------------

def test_group_by_well():
    df = pd.DataFrame({
        "plateID": ["p1"] * 4, "rowID": ["r1", "r1", "r2", "r2"],
        "columnID": ["c1", "c1", "c1", "c1"],
        "value": [10.0, 20.0, 30.0, 40.0],
        "label": ["a", "b", "c", "d"],
    })
    out = U._group_by_well(df)
    # r1/c1 numeric mean = 15, non-numeric first = 'a'
    assert out.loc[("p1", "r1", "c1"), "value"] == 15.0


# ---------------------------------------------------------------------------
# annotate_predictions
# ---------------------------------------------------------------------------

def test_annotate_predictions(tmp_path):
    csv = tmp_path / "preds.csv"
    pd.DataFrame({
        "path": [
            "/x/1_A05_1_1.png",   # plate1 col5 → screen
            "/x/5_B06_1_2.png",   # plate5 col6 → pc
            "/x/1_C02_1_3.png",   # col2 → nc
        ],
        "pred": [0.9, 0.1, 0.5],
    }).to_csv(csv, index=False)
    out = U.annotate_predictions(str(csv))
    conds = set(out["cond"])
    assert "screen" in conds and "pc" in conds and "nc" in conds


# ---------------------------------------------------------------------------
# get_files_from_dir / _find_similar_sized_images
# ---------------------------------------------------------------------------
