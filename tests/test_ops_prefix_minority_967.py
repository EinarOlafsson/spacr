"""Item 967: the spacer-prefix cell call ignores minority stray barcodes (alpha, default 3 since schema 945).

Inside a cell, a barcode with fewer than ``ops_prefix_minority_min_reads``
reads is left out of the call when another barcode has strictly more reads;
ties still block the call and a decoy-top cell is still not genotyped. 0 (the
default before schema 945, and the direct-call fallback of prefix_settings)
writes what the decode wrote before, byte for byte.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from spacr import ops_prefix as P
from tests.test_ops_read_mode_952 import _prefix, _same_files


def _reads():
    """Five cells: a stray-read cell, a 1-vs-1 tie, a clean cell, a 4-vs-3 cell, a decoy-top cell."""
    rows = []

    def add(cell, prefix, n, gene=None):
        rows.extend({"plate": "P", "well": "W", "site": 1, "cell": cell, "status": "exact",
                     "key": "ACGT", "distance": 0, "prefix": prefix, "guide": prefix,
                     "gene": gene or f"G{prefix}", "level": "guide", "quality": 1.0}
                    for _ in range(n))

    add(1, "A", 3)
    add(1, "B", 1)
    add(2, "A", 1)
    add(2, "B", 1)
    add(3, "A", 5)
    add(4, "A", 4)
    add(4, "B", 3)
    add(5, "D", 3, gene=P.DECOY_GENE)
    add(5, "A", 1)
    rows.append({**rows[0], "cell": 0})
    return pd.DataFrame(rows)


def test_zero_and_one_change_nothing():
    reads = _reads()
    for method in P.QUALITY_METHODS:
        base = P.prefix_cell_calls(reads, method=method)
        for k in (0, 1, None):
            pd.testing.assert_frame_equal(
                P.prefix_cell_calls(reads, method=method, minority_min_reads=k), base)
        assert P._called_counts(reads, method, minority_min_reads=0) == \
            P._called_counts(reads, method)


def test_the_rule_rescues_stray_reads_and_keeps_ties_blocked():
    reads = _reads()
    assert list(P.prefix_cell_calls(reads, method="recipe_simple")["quality"]) == \
        ["bad", "bad", "perfect", "bad", "bad"]
    k3 = P.prefix_cell_calls(reads, method="recipe_simple", minority_min_reads=3)
    assert list(k3["quality"]) == ["perfect", "bad", "perfect", "bad", "perfect"]
    assert list(k3["minority_rescued"]) == [True, False, False, False, True]
    assert list(k3["minority_ignored"]) == [1, 0, 0, 0, 1]
    assert k3.loc[0, "prefix"] == "A" and k3.loc[0, "barcodes"] == 1
    assert k3.loc[0, "matched"] == 4 and k3.loc[0, "top_share"] == 0.75
    k4 = P.prefix_cell_calls(reads, method="recipe_simple", minority_min_reads=4)
    assert list(k4["quality"]) == ["perfect", "bad", "perfect", "perfect", "perfect"]
    flagged = P._flag_decoy_cells(k3)
    assert bool(flagged.loc[4, "decoy"]) and flagged.loc[4, "gene"] is None
    assert P._called_counts(reads, "recipe_simple", minority_min_reads=3) == (2, 1)
    assert P._called_counts(reads, "recipe_simple", minority_min_reads=4) == (3, 1)
    assert P._called_counts(reads, "recipe_simple") == (1, 0)


def test_the_spacr_classes_keep_their_calls():
    reads = _reads()
    base = P.prefix_cell_calls(reads, method="spacr")
    k3 = P.prefix_cell_calls(reads, method="spacr", minority_min_reads=3)
    called = ("perfect", "great")
    assert list(base["quality"].isin(called)) == list(k3["quality"].isin(called))
    assert not k3["minority_rescued"].any()
    assert list(k3["quality"]) == ["perfect", "ambiguous", "perfect", "great", "perfect"]
    assert P._called_counts(reads, "spacr", minority_min_reads=3) == \
        P._called_counts(reads, "spacr")


def test_the_setting_is_alpha_with_a_registered_default_of_3():
    from spacr.ops_settings import OPS_DEFAULTS, OPS_TOOLTIPS
    from spacr.settings import ALPHA_FEATURES

    assert OPS_DEFAULTS["ops_prefix_minority_min_reads"] == 3
    assert ALPHA_FEATURES[967]["settings"] == ("ops_prefix_minority_min_reads",)
    assert "Alpha." in OPS_TOOLTIPS["ops_prefix_minority_min_reads"]
    base = {"ops_library": "x.csv"}
    assert P.prefix_settings(base)["ops_prefix_minority_min_reads"] == 0
    assert P.prefix_settings({**base, "ops_prefix_minority_min_reads": 3})[
        "ops_prefix_minority_min_reads"] == 3
    with pytest.raises(ValueError, match="ops_prefix_minority_min_reads"):
        P.prefix_settings({**base, "ops_prefix_minority_min_reads": -1})


@pytest.mark.parametrize("extra", [{}, {"ops_prefix_decoy_k": 4,
                                        "ops_prefix_quality_method": "recipe_simple"}])
def test_zero_writes_what_the_decode_wrote(tmp_path, extra):
    base = _prefix(tmp_path, "base", **extra)
    off = _prefix(tmp_path, "off", ops_prefix_minority_min_reads=0, **extra)
    _same_files(base["folder"], off["folder"])
    assert "minority_rescued" not in off
    saved = json.load(open(os.path.join(off["folder"], "quality_classes.json")))
    assert "ops_prefix_minority_min_reads" not in saved["settings"]


def test_on_the_run_reports_the_rescued_cells(tmp_path):
    lines = []
    settings = {"ops_prefix_decoy_k": 4, "ops_prefix_quality_method": "recipe_simple",
                "ops_prefix_minority_min_reads": 3}
    out = _prefix(tmp_path, "on", **settings)
    saved = json.load(open(os.path.join(out["folder"], "quality_classes.json")))
    assert saved["settings"]["ops_prefix_minority_min_reads"] == 3
    summary = pd.read_csv(os.path.join(out["folder"], "site_summary.csv"))
    cells = pd.read_csv(os.path.join(out["folder"], "cells_CP9_W1.csv"))
    assert {"minority_ignored", "minority_rescued"} <= set(cells.columns)
    rescued = int((cells["minority_rescued"] & ~cells["decoy"]).sum())
    assert int(summary["cells_minority_rescued"].sum()) == rescued
    assert out["minority_rescued"] == {"CP9_W1": rescued}
    original = P.run_prefix_decode
    tuned = {**settings, "ops_prefix_threshold_tuning": "per_well_decoy",
             "ops_read_mode": "log", "ops_prefix_fdr_target_pct": 50.0,
             "ops_prefix_tuning_grid": {"ops_read_threshold": [10, 5000]}}
    root = tmp_path / "data"
    again = original({**_settings_like(tmp_path, root), **tuned,
                      "dst_root": str(tmp_path / "tuned")}, report=lines.append)
    assert "CP9_W1" in again["minority_rescued"]
    assert any("minority-read rule (ops_prefix_minority_min_reads 3) rescued" in line
               for line in lines)


def _settings_like(tmp_path, root):
    """The settings :func:`tests.test_ops_read_mode_952._prefix` decodes with."""
    return {"ops_layout": "cpg_cropped", "ops_library_mode": "spacer_prefix",
            "cpg_root": str(root / "images_corrected_cropped"),
            "cpg_phenotype_channels": ["CorrDNA"], "ops_library": str(tmp_path / "Barcodes.csv"),
            "ops_prefix_masks": str(root / "masks" / "{plate}_{well}_{site}_cells.tiff"),
            "ops_read_threshold": 10.0, "n_workers": 1}
