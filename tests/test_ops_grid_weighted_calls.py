"""Weighted grid cells and shuffled controls follow the prefix decoder settings."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from spacr import cpg_cropped, ops_grid_search as grid, ops_prefix as prefix
from tests.test_ops_grid_search_960 import _settings
from tests.test_ops_read_weights_988 import _reads


@pytest.mark.parametrize("extra,best", [
    ({}, None),
    ({"ops_prefix_weighted_quality": True}, None),
    ({"ops_prefix_distance_weight": "decay", "ops_prefix_distance_decay_px": 4}, None),
    ({"ops_prefix_distance_weight": "decay", "ops_prefix_distance_decay_px": 4,
      "ops_prefix_distance_interior_px": 20}, None),
    ({"ops_prefix_distance_weight": "decay", "ops_prefix_distance_decay_px": 4,
      "ops_prefix_distance_min_weight": 0.8}, None),
    ({"ops_read_neighbour_discount": 0}, None),
    ({"ops_prefix_distance_weight": "distance_only", "ops_prefix_distance_decay_px": 4,
      "ops_prefix_weighted_quality": True, "ops_read_neighbour_discount": 0}, None),
    ({"ops_prefix_distance_weight": "decay", "ops_prefix_distance_decay_px": 4,
      "ops_read_neighbour_discount": 0},
     {"ops_prefix_distance_decay_px": 0, "ops_read_neighbour_discount": 1,
      "ops_prefix_weighted_quality": 1}),
], ids=["share", "quality", "distance", "interior", "floor", "discount",
        "distance-only", "tuned-weights"])
def test_grid_weighted_calls_match_the_decode_settings(monkeypatch, extra, best):
    """Each weight changes the same cells in the grid, decode and shuffle control."""
    reads = _reads()
    reads.loc[reads["prefix"] == "A", "quality"] = 0.95
    reads.loc[reads["prefix"] == "B", "quality"] = 0.05
    reads["y"] = 2
    reads["x"] = reads["cell"].map({0: 22, 1: 2, 2: 12, 3: 27, 4: 57, 5: 72})
    reads.loc[reads["prefix"] == "D", "x"] = 47
    labels = np.zeros((5, 80), np.int32)
    nuclei = np.zeros_like(labels)
    for cell, left, right, centre in ((1, 0, 10, 2), (2, 10, 20, 12),
                                     (3, 25, 50, 27), (4, 55, 65, 57), (5, 70, 80, 72)):
        labels[:, left:right] = cell
        nuclei[2, centre] = cell
    pairs = prefix._touching_cells(labels)
    neighbours = prefix._neighbour_table(pairs, "P", "W", 1)
    settings = {"ops_library": "unused.csv", "ops_prefix_quality_method": "weighted_majority",
                "ops_prefix_weighted_quality": False, "ops_prefix_weighted_min_share": 0.9,
                "ops_prefix_threshold_tuning": "none", "ops_prefix_decoy_k": 1,
                "ops_prefix_fdr_calibration": "shuffle", **extra}
    chosen = prefix.prefix_settings(settings)
    point = {key: "log" if key == "ops_read_mode" else chosen[key] for key in grid._GRID_KEYS}
    point.update(best or {})
    weights = {**prefix._weight_arguments(chosen, best), "neighbours": neighbours}
    expected = prefix._flag_decoy_cells(prefix.prefix_cell_calls(
        reads, method="weighted_majority", **weights))
    expected_hit = expected["quality"].isin(("perfect", "great"))

    def shuffled_reads(frame, *args, **kwargs):
        """Keep the barcode mixture while making the shuffled top barcode a control."""
        shuffled = frame.copy()
        shuffled.loc[shuffled["gene"] == prefix.DECOY_GENE, "gene"] = "control"
        return shuffled

    shuffled = prefix.prefix_cell_calls(shuffled_reads(reads), method="weighted_majority",
                                        **weights)
    expected_shuffle = int(shuffled["quality"].isin(("perfect", "great")).sum())

    def matched_reads(*args, extra=None, **kwargs):
        """Supply matched candidates with the geometry measured by the grid worker."""
        frame = reads.drop(columns=["plate", "well", "site", "nuc_gap_px"]).copy()
        for name, values in (extra or {}).items():
            frame[name] = values
        return frame

    monkeypatch.setattr(cpg_cropped, "load_cpg_site",
                        Mock(return_value={"sbs": np.zeros((1, 5, 80, 4))}))
    monkeypatch.setattr(prefix, "site_labels", Mock(return_value=labels))
    monkeypatch.setattr(prefix, "_site_nuclei", Mock(return_value=nuclei))
    monkeypatch.setattr(prefix, "_spot_candidates", Mock(return_value=(
        reads[["y", "x"]].to_numpy(), np.full(len(reads), 100), np.zeros((len(reads), 1, 4)))))
    monkeypatch.setattr(prefix, "_site_tuning_reads", matched_reads)
    monkeypatch.setattr(prefix, "_shuffle_frame", shuffled_reads)
    monkeypatch.setattr(grid, "_LIBRARIES", {"matching": None, "control": None,
                                            "shuffles": [object()]})
    result = grid._score_tile({"settings": settings, "index": pd.DataFrame(), "plate": "P",
                               "well": "WellW", "well_id": "W", "site": 1, "points": [point],
                               "pinned": {"cpg_sbs_bases": list("ACGT")}, "cycles": 1,
                               "loosest": 1})
    row = result["rows"][0]
    assert row["cells_real"] == int((expected_hit & ~expected["decoy"]).sum())
    assert row["cells_decoy"] == int((expected_hit & expected["decoy"]).sum())
    assert row["cells_ambiguous"] == int((expected["quality"] == "ambiguous").sum())
    assert row["shuffle_called"] == expected_shuffle


def test_the_grid_retains_the_weight_settings_chosen_by_the_decoder(tmp_path):
    """The default grid point carries the selected weight axes from per-well tuning."""
    settings = _settings(tmp_path, "grid", ops_grid_tiles="1", ops_read_mode="log",
                         ops_prefix_quality_method="weighted_majority",
                         ops_prefix_threshold_tuning="per_well_decoy",
                         ops_prefix_distance_weight="decay", ops_prefix_distance_decay_px=4,
                         ops_read_neighbour_discount=0, ops_prefix_weighted_quality=False,
                         ops_prefix_tuning_grid={"ops_read_threshold": [10],
                                                "ops_prefix_min_read_quality": [0],
                                                "ops_prefix_max_distance": [0],
                                                "ops_prefix_distance_decay_px": [4],
                                                "ops_read_neighbour_discount": [0],
                                                "ops_prefix_weighted_quality": [0]})
    settings["ops_prefix_nuclei_masks"] = settings["ops_prefix_masks"]
    result = grid._run_grid_search(settings)
    tuned_file = next(Path(settings["dst_root"]).rglob("threshold_tuning_CP9_W1.json"))
    tuned = json.loads(tuned_file.read_text())["chosen"]
    defaults = result["defaults"]["CP9:W1"]
    for key in prefix.WEIGHT_GRID_KEYS:
        assert defaults[key] == tuned[key]
    assert result["default_matches_decode"] is True
