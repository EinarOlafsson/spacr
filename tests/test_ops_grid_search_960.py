"""Item 960: the OPS parameter grid search on representative tiles.

A grid point's counts are the counts the spacer-prefix decode writes at that
setting, the default point is the decode's own per-well choice, the same
settings write the same CSV, and applying a point writes its values and the
grid run into the settings.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from spacr import ops_grid_search as G
from spacr import ops_prefix as P
from tests.test_ops_spacer_prefix_decode_920 import (SPACERS, library_table, synthetic_site,
                                                     write_cpg_site)

#: A small tuning grid, so the synthetic well tunes in seconds.
TUNING = {"ops_read_threshold": [10, 5000], "ops_prefix_min_read_quality": [0, 0.5]}

#: The counts a grid point and the decode must agree on.
COUNTS = ("reads", "exact", "corrected", "cells_in_labels", "cells_real", "cells_decoy")


def _settings(tmp_path, name, **extra):
    """A synthetic four-site cropped well and the settings that decode it."""
    root = tmp_path / "data"
    if not root.exists():
        stack, labels = synthetic_site()
        for site in (1, 2, 3, 4):
            write_cpg_site(root, site, stack, labels)
        library_table(SPACERS).to_csv(tmp_path / "Barcodes.csv", index=False)
    return {"ops_layout": "cpg_cropped", "ops_library_mode": "spacer_prefix",
            "cpg_root": str(root / "images_corrected_cropped"),
            "cpg_phenotype_channels": ["CorrDNA"], "ops_library": str(tmp_path / "Barcodes.csv"),
            "ops_prefix_masks": str(root / "masks" / "{plate}_{well}_{site}_cells.tiff"),
            "ops_read_threshold": 10.0, "n_workers": 1, "ops_prefix_decoy_k": 4,
            "dst_root": str(tmp_path / name), **extra}


def _pooled(result):
    """The pooled rows of a grid run's CSV, by point."""
    frame = pd.read_csv(result["csv"])
    return frame[frame["tile"] == "pooled"].set_index("point")


def test_parse_grid_reads_text_json_and_dicts_alike():
    text = G._parse_grid("ops_read_threshold=40,80,80; ops_read_mode: log, raw")
    assert text == {"ops_read_threshold": [40.0, 80.0], "ops_read_mode": ["log", "raw"]}
    assert G._parse_grid('{"ops_read_threshold": [40, 80], "ops_read_mode": ["log", "raw"]}') == text
    assert G._parse_grid({"ops_prefix_max_distance": [1, 0]}) == {"ops_prefix_max_distance": [1, 0]}
    assert G._parse_grid(None) == {} and G._parse_grid("  ") == {}
    with pytest.raises(ValueError, match="cannot vary 'ops_min_fraction'"):
        G._parse_grid("ops_min_fraction=0.5")
    with pytest.raises(ValueError, match="'auto' chooses per well"):
        G._parse_grid("ops_read_mode=auto")
    with pytest.raises(ValueError, match="whole numbers"):
        G._parse_grid("ops_prefix_max_distance=0.5")
    with pytest.raises(ValueError, match="has no '='"):
        G._parse_grid("ops_read_threshold")


def test_pick_tiles_is_seeded_and_takes_the_wells_in_turn():
    index = pd.DataFrame([{"plate": "P", "well": f"Well{w}", "well_id": f"W{w}", "site": s}
                          for w in (1, 2, 3) for s in range(25)])
    first = G._pick_tiles(index, n=8, seed=3)
    assert first == G._pick_tiles(index, n=8, seed=3)
    assert first != G._pick_tiles(index, n=8, seed=4)
    assert [t.split(":")[1] for t in first] == ["W1", "W2", "W3"] * 2 + ["W1", "W2"]
    edge = G._pick_tiles(index, n=6, mode="edge_centre", seed=0)
    assert {t.split(":")[2] for t in edge[:3]} <= {"0", "4", "20", "24"}
    assert {t.split(":")[2] for t in edge[3:]} == {"12"}
    assert G._pick_tiles(index, mode="user", tiles="W2:7, P:W3:1") == ["P:W2:7", "P:W3:1"]
    assert G._pick_tiles(index, n=200)[-1].startswith("P:W") and len(G._pick_tiles(index, n=200)) == 75
    with pytest.raises(ValueError, match="needs ops_grid_tiles"):
        G._pick_tiles(index, mode="user")
    with pytest.raises(ValueError, match="more than one site"):
        G._pick_tiles(index, tiles="7")
    with pytest.raises(ValueError, match="ops_grid_tile_mode must be one of"):
        G._pick_tiles(index, mode="corners")


def test_a_grid_point_counts_what_the_decode_writes_at_that_setting(tmp_path):
    grid = {"ops_read_threshold": [10, 5000], "ops_read_mode": ["log", "raw"]}
    settings = _settings(tmp_path, "grid", ops_prefix_threshold_tuning="none",
                         ops_prefix_fdr_calibration="shuffle", ops_grid_params=grid,
                         ops_grid_tiles="1,3")
    result = G._run_grid_search(settings)
    pooled = _pooled(result)
    assert len(pooled) == 5 and bool(pooled.loc[0, "is_default"])
    assert result["default_matches_decode"] is None
    lines = pd.read_csv(result["csv"])
    for point, mode, threshold in ((0, "log", 10.0), (2, "raw", 10.0), (3, "log", 5000.0)):
        decoded = P.run_prefix_decode({**settings, "ops_read_mode": mode,
                                       "ops_read_threshold": threshold, "cpg_sites": [1, 3],
                                       "dst_root": str(tmp_path / f"decode{point}")})
        summary = pd.read_csv(os.path.join(decoded["folder"], "site_summary.csv"))
        for site in (1, 3):
            ours = lines[(lines["point"] == point) & (lines["tile"] == f"CP9:W1:{site}")].iloc[0]
            theirs = summary[summary["site"] == site].iloc[0]
            assert {c: int(ours[c]) for c in COUNTS} == {c: int(theirs[c]) for c in COUNTS}
            assert ours["shuffle_called"] == pytest.approx(theirs["shuffle_called"])
    assert pooled.loc[3, "spots"] < pooled.loc[1, "spots"]
    assert pooled.loc[0, "cells_real"] > 0 and bool(pooled.loc[0, "pareto"])


def test_the_default_point_is_the_decodes_own_tuned_choice(tmp_path):
    settings = _settings(tmp_path, "tuned", ops_prefix_threshold_tuning="per_well_decoy",
                         ops_prefix_tuning_grid=TUNING, ops_prefix_fdr_target_pct=50.0,
                         ops_grid_params="ops_prefix_max_distance=0,1", ops_grid_n_tiles=4)
    result = G._run_grid_search(settings)
    assert result["default_matches_decode"] is True
    assert result["tiles"] == [f"CP9:W1:{s}" for s in (1, 2, 3, 4)] or sorted(result["tiles"]) \
        == sorted(f"CP9:W1:{s}" for s in (1, 2, 3, 4))
    default = result["defaults"]["CP9:W1"]
    assert default["ops_read_mode"] in ("log", "raw")
    assert default["ops_read_threshold"] in (10.0, 5000.0)
    assert set(_pooled(result)["ops_prefix_max_distance"].astype(int)) == {0, 1}


def test_the_same_settings_write_the_same_csv(tmp_path):
    settings = _settings(tmp_path, "again", ops_prefix_threshold_tuning="none",
                         ops_grid_params="ops_prefix_min_read_quality=0,0.5", ops_grid_seed=2,
                         ops_grid_n_tiles=2)
    first = G._run_grid_search(settings, settings_path=str(tmp_path / "s" / "settings.csv"))
    with open(first["csv"], "rb") as handle:
        written = handle.read()
    second = G._run_grid_search(settings, settings_path=str(tmp_path / "s" / "settings.csv"))
    assert second["run_id"] == first["run_id"] and second["csv"] == first["csv"]
    with open(second["csv"], "rb") as handle:
        assert handle.read() == written
    assert os.path.dirname(first["csv"]) == str(tmp_path / "s")
    for path in (first["timing_csv"], first["plot"], first["json"]):
        assert os.path.getsize(path) > 0
    timing = pd.read_csv(first["timing_csv"])
    assert set(timing.columns) >= {"detect_s", "eval_s"} and len(timing) == 3 * 2
    other = G._run_grid_search({**settings, "ops_grid_seed": 5})
    assert other["run_id"] != first["run_id"] or other["tiles"] == first["tiles"]


def test_two_workers_write_what_one_writes(tmp_path):
    settings = _settings(tmp_path, "workers", ops_prefix_threshold_tuning="none",
                         ops_grid_params="ops_read_threshold=10,5000", ops_grid_n_tiles=3)
    one = G._run_grid_search(settings, settings_path=str(tmp_path / "one" / "s.csv"))
    two = G._run_grid_search({**settings, "n_workers": 2},
                            settings_path=str(tmp_path / "two" / "s.csv"))
    with open(one["csv"], "rb") as a, open(two["csv"], "rb") as b:
        assert a.read() == b.read()


def test_apply_writes_the_point_and_the_grid_run(tmp_path):
    settings = _settings(tmp_path, "apply", ops_prefix_threshold_tuning="per_well_decoy",
                         ops_prefix_tuning_grid=TUNING, ops_prefix_fdr_target_pct=50.0,
                         ops_grid_params="ops_read_threshold=10; ops_read_mode=raw",
                         ops_grid_tiles="1,2", ops_grid_search=True)
    result = G._run_grid_search(settings)
    applied = G._apply_grid_choice(settings, result, 1)
    assert applied["ops_read_threshold"] == 10.0 and applied["ops_read_mode"] == "raw"
    assert applied["ops_prefix_threshold_tuning"] == "none"
    assert applied["ops_grid_run_id"] == result["run_id"] and applied["ops_grid_search"] is False
    assert applied["ops_grid_tiles"] == "CP9:W1:1,CP9:W1:2" and applied["ops_grid_seed"] == 0
    kept = G._apply_grid_choice(settings, result["csv"], 0)
    assert kept["ops_prefix_threshold_tuning"] == "per_well_decoy"
    assert kept["ops_grid_run_id"] == result["run_id"]
    assert "ops_read_mode" not in kept or kept["ops_read_mode"] == settings.get("ops_read_mode")
    with pytest.raises(ValueError, match="has no point 7"):
        G._apply_grid_choice(settings, result, 7)


def test_a_run_with_the_grid_switch_on_runs_only_the_grid(tmp_path):
    from spacr.ops_engine import run_ops

    settings = _settings(tmp_path, "switch", ops_prefix_threshold_tuning="none",
                         ops_grid_search=True, ops_grid_n_tiles=1)
    out = run_ops(settings)
    assert out["phases"] == ["prefix_grid_search"]
    assert os.path.isfile(out["prefix_grid_search"]["csv"])
    assert not os.path.exists(os.path.join(settings["dst_root"], "results", "ops_prefix"))
    with pytest.raises(ValueError, match="cropped-site layout"):
        G._run_grid_search({**settings, "ops_layout": "tiles"})


def test_the_new_settings_are_off_by_default_and_alpha():
    from spacr.ops_settings import OPS_DEFAULTS
    from spacr.settings import ALPHA_FEATURES

    keys = ("ops_grid_search", "ops_grid_params", "ops_grid_tiles", "ops_grid_tile_mode",
            "ops_grid_n_tiles", "ops_grid_seed", "ops_grid_run_id")
    assert {key: OPS_DEFAULTS[key] for key in keys} == {key: (False if key == "ops_grid_search"
                                                               else None) for key in keys}
    assert ALPHA_FEATURES[960]["settings"] == keys
