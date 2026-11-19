"""Item 982 small-well guard: a tuned well with few sites keeps the read-quality floor at 0.1.

Off (0, the default) the decode writes what it wrote before.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from spacr import ops_prefix as P
from tests.test_ops_read_mode_952 import _prefix

#: Per-well decoy tuning on the 4-site synthetic well, with a grid holding
#: three floors.
TUNED = {"ops_prefix_decoy_k": 4, "ops_prefix_threshold_tuning": "per_well_decoy",
         "ops_prefix_fdr_target_pct": 50.0,
         "ops_prefix_tuning_grid": {"ops_read_threshold": [10, 5000],
                                    "ops_prefix_min_read_quality": [0, 0.1, 0.5]}}


def _same_outputs(one, two):
    """Every output byte-identical; the tuning json apart from its run-folder paths."""
    names = sorted(os.listdir(one))
    assert names == sorted(os.listdir(two))
    for name in names:
        with open(os.path.join(one, name), "rb") as a, open(os.path.join(two, name), "rb") as b:
            first, second = a.read(), b.read()
        if name.startswith("threshold_tuning_") and name.endswith(".json"):
            first, second = (json.loads(x) for x in (first, second))
            first.pop("paths"), second.pop("paths")
        if name == "quality_classes.json":
            # The setting is recorded whenever it is on, even when no well is small.
            first, second = (json.loads(x) for x in (first, second))
            second["settings"].pop("ops_prefix_tuning_min_sites", None)
        assert first == second, name


def test_the_axes_keep_the_floor_only_below_the_site_count():
    axes = [("ops_read_threshold", [40.0, 80.0]),
            ("ops_prefix_min_read_quality", [0.0, 0.1, 0.25]),
            ("ops_prefix_max_distance", [1.0, 0.0])]
    assert P._small_well_axes(axes, 3, 0) == (axes, False)
    assert P._small_well_axes(axes, 4, 4) == (axes, False)
    guarded, flag = P._small_well_axes(axes, 3, 4)
    assert flag and guarded == [axes[0], ("ops_prefix_min_read_quality", [0.1]), axes[2]]
    assert P.SMALL_WELL_FLOOR == 0.1


def test_off_and_a_large_enough_well_write_what_the_decode_wrote(tmp_path):
    base = _prefix(tmp_path, "base", ops_read_mode="log", **TUNED)
    off = _prefix(tmp_path, "off", ops_read_mode="log", ops_prefix_tuning_min_sites=0, **TUNED)
    big = _prefix(tmp_path, "big", ops_read_mode="log", ops_prefix_tuning_min_sites=4, **TUNED)
    _same_outputs(base["folder"], off["folder"])
    saved = json.load(open(os.path.join(off["folder"], "quality_classes.json")))
    assert "ops_prefix_tuning_min_sites" not in saved["settings"]
    _same_outputs(base["folder"], big["folder"])
    assert "small_well_floor" not in big["tuning"]["CP9_W1"]


def test_a_small_well_keeps_the_floor_and_says_so(tmp_path, monkeypatch):
    lines = []
    original = P.run_prefix_decode
    monkeypatch.setattr(P, "run_prefix_decode",
                        lambda settings, report=None: original(settings, report=lines.append))
    out = _prefix(tmp_path, "small", ops_read_mode="log", ops_prefix_tuning_min_sites=5, **TUNED)
    choice = out["tuning"]["CP9_W1"]
    assert choice["chosen"]["ops_prefix_min_read_quality"] == 0.1
    assert choice["small_well_floor"] == 0.1 and choice["small_well_sites"] == 4
    stem = os.path.join(out["folder"], "threshold_tuning_CP9_W1")
    assert set(pd.read_csv(stem + ".csv")["ops_prefix_min_read_quality"]) == {0.1}
    assert json.load(open(stem + ".json"))["small_well_floor"] == 0.1
    saved = json.load(open(os.path.join(out["folder"], "quality_classes.json")))
    assert saved["settings"]["ops_prefix_tuning_min_sites"] == 5
    assert any("fewer than ops_prefix_tuning_min_sites 5" in line for line in lines)
    assert out["well_choices"][0].endswith("read-quality floor kept at 0.1")


def test_a_negative_site_count_is_refused(tmp_path):
    with pytest.raises(ValueError, match="ops_prefix_tuning_min_sites"):
        _prefix(tmp_path, "bad", ops_prefix_tuning_min_sites=-1, **TUNED)


def test_alpha_registration():
    from spacr.ops_settings import OPS_CATEGORIES, OPS_DEFAULTS, OPS_TOOLTIPS, OPS_TYPES
    from spacr.settings import ALPHA_FEATURES

    assert ALPHA_FEATURES[982]["settings"] == ("ops_prefix_tuning_min_sites",)
    assert OPS_DEFAULTS["ops_prefix_tuning_min_sites"] == 0
    assert OPS_TYPES["ops_prefix_tuning_min_sites"] is int
    tip = OPS_TOOLTIPS["ops_prefix_tuning_min_sites"]
    assert tip.startswith("(int) - ") and tip.endswith("Default 0. Alpha.")
    assert any("ops_prefix_tuning_min_sites" in keys for keys in OPS_CATEGORIES.values())
