"""Item 966: the spacer-prefix decode's default is the auto read with per-well decoy tuning.

The OPS barcode decode keeps the log read; settings stamped before 944 keep
the fixed thresholds and the log read, byte for byte.
"""
from __future__ import annotations

import json
import os

import pytest

from spacr import ops_prefix as P
from spacr import ops_sbs as S
from tests.test_ops_read_mode_952 import _prefix, _same_files

#: The pre-944 values of the four keys item 966 changes.
OLD = {"ops_read_mode": "log", "ops_prefix_threshold_tuning": "none",
       "ops_prefix_decoy_k": 0, "ops_prefix_fdr_target_pct": 0.1}

#: A small grid, so the synthetic well tunes in seconds.
GRID = {"ops_read_threshold": [10, 5000], "ops_prefix_min_read_quality": [0, 0.5]}


def _defaults():
    """The four keys as a run filled from :data:`spacr.ops_settings.OPS_DEFAULTS` has them."""
    from spacr.ops_settings import OPS_DEFAULTS

    return {key: OPS_DEFAULTS[key] for key in OLD}


def test_the_new_defaults_and_the_schema_row():
    from spacr.settings import DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION
    from spacr.settings_spec import convert_settings_dict_for_gui

    assert _defaults() == {"ops_read_mode": None, "ops_prefix_threshold_tuning": "per_well_decoy",
                           "ops_prefix_decoy_k": 4, "ops_prefix_fdr_target_pct": 1.0}
    assert SETTINGS_SCHEMA_VERSION >= 944
    assert DEFAULT_CHANGES[944] == OLD
    spec = convert_settings_dict_for_gui(_defaults())
    assert spec["ops_read_mode"] == ("combo", ["log", "raw", "auto", None], None)
    assert spec["ops_prefix_threshold_tuning"][2] == "per_well_decoy"


def test_an_empty_read_mode_is_each_decodes_own_default():
    tuned = {"ops_prefix_threshold_tuning": "per_well_decoy"}
    fixed = {"ops_prefix_threshold_tuning": "none"}
    for empty in (None, "", "None"):
        assert P._prefix_read_mode({"ops_read_mode": empty}, tuned) == "auto"
        assert P._prefix_read_mode({"ops_read_mode": empty}, fixed) == "log"
        assert S._read_mode_setting({"ops_read_mode": empty}) == "log"
    assert P._prefix_read_mode({"ops_read_mode": "log"}, tuned) == "log"
    assert P._prefix_read_mode({"ops_read_mode": "raw"}, fixed) == "raw"
    with pytest.raises(ValueError, match="ops_read_mode must be one of"):
        P._prefix_read_mode({"ops_read_mode": "brightest"}, tuned)


def test_the_pre_944_values_write_what_the_decode_wrote(tmp_path):
    base = _prefix(tmp_path, "base")
    old = _prefix(tmp_path, "old", **OLD)
    _same_files(base["folder"], old["folder"])
    assert "well_choices" not in old


def test_a_file_stamped_942_folds_to_the_old_values():
    from spacr.settings import _fold_older_defaults

    folded = {}
    applied = _fold_older_defaults(folded, 942)
    assert {key: folded[key] for key in OLD} == OLD
    assert {key: applied[key] for key in OLD} == {key: (value, 944) for key, value in OLD.items()}
    current = {}
    _fold_older_defaults(current, 944)
    assert not set(OLD) & set(current)


def test_the_default_tunes_reads_auto_and_reports_one_line_per_well(tmp_path, monkeypatch):
    lines = []
    original = P.run_prefix_decode
    monkeypatch.setattr(P, "run_prefix_decode",
                        lambda settings, report=None: original(settings, report=lines.append))
    settings = {**_defaults(), "ops_prefix_tuning_grid": GRID}
    out = _prefix(tmp_path, "default", **settings)
    choice = out["tuning"]["CP9_W1"]
    assert choice["read_mode"] in ("raw", "log")
    saved = json.load(open(os.path.join(out["folder"], "quality_classes.json")))
    assert saved["settings"]["ops_read_mode"] == "auto"
    assert saved["settings"]["ops_prefix_decoy_k"] == 4
    assert out["well_choices"] == lines[-1:]
    line = lines[-1]
    assert line.startswith("prefix_decode CP9 W1 choice: read ")
    assert "chosen by auto" in line and "tuning sites" in line and "report sites" in line
    forced = _prefix(tmp_path, "forced", **{**settings, "ops_read_mode": "auto"})
    for name in ("reads_CP9_W1.parquet", "cells_CP9_W1.csv", "site_summary.csv"):
        with open(os.path.join(out["folder"], name), "rb") as a, \
                open(os.path.join(forced["folder"], name), "rb") as b:
            assert a.read() == b.read(), name


def test_tuning_off_keeps_the_log_read_with_an_empty_read_mode(tmp_path):
    out = _prefix(tmp_path, "fixed", ops_read_mode=None, ops_prefix_decoy_k=4)
    log = _prefix(tmp_path, "log", ops_read_mode="log", ops_prefix_decoy_k=4)
    _same_files(out["folder"], log["folder"])
    saved = json.load(open(os.path.join(out["folder"], "quality_classes.json")))
    assert "ops_read_mode" not in saved["settings"] and not out["tuning"]
