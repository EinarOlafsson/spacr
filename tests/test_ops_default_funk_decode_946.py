"""Item 946: the default OPS decode is the proven Funk decode.

``ops_decode_preset`` defaults to ``'rule_a_s2'`` and
``ops_decode_threshold_mode`` to ``'auto'`` (calling thresholds only:
``ops_decode_auto_spot_grid`` is empty). Settings files written before keep
the decode they ran with through ``DEFAULT_CHANGES``.
"""
from spacr import ops_decode, ops_settings
from spacr.settings import (DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION,
                            _fold_older_defaults)

#: The schema row item 946 added: the highest row of DEFAULT_CHANGES.
ROW = max(key for key, row in DEFAULT_CHANGES.items()
          if "ops_decode_preset" in row)

OLD = {"ops_decode_preset": "none", "ops_decode_threshold_mode": "manual",
       "ops_decode_auto_spot_grid": [150.0, 200.0, 250.0, 315.0, 400.0]}


def test_the_new_defaults():
    defaults = ops_settings.OPS_DEFAULTS
    assert defaults["ops_decode_preset"] == "rule_a_s2"
    assert defaults["ops_decode_threshold_mode"] == "auto"
    assert defaults["ops_decode_auto_spot_grid"] == []
    tips = ops_settings.OPS_TOOLTIPS
    assert tips["ops_decode_preset"].endswith("Default 'rule_a_s2'.")
    assert tips["ops_decode_threshold_mode"].endswith("Default 'auto'.")
    assert "Default empty." in tips["ops_decode_auto_spot_grid"]


def test_the_panel_choice_defaults_to_auto():
    import spacr.settings_spec as spec

    source = open(spec.__file__, encoding="utf-8").read()
    assert "'ops_decode_threshold_mode': ('combo', ['manual', 'auto', 'dataset'], 'auto')" in source


def test_the_schema_row_keeps_older_files_on_the_old_decode():
    assert ROW <= SETTINGS_SCHEMA_VERSION
    assert DEFAULT_CHANGES[ROW] == OLD
    stamped = _fold_older_defaults({}, ROW - 1)
    assert {key: stamped[key][0] for key in OLD} == OLD
    unstamped = _fold_older_defaults({}, 0)
    assert unstamped["ops_decode_preset"][0] == "none"
    assert unstamped["ops_decode_threshold_mode"][0] == "manual"
    # Older than item 942's row: its calling-only auto mode.
    assert unstamped["ops_decode_auto_spot_grid"][0] == []
    for key in OLD:
        assert key not in _fold_older_defaults({}, SETTINGS_SCHEMA_VERSION)
    kept = {"ops_decode_preset": "rule_a_s2"}
    _fold_older_defaults(kept, ROW - 1)
    assert kept["ops_decode_preset"] == "rule_a_s2"


def test_the_defaults_resolve_to_the_rule_a_s2_auto_decode():
    settings = ops_decode.resolve_settings(dict(ops_settings.OPS_DEFAULTS))
    for key, value in ops_decode.RULE_A_S2.items():
        assert settings[key] == value, key
    rule = ops_decode.decode_rule(settings)
    assert rule["active"] and rule["decoy_k"] == 4
    thresholds = ops_decode.threshold_settings(settings, rule)
    assert thresholds["mode"] == "auto" and thresholds["spot_thresholds"] == []
    assert thresholds["fdr_max"] == 0.1
    old = dict(ops_settings.OPS_DEFAULTS, **OLD)
    assert ops_decode.resolve_settings(old) == old
