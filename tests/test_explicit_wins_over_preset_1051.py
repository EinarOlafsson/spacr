"""Item 1051 (schema 985): a value the user set beats a preset, even at the default."""
import csv

from spacr import ops_decode
from spacr.ops_settings import ops_defaults
from spacr.settings import (EXPLICIT_KEYS_KEY, LEGACY_EXPLICIT,
                            SETTINGS_SCHEMA_VERSION, SETTINGS_VERSION_KEY,
                            _fold_older_defaults, record_explicit_keys)


def test_the_schema_row_exists():
    assert SETTINGS_SCHEMA_VERSION >= 985


def test_an_explicit_default_valued_key_beats_the_preset():
    resolved = ops_decode.resolve_settings(ops_defaults(
        {"ops_decode_preset": "rule_a_s2", "ops_pixel_quality": 0.2,
         "ops_single_read_quality": 0.03, "ops_footprint": 3.0}))
    assert resolved["ops_pixel_quality"] == 0.2
    assert resolved["ops_single_read_quality"] == 0.03
    assert resolved["ops_footprint"] == 3.0


def test_an_unset_key_gets_the_preset():
    resolved = ops_decode.resolve_settings(ops_defaults(
        {"ops_decode_preset": "rule_a_s2"}))
    assert resolved["ops_pixel_quality"] == 0.1
    assert resolved["ops_single_read_quality"] == 0.06
    assert resolved["ops_channel_registration"] == "spots_refine"


def test_a_direct_mapping_with_a_record_is_read_by_the_record():
    given = record_explicit_keys({"ops_decode_preset": "rule_a_s2",
                                  "ops_pixel_quality": 0.2})
    assert ops_decode.resolve_settings(given)["ops_pixel_quality"] == 0.2


def test_an_old_stamped_file_replays_the_old_resolution():
    given = {"ops_decode_preset": "rule_a_s2", "ops_pixel_quality": 0.2}
    _fold_older_defaults(given, 983, known={"ops_decode_preset", "ops_pixel_quality"})
    assert given[EXPLICIT_KEYS_KEY] == LEGACY_EXPLICIT
    resolved = ops_decode.resolve_settings(ops_defaults(given))
    assert resolved["ops_pixel_quality"] == 0.1


def test_a_current_stamped_file_keeps_its_values(tmp_path):
    from spacr.cli import resolve_module, resolve_settings

    ops = resolve_module("ops")
    path = tmp_path / "s.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Key", "Value"])
        writer.writerows([(SETTINGS_VERSION_KEY, str(SETTINGS_SCHEMA_VERSION)),
                          ("ops_pixel_quality", "0.2")])
    loaded = resolve_settings(ops, str(path))
    assert ops_decode.resolve_settings(loaded)["ops_pixel_quality"] == 0.2
    old = tmp_path / "old.csv"
    with open(old, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Key", "Value"])
        writer.writerows([(SETTINGS_VERSION_KEY, "983"), ("ops_pixel_quality", "0.2")])
    replay = resolve_settings(ops, str(old))
    assert replay[EXPLICIT_KEYS_KEY] == LEGACY_EXPLICIT
    assert ops_decode.resolve_settings(replay)["ops_pixel_quality"] == 0.1


def test_the_in_situ_preset_leaves_an_explicit_default_alone():
    from spacr.ops_insitu import _insitu_options

    chosen = {"ops_quality_min_reads": 2, "ops_prefix_quality_method": "spacr"}
    base = {"ops_insitu_preset": "t7"}
    assert _insitu_options(base, chosen)["min_reads"] == 1
    explicit = record_explicit_keys({**base, "ops_quality_min_reads": 2})
    assert _insitu_options(explicit, chosen)["min_reads"] is None


def test_a_mapping_still_carrying_an_old_stamp_replays_the_old_resolution():
    given = {SETTINGS_VERSION_KEY: 931, "ops_decode_preset": "rule_a_s2",
             "ops_pixel_quality": 0.2}
    resolved = ops_decode.resolve_settings(ops_defaults(given))
    assert resolved["ops_pixel_quality"] == 0.1
