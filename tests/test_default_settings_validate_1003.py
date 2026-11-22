"""The shipped defaults of every app pass the settings validation (item 1003).

Shipped defaults cannot name the user's data, so a problem about a required input
(``src``, a database, a table, a channel) is expected; every other error means the
defaults and the validator disagree, as ``batch_size`` 0 (automatic, item 987b) did.
"""
from __future__ import annotations

import inspect
import re

import pytest

import spacr.settings as S
from spacr import cli
from spacr.validate import validate_settings

#: The settings the user must give before a run; defaults leave them empty or a placeholder.
REQUIRED_INPUTS = frozenset({
    "src", "db_path", "predictions_file", "guide_fractions_file", "images", "masks",
    "measurements", "inputs", "genotype_source", "seg_sites_src", "paired_data", "cell_channel",
    "results_folder", "target_guides"})


def _default_functions():
    """Every ``set_default_settings_*`` and ``get_*default*settings`` function of spacr.settings."""
    return sorted(name for name, fn in vars(S).items()
                  if inspect.isfunction(fn) and fn.__module__ == "spacr.settings"
                  and (name.startswith("set_default_settings_")
                       or re.match(r"get_.*default.*settings$", name)))


def _value_errors(settings, app):
    """The validation errors that are not about a required input."""
    return [str(p) for p in validate_settings(dict(settings), app)
            if p.is_error and str(p).split("]")[0].lstrip("[") not in REQUIRED_INPUTS]


def test_the_default_functions_are_found():
    names = _default_functions()
    assert "set_default_settings_preprocess_generate_masks" in names and len(names) >= 15


@pytest.mark.parametrize("name", _default_functions())
def test_every_default_function_passes_validation(name):
    fn = getattr(S, name)
    settings = fn({}) if inspect.signature(fn).parameters else fn()
    app = "mask" if name == "set_default_settings_preprocess_generate_masks" else ""
    assert _value_errors(settings, app) == []


@pytest.mark.parametrize("name", sorted(n for n, m in cli.MODULES.items()
                                        if getattr(m, "validate_key", None)))
def test_every_apps_shipped_defaults_pass_validation(name):
    module = cli.MODULES[name]
    assert _value_errors(cli.module_defaults(module), module.validate_key) == []


@pytest.mark.parametrize("value", [0, -1, None, "auto", "none", 8])
def test_mask_generation_accepts_every_batch_size_its_plan_accepts(value):
    assert not [p for p in validate_settings({"batch_size": value}, "mask")
                if p.is_error and "batch_size" in str(p)]


@pytest.mark.parametrize("app, value", [("mask", -3), ("mask", "many"), ("mask", "60.5"), ("ops", -2),
                                        ("train_cellpose", 0), ("", 0)])
def test_a_batch_size_the_app_cannot_use_is_still_refused(app, value):
    assert [p for p in validate_settings({"batch_size": value}, app)
            if p.is_error and "batch_size" in str(p)]


@pytest.mark.parametrize("app", ["mask", "ops"])
@pytest.mark.parametrize("value", ["auto_vram", "AUTO_VRAM", "whole", 0, -1, None, "none", "auto", 8])
def test_both_mask_plan_apps_accept_the_shared_batch_selectors(app, value):
    """Mask Generation and OPS admit both automatic modes and explicit counts."""
    assert not [p for p in validate_settings({"batch_size": value}, app)
                if p.is_error and p.setting == "batch_size"]


@pytest.mark.parametrize("app", ["mask", "ops"])
@pytest.mark.parametrize("key", ["batch_size", "mask_eval_batch_size"])
@pytest.mark.parametrize("written, expected", [("auto_vram", "auto_vram"), ("0", 0), ("16", 16)])
def test_cli_and_qt_keep_vram_batch_values(app, key, written, expected):
    """CLI coercion and the Qt run boundary accept the shared settings types."""
    from spacr.validate import coerce_expected_types

    value = cli.coerce_value(key, written, "auto_vram", S.expected_types, app)
    assert value == expected
    assert coerce_expected_types({key: value}, app) == {key: expected}
    assert not [p for p in validate_settings({key: value}, app)
                if p.is_error and p.setting == key]


@pytest.mark.parametrize("app", ["mask", "ops"])
def test_current_mask_defaults_enable_vram_batches_with_alpha_controls_hidden(app):
    """Hiding tuning controls leaves the new Mask and OPS batching defaults active."""
    defaults = cli.module_defaults(cli.MODULES[app])
    assert {key: defaults[key] for key in ("batch_size", "mask_eval_batch_size")} == {
        "batch_size": "auto_vram", "mask_eval_batch_size": "auto_vram"}
    assert {key: defaults[key] for key in ("mask_eval_batch_cap", "mask_batch_seed",
                                         "gpu_vram_budget_gb", "randomize")} == {
        "mask_eval_batch_cap": 32, "mask_batch_seed": 42,
        "gpu_vram_budget_gb": None, "randomize": True}
    assert S.ALPHA_FEATURES[1011] == {"settings": (
        "mask_eval_batch_cap", "mask_batch_seed", "gpu_vram_budget_gb")}
    assert not S._is_alpha("settings", "batch_size")
    assert not S._is_alpha("settings", "mask_eval_batch_size")
    from spacr.qt.help_api_index import SETTING_CONSUMERS

    assert ("spacr.ops_settings", "ops_defaults") in SETTING_CONSUMERS["randomize"]


@pytest.mark.parametrize("version, expected_fields, expected_tiles", [
    (0, 50, 0), (944, 50, 0), (945, 0, 0), (954, 0, 0),
    (955, "auto_vram", "auto_vram")])
@pytest.mark.parametrize("app", ["mask", "ops"])
def test_schema_955_keeps_each_historical_batch_default(version, expected_fields, expected_tiles, app):
    """Each settings era keeps its original field and tile request."""
    from spacr.ops_settings import ops_defaults

    factory = S.set_default_settings_preprocess_generate_masks if app == "mask" else ops_defaults
    known = factory({})
    loaded = {}
    S._fold_older_defaults(loaded, version, known)
    defaults = factory(loaded)
    assert defaults["batch_size"] == expected_fields
    assert defaults["mask_eval_batch_size"] == expected_tiles
    assert S.SETTINGS_SCHEMA_VERSION >= 955
    assert S.DEFAULT_CHANGES[955] == {"batch_size": 0, "mask_eval_batch_size": 0, "randomize": False}
    if app == "ops" and version < 948:
        assert [defaults[key] for key in ("ops_ph_tile_batch", "ops_sbs_tile_batch")] == [8, 16]


@pytest.mark.parametrize("version", [0, 954, 955])
@pytest.mark.parametrize("explicit", [{"batch_size": 7, "mask_eval_batch_size": 12},
                                      {"batch_size": "whole", "mask_eval_batch_size": 0},
                                      {"batch_size": "auto_vram", "mask_eval_batch_size": "auto_vram"}])
def test_schema_955_preserves_explicit_batch_selectors(version, explicit):
    """Folding old defaults never changes a request explicitly saved by the user."""
    loaded = dict(explicit)
    S._fold_older_defaults(loaded, version)
    assert {key: loaded[key] for key in explicit} == explicit


@pytest.mark.parametrize("key", ["mask_eval_batch_cap", "mask_batch_seed", "gpu_vram_budget_gb"])
def test_new_batch_controls_have_qt_links_and_shared_types(key):
    """Every new control reaches the panel, tooltip and all three value indices."""
    from spacr.ops_settings import OPS_CATEGORIES, OPS_TYPES, ops_defaults
    from spacr.qt.help_api_index import SETTING_CONSUMERS
    from spacr.qt.screens.setting_api_targets import SETTING_API_TARGETS
    from spacr.qt.screens.settings_flow_index import SETTINGS_WITH_A_FLOW_SECTION

    assert S.expected_types[key] == OPS_TYPES[key]
    assert key in S.tooltips and key in ops_defaults({})
    assert key in {name for values in S.categories.values() for name in values}
    assert key in {name for values in OPS_CATEGORIES.values() for name in values}
    assert key in SETTINGS_WITH_A_FLOW_SECTION
    assert SETTING_API_TARGETS[key] == ("spacr.ops_settings", "ops_defaults", True)
    assert SETTING_CONSUMERS[key] == (("spacr.ops_settings", "ops_defaults"),
                                      ("spacr.settings", "set_default_settings_preprocess_generate_masks"))


@pytest.mark.parametrize("value", [0, -1])
def test_mask_tile_cap_cannot_disable_the_one_tile_floor(value):
    """Invalid caps fail at validation before either mask application starts."""
    assert [p for p in validate_settings({"mask_eval_batch_cap": value}, "mask")
            if p.is_error and p.setting == "mask_eval_batch_cap"]


@pytest.mark.parametrize("key, written, expected", [
    ("batch_size", "auto_vram", "auto_vram"),
    ("mask_eval_batch_size", "auto_vram", "auto_vram"),
    ("mask_eval_batch_cap", "24", 24),
    ("mask_batch_seed", "-12", -12),
    ("gpu_vram_budget_gb", "8.0", 8.0),
    ("gpu_vram_budget_gb", "None", None)])
def test_qt_and_cli_parse_shared_mask_controls(key, written, expected):
    """The settings widgets and CLI retain selectors and parse new numeric controls."""
    from spacr.qt.screens.settings_model import SettingsWidgets

    assert SettingsWidgets._coerce_to_expected_type(key, written) == expected
    assert cli.coerce_value(key, written, None, S.expected_types, "mask") == expected


@pytest.mark.parametrize("app", ["mask", "ops"])
@pytest.mark.parametrize("version, expected_fields, expected_tiles", [
    (None, 50, 0), (954, 0, 0), (955, "auto_vram", "auto_vram")])
def test_cli_loads_each_saved_batch_schema(tmp_path, app, version, expected_fields, expected_tiles):
    """Loading a real saved file folds old defaults before the app fills new ones."""
    import json

    payload = {} if version is None else {S.SETTINGS_VERSION_KEY: version}
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = cli.resolve_settings(cli.MODULES[app], str(path))
    assert loaded["batch_size"] == expected_fields
    assert loaded["mask_eval_batch_size"] == expected_tiles


@pytest.mark.parametrize("key", ["batch_size", "mask_eval_batch_size"])
def test_shared_settings_type_admits_vram_selector_at_the_legacy_gui_boundary(key):
    """Both GUI settings boundaries can collect the automatic selector."""
    from unittest.mock import Mock

    variable = Mock()
    variable.get.return_value = "auto_vram"
    collected, errors = S.check_settings({key: (None, None, variable, None)},
                                        S.expected_types, q=Mock())
    assert not errors
    assert collected[key] == "auto_vram"


@pytest.mark.parametrize("app", ["train_cellpose", ""])
def test_vram_field_selector_does_not_widen_numeric_batch_apps(app):
    """Apps without a mask plan continue to require numerical minibatches."""
    with pytest.raises(cli.SettingsError):
        cli.coerce_value("batch_size", "auto_vram", 8, S.expected_types, app)
    assert [p for p in validate_settings({"batch_size": "auto_vram"}, app)
            if p.is_error and p.setting == "batch_size"]


@pytest.mark.parametrize('route', ['cli', 'repro', 'qt'])
@pytest.mark.parametrize('app', ['mask', 'ops'])
@pytest.mark.parametrize('version', [None, 954, 955])
@pytest.mark.parametrize('explicit', [None, True, False])
def test_saved_randomize_preserves_each_pipeline_and_explicit_choice(tmp_path, route, app, version, explicit):
    """Saved settings reach the shared field-order helper with the original app default."""
    import json
    from types import SimpleNamespace
    from spacr._mask_batches import _field_order
    from spacr.cli_repro import _replayed_settings
    from spacr.ops_settings import ops_defaults
    from spacr.qt.screens.app_screen import AppScreen

    payload={'batch_size':2,'mask_eval_batch_size':0}
    if version is not None:
        payload[S.SETTINGS_VERSION_KEY]=version
    if explicit is not None:
        payload['randomize']=explicit
    module=cli.MODULES[app]
    if route=='cli':
        path=tmp_path/'saved.json'
        path.write_text(json.dumps(payload),encoding='utf-8')
        loaded=cli.resolve_settings(module,str(path))
    elif route=='repro':
        loaded=cli.module_defaults(module)
        loaded.update(_replayed_settings(dict(payload),app))
    else:
        known=cli.module_defaults(module)
        screen=SimpleNamespace(_settings_model=SimpleNamespace(collect=lambda:known))
        loaded=cli.module_defaults(module)
        loaded.update(AppScreen._in_its_own_era(screen,dict(payload)))
    if app=='ops':
        loaded=ops_defaults(loaded)
    expected=explicit if explicit is not None else (app=='mask' or version==955)
    assert loaded['randomize'] is expected
    assert loaded['batch_size']==2 and loaded['mask_eval_batch_size']==0
    fields=[f'field{i}' for i in range(7)]
    ordered=_field_order(fields,loaded)
    assert (ordered!=fields) is expected
    if not expected:
        assert ordered==fields
