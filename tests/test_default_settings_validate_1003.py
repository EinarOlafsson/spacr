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
