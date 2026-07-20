"""
Tests for spacr.settings — mostly pure dict-manipulation helpers, so
these are quick and correctness-focused.
"""
from __future__ import annotations

import pytest

import spacr.settings as S


# ---------------------------------------------------------------------------
# 1. Every set_default_* helper: shape + None-safety + idempotence.
# ---------------------------------------------------------------------------

# The public setters that take a `settings` dict argument. All should
# accept None (thanks to the recent mutable-default refactor) OR treat a
# missing arg as "start empty".
SETTERS_NULLABLE = [
    "set_default_settings_preprocess_generate_masks",
    "set_default_umap_image_settings",
    "get_measure_crop_settings",
]

SETTERS_REQUIRED = [
    "set_default_plot_data_from_db",
    "set_default_settings_preprocess_img_data",
    "set_default_analyze_screen",
    "set_default_train_test_model",
    "set_generate_training_dataset_defaults",
    "deep_spacr_defaults",
    "get_train_test_model_settings",
    "get_analyze_recruitment_default_settings",
    "get_default_test_cellpose_model_settings",
    "get_default_apply_cellpose_model_settings",
    "default_settings_analyze_percent_positive",
    "get_analyze_reads_default_settings",
    "get_map_barcodes_default_settings",
    "get_train_cellpose_default_settings",
    "set_generate_dataset_defaults",
]


@pytest.mark.parametrize("fname", SETTERS_NULLABLE)
def test_setter_accepts_none_yields_dict(fname):
    fn = getattr(S, fname)
    out = fn(None)
    assert isinstance(out, dict) and len(out) > 0


@pytest.mark.parametrize("fname", SETTERS_NULLABLE)
def test_setter_none_and_empty_dict_agree(fname):
    fn = getattr(S, fname)
    a = fn(None)
    b = fn({})
    assert a == b, f"{fname}: passing None vs {{}} should give same defaults"


@pytest.mark.parametrize("fname", SETTERS_NULLABLE)
def test_setter_no_shared_state(fname):
    """Two calls with omitted arg must NOT share the same dict object."""
    fn = getattr(S, fname)
    a = fn(None)
    b = fn(None)
    a["__test_key__"] = "polluted"
    assert "__test_key__" not in b, (
        f"{fname}: default-arg dicts are shared between calls (mutable-default regression)"
    )


@pytest.mark.parametrize("fname", SETTERS_REQUIRED)
def test_setter_with_empty_dict_returns_populated_dict(fname):
    fn = getattr(S, fname)
    out = fn({})
    assert isinstance(out, dict)
    # Every set_default_*/get_*_settings function should have added at least
    # a couple of keys — otherwise it's a no-op.
    assert len(out) >= 2, f"{fname}({{}}) produced only {out}"


@pytest.mark.parametrize("fname", SETTERS_NULLABLE + SETTERS_REQUIRED[:5])
def test_setter_preserves_caller_supplied_values(fname):
    """setdefault semantics: user values must survive."""
    fn = getattr(S, fname)
    sentinel = "__user_supplied__"
    out = fn({"src": sentinel})
    assert out.get("src") == sentinel


# ---------------------------------------------------------------------------
# 2. Module-level dicts have expected shape and are read-only-ish.
# ---------------------------------------------------------------------------

DICT_ATTRS = [
    "categories", "category_dependencies", "category_group_dependencies",
    "category_integer_dependencies", "category_value_dependencies",
    "descriptions", "expected_types", "tooltips",
]


@pytest.mark.parametrize("name", DICT_ATTRS)
def test_module_dict_is_non_empty(name):
    d = getattr(S, name)
    assert isinstance(d, dict) and len(d) > 0, f"{name} should be a non-empty dict"


def test_descriptions_mask_help_typo_fixes_persisted():
    """Direct assertion for the fix(settings) typos in the 'mask' help text."""
    mask = S.descriptions["mask"]
    assert "Downloade" not in mask
    assert "menue" not in mask
    assert "Download the training set" in mask
    assert "menu bar" in mask


def test_expected_types_has_common_keys():
    et = S.expected_types
    # A handful of settings ubiquitous across spacr modules.
    for k in ("src", "channels"):
        assert k in et, f"expected_types missing common key {k!r}"


# ---------------------------------------------------------------------------
# 3. Object-settings helper _get_object_settings shape checks.
# ---------------------------------------------------------------------------

def test_get_object_settings_returns_dict_per_object_type():
    base = S.set_default_settings_preprocess_generate_masks({})
    for object_type in ("cell", "nucleus", "pathogen"):
        out = S._get_object_settings(object_type, base)
        assert isinstance(out, dict)
        assert len(out) > 0
