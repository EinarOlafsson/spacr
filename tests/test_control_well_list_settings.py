"""Contracts for list-valued regression control-well settings."""

import pytest

from spacr.settings import (expected_types,
                            get_perform_regression_default_settings)


CONTROL_LIST_KEYS = (
    "exclude_grnas",
    "positive_control_wells",
    "negative_control_wells",
    "mixed_control_wells",
)


@pytest.mark.parametrize("key", CONTROL_LIST_KEYS)
def test_a_legacy_scalar_is_normalized_to_one_list_item(key):
    settings = get_perform_regression_default_settings({key: " c2 "})
    assert settings[key] == ["c2"]


@pytest.mark.parametrize("key", CONTROL_LIST_KEYS)
def test_an_empty_legacy_scalar_becomes_none(key):
    settings = get_perform_regression_default_settings({key: "  "})
    assert settings[key] is None


@pytest.mark.parametrize("key", CONTROL_LIST_KEYS)
def test_the_declared_type_selects_the_list_editor(key):
    declared = expected_types[key]
    assert list in declared
    assert str not in declared
