"""Every OPS setting has a type, a category and help -- checked, not assumed.

Instruction 364 found what an unregistered module costs: a setting absent from
the shared tables is never validated, and that is how a checkbox came to ship
the string ``'False'`` -- truthy, silently, for a whole release. `spacrops.py`
carried sixty-three settings and not one of them was declared.

This is a RATCHET on that. It compares the declarations against the settings
factory itself, so adding a setting without declaring it fails here rather
than shipping a box nobody can validate or explain.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def declared():
    """The settings the factory produces, and what has been declared for them."""
    from spacr import ops_settings
    from spacr.spacrops import get_preprocess_ops_settings

    keys = set(get_preprocess_ops_settings({}))
    categorised = {k for keys_ in ops_settings.OPS_CATEGORIES.values()
                   for k in keys_}
    return keys, ops_settings, categorised


def test_no_setting_is_untyped(declared):
    """An untyped setting cannot be validated, so a wrong value reaches a run.

    A key may be typed HERE or already typed by another module -- `src` is
    shared, and `register_defaults` rightly refuses to let one module rewrite
    another's declaration. What matters is that nothing is untyped anywhere.
    """
    from spacr import settings as shared

    keys, ops_settings, _cat = declared
    missing = sorted(k for k in keys
                     if k not in ops_settings.OPS_TYPES
                     and k not in shared.expected_types)
    assert not missing, f"no declared type for: {missing}"


def test_no_setting_is_without_help(declared):
    """A setting with no tooltip is a labelled box the user has to guess at."""
    from spacr import settings as shared

    keys, ops_settings, _cat = declared
    missing = sorted(k for k in keys
                     if k not in ops_settings.OPS_TOOLTIPS
                     and k not in shared.tooltips)
    assert not missing, f"no tooltip for: {missing}"

    # A tooltip that only restates the label helps nobody. This is a floor on
    # effort, not a style rule: every one of these should say what the setting
    # does to the RESULT.
    thin = sorted(k for k, v in ops_settings.OPS_TOOLTIPS.items()
                  if len(v.split()) < 6)
    assert not thin, f"tooltips too thin to be worth reading: {thin}"


def test_no_setting_is_uncategorised(declared):
    """A setting in no category does not appear in the panel at all."""
    keys, _ops, categorised = declared
    missing = sorted(keys - categorised)
    assert not missing, f"in no category, so invisible in the GUI: {missing}"


def test_nothing_is_declared_that_does_not_exist(declared):
    """A declaration for a setting the factory does not produce is stale.

    It is harmless in itself and it is evidence that a setting was renamed or
    removed without the tables following, which is how they drift out of date.
    """
    keys, ops_settings, categorised = declared
    declared_keys = (set(ops_settings.OPS_TYPES)
                     | set(ops_settings.OPS_TOOLTIPS) | categorised)
    extra = sorted(declared_keys - keys)
    assert not extra, f"declared but not produced by the factory: {extra}"


def test_registering_twice_is_not_an_error(declared):
    """A module imported from both the GUI and a headless run registers twice."""
    _keys, ops_settings, _cat = declared
    ops_settings.register()
    ops_settings.register()


def test_the_registration_reaches_the_shared_tables(declared):
    """After registering, the shared tables must actually carry the settings."""
    keys, ops_settings, _cat = declared
    from spacr import settings as shared

    ops_settings.register()
    for key in sorted(keys):
        assert key in shared.expected_types, f"{key} never reached expected_types"
    assert shared.descriptions.get("ops"), "the module blurb did not register"
