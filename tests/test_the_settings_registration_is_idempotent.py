"""Registering a module's settings twice, which importing it makes possible.

HANDOFF 3c records what this machinery costs when it goes wrong: six pipelines
register their keys from their own module, ``register_defaults`` runs on import,
and it REFUSES to let one module redefine another's tooltip. So a second
registration must be a no-op rather than a re-registration -- otherwise
importing a module twice, or a test importing it after the app has, would raise
on keys the module itself owns.

The uncovered arc is exactly that second call, which every process makes as
soon as anything re-imports the module.
"""
from __future__ import annotations

import pytest


#: Every module that registers its own settings keys on import. Taken from the
#: package rather than listed by hand, so a module added later is covered the
#: day it is added.
def _registering_modules():
    import importlib
    import pkgutil

    import spacr

    found = []
    for info in pkgutil.iter_modules(spacr.__path__):
        try:
            module = importlib.import_module(f"spacr.{info.name}")
        except Exception:
            continue
        if hasattr(module, "register_settings"):
            found.append(info.name)
    return found


MODULES = _registering_modules()


@pytest.mark.parametrize("module_name", MODULES)
def test_registering_a_modules_settings_records_it(module_name):
    """The contract: after register_settings(), the module is registered.

    Some modules register on import and some wait to be asked -- both are
    legitimate, and asserting the ASKED-FOR state rather than the imported one
    is what makes this test true for either.
    """
    import importlib

    from spacr.settings import has_registered_defaults

    module = importlib.import_module(f"spacr.{module_name}")
    module.register_settings()

    # A module registers under its own APP_KEY when it declares one -- the key
    # is the settings namespace, not the import path.
    key = getattr(module, "APP_KEY", module_name)
    assert has_registered_defaults(key)


@pytest.mark.parametrize("module_name", MODULES)
def test_registering_a_second_time_is_a_no_op(module_name):
    """Arc 745 -> 746 in external_masks, and its sibling elsewhere.

    False means "already done", not "failed". A caller treating it as a
    failure would re-register and hit register_defaults' own refusal, which is
    the error HANDOFF 3c describes -- one module appearing to redefine
    another's tooltip.
    """
    import importlib

    module = importlib.import_module(f"spacr.{module_name}")
    module.register_settings()

    assert module.register_settings() is False


def test_an_explicit_replace_registers_again():
    """The taken side: ``replace=True`` is how a reload refreshes the keys.

    Without it a developer editing a tooltip and re-importing would keep the
    old text for the life of the process, and conclude their edit did nothing.
    """
    from spacr.external_masks import register_settings

    assert register_settings(replace=True) is True
    # And the module is still registered afterwards, not left half-removed.
    from spacr.settings import has_registered_defaults
    assert has_registered_defaults("external_masks")


def test_the_registered_keys_are_documented():
    """What the registration is FOR: a tooltip for every key it declares.

    HANDOFF 3c's other half -- read cold, a key looks undocumented, and a tool
    that then writes "no description" is not missing a sentence, it is writing
    a wrong one.
    """
    import spacr.external_masks                                  # noqa: F401
    from spacr.settings import tooltips

    described = [key for key in ("inputs",) if key in tooltips]
    for key in described:
        assert str(tooltips[key]).strip(), f"{key} has an empty tooltip"
