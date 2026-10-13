"""Snapshot and restore ``spacr.qt.app``'s process-global registry.

``spacr.qt.register_self_registering_modules()`` writes to ``APPS``,
``APP_FACTORIES``, ``APP_STAGE``, ``APP_META`` and, through
``app._META_TARGETS``, several side tables. Thirty-four test files call it --
they need the registry a launched GUI has, which importing ``spacr.qt.app``
alone does not give them -- and nine apps join the list when it runs.

WHY THESE LIVE HERE RATHER THAN IN A CONFTEST. ``tests/qt/conftest.py`` has
restored the registry since the leak was first found, and its docstring says
the helpers belong "in one place so the per-test restore and the per-module one
cannot drift apart". That was right and it was not far enough: a conftest only
covers the directory beneath it, so the protection stopped at ``tests/qt/``
while NINE of the callers are in plain ``tests/``.

Measured 2026-09-13 -- each of these, run in one process before
``tests/test_app_registry_parity.py``, makes it fail:

    test_a_settings_api_link_lands_on_the_setting
    test_the_api_homepage_shows_the_module_structure
    test_the_readme_describes_the_build_that_ships
    test_user_facing_tone

with ``FOLDED names apps that still have a registry row: ['control_chart',
'feature_explorer', 'outliers', 'trellis']``. The parity file passes alone,
every time, which is why a per-file run never found it and a batched sweep did.

Importing the folded screens does NOT reproduce it -- they register nothing on
import -- so the ghost rows come from the registration call, not from module
import. That distinction cost an experiment and is recorded so it costs nobody
another one.
"""
from __future__ import annotations

import sys


def app_registry_snapshot(app_mod):
    """Everything ``register_self_registering_modules()`` writes to.

    Driven off ``_META_TARGETS`` so a new side table is covered without an
    edit here.
    """
    side = []
    for module_name, attribute, _field in app_mod._META_TARGETS:
        module = sys.modules.get(module_name)
        table = getattr(module, attribute, None) if module else None
        if isinstance(table, dict):
            side.append((table, dict(table)))
    return (list(app_mod.APPS), dict(app_mod.APP_FACTORIES),
            dict(app_mod.APP_STAGE), dict(app_mod.APP_META), side)


def restore_app_registry_to(app_mod, snapshot):
    """Put ``snapshot`` back. ``APPS`` is rebuilt only if it actually moved."""
    apps, factories, stages, meta, side = snapshot
    if list(app_mod.APPS) != apps:
        app_mod.APPS[:] = apps
        app_mod._refresh_sections()
    app_mod.APP_FACTORIES.clear()
    app_mod.APP_FACTORIES.update(factories)
    app_mod.APP_STAGE.clear()
    app_mod.APP_STAGE.update(stages)
    app_mod.APP_META.clear()
    app_mod.APP_META.update(meta)
    for table, saved in side:
        table.clear()
        table.update(saved)
