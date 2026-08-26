"""The factory the app registry calls, and the guard against a second row.

``register_app`` stores a callable rather than a widget so a screen is not
built until someone opens it. That callable is the only thing standing
between the registry row and a live screen, and an import-time ``register()``
that ran twice would put two rows in the sidebar for one app.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import curate as curate_module
from spacr.qt.screens import lineage as lineage_module


def test_the_lineage_factory_builds_a_live_screen(qtbot):
    screen = lineage_module.make_lineage_screen()
    qtbot.addWidget(screen)

    assert isinstance(screen, lineage_module.LineageScreen)
    assert screen.tree.topLevelItemCount() == 0
    screen.unlink_selection()


def test_the_curate_factory_builds_a_live_screen(qtbot):
    screen = curate_module.make_curate_screen()
    qtbot.addWidget(screen)

    assert isinstance(screen, curate_module.CurateScreen)


@pytest.mark.parametrize("module", [lineage_module, curate_module])
def test_registering_a_screen_twice_adds_one_row(module):
    """Two rows for one app would each open their own copy of the screen."""
    from spacr.qt.app import APPS

    before = [row for row in APPS if row[0] == module.APP_KEY]
    first = module.register()
    after = [row for row in APPS if row[0] == module.APP_KEY]
    assert len(after) == 1

    assert module.register() is None
    assert [row for row in APPS if row[0] == module.APP_KEY] == after

    if not before and first is not None:
        APPS.remove(first)
