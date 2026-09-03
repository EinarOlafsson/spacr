"""The dock's top level is Home's tiles, and nothing is lost or doubled.

Asked for on 2026-09-03: "the moduals that should be in the Help category are
not yet and potentially some moduals that should be nested are not. the
modules in the dock should mimic the screen module tiles
(core:6;data:6;tools:5;assays:4) everything else is nested and in help
dropdown."

MEASURED before the change, one 1440x900 MainWindow:

    top-level dock rows   36  ->  27   (19 tiles + 8 Help)
    Home tiles            19  ->  19   (unchanged)
    keys drawn TWICE       9  ->   0
    keys unreachable       0  ->   0

The nine doubled keys were ``convert``, ``external_masks``,
``investigate_hit``, ``layer_viewer``, ``lineage``, ``plate_view``,
``profiler``, ``tabulate`` and ``train_compare`` -- each had a top-level row
AND an indented row under the host it folds into.

THE COUNTS IN THE REQUEST ARE data:6 AND tools:5; the registry's tiles are
data:5 and tools:4. This file asserts the dock MATCHES HOME rather than
either set of numbers, which is the property that stops the two drifting
again: promoting a module to a tile gives it a dock row and folding one takes
its dock row away, with no list here to update.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.app import (APPS, SECTION_HELP, SECTION_ORDER, _HELP_MODULES,
                          dock_rows, folded_children, section_members,
                          tiled_apps)


def test_every_top_level_dock_row_is_a_home_tile_or_a_help_module():
    """The one rule. A row that is neither is a row Home cannot explain."""
    tiles = {row[0] for row in tiled_apps()}
    helpers = {row[0] for row in _HELP_MODULES}
    stray = [key for key, *_ in dock_rows()
             if key not in tiles and key not in helpers]
    assert not stray, f"these dock rows are neither a tile nor Help: {stray}"


def test_every_home_tile_has_a_dock_row():
    """And the other direction, which is what "mimic" means.

    Asserted per section, because a tile that lost its dock row would still
    leave the totals right if another gained one.
    """
    listed = {key: section for key, _n, _d, section in dock_rows()}
    for section in SECTION_ORDER:
        tiles = [row[0] for row in section_members(section)]
        missing = [key for key in tiles if key not in listed]
        assert not missing, f"{section} tiles with no dock row: {missing}"
        wrong = {key: listed[key] for key in tiles
                 if listed[key] != section}
        assert not wrong, f"{section} tiles filed elsewhere in the dock: {wrong}"


def test_the_help_heading_holds_every_help_menu_module():
    """The Help modules are tileless, so the dock is where they live."""
    listed = {key: section for key, _n, _d, section in dock_rows()}
    registered = {row[0] for row in APPS}
    for key, *_ in _HELP_MODULES:
        if key not in registered:
            # Declared for the Help MENU but not a registered module. It has
            # no dock row to be in the wrong place.
            continue
        assert listed.get(key) == SECTION_HELP, (
            f"{key} is a Help-menu module but the dock files it under "
            f"{listed.get(key)!r}")


def test_no_module_appears_twice_in_the_dock():
    """A module in two places is one the reader cannot learn the place of."""
    top = [key for key, *_ in dock_rows()]
    assert len(top) == len(set(top)), "a key has two top-level dock rows"
    children = set()
    for host, kids in folded_children().items():
        children.update(kids)
    doubled = sorted(set(top) & children)
    assert not doubled, (
        f"these keys have BOTH a top-level row and an indented one: "
        f"{doubled}")


def test_no_module_falls_out_of_the_dock_altogether():
    """Every registered module is a row, or a child of one that is.

    THE LOAD-BEARING TEST OF THIS CHANGE. Narrowing the top level to
    Home's tiles removed seventeen rows; each one has to still be reachable
    from the dock, or the change hid a module instead of nesting it.
    """
    top = {key for key, *_ in dock_rows()}
    reachable = set(top)
    folded = folded_children()
    for host in top:
        reachable.update(folded.get(host, ()))
    lost = sorted({row[0] for row in APPS} - reachable)
    assert not lost, (
        f"these modules have no dock row and no host that does: {lost}")


def test_the_dock_draws_what_the_registry_says(qtbot, qt_theme_applied):
    """The rendered dock, not just the function that feeds it."""
    from spacr.qt.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    dock = win._sidebar
    hosts = [str(row.property("navKey")) for row in dock._items
             if not row.property("isFoldChild")]
    hosts = [key for key in hosts if key != "__home__"]
    assert hosts == [key for key, *_ in dock_rows()], (
        "the dock's rows are not the registry's dock rows")
    assert set(dock._section_headers) == set(SECTION_ORDER) | {SECTION_HELP}
