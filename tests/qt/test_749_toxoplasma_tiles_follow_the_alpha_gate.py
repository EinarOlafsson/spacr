"""The Toxoplasma page shows the module tiles Home shows, and says which are planned.

Maintainer decision of 2026-11-01 (UI audit): Host–Pathogen and Invasion
Assay are alpha-stage modules, so their tiles appear only with Preferences >
Show Alpha modules on, the gate Home uses for its own tiles. Egress,
Bradyzoite conversion and Host cell damage are greyed because they are not
built yet, and each says "Planned" under its name.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QLabel

from spacr.qt.screens.organism_screen import OrganismScreen
from spacr.qt.widgets.home import AppTile

ALPHA_TILES = {"host_pathogen", "invasion"}
PLANNED = {"Egress", "Bradyzoite conversion", "Host cell damage"}


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "gate.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def _shown(page):
    return {tile.property("organismModuleKey") or tile.text_label
            for tile in page.findChildren(AppTile) if not tile.isHidden()}


def _in_grid(page):
    return {page._grid.itemAt(i).widget() for i in range(page._grid.count())}


def test_alpha_stage_tiles_follow_show_alpha_modules(qtbot, prefs):
    from spacr.qt.app import app_is_visible

    prefs.set_show_alpha(False)
    assert not any(app_is_visible(key) for key in ALPHA_TILES)
    page = OrganismScreen("toxoplasma")
    qtbot.addWidget(page)
    page.resize(1400, 900)
    page.show()
    qtbot.waitUntil(lambda: page._columns > 0)
    assert not (_shown(page) & ALPHA_TILES)
    assert {"analyze_plaques", "recruitment", "replication"} <= _shown(page)
    assert not any(tile.property("organismModuleKey") in ALPHA_TILES
                   for tile in _in_grid(page))

    prefs.set_show_alpha(True)
    page.refresh_maturity_visibility()
    assert ALPHA_TILES <= _shown(page)
    assert set(page._tiles) == _in_grid(page)

    prefs.set_show_alpha(False)
    page.refresh_maturity_visibility()
    assert not (_shown(page) & ALPHA_TILES)
    assert set(page._tiles) == _in_grid(page)


def test_greyed_tiles_say_planned(qtbot, prefs):
    prefs.set_show_alpha(False)
    page = OrganismScreen("toxoplasma")
    qtbot.addWidget(page)
    page.resize(1400, 900)
    page.show()
    qtbot.waitUntil(lambda: page._columns > 0)
    greyed = [tile for tile in page.findChildren(AppTile) if not tile.isEnabled()]
    assert {tile.text_label for tile in greyed} == PLANNED
    for tile in greyed:
        caption = tile.findChild(QLabel, "OrganismTilePlanned")
        assert caption is not None and caption.isVisible()
        assert caption.text() == "Planned"
        assert caption.geometry().bottom() <= tile.contentsRect().bottom()
        assert caption.geometry().top() >= tile.name_label.geometry().bottom()
    for tile in page.findChildren(AppTile):
        if tile.isEnabled():
            assert tile.findChild(QLabel, "OrganismTilePlanned") is None
