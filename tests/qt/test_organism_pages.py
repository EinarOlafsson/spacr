"""Organism pages preserve assay routes and keep proposals inert."""
from __future__ import annotations

import hashlib
import json

import pytest
from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import QLabel, QMainWindow, QSizePolicy, QStackedWidget

from spacr.qt import app
from spacr.qt.command_palette import CommandPalette
from spacr.qt.organisms import ORGANISMS
from spacr.qt.screens.organism_screen import OrganismScreen, _IMAGES
from spacr.qt.preferences import scaled_px
from spacr.qt.theme import TILE_H, TILE_MAX_W, TILE_W
from spacr.qt.widgets.organism_diagram import _diagram_svg


ASSAYS = ("analyze_plaques", "recruitment", "invasion", "replication")


def test_home_offers_three_organisms_and_keeps_all_assay_registry_keys():
    assert [row[0] for row in app.section_members(app.SECTION_ASSAYS)] == [
        "toxoplasma", "plasmodium", "candida"]
    assert set(ASSAYS) <= {row[0] for row in app.APPS}
    assert not set(ASSAYS) & {row[0] for row in app.tiled_apps()}
    assert app._ICON_OVERRIDES["toxoplasma"] == "replication.png"


@pytest.mark.parametrize("key", ORGANISMS)
def test_each_organism_has_eight_home_tiles_and_a_credited_cell_diagram(
        qtbot, qt_theme_applied, key):
    screen = OrganismScreen(key)
    qtbot.addWidget(screen)
    screen.resize(1280, 800)
    screen.show()
    qtbot.wait(30)
    assert len(screen._tiles) == 8
    assert screen._diagram.artwork.renderer.isValid()
    credits = screen.findChildren(QLabel, "OrganismImageCredit")
    assert len(credits) == 1
    assert "Philippe Le Mercier" in credits[0].text()
    assert "swissbiopics.org" in credits[0].text()
    assert len(screen.findChildren(QLabel, "OrganismSectionText")) == 4
    for tile in screen._tiles:
        assert tile.sizeHint().width() == scaled_px(TILE_W)
        assert tile.height() == scaled_px(TILE_H)
        assert tile.maximumWidth() == scaled_px(TILE_MAX_W)
        assert tile.sizePolicy().horizontalPolicy() == QSizePolicy.Preferred
    assert all(not tile.is_name_elided() for tile in screen._tiles)
    assert screen._scroll.horizontalScrollBar().maximum() == 0
    assert screen._module_scroll.horizontalScrollBar().maximum() == 0


@pytest.mark.parametrize("key", ORGANISMS)
def test_proposals_cannot_navigate_and_explain_their_status(qtbot, key):
    screen = OrganismScreen(key)
    qtbot.addWidget(screen)
    requested = []
    screen.module_requested.connect(requested.append)
    proposals = [tile for tile in screen._tiles
                 if not tile.property("organismModuleKey")]
    assert len(proposals) == (4 if key == "toxoplasma" else 8)
    for tile in proposals:
        assert not tile.isEnabled()
        assert tile.graphicsEffect().opacity() < 0.5
        assert tile.toolTip().startswith("Coming soon")
        tile.click()
    assert requested == []


def test_the_four_existing_assays_emit_their_unchanged_keys(qtbot):
    screen = OrganismScreen("toxoplasma")
    qtbot.addWidget(screen)
    requested = []
    screen.module_requested.connect(requested.append)
    for tile in screen._tiles[:4]:
        assert tile.isEnabled()
        qtbot.mouseClick(tile, Qt.LeftButton)
    assert requested == list(ASSAYS)


@pytest.mark.parametrize("key", ORGANISMS)
def test_registry_factory_builds_the_page_and_wires_navigation(qtbot, key):
    class Host:
        def __init__(self):
            self.selected = []

        def _on_nav_selected(self, selected):
            self.selected.append(selected)

    host = Host()
    factory = app.registered_factory(key)
    screen = app._call_screen_factory(factory, key, host)
    qtbot.addWidget(screen)
    assert isinstance(screen, OrganismScreen)
    assert screen.app_key == key
    if key == "toxoplasma":
        screen._tiles[0].click()
        assert host.selected == ["analyze_plaques"]


def test_command_palette_keeps_direct_assay_navigation(qtbot):
    window = QMainWindow()
    window._stack = QStackedWidget(window)
    window._screens = {}
    selected = []
    window._on_nav_selected = selected.append
    qtbot.addWidget(window)
    palette = CommandPalette(window)
    qtbot.addWidget(palette)
    for key in ASSAYS:
        commands = [command for command in palette._commands
                    if key in command.keywords]
        assert len(commands) == 1
        commands[0].action()
    assert selected == list(ASSAYS)


def test_bundled_artwork_matches_its_attributed_source_records():
    records = json.loads((_IMAGES / "organism_sources.json").read_text())
    assert len(records) == 2
    for record in records:
        assert record["licence"] == "CC BY 4.0"
        assert "Philippe Le Mercier" in record["credit"]
        assert record["modifications"]
        assert hashlib.sha256((_IMAGES / record["file"]).read_bytes()).hexdigest() == record["sha256"]


@pytest.mark.parametrize("key", ORGANISMS)
def test_narrow_pages_reflow_without_clipping_tile_names(qtbot, qt_theme_applied, key):
    screen = OrganismScreen(key)
    qtbot.addWidget(screen)
    screen.resize(640, 800)
    screen.show()
    qtbot.wait(30)
    assert screen._scroll.horizontalScrollBar().maximum() == 0
    assert screen._module_scroll.horizontalScrollBar().maximum() == 0
    assert all(not tile.is_name_elided() for tile in screen._tiles)


def test_new_vector_art_is_distinct_and_follows_both_themes(qapp):
    from spacr.qt import iconset

    sources = sorted((_IMAGES.parent / "icons").glob("organism_*.svg"))
    assert len(sources) == 22
    rendered = set()
    for path in sources:
        dark = iconset.themed_array(str(path), "dark")
        light = iconset.themed_array(str(path), "light")
        assert dark is not None and light is not None, path.name
        assert dark[:, :, 3].max() == 255
        assert not (dark == light).all(), path.name
        rendered.add(dark.tobytes())
    assert len(rendered) == 22


@pytest.mark.parametrize("key", ORGANISMS)
def test_diagram_selectors_highlight_real_groups_and_disclose_shared_shapes(qtbot, key):
    screen = OrganismScreen(key)
    qtbot.addWidget(screen)
    diagram = screen._diagram
    baseline = _diagram_svg(diagram.artwork.source)
    for index in range(diagram.selector.count()):
        item = diagram.selector.item(index)
        location = item.data(Qt.UserRole)
        assert diagram.artwork.renderer.elementExists(location), location
        item.setCheckState(Qt.Checked)
        assert location[:2] + "-" + location[2:] in diagram.caption.text()
        assert ("SL0173" if location == "SL0171" else location) in diagram.artwork.selected
        item.setCheckState(Qt.Unchecked)
    if key == "toxoplasma":
        diagram.selector.item(0).setCheckState(Qt.Checked)
        assert "rhoptries 1 / rhoptries 2" in diagram.caption.text()
    if key == "plasmodium":
        assert "rhoptries 1" not in diagram.labels
    for index in range(diagram.selector.count()):
        diagram.selector.item(index).setCheckState(Qt.Unchecked)
    assert "Check several labels" in diagram.caption.text()
    assert _diagram_svg(diagram.artwork.source) == baseline


def test_text_expands_and_restores_and_divider_drag_reflows_tiles(qtbot, qt_theme_applied):
    screen = OrganismScreen("toxoplasma")
    qtbot.addWidget(screen)
    screen.resize(1280, 800)
    screen.show()
    qtbot.wait(30)
    initial_width = screen._intro.width()
    initial_columns = screen._columns
    qtbot.mouseClick(screen._expand, Qt.LeftButton)
    qtbot.wait(30)
    assert screen._intro.width() > initial_width + 200
    assert screen._columns < initial_columns
    assert screen._module_scroll.horizontalScrollBar().maximum() == 0
    qtbot.mouseClick(screen._expand, Qt.LeftButton)
    qtbot.wait(30)
    assert abs(screen._intro.width() - initial_width) <= 2
    handle = screen._splitter.handle(1)
    centre = handle.rect().center()
    qtbot.mousePress(handle, Qt.LeftButton, pos=centre)
    qtbot.mouseMove(handle, centre + QPoint(160, 0))
    qtbot.mouseRelease(handle, Qt.LeftButton, pos=centre + QPoint(160, 0))
    qtbot.wait(30)
    assert screen._intro.width() > initial_width + 100
    assert all(tile.height() == scaled_px(TILE_H) for tile in screen._tiles)


def test_prose_mentions_every_module_and_active_links_navigate(qtbot):
    for data in ORGANISMS.values():
        prose = " ".join(section[1] for section in data["sections"])
        assert len(prose.split()) >= 220
        assert all(title in prose for _, title, _, _ in data["modules"])
    screen = OrganismScreen("toxoplasma")
    qtbot.addWidget(screen)
    requested = []
    screen.module_requested.connect(requested.append)
    links = screen.findChildren(QLabel, "OrganismModuleLink")
    assert len(links) == 4
    for link in links:
        assert not link.openExternalLinks()
    for key in ASSAYS:
        screen._open_module_link(key)
    screen._open_module_link("https://example.org")
    screen._open_module_link("egress")
    assert requested == list(ASSAYS)


def test_help_search_retains_direct_assay_destinations():
    from spacr.qt.help_index import module_entries
    from spacr.qt.help_search import open_entry

    class Host:
        open_module = app.MainWindow.open_module

        def __init__(self):
            self.selected = []

        def _on_nav_selected(self, key):
            self.selected.append(key)

    host = Host()
    entries = {entry.payload["app"]: entry for entry in module_entries()}
    for key in ASSAYS:
        open_entry(host, entries[key])
    assert host.selected == list(ASSAYS)


def test_main_window_opens_an_assay_from_the_organism_page_and_returns_home(
        qtbot, qt_theme_applied):
    window = app.MainWindow()
    qtbot.addWidget(window)
    window.resize(1280, 800)
    window.show()
    window._on_nav_selected("toxoplasma")
    screen = window._screens["toxoplasma"]
    assert window._stack.currentWidget() is screen
    screen._tiles[3].click()
    assert window._stack.currentWidget() is window._screens["replication"]
    assert window._screens["replication"].app_key == "replication"
    window._on_nav_selected("toxoplasma")
    assert window._stack.currentWidget() is screen
    window._on_nav_selected("__home__")
    assert window._stack.currentWidget() is window._startup
