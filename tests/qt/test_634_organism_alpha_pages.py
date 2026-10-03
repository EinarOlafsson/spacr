"""Item 634: Trypanosoma, Leishmania, Giardia, virus and mammalian pages.

Five organism pages built like the Toxoplasma, Plasmodium and Candida ones,
registered, with Plasmodium and Candida, under ``ALPHA_SPECIES[634]``: absent
from Home, the dock and the spaCR menu with Preferences > Show alpha species
off, and opening with it on, whatever Show alpha features says. Their
compartment lists are UniProt subcellular locations (release 2026_03)
annotated for each taxon and drawn in the bundled artwork.
"""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QLabel

from spacr.qt import app
from spacr.qt.organisms import ORGANISMS, workflow
from spacr.qt.screens.organism_screen import OrganismScreen, _IMAGES
from spacr.qt.widgets.organism_diagram import _compartment_labels
from spacr.settings import ALPHA_FEATURES, ALPHA_SPECIES, _alpha_names

NEW = ("trypanosoma", "leishmania", "giardia", "virus", "mammalian")
SPECIES = ("plasmodium", "candida") + NEW
REFERENCE = ("toxoplasma", "plasmodium", "candida")
PAGE_NAMES = {
    "plasmodium": "PlasmodiumOrganismPage",
    "candida": "CandidaOrganismPage",
    "trypanosoma": "TrypanosomaOrganismPage",
    "leishmania": "LeishmaniaOrganismPage",
    "giardia": "GiardiaOrganismPage",
    "virus": "VirusOrganismPage",
    "mammalian": "MammalianOrganismPage",
}
_SVG = "{http://www.w3.org/2000/svg}"


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_the_species_pages_are_registered_under_their_own_switch():
    entry = ALPHA_SPECIES[634]
    assert entry["apps"] == SPECIES
    assert set(entry["widgets"]) == set(PAGE_NAMES.values())
    assert 634 not in ALPHA_FEATURES
    assert set(SPECIES) <= _alpha_names("apps")
    assert "toxoplasma" not in _alpha_names("apps")
    registry = {row[0]: row for row in app.APPS}
    for key in SPECIES:
        assert registry[key][3] == app.SECTION_ASSAYS
        assert app.app_stage(key) == app.app_stage("toxoplasma")
        assert isinstance(app.APP_FACTORIES[key], type(app.APP_FACTORIES["toxoplasma"]))


@pytest.mark.parametrize("key", SPECIES)
def test_each_page_follows_show_alpha_species_not_show_alpha_features(prefs, key):
    prefs._set_show_alpha_species(False)
    for features in (False, True):
        prefs._set_show_alpha_features(features)
        assert not app.app_is_visible(key)
        assert key not in {row[0] for row in app.tiled_apps(app.visible_apps())}
        assert app.app_is_visible("toxoplasma")
    prefs._set_show_alpha_species(True)
    for features in (False, True):
        prefs._set_show_alpha_features(features)
        assert app.app_is_visible(key)
        assert key in {row[0] for row in app.section_members(
            app.SECTION_ASSAYS, app.visible_apps())}


def test_show_alpha_features_alone_keeps_its_own_items_independent(prefs):
    prefs._set_show_alpha_species(True)
    prefs._set_show_alpha_features(False)
    assert prefs._is_alpha_visible() is False
    hidden = [name for name in _alpha_names("widgets")
              if name not in ALPHA_SPECIES[634]["widgets"]]
    assert hidden and not any(prefs._is_alpha_visible("widgets", name) for name in hidden)
    assert all(prefs._is_alpha_visible("widgets", name) for name in PAGE_NAMES.values())


def test_the_preferences_dialog_shows_and_saves_the_species_switch(
        qtbot, prefs, monkeypatch):
    from PySide6.QtWidgets import QDialogButtonBox, QWidget

    monkeypatch.setattr(prefs, "apply_preferences_to_app", lambda *args: None)
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    switch = dialog.findChild(QWidget, "ShowAlphaSpecies")
    features = dialog.findChild(QWidget, "ShowAlphaFutureFeatures")
    assert switch is not None and switch.isChecked() is False
    assert switch.parentWidget() is features.parentWidget()
    assert "Show alpha species" in switch.text() and "Giardia" in switch.text()
    switch.setChecked(True)
    dialog.findChild(QDialogButtonBox).accepted.emit()
    assert prefs._get_show_alpha_species() is True
    assert prefs._get_show_alpha_features() is False


def test_the_menu_and_the_page_follow_the_switch(qtbot, qt_theme_applied, prefs):
    from spacr.qt.preferences import _apply_alpha_widgets

    prefs._set_show_alpha_features(True)
    prefs._set_show_alpha_species(False)
    window = app.MainWindow()
    qtbot.addWidget(window)
    window.resize(1280, 800)
    for key in SPECIES:
        assert not window._organism_menus[key].menuAction().isVisible()
    assert window._organism_menus["toxoplasma"].menuAction().isVisible()
    prefs._set_show_alpha_features(False)
    prefs._set_show_alpha_species(True)
    window._refresh_app_action_visibility()
    window.show()
    for key in SPECIES:
        assert window._organism_menus[key].menuAction().isVisible()
        window._on_nav_selected(key)
        page = window._screens[key]
        assert isinstance(page, OrganismScreen) and page.app_key == key
        assert window._stack.currentWidget() is page
        assert page.objectName() == PAGE_NAMES[key]
    window._on_nav_selected("giardia")
    page = window._screens["giardia"]
    prefs._set_show_alpha_species(False)
    assert _apply_alpha_widgets(page) == 1 and page.isHidden()
    prefs._set_show_alpha_species(True)
    assert _apply_alpha_widgets(page) == 1 and not page.isHidden()
    window._on_nav_selected("__home__")
    assert window._stack.currentWidget() is window._startup


@pytest.mark.parametrize("key", NEW)
def test_each_page_has_the_reference_structure(key):
    guide = ORGANISMS[key]
    reference = ORGANISMS["candida"]
    assert set(guide) - {"source_label"} == set(reference)
    assert len(guide["sections"]) == len(reference["sections"]) == 4
    assert len(guide["modules"]) == len(reference["modules"]) == 8
    icons = [icon for _, _, _, icon in guide["modules"]]
    assert len(set(icons)) == len(icons)
    assert all(route is None for route, _, _, _ in guide["modules"])
    assert set(guide["workflows"]) <= set(icons)
    for _heading, _prose, linked in guide["sections"]:
        assert all(workflow(key, icon) for icon in linked)
    for app_key, preset, note in guide["workflows"].values():
        assert app_key in {row[0] for row in app.APPS} | {"motility"}
        assert isinstance(preset, dict) and note.startswith("Opens ")
    records = {row["file"]: row for row in json.loads(
        (_IMAGES / "organism_sources.json").read_text())}
    record = records[guide["diagram"]]
    assert record["retrieved"] == "2026-10-03" and "2026_03" in record["modifications"]
    if key != "trypanosoma":
        assert guide["diagram"] == f"organism_{key}.svg"
        assert record["credit"] == "spaCR contributors"
    assert hashlib.sha256((_IMAGES / guide["diagram"]).read_bytes()).hexdigest() == record["sha256"]


@pytest.mark.parametrize("key", NEW)
def test_compartments_are_uniprot_locations_drawn_in_the_artwork(key):
    labels = _compartment_labels(key)
    assert 10 <= len(labels) <= 14
    root = ET.parse(_IMAGES / ORGANISMS[key]["diagram"]).getroot()
    groups = {node.get("id"): node for node in root.iter() if node.get("id")}
    for name, code in labels.items():
        node = groups[code]
        uniprot = [" ".join(text.itertext()).strip()
                   for text in node.findall(_SVG + "text")
                   if text.get("property") == "name"]
        assert uniprot == [name], (key, code)
        assert any(child.tag.removeprefix(_SVG) in {"path", "ellipse", "circle", "polygon", "polyline", "rect", "line"}
                   for child in node.iter()), (key, code)
    if key == "giardia":
        assert "Mitochondrion" not in labels and "Golgi apparatus" not in labels


@pytest.mark.parametrize("key", NEW)
def test_each_page_draws_its_cell_and_every_compartment_highlights(
        qtbot, qt_theme_applied, key):
    screen = OrganismScreen(key)
    qtbot.addWidget(screen)
    screen.resize(1280, 800)
    screen.show()
    qtbot.wait(30)
    diagram = screen._diagram
    assert diagram.artwork.renderer.isValid()
    assert diagram.artwork.portrait == (key == "leishmania")
    assert diagram.selector.accessibleName() == "UniProt compartment"
    masks = diagram.artwork.masks()
    for index in range(diagram.selector.count()):
        item = diagram.selector.item(index)
        code = item.data(Qt.UserRole)
        assert masks[code][2] > 0, (key, code)
        item.setCheckState(Qt.Checked)
        assert code[:2] + "-" + code[2:] in diagram.caption.text()
        assert code in diagram.artwork.selected
    diagram.clear_button.click()
    assert not diagram.artwork.selected
    links = [label.text() for label in screen.findChildren(QLabel)]
    expected = ORGANISMS[key].get("source_label", "Biology source: CDC")
    assert any(expected in text and ORGANISMS[key]["source"] in text for text in links)
    live = [tile for tile in screen._tiles if tile.property("organismWorkflow")]
    assert len(live) == len(ORGANISMS[key]["workflows"])
    assert all(tile.isEnabled() for tile in live)


def _shapes(key, code):
    root = ET.parse(_IMAGES / ORGANISMS[key]["diagram"]).getroot()
    group = next(node for node in root.iter() if node.get("id") == code)
    return [child for child in group if child.tag.removeprefix(_SVG) != "text"]


def test_the_redrawn_cells_carry_their_defining_anatomy():
    assert len(_shapes("giardia", "SL0191")) == 2
    assert len(_shapes("giardia", "SL0117")) == 8
    assert len(_shapes("giardia", "SL0308")) == 8
    assert len(_shapes("leishmania", "SL0117")) == 2
    assert len(_shapes("leishmania", "SL0150")) == 2
    capsids = [shape for shape in _shapes("virus", "SL0274")
               if shape.tag.removeprefix(_SVG) == "path" and shape.get("fill") != "none"]
    assert all(len(shape.get("d").split("L")) == 6 for shape in capsids)
    assert len(_shapes("mammalian", "SL0066")) == 1
    assert len(_shapes("mammalian", "SL0048")) == 2


def test_each_new_page_has_its_own_home_icon():
    from spacr.qt.iconset import SHARED_ICON_ASSETS

    assert SHARED_ICON_ASSETS["trypanosoma"] == "organism_gliding.svg"
    for key in ("leishmania", "giardia", "virus", "mammalian"):
        assert SHARED_ICON_ASSETS[key] == f"organism_{key}.svg"
        assert (_IMAGES.parent / "icons" / f"organism_{key}.svg").is_file()
