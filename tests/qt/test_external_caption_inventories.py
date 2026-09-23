"""Inventory text passed to the translator through SVG metadata or job callbacks."""
import importlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def builder(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    return importlib.import_module("build_i18n_catalogs")


@pytest.mark.parametrize("app_key", ["toxoplasma", "plasmodium", "candida"])
def test_reachable_svg_descriptions_are_catalogued(builder, qtbot, app_key):
    from spacr.qt.organisms import ORGANISMS
    from spacr.qt.widgets.organism_diagram import OrganismDiagram

    path = ROOT / "spacr/resources/images" / ORGANISMS[app_key]["diagram"]
    diagram = OrganismDiagram(app_key, path)
    qtbot.addWidget(diagram)
    displayed = {diagram.descriptions[code] for code in diagram.labels.values()
                 if code in diagram.descriptions}
    assert len(displayed) >= 10
    assert displayed <= builder._organism_description_sources()


def test_installer_phase_sources_match_the_handoff(builder):
    receipt = json.loads((ROOT / "features/data/474_starplast_integration_2026-09-22.json").read_text())
    sources = builder._starplast_progress_sources()
    assert sources == set(receipt["dynamic_progress_labels"])
    assert len(sources) == 7
    assert sources <= builder._indirect_runtime_ui_sources()


def test_shortcut_inventory_tracks_the_actual_make_masks_rows(builder):
    from spacr.qt.screens.make_masks import SHORTCUT_HINTS
    from spacr.qt.i18n import _ROWS

    expected = {text for row in SHORTCUT_HINTS for text in row}
    assert builder._make_masks_shortcut_sources() == expected
    assert "Ctrl+L+right click" in expected
    assert "Lock / unlock magnifier region and zoom" in expected
    translatable = {text for text in expected if builder._looks_translatable(text)}
    assert translatable <= builder._indirect_runtime_ui_sources()
    assert translatable - _ROWS.keys() <= set(builder.extract_static_ui_sources())


def test_explicit_service_module_does_not_expose_private_helpers(builder):
    from api_visibility import explicit_page_policy, public_symbol

    assert explicit_page_policy("spacr._starplast") is False
    for name in ("spacr._starplast.apps_root", "spacr._starplast._record",
                 "spacr._starplast.Path", "spacr._segmentation_backends"):
        assert explicit_page_policy(name) is None
    assert public_symbol("spacr._starplast.apps_root")
    assert not public_symbol("spacr._starplast._record")
    assert not public_symbol("spacr._segmentation_backends.install_backend")
