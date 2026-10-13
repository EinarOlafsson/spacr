"""F534's two batch submodules are alpha page folds, reached by a switch.

Spatial Mask Generation opens from a "Spatial" switch beside OPS in Mask
Generation's actions row; Spatial Map Barcodes from the same switch in Map
Barcodes' actions row. Both reuse OPS's page-fold machinery
(:class:`spacr.qt.screens.map_barcodes.PageFold`), both switches sit behind
Preferences -> Show alpha features, and both run entries resolve to
:mod:`spacr.ops_engine`.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from spacr.qt.screens import map_barcodes as mb                # noqa: E402
from spacr.qt.screens import mask as mask_mod                  # noqa: E402

SWITCHES = {
    "mask": ("MaskSpatialSwitch", "spatial_masks"),
    "map_barcodes": ("MapBarcodesSpatialMapSwitch", "spatial_map"),
}

PAGE_CONTENT = {
    "spatial_masks": (
        ("Spatial input", "Spatial region & tiling", "Spatial segmentation"),
        ("spatial_source", "spatial_tile_overlap", "spatial_segmenter")),
    "spatial_map": (
        ("Spatial input", "Spatial mapping", "Spatial infection"),
        ("spatial_source", "spatial_masks_dir", "spatial_pathogen_genes")),
}


@pytest.fixture
def gate(tmp_path, monkeypatch):
    """Preferences read from a private file, so the gate starts shut."""
    from PySide6.QtCore import QSettings

    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def _screen(qtbot, key):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key=key)
    qtbot.addWidget(screen)
    return screen


@pytest.mark.parametrize("host", sorted(SWITCHES))
def test_the_switch_is_on_its_host_and_follows_the_alpha_gate(qtbot, gate,
                                                               host):
    from PySide6.QtCore import QObject

    from spacr.qt.widgets import AiToggleLabel
    from spacr.settings import ALPHA_FEATURES

    name, key = SWITCHES[host]
    assert name in ALPHA_FEATURES[534]["widgets"]
    screen = _screen(qtbot, host)
    switch = screen.findChild(QObject, name)
    assert isinstance(switch, AiToggleLabel)
    assert switch is screen._spatial_switch
    assert switch._full_text == mb.SPATIAL_TOGGLE_TEXT
    assert mb.page_fold(screen, key).switch is switch
    assert screen._action_hint(switch), "the switch explains nothing"
    assert switch.isHidden(), "an alpha switch shows with the gate shut"

    gate._set_show_alpha_features(True)
    gate._apply_alpha_widgets(screen)
    assert not switch.isHidden()
    gate._set_show_alpha_features(False)
    gate._apply_alpha_widgets(screen)
    assert switch.isHidden()


def test_the_mask_switch_sits_right_after_ops(qtbot, gate):
    from spacr.qt.widgets import AiToggleLabel

    screen = _screen(qtbot, "mask")
    row = [label._full_text for label in screen.findChildren(AiToggleLabel)]
    assert row.index(mb.SPATIAL_TOGGLE_TEXT) == (
        row.index(mask_mod.OPS_TOGGLE_TEXT) + 1), row


def test_only_the_two_hosts_grow_one(qtbot, gate):
    for key in ("measure", "ops", "spatial_masks"):
        assert _screen(qtbot, key)._spatial_switch is None


@pytest.mark.parametrize("host", sorted(SWITCHES))
def test_the_switch_opens_and_closes_the_settings_page(qtbot, gate, host):
    _name, key = SWITCHES[host]
    categories, keys = PAGE_CONTENT[key]
    screen = _screen(qtbot, host)
    fold = mb.page_fold(screen, key)
    assert fold.page is None, "the page is built before anyone asked"

    screen._spatial_switch.setChecked(True)
    page = fold.page
    assert page is not None and page.app_key == key
    pages = screen._fold_pages
    assert pages.indexOf(page) > 0
    assert pages.tabText(pages.indexOf(page)) == mb.fold_description(key)[0]
    assert page._header.title_label.text() == mb.fold_description(key)[0]
    assert page._header.description_label is not None

    shown = {section.property("settingsCategorySource")
             for section in page._settings_sections}
    assert set(categories) <= shown, shown
    for setting in keys:
        assert page.setting_row_is_visible(setting), setting
        control = page._settings_model._widgets.built(setting)
        help_text = (control.property("apiTooltipDescriptionSource")
                     or control.toolTip())
        assert str(help_text).strip(), f"{setting} has no help"

    screen._spatial_switch.setChecked(False)
    assert fold.page is None
    assert pages.indexOf(page) == -1

    screen._spatial_switch.setChecked(True)
    assert fold.page is page, "the page was rebuilt instead of kept"


def test_closing_the_page_by_its_cross_puts_the_switch_back(qtbot, gate):
    screen = _screen(qtbot, "mask")
    switch = screen._spatial_switch
    switch.setChecked(True)
    fold = mask_mod.spatial_masks_page(screen)
    pages = screen._fold_pages
    pages.tabCloseRequested.emit(pages.indexOf(fold.page))
    assert not switch.isChecked()
    assert fold.page is None


def test_ops_and_spatial_pages_share_the_strip_independently(qtbot, gate):
    screen = _screen(qtbot, "mask")
    screen._ops_switch.setChecked(True)
    screen._spatial_switch.setChecked(True)
    ops = mask_mod.ops_page(screen)
    spatial = mask_mod.spatial_masks_page(screen)
    assert mb.page_fold(screen, "ops") is ops
    assert ops.page.app_key == "ops" and spatial.page.app_key == "spatial_masks"
    screen._spatial_switch.setChecked(False)
    assert ops.page is not None and screen._ops_switch.isChecked()


def test_a_page_that_cannot_be_built_leaves_the_switch_off(qtbot, gate,
                                                           monkeypatch):
    screen = _screen(qtbot, "map_barcodes")
    fold = mb.page_fold(screen, "spatial_map")
    monkeypatch.setattr(fold.opener, "open", lambda *a: None)
    screen._spatial_switch.setChecked(True)
    assert not screen._spatial_switch.isChecked()
    assert fold.page is None


def test_show_page_presses_the_switch_or_reopens_behind_it(qtbot, gate):
    screen = _screen(qtbot, "map_barcodes")
    fold = mb.page_fold(screen, "spatial_map")
    fold.show_page()
    assert screen._spatial_switch.isChecked() and fold.page is not None
    page = fold.page
    mb.hide_as_page(page, screen)
    fold.show_page()
    assert screen._fold_pages.indexOf(page) > 0


def test_the_installers_refuse_other_screens_and_install_once(qtbot):
    from PySide6.QtWidgets import QCheckBox, QWidget

    other = QWidget()
    qtbot.addWidget(other)
    switch = QCheckBox()
    qtbot.addWidget(switch)
    assert mask_mod.install_spatial_masks_switch(other, switch) is None
    assert mb.install_spatial_map_switch(other, switch) is None
    assert mb.install_page_fold(other, None, "mask", "spatial_masks") is None
    assert mb.page_fold(None, "spatial_masks") is None

    host = QWidget()
    qtbot.addWidget(host)
    host.app_key = "mask"
    first = mask_mod.install_spatial_masks_switch(host, switch)
    assert first is not None and switch.objectName() == "MaskSpatialSwitch"
    assert mask_mod.install_spatial_masks_switch(host, switch) is first
    assert mask_mod.spatial_masks_page(host) is first
    assert mask_mod.ops_page(host) is None


def test_the_page_folds_are_declared_where_navigation_reads_them():
    assert "spatial_masks" in mask_mod.PAGE_FOLDS
    assert "spatial_map" in mb.PAGE_FOLDS
    assert mask_mod.PAGE_FOLDS[0] == "ops"
    for key in ("spatial_masks", "spatial_map"):
        assert key not in mask_mod.FOLDED_APPS + mb.FOLDED_APPS, (
            "an alpha page fold in FOLDED_APPS is listed in the menu with "
            "the gate shut")
    assert mb.page_fold_host("ops") == "mask"
    assert mb.page_fold_host("spatial_masks") == "mask"
    assert mb.page_fold_host("spatial_map") == "map_barcodes"
    assert mb.page_fold_host("timelapse") is None

    from spacr.qt.chaining import screen_for_module

    assert screen_for_module("spatial_masks") == "mask"
    assert screen_for_module("spatial_map") == "map_barcodes"
    for key in ("spatial_masks", "spatial_map"):
        name, description, stage = mb.fold_description(key)
        assert name.startswith("Spatial") and description and stage == "alpha"
        assert mb.folded_module_title(key) == name


def test_the_run_entries_and_registries_name_the_ops_engine():
    from spacr import cli, ops_engine
    from spacr.qt import iconset
    from spacr.qt.bridge import resolve_pipeline_entry
    from spacr.qt.screens.settings_model import _APP_API_MODULE
    from spacr.validate import ALT_SRC_KEYS, APP_FUNCTIONS

    for key, func in (("spatial_masks", ops_engine.run_spatial_masks),
                      ("spatial_map", ops_engine.run_spatial_map)):
        entry = resolve_pipeline_entry(key)
        assert entry is not None
        assert getattr(entry, "__wrapped__", None) is func or (
            entry.__name__ == func.__name__)
        assert APP_FUNCTIONS[key] == f"spacr.ops_engine.{func.__name__}"
        assert cli.MODULES[key].func_name == func.__name__
        assert cli.MODULES[key].note.startswith("ALPHA")
        assert _APP_API_MODULE[key] == "ops_engine"
        assert ALT_SRC_KEYS[key] == "spatial_source"
        assert key in iconset._NAME_TO_GLYPH


def test_open_module_lands_on_the_host_with_the_page_open(qtbot, gate):
    from spacr.qt.app import MainWindow, _opened_module_screen

    window = MainWindow()
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    for key, host in (("spatial_masks", "mask"),
                      ("spatial_map", "map_barcodes")):
        assert _opened_module_screen(window, key, host) is None
        assert window.open_module(key) == host
        screen = window._screens[host]
        fold = mb.page_fold(screen, key)
        assert screen._spatial_switch.isChecked()
        assert fold.page is not None and fold.page.app_key == key
        assert _opened_module_screen(window, key, host) is fold.page
        screen._fold_pages.tabCloseRequested.emit(
            screen._fold_pages.indexOf(fold.page))
        assert not screen._spatial_switch.isChecked()
        window.open_module(key)
        assert screen._spatial_switch.isChecked()
        assert _opened_module_screen(window, host, host) is screen


def test_navigation_edges_without_an_installed_switch(qtbot):
    from types import SimpleNamespace

    from PySide6.QtWidgets import QWidget

    from spacr.qt.app import MainWindow, _opened_module_screen

    host = QWidget()
    qtbot.addWidget(host)
    window = SimpleNamespace(_screens={"mask": host})
    assert _opened_module_screen(window, "timelapse", "mask") is host
    assert _opened_module_screen(window, "spatial_masks", "mask") is None
    MainWindow._switch_a_fold_on(window, "mask", "spatial_masks")
    assert mb.page_fold(host, "spatial_masks") is None


def test_a_switch_that_cannot_be_wired_leaves_the_screen_standing(
        qtbot, gate, monkeypatch):
    def refuse(screen, switch):
        raise RuntimeError("the spatial page is not available")

    monkeypatch.setattr(mask_mod, "install_spatial_masks_switch", refuse)
    screen = _screen(qtbot, "mask")
    assert screen._spatial_switch is not None
    assert mask_mod.spatial_masks_page(screen) is None


def test_the_page_intros_come_from_the_registered_descriptions(monkeypatch):
    import spacr.settings
    from spacr.qt.screens import app_screen

    assert app_screen.APP_INTROS["spatial_masks"] == (
        spacr.settings.descriptions["spatial_masks"])
    assert app_screen.APP_INTROS["spatial_map"] == (
        spacr.settings.descriptions["spatial_map"])
    intros = dict(app_screen.APP_INTROS)
    monkeypatch.setattr(app_screen, "APP_INTROS", {})
    monkeypatch.setattr(spacr.settings, "descriptions", {})
    app_screen._absorb_page_fold_intros()
    assert app_screen.APP_INTROS == {}
    monkeypatch.setattr(spacr.settings, "descriptions", {
        "spatial_masks": intros["spatial_masks"]})
    app_screen._absorb_page_fold_intros()
    assert set(app_screen.APP_INTROS) == {"spatial_masks"}


def test_the_alpha_gate_hides_the_modules_from_navigation(gate):
    from spacr.qt.app import app_is_visible

    assert not app_is_visible("spatial_masks")
    assert not app_is_visible("spatial_map")
    gate._set_show_alpha_features(True)
    assert app_is_visible("spatial_masks") and app_is_visible("spatial_map")
