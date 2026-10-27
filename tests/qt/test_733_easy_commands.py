"""Explicit Easy commands use native navigation, previews and input handlers."""
import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QInputDialog, QMainWindow, QPlainTextEdit,
    QWidget,
)

from spacr.qt import app
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


class _Commands(QMainWindow):
    _open_easy_command = app.MainWindow._open_easy_command
    _open_easy_input = app.MainWindow._open_easy_input
    _easy_input_paths = app.MainWindow._easy_input_paths
    _preview_easy_settings = app.MainWindow._preview_easy_settings
    _set_easy_mode_from_help = app.MainWindow._set_easy_mode_from_help

    def __init__(self):
        super().__init__()
        self.opened = []
        self.navigated = []
        self.refreshed = []
        self.urls = []
        self.inspections = []
        self._startup = SimpleNamespace(_inspect=lambda: self.inspections.append(True))
        self.screen = QWidget(self)
        self._easy_help_menu = self.menuBar().addMenu("Help")
        self._act_preferences = QAction("Preferences", self)
        self._act_easy_mode = QAction("Easy mode", self)
        self._act_easy_mode.setCheckable(True)
        self._act_easy_mode.triggered.connect(self._set_easy_mode_from_help)
        self._easy_help_menu.addAction(self._act_easy_mode)

    def _open_easy_named_module(self, key):
        self.opened.append(key)
        return self.screen

    def _on_nav_selected(self, key):
        self.navigated.append(key)

    def refresh_after_preferences(self, changed):
        self.refreshed.append(changed)

    def _open_url(self, value):
        self.urls.append(value)


@pytest.fixture
def commands(qtbot, private_ai_profile):
    window = _Commands()
    qtbot.addWidget(window)
    return window


def test_typing_never_navigates_but_enter_uses_actual_command(qtbot, commands):
    from spacr.qt.widgets.home import _EasyHomePage

    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.command_requested.connect(commands._open_easy_command)
    page.show()
    qtbot.keyClicks(page.path_edit, "mAkE mAsKs")
    page._debounce.stop()
    assert commands.opened == []
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    assert commands.opened == ["make_masks"]
    assert not page._debounce.isActive()


@pytest.mark.parametrize("text,key", [
    ("GENERATE MASKS", "mask"), ("quality CONTROL", "qc_dashboard"),
    ("TiMe", "timelapse"), ("3D", "3d"),
    ("I work on T. gondii", "toxoplasma"),
    ("I WORK ON T.g.", "toxoplasma"),
    ("TOXOPLASMA", "toxoplasma"),
    ("I work on P. falciparum", "plasmodium"),
    ("I study C. albicans", "candida"),
])
def test_commands_resolve_case_and_organism_questions(commands, text, key):
    commands._open_easy_command(text)
    assert commands.opened == [key]


def test_real_nested_help_action_and_url_aliases(commands):
    invoked = []
    nested = commands._easy_help_menu.addMenu("Learning")
    action = nested.addAction("&Keyboard shortcuts\tCtrl+?")
    action.triggered.connect(lambda: invoked.append(True))
    commands._open_easy_command("KEYBOARD SHORTCUTS")
    assert invoked == [True]
    commands._open_easy_command("RaDmE")
    assert commands.urls == ["https://github.com/EinarOlafsson/spacr#readme"]
    assert commands.opened == []


@pytest.mark.parametrize("query", ["mask:", "MASK:", "Mask:",
                                    "segmentation", "SEGMENTATION", "sEgMeNtAtIoN",
                                    "object detection", "OBJECT DETECTION", "ObJeCt DeTeCtIoN"])
def test_partial_name_lists_actual_modules_without_navigation(qtbot, commands, monkeypatch, query):
    from spacr.qt.widgets.home import _EasyHomePage

    monkeypatch.setattr(app, "app_is_visible", lambda key: True)
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.command_requested.connect(commands._open_easy_command)
    page.path_edit.setText(query)
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    keys = {card._key for card in page.field.cards}
    assert {"mask", "make_masks"} <= keys
    assert commands.opened == []
    assert not page._runner.is_busy()
    assert {row[0] for row in app._easy_module_search(query)} >= {"mask", "make_masks"}
    assert app._easy_module_search("/actual/MASK:/image.tif") == []
    assert app._easy_module_search("/actual/segmentation/image.tif") == []
    assert app._easy_module_search("/actual/object detection/image.tif") == []
    if query.casefold() in ('segmentation', 'object detection'):
        assert app._easy_module_matches(query) == ('mask', 'make_masks')


@pytest.mark.parametrize('query', ['segmentation', 'OBJECT DETECTION'])
def test_segmentation_aliases_preserve_existing_visibility_gates(qtbot, commands, monkeypatch, query):
    from spacr.qt.widgets.home import _EasyHomePage

    monkeypatch.setattr(app, 'app_is_visible', lambda key: key == 'make_masks')
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.command_requested.connect(commands._open_easy_command)
    page.path_edit.setText(query)
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    assert [card._key for card in page.field.cards] == ['make_masks']
    assert commands.opened == []
    assert not page._runner.is_busy()


def test_named_fold_uses_actual_list_opener_and_returns_child(commands, monkeypatch):
    from spacr.qt.screens.map_barcodes import FoldOpener

    child = QWidget(commands)
    host = QWidget(commands)
    opener = FoldOpener(host, "graph_builder", lambda parent: child)
    opened = []
    monkeypatch.setattr(opener, "open", lambda: opened.append(True) or child)
    host._fold_openers = [opener]
    monkeypatch.setattr(app, "folded_children", lambda: {"mask": ["graph_builder"]})
    monkeypatch.setattr(app, "app_is_visible", lambda key: True)
    commands._screens = {"mask": host}
    commands.open_module = commands.opened.append
    result = app.MainWindow._open_easy_named_module(commands, "graph_builder")
    assert commands.opened == ["mask"]
    assert opened == [True]
    assert result is child


@pytest.mark.parametrize("key,dimension", [("timelapse", "t"), ("3d", "z")])
def test_time_and_3d_commands_reveal_existing_dimension_controls(commands, monkeypatch, key, dimension):
    calls = []
    commands.screen.set_dimension = lambda axis, enabled: calls.append((axis, enabled))
    commands._screens = {"mask": commands.screen}
    commands.open_module = commands.opened.append
    monkeypatch.setattr(app, "app_is_visible", lambda key: True)
    assert app.MainWindow._open_easy_named_module(commands, key) is commands.screen
    assert commands.opened == ["mask"]
    assert calls == [(dimension, True)]


def test_help_toggle_persists_to_startup_without_preferences_checkbox(commands, qtbot):
    from spacr.qt import preferences
    from spacr.qt.widgets.setup_slides import SetupSlides
    from spacr.qt.widgets.toggle import Toggle

    preferences.set_easy_mode(False)
    commands._act_easy_mode.trigger()
    assert preferences.get_easy_mode()
    assert commands.navigated == ["__home__"]
    assert commands.refreshed == [{"easy_mode"}]
    startup = SetupSlides()
    qtbot.addWidget(startup)
    assert startup._editors["easy_mode"].isChecked()
    dialog = preferences.PreferencesDialog()
    qtbot.addWidget(dialog)
    assert dialog.findChild(Toggle, "EasyMode") is None
    assert dialog.findChild(Toggle, "EasyModeDock") is None
    commands._act_easy_mode.trigger()
    assert not preferences.get_easy_mode()
    reopened = SetupSlides()
    qtbot.addWidget(reopened)
    assert not reopened._editors["easy_mode"].isChecked()


@pytest.mark.parametrize("accept", [False, True])
def test_saved_settings_actual_preview_controls_apply(commands, qtbot, monkeypatch, accept):
    settings = {"src": "/original/plate", "diameter": 31, "timelapse": False}
    applied, shown = [], []
    commands.screen.apply_settings_dict = applied.append
    context = {"saved_settings": {"module": "mask", "path": "/saved.json", "settings": settings}}

    def respond(dialog):
        dialog.show()
        preview = dialog.findChild(QPlainTextEdit, "EasySettingsPreview")
        shown.append(json.loads(preview.toPlainText()))
        assert preview.isReadOnly()
        buttons = dialog.findChild(QDialogButtonBox)
        buttons.button(QDialogButtonBox.Open if accept else QDialogButtonBox.Cancel).click()
        return dialog.result()

    monkeypatch.setattr(QDialog, "exec", respond)
    assert commands._open_easy_input("mask", "/saved.json", context)
    assert shown == [settings]
    assert applied == ([settings] if accept else [])
    assert commands.opened == (["mask"] if accept else [])
    if accept:
        assert applied[0] is not settings


def test_native_database_and_csv_handlers_reach_real_readers(commands, qtbot, tmp_path):
    from spacr.qt.screens.db_browser import DbBrowserScreen
    from spacr.qt.screens.gate_editor import GateEditorScreen

    database = tmp_path / "measurements.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE cells (area REAL, intensity REAL)")
        connection.execute("INSERT INTO cells VALUES (12, 34)")
    browser = DbBrowserScreen(threaded=False)
    qtbot.addWidget(browser)
    commands.screen = browser
    assert commands._open_easy_input("db_browser", str(database), {"analysis_inputs": {"db_browser": {}}})
    assert browser.tables() == ["cells"]
    assert browser._db.path == str(database)
    csv = tmp_path / "cells.csv"
    csv.write_text("area,intensity\n12,34\n56,78\n")
    gate = GateEditorScreen(threaded=False)
    qtbot.addWidget(gate)
    commands.screen = gate
    assert commands._open_easy_input("gate_editor", str(csv), {})
    assert list(gate._frame["area"]) == [12, 56]
    assert list(gate._frame["intensity"]) == [34, 78]
    assert commands.opened == ["db_browser", "gate_editor"]


@pytest.mark.parametrize("key,multiple", [("gate_editor", False), ("project_browser", True)])
@pytest.mark.parametrize("cancel", [False, True])
def test_multi_plate_chooser_respects_native_handler_capability(commands, monkeypatch, key, multiple, cancel):
    from spacr.qt.dnd_handlers import get_handler

    context = {"analysis_inputs": {key: {}}, "plate_inputs": [
        {"project": "/plate/a", "database": "/plate/a/measurements.db"},
        {"project": "/plate/b", "database": "/plate/b/measurements.db"},
    ]}
    first, second = "/plate/a/measurements.db", "/plate/b/measurements.db"
    observed = []

    def choose(parent, title, label, choices, *args):
        observed.extend(choices)
        return (choices[0] if multiple else second), not cancel

    monkeypatch.setattr(QInputDialog, "getItem", choose)
    handler = get_handler(key)
    assert handler.accepts_multiple() is multiple
    result = commands._easy_input_paths(key, first, context, handler)
    assert result == ([] if cancel else [first, second] if multiple else [second])
    assert (len(observed) == 3) is multiple


def test_cancel_plate_selection_never_opens_destination(commands, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args: ("", False))
    context = {"analysis_inputs": {"gate_editor": {}}, "plate_inputs": [
        {"database": "/a.db"}, {"database": "/b.db"}]}
    assert commands._open_easy_input("gate_editor", "/a.db", context)
    assert commands.opened == []


def test_unstructured_pair_handoff_keeps_runtime_confirmation_in_make_masks(commands):
    calls = []
    commands.screen._open_paired_layout = calls.append
    pairs = {"/loose/image.tif": "/loose/image_mask.tif"}
    context = {"paired_layout": {"kind": "unstructured", "pairs": pairs}}
    assert commands._open_easy_input("make_masks", "/loose", context)
    assert calls == [pairs]
    assert commands.opened == ["make_masks"]


@pytest.mark.parametrize("query", ["mask:", "MASK:", "Mask:"])
def test_partial_suggestions_interrupt_prior_busy_inspection(qtbot, commands, monkeypatch, tmp_path, query):
    from spacr.qt.widgets import home

    started, release = threading.Event(), threading.Event()

    def slow_inspect(value):
        started.set()
        release.wait(5)
        return dict(path=value, actions=[], message="Old inspection", context={})

    monkeypatch.setattr(home, "_easy_inspect_path", slow_inspect)
    monkeypatch.setattr(app, "app_is_visible", lambda key: True)
    page = home._EasyHomePage()
    qtbot.addWidget(page)
    page.command_requested.connect(commands._open_easy_command)
    try:
        page.path_edit.setText(str(tmp_path))
        page._inspect()
        qtbot.waitUntil(started.is_set)
        assert page._runner.is_busy()
        page.path_edit.setText(query)
        qtbot.keyClick(page.path_edit, Qt.Key_Return)
        assert {"mask", "make_masks"} <= {card._key for card in page.field.cards}
        assert commands.opened == []
        assert not page._runner.is_busy()
        release.set()
        qtbot.waitUntil(lambda: page._runner.active_jobs() == 0)
        assert {"mask", "make_masks"} <= {card._key for card in page.field.cards}
        assert page.status.text() != "Old inspection"
    finally:
        release.set()
        page.close()


@pytest.mark.parametrize("suffix", [None, ".gif", ".webp"])
def test_foreign_image_folder_uses_native_images_input(commands, qtbot, tmp_path, suffix):
    from spacr.qt.screens.foreign import ForeignScreen
    from PIL import Image, features

    images = tmp_path / "raw_images"
    images.mkdir()
    source = images
    if suffix:
        if suffix == ".webp" and not features.check("webp"):
            pytest.skip("This Pillow has no WebP reader")
        source = images / ("field" + suffix)
        Image.new("RGB", (7, 5), "red").save(source)
        from spacr.qt.dnd_handlers import get_handler
        assert get_handler("convert").can_accept(source)
    screen = ForeignScreen(threaded=False)
    qtbot.addWidget(screen)
    commands.screen = screen
    original_measurements = screen._table_edit.text()
    assert commands._open_easy_input("foreign", str(source), {})
    assert screen.images_path() == str(images)
    assert screen._table_edit.text() == original_measurements
    assert commands.opened == ["foreign"]


@pytest.mark.parametrize("key", ["3d", "timelapse"])
def test_dimension_route_uses_mask_drop_without_starting_processing(commands, monkeypatch, tmp_path, key):
    from spacr.qt.dnd_handlers import MaskDropHandler

    calls = []
    monkeypatch.setattr(MaskDropHandler, "apply", lambda self, path, screen: calls.append((path, screen)))
    context = {"dimension_routes": {key: {"source": str(tmp_path)}}}
    assert commands._open_easy_input(key, "/selected/image.tif", context)
    assert commands.opened == [key]
    assert calls == [(tmp_path, commands.screen)]


def test_partial_help_search_is_case_insensitive_and_uses_help_destinations(commands):
    menu = commands._easy_help_menu.addMenu('Learning')
    menu.addAction('&Keyboard shortcuts\tCtrl+?')
    commands.menuBar().addMenu('File').addAction('Quit')
    search = app.MainWindow._easy_navigation_suggestions
    assert search(commands, 'KEYBOARD:') == search(commands, 'keyboard:')
    assert any(row[0] == '__easy_command__:Keyboard shortcuts'
               for row in search(commands, 'KeYbOaRd'))
    assert not search(commands, 'QUIT')


@pytest.mark.parametrize('query', [
    'measurement', 'MEASUREMENT', 'MeAsUrEmEnT',
    'feature', 'FEATURE', 'FeAtUrE',
    'morphological profiling', 'MORPHOLOGICAL PROFILING', 'MoRpHoLoGiCaL PrOfIlInG',
])
def test_measure_aliases_offer_the_native_module_and_preserve_explicit_paths(commands, query):
    rows = app._easy_module_search(query)
    assert sum(row[0] == 'measure' for row in rows) == 1
    assert app._easy_module_matches(query) == ('measure',)
    commands._open_easy_command(query)
    assert commands.opened == ['measure']
    assert app._easy_module_search('/actual/' + query + '/image.tif') == []


@pytest.mark.parametrize('label', ['Database Browser', '&Database Browser…', 'DATABASE BROWSER\tCtrl+B'])
def test_database_help_action_reuses_native_result_instead_of_a_generic_copy(commands, label):
    native = app._easy_module_search('database')
    commands._easy_help_menu.addAction(label)
    commands._easy_help_menu.addAction('Database reference')
    rows = app.MainWindow._easy_navigation_suggestions(commands, 'DaTaBaSe')
    assert [row for row in rows if not row[0].startswith('__easy_command__:')] == native
    assert sum(row[0] == 'db_browser' for row in rows) == 1
    assert not any(row[0].startswith('__easy_command__:')
                   and app._easy_command_text(row[1]) == 'database browser' for row in rows)
    assert any(row[0] == '__easy_command__:Database reference' for row in rows)


@pytest.mark.parametrize('query', ['plan', 'PLAN', 'Plan:'])
def test_planning_search_offers_both_actual_planning_tools(query):
    assert {'experiment_design', 'power'} <= {row[0] for row in app._easy_module_search(query)}
    assert app._easy_module_matches('EXPERIMENTAL DESIGN') == ('experiment_design',)
    assert app._easy_module_matches('PoWeR / DeSiGn') == ('power',)
    assert app._easy_module_matches('/actual/Power / Design') == ()


@pytest.mark.parametrize('accept', [False, True])
def test_inferred_submodule_settings_preview_then_apply(commands, qtbot, monkeypatch, accept):
    settings = {'spatial_source': '/slide', 'spatial_platform': 'xenium'}
    applied, titles = [], []
    commands.screen.apply_settings_dict = applied.append

    def respond(dialog):
        titles.append(dialog.windowTitle())
        assert json.loads(dialog.findChild(QPlainTextEdit, 'EasySettingsPreview').toPlainText()) == settings
        return QDialog.Accepted if accept else QDialog.Rejected

    monkeypatch.setattr(QDialog, 'exec', respond)
    context = {'submodule_settings': {'spatial_masks': {'settings': settings, 'source': '/slide'}}}
    assert commands._open_easy_input('spatial_masks', '/slide', context)
    assert titles == [app.tr('Preview settings')]
    assert commands.opened == (['spatial_masks'] if accept else [])
    assert applied == ([settings] if accept else [])
