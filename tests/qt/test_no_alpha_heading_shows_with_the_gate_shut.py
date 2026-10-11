"""With Show alpha features off, no module shows an alpha settings heading.

The workstation found Mask Generation with its Timelapse fold on showing
"LINEAGE TREES α" and "EVENT DETECTION α" while the gate was shut: the fold
mounts its categories outside the form's own section list, and switching it
on showed every one of them, rows hidden or not.  Each module the app opens
is built here the way the app builds it, with every fold, dimension switch
and waiting heading opened, and every heading still on screen is checked.

Two gates draw the same mark.  Show alpha features hides what
``ALPHA_FEATURES`` registers, and a category named with a trailing α is one
of those.  Feature maturity -> Show alpha hides the categories of a module
whose maturity stage is alpha, and their headings carry the α too.  With the
first gate shut no registered alpha category may show; with both shut no
heading may carry the mark at all.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QLabel, QWidget  # noqa: E402

from spacr.qt import app as app_module  # noqa: E402
from spacr.qt.widgets.section import Section  # noqa: E402
from spacr.settings import ALPHA_FEATURES, _alpha_names  # noqa: E402

pytestmark = pytest.mark.qt

ALPHA_MARK = "α"


def _module_keys():
    """Every module key the app registers, plus the folded Timelapse form."""
    keys = [row[0] for row in app_module.APPS] or [
        row[0] for row in app_module._BUILTIN_APPS]
    return sorted(set(keys) | {"timelapse"})


@pytest.fixture
def gate_shut(monkeypatch):
    """Show alpha features and Show alpha species both off."""
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: False)
    monkeypatch.setattr(preferences, "_get_show_alpha_species", lambda: False)


def _build(key):
    """Build ``key``'s screen as the main window does, folds included."""
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.screens.map_barcodes import install_folds_on

    factory = app_module.registered_factory(key)
    screen = None
    if factory is not None:
        try:
            screen = app_module._call_screen_factory(factory, key, None)
        except Exception:                                    # noqa: BLE001
            screen = None
    if screen is None:
        screen = AppScreen(key)
    install_folds_on(screen)
    return screen


def _open_everything(screen):
    """Every fold on, every dimension on, every waiting heading built."""
    opener = getattr(screen, "_open_every_waiting_heading", None)
    if callable(opener):
        opener()
    folds = getattr(screen, "_category_folds", None)
    for key in getattr(folds, "order", ()) or ():
        folds.set_active(key, True)
    for dimension in ("t", "z"):
        setter = getattr(screen, "set_dimension", None)
        if callable(setter):
            setter(dimension, True)
    if callable(opener):
        opener()


def _section_keys(screen, section):
    """The settings keys whose rows sit directly on ``section``."""
    widgets = getattr(getattr(screen, "_settings_model", None),
                      "_widgets", None) or {}
    built = getattr(widgets, "built_items", None)
    pairs = built() if callable(built) else widgets.items()
    keys = set()
    for key, widget in pairs:
        try:
            if isinstance(widget, QWidget) and section.isAncestorOf(widget):
                keys.add(key)
        except RuntimeError:
            continue
    for caption in section.findChildren(QLabel):
        key = caption.property("settingKey")
        if key:
            keys.add(str(key))
    return keys


def _visible_sections(key):
    """Build ``key`` with everything opened; its sections still on screen."""
    if key in app_module._alpha_hidden_apps():
        pytest.skip(f"{key} is itself behind the alpha gate")
    screen = _build(key)
    _open_everything(screen)
    return screen, [section for section in screen.findChildren(Section)
                    if section.isVisibleTo(screen)]


@pytest.mark.parametrize("key", _module_keys())
def test_no_registered_alpha_category_shows_with_the_gate_shut(
        qtbot, gate_shut, key):
    """No heading on screen is an α category or holds only alpha settings."""
    screen, sections = _visible_sections(key)
    qtbot.addWidget(screen)
    alpha = set(_alpha_names("settings", key))
    leaks = []
    for section in sections:
        text = section.header().text()
        source = str(section.property("settingsCategorySource")
                     or getattr(section, "_title_source", "") or "")
        if ALPHA_MARK in source:
            leaks.append(text)
            continue
        keys = _section_keys(screen, section)
        if keys and keys <= alpha:
            leaks.append(f"{text} (only alpha settings: {sorted(keys)})")
    assert not leaks, f"{key} shows alpha headings with the gate shut: {leaks}"


@pytest.mark.parametrize("key", _module_keys())
def test_no_heading_carries_the_mark_with_both_alpha_gates_shut(
        qtbot, monkeypatch, gate_shut, key):
    """With Show alpha off too, no visible heading contains α at all."""
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "get_show_alpha", lambda: False)
    screen, sections = _visible_sections(key)
    qtbot.addWidget(screen)
    leaks = [section.header().text() for section in sections
             if ALPHA_MARK in section.header().text()]
    assert not leaks, f"{key} shows α headings with both gates shut: {leaks}"


def test_the_timelapse_fold_holds_back_its_alpha_categories(qtbot, gate_shut):
    """The case the workstation saw: Mask with Timelapse switched on."""
    screen = _build("mask")
    qtbot.addWidget(screen)
    folds = screen._category_folds
    folds.set_active("timelapse", True)
    shown = [section.header().text() for fold in folds.folds.values()
             for section in fold.sections if section.isVisibleTo(screen)]
    assert shown, "the Timelapse fold should still show its ordinary categories"
    assert not [text for text in shown if ALPHA_MARK in text]


def test_the_timelapse_fold_shows_alpha_categories_once_the_gate_opens(
        qtbot, monkeypatch, gate_shut):
    """Turning the preference on brings the categories back live."""
    from spacr.qt import preferences

    screen = _build("mask")
    qtbot.addWidget(screen)
    folds = screen._category_folds
    folds.set_active("timelapse", True)
    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: True)
    screen._refresh_alpha_visibility()
    shown = [section.header().text() for fold in folds.folds.values()
             for section in fold.sections if section.isVisibleTo(screen)]
    assert any(ALPHA_MARK in text for text in shown)


def test_every_alpha_category_name_is_a_registered_item():
    """The α headings checked above all hide registered alpha settings."""
    assert any("timelapse_lineage" in entry.get("settings", ())
               for entry in ALPHA_FEATURES.values())
