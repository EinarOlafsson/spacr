"""The per-object grid switch and returning category bodies, when things fail.

Pins what the user keeps when the optional parts of the settings panel go
wrong:

* a screen still builds with its settings form when deciding the object
  rows or registering the right-hand text column raises;
* the per-object grid is not mounted when the module has no model, too few
  shared questions, or the grid itself cannot be built -- the flat form
  stays;
* the Preferences switch reports "no change" when the preference cannot be
  read, when a destroyed grid is treated as unmounted, and when the panel
  has no layout to mount into;
* taking the grid down gives the flat rows back even when the model cannot
  be asked, and survives a grid whose C++ half has gone;
* the "does this heading hold anything" test ignores nested non-sections
  and owners that never registered rows;
* detaching hidden category bodies skips one that dies mid-detach, and a
  body that comes back survives a failing language pass.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QVBoxLayout, QWidget  # noqa: E402

from spacr.qt.screens.app_screen import AppScreen  # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus  # noqa: E402
from spacr.qt.widgets.section import Section  # noqa: E402

pytestmark = pytest.mark.qt


def _build(qtbot, key="regression"):
    screen = AppScreen(key)
    qtbot.addWidget(screen)
    return screen


@pytest.fixture
def screen(qtbot):
    widget = _build(qtbot)
    yield widget
    retire_pyqtgraph_menus(widget)


def _grid_on(monkeypatch, value=True):
    monkeypatch.setattr("spacr.qt.preferences.get_object_grid_enabled",
                        lambda: value)


# --------------------------------------------------------------------------
# building


def test_a_screen_builds_when_optional_build_steps_raise(qtbot, monkeypatch):
    import sys

    from spacr.qt.screens.settings_model import SettingsWidgets

    real = SettingsWidgets.refresh_object_visibility
    refused = []

    def refuse_from_the_screen(model):
        caller = sys._getframe(1).f_code.co_filename
        if caller.endswith("app_screen.py"):
            refused.append(caller)
            raise ValueError("cannot decide")
        return real(model)

    def refuse(*_args, **_kwargs):
        raise ValueError("cannot register")

    monkeypatch.setattr(SettingsWidgets, "refresh_object_visibility",
                        refuse_from_the_screen)
    monkeypatch.setattr("spacr.qt.live_zoom.register_text_column", refuse)
    screen = _build(qtbot)
    try:
        assert refused, "the screen's own object-row decision was not asked"
        assert screen._settings_model is not None
        assert screen.rendered_settings_sections(), (
            "the form must still be on screen")
        assert screen._runtime_wrap is not None
    finally:
        retire_pyqtgraph_menus(screen)


# --------------------------------------------------------------------------
# mounting the per-object grid


def test_no_model_means_no_grid(screen, monkeypatch):
    _grid_on(monkeypatch)
    layout = QVBoxLayout()
    was = screen._settings_model
    screen._settings_model = types.SimpleNamespace(_widgets={})
    try:
        screen._mount_the_object_grid(layout)
    finally:
        screen._settings_model = was
    assert getattr(screen, "_object_grid", None) is None
    assert layout.count() == 0


def test_a_module_with_too_few_shared_questions_keeps_its_flat_form(
        screen, monkeypatch):
    _grid_on(monkeypatch)
    layout = QVBoxLayout()
    before = list(screen._settings_sections)
    screen._mount_the_object_grid(layout)
    assert getattr(screen, "_object_grid", None) is None
    assert layout.count() == 0
    assert list(screen._settings_sections) == before


def test_a_grid_that_cannot_be_built_leaves_the_form(screen, monkeypatch):
    _grid_on(monkeypatch)

    def broken(*_args, **_kwargs):
        raise TypeError("no grid today")

    monkeypatch.setattr(
        "spacr.qt.widgets.object_settings_grid.ObjectSettingsGrid", broken)
    layout = QVBoxLayout()
    screen._mount_the_object_grid(layout)
    assert getattr(screen, "_object_grid", None) is None
    assert layout.count() == 0


# --------------------------------------------------------------------------
# the Preferences switch


def test_an_unreadable_preference_changes_nothing(screen, monkeypatch):
    def unreadable():
        raise OSError("settings file locked")

    monkeypatch.setattr("spacr.qt.preferences.get_object_grid_enabled",
                        unreadable)
    assert screen.apply_object_grid_preference() is False


def test_a_destroyed_grid_counts_as_unmounted(screen, monkeypatch):
    class _GoneGrid:
        def parent(self):
            raise RuntimeError("Internal C++ object already deleted.")

    _grid_on(monkeypatch, False)
    screen._object_grid = _GoneGrid()
    assert screen.apply_object_grid_preference() is False


def test_no_layout_to_mount_into_changes_nothing(screen, monkeypatch):
    _grid_on(monkeypatch, True)
    screen._object_grid = None
    screen._settings_layout = None
    assert screen.apply_object_grid_preference() is False
    assert screen._object_grid is None


# --------------------------------------------------------------------------
# unmounting


def test_the_grid_comes_off_even_when_the_model_cannot_give_rows_back(
        screen):
    asked = []

    def refuse(keys):
        asked.append(tuple(keys))
        raise ValueError("model busy")

    section = Section("Per-object settings", screen)
    grid = QWidget(section)
    screen._object_grid = grid
    was = screen._settings_model
    screen._settings_model = types.SimpleNamespace(
        hide_the_rows_the_grid_speaks_for=refuse)
    try:
        screen._unmount_the_object_grid()
    finally:
        screen._settings_model = was
    assert asked == [()]
    assert screen._object_grid is None
    assert screen._object_grid_binding is None
    assert section.parent() is None


def test_a_model_that_cannot_unhide_and_a_gone_grid_are_survived(screen):
    class _GoneGrid:
        def parent(self):
            raise RuntimeError("Internal C++ object already deleted.")

    was = screen._settings_model
    sections = list(screen._settings_sections)
    screen._settings_model = types.SimpleNamespace()
    screen._object_grid = _GoneGrid()
    try:
        screen._unmount_the_object_grid()
    finally:
        screen._settings_model = was
    assert screen._object_grid is None
    assert list(screen._settings_sections) == sections


# --------------------------------------------------------------------------
# headings and bodies


def test_nested_non_sections_and_unregistered_owners_hold_nothing():
    owner = types.SimpleNamespace(
        _nested_sections=lambda: [types.SimpleNamespace(), object()])
    assert AppScreen._section_holds_anything(owner) is False


def test_a_body_that_dies_mid_detach_is_skipped(screen, monkeypatch):
    class _Dying:
        _settings_top_level_order = 0

        def _detach_body_while_hidden(self):
            raise RuntimeError("Internal C++ object already deleted.")

    class _Fine:
        _settings_top_level_order = 1

        def _detach_body_while_hidden(self):
            return True

    fine = _Fine()
    monkeypatch.setattr(screen, "rendered_settings_sections",
                        lambda: (_Dying(), fine))
    monkeypatch.setattr(screen, "_heading_is_waiting", lambda s: False)
    assert screen._detach_what_the_form_hides() == 1
    assert fine._body_came_back == screen._a_category_body_came_back


@pytest.mark.parametrize("error", [RuntimeError, ValueError])
def test_a_body_that_comes_back_survives_a_failing_language_pass(
        screen, monkeypatch, error):
    retargeted = []

    def fail(_body):
        raise error("pass failed")

    monkeypatch.setattr("spacr.qt.i18n.retranslate_widget_tree", fail)
    monkeypatch.setattr(
        "spacr.qt.screens.settings_model.retarget_field_tooltips",
        retargeted.append)
    section = Section("Advanced", screen)
    screen._a_category_body_came_back(section)
    assert retargeted == [], "the tooltip move must wait for a good pass"
    assert section._body is not None
