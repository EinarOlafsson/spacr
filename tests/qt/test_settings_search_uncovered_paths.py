"""Edges of the settings-search strip: a refused QSS block and a stale size list.

Both are places where the strip is told something it did not expect by a
part of the shell it does not own. Neither may cost a user their settings
form, which is the only thing on the page that matters.
"""
from __future__ import annotations

import importlib.util

import pytest

from PySide6.QtWidgets import QScrollArea, QSplitter, QVBoxLayout, QWidget

from spacr.qt import settings_search as SS


def _load_module_copy(name):
    """Execute ``spacr/qt/settings_search.py`` again under a throwaway name.

    Kept out of ``sys.modules`` on purpose: the real module stays the one
    every other screen is holding, so re-running the body cannot swap the
    ``SettingsSearchBar`` class out from under a live window.
    """
    spec = importlib.util.spec_from_file_location(name, SS.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_theme_registry_that_refuses_the_block_still_yields_a_usable_module(
        monkeypatch):
    """Registering the strip's QSS is decoration; importing the strip is not."""
    from spacr.qt import theme

    def _refuse(*_args, **_kwargs):
        raise RuntimeError("the widget QSS registry is closed")

    monkeypatch.setattr(theme, "register_widget_qss", _refuse)

    module = _load_module_copy("spacr.qt._settings_search_refused_qss")

    assert module.BAR_NAME == SS.BAR_NAME
    assert issubclass(module.SettingsSearchBar, QWidget)
    assert callable(module.install)


class _StubModel:
    """The two attributes ``SettingsSearchBar`` reads off a screen's model."""

    _widgets: dict = {}

    def keys_matching(self, _query):
        return []

    def modified_keys(self):
        return []

    def essential_keys(self):
        return []


def _screen_in_a_splitter(qtbot):
    """A screen shaped exactly as ``install`` requires, and nothing more."""
    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    splitter = QSplitter(host)
    layout.addWidget(splitter)

    sibling = QWidget()
    splitter.addWidget(sibling)
    scroll = QScrollArea()
    splitter.addWidget(scroll)

    screen = QWidget(host)
    screen.app_key = "probe"
    screen._settings_scroll = scroll
    screen._settings_model = _StubModel()
    screen._settings_sections = [QWidget(screen)]

    host.resize(600, 400)
    host.show()
    qtbot.waitExposed(host)
    # The host is returned so the caller holds it: nothing else owns the
    # splitter, and a collected host takes the C++ objects with it.
    return host, screen, splitter, scroll


def test_a_size_list_that_no_longer_matches_the_splitter_is_not_forced_back(
        qtbot):
    """A stale list would collapse a pane; the strip leaves the layout alone.

    ``install`` reads the splitter's sizes before it moves the scroll area
    into its new container and puts the list back afterwards. The list is
    only put back when it still has one entry per pane — here it reports a
    single zero-width entry for a two-pane splitter, and restoring that
    would leave the settings column with no width at all.
    """
    _host, screen, splitter, scroll = _screen_in_a_splitter(qtbot)
    splitter.sizes = lambda: [0]

    bar = SS.install(screen)

    assert bar is not None
    assert screen._settings_search is bar
    assert scroll.parentWidget() is not splitter, "the scroll area was not moved"
    pane = splitter.widget(1)
    assert pane.objectName() == SS.PANE_NAME
    assert pane.isAncestorOf(scroll) and pane.isAncestorOf(bar)
    qtbot.waitUntil(lambda: pane.width() > 0, timeout=2000)


def test_a_matching_size_list_is_restored_over_the_new_pane(qtbot):
    """The ordinary case: the splitter keeps the proportions it had."""
    _host, screen, splitter, scroll = _screen_in_a_splitter(qtbot)
    splitter.setSizes([120, 480])
    qtbot.waitUntil(lambda: splitter.sizes()[0] > 0, timeout=2000)
    before = list(splitter.sizes())

    bar = SS.install(screen)

    assert bar is not None
    assert splitter.count() == len(before)
    qtbot.waitUntil(lambda: splitter.sizes() == before, timeout=2000)
