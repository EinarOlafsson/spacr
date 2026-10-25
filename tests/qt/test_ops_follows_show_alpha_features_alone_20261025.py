"""OPS is reached through Show alpha features alone, and shows its settings.

2026-10-25, tutorial 76: with Show alpha features on and the separate
alpha-maturity preference off, Mask -> OPS opened a page whose every section,
the "Load test data…" one among them, was hidden at Essentials and at All.
OPS is an alpha module, so each of its sections carried alpha maturity, and
the switch that opens it answered to neither preference. OPS is now an alpha
feature (item 372 in ``ALPHA_FEATURES``): the switch shows only while Show
alpha features is on, and while it is on the page shows its own sections
whatever the maturity preference says.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QPushButton, QWidget

import spacr.qt.app as app_module
from spacr.qt import preferences
from spacr.qt import settings_search as ss
from spacr.qt.screens import mask as mask_screen
from spacr.qt.widgets.section import Section, _sections_below


@pytest.fixture
def alpha_preferences():
    saved = preferences.get_show_alpha(), preferences._get_show_alpha_features()

    def choose(*, features, maturity):
        preferences._set_show_alpha_features(features)
        preferences.set_show_alpha(maturity)

    yield choose
    preferences.set_show_alpha(saved[0])
    preferences._set_show_alpha_features(saved[1])


@pytest.fixture
def open_mask(qtbot):
    windows = []

    def open_it():
        win = app_module.MainWindow()
        windows.append(win)
        win.resize(1400, 900)
        win.show()
        win._on_nav_selected("mask")
        qtbot.waitUntil(lambda: "mask" in win._screens, timeout=10000)
        return win, win._screens["mask"]

    yield open_it
    for win in windows:
        win.close()
        win.deleteLater()


def _the_example_button(page):
    return next(button for button in page.findChildren(QPushButton)
                if button.text().startswith("Load test data"))


def _section_holding(page, widget):
    return next(section for section in _sections_below(page)
                if isinstance(section, Section) and section.isAncestorOf(widget))


def test_ops_shows_its_test_data_section_at_both_levels_with_alpha_features_alone(
        qtbot, alpha_preferences, open_mask):
    alpha_preferences(features=True, maturity=False)
    win, mask = open_mask()
    switch = mask._ops_switch
    assert switch.objectName() == "MaskOpsSwitch"
    qtbot.waitUntil(lambda: switch.isVisibleTo(mask), timeout=5000)
    fold = mask_screen.ops_page(mask)
    fold.set_shown(True)
    qtbot.waitUntil(lambda: fold.page is not None, timeout=10000)
    page = fold.page
    button = _the_example_button(page)
    section = _section_holding(page, button)
    bar = next(widget for widget in win.findChildren(QWidget)
               if widget.objectName() == ss.BAR_NAME)
    for level in (ss.ESSENTIALS, ss.ALL, ss.ESSENTIALS):
        bar.set_level(level)
        qtbot.waitUntil(lambda: section.isVisibleTo(page) and button.isVisibleTo(page),
                        timeout=5000)
    sections = [s for s in _sections_below(page) if isinstance(s, Section)]
    assert sections and all(s.isVisibleTo(page) for s in sections)


@pytest.mark.parametrize("maturity", [False, True])
def test_without_alpha_features_mask_offers_no_ops_switch(
        qtbot, alpha_preferences, open_mask, maturity):
    alpha_preferences(features=False, maturity=maturity)
    _win, mask = open_mask()
    assert mask._ops_switch.objectName() == "MaskOpsSwitch"
    qtbot.wait(50)
    assert not mask._ops_switch.isVisibleTo(mask)
