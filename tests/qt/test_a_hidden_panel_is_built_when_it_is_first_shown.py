"""A module builds the panels it shows, and the hidden ones when they appear.

ITEMS 284 AND 380. Opening a module polishes every widget on its screen
against the whole stylesheet, so a widget nobody can see still costs its
share of the open. Counted on 2026-09-21 with the page on screen, the
largest hidden shares were whole panels inside cards that start hidden:

    Regression   results tabs + figure pages     ~740 of 1,455 widgets
    Mask         Cellpose live preview            ~280 of 1,124
                 folded Timelapse track preview   ~130
    Classify     hyperparameter search            ~130 of   857
    UMAP         hyperparameter search            ~130 of   669
    Measure      crop preview                     ~115 of   591

Each is now built the first time its card is shown or its attribute is
used. These tests hold both halves: NOT built when the module opens, and
built -- whole, wired, translated, button roles intact -- when first
needed.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget     # noqa: E402

from spacr.qt.screens import app_screen                         # noqa: E402
from spacr.qt.screens.app_screen import AppScreen               # noqa: E402
from spacr.qt.widgets.card import _CardBuiltWhenShown as Card    # noqa: E402


@contextmanager
def _language(code):
    """Set the UI language through the environment for one block."""
    from spacr.qt import i18n

    before = os.environ.get(i18n.ENV_LANGUAGE)
    os.environ[i18n.ENV_LANGUAGE] = code
    try:
        yield
    finally:
        if before is None:
            os.environ.pop(i18n.ENV_LANGUAGE, None)
        else:
            os.environ[i18n.ENV_LANGUAGE] = before


def _screen(qtbot, key):
    screen = AppScreen(key)
    qtbot.addWidget(screen)
    return screen


def _has(root, class_name):
    return any(type(w).__name__ == class_name
               for w in root.findChildren(QWidget))


# -- the card ---------------------------------------------------------------

def test_a_card_does_not_build_its_body_until_it_is_shown(qtbot):
    calls = []
    card = Card(title="Probe")
    qtbot.addWidget(card)
    card.build_body_when_first_shown(lambda: calls.append(1))
    card.hide()
    assert calls == [] and card.body_is_built() is False

    card.show()
    assert calls == [1], "showing the card did not build its body"
    card.hide()
    card.show()
    assert calls == [1], "the body was built twice"


def test_the_body_is_there_before_the_card_is_visible(qtbot):
    """Built from ``setVisible``, before Qt shows anything: no empty frame."""
    seen = []
    card = Card(title="Probe")
    qtbot.addWidget(card)

    def build():
        seen.append(card.isVisible())
        card.body_layout.addWidget(QLabel("body"))

    card.build_body_when_first_shown(build)
    card.show()
    assert seen == [False]


def test_a_card_revealed_by_its_parent_builds_its_body(qtbot):
    calls = []
    host = QWidget()
    qtbot.addWidget(host)
    card = Card(title="Probe", parent=host)
    QVBoxLayout(host).addWidget(card)
    card.build_body_when_first_shown(lambda: calls.append(1))
    host.show()
    qtbot.waitUntil(lambda: calls == [1], timeout=2000)


# -- regression ---------------------------------------------------------------

def test_regression_opens_without_its_results_panel(qtbot):
    screen = _screen(qtbot, "regression")
    assert screen._part_is_owed(app_screen._REGRESSION_RESULTS)
    assert not _has(screen, "RegressionResultsPanel")
    assert not _has(screen, "MeasurementScanPanel")
    assert screen._results_panel_if_built() is None


def test_asking_for_the_results_panel_builds_all_of_it_once(qtbot):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    screen = _screen(qtbot, "regression")
    panel = screen._results_panel
    assert isinstance(panel, RegressionResultsPanel)
    assert not screen._part_is_owed(app_screen._REGRESSION_RESULTS)
    tabs = screen._results_tabs
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    assert titles == ["Runs", "Results", "Measurements", "Cells"]
    assert tabs.currentWidget() is screen._results_page
    assert screen._figure_grid is not None
    assert screen._figures_stack.count() == 3
    assert screen._results_panel is panel, "built a second time"
    assert screen._results_tabs.tabBar().usesScrollButtons() is False, (
        "the late part missed the screen's scroll-arrow sweep")


def test_showing_the_figures_card_builds_the_results(qtbot):
    screen = _screen(qtbot, "regression")
    screen._figures_card.setVisible(True)
    assert not screen._part_is_owed(app_screen._REGRESSION_RESULTS)
    assert _has(screen._figures_card, "RegressionResultsPanel")


def test_the_fold_extras_wait_for_the_panel_instead_of_building_it(qtbot):
    """Regression's fold strip adds the Hits tab to the results panel."""
    from spacr.qt.screens import regression

    screen = _screen(qtbot, "regression")
    assert regression.install_extras(screen) is True
    assert screen._part_is_owed(app_screen._REGRESSION_RESULTS), (
        "installing the extras built the panel they decorate")
    regression.project_path(screen)
    assert screen._part_is_owed(app_screen._REGRESSION_RESULTS), (
        "asking which run is loaded built the panel to say none")

    panel = screen._results_panel
    assert getattr(panel, "hits", None) is not None, (
        "the Hits tab never arrived with the panel")


def test_closing_regression_does_not_build_its_results(qtbot):
    screen = _screen(qtbot, "regression")
    screen.close()
    assert screen._part_is_owed(app_screen._REGRESSION_RESULTS)


def test_the_results_are_translated_when_they_are_built(qtbot):
    from spacr.qt.i18n import tr

    with _language("sv"):
        screen = _screen(qtbot, "regression")
        tabs = screen._results_tabs
        rendered = tabs.tabText(1)
    expected = tr("Results", "sv")
    assert expected != "Results", "no Swedish catalogue row to test with"
    assert rendered == expected


# -- hyperparameter search -------------------------------------------------

@pytest.mark.parametrize("key", ["umap", "classify_merged"])
def test_the_hyperparameter_panel_waits_for_its_switch(qtbot, key):
    from spacr.qt.screens.hyperparam import HyperparamPanel

    screen = _screen(qtbot, key)
    assert screen._part_is_owed(app_screen._HYPERPARAM_PANEL)
    assert not _has(screen, "HyperparamPanel")
    assert screen._hp_switch is not None, "the switch went with the panel"

    screen._on_hyperparam_switch(True)
    assert isinstance(screen._hyperparam, HyperparamPanel)
    assert screen._hyperparam.parent() is screen._hyperparam_card.body
    assert screen._hyperparam_card.isVisibleTo(screen)


def test_umap_keeps_its_gpu_switch(qtbot):
    screen = _screen(qtbot, "umap")
    assert screen._gpu_switch is not None
    assert screen._part_is_owed(app_screen._HYPERPARAM_PANEL)


def test_the_hyperparameter_panel_is_translated_when_built(qtbot):
    from spacr.qt.i18n import tr

    with _language("sv"):
        screen = _screen(qtbot, "umap")
        screen._on_hyperparam_switch(True)
        texts = {str(w.text()) for w in screen._hyperparam.findChildren(
            QWidget) if hasattr(w, "text") and callable(w.text)}
    expected = tr("Walk", "sv")
    assert expected != "Walk", "no Swedish catalogue row to test with"
    assert expected in texts and "Walk" not in texts


# -- mask and measure previews ---------------------------------------------

def test_mask_opens_without_its_live_preview(qtbot):
    screen = _screen(qtbot, "mask")
    assert screen._part_is_owed(app_screen._LIVE_PREVIEW)
    assert not _has(screen, "LivePreviewPanel")
    assert getattr(screen._live_preview_card, "_refresh_button", None), (
        "the card lost its Refresh button")


def test_the_live_switch_builds_and_seeds_the_preview(qtbot, monkeypatch):
    from spacr.qt.widgets.live_preview import LivePreviewPanel

    seeded = []
    monkeypatch.setattr(LivePreviewPanel, "apply_settings",
                        lambda self, settings: seeded.append(dict(settings)))
    screen = _screen(qtbot, "mask")
    screen._on_preview_switch(True)
    assert isinstance(screen._live_preview, LivePreviewPanel)
    assert seeded, "the preview opened at its own defaults, not the form's"
    assert screen._live_preview_card.isVisibleTo(screen)


def test_a_source_typed_before_the_preview_exists_is_loaded_with_it(
        qtbot, monkeypatch):
    """The eager screen's hidden panel loaded ``src`` as it was typed."""
    from spacr.qt.widgets.live_preview import LivePreviewPanel

    loads = []
    monkeypatch.setattr(LivePreviewPanel, "load_source_async",
                        lambda self, src: loads.append(src))
    screen = _screen(qtbot, "mask")
    screen._autoload_live_preview("/data/plate1")
    assert loads == [] and screen._part_is_owed(app_screen._LIVE_PREVIEW), (
        "typing a path built the preview")
    screen._live_preview_card.setVisible(True)
    assert loads == ["/data/plate1"]


def test_run_keeps_its_colour_when_the_preview_is_built_translated(qtbot):
    """Polished before translated, as the eager screen was.

    ``button_roles`` reads a button's visible text when it is polished; a
    Swedish "Kör förhandsgranskning" is not an English "Run".
    """
    from PySide6.QtWidgets import QApplication, QPushButton

    from spacr.qt.button_roles import install_button_roles

    install_button_roles(QApplication.instance())
    with _language("sv"):
        screen = _screen(qtbot, "mask")
        screen._on_preview_switch(True)
        runs = [b for b in screen._live_preview.findChildren(QPushButton)
                if b.property("_spacr_i18n_text") == "Run preview"]
    assert runs, "no Run preview button found -- wrong probe"
    assert all(b.property("buttonActionRole") == "positive" for b in runs)


def test_measure_opens_without_its_crop_preview(qtbot):
    from spacr.qt.widgets.measure_preview import MeasurePreviewPanel

    screen = _screen(qtbot, "measure")
    assert screen._part_is_owed(app_screen._MEASURE_PREVIEW)
    assert not _has(screen, "MeasurePreviewPanel")
    screen._on_preview_switch(True)
    assert isinstance(screen._measure_preview, MeasurePreviewPanel)


def test_an_attached_preview_is_built_when_it_is_first_opened(qtbot):
    """The registry's folded previews: Mask's Timelapse track preview."""
    from spacr.qt.preview_registry import attach_folded

    screen = _screen(qtbot, "mask")
    host = attach_folded(screen, "timelapse")
    assert host is not None
    assert host.panel_is_built() is False
    assert not _has(screen, "TimelapsePreviewPanel")
    host.on_toggled(True)
    assert host.panel_is_built() is True
    assert type(host.panel).__name__ == "TimelapsePreviewPanel"
    assert host.card.isVisibleTo(screen)
