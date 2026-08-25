"""Activation stopped being a tile and became a page on Classify.

An activation map is a picture of what a trained classifier looked at, so
it is read beside the model that produced it: the module is a button on
Classify's masthead that opens its own screen as a page next to the
training settings.

What these tests protect is the part of that move that is easy to lose.

* The module has to arrive WHOLE. It is not a few settings categories on
  the host's form -- nineteen of its twenty-seven settings are its own,
  ``spacr.qt.bridge`` sends its key to a different entry point than the
  host's Run button, its hyperparameter panel searches a disjoint set of
  parameters and its drop policy takes a different kind of folder. Each
  is asserted on the driven page, because each is a capability that a
  fold into the host's form would have quietly dropped.
* The page has to keep the wiring its sidebar row gave it: "Explain
  error" and "Run on a cluster" are capabilities of that screen too.
* And the one navigation that leads to it has to land. Explain CV Model
  -- Classify's other page -- offers "Open Activation Maps" and asks its
  host for ``activation_maps``, which is not a screen key: the request
  navigated to a key the registry has never heard of and built an orphan
  page with an empty form on it. It opens the page beside it now.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QWidget

from spacr.qt.screens import activation, classify, map_barcodes
from spacr.qt.screens.app_screen import AppScreen

APP_KEY = activation.APP_KEY


def _host(qtbot):
    """A Classify screen with its fold strip installed."""
    screen = AppScreen(app_key=classify.HOST_KEY)
    qtbot.addWidget(screen)
    strip = classify.install_folds(screen)
    assert strip is not None, "Classify carries no fold strip"
    return screen, strip


def _page(qtbot, host):
    """The activation page, opened the way the button opens it."""
    page = activation.open_page(host)
    assert page is not None, "the fold opened nothing"
    qtbot.addWidget(page)
    return page


class _Window(QObject):
    """A stand-in main window that records what it was asked to open."""

    def __init__(self):
        super().__init__()
        self.explained = []
        self.submitted = []
        self.opened = []

    def _on_train_requested(self, key, seed):
        self.opened.append((key, dict(seed)))

    def _on_explain_error(self, text, app_key):
        self.explained.append((text, app_key))

    def _on_remote_submit_requested(self, app_key, settings):
        self.submitted.append((app_key, settings))


# ---------------------------------------------------------------------------
# The button
# ---------------------------------------------------------------------------

def test_activation_is_the_last_of_classifys_three_folds(
        qtbot, qt_theme_applied):
    """The strip reads in the order of the visit, and ends here.

    Judge the model, ask which measured features it keyed on, then ask
    where in the image it looked -- so the order is asserted rather than
    left to a set.
    """
    _screen, strip = _host(qtbot)

    assert list(strip.keys()) == list(classify.FOLDED_APPS)
    assert classify.FOLDED_APPS[-1] == APP_KEY
    assert callable(classify.BUILDERS[APP_KEY])


def test_the_button_is_the_modules_own_icon_and_line(qtbot, qt_theme_applied):
    """No caption, the module's icon, and the sentence its tile carried.

    A button labelled with a word would be a new name for the module; the
    icon is the name it already had on Home.
    """
    _screen, strip = _host(qtbot)
    button = strip.button_for(APP_KEY)
    name, description, stage = map_barcodes.fold_description(APP_KEY)

    assert button.text() == "", "the fold button has a caption"
    assert not button.icon().isNull(), "the fold button has no icon"
    assert description and description in button.toolTip()
    assert name in button.toolTip()
    assert button.property("stage") == stage


def test_the_button_says_what_it_is_with_or_without_a_registry_row(qtbot):
    """The tooltip and the hover colour survive the row being dropped.

    Folding a module ends in its registry row going away, and the button
    then reads ``map_barcodes.FOLD_FALLBACK`` instead. Asserted as the
    pair rather than as one or the other: while the row is still there
    the two must agree, and once it goes the fallback must answer alone,
    or the button goes mute the day somebody deletes the row.
    """
    from spacr.qt.app import APPS

    name, description, stage = map_barcodes.fold_description(APP_KEY)
    assert name and description, "nothing would show on the button"
    assert stage in ("alpha", "beta", "stable")

    row = next((row for row in APPS if row[0] == APP_KEY), None)
    fallback = map_barcodes.FOLD_FALLBACK.get(APP_KEY)
    if row is not None and fallback is not None:
        assert (fallback[0], fallback[1]) == (row[1], row[2])


# ---------------------------------------------------------------------------
# What the button opens
# ---------------------------------------------------------------------------

def test_the_button_opens_the_module_as_a_page_beside_the_training_form(
        qtbot, qt_theme_applied):
    """A page on the host, not a window over it, and the module itself.

    A window is what a fold becomes only when its host has no body to
    make pages out of, and Classify has one.
    """
    screen, strip = _host(qtbot)

    strip.button_for(APP_KEY).click()

    page = activation.opener_on(screen).window
    assert page is not None, "clicking the button opened nothing"
    qtbot.addWidget(page)
    assert isinstance(page, AppScreen)
    assert page.app_key == APP_KEY
    assert not page.isWindow()
    pages = screen._fold_pages
    assert pages.currentWidget() is page
    assert pages.tabText(pages.indexOf(page)) == "Activation Maps"
    assert pages.tabText(0) == "Classify", "the host lost its own page"


def test_the_attribution_settings_arrive_and_the_host_never_had_them(
        qtbot, qt_theme_applied):
    """The nineteen settings that make this a different run, on the page.

    This is the measurement that says the fold could not have been a few
    categories on the host's form: asserted as a set difference, both
    ways, rather than by opening the page and looking at it.
    """
    host, _strip = _host(qtbot)
    page = _page(qtbot, host)

    on_page = set(page._settings_model.collect())
    on_host = set(host._settings_model.collect())
    only_here = on_page - on_host

    assert {"cam_type", "target_layer", "smoothgrad_samples",
            "smoothgrad_sigma", "occlusion_window", "occlusion_stride",
            "ig_steps", "ig_baseline", "attribution_steps",
            "attribution_baseline", "sanity_check", "overlay",
            "normalize_input", "object_type"} <= only_here
    assert len(only_here) == 19
    assert page._btn_run is not None, "the page cannot run anything"


def test_the_page_runs_the_attribution_not_the_training(
        qtbot, qt_theme_applied):
    """Its Run button starts a different job than the host's.

    The reason the settings could not simply be revealed on the host's
    form: one form has one Run button, and these settings need the other
    entry point.
    """
    from spacr.qt.bridge import resolve_pipeline_entry

    host, _strip = _host(qtbot)
    page = _page(qtbot, host)

    entry = resolve_pipeline_entry(page.app_key)
    assert entry is not None, "the page's Run button is dead"
    assert entry.__name__ == "generate_activation_map"
    assert resolve_pipeline_entry(host.app_key).__name__ != entry.__name__


def test_the_page_sweeps_the_attribution_parameters_not_the_models(
        qtbot, qt_theme_applied):
    """The hyperparameter search follows the page's own key.

    The panel searches whatever its screen's ``app_key`` names, so a
    module folded into the host's form would have had its eight
    attribution parameters searched by nothing at all.
    """
    host, _strip = _host(qtbot)
    page = _page(qtbot, host)

    assert page._hyperparam is not None, "the page lost its search"
    assert page._hyperparam.app_key == APP_KEY
    swept = set(page._hyperparam._value_edits)
    assert swept == {"cam_type", "target_layer", "smoothgrad_samples",
                     "smoothgrad_sigma", "occlusion_window",
                     "occlusion_stride", "ig_steps", "ig_baseline"}
    assert not swept & set(host._hyperparam._value_edits)


def test_the_page_keeps_its_own_drop_policy(qtbot, qt_theme_applied):
    """A folder dropped on the page is read as the page reads folders.

    The host takes a training set; this module takes the measurements or
    the model directory it attributes, so the two policies are different
    objects and the page has to carry its own.
    """
    from spacr.qt.dnd_handlers import MeasurementsDropHandler

    host, _strip = _host(qtbot)
    page = _page(qtbot, host)

    assert page.acceptDrops()
    assert isinstance(page._dnd_handler, MeasurementsDropHandler)
    assert not isinstance(host._dnd_handler, MeasurementsDropHandler)


def test_the_folded_page_keeps_its_host_connections(qtbot, qt_theme_applied):
    """"Explain error" and "Run on a cluster" survive the fold.

    Both are wired by ``MainWindow._build_screen`` on the sidebar's
    screen, so a folded screen that skipped them would silently drop two
    buttons' worth of behaviour.
    """
    from spacr.qt.chaining import HOST_CONNECTIONS

    window = _Window()
    page = classify.BUILDERS[APP_KEY](window)
    qtbot.addWidget(page)

    assert set(HOST_CONNECTIONS) == {"error_explain_requested",
                                     "remote_submit_requested"}
    page.error_explain_requested.emit("Traceback...", APP_KEY)
    page.remote_submit_requested.emit(APP_KEY, {"src": "/tmp"})

    assert window.explained == [("Traceback...", APP_KEY)]
    assert window.submitted == [(APP_KEY, {"src": "/tmp"})]


def test_pressing_the_button_twice_keeps_one_page(qtbot, qt_theme_applied):
    """A second press raises the page; it does not build a second copy.

    The screen owns a job runner and a console, so two of them is two of
    everything behind them -- and the settings typed into the first would
    be invisible on the second.
    """
    host, strip = _host(qtbot)
    first = _page(qtbot, host)

    strip.button_for(APP_KEY).click()

    assert activation.opener_on(host).window is first
    assert host._fold_pages.count() == 2, "the page was added twice"


# ---------------------------------------------------------------------------
# The navigation that leads to it
# ---------------------------------------------------------------------------

def test_explain_cv_opens_the_page_rather_than_a_key_nothing_knows(
        qtbot, qt_theme_applied):
    """"Open Activation Maps" lands on the page beside it.

    It asks its host for ``activation_maps``, which is not a screen key:
    through the main window that request navigated to a key the registry
    has never heard of and built an orphan page with an empty form. The
    host it is given now is the navigator, which answers with the page.
    """
    host, strip = _host(qtbot)
    strip.button_for("explain_cv").click()
    explain = [opener for opener in host._fold_openers
               if opener.key == "explain_cv"][0].window
    qtbot.addWidget(explain)

    explain.explain.activation_button.click()

    page = activation.opener_on(host).window
    assert page is not None, "the button opened nothing"
    qtbot.addWidget(page)
    assert page.app_key == APP_KEY
    assert host._fold_pages.currentWidget() is page


def test_the_navigator_seeds_the_page_it_opens(qtbot, qt_theme_applied):
    """A navigation carrying settings puts them in the form it opens.

    The window's own navigation seeds the screen it lands on, so one that
    stopped at opening the page would answer the same request with less.
    """
    host, _strip = _host(qtbot)
    navigator = activation.ExplainNavigator(None)
    navigator.attach(host)

    opened = navigator._on_train_requested(
        "activation_maps", {"smoothgrad_samples": 8, "target_layer": "layer4"})

    qtbot.addWidget(opened)
    settings = opened._settings_model.collect()
    assert settings["smoothgrad_samples"] == 8
    assert settings["target_layer"] == "layer4"


def test_the_navigator_forwards_every_other_destination_to_the_window(qtbot):
    """It stands in for the window on one request and no others.

    Explain CV's sibling panels seed training screens through the same
    slot, so a navigator that answered everything would strand them.
    """
    window = _Window()
    navigator = activation.ExplainNavigator(window)

    navigator._on_train_requested("umap", {"src": "/data"})

    assert window.opened == [("umap", {"src": "/data"})]


def test_the_navigator_names_the_module_the_registry_knows(qtbot):
    """With no page to open, it asks for ``activation``, not the alias.

    The path a folded screen that ended up in a window takes: there is no
    host above it, so the request goes on to the window -- under the key
    that names the module rather than the one that reaches nothing.
    """
    window = _Window()
    navigator = activation.ExplainNavigator(window)
    navigator.attach(QWidget())

    navigator._on_train_requested("activation_maps", {})

    assert window.opened == [(APP_KEY, {})]


def test_a_navigator_with_nothing_to_ask_answers_nothing(qtbot):
    """No page and no navigable window is the path that used to crash.

    ``ExplainCvPanel`` called ``host._on_train_requested`` on whatever it
    was given, so a host without that slot turned its button into an
    ``AttributeError``. The navigator is always a host that has it, and
    says so by returning None rather than by raising.
    """
    assert activation.ExplainNavigator(None)._on_train_requested(
        "activation_maps", {}) is None
    assert activation.ExplainNavigator(QObject())._on_train_requested(
        "umap", {}) is None
