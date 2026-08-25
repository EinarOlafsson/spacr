"""Five modules that stopped being tiles and became buttons on a host.

Barcode QC belongs with Map Barcodes, Classifier Evaluation and Explain
CV Model with Classify, Annotator Agreement with Annotate and AnnData
Export with Measure: each is the second half of a visit to the screen
that produced what it looks at, so each is reached from that screen
rather than from a tile of its own.

What these tests protect is the part of a fold that is easy to lose.

* The button has to be recognisable as the module it replaced -- its own
  icon, its own description, and the maturity colour its TILE lit up in,
  read from the one table rather than retyped, so a button and the tile
  it replaced can never disagree.
* The button has to open the module ITSELF. A fold that shipped a
  summary of the folded screen would be a fold that lost the rest of it,
  so each test names a capability that only the real screen has -- the
  bundle browser, the SHAP panel, the settings form and its Run button --
  and asserts it arrived.
* A settings-driven module opened this way has to keep the host wiring
  its sidebar row gave it: "Explain error" and "Run on a cluster" are
  capabilities of that screen too.
* And none of it may cost the host screen. A strip that cannot be built
  is a missing button; a strip that raises is a module a user cannot
  open at all.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from importlib import import_module

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QPushButton, QStackedWidget, QWidget

from spacr.qt.app import app_stage
from spacr.qt.screens import classify, map_barcodes, measure
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.screens.annotate import FOLDED_APPS as ANNOTATE_FOLDS
from spacr.qt.widgets.fold_strip import FoldStrip


#: (module, host key) for the three hosts whose screen is the generic
#: settings form. Annotate builds its own masthead and is tested apart.
SETTINGS_HOSTS = [
    (map_barcodes, "map_barcodes"),
    (classify, "classify_merged"),
    (measure, "measure"),
]


def _host_screen(qtbot, module, host_key):
    """A host screen with its fold strip installed."""
    screen = AppScreen(app_key=host_key)
    qtbot.addWidget(screen)
    strip = module.install_folds(screen)
    assert strip is not None, f"{host_key}: no fold strip was installed"
    return screen, strip


def _opened(qtbot, opener):
    """Press a fold button's opener and register what it opened."""
    window = opener.open()
    assert window is not None, f"{opener.key}: the button opened nothing"
    qtbot.addWidget(window)
    return window


# ---------------------------------------------------------------------------
# The strip itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module,host_key", SETTINGS_HOSTS,
                         ids=[key for _module, key in SETTINGS_HOSTS])
def test_each_host_carries_its_folded_modules_as_buttons(
        qtbot, qt_theme_applied, module, host_key):
    """The folded modules appear on the host masthead, in declared order.

    The order is the reading order of the visit -- judge the model, then
    ask what it keyed on -- so it is asserted rather than left to a set.
    """
    _screen, strip = _host_screen(qtbot, module, host_key)
    assert list(strip.keys()) == list(module.FOLDED_APPS)


def test_annotate_carries_the_agreement_button(qtbot, qt_theme_applied):
    """Annotate builds its own masthead, so its strip is built with it.

    Nothing installs a strip on this screen from outside: if the header
    stopped making one, the button would simply never appear.
    """
    from spacr.qt.screens.annotate import AnnotateScreen

    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    assert list(screen._fold_strip.keys()) == list(ANNOTATE_FOLDS)
    assert screen._fold_strip.parent() is not None


@pytest.mark.parametrize("module,host_key", SETTINGS_HOSTS,
                         ids=[key for _module, key in SETTINGS_HOSTS])
def test_a_fold_button_is_its_module_icon_and_description(
        qtbot, qt_theme_applied, module, host_key):
    """No text, the module's own icon, the module's own line as tooltip.

    A button labelled with a word would be a new name for the module; the
    icon is the name it already had on Home.
    """
    from spacr.qt.app import APPS

    descriptions = {row[0]: row[2] for row in APPS}
    _screen, strip = _host_screen(qtbot, module, host_key)
    for key in module.FOLDED_APPS:
        button = strip.button_for(key)
        assert button is not None
        assert button.text() == "", f"{key}: the fold button has a caption"
        assert not button.icon().isNull(), f"{key}: the fold button has no icon"
        assert descriptions[key] in button.toolTip()


@pytest.mark.parametrize("module,host_key", SETTINGS_HOSTS,
                         ids=[key for _module, key in SETTINGS_HOSTS])
def test_a_fold_button_lights_in_the_stage_its_tile_lit_in(
        qtbot, qt_theme_applied, module, host_key):
    """The hover colour is the module's maturity, from the one table.

    Two colour tables drift. This asserts the button's ``stage`` property
    -- what the stylesheet selects on -- against ``app_stage``, which is
    what the tile reads, so signing a module off recolours both or
    neither.
    """
    from spacr.qt.theme import STAGE_HOVER, stylesheet

    _screen, strip = _host_screen(qtbot, module, host_key)
    sheet = stylesheet()
    for key in module.FOLDED_APPS:
        stage = app_stage(key)
        button = strip.button_for(key)
        assert button.property("stage") == stage
        assert stage in STAGE_HOVER
        rule = f'QPushButton#FoldButton[stage="{stage}"]:hover'
        assert rule in sheet, f"{key}: nothing lights the button on hover"


# ---------------------------------------------------------------------------
# What the button opens
# ---------------------------------------------------------------------------

def test_barcode_qc_opens_its_own_settings_form_and_run_button(
        qtbot, qt_theme_applied):
    """Map Barcodes' button opens Barcode QC, not a digest of it.

    The abundance-threshold sweep is settings plus a Run button, and
    neither exists on the mapping screen -- so the form arriving with its
    keys is what says nothing was lost.
    """
    screen, _strip = _host_screen(qtbot, map_barcodes, "map_barcodes")
    window = _opened(qtbot, screen._fold_openers[0])

    assert isinstance(window, AppScreen)
    assert window.app_key == "barcode_qc"
    # A PAGE, NOT A WINDOW. A window is the last resort for a fold, and
    # this host has a body to put pages in; the assertion moved with the
    # behaviour rather than being dropped.
    assert not window.isWindow()
    assert screen._fold_pages.currentWidget() is window
    assert screen._fold_pages.tabText(
        screen._fold_pages.indexOf(window)) == "Barcode QC"
    assert len(window._settings_model.collect()) > 0
    assert window._btn_run is not None


def test_measure_opens_the_anndata_export_form(qtbot, qt_theme_applied):
    """Measure's button opens the export module, settings and all.

    AnnData Export has no screen but its settings; a fold that offered
    only "export" with no keys would drop every choice it has.
    """
    screen, _strip = _host_screen(qtbot, measure, "measure")
    window = _opened(qtbot, screen._fold_openers[0])

    assert isinstance(window, AppScreen)
    assert window.app_key == "anndata_export"
    assert not window.isWindow()
    assert screen._fold_pages.tabText(
        screen._fold_pages.indexOf(window)) == "AnnData Export"
    assert len(window._settings_model.collect()) > 0


def test_classify_opens_both_judgements_with_their_own_panels(
        qtbot, qt_theme_applied):
    """Classify's two buttons open the two real screens.

    Each is named by a capability Classify itself has none of: the
    evaluation bundle browser, and the explanation screen's CV panel.
    """
    from spacr.qt.screens.classifier_evaluation import (
        ClassifierEvaluationScreen)
    from spacr.qt.screens.model_explanation import ModelExplanationScreen

    screen, _strip = _host_screen(qtbot, classify, "classify_merged")
    openers = {opener.key: opener for opener in screen._fold_openers}

    evaluation = _opened(qtbot, openers["classifier_evaluation"])
    assert isinstance(evaluation, ClassifierEvaluationScreen)
    assert hasattr(evaluation, "bundles")

    explain = _opened(qtbot, openers["explain_cv"])
    assert isinstance(explain, ModelExplanationScreen)
    assert explain.explain is not None


def test_annotate_opens_the_agreement_screen(qtbot, qt_theme_applied):
    """Annotate's button opens the κ screen, disagreement review included."""
    from spacr.qt.screens.agreement import AgreementScreen
    from spacr.qt.screens.annotate import AnnotateScreen

    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    window = _opened(qtbot, screen._fold_openers[0])

    assert isinstance(window, AgreementScreen)
    # A page beside the annotation grid rather than a window over it --
    # see the barcode QC test for why the assertion moved.
    assert not window.isWindow()
    assert screen._fold_pages.tabText(0) == "Annotate"
    assert screen._fold_pages.tabText(
        screen._fold_pages.indexOf(window)) == "Annotator Agreement"


def test_the_button_press_itself_opens_the_module(qtbot, qt_theme_applied):
    """The wiring, not just the opener: clicking the button opens it.

    ``FoldStrip`` connects through a lambda that drops the click's
    argument; a callback taking no default would raise there and nowhere
    else.
    """
    screen, strip = _host_screen(qtbot, measure, "measure")
    button = strip.button_for("anndata_export")

    button.click()

    window = screen._fold_openers[0].window
    assert window is not None, "clicking the button opened nothing"
    qtbot.addWidget(window)
    assert window.app_key == "anndata_export"


# ---------------------------------------------------------------------------
# The window, once it is open
# ---------------------------------------------------------------------------

def test_pressing_a_fold_button_twice_reuses_one_window(
        qtbot, qt_theme_applied):
    """A second press raises the window; it does not open a second copy.

    These screens own a job runner and a database handle apiece, so two
    of them is two of everything behind them.
    """
    screen, _strip = _host_screen(qtbot, map_barcodes, "map_barcodes")
    opener = screen._fold_openers[0]

    first = _opened(qtbot, opener)
    second = opener.open()

    assert second is first


def test_a_closed_and_deleted_window_is_rebuilt(qtbot, qt_theme_applied):
    """Reopening after Qt deleted the window builds a fresh one.

    The opener keeps a Python reference to a widget whose C++ side can go
    away underneath it; touching that wrapper raises ``RuntimeError``,
    which must read as "gone", not as "broken".
    """
    import shiboken6

    screen, _strip = _host_screen(qtbot, measure, "measure")
    opener = screen._fold_openers[0]
    # Deliberately NOT registered with qtbot: this widget is destroyed
    # mid-test, and a teardown that tried to close it again would report
    # the deletion as a failure of whatever ran next.
    first = opener.open()
    assert first is not None

    shiboken6.delete(first)
    second = opener.open()

    assert second is not None
    assert second is not first
    assert second.app_key == "anndata_export"
    qtbot.addWidget(second)


def test_a_module_that_cannot_be_built_costs_the_window_not_the_host(
        qtbot, qt_theme_applied):
    """A folded module that raises leaves the host screen usable."""
    screen, strip = _host_screen(qtbot, measure, "measure")
    opener = screen._fold_openers[0]

    def boom(_host_window):
        raise RuntimeError("no exporter today")

    opener._build = boom

    assert opener.open() is None
    assert opener.window is None
    assert strip.button_for("anndata_export").isEnabled()


# ---------------------------------------------------------------------------
# Nothing lost in the move
# ---------------------------------------------------------------------------

class _Recorder(QObject):
    """A stand-in main window that records the two host connections."""

    def __init__(self):
        super().__init__()
        self.explained = []
        self.submitted = []

    def _on_explain_error(self, text, app_key):
        self.explained.append((text, app_key))

    def _on_remote_submit_requested(self, app_key, settings):
        self.submitted.append((app_key, settings))


def test_a_folded_settings_screen_keeps_its_host_connections(
        qtbot, qt_theme_applied):
    """"Explain error" and "Run on a cluster" survive the fold.

    Both are wired by ``MainWindow._build_screen`` on the sidebar's
    screen, so a folded screen that skipped them would silently drop two
    buttons' worth of behaviour. The pairs are read from
    ``chaining.HOST_CONNECTIONS``, the same table that wiring is checked
    against.
    """
    from spacr.qt.chaining import HOST_CONNECTIONS

    recorder = _Recorder()
    window = map_barcodes.build_settings_screen("barcode_qc", recorder)
    qtbot.addWidget(window)

    assert set(HOST_CONNECTIONS) == {"error_explain_requested",
                                     "remote_submit_requested"}
    window.error_explain_requested.emit("Traceback...", "barcode_qc")
    window.remote_submit_requested.emit("barcode_qc", {"src": "/tmp"})

    assert recorder.explained == [("Traceback...", "barcode_qc")]
    assert recorder.submitted == [("barcode_qc", {"src": "/tmp"})]


def test_explain_cv_only_gets_a_host_it_can_navigate_through(
        qtbot, qt_theme_applied):
    """The explanation screen sends the user on, or does not offer to.

    It calls ``host._on_train_requested`` directly, so being handed a
    window without that slot would turn one of its buttons into an
    ``AttributeError`` at the moment it was pressed.
    """
    class _Bare(QObject):
        pass

    assert classify._navigable(None) is None
    assert classify._navigable(_Bare()) is None
    recorder = _Recorder()
    recorder._on_train_requested = lambda *_a, **_k: None
    assert classify._navigable(recorder) is recorder


def test_every_folded_key_has_a_builder(qtbot):
    """A key in the strip with no builder would be a dead button."""
    for module in (map_barcodes, classify, measure):
        assert set(module.FOLDED_APPS) <= set(module.BUILDERS), module.HOST_KEY
    from spacr.qt.screens import annotate
    assert set(annotate.FOLDED_APPS) <= set(annotate.FOLD_BUILDERS)


def test_the_folded_modules_still_have_their_registry_metadata(qtbot):
    """The window title, icon and tooltip all come from the registry.

    Folding a module removes its tile, not its row: drop the row and the
    settings-driven folds open on an empty form and the buttons lose the
    descriptions they show. Naming that dependency here means the fold
    fails loudly rather than quietly.
    """
    from spacr.qt.app import APP_META

    folded = (list(map_barcodes.FOLDED_APPS) + list(classify.FOLDED_APPS)
              + list(measure.FOLDED_APPS) + list(ANNOTATE_FOLDS))
    for key in folded:
        assert key in APP_META, f"{key} has no registry metadata"
        assert APP_META[key]["intro"], f"{key} has no description to show"
    for key in ("barcode_qc", "anndata_export"):
        assert APP_META[key]["defaults_module"], (
            f"{key} would open on an empty form")


# ---------------------------------------------------------------------------
# The strip never costs the screen
# ---------------------------------------------------------------------------

def test_a_screen_that_is_not_the_host_gets_no_strip(qtbot, qt_theme_applied):
    """Installing into the wrong screen does nothing at all.

    The seam that calls this walks every module screen, so being asked
    about the wrong one has to be free rather than wrong.
    """
    screen = AppScreen(app_key="mask")
    qtbot.addWidget(screen)

    assert measure.install_folds(screen) is None
    assert getattr(screen, "_fold_strip", None) is None


def test_a_screen_with_no_masthead_gets_no_strip(qtbot):
    """A bespoke screen without a ``ModuleHeader`` is left alone."""
    bare = QWidget()
    qtbot.addWidget(bare)
    bare.app_key = "measure"

    assert measure.install_folds(bare) is None


def test_installing_twice_does_not_add_a_second_strip(
        qtbot, qt_theme_applied):
    """The seam may fire more than once; the masthead may not grow."""
    screen, strip = _host_screen(qtbot, classify, "classify_merged")

    assert classify.install_folds(screen) is strip
    assert len(screen.findChildren(FoldStrip)) == 1


def test_a_strip_that_cannot_be_built_never_takes_the_host_down(
        qtbot, qt_theme_applied, monkeypatch):
    """A broken strip is a missing button, not a module you cannot open."""
    class _Exploding(FoldStrip):
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("strip on fire")

    monkeypatch.setattr(map_barcodes, "FoldStrip", _Exploding)
    screen = AppScreen(app_key="map_barcodes")
    qtbot.addWidget(screen)

    assert map_barcodes.install_folds(screen) is None
    assert screen._header is not None


def test_annotate_opens_even_when_its_strip_cannot_be_built(
        qtbot, qt_theme_applied, monkeypatch):
    """The annotation grid is worth more than the button beside it."""
    from spacr.qt.screens import annotate as annotate_module

    class _Exploding(FoldStrip):
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("strip on fire")

    monkeypatch.setattr(annotate_module, "FoldStrip", _Exploding)
    screen = annotate_module.AnnotateScreen()
    qtbot.addWidget(screen)

    assert getattr(screen, "_fold_strip", None) is None
    assert screen._src_label is not None


def test_the_window_title_falls_back_to_the_registry_name(qtbot):
    """A module with no long-form title still names its own window."""
    assert map_barcodes.folded_module_title("agreement") == \
        "Annotator Agreement"
    assert map_barcodes.folded_module_title("not_an_app") == "Not An App"


def test_connect_host_without_a_window_is_a_no_op(qtbot, qt_theme_applied):
    """A folded module opened outside a window still opens.

    ``connect_host(screen, None)`` is the path a screen built in a test,
    or before a window exists, takes.
    """
    screen = AppScreen(app_key="barcode_qc")
    qtbot.addWidget(screen)

    map_barcodes.connect_host(screen, None)     # must not raise

    assert screen.app_key == "barcode_qc"


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------

class _StackWindow(QWidget):
    """The part of the main window the fold hook uses: a screen stack."""

    def __init__(self):
        super().__init__()
        self._stack = QStackedWidget(self)


def test_every_fold_host_is_named_by_the_lookup(qtbot):
    """A host whose module is not in the table never gets its strip.

    The strip is hung on the shared settings screen from outside, so the
    table is the only thing that says which module owns which host.
    """
    for module in (map_barcodes, classify, measure):
        assert map_barcodes.FOLD_HOST_MODULES.get(module.HOST_KEY), module.HOST_KEY
    for host_key, module_name in map_barcodes.FOLD_HOST_MODULES.items():
        module = import_module(f"spacr.qt.screens.{module_name}")
        assert module.HOST_KEY == host_key
        assert callable(module.install_folds)


def test_the_window_hook_strips_each_host_as_the_stack_reaches_it(
        qtbot, qt_theme_applied):
    """The production path: opening Measure shows the export button.

    ``AppScreen`` is one class serving every module and knows nothing
    about who folded into it, so the strip arrives from outside as each
    screen becomes current -- the same route the settings previews and
    the recipe button take.
    """
    window = _StackWindow()
    qtbot.addWidget(window)
    # A placeholder takes the "already current" slot, so every host below
    # arrives through a real tab change rather than by being added first.
    window._stack.addWidget(QWidget())
    hosts = {}
    for host_key in ("map_barcodes", "classify_merged", "measure"):
        screen = AppScreen(app_key=host_key)
        hosts[host_key] = screen
        window._stack.addWidget(screen)
    # `external_masks` rather than `mask`: Mask Generation hosts folds of
    # its own now, so it is no longer a screen the lookup has never heard
    # of.
    other = AppScreen(app_key="external_masks")
    window._stack.addWidget(other)

    watcher = map_barcodes.install_window_hooks(window)
    assert watcher is not None
    assert map_barcodes.install_window_hooks(window) is watcher

    for host_key, screen in hosts.items():
        window._stack.setCurrentWidget(screen)
        assert isinstance(getattr(screen, "_fold_strip", None), FoldStrip), (
            f"{host_key} became current without its fold buttons")

    window._stack.setCurrentWidget(other)
    assert getattr(other, "_fold_strip", None) is None


def test_the_window_hook_needs_a_stack_and_says_so(qtbot):
    """A window with no screen stack is left alone rather than patched."""
    bare = QWidget()
    qtbot.addWidget(bare)

    assert map_barcodes.install_window_hooks(bare) is None


def test_the_first_screen_is_stripped_without_a_tab_change(
        qtbot, qt_theme_applied):
    """The screen already showing when the hook lands gets its strip too.

    No ``currentChanged`` is coming for it, so a hook that only listened
    would leave the launch screen without its buttons until the user
    navigated away and back.
    """
    window = _StackWindow()
    qtbot.addWidget(window)
    screen = AppScreen(app_key="measure")
    window._stack.addWidget(screen)

    watcher = map_barcodes.install_window_hooks(window)
    assert getattr(screen, "_fold_strip", None) is None
    watcher.install_current()

    assert isinstance(screen._fold_strip, FoldStrip)


def test_a_stack_that_raises_costs_the_strip_not_the_window(
        qtbot, qt_theme_applied):
    """A window whose stack has gone away must not raise on a tab change."""
    window = _StackWindow()
    qtbot.addWidget(window)
    watcher = map_barcodes.install_window_hooks(window)
    del window._stack

    assert watcher.install_current() is None


def test_the_shortcuts_seam_installs_the_fold_hooks(qtbot, qt_theme_applied):
    """The window hooks are asked for where every other screen hook is.

    ``spacr.qt.shortcuts`` is the one place a module reaches a live
    window, and the entry lands with that file rather than with this one
    -- so this asserts the call when it is present instead of turning red
    on a checkout that has the folds but not yet the line.
    """
    import inspect

    from spacr.qt import shortcuts

    source = inspect.getsource(shortcuts._install_window_hooks)
    if "map_barcodes" not in source:
        pytest.skip("the fold window hook is not wired into shortcuts yet")

    window = _StackWindow()
    qtbot.addWidget(window)
    screen = AppScreen(app_key="classify_merged")
    window._stack.addWidget(screen)

    shortcuts._install_window_hooks(window)
    watcher = getattr(window, "_fold_watcher", None)
    assert watcher is not None
    watcher.install_current()
    assert isinstance(screen._fold_strip, FoldStrip)


def test_the_fold_buttons_are_not_confused_with_the_screens_own_actions(
        qtbot, qt_theme_applied):
    """The strip's buttons carry the fold object name and no caption.

    One QSS rule styles every fold button by that name; a button that
    lost it would be styled as an ordinary action and would light in the
    wrong colour.
    """
    from spacr.qt.widgets.fold_strip import BUTTON_NAME

    _screen, strip = _host_screen(qtbot, classify, "classify_merged")
    named = [button for button in strip.findChildren(QPushButton)
             if button.objectName() == BUTTON_NAME]

    assert len(named) == len(classify.FOLDED_APPS)
    assert all(button.accessibleName() for button in named)


# ---------------------------------------------------------------------------
# After the tile is gone
# ---------------------------------------------------------------------------

def test_the_fold_fallback_agrees_with_the_registry(qtbot):
    """What the fallback says a module is must be what its tile said.

    Two tables that describe the same module drift, and this one is only
    consulted after the row it duplicates has been deleted -- the moment
    nobody can compare them any more. So they are compared now, while
    both exist.
    """
    from spacr.qt.app import APPS

    rows = {row[0]: row for row in APPS}
    for key, (name, description, stage) in map_barcodes.FOLD_FALLBACK.items():
        row = rows.get(key)
        if row is None:
            continue
        assert name == row[1], f"{key}: the fallback name is out of date"
        assert description == row[2], f"{key}: the fallback line is stale"
        assert stage == app_stage(key), f"{key}: the fallback stage is stale"


def test_every_folded_module_has_a_fallback(qtbot):
    """A folded key with no fallback loses its name the day its row goes."""
    folded = (list(map_barcodes.FOLDED_APPS) + list(classify.FOLDED_APPS)
              + list(measure.FOLDED_APPS) + list(ANNOTATE_FOLDS))
    for key in folded:
        assert key in map_barcodes.FOLD_FALLBACK, key


def test_a_button_keeps_its_colour_when_the_registry_row_is_dropped(
        qtbot, qt_theme_applied, monkeypatch):
    """Folding a module ends in deleting its row; the button survives it.

    ``unregister_app`` takes the key out of ``APP_STAGE`` as well as out
    of ``APPS``, so with nothing else in place an alpha module's button
    would come back stable-blue with an empty tooltip -- looking finished
    and saying nothing.
    """
    import spacr.qt.app as app_module

    monkeypatch.setattr(
        app_module, "APPS",
        [row for row in app_module.APPS if row[0] != "anndata_export"])
    stages = dict(app_module.APP_STAGE)
    stages.pop("anndata_export", None)
    monkeypatch.setattr(app_module, "APP_STAGE", stages)
    assert app_stage("anndata_export") == "stable"    # the row really is gone

    screen = AppScreen(app_key="measure")
    qtbot.addWidget(screen)
    strip = measure.install_folds(screen)
    button = strip.button_for("anndata_export")

    assert button.property("stage") == "alpha"
    assert "Write the measurements as .h5ad" in button.toolTip()
    assert button.accessibleName() == "AnnData Export"


def test_a_folded_window_is_still_named_when_its_row_is_dropped(
        qtbot, monkeypatch):
    """The window title comes from the fallback once the row is gone.

    A window headed "Barcode Qc" is the key showing through, which is the
    visible half of having deleted the row the name lived in.
    """
    import spacr.qt.app as app_module
    from spacr.qt.screens import app_screen as app_screen_module

    monkeypatch.setattr(
        app_module, "APPS",
        [row for row in app_module.APPS if row[0] != "barcode_qc"])
    titles = dict(app_screen_module.APP_TITLES)
    titles.pop("barcode_qc", None)
    monkeypatch.setattr(app_screen_module, "APP_TITLES", titles)

    assert map_barcodes.folded_module_title("barcode_qc") == "Barcode QC"


def test_an_unknown_key_still_gets_a_readable_title(qtbot):
    """A key with neither a row nor a fallback is titled from the key."""
    assert map_barcodes.fold_description("not_an_app") == ("", "", "")
    assert map_barcodes.folded_module_title("not_an_app") == "Not An App"


def test_restating_a_button_nobody_knows_leaves_it_alone(qtbot,
                                                        qt_theme_applied):
    """A button for an unknown module keeps whatever it already said.

    Blanking a tooltip because a lookup missed would turn a working
    button into a mute one.
    """
    _screen, strip = _host_screen(qtbot, measure, "measure")
    button = strip.button_for("anndata_export")
    before = button.toolTip()

    map_barcodes.restate_fold_button(button, "not_an_app")
    map_barcodes.restate_fold_button(None, "anndata_export")   # must not raise

    assert button.toolTip() == before


# ---------------------------------------------------------------------------
# Every guard, driven into failure
# ---------------------------------------------------------------------------

def test_a_host_module_that_cannot_be_imported_costs_only_its_strip(
        qtbot, qt_theme_applied, monkeypatch):
    """A host named by the lookup but missing from disk is survivable."""
    monkeypatch.setitem(map_barcodes.FOLD_HOST_MODULES, "measure",
                        "no_such_screen_module")
    screen = AppScreen(app_key="measure")
    qtbot.addWidget(screen)

    assert map_barcodes.install_folds_on(screen) is None
    assert getattr(screen, "_fold_strip", None) is None


def test_a_screen_the_lookup_has_never_heard_of_is_skipped(
        qtbot, qt_theme_applied):
    """Most screens host nothing; asking about them must cost nothing.

    Asked about `external_masks` rather than `mask`, which became a fold
    host when Timelapse and Motility moved onto it.
    """
    screen = AppScreen(app_key="external_masks")
    qtbot.addWidget(screen)

    assert map_barcodes.install_folds_on(screen) is None
    assert map_barcodes.install_folds_on(QWidget()) is None


def test_a_folded_screen_opens_even_if_the_chaining_strip_fails(
        qtbot, qt_theme_applied, monkeypatch):
    """The module is worth more than the strip above its Run button."""
    from spacr.qt import chaining

    def boom(*_args, **_kwargs):
        raise RuntimeError("chaining on fire")

    monkeypatch.setattr(chaining, "install_chaining", boom)
    window = map_barcodes.build_settings_screen("anndata_export")
    qtbot.addWidget(window)

    assert window.app_key == "anndata_export"


def test_a_folded_screen_opens_when_the_chaining_module_is_gone(
        qtbot, qt_theme_applied, monkeypatch):
    """Host wiring is a bonus; its absence must not block the window."""
    monkeypatch.setitem(__import__("sys").modules, "spacr.qt.chaining", None)
    screen = AppScreen(app_key="barcode_qc")
    qtbot.addWidget(screen)
    recorder = _Recorder()

    map_barcodes.connect_host(screen, recorder)     # must not raise

    screen.error_explain_requested.emit("x", "barcode_qc")
    assert recorder.explained == []


def test_a_module_title_survives_an_unreadable_title_table(
        qtbot, monkeypatch):
    """A broken title table falls through to what the tile was called."""
    from spacr.qt.screens import app_screen as app_screen_module

    class _Hostile:
        def get(self, _key):
            raise RuntimeError("titles on fire")

    monkeypatch.setattr(app_screen_module, "APP_TITLES", _Hostile())

    assert map_barcodes.folded_module_title("agreement") == \
        "Annotator Agreement"


def test_a_description_survives_an_unreadable_registry(qtbot, monkeypatch):
    """A registry that raises reads as "no answer", not as a crash.

    The maturity lookup is the half most likely to go missing -- folding
    a module ends in deleting its stage -- so it is the half driven into
    failure here.
    """
    import spacr.qt.app as app_module

    def boom(_key):
        raise RuntimeError("registry on fire")

    monkeypatch.setattr(app_module, "app_stage", boom)

    name, description, stage = map_barcodes.fold_description("explain_cv")
    assert name == "Explain CV Model"
    assert stage == "alpha"          # from the fallback, not from the row
    assert "SHAP" in description


def test_a_stack_that_refuses_to_be_followed_installs_no_hook(qtbot):
    """A window whose stack will not connect is left without the hook."""
    class _Refusing:
        @property
        def currentChanged(self):
            raise RuntimeError("no signals here")

    window = QWidget()
    qtbot.addWidget(window)
    window._stack = _Refusing()

    assert map_barcodes.install_window_hooks(window) is None
    assert getattr(window, "_fold_watcher", None) is None


def test_a_signal_that_refuses_to_connect_costs_only_that_wire(qtbot):
    """One unwirable host signal must not cost the other.

    The two connections are independent capabilities, and a screen that
    opened with neither because one refused would be worse than a screen
    that opened with one.
    """
    class _Refusing:
        def connect(self, _slot):
            raise RuntimeError("this signal is spoken for")

    class _Accepting:
        def __init__(self):
            self.slots = []

        def connect(self, slot):
            self.slots.append(slot)

    class _Screen:
        error_explain_requested = _Refusing()

        def __init__(self):
            self.remote_submit_requested = _Accepting()

    screen = _Screen()
    recorder = _Recorder()

    map_barcodes.connect_host(screen, recorder)     # must not raise

    assert screen.remote_submit_requested.slots == [
        recorder._on_remote_submit_requested]


def test_a_host_with_no_buildable_folds_gets_no_empty_strip(
        qtbot, qt_theme_applied):
    """A strip of nothing is chrome; the masthead stays as it was."""
    screen = AppScreen(app_key="measure")
    qtbot.addWidget(screen)

    assert map_barcodes.install_fold_strip(
        screen, "measure", ("not_a_module",), {}) is None
    assert not screen.findChildren(FoldStrip)


def test_an_empty_stack_gives_the_hook_nothing_to_do(qtbot,
                                                     qt_theme_applied):
    """A window whose stack is still empty is asked again later."""
    window = _StackWindow()
    qtbot.addWidget(window)
    watcher = map_barcodes.install_window_hooks(window)

    assert watcher.install_current() is None
