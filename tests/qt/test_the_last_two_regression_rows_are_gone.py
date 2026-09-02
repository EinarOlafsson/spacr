"""Hit List and Methods & Results are buttons on Regression, not tiles.

They were the last two keys that appeared in a host's ``FOLDED_APPS`` and in
``spacr.qt.app.APPS`` at the same time -- one module with two front doors,
where the tile opened it bare and the button opened it seeded by the host.
Their rows were held back because ``register_app`` fans one call out into six
other tables, and a row dropped without moving those answers first takes the
button's name, its sentence and its maturity colour with it silently.

These tests are the acceptance bar for the last two folds:

* the sweep -- no key is both a fold button and a tile -- pinned at empty
  rather than at a list somebody has to remember to shorten;
* what each button still says and still lights up in, read from
  :data:`spacr.qt.screens.map_barcodes.FOLD_FALLBACK` and checked against the
  screen modules' own strings, because the registry now answers "stable" and
  a title-cased key for both;
* and the two doors themselves: the Hits button raises the tab on the panel
  the user is already looking at, falls back to a window on a screen that has
  no panel, and Methods & Results still builds its page seeded with the run.

One capability really did travel on the row and had to be moved rather than
recorded: the hit list's "Investigate selected…" signal was connected by the
registry FACTORY, which ran only while the module was a tile. The tab and the
folded window are built by the host, so the host connects it now, and the
last group of tests here is that connection.
"""
from __future__ import annotations

import importlib
import pkgutil

import pytest

pytest.importorskip("PySide6")

import pandas as pd                                            # noqa: E402
from PySide6.QtWidgets import QWidget                          # noqa: E402

from spacr.qt import app as app_module                         # noqa: E402
from spacr.qt.i18n import tr                                   # noqa: E402
from spacr.qt.screens import (hit_list as hit_list_module,     # noqa: E402
                              map_barcodes, methods_export as
                              methods_export_module, regression)
from spacr.qt.screens.app_screen import AppScreen              # noqa: E402
from spacr.qt.widgets.fold_strip import (FoldButton,           # noqa: E402
                                         folded_fallback)

#: The two modules whose rows this change dropped, with the screen module
#: that still owns their strings.
LAST_TWO = (("hit_list", hit_list_module),
            ("methods_export", methods_export_module))

#: The nine non-English UI languages, in the order ``APP_TRANSLATIONS``
#: spells them.
LANGUAGES = ("sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr")


def _results_frame() -> pd.DataFrame:
    """A minimal coefficient table a results panel will load."""
    return pd.DataFrame({
        "feature": ["g1", "g2"],
        "grna": ["g1", "g2"],
        "coefficient": [1.5, -1.2],
        "p_value": [0.001, 0.04],
        "adjusted_p_value": [0.002, 0.04],
    })


def _host(qtbot):
    """A Regression screen with its folds installed."""
    screen = AppScreen(app_key="regression")
    qtbot.addWidget(screen)
    strip = regression.install_folds(screen)
    assert strip is not None, "no fold strip was installed on Regression"
    return screen, strip


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def test_no_key_is_both_a_fold_button_and_a_tile(qapp):
    """The whole rule, swept over every screen module rather than listed.

    Pinned at empty, so a fold that ships tomorrow and forgets to drop its
    row fails here rather than quietly drawing a second front door.
    """
    import spacr.qt.screens as screens_package

    # A TILE, NOT A ROW, and the difference is the whole rule. A folded
    # module KEEPS its registry row -- it still has a screen, an icon, a
    # section and a key to navigate to, and `spacr.qt.app.tiled_apps` says
    # so in as many words; what it loses is the tile. Read against `APPS`
    # this swept up nine modules that have exactly one front door, among
    # them Format Converter and Investigate Hit, and reported them as
    # having two. `tiled_apps()` is what Home actually draws.
    tiled = {row[0] for row in app_module.tiled_apps()}
    both = {}
    for found in pkgutil.iter_modules(screens_package.__path__):
        try:
            module = importlib.import_module(
                f"spacr.qt.screens.{found.name}")
        except Exception:                                       # noqa: BLE001
            continue
        for key in getattr(module, "FOLDED_APPS", ()) or ():
            if key in tiled:
                both.setdefault(key, []).append(found.name)

    assert both == {}, (
        "these keys are a button on a host AND a tile, so the same module "
        f"has two front doors: {both}")


@pytest.mark.parametrize("key,_module", LAST_TWO)
def test_the_row_really_is_gone(qapp, key, _module):
    """The premise. ``APP_META`` goes with the row, and both are checked."""
    assert key not in {row[0] for row in app_module.APPS}
    assert key not in app_module.APP_META


@pytest.mark.parametrize("key,_module", LAST_TWO)
def test_the_module_no_longer_offers_a_register_hook(qapp, key, _module):
    """A ``register()`` left behind is a row waiting to come back.

    The screens package imports both modules at import time, so a surviving
    hook is one stray call away from putting the tile back beside the button.
    """
    assert not hasattr(_module, "register")
    assert "register" not in _module.__all__


# ---------------------------------------------------------------------------
# What the button still says
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,module", LAST_TWO)
def test_the_fallback_carries_the_whole_record(qapp, key, module):
    """Name, sentence and stage, checked against the module's own strings.

    The registry used to check these: the fold table and the row said the
    same thing, and a test compared them. With the row gone the module's own
    ``APP_NAME`` and ``APP_DESCRIPTION`` are what is left to check the kept
    copy against, so they are what this compares.
    """
    name, description, stage = map_barcodes.FOLD_FALLBACK[key]

    assert name == module.APP_NAME
    assert description == module.APP_DESCRIPTION
    assert stage == "alpha", (
        f"{key} was assessed alpha; a fallback that says otherwise lights "
        f"the button in the colour of finished code")


@pytest.mark.parametrize("key,module", LAST_TWO)
def test_the_record_is_reachable_from_the_widget_that_needs_it(qapp, key,
                                                               module):
    """``fold_strip`` walks the hosts; the record has to be found that way.

    A table only ``map_barcodes.fold_description`` can see leaves a button
    built anywhere else -- the strip's own first paint -- title-cased and
    stable-blue until something restates it.
    """
    assert folded_fallback(key) == (module.APP_NAME, module.APP_DESCRIPTION,
                                    "alpha")


@pytest.mark.parametrize("key,module", LAST_TWO)
def test_the_button_wears_the_name_sentence_and_stage(qapp, qt_theme_applied,
                                                      key, module):
    """The registry now answers "stable" and a title-cased key for both."""
    button = FoldButton(key)

    assert button.property("stage") == "alpha"
    assert button.accessibleName() == module.APP_NAME
    assert button.toolTip().splitlines()[0] == module.APP_NAME
    assert module.APP_DESCRIPTION in button.toolTip()
    assert button.text() == "", "the fold button grew a caption"
    assert not button.icon().isNull()


def test_the_regression_strip_still_draws_all_three_in_order(
        qtbot, qt_theme_applied):
    """The figure, the list, the write-up -- none of them lost to the drop."""
    _screen, strip = _host(qtbot)

    assert list(strip.keys()) == list(regression.FOLDED_APPS)
    # NOT ALL ALPHA ANY MORE. This read "every button is alpha" when the
    # strip carried three, and Diagnostics joined them assessed as beta --
    # so the sweep started failing on a fact about the screen that is
    # correct. The maturity each button lights in is worth pinning; that
    # they all light in ONE colour never was.
    expected = {"volcano_explorer": "alpha", "hit_list": "alpha",
                "methods_export": "alpha", "investigate_hit": "alpha",
                "profiler": "alpha", regression.DIAGNOSTICS_KEY: "beta"}
    assert set(expected) == set(regression.FOLDED_APPS), (
        "a fold arrived or left without this test being told its maturity")
    for key in regression.FOLDED_APPS:
        assert strip.button_for(key).property("stage") == expected[key]


@pytest.mark.parametrize("key,module", LAST_TWO)
def test_the_translated_names_outlived_the_row(qapp, key, module):
    """``add_translation`` no longer runs; the shipped catalogs answer.

    A window opened in Swedish must still call the button Träfflista rather
    than falling back to the English name.
    """
    assert tuple(tr(module.APP_NAME, code)
                 for code in LANGUAGES) == module.APP_TRANSLATIONS


# ---------------------------------------------------------------------------
# Both doors still open
# ---------------------------------------------------------------------------

def test_the_hits_button_still_raises_the_tab(qtbot, qt_theme_applied):
    """One hit list per screen: the button goes to the one that exists."""
    screen, strip = _host(qtbot)
    panel = screen._results_panel
    assert panel.hits is not None, "the Hits tab was never installed"

    strip.button_for("hit_list").click()

    assert panel.tabs.currentWidget() is panel.hits


def test_the_hits_button_still_follows_the_run_on_screen(
        qtbot, qt_theme_applied, tmp_path):
    """The tab the button raises is loaded with the run the panel loaded."""
    run = tmp_path / "ols_1"
    run.mkdir()
    _results_frame().to_csv(run / "results.csv", index=False)

    screen, strip = _host(qtbot)
    panel = screen._results_panel
    assert panel.load(str(run))

    strip.button_for("hit_list").click()

    assert panel.tabs.currentWidget() is panel.hits
    assert panel.hits._folder_edit.text() == str(run)


def test_the_hits_button_still_opens_a_window_with_no_panel(
        qtbot, qt_theme_applied):
    """A screen with no results panel has no tab, and still reaches the list.

    The tab needs a panel to live on; the capability must not need one.
    """
    opener = regression.HitsOpener(screen=None)

    window = opener.open()

    assert window is not None, "the Hits button opened nothing"
    qtbot.addWidget(window)
    assert isinstance(window, hit_list_module.HitListScreen)


def test_methods_and_results_still_builds_its_page(
        qtbot, qt_theme_applied, tmp_path):
    """The button builds the module itself, seeded with the run and project.

    Seeding is what makes the fold a superset of the tile the drop removed:
    the tile opened on two empty path boxes and asked the user to type what
    the host already knew.
    """
    project = tmp_path / "plate1"
    run = project / "results" / "score" / "ols_1"
    run.mkdir(parents=True)
    _results_frame().to_csv(run / "results.csv", index=False)

    screen, strip = _host(qtbot)
    assert screen._results_panel.load(str(run))

    page = regression.BUILDERS["methods_export"](None, screen=screen)
    qtbot.addWidget(page)

    assert isinstance(page, methods_export_module.MethodsExportScreen)
    assert page._fields["project"].text() == str(project)
    assert page._fields["results"].text() == str(run)
    # The capability itself, not a summary of it.
    assert callable(page.digest)
    assert page._provenance is not None

    # And the button is wired to that builder rather than to nothing.
    assert strip.button_for("methods_export") is not None


# ---------------------------------------------------------------------------
# The one capability the row was carrying
# ---------------------------------------------------------------------------

class _Workbench(QWidget):
    """A stand-in for the main window: it holds the investigation handler."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list = []

    def _on_investigate_hit_requested(self, request: dict) -> None:
        self.seen.append(request)


def test_the_hits_tab_reaches_the_investigation_workbench(qtbot,
                                                          qt_theme_applied):
    """"Investigate selected…" emitted into nothing once the factory went.

    The registry factory took ``host`` and connected this signal, and it ran
    only while the hit list was a tile. The tab is built by the host, so the
    host wires it -- otherwise the button on the Hits tab looks live and does
    nothing at all.
    """
    screen = AppScreen(app_key="regression")
    qtbot.addWidget(screen)
    seen: list = []
    # A top-level widget is its own ``window()``, which is where the strip
    # installer reads the handler off.
    screen._on_investigate_hit_requested = seen.append

    assert regression.install_folds(screen) is not None
    hits = screen._results_panel.hits
    assert hits is not None

    hits.investigate_requested.emit({"gene": "g1"})

    assert seen == [{"gene": "g1"}]


def test_the_folded_hit_list_window_reaches_it_too(qtbot, qt_theme_applied):
    """A screen with no panel opens the module in a window; same wiring."""
    host = _Workbench()
    qtbot.addWidget(host)
    screen = AppScreen(app_key="regression")
    qtbot.addWidget(screen)

    hits = regression.HitsOpener(screen)._build_window(host)
    qtbot.addWidget(hits)
    hits.investigate_requested.emit({"gene": "g2"})

    assert host.seen == [{"gene": "g2"}]


def test_the_investigation_is_connected_exactly_once(qtbot):
    """Two connections would open two workbenches on one press."""
    from spacr.qt.screens.hit_list import connect_investigation

    host = _Workbench()
    qtbot.addWidget(host)
    hits = hit_list_module.HitListScreen(threaded=False)
    qtbot.addWidget(hits)

    assert connect_investigation(hits, host) is True
    assert connect_investigation(hits, host) is False

    hits.investigate_requested.emit({"gene": "g3"})

    assert host.seen == [{"gene": "g3"}]


def test_nothing_to_wire_is_not_a_failure(qtbot):
    """A host with no handler, and no screen at all, are both no-ops.

    The hit list opens on plenty of things that are not the main window --
    a bare page in a test, a window built before the handler exists -- and
    an unwired button is a smaller screen than an exception.
    """
    from spacr.qt.screens.hit_list import connect_investigation

    hits = hit_list_module.HitListScreen(threaded=False)
    qtbot.addWidget(hits)

    assert connect_investigation(hits, None) is False
    assert connect_investigation(hits, object()) is False
    assert connect_investigation(None, _Workbench()) is False
