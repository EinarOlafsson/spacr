"""Three modules that are NOT redundant, folded into Regression.

Volcano Explorer, Hit List and Methods & Results each do something the
regression results panel structurally cannot, so a fold that shipped a
summary of any of them would be a fold that deleted a feature. These
tests name the capability in each case and assert it arrived:

* the publication renderer, offered from the pyqtgraph volcano's own menu
  and NOT as 56 style fields on a plot that can honour none of them;
* the whole-table annotation join with its filter bar and its three
  export formats, as a tab beside Coefficients and Guide support;
* the run digest, opened already pointed at the project the regression
  screen is reading.

They also protect the two things a fold is easy to get wrong: the button
has to be recognisable as the module it replaced -- its own icon, its own
sentence, and the maturity colour its TILE lit up in, read from the one
table rather than retyped -- and none of it may cost the host screen.

And the panel defect the Volcano Explorer fold exposed: a guide
permutation stacks every minimum-support family and every response into
one long table, each corrected separately, so drawing it whole puts two
Benjamini-Hochberg families on one volcano and the same guide on it
several times.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

import pandas as pd

from spacr.qt.app import app_stage
from spacr.qt.screens import regression
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.screens.hit_list import HitListScreen
from spacr.qt.widgets.fold_strip import FoldStrip


#: A long guide-permutation table: two support families over the same two
#: guides, each family corrected on its own. Four rows, two guides.
def _two_family_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "feature": ["g1", "g2", "g1", "g2"],
        "grna": ["g1", "g2", "g1", "g2"],
        "coefficient": [1.5, -1.2, 1.5, -1.2],
        "p_value": [0.001, 0.04, 0.001, 0.04],
        "adjusted_p_value": [0.002, 0.04, 0.004, 0.08],
        "minimum_wells_threshold": [1, 1, 2, 2],
    })


def _host(qtbot):
    """A Regression screen with its folds installed."""
    screen = AppScreen(app_key="regression")
    qtbot.addWidget(screen)
    strip = regression.install_folds(screen)
    assert strip is not None, "no fold strip was installed on Regression"
    return screen, strip


def _opened(qtbot, opener):
    """Press one fold opener and register what it opened."""
    window = opener.open()
    assert window is not None, f"{opener.key}: the button opened nothing"
    qtbot.addWidget(window)
    return window


# ---------------------------------------------------------------------------
# The strip
# ---------------------------------------------------------------------------

def test_regression_carries_the_three_folded_modules_as_buttons(
        qtbot, qt_theme_applied):
    """The three appear on the masthead, in declared order.

    The order is the reading order after a run finishes -- the figure, the
    list, the write-up -- so it is asserted rather than left to a set.
    """
    _screen, strip = _host(qtbot)
    assert list(strip.keys()) == list(regression.FOLDED_APPS)


def test_a_regression_fold_button_is_its_module_icon_and_description(
        qtbot, qt_theme_applied):
    """No text, the module's own icon, the module's own line as tooltip.

    A button labelled with a word would be a new name for the module; the
    icon is the name it already had on Home.
    """
    from spacr.qt.app import APPS

    descriptions = {row[0]: row[2] for row in APPS}
    _screen, strip = _host(qtbot)
    for key in regression.FOLDED_APPS:
        button = strip.button_for(key)
        assert button is not None
        assert button.text() == "", f"{key}: the fold button has a caption"
        assert not button.icon().isNull(), f"{key}: no icon on the button"
        assert descriptions[key] in button.toolTip()


def test_a_regression_fold_button_lights_in_the_stage_its_tile_lit_in(
        qtbot, qt_theme_applied):
    """The hover colour is the module's maturity, from the one table.

    Two colour tables drift. This asserts the button's ``stage`` property
    -- what the stylesheet selects on -- against ``app_stage``, which is
    what the tile reads, so signing a module off recolours both or
    neither.
    """
    from spacr.qt.theme import STAGE_HOVER, stylesheet

    _screen, strip = _host(qtbot)
    sheet = stylesheet()
    for key in regression.FOLDED_APPS:
        stage = app_stage(key)
        assert strip.button_for(key).property("stage") == stage
        assert stage in STAGE_HOVER
        rule = f'QPushButton#FoldButton[stage="{stage}"]:hover'
        assert rule in sheet, f"{key}: nothing lights the button on hover"


def test_the_fold_fallback_says_what_the_tile_said(qtbot):
    """The kept copy of each tile's line agrees with the registry.

    Folding a module ends in its row being dropped, and the strip reads a
    button's name, tooltip and hover colour out of that row. The fallback
    is what the button says afterwards, so while both exist they have to
    agree or the button changes its mind the day the row goes.
    """
    from spacr.qt.app import APPS

    rows = {row[0]: row for row in APPS}
    for key, (name, description, stage) in regression.FOLD_FALLBACK.items():
        assert key in rows, f"{key} is not in the registry"
        assert rows[key][1] == name
        assert rows[key][2] == description
        assert app_stage(key) == stage


def test_every_folded_key_can_be_opened(qtbot, qt_theme_applied):
    """A key in the strip with nothing behind it would be a dead button."""
    screen, _strip = _host(qtbot)
    assert [o.key for o in screen._fold_openers] == list(
        regression.FOLDED_APPS)


# ---------------------------------------------------------------------------
# Volcano Explorer: the publication figure
# ---------------------------------------------------------------------------

def test_the_volcano_offers_the_figure_it_cannot_draw_itself(
        qtbot, qt_theme_applied):
    """"Publication figure…" is on the volcano's own right-click menu.

    That is where a user who wants the figure is already looking. It is
    deliberately not ``offer_style`` on this plot: the panel's volcano is
    pyqtgraph and would then carry 56 style entries a pyqtgraph renderer
    can honour none of.
    """
    from spacr.qt.widgets.fast_plots import menu_reading_order

    screen, _strip = _host(qtbot)
    plot = screen._results_panel.volcano

    menu = plot.build_style_menu()
    labels = menu_reading_order(menu)

    assert regression.PUBLICATION_FIGURE_LABEL in labels
    assert plot._style is None, (
        "the pyqtgraph volcano was given a figure style it cannot honour")


def test_the_publication_figure_opens_the_table_that_is_on_screen(
        qtbot, qt_theme_applied):
    """The explorer arrives seeded with the frame, not with a file path.

    The panel may be showing a table nobody can find again -- a live run,
    a bare CSV, a frame handed straight in -- so re-reading the folder
    would publish a different table from the one the user asked for.
    """
    import dataclasses

    screen, strip = _host(qtbot)
    frame = _two_family_frame()
    assert screen._results_panel.set_frame(frame, source="")

    window = _opened(qtbot, screen._publication_opener)

    published = window.explorer.results()
    assert list(published["feature"]) == ["g1", "g2"]
    # The capability that made this fold worth keeping: a matplotlib
    # renderer behind a style with dozens of fields, savable as JSON.
    assert len(dataclasses.fields(window.explorer.style())) > 50


def test_the_menu_entry_and_the_button_open_one_explorer(
        qtbot, qt_theme_applied):
    """Two handles, one door. A second press raises the window it has.

    Two explorers of the same run is two matplotlib canvases and two
    styles the user has to keep in step by hand.
    """
    screen, strip = _host(qtbot)
    assert screen._results_panel.set_frame(_two_family_frame(), source="")

    first = _opened(qtbot, screen._publication_opener)
    strip.button_for("volcano_explorer").click()

    assert screen._publication_opener.window is first


def test_the_publication_figure_falls_back_to_the_run_folder(
        qtbot, qt_theme_applied, tmp_path):
    """With no frame on screen it opens the folder the panel was pointed at.

    A user who has loaded a run and not yet drawn anything still means
    "publish this run" when they ask for the figure.
    """
    folder = tmp_path / "ols_1"
    folder.mkdir()
    table = folder / "results.csv"
    _two_family_frame().to_csv(table, index=False)

    panel = type("_Panel", (), {
        "results_frame": lambda self: None,
        "run_folder": lambda self, path=str(folder): path,
    })()
    window = regression.build_publication_figure(panel)
    qtbot.addWidget(window)

    assert len(window.explorer.results()) == 2


# ---------------------------------------------------------------------------
# Hit List: the tab
# ---------------------------------------------------------------------------

def test_the_hits_tab_sits_beside_guide_support(qtbot, qt_theme_applied):
    """The hit list is a tab on the panel, not a separate destination.

    Beside Coefficients and Guide support, because it is the third view of
    the same run: per coefficient, per guide, per gene.
    """
    screen, _strip = _host(qtbot)
    tabs = screen._results_panel.tabs
    titles = [tabs.tabText(i) for i in range(tabs.count())]

    assert regression.HITS_TAB_TITLE in titles
    assert titles.index(regression.HITS_TAB_TITLE) == (
        titles.index(regression.HITS_TAB_AFTER) + 1)


def test_the_hits_tab_brought_the_filter_bar_and_the_three_exports(
        qtbot, qt_theme_applied):
    """The whole module went in, not a copy of its table.

    The filter bar and the three export formats ARE the capability the
    panel has none of: the panel annotates the one selected gene, and this
    joins annotation across the whole table, recomputes BH over the gene
    family and writes the exact list on screen out three ways.
    """
    from PySide6.QtWidgets import QPushButton

    screen, _strip = _host(qtbot)
    hits = screen._results_panel.hits

    assert isinstance(hits, HitListScreen)
    # The filter bar is live, and what it says is recorded as data on the
    # list -- which is what makes an exported subset self-describing.
    hits._direction.setCurrentText("up")
    hits._guides_spin.setValue(2)
    assert hits.current_filters() == {"direction": "up", "min_guides": 2}

    labels = {button.text() for button in hits.findChildren(QPushButton)}
    assert {"Export CSV…", "Export Markdown…", "Export HTML…"} <= labels
    # And the join the panel cannot do: annotation CSVs across the whole
    # table rather than the one selected gene.
    assert callable(hits.set_metadata_files)


def test_the_hits_tab_follows_the_run_the_panel_loads(
        qtbot, qt_theme_applied, tmp_path):
    """One run on screen, not a hit list of some earlier one.

    A tab showing yesterday's hits beside today's coefficients is worse
    than an empty tab, because nothing on it says the two disagree.
    """
    folder = tmp_path / "ols_2"
    folder.mkdir()
    _two_family_frame().to_csv(folder / "results.csv", index=False)

    screen, _strip = _host(qtbot)
    panel = screen._results_panel
    assert panel.load(str(folder))

    assert panel.hits._folder_edit.text() == str(folder)


def test_the_hits_button_raises_the_tab_rather_than_opening_a_second_list(
        qtbot, qt_theme_applied):
    """One hit list per screen. The button goes to the one that exists."""
    screen, strip = _host(qtbot)
    panel = screen._results_panel

    strip.button_for("hit_list").click()

    assert panel.tabs.currentWidget() is panel.hits


def test_the_hits_button_opens_the_module_when_there_is_no_panel(
        qtbot, qt_theme_applied):
    """A screen with no results panel still reaches the hit list.

    The tab needs a panel to live on; the capability must not need one.
    """
    opener = regression.HitsOpener(screen=None)
    window = _opened(qtbot, opener)

    assert isinstance(window, HitListScreen)


# ---------------------------------------------------------------------------
# Methods & Results: the run digest
# ---------------------------------------------------------------------------

def test_methods_and_results_opens_seeded_with_the_project(
        qtbot, qt_theme_applied, tmp_path):
    """The path this screen asks the user to type is filled in.

    The regression screen already holds the project the run came from, and
    typing it a second time is the friction this fold removes.
    """
    project = tmp_path / "plate1"
    run = project / "results" / "score" / "ols_1"
    run.mkdir(parents=True)
    _two_family_frame().to_csv(run / "results.csv", index=False)

    screen, strip = _host(qtbot)
    assert screen._results_panel.load(str(run))

    window = _opened(qtbot, screen._fold_openers[2])

    assert window._fields["project"].text() == str(project)
    assert window._fields["results"].text() == str(run)
    # The capability: it is the digest builder itself, not a summary of
    # it -- and it is the one screen in spaCR that treats a model's
    # output as untrusted, so the verification strip comes with it.
    assert window._build_button is not None
    assert callable(window.digest)
    assert window._provenance is not None


def test_the_project_path_survives_a_list_of_plates():
    """A host that DOES carry ``src`` yields the first plate, not the list.

    Regression is handed score and count tables rather than a plate
    folder, so the project comes from the run folder; the fallback is what
    serves a host whose settings form names the project directly, and
    handed a list it would otherwise fill the path box with a Python repr.
    """
    class _Model:
        def collect(self):
            return {"src": ["/data/plate1", "/data/plate2"]}

    screen = type("_Screen", (), {"_settings_model": _Model()})()

    assert regression.project_path(screen) == os.path.abspath("/data/plate1")


def test_the_project_is_the_folder_above_the_run(tmp_path):
    """A run writes to ``<project>/results/<score>/<kind>``.

    The digest reads the project's manifest, its measurements and its QC,
    none of which are inside the run folder the panel is pointed at.
    """
    run = tmp_path / "plate1" / "results" / "score" / "ols_1"
    run.mkdir(parents=True)

    panel = type("_Panel", (), {
        "run_folder": lambda self, path=str(run): path})()
    screen = type("_Screen", (), {"_results_panel": panel})()

    assert regression.project_path(screen) == str(tmp_path / "plate1")


# ---------------------------------------------------------------------------
# One correction family on one volcano
# ---------------------------------------------------------------------------

def test_two_support_families_do_not_share_one_volcano(
        qtbot, qt_theme_applied):
    """A permutation run draws each guide once, corrected once.

    The long table holds the same guides once per minimum-support
    threshold with a separate Benjamini-Hochberg correction each, so drawn
    whole it puts two correction families on one axis and the same guide
    on the plot two to four times.
    """
    screen, _strip = _host(qtbot)
    panel = screen._results_panel

    assert panel.set_frame(_two_family_frame(), source="")

    shown = panel.results_frame()
    assert len(shown) == 2
    assert set(shown["minimum_wells_threshold"]) == {1}


def test_two_responses_are_two_corrections_and_only_one_is_drawn(
        qtbot, qt_theme_applied):
    """Each fitted response is its own family and must not be pooled."""
    frame = pd.DataFrame({
        "feature": ["g1", "g2", "g1", "g2"],
        "coefficient": [1.0, -1.0, 0.5, -0.5],
        "p_value": [0.01, 0.2, 0.03, 0.4],
        "outcome": ["score_a", "score_a", "score_b", "score_b"],
    })
    screen, _strip = _host(qtbot)
    panel = screen._results_panel

    assert panel.set_frame(frame, source="")

    assert set(panel.results_frame()["outcome"]) == {"score_a"}


def test_an_ordinary_coefficient_table_is_handed_through_untouched():
    """Every parametric run has neither column, and pays nothing.

    Asserted on identity rather than on equality: a cut that copied every
    frame would double the memory of the largest table spaCR draws.
    """
    frame = pd.DataFrame({"feature": ["g1"], "coefficient": [1.0]})

    assert regression.single_correction_family(frame) is frame


def test_the_cut_agrees_with_the_explorers_own(tmp_path):
    """The panel and the Volcano Explorer keep the same rows.

    They draw the same run side by side. Two rules for "which family" is
    two different pictures of one table, and only one of them can be
    right.
    """
    from spacr.qt.screens import volcano

    table = tmp_path / "guide_permutation_results_long.csv"
    frame = _two_family_frame()
    frame.to_csv(table, index=False)

    mine = regression.single_correction_family(frame)
    theirs = volcano.load_results(str(table))

    assert list(mine["feature"]) == list(theirs["feature"])
    assert list(mine["minimum_wells_threshold"]) == list(
        theirs["minimum_wells_threshold"])


def test_installing_the_cut_twice_does_not_stack_two_of_them(
        qtbot, qt_theme_applied):
    """A second install would wrap the wrapper and cut an already-cut frame."""
    screen, _strip = _host(qtbot)
    panel = screen._results_panel

    assert regression.install_correction_families(panel) is False


# ---------------------------------------------------------------------------
# The strip never costs the screen
# ---------------------------------------------------------------------------

def test_a_screen_that_is_not_the_host_gets_no_strip(qtbot, qt_theme_applied):
    """Installing into the wrong screen does nothing at all.

    The seam that calls this walks every module screen, so being asked
    about the wrong one has to be free rather than wrong.
    """
    screen = AppScreen(app_key="measure")
    qtbot.addWidget(screen)

    assert regression.install_folds(screen) is None
    assert regression.install_extras(screen) is False


def test_a_second_install_returns_the_strip_the_first_one_made(
        qtbot, qt_theme_applied):
    """The stack watcher installs on every tab change; twice is not two."""
    screen, strip = _host(qtbot)

    assert regression.install_folds(screen) is strip
    assert isinstance(strip, FoldStrip)


# ---------------------------------------------------------------------------
# None of it may take the screen down with it
# ---------------------------------------------------------------------------
#
# Every guard below is driven into the failure it guards against rather
# than left to be read: a fold that raises is a module the user cannot
# open at all, and a screen that raises while being folded into is no
# screen at all.

def test_a_table_that_is_not_a_table_is_handed_straight_back():
    """The cut is asked about whatever reaches ``set_frame``.

    A None, an empty frame and an object that is not a table all reach it,
    because the panel is what decides they are unusable and says so.
    """
    empty = pd.DataFrame({"feature": []})
    stranger = object()

    assert regression.single_correction_family(None) is None
    assert regression.single_correction_family(empty) is empty
    assert regression.single_correction_family(stranger) is stranger


def test_the_cut_is_not_installed_where_it_cannot_be(qtbot):
    """A panel with no ``set_frame`` has nothing to put the cut in front of."""
    class _NoSetFrame:
        set_frame = "not callable"

    assert regression.install_correction_families(None) is False
    assert regression.install_correction_families(_NoSetFrame()) is False


def test_a_panel_that_cannot_name_its_run_is_not_an_error(
        qtbot, qt_theme_applied):
    """A run folder that raises reads as "no folder", not as a crash."""
    class _Panel:
        def run_folder(self):
            raise RuntimeError("no path today")

        def results_frame(self):
            return None

    screen = type("_Screen", (), {"_results_panel": _Panel()})()
    assert regression.project_path(screen) == ""

    window = regression.build_publication_figure(_Panel())
    qtbot.addWidget(window)
    assert window.explorer.results().empty


def test_a_panel_whose_frame_cannot_be_read_falls_back_to_the_folder(
        qtbot, qt_theme_applied, tmp_path):
    """A frame that raises must not cost the figure the run it can find."""
    folder = tmp_path / "ols_9"
    folder.mkdir()
    _two_family_frame().to_csv(folder / "results.csv", index=False)

    class _Panel:
        def results_frame(self):
            raise RuntimeError("the table is gone")

        def run_folder(self, path=None):
            return str(folder)

    window = regression.build_publication_figure(_Panel())
    qtbot.addWidget(window)

    assert len(window.explorer.results()) == 2


def test_the_publication_entry_is_not_offered_where_there_is_no_volcano(
        qtbot):
    """A panel with no plot, or a plot with no menu, is left alone."""
    class _NoMenu:
        pass

    assert regression.install_publication_figure(None, lambda: None) is False
    panel = type("_Panel", (), {"volcano": _NoMenu()})()
    assert regression.install_publication_figure(panel, lambda: None) is False


def test_the_hits_tab_is_not_built_where_it_cannot_live(
        qtbot, qt_theme_applied):
    """No panel and no tab bar are both "nowhere to put it", not failures."""
    assert regression.install_hits_tab(None) is None
    assert regression.install_hits_tab(object()) is None
    assert regression.raise_hits_tab(None) is False


def test_a_hits_tab_that_cannot_be_built_costs_the_tab_not_the_panel(
        qtbot, qt_theme_applied, monkeypatch):
    """A hit list that raises leaves the results panel whole.

    The tab is one view of the run; the panel is every other view of it.
    """
    from spacr.qt.screens import hit_list as hit_list_module

    class _Explodes:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no hit list today")

    monkeypatch.setattr(hit_list_module, "HitListScreen", _Explodes)

    screen = AppScreen(app_key="regression")
    qtbot.addWidget(screen)
    panel = screen._results_panel

    assert regression.install_hits_tab(panel) is None
    assert panel.tabs.count() > 0


def test_a_second_call_returns_the_tab_the_first_one_made(
        qtbot, qt_theme_applied):
    """The stack watcher installs on every tab change; twice is not two tabs."""
    screen, _strip = _host(qtbot)
    panel = screen._results_panel

    assert regression.install_hits_tab(panel) is panel.hits


def test_a_hits_tab_that_cannot_follow_the_run_says_nothing(
        qtbot, qt_theme_applied):
    """A failed follow is a stale tab, not a failed load.

    The panel has already put the run on screen by the time this runs, and
    an exception here would undo that.
    """
    class _Hits:
        def load_folder(self, _folder):
            raise RuntimeError("cannot read that folder")

    panel = type("_Panel", (), {
        "hits": _Hits(),
        "run_folder": lambda self: "/somewhere/ols_1",
    })()

    regression._follow_the_run(panel)    # must not raise


def test_a_panel_with_no_loaded_signal_still_gets_its_tab(
        qtbot, qt_theme_applied):
    """The tab is worth having even where nothing tells it a run arrived."""
    from PySide6.QtWidgets import QTabWidget, QWidget

    holder = QWidget()
    qtbot.addWidget(holder)
    tabs = QTabWidget(holder)
    tabs.addTab(QWidget(holder), regression.HITS_TAB_AFTER)
    panel = type("_Panel", (QWidget,), {})(holder)
    panel.tabs = tabs

    hits = regression.install_hits_tab(panel)

    assert hits is not None
    assert tabs.tabText(tabs.indexOf(hits)) == regression.HITS_TAB_TITLE


def test_the_project_path_of_a_screen_that_has_none_is_empty():
    """Every way the project is unknown answers "", not a broken path."""
    class _Raises:
        def collect(self):
            raise RuntimeError("no settings")

    assert regression.project_path(None) == ""
    assert regression.project_path(object()) == ""
    assert regression.project_path(
        type("_S", (), {"_settings_model": _Raises()})()) == ""
    assert regression.project_path(
        type("_S", (), {"_settings_model": type(
            "_M", (), {"collect": lambda self: {"src": ""}})()})()) == ""


def test_the_hits_button_survives_a_screen_that_cannot_raise_its_results(
        qtbot, qt_theme_applied):
    """A results page that refuses to come forward still opens the list."""
    from PySide6.QtWidgets import QWidget

    class _Screen(QWidget):
        #: A panel with no Hits tab: the button has a panel to try and
        #: nothing on it to raise, which is the state a screen is in
        #: before the tab is installed.
        _results_panel = object()

        def _raise_the_results_tab(self):
            raise RuntimeError("no results page")

    host = _Screen()
    qtbot.addWidget(host)
    opener = regression.HitsOpener(host)
    window = _opened(qtbot, opener)

    assert isinstance(window, HitListScreen)


def test_the_extras_are_not_installed_where_there_is_no_panel(qtbot):
    """The host key alone is not enough; the panel is what carries them."""
    screen = type("_Screen", (), {"app_key": "regression"})()

    assert regression.install_extras(screen) is False


def test_a_host_with_no_masthead_gets_no_strip(qtbot):
    """A screen that cannot show buttons is a smaller screen, not an error."""
    screen = type("_Screen", (), {"app_key": "regression"})()

    assert regression.install_folds(screen) is None


def test_a_strip_that_cannot_be_built_costs_the_buttons_not_the_screen(
        qtbot, qt_theme_applied):
    """A masthead that refuses the strip leaves the regression screen up."""
    screen = AppScreen(app_key="regression")
    qtbot.addWidget(screen)

    def refuse(_widget, *_args, **_kwargs):
        raise RuntimeError("no room on this masthead")

    screen._header.add_trailing = refuse

    assert regression.install_folds(screen) is None
    assert screen._results_panel is not None


def test_a_panel_whose_signal_refuses_the_tab_still_gets_the_tab(
        qtbot, qt_theme_applied):
    """A follow that cannot be wired costs the following, not the tab.

    The list is still reachable and still loadable by hand; only the
    automatic re-point is lost.
    """
    from PySide6.QtWidgets import QTabWidget, QWidget

    class _Refuses:
        def connect(self, _slot):
            raise RuntimeError("this signal takes no slots")

    holder = QWidget()
    qtbot.addWidget(holder)
    tabs = QTabWidget(holder)
    tabs.addTab(QWidget(holder), regression.HITS_TAB_AFTER)
    panel = type("_Panel", (QWidget,), {})(holder)
    panel.tabs = tabs
    panel.loaded = _Refuses()

    hits = regression.install_hits_tab(panel)

    assert hits is not None
    assert tabs.indexOf(hits) >= 0
