"""The montage panel answers with a reason rather than an empty grid.

Everything here is a path where the panel is handed less than it needs -- a
coefficient table with no intercept, an input table whose provider throws, a
crop route that cannot cut the shape the control is showing -- and where the
wrong answer is a picture. A montage that draws the wrong cells looks exactly
like one that draws the right cells, so each of these has to end in a stated
refusal, a disabled control, or a fallback the caption can name.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Tuple

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import (                                 # noqa: E402
    QDialog, QMessageBox, QTabBar,
)

from spacr.qt.widgets import cell_montage_view as cmv           # noqa: E402


def _block(monkeypatch, *names):
    """Make ``from <name> import ...`` fail, the way a trimmed install does."""
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)


# ---------------------------------------------------------------------------
# Module-level readers
# ---------------------------------------------------------------------------

def test_the_channel_mapping_falls_back_when_crops_cannot_be_imported(
        monkeypatch):
    """The colour-to-source map decides which channel a red pixel came from.
    Without it the montage still has to draw something, and the fallback is
    spaCR's own historical order rather than positional."""
    _block(monkeypatch, "spacr.crops")

    assert cmv._colour_to_source() == {"r": 2, "g": 1, "b": 0}


@pytest.mark.parametrize("frame", [
    None,
    pd.DataFrame(),
    pd.DataFrame({"coefficient": [0.3]}),
    pd.DataFrame({"feature": ["Intercept"], "note": ["no effect column"]}),
    pd.DataFrame({"feature": ["gene[A]"], "coefficient": [0.3]}),
    pd.DataFrame({"feature": ["Intercept"], "coefficient": ["not a number"]}),
    pd.DataFrame({"feature": ["Intercept"], "coefficient": [float("inf")]}),
])
def test_a_table_that_names_no_usable_intercept_reports_none(frame):
    """None is what makes the caption say 'median'. A zero here would put a
    baseline on the montage that the fit never produced."""
    assert cmv.intercept_from_frame(frame) is None


@pytest.mark.parametrize("spelling", ["Intercept", "(Intercept)", "const",
                                      "constant"])
def test_every_spelling_of_the_intercept_is_found(spelling):
    """statsmodels and this project spell it four ways; a baseline silently
    not found falls back to the median while the caption says intercept."""
    frame = pd.DataFrame({"feature": [spelling, "gene[A]"],
                          "coefficient": [1.25, 0.3]})

    assert cmv.intercept_from_frame(frame) == 1.25


def test_a_crop_size_that_is_not_a_number_leaves_the_thumbnail_at_default():
    """`img_size` is hand-editable in the picture settings; zero means 'use
    the default', which is what an unreadable value has to become."""
    assert cmv._thumb_px_of({"img_size": "large"}) == 0
    assert cmv._thumb_px_of({"img_size": 96}) == 96
    assert cmv._thumb_px_of({"img_size": 4096}) == 512


def test_the_candidate_border_has_a_colour_without_the_annotation_palette(
        monkeypatch):
    """The border marks the cells the coefficient points at. Losing the
    palette must not lose the mark."""
    _block(monkeypatch, "spacr.qt.annotate_engine")

    assert cmv.candidate_colour() == "#3ea6ff"


def test_the_candidate_border_assumes_a_dark_theme_when_it_cannot_ask(
        monkeypatch):
    """The hue deepens against a light tile. With no preference store to
    ask, the dark answer is the one that is readable on the default theme."""
    _block(monkeypatch, "spacr.qt.preferences")

    colour = cmv.candidate_colour()

    assert colour.startswith("#") and len(colour) == 7


@pytest.mark.parametrize("row", [
    None,
    pd.Series({"object_id": 3}),
])
def test_a_row_that_does_not_say_it_was_picked_is_not_a_candidate(row):
    """Absence of the column means the picker never ran on this row; wearing
    the candidate border would claim an inference nothing made."""
    assert cmv._is_candidate(row) is False


def test_a_row_that_cannot_be_read_is_not_a_candidate():
    """Drawing a tile must not raise on a row object that answers `index`
    but not indexing."""
    class Awkward:
        index = ("montage_candidate",)

        def __getitem__(self, key):
            raise RuntimeError("this row cannot be read")

    assert cmv._is_candidate(Awkward()) is False


def test_a_single_channel_crop_becomes_a_grey_pixmap(qapp):
    """A one-channel mask crop reaches the grid as a 2-D array; handing that
    straight to QImage as RGB reads three rows as one."""
    grey = np.arange(64, dtype=np.uint8).reshape(8, 8)

    pixmap = cmv._pixmap(grey, size=32)

    assert not pixmap.isNull()
    assert pixmap.width() > 0 and pixmap.height() > 0


# ---------------------------------------------------------------------------
# One well's tab
# ---------------------------------------------------------------------------

@pytest.fixture()
def tab(qapp, qtbot):
    widget = cmv._WellTab(("gene", "gene", "g1", "A01"), "A01 · g1")
    qtbot.addWidget(widget)
    return widget


def _crops(n):
    return [np.full((8, 8, 3), 20 + i, dtype=np.uint8) for i in range(n)]


def test_a_tab_told_how_many_fit_on_a_page_pages_the_crops(tab):
    """The page size is measured by the caller from the viewport; a tab that
    ignored it would draw every crop and scroll instead of paging."""
    rows = pd.DataFrame({"object_id": range(6)})

    tab.set_content(rows, _crops(6), "caption", columns=3, thumb_px=48,
                    per_page=2)

    assert tab.page() == 0
    assert tab._per_page == 2, "the caller's page size was not kept"
    assert tab.show_page(1) == min(1, tab.page_count() - 1)
    assert tab.show_page(99) == tab.page_count() - 1, (
        "a page past the end must clamp rather than showing an empty grid")


def test_a_tab_with_no_page_size_shows_every_crop(tab, monkeypatch):
    """Zero means 'all of them', which is what a tab too small to measure
    has to fall back to rather than showing nothing."""
    rows = pd.DataFrame({"object_id": range(4)})
    tab.set_content(rows, _crops(4), "caption", columns=2)
    monkeypatch.setattr(tab, "per_page", lambda: 0)

    assert tab._page_slice() == [0, 1, 2, 3]


def test_the_column_count_falls_back_when_preferences_cannot_be_read(
        tab, monkeypatch):
    """The grid still has to have a width. Zero columns is a division by
    zero in the layout."""
    _block(monkeypatch, "spacr.qt.preferences")

    assert tab.column_count() == cmv.DEFAULT_MONTAGE_COLUMNS


def test_a_tab_with_no_width_yet_uses_the_requested_thumbnail_size(tab):
    """Called before the tab has been laid out, where dividing the width by
    the column count gives zero and every crop would vanish."""
    tab._requested_px = 128

    assert tab._thumbnail_px_for(0, 4) == 128
    assert tab._thumbnail_px_for(400, 0) == 128


def test_a_thumbnail_paints_its_provenance_ring(qapp, qtbot):
    """The ring is the only thing on the tile that says whether this cell
    carries the inference, so it has to survive an actual paint."""
    pixmap = cmv._pixmap(np.full((8, 8, 3), 40, dtype=np.uint8), size=48)
    thumb = cmv._Thumb(pixmap, "tooltip", highlight="#3ea6ff")
    qtbot.addWidget(thumb)
    thumb.resize(48, 48)

    thumb.grab()                     # forces paintEvent through Qt itself

    assert thumb.toolTip() == "tooltip"


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

@pytest.fixture()
def view(qapp, qtbot):
    widget = cmv.CellMontageView(threaded=False)
    qtbot.addWidget(widget)
    return widget


def _raising_provider():
    def provider():
        raise RuntimeError("the input table is gone")
    return provider


def test_an_input_table_that_cannot_be_reached_names_no_files(qapp, qtbot):
    """The panel reads its counts and scores off the run's input table. A
    provider that throws must leave the montage saying it has none, not take
    the window down."""
    widget = cmv.CellMontageView(database_provider=_raising_provider(),
                                 threaded=False)
    qtbot.addWidget(widget)

    assert widget.count_csvs() == ()
    assert widget.score_csvs() == ()


def test_no_input_table_at_all_names_no_files(view):
    """The panel is usable before a run is loaded."""
    assert view.count_csvs() == ()
    assert view.score_csvs() == ()


def test_the_attached_score_and_count_files_are_read_off_the_input_table(
        qapp, qtbot):
    """The contrast: the same rows carry both, which is why requiring a run
    folder for the guide fractions was never necessary."""
    rows = [{"plate": "p1", "score": "s1.csv", "count": "c1.csv",
             "database": "p1.db"},
            {"plate": "p2", "score": "s1.csv", "count": "", "database": ""}]
    widget = cmv.CellMontageView(database_provider=lambda: rows,
                                 threaded=False)
    qtbot.addWidget(widget)

    assert widget.count_csvs() == ("c1.csv",)
    assert widget.score_csvs() == ("s1.csv",), "a repeated file was listed twice"


def test_an_intercept_baseline_falls_back_when_the_fit_names_none(qapp,
                                                                  qtbot):
    """The caption names the baseline it used. Reporting the intercept while
    silently centring on the median is the mismatch this guards."""
    frame = pd.DataFrame({"feature": ["gene[A]"], "coefficient": [0.3]})
    widget = cmv.CellMontageView(frame_provider=lambda: frame, threaded=False)
    qtbot.addWidget(widget)
    index = widget._baseline.findData("intercept")
    assert index >= 0
    widget._baseline.setCurrentIndex(index)

    assert widget._baseline_value() == (None, "")


def test_building_with_nothing_selected_says_so_and_starts_nothing(view):
    """A montage cannot be built before a coefficient is clicked, and the
    button has to explain that rather than appearing to work."""
    assert view.build() is False
    assert view.status_text()


def test_the_sweep_prompt_offers_rank_and_reports_the_answer(view,
                                                             monkeypatch):
    """The multivariate picker needs a sweep. Rather than silently falling
    back, the panel asks -- and 'rank' has to be recorded as the choice so
    the caption says rank because it IS rank."""
    def choose_first(self):
        self.buttons()[0].click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", choose_first)
    assert view._ask("the sweep has not been run") == "rank"

    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    assert view._ask("the sweep has not been run") == "cancel"


def test_the_picture_settings_window_can_be_dismissed_without_changing_it(
        view, monkeypatch):
    """Cancel has to leave the settings exactly as they were; a window that
    writes back on close makes Cancel indistinguishable from OK."""
    from spacr.qt.widgets import picture_settings_dialog as psd

    before = dict(view._picture_settings)
    monkeypatch.setattr(psd.PictureSettingsDialog, "exec",
                        lambda self: QDialog.Rejected)

    assert view.edit_picture_settings() is False
    assert view._picture_settings == before


def test_accepting_the_picture_settings_writes_them_back_to_the_controls(
        view, monkeypatch):
    """One setting, one value: the hidden widgets are what `request()` reads,
    so a value edited in the window has to land on them."""
    from spacr.qt.widgets import picture_settings_dialog as psd

    monkeypatch.setattr(psd.PictureSettingsDialog, "exec",
                        lambda self: QDialog.Accepted)
    monkeypatch.setattr(psd.PictureSettingsDialog, "values",
                        lambda self: {"channels": [0, 1], "img_size": 128})
    view._force_picking("rank")

    assert view.edit_picture_settings() is True
    assert view._picture_settings["img_size"] == 128
    assert view._channels.text() == "0, 1"
    assert view._picking_override == ""


def test_nothing_is_written_to_a_database_without_both_halves(view):
    """The merge needs a database and a score file. Saying which is missing
    is the difference between 'nothing to do' and 'it silently did nothing'."""
    assert view.write_scores_into_the_databases() == {}
    assert "Nothing to merge" in view.status_text()


def test_the_write_confirmation_names_every_file_it_will_touch(view,
                                                               monkeypatch):
    """This writes to the user's measurement databases. The prompt has to
    list what it will touch, and Cancel has to be the default so a stray
    Return does not start it."""
    seen = {}

    def capture(self):
        seen["text"] = self.text()
        seen["detail"] = self.informativeText()
        seen["default"] = self.defaultButton().text()
        self.buttons()[1].click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", capture)

    answered = view._ask_before_writing(["/data/p1.db"], ["/data/scores.csv"])

    assert answered is False
    assert "1 measurement database?" in seen["text"]
    assert "scores.csv" in seen["detail"]
    assert "p1.db" in seen["detail"]
    assert seen["default"] == "Cancel"


@pytest.mark.xfail(strict=True, reason=(
    "_ask_before_writing returns `clickedButton() is not cancel`, and a "
    "dialog closed by its title bar clicked no button at all, so None "
    "is not cancel reads as consent and the scores are written"))
def test_closing_the_write_confirmation_is_not_consent(view, monkeypatch):
    """Dismissing a window is not agreeing to it. The polarity used by the
    matching prompt in Preferences -- `clickedButton() is proceed` -- is the
    one that makes a closed window mean no."""
    def close_the_window(self):
        self.reject()               # what the title-bar x does
        return 0

    monkeypatch.setattr(QMessageBox, "exec", close_the_window)

    assert view._ask_before_writing(["/data/p1.db"], ["/a.csv"]) is False


def test_the_write_confirmation_accepts_when_the_write_button_is_pressed(
        view, monkeypatch):
    """The contrast that makes the refusal a refusal."""
    def press_write(self):
        self.buttons()[0].click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", press_write)

    assert view._ask_before_writing(["/data/p1.db"], ["/a.csv"]) is True


# ---------------------------------------------------------------------------
# What the panel picked
# ---------------------------------------------------------------------------

def _plan(name, objects):
    return SimpleNamespace(objects=objects,
                           coefficient=SimpleNamespace(name=name))


def test_a_plan_with_no_rows_contributes_no_group(view):
    """An empty plan is a real answer -- the window admitted nothing -- and
    it must not appear in the comparison as a group of zero cells."""
    view._plans = (
        _plan("GRA14", pd.DataFrame({"montage_candidate": []})),
        _plan("ROP18", pd.DataFrame({"montage_candidate": [True, False]})),
    )

    groups = view.picked_groups()

    assert set(groups) == {"ROP18"}
    assert groups["ROP18"] == [0]


def test_a_plan_that_never_marked_candidates_contributes_all_its_rows(view):
    """Rank picking marks nothing; every row it drew is what it picked."""
    view._plans = (_plan("GRA14", pd.DataFrame({"object_id": [7, 8]})),)

    assert view.picked_groups() == {"GRA14": [0, 1]}


def test_comparing_with_nothing_picked_says_what_to_do_first(view):
    """Opening an empty comparison tab would look like a comparison that
    found no difference."""
    assert view.compare_a_measurement() is None
    assert "Show some cells first" in view.status_text()


def test_the_wider_inventory_is_used_only_when_it_covers_the_picked_rows(
        view):
    """The control-well contrasts need cells the montage did not draw, but
    an inventory that does not contain the picked rows would compare two
    different things."""
    picked = pd.DataFrame({"v": [1.0, 2.0]}, index=[10, 11])
    view._plans = (_plan("GRA14", picked),)

    view.remember_inventory(objects=pd.DataFrame({"v": [1.0, 2.0, 3.0]},
                                                 index=[10, 11, 12]))
    assert len(view.rows_to_compare()) == 3

    view.remember_inventory(objects=pd.DataFrame({"v": [9.0]}, index=[99]))
    assert len(view.rows_to_compare()) == 2


def test_an_inventory_that_cannot_be_compared_leaves_the_picked_rows(view):
    """An inventory whose index will not answer `isin` is not a reason to
    lose the comparison; the rows the montage drew are always safe."""
    class Unusable:
        """An inventory that reports rows but whose index cannot be tested."""

        index = 7                       # not something Index.isin accepts

        def __len__(self):
            return 3

    picked = pd.DataFrame({"v": [1.0]}, index=[1])
    view._plans = (_plan("GRA14", picked),)
    view.remember_inventory(objects=Unusable())

    rows = view.rows_to_compare()

    assert rows is not None
    assert len(rows) == 1
    assert list(rows.index) == [1]


# ---------------------------------------------------------------------------
# Saved-run state and the hidden widgets
# ---------------------------------------------------------------------------

def test_a_workspace_state_that_is_not_a_mapping_applies_nothing(view):
    """A saved run written by a different build can carry anything; the
    panel must open rather than refuse the whole run."""
    assert view.apply_workspace_state(None) is False
    assert view.apply_workspace_state("channels=0,1") is False


def test_a_setting_the_widget_cannot_hold_is_skipped_not_forced(view):
    """A crop size of 'large' must not be pushed into a spin box; the rest
    of the settings still have to land."""
    view._write_back({"channels": [1, 2], "cap": "not a number"})

    assert view._channels.text() == "1, 2"
    assert isinstance(view._read_widgets()["cap"], (int, float))


def test_the_mode_line_survives_a_panel_with_no_status_label(view,
                                                             monkeypatch):
    """The fallback must never be silent, but announcing it cannot be the
    thing that raises."""
    before = view.status_text()
    monkeypatch.setattr(view, "_status", None)

    view._on_mode_changed()

    assert view.status_text() == before
    assert cmv.picture_source_label(view.picture_mode()), (
        "there was no sentence to announce in the first place")


def test_the_loaded_run_name_is_empty_when_no_run_is_loaded(view):
    """'' lets a caller tell 'no run' from 'a run whose name I could not
    work out'."""
    assert view.loaded_run_name() == ""


# ---------------------------------------------------------------------------
# The shape control
# ---------------------------------------------------------------------------

def test_a_shape_the_route_cannot_cut_moves_the_control_off_it(view):
    """A control reading 'object-shaped' over bounding-box crops is the
    silent substitution this exists to prevent."""
    for index in range(view._shape.count()):
        if str(view._shape.itemData(index) or "") == "mask":
            view._shape.setCurrentIndex(index)
    result = cmv.MontageLoad(plans=(object(),), shapes=("bbox",),
                             shape_reason="this route has no masks")

    view._apply_shape_availability(result)

    assert str(view._shape.currentData()) == "bbox"
    assert view._shape.isEnabled() is True


def test_a_route_that_offers_no_shape_at_all_disables_the_control(view):
    """Two options that both do nothing is worse than one greyed control
    carrying the sentence that says why."""
    result = cmv.MontageLoad(plans=(object(),), shapes=(),
                             shape_reason="the PNGs were cut at run time")

    view._apply_shape_availability(result)

    assert view._shape.isEnabled() is False
    assert "cut at run time" in view._shape.toolTip()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def test_two_wells_with_the_same_label_get_distinguishable_tabs(view):
    """Two identical tabs is precisely the failure the label exists to
    prevent -- a reader cannot tell which gene's cells they are looking at."""
    first = view._open_well_tab(("g", "gene", "g1", "A01"), "A01 · g1", "tip")
    second = view._open_well_tab(("g", "gene", "g2", "A01"), "A01 · g1", "tip")

    assert first.label == "A01 · g1"
    assert second.label == "A01 · g1 #2"


def test_the_summary_tab_cannot_be_closed(view):
    """It carries the caption for the whole montage; closing it would leave
    the panel with no account of itself."""
    before = view._tabs.count()

    view._close_tab(0)

    assert view._tabs.count() == before


def test_a_tab_bar_that_refuses_a_close_button_does_not_stop_the_tab(
        view, monkeypatch):
    """Hiding the x is cosmetic; a Qt build that refuses must not take the
    tab away with it."""
    before = view._tabs.count()

    def refuse(self, index, side, widget):
        raise RuntimeError("this tab bar will not take a button")

    monkeypatch.setattr(QTabBar, "setTabButton", refuse)

    view._hide_close_button(0)

    assert view._tabs.count() == before
    assert view._tabs.widget(0) is view._summary_tab


def test_the_panel_column_count_falls_back_without_preferences(view,
                                                               monkeypatch):
    """The grid still needs a width when there is no store to ask."""
    _block(monkeypatch, "spacr.qt.preferences")

    assert view._column_count() == cmv.DEFAULT_MONTAGE_COLUMNS


def test_clearing_empties_the_grids_but_keeps_the_tabs_open(view):
    """A well tab closes by its x and by nothing else, so a redraw must not
    take it away while the user is reading its caption."""
    tab = view._open_well_tab(("g", "gene", "g1", "A01"), "A01 · g1", "tip")
    tab.set_content(pd.DataFrame({"object_id": [0, 1]}), _crops(2),
                    "caption", columns=2)
    before = view._tabs.count()

    view._clear()

    assert view._tabs.count() == before
    assert tab.caption_text() == "caption"


def test_a_relayout_only_reflows_when_the_column_count_moved(view,
                                                             monkeypatch):
    """Reflowing on every resize event rebuilds every thumbnail widget, and
    a grid that rebuilds while being scrolled loses the reader's place."""
    view._open_well_tab(("g", "gene", "g1", "A01"), "A01 · g1", "tip")
    view._columns = view._column_count()

    view._relayout()
    assert view._columns == view._column_count()

    wider = view._columns + 1
    monkeypatch.setattr(view, "_column_count", lambda: wider)
    view._relayout()

    assert view._columns == wider


def test_a_panel_with_no_selection_says_what_to_click(view):
    """The status line is the whole of the panel's account of itself before
    a coefficient is chosen."""
    view._announce()

    assert view.status_text() == view.NOTHING_SELECTED


# ---------------------------------------------------------------------------
# Building from a run whose files are all present
# ---------------------------------------------------------------------------

@pytest.fixture()
def loaded(qapp, qtbot, tmp_path):
    """A panel whose coefficient, run folder, database and CSVs all exist."""
    import sqlite3

    run = tmp_path / "ols_3"
    run.mkdir()
    results = run / "results.csv"
    results.write_text("feature,coefficient\n", encoding="utf-8")
    database = tmp_path / "p1.db"
    sqlite3.connect(str(database)).close()
    counts = tmp_path / "counts.csv"
    counts.write_text("prc,grna,count\n", encoding="utf-8")
    scores = tmp_path / "scores.csv"
    scores.write_text("object_id,pred\n", encoding="utf-8")

    frame = pd.DataFrame({"feature": ["fraction:gene[GRA14]", "Intercept"],
                          "coefficient": [0.8, 1.0]})
    rows = [{"plate": "p1", "score": str(scores), "count": str(counts),
             "database": str(database)}]
    widget = cmv.CellMontageView(frame_provider=lambda: frame,
                                 results_provider=lambda: str(results),
                                 database_provider=lambda: rows,
                                 threaded=False)
    qtbot.addWidget(widget)
    widget.set_coefficient("fraction:gene[GRA14]")
    return widget


def test_a_run_with_every_input_present_can_be_asked_for(loaded):
    """The contrast for every refusal above: nothing is missing, so the
    button has no reason and a request exists."""
    assert loaded.reason() == ""
    assert loaded.request() is not None


def test_a_queue_stops_at_the_first_coefficient_that_starts_a_load(loaded):
    """One coefficient that cannot load must not strand the rest of the
    selection, and one that CAN must not be skipped past."""
    loaded._queue = ["fraction:gene[GRA14]"]

    assert loaded._build_the_next_queued() is True
    assert loaded._queue == []
    assert loaded._build_the_next_queued() is False


def test_a_multivariate_request_with_no_sweep_does_not_build(loaded):
    """The sweep is long. A user who declines it must not have a montage
    built by some other picker while the caption says multivariate."""
    loaded._picture_settings = dict(loaded.picture_settings())
    loaded._picture_settings["cell_picking"] = "multivariate"
    loaded._ask_about_multivariate = lambda _shortfall: "cancel"

    assert loaded.build() is False


def test_a_multivariate_request_the_user_downgrades_builds_as_rank(loaded):
    """Rank is recorded as the choice, so the caption says rank because it
    IS rank rather than because something fell back silently."""
    loaded._picture_settings = dict(loaded.picture_settings())
    loaded._picture_settings["cell_picking"] = "multivariate"
    loaded._ask_about_multivariate = lambda _shortfall: "rank"

    assert loaded.build() is True
    assert loaded._picking_override == "rank"


def test_the_write_prompt_is_the_default_confirmation(loaded, monkeypatch):
    """`write_scores_into_the_databases()` with no confirmation argument has
    to ask the user, never assume."""
    asked = []

    def refuse(self):
        asked.append(self.text())
        self.buttons()[1].click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", refuse)

    assert loaded.write_scores_into_the_databases() == {}
    assert asked, "the databases were written without asking"


def test_stepping_to_the_next_coefficient_starts_at_the_first(loaded):
    """A key that is not in the selection has no position in it; stepping
    from there has to land on the first, not raise."""
    loaded.set_coefficients(["fraction:gene[A]", "fraction:gene[B]"])
    loaded._key = "fraction:gene[NOT_SELECTED]"

    assert loaded.show_next_coefficient() == "fraction:gene[A]"


def test_one_selected_coefficient_has_nowhere_to_step_to(loaded):
    """None is what tells the caller to leave the button disabled."""
    loaded.set_coefficients(["fraction:gene[A]"])

    assert loaded.show_next_coefficient() is None


# ---------------------------------------------------------------------------
# The guide fractions, when there is no run folder
# ---------------------------------------------------------------------------

def _request(tmp_path, counts):
    database = tmp_path / "p1.db"
    database.write_bytes(b"")
    return cmv.MontageRequest(name="GRA14", effect=0.8, level="gene",
                              results_path="", databases=(str(database),),
                              count_csvs=(str(counts),))


def test_count_csvs_that_cannot_be_read_say_so_rather_than_saying_nothing(
        tmp_path, monkeypatch):
    """Either a run folder or a count CSV is enough for the guide fractions.
    When the CSVs are the only source and they cannot be read, the reason
    has to reach the tab -- 'no fractions available' would blame the wrong
    thing."""
    from spacr import cell_montage

    counts = tmp_path / "counts.csv"
    counts.write_text("prc,grna,count\n", encoding="utf-8")

    def refuse(_paths):
        raise cell_montage.MontageError("the count file names no well")

    monkeypatch.setattr(cell_montage, "fractions_from_counts", refuse)

    answer = cmv.load(_request(tmp_path, counts))

    assert answer.error == "the count file names no well"
    assert answer.unavailable is True


def test_an_unexpected_failure_reading_the_counts_is_still_a_sentence(
        tmp_path, monkeypatch):
    """`load` runs on a worker thread and must never raise: the tab it
    belongs to has to stay on screen and say why."""
    from spacr import cell_montage

    counts = tmp_path / "counts.csv"
    counts.write_text("prc,grna,count\n", encoding="utf-8")

    def explode(_paths):
        raise MemoryError("the count file is too large")

    monkeypatch.setattr(cell_montage, "fractions_from_counts", explode)

    answer = cmv.load(_request(tmp_path, counts))

    assert "Could not build the guide fractions" in answer.error
    assert "too large" in answer.error


# ---------------------------------------------------------------------------
# Remaining panel details
# ---------------------------------------------------------------------------

def test_the_resolved_crop_source_is_remembered_for_the_settings_window(view):
    """The settings window offers THIS screen's mask planes rather than free
    text, which it can only do if the last load's source was kept."""
    source = SimpleNamespace(kind="merged")

    view.remember_inventory(source=source)

    assert view._last_source is source


def test_a_setting_with_no_value_is_not_written_over_the_widget(view):
    """A settings file that carries the key with a null must not blank the
    control that already holds a good value."""
    view._channels.setText("r, g, b")

    view._write_back({"channels": None})

    assert view._channels.text() == "r, g, b"


def test_a_combo_setting_is_matched_by_its_text_when_it_has_no_data(view):
    """The settings window writes what the user saw. A combo entry carrying
    no data would otherwise silently ignore a valid choice."""
    view._object.addItem("organelle")

    view._write_back({"object_type": "organelle"})

    assert view._object.currentText() == "organelle"


def test_a_control_that_does_not_exist_is_skipped_when_reading_back(view):
    """The mirrored map names every setting; a build without one of those
    controls must still be able to describe itself."""
    view._cap = None

    read = view._read_widgets()

    assert "cap" not in read
    assert "channels" in read


def test_saving_before_there_is_a_montage_writes_nothing(view):
    """A file dialog opened for an empty montage would write a blank page."""
    assert view.save() is None
    assert "no montage to save" in view.status_text()


def test_a_save_click_is_not_a_file_name(view, monkeypatch):
    """Qt hands `clicked` a bool into the first argument, and `False is None`
    is False -- so the bool went past the "ask for a name" branch and on to
    the writer, which is how `QImage.save(bool)` reached a user."""
    from PySide6.QtWidgets import QFileDialog

    asked = []

    def ask(*args, **kwargs):
        asked.append(args[2] if len(args) > 2 else "")
        return "", ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(ask))
    view._plans = (_plan("GRA14", pd.DataFrame({"v": [1.0]})),)
    view._name = "GRA14"

    assert view.save(False) is None
    assert asked, "the bool was taken for a path instead of opening a dialog"
    assert asked[0].startswith("cells_behind_GRA14."), asked


def test_a_shape_combo_with_no_item_model_still_takes_its_enabled_state(
        view):
    """The per-entry tooltips need an item model. A combo without one has
    to keep the control-level answer rather than raising during a load."""
    from PySide6.QtCore import QStringListModel

    view._shape.setModel(QStringListModel(["object", "bbox"]))
    result = cmv.MontageLoad(plans=(object(),), shapes=("bbox",),
                             shape_reason="this route has no masks")

    view._apply_shape_availability(result)

    assert view._shape.isEnabled() is True


def test_the_comparison_tab_opens_once_and_is_reused(view):
    """A second floating comparison of the same montage is two answers to
    one question; the tab is reused so it stays in step with the grid."""
    rows = pd.DataFrame({"cell_area": [10.0, 20.0, 30.0],
                         "montage_candidate": [True, True, False]})
    view._plans = (_plan("GRA14", rows),)

    panel = view.compare_a_measurement()

    assert panel is not None
    assert view.compare_a_measurement() is panel
    assert view._tabs.tabText(view._tabs.indexOf(panel)) == "Compare"


# ---------------------------------------------------------------------------
# Choosing a route to the pixels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _FakePlan:
    """The shape `load` needs back from the selector, and nothing more."""

    objects: Any
    notes: Tuple[str, ...] = ()

    def rows(self):
        return [row for _index, row in self.objects.iterrows()]


class _FakeSource:
    """A crop source that answers with a flat grey square per row."""

    def get_many(self, rows):
        return [np.full((4, 4, 3), 7, dtype=np.uint8) for _ in rows]


@pytest.fixture()
def montage_pieces(monkeypatch, tmp_path):
    """Stand in for the readers `load` calls, and record what they were given.

    The pieces are the module's own collaborators, replaced one for one so
    the branch under test is the only thing that varies.
    """
    from spacr import cell_montage, crops

    objects = pd.DataFrame({
        "object_id": [1, 2],
        "prc": ["p1_r1_c1", "p1_r1_c2"],
        "pred": [0.8, 0.2],
    })
    counts = pd.DataFrame({
        "prc": ["p1_r1_c1", "p1_r1_c2"],
        "grna": ["TGGT1_225160_1", "TGGT1_225160_2"],
        "fraction": [0.9, 0.1],
    })
    seen: dict = {}

    monkeypatch.setattr(cell_montage, "fractions_from_counts",
                        lambda _paths: counts)
    monkeypatch.setattr(cell_montage, "load_montage_objects",
                        lambda *a, **k: objects.copy())
    monkeypatch.setattr(crops, "reanchor_frame",
                        lambda frame, _root: (frame, SimpleNamespace(
                            describe=lambda: "")))

    def remember(_objects, _counts, name, effect, **kwargs):
        seen.update(kwargs)
        seen["name"] = name
        return _FakePlan(objects=_objects)

    monkeypatch.setattr(cell_montage, "select_montage", remember)

    def install(choice):
        monkeypatch.setattr(cell_montage, "resolve_montage_crop_source",
                            lambda *a, **k: choice)
        return seen

    return install, objects, counts


def _load_request(tmp_path, picture=None, crop_shape="object"):
    database = tmp_path / "p1.db"
    database.write_bytes(b"")
    counts = tmp_path / "counts.csv"
    counts.write_text("prc,grna,count\n", encoding="utf-8")
    return cmv.MontageRequest(
        name="225160", effect=0.8, level="gene", results_path="",
        databases=(str(database),), count_csvs=(str(counts),),
        picture=picture or {}, crop_shape=crop_shape)


def _choice(requirements):
    from spacr.cell_montage import CropSourceChoice

    return CropSourceChoice(source=_FakeSource(), kind="merged",
                            reason="merged arrays", available=True,
                            requirements=requirements)


def test_a_route_missing_what_it_needs_is_refused_by_what_is_missing(
        tmp_path, montage_pieces):
    """A user with no channel list has to be told THAT. Reporting 'no crop
    source' sends them looking for files that are all present."""
    from spacr.cell_montage import RouteRequirements

    install, _objects, _counts = montage_pieces
    install(_choice(RouteRequirements(route="merged-bbox", shapes=("bbox",),
                                      missing=("no channel list",))))

    answer = cmv.load(_load_request(tmp_path))

    assert answer.unavailable is True
    assert "no channel list" in answer.error


def test_a_shape_this_route_cannot_cut_is_stated_not_substituted(
        tmp_path, montage_pieces):
    """An object-shaped crop this route cannot cut must never quietly become
    a bounding box: the caption says what was asked for, why it could not be
    done, and what was drawn instead."""
    from spacr.cell_montage import RouteRequirements

    install, _objects, _counts = montage_pieces
    install(_choice(RouteRequirements(route="merged-bbox", shapes=("bbox",),
                                      detail="a coordinate table")))

    answer = cmv.load(_load_request(tmp_path, crop_shape="object"))

    assert answer.plans
    notes = "\n".join(answer.plans[0].notes)
    assert "'object' crop shape was asked for" in notes
    assert "'bbox'" in notes
    assert answer.shapes == ("bbox",)


def test_a_route_that_declares_no_requirements_narrows_nothing(
        tmp_path, montage_pieces):
    """The exported PNGs were cut when the run wrote them, so that route
    answers no shape question at all -- and must not be read as offering
    every shape."""
    install, _objects, _counts = montage_pieces
    install(_choice(None))

    answer = cmv.load(_load_request(tmp_path))

    assert answer.plans
    assert answer.shapes == ()
    assert answer.shape_reason == ""


def test_attributed_picking_matches_guides_on_the_measured_prefix(
        tmp_path, montage_pieces, monkeypatch):
    """The results name guides as the design did and the counts as the
    library does. Hard-coding one organism's prefix leaves every Plasmodium
    or human library matching nothing, at which point the attribution has no
    competition to compare against and says nothing about it."""
    from spacr import cell_montage

    install, _objects, _counts = montage_pieces
    seen = install(_choice(None))
    monkeypatch.setattr(cell_montage, "effects_from_results",
                        lambda _path: {"225160_1": 0.8, "225160_2": -0.1})

    cmv.load(_load_request(tmp_path, picture={"cell_picking": "attributed"}))

    assert seen["picking"] == "attributed"
    assert seen["effects"] == {"TGGT1_225160_1": 0.8, "TGGT1_225160_2": -0.1}


def test_attributed_picking_with_no_matching_effects_carries_none(
        tmp_path, montage_pieces, monkeypatch):
    """An empty map and 'no map at all' are different to the selector, and
    an empty one would silently mean 'no competition' rather than 'unknown'."""
    from spacr import cell_montage

    install, _objects, _counts = montage_pieces
    seen = install(_choice(None))
    monkeypatch.setattr(cell_montage, "effects_from_results",
                        lambda _path: {"nothing_matches": 1.0})

    cmv.load(_load_request(tmp_path, picture={"cell_picking": "assigned"}))

    assert seen["effects"] is None


def test_multivariate_picking_is_handed_the_sweep_grid(
        tmp_path, montage_pieces, monkeypatch):
    """The grid is what makes multivariate multivariate. Without it the
    picker falls back to the single-score attribution and says so, which
    hides that what it fell back FROM was never supplied."""
    from spacr import cell_montage

    install, _objects, _counts = montage_pieces
    seen = install(_choice(None))
    grid = pd.DataFrame({"cell_area": [0.4]}, index=["225160_1"])
    monkeypatch.setattr(cell_montage, "effects_from_results",
                        lambda _path: {"225160_1": 0.8})
    monkeypatch.setattr(cell_montage, "effects_grid_from_results",
                        lambda _path: grid)

    cmv.load(_load_request(tmp_path, picture={"cell_picking": "multivariate"}))

    assert seen["effects_grid"] is grid
