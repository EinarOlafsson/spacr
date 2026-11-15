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
from types import SimpleNamespace

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
    """`crop_size` is hand-editable in the picture settings; zero means 'use
    the default', which is what an unreadable value has to become. It was
    `img_size` until 2026-09-19 (364)."""
    assert cmv._thumb_px_of({"crop_size": "large"}) == 0
    assert cmv._thumb_px_of({"crop_size": 96}) == 96
    assert cmv._thumb_px_of({"crop_size": 4096}) == 512


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
                        lambda self: {"channels": [0, 1], "crop_size": 128})
    view._force_picking("rank")

    assert view.edit_picture_settings() is True
    assert view._picture_settings["crop_size"] == 128
    assert view._channels.text() == "0, 1"
    assert view._picking_override == ""


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


def test_the_loaded_run_name_is_empty_when_no_run_is_loaded(view):
    """'' lets a caller tell 'no run' from 'a run whose name I could not
    work out'."""
    assert view.loaded_run_name() == ""


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
# Remaining panel details
# ---------------------------------------------------------------------------

def test_the_resolved_crop_source_is_remembered_for_the_settings_window(view):
    """The settings window offers THIS screen's mask planes rather than free
    text, which it can only do if the last load's source was kept."""
    source = SimpleNamespace(kind="merged")

    view.remember_inventory(source=source)

    assert view._last_source is source


def test_a_control_that_does_not_exist_is_skipped_when_reading_back(view):
    """The mirrored map names every setting; a build without one of those
    controls must still be able to describe itself."""
    view._cap = None

    read = view._read_widgets()

    assert "cap" not in read
    assert "channels" in read


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
