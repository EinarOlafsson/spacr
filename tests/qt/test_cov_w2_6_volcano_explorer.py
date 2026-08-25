"""Clicking, dropping, restyling and exporting a real volcano.

Every case drives the widget with a real results frame and the real
renderer: the point of the module is that one function draws both the
headless PDF and the widget, so a stubbed renderer would test nothing that
ships. Only the file dialogs -- which cannot be answered offscreen -- are
replaced, and only by making them return a path.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import QMimeData, QPoint, QUrl, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QFileDialog, QMessageBox

from spacr.qt.widgets.volcano_explorer import VolcanoExplorer
from spacr.volcano_style import VolcanoStyle


def _results(n: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "guide": [f"G{i:03d}" for i in range(n)],
        "gene": [f"TGGT1_{i // 3:06d}" for i in range(n)],
        "standardized_marginal_effect": rng.normal(size=n),
        "adjusted_p_value": rng.random(n) * 0.5 + 1e-6,
        "p_value": rng.random(n),
    })


@pytest.fixture
def explorer(qapp, qtbot):
    widget = VolcanoExplorer(_results())
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def annotation_csv(tmp_path):
    path = tmp_path / "annotations.csv"
    pd.DataFrame({
        "gene": [f"TGGT1_{i:06d}" for i in range(8)],
        "localisation": ["rhoptry", "micronemes"] * 4,
        "essentiality": np.linspace(0.0, 1.0, 8),
    }).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# results in, plot out
# --------------------------------------------------------------------------

def test_new_results_replace_the_plot_and_clear_the_selection(explorer):
    explorer.select_point(3)
    assert explorer.selected_index() == 3
    explorer.set_results(_results(10))
    assert explorer.selected_index() is None
    assert len(explorer.results()) == 10
    assert explorer._controls["x_column"].count() > 0


def test_the_results_handed_back_are_a_copy(explorer):
    taken = explorer.results()
    taken.loc[0, "guide"] = "EDITED"
    assert explorer.results().loc[0, "guide"] != "EDITED"


def test_a_new_style_redraws_and_moves_the_side_panel_with_it(explorer):
    """Two ways to change one setting that disagree about what it now is
    would be worse than having only one of them."""
    style = VolcanoStyle()
    style.title = "A different title"
    style.point_size = 44
    explorer.set_style(style)
    assert explorer.style() is style
    assert explorer._controls["title"].text() == "A different title"


def test_an_empty_result_set_draws_nothing_rather_than_crashing(qapp, qtbot):
    empty = VolcanoExplorer(pd.DataFrame())
    qtbot.addWidget(empty)
    empty.refresh()
    assert empty.nearest_point(0.0, 0.0) is None
    assert empty.export("pdf") is None


# --------------------------------------------------------------------------
# clicking is a lookup, not a hit test on pixels
# --------------------------------------------------------------------------

class _Event:
    def __init__(self, xdata, ydata, inaxes):
        self.xdata, self.ydata, self.inaxes = xdata, ydata, inaxes


def test_clicking_a_point_selects_the_row_it_belongs_to(explorer):
    explorer.refresh()
    axes = explorer._panels[0]
    frame = explorer.results()
    row = 5
    x = float(frame["standardized_marginal_effect"][row])
    y = float(-np.log10(frame["adjusted_p_value"][row]))
    seen = []
    explorer.point_selected.connect(seen.append)
    explorer._on_click(_Event(x, y, axes))
    assert explorer.selected_index() == row
    assert seen and seen[0]["guide"] == frame["guide"][row]
    assert explorer._detail_hint.text() == f"Selected {frame['guide'][row]}"


def test_a_click_off_the_axes_selects_nothing(explorer):
    explorer._on_click(_Event(0.0, 0.0, None))
    assert explorer.selected_index() is None


def test_a_click_on_empty_space_selects_nothing(explorer):
    """5% of the diagonal is about the radius of a marker; further than that
    is the background."""
    explorer.refresh()
    axes = explorer._panels[0]
    explorer._on_click(_Event(1e6, 1e6, axes))
    assert explorer.selected_index() is None


def test_a_click_where_no_point_has_a_finite_position_selects_nothing(
        qapp, qtbot):
    frame = _results(4)
    frame["standardized_marginal_effect"] = ["a", "b", "c", "d"]
    widget = VolcanoExplorer(frame)
    qtbot.addWidget(widget)
    assert widget.nearest_point(0.0, 0.0) is None


def test_the_nearest_point_is_normalised_by_each_axis_range(explorer):
    """Raw Euclidean distance picks the wrong point whenever the axes have
    different scales, which on a volcano they always do."""
    explorer.refresh()
    axes = explorer._panels[0]
    frame = explorer.results()
    row = 2
    x = float(frame["standardized_marginal_effect"][row])
    y = float(-np.log10(frame["adjusted_p_value"][row]))
    assert explorer.nearest_point(x, y, axes) == row
    # Without the axis ranges the same click is measured in raw data units.
    assert explorer.nearest_point(x, y, None) in (row, None)


# --------------------------------------------------------------------------
# the split axis
# --------------------------------------------------------------------------

def test_a_split_axis_gets_limits_derived_rather_than_refusing_to_draw(
        explorer):
    explorer._style.split_axis = True
    explorer._style.split_y_lims = None
    explorer.refresh()
    assert explorer._style.split_y_lims is not None
    (low_a, low_b), (high_a, high_b) = explorer._style.split_y_lims
    assert low_a == 0.0 and low_b < high_a < high_b


def test_no_y_values_means_no_split_to_suggest(qapp, qtbot):
    widget = VolcanoExplorer(_results(0))
    qtbot.addWidget(widget)
    assert widget._suggest_split() is None


def test_a_flat_y_axis_has_no_split_worth_making(qapp, qtbot):
    frame = _results(8)
    frame["adjusted_p_value"] = 0.5
    widget = VolcanoExplorer(frame)
    qtbot.addWidget(widget)
    assert widget._suggest_split() is None


def test_the_plotted_y_is_the_raw_column_when_the_log_is_off(explorer):
    explorer._style.y_neg_log10 = False
    assert np.allclose(explorer._plotted_y(),
                       explorer.results()["adjusted_p_value"].to_numpy(float))


def test_a_bad_style_draws_a_message_instead_of_crashing(explorer):
    explorer._style.x_column = "no_such_column"
    explorer.refresh()
    assert explorer._panels
    texts = [t.get_text() for t in explorer._panels[0].texts]
    assert any("Cannot draw this plot" in t for t in texts)


# --------------------------------------------------------------------------
# annotation merge
# --------------------------------------------------------------------------

def test_a_dropped_annotation_table_becomes_colour_and_shape_sources(
        explorer, annotation_csv):
    added = explorer.merge_annotation_file(annotation_csv)
    assert added == 2
    assert "localisation" in explorer.results().columns
    offered = [explorer._controls["color_by"].itemData(i)
               for i in range(explorer._controls["color_by"].count())]
    assert "localisation" in offered


def test_the_join_column_is_the_one_that_matches_the_most_rows(explorer,
                                                               tmp_path):
    """Not the first one that happens to share a name: a file keyed on gene
    and one keyed on guide both have to work without being asked which."""
    path = tmp_path / "both.csv"
    pd.DataFrame({
        "guide": ["nothing_matches"] * 8,
        "gene": [f"TGGT1_{i:06d}" for i in range(8)],
        "localisation": ["rhoptry"] * 8,
    }).to_csv(path, index=False)
    explorer.merge_annotation_file(path)
    merged = explorer.results()
    assert merged.loc[0, "localisation"] == "rhoptry"


def test_a_caller_may_name_the_join_column_itself(explorer, annotation_csv):
    assert explorer.merge_annotation_file(annotation_csv, on="gene") == 2


def test_an_empty_annotation_file_adds_nothing(explorer, tmp_path):
    path = tmp_path / "empty.csv"
    pd.DataFrame({"gene": []}).to_csv(path, index=False)
    assert explorer.merge_annotation_file(path) == 0


def test_a_file_that_brings_no_new_column_adds_nothing(explorer, tmp_path):
    path = tmp_path / "same.csv"
    explorer.results()[["gene"]].to_csv(path, index=False)
    assert explorer.merge_annotation_file(path) == 0


def test_a_file_sharing_no_column_says_what_it_could_have_joined_on(
        explorer, tmp_path):
    path = tmp_path / "unrelated.csv"
    pd.DataFrame({"orf": ["a"], "note": ["b"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="shares no column"):
        explorer.merge_annotation_file(path)


def test_an_excel_annotation_table_merges_too(explorer, tmp_path):
    path = tmp_path / "annotations.xlsx"
    pd.DataFrame({
        "gene": [f"TGGT1_{i:06d}" for i in range(8)],
        "localisation": ["rhoptry"] * 8,
    }).to_excel(path, index=False)
    assert explorer.merge_annotation_file(path) == 1


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def test_an_export_re_renders_at_print_size_rather_than_screenshotting(
        explorer, tmp_path):
    path = tmp_path / "volcano.pdf"
    written = explorer.export("pdf", str(path))
    assert written == str(path)
    assert path.stat().st_size > 1000


def test_an_export_adds_the_suffix_the_format_names(explorer, tmp_path):
    written = explorer.export("png", str(tmp_path / "volcano"))
    assert written.endswith(".png")
    assert os.path.exists(written)


def test_an_export_asks_where_to_put_it_when_nobody_said(explorer, tmp_path,
                                                         monkeypatch):
    target = str(tmp_path / "chosen.pdf")
    asked = {}

    def _ask(parent, caption, name, filters):
        asked["caption"] = caption
        asked["filters"] = filters
        return target, filters

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(_ask))
    assert explorer.export("pdf") == target
    assert "PDF" in asked["caption"] and "*.pdf" in asked["filters"]


def test_a_cancelled_export_writes_nothing(explorer, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    assert explorer.export("pdf") is None


def test_an_export_that_fails_says_so_rather_than_raising(explorer,
                                                          monkeypatch,
                                                          tmp_path):
    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *args: warned.append(args)))
    explorer._style.x_column = "no_such_column"
    assert explorer.export("pdf", str(tmp_path / "x.pdf")) is None
    assert warned and "Export failed" in warned[0][1]


# --------------------------------------------------------------------------
# the style is a value: saved, loaded, dropped
# --------------------------------------------------------------------------

def test_a_style_can_be_saved_and_loaded_back(explorer, tmp_path,
                                              monkeypatch):
    path = str(tmp_path / "style.json")
    explorer._style.title = "Saved title"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (path, "")))
    assert explorer._save_style() == path

    explorer._style.title = "Something else"
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (path, "")))
    assert explorer._load_style() == path
    assert explorer.style().title == "Saved title"


def test_cancelling_the_style_dialogs_changes_nothing(explorer, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    before = explorer.style().title
    assert explorer._save_style() is None
    assert explorer._load_style() is None
    assert explorer.style().title == before


def test_picking_an_annotation_file_reports_what_it_added(explorer,
                                                          annotation_csv,
                                                          monkeypatch):
    told = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(annotation_csv), "")))
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *args: told.append(args)))
    explorer._pick_annotation_file()
    assert told and "Added 2 columns" in told[0][2]


def test_cancelling_the_annotation_picker_merges_nothing(explorer,
                                                         monkeypatch):
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    before = list(explorer.results().columns)
    explorer._pick_annotation_file()
    assert list(explorer.results().columns) == before


def test_an_annotation_file_that_cannot_be_merged_says_why(explorer,
                                                           tmp_path,
                                                           monkeypatch):
    path = tmp_path / "unrelated.csv"
    pd.DataFrame({"orf": ["a"]}).to_csv(path, index=False)
    warned = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *args: warned.append(args)))
    explorer._pick_annotation_file()
    assert warned and "Could not merge annotations" in warned[0][1]


# --------------------------------------------------------------------------
# drag and drop
# --------------------------------------------------------------------------

def _urls(paths):
    """A live QMimeData carrying local file URLs.

    The caller HOLDS it: a Qt drag event keeps a bare pointer to the mime
    data, so letting the temporary die takes the interpreter with it.
    """
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return mime


def test_a_drag_carrying_files_is_accepted(explorer, annotation_csv):
    mime = _urls([annotation_csv])
    event = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, mime, Qt.LeftButton,
                            Qt.NoModifier)
    explorer.dragEnterEvent(event)
    assert event.isAccepted()


def test_a_drag_carrying_no_file_is_refused(explorer):
    mime = QMimeData()
    mime.setText("just some text")
    event = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, mime, Qt.LeftButton,
                            Qt.NoModifier)
    explorer.dragEnterEvent(event)
    assert not event.isAccepted()


def test_dropping_an_annotation_table_merges_it(explorer, annotation_csv):
    mime = _urls([annotation_csv])
    event = QDropEvent(QPoint(5, 5), Qt.CopyAction, mime,
                       Qt.LeftButton, Qt.NoModifier)
    explorer.dropEvent(event)
    assert event.isAccepted()
    assert "localisation" in explorer.results().columns


def test_dropping_a_style_file_restyles_the_plot(explorer, tmp_path):
    style = VolcanoStyle()
    style.title = "Dropped style"
    path = tmp_path / "style.json"
    style.save(str(path))
    mime = _urls([path])
    event = QDropEvent(QPoint(5, 5), Qt.CopyAction, mime,
                       Qt.LeftButton, Qt.NoModifier)
    explorer.dropEvent(event)
    assert explorer.style().title == "Dropped style"


def test_one_unmergeable_drop_does_not_cost_the_others(explorer, tmp_path,
                                                       annotation_csv):
    bad = tmp_path / "unrelated.csv"
    pd.DataFrame({"orf": ["a"]}).to_csv(bad, index=False)
    mime = _urls([bad, annotation_csv])
    event = QDropEvent(QPoint(5, 5), Qt.CopyAction, mime, Qt.LeftButton,
                       Qt.NoModifier)
    explorer.dropEvent(event)
    assert event.isAccepted()
    assert "localisation" in explorer.results().columns


def test_a_drop_with_nothing_local_in_it_is_ignored(explorer):
    mime = QMimeData()
    mime.setUrls([QUrl("https://example.org/volcano.csv")])
    event = QDropEvent(QPoint(5, 5), Qt.CopyAction, mime, Qt.LeftButton,
                       Qt.NoModifier)
    explorer.dropEvent(event)
    assert not event.isAccepted()


# --------------------------------------------------------------------------
# the right-click menu
# --------------------------------------------------------------------------

def test_the_style_menu_is_built_from_the_style_itself(explorer):
    """A style that gains a field gains a menu entry without anyone
    remembering to add one."""
    menu = explorer.build_style_menu()
    assert menu.actions()
    assert menu.toolTipsVisible()


def test_right_clicking_the_canvas_shows_that_menu_at_the_pointer(explorer,
                                                                  monkeypatch):
    shown = []

    class _Menu:
        def exec(self, position):
            shown.append(position)

    monkeypatch.setattr(explorer, "build_style_menu", _Menu)
    explorer._style_menu(QPoint(4, 4))
    assert shown == [explorer._canvas.mapToGlobal(QPoint(4, 4))]


def test_the_closed_set_fields_offer_the_columns_the_panel_offers(explorer):
    choices = explorer._style_choices()
    assert set(choices) <= {"x_column", "y_column", "color_by", "shape_by",
                            "colour_by", "label_by", "label_column"}
    assert "standardized_marginal_effect" in choices["x_column"]


@pytest.mark.xfail(strict=True, reason=(
    "`_style_choices` asks `_controls` for 'colour_by' and 'label_by', but "
    "the controls (and VolcanoStyle) spell them 'color_by' and "
    "'label_column'. So the right-click menu never gets the closed column "
    "list for those two and offers free text where the side panel offers a "
    "picker -- the two routes disagree about what the setting is."))
def test_colour_by_and_the_label_column_are_offered_as_closed_sets(explorer):
    choices = explorer._style_choices()
    assert "color_by" in choices
    assert "label_column" in choices
    assert "gene" in choices["color_by"]


def test_a_mapping_menu_that_is_not_on_this_panel_is_skipped(explorer):
    """The column menus are refilled by name, and a panel that does not
    build one of them must not cost the others their columns."""
    del explorer._controls["shape_by"]
    explorer.set_results(_results(6))
    offered = [explorer._controls["color_by"].itemData(i)
               for i in range(explorer._controls["color_by"].count())]
    assert "gene" in offered
