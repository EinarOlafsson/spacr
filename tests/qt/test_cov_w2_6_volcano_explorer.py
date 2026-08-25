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
    """Widened from a fixed list of five names to the panel itself.

    `_style_choices` used to be a hand-written list of field names, so this
    asserted that it stayed inside that list. It is now read off the panel --
    a control that IS a closed list is the definition of one -- which is what
    stops the menu offering free text where the panel offers a picker. The
    claim worth holding is therefore that every closed set is a real style
    field the panel closes, not that there are at most five of them.
    """
    import dataclasses

    from spacr.volcano_style import VolcanoStyle

    choices = explorer._style_choices()
    fields = {f.name for f in dataclasses.fields(VolcanoStyle)}
    assert set(choices) <= fields
    assert "standardized_marginal_effect" in choices["x_column"]
    # The pickers the panel has always had, which the menu never got.
    for closed in ("marker", "colormap", "line_style", "font_family"):
        assert closed in choices, closed


def test_colour_by_and_the_label_column_are_offered_as_closed_sets(explorer):
    """Was a strict xfail: `_style_choices` asked `_controls` for 'colour_by'
    and 'label_by' while the controls -- and VolcanoStyle -- spell them
    'color_by' and 'label_column', so the menu offered free text where the
    panel offered a picker. The choices are now read off the panel's own
    controls, so a name can no longer be spelled two ways."""
    choices = explorer._style_choices()
    assert "color_by" in choices
    assert "label_column" in choices
    assert "gene" in choices["color_by"]
    # `None` is a value there -- it is how a colour-by column is taken back
    # off -- so the menu has to offer it too.
    assert None in choices["color_by"]


def test_a_mapping_menu_that_is_not_on_this_panel_is_skipped(explorer):
    """The column menus are refilled by name, and a panel that does not
    build one of them must not cost the others their columns."""
    del explorer._controls["shape_by"]
    explorer.set_results(_results(6))
    offered = [explorer._controls["color_by"].itemData(i)
               for i in range(explorer._controls["color_by"].count())]
    assert "gene" in offered


# --------------------------------------------------------------------------
# the menu and the panel are two routes to one figure
# --------------------------------------------------------------------------

def _menu_words(menu) -> list:
    """Every word the menu shows a reader: entries and the groups holding
    them."""
    from spacr.qt.widgets.fast_plots import menu_entries, menu_groups

    return ([action.text() for action in menu_entries(menu)]
            + list(menu_groups(menu)))


def _menu_action(menu, prefix: str):
    from spacr.qt.widgets.fast_plots import menu_entries

    for action in menu_entries(menu):
        if action.text().startswith(prefix):
            return action
    raise AssertionError(
        f"nothing on the menu starts with {prefix!r}: {_menu_words(menu)}")


def test_the_menu_and_the_side_panel_offer_the_same_settings(explorer):
    """Asserted as a set comparison, not by eye: two routes to one figure
    that disagree about what can be set is the defect this closes."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(VolcanoStyle)}
    assert explorer.menu_settings() == explorer.panel_settings()
    assert explorer.panel_settings() == fields


def test_the_axis_limits_and_labels_are_on_both_routes(explorer):
    """Measured before the change: neither the menu NOR the side panel
    offered `x_lim`/`y_lim`, though the style carried both."""
    for setting in ("x_lim", "y_lim", "x_label", "y_label"):
        assert setting in explorer.panel_settings(), setting
        assert setting in explorer.menu_settings(), setting


def test_every_appearance_setting_is_on_both_routes(explorer):
    """Point size and opacity by name, because the ask named them."""
    for setting in ("marker_size", "significant_marker_size", "marker_alpha",
                    "marker_edge_width", "colormap", "color_vmin",
                    "color_vmax", "background_color"):
        assert setting in explorer.panel_settings(), setting
        assert setting in explorer.menu_settings(), setting


def test_a_setting_added_to_the_style_would_be_missing_from_the_panel(
        explorer):
    """The set comparison is the check that keeps the two in step, so it has
    to be able to fail: dropping one control makes the sets differ."""
    del explorer._controls["marker_size"]
    assert explorer.menu_settings() != explorer.panel_settings()


def test_a_limit_typed_into_the_menu_reaches_the_figure_and_the_file(
        explorer, tmp_path, monkeypatch):
    """Driven, then measured off the axes and off the written file -- an
    x limit of ±8 puts tick labels on the page that the data alone
    (roughly ±2.5) would never produce."""
    from PySide6.QtWidgets import QInputDialog

    answers = iter([(-8.0, True), (8.0, True)])
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: next(answers)))
    _menu_action(explorer.build_style_menu(), "X lim").trigger()

    assert explorer.style().x_lim == (-8.0, 8.0)
    explorer.refresh()
    assert explorer._panels[0].get_xlim() == (-8.0, 8.0)
    # And the side panel followed, rather than still showing "automatic".
    assert explorer._controls["x_lim"].value() == (-8.0, 8.0)

    path = tmp_path / "limited.svg"
    assert explorer.export("svg", str(path)) == str(path)
    written = path.read_text(encoding="utf-8")
    ticks = [t.get_text() for t in explorer._panels[0].get_xticklabels()]
    assert "−8" in ticks, ticks
    for tick in ticks:
        assert tick in written, f"{tick} never reached the file"


def test_an_axis_label_typed_into_the_menu_reaches_the_figure_and_the_file(
        explorer, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("Effect, typed", True)))
    _menu_action(explorer.build_style_menu(), "X label").trigger()

    assert explorer.style().x_label == "Effect, typed"
    explorer.refresh()
    assert explorer._panels[0].get_xlabel() == "Effect, typed"
    assert explorer._controls["x_label"].text() == "Effect, typed"

    path = tmp_path / "labelled.svg"
    explorer.export("svg", str(path))
    assert "Effect, typed" in path.read_text(encoding="utf-8")


def test_a_limit_can_be_taken_back_off_from_the_panel(explorer):
    """A spin box alone cannot say `None`, so a limit set once could never be
    cleared; the 'auto' tick is that third state."""
    explorer._controls["x_lim"].setValue((-3.0, 3.0))
    explorer._on_control_changed()
    assert explorer.style().x_lim == (-3.0, 3.0)

    explorer._controls["x_lim"].setValue(None)
    explorer._on_control_changed()
    assert explorer.style().x_lim is None
    explorer.refresh()
    assert explorer._panels[0].get_xlim() != (-3.0, 3.0)


def test_an_effect_threshold_left_automatic_stays_none(explorer):
    """It was a plain spin box, so the first touch of any control anywhere in
    the panel wrote 0.0 over a threshold nobody had set."""
    explorer._controls["marker_size"].setValue(31.0)
    assert explorer.style().marker_size == 31.0
    assert explorer.style().effect_threshold is None


# --------------------------------------------------------------------------
# colour by localization, several at once
# --------------------------------------------------------------------------

@pytest.fixture
def localised(qapp, qtbot):
    """A screen whose gene ids the bundled LOPIT table actually knows.

    Two compartments, ten genes each, so "dense granules and rhoptries 1" is
    a question this frame can answer.
    """
    from spacr.localisation import table

    lookup = table()
    if not lookup:
        pytest.skip("no bundled localisation table to colour by")
    genes = []
    for compartment in ("dense granules", "rhoptries 1"):
        genes += [f"TGGT1_{key}" for key, place in lookup.items()
                  if place == compartment][:10]
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({
        "guide": [f"{gene}_1" for gene in genes],
        "gene": genes,
        "standardized_marginal_effect": rng.normal(size=len(genes)),
        "adjusted_p_value": rng.random(len(genes)) * 0.5 + 1e-6,
    })
    widget = VolcanoExplorer(frame)
    qtbot.addWidget(widget)
    return widget


def test_only_the_compartments_this_screen_has_are_offered(localised):
    """A tick box that would colour nothing is indistinguishable from a
    broken one."""
    assert localised.compartments() == ["dense granules", "rhoptries 1"]
    offered = localised._controls["localizations"].options()
    assert offered == ["dense granules", "rhoptries 1"]


def test_two_compartments_can_be_ticked_at_once_on_the_menu(localised):
    """"all or any combination of localizations, e.g. dense granuals and
    rhoptries 1 should be possible" -- so this is a submenu of tick boxes,
    not a pick-one."""
    menu = localised.build_style_menu()
    ticks = [a for a in _menu_words(menu)]
    assert "dense granules" in ticks and "rhoptries 1" in ticks

    from spacr.qt.widgets.fast_plots import menu_entries

    by_text = {a.text(): a for a in menu_entries(menu)}
    by_text["dense granules"].setChecked(True)
    assert localised.style().localizations == ("dense granules",)
    by_text["rhoptries 1"].setChecked(True)
    assert localised.style().localizations == ("dense granules",
                                               "rhoptries 1")


def test_ticking_two_compartments_recolours_the_points(localised):
    localised._style.localizations = ("dense granules", "rhoptries 1")
    localised.refresh()

    drawn = {c.get_label(): tuple(np.asarray(c.get_facecolor())[0][:3])
             for c in localised._panels[0].collections}
    assert set(drawn) == {"dense granules", "rhoptries 1"}
    # Two compartments, two colours: one colour for both would be a picture
    # that cannot answer the question it was asked.
    assert len(set(drawn.values())) == 2


def test_unticking_a_compartment_puts_the_grey_back(localised):
    localised._style.localizations = ("dense granules", "rhoptries 1")
    localised.refresh()
    localised._style.localizations = ("dense granules",)
    localised.refresh()

    labels = {c.get_label() for c in localised._panels[0].collections}
    assert labels == {"dense granules", "elsewhere"}


def test_the_menus_ticks_reach_the_side_panel(localised):
    """Two ways to change one setting that disagree about what it now is
    would be worse than having only one of them."""
    from spacr.qt.widgets.fast_plots import menu_entries

    menu = localised.build_style_menu()
    next(a for a in menu_entries(menu)
         if a.text() == "rhoptries 1").setChecked(True)

    assert localised._controls["localizations"].values() == ("rhoptries 1",)


def test_the_panels_ticks_reach_the_style_and_the_plot(localised):
    widget = localised._controls["localizations"]
    widget.setValues(("dense granules", "rhoptries 1"))
    localised._on_control_changed()

    assert localised.style().localizations == ("dense granules",
                                               "rhoptries 1")
    labels = {c.get_label() for c in localised._panels[0].collections}
    assert labels == {"dense granules", "rhoptries 1"}


def test_a_compartment_this_screen_lacks_survives_a_refill(localised):
    """A style file naming a compartment the screen has none of must not
    lose it the moment the panel is refilled."""
    localised._style.localizations = ("apicoplast",)
    localised.set_style(localised._style)
    localised._repopulate_column_menus()

    assert "apicoplast" in localised._controls["localizations"].values()


def test_a_screen_with_no_recognised_genes_offers_no_compartments(explorer):
    """`_results` names genes the reference table has never heard of, and a
    volcano is still a volcano without compartment colouring."""
    assert explorer.compartments() == []
    assert explorer._controls["localizations"].count() == 0


# --------------------------------------------------------------------------
# nothing says "aspect ratio"
# --------------------------------------------------------------------------

def test_nothing_the_volcano_says_to_a_user_mentions_an_aspect_ratio(
        explorer):
    """A ratio is a number and the choice is a shape."""
    from PySide6.QtWidgets import QAbstractButton, QGroupBox, QLabel, QWidget

    words = _menu_words(explorer.build_style_menu())
    for widget in explorer.findChildren(QWidget):
        words.append(widget.toolTip())
        caption = widget.property("caption")
        if caption:
            words.append(str(caption))
        if isinstance(widget, QGroupBox):
            words.append(widget.title())
        elif isinstance(widget, (QAbstractButton, QLabel)):
            words.append(widget.text())
    offending = [word for word in words
                 if word and "aspect ratio" in word.lower()]
    assert offending == [], offending
