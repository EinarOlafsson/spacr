"""Every control in the figure dialog changes the figure it is opened on.

A control that is built, laid out and connected to nothing reads as a
capability the figure does not have. These drive the widgets the user
drives -- the spin boxes, the check boxes, the colour buttons -- and assert
against the matplotlib artists, not against the dialog's own state.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtWidgets import QFormLayout, QPushButton  # noqa: E402

from spacr.qt.widgets import figure_settings as fs  # noqa: E402

pytestmark = pytest.mark.qt

#: More series than the dialog styles one by one, so the shared rules appear.
MANY = fs.FigureSettingsDialog.SERIES_DETAIL_LIMIT + 2


def rows_of(dialog, tab_index):
    """``{row label: field widget}`` for one tab of the dialog."""
    page = dialog.tabs.widget(tab_index)
    found = {}
    for form in page.findChildren(QFormLayout):
        for row in range(form.rowCount()):
            label = form.itemAt(row, QFormLayout.LabelRole)
            field = form.itemAt(row, QFormLayout.FieldRole)
            if label is None or field is None:
                continue
            widget = label.widget()
            text = widget.text() if hasattr(widget, "text") else None
            if text and field.widget() is not None:
                found[text] = field.widget()
    return found


@pytest.fixture()
def figure():
    fig = plt.figure()
    crowded = fig.add_subplot(211)
    for index in range(MANY):
        crowded.plot([0, 1, 2], [index, index + 1, index + 2],
                     label=f"series {index}")
    # A scatter among them: a collection takes its size as an AREA and a
    # line takes a width, so one rule has to reach both spellings.
    crowded.scatter([0, 1], [0.5, 1.5], label="scattered")
    crowded.set_title("crowded")
    sparse = fig.add_subplot(212)
    sparse.plot([0, 1, 2], [1, 2, 3], label="one line")
    sparse.scatter([0, 1], [1, 2], label="points")
    sparse.legend()
    sparse.set_title("sparse")
    yield fig
    plt.close(fig)


@pytest.fixture()
def dialog(qapp, figure):
    edits = []
    widget = fs.FigureSettingsDialog(
        figure, on_change=lambda **_kwargs: edits.append(1))
    widget._edits = edits
    yield widget
    widget.deleteLater()


def tab_named(dialog, title):
    for index in range(dialog.tabs.count()):
        if dialog.tabs.tabText(index).startswith(title):
            return index
    raise AssertionError(f"no {title!r} tab: "
                         f"{[dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]}")


# --------------------------------------------------------------------------
# the shared rules an axes with many series gets instead of one block each
# --------------------------------------------------------------------------

def test_a_crowded_axes_is_given_rules_rather_than_a_block_per_series(dialog):
    rows = rows_of(dialog, tab_named(dialog, "crowded"))

    assert "Palette" in rows
    assert "  Colour" not in rows, (
        f"{MANY} series would be {MANY * 5} controls, not a screen")


def test_keeping_the_current_colours_changes_nothing(dialog, figure):
    rows = rows_of(dialog, tab_named(dialog, "crowded"))
    palette = rows["Palette"]
    before = [line.get_color() for line in figure.axes[0].lines]

    palette.setCurrentIndex(palette.findData("tab10"))
    recoloured = [line.get_color() for line in figure.axes[0].lines]
    palette.setCurrentIndex(0)

    assert recoloured != before, "a palette has to reach the lines"
    assert [line.get_color() for line in figure.axes[0].lines] == recoloured, (
        "'Keep current colours' keeps them, it does not re-apply a palette")


def test_the_point_size_rule_reaches_every_series(dialog, figure):
    rows = rows_of(dialog, tab_named(dialog, "crowded"))

    rows["Point size (all)"].setValue(64.0)

    assert all(line.get_markersize() == pytest.approx(8.0)
               for line in figure.axes[0].lines), (
        "a line takes the square root: the box is an area, the marker a width")
    assert figure.axes[0].collections[0].get_sizes() == pytest.approx([64.0]), (
        "a collection takes the area unchanged")


def test_the_opacity_rule_reaches_every_series(dialog, figure):
    rows = rows_of(dialog, tab_named(dialog, "crowded"))

    rows["Opacity (all)"].setValue(0.25)

    assert all(line.get_alpha() == pytest.approx(0.25)
               for line in figure.axes[0].lines)


def test_the_outline_width_rule_reaches_every_series(dialog, figure):
    rows = rows_of(dialog, tab_named(dialog, "crowded"))

    rows["Outline width (all)"].setValue(3.0)

    assert all(line.get_linewidth() == pytest.approx(3.0)
               for line in figure.axes[0].lines)


# --------------------------------------------------------------------------
# the axes controls
# --------------------------------------------------------------------------

def test_autoscale_puts_the_limits_back_around_the_data(dialog, figure):
    axis = figure.axes[1]
    axis.set_xlim(100.0, 200.0)
    page = dialog.tabs.widget(tab_named(dialog, "sparse"))
    button = [widget for widget in page.findChildren(QPushButton)
              if widget.text() == "Autoscale to data"]
    assert button, "the autoscale button is on the axes tab"
    button[0].click()

    low, high = axis.get_xlim()
    assert low < 2.0 < high, "the data is back inside the view"


def test_a_zero_width_limit_is_not_applied(dialog, figure):
    axis = figure.axes[1]
    before = axis.get_xlim()
    rows = rows_of(dialog, tab_named(dialog, "sparse"))
    boxes = rows["X limits"].findChildren(type(rows["Spine width"]))

    boxes[0].setValue(5.0)
    boxes[1].setValue(5.0)

    assert axis.get_xlim() != (5.0, 5.0), (
        "matplotlib throws on a zero-width axis; the box waits for the other")
    assert axis.get_xlim() != before, "the first edit did land"


def test_the_grid_switch_turns_the_grid_on_and_off(dialog, figure):
    axis = figure.axes[1]
    rows = rows_of(dialog, tab_named(dialog, "sparse"))

    rows["Grid"].setChecked(True)
    assert any(line.get_visible() for line in axis.get_xgridlines())

    rows["Grid"].setChecked(False)
    assert not any(line.get_visible() for line in axis.get_xgridlines()), (
        "line properties passed while disabling turn the grid back on")


def test_the_spine_width_reaches_every_spine(dialog, figure):
    axis = figure.axes[1]
    rows = rows_of(dialog, tab_named(dialog, "sparse"))

    rows["Spine width"].setValue(2.5)

    assert all(spine.get_linewidth() == pytest.approx(2.5)
               for spine in axis.spines.values())


def test_hiding_the_top_and_right_hides_only_those_two(dialog, figure):
    axis = figure.axes[1]
    rows = rows_of(dialog, tab_named(dialog, "sparse"))

    rows["Hide top/right"].setChecked(True)

    assert axis.spines["top"].get_visible() is False
    assert axis.spines["right"].get_visible() is False
    assert axis.spines["left"].get_visible() is True


# --------------------------------------------------------------------------
# the legend
# --------------------------------------------------------------------------

def test_switching_the_legend_off_hides_it_rather_than_losing_it(dialog,
                                                                 figure):
    axis = figure.axes[1]
    rows = rows_of(dialog, tab_named(dialog, "sparse"))

    rows["Legend"].setChecked(False)

    assert axis.get_legend() is not None, "hidden, not destroyed"
    assert axis.get_legend().get_visible() is False


def test_switching_it_back_on_rebuilds_it_where_it_was_asked_for(dialog,
                                                                 figure):
    axis = figure.axes[1]
    rows = rows_of(dialog, tab_named(dialog, "sparse"))
    rows["Legend"].setChecked(False)

    rows["Legend position"].setCurrentText("lower left")
    rows["Legend"].setChecked(True)

    legend = axis.get_legend()
    assert legend is not None and legend.get_visible() is True
    assert legend._loc == 3, "'lower left' is where it was asked for"


def test_an_unlabelled_axes_keeps_the_legend_it_already_had(qapp):
    """Rebuilding needs labelled artists; restyling the existing one does not."""
    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.plot([0, 1], [1, 2], label="named")
    axis.set_title("relabelled")
    axis.legend()
    for line in axis.lines:
        line.set_label("_nolegend_")

    widget = fs.FigureSettingsDialog(fig)
    try:
        rows = rows_of(widget, tab_named(widget, "relabelled"))
        rows["Legend"].setChecked(False)
        rows["Legend text size"].setValue(17)
        rows["Legend"].setChecked(True)

        legend = axis.get_legend()
        assert legend is not None, (
            "legend() with no labelled artists returns nothing and would "
            "throw away the legend the figure already had")
        assert legend.get_visible() is True
        assert all(text.get_fontsize() == 17 for text in legend.get_texts())
    finally:
        widget.deleteLater()
        plt.close(fig)


# --------------------------------------------------------------------------
# the colour buttons
# --------------------------------------------------------------------------

@pytest.fixture()
def picks(monkeypatch):
    """Answer the colour picker without opening it."""
    chosen = {"colour": "#123456"}
    monkeypatch.setattr(fs, "pick_colour",
                        lambda *args, **kwargs: QColor(chosen["colour"]))
    return chosen


def test_the_background_button_repaints_the_figure_patch(dialog, figure,
                                                          picks):
    rows = rows_of(dialog, tab_named(dialog, "Figure"))

    rows["Background"].click()

    assert matplotlib.colors.to_hex(figure.patch.get_facecolor()) == "#123456"


def test_the_line_and_the_font_colours_are_two_separate_buttons(dialog,
                                                                figure,
                                                                picks):
    rows = rows_of(dialog, tab_named(dialog, "Figure"))
    axis = figure.axes[1]

    picks["colour"] = "#ff0000"
    rows["Line colour"].click()
    picks["colour"] = "#00ff00"
    rows["Font colour"].click()

    assert matplotlib.colors.to_hex(
        axis.spines["left"].get_edgecolor()) == "#ff0000"
    assert matplotlib.colors.to_hex(
        axis.title.get_color()) == "#00ff00", (
        "changing the axis colour must not drag the text with it")


def test_a_series_colour_button_recolours_that_series_only(dialog, figure,
                                                            picks):
    axis = figure.axes[1]
    other = axis.collections[0].get_facecolor().copy()
    rows_by_label = []
    page = dialog.tabs.widget(tab_named(dialog, "sparse"))
    for form in page.findChildren(QFormLayout):
        for row in range(form.rowCount()):
            label = form.itemAt(row, QFormLayout.LabelRole)
            field = form.itemAt(row, QFormLayout.FieldRole)
            if label and field and getattr(label.widget(), "text", None) \
                    and label.widget().text() == "  Colour":
                rows_by_label.append(field.widget())

    assert rows_by_label, "each named series gets its own colour button"
    rows_by_label[0].click()

    assert matplotlib.colors.to_hex(axis.lines[0].get_color()) == "#123456"
    assert (axis.collections[0].get_facecolor() == other).all(), (
        "the other series was not touched")


# --------------------------------------------------------------------------
# the figure size
# --------------------------------------------------------------------------

def test_the_size_boxes_resize_the_figure(dialog, figure):
    rows = rows_of(dialog, tab_named(dialog, "Figure"))

    rows["Width (in)"].setValue(9.0)
    rows["Height (in)"].setValue(4.0)

    assert figure.get_size_inches() == pytest.approx([9.0, 4.0])


# --------------------------------------------------------------------------
# the Image UMAP half, live against the figure it was drawn from
# --------------------------------------------------------------------------

@pytest.fixture()
def umap_figure():
    """A finished Image UMAP: a scatter plus the payload it was drawn from."""
    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.scatter([0.0, 1.0, 2.0], [0.0, 1.0, 0.5])
    axis.set_title("umap")
    fig._spacr_umap_payload = {"settings": {"dot_size": 20}}
    yield fig
    plt.close(fig)


def test_a_umap_figure_gets_the_umap_tab(qapp, umap_figure):
    dialog = fs.FigureSettingsDialog(umap_figure)
    try:
        titles = [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]
        assert "Image UMAP" in titles
        assert dialog.umap_values(), "the tab carries the payload's settings"
    finally:
        dialog.deleteLater()


def test_a_plain_figure_has_no_umap_tab(qapp, figure):
    dialog = fs.FigureSettingsDialog(figure)
    try:
        titles = [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]
        assert "Image UMAP" not in titles, (
            "live would mean re-running the reduction, and every point moving")
        assert dialog.umap_values() == {}
    finally:
        dialog.deleteLater()


def test_a_changed_umap_setting_restyles_the_points_it_was_drawn_with(
        qapp, umap_figure):
    dialog = fs.FigureSettingsDialog(umap_figure)
    try:
        values = dict(dialog.umap_values())
        values["dot_size"] = 91
        dialog._umap_settings.settings_changed.emit(values)

        assert umap_figure.axes[0].collections[0].get_sizes() \
            == pytest.approx([91.0]), (
            "the embedding is read, never recomputed: only the style moves")
        assert dialog._umap_applied["dot_size"] == 91, (
            "what was applied is remembered, so the next change is a delta")
    finally:
        dialog.deleteLater()


def test_the_umap_values_are_propagated_into_the_module_panel(qapp,
                                                              umap_figure):
    written = []
    dialog = fs.FigureSettingsDialog(
        umap_figure, propagate_callback=lambda values: written.append(values))
    try:
        dialog._propagate_btn.click()
    finally:
        dialog.deleteLater()

    assert written and "dot_size" in written[0], (
        "the next run starts from what the figure now shows")
