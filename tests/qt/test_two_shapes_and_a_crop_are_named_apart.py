"""Three controls, three quantities, and not one of them "aspect ratio".

A graph offers TWO things a reader is entitled to tell apart:

    Graph shape        square / horizontal rectangle / vertical rectangle --
                       the proportions of the FIGURE, which is what gets
                       exported and what "save it as a square" asks for.
    Lock axis scales   one y unit drawn as n x units -- a statement about
                       the DATA, which is what a Q-Q's 45-degree diagonal
                       needs and what the page has nothing to do with.

"Aspect ratio" named both at once, which is how they came to be read as one
control. The graph's own right-click menu was named apart first; these pin
the other places a user meets either quantity to the same words, because a
second name for one thing is the same failure as one name for two.

A third quantity turns up on the Measure crop panel -- the pixel box an
object is cut out into. It is neither graph control, and it must not borrow
either word.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QWidget


# --------------------------------------------------------------------------- #
#  reading a built form the way a user reads it: by the words beside a control
# --------------------------------------------------------------------------- #

def _label_text(widget) -> str:
    """The words shown for a form row's label.

    A row's label is not always the ``QLabel`` itself: a settings form wraps
    it together with its API link, so the text lives one level down.
    """
    if isinstance(widget, QLabel):
        return widget.text()
    for child in widget.findChildren(QLabel):
        if (child.text() or "").strip():
            return child.text()
    return ""


def _row_label(container, field) -> str:
    """What ``container`` writes beside ``field``, or "" if it writes nothing."""
    for form in container.findChildren(QFormLayout):
        for index in range(form.rowCount()):
            item = form.itemAt(index, QFormLayout.FieldRole)
            if item is None or item.widget() is not field:
                continue
            label_item = form.itemAt(index, QFormLayout.LabelRole)
            widget = label_item.widget() if label_item is not None else None
            return _label_text(widget) if widget is not None else ""
    return ""


def _every_word_shown(container) -> list:
    """Every label a reader can see in ``container``."""
    return [label.text() for label in container.findChildren(QLabel)
            if (label.text() or "").strip()]


def _axis_lock_entry_in_the_graph_menu() -> str:
    """What the graph's own right-click menu calls the data lock."""
    from spacr.qt.widgets.fast_plots import FastPlot, menu_entries

    plot = FastPlot()
    try:
        entries = [action.text() for action
                   in menu_entries(plot.build_style_menu())]
    finally:
        plot.deleteLater()
    locks = [text for text in entries if "axis scales" in text.lower()]
    assert locks, entries
    return locks[0]


# --------------------------------------------------------------------------- #
#  the save dialog: file shape here, data lock on the plot
# --------------------------------------------------------------------------- #

@pytest.fixture()
def save_dialog(qtbot):
    """The dialog with no figure behind it: this is about its words."""
    from spacr.qt.widgets.save_figure_dialog import SaveFigureDialog

    dialog = SaveFigureDialog(None)
    qtbot.addWidget(dialog)
    return dialog


def test_the_save_dialog_does_not_duplicate_the_plots_data_lock(save_dialog):
    """The data lock belongs to the plot and therefore every export."""
    assert not hasattr(save_dialog, "aspect")
    assert all("lock axis scales" not in text.lower()
               for text in _every_word_shown(save_dialog))


def test_the_graph_menu_uses_one_unambiguous_name_for_the_data_lock():
    assert "lock axis scales" in _axis_lock_entry_in_the_graph_menu().lower()


def test_the_save_shape_points_to_the_separate_axis_lock(save_dialog):
    """The file control says where the data-owned quantity lives."""
    tip = save_dialog.graph_shape.toolTip().lower()

    assert "page" in tip and "axis lock" in tip and "data" in tip, tip


def test_the_saved_figures_shape_is_offered_in_the_menus_words(save_dialog):
    """The other quantity, and the same three shapes the menu draws."""
    assert _row_label(save_dialog, save_dialog.graph_shape) == "graph shape"
    offered = [save_dialog.graph_shape.itemText(index)
               for index in range(save_dialog.graph_shape.count())]

    assert offered == ["as drawn", "square", "horizontal rectangle",
                       "vertical rectangle"], offered


def test_no_row_of_the_save_dialog_says_aspect_ratio(save_dialog):
    """The word that meant both quantities is gone from the dialog, labels
    and tooltips alike -- a tooltip is read by exactly the user who is
    trying to tell the two apart."""
    said = [text for text in _every_word_shown(save_dialog)
            if "aspect ratio" in text.lower()]
    tips = [widget.toolTip() for widget in save_dialog.findChildren(QWidget)
            if "aspect ratio" in (widget.toolTip() or "").lower()]

    assert not said, said
    assert not tips, tips


# --------------------------------------------------------------------------- #
#  the figure-style preference: which quantity the row actually is
# --------------------------------------------------------------------------- #

def _aspect_row_label() -> str:
    from spacr.qt.widgets.figure_settings import style_setting_label

    return style_setting_label("aspect")


def test_the_style_row_for_aspect_is_explained_as_the_axis_lock():
    """`aspect` takes 'equal' or 'auto' -- matplotlib's axes aspect, which
    ties one data unit in y to one in x. It is the axis-scale lock, and its
    explanation has to say so instead of promising proportions."""
    from spacr.qt.preferences import PREFERENCE_TIPS

    tip = PREFERENCE_TIPS[_aspect_row_label()]

    assert "axis scales" in tip.lower(), tip
    assert "rectangle" not in tip.lower(), (
        "this row cannot take a rectangle; the shape of the figure is a "
        f"different setting: {tip}")


def test_that_explanation_names_the_values_the_row_actually_offers(
        qtbot, tmp_path, monkeypatch):
    """Measured against the built control, so an explanation describing a
    choice the row does not offer is caught by the row itself."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from spacr.qt.preferences import PREFERENCE_TIPS
    from spacr.qt.widgets.figure_settings import FigureStylePreferences

    panel = FigureStylePreferences()
    qtbot.addWidget(panel)
    label = _aspect_row_label()
    combo = None
    for form in panel.findChildren(QFormLayout):
        for index in range(form.rowCount()):
            label_item = form.itemAt(index, QFormLayout.LabelRole)
            field_item = form.itemAt(index, QFormLayout.FieldRole)
            if label_item is None or field_item is None:
                continue
            if _label_text(label_item.widget()) != label:
                continue
            if isinstance(field_item.widget(), QComboBox):
                combo = field_item.widget()
    assert combo is not None, f"no {label!r} row with a choice on it"
    offered = [combo.itemText(index) for index in range(combo.count())]

    assert offered == ["equal", "auto"], offered
    tip = PREFERENCE_TIPS[label]
    for value in offered:
        assert f"'{value}'" in tip, (value, tip)


# --------------------------------------------------------------------------- #
#  the crop box: a third quantity, and it borrows neither word
# --------------------------------------------------------------------------- #

@pytest.fixture()
def crop_panel(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from spacr.qt.widgets.measure_preview import MeasurePreviewPanel

    panel = MeasurePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    return panel


@pytest.fixture()
def crop_settings(qtbot, crop_panel):
    from spacr.qt.widgets.measure_preview import CropSettingsDialog

    dialog = CropSettingsDialog(crop_panel)
    qtbot.addWidget(dialog)
    return dialog


def test_the_crop_toggle_is_named_for_the_crop(crop_settings, crop_panel):
    """It sits between "Crop width" and "Crop height" and it holds the
    second at the first. Neither graph word describes it, and either one
    would tell a reader it changes a plot."""
    assert (_row_label(crop_settings, crop_panel._lock_aspect)
            == "Match crop height to width")


def test_the_crop_panel_says_nothing_about_aspect_ratios(crop_settings):
    said = [text for text in _every_word_shown(crop_settings)
            if "aspect" in text.lower()]

    assert not said, said


def test_the_crop_toggle_carries_the_width_over_to_the_height(crop_panel):
    """The label is only true if this is what the toggle does."""
    crop_panel._lock_aspect.setChecked(True)

    crop_panel._crop_width.setValue(128)

    assert crop_panel._crop_height.value() == 128


def test_an_unlocked_crop_keeps_the_height_it_was_given(crop_panel):
    crop_panel._lock_aspect.setChecked(True)
    crop_panel._crop_width.setValue(128)
    crop_panel._lock_aspect.setChecked(False)

    crop_panel._crop_width.setValue(96)

    assert crop_panel._crop_height.value() == 128
