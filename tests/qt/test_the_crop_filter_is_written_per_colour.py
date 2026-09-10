"""Six two-field rows instead of one number for every colour.

The annotator's size filter was a single window applied to every outlined
plane. Red, green and blue hold different objects, so one window is nonsense
for two of them; the ask was a size window AND a brightness window per
colour, written as six rows of two fields rather than twelve settings.

Two things are easy to get wrong here and both are measured:

* **an empty field means NO BOUND**, and that is how a user turns half a
  filter off. It has to survive being saved and read back rather than being
  helpfully filled in with a zero, which is a different filter.
* **the value a project was already filtering on must not vanish.** The old
  single window is migrated onto the three area rows, where the user can see
  it, instead of staying in force somewhere they cannot.

The filtering itself is measured on pixels: an outline is drawn or it is
not, on a field built so that each object's area and mean intensity are
known.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QLabel, QLineEdit


# ---------------------------------------------------------------------------
# The form
# ---------------------------------------------------------------------------

def _dialog(qtbot, settings):
    from spacr.qt.screens.annotate import _SettingsDialog

    dialog = _SettingsDialog(settings)
    qtbot.addWidget(dialog)
    return dialog


def test_the_form_holds_six_rows_of_two_fields(qtbot):
    """Twelve fields, six labels, one row per colour per measure."""
    from spacr.qt.annotate_engine import AnnotateSettings
    from spacr.qt.screens.annotate import FILTER_ROW_LABELS

    dialog = _dialog(qtbot, AnnotateSettings())
    fields = dialog._object_filter_fields
    assert set(fields) == {("r", "area"), ("r", "intensity"),
                           ("g", "area"), ("g", "intensity"),
                           ("b", "area"), ("b", "intensity")}
    edits = [widget for pair in fields.values() for widget in pair]
    assert len(edits) == 12
    assert all(isinstance(widget, QLineEdit) for widget in edits)
    assert all(widget.parent() is not None for widget in edits)

    shown = {label.text() for label in dialog.findChildren(QLabel)}
    for caption in FILTER_ROW_LABELS.values():
        assert caption in shown, f"the form has no {caption!r} row"

    # A min and a max, said in the field rather than in a second label.
    for low, high in fields.values():
        assert low.placeholderText() == "min"
        assert high.placeholderText() == "max"


def test_an_empty_field_means_no_bound_and_survives_a_round_trip(qtbot):
    """Empty is a value, and it is the one that turns a filter off."""
    from spacr.qt.annotate_engine import AnnotateSettings

    settings = AnnotateSettings()
    dialog = _dialog(qtbot, settings)
    dialog._object_filter_fields[("g", "area")][1].setText("900")
    saved = dialog.collect()

    assert saved.object_filters["g_area"] == (None, 900.0)
    assert saved.object_filters["r_area"] == (None, None)

    reopened = _dialog(qtbot, saved)
    assert reopened._object_filter_fields[("g", "area")][0].text() == "", (
        "an empty minimum came back filled in")
    assert reopened._object_filter_fields[("g", "area")][1].text() == "900"
    assert reopened._object_filter_fields[("r", "intensity")][0].text() == ""

    # And through the shape a settings file has: a plain dict and back.
    restored = AnnotateSettings(**dataclasses.asdict(saved))
    assert restored.object_filters["g_area"] == (None, 900.0)
    assert restored.object_filters["b_intensity"] == (None, None)


def test_clearing_a_field_is_not_undone_by_the_old_setting(qtbot):
    """The migration must not put a bound back on a row just emptied."""
    from spacr.qt.annotate_engine import AnnotateSettings

    settings = AnnotateSettings()
    settings.object_size = (200, 900)
    dialog = _dialog(qtbot, settings)
    # It arrives in the new fields rather than staying invisible.
    assert dialog._object_filter_fields[("r", "area")][0].text() == "200"
    assert dialog._object_filter_fields[("b", "area")][1].text() == "900"

    dialog._object_filter_fields[("r", "area")][0].setText("")
    saved = dialog.collect()
    assert saved.object_filters["r_area"] == (None, 900.0)

    reopened = _dialog(qtbot, saved)
    assert reopened._object_filter_fields[("r", "area")][0].text() == "", (
        "the old single window came back on a row the user had emptied")


def test_a_legacy_window_lands_on_every_colours_area_row():
    """It applied to every outlined plane, so it migrates onto all three."""
    from spacr.qt.annotate_engine import normalize_object_filters

    bounds = normalize_object_filters(None, (200, 0))
    for channel in ("r", "g", "b"):
        assert bounds[f"{channel}_area"] == (200.0, None), (
            "a project that hid debris below 200px stopped hiding it")
        assert bounds[f"{channel}_intensity"] == (None, None)


@pytest.mark.parametrize("typed", ["", "   ", None, "large", "1e"])
def test_a_field_that_is_not_a_number_is_no_bound(typed):
    """The fields are free text; a half-typed number must not filter."""
    from spacr.qt.annotate_engine import filter_bound

    assert filter_bound(typed) is None


# ---------------------------------------------------------------------------
# What the filter actually does to the picture
# ---------------------------------------------------------------------------

def _two_object_field():
    """A red plane holding a big dim object and a small bright one.

    Their masks measure 400px at mean 90 and 96px at mean 160 -- the second
    is larger than the drawn block because the foreground mask is smoothed
    before it is thresholded, which is why the numbers below are the ones
    they are.
    """
    array = np.zeros((64, 64, 3), dtype=np.uint8)
    array[6:26, 6:26, 0] = 90
    array[40:48, 40:48, 0] = 240
    return Image.fromarray(array)


def _outlined(**filters):
    """``(big object outlined, small object outlined)``, from the pixels."""
    from spacr.qt.annotate_engine import outline_image

    field = _two_object_field()
    drawn = np.asarray(outline_image(
        field, field, outline_channels=["r"], edge_transparency=100.0,
        object_filters=filters))[:, :, 0]
    return bool(drawn[0:32, 0:32].max()), bool(drawn[32:64, 32:64].max())


def test_both_objects_are_outlined_with_no_filter():
    assert _outlined() == (True, True)


def test_an_area_window_selects_by_size():
    assert _outlined(r_area=(None, 100)) == (False, True)
    assert _outlined(r_area=(200, None)) == (True, False)


def test_an_intensity_window_selects_by_brightness():
    """The measure the single size filter never had."""
    assert _outlined(r_intensity=(120, None)) == (False, True)
    assert _outlined(r_intensity=(None, 100)) == (True, False)


def test_the_two_windows_compose():
    assert _outlined(r_area=(200, None), r_intensity=(None, 100)) == (
        True, False)


def test_another_colours_row_leaves_this_plane_alone():
    """Per colour is the whole point: green's window is not red's."""
    assert _outlined(g_area=(999999, None)) == (True, True)
    assert _outlined(b_intensity=(250, None)) == (True, True)


def test_the_legacy_setting_still_filters_exactly_as_it_did():
    """A caller that only knows ``object_size`` gets what it always got."""
    from spacr.qt.annotate_engine import outline_image

    field = _two_object_field()

    def drawn(object_size):
        return np.asarray(outline_image(
            field, field, outline_channels=["r"], edge_transparency=100.0,
            object_size=object_size))[:, :, 0].max()

    assert drawn(None) == 255
    assert drawn((1, 100000)) == 255
    assert drawn((100000, 200000)) == 0
