"""One uncovered decision each, in five widgets that are otherwise complete.

Four shapes recur across the Qt tree and are all here: a layout item that
is not a widget, a layout built on demand and asked for twice, a signal
handler reached without a sender, and a rectangle whose inner box has
collapsed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

import numpy as np
from PySide6.QtWidgets import QLabel, QSpacerItem, QSizePolicy

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# save_figure_dialog / figure_grid -- a spacer among the widgets
# ---------------------------------------------------------------------------

class TestLayoutItemsThatAreNotWidgets:
    """`item.widget()` is None for a spacer, and both teardown loops skip it.

    Calling setParent or deleteLater on that None raises inside the loop,
    which would leave the rest of the layout undeleted -- the figures
    behind a re-render would stack up instead of being replaced.
    """

    def test_the_save_dialog_clears_a_holder_containing_a_spacer(self, qtbot):
        from spacr.qt.widgets.save_figure_dialog import SaveFigureDialog

        dialog = SaveFigureDialog(figure=None)
        qtbot.addWidget(dialog)

        dialog._holder.addWidget(QLabel("a preview"))
        dialog._holder.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Minimum,
                        QSizePolicy.Policy.Fixed))
        dialog._holder.addWidget(QLabel("another"))

        dialog._clear_holder()
        assert dialog._holder.count() == 0, (
            "a non-widget item stopped the holder being emptied")

    def test_the_figure_grid_relays_out_over_a_spacer(self, qtbot):
        from spacr.qt.widgets.figure_grid import SearchFigureGrid

        grid = SearchFigureGrid(["alpha", "beta"])
        qtbot.addWidget(grid)

        grid._grid.addWidget(QLabel("a figure"), 0, 0)
        grid._grid.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Minimum,
                        QSizePolicy.Policy.Fixed), 0, 1)

        grid.relayout()
        assert grid._grid.count() == 0
        assert grid._labels == []


# ---------------------------------------------------------------------------
# section.py -- the header layout is built once
# ---------------------------------------------------------------------------

class TestTheHeaderLayoutIsBuiltOnDemandAndOnlyOnce:

    def test_asking_twice_returns_the_same_layout(self, qtbot):
        """THE UNCOVERED ARC: the second call finds a layout already there.

        A second QHBoxLayout on the same header would be refused by Qt
        with a warning and the mark would never be placed, so the second
        module mark on a section has to reuse the first one's row.
        """
        from spacr.qt.widgets.section import Section

        section = Section("A heading")
        qtbot.addWidget(section)

        first = section._source_row()
        assert first is not None
        second = section._source_row()

        assert second is first, "the header layout was rebuilt"
        assert section._header.layout() is first

    def test_a_section_with_no_mark_never_grows_a_header_layout(self, qtbot):
        """Why it is on demand: every category that has no mark is
        the widget it always was."""
        from spacr.qt.widgets.section import Section

        section = Section("A heading")
        qtbot.addWidget(section)

        assert section._header.layout() is None


# ---------------------------------------------------------------------------
# channel_picker.py -- the correction runs without a Toggle to correct
# ---------------------------------------------------------------------------

class TestUncheckingTheLastChannel:

    def test_the_last_box_is_put_back_when_none_is_not_allowed(self, qtbot):
        from spacr.qt.widgets.channel_picker import ChannelPicker

        picker = ChannelPicker("r")
        qtbot.addWidget(picker)
        picker._allow_none = False

        box = next(iter(picker._boxes.values()))
        box.setChecked(True)
        for other in picker._boxes.values():
            if other is not box:
                other.setChecked(False)

        box.setChecked(False)          # emits toggled with box as sender
        assert picker.value(), "the picture was allowed to go blank"

    def test_called_with_no_sender_the_correction_is_skipped(self, qtbot):
        """THE UNCOVERED ARC.

        `sender()` is None when the slot is called directly rather than
        through the signal, and it is typed as QObject even when it is
        not. Neither is a Toggle, so there is nothing to put back and
        the handler falls through to announcing the value -- which is
        the honest thing to do: an empty value the code did not cause
        must still reach the listeners.
        """
        from spacr.qt.widgets.channel_picker import ChannelPicker

        picker = ChannelPicker("")
        qtbot.addWidget(picker)
        picker._allow_none = False
        for box in picker._boxes.values():
            box.blockSignals(True)
            box.setChecked(False)
            box.blockSignals(False)

        assert picker.value() == ""
        with qtbot.waitSignal(picker.changed, timeout=500) as caught:
            picker._on_toggled(False)   # no sender: nothing to correct
        assert caught.args == [""]


# ---------------------------------------------------------------------------
# animation_zoom.py -- a ring with no inside left
# ---------------------------------------------------------------------------

class TestAFieldRingThickerThanItsOwnWell:

    def test_a_normal_ring_has_a_hole_in_it(self):
        from spacr.qt.widgets.animation_zoom import _field_masks

        ring, inside = _field_masks(64)
        assert ring.any() and inside.any()
        assert int(inside.sum()) > int(ring.sum()), (
            "the ring is not thinner than the area it encloses")

    def test_padding_wider_than_the_box_leaves_a_solid_disc(self):
        """THE UNCOVERED ARC: the inner rectangle has collapsed.

        Drawing a rounded rectangle whose right edge is left of its left
        edge is not a smaller shape, it is an error from Pillow -- so
        the inner mask is left empty and the ring becomes the whole
        disc. That is the correct picture for a pad that thick.
        """
        from spacr.qt.widgets.animation_zoom import _field_masks

        ring, inside = _field_masks(64, box=(20.0, 20.0, 24.0, 24.0),
                                    radius=1.0, pad=40.0)

        assert np.array_equal(ring, inside), (
            "an inner hole was cut from a box that has no inside")
        assert ring.any(), "the whole disc vanished"


# ---------------------------------------------------------------------------
# gate_settings.py -- a projection control that was never built
# ---------------------------------------------------------------------------

class TestGreyingTheControlsAProjectionDoesNotRead:

    def test_only_the_chosen_projections_controls_stay_enabled(self, qtbot):
        from spacr.qt.widgets.gate_settings import (GateEditorSettings,
                                                    GateSettingsDialog)

        dialog = GateSettingsDialog(GateEditorSettings())
        qtbot.addWidget(dialog)

        dialog._reduction.setCurrentText("umap")
        dialog._grey_irrelevant_methods()
        assert dialog._n_neighbors.isEnabled()
        assert dialog._perplexity.isEnabled() is False

        dialog._reduction.setCurrentText("tsne")
        dialog._grey_irrelevant_methods()
        assert dialog._perplexity.isEnabled()
        assert dialog._n_neighbors.isEnabled() is False

    def test_a_control_that_is_not_there_is_skipped_not_crashed_on(
            self, qtbot):
        """THE UNCOVERED ARC.

        The table names controls by string, so it can name one the
        dialog does not have -- a control removed from the form but left
        in the table. Greying is cosmetic and must not be the thing that
        stops the settings window opening, so a missing name is skipped.
        """
        from spacr.qt.widgets.gate_settings import (GateEditorSettings,
                                                    GateSettingsDialog)

        dialog = GateSettingsDialog(GateEditorSettings())
        qtbot.addWidget(dialog)

        assert hasattr(dialog, "_perplexity")
        setattr(dialog, "_perplexity", None)

        dialog._reduction.setCurrentText("umap")
        dialog._grey_irrelevant_methods()   # must not raise
        assert dialog._n_neighbors.isEnabled(), (
            "the controls that DO exist were not greyed after a missing one")

    def test_every_name_in_the_table_is_a_real_control(self, qtbot):
        """The pin behind the skip: today nothing is missing."""
        from spacr.qt.widgets.gate_settings import (GateEditorSettings,
                                                    GateSettingsDialog)

        dialog = GateSettingsDialog(GateEditorSettings())
        qtbot.addWidget(dialog)

        for controls in GateSettingsDialog._METHOD_CONTROLS.values():
            for name in controls:
                assert getattr(dialog, name, None) is not None, (
                    f"{name} is named in _METHOD_CONTROLS but never built")
