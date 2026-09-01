"""Switching the primary object shows that object's own channel.

Choosing "cell" while a nucleus plane was displayed left the user tuning cell
diameter, flow threshold and background against nucleus pixels, with nothing on
screen saying so. Every object already states the channel it is segmented from,
in the same spinner the run reads, so the view can follow it.

Reported 2026-09-01: "if i have a nucleus image loaded and i filled in the
nucleus channel and cell channel and i switch primary object to cell the image
should automatically switch to the cell image in the same series".
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.qt.widgets.live_preview import LivePreviewPanel
from spacr.qt.widgets.preview_controls import populate_channel_combo


@pytest.fixture
def panel(qapp):
    p = LivePreviewPanel()
    populate_channel_combo(p._channel_box, 4)
    p._image = np.zeros((8, 8, 4), dtype=np.uint8)
    p._cell_channel.setValue(2)
    p._nucleus_channel.setValue(1)
    p._pathogen_channel.setValue(3)
    p._organelle_channel.setValue(0)
    return p


def test_choosing_cell_shows_the_cell_channel(panel):
    # Start on the nucleus, the way the report describes: a nucleus image is
    # loaded and both channels have been filled in.
    panel._object_box.setCurrentText("nucleus")
    assert panel._channel_box.currentText() == "Ch 1"

    panel._object_box.setCurrentText("cell")
    assert panel._channel_box.currentText() == "Ch 2"


def test_every_object_follows_its_own_spinner(panel):
    """Not a cell special case: the same rule for each compartment."""
    for obj, expected in (("nucleus", "Ch 1"), ("pathogen", "Ch 3"),
                          ("organelle", "Ch 0"), ("cell", "Ch 2")):
        panel._object_box.setCurrentText(obj)
        assert panel._channel_box.currentText() == expected, obj


def test_retyping_the_channel_moves_the_view_with_it(panel):
    """The spinner is the statement of which plane the object lives on, so
    correcting it while that object is primary must move the view too."""
    panel._object_box.setCurrentText("cell")
    panel._cell_channel.setValue(3)
    assert panel._channel_box.currentText() == "Ch 3"


def test_changing_a_channel_that_is_not_primary_leaves_the_view_alone(panel):
    panel._object_box.setCurrentText("cell")
    panel._nucleus_channel.setValue(0)
    assert panel._channel_box.currentText() == "Ch 2", (
        "editing the nucleus channel moved a view showing the cell")


def test_cell_plus_nucleus_does_not_pick_a_side(panel):
    """Both are being segmented, so neither is the answer.

    Following one of them would make the view flicker between the two as the
    spinners are edited, and would silently claim one of the two objects is
    the one being looked at.
    """
    panel._object_box.setCurrentText("cell")
    assert panel._channel_box.currentText() == "Ch 2"
    panel._object_box.setCurrentText("cell + nucleus")
    assert panel._channel_box.currentText() == "Ch 2", "the view was moved"


def test_a_channel_the_image_does_not_have_is_not_selected(panel):
    """A three-channel image with the pathogen spinner on 7 must not blank the
    view or select something arbitrary -- it stays where it is."""
    panel._object_box.setCurrentText("cell")
    populate_channel_combo(panel._channel_box, 3, keep="Ch 2")
    panel._pathogen_channel.setValue(7)      # no such plane in this image

    panel._object_box.setCurrentText("pathogen")

    assert panel._channel_box.currentText() == "Ch 2", (
        "an out-of-range channel blanked the view or selected arbitrarily")
