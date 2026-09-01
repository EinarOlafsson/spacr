"""With two objects segmented, the source view can show either or both.

"cell + nucleus" runs Cellpose on two channels, and the source view could only
ever show one of them -- so half of what was being tuned was never on screen.

Reported 2026-09-01: "if primary object is cell + nucleus that the cell channel
array and the nucleus channel array are both loaded and the images should get
back and forth arrows to cycle from showing cell nucleus or both".
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
    # Distinguishable planes, so a composite can be told from a single one.
    image = np.zeros((6, 6, 4), dtype=np.uint8)
    image[..., 1] = 40          # cell channel
    image[..., 2] = 90          # nucleus channel
    p._image = image
    p._cell_channel.setValue(1)
    p._nucleus_channel.setValue(2)
    return p


def test_one_object_has_nothing_to_cycle(panel):
    panel._object_box.setCurrentText("cell")
    assert panel._cycle_stops() == []
    assert panel._cycle_next_btn.isHidden()


def test_two_objects_give_three_stops(panel):
    panel._object_box.setCurrentText("cell + nucleus")
    assert panel._cycle_stops() == [("cell",), ("nucleus",),
                                    ("cell", "nucleus")]


def test_the_arrows_appear_only_for_two_objects(panel):
    panel._object_box.setCurrentText("cell + nucleus")
    assert not panel._cycle_next_btn.isHidden()
    panel._object_box.setCurrentText("cell")
    assert panel._cycle_next_btn.isHidden()


def test_stepping_forward_walks_cell_nucleus_both(panel):
    panel._object_box.setCurrentText("cell + nucleus")
    assert panel._cycle_label.text() == "cell"
    panel._cycle_view(1)
    assert panel._cycle_label.text() == "nucleus"
    panel._cycle_view(1)
    assert panel._cycle_label.text() == "both"


def test_stepping_wraps_at_both_ends(panel):
    panel._object_box.setCurrentText("cell + nucleus")
    panel._cycle_view(-1)
    assert panel._cycle_label.text() == "both", "backwards from the first"
    panel._cycle_view(1)
    assert panel._cycle_label.text() == "cell", "forwards from the last"


def test_a_single_stop_drives_the_ordinary_channel_view(panel):
    """One object is shown the ordinary way, so the channel dropdown and the
    view cannot disagree about which plane is up."""
    panel._object_box.setCurrentText("cell + nucleus")
    panel._cycle_view(1)                       # nucleus
    assert panel._composite_roles == ()
    assert panel._channel_box.currentText() == "Ch 2"


def test_both_composites_the_two_channels(panel):
    panel._object_box.setCurrentText("cell + nucleus")
    panel._cycle_view(2)                       # both
    assert panel._composite_roles == ("cell", "nucleus")
    shown = panel._display_image()
    assert shown.ndim == 3 and shown.shape[-1] == 3, (
        "both must be a composite, not a single plane")


def test_the_composite_carries_both_planes(panel):
    """The point of "both": neither object may be missing from it."""
    panel._object_box.setCurrentText("cell + nucleus")
    panel._cycle_view(2)
    shown = panel._display_image()
    assert 40 in np.unique(shown), "the cell plane is not in the composite"
    assert 90 in np.unique(shown), "the nucleus plane is not in the composite"


def test_the_second_object_lands_in_red_and_blue(panel):
    """Green cells, magenta nuclei -- the outline colours the panel already
    uses, reproduced without a second colour table to keep in step."""
    panel._object_box.setCurrentText("cell + nucleus")
    panel._cycle_view(2)
    shown = panel._display_image()
    assert shown[0, 0, 1] == 40, "the cell plane should be green"
    assert shown[0, 0, 0] == shown[0, 0, 2] == 90, "the nucleus is magenta"


def test_choosing_a_channel_by_hand_ends_the_composite(panel):
    """The dropdown names ONE plane; leaving a composite up would show
    something the control does not describe."""
    panel._object_box.setCurrentText("cell + nucleus")
    panel._cycle_view(2)
    assert panel._composite_roles == ("cell", "nucleus")

    panel._channel_box.setCurrentIndex(panel._channel_box.findText("Ch 3"))

    assert panel._composite_roles == ()
    assert panel._display_image().ndim == 2


def test_changing_the_object_resets_the_cycle(panel):
    """The stops belong to the selection, so an index into the old one means
    nothing."""
    panel._object_box.setCurrentText("cell + nucleus")
    panel._cycle_view(2)
    panel._object_box.setCurrentText("pathogen")
    assert panel._composite_roles == ()
    panel._object_box.setCurrentText("cell + nucleus")
    assert panel._cycle_label.text() == "cell"
