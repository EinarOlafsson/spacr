"""The documentation dot paints a different colour, and size, in each of its states.

The dot IS the control -- no border, label or icon -- so normal, hover,
pressed and disabled must each paint their own fill, availability outranks
press state, and the hovered dot is drawn larger than the resting one.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt                        # noqa: E402

from spacr.qt.widgets.dot_link import DotLink        # noqa: E402

#: normal, hover, pressed, disabled -- far apart so the centre pixel names the state.
_COLOURS = ("#ff0000", "#00ff00", "#0000ff", "#ffff00")


def _dot(**state):
    widget = DotLink(tooltip="What this step does", colours=_COLOURS,
                     accessible_description="Open the documentation")
    if state.get("hover"):
        widget.setAttribute(Qt.WA_UnderMouse, True)
    if state.get("down"):
        widget.setDown(True)
    if state.get("disabled"):
        widget.setEnabled(False)
    return widget


def _paint(widget):
    """The RGB of the middle pixel and the number of pixels the dot covers."""
    image = widget.grab().toImage()
    centre = image.pixelColor(image.width() // 2, image.height() // 2)
    background = image.pixelColor(0, 0).rgb()
    area = sum(1 for y in range(image.height()) for x in range(image.width())
               if image.pixelColor(x, y).rgb() != background)
    return (centre.red(), centre.green(), centre.blue()), area


def test_each_dot_state_paints_its_own_colour_and_hover_grows(qapp, subtests):
    """Centre colour per state; disabled wins over pressed; hover covers more pixels."""
    painted = {}
    for case, state, colour in (
            ("normal", {}, (255, 0, 0)),
            ("hover", {"hover": True}, (0, 255, 0)),
            ("pressed", {"down": True}, (0, 0, 255)),
            ("disabled", {"disabled": True}, (255, 255, 0)),
            ("disabled while pressed", {"disabled": True, "down": True},
             (255, 255, 0)),
    ):
        with subtests.test(msg=case):
            painted[case] = _paint(_dot(**state))
            assert painted[case][0] == colour
    with subtests.test(msg="the hovered dot is drawn larger than the resting one"):
        assert painted["hover"][1] > painted["normal"][1] > 0
