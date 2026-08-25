"""The documentation dot paints a different colour in each of its states.

The dot IS the control -- there is no border, no label and no platform icon
behind it. If disabled, pressed and hovered all painted the same fill, the
only feedback the control has would be gone: a user could not tell a link
they may follow from one they may not.
"""
from __future__ import annotations

from PySide6.QtCore import Qt

from spacr.qt.widgets.dot_link import DotLink

#: normal, hover, pressed, disabled -- deliberately far apart so the sampled
#: centre pixel identifies the state unambiguously.
_COLOURS = ("#ff0000", "#00ff00", "#0000ff", "#ffff00")


def _dot(qapp):
    return DotLink(tooltip="What this step does",
                   colours=_COLOURS,
                   accessible_description="Open the documentation")


def _centre_pixel(widget):
    """Paint the widget and return the RGB of its middle pixel."""
    image = widget.grab().toImage()
    colour = image.pixelColor(image.width() // 2, image.height() // 2)
    return colour.red(), colour.green(), colour.blue()


def _painted_area(widget):
    """How many pixels of the widget the dot actually covers."""
    image = widget.grab().toImage()
    background = image.pixelColor(0, 0).rgb()
    return sum(1
               for y in range(image.height())
               for x in range(image.width())
               if image.pixelColor(x, y).rgb() != background)


def test_each_state_paints_its_own_colour(qapp):
    """Normal, hover, pressed and disabled are four visibly different dots."""
    resting = _dot(qapp)

    hovered = _dot(qapp)
    hovered.setAttribute(Qt.WA_UnderMouse, True)

    pressed = _dot(qapp)
    pressed.setDown(True)

    disabled = _dot(qapp)
    disabled.setEnabled(False)

    seen = {
        "normal": _centre_pixel(resting),
        "hover": _centre_pixel(hovered),
        "pressed": _centre_pixel(pressed),
        "disabled": _centre_pixel(disabled),
    }
    assert seen["normal"] == (255, 0, 0), seen
    assert seen["hover"] == (0, 255, 0), seen
    assert seen["pressed"] == (0, 0, 255), seen
    assert seen["disabled"] == (255, 255, 0), seen


def test_a_disabled_dot_stays_disabled_even_while_pressed(qapp):
    """Availability outranks press state, so a dead link cannot look live."""
    widget = _dot(qapp)
    widget.setEnabled(False)
    widget.setDown(True)
    assert _centre_pixel(widget) == (255, 255, 0)


def test_the_hovered_dot_is_drawn_larger_than_the_resting_one(qapp):
    """Growing under the pointer is what makes a 7px mark findable."""
    resting = _dot(qapp)
    hovered = _dot(qapp)
    hovered.setAttribute(Qt.WA_UnderMouse, True)
    assert _painted_area(hovered) > _painted_area(resting)
