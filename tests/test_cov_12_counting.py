"""Counting-session edges: a blank class, a stack with no plane, an emptied layer.

A counting session is thousands of keystrokes over hours, so the ways it can be
set up wrong have to fail at construction rather than half way through a score:
a class with no name is unclickable, and a stack holding only a volume still has
to give its markers a spacing. The third case is the one that bites a scorer --
undo after the markers were cleared out from under it must be a no-op, not a
crash that loses the rest of the count.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.counting import CountClass, CountingSession
from spacr.layers import LayerError, LayerStack, Spacing


def flat_stack(size=32):
    stack = LayerStack()
    stack.add_image(np.zeros((size, size), np.uint16), name='image',
                    spacing=Spacing.isotropic(2, 1.0, units='px'))
    return stack


def volume_stack():
    """A stack whose only layer is 3-D, so no plane supplies the spacing."""
    stack = LayerStack()
    stack.add_image(np.zeros((4, 32, 32), np.uint16), name='volume',
                    spacing=Spacing.isotropic(3, 1.0, units='um'))
    return stack


def test_a_class_with_a_blank_name_is_refused_at_construction():
    """A nameless class has no legend entry and no shortcut to select it.

    Left to stand it produces a session where one of the tallies cannot be
    named in the export, which is only noticed after the counting is done.
    """
    with pytest.raises(LayerError) as caught:
        CountClass('   ')
    assert 'non-blank name' in str(caught.value)

    with pytest.raises(LayerError):
        CountingSession(flat_stack(), classes=[''])


def test_counting_on_a_volume_still_gets_a_two_dimensional_spacing():
    """A stack with no 2-D layer falls back to isotropic spacing in its units.

    Markers are placed on the displayed plane, so a session opened over a
    z-stack must still have somewhere to put them; inheriting the stack's units
    is what keeps the exported coordinates honest about being microns.
    """
    counting = CountingSession(volume_stack(), size=6.0)
    spacing = counting.layer('infected').spacing
    assert spacing.ndim == 2
    assert spacing.units == 'um'
    assert spacing.scale == (1.0, 1.0)

    index = counting.add({'y': 4.0, 'x': 5.0})
    assert index == 0
    assert counting.counts()['infected'] == 1


def test_undo_after_the_markers_were_emptied_is_a_quiet_no_op():
    """Undoing an add whose marker is already gone reports the class, not a crash.

    The layer can be emptied from the layer list while the session's history
    still remembers the click. Undo has to survive that: a scorer who presses
    undo once too often must not lose the rest of the count to an exception.
    """
    counting = CountingSession(flat_stack(), size=6.0)
    counting.add({'y': 3.0, 'x': 3.0})
    layer = counting.layer('infected')
    layer.data = np.zeros((0, 2), dtype=np.float64)

    assert counting.undo() == ('add', 'infected')
    assert counting.counts()['infected'] == 0
    assert counting.undo() is None


@pytest.mark.parametrize("size", [0, -1, float("nan"), float("inf"), "wide"])
def test_a_marker_diameter_must_be_positive_and_finite(size):
    with pytest.raises(LayerError, match="positive finite"):
        CountingSession(flat_stack(), size=size)


def test_a_negative_shortcut_position_is_refused():
    counting = CountingSession(flat_stack(), classes=[])

    with pytest.raises(LayerError, match="shortcut_index"):
        counting.add_class("infected", shortcut_index=-1)


def test_two_classes_cannot_share_a_nonblank_shortcut():
    counting = CountingSession(
        flat_stack(),
        classes=[CountClass("infected", shortcut="i")],
    )
    distinct = counting.add_class(CountClass("uninfected", shortcut="u"))
    assert distinct.shortcut == "u"

    with pytest.raises(LayerError, match="shortcut.*already selects"):
        counting.add_class(CountClass("mitotic", shortcut="i"))
