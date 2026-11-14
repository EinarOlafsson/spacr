"""The layer viewer's quiet paths — the ones that do nothing, on purpose.

Every branch exercised here is a place where the viewer is asked to react to
something it must leave alone: a depth change before anything has been
painted, a list row naming a layer that has been deleted, a colour the
combo box has never heard of, a cancelled file dialog, a volume that cannot
donate a 2-D spacing, and a hover that must not stamp on the status line.
Each one is a "do nothing" that is only correct because the alternative
(crash, wrong selection, wiped calibration) is what the user would see.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QWidget

from spacr.layers import LayerStack, Spacing
from spacr.qt import layer_viewer as lv


def _sized(qtbot, widget: QWidget, width=420, height=420):
    qtbot.addWidget(widget)
    widget.resize(width, height)
    return widget


def _volume_stack():
    """A z-stack in µm plus a 2-D mask calibrated at 0.65 µm/px."""
    stack = LayerStack()
    volume = np.zeros((4, 30, 30), dtype=np.uint16)
    volume[2] = 2000
    stack.add_image(volume, name="volume",
                    spacing=Spacing.isotropic(3, 2.0, units="um"))
    stack.add_labels(np.zeros((30, 30), dtype=np.int32), name="mask",
                     spacing=Spacing.isotropic(2, 0.65, units="um"))
    return stack


# ---------------------------------------------------------------------------
# The canvas: moving to a slice nobody has drawn yet
# ---------------------------------------------------------------------------

def test_a_slice_chosen_before_the_first_paint_is_still_the_slice_drawn(
        qtbot, qt_theme_applied):
    """A z-slider moved while the page is still hidden must not be discarded.

    ``set_depth`` is what the orthogonal-view sliders call. A screen that
    restores its saved slice in ``__init__`` calls it before the widget has
    ever been painted, so there is no world window yet to move. If the
    depth were only ever pushed into an existing window, that restored
    slice would be silently dropped and the user would come back to plane 0
    instead of the plane they left the page on.
    """
    canvas = _sized(qtbot, lv.LayerCanvas(_volume_stack()), 200, 200)
    canvas.set_plane(("y", "x"), {"z": 0.0})
    assert canvas.canvas is None          # nothing painted yet

    moved = []
    canvas.view_changed.connect(lambda: moved.append(True))

    canvas.set_depth(z=3.0)               # no window to move: remembered only
    assert canvas.canvas is None
    assert canvas._depth == {"z": 3.0}
    assert moved == [True]

    built = canvas._ensure_canvas()       # the first paint honours it
    assert built is not None
    assert built.depth == {"z": 3.0}

    canvas.set_depth(z=1.0)               # now there IS a window to move
    assert canvas.canvas.depth == {"z": 1.0}
    assert moved == [True, True]


# ---------------------------------------------------------------------------
# The layer list: a row that outlived its layer
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The property panel: a colour with no entry in the combo
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The add buttons: a cancelled dialog
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# New layers inherit a calibrated spacing
# ---------------------------------------------------------------------------

def test_a_new_points_layer_takes_its_spacing_past_the_volume(
        qtbot, qt_theme_applied):
    """A points layer added over a z-stack must land in µm, not in pixels.

    ``_default_spacing`` copies the spacing of an existing *2-D* layer, and
    a volume cannot donate one — its spacing has three axes, so a points
    layer built from it would not even be accepted by the stack. The loop
    therefore has to walk past the volume and keep looking. If it stopped at
    the first layer instead, every counted marker dropped on a µm-calibrated
    field would be placed with 1 px steps and would sit in the wrong place
    the moment the field was not 1 µm/px.
    """
    stack = _volume_stack()
    viewer = _sized(qtbot, lv.LayerViewer(stack))
    assert [layer.ndim for layer in stack] == [3, 2]

    spacing = viewer._default_spacing()
    assert spacing.ndim == 2
    assert spacing.scale == (0.65, 0.65)
    assert spacing.units == "um"

    viewer.add_points_button.click()
    assert "points" in stack.names
    assert stack["points"].spacing.scale == (0.65, 0.65)
    assert stack["points"].spacing.units == "um"


def test_with_no_two_d_layer_at_all_the_spacing_falls_back_to_pixels(
        qtbot, qt_theme_applied):
    """A volume-only stack still has to answer, and the answer is 1 px.

    This is the same walk with nothing to find. The fallback is what makes
    "＋ Points" work on a page that is showing only a z-stack; without it
    the loop would fall off the end and the button would raise instead of
    adding a layer.
    """
    stack = LayerStack()
    stack.add_image(np.zeros((4, 20, 20), dtype=np.uint16), name="volume",
                    spacing=Spacing.isotropic(3, 1.0, units="px"))
    viewer = _sized(qtbot, lv.LayerViewer(stack))

    spacing = viewer._default_spacing()
    assert spacing.ndim == 2
    assert spacing.scale == (1.0, 1.0)
    assert spacing.units == "px"


# ---------------------------------------------------------------------------
# Hovering
# ---------------------------------------------------------------------------
