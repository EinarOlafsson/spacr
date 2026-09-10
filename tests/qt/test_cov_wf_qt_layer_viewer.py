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

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from spacr.layers import LayerStack, Spacing
from spacr.qt import layer_viewer as lv


def _sized(qtbot, widget: QWidget, width=420, height=420):
    qtbot.addWidget(widget)
    widget.resize(width, height)
    return widget


def _flat_stack():
    """Two ordinary 2-D layers, the shape every screen in spaCR starts from."""
    stack = LayerStack()
    image = np.full((40, 40), 1000, dtype=np.uint16)
    image[10:20, 10:20] = 4000
    stack.add_image(image, name="image", contrast_limits=(0.0, 4095.0))
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[10:20, 10:20] = 3
    stack.add_labels(mask, name="mask")
    return stack


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

def test_clicking_a_row_whose_layer_is_gone_keeps_the_real_selection(
        qtbot, qt_theme_applied):
    """A stale row must not blank the selection the user still has.

    Rows carry the layer's name, not the layer. Anything that renames or
    removes a layer between a repaint and a click — a background load, an
    undo — leaves a row pointing at a name the stack no longer holds.
    Selecting it must be inert: if it instead propagated ``None`` into
    ``stack.select`` the property panel would grey out and the Fit/remove
    buttons would disable under a user who never asked for that.
    """
    stack = _flat_stack()
    viewer = _sized(qtbot, lv.LayerViewer(stack))
    layer_list = viewer.layer_list
    assert [layer_list.item(i).text() for i in range(layer_list.count())] == \
        ["mask", "image"]

    layer_list.setCurrentRow(0)                       # a live row: it works
    assert stack.selected.name == "mask"
    assert viewer.status.text().startswith("mask (labels)")

    stale = layer_list.item(1)                        # the row goes stale
    stale.setData(Qt.UserRole, "a layer that was removed")
    assert layer_list._layer_for(stale) is None
    layer_list.setCurrentRow(1)

    assert stack.selected.name == "mask"              # untouched
    assert viewer.status.text().startswith("mask (labels)")
    assert viewer.remove_button.isEnabled()


# ---------------------------------------------------------------------------
# The property panel: a colour with no entry in the combo
# ---------------------------------------------------------------------------

def test_a_custom_colour_is_shown_without_recolouring_it(qtbot,
                                                         qt_theme_applied):
    """Selecting a hex-coloured channel must show its truthful colour.

    ``ImageLayer`` accepts any colour spec, so a channel can legitimately be
    a ``#rrggbb`` ramp that the combo box — filled only from the built-in
    ``COLORMAPS`` — did not previously display. The selected layer must not
    inherit the stale text from another layer, and synchronising the combo
    must not emit the signal that rewrites the custom colour.
    """
    stack = LayerStack()
    stack.add_image(np.full((20, 20), 500, dtype=np.uint16),
                    name="image", colormaps="#ff00aa")
    viewer = _sized(qtbot, lv.LayerViewer(stack))
    assert stack["image"].colormap.name == "#ff00aa"
    assert viewer.colormap_combo.findText("#ff00aa") >= 0
    assert viewer.colormap_combo.currentText() == "#ff00aa"
    assert stack["image"].colormap.name == "#ff00aa"   # not rewritten

    stack["image"].colormap = "magenta"                # a colour it CAN show
    stack.select(None)
    stack.select(stack["image"])
    assert viewer.colormap_combo.currentText() == "magenta"
    assert viewer.colormap_combo.findText("#ff00aa") == -1


# ---------------------------------------------------------------------------
# The add buttons: a cancelled dialog
# ---------------------------------------------------------------------------

def test_cancelling_the_mask_dialog_adds_no_layer(qtbot, qt_theme_applied,
                                                  tmp_path, monkeypatch):
    """Pressing Escape in the mask chooser must leave the stack as it was.

    ``QFileDialog.getOpenFileName`` returns ``("", "")`` when the user
    cancels. Handing that empty string on to the loader would push a
    ``FileNotFoundError`` through ``add_labels_file``, which catches it and
    writes "Could not load " into the status line — an error message for an
    action the user deliberately abandoned, on a screen that now claims
    something went wrong.
    """
    pytest.importorskip("PIL")
    from PIL import Image

    mask_png = tmp_path / "mask.png"
    labels = np.zeros((16, 16), dtype=np.uint8)
    labels[4:9, 4:9] = 7
    Image.fromarray(labels).save(mask_png)

    viewer = _sized(qtbot, lv.LayerViewer())
    chosen = [str(mask_png)]
    monkeypatch.setattr(lv.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (chosen[0], "")))

    viewer.add_mask_button.click()                     # the user picks a file
    assert viewer.stack.names == ("mask",)
    assert int(viewer.stack["mask"].data.max()) == 7

    chosen[0] = ""                                     # the user cancels
    viewer.add_mask_button.click()
    assert viewer.stack.names == ("mask",)
    assert "Could not load" not in viewer.status.text()


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

def test_hovering_does_not_overwrite_the_selected_layer_s_description(
        qtbot, qt_theme_applied):
    """The status line belongs to the selection once there is one.

    There is one status label and two things want it: the cursor's world
    position, and the description of the layer being edited. The rule is
    that a selection wins, because the description carries the opacity and
    the spacing the user is in the middle of adjusting — if every mouse move
    across the canvas replaced it with a coordinate pair, that readout would
    be unreadable for as long as the pointer was over the picture.
    """
    stack = _flat_stack()
    viewer = _sized(qtbot, lv.LayerViewer(stack))

    stack.select(None)                          # nothing selected: coordinates
    assert viewer.status.text() == "No layer selected"
    viewer.canvas.hovered.emit({"y": 12.5, "x": 4.0})
    assert viewer.status.text() == "y 12.5 · x 4"

    stack.select(stack["image"])                # selected: the description
    described = viewer.status.text()
    assert described.startswith("image (image)")
    viewer.canvas.hovered.emit({"y": 31.25, "x": 9.0})
    assert viewer.status.text() == described
    assert "31.25" not in viewer.status.text()
