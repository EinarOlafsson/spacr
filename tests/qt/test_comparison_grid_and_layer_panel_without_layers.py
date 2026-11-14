"""Table-driven tests of the layer views when a ``LayerStack`` has nothing to show.

A channel that failed to load, a timepoint whose file is missing, a panel
whose layers were cleared: each leaves a live widget over an empty stack.  The
comparison grid (``spacr.qt.comparison_grid``) must keep such a panel without
putting it on the shared ``CanvasLink`` and without disturbing the loaded
panels beside it; the layer panel (``spacr.qt.layer_viewer``) must treat every
control edit as a no-op; and a pick on labels that name no field must publish
nothing.  Every case runs against a loaded panel so the skip is visible.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.layers import FieldKey, LayerStack  # noqa: E402
from spacr.qt import comparison_grid as cg  # noqa: E402
from spacr.qt import layer_viewer as lv  # noqa: E402

pytestmark = pytest.mark.qt

PLATE_KEY_VALUES = ("plate1", "A", "1", "1")


def _channel(seed=0, *, field=True, size=64, label=17):
    """A loaded panel: one image channel with a mask over it."""
    stack = LayerStack()
    rng = np.random.default_rng(seed)
    stack.add_image(rng.integers(0, 4000, (size, size)).astype(np.uint16),
                    name=f"ch{seed}", contrast_limits=(0.0, 4000.0))
    mask = np.zeros((size, size), np.int32)
    mask[20:30, 20:30] = label
    key = FieldKey(values=dict(zip(FieldKey.columns(), PLATE_KEY_VALUES)))
    stack.add_labels(mask, name="mask", field=key if field else None,
                     opacity=0.5)
    return stack


def _grid(qtbot, panels):
    grid = cg.ComparisonGrid(panels)
    qtbot.addWidget(grid)
    grid.resize(400, 400)
    grid.show()
    qtbot.waitExposed(grid)
    return grid


def _visible_width(panel):
    """How much of the sample, in world units, this panel is showing."""
    canvas = panel.canvas.canvas
    return canvas.step[1] * canvas.shape[1]


def _zoom(panel):
    panel.canvas._canvas = panel.canvas.canvas.zoomed(6.0)
    panel.canvas.view_changed.emit()


def test_the_grid_carries_an_unloaded_panel_without_linking_it(qtbot, subtests):
    """Add, refill, remove, link-all and fit all step over the empty panel."""
    with subtests.test(msg="added_but_not_linked"):
        grid = _grid(qtbot, {"loaded": _channel(0)})
        empty_panel = grid.add_panel("unloaded", LayerStack())
        assert grid.canvas_link.keys == ("loaded",)
        assert grid.canvas_link["loaded"].shape[0] > 0
        assert list(grid.panels) == ["loaded", "unloaded"]
        assert grid.panels["unloaded"] is empty_panel
        assert empty_panel.caption.text() == "unloaded"
        assert empty_panel.canvas.canvas is None
        assert grid.status.text() == "2 panel(s) · 1 not linked (unloaded)"

    with subtests.test(msg="refilled_panel_joins_at_the_same_window"):
        grid = _grid(qtbot, {"loaded": _channel(0)})
        grid.add_panel("late", LayerStack())
        assert "late" not in grid.canvas_link
        grid.remove_panel("late")
        grid.add_panel("late", _channel(1))
        assert grid.canvas_link.keys == ("loaded", "late")
        assert grid.canvas_link["late"].step == pytest.approx(
            grid.canvas_link["loaded"].step)
        assert grid.canvas_link["late"].origin == pytest.approx(
            grid.canvas_link["loaded"].origin)

    with subtests.test(msg="removing_the_unloaded_panel"):
        grid = _grid(qtbot, {"loaded": _channel(0)})
        unloaded = grid.add_panel("unloaded", LayerStack())
        assert grid.remove_panel("unloaded") is unloaded
        assert list(grid.panels) == ["loaded"]
        assert grid.canvas_link.keys == ("loaded",)
        assert grid.status.text() == "1 panel(s) · all linked"
        grid.remove_panel("loaded")
        assert grid.canvas_link.keys == ()
        assert grid.status.text() == "No panels"

    with subtests.test(msg="link_all_relinks_and_steps_over_the_empty_one"):
        grid = _grid(qtbot, {"a": _channel(0), "b": _channel(1)})
        grid.add_panel("unloaded", LayerStack())
        grid.panels["b"].lock_box.setChecked(False)
        free = grid.panels["b"].canvas
        free._canvas = free.canvas.panned(20, 20)
        free.view_changed.emit()
        assert grid.canvas_link["b"].origin != pytest.approx(
            grid.canvas_link["a"].origin)
        grid.lock_all()
        assert grid.canvas_link.keys == ("a", "b")
        assert grid.canvas_link["b"].origin == pytest.approx(
            grid.canvas_link["a"].origin)
        assert grid.canvas_link.is_locked("b") is True
        assert grid.panels["unloaded"].lock_box.isChecked() is True
        assert grid.status.text() == "3 panel(s) · 1 not linked (unloaded)"

    with subtests.test(msg="fit_skips_the_unlocked_panel"):
        grid = _grid(qtbot, {"free": _channel(0), "linked": _channel(1)})
        grid.panels["free"].lock_box.setChecked(False)
        _zoom(grid.panels["free"])
        parked = grid.canvas_link["free"]
        _zoom(grid.panels["linked"])
        assert _visible_width(grid.panels["linked"]) < 64.0
        grid.reset_view()
        assert (grid.canvas_link["free"].origin,
                grid.canvas_link["free"].step) == (parked.origin, parked.step)
        assert _visible_width(grid.panels["linked"]) >= 64.0
        assert "1 free (free)" in grid.status.text()

    with subtests.test(msg="a_pick_on_labels_without_a_field"):
        grid = _grid(qtbot, {"ch0": _channel(0, field=False),
                             "ch1": _channel(1, field=False)})
        published = []
        grid.object_picked.connect(published.append)
        layer = [lay for lay in grid.panels["ch0"].stack
                 if lay.name == "mask"][0]
        grid._on_panel_picked("ch0", layer, {"y": 25.0, "x": 25.0}, 17)
        assert published == []
        assert layer.selected_label in (None, 0)


def test_with_no_layer_every_layer_panel_edit_is_a_no_op(qtbot):
    """Each control handler on an empty stack changes nothing and stays disabled."""
    stack = LayerStack()
    panel = lv._LayerPanel(stack)
    qtbot.addWidget(panel)
    moved = []

    panel._on_opacity(30)
    panel._on_blending(1)
    panel._on_colormap("gray")
    panel._on_limits()
    panel._on_auto()
    panel._on_labels_display(1)
    panel._reorder(moved.append)

    assert moved == []
    assert stack.selected is None
    assert len(stack) == 0
    assert not panel.opacity_slider.isEnabled()
    assert not panel.raise_button.isEnabled()
