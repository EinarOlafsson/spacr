"""Fit on the comparison grid: which panel it fits, and when it leaves the window alone.

``ComparisonGrid.reset_view`` fits the first panel that is on the shared
window and locked, pushes that window to the others, and otherwise changes
nothing; the status line has to stay truthful through every case.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.layers import FieldKey, LayerStack
from spacr.qt import comparison_grid as cg

pytestmark = pytest.mark.qt


def _channel(seed=0, *, size=64, label=17):
    """A loaded panel: one image channel with a field-keyed mask over it."""
    stack = LayerStack()
    rng = np.random.default_rng(seed)
    stack.add_image(rng.integers(0, 4000, (size, size)).astype(np.uint16),
                    name=f"ch{seed}", contrast_limits=(0.0, 4000.0))
    mask = np.zeros((size, size), np.int32)
    mask[20:30, 20:30] = label
    field = FieldKey(values=dict(zip(FieldKey.columns(),
                                     ("plate1", "A", "1", "1"))))
    stack.add_labels(mask, name="mask", field=field, opacity=0.5)
    return stack


def _grid(qtbot, panels):
    grid = cg.ComparisonGrid(panels)
    qtbot.addWidget(grid)
    grid.resize(400, 400)
    grid.show()
    qtbot.waitExposed(grid)
    return grid


def _window(grid, key):
    canvas = grid.canvas_link[key]
    return (canvas.origin, canvas.step, canvas.shape)


def _zoom(grid, key, factor=6.0):
    canvas = grid.panels[key].canvas
    canvas._canvas = canvas.canvas.zoomed(factor)
    canvas.view_changed.emit()


def _visible_width(grid, key):
    canvas = grid.panels[key].canvas.canvas
    return canvas.step[1] * canvas.shape[1]


def test_reset_view_fits_the_first_locked_linked_panel_only(subtests, qtbot):
    """Empty grids, emptied panels and unlocked panels are passed over."""
    with subtests.test(msg="no panels yet: says so, then fits once one arrives"):
        grid = _grid(qtbot, {})
        grid.reset_view()
        assert grid.status.text() == "No panels"
        grid.add_panel("a", _channel(0))
        _zoom(grid, "a")
        assert _visible_width(grid, "a") < 64.0
        grid.reset_view()
        assert grid.status.text() == "1 panel(s) · all linked"
        assert _visible_width(grid, "a") >= 64.0

    with subtests.test(msg="an emptied panel leaves the last window in place"):
        grid = _grid(qtbot, {"a": _channel(0)})
        _zoom(grid, "a")
        zoomed = _window(grid, "a")
        grid.reset_view()
        fitted = _window(grid, "a")
        assert fitted != zoomed
        assert _visible_width(grid, "a") >= 64.0
        grid.panels["a"].stack.clear()
        grid.reset_view()
        assert _window(grid, "a") == fitted
        assert grid.panels["a"].canvas.canvas is None
        assert grid.status.text() == "1 panel(s) · all linked"

    with subtests.test(msg="an unlocked panel keeps its own view"):
        grid = _grid(qtbot, {"ch0": _channel(0), "ch1": _channel(1)})
        grid.panels["ch0"].lock_box.setChecked(False)
        _zoom(grid, "ch0")
        parked = _window(grid, "ch0")
        _zoom(grid, "ch1")
        assert _visible_width(grid, "ch1") < 64.0
        grid.reset_view()
        assert grid.canvas_link.is_locked("ch0") is False
        assert _window(grid, "ch0") == parked
        assert _visible_width(grid, "ch1") >= 64.0
        assert grid.status.text() == "2 panel(s) · 1 free (ch0)"

    with subtests.test(msg="a panel missing from the link is skipped"):
        grid = _grid(qtbot, {"ch0": _channel(0), "ch1": _channel(1)})
        grid.canvas_link.remove("ch0")
        _zoom(grid, "ch1")
        grid.reset_view()
        assert "ch0" not in grid.canvas_link
        assert _visible_width(grid, "ch1") >= 64.0
        assert grid.status.text() == "2 panel(s) · 1 not linked (ch0)"
