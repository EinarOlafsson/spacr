"""The comparison grid answers rather than crashes when a panel is unknown.

Four views of one field are only one observation if they stay on the same
world window and the same object. Each guard here is a place where the grid
is asked about something it does not hold -- a panel with no canvas in the
link, a click on labels that were never told which field they segment -- and
the answer has to be "nothing to do", because the other panels are still
showing a live comparison.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.layers import FieldKey, LayerStack
from spacr.qt import comparison_grid as cg

PLATE_KEY_VALUES = ("plate1", "A", "1", "1")


def _field_key():
    return FieldKey(values=dict(zip(FieldKey.columns(), PLATE_KEY_VALUES)))


def _channel(seed=0, *, field=True, size=64, label=17):
    stack = LayerStack()
    rng = np.random.default_rng(seed)
    stack.add_image(rng.integers(0, 4000, (size, size)).astype(np.uint16),
                    name=f"ch{seed}", contrast_limits=(0.0, 4000.0))
    mask = np.zeros((size, size), np.int32)
    mask[20:30, 20:30] = label
    stack.add_labels(mask, name="mask",
                     field=_field_key() if field else None, opacity=0.5)
    return stack


def _grid(qtbot, panels=None, **kwargs):
    grid = cg.ComparisonGrid(
        panels if panels is not None
        else {f"ch{i}": _channel(i) for i in range(2)}, **kwargs)
    qtbot.addWidget(grid)
    grid.resize(400, 400)
    grid.show()
    qtbot.waitExposed(grid)
    return grid


# ---------------------------------------------------------------------------
# A panel names itself
# ---------------------------------------------------------------------------

def test_a_panel_knows_the_name_it_has_in_the_link(qtbot):
    """The key is how every message about a panel is addressed.

    View changes, lock toggles and picks all travel as ``(key, ...)``, so a
    panel that could not say its own name would be unaddressable.
    """
    panel = cg.ComparisonPanel("dapi", _channel(0))
    qtbot.addWidget(panel)

    assert panel.key == "dapi"


# ---------------------------------------------------------------------------
# A grid with no panels
# ---------------------------------------------------------------------------

def test_a_grid_can_open_with_no_panels_at_all(qtbot):
    """The window exists before anything is loaded into it."""
    grid = cg.ComparisonGrid(None)
    qtbot.addWidget(grid)

    assert grid.panels == {}
    assert grid.status.text() == "No panels"


# ---------------------------------------------------------------------------
# A panel the link does not hold
# ---------------------------------------------------------------------------

def test_locking_a_panel_the_link_does_not_hold_changes_nothing(qtbot):
    """A panel with no canvas in the link cannot be put on the shared window.

    Its checkbox still moves -- it is a live widget -- and the grid must
    treat the toggle as a no-op rather than raising into the signal.
    """
    grid = _grid(qtbot)
    grid.canvas_link.remove("ch1")
    before = grid.status.text()

    grid.panels["ch1"].lock_box.setChecked(False)
    grid.panels["ch1"].lock_box.setChecked(True)

    assert grid.status.text() == before
    assert "ch1" not in grid.canvas_link


def test_fitting_skips_panels_that_are_not_on_the_shared_window(qtbot):
    """Fit puts the LINKED panels back; an unlocked one keeps its own view.

    Unlocking a panel exists so a user can look closely at it without losing
    where the others are; Fit must not undo that.
    """
    grid = _grid(qtbot)
    grid.panels["ch0"].lock_box.setChecked(False)
    before = grid.canvas_link["ch0"]

    grid.reset_view()

    after = grid.canvas_link["ch0"]
    assert grid.canvas_link.is_locked("ch0") is False
    assert (after.origin, after.step) == (before.origin, before.step)


# ---------------------------------------------------------------------------
# A click that names no object
# ---------------------------------------------------------------------------

def test_clicking_labels_that_name_no_field_publishes_nothing(qtbot):
    """Labels with no field key cannot be named in measurement-table terms.

    Publishing the raw label number instead would highlight whichever object
    happened to carry that number in another view -- a different cell.
    """
    grid = _grid(qtbot, {"ch0": _channel(0, field=False),
                         "ch1": _channel(1, field=False)})
    published = []
    grid.object_picked.connect(published.append)

    layer = [lay for lay in grid.panels["ch0"].stack
             if lay.name == "mask"][0]
    grid._on_panel_picked("ch0", layer, {"y": 25.0, "x": 25.0}, 17)

    assert published == []
    assert layer.selected_label in (None, 0)
