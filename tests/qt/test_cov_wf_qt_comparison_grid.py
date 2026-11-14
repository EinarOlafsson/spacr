"""``B16`` — a comparison grid holding a panel that has nothing to show.

Every panel in :mod:`spacr.qt.comparison_grid` is meant to be on the shared
world window, and the code that keeps them there — ``add_panel``,
``remove_panel``, ``lock_all``, ``reset_view`` — is written on the assumption
that each key names a canvas in the :class:`~spacr.layers.CanvasLink`. A panel
whose stack is empty breaks that assumption: ``LayerCanvas._ensure_canvas``
has no extent to fit, returns ``None``, and the key never enters the link.

That is not a hypothetical. A channel that failed to load, a timepoint whose
file is missing, a panel whose layers were cleared to free memory — all leave
a live widget with an empty stack sitting in the grid. The other panels are
still showing a real comparison, so the grid has to carry the empty one
without raising into a Qt signal handler and without disturbing the shared
window. These tests drive each of those four paths with an unloaded panel
beside a loaded one, so the skip is visible against a working panel rather
than asserted into a vacuum.
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


def _channel(seed=0, *, size=64, label=17):
    """A loaded panel: one image channel with a mask over it."""
    stack = LayerStack()
    rng = np.random.default_rng(seed)
    stack.add_image(rng.integers(0, 4000, (size, size)).astype(np.uint16),
                    name=f"ch{seed}", contrast_limits=(0.0, 4000.0))
    mask = np.zeros((size, size), np.int32)
    mask[20:30, 20:30] = label
    stack.add_labels(mask, name="mask", field=_field_key(), opacity=0.5)
    return stack


def _grid(qtbot, panels, **kwargs):
    grid = cg.ComparisonGrid(panels, **kwargs)
    qtbot.addWidget(grid)
    grid.resize(400, 400)
    grid.show()
    qtbot.waitExposed(grid)
    return grid


def _visible_width(panel):
    """How much of the sample, in world units, this panel is showing."""
    canvas = panel.canvas.canvas
    return canvas.step[1] * canvas.shape[1]


# ---------------------------------------------------------------------------
# Adding a panel with nothing in it
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Removing a panel the link never held
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# "Link all" over a panel the link never held
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# "Fit" over a panel with nothing left to fit
# ---------------------------------------------------------------------------

def test_fit_leaves_the_shared_window_alone_when_no_panel_can_be_fitted(
        qtbot):
    """Fit on a grid whose layers went away must not move the window.

    ``reset_view`` fits the first linked panel and then pushes that window to
    the rest. When a panel's stack has been emptied, the fit produces nothing,
    and the grid has to go on to the next panel rather than push a ``None``.
    With only emptied panels left there is nothing to push at all, and the
    last good window must survive so that reloading the channel comes back to
    where the user was.
    """
    grid = _grid(qtbot, {"a": _channel(0)})
    panel = grid.panels["a"]
    panel.canvas._canvas = panel.canvas.canvas.zoomed(6.0)
    panel.canvas.view_changed.emit()
    assert _visible_width(panel) < 64.0, "the zoom did not take"

    # A panel that CAN be fitted is fitted: the whole 64-unit field is back.
    grid.reset_view()
    assert _visible_width(panel) >= 64.0
    fitted = grid.canvas_link["a"]

    # Now the layers go away and the same button must change nothing.
    panel.stack.clear()
    grid.reset_view()

    after = grid.canvas_link["a"]
    assert (after.origin, after.step, after.shape) == (
        fitted.origin, fitted.step, fitted.shape)
    assert panel.canvas.canvas is None
    assert grid.status.text() == "1 panel(s) · all linked"


def test_fit_on_a_grid_with_no_panels_yet_says_so_instead_of_raising(qtbot):
    """The Fit button exists before anything has been loaded into the grid.

    The comparison window opens empty and the toolbar is live from the first
    frame, so a user can press Fit with nothing in the grid. The status line
    is the only feedback there is, and it has to keep telling the truth
    through the press and afterwards, once panels arrive.
    """
    grid = _grid(qtbot, {})

    grid.reset_view()
    assert grid.status.text() == "No panels"

    grid.add_panel("a", _channel(0))
    grid.reset_view()
    assert grid.status.text() == "1 panel(s) · all linked"
    assert _visible_width(grid.panels["a"]) >= 64.0
