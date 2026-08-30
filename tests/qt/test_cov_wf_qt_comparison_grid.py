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


def _empty():
    """An unloaded panel: a live stack with nothing in it to fit a canvas to."""
    return LayerStack()


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

def test_a_panel_with_an_empty_stack_is_kept_but_not_put_on_the_link(qtbot):
    """A channel that failed to load must not take the grid down with it.

    ``add_panel`` asks the new panel's canvas to fit itself, and an empty
    stack has no extent to fit — the answer is ``None``. Handing that ``None``
    to ``CanvasLink.add`` would raise ``LayerError`` out of the middle of a
    user's "add this channel" click and lose the panels that DID load. The
    empty panel stays a real widget in the layout; it is only the shared
    world window it is left out of.
    """
    grid = _grid(qtbot, {"loaded": _channel(0)})

    empty_panel = grid.add_panel("unloaded", _empty())

    # The loaded panel went onto the shared window; the empty one did not.
    assert grid.canvas_link.keys == ("loaded",)
    assert grid.canvas_link["loaded"].shape[0] > 0
    # ...and the empty panel is still a panel of this grid, with a caption.
    assert list(grid.panels) == ["loaded", "unloaded"]
    assert grid.panels["unloaded"] is empty_panel
    assert empty_panel.caption.text() == "unloaded"
    assert empty_panel.canvas.canvas is None
    assert grid.status.text() == "2 panel(s) · all linked"


def test_an_unloaded_panel_that_gains_layers_can_join_the_link_later(qtbot):
    """The empty panel is a placeholder, not a dead cell.

    The grid keeps the widget so that the slot for a channel still loading is
    visible in the layout. What makes that worth doing is that filling the
    stack and re-adding the panel puts it on the same world window as the
    others, at the same magnification — which is the whole promise of the
    grid.
    """
    grid = _grid(qtbot, {"loaded": _channel(0)})
    grid.add_panel("late", _empty())
    assert "late" not in grid.canvas_link, "an empty stack has nothing to fit"

    grid.remove_panel("late")
    grid.add_panel("late", _channel(1))

    assert grid.canvas_link.keys == ("loaded", "late")
    assert grid.canvas_link["late"].step == pytest.approx(
        grid.canvas_link["loaded"].step), (
        "the late panel came in at a different magnification")
    assert grid.canvas_link["late"].origin == pytest.approx(
        grid.canvas_link["loaded"].origin)


# ---------------------------------------------------------------------------
# Removing a panel the link never held
# ---------------------------------------------------------------------------

def test_removing_an_unloaded_panel_does_not_ask_the_link_for_it(qtbot):
    """Closing the cell of a channel that never loaded must still close it.

    ``CanvasLink.remove`` raises ``LayerError`` for a key it does not hold, so
    the grid checks first. Without the check, the one panel a user is most
    likely to want rid of — the one showing nothing — would be the one that
    could not be removed.
    """
    grid = _grid(qtbot, {"loaded": _channel(0)})
    unloaded = grid.add_panel("unloaded", _empty())

    returned = grid.remove_panel("unloaded")

    assert returned is unloaded
    assert list(grid.panels) == ["loaded"]
    assert grid.canvas_link.keys == ("loaded",)
    assert grid.status.text() == "1 panel(s) · all linked"

    # And a panel the link DOES hold leaves the link when it is removed, so
    # the guard above is a narrow skip rather than a removal that never runs.
    grid.remove_panel("loaded")
    assert grid.canvas_link.keys == ()
    assert grid.status.text() == "No panels"


# ---------------------------------------------------------------------------
# "Link all" over a panel the link never held
# ---------------------------------------------------------------------------

def test_link_all_relinks_the_loaded_panels_and_steps_over_the_empty_one(
        qtbot):
    """"Link all" is the way back from an experiment, and must always work.

    A user unlocks a panel, pans it somewhere, then presses "Link all" to put
    everything back together. If the button raised on a panel the link does
    not hold, the recovery gesture would be broken precisely in the grid where
    something already went wrong — one channel failed to load.
    """
    grid = _grid(qtbot, {"a": _channel(0), "b": _channel(1)})
    grid.add_panel("unloaded", _empty())
    grid.panels["b"].lock_box.setChecked(False)
    free = grid.panels["b"].canvas
    free._canvas = free.canvas.panned(20, 20)
    free.view_changed.emit()
    assert grid.canvas_link["b"].origin != pytest.approx(
        grid.canvas_link["a"].origin), "the free panel did not move away"

    grid.lock_all()

    assert grid.canvas_link.keys == ("a", "b")
    assert grid.canvas_link["b"].origin == pytest.approx(
        grid.canvas_link["a"].origin)
    assert grid.canvas_link.is_locked("b") is True
    assert grid.panels["unloaded"].lock_box.isChecked() is True
    assert grid.status.text() == "3 panel(s) · all linked"


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


def test_fit_skips_a_panel_the_user_unlocked_and_fits_the_next_one(qtbot):
    """Unlocking is how a user studies one panel closely; Fit must respect it.

    ``reset_view`` walks the panels and fits the first one that is both in the
    link and locked. An unlocked panel is passed over — otherwise the button
    that tidies the linked panels would also yank the one panel the user
    deliberately parked somewhere else.
    """
    grid = _grid(qtbot, {"free": _channel(0), "linked": _channel(1)})
    grid.panels["free"].lock_box.setChecked(False)
    free_canvas = grid.panels["free"].canvas
    free_canvas._canvas = free_canvas.canvas.zoomed(6.0)
    free_canvas.view_changed.emit()
    parked = grid.canvas_link["free"]
    grid.panels["linked"].canvas._canvas = (
        grid.panels["linked"].canvas.canvas.zoomed(6.0))
    grid.panels["linked"].canvas.view_changed.emit()

    grid.reset_view()

    assert (grid.canvas_link["free"].origin,
            grid.canvas_link["free"].step) == (parked.origin, parked.step)
    assert _visible_width(grid.panels["linked"]) >= 64.0
    assert "1 free (free)" in grid.status.text()
