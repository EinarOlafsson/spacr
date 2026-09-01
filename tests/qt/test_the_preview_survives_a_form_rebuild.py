"""A shaping edit must not empty the live preview.

Reported 2026-09-01: "the live preview loaded images are gone after every time
i put in a number for an object channel because doing this reloads the module
with the new settings for the unlocked object."

That is exactly what happens. `_rebuild_the_form` asks the window to build the
screen AGAIN -- deliberately, because which settings a form holds depends on a
few of its own values -- and `rebuild_app_screen` carries the user's VALUES
across. It did not carry the live preview's loaded image, and the preview lives
on the screen being replaced.

The cost is not an inconvenience: setting the channels while watching the
result is the task the preview exists for, and the channel numbers are the
values that trigger the rebuild.
"""
from __future__ import annotations

import numpy as np
import pytest


class _Panel:
    """The parts of LivePreviewPanel this transfer touches."""

    def __init__(self):
        self._image = None
        self._image_path = None
        self._path_full = ""
        self._settings = {}
        self.redrawn = False
        self.table_rebuilt = False

    def _show_elided_path(self):
        pass

    def _refresh_canvases(self):
        self.redrawn = True

    def _refresh_source_selectors(self):
        self.table_rebuilt = True


class _Screen:
    def __init__(self, panel=None):
        self._live_preview = panel


def test_the_loaded_image_moves_to_the_replacement_screen():
    from spacr.qt.app import _carry_preview_state

    image = np.zeros((4, 4), dtype=np.uint8)
    old = _Screen(_Panel())
    old._live_preview._image = image
    old._live_preview._path_full = "/data/plate1/field.tif"

    fresh = _Screen(_Panel())
    _carry_preview_state(old, fresh)

    assert fresh._live_preview._image is image
    assert fresh._live_preview._path_full == "/data/plate1/field.tif"


def test_the_replacement_is_redrawn_so_the_image_is_actually_shown():
    """Carrying the array without redrawing leaves a blank canvas holding an
    image, which looks identical to having lost it."""
    from spacr.qt.app import _carry_preview_state

    old = _Screen(_Panel())
    old._live_preview._image = np.zeros((4, 4), dtype=np.uint8)
    fresh = _Screen(_Panel())

    _carry_preview_state(old, fresh)
    assert fresh._live_preview.redrawn is True


def test_the_set_table_is_rebuilt_not_only_the_canvas():
    """Carrying the image alone left a picture above an EMPTY table.

    The table is how a field is chosen, so the preview looked loaded and could
    not be driven -- and `_image_path` came across too, so the panel read as
    already-loaded and pressing Choose appeared to do nothing. Reported
    2026-09-01 as "the images are now still there after reloading but the
    table is gone. and pressing import again does nothing".
    """
    from spacr.qt.app import _carry_preview_state

    old = _Screen(_Panel())
    old._live_preview._image = np.zeros((4, 4), dtype=np.uint8)
    fresh = _Screen(_Panel())

    _carry_preview_state(old, fresh)
    assert fresh._live_preview.table_rebuilt is True, (
        "the image came across but its set table did not")


def test_nothing_loaded_means_nothing_carried_and_no_redraw():
    from spacr.qt.app import _carry_preview_state

    fresh = _Screen(_Panel())
    _carry_preview_state(_Screen(_Panel()), fresh)

    assert fresh._live_preview._image is None
    assert fresh._live_preview.redrawn is False


def test_a_screen_without_a_preview_is_not_an_error():
    """Most modules have no live preview at all, and every one of them goes
    through this rebuild."""
    from spacr.qt.app import _carry_preview_state

    _carry_preview_state(_Screen(None), _Screen(None))
    _carry_preview_state(None, _Screen(_Panel()))


def test_a_panel_that_raises_does_not_take_the_rebuild_down():
    """Costing the user their whole screen to save them a re-load is the
    wrong trade."""
    from spacr.qt.app import _carry_preview_state

    class Exploding(_Panel):
        def _refresh_canvases(self):
            raise RuntimeError("boom")

    old = _Screen(_Panel())
    old._live_preview._image = np.zeros((2, 2), dtype=np.uint8)
    _carry_preview_state(old, _Screen(Exploding()))


def test_the_rebuild_actually_calls_the_carry():
    """The wiring, not the mechanism.

    Every test above drives `_carry_preview_state` directly, so all five pass
    with the CALL removed from `rebuild_app_screen` -- a mutation confirmed
    that, and a helper nothing calls fixes nothing. Driving the real rebuild
    needs a MainWindow and a built screen, which is far more machinery than
    this is worth, so the call site is asserted at source level instead.

    Weaker than driving it, and stated as such: this catches the call being
    deleted, not the call being made at the wrong moment.
    """
    import inspect

    from spacr.qt.app import MainWindow

    source = inspect.getsource(MainWindow.rebuild_app_screen)
    assert "_carry_preview_state(old, fresh)" in source, (
        "rebuild_app_screen no longer carries the preview state, so a shaping "
        "edit empties the live preview again")
