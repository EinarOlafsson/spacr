""""No upper limit" must survive Propagate, or the run erases the organelles.

A spin box cannot hold ``None``, so the live preview says "no area cap" with
0. Two filters read an upper area bound and they do NOT agree on what
switches it off:

* ``spacr.utils._filter_objects`` tests ``max_area > 0`` -- 0 disables it.
* ``spacr.object._postprocess_masks`` tests ``max_size is not None`` -- 0
  there means "remove every object bigger than nothing", which is all of
  them.

``organelle_max_area`` goes to the second one. So propagating the preview's 0
into the Mask panel replaced "no cap" with "delete everything", and the run
then wrote empty organelle masks with no error and nothing in the log. The
module ships ``None`` for exactly this reason, and Propagate was overwriting
it.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")


def test_zero_really_would_delete_every_organelle():
    """The consequence, so the rest of this file is not about a style point."""
    from spacr.object import _postprocess_masks

    mask = np.zeros((40, 40), dtype=np.uint16)
    mask[0:5, 0:5] = 1
    mask[10:15, 10:15] = 2
    mask[20:30, 20:30] = 3

    kept_none = _postprocess_masks([mask.copy()], min_size=10, max_size=None)[0]
    kept_zero = _postprocess_masks([mask.copy()], min_size=10, max_size=0)[0]

    assert len(set(np.unique(kept_none)) - {0}) == 3
    assert len(set(np.unique(kept_zero)) - {0}) == 0, (
        "max_size=0 no longer erases everything; re-check what Propagate "
        "should write")


def test_the_preview_propagates_no_cap_as_no_cap(qtbot, qt_theme_applied):
    """The fix: the untouched preview must not change what the run does."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)

    propagated = screen._live_preview.settings_for_propagation()

    assert propagated["organelle_max_area"] is None, (
        "the preview propagated a real cap of 0 px, which deletes every "
        "organelle the run segments")


def test_a_key_whose_reader_treats_zero_as_off_still_propagates_zero(
        qtbot, qt_theme_applied):
    """The rule is per key, not "None everywhere".

    `cell_max_area` reaches `_filter_objects`, where 0 IS the way off. Turning
    that into None would hand `None > 0` to a comparison that raises.
    """
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)

    propagated = screen._live_preview.settings_for_propagation()

    assert propagated["cell_max_area"] == 0


def test_the_rule_is_read_from_the_modules_own_defaults(qtbot,
                                                        qt_theme_applied):
    """Which keys spell "off" as None is not a hand-written list.

    A hand-written one is how the two conventions drifted apart in the first
    place, and a new object type would inherit whichever the author guessed.
    """
    from spacr.qt.widgets.live_preview import LivePreviewPanel
    from spacr.settings import (
        set_default_settings_preprocess_generate_masks as defaults)

    shipped = defaults({})
    off_is_none = LivePreviewPanel._keys_whose_off_is_none()

    assert off_is_none, "no key was recognised, so the rule does nothing"
    for key in off_is_none:
        assert shipped[key] is None, f"{key} does not ship None"
