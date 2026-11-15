"""The spaceout ink solver's value-ramp guard on a hue with no zero channel.

The ink solver's general contract (argmax chroma in the band, the plain hue
shift for an empty band, the strictly rising value ramp) is pinned in
``tests/qt/test_theme_sheets_and_inks.py``; this file keeps the one guard case
a real HSV hue cannot reach.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt import theme                         # noqa: E402

pytestmark = pytest.mark.qt



def test_the_value_ramp_guard_keeps_the_first_of_equally_coloured_candidates(
        monkeypatch):
    """A hue with no zero channel (not a real HSV hue, but what a user theme or
    a half-applied dressing can hand ``_hue_rgb``) makes the value ramp
    non-strict: level 2 is no more coloured than level 1.  The guard then
    keeps the earlier candidate, and the tint ramp supplies the answer."""
    monkeypatch.setattr(theme, "_hue_rgb", lambda hue, saturation=1.0: (1.0, 0.5, 0.5))
    ink = theme._hue_ink.__wrapped__(10.0, 0.0, 1.0, "#123456")
    assert ink == "#fd7e7e"
    values = [int(ink[i:i + 2], 16) for i in (1, 3, 5)]
    assert max(values) - min(values) == 0xfd - 0x7e
