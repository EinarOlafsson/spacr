"""The field ring mask, and loading an animation that cannot be shown.

The ring is the field's stroke alone (part of the chrome, never the outside
corner); a well too small for an inside is the whole rounded box.  A missing,
corrupt or frameless animation degrades to "no animation" (None), and a
failed read is cached until `clear_cache`.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.qt.widgets import animation_zoom as az

SIDE = az.SOURCE_SIZE


def test_field_ring_mask_is_the_stroke_or_the_whole_box_when_there_is_no_inside(
        subtests):
    """Shape, dtype, what is inside the ring and its place in the chrome."""
    with subtests.test(msg="the packaged field: a stroke inside the chrome"):
        ring = az.field_ring_mask(SIDE)
        chrome = az.chrome_mask(SIDE)
        assert (ring.shape, ring.dtype) == ((SIDE, SIDE), np.dtype(bool))
        assert bool(ring.any()) is True
        assert (bool(ring[0, 0]), bool(chrome[0, 0])) == (False, True)
        assert bool(((chrome | ring) == chrome).all()) is True
        assert bool(ring[SIDE // 2, SIDE // 2]) is False

    for case, size, box, radius, centre, expected in (
            ("a normal well is a ring with a hole", 64,
             (8.0, 8.0, 56.0, 56.0), 6.0, 32, False),
            ("a box narrower than the padding has no hole", 32,
             (14.0, 14.0, 18.0, 18.0), 1.0, 16, True),
    ):
        with subtests.test(msg=case):
            ring = az.field_ring_mask(size, box=box, radius=radius)
            assert (ring.shape, ring.dtype) == ((size, size), np.dtype(bool))
            assert bool(ring[0, 0]) is False
            assert bool(ring[centre, centre]) is expected

    with subtests.test(msg="the tiny box is filled solid"):
        ring = az.field_ring_mask(32, box=(14.0, 14.0, 18.0, 18.0), radius=1.0)
        assert bool(ring[14:18, 14:18].all()) is True


def test_an_animation_that_cannot_be_read_is_none_and_cached(tmp_path,
                                                             monkeypatch,
                                                             subtests):
    """zoomed_animation answers None for absent, corrupt and frameless files, once per cache."""
    az.clear_cache()
    try:
        with subtests.test(msg="an absent and a corrupt file"):
            assert az.zoomed_animation(str(tmp_path / "absent.gif"), 64) is None
            broken = tmp_path / "broken.gif"
            broken.write_bytes(b"GIF89a-not-really")
            assert az.zoomed_animation(str(broken), 64) is None

        with subtests.test(msg="a file that decodes to no frames"):
            with monkeypatch.context() as patch:
                patch.setattr(az, "read_frames", lambda path: ((), ()))
                assert az.source_content_extent("anything.gif") == 0.0
                assert az.zoomed_animation("no-such-animation.gif", 60) is None

        with subtests.test(msg="a failed read is cached until clear_cache"):
            reads = []

            def fake_read(path):
                reads.append(str(path))
                raise OSError("no such animation")

            monkeypatch.setattr(az, "read_frames", fake_read)
            az.clear_cache()
            assert az.zoomed_animation("a.gif", 64) is None
            assert az.zoomed_animation("a.gif", 64) is None
            assert reads == ["a.gif"]
            az.clear_cache()
            assert az.zoomed_animation("a.gif", 64) is None
            assert reads == ["a.gif", "a.gif"]
    finally:
        az.clear_cache()
