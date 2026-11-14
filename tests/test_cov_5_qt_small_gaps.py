"""Narrow gaps across the small Qt helpers: caches, marks, masks, cancels.

Each of these guards a moment nobody watches — a crop whose dataset moved, a
settings category re-attributed to a module with no picture, a preview result
that lands after the user moved on. None of them may raise out of a Qt slot,
and none may leave something on screen that is no longer true.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtGui import QIcon                                   # noqa: E402

from spacr.qt.widgets import animation_zoom as az                 # noqa: E402
from spacr.qt.widgets.section import Section, module_mark         # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# crop_thumbs: a dataset that moved, and a cache that must not throw
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# section: attributing a settings category to the module it came from
# ---------------------------------------------------------------------------

def test_a_key_with_no_picture_of_its_own_gets_no_mark(qapp):
    """A generic square would claim these settings came from a module."""
    assert module_mark("not_a_registered_app_at_all") is None


def test_an_iconset_that_raises_leaves_the_heading_plain(qapp, monkeypatch):
    from spacr.qt import iconset

    def refuse(*_args, **_kwargs):
        raise RuntimeError("icon resources are not installed")

    monkeypatch.setattr(iconset, "bundled_icon_path", refuse)

    assert module_mark("measure") is None


def test_an_icon_that_decodes_to_nothing_is_not_drawn(qapp, monkeypatch):
    """A null QIcon paints an empty box, which reads as a broken heading."""
    from spacr.qt import iconset

    monkeypatch.setattr(iconset, "app_icon", lambda _key: QIcon())

    assert module_mark("measure") is None


def test_a_category_shows_the_mark_of_the_module_it_was_folded_from(qapp):
    section = Section("Object measurements")

    assert section.source_app() == ""
    assert section.source_mark() is None

    assert section.set_source_app("measure", "Measure") is True
    assert section.source_app() == "measure"
    badge = section.source_mark()
    assert badge is not None
    assert badge.accessibleName() == "Measure"
    assert not badge.pixmap().isNull()


def test_re_attributing_to_a_module_with_no_picture_takes_the_mark_away(qapp):
    """The old module's icon left behind would attribute the wrong settings."""
    section = Section("Object measurements")
    section.set_source_app("measure", "Measure")

    assert section.set_source_app("no_such_module") is False

    assert section.source_app() == "no_such_module"
    assert section.source_mark() is None, "the previous module's mark is stale"


def test_dropping_the_attribution_entirely_hides_the_mark(qapp):
    section = Section("Object measurements")
    section.set_source_app("measure", "Measure")

    assert section.set_source_app("") is False
    assert section.source_app() == ""
    assert section.source_mark() is None


# ---------------------------------------------------------------------------
# animation_zoom: the masks and the cache
# ---------------------------------------------------------------------------

SIDE = az.SOURCE_SIZE


def test_the_ring_mask_is_the_stroke_and_nothing_outside_it():
    """Measuring content needs the stroke alone; the chrome mask adds the rest."""
    ring = az.field_ring_mask(SIDE)
    chrome = az.chrome_mask(SIDE)

    assert ring.shape == (SIDE, SIDE)
    assert ring.any()
    assert not ring[0, 0], "the corner is outside the field, not on its stroke"
    assert chrome[0, 0], "the corner is chrome"
    assert (chrome | ring == chrome).all(), "the ring is part of the chrome"


def test_asking_for_no_speck_removal_returns_the_mask_untouched():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True

    assert az.drop_specks(mask, minimum=0) is mask
    assert not az.drop_specks(mask, minimum=3).any(), "a lone pixel is a speck"


def test_a_zoom_that_kept_the_well_measures_itself_through_the_same_rule():
    """The output has to be measured by the mask the input was measured by."""
    kept = az.ZoomedAnimation(
        path="x.gif", size=64,
        frames=(np.zeros((64, 64, 3), np.uint8),), delays=(40,),
        source_extent=0.4, fill=0.6, crop=(0, 0, az.SOURCE_SIZE), shows_field=True)

    mask = kept.chrome_mask()
    assert mask is not None and mask.shape == (64, 64)
    assert mask.any() and not mask.all()
    assert kept.measured_fill() == 0.0, "blank frames have no content"

    erased = az.ZoomedAnimation(
        path="x.gif", size=64,
        frames=(np.zeros((64, 64, 3), np.uint8),), delays=(40,),
        source_extent=0.4, fill=0.6, crop=(0, 0, az.SOURCE_SIZE), shows_field=False)
    assert erased.chrome_mask() is None


def test_an_undecodable_animation_degrades_to_no_tooltip_not_an_exception(
        tmp_path):
    """A missing asset must not raise into the event loop from a hover."""
    az.clear_cache()
    try:
        assert az.zoomed_animation(str(tmp_path / "absent.gif"), 64) is None
        broken = tmp_path / "broken.gif"
        broken.write_bytes(b"GIF89a-not-really")
        assert az.zoomed_animation(str(broken), 64) is None
    finally:
        az.clear_cache()


def test_clearing_the_cache_makes_the_next_read_happen_again(tmp_path,
                                                             monkeypatch):
    reads = []

    def fake_read(path):
        reads.append(str(path))
        raise OSError("no such animation")

    monkeypatch.setattr(az, "read_frames", fake_read)
    az.clear_cache()

    assert az.zoomed_animation("a.gif", 64) is None
    assert az.zoomed_animation("a.gif", 64) is None
    assert reads == ["a.gif"], "the second call came from the cache"

    az.clear_cache()
    assert az.zoomed_animation("a.gif", 64) is None
    assert reads == ["a.gif", "a.gif"]
