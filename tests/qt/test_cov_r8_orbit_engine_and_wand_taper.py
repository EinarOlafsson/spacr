"""The fractal's four-frame temporal window, and the taper's two markers.

``OrbitEngine`` is the whole of the CPU backend's antialiasing: four
sub-pixel jitter phases in a ring, blended each frame. Nothing had ever
constructed one under test, so its buffer allocation, its first-frame
fill and its ring advance were all unexecuted.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# OrbitEngine
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    numba = pytest.importorskip("numba")     # noqa: F841 - the CPU backend
    from spacr.qt.widgets.fractal_travel import OrbitEngine

    return OrbitEngine(1)


class TestTheFourFrameRing:

    def test_a_first_render_produces_a_picture_of_the_asked_for_size(
            self, engine):
        frame = engine.render(16, 12, 0.0, 1.0, 0.5, 8)

        assert frame.shape == (12, 16, 3)
        assert frame.dtype == np.uint8

    def test_the_opening_frame_is_the_picture_not_a_fade_from_black(
            self, engine):
        """THE UNCOVERED BRANCH: ``frames == 0`` fills the history.

        The blend averages four ring slots. On the first frame three of
        them have never been written, so without the fill the opening
        image would be a quarter-strength version of itself -- a fade up
        from black on every launch.
        """
        from spacr.qt.widgets.fractal_travel import OrbitEngine

        fresh = OrbitEngine(1)
        assert fresh.frames == 0

        first = fresh.render(12, 10, 0.0, 1.0, 0.5, 6)

        assert fresh.frames == 1
        assert int(first.max()) > 0, "the opening frame was black"
        assert fresh.ring is not None
        for index in range(4):
            assert np.array_equal(fresh.ring[index], fresh.ring[0]), (
                f"ring slot {index} was left unwritten on the first frame")

    def test_the_ring_advances_one_jitter_phase_per_frame(self):
        """Its own engine: a size change would restart the ring."""
        from spacr.qt.widgets.fractal_travel import JITTERS, OrbitEngine

        assert len(JITTERS) == 4
        walking = OrbitEngine(1)

        for step in range(len(JITTERS) + 1):
            walking.render(12, 10, 0.05 * step, 1.0, 0.5, 6)
            assert walking.slot == (step + 1) % 4, (
                f"the ring was at the wrong phase after frame {step}")

        assert walking.frames == len(JITTERS) + 1

    def test_the_returned_frame_is_a_copy_the_next_render_cannot_change(
            self, engine):
        first = engine.render(12, 10, 0.0, 1.0, 0.5, 6)
        keep = first.copy()

        engine.render(12, 10, 9.0, 1.0, 0.5, 6)

        assert np.array_equal(first, keep), (
            "the caller's frame was overwritten by the next render")


class TestResizingTheRing:

    def test_the_same_size_twice_keeps_the_buffers(self):
        from spacr.qt.widgets.fractal_travel import OrbitEngine

        resized = OrbitEngine(1)
        resized.render(12, 10, 0.0, 1.0, 0.5, 6)
        ring, output = resized.ring, resized.output

        resized.render(12, 10, 0.1, 1.0, 0.5, 6)

        assert resized.ring is ring, "the ring was reallocated for no reason"
        assert resized.output is output

    def test_a_new_size_reallocates_and_restarts_the_window(self):
        """THE UNCOVERED BRANCH: the size changed.

        The ring holds four frames of the OLD size. Blending those into
        a new-sized output is a shape error at best; restarting the
        window is what makes a resize show the new picture rather than a
        crash or four frames of the last one.
        """
        from spacr.qt.widgets.fractal_travel import OrbitEngine

        resized = OrbitEngine(1)
        resized.render(12, 10, 0.0, 1.0, 0.5, 6)
        assert resized.frames == 1

        frame = resized.render(20, 14, 0.1, 1.0, 0.5, 6)

        assert frame.shape == (14, 20, 3)
        assert resized.ring.shape == (4, 14, 20, 3)
        assert resized.frames == 1, "the resize did not restart the window"
        assert resized.slot == 1

    def test_a_fresh_engine_holds_no_buffers_at_all(self):
        from spacr.qt.widgets.fractal_travel import OrbitEngine

        fresh = OrbitEngine(1)
        assert fresh.ring is None and fresh.output is None
        assert (fresh.width, fresh.height) == (0, 0)

    def test_the_thread_count_is_never_below_one(self):
        from spacr.qt.widgets.fractal_travel import OrbitEngine

        assert OrbitEngine(0).thread_count == 1
        assert OrbitEngine(-4).thread_count == 1
        assert OrbitEngine(3).thread_count == 3


# ---------------------------------------------------------------------------
# wand_rescue.taper_region_to_intensity -- the two watershed markers
# ---------------------------------------------------------------------------

def _scene(size=40):
    """A bright square inside a dimmer flood, and the cut that split it."""
    image = np.zeros((size, size), dtype=np.float32)
    image[10:30, 10:30] = 200.0
    flooded = np.zeros((size, size), dtype=bool)
    flooded[8:32, 8:32] = True
    provisional = np.zeros((size, size), dtype=bool)
    provisional[8:32, 8:20] = True          # a straight cut down the middle
    return image, flooded, provisional


class TestTaperingACutOntoAnIntensityEdge:

    def test_the_boundary_moves_and_stays_inside_the_flood(self):
        from spacr.qt.wand_rescue import taper_region_to_intensity

        image, flooded, provisional = _scene()

        out = taper_region_to_intensity(image, flooded, provisional,
                                        seed_yx=(20, 14))

        assert out.dtype == bool
        assert not (out & ~flooded).any(), "the result left the flood"
        assert out[20, 14], "the piece the click is in was dropped"

    def test_a_seed_outside_the_provisional_region_changes_nothing(self):
        from spacr.qt.wand_rescue import taper_region_to_intensity

        image, flooded, provisional = _scene()

        out = taper_region_to_intensity(image, flooded, provisional,
                                        seed_yx=(20, 28))

        assert np.array_equal(out, provisional & flooded)

    def test_nothing_thrown_away_means_nothing_to_taper(self):
        from spacr.qt.wand_rescue import taper_region_to_intensity

        image, flooded, _provisional = _scene()

        out = taper_region_to_intensity(image, flooded, flooded,
                                        seed_yx=(20, 14))

        assert np.array_equal(out, flooded)

    def test_a_discarded_sliver_thinner_than_the_margin_still_tapers(self):
        """The fallback: the deepest quarter instead of giving up.

        A one-pixel-wide discard has no room for an 8-deep band, and
        leaving the straight edge is the outcome the taper exists to
        avoid.
        """
        from spacr.qt.wand_rescue import taper_region_to_intensity

        image, flooded, provisional = _scene()
        provisional = flooded.copy()
        provisional[:, 31] = False           # a one-pixel discard

        out = taper_region_to_intensity(image, flooded, provisional,
                                        seed_yx=(20, 14), margin=8)

        assert out[20, 14]
        assert not (out & ~flooded).any()

    def test_both_markers_are_always_present_by_the_time_they_are_used(self):
        """THE PIN.

        ``if not background.any() or not foreground.any()`` gives up
        before the watershed, and neither half can be true here.
        Foreground always holds the clicked disc -- the click is trusted
        and a few pixels around it are marked object unconditionally.
        Background is non-empty whenever anything was discarded at all,
        because the fallback cutoff is taken from the discard's own
        maximum depth, and that was already checked by the return above.

        Marking nothing as background makes the watershed flood the
        whole object, which is the failure this stops. The pin fails if
        the unconditional disc or the depth fallback is removed.
        """
        from spacr.qt import wand_rescue

        source = inspect.getsource(wand_rescue.taper_region_to_intensity)
        assert "<= 4) & provisional" in source, (
            "the clicked disc is no longer forced into the foreground")
        assert "float(depth.max()) * 0.75" in source, (
            "the depth fallback for a thin discard is gone")

        from scipy.ndimage import distance_transform_edt

        image, flooded, provisional = _scene()
        for width in (1, 2, 4, 12):
            trial = flooded.copy()
            trial[:, 32 - width:32] = False
            removed = flooded & ~trial
            assert removed.any()
            depth = distance_transform_edt(removed)
            cutoff = max(1.0, float(depth.max()) * 0.75)
            assert (removed & (depth >= cutoff)).any(), (
                f"a discard {width} pixels wide left no background marker")
