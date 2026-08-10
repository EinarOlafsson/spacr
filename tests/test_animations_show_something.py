"""An animation that changes nothing teaches nothing.

`validate_setting_animation_assets` checks the FILES are intact. A GIF can
pass that -- right size, right digest -- and still show the viewer nothing,
which is invisible to every other check in the project because it is a
property of the pixels rather than of the code.
"""

import numpy as np
import pytest
from PIL import Image

from spacr.setting_animations import (
    MIN_VISIBLE_CHANGE, measure_visible_change,
    validate_animations_show_something,
)


def _gif(path, frames):
    images = [Image.fromarray(f.astype(np.uint8), "RGB") for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], loop=0)
    return str(path)


class TestMeasurement:

    def test_a_still_animation_measures_zero(self, tmp_path):
        frame = np.zeros((40, 40, 3))
        assert measure_visible_change(
            _gif(tmp_path / "still.gif", [frame, frame.copy()])) == 0.0

    def test_a_changing_animation_measures_the_changed_fraction(self, tmp_path):
        a = np.zeros((40, 40, 3))
        b = a.copy()
        b[:20, :, :] = 255                      # half the frame
        got = measure_visible_change(_gif(tmp_path / "half.gif", [a, b]))
        assert 0.4 < got < 0.6

    def test_it_compares_against_the_MOST_different_frame(self, tmp_path):
        """These GIFs loop, so the LAST frame is the first one again.
        Comparing first to last reports zero change for every animation
        ever made -- which is exactly what a first attempt did.
        """
        a = np.zeros((40, 40, 3))
        b = a.copy()
        b[:20, :, :] = 255
        looped = _gif(tmp_path / "loop.gif", [a, b, a.copy()])
        assert measure_visible_change(looped) > 0.4

    def test_an_unreadable_file_reads_as_showing_nothing(self, tmp_path):
        """Not an exception: the caller wants "this shows nothing" in its
        report, beside the others, rather than a traceback."""
        broken = tmp_path / "broken.gif"
        broken.write_bytes(b"not a gif")
        assert measure_visible_change(str(broken)) == 0.0

    def test_a_single_frame_shows_nothing(self, tmp_path):
        frame = np.zeros((40, 40, 3))
        assert measure_visible_change(
            _gif(tmp_path / "one.gif", [frame])) == 0.0


class TestTheShippedAnimations:

    def test_the_threshold_is_a_visible_amount(self):
        assert 0.001 <= MIN_VISIBLE_CHANGE <= 0.05

    def test_it_returns_every_failure_not_just_the_first(self):
        """Raising on the first would make fixing them a 27-round loop."""
        failures = validate_animations_show_something()
        assert isinstance(failures, dict)

    def test_the_number_of_silent_animations_never_grows(self):
        """A ratchet, because the previous version of this test pinned one
        slug and so FAILED the moment someone fixed it.

        History: the 2026-08 audit found 27 animations below 1%.
        `_diameter_scene` drew a caliper over an object that never changed
        size (4 fixed, 23 left). `_filter_scene` faded a single object where
        the setting is a THRESHOLD, and for the `minimum` variants that one
        object is the smallest of the four (8 fixed, 15 left).
        `_umap_scene` drew three clusters of five dots at radius 1.5 --
        about 106 px of a 129,600 px frame -- so an animation that changed
        every point still measured a fifth of a percent (7 fixed, 8 left).
        The last eight were one-offs, mostly non-cell objects drawn 10-16 px
        wide in a 360 px frame.

        All 94 clear the threshold now, so this asserts zero. It may
        never rise.
        """
        failures = validate_animations_show_something()
        assert len(failures) == 0, (
            "an animation has gone under the visible-change threshold: "
            f"{sorted(failures)}")

    def test_the_repaired_families_stay_fixed(self):
        """Pinned against a regeneration that silently undoes the fix."""
        failures = validate_animations_show_something()
        repaired = (
            "cell_diameter", "nucleus_diameter",
            "pathogen_diameter", "organelle_diameter",
            "cell_min_area", "nucleus_min_area",
            "pathogen_min_area", "organelle_min_area",
            "cell_min_intensity_percentile",
            "nucleus_min_intensity_percentile",
            "pathogen_min_intensity_percentile",
            "organelle_min_intensity_percentile",
            "remove_cluster_noise", "plot_points", "plot_by_cluster",
            "min_dist", "plot_images", "remove_image_canvas", "dot_size",
        )
        back = [slug for slug in repaired if slug in failures]
        assert not back, f"back under the visible-change threshold: {back}"

    def test_a_generous_threshold_passes_everything(self):
        """Sanity: the measurement is not returning zero for everything."""
        assert validate_animations_show_something(minimum=0.0) == {}
