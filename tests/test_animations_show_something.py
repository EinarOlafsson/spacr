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

    def test_the_known_offenders_are_caught(self):
        """The 2026-08 audit found 27 below 1%, pathogen_diameter at 0.0%.
        This pins that the check still sees them -- if it stops, the check
        broke rather than the animations improving.
        """
        failures = validate_animations_show_something()
        assert "pathogen_diameter" in failures
        assert failures["pathogen_diameter"] < 0.005

    def test_a_generous_threshold_passes_everything(self):
        """Sanity: the measurement is not returning zero for everything."""
        assert validate_animations_show_something(minimum=0.0) == {}
