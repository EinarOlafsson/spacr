"""Two loops that never leave by their own back door.

The magic wand's queue never holds a coordinate off the image, and the
byte formatter never runs out of units. Both guards are one line each and
both are correct to keep -- an off-image index into a numpy array is an
IndexError inside a paint handler, and a formatter that fell off the end
of its unit list would return an unlabelled number. Neither can fire
today, so each is pinned to the code that makes it so.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from spacr.qt.mask_engine import magic_wand
from spacr.qt.resource_cleanup import human_bytes


class TestTheWandsQueueNeverHoldsAnOffImageCoordinate:

    def _image(self, size=8):
        image = np.zeros((size, size, 3), dtype=np.uint8)
        image[:4, :, :] = 200          # a bright band across the top
        return image

    def test_a_wand_from_inside_the_band_fills_the_band(self):
        image = self._image()
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        out = magic_wand(image, mask, seed_x=3, seed_y=1, tolerance=5.0)

        assert out[:4, :].all(), "the band was not filled"
        assert not out[4:, :].any(), "the fill leaked past the band"

    def test_a_wand_seeded_in_the_very_corner_stays_on_the_image(self):
        """The corner is where an unchecked neighbour would go negative.

        Every one of the four neighbours of (0, 0) is off the image on
        two sides. They are refused at the push, not at the pop.
        """
        image = self._image()
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        out = magic_wand(image, mask, seed_x=0, seed_y=0, tolerance=5.0)

        assert out.shape == mask.shape
        assert out[0, 0] == 255

    def test_a_seed_off_the_image_returns_the_mask_untouched(self):
        image = self._image()
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[2, 2] = 7

        for seed in ((-1, 0), (0, -1), (8, 0), (0, 8)):
            out = magic_wand(image, mask, seed_x=seed[0], seed_y=seed[1],
                             tolerance=5.0)
            assert out is mask, f"a seed at {seed} copied the mask anyway"
            assert out[2, 2] == 7

    def test_nothing_out_of_bounds_can_be_queued(self):
        """THE PIN.

        The seed is bounds-checked before the loop starts, and every
        neighbour is bounds-checked before it is appended -- so the
        `continue` at the top of the loop guards against a coordinate
        that cannot arrive.

        It is still the right line to keep: an off-image index into a
        numpy array is an IndexError raised inside a paint handler,
        where it takes the whole canvas down. This fails if the push
        stops checking, which is the change that would make it live.
        """
        source = inspect.getsource(magic_wand)
        push = source[source.index("for dx, dy in"):]
        assert "0 <= nx < image.shape[1]" in push
        assert "0 <= ny < image.shape[0]" in push, (
            "neighbours are no longer bounds-checked before being queued")

    def test_the_pixel_budget_stops_a_fill_of_the_whole_image(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        out = magic_wand(image, mask, seed_x=16, seed_y=16, tolerance=255.0,
                         max_pixels=10)

        assert int((out > 0).sum()) <= 11, (
            "the fill ran past its pixel budget")


class TestTheByteFormatterNeverRunsOutOfUnits:

    @pytest.mark.parametrize("count,expected", [
        (0, "0 B"),
        (512, "512 B"),
        (1536, "1.5 KB"),
        (1024 ** 2 + 1024 ** 2 // 2, "1.5 MB"),
        (3 * 1024 ** 3, "3.0 GB"),
        (2 * 1024 ** 4, "2.0 TB"),
    ])
    def test_it_labels_the_number_at_every_scale(self, count, expected):
        assert human_bytes(count) == expected

    def test_bytes_are_whole_because_half_a_byte_is_a_fake_place(self):
        assert human_bytes(1) == "1 B"
        assert human_bytes(1023) == "1023 B"

    def test_a_negative_count_reads_as_nothing_rather_than_as_minus(self):
        assert human_bytes(-5) == "0 B"

    def test_a_number_past_every_unit_still_says_terabytes(self):
        """No unit above TB, so TB has to absorb everything above it."""
        assert human_bytes(9999 * 1024 ** 4) == "9999.0 TB"

    def test_the_loop_cannot_finish_without_returning(self):
        """THE PIN.

        `unit == "TB"` forces the return on the last pass, so the line
        after the loop is unreachable. Keeping it means a future unit
        appended to the tuple without extending that condition returns a
        number instead of falling off the end -- which is why the pin
        checks the tuple's last member rather than just the condition.
        """
        source = inspect.getsource(human_bytes)
        units = source[source.index("for unit in ("):source.index("):")]
        assert units.rstrip().endswith('"TB"'), (
            "the last unit is no longer TB, so the loop can now finish "
            "without returning and the line below it is live")
        assert 'unit == "TB"' in source
