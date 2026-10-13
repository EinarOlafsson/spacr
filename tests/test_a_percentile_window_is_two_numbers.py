"""A percentile window is two numbers, and every spelling of one is read.

THE SHIPPED DEFAULT NEVER REACHED THE PICTURE. ``percentiles`` was parsed by
the channel-list reader, which maps the POSITION strings '0', '1' and '2'
onto the colours 'r', 'g' and 'b' -- right for ``normalize_channels`` and
silently destructive here. The annotator's own ``[2, 98]`` became
``['b', '98']``, ``float('b')`` raised, and the caller fell back to a pair
nobody asked for. Every window whose low end was 0, 1 or 2 was replaced.

The other half is the one the maintainer named: a window typed into a text
box is a parsing problem handed to the user, and ``[1 99]`` is what a user
who meant ``[1, 99]`` types. Two numeric fields ask for two numbers, and
every text spelling already on disk is migrated rather than refused.
"""
from __future__ import annotations

import pytest

from spacr.crops import (DEFAULT_PERCENTILES, crop_spec_from_settings,
                         percentile_pair)


class TestEverySpellingOfAPair:

    def test_the_annotators_own_default_survives(self):
        """``[2, 98]`` is what ``set_annotate_default_settings`` ships."""
        from spacr.settings import set_annotate_default_settings

        shipped = set_annotate_default_settings({})["percentiles"]
        assert percentile_pair(shipped) == (2.0, 98.0)

    @pytest.mark.parametrize("written", [
        [2, 98], (2, 98), "2,98", "[2, 98]", "[2,98]", "(2, 98)", "2 98",
        "[2 98]", "2;98",
    ])
    def test_a_bracketed_or_spaced_pair_is_migrated_not_refused(self, written):
        """Whatever is already on disk still means the window it spells."""
        assert percentile_pair(written) == (2.0, 98.0)

    def test_a_pair_given_high_first_is_put_in_order(self):
        """``98, 2`` describes one window and there is one thing it can mean."""
        assert percentile_pair("98, 2") == (2.0, 98.0)

    def test_a_percentile_outside_zero_to_a_hundred_is_brought_back(self):
        """numpy raises on one, and the raise surfaces as a lost montage."""
        assert percentile_pair([-5, 150]) == (0.0, 100.0)

    @pytest.mark.parametrize("nothing", [None, "", "nonsense", [3], True, 3.5])
    def test_something_that_is_not_a_pair_answers_the_default(self, nothing):
        """A picture is the last thing this produces and the least important."""
        assert percentile_pair(nothing) == DEFAULT_PERCENTILES

    def test_the_default_is_the_pair_the_rest_of_spacr_stretches_to(self):
        """One number, so the parser and every panel cannot drift apart."""
        assert DEFAULT_PERCENTILES == (2.0, 98.0)


class TestTheCropLayerReadsAPairWrittenAsText:

    def test_a_space_separated_pair_reaches_the_cut(self):
        """``[1 99]`` is not valid Python, so it arrived as a string.

        A non-empty string is TRUTHY but is not a sequence, so the cut fell
        through to the full 0-100 stretch: the user configured a window, the
        crop ignored it, and nothing said so.
        """
        spec = crop_spec_from_settings(
            {"normalize": "[1 99]", "crop_mode": ["cell"]})

        assert list(spec.normalize) == [1.0, 99.0]

    def test_a_comma_separated_pair_reaches_the_cut(self):
        spec = crop_spec_from_settings(
            {"normalize": "2,98", "crop_mode": ["cell"]})

        assert list(spec.normalize) == [2.0, 98.0]

    def test_a_real_list_is_left_exactly_as_it_was(self):
        """The migration must not rewrite what already worked."""
        spec = crop_spec_from_settings(
            {"normalize": [5, 95], "crop_mode": ["cell"]})

        assert spec.normalize == [5, 95]

    def test_a_sequence_that_is_not_a_pair_is_still_discarded(self):
        spec = crop_spec_from_settings(
            {"normalize": [2, 98, 99], "crop_mode": ["cell"]})

        assert spec.normalize is False


class TestThePairReachesTheRenderer:

    def test_the_configured_window_is_what_normalisation_is_given(self,
                                                                 monkeypatch):
        """The end-to-end proof: what the settings say is what is applied.

        Driven through ``draw_crop`` rather than the parser, because the
        defect was in the step BETWEEN the setting and the annotator's own
        ``normalize_pil`` -- both of which were correct on their own.
        """
        numpy = pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        engine = pytest.importorskip("spacr.qt.annotate_engine")

        seen = []

        def remember(image, percentiles, channels):
            seen.append(tuple(percentiles))
            return image

        monkeypatch.setattr(engine, "normalize_pil", remember)

        from spacr.picture_settings import draw_crop

        crop = numpy.zeros((8, 8, 3), dtype="uint8")
        draw_crop(crop, {"normalize_channels": "r,g,b",
                         "percentiles": [2, 98]})

        assert seen == [(2.0, 98.0)]
