"""Picture settings translate one vocabulary into another without guessing.

The annotator names a setting one way and the crop layer another, and the two
disagree about what a channel is: the annotator's ``channels`` are colour
planes of a picture already drawn, while the crop layer's are which source
array planes to cut. Every branch here is a place where a value that cannot
be translated must be left alone rather than coerced -- an index read as a
colour, or a colour read as an index, produces a crop that looks plausible
and is wrong.
"""
from __future__ import annotations

import builtins

import numpy as np
import pytest

from spacr import picture_settings as ps


# -- which settings apply ----------------------------------------------------

def test_a_threshold_belongs_only_to_the_picker_that_computes_one():
    """The other pickers produce no probability for a threshold to cut."""
    assert ps.applies_to_picking("picking_threshold", "attributed") is True
    assert ps.applies_to_picking("picking_threshold", "rank") is False


def test_a_setting_this_module_never_heard_of_still_applies():
    """Hiding the unknown would hide every setting added after this one."""
    assert ps.applies_to_picking("some_new_setting", "rank") is True


def test_a_cut_setting_that_does_not_apply_to_the_mode_is_left_out(
        monkeypatch):
    """The translation honours the mode filter for every mapped setting,
    including ones added to the table after this rule was written."""
    monkeypatch.setitem(ps.CUT_SETTINGS, "image_type", "image_type")
    out = ps.to_crop_settings({"crop_source": ps.STREAM_IMAGES,
                               "image_type": "png"})
    assert "image_type" not in out


# -- what shape the cut can take ---------------------------------------------

def test_something_without_settings_cannot_be_asked_about_its_cut():
    """A caller holding a list rather than a settings dict gets False."""
    assert ps.bounding_box_only(["crop_source", "merged"]) is False


def test_the_database_route_can_only_cut_a_box():
    """A coordinate column has no outline to follow."""
    assert ps.bounding_box_only({"crop_source": ps.STREAM_FROM_DB}) is True


# -- translating channels ----------------------------------------------------

def test_a_bare_channel_number_is_carried_over_as_an_index():
    """One spin box holding 2 means source channel two, not colour 'r'."""
    out = ps.to_crop_settings({"crop_source": ps.STREAM_IMAGES, "channels": 2})
    assert out["png_dims"] == [2]


def test_a_channel_value_that_is_neither_letters_nor_indices_is_dropped():
    """Guessing a number for it would cut the wrong plane silently."""
    out = ps.to_crop_settings({"crop_source": ps.STREAM_IMAGES,
                               "channels": 1.5})
    assert "png_dims" not in out
    assert "png_channel_mapping" not in out


def test_a_channel_string_of_only_separators_is_dropped():
    """','' names no channel; an empty index list would cut nothing."""
    out = ps.to_crop_settings({"crop_source": ps.STREAM_IMAGES,
                               "channels": ",,"})
    assert "png_dims" not in out


def test_settings_that_are_not_a_mapping_translate_to_nothing():
    """A caller handing over an int must not raise inside the translator."""
    assert ps.to_crop_settings(7) == {}


def test_a_mask_plane_that_is_not_a_number_is_left_unset():
    """An unparseable plane must not become plane zero by accident."""
    out = ps.to_crop_settings({"crop_source": ps.STREAM_IMAGES,
                               "mask_array": "outer"})
    assert "mask_array" not in out
    assert out["stream_method"] == "array"


# -- drawing -----------------------------------------------------------------

def _crop():
    return np.full((8, 8, 3), 120, dtype="uint8")


def test_a_bracketed_channel_list_is_read_as_channels():
    """A settings file round-tripped through a text field keeps its brackets."""
    drawn = ps.draw_crop(_crop(), {"channels": "['r', 'g']"})
    assert drawn.shape == (8, 8, 3)
    assert drawn[:, :, 2].max() == 0


def test_a_channel_setting_that_cannot_be_iterated_is_ignored():
    """A stray number in the outline field must not lose the whole crop."""
    drawn = ps.draw_crop(_crop(), {"outline": 5})
    np.testing.assert_array_equal(drawn, _crop())


def test_percentiles_that_are_not_a_pair_fall_back_to_the_default():
    """A half-filled percentile field must not normalise against nothing."""
    drawn = ps.draw_crop(_crop(), {"normalize_channels": "r",
                                   "percentiles": "5"})
    assert drawn.shape == (8, 8, 3)


def test_an_object_size_that_is_not_a_number_outlines_without_a_bound():
    """Text in the size field must not stop the outline being drawn."""
    drawn = ps.draw_crop(_crop(), {"outline": "r", "object_size": "large"})
    assert drawn.shape == (8, 8, 3)


def test_an_unavailable_drawing_pipeline_returns_the_crop_unchanged(
        monkeypatch):
    """Losing a montage to a missing optional import is the worst trade."""
    real_import = builtins.__import__

    def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith("annotate_engine") or "annotate_engine" in name:
            raise ImportError("Qt is not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    drawn = ps.draw_crop(_crop(), {"outline": "r"})
    np.testing.assert_array_equal(drawn, _crop())
