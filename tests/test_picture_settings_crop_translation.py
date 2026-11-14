"""Picture settings translated to the crop layer and drawn, without guessing.

The annotator names a setting one way and the crop layer another, and they
disagree about what a channel is: the annotator's ``channels`` are colour
planes of a picture already drawn, the crop layer's are source planes to
cut.  ``to_crop_settings`` must carry over only what it can translate and
leave the rest out (an index read as a colour, or a colour as an index, cuts
a plausible wrong crop); ``draw_crop`` must ignore what it cannot read and
never lose the crop.
"""
from __future__ import annotations

import builtins

import numpy as np

from spacr import picture_settings as ps


def _crop():
    """An 8x8 RGB crop: a red ramp, flat green and a blue square."""
    crop = np.zeros((8, 8, 3), dtype=np.uint8)
    crop[..., 0] = np.arange(64, dtype=np.uint8).reshape(8, 8) * 3
    crop[..., 1] = 200
    crop[2:6, 2:6, 2] = 250
    return crop


def test_to_crop_settings_carries_over_only_what_translates(subtests,
                                                            monkeypatch):
    """Exact translations per input: a bare number is an index list, letters
    are a colour mapping, separators-only, non-integer or non-mapping inputs
    give nothing, a non-numeric plane is left unset, streaming sources get
    their ``stream_method``, and a mapped setting the mode does not use is
    left out."""
    images, database = ps.STREAM_IMAGES, ps.STREAM_FROM_DB
    cases = [
        ("bare channel number", {"crop_source": images, "channels": 2},
         {"png_dims": [2], "stream_method": "array"}),
        ("index string", {"crop_source": images, "channels": "0;1",
                          "object_array": "3"},
         {"png_dims": [0, 1], "stream_method": "array", "object_array": "3"}),
        ("neither letters nor indices",
         {"crop_source": images, "channels": 1.5},
         {"stream_method": "array"}),
        ("only separators", {"crop_source": images, "channels": ",,"},
         {"stream_method": "array"}),
        ("mask plane not a number",
         {"crop_source": images, "mask_array": "outer"},
         {"stream_method": "array"}),
        ("colour letters, size and box on load",
         {"channels": "r,b", "crop_size": 64, "crop_shape": "bbox"},
         {"png_size": [64, 64],
          "png_channel_mapping": {"r": 2, "g": None, "b": 0},
          "use_bounding_box": True}),
        ("database stream",
         {"crop_source": database, "object_type": "Cell",
          "red_channel": "2", "green_channel": "x", "blue_channel": -1},
         {"object_array": "cell",
          "png_channel_mapping": {"r": 2, "g": None, "b": None},
          "stream_method": "column", "use_bounding_box": True}),
        ("not a mapping", 7, {}),
        ("nothing", None, {}),
    ]
    for case_id, picture, expected in cases:
        with subtests.test(msg=case_id):
            assert ps.to_crop_settings(picture) == expected

    with subtests.test(msg="a mapped setting the mode does not use"):
        monkeypatch.setitem(ps.CUT_SETTINGS, "image_type", "image_type")
        assert ps.to_crop_settings({"crop_source": images,
                                    "image_type": "png"}) == {
            "stream_method": "array"}


def test_draw_crop_ignores_what_it_cannot_read_and_never_loses_the_crop(
        subtests, monkeypatch):
    """A bracketed channel list keeps r and g and blanks b; an unreadable
    channel value, half a percentile pair or a textual object size each draws
    exactly as the default would; no draw setting, or a missing drawing
    pipeline, returns the very same array."""
    crop = _crop()
    defaults = list(ps.DEFAULT_PERCENTILES)
    equivalent = [
        ("percentiles not a pair",
         {"normalize_channels": "r", "percentiles": "5"},
         {"normalize_channels": "r", "percentiles": defaults}),
        ("object size not a number",
         {"outline": "b", "object_size": "large"},
         {"outline": "b", "object_size": 0}),
        ("bracketed channel list", {"channels": "['r', 'g']"},
         {"channels": ["r", "g"]}),
        ("positions are colours", {"channels": "0,1"},
         {"channels": ["r", "g"]}),
    ]
    for case_id, picture, reference in equivalent:
        with subtests.test(msg=case_id):
            drawn = ps.draw_crop(crop, picture)
            assert drawn.shape == (8, 8, 3) and drawn.dtype == np.uint8
            np.testing.assert_array_equal(drawn, ps.draw_crop(crop, reference))
            assert not np.array_equal(drawn, crop)

    with subtests.test(msg="channel filter keeps r and g exactly"):
        drawn = ps.draw_crop(crop, {"channels": "['r', 'g']"})
        np.testing.assert_array_equal(drawn[..., :2], crop[..., :2])
        assert drawn[..., 2].max() == 0

    with subtests.test(msg="outline value that cannot be iterated"):
        np.testing.assert_array_equal(ps.draw_crop(crop, {"outline": 5}), crop)

    with subtests.test(msg="no draw setting"):
        assert ps.draw_crop(crop, {"crop_size": 64}) is crop

    with subtests.test(msg="drawing pipeline unavailable"):
        real_import = builtins.__import__

        def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
            if "annotate_engine" in name:
                raise ImportError("Qt is not installed")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        assert ps.draw_crop(crop, {"outline": "r"}) is crop
