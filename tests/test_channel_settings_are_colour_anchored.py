"""Channel settings must not encode a colour by list position.

The rule, and the reason for it, is INVARIANTS 13. ``png_dims=[0,1,2]`` meant
"0 is blue" for the whole life of the project -- not because anything said
so, but because cv2 reads a 3-channel array as BGR and the writer handed it
the array unchanged. When someone read the list the other way, every crop
written for eleven days put the 405/DAPI plane in the red channel.

The survey behind this file (instructions/done/20) classified every channel
setting spaCR has into three kinds:

* a SET of source channels to process -- ``channels``, ``channel_dims``.
  No colour is involved and forcing r/g/b onto them would invent a meaning
  that is not there. These stay lists.
* a named role -> one source channel -- ``cell_channel``,
  ``nucleus_channel``, ``pathogen_channel``, ``organelle_channel``,
  ``outside_channel``, ``total_channel``, ``channel_of_interest``. Already
  explicit: the NAME says which stain, so position decides nothing.
* a COLOUR mapping -- ``png_channel_mapping``, and ``train_channels``, which
  names its planes 'r'/'g'/'b' directly. These are the only two that can be
  read backwards, and neither is positional.

So the tests here are guards, not fixes. They exist so that a future setting
cannot quietly reintroduce the pattern.
"""
from __future__ import annotations

import pytest

from spacr import crops


#: Settings that legitimately hold a list of SOURCE CHANNEL INDICES with no
#: colour meaning. Adding to this list is a claim that the setting decides
#: which channels are processed, not what colour anything is drawn in.
CHANNEL_SETS = ("channels", "channel_dims")

#: The only settings that carry colour. Anything else that starts to would
#: have to be added here deliberately.
COLOUR_SETTINGS = ("png_channel_mapping", "train_channels")


def test_the_declared_mapping_is_a_dict_keyed_by_colour():
    """Not a list. A dict cannot be read backwards -- that is the whole fix."""
    mapping = crops.resolve_png_channel_mapping({})
    assert set(mapping) == set(crops.PNG_COLOR_KEYS) == {"r", "g", "b"}
    assert mapping == crops.DEFAULT_PNG_CHANNEL_MAPPING


def test_the_default_puts_the_405_plane_in_blue():
    """Microscope channels arrive in wavelength order: 0 is 405, and 405 is
    blue. The default has to match what a biologist means, because it is what
    every existing settings file will be translated into."""
    assert crops.DEFAULT_PNG_CHANNEL_MAPPING["b"] == 0
    assert crops.DEFAULT_PNG_CHANNEL_MAPPING["g"] == 1
    assert crops.DEFAULT_PNG_CHANNEL_MAPPING["r"] == 2


@pytest.mark.parametrize("dims,expected", [
    ([0, 1, 2], {"r": 2, "g": 1, "b": 0}),
    ([1, 2, 3], {"r": 3, "g": 2, "b": 1}),
    ([0, 1], {"r": None, "g": 1, "b": 0}),
    ([2], {"r": 2, "g": 2, "b": 2}),
])
def test_a_legacy_png_dims_list_is_translated_the_way_it_always_looked(
        dims, expected):
    """The legacy reading, because that is the one that was ever on screen.

    A settings CSV written by any older build lands here, and it has to keep
    rendering the way it always did or the user's data changes colour under
    them on upgrade.
    """
    assert crops.png_dims_to_channel_mapping(dims) == expected


def test_png_dims_still_works_and_loses_to_an_explicit_mapping():
    """Both accepted, and the explicit one wins. A user who has said outright
    which channel is red must not have it overridden by a stale list."""
    both = {"png_dims": [0, 1, 2],
            "png_channel_mapping": {"r": 0, "g": 1, "b": 2}}
    assert crops.resolve_png_channel_mapping(both) == {"r": 0, "g": 1, "b": 2}


def test_a_miskeyed_colour_is_an_error_not_a_silent_drop():
    """A mis-keyed mapping would delete a whole stain from every crop in the
    run and say nothing, which is the failure mode that has to be loud."""
    with pytest.raises(crops.CropError, match="no colour"):
        crops.resolve_png_channel_mapping(
            {"png_channel_mapping": {"red": 2, "g": 1, "b": 0}})
    with pytest.raises(crops.CropError, match="every colour empty"):
        crops.resolve_png_channel_mapping(
            {"png_channel_mapping": {"r": None, "g": None, "b": None}})


def test_the_channel_sets_are_still_plain_lists():
    """`channels` decides which channels are PROCESSED. It has no colour
    semantics, and converting it to an r/g/b dict would invent one."""
    from spacr.settings import expected_types

    for key in CHANNEL_SETS:
        if key in expected_types:
            declared = expected_types[key]
            allowed = declared if isinstance(declared, tuple) else (declared,)
            assert list in allowed, (
                f"{key} stopped being a list; if it became a colour mapping "
                f"that is a decision to record in COLOUR_SETTINGS and in "
                f"instructions/done/20, not a quiet type change")


def test_train_channels_names_colours_rather_than_positions():
    """It is already colour-anchored -- 'r', 'g', 'b' -- which is why it
    needed no change. But its meaning DEPENDS on png_channel_mapping, and
    `crops.legacy_channel_names` is what translates a model trained on legacy
    crops onto the corrected order."""
    assert crops.legacy_channel_names(["r", "g"]) == ["b", "g"]
    assert crops.legacy_channel_names(["b"]) == ["r"]
    # Anything that is not a colour name passes through untouched rather than
    # raising: this runs over user-supplied model metadata.
    assert crops.legacy_channel_names(["x"]) == ["x"]


def test_channels_from_settings_is_the_single_translation_point():
    """Everything downstream of this is in colour order, so there is exactly
    one place the legacy list is interpreted."""
    assert crops.channels_from_settings({}) == (2, 1, 0)
    assert crops.channels_from_settings({"png_dims": [0, 1, 2]}) == (2, 1, 0)
    assert crops.channels_from_settings(
        {"png_channel_mapping": {"r": 0, "g": 1, "b": 2}}) == (0, 1, 2)
    # A greyscale mapping stays one plane rather than three identical ones.
    assert crops.channels_from_settings({"png_dims": [1]}) == (1,)
