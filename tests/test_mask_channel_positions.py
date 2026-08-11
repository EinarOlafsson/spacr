"""The merged stack's channel axis is in ROLE order, not sorted order.

`io.preprocess_img_data` walks nucleus, cell, pathogen, organelle and gives
each newly seen raw channel the next dense position -- ``seen[ch] =
len(mask_channels)`` -- then slices the stack with that list. So the axis is
role-ordered and deduplicated.

Two places computed the position as ``sorted({nucleus, cell, pathogen,
organelle})`` instead. Sorted order equals role order only when the roles
happen to be in ascending channel order, which is why this survived: the
common ``nucleus=0, cell=1`` layout agrees, and everything is fine until
someone images the nucleus on a later channel.

When they disagree, Cellpose segments each object on another object's plane.
Nothing raises. The masks look plausible, and every count, intensity and
recruitment number downstream is measured from the wrong image.
"""

import pytest

from spacr.utils import (MASK_CHANNEL_ROLE_ORDER,
                         dense_mask_channel_positions)


def test_role_order_not_sorted_order():
    """The case the sorted reading gets wrong, stated as a number.

    Stack axis is [2, 0, 1]: nucleus first because nucleus is the first role,
    not because 2 is the smallest channel. Raw channel 1 is therefore at
    position 2 -- the sorted reading says 1, which holds the cell.
    """
    settings = {"nucleus_channel": 2, "cell_channel": 0,
                "pathogen_channel": None, "organelle_channel": 1}
    positions = dense_mask_channel_positions(settings)

    assert positions == {2: 0, 0: 1, 1: 2}
    assert positions[1] == 2, "the organelle plane is not where sorted says"

    sorted_reading = {ch: i for i, ch in enumerate(sorted(positions))}
    assert sorted_reading != positions, "pick a case where the two differ"


def test_the_usual_layout_is_unchanged():
    """nucleus=0, cell=1 is why this went unnoticed: both readings agree."""
    settings = {"nucleus_channel": 0, "cell_channel": 1,
                "pathogen_channel": 2, "organelle_channel": None}
    assert dense_mask_channel_positions(settings) == {0: 0, 1: 1, 2: 2}


def test_a_channel_used_by_two_roles_gets_one_position():
    """The writer deduplicates, so the reader must too -- otherwise every
    position after the repeat is off by one."""
    settings = {"nucleus_channel": 1, "cell_channel": 1,
                "pathogen_channel": 0, "organelle_channel": None}
    assert dense_mask_channel_positions(settings) == {1: 0, 0: 1}


def test_string_channels_are_coerced_like_the_writer_does():
    """A settings CSV round-trips channel indices as strings, and
    io.preprocess_img_data int()s them before keying `seen`."""
    settings = {"nucleus_channel": "1", "cell_channel": "0",
                "pathogen_channel": None, "organelle_channel": None}
    assert dense_mask_channel_positions(settings) == {1: 0, 0: 1}


def test_unusable_channels_are_absent_rather_than_guessed():
    settings = {"nucleus_channel": None, "cell_channel": "not a channel",
                "pathogen_channel": 3, "organelle_channel": None}
    assert dense_mask_channel_positions(settings) == {3: 0}


def test_the_role_order_matches_what_the_writer_walks():
    """If io.preprocess_img_data's list is ever reordered, this fails.

    The two are the same fact written twice; the alternative is discovering
    the disagreement as mis-segmented plates.
    """
    assert MASK_CHANNEL_ROLE_ORDER == (
        "nucleus_channel", "cell_channel", "pathogen_channel",
        "organelle_channel")

    import inspect
    from spacr import io
    source = inspect.getsource(io.preprocess_img_data)
    marker = ("mask_channels_raw = [settings.get('nucleus_channel'), "
              "settings.get('cell_channel'), "
              "settings.get('pathogen_channel'), "
              "settings.get('organelle_channel')]")
    assert marker in source, (
        "preprocess_img_data no longer builds the channel list in the order "
        "MASK_CHANNEL_ROLE_ORDER claims; one of the two moved")


# ---------------------------------------------------------------------------
# the organelle path, which was wrong on a FIRST run
# ---------------------------------------------------------------------------

def test_the_recorded_dense_position_wins_when_it_exists():
    """io.preprocess_img_data records the position it actually used, so that
    is the authority whenever it is on hand."""
    settings = {"nucleus_channel": 2, "cell_channel": 0,
                "organelle_channel": 1, "cellpose_organelle_channel": 2}
    recorded = settings.get("cellpose_organelle_channel")
    assert recorded is not None
    assert int(recorded) == dense_mask_channel_positions(settings)[1]


@pytest.mark.parametrize(("settings", "expected"), [
    ({"nucleus_channel": 2, "cell_channel": 0, "organelle_channel": 1}, 2),
    ({"nucleus_channel": 0, "cell_channel": 1, "organelle_channel": 2}, 2),
    ({"nucleus_channel": 3, "cell_channel": 2, "organelle_channel": 0}, 2),
    ({"nucleus_channel": None, "cell_channel": None,
      "organelle_channel": 3}, 0),
])
def test_the_organelle_falls_back_to_its_role_order_position(settings,
                                                             expected):
    """The fallback has to be RIGHT, not merely present: it is what a resumed
    run uses, and a resumed run is the normal state after the raws are moved
    into src/orig."""
    settings.setdefault("pathogen_channel", None)
    positions = dense_mask_channel_positions(settings)
    assert positions[settings["organelle_channel"]] == expected
