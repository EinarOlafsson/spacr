"""Three crop modes, and each asks only what its own route needs.

The Cells tab cuts pictures three ways: from crops already written to
disk, from a merged array located by a labelled plane, or from a merged
array located by a row in the measurement database. Which settings mean
anything depends entirely on which of those is chosen, and the panel used
to offer settings from all three at once -- including two different
fields for the same plane, and a coordinate-column picker for a fact the
object type already decides.
"""

from __future__ import annotations

import pytest

from spacr import picture_settings as ps
from spacr.crops import LOAD_IMAGES, STREAM_FROM_DB, STREAM_IMAGES


def _values(offered):
    return [o[0] if isinstance(o, tuple) else o for o in offered]


def test_all_three_modes_are_offered():
    """The dropdown built its own pair and left the database route out.

    `modes()` has held three entries since the database route shipped, but
    the panel's option list was a separate literal naming two of them, so
    no amount of filling in the database settings could reach a mode the
    user could not pick.
    """
    assert _values(ps.offered_values("crop_source")) == [
        LOAD_IMAGES, STREAM_IMAGES, STREAM_FROM_DB]


@pytest.mark.parametrize("mode, live", [
    (LOAD_IMAGES, {"crop_shape"}),
    (STREAM_IMAGES, {"crop_shape", "object_array"}),
    (STREAM_FROM_DB, {"object_type"}),
])
def test_each_mode_asks_for_exactly_its_own_settings(mode, live):
    asked = {"crop_shape", "object_array", "object_type"}
    assert {k for k in asked if ps.applies_to(k, mode)} == live


def test_the_plane_is_one_field_not_two():
    """`object_array` and `mask_array` asked the same question twice.

    They were described differently -- one "the mask plane the intensity
    channels are cut by", one "the labelled plane the object number is
    read from" -- and did the same job, so the panel could hold two
    answers that disagreed.
    """
    assert "object_array" in ps.ALL_KEYS
    assert "mask_array" not in ps.ALL_KEYS


def test_the_coordinate_column_is_derived_not_asked():
    """The object type names the columns, so asking again invites a clash."""
    assert "coordinate_columns" not in ps.ALL_KEYS
    # The derivation still has something to check itself against.
    assert callable(ps.available_coordinate_columns)


def test_only_the_database_route_is_forced_to_a_box():
    assert ps.bounding_box_only({"crop_source": STREAM_FROM_DB})
    assert not ps.bounding_box_only({"crop_source": STREAM_IMAGES})
    assert not ps.bounding_box_only({"crop_source": LOAD_IMAGES})


def test_the_array_route_cuts_by_the_plane_it_was_given():
    cut = ps.to_crop_settings({"crop_source": STREAM_IMAGES,
                               "object_array": "2", "crop_shape": "object"})
    assert cut["stream_method"] == "array"
    assert cut["mask_array"] == 2
    assert cut["use_bounding_box"] is False


def test_the_array_route_takes_an_object_name_for_its_plane():
    """Blank means the type's own plane; a name asks for a named one."""
    cut = ps.to_crop_settings({"crop_source": STREAM_IMAGES,
                               "object_array": "nucleus"})
    assert cut["object_array"] == "nucleus"
    assert "mask_array" not in cut


def test_the_database_route_locates_by_object_type():
    cut = ps.to_crop_settings({"crop_source": STREAM_FROM_DB,
                               "object_type": "nucleus"})
    assert cut["stream_method"] == "column"
    assert cut["object_array"] == "nucleus"


def test_the_database_route_returns_a_box_even_when_asked_for_an_outline():
    """It has no outline to follow, so it says so rather than pretending."""
    cut = ps.to_crop_settings({"crop_source": STREAM_FROM_DB,
                               "object_type": "cell", "crop_shape": "object"})
    assert cut["use_bounding_box"] is True


def test_the_array_route_ignores_a_plane_meant_for_the_other_one():
    """A leftover object_type must not steer the array route."""
    cut = ps.to_crop_settings({"crop_source": STREAM_IMAGES,
                               "object_type": "pathogen"})
    assert cut.get("object_array") != "pathogen"


def test_a_greyed_control_says_which_mode_silenced_it():
    reason = ps.why_not("object_array", STREAM_FROM_DB)
    assert reason and "array route" in reason
