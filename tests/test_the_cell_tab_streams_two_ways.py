"""Two ways to locate a streamed crop, and what each can cut.

Both cut from ``merged/*.npy`` and both use the channels the montage
already asks for. What differs is how the object is found in the field:

1. A COORDINATE COLUMN. The measurement table names the object, and the
   column is always called after the object type -- so the setting is the
   object TYPE and the object number, and nothing else. There is no
   outline in a table, so this route can only cut the recorded box.
2. A LABELLED PLANE. The object number IS the label at that object's
   pixels in the plane `object_array` names, so this route can cut the
   box or follow the outline.

The two must produce the same picture when both are asked for a box, and
each asks only for what its own route uses: the plane is meaningless to
the table route, and the object type is how the table route finds its
columns.
"""
from __future__ import annotations

import pytest

from spacr.picture_settings import (LOAD_IMAGES, OWN_DEFAULTS, STREAM_IMAGES,
                                    ALL_KEYS, applies_to, bounding_box_only,
                                    offered_values, to_crop_settings, why_not)


def _streaming(**overrides):
    return dict(OWN_DEFAULTS, crop_source=STREAM_IMAGES, **overrides)


def test_the_panel_offers_the_three_settings():
    """Object type, the labelled plane and the cut shape.

    The plane was `mask_array` here and `object_array` a few lines away in
    the same panel -- two fields, two descriptions, one question. It is
    `object_array` now, and there is one of it.
    """
    for key in ("object_type", "object_array", "crop_shape"):
        assert key in ALL_KEYS, f"{key} is not offered"
        assert key in OWN_DEFAULTS or key == "object_type"


def test_the_object_types_are_the_ones_a_run_can_write():
    """"object type: cell, nucleus, pathogen, ..." -- the ... included."""
    offered = offered_values("object_type")
    names = tuple(c[0] if isinstance(c, tuple) else c for c in offered)

    assert {"cell", "nucleus", "pathogen", "cytoplasm"} <= set(names)
    # The organelle planes a measure run can write, which had no column
    # and so could not be streamed at all.
    assert {"organelle", "organelleb", "organellec", "organelled"} <= set(names)


def test_every_object_type_has_its_coordinate_column():
    """The column is derived from the type, never asked for."""
    from spacr.stream_dataset import coordinate_column

    for name in offered_values("object_type"):
        key = name[0] if isinstance(name, tuple) else name
        assert coordinate_column(key) == f"{key}_id"


def test_the_coordinate_route_can_only_cut_a_box():
    """No outline in a table, so the panel must not offer one."""
    from spacr.crops import STREAM_FROM_DB

    settings = dict(OWN_DEFAULTS, crop_source=STREAM_FROM_DB,
                    object_type="nucleus")

    assert bounding_box_only(settings)
    cut = to_crop_settings(settings)
    assert cut["stream_method"] == "column"
    assert cut["object_array"] == "nucleus"
    assert cut["use_bounding_box"] is True


def test_the_labelled_plane_route_can_cut_either():
    """A labelled plane carries the outline, so both cuts are available.

    The plane is named by `object_array`; this test set `mask_array`,
    which was the duplicate field.
    """
    shaped = _streaming(object_array=2, crop_shape="object")

    assert not bounding_box_only(shaped)
    cut = to_crop_settings(shaped)
    assert cut["stream_method"] == "array"
    assert cut["mask_array"] == 2
    assert cut["use_bounding_box"] is False

    boxed = _streaming(object_array=2, crop_shape="bbox")
    assert to_crop_settings(boxed)["use_bounding_box"] is True


def test_both_routes_ask_for_the_same_object_and_the_same_box():
    """Same object type, same cut -- the two routes are two ways to find
    one crop, not two crops."""
    from spacr.crops import STREAM_FROM_DB

    column = to_crop_settings(dict(OWN_DEFAULTS, crop_source=STREAM_FROM_DB,
                                   object_type="pathogen"))
    array = to_crop_settings(
        _streaming(object_array="pathogen", crop_shape="bbox"))

    # Each route NAMES the object its own way -- the table route by the
    # type it looks the columns up from, the array route by the plane it
    # reads the labels out of -- and they arrive at the same object.
    assert column["object_array"] == array["object_array"] == "pathogen"
    assert column["use_bounding_box"] == array["use_bounding_box"] is True


def test_the_plane_means_nothing_when_the_crops_are_on_disk():
    """Greyed with a reason, never hidden.

    `crop_shape` left this test: a crop on disk still has a mask to cut
    against, so the shape is a real choice there and only the plane is
    silenced. `mask_array` left because it no longer exists.
    """
    assert not applies_to("object_array", LOAD_IMAGES)
    assert why_not("object_array", LOAD_IMAGES).strip()
    assert applies_to("object_array", STREAM_IMAGES)
