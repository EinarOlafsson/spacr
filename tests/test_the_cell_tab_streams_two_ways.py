"""Two ways to locate a streamed crop, and what each can cut.

Both cut from ``merged/*.npy`` and both use the channels the montage
already asks for. What differs is how the object is found in the field:

1. A COORDINATE COLUMN. The measurement table names the object, and the
   column is always called after the object type -- so the setting is the
   object TYPE and the object number, and nothing else. There is no
   outline in a table, so this route can only cut the recorded box.
2. A MASK ARRAY. The object number IS the label at that object's pixels
   in the labelled plane, so this route can cut the box or follow the
   outline.

The two must produce the same picture when both are asked for a box.
"""
from __future__ import annotations

import pytest

from spacr.picture_settings import (LOAD_IMAGES, OWN_DEFAULTS, STREAM_IMAGES,
                                    ALL_KEYS, applies_to, bounding_box_only,
                                    offered_values, to_crop_settings, why_not)


def _streaming(**overrides):
    return dict(OWN_DEFAULTS, crop_source=STREAM_IMAGES, **overrides)


def test_the_panel_offers_the_three_settings():
    """Object type, mask array and bounding box, in the settings panel."""
    for key in ("object_type", "mask_array", "crop_shape"):
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


def test_the_mask_array_route_can_cut_either(monkeypatch):
    """A labelled plane carries the outline, so both cuts are available."""
    shaped = _streaming(object_type="cell", mask_array=2,
                        crop_shape="object")

    # A MASK ARRAY OVERRIDES THE COLUMNS. Both may be present -- the table
    # is how the object is named either way -- and the labelled plane is
    # what makes the outline available.
    assert not bounding_box_only(shaped)
    cut = to_crop_settings(shaped)
    assert cut["stream_method"] == "array"
    assert cut["mask_array"] == 2
    assert cut["use_bounding_box"] is False

    boxed = _streaming(object_type="cell", mask_array=2, crop_shape="bbox")
    assert to_crop_settings(boxed)["use_bounding_box"] is True


def test_both_routes_ask_for_the_same_object_and_the_same_box():
    """Same object type, same cut -- the two routes are two ways to find
    one crop, not two crops."""
    from spacr.crops import STREAM_FROM_DB

    column = to_crop_settings(dict(OWN_DEFAULTS, crop_source=STREAM_FROM_DB,
                                   object_type="pathogen"))
    array = to_crop_settings(
        _streaming(object_type="pathogen", mask_array=1, crop_shape="bbox"))

    assert column["object_array"] == array["object_array"]
    assert column["use_bounding_box"] == array["use_bounding_box"] is True


def test_neither_setting_means_anything_when_the_crops_are_on_disk():
    """Greyed with a reason, never hidden."""
    for key in ("mask_array", "crop_shape"):
        assert not applies_to(key, LOAD_IMAGES)
        assert why_not(key, LOAD_IMAGES).strip()
        assert applies_to(key, STREAM_IMAGES)
