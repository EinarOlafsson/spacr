"""Where a classifier's training images come from.

Pipeline code, so Qt-free and tested without a GUI.
"""
import numpy as np
import pytest

from spacr.crop_source import (
    CROP_SOURCES, CropSourceError, crop_at, crop_object, crops_from_merged,
    inapplicable_settings, mask_plane_for, matches_path, normalise_extension,
    resolve_source, select_crops, validate,
)


# ---------------------------------------------------------------------------
# Which source
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("given, expected", [
    ("pre_generated", "pre_generated"),
    ("on_demand", "on_demand"),
    ("generate", "generate"),
    ("auto", "pre_generated"),
    ("", "pre_generated"),
    (None, "pre_generated"),
])
def test_the_source_is_resolved_explicitly(given, expected):
    assert resolve_source({"crop_source": given}) == expected


def test_an_unknown_source_is_refused():
    """Guessing would train on a different set of images and report success."""
    with pytest.raises(CropSourceError, match="not one of"):
        resolve_source({"crop_source": "magic"})


def test_each_source_greys_the_others_settings():
    greyed = inapplicable_settings("pre_generated")
    assert "extract_channels" in greyed and "object_array" in greyed
    assert "path_string" not in greyed

    greyed = inapplicable_settings("on_demand")
    assert "path_string" in greyed and "file_type" in greyed
    assert "extract_channels" not in greyed


# ---------------------------------------------------------------------------
# file_type as an extension filter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("given, expected", [
    ("png", "png"), (".PNG", "png"), ("tif", "tif"), ("TIFF", "tiff"),
    ("", ""),
])
def test_file_type_names_an_extension(given, expected):
    assert normalise_extension(given) == expected


def test_the_old_object_png_value_still_reads():
    """It is what every settings CSV written before this holds."""
    assert normalise_extension("cell_png") == "png"
    assert normalise_extension("nucleus_png") == "png"


def test_a_format_spacr_cannot_read_is_named():
    with pytest.raises(CropSourceError, match="not an image type"):
        normalise_extension("psd")


def test_path_and_format_are_independent_filters():
    """One setting could never express "every nucleus crop, whatever format"
    or "every TIFF, whatever object"."""
    paths = ["/a/cell_png/1.png", "/a/nucleus_png/2.png",
             "/a/nucleus_png/3.tif"]

    assert select_crops(paths, {"path_string": "nucleus_png"}) == paths[1:]
    assert select_crops(paths, {"file_type": "tif"}) == [paths[2]]
    assert select_crops(paths, {"path_string": "nucleus_png",
                                "file_type": "png"}) == [paths[1]]


def test_png_type_is_still_accepted_as_the_path_filter():
    paths = ["/a/cell_png/1.png", "/a/nucleus_png/2.png"]
    assert select_crops(paths, {"png_type": "cell_png"}) == [paths[0]]


def test_tif_and_tiff_are_the_same_format():
    assert matches_path("/a/1.tiff", file_type="tif")


# ---------------------------------------------------------------------------
# Cutting from merged
# ---------------------------------------------------------------------------

def _merged():
    """Two planes of intensity and one of masks, with two objects in it."""
    array = np.zeros((10, 10, 3), dtype=np.float32)
    array[:, :, 0] = 7.0                       # intensity
    array[:, :, 1] = 3.0                       # intensity
    array[2:5, 2:5, 2] = 1                     # object 1
    array[6:9, 6:8, 2] = 2                     # object 2
    return array


SETTINGS = {"extract_channels": [0, 1], "object_array": "cell",
            "cell_mask_dim": 2, "crop_shape": "bounding_box"}


def test_a_crop_is_the_objects_own_bounding_box():
    array = _merged()
    cut = crop_object(array, array[:, :, 2], 1, channels=[0, 1])
    assert cut.shape == (3, 3, 2), "the box is not the object's extent"
    assert np.allclose(cut[:, :, 0], 7.0)


def test_only_the_named_planes_become_channels():
    array = _merged()
    assert crop_object(array, array[:, :, 2], 1, channels=[0]).shape[2] == 1
    assert crop_object(array, array[:, :, 2], 1,
                       channels=[1, 0]).shape[2] == 2


def test_the_channel_order_is_the_order_given():
    array = _merged()
    cut = crop_object(array, array[:, :, 2], 1, channels=[1, 0])
    assert cut[0, 0, 0] == 3.0 and cut[0, 0, 1] == 7.0


def test_an_object_crop_masks_the_background_away():
    """The background around a cell is sometimes signal and sometimes
    contamination, which is why it is a choice."""
    array = _merged()
    array[2:5, 2:5, 2] = 0
    array[3, 3, 2] = 1                          # one pixel object
    cut = crop_object(array, array[:, :, 2], 1, channels=[0], shape="object")
    assert cut.shape == (1, 1, 1)

    array[2:5, 2:5, 2] = 1
    array[2, 2, 2] = 0                          # a notch out of the corner
    boxed = crop_object(array, array[:, :, 2], 1, channels=[0])
    masked = crop_object(array, array[:, :, 2], 1, channels=[0],
                         shape="object")
    assert boxed[0, 0, 0] == 7.0
    assert masked[0, 0, 0] == 0.0, "the notch kept its background"


def test_a_missing_object_gives_nothing_rather_than_an_empty_image():
    array = _merged()
    assert crop_object(array, array[:, :, 2], 99, channels=[0]) is None


def test_a_crop_can_be_resized():
    array = _merged()
    cut = crop_object(array, array[:, :, 2], 1, channels=[0, 1], size=8)
    assert cut.shape == (8, 8, 2)


def test_a_plane_the_array_does_not_have_is_named():
    array = _merged()
    with pytest.raises(CropSourceError, match="plane 9"):
        crop_object(array, array[:, :, 2], 1, channels=[9])


def test_every_object_is_cut_from_one_merged_array():
    crops = crops_from_merged(_merged(), SETTINGS)
    assert [label for label, _ in crops] == [1, 2]
    assert crops[0][1].shape == (3, 3, 2)
    assert crops[1][1].shape == (3, 2, 2)


def test_only_the_requested_objects_are_cut():
    crops = crops_from_merged(_merged(), SETTINGS, labels=[2])
    assert [label for label, _ in crops] == [2]


def test_the_mask_plane_comes_from_the_mask_step_settings():
    """So the two cannot disagree about which plane is which."""
    assert mask_plane_for("cell", {"cell_mask_dim": 2}) == 2
    assert mask_plane_for("nucleus", {"nucleus_mask_dim": 1}) == 1


def test_an_object_with_no_mask_plane_names_the_setting():
    with pytest.raises(CropSourceError, match="pathogen_mask_dim"):
        mask_plane_for("pathogen", {"cell_mask_dim": 2})


# ---------------------------------------------------------------------------
# Coordinates from a database
# ---------------------------------------------------------------------------

def test_a_coordinate_crop_is_a_fixed_box():
    array = _merged()
    cut = crop_at(array, 5, 5, channels=[0, 1], size=4)
    assert cut.shape == (4, 4, 2)


def test_a_coordinate_at_the_edge_is_clipped_not_refused():
    array = _merged()
    cut = crop_at(array, 0, 0, channels=[0], size=6)
    assert cut is not None and cut.shape[0] == 3


def test_a_coordinate_outside_the_image_gives_nothing():
    assert crop_at(_merged(), 500, 500, channels=[0], size=4) is None


# ---------------------------------------------------------------------------
# Refusing early, with the setting to change
# ---------------------------------------------------------------------------

def test_validate_passes_a_workable_on_demand_setup():
    assert validate({"crop_source": "on_demand", **SETTINGS}) == "on_demand"


def test_on_demand_without_channels_is_refused_before_training():
    """Discovering it after an hour of dataset building is a worse failure."""
    with pytest.raises(CropSourceError, match="extract_channels"):
        validate({"crop_source": "on_demand", "object_array": "cell",
                  "cell_mask_dim": 2})


def test_database_objects_can_only_be_bounding_boxes():
    """A coordinate has no outline, so a crop claiming to be object-shaped
    would be a rectangle wearing the wrong name."""
    with pytest.raises(CropSourceError, match="bounding boxes"):
        validate({"crop_source": "on_demand", "extract_channels": [0],
                  "coordinate_columns": ["y", "x"], "crop_shape": "object",
                  "image_size": 64})


def test_coordinate_crops_need_a_size():
    with pytest.raises(CropSourceError, match="image_size"):
        validate({"crop_source": "on_demand", "extract_channels": [0],
                  "coordinate_columns": ["y", "x"]})


def test_coordinate_columns_need_both_axes():
    with pytest.raises(CropSourceError, match="row and a column"):
        validate({"crop_source": "on_demand", "extract_channels": [0],
                  "coordinate_columns": ["y"], "image_size": 64})


def test_a_pre_generated_setup_with_a_bad_extension_is_refused():
    with pytest.raises(CropSourceError, match="not an image type"):
        validate({"crop_source": "pre_generated", "file_type": "psd"})
