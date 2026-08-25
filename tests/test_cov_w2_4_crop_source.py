"""Crop source — every spelling, every refusal, and the pixels that come out.

This module is the door every computer-vision run goes through, and it has
been broken twice by a rename that left it behind. So the aliases are
asserted exhaustively rather than by sample, and each refusal is asserted
to NAME the setting to change: a message that says only "invalid" costs the
user the hour it takes to find out which of six settings it meant.

The cutting functions are driven on real arrays with real labels, and what
is asserted is the pixels -- an object-shaped crop that still carried its
neighbour's signal would pass any shape-only check.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import crop_source as cs
from spacr.crop_source import CropSourceError


# ---------------------------------------------------------------------------
# resolve_source
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stored,expected", sorted(
    cs.CROP_SOURCE_ALIASES.items()))
def test_every_spelling_any_panel_ever_wrote_still_resolves(stored, expected):
    assert cs.resolve_source({"crop_source": stored}) == expected


def test_an_unset_source_means_load_images(monkeypatch):
    assert cs.resolve_source({}) == "png"
    assert cs.resolve_source({"crop_source": ""}) == "png"
    assert cs.resolve_source({"crop_source": None}) == "png"


def test_the_stored_value_is_read_case_and_space_insensitively():
    assert cs.resolve_source({"crop_source": "  ON_DEMAND "}) == "merged"


def test_an_unknown_source_is_refused_with_the_spellings_it_knows():
    with pytest.raises(CropSourceError) as caught:
        cs.resolve_source({"crop_source": "whatever_is_there"})
    message = str(caught.value)
    assert "whatever_is_there" in message
    assert "load_images" in message and "on_demand" in message


# ---------------------------------------------------------------------------
# inapplicable_settings
# ---------------------------------------------------------------------------

def test_greying_lists_the_other_sources_settings_and_not_its_own():
    greyed = cs.inapplicable_settings("png")
    assert "path_string" not in greyed
    assert "file_type" not in greyed
    assert "extract_channels" in greyed
    assert "coordinate_columns" in greyed


def test_a_setting_two_other_sources_share_is_listed_once():
    greyed = cs.inapplicable_settings("png")
    assert greyed.count("extract_channels") == 1
    assert len(greyed) == len(set(greyed))


def test_greying_accepts_the_stored_spelling_not_only_the_resolved_one():
    """The panel has the settings-file value in hand, not our name for it."""
    assert cs.inapplicable_settings("on_demand") == cs.inapplicable_settings(
        "merged")


def test_greying_refuses_a_source_that_is_not_one():
    with pytest.raises(CropSourceError, match="not one of"):
        cs.inapplicable_settings("sideways")


# ---------------------------------------------------------------------------
# normalise_extension / matches_path / select_crops
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("given,expected", [
    ("", ""), (None, ""), (".TIF", "tif"), ("tif", "tif"), ("tiff", "tiff"),
    ("PNG", "png"), ("cell_png", "png"), ("nucleus_tif", "tif"),
])
def test_a_file_type_becomes_a_bare_lowercase_extension(given, expected):
    assert cs.normalise_extension(given) == expected


def test_an_extension_spacr_cannot_read_names_the_ones_it_can():
    with pytest.raises(CropSourceError) as caught:
        cs.normalise_extension("czi")
    assert "czi" in str(caught.value)
    assert "png" in str(caught.value) and "npy" in str(caught.value)


def test_the_object_filter_and_the_format_filter_are_independent():
    """The whole point of splitting them: either one alone must work."""
    path = "/data/plate1/cell_png/A01_1_3.png"
    assert cs.matches_path(path, path_string="cell_png") is True
    assert cs.matches_path(path, path_string="nucleus_png") is False
    assert cs.matches_path(path, file_type="png") is True
    assert cs.matches_path(path, file_type="tif") is False
    # No filter at all keeps everything.
    assert cs.matches_path(path) is True


def test_tif_and_tiff_are_the_same_format():
    assert cs.matches_path("/d/a.tiff", file_type="tif") is True
    # ...but the reverse is a different, stricter request.
    assert cs.matches_path("/d/a.tif", file_type="tiff") is False


def test_select_crops_keeps_the_order_it_was_given():
    paths = ["/d/cell_png/c.png", "/d/nucleus_png/a.png",
             "/d/cell_png/b.tif", "/d/cell_png/a.png"]
    assert cs.select_crops(paths, {"path_string": "cell_png",
                                   "file_type": "png"}) == [
        "/d/cell_png/c.png", "/d/cell_png/a.png"]


def test_select_crops_still_reads_the_old_png_type_spelling():
    """Every old settings CSV holds it; refusing it refuses those runs."""
    paths = ["/d/cell_png/a.png", "/d/nucleus_png/a.png"]
    assert cs.select_crops(paths, {"png_type": "cell_png"}) == [
        "/d/cell_png/a.png"]


# ---------------------------------------------------------------------------
# _as_indices
# ---------------------------------------------------------------------------

def test_planes_may_be_one_number_a_list_or_a_numpy_integer():
    assert cs._as_indices(2, "extract_channels") == [2]
    assert cs._as_indices(np.int64(3), "extract_channels") == [3]
    assert cs._as_indices([0, 2], "extract_channels") == [0, 2]
    assert cs._as_indices(np.array([1, 0]), "extract_channels") == [1, 0]


def test_unset_planes_name_the_setting_that_is_unset():
    with pytest.raises(CropSourceError, match="extract_channels is not set"):
        cs._as_indices(None, "extract_channels")


def test_planes_that_are_not_numbers_are_refused():
    with pytest.raises(CropSourceError, match="not a list of planes"):
        cs._as_indices(["cell", "nucleus"], "extract_channels")


# ---------------------------------------------------------------------------
# object_bounds / crop_object
# ---------------------------------------------------------------------------

@pytest.fixture
def merged():
    """(24, 24, 3): two intensity planes and a label plane with two cells."""
    rows, cols = np.indices((24, 24))
    mask = np.zeros((24, 24), dtype=np.uint16)
    mask[4:10, 5:11] = 1
    mask[14:20, 15:21] = 2
    array = np.stack([(rows * 24 + cols).astype(np.float32),
                      np.full((24, 24), 7.0, dtype=np.float32),
                      mask.astype(np.float32)], axis=2)
    return array


def test_bounds_are_half_open_so_they_slice_directly(merged):
    mask = merged[:, :, 2]
    assert cs.object_bounds(mask, 1) == (4, 10, 5, 11)
    assert cs.object_bounds(mask, 2) == (14, 20, 15, 21)
    assert cs.object_bounds(mask, 9) is None


def test_a_bounding_box_crop_keeps_the_rectangle_and_its_channel_order(
        merged):
    mask = merged[:, :, 2]
    cut = cs.crop_object(merged, mask, 1, channels=[1, 0])
    assert cut.shape == (6, 6, 2)
    assert cut.dtype == np.float32
    assert np.all(cut[:, :, 0] == 7.0)
    assert cut[0, 0, 1] == 4 * 24 + 5


def test_an_object_shaped_crop_zeroes_everything_outside_the_object(merged):
    mask = merged[:, :, 2]
    mask[6, 5] = 0                       # a notch bitten out of cell 1
    cut = cs.crop_object(merged, mask, 1, channels=[1], shape="object")
    assert cut[2, 0, 0] == 0.0
    assert cut[0, 0, 0] == 7.0


def test_padding_grows_the_box_and_stops_at_the_array_edge(merged):
    mask = merged[:, :, 2]
    unpadded = cs.crop_object(merged, mask, 2, channels=[0])
    padded = cs.crop_object(merged, mask, 2, channels=[0], padding=6)
    assert unpadded.shape[:2] == (6, 6)
    # 6 px of padding on each side would be 18, but the array runs out at 24.
    assert padded.shape[:2] == (16, 15)


def test_a_size_resizes_the_crop_to_a_square(merged):
    mask = merged[:, :, 2]
    cut = cs.crop_object(merged, mask, 1, channels=[0, 1], size=12)
    assert cut.shape == (12, 12, 2)


def test_a_crop_already_the_right_size_is_returned_untouched(merged):
    mask = merged[:, :, 2]
    cut = cs.crop_object(merged, mask, 1, channels=[0], size=6)
    assert cut.shape == (6, 6, 1)
    assert cut[0, 0, 0] == 4 * 24 + 5


def test_an_absent_object_is_none_rather_than_an_empty_crop(merged):
    mask = merged[:, :, 2]
    assert cs.crop_object(merged, mask, 99, channels=[0]) is None


def test_a_crop_shape_that_is_not_one_is_refused(merged):
    with pytest.raises(CropSourceError, match="crop_shape"):
        cs.crop_object(merged, merged[:, :, 2], 1, channels=[0],
                       shape="outline")


def test_a_two_dimensional_array_is_not_a_merged_stack(merged):
    with pytest.raises(CropSourceError, match="height, width, planes"):
        cs.crop_object(merged[:, :, 0], merged[:, :, 2], 1, channels=[0])


def test_a_plane_the_array_does_not_have_is_refused_before_cutting(merged):
    with pytest.raises(CropSourceError, match="plane 7"):
        cs.crop_object(merged, merged[:, :, 2], 1, channels=[0, 7])


# ---------------------------------------------------------------------------
# crop_at
# ---------------------------------------------------------------------------

def test_a_coordinate_crop_is_centred_and_rounded_half_to_even(merged):
    cut = cs.crop_at(merged, 10.5, 10.5, channels=[0], size=4)
    # 10.5 rounds to 10, so rows 8..12 and columns 8..12.
    assert cut.shape == (4, 4, 1)
    assert cut[0, 0, 0] == 8 * 24 + 8


def test_the_side_is_rounded_down_to_even_and_never_below_two(merged):
    assert cs.crop_at(merged, 12, 12, channels=[0], size=5).shape[:2] == (4, 4)
    assert cs.crop_at(merged, 12, 12, channels=[0], size=0).shape[:2] == (2, 2)


def test_an_edge_coordinate_yields_a_smaller_crop_not_a_padded_one(merged):
    cut = cs.crop_at(merged, 0, 0, channels=[0], size=8)
    assert cut.shape[:2] == (4, 4)


def test_a_box_entirely_off_the_array_is_none(merged):
    assert cs.crop_at(merged, -50, 5, channels=[0], size=4) is None
    assert cs.crop_at(merged, 5, 500, channels=[0], size=4) is None


def test_a_coordinate_crop_with_no_planes_named_is_refused(merged):
    with pytest.raises(CropSourceError, match="extract_channels is not set"):
        cs.crop_at(merged, 5, 5, channels=None, size=4)


# ---------------------------------------------------------------------------
# mask_plane_for
# ---------------------------------------------------------------------------

def test_the_mask_plane_comes_from_the_setting_the_mask_step_wrote():
    assert cs.mask_plane_for("cell", {"cell_mask_dim": 3}) == 3
    assert cs.mask_plane_for(" Nucleus ", {"nucleus_mask_dim": "2"}) == 2


def test_an_object_with_no_mask_plane_names_the_setting_to_add():
    with pytest.raises(CropSourceError) as caught:
        cs.mask_plane_for("pathogen", {"cell_mask_dim": 3})
    assert "pathogen_mask_dim" in str(caught.value)


def test_a_mask_plane_that_is_not_a_number_names_the_setting():
    with pytest.raises(CropSourceError, match="cell_mask_dim"):
        cs.mask_plane_for("cell", {"cell_mask_dim": "the third one"})


# ---------------------------------------------------------------------------
# crops_from_merged
# ---------------------------------------------------------------------------

def test_every_label_in_the_mask_gets_a_crop(merged):
    got = cs.crops_from_merged(merged, {"object_array": "cell",
                                        "cell_mask_dim": 2,
                                        "extract_channels": [0, 1],
                                        "image_size": 8})
    assert [label for label, _ in got] == [1, 2]
    assert all(image.shape == (8, 8, 2) for _, image in got)


def test_asking_for_specific_labels_skips_the_ones_that_are_absent(merged):
    got = cs.crops_from_merged(merged, {"object_array": "cell",
                                        "cell_mask_dim": 2,
                                        "extract_channels": [0]},
                               labels=[2, 99])
    assert [label for label, _ in got] == [2]


def test_a_mask_plane_beyond_the_array_is_refused_with_both_numbers(merged):
    with pytest.raises(CropSourceError) as caught:
        cs.crops_from_merged(merged, {"object_array": "cell",
                                      "cell_mask_dim": 9,
                                      "extract_channels": [0]})
    assert "9" in str(caught.value) and "3 plane" in str(caught.value)


# ---------------------------------------------------------------------------
# stream_planes
# ---------------------------------------------------------------------------

def test_the_current_spelling_wins_over_the_older_one():
    assert cs.stream_planes({"channel_arrays": [1, 2],
                             "extract_channels": [0]}) == [1, 2]


def test_the_older_spelling_is_still_read():
    assert cs.stream_planes({"extract_channels": [0, 3]}) == [0, 3]


def test_neither_spelling_set_names_the_one_to_set():
    with pytest.raises(CropSourceError, match="channel_arrays is not set"):
        cs.stream_planes({"crop_shape": "object"})


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def test_loading_images_needs_nothing_but_a_readable_file_type():
    assert cs.validate({"crop_source": "png"}) == "png"
    assert cs.validate({"crop_source": "png", "file_type": "cell_tif"}) == "png"


def test_loading_images_refuses_a_file_type_it_cannot_read():
    with pytest.raises(CropSourceError, match="czi"):
        cs.validate({"crop_source": "png", "file_type": "czi"})


def test_streaming_needs_planes_a_shape_and_a_mask_plane():
    settings = {"crop_source": "merged", "channel_arrays": [0, 1],
                "object_array": "cell", "cell_mask_dim": 2}
    assert cs.validate(settings) == "merged"


def test_streaming_without_a_mask_plane_is_refused_before_the_run():
    with pytest.raises(CropSourceError, match="cell_mask_dim"):
        cs.validate({"crop_source": "merged", "channel_arrays": [0]})


def test_streaming_refuses_a_crop_shape_that_is_not_one():
    with pytest.raises(CropSourceError, match="crop_shape"):
        cs.validate({"crop_source": "merged", "channel_arrays": [0],
                     "crop_shape": "outline"})


def test_a_coordinate_run_needs_an_image_size_and_no_mask_plane():
    """A coordinate has no mask, so demanding one would refuse a valid run."""
    settings = {"crop_source": "merged", "channel_arrays": [0, 1],
                "coordinate_columns": ["cell_id"], "image_size": 64}
    assert cs.validate(settings) == "merged"


def test_a_coordinate_run_without_a_size_has_no_box_to_cut():
    with pytest.raises(CropSourceError, match="image_size"):
        cs.validate({"crop_source": "merged", "channel_arrays": [0],
                     "coordinate_columns": ["centroid_y", "centroid_x"]})


def test_a_coordinate_run_cannot_ask_for_object_shaped_crops():
    with pytest.raises(CropSourceError, match="no outline"):
        cs.validate({"crop_source": "merged", "channel_arrays": [0],
                     "coordinate_columns": ["cell_id"], "image_size": 64,
                     "crop_shape": "object"})


def test_generating_is_validated_like_streaming():
    assert cs.validate({"crop_source": "generate", "channel_arrays": [0],
                        "object_array": "cell",
                        "cell_mask_dim": 2}) == "generate"
