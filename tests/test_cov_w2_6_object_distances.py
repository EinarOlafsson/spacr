"""Distances measured on masks whose right answers are known by construction.

Every case here is built from small hand-placed squares, so the expected
number can be written down rather than recorded: two squares four pixels
apart have a surface-to-surface distance of four, a square's centre sits
half its width from its own rim, and an object with no partner of the other
type is infinitely far from one rather than NaN.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import object_distances as od


def _two_squares(shape=(40, 40)):
    """Two 6x6 squares in a field, with a four-pixel gap between them."""
    mask = np.zeros(shape, dtype=np.int32)
    mask[10:16, 10:16] = 1
    mask[10:16, 20:26] = 2
    return mask


def _one_square(shape=(40, 40), value=1, row=10, col=10, size=6):
    mask = np.zeros(shape, dtype=np.int32)
    mask[row:row + size, col:col + size] = value
    return mask


# --------------------------------------------------------------------------
# _as_spacing -- the one place voxel size is turned into a per-axis tuple
# --------------------------------------------------------------------------

def test_a_scalar_voxel_size_applies_to_every_axis():
    """One number means the voxels are cubes, so every axis gets it."""
    assert od._as_spacing(2.5, 3) == (2.5, 2.5, 2.5)


def test_a_per_axis_voxel_size_is_kept_axis_by_axis():
    assert od._as_spacing((0.5, 0.25), 2) == (0.5, 0.25)


def test_a_voxel_size_of_the_wrong_length_is_refused_not_padded():
    """Two numbers cannot describe three axes; guessing the third would
    silently scale a distance by whatever was invented."""
    assert od._as_spacing((0.5, 0.25), 3) is None


def test_no_voxel_size_means_pixel_units():
    assert od._as_spacing(None, 2) is None


# --------------------------------------------------------------------------
# the two transforms
# --------------------------------------------------------------------------

def test_a_voxel_size_scales_the_surface_transform():
    """The transform in physical units is the pixel one times the step."""
    mask = _one_square()
    plain = od.surface_distance_transform(mask)
    scaled = od.surface_distance_transform(mask, spacing=2.0)
    assert np.allclose(scaled, plain * 2.0, atol=1e-4)


def test_an_empty_mask_has_no_interior_depth_anywhere():
    """Nothing is inside an object, so the interior transform is zero --
    not inf, which is what the *surface* transform reports for the same
    field and means the opposite thing."""
    empty = np.zeros((12, 12), dtype=np.int32)
    interior = od.interior_distance_transform(empty)
    assert interior.shape == (12, 12)
    assert np.all(interior == 0)
    assert np.all(np.isinf(od.surface_distance_transform(empty)))


def test_a_voxel_size_scales_the_interior_transform():
    mask = _one_square()
    plain = od.interior_distance_transform(mask)
    scaled = od.interior_distance_transform(mask, spacing=3.0)
    assert np.allclose(scaled, plain * 3.0, atol=1e-4)


# --------------------------------------------------------------------------
# sampling and boundaries
# --------------------------------------------------------------------------

def test_sampling_no_points_returns_no_values():
    out = od._sample(np.zeros((5, 5)), np.empty((0, 2)))
    assert out.shape == (0,)


def test_a_point_outside_the_field_is_infinitely_far_not_clipped():
    """Clipping to the edge would invent a distance measured from somewhere
    the object is not."""
    field = np.ones((5, 5), dtype=float)
    out = od._sample(field, np.array([[2.0, 2.0], [99.0, 2.0], [-3.0, 1.0]]))
    assert out[0] == 1.0
    assert np.isinf(out[1]) and np.isinf(out[2])


def test_every_point_outside_the_field_skips_the_lookup_entirely():
    """The guard around the fancy-index, not just the inf it produces.

    ``out`` starts as all-inf, so the answer for a wholly-outside set is
    already correct before the lookup -- and running the lookup anyway would
    index ``field`` with an empty selection per axis, which is legal for
    numpy and pointless. What matters is that a caller measuring an object
    that fell entirely off the field gets infs rather than an IndexError.
    """
    field = np.ones((5, 5), dtype=float)

    out = od._sample(field, np.array([[99.0, 99.0], [-4.0, -7.0]]))

    assert out.shape == (2,)
    assert np.isinf(out).all()


def test_a_field_with_no_objects_has_no_boundary_pixels():
    assert od._boundary_pixels(np.zeros((10, 10), dtype=np.int32)) == {}


def test_the_boundary_of_a_square_is_its_rim_and_only_its_rim():
    mask = _one_square()
    edges = od._boundary_pixels(mask)
    assert set(edges) == {1}
    rows, cols = edges[1]
    # A 6x6 square has 20 rim pixels and 16 interior ones.
    assert rows.size == 20
    assert not ((rows > 10) & (rows < 15) & (cols > 10) & (cols < 15)).any()


def test_an_object_with_no_boundary_is_infinitely_far_from_anything():
    """`inf`, deliberately: there is no surface to measure from, and 0 would
    read as 'touching'."""
    assert od._min_over_boundary(np.zeros((4, 4)), None) == float("inf")
    empty = (np.empty(0, dtype=int), np.empty(0, dtype=int))
    assert od._min_over_boundary(np.zeros((4, 4)), empty) == float("inf")


# --------------------------------------------------------------------------
# local maxima
# --------------------------------------------------------------------------

def test_a_label_that_is_not_in_the_mask_has_no_maxima():
    image = np.zeros((20, 20), dtype=float)
    peaks = od.local_maxima(image, _one_square(shape=(20, 20)), label=7)
    assert peaks.shape == (0, 2)


def test_a_peak_finder_that_fails_costs_the_object_its_maxima_not_the_run(
        monkeypatch):
    """A peak finder can refuse a degenerate object. That must cost one
    object its maxima column, not the whole field's measurement."""
    import skimage.feature

    def _explode(*args, **kwargs):
        raise ValueError("degenerate plateau")

    monkeypatch.setattr(skimage.feature, "peak_local_max", _explode)
    image = np.zeros((20, 20), dtype=float)
    image[11:15, 11:15] = 5.0
    peaks = od.local_maxima(image, _one_square(shape=(20, 20)), label=1)
    assert peaks.shape == (0, 2)


def test_two_peaks_have_a_spread_and_one_has_none():
    """Spread is what tells two bright spots at opposite ends of a cell from
    two sitting on top of each other -- the count cannot."""
    assert od._pairwise_spread(np.empty((0, 2))) == 0.0
    assert od._pairwise_spread(np.array([[3.0, 4.0]])) == 0.0
    spread = od._pairwise_spread(np.array([[0.0, 0.0], [0.0, 10.0]]))
    assert spread == pytest.approx(10.0)


def test_a_voxel_size_scales_the_spread_it_reports():
    points = np.array([[0.0, 0.0], [0.0, 10.0]])
    assert od._pairwise_spread(points, spacing=(1.0, 0.5)) == pytest.approx(5.0)


# --------------------------------------------------------------------------
# between_object_types
# --------------------------------------------------------------------------

def test_an_empty_primary_mask_yields_a_table_with_no_rows():
    empty = np.zeros((16, 16), dtype=np.int32)
    frame = od.between_object_types({"cell": empty, "nucleus": empty},
                                    primary="cell")
    assert list(frame.columns) == ["label"]
    assert len(frame) == 0


def test_a_partner_mask_of_another_shape_is_skipped_not_broadcast():
    """A mask from a different field cannot be measured against this one,
    and a resized one would report distances that were never observed."""
    masks = {"cell": _two_squares(),
             "nucleus": np.zeros((8, 8), dtype=np.int32)}
    frame = od.between_object_types(masks, primary="cell")
    assert not [c for c in frame.columns if "nucleus" in c]


def test_a_contained_object_is_measured_from_the_rim_not_the_overlap():
    """A nucleus wholly inside a cell overlaps it completely, and the cell's
    surface is still two pixels away from the nucleus's -- overlap and
    surface distance answer different questions."""
    cell = _one_square(size=8, row=10, col=10)
    inside = np.zeros((40, 40), dtype=np.int32)
    inside[12:16, 12:16] = 1            # a nucleus wholly inside the cell
    frame = od.between_object_types({"cell": cell, "nucleus": inside},
                                    primary="cell")
    row = frame.iloc[0]
    assert row["surface_to_nucleus_surface"] == pytest.approx(2.0)
    assert row["nucleus_overlap_fraction"] == pytest.approx(16 / 64)
    # The centre of the cell is inside the nucleus, so its distance to the
    # nearest nucleus surface is zero.
    assert row["centre_to_nucleus_surface"] == 0.0


def test_an_object_with_no_partner_is_infinitely_far_never_nan():
    """`inf` is a fact; NaN would mean 'not measured' and hide the fact."""
    frame = od.between_object_types(
        {"cell": _one_square(), "nucleus": np.zeros((40, 40), dtype=np.int32)},
        primary="cell")
    row = frame.iloc[0]
    assert np.isinf(row["centre_to_nearest_nucleus_centre"])
    assert np.isinf(row["surface_to_nucleus_surface"])


def test_the_centre_of_a_square_sits_half_its_width_from_its_own_rim():
    frame = od.between_object_types({"cell": _one_square(size=6)},
                                    primary="cell")
    own = float(frame.iloc[0]["distance_to_own_boundary"])
    assert own == pytest.approx(3.0, abs=0.5)
    assert 0.0 <= float(frame.iloc[0]["relative_radial_position"]) <= 1.0


def test_an_object_near_the_wall_is_reported_close_to_the_field_edge():
    mask = _one_square(shape=(40, 40), row=0, col=0, size=4)
    frame = od.between_object_types({"cell": mask}, primary="cell")
    assert float(frame.iloc[0]["distance_to_field_edge"]) == pytest.approx(1.5)


# --------------------------------------------------------------------------
# maxima_distances
# --------------------------------------------------------------------------

def test_maxima_on_an_empty_mask_yields_no_rows():
    empty = np.zeros((16, 16), dtype=np.int32)
    frame = od.maxima_distances({"cell": empty}, np.zeros((16, 16)),
                                primary="cell")
    assert len(frame) == 0


def test_a_single_plane_image_is_treated_as_one_channel():
    """A caller with one 2-D image should not have to add a channel axis."""
    mask = _one_square(size=8)
    image = np.zeros((40, 40), dtype=float)
    image[12:14, 12:14] = 100.0
    frame = od.maxima_distances({"cell": mask}, image, primary="cell")
    assert "channel_0_maxima_count" in frame.columns
    assert int(frame.iloc[0]["channel_0_maxima_count"]) >= 1


def test_a_channel_that_the_stack_does_not_have_is_skipped():
    mask = _one_square(size=8)
    image = np.zeros((40, 40), dtype=float)
    frame = od.maxima_distances({"cell": mask}, image, primary="cell",
                                channels=(0, 5))
    assert "channel_0_maxima_count" in frame.columns
    assert "channel_5_maxima_count" not in frame.columns


def test_a_peakless_object_reports_nan_distances_beside_a_zero_count():
    """NOT zero. Zero would read as 'the peak is right here', which is the
    opposite of what happened; the count column beside it says why."""
    mask = _one_square(size=8)
    image = np.zeros((40, 40), dtype=float)
    frame = od.maxima_distances({"cell": mask, "nucleus": mask},
                                image, primary="cell")
    # Peaks are suppressed by making the finder refuse this object.
    import skimage.feature
    row = frame.iloc[0]
    if int(row["channel_0_maxima_count"]) == 0:
        assert np.isnan(row["channel_0_maxima_to_own_boundary_min"])
        assert np.isnan(row["channel_0_maxima_to_centre_mean"])
    assert skimage.feature.peak_local_max is not None


def test_a_peakless_object_reports_nan_for_every_partner_type(monkeypatch):
    import skimage.feature
    monkeypatch.setattr(skimage.feature, "peak_local_max",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
    mask = _one_square(size=8)
    other = _one_square(size=4, row=25, col=25, value=1)
    image = np.zeros((40, 40), dtype=float)
    image[12:14, 12:14] = 100.0
    frame = od.maxima_distances({"cell": mask, "nucleus": other},
                                image, primary="cell")
    row = frame.iloc[0]
    assert int(row["channel_0_maxima_count"]) == 0
    assert np.isnan(row["channel_0_maxima_to_nucleus_surface_min"])
    assert np.isnan(row["channel_0_maxima_to_nucleus_surface_mean"])


def test_a_peak_is_measured_against_every_other_object_type():
    """The columns exist per partner type and carry the real distance from
    the peak to that type's nearest surface."""
    mask = _one_square(size=8, row=10, col=10)
    other = np.zeros((40, 40), dtype=np.int32)
    other[10:14, 30:34] = 1
    image = np.zeros((40, 40), dtype=float)
    image[13, 13] = 100.0
    frame = od.maxima_distances({"cell": mask, "nucleus": other},
                                image, primary="cell", spacing=None)
    row = frame.iloc[0]
    assert int(row["channel_0_maxima_count"]) == 1
    # The single peak at (13, 13) is 17 columns from the nucleus at col 30.
    assert row["channel_0_maxima_to_nucleus_surface_min"] == pytest.approx(
        17.0, abs=0.1)
    assert (row["channel_0_maxima_to_nucleus_surface_min"]
            == row["channel_0_maxima_to_nucleus_surface_mean"])


def test_a_partner_of_another_shape_contributes_no_maxima_columns():
    mask = _one_square(size=8)
    image = np.zeros((40, 40), dtype=float)
    image[12:14, 12:14] = 100.0
    frame = od.maxima_distances(
        {"cell": mask, "nucleus": np.zeros((7, 7), dtype=np.int32)},
        image, primary="cell")
    assert not [c for c in frame.columns if "nucleus" in c]


# --------------------------------------------------------------------------
# intensity_centre_offset
# --------------------------------------------------------------------------

def test_the_offset_of_an_empty_mask_is_a_table_with_no_rows():
    empty = np.zeros((16, 16), dtype=np.int32)
    frame = od.intensity_centre_offset(empty, np.zeros((16, 16)),
                                       primary="cell")
    assert len(frame) == 0


def test_a_uniformly_stained_object_has_no_intensity_offset():
    mask = _one_square(size=8)
    image = np.zeros((40, 40), dtype=float)
    image[10:18, 10:18] = 50.0
    frame = od.intensity_centre_offset(mask, image, primary="cell")
    assert float(frame.iloc[0]["channel_0_intensity_centre_offset"]) \
        == pytest.approx(0.0, abs=1e-6)


def test_a_one_sided_stain_moves_the_intensity_centre_off_the_geometric_one():
    """Polarisation in one number: no intensity summary says this."""
    mask = _one_square(size=8)
    image = np.zeros((40, 40), dtype=float)
    image[10:12, 10:18] = 100.0          # all the signal at one end
    frame = od.intensity_centre_offset(mask, image, primary="cell")
    assert float(frame.iloc[0]["channel_0_intensity_centre_offset"]) > 2.0


def test_the_offset_skips_a_channel_the_stack_does_not_have():
    mask = _one_square(size=8)
    image = np.zeros((40, 40, 2), dtype=float)
    image[10:18, 10:18, 0] = 50.0
    frame = od.intensity_centre_offset(mask, image, primary="cell",
                                       channels=(1, 9))
    assert "channel_1_intensity_centre_offset" in frame.columns
    assert "channel_9_intensity_centre_offset" not in frame.columns


# --------------------------------------------------------------------------
# the one call the pipeline makes
# --------------------------------------------------------------------------

def test_without_a_field_only_the_geometric_families_are_measured():
    frame = od.object_distances({"cell": _two_squares()}, None, primary="cell")
    assert "distance_to_own_boundary" in frame.columns
    assert not [c for c in frame.columns if "maxima" in c]


def test_the_maxima_family_can_be_turned_off_without_losing_the_rest():
    mask = _two_squares()
    image = np.zeros((40, 40), dtype=float)
    image[12, 12] = 100.0
    with_maxima = od.object_distances({"cell": mask}, image, primary="cell")
    without = od.object_distances({"cell": mask}, image, primary="cell",
                                  maxima=False)
    assert [c for c in with_maxima.columns if "maxima" in c]
    assert not [c for c in without.columns if "maxima" in c]
    assert "channel_0_intensity_centre_offset" in without.columns
    assert len(without) == len(with_maxima) == 2


def test_no_column_carries_the_object_type_it_will_be_prefixed_with():
    """`measure` prefixes every family with the object it belongs to, so a
    column named `cell_...` here reaches the database as `cell_cell_...`."""
    mask = _two_squares()
    image = np.zeros((40, 40), dtype=float)
    image[12, 12] = 100.0
    frame = od.object_distances({"cell": mask}, image, primary="cell")
    assert not [c for c in frame.columns if c.startswith("cell_")]
