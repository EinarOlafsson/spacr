"""The z and t plumbing refuses a geometry it cannot honour.

Every number in this module is a fact about the sample: which axis is z, how
much thicker a z step is than a pixel, how far an object may have moved
between two timepoints. A wrong one does not fail -- it produces a volume, a
distance, a track, and every measurement downstream is silently in the wrong
units. So each of them is either given, derivable, or refused by name.

The empty cases matter for the opposite reason: a stack with no objects, a
plane pair with nothing in common, a first timepoint that detected nothing.
Those are ordinary answers, and they have to come back as zero and ``{}``
rather than as a division by a count of nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import zstack


# -- the spec ---------------------------------------------------------------

def test_a_non_positive_anisotropy_is_refused():
    """``dz/dxy`` must be positive; zero or negative is not a ratio.

    A negative ratio would flip the z axis of every resampled volume, which
    produces a picture that looks fine and is upside down.
    """
    with pytest.raises(zstack.ZStackError, match="anisotropy"):
        zstack.ZStackSpec(anisotropy=-2.0)

    with pytest.raises(zstack.ZStackError, match="anisotropy"):
        zstack.ZStackSpec(anisotropy=0.0)


def test_a_stitch_threshold_outside_zero_to_one_is_refused():
    """The stitch threshold is an IoU, so it lives in [0, 1]."""
    with pytest.raises(zstack.ZStackError, match="stitch_threshold"):
        zstack.ZStackSpec(stitch_threshold=1.5)

    with pytest.raises(zstack.ZStackError, match="stitch_threshold"):
        zstack.ZStackSpec(stitch_threshold=-0.1)


# -- the result -------------------------------------------------------------

def test_an_empty_result_counts_no_objects_and_no_truncation():
    """A stack with no labels reports zero rather than dividing by zero."""
    empty = zstack.ZStackResult(labels=np.empty((0, 0, 0), dtype=np.int64),
                                mode=zstack.MODE_STITCH)

    assert empty.n_objects == 0
    assert empty.truncated_fraction == 0.0


def test_a_result_with_no_objects_in_a_real_volume_is_also_zero():
    """An all-background volume has objects to divide by: none."""
    blank = zstack.ZStackResult(labels=np.zeros((3, 8, 8), dtype=np.int64),
                                mode=zstack.MODE_STITCH)

    assert blank.n_objects == 0
    assert blank.truncated_fraction == 0.0


# -- axis handling ----------------------------------------------------------

def test_a_two_dimensional_array_has_no_z_axis_to_move():
    """A plane is not a volume, and is refused before anything is moved."""
    with pytest.raises(ValueError, match="at least 3 axes"):
        zstack._as_z_first(np.zeros((8, 8)), None)


def test_a_four_dimensional_array_must_be_told_which_axis_is_z():
    """Detection is only defined for a plain 3-D volume.

    Guessing on a ``(Z, Y, X, C)`` array picks the channel axis about as often
    as it picks z, and a transposed volume segments into nonsense.
    """
    with pytest.raises(zstack.AmbiguousZAxisError, match="z_axis must be given"):
        zstack._as_z_first(np.zeros((4, 64, 64, 3)), None)


def test_a_plain_volume_has_its_z_axis_detected():
    """A ``(Z, Y, X)`` volume needs no ``z_axis`` argument."""
    moved, axis = zstack._as_z_first(np.zeros((5, 64, 64)), None)

    assert axis == 0
    assert moved.shape == (5, 64, 64)


def test_an_ambiguous_volume_is_refused_rather_than_guessed():
    """A cube gives detection nothing to go on and raises in strict mode."""
    with pytest.raises(zstack.AmbiguousZAxisError):
        zstack._as_z_first(np.zeros((64, 64, 64)), None)


# -- anisotropy -------------------------------------------------------------

def test_a_voxel_size_with_a_zero_side_is_refused():
    """A voxel with a zero or infinite side is not a measurement."""
    with pytest.raises(zstack.ZStackError, match="voxel_size_um"):
        zstack.resolve_anisotropy(voxel_size_um=(0.0, 0.65, 0.65))

    with pytest.raises(zstack.ZStackError, match="voxel_size_um"):
        zstack.resolve_anisotropy(voxel_size_um=(2.0, float("nan"), 0.65))


def test_a_good_voxel_size_gives_the_ratio():
    """The same call does return dz over the mean xy pitch."""
    assert zstack.resolve_anisotropy(voxel_size_um=(2.0, 0.5, 0.5)) == 4.0


# -- projection -------------------------------------------------------------

def test_best_focus_on_a_single_plane_returns_that_plane():
    """One plane is trivially the sharpest, with no focus scoring at all."""
    only = np.arange(64, dtype=np.float32).reshape(1, 8, 8)

    projected = zstack.project(only, mode="best_focus", z_axis=0)

    assert projected.shape == (8, 8)
    assert np.array_equal(projected, only[0])


# -- stitching --------------------------------------------------------------

def test_stitching_accepts_a_list_of_planes():
    """A sequence of 2-D masks stitches like a stacked array does.

    Cellpose hands back a list, and materialising it into an array first would
    double the peak memory of the largest step in the run.
    """
    plane = np.zeros((8, 8), dtype=np.int32)
    plane[2:6, 2:6] = 1

    stitched = zstack.stitch_planes([plane, plane.copy()], iou_threshold=0.25)

    assert stitched.shape == (2, 8, 8)
    assert set(np.unique(stitched)) == {0, 1}


def test_stitching_an_empty_stack_is_refused():
    """No planes is not a volume of zero objects, it is a caller error."""
    with pytest.raises(ValueError, match="empty stack"):
        zstack.stitch_planes([])


# -- surface area -----------------------------------------------------------

def test_a_volume_with_no_objects_exposes_no_faces():
    """An all-background volume has no label boundaries along any axis."""
    faces = zstack._surface_faces_per_label(np.zeros((4, 4, 4), np.int64), 0)

    assert faces.shape == (3, 1)
    assert not faces.any()


# -- 4-D axis order ---------------------------------------------------------

def test_explicit_time_and_z_axes_need_a_four_axis_array():
    """Naming both axes on a 3-D array cannot be honoured."""
    with pytest.raises(zstack.TStackError, match="4-D"):
        zstack.resolve_axis_order(np.zeros((5, 64, 64)), t_axis=0, z_axis=1)


def test_a_named_axis_outside_the_two_leading_axes_cannot_infer_the_other():
    """``t_axis=2`` on ``(T, Z, Y, X)`` leaves nothing to infer z from."""
    with pytest.raises(zstack.TStackError, match="cannot be inferred"):
        zstack.resolve_axis_order(np.zeros((4, 5, 64, 64)), t_axis=2)


def test_a_channel_axis_adds_an_axis_to_what_the_spec_needs(tmp_path):
    """A spec naming a channel axis needs one more axis than without it.

    An array short of that is refused by name rather than being reshaped into
    something with the channels folded into z.
    """
    spec = zstack.TStackSpec(t_axis=0, z_axis=1, channel_axis=4)

    with pytest.raises(zstack.TAxisNotPresentError, match="channel_axis=4"):
        zstack.as_t_first(np.zeros((3, 4, 8, 8)), spec)


def test_a_flat_time_series_must_put_time_on_axis_zero():
    """``t_axis_order='TYX'`` names axis 0 as time; anything else contradicts it."""
    with pytest.raises(zstack.TStackError, match="t_axis=2"):
        zstack.plan_4d_from_settings(
            {"t_stack": True, "t_axis_order": "TYX", "t_axis": 2})


def test_a_flat_time_series_with_time_on_axis_zero_is_accepted():
    """The consistent spelling of the same settings does produce a spec."""
    spec = zstack.plan_4d_from_settings(
        {"t_stack": True, "t_axis_order": "TYX", "t_axis": 0})

    assert spec is not None
    assert spec.t_axis == 0
    assert spec.z_axis is None


# -- tracking ---------------------------------------------------------------

def test_a_timepoint_with_no_objects_matches_nothing():
    """An empty frame links to nothing rather than raising in the solver."""
    scale = zstack._displacement_scale(2, 1.0, None, False)
    populated = np.zeros((8, 8), dtype=np.int64)
    populated[2:5, 2:5] = 1

    assert zstack._centroid_matches(np.zeros((8, 8), np.int64), populated,
                                    scale, 5.0) == {}
    assert zstack._centroid_matches(populated, np.zeros((8, 8), np.int64),
                                    scale, 5.0) == {}


def test_trackpy_links_a_flat_time_series_with_a_defaulted_spec():
    """A ``(T, Y, X)`` movie tracks with no spec and no z coordinate.

    trackpy is asked for a 3-D link whatever the input, so a flat series has
    to supply a constant z rather than a DataFrame with one column missing --
    trackpy's own column guess would then silently link in 2-D on some frames
    and 3-D on others.
    """
    pytest.importorskip("trackpy")

    labels = np.zeros((3, 16, 16), dtype=np.int64)
    for t in range(3):
        labels[t, 4 + t:7 + t, 4:7] = 1

    result = zstack.track_4d(labels, backend="trackpy",
                             max_displacement_px=6.0)

    assert result.labels.shape == labels.shape
    assert set(np.unique(result.labels)) == {0, 1}
    assert result.backend == "trackpy"
    assert result.n_tracks == 1
