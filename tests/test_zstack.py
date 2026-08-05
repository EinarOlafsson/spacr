"""The 3D (Beta) z axis: :mod:`spacr.zstack` and its wiring into ``spacr.object``.

Everything here is CPU-only, offline and model-free. The z logic is exercised
against synthetic label volumes and against deterministic segmenters written
in the test, and the two tests that go through
``object.generate_cellpose_masks_sam`` monkeypatch ``cellpose.models`` with the
same fake used by ``test_cov_object_masks_sam.py``. No real network is loaded
and nothing is downloaded.

The load-bearing tests, in the order they matter:

1. ``test_2d_is_untouched_*`` -- the acceptance criterion. A user who does not
   opt in must not see any change at all, so the masks, the eval kwargs and
   the database rows are compared byte for byte between a run whose settings
   have never heard of z and a run whose z settings are present but off.
2. ``test_anisotropy_decides_whether_objects_in_z_merge`` -- two objects
   separated by three planes fuse at anisotropy 1.0 and stay apart at the true
   5.0. This is the whole reason anisotropy is a required input.
3. ``test_z_stack_on_but_no_z_axis_is_a_loud_error`` -- opting in when the
   ingest has already projected z away must stop the run, not silently
   segment the projection and label it 3-D.
"""
from __future__ import annotations

import sqlite3
import types
from pathlib import Path

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

import spacr.object as O
import spacr.settings as S
import spacr.zstack as Z

from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call


# ===========================================================================
# Helpers
# ===========================================================================

def _ball(radius: int, ndim: int = 3) -> np.ndarray:
    """Spherical structuring element of the given radius, in voxels."""
    r = int(radius)
    axes = np.ogrid[tuple(slice(-r, r + 1) for _ in range(ndim))]
    return sum(a ** 2 for a in axes) <= r ** 2


def _proximity_segmenter(radius: int = 2, threshold: float = 0.5):
    """A deterministic segmenter that fuses objects closer than ``radius``.

    Dilates the foreground by a *spherical* element, labels the result, then
    restricts the labels back to the original foreground. Because the element
    is spherical it measures distance in whatever voxel grid it is handed --
    which is exactly why an un-corrected anisotropy makes it merge objects
    that are physically far apart in z, and exactly what a real 3-D segmenter
    does wrong for the same reason.
    """
    from scipy.ndimage import binary_dilation, label as ndi_label

    def _fn(array, do_3D=False, anisotropy=None, z_axis=None, stitch=False):
        binary = np.asarray(array) >= threshold
        grown = binary_dilation(binary, structure=_ball(radius, binary.ndim))
        labels, _ = ndi_label(grown)
        return np.where(binary, labels, 0).astype(np.int32)

    return _fn


def _two_blobs_separated_in_z(n_z=9, size=20, gap_start=3, gap_stop=6):
    """A volume with two xy-identical blobs separated by three empty planes.

    Planes 1-2 hold one blob, planes 6-7 the other, and 3-5 are empty. In
    voxels the gap is 3; at anisotropy 5 it is really 15.
    """
    volume = np.zeros((n_z, size, size), dtype=np.float32)
    volume[1:gap_start, 4:16, 4:16] = 1.0
    volume[gap_stop:n_z - 1, 4:16, 4:16] = 1.0
    return volume


def _square(plane_shape, rows, cols, value):
    out = np.zeros(plane_shape, dtype=np.int32)
    out[rows[0]:rows[1], cols[0]:cols[1]] = value
    return out


# ===========================================================================
# 1. Axis detection
# ===========================================================================

@pytest.mark.parametrize("shape,expected", [
    ((21, 512, 512), 0),      # (Z, Y, X) -- the canonical microscope layout
    ((512, 512, 21), 2),      # (Y, X, Z) -- what a tiff reader often hands you
    ((512, 21, 512), 1),      # z in the middle, still unambiguous
    ((100, 512, 512), 0),     # no axis is short, but two are equal
    ((512, 512, 100), 2),
])
def test_detect_z_axis_finds_the_short_axis(shape, expected):
    assert Z.detect_z_axis(np.zeros(shape, dtype=np.uint8)) == expected
    # A shape tuple is accepted too, so callers need not allocate.
    assert Z.detect_z_axis(shape) == expected


@pytest.mark.parametrize("shape", [
    (64, 64, 64),      # a cube: nothing distinguishes any axis
    (10, 20, 512),     # two short axes: which one is z?
    (8, 16, 24),       # three different, none equal, all short
])
def test_ambiguous_shapes_are_reported_not_guessed(shape):
    """None means 'I do not know', which the caller must handle."""
    assert Z.detect_z_axis(shape) is None


def test_strict_detection_explains_the_ambiguity(shape=(64, 64, 64)):
    with pytest.raises(Z.AmbiguousZAxisError) as excinfo:
        Z.detect_z_axis(shape, strict=True)
    message = str(excinfo.value)
    assert "64" in message
    assert "z_axis" in message, "the message must name the setting that fixes it"


def test_detect_z_axis_rejects_non_3d_input():
    with pytest.raises(ValueError, match="3-D"):
        Z.detect_z_axis(np.zeros((4, 8, 8, 2)))


# ===========================================================================
# 2. Anisotropy -- the load-bearing behaviour
# ===========================================================================

def test_anisotropy_decides_whether_objects_in_z_merge():
    """Two objects three planes apart: fused at 1.0, separate at the true 5.0.

    This is the single most important behaviour in the module. A confocal z
    step is routinely 3-10x the xy pixel size, so a volume segmented as if it
    were isotropic reads a 15 um gap as a 3 pixel gap and welds the two
    objects into one column. Nothing downstream can detect that this happened.
    """
    volume = _two_blobs_separated_in_z()
    segment_fn = _proximity_segmenter(radius=2)

    fused = Z.segment_3d(
        volume, segment_fn=segment_fn, mode=Z.MODE_VOLUMETRIC,
        anisotropy=1.0, resample_to_isotropic=True,
    )
    separate = Z.segment_3d(
        volume, segment_fn=segment_fn, mode=Z.MODE_VOLUMETRIC,
        anisotropy=5.0, resample_to_isotropic=True,
    )

    assert fused.n_objects == 1, (
        "at anisotropy 1.0 the 3-plane gap is only 3 voxels wide, so a "
        "segmenter with a 2-voxel reach bridges it and reports one object"
    )
    assert separate.n_objects == 2, (
        "at the true anisotropy 5.0 the same gap is 15 isotropic voxels and "
        "the two objects must stay apart"
    )
    assert separate.anisotropy == 5.0
    assert separate.mode == Z.MODE_VOLUMETRIC


def test_unknown_anisotropy_is_reported_not_assumed():
    """Volumetric segmentation with no idea of the voxel ratio must stop."""
    volume = _two_blobs_separated_in_z()

    with pytest.raises(Z.UnknownAnisotropyError) as excinfo:
        Z.segment_3d(volume, segment_fn=_proximity_segmenter(),
                     mode=Z.MODE_VOLUMETRIC)

    message = str(excinfo.value)
    assert "1.0" in message, "must say what it refuses to assume"
    assert "voxel_size_z_um" in message, "must name the way out"


def test_anisotropy_is_derived_from_the_voxel_size():
    # dz / mean(dy, dx)
    assert Z.resolve_anisotropy(voxel_size_um=(5.0, 1.0, 1.0)) == 5.0
    assert Z.resolve_anisotropy(voxel_size_um=(1.0, 0.25, 0.25)) == 4.0
    # An explicit value wins over the derived one.
    assert Z.resolve_anisotropy(anisotropy=3.0, voxel_size_um=(5.0, 1.0, 1.0)) == 3.0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_nonsense_anisotropy_is_rejected(bad):
    with pytest.raises(Z.ZStackError):
        Z.resolve_anisotropy(anisotropy=bad)


def test_stitch_mode_records_that_it_ignores_anisotropy():
    """Cellpose ignores anisotropy without do_3D, silently. spaCR says so."""
    volume = _two_blobs_separated_in_z()
    result = Z.segment_3d(
        volume, segment_fn=lambda a, **kw: _proximity_segmenter()(a),
        mode=Z.MODE_STITCH, anisotropy=5.0,
    )
    assert any("ignored in stitch mode" in note for note in result.notes)
    assert result.anisotropy is None, (
        "recording an anisotropy on a result that never used one would make "
        "the number look meaningful"
    )


def test_resample_isotropic_stretches_z_and_restores_it():
    volume = np.zeros((6, 12, 12), dtype=np.float32)
    volume[2] = 1.0

    stretched = Z.resample_isotropic(volume, anisotropy=4.0)
    assert stretched.shape == (24, 12, 12)

    back = Z.restore_anisotropic(stretched, n_z=6)
    assert back.shape == volume.shape
    # anisotropy 1.0 is a no-op, not a resize through the interpolator.
    assert Z.resample_isotropic(volume, anisotropy=1.0) is not None
    assert np.array_equal(Z.resample_isotropic(volume, anisotropy=1.0), volume)


# ===========================================================================
# 3. Stitching
# ===========================================================================

def test_stitching_links_the_same_object_across_planes():
    """Plane-local label ids are meaningless; the linking must see through them."""
    shape = (24, 24)
    stack = np.stack([
        _square(shape, (4, 16), (4, 16), 1),
        _square(shape, (4, 16), (4, 16), 7),    # same object, different local id
        _square(shape, (4, 16), (4, 16), 3),
    ])

    linked = Z.stitch_planes(stack, iou_threshold=0.25)

    ids = np.unique(linked)
    assert list(ids[ids > 0]) == [1], "one object spanning three planes"
    assert (linked[0] == linked[1]).all()
    assert (linked[1] == linked[2]).all()


def test_stitching_does_not_fuse_two_objects_that_merely_overlap_in_xy():
    """Matching is one-to-one: only the better match inherits the label.

    Two distinct objects in the upper plane both overlap the same object
    below. Linking both would weld them into one 3-D object; instead the
    higher-IoU one continues the chain and the other starts its own.
    """
    shape = (24, 24)
    lower = _square(shape, (2, 12), (2, 12), 1)          # 100 px

    upper = np.zeros(shape, dtype=np.int32)
    upper[2:7, 2:12] = 1                                  # 50 px -> IoU 0.50
    upper[8:12, 2:12] = 2                                 # 40 px -> IoU 0.40

    linked = Z.stitch_planes(np.stack([lower, upper]), iou_threshold=0.25)

    top_ids = np.unique(linked[1])
    top_ids = top_ids[top_ids > 0]
    assert len(top_ids) == 2, "the two upper objects must stay distinct"

    bottom_id = int(np.unique(linked[0])[1:][0])
    assert bottom_id in top_ids, "the better match continues the object"
    assert len({int(i) for i in top_ids}) == 2
    # Three 2-D regions, two 3-D objects.
    all_ids = np.unique(linked)
    assert len(all_ids[all_ids > 0]) == 2


def test_stitching_leaves_non_overlapping_objects_apart():
    shape = (24, 24)
    stack = np.stack([
        _square(shape, (2, 6), (2, 6), 1),
        _square(shape, (16, 20), (16, 20), 1),   # same local id, elsewhere
    ])
    linked = Z.stitch_planes(stack, iou_threshold=0.25)
    ids = np.unique(linked)
    assert len(ids[ids > 0]) == 2


def test_stitching_below_the_threshold_does_not_link():
    shape = (24, 24)
    lower = _square(shape, (2, 12), (2, 12), 1)
    upper = _square(shape, (10, 20), (2, 12), 1)   # small overlap, low IoU
    loose = Z.stitch_planes(np.stack([lower, upper]), iou_threshold=0.05)
    strict = Z.stitch_planes(np.stack([lower, upper]), iou_threshold=0.9)

    assert len(np.unique(loose)) - 1 == 1, "a low threshold links them"
    assert len(np.unique(strict)) - 1 == 2, "a high threshold does not"


def test_an_empty_plane_does_not_cause_a_label_collision():
    """The bug in ``cellpose.utils.stitch3D``, which spaCR must not inherit.

    Cellpose resets its label counter after an empty plane, so the object
    above and the object below both end up as label 1 and are fused into one
    3-D object even though they never touch.
    """
    shape = (24, 24)
    stack = np.stack([
        _square(shape, (2, 6), (2, 6), 1),
        np.zeros(shape, dtype=np.int32),          # nothing segmented here
        _square(shape, (16, 20), (16, 20), 1),    # same local id again
    ])

    linked = Z.stitch_planes(stack, iou_threshold=0.25)
    ids = np.unique(linked)
    assert len(ids[ids > 0]) == 2, "two unrelated objects, not one"
    assert set(np.unique(linked[0])) != set(np.unique(linked[2]))


def test_stitch_threshold_must_be_an_iou():
    with pytest.raises(Z.ZStackError, match=r"\[0, 1\]"):
        Z.stitch_planes(np.zeros((2, 4, 4), dtype=np.int32), iou_threshold=1.5)


# ===========================================================================
# 4. Truncation at the ends of the stack
# ===========================================================================

def test_objects_touching_the_first_or_last_plane_are_flagged():
    """The z equivalent of seg_qc's xy border rule."""
    labels = np.zeros((6, 20, 20), dtype=np.int32)
    labels[0:3, 2:8, 2:8] = 1        # runs off the bottom of the stack
    labels[2:4, 12:18, 12:18] = 2    # wholly inside
    labels[4:6, 2:8, 12:18] = 3      # runs off the top

    truncated = Z.flag_truncated_z(labels)
    assert sorted(truncated.tolist()) == [1, 3]
    assert 2 not in truncated


def test_a_2d_mask_has_no_z_truncation():
    assert Z.flag_truncated_z(np.ones((10, 10), dtype=np.int32)).size == 0


def test_truncation_is_reported_on_the_result():
    volume = np.zeros((6, 20, 20), dtype=np.float32)
    volume[0:2, 4:12, 4:12] = 1.0    # touches plane 0

    result = Z.segment_3d(
        volume, segment_fn=_proximity_segmenter(radius=1),
        mode=Z.MODE_VOLUMETRIC, anisotropy=1.0,
    )
    assert result.truncated_labels.size == 1
    assert result.truncated_fraction == 1.0
    assert any("truncated" in note for note in result.notes)


# ===========================================================================
# 5. A single plane is 2-D, not degenerate 3-D
# ===========================================================================

def test_a_single_plane_volume_matches_the_2d_path_exactly():
    """n_z == 1 must be the ordinary 2-D path: 2-D in, 2-D out, no z code."""
    plane = np.zeros((20, 20), dtype=np.float32)
    plane[4:12, 4:12] = 1.0
    volume = plane[np.newaxis, ...]

    segment_fn = _proximity_segmenter(radius=1)
    seen = []

    def _recording_fn(array, **kwargs):
        seen.append((np.asarray(array).shape, dict(kwargs)))
        return segment_fn(array)

    result = Z.segment_3d(volume, segment_fn=_recording_fn,
                          mode=Z.MODE_VOLUMETRIC, anisotropy=None)

    assert result.mode == Z.MODE_SINGLE_PLANE
    assert result.labels.ndim == 2, "a 2-D image must give a 2-D mask"
    assert result.n_z == 1
    assert result.anisotropy is None
    # The segmenter saw the plain 2-D plane and no 3-D kwargs at all.
    assert len(seen) == 1
    assert seen[0][0] == (20, 20)
    assert seen[0][1] == {}
    # Missing anisotropy is not even consulted -- a 1-plane stack has no z.
    assert np.array_equal(result.labels, segment_fn(plane))


# ===========================================================================
# 6. Projection
# ===========================================================================

@pytest.mark.parametrize("mode,expected", [
    ("max", 9.0),
    ("mean", 3.0),
    ("sum", 12.0),
])
def test_projection_reducers(mode, expected):
    volume = np.zeros((4, 6, 6), dtype=np.float32)
    volume[0] = 1.0
    volume[1] = 2.0
    volume[2] = 9.0
    volume[3] = 0.0
    assert project_value(volume, mode) == expected


def project_value(volume, mode):
    out = Z.project(volume, mode=mode)
    assert out.shape == (6, 6)
    return float(out[0, 0])


def test_best_focus_keeps_the_sharpest_plane_not_a_blend():
    volume = np.zeros((3, 20, 20), dtype=np.float32)
    volume[0] = 0.5                       # flat: no detail at all
    volume[1, ::2, ::2] = 1.0             # high-frequency detail: sharpest
    volume[2, 5:15, 5:15] = 1.0           # one soft block

    out = Z.project(volume, mode="best_focus")
    assert np.array_equal(out, volume[1])


def test_projection_none_returns_the_volume():
    volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    assert np.array_equal(Z.project(volume, mode=None), volume)


def test_unknown_projection_is_rejected():
    with pytest.raises(Z.ZStackError, match="z_projection"):
        Z.project(np.zeros((2, 4, 4)), mode="median")


# ===========================================================================
# 7. Relabelling
# ===========================================================================

def test_relabelling_is_contiguous_with_no_gaps():
    labels = np.zeros((3, 6, 6), dtype=np.int32)
    labels[0, 0:2, 0:2] = 5
    labels[1, 3:5, 3:5] = 9
    labels[2, 0:2, 3:5] = 100

    out = Z.relabel_volume(labels)
    ids = np.unique(out)
    assert list(ids) == [0, 1, 2, 3], "1..N with no holes, background still 0"
    assert out.shape == labels.shape
    # The same voxels are still labelled; only the numbers changed.
    assert np.array_equal(out > 0, labels > 0)


def test_relabelling_an_empty_volume_is_empty():
    out = Z.relabel_volume(np.zeros((2, 4, 4), dtype=np.int32))
    assert not out.any()


# ===========================================================================
# 8. Volume statistics
# ===========================================================================

def test_volume_stats_with_anisotropic_voxels_differ_from_isotropic():
    """Anisotropy is not a constant scale factor on the derived quantities."""
    labels = np.zeros((6, 20, 20), dtype=np.int32)
    labels[2:5, 4:10, 4:10] = 1          # 3 planes x 6 x 6 = 108 voxels

    isotropic = Z.volume_stats(labels, voxel_size=(1.0, 1.0, 1.0))
    anisotropic = Z.volume_stats(labels, voxel_size=(5.0, 1.0, 1.0))

    row_i = isotropic.iloc[0]
    row_a = anisotropic.iloc[0]

    # The voxel count is a property of the mask and does not move.
    assert row_i["volume_voxels"] == row_a["volume_voxels"] == 108
    assert row_i["volume_um3"] == 108.0
    assert row_a["volume_um3"] == 540.0

    # Surface area is NOT simply 5x: the z-facing faces have area dy*dx and
    # are unaffected, while the x- and y-facing faces are 5x taller. This is
    # exactly why an unspaced measurement of an anisotropic volume is
    # meaningless rather than merely mis-scaled.
    assert row_a["surface_um2"] != pytest.approx(row_i["surface_um2"] * 5.0)
    assert row_a["surface_um2"] > row_i["surface_um2"]

    assert row_i["z_extent_um"] == 3.0
    assert row_a["z_extent_um"] == 15.0
    assert row_a["z_extent_planes"] == 3


def test_volume_stats_states_its_units():
    """A voxel count in a column named like a px^2 area is how a screen
    silently gets corrupted, so every column carries its unit."""
    labels = np.zeros((4, 10, 10), dtype=np.int32)
    labels[1:3, 2:6, 2:6] = 1

    table = Z.volume_stats(labels, voxel_size=(2.0, 0.5, 0.5))
    for column in table.columns:
        assert column in Z.VOLUME_STATS_UNITS, f"{column} has no declared unit"

    assert Z.VOLUME_STATS_UNITS["volume_voxels"] == "voxels"
    assert Z.VOLUME_STATS_UNITS["volume_um3"] == "um^3"
    assert Z.VOLUME_STATS_UNITS["z_extent_um"] == "um"


def test_volume_stats_without_a_voxel_size_omits_the_physical_columns():
    labels = np.zeros((4, 10, 10), dtype=np.int32)
    labels[1:3, 2:6, 2:6] = 1

    table = Z.volume_stats(labels)
    assert "volume_voxels" in table.columns
    assert "volume_um3" not in table.columns
    assert "z_extent_um" not in table.columns


def test_volume_stats_marks_truncated_objects_and_measures_z_extent():
    labels = np.zeros((6, 20, 20), dtype=np.int32)
    labels[0:2, 2:8, 2:8] = 1        # truncated at the bottom
    labels[2:5, 12:18, 12:18] = 2    # interior

    table = Z.volume_stats(labels, voxel_size=(1.0, 1.0, 1.0)).set_index("label")
    assert bool(table.loc[1, "truncated_z"]) is True
    assert bool(table.loc[2, "truncated_z"]) is False
    assert table.loc[1, "z_min"] == 0
    assert table.loc[2, "z_extent_planes"] == 3


def test_volume_stats_needs_a_3d_volume():
    with pytest.raises(ValueError, match="3-D"):
        Z.volume_stats(np.zeros((10, 10), dtype=np.int32))


def test_volume_stats_of_an_empty_volume_is_an_empty_table():
    table = Z.volume_stats(np.zeros((3, 8, 8), dtype=np.int32))
    assert len(table) == 0
    assert "volume_voxels" in table.columns


# ===========================================================================
# 9. Modes are explicit and recorded
# ===========================================================================

def test_project_mode_collapses_z_and_says_so():
    volume = _two_blobs_separated_in_z()
    result = Z.segment_3d(volume, segment_fn=_proximity_segmenter(radius=1),
                          mode=Z.MODE_PROJECT, projection="max")

    assert result.mode == Z.MODE_PROJECT
    assert result.labels.ndim == 2, "projecting gives a 2-D mask"
    assert result.n_z == 9
    assert any("projection" in note for note in result.notes)


def test_stitch_and_volumetric_give_different_answers_and_both_say_which():
    """They are different measurements of the same sample, not variants."""
    volume = _two_blobs_separated_in_z()

    stitched = Z.segment_3d(
        volume, segment_fn=lambda a, **kw: _proximity_segmenter(radius=2)(a),
        mode=Z.MODE_STITCH, stitch_threshold=0.25,
    )
    volumetric = Z.segment_3d(
        volume, segment_fn=_proximity_segmenter(radius=2),
        mode=Z.MODE_VOLUMETRIC, anisotropy=1.0, resample_to_isotropic=True,
    )

    assert stitched.mode == Z.MODE_STITCH
    assert volumetric.mode == Z.MODE_VOLUMETRIC
    # Per-plane segmentation cannot bridge the three empty planes; volumetric
    # segmentation at anisotropy 1.0 does. Same data, different answers.
    assert stitched.n_objects == 2
    assert volumetric.n_objects == 1
    assert any("not volumetric" in note for note in stitched.notes)


def test_an_unknown_mode_is_rejected():
    with pytest.raises(Z.ZStackError, match="z_segmentation_mode"):
        Z.segment_3d(np.zeros((3, 8, 8)), segment_fn=lambda a, **kw: a,
                     mode="3d-ish")


def test_estimate_peak_bytes_grows_with_the_mode_and_anisotropy():
    shape = (21, 512, 512)
    project = Z.estimate_peak_bytes(shape, np.float32, Z.MODE_PROJECT)
    stitch = Z.estimate_peak_bytes(shape, np.float32, Z.MODE_STITCH)
    volumetric = Z.estimate_peak_bytes(shape, np.float32, Z.MODE_VOLUMETRIC,
                                       anisotropy=5.0)
    assert project < stitch < volumetric
    # One field, not a plate: even the worst case stays in single-digit GB.
    assert volumetric < 16 * 1024 ** 3


# ===========================================================================
# 10. The settings bridge
# ===========================================================================

def test_plan_is_none_when_the_3d_settings_are_absent_or_off():
    """The contract that keeps the 2-D path untouched."""
    assert Z.plan_from_settings({}) is None
    assert Z.plan_from_settings({"z_stack": False}) is None
    # Present but off, with every other z setting deliberately non-default.
    assert Z.plan_from_settings({
        "z_stack": False,
        "z_segmentation_mode": "volumetric",
        "anisotropy": 5.0,
        "z_axis": 2,
        "stitch_threshold": 0.9,
    }) is None


def test_plan_is_built_when_3d_is_on():
    spec = Z.plan_from_settings({
        "z_stack": True,
        "z_segmentation_mode": "stitch",
        "z_axis": 0,
        "z_projection": "max",
        "voxel_size_z_um": 4.0,
        "voxel_size_xy_um": 0.5,
        "stitch_threshold": 0.4,
    })
    assert spec is not None
    assert spec.mode == Z.MODE_STITCH
    assert spec.stitch_threshold == 0.4
    assert spec.voxel_size_um == (4.0, 0.5, 0.5)
    assert spec.require_anisotropy() == 8.0


def test_a_volumetric_plan_without_anisotropy_fails_at_setup_time():
    """Before the model is loaded and the first field read, not after."""
    with pytest.raises(Z.UnknownAnisotropyError):
        Z.plan_from_settings({"z_stack": True,
                              "z_segmentation_mode": "volumetric"})


def test_an_unknown_mode_in_the_settings_is_rejected():
    with pytest.raises(Z.ZStackError, match="z_segmentation_mode"):
        Z.plan_from_settings({"z_stack": True, "z_segmentation_mode": "nope"})


def test_the_mask_defaults_ship_the_3d_settings_off():
    defaults = S.set_default_settings_preprocess_generate_masks({})
    assert defaults["z_stack"] is False
    assert defaults["z_segmentation_mode"] == "project"
    assert defaults["z_axis"] is None
    assert defaults["anisotropy"] is None
    assert defaults["voxel_size_z_um"] is None
    assert defaults["voxel_size_xy_um"] is None
    assert defaults["z_projection"] == "max"
    assert defaults["stitch_threshold"] == 0.25
    assert Z.plan_from_settings(defaults) is None, (
        "the shipped defaults must not switch any z code on"
    )


NEW_3D_KEYS = [
    "z_stack", "z_segmentation_mode", "z_axis", "z_projection",
    "anisotropy", "voxel_size_z_um", "voxel_size_xy_um", "stitch_threshold",
]


@pytest.mark.parametrize("key", NEW_3D_KEYS)
def test_every_new_setting_is_categorised_typed_and_documented(key):
    """The invariants tests/test_settings_categories.py pins, per key."""
    categorised = [k for keys in S.categories.values() for k in keys]
    assert categorised.count(key) == 1, "exactly one category"
    assert key in S.categories["3D Settings (Beta)"]
    assert key in S.expected_types
    assert key in S.tooltips

    tooltip = S.tooltips[key]
    assert tooltip.startswith("("), "house style: leading (type)"
    assert "Default" in tooltip
    assert len(tooltip) > 200, (
        "a tooltip must say what the setting does and what changes when you "
        "alter it, not restate its name"
    )
    assert key.replace("_", " ") not in tooltip.lower()[:40]


def test_the_3d_tooltips_are_honest_about_the_ingest_limitation():
    """A user must not read these and believe spaCR measures volumes today."""
    assert "collapses" in S.tooltips["z_stack"]
    assert "Measure" in S.tooltips["z_segmentation_mode"]


# ===========================================================================
# 11. Wiring into spacr.object
# ===========================================================================

@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


@pytest.fixture
def fake_model(monkeypatch):
    """A deterministic stand-in for ``cellpose.models.CellposeModel``.

    Handles both call shapes spaCR uses: a list of 2-D images (the ordinary
    path and the per-plane stitch path) and a single (Z, Y, X, C) volume
    (the volumetric path).
    """
    holder = {"model": None}

    class _M:
        def __init__(self, gpu=None, pretrained_model=None, device=None, **kw):
            self.gpu = gpu
            self.pretrained_model = pretrained_model
            self.device = device
            self.eval_kwargs = []
            self.eval_shapes = []
            holder["model"] = self

        @staticmethod
        def _label(image):
            out = np.zeros(image.shape[:2], dtype=np.uint16)
            out[2:8, 2:8] = 1
            out[12:18, 12:18] = 2
            return out

        def eval(self, x=None, channel_axis=MISSING_CHANNEL_AXIS, **kwargs):
            # channel_axis is named, not swallowed by **kwargs: a mock that
            # accepts any axis cannot tell the volumetric call (-1 on a
            # (Z, Y, X, C) volume) from an axis Cellpose would reject.
            check_cellpose_eval_call(
                x, channel_axis,
                z_axis=kwargs.get("z_axis"),
                do_3D=kwargs.get("do_3D", False),
                stitch_threshold=kwargs.get("stitch_threshold", 0.0))
            # Recorded back into the kwargs dict so the "the 3D settings must
            # not leak into the 2-D call" comparison still sees every argument.
            self.eval_kwargs.append({"channel_axis": channel_axis, **kwargs})
            if isinstance(x, list):
                self.eval_shapes.append([np.asarray(i).shape for i in x])
                masks = [self._label(np.asarray(i)) for i in x]
                flows = [np.zeros(m.shape, np.float32) for m in masks]
                return masks, flows, None, None
            volume = np.asarray(x)
            self.eval_shapes.append(volume.shape)
            labels = np.stack([self._label(volume[z])
                               for z in range(volume.shape[0])])
            return labels, [np.zeros(labels.shape, np.float32)], None, None

    monkeypatch.setattr(O, "cp_models", types.SimpleNamespace(CellposeModel=_M))
    return holder


def _write_npz(src: Path, shape, name="batch1.npz", seed=0):
    """Write one pre-batched npz. ``shape`` is (N, Y, X, C) or (N, Z, Y, X, C)."""
    src.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 4000, size=shape).astype(np.uint16)
    filenames = np.array([f"plate1_A01_{i + 1}.npy" for i in range(shape[0])])
    np.savez(src / name, data=data, filenames=filenames)
    return data


def _base_settings(src, **over):
    settings = {
        "src": str(src),
        "cell_channel": 0,
        "nucleus_channel": 1,
        "pathogen_channel": None,
        "magnification": 20,
        "batch_size": 50,
        "verbose": False,
        "plot": False,
        "save": True,
        "timelapse": False,
        "n_jobs": 1,
        "seg_qc": "off",
    }
    settings.update(over)
    return settings


def _artifacts(src, object_type="cell"):
    """Everything a run leaves behind, for a byte-for-byte comparison."""
    folder = Path(src) / f"{object_type}_mask_stack"
    masks = {p.name: p.read_bytes() for p in sorted(folder.iterdir())}

    db = Path(src).parent / "measurements" / "measurements.db"
    con = sqlite3.connect(str(db))
    try:
        rows = sorted(con.execute(
            "SELECT file_name, count_type, object_count FROM object_counts"))
    finally:
        con.close()
    return masks, rows


# --- the acceptance criterion ---------------------------------------------

def test_2d_is_untouched_when_the_3d_settings_are_absent_or_present_but_off(
        tmp_path, fake_model):
    """A user who does not opt in must not see any change at all.

    Two runs on identical input: one whose settings have never heard of z,
    one carrying every 3D key explicitly set but with ``z_stack`` off (and the
    rest at deliberately provocative values). The masks on disk, the rows in
    the database and the kwargs handed to Cellpose must all match exactly.
    """
    src_a = tmp_path / "a" / "stack"
    src_b = tmp_path / "b" / "stack"
    _write_npz(src_a, (3, 32, 32, 2), seed=7)
    _write_npz(src_b, (3, 32, 32, 2), seed=7)

    O.generate_cellpose_masks_sam(str(src_a), _base_settings(src_a), "cell")
    kwargs_a = [dict(k) for k in fake_model["model"].eval_kwargs]

    O.generate_cellpose_masks_sam(str(src_b), _base_settings(
        src_b,
        z_stack=False,              # the switch, off
        z_segmentation_mode="volumetric",
        z_axis=2,
        z_projection="best_focus",
        anisotropy=5.0,
        voxel_size_z_um=5.0,
        voxel_size_xy_um=1.0,
        stitch_threshold=0.9,
    ), "cell")
    kwargs_b = [dict(k) for k in fake_model["model"].eval_kwargs]

    masks_a, rows_a = _artifacts(src_a)
    masks_b, rows_b = _artifacts(src_b)

    assert masks_a.keys() == masks_b.keys()
    for name in masks_a:
        assert masks_a[name] == masks_b[name], (
            f"{name} differs: turning the 3D settings on-but-off changed the "
            f"2-D masks"
        )
    assert rows_a == rows_b
    assert kwargs_a == kwargs_b, (
        "Cellpose was called differently; the 3D settings must not leak into "
        "the 2-D eval call"
    )
    # And in particular no 3-D argument was passed at all.
    for kwargs in kwargs_a:
        assert "do_3D" not in kwargs
        assert "anisotropy" not in kwargs
        assert "z_axis" not in kwargs
        assert "stitch_threshold" not in kwargs


def test_z_stack_on_but_no_z_axis_is_a_loud_error(tmp_path, fake_model):
    """Silently segmenting the projection and calling it 3-D is the one
    outcome worse than having no 3-D at all."""
    src = tmp_path / "stack"
    _write_npz(src, (2, 32, 32, 2))

    settings = _base_settings(src, z_stack=True, z_segmentation_mode="project")

    with pytest.raises(Z.ZAxisNotPresentError) as excinfo:
        O.generate_cellpose_masks_sam(str(src), settings, "cell")

    message = str(excinfo.value)
    assert "no z" in message.lower()
    assert "io._rename_and_organize_image_files" in message, (
        "the message must name where z was lost, not just that it is missing"
    )
    assert "z_stack off" in message, "and how to proceed"


def test_an_opted_in_run_with_real_volumes_segments_in_3d(tmp_path, fake_model):
    """The volumetric path reaches Cellpose with do_3D and the anisotropy."""
    src = tmp_path / "stack"
    _write_npz(src, (2, 5, 32, 32, 2))     # (fields, Z, Y, X, C)

    settings = _base_settings(
        src, z_stack=True, z_segmentation_mode="volumetric",
        voxel_size_z_um=5.0, voxel_size_xy_um=1.0,
    )
    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    model = fake_model["model"]
    assert len(model.eval_kwargs) == 2, "one call per field, not per batch"
    for kwargs in model.eval_kwargs:
        assert kwargs["do_3D"] is True
        assert kwargs["anisotropy"] == 5.0, "derived from the voxel size"
        assert kwargs["z_axis"] == 0
        assert kwargs["channel_axis"] == -1
    # Each call saw one whole (Z, Y, X, C) volume.
    assert model.eval_shapes == [(5, 32, 32, 2), (5, 32, 32, 2)]

    # The masks written to disk are 3-D.
    folder = src / "cell_mask_stack"
    written = sorted(folder.iterdir())
    assert len(written) == 2
    for path in written:
        assert np.load(path).shape == (5, 32, 32)


def test_the_stitch_path_asks_cellpose_for_plain_2d_planes(tmp_path, fake_model):
    """spaCR links the planes itself rather than using cellpose's stitch3D,
    which reuses label ids after an empty plane."""
    src = tmp_path / "stack"
    _write_npz(src, (1, 4, 32, 32, 2))

    settings = _base_settings(src, z_stack=True, z_segmentation_mode="stitch",
                              stitch_threshold=0.3)
    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    model = fake_model["model"]
    assert len(model.eval_kwargs) == 1
    kwargs = model.eval_kwargs[0]
    assert "do_3D" not in kwargs, "stitch mode must not ask for 3-D flows"
    assert "stitch_threshold" not in kwargs, (
        "cellpose must not do the stitching; spaCR does"
    )
    assert "anisotropy" not in kwargs, (
        "cellpose ignores anisotropy without do_3D, so passing it would be "
        "a lie about what ran"
    )
    # It was handed the planes one by one, as 2-D images.
    assert model.eval_shapes == [[(32, 32, 2)] * 4]

    mask = np.load(src / "cell_mask_stack" / "plate1_A01_1.npy")
    assert mask.shape == (4, 32, 32)
    # The fake returns the same two blobs on every plane, so linking across z
    # must yield exactly two objects, not two per plane.
    ids = np.unique(mask)
    assert len(ids[ids > 0]) == 2


def test_project_mode_filters_against_the_projection_it_segmented(
        tmp_path, fake_model, capsys):
    """merge/split/filter scores masks against intensities, so in project mode
    it must see the projected plane, not the volume it came from."""
    src = tmp_path / "stack"
    _write_npz(src, (1, 4, 32, 32, 2))

    settings = _base_settings(
        src, z_stack=True, z_segmentation_mode="project", z_projection="max",
        # force merge/split/filter to actually run
        cell_min_object_area=1,
    )
    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    # Reaching here at all is the assertion: handing merge_split_filter_masks
    # the raw (N, Z, Y, X, C) volume raises "Unsupported intensity_images
    # ndim: 5".
    out = capsys.readouterr().out
    assert "merge_split_filter_masks(cell): skipped" not in out, (
        "project mode produces 2-D masks, so the 2-D filters still apply"
    )
    assert "perimeter_merge" in out, "the filter step really ran"

    mask = np.load(src / "cell_mask_stack" / "plate1_A01_1.npy")
    assert mask.shape == (32, 32), "project mode gives a 2-D mask"


def test_the_3d_modes_skip_the_2d_merge_split_filter_step(tmp_path, fake_model,
                                                          capsys):
    """Applying a 2-D area filter per plane would tear the 3-D labels apart."""
    src = tmp_path / "stack"
    _write_npz(src, (1, 4, 32, 32, 2))

    settings = _base_settings(
        src, z_stack=True, z_segmentation_mode="volumetric", anisotropy=2.0,
        cell_min_object_area=1,
    )
    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    out = capsys.readouterr().out
    assert "merge_split_filter_masks(cell): skipped" in out
    assert "2-D only" in out


def test_plotting_is_skipped_rather_than_crashing_in_3d(tmp_path, fake_model,
                                                        capsys):
    """The z paths have no per-image flow field for plot_cellpose4_output."""
    src = tmp_path / "stack"
    _write_npz(src, (1, 4, 32, 32, 2))

    settings = _base_settings(
        src, plot=True, z_stack=True, z_segmentation_mode="volumetric",
        anisotropy=2.0,
    )
    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    out = capsys.readouterr().out
    assert "plot skipped" in out
    assert "volumetric" in out


def test_verbose_reports_which_mode_ran_and_why(tmp_path, fake_model, capsys):
    """A number without its mode cannot be compared with anything."""
    src = tmp_path / "stack"
    _write_npz(src, (1, 4, 32, 32, 2))

    settings = _base_settings(
        src, verbose=True, z_stack=True, z_segmentation_mode="stitch",
    )
    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    out = capsys.readouterr().out
    assert "[3D]" in out
    assert "not volumetric" in out, (
        "the user must be told that stitching is not 3-D segmentation"
    )
