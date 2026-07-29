"""The 4D (Beta) half of :mod:`spacr.zstack`: x, y, z, t.

The tests are grouped by the claim they defend, and the first group is the
acceptance criterion for the whole feature: **with the 4D settings off, the
2-D and 3-D paths behave exactly as they did before it existed**. Everything
after that only matters if that holds.

Nothing here runs Cellpose, trackpy, btrack, Trackastra or Ultrack. Every
segmentation goes through a passthrough ``segment_fn`` and every link is
computed on a synthetic label volume, so the whole file is CPU-only, offline
and runs in about a second. The one place a real library would be called --
the trackpy backend -- is exercised through a stub that records what it was
handed, which is the part worth pinning: whether ``pos_columns`` is passed
explicitly and whether z was scaled by the anisotropy first.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

import spacr.settings as S
import spacr.zstack as Z
# Imported once, up here, so that the sys.modules stubbing further down never
# races with this module's own `import trackpy` at import time.
import spacr.timelapse as TL


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

def passthrough(array, do_3D=False, anisotropy=None, z_axis=None, stitch=False):
    """A ``segment_fn`` that returns its input as labels.

    Lets a synthetic label array be pushed through the real segmentation
    driver, so the 4-D plumbing is tested rather than a segmenter.
    """
    return np.asarray(array).astype(np.int32)


def one_object_per_plane(n_t=3, n_z=3, size=48, box=8, step=2):
    """``(T, Z, Y, X)``: one object per z plane, all at the same drifting xy.

    This is the shape of data on which confusing t and z produces *plausible*
    output rather than an error, which is why it is the centrepiece of this
    file. Read as ``TZYX`` it is three objects, one per plane, each drifting
    ``step`` px in x per timepoint. Read as ``ZTYX`` it is one object observed
    over three frames -- a perfectly ordinary-looking track, and fiction.
    """
    out = np.zeros((n_t, n_z, size, size), dtype=np.int32)
    for t in range(n_t):
        x0 = 10 + step * t
        for z in range(n_z):
            out[t, z, 10:10 + box, x0:x0 + box] = z + 1
    return out


def stacked_pair(n_t=3, n_z=2, size=32):
    """``(T, Z, Y, X)``: two stationary objects sitting one above the other.

    They are separate in z and identical in xy, so they stay two objects under
    volumetric linking and become one the moment z is projected away.
    """
    out = np.zeros((n_t, n_z, size, size), dtype=np.int32)
    for t in range(n_t):
        out[t, 0, 10:16, 10:16] = 1
        out[t, 1, 10:16, 10:16] = 2
    return out


def default_mask_settings():
    return S.set_default_settings_preprocess_generate_masks({})


NEW_4D_KEYS = [
    "t_stack", "t_axis_order", "t_axis", "frame_interval_s",
    "t_track_backend", "t_link_threshold", "t_max_displacement_px",
    "t_max_displacement_um", "t_project_for_tracking",
]


# ===========================================================================
# 1. ACCEPTANCE: 2-D and 3-D are unchanged when the 4D settings are off
# ===========================================================================

def test_the_4d_settings_are_off_by_default():
    """`t_stack` is the master switch and it starts off, like `z_stack`."""
    settings = default_mask_settings()
    assert settings["t_stack"] is False
    assert settings["z_stack"] is False


def test_plan_is_none_when_4d_is_off_so_no_4d_code_runs():
    """The contract that keeps every other path bit-identical."""
    assert Z.plan_4d_from_settings(default_mask_settings()) is None
    assert Z.plan_4d_from_settings({}) is None
    assert Z.plan_4d_from_settings({"t_stack": False, "t_axis_order": "TZYX"}) is None


def test_the_3d_plan_is_untouched_by_the_new_keys():
    """The 4D keys must not perturb what the 3D switch produces."""
    settings = default_mask_settings()
    assert Z.plan_from_settings(settings) is None

    settings["z_stack"] = True
    spec = Z.plan_from_settings(settings)
    assert spec == Z.ZStackSpec(
        z_axis=None, n_z=None, anisotropy=None, voxel_size_um=None,
        projection="max", mode=Z.MODE_PROJECT, stitch_threshold=0.25,
        resample_to_isotropic=False,
    )


def test_the_3d_defaults_are_untouched_by_the_new_keys():
    settings = default_mask_settings()
    assert settings["z_segmentation_mode"] == "project"
    assert settings["z_projection"] == "max"
    assert settings["z_axis"] is None
    assert settings["anisotropy"] is None
    assert settings["voxel_size_z_um"] is None
    assert settings["voxel_size_xy_um"] is None
    assert settings["stitch_threshold"] == 0.25


@pytest.mark.parametrize("mode", [Z.MODE_PROJECT, Z.MODE_STITCH, Z.MODE_VOLUMETRIC])
def test_3d_segmentation_is_byte_identical_to_the_4d_single_timepoint(mode):
    """The 3-D path is not reimplemented: segment_4d loops around segment_3d."""
    volume = one_object_per_plane(n_t=1)[0].astype(np.float32)
    kwargs = dict(anisotropy=3.0) if mode == Z.MODE_VOLUMETRIC else {}

    three_d = Z.segment_3d(volume, passthrough, mode=mode, z_axis=0, **kwargs)
    four_d = Z.segment_4d(
        volume[None],
        Z.TStackSpec(t_axis=0, z_axis=1, z_mode=mode, **kwargs),
        passthrough,
    )
    assert np.array_equal(four_d.labels[0], three_d.labels)
    assert four_d.z_mode == three_d.mode


def test_the_existing_2d_iou_tracker_still_works_on_a_2d_stack():
    """The guard added to spacr.timelapse must not disturb the 2-D path."""
    masks = np.zeros((3, 32, 32), dtype=np.int32)
    for t in range(3):
        masks[t, 10:18, 10 + t:18 + t] = 1
    df = TL._track_by_iou(masks, iou_threshold=0.1)
    assert sorted(df["frame"]) == [0, 1, 2]
    assert df["track_id"].nunique() == 1


# ===========================================================================
# 2. Axis order: the ambiguous case is refused, the explicit case works
# ===========================================================================

def test_an_ambiguous_shape_is_refused_not_guessed():
    """(3, 3, 48, 48) is three timepoints of three planes, or the reverse."""
    assert Z.detect_axes((3, 3, 48, 48)) is None
    assert Z.detect_axes((3, 3, 48, 48), n_t=3) is None   # fits both readings
    assert Z.detect_axes((3, 3, 48, 48), n_z=3) is None


def test_no_hint_at_all_is_ambiguous_even_when_the_lengths_differ():
    """"Time series are longer than z stacks" is a heuristic, not a fact."""
    assert Z.detect_axes((41, 21, 512, 512)) is None
    assert Z.detect_axes((21, 41, 512, 512)) is None


def test_strict_detection_names_both_readings():
    with pytest.raises(Z.AmbiguousAxisOrderError) as excinfo:
        Z.detect_axes((10, 21, 512, 512), strict=True)
    message = str(excinfo.value)
    assert "TZYX" in message and "ZTYX" in message
    assert "10 timepoints of 21 planes" in message
    assert "21 timepoints of 10 planes" in message
    assert "t_axis_order" in message


def test_a_hint_that_discriminates_settles_the_order():
    tzyx = Z.detect_axes((10, 21, 512, 512), n_t=10)
    assert (tzyx.t_axis, tzyx.z_axis) == (0, 1)
    assert tzyx.name == "TZYX"
    assert tzyx.source == "n_t"

    ztyx = Z.detect_axes((10, 21, 512, 512), n_z=10)
    assert (ztyx.t_axis, ztyx.z_axis) == (1, 0)
    assert ztyx.name == "ZTYX"


def test_a_hint_matching_neither_reading_is_an_error():
    with pytest.raises(Z.TStackError, match="match neither reading"):
        Z.detect_axes((10, 21, 512, 512), n_t=7)


def test_both_hints_together_settle_it():
    order = Z.detect_axes((10, 21, 512, 512), n_t=21, n_z=10)
    assert (order.t_axis, order.z_axis) == (1, 0)
    assert order.source == "n_t+n_z"


def test_detection_refuses_arrays_that_are_not_four_dimensional():
    with pytest.raises(ValueError, match="cannot tell which"):
        Z.detect_axes((21, 512, 512))
    with pytest.raises(ValueError, match="4-D"):
        Z.detect_axes((5, 512, 512, 3, 2, 2))


def test_detection_refuses_a_shape_whose_trailing_axes_are_not_a_plane():
    """A (Y, X, Z, T) layout must not be silently read as (T, Z, Y, X)."""
    with pytest.raises(ValueError, match="not an image plane"):
        Z.detect_axes((512, 512, 21, 10), n_t=10)


def test_a_channel_axis_is_excluded_from_the_question():
    order = Z.detect_axes((10, 3, 21, 512, 512), n_t=10, channel_axis=1)
    assert (order.t_axis, order.z_axis, order.channel_axis) == (0, 2, 1)
    assert order.name == "TCZYX"


def test_an_explicit_order_needs_no_evidence():
    for name in ("TZYX", "ZTYX"):
        order = Z.resolve_axis_order((3, 3, 48, 48), axis_order=name)
        assert order.name == name
        assert order.source == "explicit"


def test_an_explicit_order_that_contradicts_a_count_is_refused():
    with pytest.raises(Z.TStackError, match="disagree"):
        Z.resolve_axis_order((10, 21, 64, 64), axis_order="TZYX", n_t=21)
    with pytest.raises(Z.TStackError, match="disagree"):
        Z.resolve_axis_order((10, 21, 64, 64), axis_order="TZYX", n_z=10)


def test_an_unknown_order_name_is_refused():
    with pytest.raises(Z.TStackError, match="not one of"):
        Z.resolve_axis_order((3, 3, 48, 48), axis_order="XYZT")


def test_one_explicit_index_implies_the_other():
    assert Z.resolve_axis_order((3, 3, 48, 48), t_axis=1).name == "ZTYX"
    assert Z.resolve_axis_order((3, 3, 48, 48), z_axis=1).name == "TZYX"


def test_resolve_falls_through_to_strict_detection():
    with pytest.raises(Z.AmbiguousAxisOrderError):
        Z.resolve_axis_order((3, 3, 48, 48))


# ===========================================================================
# 3. Tracking across t does not link across z
# ===========================================================================

def _tracked(array, order_name, backend=Z.BACKEND_IOU, **kwargs):
    """Declare the axis order, put t first, and link."""
    order = Z.resolve_axis_order(array, axis_order=order_name)
    spec = Z.TStackSpec(t_axis=order.t_axis, z_axis=order.z_axis, **kwargs)
    return Z.track_4d(Z.as_t_first(array, spec), spec, backend=backend)


def test_the_correct_axis_order_gives_one_track_per_object():
    """Three objects, one per z plane, drifting in x: three tracks."""
    array = one_object_per_plane()
    result = _tracked(array, "TZYX", link_threshold=0.3)

    assert result.n_tracks == 3
    tracks = result.tracks
    assert sorted(tracks["frame"].unique().tolist()) == [0, 1, 2]
    for track_id, rows in tracks.groupby("track_id"):
        assert sorted(rows["frame"].tolist()) == [0, 1, 2]


def test_a_track_never_spans_two_z_planes():
    """The failure this whole feature exists to prevent, asserted directly."""
    array = one_object_per_plane()
    result = _tracked(array, "TZYX", link_threshold=0.3)

    for t, volume in enumerate(result.labels):
        for value in np.unique(volume)[1:]:
            planes = np.unique(np.nonzero(volume == value)[0])
            assert planes.size == 1, (
                f"track {value} occupies z planes {planes.tolist()} at t={t}; "
                f"the linker joined objects across z"
            )


def test_declaring_the_wrong_axis_order_gives_a_different_plausible_answer():
    """Read as ZTYX the same buffer yields one smooth, entirely fake track.

    Nothing about the output says it is wrong -- which is exactly why
    detect_axes refuses to choose and the order has to be declared.
    """
    array = one_object_per_plane()
    right = _tracked(array, "TZYX", link_threshold=0.3)
    wrong = _tracked(array, "ZTYX", link_threshold=0.3)

    assert right.n_tracks == 3
    assert wrong.n_tracks == 1
    # And the fake one looks perfectly healthy: one object, present in every
    # "frame", no gaps.
    assert sorted(wrong.tracks["frame"].unique().tolist()) == [0, 1, 2]
    assert not right.tracks.equals(wrong.tracks)


def test_two_objects_stacked_in_z_stay_two_objects():
    array = stacked_pair()
    result = _tracked(array, "TZYX")
    assert result.n_tracks == 2
    for volume in result.labels:
        assert np.unique(volume)[1:].size == 2


# ===========================================================================
# 4. A 2-D-only backend handed a volume raises and names itself
# ===========================================================================

@pytest.mark.parametrize(
    "backend", [Z.BACKEND_BTRACK, Z.BACKEND_TRACKASTRA, Z.BACKEND_ULTRACK])
def test_a_two_d_only_backend_refuses_a_volume(backend):
    labels = stacked_pair()
    with pytest.raises(Z.TrackerIsTwoDError) as excinfo:
        Z.track_4d(labels, Z.TStackSpec(), backend=backend)
    message = str(excinfo.value)
    assert backend in message
    assert "spacr.timelapse" in message
    # It names which limit it is: the library's, or spaCR's adapter's.
    assert "3-D" in message or "3D" in message


@pytest.mark.parametrize(
    "backend", [Z.BACKEND_BTRACK, Z.BACKEND_TRACKASTRA, Z.BACKEND_ULTRACK])
def test_project_for_tracking_does_not_unlock_a_backend_spacr_cannot_drive(backend):
    """Projecting and then linking with a *different* linker would be worse."""
    labels = stacked_pair()
    with pytest.raises(Z.TrackerIsTwoDError):
        Z.track_4d(labels, Z.TStackSpec(), backend=backend,
                   project_for_tracking=True)


@pytest.mark.parametrize(
    "backend", [Z.BACKEND_BTRACK, Z.BACKEND_TRACKASTRA, Z.BACKEND_ULTRACK])
def test_a_two_d_only_backend_on_a_flat_stack_points_at_timelapse(backend):
    flat = stacked_pair()[:, 0]
    with pytest.raises(Z.TStackError, match="spacr.timelapse"):
        Z.track_4d(flat, Z.TStackSpec(), backend=backend)


def test_the_backend_table_separates_the_library_from_the_adapter():
    """Saying "spaCR cannot" is not the same as saying "the library cannot"."""
    for name in (Z.BACKEND_BTRACK, Z.BACKEND_TRACKASTRA, Z.BACKEND_ULTRACK):
        record = Z.TRACK_BACKENDS[name]
        assert record.links_3d is False
        assert record.library_links_3d is True
    for name in (Z.BACKEND_IOU, Z.BACKEND_CENTROID, Z.BACKEND_TRACKPY):
        assert Z.TRACK_BACKENDS[name].links_3d is True


def test_an_unknown_backend_is_refused():
    with pytest.raises(Z.TStackError, match="not one of"):
        Z.track_4d(stacked_pair(), Z.TStackSpec(), backend="magic")
    with pytest.raises(Z.TStackError, match="not one of"):
        Z.TStackSpec(track_backend="magic")


# ===========================================================================
# 5. Projection is a choice, and it says what it destroyed
# ===========================================================================

def test_project_for_tracking_merges_what_it_projects_and_says_so():
    labels = stacked_pair()
    kept = Z.track_4d(labels, Z.TStackSpec(), backend=Z.BACKEND_IOU)
    flattened = Z.track_4d(labels, Z.TStackSpec(), backend=Z.BACKEND_IOU,
                           project_for_tracking=True)

    assert kept.n_tracks == 2
    assert flattened.n_tracks == 1
    assert flattened.projected is True
    assert any("collapsed" in note for note in flattened.notes)
    assert "COLLAPSED" in Z.format_4d(flattened)


def test_label_projection_takes_the_majority_not_the_maximum():
    """Max along z would just pick the highest-numbered label."""
    volume = np.zeros((4, 8, 8), dtype=np.int32)
    volume[0:3, 2:6, 2:6] = 1     # three planes
    volume[3, 2:6, 2:6] = 7       # one plane, but a bigger id
    assert set(np.unique(Z.project_labels(volume)).tolist()) == {0, 1}


def test_projecting_a_two_d_plane_is_a_no_op():
    plane = np.array([[0, 1], [2, 0]], dtype=np.int32)
    assert np.array_equal(Z.project_labels(plane), plane)


# ===========================================================================
# 6. Anisotropy applies to linking, not just to segmentation
# ===========================================================================

def _two_plane_jump(n_z=5, size=32, jump=2):
    """An object at z=0 at t=0 and at z=jump at t=1, same xy."""
    labels = np.zeros((2, n_z, size, size), dtype=np.int32)
    labels[0, 0, 8:12, 8:12] = 1
    labels[1, jump, 8:12, 8:12] = 1
    return labels


def test_anisotropy_changes_which_objects_link():
    """At dz/dxy = 5 a two-plane jump is ten pixels, not two."""
    labels = _two_plane_jump()

    linked = Z.track_4d(labels, Z.TStackSpec(anisotropy=1.0),
                        backend=Z.BACKEND_CENTROID, max_displacement_px=3.0)
    assert linked.n_tracks == 1, "at anisotropy 1.0 the jump is inside the gate"

    split = Z.track_4d(labels, Z.TStackSpec(anisotropy=5.0),
                       backend=Z.BACKEND_CENTROID, max_displacement_px=3.0)
    assert split.n_tracks == 2, "at anisotropy 5.0 the jump is 10 px, outside it"
    assert split.anisotropy == 5.0


def test_a_volumetric_centroid_link_will_not_assume_isotropy():
    with pytest.raises(Z.UnknownAnisotropyError):
        Z.track_4d(_two_plane_jump(), Z.TStackSpec(),
                   backend=Z.BACKEND_CENTROID, max_displacement_px=3.0)


def test_the_gate_can_be_stated_in_micrometres_instead():
    """In physical coordinates the anisotropy is already baked in."""
    labels = _two_plane_jump()
    voxel = (5.0, 1.0, 1.0)   # dz = 5 um, dxy = 1 um -> anisotropy 5

    split = Z.track_4d(labels, Z.TStackSpec(voxel_size_um=voxel),
                       backend=Z.BACKEND_CENTROID, max_displacement_um=3.0)
    assert split.n_tracks == 2
    linked = Z.track_4d(labels, Z.TStackSpec(voxel_size_um=voxel),
                        backend=Z.BACKEND_CENTROID, max_displacement_um=12.0)
    assert linked.n_tracks == 1


def test_micrometres_without_a_voxel_size_is_refused():
    with pytest.raises(Z.TStackError, match="voxel size is not known"):
        Z.track_4d(_two_plane_jump(), Z.TStackSpec(),
                   backend=Z.BACKEND_CENTROID, max_displacement_um=3.0)


def test_a_distance_backend_without_a_gate_is_refused():
    with pytest.raises(Z.UnknownDisplacementError, match="will not pick a default"):
        Z.track_4d(_two_plane_jump(), Z.TStackSpec(anisotropy=5.0),
                   backend=Z.BACKEND_CENTROID)


def test_the_gate_cannot_be_given_in_two_units_at_once():
    with pytest.raises(Z.TStackError, match="will not pick one"):
        Z.track_4d(_two_plane_jump(), Z.TStackSpec(anisotropy=5.0),
                   backend=Z.BACKEND_CENTROID,
                   max_displacement_px=3.0, max_displacement_um=3.0)
    with pytest.raises(Z.TStackError, match="Set exactly one"):
        Z.TStackSpec(max_displacement_px=3.0, max_displacement_um=3.0)


def test_the_overlap_backend_genuinely_ignores_anisotropy():
    """It computes no distance, so there is nothing for anisotropy to scale."""
    labels = one_object_per_plane()
    a = Z.track_4d(labels, Z.TStackSpec(anisotropy=1.0), backend=Z.BACKEND_IOU,
                   link_threshold=0.3)
    b = Z.track_4d(labels, Z.TStackSpec(anisotropy=9.0), backend=Z.BACKEND_IOU,
                   link_threshold=0.3)
    assert np.array_equal(a.labels, b.labels)
    assert a.anisotropy is None
    assert any("anisotropy is ignored" in note for note in b.notes)


def test_a_flat_stack_needs_no_anisotropy_for_a_distance_link():
    flat = np.zeros((2, 32, 32), dtype=np.int32)
    flat[0, 8:12, 8:12] = 1
    flat[1, 9:13, 8:12] = 1
    result = Z.track_4d(flat, Z.TStackSpec(), backend=Z.BACKEND_CENTROID,
                        max_displacement_px=3.0)
    assert result.n_tracks == 1


# ===========================================================================
# 7. The degenerate cases are exact
# ===========================================================================

def test_a_single_z_plane_is_the_ordinary_2d_path():
    array = one_object_per_plane(n_z=1).astype(np.float32)
    spec = Z.TStackSpec(t_axis=0, z_axis=1)
    result = Z.segment_4d(array, spec, passthrough)

    assert result.labels.shape == (3, 48, 48)
    assert result.z_mode == Z.MODE_SINGLE_PLANE
    for t in range(array.shape[0]):
        assert np.array_equal(result.labels[t], passthrough(array[t, 0]))


def test_a_spec_with_no_z_axis_is_a_plain_time_series():
    array = one_object_per_plane(n_z=1)[:, 0].astype(np.float32)
    spec = Z.TStackSpec(t_axis=0, z_axis=None)
    result = Z.segment_4d(array, spec, passthrough)
    assert result.labels.shape == array.shape
    assert result.has_z is False


def test_the_2d_table_matches_the_existing_tracks_table_exactly():
    """The visualiser and the motility assay must need no change."""
    array = one_object_per_plane(n_z=1)
    spec = Z.TStackSpec(t_axis=0, z_axis=1)
    tracked = Z.track_4d(Z.as_t_first(array, spec)[:, 0], spec,
                         backend=Z.BACKEND_IOU, link_threshold=0.3)

    mine = tracked.tracks[list(Z.BASE_TRACK_COLUMNS)]
    theirs = TL._relabelled_stack_to_tracks_df(tracked.labels)[
        list(Z.BASE_TRACK_COLUMNS)]
    pd.testing.assert_frame_equal(mine, theirs)


def test_a_single_timepoint_carries_the_3d_note():
    volume = one_object_per_plane(n_t=1)[0].astype(np.float32)
    result = Z.segment_4d(volume[None], Z.TStackSpec(), passthrough)
    assert result.n_t == 1
    assert any("3-D run" in note for note in result.notes)


# ===========================================================================
# 8. Truncation in z AND in t
# ===========================================================================

def _truncation_case(n_t=4, n_z=3, size=24):
    """Three tracks with different truncation defects.

    * 1 -- every timepoint, touching z=0            -> truncated in z and in t
    * 2 -- middle timepoints only, interior in z    -> truncated in neither
    * 3 -- last two timepoints, touching the last z -> truncated in z and in t
    """
    labels = np.zeros((n_t, n_z, size, size), dtype=np.int32)
    labels[:, 0:2, 2:6, 2:6] = 1
    labels[1:3, 1:2, 10:14, 10:14] = 2
    labels[2:, 1:3, 18:22, 18:22] = 3
    return labels


def test_truncation_in_z_and_in_t_are_flagged_separately():
    table = Z.volume_tracks(_truncation_case())
    flags = table.groupby("track_id")[["truncated_z", "truncated_t"]].any()

    assert flags.loc[1, "truncated_z"] and flags.loc[1, "truncated_t"]
    assert not flags.loc[2, "truncated_z"] and not flags.loc[2, "truncated_t"]
    assert flags.loc[3, "truncated_z"] and flags.loc[3, "truncated_t"]


def test_flag_truncated_t_finds_only_the_end_timepoints():
    assert Z.flag_truncated_t(_truncation_case()).tolist() == [1, 3]
    assert Z.flag_truncated_t(np.zeros((0, 2, 4, 4))).size == 0
    assert Z.flag_truncated_t(np.zeros((4, 4))).size == 0


def test_the_track_result_reports_its_truncated_fraction():
    result = Z.track_4d(_truncation_case(), Z.TStackSpec(),
                        backend=Z.BACKEND_IOU)
    assert result.truncated_tracks.size == 2
    assert result.truncated_fraction == pytest.approx(2 / 3)
    assert any("truncated in t" in note for note in result.notes)


def test_an_empty_result_has_no_truncated_fraction():
    empty = Z.TrackResult(labels=np.zeros((2, 2, 4, 4), dtype=np.int64))
    assert empty.truncated_fraction == 0.0


# ===========================================================================
# 9. The output table
# ===========================================================================

def test_the_base_columns_come_first_and_unchanged():
    table = Z.volume_tracks(_truncation_case())
    assert list(table.columns)[:5] == list(Z.BASE_TRACK_COLUMNS)
    assert (table["track_id"] == table["original_label"]).all()


def test_a_volume_gets_a_volume_column_and_a_plane_gets_an_area_column():
    """A voxel count and a px^2 area never share a column."""
    volumetric = Z.volume_tracks(_truncation_case())
    assert "volume_voxels" in volumetric and "area_px2" not in volumetric
    assert "z" in volumetric

    flat = Z.volume_tracks(_truncation_case()[:, 0])
    assert "area_px2" in flat and "volume_voxels" not in flat
    assert "z" not in flat


def test_physical_columns_appear_only_when_the_physics_is_known():
    labels = _truncation_case()
    bare = Z.volume_tracks(labels, Z.TStackSpec())
    assert "volume_um3" not in bare and "time_s" not in bare

    spec = Z.TStackSpec(voxel_size_um=(5.0, 0.5, 0.5), frame_interval_s=30.0)
    full = Z.volume_tracks(labels, spec)
    assert {"volume_um3", "z_um", "time_s"} <= set(full.columns)
    row = full.iloc[0]
    assert row["volume_um3"] == pytest.approx(row["volume_voxels"] * 5.0 * 0.25)
    assert full["time_s"].max() == pytest.approx(3 * 30.0)


def test_every_emitted_column_has_a_declared_unit():
    labels = _truncation_case()
    spec = Z.TStackSpec(voxel_size_um=(5.0, 0.5, 0.5), frame_interval_s=30.0)
    for table in (Z.volume_tracks(labels, spec),
                  Z.volume_tracks(labels[:, 0], spec)):
        assert set(table.columns) <= set(Z.TRACK_COLUMN_UNITS)


def test_an_empty_stack_still_gives_the_right_columns():
    table = Z.volume_tracks(np.zeros((2, 3, 8, 8), dtype=np.int32))
    assert list(table.columns)[:5] == list(Z.BASE_TRACK_COLUMNS)
    assert table.empty


def test_the_table_refuses_an_array_that_is_not_t_first():
    with pytest.raises(Z.TStackError, match="t-first"):
        Z.volume_tracks(np.zeros((4, 4), dtype=np.int32))
    with pytest.raises(Z.TStackError, match="t-first"):
        Z.track_4d(np.zeros((2, 2, 2, 8, 8), dtype=np.int32), Z.TStackSpec())


# ===========================================================================
# 10. Memory: volumes are iterated, never materialised
# ===========================================================================

def test_iter_volumes_yields_views_and_never_copies():
    array = np.arange(3 * 2 * 64 * 64, dtype=np.float32).reshape(3, 2, 64, 64)
    spec = Z.TStackSpec(t_axis=0, z_axis=1)
    volumes = list(Z.iter_volumes(array, spec))

    assert len(volumes) == 3
    for t, volume in enumerate(volumes):
        assert volume.shape == (2, 64, 64)
        assert volume.base is not None, "a copy was made"
        assert np.shares_memory(volume, array)
        assert np.array_equal(volume, array[t])


def test_iter_volumes_reorders_a_z_first_acquisition_without_copying():
    array = np.arange(2 * 3 * 64 * 64, dtype=np.float32).reshape(2, 3, 64, 64)
    spec = Z.TStackSpec(t_axis=1, z_axis=0)
    volumes = list(Z.iter_volumes(array, spec))
    assert len(volumes) == 3
    assert np.array_equal(volumes[0], array[:, 0])
    assert np.shares_memory(volumes[0], array)


def test_iter_volumes_refuses_a_shape_that_contradicts_the_spec():
    array = np.zeros((3, 2, 8, 8), dtype=np.float32)
    with pytest.raises(Z.TStackError, match="n_t"):
        list(Z.iter_volumes(array, Z.TStackSpec(n_t=5)))
    with pytest.raises(Z.TStackError, match="n_z"):
        list(Z.iter_volumes(array, Z.TStackSpec(n_z=5)))


def test_a_flat_array_under_a_4d_spec_names_the_ingest_as_the_cause():
    with pytest.raises(Z.TAxisNotPresentError) as excinfo:
        Z.as_t_first(np.zeros((3, 64, 64)), Z.TStackSpec())
    message = str(excinfo.value)
    assert "spacr.io" in message
    assert "will not segment the projection" in message


def test_the_peak_footprint_is_one_volume_plus_every_label():
    peak = Z.estimate_peak_bytes_4d((41, 21, 2048, 2048), z_mode=Z.MODE_VOLUMETRIC,
                                    anisotropy=5.0)
    labels = 41 * 21 * 2048 * 2048 * 4
    live = Z.estimate_peak_bytes((21, 2048, 2048), mode=Z.MODE_VOLUMETRIC,
                                 anisotropy=5.0)
    assert peak == labels + live

    # Under 'project' the labels lose their z axis and the number collapses.
    projected = Z.estimate_peak_bytes_4d((41, 21, 2048, 2048),
                                         z_mode=Z.MODE_PROJECT)
    assert projected < peak / 10


def test_the_peak_estimate_needs_a_four_dimensional_shape():
    with pytest.raises(ValueError, match="at least"):
        Z.estimate_peak_bytes_4d((21, 512, 512))


# ===========================================================================
# 11. The trackpy backend: stubbed, never run for real
# ===========================================================================

class _StubTrackpy:
    """Records what it was handed and links everything into one track."""

    def __init__(self):
        self.calls = []

    def link(self, features, search_range, pos_columns=None, t_column="frame",
             memory=0):
        self.calls.append(dict(search_range=search_range,
                               pos_columns=pos_columns, t_column=t_column,
                               memory=memory, features=features.copy()))
        out = features.copy()
        out["particle"] = 0
        return out


def test_the_trackpy_backend_links_in_three_dimensions_explicitly(monkeypatch):
    stub = _StubTrackpy()
    monkeypatch.setitem(sys.modules, "trackpy", stub)

    result = Z.track_4d(_two_plane_jump(), Z.TStackSpec(anisotropy=5.0),
                        backend=Z.BACKEND_TRACKPY, max_displacement_px=20.0)

    assert result.n_tracks == 1
    call = stub.calls[0]
    assert call["pos_columns"] == ["z", "y", "x"], (
        "trackpy guesses its position columns from whatever the frame happens "
        "to carry; the dimensionality of the link must be a decision, not an "
        "accident"
    )
    assert call["search_range"] == 20.0
    # z is pre-scaled: plane 2 at anisotropy 5 must arrive as 10.
    assert sorted(call["features"]["z"].tolist()) == [0.0, 10.0]


def test_the_trackpy_backend_says_what_to_install_when_it_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "trackpy", None)
    with pytest.raises(RuntimeError, match="pip install trackpy"):
        Z.track_4d(_two_plane_jump(), Z.TStackSpec(anisotropy=5.0),
                   backend=Z.BACKEND_TRACKPY, max_displacement_px=20.0)


def test_the_trackpy_backend_handles_an_empty_stack(monkeypatch):
    stub = _StubTrackpy()
    monkeypatch.setitem(sys.modules, "trackpy", stub)
    result = Z.track_4d(np.zeros((2, 3, 8, 8), dtype=np.int32),
                        Z.TStackSpec(anisotropy=2.0),
                        backend=Z.BACKEND_TRACKPY, max_displacement_px=5.0)
    assert result.n_tracks == 0
    assert stub.calls == []


# ===========================================================================
# 12. Segmentation driver
# ===========================================================================

def test_segment_4d_delegates_every_z_mode_to_the_z_half():
    array = one_object_per_plane().astype(np.float32)
    spec = Z.TStackSpec(z_mode=Z.MODE_STITCH, stitch_threshold=0.3)
    result = Z.segment_4d(array, spec, passthrough)
    assert result.labels.shape[:2] == (3, 3)
    assert result.z_mode == Z.MODE_STITCH
    assert len(result.z_results) == 3

    for t in range(array.shape[0]):
        expected = Z.segment_3d(array[t], passthrough, mode=Z.MODE_STITCH,
                                stitch_threshold=0.3, z_axis=0)
        assert np.array_equal(result.labels[t], expected.labels)

    # Proof the z half really ran rather than the labels passing through: the
    # three per-plane objects sit exactly on top of one another in xy, so
    # stitching links them into a single 3-D object per timepoint.
    assert result.objects_per_timepoint == [1, 1, 1]


def test_segment_4d_is_verbose_on_request(capsys):
    array = one_object_per_plane().astype(np.float32)
    Z.segment_4d(array, Z.TStackSpec(z_mode=Z.MODE_PROJECT), passthrough,
                 verbose=True)
    assert "[4D] t=0:" in capsys.readouterr().out


def test_segment_4d_refuses_an_acquisition_with_no_timepoints():
    array = np.zeros((0, 3, 8, 8), dtype=np.float32)
    with pytest.raises(Z.TStackError, match="zero timepoints"):
        Z.segment_4d(array, Z.TStackSpec(), passthrough)


def test_segment_4d_refuses_timepoints_that_segment_to_different_shapes():
    def ragged(array, **kwargs):
        array = np.asarray(array)
        ragged.calls += 1
        return array[..., :-1] if ragged.calls > 1 else array
    ragged.calls = 0

    array = one_object_per_plane().astype(np.float32)
    with pytest.raises(Z.TStackError, match="different shapes"):
        Z.segment_4d(array, Z.TStackSpec(z_mode=Z.MODE_PROJECT), ragged)


def test_the_result_can_be_tracked_without_restating_the_spec():
    array = one_object_per_plane().astype(np.float32)
    spec = Z.TStackSpec(z_mode=Z.MODE_VOLUMETRIC, anisotropy=3.0,
                        link_threshold=0.3)
    segmented = Z.segment_4d(array, spec, passthrough)
    tracked = Z.track_4d(segmented)
    assert tracked.n_tracks == 3


# ===========================================================================
# 13. The spec itself
# ===========================================================================

def test_the_spec_refuses_to_be_self_contradictory():
    with pytest.raises(Z.TStackError, match="cannot be both"):
        Z.TStackSpec(t_axis=1, z_axis=1)
    with pytest.raises(Z.TStackError, match="collides"):
        Z.TStackSpec(t_axis=0, z_axis=1, channel_axis=1)
    with pytest.raises(Z.TStackError, match="IoU in"):
        Z.TStackSpec(link_threshold=1.5)
    with pytest.raises(Z.TStackError, match="finite number > 0"):
        Z.TStackSpec(max_displacement_px=0)
    with pytest.raises(Z.TStackError, match="seconds"):
        Z.TStackSpec(frame_interval_s=-1)


def test_the_spec_validates_its_z_half_at_construction():
    """A bad z mode must not survive until the first field has been read."""
    with pytest.raises(Z.ZStackError, match="z_segmentation_mode"):
        Z.TStackSpec(z_mode="sideways")
    with pytest.raises(Z.ZStackError, match="z_projection"):
        Z.TStackSpec(projection="median")


def test_the_spec_hands_its_z_settings_straight_through():
    spec = Z.TStackSpec(z_mode=Z.MODE_VOLUMETRIC, anisotropy=4.0,
                        voxel_size_um=(4.0, 1.0, 1.0), stitch_threshold=0.4)
    z_spec = spec.to_z_spec()
    assert z_spec.mode == Z.MODE_VOLUMETRIC
    assert z_spec.anisotropy == 4.0
    assert z_spec.stitch_threshold == 0.4
    assert z_spec.z_axis == 0
    assert spec.require_anisotropy() == 4.0
    assert spec.voxel_size == spec.voxel_size_um


def test_the_spec_names_its_axis_order():
    assert Z.TStackSpec(t_axis=0, z_axis=1).axis_order == "TZYX"
    assert Z.TStackSpec(t_axis=1, z_axis=0).axis_order == "ZTYX"
    assert Z.TStackSpec(t_axis=0, z_axis=None).axis_order is None
    assert Z.TStackSpec(t_axis=0, z_axis=2).axis_order is None
    assert Z.TStackSpec().backend.name == Z.BACKEND_IOU


def test_an_axis_order_cannot_reuse_an_index():
    with pytest.raises(Z.TStackError, match="same axis index"):
        Z.AxisOrder(t_axis=0, z_axis=0, y_axis=2, x_axis=3)


# ===========================================================================
# 14. Settings
# ===========================================================================

def test_turning_4d_on_without_an_axis_order_stops_the_run():
    with pytest.raises(Z.AmbiguousAxisOrderError, match="will not guess"):
        Z.plan_4d_from_settings({"t_stack": True})


def test_the_axis_order_setting_reaches_the_spec():
    for name, axes in (("TZYX", (0, 1)), ("ZTYX", (1, 0))):
        spec = Z.plan_4d_from_settings({"t_stack": True, "t_axis_order": name})
        assert (spec.t_axis, spec.z_axis) == axes
        assert spec.axis_order == name


def test_a_bad_axis_order_setting_is_refused():
    with pytest.raises(Z.TStackError, match="not one of"):
        Z.plan_4d_from_settings({"t_stack": True, "t_axis_order": "XYZT"})


def test_either_axis_index_alone_implies_the_other():
    assert Z.plan_4d_from_settings({"t_stack": True, "t_axis": 1}).z_axis == 0
    assert Z.plan_4d_from_settings({"t_stack": True, "z_axis": 1}).t_axis == 0


def test_an_explicit_index_that_contradicts_the_order_setting_is_refused():
    """Letting t_axis_order silently win would transpose the array unnoticed."""
    with pytest.raises(Z.TStackError, match="will not pick one"):
        Z.plan_4d_from_settings({"t_stack": True, "t_axis_order": "TZYX",
                                 "t_axis": 1})
    with pytest.raises(Z.TStackError, match="will not pick one"):
        Z.plan_4d_from_settings({"t_stack": True, "t_axis_order": "TZYX",
                                 "z_axis": 0})
    # Agreeing values are fine.
    spec = Z.plan_4d_from_settings({"t_stack": True, "t_axis_order": "ZTYX",
                                    "t_axis": 1, "z_axis": 0})
    assert spec.axis_order == "ZTYX"


def test_a_trailing_z_axis_setting_cannot_imply_a_time_axis():
    """`z_axis=2` is a valid 3-D setting; arithmetic on it would give t=-1."""
    for key in ("z_axis", "t_axis"):
        with pytest.raises(Z.TStackError, match="two leading axes"):
            Z.plan_4d_from_settings({"t_stack": True, key: 2})


def test_the_frame_interval_falls_back_to_the_motility_setting():
    """One physical number, not two competing settings."""
    base = {"t_stack": True, "t_axis_order": "TZYX"}
    assert Z.plan_4d_from_settings(base).frame_interval_s is None
    assert Z.plan_4d_from_settings(
        {**base, "seconds_per_frame": 60}).frame_interval_s == 60.0
    assert Z.plan_4d_from_settings(
        {**base, "seconds_per_frame": 60,
         "frame_interval_s": 12.5}).frame_interval_s == 12.5


def test_the_z_half_of_the_settings_is_read_by_both_planners():
    settings = default_mask_settings()
    settings.update(t_stack=True, t_axis_order="TZYX", z_stack=True,
                    z_segmentation_mode="volumetric", anisotropy=3.0)
    z_spec = Z.plan_from_settings(settings)
    t_spec = Z.plan_4d_from_settings(settings)
    assert t_spec.to_z_spec().mode == z_spec.mode
    assert t_spec.to_z_spec().anisotropy == z_spec.anisotropy


def test_a_volumetric_run_without_an_anisotropy_stops_at_plan_time():
    with pytest.raises(Z.UnknownAnisotropyError):
        Z.plan_4d_from_settings({"t_stack": True, "t_axis_order": "TZYX",
                                 "z_segmentation_mode": "volumetric"})


def test_a_distance_backend_without_a_gate_stops_at_plan_time():
    with pytest.raises(Z.UnknownDisplacementError):
        Z.plan_4d_from_settings({"t_stack": True, "t_axis_order": "TZYX",
                                 "t_track_backend": "centroid"})


def test_a_distance_backend_on_volumes_needs_the_anisotropy_at_plan_time():
    with pytest.raises(Z.UnknownAnisotropyError):
        Z.plan_4d_from_settings({
            "t_stack": True, "t_axis_order": "TZYX",
            "z_segmentation_mode": "stitch",
            "t_track_backend": "centroid", "t_max_displacement_px": 10.0,
        })


def test_a_2d_only_backend_with_a_volumetric_z_mode_stops_at_plan_time():
    with pytest.raises(Z.TrackerIsTwoDError, match="does not rescue"):
        Z.plan_4d_from_settings({
            "t_stack": True, "t_axis_order": "TZYX",
            "z_segmentation_mode": "stitch", "t_track_backend": "btrack",
            "t_project_for_tracking": True,
        })


def test_a_2d_only_backend_is_fine_when_z_is_projected():
    spec = Z.plan_4d_from_settings({
        "t_stack": True, "t_axis_order": "TZYX",
        "z_segmentation_mode": "project", "t_track_backend": "btrack",
    })
    assert spec.track_backend == "btrack"


def test_the_voxel_size_settings_derive_the_spec_voxel_size():
    spec = Z.plan_4d_from_settings({
        "t_stack": True, "t_axis_order": "TZYX",
        "voxel_size_z_um": 5.0, "voxel_size_xy_um": 0.5,
    })
    assert spec.voxel_size_um == (5.0, 0.5, 0.5)


# --- category / tooltip invariants for the new keys -------------------------

def test_every_new_key_is_declared_typed_and_described():
    for key in NEW_4D_KEYS:
        assert key in S.expected_types, f"{key} has no declared type"
        assert key in S.tooltips, f"{key} has no tooltip"
        assert S.tooltips[key].startswith("("), f"{key} tooltip has no type prefix"


def test_every_new_key_appears_in_exactly_one_category():
    listed = [k for keys in S.categories.values() for k in keys]
    for key in NEW_4D_KEYS:
        assert listed.count(key) == 1, f"{key} is categorised {listed.count(key)} times"


def test_the_new_keys_have_their_own_4d_beta_panel():
    panel = S.categories["4D Settings (Beta)"]
    assert set(NEW_4D_KEYS) <= set(panel)
    assert "t_stack" == panel[0]
    assert not set(NEW_4D_KEYS) & set(S.categories["3D Settings (Beta)"])


def test_every_new_key_is_offered_by_the_mask_and_timelapse_panels():
    for factory in (S.set_default_settings_preprocess_generate_masks,
                    S.get_timelapse_settings):
        offered = factory({})
        for key in NEW_4D_KEYS:
            assert key in offered, f"{key} missing from {factory.__name__}"


# ===========================================================================
# 15. The guard added to spacr.timelapse
# ===========================================================================

def test_the_iou_linker_no_longer_silently_tracks_a_volume():
    """It used to return plausible tracks for a (T, Z, Y, X) array."""
    with pytest.raises(ValueError, match="along z"):
        TL._track_by_iou(stacked_pair(), iou_threshold=0.1)


def test_the_feature_table_names_z_instead_of_failing_out_of_skimage():
    with pytest.raises(ValueError, match="4D"):
        TL._prepare_for_tracking(stacked_pair())


def test_the_relabelled_stack_table_refuses_a_volume():
    with pytest.raises(ValueError, match="2-D frames"):
        TL._relabelled_stack_to_tracks_df(stacked_pair())


def test_the_guard_accepts_a_list_of_2d_frames():
    frames = [np.zeros((8, 8), dtype=np.int32) for _ in range(3)]
    TL._require_2d_frames(frames, "test")          # does not raise
    TL._require_2d_frames(np.zeros((0, 8, 8)), "test")


# ===========================================================================
# 16. Reporting
# ===========================================================================

def test_format_4d_renders_a_segmentation_result():
    array = one_object_per_plane().astype(np.float32)
    spec = Z.TStackSpec(z_mode=Z.MODE_VOLUMETRIC, anisotropy=3.0,
                        frame_interval_s=30.0)
    text = Z.format_4d(Z.segment_4d(array, spec, passthrough))
    assert "axis order      : TZYX" in text
    assert "timepoints      : 3" in text
    assert "anisotropy      : 3 (dz/dxy)" in text
    assert "frame interval  : 30 s" in text


def test_format_4d_renders_a_tracking_result():
    text = Z.format_4d(Z.track_4d(_truncation_case(), Z.TStackSpec(),
                                  backend=Z.BACKEND_IOU))
    assert "backend         : iou" in text
    assert "not used by this backend" in text
    assert "truncated in t" in text


def test_format_4d_refuses_anything_else():
    with pytest.raises(TypeError, match="TStackResult"):
        Z.format_4d({"labels": None})


def test_a_segmentation_result_with_no_timepoints_reports_a_2d_mode():
    assert Z.TStackResult(labels=np.zeros((0, 4, 4))).z_mode == Z.MODE_SINGLE_PLANE
