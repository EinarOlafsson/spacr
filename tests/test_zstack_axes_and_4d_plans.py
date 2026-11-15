"""Table-driven tests of the z-axis handling and the 4-D (t + z) plumbing in ``spacr.zstack``.

Every number here is a fact about the sample -- which axis is z, which is
time, how far an object may move between frames -- and a wrong one does not
fail, it silently produces a volume or a track in the wrong units.  So each
unit is walked over the spellings that must be accepted (with the exact
spec, view or summary they give) and the ones that must be refused by name.
All label arrays are synthetic; linking uses the built-in ``iou`` backend,
plus ``trackpy`` when it is installed.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacr.zstack as Z


def drifting_movie(n_t=3, size=16, box=6, step=1):
    """``(T, Y, X)`` labels: one square that drifts ``step`` px in x per frame."""
    out = np.zeros((n_t, size, size), dtype=np.int32)
    for t in range(n_t):
        x0 = 4 + step * t
        out[t, 4:4 + box, x0:x0 + box] = 1
    return out


def two_then_one(size=12):
    """``(T, Y, X)`` labels: two objects at t=0, one at t=1."""
    out = np.zeros((2, size, size), dtype=np.int32)
    out[0, 1:4, 1:4] = 1
    out[0, 7:10, 7:10] = 2
    out[1, 1:4, 1:4] = 1
    return out


def test_as_z_first_moves_a_known_or_detected_z_axis_to_the_front(subtests):
    """A 3-D volume's short axis is detected; 4-D needs it named; planes and cubes are refused."""
    accepted = [
        # id, shape, z_axis, resolved axis
        ("zyx-detected", (5, 64, 64), None, 0),
        ("yxz-detected", (64, 64, 5), None, 2),
        ("yzx-detected", (64, 5, 64), None, 1),
        ("negative-axis-wraps", (5, 64, 64), -3, 0),
        ("4d-named-front", (4, 64, 64, 3), 0, 0),
        ("4d-named-middle", (64, 64, 4, 3), 2, 2),
    ]
    for case_id, shape, z_axis, resolved in accepted:
        with subtests.test(msg=case_id):
            volume = np.arange(np.prod(shape)).reshape(shape)
            moved, axis = Z._as_z_first(volume, z_axis)
            assert axis == resolved
            assert np.array_equal(moved, np.moveaxis(volume, resolved, 0))

    refused = [
        ("a-plane", np.zeros((8, 8)), ValueError,
         "expected a volume with at least 3 axes, got ndim=2 (shape (8, 8))"),
        ("4d-without-z", np.zeros((4, 64, 64, 3)), Z.AmbiguousZAxisError,
         "z_axis must be given for a 4-D array (shape (4, 64, 64, 3)); it is "
         "only detectable for a plain 3-D volume"),
        ("a-cube", np.zeros((64, 64, 64)), Z.AmbiguousZAxisError,
         "cannot tell which axis of shape (64, 64, 64) is z: no axis stands "
         "out as shorter than the other two. Set the `z_axis` setting "
         "explicitly (0, 1 or 2) rather than letting spaCR guess -- guessing "
         "wrong segments a transposed volume."),
    ]
    for case_id, volume, error, message in refused:
        with subtests.test(msg=case_id):
            with pytest.raises(error) as raised:
                Z._as_z_first(volume, None)
            assert str(raised.value) == message


def test_plan_4d_from_settings_builds_the_exact_spec_or_refuses_by_name(
        subtests):
    """Leading axes come from the order or the indices, never a guess; distance trackers need a gate."""
    accepted = [
        ("off", {"t_stack": False}, None),
        ("absent", {}, None),
        ("flat-series-time-on-axis-0",
         {"t_stack": True, "t_axis_order": "TYX", "t_axis": 0},
         Z.TStackSpec(t_axis=0, z_axis=None)),
        ("both-axes-named", {"t_stack": True, "t_axis": 1, "z_axis": 0},
         Z.TStackSpec(t_axis=1, z_axis=0)),
        ("only-t-named", {"t_stack": True, "t_axis": 1},
         Z.TStackSpec(t_axis=1, z_axis=0)),
        ("only-z-named", {"t_stack": True, "z_axis": 1},
         Z.TStackSpec(t_axis=0, z_axis=1)),
        ("centroid-on-a-projection", {
            "t_stack": True, "t_axis_order": Z.AXIS_ORDER_TZYX,
            "z_segmentation_mode": Z.MODE_PROJECT,
            "t_track_backend": Z.BACKEND_CENTROID,
            "t_max_displacement_px": 6},
         Z.TStackSpec(t_axis=0, z_axis=1, z_mode=Z.MODE_PROJECT,
                      track_backend=Z.BACKEND_CENTROID,
                      max_displacement_px=6)),
        ("centroid-on-a-flat-movie", {
            "t_stack": True, "t_axis_order": Z.AXIS_ORDER_TYX,
            "t_track_backend": Z.BACKEND_CENTROID,
            "t_max_displacement_px": 4.5},
         Z.TStackSpec(t_axis=0, z_axis=None,
                      track_backend=Z.BACKEND_CENTROID,
                      max_displacement_px=4.5)),
    ]
    for case_id, settings, spec in accepted:
        with subtests.test(msg=case_id):
            assert Z.plan_4d_from_settings(settings) == spec

    planned = Z.plan_4d_from_settings(
        {"t_stack": True, "t_axis": 1, "z_axis": 0})
    assert planned.axis_order == Z.AXIS_ORDER_ZTYX

    refused = [
        ("flat-series-time-elsewhere",
         {"t_stack": True, "t_axis_order": "TYX", "t_axis": 2},
         Z.TStackError,
         "t_axis_order='TYX' describes a (T, Y, X) acquisition, whose time "
         "axis is axis 0, but t_axis=2 was given. Give t_axis=0, or state the "
         "whole order with t_axis/z_axis indices instead."),
        ("neither-axis-nor-order", {"t_stack": True},
         Z.AmbiguousAxisOrderError,
         "t_stack is on but neither t_axis_order nor t_axis/z_axis is set, so "
         "spaCR does not know whether the leading axes of your data are "
         "(T, Z, ...) or (Z, T, ...). It will not guess: reading one as the "
         "other links objects across z and reports them as trajectories "
         "through time, which looks entirely plausible and is entirely "
         "fictional. Set t_axis_order to 'TZYX' or 'ZTYX' -- or to 'TYX' if "
         "your acquisition has no z axis at all."),
        ("centroid-on-a-kept-volume-without-anisotropy", {
            "t_stack": True, "t_axis_order": Z.AXIS_ORDER_TZYX,
            "z_segmentation_mode": Z.MODE_STITCH,
            "t_track_backend": Z.BACKEND_CENTROID,
            "t_max_displacement_px": 6},
         Z.UnknownAnisotropyError,
         "3-D segmentation needs the z/xy voxel ratio and this run does not "
         "know it. Set `anisotropy` directly (dz / dxy), or set "
         "`voxel_size_z_um` and `voxel_size_xy_um` and let spaCR derive it. "
         "spaCR will not assume 1.0: on a confocal stack the z step is "
         "typically 3-10x the xy pixel size, and segmenting such a volume as "
         "if it were isotropic fuses objects that are separated in z."),
        ("centroid-without-a-gate", {
            "t_stack": True, "t_axis_order": Z.AXIS_ORDER_TYX,
            "t_track_backend": Z.BACKEND_CENTROID},
         Z.UnknownDisplacementError,
         "t_track_backend='centroid' links by distance but neither "
         "t_max_displacement_px nor t_max_displacement_um is set. Set one; "
         "spaCR will not pick a default gate."),
    ]
    for case_id, settings, error, message in refused:
        with subtests.test(msg=case_id):
            with pytest.raises(error) as raised:
                Z.plan_4d_from_settings(settings)
            assert str(raised.value) == message


def test_track_4d_links_with_the_spec_passed_or_the_one_the_result_carries(
        subtests):
    """An explicit ``spec=`` beats the result's stale spec; without one the carried geometry is kept."""
    labels = drifting_movie()
    stale = Z.TStackResult(labels=labels,
                           spec=Z.TStackSpec(z_axis=None,
                                             frame_interval_s=None),
                           n_t=3, n_z=1)
    carried = Z.TStackResult(labels=labels,
                             spec=Z.TStackSpec(z_axis=None,
                                               frame_interval_s=4.0,
                                               voxel_size_um=(1.0, 0.5, 0.5)),
                             n_t=3, n_z=1)
    common = {"frame": [0, 1, 2], "track_id": [1, 1, 1],
              "original_label": [1, 1, 1], "x": [6.5, 7.5, 8.5],
              "y": [6.5, 6.5, 6.5], "area_px2": [36, 36, 36]}
    flags = {"truncated_z": [False, False, False],
             "truncated_t": [True, True, True]}
    cases = [
        ("explicit-spec-wins", stale,
         {"spec": Z.TStackSpec(z_axis=None, frame_interval_s=2.5)},
         {**common, "time_s": [0.0, 2.5, 5.0], **flags}),
        ("stale-spec-has-no-time", stale, {}, {**common, **flags}),
        ("carried-spec-kept", carried, {},
         {**common, "time_s": [0.0, 4.0, 8.0], **flags}),
    ]
    for case_id, result, kwargs, table in cases:
        with subtests.test(msg=case_id):
            tracked = Z.track_4d(result, **kwargs)
            assert tracked.tracks.to_dict("list") == table
            assert list(tracked.tracks.columns) == list(table)
            assert (tracked.backend, tracked.n_tracks) == (Z.BACKEND_IOU, 1)
            assert np.array_equal(tracked.labels, labels)

    with subtests.test(msg="trackpy-flat-series-defaulted-spec"):
        pytest.importorskip("trackpy")
        movie = np.zeros((3, 16, 16), dtype=np.int64)
        for t in range(3):
            movie[t, 4 + t:7 + t, 4:7] = 1
        tracked = Z.track_4d(movie, backend="trackpy",
                             max_displacement_px=6.0)
        assert (tracked.backend, tracked.n_tracks) == ("trackpy", 1)
        assert np.array_equal(tracked.labels, movie)
        assert tracked.tracks.to_dict("list") == {
            "frame": [0, 1, 2], "track_id": [1, 1, 1],
            "original_label": [1, 1, 1], "x": [5.0, 5.0, 5.0],
            "y": [5.0, 6.0, 7.0], "area_px2": [9, 9, 9], **flags}


def test_format_4d_prints_only_the_numbers_the_run_has():
    """Optional anisotropy/interval lines and the object range appear only when known; other objects are refused."""
    head = ("4D (Beta) segmentation\n  axis order      : custom/2-D\n"
            "  timepoints      : {n}\n  planes per t    : 1\n"
            "  z mode          : single_plane\n")
    cases = [
        ("anisotropy-and-interval-known",
         Z.TStackResult(labels=two_then_one(),
                        spec=Z.TStackSpec(z_axis=None, anisotropy=3.0,
                                          frame_interval_s=1.5),
                        n_t=2, n_z=1),
         head.format(n=2) + "  anisotropy      : 3 (dz/dxy)\n"
         "  frame interval  : 1.5 s\n  objects per t   : min 1, max 2"),
        ("anisotropy-and-interval-unknown",
         Z.TStackResult(labels=two_then_one(),
                        spec=Z.TStackSpec(z_axis=None, anisotropy=None,
                                          frame_interval_s=None),
                        n_t=2, n_z=1),
         head.format(n=2) + "  objects per t   : min 1, max 2"),
        ("no-timepoints-no-range",
         Z.TStackResult(labels=np.zeros((0, 8, 8), dtype=np.int32), spec=None,
                        n_t=0, n_z=1,
                        notes=["no timepoints matched the file pattern"]),
         head.format(n=0) + "  - no timepoints matched the file pattern"),
        ("spec-less-with-notes",
         Z.TStackResult(labels=two_then_one(), spec=None, n_t=2, n_z=1,
                        notes=["segmented from a synthetic stack"]),
         head.format(n=2) + "  objects per t   : min 1, max 2\n"
         "  - segmented from a synthetic stack"),
        ("a-track-result",
         Z.TrackResult(labels=drifting_movie(), n_tracks=1),
         "4D (Beta) tracking\n  backend         : iou\n"
         "  tracks          : 1\n"
         "  anisotropy      : not used by this backend\n"
         "  truncated in t  : 1 (100% of tracks touch the first or last "
         "timepoint)"),
    ]
    for case_id, result, text in cases:
        assert Z.format_4d(result) == text, case_id

    with pytest.raises(TypeError) as raised:
        Z.format_4d(np.zeros((2, 4, 4), dtype=np.int32))
    assert str(raised.value) == \
        "format_4d takes a TStackResult or a TrackResult, got ndarray"
