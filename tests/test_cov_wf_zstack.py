"""spacr.zstack's 4-D half: the answers it gives when a fact is simply absent.

Every branch pinned here is a place where the 4-D plumbing has to cope with a
missing number rather than a wrong one -- a result that carries no spec, an
acquisition that segmented no timepoints, a settings dict that names both
leading axes outright, a distance tracker on a flat movie that has no z to
scale.  Each of those is an ordinary way to use the feature, and each is one
line away from printing a unit it does not know, dividing by a count of
nothing, or demanding an anisotropy for an axis that does not exist.

Nothing here runs Cellpose or any tracking library: every label array is
synthetic and every link is computed by the built-in ``iou`` and ``centroid``
backends, so the file is CPU-only and offline.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacr.zstack as Z


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def drifting_movie(n_t=3, size=16, box=6, step=1):
    """``(T, Y, X)`` labels: one square that drifts ``step`` px in x per frame.

    Consecutive frames overlap heavily, so both the overlap backend and the
    centroid backend link it into a single track -- which is what makes the
    tracks table below a readable check on which spec was used.
    """
    out = np.zeros((n_t, size, size), dtype=np.int32)
    for t in range(n_t):
        x0 = 4 + step * t
        out[t, 4:4 + box, x0:x0 + box] = 1
    return out


def two_then_one(size=12):
    """``(T, Y, X)`` labels: two objects at t=0, one at t=1.

    Gives ``objects_per_timepoint`` a genuine min and max to report, so the
    reported range is a fact about the data and not a constant.
    """
    out = np.zeros((2, size, size), dtype=np.int32)
    out[0, 1:4, 1:4] = 1
    out[0, 7:10, 7:10] = 2
    out[1, 1:4, 1:4] = 1
    return out


# ---------------------------------------------------------------------------
# track_4d: an explicitly passed spec outranks the one the result carries
# ---------------------------------------------------------------------------

def test_a_spec_passed_to_track_4d_beats_the_one_the_result_carries():
    """Re-tracking a segmentation with new settings must use the new settings.

    ``track_4d`` accepts the :class:`TStackResult` that segmentation returned,
    and that result remembers the spec it was segmented with.  A user who
    re-links the same labels with a corrected spec -- they found the real frame
    interval, or want a different gate -- passes it as ``spec=``.  If the
    result's stale spec quietly won instead, the tracks table would come back
    with the OLD frame interval, and every velocity computed from it would be
    wrong by that ratio while looking entirely normal.
    """
    labels = drifting_movie()
    segmented_with = Z.TStackSpec(z_axis=None, frame_interval_s=None)
    result = Z.TStackResult(labels=labels, spec=segmented_with, n_t=3, n_z=1)

    corrected = Z.TStackSpec(z_axis=None, frame_interval_s=2.5)
    tracked = Z.track_4d(result, spec=corrected)

    # The corrected spec supplies a frame interval, so the table gains the
    # column the stale spec could not have produced at all.
    assert "time_s" in tracked.tracks.columns
    assert sorted(tracked.tracks["time_s"].tolist()) == [0.0, 2.5, 5.0]
    assert tracked.n_tracks == 1
    assert tracked.backend == Z.BACKEND_IOU

    # And the same call without the override does not invent the column, which
    # is what makes the assertion above evidence rather than a coincidence.
    stale = Z.track_4d(result)
    assert "time_s" not in stale.tracks.columns
    assert stale.n_tracks == 1


def test_track_4d_still_reads_the_results_own_spec_when_none_is_given():
    """Passing only the result must keep the geometry it was segmented with.

    This is the ordinary call -- segment, then track -- and the spec is how the
    voxel size reaches the tracks table.  If it were dropped, ``volume_tracks``
    would fall back to pixel columns and a screen measured in micrometres would
    silently change units halfway through the pipeline.
    """
    labels = drifting_movie()
    carried = Z.TStackSpec(z_axis=None, frame_interval_s=4.0,
                           voxel_size_um=(1.0, 0.5, 0.5))
    result = Z.TStackResult(labels=labels, spec=carried, n_t=3, n_z=1)

    tracked = Z.track_4d(result)

    assert tracked.tracks["time_s"].tolist() == [0.0, 4.0, 8.0]
    assert tracked.n_tracks == 1


# ---------------------------------------------------------------------------
# format_4d: it never prints a number the run does not have
# ---------------------------------------------------------------------------

def test_a_summary_omits_the_anisotropy_and_interval_it_was_never_told():
    """The 4-D summary is pasted next to the numbers it describes.

    ``anisotropy`` and ``frame_interval_s`` are optional: a flat movie has no
    z step, and an acquisition whose metadata was lost has no interval.  If
    the summary printed a placeholder for them anyway, someone reading the
    report would believe a ratio and a seconds-per-frame that the run never
    used -- and both are exactly the numbers a reviewer checks first.
    """
    labels = two_then_one()

    known = Z.TStackResult(
        labels=labels,
        spec=Z.TStackSpec(z_axis=None, anisotropy=3.0, frame_interval_s=1.5),
        n_t=2, n_z=1,
    )
    unknown = Z.TStackResult(
        labels=labels,
        spec=Z.TStackSpec(z_axis=None, anisotropy=None, frame_interval_s=None),
        n_t=2, n_z=1,
    )

    with_numbers = Z.format_4d(known)
    without = Z.format_4d(unknown)

    # The spec that has them prints them, in the documented units.
    assert "anisotropy      : 3 (dz/dxy)" in with_numbers
    assert "frame interval  : 1.5 s" in with_numbers

    # The spec that does not have them prints neither line, and keeps the rest.
    assert "anisotropy" not in without
    assert "frame interval" not in without
    assert "  timepoints      : 2" in without
    assert "  objects per t   : min 1, max 2" in without


def test_a_run_that_segmented_no_timepoints_reports_no_object_range():
    """An empty acquisition is an answer, not a crash.

    ``objects per t`` is printed as ``min ..., max ...``, and ``min()`` of an
    empty sequence raises.  A user who points the 4-D path at a folder whose
    files did not match the pattern gets zero timepoints, and the summary is
    the first thing they read: it has to say "0 timepoints" rather than die
    inside the reporter with a ValueError about an empty sequence.
    """
    empty = Z.TStackResult(
        labels=np.zeros((0, 8, 8), dtype=np.int32),
        spec=None, n_t=0, n_z=1,
        notes=["no timepoints matched the file pattern"],
    )
    populated = Z.TStackResult(
        labels=two_then_one(), spec=None, n_t=2, n_z=1,
        notes=["segmented from a synthetic stack"],
    )

    empty_text = Z.format_4d(empty)
    populated_text = Z.format_4d(populated)

    # The populated run prints the range, so its absence below is meaningful.
    assert "  objects per t   : min 1, max 2" in populated_text

    assert "objects per t" not in empty_text
    assert "  timepoints      : 0" in empty_text
    # A spec-less result still names its axis order honestly rather than
    # guessing one, and still carries its notes to the reader.
    assert "  axis order      : custom/2-D" in empty_text
    assert "  - no timepoints matched the file pattern" in empty_text


def test_format_4d_refuses_an_object_that_is_neither_result():
    """The reporter must name what it was handed, not print half a summary.

    ``format_4d`` is called on whatever a pipeline step returned.  When a step
    starts returning a bare array -- a refactor away -- a partially filled
    report is far worse than a TypeError, because the missing lines look like
    settings that were switched off.
    """
    with pytest.raises(TypeError, match="TStackResult or a TrackResult"):
        Z.format_4d(np.zeros((2, 4, 4), dtype=np.int32))

    # A TrackResult, by contrast, is summarised rather than refused.
    text = Z.format_4d(Z.TrackResult(labels=drifting_movie(), n_tracks=1))
    assert "4D (Beta) tracking" in text
    assert "  tracks          : 1" in text


# ---------------------------------------------------------------------------
# plan_4d_from_settings: both leading axes named outright
# ---------------------------------------------------------------------------

def test_naming_both_leading_axes_needs_no_partner_to_be_deduced():
    """``t_axis`` and ``z_axis`` together are a complete order on their own.

    ``t_axis_order`` is the recommended spelling, but the two index settings
    are the older one and still supported.  When both are given there is
    nothing left to infer, and the inference step must not fire and overwrite
    one of them: reading a ``(Z, T, Y, X)`` acquisition as ``(T, Z, Y, X)``
    links objects across z and reports them as trajectories through time,
    which looks plausible and is fiction.
    """
    both = Z.plan_4d_from_settings(
        {"t_stack": True, "t_axis": 1, "z_axis": 0}
    )

    assert (both.t_axis, both.z_axis) == (1, 0)
    assert both.axis_order == Z.AXIS_ORDER_ZTYX

    # Give only one of them and the other IS deduced -- the branch above is
    # skipping exactly this arithmetic, not dead code.
    only_t = Z.plan_4d_from_settings({"t_stack": True, "t_axis": 1})
    assert (only_t.t_axis, only_t.z_axis) == (1, 0)

    only_z = Z.plan_4d_from_settings({"t_stack": True, "z_axis": 1})
    assert (only_z.t_axis, only_z.z_axis) == (0, 1)


def test_a_settings_dict_with_neither_axis_nor_order_is_refused_by_name():
    """Guessing the leading axes is the one thing this feature must never do.

    Without an order, ``(T, Z, Y, X)`` and ``(Z, T, Y, X)`` are the same shape,
    and picking wrong produces a full tracks table of trajectories that never
    happened.  The refusal has to name the setting to fix.
    """
    with pytest.raises(Z.AmbiguousAxisOrderError, match="t_axis_order"):
        Z.plan_4d_from_settings({"t_stack": True})

    # And with the feature off, the same dict is simply not a 4-D run.
    assert Z.plan_4d_from_settings({"t_stack": False}) is None


# ---------------------------------------------------------------------------
# plan_4d_from_settings: a distance tracker only needs an anisotropy for z
# ---------------------------------------------------------------------------

def test_a_distance_tracker_asks_for_anisotropy_only_when_z_survives():
    """The z step only has to be known when there is a z to measure along.

    The distance backends scale the z component of a displacement by
    ``dz/dxy``.  In ``project`` mode z is gone before linking, so there is no z
    component and no ratio to demand -- insisting on one would refuse a
    perfectly well-specified projection run and send the user hunting for a
    voxel size their acquisition never recorded.  In ``stitch`` mode the volume
    survives, and then a missing ratio silently lets objects several planes
    apart link as one.
    """
    projected = Z.plan_4d_from_settings({
        "t_stack": True,
        "t_axis_order": Z.AXIS_ORDER_TZYX,
        "z_segmentation_mode": Z.MODE_PROJECT,
        "t_track_backend": Z.BACKEND_CENTROID,
        "t_max_displacement_px": 6,
    })

    assert projected.track_backend == Z.BACKEND_CENTROID
    assert projected.max_displacement_px == 6
    assert projected.anisotropy is None
    assert projected.z_mode == Z.MODE_PROJECT

    # Keep the volume instead and the very same settings are refused, which is
    # what makes the accepted spec above a decision rather than an oversight.
    with pytest.raises(Z.UnknownAnisotropyError):
        Z.plan_4d_from_settings({
            "t_stack": True,
            "t_axis_order": Z.AXIS_ORDER_TZYX,
            "z_segmentation_mode": Z.MODE_STITCH,
            "t_track_backend": Z.BACKEND_CENTROID,
            "t_max_displacement_px": 6,
        })


def test_a_distance_tracker_without_a_gate_is_refused_before_the_run():
    """A displacement gate has no safe default, so it is demanded up front.

    Too large a gate fuses neighbouring objects into one track; too small
    breaks one object into a track per frame.  Both produce a full, plausible
    tracks table.  The refusal has to arrive while the settings are being read
    -- not after a model has loaded and the first timepoint has been segmented.
    """
    with pytest.raises(Z.UnknownDisplacementError,
                       match="t_max_displacement_px"):
        Z.plan_4d_from_settings({
            "t_stack": True,
            "t_axis_order": Z.AXIS_ORDER_TYX,
            "t_track_backend": Z.BACKEND_CENTROID,
        })

    # With the gate supplied, the same flat movie plans without complaint and
    # keeps the number it was given.
    ok = Z.plan_4d_from_settings({
        "t_stack": True,
        "t_axis_order": Z.AXIS_ORDER_TYX,
        "t_track_backend": Z.BACKEND_CENTROID,
        "t_max_displacement_px": 4.5,
    })
    assert ok.z_axis is None
    assert ok.max_displacement_px == 4.5
