"""Intensity-plan decisions for fields that carry no intensity, or no name.

Every branch here changes the number a measurement is finally written at, so
each one has to be reachable and each one has to be labelled. The cases below
are the ones a clean plate never produces: a merged array whose only planes
are label masks, a plane that is entirely non-finite, a file that is not a
merged stack at all, and a field whose name the schema cannot parse. A worker
that guessed in any of them would write intensities on a scale nothing else
on the plate shares, and the record would not say so.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import intensity_rescale
from spacr.intensity_rescale import (
    PLAN_SETTINGS_KEY,
    UINT16_MAX,
    build_plate_plan,
    fallback_record,
    resolve_record,
    signal_max,
)


#: Both planes of a two-plane array are label masks, so nothing is intensity.
ALL_MASKS = {"cell_mask_dim": 0, "nucleus_mask_dim": 1, "timelapse": False}

#: Plane 1 is a label mask; plane 0 is the intensity channel.
ONE_INTENSITY = {"cell_mask_dim": 1, "timelapse": False}


def _stack(top, dtype=np.float32, planes=2):
    data = np.zeros((3, 3, planes), dtype=dtype)
    data[..., 0] = top
    return data


def test_a_stack_that_is_all_label_planes_reports_no_intensity():
    """With every plane a mask there is no intensity to scale, and the
    plan must say so rather than reporting a maximum of zero as a real one."""
    top, has_intensity = signal_max(_stack(5.0), ALL_MASKS)
    assert (top, has_intensity) == (0.0, False)


def test_an_intensity_plane_of_only_nan_is_still_an_intensity_plane():
    """A channel that is present but entirely non-finite has a maximum of
    zero and must not be mistaken for a stack with no intensity at all: the
    two get different scaling rules."""
    data = _stack(0.0)
    data[..., 0] = np.nan
    top, has_intensity = signal_max(data, ONE_INTENSITY)
    assert (top, has_intensity) == (0.0, True)


def test_a_maskless_field_is_planned_as_no_intensity_and_left_alone(tmp_path):
    """The plate plan gives a label-only field factor 1 under its own scope
    name, so a reader can tell "nothing to scale" from "scaled by one"."""
    name = "plate1_A01_1.npy"
    np.save(tmp_path / name, _stack(5.0, dtype=np.uint16))
    plan = build_plate_plan(tmp_path, [name], ALL_MASKS)
    record = plan["fields"][name]
    assert record["kind"] == "no_intensity"
    assert record["rescale_scope"] == "no_intensity"
    assert record["rescale_factor"] == 1.0
    assert record["comparable_within_plate"] is True


def test_an_array_without_a_channel_axis_is_a_named_failure(tmp_path):
    """A two-dimensional .npy is not a merged field. It has to land in
    ``failures`` with the shape it actually had, because the worker's own
    fallback is the only thing that can rescue it."""
    name = "plate1_A01_1.npy"
    np.save(tmp_path / name, np.zeros((4, 4), dtype=np.uint16))
    plan = build_plate_plan(tmp_path, [name], ONE_INTENSITY)
    assert name not in plan["fields"]
    assert "ValueError" in plan["failures"][name]
    assert "(4, 4)" in plan["failures"][name]


def test_a_maskless_field_falls_back_to_no_intensity_and_stays_comparable():
    """Without a plan the worker must reach the same "nothing to scale"
    conclusion, and must still call it comparable -- an untouched label
    plane is comparable with every other untouched label plane."""
    record = fallback_record(_stack(5.0, dtype=np.uint16), "plate1_A01_1.npy",
                             ALL_MASKS)
    assert record["kind"] == "no_intensity"
    assert record["rescale_factor"] == 1.0
    assert record["rescale_scope"] == "no_intensity"
    assert record["comparable_within_plate"] is True


def test_a_field_name_the_schema_cannot_parse_is_recorded_as_an_error_plate():
    """``fallback_record`` never propagates a name-parsing failure: the
    measurement still has to be written, under a plate id that is visibly
    not a real plate."""
    record = fallback_record(_stack(3.0, dtype=np.uint16), "nonsense.npy",
                             ONE_INTENSITY)
    assert record["plateID"] == "error"
    assert record["comparable_within_plate"] is False


def test_an_unparseable_name_makes_the_worker_fall_back_even_with_a_plan():
    """A plan is keyed by plate. A field whose plate cannot be read from its
    name cannot be matched against one, so it takes the explicitly
    non-comparable per-field route instead of borrowing another plate's
    factor."""
    settings = dict(ONE_INTENSITY)
    settings[PLAN_SETTINGS_KEY] = {"version": 1, "fields": {}, "failures": {},
                                   "plates": {"plate1": 131070.0}}
    record = resolve_record(_stack(3.0, dtype=np.uint16), "nonsense.npy",
                            settings)
    assert record["plateID"] == "error"
    assert record["rescale_scope"] == "field_fallback"
    assert record["plate_intensity_max"] is None


def test_a_normalised_float_field_keeps_its_fixed_factor_under_a_plan():
    """A float field inside [0, 1] converts by 65535 whatever the plate's raw
    maximum is; the plate factor would destroy a normalised image."""
    settings = dict(ONE_INTENSITY)
    settings[PLAN_SETTINGS_KEY] = {"version": 1, "fields": {}, "failures": {},
                                   "plates": {"plate1": 131070.0}}
    record = resolve_record(_stack(0.5, dtype=np.float32), "plate1_A01_1.npy",
                            settings)
    assert record["kind"] == "fixed_normalized"
    assert record["rescale_factor"] == UINT16_MAX
    assert record["comparable_within_plate"] is True


def test_a_label_only_field_under_a_plan_is_left_at_factor_one():
    """The plan path must reach the same no-intensity answer as the fallback
    path, so a resumed plate does not rescale label planes it skipped before."""
    settings = dict(ALL_MASKS)
    settings[PLAN_SETTINGS_KEY] = {"version": 1, "fields": {}, "failures": {},
                                   "plates": {"plate1": 131070.0}}
    record = resolve_record(_stack(5.0, dtype=np.uint16), "plate1_A01_1.npy",
                            settings)
    assert record["kind"] == "no_intensity"
    assert record["rescale_scope"] == "no_intensity"
    assert record["rescale_factor"] == 1.0


def test_a_plate_that_fits_in_uint16_scales_nothing():
    """When the plate maximum is already inside the uint16 ceiling the only
    honest factor is 1, recorded as ``identity`` so nobody looks for a scale
    that was never applied."""
    settings = dict(ONE_INTENSITY)
    settings[PLAN_SETTINGS_KEY] = {"version": 1, "fields": {}, "failures": {},
                                   "plates": {"plate1": 40000.0}}
    record = resolve_record(_stack(1000, dtype=np.uint16), "plate1_A01_1.npy",
                            settings)
    assert record["rescale_scope"] == "identity"
    assert record["rescale_factor"] == 1.0
    assert record["plate_intensity_max"] == 40000.0
    assert intensity_rescale.needs_warning(record["rescale_factor"]) is False
