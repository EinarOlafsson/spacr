"""The intensity record :func:`spacr.intensity_rescale.resolve_record` writes for a field under a plate plan.

Every branch changes the number a measurement is finally written at, so each
record is asserted in full: a field whose plate cannot be read from its name
takes the explicitly non-comparable per-field route, a normalised float keeps
its fixed factor, a label-only field stays at factor one, a plate inside the
uint16 ceiling is an identity, and a plate above it is scaled down.
"""
from __future__ import annotations

import numpy as np

from spacr import intensity_rescale
from spacr.intensity_rescale import PLAN_SETTINGS_KEY, UINT16_MAX, resolve_record

#: Both planes of a two-plane array are label masks, so nothing is intensity.
ALL_MASKS = {"cell_mask_dim": 0, "nucleus_mask_dim": 1, "timelapse": False}

#: Plane 1 is a label mask; plane 0 is the intensity channel.
ONE_INTENSITY = {"cell_mask_dim": 1, "timelapse": False}


def _stack(top, dtype):
    data = np.zeros((3, 3, 2), dtype=dtype)
    data[..., 0] = top
    return data


def _planned(settings, plate_max):
    planned = dict(settings)
    planned[PLAN_SETTINGS_KEY] = {"version": 1, "fields": {}, "failures": {},
                                  "plates": {"plate1": plate_max}}
    return planned


def _record(plate, dtype, top, kind, factor, scope, plate_max, comparable):
    return {"plateID": plate, "original_dtype": dtype, "original_intensity_max": top,
            "kind": kind, "rescale_factor": factor, "rescale_scope": scope,
            "plate_intensity_max": plate_max, "comparable_within_plate": comparable}


CASES = [
    ("an_unparseable_name_falls_back_per_field",
     _stack(3, np.uint16), "nonsense.npy", _planned(ONE_INTENSITY, 131070.0),
     _record("error", "uint16", 3.0, "raw", 1.0, "field_fallback", None, False), False),
    ("a_normalised_float_keeps_its_fixed_factor",
     _stack(0.5, np.float32), "plate1_A01_1.npy", _planned(ONE_INTENSITY, 131070.0),
     _record("plate1", "float32", 0.5, "fixed_normalized", UINT16_MAX,
             "fixed_normalized", 131070.0, True), False),
    ("a_label_only_field_stays_at_factor_one",
     _stack(5, np.uint16), "plate1_A01_1.npy", _planned(ALL_MASKS, 131070.0),
     _record("plate1", "uint16", 0.0, "no_intensity", 1.0, "no_intensity", 131070.0, True),
     False),
    ("a_plate_inside_uint16_is_an_identity",
     _stack(1000, np.uint16), "plate1_A01_1.npy", _planned(ONE_INTENSITY, 40000.0),
     _record("plate1", "uint16", 1000.0, "raw", 1.0, "identity", 40000.0, True), False),
    ("a_plate_above_uint16_is_scaled_down",
     _stack(1000, np.uint16), "plate1_A01_1.npy", _planned(ONE_INTENSITY, 131070.0),
     _record("plate1", "uint16", 1000.0, "raw", 0.5, "plate", 131070.0, True), True),
]


def test_resolve_record_labels_every_scaling_decision(subtests):
    """The full record is exact for each case, and needs_warning agrees with the factor."""
    for case_id, data, name, settings, expected, warns in CASES:
        with subtests.test(msg=case_id):
            assert resolve_record(data, name, settings) == expected
            assert intensity_rescale.needs_warning(expected["rescale_factor"]) is warns
