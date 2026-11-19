"""Measure input refusals preserve data and give a specific correction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import measure
from spacr.errors import ConfigurationError


def test_invalid_worker_count_names_the_value_and_correction():
    """A noninteger worker count stays invalid and explains the accepted input."""
    with pytest.raises(ConfigurationError) as caught:
        measure.resolve_n_jobs("four", cpu_count=8)
    assert "'four'" in str(caught.value)
    assert "n_jobs" in str(caught.value)
    assert "Set n_jobs to a positive whole number" in str(caught.value)


@pytest.mark.parametrize("key, value", [
    ("voxel_size_z_um", -1),
    ("voxel_size_xy_um", 0),
    ("anisotropy", -2),
])
def test_invalid_voxel_calibration_names_the_input_without_replacing_it(key, value):
    """Bad physical calibration is refused without inventing a replacement."""
    settings = {"voxel_size_z_um": 1.0, "voxel_size_xy_um": 0.25, key: value}
    before = dict(settings)
    with pytest.raises(ConfigurationError) as caught:
        measure.resolve_measurement_spacing(settings, ndim=3, n_z=3)
    assert key in str(caught.value)
    assert str(value) in str(caught.value)
    recovery = ("Set anisotropy to the recorded Z step divided by the XY pixel size"
                if key == "anisotropy" else "Enter the recorded microscope voxel size")
    assert recovery in str(caught.value)
    assert settings == before


def test_duplicate_channel_names_require_distinct_names_and_preserve_settings():
    """Case-insensitive duplicate names remain invalid without rewriting settings."""
    settings = {"channels": [0, 1], "channel_names": ["DAPI", "dapi"]}
    with pytest.raises(ValueError) as caught:
        measure._measurement_channel_names(settings)
    assert "repeats a name" in str(caught.value)
    assert "Give every measured channel a distinct name" in str(caught.value)
    assert settings == {"channels": [0, 1], "channel_names": ["DAPI", "dapi"]}


def test_confluency_channel_refusal_keeps_the_index_and_image():
    """An out-of-range channel is refused before selecting or changing image pixels."""
    data = np.arange(8).reshape(2, 2, 2)
    before = data.copy()
    with pytest.raises(ValueError) as caught:
        measure._measure_field_confluency(
            data, {"confluency_source": "intensity", "confluency_channel": 9})
    assert "confluency_channel is 9" in str(caught.value)
    assert "2 planes" in str(caught.value)
    assert "Choose confluency_channel from the merged-array intensity planes" in str(caught.value)
    np.testing.assert_array_equal(data, before)


def test_fucci_channel_count_refusal_preserves_the_requested_reporter():
    """A single FUCCI reporter is not silently replaced with a guessed channel."""
    channels = [3]
    with pytest.raises(ValueError) as caught:
        measure._fucci_states(pd.DataFrame(), {}, channels)
    assert "got [3]" in str(caught.value)
    assert "G1 reporter then the S/G2/M reporter" in str(caught.value)
    assert "Set cell_cycle_fucci_channels to [G1 channel, S/G2/M channel]" in str(caught.value)
    assert channels == [3]


@pytest.mark.parametrize("thresholds", [[1, 2, 3], ["unreadable"]])
def test_manual_viability_threshold_refusal_explains_the_entry(thresholds):
    """Malformed manual thresholds keep their values and their refusal behavior."""
    before = list(thresholds)
    with pytest.raises(ValueError) as caught:
        measure._viability_manual(thresholds)
    assert "viability_thresholds" in str(caught.value)
    recovery = ("Replace the invalid entry with a numeric threshold or None"
                if thresholds == ["unreadable"] else "Enter viability_thresholds as [dead, live]")
    assert recovery in str(caught.value)
    assert thresholds == before
    if thresholds == ["unreadable"]:
        assert caught.value.__cause__ is None
        assert caught.value.__suppress_context__


def test_overlapping_viability_controls_require_separate_assignments():
    """A well cannot become both controls and the input table stays unchanged."""
    frame = pd.DataFrame({"rowID": ["r1"], "columnID": ["c1"]})
    before = frame.copy(deep=True)
    settings = {"viability_negative_wells": "A01", "viability_positive_wells": "A01"}
    with pytest.raises(ValueError) as caught:
        measure._well_roles(frame, settings)
    assert "1 well(s)" in str(caught.value)
    assert "viability_negative_wells" in str(caught.value)
    assert "viability_positive_wells" in str(caught.value)
    assert "Assign each control well to only one of these two settings" in str(caught.value)
    pd.testing.assert_frame_equal(frame, before)
    assert settings == {"viability_negative_wells": "A01", "viability_positive_wells": "A01"}


@pytest.mark.parametrize("conditions, diagnostic, recovery", [
    ({"control": "not-a-well"}, "wound_conditions['control']", "Use well names such as B03"),
    ({"control": "A01", "treated": "A01"}, "both 'control' and 'treated'",
     "Remove the overlapping well from one condition"),
])
def test_wound_condition_refusal_keeps_the_assignments_and_names_recovery(
        conditions, diagnostic, recovery):
    """Malformed and overlapping well specifications stay invalid and unchanged."""
    before = dict(conditions)
    with pytest.raises(ValueError) as caught:
        measure._wound_condition_lookup(conditions)
    assert diagnostic in str(caught.value)
    assert recovery in str(caught.value)
    assert conditions == before


def test_nonfinite_intensity_names_the_source_without_replacing_pixels():
    """Invalid intensity data stay refused without an invented pixel replacement."""
    plane = np.array([[1.0, np.nan]])
    before = plane.copy()
    with pytest.raises(ConfigurationError) as caught:
        measure._checked_intensity(plane, "plate1_A01_1", "source.tif")
    assert "source.tif: plate1_A01_1" in str(caught.value)
    assert "Correct the source image export so every intensity is finite" in str(caught.value)
    np.testing.assert_array_equal(plane, before)


@pytest.mark.parametrize("plane, shape, recovery", [
    (np.zeros((2, 3), dtype=np.int32), (2, 2), "Select an aligned mask and intensity image"),
    (np.array([[-1, 0]]), (1, 2), "Re-export the labelled mask with zero background"),
    (np.array([[70000, 0]]), (1, 2), "Regenerate a compatible image-and-mask project"),
])
def test_incompatible_mask_names_recovery_without_changing_labels(plane, shape, recovery):
    """Mask shape and identifier guards refuse data without clipping or relabelling."""
    before = plane.copy()
    with pytest.raises(ConfigurationError) as caught:
        measure._checked_label(plane, "plate1_A01_1", "mask.tif", shape)
    assert "mask.tif" in str(caught.value)
    assert recovery in str(caught.value)
    np.testing.assert_array_equal(plane, before)
